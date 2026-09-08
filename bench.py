# -*- coding: utf-8 -*-
"""测评主脚本：对两家 vision API 跑全部探针，判分，输出成本/延迟/明细报告"""
import base64
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI

ASSET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")

MODELS = {
    "deepseek": {
        "label": "deepseek-v4-flash-vision-exp",
        "base_url": "https://api.deepseek.com",
        "key_env": "DS_API_KEY",
        "price": {"in": 1.5, "out": 4.5},   # 元/百万，空闲时段未命中缓存
    },
    "deepseek41": {
        "label": "deepseek-v4.1-flash-expires-on-0910",
        "base_url": "https://api.deepseek.com",
        "key_env": "DS41_API_KEY",
        "price": {"in": 1.5, "out": 4.5},   # 占位，按 flash 同价估计
    },
    "mimo": {
        "label": "mimo-v2.5",
        "base_url": "https://api.xiaomimimo.com/v1",
        "key_env": "MIMO_API_KEY",
        "price": {"in": 1.0, "out": 2.0},   # 元/百万，未命中缓存
    },
}

SCORE_FN = {
    "char_acc": "字符准确率",
    "exact": "精确匹配",
    "numeric": "数值容差",
    "rowcol": "行列匹配",
    "grid": "网格字母序列",
}


def b64(png):
    with open(os.path.join(ASSET_DIR, png), "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode()


def norm(s):
    s = s.translate(str.maketrans("，。：；！？（）", ",.:;!?()"))
    s = re.sub(r"\s+", "", (s or "").strip().lower())
    return s


def lev(a, b):
    if len(a) < len(b):
        a, b = b, a
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def score(probe, answer):
    st = probe["score_type"]
    gt = probe["gt"]
    a = answer or ""
    if st == "char_acc":
        g, x = norm(gt), norm(a)
        return max(0.0, 1 - lev(g, x) / max(len(g), 1))
    if st == "exact":
        g, x = norm(gt), norm(a)
        return 1.0 if (g == x or g in x) else 0.0
    if st == "numeric":
        m = re.search(r"-?\d+(?:\.\d+)?", a)
        if not m:
            return 0.0
        return 1.0 if abs(float(m.group()) - float(gt)) <= 0.01 else 0.0
    if st == "rowcol":
        nums = re.findall(r"\d+", a)
        if len(nums) < 2:
            return 0.0
        return 1.0 if (int(nums[0]), int(nums[1])) == tuple(int(x) for x in gt.split(",")) else 0.0
    if st == "grid":
        g = "".join(re.findall(r"[A-Za-z]", gt)).upper()
        x = "".join(re.findall(r"[A-Za-z]", a)).upper()
        if not g:
            return 0.0
        return 1.0 if g == x else 0.0
    return 0.0


def call_api(cfg, probe, max_retries=6):
    client = OpenAI(api_key=os.environ[cfg["key_env"]], base_url=cfg["base_url"])
    content = [{"type": "text", "text": probe["question"]}]
    for img in probe["images"]:
        content.append({"type": "image_url", "image_url": {"url": b64(img)}})
    kwargs = dict(
        model=cfg["label"],
        messages=[{"role": "user", "content": content}],
    )
    if NOTHINK:
        kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
        kwargs["max_tokens"] = 2048
    else:
        kwargs["extra_body"] = {"thinking": {"type": "enabled"}}
        if cfg["label"].startswith("mimo"):
            kwargs["max_completion_tokens"] = 65536
        else:
            kwargs["max_tokens"] = 65536
            kwargs["reasoning_effort"] = "max"
    last = None
    for attempt in range(max_retries):
        try:
            t0 = time.time()
            resp = client.chat.completions.create(**kwargs)
            dt = time.time() - t0
            text = (resp.choices[0].message.content or "").strip()
            msg = resp.choices[0].message
            rt = 0
            if getattr(resp.usage, "completion_tokens_details", None) is not None:
                rt = getattr(resp.usage.completion_tokens_details, "reasoning_tokens", 0) or 0
            fr = resp.choices[0].finish_reason
            return text, resp.usage.prompt_tokens, resp.usage.completion_tokens, rt, dt, fr, None
        except Exception as e:  # noqa: BLE001
            last = e
            es = str(e).lower()
            if "429" in es or "rate" in es:
                time.sleep(min(30, 5 * 2 ** attempt))
            else:
                time.sleep(2 * (attempt + 1))
    return "", 0, 0, 0, 0.0, None, str(last)[:300]


def cost(cfg, pt, ct):
    p = cfg["price"]
    return pt / 1e6 * p["in"] + ct / 1e6 * p["out"]


_LOCK = threading.Lock()
NOTHINK = False


def _append(outfile, rec):
    with open(outfile, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        f.flush()


def run_model(key, cfg, probes, outfile, resume, workers=1, max_retries=3):
    """跑单个模型，每完成一题立即追加写入 outfile；resume 时跳过已正确完成、重跑失败的题。
    workers>1 时用线程池并发跑探针。"""
    done = {}
    if resume and os.path.exists(outfile):
        for line in open(outfile, encoding="utf-8"):
            r = json.loads(line)
            done[r["id"]] = r["score"]
    todo = [p for p in probes if not (resume and p["id"] in done and done[p["id"]] == 1.0)]
    skipped = len(probes) - len(todo)
    if skipped:
        print(f"  ({skipped} already-correct probes skipped via resume)", flush=True)

    def work(p):
        if resume and p["id"] in done and done[p["id"]] == 1.0:
            return
        text, pt, ct, rt, dt, fr, err = call_api(cfg, p, max_retries)
        sc = 0.0 if err else score(p, text)
        rec = {
            "model": key, "id": p["id"], "task": p["task"],
            "score": round(sc, 4), "score_type": p["score_type"],
            "answer": text[:500], "gt": p["gt"], "error": err,
            "prompt_tokens": pt, "completion_tokens": ct, "reasoning_tokens": rt,
            "finish_reason": fr,
            "latency_s": round(dt, 2), "cost_yuan": round(cost(cfg, pt, ct), 6),
        }
        _append(outfile, rec)
        stat = "OK" if sc == 1 else ("PART" if 0 < sc < 1 else "ERR" if err else "FAIL")
        trunc = " [TRUNC]" if fr == "length" else ""
        with _LOCK:
            print(f"  {p['id']:24s} {stat:4s} {sc:6.3f}  {p['task']:12s} think={rt:>6d} tot={ct:>6d}{trunc} cost={rec['cost_yuan']:8.6f}元 {dt:5.1f}s", flush=True)

    if workers > 1 and len(todo) > 1:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            list(ex.map(work, todo))
    else:
        for p in todo:
            work(p)


def load_results():
    """合并所有 results_<model>.jsonl 与 results.jsonl（兼容旧格式），(model,id) 去重保留最新。"""
    rows = []
    for fn in sorted(os.listdir(os.path.dirname(os.path.abspath(__file__)))):
        if fn.startswith("results_") and fn.endswith(".jsonl"):
            for line in open(os.path.join(os.path.dirname(os.path.abspath(__file__)), fn), encoding="utf-8"):
                rows.append(json.loads(line))
    legacy = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.jsonl")
    if os.path.exists(legacy):
        for line in open(legacy, encoding="utf-8"):
            rows.append(json.loads(line))
    dedup = {}
    for r in rows:
        dedup[(r["model"], r["id"])] = r
    return list(dedup.values())


def main():
    args = sys.argv[1:]
    if "--report" in args:
        manifest = next((a for a in args if a.endswith(".json")), "probes.json")
        probes = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), manifest), encoding="utf-8"))["probes"]
        report(load_results(), probes)
        return
    manifest = "probes.json"
    model_only = None
    resume = "--resume" in args
    workers = 1
    allowed = None
    global NOTHINK
    NOTHINK = "--nothink" in args
    for a in args:
        if a.endswith(".json"):
            manifest = a
        elif a == "--model":
            continue
        elif a.startswith("--model="):
            model_only = a.split("=", 1)[1]
        elif a.startswith("--workers="):
            workers = int(a.split("=", 1)[1])
        elif a in MODELS and a not in ("--model",):
            model_only = a
        elif a == "--resume" or a.startswith("--"):
            continue
        else:
            allowed = (a.split(",") if a else None)
    probes = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), manifest), encoding="utf-8"))["probes"]
    if allowed is not None:
        probes = [p for p in probes if p["id"] in allowed]
    keys = [model_only] if model_only else list(MODELS.keys())
    for key in keys:
        cfg = MODELS[key]
        outfile = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"results_{key}.jsonl")
        print(f"\n=== {cfg['label']} ({len(probes)} probes, workers={workers}) -> {outfile} ===", flush=True)
        run_model(key, cfg, probes, outfile, resume, workers)
    if model_only is None:
        report(load_results(), probes)


def report(results, probes):
    lines = []
    def emit(s=""):
        lines.append(s)
        print(s)
    models = sorted(set(r["model"] for r in results))
    width = 14
    emit("=" * 72)
    emit("汇总（分数=正确率，partially = 字符级准确率均值）")
    tasks = sorted(set(p["task"] for p in probes))
    header = f"{'任务':<16}{'题数':>4}" + "".join(f"{m:>{width}}" for m in models)
    emit(header)
    t_acc = {m: 0.0 for m in models}
    t_ok = {m: 0.0 for m in models}
    t_cnt = {m: 0 for m in models}
    t_cost = {m: 0.0 for m in models}
    for task in tasks:
        row = {m: [] for m in models}
        for r in results:
            if r["task"] == task:
                row[r["model"]].append(r)
        parts = []
        for m in models:
            rs = row[m]
            if not rs:
                parts.append(f"{'-':>{width}}")
                continue
            s = sum(r["score"] for r in rs) / len(rs)
            ok = sum(1 for r in rs if r["score"] == 1.0)
            t_acc[m] += s * len(rs)
            t_ok[m] += ok
            t_cnt[m] += len(rs)
            t_cost[m] += sum(r["cost_yuan"] for r in rs)
            parts.append(f"{s:6.3f} ({ok}/{len(rs)})".rjust(width))
        emit(f"{task:<16}{len(row[models[0]]):>4}" + "".join(parts))
    emit("-" * 72)
    emit(f"{'整体平均':<16}{sum(t_cnt.values()):>4}" + "".join(f"{t_acc[m]/max(t_cnt[m],1):>{width}.3f}" for m in models))
    emit(f"{'全对题数':<16}{'':>4}" + "".join(f"{t_ok[m]:>{width}.0f}" for m in models))
    emit(f"{'总成本(元)':<16}{'':>4}" + "".join(f"{t_cost[m]:>{width}.6f}" for m in models))
    emit("")
    emit("分辨率 A/B（同一图高清 vs 缩至700px）：full - ds700")
    for task, prefix in (("res_en", "ocr_en_12"), ("res_cn", "ocr_cn_0"), ("res_chart", "crowd_chart_0")):
        full = {r["model"]: r["score"] for r in results if r["id"] == f"{prefix}_full"}
        ds = {r["model"]: r["score"] for r in results if r["id"] == f"{prefix}_ds700"}
        bits = []
        for m in models:
            bits.append(f"{m}: {full.get(m,'-')} - {ds.get(m,'-')}")
        emit(f"  {task:<12}" + "   ".join(bits))
    errs = [r for r in results if r["error"]]
    if errs:
        emit(f"\n调用错误 {len(errs)} 次: {[(r['model'], r['id'], r['error'][:60]) for r in errs]}")
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "report.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
