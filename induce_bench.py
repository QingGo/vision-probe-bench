# -*- coding: utf-8 -*-
"""视觉原语诱导实验评测器：A/B/C 三臂对比，每题 2 次采样，deepseek 与 flash-vision-exp 通用同一 key"""
import base64
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI

from bench import ASSET_DIR, b64, score as bench_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODELS = {
    "deepseek41": {"label": "deepseek-v4.1-flash-expires-on-0910", "key_env": "DS41_API_KEY",
                   "price": {"in": 1.5, "out": 4.5}},
    "deepseek": {"label": "deepseek-v4-flash-vision-exp", "key_env": "DS41_API_KEY",
                 "price": {"in": 1.5, "out": 4.5}},
}
_LOCK = threading.Lock()
SAMPLES = 2


def call_api(cfg, probe):
    client = OpenAI(api_key=os.environ[cfg["key_env"]], base_url="https://api.deepseek.com")
    content = [{"type": "text", "text": probe["question"]}]
    for png in probe["images"]:
        content.append({"type": "image_url", "image_url": {"url": b64(png)}})
    kwargs = dict(
        model=cfg["label"],
        messages=[{"role": "user", "content": content}],
        max_tokens=8192,
        extra_body={"thinking": {"type": "enabled"}},
        reasoning_effort="high",
    )
    for attempt in range(4):
        try:
            t0 = time.time()
            resp = client.chat.completions.create(**kwargs)
            return (resp.choices[0].message.content or "").strip(), resp.usage.prompt_tokens, \
                   resp.usage.completion_tokens, time.time() - t0, None
        except Exception as e:  # noqa: BLE001
            time.sleep(3 * (attempt + 1))
            last = e
    return "", 0, 0, 0.0, str(last)[:200]


def extract_answer(text, arm):
    if arm == "C":
        m = re.search(r"ANSWER\s*[:：]\s*(.+)", text, flags=re.S)
        if m:
            return m.group(1).strip()
        m = re.search(r"[^A-Za-z]*ANSWER\s*[:：]\s*(.+)", text, flags=re.S)
        if m:
            return m.group(1).strip()
    return text


def score_probe(probe, answer):
    return bench_score(probe, answer)


def cost(cfg, pt, ct):
    return pt / 1e6 * cfg["price"]["in"] + ct / 1e6 * cfg["price"]["out"]


def run_model(key, cfg, probes, outfile, resume=False):
    done = {}
    if resume and os.path.exists(outfile):
        for line in open(outfile, encoding="utf-8"):
            r = json.loads(line)
            done[(r["id"], r["sample"])] = r["score"]
    todo = []
    for p in probes:
        for s in range(SAMPLES):
            if resume and (p["id"], s) in done:
                continue
            todo.append((p, s))

    def work(item):
        p, s = item
        raw, pt, ct, dt, err = call_api(cfg, p)
        ans = extract_answer(raw, p["arm"])
        sc = 0.0 if err else score_probe(p, ans)
        rec = {"model": key, "id": p["id"], "base_id": p["base_id"], "task": p["task"],
               "arm": p["arm"], "sample": s, "score": round(sc, 4),
               "answer": raw[:400], "extracted": ans[:400], "gt": p["gt"], "error": err,
               "prompt_tokens": pt, "completion_tokens": ct,
               "latency_s": round(dt, 2), "cost_yuan": round(cost(cfg, pt, ct), 6)}
        with open(outfile, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        stat = "OK" if sc == 1 else ("PART" if 0 < sc < 1 else "ERR" if err else "FAIL")
        with _LOCK:
            print(f"  [{p['id']:20s}] {p['arm']} s{s} {stat:4s} {sc:6.3f}  cost={rec['cost_yuan']:8.6f}元 {dt:5.1f}s", flush=True)

    with ThreadPoolExecutor(max_workers=30) as ex:
        list(ex.map(work, todo))


def load_results():
    rows = []
    for fn in sorted(os.listdir(BASE_DIR)):
        if fn.startswith("induce_results_") and fn.endswith(".jsonl"):
            for line in open(os.path.join(BASE_DIR, fn), encoding="utf-8"):
                rows.append(json.loads(line))
    return rows


def report():
    rows = load_results()
    models = sorted(set(r["model"] for r in rows))
    tasks = sorted(set(r["task"] for r in rows), key=lambda t: (t == "crowd_chart", t))
    print("=" * 78)
    for m in models:
        print(f"--- {m} ---")
        print(f"{'任务':<14}{'题':<14}{'A均值':>8}{'B均值':>8}{'C均值':>8}{'C-A':>8}{'C-B':>8}")
        ag = {"A": [], "B": [], "C": []}
        for t in tasks:
            rows_t = [r for r in rows if r["model"] == m and r["task"] == t]
            names = sorted(set(r["id"] for r in rows_t))
            if not names:
                continue
            name = names[0]
            line = f"{t:<14}{name:.<14}"
            arms = {}
            for arm in ("A", "B", "C"):
                scs = [r["score"] for r in rows_t if r["arm"] == arm]
                arms[arm] = sum(scs) / len(scs) if scs else None
                line += f"{(f'{arms[arm]:.3f}' if arms[arm] is not None else '-'):>8}"
            ca = (arms["C"] or 0) - (arms["A"] or 0)
            cb = (arms["C"] or 0) - (arms["B"] or 0)
            line += f"{ca:>8.3f}{cb:>8.3f}"
            print(line)
            for a in ("A", "B", "C"):
                if arms[a] is not None:
                    ag[a].append(arms[a])
        print(f"--- 整体均值 ---")
        for a in ("A", "B", "C"):
            v = sum(ag[a]) / len(ag[a]) if ag[a] else 0
            print(f"  {a}: {v:.3f}", end="")
        print()
        cost = sum(r["cost_yuan"] for r in rows if r["model"] == m)
        print(f"  总成本: {cost:.4f} 元")
        print()


def main():
    args = sys.argv[1:]
    manifest = "probes_induce.json"
    model_only = None
    for a in args:
        if a.endswith(".json"):
            manifest = a
        elif a in MODELS:
            model_only = a
    probes = json.load(open(os.path.join(BASE_DIR, manifest), encoding="utf-8"))["probes"]
    keys = [model_only] if model_only else list(MODELS.keys())
    for key in keys:
        cfg = MODELS[key]
        outfile = os.path.join(BASE_DIR, f"induce_results_{key}.jsonl")
        print(f"\n=== Induce [{cfg['label']}] ({len(probes)} probes x {SAMPLES} samples) ===", flush=True)
        run_model(key, cfg, probes, outfile)
    report()


if __name__ == "__main__":
    main()
