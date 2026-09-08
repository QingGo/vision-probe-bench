# -*- coding: utf-8 -*-
"""多模态 Agent 评测：看图 + 工具调用 + 多步迭代，验证 Agent 环路是否放大视觉能力"""
import base64
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI

from bench import ASSET_DIR, MODELS, b64, norm, score as bench_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_LOCK = threading.Lock()
CHART_DATA = json.load(open(os.path.join(BASE_DIR, "chart_data.json"), encoding="utf-8"))
THINK = False


# ---------- 工具定义 ----------
def tools_for(toolset, probe):
    if toolset == "chart_lookup":
        data = CHART_DATA[probe["images"][0]]
        return ([{"type": "function", "function": {"name": "get_value",
                  "description": "按类别标签查询该柱状图中此类的精确数值",
                  "parameters": {"type": "object", "properties": {
                      "category": {"type": "string", "description": "类别标签，例如 g000"}},
                      "required": ["category"]}}}],
                lambda a: str(data.get(a.get("category", ""), "无此类别")))
    if toolset == "ui_click":
        return ([{"type": "function", "function": {"name": "click",
                  "description": "点击界面上的某个按钮",
                  "parameters": {"type": "object", "properties": {
                      "button": {"type": "string", "description": "按钮文字"}},
                      "required": ["button"]}}}],
                lambda a: f"已点击按钮：{a.get('button')}")
    if toolset == "form_submit":
        return ([{"type": "function", "function": {"name": "submit",
                  "description": "提交表单，字段顺序为 name, phone, email",
                  "parameters": {"type": "object", "properties": {
                      "name": {"type": "string"}, "phone": {"type": "string"}, "email": {"type": "string"}},
                      "required": ["name", "phone", "email"]}}}],
                lambda a: "提交成功")
    if toolset == "verify":
        gt = probe["gt"]
        return ([{"type": "function", "function": {"name": "verify",
                  "description": "提交你推测的柱子数值，返回 correct 或 wrong",
                  "parameters": {"type": "object", "properties": {
                      "value": {"type": "integer"}}, "required": ["value"]}}}],
                lambda a: "correct" if str(a.get("value")) == str(gt) else "wrong，请重新仔细看图后重试")
    return [], lambda a: ""


def score_agent(probe, answer, last_args):
    if probe["score_type"] == "form_args":
        gt = probe["gt"].split("|")
        got = [last_args.get(k, "") for k in ("name", "phone", "email")]
        if all(g for g in got):
            return 1.0 if all(norm(g) == norm(v) for g, v in zip(gt, got)) else 0.0
        a = answer or ""
        phone = re.search(r"\d{11}", a)
        email = re.search(r"[\w.]+@[\w.]+", a)
        name = re.search(r"姓名[:：]?\s*([\u4e00-\u9fa5A-Za-z]+)", a)
        got = [name.group(1) if name else "", phone.group(0) if phone else "", email.group(0) if email else ""]
        return 1.0 if all(g and norm(g) == norm(v) for g, v in zip(got, gt)) else 0.0
    return bench_score(probe, answer)


def run_agent(cfg, probe, max_steps=6, max_retries=3):
    client = OpenAI(api_key=os.environ[cfg["key_env"]], base_url=cfg["base_url"])
    tools, fn = tools_for(probe["toolset"], probe)
    content = [{"type": "text", "text": probe["question"]}]
    for img in probe["images"]:
        content.append({"type": "image_url", "image_url": {"url": b64(img)}})
    messages = [{"role": "user", "content": content}]
    kwargs = dict(
        model=cfg["label"],
        tools=tools,
        tool_choice="auto",
    )
    if THINK:
        kwargs["extra_body"] = {"thinking": {"type": "enabled"}}
        if cfg["label"].startswith("mimo"):
            kwargs["max_completion_tokens"] = 65536
        else:
            kwargs["max_tokens"] = 65536
            kwargs["reasoning_effort"] = "max"
    else:
        kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
        if cfg["label"].startswith("mimo"):
            kwargs["max_completion_tokens"] = 8192
        else:
            kwargs["max_tokens"] = 8192
    pt_tot = ct_tot = rt_tot = 0
    steps = []
    final = ""
    last_args = {}
    t0 = time.time()
    for step in range(max_steps):
        last = None
        for attempt in range(max_retries):
            try:
                resp = client.chat.completions.create(messages=messages, **kwargs)
                break
            except Exception as e:  # noqa: BLE001
                last = e
                time.sleep(2 * (attempt + 1))
        else:
            return "", pt_tot, ct_tot, rt_tot, time.time() - t0, [f"ERR {str(last)[:100]}"], {}, 0.0
        msg = resp.choices[0].message
        pt_tot += resp.usage.prompt_tokens
        ct_tot += resp.usage.completion_tokens
        if getattr(resp.usage, "completion_tokens_details", None) is not None:
            rt_tot += getattr(resp.usage.completion_tokens_details, "reasoning_tokens", 0) or 0
        content_text = (msg.content or "").strip()
        if not msg.tool_calls:
            final = content_text
            break
        asst = {"role": "assistant", "content": content_text,
                "tool_calls": [{"id": tc.id, "type": "function",
                                "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                               for tc in msg.tool_calls]}
        rc = getattr(msg, "reasoning_content", None)
        if rc:
            asst["reasoning_content"] = rc
        messages.append(asst)
        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except Exception:
                args = {}
            result = fn(args)
            if tc.function.name == "submit":
                last_args = args
            steps.append(f"{tc.function.name}({json.dumps(args, ensure_ascii=False)}) -> {result}")
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
    return final, pt_tot, ct_tot, rt_tot, time.time() - t0, steps, last_args, cost_total(cfg, pt_tot, ct_tot)


def cost_total(cfg, pt, ct):
    return pt / 1e6 * cfg["price"]["in"] + ct / 1e6 * cfg["price"]["out"]


def run_model(key, cfg, probes, outfile, resume, workers=1):
    done = {}
    if resume and os.path.exists(outfile):
        for line in open(outfile, encoding="utf-8"):
            r = json.loads(line)
            done[r["id"]] = r["score"]
    todo = [p for p in probes if not (resume and p["id"] in done and done[p["id"]] == 1.0)]

    def work(p):
        final, pt, ct, rt, dt, steps, args, c = run_agent(cfg, p)
        sc = score_agent(p, final, args)
        rec = {"model": key, "id": p["id"], "task": p["task"], "score": round(sc, 4),
               "answer": final[:400], "gt": p["gt"], "steps": steps,
               "prompt_tokens": pt, "completion_tokens": ct, "reasoning_tokens": rt,
               "latency_s": round(dt, 2), "cost_yuan": round(c, 6)}
        with open(outfile, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
        stat = "OK" if sc == 1 else "FAIL"
        with _LOCK:
            print(f"  {p['id']:20s} {stat:4s} {sc:6.3f}  {p['task']:12s} steps={len(steps)} cost={rec['cost_yuan']:8.6f}元 {dt:5.1f}s", flush=True)

    if workers > 1 and len(todo) > 1:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            list(ex.map(work, todo))
    else:
        for p in todo:
            work(p)


def load_results():
    rows = []
    for fn in sorted(os.listdir(BASE_DIR)):
        if fn.startswith("agent_results_") and fn.endswith(".jsonl"):
            for line in open(os.path.join(BASE_DIR, fn), encoding="utf-8"):
                rows.append(json.loads(line))
    dedup = {}
    for r in rows:
        dedup[(r["model"], r["id"])] = r
    return list(dedup.values())


def report(results):
    tasks = ["chart_lookup", "ui_click", "form_submit", "verify"]
    print("=" * 70)
    print(f"{'任务':<16}{'题数':>4}{'deepseek-agent':>16}{'mimo-agent':>16}")
    t_acc = {"deepseek": 0.0, "mimo": 0.0}
    t_cnt = {"deepseek": 0, "mimo": 0}
    t_cost = {"deepseek": 0.0, "mimo": 0.0}
    for task in tasks:
        row = {"deepseek": [], "mimo": []}
        for r in results:
            if r["task"] == task:
                row[r["model"]].append(r)
        parts = []
        for m in ("deepseek", "mimo"):
            rs = row[m]
            if not rs:
                parts.append("-")
                continue
            s = sum(r["score"] for r in rs) / len(rs)
            t_acc[m] += s * len(rs)
            t_cnt[m] += len(rs)
            t_cost[m] += sum(r["cost_yuan"] for r in rs)
            parts.append(f"{s:.3f} ({sum(1 for r in rs if r['score']==1)}/{len(rs)})")
        print(f"{task:<16}{len(row['deepseek']):>4}{parts[0]:>16}{parts[1]:>16}")
    print("-" * 70)
    print(f"{'整体':<16}{sum(t_cnt.values()):>4}{t_acc['deepseek']/max(t_cnt['deepseek'],1):>16.3f}{t_acc['mimo']/max(t_cnt['mimo'],1):>16.3f}")
    print(f"{'总成本(元)':<16}{'':>4}{t_cost['deepseek']:>16.6f}{t_cost['mimo']:>16.6f}")
    print()
    print("放大对比（同图：纯感知 vs Agent）：")
    for r in results:
        if r["task"] == "chart_lookup":
            base = {"chart_lookup_0": 0.333, "chart_lookup_1": 0.333, "chart_lookup_2": 0.333}
            print(f"  {r['id']:<16} {r['model']:<10} 纯感知≈{base.get(r['id'],'-')}  Agent={r['score']}")


def main():
    args = sys.argv[1:]
    global THINK
    THINK = "--think" in args
    manifest = "agent_probes.json"
    model_only = None
    resume = "--resume" in args
    workers = 1
    for a in args:
        if a.endswith(".json"):
            manifest = a
        elif a.startswith("--model="):
            model_only = a.split("=", 1)[1]
        elif a in MODELS:
            model_only = a
        elif a.startswith("--workers="):
            workers = int(a.split("=", 1)[1])
    probes = json.load(open(os.path.join(BASE_DIR, manifest), encoding="utf-8"))["probes"]
    keys = [model_only] if model_only else list(MODELS.keys())
    for key in keys:
        cfg = MODELS[key]
        outfile = os.path.join(BASE_DIR, f"agent_results_{key}{'_think' if THINK else ''}.jsonl")
        print(f"\n=== Agent[{cfg['label']}]{' THINK' if THINK else ''} ({len(probes)} probes, workers={workers}) ===", flush=True)
        run_model(key, cfg, probes, outfile, resume, workers)
    if model_only is None:
        report(load_results())


if __name__ == "__main__":
    main()
