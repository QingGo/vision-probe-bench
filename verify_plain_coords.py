# -*- coding: utf-8 -*-
"""纯文本坐标精度（不诱导视觉原语）：3 档尺寸 × 告知/不告知尺寸，输出 (x, y) 普通文本"""
import base64
import json
import os
import re

from openai import OpenAI

ASSET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")

# 复用坐标矩阵实验中生成的图与真值（coord_results.json 已存 truth）
truth_map = {}
for rec in json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "coord_results.json"), encoding="utf-8")):
    truth_map[rec["variant"]] = {"size": (rec["variant"] in ("small_300x300",)) and None or None,
                                 "truth": [(t[0], t[1]) for t in rec["truth"]]}
# 实际尺寸
SIZES = {"small_300x300": (300, 300), "mid_600x500": (600, 500), "big_1600x1400": (1600, 1400)}
VARIANT_ORDER = {"small_300x300": "small_300x300", "mid_600x500": "mid_600x500", "big_1600x1400": "big_1600x1400"}

COLOR_ORDER = ["红", "蓝", "绿", "橙", "紫", "青"]

client = None


def ask(model, fn, W, H, with_size):
    q = ("图中有 6 个不同颜色的圆点。请按 红、蓝、绿、橙、紫、青 的顺序，每行输出一个圆点的圆心坐标，"
         "格式（纯文本，不要任何特殊标记或标签）：(x, y)")
    if with_size:
        q = f"图片尺寸是 {W}×{H} 像素。" + q
    m = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": [
            {"type": "text", "text": q},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(
                open(os.path.join(ASSET_DIR, f"coord_{VARIANT_ORDER[fn]}.png"), "rb").read()).decode()}},
        ]}],
        max_tokens=4096,
        extra_body={"thinking": {"type": "enabled"}},
        reasoning_effort="high",
    )
    return m.choices[0].message.content or ""


def parse(content):
    nums = re.findall(r"\((\d+)[,，]\s*(\d+)\)", content)
    if not nums:
        nums = re.findall(r"(\d+)[,，]\s*(\d+)", content)
    return [(int(a), int(b)) for a, b in nums]


if __name__ == "__main__":
    import os as _os
    client = OpenAI(api_key=_os.environ["DS41_API_KEY"], base_url="https://api.deepseek.com")
    models = [("deepseek-v4.1-flash-expires-on-0910", "deepseek41"), ("deepseek-v4-flash-vision-exp", "deepseek")]
    results = []
    for model, mname in models:
        print(f"\n===== {mname} =====")
        for vname in ("small_300x300", "mid_600x500", "big_1600x1400"):
            W, H = SIZES[vname]
            truth = truth_map[vname]["truth"]
            for with_size in (True, False):
                content = ask(model, vname, W, H, with_size)
                preds = parse(content)
                errs = []
                for i, (tx, ty) in enumerate(truth):
                    if i < len(preds):
                        px, py = preds[i]
                        errs.append(round(((px - tx) ** 2 + (py - ty) ** 2) ** 0.5, 1))
                stat = f"解析{len(preds)}/6"
                if errs:
                    stat += f" 平均{sum(errs)/len(errs):.1f}px 中位{sorted(errs)[len(errs)//2]:.1f} 最大{max(errs):.1f}"
                print(f"  {vname} {'告知' if with_size else '不告知'}: {stat}")
                firstlines = content.splitlines()[:7]
                print("    ", " / ".join(firstlines)[:180])
                results.append({"model": mname, "variant": vname, "with_size": with_size,
                                "errors": errs, "answer": content[:1500]})
    json.dump(results, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "plain_coord_results.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
