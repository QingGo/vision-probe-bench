# -*- coding: utf-8 -*-
"""坐标精度量化：三档尺寸 × 随机圆点 × 告知/不告知尺寸，测像素误差"""
import base64
import json
import os
import random

from PIL import Image, ImageDraw
from openai import OpenAI

ASSET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
COLORS = [
    ("红", (220, 40, 40)), ("蓝", (40, 90, 220)), ("绿", (40, 180, 90)),
    ("橙", (240, 140, 20)), ("紫", (150, 60, 200)), ("青", (0, 170, 180)),
]
# 尺寸档位：(name, W, H)  300x300=9万px(放大档) 600x500=30万px(不缩放) 1600x1400=224万px(缩小档)
SIZES = [(("small_300x300", 300, 300)), ("mid_600x500", 600, 500), ("big_1600x1400", 1600, 1400)]

rng = random.Random(2026)
variants = []
for name, W, H in SIZES:
    pts = []
    margin = int(min(W, H) * 0.15)
    for cname, col in COLORS:
        while True:
            x, y = rng.randint(margin, W - margin), rng.randint(margin, H - margin)
            if all((x - px) ** 2 + (y - py) ** 2 > (min(W, H) * 0.09) ** 2 for px, py, _ in pts):
                pts.append((x, y, cname))
                break
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    r = max(8, int(min(W, H) * 0.028))
    for x, y, cname in pts:
        col = dict(COLORS)[cname]
        d.ellipse([x - r, y - r, x + r, y + r], fill=col)
    fn = f"coord_{name}.png"
    img.save(os.path.join(ASSET_DIR, fn))
    variants.append((name, W, H, fn, pts, r))
    print("variant", name, W, "x", H, "pts", [(cname, x, y) for x, y, cname in pts])

client = OpenAI(api_key=os.environ["DS41_API_KEY"], base_url="https://api.deepseek.com")

def ask(fn, W, H, with_size):
    q = "图中有 6 个不同颜色的圆点（红、蓝、绿、橙、紫、青）。请逐个输出每个圆点圆心的像素坐标。格式：颜色名 + <｜point｜>[[x,y]]<｜/point｜>，每行一个。"
    if with_size:
        q = f"图片尺寸是 {W}×{H} 像素。" + q
    m = client.chat.completions.create(
        model="deepseek-v4.1-flash-expires-on-0910",
        messages=[{"role": "user", "content": [
            {"type": "text", "text": q},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(open(os.path.join(ASSET_DIR, fn), "rb").read()).decode()}},
        ]}],
        max_tokens=4096,
        extra_body={"thinking": {"type": "enabled"}},
        reasoning_effort="high",
    )
    return m.choices[0].message.content or ""

import re
results = []
for name, W, H, fn, pts, r in variants:
    for with_size in (True, False):
        print(f"\n===== {name}  {'告知尺寸' if with_size else '不告知'} =====")
        content = ask(fn, W, H, with_size)
        print(content[:500].replace("\n", " / "))
        preds = {}
        for cname, _ in COLORS:
            m = re.search(re.escape(cname) + r"[^\d]*<｜point｜>\[\[(\d+),(\d+)\]\]<｜point｜>", content)
            if m:
                preds[cname] = (int(m.group(1)), int(m.group(2)))
        errs = []
        for x, y, cname in pts:
            if cname in preds:
                px, py = preds[cname]
                errs.append(((px - x) ** 2 + (py - y) ** 2) ** 0.5)
        rec = {"variant": name, "with_size": with_size, "truth": [(x, y, c) for x, y, c in pts],
               "preds": {k: list(v) for k, v in preds.items()},
               "errors": [round(e, 1) for e in errs],
               "answer": content[:2000]}
        results.append(rec)
        if errs:
            print(f"  点数: {len(errs)}/6  平均误差: {sum(errs)/len(errs):.1f}px  中位: {sorted(errs)[len(errs)//2]:.1f}px  最大: {max(errs):.1f}px")
        else:
            print("  未解析到坐标")
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "coord_results.json"), "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=1)
