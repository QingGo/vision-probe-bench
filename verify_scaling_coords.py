# -*- coding: utf-8 -*-
"""验证自动缩放对坐标的影响：同一路径图 × 三个尺寸档位"""
import base64
import json
import os

from PIL import Image, ImageDraw
from openai import OpenAI

ASSET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")

# 基本几何：线段 (50,300)->(130,100)->(250,250)->(350,100)，绿起点 (50,300)，红终点 (350,100)
BASE_PTS = [(50, 300), (130, 100), (250, 250), (350, 100)]
SIZES = [("native_400x350", (400, 350), 1.0), ("mid_600x525", (600, 525), 1.5), ("big_1600x1400", (1600, 1400), 4.0)]

variants = []
for name, (W, H), s in SIZES:
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    pts = [(int(x * s), int(y * s)) for x, y in BASE_PTS]
    d.line(pts, fill=(0, 0, 0), width=int(5 * s))
    d.ellipse([pts[0][0] - 14 * s, pts[0][1] - 14 * s, pts[0][0] + 14 * s, pts[0][1] + 14 * s], fill=(40, 180, 90))
    d.rectangle([pts[-1][0] - 16 * s, pts[-1][1] - 16 * s, pts[-1][0] + 16 * s, pts[-1][1] + 16 * s], fill=(220, 40, 40))
    fn = f"vp_scale_{name}.png"
    img.save(os.path.join(ASSET_DIR, fn))
    variants.append((name, W, H, fn, pts[0], pts[-1]))
    print("variant:", name, W, "x", H, "start", pts[0], "end", pts[-1])

client = OpenAI(api_key=os.environ["DS41_API_KEY"], base_url="https://api.deepseek.com")

def ask(fn):
    m = client.chat.completions.create(
        model="deepseek-v4.1-flash-expires-on-0910",
        messages=[{"role": "user", "content": [
            {"type": "text", "text": "图中有绿色圆点（起点）和红色方块（终点），由黑色折线相连。请回答（像素坐标，整数）：1) 图片尺寸是宽几像素×高几像素？2) 绿色圆点中心坐标 3) 红色方块中心坐标。用 <｜point｜>[[x,y]]<｜/point｜> 输出两个点。"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(open(os.path.join(ASSET_DIR, fn), "rb").read()).decode()}},
        ]}],
        max_tokens=4096,
        extra_body={"thinking": {"type": "enabled"}},
        reasoning_effort="high",
    )
    msg = m.choices[0].message
    return (msg.content or "")

results = []
for name, W, H, fn, start, end in variants:
    print(f"\n===== {name} (真值 {W}x{H}, start={start}, end={end}) =====")
    try:
        content = ask(fn)
        print(content[:600])
        results.append({"variant": name, "truth": [W, H, start, end], "answer": content})
    except Exception as e:
        print("ERR", str(e)[:150])
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "vp_scale_results.json"), "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=1)
