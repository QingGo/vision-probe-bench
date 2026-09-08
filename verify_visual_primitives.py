# -*- coding: utf-8 -*-
"""视觉原语验证：dense count + box / path trace + point / polygon / 显式触发测试（v4.1 flash）"""
import base64
import json
import os
import random

from PIL import Image, ImageDraw, ImageFont
from openai import OpenAI

ASSET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
FONT_EN = "/System/Library/Fonts/Supplemental/Arial.ttf"
rng = random.Random(7)


def font(p, s):
    return ImageFont.truetype(p, s)


def b64(png):
    return "data:image/png;base64," + base64.b64encode(open(os.path.join(ASSET_DIR, png), "rb").read()).decode()


# ---- 图 1：12 red circles / 8 blue squares / 5 green triangles（坐标已知） ----
img = Image.new("RGB", (600, 400), "white")
d = ImageDraw.Draw(img)
reds, blues, tris = [], [], []
def place(coll, n, color, kind):
    while len(coll) < n:
        x, y = rng.randint(30, 550), rng.randint(30, 350)
        if all(abs(x - a) > 45 or abs(y - b) > 45 for a, b, _ in coll):
            coll.append((x, y, color))
nf = font(FONT_EN, 16)
for coll in (reds, blues, tris):
    pass
for coll, n, color, kind in ((reds, 12, (220, 40, 40), "circle"), (blues, 8, (40, 90, 220), "square"),
                             (tris, 5, (40, 180, 90), "tri")):
    while len(coll) < n:
        x, y = rng.randint(30, 550), rng.randint(30, 350)
        if all(abs(x - a) > 45 or abs(y - b) > 45 for a, b, _ in coll):
            coll.append((x, y, color))
    if kind == "circle":
        for x, y, c in coll:
            d.ellipse([x - 20, y - 20, x + 20, y + 20], fill=c)
    elif kind == "square":
        for x, y, c in coll:
            d.rectangle([x - 18, y - 18, x + 18, y + 18], fill=c)
    else:
        for x, y, c in coll:
            d.polygon([(x - 22, y + 16), (x + 22, y + 16), (x, y - 22)], fill=c)
img.save(os.path.join(ASSET_DIR, "vp_count.png"))
print("reds:", len(reds), blues, tris)

# ---- 图 2：折线路径（已知关键点） ----
img2 = Image.new("RGB", (400, 350), "white")
d2 = ImageDraw.Draw(img2)
pts = [(50, 300), (130, 100), (250, 250), (350, 100)]
d2.line(pts, fill=(0, 0, 0), width=5)
d2.ellipse([50 - 14, 300 - 14, 50 + 14, 300 + 14], fill=(40, 180, 90))
d2.rectangle([350 - 16, 100 - 16, 350 + 16, 100 + 16], fill=(220, 40, 40))
img2.save(os.path.join(ASSET_DIR, "vp_path.png"))
print("path pts:", pts)

# ---- 图 3：绿色三角形（polygon） ----
img3 = Image.new("RGB", (300, 300), "white")
d3 = ImageDraw.Draw(img3)
d3.polygon([(220, 130), (330, 130), (275, 70)], fill=(40, 180, 90))
img3.save(os.path.join(ASSET_DIR, "vp_poly.png"))

# ---- 调 API ----
client = OpenAI(api_key=os.environ["DS41_API_KEY"], base_url="https://api.deepseek.com")
def ask(png, text, max_tokens=8192):
    msg = client.chat.completions.create(
        model="deepseek-v4.1-flash-expires-on-0910",
        messages=[{"role": "user", "content": [
            {"type": "text", "text": text},
            {"type": "image_url", "image_url": {"url": b64(png)}},
        ]}],
        max_tokens=max_tokens,
        extra_body={"thinking": {"type": "enabled"}},
        reasoning_effort="high",
    )
    m = msg.choices[0].message
    return (m.reasoning_content or "") + "\n[CONTENT]\n" + (m.content or "")

tests = [
    ("T1 密集计数+box(显式触发)", "vp_count.png",
     "仔细数图中有多少个红色圆形。请逐个找出每一个红色圆形，并用 <｜ref｜>红色圆形<｜/ref｜><｜box｜>[[x1,y1,x2,y2]]<｜/box｜> 标注其包围盒（图片 600×400，像素坐标）。思考过程中也必须逐个输出标注。最终输出所有标注和总数。"),
    ("T2 路径追踪+point", "vp_path.png",
     "从绿色圆点出发沿黑色折线追踪到红色方块。思考过程中用 <｜point｜>[[x,y]]<｜/point｜> 逐步记录经过的关键点（像素坐标）。"),
    ("T3 polygon", "vp_poly.png",
     "找出图中的绿色三角形，用 <｜polygon｜>[[x1,y1,x2,y2,x3,y3]]<｜/polygon｜> 输出其顶点坐标。"),
    ("T4 自然描述(不触发)", "vp_count.png", "这张图里有什么？请描述一下。"),
]
out = []
for name, png, q in tests:
    print(f"\n===== {name} =====")
    try:
        resp = ask(png, q)
        print(resp[:1200])
        out.append({"name": name, "resp": resp})
    except Exception as e:
        print("ERR", str(e)[:200])
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "vp_results.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
