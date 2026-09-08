# -*- coding: utf-8 -*-
"""合成探针图片生成器：生成带已知答案(GT)的测试图，写入 assets/*.png 和 probes.json"""
import base64
import io
import json
import math
import os
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont, ImageFilter

ASSET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
FONT_EN = "/System/Library/Fonts/Supplemental/Arial.ttf"
FONT_CN = "/System/Library/Fonts/Hiragino Sans GB.ttc"
FONT_SYM = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data", "fonts", "ttf", "DejaVuSans.ttf")

rng = random.Random(42)
probes = []


def font(path, size):
    return ImageFont.truetype(path, size)


def save_img(img, name):
    p = os.path.join(ASSET_DIR, name)
    img.save(p)
    return name


def b64(png_path):
    with open(os.path.join(ASSET_DIR, png_path), "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode()


def add(id_, task, question, gt, images, score_type="char_acc", extra=None):
    probes.append({"id": id_, "task": task, "question": question, "gt": gt,
                   "images": images, "score_type": score_type,
                   "extra": extra or {}})


def render_text(text, size, fpath, canvas_w, bg=(255, 255, 255), fg=(0, 0, 0)):
    """把文本渲染到宽画布上（模拟单据/长行文字），返回 PIL Image"""
    img = Image.new("RGB", (canvas_w, max(size * 2 + 20, 64)), bg)
    d = ImageDraw.Draw(img)
    d.text((10, 14), text, font=font(fpath, size), fill=fg)
    return img


def random_str(n, chars="ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnpqrstuvwxyz23456789"):
    return "".join(rng.choice(chars) for _ in range(n))


def random_float(lo=5.0, hi=98.0):
    return round(rng.uniform(lo, hi), 2)


# ---------- 1. 英文 OCR（小字号 + 宽画布 → DS 会缩小） ----------
sent_en = "The quick brown fox jumps over the lazy dog and flies away"
ocr_en = [
    (9, random_str(40)),
    (12, random_str(40)),
    (11, sent_en),
    (15, sent_en),
]
for size, txt in ocr_en:
    name = f"ocr_en_{size}.png"
    save_img(render_text(txt, size, FONT_EN, 1900), name)
    add(f"ocr_en_{size}", "ocr_en", "请逐字转录图片中的全部文字，只输出文字本身，不要加引号或解释。",
        txt, [name], "char_acc")

# ---------- 2. 中文 OCR ----------
sent_cn = [
    "北方的冬天格外寒冷，清晨的湖面结满厚冰",
    "智能手表可以实时监测心率血压与睡眠质量",
    "她把行李箱放在候车室的座位上，然后去买票",
    "工程师正在调试新交付的生产线，注意安全",
]
for i, txt in enumerate(sent_cn):
    size = 10 if i < 2 else 13
    name = f"ocr_cn_{i}.png"
    save_img(render_text(txt, size, FONT_CN, 1900), name)
    add(f"ocr_cn_{i}", "ocr_cn", "请逐字转录图片中的全部中文文字，只输出文字本身。",
        txt, [name], "char_acc")

# ---------- 3. 中文场景文字（彩色底 + 旋转 + 噪声） ----------
scene_cn = [
    "奶茶店今日买一送一",
    "扫码支付立减五元",
    "地铁站出口向左转",
]
for i, txt in enumerate(scene_cn):
    bg = Image.new("RGB", (900, 260), (rng.randint(60, 200), rng.randint(60, 200), rng.randint(60, 200)))
    for _ in range(12000):
        x, y = rng.randint(0, 899), rng.randint(0, 259)
        c = (rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255))
        bg.putpixel((x, y), c)
    txt_img = Image.new("RGBA", (1000, 200), (0, 0, 0, 0))
    ImageDraw.Draw(txt_img).text((10, 40), txt, font=font(FONT_CN, 40), fill=(255, 255, 255, 255))
    txt_img = txt_img.rotate(rng.choice([-3, 3, 5, -5]), expand=True, resample=Image.BICUBIC)
    bg.paste(txt_img, (60, 20), txt_img)
    bg = bg.filter(ImageFilter.GaussianBlur(0.4))
    name = f"scene_cn_{i}.png"
    save_img(bg, name)
    add(f"scene_cn_{i}", "scene_cn", "请识别图片中的白色文字并转录，只输出文字本身。",
        txt, [name], "char_acc")

# ---------- 4. 密集数字表格（小字号，DS 缩小后更难） ----------
for t in range(3):
    rows, cols = 8, 6
    data = [[random_float() for _ in range(cols)] for _ in range(rows)]
    cell_w, cell_h = 150, 46
    margin_l, margin_t = 110, 70
    W = margin_l + cols * cell_w + 20
    H = margin_t + rows * cell_h + 20
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    fs = 13
    for r in range(rows):
        d.text((30, margin_t + r * cell_h + 14), str(r + 1), font=font(FONT_EN, fs), fill=(0, 0, 0))
        for c in range(cols):
            v = data[r][c]
            d.text((margin_l + c * cell_w + 40, margin_t + r * cell_h + 14),
                   f"{v:.2f}", font=font(FONT_EN, fs), fill=(0, 0, 0))
    for c in range(cols):
        d.text((margin_l + c * cell_w + 60, 30), chr(65 + c), font=font(FONT_EN, 16), fill=(0, 0, 0))
    for x in range(0, W, cell_w):
        d.line([(margin_l + x, margin_t), (margin_l + x, H)], fill=(0, 0, 0), width=1)
    for y in range(0, H, cell_h):
        d.line([(0, margin_t + y), (W, margin_t + y)], fill=(0, 0, 0), width=1)
    name = f"dense_table_{t}.png"
    save_img(img, name)
    qs = [(2, 2), (5, 4), (7, 1)]
    for j, (r, c) in enumerate(qs):
        add(f"dense_table_{t}_{j}", "dense_table",
            f"图中表格第 {r+1} 行、第 {chr(65+c)} 列（左起第 {c+1} 列，行号从 1 开始，列字母 A 起）的数值是多少？只输出数字。",
            f"{data[r][c]:.2f}", [name], "numeric")

# ---------- 5. 拥挤柱状图（22 类，标签小） ----------
chart0_top = None
chart_data = {}
for t in range(3):
    cats = [f"g{t}{i:02d}" for i in range(22)]
    vals = [rng.randint(5, 95) for _ in cats]
    chart_data[f"crowd_chart_{t}.png"] = {c: v for c, v in zip(cats, vals)}
    fig, ax = plt.subplots(figsize=(16, 5), dpi=120)
    ax.bar(range(len(cats)), vals, width=0.8)
    ax.set_xticks(range(len(cats)))
    ax.set_xticklabels(cats, rotation=90, fontsize=4.5)
    ax.set_yticks([])
    ax.set_ylim(0, 100)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.savefig(os.path.join(ASSET_DIR, f"crowd_chart_{t}.png"), bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    name = f"crowd_chart_{t}.png"
    hi = sorted(zip(vals, cats), reverse=True)
    top_cat = hi[0][1]
    if t == 0:
        chart0_top = top_cat
    mid_idx = 10
    mid_cat, mid_val = cats[mid_idx], vals[mid_idx]
    add(f"crowd_chart_{t}_top", "crowd_chart",
        "图中 x 轴标签是 g 开头的编号（例如 g000）。数值最高的类别标签是什么？只输出该标签（如 g000）。",
        top_cat, [name], "exact")
    add(f"crowd_chart_{t}_val", "crowd_chart",
        f"图中 x 轴标签为 {mid_cat} 的那根柱子，数值是多少？只输出整数。",
        str(mid_val), [name], "numeric")

# ---------- 6. 计数（网格抖动摆放，无重叠歧义） ----------
for n in [13, 19, 26, 31, 42, 55, 64, 77]:
    img = Image.new("RGB", (900, 900), "white")
    d = ImageDraw.Draw(img)
    palette = ["#E63946", "#457B9D", "#F4A261", "#2A9D8F", "#9D4EDD", "#E9C46A"]
    cols = math.ceil(math.sqrt(n * 900 * 900 / (36 * 36 * 900)))  # ~approx
    cols = int(math.ceil(math.sqrt(n * 1.6)))
    step = 900 // (cols + 1)
    placed = []
    idx = 0
    while len(placed) < n:
        for r in range(1, cols + 1):
            for c in range(1, cols + 1):
                if len(placed) >= n:
                    break
                cx = c * step + rng.randint(-4, 4)
                cy = r * step + rng.randint(-4, 4)
                if 15 < cx < 885 and 15 < cy < 885:
                    placed.append((cx, cy, palette[idx % len(palette)]))
                    idx += 1
    for cx, cy, col in placed:
        d.ellipse([cx - 13, cy - 13, cx + 13, cy + 13], fill=col)
    name = f"count_{n}.png"
    save_img(img, name)
    add(f"count_{n}", "counting", "图中有多少个彩色圆点？只输出数字。", str(n), [name], "numeric")

# ---------- 7. 空间推理（3×3 网格） ----------
cell = 150
for t, (rsq, csq, rbl, cbl, rstar, cstar) in enumerate([
    (1, 0, 2, 1, 0, 2),
    (2, 2, 0, 0, 1, 2),
    (0, 1, 2, 2, 2, 0),
    (1, 1, 0, 2, 2, 1),
    (2, 0, 1, 2, 0, 1),
]):
    img = Image.new("RGB", (cell * 3, cell * 3), "white")
    d = ImageDraw.Draw(img)
    for g in range(4):
        d.line([(0, g * cell), (cell * 3, g * cell)], fill=(0, 0, 0), width=2)
        d.line([(g * cell, 0), (g * cell, cell * 3)], fill=(0, 0, 0), width=2)
    def put(r, c, shape, color):
        x, y = c * cell, r * cell
        pad = 35
        if shape == "square":
            d.rectangle([x + pad, y + pad, x + cell - pad, y + cell - pad], fill=color)
        elif shape == "circle":
            d.ellipse([x + pad, y + pad, x + cell - pad, y + cell - pad], fill=color)
        else:
            d.text((x + 55, y + 45), "★", font=font(FONT_CN, 60), fill=color)
    put(rsq, csq, "square", "red")
    put(rbl, cbl, "circle", "blue")
    put(rstar, cstar, "star", "black")
    name = f"spatial_{t}.png"
    save_img(img, name)
    add(f"spatial_{t}", "spatial",
        "图中是一个 3×3 网格。请回答：红色方块在第几行第几列？（行、列都从 1 开始数，用“行,列”格式回答，如 2,3）",
        f"{rsq+1},{csq+1}", [name], "exact")

# ---------- 8. 多图对比 ----------
def dot_grid(count):
    img = Image.new("RGB", (700, 700), "white")
    d = ImageDraw.Draw(img)
    cols = int(math.ceil(math.sqrt(count * 1.6)))
    step = 700 // (cols + 1)
    placed = []
    i = 0
    for r in range(1, cols + 1):
        for c in range(1, cols + 1):
            if len(placed) >= count:
                break
            cx, cy = c * step + rng.randint(-4, 4), r * step + rng.randint(-4, 4)
            if 15 < cx < 685 and 15 < cy < 685:
                placed.append((cx, cy, "#E63946"))
                i += 1
    for cx, cy, col in placed:
        d.ellipse([cx - 11, cy - 11, cx + 11, cy + 11], fill=col)
    return img

m1, m2 = dot_grid(15), dot_grid(15)
n1 = "multi_same_a.png"; n2 = "multi_same_b.png"
save_img(m1, n1); save_img(m2, n2)
add("multi_same", "multi_image",
    "请比较这两张图片。两张图是否完全相同？只回答“相同”或“不同”。",
    "相同", [n1, n2], "exact")

m3, m4 = dot_grid(14), dot_grid(15)
n3 = "multi_cnt_a.png"; n4 = "multi_cnt_b.png"
save_img(m3, n3); save_img(m4, n4)
add("multi_diff_cnt", "multi_image",
    "请比较这两张图片。两张图是否完全相同？如果不同，两图中红色圆点的数量分别是什么？回答格式：相同或不同, 左图数量, 右图数量（如：不同, 14, 15）",
    "不同, 14, 15", [n3, n4], "exact")

# 颜色差异
def color_grid(color):
    img = Image.new("RGB", (400, 400), "white")
    d = ImageDraw.Draw(img)
    pad = 40
    d.rectangle([pad, pad, 400 - pad, 400 - pad], fill=color)
    d.ellipse([160, 160, 240, 240], fill="black")
    return img
save_img(color_grid((220, 50, 50)), "multi_col_a.png")
save_img(color_grid((60, 60, 210)), "multi_col_b.png")
add("multi_diff_color", "multi_image",
    "请比较这两张图片。两张图是否完全相同？只回答“相同”或“不同”。",
    "不同", ["multi_col_a.png", "multi_col_b.png"], "exact")

# ---------- 9. 图中数学 ----------
math_items = [
    ("math_eq", "3x + 7 = 22    求 x 的值", "5"),
    ("math_angle", "直角三角形，已知一个锐角为 35°，另一个锐角是多少度？", "55"),
    ("math_ops", "17 + 8 × 3 = ？", "41"),
    ("math_square", "正方形边长为 6，它的面积是多少？", "36"),
]
for i, (mid, txt, gt) in enumerate(math_items):
    img = Image.new("RGB", (900, 220), "white")
    d = ImageDraw.Draw(img)
    d.text((30, 55), txt, font=font(FONT_CN, 38), fill=(0, 0, 0))
    name = f"math_{i}.png"
    save_img(img, name)
    add(f"math_{i}", "math_img", "请解答图中的问题，只输出最终数值答案（整数）。", gt, [name], "numeric")

# ---------- 10. 细粒度符号 ----------
symbols = [
    ("α", "alpha"), ("β", "beta"), ("∂", "partial"), ("∇", "nabla"),
    ("∈", "element-of"), ("∉", "not-element-of"), ("⊆", "subset-of"),
    ("∪", "union"), ("∩", "intersection"), ("∑", "sum"), ("√", "sqrt"),
]
options = "、".join([s[1] for s in symbols])
for i, (sym, name_) in enumerate(symbols[:9]):
    img = Image.new("RGB", (300, 300), "white")
    d = ImageDraw.Draw(img)
    d.text((110, 80), sym, font=font(FONT_SYM, 130), fill=(0, 0, 0))
    nm = f"symbol_{i}.png"
    save_img(img, nm)
    add(f"symbol_{i}", "fine_symbol",
        f"图片中央显示的是哪个数学符号？从以下选项中选择并只输出英文名：{options}",
        name_, [nm], "exact")

# ---------- 11. 分辨率 A/B：同一图的高清版 vs 缩小版 ----------
def downscale(png_path, max_w):
    img = Image.open(os.path.join(ASSET_DIR, png_path))
    if img.width > max_w:
        h = int(img.height * max_w / img.width)
        img = img.resize((max_w, h), Image.LANCZOS)
    out = png_path.replace(".png", f"_ds{max_w}.png")
    save_img(img, out)
    return out

res_items = [
    ("ocr_en_12.png", "char_acc", "请逐字转录图片中的全部文字，只输出文字本身。",
     "ocr_en_12", "res_en", "3v8K2mQ9xT4zP6nL1rA5cJ7bH9dF2eG8wS3uY"),
    ("ocr_cn_0.png", "char_acc", "请逐字转录图片中的全部中文文字，只输出文字本身。",
     "ocr_cn_0", "res_cn", "北方的冬天格外寒冷，清晨的湖面结满厚冰"),
    ("crowd_chart_0.png", "exact", "图中 x 轴标签是 g 开头的编号。数值最高的类别标签是什么？只输出该标签。",
     "crowd_chart_0", "res_chart", chart0_top),
]
for base_png, stype, q, base_id, task, gt in res_items:
    full = base_png
    small = downscale(base_png, 700)
    add(f"{base_id}_full", task, q, gt, [full], stype)
    add(f"{base_id}_ds700", task, q, gt, [small], stype)

# ---------- 输出 ----------
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "chart_data.json"), "w", encoding="utf-8") as f:
    json.dump(chart_data, f, ensure_ascii=False, indent=1)

manifest = {"probes": probes}
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "probes.json"), "w", encoding="utf-8") as f:
    json.dump(manifest, f, ensure_ascii=False, indent=1)

print(f"generated {len(probes)} probes, {len(os.listdir(ASSET_DIR))} images")
from collections import Counter
print(Counter(p["task"] for p in probes))
