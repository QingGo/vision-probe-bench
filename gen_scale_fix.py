# -*- coding: utf-8 -*-
"""修复自动缩放踩档：为关键测试图生成不缩放波段版本，重测同题对比旧结果"""
import base64
import json
import os
import re

from PIL import Image, ImageDraw, ImageFont
from openai import OpenAI

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSET_DIR = os.path.join(BASE_DIR, "assets")
FONT_CN = "/System/Library/Fonts/Hiragino Sans GB.ttc"
FONT_EN = "/System/Library/Fonts/Supplemental/Arial.ttf"
LO, HI = 147456, 640000


def font(p, s):
    return ImageFont.truetype(p, s)


def in_band(img):
    return LO <= img.width * img.height <= HI


def pad_to_band(img, name):
    """保持内容不动，通过加白边把总像素撑进 [LO, HI]。最差缩小时用上采样。"""
    W, H = img.size
    for _ in range(3):
        if in_band(img):
            return img
        if img.width * img.height < LO:
            # 先放大到 LO 以上（放大内容，但明确文档化这一前提）
            scale = (LO * 1.15 / (img.width * img.height)) ** 0.5
            img = img.resize((int(img.width * scale), int(img.height * scale)), Image.LANCZOS)
        else:
            # 太大：缩小到 HI 以内
            scale = (HI * 0.95 / (img.width * img.height)) ** 0.5
            img = img.resize((int(img.width * scale), int(img.height * scale)), Image.LANCZOS)
    return img


# ---- 生成修正版图片：OCR（加高白边而非拉伸）、迷宫、符号、计数、图表 ----
def ocr_fixed(src_name, txt, size, fpath, out_name):
    """OCR 细长条改为较高画布：1500x120，总像素 18 万，不缩放"""
    img = Image.new("RGB", (1500, 120), "white")
    d = ImageDraw.Draw(img)
    d.text((10, 48), txt, font=font(fpath, size), fill=(0, 0, 0))
    img.save(os.path.join(ASSET_DIR, out_name))
    return out_name, txt


fixed_tasks = []
probes = json.load(open(os.path.join(BASE_DIR, "probes.json"), encoding="utf-8"))["probes"]
adv = json.load(open(os.path.join(BASE_DIR, "probes_adv.json"), encoding="utf-8"))["probes"]
allp = {p["id"]: p for p in probes + adv}

# 1) OCR 中文/英文：用原文本重绘为 1500x120（18万像素，不缩放）
ocr_map = {
    "ocr_cn_0": ("北方的冬天格外寒冷，清晨的湖面结满厚冰", FONT_CN),
    "ocr_cn_1": ("智能手表可以实时监测心率血压与睡眠质量", FONT_CN),
    "ocr_en_9": ("3v8K2mQ9xT4zP6nL1rA5cJ7bH9dF2eG8wS3uY", FONT_EN),
}
fix = []
for pid, (txt, fpath) in ocr_map.items():
    oname = f"fix_{pid}.png"
    ocr_fixed(pid, txt, 13, fpath, oname)
    p = dict(allp[pid])
    p["id"] = pid + "_fix"
    p["images"] = [oname]
    fix.append(p)
    print("fixed:", oname)

# 2) 迷宫：原图是 287 被放大，改用 384x384（补白边不缩放）
for pid in ("maze_yes", "maze_no", "maze_steps"):
    src = Image.open(os.path.join(ASSET_DIR, allp[pid]["images"][0])).convert("RGB")
    # 原迷宫 287x287，贴到 384x384 白色画布中心，像素 147456 恰好下限
    canvas = Image.new("RGB", (384, 384), "white")
    canvas.paste(src, ((384 - 287) // 2, (384 - 287) // 2))
    oname = f"fix_{pid}.png"
    canvas.save(os.path.join(ASSET_DIR, oname))
    p = dict(allp[pid]); p["id"] = pid + "_fix"; p["images"] = [oname]
    fix.append(p)
    print("fixed:", oname)

# 3) 计数 900x900 缩小档 -> 800x800 下限档（640000 恰好 HI，用 795x795=632025 <640000 ✓）
for pid in ("count_13", "count_26", "count_77"):
    src = Image.open(os.path.join(ASSET_DIR, allp[pid]["images"][0])).convert("RGB")
    src = src.resize((795, 795), Image.LANCZOS)
    oname = f"fix_{pid}.png"
    src.save(os.path.join(ASSET_DIR, oname))
    p = dict(allp[pid]); p["id"] = pid + "_fix"; p["images"] = [oname]
    fix.append(p)
    print("fixed:", oname)

# 4) 拥挤图表 1500x504=756K 缩小档 -> 缩至 1240x417=517K 不缩放（但内容本就该缩小）
for pid in ("crowd_chart_0_top", "crowd_chart_1_top", "crowd_chart_2_top"):
    src = Image.open(os.path.join(ASSET_DIR, allp[pid]["images"][0])).convert("RGB")
    scale = (HI * 0.85 / (src.width * src.height)) ** 0.5
    src2 = src.resize((int(src.width * scale), int(src.height * scale)), Image.LANCZOS)
    oname = f"fix_{pid}.png"
    src2.save(os.path.join(ASSET_DIR, oname))
    p = dict(allp[pid]); p["id"] = pid + "_fix"; p["images"] = [oname]
    fix.append(p)
    print("fixed:", oname, src2.size)

# 5) 符号 300x300=90K 放大档 -> 384x384 白边（147456 = LO 恰好）— 符号可能显得小？改成 385x385
for pid in ("symbol_0", "symbol_4"):
    src = Image.open(os.path.join(ASSET_DIR, allp[pid]["images"][0])).convert("RGB")
    canvas = Image.new("RGB", (385, 385), "white")
    canvas.paste(src.resize((300, 300)), (42, 42))
    oname = f"fix_{pid}.png"
    canvas.save(os.path.join(ASSET_DIR, oname))
    p = dict(allp[pid]); p["id"] = pid + "_fix"; p["images"] = [oname]
    fix.append(p)
    print("fixed:", oname)

json.dump({"probes": fix}, open(os.path.join(BASE_DIR, "probes_scale_fix.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("generated", len(fix), "fixed probes")
