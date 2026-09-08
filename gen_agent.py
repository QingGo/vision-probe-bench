# -*- coding: utf-8 -*-
"""多模态 Agent 探针生成器：UI 截图、表单图片 + agent_probes.json"""
import json
import os

from PIL import Image, ImageDraw, ImageFont

ASSET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
FONT_CN = "/System/Library/Fonts/Hiragino Sans GB.ttc"
FONT_EN = "/System/Library/Fonts/Supplemental/Arial.ttf"

probes = []


def font(path, size):
    return ImageFont.truetype(path, size)


def save_img(img, name):
    p = os.path.join(ASSET_DIR, name)
    img.save(p)
    return name


def add(id_, task, question, gt, images, toolset, score_type="exact"):
    probes.append({"id": id_, "task": task, "question": question, "gt": gt,
                   "images": images, "toolset": toolset, "score_type": score_type})


# ---------- UI 面板（点击按钮） ----------
def ui_panel(buttons, title, W=620, H=300, btn_h=52, gap=16):
    img = Image.new("RGB", (W, H), (235, 238, 242))
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, W, 44], fill=(45, 62, 80))
    d.text((18, 10), title, font=font(FONT_CN, 24), fill=(255, 255, 255))
    n = len(buttons)
    total_w = n * 140 + (n - 1) * gap
    x0 = (W - total_w) // 2
    y = 140
    for i, (label, col) in enumerate(buttons):
        x = x0 + i * (140 + gap)
        d.rounded_rectangle([x, y, x + 140, y + btn_h], radius=8, fill=col)
        d.text((x + 55, y + 12), label, font=font(FONT_CN, 24), fill=(255, 255, 255))
    return img


ui_cases = [
    ("ui_click_0", "数据导出面板", [("保存", (52, 152, 219)), ("取消", (149, 165, 166)),
                                   ("删除", (231, 76, 60)), ("导出", (46, 204, 113)), ("打印", (155, 89, 182))],
     "导出", "用户想导出报表数据。请读取面板上的按钮文字，调用 click 点击正确的按钮，最后输出你点击的按钮文字。"),
    ("ui_click_1", "协议确认弹窗", [("同意", (46, 204, 113)), ("拒绝", (231, 76, 60)),
                                   ("更多", (52, 152, 219)), ("稍后", (149, 165, 166))],
     "同意", "用户同意服务协议。请读取面板上的按钮文字，调用 click 点击正确的按钮，最后输出你点击的按钮文字。"),
]
for mid, title, buttons, gt, q in ui_cases:
    img = ui_panel(buttons, title)
    n = f"{mid}.png"
    save_img(img, n)
    add(mid, "ui_click", q, gt, [n], "ui_click")

# ---------- 表单（读取字段 → submit 工具） ----------
def form_img(fields, W=760, H=430):
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    d.text((40, 24), "用户信息登记表", font=font(FONT_CN, 30), fill=(30, 30, 30))
    y = 96
    for label, value in fields:
        d.rectangle([40, y, W - 40, y + 74], outline=(180, 180, 180), width=2, fill=(250, 250, 250))
        d.text((60, y + 22), label, font=font(FONT_CN, 22), fill=(90, 90, 90))
        d.text((230, y + 22), value, font=font(FONT_CN, 22), fill=(0, 0, 0))
        y += 92
    return img


form_cases = [
    ("form_0", [("姓名", "李小明"), ("电话", "13812345678"), ("邮箱", "xiaoming@mail.com")],
     {"name": "李小明", "phone": "13812345678", "email": "xiaoming@mail.com"}),
    ("form_1", [("姓名", "王芳"), ("电话", "15900001111"), ("邮箱", "wangfang@test.cn")],
     {"name": "王芳", "phone": "15900001111", "email": "wangfang@test.cn"}),
]
for mid, fields, gt in form_cases:
    img = form_img(fields)
    n = f"{mid}.png"
    save_img(img, n)
    add(mid, "form_submit",
        "图中是一张信息登记表。请读取表单中姓名、电话、邮箱三个字段的值，调用 submit 工具提交，字段顺序为 (name, phone, email)。最后输出你提交的值。",
        f"{gt['name']}|{gt['phone']}|{gt['email']}", [n], toolset="form_submit", score_type="form_args")

# ---------- 复用拥挤柱状图：chart_lookup（放大测试）与 verify（纠错循环） ----------
chart_data = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "chart_data.json"), encoding="utf-8"))
for t in range(3):
    img = f"crowd_chart_{t}.png"
    data = chart_data[img]
    top = max(data, key=data.get)
    add(f"chart_lookup_{t}", "chart_lookup",
        "图中是柱状图，x 轴标签为 g 开头的编号（如 g000）。x 轴标签较小，请先用 get_value 工具查询若干候选类别的精确数值，再回答：数值最高的类别标签是什么？最后只输出该标签。",
        top, [img], "chart_lookup")

mid_img = "ui_click_0.png"
add("verify_0", "verify",
    "图中是一个操作面板。请数一数面板上有几个按钮。先用 verify 工具提交你的答案，若返回 wrong 说明数错了，请重新看图再试，直到返回 correct。最后输出数量。",
    "5", [mid_img], toolset="verify", score_type="numeric")

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "agent_probes.json"), "w", encoding="utf-8") as f:
    json.dump({"probes": probes}, f, ensure_ascii=False, indent=1)

from collections import Counter
print("generated", len(probes), "agent probes")
print(Counter(p["task"] for p in probes))
