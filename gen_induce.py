# -*- coding: utf-8 -*-
"""视觉原语诱导实验：从已有探针集挑选 16 题，生成 A(基线)/B(详细提示无原语)/C(诱导原语+声明尺寸) 三臂 prompt"""
import json
import os

from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSET_DIR = os.path.join(BASE_DIR, "assets")

# 受益候选（高概率）与回归对照
PICK = [
    # 受益候选
    ("dense_count", "dense_red_only", "dense_red_only"),
    ("dense_count", "dense_red_sub", "dense_red_sub"),
    ("dense_count", "dense_red_sub2", "dense_red_sub2"),
    ("maze", "maze_yes", "maze_yes"),
    ("maze", "maze_no", "maze_no"),
    ("maze", "maze_steps", "maze_steps"),
    ("path_trace", "tangle_0", "tangle_0"),
    ("path_trace", "tangle_1", "tangle_1"),
    ("path_trace", "tangle_2", "tangle_2"),
    ("spatial_hop", "spatial_hop_0", "spatial_hop_0"),
    ("spatial_hop", "spatial_hop_4", "spatial_hop_4"),
    # 回归对照（预期无提升）
    ("arc_grid", "arc_recolor", "arc_recolor"),
    ("arc_grid", "arc_mirror", "arc_mirror"),
    ("ocr_cn", "ocr_cn_0", "ocr_cn_0"),
    ("ocr_en", "ocr_en_9", "ocr_en_9"),
    ("crowd_chart", "crowd_chart_0_top", "crowd_chart_0_top"),
]

INDUCE_TEXT = {
    "dense_count": "推理过程中，请先用 <｜ref｜>红色圆点<｜/ref｜><｜box｜>[[x1,y1,x2,y2]]<｜box｜> 逐个标注每一个红色圆点，确保不遗漏，然后再给出总数。",
    "maze": "推理过程中，请像深度优先搜索一样探索迷宫，并用 <｜point｜>[[x,y]]<｜point｜> 逐一记录你当前所在格子的坐标，回溯到死路时也要记录。",
    "path_trace": "推理过程中，请沿着曲线用 <｜point｜>[[x,y]]<｜point｜> 逐步记录路径上的关键点，注意在交叉处按连续性判断延续。",
    "spatial_hop": "推理过程中，请先用 <｜box｜>[[x1,y1,x2,y2]]<｜box｜> 标注 A、B、C 图形的位置以辅助推理。",
    "arc_grid": "推理过程中，请先用 <｜box｜>[[x1,y1,x2,y2]]<｜box｜> 标注示例网格与测试网格的位置，再推导变换规则。",
    "ocr_cn": "推理过程中，请先用 <｜box｜>[[x1,y1,x2,y2]]<｜box｜> 标注文字所在区域。",
    "ocr_en": "推理过程中，请先用 <｜box｜>[[x1,y1,x2,y2]]<｜box｜> 标注文字所在区域。",
    "crowd_chart": "推理过程中，请先用 <｜box｜>[[x1,y1,x2,y2]]<｜box｜> 标注你认为数值最高的那根柱子。",
}

B_TEXT = "请先仔细观察图片，分步骤分析后，再给出最终答案。"

full = json.load(open(os.path.join(BASE_DIR, "probes.json"), encoding="utf-8"))["probes"]
adv = json.load(open(os.path.join(BASE_DIR, "probes_adv.json"), encoding="utf-8"))["probes"]
allp = {p["id"]: p for p in full + adv}

out = []
for task, pid, _ in PICK:
    p = allp[pid]
    img = os.path.join(ASSET_DIR, p["images"][0])
    W, H = Image.open(img).size
    for arm in ("A", "B", "C"):
        q = p["question"]
        if arm == "B":
            q = q + "\n\n" + B_TEXT
        elif arm == "C":
            q = q + "\n\n" + INDUCE_TEXT[task] + f"\n注意：图片尺寸是 {W}×{H} 像素。最后一行请单独输出：ANSWER: <最终答案>"
        out.append({"id": f"{pid}_{arm}", "base_id": pid, "task": task, "arm": arm,
                    "question": q, "images": p["images"], "gt": p["gt"],
                    "score_type": p["score_type"]})

json.dump({"probes": out}, open(os.path.join(BASE_DIR, "probes_induce.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
from collections import Counter
print("generated", len(out), "induce probes")
print(Counter((p["task"], p["arm"]) for p in out))
