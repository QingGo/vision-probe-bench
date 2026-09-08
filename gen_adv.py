# -*- coding: utf-8 -*-
"""第二轮探针：瞄准 DeepSeek V4 Vision 论文声称的强项
迷宫 / 路径追踪 / 多跳空间推理 / 密集计数 / ARC 风格网格推理"""
import base64
import io
import json
import math
import os
import random
from collections import deque

from PIL import Image, ImageDraw, ImageFont

ASSET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
FONT_EN = "/System/Library/Fonts/Supplemental/Arial.ttf"
FONT_CN = "/System/Library/Fonts/Hiragino Sans GB.ttc"

rng = random.Random(2026)
probes = []


def font(path, size):
    return ImageFont.truetype(path, size)


def save_img(img, name):
    p = os.path.join(ASSET_DIR, name)
    img.save(p)
    return name


def add(id_, task, question, gt, images, score_type="exact", extra=None):
    probes.append({"id": id_, "task": task, "question": question, "gt": gt,
                   "images": images, "score_type": score_type, "extra": extra or {}})


# ---------- 迷宫 ----------
def gen_maze(rows, cols):
    walls = {(r, c): [True, True, True, True] for r in range(rows) for c in range(cols)}  # top right bottom left
    visited = [[False] * cols for _ in range(rows)]
    dirs = [(-1, 0, 0, 2), (0, 1, 1, 3), (1, 0, 2, 0), (0, -1, 3, 1)]  # (dr,dc,wall_idx_of_current, wall_idx_of_neighbor)
    stack = [(0, 0)]
    visited[0][0] = True
    while stack:
        r, c = stack[-1]
        nbr = []
        for dr, dc, wi, wj in dirs:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc]:
                nbr.append((nr, nc, wi, wj))
        if not nbr:
            stack.pop()
            continue
        nr, nc, wi, wj = rng.choice(nbr)
        walls[(r, c)][wi] = False
        walls[(nr, nc)][wj] = False
        visited[nr][nc] = True
        stack.append((nr, nc))
    return walls


def render_maze(rows, cols, walls, P=26, W=5):
    img = Image.new("RGB", (cols * P + 1, rows * P + 1), (0, 0, 0))
    d = ImageDraw.Draw(img)
    for r in range(rows):
        for c in range(cols):
            x, y = c * P, r * P
            d.rectangle([x + W, y + W, x + P - W - 1, y + P - W - 1], fill=(255, 255, 255))
            t, rt, b, lf = walls[(r, c)]
            if t and r > 0:
                d.rectangle([x + W, y + W - W, x + P - W - 1, y + W], fill=(0, 0, 0))
            if lf and c > 0:
                d.rectangle([x + W - W, y + W, x + W, y + P - W - 1], fill=(0, 0, 0))
    return img


def draw_marker(img, r, c, color, P=26):
    d = ImageDraw.Draw(img)
    cx, cy = c * P + P / 2, r * P + P / 2
    d.ellipse([cx - P / 3, cy - P / 3, cx + P / 3, cy + P / 3], fill=color)


def shortest_path(rows, cols, walls, start, end):
    q = deque([start])
    prev = {start: None}
    while q:
        cur = q.popleft()
        if cur == end:
            break
        r, c = cur
        for dr, dc, wi, _ in [(-1, 0, 0, None), (0, 1, 1, None), (1, 0, 2, None), (0, -1, 3, None)]:
            if walls[(r, c)][wi]:
                continue
            nr, nc = r + dr, c + dc
            nxt = (nr, nc)
            if 0 <= nr < rows and 0 <= nc < cols and nxt not in prev:
                prev[nxt] = cur
                q.append(nxt)
    if end not in prev:
        return None
    path = []
    cur = end
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    return list(reversed(path))


maze_items = []
rows, cols = 11, 11
w1 = gen_maze(rows, cols)
sp = shortest_path(rows, cols, w1, (0, 0), (rows - 1, cols - 1))
img = render_maze(rows, cols, w1)
draw_marker(img, 0, 0, (46, 204, 113))
draw_marker(img, rows - 1, cols - 1, (231, 76, 60))
n1 = "maze_yes.png"
save_img(img, n1)
add("maze_yes", "maze", "图中是迷宫，绿色圆点 S 是起点，红色圆点 E 是终点。从 S 能否走到 E？只回答“是”或“否”。",
    "是", [n1], "exact")

# 不可解迷宫：上半区与下半区各自独立成迷宫
half = rows // 2
wtop = gen_maze(half, cols)
wbot = gen_maze(rows - half, cols)
walls2 = {}
for r in range(rows):
    for c in range(cols):
        walls2[(r, c)] = list(wtop[(r, c)]) if r < half else list(wbot[(r - half, c)])
img = render_maze(rows, cols, walls2)
draw_marker(img, 1, 1, (46, 204, 113))
draw_marker(img, rows - 2, cols - 2, (231, 76, 60))
n2 = "maze_no.png"
save_img(img, n2)
add("maze_no", "maze", "图中是迷宫，绿色圆点 S 是起点，红色圆点 E 是终点。从 S 能否走到 E？只回答“是”或“否”。",
    "否", [n2], "exact")

w3 = gen_maze(rows, cols)
sp3 = shortest_path(rows, cols, w3, (0, 0), (rows - 1, cols - 1))
img = render_maze(rows, cols, w3)
draw_marker(img, 0, 0, (46, 204, 113))
draw_marker(img, rows - 1, cols - 1, (231, 76, 60))
n3 = "maze_steps.png"
save_img(img, n3)
add("maze_steps", "maze", "图中是迷宫，绿色 S 是起点，红色 E 是终点。从 S 到 E 的最短路径一共经过多少个格子（把 S 和 E 都算入）？只输出数字。",
    str(len(sp3)), [n3], "numeric")

# ---------- 路径追踪（同色曲线交叉） ----------
def bezier_pt(p0, p1, p2, p3, t):
    mt = 1 - t
    x = mt ** 3 * p0[0] + 3 * mt * mt * t * p1[0] + 3 * mt * t * t * p2[0] + t ** 3 * p3[0]
    y = mt ** 3 * p0[1] + 3 * mt * mt * t * p1[1] + 3 * mt * t * t * p2[1] + t ** 3 * p3[1]
    return (x, y)


def gen_tangle(perm, seed):
    rng2 = random.Random(seed)
    W, H = 700, 260
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    ys = [60, 130, 200]
    left = [(30, y) for y in ys]
    right = [(W - 30, y) for y in ys]
    color = (70, 70, 70)
    for i in range(3):
        ty = ys[perm[i] - 1]
        p0 = left[i]
        p3 = (W - 30, ty)
        p1 = (W * 0.35, rng2.randint(20, H - 20))
        p2 = (W * 0.65, rng2.randint(20, H - 20))
        prev = None
        for t in [k / 120 for k in range(121)]:
            pt = bezier_pt(p0, p1, p2, p3, t)
            x, y = int(pt[0]), int(pt[1])
            if prev:
                d.line([prev, (x, y)], fill=color, width=6)
            prev = (x, y)
    for i, (x, y) in enumerate(left):
        d.ellipse([x - 11, y - 11, x + 11, y + 11], fill=color)
        d.text((x - 14, y - 32), chr(65 + i), font=font(FONT_EN, 22), fill=(0, 0, 0))
    for i, (x, y) in enumerate(right):
        d.ellipse([x - 11, y - 11, x + 11, y + 11], fill=color)
        d.text((x - 6, y - 32), str(i + 1), font=font(FONT_EN, 22), fill=(0, 0, 0))
    return img, perm


for k, perm in enumerate([[2, 3, 1], [3, 1, 2], [1, 3, 2]]):
    img, perm = gen_tangle(perm, 100 + k)
    n = f"tangle_{k}.png"
    save_img(img, n)
    gt = ", ".join(f"{c}->{p}" for c, p in zip("ABC", perm))
    add(f"tangle_{k}", "path_trace",
        "图中的三条曲线颜色完全相同、会互相交叉。左侧端点标记为 A、B、C，右侧端点标记为 1、2、3。请沿曲线追踪，确定 A、B、C 各自连到右侧的哪个端点。回答格式：A->数字, B->数字, C->数字。",
        gt, [n], "exact")

# ---------- 多跳空间推理（箭头关系图） ----------
def arrow_diag(d, x0, y0, A, B, direction, w=260):
    """在 (x0,y0) 画一个小图：A ->(direction) B"""
    if direction == "right":  # A 在左，B 在右，箭头向右
        ax, ay = x0 + 55, y0 + 55
        bx, by = x0 + 205, y0 + 55
    elif direction == "down":  # A 在上，B 在下，箭头向下
        ax, ay = x0 + 55, y0 + 55
        bx, by = x0 + 55, y0 + 205
    elif direction == "left":
        bx, by = x0 + 55, y0 + 55
        ax, ay = x0 + 205, y0 + 55
    elif direction == "up":
        bx, by = x0 + 55, y0 + 55
        ax, ay = x0 + 55, y0 + 205
    cx, cy = bx, by
    d.ellipse([ax - 22, ay - 22, ax + 22, ay + 22], outline=(30, 30, 30), width=3, fill="white")
    d.ellipse([bx - 22, by - 22, bx + 22, by + 22], outline=(30, 30, 30), width=3, fill="white")
    d.text((ax - 8, ay - 15), A, font=font(FONT_EN, 28), fill=(0, 0, 0))
    d.text((bx - 8, by - 15), B, font=font(FONT_EN, 28), fill=(0, 0, 0))
    d.line([(ax, ay), (cx, cy)], fill=(0, 0, 0), width=4)
    # 箭头
    import math as _m
    ang = _m.atan2(cy - ay, cx - ax)
    for s in (0.75, 1.05):
        px, py = cx - s * 14 * _m.cos(ang), cy - s * 14 * _m.sin(ang)
        d.line([(cx, cy), (px - 8 * _m.cos(ang + 0.5), py - 8 * _m.sin(ang + 0.5))], fill=(0, 0, 0), width=3)
        d.line([(cx, cy), (px - 8 * _m.cos(ang - 0.5), py - 8 * _m.sin(ang - 0.5))], fill=(0, 0, 0), width=3)


hops_cases = [
    ("right", "down", "右下"),
    ("left", "down", "左下"),
    ("right", "up", "右上"),
    ("down", "right", "右下"),
    ("up", "left", "左上"),
]
for k, (d1, d2, gt) in enumerate(hops_cases):
    img = Image.new("RGB", (560, 340), "white")
    dd = ImageDraw.Draw(img)
    if d1 in ("right", "left"):
        arrow_diag(dd, 20, 40, "A", "B", d1)
    else:
        arrow_diag(dd, 20, 40, "A", "B", d1)
    if d2 in ("right", "left"):
        arrow_diag(dd, 280, 80, "B", "C", d2)
    else:
        arrow_diag(dd, 280, 80, "B", "C", d2)
    n = f"spatial_hop_{k}.png"
    save_img(img, n)
    add(f"spatial_hop_{k}", "spatial_hop",
        "图中有两个小图。左图用箭头表示 A 和 B 的位置关系（箭头从 A 指向 B 所在方向）；右图用箭头表示 B 和 C 的位置关系。请综合这两个关系，推断 C 相对于 A 在哪个方向？只输出一个方位词（上/下/左/右/左上/左下/右上/右下）。",
        gt, [n], "exact")

# ---------- 密集计数（重叠小圆） ----------
def dense_dots(counts, seed, size=700, r=9):
    rng2 = random.Random(seed)
    img = Image.new("RGB", (size, size), "white")
    d = ImageDraw.Draw(img)
    colors = [(231, 76, 60), (52, 152, 219), (46, 204, 113)]
    pts = []
    for _ in range(sum(counts)):
        pts.append((rng2.randint(r, size - r), rng2.randint(r, size - r)))
    idx = 0
    for color, cnt in zip(colors, counts):
        for _ in range(cnt):
            x, y = pts[idx]
            idx += 1
            d.ellipse([x - r, y - r, x + r, y + r], fill=color)
    return img


dense = [
    (("dense_red_only", [47, 0, 0]), "图中有多少个红色圆点？只输出数字。"),
    (("dense_red_sub", [23, 17, 12]), "图中有三种颜色的圆点。其中红色圆点有多少个？只输出数字。"),
    (("dense_red_sub2", [35, 10, 9]), "图中有三种颜色的圆点。其中红色圆点有多少个？只输出数字。"),
]
for (mid, counts), q in dense:
    img = dense_dots(counts, hash(mid) % 10000)
    n = f"{mid}.png"
    save_img(img, n)
    add(mid, "dense_count", q, str(counts[0]), [n], "numeric")

# ---------- ARC 风格网格推理 ----------
CELL = 26
GAP = 18
PAL = {"K": (0, 0, 0), "R": (220, 40, 40), "G": (30, 170, 60), "B": (40, 80, 220),
       "Y": (230, 200, 30), "P": (180, 40, 200), "O": (240, 130, 20), "W": (250, 250, 250)}
COLORS = list(PAL.keys())


def rand_grid(rows, cols, fill):
    return [[rng.choice(fill) for _ in range(cols)] for _ in range(rows)]


def trans_recolor(g, a, b):
    return [[a if x == b else x for x in row] for row in g]


def trans_mirror(g):
    return [row[::-1] for row in g]


def trans_rotcw(g):
    return [list(row) for row in zip(*g[::-1])]


def trans_outline(g):
    rows, cols = len(g), len(g[0])
    rs, cs = set(), set()
    for r in range(rows):
        for c in range(cols):
            if g[r][c] != "K":
                rs.add(r)
                cs.add(c)
    rmin, rmax, cmin, cmax = min(rs), max(rs), min(cs), max(cs)
    out = [["K"] * cols for _ in range(rows)]
    for r in range(rmin, rmax + 1):
        for c in range(cmin, cmax + 1):
            if r in (rmin, rmax) or c in (cmin, cmax):
                out[r][c] = g[r][c]
    return out


def trans_fillhole(g):
    rows, cols = len(g), len(g[0])
    out = [list(row) for row in g]
    fg = None
    for r in range(rows):
        for c in range(cols):
            if g[r][c] != "K":
                fg = g[r][c]
                break
        if fg:
            break
    # flood fill from border black cells
    seen = set()
    q = deque()
    for r in range(rows):
        for c in range(cols):
            if (r in (0, rows - 1) or c in (0, cols - 1)) and g[r][c] == "K":
                q.append((r, c))
                seen.add((r, c))
    while q:
        r, c = q.popleft()
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in seen and g[nr][nc] == "K":
                seen.add((nr, nc))
                q.append((nr, nc))
    for r in range(rows):
        for c in range(cols):
            if g[r][c] == "K" and (r, c) not in seen:
                out[r][c] = fg
    return out


def draw_grid(d, g, x, y):
    for r, row in enumerate(g):
        for c, ch in enumerate(row):
            d.rectangle([x + c * CELL, y + r * CELL, x + (c + 1) * CELL - 1, y + (r + 1) * CELL - 1],
                        fill=PAL[ch], outline=(0, 0, 0))


def arc_row(d, g_in, g_out, y, label):
    d.text((8, y + CELL * len(g_in) / 2 - 10), label, font=font(FONT_CN, 16), fill=(0, 0, 0))
    draw_grid(d, g_in, 50, y)
    d.text((50 + len(g_in[0]) * CELL + 8, y + CELL * len(g_in) / 2 - 10), "→", font=font(FONT_EN, 24), fill=(0, 0, 0))
    draw_grid(d, g_out, 50 + len(g_in[0]) * CELL + 30, y)


def make_arc(transform, tag, rows=6, cols=6, seed=None):
    rng3 = random.Random(seed)
    W = 50 + 2 * (cols * CELL) + 40 + 30 + 20
    H = 3 * (rows * CELL) + 60
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    demos = []
    for _ in range(2):
        if transform is trans_recolor:
            a, b = rng3.sample([c for c in COLORS if c not in ("K", "W")], 2)
            g = rand_grid(rows, cols, ["K", "K", a, a])
            out = trans_recolor(g, a, b)
        elif transform is trans_fillhole:
            fg = rng3.choice([c for c in COLORS if c not in ("K", "W")])
            g = [["K"] * cols for _ in range(rows)]
            rr, cc = 2, 3
            for r in range(rr, rr + 3):
                for c in range(cc, cc + 3):
                    g[r][c] = fg
            for r in range(rr + 1, rr + 2):
                for c in range(cc + 1, cc + 2):
                    g[r][c] = "K"
            out = trans_fillhole(g)
        elif transform is trans_outline:
            fg = rng3.choice([c for c in COLORS if c not in ("K", "W")])
            g = rand_grid(rows, cols, ["K", "K", fg])
            out = trans_outline(g)
        else:
            fg = rng3.choice([c for c in COLORS if c not in ("K", "W")])
            g = rand_grid(rows, cols, ["K", fg])
            out = transform(g)
        demos.append((g, out))
    # test input
    if transform is trans_recolor:
        a, b = demos[0][0] and rng3.sample([c for c in COLORS if c not in ("K", "W")], 2)
        g_test = rand_grid(rows, cols, ["K", "K", a, a])
        out_test = trans_recolor(g_test, a, b)
    elif transform is trans_fillhole:
        fg = rng3.choice([c for c in COLORS if c not in ("K", "W")])
        g_test = [["K"] * cols for _ in range(rows)]
        rr, cc = 1, 2
        for r in range(rr, rr + 3):
            for c in range(cc, cc + 3):
                g_test[r][c] = fg
        g_test[rr + 1][cc + 1] = "K"
        out_test = trans_fillhole(g_test)
    elif transform is trans_outline:
        fg = rng3.choice([c for c in COLORS if c not in ("K", "W")])
        g_test = rand_grid(rows, cols, ["K", "K", fg])
        out_test = trans_outline(g_test)
    else:
        fg = rng3.choice([c for c in COLORS if c not in ("K", "W")])
        g_test = rand_grid(rows, cols, ["K", fg])
        out_test = transform(g_test)
    arc_row(d, demos[0][0], demos[0][1], 8, "示例1")
    arc_row(d, demos[1][0], demos[1][1], rows * CELL + 14, "示例2")
    d.text((8, 2 * rows * CELL + 18 + CELL * rows / 2 - 10), "测试", font=font(FONT_CN, 16), fill=(0, 0, 0))
    draw_grid(d, g_test, 50, 2 * rows * CELL + 18)
    gt = "".join("".join(row) for row in out_test)
    return img, gt


arc_trans = [("arc_recolor", trans_recolor, 101), ("arc_mirror", trans_mirror, 202),
             ("arc_rot", trans_rotcw, 303), ("arc_outline", trans_outline, 404),
             ("arc_fill", trans_fillhole, 505), ("arc_mirror2", trans_mirror, 606)]
for tag, trans, seed in arc_trans:
    img, gt = make_arc(trans, tag, rows=6, cols=6, seed=seed)
    n = f"{tag}.png"
    save_img(img, n)
    add(tag, "arc_grid",
        "图中包含同一个变换规则的两个示例（输入→输出）和一个测试输入网格。颜色字母：K=黑 R=红 G=绿 B=蓝 Y=黄 P=紫 O=橙。请推断规则，并给出测试输入对应的输出网格，按行输出字母，行之间用换行分隔，不要输出其它内容。",
        gt, [n], "grid")

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "probes_adv.json"), "w", encoding="utf-8") as f:
    json.dump({"probes": probes}, f, ensure_ascii=False, indent=1)

from collections import Counter
print("generated", len(probes), "advanced probes")
print(Counter(p["task"] for p in probes))
