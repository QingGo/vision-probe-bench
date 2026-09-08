# Vision Bench：多模态 API 对比测评（合成探针）

程序化生成带标准答案（Ground Truth）的视觉探针，自动判分，对比不同多模态 API 的感知、推理与 Agent 能力。总 API 成本可控（约 0.1~1.5 元/轮，取决于思考档位）。

实测对象：`deepseek-v4-flash-vision-exp`（DeepSeek）vs `mimo-v2.5`（小米 MiMo）。

## 目录结构

```
├── gen_images.py         # 61 题基础探针生成器（OCR/表格/图表/计数/空间/多图/数学/符号/分辨率A-B）
├── gen_adv.py            # 20 题进阶探针生成器（迷宫/路径追踪/多跳空间/密集计数/ARC 网格）
├── gen_agent.py          # 8 题 Agent 探针生成器（UI 点击/表单提交/校验循环/图表查询）
├── bench.py              # 纯感知评测器（默认 / max 思考 / nothink 三档）
├── agent_bench.py        # Agent 工具调用环路评测器（think / nothink）
├── probes.json           # 探针清单（题目、GT、判分类型）
├── probes_adv.json       # 进阶探针清单
├── agent_probes.json     # Agent 探针清单
├── chart_data.json       # 图表真值数据（供 Agent 的 get_value 工具使用）
└── assets/               # 生成的测试图片
```

## 快速开始

```bash
pip install -r requirements.txt

# 1. 生成探针图片与清单（可选，仓库已附带生成结果）
python3 gen_images.py
python3 gen_adv.py
python3 gen_agent.py

# 2. 设置 API Key（环境变量，切勿写入文件）
export DS_API_KEY=sk-xxx
export MIMO_API_KEY=sk-xxx

# 3. 跑纯感知（默认配置）
python3 bench.py
# 3b. max 思考档（DeepSeek reasoning_effort=max，max_tokens=65536）
python3 bench.py probes_adv.json
# 3c. 非思考档 + 指定探针 + 50 线程并发 + 断点续跑
python3 bench.py --nothink --workers=50 probes_pure.json --resume

# 4. 跑 Agent 环路（nothink / think）
python3 agent_bench.py --workers=50
python3 agent_bench.py --think --workers=50

# 5. 生成报告
python3 bench.py --report [manifest.json]
```

## 参数说明

| 参数 | 说明 |
|---|---|
| `--model=<deepseek\|mimo>` | 只跑单个模型，输出到 `results_<model>.jsonl`，可与另一模型并行 |
| `--workers=N` | 线程池并发跑探针（实测 50 线程安全） |
| `--resume` | 断点续跑：跳过已正确完成的题、重跑失败的题（每题结果即时写盘） |
| `--nothink` | 关闭 thinking（`thinking.type=disabled`，max_tokens=2048） |
| 默认（无 `--nothink`） | 开启 thinking，DeepSeek 用 `reasoning_effort=max`，max_tokens=65536 |
| `--report` | 只读合并结果文件出报告，不调用 API |

## 判分方式

- `char_acc`：字符级 Levenshtein 准确率（忽略空白），用于 OCR 转录
- `exact`：规范化后精确匹配（去空白、转小写、全角转半角）
- `numeric`：数值容差 ±0.01
- `grid`：网格字母序列比对（ARC 任务）
- `form_args`：Agent 表单任务的工具参数比对（含文本兜底提取）

## 结果摘要（2026-08 实测）

| 轮次 | 配置 | DeepSeek | MiMo |
|---|---|---|---|
| 基础 61 题 | 默认 | 69% | **85%** |
| 进阶 20 题 | max 思考 | 45% | 45%（成本 DS 1.45 元 vs MiMo 0.10 元） |
| Agent 8 题 | think/nothink 一致 | 75% | 75% |

要点：MiMo 赢纯感知（中文 OCR 100% vs 0%、吃满高分辨率）；DeepSeek 仅在 max 思考下赢迷宫/路径追踪，成本贵约 14 倍；ARC 两家均 0 分；Agent 环路对两家放大相同。

## 注意事项

- **API Key 通过环境变量传入，禁止提交到仓库**（参考 `.env.example`）。
- 生成器使用 macOS 系统字体（Arial / Hiragino Sans GB），换平台需修改 `gen_images.py` 中的字体路径。
- 模型为实验版本，结果可能随模型更新而变化。
- 每题默认单次采样，存在运行间方差，结论请以整体趋势为准。

## 相关文章

知乎文章草稿见 `article/zhihu-draft.md`。
