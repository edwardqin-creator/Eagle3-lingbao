# 评测矩阵、请求级诊断与实验台账

本文把三类日常操作固化下来：冻结 Draft 后搜索候选树、生成请求级接受长度、按任务族和 Train/Validation 对比。线上仓库默认位于：

```text
/data/home/leonardoqin/Eagle3-lingbao
```

## 1. 三种口径不能混用

| 口径 | 配置 | 用途 | 能否做最终吞吐验收 |
|---|---|---|---|
| Formal | `EVAL_REQUEST_DIAGNOSTICS=0`、默认 decode log interval 40 | Baseline/EAGLE3 真实性能 | 是 |
| Diagnostic | 并发1、`EVAL_REQUEST_DIAGNOSTICS=1`、interval 1 | 每请求 ACC、任务族、失败样本 | 否 |
| Registry fallback | `experiments/registry.json` 的历史观测 | 换机后查历史结论 | 否；只做台账 |

诊断模式里，客户端在请求完成计时之后额外等待 20ms，让 SGLang 日志落盘并归属到当前请求。因此单请求 `ttft_s/tpot_s/e2e_s` 不直接包含这 20ms，但整个 benchmark 的 `duration_s` 包含等待；并发1跑1500条时约额外30秒。所以：

- 请求级 TPOT/E2E 的相关性分析可以使用；
- 汇总的请求吞吐、输出吞吐不能用于上线 Gate；
- `--decode-log-interval 1` 仍可能带来少量服务端日志 I/O 干扰，最终结论必须回到 Formal 复测。

## 2. 冻结 Draft 后批量测试树和 Split

一次只定义四类输入：基础 `.env`、冻结 Draft、T=0 Validation、T=0 Train。脚本会为每个组合生成独立配置和实验名：

```bash
cd /data/home/leonardoqin/Eagle3-lingbao

export BASE_CONFIG=configs/lingbao.env
export MATRIX_NAME=v2-i8k-e5-t00
export DRAFT_MODEL=/data/home/leonardoqin/models/exports/lingbao-eagle3-v2-all-i8k-oldopt-e5-v1-sglang
export VAL_T0_DATA=/data/home/leonardoqin/datasets/lingbao_eagle3_data/data_version2/val_requests.temperature_0.jsonl
export TRAIN_T0_DATA=/data/home/leonardoqin/datasets/lingbao_eagle3_data/data_version2/train_requests.temperature_0.jsonl
export SPLITS="val train"
export TREES="s2k2d4:2:2:4 s3k1d4:3:1:4 s2k2d3:2:2:3"
export EVAL_GPU_DEVICES="6,7"
export EVAL_PORT_BASE=34600
export EVAL_LIMIT=500
export EVAL_SAMPLE_SEED=20260805

bash scripts/06_evaluate_matrix.sh plan
bash scripts/06_evaluate_matrix.sh run
```

`TREES` 每项格式为 `标签:steps:topk:draft_tokens`。先 `plan` 只生成配置，人工确认 Draft、数据、GPU、端口和树，再 `run`。已有验收报告时脚本沿用 `05_evaluate.sh` 的防覆盖保护；确认重跑才设置：

```bash
EVAL_OVERWRITE=1 bash scripts/06_evaluate_matrix.sh run
```

### T=0 的要求

不要只在实验名写 `t00`。请求文件中每条 `sampling_params.temperature` 必须确实为0，并冻结同一个文件和 SHA256。训练轨迹仍可来自 T=0.9；“训练温度”和“服务评测温度”是两个变量，台账里必须分别说明。

## 3. 生成请求级接受长度

Formal 轮不能从汇总 decode 日志可靠映射到单条请求。要逐请求归因，重跑 Diagnostic：

```bash
export EVAL_MODE=diagnostic
export EVAL_LIMIT=1500
export EVAL_CONCURRENCIES=1
export SPLITS="val train"
export TREES="s2k2d4:2:2:4"

bash scripts/06_evaluate_matrix.sh plan
bash scripts/06_evaluate_matrix.sh run
```

它会在 `eagle3_c1.details.jsonl` 每条记录写入：

```json
{
  "request_acceptance": {
    "decode_log_samples": 12,
    "avg_accept_length": 2.58,
    "p50_accept_length": 3.0,
    "p90_accept_length": 3.0,
    "p95_accept_length": 3.0,
    "p99_accept_length": 3.0,
    "avg_accept_rate": 0.53
  }
}
```

这里一个请求会经历多次 speculative verification，所以会同时存在均值、P50、P90、P99。它们描述的是“该请求内部多次验证循环”的分布，不是500/1500条请求之间的分位数。树 `2/2/4` 下单次可推进的有效长度上限通常是3，因此很多请求的P90/P99会等于3；请求间分位数由后处理脚本重新计算。

检查诊断覆盖率：

```bash
EAGLE_DETAILS=/path/to/eagle3_c1.details.jsonl \
bash scripts/07_postprocess.sh coverage
```

覆盖率必须为100%，否则不能做任务族 ACC 结论。

## 4. 合并 Manifest、请求和两套 Details

Validation 示例：

```bash
DATA_ROOT=/data/home/leonardoqin/datasets/lingbao_eagle3_data

MANIFEST=${DATA_ROOT}/data_version2/split_manifest.jsonl \
REQUESTS=${DATA_ROOT}/data_version2/val_requests.temperature_0.jsonl \
RESULT_DIR=${DATA_ROOT}/benchmark_results/lingbao-v2-e5-t00-s2k2d4-requestdiag-v1/validation \
SPLIT=val \
ANALYSIS_NAME=v2-i8k-e5-val-t00-s2k2d4-diag1500 \
DRAFT_TOKENS=4 \
bash scripts/07_postprocess.sh analyze
```

Train 只替换请求、结果目录、Split和分析名：

```bash
MANIFEST=${DATA_ROOT}/data_version2/split_manifest.jsonl \
REQUESTS=${DATA_ROOT}/data_version2/train_requests.temperature_0.jsonl \
RESULT_DIR=${DATA_ROOT}/benchmark_results/lingbao-v2-e5-train-t00-s2k2d4-requestdiag-gpu67-v1/validation \
SPLIT=train \
ANALYSIS_NAME=v2-i8k-e5-train-t00-s2k2d4-diag1500 \
DRAFT_TOKENS=4 \
bash scripts/07_postprocess.sh analyze
```

输出目录默认在 `artifacts/postprocess/<ANALYSIS_NAME>/`：

| 文件 | 内容 |
|---|---|
| `request_analysis.csv/jsonl` | 每请求 Prompt末句、任务族、ACC、TTFT/TPOT/E2E、加速比、输出哈希 |
| `task_family_summary.csv/json` | `all_pairs`、同输出长度、同输出哈希三种任务族汇总 |
| `request_group_summary.csv/json` | 按任务族、ACC桶、输出长度桶聚合 |

请求级 `TPOT speedup = baseline_tpot_ms / eagle_tpot_ms`；大于1表示 EAGLE3 更快。`E2E reduction = (baseline-eagle)/baseline`；正数表示延迟下降。不同输出Token数会扭曲单请求比较，优先看 `same_output_sha256`，其次 `same_output_tokens`，最后才是 `all_pairs`。

## 5. Train / Validation 任务族对比

完成两边 `analyze` 后：

```bash
TRAIN_ANALYSIS_DIR=artifacts/postprocess/v2-i8k-e5-train-t00-s2k2d4-diag1500 \
VAL_ANALYSIS_DIR=artifacts/postprocess/v2-i8k-e5-val-t00-s2k2d4-diag1500 \
SUMMARY_SCOPE=all_pairs \
OUTPUT_DIR=artifacts/postprocess/v2-i8k-e5-train-vs-val \
bash scripts/07_postprocess.sh compare-splits
```

如要控制 Target-only 与 EAGLE3 输出完全一致：

```bash
SUMMARY_SCOPE=same_output_sha256 \
bash scripts/07_postprocess.sh compare-splits
```

经验判读：

- Train ACC显著高于Val，同时Val性能差：过拟合/分布偏移；
- Train和Val都低且接近：不是经典过拟合，应看容量、监督目标、Hidden State层或数据难例；
- 当前 I8K E5 的 Train/Val ACC 只差约0.02，未见明显记忆优势；
- 信息处理、决策辅助稳定正收益；闲聊、知识问答是优先难例挖掘方向。

## 6. 实验台账

所有重要结果先登记 `experiments/registry.json`，然后生成机器可读与 Markdown 台账：

```bash
bash scripts/07_postprocess.sh ledger
```

产物：

```text
artifacts/postprocess/ledger/
├── experiment_ledger.csv
├── experiment_ledger.json
└── experiment_ledger.md
```

如果结果目录在本机存在，程序直接读取 `baseline_c1.json/eagle3_c1.json/comparison_c1.json`；迁移到新机器后文件不存在，则使用注册表里经过复核的 `observed` 结果，并标记 `source=registry`。正式发布结论时可加 `LEDGER_STRICT=1`，强制所有路径可访问。

## 7. 每轮实验完成后的固定清单

1. 冻结 `.env`、Draft目录、请求文件与SHA256；
2. Formal 500/1500条跑真实性能；
3. 需要归因时另跑 Diagnostic，不覆盖 Formal 实验名；
4. 生成任务族与请求级产物；
5. 更新 `experiments/registry.json`；
6. 更新 [EXPERIMENT_OVERVIEW.md](EXPERIMENT_OVERVIEW.md) 的结论与待办；
7. 最后才跑冻结 Test，禁止看 Test 后继续调参。
