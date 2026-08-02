# Lingbao EAGLE3：从数据到验收的 5 步工作流

这个仓库把 TENCENT64 上逐步积累的 30 多个实验脚本收敛成 5 个用户入口。目标不是隐藏关键步骤，而是让每一步都有明确输入、输出、检查点和失败恢复方式。

```text
原始请求日志
  └─ 01 数据清洗/去重/切分
       └─ 02 Target 35B 重放（只重放 Train）
            └─ 03 生成 input_ids/loss_mask/vocab mapping
                 └─ 04 Plan → Train → Checkpoint → Export
                      └─ 05 Baseline/EAGLE3 Validation → 冻结参数 → Test
```

## 一、先准备配置

在远端开发容器中：

```bash
cd /data/home/leonardoqin
git clone https://github.com/edwardqin-creator/Eagle3-lingbao.git
cd Eagle3-lingbao

cp configs/lingbao.env.example configs/lingbao.env
vim configs/lingbao.env

chmod +x scripts/*.sh scripts/*.py tools/*.py
```

默认配置已对应当前服务器的重要固定路径：

- Target：`/data/home/leonardoqin/models/lingbao-response`
- SpecForge：`/workspace/SpecForge`
- Python：`/workspace/SpecForge/.venv/bin/python`
- 训练数据根目录：`/data/home/leonardoqin/datasets/lingbao_eagle3_data`
- SGLang：`0.5.14`
- Target capture：GPU 2、3（TP2）
- Draft training：GPU 4、5、6、7（4 个数据并行 rank）
- 所有服务使用 33xxx 高端口，避开历史 30000/30001 冲突。

容器内还需要已有的 `aiohttp`、`transformers`、`sglang==0.5.14`、SpecForge、Mooncake CUDA13、FlashAttention，以及系统命令 `curl`、`ss`、`setsid`。`04_train.sh doctor` 会检查训练链路的关键依赖。

每一轮实验至少要改：

```bash
EXPERIMENT_NAME=一个全新的实验名
RAW_DATA=本轮原始请求文件
REQUEST_DIR=本轮请求切分目录
REPLAY_OUTPUT=本轮重放输出
PRETOKENIZED_DATA=本轮预分词输出
VOCAB_MAPPING=本轮词表映射
```

配置支持另存多份，运行时选择：

```bash
LINGBAO_CONFIG=/path/to/experiment.env bash scripts/04_train.sh plan
```

## 二、5 个步骤

### Step 1：清洗、精确去重、固定切分

```bash
source /workspace/SpecForge/.venv/bin/activate
set -a
source configs/lingbao.env
set +a

python scripts/01_prepare_data.py \
  --input "${RAW_DATA}" \
  --input-format msg-log \
  --output-dir "${REQUEST_DIR}" \
  --train-size "${TRAIN_SIZE}" \
  --val-size "${VAL_SIZE}" \
  --seed "${SPLIT_SEED}"
```

它做四件保守的事：解析 `msg` 中的 request payload、NFKC/不可见字符规范化、异常长度过滤、Prompt 精确去重，然后一次性固定 Train/Validation/Test。它不会把数字替换成 `<NUM>`，因为 Draft 必须拟合 Target 在真实 token 上的条件分布。

输出：

```text
requests_v2/
├── train_requests.jsonl
├── val_requests.jsonl
├── test_requests.jsonl
├── prepare_rejects.jsonl
└── prepare_report.json
```

当前历史基线是 38,663 条：Train 30,000、Validation 3,000、Test 5,663。新版本以本次报告为准，不能用旧数字硬校验。

> Data-Juicer 的 LID、MinHash 和质量模型可以作为 Step 1 的上游增强，但不要把它与每次训练强耦合。38K 业务请求首先保证规则透明、可回溯；近重复删除要在切分前完成，防止跨集合泄漏。

### Step 2：用 Target 35B 重放 Train

```bash
# 一条命令启动 3 个 TP2 副本、并发重放、校验并停止服务。
bash scripts/02_replay.sh all

# 断线后可直接重跑，脚本按 ID 断点续传。
bash scripts/02_replay.sh run

# 单独检查结果。
bash scripts/02_replay.sh validate
```

重放的目的不是评测 Target，而是生成教师答案：

```json
{"id":"...","text":"原prompt + Target completion + <|im_end|>\n","completion":"...","finish_reason":"stop"}
```

原请求有 `seed` 就原样传递；没有就完全不传。`temperature=0.9/top_p=0.8/top_k=20/repetition_penalty=1.05/max_tokens=200` 等参数来自每条请求，不由脚本擅自重写。只有 Train 需要重放；Validation/Test 保留请求即可用于 A/B。

Client 进度保存在 `replay_runtime/logs/replay_client.log`；最终还会生成 `.run_report.json` 和 `.validation.json`。不要只看“进程结束”，必须以 ID 完整、无重复、无 `finish_reason=length` 的 validation 报告为准。

### Step 3：生成训练资产

```bash
bash scripts/03_prepare_training.sh build
bash scripts/03_prepare_training.sh inspect
```

输出三个真正给训练使用的资产：

- `input_ids`：Target tokenizer 对完整 `prompt + completion + EOS` 的编码；
- `loss_mask`：Prompt 为 0，Target completion 与 EOS 为 1；
- `vocab mapping`：从 Target 248,320 词表映射到 Draft 32,000 词表。

默认使用 `target-faithful` mask。`inspect` 会直接解码三条监督文本，必须人工确认看到的是回答和 EOS，而不是整段长 Prompt。脚本逐条调用 SpecForge 预处理，避免历史上“30,000 条最后只剩 32 条”的 batch 维度错误。

### Step 4：Plan、训练、检查、导出

先做预检和 Plan：

```bash
bash scripts/04_train.sh doctor
bash scripts/04_train.sh render
bash scripts/04_train.sh plan
```

`plan` 成功后才开始训练：

```bash
bash scripts/04_train.sh run
bash scripts/04_train.sh status
bash scripts/04_train.sh tail
```

脚本内部使用的关键原生命令没有被隐藏：

```bash
export PYTHONPATH=/workspace/SpecForge

# 只生成/检查进程计划，不占卡训练。
/workspace/SpecForge/.venv/bin/specforge train \
  -c "${TRAIN_CONFIG}" \
  --plan

# 真正训练；04_train.sh 额外负责 nohup、PID 和日志文件。
/workspace/SpecForge/.venv/bin/specforge train \
  -c "${TRAIN_CONFIG}"
```

训练结束：

```bash
bash scripts/04_train.sh checkpoints
bash scripts/04_train.sh export
```

必须理解这两个产物的区别：

- `training_state.pt` 是断点续训状态，不能直接给 SGLang；
- `specforge export --to sglang` 生成的目录才是可部署 Draft Model。

对应的原生导出命令是：

```bash
/workspace/SpecForge/.venv/bin/specforge export --to sglang \
  --checkpoint "${EXPORT_CHECKPOINT}" \
  --draft-config "${DRAFT_CONFIG}" \
  --vocab-mapping "${VOCAB_MAPPING}" \
  --output-dir "${EXPORT_DIR}"
```

当前模板的核心训练配置：

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `num_epochs` | 5 | 每条 Train 样本看到 5 次；不是越大越好 |
| `batch_size × world_size` | 1 × 4 | 每个 optimizer step 消耗 4 条样本 |
| `learning_rate` | 1e-4 | Draft 的初始学习率 |
| `ttt_length` | 7 | 训练 7 个未来位置的预测目标 |
| `attention_backend` | fa | 避开当前 SM120 FlexAttention Triton 共享内存错误 |
| `compact_teacher` | true | 当前本地补丁下分块算教师投影，降低长序列峰值显存 |
| Draft layer | 1 | 只训练 1 层 Draft Transformer，不改 Target 结构或权重 |
| Draft MLP | 8192 | 比 4096 容量更高，也更慢、更占显存 |
| aux layers | 3,19,35 | 采集 Target 浅/中/深三层表示 |

有效总 step 由 SpecForge 的 sampler/world size 决定，以 `--plan` 输出为唯一真值，不要用文件行数直接猜。

### Step 5：Validation 调参和最终 Test

先做 8 条全链路 smoke：

```bash
bash scripts/05_evaluate.sh all smoke
```

再用固定的 500 条 Validation 对 Baseline/EAGLE3 做同卡、同 Prompt、同 seed、同采样参数 A/B：

```bash
bash scripts/05_evaluate.sh all validation
```

默认会测并发 1/2/4/8，并生成：

```text
benchmark_results/<EXPERIMENT_NAME>/validation/
├── baseline_c*.json
├── eagle3_c*.json
├── comparison_c*.json
└── ACCEPTANCE_REPORT.json
```

若要隔离短回答带来的噪声，再单独跑固定 128-token Decode 微基准（它不能代替业务 E2E）：

```bash
EVAL_FIXED_OUTPUT_TOKENS=128 \
EVAL_IGNORE_EOS=1 \
EVAL_OVERWRITE=1 \
bash scripts/05_evaluate.sh all validation
```

Validation 上一次只改变一个变量：

1. 先固定 checkpoint，搜索 `SPEC_NUM_STEPS`、`SPEC_TOPK`、`SPEC_DRAFT_TOKENS`；
2. 再比较不同 checkpoint/epoch；
3. 选择真实 TPOT/E2E/吞吐最好的组合，而不是只选接受长度最大的组合；
4. 把 checkpoint 与 serving 参数写回独立 `.env`，冻结后才碰 Test。

一个建议的小范围搜索表：

| 轮次 | steps | top-k | draft tokens | 目的 |
|---|---:|---:|---:|---|
| A | 3 | 1 | 4 | 低开销基线 |
| B | 3 | 4 | 8 | 增加候选宽度 |
| C | 3 | 4 | 16 | 当前接受长度较高配置 |
| D | 4 | 4 | 16 | 判断更深树是否抵消收益 |

最终盲测：

```bash
CONFIRM_FINAL_TEST=1 bash scripts/05_evaluate.sh all test
```

Test 只回答“冻结后的方案能否上线”，不能看完 Test 再回头调参数，否则 Test 就退化成另一个 Validation。

## 三、如何读验收结果

最关心的 `avg_accept_length` 表示一次 Target 验证平均推进多少 token。它从 1.5 提升到 2.0，理论 Target 调用步数下降，但不代表端到端一定加速，因为 Draft 建树、采样、KV 和验证都有额外开销。

优先级应是：

1. 500/500 成功，输出质量与 Baseline 等价；
2. `avg_accept_length` 明显高于 1，且跨并发稳定；
3. TPOT speedup > 1；
4. 输出 token 吞吐 speedup > 1；
5. E2E 不回退，并观察 p90/p99；
6. TTFT 单独看，投机解码主要优化 Decode，不保证 TTFT 改善。

默认 Gate 是接受长度 ≥ 2.0、输出吞吐不回退、E2E 回退不超过 5%。阈值可以通过环境变量修改，但修改要写进实验记录。

## 四、代码地图

用户只运行：

```text
scripts/01_prepare_data.py       数据清洗、去重、切分
scripts/02_replay.sh             多服务 Target 重放
scripts/03_prepare_training.sh   token/mask/mapping
scripts/04_train.sh              doctor/plan/train/export
scripts/05_evaluate.sh           smoke/validation/test A/B
```

`tools/` 是上述入口调用的内部实现，不需要日常逐个执行。完整故障记录见 [docs/DEBUGGING.md](docs/DEBUGGING.md)，调参决策见 [docs/VALIDATION.md](docs/VALIDATION.md)。

## 五、与 SpecForge 官方流程的关系

本仓库沿用官方的 `specforge train --config`、checkpoint 和 `specforge export --to sglang` 主流程。需要特别注意：当前官方文档把 compact teacher 标为离线文本 EAGLE3 能力；本项目历史在线成功运行依赖 `/workspace/SpecForge` 的本地 streaming capability 补丁。因此每次都必须执行 `doctor` 和 `plan`，不能假设换一套 SpecForge 环境仍兼容。

- SpecForge Training：https://sgl-project.github.io/SpecForge/basic_usage/training.html
- Data Preparation：https://sgl-project.github.io/SpecForge/basic_usage/data_preparation.html
- Benchmark：https://sgl-project.github.io/SpecForge/benchmarks/benchmark.html
