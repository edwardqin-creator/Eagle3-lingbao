# 灵宝 EAGLE-3 实验全景总结

更新时间：2026-08-13

线上仓库：`/data/home/leonardoqin/Eagle3-lingbao`

结果根目录：`/data/home/leonardoqin/datasets/lingbao_eagle3_data/benchmark_results`

本文只总结灵宝业务模型的 EAGLE-3 训练、候选树、采样温度、模型容量和请求级诊断。每个“提升”都写明对照组；验证集不同或一次修改多个变量的实验不会被当成严格单变量结论。

## 1. 当前主线到底是什么配置

当前主线不是最新跑完的 I4K，而是目前唯一通过正式 Validation Gate 的：

```text
实验名：lingbao-eagle3-v2-all-i8k-oldopt-e5-v1
Target：/data/home/leonardoqin/models/lingbao-response
Draft：/data/home/leonardoqin/models/exports/lingbao-eagle3-v2-all-i8k-oldopt-e5-v1-sglang
训练YAML：/data/home/leonardoqin/datasets/lingbao_eagle3_data/configs/lingbao-eagle3-v2-all-i8k-oldopt-e5-v1.yaml
Draft JSON：/data/home/leonardoqin/datasets/lingbao_eagle3_data/configs/qwen3.5-35b-a3b-eagle3-i8k-data-v2-all-v1.json
预分词数据：/data/home/leonardoqin/datasets/lingbao_eagle3_data/pretokenized_v2/train_eagle3_data_v2_all_tokens_10240.jsonl
Vocab Mapping：/data/home/leonardoqin/datasets/lingbao_eagle3_data/cache/vocab_mapping/qwen3.5-35b-a3b-eagle3-data-v2-all-v1.pt
训练数据：Data V2 全任务 Train，77,384 条
训练轨迹温度：T=0.9
评测/服务温度：T=0
推理树：Steps=2，TopK=2，Draft Tokens=4
评测请求：/data/home/leonardoqin/datasets/lingbao_eagle3_data/data_version2/val_requests.temperature_0.jsonl
正式口径：Validation 500 条，并发 1，sample_seed=20260805
```

### 1.1 Draft 结构

| 配置项 | 当前值 |
|---|---:|
| 架构 | `LlamaForCausalLMEagle3` |
| Target Hidden Streams | `[3, 19, 35]` |
| Draft Transformer层数 | 1 |
| Hidden Size | 2,048 |
| Intermediate Size | 8,192（I8K） |
| Attention Heads / KV Heads | 16 / 2 |
| Head Dim | 256 |
| 激活函数 | SiLU |
| 精度 | BF16 |
| Target词表 | 248,320 |
| Draft词表 | 32,000 |
| TTT Length | 7 |

配置文件：`configs/qwen3.5-35b-a3b-eagle3-i8k.json`。

### 1.2 数据与训练配置

| 配置项 | 当前值 |
|---|---:|
| Train / Validation / Test | 77,384 / 7,827 / 14,775 |
| 任务分布 | 闲聊51.25%、决策26.24%、知识13.95%、信息8.56% |
| 最大训练序列 | 10,240 Token |
| Epoch | 5 |
| 每卡Batch | 1 |
| Draft训练Rank | 4 |
| Global Batch | 4 |
| Gradient Accumulation | 1 |
| Learning Rate | `1e-4` |
| Warmup Ratio | 0.015 |
| Max Grad Norm | 0.5 |
| AdamW β₁ / β₂ | 0.9 / 0.999 |
| AdamW ε / Weight Decay | `1e-8` / 0 |
| Attention Backend | FA |

本文把这套优化器配置简称为 **OldOpt**。训练时通常由 2 张 GPU 运行 Target Capture、4 张 GPU 训练 Draft；GPU 编号只是资源编排，不属于模型超参数。

### 1.3 当前服务与评测配置

| 配置项 | 当前值 |
|---|---:|
| Target Tensor Parallel | 2 |
| Target / Draft精度 | BF16 |
| Context Length | 16,384 |
| Static Memory Fraction | 0.60 |
| Max Running Requests | 8 |
| EAGLE-3 Steps / TopK / Draft Tokens | 2 / 2 / 4 |
| Sampling Temperature | 0 |
| 正式并发 | 1 |
| 正式样本 | 500 |
| Warmup | 8 |
| Decode Log Interval | 40 |
| Request Diagnostics | 关闭 |

### 1.4 训练轨迹与正式评测不是同一个温度

- Draft 使用 Target 在 `T=0.9、top_p=0.8、top_k=20、repetition_penalty=1.05` 下重放的轨迹训练。
- 当前唯一 PASS 是把同一个 Draft 放到 `T=0` 的 Target 轨迹上评测。
- 所以当前已经证明的是“`T=0.9` 轨迹训练的 Draft 可以在 `T=0` 服务口径下获益”，不是“已用 `T=0` 重新训练”。
- 如果线上必须保持 `T=0.9`，只能引用 `T=0.9` 的正式结果，不能拿 `T=0` PASS 代替线上验收。

## 2. 当前唯一正式 PASS：绝对值与真实增长

实验目录：

```text
/data/home/leonardoqin/datasets/lingbao_eagle3_data/benchmark_results/
lingbao-eagle3-v2-all-i8k-e5-eval-s2k2d4-t00/validation
```

配置：Data V2 全任务、I8K、OldOpt、E5、Validation `T=0`、树 `2/2/4`、并发 1、500 条。

| 指标 | Target-only | EAGLE-3 | EAGLE-3相对增长 |
|---|---:|---:|---:|
| 请求吞吐 | 3.966 req/s | 3.985 req/s | **1.005x，+0.5%** |
| 输出吞吐 | 77.86 tok/s | 78.45 tok/s | **1.008x，+0.8%** |
| E2E p50 | 245.49 ms | 245.09 ms | p50下降0.16%；mean **改善+0.48%** |
| E2E p90 | 303.51 ms | 299.12 ms | 下降1.45% |
| E2E p99 | 335.25 ms | 330.47 ms | 下降1.43% |
| TTFT p50 | 157.54 ms | 166.55 ms | 回退5.72%；mean **回退5.45%** |
| TPOT p50 | 4.65 ms/token | 3.93 ms/token | p50下降15.5%；mean **加速1.133x** |
| 平均接受长度 | — | 2.109 | 每次Target验证平均产出2.109 Token |
| 接受率 | — | 36.95% | 只在同一个`2/2/4`树内比较 |

结论不是“已经大幅加速”，而是：EAGLE-3 首次跨过端到端正收益线。TPOT 已有明确收益，但输出吞吐仅 `+0.8%`、E2E mean 仅 `+0.48%`，仍需重复跑和并发测试确认稳定性。

## 3. 增长账本：到底是哪一步增长了多少

### 3.1 严格或近似可控的关键对照

| 改动与固定条件 | 接受长度变化 | 接受率变化 | TPOT变化 | 输出吞吐变化 | E2E改善变化 | 判断 |
|---|---:|---:|---:|---:|---:|---|
| 数据/覆盖：旧Chat→Data V2；I8K E5、T0.9、`3/4/16` | 2.267→2.546，**+0.279（+12.3%）** | 8.45%→10.29%，**+1.84pp** | 0.858x→0.972x，**+0.114x（+13.3%）** | 0.904x→0.955x，**+0.051x（+5.6%）** | -10.54%→-6.05%，**+4.49pp** | 数据/覆盖扩大明显正向 |
| 数据/覆盖：旧Chat→Data V2；I8K E5、T0.9、`2/2/4` | 1.835→2.016，**+0.181（+9.9%）** | 27.88%→33.90%，**+6.02pp** | 0.964x→1.065x，**+0.101x（+10.5%）** | 0.958x→0.995x，**+0.037x（+3.9%）** | -6.43%→-1.46%，**+4.97pp** | 小树下同样正向 |
| 温度：Data V2同Draft、`2/2/4`，评测T0.9→T0 | 2.016→2.109，**+0.093（+4.6%）** | 33.90%→36.95%，**+3.05pp** | 1.065x→1.133x，**+0.068x（+6.4%）** | 0.995x→1.008x，**+0.013x（+1.3%）** | -1.46%→+0.48%，**+1.94pp** | 当前最明确的服务侧正变量 |
| 温度：Data V2同Draft、`3/4/16`，评测T0.9→T0 | 2.546→2.715，**+0.169（+6.6%）** | 10.29%→11.47%，**+1.18pp** | 0.972x→1.052x，**+0.080x（+8.2%）** | 0.955x→0.969x，**+0.014x（+1.5%）** | -6.05%→-2.66%，**+3.39pp** | T0提升可在大树复现 |
| 树：Data V2、T0，同Draft，`3/4/16`→`2/2/4` | 2.715→2.109，**-0.606（-22.3%）** | 候选数不同，不横比 | 1.052x→1.133x，**+0.081x（+7.7%）** | 0.969x→1.008x，**+0.039x（+4.0%）** | -2.66%→+0.48%，**+3.14pp** | 接受长度下降，但真实性能转正 |
| 全链路：旧Chat T0.9 `2/2/4`→Data V2 T0 `2/2/4` | 1.835→2.109，**+0.274（+14.9%）** | 27.88%→36.95%，**+9.08pp** | 0.964x→1.133x，**+0.169x（+17.5%）** | 0.958x→1.008x，**+0.050x（+5.2%）** | -6.43%→+0.48%，**+6.91pp** | 数据扩大+低熵评测+小树形成当前PASS |
| 容量：Data V2 E5 T0 `2/2/4`，I8K→I4K | 2.109→2.077，**-0.031（-1.5%）** | 36.95%→35.98%，**-0.97pp** | 1.133x→1.118x，**-0.015x（-1.3%）** | 1.008x→0.997x，**-0.011x（-1.1%）** | +0.48%→-0.47%，**-0.95pp** | I4K由PASS退为FAIL |

`Data V2 T=0.9、2/2/4` 的 `2.016 / 33.90% / 1.065x / 0.995x / -1.46%` 来自历史记录，但其结果目录尚未补入注册表；后续应回填原始 JSON 后再作为最终审计证据。

### 3.2 非严格单变量，但已观察到的负向差异

| 实验包 | 对照→实验 | 变化量 | 能否单独归因 |
|---|---|---|---|
| Adam95重参数包 | Brief I12K OldOpt E5→I12K `LR5e-5/Accum2/β₂.95` E7 | ACC **-0.385（-17.0%）**；TPOT **-0.174x（-20.5%）**；输出 **-0.086x（-9.6%）**；E2E **-8.59pp** | 不能；LR、Accum、β₂、Epoch一起变了 |
| Adam95继续到E10 | Brief I12K OldOpt E5→Adam95 E10 | ACC **-0.427（-18.9%）**；TPOT **-0.187x（-22.0%）**；输出 **-0.085x（-9.5%）**；E2E **-10.85pp** | 不能；证明多训没有救回组合退化 |
| Epoch E4→E5 | Data V2 I8K、T0.9、`3/4/16` | ACC 2.582→2.546，**-1.4%**；TPOT 0.988x→0.972x；E2E -5.07%→-6.05% | 近似可比；没有E5优于E4的证据 |
| 全任务→Chat-only专项 | 全任务模型/全任务Val→Chat模型/Chat Val，均T0 `2/2/4` | ACC **-0.202（-9.6%）**；TPOT **-0.106x（-9.4%）**；输出 **-0.036x（-3.6%）**；E2E **-4.02pp** | 不能纯归因；训练集和验证集都变了 |
| I8K→I12K观察 | 旧Chat I8K→Brief I12K OldOpt E5，T0.9 `3/4/16` | ACC 2.267→2.264，基本持平；TPOT -0.008x；输出 -0.005x；E2E -0.96pp | 不是严格同数据；至少没有看到容量收益 |

## 4. 实验时间线与结果路径

### 4.1 旧Chat与Brief阶段

| 日期/实验 | 训练配置 | 评测配置 | ACC / Rate | TPOT / 输出 / E2E | 结果目录 |
|---|---|---|---|---|---|
| 旧Chat E5 | 旧Chat约25K、I8K、OldOpt、E5、轨迹T0.9 | Val T0.9、`3/4/16` | 2.267 / 8.45% | 0.858x / 0.904x / -10.54% | `benchmark_results/lingbao-eagle3-chatonly-fullattn-i8k-e5-v1/validation` |
| 旧Chat小树 | 同上 | Val T0.9、`2/2/4` | 1.835 / 27.88% | 0.964x / 0.958x / -6.43% | `benchmark_results/lingbao-chatonly-e5-s2c-steps2-topk2-d4/validation` |
| Brief Adam95 E7 | Brief 25,181、I12K、LR5e-5、Accum2、β₂.95 | Val T0.9、`3/4/16` | 1.879 / 5.91% | 0.676x / 0.813x / -20.09% | `benchmark_results/lingbao-eagle3-briefchat-i12k-adam95-e10-v1-e7/validation` |
| Brief Adam95 E10 | 同上，E10 | Val T0.9、`3/4/16` | 1.837 / 5.56% | 0.663x / 0.814x / -22.35% | `benchmark_results/lingbao-eagle3-briefchat-i12k-adam95-e10-v1-e10/validation` |
| A I12K OldOpt E5 | Brief 25,181、I12K、OldOpt、E5 | Val T0.9、`3/4/16` | 2.264 / 8.44% | 0.850x / 0.899x / -11.50% | `benchmark_results/briefchat-e5-a-i12k-max-s3k4d16-prod-t09/validation` |
| B I8K Greedy E5 | Brief 25,181、I8K、OldOpt、T0轨迹、E5 | Val T0、`3/4/16` | 2.334 / 8.98% | 0.903x / 0.926x / -8.75% | `benchmark_results/valcheck-b-greedy-s3k4d16-t00/validation` |

阶段结论：旧模型已经具备预测能力，但短输出摊不平大树成本；Adam95组合显著负向；恢复OldOpt后能力恢复；I12K没有证明比I8K更好。

### 4.2 Data V2主线

| 实验 | 训练配置 | 评测配置 | ACC / Rate | TPOT / 输出 / E2E | 结果目录 |
|---|---|---|---|---|---|
| V2 I8K E4 | 全任务77,384、I8K、OldOpt、E4、轨迹T0.9 | Val T0.9、`3/4/16` | 2.582 / 10.54% | 0.988x / 0.951x / -5.07% | `benchmark_results/lingbao-eagle3-v2-all-i8k-e4-eval-s3k4d16/validation` |
| V2 I8K E5 | 全任务77,384、I8K、OldOpt、E5、轨迹T0.9 | Val T0.9、`3/4/16` | 2.546 / 10.29% | 0.972x / 0.955x / -6.05% | `benchmark_results/lingbao-eagle3-v2-all-i8k-e5-eval-s3k4d16/validation` |
| V2 I8K E5 T0大树 | 同一个E5 Draft | Val T0、`3/4/16` | 2.715 / 11.47% | 1.052x / 0.969x / -2.66% | `benchmark_results/lingbao-eagle3-v2-all-i8k-e5-eval-s3k4d16-t00/validation` |
| **V2 I8K E5 T0小树** | **同一个E5 Draft** | **Val T0、`2/2/4`** | **2.109 / 36.95%** | **1.133x / 1.008x / +0.48%** | `benchmark_results/lingbao-eagle3-v2-all-i8k-e5-eval-s2k2d4-t00/validation` |
| V2 I8K E5 Train Check | 同一个E5 Draft | Train T0、`2/2/4` | 2.146 / 38.17% | 1.165x / 1.027x / +0.97% | `benchmark_results/lingbao-v2-e5-train-t00-s2k2d4-v1/validation` |
| V2 Chat-only I8K E5 | Chat-only39,657、I8K、OldOpt、E5 | Chat Val T0、`2/2/4` | 1.907 / 30.30% | 1.027x / 0.972x / -3.54% | `benchmark_results/lingbao-eagle3-v2-chatonly-i8k-e5-eval-s2k2d4-t00/validation` |
| V2 I4K E5 | 全任务77,384、仅MLP改4K | Val T0、`2/2/4` | 2.077 / 35.98% | 1.118x / 0.997x / -0.47% | `benchmark_results/lingbao-eagle3-v2-all-i4k-e5-eval-s2k2d4-t00-val-v1/validation` |
| V2 I4K Train Check | 同一个I4K Draft | Train T0、`2/2/4` | 2.101 / 36.67% | 1.143x / 1.015x / +0.09% | `benchmark_results/lingbao-eagle3-v2-all-i4k-e5-eval-s2k2d4-t00-train-v1/validation` |

阶段结论：Data V2是最可靠的训练侧正变量；T0是最明确的服务侧正变量；`2/2/4`是当前唯一把两者转成端到端正收益的树；Chat-only和I4K都没有超过全任务I8K。

## 5. 候选树：接受长度增加了多少，代价是多少

以下来自 A I12K OldOpt E5、Validation T0.9。以最小树 `2/2/4` 为对照：

| Tree | ACC | ACC增量 | TPOT | TPOT变化 | 输出 | 输出变化 | E2E | E2E变化 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2/2/4 | 1.814 | 基准 | 0.955x | 基准 | 0.951x | 基准 | -6.60% | 基准 |
| 3/2/4 | 1.858 | +0.044 | 0.928x | -0.027x | 0.936x | -0.015x | -6.80% | -0.20pp |
| 3/4/16 | 2.264 | +0.450 | 0.850x | -0.105x | 0.899x | -0.052x | -11.50% | -4.90pp |
| 4/4/16 | 2.299 | +0.485 | 0.822x | -0.133x | 0.861x | -0.090x | -11.27% | -4.67pp |
| 4/4/32 | 2.391 | +0.577 | 0.703x | -0.252x | 0.814x | -0.137x | -20.38% | -13.78pp |
| 4/8/48 | 2.492 | +0.678 | 0.606x | -0.349x | 0.769x | -0.182x | -32.83% | -26.23pp |
| 5/8/64 | 2.588 | +0.774 | 0.535x | -0.420x | 0.722x | -0.229x | -41.34% | -34.74pp |

规律很清楚：从 `2/2/4` 到 `5/8/64`，接受长度只增加 `0.774`，输出吞吐却再下降 `22.9%` 基准量、E2E再恶化 `34.74` 个百分点。当前约20 Token的短回答没有足够Decode长度摊销大树。

因此：

- `2/2/4`：真实性能与上线候选树。
- `3/4/16`：诊断Draft接受能力的统一上限树。
- 更大树：已经证明不适合当前业务，不再作为提升平均接受长度的主路径。

## 6. Train、Validation与任务族诊断

### 6.1 500条正式Train/Val

| Split | ACC | Rate | TPOT | 请求吞吐 | 输出吞吐 | E2E | TTFT |
|---|---:|---:|---:|---:|---:|---:|---:|
| Validation T0 | 2.109 | 36.95% | 1.133x | 1.005x | 1.008x | +0.48% | -5.45% |
| Train T0 | 2.146 | 38.17% | 1.165x | 1.010x | 1.027x | +0.97% | -5.14% |
| Train相对Val | **+0.037（+1.75%）** | **+1.21pp** | **+0.032x** | +0.005x | +0.019x | +0.49pp | +0.31pp |

Train只比Val高1.75%，不支持明显记忆或严重过拟合；也不能因为Train没有远高于Val就直接判定欠拟合。

### 6.2 1500条请求级诊断

| Split | N | ACC | Rate | TPOT | E2E | TTFT | 墙钟输出吞吐 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Validation | 1,500 | 2.126 | 37.57% | 1.149x | +0.66% | -5.64% | 0.937x（诊断等待影响，不验收） |
| Train | 1,500 | 2.148 | 38.31% | 1.160x | +1.22% | -4.82% | 0.949x（诊断等待影响，不验收） |

路径：

- Validation：`benchmark_results/lingbao-v2-e5-t00-s2k2d4-requestdiag-v1/validation`
- Train：`benchmark_results/lingbao-v2-e5-train-t00-s2k2d4-requestdiag-gpu67-v1/validation`

请求诊断会在每个请求计时结束后等待服务日志，因此单请求的 TTFT/TPOT/E2E 不包含等待，但整轮 `duration_s`、请求吞吐和输出吞吐包含等待。诊断轮不能替代 Formal Gate。

### 6.3 Validation任务族（1500条，all_pairs）

| 任务族 | N | ACC | 相对全体ACC 2.126 | TPOT | 输出 | E2E | 结论 |
|---|---:|---:|---:|---:|---:|---:|---|
| 信息处理 | 146 | 2.408 | **+0.282（+13.3%）** | 1.278x | 1.028x | +1.52% | 稳定正收益 |
| 决策辅助 | 380 | 2.336 | **+0.210（+9.9%）** | 1.262x | 1.047x | +4.49% | 当前收益最佳任务族 |
| 对局闲聊 | 776 | 2.092 | **-0.034（-1.6%）** | 1.101x | 0.995x | -0.91% | 体量最大，决定总盘临界 |
| 知识问答 | 198 | 2.024 | **-0.102（-4.8%）** | 1.056x | 0.996x | -2.43% | 当前最弱任务族 |

同输出文本的 `same_output_sha256` 口径下：信息处理/决策辅助 E2E 分别 `+2.04%/+4.69%`；闲聊接近持平 `+0.06%`；知识问答仍回退 `-1.07%`。这说明结果不是只由两次采样输出文本不同造成。

### 6.4 Train任务族（1500条，all_pairs）

| 任务族 | N | ACC | TPOT | 输出 | E2E |
|---|---:|---:|---:|---:|---:|
| 信息处理 | 134 | 2.399 | 1.286x | 1.033x | +1.98% |
| 决策辅助 | 382 | 2.352 | 1.270x | 1.058x | +4.85% |
| 对局闲聊 | 744 | 2.103 | 1.106x | 1.010x | -0.57% |
| 知识问答 | 240 | 2.119 | 1.115x | 1.020x | -0.53% |

Train/Val的任务排序一致，说明下一轮应优先优化闲聊与知识问答的首Token/次Token，而不是给所有任务均匀增加训练量。

## 7. 已证明、未证明和已经否定的方向

### 7.1 已有数据支持

1. **扩大业务数据/任务覆盖正向。**同树同温度下，ACC提升约9.9%～12.3%，TPOT提升约10.5%～13.3%。
2. **降低Target采样熵正向。**同Draft同树把T0.9改T0，ACC提升4.6%，TPOT提升6.4%，使输出吞吐从0.995x跨到1.008x。
3. **短输出必须用小树。**T0下从`3/4/16`缩到`2/2/4`，ACC下降22.3%，但输出提升4.0%，E2E改善3.14pp并转正。
4. **I8K是当前容量甜点。**I4K节省计算后ACC下降1.5%，正式Val反而退为FAIL；I12K未看到接受长度收益。
5. **当前没有明显Train记忆优势。**Train/Val ACC只差约1%～2%。

### 7.2 尚未严格证明

1. AdamW β₂ `.95` 是否单独负向：现有实验同时改了LR、Accum和Epoch。
2. Chat-only是否一定差：当前训练集和验证集都不同，需要把全任务Draft与Chat-only Draft放到同一个Chat-only Val上公平比较。
3. `T=0`重放训练是否优于`T=0.9`重放训练：当前主线Draft仍由T0.9轨迹训练。
4. Hidden Streams `[3,19,35]` 是否最优：尚未做单变量层选择消融。
5. Post-Norm、接受感知Loss、Soft-logit/KL蒸馏：尚未进入正式对照实验。

### 7.3 当前不再优先

1. 继续扩大候选树追求ACC=3：已多次证明真实性能恶化。
2. 无条件增加Epoch：E5不优于E4，E10不优于E7。
3. 把MLP降到4K换速度：正式Val已经失败。
4. 直接只用Chat-only缩窄数据：现有专项模型没有复现全任务PASS。
5. 扩大32K Mapping：当前监督Token、可见回答和回答前4 Token覆盖率均为100%。

## 8. 当前结论与下一步

当前结果的完整因果链是：

```text
旧Chat I8K E5、T0.9、2/2/4
ACC 1.835 / TPOT 0.964x / 输出0.958x / E2E -6.43%
        │
        ├─ Data V2全任务扩大覆盖
        │  ACC +9.9%，TPOT +10.5%，E2E +4.97pp
        │
        ├─ 评测T0降低Target采样熵
        │  ACC再+4.6%，TPOT再+6.4%，E2E再+1.94pp
        │
        └─ 保持2/2/4控制树开销
           ACC 2.109 / TPOT 1.133x / 输出1.008x / E2E +0.48%
```

下一阶段按以下顺序推进：

1. **稳住收益：**同一配置重复 Formal 2～3 次，并补并发2/4/8；报告均值、标准差和最差值。
2. **冻结Test：**所有Val参数冻结后跑一次Test，禁止再用Test调参。
3. **不扩大树提升前两Token：**对闲聊、知识问答做低接受难例挖掘；优先验证前位置加权Loss或TTT=4。
4. **蒸馏：**只保存32K映射空间内前2～3位置的Target TopK logits，做小规模KL消融。
5. **模型结构单变量：**Hidden Stream层选择、输入stream RMSNorm、Draft Post-Norm分别独立实验，不能与数据/优化器一起改。
6. **线上温度约束：**若生产必须T0.9，优化目标应回到T0.9的`0.995x`临界点，不能把T0 PASS视作已经完成上线验收。

机器可读实验台账：`experiments/registry.json`。

复现、温度派生、树搜索、请求级诊断和任务族后处理方法见 [POSTPROCESSING.md](POSTPROCESSING.md)。
