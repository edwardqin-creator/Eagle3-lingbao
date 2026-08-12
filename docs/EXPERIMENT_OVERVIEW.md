# 灵宝 EAGLE-3 实验全景总结

更新时间：2026-08-12

范围：旧 Chat-only E5、Brief 参数消融、Data Version 2 全任务/Chat-only、I4K容量消融、请求级与任务族诊断。

线上仓库：`/data/home/leonardoqin/Eagle3-lingbao`

结果根目录：`/data/home/leonardoqin/datasets/lingbao_eagle3_data/benchmark_results`

> 除特别说明外，性能变化均相对于该轮自己的 Target-only Baseline。接受长度越大越好；TPOT、请求/输出吞吐大于 `1.0x` 才是正收益；E2E 改善正数表示延迟下降。Diagnostic 轮的墙钟吞吐不用于验收。

## 1. 一页结论

1. **当前唯一正式 Validation PASS** 是 `Data V2全任务 + I8K + OldOpt + E5 + T=0 + 2/2/4 + c1`：500条接受长度 `2.109`、接受率 `36.95%`、TPOT `1.133x`、输出吞吐 `1.008x`、E2E改善 `+0.48%`。
2. 当前收益主要来自 Decode。TTFT仍回退约5.45%，平均输出仅约20 Token，Prefill/TTFT和候选验证开销很难摊薄。
3. **数据扩容有效**：旧Chat-only约25K扩到Data V2全任务77,384后，同类配置的接受长度与TPOT明显提高。
4. **T=0是最明确的服务侧正变量**：固定Target argmax轨迹后，连续分叉减少；同一Data V2 E5 Draft的`2/2/4`从临界转为首个PASS。但如果线上仍为T=0.9，T=0成绩不能直接等同线上收益。
5. **`2/2/4`是当前真实性能树**；`3/4/16`只适合诊断接受上限。激进树能把接受长度抬至2.5左右，却会让吞吐和E2E大幅回退。
6. **I4K容量消融是负向的**：Val从I8K的`2.109/1.133x/1.008x/+0.48%`退化到`2.077/1.118x/0.997x/-0.47%`。节省MLP计算没有抵消接受能力损失，当前保留I8K。
7. **Chat-only专项化是负向的**：Data V2 Chat-only模型在T=0、`2/2/4`上仅`1.907/1.027x/0.972x/-3.54%`。减少样本量和跨任务覆盖没有自动提高闲聊能力。
8. **Train/Val接近，不支持明显过拟合**：1500条请求诊断中，Train/Val接受长度约`2.148/2.126`，只差`0.022`。
9. **任务族差异非常明确**：信息处理、决策辅助稳定正收益；对局闲聊、知识问答接受长度更低，E2E仍临界或回退，是难例挖掘和蒸馏的优先对象。
10. LR减半、accumulation翻倍、AdamW beta2改为`.95`、I12K与E10同时修改的Brief实验明显负向；恢复OldOpt后能力恢复。后续坚持单变量消融。

## 2. 评测口径

| 口径 | 样本/配置 | 作用 | 是否用于最终Gate |
|---|---|---|---|
| Formal | 常用500条；request diagnostics关闭 | 真实性能、吞吐、E2E | 是 |
| Request Diagnostic | 常用1500条；c1、decode interval 1 | 每请求ACC、任务族、难例 | 否 |
| Train check | 冻结Train请求的随机子集 | 欠拟合/记忆诊断 | 不能代替Val/Test |
| Test | 参数全部冻结后一次性运行 | 最终泛化验收 | 是；禁止反向调参 |

Diagnostic客户端在请求计时结束后等待20ms读取服务日志，因此请求级TTFT/TPOT/E2E本身不直接包含这20ms，但整轮 `duration_s` 与墙钟吞吐包含等待。诊断轮输出吞吐低于1不等于正式性能回退。

## 3. 演进时间线

| 时间 | 阶段 | 核心变化 | 代表结果 | 结论 |
|---|---|---|---|---|
| 2026-07-31 | 旧Chat E5 | I8K、OldOpt、E5、T0.9 | 3/4/16 ACC 2.267，输出0.904x | Draft已学习，但大树无法回本 |
| 2026-08-01 | 树搜索 | 同Draft缩到2/2/4 | ACC 1.835，TPOT0.964x | 小树方向正确 |
| 2026-08-04 | Brief重参数 | I12K、LR5e-5、accum2、beta2=.95、E7/E10 | ACC 1.879/1.837 | 组合修改显著负向 |
| 2026-08-05 | A/B隔离 | 恢复OldOpt；另做Greedy训练/评测 | A恢复2.264；B T0评测2.334 | OldOpt恢复；T0降低轨迹熵 |
| 2026-08-05 | 激进树 | 4/4/32至5/8/64 | 最大ACC2.588，输出0.722x | 接受长度不能脱离开销 |
| 2026-08-07 | Data V2 | 全任务77,384、I8K、OldOpt、E5 | T0.9大树ACC2.546 | 数据扩容正向 |
| 2026-08-07 | 温度/小树 | 同Draft改T0与2/2/4 | ACC2.109、输出1.008x、E2E+0.48% | 首个正式Val PASS |
| 2026-08-10 | Chat-only专项 | 39,657条、其余同主线 | ACC1.907、输出0.972x | 专项化退化 |
| 2026-08-11 | 请求诊断 | Train/Val各1500、每请求ACC | 2.148/2.126 | 未见明显过拟合；任务族分化 |
| 2026-08-12 | I4K消融 | 只改MLP 8192→4096 | Val输出0.997x、E2E-0.47% | 容量减半不划算 |

## 4. 训练模型与主要结果

| 模型 | Train数据 | 轨迹T | MLP | Epoch | LR/Accum/β₂ | 评测 | Tree | ACC | 接受率 | TPOT | 输出 | E2E |
|---|---|---:|---:|---:|---|---|---|---:|---:|---:|---:|---:|
| 旧Chat E5 | 旧Chat-only约25K | 0.9 | 8K | 5 | `1e-4/1/.999` | Val T0.9 | 3/4/16 | 2.267 | 8.45% | 0.858x | 0.904x | -10.54% |
| Brief Adam95 E7 | Brief 25,181 | 0.9 | 12K | 7 | `5e-5/2/.95` | Val T0.9 | 3/4/16 | 1.879 | 5.91% | 0.676x | 0.813x | -20.09% |
| Brief Adam95 E10 | Brief 25,181 | 0.9 | 12K | 10 | `5e-5/2/.95` | Val T0.9 | 3/4/16 | 1.837 | 5.56% | 0.663x | 0.814x | -22.35% |
| A I12K OldOpt E5 | Brief 25,181 | 0.9 | 12K | 5 | `1e-4/1/.999` | Val T0.9 | 3/4/16 | 2.264 | 8.44% | 0.850x | 0.899x | -11.50% |
| B I8K Greedy E5 | Brief 25,181 | 0 | 8K | 5 | OldOpt | Val T0 | 3/4/16 | 2.334 | 8.98% | 0.903x | 0.926x | -8.75% |
| Data V2 E4 | 全任务77,384 | 0.9 | 8K | 4 | OldOpt | Val T0.9 | 3/4/16 | 2.582 | 10.54% | 0.988x | 0.951x | -5.07% |
| Data V2 E5 | 全任务77,384 | 0.9 | 8K | 5 | OldOpt | Val T0 | 3/4/16 | 2.715 | 11.47% | 1.052x | 0.969x | -2.66% |
| **Data V2 I8K E5** | **全任务77,384** | **0.9** | **8K** | **5** | **OldOpt** | **Val T0** | **2/2/4** | **2.109** | **36.95%** | **1.133x** | **1.008x** | **+0.48%** |
| Data V2 Chat-only E5 | Chat-only39,657 | 0.9 | 8K | 5 | OldOpt | Chat Val T0 | 2/2/4 | 1.907 | 30.30% | 1.027x | 0.972x | -3.54% |
| Data V2 I4K E5 | 全任务77,384 | 0.9 | 4K | 5 | OldOpt | Val T0 | 2/2/4 | 2.077 | 35.98% | 1.118x | 0.997x | -0.47% |

### 4.1 参数归因

| 修改 | 观察 | 当前行动 |
|---|---|---|
| 删除347条异常 | 恢复OldOpt后与旧模型基本持平 | 保留清洗；不是跃升来源 |
| MLP 8K→12K | ACC未提高，Draft更贵 | 不采用I12K |
| MLP 8K→4K | Val ACC、TPOT、输出、E2E全面小幅退化 | 保留I8K |
| LR 1e-4→5e-5 | 与其他改动组合后学习能力下降 | 主线恢复1e-4 |
| Accum 1→2 | 每Epoch优化器更新数减半 | 主线恢复1 |
| β₂ .999→.95 | 当前短输出任务未受益 | 主线恢复.999 |
| Epoch 5→10 | E10不及E7；Data V2 E5不及E4大树 | 不无条件加Epoch，按Val选checkpoint |
| 25K→77K | ACC和TPOT显著提高 | 当前最可靠训练侧方向 |
| 全任务→Chat-only | 专项模型明显退化 | 暂不做前置路由/多Draft生产化 |

## 5. 候选树搜索

### 5.1 旧Chat E5，Val T=0.9

| Tree | ACC | 接受率 | TPOT | 输出 | E2E |
|---|---:|---:|---:|---:|---:|
| 3/4/16 | 2.267 | 8.45% | 0.858x | 0.904x | -10.54% |
| 3/2/8 | 2.056 | 15.05% | 0.916x | 0.941x | -7.69% |
| 3/1/4 | 1.783 | 26.15% | 0.914x | 0.929x | -7.77% |
| 3/2/4 | 1.870 | 29.02% | 0.957x | 0.954x | -6.31% |
| 2/2/4 | 1.835 | 27.88% | 0.964x | 0.958x | -6.43% |

### 5.2 A I12K OldOpt E5，Val T=0.9激进树

| Tree | ACC | TPOT | 输出 | E2E |
|---|---:|---:|---:|---:|
| 2/2/4 | 1.814 | 0.955x | 0.951x | -6.60% |
| 3/2/4 | 1.858 | 0.928x | 0.936x | -6.80% |
| 3/4/16 | 2.264 | 0.850x | 0.899x | -11.50% |
| 4/4/16 | 2.299 | 0.822x | 0.861x | -11.27% |
| 4/4/32 | 2.391 | 0.703x | 0.814x | -20.38% |
| 4/8/48 | 2.492 | 0.606x | 0.769x | -32.83% |
| 5/8/64 | 2.588 | 0.535x | 0.722x | -41.34% |

规律：树宽/深增加时ACC只缓慢上升，候选生成和Target验证成本快速上升。平均输出约20 Token时，32/48/64个Draft Tokens没有摊销空间。

### 5.3 Data V2 E5关键温度/树对照

| 评测T | Tree | ACC | 接受率 | TPOT | 输出 | E2E | 状态 |
|---:|---|---:|---:|---:|---:|---:|---|
| 0.9 | 3/4/16 | 2.546 | 10.29% | 0.972x | 0.955x | -6.05% | FAIL |
| 0.9 | 2/2/4 | 2.016 | 33.90% | 1.065x | 0.995x | -1.46% | 临界 |
| 0 | 3/4/16 | 2.715 | 11.47% | 1.052x | 0.969x | -2.66% | ACC诊断上限 |
| **0** | **2/2/4** | **2.109** | **36.95%** | **1.133x** | **1.008x** | **+0.48%** | **PASS** |

结论：生产真实性能先固定 `2/2/4`；`3/4/16`仅用于回答“Draft理论接受能力有没有提高”，不能作为上线配置。

## 6. 当前正式500条结果

### 6.1 I8K主线

| Split | ACC | 接受率 | TPOT | 请求吞吐 | 输出吞吐 | E2E | TTFT | 结果目录 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Val T0 | 2.109 | 36.95% | 1.133x | 1.005x | 1.008x | +0.48% | -5.45% | `benchmark_results/lingbao-eagle3-v2-all-i8k-e5-eval-s2k2d4-t00/validation` |
| Train T0 | 2.146 | 38.17% | 1.165x | 1.010x | 1.027x | +0.97% | -5.14% | `benchmark_results/lingbao-v2-e5-train-t00-s2k2d4-v1/validation` |

Train只比Val高约0.037（约1.75%）。训练数据记忆不是当前主要矛盾。

### 6.2 I4K单变量消融

| Split | I8K ACC | I4K ACC | I8K TPOT | I4K TPOT | I8K输出 | I4K输出 | I8K E2E | I4K E2E |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Val T0 | 2.109 | 2.077 | 1.133x | 1.118x | 1.008x | 0.997x | +0.48% | -0.47% |
| Train T0 | 2.146 | 2.101 | 1.165x | 1.143x | 1.027x | 1.015x | +0.97% | +0.09% |

I4K的Val ACC下降约1.5%，接受率下降约0.97个百分点，正式Val由PASS退为FAIL。更小MLP虽减少单次Draft成本，但正确前缀减少导致Target验证次数上升，净收益变差。

结果目录：

- Val：`benchmark_results/lingbao-eagle3-v2-all-i4k-e5-eval-s2k2d4-t00-val-v1/validation`
- Train：`benchmark_results/lingbao-eagle3-v2-all-i4k-e5-eval-s2k2d4-t00-train-v1/validation`

## 7. 1500条请求级诊断

### 7.1 全局

| Split | N | ACC | 接受率 | TPOT | E2E | TTFT | 墙钟输出吞吐 | 解释 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Val T0 | 1500 | 2.126 | 37.57% | 1.149x | +0.66% | -5.64% | 0.937x | 诊断等待导致，不验收 |
| Train T0 | 1500 | 2.148 | 38.31% | 1.160x | +1.22% | -4.82% | 0.949x | 诊断等待导致，不验收 |

结果目录：

- Val：`benchmark_results/lingbao-v2-e5-t00-s2k2d4-requestdiag-v1/validation`
- Train：`benchmark_results/lingbao-v2-e5-train-t00-s2k2d4-requestdiag-gpu67-v1/validation`

### 7.2 Validation任务族（all_pairs）

| 任务族 | N | ACC | TPOT | 输出 | E2E |
|---|---:|---:|---:|---:|---:|
| 信息处理 | 146 | 2.408 | 1.278x | 1.028x | +1.52% |
| 决策辅助 | 380 | 2.336 | 1.262x | 1.047x | +4.49% |
| 对局闲聊 | 776 | 2.092 | 1.101x | 0.995x | -0.91% |
| 知识问答 | 198 | 2.024 | 1.056x | 0.996x | -2.43% |

### 7.3 Validation同输出文本（same_output_sha256）

| 任务族 | N | ACC | TPOT | 输出 | E2E |
|---|---:|---:|---:|---:|---:|
| 信息处理 | 122 | 2.438 | 1.308x | 1.021x | +2.04% |
| 决策辅助 | 164 | 2.358 | 1.273x | 1.049x | +4.69% |
| 对局闲聊 | 377 | 2.151 | 1.135x | 1.001x | +0.06% |
| 知识问答 | 88 | 2.064 | 1.082x | 0.989x | -1.07% |

### 7.4 Train任务族（all_pairs）

| 任务族 | N | ACC | 请求ACC P50 | 请求ACC P90 | TPOT | 输出 | E2E |
|---|---:|---:|---:|---:|---:|---:|---:|
| 信息处理 | 134 | 2.399 | 2.414 | 2.710 | 1.286x | 1.033x | +1.98% |
| 决策辅助 | 382 | 2.352 | 2.364 | 2.667 | 1.270x | 1.058x | +4.85% |
| 对局闲聊 | 744 | 2.103 | 2.111 | 2.571 | 1.106x | 1.010x | -0.57% |
| 知识问答 | 240 | 2.119 | 2.125 | 2.556 | 1.115x | 1.020x | -0.53% |

### 7.5 归因

- 信息处理与决策辅助通常有更稳定、受盘面约束的答案前缀，Target熵较低，Draft更容易连续命中；决策辅助是当前最明确的业务收益族。
- 闲聊表达自由度高，第一、第二Token有多种合理说法；即使TPOT改善，短输出和TTFT仍使E2E临界。
- 知识问答既要求事实Token准确，又可能产生多种句式，当前ACC最低、Val E2E回退最大。
- Train/Val每族ACC都接近，优先做“低接受且高频”的难例加权/蒸馏，而不是简单增加Epoch。

## 8. 数据和模型资产

### 8.1 Data V2

| 数据 | Train | Val | Test | 合计 |
|---|---:|---:|---:|---:|
| 全任务 | 77,384 | 7,827 | 14,775 | 99,986 |
| Chat-only | 39,657 | 4,011 | 7,572 | 51,240 |

Data V2从100,000条原始请求中删除14条，仅做结构异常、精确去重和控制字符规范化，没有改写业务模板。全任务分布：闲聊51.25%、决策26.24%、知识13.95%、信息8.56%。

### 8.2 当前主线路径

```text
Target:
/data/home/leonardoqin/models/lingbao-response

Draft (I8K E5):
/data/home/leonardoqin/models/exports/lingbao-eagle3-v2-all-i8k-oldopt-e5-v1-sglang

Draft (I4K E5 ablation):
/data/home/leonardoqin/models/exports/lingbao-eagle3-v2-all-i4k-oldopt-e5-v1-sglang

Data V2 requests:
/data/home/leonardoqin/datasets/lingbao_eagle3_data/data_version2

Pretokenized:
/data/home/leonardoqin/datasets/lingbao_eagle3_data/pretokenized_v2/train_eagle3_data_v2_all_tokens_10240.jsonl

Vocab mapping:
/data/home/leonardoqin/datasets/lingbao_eagle3_data/cache/vocab_mapping/qwen3.5-35b-a3b-eagle3-data-v2-all-v1.pt

Manifest:
/data/home/leonardoqin/datasets/lingbao_eagle3_data/data_version2/split_manifest.jsonl
```

32K mapping对当前预分词训练数据的监督Token、可见回答及前4Token覆盖均为100%，继续扩大mapping不是当前优先方向。

## 9. 实验注册表

机器可读台账：`experiments/registry.json`。当前登记16组关键实验，记录日期、模型、数据、Split、温度、树、结果目录、Draft和结论。生成汇总：

```bash
cd /data/home/leonardoqin/Eagle3-lingbao
bash scripts/07_postprocess.sh ledger
```

线上结果存在时读取真实JSON；迁移机器后缺失时保留登记观测值并标记来源。新增实验必须先创建唯一 `EXPERIMENT_NAME`，完成后再登记，禁止覆盖历史报告。

## 10. 当前推荐配置

```text
模型：lingbao-eagle3-v2-all-i8k-oldopt-e5-v1
训练数据：Data V2全任务Train 77,384
训练轨迹：T=0.9
Draft：1层，MLP 8192，ttt_length=7
Hidden streams：当前固定3层选择
优化器：LR=1e-4，accumulation=1，AdamW beta2=.999
服务树：2/2/4
服务温度：业务允许时T=0；T=0.9需单独验收
Formal：request diagnostics关闭
Diagnostic：c1、interval1、单独实验名
```

## 11. 下一步优先级

### P0：稳住正式收益

1. 用冻结I8K E5 + T0 + 2/2/4补并发2/4/8 Formal，确认c1收益是否可扩展。
2. 固定配置完成Test盲测；Test只运行一次，不用于调参。
3. 重复c1正式轮2～3次，报告均值与抖动区间，因为当前输出收益仅+0.8%、E2E仅+0.48%。

### P1：提高ACC而不扩大树

1. 用请求级诊断筛选“首/次Token拒绝、高频、Target低熵”的闲聊和知识问答难例，提高训练采样权重。
2. 对2/2/4服务目标做前位置加权Loss或缩短TTT消融，重点保证第1、2步，不把算力平均花在第4～7步。
3. 做Target soft-logit/KL蒸馏小实验，优先监督前2～3Token和32K映射空间的TopK分布。
4. Hidden stream层选择、RMSNorm/Post-Norm各自单变量验证；不要与数据/MLP/优化器同时改变。

### P2：数据策略

1. 保持全任务训练，不继续盲目缩成Chat-only。
2. 不是随机多Seed扩充；只对Target高熵且线上确有多表达的样本生成额外轨迹。
3. 任务路由只有在每个子模型都能在相同请求集上超过全任务Draft后才进入工程化。

## 12. 最终判断

```text
旧数据 + 大树：有接受能力，但开销无法回本
            ↓
Data V2全任务：提高了Draft覆盖与连续命中
            ↓
T=0：降低Target轨迹熵
            ↓
2/2/4：把候选树开销压到短输出可摊销范围
            ↓
I8K E5首次取得正式Validation正收益
            ↓
I4K/Chat-only消融说明：继续减容量或缩数据会损伤命中
            ↓
下一阶段应在不扩大树的前提下，针对闲聊/知识问答前2Token做难例与蒸馏
```

复现命令与后处理方法见 [POSTPROCESSING.md](POSTPROCESSING.md)。
