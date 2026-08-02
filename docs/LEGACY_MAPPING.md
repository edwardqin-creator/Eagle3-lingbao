# 原 30+ 脚本如何收敛到 5 个入口

`scripts.tar.gz` 中的脚本不是全部错误，而是同一流程在排障过程中不断生成了 smoke、修复版和一次性包装。这个仓库保留能力，删除日常入口的重复。

| 旧脚本/范围 | 新入口 | 处理方式 |
|---|---|---|
| `01_prepare_requests.py`, `02_token_stats.py` | `01_prepare_data.py` | 合并解析、保守清洗、去重、切分和报告 |
| `03_replay.py`, `04_start_4_tp2.sh`, `05_run_4_replay.sh` | `02_replay.sh` | 多服务生命周期、断点重放和校验成为一个入口 |
| `06_quick_validate.py`, `07_full_token_stats.py` | `02_replay.sh validate` + Step 3 报告 | ID/结构由重放校验，真实 token 统计由预分词报告负责 |
| `08_build_vocab_mapping.py`, `09_prepare_tokenized_prompts.py`, `28/31/33_*` | `03_prepare_training.sh` | 固定逐条预处理、可解释 loss mask、mapping 与硬校验 |
| `08_environment_preflight.sh` | `04_train.sh doctor` | 版本、导入路径、capture patch、Mooncake、GPU、端口集中检查 |
| 多个手写 YAML/启动命令 | `04_train.sh render/plan/run` | 中央 `.env` + 一份模板；渲染 YAML 仍完整保留便于审计 |
| `10_test_eagle3_real_prompt.py` | `05_evaluate.sh all smoke` | 单请求改为固定 8 条全链路 smoke |
| `11/12_start_*`, `13_benchmark_*`, `14_compare_*`, `15_run_*` | `05_evaluate.sh` | Baseline/EAGLE3 同卡顺序启动、流式指标、比较与 Gate |
| `17`～`27` Data-Juicer/Stage2/重切分实验 | Step 1 的可选上游 | 不再耦合进每轮训练；产出 request JSONL 后交给 `01 --input-format request-jsonl` |
| DFlash 专用脚本 | 不放进 EAGLE3 主入口 | DFlash 的 block/objective/export 不同，避免一个脚本用大量条件分支混跑两种算法 |

这样做之后，用户需要理解的仍是完整流程，而不是 30 个文件名。`tools/` 中的内部模块可以单独测试，但不构成新的操作步骤。
