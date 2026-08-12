# Validation、调参和 Test 冻结规则

## 1. 三个集合各自只做一件事

- Train：重放 Target 答案，训练 Draft 权重。
- Validation：选 checkpoint、epoch 和 serving 参数。
- Test：只在方案冻结后做一次最终结论。

三者必须在清洗/近重复去重之后再切分。任何跨集合 Prompt 泄漏都会让接受长度显得虚高。

## 2. 每个候选实验必须记录

```text
数据版本与 SHA256
Target checkpoint
Draft checkpoint
Draft config（MLP、aux layers、vocab mapping）
训练 epochs、LR、ttt_length、world size
SGLang/SpecForge commit 与本地 patch
steps/top-k/draft-tokens
GPU、TP、并发、样本 seed、采样参数
```

`configs/lingbao.env`、渲染后的训练 YAML、Plan 日志和 `ACCEPTANCE_REPORT.json` 一起构成一轮完整实验记录。

## 3. 现象 → 判断 → 下一步

| 现象 | 判断 | 下一步 |
|---|---|---|
| Train/Val 接受长度都低且接近 | 不是经典过拟合；Draft 容量、目标、优化或服务树配置受限 | 先搜 serving 参数，再比较 epoch/checkpoint；必要时增加容量或挖 hard cases |
| Train 高、Val 低 | 分布偏移或过拟合 | 去重、扩大覆盖、减少 epoch/容量 |
| 接受长度升高但 TPOT/吞吐下降 | 建树/验证额外开销超过节省 | 降 top-k、draft tokens 或 steps |
| TPOT 改善但 E2E 不改善 | 输出过短，TTFT/Prefill 占主导 | 固定输出长度做 decode 微基准，同时保留业务 E2E 测试 |
| 并发 1 加速、并发 8 回退 | Draft/验证 kernel 或调度不适合高并发 | 按生产并发决策，不能只看 c1 |
| 接受长度日志为空 | server log offset/路径错误，或 EAGLE3 未真正启用 | 检查启动参数和日志中的 speculative algorithm |

## 4. 两类测试都需要

业务测试保留 EOS 和原始采样参数，回答真实 E2E。Decode 微基准使用固定输出长度与 `ignore_eos`，隔离短回答对 TPOT 的噪声。后者不能代替业务结果。

## 5. 当前历史结果的正确结论

截至 2026-08-12，Data V2 全任务 I8K OldOpt E5 在 Validation T=0、`2/2/4`、并发1的正式500条实验中达到接受长度 `2.109`、TPOT `1.133x`、输出吞吐 `1.008x`、E2E改善 `+0.48%`，是当前主线。I4K同口径退化为 `2.077/1.118x/0.997x/-0.47%`；Chat-only专项模型退化为 `1.907/1.027x/0.972x/-3.54%`。

Train/Validation 1500条请求诊断 ACC 分别约 `2.148/2.126`，差值很小，不支持明显过拟合。任务族结果显示信息处理、决策辅助有稳定正收益，闲聊、知识问答仍是主要瓶颈。完整历史和路径见 [EXPERIMENT_OVERVIEW.md](EXPERIMENT_OVERVIEW.md)，复现方法见 [POSTPROCESSING.md](POSTPROCESSING.md)。

## 6. Formal 与 Request Diagnostics

`EVAL_REQUEST_DIAGNOSTICS=1` 会把每条请求对应的 SGLang decode 日志解析进 details；它要求并发1和 `EVAL_DECODE_LOG_INTERVAL=1`。客户端在请求计时结束后等待20ms日志刷新，因此逐请求 TTFT/TPOT/E2E 不直接包含等待，但整轮 `duration_s` 和墙钟吞吐包含等待。诊断结果用于相关性、任务族和难例筛选，最终吞吐必须用 `EVAL_REQUEST_DIAGNOSTICS=0` 重新正式测试。
