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

历史 EAGLE3 从 top-k=1/draft=4 的约 1.58，提高到 top-k=4/draft=16 的约 1.94；Train 与 Validation 接近，说明主要矛盾不是普通数据集过拟合。与此同时部分配置 TPOT/E2E 仍回退，说明 1.94 尚不足以覆盖更宽候选树的系统开销。因此下一轮必须把“训练质量”和“serving 搜索”拆开比较。
