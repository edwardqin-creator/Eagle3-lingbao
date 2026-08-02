# Lingbao + SpecForge EAGLE3 全量 Debug 记录

下面按“症状—原因—修复—验证”记录本项目实际遇到的问题。先定位第一条真实异常，不要把后续清理错误当成新的训练错误。

## 1. Docker、网络和环境

### Docker tag/push 名称不一致

- 症状：`image does not exist locally with the tag` 或 `tag does not exist`。
- 原因：`docker tag` 的目标 tag 与随后 `docker push` 使用的 tag 不同，或把反斜杠和参数写在同一行造成多余参数。
- 修复：先 `docker image ls` 确认源名，再严格执行 `docker tag SOURCE TARGET`，最后 push 完全相同的 TARGET；凭证用 `--password-stdin`，不要写进 shell history。

### 65GB 模型上传命令失败/看不到进度

- 症状：`curl -F` 字段格式错误、旧 curl 不认识 `--progress-meter`，或 Python 服务一次 `read()` 整个文件。
- 修复：表单必须写成 `-F 'file=@archive.tar'`；旧 curl 默认显示进度，也可用 `-#`；超大文件优先对象存储/rsync，临时 HTTP 服务必须流式分块写盘，不能把 65GB 读入内存。

### Docker daemon permission denied

- 症状：`permission denied while trying to connect to /var/run/docker.sock`。
- 原因：用户不在 docker 组或 daemon 只允许 root。
- 修复：使用已授权的 `sudo docker ...`，或由管理员加入 docker 组；不要为了省事暴露 daemon TCP。
- 验证：`docker info`、容器内 `nvidia-smi`。

### 只通内网，镜像/PyPI/TCR 超时

- 症状：TCR login deadline、PyPI wheel 下载超时。
- 原因：节点无公网出口或镜像源没有 CUDA13 包。
- 修复：在有公网的同架构 Python 3.11 Linux 节点下载 wheelhouse，校验后通过内网上传；内网用 `uv pip install --offline --no-deps wheel.whl`。
- 验证：导入包、打印 `.so` 路径和版本。

### uv venv 没有 pip

- 症状：`python -m pip` / `pip3` 都不存在。
- 原因：`uv venv` 可以不安装 pip 模块。
- 修复：使用 `uv pip install --python .venv/bin/python ...` 和 `uv pip check --python ...`。

### NUMA affinity warning

- 症状：SGLang 提示没有权限设置 NUMA affinity。
- 原因：容器缺 `SYS_NICE` 或宿主限制。
- 影响：主要是 CPU 调度/内存局部性噪声，不会让投机解码算法失效。
- 修复：允许时容器增加 `--cap-add SYS_NICE`；A/B 必须保持相同容器条件。

## 2. 数据解析、重放和预分词

### 把 seed 当作必填字段

- 症状：`ValueError: 缺少字段 ['seed']`。
- 原因：生产请求有的没有 seed，脚本错误地强制要求。
- 修复：有 seed 原样传递；没有就省略。评测 A/B 为了可比性，缺失 seed 时才用 ID 生成稳定 seed。

### 长度统计很慢或看似卡住

- 原因：38K 条、每条约 5K token，Tokenizer 总工作量很大；旧脚本没有进度。
- 修复：真实 tokenizer、批量/多进程或逐 100 条打印速度。历史结果完整序列 max=9329，因此训练 `max_length=10240`、服务 `context_length=16384`。

### 重放 32 条一直不打印

- 原因：并发请求仍在生成，旧脚本只在完成阈值时打印；服务端动态 batching 也不会逐请求刷日志。
- 修复：完成计数实时输出；多个 TP2 副本并行；支持按 ID 续跑。

### 端口 30001 已被占用

- 症状：`Errno 98 address already in use`。
- 原因：旧服务或其他业务占用通用端口。
- 修复：启动前 `ss -ltnp 'sport = :PORT'`，统一使用 33xxx 实验端口；PID 文件只作线索，端口/进程状态才是真相。

### 30,000 条预处理后只剩 32 条

- 症状：Data map 完成，但 `有效训练记录数: 32`。
- 原因：把整个数据集作为一个 batch 传给期望逐样本结构的预处理函数，batch 维度被错误解释。
- 修复：逐条调用 `preprocess_conversations(conversations=[text])`，最终硬校验 input/output ID 集和期望记录数。

### loss_mask 监督范围不清楚

- 原理：`input_ids` 是 Prompt + Target answer + EOS；Prompt 对应 mask=0，answer/EOS 对应 mask=1。只有 mask=1 参与 Draft loss。
- 修复：使用 `target-faithful`，按原 request.prompt 的 token 前缀精确切边界；运行 `03_prepare_training.sh inspect` 人工解码。
- 注意：历史 `train_only_last_turn` 可能连 `</think>` 一起监督；切换策略属于数据版本变化，必须新建实验名。

## 3. Qwen3.5 Target 与 SGLang Capture 兼容

### Capture logits 一维导致 IndexError

- 症状：`IndexError: too many indices for tensor of dimension 1`，位置在 logits buffer copy。
- 原因：Qwen3.5 的 capture 路径返回形状与普通 Llama 假设不同。
- 修复：应用 SpecForge 针对 SGLang 0.5.14 的 `spec-capture.patch`，确认 `sglang.srt.spec_capture_sink` 存在。
- 验证：`04_train.sh doctor`，再跑 8-step smoke。

### TargetHead 读取顶层 Llama 字段失败

- 原因：Qwen3.5 MoE/VL config 把语言模型字段放在 `text_config`，普通 Llama 在顶层。
- 修复：TargetHead 配置读取使用 `getattr(config, 'text_config', config)`；这不是微调权重问题，也不是 MoE 路由本身导致。

### Source SpecForge 与 site-packages 混用

- 症状：明明改过源码，Plan 仍报旧 capability 错误。
- 原因：`specforge` 命令/子进程导入 `.venv/site-packages/specforge`，没有使用 `/workspace/SpecForge/specforge`。
- 修复：所有入口统一 `PYTHONPATH=/workspace/SpecForge`；最好对同一源码执行 editable install；doctor 断言 `specforge.__file__` 位于源码目录。

## 4. 长序列 Attention 和教师投影显存

### FlexAttention / Triton 无合法配置

- 症状：`No valid triton configs`，`Required: 114688`，`Hardware limit: 101376`。
- 原因：当前 SM120 Blackwell 上生成的 backward kernel 共享内存需求超过硬件限制。
- 修复：本项目改用 FlashAttention (`attention_backend: fa`)；先做 10,240 token BF16 forward/backward smoke。不是把 Flex 的 Triton 后端替换成 FlashInfer——训练 Attention backend 与 SGLang 推理 FlashInfer 是不同层次。

### SDPA 长序列显存爆掉

- 原因：标准 attention 中间量随序列长度二次增长；短 32 条 smoke 没覆盖最坏长度。
- 修复：FlashAttention；smoke 必须专门包含最长 32 条，而不是随机 32 条。

### 为什么日志里出现 FP32，能否改 BF16

- 原因：教师 full-vocab logits/soft target 的部分计算为数值稳定使用 FP32；长序列 × 248K vocab 形成巨大临时张量。
- 正确修复：分块计算 compact teacher projection，保持每块 FP32 数值语义，不是粗暴把教师概率改 BF16。
- 风险：当前上游文档把 compact teacher 限定为 offline text EAGLE3。本项目在线成功依赖本地 capability 补丁，换环境必须重新 Plan/smoke。

### `algorithm eagle3 does not support compact teacher for streaming`

- 原因 A：使用上游默认能力门；原因 B：又导入 site-packages 旧代码。
- 历史本地修复：允许 `mode=streaming && algorithm=eagle3` 的自定义路径，并确保源码优先导入。
- 验证：不能只改 YAML；必须 `specforge train --plan` 成功且最长样本 smoke 成功。

## 5. Mooncake 与进程生命周期

### Mooncake 缺失

- 症状：`ModuleNotFoundError: mooncake`，找不到 `mooncake_master`。
- 修复：安装匹配 CPython 3.11、manylinux x86_64、CUDA13 的 wheel；doctor 同时检查 Python extension 和 executable。

### managed_local port unavailable

- 原因：上一次异常退出残留服务，或多个实验复用了 Mooncake RPC/metadata/metrics/capture 端口。
- 修复：为每个实验分配唯一端口；启动前逐端口检查；确认 PID/命令后优雅终止。

### `<defunct>` mooncake_master 杀不掉

- 原因：僵尸已经退出，只剩父进程未 wait 的进程表项；`kill` 对它无效，也不占 GPU/端口。
- 修复：容器用 `--init`/tini；launcher 在 finally 中 wait/reap child。不要把僵尸当 GPU OOM 根因。

### 训练成功后出现清理连锁异常

- 判断：先看最后 checkpoint、训练退出码和第一条 traceback。Mooncake drain/consumer cleanup 可能发生在训练已完成之后。
- 修复：修 launcher 的 shutdown 顺序与 grace period；不能因为最后一屏有清理报错就否定已经保存的 checkpoint，也不能忽略非零退出码。

## 6. Checkpoint、导出与服务

### `training_state.pt` 存在，能否直接推理

- 不能。它包含 Draft state、global step、epoch 等训练状态。
- 修复：`04_train.sh export` 调用官方 exporter，得到 SGLang 目录。

### 服务启动但 Draft 权重未真正加载

- 症状：日志有 skipped/missing keys，接受长度长期为 1。
- 修复：检查 Draft config/aux layers/vocab mapping 与训练完全一致；查看加载日志；真实请求必须出现 `accept len`。

### PID 文件指向 zombie

- 症状：脚本说服务存在，`ps` 显示 `<defunct>`。
- 原因：只使用 `kill -0` 判断，zombie 也可能返回存在。
- 修复：同时看 `ps -o stat` 和端口健康检查；新评测脚本以 `/health` 为准。

## 7. 指标误读

### 7 个 acceptance_rate

- 原因：训练 `ttt_length=7`，分别统计未来第 1～7 个位置的命中/接受估计；不是 7 个模型。

### `draft_tokens=16` 为什么接受率只有 6%，接受长度却约 1.94

- Draft tokens 是候选树预算，不是线性承诺一次产生 16 个有效 token；接受率的分母与树候选相关，接受长度表示每次验证真正推进的 token。两者不能直接相除。

### 接受长度提高但速度下降

- 原因：top-k/候选 token 增大也增加 Draft、建树、验证和调度成本。只有节省的 Target decode 成本超过额外开销才加速。
- 修复：同卡 Baseline A/B，同时看接受长度、TPOT、输出吞吐和 E2E；Validation 选配置，Test 最终验收。

## 8. 一条标准排障顺序

```text
第一条 traceback
→ Python/SpecForge/SGLang 实际导入路径与版本
→ 数据行数、ID、loss_mask 和最长样本
→ GPU/端口/残留进程
→ Plan
→ 最长样本 smoke
→ checkpoint 与退出码
→ export 加载日志
→ 8 条推理 smoke
→ 500 条同条件 A/B
```

不要同时修改数据、模型宽度、epoch 和 serving 树参数；否则即使结果改善，也无法知道是哪一个因素造成的。
