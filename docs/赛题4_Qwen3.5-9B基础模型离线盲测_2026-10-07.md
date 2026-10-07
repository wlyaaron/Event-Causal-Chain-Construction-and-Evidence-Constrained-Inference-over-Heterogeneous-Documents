# 赛题 4：Qwen3.5-9B 基础模型离线盲测记录（2026-10-07）

## 代码与输入来源

- 服务器仓库在运行时为 `main` 的 `50e81d8895f6764a2ec9badffaaaaed110999d84`，与当时 GitHub `main` 一致，已有未提交的微调准备文件原样保留。该提交包含 `9641671d` 的 A 档问题条件化路径索引及 760 题 DeepSeek 研究复跑。A 用 `experiments.run_full_causal_path_research.input_for()` 的优化输入，B/C 用 `task4_core.build_input()` 原输入。
- 仅新增离线调用适配器 `experiments/run_local_qwen35_blind.py`，提交为 `32d09c73`（分支 `task4-server-sft-20261006`）。它复用冻结的输入函数、`task4_core.parse_prediction()` 和 `PredictionStore`，没有调用 API，也不读取训练 gold 或托管 API 输出。服务器只复制了该文件，主线其他代码未改。
- 本地权重为官方 `Qwen/Qwen3.5-9B` 基础模型，先前下载记录的版本为 `460979c3d11864dd16408d860ac930a360a2fac2`。本次使用文件的 `config.json` SHA-256 为 `d0883072e01861ed0b2d47be3c16c36a8e81c224c7ffaa310c6558fb3f932b05`，`tokenizer.json` 为 `5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42`。没有加载 LoRA 或进行微调。
- 单张 RTX 4090（49140 MiB），PyTorch 2.6.0、Transformers 5.6.0、flash-linear-attention/fla-core 0.4.2、causal-conv1d 1.6.2.post1。BF16、SDPA、单题串行、`do_sample=False`、`enable_thinking=False`、最多输出 1536 token、总上下文上限 16384 token。Qwen 官方[模型说明](https://huggingface.co/Qwen/Qwen3.5-9B)列出非思考模式和专用推理引擎；这里选确定性 Transformers 基线，便于先验证流程，吞吐量不能代表最优服务部署。

## 命令与完整性

```bash
cd /root/autodl-tmp/event-causal-chain-repo
/root/autodl-tmp/task4-qwen35-venv/bin/python -m experiments.run_local_qwen35_blind \
  --model-dir outputs/qwen3.5-9b \
  --output outputs/qwen35_base_causal_path_760.json
```

先用 `--dry-run` 和同一 tokenizer 审计了全部 760 题：A/B/C 为 210/350/200；提示平均 5048.7 token，最长 7343 token。加上 1536 输出预算仍低于 16384，未截断任何材料。3 题冒烟后重新从零运行完整 760 题；正式 JSON 通过 `task4_core.validate_file()` 的样本覆盖、字段、拒答和证据 ID 校验。最终输出 SHA-256：`34bfb28993c8ffa5065635741061cba42e528920be499e4a029bf52e7e0ff2d0`。预测、逐题耗时和错误记录在 Git 忽略的 `outputs/`，并复制回本地工作机；模型权重、研究 API 输出和预测均未提交 Git。GPU 推理进程已退出，未向赛事平台提交。

## 无 gold 诊断结果

| 档位 | 题数 | 精确拒答 | 平均答案字数 | 平均链节点 | 本地模型计算时间 | 首轮格式重试 | 两轮失败后格式回退 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A | 210 | 70 | 143.44 | 2.452 | 1398.9 秒 | 18 | 0 |
| B | 350 | 196 | 84.27 | 1.966 | 1836.2 秒 | 47 | 6 |
| C | 200 | 126 | 75.85 | 1.620 | 956.0 秒 | 18 | 3 |
| 合计 | 760 | 392 | 98.41 | 2.009 | 4191.15 秒（约 70 分钟） | 83 | 9 |

没有输出达到 1536 token 上限。A 档生成链中，相邻节点可在给定边表直接找到的比例为 88.53%；38 条非空链至少有一段不在给定边表。赛事训练 gold 也有缺边，故这些不是自动判错。给定的 `question_type=unanswerable` 共 133 题，其中模型精确拒答 126 题；其他题型有 266 题精确拒答。这只是可见输入类别与输出行为的交叉统计，不等于正确拒答率，不能据题型强制拒答。

与 2026-10-06 同流程 DeepSeek 研究输出比较（旧文件 SHA-256 `124e52d92e76df261e48c3ded41e01356ee849996cd04a917fb4234a32265d3c`）：

| 档位 | DeepSeek → 本地 Qwen 拒答 | DeepSeek → 本地 Qwen 平均链节点 | 其他无 gold 差异 |
| --- | ---: | ---: | --- |
| A | 17 → 70 | 3.600 → 2.452 | 给定边直接支持比例 97.87% → 88.53% |
| B | 79 → 196 | 2.509 → 1.966 | — |
| C | 56 → 126 | 2.370 → 1.620 | — |

原 `deepseek-flash` 别名当时实际指向 V4.1 Flash，并非赛事表中的原版 V4 Flash。两种模型、解码与服务实现不同；这些对照只说明**未微调 Qwen 输出明显更保守、链更短**，不能证明真实答案或链的正确率下降，也不能推算赛事分数。760 题没有 gold，不用于调参、人工标注或伪标签。两轮格式失败的 9 题被运行器按旧流程回退为精确拒答；它们可能包含本可回答的题，应作为负结果保留。数值 confidence 是基础模型的原始输出，未经校准，不可解释为真实概率。

## 后续判断

这次实验已验证最新主线流程能在本地开源权重上完整离线运行，但基础 Qwen3.5-9B 的高拒答和短链行为使它不宜仅凭盲测结构统计直接当作已验证的正式方案。真正的答案字符 F1、语义、关键事实、证据节点/边 F1、链完全匹配和拒答正确率，需要在**固定包隔离的有 gold 验证/留出集**上计算；盲测集本身不能给出这些指标。单题串行 Transformers 推理约 70 分钟，而先前托管 API 使用并发服务；该时间差主要反映执行方式，不能归因于路径算法。
