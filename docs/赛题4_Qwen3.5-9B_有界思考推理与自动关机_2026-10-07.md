# 赛题 4：Qwen3.5-9B 有界思考推理与自动关机（2026-10-07）

## 来源和方法边界

- 模型仍是本地官方 `Qwen/Qwen3.5-9B` 权重，单张 RTX 4090（49140 MiB），不使用托管 API、API 预测或测试集答案。Qwen3.5 [官方模型卡](https://huggingface.co/Qwen/Qwen3.5-9B)说明它默认先生成 `<think>…</think>`，并推荐思考模式采样 `temperature=1.0, top_p=0.95, top_k=20`。运行器使用这三个生成参数；官方还建议的 `presence_penalty=1.5` 未在本 Transformers 路径实现，不应把本实验说成完整复制官方服务配置。
- 思考预算的两段生成来自 [Qwen3 官方教程](https://github.com/QwenLM/Qwen3/blob/main/docs/source/getting_started/quickstart.md#thinking-budget)：先生成有限的思考 token，未自行结束时插入提前结束提示和 `</think>`，再续写最终答案。**教程是 Qwen3 的，不是 Qwen3.5-9B 的官方保证**；应用到本模型是工程假设，必须以本题验证集实测为准。Qwen3.5 模型卡也明确不支持 Qwen3 的 `/think`、`/nothink` 软开关。当前实验保持聊天模板 `enable_thinking=True`，不使用软开关。
- `experiments/run_local_qwen35_thinking.py` 保留原 A 档问题条件化路径输入及 B/C 原输入，仅从生成结果的 `</think>` 后提取最终 JSON，并沿用 `task4_core.parse_prediction()` 和完整输出校验。首轮输出格式失败时，在同一原始输入上做一次直接格式修复；若仍失败，记录并回退为拒答。修复次数和回退次数写在忽略目录的进度文件中，不能算作纯思考答案。

## 子集核查和负结果

- 固定包隔离验证集的 4 个材料包、4 种题型，在 A/B/C 三视图共 12 题。子集仅使用赛事训练集材料和 gold。完整测试集 760 题没有 gold，没有拿来选择设置或人工标注。
- 自然思考上限 2048 token 的早期 3 题里有 2 题未生成 `</think>`；仅在系统提示中要求 800 token 内结束思考，另一组 3 题仍全部碰到 3072 token 上限。这两组负结果保存在忽略目录 `outputs/`，未用于正式输出。
- 采用 1024 token 的两段思考预算、总输出上限 2048、单题串行时，12 题全部形成有效最终 JSON；11 题需要插入提前结束提示，1 题触及最终输出上限并经一次格式修复，无二次失败后的强制拒答。答案字符 F1 代理均值 0.2904、证据节点 F1 0.6625、链完全匹配 0.4167。这是小型验证子集的本地代理，不能外推为整体验证集或盲测成绩。
- 针对触及上限的 `industrial_production_01_Q06/A` 单题，将总上限提高到 3072 后得到有效 JSON，生成 1215 token，没有碰到新上限；因此正式运行给最终答案保留 3072 token 总预算。
- 同 3 题的 batch 1 与 batch 2 分别耗时 132.08 秒／生成 3365 token 和 106.50 秒／生成 2688 token。因采样输出长度不同，按实际生成量计的吞吐分别约 25.5 和 25.2 token/秒；batch 2 未显示可信的吞吐改进。batch 2 观察时显存约 21.4 GiB，没有 OOM；正式运行仍选 batch 1，避免为不确定的收益增加批处理风险。
- 用同一 tokenizer 预审全部 760 道测试输入，最长提示 7341 token；即使给最终输出预留 3072 token，也未达到 16384 token 上下文上限，没有静默截断材料。测试输入只用于这个长度完整性检查，不用于调参。

## 代码和无人值守运行

- 推理脚本：`experiments/run_local_qwen35_thinking.py`；关机守护：`experiments/inference_shutdown_guard.py`；守护模拟测试：`test_inference_shutdown_guard.py`。
- 推理使用单份本地 BF16 模型、Transformers 5.6.0、SDPA、1024 token 思考预算、3072 token 总输出上限、batch 1。正式输出及进度均位于 Git 忽略的 `outputs/`；推理结束后用 `task4_core.validate_file()` 核查 760 题覆盖、字段和证据 ID。
- 守护进程先等推理进程退出。成功时要求最终 JSON 覆盖全部 760 题并校验进度元数据；失败时先写明退出码或校验错误。无论成功还是失败，都仅在 GPU 利用率不超过 5%、显存占用不超过 2048 MiB 且无 GPU 计算进程持续 10 分钟后，才请求专用 AutoDL 实例关机。GPU 查询异常或持续繁忙时不关机。模拟测试已验证成功、失败两条路径不会误执行真实关机命令。

```bash
cd /root/autodl-tmp/event-causal-chain-repo
nohup setsid /root/autodl-tmp/task4-qwen35-venv/bin/python \
  -m experiments.inference_shutdown_guard \
  --model-dir outputs/qwen3.5-9b \
  --output outputs/qwen35_thinking_budget1024_blind760.json \
  --batch-size 1 --thinking-budget-tokens 1024 --max-new-tokens 3072 \
  --arm-shutdown \
  > outputs/qwen35_thinking_budget1024_blind760.launch.log 2>&1 < /dev/null &
```

上面的命令是 **10 月 7 日历史运行命令**。当日 19:14（北京时间）推理完成并验证成功，随后用户要求保留服务器继续工作；约 19:21 已终止自动关机守护并检查进程退出，**不要重新运行带 `--arm-shutdown` 的命令**。守护状态：`outputs/qwen35_thinking_budget1024_blind760.guard.json`；运行日志：`outputs/qwen35_thinking_budget1024_blind760.guard.log`；逐题进度和诊断：`outputs/qwen35_thinking_budget1024_blind760.json.progress.json`。运行脚本本身不向赛事平台提交预测；用户随后对输出做了线上评测。输出不用于微调、蒸馏或伪标签。

## 全量完成与完整性核查

最终 JSON 独立通过 `run_deepseek.py --track all --limit 0 --validate`：A/B/C **210／350／200**，合计 **760**，检查题目覆盖、字段、拒答约束及证据 ID。文件 SHA-256 为 `bd2c02a550eef8f503a316b73a06d39b9183fe270f500fb1ddcd3c22bb58aca9`。最终 JSON、逐题进度和对照 JSON 只在 Git 忽略的 `outputs/` 中；测试 gold 不公开。10 月 7 日用户提供的线上截图给出本文件 A/B/C **61.86／43.44／45.20**、总分 **54.67**，未微调非思考版为 **45.92／33.14／42.19**、总分 **41.71**。

| 输出行为／成本 | 非思考 760 题 | 有界思考 760 题 |
| --- | ---: | ---: |
| 精确拒答 | 392（51.58%） | 191（25.13%） |
| 平均链节点 | 2.009 | 2.800 |
| 平均答案字符 | 98.41 | 149.17 |
| 生成 token | 110,003 | 920,788 |
| 模型生成时间 | 4,191.15 秒 | 27,213.59 秒 |
| 格式重试／二次失败回退 | 83／9 | 119／13 |

按 A/B/C 分档，思考模式拒答 **24／107／60**，非思考为 **70／196／126**；逐题配对有 **228** 题由拒答变回答、**27** 题反向变化。A/B/C 思考模式平均链节点分别 **3.248／2.671／2.555**，非思考为 **2.452／1.966／1.620**。两次链完全相同 **280／760**。思考模式生成 token 为非思考的 **8.37 倍**、模型生成时间为 **6.49 倍**；按记录的 27,213.59 秒约 **7 小时 33 分 34 秒**。其中 **741／760** 题需要注入提前结束标记，**4** 题触及总输出上限，**13** 题二次格式失败后回退拒答，都是应保留的负结果。

线上总分证实有界思考这一**端到端配置**高于非思考配置 **12.96** 分；仅凭无 gold 的逐题诊断仍无法判断哪些多出的回答、节点正确。非思考用确定性生成，思考用采样，不能把全部差值单独归于思考。DeepSeek 10 月 6 日研究复跑的线上分档为 **63.85／50.10／52.24**、总分 **58.56**；其原始逐题输出在原机器的忽略目录，新服务器没有可比的本地生成时间。四份输出的完整分档分析见[进度总览](赛题4_2026-10-02至10-07_研究与工程进展总览.md)第 3.9 节。
