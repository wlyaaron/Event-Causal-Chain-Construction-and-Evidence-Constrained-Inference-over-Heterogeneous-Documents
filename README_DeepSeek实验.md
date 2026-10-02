# 赛题 4：DeepSeek API 实验基线

这个脚本读取本仓库的 `数据集/抽样测试集_100`，逐题向兼容 Chat Completions 的 API 发送**原始文档、问题，以及该档位实际提供的事件/因果边**，生成赛事要求的单个 JSON 数组。输入材料会发送到你填写的 API 服务。API URL、模型名和 Key 由你运行时输入；Key 不写入仓库或输出文件。只需 Python 3.10+，无需安装第三方包。

## 先看两点

1. **赛事资格**：[提交要求](docs/赛事网页内容/04_提交要求.md)写明禁止直接用非开源模型预测，且对闭源 API 模型有限制。这个程序先用于研究、调试和对照实验；**不要在未向组委会确认许可与复现方式前，直接把其预测作为正式参赛作品提交**。
2. **实际数据与页面说明有差异**：仓库中的抽样测试集共有 A 30 个主题 / 210 题、B 50 个主题 / 350 题、C 20 个主题 / 200 题。A 有事件列表和因果边；B、C 只有文档和问题，没有事件列表或因果边。程序按磁盘上的实际文件处理，B/C 使用文档 ID（如 `D001`）作为候选证据 ID。**这 760 题是仓库中的抽样测试集；正式评测范围以平台为准。**

## 在 Windows PowerShell 中运行

在仓库根目录执行：

```powershell
python run_deepseek.py --dry-run --track A --limit 3
python run_deepseek.py --track A --limit 3 --output outputs/a_preview.json
```

第二条命令会提示输入 API URL、模型名和 Key。URL 可填基础地址（例如 `https://api.deepseek.com`）或完整的 `/chat/completions` 地址；模型名请填你账户实际可用的名称（例如 `deepseek-v4-pro`）。Key 输入时不回显。也可以用 `--api-url`、`--model`，或环境变量 `DEEPSEEK_API_URL`、`DEEPSEEK_MODEL`、`DEEPSEEK_API_KEY` 设置，但**不要将 Key 写进脚本、命令行参数、README 或 Git 提交**。

官方 API 文档：[接入说明](https://api-docs.deepseek.com/)、[Chat Completions](https://api-docs.deepseek.com/api/create-chat-completion/)、[模型更新记录](https://api-docs.deepseek.com/updates/)。当前官方文档接受 `deepseek-v4-pro`；旧的 `deepseek-v4-flash` 名称虽仍可用，但已转到更新的 Flash 模型。若要复现赛事表中 `DeepSeek-V4-Flash` 基线，必须先确认服务端实际模型版本。

先分别试 A/B/C，再考虑全量：

```powershell
python run_deepseek.py --track B --limit 3 --output outputs/b_preview.json
python run_deepseek.py --track C --limit 3 --output outputs/c_preview.json
python run_deepseek.py --track all --limit 0 --output outputs/sample_test_760.json
python run_deepseek.py --track all --limit 0 --validate outputs/sample_test_760.json
```

默认只处理 3 题，`--limit 0` 才处理所选范围内全部问题。每完成一题会把结果存到同名 `.partial` 文件；网络或格式错误时，修复后重复原命令即可续跑，不会重问已经保存的问题。`--overwrite` 会丢弃现有输出重新计算，可能重复产生 API 费用。

输出字段为 `sample_id`、`answer`、`evidence_chain`、`confidence`、`question_type`。证据 ID 必须来自当前材料；拒答时写 `"无法确定"`、空链和 `null`。校验命令检查字段、ID 与选中问题的覆盖率，**不能代替平台评分或人工核对答案质量**。A 档若链上相邻事件不在给定边中，程序只提示人工复核，因为训练标注里也存在未由给定边直接连接的可接受链。

实验输出目录已在 `.gitignore` 中忽略。程序不读取 `gold/问答对_答案.json`，标准答案不会发到 API。

## 离线自测

```powershell
python -m unittest -v test_run_deepseek.py
```

测试使用本机模拟 API，不需要真实 Key，也不会调用 DeepSeek。
