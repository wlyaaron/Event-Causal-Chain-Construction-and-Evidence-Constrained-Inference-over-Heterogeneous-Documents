# 赛题 4：DeepSeek API 实验基线

这个脚本读取本仓库的 `数据集/抽样测试集_100`，逐题向兼容 Chat Completions 的服务发送**原始文档、问题，以及该档位实际提供的事件/因果边**，生成赛事要求的单个 JSON 数组。远程服务会收到输入材料。只需 Python 3.10+，无需安装第三方包。

## 先看两点

1. **赛事资格**：[规则要求和常见问题](docs/赛事网页内容/06_规则要求和常见问题.md)明确写着“不允许使用闭源API模型”“允许离线使用开源模型”；[提交要求](docs/赛事网页内容/04_提交要求.md)还要求禁止手工标注、算法可复现。官方托管的 DeepSeek API 调用仅作为调试、研究基线；**不要直接把该 API 生成的结果用于正式参赛提交**。若组委会明确书面允许某种托管服务，依其最新裁定执行。正式方案应离线运行开源权重，保存代码、权重、环境和复现步骤。
2. **实际数据与页面说明有差异**：仓库中的抽样测试集共有 A 30 个主题 / 210 题、B 50 个主题 / 350 题、C 20 个主题 / 200 题。A 有事件列表和因果边；B、C 只有文档和问题，没有事件列表或因果边。程序按磁盘上的实际文件处理，B/C 使用文档 ID（如 `D001`）作为候选证据 ID。**这 760 题是仓库中的抽样测试集；正式评测范围以平台为准。**

## 在 Windows PowerShell 中运行

在仓库根目录执行：

```powershell
python run_deepseek.py --dry-run --track A --limit 3
python run_deepseek.py --track A --limit 3 --output outputs/a_preview.json
```

第二条命令会提示输入 API URL、模型名和 Key。URL 可填基础地址（例如 `https://api.deepseek.com`）或完整的 `/chat/completions` 地址；模型名须是账户实际可用的名称，例如 `deepseek-flash`。Key 输入时不回显。也可以传 `--api-url`、`--model`、`--api-key-file`，或设置环境变量 `DEEPSEEK_API_URL`、`DEEPSEEK_MODEL`、`DEEPSEEK_API_KEY_FILE`、`DEEPSEEK_API_KEY`。**Key 文件只保存在本机；不要将其内容写进脚本、README、命令行参数或 Git 提交。**

已有 Key 文件时可直接运行，不用粘贴 Key：

```powershell
python run_deepseek.py --probe-api --api-url https://api.deepseek.com --model deepseek-flash --api-key-file "C:\path\to\key.txt"
python run_deepseek.py --track A --limit 1 --api-url https://api.deepseek.com --model deepseek-flash --api-key-file "C:\path\to\key.txt" --output outputs/a_preview.json
```

已用真实 A 档问题验证：`max_tokens=1200` 会因模型思考文本耗尽而截断，`8192` 能完成该样例。因此默认上限调整为 8192；若别的题仍被截断，可单独提高 `--max-tokens`，但会增加调用时间和费用。

如果调用返回 HTTP 400，先运行下面的诊断命令。它先检查官方 API Key 是否可用，再发送两条短测试消息，**不会发送数据集，也不会显示 Key 或账户金额**：

```powershell
python run_deepseek.py --probe-api --api-url https://api.deepseek.com --model deepseek-flash
```

诊断会依次区分认证、极简聊天请求和 JSON 输出模式。若基础请求成功而 JSON 模式失败，可加入 `--no-json-mode --overwrite` 试运行；脚本仍要求模型返回 JSON 并逐条校验。首次请求失败后留下的 `.partial.meta.json` 可以直接续跑，或用 `--overwrite` 清除后重新运行。

官方 API 文档：[接入说明](https://api-docs.deepseek.com/)、[Chat Completions](https://api-docs.deepseek.com/api/create-chat-completion/)、[模型更新记录](https://api-docs.deepseek.com/updates/)。托管模型别名可迁移到更新版本；研究时应记录实际模型版本。

正式参赛可把**自行部署的开源权重**暴露为兼容 Chat Completions 的本机接口，再复用相同的输入、输出和格式校验代码。例如本机模型服务监听 `http://127.0.0.1:8000/v1`、模型名为服务实际名称时，运行：

```powershell
python run_deepseek.py --track A --limit 3 --api-url http://127.0.0.1:8000/v1 --model YOUR_LOCAL_MODEL --output outputs/a_local_preview.json
```

本机地址不要求 DeepSeek Key；若本机服务不支持 `response_format`，加 `--no-json-mode`。这个命令**只连接已启动的模型服务**；模型权重、运行环境、许可证及可复现性仍需单独落实。不能把第三方托管服务填成“本地地址”规避赛事规则。

先分别试 A/B/C，再考虑全量：

```powershell
python run_deepseek.py --track B --limit 3 --output outputs/b_preview.json
python run_deepseek.py --track C --limit 3 --output outputs/c_preview.json
python run_deepseek.py --track all --limit 0 --output outputs/sample_test_760.json
python run_deepseek.py --track all --limit 0 --validate outputs/sample_test_760.json
```

默认只处理 3 题，`--limit 0` 才处理所选范围内全部问题。每完成一题会把结果存到同名 `.partial` 文件；网络或格式错误时，修复后重复原命令即可续跑，不会重问已经保存的问题。`--overwrite` 会丢弃现有输出重新计算，可能重复产生 API 费用。

输出字段为 `sample_id`、`answer`、`evidence_chain`、`confidence`、`question_type`。证据 ID 必须来自当前材料；拒答时写 `"无法确定"`、空链和 `null`。校验命令检查字段、ID 与选中问题的覆盖率，**不能代替平台评分或人工核对答案质量**。A 档若链上相邻事件不在给定边中，程序只提示人工复核，因为训练标注里也存在未由给定边直接连接的可接受链。主办方网页称会提供官方格式校验脚本和 `sample_submission.json`；本仓库目前未见这两个文件，取得后应再跑官方校验。

实验输出目录已在 `.gitignore` 中忽略。程序不读取 `gold/问答对_答案.json`，标准答案不会发到 API。

## 离线自测

```powershell
python -m unittest -v test_run_deepseek.py
```

测试使用本机模拟 API，不需要真实 Key，也不会调用 DeepSeek。
