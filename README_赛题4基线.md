# 赛题 4：可运行的对照方法

**这些是依据公开描述独立实现的同类方法，不是主办方产生分数表的原始代码。** 官网只公布名称与分数，没有公布 BM25+BERT、普通 RAG、图推理和 Llama3-8B 的模型版本、训练策略、提示词或源代码。因此下面的结果不能声称复现官网分数。

## 先区分“格式正确”和“答得正确”

所有方法统一输出 UTF-8 JSON 数组，每题严格包含 `sample_id`、`answer`、`evidence_chain`、`confidence`、`question_type` 五个字段；证据链按原因到结果排列。拒答必须是 `"无法确定"`、`[]`、`null`。`run_deepseek.py --validate` 和 `run_baselines.py --validate` 可检查字段、ID 和覆盖率，**无法证明答案事实、边方向、置信度校准或平台得分正确**。

此前单题 `deepseek-flash` 已通过结构校验。根据 [DeepSeek 官方更新日志](https://api-docs.deepseek.com/updates/)，2026 年 9 月后这个 API 名称调用的是 **DeepSeek-V4.1-Flash**；旧的 `deepseek-v4-flash` 也临时路由到 V4.1。因此它并非官网表格中的原版 **DeepSeek-V4-Flash**，57.61 分不能直接套用到我们的运行结果。

## 方法如何工作

| 方法 | 实现 | 必要资源 | 当前局限 |
| --- | --- | --- | --- |
| `graph` 图推理 | A 档沿给定有向边找链，用边的 `confidence_level` 给 0.9/0.7/0.5；B/C 无给定边时只取一条可定位证据 | Python 标准库 | B/C 不能可靠恢复多跳链；模板答案较粗 |
| `bm25_bert` | 中文字符 BM25 检索候选事件，再用本地 BERT 架构的 BGE-small-zh-v1.5 语义重排；图链与答案按上述规则生成 | 开源 BERT 权重、`sentence-transformers`、CPU 可运行 | 不是训练后的因果分类器；B/C 多跳能力仍弱 |
| `rag` | BM25 找相关文档，保留题目明确引用的文档，再连同可用事件和因果边交给生成模型；强制结构校验，首次无效时纠错重试一次 | 本地 Hugging Face 开源权重或兼容 Chat Completions 的模型服务 | 检索可能漏证据；用托管 API 的实验结果不可直接参赛 |
| `llama3` | 完整材料送到**本地**兼容接口的 Llama3-8B 开源权重，使用同一输出校验 | 已运行的本地 Llama3-8B 服务 | 本机没有该模型权重和服务，只有模拟接口测试通过 |

官网 [赛题](https://www.heywhale.com/org/cunsl/competition/area/6aba35f66e5beff2fca5b1e8/content/1)将 BM25、BERT、向量检索列为候选事件匹配基础方法，将 RAG 列为挑战任务可选路线；[评测](https://www.heywhale.com/org/cunsl/competition/area/6aba35f66e5beff2fca5b1e8/content/4)给出的综合分分别是图推理 25.52、BM25+BERT 40.20、普通 RAG 40.48、Llama3-8B 50.85、DeepSeek-V4-Flash 57.61。我们没有原始基线实现，不能与这些分数做同条件比较。

## 本机使用

Python 环境已安装 `torch 2.10.0+cpu`、`transformers 4.45.2`、`sentence-transformers 5.2.3`。BERT 权重采用 [BAAI/bge-small-zh-v1.5](https://huggingface.co/BAAI/bge-small-zh-v1.5)，MIT 许可证，固定版本 `7999e1d3359715c523056ef9478215996d62a620`。权重约 96 MB，留在本地 `models/bge-small-zh-v1.5`，未上传 Git。新机器先下载：

```python
from huggingface_hub import snapshot_download
snapshot_download("BAAI/bge-small-zh-v1.5",
                  revision="7999e1d3359715c523056ef9478215996d62a620",
                  local_dir="models/bge-small-zh-v1.5",
                  allow_patterns=["*.json", "*.txt", "*.safetensors"])
```

在仓库根目录执行：

```powershell
python run_baselines.py --method graph --track A --limit 3 --output outputs/graph_a.json
python run_baselines.py --method bm25_bert --track A --limit 3 --output outputs/bm25_bert_a.json
python run_baselines.py --method rag --track A --limit 3 --api-url http://127.0.0.1:8000/v1 --model YOUR_LOCAL_MODEL --output outputs/rag_a.json
python run_baselines.py --method rag --track A --limit 1 --top-k 2 --local-hf-model models/YOUR_OPEN_MODEL --output outputs/rag_offline_a.json
python run_baselines.py --method llama3 --track A --limit 3 --api-url http://127.0.0.1:8000/v1 --model YOUR_LOCAL_LLAMA3_MODEL --output outputs/llama3_a.json
python run_baselines.py --method graph --track all --limit 0 --output outputs/graph_760.json
python run_baselines.py --method graph --track all --limit 0 --validate outputs/graph_760.json
```

`--limit 0` 处理全部所选题目。每题保存到 `.partial`，重跑同一命令会续跑；`--overwrite` 清除旧结果。`--local-hf-model` 直接在本机加载开源权重，不调用外部 API，可用 `--hf-context-tokens`、`--hf-max-new-tokens` 控制上下文和生成长度。`--method rag` 也可以用 `--api-key-file` 指向本机 Key 文件测试托管 API，**这仅是研究调试，不是符合赛事模型使用规则的正式提交路径**。不要上传 Key 或实验输出到仓库。

## 已完成的验证与边界

- 图推理：抽样测试集 A/B/C **760/760** 题生成并通过结构和覆盖率校验。
- BM25+BERT：抽样测试集 A/B/C **760/760** 题生成并通过结构和覆盖率校验；B/C 目前仅能稳定定位单个文档，答案质量有限。
- RAG：A/B/C 各 1 道真实题通过。这里使用 DeepSeek 官方托管 API 仅验证程序；B 档首次因无证据链被校验拒绝，纠错重试后通过。
- Llama3-8B：本地兼容接口路径通过模拟服务测试，**尚未下载权重或完成真实模型推理**。
- 图推理在训练集前 100 题的本地诊断：证据链完全匹配率 0.38；与多条标准链比较后的最佳事件集合 F1 为 0.657；8 道空链题均拒答。这不包含答案关键事实、链边方向与置信度评分，**不是官网 25.52 分**。代码见 `evaluate_baselines.py`，只读取训练集 `gold` 作评估；推理代码不读取标准答案。
- BM25+BERT 在同一训练集前 100 题：证据链完全匹配率 0.37、最佳事件集合 F1 为 0.652；这也不是官网 40.20 分。
- 离线 RAG 适配器使用 [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)（Apache-2.0，版本 `7ae557604adf67be50417f59c2c2f167def9a775`）在 CPU 上完成实际模型生成，但两道真实样例产生了不完整 JSON 或无依据推断，**被提交校验拒绝，未生成合格文件**。这个小模型仅验证本地加载链路，不能作为当前参赛模型。权重留在本地 `models/`，未上传 Git。

网页称会提供官方 `sample_submission.json` 和格式校验脚本，本仓库当前尚未收到。取得后要再次以官方脚本校验。正式结果还需离线开源生成模型、平台测分与复现资料。
