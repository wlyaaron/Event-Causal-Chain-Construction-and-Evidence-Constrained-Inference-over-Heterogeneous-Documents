# 面向多源异构文档的事件因果链构建与证据约束推演

2026 大数据与计算智能挑战赛赛题 4 团队资料仓库。赛事截图保存于 **2026-10-02**；时间、规则和格式以[赛事页面](https://www.heywhale.com/org/cunsl/competition/area/6aba35f66e5beff2fca5b1e8/content/0)最新内容为准。

## 赛事入口与交流群

- [大赛官网](https://d2.nudt.edu.cn/tzs) · [赛题详情推文](https://mp.weixin.qq.com/s/NhCaa2aOPc789S-SWOLwbA) · [组委会 FAQ](https://dcnmzcw6gpr9.feishu.cn/share/base/view/shrcnghueQSnnSX7THcWfOVWCHb)
- 大赛已上线，请在官网报名组队；集中问题可先查 FAQ，其他问题可在群内提出。

赛事群通知原文：

> 1️⃣各位参赛的老师、同学，感谢大家的耐心等待，🔥 2026大数据与计算智能挑战赛正式上线！
>
> 官网：https://d2.nudt.edu.cn/tzs
>
> 👇 赛题详情请看推文，抓紧报名组队啦！推文链接：https://mp.weixin.qq.com/s/NhCaa2aOPc789S-SWOLwbA

> 2️⃣ 感谢各位老师、同学对本次大赛的支持和参与🌹🌹
>
> 针对大家集中咨询的问题，运营团队整理了一份FAQ，欢迎大家查阅，如有其他疑问可随时在群内提出。文档链接：https://dcnmzcw6gpr9.feishu.cn/share/base/view/shrcnghueQSnnSX7THcWfOVWCHb

![赛题 4 交流群二维码](docs/赛题4交流群二维码.png)

## 赛题、数据与提交

目标是读取多篇文档，回答因果问题，并给出从原因到结果排列的证据事件 ID 和置信度。任务 A 给文档、事件列表和因果边，权重 **60%**；B 给文档和候选事件但需自行判断因果方向，权重 **30%**；C 涵盖事件抽取、反事实、冲突证据和无答案问题，权重 **10%**。[本地抽样测试集](数据集/抽样测试集_100/)共 A 210、B 350、C 200，合计 **760 题**。其中本地 B/C 文件包只有文档和问题，缺少网页描述中的候选事件列表。代码按实际文件处理，暂用文档 ID 作 B/C 证据候选；取得正式数据后须核对。

初赛提交单个 UTF-8 JSON 数组，网页示例的每题字段如下：

```json
[{"sample_id":"safety_001_Q01","answer":"冷藏温度超标且整改未落实，最终导致食物中毒","evidence_chain":["S001","S003","S004"],"confidence":0.91,"question_type":"retrospective"}]
```

无法确定时按网页示例写 `"answer":"无法确定"`、`"evidence_chain":[]`、`"confidence":null`。评测按答案正确性 **40%**、证据链准确性 **40%**、置信度与拒答 **20%** 计分；单个 JSON 覆盖 A/B/C。每天最多提交三次，算法和结果须可复现。规则要求符合条件的开源模型，禁止直接用闭源 API 生成正式参赛预测，也禁止人工标注测试答案。DeepSeek 托管 API 脚本仅作研究诊断。

**待平台澄清：** 当前数据存在 `question_type=unanswerable`，网页示例主要列出 `retrospective`、`prospective`、`counterfactual`；FAQ 对拒答又提到 `null`。程序目前原样保留输入的 `question_type`，只做本地结构校验，不擅自改映射。线上评测和官方 `sample_submission.json` / 校验脚本开放后，再按官方要求复核。

## 方法核查

这些代码依据公开描述**独立实现同类方法**。主办方未公开基线源代码、模型权重或提示词，不能声称复现其分数。

| 方法 | 实际使用 | 当前边界 |
| --- | --- | --- |
| `graph` | A 档给定有向因果边搜索路径、模板作答 | B/C 无边时只保守定位单个文档；答案模板粗 |
| `bm25_bert` | 中文字符词 BM25（词频、IDF、长度归一化）加本地 BGE-small-zh-v1.5 BERT 编码器余弦相似度排序，复用图链与模板答案 | 真正使用了 BM25 与 BERT 语义重排；并非 BERT 因果分类器，答案生成能力与 `graph` 相同 |
| `rag` | BM25 检索候选，保留题目明确点名的文档，再交本地模型生成 | 真正先检索再生成；并非向量 RAG，检索可能漏跨文档证据 |
| `llama3` | 全量材料送本地兼容接口或离线 Hugging Face 模型，复用统一校验 | 通用本地模型适配器；仓库没有 Llama3-8B 权重，未验证真正 Llama3-8B 效果 |
| `run_deepseek.py` | 全量材料送兼容 Chat Completions 接口 | 已跑通托管 DeepSeek 单题结构；托管 API 不作为正式提交路径 |

本地校验检查五个字段、证据 ID 与题目覆盖，**不证明答案事实、因果方向或官方得分正确**。图方法和 BM25+BERT 曾对本地 760 题全部生成并通过结构校验；训练集前 100 题的证据链完全匹配率分别约 0.38 和 0.37。RAG 接口与离线模型加载路径已试过，所用 0.5B 小模型在真实样例上未稳定给出合格 JSON。官方公开综合分（图 25.52、BM25+BERT 40.20、普通 RAG 40.48、Llama3-8B 50.85、DeepSeek-V4-Flash 57.61）不能直接套用到本仓库实现。

## 代码结构

| 位置 | 职责 |
| --- | --- |
| [`task4_core.py`](task4_core.py) | 发现题目、读取推理材料、构造输入、解析和校验统一提交格式、原子保存 |
| [`task4_api.py`](task4_api.py) | 兼容 Chat Completions 的 URL、Key、诊断和请求 |
| [`run_baselines.py`](run_baselines.py) | BM25、BERT 排序、图路径、RAG、本地模型与方法命令行 |
| [`run_deepseek.py`](run_deepseek.py) | DeepSeek/兼容接口的研究命令行，复用核心与接口代码 |
| [`evaluate_baselines.py`](evaluate_baselines.py) | 仅训练集诊断时读取标准答案；推理代码不读 `gold` |
| `test_run_*.py` | 不需真实 Key 的测试 |
| [`数据集/`](数据集/) · `docs/赛事网页内容/` | 原始数据 · 六张赛事截图 |

## 运行

在仓库根目录用 Python 3.10+ 执行。默认只处理 3 题；`--limit 0` 表示所选范围全部题。输出放在已忽略的 `outputs/`；中断后用相同命令可从 `.partial` 续跑。

```powershell
python run_baselines.py --method graph --track all --limit 0 --output outputs/graph_760.json
python run_baselines.py --method graph --track all --limit 0 --validate outputs/graph_760.json
python run_baselines.py --method bm25_bert --track A --limit 3 --embedding-model models/bge-small-zh-v1.5 --output outputs/bm25_bert_a.json
python run_baselines.py --method rag --track A --limit 3 --local-hf-model models/YOUR_OPEN_MODEL --output outputs/rag_a.json
python run_baselines.py --method llama3 --track A --limit 3 --api-url http://127.0.0.1:8000/v1 --model YOUR_LOCAL_LLAMA3_MODEL --output outputs/llama3_a.json
python run_deepseek.py --dry-run --track A --limit 3
python -m unittest -q test_run_deepseek.py test_run_baselines.py
```

`bm25_bert` 需要本地 [BAAI/bge-small-zh-v1.5](https://huggingface.co/BAAI/bge-small-zh-v1.5) 权重和 `sentence-transformers`；RAG 离线模式需要本地开源生成权重、`torch`、`transformers`。模型权重不在 Git 中。RAG 也可接已启动的本机 Chat Completions 服务（`--api-url`、`--model`）。研究 DeepSeek 托管 API 可运行 `python run_deepseek.py --probe-api --api-url https://api.deepseek.com --model deepseek-flash`；Key 从交互输入或 `--api-key-file` 读取，不能提交 Key 或实验输出。取得官方校验脚本后还须用它复核。

## 赛事页面截图

以下图片保存于 2026-10-02；点击标题可看对应网页最新版本。

### [赛程安排](https://www.heywhale.com/org/cunsl/competition/area/6aba35f66e5beff2fca5b1e8/content/0)

初赛 9 月 30 日至 10 月 14 日；**10 月 4 日开启线上提交和评测**。复赛 10 月 15 日至 17 日，决赛 10 月 23 日。

![赛程安排截图](docs/赛事网页内容/01_赛程安排.png)

### [赛题](https://www.heywhale.com/org/cunsl/competition/area/6aba35f66e5beff2fca5b1e8/content/1)

![赛题截图](docs/赛事网页内容/02_赛题.png)

### [数据](https://www.heywhale.com/org/cunsl/competition/area/6aba35f66e5beff2fca5b1e8/content/2)

![数据说明截图](docs/赛事网页内容/03_数据.png)

### [提交要求](https://www.heywhale.com/org/cunsl/competition/area/6aba35f66e5beff2fca5b1e8/content/3)

![提交要求截图](docs/赛事网页内容/04_提交要求.png)

### [评测](https://www.heywhale.com/org/cunsl/competition/area/6aba35f66e5beff2fca5b1e8/content/4)

![评测截图](docs/赛事网页内容/05_评测.png)

### [规则要求和常见问题](https://www.heywhale.com/org/cunsl/competition/area/6aba35f66e5beff2fca5b1e8/content/5)

![规则与常见问题截图](docs/赛事网页内容/06_规则要求和常见问题.png)
