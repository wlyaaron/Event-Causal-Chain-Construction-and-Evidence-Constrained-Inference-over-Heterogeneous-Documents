# 赛题 4：Qwen3.5-9B 微调来源与环境核查（2026-10-06）

## 结论范围

本记录区分公开来源、工程选择和本机实测。当前只使用赛事合法训练集；验证、留出包不进入训练，760 道无 gold 测试题不用于调参。DeepSeek API 研究输出不进入本微调数据。两步 GPU 冒烟训练已经完成；它证明软件和短样本链路可运行，不代表模型质量或完整训练已验证。

## 可向老师说明的来源

- **模型**：[Qwen 官方 Qwen3.5-9B 模型卡](https://huggingface.co/Qwen/Qwen3.5-9B)给出 9B 语言模型、262144 原生上下文、Apache-2.0 许可及官方权重。下载源为官方 ModelScope `Qwen/Qwen3.5-9B`，Git 提交 `460979c3d11864dd16408d860ac930a360a2fac2`。四个权重分片 SHA-256 与官方 Hugging Face 文件页面逐一相同；模型、数据、LoRA 产物只在服务器 `outputs/` 中。
- **微调框架**：[Qwen3 官方仓库](https://github.com/QwenLM/Qwen3#finetuning)把 Axolotl、Unsloth、Swift、LLaMA-Factory 并列列为 SFT 等微调可用框架。Qwen3.5-9B 模型卡未指定唯一优选微调框架，因此“必须用 LLaMA-Factory”不是官方结论。选择它是工程选择：[LLaMA-Factory v0.9.5](https://github.com/hiyouga/LlamaFactory/releases/tag/v0.9.5)明确加入 Qwen3.5 与 Transformers v5 支持。使用上游未修改 Git 提交 `7af909522a951e3ad9f022ea6f88b6755257eaa5`，另装于数据盘，保留系统盘原有 2024 年 LLaMA-Factory 和其中工作树。
- **方法**：[LoRA](https://arxiv.org/abs/2106.09685)提供低秩参数更新依据；[QLoRA](https://arxiv.org/abs/2305.14314)提供 4-bit 冻结底座加 LoRA 的省显存依据。[Liger Kernel 上游](https://github.com/linkedin/Liger-Kernel)明确列有 Qwen3.5 的融合线性层交叉熵支持。是否优于其他方法、学习率和 rank 取值，均需本题验证集实验，不能从论文直接推出。
- **长材料与证据**：[Lost in the Middle](https://aclanthology.org/2024.tacl-1.9/)提示长输入中的证据位置会影响使用效果。本题采用完整材料，并审计 token 数、证据位置与截断；这不是对 16K 一定优于检索压缩的证明。

## 服务器与依赖实测

- 新服务器单张 RTX 4090，显存 49140 MiB；内存 cgroup 约 92 GiB，25 CPU 核。数据盘 `/root/autodl-tmp` 50 GB，模型和 Python 环境均在数据盘；系统盘旧工作保留。
- 独立环境 `/root/autodl-tmp/task4-qwen35-venv`：Python 3.11，PyTorch 2.6.0+cu124，Transformers 5.6.0，PEFT 0.18.1，TRL 0.24.0，Accelerate 1.11.0，bitsandbytes 0.47.0，LLaMA-Factory 0.9.5。`pip check` 无冲突，CUDA 可用。环境通过 `--system-site-packages` 创建，升级中发现 pip 移除了原环境的 torchvision/torchaudio，已经恢复原环境的 0.19.1+cu121 / 2.4.1+cu121；原环境 `pip check` 通过。
- 官方模型通过 ModelScope 镜像下载，约 18 分 48 秒；服务器直连 Hugging Face 被拒，直连 GitHub 克隆超时。LLaMA-Factory 上游仓库先在本机克隆，再传至服务器数据盘；PyPI 清华镜像可用。原有系统盘框架不具备本次模型的明确支持，未被改动。
- 首次 GPU 冒烟失败：继承的 PyTorch 2.4.1 缺少 Transformers 5.6 量化路径调用的 `nn.Module.set_submodule`。失败日志留在忽略目录 `outputs/llamafactory_logs/qwen35_smoke_4k_torch241_failed.log`。在任务环境安装 PyTorch 2.6.0 后重跑成功。Qwen Gated DeltaNet 的 fast path 依赖尚未安装，当前使用 Transformers 的 torch fallback。
- 首次 16K 压力测试失败：在 16356-token 样本计算全词表交叉熵时，CUDA 还需分配 14.78 GiB 而仅余 12.56 GiB，日志为 `outputs/llamafactory_logs/qwen35_long_16k_no_liger_oom.log`。随后仅在任务环境安装 `liger-kernel==0.8.4`，并在正式与压力配置启用 `enable_liger_kernel: true`。重跑的一步训练成功，训练段 82.92 秒，日志为 `outputs/llamafactory_logs/qwen35_long_16k_liger.log`；观察时显存约 25 GiB，非全程峰值。此压力测试不等同于完整训练时长评估。

## 数据与截断

- 固定包隔离划分 `experiments/finetune_split_20261005.json`：fit/validation/holdout 为 801/101/98 包。v3 数据 A/B/C fit 5767/5767/5672，validation 709/709/706，holdout 702/702/685。完整 gold 候选链保留，单次监督选一条完整链；原始数据可重建，JSONL 不进 Git。
- 使用 Qwen3.5-9B tokenizer 和框架真实 `qwen3_5_nothink` 模板测长。训练混合集原 11562 行，16K `cutoff_len: 16384` 下保留 11538 行，超长 24 行单列于 manifest，不截断后训练。保留集按视图 A/B/C 为 6843/3520/1175 行，最长 16356 token；源 SHA-256 为 `e148764714d546a6b4179909f808ddbdb0abd22446e93b54ba409cd63fea3570`，保留集 SHA-256 为 `dcfd46ad3383eb6e6d6f36cac30331655e42cea25d90aaf106d608a5abd257d0`。
- 以 8K 限制会导致大量训练样本超长，且部分 gold 证据落在界外。16K 是本次无静默截断的起点，不表示所有比赛材料都小于该窗口。需要将 24 行作为长上下文待处理组单独评估。
- 数据已有风险分层：18 题链引用缺失事件、789 题有步骤不在给定边表（不能据此自动判错）、93 道 unanswerable 有解释性答案/证据、703 题缺 required_facts、2722 题多 gold 路径、跨划分同文问题加答案 498 题。评估要保留这些分层，不能简单删除或制造数值 confidence 监督标签。

## 已跑通的训练与待验证项

- 短样本配置 `experiments/configs/task4_qwen35_9b_llamafactory_smoke.yaml`：六条 fit 样本，A/B/C 各两条，最大 3608 token；4-bit QLoRA、LoRA rank 16、batch 1、BF16、梯度检查点、冻结视觉塔。两步完成并保存约 700 MB adapter，训练段 12.15 秒，train loss 1.438；这一损失值只用于证明反向传播成功。
- 近 16K 配置 `experiments/configs/task4_qwen35_9b_llamafactory_long_16k.yaml`：从 fit 各视图挑一条最长且不超限的完整样本，A/B/C 分别为 16356/15905/14577 token。第一步恰好训练最长的 A 样本；启用 Liger 后一步完成并保存 adapter，训练段 82.92 秒。该单条 loss 0.6601 也不能用于判断质量。
- 正式 16K 输入配置为 `experiments/configs/task4_qwen35_9b_llamafactory_fit_16k.yaml`，仅含 fit 数据；`num_train_epochs: 1`、有效 batch 8、LoRA rank 16、学习率 `1e-4` 都是待验证的初始假设。验证/留出评估须用同一份固定划分和已有评价脚本另行运行，不在训练时随机切分。
- 正式数据生成命令如下。现存目录已经生成且脚本拒绝覆盖；复核现有 `manifest.json` 与 SHA-256 后即可训练。若重新生成，先使用新的输出目录并相应改配置路径，保留旧数据供对照。

```bash
cd /root/autodl-tmp/event-causal-chain-repo
PYTHONPATH=. /root/autodl-tmp/task4-qwen35-venv/bin/python experiments/prepare_llamafactory_data.py \
  --source outputs/finetune_2026-10-06_v3/fit_mixed_fixed_2epoch.jsonl \
  --tokenizer outputs/qwen3.5-9b-tokenizer \
  --output-dir outputs/llamafactory_qwen35_16k \
  --cutoff-len 16384 --dataset-name task4_fit_fixed_qwen35_16k \
  --llamafactory-commit 7af909522a951e3ad9f022ea6f88b6755257eaa5
HF_HOME=/root/autodl-tmp/task4-hf-cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  CUDA_VISIBLE_DEVICES=0 /root/autodl-tmp/task4-qwen35-venv/bin/llamafactory-cli train \
  experiments/configs/task4_qwen35_9b_llamafactory_fit_16k.yaml
```

当前准备工作不等于已经完成一轮正式训练；配置中的一步压力测试只验证目标长度可训练。

## 版本与边界

原始模型、赛事原始数据、重建 JSONL、训练 adapter、失败日志、SSH 凭证均放在 Git 忽略目录，不提交 Git。正式效果仍须按 A/B/C 分档报告答案字符 F1、语义、关键事实覆盖代理、证据节点/边 F1、完整链、拒答和成本；数值 confidence 只能在验证集上研究校准。未经授权不向赛事平台提交预测。
