"""Render reproducible teacher summary and 50-question side-by-side audit."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCORE = ROOT / "outputs" / "a_candidate_v4_pair_holdout50_20261010" / "score" / "all_scores.json"
FIT = ROOT / "experiments" / "research_audit_20261010" / "a_candidate_v3_fit.json"
HOLDOUT = ROOT / "experiments" / "research_audit_20261010" / "a_candidate_v3_holdout.json"
META = ROOT / "outputs" / "a_candidate_v4_pair_holdout50_20261010" / "metadata.json"
SUMMARY = ROOT / "docs" / "赛题4_A类候选池与V4提示词_50题配对实验_2026-10-10.md"
DETAIL = ROOT / "docs" / "赛题4_A类候选池50题逐题对照_2026-10-10.md"
CHART = ROOT / "docs" / "赛题4_A类候选池50题配对实验_结果图_2026-10-10.svg"


def fmt(value: object) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def cell(value: object) -> str:
    content = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    return content.replace("|", "\\|").replace("\n", "<br>")


def main() -> None:
    score = json.loads(SCORE.read_text(encoding="utf-8"))
    fit = json.loads(FIT.read_text(encoding="utf-8"))
    holdout = json.loads(HOLDOUT.read_text(encoding="utf-8"))
    meta = json.loads(META.read_text(encoding="utf-8"))
    cases = score["cases"]
    all_summary = score["summary"]["all"]
    paired = score["paired_guided_minus_base"]
    metrics = [
        ("有效输出", "valid"), ("答案字符 F1", "answer_char_f1"),
        ("答案语义余弦", "answer_semantic_cosine"),
        ("证据节点 F1", "chain_node_f1"), ("有向边 F1", "chain_edge_f1"),
        ("完整链命中", "chain_exact"),
    ]
    lines = [
        "# 赛题4：A 类候选池与 DeepSeek V4 的 50 题配对实验",
        "",
        "## 结论",
        "",
        "程序候选池的**可用性**较高，但本轮把候选直接加入 V4 提示词，未形成明确的端到端提升。"
        "在预先冻结的 50 道 A 视图题上，候选前 5 条覆盖 47 道题的至少一条完整原始 gold 链；"
        "完整链命中由 26/50 升至 28/50（4 题改善、2 题退步，配对 bootstrap 95% 区间跨 0）。"
        "候选组有 6 次无效输出，纯 V4 组为 0 次；因此按全 50 题计，答案字符 F1、语义和节点 F1 均下降。"
        "这说明当前主要瓶颈已经从‘候选是否含金链’转到‘模型如何单选合法链及维持输出格式’，"
        "尚不能认定候选提示带来了稳定增益。",
        "",
        "![50题候选覆盖与模型完整链命中](赛题4_A类候选池50题配对实验_结果图_2026-10-10.svg)",
        "",
        "## 实验设置",
        "",
        "- 数据：赛事训练集预先划分出的 98 个 holdout 材料包；50 道题来自 50 个不同包。"
        "仅用题面、事件和图按固定哈希抽题，未用 gold 选题。全部模拟 A 类视图，完整文档、事件、关系同时提供。",
        "- 类型配额：明确完整路径 15、明确后续直接事件 15、两 ID 时间/因果判断 15、"
        "单 ID 不定方向直接因果 2、单事件事实 3。全部属于 retrospective；本子集没有严格拒答 gold，"
        "不能用于判断拒答方法的优劣。",
        "- 模型：两组均为 DeepSeek Chat Completions 的 `deepseek-flash`，JSON mode，"
        "`max_tokens=8192`，未显式设置 temperature 或 reasoning_effort。两组的 V4 system prompt 完全相同，"
        "SHA-256 为 `" + meta["system_prompt_sha256"] + "`。",
        "- 唯一实验变量：候选组在相同原始 user 输入前增加下述短段与最多五条程序候选链；"
        "没有 ICL、二次调用、gold 或答案提示。候选不保证正确，允许模型否决或自行构链。",
        "",
        "> 以下是程序依据当前材料中的有向因果图和问题生成的候选 evidence_chain，按相关性排序，仅供核对，不保证正确或完整。"
        "请以完整材料自行判断；可以选用、修正或否决候选并自行构建合法链。不要仅因候选存在就强行回答。",
        "> 候选 evidence_chain：`[最多五条路径]`",
        "",
        "## 程序候选池先验核查",
        "",
        "两条局部规则：题面明确给出两个 ID 且询问时间/因果时，若图中存在所提顺序的有向边，"
        "优先展示该二节点候选；题面给出单 ID 并询问‘与哪一事件存在直接因果’时，"
        "优先展示该节点两侧被标为直接因果的边。规则只读输入，保留其他路径作为候选。",
        "",
        "| 完整原始 gold 链前 5 命中 | 原 V2 | 局部 V3 | 新增/损失 |",
        "|---|---:|---:|---:|",
    ]
    for label, audit in (("fit 其他回溯题", fit), ("holdout 其他回溯题", holdout)):
        group = audit["groups"]["other_retrospective"]
        lines.append(f"| {label} | {group['v2_hit']}/{group['n']} | "
                     f"{group['v3_hit']}/{group['n']} | +{group.get('gains', group.get('gain', 0))}"
                     f" / -{group.get('losses', group.get('loss', 0))} |")
    lines += ["", "其他题型在两次审计中均无命中变化；这是候选覆盖审计，不能代替模型作答实验。",
              "", "## 50 题模型结果", "",
              "| 指标（全 50 题，越高越好） | 纯 V4 | V4 + 候选 | 配对差值 95% 区间 |",
              "|---|---:|---:|---:|"]
    for label, key in metrics:
        base = all_summary["base"][key]
        guided = all_summary["guided"][key]
        if key == "valid":
            lines.append(f"| {label} | {base}/50 | {guided}/50 | — |")
        else:
            ci = paired[key]["ci95"]
            lines.append(f"| {label} | {fmt(base)} | {fmt(guided)} | "
                         f"{fmt(paired[key]['mean_delta'])} [{fmt(ci[0])}, {fmt(ci[1])}] |")
    lines += ["", "完整链配对：两组都中 24；仅候选组中 4；仅纯 V4 中 2；两组都不中 20。"
              "47/50 的候选召回并未转化为同等幅度的模型命中。",
              "", "| 题型 | 题数 | 候选含完整 gold | 纯 V4 完整链 | 候选组完整链 | 候选组无效 |",
              "|---|---:|---:|---:|---:|---:|"]
    labels = {"explicit_path": "明确完整路径", "direct_effect": "明确后续直接事件",
              "temporal_causal_pair": "两 ID 时间/因果", "unspecified_direct": "单 ID 不定方向直接因果",
              "event_facts": "单事件事实"}
    for category, label in labels.items():
        group = score["summary"][category]
        lines.append(f"| {label} | {group['base']['n']} | {group['candidate_hit']} | "
                     f"{group['base']['chain_exact_count']} | {group['guided']['chain_exact_count']} | "
                     f"{group['guided']['invalid']} |")
    valid = [case for case in cases if all(case["results"][arm]["valid"]
                                       for arm in ("base", "guided"))]
    lines += ["", "### 有效题诊断（非主结果）", "",
              f"两组都有效的 {len(valid)} 题里，完整链为 "
              f"{sum(x['results']['base']['metrics']['chain_exact'] for x in valid)}/{len(valid)} "
              f"→ {sum(x['results']['guided']['metrics']['chain_exact'] for x in valid)}/{len(valid)}；"
              "答案字符 F1 为 0.4060 → 0.4075，语义余弦为 0.7948 → 0.7980。"
              "这些条件统计只能定位无效输出的影响，不能替代全 50 题结论。",
              "", "### 明确的收益与损失", "",
              "- `natural_disaster_124_Q01`：纯 V4 输出过长的 `D001-D002-D003-D004-D009`；"
              "候选组选择完整 gold 备选链 `D001-D003-D004-D009`。",
              "- `natural_disaster_005_Q04`：纯 V4 缺少 gold 起点 `D002`；"
              "候选组改为 `D002-D001-D009-D010`，命中完整备选链。",
              "- `diplomacy_134_Q02`：图上候选 `D004-D001` 正好是 gold，纯 V4 也给出该链；"
              "候选组反而只给 `D004`，并否定题目中的因果关系。"
              "这是模型根据原文复核后与标注不一致的实例，不能简单用候选强制覆盖。",
              "- 6 次候选组无效中，3 次把多条候选照抄为二维 `evidence_chain`，"
              "3 次思考 token 耗尽后截断。原协议只允许一维单条链；无效按零计。",
              "", "## 成本与判断", "",
              f"两组输入 token 分别为 {all_summary['base']['prompt_tokens']:,}、"
              f"{all_summary['guided']['prompt_tokens']:,}；输出含思考 token 分别为 "
              f"{all_summary['base']['completion_tokens']:,}、"
              f"{all_summary['guided']['completion_tokens']:,}；请求耗时总和分别为 "
              f"{all_summary['base']['latency_seconds_sum']:.2f} 秒、"
              f"{all_summary['guided']['latency_seconds_sum']:.2f} 秒。"
              "100 次请求的余额从 52.29 元到最后一次查询的 51.36 元，已显示扣费 0.93 元；"
              "账务可能滞后，因此这是当前可见扣费，不当作最终结算额。",
              "", "**判断**：本轮不支持将当前候选提示直接替换纯 V4。"
              "继续研究时，先解决“候选池是多个可选链，但输出只能是一条链”的接口表达，"
              "并限制候选引发的额外思考/截断；之后再在新的冻结题集复验。"
              "不得用本 50 题的 gold 反向调候选或把无效输出事后修成有效。",
              "", "详细的 50 题 gold、两组原答案、链和逐题分数见"
              " [逐题对照](赛题4_A类候选池50题逐题对照_2026-10-10.md)。"
              "这些数字均为本地原始 gold 诊断，不是官方测试分数；没有使用 760 道无 gold 测试题调参或提交。",
              ""]
    SUMMARY.write_text("\n".join(lines), encoding="utf-8")

    bars = [("程序候选含完整 gold", 47, "#4f7dba"),
            ("纯 V4 完整链命中", 26, "#667085"),
            ("V4 + 候选完整链命中", 28, "#247d64")]
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="930" height="360" viewBox="0 0 930 360">',
           '<rect width="930" height="360" fill="#ffffff"/>',
           '<text x="34" y="43" font-family="Arial, sans-serif" font-size="23" font-weight="700" fill="#13253a">候选覆盖较高，模型收益尚不明确</text>',
           '<text x="34" y="72" font-family="Arial, sans-serif" font-size="14" fill="#526174">50 道训练集 holdout 包题目 · 模拟 A 视图 · 同题配对</text>']
    for index, (label, count, color) in enumerate(bars):
        y = 112 + index * 70
        width = 430 * count / 50
        svg += [f'<text x="34" y="{y+19}" font-family="Arial, sans-serif" font-size="16" fill="#13253a">{label}</text>',
                f'<rect x="330" y="{y}" width="430" height="29" rx="6" fill="#edf1f5"/>',
                f'<rect x="330" y="{y}" width="{width:.1f}" height="29" rx="6" fill="{color}"/>',
                f'<text x="775" y="{y+21}" font-family="Arial, sans-serif" font-size="17" font-weight="700" fill="#13253a">{count}/50</text>']
    svg += ['<text x="34" y="333" font-family="Arial, sans-serif" font-size="14" fill="#9b2c2c">候选组另有 6 次无效输出；完整链 +2 题的配对区间跨 0。</text>',
            '</svg>']
    CHART.write_text("\n".join(svg) + "\n", encoding="utf-8")

    detail = ["# 50 题逐题对照：原始 gold 与两组 DeepSeek V4", "",
              "同一材料包的完整输入只送给模型，不在本页重复新闻原文。"
              "每题展示题面、程序候选、原始完整 gold 备选链及两组输出。"
              "gold 仅在全部预测完成后由本地评分器读取。", ""]
    for index, case in enumerate(cases, 1):
        base = case["results"]["base"]
        guided = case["results"]["guided"]
        def answer(record: dict) -> str:
            return record["prediction"]["answer"] if record["prediction"] else (
                "无效输出：" + str(record["error"]) + "；原始响应：" +
                str(record["raw"] or "<无内容>"))
        def chain(record: dict) -> object:
            return record["prediction"]["evidence_chain"] if record["prediction"] else "无效"
        def result(record: dict) -> str:
            m = record["metrics"]
            return (f"完整链 {m['chain_exact']}；节点 F1 {fmt(m['chain_node_f1'])}；"
                    f"边 F1 {fmt(m['chain_edge_f1'])}；答案字符 F1 {fmt(m['answer_char_f1'])}；"
                    f"语义 {fmt(m['answer_semantic_cosine'])}")
        detail += [f"## {index}. {case['sample_id']}（{labels[case['category']]}）", "",
                   f"**问题**：{case['question']}", "",
                   f"**程序候选前 5**：`{json.dumps(case['candidate_paths'], ensure_ascii=False)}`；"
                   f"含完整 gold：{'是' if case['candidate_hit'] else '否'}。", "",
                   "| 项目 | 原始 gold | 纯 V4 | V4 + 候选 |",
                   "|---|---|---|---|",
                   f"| 答案 | {cell(case['gold_answers'])} | {cell(answer(base))} | {cell(answer(guided))} |",
                   f"| 证据链 | {cell(case['gold_chains'])} | {cell(chain(base))} | {cell(chain(guided))} |",
                   f"| 本地指标 | — | {cell(result(base))} | {cell(result(guided))} |",
                   ""]
    DETAIL.write_text("\n".join(detail), encoding="utf-8")
    print(json.dumps({"summary": str(SUMMARY), "detail": str(DETAIL),
                      "chart": str(CHART),
                      "cases": len(cases)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
