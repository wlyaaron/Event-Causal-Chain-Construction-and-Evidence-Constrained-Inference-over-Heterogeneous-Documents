"""Merge six post-hoc repairs into the original 50 for failure analysis.

This never overwrites the prospective paired score. The merged row is a
post-hoc debugging view and must not be reported as a fresh efficacy test.
"""
from __future__ import annotations

import copy
import json
import re
from collections import Counter
from pathlib import Path

from experiments.score_a_candidate_v4_pair_20261010 import summarize


ROOT = Path(__file__).resolve().parents[1]
SCORE = ROOT / "outputs" / "a_candidate_v4_pair_holdout50_20261010" / "score" / "all_scores.json"
REPAIR = ROOT / "outputs" / "a_candidate_v4_repair6_20261010" / "score.json"
OUT = ROOT / "outputs" / "a_candidate_v4_repair6_20261010" / "posthoc_50_diagnostic.json"
DOC = ROOT / "docs" / "赛题4_A类候选池事后修复并入50题_2026-10-10.md"
ID = re.compile(r"D\d{3,}")

REASONS = {
    "economy_trade_117_Q07": "题问单个直接后果，模型把 D005、D006 两条并行直接边拼成一维序列；候选第1条就是 gold。",
    "economy_trade_123_Q06": "题问直接后果，模型列尽五个并行后果，并将它们拼成链；gold 在候选第3条。",
    "diplomacy_157_Q06": "题问直接后果，模型列尽五个并行后果；gold 在候选第1条。",
    "public_safety_075_Q06": "题问直接后果，模型把多个直接后果及后续事件串联；gold 在候选第4条。",
    "public_safety_183_Q06": "题问直接后果，模型跨越多个并行分支，还在 answer 提及非链上事件；gold 在候选第1条。",
    "diplomacy_101_Q06": "题问直接后果，模型列四个并行结果；gold 在候选第4条。",
    "economy_trade_045_Q06": "题问直接后果，模型列尽五个并行结果；gold 在候选第3条。",
    "economy_trade_066_Q06": "题问直接后果，模型列尽五个并行结果；gold 在候选第5条。",
    "natural_disaster_026_Q06": "修复后已输出单链，但挑了候选第1条 D001→D003，应选第3条 D001→D002；多个直接后果的优先级不由图边自身决定。",
    "natural_disaster_183_Q06": "题问直接后果，模型列四个并行结果并附加 D007；gold 在候选第2条。",
    "public_safety_106_Q06": "题问直接后果，模型列尽五个并行结果；gold 在候选第4条。",
    "economy_trade_133_Q06": "题问直接后果，模型列尽五个并行结果；gold 在候选第1条。",
    "natural_disaster_121_Q06": "候选无完整 gold；模型依据材料日期反驳题设先后，并反向提交 D008→D005。需人工核对 gold 与事件时间。",
    "public_safety_099_Q02": "候选第1条为 gold D003→D001，但模型依据事故先于调查的现实时间否定它；gold/图边与时间语义有冲突风险。",
    "diplomacy_134_Q02": "候选第1条为 gold D004→D001，模型认为回顾性报道不可能先于条约并拒绝其因果；将历史历程与报道事件区分的语义判断不同于标注。",
    "economy_trade_174_Q06": "候选无完整 gold D004→D006；模型选 D005→D006 作为直接诉讼原因，虽然 answer 同样否定仅凭时间推因果，链指向另一条路线。",
    "diplomacy_070_Q01": "完整 gold 不在前五候选；模型从图上自构更长路径并加入 D006、D010，形成多余中间节点。",
}


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    original = read(SCORE)
    repairs = {row["sample_id"]: row["repair"] for row in read(REPAIR)["cases"]}
    if len(original["cases"]) != 50 or len(repairs) != 6:
        raise ValueError("Incomplete input")
    cases = copy.deepcopy(original["cases"])
    for case in cases:
        sid = case["sample_id"]
        if sid in repairs:
            case["results"]["guided"] = repairs[sid]
            case["posthoc_repaired"] = True
        else:
            case["posthoc_repaired"] = False
    misses = []
    classification = Counter()
    for case in cases:
        result = case["results"]["guided"]
        pred = result["prediction"] or {}
        chosen = pred.get("evidence_chain")
        selected_ids = set(chosen) if isinstance(chosen, list) else set()
        answer_ids = set(ID.findall(pred.get("answer") or ""))
        case["answer_mentions_ids_outside_chosen_chain"] = sorted(answer_ids - selected_ids)
        if result["metrics"]["chain_exact"]:
            continue
        sid = case["sample_id"]
        gold = {tuple(x) for x in case["gold_chains"]}
        pool = case["candidate_paths"]
        ranks = [i + 1 for i, x in enumerate(pool) if tuple(x) in gold]
        chosen_rank = next((i + 1 for i, x in enumerate(pool) if x == chosen), None)
        bucket = "candidate_present_chose_wrong" if ranks else "candidate_absent_no_recovery"
        classification[bucket] += 1
        misses.append({"sample_id": sid, "category": case["category"],
                       "candidate_gold_ranks": ranks, "selected_candidate_rank": chosen_rank,
                       "predicted_chain": chosen, "gold_chains": case["gold_chains"],
                       "candidate_paths": pool, "question": case["question"],
                       "gold_answer": case["gold_answers"],
                       "predicted_answer": pred.get("answer"),
                       "reason": REASONS[sid]})
    if set(REASONS) != {x["sample_id"] for x in misses}:
        raise ValueError("Manual reason mapping needs review")
    report = {
        "warning": "Post-hoc replacement of six previously invalid outputs; not a new 50-question efficacy estimate.",
        "original_score": str(SCORE), "repair_score": str(REPAIR),
        "summary": {"base": summarize(cases, "base"),
                    "original_guided": original["summary"]["all"]["guided"],
                    "posthoc_guided": summarize(cases, "guided")},
        "miss_classification": dict(classification), "misses": misses,
        "cases": cases,
    }
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# A 类候选池：六题事后修复并入 50 题的诊断视图", "",
             "六题是按上一轮无效输出事后选择并补跑的。下表仅用于定位剩余错误，**不是一轮新的 50 题配对实验，也不能作为泛化提升证据**。原始 50 题分数及原始 JSON 保持不动。", "",
             "| 指标 | 纯 V4 原始 50 | 候选 V4 原始 50 | 候选 V4 + 六题事后修复 |",
             "|---|---:|---:|---:|"]
    for label, key in (("有效输出", "valid"), ("完整链命中", "chain_exact_count"),
                       ("answer 字符 F1", "answer_char_f1"),
                       ("answer 语义余弦", "answer_semantic_cosine"),
                       ("节点 F1", "chain_node_f1"), ("有向边 F1", "chain_edge_f1")):
        data = report["summary"]
        lines.append(f"| {label} | {data['base'][key]} | {data['original_guided'][key]} | {data['posthoc_guided'][key]} |")
    lines += ["", f"剩余未命中 {len(misses)} 题：候选中已有完整 gold 链 {classification['candidate_present_chose_wrong']} 题；候选中没有且模型未自行找回 {classification['candidate_absent_no_recovery']} 题。", "",
              "## 剩余未命中的逐题原因", "",
              "| 题目 | 类型 | gold 在候选中的位次 | 模型所选 | 原始完整 gold | 诊断 |",
              "|---|---|---|---|---|---|"]
    for row in misses:
        ranks = ",".join(map(str, row["candidate_gold_ranks"])) or "无"
        lines.append("| `{sample_id}` | {category} | {ranks} | `{chosen}` | `{gold}` | {reason} |".format(
            sample_id=row["sample_id"], category=row["category"], ranks=ranks,
            chosen="→".join(row["predicted_chain"] or []),
            gold="; ".join("→".join(x) for x in row["gold_chains"]), reason=row["reason"]))
    lines += ["", "## 解释与下一步", "",
              "12 题是直接后果或未指定终点的直接关系题：图上有多个平行直接边，gold 通常只取其中一条。11 题仍把平行结果写成单条链或全部列进 answer；另 1 题虽然只选一条，却选错分支。候选召回在这 12 题均已命中，因此瓶颈是问题所指的单个主结论与平行边的取舍。", "",
              "全量训练集复核发现一种强烈的标注规律：639 道包含‘之后直接发生了什么’的题中，512 道从问题起点有多个图上的直接后果，gold 却在 636/639 道选择 event_id 最小的那个直接后果；fit 包单独是 503/505。三道例外分别为 `public_safety_229_Q04`、`healthcare_02_Q02`、`natural_disaster_184_Q06`。这是标注分布，不是说明最小编号的后果在因果语义上总是唯一正确。详见 [统计 JSON](../experiments/research_audit_20261010/direct_effect_gold_priority.json)。", "",
              "4 题是时间／因果判断：其中 2 题候选本就缺 gold，另 2 题模型与 gold 对事件先后或报道语义的判断冲突。不能把这类现象简单归咎于链长；需查看原文与图边，并允许标注本身存在争议。", "",
              "1 题是多阶段完整链：前五候选不含任何完整 gold 备选，模型自构链又增加了额外中间节点。需要改善召回排序，再检查长链选择。", "",
              "在 33 道链命中题中，仍有 3 道 answer 明确写出所选链外的事件 ID：`natural_disaster_193_Q06` 写 D003/D004/D005，`diplomacy_170_Q01` 与 `public_safety_191_Q01` 都写 D002/D004。这是链对但答案可能扩写的直接证据；不能用链命中代表答案也正确。", "",
              "### 时间题的原文复核", "",
              "`natural_disaster_121_Q06` 的题设与 gold answer 说 D005 早于 D008，但事件列表 `occur_time` 分别是 2025-01-09 和 2025-01-08；模型反驳该先后关系有原文依据。`public_safety_099_Q02` 的图确实标了 D003→D001 直接因果，但 D001 事故日期是 2021-06-25、D003 调查日期是 2021-06-27；模型按时间拒绝逆向因果也有依据。`diplomacy_134_Q02` 的图标 D004→D001 间接传导，D004 却是 2001-07-16 对此前历史进程的回顾报道，与 D001 条约签署同日；“历史进程促成条约”和“回顾报道促成条约”有语义差别。`economy_trade_174_Q06` 的 gold D004→D006 是非因果判断的两个证据锚点，图中无对应边；只从图路径召回候选当然找不到。这些题的原始 gold 仍照收评分，但归因必须标注争议，不能把它们都算成模型忽略正确链。", "",
              "建议下一版只修改通用输出约束：候选继续编号展示；最终 `evidence_chain` 只能是一条一维数组；answer 的核心结论由这条路线直接支持，并避免把平行后果和链外背景展开为答案。遇到比较、反事实、冲突或干扰识别等问法，允许简要引用链外对照材料，但必须区分其作用，不得将它们拼成同一因果链。该规则仍需在完整 50 题上重新调用验证，且因这 50 题已经用于诊断，结果只能视作探索性复跑。", ""]
    DOC.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"summary": report["summary"], "miss_classification": classification,
                      "output": str(OUT), "doc": str(DOC)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
