"""Make an exploratory same-50 comparison without overstating causality."""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = ROOT / "outputs/a_candidate_v4_pair_holdout50_20261010/score/all_scores.json"
POSTHOC = ROOT / "outputs/a_candidate_v4_repair6_20261010/posthoc_50_diagnostic.json"
NEW = ROOT / "outputs/a_candidate_answer_aligned_v2_20261010/score/all_scores.json"
DOC = ROOT / "docs/赛题4_A类候选池答案对齐提示词_50题探索性复跑_2026-10-10.md"
OUT = ROOT / "outputs/a_candidate_answer_aligned_v2_20261010/score/paired_diagnostic.json"
ID = re.compile(r"D\d{3,}")


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(value):
    return "—" if value is None else str(value)


def main() -> None:
    original, posthoc, new = read(ORIGINAL), read(POSTHOC), read(NEW)
    old_cases = {c["sample_id"]: c for c in original["cases"]}
    post_cases = {c["sample_id"]: c for c in posthoc["cases"]}
    new_cases = {c["sample_id"]: c for c in new["cases"]}
    if not (len(old_cases) == len(post_cases) == len(new_cases) == 50):
        raise ValueError("Case mismatch")
    if set(old_cases) != set(post_cases) or set(old_cases) != set(new_cases):
        raise ValueError("Different questions")
    transitions = Counter()
    rows = []
    for sid in old_cases:
        old, post, fresh = old_cases[sid], post_cases[sid], new_cases[sid]
        orig = old["results"]["guided"]
        previous = post["results"]["guided"]
        current = fresh["results"]["new"]
        pred = current["prediction"] or {}
        chain = pred.get("evidence_chain")
        extras = sorted(set(ID.findall(pred.get("answer") or "")) - set(chain or []))
        old_hit = bool(previous["metrics"]["chain_exact"])
        new_hit = bool(current["metrics"]["chain_exact"])
        transitions["gain" if new_hit and not old_hit else
                    "loss" if old_hit and not new_hit else
                    "both_hit" if old_hit else "both_miss"] += 1
        rows.append({"sample_id": sid, "category": old["category"],
                     "candidate_hit": old["candidate_hit"],
                     "gold_chains": old["gold_chains"], "gold_answer": old["gold_answers"],
                     "base": old["results"]["base"], "original_guided": orig,
                     "posthoc_guided": previous, "new": current,
                     "new_answer_ids_outside_chain": extras,
                     "posthoc_repaired": post.get("posthoc_repaired", False)})
    summary = {"base": original["summary"]["all"]["base"],
               "original_guided": original["summary"]["all"]["guided"],
               "posthoc_guided": posthoc["summary"]["posthoc_guided"],
               "new": new["summary"]}
    report = {"warning": "Same 50 cases informed revisions; diagnostic only. New run changed candidate display, answer instruction, one-chain instruction, and max_tokens versus the original guided arm.",
              "summary": summary, "posthoc_to_new_transitions": dict(transitions), "rows": rows}
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# A 类候选池：答案与单链对齐提示词的 50 题探索性复跑", "",
             "本轮与原始 50 题使用相同的冻结问题、完整 A 视图材料和 DeepSeek `deepseek-flash`；gold 从未发给模型。V4 system prompt 未改，用户提示前缀加入一条路线、答案依据链内事件的条件约束；候选采用编号文字，输出上限为 32768。**这 50 题已经用于诊断并影响提示词设计，因此本轮不是独立验证；同时改了候选呈现与上限，不能把差额全归因于答案约束。**", "",
             "| 指标 | 纯 V4 原始 | 候选 V4 原始 | 六题事后修复视图 | 本轮新提示词 |",
             "|---|---:|---:|---:|---:|"]
    for label, key in (("有效", "valid"), ("完整链", "chain_exact_count"),
                       ("答案字符 F1", "answer_char_f1"),
                       ("答案语义余弦", "answer_semantic_cosine"),
                       ("节点 F1", "chain_node_f1"), ("有向边 F1", "chain_edge_f1"),
                       ("输入 token", "prompt_tokens"), ("输出 token", "completion_tokens"),
                       ("累计请求耗时/秒", "latency_seconds_sum")):
        lines.append(f"| {label} | " + " | ".join(fmt(summary[arm][key]) for arm in
             ("base", "original_guided", "posthoc_guided", "new")) + " |")
    lines += ["", "请求完成时 DeepSeek 余额从 50.42 元显示为 50.21 元，**即时可见余额差 0.21 元**；服务商结算可能滞后，不能将此数视为最终费用。无赛事平台提交。", "",
              f"与六题事后修复视图相比：新命中 {transitions['gain']} 题，退步 {transitions['loss']} 题，两者都命中 {transitions['both_hit']} 题，两者都未命中 {transitions['both_miss']} 题。", "",
              "| 问题子类 | 题数 | 事后视图完整链 | 新提示词完整链 | 事后视图答案 F1 | 新提示词答案 F1 |",
              "|---|---:|---:|---:|---:|---:|"]
    for category in ("event_facts", "unspecified_direct", "direct_effect",
                     "temporal_causal_pair", "explicit_path"):
        subset = [r for r in rows if r["category"] == category]
        count = len(subset)
        old_count = sum(bool(r["posthoc_guided"]["metrics"]["chain_exact"]) for r in subset)
        new_count = sum(bool(r["new"]["metrics"]["chain_exact"]) for r in subset)
        old_f1 = sum(r["posthoc_guided"]["metrics"]["answer_char_f1"] for r in subset) / count
        new_f1 = sum(r["new"]["metrics"]["answer_char_f1"] for r in subset) / count
        lines.append(f"| {category} | {count} | {old_count} | {new_count} | {old_f1:.4f} | {new_f1:.4f} |")
    lines += ["",
              "## 变化的原因与边界", "",
              "新提示词把 answer 明写所选链外事件 ID 的题数从事后修复视图的 12/50 降到 1/50；在链命中题中则从 3/33 降到 0/40。链与答案的**形式一致性**改善了，但这不能自动证明每一项未标 ID 的事实都只来自链内材料。", "",
              "完整链多命中 7 题，均为原来漏掉的题，没有链命中的回退。新增命中中，6 题是直接后果/直接关系题，模型不再把若干平行直接边串成一条；另 1 题是 `diplomacy_134_Q02`，模型采纳图中 D004→D001 的标注，但该标注对回顾报道与历史进程的语义仍有争议。", "",
              "仍未命中 10 题：6 题是候选含 gold 但模型选了另一条**同样有图边的直接后果**；`public_safety_099_Q02` 候选含 gold，却因图标 D003→D001 与事故/调查日期冲突而反向选择；`natural_disaster_121_Q06` 与 `economy_trade_174_Q06` 的 gold 是非因果判断锚点，图路径候选不含它；`diplomacy_070_Q01` 的长链 gold 不在前五，模型自行生成仍多出 D006、D010。因此上一轮的“长链拼接”在直接后果题上已明显缓解，剩余主要是**多个平行后果究竟选哪个**，以及图与原文/标注冲突。", "",
              "为什么这 6 题会选错：题目都问‘之后直接发生了什么’，图中确有多个并行直接后果；模型现在只选一条，却按新闻内容的显著性选了工作部署、伤员救治或社会反响等另一条。训练集同类问法 639 题中，512 题有多个直接后果，gold 在 636/639 题选择了 ID 顺序最靠前的那个；fit 包独立统计为 503/505。这解释了为什么候选中有正确链，模型仍可能按语义选择另一条合法边。该规律有 3 题例外，不能写成‘最小 ID 必然正确’的通用因果规则，更不能把这 50 题继续反复调成迎合编号。见 [全量统计](../experiments/research_audit_20261010/direct_effect_gold_priority.json)。", "",
              "答案指标出现负结果：字符 F1 从事后视图 0.4124 降为 0.3781，语义余弦从 0.8006 降为 0.7777；逐题字符 F1 17 题上升、33 题下降。新答案平均长度从 157.94 字增加到 175.96 字，说明“围绕链”没有让答案自动变简洁。`natural_disaster_072_Q04` 的链前后都正确，但新答案只写 D001→D002→D003→D010 的编号与边型，省掉 gold 中的事件名称、起点和终点效果；`natural_disaster_026_Q06` 虽改对 D001→D002，却把领导指示及后续部署讲得过长，偏离 gold 的简短直接后果格式。`diplomacy_070_Q01` 继续列出 D006、D010 和许多细节，链与答案均未收敛。故本轮不能宣称答案质量提升。", "",
              "这 50 题已经参与规则设计，不能用 40/50 推断对新题也会达到 80%；若要确认泛化，须冻结代码和提示词后另选未看过的包重新评估。当前改进点已从格式/长链转移到平行直接后果的选择与答案简洁性，但不应根据这 50 题继续定制特定 ID 的优先规则。", "",
              "## 逐题变化", "",
              "| 题目 | 类型 | gold 完整链 | 六题修复视图链 | 新提示词链 | 变化 | 答案链外 ID |",
              "|---|---|---|---|---|---|---|"]
    for row in rows:
        prev = row["posthoc_guided"]["prediction"] or {}
        cur = row["new"]["prediction"] or {}
        p_hit = bool(row["posthoc_guided"]["metrics"]["chain_exact"])
        c_hit = bool(row["new"]["metrics"]["chain_exact"])
        change = "↑" if c_hit and not p_hit else "↓" if p_hit and not c_hit else "="
        lines.append("| `{sid}` | {cat} | `{gold}` | `{before}` | `{after}` | {change} | {extras} |".format(
            sid=row["sample_id"], cat=row["category"],
            gold="; ".join("→".join(x) for x in row["gold_chains"]),
            before="→".join(prev.get("evidence_chain") or []),
            after="→".join(cur.get("evidence_chain") or []),
            change=change, extras=",".join(row["new_answer_ids_outside_chain"]) or "—"))
    lines += ["", "逐题完整答案、原始 gold、候选、四组预测与指标见 [配对诊断 JSON](../outputs/a_candidate_answer_aligned_v2_20261010/score/paired_diagnostic.json)。评价是本地原始完整 gold 备选链诊断，不是官方测试分。", ""]
    DOC.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"summary": summary["new"], "transitions": transitions,
                      "output": str(DOC)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
