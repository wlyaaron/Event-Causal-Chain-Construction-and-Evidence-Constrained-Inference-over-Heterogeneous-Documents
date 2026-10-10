"""Render paired, verbatim model answers for the Task 4 stage report.

Case selection is diagnostic and explicitly non-representative. Every output
cell is loaded from saved scored predictions or, for an invalid output, raw
model text. No model is called and no output is rewritten.
"""
from __future__ import annotations

import html
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs" / "赛题4_提示词到程序候选_阶段实验真实输出横排对照_2026-10-10.md"
CASE_DATA = ROOT / "experiments" / "results" / "task4_stage_progress_cases_20261010.json"
SOURCES = {
    "v3": ROOT / "outputs/reasoning_type_90_pair_20261009/score/all_scores.json",
    "v4": ROOT / "outputs/deepseek_v4_20261009/score/all_scores.json",
    "hybrid": ROOT / "outputs/v4_hybrid12_20261010/score/all_scores.json",
    "route": ROOT / "outputs/deepseek_route_candidate_20261010/score/all_scores.json",
    "pair": ROOT / "outputs/a_candidate_v4_pair_holdout50_20261010/score/all_scores.json",
    "repair": ROOT / "outputs/a_candidate_v4_repair6_20261010/score.json",
    "final": ROOT / "outputs/a_candidate_answer_aligned_v2_20261010/score/paired_diagnostic.json",
}
CASES = [
    ("规则提示词：先看拒答边界", "v3", "public_safety_067_Q02", "base", "v3",
     "基础版", "合并问答与证据链规则 V3",
     "严格 gold 拒答被 V3 改写为有链解释；这是 V3 整体下降的核心类型。"),
    ("规则提示词：先看拒答边界", "v4", "energy_environment_04_Q02", "v3", "v4",
     "V3", "V4 零示例",
     "V4 恢复该题的严格拒答与空链；它也会在后面的两阶段流程里再次失守。"),
    ("小规模程序试探：候选不可盲加", "hybrid", "economy_trade_201_Q05", "first", "guided",
     "V4 原答", "V4＋程序候选复核",
     "原答已严格拒答；加入图路径后，模型把‘无法精确量化’扩成可回答的解释。"),
    ("两阶段路由：新增一次模型调用的代价", "route", "energy_environment_04_Q02", "v4", "method",
     "V4 原答", "路由＋条件复核",
     "路线判断把严格 gold 拒答变成纠错性解释；同一道题再次展示门控风险。"),
    ("两阶段路由：新增一次模型调用的代价", "route", "public_safety_129_Q09", "v4", "method",
     "V4 原答", "路由＋条件复核",
     "模型中间结果给出非法 basis ID，程序发现但流程未回退，吞掉有效初答。"),
    ("候选池直接加入 V4：格式风险", "pair", "natural_disaster_193_Q06", "base", "guided",
     "纯 V4", "V4＋原候选列表",
     "候选包含 gold，但模型把四条平行路线输出成二维 evidence_chain，按原协议无效。"),
    ("六题事后修复：仅供排错", "repair", "natural_disaster_193_Q06", "old_guided", "repair",
     "原候选输出", "编号候选＋单链要求＋较高上限",
     "格式与链修复成功，但新 answer 仍列入 D003/D004/D005，说明链对不等于答案对。"),
    ("最终同题探索复跑：链与答案一起看", "final", "diplomacy_101_Q06", "base", "new",
     "纯 V4", "候选＋单链及答案对齐",
     "平行后果收敛为 gold 的 D001→D002；链、字符 F1 与语义均提高。"),
    ("最终同题探索复跑：链与答案一起看", "final", "natural_disaster_072_Q04", "base", "new",
     "纯 V4", "候选＋单链及答案对齐",
     "两次都命中完整链，但新 answer 主要写编号与边型，弱化事件含义和结果解释；答案两项指标下降。"),
    ("最终同题探索复跑：链与答案一起看", "final", "natural_disaster_026_Q06", "base", "new",
     "纯 V4", "候选＋单链及答案对齐",
     "新链由错到对；新 answer 加入大量领导指示和部署细节，和简短 gold 答法距离更远。"),
    ("最终同题探索复跑：链与答案一起看", "final", "economy_trade_123_Q06", "base", "new",
     "纯 V4", "候选＋单链及答案对齐",
     "新版本只选一条图上直接边，但选了 D006 而非 gold D002；格式修复并不保证分支选择。"),
    ("最终同题探索复跑：链与答案一起看", "final", "diplomacy_070_Q01", "base", "new",
     "纯 V4", "候选＋单链及答案对齐",
     "前五候选缺完整 gold；模型在池外自构长链，仍多出 D006/D010。字符 F1 下降，语义接近。"),
    ("最终同题探索复跑：链与答案一起看", "final", "natural_disaster_121_Q06", "base", "new",
     "纯 V4", "候选＋单链及答案对齐",
     "两版都与 gold 链相反；事件表日期表明题设的 D005 早于 D008 不成立，需将此例标为标注争议。"),
]


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def fetch(source: str, sid: str, data: dict) -> tuple[dict, dict, dict]:
    if source == "v3":
        row = next(x for x in data["cases"]["deepseek"] if x["sample_id"] == sid)
        return row, row["results"], {"answer": row["gold_answers"], "chains": row["gold_chains"]}
    if source == "v4":
        row = next(x for x in data["sets"]["fresh30"]["cases"] if x["sample_id"] == sid)
        return row, row["results"], {"answer": row["gold_answers"], "chains": row["gold_chains"]}
    if source == "hybrid":
        row = next(x for x in data["cases"] if x["sample_id"] == sid)
        return row, row["results"], {"answer": row["gold_answers"], "chains": row["gold_chains"]}
    if source == "route":
        row = next(x for x in data["cases"] if x["sample_id"] == sid)
        results = {arm: {"prediction": row[arm], "valid": row[arm] is not None,
                         "metrics": row["metrics"][arm],
                         "error": row.get("decision") if row[arm] is None else None}
                   for arm in ("v4", "method")}
        return row, results, {"answer": row["gold_answer"], "chains": row["gold_chains"]}
    if source == "pair":
        row = next(x for x in data["cases"] if x["sample_id"] == sid)
        return row, {"base": row["results"]["base"], "guided": row["results"]["guided"]}, {"answer": row["gold_answers"], "chains": row["gold_chains"]}
    if source == "repair":
        row = next(x for x in data["cases"] if x["sample_id"] == sid)
        return row, {"old_guided": row["old_guided"], "repair": row["repair"]}, {"answer": row["gold_answers"], "chains": row["gold_chains"]}
    if source == "final":
        row = next(x for x in data["rows"] if x["sample_id"] == sid)
        if "question" not in row:
            original = read(SOURCES["pair"])
            row = {**row, "question": next(x["question"] for x in original["cases"]
                                            if x["sample_id"] == sid)}
        return row, {"base": row["base"], "new": row["new"]}, {"answer": row["gold_answer"], "chains": row["gold_chains"]}
    raise KeyError(source)


def cell(value: object) -> str:
    return html.escape(str(value), quote=False).replace("|", "&#124;").replace("\r", "").replace("\n", "<br>")


def chain(value: object) -> str:
    return cell(json.dumps(value, ensure_ascii=False))


def output(result: dict) -> str:
    prediction = result.get("prediction")
    metrics = result.get("metrics") or {}
    if prediction:
        answer = prediction.get("answer", "")
        evidence = prediction.get("evidence_chain")
        display = f"**answer**：{cell(answer)}<br>**chain**：`{chain(evidence)}`"
    else:
        raw = result.get("raw")
        display = "**无有效预测**"
        if raw:
            display += f"；原始输出：{cell(raw)}"
        if result.get("error"):
            display += f"<br>解析错误：{cell(result['error'])}"
    f1 = metrics.get("answer_char_f1")
    sem = metrics.get("answer_semantic_cosine")
    exact = metrics.get("chain_exact")
    return (display + "<br>**本地代理**：字 F1 " +
            ("—" if f1 is None else f"{f1:.3f}") + "；语义 " +
            ("—" if sem is None else f"{sem:.3f}") + "；整链 " +
            ("—" if exact is None else str(exact)))


def main() -> None:
    source_data = {name: read(path) for name, path in SOURCES.items()}
    source_hashes = {name: hashlib.sha256(path.read_bytes()).hexdigest()
                     for name, path in SOURCES.items()}
    case_records = []
    lines = ["# 赛题 4：从提示词到程序候选的真实输出横排对照", "",
             "这页供老师点开看**同一道题**的变化。每一行左侧是该轮基线原答，右侧是新增方法的原答；上方保留原始 gold 答案与**完整备选链**，没有摘编新闻原文，也没有改写模型输出。无效输出直接展示原始文本或错误。", "",
             "**表格读法示意：**`字 F1 0.345；语义 0.769；整链 1` 表示答案与 gold 的本地字符重合为 0.345、本地 BGE 语义余弦为 0.769，而提交的单条链恰好等于某一条完整 gold 备选链。整链为 1 **不表示答案正确**。每轮不同题集的均值不能横向相减；逐题选择是展示失败机制，不能视为随机抽样。", "",
             "所有统计均来自训练集有 gold 题的**本地代理**，非官方测试评分。早期 90／30／12 题属于 fit 研究；最后 50 题来自原先冻结的 holdout 包，但已多次用于诊断，因此最后复跑是探索性。DeepSeek API 输出仅用于研究对照。", ""]
    previous_section = None
    for idx, (section, source, sid, left, right, left_label, right_label, reason) in enumerate(CASES, 1):
        if section != previous_section:
            lines += [f"## {section}", ""]
            previous_section = section
        row, results, gold = fetch(source, sid, source_data[source])
        case_records.append({
            "stage": section,
            "source": source,
            "sample_id": sid,
            "question": row["question"],
            "gold": gold,
            "left_label": left_label,
            "right_label": right_label,
            "left_result": results[left],
            "right_result": results[right],
            "observation": reason,
        })
        if not gold["chains"] and gold["answer"] != "无法确定":
            raise ValueError(f"Review explanatory no-chain gold: {sid}")
        lines += [f"### {idx}. `{sid}`", "", f"**问题：**{cell(row['question'])}", "",
                  f"**原始 gold：**{cell(gold['answer'])}<br>完整备选链：`{chain(gold['chains'])}`", "",
                  f"| {left_label} | {right_label} |",
                  "|---|---|",
                  f"| {output(results[left])} | {output(results[right])} |", "",
                  f"**观察：**{reason}", ""]
    lines += ["## 数据与复核入口", "",
              "- [本页 13 题的结构化原答、gold、逐题代理指标与来源文件 SHA-256](../experiments/results/task4_stage_progress_cases_20261010.json)。来源输出文件保存在本地实验目录；此精简数据随报告上传，便于独立核对表格。",
              "- [90 题 V3 配对报告](赛题4_九类问题90题_双模型基础版对合并规则V3_实验结果_2026-10-09.md)、[V4 新 30 题报告](赛题4_DeepSeek_V4零示例复核_结果_2026-10-09.md)、[12 题代码复核报告](赛题4_证据链与严格拒答_提示词对代码辅助_2026-10-10.md)、[30 题两阶段路由报告](赛题4_DeepSeek路由与候选链实验_2026-10-10.md)。",
              "- [原始 50 题配对](../outputs/a_candidate_v4_pair_holdout50_20261010/score/all_scores.json)、[六题修复](../outputs/a_candidate_v4_repair6_20261010/score.json)、[最终 50 题逐题配对](../outputs/a_candidate_answer_aligned_v2_20261010/score/paired_diagnostic.json)。", ""]
    CASE_DATA.parent.mkdir(parents=True, exist_ok=True)
    CASE_DATA.write_text(json.dumps({"schema": "task4_stage_progress_cases_v1",
                                     "sources_sha256": source_hashes,
                                     "cases": case_records}, ensure_ascii=False, indent=2),
                         encoding="utf-8")
    DOC.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"cases": len(CASES), "output": str(DOC)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
