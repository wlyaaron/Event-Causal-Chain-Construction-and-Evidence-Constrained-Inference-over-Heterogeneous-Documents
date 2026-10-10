"""Post-call original-gold diagnostics for six repaired guided responses."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import task4_core as core  # noqa: E402
from experiments.evaluate_validation_diagnostic import evaluate  # noqa: E402
from experiments.freeze_a_candidate_v4_repair6_20261010 import OUT as MANIFEST  # noqa: E402
from experiments.run_a_candidate_v4_repair6_20261010 import OUT  # noqa: E402
from experiments.score_a_candidate_v4_pair_20261010 import EMBEDDING  # noqa: E402

ORIGINAL = (ROOT / "outputs" / "a_candidate_v4_pair_holdout50_20261010" /
            "score" / "all_scores.json")
REPORT = ROOT / "docs" / "赛题4_A类候选池六题修复诊断_2026-10-10.md"


def md(value: object) -> str:
    content = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    return content.replace("|", "\\|").replace("\n", "<br>")


def main() -> None:
    frozen = core.read_json(MANIFEST)
    rows = frozen["rows"]
    attempts_path = OUT / "attempts.jsonl"
    attempts = [json.loads(line) for line in attempts_path.read_text(
        encoding="utf-8").splitlines()]
    by_id = {item["sample_id"]: item for item in attempts}
    if len(attempts) != 6 or set(by_id) != {row["sample_id"] for row in rows}:
        raise ValueError("Need all six unique repair attempts before scoring")
    predictions = core.read_json(OUT / "predictions.json")
    valid_ids = {item["sample_id"] for item in predictions}
    if valid_ids != {sid for sid, item in by_id.items() if item["parsed"] is not None}:
        raise ValueError("Prediction/attempt mismatch")
    metrics = {}
    if predictions:
        valid_manifest = OUT / "valid_manifest.json"
        core.atomic_json(valid_manifest, {"rows": [
            {**row, "risk_flags": []} for row in rows if row["sample_id"] in valid_ids]})
        evaluation = evaluate(valid_manifest, {"repair": OUT / "predictions.json"},
                              ROOT / "数据集" / "训练集", str(EMBEDDING))
        core.atomic_json(OUT / "gold_metrics.json", evaluation)
        metrics = {item["sample_id"]: item["metrics"]["repair"]
                   for item in evaluation["rows"]}
    old_report = core.read_json(ORIGINAL)
    old = {item["sample_id"]: item for item in old_report["cases"]}
    cases = []
    for row in rows:
        sid = row["sample_id"]
        attempt = by_id[sid]
        original = old[sid]
        new = attempt["parsed"]
        cases.append({"sample_id": sid, "question": row["question"],
                      "previous_failure_type": row["previous_failure_type"],
                      "candidate_paths": row["candidate_paths"],
                      "gold_answers": original["gold_answers"],
                      "gold_chains": original["gold_chains"],
                      "old_base": original["results"]["base"],
                      "old_guided": original["results"]["guided"],
                      "repair": {"valid": new is not None, "prediction": new,
                                 "raw": attempt["attempt"].get("raw"),
                                 "error": attempt["attempt"].get("validation_error") or
                                          attempt["attempt"].get("error"),
                                 "finish_reason": attempt["attempt"].get("finish_reason"),
                                 "usage": attempt["attempt"].get("usage"),
                                 "elapsed_seconds": attempt["attempt"].get("elapsed_seconds"),
                                 "metrics": metrics.get(sid)}})
    valid = [case for case in cases if case["repair"]["valid"]]
    summary = {"n": 6, "valid": len(valid),
               "chain_exact": sum(case["repair"]["metrics"]["chain_exact"] for case in valid),
               "candidate_chain_chosen": sum(tuple(case["repair"]["prediction"]["evidence_chain"])
                                             in {tuple(path) for path in case["candidate_paths"]}
                                             for case in valid),
               "prompt_tokens": sum((case["repair"]["usage"] or {}).get("prompt_tokens", 0)
                                    for case in cases),
               "completion_tokens": sum((case["repair"]["usage"] or {}).get("completion_tokens", 0)
                                        for case in cases),
               "reasoning_tokens": sum(((case["repair"]["usage"] or {}).get(
                   "completion_tokens_details") or {}).get("reasoning_tokens", 0)
                   for case in cases),
               "elapsed_seconds_sum": round(sum(case["repair"]["elapsed_seconds"] or 0
                                                for case in cases), 2)}
    report = {"scope": "post-hoc repair of six previous invalid guided outputs",
              "method_note": "Same V4 system and frozen full inputs; numbered candidates, one-chain instruction, max_tokens 32768. Gold read only after calls. This is not a new 50-question efficacy estimate.",
              "summary": summary, "cases": cases}
    core.atomic_json(OUT / "score.json", report)

    lines = ["# A 类候选池六题修复诊断", "",
             "这 6 题由上一轮候选组的**无效输出**预先确定：3 题把多条候选写成二维数组，"
             "3 题思考 token 用完而截断。模型调用前不读取 gold。本轮仅作错误修复诊断，"
             "不能用它把原 50 题结果改写成新成绩。", "",
             "## 修改", "",
             "- DeepSeek `deepseek-flash`、V4 system prompt、完整原始 A 视图材料保持不变。",
             "- 候选由二维 JSON 列表改为编号的文本路线；明确最终 `evidence_chain` 只能是一条一维 ID 数组。",
             "- `max_tokens` 从 8192 提高到 32768；思考模式及 effort 仍由服务商默认设置。",
             "", "## 结果", "",
             f"有效输出 **{summary['valid']}/6**；完整原始 gold 备选链命中 "
             f"**{summary['chain_exact']}/6**。最终回答累计消耗输入 "
             f"{summary['prompt_tokens']:,} token、输出 {summary['completion_tokens']:,} token"
             f"（其中思考 {summary['reasoning_tokens']:,}），累计请求耗时 "
             f"{summary['elapsed_seconds_sum']:.2f} 秒。", "",
             "本轮同时改变了候选呈现方式和输出上限，不能凭这六题区分两项修改各自的贡献。"
             "三道原截断题修复后的输出 token 分别为 8157、5227、8950；最后一题超过旧上限 8192。", "",
             "**有效不等于全答对。**六题的答案字符 F1 均值从对应纯 V4 的 "
             f"{sum(case['old_base']['metrics']['answer_char_f1'] for case in cases)/6:.4f} "
             "到修复后的 "
             f"{sum(case['repair']['metrics']['answer_char_f1'] for case in cases)/6:.4f}；"
             "语义余弦从 "
             f"{sum(case['old_base']['metrics']['answer_semantic_cosine'] for case in cases)/6:.4f} "
             "到 "
             f"{sum(case['repair']['metrics']['answer_semantic_cosine'] for case in cases)/6:.4f}。"
             "`natural_disaster_026_Q06` 虽恢复合法格式，却选择了 `D001→D003`，"
             "原始 gold 是 `D001→D002`；`natural_disaster_193_Q06` 的链虽命中，"
             "answer 仍列举了多个直接后续事件，不能据链命中宣称整个回答正确。", "",
             "上一轮全 50 题中，候选仅使输入 token 从 "
             f"{old_report['summary']['all']['base']['prompt_tokens']:,} 增至 "
             f"{old_report['summary']['all']['guided']['prompt_tokens']:,}，"
             "却使输出 token 从 "
             f"{old_report['summary']['all']['base']['completion_tokens']:,} 增至 "
             f"{old_report['summary']['all']['guided']['completion_tokens']:,}。"
             "这与模型花更多思考 token 比较多条路线相符，但单次随机调用不能严格证明因果。"
             "本次六题输出中思考 token 占输出的 "
             f"{summary['reasoning_tokens']/summary['completion_tokens']:.1%}。", "",
             "| 题目 | 原失败 | 修复后有效 | 修复后完整链 | 纯 V4 完整链 | 修复后输出 token |",
             "|---|---|---:|---:|---:|---:|"]
    for case in cases:
        repair = case["repair"]
        lines.append(f"| `{case['sample_id']}` | {case['previous_failure_type']} | "
                     f"{'是' if repair['valid'] else '否'} | "
                     f"{repair['metrics']['chain_exact'] if repair['metrics'] else '—'} | "
                     f"{case['old_base']['metrics']['chain_exact']} | "
                     f"{(repair['usage'] or {}).get('completion_tokens', 0)} |")
    lines += ["", "## 逐题对照", ""]
    for case in cases:
        repair = case["repair"]
        prediction = repair["prediction"]
        lines += [f"### {case['sample_id']}", "",
                  f"问题：{case['question']}", "",
                  f"候选：`{json.dumps(case['candidate_paths'], ensure_ascii=False)}`", "",
                  "| 项目 | 原始 gold | 纯 V4 | 修复后候选组 |",
                  "|---|---|---|---|",
                  f"| 答案 | {md(case['gold_answers'])} | "
                  f"{md(case['old_base']['prediction']['answer'])} | "
                  f"{md(prediction['answer'] if prediction else repair['error'])} |",
                  f"| 证据链 | {md(case['gold_chains'])} | "
                  f"{md(case['old_base']['prediction']['evidence_chain'])} | "
                  f"{md(prediction['evidence_chain'] if prediction else '无效')} |",
                  f"| 答案字符 F1 / 语义 / 完整链 | — | "
                  f"{case['old_base']['metrics']['answer_char_f1']:.4f} / "
                  f"{case['old_base']['metrics']['answer_semantic_cosine']:.4f} / "
                  f"{case['old_base']['metrics']['chain_exact']} | "
                  f"{repair['metrics']['answer_char_f1']:.4f} / "
                  f"{repair['metrics']['answer_semantic_cosine']:.4f} / "
                  f"{repair['metrics']['chain_exact']} |" if repair["metrics"] else
                  f"| 答案字符 F1 / 语义 / 完整链 | — | "
                  f"{case['old_base']['metrics']['answer_char_f1']:.4f} / "
                  f"{case['old_base']['metrics']['answer_semantic_cosine']:.4f} / "
                  f"{case['old_base']['metrics']['chain_exact']} | 无效 |",
                  ""]
    lines += ["本地指标依照原始完整 gold 备选链计算，不是官方测试分数。"
              "本轮 6 题已经暴露在调试过程中；任何后续方法有效性结论都需要新的冻结题集。", ""]
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"summary": summary, "report": str(REPORT),
                      "score": str(OUT / "score.json")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
