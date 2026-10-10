"""Score paired research predictions against original, complete validation gold.

Predictions: JSON list with sample_id, view, answer, evidence_chain. Additional
fields (usage, latency, model metadata) are preserved outside this evaluator.
Only organizer training gold is read; no blind-test path is accepted.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import evaluate_baselines as basic
import task4_core as core
import task4_workflow as workflow

ROOT = Path(__file__).resolve().parents[1]


def mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None


def edges(chain: list[str]) -> list[str]:
    return [f"{a}>{b}" for a, b in zip(chain, chain[1:])]


def gold_map(root: Path, packs: set[str]) -> dict[str, dict]:
    out = {}
    for pack in sorted(root.iterdir()):
        if pack.name in packs:
            for item in core.read_json(pack / "gold" / "问答对_答案.json"):
                if item["sample_id"] in out:
                    raise ValueError(f"Duplicate gold ID: {item['sample_id']}")
                out[item["sample_id"]] = item
    return out


def evaluate(manifest: Path, prediction_paths: dict[str, Path], train_root: Path,
             embedding_model: str | None = None) -> dict:
    frozen = core.read_json(manifest)["rows"]
    expected = {(r["sample_id"], r["view"]): r for r in frozen}
    if len(expected) != len(frozen):
        raise ValueError("Frozen manifest has duplicate keys")
    samples = {s.sample_id: s for s in core.discover_samples(train_root, limit=0)
               if s.pack.name in {r["pack"] for r in frozen}}
    gold = gold_map(train_root, {r["pack"] for r in frozen})
    encoder = None
    if embedding_model:
        from sentence_transformers import SentenceTransformer
        encoder = SentenceTransformer(embedding_model, device="cpu")
    predictions = {}
    for arm, path in prediction_paths.items():
        records = core.read_json(path)
        arm_rows = {}
        for record in records:
            key = (record.get("sample_id"), record.get("view"))
            if key not in expected or key in arm_rows:
                raise ValueError(f"{arm}: unexpected or duplicate key {key}")
            answer, chain = record.get("answer"), record.get("evidence_chain")
            if not isinstance(answer, str) or not answer.strip() or not isinstance(chain, list):
                raise ValueError(f"{arm}: malformed answer/chain {key}")
            _, allowed, _ = workflow.training_view_input(samples[key[0]], key[1])
            if (any(not isinstance(node, str) or node not in allowed for node in chain)
                    or len(chain) != len(set(chain))):
                raise ValueError(f"{arm}: invalid evidence ID {key}")
            if (answer == "无法确定") != (chain == []):
                raise ValueError(f"{arm}: refusal/chain conflict {key}")
            arm_rows[key] = record
        if set(arm_rows) != set(expected):
            raise ValueError(f"{arm}: missing {len(set(expected) - set(arm_rows))} frozen items")
        predictions[arm] = arm_rows

    cache = {}
    def semantic(text: str, refs: list[str]) -> float | None:
        if not encoder or not refs:
            return None
        import numpy as np
        for value in [text, *refs]:
            if value not in cache:
                cache[value] = np.asarray(encoder.encode(value, normalize_embeddings=True,
                                                          show_progress_bar=False))
        return max(float(cache[text] @ cache[ref]) for ref in refs)

    rows = []
    for key, frozen_row in expected.items():
        item = gold[key[0]]
        alternatives = item.get("evidence_chains") or ([[]] if item.get("answers") == "无法确定" else [])
        if not alternatives:
            raise ValueError(f"No complete gold alternative: {key}")
        references = basic.answer_references(item)
        facts = basic.facts(item, "required_facts")
        truth_refusal = all(not chain for chain in alternatives) and item.get("answers") == "无法确定"
        metrics = {}
        for arm, arm_rows in predictions.items():
            pred = arm_rows[key]
            chain, answer = pred["evidence_chain"], pred["answer"]
            answer_f1 = max((basic.char_f1(answer, ref) for ref in references), default=0)
            metrics[arm] = {
                "answer_char_f1": answer_f1,
                "answer_semantic_cosine": semantic(answer, references),
                "required_fact_coverage_lit": (sum(fact in answer or basic.char_f1(answer, fact) >= .5
                                                    for fact in facts) / len(facts) if facts else None),
                "chain_node_f1": max(basic.set_f1(chain, ref) for ref in alternatives),
                "chain_edge_f1": max(basic.set_f1(edges(chain), edges(ref)) for ref in alternatives),
                "chain_exact": int(chain in alternatives),
                "strict_refusal_correct": int((answer == "无法确定") == truth_refusal),
                "refusal": int(answer == "无法确定"),
                "answer_right_chain_wrong": int(answer_f1 >= .6 and chain not in alternatives),
            }
        rows.append({**frozen_row, "strict_refusal_gold": truth_refusal,
                     "gold_answers": references, "gold_chains": alternatives,
                     "predictions": {a: {"answer": predictions[a][key]["answer"],
                                         "evidence_chain": predictions[a][key]["evidence_chain"]}
                                     for a in predictions}, "metrics": metrics})

    def summarize(subset: list[dict]) -> dict:
        return {arm: {name: mean([r["metrics"][arm][name] for r in subset
                                  if r["metrics"][arm][name] is not None])
                      for name in next(iter(subset))["metrics"][arm]}
                | {"n": len(subset), "n_with_required_facts": sum(
                    r["metrics"][arm]["required_fact_coverage_lit"] is not None for r in subset)}
                for arm in predictions}

    grouped = {"all": rows}
    for view in "ABC":
        grouped[f"view:{view}"] = [r for r in rows if r["view"] == view]
    for flag in ("long_chain", "graph_gap", "multiple_gold_paths",
                 "explanatory_unanswerable", "long_material"):
        grouped[f"risk:{flag}"] = [r for r in rows if flag in r["risk_flags"]]
    grouped["gold:strict_refusal"] = [r for r in rows if r["strict_refusal_gold"]]
    for kind in ("retrospective", "prospective", "counterfactual", "unanswerable"):
        grouped[f"type:{kind}"] = [r for r in rows if r["question_type"] == kind]

    buckets = Counter()
    if "qwen" in predictions and "deepseek" in predictions:
        for row in rows:
            q, d = (row["metrics"][name] for name in ("qwen", "deepseek"))
            def good(m: dict) -> bool:
                return m["answer_char_f1"] >= .6 and bool(m["chain_exact"])
            category = ("both_good" if good(q) and good(d) else
                        "qwen_only_good" if good(q) else
                        "deepseek_only_good" if good(d) else "both_poor")
            row["qwen_deepseek_bucket"] = category
            buckets[category] += 1

    def weighted_proxy(subset: list[dict], arm: str,
                       include_semantic: bool = True) -> float | None:
        total = 0.0
        for view, weight in (("A", .6), ("B", .3), ("C", .1)):
            part = [r["metrics"][arm] for r in subset if r["view"] == view]
            if not part:
                return None
            facts_values = [m["required_fact_coverage_lit"] for m in part
                            if m["required_fact_coverage_lit"] is not None]
            semantic_values = [m["answer_semantic_cosine"] for m in part
                               if m["answer_semantic_cosine"] is not None]
            if not facts_values or (include_semantic and len(semantic_values) != len(part)):
                return None
            score = (25 * sum(facts_values) / len(facts_values)
                     + 10 * sum(m["answer_char_f1"] for m in part) / len(part)
                     + (5 * max(0, min(1, sum(semantic_values) / len(part)))
                        if include_semantic else 0)
                     + 25 * sum(m["chain_node_f1"] for m in part) / len(part)
                     + 7.5 * sum(m["chain_edge_f1"] for m in part) / len(part)
                     + 7.5 * sum(m["chain_exact"] for m in part) / len(part))
            total += weight * score
        return total

    proxy = {arm: weighted_proxy(rows, arm) for arm in predictions}
    proxy_75 = {arm: weighted_proxy(rows, arm, include_semantic=False)
                for arm in predictions}
    paired = {}
    if all(value is not None for value in proxy.values()):
        rng = random.Random(20261008)
        by_view = {view: [r for r in rows if r["view"] == view] for view in "ABC"}
        for left in predictions:
            for right in predictions:
                if left >= right:
                    continue
                observed = proxy[left] - proxy[right]
                boot = []
                for _ in range(10000):
                    sampled = [rng.choice(by_view[view]) for view in "ABC"
                               for _ in range(len(by_view[view]))]
                    lval, rval = weighted_proxy(sampled, left), weighted_proxy(sampled, right)
                    if lval is not None and rval is not None:
                        boot.append(lval - rval)
                boot.sort()
                paired[f"{left}_minus_{right}"] = {
                    "delta": round(observed, 4), "bootstrap_replicates": len(boot),
                    "ci95": [round(boot[int(.025 * len(boot))], 4),
                             round(boot[min(len(boot)-1, int(.975 * len(boot)))], 4)]
                    if boot else None}
    return {"manifest": str(manifest), "arms": list(predictions),
            "metric_note": "Local proxies, not official scores. Each node/edge maximum is over original intact gold alternatives; exact compares whole chains. Answer-right threshold .6 is a prespecified diagnostic flag, not semantic correctness.",
            "semantic_model": embedding_model,
            "weighted_80_proxy": {a: round(v, 4) if v is not None else None
                                  for a, v in proxy.items()},
            "weighted_75_proxy_without_semantic": {
                a: round(v, 4) if v is not None else None for a, v in proxy_75.items()},
            "paired_80_proxy": paired,
            "groups": {name: summarize(group)
                for name, group in grouped.items() if group}, "buckets": dict(buckets),
            "rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "experiments" /
                        "validation_diagnostic_20261008.json")
    parser.add_argument("--train-root", type=Path, default=ROOT / "数据集" / "训练集")
    parser.add_argument("--prediction", action="append", required=True,
                        help="arm=path; use qwen= and deepseek= for four buckets")
    parser.add_argument("--embedding-model")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = dict(entry.split("=", 1) for entry in args.prediction)
    if len(paths) != len(args.prediction):
        raise ValueError("Duplicate arm name")
    if not args.output.resolve().is_relative_to((ROOT / "outputs").resolve()):
        raise ValueError("Gold-bearing report must stay under ignored outputs/")
    report = evaluate(args.manifest, {a: Path(p) for a, p in paths.items()},
                      args.train_root, args.embedding_model)
    core.atomic_json(args.output, report)
    print(json.dumps({k: v for k, v in report.items() if k != "rows"},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
