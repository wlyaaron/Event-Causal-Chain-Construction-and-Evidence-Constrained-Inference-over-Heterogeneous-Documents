"""Audit TRAIN-only supervision before constructing fine-tuning examples.

No test files, external data, API predictions, or private credentials are read.
The report contains counts and sample IDs, never source document text.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import task4_core as core


def normalized(value: str) -> str:
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", value)).casefold()


def quantiles(values: list[int]) -> dict[str, int]:
    ordered = sorted(values)
    if not ordered:
        return {}
    return {str(percentile): ordered[round((len(ordered) - 1) * percentile / 100)]
            for percentile in (0, 50, 90, 95, 99, 100)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-root", type=Path,
                        default=core.REPO_ROOT / "数据集" / "训练集")
    parser.add_argument("--output", type=Path,
                        default=core.REPO_ROOT / "outputs" / "training_data_audit.json")
    args = parser.parse_args()
    samples = core.discover_samples(args.train_root, limit=0)
    by_pack = defaultdict(list)
    for sample in samples:
        by_pack[sample.pack].append(sample)
    question_types = Counter()
    confidence_levels = Counter()
    answer_forms = Counter()
    anomaly_counts = Counter()
    anomaly_examples = defaultdict(list)
    prompt_lengths = []
    answer_lengths = []
    chain_lengths = []
    alternative_counts = []
    document_hashes = defaultdict(set)
    question_hashes = defaultdict(set)
    domains = Counter()
    compatible_document_chain_questions = 0
    nonempty_chain_questions = 0
    for pack, pack_samples in sorted(by_pack.items()):
        domains[pack.name.rsplit("_", 1)[0]] += 1
        documents, events, edges = core.load_pack(pack)
        event_ids = {event.get("event_id") for event in events or []
                     if isinstance(event, dict) and isinstance(event.get("event_id"), str)}
        edge_ids = {(edge.get("cause_event_id"),
                     edge.get("result_event_id", edge.get("effect_event_id")))
                    for edge in edges or [] if isinstance(edge, dict)}
        for doc_id, document in documents.items():
            digest = hashlib.sha256(normalized(document).encode("utf-8")).hexdigest()
            document_hashes[digest].add(pack.name)
        gold_path = pack / "gold" / "问答对_答案.json"
        if not gold_path.exists():
            raise ValueError(f"缺少训练金标：{gold_path}")
        gold = {item["sample_id"]: item for item in core.read_json(gold_path)}
        for sample in pack_samples:
            if sample.sample_id not in gold:
                raise ValueError(f"训练问题缺少金标：{sample.sample_id}")
            item = gold[sample.sample_id]
            question_types[sample.question_type] += 1
            question_hashes[hashlib.sha256(normalized(sample.question).encode("utf-8"))
                            .hexdigest()].add(pack.name)
            prompt_lengths.append(len(core.build_input(sample, max_input_chars=1_000_000)[0]))
            answer = item.get("answers", item.get("answer"))
            answer_forms[type(answer).__name__] += 1
            if isinstance(answer, str):
                answer_lengths.append(len(answer))
            elif isinstance(answer, list) and answer and isinstance(answer[0], str):
                answer_lengths.append(len(answer[0]))
            else:
                anomaly_counts["missing_or_invalid_answer"] += 1
            confidence_levels[str(item.get("confidence_level"))] += 1
            chains = item.get("evidence_chains") or []
            if not isinstance(chains, list):
                chains = []
                anomaly_counts["invalid_evidence_chains_type"] += 1
            alternative_counts.append(len(chains))
            valid_chains = [chain for chain in chains
                            if isinstance(chain, list)
                            and all(isinstance(node, str) for node in chain)]
            if len(valid_chains) != len(chains):
                anomaly_counts["invalid_chain_shape"] += 1
            nonempty = [chain for chain in valid_chains if chain]
            if nonempty:
                nonempty_chain_questions += 1
                if all(node in documents for chain in nonempty for node in chain):
                    compatible_document_chain_questions += 1
            chain_lengths.extend(map(len, nonempty))
            def note(key: str) -> None:
                anomaly_counts[key] += 1
                if len(anomaly_examples[key]) < 10:
                    anomaly_examples[key].append(sample.sample_id)
            if any(node not in event_ids for chain in nonempty for node in chain):
                note("chain_refers_to_missing_event")
            if any((left, right) not in edge_ids
                   for chain in nonempty for left, right in zip(chain, chain[1:])):
                note("chain_has_step_absent_from_given_edges")
            if any(len(chain) != len(set(chain)) for chain in nonempty):
                note("chain_repeats_event")
            if len({tuple(chain) for chain in valid_chains}) != len(valid_chains):
                note("duplicate_gold_chain_alternatives")
            if sample.question_type == "unanswerable" and nonempty:
                note("unanswerable_type_with_nonempty_chain")
            if isinstance(answer, str) and answer.strip() == "无法确定" and nonempty:
                note("exact_refusal_with_nonempty_chain")
            if isinstance(answer, str) and answer.strip() != "无法确定" and not nonempty:
                note("non_refusal_without_chain")
            required = (item.get("answer_facts") or {}).get("required_facts") or []
            if not required:
                note("no_required_facts")
    duplicated_docs = {digest: names for digest, names in document_hashes.items()
                       if len(names) > 1}
    repeated_questions = {digest: names for digest, names in question_hashes.items()
                          if len(names) > 1}
    report = {
        "scope": "TRAIN ONLY; normalized exact duplicates, not semantic near duplicates",
        "packs": len(by_pack), "questions": len(samples),
        "domains_packs": dict(sorted(domains.items())),
        "question_types": dict(sorted(question_types.items())),
        "confidence_levels": dict(sorted(confidence_levels.items())),
        "answer_field_types": dict(sorted(answer_forms.items())),
        "prompt_character_quantiles": quantiles(prompt_lengths),
        "answer_character_quantiles": quantiles(answer_lengths),
        "nonempty_gold_chain_length_quantiles": quantiles(chain_lengths),
        "gold_chain_alternatives_quantiles": quantiles(alternative_counts),
        "questions_with_nonempty_gold_chain": nonempty_chain_questions,
        "questions_whose_all_gold_nodes_are_document_ids":
            compatible_document_chain_questions,
        "cross_pack_normalized_exact_duplicate_document_groups": len(duplicated_docs),
        "packs_in_exact_duplicate_document_groups": len(set().union(*duplicated_docs.values()))
            if duplicated_docs else 0,
        "cross_pack_exact_question_text_groups": len(repeated_questions),
        "packs_in_exact_question_text_groups": len(set().union(*repeated_questions.values()))
            if repeated_questions else 0,
        "anomaly_counts_by_question": dict(sorted(anomaly_counts.items())),
        "anomaly_example_ids": dict(sorted(anomaly_examples.items())),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
