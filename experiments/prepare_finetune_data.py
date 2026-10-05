"""Build TRAIN-only, pack-grouped SFT data for A/B/C inference views.

Outputs contain organizer training text and gold, so keep JSONL under outputs/.
No blind-test data, API predictions, or confidence pseudo-labels are used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import task4_core as core
import task4_workflow as workflow
from experiments.audit_training_data import normalized, quantiles


SFT_SYSTEM_PROMPT = (
    "你是证据约束事件因果问答系统。只依据用户消息给出的文档、事件和因果关系回答。"
    "只输出含 answer 和 evidence_chain 的 JSON 对象；证据链按原因到结果顺序，"
    "只能使用当前输入视图允许的事件 ID。无证据时输出 answer='无法确定'、evidence_chain=[]。"
    "unanswerable 是题型提示，仍须核查原文，不得按题型一律拒答。"
    "反事实只作有边界推断，不得把时间先后直接当作因果。"
    "数值 confidence 由推理时独立校准器负责，本任务不预测它。"
)


def document_groups(packs: list[Path]) -> list[list[str]]:
    """Keep packs sharing a normalized exact document in one split."""
    parent = {pack.name: pack.name for pack in packs}

    def root(name: str) -> str:
        while parent[name] != name:
            parent[name] = parent[parent[name]]
            name = parent[name]
        return name

    seen: dict[str, str] = {}
    for pack in packs:
        documents, _, _ = core.load_pack(pack)
        for text in documents.values():
            digest = hashlib.sha256(normalized(text).encode("utf-8")).hexdigest()
            if digest in seen:
                parent[root(pack.name)] = root(seen[digest])
            else:
                seen[digest] = pack.name
    groups: dict[str, list[str]] = defaultdict(list)
    for pack in packs:
        groups[root(pack.name)].append(pack.name)
    return [sorted(group) for group in groups.values()]


def make_split(groups: list[list[str]], seed: int) -> dict[str, list[str]]:
    """Deterministic approximate 80/10/10 allocation by domain and pack count."""
    domains: dict[str, list[list[str]]] = defaultdict(list)
    for group in groups:
        # A duplicate group can span domains; its first member fixes allocation.
        domains[group[0].rsplit("_", 1)[0]].append(group)
    splits = {"fit": [], "validation": [], "holdout": []}
    rng = random.Random(seed)
    for domain in sorted(domains):
        shuffled = sorted(domains[domain])
        rng.shuffle(shuffled)
        targets = {"fit": round(sum(map(len, shuffled)) * .8),
                   "validation": round(sum(map(len, shuffled)) * .1)}
        counts = Counter()
        for group in shuffled:
            if counts["fit"] < targets["fit"]:
                name = "fit"
            elif counts["validation"] < targets["validation"]:
                name = "validation"
            else:
                name = "holdout"
            splits[name].extend(group)
            counts[name] += len(group)
    return {name: sorted(values) for name, values in splits.items()}


def flags_for(item: dict, sample: core.Sample, event_ids: set[str],
              doc_ids: set[str], edges: set[tuple[str, str]]) -> list[str]:
    chains = item.get("evidence_chains") or []
    nonempty = [chain for chain in chains if isinstance(chain, list) and chain]
    flags = []
    if any(node not in event_ids for chain in nonempty for node in chain):
        flags.append("missing_event_id")
    if any(node not in doc_ids for chain in nonempty for node in chain):
        flags.append("not_document_id_compatible")
    if any((a, b) not in edges for chain in nonempty for a, b in zip(chain, chain[1:])):
        flags.append("step_absent_from_given_edges")
    if sample.question_type == "unanswerable" and nonempty:
        flags.append("unanswerable_with_evidence")
    if not (item.get("answer_facts") or {}).get("required_facts"):
        flags.append("no_required_facts")
    if len({tuple(chain) for chain in chains}) > 1:
        flags.append("multiple_gold_paths")
    if item.get("confidence_level") not in {"certain", "probable", "possible", None}:
        flags.append("nonstandard_confidence_level")
    return flags


def choose_chain(item: dict, allowed: set[str], question: str,
                 supported_edges: set[tuple[str, str]] | None = None) -> list[str] | None:
    """Choose one intact gold alternative; never splice nodes across paths."""
    chains = item.get("evidence_chains") or []
    if not chains and str(item.get("answers", "")).strip() == "无法确定":
        return []
    alternatives = [chain for chain in chains if isinstance(chain, list)
                    and len(chain) == len(set(chain)) and set(chain) <= allowed]
    if supported_edges is not None:
        alternatives = [chain for chain in alternatives if all(
            (a, b) in supported_edges for a, b in zip(chain, chain[1:]))]
    if not alternatives:
        return None
    if all(not chain for chain in alternatives):
        return []
    alternatives = [chain for chain in alternatives if chain]
    if not alternatives:
        return []
    answer = str(item.get("answers", ""))
    anchors = set(re.findall(r"D\d{3,}", question))
    annotated = (item.get("annotations") or {}).get("causal_path")

    def score(chain: list[str]) -> tuple:
        # Reference path and answer mention order give a reproducible choice.
        positions = [answer.find(node) for node in chain]
        ordered_in_answer = all(position >= 0 for position in positions) and positions == sorted(positions)
        return (chain == annotated, ordered_in_answer,
                len(anchors & set(chain)),
                len(chain) if "完整" in question or "多个阶段" in question else -len(chain),
                tuple(chain))

    return list(max(alternatives, key=score))


def prepare(train_root: Path, output_dir: Path, manifest_path: Path,
            seed: int = 20261005) -> dict:
    if not output_dir.resolve().is_relative_to((core.REPO_ROOT / "outputs").resolve()):
        raise ValueError("Fine-tuning examples must be written under ignored outputs/")
    samples = core.discover_samples(train_root, limit=0)
    packs = sorted({sample.pack for sample in samples})
    groups = document_groups(packs)
    splits = make_split(groups, seed)
    assignment = {pack: split for split, names in splits.items() for pack in names}
    if len(assignment) != len(packs):
        raise AssertionError("some packs missing from split")
    if any(len({assignment[pack] for pack in group}) != 1 for group in groups):
        raise AssertionError("duplicate document group crosses splits")
    fit_qa = set()
    for pack in packs:
        if assignment[pack.name] != "fit":
            continue
        gold_items = {item["sample_id"]: item for item in core.read_json(
            pack / "gold" / "问答对_答案.json")}
        for question in core.read_json(pack / "问题.json"):
            answer = gold_items[question["sample_id"]].get("answers")
            if isinstance(answer, str):
                fit_qa.add((normalized(question["question"]), normalized(answer)))
    manifest = {"seed": seed, "scope": "organizer train packs only",
                "normalized_exact_document_group_count": len(groups),
                "splits": splits}
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                             encoding="utf-8")
    output_dir.mkdir(parents=True, exist_ok=True)
    handles = {(split, view): (output_dir / f"{split}_{view}.jsonl").open("w", encoding="utf-8")
               for split in splits for view in "ABC"}
    by_pack: dict[Path, list[core.Sample]] = defaultdict(list)
    for sample in samples:
        by_pack[sample.pack].append(sample)
    counts = Counter()
    flag_counts = Counter()
    input_lengths = defaultdict(list)
    quarantined = []
    try:
        for pack in packs:
            documents, events, relations = core.load_pack(pack)
            event_ids = {event["event_id"] for event in events or []}
            doc_ids = set(documents)
            edges = {(edge.get("cause_event_id"), edge.get("result_event_id", edge.get("effect_event_id")))
                     for edge in relations or []}
            gold = {item["sample_id"]: item for item in core.read_json(
                pack / "gold" / "问答对_答案.json")}
            for sample in by_pack[pack]:
                item = gold[sample.sample_id]
                flags = flags_for(item, sample, event_ids, doc_ids, edges)
                flag_counts.update(flags)
                split = assignment[pack.name]
                for view in "ABC":
                    content, allowed, _ = workflow.training_view_input(sample, view)
                    input_lengths[view].append(len(content))
                    chosen = choose_chain(item, allowed, sample.question,
                                          edges if view == "A" else None)
                    answer = item.get("answers")
                    reason = None
                    if chosen is None:
                        reason = ("no_supported_visible_graph_path" if view == "A"
                                  else "no_compatible_gold_path")
                    elif not isinstance(answer, str) or not answer.strip():
                        reason = "invalid_answer"
                    elif (answer.strip() == "无法确定") != (chosen == []):
                        reason = "answer_chain_refusal_conflict"
                    elif split != "fit" and (normalized(sample.question),
                                               normalized(answer)) in fit_qa:
                        reason = "same_question_answer_as_fit"
                    if reason:
                        quarantined.append({"pack": pack.name, "sample_id": sample.sample_id,
                                            "split": split, "view": view, "reason": reason,
                                            "flags": flags})
                        counts[f"quarantine_{view}"] += 1
                        continue
                    row = {"pack": pack.name, "sample_id": sample.sample_id,
                           "view": view, "question_type": sample.question_type,
                           "messages": [{"role": "system", "content": SFT_SYSTEM_PROMPT},
                                        {"role": "user", "content": content},
                                        {"role": "assistant", "content": json.dumps(
                                            {"answer": answer, "evidence_chain": chosen},
                                            ensure_ascii=False)}],
                           "selection": "one_intact_gold_alternative",
                           "gold_alternative_count": len(item.get("evidence_chains") or []),
                           "flags": flags}
                    handles[split, view].write(json.dumps(row, ensure_ascii=False) + "\n")
                    counts[f"{split}_{view}"] += 1
    finally:
        for handle in handles.values():
            handle.close()
    core.atomic_json(output_dir / "quarantine.json", quarantined)
    summary = {"packs": len(packs), "questions": len(samples),
               "split_pack_counts": {name: len(names) for name, names in splits.items()},
               "duplicate_document_groups": sum(len(group) > 1 for group in groups),
               "examples": dict(sorted(counts.items())),
               "flags_by_question": dict(sorted(flag_counts.items())),
               "input_character_quantiles_by_view": {
                   view: quantiles(input_lengths[view]) for view in "ABC"},
               "quarantine_count": len(quarantined),
               "quarantine_reasons_by_view": dict(sorted(Counter(
                   f"{row['view']}:{row['reason']}" for row in quarantined).items())),
               "note": "A/B/C are separate input views. Targets omit numeric confidence. Structurally incompatible labels and validation/holdout exact question+answer matches to fit are quarantined; question-only templates are audited, not called answer leaks."}
    core.atomic_json(output_dir / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-root", type=Path,
                        default=core.REPO_ROOT / "数据集" / "训练集")
    parser.add_argument("--output-dir", type=Path,
                        default=core.REPO_ROOT / "outputs" / "finetune_2026-10-05")
    parser.add_argument("--manifest", type=Path,
                        default=core.REPO_ROOT / "experiments" / "finetune_split_20261005.json")
    parser.add_argument("--seed", type=int, default=20261005)
    args = parser.parse_args()
    print(json.dumps(prepare(args.train_root, args.output_dir, args.manifest, args.seed),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
