"""TRAIN-only learned event retriever with pack-grouped validation.

The model ranks candidate evidence IDs; it never predicts final answers. Gold
chains are read only while fitting and evaluating this small retriever.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

import numpy as np
import joblib
from sklearn.ensemble import HistGradientBoostingClassifier

import run_baselines as baseline
import task4_core as core


KINDS = ("retrospective", "prospective", "counterfactual", "unanswerable")


def features(sample: core.Sample, pack: baseline.Pack,
             graph_visible: bool,
             ablate: str = "none") -> tuple[list[str], np.ndarray, list[float]]:
    candidates = pack.candidates
    ids = [candidate.event_id for candidate in candidates]
    texts = [candidate.text for candidate in candidates]
    lexical = baseline.bm25_scores(sample.question, texts)
    peak = max(lexical, default=0.0) or 1.0
    ranks = {index: rank for rank, index in enumerate(
        sorted(range(len(ids)), key=lambda index: lexical[index], reverse=True))}
    explicit = set(baseline.ID_PATTERN.findall(sample.question)) & set(ids)
    outgoing, incoming = baseline.directed_graph(pack) if graph_visible else ({}, {})
    question_tokens = set(baseline.tokens(sample.question))
    rows = []
    for index, candidate in enumerate(candidates):
        label_tokens = set(baseline.tokens(candidate.label))
        title_overlap = len(question_tokens & label_tokens) / max(1, len(label_tokens))
        event_tokens = set(baseline.tokens(candidate.text[:600]))
        text_overlap = len(question_tokens & event_tokens) / max(1, len(question_tokens))
        number = re.search(r"\d+", candidate.event_id)
        normalized_number = (int(number.group()) / max(1, len(candidates))
                             if number else index / max(1, len(candidates)))
        row = [
            lexical[index] / peak,
            ranks[index] / max(1, len(candidates) - 1),
            float(candidate.event_id in explicit),
            title_overlap,
            text_overlap,
            normalized_number,
            min(len(candidate.text), 1200) / 1200,
            len(explicit) / max(1, len(candidates)),
            min(len(sample.question), 500) / 500,
            float(graph_visible),
            len(outgoing.get(candidate.event_id, [])) / max(1, len(candidates)),
            len(incoming.get(candidate.event_id, [])) / max(1, len(candidates)),
            *(float(sample.question_type == kind) for kind in KINDS),
        ]
        if ablate in {"id_position", "id_position_and_type"}:
            row[2] = 0.0
            row[5] = 0.0
        if ablate == "id_position_and_type":
            row[-len(KINDS):] = [0.0] * len(KINDS)
        rows.append(row)
    return ids, np.asarray(rows, dtype=np.float32), lexical


def best_chain_recall(selected: set[str], chains: list[list[str]]) -> float:
    return max((len(selected & set(chain)) / len(chain) for chain in chains if chain),
               default=0.0)


def summarize(rows: list[dict], name: str) -> dict:
    answerable = [row for row in rows if row["chains"]]
    result = {"samples": len(rows), "with_nonempty_gold_chain": len(answerable)}
    for k in (3, 5):
        recalls = [best_chain_recall(set(row[name][:k]), row["chains"])
                   for row in answerable]
        result[f"top{k}_mean_best_gold_recall"] = (
            round(sum(recalls) / len(recalls), 4) if recalls else None)
        result[f"top{k}_full_gold_chain_rate"] = (
            round(sum(value == 1 for value in recalls) / len(recalls), 4)
            if recalls else None)
    return result


def document_only_pack(pack: baseline.Pack) -> baseline.Pack:
    """Simulate track C by hiding event and edge files during retrieval."""
    candidates = []
    for doc_id, document in pack.documents.items():
        title = next((line.strip(" #：:。\t") for line in document.splitlines()
                      if line.strip()), doc_id)
        candidates.append(baseline.Candidate(doc_id, doc_id, title[:60],
                                              document[:900]))
    return baseline.Pack(pack.documents, candidates, [], None)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-root", type=Path,
                        default=core.REPO_ROOT / "数据集" / "训练集")
    parser.add_argument("--output", type=Path,
                        default=core.REPO_ROOT / "outputs" / "learned_retriever_audit.json")
    parser.add_argument("--seed", type=int, default=20261004)
    parser.add_argument("--validation-domain", type=str,
                        help="Hold out all packs with this Chinese domain prefix")
    parser.add_argument("--ablate", choices=["none", "id_position",
                                            "id_position_and_type"],
                        default="none")
    parser.add_argument("--model-output", type=Path,
                        help="Optional local joblib checkpoint; do not load untrusted files")
    parser.add_argument("--include-doc-only", action="store_true",
                        help="Also simulate track C with only source documents")
    args = parser.parse_args()
    samples = core.discover_samples(args.train_root, limit=0)
    packs = sorted({sample.pack for sample in samples})
    rng = random.Random(args.seed)
    rng.shuffle(packs)
    if args.validation_domain:
        validation_packs = {pack for pack in packs
                            if pack.name.split("_")[0] == args.validation_domain}
        if not validation_packs:
            raise ValueError(f"No packs in domain {args.validation_domain}")
        train_packs = set(packs) - validation_packs
    else:
        train_packs = set(packs[:800])
        validation_packs = set(packs[800:])
    gold = {}
    for path in args.train_root.rglob("gold/问答对_答案.json"):
        for item in core.read_json(path):
            gold[item["sample_id"]] = item

    fit_x = []
    fit_y = []
    validation = []
    cached_packs = {}
    for index, sample in enumerate(samples, start=1):
        if sample.pack not in cached_packs:
            cached_packs[sample.pack] = baseline.read_pack(sample.pack)
        pack = cached_packs[sample.pack]
        chains = gold[sample.sample_id].get("evidence_chains") or []
        positive = {node for chain in chains for node in chain}
        views = [("with_supplied_graph", pack, True),
                 ("graph_hidden", pack, False)]
        if args.include_doc_only:
            views.append(("documents_only", document_only_pack(pack), False))
        for view, view_pack, graph_visible in views:
            ids, matrix, lexical = features(sample, view_pack, graph_visible,
                                            args.ablate)
            if sample.pack in train_packs:
                fit_x.append(matrix)
                fit_y.extend(int(event_id in positive) for event_id in ids)
            else:
                bm25 = [ids[i] for i in sorted(range(len(ids)),
                         key=lambda i: lexical[i], reverse=True)]
                validation.append({"sample_id": sample.sample_id,
                                   "question_type": sample.question_type,
                                   "view": view,
                                   "ids": ids, "matrix": matrix,
                                   "bm25": bm25, "chains": chains})
        if index % 1000 == 0:
            print(f"prepared {index}/{len(samples)} questions", flush=True)

    model = HistGradientBoostingClassifier(max_iter=120, max_leaf_nodes=15,
                                            learning_rate=0.07,
                                            l2_regularization=1.0,
                                            random_state=args.seed)
    x = np.concatenate(fit_x)
    y = np.asarray(fit_y)
    model.fit(x, y)
    for row in validation:
        scores = model.predict_proba(row.pop("matrix"))[:, 1]
        row["learned"] = [row["ids"][index] for index in
                          np.argsort(-scores)]
        row.pop("ids")

    results = {}
    view_names = ("with_supplied_graph", "graph_hidden", "documents_only") \
        if args.include_doc_only else ("with_supplied_graph", "graph_hidden")
    for key in view_names:
        subset = [row for row in validation if row["view"] == key]
        results[key] = {"bm25": summarize(subset, "bm25"),
                        "learned": summarize(subset, "learned"),
                        "by_question_type": {
                            kind: {"bm25": summarize([row for row in subset
                                                      if row["question_type"] == kind], "bm25"),
                                   "learned": summarize([row for row in subset
                                                        if row["question_type"] == kind], "learned")}
                            for kind in KINDS
                            if any(row["question_type"] == kind for row in subset)}}
    report = {"scope": "training set only; grouped holdout",
              "seed": args.seed, "train_packs": len(train_packs),
              "validation_packs": len(validation_packs),
              "validation_domain": args.validation_domain,
              "ablation": args.ablate,
              "include_doc_only": args.include_doc_only,
              "feature_count": x.shape[1],
              "graph_hidden_view": "All given causal edges masked from feature extraction",
              "results": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    if args.model_output:
        args.model_output.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": model, "ablation": args.ablate,
                     "seed": args.seed,
                     "train_pack_names": sorted(pack.name for pack in train_packs),
                     "validation_pack_names": sorted(pack.name for pack in validation_packs)},
                    args.model_output)
    print(json.dumps({key: results[key] for key in results},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
