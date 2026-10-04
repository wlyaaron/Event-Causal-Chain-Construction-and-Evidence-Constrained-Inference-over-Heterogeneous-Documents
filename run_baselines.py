"""Reproducible task-4 baselines: graph, BM25+BERT, RAG, and local Llama.

These are independently specified examples. The organizers publish scores and
method names, but not the implementations that produced their score table.
No gold answer is read for inference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

import task4_api as api
import task4_core as common


ROOT = Path(__file__).resolve().parent
DEFAULT_EMBEDDING = ROOT / "models" / "bge-small-zh-v1.5"
ID_PATTERN = re.compile(r"D\d{3,}")
TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]")
CAUSAL_WORDS = ("导致", "引发", "造成", "推动", "促成", "触发", "使得")


@dataclass(frozen=True)
class Candidate:
    event_id: str
    doc_id: str
    label: str
    text: str


@dataclass
class Pack:
    documents: dict[str, str]
    candidates: list[Candidate]
    edges: list[dict]
    events: list[dict] | None


def read_pack(path: Path) -> Pack:
    documents, events, edges = common.load_pack(path)
    candidates: list[Candidate] = []
    if events is not None:
        for event in events:
            event_id = event["event_id"]
            doc_id = event.get("doc_id", event_id)
            label = str(event.get("event_type") or event.get("trigger_word") or event_id)
            arguments = event.get("argument") or {}
            argument_text = "；".join(str(v) for v in arguments.values()) if isinstance(arguments, dict) else str(arguments)
            text = f"{event_id} {label} {event.get('trigger_word', '')} {argument_text[:450]} {documents.get(doc_id, '')[:450]}"
            candidates.append(Candidate(event_id, doc_id, label, text))
    else:
        for doc_id, document in documents.items():
            title = next((s.strip(" #：:。\t") for s in document.splitlines() if s.strip()), doc_id)
            candidates.append(Candidate(doc_id, doc_id, title[:60], document[:900]))
    return Pack(documents, candidates, edges or [], events)


def tokens(text: str) -> list[str]:
    """Character 1/2-grams support Chinese without a tokenizer dependency."""
    raw = TOKEN_PATTERN.findall(text.lower())
    result = list(raw)
    result.extend(raw[i] + raw[i + 1] for i in range(len(raw) - 1)
                  if len(raw[i]) == 1 and len(raw[i + 1]) == 1)
    return result


def bm25_scores(query: str, texts: list[str], k1: float = 1.2,
                b: float = 0.75) -> list[float]:
    if not texts:
        return []
    corpus = [Counter(tokens(t)) for t in texts]
    lengths = [sum(c.values()) for c in corpus]
    avg_length = sum(lengths) / len(lengths) or 1.0
    df = Counter(term for document in corpus for term in document)
    q_terms = set(tokens(query))
    scores = []
    for document, length in zip(corpus, lengths):
        score = 0.0
        for term in q_terms:
            frequency = document.get(term, 0)
            if not frequency:
                continue
            idf = math.log(1 + (len(corpus) - df[term] + 0.5) / (df[term] + 0.5))
            score += idf * frequency * (k1 + 1) / (frequency + k1 * (1 - b + b * length / avg_length))
        scores.append(score)
    return scores


class BertEncoder:
    """BERT architecture, locally loaded open BGE-small-zh-v1.5 weights."""

    def __init__(self, path: Path):
        if not path.is_dir():
            raise ValueError(f"本地 BERT 权重目录不存在：{path}")
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(str(path), device="cpu")
        self.cache: dict[tuple[str, ...], object] = {}

    def scores(self, query: str, texts: list[str]) -> list[float]:
        import numpy as np
        key = tuple(texts)
        if key not in self.cache:
            self.cache[key] = self.model.encode(texts, batch_size=8,
                                                normalize_embeddings=True, show_progress_bar=False)
        question = self.model.encode(query, normalize_embeddings=True,
                                     show_progress_bar=False)
        return [float(x) for x in np.asarray(self.cache[key]) @ question]


class LocalHFGenerator:
    """Offline open-weight chat generation without a remote API server."""

    def __init__(self, path: Path, context_tokens: int):
        if not path.is_dir():
            raise ValueError(f"本地生成模型权重目录不存在：{path}")
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        torch.set_num_threads(min(torch.get_num_threads(), 4))
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            str(path), local_files_only=True, torch_dtype=torch.float32).eval()
        self.model.generation_config.temperature = 1.0
        self.model.generation_config.top_p = 1.0
        self.model.generation_config.top_k = 50
        self.context_tokens = context_tokens

    def generate(self, content: str, max_new_tokens: int) -> str:
        messages = [{"role": "system", "content": common.SYSTEM_PROMPT},
                    {"role": "user", "content": content}]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False,
                                                    add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        length = inputs["input_ids"].shape[-1]
        if length + max_new_tokens > self.context_tokens:
            raise ValueError(f"本地模型输入 {length} token 加输出 {max_new_tokens} 超过 --hf-context-tokens={self.context_tokens}；请减小 --top-k 或提高上下文上限")
        with self.torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=max_new_tokens,
                                         do_sample=False, pad_token_id=self.tokenizer.eos_token_id)
        return self.tokenizer.decode(output[0][length:], skip_special_tokens=True)


def rank_candidates(question: str, pack: Pack, encoder: BertEncoder | None,
                    hybrid: bool) -> list[tuple[Candidate, float]]:
    texts = [candidate.text for candidate in pack.candidates]
    lexical = bm25_scores(question, texts)
    maximum = max(lexical, default=0) or 1.0
    lexical = [score / maximum for score in lexical]
    if hybrid:
        if encoder is None:
            raise ValueError("BM25+BERT 需要本地 BERT 权重")
        semantic = encoder.scores(question, texts)
        # Cosine similarity is used for ranking, not treated as calibrated confidence.
        ranking = [0.45 * a + 0.55 * max(0.0, b) for a, b in zip(lexical, semantic)]
    else:
        ranking = lexical
    return sorted(zip(pack.candidates, ranking), key=lambda item: item[1], reverse=True)


def directed_graph(pack: Pack) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    ids = {candidate.event_id for candidate in pack.candidates}
    outgoing = {name: [] for name in sorted(ids)}
    incoming = {name: [] for name in sorted(ids)}
    for edge in pack.edges:
        cause = edge.get("cause_event_id")
        result = edge.get("result_event_id", edge.get("effect_event_id"))
        if cause in ids and result in ids and cause != result:
            outgoing[cause].append(result)
            incoming[result].append(cause)
    return outgoing, incoming


def paths_to(target: str, outgoing: dict[str, list[str]],
             incoming: dict[str, list[str]], max_nodes: int = 12) -> list[list[str]]:
    roots = [name for name in outgoing if not incoming[name]] or list(outgoing)
    paths: list[list[str]] = []

    def visit(node: str, path: list[str]) -> None:
        if node == target:
            paths.append(path)
            return
        if len(path) >= max_nodes:
            return
        for nxt in outgoing[node]:
            if nxt not in path:
                visit(nxt, [*path, nxt])

    for root in roots:
        visit(root, [root])
    return paths


def select_chain(question: str, pack: Pack,
                 ranking: list[tuple[Candidate, float]]) -> list[str]:
    if not ranking:
        return []
    ids = {candidate.event_id for candidate in pack.candidates}
    explicit = [name for name in ID_PATTERN.findall(question) if name in ids]
    scores = {candidate.event_id: score for candidate, score in ranking}
    outgoing, incoming = directed_graph(pack)
    edge_strength = {}
    for edge in pack.edges:
        pair = (edge.get("cause_event_id"), edge.get("result_event_id", edge.get("effect_event_id")))
        edge_strength[pair] = {"certain": 1.0, "probable": 0.7, "possible": 0.4}.get(edge.get("confidence_level"), 0.7)

    def path_score(path: list[str]) -> tuple[float, float]:
        strength = sum(edge_strength.get((a, b), 0.5) for a, b in zip(path, path[1:]))
        relevance = sum(scores.get(node, 0) for node in path) / len(path)
        return strength, relevance

    if pack.edges and explicit:
        target = explicit[-1]
        if "直接原因" in question and incoming[target]:
            parent = max(incoming[target], key=lambda name: scores.get(name, 0))
            return [parent, target]
        if ("后续" in question or "影响" in question or "引发" in question) and outgoing[target]:
            child = max(outgoing[target], key=lambda name: scores.get(name, 0))
            return [target, child]
        if len(explicit) >= 2 and explicit[0] != target:
            candidates = [p for p in paths_to(target, outgoing, incoming) if explicit[0] in p]
            if candidates:
                return max(candidates, key=lambda p: (len(p), *path_score(p)))
        candidates = paths_to(target, outgoing, incoming)
        if candidates:
            if any(word in question for word in ("自起点", "一步步", "完整", "多跳", "依次")):
                return max(candidates, key=lambda p: (len(p), *path_score(p)))
            return max(candidates, key=lambda p: (sum(scores.get(x, 0) for x in p) / len(p), -len(p)))
    if pack.edges:
        candidate = ranking[0][0].event_id
        if incoming[candidate]:
            return [max(incoming[candidate], key=lambda name: scores.get(name, 0)), candidate]
        if outgoing[candidate]:
            return [candidate, max(outgoing[candidate], key=lambda name: scores.get(name, 0))]
    # B now has event lists but no known causal edges; C has neither.
    # Retrieval alone does not prove a causal edge.
    return [explicit[-1]] if explicit else [ranking[0][0].event_id]


def evidence_sentences(question: str, pack: Pack, chain: list[str],
                       max_chars: int = 420,
                       encoder: BertEncoder | None = None) -> list[str]:
    """Extract source sentences from documents attached to the selected events."""
    lookup = {candidate.event_id: candidate for candidate in pack.candidates}
    selected: list[str] = []
    used: set[str] = set()
    for event_id in chain:
        candidate = lookup[event_id]
        document = pack.documents.get(candidate.doc_id, "")
        sentences = [part.strip() for part in re.split(r"(?<=[。！？；;])|\n+", document)
                     if part.strip()]
        sentences = [part for part in sentences if 8 <= len(part) <= 250] or sentences
        if not sentences:
            continue
        scores = bm25_scores(question + " " + candidate.label, sentences)
        if encoder is not None:
            maximum = max(scores, default=0) or 1.0
            semantic = encoder.scores(question, sentences)
            scores = [0.4 * lexical / maximum + 0.6 * max(0.0, meaning)
                      for lexical, meaning in zip(scores, semantic)]
        best = max(range(len(sentences)), key=lambda i: scores[i])
        sentence = sentences[best]
        if sentence not in used and sum(map(len, selected)) + len(sentence) <= max_chars:
            selected.append(sentence)
            used.add(sentence)
    return selected


def template_prediction(sample: common.Sample, pack: Pack,
                        ranking: list[tuple[Candidate, float]],
                        encoder: BertEncoder | None = None) -> dict:
    if sample.question_type == "unanswerable":
        raw = {"answer": "无法确定", "evidence_chain": [], "confidence": None}
    else:
        chain = select_chain(sample.question, pack, ranking)
        labels = {candidate.event_id: candidate.label for candidate in pack.candidates}
        excerpts = evidence_sentences(sample.question, pack, chain, encoder=encoder)
        factual_text = " ".join(excerpts)
        if not chain:
            raw = {"answer": "无法确定", "evidence_chain": [], "confidence": None}
        elif len(chain) == 1:
            node = chain[0]
            raw = {"answer": factual_text or labels[node],
                   "evidence_chain": chain, "confidence": 0.5}
        else:
            path = " → ".join(f"{name}（{labels[name]}）" for name in chain)
            strength = {(e.get("cause_event_id"), e.get("result_event_id", e.get("effect_event_id"))):
                        e.get("confidence_level") for e in pack.edges}
            levels = [strength.get((a, b), "possible") for a, b in zip(chain, chain[1:])]
            confidence = (0.9 if levels and all(level == "certain" for level in levels)
                          else 0.7 if levels and all(level in {"certain", "probable"} for level in levels)
                          else 0.5)
            if sample.question_type == "counterfactual":
                answer = (f"若题述原因未发生，给定因果链 {path} 的对应传导将失去依据；"
                          "其它因素可能造成的结果无法仅凭现有材料确定。")
            else:
                answer = f"{factual_text} 据给定因果关系，事件顺序为：{path}。".strip()
            raw = {"answer": answer,
                   "evidence_chain": chain, "confidence": confidence}
    allowed = {candidate.event_id for candidate in pack.candidates}
    return common.parse_prediction(json.dumps(raw, ensure_ascii=False), sample, allowed)


def rag_input(sample: common.Sample, pack: Pack,
              ranking: list[tuple[Candidate, float]], top_k: int) -> str:
    selected_ids = {candidate.doc_id for candidate, _ in ranking[:top_k]}
    chain_ids = select_chain(sample.question, pack, ranking)
    lookup = {candidate.event_id: candidate for candidate in pack.candidates}
    selected_ids.update(lookup[event_id].doc_id for event_id in chain_ids)
    # Explicitly referenced documents must not be dropped by retrieval.
    selected_ids.update(name for name in ID_PATTERN.findall(sample.question) if name in pack.documents)
    payload = {
        "track": sample.track, "sample_id": sample.sample_id,
        "question_type": sample.question_type, "question": sample.question,
        "retrieved_documents": [{"doc_id": name, "text": text}
                                for name, text in pack.documents.items() if name in selected_ids],
        "events": pack.events,
        "causal_relations": pack.edges or None,
        "candidate_chain": chain_ids,
        "retrieval_note": "只可用已给材料。未检索到的文档不能凭记忆补全；若关键证据不足，请拒答。",
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def run(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="赛题4可复现方法示例；非主办方原始基线代码")
    parser.add_argument("--method", choices=["graph", "bm25_bert", "rag", "llama3"], required=True)
    parser.add_argument("--dataset", type=Path, default=common.DEFAULT_DATASET)
    parser.add_argument("--track", choices=["all", "A", "B", "C", "train"], default="all")
    parser.add_argument("--limit", type=int, default=3, help="0 表示所选范围的全部问题")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--embedding-model", type=Path, default=DEFAULT_EMBEDDING)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--api-url")
    parser.add_argument("--model")
    parser.add_argument("--api-key-file", type=Path)
    parser.add_argument("--local-hf-model", type=Path,
                        help="离线 Hugging Face 聊天模型权重目录；与 --api-url 二选一")
    parser.add_argument("--hf-context-tokens", type=int, default=8192)
    parser.add_argument("--hf-max-new-tokens", type=int, default=512)
    parser.add_argument("--no-json-mode", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--validate", type=Path, help="只检查已有输出的格式、证据 ID 和问题覆盖率")
    args = parser.parse_args(argv)
    try:
        samples = common.discover_samples(args.dataset, args.track, args.limit)
        if args.validate:
            n = common.validate_file(args.validate, samples, 80000)
            print(f"格式与覆盖率校验通过：{n} 题")
            return 0
        if args.top_k < 1:
            raise ValueError("--top-k 必须至少为 1")
        output = args.output or ROOT / "outputs" / f"{args.method}.json"
        encoder = BertEncoder(args.embedding_model) if args.method == "bm25_bert" else None
        endpoint = key = model = None
        local_generator = None
        if args.method in {"rag", "llama3"}:
            if args.local_hf_model:
                if args.api_url:
                    raise ValueError("--local-hf-model 与 --api-url 不能同时使用")
                local_generator = LocalHFGenerator(args.local_hf_model, args.hf_context_tokens)
                model = str(args.local_hf_model.resolve())
                endpoint = "offline-transformers"
            else:
                if args.method == "llama3" and not args.api_url:
                    raise ValueError("llama3 需要已启动的本地兼容接口或 --local-hf-model")
                endpoint, key, model = api.read_api_settings(args)
                if args.method == "llama3" and urlsplit(endpoint).hostname not in {"localhost", "127.0.0.1", "::1"}:
                    raise ValueError("llama3 方法须连接本地部署的开源权重")
        signature = hashlib.sha256("\n".join(s.sample_id for s in samples).encode()).hexdigest()
        metadata = {"method": args.method, "dataset": str(args.dataset.resolve()),
                    "track": args.track, "limit": args.limit, "sample_ids_sha256": signature,
                    "embedding_model": str(args.embedding_model.resolve()) if encoder else None,
                    "endpoint": endpoint, "model": model, "top_k": args.top_k,
                    "json_mode": not args.no_json_mode,
                    "max_tokens": args.max_tokens if local_generator is None else None,
                    "hf_context_tokens": args.hf_context_tokens if local_generator else None,
                    "hf_max_new_tokens": args.hf_max_new_tokens if local_generator else None}
        store = common.PredictionStore(output, samples, metadata, overwrite=args.overwrite)
        cache: dict[Path, Pack] = {}
        for index, sample in enumerate(samples[len(store.completed):], len(store.completed) + 1):
            if sample.pack not in cache:
                cache[sample.pack] = read_pack(sample.pack)
            pack = cache[sample.pack]
            ranking = rank_candidates(sample.question, pack, encoder,
                                      hybrid=args.method == "bm25_bert")
            if args.method in {"graph", "bm25_bert"}:
                record = template_prediction(sample, pack, ranking, encoder)
            else:
                content = (rag_input(sample, pack, ranking, args.top_k)
                           if args.method == "rag" else common.build_input(sample)[0])
                allowed = {candidate.event_id for candidate in pack.candidates}
                for correction in range(2):
                    raw = (local_generator.generate(content, args.hf_max_new_tokens)
                           if local_generator else
                           api.call_model(endpoint, key, model, content,
                                             args.timeout, args.retries, args.max_tokens,
                                             json_mode=not args.no_json_mode))
                    try:
                        record = common.parse_prediction(raw, sample, allowed)
                        break
                    except ValueError as exc:
                        if correction:
                            raise
                        content += ("\n\n上一版输出未通过提交格式校验：" + str(exc)
                                    + "。请重新输出 JSON；可回答时必须给出真实有序事件 ID 证据链。"
                                      "若仅能说时间先后、缺少可核查的因果证据，请输出"
                                      '{"answer":"无法确定","evidence_chain":[],"confidence":null}。')
            store.append(record)
            print(f"[{index}/{len(samples)}] {sample.sample_id} 已保存", flush=True)
        store.finish()
        print(f"完成：{store.output}；{len(samples)} 题通过结构校验。此结果不是官方评分。")
        return 0
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(run())
