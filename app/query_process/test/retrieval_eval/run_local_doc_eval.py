import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from dotenv import load_dotenv

from app.lm.embedding_utils import generate_embeddings
from app.lm.lm_utils import get_llm_client


load_dotenv(".env")


@dataclass
class ChunkDoc:
    chunk_id: str
    file_title: str
    title: str
    parent_title: str
    content: str
    source_file: str

    @property
    def text(self) -> str:
        parts = [self.title, self.parent_title, self.content]
        return "\n".join(x for x in parts if x)


@dataclass
class CorpusEmbeddings:
    docs: List[ChunkDoc]
    dense: List[List[float]]
    sparse: List[Dict[int, float]]


def load_questions(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def normalize_doc_title_from_source(source_doc: str) -> str:
    name = Path(source_doc).name
    if name.endswith(".pdf"):
        return name[:-4]
    return name


def load_local_chunks() -> List[ChunkDoc]:
    corpora = [
        Path(r"output\20260420\ae4bb064-06d2-4e7b-85ab-4bb501986dad\chunks.json"),
        Path(r"output\20260420\1df8dfde-855b-4716-a9e5-8e4a3d8e1294\chunks.json"),
        Path(r"output\20260420\5e41ddfd-8f0a-4a7f-9482-eef09af45b4b\chunks.json"),
    ]
    docs: List[ChunkDoc] = []
    for p in corpora:
        data = json.loads(p.read_text(encoding="utf-8"))
        dir_id = p.parent.name
        for idx, item in enumerate(data, start=1):
            if not isinstance(item, dict):
                continue
            docs.append(
                ChunkDoc(
                    chunk_id=f"{dir_id}#{idx}",
                    file_title=(item.get("file_title") or "").strip(),
                    title=(item.get("title") or "").strip(),
                    parent_title=(item.get("parent_title") or "").strip(),
                    content=(item.get("content") or "").strip(),
                    source_file=str(p),
                )
            )
    return docs


def dot_sparse(a: Dict[int, float], b: Dict[int, float]) -> float:
    if len(a) > len(b):
        a, b = b, a
    score = 0.0
    for k, v in a.items():
        if k in b:
            score += v * b[k]
    return score


def build_corpus_cache(docs: List[ChunkDoc]) -> Dict[str, CorpusEmbeddings]:
    grouped: Dict[str, List[ChunkDoc]] = {}
    for doc in docs:
        grouped.setdefault(doc.file_title, []).append(doc)
    cache: Dict[str, CorpusEmbeddings] = {}
    for file_title, items in grouped.items():
        emb = generate_embeddings([d.text for d in items])
        cache[file_title] = CorpusEmbeddings(
            docs=items,
            dense=emb["dense"],
            sparse=emb["sparse"],
        )
    return cache


def weighted_hybrid_rank(query_text: str, corpus: CorpusEmbeddings, top_k: int = 10) -> List[dict]:
    embeddings = generate_embeddings([query_text])
    q_dense = embeddings["dense"][0]
    q_sparse = embeddings["sparse"][0]
    doc_dense = corpus.dense
    doc_sparse = corpus.sparse

    scored = []
    for doc, dense_vec, sparse_vec in zip(corpus.docs, doc_dense, doc_sparse):
        dense_score = sum(x * y for x, y in zip(q_dense, dense_vec))
        sparse_score = dot_sparse(q_sparse, sparse_vec)
        score = 0.8 * dense_score + 0.2 * sparse_score
        scored.append(
            {
                "chunk_id": doc.chunk_id,
                "score": float(score),
                "dense_score": float(dense_score),
                "sparse_score": float(sparse_score),
                "file_title": doc.file_title,
                "title": doc.title,
                "content_preview": doc.content[:160].replace("\n", " "),
            }
        )
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


def generate_hyde_doc(query_text: str) -> str:
    llm = get_llm_client()
    prompt = (
        "你是一个企业技术资料检索助手。\n"
        "请针对下面的问题，生成一段最可能命中文档证据的中文说明文字。\n"
        "要求：\n"
        "1. 只输出说明文字，不要加解释。\n"
        "2. 尽量包含产品名、章节术语、部署方式、参数或能力点。\n"
        "3. 120到220字。\n\n"
        f"问题：{query_text}"
    )
    response = llm.invoke(prompt)
    return response.content.strip()


def rrf_merge(left: List[dict], right: List[dict], top_k: int = 10, k: int = 60) -> List[dict]:
    score_map: Dict[str, float] = {}
    item_map: Dict[str, dict] = {}
    for items in (left, right):
        for rank, item in enumerate(items, start=1):
            chunk_id = item["chunk_id"]
            score_map[chunk_id] = score_map.get(chunk_id, 0.0) + 1.0 / (k + rank)
            item_map.setdefault(chunk_id, item)
    merged = []
    for chunk_id, score in score_map.items():
        item = dict(item_map[chunk_id])
        item["rrf_score"] = float(score)
        merged.append(item)
    merged.sort(key=lambda x: x["rrf_score"], reverse=True)
    return merged[:top_k]


def select_docs_for_question(question: dict, docs: List[ChunkDoc]) -> List[ChunkDoc]:
    source_title = normalize_doc_title_from_source(question["source_doc"])
    selected = [d for d in docs if d.file_title == source_title]
    if selected:
        return selected
    return docs


def build_baseline_query(question: dict) -> str:
    rewritten = (question.get("expected_rewrite") or "").strip()
    return rewritten or question["query"].strip()


def write_run(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local retrieval eval over imported doc chunks")
    parser.add_argument(
        "--questions",
        default=r"app\query_process\test\retrieval_eval\current_kb_questions.csv",
        help="Question CSV path",
    )
    parser.add_argument(
        "--out-dir",
        default=r"app\query_process\test\retrieval_eval\runs",
        help="Directory to write run files",
    )
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    questions = load_questions(Path(args.questions))
    docs = load_local_chunks()
    corpus_cache = build_corpus_cache(docs)
    out_dir = Path(args.out_dir)

    baseline_rows = []
    final_rows = []

    for idx, q in enumerate(questions, start=1):
        qid = q["qid"]
        candidate_docs = select_docs_for_question(q, docs)
        source_title = candidate_docs[0].file_title if candidate_docs else ""
        corpus = corpus_cache[source_title]
        baseline_query = build_baseline_query(q)
        baseline_hits = weighted_hybrid_rank(baseline_query, corpus, top_k=args.top_k)

        hyde_doc = generate_hyde_doc(baseline_query)
        hyde_query = baseline_query + "\n" + hyde_doc
        hyde_hits = weighted_hybrid_rank(hyde_query, corpus, top_k=args.top_k)
        merged_hits = rrf_merge(baseline_hits, hyde_hits, top_k=args.top_k)

        baseline_rows.append(
            {
                "qid": qid,
                "retrieved_chunk_ids": [x["chunk_id"] for x in baseline_hits],
                "results": baseline_hits,
            }
        )
        final_rows.append(
            {
                "qid": qid,
                "retrieved_chunk_ids": [x["chunk_id"] for x in merged_hits],
                "hyde_doc": hyde_doc,
                "baseline_results": baseline_hits,
                "hyde_results": hyde_hits,
                "results": merged_hits,
            }
        )
        print(f"[{idx}/{len(questions)}] {qid} done")

    write_run(out_dir / "baseline_rewrite.local.jsonl", baseline_rows)
    write_run(out_dir / "final_rrf.local.jsonl", final_rows)
    print("baseline:", out_dir / "baseline_rewrite.local.jsonl")
    print("final   :", out_dir / "final_rrf.local.jsonl")


if __name__ == "__main__":
    main()
