import sys
from pathlib import Path
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent))

from sentence_transformers import CrossEncoder

from src.embed import embed_text, get_collection, get_embedding_model

RERANKER_MODEL_ID = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_reranker_cache: CrossEncoder | None = None


def load_reranker() -> CrossEncoder:
    global _reranker_cache
    if _reranker_cache is not None:
        return _reranker_cache
    _reranker_cache = CrossEncoder(RERANKER_MODEL_ID)
    return _reranker_cache


def rerank(
    model: CrossEncoder,
    query: str,
    candidates: list[dict],
    top_k: int = 3,
) -> list[dict]:
    if not candidates:
        return []
    pairs = [(query, c["text"]) for c in candidates]
    scores = model.predict(pairs)
    results = [
        {**candidate, "rerank_score": float(score)}
        for candidate, score in zip(candidates, scores)
    ]
    results.sort(key=lambda x: x["rerank_score"], reverse=True)
    return results[:top_k]


def retrieve(query: str, top_n: int = 10, top_k: int = 3) -> list[dict]:
    # Stage 1: dense retrieval from ChromaDB
    embed_model = get_embedding_model()
    collection = get_collection()

    query_embedding = embed_text(embed_model, [query], is_query=True)
    raw = collection.query(
        query_embeddings=query_embedding,
        n_results=top_n,
        include=["metadatas", "documents", "distances"],
    )

    candidates = [
        {
            "chunk_id": chunk_id,
            "source_id": meta.get("source_id", ""),
            "title":    meta.get("title", ""),
            "text":     doc,
        }
        for chunk_id, meta, doc in zip(
            raw["ids"][0],
            raw["metadatas"][0],
            raw["documents"][0],
        )
    ]

    # Stage 2: cross-encoder reranking
    rerank_model = load_reranker()
    ranked = rerank(rerank_model, query, candidates, top_k=top_k)

    return [
        {
            "chunk_id":     r["chunk_id"],
            "title":        r["title"],
            "text":         r["text"],
            "rerank_score": r["rerank_score"],
        }
        for r in ranked
    ]


def main() -> None:
    query = "Python developer with machine learning skills"
    print(f"Query: {query}\n")

    try:
        results = retrieve(query, top_n=10, top_k=3)
    except Exception as e:
        print(f"[ERROR] Retrieval failed: {e}")
        sys.exit(1)

    print("Top 3 results:")
    for i, r in enumerate(results[:3], start=1):
        print(f"\n[{i}] {r['title']}")
        print(f"    Rerank score : {r['rerank_score']:.4f}")
        print(f"    Text         : {r['text'][:200]}...")


if __name__ == "__main__":
    main()
