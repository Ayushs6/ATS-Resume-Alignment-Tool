import json
import sys
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

CHROMA_PATH = "data/chroma_db"
COLLECTION_NAME = "ats_corpus"
EMBED_MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"
BATCH_SIZE = 64
PROGRESS_EVERY = 100

_embed_model_cache: SentenceTransformer | None = None
_collection_cache: chromadb.Collection | None = None


def get_embedding_model() -> SentenceTransformer:
    global _embed_model_cache
    if _embed_model_cache is None:
        _embed_model_cache = SentenceTransformer(EMBED_MODEL_ID)
    return _embed_model_cache


def embed_text(
    model: SentenceTransformer,
    texts: list[str],
    is_query: bool = False,
) -> list[list[float]]:
    prompt_name = "query" if is_query else "document"
    vectors = model.encode(
        texts,
        prompt_name=prompt_name,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return vectors.tolist()


def build_index(chunks_path: str = "data/processed/all_chunks.jsonl") -> None:
    path = Path(chunks_path)
    if not path.exists():
        print(f"[ERROR] Chunks file not found: {path}")
        sys.exit(1)

    print("Loading chunks...")
    chunks = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    total = len(chunks)
    print(f"  {total} chunks loaded")

    print(f"Loading embedding model: {EMBED_MODEL_ID}")
    model = get_embedding_model()

    print(f"Connecting to ChromaDB at: {CHROMA_PATH}")
    Path(CHROMA_PATH).mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=None,
        metadata={"hnsw:space": "cosine"},
    )

    print(f"Embedding and indexing in batches of {BATCH_SIZE}...")
    processed = 0
    milestone = PROGRESS_EVERY

    while processed < total:
        batch = chunks[processed : processed + BATCH_SIZE]

        ids = [c["chunk_id"] for c in batch]
        texts = [c["text"] for c in batch]
        metadatas = [
            {
                "chunk_id": c["chunk_id"],
                "source_id": c["source_id"],
                "title": c["title"],
            }
            for c in batch
        ]

        embeddings = embed_text(model, texts, is_query=False)

        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=texts,
        )

        processed += len(batch)

        if processed >= milestone or processed == total:
            print(f"  {processed}/{total} chunks embedded...")
            milestone += PROGRESS_EVERY

    print(f"Indexing complete. Collection '{COLLECTION_NAME}' has {collection.count()} vectors.")


def get_collection() -> chromadb.Collection:
    global _collection_cache
    if _collection_cache is None:
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        _collection_cache = client.get_collection(name=COLLECTION_NAME, embedding_function=None)
    return _collection_cache


def main() -> None:
    build_index()
    collection = get_collection()
    total_vectors = collection.count()
    print(f"Index built successfully. Total vectors: {total_vectors}")


if __name__ == "__main__":
    main()
