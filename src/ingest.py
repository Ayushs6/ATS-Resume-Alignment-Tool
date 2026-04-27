import json
import sys
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

ONET_PATH = Path("data/processed/onet_profiles.jsonl")
JD_PATH = Path("data/processed/jd_corpus.jsonl")
OUT_PATH = Path("data/processed/all_chunks.jsonl")


def chunk_documents(
    jsonl_path: str,
    chunk_size: int = 512,
    overlap: int = 64,
) -> list[dict]:
    path = Path(jsonl_path)
    if not path.exists():
        raise FileNotFoundError(f"JSONL file not found: {path}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=len,
    )

    chunks = []
    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                doc = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"  [WARN] Skipping malformed line {line_num} in {path.name}: {e}")
                continue

            source_id = doc.get("id", f"doc_{line_num}")
            title = doc.get("title", "")
            text = doc.get("text", "")

            if not text.strip():
                continue

            parts = splitter.split_text(text)
            for n, part in enumerate(parts):
                chunks.append({
                    "chunk_id": f"{source_id}_chunk_{n}",
                    "source_id": source_id,
                    "title": title,
                    "text": part,
                })

    return chunks


def main():
    sources = [ONET_PATH, JD_PATH]
    all_chunks = []

    for path in sources:
        print(f"Chunking {path.name}...")
        try:
            chunks = chunk_documents(str(path))
        except FileNotFoundError as e:
            print(f"  [ERROR] {e}")
            sys.exit(1)
        print(f"  {path.name}: {chunks[0]['source_id']!r} … → {len(chunks)} chunks")
        all_chunks.extend(chunks)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    print(f"\nTotal chunks : {len(all_chunks)}")
    print(f"Saved to     : {OUT_PATH}")


if __name__ == "__main__":
    main()
