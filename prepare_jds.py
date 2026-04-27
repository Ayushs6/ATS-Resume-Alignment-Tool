import json
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

OUT_PATH = Path("data/processed/jd_corpus.jsonl")

MIN_DESC_LENGTH = 100
DEDUP_KEY_LENGTH = 200
MAX_RECORDS = 2000


@dataclass
class DatasetConfig:
    path: Path
    col_id: str | None      # None = use row index with id_prefix
    col_title: str
    col_description: str
    id_prefix: str          # prepended to every id to guarantee uniqueness across sources


SOURCES = [
    DatasetConfig(
        path=Path("data/raw/jd_dataset.csv"),
        col_id="JobID",
        col_title="Title",
        col_description="Responsibilities",
        id_prefix="DS1-",
    ),
    DatasetConfig(
        path=Path("data/raw/jd_dataset2.csv"),
        col_id=None,
        col_title="Job Title",
        col_description="Job Description",
        id_prefix="DS2-",
    ),
]


def load_source(cfg: DatasetConfig) -> pd.DataFrame | None:
    if not cfg.path.exists():
        print(f"  [ERROR] File not found: {cfg.path}")
        return None

    try:
        df = pd.read_csv(cfg.path, dtype=str)
    except Exception as e:
        print(f"  [ERROR] Could not read {cfg.path.name}: {e}")
        return None

    for col in (cfg.col_title, cfg.col_description):
        if col not in df.columns:
            print(f"  [ERROR] Column '{col}' missing in {cfg.path.name}. Found: {df.columns.tolist()}")
            return None

    raw_count = len(df)

    # Build id column
    if cfg.col_id and cfg.col_id in df.columns:
        df["_id"] = cfg.id_prefix + df[cfg.col_id].astype(str).str.strip()
    else:
        df["_id"] = cfg.id_prefix + df.index.astype(str)

    df = df.rename(columns={cfg.col_title: "_title", cfg.col_description: "_description"})
    df = df[["_id", "_title", "_description"]]

    # Drop null or short descriptions
    before_drop = len(df)
    df = df.dropna(subset=["_title", "_description"])
    df = df[df["_description"].str.strip().str.len() >= MIN_DESC_LENGTH]
    dropped = before_drop - len(df)

    print(f"  {cfg.path.name}: {raw_count} rows loaded, {dropped} dropped (null/short) → {len(df)} kept")
    return df


def main():
    all_frames = []

    for cfg in SOURCES:
        print(f"Loading {cfg.path.name}...")
        df = load_source(cfg)
        if df is not None:
            all_frames.append(df)

    if not all_frames:
        print("[ERROR] No data loaded.")
        sys.exit(1)

    combined = pd.concat(all_frames, ignore_index=True)
    total_combined = len(combined)

    # Deduplicate across all sources by first 200 chars of description
    combined["_dedup_key"] = combined["_description"].str.strip().str[:DEDUP_KEY_LENGTH]
    before_dedup = len(combined)
    combined = combined.drop_duplicates(subset="_dedup_key", keep="first").drop(columns="_dedup_key")
    dropped_dupes = before_dedup - len(combined)

    # Sample to max 2000
    sampled = False
    if len(combined) > MAX_RECORDS:
        combined = combined.sample(n=MAX_RECORDS, random_state=42)
        sampled = True

    # Build text field and save
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for _, row in combined.iterrows():
            profile = {
                "id": row["_id"],
                "text": (
                    f"Job Title: {row['_title'].strip()}. "
                    f"Description: {row['_description'].strip()}"
                ),
                "title": row["_title"].strip(),
            }
            f.write(json.dumps(profile, ensure_ascii=False) + "\n")

    final_count = len(combined)

    # Summary
    print()
    print(f"Total rows (combined) : {total_combined}")
    print(f"Dropped (duplicates)  : {dropped_dupes}")
    print(f"Sampled to {MAX_RECORDS}       : {sampled}")
    print(f"Final count           : {final_count}")
    print(f"Saved to              : {OUT_PATH}")


if __name__ == "__main__":
    main()
