import json
import sys
from pathlib import Path

import pandas as pd

DATA_DIR = Path("data/raw/onet")
OUT_PATH = Path("data/processed/onet_profiles.jsonl")

SKILL_SCORE_THRESHOLD = 3.0


def find_file(stem: str) -> Path:
    """Try CSV first (as specified), fall back to xlsx if not found."""
    for ext in (".csv", ".xlsx"):
        path = DATA_DIR / f"{stem}{ext}"
        if path.exists():
            return path
    raise FileNotFoundError(f"No .csv or .xlsx found for '{stem}' in {DATA_DIR}")


def read_file(path: Path, **kwargs) -> pd.DataFrame:
    if path.suffix == ".csv":
        return pd.read_csv(path, **kwargs)
    return pd.read_excel(path, **kwargs)


def load_occupations(path: Path) -> pd.DataFrame:
    df = read_file(path, usecols=["O*NET-SOC Code", "Title", "Description"], dtype=str)
    df.columns = df.columns.str.strip()
    return df.dropna(subset=["O*NET-SOC Code"])


def load_skills(path: Path) -> pd.DataFrame:
    df = read_file(
        path,
        usecols=["O*NET-SOC Code", "Element Name", "Scale Name", "Data Value"],
        dtype={"O*NET-SOC Code": str, "Data Value": float},
    )
    df.columns = df.columns.str.strip()
    return df[df["Data Value"] > SKILL_SCORE_THRESHOLD].dropna(subset=["O*NET-SOC Code", "Element Name"])


def main():
    try:
        occ_path = find_file("Occupation Data")
        skills_path = find_file("Skills")
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    print(f"Loading {occ_path.name}...")
    try:
        occupations = load_occupations(occ_path)
    except Exception as e:
        print(f"[ERROR] Failed to load occupation data: {e}")
        sys.exit(1)
    print(f"  {len(occupations)} occupations loaded")

    print(f"Loading {skills_path.name}...")
    try:
        skills = load_skills(skills_path)
    except Exception as e:
        print(f"[ERROR] Failed to load skills data: {e}")
        sys.exit(1)
    print(f"  {len(skills)} skill rows with Data Value > {SKILL_SCORE_THRESHOLD}")

    # Join on O*NET-SOC Code
    skill_groups = (
        skills.groupby("O*NET-SOC Code")["Element Name"]
        .apply(lambda names: ", ".join(sorted(set(names))))
        .reset_index()
        .rename(columns={"Element Name": "skills_text"})
    )
    merged = occupations.merge(skill_groups, on="O*NET-SOC Code", how="left")
    merged["skills_text"] = merged["skills_text"].fillna("N/A")

    # Build and save profiles
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    profiles = []
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for _, row in merged.iterrows():
            profile = {
                "id": row["O*NET-SOC Code"],
                "text": (
                    f"Job Title: {row['Title']}. "
                    f"Description: {row['Description']}. "
                    f"Required Skills: {row['skills_text']}"
                ),
                "title": row["Title"],
            }
            profiles.append(profile)
            f.write(json.dumps(profile, ensure_ascii=False) + "\n")

    print(f"\n{len(profiles)} occupation profiles created.")


if __name__ == "__main__":
    main()
