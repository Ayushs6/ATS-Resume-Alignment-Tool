import json
import re
import sys
import time
from pathlib import Path
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent))

import src.embed as embed_module
import src.generate as generate_module
import src.retrieve as retrieve_module
from src.generate import build_prompt, generate_feedback, load_llm
from src.retrieve import retrieve

# ---------------------------------------------------------------------------
# Keyword overlap — used to anchor the LLM's match_score to real data
# ---------------------------------------------------------------------------
_STOPWORDS = {
    "about", "above", "after", "again", "against", "also", "although", "among",
    "another", "around", "because", "been", "before", "being", "below", "between",
    "both", "cannot", "could", "does", "doing", "done", "during", "each",
    "either", "every", "further", "great", "have", "having", "here", "high",
    "however", "including", "into", "just", "know", "large", "like", "looking",
    "make", "making", "many", "more", "most", "must", "need", "needs", "never",
    "other", "others", "over", "owned", "people", "place", "please", "plus",
    "provide", "rather", "related", "role", "same", "since", "some", "strong",
    "such", "support", "team", "than", "that", "their", "them", "then", "there",
    "these", "they", "this", "those", "through", "time", "together", "under",
    "until", "upon", "used", "using", "various", "very", "want", "well", "were",
    "what", "when", "where", "which", "while", "will", "with", "within",
    "without", "work", "working", "would", "year", "years", "your",
}


def _extract_keywords(text: str) -> set[str]:
    tokens = re.findall(r"\b[a-zA-Z][a-zA-Z+#.-]*[a-zA-Z]\b", text)
    return {
        t.lower() for t in tokens
        if len(t) > 4 and t.lower() not in _STOPWORDS
    }


def _keyword_match_rate(resume_text: str, jd_text: str) -> float:
    """Fraction of JD keywords found verbatim in the resume (0.0–1.0)."""
    jd_kw = _extract_keywords(jd_text)
    if not jd_kw:
        return 0.0
    rl = resume_text.lower()
    matched = sum(
        1 for kw in jd_kw
        if re.search(r"\b" + re.escape(kw) + r"\b", rl)
    )
    return matched / len(jd_kw)


# Module-level model references — populated once by load_all()
_llm = None
_llm_tokenizer = None
_loaded = False


def load_all() -> None:
    """Load every model once and cache them at module level."""
    global _llm, _llm_tokenizer, _loaded
    if _loaded:
        return

    print("[1/4] Loading embedding model...")
    embed_module.get_embedding_model()

    print("[2/4] Loading ChromaDB collection...")
    embed_module.get_collection()

    print("[3/4] Loading reranker...")
    retrieve_module.load_reranker()

    print("[4/4] Loading LLM...")
    _llm, _llm_tokenizer = load_llm()

    _loaded = True
    print("All models loaded.\n")


def analyze(resume_text: str, job_description: str) -> dict:
    """Run the full RAG pipeline and return structured feedback as a dict."""
    if not _loaded:
        load_all()

    # Step 1: pre-compute keyword overlap as a numeric anchor for the LLM
    kw_pct = _keyword_match_rate(resume_text, job_description)

    # Step 2: build a retrieval query from both inputs
    query = f"{resume_text.strip()} {job_description.strip()}"

    # Step 3: two-stage retrieval (dense → rerank), top 3
    chunks = retrieve(query, top_n=15, top_k=3)

    # Step 4: build prompt with resume, JD, retrieved context, and keyword anchor
    prompt = build_prompt(resume_text, job_description, chunks, keyword_match_pct=kw_pct)

    # Step 5: generate and parse structured feedback
    return generate_feedback(_llm, _llm_tokenizer, prompt)


def main() -> None:
    sample_resume = """
Jane Smith | jane@example.com | linkedin.com/in/janesmith

EXPERIENCE
Data Analyst — RetailCo (2022–Present)
- Analysed customer purchase data using Python and pandas
- Built dashboards in Tableau to track KPIs for 3 business units
- Wrote SQL queries against a PostgreSQL warehouse
- Automated weekly reports with Python scripts, saving 4 hours/week

SKILLS
Python, pandas, SQL, PostgreSQL, Tableau, Excel, Git

EDUCATION
B.Sc. Statistics, City University, 2022
"""

    sample_jd = """
We are looking for a Data Scientist to join our growing analytics team.
Requirements:
- 2+ years of experience with Python for data analysis or ML
- Proficiency in SQL and working with relational databases
- Familiarity with machine learning libraries (scikit-learn, XGBoost)
- Experience building visualisations or dashboards
- Strong communication skills; ability to present findings to non-technical stakeholders
- Experience with cloud platforms (AWS, GCP) is a plus
"""

    print("=== ATS Resume Analyser ===\n")

    t0 = time.perf_counter()
    load_all()

    print("Running analysis...\n")
    try:
        result = analyze(sample_resume, sample_jd)
    except ValueError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    elapsed = time.perf_counter() - t0
    print(json.dumps(result, indent=2))
    print(f"\nCompleted in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
