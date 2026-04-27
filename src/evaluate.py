import re
import sys
import time
from pathlib import Path
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import analyze, load_all

# ---------------------------------------------------------------------------
# Common English stopwords — avoids NLTK dependency
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

# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------
TEST_CASES = [
    {
        "label": "Software Engineer (Python / PyTorch / LLM)",
        "resume": """
Alex Rivera | alex@example.com | github.com/alexrivera

EXPERIENCE
Backend Software Engineer — CloudSys Inc. (2021–Present)
- Developed and maintained RESTful APIs using Python and FastAPI
- Built data ingestion pipelines processing 10M+ records daily using Apache Kafka
- Optimised PostgreSQL queries, reducing average response time by 35%
- Collaborated with frontend teams to integrate APIs with React applications
- Wrote unit and integration tests achieving 90% code coverage with pytest
- Containerised services using Docker and deployed via GitHub Actions CI/CD

SKILLS
Python, FastAPI, PostgreSQL, Docker, Kafka, Git, pytest, Linux, REST APIs

EDUCATION
B.Sc. Computer Science, State University, 2021
""",
        "jd": """
We are hiring a Machine Learning Engineer to build and fine-tune large language models.
Requirements:
- Strong Python skills with hands-on PyTorch experience
- Experience fine-tuning or pre-training transformer-based LLMs (GPT, LLaMA, Mistral)
- Familiarity with HuggingFace Transformers and the Datasets library
- Experience with CUDA and GPU-accelerated training workflows
- Knowledge of RLHF, LoRA, or other parameter-efficient fine-tuning techniques
- Experience deploying ML models to production (REST APIs or batch inference)
- Familiarity with experiment tracking tools such as MLflow or Weights & Biases
- Strong collaboration and communication skills
""",
    },
    {
        "label": "Data Scientist (SQL / scikit-learn / A/B Testing)",
        "resume": """
Priya Nair | priya@example.com | linkedin.com/in/priyanair

EXPERIENCE
Junior Data Analyst — RetailMetrics (2022–Present)
- Generated weekly performance dashboards using Tableau and Excel
- Wrote SQL queries to extract sales and customer data from Oracle databases
- Cleaned and merged datasets using Python (pandas) for reporting purposes
- Presented monthly KPI summaries to department heads
- Assisted in building automated email reports using Python scripts

SKILLS
SQL, Python, pandas, Tableau, Excel, Oracle, data cleaning, reporting

EDUCATION
B.Sc. Statistics, City University, 2022
""",
        "jd": """
We are looking for a Data Scientist to join our growth analytics team.
Requirements:
- Proficiency in SQL for complex querying and data extraction
- Hands-on experience with scikit-learn for building classification and regression models
- Strong understanding of A/B testing methodology and statistical significance
- Experience with feature engineering and model evaluation techniques
- Familiarity with experiment design and hypothesis testing
- Ability to translate analytical findings into business recommendations
- Experience with Python libraries including NumPy, pandas, and Matplotlib
- Excellent communication skills for presenting to non-technical stakeholders
""",
    },
    {
        "label": "Product Manager (Roadmap / Agile / Stakeholder)",
        "resume": """
Daniel Kim | daniel@example.com | linkedin.com/in/danielkim

EXPERIENCE
Project Coordinator — HealthTech Solutions (2020–Present)
- Coordinated cross-functional teams across engineering, design, and QA departments
- Maintained project timelines and tracked deliverables using Jira and Confluence
- Facilitated weekly team meetings and documented action items
- Assisted senior managers in preparing presentations for executive reviews
- Managed vendor relationships and negotiated delivery schedules
- Tracked project budgets and flagged risks to leadership

SKILLS
Jira, Confluence, project coordination, team communication, Excel, PowerPoint

EDUCATION
B.A. Business Administration, Metro University, 2020
""",
        "jd": """
We are hiring a Product Manager to lead our core platform team.
Requirements:
- Experience owning and managing a product roadmap from ideation to launch
- Proficiency with agile methodologies — sprint planning, retrospectives, backlog grooming
- Proven stakeholder management skills, including executive-level communication
- Ability to define OKRs and success metrics aligned with business goals
- Experience conducting user research and synthesising customer feedback
- Familiarity with go-to-market planning and feature launch coordination
- Data-driven mindset — comfortable analysing usage metrics to prioritise features
- Experience working directly with engineering and design teams in a product role
""",
    },
    {
        "label": "Marketing Manager (SEO / Google Ads)",
        "resume": """
Sofia Reyes | sofia@example.com | linkedin.com/in/sofiareyes

EXPERIENCE
Social Media Coordinator — BrandWave Agency (2021–Present)
- Managed organic content calendars for Facebook, Instagram, and LinkedIn
- Wrote blog posts and email newsletters for three B2C clients
- Tracked engagement metrics using native platform analytics and Google Sheets
- Coordinated with designers for campaign creative assets
- Assisted in setting up and monitoring Facebook Ads campaigns
- Reported monthly on follower growth and engagement rates to account managers

SKILLS
Facebook Ads, Instagram, content writing, email marketing, Canva, Google Sheets

EDUCATION
B.A. Communications, Westfield University, 2021
""",
        "jd": """
We are hiring a Digital Marketing Manager to lead our search and performance channels.
Requirements:
- Proven SEO experience including keyword research, on-page optimisation, and technical SEO audits
- Hands-on experience managing Google Ads and paid search (SEM) campaigns
- Proficiency with Google Analytics 4 and Google Search Console
- Experience with A/B testing and conversion rate optimisation (CRO)
- Familiarity with marketing automation platforms such as HubSpot or Marketo
- Strong analytical skills — comfortable interpreting ROAS, CPA, and attribution data
- Content strategy experience including editorial calendar management
- Excellent written communication and cross-functional collaboration skills
""",
    },
    {
        "label": "Data Analyst (Tableau / SQL / BI)",
        "resume": """
Marcus Thompson | marcus@example.com | linkedin.com/in/marcust

EXPERIENCE
Reporting Analyst — FinanceGroup Ltd. (2022–Present)
- Built and maintained Excel-based reports for sales, operations, and finance teams
- Wrote basic SQL queries to pull data from a MySQL database for ad-hoc requests
- Created PowerPoint presentations summarising monthly performance for management
- Automated repetitive reporting tasks using Excel macros (VBA)
- Liaised with IT to resolve data quality issues in source systems
- Assisted in migrating legacy reports to a shared SharePoint portal

SKILLS
Excel, VBA, SQL (basic), PowerPoint, SharePoint, data reporting

EDUCATION
B.Sc. Business Information Systems, North University, 2022
""",
        "jd": """
We are looking for a Business Intelligence Analyst to join our data team.
Requirements:
- Advanced SQL skills including window functions, CTEs, and query optimisation
- Proficiency in Tableau for building interactive dashboards and executive reports
- Experience with data warehousing concepts and platforms such as Redshift or BigQuery
- Familiarity with ETL pipeline tools and data transformation workflows
- Ability to translate business questions into analytical frameworks
- Experience with Power BI is a plus
- Strong storytelling skills — presenting insights clearly to non-technical audiences
- Attention to data accuracy and experience with data validation techniques
""",
    },
]


# ---------------------------------------------------------------------------
# Keyword utilities
# ---------------------------------------------------------------------------

def _extract_keywords(text: str) -> set[str]:
    """Tokenise text, drop stopwords, keep tokens longer than 4 chars."""
    tokens = re.findall(r"\b[a-zA-Z][a-zA-Z+#.-]*[a-zA-Z]\b", text)
    return {
        t.lower()
        for t in tokens
        if len(t) > 4 and t.lower() not in _STOPWORDS
    }


def keyword_match_rate(resume_text: str, jd_text: str) -> float:
    """Return the fraction of JD keywords that appear in the resume (0.0–1.0)."""
    jd_keywords = _extract_keywords(jd_text)
    if not jd_keywords:
        return 0.0
    resume_lower = resume_text.lower()
    matched = sum(
        1 for kw in jd_keywords
        if re.search(r"\b" + re.escape(kw) + r"\b", resume_lower)
    )
    return matched / len(jd_keywords)


def _apply_rewrites(resume_text: str, rewritten_bullets: list[dict]) -> str:
    """Replace original bullet text with improved versions inside the resume."""
    modified = resume_text
    for bullet in rewritten_bullets:
        original = bullet.get("original", "").strip()
        improved = bullet.get("improved", "").strip()
        if original and improved and original in modified:
            modified = modified.replace(original, improved, 1)
    return modified


# ---------------------------------------------------------------------------
# Pipeline evaluation
# ---------------------------------------------------------------------------

def evaluate_pipeline(test_cases: list[dict]) -> dict:
    """Run analyze() on every test case and compute before/after keyword match."""
    load_all()

    case_results = []
    for i, tc in enumerate(test_cases, start=1):
        label = tc["label"]
        resume = tc["resume"]
        jd = tc["jd"]
        print(f"[{i}/{len(test_cases)}] Evaluating: {label}")

        before = keyword_match_rate(resume, jd)

        t0 = time.perf_counter()
        try:
            feedback = analyze(resume, jd)
        except Exception as e:
            print(f"  [ERROR] Pipeline failed: {e}")
            case_results.append({
                "label": label,
                "before_match_rate": before,
                "after_match_rate": before,
                "improvement": 0.0,
                "match_score": None,
                "error": str(e),
            })
            continue
        elapsed = time.perf_counter() - t0

        rewritten = feedback.get("rewritten_bullets", [])
        improved_resume = _apply_rewrites(resume, rewritten)
        after = keyword_match_rate(improved_resume, jd)

        print(
            f"  Before: {before:.1%}  →  After: {after:.1%}  "
            f"(Δ {after - before:+.1%})  LLM score: {feedback.get('match_score', '?')}  "
            f"[{elapsed:.1f}s]"
        )
        case_results.append({
            "label": label,
            "before_match_rate": before,
            "after_match_rate": after,
            "improvement": after - before,
            "match_score": feedback.get("match_score"),
        })

    valid = [r for r in case_results if "error" not in r]
    avg_before = sum(r["before_match_rate"] for r in valid) / len(valid) if valid else 0.0
    avg_after = sum(r["after_match_rate"] for r in valid) / len(valid) if valid else 0.0

    return {
        "cases": case_results,
        "avg_before": avg_before,
        "avg_after": avg_after,
        "avg_improvement": avg_after - avg_before,
    }


# ---------------------------------------------------------------------------
# Report printer
# ---------------------------------------------------------------------------

def print_report(results: dict) -> None:
    col_w = 36
    num_w = 10

    header = f"{'Test Case':<{col_w}} {'Before':>{num_w}} {'After':>{num_w}} {'Δ Improvement':>{num_w}} {'LLM Score':>{num_w}}"
    divider = "-" * len(header)

    print()
    print("=" * len(header))
    print("  ATS ALIGNMENT EVALUATION REPORT")
    print("=" * len(header))
    print(header)
    print(divider)

    for r in results["cases"]:
        error_flag = " [ERROR]" if "error" in r else ""
        llm_score = f"{r['match_score']}/100" if r["match_score"] is not None else "N/A"
        print(
            f"{r['label'][:col_w]:<{col_w}} "
            f"{r['before_match_rate']:>{num_w}.1%} "
            f"{r['after_match_rate']:>{num_w}.1%} "
            f"{r['improvement']:>+{num_w}.1%} "
            f"{llm_score:>{num_w}}"
            f"{error_flag}"
        )

    print(divider)
    print(
        f"{'AVERAGE':<{col_w}} "
        f"{results['avg_before']:>{num_w}.1%} "
        f"{results['avg_after']:>{num_w}.1%} "
        f"{results['avg_improvement']:>+{num_w}.1%} "
        f"{'':>{num_w}}"
    )
    print("=" * len(header))
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    results = evaluate_pipeline(TEST_CASES)
    print_report(results)


if __name__ == "__main__":
    main()
