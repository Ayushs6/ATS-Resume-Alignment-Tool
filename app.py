import gradio as gr

from src.pipeline import analyze, load_all

# ---------------------------------------------------------------------------
# Example library (5 roles for the gr.Examples picker)
# ---------------------------------------------------------------------------
_EXAMPLES = [
    [
        """Sarah Lee | sarah@example.com | linkedin.com/in/sarahlee

EXPERIENCE
Software Developer — FinTech Solutions (2021–Present)
- Developed backend services using Django and PostgreSQL
- Built REST APIs consumed by mobile and web clients
- Wrote unit tests with pytest; maintained 85% code coverage
- Deployed applications to AWS using Elastic Beanstalk

SKILLS
Python, Django, PostgreSQL, REST APIs, pytest, Git, AWS (basic)

EDUCATION
B.Sc. Computer Science, Metro University, 2021""",
        """We are hiring a Backend Python Engineer.

Requirements:
- 2+ years Python experience with FastAPI or Django
- Strong PostgreSQL and SQLAlchemy skills
- Experience with Docker and CI/CD pipelines (GitHub Actions)
- Familiarity with Redis for caching
- AWS deployment experience (EC2, RDS, or Lambda)
- Test-driven development with pytest
- Good communication and agile team experience""",
    ],
    [
        """Alex Rivera | alex@example.com | github.com/alexrivera

EXPERIENCE
Backend Software Engineer — CloudSys Inc. (2021–Present)
- Developed and maintained RESTful APIs using Python and FastAPI
- Built data ingestion pipelines processing 10M+ records daily using Apache Kafka
- Optimised PostgreSQL queries, reducing average response time by 35%
- Wrote unit and integration tests achieving 90% code coverage with pytest
- Containerised services using Docker and deployed via GitHub Actions CI/CD

SKILLS
Python, FastAPI, PostgreSQL, Docker, Kafka, Git, pytest, Linux, REST APIs

EDUCATION
B.Sc. Computer Science, State University, 2021""",
        """We are hiring a Machine Learning Engineer to build and fine-tune large language models.
Requirements:
- Strong Python skills with hands-on PyTorch experience
- Experience fine-tuning or pre-training transformer-based LLMs (GPT, LLaMA, Mistral)
- Familiarity with HuggingFace Transformers and the Datasets library
- Experience with CUDA and GPU-accelerated training workflows
- Knowledge of RLHF, LoRA, or other parameter-efficient fine-tuning techniques
- Familiarity with experiment tracking tools such as MLflow or Weights & Biases
- Strong collaboration and communication skills""",
    ],
    [
        """Priya Nair | priya@example.com | linkedin.com/in/priyanair

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
B.Sc. Statistics, City University, 2022""",
        """We are looking for a Data Scientist to join our growth analytics team.
Requirements:
- Proficiency in SQL for complex querying and data extraction
- Hands-on experience with scikit-learn for building classification and regression models
- Strong understanding of A/B testing methodology and statistical significance
- Experience with feature engineering and model evaluation techniques
- Familiarity with experiment design and hypothesis testing
- Ability to translate analytical findings into business recommendations
- Excellent communication skills for presenting to non-technical stakeholders""",
    ],
    [
        """Daniel Kim | daniel@example.com | linkedin.com/in/danielkim

EXPERIENCE
Project Coordinator — HealthTech Solutions (2020–Present)
- Coordinated cross-functional teams across engineering, design, and QA departments
- Maintained project timelines and tracked deliverables using Jira and Confluence
- Facilitated weekly team meetings and documented action items
- Assisted senior managers in preparing presentations for executive reviews
- Managed vendor relationships and negotiated delivery schedules

SKILLS
Jira, Confluence, project coordination, team communication, Excel, PowerPoint

EDUCATION
B.A. Business Administration, Metro University, 2020""",
        """We are hiring a Product Manager to lead our core platform team.
Requirements:
- Experience owning and managing a product roadmap from ideation to launch
- Proficiency with agile methodologies — sprint planning, retrospectives, backlog grooming
- Proven stakeholder management skills, including executive-level communication
- Ability to define OKRs and success metrics aligned with business goals
- Experience conducting user research and synthesising customer feedback
- Familiarity with go-to-market planning and feature launch coordination
- Data-driven mindset — comfortable analysing usage metrics to prioritise features""",
    ],
    [
        """Sofia Reyes | sofia@example.com | linkedin.com/in/sofiareyes

EXPERIENCE
Social Media Coordinator — BrandWave Agency (2021–Present)
- Managed organic content calendars for Facebook, Instagram, and LinkedIn
- Wrote blog posts and email newsletters for three B2C clients
- Tracked engagement metrics using native platform analytics and Google Sheets
- Coordinated with designers for campaign creative assets
- Assisted in setting up and monitoring Facebook Ads campaigns

SKILLS
Facebook Ads, Instagram, content writing, email marketing, Canva, Google Sheets

EDUCATION
B.A. Communications, Westfield University, 2021""",
        """We are hiring a Digital Marketing Manager to lead our search and performance channels.
Requirements:
- Proven SEO experience including keyword research, on-page optimisation, and technical SEO audits
- Hands-on experience managing Google Ads and paid search (SEM) campaigns
- Proficiency with Google Analytics 4 and Google Search Console
- Experience with A/B testing and conversion rate optimisation (CRO)
- Familiarity with marketing automation platforms such as HubSpot or Marketo
- Strong analytical skills — comfortable interpreting ROAS, CPA, and attribution data""",
    ],
    [
        """Marcus Thompson | marcus@example.com | linkedin.com/in/marcust

EXPERIENCE
Reporting Analyst — FinanceGroup Ltd. (2022–Present)
- Built and maintained Excel-based reports for sales, operations, and finance teams
- Wrote basic SQL queries to pull data from a MySQL database for ad-hoc requests
- Created PowerPoint presentations summarising monthly performance for management
- Automated repetitive reporting tasks using Excel macros (VBA)
- Liaised with IT to resolve data quality issues in source systems

SKILLS
Excel, VBA, SQL (basic), PowerPoint, SharePoint, data reporting

EDUCATION
B.Sc. Business Information Systems, North University, 2022""",
        """We are looking for a Business Intelligence Analyst to join our data team.
Requirements:
- Advanced SQL skills including window functions, CTEs, and query optimisation
- Proficiency in Tableau for building interactive dashboards and executive reports
- Experience with data warehousing concepts and platforms such as Redshift or BigQuery
- Familiarity with ETL pipeline tools and data transformation workflows
- Ability to translate business questions into analytical frameworks
- Strong storytelling skills — presenting insights clearly to non-technical audiences""",
    ],
]


# ---------------------------------------------------------------------------
# File extraction (PDF / DOCX)
# ---------------------------------------------------------------------------
def _extract_resume_text(filepath: str) -> str:
    path = filepath.lower()
    if path.endswith(".pdf"):
        import pdfplumber
        with pdfplumber.open(filepath) as pdf:
            return "\n".join(page.extract_text() or "" for page in pdf.pages).strip()
    if path.endswith(".docx"):
        from docx import Document
        doc = Document(filepath)
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip()).strip()
    return ""


# ---------------------------------------------------------------------------
# HTML helpers
# ---------------------------------------------------------------------------
def _score_badge(score: int) -> str:
    if score >= 70:
        color, bg, label = "#166534", "#dcfce7", "Strong Match"
    elif score >= 50:
        color, bg, label = "#92400e", "#fef3c7", "Moderate Match"
    else:
        color, bg, label = "#991b1b", "#fee2e2", "Weak Match"
    return (
        f'<div style="text-align:center;padding:16px 0 8px 0;">'
        f'<span style="font-size:60px;font-weight:800;color:{color};line-height:1;">{score}</span>'
        f'<span style="font-size:26px;color:#94a3b8;font-weight:400;">/100</span>'
        f'<div style="margin-top:10px;">'
        f'<span style="background:{bg};color:{color};padding:5px 20px;border-radius:20px;'
        f'font-size:14px;font-weight:600;">{label}</span>'
        f'</div></div>'
    )


def _summary_card(score: int, n_matched: int, n_missing: int, n_bullets: int) -> str:
    return (
        f'<div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;'
        f'padding:10px 16px;font-size:14px;color:#475569;">'
        f'<strong>{score}/100</strong>&nbsp;&nbsp;·&nbsp;&nbsp;'
        f'<span style="color:#166534;font-weight:600;">✓ {n_matched} skills matched</span>'
        f'&nbsp;&nbsp;·&nbsp;&nbsp;'
        f'<span style="color:#991b1b;font-weight:600;">✗ {n_missing} gaps found</span>'
        f'&nbsp;&nbsp;·&nbsp;&nbsp;{n_bullets} bullets rewritten</div>'
    )


_PLACEHOLDER_BADGE = (
    '<div style="text-align:center;padding:16px 0 8px 0;">'
    '<span style="font-size:60px;font-weight:800;color:#cbd5e1;line-height:1;">—</span>'
    '<span style="font-size:26px;color:#cbd5e1;">/100</span>'
    '<div style="margin-top:10px;">'
    '<span style="background:#f1f5f9;color:#94a3b8;padding:5px 20px;border-radius:20px;'
    'font-size:14px;font-weight:600;">Awaiting Analysis</span>'
    '</div></div>'
)

_PLACEHOLDER_SUMMARY = (
    '<div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;'
    'padding:10px 16px;font-size:14px;color:#94a3b8;text-align:center;">'
    'Analysis summary will appear here after submission.</div>'
)

# Tuple shape: score_html, summary_html, job_title, m_hard, m_soft, miss_hard, miss_soft, placement, bullets_df, bullets_copy, cover_letter
_EMPTY = (_PLACEHOLDER_BADGE, _PLACEHOLDER_SUMMARY, "", "", "", "", "", "", [], "", "")


# ---------------------------------------------------------------------------
# Core handler
# ---------------------------------------------------------------------------
def run_analysis(resume_file, resume_text: str, jd_text: str):
    # File upload takes priority over pasted text
    if resume_file is not None:
        try:
            extracted = _extract_resume_text(resume_file)
            if extracted:
                resume_text = extracted
        except Exception as e:
            gr.Warning(f"Could not read uploaded file: {e}. Using pasted text instead.")

    if not resume_text.strip():
        gr.Warning("Please paste your resume or upload a PDF / DOCX file.")
        return _EMPTY
    if not jd_text.strip():
        gr.Warning("Please paste a job description before submitting.")
        return _EMPTY

    try:
        result = analyze(resume_text, jd_text)
    except Exception as e:
        gr.Warning(f"Analysis failed: {e}", title="Error")
        return _EMPTY

    score = result.get("match_score", 0)

    jtm = result.get("job_title_match", {})
    suggestion = jtm.get("suggestion", "").strip()
    job_title_text = (
        f"Your title:   {jtm.get('resume_title', 'N/A')}\n"
        f"Target title: {jtm.get('target_title', 'N/A')}\n"
        f"Match level:  {jtm.get('match_level', 'Unknown')}"
        + (f"\nTip: {suggestion}" if suggestion else "")
    )

    matched_hard_list = result.get("matched_hard_skills", [])
    matched_soft_list = result.get("matched_soft_skills", [])
    missing_hard_list = result.get("missing_hard_skills", [])
    missing_soft_list = result.get("missing_soft_skills", [])

    placement = "\n".join(f"• {s}" for s in result.get("placement_suggestions", []))

    rewritten = result.get("rewritten_bullets", [])
    bullets_df = [
        [b.get("original", ""), b.get("improved", "")]
        for b in rewritten
    ]
    bullets_copy = "\n\n".join(
        f"{i}. {b.get('improved', '')}"
        for i, b in enumerate(rewritten, start=1)
    )

    cover_letter = result.get("cover_letter_snippet", "").strip()

    n_matched = len(matched_hard_list) + len(matched_soft_list)
    n_missing = len(missing_hard_list) + len(missing_soft_list)

    if score < 50:
        gr.Warning(
            f"Low ATS match: {score}/100. "
            "Add the missing keywords to significantly improve your chances.",
            title="Low Match Score",
        )

    return (
        _score_badge(score),
        _summary_card(score, n_matched, n_missing, len(rewritten)),
        job_title_text,
        ", ".join(matched_hard_list),
        ", ".join(matched_soft_list),
        ", ".join(missing_hard_list),
        ", ".join(missing_soft_list),
        placement,
        bullets_df,
        bullets_copy,
        cover_letter,
    )


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
with gr.Blocks(title="ATS Resume Alignment Tool") as demo:
    gr.Markdown(
        """
        # 🎯 ATS Resume Alignment Tool
        ### Powered by Qwen3-8B + Two-Stage RAG &nbsp;|&nbsp; Local & Private
        Upload or paste your resume and the job description. The tool scores your ATS
        alignment, surfaces missing keywords with placement advice, rewrites weak bullets,
        and drafts a tailored cover letter — all on-device.
        """
    )

    with gr.Row():
        # ── Left panel: inputs ────────────────────────────────────────────
        with gr.Column(scale=1):
            gr.Markdown("### 📄 Your Resume & JD")

            resume_file = gr.File(
                label="Upload Resume (PDF or DOCX)",
                file_types=[".pdf", ".docx"],
                type="filepath",
            )
            resume_input = gr.Textbox(
                label="Or Paste Your Resume",
                lines=12,
                placeholder="Paste the full text of your resume here…",
            )
            jd_input = gr.Textbox(
                label="Paste Job Description",
                lines=10,
                placeholder="Paste the job description here…",
            )
            with gr.Row():
                submit_btn = gr.Button("Analyze My Resume", variant="primary", size="lg")
                clear_btn  = gr.Button("Clear", variant="secondary", size="lg")

            gr.Examples(
                examples=_EXAMPLES,
                inputs=[resume_input, jd_input],
                label="📎 Load Example (6 roles)",
                examples_per_page=6,
            )

        # ── Right panel: results ──────────────────────────────────────────
        with gr.Column(scale=1):
            gr.Markdown("### 📊 Analysis Results")

            score_html   = gr.HTML(value=_PLACEHOLDER_BADGE)
            summary_html = gr.HTML(value=_PLACEHOLDER_SUMMARY)

            gr.Markdown("#### 🎯 Job Title Match")
            job_title_match = gr.Textbox(lines=4, interactive=False, show_label=False)

            gr.Markdown("#### ✅ Matched Skills")
            with gr.Row():
                matched_hard = gr.Textbox(label="Hard Skills", lines=3, interactive=False)
                matched_soft = gr.Textbox(label="Soft Skills", lines=3, interactive=False)

            gr.Markdown("#### ❌ Missing Skills to Add")
            with gr.Row():
                missing_hard = gr.Textbox(label="Hard Skills", lines=3, interactive=False)
                missing_soft = gr.Textbox(label="Soft Skills", lines=3, interactive=False)

            placement_suggestions = gr.Textbox(
                label="📍 Where & How to Add Missing Keywords",
                lines=10,
                interactive=False,
            )
            bullets_output = gr.Dataframe(
                headers=["Original Bullet", "AI-Improved Bullet"],
                label="✏️ Resume Bullet Rewrites",
                interactive=False,
                wrap=True,
            )
            bullets_copy = gr.Textbox(
                label="📋 Copy All Improved Bullets",
                lines=8,
                interactive=False,
            )
            cover_letter = gr.Textbox(
                label="✉️ Tailored Cover Letter (200+ words)",
                lines=12,
                interactive=False,
            )

    # ── Wire submit & clear ───────────────────────────────────────────────
    _outputs = [
        score_html, summary_html, job_title_match,
        matched_hard, matched_soft,
        missing_hard, missing_soft,
        placement_suggestions, bullets_output, bullets_copy,
        cover_letter,
    ]

    submit_btn.click(
        fn=run_analysis,
        inputs=[resume_file, resume_input, jd_input],
        outputs=_outputs,
        show_progress="full",
    )

    clear_btn.click(
        fn=lambda: (None, "", "") + _EMPTY,
        inputs=[],
        outputs=[resume_file, resume_input, jd_input] + _outputs,
    )


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Loading models at startup…")
    load_all()
    print("Models ready. Starting Gradio…\n")
    demo.launch()
