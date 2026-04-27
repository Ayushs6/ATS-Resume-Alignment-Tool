import json
import re
import sys

from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler

MODEL_ID = "mlx-community/Qwen3-8B-4bit"

SYSTEM_PROMPT = (
    "You are an expert ATS-aware resume coach. "
    "Analyze the resume against the job description and retrieved skill context. "
    "Return ONLY valid JSON — no explanation, no markdown, no code fences. "
    "Hard skills are technical tools, languages, frameworks, and platforms. "
    "Soft skills are interpersonal abilities like communication, leadership, and collaboration."
)

JSON_SCHEMA = """{
  "match_score": <int 0-100>,
  "job_title_match": {
    "resume_title": "<candidate's most recent job title>",
    "target_title": "<job title from JD>",
    "match_level": "<Strong | Partial | Weak>",
    "suggestion": "<1 sentence advice if Partial or Weak, else empty string>"
  },
  "matched_hard_skills": [<up to 10 technical skills/tools found in both resume and JD>],
  "matched_soft_skills": [<up to 5 soft skills found in both resume and JD>],
  "missing_hard_skills": [<up to 10 technical skills/tools in JD but absent from resume>],
  "missing_soft_skills": [<up to 5 soft skills in JD but absent from resume>],
  "placement_suggestions": [
    "<Add 'keyword' to [Section] — example: 'Leveraged X to achieve Y'>"
  ],
  "rewritten_bullets": [
    {"original": "<exact bullet from resume>", "improved": "<rewritten to naturally embed one missing hard skill>"}
  ],
  "cover_letter_snippet": "<a tailored cover letter, MINIMUM 200 words, 3-4 paragraphs>"
}

Rules:
- job_title_match: compare the candidate's most recent role to the target title.
- matched_hard_skills: technical matches only. Max 10 items.
- matched_soft_skills: interpersonal matches only. Max 5 items.
- missing_hard_skills: technical gaps from JD. Max 10 items.
- missing_soft_skills: soft skill gaps from JD. Max 5 items.
- placement_suggestions: for the top 6 missing hard skills, specify which resume section to add them to and give a 1-line example sentence that embeds the keyword naturally.
- rewritten_bullets: pick the 6 weakest bullets and rewrite each to naturally embed one missing hard skill. Max 6 items.
- cover_letter_snippet: write a complete, professional cover letter of AT LEAST 200 words. Open with the role and company interest, body should weave in 3-4 specific matched skills and 1-2 quantified achievements from the resume, and close with a confident call to action. Use natural, polished prose — no bullet points, no markdown."""

# Module-level tokenizer cache — populated by load_llm(), reused by build_prompt()
_tokenizer_cache = None


def load_llm():
    global _tokenizer_cache
    model, tokenizer = load(MODEL_ID)
    _tokenizer_cache = tokenizer
    return model, tokenizer


MAX_RESUME_CHARS = 4500   # ~600 words
MAX_JD_CHARS = 2500
MAX_CHUNK_CHARS = 300


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n[truncated]"


def build_prompt(
    resume_text: str,
    job_description: str,
    retrieved_chunks: list[dict],
    keyword_match_pct: float | None = None,
) -> str:
    global _tokenizer_cache
    if _tokenizer_cache is None:
        load_llm()
    tokenizer = _tokenizer_cache

    context_blocks = ""
    if retrieved_chunks:
        top_chunks = retrieved_chunks[:5]
        context_blocks = "\n\n".join(
            f"[{i+1}] {_truncate(chunk['text'], MAX_CHUNK_CHARS)}"
            for i, chunk in enumerate(top_chunks)
        )
    else:
        context_blocks = "(No retrieved context provided)"

    pct_note = ""
    if keyword_match_pct is not None:
        pct_note = (
            f"\nPre-computed keyword overlap: {keyword_match_pct:.0%} of JD keywords "
            f"were found verbatim in the resume. "
            f"Use this as the primary anchor for match_score — "
            f"low overlap (≤20%) → score 10–35; moderate (21–45%) → 36–58; "
            f"good (46–65%) → 59–74; strong (66–80%) → 75–88; "
            f"excellent (>80%) → 89–100. "
            f"Adjust by ±8 points only for seniority fit and soft-skill alignment.\n"
        )

    user_message = f"""Resume:
{_truncate(resume_text.strip(), MAX_RESUME_CHARS)}

Job Description:
{_truncate(job_description.strip(), MAX_JD_CHARS)}
{pct_note}
Retrieved Context (relevant skills and job profiles):
{context_blocks}

Return a JSON object with EXACTLY this structure:
{JSON_SCHEMA}"""

    return tokenizer.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_message},
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


_DEFAULTS = {
    "match_score": 0,
    "job_title_match": {
        "resume_title": "",
        "target_title": "",
        "match_level": "Unknown",
        "suggestion": "",
    },
    "matched_hard_skills": [],
    "matched_soft_skills": [],
    "missing_hard_skills": [],
    "missing_soft_skills": [],
    "placement_suggestions": [],
    "rewritten_bullets": [],
    "cover_letter_snippet": "",
}


def _extract_json(text: str) -> dict:
    text = text.strip()
    # Strip markdown code fences if present
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    # Try direct parse first
    try:
        result = json.loads(text)
        return {**_DEFAULTS, **result}
    except json.JSONDecodeError:
        pass

    # Try extracting the outermost {...} block
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            result = json.loads(match.group())
            return {**_DEFAULTS, **result}
        except json.JSONDecodeError:
            pass

    # Last resort: patch a truncated JSON by closing open brackets
    brace_match = re.search(r"\{.*", text, re.DOTALL)
    if brace_match:
        fragment = brace_match.group()
        # Close any open arrays and the object
        open_arrays = fragment.count("[") - fragment.count("]")
        open_braces = fragment.count("{") - fragment.count("}")
        patched = fragment + "]" * max(open_arrays, 0) + "}" * max(open_braces, 0)
        try:
            result = json.loads(patched)
            return {**_DEFAULTS, **result}
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not parse JSON from model output:\n{text[:400]}")


def generate_feedback(model, tokenizer, prompt: str) -> dict:
    raw_output = generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=2500,
        sampler=make_sampler(temp=0.6),
        verbose=False,
    )
    return _extract_json(raw_output)


def main() -> None:
    sample_resume = """
John Doe | john@example.com | github.com/johndoe

EXPERIENCE
Software Engineer — Acme Corp (2021–Present)
- Built REST APIs using Flask and deployed on AWS EC2
- Wrote SQL queries to support analytics dashboards
- Collaborated with frontend team on React integration

SKILLS
Python, Flask, SQL, Git, REST APIs

EDUCATION
B.Sc. Computer Science, State University, 2021
"""

    sample_jd = """
We are hiring a Backend Python Developer.
Requirements:
- 2+ years of experience with Python (FastAPI or Django preferred)
- Strong knowledge of PostgreSQL and ORMs (SQLAlchemy)
- Experience with Docker and CI/CD pipelines
- Familiarity with machine learning workflows is a plus
- Strong communication and teamwork skills
"""

    print("Loading LLM...")
    model, tokenizer = load_llm()

    print("Building prompt...")
    prompt = build_prompt(sample_resume, sample_jd, retrieved_chunks=[])

    print("Generating feedback...\n")
    try:
        result = generate_feedback(model, tokenizer, prompt)
    except ValueError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
