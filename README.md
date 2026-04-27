# 🎯 ATS Resume Alignment Tool

> **CSIT 595 — Generative AI Applications | Montclair State University**

---

## 📋 Project Overview

The ATS Resume Alignment Tool is a fully local, on-device application that analyzes a resume against a job description and produces an ATS match score, matched and missing skills (hard and soft), keyword placement advice, AI-rewritten bullet points, and a tailored cover letter. The system uses a two-stage RAG pipeline — dense vector retrieval followed by cross-encoder reranking — backed by a Qwen3-8B language model running entirely on Apple Silicon via MLX. No API keys, no cloud calls, and no data ever leaves the machine.

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **LLM** | `mlx-community/Qwen3-8B-4bit` via `mlx-lm` |
| **Embeddings** | `Qwen3-Embedding-0.6B` (MLX) |
| **Vector Store** | ChromaDB (persistent, local) |
| **Reranker** | `cross-encoder/ms-marco-MiniLM-L-6-v2` via `sentence-transformers` |
| **UI** | Gradio 6 |
| **Document Parsing** | `pdfplumber` (PDF), `python-docx` (DOCX) |
| **Runtime** | Python 3.10+, MLX, Apple Metal |

---

## ✅ Requirements

- **OS:** macOS with Apple Silicon (M1 / M2 / M3 / M4)
- **Python:** 3.10 or higher
- **Disk space:** ~7 GB (model weights + ChromaDB index)
- **RAM:** 16 GB recommended (8 GB minimum)
- **Browser:** Chrome or Firefox (Safari not supported)

---

## 🚀 Installation

**1. Clone the repository**
```bash
git clone <your-repo-url>
cd "GenAI Project"
```

**2. Create and activate a virtual environment**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Download the LLM weights**

The model (`Qwen3-8B-4bit`, ~5 GB) is downloaded automatically from Hugging Face on first run. Ensure you have a stable internet connection for the initial launch. Subsequent runs use the local cache and are fully offline.

**5. Verify the ChromaDB index**

The pre-built vector index lives in `data/chroma_db/`. It is populated from O*NET occupation profiles and a curated job description corpus. If the directory is missing or empty, run the ingestion scripts once:
```bash
python prepare_onet.py
python prepare_jds.py
python -m src.ingest
```

---

## ▶️ Running the App

```bash
source .venv/bin/activate
python app.py
```

Then open **http://localhost:7860** in Chrome or Firefox.

From the UI you can:
- Paste a resume and job description directly, **or** upload a `.pdf` / `.docx` file
- Load one of 6 built-in example pairs from the **Example** picker
- Click **Analyze** to run the full pipeline (~40 s on M-series hardware)
- Click **Clear** to reset all inputs and outputs

---

## 📊 Running the Evaluation

A 5-case automated evaluation harness measures keyword alignment improvement and LLM match scores across diverse role types.

```bash
source .venv/bin/activate
python -m src.evaluate
```

Sample output:
```
Test Case                                Before      After  Δ Improvement  LLM Score
Software Engineer (Python / PyTorch)      9.5%      28.6%        +19.0%     38/100
Data Scientist (SQL / scikit-learn)      10.9%      30.4%        +19.6%     45/100
Product Manager (Roadmap / Agile)        15.4%      40.4%        +25.0%     42/100
Marketing Manager (SEO / Google Ads)     15.6%      35.6%        +20.0%     38/100
Data Analyst (Tableau / SQL / BI)        11.4%      34.1%        +22.7%     36/100
AVERAGE                                  12.5%      33.8%        +21.3%
```

---

## 📁 Project Structure

```
GenAI Project/
├── app.py                  # Gradio UI — file upload, example picker, run_analysis handler
├── requirements.txt        # Pinned dependencies
├── prepare_onet.py         # One-time: parse O*NET XLSX → JSONL chunks
├── prepare_jds.py          # One-time: parse job description CSVs → JSONL chunks
│
├── src/
│   ├── __init__.py
│   ├── pipeline.py         # Orchestrator: analyze(), load_all(), keyword overlap
│   ├── generate.py         # LLM prompt, JSON schema, generate_feedback()
│   ├── retrieve.py         # Two-stage retrieval: ChromaDB dense → CrossEncoder rerank
│   ├── embed.py            # Embedding model + ChromaDB collection singleton
│   ├── ingest.py           # Ingests processed chunks into ChromaDB
│   └── evaluate.py         # 5-case eval harness with before/after keyword metrics
│
└── data/
    ├── chroma_db/          # Persistent ChromaDB vector index
    ├── processed/
    │   ├── all_chunks.jsonl
    │   ├── jd_corpus.jsonl
    │   └── onet_profiles.jsonl
    └── raw/
        ├── jd_dataset.csv
        ├── jd_dataset2.csv
        └── onet/
            ├── Occupation Data.xlsx
            └── Skills.xlsx
```

---

## 👥 Team

| Name | Program |
|---|---|
| **Ayush Shrivastava** | CSIT 595 — Generative AI Applications, Montclair State University |
| **Vishwa Patel** | CSIT 595 — Generative AI Applications, Montclair State University |
| **Senoussi Abdoulkarim** | CSIT 595 — Generative AI Applications, Montclair State University |

---

## 📝 License

This project was developed for academic purposes as part of the CSIT 595 course at Montclair State University. Not intended for commercial use.
