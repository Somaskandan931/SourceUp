# SOURCEUP
![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat&logo=python&logoColor=white)
![Java](https://img.shields.io/badge/Java-11+-ED8B00?style=flat&logo=openjdk&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=flat&logo=fastapi&logoColor=white)
![React](https://img.shields.io/badge/React-18+-61DAFB?style=flat&logo=react&logoColor=black)
![LightGBM](https://img.shields.io/badge/LightGBM-LambdaRank-brightgreen?style=flat)
![scikit-learn](https://img.shields.io/badge/scikit--learn-metrics-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![Groq](https://img.shields.io/badge/Groq-LLM-F55036?style=flat)
![FAISS](https://img.shields.io/badge/FAISS-vector%20search-blue?style=flat)
![Sentence Transformers](https://img.shields.io/badge/SBERT-all--MiniLM--L6--v2-yellow?style=flat)
![Redis](https://img.shields.io/badge/Redis-session%20memory-DC382D?style=flat&logo=redis&logoColor=white)
![Airflow](https://img.shields.io/badge/Airflow-2.x-017CEE?style=flat&logo=apache-airflow&logoColor=white)
![Maven](https://img.shields.io/badge/Maven-build-C71A36?style=flat&logo=apache-maven&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat)
![Research](https://img.shields.io/badge/IEEE-In%20Progress-orange?style=flat)
**Constraint-Aware Explainable AI Framework for SME Procurement**

`FastAPI` · `LightGBM LambdaRank` · `FAISS + SBERT` · `Groq LLM` · `React` · `Java Selenium`

---

## Table of Contents

1. [Overview](#1-overview)
2. [System Architecture](#2-system-architecture)
3. [Core Technical Contributions](#3-core-technical-contributions)
4. [Project Structure](#4-project-structure)
5. [Prerequisites & Installation](#5-prerequisites--installation)
6. [How to Run — Step by Step](#6-how-to-run--step-by-step)
7. [API Reference](#7-api-reference)
8. [Evaluation & Research Metrics](#8-evaluation--research-metrics)
9. [Airflow DAG (Optional)](#9-airflow-dag-optional)
10. [Known Issues & Configuration Notes](#10-known-issues--configuration-notes)
11. [Research Roadmap](#11-research-roadmap)
12. [Academic Context](#12-academic-context)

---

## 1. Overview

SourceUp is a research-grade AI procurement framework designed for small and medium enterprises (SMEs). It goes beyond conventional supplier directories by combining semantic retrieval, hard business-constraint filtering, and a Learning-to-Rank (LTR) model trained with weak supervision. Every recommendation comes with a transparent, auditable decision trace — telling users not just *what* was recommended, but *why*.

**Research question:**
> "How can constraint-aware explainable ranking improve decision quality, feasibility, and trust in SME supplier selection compared to traditional recommender systems?"

This reframes SourceUp from an application into a measurable AI decision-making framework suitable for academic publication.

| Before | After |
|--------|-------|
| AI sourcing app | AI research framework |
| Filter + rank pipeline | Constraint-aware ranking problem |
| UI explainability | Measurable research variable |
| Heuristic scoring | Learnable ranking objective |
| Metrics only | Controlled experiments |

---

## 2. System Architecture

SourceUp is composed of six integrated layers that form the end-to-end procurement decision pipeline:

| Component | Responsibility |
|-----------|---------------|
| **Java Scraper (`App.java`)** | Selenium-based crawler for TradeIndia. Extracts product cards, prices, MOQ, certifications, phone numbers, and computes a composite feasibility score per supplier. |
| **Data Pipeline (`pipeline/`)** | Three-stage ETL: `validate_and_merge` → `clean_normalize` → `incremental_faiss`. Produces `suppliers_clean.csv` and a FAISS flat-L2 index over `all-MiniLM-L6-v2` embeddings. |
| **Feature Builder (`feature_builder.py`)** | Constructs 10 LTR features per (query, supplier) pair and assigns weakly-supervised relevance labels (0–5 scale) derived from the composite heuristic score. |
| **FastAPI Backend (`backend/`)** | Serves `/recommend` and `/chat`. Orchestrates retrieval → constraint filtering → LightGBM ranking → explanation generation. Powered by Groq LLM for conversational queries. |
| **Sourcebot NLU (`sourcebot/`)** | Rule-based + GPT-fallback parser that classifies user intent and extracts product, budget, delivery, and certification constraints from natural language queries. |
| **React Frontend (`App.js`)** | Single-page UI. Calls the FastAPI backend at port 8000. Displays ranked suppliers with decision traces and what-if simulation controls. |

---

## 3. Core Technical Contributions

### 3.1 Constraint-Aware Learning-to-Rank

The primary ranking function is a constrained optimisation problem:

```
Score(q, d) = f_θ(q, d) − γ · ConstraintViolation(d, C)
```

Where:
- `q` = user query
- `d` = supplier candidate
- `C` = SME constraints (budget, delivery, MOQ, certifications)
- `f_θ` = LightGBM LambdaRank model
- `γ` = constraint penalty weight (tunable via `--gamma` flag)

Extended formulation with soft constraint relaxation:

```
Score(q, d) = λ₁ · Relevance(q, d) + λ₂ · Feasibility(d, C)
Feasibility(d) = Σ wᵢ · 1(constraint_i satisfied)
```

### 3.2 LightGBM LambdaRank (`train_lambdarank.py`)

The ranker uses a pairwise learning-to-rank objective that optimises NDCG directly via gradient approximation (Burges et al., 2006). Ten features are used per (query, supplier) pair:

| Feature | Description |
|---------|-------------|
| `price_match` | Normalised price within budget ceiling |
| `price_competitiveness` | Relative price vs. the full query candidate set |
| `location_match` | City-level delivery score (lookup table of 17 Indian cities) |
| `cert_match` | Certification overlap with user requirements |
| `years_normalized` | Years on platform, capped at 10 |
| `is_manufacturer` | Binary business-type flag |
| `faiss_score` | Semantic similarity distance from SBERT retrieval |
| `moq_feasibility` | Minimum order quantity vs. SME capacity |
| `composite_score` | Heuristic baseline score from the Java scraper |
| `violation_count` | Number of hard constraints breached |

Pre-trained models `ranker_lightgbm.pkl` and `ranker_xgboost.pkl` are included in `backend/app/models/embeddings/`. The ranker degrades gracefully to a rule-based scorer if neither LightGBM nor XGBoost is installed.

### 3.3 Constraint Engine (`constraint_engine.py`)

Enforces hard filters before ranking. Supported constraints:

- `max_price` — budget ceiling per unit
- `max_moq` — maximum affordable minimum order quantity
- `max_lead_time` — delivery urgency in days
- `min_certifications` — minimum certification count required
- `location` — domestic vs. international preference

Each filtered supplier is annotated with the specific constraint it failed, enabling the decision trace to explain exclusions.

### 3.4 Explainability Services

Three dedicated modules provide transparency at different levels:

| Module | Function |
|--------|----------|
| `explanation.py` | Generates human-readable reason strings for each recommendation: price fit, location proximity, certification coverage, experience level. |
| `decision_trace.py` | Full audit trail breaking down semantic similarity, price, location, certification, and experience contributions plus constraint penalties. |
| `what_if_simulator.py` | Simulates ranking changes when users adjust priorities. Supports: *"What if I increase my budget by 10%?"*, *"What if I relax certification requirements?"*, *"What if I prioritise price over speed?"* |

### 3.5 Sourcebot NLU (`sourcebot/`)

A two-layer NLU pipeline:

- **Rule-based parser (`rules.py`)** — fast regex extraction of product, budget, delivery days, location, and certifications. Blocks question starters (`who`, `what`, `why`, `how`…) from being misclassified as product queries.
- **GPT fallback (`gpt_fallback.py`)** — invoked when the rule parser returns low-confidence results. Uses Groq or Ollama for structured constraint extraction.
- **Intent classifier (`parser.py`)** — routes queries to `product_search`, `information`, `conversation`, or `explanation_request`.
- **Session memory (`session.py`)** — Redis/Memurai-backed conversation context. Stores parsed state across multi-turn dialogue.

---

## 4. Project Structure

```
SourceUp/
├── App.java                          Java scraper (TradeIndia / Selenium)
├── App.js                            React frontend
├── feature_builder.py                LTR feature + weak-supervision label construction
├── metrics.py                        IEEE-compliant ranking metrics (standalone)
├── sourceup_dag.py                   Airflow DAG (optional daily pipeline automation)
│
├── backend/
│   └── app/
│       ├── main.py                   FastAPI entry point (loads .env, mounts routers)
│       ├── api/
│       │   ├── recommend.py          /recommend endpoint — full ranking pipeline
│       │   └── chat.py               /chat endpoint — Groq LLM conversational interface
│       ├── models/
│       │   ├── retriever.py          FAISS semantic search (all-MiniLM-L6-v2)
│       │   ├── ranker.py             LightGBM / XGBoost / rule-based scorer
│       │   ├── constraint_engine.py  Hard constraint filtering
│       │   ├── train_lambdarank.py   LambdaRank training script (primary)
│       │   ├── train_ranker.py       Legacy XGBoost training script
│       │   └── embeddings/           Pre-trained .pkl model files
│       └── services/
│           ├── explanation.py        Human-readable recommendation reasons
│           ├── decision_trace.py     Full per-supplier audit trail
│           └── what_if_simulator.py  Priority/constraint trade-off simulation
│
├── sourcebot/
│   ├── orchestrator.py               Main conversation handler + intent routing
│   ├── nlu/
│   │   ├── parser.py                 Intent classification + dispatch
│   │   ├── rules.py                  Regex-based constraint extraction
│   │   └── gpt_fallback.py           LLM-based fallback parsing
│   ├── memory/
│   │   └── session.py                Redis/Memurai session storage
│   └── responses/
│       └── info_responses.py         AI-generated informational answers
│
├── pipeline/
│   ├── run_all.py                    Full ETL pipeline entry point
│   ├── validate_merge.py             Merge and validate all scraped CSVs
│   ├── clean_normalize.py            Data cleaning and normalisation
│   └── incremental_faiss.py          SBERT embedding + FAISS index build
│
├── eval/
│   ├── ablation.py                   5-variant ablation study (IEEE-compliant)
│   ├── baselines.py                  BM25 / SBERT / rule-based baseline comparison
│   └── sensitivity.py                Constraint-weight sensitivity analysis
│
└── data/                             Created at runtime — not committed to repo
    ├── outputs/                      Raw scraped CSVs from Java scraper
    ├── merged/                       suppliers_all.csv
    ├── clean/                        suppliers_clean.csv
    ├── embeddings/                   suppliers.faiss + suppliers_meta.csv
    ├── training/                     ranking_data.csv (from feature_builder.py)
    └── eval/                         ablation_results.csv + plots
```

---

## 5. Prerequisites & Installation

### 5.1 System Requirements

| Dependency | Details |
|------------|---------|
| Python | 3.9 or higher |
| Java + Maven | JDK 11+ and Maven 3.6+ (scraper only) |
| Node.js | 18+ (React frontend) |
| Chrome + ChromeDriver | Auto-managed by WebDriverManager in the Java scraper |
| Redis or Memurai | For chat session memory. Memurai is a Windows-native Redis-compatible server. Optional — the `/recommend` API works without it, but multi-turn chat loses context between requests. |
| Groq API key | Free tier at [console.groq.com](https://console.groq.com). Required for `/chat` and GPT-fallback NLU. `/recommend` works without it using the rule-based parser. |

### 5.2 Python Dependencies

```bash
pip install fastapi uvicorn python-dotenv groq \
            sentence-transformers faiss-cpu \
            lightgbm xgboost pandas numpy scikit-learn \
            scipy redis matplotlib seaborn
```

### 5.3 Frontend Dependencies

```bash
# From the directory containing App.js
npm install
```

### 5.4 Environment Variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
REDIS_HOST=localhost        # optional, default: localhost
REDIS_PORT=6379             # optional, default: 6379
SOURCEUP_DIR=/path/to/project   # optional, overrides hardcoded paths
```

### 5.5 Fix Hardcoded Paths

> **Important:** Several files contain absolute Windows paths (`C:/Users/somas/PycharmProjects/SourceUp` or `D:/PycharmProjects/SourceUp`). Before running anything, do a project-wide find-and-replace with your actual project directory path, or set the `SOURCEUP_DIR` environment variable.

Files to fix:

- `backend/app/models/retriever.py`
- `backend/app/models/ranker.py`
- `pipeline/validate_merge.py`
- `pipeline/clean_normalize.py`
- `pipeline/incremental_faiss.py`
- `feature_builder.py`

---

## 6. How to Run — Step by Step

Run these steps in order. Steps 1–3 are one-time setup. Steps 4–5 are the running application.

---

### Step 1 — Java Scraper

Collects supplier data from TradeIndia and writes CSVs to `data/outputs/`.

```bash
# Build the JAR
mvn clean package -q

# Run the scraper
java -jar target/global-sources-scraper-1.0-SNAPSHOT.jar \
     input.csv output.csv [max_budget] [max_delivery_days] [start_page] [end_page]
```

**Example:**

```bash
java -jar target/*.jar queries.csv data/outputs/result.csv 50000 7 1 5
```

`input.csv` must have a header row with search queries in the first column (e.g., `"industrial bearings"`, `"LED strips wholesale"`).

You also need a `data/test_output.csv` as the canonical schema reference. Create it by saving any one scraper output CSV as `data/test_output.csv`.

---

### Step 2 — Data Pipeline

Merges all CSVs in `data/outputs/`, cleans and normalises the data, then builds the FAISS semantic index.

```bash
python pipeline/run_all.py
```

This produces:

- `data/merged/suppliers_all.csv`
- `data/clean/suppliers_clean.csv`
- `data/embeddings/suppliers.faiss`
- `data/embeddings/suppliers_meta.csv`

---

### Step 3 — Build Features & Train Ranker

Both commands must complete before starting the backend. They can be run in any order.

```bash
# Build LTR training dataset
python feature_builder.py

# Train LightGBM LambdaRank model
python backend/app/models/train_lambdarank.py

# Optional: tune the constraint penalty weight
python backend/app/models/train_lambdarank.py --gamma 0.3
```

This outputs `ranker_lightgbm.pkl` to `backend/app/models/embeddings/`.

> **Note:** Pre-trained models are already included in `backend/app/models/embeddings/`. Skip this step if the scraped data has not changed.

---

### Step 4 — Start the Backend

```bash
# From the project root
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```

Verify it is running:
- Health check: `http://localhost:8000`
- Swagger UI: `http://localhost:8000/docs`

On startup the backend prints environment diagnostics including whether `GROQ_API_KEY` is loaded and how many suppliers are in the FAISS index.

---

### Step 5 — Start the React Frontend

```bash
# From the frontend directory
npm start
```

Opens `http://localhost:3000`. The frontend makes API calls to `http://localhost:8000`.

---

### Step 6 — Evaluation Scripts (Optional — for Research / IEEE Paper)

```bash
python eval/ablation.py        # 5-variant ablation study
python eval/baselines.py       # BM25 / SBERT / rule-based baselines
python eval/sensitivity.py     # Constraint-weight sensitivity analysis
python metrics.py              # Standalone metric unit tests
```

Results and plots are saved to `data/eval/`.

---

## 7. API Reference

### `POST /recommend`

The primary procurement recommendation endpoint.

```http
POST http://localhost:8000/recommend
Content-Type: application/json

{
  "query": "industrial ball bearings",
  "max_price": 500,
  "max_lead_time": 7,
  "min_certifications": 1,
  "location": "Mumbai",
  "top_k": 10
}
```

Response includes ranked suppliers, constraint metadata, decision traces, and human-readable explanation strings for each result.

---

### `POST /chat`

Conversational interface. Accepts free-text queries and routes through the Sourcebot NLU pipeline before calling `/recommend` internally.

```http
POST http://localhost:8000/chat
Content-Type: application/json

{
  "session_id": "user-abc-123",
  "message": "Find me a PCB manufacturer under Rs 200 with ISO certification"
}
```

Supports follow-up queries such as:
- *"Why was the first supplier ranked higher than the third?"*
- *"What if I prioritise price over delivery speed?"*
- *"Explain why this supplier was recommended."*

---

### `POST /what-if`

Simulates how rankings change when priorities or constraints are adjusted.

```http
POST http://localhost:8000/what-if
Content-Type: application/json

{
  "suppliers": [...],
  "scenario": "increase_budget",
  "parameters": { "budget_multiplier": 1.1 }
}
```

---

### `GET /health`

Returns backend health status, FAISS index size, total supplier count, and whether Groq is configured.

---

## 8. Evaluation & Research Metrics

SourceUp implements IEEE-standard learning-to-rank evaluation in `metrics.py` and the `eval/` scripts.

### 8.1 Ranking Quality Metrics

| Metric | Description |
|--------|-------------|
| **NDCG@5, NDCG@10** | Primary ranking quality metric. Implemented per-query in `metrics.py` and batch (query-grouped) in eval scripts. |
| **Precision@5, Precision@10** | Fraction of top-k results that are relevant (relevance threshold > 2). |
| **MAP** | Mean Average Precision across all queries. |
| **Kendall's Tau** | Rank correlation between predicted and ground-truth orderings. Measures ranking stability under small input changes. |
| **MRR** | Mean Reciprocal Rank — position of the first relevant result across queries. |

### 8.2 Ablation Study Variants

`eval/ablation.py` evaluates five system configurations:

| Variant | Configuration |
|---------|--------------|
| **V1 — Full Model** | SBERT retrieval + Constraint Engine + LightGBM LambdaRank. Primary contribution. |
| **V2 — No Constraints** | SBERT + LightGBM, constraint filtering removed. Isolates the impact of constraint enforcement. |
| **V3 — No LTR** | SBERT + Constraints + rule-based scorer. Isolates the contribution of the learned ranker. |
| **V4 — No Semantic** | BM25 retrieval + Constraints + LightGBM. Removes SBERT semantic matching. |
| **V5 — Rule-Based Only** | Heuristic scorer only. No SBERT, no LTR, no constraints. Weakest baseline. |

Metrics reported per variant: NDCG@10, Precision@5, MAP, Constraint Violation Rate, Kendall's Tau.

### 8.3 Constraint Metrics

- **Constraint Violation Rate (CVR)** — percentage of top-10 results that violate at least one hard constraint.
- **Feasibility Score** — weighted sum of satisfied constraints across the returned candidate set.
- **Sensitivity analysis** (`eval/sensitivity.py`) varies γ (constraint penalty weight) from 0.0 to 1.0 and plots NDCG vs. CVR trade-off curves.

### 8.4 Weak Supervision Justification

Training labels are derived from the composite heuristic score computed in the Java scraper:

```
composite_score = 0.35 × price_score + 0.25 × delivery_score + 0.40 × reliability_score
```

This is the standard bootstrapping approach for LTR datasets without explicit human relevance judgements (Joachims et al., 2002). Robustness to label noise is validated by the sensitivity analysis in `eval/sensitivity.py`.

---

## 9. Airflow DAG (Optional)

`sourceup_dag.py` defines an Airflow DAG that runs `python pipeline/run_all.py` on a daily schedule, automating the ETL refresh without manual intervention.

```python
# sourceup_dag.py
with DAG("sourceup", start_date=datetime(2024,1,1), schedule="@daily") as dag:
    run = BashOperator(
        task_id="pipeline",
        bash_command="python pipeline/run_all.py"
    )
```

To use it, copy `sourceup_dag.py` into your Airflow `dags/` directory. Requires Apache Airflow 2.x installed and configured separately. The rest of SourceUp runs fully without Airflow — it is an optional production convenience.

---

## 10. Known Issues & Configuration Notes

| Issue | Resolution |
|-------|-----------|
| **Hardcoded Windows paths** | Find-and-replace `C:/Users/somas/PycharmProjects/SourceUp` throughout the project, or set the `SOURCEUP_DIR` environment variable. |
| **Redis / session memory unavailable** | Install Redis (Linux/macOS) or Memurai (Windows, Redis-compatible). If unavailable, session context is lost between chat turns but `/recommend` still works fully. |
| **FAISS index missing on startup** | Run `pipeline/run_all.py` first. The backend raises `RuntimeError` if `suppliers.faiss` does not exist. |
| **`GROQ_API_KEY` not set** | The `/chat` endpoint and GPT-fallback NLU will fail. `/recommend` continues to work using the rule-based parser and the pre-trained ranker. |
| **Scraper captchas / rate limiting** | TradeIndia may throttle requests. Increase sleep delays in `App.java` or add a residential proxy. The scraper already randomises wait times and uses stealth Chrome flags to reduce detection. |
| **`data/test_output.csv` missing** | This file defines the canonical CSV schema for the merge step. Create it by saving any one scraper output CSV as `data/test_output.csv`. |

---

## 11. Research Roadmap

Proposed extensions for future work:

- **Multilingual sourcing** — extend the NLU to support Tamil, Hindi, and other Indian regional languages for broader SME reach.
- **Reinforcement learning for negotiation** — an agent that optimises multi-round price negotiation within supplier constraints.
- **Real interaction logs** — replace weak supervision labels with click-through data from deployed production usage.
- **Fairness-aware ranking** — add exposure parity constraints to prevent systematic under-exposure of smaller or newer suppliers.
- **Constraint relaxation engine** — suggest the minimum constraint adjustment that would expand the feasible supplier set when no results satisfy all hard constraints.
- **Interactive feedback loop** — allow users to refine constraints dynamically and observe real-time rank changes within the UI.

---

## 12. Academic Context

### Problem Framing

SourceUp is positioned as a constraint-aware learning-to-rank framework, not merely a sourcing application. The core contribution is the formalisation of SME procurement as a constrained optimisation problem:

```
Score(q, d) = f_θ(q, d) − γ · ConstraintViolation(d, C)
```

This enables ablation, sensitivity analysis, and controlled comparison against standard IR baselines — the requirements for IEEE-level publication.

### Key References

**LambdaRank:**
Burges et al. (2006). *Learning to Rank using Gradient Descent.* ICML 2005.
Burges, C. (2010). *From RankNet to LambdaRank to LambdaMART: An Overview.* MSR-TR-2010-82.

**Weak supervision:**
Joachims, T. (2002). *Optimizing Search Engines using Clickthrough Data.* KDD 2002.
Labels derived from production heuristics are the standard approach for bootstrapping LTR datasets without explicit relevance judgements.

**SBERT retrieval:**
Reimers & Gurevych (2019). *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks.* EMNLP 2019.

### Publication Readiness Checklist

- [x] Research problem formally defined
- [x] Mathematical formulation with learnable objective
- [x] Weak supervision labels scientifically justified
- [x] Strong baselines included (BM25, SBERT cosine, rule-based)
- [x] Ablation study across 5 system variants
- [x] Sensitivity analysis on constraint weight γ
- [x] IEEE-standard metrics (NDCG, MAP, P@k, Kendall's Tau)
- [x] Constraint-specific metrics (CVR, feasibility score)
- [x] Case studies (budget-constrained, urgent, conflicting constraints)
- [x] Failure and trade-off analysis
- [ ] User study / human evaluation (future work)
- [ ] Real interaction log data (future work)

---

*SourceUp — Built for SME procurement research. All components are modular and independently testable.*