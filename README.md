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
![Razorpay](https://img.shields.io/badge/Razorpay-billing-02042B?style=flat&logo=razorpay&logoColor=white)
![JWT](https://img.shields.io/badge/JWT-auth-black?style=flat&logo=jsonwebtokens&logoColor=white)
![SHAP](https://img.shields.io/badge/SHAP-XAI-orange?style=flat)
![License](https://img.shields.io/badge/License-MIT-green?style=flat)
![Research](https://img.shields.io/badge/IEEE-In%20Progress-orange?style=flat)

**Constraint-Aware Explainable AI Framework for SME Procurement**

`FastAPI` · `LightGBM LambdaRank` · `FAISS + SBERT` · `Groq LLM` · `React` · `Java Selenium` · `Razorpay` · `JWT Auth` · `SHAP`

---

## Overview

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

## TL;DR (What this does in 30 seconds)

- Enter a product query (e.g., "industrial bearings under ₹500")
- System retrieves suppliers using SBERT + FAISS
- Filters by real-world constraints (budget, MOQ, delivery, certifications)
- Ranks using LightGBM LambdaRank
- Explains *why* each supplier is recommended (SHAP + decision trace)
- Generates RFQ emails instantly via LLM

Result: A transparent, constraint-aware procurement decision system for SMEs.

## Example Output

**Query:**
"PCB manufacturer under ₹200 with ISO certification"

**Top Result:**
- Supplier: ABC Circuits Pvt Ltd
- Score: 0.91
- Rank: #1

**Why recommended:**
- ✅ Price within budget (₹180)
- ✅ ISO 9001 certified
- ✅ 5+ years experience
- ⚠️ Slightly higher MOQ

**Decision Trace (simplified):**
- Semantic similarity: 0.82
- Price score: +0.25
- Certification match: +0.30
- Constraint penalty: -0.05

## System Architecture

SourceUp is composed of eight integrated layers that form the end-to-end procurement decision pipeline:

| Component | Responsibility |
|-----------|---------------|
| **Java Scraper (`App.java`)** | Selenium-based crawler for TradeIndia. Extracts product cards, prices, MOQ, certifications, phone numbers, and computes a composite feasibility score per supplier. |
| **Data Pipeline (`pipeline/`)** | Four-stage ETL: `validate_and_merge` → `clean_normalize` → `incremental_faiss` → `feature_builder`. Produces `suppliers_clean.csv`, a FAISS flat-L2 index over `all-MiniLM-L6-v2` embeddings, and LTR training data. |
| **Feature Builder (`feature_builder.py`)** | Constructs 10 LTR features per (query, supplier) pair and assigns weakly-supervised relevance labels (0–5 scale) derived from the composite heuristic score. |
| **FastAPI Backend (`backend/`)** | Serves `/recommend`, `/chat`, `/quote`, `/what-if`, `/compare`, `/stats`, and `/auth`. Orchestrates retrieval → constraint filtering → LightGBM ranking → explanation generation. Powered by Groq LLM for conversational and RFQ endpoints. |
| **Sourcebot NLU (`sourcebot/`)** | Rule-based + GPT-fallback parser that classifies user intent and extracts product, budget, delivery, and certification constraints from natural language queries. Redis-backed session memory for multi-turn dialogue. |
| **React Frontend (`App.js`)** | Single-page UI with tabbed Search and Chat interfaces, onboarding wizard, JWT authentication modal, Razorpay billing modal, RFQ quote-draft modal, decision-trace display, and what-if simulation controls. Calls the FastAPI backend at port 8000. |
| **Auth & Billing (`backend/app/api/auth.py`)** | JWT-based user registration and login. Razorpay payment order creation and webhook verification. Plan management (Free / Pro / Enterprise). |
| **Quote Generation (`backend/app/api/quote.py`)** | Groq LLM-powered RFQ email drafting. Accepts supplier and product context; returns a ready-to-send email subject and body. Supports iterative refinement via `/quote/refine`. |

---

## Core Technical Contributions

### 1. Constraint-Aware Learning-to-Rank

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

### 2. LightGBM LambdaRank (`train_lambdarank.py`)

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

### 3. Constraint Engine (`constraint_engine.py`)

Enforces hard filters before ranking. Supported constraints:

- `max_price` — budget ceiling per unit
- `moq_budget` — total affordable MOQ spend
- `max_lead_time` — delivery urgency in days
- `min_certifications` — minimum certification count required
- `location` — preferred location (hard or soft, controlled by `location_mandatory` flag)
- `min_years_experience` — minimum years on platform
- `required_certifications` — list of specific certifications required

Each filtered supplier is annotated with the specific constraint it failed, enabling the decision trace to explain exclusions.

### 4. Explainability Services

Three dedicated modules provide transparency at different levels:

| Module | Function |
|--------|----------|
| `explanation.py` | Generates human-readable reason strings for each recommendation: price fit, location proximity, certification coverage, experience level. |
| `decision_trace.py` | Full audit trail breaking down semantic similarity, price, location, certification, and experience contributions plus constraint penalties. Also supports comparative trace between two suppliers. |
| `what_if_simulator.py` | Simulates ranking changes when users adjust priorities. Supports: *"What if I increase my budget by 10%?"*, *"What if I relax certification requirements?"*, *"What if I prioritise price over speed?"* |

### 5. SHAP Feature Attribution (`eval/shap_analysis.py`)

Uses the SHAP library (Lundberg & Lee, NeurIPS 2017) to generate model-grounded feature attributions from the trained LightGBM ranker. Produces IEEE-publication-grade plots:

- Global feature importance (beeswarm + bar)
- Per-query SHAP beeswarm (each supplier's score composition)
- Dependence plots for top 3 features
- Force plot for the top-ranked supplier per example query
- SHAP heatmap + reproducible CSV

This upgrades decision traces from *heuristic reasons* to *model-grounded explanations*.

### 6. Sourcebot NLU (`sourcebot/`)

A two-layer NLU pipeline:

- **Rule-based parser (`rules.py`)** — fast regex extraction of product, budget, delivery days, location, and certifications. Blocks question starters (`who`, `what`, `why`, `how`…) from being misclassified as product queries.
- **GPT fallback (`gpt_fallback.py`)** — invoked when the rule parser returns low-confidence results. Uses Groq or Ollama for structured constraint extraction.
- **Intent classifier (`parser.py`)** — routes queries to `product_search`, `information`, `conversation`, or `explanation_request`.
- **Session memory (`session.py`)** — Redis/Memurai-backed conversation context. Stores parsed state across multi-turn dialogue.

### 7. Auth & Billing (`auth.py`)

- JWT user registration and login via `python-jose` + `passlib[bcrypt]`.
- Razorpay payment order creation, webhook HMAC verification, and plan management (Free / Pro / Enterprise).
- All auth endpoints are protected via `HTTPBearer` token middleware.
- Graceful degradation: the `/recommend` and `/chat` endpoints continue working if auth dependencies are not installed.

### 8. Quote Generation (`quote.py`)

- `POST /quote/draft` — accepts supplier name, product, quantity, target price, delivery location, certification, and buyer details. Returns a ready-to-send RFQ email (subject + body) generated by Groq LLM.
- `POST /quote/refine` — accepts an existing draft and a refinement instruction (e.g. *"Make it shorter"*, *"Add ISO 9001 requirement"*). Returns the revised draft.
- Frontend exposes a **"Draft RFQ"** button per supplier card that opens a pre-filled quote modal.

---

## Why SourceUp is Different

- Combines **IR + ML + Systems + LLMs** (rare combo)
- Moves beyond recommendation → **decision intelligence**
- Built with **research rigor (IEEE-level evaluation)**
- Fully **explainable + auditable pipeline**

## Project Structure

```
SourceUp/
├── App.java                          Java scraper (TradeIndia / Selenium)
├── App.js                            React frontend (tabbed UI, auth modal, billing, quote modal)
├── feature_builder.py                LTR feature + weak-supervision label construction
├── metrics.py                        IEEE-compliant ranking metrics (standalone)
├── sourceup_dag.py                   Airflow DAG (optional daily pipeline automation)
├── config.py                         Centralised path + env config (no hardcoded paths)
│
├── backend/
│   └── app/
│       ├── main.py                   FastAPI entry point (loads config, mounts routers)
│       ├── api/
│       │   ├── recommend.py          /recommend, /what-if, /compare — full ranking pipeline
│       │   ├── chat.py               /chat — Groq LLM conversational interface
│       │   ├── quote.py              /quote/draft, /quote/refine — RFQ email generation
│       │   └── auth.py               /auth/* — JWT auth + Razorpay billing
│       ├── models/
│       │   ├── retriever.py          FAISS semantic search (all-MiniLM-L6-v2)
│       │   ├── ranker.py             LightGBM / XGBoost / rule-based scorer
│       │   ├── constraint_engine.py  Hard constraint filtering
│       │   ├── train_lambdarank.py   LambdaRank training script (primary)
│       │   ├── train_ranker.py       Legacy XGBoost training script
│       │   └── embeddings/           Pre-trained .pkl model files
│       └── services/
│           ├── explanation.py        Human-readable recommendation reasons
│           ├── decision_trace.py     Full per-supplier audit trail + comparative trace
│           └── what_if_simulator.py  Priority/constraint trade-off simulation
│
├── sourcebot/
│   ├── orchestrator.py               Main conversation handler + intent routing
│   ├── nlu/
│   │   ├── parser.py                 Intent classification + dispatch
│   │   ├── rules.py                  Regex-based constraint extraction
│   │   └── gpt_fallback.py           LLM-based fallback parsing
│   ├── memory/
│   │   ├── session.py                Redis/Memurai session storage
│   │   └── test_memurai.py           Redis connection sanity test
│   └── responses/
│       └── info_responses.py         AI-generated informational answers
│
├── pipeline/
│   ├── run_all.py                    Full ETL pipeline entry point (4-stage)
│   ├── validate_merge.py             Merge and validate all scraped CSVs
│   ├── clean_normalize.py            Data cleaning and normalisation
│   └── incremental_faiss.py          SBERT embedding + FAISS index build
│
├── eval/
│   ├── ablation.py                   5-variant ablation study (IEEE-compliant)
│   ├── baselines.py                  BM25 / SBERT / rule-based baseline comparison
│   ├── sensitivity.py                Constraint-weight sensitivity analysis
│   ├── shap_analysis.py              SHAP feature attribution (NeurIPS 2017)
│   ├── fairness.py                   Geographic exposure-disparity analysis
│   ├── stability.py                  Rank stability under input perturbations
│   └── label_noise_analysis.py       Weak-supervision robustness (10–40% label noise)
│
├── evaluation/
│   └── metrics.py                    Standalone IEEE metric unit tests
│
└── data/                             Created at runtime — not committed to repo
    ├── outputs/                      Raw scraped CSVs from Java scraper
    ├── merged/                       suppliers_all.csv
    ├── clean/                        suppliers_clean.csv
    ├── embeddings/                   suppliers.faiss + suppliers_meta.csv
    ├── training/                     ranking_data.csv (from feature_builder.py)
    └── eval/                         ablation_results.csv + plots/
```

---

## Prerequisites & Installation

### 1. System Requirements

| Dependency | Details |
|------------|---------|
| Python | 3.9 or higher |
| Java + Maven | JDK 11+ and Maven 3.6+ (scraper only) |
| Node.js | 18+ (React frontend) |
| Chrome + ChromeDriver | Auto-managed by WebDriverManager in the Java scraper |
| Redis or Memurai | For chat session memory. Memurai is a Windows-native Redis-compatible server. Optional — the `/recommend` API works without it, but multi-turn chat loses context between requests. |
| Groq API key | Free tier at [console.groq.com](https://console.groq.com). Required for `/chat`, `/quote`, and GPT-fallback NLU. `/recommend` works without it using the rule-based parser. |

### 2. Python Dependencies

```bash
pip install fastapi uvicorn python-dotenv groq \
            sentence-transformers faiss-cpu \
            lightgbm xgboost pandas numpy scikit-learn \
            scipy redis matplotlib seaborn shap \
            python-jose[cryptography] passlib[bcrypt] razorpay
```

### 3. Frontend Dependencies

```bash
# From the directory containing App.js
npm install
```

### 4. Environment Variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here

# Auth & Billing
SECRET_KEY=your_jwt_secret_here        # generate: openssl rand -hex 32
RAZORPAY_KEY_ID=your_razorpay_key
RAZORPAY_KEY_SECRET=your_razorpay_secret

# Optional Redis
REDIS_HOST=localhost
REDIS_PORT=6379

# Optional path override
SOURCEUP_DIR=/path/to/project
```

Billing endpoints (`/auth/billing/*`) and auth endpoints (`/auth/register`, `/auth/login`) require all three auth+billing keys. The `/recommend` and `/chat` endpoints work without them.

### 5. Path Configuration

> **Important:** The project now ships with a centralised `config.py` that reads `SOURCEUP_DIR` from the environment. If you set `SOURCEUP_DIR` in `.env`, no manual path editing is required.
>
> If you prefer not to use the env var, do a project-wide find-and-replace of `C:/Users/somas/PycharmProjects/SourceUp` with your actual project path in:
> `backend/app/models/retriever.py`, `backend/app/models/ranker.py`,
> `pipeline/validate_merge.py`, `pipeline/clean_normalize.py`,
> `pipeline/incremental_faiss.py`, `feature_builder.py`

---

## How to Run — Step by Step

Run these steps in order. Steps 1–3 are one-time setup. Steps 5–6 are the running application.

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

After the first run, create the canonical schema reference:

```bash
cp data/outputs/result.csv data/test_output.csv
```

---

### Step 2 — Data Pipeline

Merges all CSVs in `data/outputs/`, cleans and normalises the data, builds the FAISS semantic index, and generates LTR training data.

```bash
python pipeline/run_all.py

# To skip the (slow) feature-builder step:
python pipeline/run_all.py --skip-features
```

This produces:

- `data/merged/suppliers_all.csv`
- `data/clean/suppliers_clean.csv`
- `data/embeddings/suppliers.faiss`
- `data/embeddings/suppliers_meta.csv`
- `data/training/ranking_data.csv` (unless `--skip-features`)

---

### Step 3 — Build Features & Train Ranker

```bash
# Build LTR training dataset (if skipped in Step 2)
python feature_builder.py

# Train LightGBM LambdaRank model (primary)
python backend/app/models/train_lambdarank.py

# Optional: tune the constraint penalty weight
python backend/app/models/train_lambdarank.py --gamma 0.3

# Optional: train XGBoost backup
python backend/app/models/train_ranker.py
```

Outputs `ranker_lightgbm.pkl` and `ranker_xgboost.pkl` to `backend/app/models/embeddings/`.

> **Note:** Pre-trained models are already included. Skip this step if the scraped data has not changed.

---

### Step 4 — Evaluation Scripts (Optional — for Research / IEEE Paper)

```bash
python eval/ablation.py              # 5-variant ablation study
python eval/baselines.py             # BM25 / SBERT / rule-based baselines
python eval/sensitivity.py           # Constraint-weight sensitivity analysis
python eval/shap_analysis.py         # SHAP feature attribution
python eval/fairness.py              # Geographic exposure-disparity analysis
python eval/stability.py             # Rank stability under perturbations
python eval/label_noise_analysis.py  # Label noise robustness analysis
python evaluation/metrics.py         # Standalone metric unit tests
```

Results and plots are saved to `data/eval/`.

---

### Step 5 — Start the Backend

```bash
# From the project root
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```

Verify it is running:
- Health check: `http://localhost:8000/health`
- Swagger UI: `http://localhost:8000/docs`

On startup the backend prints environment diagnostics including Groq key status, Razorpay billing status, project root path, and any missing-path warnings.

---

### Step 6 — Start the React Frontend

```bash
# From the frontend directory
npm start
```

Opens `http://localhost:3000`. The frontend calls `http://localhost:8000`.

On first visit, an **onboarding wizard** walks new users through the four core features (search, chat, quote, and explanations). Users can register or log in via the auth modal. The **Search** and **Chat** tabs are the primary interfaces. Each supplier card exposes a **"Draft RFQ"** button that opens the quote-generation modal powered by Groq.

---

## API Reference

### `POST /recommend`

The primary procurement recommendation endpoint.

```http
POST http://localhost:8000/recommend
Content-Type: application/json

{
  "product": "industrial ball bearings",
  "max_price": 500,
  "moq_budget": 10000,
  "max_lead_time": 7,
  "location": "Mumbai",
  "location_mandatory": false,
  "required_certifications": ["ISO 9001"],
  "min_years_experience": 3,
  "enable_explanations": true,
  "enable_what_if": false,
  "top_k": 10
}
```

Response includes ranked suppliers with `score`, `rank`, `reasons`, `decision_trace`, `constraint_results`, and `confidence_score` per result, plus `metadata` showing candidate counts and ranking method used.

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
- General B2B knowledge queries (ISO standards, compliance, export regulations) — answered directly by Groq LLM.

---

### `POST /what-if`

Simulates how rankings change when priorities or constraints are adjusted.

```http
POST http://localhost:8000/what-if
Content-Type: application/json

{
  "product": "industrial ball bearings",
  "constraints": { "max_price": 500 },
  "scenario": "price_over_speed"
}
```

Supported scenarios: `price_over_speed`, `speed_over_price`, `quality_over_cost`.

---

### `POST /compare`

Side-by-side supplier comparison with a comparative decision trace explaining exactly why one ranks higher than the other.

```http
POST http://localhost:8000/compare
Content-Type: application/json

[0, 2]   // supplier indices from a prior /recommend response
```

---

### `POST /quote/draft`

Generates a professional RFQ email using Groq LLM.

```http
POST http://localhost:8000/quote/draft
Content-Type: application/json

{
  "supplier_name": "ABC Bearings Pvt Ltd",
  "product_name": "Industrial Ball Bearings",
  "quantity": 500,
  "target_price": 450,
  "delivery_location": "Chennai",
  "required_certification": "ISO 9001",
  "lead_time_days": 7,
  "buyer_company": "My Company",
  "buyer_name": "Procurement Team"
}
```

Returns `{ "subject": "...", "body": "...", "tone": "professional" }`.

---

### `POST /quote/refine`

Refines an existing RFQ draft based on a user instruction.

```http
POST http://localhost:8000/quote/refine
Content-Type: application/json

{
  "original_draft": "...",
  "refinement_instruction": "Make it shorter and add ISO 14001 requirement"
}
```

---

### `POST /auth/register`

Registers a new user. Returns a JWT access token.

```http
POST http://localhost:8000/auth/register
Content-Type: application/json

{ "email": "user@example.com", "password": "secret" }
```

---

### `POST /auth/login`

Logs in and returns a JWT access token.

---

### `POST /auth/billing/create-order`

Creates a Razorpay payment order for plan upgrade.

---

### `POST /auth/billing/verify`

Verifies a Razorpay webhook payment signature.

---

### `GET /auth/billing/plans`

Returns available plans (Free / Pro / Enterprise) with features and pricing.

---

### `GET /health`

Returns backend health status, FAISS index size, total supplier count, Groq configuration status, and Razorpay configuration status.

---

### `GET /stats`

Returns top supplier locations and product categories from the FAISS index.

---

## Evaluation & Research Metrics

SourceUp implements IEEE-standard learning-to-rank evaluation across seven dedicated scripts.

### Ranking Quality Metrics

| Metric | Description |
|--------|-------------|
| **NDCG@5, NDCG@10** | Primary ranking quality metric. Per-query in `metrics.py`; batch (query-grouped) in eval scripts. |
| **Precision@5, Precision@10** | Fraction of top-k results that are relevant (relevance threshold > 2). |
| **MAP** | Mean Average Precision across all queries. |
| **Kendall's Tau** | Rank correlation between predicted and ground-truth orderings. Stability criterion: τ ≥ 0.85 at σ=0.03. |
| **MRR** | Mean Reciprocal Rank — position of the first relevant result. |

### Ablation Study Variants

`eval/ablation.py` evaluates five system configurations:

| Variant | Configuration |
|---------|--------------|
| **V1 — Full Model** | SBERT retrieval + Constraint Engine + LightGBM LambdaRank. Primary contribution. |
| **V2 — No Constraints** | SBERT + LightGBM, constraint filtering removed. Isolates constraint enforcement impact. |
| **V3 — No LTR** | SBERT + Constraints + rule-based scorer. Isolates the learned ranker's contribution. |
| **V4 — No Semantic** | BM25 retrieval + Constraints + LightGBM. Removes SBERT semantic matching. |
| **V5 — Rule-Based Only** | Heuristic scorer only. No SBERT, no LTR, no constraints. Weakest baseline. |

Metrics reported per variant: NDCG@10, Precision@5, MAP, Constraint Violation Rate, Kendall's Tau.

### Constraint Metrics

- **Constraint Violation Rate (CVR)** — percentage of top-10 results violating at least one hard constraint.
- **Feasibility Score** — weighted sum of satisfied constraints across the returned candidate set.
- **Sensitivity analysis** (`eval/sensitivity.py`) varies γ from 0.0 to 1.0 and plots NDCG vs. CVR trade-off curves.

### SHAP Feature Attribution (`eval/shap_analysis.py`)

Generates model-grounded explanations using the SHAP TreeExplainer on the trained LightGBM ranker:

- Global feature importance (beeswarm, bar plots)
- Per-query SHAP beeswarm showing each supplier's score composition
- Dependence plots for the top 3 most important features
- Force plot for the top-ranked supplier per example query
- SHAP heatmap + reproducible `shap_values.csv`

All plots saved to `data/eval/plots/`.

### Fairness Analysis (`eval/fairness.py`)

Measures geographic exposure disparity between metro (Mumbai, Delhi, Chennai, Bengaluru) and Tier-2/3 city suppliers at matched score levels:

- Exposure Ratio = avg_rank(Metro) / avg_rank(Tier-2) — ideal ≈ 1.0
- Disparate Impact Ratio (DIR) — IEEE fairness criterion: DIR ≥ 0.8 (EEOC 80% rule adapted)
- Counterfactual fairness test: same feature vector, only location differs

### Rank Stability Analysis (`eval/stability.py`)

Validates ranker robustness under minor input perturbations — a key IEEE publication requirement:

- Score perturbation stability: Gaussian noise (σ ∈ {0.01, 0.03, 0.05}) injected into numeric features; Kendall's Tau measured per query
- Query paraphrase stability: SBERT retrieval noise simulation via FAISS score perturbation

### Label Noise Analysis (`eval/label_noise_analysis.py`)

Justifies weak supervision by demonstrating graceful degradation:

- Sweeps label noise K ∈ {0%, 10%, 20%, 30%, 40%}
- Re-trains LightGBM on noisy labels; evaluates on the original clean test set
- Compares degradation against the rule-based baseline at each noise level
- Scientifically validates that the model consistently outperforms baselines even at 30% label noise

### Weak Supervision Justification

Training labels are derived from the composite heuristic score computed in the Java scraper:

```
composite_score = 0.35 × price_score + 0.25 × delivery_score + 0.40 × reliability_score
```

This is the standard bootstrapping approach for LTR datasets without explicit human relevance judgements (Joachims et al., 2002). Robustness to label noise is validated by `eval/label_noise_analysis.py`.

---

## Airflow DAG

`sourceup_dag.py` defines an Airflow DAG that runs `python pipeline/run_all.py` on a daily schedule, automating the ETL refresh without manual intervention.

```python
with DAG("sourceup", start_date=datetime(2024,1,1), schedule="@daily") as dag:
    run = BashOperator(
        task_id="pipeline",
        bash_command="python pipeline/run_all.py"
    )
```

Copy `sourceup_dag.py` into your Airflow `dags/` directory. Requires Apache Airflow 2.x. The rest of SourceUp runs fully without Airflow — it is an optional production convenience.

---

## Known Issues & Configuration Notes

| Issue | Resolution |
|-------|-----------|
| **`SOURCEUP_DIR` not set / hardcoded Windows paths** | Set `SOURCEUP_DIR` in `.env`, or find-and-replace `C:/Users/somas/PycharmProjects/SourceUp` throughout the project. |
| **Redis / session memory unavailable** | Install Redis (Linux/macOS) or Memurai (Windows). If unavailable, session context is lost between chat turns but `/recommend` still works fully. |
| **FAISS index missing on startup** | Run `python pipeline/run_all.py` first. The backend raises `RuntimeError` if `suppliers.faiss` does not exist. |
| **`GROQ_API_KEY` not set** | The `/chat` and `/quote` endpoints fail. `/recommend` continues to work using the rule-based parser and pre-trained ranker. |
| **Auth endpoints return 500** | Install `python-jose[cryptography]` and `passlib[bcrypt]`; set `SECRET_KEY` in `.env`. |
| **Billing endpoints return 500** | Install `razorpay`; set `RAZORPAY_KEY_ID` and `RAZORPAY_KEY_SECRET` in `.env`. |
| **SHAP analysis fails** | `pip install shap` |
| **Scraper captchas / rate limiting** | Increase sleep delays in `App.java` or add a residential proxy. The scraper already randomises wait times and uses stealth Chrome flags. |
| **`data/test_output.csv` missing** | `cp data/outputs/<any_result>.csv data/test_output.csv` |
| **LightGBM not installed** | The ranker degrades gracefully to XGBoost, then to the rule-based scorer. Install with `pip install lightgbm`. |

---

## Real-World Use Cases

- SME procurement teams sourcing verified suppliers
- B2B marketplaces improving recommendation quality
- AI assistants for industrial sourcing
- Research in explainable ranking systems (XAI + IR)

## Research Roadmap

Proposed extensions for future work:

- **Multilingual sourcing** — extend the NLU to support Tamil, Hindi, and other Indian regional languages for broader SME reach.
- **Reinforcement learning for negotiation** — an agent that optimises multi-round price negotiation within supplier constraints.
- **Real interaction logs** — replace weak supervision labels with click-through data from deployed production usage.
- **Constraint relaxation engine** — suggest the minimum constraint adjustment that would expand the feasible supplier set when no results satisfy all hard constraints.
- **Interactive feedback loop** — allow users to refine constraints dynamically and observe real-time rank changes within the UI.
- **Production billing tier enforcement** — gate `/recommend` top-k results and decision traces behind Pro/Enterprise plan checks.

---

## Academic Context

### Problem Framing

SourceUp is positioned as a constraint-aware learning-to-rank framework, not merely a sourcing application. The core contribution is the formalisation of SME procurement as a constrained optimisation problem:

```
Score(q, d) = f_θ(q, d) − γ · ConstraintViolation(d, C)
```

This enables ablation, sensitivity analysis, and controlled comparison against standard IR baselines — the requirements for IEEE-level publication.


*SourceUp — Built for SME procurement research. All components are modular and independently testable.*