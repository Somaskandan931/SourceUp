# SourceUp — Execution Order Guide

Run these steps **in order**. Steps 1–4 are one-time setup. Steps 5–6 are the running application.

---

## Prerequisites — Install Everything First

### Python packages
```bash
pip install fastapi uvicorn python-dotenv groq \
            sentence-transformers faiss-cpu \
            lightgbm xgboost pandas numpy scikit-learn \
            scipy redis matplotlib seaborn shap \
            python-jose[cryptography] passlib[bcrypt] \
            langchain-groq langchain-core
```

### Node.js (frontend)
```bash
npm install   # run from the directory containing App.js
```

### Java + Maven (scraper only)
JDK 11+ and Maven 3.6+ must be on your PATH.

---

## Environment Setup

Create a `.env` file in the **project root** (same level as `backend/`, `pipeline/`, etc.):

```env
GROQ_API_KEY=your_groq_api_key_here

# Auth (required for /auth endpoints)
SECRET_KEY=your_jwt_secret_here        # generate: openssl rand -hex 32

# UPI Billing (required for upgrade payments)
UPI_ID=yourname@upi                    # your UPI VPA (e.g. yourname@okhdfcbank)

# Optional — Redis session memory (chat loses context without it)
REDIS_HOST=localhost
REDIS_PORT=6379

# Optional — overrides project root path detection
SOURCEUP_DIR=D:\PycharmProjects\SourceUp
```

> **If you do NOT set `SOURCEUP_DIR`**, the project root defaults to the directory
> containing `config.py`, making the project self-locating on any machine.

---

## Step 1 — Java Scraper (data collection)

Build the JAR, then run a scrape. Output CSVs go to `data/outputs/`.

```bash
# Build
mvn clean package -q

# Run
java -jar target/global-sources-scraper-1.0-SNAPSHOT.jar \
     queries.csv data/outputs/result.csv 50000 7 1 5
#    <input>    <output>               <budget> <delivery_days> <start_page> <end_page>
```

`queries.csv` must have a header row; search terms go in the first column.

After the first run, copy any one output CSV as the canonical schema reference:
```bash
cp data/outputs/result.csv data/test_output.csv
```

---

## Step 2 — Data Pipeline (merge → clean → FAISS index)

```bash
python pipeline/run_all.py
```

This runs four sub-steps internally:

1. `validate_merge` — merges all CSVs in `data/outputs/` into `data/merged/suppliers_all.csv`
2. `clean_normalize` — deduplicates and normalises → `data/clean/suppliers_clean.csv`
3. `incremental_faiss` — builds SBERT embeddings → `data/embeddings/suppliers.faiss` + `suppliers_meta.csv`
4. `feature_builder` — builds LTR training dataset → `data/training/ranking_data.csv`

To skip the (slow) feature-builder step:
```bash
python pipeline/run_all.py --skip-features
```

---

## Step 3 — Train the Ranker

Both commands must complete before the backend will use ML ranking.

```bash
# Primary: LightGBM LambdaRank
python backend/app/models/train_lambdarank.py

# Optional: tune constraint-penalty weight (default γ = 0.3)
python backend/app/models/train_lambdarank.py --gamma 0.3

# Legacy: XGBoost backup
python backend/app/models/train_ranker.py
```

Outputs: `ranker_lightgbm.pkl` and `ranker_xgboost.pkl` in `backend/app/models/embeddings/`.

> **Skip this step** if the scraped data has not changed — pre-trained models are already included.

---

## Step 4 — Evaluation Scripts

Run these before starting the backend if you need research metrics.
All results and plots are saved to `data/eval/`.

```bash
python eval/ablation.py            # 5-variant ablation study
python eval/baselines.py           # BM25 / SBERT / rule-based comparison
python eval/sensitivity.py         # γ sweep — NDCG vs. CVR trade-off
python eval/shap_analysis.py       # SHAP feature attribution (IEEE-grade XAI)
python eval/fairness.py            # Geographic exposure-disparity analysis
python eval/stability.py           # Kendall's Tau rank stability under perturbations
python eval/label_noise_analysis.py # Weak-supervision robustness (10–40% noise)
python evaluation/metrics.py       # Standalone metric unit tests
```

---

## Step 5 — Start the Backend

```bash
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```

Startup banner will confirm:
- Project root path
- Groq API key status
- UPI billing status
- Any missing-path warnings

Verify:
- Health check: `http://localhost:8000/health`
- Swagger docs: `http://localhost:8000/docs`

---

## Step 6 — Start the React Frontend

```bash
npm start   # from the directory containing App.js
```

Opens `http://localhost:3000`.

---

## Quick-Reference: All Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/recommend` | Main procurement ranking (constraint-aware + explainable) |
| POST | `/chat` | Conversational interface via Groq LLM + Sourcebot NLU |
| POST | `/what-if` | Priority/constraint trade-off simulation |
| POST | `/compare` | Side-by-side supplier decision trace |
| POST | `/quote/draft` | Generate RFQ email draft via Groq |
| POST | `/quote/refine` | Refine an existing RFQ draft |
| POST | `/auth/register` | User registration (JWT) |
| POST | `/auth/login` | Login → JWT token |
| POST | `/auth/billing/order` | Create UPI payment order |
| POST | `/auth/billing/verify` | Verify UPI payment (UTR submission) |
| GET  | `/auth/billing/plans` | List Free / Pro / Enterprise plans |
| GET  | `/health` | System health + FAISS index size |
| GET  | `/stats` | Supplier DB stats by location and category |
| GET  | `/recommend/test` | Sanity check — returns first supplier in index |
| GET  | `/chat/test` | Sanity check — verifies session layer |

---

## UPI Payment Flow

1. User calls `POST /auth/billing/order` with their plan choice.
2. Backend returns a `upi_link` (UPI deep-link) and `order_id`.
3. Frontend opens the UPI link / renders a QR code.
4. User completes payment in their UPI app (GPay, PhonePe, Paytm, etc.).
5. User copies the **UTR / transaction ID** from their UPI app.
6. Frontend calls `POST /auth/billing/verify` with `order_id` + `upi_transaction_id`.
7. Backend upgrades the user's plan and returns a new JWT.

> **Production note:** For automatic verification, integrate with your bank's
> payment API or a payment aggregator (Cashfree, Razorpay UPI, Paytm PG, etc.)
> to auto-match UTRs via webhook.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `RuntimeError: suppliers.faiss not found` | Run Step 2 (`pipeline/run_all.py`) first |
| `/chat` returns 500 | Check `GROQ_API_KEY` in `.env` |
| Auth endpoints return 500 | Install `python-jose[cryptography] passlib[bcrypt]`; set `SECRET_KEY` in `.env` |
| Billing order returns 503 | Set `UPI_ID` in `.env` |
| Chat loses context between turns | Install Redis (Linux/macOS) or Memurai (Windows) |
| Scraper blocked / captcha | Increase sleep delays in `App.java`; consider a residential proxy |
| `data/test_output.csv` missing | `cp data/outputs/<any_result>.csv data/test_output.csv` |
| SHAP plots fail | `pip install shap` |
| `python-jose` not found | `pip install python-jose[cryptography]` |
| `langchain-groq` not found | `pip install langchain-groq langchain-core` |
