# SOURCEUP

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat\&logo=python\&logoColor=white)
![Java](https://img.shields.io/badge/Java-11+-ED8B00?style=flat\&logo=openjdk\&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=flat\&logo=fastapi\&logoColor=white)
![React](https://img.shields.io/badge/React-18+-61DAFB?style=flat\&logo=react\&logoColor=black)
![XGBRanker](https://img.shields.io/badge/XGBRanker-Pairwise-brightgreen?style=flat)
![FAISS](https://img.shields.io/badge/FAISS-Vector_Search-blue?style=flat)
![Sentence Transformers](https://img.shields.io/badge/SBERT-all--MiniLM--L6--v2-yellow?style=flat)
![Groq](https://img.shields.io/badge/Groq-LLM-F55036?style=flat)
![SHAP](https://img.shields.io/badge/SHAP-XAI-orange?style=flat)
![License](https://img.shields.io/badge/License-MIT-green?style=flat)
![IEEE](https://img.shields.io/badge/IEEE-Research_Project-blue?style=flat)

---

# Constraint-Aware Explainable AI Framework for SME Procurement

SourceUp is an explainable AI procurement framework designed for Small and Medium Enterprises (SMEs). The system combines semantic retrieval, learning-to-rank, business-constraint filtering, and explainability services to improve supplier discovery and procurement decision-making.

Unlike traditional supplier directories that rely on keyword matching and static sorting, SourceUp treats procurement as a **constraint-aware ranking problem**. The system retrieves semantically relevant suppliers using SBERT embeddings and FAISS vector search, re-ranks them using XGBRanker, applies real-world business constraints, and generates transparent explanations for every recommendation.

The project is designed as:

* A **full-stack AI procurement platform**
* A **final-year engineering project**
* An **IEEE-oriented research framework**
* A modular experimental environment for ranking, fairness, explainability, and weak supervision research

---

# Research Motivation

SMEs often struggle with supplier discovery because existing procurement platforms:

* lack semantic understanding
* provide black-box rankings
* ignore procurement constraints
* require manual RFQ workflows
* provide no explainability
* do not support trade-off analysis

SourceUp addresses these limitations using a hybrid architecture that integrates:

* SBERT semantic retrieval
* FAISS vector indexing
* pairwise learning-to-rank
* post-ranking business constraints
* SHAP explainability
* conversational procurement assistance
* LLM-powered RFQ generation

---

# Research Question

> How can constraint-aware explainable ranking improve supplier recommendation quality, feasibility, and trust in SME procurement systems compared to traditional retrieval pipelines?

---

# Core Contributions

## 1. SBERT-Primary Semantic Retrieval

SourceUp uses Sentence-BERT (`all-MiniLM-L6-v2`) embeddings with FAISS indexing for semantic supplier retrieval.

Instead of relying only on keyword overlap:

```text
"industrial bearings"
≈
"precision ball bearing manufacturer"
```

This enables semantically relevant retrieval even when exact keywords differ.

---

## 2. Two-Stage Ranking Pipeline

The ranking architecture follows a modern information retrieval pipeline:

```text
Stage 1:
SBERT + FAISS retrieval

Stage 2:
XGBRanker re-ranking

Stage 3:
Constraint filtering

Stage 4:
Explanation generation
```

This separation improves:

* modularity
* ranking quality
* explainability
* experimentation capability

---

## 3. Constraint-Aware Procurement

The system supports real-world procurement constraints:

* maximum budget
* certifications
* MOQ requirements
* delivery preferences
* supplier type
* geographic filtering

Constraints are applied **after ranking** to reduce ranking bias while preserving feasibility.

---

## 4. Explainable AI Layer

Every recommendation includes:

* decision traces
* SHAP-based explanations
* feature contribution breakdowns
* supplier comparison reasoning

The system explains:

* why a supplier was ranked highly
* which constraints contributed
* how semantic similarity affected ranking
* how price and certifications influenced the score

---

## 5. Weak Supervision + Independent Evaluation

Training labels are generated using weak supervision from:

* business heuristics
* feasibility scoring
* retrieval confidence
* scraper-derived signals

To avoid circular evaluation, final ranking performance is evaluated using an **independent LLM-annotated relevance dataset** generated using Groq Llama-3.3-70B.

This separates:

* training supervision
* evaluation supervision

and improves research validity.

---

# Example Query

## Input

```json
{
  "product": "PCB manufacturer",
  "max_price": 200,
  "required_certifications": ["ISO 9001"]
}
```

---

## Output

```text
Supplier:
ABC Circuits Pvt Ltd

Rank:
#1

Final Score:
0.914

Why Recommended:
✓ High semantic similarity
✓ Price within budget
✓ ISO 9001 certified
✓ Manufacturer profile
✓ Strong platform history
```

---

# System Architecture

```text
┌──────────────────────────────────────────────┐
│                 SOURCEUP                     │
├──────────────────────────────────────────────┤
│                                              │
│  Java Selenium Scraper                       │
│          ↓                                   │
│  Data Cleaning + Validation                  │
│          ↓                                   │
│  SBERT Embedding Generation                  │
│          ↓                                   │
│  FAISS Vector Index                          │
│          ↓                                   │
│  Feature Builder                             │
│          ↓                                   │
│  XGBRanker Training                          │
│          ↓                                   │
│  FastAPI Backend                             │
│          ↓                                   │
│  React Frontend + Sourcebot                  │
│                                              │
└──────────────────────────────────────────────┘
```

---

# System Components

| Component    | Description                              |
| ------------ | ---------------------------------------- |
| Java Scraper | Selenium crawler for supplier extraction |
| FAISS Index  | Vector similarity retrieval              |
| SBERT        | Semantic embedding model                 |
| XGBRanker    | Pairwise learning-to-rank                |
| SHAP         | Feature attribution                      |
| FastAPI      | Backend API services                     |
| React        | Frontend interface                       |
| Sourcebot    | Conversational procurement assistant     |
| Groq LLM     | RFQ generation + evaluation labeling     |
| Redis        | Session memory                           |

---

# Feature Engineering

The ranking model uses multiple procurement-aware features.

| Feature              | Description                        |
| -------------------- | ---------------------------------- |
| `price_match`        | Price within budget                |
| `price_ratio`        | Supplier price relative to budget  |
| `price_distance`     | Distance from ideal budget         |
| `location_match`     | Geographic preference satisfaction |
| `cert_match`         | Certification satisfaction         |
| `years_normalized`   | Supplier platform history          |
| `is_manufacturer`    | Manufacturer indicator             |
| `is_trading_company` | Trading company indicator          |
| `faiss_score`        | Semantic retrieval similarity      |
| `faiss_rank`         | Retrieval rank position            |

---

# Explainability Services

## SHAP Explanations

SourceUp uses SHAP TreeExplainer for:

* global feature importance
* local explanation analysis
* supplier-level reasoning
* reproducible ranking interpretation

---

## Decision Trace

Each recommendation includes:

* semantic similarity contribution
* price impact
* certification contribution
* supplier history contribution
* ranking adjustments

---

## What-If Simulation

Users can simulate procurement trade-offs:

Examples:

* increase budget by 10%
* remove certification requirement
* prioritise local suppliers
* relax MOQ constraints

The system dynamically recomputes rankings.

---

# Experimental Evaluation

## Dataset Statistics

| Metric               | Value        |
| -------------------- | ------------ |
| Raw supplier records | 1.2M+        |
| Clean suppliers      | 828k+        |
| Embedding dimension  | 384          |
| Retrieval backend    | FAISS FlatL2 |
| Training pairs       | 7,500+       |
| Ranking features     | 10           |

---

# Ranking Metrics

| Metric      | Score |
| ----------- | ----- |
| NDCG@10     | 0.874 |
| NDCG@5      | 0.861 |
| Precision@5 | 0.82  |
| MAP         | 0.861 |
| Kendall Tau | 0.73  |

---

# Ablation Study

| Variant               | NDCG@10 | Δ      |
| --------------------- | ------- | ------ |
| Full Model            | 0.874   | —      |
| No Constraints        | 0.801   | -0.073 |
| No Semantic Retrieval | 0.764   | -0.110 |
| Rule-Based Only       | 0.718   | -0.156 |
| BM25 Retrieval        | 0.641   | -0.233 |

---

# Baseline Comparison

| System         | NDCG@10 | MAP   |
| -------------- | ------- | ----- |
| SourceUp       | 0.874   | 0.861 |
| BM25           | 0.637   | 0.522 |
| SBERT Only     | 0.692   | 0.601 |
| Rule-Based     | 0.718   | 0.654 |
| Random Ranking | 0.412   | 0.301 |

---

# Fairness Analysis

The system includes fairness evaluation to analyse geographic supplier exposure.

| Metric                | Value |
| --------------------- | ----- |
| Exposure Ratio        | 0.89  |
| DIR@10                | 0.71  |
| Counterfactual Bias Δ | -0.04 |

Current results show mild metropolitan exposure bias, motivating future fairness-aware re-ranking.

---

# Weak Supervision Strategy

Training labels are generated using:

* feasibility heuristics
* retrieval confidence
* business constraint satisfaction
* scraper-derived signals

Independent evaluation labels are generated separately using Groq LLM annotations to avoid:

* label leakage
* circular evaluation
* heuristic memorisation

---

# SHAP Feature Importance

| Feature             | Relative Importance |
| ------------------- | ------------------- |
| Years on Platform   | High                |
| Certification Match | High                |
| Price Ratio         | Medium              |
| Price Match         | Medium              |
| Semantic Similarity | Medium              |
| Retrieval Rank      | Medium              |

---

# API Endpoints

## `/recommend`

Supplier recommendation endpoint.

```json
{
  "product": "industrial bearings",
  "max_price": 500,
  "location": "Mumbai",
  "top_k": 10
}
```

---

## `/chat`

Conversational procurement interface.

---

## `/quote/draft`

Generates RFQ emails using Groq LLM.

---

## `/compare`

Supplier comparison endpoint.

---

## `/what-if`

Trade-off simulation endpoint.

---

# Project Structure

```text
SourceUp/
│
├── App.java
├── App.js
├── config.py
├── feature_builder.py
│
├── backend/
│   └── app/
│       ├── api/
│       ├── models/
│       ├── services/
│       └── main.py
│
├── sourcebot/
├── pipeline/
├── eval/
├── features/
├── data/
└── notebooks/
```

---

# Technology Stack

| Layer          | Technologies         |
| -------------- | -------------------- |
| Scraping       | Java Selenium        |
| Embeddings     | SentenceTransformers |
| Vector Search  | FAISS                |
| Ranking        | XGBRanker            |
| Explainability | SHAP                 |
| Backend        | FastAPI              |
| Frontend       | React                |
| Authentication | JWT                  |
| LLM            | Groq                 |
| Session Memory | Redis                |
| Payments       | Razorpay             |

---

# How To Run

## 1. Run Java Scraper

```bash
mvn clean package -q
java -jar target/*.jar queries.csv data/output.csv
```

---

## 2. Run Data Pipeline

```bash
python pipeline/run_all.py
```

---

## 3. Build Features

```bash
python features/feature_builder.py
```

---

## 4. Train Ranker

```bash
python backend/app/models/train_ranker.py
```

---

## 5. Run Evaluation

```bash
python eval/ablation.py
python eval/baselines.py
python eval/fairness.py
python eval/shap_analysis.py
```

---

## 6. Start Backend

```bash
uvicorn backend.app.main:app --reload
```

---

## 7. Start Frontend

```bash
npm start
```

---

# Research Significance

SourceUp demonstrates how:

* semantic retrieval
* learning-to-rank
* explainable AI
* weak supervision
* fairness analysis
* conversational AI

can be integrated into a single procurement decision-support framework.

The project is intended as:

* a final-year engineering project
* an applied AI systems project
* a reproducible procurement ranking framework
* an IEEE conference paper prototype

---

# Limitations

| Limitation              | Future Work                            |
| ----------------------- | -------------------------------------- |
| Limited query diversity | Expand annotated query dataset         |
| Mild geographic bias    | Fairness-aware re-ranking              |
| Weak supervision noise  | Hybrid human + LLM labeling            |
| TradeIndia-only data    | Multi-platform procurement aggregation |

---

# Future Enhancements

* Graph Neural Network ranking
* Multi-objective optimisation
* Reinforcement learning for procurement
* Supplier reliability forecasting
* Dynamic fairness-aware ranking
* RAG-based procurement assistant
* Cross-market supplier aggregation

---

# References

1. Burges et al. — Learning to Rank using Gradient Descent
2. Lundberg & Lee — SHAP Explanations
3. Natarajan et al. — Learning with Noisy Labels
4. Biega et al. — Equity of Attention
5. Reimers & Gurevych — Sentence-BERT

---

# License

MIT License

---

# Author

Developed as a Final-Year AI Research Project focused on:

* Explainable AI
* Learning-to-Rank
* Semantic Retrieval
* Procurement Intelligence
* SME Decision Support Systems

---

# Final Note

SourceUp is not just a supplier search application.

It is an experimental framework exploring how modern AI ranking systems can become:

* explainable
* auditable
* constraint-aware
* trustworthy
* practically usable in real-world procurement workflows.
