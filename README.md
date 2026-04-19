# SourceUp
### Constraint-Aware Explainable AI for Supplier Selection

> AI-powered decision support for SMEs  replacing rigid filtering with constraint-aware ranking and transparent, explainable decisions.

---

## Why SourceUp?

Traditional sourcing platforms fall short in predictable ways:

| Problem | SourceUp Solution |
|---|---|
| Keyword filtering only | Semantic, intent-aware retrieval |
| No trade-off visibility | Multi-factor constraint optimization |
| Black-box decisions | Explainable ranking with feature contributions |

Built to solve a real problem: helping SMEs choose cost-effective, reliable suppliers under real-world constraints.

---

## Core Innovation

SourceUp moves beyond *filter-then-rank* by directly optimizing:

$$\text{Score}(q, d) = f_\theta(q, d) - \gamma \cdot \text{ConstraintViolation}(d, C)$$

This enables:
- **Trade-off reasoning** across price, delivery time, and reliability
- **Soft constraint handling**  penalization instead of rigid exclusion
- **Transparent ranking**  every decision is explainable

---

## System Architecture

```
Scraper  Feature Engineering  Semantic Search (FAISS)
         Constraint-Aware Ranking  Explainability Layer  API / SourceBot
```

---

## Key Features

### Semantic Search
SBERT embeddings + FAISS index for high-quality supplier retrieval that understands query intent, not just keywords.

### Constraint-Aware Ranking
Budget, delivery windows, and feasibility constraints are integrated directly into the scoring function  not applied as post-hoc filters.

### Explainable AI Layer
- Feature contribution breakdowns per supplier
- Trade-off explanations (why Supplier A ranks above Supplier B)
- Human-readable decision rationale

### SourceBot (Conversational Interface)
Natural language queries are parsed into structured constraints via LangChain + LLM, enabling non-technical users to interact without knowing query syntax.

### Automated Data Pipeline
End-to-end scraping, cleaning, and feature engineering  from raw web data to model-ready supplier records.

---

## Example

**Query:**
```
"Need disposable food containers under ₹5 with fast delivery"
```

**Output:** Ranked supplier list with explanations

| Supplier | Score | Explanation |
|---|---|---|
| Supplier A | 0.91 | Low unit price + high reliability rating |
| Supplier B | 0.74 | Penalized for estimated delivery delay |
| Supplier C | 0.61 | Budget exceeded soft constraint threshold |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend API | FastAPI |
| Semantic Search | Sentence Transformers (SBERT) + FAISS |
| Ranking Model | LightGBM / XGBoost *(planned)* |
| Scraping | Java (Jsoup) + Python |
| LLM / Chatbot | LangChain + OpenAI API |
| Frontend | Streamlit |
| ML Utilities | Scikit-learn |

---

## Project Status

- [x] Data scraping pipeline
- [x] Semantic retrieval system (SBERT + FAISS)
- [ ] Feature engineering *(in progress)*
- [ ] Learning-to-Rank model *(in progress)*
- [ ] Evaluation experiments *(planned)*

---

## Getting Started

```bash
git clone https://github.com/your-username/sourceup.git
cd sourceup
pip install -r requirements.txt
uvicorn main:app --reload
```

Run the Java scraper:

```bash
java -jar scraper.jar input.csv output.csv
```

---

## Roadmap

- [ ] Learning-to-Rank with LambdaRank
- [ ] Fairness-aware supplier recommendations
- [ ] Reinforcement learning for dynamic scoring
- [ ] Multilingual query support
- [ ] Cloud deployment (Docker + Kubernetes)

---

## Skills Demonstrated

- End-to-end ML system design
- Semantic search with vector databases
- Constraint-aware ranking formulation
- Explainable AI for decision support systems
- Data engineering: scraping  features  pipeline
- API development + conversational AI integration

---

## About

SourceUp is not just a recommender system. It is a step toward building **transparent, constraint-aware AI systems** for real-world procurement decisions  where trust and explainability matter as much as accuracy.

---

*Built for SMEs. Designed for clarity. Powered by constraint-aware AI.*