# SourceUp

> Constraint-aware, explainable, and fair semantic supplier discovery for SME
> procurement — record a query, retrieve semantically matched candidates from
> 871k supplier records, enforce hard procurement constraints, rank with a
> learned LambdaRank model, and explain every decision with SHAP attribution.
> No opaque black boxes.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=flat&logo=fastapi&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-LambdaRank-brightgreen?style=flat)
![FAISS](https://img.shields.io/badge/FAISS-Vector_Search-blue?style=flat)
![SBERT](https://img.shields.io/badge/SBERT-all--mpnet--base--v2-yellow?style=flat)
![SHAP](https://img.shields.io/badge/SHAP-Explainable_AI-orange?style=flat)
![MongoDB](https://img.shields.io/badge/MongoDB-Motor_Async-47A248?style=flat&logo=mongodb&logoColor=white)
![Groq](https://img.shields.io/badge/Groq-LLM_Powered-F55036?style=flat)
![IEEE](https://img.shields.io/badge/IEEE-Conference_Paper-blue?style=flat)
![License](https://img.shields.io/badge/License-MIT-green?style=flat)

---

## Project Context

SourceUp was built to answer a specific question:

> Can an SME buyer describe a procurement need in natural language, and receive
> a ranked, explainable, constraint-satisfied shortlist of real suppliers —
> without relying on a keyword search or a manual spreadsheet?

The answer required building an entire pipeline: semantic retrieval, constraint
enforcement, learning-to-rank, SHAP attribution, geographic fairness evaluation,
and a conversational interface layered on top. Each of those pieces exists because
none of the others was sufficient alone.

This repository is therefore written for two audiences:

| Audience | What they should get from this repo |
|---|---|
| Evaluators / reviewers | A complete engineering and research story: problem framing, design trade-offs, honest evaluation methodology, and explicit limitations |
| Developers / contributors | A practical path from a procurement query to a ranked, explained supplier list, with enough background to understand every moving part |
| Researchers | Transparent learning-to-rank internals, weak-supervision labelling methodology, a label-independence audit, and reproducible eval scripts |
| Builders / product teams | A production-minded FastAPI backend, React frontend, MongoDB persistence, RFQ generation, and billing scaffolding to build on |

SourceUp is intentionally not a black-box recommender. The ranking decision for
every supplier is fully traceable: which features drove the score up, which
constraint was violated, and what would change if the buyer relaxed a filter.

---

## Why This Project Exists

SME procurement is still largely manual. Buyers search fragmented B2B directories,
filter results by keyword, shortlist candidates in spreadsheets, and select based
on incomplete information — often without knowing why one supplier is better than
another for their specific constraints.

The problem has four distinct parts that individually have solutions but rarely
appear together in one system:

```
Scale             →  871,467 supplier records cannot be exhaustively scored per query
Semantic mismatch →  "ISO textile exporter" ≠ "garment manufacturer with quality compliance"
                      in lexical search, but they mean the same thing to the buyer
Constraint logic  →  Budget, MOQ, certification, and location are hard gates, not preferences
Exposure bias     →  Popularity-correlated ranking over-surfaces established metro suppliers
                      and buries newly-onboarded or Tier-2 city suppliers
```

SourceUp connects those four problems into one pipeline:

```
Query → Semantic Retrieval → Constraint Enforcement → Learned Ranking → SHAP Attribution → Ranked Results
```

---

## What Users Should Know First

SourceUp is not a search engine that retrieves matching records. It is a
**ranking system** that takes a set of semantically retrieved candidates and
decides the order they should be presented to a procurement officer, while
respecting hard operational constraints and explaining every position in the
final list.

| Concept | Plain-English explanation |
|---|---|
| Semantic retrieval | SBERT converts both the query and every supplier description into a dense vector. FAISS finds the closest supplier vectors to the query vector. This is why "packaging exporter" matches "container manufacturer" even with no shared keywords |
| Cross-encoder reranking | A second, slower model reads each (query, supplier) pair together and rescores the top-100 candidates more accurately than the bi-encoder can |
| Hard constraint enforcement | Budget, MOQ, certification, and location are checked as pass/fail gates. Suppliers that fail are tagged with the reason and penalised in the ranking rather than silently dropped — so the user can see exactly what failed |
| LambdaRank | A LightGBM model trained with a list-wise ranking objective. It learns which combinations of price compliance, semantic similarity, certification, and location produce the best supplier ordering for a given query group |
| SHAP attribution | Every ranking decision is decomposed into per-feature contributions. The buyer can see that cert\_match drove the top result up and location\_match drove it down, with numeric values attached |
| Weak-supervision labels | Because no ground-truth click data exists, relevance labels are generated by a heuristic function. This creates a circularity risk that is explicitly addressed and mitigated — see the Labelling section |
| Geographic fairness | The system tracks whether Metro suppliers receive systematically more exposure than Tier-2 or Tier-3 suppliers, and reports exposure ratios alongside ranking metrics |
| What-if simulation | The buyer can ask "what if I increase my budget by 20%?" and see a simulated re-ranking without running the full pipeline again |

The quality of recommendations depends on corpus coverage. SourceUp indexes
19,750 of 871,467 cleaned supplier profiles — enough for a functioning research
benchmark, not yet a production catalogue. This limitation is explicit throughout
the codebase and the paper.

---

## Design Preferences

SourceUp is built with a few strong preferences that guide every implementation
decision.

### 1. Repository as Source of Truth

Core logic belongs in the repository. The pipeline scripts, evaluation modules,
labelling logic, constraint engine, and ranking model are all version-controlled.
A notebook or a web UI may launch a pipeline run, but it must not redefine the
algorithm. This is what makes the system reproducible.

### 2. Honest Evaluation Over Flattering Metrics

The most important design decision in this project is not a model architecture
choice — it is the decision to report two rule-based baselines instead of one.
The legacy baseline reuses the same signals as the weak-supervision labelling
function. Its NDCG@10 of 0.8825 looks competitive with the full model because
it is, structurally, grading its own homework. The label-independent baseline
uses only signals the labelling function never sees, and scores 0.5963.

Only reporting the legacy baseline would tell a misleading story. Reporting both,
and treating the label-independent comparison as primary, is the right scientific
choice even though it reduces the apparent gap.

### 3. Constraint Violation Is Information, Not Failure

When a supplier fails a hard constraint, it is not removed from the results. It
is tagged with the specific constraint it violated, penalised in the ranking
score, and surfaced at the bottom of the list with the violation reason visible
in the SHAP trace. A buyer searching for ISO-certified Chennai suppliers who gets
zero fully-compliant results should still see the closest available options —
and understand exactly why each one fell short.

### 4. Explainability as a First-Class Output

SHAP attribution is not a diagnostic tool added after training. It is part of
the production response. Every `/recommend` API call returns, alongside the
ranked supplier list, a per-feature contribution breakdown for each result. The
decision trace module translates those SHAP values into natural-language summaries
("Price £0.04 is within budget of £50,000", "ISO certification match confirmed").

### 5. Fairness as a Measurable Metric, Not a Post-Hoc Claim

Geographic supplier exposure is measured at evaluation time with exposure ratios,
disparate-impact ratios, Mann-Whitney tests, Kolmogorov-Smirnov tests, and a
counterfactual test that flips supplier locations while holding all other features
fixed. The fairness evaluation is not a section added to look responsible — it
directly tests a real risk: that a ranking system trained on scraped B2B data
might systematically favour large metro suppliers.

---

## System Architecture

The diagram below is the architecture figure used in the IEEE paper. It traces a
query end-to-end through candidate generation, constraint filtering, learning-to-rank,
output, and the evaluation/analysis framework that surrounds the live pipeline.

![SourceUp end-to-end architecture](IEEE_ARCHITECTURE.png)

---

## How the Ranking Pipeline Works

**SourceUp is fundamentally different from a search engine.** A search engine
retrieves documents that match a query. SourceUp retrieves candidate suppliers
and then re-orders them according to a learned model that weighs semantic fit,
price compliance, certification, location, and supplier experience together.

The distinction matters because retrieval and ranking have different failure modes.
Retrieval fails when vocabulary gaps prevent the right supplier from appearing
in the candidate pool. Ranking fails when the ordering of an already-correct
candidate pool is wrong. SourceUp addresses both.

```
Query: "ISO certified food packaging, budget £50k, Chennai"
   │
   ▼
SBERT Encoding         Query → dense vector (768-dim, all-mpnet-base-v2)
   │
   ▼
FAISS Retrieval        Top-100 semantically similar suppliers from 19,750-vector index
   │                   Inner-product search over normalised embeddings
   ▼
Cross-Encoder Rerank   ms-marco-MiniLM-L-6-v2 reads each (query, supplier) pair jointly
   │                   More accurate than bi-encoder; scores all 100 candidates
   ▼
Constraint Engine      Check budget, MOQ, certification, location for each candidate
   │                   Tag violations; compute penalty score V(s,C)
   ▼
Feature Engineering    9-dim feature vector per (query, supplier) pair
   │                   price_match, price_ratio, price_distance, location_match,
   │                   cert_match, years_normalized, is_manufacturer,
   │                   is_trading_company, faiss_score
   ▼
LightGBM LambdaRank   Query-grouped ranking; Score = f_θ(q,s) - γ·V(s,C)
   │                   Trained on 18,525 pairs across 247 queries
   ▼
MMR Diversification    Maximal Marginal Relevance selects top-10 diverse results
   │
   ▼
SHAP Attribution       TreeExplainer decomposes each score into feature contributions
   │                   Natural-language decision trace generated per supplier
   ▼
Response               Ranked suppliers + SHAP breakdown + constraint violation tags
                        Warm-start latency: ~578 ms
```

---

## The Label-Independence Problem

This is the most important methodological decision in the project, and it is
worth explaining in full before looking at any evaluation numbers.

Weak-supervision labelling assigns relevance scores to supplier-query pairs using
a heuristic function of `price_match`, `location_match`, `cert_match`,
`years_normalized`, and `faiss_score`. A naive rule-based baseline that also
scores suppliers using those same signals can, in principle, match or beat the
learned model — not because it is a better ranker, but because it is
*scoring its own exam*. Its high NDCG@10 is a structural artefact of benchmark
construction, not evidence of genuine ranking skill.

```
Weak-label function uses:  price_match, location_match, cert_match, faiss_score
Legacy rule baseline uses: price_match, location_match, cert_match, faiss_score
                                                                      ↑
                                         These overlap → inflated baseline NDCG
```

To avoid this circularity, SourceUp additionally constructs a **label-independent
rule-based baseline** using only features the labelling function never sees:
`category_overlap_score`, `price_distance`, `certification_count`, and
`is_manufacturer`. This baseline scores 0.5963 NDCG@10 — comparable to BM25 and
SBERT-cosine, as you would expect from a scorer operating without privileged
access to the evaluation signal.

The +0.2747 improvement of the full LambdaRank model over the label-independent
baseline (p < 0.001) is the honest test. Everything else is context.

---

## Results

Evaluated on 23,175 supplier-query pairs across 309 queries, 62 held-out test
queries (4,650 test pairs).

### Baseline Comparison

| System | NDCG@10 | P@5 | MAP | MRR |
|---|---|---|---|---|
| **SourceUp (LambdaRank)** | **0.8710** | 0.7443 | 0.7183 | 0.8892 |
| Legacy Rule-Based Ranker | 0.8825 | 0.7574 | 0.7462 | 0.9119 |
| Independent Rule-Based Ranker | 0.5963 | 0.3770 | 0.3977 | 0.6186 |
| SBERT Cosine | 0.5051 | 0.2984 | 0.3132 | 0.4023 |
| Random | 0.5045 | 0.3016 | 0.3286 | 0.4456 |
| BM25 | 0.4927 | 0.2557 | 0.3070 | 0.3797 |

The legacy baseline ties the full model because it shares signals with the
labelling function. Once that overlap is removed, it drops to 0.5963. That
drop is the real signal. The learned model's +0.2747 improvement over the
label-independent baseline at p < 0.001 is the primary evidence that
learning-to-rank adds value over heuristics.

![Baseline comparison](data/eval/plots/baseline_comparison_bar.png)
![Per-query NDCG@10 across baselines](data/eval/plots/baseline_ndcg_per_query.png)

### Statistical Significance (Wilcoxon Signed-Rank, n=62 queries)

| Comparison | Mean ΔNDCG@10 | p-value | Significant |
|---|---|---|---|
| vs. BM25 | +0.3783 | < 0.001 | ✓ |
| vs. SBERT Cosine | +0.3659 | < 0.001 | ✓ |
| vs. Random | +0.3665 | < 0.001 | ✓ |
| vs. Independent Rule-Based | +0.2747 | < 0.001 | ✓ |
| vs. Legacy Rule-Based | −0.0115 | > 0.05 | ✗ (expected) |
| vs. XGBoost LambdaMART | −0.0061 | > 0.05 | ✗ |
| vs. LightGBM Regression | −0.0078 | > 0.05 | ✗ |

![Statistical significance table](data/eval/plots/baseline_significance_table.png)

### Ablation Study

| Variant | NDCG@10 | CVR | ΔNDCG |
|---|---|---|---|
| **Full Model** | **0.8710** | 0.4581 | — |
| No Constraints | 0.8328 | 0.5548 | −0.0382 |
| Independent Rule-Based Ranker | 0.5963 | 0.5355 | −0.2747 |
| No Semantic Retrieval | 0.5528 | 0.4419 | −0.3182 |
| Rule-Based Only | 0.5066 | 0.5419 | −0.3644 |

Removing semantic retrieval causes Kendall's τ to collapse from 0.5277 to 0.0140
— an almost complete loss of ranking consistency. The constraint engine's
NDCG impact is small (−0.0382) but its CVR impact is large (+21.1% relative).
Both matter.

![Ablation NDCG@10](data/eval/plots/ablation_ndcg_bar.png)
![Ablation NDCG/CVR trade-off](data/eval/plots/ablation_tradeoff.png)
![Ablation results table](data/eval/plots/ablation_table.png)

### Robustness

| Label Noise Rate | NDCG@10 |
|---|---|
| 0% | 0.8750 |
| 20% | 0.8653 |
| 40% | 0.8296 |

Kendall's τ = 0.9505 under moderate score perturbation (σ = 0.03), 0.9016 at
σ = 0.10. The system is stable.

![NDCG@10 under label noise](data/eval/plots/label_noise_robustness.png)
![Kendall's τ under score perturbation (boxplot)](data/eval/plots/stability_tau_boxplot.png)
![Stability heatmap across perturbation regimes](data/eval/plots/stability_heatmap.png)
![NDCG@10 under retrieval/score noise](data/eval/plots/stability_noise_curve.png)

### Sensitivity

γ (constraint penalty coefficient) was swept from 0 to 1, a constraint stress
test was run across loose/medium/strict regimes, and cross-category generalisation
was checked across query splits.

![γ sweep, stress test, and generalisation (combined)](data/eval/plots/sensitivity_combined.png)
![NDCG@10 across the γ sweep](data/eval/plots/sensitivity_gamma_curve.png)
![Constraint stress test](data/eval/plots/sensitivity_stress_bar.png)
![Cross-category generalisation](data/eval/plots/sensitivity_generalization.png)

### Fairness

| Metric | Value | Interpretation |
|---|---|---|
| Exposure ratio (Tier-2 / Metro) | 1.0099 | Near-ideal 1.0 |
| Disparate Impact Ratio @ k=10 | 0.9314 | Within accepted 0.8–1.25 range |
| Mann-Whitney p-value | 0.3084 | No significant score difference |
| KS test p-value | 0.4522 | No significant distribution difference |
| Counterfactual score change | 0.0000 | Flipping location tag → zero score change |

![Supplier exposure by geographic tier](data/eval/plots/fairness_exposure_bar.png)
![Disparate impact ratio curve](data/eval/plots/fairness_dir_curve.png)
![Score distribution: Metro vs. Tier-2/3](data/eval/plots/fairness_score_distribution.png)
![Counterfactual location-flip test](data/eval/plots/fairness_counterfactual.png)

### SHAP Explainability

![SHAP mean |value| summary](data/eval/plots/shap_summary_bar.png)
![SHAP summary beeswarm](data/eval/plots/shap_summary_beeswarm.png)
![SHAP feature correlation heatmap](data/eval/plots/shap_heatmap.png)
![SHAP waterfall — single supplier decision](data/eval/plots/shap_waterfall.png)
![SHAP force plot — top-ranked supplier](data/eval/plots/shap_force_top_supplier.png)
![SHAP dependence — faiss_score](data/eval/plots/shap_dependence_faiss_score.png)
![SHAP dependence — cert_match](data/eval/plots/shap_dependence_cert_match.png)
![SHAP dependence — location_match](data/eval/plots/shap_dependence_location_match.png)
![SHAP dependence — price_ratio](data/eval/plots/shap_dependence_price_ratio.png)

### Latency (Warm-Start, Models Resident in Memory)

| Stage | Latency |
|---|---|
| SBERT query encode + FAISS top-100 | 26 ms |
| Cross-encoder rerank (100 candidates) | 235 ms |
| Constraint filtering | 1 ms |
| LightGBM ranking | 4 ms |
| SHAP decision trace | 8 ms |
| **Total** | **578 ms** |

---

## Evaluation Artefacts

Every figure above is generated by a corresponding script in `eval/` and backed
by a raw CSV/JSON file, so every number in the README and the paper is
reproducible end-to-end.

| Artefact | Generated by | Raw data |
|---|---|---|
| Baseline comparison | `eval/baselines.py` | `eval/baseline_results.csv`, `eval/significance_results.csv` |
| Ablation study | `eval/ablation.py` | `eval/ablation_results.csv` |
| Stability / robustness | `eval/stability.py` | `eval/stability_results.csv`, `eval/stability_score_perturbation.csv`, `eval/stability_retrieval_noise.csv` |
| Sensitivity (γ sweep, stress test, generalisation) | `eval/sensitivity.py` | `eval/sensitivity_gamma.csv`, `eval/sensitivity_stress.csv`, `eval/sensitivity_generalization.csv` |
| Fairness | `eval/fairness.py` | `eval/fairness_results.csv` |
| SHAP attribution | `eval/shap_analysis.py` | `eval/shap_values.csv`, `eval/shap_summary_statistics.csv` |
| Label-noise robustness | `eval/label_noise_analysis.py` | `eval/label_noise_results.csv` |
| Label/baseline overlap audit | `eval/check_label_baseline_overlap.py` | — |
| Case study (Section RQ5) | `eval/case_study.py`, `eval/find_successful_case_study.py` | `eval/case_study.md`, `eval/case_study_success.md`, `eval/case_study.json`, `eval/case_study_candidates.json` |

All generated plots live in `eval/plots/`; all generated tables live alongside
the CSVs in `eval/`.

---

## Repository Reading Guide

If you are reviewing the project quickly, start here:

| Need | Read or run |
|---|---|
| Understand the full idea | This README: pipeline walkthrough, label-independence problem, honest evaluation |
| See the architecture | `IEEE_ARCHITECTURE.png` (also embedded above) |
| Run the full data pipeline | `python pipeline/run_all.py --full` |
| Run evaluation | `python eval/baselines.py` → `eval/ablation.py` → `eval/fairness.py` → `eval/shap_analysis.py` |
| Start the API | `uvicorn backend.app.main:app --reload --port 8000` |
| Understand constraint logic | `backend/app/models/constraint_engine.py` |
| Understand ranking model | `backend/app/models/ranker.py` + `backend/app/models/train_lambdarank.py` |
| Understand labelling decisions | `backend/app/training/weak_label_generator.py` + `configs/weak_labels.yaml` |
| Understand SHAP output | `backend/app/services/explanation.py` + `backend/app/services/decision_trace.py` |
| Inspect feature engineering | `features/feature_builder.py` |
| Change training parameters | `configs/weak_labels.yaml` |

---

## Project Architecture

```
┌───────────────────────────────────────────────────────────────────┐
│                         INPUT                                     │
│           Procurement query with constraints                      │
│    "ISO certified food packaging, £50k budget, Chennai"           │
└───────────────────────────┬───────────────────────────────────────┘
                            │
                            ▼
┌───────────────────────────────────────────────────────────────────┐
│               Semantic Retrieval Layer                            │
│   backend/app/models/retriever.py                                 │
│   SBERT (all-mpnet-base-v2) → FAISS flat inner-product index      │
│   Cross-encoder (ms-marco-MiniLM-L-6-v2) reranks top-100         │
│   Query expansion via synonym map for procurement vocabulary      │
└───────────────────────────┬───────────────────────────────────────┘
                            │
                            ▼
┌───────────────────────────────────────────────────────────────────┐
│               Constraint Engine                                   │
│   backend/app/models/constraint_engine.py                        │
│   Hard checks: budget / MOQ / certification / location            │
│   Violations → tagged with reason + penalty score V(s,C)          │
│   Suppliers are never silently dropped                            │
└───────────────────────────┬───────────────────────────────────────┘
                            │
                            ▼
┌───────────────────────────────────────────────────────────────────┐
│               Feature Engineering                                 │
│   features/feature_builder.py                                     │
│   9-dimensional vector per (query, supplier) pair                 │
│   price_match · price_ratio · price_distance · location_match     │
│   cert_match · years_normalized · is_manufacturer                 │
│   is_trading_company · faiss_score                                │
└───────────────────────────┬───────────────────────────────────────┘
                            │
                            ▼
┌───────────────────────────────────────────────────────────────────┐
│               LightGBM LambdaRank                                 │
│   backend/app/models/ranker.py                                    │
│   Score(q,s) = f_θ(q,s) − γ·V(s,C)                              │
│   Trained on 18,525 query-grouped pairs (247 queries)             │
│   XGBoost and pointwise-regression backends also available        │
│   MMR diversification on final top-k                              │
└───────────────────────────┬───────────────────────────────────────┘
                            │
                            ▼
┌───────────────────────────────────────────────────────────────────┐
│               Explainability Layer                                │
│   backend/app/services/explanation.py                             │
│   backend/app/services/decision_trace.py                         │
│   backend/app/services/what_if_simulator.py                      │
│   SHAP TreeExplainer → per-feature contributions                  │
│   Natural-language trace per supplier                             │
│   What-if constraint counterfactuals                              │
└───────────────────────────┬───────────────────────────────────────┘
                            │
                            ▼
┌───────────────────────────────────────────────────────────────────┐
│               API + Application Layer                             │
│   backend/app/api/recommend.py  →  /recommend, /what-if, /compare │
│   backend/app/api/chat.py       →  /chat  (SourceBot, Groq-backed)│
│   backend/app/api/quote.py      →  /quote/draft, /refine, /pdf    │
│   backend/app/api/auth.py       →  JWT, OAuth, billing            │
└───────────────────────────┬───────────────────────────────────────┘
                            │
                            ▼
┌───────────────────────────────────────────────────────────────────┐
│               Offline Evaluation Pipeline                         │
│   eval/baselines.py          BM25, SBERT-cosine, rule-based, LTR  │
│   eval/ablation.py           Component removal experiments        │
│   eval/fairness.py           Exposure ratio, DI, KS, counterfactual│
│   eval/shap_analysis.py      Summary bar, beeswarm, waterfall     │
│   eval/stability.py          Perturbation + label noise           │
│   eval/sensitivity.py        γ sweep, stress test, cross-category │
│   eval/label_noise_analysis.py  NDCG under 0–40% corruption       │
└───────────────────────────────────────────────────────────────────┘
```

A rendered, IEEE-paper version of this same pipeline is shown in
[System Architecture](#system-architecture) above.

---

## Codebase Structure

```
SourceUp/
├── IEEE_ARCHITECTURE.png       # End-to-end architecture figure used in the paper
├── backend/
│   └── app/
│       ├── api/
│       │   ├── recommend.py        # /recommend, /what-if, /compare, /stats
│       │   │                       # SearchQuery Pydantic model; async handlers
│       │   │                       # Retrieval mode: "faiss" | "bm25" | "hybrid"
│       │   ├── chat.py             # /chat — SourceBot conversational assistant
│       │   ├── quote.py            # /quote/draft, /refine, /export-pdf (ReportLab)
│       │   └── auth.py             # /auth/* — JWT, bcrypt, Google OAuth, billing
│       ├── models/
│       │   ├── retriever.py        # SBERT encoding, FAISS search, query expansion
│       │   │                       # Cross-encoder reranking, BM25 fallback
│       │   │                       # Singleton index: load once, reuse forever
│       │   ├── ranker.py           # LightGBM inference + MMR diversification
│       │   │                       # extract_features_batch(), apply_mmr()
│       │   │                       # FEATURE_COLS: canonical 9-feature list
│       │   ├── constraint_engine.py# Hard constraint checking and penalty tagging
│       │   │                       # Budget / MOQ / certification / location
│       │   │                       # Violations tagged, never silently dropped
│       │   ├── train_lambdarank.py # LambdaRank training entry point
│       │   ├── train_ranker.py     # XGBoost/regression ranker training
│       │   └── embeddings/         # Serialised .pkl model files (not committed)
│       │       ├── ranker_lightgbm.pkl
│       │       ├── xgb_ranker.pkl
│       │       └── fairness_weights.pkl
│       ├── services/
│       │   ├── explanation.py      # SHAP TreeExplainer wrappers
│       │   ├── decision_trace.py   # Per-supplier natural-language audit trail
│       │   │                       # 7 contribution factors + constraint effects
│       │   │                       # generate_comparative_trace() for A/B comparisons
│       │   └── what_if_simulator.py# Constraint counterfactual simulation
│       │                           # "What if budget +20%?" → simulated re-ranking
│       ├── training/
│       │   └── weak_label_generator.py  # Heuristic relevance labelling
│       │                                # Quantile-based rebucketing (v3.0)
│       │                                # Feasibility gate to decouple high labels
│       │                                # from constraint violations (v3.1)
│       ├── database/mongodb.py     # Motor async MongoDB client
│       └── main.py                 # FastAPI app factory; startup banner; health check
├── sourcebot/
│   ├── orchestrator.py             # Intent routing: search / explain / what-if / info
│   ├── nlu/
│   │   ├── parser.py               # Rule-based NLU + intent classification
│   │   │                           # 'information' | 'product_search' | 'explanation_request'
│   │   │                           # 'what_if' intents with confidence scores
│   │   ├── rules.py                # Deterministic procurement entity extraction
│   │   └── gpt_fallback.py         # Groq fallback for ambiguous queries
│   ├── memory/session.py           # Redis/Memurai session memory
│   └── responses/info_responses.py # Groq-backed informational response generator
├── pipeline/
│   ├── validate_merge.py           # Raw data validation and source merging
│   ├── clean_normalize.py          # ETL: dedup, normalise, standardise fields
│   ├── assign_locations.py         # Metro / Tier-2 / Tier-3 deterministic assignment
│   │                               # 390,198 Metro / 320,840 Tier-2 / 160,429 Tier-3
│   ├── incremental_faiss.py        # Embed + index; append without full rebuild
│   └── run_all.py                  # CLI orchestrator: --full / --train-lambdarank
│                                   # --run-analysis / --shap-analysis / --limit N
├── features/
│   └── feature_builder.py          # 9-feature vector construction per (query, supplier)
│                                   # _stable_hash() for reproducible RNG seeds
│                                   # Hard-negative sampling: 40% Metro / 40% Tier-2 / 20% International
├── eval/
│   ├── baselines.py                # BM25 / SBERT-cosine / rule-based / random / LambdaRank
│   ├── ablation.py                 # Remove semantic retrieval / constraints / LTR
│   ├── stability.py                # Score perturbation + retrieval noise
│   ├── sensitivity.py              # γ sweep / constraint stress test / cross-category
│   ├── fairness.py                 # Exposure ratio / DI / KS / counterfactual
│   ├── shap_analysis.py            # Summary bar / beeswarm / waterfall / dependence plots
│   ├── label_noise_analysis.py     # NDCG under 0–40% label corruption
│   ├── case_study.py               # End-to-end case study runner
│   ├── find_successful_case_study.py  # Search for a fully-feasible case-study query
│   ├── check_label_baseline_overlap.py  # Audits signal overlap between labeller and baseline
│   ├── dataset_diagnostic.py       # Dataset sanity checks
│   ├── verify_location_match.py    # Location-match field verification
│   ├── *.csv / *.json / *.md       # Generated evaluation results and case studies
│   └── plots/                      # All generated evaluation figures (PNG)
├── evaluation/
│   └── metrics.py                  # NDCG@k, MAP, MRR, P@k, Kendall τ, CVR
├── rule_baseline.py                # Label-independent rule-based scorer
│                                   # Uses only features labelling function never sees
├── configs/
│   └── weak_labels.yaml            # Heuristic weights, quantile thresholds, feasibility floor
├── config.py                       # Centralised path + environment configuration
│                                   # SOURCEUP_DIR override; cfg.validate() warnings
└── somasjar.jar                    # Java/Selenium scraper for GlobalSources/IndiaMart/TradeIndia
```

---

## Design Deep-Dives

### Constraint Engine (`backend/app/models/constraint_engine.py`)

The constraint engine runs between retrieval and ranking. It checks each of the
100 retrieved candidates against four hard constraints and tags failures rather
than removing them.

#### Why Tag Instead of Drop

Silently dropping constraint-violating suppliers before ranking has a practical
problem: if all 100 retrieved candidates fail a constraint (as happens when the
corpus has no Chennai-based suppliers), the system would return an empty list.
Instead, SourceUp tags every violation with the specific constraint it failed,
applies a penalty to the ranking score, and surfaces the best available options
with violation reasons visible in the decision trace. A buyer searching for
Chennai ISO suppliers who gets only Chinese suppliers back can at least see why
— and decide whether to relax the location constraint.

#### Constraint Checks

| Constraint | Check | Failure tag |
|---|---|---|
| Budget | `supplier_price ≤ max_price` | `budget_violation` |
| MOQ | `moq ≤ moq_budget` | `moq_affordability` |
| Certification | fuzzy match via rapidfuzz | `certifications` |
| Location | city-level string match | `location` |

The penalty term in the ranking score is:

```
V(s,C) = (1/3) × [(1 − price_match) + (1 − location_match) + (1 − cert_match)]
Score(q,s) = f_θ(q,s) − γ · V(s,C)
```

Sensitivity analysis (γ ∈ [0, 1]) showed NDCG@10 changed by at most 0.0034
across the full sweep. Binary constraint features, not the continuous penalty
term, dominate the ranking signal. γ = 0.3 is the production default.

---

### Weak-Supervision Labelling (`backend/app/training/weak_label_generator.py`)

This is where the benchmark is constructed and where the most careful design
decisions were made.

#### Version History

The labelling logic went through three versions, each fixing a discovered failure
mode:

**v1.0 — Fixed absolute thresholds:** Labels 0–5 assigned by comparing the
continuous weak-label score against fixed cutoffs (0.25, 0.40, 0.55, 0.70, 0.85).
Diagnostic finding: the weighted score rarely exceeds ~0.55 even for the
best candidates (most component signals are sparse binary values; faiss_score
averages ~0.25). Result: 83%+ of rows in labels {1, 2}, only ~1.1% in
labels {4, 5}. LambdaRank needs graded relevance to optimise; this distribution
gave it almost nothing to work with.

**v2.0 (v3.0 in code) — Quantile-based rebucketing:** Labels assigned by
percentile of the actual weak-label score distribution, not by absolute cutoffs.
Top 2% → label 5, next 8% → label 4, and so on. This spread labels meaningfully
across all six buckets. New failure mode discovered: `faiss_score` has far more
variance than binary `location_match` and `cert_match`, so semantically-excellent-
but-constraint-violating rows could land in the top quantiles. CVR rose from
~0.37 to ~0.45–0.47 in two seeded reruns — not sampling noise.

**v2.1 (v3.1 in code) — Feasibility gate:** A minimum number of hard constraints
must be satisfied before a quantile-assigned high label is kept. A row in the
top 2% of weak-label scores but satisfying zero constraints is downgraded one
tier at a time until it reaches a label whose feasibility floor it meets.
Result: CVR returns to acceptable range without disturbing the quantile class
balance.

```yaml
# configs/weak_labels.yaml
feasibility_floor:
  label_5: 2   # top label requires ≥2 of 3 constraints satisfied
  label_4: 2
  label_3: 1
  label_2: 0
  label_1: 0
```

#### Reproducibility Fix

The feature builder previously seeded per-query RNG using Python's built-in
`hash(str)`. Python randomises string hashing per-process (PYTHONHASHSEED is
a random salt), so "seeded" generators were not reproducible across runs. This
caused run-to-run differences in retrieved candidates, sampled hard negatives,
and injected FAISS noise — producing misleading swings in CVR and NDCG that
had nothing to do with model or labelling changes.

Fixed by `_stable_hash()` in `features/feature_builder.py`, which uses
MD5-based deterministic hashing. Seeds are now reproducible across machines,
operating systems, and Python versions.

---

### LambdaRank Training (`backend/app/models/train_lambdarank.py`)

#### Why LambdaRank Over Pointwise Regression

Pointwise regression treats each (query, supplier) pair independently and
minimises a regression loss against the continuous relevance label. This is
efficient but ignores the structure of the problem: what matters is not the
absolute score of each supplier but the relative ordering within a query group.
LambdaRank optimises NDCG directly by computing pairwise gradient contributions
weighted by the change in NDCG that swapping two items would produce. It sees
the entire query group at once and optimises ranking quality rather than
score prediction.

The XGBoost and LightGBM pointwise-regression backends are trained on the same
splits for comparison. Both score within 0.01 NDCG@10 of LambdaRank — consistent
with the literature finding that backend choice matters less than feature quality
and query-level supervision signal.

#### Feature Column Stability

`FEATURE_COLS` is defined once in `ranker.py` and imported everywhere that
builds or consumes feature vectors. This prevents the training/inference
column mismatch bug that previously caused silent failures when feature
sets diverged between `feature_builder.py` and `ranker.py`.

Note: `faiss_rank` (the ordinal retrieval rank) was initially included but
removed after SHAP analysis showed mean |SHAP| < 0.03 — negligible attribution
relative to the highly correlated `faiss_score`. Including a near-zero-attribution
feature adds noise and makes the feature importance analysis harder to read.

---

### Decision Trace (`backend/app/services/decision_trace.py`)

The decision trace is the production explainability output. It is not a
diagnostic tool — it is part of every `/recommend` response.

For each ranked supplier, `generate_trace()` returns:

```json
{
  "supplier_name": "Dalian Huiyou Packaging Co., Ltd.",
  "final_score": 0.06,
  "contributions": {
    "semantic_match": {"raw_score": 0.312, "contribution": 0.016, "explanation": "..."},
    "price": {"raw_score": 1.0, "contribution": 0.350, "explanation": "Price £0.04 within budget £50,000"},
    "location": {"raw_score": 0.0, "contribution": 0.0, "explanation": "No match: China ≠ Chennai"},
    "certification": {"raw_score": 1.0, "contribution": 0.200, "explanation": "ISO certification match confirmed"}
  },
  "constraints": {"passed_all": false, "details": {"location": {"passed": false}}},
  "summary": ["ISO certification match confirmed", "Price within budget", "Product matches with 0.31 similarity"]
}
```

`generate_comparative_trace()` accepts two suppliers and returns a ranked list
of the factors that most explain their score difference, in descending order of
magnitude. This powers the "Why was A ranked higher than B?" intent in SourceBot.

Summary strings are truncated to 60 characters and capped at three items to
fit comfortably as UI tags. Factors with contribution ≤ 0.01 and unconstrained
fields ("No price constraint specified") are suppressed — they are meaningless
to the user and produce visual clutter.

---

### SourceBot (`sourcebot/orchestrator.py`)

SourceBot is a Groq-backed procurement conversational assistant with a rule-based
NLU front end. Intent classification runs before any LLM call, routing common
patterns deterministically without burning API tokens.

#### Intent Routing

```
User message → classify_intent()
    │
    ├── 'explanation_request'  → handle_explanation_request()
    │                            Calls decision trace for the last result set
    │
    ├── 'what_if'              → handle_what_if_query()
    │                            Calls WhatIfSimulator with parsed constraint changes
    │
    ├── 'information'          → generate_info_response_enhanced()
    │                            Groq-backed; receives only raw text, not parsed_data
    │                            (FIX: passing parsed_data caused prompt variable collisions)
    │
    └── 'product_search'       → parse() → /recommend API call
                                 rule_parse() extracts product, price, location, cert
                                 gpt_parse() fallback for ambiguous queries
```

The NLU parser uses pattern matching over 30+ procurement-specific phrases
before falling back to Groq. "What is ISO certification?" → `information`.
"Best ISO packaging suppliers under £2" → `product_search`. The distinction
matters because `information` queries should never trigger a FAISS retrieval.

---

## Quick Start

### 1. Clone and configure

```bash
git clone https://github.com/Somaskandan931/SourceUp.git
cd SourceUp
```

Create `.env` in the project root:

```env
# Minimum required
GROQ_API_KEY=your_groq_api_key
SECRET_KEY=replace_with_openssl_rand_hex_32
MONGODB_URI=mongodb://localhost:27017
MONGODB_DB=sourceup

# Optional — billing disabled without these
UPI_ID=yourname@upi

# Optional — Google OAuth disabled without these
GOOGLE_CLIENT_ID=your_google_client_id
GOOGLE_CLIENT_SECRET=your_google_client_secret
GOOGLE_REDIRECT_URI=http://localhost:8000/auth/google/callback
FRONTEND_URL=http://localhost:3000

# Demo access
DEMO_EMAIL=demo@sourceup.com
DEMO_PASSWORD=demopass123
DEMO_PLAN=pro

# Optional — session memory
REDIS_HOST=localhost
REDIS_PORT=6379
```

### 2. Install Python dependencies

```bash
pip install fastapi uvicorn python-dotenv groq httpx reportlab
pip install sentence-transformers faiss-cpu lightgbm xgboost shap
pip install pandas numpy scikit-learn scipy matplotlib seaborn
pip install motor pymongo python-jose[cryptography] passlib[bcrypt]
pip install redis rapidfuzz
```

### 3. Build the data pipeline

If you have raw supplier CSVs in `data/merged/suppliers_all.csv`:

```bash
# Full pipeline: validate → clean → assign locations → embed → index → train → evaluate
python pipeline/run_all.py --full

# Individual steps:
python pipeline/run_all.py --train-lambdarank    # train LambdaRank only
python pipeline/run_all.py --run-analysis        # baselines + ablation
python pipeline/run_all.py --shap-analysis       # SHAP feature attribution
python pipeline/run_all.py --limit 5000          # limit dataset size for testing
```

The pipeline writes clean data to `data/clean/suppliers_clean.csv`, the FAISS
index to `data/embeddings/suppliers.faiss`, and the trained model to
`backend/app/models/embeddings/ranker_lightgbm.pkl`.

> **Note:** The FAISS index, trained `.pkl` files, and raw supplier CSVs are
> not committed. Run the pipeline to generate them, or contact the author for
> pre-built artefacts.

### 4. Start the backend

```bash
python -m uvicorn backend.app.main:app --reload --port 8000
```

The startup banner prints the status of every critical file:

```
🚀 SourceUp API starting up
  Clean Data:    ✅ data/clean/suppliers_clean.csv
  FAISS Index:   ✅ data/embeddings/suppliers.faiss
  LGBM Model:    ✅ backend/app/models/embeddings/ranker_lightgbm.pkl
  Groq:          ✅ configured
```

API docs: http://localhost:8000/docs  
Health check: http://localhost:8000/health

### 5. Start the frontend

```bash
cd frontend-react
npm install
npm start
# → http://localhost:3000
```

### 6. Demo access

```
Email:    demo@sourceup.com
Password: demopass123
```

Or one-click via `POST /auth/demo-login` — no sign-up required.

---

## API Reference

### Supplier Recommendation

```http
POST /recommend
```

```json
{
  "product": "ISO certified food packaging",
  "max_price": 2.0,
  "moq_budget": 500,
  "location": "Mumbai",
  "location_mandatory": false,
  "certification": "ISO",
  "max_lead_time": 30,
  "min_years_experience": 3,
  "retrieval_mode": "hybrid",
  "enable_explanations": true,
  "enable_cross_encoder": true,
  "enable_diversity": true,
  "top_k": 10
}
```

`retrieval_mode` options: `"faiss"` (semantic only), `"bm25"` (lexical only),
`"hybrid"` (both, recommended). `enable_explanations: true` returns the full
SHAP decision trace per supplier. `enable_cross_encoder: true` adds ~235 ms of
latency but improves candidate quality. `location_mandatory: false` means
location is used as a soft ranking signal rather than a hard filter.

```http
POST /what-if     # Re-rank with modified constraints; no new retrieval
POST /compare     # Compare two suppliers or scenarios
GET  /stats       # Dataset and index statistics
```

### Conversational Assistant

```http
POST /chat        # SourceBot — natural-language procurement queries
                  # Handles: product search / explain / what-if / informational
```

### RFQ Workflow

```http
POST /quote/draft       # Generate RFQ email from supplier + requirements
POST /quote/refine      # Refine draft: shorten / formalise / rewrite
POST /quote/export-pdf  # Download print-ready RFQ PDF (ReportLab)
```

### Auth and Billing

```http
POST /auth/register
POST /auth/login
GET  /auth/me
POST /auth/demo-login
GET  /auth/google/login
GET  /auth/google/callback
GET  /auth/billing/plans
POST /auth/billing/order
POST /auth/billing/verify
```

---

## Evaluation Pipeline

All evaluation scripts are independently re-runnable with fixed random seeds.

```bash
python eval/baselines.py            # Full baseline comparison table + plots
python eval/ablation.py             # Component removal study
python eval/stability.py            # Kendall τ under score perturbation and retrieval noise
python eval/sensitivity.py          # γ sweep, constraint stress test, cross-category split
python eval/fairness.py             # Exposure ratio, DI, KS test, counterfactual
python eval/shap_analysis.py        # Feature importance summary, dependence, waterfall
python eval/label_noise_analysis.py # NDCG under 0–40% label corruption
python eval/case_study.py           # End-to-end case study: hard + soft constraint queries
python eval/check_label_baseline_overlap.py  # Audit signal overlap between labeller and baselines
```

Outputs: CSVs/JSON/Markdown in `eval/`, PNGs in `eval/plots/` (see
[Evaluation Artefacts](#evaluation-artefacts) for the full mapping).

---

## Case Study: What the System Actually Does

Two contrasting queries from `eval/case_study.md`, reproduced exactly from the
deployed system. No values are adjusted.

### Case A — Hard Constraint: No Feasible Supplier Exists

```
Query:  food packaging suppliers
Budget: £50,000 · MOQ: 500 · Location: Chennai (mandatory) · Certification: ISO

Retrieved (FAISS):           100
After cross-encoder rerank:  100
Fully feasible:              0      ← zero Chennai ISO suppliers in corpus
Location matches:            0
Final recommendations:       10     ← 10 best-available constraint-violators
Top supplier score:          0.06
```

SHAP attribution for the top result:

| Feature | Contribution |
|---|---|
| cert_match | +0.0750 (dominant positive) |
| faiss_score | +0.0060 |
| price_match | +0.0017 |
| location_match | −0.0039 (dominant negative) |

The system correctly surfaces: the unmet location constraint is visible in the
SHAP trace as the dominant negative contributor. The buyer sees 10 Chinese
food-packaging suppliers with ISO certification and their violation tags — not
an empty list and not a misleading "no results" error.

### Case B — Soft Constraint: Price and MOQ Satisfied

```
Query:  packaging materials supplier
Budget: £1,00,000 · MOQ: 500 · Location: Mumbai (soft preference) · Certification: none

Retrieved (FAISS):           100
Price + MOQ feasible:        76
Location matches:            0
Final recommendations:       10     ← diverse set: food boxes, cartons, storage, cosmetic
Top supplier score:          0.0469
```

SHAP attribution for the top result:

| Feature | Contribution |
|---|---|
| faiss_score | +0.0698 (dominant positive) |
| price_match | +0.0022 |
| location_match | −0.0029 |

With location as a soft preference, semantic similarity takes over as the
dominant driver. The absence of a location match is still reported in the trace
rather than concealed — the buyer knows the top result is not in Mumbai.

Full write-ups, including the search for the most representative cases, live in
`eval/case_study.md` and `eval/case_study_success.md`, with the raw candidate
pools in `eval/case_study.json` and `eval/case_study_candidates.json`.

---

## Feature Set

The LambdaRank model uses a 9-dimensional feature vector per (query, supplier)
pair. All features are normalised to [0, 1].

| Feature | Description | Type |
|---|---|---|
| `price_match` | Supplier price ≤ budget ceiling | {0, 1} |
| `price_ratio` | Budget / price, capped at 1 | [0, 1] |
| `price_distance` | Normalised gap to budget threshold | [0, 1] |
| `location_match` | City-level delivery compliance | {0, 1} |
| `cert_match` | Required certification satisfied | {0, 1} |
| `years_normalized` | Platform tenure, capped at 10 years | [0, 1] |
| `is_manufacturer` | Supplier type flag | {0, 1} |
| `is_trading_company` | Supplier type flag | {0, 1} |
| `faiss_score` | Bi-encoder / cross-encoder similarity | [0, 1] |

SHAP attribution (mean |SHAP| across 62 test queries):

```
faiss_score     ████████████████████████████████  0.6513
location_match  ███                               0.0780
cert_match      █                                 0.0338
```

Semantic similarity dominates by an order of magnitude. This is expected:
buyer queries and supplier descriptions use different vocabulary for the same
need, so a feature capturing meaning rather than surface wording absorbs most
discriminative signal. Price features rank lower because the hard constraint
filter captures most budget signal before the learned model sees the candidates.

---

## Dataset

| Statistic | Value |
|---|---|
| Raw merged supplier records | 1,296,743 |
| Cleaned, deduplicated | 871,467 |
| Indexed in FAISS | 19,750 |
| Procurement queries | 309 |
| Supplier-query pairs | 23,175 |
| Train split | 247 queries / 18,525 pairs |
| Test split | 62 queries / 4,650 pairs |
| Geographic groups | Metro 390,198 · Tier-2 320,840 · Tier-3 160,429 |
| Hard-negative sampling | 40% Metro · 40% Tier-2 · 20% International |

Supplier data sourced from GlobalSources, IndiaMart, and TradeIndia via
`somasjar.jar` (Java/Selenium). The scraper runs separately; cleaned output
feeds the Python pipeline. A small enrichment subset (80 of 871,467 rows)
supplies buyer ratings, city, and certification-count fields; outside this
subset these fields are absent rather than imputed.

---

## Limitations

These are stated explicitly because they are real constraints, not caveats added
for appearance.

**Evaluation is entirely offline.** No online A/B testing, no click-through
measurement. Real-world translation of offline NDCG gains is unknown.

**The benchmark indexes 19,750 of 871,467 cleaned suppliers.** Reported results
characterise ranking quality over this indexed subset. Full-catalogue performance
has not been validated.

**Buyer rating coverage is extremely sparse.** 80 of 871,467 suppliers have
buyer ratings. The field carries near-zero variance in training and contributes
negligible effective signal despite being collected.

**The benchmark covers a single geographic market.** India-facing procurement,
sourced largely from international suppliers. Performance in other markets may
differ.

**Warm-start latency is a single representative measurement.** A repeated-trial
distribution (mean, P95, P99) across many queries is future work.

**The residual circularity between labelling and learned features is mitigated,
not eliminated.** The label-independent baseline addresses the most obvious
overlap. The learned model still receives `faiss_score` as a feature, and
`faiss_score` is also an input to the weak-label function. This is noted
explicitly and not correctable without real procurement feedback data.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | FastAPI, Pydantic, Uvicorn |
| Retrieval | SBERT (`all-mpnet-base-v2`), FAISS (flat inner-product) |
| Cross-encoder | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Ranking | LightGBM (`lambdarank` objective), XGBoost (`rank:pairwise`) |
| Explainability | SHAP TreeExplainer |
| LLM | Groq API |
| Database | MongoDB (Motor async) |
| Auth | JWT (`python-jose`), bcrypt (`passlib`), Google OAuth |
| PDF | ReportLab |
| Frontend | React 19, React Router |
| Session memory | Redis / Memurai (optional) |
| Scraper | Java / Selenium (`somasjar.jar`) |
| Evaluation | pandas, NumPy, scikit-learn, scipy, matplotlib, seaborn |

---

## Plans

| Feature | Free | Pro | Enterprise |
|---|---|---|---|
| Supplier searches | Limited | Unlimited | Unlimited |
| SourceBot chat | ✓ | ✓ | ✓ |
| RFQ wizard | Limited | ✓ | ✓ |
| RFQ PDF export | Limited | ✓ | ✓ |
| Decision traces | Limited | ✓ | ✓ |
| What-if scenarios | — | ✓ | ✓ |
| API access | — | — | ✓ |

---

## Future Work

- Replace weak supervision with real procurement interaction data (click-through,
  purchase confirmation, RFQ completion) and re-evaluate offline metrics against
  real relevance signals
- Quantify cold-start/new-supplier exposure fairness via exploration- or
  diversity-aware reranking
- Structured certification-field extraction and city-level location normalisation
  to reduce the 0% location-match rate in the current corpus
- Extend FAISS indexing from 19,750 toward the full 871,467-supplier catalogue
  as compute permits
- Repeated-trial latency benchmark (mean, median, P95, P99) to complement the
  single-measurement figures
- Multilingual procurement query support for Indian regional languages

---

## Security Note

This is a research prototype. Before any production deployment: rotate all
secrets, remove demo credentials from the codebase, enforce HTTPS, harden CORS
origins, add rate limiting, replace the UPI prototype with a production payment
provider, and conduct a full security audit of the JWT implementation.

---

## Citation

If you use SourceUp in research, please cite:

```bibtex
@inproceedings{somaskandan2026sourceup,
  author      = {R. Somaskandan},
  title       = {{SourceUp}: A Constraint-Aware, Explainable, and Fair
                 Semantic Retrieval Framework for {SME} Supplier Discovery},
  booktitle   = {Proceedings of the IEEE Conference},
  year        = {2026},
  institution = {Sathyabama Institute of Science and Technology}
}
```

---

## Acknowledgements

SourceUp uses the following open-source components:

- **Sentence-BERT** — Reimers and Gurevych, EMNLP 2019 — dense query/supplier
  embeddings via `all-mpnet-base-v2`
- **FAISS** — Johnson, Douze, Jégou, IEEE Trans. Big Data 2021 — approximate
  nearest-neighbour search over supplier embeddings
- **LightGBM** — Ke et al., NeurIPS 2017 — gradient-boosted LambdaRank model
- **SHAP** — Lundberg and Lee, NeurIPS 2017 — TreeExplainer for feature attribution
- **LambdaRank** — Burges et al., Microsoft Research Technical Report 2010 —
  pairwise ranking objective with NDCG-weighted gradients

---

## Author

**R. Somaskandan** — M.Sc. Computer Science (AI), Sathyabama Institute of Science and Technology, Chennai  
somaskandan931@gmail.com · [GitHub](https://github.com/Somaskandan931) · [Portfolio](https://rajagopal-somaskandan.netlify.app)

---

## License

MIT License — see [LICENSE](LICENSE) for details.