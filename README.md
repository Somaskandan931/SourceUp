SourceUp
Constraint-Aware Explainable AI Framework for SME Supplier Selection
Overview

SourceUp is an AI-powered supplier sourcing system designed for small and medium enterprises (SMEs). It combines semantic search, machine learning, and explainability to improve supplier discovery and procurement decision-making.

Unlike traditional sourcing platforms that rely on keyword matching or rule-based filtering, SourceUp models supplier selection as a constraint-aware ranking problem, where factors such as budget, delivery time, and supplier reliability are integrated directly into the ranking process.

The system is designed both as a functional platform and as a research framework for explainable AI in decision support systems.

Key Idea

Traditional pipeline:

Filter → Rank

SourceUp approach:

Constraint-Aware Ranking
Score(q,d)=f
θ
	​

(q,d)−γ⋅ConstraintViolation(d,C)

Where:

q = user query
d = supplier
C = constraints (budget, delivery, etc.)
fθ = learned ranking model
γ = penalty weight

This enables:

Constraint-aware recommendations
Trade-off modeling between cost, delivery, and quality
Explainable supplier selection
Features
Semantic supplier search using embeddings (SBERT + FAISS)
Constraint-aware ranking (budget, delivery, feasibility)
Explainable recommendations with decision traces
Conversational interface for natural language queries
Automated supplier data collection via scraping pipelines
Modular architecture for research and experimentation
System Architecture
Scraped Supplier Data
        ↓
Data Processing & Feature Engineering
        ↓
Semantic Retrieval (FAISS)
        ↓
Ranking Engine (Constraint-Aware LTR)
        ↓
Explainability Layer
        ↓
API / SourceBot Interface
Modules
1. Data Collection
Web scraping from B2B platforms (IndiaMART, GlobalSources)
Structured CSV dataset generation
Data cleaning and normalization
2. Feature Engineering Pipeline
Price normalization
Supplier reliability scoring
Delivery estimation (proxy features)
Constraint feasibility flags
3. Supplier Recommendation Engine
Semantic search using embeddings
Learning-to-Rank (planned: LightGBM / LambdaRank)
Constraint-aware scoring
4. Explainability Layer
Feature-level contribution analysis
Trade-off insights (cost vs delivery vs rating)
Decision trace for ranking comparisons
5. SourceBot (Conversational Interface)
Natural language query handling
Constraint extraction
Session-based interaction
6. Simulation & Evaluation
What-if analysis
Constraint sensitivity testing
Ranking stability experiments
Tech Stack
Backend: FastAPI (Python)
ML: Sentence Transformers, FAISS, Scikit-learn
Ranking (planned): LightGBM / XGBoost
Scraping: Java (Jsoup) + Python pipelines
Conversational AI: LangChain + OpenAI API
Storage: CSV (PostgreSQL-ready)
Frontend: Streamlit
Example Use Case

Input:
"Need disposable food containers under ₹5 with fast delivery"

System Output:

Ranked suppliers based on relevance + constraints
Explanation:
Supplier A ranked higher due to lower price and better rating
Supplier B penalized due to delivery delay
Research Alignment

This project explores:

Constraint-aware recommendation systems
Explainable AI in decision support
Learning-to-Rank for structured decision problems
Trade-offs between feasibility and relevance
Current Status
Scraping pipeline: Implemented
Semantic retrieval: Implemented
Feature engineering: In progress
Ranking model: In progress
Experimental evaluation: Planned
Future Work
Full Learning-to-Rank integration
Fairness-aware ranking
Reinforcement learning for adaptive recommendations
Multilingual support
Cloud deployment (Docker + Kubernetes)
Getting Started
git clone https://github.com/your-username/sourceup.git
cd sourceup
Run Backend
uvicorn main:app --reload
Run Scraper
java -jar scraper.jar input.csv output.csv
Project Structure
sourceup/
│
├── data/                  # Scraped datasets
├── scraper/               # Java scraping module
├── backend/               # FastAPI services
├── ranking/               # ML ranking models
├── chatbot/               # SourceBot module
├── experiments/           # Evaluation scripts
└── notebooks/             # Research analysis
Design Principles
Clear separation between data, ranking, and interface
Explainability over black-box decisions
Constraint-aware modeling instead of filtering
Modular and research-extensible architecture
License

This project is intended for academic and research use.
Licensing for commercial use can be defined in future versions.

Final Note

SourceUp is not just a sourcing tool.

It is a step toward building explainable, constraint-aware AI systems for real-world decision-making.