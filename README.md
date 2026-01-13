# SourceUp – Intelligent Supplier Sourcing Platform for SMEs

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.0+-yellow.svg)](https://lightgbm.readthedocs.io/)
[![License](https://img.shields.io/badge/License-Research-orange.svg)]()

---

## Overview

SourceUp is a **constraint-aware, AI-driven conversational supplier recommendation platform** designed specifically for resource-constrained Micro, Small, and Medium Enterprises (MSMEs). 

The platform addresses critical challenges in B2B procurement by:
- **Reducing supplier discovery time by 70%** through semantic search and intelligent ranking
- **Enforcing hard business constraints** (budget, MOQ, delivery deadlines, certifications) before ranking
- **Providing explainable recommendations** with transparent decision traces
- **Enabling what-if simulations** for trade-off analysis

Unlike traditional B2B marketplaces that treat constraints as soft preferences, SourceUp ensures all recommended suppliers are **operationally viable** by filtering infeasible candidates before ranking, reducing cognitive load and decision-making errors.

**Research-backed performance:**
- **NDCG@10: 0.7439** (9.2% improvement over rule-based baseline)
- **Precision@5: 92.5%** (9 out of 10 top-5 recommendations are highly relevant)
- **40-60% reduction** in infeasible recommendations through constraint-first design

📄 **Full IEEE research paper**: Available in `docs/paper.tex`

---

## Features

### **Conversational AI Interface**
- Natural language query understanding with **hybrid NLU pipeline** (rule-based + LLM fallback)
- **Intent classification** distinguishing product search from informational queries
- **Multi-turn conversations** with Redis-backed session memory
- Dynamic **LLM-powered responses** for procurement questions (ISO standards, certifications, trade terms)

### **Explainable Ranking System**
- **LightGBM-based Learning-to-Rank** optimized for NDCG@10
- **Transparent decision traces** breaking down ranking into feature contributions
- **Feature importance analysis** (Price: 77%, Semantic Relevance: 16%, Location: 6%)
- **Comparative explanations**: "Why was Supplier A ranked higher than Supplier B?"

### **Hard Constraint Enforcement**
- **Pre-ranking feasibility filtering** (reduces candidates by 40-60%)
- Enforces:
  - Budget limits and MOQ affordability
  - Delivery lead time deadlines
  - Required certifications (ISO, FDA, CE, RoHS, etc.)
  - Geographic location preferences
  - Minimum platform experience
- **Constraint satisfaction scoring** for soft preferences within viable suppliers

### **What-If Simulation**
- Explore trade-offs: *"What if I prioritize price over speed?"*
- **Budget sensitivity analysis**: Impact of increasing/decreasing budget
- **Constraint relaxation**: Effects of relaxing delivery or certification requirements
- **Priority rebalancing**: Simulating price-focused vs. quality-focused procurement

### **Production-Ready Architecture**
- **Automated Java-based scraper** for B2B marketplace data ingestion
- **FAISS-indexed semantic search** with 384D Sentence-BERT embeddings
- **Modular FastAPI backend** with health checks and comprehensive error handling
- **Streamlit MVP interface** for rapid prototyping
- **Redis/Memurai session management** for stateful conversations

---

## Tech Stack

### **Backend & APIs**
- **FastAPI** – Modern, high-performance web framework
- **Python 3.10+** – Core programming language
- **Uvicorn** – ASGI server for FastAPI

### **Machine Learning & Ranking**
- **LightGBM** – Primary learning-to-rank model (LambdaRank objective)
- **XGBoost** – Secondary LTR model (fallback)
- **Scikit-learn** – Feature engineering and validation
- **Sentence-BERT** (all-MiniLM-L6-v2) – 384D text embeddings
- **FAISS** (Facebook AI Similarity Search) – Billion-scale vector search

### **Natural Language Processing**
- **LangChain** – LLM orchestration framework
- **Groq/Ollama/OpenAI** – LLM providers for fallback parsing and info responses
- **Pydantic** – Data validation and schema enforcement

### **Data & Storage**
- **Redis/Memurai** – Session management and caching
- **PostgreSQL** (planned) – Structured data persistence
- **CSV** (current) – Lightweight data pipeline

### **Data Collection**
- **Java** – High-performance web scraper
- **Selenium** – Browser automation for dynamic content
- **Apache Airflow** (optional) – Workflow orchestration

### **Frontend & Visualization**
- **Streamlit** – Rapid prototyping interface
- **Matplotlib/Seaborn** – IEEE-compliant research visualizations

### **Development Tools**
- **Conda** – Environment management
- **Git** – Version control
- **dotenv** – Environment variable management

---

## Installation

### Prerequisites

Ensure you have the following installed:
```bash
# Python 3.10 or higher
python --version  # Should return 3.10+

# Redis (Linux/Mac) or Memurai (Windows)
redis-server --version  # or memurai --version

# Conda (recommended for environment management)
conda --version
```

### Step 1: Clone Repository
```bash
git clone https://github.com/yourusername/SourceUp.git
cd SourceUp
```

### Step 2: Create Virtual Environment
```bash
# Using Conda (recommended)
conda create -n sourceup python=3.10
conda activate sourceup

# Or using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Python Dependencies
```bash
pip install -r requirements.txt
```

**Key dependencies:**
```
fastapi>=0.100.0
uvicorn[standard]>=0.23.0
sentence-transformers>=2.2.0
faiss-cpu>=1.7.4
lightgbm>=4.0.0
xgboost>=2.0.0
langchain>=0.1.0
langchain-groq>=0.1.0
redis>=5.0.0
pydantic>=2.0.0
streamlit>=1.28.0
python-dotenv>=1.0.0
```

### Step 4: Configure Environment Variables
```bash
# Copy example environment file
cp .env.example .env

# Edit .env with your API keys
nano .env  # or use any text editor
```

**Required environment variables:**
```env
# LLM Provider Configuration
GROQ_API_KEY=your_groq_api_key_here
AI_PROVIDER=groq  # Options: groq, ollama, openai
INFO_AI_PROVIDER=groq

# Redis Configuration
REDIS_HOST=localhost
REDIS_PORT=6379

# Optional: OpenAI (if using openai provider)
OPENAI_API_KEY=your_openai_key_here
```

### Step 5: Start Redis/Memurai
```bash
# On Linux/Mac
redis-server

# On Windows (Memurai)
# Start Memurai service from Windows Services
# Or run: memurai
```

### Step 6: Run Data Pipeline (First-Time Setup)

This step builds the FAISS index and trains ranking models:
```bash
python pipeline/run_all.py
```

**This will:**
1. Validate and clean supplier data
2. Generate sentence embeddings for all suppliers
3. Build FAISS index for semantic search
4. Train LightGBM and XGBoost ranking models
5. Generate IEEE-compliant evaluation plots

**Expected output:**
```
✅ Loaded 10,247 supplier records
✅ Built FAISS index with 10,247 vectors
✅ Trained LightGBM ranker (NDCG@10: 0.7439)
✅ Trained XGBoost ranker (NDCG@10: 0.7295)
✅ Saved models to backend/app/models/embeddings/
```

### Step 7: Start Backend API
```bash
cd backend
uvicorn app.main:main --reload --port 8000
```

**Backend will be available at:**
- API Base: http://localhost:8000
- Swagger Docs: http://localhost:8000/docs
- Health Check: http://localhost:8000/health

### Step 8: Launch Frontend (New Terminal)
```bash
# Open a new terminal, activate environment
conda activate sourceup

# Start Streamlit interface
cd frontend
streamlit run app.py
```

**Frontend will open at:** http://localhost:8501

---

## Usage

### 1. **Web Interface (Streamlit)**

Navigate to http://localhost:8501 and try:

**Product Search:**
```
Find ISO 9001 certified plastic manufacturers in China under $2
```

**Informational Query:**
```
What is ISO 9001 certification?
```

**What-If Simulation:**
```
What if I prioritize price over delivery speed?
```

### 2. **API Endpoints**

#### **A. Supplier Recommendation**
```bash
curl -X POST "http://localhost:8000/recommend" \
  -H "Content-Type: application/json" \
  -d '{
    "product": "biodegradable food containers",
    "max_price": 2.0,
    "location": "china",
    "certification": "fda",
    "enable_explanations": true,
    "enable_what_if": true
  }'
```

**Response:**
```json
{
  "results": [
    {
      "supplier": "GreenPack Industries",
      "product": "Compostable Food Containers",
      "score": 0.8642,
      "rank": 1,
      "price": "$1.80",
      "location": "Guangdong, China",
      "moq": "1000 pieces",
      "lead_time": "15 days",
      "url": "https://...",
      "reasons": [
        "Within budget ($1.80)",
        "FDA certified",
        "Located in China",
        "Direct manufacturer"
      ],
      "decision_trace": {
        "contributions": {
          "price": {
            "contribution": 0.35,
            "explanation": "Price $1.80 is within budget of $2.00"
          },
          "semantic_match": {
            "contribution": 0.04,
            "explanation": "Product matches query with 0.92 similarity"
          },
          "certification": {
            "contribution": 0.20,
            "explanation": "Has required FDA certification"
          }
        },
        "summary": [
          "✓ Price: +0.350 (Price $1.80 is within budget of $2.00)",
          "✓ Certification: +0.200 (Has required FDA certification)",
          "✓ Location: +0.200 (Exact match: Guangdong, China)"
        ]
      },
      "confidence_score": 0.864
    }
  ],
  "metadata": {
    "total_candidates": 87,
    "after_constraints": 34,
    "filters_applied": ["budget", "certification"],
    "ranking_method": "lightgbm"
  }
}
```

#### **B. Conversational Chat**
```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "user_123",
    "message": "Find ISO certified electronics suppliers in Vietnam"
  }'
```

**Response:**
```json
{
  "message": "I found 12 suppliers for electronics from Vietnam with ISO certification. Here are the top 5 recommendations:",
  "suppliers": [
    {
      "supplier": "VietTech Electronics Co.",
      "product": "Consumer Electronics",
      "score": 0.7821,
      "reasons": ["ISO 9001 certification", "Located in Vietnam", "8+ years verified"]
    }
  ],
  "session_id": "user_123"
}
```

#### **C. What-If Simulation**
```bash
curl -X POST "http://localhost:8000/what-if" \
  -H "Content-Type: application/json" \
  -d '{
    "product": "plastic containers",
    "constraints": {
      "max_price": 1.5,
      "location": "vietnam"
    },
    "scenario": "price_over_speed"
  }'
```

**Response:**
```json
{
  "scenario": "Price Focused",
  "original_weights": {
    "price_match": 0.35,
    "location_match": 0.20
  },
  "new_weights": {
    "price_match": 0.50,
    "location_match": 0.10
  },
  "top_10_changes": [
    {
      "supplier": "BudgetPack Ltd.",
      "original_rank": 8,
      "new_rank": 2,
      "rank_change": 6,
      "explanation": "Lower price became more valuable"
    }
  ],
  "new_top_supplier": "BudgetPack Ltd.",
  "original_top_supplier": "QualityFirst Co."
}
```

### 3. **Python SDK Example**
```python
import requests

# Initialize session
session_id = "user_001"

# First query: Product search
response = requests.post(
    "http://localhost:8000/chat",
    json={
        "session_id": session_id,
        "message": "Find plastic containers from China under $1"
    }
)
results = response.json()

# Follow-up: Refine constraints
response = requests.post(
    "http://localhost:8000/chat",
    json={
        "session_id": session_id,
        "message": "Show only FDA certified suppliers"
    }
)
refined_results = response.json()

# What-if: Explore trade-offs
response = requests.post(
    "http://localhost:8000/what-if",
    json={
        "product": "plastic containers",
        "constraints": {"max_price": 1.0, "certification": "fda"},
        "scenario": "quality_over_cost"
    }
)
simulation = response.json()
```

---


## Acknowledgments

- **Sentence-BERT** by Reimers & Gurevych (2019) for semantic embeddings
- **FAISS** by Facebook AI Research for efficient similarity search
- **LightGBM** by Microsoft Research for gradient boosting framework
- **GlobalSources** for B2B marketplace data access
- **Groq** for high-performance LLM inference

---



**Built with ❤️ for MSMEs worldwide.**

[![Star on GitHub](https://img.shields.io/github/stars/yourusername/SourceUp?style=social)](https://github.com/yourusername/SourceUp)
