"""
FastAPI Application Entry Point — SourceUp
-------------------------------------------
Registers all routers: recommend, chat, quote, auth.
Uses centralised config — NO hardcoded paths anywhere.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)
))))

from dotenv import load_dotenv
load_dotenv()

from config import cfg

# Print startup banner
print("=" * 60)
print("🚀 SourceUp API starting up")
print(f"   Root      : {cfg.ROOT}")
print(f"   Groq      : {'✅ configured' if cfg.GROQ_API_KEY else '❌ NOT SET'}")
print(f"   UPI       : {'✅ configured' if cfg.UPI_ID else '⚠️  not set (billing disabled)'}")
for warn in cfg.validate():
    print(f"   ⚠️  {warn}")
print("=" * 60)

cfg.ensure_dirs()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api.recommend import router as recommend_router
from backend.app.api.chat      import router as chat_router
from backend.app.api.quote     import router as quote_router
from backend.app.api.auth      import router as auth_router

app = FastAPI(
    title="SourceUp API",
    description=(
        "Constraint-aware explainable supplier recommendation. "
        "Includes chat, quote drafting, and billing."
    ),
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # tighten in production: ["https://yourdomain.com"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(recommend_router)
app.include_router(chat_router)
app.include_router(quote_router)
app.include_router(auth_router)


@app.get("/")
def root():
    return {
        "service":   "SourceUp API",
        "version":   "2.0.0",
        "endpoints": {
            "docs":     "/docs",
            "recommend":"/recommend",
            "chat":     "/chat",
            "quote":    "/quote/draft",
            "auth":     "/auth/login",
            "billing":  "/auth/billing/plans",
            "health":   "/health",
        },
    }


@app.get("/health")
def health():
    status = {"service": "SourceUp API", "version": "2.0.0"}
    try:
        from backend.app.models.retriever import load_index
        index, meta = load_index()
        status["faiss"]    = f"ok ({len(meta)} suppliers)"
        status["database"] = "connected"
    except Exception as e:
        status["faiss"]    = f"error: {e}"
        status["database"] = "unavailable"
    status["groq"] = "configured" if cfg.GROQ_API_KEY else "not configured"
    status["upi"]  = "configured" if cfg.UPI_ID else "not configured"
    return status


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app.main:app", host="0.0.0.0", port=8000, reload=True)
