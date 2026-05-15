"""
FastAPI Application Entry Point — SourceUp
-------------------------------------------
Registers all routers: recommend, chat, quote, auth.
Uses centralised config — NO hardcoded paths anywhere.
"""

import sys
import os
from contextlib import asynccontextmanager

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)
))))

from dotenv import load_dotenv
load_dotenv()

from config import cfg

# Ensure directories exist BEFORE setting up logging
cfg.ensure_dirs()

from backend.app.utils.logging_config import setup_logging

# Setup logging after directories are created
log_file_path = cfg.LOGS_DIR / "app.log"
setup_logging(level="INFO", log_file=str(log_file_path))

from backend.app.utils.logging_config import get_logger

logger = get_logger(__name__)

# Print startup banner
print("=" * 60)
print("🚀 SourceUp API starting up")
print(f"   Root      : {cfg.ROOT}")
print(f"   Data Dir  : {cfg.DATA_DIR}")
print(f"   Logs Dir  : {cfg.LOGS_DIR}")
print(f"   Groq      : {'✅ configured' if cfg.GROQ_API_KEY else '❌ NOT SET'}")
print(f"   UPI       : {'✅ configured' if cfg.UPI_ID else '⚠️ not set (billing disabled)'}")
print(f"   MongoDB   : {os.getenv('MONGODB_URI', 'mongodb://localhost:27017')}")
for warn in cfg.validate():
    print(f"   ⚠️  {warn}")
print("=" * 60)

# Display file status
print(f"\n📁 File Status:")
print(f"  Clean Data:      {'✅' if cfg.CLEAN_DATA.exists() else '❌'} {cfg.CLEAN_DATA}")
print(f"  FAISS Index:     {'✅' if cfg.FAISS_INDEX.exists() else '❌'} {cfg.FAISS_INDEX}")
print(f"  FAISS Meta:      {'✅' if cfg.FAISS_META.exists() else '❌'} {cfg.FAISS_META}")
print(f"  LGBM Model:      {'✅' if cfg.LGBM_MODEL.exists() else '❌'} {cfg.LGBM_MODEL}")
print("=" * 60 + "\n")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api.recommend import router as recommend_router
from backend.app.api.chat import router as chat_router
from backend.app.api.quote import router as quote_router
from backend.app.api.auth import router as auth_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    # Startup
    logger.info("Starting up SourceUp API...")

    # Initialize MongoDB
    try:
        from backend.app.database.mongodb import MongoDB
        mongodb_uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
        mongodb_db = os.getenv("MONGODB_DB", "sourceup")
        await MongoDB.connect(mongodb_uri, mongodb_db)
        logger.info("✅ MongoDB connected successfully")

        # Demo user info
        demo_email = os.getenv("DEMO_EMAIL", "demo@sourceup.com")
        demo_plan = os.getenv("DEMO_PLAN", "pro")
        logger.info(f"✅ Demo user available at: /auth/demo-login (plan: {demo_plan})")

    except Exception as e:
        logger.warning(f"⚠️ MongoDB connection failed: {e}")
        logger.warning("   Continuing without database - auth will use in-memory fallback")

    # Validate critical services
    warnings = []

    if not cfg.GROQ_API_KEY:
        warnings.append("GROQ_API_KEY not set - chat & quote features will be limited")

    if not cfg.UPI_ID:
        warnings.append("UPI_ID not set - billing will be disabled")

    # Check FAISS index
    try:
        from backend.app.models.retriever import load_index
        index, meta = load_index()
        logger.info(f"✅ Loaded {len(meta)} suppliers into FAISS index")
    except Exception as e:
        warnings.append(f"FAISS index load failed: {e}")

    for w in warnings:
        logger.warning(w)

    yield

    # Shutdown
    try:
        from backend.app.database.mongodb import MongoDB
        await MongoDB.close()
        logger.info("✅ MongoDB connection closed")
    except:
        pass

    logger.info("Shutting down SourceUp API...")


app = FastAPI(
    title="SourceUp API",
    description="Constraint-aware explainable supplier recommendation. Includes chat, quote drafting, and billing.",
    version="2.0.0",
    lifespan=lifespan,
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3001",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:8080",
        "http://127.0.0.1:8080",
    ],
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
        "service": "SourceUp API",
        "version": "2.0.0",
        "endpoints": {
            "docs": "/docs",
            "recommend": "/recommend",
            "chat": "/chat",
            "quote": "/quote/draft",
            "auth": "/auth/login",
            "demo": "/auth/demo-login",
            "billing": "/auth/billing/plans",
            "health": "/health",
        },
    }


@app.get("/health")
def health():
    status = {
        "service": "SourceUp API",
        "version": "2.0.0",
        "status": "healthy"
    }

    try:
        from backend.app.models.retriever import load_index
        index, meta = load_index()
        status["faiss"] = f"ok ({len(meta)} suppliers)"
        status["database"] = "connected"
    except Exception as e:
        status["faiss"] = f"error: {e}"
        status["database"] = "unavailable"
        status["status"] = "degraded"

    status["groq"] = "configured" if cfg.GROQ_API_KEY else "not configured"
    status["upi"] = "configured" if cfg.UPI_ID else "not configured"

    # MongoDB status
    try:
        from backend.app.database.mongodb import MongoDB
        status["mongodb"] = "connected" if MongoDB.db is not None else "disconnected"
    except:
        status["mongodb"] = "not configured"

    return status


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app.main:app", host="0.0.0.0", port=8000, reload=True)
