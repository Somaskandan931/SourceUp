"""
FastAPI Application Entry Point
--------------------------------
Main backend API server for SourceUp with chat support.
"""

# CRITICAL: Load environment variables FIRST before any other imports
from dotenv import load_dotenv
import os

load_dotenv()

# Verify environment variables are loaded
print("=" * 60)
print("🔧 Environment Check:")
print(f"✅ GROQ_API_KEY loaded: {bool(os.getenv('GROQ_API_KEY'))}")
if os.getenv('GROQ_API_KEY'):
    key = os.getenv('GROQ_API_KEY')
    print(f"✅ Key starts with: {key[:10]}...")
    print(f"✅ Key length: {len(key)}")
else:
    print("❌ GROQ_API_KEY not found!")
    print("❌ Make sure you have a .env file with GROQ_API_KEY=your_key")
print("=" * 60)

# Now import everything else
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.app.api.recommend import router as recommend_router
from backend.app.api.chat import router as chat_router

app = FastAPI(
    title="SourceUP API",
    description="Supplier recommendation and search API with chat support",
    version="1.0.0"
)

# Enable CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(recommend_router, tags=["recommendations"])
app.include_router(chat_router, tags=["chat"])


@app.get("/")
def root():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "SourceUP API",
        "version": "1.0.0",
        "endpoints": {
            "recommend": "/recommend",
            "chat": "/chat",
            "docs": "/docs"
        }
    }


@app.get("/health")
def health_check():
    """Detailed health check."""
    try:
        from backend.app.models.retriever import load_index
        index, meta = load_index()

        return {
            "status": "healthy",
            "database": "connected",
            "faiss_index": "loaded",
            "total_suppliers": len(meta),
            "groq_configured": bool(os.getenv("GROQ_API_KEY"))
        }
    except Exception as e:
        return {
            "status": "degraded",
            "error": str(e),
            "groq_configured": bool(os.getenv("GROQ_API_KEY"))
        }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)