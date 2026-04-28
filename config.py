"""
SourceUp — Centralised Configuration
--------------------------------------
ALL file paths and environment settings live here.
No hardcoded paths anywhere else in the codebase.

Usage:
    from config import cfg

    cfg.CLEAN_DATA        # path to suppliers_clean.csv
    cfg.FAISS_INDEX       # path to suppliers.faiss
    cfg.LGBM_MODEL        # path to ranker_lightgbm.pkl
    cfg.GROQ_API_KEY      # Groq API key from .env

The SOURCEUP_DIR environment variable controls the project root.
It defaults to the directory containing this file (config.py),
making the project fully portable across Windows / Linux / Mac / Docker.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


class _Config:
    # ── Project root ─────────────────────────────────────────────────
    # Set SOURCEUP_DIR in your shell or .env to override.
    # If not set, the root is the directory that contains config.py.
    ROOT: Path = Path(
        os.getenv("SOURCEUP_DIR", str(Path(__file__).resolve().parent))
    )

    # ── Data directories ─────────────────────────────────────────────
    @property
    def DATA_DIR(self)        -> Path: return self.ROOT / "data"

    @property
    def OUTPUTS_DIR(self)     -> Path: return self.DATA_DIR / "outputs"

    @property
    def MERGED_DIR(self)      -> Path: return self.DATA_DIR / "merged"

    @property
    def CLEAN_DIR(self)       -> Path: return self.DATA_DIR / "clean"

    @property
    def EMBEDDINGS_DIR(self)  -> Path: return self.DATA_DIR / "embeddings"

    @property
    def TRAINING_DIR(self)    -> Path: return self.DATA_DIR / "training"

    @property
    def EVAL_DIR(self)        -> Path: return self.DATA_DIR / "eval"

    @property
    def EVAL_PLOTS_DIR(self)  -> Path: return self.EVAL_DIR / "plots"

    # ── Key files ────────────────────────────────────────────────────
    @property
    def CLEAN_DATA(self)      -> Path: return self.CLEAN_DIR    / "suppliers_clean.csv"

    @property
    def MERGED_DATA(self)     -> Path: return self.MERGED_DIR   / "suppliers_all.csv"

    @property
    def SCHEMA_FILE(self)     -> Path: return self.DATA_DIR     / "test_output.csv"

    @property
    def QUERY_FILE(self)      -> Path: return self.DATA_DIR     / "search_query.csv"

    @property
    def TRAINING_DATA(self)   -> Path: return self.TRAINING_DIR / "ranking_data.csv"

    @property
    def FAISS_INDEX(self)     -> Path: return self.EMBEDDINGS_DIR / "suppliers.faiss"

    @property
    def FAISS_META(self)      -> Path: return self.EMBEDDINGS_DIR / "suppliers_meta.csv"

    # ── ML model files ───────────────────────────────────────────────
    @property
    def MODELS_DIR(self)      -> Path:
        return self.ROOT / "backend" / "app" / "models" / "embeddings"

    @property
    def LGBM_MODEL(self)      -> Path: return self.MODELS_DIR / "ranker_lightgbm.pkl"

    @property
    def XGB_MODEL(self)       -> Path: return self.MODELS_DIR / "ranker_xgboost.pkl"

    # ── Scraper ──────────────────────────────────────────────────────
    @property
    def JAR_PATH(self)        -> Path: return self.ROOT / "somasjar.jar"

    @property
    def SESSIONS_DIR(self)    -> Path: return self.ROOT / "sessions"

    # ── Environment / secrets ────────────────────────────────────────
    GROQ_API_KEY:   str = os.getenv("GROQ_API_KEY", "")
    SECRET_KEY:     str = os.getenv("SECRET_KEY", "change-me-generate-with-openssl-rand-hex-32")
    UPI_ID:         str = os.getenv("UPI_ID", "")
    AI_PROVIDER:    str = os.getenv("AI_PROVIDER", "groq")
    REDIS_HOST:     str = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT:     int = int(os.getenv("REDIS_PORT", "6379"))

    def ensure_dirs(self):
        """Create all output directories (idempotent)."""
        for d in [
            self.OUTPUTS_DIR, self.MERGED_DIR, self.CLEAN_DIR,
            self.EMBEDDINGS_DIR, self.TRAINING_DIR,
            self.EVAL_DIR, self.EVAL_PLOTS_DIR,
            self.MODELS_DIR, self.SESSIONS_DIR,
        ]:
            d.mkdir(parents=True, exist_ok=True)

    def validate(self) -> list:
        """Return list of warnings about missing config."""
        warnings = []
        if not self.GROQ_API_KEY:
            warnings.append("GROQ_API_KEY not set — LLM features will be disabled")
        if not self.UPI_ID:
            warnings.append("UPI_ID not set — billing will be disabled")
        if not self.FAISS_INDEX.exists():
            warnings.append(f"FAISS index missing: {self.FAISS_INDEX} — run pipeline/run_all.py")
        if not self.CLEAN_DATA.exists():
            warnings.append(f"Clean data missing: {self.CLEAN_DATA} — run pipeline/run_all.py")
        return warnings

    def __repr__(self):
        return f"<SourceUpConfig root={self.ROOT}>"


cfg = _Config()
