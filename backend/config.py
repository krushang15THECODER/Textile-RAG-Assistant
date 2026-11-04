import os
from typing import List, Optional
from dotenv import load_dotenv

# Load .env once at module initialization
load_dotenv(override=True)


def get_mongodb_uri() -> str:
    val = os.getenv("MONGODB_URI", "").strip()
    if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
        val = val[1:-1].strip()
    return val


def get_mongodb_db_name() -> str:
    return os.getenv("MONGODB_DB_NAME", "labhtex_textile_rag").strip()


def get_admin_registration_key() -> str:
    """Read the admin registration key from the environment.
    
    Returns empty string if not configured (never falls back to a hardcoded secret).
    """
    return os.getenv("ADMIN_REGISTRATION_KEY", "").strip()


def validate_config() -> None:
    """Fail application startup if essential security configuration is missing.
    
    Must be called during application startup (e.g., FastAPI lifespan).
    """
    admin_key = get_admin_registration_key()
    if not admin_key:
        raise RuntimeError(
            "Configuration Error: ADMIN_REGISTRATION_KEY environment variable is missing or empty. "
            "The application cannot start without a secure, configured admin registration key. "
            "Please define ADMIN_REGISTRATION_KEY in your .env file or host environment."
        )


def get_backend_url() -> str:
    # 1. Check streamlit secrets if running under Streamlit
    try:
        import streamlit as st
        if hasattr(st, "secrets") and "BACKEND_URL" in st.secrets:
            val = str(st.secrets["BACKEND_URL"]).strip().rstrip("/")
            if val:
                return val
    except Exception:
        pass
    # 2. Check environment variable
    return os.getenv("BACKEND_URL", "http://127.0.0.1:8000").strip().rstrip("/")


def get_cookie_secure() -> bool:
    """Parse COOKIE_SECURE environment setting.
    
    Rules:
    - If COOKIE_SECURE is set: strictly parses ('true', '1', 'yes') as True,
      ('false', '0', 'no') as False. Raises ValueError on invalid values.
    - If COOKIE_SECURE is unset: defaults to True if BACKEND_URL uses https://
      (production / cloud deployments), otherwise False for local HTTP development.
    """
    raw = os.getenv("COOKIE_SECURE")
    if raw is not None:
        clean = raw.strip().lower()
        if clean in ("true", "1", "yes"):
            return True
        elif clean in ("false", "0", "no"):
            return False
        else:
            raise ValueError(
                f"Invalid COOKIE_SECURE setting: '{raw}'. Expected 'true', 'false', '1', or '0'."
            )
    backend_url = get_backend_url()
    return backend_url.startswith("https://")


def get_retrieval_score_threshold() -> float:
    """Read the minimum cosine similarity threshold for vector retrieval.
    
    Default is 0.40 based on empirical benchmarking on the textile dataset.
    """
    raw = os.getenv("RETRIEVAL_SCORE_THRESHOLD", "0.40").strip()
    try:
        val = float(raw)
        return max(0.0, min(1.0, val))
    except ValueError:
        return 0.40


# Exported constants for compatibility
MONGODB_URI = get_mongodb_uri()
MONGODB_DB_NAME = get_mongodb_db_name()
ADMIN_REGISTRATION_KEY = get_admin_registration_key()

SESSION_COOKIE_NAME = os.getenv("SESSION_COOKIE_NAME", "labhtex_session").strip()
BACKEND_URL = get_backend_url()
COOKIE_SECURE = get_cookie_secure()
RETRIEVAL_SCORE_THRESHOLD = get_retrieval_score_threshold()

SESSION_IDLE_TIMEOUT_MINUTES = int(os.getenv("SESSION_IDLE_TIMEOUT_MINUTES", "30"))
SESSION_ABSOLUTE_LIFETIME_HOURS = int(os.getenv("SESSION_ABSOLUTE_LIFETIME_HOURS", "8"))
MIN_PASSWORD_LENGTH = 12

raw_cors = os.getenv("CORS_ORIGINS", "http://localhost:8501,http://127.0.0.1:8501,http://localhost:8000")
CORS_ORIGINS: List[str] = [origin.strip() for origin in raw_cors.split(",") if origin.strip()]
