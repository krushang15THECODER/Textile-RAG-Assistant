import os
from typing import List
from dotenv import load_dotenv

load_dotenv(override=True)

def get_mongodb_uri() -> str:
    load_dotenv(override=True)
    val = os.getenv("MONGODB_URI", "").strip()
    if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
        val = val[1:-1].strip()
    return val

def get_mongodb_db_name() -> str:
    load_dotenv(override=True)
    return os.getenv("MONGODB_DB_NAME", "labhtex_textile_rag").strip()

def get_admin_registration_key() -> str:
    load_dotenv(override=True)
    return os.getenv("ADMIN_REGISTRATION_KEY", "LabhTexAdminSecret2026!").strip()

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
    load_dotenv(override=True)
    return os.getenv("BACKEND_URL", "http://127.0.0.1:8000").strip().rstrip("/")

# Properties for backwards compatibility
MONGODB_URI = get_mongodb_uri()
MONGODB_DB_NAME = get_mongodb_db_name()
ADMIN_REGISTRATION_KEY = get_admin_registration_key()

SESSION_COOKIE_NAME = os.getenv("SESSION_COOKIE_NAME", "labhtex_session").strip()
BACKEND_URL = get_backend_url()

SESSION_IDLE_TIMEOUT_MINUTES = int(os.getenv("SESSION_IDLE_TIMEOUT_MINUTES", "30"))
SESSION_ABSOLUTE_LIFETIME_HOURS = int(os.getenv("SESSION_ABSOLUTE_LIFETIME_HOURS", "8"))
MIN_PASSWORD_LENGTH = 12

raw_cors = os.getenv("CORS_ORIGINS", "http://localhost:8501,http://127.0.0.1:8501,http://localhost:8000")
CORS_ORIGINS: List[str] = [origin.strip() for origin in raw_cors.split(",") if origin.strip()]
