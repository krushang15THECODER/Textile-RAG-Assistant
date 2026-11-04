import os

# Set thread limits BEFORE importing torch/numpy/OpenMP to cap memory within 512MB on Render/cloud
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.config import CORS_ORIGINS, validate_config
from backend.database import DatabaseUnavailableError, get_db, init_indexes
from backend.routes import admin, auth, conversations, messages


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. Validate security configuration on startup
    validate_config()

    # 2. Initialize indexes if MongoDB is available
    try:
        db = get_db()
        init_indexes(db)
        print("MongoDB indexes successfully initialized.")
    except DatabaseUnavailableError as db_err:
        print(f"Warning: Database unavailable during startup: {db_err}")
    except Exception as exc:
        print(f"Warning: Index initialization note: {exc}")
    yield


app = FastAPI(
    title="LabhTex Textile Intelligence API",
    description="Backend API for Authentication, Authorization, Persistent Chat, and Knowledge Base Management",
    version="2.5.0",
    lifespan=lifespan
)

# CORS Middleware with explicit origins and Streamlit Cloud regex support
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS or ["http://localhost:8501", "http://127.0.0.1:8501"],
    allow_origin_regex=r"^https://.*\.streamlit\.app$",
    allow_credentials=True,
    allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS", "HEAD"],
    allow_headers=["*"],
)


@app.exception_handler(DatabaseUnavailableError)
async def database_unavailable_handler(request: Request, exc: DatabaseUnavailableError):
    # Log technical detail for backend diagnostic logs
    print(f"[ERROR] Database unavailable: {exc}")
    clean_msg = "Database service is temporarily unavailable. Please verify connectivity or try again shortly."
    if "SSL handshake failed" in str(exc) or "TLSV1_ALERT" in str(exc):
        clean_msg = "Database connection failed (SSL handshake). Please ensure '0.0.0.0/0' is whitelisted in MongoDB Atlas Network Access."
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={"detail": clean_msg, "diagnostic": str(exc)}
    )


# Include Routers
app.include_router(auth.router)
app.include_router(conversations.router)
app.include_router(messages.router)
app.include_router(admin.router)


@app.api_route("/", methods=["GET", "HEAD"])
async def root():
    """Root endpoint welcoming visitors and providing navigation to documentation."""
    return {
        "status": "online",
        "service": "LabhTex Textile Intelligence Backend",
        "version": "2.5.0",
        "documentation": "/docs",
        "health": "/health",
        "live": "/live"
    }


@app.api_route("/live", methods=["GET", "HEAD"])
async def liveness_probe():
    """Liveness probe: returns 200 OK immediately if the web process is running."""
    return {"status": "alive"}


@app.api_route("/health", methods=["GET", "HEAD"])
async def health_check():
    """Readiness probe: validates database connectivity, ONNX runtime model, and runtime state."""
    db_connected = False
    db_status = "connected"
    try:
        db = get_db()
        db.command("ping")
        db_connected = True
    except Exception as e:
        db_status = f"unavailable: {e}"

    import sys
    import ssl
    from pathlib import Path
    onnx_present = (Path("models") / "all-MiniLM-L6-v2" / "model.onnx").exists()
    is_ready = db_connected and onnx_present

    return {
        "status": "online" if is_ready else "degraded",
        "service": "LabhTex Textile Intelligence Backend",
        "version": "2.5.0",
        "ready": is_ready,
        "onnx_ready": onnx_present,
        "python": sys.version.split()[0],
        "openssl": ssl.OPENSSL_VERSION,
        "database": db_status
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)
