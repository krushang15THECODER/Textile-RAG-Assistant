from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.config import CORS_ORIGINS
from backend.database import DatabaseUnavailableError, get_db, init_indexes
from backend.routes import admin, auth, conversations, messages


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize indexes if MongoDB is available
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
    version="2.0.0",
    lifespan=lifespan
)

# Strict CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS or ["http://localhost:8501"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)


@app.exception_handler(DatabaseUnavailableError)
async def database_unavailable_handler(request: Request, exc: DatabaseUnavailableError):
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={"detail": str(exc)}
    )


# Include Routers
app.include_router(auth.router)
app.include_router(conversations.router)
app.include_router(messages.router)
app.include_router(admin.router)


@app.get("/health")
async def health_check():
    """Health check endpoint validating service and database connectivity."""
    db_status = "connected"
    try:
        db = get_db()
        db.command("ping")
    except Exception as e:
        db_status = f"unavailable: {e}"

    return {
        "status": "online",
        "service": "LabhTex Textile Intelligence Backend",
        "database": db_status
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)
