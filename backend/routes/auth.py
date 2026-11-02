from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from bson import ObjectId

from backend.config import SESSION_COOKIE_NAME, SESSION_ABSOLUTE_LIFETIME_HOURS
from backend.database import get_db
from backend.models import LoginResponse, UserLoginRequest, UserRegisterRequest, UserResponse
from backend.security import (
    check_login_rate_limit,
    create_user_session,
    get_current_user_and_session,
    hash_password,
    hash_session_token,
    record_failed_login,
    reset_login_rate_limit,
    verify_admin_key,
    verify_password,
)

router = APIRouter(prefix="/api/auth", tags=["Authentication"])


def serialize_user(user: dict) -> UserResponse:
    """Format MongoDB user document into a safe Pydantic response."""
    return UserResponse(
        id=str(user["_id"]),
        name=user.get("name", "User"),
        email=user["email"],
        role=user.get("role", "user"),
        is_active=user.get("is_active", True),
        created_at=user.get("created_at", datetime.now(timezone.utc)),
    )


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
@router.post("/register/", response_model=UserResponse, status_code=status.HTTP_201_CREATED, include_in_schema=False)
async def register(req: UserRegisterRequest):
    """Register a new user account with bcrypt password hashing."""
    db = get_db()
    email_clean = req.email.lower().strip()

    # Check for existing account
    existing = db.users.find_one({"email": email_clean})
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="An account with this email address already exists."
        )

    # Determine role based on admin key check
    role = "user"
    if req.admin_key:
        if verify_admin_key(req.admin_key):
            role = "admin"
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid administrator registration key."
            )

    now = datetime.now(timezone.utc)
    hashed_pwd = hash_password(req.password)

    new_user = {
        "name": req.name.strip(),
        "email": email_clean,
        "password_hash": hashed_pwd,
        "role": role,
        "is_active": True,
        "created_at": now,
        "updated_at": now
    }

    result = db.users.insert_one(new_user)
    new_user["_id"] = result.inserted_id

    return serialize_user(new_user)


@router.post("/login", response_model=LoginResponse)
@router.post("/login/", response_model=LoginResponse, include_in_schema=False)
async def login(req: UserLoginRequest, request: Request, response: Response):
    """Authenticate user, create session in MongoDB, and set HttpOnly session cookie."""
    client_ip = request.client.host if request.client else "unknown"
    check_login_rate_limit(client_ip)

    db = get_db()
    email_clean = req.email.lower().strip()

    user = db.users.find_one({"email": email_clean})
    if not user or not verify_password(req.password, user.get("password_hash", "")):
        record_failed_login(client_ip)
        # Generic error message to prevent account enumeration
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password."
        )

    if not user.get("is_active", True):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Your account has been deactivated. Please contact an administrator."
        )

    reset_login_rate_limit(client_ip)

    raw_token, expires_at = create_user_session(db, user["_id"])

    # Set secure HttpOnly session cookie
    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=raw_token,
        max_age=SESSION_ABSOLUTE_LIFETIME_HOURS * 3600,
        httponly=True,
        secure=False,  # Set to True behind HTTPS reverse-proxy / Render
        samesite="lax",
        path="/"
    )

    return LoginResponse(
        user=serialize_user(user),
        token=raw_token,
        expires_at=expires_at
    )


@router.get("/login", include_in_schema=False)
@router.get("/login/", include_in_schema=False)
async def get_login_not_allowed():
    raise HTTPException(
        status_code=status.HTTP_405_METHOD_NOT_ALLOWED,
        detail="Login endpoint only accepts POST requests. If you are accessing this from a client, ensure BACKEND_URL uses 'https://' rather than 'http://' to avoid redirect method stripping."
    )


@router.post("/logout")
async def logout(response: Response, auth_data=Depends(get_current_user_and_session)):
    """Revoke active session in MongoDB and clear session cookie."""
    _, session_doc = auth_data
    db = get_db()
    now = datetime.now(timezone.utc)

    db.sessions.update_one(
        {"_id": session_doc["_id"]},
        {"$set": {"revoked_at": now}}
    )

    response.delete_cookie(
        key=SESSION_COOKIE_NAME,
        path="/"
    )

    return {"message": "Successfully logged out and session revoked."}


@router.get("/me", response_model=UserResponse)
async def get_me(auth_data=Depends(get_current_user_and_session)):
    """Return currently authenticated user profile."""
    user, _ = auth_data
    return serialize_user(user)
