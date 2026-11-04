import hashlib
import secrets
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional, Tuple
import bcrypt
from fastapi import Cookie, Depends, Header, HTTPException, Request, status
from bson import ObjectId

from backend.config import (
    ADMIN_REGISTRATION_KEY,
    SESSION_ABSOLUTE_LIFETIME_HOURS,
    SESSION_COOKIE_NAME,
    SESSION_IDLE_TIMEOUT_MINUTES,
    get_admin_registration_key,
)
from backend.database import get_db

# In-memory sliding window rate limiter for login attempts (IP -> list of timestamps)
_login_attempts: Dict[str, list] = {}
MAX_LOGIN_ATTEMPTS = 5
RATE_LIMIT_WINDOW_SECONDS = 300  # 5 minutes lock out after 5 failures


def hash_password(password: str) -> str:
    """Hash a plaintext password using bcrypt with a secure salt."""
    salt = bcrypt.gensalt(rounds=12)
    hashed = bcrypt.hashpw(password.encode("utf-8"), salt)
    return hashed.decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password against bcrypt hash safely."""
    try:
        return bcrypt.checkpw(plain_password.encode("utf-8"), hashed_password.encode("utf-8"))
    except Exception:
        return False


def generate_session_token() -> str:
    """Generate a high-entropy cryptographically secure random session token."""
    return secrets.token_urlsafe(32)


def hash_session_token(token: str) -> str:
    """Return SHA-256 hash of session token to prevent database token leakage."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def verify_admin_key(provided_key: Optional[str]) -> bool:
    """Verify administrator registration key in constant time."""
    if not provided_key or not provided_key.strip():
        return False
    # Check monkeypatched ADMIN_REGISTRATION_KEY in module, fallback to dynamic get_admin_registration_key()
    configured_key = ADMIN_REGISTRATION_KEY or get_admin_registration_key()
    if not configured_key or not configured_key.strip():
        return False
    return secrets.compare_digest(provided_key.strip(), configured_key.strip())


def check_login_rate_limit(client_ip: str) -> None:
    """Enforce rate limits on failed login attempts to prevent brute-force attacks."""
    now = time.time()
    attempts = _login_attempts.get(client_ip, [])
    # Filter attempts within current sliding window
    attempts = [t for t in attempts if now - t < RATE_LIMIT_WINDOW_SECONDS]
    _login_attempts[client_ip] = attempts

    if len(attempts) >= MAX_LOGIN_ATTEMPTS:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many failed login attempts. Please wait 5 minutes before trying again."
        )


def record_failed_login(client_ip: str) -> None:
    """Record a failed login attempt for the given IP address."""
    now = time.time()
    attempts = _login_attempts.get(client_ip, [])
    attempts.append(now)
    _login_attempts[client_ip] = attempts


def reset_login_rate_limit(client_ip: str) -> None:
    """Clear failed attempts on successful login."""
    _login_attempts.pop(client_ip, None)


def create_user_session(db, user_id: str) -> Tuple[str, datetime]:
    """Create a new session in MongoDB using token SHA-256 hash."""
    now = datetime.now(timezone.utc)
    raw_token = generate_session_token()
    token_hash = hash_session_token(raw_token)
    expires_at = now + timedelta(hours=SESSION_ABSOLUTE_LIFETIME_HOURS)

    session_doc = {
        "user_id": str(user_id),
        "token_hash": token_hash,
        "created_at": now,
        "last_activity": now,
        "expires_at": expires_at,
        "revoked_at": None
    }
    db.sessions.insert_one(session_doc)
    return raw_token, expires_at


def extract_token_from_request(
    authorization: Optional[str] = Header(None),
    cookie_token: Optional[str] = Cookie(None, alias=SESSION_COOKIE_NAME)
) -> Optional[str]:
    """Extract session token from either Authorization header or HttpOnly cookie."""
    if authorization and authorization.startswith("Bearer "):
        return authorization.split(" ", 1)[1].strip()
    if cookie_token:
        return cookie_token.strip()
    return None


async def get_current_user_and_session(
    token: Optional[str] = Depends(extract_token_from_request)
) -> Tuple[dict, dict]:
    """Validate session token against MongoDB and return active (user, session)."""
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required. No session token provided."
        )

    db = get_db()
    token_hash = hash_session_token(token)
    session_doc = db.sessions.find_one({"token_hash": token_hash})

    if not session_doc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid session token."
        )

    if session_doc.get("revoked_at") is not None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Session has been revoked. Please log in again."
        )

    now = datetime.now(timezone.utc)

    # Convert session dates to timezone-aware if needed
    expires_at = session_doc["expires_at"]
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=timezone.utc)

    last_activity = session_doc["last_activity"]
    if last_activity.tzinfo is None:
        last_activity = last_activity.replace(tzinfo=timezone.utc)

    # 1. Absolute session lifetime check (8 hours)
    if now > expires_at:
        db.sessions.update_one({"_id": session_doc["_id"]}, {"$set": {"revoked_at": now}})
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Session has expired. Maximum session lifetime reached."
        )

    # 2. Idle timeout check (30 minutes)
    if now > last_activity + timedelta(minutes=SESSION_IDLE_TIMEOUT_MINUTES):
        db.sessions.update_one({"_id": session_doc["_id"]}, {"$set": {"revoked_at": now}})
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Session timed out due to 30 minutes of inactivity."
        )

    # Update last_activity sliding window
    db.sessions.update_one({"_id": session_doc["_id"]}, {"$set": {"last_activity": now}})

    # Fetch user
    user_id = session_doc["user_id"]
    try:
        user_query = {"_id": ObjectId(user_id)}
    except Exception:
        user_query = {"_id": user_id}

    user = db.users.find_one(user_query)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User account no longer exists."
        )

    if not user.get("is_active", True):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is deactivated."
        )

    return user, session_doc


async def get_current_user(
    auth_data: Tuple[dict, dict] = Depends(get_current_user_and_session)
) -> dict:
    """FastAPI dependency to retrieve currently authenticated user."""
    user, _ = auth_data
    return user


async def require_admin(
    user: dict = Depends(get_current_user)
) -> dict:
    """FastAPI dependency ensuring the authenticated user has the 'admin' role."""
    if user.get("role") != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Administrative privileges required to access this resource."
        )
    return user
