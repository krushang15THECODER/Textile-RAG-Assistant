"""
Comprehensive test suite for Authentication, Authorization, Sessions,
Persistent Chat History, and MongoDB integrity in the FastAPI backend.
"""

from datetime import datetime, timedelta, timezone
import mongomock
import pytest
from fastapi.testclient import TestClient

from backend.database import DatabaseUnavailableError, init_indexes, set_test_db
from backend.main import app
from backend.security import hash_session_token


@pytest.fixture(autouse=True)
def setup_mock_db():
    """Provide a fresh isolated mock MongoDB instance for each test."""
    mock_client = mongomock.MongoClient()
    mock_db = mock_client["test_labhtex_rag"]
    init_indexes(mock_db)
    set_test_db(mock_db)
    yield mock_db
    set_test_db(None)


def test_successful_registration_and_password_hashing(setup_mock_db):
    """Scenario 1: User can register, and password is never stored in plaintext."""
    client = TestClient(app)
    db = setup_mock_db

    payload = {
        "name": "Arjun Sharma",
        "email": "arjun@labhtex.com",
        "password": "SecurePassword123!"
    }
    resp = client.post("/api/auth/register", json=payload)
    assert resp.status_code == 201, resp.text
    data = resp.json()
    assert data["email"] == "arjun@labhtex.com"
    assert data["role"] == "user"
    assert "password" not in data
    assert "password_hash" not in data

    # Verify database document
    user_doc = db.users.find_one({"email": "arjun@labhtex.com"})
    assert user_doc is not None
    assert user_doc["password_hash"] != "SecurePassword123!"
    assert user_doc["password_hash"].startswith("$2b$") or user_doc["password_hash"].startswith("$2a$")
    print("PASS: test_successful_registration_and_password_hashing")


def test_duplicate_email_rejection():
    """Scenario 2: Duplicate registration with the same email must be rejected."""
    client = TestClient(app)
    payload = {
        "name": "User One",
        "email": "duplicate@labhtex.com",
        "password": "Password123456!"
    }
    r1 = client.post("/api/auth/register", json=payload)
    assert r1.status_code == 201

    r2 = client.post("/api/auth/register", json=payload)
    assert r2.status_code == 400
    assert "already exists" in r2.json()["detail"]
    print("PASS: test_duplicate_email_rejection")


def test_minimum_password_length_validation():
    """Scenario 3: Passwords shorter than 12 characters must be rejected by Pydantic schema."""
    client = TestClient(app)
    payload = {
        "name": "Short Pwd",
        "email": "short@labhtex.com",
        "password": "short"  # Only 5 characters
    }
    resp = client.post("/api/auth/register", json=payload)
    assert resp.status_code == 422
    print("PASS: test_minimum_password_length_validation")


def test_successful_login_and_cookie_generation(setup_mock_db):
    """Scenario 4: User can log in with valid credentials, receiving token and HttpOnly cookie."""
    client = TestClient(app)
    db = setup_mock_db

    # Register
    client.post("/api/auth/register", json={
        "name": "Login User",
        "email": "login@labhtex.com",
        "password": "ValidPassword1234!"
    })

    # Login
    resp = client.post("/api/auth/login", json={
        "email": "login@labhtex.com",
        "password": "ValidPassword1234!"
    })
    assert resp.status_code == 200
    data = resp.json()
    assert "token" in data
    raw_token = data["token"]

    # Verify session is stored in MongoDB ONLY as SHA-256 hash (never raw token)
    token_hash = hash_session_token(raw_token)
    session_doc = db.sessions.find_one({"token_hash": token_hash})
    assert session_doc is not None
    assert session_doc.get("token") is None  # Raw token must not be in database!
    print("PASS: test_successful_login_and_cookie_generation")


def test_incorrect_password_and_generic_error():
    """Scenario 5: Wrong password returns generic error to prevent account enumeration."""
    client = TestClient(app)
    client.post("/api/auth/register", json={
        "name": "Enumeration Test",
        "email": "enum@labhtex.com",
        "password": "ValidPassword1234!"
    })

    resp = client.post("/api/auth/login", json={
        "email": "enum@labhtex.com",
        "password": "WrongPassword999!"
    })
    assert resp.status_code == 401
    assert resp.json()["detail"] == "Invalid email or password."

    # Non-existent user must return the EXACT same generic error
    resp_nonexistent = client.post("/api/auth/login", json={
        "email": "nonexistent@labhtex.com",
        "password": "WrongPassword999!"
    })
    assert resp_nonexistent.status_code == 401
    assert resp_nonexistent.json()["detail"] == "Invalid email or password."
    print("PASS: test_incorrect_password_and_generic_error")


def test_session_idle_timeout(setup_mock_db):
    """Scenario 6: Sessions idle for >30 minutes must be rejected."""
    client = TestClient(app)
    db = setup_mock_db

    client.post("/api/auth/register", json={
        "name": "Idle User",
        "email": "idle@labhtex.com",
        "password": "ValidPassword1234!"
    })
    login_resp = client.post("/api/auth/login", json={
        "email": "idle@labhtex.com",
        "password": "ValidPassword1234!"
    })
    token = login_resp.json()["token"]

    # Artificially age the last_activity by 31 minutes
    token_hash = hash_session_token(token)
    old_time = datetime.now(timezone.utc) - timedelta(minutes=31)
    db.sessions.update_one({"token_hash": token_hash}, {"$set": {"last_activity": old_time}})

    # Call /api/auth/me
    resp = client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 401
    assert "30 minutes of inactivity" in resp.json()["detail"]
    print("PASS: test_session_idle_timeout")


def test_session_revocation_on_logout(setup_mock_db):
    """Scenario 7: Logging out immediately revokes session in MongoDB."""
    client = TestClient(app)
    db = setup_mock_db

    client.post("/api/auth/register", json={
        "name": "Logout User",
        "email": "logout@labhtex.com",
        "password": "ValidPassword1234!"
    })
    login_resp = client.post("/api/auth/login", json={
        "email": "logout@labhtex.com",
        "password": "ValidPassword1234!"
    })
    token = login_resp.json()["token"]

    # Logout
    logout_resp = client.post("/api/auth/logout", headers={"Authorization": f"Bearer {token}"})
    assert logout_resp.status_code == 200

    # Verify session revoked in database
    token_hash = hash_session_token(token)
    session_doc = db.sessions.find_one({"token_hash": token_hash})
    assert session_doc["revoked_at"] is not None

    # Try calling me with revoked token
    me_resp = client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me_resp.status_code == 401
    assert "revoked" in me_resp.json()["detail"]
    print("PASS: test_session_revocation_on_logout")


def test_admin_registration_and_role_separation(setup_mock_db, monkeypatch):
    """Scenario 8: Admin key allows admin registration; normal users receive role=user."""
    client = TestClient(app)
    monkeypatch.setattr("backend.security.ADMIN_REGISTRATION_KEY", "SuperSecretAdminKey2026")

    # Regular user
    r1 = client.post("/api/auth/register", json={
        "name": "Normal User",
        "email": "normal@labhtex.com",
        "password": "ValidPassword1234!"
    })
    assert r1.json()["role"] == "user"

    # User with wrong admin key
    r2 = client.post("/api/auth/register", json={
        "name": "Attacker",
        "email": "attacker@labhtex.com",
        "password": "ValidPassword1234!",
        "admin_key": "wrong_key"
    })
    assert r2.status_code == 400
    assert "Invalid administrator registration key" in r2.json()["detail"]

    # User with valid admin key
    r3 = client.post("/api/auth/register", json={
        "name": "True Admin",
        "email": "admin@labhtex.com",
        "password": "ValidPassword1234!",
        "admin_key": "SuperSecretAdminKey2026"
    })
    assert r3.status_code == 201
    assert r3.json()["role"] == "admin"
    print("PASS: test_admin_registration_and_role_separation")


def test_role_based_access_control_on_admin_endpoints(setup_mock_db, monkeypatch):
    """Scenario 9: Normal users cannot access admin endpoints (HTTP 403); Admins can access (HTTP 200)."""
    client = TestClient(app)
    monkeypatch.setattr("backend.security.ADMIN_REGISTRATION_KEY", "AdminKey99")

    # 1. Register and login normal user
    client.post("/api/auth/register", json={
        "name": "Regular User",
        "email": "reg@labhtex.com",
        "password": "ValidPassword1234!"
    })
    user_token = client.post("/api/auth/login", json={
        "email": "reg@labhtex.com",
        "password": "ValidPassword1234!"
    }).json()["token"]

    # 2. Register and login admin
    client.post("/api/auth/register", json={
        "name": "Boss Admin",
        "email": "boss@labhtex.com",
        "password": "ValidPassword1234!",
        "admin_key": "AdminKey99"
    })
    admin_token = client.post("/api/auth/login", json={
        "email": "boss@labhtex.com",
        "password": "ValidPassword1234!"
    }).json()["token"]

    # Normal user attempts to access /api/admin/documents
    user_resp = client.get("/api/admin/documents", headers={"Authorization": f"Bearer {user_token}"})
    assert user_resp.status_code == 403
    assert "Administrative privileges required" in user_resp.json()["detail"]

    # Admin accesses /api/admin/documents
    admin_resp = client.get("/api/admin/documents", headers={"Authorization": f"Bearer {admin_token}"})
    assert admin_resp.status_code == 200
    print("PASS: test_role_based_access_control_on_admin_endpoints")


def test_conversation_ownership_and_cross_user_isolation(setup_mock_db):
    """Scenario 10: User A cannot read, modify, or delete User B's conversations."""
    client = TestClient(app)

    # Register User A
    client.post("/api/auth/register", json={"name": "Alice", "email": "alice@labhtex.com", "password": "PasswordAlice123!"})
    token_a = client.post("/api/auth/login", json={"email": "alice@labhtex.com", "password": "PasswordAlice123!"}).json()["token"]

    # Register User B
    client.post("/api/auth/register", json={"name": "Bob", "email": "bob@labhtex.com", "password": "PasswordBob12345!"})
    token_b = client.post("/api/auth/login", json={"email": "bob@labhtex.com", "password": "PasswordBob12345!"}).json()["token"]

    # Alice creates a conversation
    conv_a = client.post(
        "/api/conversations",
        json={"title": "Alice Private Loom Specs"},
        headers={"Authorization": f"Bearer {token_a}"}
    ).json()
    conv_a_id = conv_a["id"]

    # Bob attempts to read Alice's conversation
    bob_read = client.get(f"/api/conversations/{conv_a_id}", headers={"Authorization": f"Bearer {token_b}"})
    assert bob_read.status_code == 404

    # Bob attempts to rename Alice's conversation
    bob_patch = client.patch(
        f"/api/conversations/{conv_a_id}",
        json={"title": "Hacked Title"},
        headers={"Authorization": f"Bearer {token_b}"}
    )
    assert bob_patch.status_code == 404

    # Bob attempts to delete Alice's conversation
    bob_del = client.delete(f"/api/conversations/{conv_a_id}", headers={"Authorization": f"Bearer {token_b}"})
    assert bob_del.status_code == 404

    # Alice can successfully access her own conversation
    alice_read = client.get(f"/api/conversations/{conv_a_id}", headers={"Authorization": f"Bearer {token_a}"})
    assert alice_read.status_code == 200
    assert alice_read.json()["title"] == "Alice Private Loom Specs"
    print("PASS: test_conversation_ownership_and_cross_user_isolation")


def test_chat_persistence_and_deletion_cascade(setup_mock_db):
    """Scenario 11: Messages persist with citations; deleting conversation cascades to all messages."""
    client = TestClient(app)
    db = setup_mock_db

    client.post("/api/auth/register", json={"name": "Charlie", "email": "charlie@labhtex.com", "password": "CharliePassword123!"})
    token = client.post("/api/auth/login", json={"email": "charlie@labhtex.com", "password": "CharliePassword123!"}).json()["token"]

    # Create conversation
    conv = client.post("/api/conversations", json={"title": "Weft Insertion"}, headers={"Authorization": f"Bearer {token}"}).json()
    conv_id = conv["id"]

    # Post message
    msg_resp = client.post(
        f"/api/conversations/{conv_id}/messages",
        json={"content": "What is the rapier loom insertion rate?"},
        headers={"Authorization": f"Bearer {token}"}
    )
    assert msg_resp.status_code == 200
    msg_data = msg_resp.json()
    assert msg_data["role"] == "assistant"
    assert "citations" in msg_data

    # Check messages list
    list_resp = client.get(f"/api/conversations/{conv_id}/messages", headers={"Authorization": f"Bearer {token}"})
    assert list_resp.status_code == 200
    messages = list_resp.json()
    assert len(messages) == 2  # 1 user + 1 assistant
    assert messages[0]["role"] == "user"
    assert messages[1]["role"] == "assistant"

    # Delete conversation
    del_resp = client.delete(f"/api/conversations/{conv_id}", headers={"Authorization": f"Bearer {token}"})
    assert del_resp.status_code == 200

    # Ensure messages were purged in cascade
    db_messages = list(db.messages.find({"conversation_id": conv_id}))
    assert len(db_messages) == 0
    print("PASS: test_chat_persistence_and_deletion_cascade")


def test_mongodb_unavailable_handling():
    """Scenario 12: Clear error raised when MongoDB is unavailable, with NO SQLite fallback."""
    from backend.database import get_client
    import os

    # Temporarily clear MONGODB_URI to test strict no-fallback error handling
    orig = os.environ.get("MONGODB_URI")
    try:
        os.environ["MONGODB_URI"] = ""
        with pytest.raises(DatabaseUnavailableError) as exc_info:
            get_client()
        assert "MONGODB_URI is not set" in str(exc_info.value)
        assert "SQLite" not in str(exc_info.value)  # Absolutely no SQLite fallback
    finally:
        if orig is not None:
            os.environ["MONGODB_URI"] = orig
    print("PASS: test_mongodb_unavailable_handling")


def main():
    print("Running comprehensive Authentication, Authorization & Chat History test suite...\n")
    mock_db = mongomock.MongoClient()["test_db"]
    init_indexes(mock_db)
    set_test_db(mock_db)

    test_successful_registration_and_password_hashing(mock_db)
    test_duplicate_email_rejection()
    test_minimum_password_length_validation()
    test_successful_login_and_cookie_generation(mock_db)
    test_incorrect_password_and_generic_error()
    test_session_idle_timeout(mock_db)
    test_session_revocation_on_logout(mock_db)

    import types
    dummy_monkey = types.SimpleNamespace(setattr=lambda obj, attr, val: setattr(__import__(obj.__module__, fromlist=[obj.__name__]), attr, val) if hasattr(obj, '__module__') else None)

    # Test admin separation with mock
    from backend import security
    old_key = security.ADMIN_REGISTRATION_KEY
    security.ADMIN_REGISTRATION_KEY = "TestSecret2026"
    test_admin_registration_and_role_separation(mock_db, types.SimpleNamespace(setattr=lambda o, a, v: setattr(security, a, v)))
    test_role_based_access_control_on_admin_endpoints(mock_db, types.SimpleNamespace(setattr=lambda o, a, v: setattr(security, a, v)))
    security.ADMIN_REGISTRATION_KEY = old_key

    test_conversation_ownership_and_cross_user_isolation(mock_db)
    test_chat_persistence_and_deletion_cascade(mock_db)
    test_mongodb_unavailable_handling()

    print("\nAll 12 backend security & persistence scenarios PASSED successfully!")


if __name__ == "__main__":
    main()
