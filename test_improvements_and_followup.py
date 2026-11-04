import os
import pytest
from datetime import datetime, timezone
from bson import ObjectId
from fastapi.testclient import TestClient

from backend.config import (
    get_admin_registration_key,
    get_cookie_secure,
    get_retrieval_score_threshold,
    validate_config,
)
from backend.main import app
from src.api_client import APIClient
from src.generation import (
    generate_answer,
    is_dependent_followup,
    rewrite_query_for_retrieval,
)
from src.retrieval import retrieve_context, get_configured_threshold


client = TestClient(app)


# ------------------------------------------------------------------------------
# 1. Requirement 1 Tests: Admin Secret Configuration & Startup Validation
# ------------------------------------------------------------------------------
def test_admin_secret_missing_fails_validation(monkeypatch):
    """Application validation must fail if ADMIN_REGISTRATION_KEY is missing or empty."""
    monkeypatch.setenv("ADMIN_REGISTRATION_KEY", "")
    with pytest.raises(RuntimeError) as exc_info:
        validate_config()
    assert "ADMIN_REGISTRATION_KEY environment variable is missing or empty" in str(exc_info.value)


def test_admin_secret_configured_passes_validation(monkeypatch):
    """Application validation succeeds when a non-empty key is provided."""
    monkeypatch.setenv("ADMIN_REGISTRATION_KEY", "ConfiguredSecretKey123!")
    # Should not raise
    validate_config()
    assert get_admin_registration_key() == "ConfiguredSecretKey123!"


# ------------------------------------------------------------------------------
# 2. Requirement 2 Tests: COOKIE_SECURE Boolean Configuration & Behavior
# ------------------------------------------------------------------------------
@pytest.mark.parametrize("val,expected", [
    ("true", True),
    ("TRUE", True),
    ("1", True),
    ("yes", True),
    ("false", False),
    ("FALSE", False),
    ("0", False),
    ("no", False),
])
def test_cookie_secure_parsing(monkeypatch, val, expected):
    """Ensure strict parsing of COOKIE_SECURE without silent misinterpretation."""
    monkeypatch.setenv("COOKIE_SECURE", val)
    assert get_cookie_secure() is expected


def test_cookie_secure_invalid_string_raises(monkeypatch):
    """Arbitrary strings must raise ValueError rather than silently defaulting."""
    monkeypatch.setenv("COOKIE_SECURE", "maybe")
    with pytest.raises(ValueError) as exc_info:
        get_cookie_secure()
    assert "Invalid COOKIE_SECURE setting" in str(exc_info.value)


def test_cookie_secure_default_based_on_backend_url(monkeypatch):
    """When COOKIE_SECURE is unset, defaults to True for HTTPS, False for HTTP."""
    monkeypatch.delenv("COOKIE_SECURE", raising=False)
    monkeypatch.setenv("BACKEND_URL", "https://api.labhtex.com")
    assert get_cookie_secure() is True

    monkeypatch.setenv("BACKEND_URL", "http://127.0.0.1:8000")
    assert get_cookie_secure() is False


# ------------------------------------------------------------------------------
# 3. Requirement 3 Tests: Retrieval Threshold & Honest Abstention
# ------------------------------------------------------------------------------
def test_retrieval_threshold_configuration(monkeypatch):
    """Test retrieval threshold configuration loading."""
    monkeypatch.setenv("RETRIEVAL_SCORE_THRESHOLD", "0.45")
    assert get_retrieval_score_threshold() == 0.45

    monkeypatch.setenv("RETRIEVAL_SCORE_THRESHOLD", "invalid")
    assert get_retrieval_score_threshold() == 0.40


def test_relevant_astm_query_passes_threshold():
    """Exact technical code queries must retrieve high-similarity passages above threshold."""
    query = "What is the standard test method and pressure specification for burst strength testing of stretch lace fabrics according to ASTM D3786?"
    results = retrieve_context(query, top_k=4, score_threshold=0.40)
    assert len(results) > 0
    top_similarity = max(r["similarity"] for r in results)
    assert top_similarity >= 0.70, f"Expected top similarity >= 0.70, got {top_similarity}"
    assert "ASTM D3786" in results[0]["text"] or "burst" in results[0]["text"].lower()


def test_irrelevant_out_of_domain_query_rejected_by_threshold():
    """Completely out-of-domain queries must yield zero chunks under default threshold."""
    irrelevant_query = "How do you calculate the thermal efficiency of a combined-cycle Brayton-Rankine gas turbine?"
    results = retrieve_context(irrelevant_query, top_k=4, score_threshold=0.40)
    assert len(results) == 0, f"Expected 0 chunks for out-of-domain query, got {len(results)}"


def test_empty_knowledge_base_or_no_match_returns_abstention():
    """Empty retrieval context must trigger clean, honest abstention without calling LLM."""
    res = generate_answer(query="Unknown question", retrieved_chunks=[])
    assert res["abstained"] is True
    assert "I couldn't find sufficient information in the available documents" in res["answer"]
    assert len(res["citations"]) == 0


# ------------------------------------------------------------------------------
# 4. Requirement 4 Tests: Conversation History & Follow-Up Question Rewriting
# ------------------------------------------------------------------------------
def test_is_dependent_followup_detection():
    """Test detection of context-dependent questions vs independent standalone questions."""
    # Dependent queries
    assert is_dependent_followup("What about its washing requirements?") is True
    assert is_dependent_followup("What is its price?") is True
    assert is_dependent_followup("Tell me more about them.") is True
    assert is_dependent_followup("How much does that cost?") is True
    assert is_dependent_followup("Can it be dyed?") is True

    # Standalone queries
    assert is_dependent_followup("What is the burst strength of Raschel warp-knitted lace according to ASTM D3786?") is False
    assert is_dependent_followup("What fibers are used in Chemical Guipure lace embroidery?") is False


def test_query_rewriting_without_history_returns_unchanged():
    """If no history exists, the query must be returned unchanged."""
    query = "What is its price?"
    rewritten = rewrite_query_for_retrieval(query, history=[])
    assert rewritten == query


def test_query_rewriting_standalone_returns_unchanged():
    """Independent questions must not be modified even if history is present."""
    history = [
        {"role": "user", "content": "Tell me about bridal lace."},
        {"role": "assistant", "content": "Leavers lace is ideal for bridal gowns."}
    ]
    standalone = "What is the standard test method for ASTM D3786 burst strength?"
    rewritten = rewrite_query_for_retrieval(standalone, history=history)
    assert rewritten == standalone


def test_query_rewriting_fallback_combines_topic():
    """When LLM is unavailable or fails, heuristic fallback incorporates recent subject."""
    history = [
        {"role": "user", "content": "Recommend a lace for bridal wear."},
        {"role": "assistant", "content": "Leavers lace is traditionally used."}
    ]
    # Pass empty api_key to force fallback
    rewritten = rewrite_query_for_retrieval("What about its washing requirements?", history=history, api_key="")
    assert "bridal" in rewritten.lower() or "lace" in rewritten.lower()
    assert "washing" in rewritten.lower()


# ------------------------------------------------------------------------------
# 5. Requirement 5 Tests: Liveness, Readiness & Health Endpoint
# ------------------------------------------------------------------------------
def test_liveness_endpoint():
    """Liveness probe /live must return 200 OK with status alive."""
    response = client.get("/live")
    assert response.status_code == 200
    assert response.json()["status"] == "alive"


def test_health_readiness_endpoint():
    """Health endpoint /health must return status and component readiness details."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "ready" in data
    assert "onnx_ready" in data
    assert "database" in data
    assert data["onnx_ready"] is True


def test_api_client_check_health():
    """APIClient check_health method must communicate with health endpoint."""
    api = APIClient(base_url="http://testserver")
    # Using TestClient session directly
    api.session = client
    ok, info = api.check_health(timeout=5)
    assert ok is True
    assert info.get("status") in ("online", "degraded")
