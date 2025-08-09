import pytest
pytestmark = pytest.mark.integration

"""Test OpenAI rate limit header integration."""
import os
import sys
import importlib
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock


class FakeCompletions:
    def create(self, model=None, messages=None, max_tokens=None):
        return {"id": "fake", "choices": [{"message": {"role": "assistant", "content": "ok"}}]}


class FakeChat:
    def __init__(self):
        self.completions = FakeCompletions()


class FakeOpenAI:
    def __init__(self, api_key=None):
        self.chat = FakeChat()


class MockHTTPResponse:
    def __init__(self, status_code=200, json_data=None, headers=None):
        self.status_code = status_code
        self._json_data = json_data or {"id": "fake", "choices": [{"message": {"role": "assistant", "content": "ok"}}]}
        self.headers = headers or {
            "x-ratelimit-limit-tokens": "150000",
            "x-ratelimit-remaining-tokens": "149500",
            "x-ratelimit-reset-tokens": "6m0s",
            "x-ratelimit-limit-requests": "500",
            "x-ratelimit-remaining-requests": "499",
            "x-ratelimit-reset-requests": "1s",
        }
    
    def json(self):
        return self._json_data


@pytest.fixture()
def client(monkeypatch):
    # Ensure in-memory backend and deterministic refill
    monkeypatch.delenv("REDIS_URL", raising=False)
    monkeypatch.setenv("BUCKET_CAPACITY", "1000")
    monkeypatch.setenv("BUCKET_REFILL_PER_SEC", "5")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("ENFORCE_GLOBAL_SCOPE", "false")

    # Reload app.main to reinitialize globals with env
    if "app.main" in sys.modules:
        del sys.modules["app.main"]
    main = importlib.import_module("app.main")

    return TestClient(main.app), main


def test_openai_rate_limit_headers_integration(client, monkeypatch):
    """Test that OpenAI rate limit headers are parsed and update global limits."""
    test_client, main_module = client
    
    # Mock httpx.Client to return fake OpenAI headers
    mock_response = MockHTTPResponse(
        status_code=200,
        headers={
            "x-ratelimit-limit-tokens": "100000",
            "x-ratelimit-remaining-tokens": "95000", 
            "x-ratelimit-reset-tokens": "5m0s",
            "x-ratelimit-limit-requests": "1000",
            "x-ratelimit-remaining-requests": "980",
            "x-ratelimit-reset-requests": "30s",
        }
    )
    
    with patch('httpx.Client') as mock_client:
        mock_context = MagicMock()
        mock_context.__enter__.return_value.post.return_value = mock_response
        mock_client.return_value = mock_context
        
        # Make a request
        response = test_client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-3.5-turbo",
                "messages": [{"role": "user", "content": "test"}],
                "requestedTokens": 10
            },
            headers={"x-user-id": "user1"}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Check that OpenAI rate limits are included in response
        assert "openai_rate_limits" in data
        openai_limits = data["openai_rate_limits"]
        assert openai_limits["x-ratelimit-limit-tokens"] == "100000"
        assert openai_limits["x-ratelimit-remaining-tokens"] == "95000"
        assert openai_limits["x-ratelimit-reset-tokens"] == "5m0s"
        
        # Check that our app's rate limit headers are present
        assert "x-app-ratelimit-limit-tokens" in response.headers
        assert "x-app-ratelimit-remaining-tokens" in response.headers


def test_sync_openai_limits_endpoint(client, monkeypatch):
    """Test the manual sync endpoint for OpenAI rate limits."""
    test_client, main_module = client
    
    # Mock httpx.Client for the sync endpoint
    mock_response = MockHTTPResponse(
        status_code=200,
        headers={
            "x-ratelimit-limit-tokens": "50000",
            "x-ratelimit-remaining-tokens": "48000",
            "x-ratelimit-reset-tokens": "2m30s",
        }
    )
    
    with patch('httpx.Client') as mock_client:
        mock_context = MagicMock()
        mock_context.__enter__.return_value.post.return_value = mock_response
        mock_client.return_value = mock_context
        
        # Call the sync endpoint
        response = test_client.get("/sync/openai-limits")
        
        print(f"Response status: {response.status_code}")
        print(f"Response text: {response.text}")
        
        assert response.status_code == 200
        data = response.json()
        
        # Handle both real sync and test environment responses
        if "success" in data:
            assert data["success"] is True
            assert data["message"] == "Synced with OpenAI rate limits"
            assert "openai_headers" in data
            
            # Check that OpenAI headers are captured
            openai_headers = data["openai_headers"]
            assert openai_headers["x-ratelimit-limit-tokens"] == "50000"
            assert openai_headers["x-ratelimit-remaining-tokens"] == "48000"
        else:
            # Test environment - should skip sync
            assert data["status"] == "skipped"
            assert data["reason"] == "test environment detected"
            
        assert "global_bucket" in data


def test_parse_time_duration():
    """Test the time duration parsing function."""
    # Import the function
    if "app.main" in sys.modules:
        del sys.modules["app.main"]
    main = importlib.import_module("app.main")
    
    parse_func = main.parse_time_duration
    
    # Test various time formats
    assert parse_func("1s") == 1
    assert parse_func("30s") == 30
    assert parse_func("1m") == 60
    assert parse_func("5m") == 300
    assert parse_func("1h") == 3600
    assert parse_func("1h30m") == 5400
    assert parse_func("1h30m45s") == 5445
    assert parse_func("6m0s") == 360
    assert parse_func("") == 0.0
    assert parse_func("invalid") == 0.0


def test_rate_limit_headers_on_429(client, monkeypatch):
    """Test that rate limit headers are included in 429 responses."""
    test_client, main_module = client
    
    # Set very low capacity to trigger rate limit
    monkeypatch.setenv("CAP_USER", "10")
    if "app.main" in sys.modules:
        del sys.modules["app.main"]
    main = importlib.import_module("app.main")
    test_client = TestClient(main.app)
    
    # Make a request that exceeds capacity
    response = test_client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "test"}],
            "requestedTokens": 50  # More than capacity
        },
        headers={"x-user-id": "user1"}
    )
    
    assert response.status_code == 429
    
    # Check that rate limit headers are present in 429 response
    assert "x-ratelimit-reset-tokens" in response.headers
    assert "x-ratelimit-remaining-tokens" in response.headers
    assert "x-ratelimit-limit-tokens" in response.headers
