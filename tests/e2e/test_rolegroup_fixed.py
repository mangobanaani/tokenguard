import pytest
pytestmark = pytest.mark.e2e
"""Fixed role group tests without module reloading."""
import os
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


@pytest.fixture(scope="session")
def app_with_env():
    """Create app with test environment once per session."""
    # Set test environment before importing
    test_env = {
        "REDIS_URL": "",
        "BUCKET_CAPACITY": "100",
        "BUCKET_REFILL_PER_SEC": "0",
        "OPENAI_API_KEY": "test-key",
        "ENFORCE_GLOBAL_SCOPE": "false",
        "ENABLE_BACKGROUND_SYNC": "false",  # Disable background sync in tests
        "TOKEN_ADJUSTMENT_THRESHOLD": "10.0",  # Disable token adjustment in tests (1000% threshold)
        "CAP_ROLEGROUP": "100",
        "RATE_ROLEGROUP": "1",
        "CAP_ROLE": "200",
        "RATE_ROLE": "1",
    }
    
    # Apply environment
    original_env = {}
    for key, value in test_env.items():
        original_env[key] = os.environ.get(key)
        os.environ[key] = value
    
    # Import app after setting environment
    from app.main import app
    
    yield app
    
    # Restore environment
    for key, value in original_env.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


@pytest.fixture()
def client(app_with_env):
    """Create test client with mocked OpenAI."""
    with patch('app.main.OpenAI', FakeOpenAI):
        # Also mock httpx to prevent real HTTP requests
        def mock_post(url, json=None, headers=None, timeout=None):
            mock_response = MagicMock()
            mock_response.status_code = 200
            
            # For testing: return realistic usage that's close to what a real API would return
            # This avoids issues with token adjustment while still testing rate limiting
            if json and "messages" in json:
                # Estimate based on actual message content - this will be closer to real usage
                content_length = sum(len(msg.get("content", "")) for msg in json["messages"])
                base_tokens = max(5, content_length // 4)  # ~4 chars per token
                # Add some overhead for chat formatting
                estimated_tokens = int(base_tokens * 1.3)  # 30% overhead
                # For very small test messages, return a reasonable minimum
                estimated_tokens = max(estimated_tokens, 10)
            else:
                estimated_tokens = 15  # Default for unknown content
            
            mock_response.json.return_value = {
                "id": "fake",
                "choices": [{"message": {"role": "assistant", "content": "ok"}}],
                "usage": {"total_tokens": estimated_tokens, "prompt_tokens": estimated_tokens//2, "completion_tokens": estimated_tokens//2}
            }
            mock_response.headers = {}
            return mock_response
        
        with patch('httpx.Client') as mock_httpx:
            mock_httpx.return_value.__enter__.return_value.post.side_effect = mock_post
            yield TestClient(app_with_env)


@pytest.fixture(autouse=True)
def reset_buckets():
    """Reset bucket state between tests to ensure test isolation."""
    # This runs before each test
    yield
    # This runs after each test - clear any in-memory bucket state
    try:
        from app.main import buckets, capacity_overrides, refill_overrides
        buckets.clear()
        capacity_overrides.clear()
        refill_overrides.clear()
    except ImportError:
        pass  # Module not imported yet


def test_rolegroup_exact_matching(client):
    """Test that x-llm-rolegroup matches exactly, not as substring."""
    
    # First, consume tokens for rolegroup "foobar"
    response1 = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "test"}],
            "requestedTokens": 50
        },
        headers={
            "x-user-id": "user1",
            "x-llm-rolegroup": "foobar"
        }
    )
    assert response1.status_code == 200, f"Response 1 failed: {response1.text}"
    
    # Consume more tokens for rolegroup "foobar" (should work, within limit)
    response2 = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-3.5-turbo", 
            "messages": [{"role": "user", "content": "test"}],
            "requestedTokens": 40
        },
        headers={
            "x-user-id": "user1",
            "x-llm-rolegroup": "foobar"
        }
    )
    assert response2.status_code == 200, f"Response 2 failed: {response2.text}"
    
    # Try to consume more tokens for rolegroup "foobar" (should fail, over limit)
    response3 = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "test"}], 
            "requestedTokens": 20
        },
        headers={
            "x-user-id": "user1",
            "x-llm-rolegroup": "foobar"
        }
    )
    assert response3.status_code == 429, f"Response 3 should be rate limited: {response3.text}"
    
    # But rolegroup "barfoo" should have its own separate bucket
    response4 = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "test"}],
            "requestedTokens": 50
        },
        headers={
            "x-user-id": "user2", 
            "x-llm-rolegroup": "barfoo"
        }
    )
    assert response4.status_code == 200, f"Response 4 should work (different rolegroup): {response4.text}"


def test_rolegroup_hierarchy_precedence(client):
    """Test that rolegroup takes precedence over role in hierarchy."""
    
    # Test that rolegroup limit (100) is enforced instead of role limit (200)
    response1 = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "test"}],
            "requestedTokens": 40
        },
        headers={
            "x-user-id": "user1",
            "x-role": "admin",
            "x-llm-rolegroup": "special"
        }
    )
    assert response1.status_code == 200
    
    # This should fail because rolegroup limit (100) is reached, not role limit (200)
    response2 = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "test"}],
            "requestedTokens": 70  # 40 + 70 = 110 > 100 (rolegroup limit)
        },
        headers={
            "x-user-id": "user1",
            "x-role": "admin",
            "x-llm-rolegroup": "special"
        }
    )
    assert response2.status_code == 429


def test_rolegroup_case_sensitivity(client):
    """Test that rolegroup matching is case-sensitive."""
    
    # Use tokens for "Special" rolegroup
    response1 = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "test"}],
            "requestedTokens": 90
        },
        headers={
            "x-user-id": "user1",
            "x-llm-rolegroup": "Special"
        }
    )
    print(f"Response 1 status: {response1.status_code}")
    if response1.status_code == 200:
        data1 = response1.json()
        print(f"Response 1 buckets: {data1.get('buckets', {})}")
    assert response1.status_code == 200
    
    # "special" (lowercase) should be a different bucket
    response2 = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "test"}],
            "requestedTokens": 90
        },
        headers={
            "x-user-id": "user2",
            "x-llm-rolegroup": "special"
        }
    )
    print(f"Response 2 status: {response2.status_code}")
    if response2.status_code == 200:
        data2 = response2.json()
        print(f"Response 2 buckets: {data2.get('buckets', {})}")
    else:
        print(f"Response 2 error: {response2.text}")
    assert response2.status_code == 200, "Case-sensitive: 'special' != 'Special'"
