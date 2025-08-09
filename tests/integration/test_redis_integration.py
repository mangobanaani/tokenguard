import pytest
pytestmark = pytest.mark.integration

import time
import redis
import os
from fastapi.testclient import TestClient


@pytest.fixture(scope="session")
def redis_url():
    """Provide Redis URL for integration tests"""
    return os.getenv("TEST_REDIS_URL", "redis://localhost:6379/1")


@pytest.fixture
def redis_client(redis_url):
    """Create Redis client and clean up test data"""
    try:
        client = redis.Redis.from_url(redis_url, decode_responses=True)
        client.ping()  # Test connection
        
        # Clean up any existing test data
        for key in client.scan_iter(match="bucket:*"):
            client.delete(key)
            
        yield client
        
        # Clean up after test
        for key in client.scan_iter(match="bucket:*"):
            client.delete(key)
    except (redis.ConnectionError, redis.TimeoutError):
        pytest.skip("Redis not available for integration tests")


@pytest.fixture
def redis_app(redis_client, monkeypatch):
    """Create FastAPI app with Redis backend"""
    import sys
    import importlib
    
    # Mock OpenAI for tests
    class FakeOpenAI:
        def __init__(self, api_key=None):
            self.chat = lambda: None
            self.chat.completions = lambda: None
            self.chat.completions.create = lambda **kwargs: {
                "id": "fake", 
                "choices": [{"message": {"role": "assistant", "content": "test response"}}]
            }
    
    # Set up environment for Redis backend
    monkeypatch.setenv("REDIS_URL", redis_client.connection_pool.connection_kwargs['host'] + ":" + str(redis_client.connection_pool.connection_kwargs['port']) + "/1")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("BUCKET_CAPACITY", "100")
    monkeypatch.setenv("BUCKET_REFILL_PER_SEC", "5")
    
    # Reload app with Redis
    if "app.main" in sys.modules:
        del sys.modules["app.main"]
    
    main = importlib.import_module("app.main")
    monkeypatch.setattr(main, "OpenAI", FakeOpenAI)
    
    return TestClient(main.app)


class TestRedisIntegration:
    """Integration tests with Redis backend"""
    
    def test_redis_backend_initialization(self, redis_app):
        """Test that Redis backend initializes correctly"""
        r = redis_app.get("/")
        assert r.status_code == 200
        data = r.json()
        assert data["backend"] == "redis"
    
    def test_redis_bucket_persistence(self, redis_app, redis_client):
        """Test that bucket state persists in Redis"""
        # Set up a bucket
        redis_app.post("/config/budget", json={
            "scope": "user", 
            "id": "persist_user", 
            "capacity": 100, 
            "refillPerSec": 10
        })
        
        # Make a request to consume tokens
        redis_app.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "test"}],
            "requestedTokens": 30
        }, headers={"x-user-id": "persist_user"})
        
        # Check Redis directly for bucket state
        tokens_key = "bucket:user:persist_user:tokens"
        rate_key = "bucket:user:persist_user:rate" 
        cap_key = "bucket:user:persist_user:cap"
        
        assert redis_client.exists(tokens_key)
        assert float(redis_client.get(tokens_key)) == 70  # 100 - 30
        assert float(redis_client.get(rate_key)) == 10
        assert float(redis_client.get(cap_key)) == 100
    
    def test_redis_multi_scope_atomicity(self, redis_app):
        """Test that multi-scope operations are atomic in Redis"""
        # Set up hierarchical budgets
        redis_app.post("/config/budget", json={
            "scope": "global", 
            "capacity": 200, 
            "refillPerSec": 0
        })
        redis_app.post("/config/budget", json={
            "scope": "team", 
            "id": "atomic_team", 
            "capacity": 150, 
            "refillPerSec": 0
        })
        redis_app.post("/config/budget", json={
            "scope": "user", 
            "id": "atomic_user", 
            "capacity": 100, 
            "refillPerSec": 0
        })
        
        # Enable global scope
        import os
        os.environ["ENFORCE_GLOBAL_SCOPE"] = "true"
        
        # Make request that should succeed
        r = redis_app.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "test"}],
            "requestedTokens": 80
        }, headers={
            "x-user-id": "atomic_user",
            "x-team-id": "atomic_team"
        })
        
        assert r.status_code == 200
        buckets = r.json()["buckets"]
        
        # All scopes should be decremented
        assert buckets["global"]["tokens"] == 120  # 200 - 80
        assert buckets["team:atomic_team"]["tokens"] == 70  # 150 - 80  
        assert buckets["user:atomic_user"]["tokens"] == 20  # 100 - 80
    
    def test_redis_concurrent_access(self, redis_app):
        """Test concurrent access with Redis backend"""
        import concurrent.futures
        import threading
        
        # Set up bucket
        redis_app.post("/config/budget", json={
            "scope": "user",
            "id": "concurrent_redis_user", 
            "capacity": 1000,
            "refillPerSec": 0
        })
        
        results = []
        
        def make_request():
            try:
                r = redis_app.post("/v1/chat/completions", json={
                    "messages": [{"role": "user", "content": "test"}],
                    "requestedTokens": 10
                }, headers={"x-user-id": "concurrent_redis_user"})
                results.append(r.status_code)
            except Exception as e:
                results.append(f"error: {e}")
        
        # Launch concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(make_request) for _ in range(50)]
            concurrent.futures.wait(futures)
        
        # Should have exactly 50 successful requests (1000 capacity / 10 tokens each = 100 max)
        success_count = sum(1 for r in results if r == 200)
        assert success_count == 50
    
    def test_redis_refill_accuracy(self, redis_app):
        """Test that Redis-based refill is accurate over time"""
        # Set up bucket with known refill rate
        redis_app.post("/config/budget", json={
            "scope": "user",
            "id": "refill_test_user",
            "capacity": 100,
            "refillPerSec": 20  # 20 tokens per second
        })
        
        # Drain bucket completely
        r = redis_app.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "test"}],
            "requestedTokens": 100
        }, headers={"x-user-id": "refill_test_user"})
        assert r.status_code == 200
        
        # Should be empty now
        r = redis_app.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "test"}],
            "requestedTokens": 1
        }, headers={"x-user-id": "refill_test_user"})
        assert r.status_code == 429
        
        # Wait 2 seconds (should refill 40 tokens)
        time.sleep(2.1)
        
        # Should be able to use ~40 tokens
        r = redis_app.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "test"}],
            "requestedTokens": 35
        }, headers={"x-user-id": "refill_test_user"})
        assert r.status_code == 200
        
        # But not 45 tokens
        r = redis_app.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "test"}],
            "requestedTokens": 10
        }, headers={"x-user-id": "refill_test_user"})
        assert r.status_code == 429
    
    def test_redis_capacity_updates(self, redis_app, redis_client):
        """Test that capacity updates work correctly with Redis"""
        user_id = "capacity_test_user"
        
        # Set initial capacity
        redis_app.post("/config/budget", json={
            "scope": "user",
            "id": user_id,
            "capacity": 100,
            "refillPerSec": 0
        })
        
        # Use some tokens
        redis_app.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "test"}],
            "requestedTokens": 30
        }, headers={"x-user-id": user_id})
        
        # Reduce capacity below current tokens
        redis_app.post("/config/budget", json={
            "scope": "user",
            "id": user_id, 
            "capacity": 50
        })
        
        # Current tokens should be capped at new capacity
        tokens_key = f"bucket:user:{user_id}:tokens"
        current_tokens = float(redis_client.get(tokens_key) or 0)
        assert current_tokens <= 50
    
    def test_redis_key_expiration_behavior(self, redis_app, redis_client):
        """Test behavior when Redis keys don't exist (first time use)"""
        # Make request for new user (no existing keys)
        r = redis_app.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "test"}],
            "requestedTokens": 20
        }, headers={"x-user-id": "new_redis_user"})
        
        assert r.status_code == 200
        buckets = r.json()["buckets"]
        
        # Should have initialized with default values
        user_bucket = buckets["user:new_redis_user"]
        assert user_bucket["capacity"] == 100  # Default from env
        assert user_bucket["tokens"] == 80     # 100 - 20
        
        # Keys should now exist in Redis
        tokens_key = "bucket:user:new_redis_user:tokens"
        ts_key = "bucket:user:new_redis_user:ts"
        
        assert redis_client.exists(tokens_key)
        assert redis_client.exists(ts_key)


class TestRedisFailover:
    """Test behavior when Redis is unavailable"""
    
    def test_redis_connection_failure_fallback(self, monkeypatch):
        """Test that app gracefully handles Redis connection failures"""
        import sys
        import importlib
        
        # Set invalid Redis URL
        monkeypatch.setenv("REDIS_URL", "redis://invalid-host:6379/0")
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        
        # This should fall back to in-memory backend if Redis connection fails
        # The app should still start and work
        if "app.main" in sys.modules:
            del sys.modules["app.main"]
        
        try:
            main = importlib.import_module("app.main")
            # If Redis connection fails during import, it should fall back to memory
            # This test verifies the app doesn't crash on Redis connection errors
            assert True  # If we get here, the import succeeded
        except Exception as e:
            # If there's an error, it should be a controlled fallback, not a crash
            pytest.fail(f"App failed to handle Redis connection error gracefully: {e}")


@pytest.mark.performance
class TestPerformance:
    """Performance tests for bucket operations"""
    
    def test_redis_lua_script_performance(self, redis_app):
        """Test performance of Redis Lua script operations"""
        import time
        
        # Set up test bucket
        redis_app.post("/config/budget", json={
            "scope": "user",
            "id": "perf_user", 
            "capacity": 10000,
            "refillPerSec": 100
        })
        
        # Measure time for multiple operations
        start_time = time.time()
        
        for i in range(100):
            r = redis_app.post("/v1/chat/completions", json={
                "messages": [{"role": "user", "content": "test"}],
                "requestedTokens": 10
            }, headers={"x-user-id": "perf_user"})
            assert r.status_code == 200
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Should complete 100 operations in reasonable time (adjust threshold as needed)
        assert duration < 10.0, f"100 operations took {duration:.2f}s, too slow"
        
        # Calculate operations per second
        ops_per_sec = 100 / duration
        print(f"Redis operations per second: {ops_per_sec:.1f}")
        assert ops_per_sec > 10  # Minimum performance threshold
