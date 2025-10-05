import pytest
pytestmark = pytest.mark.unit

import time
import threading
from unittest.mock import Mock, patch
from fastapi.testclient import TestClient


class TestTokenBucketLogic:
    """Unit tests for TokenBucket class logic"""
    
    def test_token_bucket_initialization(self):
        """Test TokenBucket initializes correctly"""
        from app.main import TokenBucket
        
        bucket = TokenBucket(capacity=100, refill_rate_per_sec=5)
        assert bucket.capacity == 100
        assert bucket.refill_rate_per_sec == 5
        assert bucket.tokens == 100  # Should start full
        assert bucket.last_refill <= time.time()
    
    def test_token_bucket_refill(self):
        """Test token bucket refill logic"""
        from app.main import TokenBucket
        
        bucket = TokenBucket(capacity=100, refill_rate_per_sec=10)
        
        # Consume some tokens
        assert bucket.try_remove(50) is True
        assert bucket.tokens == 50
        
        # Manually advance time and test refill
        bucket.last_refill = time.time() - 2.0  # 2 seconds ago
        bucket._refill()
        
        # Should have refilled 20 tokens (10 tokens/sec * 2 sec)
        # Allow for small floating point precision differences
        assert abs(bucket.tokens - 70) < 0.01
    
    def test_token_bucket_capacity_limit(self):
        """Test that tokens don't exceed capacity during refill"""
        from app.main import TokenBucket
        
        bucket = TokenBucket(capacity=100, refill_rate_per_sec=10)
        
        # Consume small amount
        bucket.try_remove(10)
        
        # Simulate long time passing
        bucket.last_refill = time.time() - 100.0
        bucket._refill()
        
        # Should be capped at capacity
        assert bucket.tokens == 100
    
    def test_token_bucket_set_capacity(self):
        """Test setting capacity adjusts current tokens if needed"""
        from app.main import TokenBucket
        
        bucket = TokenBucket(capacity=100, refill_rate_per_sec=5)
        assert bucket.tokens == 100
        
        # Reduce capacity below current tokens
        bucket.set_capacity(50)
        assert bucket.capacity == 50
        assert bucket.tokens == 50  # Should be capped
        
        # Increase capacity
        bucket.set_capacity(150)
        assert bucket.capacity == 150
        assert bucket.tokens == 50  # Doesn't increase existing tokens
    
    def test_token_bucket_negative_values(self):
        """Test handling of negative values"""
        from app.main import TokenBucket
        
        bucket = TokenBucket(capacity=100, refill_rate_per_sec=-5)
        assert bucket.refill_rate_per_sec == 0  # Should be clamped to 0
        
        bucket.set_refill_rate(-10)
        assert bucket.refill_rate_per_sec == 0
        
        bucket.set_capacity(-50)
        assert bucket.capacity == 0
        assert bucket.tokens == 0


class TestBucketManagerLogic:
    """Unit tests for bucket manager implementations"""
    
    def test_in_memory_manager_scope_keys(self):
        """Test scope key generation for in-memory manager"""
        from app.main import InMemoryManager
        
        # Mock the globals that InMemoryManager needs
        with patch('app.main.buckets', {}), \
             patch('app.main.capacity_overrides', {}), \
             patch('app.main.refill_overrides', {}), \
             patch('app.main.lock', threading.Lock()), \
             patch('app.main.scope_defaults', return_value=(100, 5)):
            
            manager = InMemoryManager()
            
            assert manager._key("global", None) == "global"
            assert manager._key("user", "alice") == "user:alice"
            assert manager._key("team", "teamA") == "team:teamA"
            assert manager._key("role", "admin") == "role:admin"
    
    def test_redis_manager_scope_keys(self):
        """Test scope key generation for Redis manager"""
        from app.main import RedisBucketManager
        
        manager = RedisBucketManager(
            client=Mock(),
            default_caps={"user": 100, "team": 200},
            default_rates={"user": 5, "team": 10}
        )
        
        assert manager._scope_key("global", None) == "global"
        assert manager._scope_key("user", "alice") == "user:alice"
        assert manager._scope_key("team", "teamA") == "team:teamA"
        
        # Test key generation
        tokens_key, ts_key, rate_key, cap_key = manager._keys_for("user:alice")
        assert tokens_key == "bucket:user:alice:tokens"
        assert ts_key == "bucket:user:alice:ts"
        assert rate_key == "bucket:user:alice:rate"
        assert cap_key == "bucket:user:alice:cap"


class TestInputValidation:
    """Test input validation and edge cases"""
    
    def test_chat_request_validation(self):
        """Test ChatRequest model validation"""
        from app.main import ChatRequest, ChatMessage
        
        # Valid request
        request = ChatRequest(
            model="gpt-4o",
            messages=[ChatMessage(role="user", content="test")],
            max_tokens=100,
            requestedTokens=50,
            userId="test_user"
        )
        assert request.model == "gpt-4o"
        assert len(request.messages) == 1
        assert request.max_tokens == 100
        assert request.requestedTokens == 50
        assert request.userId == "test_user"
        
        # Minimal valid request
        minimal = ChatRequest(messages=[ChatMessage(role="user", content="hi")])
        assert minimal.model is None
        assert minimal.max_tokens is None
        assert minimal.requestedTokens is None
        assert minimal.userId is None
    
    def test_budget_config_validation(self):
        """Test BudgetConfig model validation"""
        from app.main import BudgetConfig
        
        # Valid configs
        global_config = BudgetConfig(scope="global", capacity=1000, refillPerSec=10)
        assert global_config.scope == "global"
        assert global_config.id is None
        
        user_config = BudgetConfig(scope="user", id="alice", capacity=100)
        assert user_config.scope == "user"
        assert user_config.id == "alice"
        assert user_config.refillPerSec is None
    
    def test_refill_config_validation(self):
        """Test RefillConfig model validation"""
        from app.main import RefillConfig
        
        config = RefillConfig(userId="test", refillPerSec=5.5)
        assert config.userId == "test"
        assert config.refillPerSec == 5.5


class TestEnvironmentConfiguration:
    """Test environment variable configuration"""
    
    def test_default_values(self, monkeypatch):
        """Test default environment values"""
        # Clear relevant env vars (including those from .env file)
        for var in ["BUCKET_CAPACITY", "BUCKET_REFILL_PER_SEC", "CAP_GLOBAL", "RATE_USER", "RATE_GLOBAL"]:
            monkeypatch.delenv(var, raising=False)
        
        # Mock dotenv module to prevent loading from .env file
        import dotenv
        monkeypatch.setattr(dotenv, "load_dotenv", lambda: None)
        
        import sys
        if "app.main" in sys.modules:
            del sys.modules["app.main"]
        
        main = __import__("app.main", fromlist=[""])
        
        assert main.DEFAULT_CAPACITY == 1000.0
        assert main.DEFAULT_REFILL_PER_SEC == 5.0
        assert main.DEFAULTS_CAP["global"] == 1000.0
        assert main.DEFAULTS_RATE["user"] == 5.0
    
    def test_custom_environment_values(self, monkeypatch):
        """Test custom environment configuration"""
        monkeypatch.setenv("BUCKET_CAPACITY", "500")
        monkeypatch.setenv("BUCKET_REFILL_PER_SEC", "2")
        monkeypatch.setenv("CAP_GLOBAL", "2000")
        monkeypatch.setenv("RATE_TEAM", "15")
        monkeypatch.setenv("ENFORCE_GLOBAL_SCOPE", "true")
        
        import sys
        if "app.main" in sys.modules:
            del sys.modules["app.main"]
        
        main = __import__("app.main", fromlist=[""])
        
        assert main.DEFAULT_CAPACITY == 500.0
        assert main.DEFAULT_REFILL_PER_SEC == 2.0
        assert main.DEFAULTS_CAP["global"] == 2000.0
        assert main.DEFAULTS_RATE["team"] == 15.0
        assert main.ENFORCE_GLOBAL is True
    
    def test_boolean_environment_parsing(self, monkeypatch):
        """Test boolean environment variable parsing"""
        test_cases = [
            ("true", True),
            ("1", True), 
            ("yes", True),
            ("on", True),
            ("TRUE", True),
            ("false", False),
            ("0", False),
            ("no", False),
            ("off", False),
            ("anything_else", False),
            ("", False)
        ]
        
        for value, expected in test_cases:
            monkeypatch.setenv("ENFORCE_GLOBAL_SCOPE", value)
            
            import sys
            if "app.main" in sys.modules:
                del sys.modules["app.main"]
            
            main = __import__("app.main", fromlist=[""])
            assert main.ENFORCE_GLOBAL == expected, f"Value '{value}' should parse to {expected}"


class TestErrorConditions:
    """Test various error conditions and edge cases"""
    
    def test_zero_capacity_bucket(self):
        """Test bucket with zero capacity"""
        from app.main import TokenBucket
        
        bucket = TokenBucket(capacity=0, refill_rate_per_sec=5)
        assert bucket.capacity == 0
        assert bucket.tokens == 0
        
        # Should not be able to remove any tokens
        assert bucket.try_remove(1) is False
        assert bucket.try_remove(0) is True  # Zero cost should succeed
    
    def test_very_large_numbers(self):
        """Test handling of very large numbers"""
        from app.main import TokenBucket
        
        bucket = TokenBucket(capacity=1e9, refill_rate_per_sec=1e6)
        assert bucket.capacity == 1e9
        assert bucket.refill_rate_per_sec == 1e6
        
        # Should handle large token removal
        assert bucket.try_remove(1e8) is True
        assert bucket.tokens == 9e8
    
    def test_time_going_backwards(self):
        """Test handling when system time goes backwards"""
        from app.main import TokenBucket
        
        bucket = TokenBucket(capacity=100, refill_rate_per_sec=10)
        bucket.try_remove(50)  # Use some tokens
        
        # Simulate time going backwards
        bucket.last_refill = time.time() + 100  # Future time
        initial_tokens = bucket.tokens
        
        bucket._refill()
        
        # Tokens should not decrease when time goes backwards
        assert bucket.tokens >= initial_tokens
    
    def test_fractional_tokens(self):
        """Test handling of fractional token amounts"""
        from app.main import TokenBucket
        
        bucket = TokenBucket(capacity=100.5, refill_rate_per_sec=1.5)
        assert bucket.capacity == 100.5
        assert bucket.refill_rate_per_sec == 1.5
        
        # Should handle fractional removal
        assert bucket.try_remove(10.5) is True
        assert bucket.tokens == 90.0
        
        # Test fractional refill
        bucket.last_refill = time.time() - 2.0  # 2 seconds ago
        bucket._refill()
        assert abs(bucket.tokens - 93.0) < 0.001  # 90 + (1.5 * 2), allow floating point error


class TestConcurrencyEdgeCases:
    """Test edge cases in concurrent scenarios"""
    
    def test_rapid_successive_requests(self):
        """Test handling of rapid successive requests"""
        from app.main import TokenBucket
        
        bucket = TokenBucket(capacity=100, refill_rate_per_sec=0)  # No refill
        
        # Rapid successive calls
        results = []
        for i in range(150):  # More than capacity
            results.append(bucket.try_remove(1))
        
        # Should succeed exactly 100 times
        success_count = sum(1 for r in results if r)
        assert success_count == 100
        
        # All subsequent should fail
        remaining_failures = results[100:]
        assert all(not r for r in remaining_failures)
    
    def test_race_condition_simulation(self):
        """Simulate race conditions in token removal"""
        from app.main import TokenBucket
        import threading
        import random
        
        bucket = TokenBucket(capacity=1000, refill_rate_per_sec=0)
        results = []
        
        def worker():
            for _ in range(10):
                # Random small delay to increase chance of race conditions
                time.sleep(random.uniform(0.001, 0.01))
                results.append(bucket.try_remove(10))
        
        # Start multiple threads
        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # Should have exactly 100 successful operations (1000 capacity / 10 tokens each)
        success_count = sum(1 for r in results if r)
        assert success_count == 100


@pytest.mark.parametrize("scope,sid,expected_key", [
    ("global", None, "global"),
    ("user", "alice", "user:alice"),
    ("team", "teamA", "team:teamA"),
    ("role", "admin", "role:admin"),
    ("user", "user:with:colons", "user:user:with:colons"),
])
def test_scope_key_formatting(scope, sid, expected_key):
    """Test scope key formatting with various inputs"""
    from app.main import RedisBucketManager
    
    manager = RedisBucketManager(Mock(), {}, {})
    assert manager._scope_key(scope, sid) == expected_key


@pytest.mark.parametrize("cost,capacity,expected", [
    (10, 100, True),   # Normal case
    (100, 100, True),  # Exact capacity
    (101, 100, False), # Over capacity  
    (0, 100, True),    # Zero cost
    (50, 0, False),    # Zero capacity
    (0, 0, True),      # Both zero
])
def test_token_removal_edge_cases(cost, capacity, expected):
    """Test token removal with various cost/capacity combinations"""
    from app.main import TokenBucket
    
    bucket = TokenBucket(capacity=capacity, refill_rate_per_sec=0)
    assert bucket.try_remove(cost) == expected
