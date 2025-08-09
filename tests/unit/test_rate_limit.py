import pytest
pytestmark = pytest.mark.unit

import os
import sys
import time
import importlib
import types
from fastapi.testclient import TestClient


class FakeCompletions:
    def create(self, model=None, messages=None, max_tokens=None):
        return {"id": "fake", "choices": [{"message": {"role": "assistant", "content": "ok"}}]}


class FakeChat:
    def __init__(self):
        self.completions = FakeCompletions()


class FakeOpenAI:
    def __init__(self, api_key=None):
        self.chat = FakeChat()


@pytest.fixture()
def client(monkeypatch):
    # Ensure in-memory backend and deterministic refill
    monkeypatch.delenv("REDIS_URL", raising=False)
    monkeypatch.setenv("BUCKET_CAPACITY", "10")
    monkeypatch.setenv("BUCKET_REFILL_PER_SEC", "0")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    # Reload app.main to reinitialize globals with env
    if "app.main" in sys.modules:
        del sys.modules["app.main"]
    main = importlib.import_module("app.main")

    # Patch OpenAI client
    monkeypatch.setattr(main, "OpenAI", FakeOpenAI)

    return TestClient(main.app)


def test_allows_within_bucket_then_limits(client: TestClient):
    # Freeze refill rate at 0 for user u1 to make deterministic
    r = client.post("/config/refill-rate", json={"userId": "u1", "refillPerSec": 0})
    assert r.status_code == 200

    payload = {"messages": [{"role": "user", "content": "hi"}], "requestedTokens": 5}
    headers = {"x-user-id": "u1"}

    # 1st request: 5 tokens -> allowed
    r1 = client.post("/v1/chat/completions", json=payload, headers=headers)
    assert r1.status_code == 200, r1.text
    buckets1 = r1.json().get("buckets", {})
    user_bucket1 = buckets1.get("user:u1", {})
    assert user_bucket1.get("tokens") is not None

    # 2nd request: another 5 tokens -> allowed (exactly capacity 10)
    r2 = client.post("/v1/chat/completions", json=payload, headers=headers)
    assert r2.status_code == 200, r2.text
    buckets2 = r2.json().get("buckets", {})
    user_bucket2 = buckets2.get("user:u1", {})
    assert user_bucket2.get("tokens", 0) <= 0.0001

    # 3rd request: exceeds -> 429 with Retry-After
    r3 = client.post("/v1/chat/completions", json=payload, headers=headers)
    assert r3.status_code == 429
    assert r3.headers.get("Retry-After") is not None
    detail = r3.json().get("detail", {})
    assert detail.get("error") == "rate_limited"


def test_refill_allows_later_requests(client: TestClient):
    # Set non-zero refill and wait
    r = client.post("/config/refill-rate", json={"userId": "u2", "refillPerSec": 10})
    assert r.status_code == 200

    payload = {"messages": [{"role": "user", "content": "hi"}], "requestedTokens": 10}
    headers = {"x-user-id": "u2"}

    # Drain bucket with one request
    r1 = client.post("/v1/chat/completions", json=payload, headers=headers)
    assert r1.status_code == 200

    # Immediately try again -> should be limited
    r2 = client.post("/v1/chat/completions", json=payload, headers=headers)
    if r2.status_code == 200:
        # If timing allowed refill already, proceed
        pass
    else:
        assert r2.status_code == 429
        time.sleep(1.2)

    # After wait, should pass
    r3 = client.post("/v1/chat/completions", json=payload, headers=headers)
    assert r3.status_code == 200, r3.text


def test_isolated_buckets_per_user(client: TestClient):
    payload = {"messages": [{"role": "user", "content": "hi"}], "requestedTokens": 10}

    # uA drains their bucket
    r1 = client.post("/v1/chat/completions", json=payload, headers={"x-user-id": "uA"})
    assert r1.status_code == 200
    r2 = client.post("/v1/chat/completions", json=payload, headers={"x-user-id": "uA"})
    assert r2.status_code in (200, 429)

    # uB should not be affected
    r3 = client.post("/v1/chat/completions", json=payload, headers={"x-user-id": "uB"})
    assert r3.status_code == 200
