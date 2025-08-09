import os
import time
import re
import httpx
import asyncio
import logging
import uuid
from typing import Dict, Optional, List, Tuple
from fastapi import FastAPI, Header, HTTPException, Response, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI
import redis
import threading
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("llmroute")

load_dotenv()

# Token estimation functions
def estimate_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    """
    Estimate token count for text. 
    Uses simple heuristic: ~4 characters per token for most models.
    This is conservative and will be adjusted based on real OpenAI usage.
    """
    # Different models have different token ratios
    char_per_token = {
        "gpt-4": 4.0,
        "gpt-4-turbo": 4.0, 
        "gpt-4o": 4.0,
        "gpt-4o-mini": 4.0,
        "gpt-3.5-turbo": 4.0,
        "text-davinci-003": 4.0,
    }
    
    ratio = char_per_token.get(model, 4.0)
    estimated = max(1, int(len(text) / ratio))
    
    # Add some overhead for chat format and system prompts
    if "chat" in model or "gpt" in model:
        estimated = int(estimated * 1.2)  # 20% overhead for chat formatting
    
    return estimated

def estimate_chat_tokens(messages: List[Dict], model: str = "gpt-3.5-turbo") -> Dict[str, int]:
    """Estimate tokens for a chat conversation, returning input and output estimates separately."""
    input_tokens = 0
    
    # Base overhead for chat format
    input_tokens += 4  # Base tokens for chat format
    
    for message in messages:
        role = message.get("role", "")
        content = message.get("content", "")
        
        # Role tokens (role names + formatting)
        input_tokens += 4  # For role formatting
        input_tokens += estimate_tokens(content, model)
    
    # Additional overhead for completion
    input_tokens += 3  # For assistant response preparation
    
    return {
        "input_tokens": input_tokens,
        "output_tokens": 150  # Default estimate, will be updated by max_tokens if provided
    }

class TokenBucket:
    def __init__(self, capacity: float, refill_rate_per_sec: float) -> None:
        self.capacity = capacity
        self.refill_rate_per_sec = max(0.0, refill_rate_per_sec)
        self.tokens = capacity
        self.last_refill = time.time()

    def set_refill_rate(self, rate: float) -> None:
        self.refill_rate_per_sec = max(0.0, rate)

    def _refill(self) -> None:
        now = time.time()
        elapsed = now - self.last_refill
        if elapsed > 0:
            self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate_per_sec)
            self.last_refill = now

    def try_remove(self, cost: float) -> bool:
        self._refill()
        if self.tokens >= cost:
            self.tokens -= cost
            return True
        return False

    def snapshot(self) -> Dict[str, float]:
        self._refill()
        return {
            "tokens": self.tokens,
            "capacity": self.capacity,
            "refillPerSec": self.refill_rate_per_sec,
        }

    def set_capacity(self, capacity: float) -> None:
        capacity = float(capacity)
        self.capacity = max(0.0, capacity)
        if self.tokens > self.capacity:
            self.tokens = self.capacity

# Redis-backed bucket manager (shared state across replicas)
class RedisBucketManager:
    # Multi-scope: enforce all scopes atomically (global/team/role/user)
    # KEYS: for each scope i, 4 keys [tokens_i, ts_i, rate_i, cap_i]
    # ARGV: n, cost, now, then pairs (def_cap_i, def_rate_i) repeated n times
    LUA_MULTI = """
    local n = tonumber(ARGV[1])
    local cost = tonumber(ARGV[2])
    local now = tonumber(ARGV[3])

    local ok = 1
    local idx = 4

    local tokens_list = {}
    local rate_list = {}
    local cap_list = {}

    -- First pass: refill and check
    for i = 0, n-1 do
      local base = (i*4)
      local tkey = KEYS[base+1]
      local tskey = KEYS[base+2]
      local rkey = KEYS[base+3]
      local ckey = KEYS[base+4]

      local def_cap = tonumber(ARGV[idx]); idx = idx + 1
      local def_rate = tonumber(ARGV[idx]); idx = idx + 1

      local cap_override = redis.call('GET', ckey)
      local capacity = def_cap
      if cap_override then capacity = tonumber(cap_override) end

      local rate_override = redis.call('GET', rkey)
      local refill_rate = def_rate
      if rate_override then refill_rate = tonumber(rate_override); if refill_rate < 0 then refill_rate = 0 end end

      local tokens = tonumber(redis.call('GET', tkey))
      local last = tonumber(redis.call('GET', tskey))
      if tokens == nil then tokens = capacity end
      if last == nil then last = now end

      local elapsed = now - last
      if elapsed < 0 then elapsed = 0 end
      tokens = math.min(capacity, tokens + (elapsed * refill_rate))

      if cost > tokens then ok = 0 end

      tokens_list[i+1] = tokens
      rate_list[i+1] = refill_rate
      cap_list[i+1] = capacity
    end

    -- Second pass: if ok, subtract and write back tokens and ts
    if ok == 1 then
      for i = 0, n-1 do
        local base = (i*4)
        local tkey = KEYS[base+1]
        local tskey = KEYS[base+2]
        local tokens = tokens_list[i+1] - cost
        redis.call('SET', tkey, tokens)
        redis.call('SET', tskey, now)
      end
    else
      -- Even if not ok, write back refreshed tokens and ts
      for i = 0, n-1 do
        local base = (i*4)
        local tkey = KEYS[base+1]
        local tskey = KEYS[base+2]
        redis.call('SET', tkey, tokens_list[i+1])
        redis.call('SET', tskey, now)
      end
    end

    return ok
    """

    # Lua script for atomic token adjustment (add/subtract tokens)
    LUA_ADJUST = """
    local n = tonumber(ARGV[1])
    local adjustment = tonumber(ARGV[2])
    local now = tonumber(ARGV[3])

    local idx = 4
    local ok = 1

    -- For each scope, adjust tokens atomically
    for i = 0, n-1 do
      local base = (i*4)
      local tkey = KEYS[base+1]
      local tskey = KEYS[base+2]
      local rkey = KEYS[base+3]
      local ckey = KEYS[base+4]

      local def_cap = tonumber(ARGV[idx]); idx = idx + 1
      local def_rate = tonumber(ARGV[idx]); idx = idx + 1

      local cap_override = redis.call('GET', ckey)
      local capacity = def_cap
      if cap_override then capacity = tonumber(cap_override) end

      local rate_override = redis.call('GET', rkey)
      local refill_rate = def_rate
      if rate_override then refill_rate = tonumber(rate_override); if refill_rate < 0 then refill_rate = 0 end end

      local tokens = tonumber(redis.call('GET', tkey))
      local last = tonumber(redis.call('GET', tskey))
      if tokens == nil then tokens = capacity end
      if last == nil then last = now end

      -- Refill before adjustment
      local elapsed = now - last
      if elapsed < 0 then elapsed = 0 end
      tokens = math.min(capacity, tokens + (elapsed * refill_rate))

      -- Apply adjustment
      if adjustment > 0 then
        -- Removing more tokens (we underestimated)
        if tokens >= adjustment then
          tokens = tokens - adjustment
        else
          ok = 0  -- Not enough tokens to adjust
        end
      else
        -- Adding tokens back (we overestimated)
        tokens = math.min(capacity, tokens + math.abs(adjustment))
      end

      -- Write back
      redis.call('SET', tkey, tokens)
      redis.call('SET', tskey, now)
    end

    return ok
    """

    def __init__(self, client: "redis.Redis", default_caps: Dict[str, float], default_rates: Dict[str, float]) -> None:
        self.client = client
        self.default_caps = default_caps
        self.default_rates = default_rates

    def _scope_key(self, scope: str, sid: Optional[str]) -> str:
        if scope == "global":
            return "global"
        return f"{scope}:{sid}"

    def _keys_for(self, scope_key: str) -> Tuple[str, str, str, str]:
        prefix = f"bucket:{scope_key}"
        return f"{prefix}:tokens", f"{prefix}:ts", f"{prefix}:rate", f"{prefix}:cap"

    def set_budget(self, scope: str, sid: Optional[str], capacity: Optional[float], refill: Optional[float]) -> Dict[str, float]:
        sk = self._scope_key(scope, sid)
        t_key, ts_key, r_key, c_key = self._keys_for(sk)
        if capacity is not None:
            self.client.set(c_key, max(0.0, float(capacity)))
        if refill is not None:
            self.client.set(r_key, max(0.0, float(refill)))
        # Return a snapshot-like view (by running multi with n=1 and cost=0)
        now = time.time()
        def_cap = self.default_caps.get(scope, 1000.0)
        def_rate = self.default_rates.get(scope, 5.0)
        ok = self.client.eval(self.LUA_MULTI, 4, t_key, ts_key, r_key, c_key, 1, 0.0, now, def_cap, def_rate)
        # we don't need ok here; snapshot is in keys now
        tokens = float(self.client.get(t_key) or def_cap)
        rate = float(self.client.get(r_key) or def_rate)
        cap = float(self.client.get(c_key) or def_cap)
        return {"tokens": tokens, "capacity": cap, "refillPerSec": rate}

    def try_consume_scoped(self, scopes: List[Tuple[str, Optional[str]]], cost: float) -> Tuple[bool, Dict[str, Dict[str, float]]]:
        keys: List[str] = []
        argv: List[float] = []
        now = time.time()
        for (scope, sid) in scopes:
            sk = self._scope_key(scope, sid)
            t, ts, r, c = self._keys_for(sk)
            keys.extend([t, ts, r, c])
            argv.extend([self.default_caps.get(scope, 1000.0), self.default_rates.get(scope, 5.0)])
        ok = int(self.client.eval(self.LUA_MULTI, len(keys), *keys, len(scopes), float(cost), now, *argv))
        # Build snapshots by reading keys back (post-refill state)
        snapshots: Dict[str, Dict[str, float]] = {}
        for (scope, sid) in scopes:
            sk = self._scope_key(scope, sid)
            t, ts, r, c = self._keys_for(sk)
            tokens = float(self.client.get(t) or self.default_caps.get(scope, 1000.0))
            rate = float(self.client.get(r) or self.default_rates.get(scope, 5.0))
            cap = float(self.client.get(c) or self.default_caps.get(scope, 1000.0))
            snapshots[sk] = {"tokens": tokens, "capacity": cap, "refillPerSec": rate}
        return bool(ok), snapshots

    def adjust_scoped(self, scopes: List[Tuple[str, Optional[str]]], adjustment: float) -> bool:
        """Atomically adjust tokens across multiple scopes."""
        keys: List[str] = []
        argv: List[float] = []
        now = time.time()
        for (scope, sid) in scopes:
            sk = self._scope_key(scope, sid)
            t, ts, r, c = self._keys_for(sk)
            keys.extend([t, ts, r, c])
            argv.extend([self.default_caps.get(scope, 1000.0), self.default_rates.get(scope, 5.0)])
        ok = int(self.client.eval(self.LUA_ADJUST, len(keys), *keys, len(scopes), float(adjustment), now, *argv))
        return bool(ok)

    # Back-compat single-scope methods
    def set_refill_rate(self, user_id: str, rate: float) -> None:
        self.set_budget("user", user_id, None, rate)

    def try_remove(self, user_id: str, cost: float) -> Tuple[bool, Dict[str, float]]:
        ok, snaps = self.try_consume_scoped([("user", user_id)], cost)
        return ok, snaps.get(f"user:{user_id}", {})

    def set_bucket(self, scope: str, sid: Optional[str], capacity: float, refill_rate: float) -> None:
        """Set bucket capacity and refill rate, and update current tokens."""
        sk = self._scope_key(scope, sid)
        t, ts, r, c = self._keys_for(sk)
        
        # Set the new capacity and refill rate
        self.client.set(c, str(capacity))
        self.client.set(r, str(refill_rate))
        
        # Get current tokens and cap to new capacity if needed
        current_tokens = float(self.client.get(t) or capacity)
        if current_tokens > capacity:
            current_tokens = capacity
        self.client.set(t, str(current_tokens))

class RefillConfig(BaseModel):
    userId: str
    refillPerSec: float

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: Optional[str] = None
    messages: List[ChatMessage]
    max_tokens: Optional[int] = None
    requestedTokens: Optional[int] = None
    userId: Optional[str] = None

class BudgetConfig(BaseModel):
    scope: str  # global|team|role|user
    id: Optional[str] = None  # required for team/role/user
    capacity: Optional[float] = None
    refillPerSec: Optional[float] = None

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
DEFAULT_CAPACITY = float(os.getenv("BUCKET_CAPACITY", "1000"))
DEFAULT_REFILL_PER_SEC = float(os.getenv("BUCKET_REFILL_PER_SEC", "5"))
# Per-scope defaults (fallback to globals if not provided)
DEFAULTS_CAP = {
    "global": float(os.getenv("CAP_GLOBAL", str(DEFAULT_CAPACITY))),
    "team": float(os.getenv("CAP_TEAM", str(DEFAULT_CAPACITY))),
    "rolegroup": float(os.getenv("CAP_ROLEGROUP", str(DEFAULT_CAPACITY))),
    "role": float(os.getenv("CAP_ROLE", str(DEFAULT_CAPACITY))),
    "user": float(os.getenv("CAP_USER", str(DEFAULT_CAPACITY))),
}
DEFAULTS_RATE = {
    "global": float(os.getenv("RATE_GLOBAL", str(DEFAULT_REFILL_PER_SEC))),
    "team": float(os.getenv("RATE_TEAM", str(DEFAULT_REFILL_PER_SEC))),
    "rolegroup": float(os.getenv("RATE_ROLEGROUP", str(DEFAULT_REFILL_PER_SEC))),
    "role": float(os.getenv("RATE_ROLE", str(DEFAULT_REFILL_PER_SEC))),
    "user": float(os.getenv("RATE_USER", str(DEFAULT_REFILL_PER_SEC))),
}
ENFORCE_GLOBAL = os.getenv("ENFORCE_GLOBAL_SCOPE", "false").lower() in ("1", "true", "yes", "on")
REDIS_URL = os.getenv("REDIS_URL")

# Global sync state
last_openai_sync = 0
SYNC_INTERVAL = 300  # Sync every 5 minutes
ENABLE_BACKGROUND_SYNC = os.getenv("ENABLE_BACKGROUND_SYNC", "true").lower() in ("1", "true", "yes", "on")
TOKEN_ADJUSTMENT_THRESHOLD = float(os.getenv("TOKEN_ADJUSTMENT_THRESHOLD", "0.1"))  # 10% default

def background_openai_sync():
    """Periodically sync with OpenAI rate limits in background"""
    global last_openai_sync
    try:
        current_time = time.time()
        if current_time - last_openai_sync > SYNC_INTERVAL:
            logger.info("Background sync with OpenAI rate limits...")
            sync_openai_limits()
            last_openai_sync = current_time
    except Exception as e:
        logger.warning(f"Background OpenAI sync failed: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    sync_task = None
    executor = None
    
    if ENABLE_BACKGROUND_SYNC:
        # Startup
        executor = ThreadPoolExecutor(max_workers=1)
        
        async def periodic_sync():
            while True:
                await asyncio.sleep(SYNC_INTERVAL)
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(executor, background_openai_sync)
        
        # Start background sync task
        sync_task = asyncio.create_task(periodic_sync())
    
    yield  # Application runs here
    
    # Shutdown
    if sync_task:
        sync_task.cancel()
        try:
            await sync_task
        except asyncio.CancelledError:
            pass
    if executor:
        executor.shutdown(wait=False)

app = FastAPI(title="tokenguard", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# In-memory bucket manager (for when Redis is not available and testing)
buckets: Dict[str, TokenBucket] = {}
capacity_overrides: Dict[str, float] = {}
refill_overrides: Dict[str, float] = {}
lock = threading.Lock()

def scope_defaults(scope: str) -> Tuple[float, float]:
    cap = DEFAULTS_CAP.get(scope, DEFAULT_CAPACITY)
    rate = DEFAULTS_RATE.get(scope, DEFAULT_REFILL_PER_SEC)
    return cap, rate

def parse_time_duration(duration_str: str) -> float:
    """Parse OpenAI duration strings like '1s', '6m0s', '1h30m' to seconds."""
    if not duration_str:
        return 0.0
    
    # Remove whitespace and convert to lowercase
    duration_str = duration_str.strip().lower()
    
    # Parse components like 1h30m45s
    pattern = r'(?:(\d+)h)?(?:(\d+)m)?(?:(\d+)s)?'
    match = re.match(pattern, duration_str)
    
    if not match:
        return 0.0
    
    hours = int(match.group(1) or 0)
    minutes = int(match.group(2) or 0) 
    seconds = int(match.group(3) or 0)
    
    return hours * 3600 + minutes * 60 + seconds

def update_global_limits_from_openai_headers(headers: Dict[str, str]):
    """Update global bucket limits based on OpenAI rate limit headers."""
    try:
        # Parse OpenAI rate limit headers
        token_limit = headers.get('x-ratelimit-limit-tokens')
        token_remaining = headers.get('x-ratelimit-remaining-tokens')
        token_reset = headers.get('x-ratelimit-reset-tokens')
        request_limit = headers.get('x-ratelimit-limit-requests')
        request_remaining = headers.get('x-ratelimit-remaining-requests')
        request_reset = headers.get('x-ratelimit-reset-requests')
        
        if token_limit and token_remaining and token_reset:
            # Update global token bucket capacity to match OpenAI limits
            token_capacity = float(token_limit)
            tokens_remaining = float(token_remaining)
            reset_seconds = parse_time_duration(token_reset)
            
            # Calculate refill rate: tokens consumed / time until reset
            if reset_seconds > 0:
                tokens_consumed = token_capacity - tokens_remaining
                refill_rate = tokens_consumed / reset_seconds
            else:
                refill_rate = DEFAULTS_RATE.get("global", DEFAULT_REFILL_PER_SEC)
            
            # Update the global bucket configuration
            bucket_manager.set_bucket("global", None, token_capacity, refill_rate)
            
            # Also update current token count to match OpenAI's view
            bucket = bucket_manager.get("global", None)
            bucket.tokens = tokens_remaining
            
            logger.info(f"Updated global limits from OpenAI: capacity={token_capacity}, "
                       f"remaining={tokens_remaining}, refill_rate={refill_rate:.2f}/sec")
                  
    except Exception as e:
        logger.warning(f"Failed to parse OpenAI rate limit headers: {e}")
        # Continue without updating limits

class InMemoryManager:
    def _key(self, scope: str, sid: Optional[str]) -> str:
        if scope == "global":
            return "global"
        return f"{scope}:{sid}"
    def get(self, scope: str, sid: Optional[str]) -> TokenBucket:
        sk = self._key(scope, sid)
        b = buckets.get(sk)
        if not b:
            cap_def, rate_def = scope_defaults(scope)
            cap = capacity_overrides.get(sk, cap_def)
            rate = refill_overrides.get(sk, rate_def)
            b = TokenBucket(cap, rate)
            buckets[sk] = b
        return b
    def set_budget(self, scope: str, sid: Optional[str], capacity: Optional[float], refill: Optional[float]) -> Dict[str, float]:
        sk = self._key(scope, sid)
        with lock:
            b = self.get(scope, sid)
            if capacity is not None:
                capacity_overrides[sk] = max(0.0, float(capacity))
                b.set_capacity(capacity_overrides[sk])
            if refill is not None:
                refill_overrides[sk] = max(0.0, float(refill))
                b.set_refill_rate(refill_overrides[sk])
            return b.snapshot()
    def try_consume_scoped(self, scopes: List[Tuple[str, Optional[str]]], cost: float) -> Tuple[bool, Dict[str, Dict[str, float]]]:
        with lock:
            # Refill and check
            bs: List[Tuple[str, TokenBucket]] = []
            for (scope, sid) in scopes:
                sk = self._key(scope, sid)
                b = self.get(scope, sid)
                bs.append((sk, b))
            if any(b.snapshot()["tokens"] < cost for _, b in bs):
                # Even if not ok, we already refreshed via snapshot()
                return False, {sk: b.snapshot() for sk, b in bs}
            # All ok, subtract
            for _, b in bs:
                b.try_remove(cost)
            return True, {sk: b.snapshot() for sk, b in bs}
    
    def adjust_scoped(self, scopes: List[Tuple[str, Optional[str]]], adjustment: float) -> bool:
        """Atomically adjust tokens across multiple scopes."""
        with lock:
            bs: List[Tuple[str, TokenBucket]] = []
            for (scope, sid) in scopes:
                sk = self._key(scope, sid)
                b = self.get(scope, sid)
                bs.append((sk, b))
            
            # Check if adjustment is possible for all scopes
            if adjustment > 0:  # Removing more tokens
                for _, b in bs:
                    b._refill()  # Ensure up-to-date
                    if b.tokens < adjustment:
                        return False
            
            # Apply adjustment to all scopes
            for _, b in bs:
                if adjustment > 0:
                    b.try_remove(adjustment)
                else:
                    b.tokens = min(b.capacity, b.tokens + abs(adjustment))
            
            return True
    # Back-compat
    def set_refill_rate(self, user_id: str, rate: float) -> None:
        self.set_budget("user", user_id, None, rate)
    def try_remove(self, user_id: str, cost: float) -> Tuple[bool, Dict[str, float]]:
        ok, snaps = self.try_consume_scoped([("user", user_id)], cost)
        return ok, snaps.get(f"user:{user_id}", {})
    
    def set_bucket(self, scope: str, sid: Optional[str], capacity: float, refill_rate: float) -> None:
        """Set bucket capacity and refill rate, and update current tokens."""
        sk = self._key(scope, sid)
        with lock:
            # Create or get existing bucket
            b = self.get(scope, sid)
            # Update capacity and refill rate
            capacity_overrides[sk] = capacity
            refill_overrides[sk] = refill_rate
            b.set_capacity(capacity)
            b.set_refill_rate(refill_rate)

# Choose backend: Redis if configured, else in-memory
redis_client: Optional[redis.Redis] = None
bucket_manager = None
if REDIS_URL:
    redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True)
    bucket_manager = RedisBucketManager(redis_client, DEFAULTS_CAP, DEFAULTS_RATE)
else:
    bucket_manager = InMemoryManager()

@app.get("/")
def health():
    return {"name": "tokenguard", "status": "ok", "time": time.time(), "backend": "redis" if REDIS_URL else "memory"}

@app.post("/config/refill-rate")
def set_refill(cfg: RefillConfig):
    if cfg.refillPerSec < 0:
        raise HTTPException(status_code=400, detail="refillPerSec must be non-negative")
    # Set and return snapshot for user scope
    snap = bucket_manager.set_budget("user", cfg.userId, None, cfg.refillPerSec)
    return {"ok": True, "userId": cfg.userId, "refillPerSec": cfg.refillPerSec, "bucket": snap}

@app.post("/config/budget")
def set_budget(cfg: BudgetConfig):
    scope = cfg.scope
    sid = cfg.id
    if scope not in ("global", "team", "rolegroup", "role", "user"):
        raise HTTPException(status_code=400, detail="invalid scope")
    if scope != "global" and not sid:
        raise HTTPException(status_code=400, detail="id required for non-global scope")
    snap = bucket_manager.set_budget(scope, sid, cfg.capacity, cfg.refillPerSec)
    key = "global" if scope == "global" else f"{scope}:{sid}"
    return {"ok": True, "scope": key, "bucket": snap}

@app.get("/sync/openai-limits")
def sync_openai_limits():
    """Manually sync global rate limits with OpenAI API by making a test request."""
    # Skip sync in test environments
    if OPENAI_API_KEY == "test-key" or not OPENAI_API_KEY:
        return {
            "status": "skipped",
            "reason": "test environment detected",
            "global_bucket": {
                "tokens": bucket_manager.get("global", None).tokens,
                "capacity": bucket_manager.get("global", None).capacity,
                "refill_per_sec": bucket_manager.get("global", None).refill_rate_per_sec
            }
        }
    
    try:
        # Make a minimal test request to get current rate limit headers
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json"
        }
        
        test_request = {
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 1
        }
        
        with httpx.Client() as client:
            openai_response = client.post(
                "https://api.openai.com/v1/chat/completions",
                json=test_request,
                headers=headers,
                timeout=10.0
            )
            
            # Update global limits from OpenAI headers
            update_global_limits_from_openai_headers(dict(openai_response.headers))
            
            # Return current state
            global_bucket = bucket_manager.get("global", None)
            
            return {
                "success": True,
                "message": "Synced with OpenAI rate limits",
                "openai_headers": {
                    "x-ratelimit-limit-tokens": openai_response.headers.get("x-ratelimit-limit-tokens"),
                    "x-ratelimit-remaining-tokens": openai_response.headers.get("x-ratelimit-remaining-tokens"),
                    "x-ratelimit-reset-tokens": openai_response.headers.get("x-ratelimit-reset-tokens"),
                    "x-ratelimit-limit-requests": openai_response.headers.get("x-ratelimit-limit-requests"),
                    "x-ratelimit-remaining-requests": openai_response.headers.get("x-ratelimit-remaining-requests"),
                    "x-ratelimit-reset-requests": openai_response.headers.get("x-ratelimit-reset-requests"),
                },
                "global_bucket": {
                    "tokens": global_bucket.tokens,
                    "capacity": global_bucket.capacity,
                    "refill_per_sec": global_bucket.refill_rate_per_sec
                }
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to sync with OpenAI: {str(e)}")

@app.get("/buckets/global")
def get_global_bucket():
    """Get global bucket status."""
    bucket = bucket_manager.get("global", None)
    return {"scope": "global", "bucket": bucket.snapshot()}

@app.get("/buckets/{scope}/{id}")
def get_bucket(scope: str, id: str):
    """Get bucket status for a specific scope and ID."""
    if scope not in ("team", "rolegroup", "role", "user"):
        raise HTTPException(status_code=400, detail="invalid scope")
    
    bucket = bucket_manager.get(scope, id)
    return {"scope": f"{scope}:{id}", "bucket": bucket.snapshot()}

@app.post("/v1/chat/completions")
def chat(request: Request, body: ChatRequest, response: Response, x_user_id: Optional[str] = Header(default=None), x_team_id: Optional[str] = Header(default=None), x_role: Optional[str] = Header(default=None), x_llm_rolegroup: Optional[str] = Header(default=None)):
    # Generate request ID for tracing
    request_id = str(uuid.uuid4())[:8]
    response.headers["x-request-id"] = request_id
    
    user_id = x_user_id or body.userId or "anon"
    team_id = x_team_id
    role = x_role
    role_group = x_llm_rolegroup
    
    # Estimate tokens if not provided
    if body.requestedTokens is not None:
        estimated_tokens = float(body.requestedTokens)
        input_tokens_estimate = estimated_tokens * 0.7  # Assume 70% input, 30% output
        output_tokens_estimate = estimated_tokens * 0.3
    else:
        # Estimate based on input messages + expected output
        token_estimates = estimate_chat_tokens([m.model_dump() for m in body.messages], body.model or "gpt-4o-mini")
        input_tokens_estimate = token_estimates["input_tokens"]
        output_tokens_estimate = body.max_tokens or token_estimates["output_tokens"]
        estimated_tokens = input_tokens_estimate + output_tokens_estimate
    
    cost = max(1.0, estimated_tokens)

    scopes: List[Tuple[str, Optional[str]]] = [("user", user_id)]
    if role:
        scopes.insert(0, ("role", role))
    if role_group:
        scopes.insert(0, ("rolegroup", role_group))
    if team_id:
        scopes.insert(0, ("team", team_id))
    if ENFORCE_GLOBAL:
        scopes.insert(0, ("global", None))

    ok, snaps = bucket_manager.try_consume_scoped(scopes, cost)
    if not ok:
        logger.warning(f"Rate limited - request_id={request_id}, user_id={user_id}, scopes={[f'{s}:{sid}' for s, sid in scopes]}, cost={cost:.1f}")
        # Build rate limit headers for 429 response
        rate_limit_headers = {"Retry-After": "1"}
        for scope_key, bucket_snap in snaps.items():
            if bucket_snap["tokens"] < cost:
                # Calculate time until this bucket refills enough tokens
                tokens_needed = cost - bucket_snap["tokens"]
                time_to_refill = tokens_needed / bucket_snap["refillPerSec"] if bucket_snap["refillPerSec"] > 0 else 3600
                rate_limit_headers["x-ratelimit-reset-tokens"] = f"{int(time_to_refill)}s"
                rate_limit_headers["x-ratelimit-remaining-tokens"] = str(int(bucket_snap["tokens"]))
                rate_limit_headers["x-ratelimit-limit-tokens"] = str(int(bucket_snap["capacity"]))
                break
        
        raise HTTPException(
            status_code=429,
            detail={"error": "rate_limited", "buckets": snaps},
            headers=rate_limit_headers
        )

    try:
        # Try direct HTTP request first to capture rate limit headers
        request_data = {
            "model": body.model or "gpt-4o-mini",
            "messages": [m.model_dump() for m in body.messages],
            "max_tokens": body.max_tokens
        }
        
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json"
        }
        
        try:
            with httpx.Client() as client:
                openai_response = client.post(
                    "https://api.openai.com/v1/chat/completions",
                    json=request_data,
                    headers=headers,
                    timeout=30.0
                )
                
                # Update global limits from OpenAI headers
                update_global_limits_from_openai_headers(dict(openai_response.headers))
                
                # Parse the response
                if openai_response.status_code == 200:
                    resp_data = openai_response.json()
                    
                    # Get actual token usage from OpenAI response
                    actual_tokens = None
                    actual_input_tokens = None
                    actual_output_tokens = None
                    
                    if "usage" in resp_data:
                        actual_tokens = resp_data["usage"].get("total_tokens")
                        actual_input_tokens = resp_data["usage"].get("prompt_tokens")
                        actual_output_tokens = resp_data["usage"].get("completion_tokens")
                    
                    # Adjust token buckets if actual usage differs significantly from estimate
                    if actual_tokens and abs(actual_tokens - cost) > cost * TOKEN_ADJUSTMENT_THRESHOLD:
                        adjustment = actual_tokens - cost
                        logger.info(f"Token adjustment - request_id={request_id}, user_id={user_id}, estimated={cost:.1f} (input: {input_tokens_estimate:.1f}, output: {output_tokens_estimate:.1f}), actual={actual_tokens} (input: {actual_input_tokens}, output: {actual_output_tokens}), diff={adjustment:.1f}")
                        # Apply adjustment to all consumed scopes atomically
                        adjust_success = bucket_manager.adjust_scoped(scopes, adjustment)
                        if not adjust_success:
                            logger.warning(f"Token adjustment failed - request_id={request_id}, adjustment={adjustment:.1f}")
                    elif actual_tokens:
                        logger.debug(f"Token usage within threshold - request_id={request_id}, estimated={cost:.1f}, actual={actual_tokens} (input: {actual_input_tokens}, output: {actual_output_tokens})")
                    
                    # Include rate limit info in response
                    rate_limit_headers = {
                        "x-ratelimit-limit-tokens": openai_response.headers.get("x-ratelimit-limit-tokens"),
                        "x-ratelimit-remaining-tokens": openai_response.headers.get("x-ratelimit-remaining-tokens"),
                        "x-ratelimit-reset-tokens": openai_response.headers.get("x-ratelimit-reset-tokens"),
                        "x-ratelimit-limit-requests": openai_response.headers.get("x-ratelimit-limit-requests"),
                        "x-ratelimit-remaining-requests": openai_response.headers.get("x-ratelimit-remaining-requests"),
                        "x-ratelimit-reset-requests": openai_response.headers.get("x-ratelimit-reset-requests"),
                    }
                    
                    # Add OpenAI rate limit headers to HTTP response
                    for header_name, header_value in rate_limit_headers.items():
                        if header_value:
                            response.headers[f"x-openai-{header_name}"] = header_value
                    
                    # Add our own rate limit headers to the response
                    user_bucket = snaps.get(f"user:{user_id}")
                    if user_bucket:
                        response.headers["x-app-ratelimit-limit-tokens"] = str(int(user_bucket["capacity"]))
                        response.headers["x-app-ratelimit-remaining-tokens"] = str(int(user_bucket["tokens"]))
                        if user_bucket["refillPerSec"] > 0:
                            time_to_full = (user_bucket["capacity"] - user_bucket["tokens"]) / user_bucket["refillPerSec"]
                            response.headers["x-app-ratelimit-reset-tokens"] = f"{int(time_to_full)}s"
                    
                    logger.info(f"Chat completion success - request_id={request_id}, user_id={user_id}, model={body.model or 'gpt-4o-mini'}, estimated_tokens={cost:.1f}, actual_tokens={actual_tokens or 'unknown'}")
                    
                    return {
                        "success": True, 
                        "data": resp_data, 
                        "buckets": snaps,
                        "openai_rate_limits": rate_limit_headers
                    }
                else:
                    logger.error(f"OpenAI API error - request_id={request_id}, status={openai_response.status_code}, response={openai_response.text}")
                    raise HTTPException(status_code=openai_response.status_code, detail=openai_response.text)
        
        except Exception as http_error:
            # Fallback to regular OpenAI client (for testing/mocked scenarios)
            try:
                client = OpenAI(api_key=OPENAI_API_KEY)
                model = body.model or "gpt-4o-mini"
                messages = [m.model_dump() for m in body.messages]
                max_tokens = body.max_tokens
                resp = client.chat.completions.create(model=model, messages=messages, max_tokens=max_tokens)
                
                # Add our own rate limit headers to the response
                user_bucket = snaps.get(f"user:{user_id}")
                if user_bucket:
                    response.headers["x-app-ratelimit-limit-tokens"] = str(int(user_bucket["capacity"]))
                    response.headers["x-app-ratelimit-remaining-tokens"] = str(int(user_bucket["tokens"]))
                    if user_bucket["refillPerSec"] > 0:
                        time_to_full = (user_bucket["capacity"] - user_bucket["tokens"]) / user_bucket["refillPerSec"]
                        response.headers["x-app-ratelimit-reset-tokens"] = f"{int(time_to_full)}s"
                
                return {"success": True, "data": resp, "buckets": snaps}
            except Exception as fallback_error:
                logger.error(f"Fallback OpenAI client error - request_id={request_id}: {fallback_error}")
                raise HTTPException(status_code=500, detail=f"openai_fallback_error: {str(fallback_error)}") from fallback_error
                
    except Exception as e:
        logger.error(f"Chat endpoint error - request_id={request_id}: {e}")
        raise HTTPException(status_code=500, detail=f"chat_error: {str(e)}") from e
