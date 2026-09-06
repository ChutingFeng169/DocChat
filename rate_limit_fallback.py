"""
rate_limit_fallback.py
=====================
Rate-limit handling for the primary LLM call with exponential-backoff retries
and an automatic fallback to a secondary endpoint after MAX_RETRIES failures.

Uses tenacity (the real @retry engine) via its `Retrying` controller. The retry
controller is constructed *per call* on purpose, so the wait/stop/sleep settings
are read from module-level globals at call time -- this lets tests patch them
(SLEEP recorder, tiny backoff) without touching production behaviour.

Env (all optional; sensible defaults fall back to the primary DeepSeek proxy):
    OPENAI_API_KEY / OPENAI_BASE_URL / LLM_MODEL   -> primary
    FALLBACK_API_KEY / FALLBACK_BASE_URL / FALLBACK_MODEL -> fallback (defaults to primary)
    BACKOFF_BASE / BACKOFF_MIN / BACKOFF_MAX       -> exponential backoff seconds
"""

import os
import time

import httpx
from dotenv import load_dotenv
from tenacity import (
    Retrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

# Load .env (if present) so the module works standalone, not just when imported by app.py.
load_dotenv()

# ---------------------------------------------------------------- env config
PRIMARY_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
PRIMARY_API_KEY = os.getenv("OPENAI_API_KEY", "")
PRIMARY_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")

FALLBACK_BASE_URL = os.getenv("FALLBACK_BASE_URL", PRIMARY_BASE_URL).rstrip("/")
FALLBACK_API_KEY = os.getenv("FALLBACK_API_KEY", PRIMARY_API_KEY)
FALLBACK_MODEL = os.getenv("FALLBACK_MODEL", PRIMARY_MODEL)

# ----------------------------------------------------- tunable retry settings
# These are read at call time inside answer_with_retry_and_fallback(), so tests
# can monkeypatch them (e.g. an instant sleep recorder, small backoff).
MAX_RETRIES = 10
BACKOFF_BASE = float(os.getenv("BACKOFF_BASE", "1"))     # multiplier (seconds)
BACKOFF_MIN = float(os.getenv("BACKOFF_MIN", "1"))        # floor wait (seconds)
BACKOFF_MAX = float(os.getenv("BACKOFF_MAX", "60"))       # cap wait (seconds)
_SLEEP = time.sleep  # tests replace this with a recorder that returns instantly


class RateLimitError(Exception):
    """Raised when the primary model endpoint returns HTTP 429 (rate limited)."""


def _chat(base_url: str, api_key: str, model: str, prompt: str) -> str:
    """Call an OpenAI-compatible /chat/completions endpoint. Raise RateLimitError on 429."""
    client = httpx.Client(
        base_url=base_url,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        timeout=httpx.Timeout(120.0, connect=30.0),
    )
    try:
        resp = client.post(
            "/chat/completions",
            json={"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0},
        )
        if resp.status_code == 429:
            raise RateLimitError(f"429 Too Many Requests from {base_url}")
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    finally:
        client.close()


def call_primary_model(prompt: str) -> str:
    """Call the primary model. Raises RateLimitError when the endpoint returns 429."""
    return _chat(PRIMARY_BASE_URL, PRIMARY_API_KEY, PRIMARY_MODEL, prompt)


def call_fallback_model(prompt: str) -> str:
    """Call the fallback model (defaults to the primary proxy if no fallback env set)."""
    return _chat(FALLBACK_BASE_URL, FALLBACK_API_KEY, FALLBACK_MODEL, prompt)


def _default_reporter(event):
    """Default progress reporter: prints to stdout. Emits structured event dicts."""
    t = event.get("type")
    if t == "attempt":
        print(f"[retry] attempt {event['n']}/{event['total']} -> calling primary model", flush=True)
    elif t == "rate_limited":
        print(f"[retry] attempt {event['n']} rate-limited (429); waiting {event['wait']:.2f}s then retrying…", flush=True)
    elif t == "exhausted":
        print(f"[retry] exhausted after {event['n']} attempts -> switching to fallback model", flush=True)
    elif t == "fallback":
        print("[retry] calling fallback model…", flush=True)
    elif t == "fallback_done":
        print("[retry] fallback model responded", flush=True)
    elif t == "done":
        print("[retry] done", flush=True)


def answer_with_retry_and_fallback(prompt: str, primary_fn=None, fallback_fn=None, on_progress=None) -> str:
    """
    Retry the primary model with exponential backoff (up to MAX_RETRIES attempts) on
    RateLimitError; if all attempts are rate-limited, fall back to the secondary model.

    primary_fn / fallback_fn default to the module's call_primary_model / call_fallback_model,
    but callers (e.g. the app) can plug their OWN (traced) callables so LangSmith tracing
    and token-cost reporting are preserved.

    on_progress(event_dict) is called with structured progress events so a UI can show
    the live retry/fallback flow:
      {"type":"attempt","n":N,"total":MAX_RETRIES}
      {"type":"rate_limited","n":N,"wait":W}
      {"type":"exhausted","n":MAX_RETRIES}
      {"type":"fallback"}
      {"type":"fallback_done"}
      {"type":"done"}
    """
    primary_fn = primary_fn or call_primary_model
    fallback_fn = fallback_fn or call_fallback_model
    if on_progress is None:
        on_progress = _default_reporter

    total = MAX_RETRIES
    wait_strategy = wait_exponential(multiplier=BACKOFF_BASE, min=BACKOFF_MIN, max=BACKOFF_MAX)

    def before(rs):
        on_progress({"type": "attempt", "n": rs.attempt_number, "total": total})

    def before_sleep(rs):
        try:
            w = float(wait_strategy(rs))
        except Exception:
            w = 0.0
        on_progress({"type": "rate_limited", "n": rs.attempt_number, "wait": w})

    retryer = Retrying(
        retry=retry_if_exception_type(RateLimitError),
        wait=wait_strategy,
        stop=stop_after_attempt(MAX_RETRIES),
        sleep=_SLEEP,
        reraise=True,
        before=before,
        before_sleep=before_sleep,
    )
    try:
        result = retryer(primary_fn, prompt)
        on_progress({"type": "done"})
        return result
    except RateLimitError:
        on_progress({"type": "exhausted", "n": MAX_RETRIES})
        on_progress({"type": "fallback"})
        fb = fallback_fn(prompt)
        on_progress({"type": "fallback_done"})
        return fb


if __name__ == "__main__":
    # Tiny manual smoke (hits the real primary endpoint once).
    print("primary ->", call_primary_model("Reply with exactly: OK")[:60])
