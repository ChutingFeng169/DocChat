"""
streaming_tokens_retry.py — three capabilities:

  1. Streaming   : stream_chat() generator yielding content deltas (httpx stream=True + SSE).
  2. Token mgmt  : count_tokens() (tiktoken cl100k_base) + truncate_context_to_fit().
  3. Content retry: is_valid_response() + call_with_content_retry()
                   (retry when the CONTENT is empty/malformed -- not on network 429).
"""
import json
import time

import httpx
import tiktoken

_encoding = tiktoken.get_encoding("cl100k_base")


# ---------------------------------------------------------------- 2. token mgmt
def count_tokens(text: str) -> int:
    return len(_encoding.encode(text or ""))


def truncate_context_to_fit(contexts, max_tokens: int = 3000):
    """Keep contexts (in order) until the token budget is exceeded."""
    result = []
    total = 0
    for ctx in contexts:
        n = count_tokens(ctx)
        if total + n > max_tokens:
            break
        result.append(ctx)
        total += n
    return result


# ---------------------------------------------------------------- 3. content retry
def is_valid_response(response, expected_format=None):
    if not response or not str(response).strip():
        return False
    if expected_format == "json":
        try:
            json.loads(response)
        except (json.JSONDecodeError, TypeError):
            return False
    return True


def call_with_content_retry(call_fn, prompt, max_attempts=3, expected_format=None, on_progress=None):
    """Retry call_fn(prompt) when the CONTENT is invalid (empty/malformed)."""
    for attempt in range(1, max_attempts + 1):
        if on_progress:
            on_progress({"type": "attempt", "n": attempt, "total": max_attempts})
        try:
            response = call_fn(prompt)
        except Exception as e:
            if on_progress:
                on_progress({"type": "error", "n": attempt, "error": str(e)})
            response = ""
        if is_valid_response(response, expected_format):
            if on_progress:
                on_progress({"type": "valid", "n": attempt})
            return response
        if on_progress:
            on_progress({"type": "invalid", "n": attempt})
    fallback = "抱歉，暂时无法生成有效回答，请换个问法试试。"
    if on_progress:
        on_progress({"type": "fallback"})
    return fallback


# ---------------------------------------------------------------- 1. streaming
def stream_chat(base_url, api_key, model, messages, on_chunk=None, timeout=120.0):
    """Generator yielding content deltas from an OpenAI-compatible streaming endpoint."""
    client = httpx.Client(
        base_url=base_url,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        timeout=httpx.Timeout(timeout, connect=30.0),
    )
    payload = {"model": model, "messages": messages, "stream": True, "stream_options": {"include_usage": True}}
    try:
        with client.stream("POST", "/chat/completions", json=payload) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                if isinstance(line, bytes):
                    line = line.decode("utf-8", errors="ignore")
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data.strip() == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    continue
                choices = obj.get("choices") or []
                delta = choices[0].get("delta", {}).get("content") if choices else None
                if delta:
                    if on_chunk:
                        on_chunk({"type": "chunk", "text": delta})
                    yield delta
    finally:
        client.close()
