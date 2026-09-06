"""
test_streaming_tokens_retry.py — token mgmt + content retry + streaming (mocked).
"""
import json
from unittest.mock import patch, MagicMock
import streaming_tokens_retry as str_mod
from streaming_tokens_retry import (
    count_tokens, truncate_context_to_fit, is_valid_response,
    call_with_content_retry, stream_chat,
)


def test_token_management():
    assert count_tokens("hello world") == 2
    assert count_tokens("") == 0
    ctx = ["a b c", "d e f", "g h i"]
    n = count_tokens(ctx[0])  # tokens per item
    kept = truncate_context_to_fit(ctx, max_tokens=2 * n)  # exactly 2 fit, 3rd would exceed
    assert len(kept) == 2, kept
    print(f"✅ token mgmt: count ok, truncated to {len(kept)} chunks (item={n} tok)")


def test_content_retry():
    attempts = {"n": 0}
    def flaky(prompt):
        attempts["n"] += 1
        if attempts["n"] < 3:
            return ""  # invalid (empty) first 2 times
        return "valid answer"
    events = []
    result = call_with_content_retry(flaky, "p", max_attempts=3, on_progress=events.append)
    assert result == "valid answer", result
    assert attempts["n"] == 3, attempts["n"]
    assert any(e["type"] == "valid" for e in events), events
    print(f"✅ content retry: succeeded on attempt {attempts['n']}")

    # all invalid -> fallback
    always_empty = lambda p: ""
    r2 = call_with_content_retry(always_empty, "p", max_attempts=2)
    assert "无法生成" in r2, r2
    print("✅ content retry: fallback after all invalid")


def _fake_stream_lines():
    # emulate SSE lines from an OpenAI-compatible streaming endpoint
    yield 'data: {"choices":[{"delta":{"content":"Hello"}}]}'
    yield 'data: {"choices":[{"delta":{"content":" world"}}]}'
    yield 'data: {"choices":[{"delta":{}},{"delta":{}}]}'
    yield 'data: [DONE]'


def test_stream_chat_yields_chunks():
    fake_resp = MagicMock()
    fake_resp.iter_lines = _fake_stream_lines
    fake_resp.raise_for_status = MagicMock()
    fake_cm = MagicMock()
    fake_cm.__enter__ = MagicMock(return_value=fake_resp)
    fake_cm.__exit__ = MagicMock(return_value=False)
    fake_client = MagicMock()
    fake_client.stream = MagicMock(return_value=fake_cm)
    fake_client.close = MagicMock()
    with patch("streaming_tokens_retry.httpx.Client", return_value=fake_client):
        chunks = list(stream_chat("http://x", "k", "m", [{"role": "user", "content": "hi"}]))
    assert chunks == ["Hello", " world"], chunks
    print(f"✅ streaming: yielded chunks -> {''.join(chunks)!r}")


if __name__ == "__main__":
    test_token_management()
    test_content_retry()
    test_stream_chat_yields_chunks()
    print("\nALL TESTS PASSED ✅")
