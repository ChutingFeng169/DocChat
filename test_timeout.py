"""
test_timeout.py — test the REAL timeout logic in timeout_handling.call_model_with_timeout.

Verifies:
  1. A stuck call (> timeout) gives up near the set timeout (not infinite wait),
     and emits calling -> timeout events.
  2. A call that finishes within the timeout returns normally (not falsely timed out),
     and emits calling -> done events.
"""

import asyncio
import time
from timeout_handling import call_model_with_timeout


def _slow_model(prompt):
    time.sleep(5)  # stuck longer than the timeout
    return "不应该走到这里"


def _fast_model(prompt):
    time.sleep(1)
    return "正常回答"


async def test_timeout_triggers_correctly():
    """Stuck call (5s) with timeout=3s -> fires ~3s, raises TimeoutError, not infinite."""
    events = []
    start = time.time()
    try:
        await call_model_with_timeout("测试问题", timeout_seconds=3, model_fn=_slow_model, on_progress=events.append)
        raise AssertionError("❌ 测试失败：应该超时但没有超时")
    except TimeoutError:
        elapsed = time.time() - start
        print(f"✅ 测试1通过：{elapsed:.2f}秒后正确触发超时（设定值3秒）")
        assert 2.5 <= elapsed <= 3.5, f"超时触发时间偏差太大: {elapsed}秒"
        assert any(e["type"] == "calling" for e in events), events
        assert any(e["type"] == "timeout" for e in events), events


async def test_normal_call_not_falsely_timed_out():
    """Fast call (1s) with timeout=5s -> returns normally, not timed out."""
    events = []
    start = time.time()
    result = await call_model_with_timeout("测试问题", timeout_seconds=5, model_fn=_fast_model, on_progress=events.append)
    elapsed = time.time() - start
    assert result == "正常回答", result
    assert any(e["type"] == "done" for e in events), events
    assert not any(e["type"] == "timeout" for e in events), events
    print(f"✅ 测试2通过：{elapsed:.2f}秒内正常返回，没有被误判超时")


async def main():
    await test_timeout_triggers_correctly()
    await test_normal_call_not_falsely_timed_out()
    print("\nALL TESTS PASSED ✅")


if __name__ == "__main__":
    asyncio.run(main())
