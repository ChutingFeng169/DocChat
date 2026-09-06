"""
timeout_handling.py
===================
Wrap a (blocking) model call with a timeout, and make the wait VISIBLE.

Two entry points:
  * call_model_with_timeout(prompt, timeout_seconds, model_fn, on_progress)  -- ASYNC
      asyncio.wait_for + run_in_executor (the original design). Emits: calling/done/timeout.
  * call_with_timeout_sync(prompt, timeout_seconds, model_fn, on_progress)  -- SYNC
      Runs the blocking call in a DAEMON thread and polls, emitting live
      "waiting" ticks (elapsed/timeout) so a UI can show the wait counting up.
      Used by the app + demo for live visibility (mirrors the rate-limit st.status flow).

on_progress event dicts:
  {"type":"calling",  "timeout": float}
  {"type":"waiting",  "elapsed": float, "timeout": float}   # sync version only
  {"type":"done",     "elapsed": float}
  {"type":"timeout",  "timeout": float, "elapsed": float}

Limitation: Python threads cannot be force-killed. On timeout we stop WAITING
and raise TimeoutError, but the blocking call keeps running in the background
until it finishes on its own (daemon thread => it won't hang the process).
In production, also set httpx/connect timeouts so the lingering call dies naturally.
"""

import asyncio
import threading
import time
from typing import Callable, Optional


def _default_model_call(prompt: str) -> str:
    # TODO: wire to your real synchronous model call, e.g.:
    # from rate_limit_fallback import call_primary_model
    # return call_primary_model(prompt)
    raise NotImplementedError("Provide model_fn or wire _default_model_call to the real model call.")


async def call_model_with_timeout(prompt: str, timeout_seconds: float = 15.0,
                                   model_fn: Optional[Callable[[str], str]] = None,
                                   on_progress=None) -> str:
    """Async: run the sync model_fn in an executor, give up after timeout_seconds."""
    fn = model_fn or _default_model_call
    loop = asyncio.get_running_loop()
    if on_progress:
        on_progress({"type": "calling", "timeout": timeout_seconds})
    try:
        result = await asyncio.wait_for(loop.run_in_executor(None, fn, prompt), timeout=timeout_seconds)
        if on_progress:
            on_progress({"type": "done"})
        return result
    except asyncio.TimeoutError:
        if on_progress:
            on_progress({"type": "timeout", "timeout": timeout_seconds})
        raise TimeoutError(f"调用超过{timeout_seconds}秒未响应")


def call_with_timeout_sync(prompt: str, timeout_seconds: float = 15.0,
                           model_fn: Optional[Callable[[str], str]] = None,
                           on_progress=None, poll_interval: float = 0.5) -> str:
    """Sync: run model_fn in a DAEMON thread; poll and emit live 'waiting' ticks;
    raise TimeoutError if it exceeds timeout_seconds. Daemon thread => no process hang."""
    fn = model_fn or _default_model_call
    box: dict = {}

    def worker():
        try:
            box["value"] = fn(prompt)
        except BaseException as e:  # noqa: BLE001
            box["error"] = e

    th = threading.Thread(target=worker, daemon=True)
    th.start()
    if on_progress:
        on_progress({"type": "calling", "timeout": timeout_seconds})
    start = time.time()
    while True:
        if "value" in box:
            elapsed = time.time() - start
            if on_progress:
                on_progress({"type": "done", "elapsed": elapsed})
            return box["value"]
        if "error" in box:
            raise box["error"]
        elapsed = time.time() - start
        if elapsed >= timeout_seconds:
            if on_progress:
                on_progress({"type": "timeout", "timeout": timeout_seconds, "elapsed": elapsed})
            raise TimeoutError(f"调用超过{timeout_seconds}秒未响应")
        if on_progress:
            on_progress({"type": "waiting", "elapsed": elapsed, "timeout": timeout_seconds})
        th.join(poll_interval)
