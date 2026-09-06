"""
async_pipelines.py — sequential vs concurrent multi-file processing.

The expensive, I/O-bound step per file (here: "embedding") is where concurrency
pays off: while one file WAITs on the embedding API, another can be loaded/
chunked/embedded too. asyncio.gather overlaps those waits; a Semaphore caps
concurrency to avoid rate-limits.

Two runtimes:
  * asyncio  (process_sequential / process_concurrent)        -- for the demo/tests.
  * sync     (process_files_sync via ThreadPoolExecutor)       -- for the Streamlit app
    (Streamlit is sync; asyncio.run would block live UI updates, so we use a thread
    pool + as_completed to get live per-file progress -- same concurrency benefit).
"""

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor, as_completed


# ---- default fakes (load_fn / embed_fn) for the demo & tests ----
def _fake_load(name):
    return [f"{name} content chunk"]  # pretend this is the parsed doc


def _fake_embed(chunks):
    time.sleep(1.5)  # simulate a network embedding API call (the I/O wait we overlap)
    return [f"emb({c[:8]})" for c in chunks]


def _name(item):
    return getattr(item, "name", str(item))


async def process_single_file(name, load_fn=None, embed_fn=None, on_progress=None):
    load_fn = load_fn or _fake_load
    embed_fn = embed_fn or _fake_embed
    if on_progress:
        on_progress({"type": "file_start", "name": _name(name)})
    docs = load_fn(name)
    # run the (blocking) embed call in a thread so the event loop stays free -> overlaps waits
    embeddings = await asyncio.get_running_loop().run_in_executor(None, embed_fn, docs)
    if on_progress:
        on_progress({"type": "file_done", "name": _name(name)})
    return {"name": _name(name), "docs": docs, "embeddings": embeddings}


async def process_sequential(items, load_fn=None, embed_fn=None, on_progress=None):
    start = time.time()
    results = []
    for it in items:
        results.append(await process_single_file(it, load_fn, embed_fn, on_progress))
    return results, time.time() - start


async def process_concurrent(items, load_fn=None, embed_fn=None, max_concurrent=2, on_progress=None):
    start = time.time()
    sem = asyncio.Semaphore(max_concurrent)

    async def _limited(name):
        async with sem:
            return await process_single_file(name, load_fn, embed_fn, on_progress)

    results = await asyncio.gather(*[_limited(it) for it in items])
    return results, time.time() - start


def process_files_sync(mode, items, work_fn, max_concurrent=2, on_progress=None, poll_interval=0.15):
    """Sync wrapper for the Streamlit app with LIVE progress (in_flight events).

    Emits:
      {"type":"in_flight","names":[...],"elapsed":float}  -- files not yet done (LIVE label)
      {"type":"file_done","name":...,"elapsed":float}     -- a file finished
    mode: "sequential" or "concurrent". work_fn(item) -> result.
    Returns (results_in_input_order, elapsed_seconds).
    """
    start = time.time()
    results = [None] * len(items)
    if mode == "sequential":
        for i, it in enumerate(items):
            if on_progress:
                on_progress({"type": "in_flight", "names": [_name(it)], "elapsed": time.time() - start})
            results[i] = work_fn(it)
            if on_progress:
                on_progress({"type": "file_done", "name": _name(it), "elapsed": time.time() - start})
    else:  # concurrent -- submit all, poll, emit live in_flight names
        with ThreadPoolExecutor(max_workers=max(1, max_concurrent)) as ex:
            fut_to_idx = {ex.submit(work_fn, it): i for i, it in enumerate(items)}
            remaining = dict(fut_to_idx)
            while remaining:
                for f in [x for x in remaining if x.done()]:
                    i = remaining.pop(f)
                    results[i] = f.result()
                    if on_progress:
                        on_progress({"type": "file_done", "name": _name(items[i]), "elapsed": time.time() - start})
                if remaining:
                    if on_progress:
                        on_progress({"type": "in_flight", "names": [_name(items[remaining[f]]) for f in remaining], "elapsed": time.time() - start})
                    time.sleep(poll_interval)  # releases GIL -> worker threads actually run concurrently
    return results, time.time() - start
