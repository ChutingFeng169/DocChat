"""
test_async_pipelines.py — verify sequential vs concurrent timing.

  * sequential:    elapsed ~ N * per_file (sum)
  * concurrent:    elapsed ~ per_file (waits overlap)  -> much faster
  * concurrent(max=1): semaphore forces one-at-a-time -> behaves like sequential
"""

import asyncio
import time
from async_pipelines import process_sequential, process_concurrent


def _embed_s(chunks):
    time.sleep(0.6)  # short, for fast tests
    return chunks


async def main():
    files = ["a.pdf", "b.pdf", "c.pdf"]

    _, seq_t = await process_sequential(files, embed_fn=_embed_s)
    _, conc_t = await process_concurrent(files, embed_fn=_embed_s, max_concurrent=3)
    _, conc1_t = await process_concurrent(files, embed_fn=_embed_s, max_concurrent=1)

    print(f"sequential={seq_t:.2f}s  concurrent={conc_t:.2f}s  concurrent(max=1)={conc1_t:.2f}s")

    assert seq_t >= 1.5, f"sequential should be ~3x0.6=1.8s, got {seq_t}"
    assert conc_t < 1.1, f"concurrent should be ~0.6s, got {conc_t}"
    assert conc_t < seq_t * 0.7, f"concurrent should be much faster than sequential, got conc={conc_t} seq={seq_t}"
    assert conc1_t >= 1.5, f"concurrent(max=1) should behave like sequential (~1.8s), got {conc1_t}"
    print("ALL TESTS PASSED ✅")


if __name__ == "__main__":
    asyncio.run(main())
