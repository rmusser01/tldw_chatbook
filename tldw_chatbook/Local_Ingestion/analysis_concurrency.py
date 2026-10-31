# tldw_chatbook/Local_Ingestion/analysis_concurrency.py
"""Bounded-concurrency chunk analysis shared by the ingestion processors.

Review-B task B14: the PDF / EPUB / markup ingestion processors summarized
their chunks strictly serially, so a 30-chunk document paid 30 sequential LLM
round-trips. This module is the ONE shared seam for that fan-out.

It is stdlib-only on purpose: every ``Local_Ingestion`` module (including the
lazily-loaded processor libraries and ``local_file_ingestion``, whose import
weight the spawn-safe parse worker must not pay -- see
``Tests/Local_Ingestion/test_ingest_import_weight.py``) can import it at
module level without cycles or heavy transitive imports.

Ordering contract: ``result[i]`` corresponds to ``chunks[i]`` exactly.

Failure contract: every submitted future is allowed to settle before anything
raises, and the first exception in chunk order is re-raised after the join.
Callers that need the historical per-chunk "record the error on the chunk and
keep going" isolation implement it inside their own ``analyze_fn`` wrapper
(see the ingestion processors' chunk loops) -- the helper itself never
abandons in-flight work on the first failure.
"""

import concurrent.futures
from typing import Callable, List, Optional, Sequence

#: Default bound on simultaneously in-flight chunk analyses.
DEFAULT_MAX_WORKERS = 3

__all__ = ["analyze_chunks_concurrently", "DEFAULT_MAX_WORKERS"]


def analyze_chunks_concurrently(
    chunks: Sequence[str],
    analyze_fn: Callable[[str], str],
    max_workers: int = DEFAULT_MAX_WORKERS,
) -> List[str]:
    """
    Run ``analyze_fn`` over ``chunks`` with bounded thread concurrency.

    Args:
        chunks: Chunk texts to analyze, in processing order.
        analyze_fn: Callable applied to each chunk text. May raise; see the
            failure contract in the module docstring.
        max_workers: Upper bound on concurrently in-flight analyses. Clamped
            to ``[1, len(chunks)]``.

    Returns:
        A list ``results`` where ``results[i]`` is the (order-preserving)
        return value of ``analyze_fn(chunks[i])``.

    Raises:
        Exception: The first exception (by chunk order) raised by
            ``analyze_fn``, after every submitted future has settled.
    """
    if not chunks:
        return []

    results: List[Optional[str]] = [None] * len(chunks)
    errors: dict[int, Exception] = {}
    workers = max(1, min(int(max_workers), len(chunks)))

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        future_to_index = {
            pool.submit(analyze_fn, chunk): index
            for index, chunk in enumerate(chunks)
        }
        for future in concurrent.futures.as_completed(future_to_index):
            index = future_to_index[future]
            try:
                results[index] = future.result()
            except Exception as exc:  # noqa: BLE001 - re-raised below after the join
                errors[index] = exc

    if errors:
        raise errors[min(errors)]

    # By this point every slot is filled (a missing slot implies an error,
    # and errors raise above).
    return results  # type: ignore[return-value]
