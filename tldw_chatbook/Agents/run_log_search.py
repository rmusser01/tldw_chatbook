# tldw_chatbook/Agents/run_log_search.py
"""Query the run log: literal by default, structured filters, bounded regex.

Pure module. Literal substring search is the DEFAULT and carries no
line-length cap: `str.__contains__` is linear and cannot backtrack. Regex
is opt-in and scan-bounded: the bound makes catastrophic-backtracking worst
cases finite, not fast. A sufficiently adversarial pattern can still be
expensive within the window. Python's `re` has no match timeout, and
`agent_service._call_with_timeout` abandons rather than kills its worker
thread -- bounding the scan to the first 500 characters avoids runaway
execution, not catastrophe within the bound. See also
`file_operation_tools._MAX_GREP_LINE_SEARCH_CHARS`.
"""

from __future__ import annotations

import re
from pathlib import Path

from .run_log_format import RunLogRecord, iter_records

#: Per-record scan window for opt-in regex mode. This bound makes
#: catastrophic-backtracking worst cases finite, not fast. Neither bound
#: makes such a pattern fast — only bounded. Mirrors
#: `file_operation_tools._MAX_GREP_LINE_SEARCH_CHARS`.
MAX_REGEX_SCAN_CHARS = 500


def load_records(log_dir: Path) -> list[RunLogRecord]:
    """Load every complete record from every segment, in order.

    Segment discovery is glob + sort, never the MANIFEST: a crashed run
    writes no manifest, and those are exactly the runs worth inspecting.

    Args:
        log_dir: The run's log directory.

    Returns:
        All records in record-number order; empty when unreadable.
    """
    records: list[RunLogRecord] = []
    try:
        for segment in sorted(log_dir.glob("logs.*.txt")):
            records.extend(iter_records(segment.read_bytes()))
    except OSError:
        return records
    return sorted(records, key=lambda r: r.number)


def search_records(
    records: list[RunLogRecord],
    *,
    contains: str = "",
    pattern: str = "",
    tool: str = "",
    type: str = "",
    status: str = "",
    kind: str = "",
    from_record: int = 0,
    to_record: int = 0,
    context: int = 0,
    limit: int = 50,
) -> list[RunLogRecord]:
    """Filter ``records``; return hits plus optional neighbouring context.

    Args:
        records: All loaded records, in order.
        contains: Literal substring (case-insensitive). Never compiled.
        pattern: Opt-in regex, searched only over the first
            ``MAX_REGEX_SCAN_CHARS`` characters of each record.
        tool: Exact tool-name filter.
        type: Exact record-type filter.
        status: Exact status filter.
        kind: Exact agent-kind filter.
        from_record: Inclusive lower bound on record number.
        to_record: Inclusive upper bound on record number.
        context: Include this many records either side of each hit.
        limit: Maximum number of matching records returned; context records
            are returned in addition to this limit.

    Returns:
        Matching records in record order, deduplicated, with context records
        included (result may exceed ``limit`` when context is used).
    """
    compiled = None
    if pattern:
        try:
            compiled = re.compile(pattern, re.IGNORECASE)
        except re.error:
            return []
    needle = contains.lower()
    hit_indexes: list[int] = []
    for index, record in enumerate(records):
        if from_record and record.number < from_record:
            continue
        if to_record and record.number > to_record:
            continue
        if tool and record.tool != tool:
            continue
        if type and record.type != type:
            continue
        if status and record.status != status:
            continue
        if kind and record.kind != kind:
            continue
        if needle and needle not in record.content.lower():
            continue
        if compiled is not None and not compiled.search(
            record.content[:MAX_REGEX_SCAN_CHARS]
        ):
            continue
        hit_indexes.append(index)
    # Apply limit to hits first, then expand context around the limited hits.
    # Context records are returned in addition to the limit.
    limited_hit_indexes = hit_indexes[:limit]
    selected: set[int] = set()
    # A negative context would make low > high below, so range(low, high+1)
    # comes back empty and even the hit itself is dropped -- a caller (or a
    # model guessing at search_run_log's args) passing context=-5 would be
    # told "No matching records." even though a match exists, which is
    # worse than an error. Clamp here, at the point of use.
    context = max(0, context)
    for index in limited_hit_indexes:
        low = max(0, index - context)
        high = min(len(records) - 1, index + context)
        selected.update(range(low, high + 1))
    return [records[i] for i in sorted(selected)]


def format_results(records: list[RunLogRecord], *, max_chars: int = 400) -> str:
    """Render results for the model.

    Args:
        records: Records to render.
        max_chars: Per-record content ceiling in the rendering.

    Returns:
        One block per record, or a plain no-matches line.
    """
    if not records:
        return "No matching records."
    blocks = []
    for record in records:
        body = record.content
        if len(body) > max_chars:
            body = body[:max_chars] + f"… (+{len(record.content) - max_chars} chars)"
        blocks.append(
            f"record {record.number:06d} [{record.type}"
            f"{'/' + record.tool if record.tool and record.tool != '-' else ''}"
            f"{'/' + record.status if record.status and record.status != '-' else ''}]"
            f"\n{body}"
        )
    return "\n\n".join(blocks)
