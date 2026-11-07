"""Textual body wrapping projected into plain-text source offsets."""

from __future__ import annotations

from textual.content import Content


def build_body_wrap_table(
    text: str, width: int, *, content_type: type[Content]
) -> tuple[tuple[tuple[int, str], ...], int]:
    """Memoized wrap table: each wrapped body line with its source offset.

    TASK-21114: a drag delivers MouseMove at 50-100 Hz and every event needs
    the wrapped layout of the SAME (text, width) -- re-running
    ``Content.wrap`` plus the offset-alignment scan over a multi-KB body per
    event was the dominant per-move cost. The table is pure in its key, so a
    small LRU covers a drag's lifetime while text growth (streaming) and
    width changes (resize mid-drag) each miss into a fresh entry and the old
    one ages out.

    Returns:
        ``(table, total_lines)`` where ``table[i]`` is ``(source_start,
        line_text)`` for wrapped line ``i`` and ``total_lines`` counts ALL
        wrapped lines. ``len(table) < total_lines`` means the alignment scan
        hit a wrap edge case it does not model at index ``len(table)`` --
        cells on or below that line fall back to the single-line mapping
        (exactly where the pre-memoization loop bailed out).
    """
    wrapped = [
        line.plain for line in content_type(text, strip_control_codes=False).wrap(width)
    ]
    table: list[tuple[int, str]] = []
    source_offset = 0
    for line in wrapped:
        if line:
            start = text.find(line, source_offset)
            if start == -1 or text[source_offset:start].strip():
                # Wrap edge case not modeled (defensive): stop here; the
                # caller falls back to the single-line mapping for this
                # line and everything after it.
                break
            table.append((start, line))
            source_offset = start + len(line)
        else:
            # Blank wrapped line: anchors at the current position.
            table.append((source_offset, ""))
            # Consume the blank line's own break so later lines stay
            # aligned; any other inter-line whitespace is absorbed by the
            # next line's find() above.
            if source_offset < len(text) and text[source_offset] in "\r\n":
                source_offset += 1
    return tuple(table), len(wrapped)
