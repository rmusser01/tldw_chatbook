"""The one front matter renderer both note exporters write through.

TASK-34000.31 (review finding N-20): the single-note **Export Markdown** file
and the chatbook bundle's note files used to f-string each value into a
``---`` block, so a title holding a colon, a leading ``*``/``#``/``-``/``?``
or a quote produced YAML that ``yaml.safe_load`` rejects or reads wrongly, and
keywords were one comma-joined line. This module serializes the block with a
real dumper, and the importer (``Notes/note_import_parsers.split_frontmatter``)
reads the result back -- the two halves of the round trip share this shape.

Pure: no I/O, no app import. ``yaml`` is imported inside the function so a
caller that is pre-imported at UI ready (``Library/library_notes_state``) adds
nothing to the boot payload (ADR-097).
"""

from __future__ import annotations

from collections.abc import Mapping

__all__ = ["render_note_frontmatter"]

# A folded scalar is valid YAML but ugly in Obsidian's Properties view; a very
# wide limit keeps one value on one line.
_NO_FOLDING_WIDTH = 4096


def render_note_frontmatter(fields: Mapping[str, object]) -> str:
    """Serialize ``fields`` as a ``---`` fenced YAML block the app's own importer reads back.

    A real dumper quotes what needs quoting and nothing else, so a plain
    title stays a plain line while ``Trip planning: Lisbon`` is quoted and a
    string that looks like another type (``"2026-10-24"``, ``"true"``,
    ``"123"``) reads back as the string. Lists stay lists. Insertion order is
    kept so ``title`` stays the first line a human reads.

    Args:
        fields: The front matter properties in the order they should appear.
            Values are plain scalars (``str``, ``int``, ``bool``, ``None``) or
            lists of them -- nothing a safe loader cannot read back.

    Returns:
        The block, ``---`` fences included, ending in a newline. The caller
        adds the blank line that separates it from the body.
    """
    import yaml  # lazy: keeps the exporters' import closure unchanged (ADR-097)

    data = dict(fields)
    body = _dump(yaml, data, allow_unicode=True)
    if yaml.safe_load(body) != data:
        # PyYAML's emitter writes a NEL (U+0085) raw inside a quoted scalar
        # when ``allow_unicode`` is on, and its own loader then reads the
        # character back as a space (found by the Hypothesis property,
        # TASK-34000.31). Escaping every non-ASCII character is uglier but
        # exact, so the block is re-dumped that way only when the readable
        # form does not read back -- the renderer's promise is the round trip.
        body = _dump(yaml, data, allow_unicode=False)
    return f"---\n{body}---\n"


def _dump(yaml, data: dict[str, object], *, allow_unicode: bool) -> str:
    return yaml.safe_dump(
        data,
        sort_keys=False,
        allow_unicode=allow_unicode,
        default_flow_style=False,
        width=_NO_FOLDING_WIDTH,
    )
