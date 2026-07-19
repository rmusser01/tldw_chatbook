"""Pure Markdown export for a single Library prompt.

Emits the exact custom ``### SECTION ###`` grammar that
``Prompt_Management.Prompts_Interop.parse_markdown_prompts_from_content``
reads (``TITLE`` -- name + optional details, ``AUTHOR``, ``SYSTEM``,
``USER``, ``KEYWORDS``), so a Library prompt exported via
``render_prompt_markdown`` and re-imported via that parser round-trips its
name/system prompt/user prompt (and author/details/keywords, modulo the
parser's own empty-string-vs-``None`` normalization -- see the docstring
below) unchanged. No Textual/DB imports -- this module only renders text.
"""

from __future__ import annotations

from typing import Any, Mapping

__all__ = ["render_prompt_markdown"]


def _text(value: Any) -> str:
    return "" if value is None else str(value)


def _keywords_csv(value: Any) -> str:
    """Render a prompt's keywords (list or CSV string) as one CSV line."""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (list, tuple)):
        items = [str(item).strip() for item in value if str(item).strip()]
        return ", ".join(items)
    return ""


def render_prompt_markdown(detail: Mapping[str, Any]) -> str:
    """Render a prompt detail mapping into the parser's custom MD grammar.

    Args:
        detail: A prompt detail-shaped mapping (as built by
            ``library_prompts_state.build_prompt_editor_state``/the raw
            ``PromptsDatabase.fetch_prompt_details`` row, or the Library
            prompt editor's live field values) with ``name``, ``author``,
            ``details``, ``system_prompt``, ``user_prompt``, ``keywords``
            keys. ``keywords`` may be a list of strings or a single
            comma-separated string; missing/``None`` fields render blank.

    Returns:
        Markdown text using the ``### TITLE ###`` / ``### AUTHOR ###`` /
        ``### SYSTEM ###`` / ``### USER ###`` / ``### KEYWORDS ###`` section
        grammar ``parse_markdown_prompts_from_content`` parses.

        The ``AUTHOR`` section is ALWAYS emitted, even when ``author`` is
        blank: the parser's ``TITLE`` block only stops capturing "details"
        at a literal ``\\n### AUTHOR ###`` line (or end of string) --
        omitting that section here would let the parser swallow every
        later section (SYSTEM/USER/KEYWORDS) into "details" instead. The
        ``KEYWORDS`` section is omitted entirely when there are no
        keywords, matching the parser's own "no section -> ``[]``" default
        (an emitted-but-blank ``KEYWORDS`` section parses identically, but
        omitting it keeps the output free of an empty trailing section).

    Note:
        Two pre-existing quirks of ``parse_markdown_prompts_from_content``
        itself (not something this renderer works around, and out of this
        module's scope to fix -- see ``Tests/Library/test_prompt_export_roundtrip.py``'s
        "known limitation" characterization tests for verified examples):

        * A blank ``AUTHOR``/``SYSTEM``/``USER`` value, when another
          section follows it (the common case), does NOT parse back as
          ``""``/``None`` -- the parser's generic section regex
          backtrack-swallows the blank line into its own header match,
          so the capture bleeds into the *next* section's literal header
          text instead. (A blank value with nothing at all following it
          would parse back as ``None``, but that is not the layout this
          renderer produces, since AUTHOR/SYSTEM/USER/KEYWORDS always
          precede/follow each other in a fixed section order.)
        * A multi-line ``SYSTEM``/``USER`` value is truncated after its
          first line: the parser's per-section capture stops at a bare
          ``$`` (under ``re.MULTILINE``, which matches before every
          newline, not just end-of-string), not just at the next ``###``
          marker.

        Only ``details`` (derived from the ``TITLE`` block's own separate
        regex, which requires a literal ``### AUTHOR ###``/end-of-string
        terminator rather than a generic ``$``) is unaffected by either
        quirk -- it keeps an explicit ``""`` and supports multi-line text
        correctly.
    """
    name = _text(detail.get("name")).strip() or "Untitled prompt"
    details = _text(detail.get("details"))
    author = _text(detail.get("author"))
    system_prompt = _text(detail.get("system_prompt"))
    user_prompt = _text(detail.get("user_prompt"))
    keywords_csv = _keywords_csv(detail.get("keywords"))

    lines = ["### TITLE ###", name]
    if details:
        lines.append(details)
    lines.append("### AUTHOR ###")
    lines.append(author)
    lines.append("### SYSTEM ###")
    lines.append(system_prompt)
    lines.append("### USER ###")
    lines.append(user_prompt)
    if keywords_csv:
        lines.append("### KEYWORDS ###")
        lines.append(keywords_csv)
    return "\n".join(lines) + "\n"
