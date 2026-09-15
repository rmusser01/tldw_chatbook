"""Pure Markdown export for a single Library prompt.

Emits the exact custom ``### SECTION ###`` grammar that
``Prompt_Management.Prompts_Interop.parse_markdown_prompts_from_content``
reads (``TITLE`` -- name + optional details, ``AUTHOR``, ``SYSTEM``,
``USER``, ``KEYWORDS``; structured artifacts also append ``ARTIFACT_TYPE``
and ``STRUCTURE``), so a Library prompt exported via
``render_prompt_markdown`` and re-imported via that parser round-trips its
name/system prompt/user prompt (and author/details/keywords, modulo the
parser's own empty-string-vs-``None`` normalization -- see the docstring
below) unchanged. No Textual/DB imports -- this module only renders text.
"""

from __future__ import annotations

import json
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


def _canonical_definition(value: Any) -> Mapping[str, Any] | None:
    """Return a definition object without transforming user-authored content."""
    if isinstance(value, Mapping):
        return value
    if not isinstance(value, str):
        return None
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError):
        return None
    return parsed if isinstance(parsed, Mapping) else None


def _structured_markdown_sections(detail: Mapping[str, Any]) -> str:
    """Render optional structured metadata after the legacy-compatible body."""
    if detail.get("prompt_format") != "structured":
        return ""
    definition = _canonical_definition(detail.get("prompt_definition"))
    if definition is None:
        return ""
    artifact_type = detail.get("artifact_type") or "prompt"
    if artifact_type not in {"prompt", "recipe"}:
        return ""
    structure = json.dumps(
        definition,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return (
        f"\n### ARTIFACT_TYPE ###\n{artifact_type}\n"
        f"\n### STRUCTURE ###\n```json\n{structure}\n```\n"
    )


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
        grammar ``parse_markdown_prompts_from_content`` parses. Structured
        records with a JSON-object definition append canonical
        ``ARTIFACT_TYPE`` and fenced ``STRUCTURE`` sections after that
        compatibility body.

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
        The parser normalizes blank ``AUTHOR``/``SYSTEM``/``USER`` values
        to ``None``; blank ``details`` remains ``""``. Multiline System
        and User content, including interior blank lines, is preserved.
        Regression coverage for empty sections and multiline content lives
        in ``Tests/Library/test_prompt_export_roundtrip.py``.
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
    return "\n".join(lines) + "\n" + _structured_markdown_sections(detail)
