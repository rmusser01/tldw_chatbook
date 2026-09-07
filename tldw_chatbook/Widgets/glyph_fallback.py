"""ASCII-safe glyph fallback mode for narrow-font terminals (AC-01, TASK-2154.19).

Some terminals render the Console glyph vocabulary poorly: "◌" (U+25CC) is
commonly missing or narrow, geometric shapes are font-dependent, and emoji
paint double-width or not at all. This module owns the opt-in escape hatch:
when ``appearance.ascii_glyphs`` is enabled, every status marker resolves to
a pure-ASCII substitute via ``resolve_glyph`` / ``resolve_glyph_text``.

Deliberately a zero-import leaf: ``Widgets.destination_rail`` must stay free
of the Chat layer (ADR-034) and ``Workspaces.conversation_browser_state``
threads glyph strings without model imports, so the fallback machinery lives
here, below both. ``Chat.console_glyphs`` documents the vocabulary; this
module decides how it renders.

The map is keyed by CHARACTER, not by constant, so every consumer of the
same glyph gets the same substitute no matter which import path it took.
Substitutes are bracketed where the marker carries meaning (state, urgency)
and bare punctuation where it is a pure geometric affordance (disclosure
triangles, the draft caret) -- a lone ``x`` on a button reads "close", while
``[x]`` in a tab label reads "done".
"""

from __future__ import annotations

#: Character -> ASCII substitute, applied only while ASCII mode is on.
#: Keep in sync with the status-marker vocabulary in
#: ``Chat/console_glyphs.py``, ``Chat/console_chat_models.py``
#: (``CONSOLE_RUN_MARKER_GLYPHS``), and ``Widgets/destination_rail.py``;
#: ``Tests/Chat/test_console_glyphs.py`` pins the completeness of this map.
ASCII_GLYPH_FALLBACKS: dict[str, str] = {
    # Run markers (console_chat_models.CONSOLE_RUN_MARKER_GLYPHS) + the
    # shared in-progress/done vocabulary (console_glyphs). The urgency order
    # lives in Workspaces/conversation_browser_state._RUN_MARKER_URGENCY,
    # which mirrors these keys as literals.
    "●": "[*]",  # agent running / in progress
    "◆": "[!]",  # waiting for approval
    "✗": "[X]",  # failed
    "✓": "[x]",  # finished ok / done
    "◈": "[s]",  # background sub-agent ended -- unseen (PR3a-2 Task 4)
    # Session + voice lifecycle (console_glyphs).
    "◌": "[~]",  # temporary session
    "◉": "(rec)",  # voice capture live
    "◐": "(~)",  # voice pipeline working
    "✕": "x",  # close/clear buttons
    # Geometric affordances: disclosure and direction.
    "▸": ">",  # collapsed / active / collapse-right
    "◂": "<",  # collapse-left
    "▾": "v",  # expanded
    # RAG scope source types.
    "▦": "[M]",  # media
    "✎": "[N]",  # note
    # Composer furniture.
    "▌": "|",  # draft caret
    "📎": "[+]",  # staged attachment indicator
}

_ASCII_MODE = False


def set_ascii_glyph_mode(enabled: bool) -> None:
    """Enable or disable ASCII-safe glyph substitution process-wide.

    Called once at app compose from ``appearance.ascii_glyphs`` and live by
    the Settings > Appearance toggle; widgets painted before a flip keep
    their current glyphs until their next sync/render.

    Args:
        enabled: True to map every known marker to its ASCII substitute,
            False to render the unicode glyph vocabulary as-is.
    """
    global _ASCII_MODE
    _ASCII_MODE = bool(enabled)


def ascii_glyph_mode() -> bool:
    """Return whether ASCII-safe glyph substitution is active.

    Returns:
        True while ASCII mode is on, False otherwise.
    """
    return _ASCII_MODE


def resolve_glyph(glyph: str) -> str:
    """Return the ASCII substitute for ``glyph`` in ASCII mode, else ``glyph``.

    Unknown glyphs pass through untouched in both modes, so a glyph with no
    assigned fallback can never be swallowed by the resolver.

    Args:
        glyph: The single status-marker character to resolve.

    Returns:
        The mapped ASCII substitute while ASCII mode is on and a mapping
        exists; ``glyph`` unchanged otherwise.
    """
    if not _ASCII_MODE:
        return glyph
    return ASCII_GLYPH_FALLBACKS.get(glyph, glyph)


def resolve_glyph_text(text: str) -> str:
    """Map every character of ``text`` through the fallback table.

    Identity when ASCII mode is off. Used for composed labels that embed a
    marker next to words ("◐ Transcribing…", "📎 2 files", "Composer ▾");
    ordinary ASCII text passes through unchanged in both modes.

    Args:
        text: The label text to resolve character by character.

    Returns:
        ``text`` with each known marker replaced by its ASCII substitute
        while ASCII mode is on; ``text`` unchanged otherwise.
    """
    if not _ASCII_MODE:
        return text
    return "".join(ASCII_GLYPH_FALLBACKS.get(char, char) for char in text)
