"""The Library Notes Preview link policy, pure (TASK-34000.11, final fix round).

What a clicked Preview href may do: open a note (``note://<id>``), go to the
system browser (a clean ``http``/``https`` url), or be refused with a notice
that names the reason. Pure functions over the href string, so the policy is
pinned by a sub-second table in the PR lane
(``Tests/Notes/test_note_link_policy.py``) without booting the app.

Rules (fix round 1 M-9; final round I-2, M-2, M-3):

- The scheme is matched case-insensitively through ``urlsplit`` --
  ``HTTPS://x`` is an https link, not an "unsupported scheme".
- A scheme-less href with no host (``#heading``, ``docs/a.md``, ``""``)
  points nowhere Chatbook can open; it gets its own notice, never the
  "unsupported scheme" one. A scheme-relative ``//host/`` IS refused as a
  scheme Chatbook does not open.
- An http(s) href carrying whitespace, a C0/C1 control character or a
  bidirectional control (U+202A-U+202E, U+2066-U+2069, U+200E/U+200F) is
  refused: an escape sequence must never reach the browser, and a bidi
  override must never make the notice read backwards.
- The notice text drops the C0/C1 and bidi controls from the href it shows.
"""

from __future__ import annotations

from typing import Literal
from urllib.parse import urlsplit

LinkKind = Literal["note", "http", "no_target", "refused"]

NOTE_LINK_PREFIX = "note://"

#: Bidirectional formatting controls (final fix round, M-3): the embeddings,
#: overrides and isolates plus the two marks.
BIDI_CONTROLS: frozenset[str] = frozenset(
    {chr(code) for code in range(0x202A, 0x202F)}
    | {chr(code) for code in range(0x2066, 0x206A)}
    | {"\u200e", "\u200f"}
)


def _is_control(ch: str) -> bool:
    code = ord(ch)
    return code < 32 or 127 <= code <= 159 or ch in BIDI_CONTROLS


def link_notice_text(href: str) -> str:
    """The href as a notice may show it: C0/C1 and bidi controls dropped.

    Args:
        href: The href as the Markdown widget reported it.

    Returns:
        ``href`` with every control character removed; never raises.
    """
    return "".join(ch for ch in href if not _is_control(ch))


def is_clean_http_link(href: str) -> bool:
    """Whether ``href`` is an ``http``/``https`` url the browser may be handed.

    The scheme is read case-insensitively; any whitespace or control
    character anywhere in the href refuses it.

    Args:
        href: The href as the Markdown widget reported it.
    """
    if any(ch.isspace() or _is_control(ch) for ch in href):
        return False
    try:
        scheme = urlsplit(href).scheme.lower()
    except ValueError:
        return False
    return scheme in {"http", "https"}


def link_kind(href: str) -> LinkKind:
    """Classify a Preview href for the link policy.

    Args:
        href: The href as the Markdown widget reported it (already stripped).

    Returns:
        ``"note"`` for ``note://<id>``; ``"http"`` for a clean http(s) url;
        ``"no_target"`` for a scheme-less href with no host (a fragment, a
        relative path, an empty href); ``"refused"`` for everything else --
        another scheme, a scheme-relative ``//host``, an http(s) url carrying
        whitespace or control characters, or an href ``urlsplit`` rejects.
    """
    if href.startswith(NOTE_LINK_PREFIX):
        return "note"
    if is_clean_http_link(href):
        return "http"
    try:
        parts = urlsplit(href)
    except ValueError:
        return "refused"
    if not parts.scheme and not parts.netloc:
        return "no_target"
    return "refused"


__all__ = [
    "BIDI_CONTROLS",
    "NOTE_LINK_PREFIX",
    "LinkKind",
    "is_clean_http_link",
    "link_kind",
    "link_notice_text",
]
