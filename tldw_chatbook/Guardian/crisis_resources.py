"""Crisis resources + disclaimer for Guardian crisis-flagged notices.

Extracted verbatim from ``check_pipeline`` (Task 1's design-review P1
ruling kept them inline for Task 1; Task 2 owns this module). Content is
ported from tldw_server's
``Docs/Design/Guardian_Self_Monitoring.md`` §Crisis Resources -- keep
synchronized with the server's Guardian (ADR-204 contract 7's maintenance
obligation).

Everything here is markup-safe plain text: these strings ride inline
transcript rows and notification toasts, so no Textual square-bracket
markup may ever appear in them.
"""
from __future__ import annotations

#: Resources + disclaimer carried by every surfaced crisis-flagged notice
#: (ADR-204 contract 7). Ported content; keep synchronized with
#: tldw_server's Guardian (the ADR's maintenance obligation).
CRISIS_RESOURCES_TEXT = (
    "If you or someone you know is struggling or in crisis, help is available:\n"
    "- 988 Suicide & Crisis Lifeline (US): call or text 988\n"
    "- Crisis Text Line: text HOME to 741741\n"
    "- SAMHSA National Helpline (US): 1-800-662-4357\n"
    "- IASP Find a Helpline (international): findahelpline.com"
)
CRISIS_DISCLAIMER = "tldw is not a mental health service."


def render_crisis_block() -> str:
    """Return the crisis resources + disclaimer as one plain-text block.

    Returns:
        ``CRISIS_RESOURCES_TEXT`` and ``CRISIS_DISCLAIMER`` joined by one
        blank line -- markup-safe plain text suitable for inline transcript
        rows, notices, and notification toasts.
    """
    return f"{CRISIS_RESOURCES_TEXT}\n\n{CRISIS_DISCLAIMER}"
