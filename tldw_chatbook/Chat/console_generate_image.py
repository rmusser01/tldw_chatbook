"""Pure parsing/formatting for the native Console ``/generate-image`` command.

No dependency on Textual, the running app, or any I/O — mirrors
``console_prefill.py``. The screen layer owns all UI wiring and the
generation worker; this module only splits the ``/generate-image`` args
string into an optional backend override plus prompt, and renders the
single-line content marker stored on the generation card's message row.

Grammar: an optional leading ``:backend`` token selects a non-default
backend (``/generate-image :swarmui a dragon``). A bare ``:`` is NOT a
backend token — it stays part of the prompt. ``:backend`` with no
trailing text parses to an empty prompt, which the caller refuses.
"""

from __future__ import annotations

from dataclasses import dataclass

GENERATION_MARKER_PREFIX = "[image] "
"""Prefix identifying a generation card's content marker in a message row."""

_MARKER_PROMPT_MAX_CHARS = 80


@dataclass(frozen=True)
class GenerateImageArgs:
    """One parsed ``/generate-image`` invocation.

    Args:
        backend: Backend id from a leading ``:backend`` token, or ``None``
            when the command should use the configured default.
        prompt: Generation prompt text (stripped). Empty when the user
            supplied no prompt — the caller refuses to dispatch then.
    """

    backend: str | None
    prompt: str


def parse_generate_image_args(args: str) -> GenerateImageArgs:
    """Split the args string of one ``/generate-image`` invocation.

    A leading ``:backend`` token (first whitespace-delimited token starting
    with ``:`` and longer than the bare colon) selects a backend override;
    everything after it is the prompt. Without such a token the whole
    stripped string is the prompt and ``backend`` is ``None``.

    Args:
        args: Raw text after the ``/generate-image`` command word.

    Returns:
        A `GenerateImageArgs` with the optional backend override and the
        stripped prompt (empty string when no usable prompt was given).
    """
    stripped = args.strip()
    if not stripped:
        return GenerateImageArgs(backend=None, prompt="")
    first, *rest = stripped.split(None, 1)
    if first.startswith(":") and first != ":":
        remainder = rest[0].strip() if rest else ""
        return GenerateImageArgs(backend=first[1:], prompt=remainder)
    return GenerateImageArgs(backend=None, prompt=stripped)


def generation_content_marker(
    prompt: str, limit: int = _MARKER_PROMPT_MAX_CHARS
) -> str:
    """Render the single-line content marker for a generation card.

    Args:
        prompt: Full generation prompt.
        limit: Maximum rendered prompt length, including the ellipsis.

    Returns:
        ``"[image] "`` followed by ``prompt`` with whitespace runs
        collapsed to single spaces, cut to ``limit`` chars with a trailing
        ``…`` when longer.
    """
    flattened = " ".join(prompt.split())
    if len(flattened) > limit:
        flattened = flattened[: limit - 1] + "…"
    return GENERATION_MARKER_PREFIX + flattened
