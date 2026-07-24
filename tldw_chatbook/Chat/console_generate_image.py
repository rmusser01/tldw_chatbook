"""Parsing/formatting plus the blocking generation batch for the native
Console ``/generate-image`` command.

``parse_generate_image_args``/``generation_content_marker`` (and the module
docstring's original promise) have no dependency on Textual, the running
app, or any I/O — mirroring ``console_prefill.py``. ``run_generation_batch``
is the one deliberate exception: it drives the blocking, network-calling
``Image_Generation.worker`` entry points, so it must run off the UI loop
(the screen layer offloads it via ``asyncio.to_thread``, exactly like
``run_generation`` itself demands). Its adapter dependencies
(``worker.run_generation``/``worker.build_request``) are imported lazily
inside the function rather than at module scope, so importing this module
for its pure helpers never eagerly pulls in the Image_Generation package.

Grammar: an optional leading ``:backend`` token selects a non-default
backend (``/generate-image :swarmui a dragon``). A bare ``:`` is NOT a
backend token — it stays part of the prompt. ``:backend`` with no
trailing text parses to an empty prompt, which the caller refuses.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from tldw_chatbook.Chat.console_chat_models import GenerationVariantMeta

GENERATION_MARKER_PREFIX = "[image] "
"""Prefix identifying a generation card's content marker in a message row."""

_MARKER_PROMPT_MAX_CHARS = 80


def clamp_initial_batch(default_batch: int, max_variants: int) -> int:
    """Clamp the initial batch count to not exceed the variant cap.

    Args:
        default_batch: Configured default batch size (≥ 1).
        max_variants: Configured maximum variants per message (≥ 1).

    Returns:
        The minimum of the two values, preserving ≥ 1 semantics.
    """
    return min(default_batch, max_variants)


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


@dataclass(frozen=True)
class BatchResult:
    """Outcome of one ``run_generation_batch`` call.

    Args:
        successes: Ordered ``(data, mime_type, meta)`` tuples for every
            variant that generated successfully, in generation order.
            Shaped exactly for
            ``ConsoleChatStore.append_generation_message(variants=...)``.
        errors: ``str(exception)`` for every variant that raised, in
            generation order. Empty when every variant succeeded.
    """

    successes: list[tuple[bytes, str, GenerationVariantMeta]]
    errors: list[str]


def run_generation_batch(
    *,
    backend: str,
    prompt: str,
    negative_prompt: str | None,
    seed: int | None,
    count: int,
    generate: Callable[[Any], Any] | None = None,
    build: Callable[..., Any] | None = None,
) -> BatchResult:
    """Run one blocking batch of ``count`` image-generation variants.

    Blocking: every call to ``generate`` (default
    ``Image_Generation.worker.run_generation``) is synchronous and may hit
    the network or a local subprocess. Callers MUST run this off the UI
    loop (e.g. ``await asyncio.to_thread(run_generation_batch, ...)``) —
    this function performs no offloading itself.

    Applies the identical-image guard: when ``seed`` is an explicit value
    (not ``None``), only the first variant (index 0) uses it — every later
    variant is generated with ``seed=-1`` so a batch of N never produces N
    copies of the same image. A ``None`` seed (the common case — no
    explicit seed configured) is passed through unchanged to every variant,
    since there is nothing to force away from.

    A per-variant failure (``generate`` raising) is caught and recorded in
    ``errors``; it never aborts the remaining variants in the batch.

    Args:
        backend: Backend id to generate with (already resolved/validated
            by the caller).
        prompt: Positive prompt text, shared by every variant.
        negative_prompt: Optional negative prompt, shared by every variant.
        seed: Optional explicit seed for variant 0; ``None`` for no
            explicit seed (every variant generates with ``seed=None``).
        count: Number of variants to generate (``>= 1``).
        generate: Blocking single-request entry point. Defaults to
            ``Image_Generation.worker.run_generation``, imported lazily.
        build: Request builder. Defaults to
            ``Image_Generation.worker.build_request``, imported lazily.

    Returns:
        A `BatchResult` with every successful variant's
        ``(data, mime_type, meta)`` plus every failure's error string.
    """
    if generate is None or build is None:
        from tldw_chatbook.Image_Generation import worker as _worker

        if generate is None:
            generate = _worker.run_generation
        if build is None:
            build = _worker.build_request

    successes: list[tuple[bytes, str, GenerationVariantMeta]] = []
    errors: list[str] = []
    for index in range(count):
        variant_seed = seed if (index == 0 or seed is None) else -1
        try:
            request = build(
                backend=backend,
                prompt=prompt,
                negative_prompt=negative_prompt,
                seed=variant_seed,
            )
            result = generate(request)
        except Exception as exc:  # noqa: BLE001 - collected per-variant, never aborts the batch
            errors.append(str(exc))
            continue
        meta = GenerationVariantMeta(
            prompt=prompt,
            negative_prompt=negative_prompt or "",
            backend=backend,
            model=None,
            seed=variant_seed,
            style=None,
            params={},
        )
        successes.append((result.content, result.content_type, meta))
    return BatchResult(successes=successes, errors=errors)
