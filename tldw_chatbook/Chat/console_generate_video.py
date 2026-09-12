"""Parsing/formatting plus the blocking generation for the native Console
``/generate-video`` command (task-3401.5).

``parse_generate_video_args`` has no dependency on Textual, the running app,
or any I/O -- mirroring ``console_generate_image.py``'s pure-helpers rule.
``run_video_generation`` is the one deliberate exception: it drives the
blocking, network-calling ``Video_Generation.worker`` entry points plus the
VideoStore file write, so it must run off the UI loop (the screen offloads
it via ``asyncio.to_thread``, exactly like the image batch).

Grammar: an optional leading ``:backend`` token selects a non-default
backend (``/generate-video :comfyui a dragon``). Token consumption stops at
the first token that isn't prefixed with ``:``; everything from there on is
the prompt. A bare ``:`` is NOT a token -- it stays part of the prompt.
(``@style`` templates are deferred to task-3401.12, so the video grammar
has no ``@`` token yet.)
"""

from __future__ import annotations

import threading
import tempfile
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO, Literal

from tldw_chatbook.Chat.console_command_grammar import (
    COMMAND_PREFIX,
    GENERATE_VIDEO_COMMAND_NAME,
)
from tldw_chatbook.Video_Generation.video_metadata import VideoGenerationMetadata
from tldw_chatbook.Video_Generation.video_formats import canonical_video_extension
from tldw_chatbook.Video_Generation.video_store import (
    VideoCapacityExceeded,
    VideoPublicationGate,
    VideoStore,
    VideoStoreSaveError,
)

GENERATE_VIDEO_COMMAND_WORD = COMMAND_PREFIX + GENERATE_VIDEO_COMMAND_NAME
"""The full leading command word (``"/generate-video"``), as registered."""

GENERATE_VIDEO_USAGE_TEXT = "Usage: /generate-video [:backend] <prompt>"
"""Status text for a ``/generate-video`` invocation with nothing to work with."""

#: Backends billed per generated second (for the cost-confirm gate). Local
#: backends (comfyui, stable_diffusion_cpp) are free at the margin.
_PAID_BACKENDS = frozenset({"minimax"})

PendingReason = Literal["over_capacity", "store_failure"]


@dataclass
class PendingVideoArtifact:
    """One generated video awaiting a user-selected storage outcome."""

    metadata: VideoGenerationMetadata
    message_id: str
    slug: str
    extension: str
    size_bytes: int
    max_bytes: int
    reason: PendingReason
    stream: BinaryIO = field(repr=False)
    error_type: str | None = None

    def rewind(self) -> None:
        """Position the owned payload stream at its beginning."""
        self.stream.seek(0)

    def close(self) -> None:
        """Release the owned payload stream; repeated calls are harmless."""
        if not self.stream.closed:
            self.stream.close()


def _stage_pending_video(
    *,
    metadata: VideoGenerationMetadata,
    message_id: str,
    slug: str,
    extension: str,
    content: bytes,
    max_bytes: int,
    reason: PendingReason,
    error_type: str | None = None,
) -> PendingVideoArtifact:
    """Move an unstored generation into an auto-deleting temporary stream."""
    stream = tempfile.TemporaryFile(mode="w+b")
    try:
        written = stream.write(content)
        if written != len(content):
            raise OSError("pending video staging was incomplete")
        artifact = PendingVideoArtifact(
            metadata=metadata,
            message_id=message_id,
            slug=slug,
            extension=extension,
            size_bytes=len(content),
            max_bytes=max_bytes,
            reason=reason,
            stream=stream,
            error_type=error_type,
        )
        artifact.rewind()
    except Exception:
        with suppress(Exception):
            stream.close()
        raise
    return artifact


@dataclass(frozen=True)
class GenerateVideoArgs:
    """One parsed ``/generate-video`` invocation.

    Args:
        backend: Backend id from a leading ``:backend`` token, or ``None``
            when the command should use the configured default.
        prompt: Generation prompt text (stripped). Empty when the user
            supplied no prompt -- the caller refuses to dispatch then.
        style: Raw text of a leading ``@style`` token (without the ``@``),
            unresolved against the template catalog (task-3401.12). ``None``
            when no ``@style`` token was present.
    """

    backend: str | None
    prompt: str
    style: str | None = None


def parse_generate_video_args(args: str) -> GenerateVideoArgs:
    """Split the args string of one ``/generate-video`` invocation.

    Consumes leading whitespace-delimited tokens in any order/combination:
    a token starting with ``:`` (longer than the bare colon) sets the
    backend override; a token starting with ``@`` (longer than the bare
    ``@``) sets the raw style token. Consumption stops at the first token
    that matches neither shape -- that token and everything after it is the
    prompt. A bare ``:`` or ``@`` stays part of the prompt. (Same grammar
    shape as ``parse_generate_image_args``.)
    """
    remaining = args.strip()
    backend: str | None = None
    style: str | None = None
    while remaining:
        parts = remaining.split(None, 1)
        token = parts[0]
        rest = parts[1] if len(parts) > 1 else ""
        if token.startswith(":") and token != ":":
            backend = token[1:]
            remaining = rest
            continue
        if token.startswith("@") and token != "@":
            style = token[1:]
            remaining = rest
            continue
        break
    return GenerateVideoArgs(backend=backend, prompt=remaining.strip(), style=style)


def is_paid_backend(backend: str) -> bool:
    """Whether the backend bills per generation (for the cost-confirm gate)."""
    return backend.strip().lower() in _PAID_BACKENDS


def estimate_video_cost_text(backend: str, duration_seconds: int | None) -> str:
    """Return the human-readable cost line for the pre-dispatch confirm modal.

    No live pricing is wired in, so paid backends get the honest billing
    SHAPE (per generated second, at MiniMax's current rates) rather than a
    fabricated dollar figure; local backends read free.
    """
    duration = duration_seconds if duration_seconds is not None else 5
    if is_paid_backend(backend):
        return (
            f"This generates a {duration}s video on '{backend}', billed per "
            "generated second at MiniMax's current rates."
        )
    return f"This generates a {duration}s video on '{backend}' (local, no per-clip charge)."


def run_video_generation(
    *,
    backend: str,
    prompt: str,
    message_id: str,
    negative_prompt: str | None = None,
    style_negative_prompt: bool = False,
    duration_seconds: int | None = None,
    fps: int | None = None,
    width: int | None = None,
    height: int | None = None,
    ratio: str | None = None,
    seed: int | None = None,
    model: str | None = None,
    video_format: str = "mp4",
    cancel_event: threading.Event | None = None,
    publication_gate: VideoPublicationGate | None = None,
    video_store: VideoStore | None = None,
) -> tuple[VideoGenerationMetadata, Path] | PendingVideoArtifact:
    """Run one video generation and persist the bytes to the VideoStore.

    Blocking: must run off the UI loop. Allocates the slug BEFORE saving so
    the metadata's name and the on-disk file always agree; the message id is
    pre-allocated by the caller (the message row is appended only after the
    bytes exist, but the VideoStore keys by that id -- ADR-044).

    Args:
        backend: Resolved backend id.
        prompt: Generation prompt.
        message_id: Pre-allocated Console message id owning the file.
        negative_prompt: Optional negative prompt (local backends).
        style_negative_prompt: Whether ``negative_prompt`` came from a style
            template and may be suppressed for an incompatible workflow.
        duration_seconds/fps/width/height/ratio/seed/model: Optional params.
        video_format: Canonical requested output container.
        cancel_event: Optional cooperative-cancellation event, threaded to
            adapters that support it (minimax).
        publication_gate: Optional gate that linearizes managed publication
            against owning-screen teardown.
        video_store: Injected store (tests); defaults to a live-config store.

    Returns:
        ``(metadata, path)`` for a managed save, or a temporary-file-backed
        :class:`PendingVideoArtifact` when storage needs a user decision.

    Raises:
        VideoGenerationError: Propagated from validation/dispatch/adapter.
        OSError: Temporary staging failed after managed storage failed.
    """
    if (
        style_negative_prompt
        and negative_prompt
        and backend.strip().lower() == "comfyui"
    ):
        from tldw_chatbook.Video_Generation.adapter_registry import get_registry

        registry = get_registry()
        resolved_backend = registry.resolve_backend(backend)
        adapter = (
            registry.get_adapter(resolved_backend)
            if resolved_backend == "comfyui"
            else None
        )
        classify_workflow = getattr(adapter, "selected_workflow_is_h3", None)
        if callable(classify_workflow) and classify_workflow():
            negative_prompt = None

    from tldw_chatbook.Video_Generation.worker import build_request, run_generation

    store = video_store if video_store is not None else VideoStore()
    slug = store.allocate_slug(message_id, prompt)
    request = build_request(
        backend=backend,
        prompt=prompt,
        negative_prompt=negative_prompt,
        duration_seconds=duration_seconds,
        fps=fps,
        width=width,
        height=height,
        ratio=ratio,
        seed=seed,
        model=model,
        video_format=video_format,
    )
    # worker.run_generation is the single validation choke point AND the
    # dispatch seam; adapters that don't declare cancel_event support are
    # called without it (signature-detected there, never TypeError-sniffed).
    result = run_generation(request, cancel_event=cancel_event)
    metadata = VideoGenerationMetadata(
        name=slug,
        prompt=prompt,
        negative_prompt=negative_prompt or "",
        backend=request.backend,
        container=result.container,
        model=result.resolved_model or model,
        seed=result.resolved_seed if result.resolved_seed is not None else seed,
        duration_seconds=result.duration_seconds
        if result.duration_seconds is not None
        else (float(duration_seconds) if duration_seconds is not None else None),
        fps=result.fps
        if result.fps is not None
        else (float(fps) if fps is not None else None),
        width=result.width or width,
        height=result.height or height,
        ratio=ratio,
    )
    extension = canonical_video_extension(result.container)
    try:
        outcome = store.save(
            message_id,
            slug,
            result.content,
            extension=extension,
            publication_gate=publication_gate,
        )
    except VideoStoreSaveError as exc:
        return _stage_pending_video(
            metadata=metadata,
            message_id=message_id,
            slug=slug,
            extension=extension,
            content=result.content,
            max_bytes=store.capacity_bytes,
            reason="store_failure",
            error_type=type(exc).__name__,
        )
    if isinstance(outcome, VideoCapacityExceeded):
        return _stage_pending_video(
            metadata=metadata,
            message_id=message_id,
            slug=slug,
            extension=extension,
            content=result.content,
            max_bytes=outcome.max_bytes,
            reason="over_capacity",
        )
    return metadata, outcome
