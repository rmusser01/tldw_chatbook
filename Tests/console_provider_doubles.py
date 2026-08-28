"""One shared shape for Console provider-gateway doubles.

`a26cdafd8` made a typed `ConsoleResolvedDestination` mandatory on any READY
provider resolution: `ConsoleChatController._resolved_destination_for_context`
raises ``ValueError("Ready provider resolution omitted its typed
destination.")`` without one, and the submit is refused with the generic
"Provider destination is incomplete." copy. Every double that hands a REAL
controller a hand-built ``SimpleNamespace`` therefore stopped working on that
commit -- silently, because the refusal reads like an ordinary blocked send.

The destination is DERIVED here through the production classifier
(`resolve_console_destination`) rather than hand-built, so a double cannot
drift from the rule it stands in for: if the classifier's output shape
changes, every double follows automatically.

Only a READY resolution carries one. A double that deliberately reports
not-ready -- or one that exists precisely to exercise the missing-destination
refusal -- must keep returning a resolution without the field, which is why
this helper attaches it only when ``ready`` is true.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from types import SimpleNamespace
from typing import Any, TypeVar

from tldw_chatbook.Chat.console_library_destination import (
    resolve_console_destination,
)

_R = TypeVar("_R")

#: What a cold local provider says while its readiness probe is still warming.
DEFAULT_BLOCKED_COPY = "WIP: provider warming up"


def provider_resolution(
    *,
    ready: bool = True,
    provider: str = "llama_cpp",
    model: str = "test-model",
    base_url: str | None = None,
    visible_copy: str | None = None,
    **extra: Any,
) -> SimpleNamespace:
    """Build the resolution shape a real controller accepts.

    Args:
        ready: Whether the readiness probe succeeded.
        provider: Provider key the turn resolved to.
        model: Model id the turn resolved to.
        base_url: Effective endpoint, or None when the double does not model one.
        visible_copy: Refusal copy; defaults to empty when ready and to
            :data:`DEFAULT_BLOCKED_COPY` when not.
        **extra: Any further attributes the calling test asserts on.

    Returns:
        A resolution namespace carrying a typed ``resolved_destination``
        whenever ``ready`` is true.
    """

    if visible_copy is None:
        visible_copy = "" if ready else DEFAULT_BLOCKED_COPY
    resolution = SimpleNamespace(
        ready=ready,
        provider=provider,
        model=model,
        base_url=base_url,
        visible_copy=visible_copy,
        **extra,
    )
    if ready:
        resolution.resolved_destination = resolve_console_destination(resolution)
    return resolution


def persisted_console_store(
    *,
    db_path: str | Path = ":memory:",
    workspace_registry: Any | None = None,
    **kwargs: Any,
):
    """A `ConsoleChatStore` wired to persistence, as production always is.

    A bare `ConsoleChatStore()` has `persistence is None`. Since `a26cdafd8`,
    `submit_draft` refuses a MANUAL or QUEUED send on a non-ephemeral session
    without the adapter's `commit_durable_turn`, returning "Durable turn
    acceptance is unavailable; the provider was not called." — so a rig built
    that way never reaches whatever it meant to test. Production wires one
    whenever the DB opens (`ConsoleRuntime.ensure_chat_store`).

    Wake turns and ephemeral sessions are exempt from that rule, which is why
    only some bare-store rigs were affected.

    Args:
        db_path: Backing database. Use a file for cross-thread controller tests.
        workspace_registry: Optional durable workspace authority.
        **kwargs: Passed through to `ConsoleChatStore`.

    Returns:
        A store backed by an in-memory ChaChaNotes DB.
    """

    from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
    from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

    return ConsoleChatStore(
        persistence=ChatPersistenceService(
            CharactersRAGDB(str(db_path), "console-doubles"),
            workspace_registry=workspace_registry,
        ),
        **kwargs,
    )


def with_destination(resolution: _R) -> _R:
    """Attach the destination the production gateway attaches to a REAL one.

    `ConsoleProviderResolution.resolved_destination` defaults to None, and the
    real gateway fills it in immediately after building the resolution. A test
    that constructs the dataclass directly skips that step and hands the
    controller a ready resolution with no destination -- the same refusal
    :func:`provider_resolution` exists to prevent, in the typed shape.

    Args:
        resolution: A ready or not-ready provider resolution.

    Returns:
        The resolution carrying a typed destination when it is ready, and
        unchanged when it is not. Frozen dataclasses are replaced rather than
        mutated.
    """

    if not getattr(resolution, "ready", False):
        return resolution
    destination = resolve_console_destination(resolution)
    if dataclasses.is_dataclass(resolution):
        return dataclasses.replace(resolution, resolved_destination=destination)
    resolution.resolved_destination = destination
    return resolution
