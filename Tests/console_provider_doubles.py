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

from types import SimpleNamespace
from typing import Any

from tldw_chatbook.Chat.console_library_destination import (
    resolve_console_destination,
)

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
