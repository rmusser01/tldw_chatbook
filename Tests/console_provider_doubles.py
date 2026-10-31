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
import json
from collections.abc import AsyncIterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any, TypeVar

from tldw_chatbook.Chat.console_library_destination import (
    resolve_console_destination,
)
from tldw_chatbook.Chat.console_provider_gateway import (
    ProviderToolCalls,
    ProviderTurnMetadata,
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
    database_path: str | Path | None = None,
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
        database_path: Backward-compatible alias for ``db_path``.
        workspace_registry: Optional durable workspace authority.
        **kwargs: Passed through to `ConsoleChatStore`.

    Returns:
        A store backed by an in-memory ChaChaNotes DB.
    """

    from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
    from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

    if database_path is not None:
        db_path = database_path
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


class AgentModeGateway:
    """Agent-capable gateway double: emits one native tool call when the
    caller passes ``tools=``, then a final text answer once the tool result
    returns. ``resolve_for_send`` reports an execution key and model id that
    are native-tools-capable (NATIVE_TOOLS_PROVIDERS; capability-known id).

    The conftest ``StreamingGateway`` (Tests/Chat/conftest.py) cannot drive
    agent mode: its ``stream_chat`` yields only ("hel", "lo") text chunks, so
    it can never emit a native tool call and only exercises plain chat. This
    double implements the same two-method protocol but yields a
    ``ProviderToolCalls`` item (OpenAI-shape dicts -- the Plan-B contract the
    agent bridge consumes) when the caller passes ``tools=`` and no tool
    result is present yet, and the final text answer on the follow-up call
    carrying the tool result. Ported from the Binding-1 spike probe (spike
    condition 5).

    Args:
        tool_name: Name of the single tool the gateway calls out to.
        final_answer: Text the follow-up call (the one carrying the tool
            result) yields as its answer.
    """

    def __init__(
        self,
        *,
        tool_name: str = "fs_list",
        final_answer: str = "MOBILE_AGENT_ROUND_TRIP_COMPLETE",
    ) -> None:
        self.tool_name = tool_name
        self.final_answer = final_answer
        self.stream_calls: list[dict[str, Any]] = []
        self.tool_result_messages: list[dict[str, Any]] = []

    async def resolve_for_send(self, selection: Any) -> SimpleNamespace:
        # "openai" (an execution key in Agents/native_tools.py's
        # NATIVE_TOOLS_PROVIDERS) + a capability-known model id
        # ("gpt-4o-mini"): an unknown model id makes the first-request
        # planner fall back to find_tools/load_tools discovery mode, and
        # "llama_cpp" (the conftest double's provider) is gated on
        # reasoning_replay.native_tools -- either way no tools are
        # advertised and the round cannot happen.
        return provider_resolution(
            ready=True,
            provider="openai",
            model="gpt-4o-mini",
            base_url="https://mobile.invalid/v1",
            visible_copy="",
        )

    async def stream_chat(
        self,
        resolution: Any,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> AsyncIterator[str | ProviderToolCalls]:
        tools = kwargs.get("tools")
        saw_tool_result = any(
            isinstance(m, dict) and m.get("role") == "tool" for m in messages
        )
        self.stream_calls.append(
            {
                "tools_passed": tools is not None,
                "tool_advertised": any(
                    isinstance(t, dict)
                    and (
                        t.get("function", {}).get("name") == self.tool_name
                        or t.get("name") == self.tool_name
                    )
                    for t in (tools or ())
                ),
                "n_messages": len(messages),
                "saw_tool_result": saw_tool_result,
            }
        )
        if tools and not saw_tool_result:
            yield "Checking the working folder first.\n"
            yield ProviderToolCalls(
                (
                    {
                        "id": "call-spike-1",
                        "type": "function",
                        "function": {
                            "name": self.tool_name,
                            "arguments": json.dumps({"path": "."}),
                        },
                    },
                ),
                metadata=ProviderTurnMetadata(finish_reason="tool_calls"),
            )
            return
        if saw_tool_result:
            self.tool_result_messages = [
                dict(m)
                for m in messages
                if isinstance(m, dict) and m.get("role") == "tool"
            ]
        yield self.final_answer
