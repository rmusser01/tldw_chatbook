"""What the Console chat controller receives instead of the app (TASK-36947.4).

ADR-250 section 7: the controller holds no reference to the app. It receives
the service members it reads, taken from ``app.services`` where it is built,
and narrow callables for the UI capabilities and app state the bundle does
not hold. Members keep their app alias names, so the duck-typed turn-context
readers (``resolve_turn_tool_policy_profile_id`` and the rest) accept a host
where they used to receive the app.
"""

from __future__ import annotations

import functools
import threading
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

#: Each bundle member the controller reads: its ``ApplicationServices`` slice
#: and field. ``raw_cli_runtime`` (rule 4) is passed in by the runtime.
MEMBER_SOURCES = {
    "unified_mcp_service": ("mcp", "unified_mcp_service"),
    "workspace_registry_service": ("workspace", "workspace_registry_service"),
    "change_review_consent_service": ("workspace", "change_review_consent_service"),
    "chachanotes_db": ("storage", "chachanotes"),
    "local_character_persona_service": ("character_persona", "local"),
    "prompt_scope_service": ("prompts", "scope"),
    "subscriptions_db": ("watchlists", "subscriptions_db"),
    "local_watchlists_service": ("watchlists", "local"),
    "watchlist_bundle_service": ("watchlists", "watchlist_bundle_service"),
    "library_rag_search_service": ("library", "library_rag_search_service"),
}


@dataclass(frozen=True, slots=True, kw_only=True)
class ConsoleControllerHost:
    """Members and narrow callables the controller's host app supplies.

    Every field is optional. A missing member or capability is ``None``, and
    a reader without an app returns what ``getattr``'s default returned, so
    the controller behaves exactly as it did with a missing app attribute.
    """

    # Service members, from ``app.services`` (``raw_cli_runtime`` is rule 4).
    unified_mcp_service: Any = None
    workspace_registry_service: Any = None
    change_review_consent_service: Any = None
    chachanotes_db: Any = None
    local_character_persona_service: Any = None
    prompt_scope_service: Any = None
    subscriptions_db: Any = None
    local_watchlists_service: Any = None
    watchlist_bundle_service: Any = None
    library_rag_search_service: Any = None
    raw_cli_runtime: Any = None
    # UI capabilities (ADR-250 section 7).
    notify: Callable[..., Any] | None = None
    call_from_thread: Callable[..., Any] | None = None
    post_message: Callable[[Any], Any] | None = None
    on_ui_thread: Callable[[], bool] = lambda: False
    is_headless: Callable[[], bool] = lambda: False
    console_attention: Callable[[int, bool], None] | None = None
    # The fleet wake's durable unseen-completion mark (Chat/console_fleet_attention).
    set_fleet_unseen_completion: Callable[[str], bool] | None = None
    clear_fleet_unseen_completion: Callable[[str], bool] | None = None
    # Readers of app state the bundle does not hold, read when used.
    read_app_config: Callable[[Any], Any] = lambda default=None: default
    read_skills_scope_service: Callable[[], Any] = lambda: None
    read_local_skills_service: Callable[[], Any] = lambda: None
    read_settings_durability_owner: Callable[[], Any] = lambda: None
    read_new_chat_default_generation: Callable[[], Any] = lambda: 0
    read_watchlists_operation_coordinator: Callable[[], Any] = lambda: None
    read_watchlists_command_service: Callable[[], Any] = lambda: None
    read_personal_context_service: Callable[[], Any] | None = None
    # Operations anchored on the app (ADR-250 rule 5 keeps the skills stack there).
    send_refusal: Callable[[str | None], Awaitable[str | None]] | None = None
    capture_turn_configuration: Callable[..., Awaitable[Any]] | None = None
    read_project_index_service: Callable[[], Any] | None = None
    publish_mcp_inspector_counts: Callable[..., None] | None = None
    publish_selected_tool_count: Callable[[str, int], None] | None = None

    @property
    def skills_scope_service(self) -> Any:
        """The app's lazily built skills scope (duck-typed capture reader)."""
        return self.read_skills_scope_service()

    @property
    def local_skills_service(self) -> Any:
        """The app's lazily built local skills service (duck-typed reader)."""
        return self.read_local_skills_service()


def controller_members(services: Any) -> dict[str, Any]:
    """Take the controller's members from an ``ApplicationServices`` bundle."""
    if services is None:
        return {}
    return {
        name: getattr(getattr(services, slice_name), field)
        for name, (slice_name, field) in MEMBER_SOURCES.items()
    }


def controller_host_for_app(app: Any, **members: Any) -> ConsoleControllerHost | None:
    """Build the production host: members from ``app.services``, callables over ``app``.

    ``None`` without an app. An app double that never published a bundle gets
    no members, as a controller with no services had before.
    """
    if app is None:
        return None
    try:
        services = app.services
    except AttributeError:
        services = None
    return console_controller_host(app, **{**controller_members(services), **members})


def console_controller_host(app: Any, **members: Any) -> ConsoleControllerHost:
    """Build the host's callables over ``app`` around the given members."""
    return ConsoleControllerHost(
        **members,
        notify=getattr(app, "notify", None),
        call_from_thread=getattr(app, "call_from_thread", None),
        post_message=getattr(app, "post_message", None),
        on_ui_thread=lambda: threading.get_ident() == getattr(app, "_thread_id", None),
        is_headless=lambda: bool(getattr(app, "is_headless", False)),
        console_attention=functools.partial(_console_attention, app),
        set_fleet_unseen_completion=functools.partial(_fleet_mark, "set", app),
        clear_fleet_unseen_completion=functools.partial(_fleet_mark, "clear", app),
        read_app_config=lambda default=None: getattr(app, "app_config", default),
        read_skills_scope_service=lambda: getattr(app, "skills_scope_service", None),
        read_local_skills_service=lambda: getattr(app, "local_skills_service", None),
        read_settings_durability_owner=lambda: getattr(
            app, "console_settings_durability_owner", None
        ),
        read_new_chat_default_generation=lambda: getattr(
            app, "console_new_chat_default_generation", 0
        ),
        read_watchlists_operation_coordinator=lambda: getattr(
            app, "watchlists_operation_coordinator", None
        ),
        read_watchlists_command_service=lambda: getattr(
            app, "watchlists_command_service", None
        ),
        read_personal_context_service=getattr(app, "get_personal_context_service", None),
        send_refusal=functools.partial(_send_refusal, app),
        capture_turn_configuration=functools.partial(_capture_turn_configuration, app),
        read_project_index_service=functools.partial(_project_index_service, app),
        publish_mcp_inspector_counts=functools.partial(_publish_mcp_inspector_counts, app),
        publish_selected_tool_count=functools.partial(_publish_selected_tool_count, app),
    )


def _console_attention(app: Any, pending: int, ring: bool) -> None:
    """UI THREAD: show the pending-round badge and ring the bell when asked."""
    from .console_interrupt_rounds import apply_console_attention

    apply_console_attention(app, pending, ring)


def _fleet_mark(action: str, app: Any, conversation_id: str) -> bool:
    """Set or clear one conversation's durable unseen-completion mark."""
    from . import console_fleet_attention as attention

    if action == "set":
        return attention.set_fleet_unseen_completion(app, conversation_id)
    return attention.clear_fleet_unseen_completion(app, conversation_id)


async def _send_refusal(app: Any, conversation_id: str | None) -> str | None:
    """Reconcile archive state before a send (``conversation_send_refusal``)."""
    from .conversation_archive_actions import conversation_send_refusal

    return await conversation_send_refusal(app, conversation_id)


async def _capture_turn_configuration(app: Any, controller: Any, session_id: str, **kwargs: Any) -> Any:
    """Run the app-anchored configuration capture for ``controller``."""
    from .console_configuration_preparation import (
        _controller_capture_turn_configuration_snapshot,
    )

    return await _controller_capture_turn_configuration_snapshot(
        app, controller, session_id, **kwargs
    )


def _project_index_service(app: Any) -> Any:
    """The app's project index service, built once off the UI thread."""
    from tldw_chatbook.Workspaces.project_index.runtime import get_project_index_service

    return get_project_index_service(app)


def _publish_mcp_inspector_counts(
    app: Any,
    session_id: str | None,
    builtin_count: Any,
    tool_count: int | None,
    not_connected_count: int | None,
) -> None:
    """Publish one run's catalog counts where the inspector's MCP row reads them."""
    try:
        counts = app.console_selected_tool_counts
    except AttributeError:
        counts = {}
    counts.pop(session_id, None)
    app.console_tool_count = builtin_count
    app.console_mcp_tool_count = tool_count
    app.console_mcp_not_connected_count = not_connected_count


def _publish_selected_tool_count(app: Any, session_id: str, count: int) -> None:
    """Publish the selected first request's tool count for the inspector."""
    try:
        counts = app.console_selected_tool_counts
    except AttributeError:
        counts = {}
    counts[session_id] = count
    app.console_selected_tool_counts = counts


#: The host of a controller built without one: every bridge is a no-op.
NO_HOST = ConsoleControllerHost()
