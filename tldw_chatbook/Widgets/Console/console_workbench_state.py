"""Console adapters for shared Workbench UI state."""

from __future__ import annotations

from tldw_chatbook.Chat.console_display_state import ConsoleControlState
from tldw_chatbook.UI.Workbench.workbench_state import (
    Density,
    WorkbenchAction,
    WorkbenchHeaderState,
    WorkbenchMode,
    WorkbenchPaneState,
    WorkbenchState,
)


def build_console_workbench_state(
    *,
    control_state: ConsoleControlState,
    provider_blocker_copy: str = "",
    provider_action_label: str = "Open Settings",
    can_send: bool = False,
    can_stop: bool = False,
    can_save_chatbook: bool = False,
    density: str = "normal",
) -> WorkbenchState:
    """Return a shared Workbench state snapshot for Console.

    Args:
        control_state: Current Console control labels and readiness state.
        provider_blocker_copy: Provider/setup blocker copy, if send is blocked.
            Only used to derive header/mode "blocked" status; the shared
            Workbench recovery banner is never populated from it. First-run
            and setup guidance now live in the empty-transcript setup card
            and the composer disabled-reason (see the Phase 2 spec, section
            2), so this state never duplicates that guidance in a banner.
        provider_action_label: Reserved for callers; unused by this function
            now that Workbench recovery is never populated. Kept so existing
            call sites do not need to change.
        can_send: Whether the visible composer draft can be sent.
        can_stop: Whether an active generation can be stopped.
        can_save_chatbook: Whether the current session can be saved as a Chatbook.
        density: Requested Workbench density, currently ``normal`` or ``compact``.

    Returns:
        Immutable shared Workbench state used by Console widgets.
    """
    blocker = provider_blocker_copy.strip()
    workbench_density: Density = "compact" if density == "compact" else "normal"
    provider_status = "blocked" if blocker else "ready"
    send_available = can_send and not blocker

    actions = (
        WorkbenchAction(
            id="new-tab",
            label="New tab",
            tooltip="Create a Console tab",
        ),
        WorkbenchAction(
            id="settings",
            label="Settings",
            tooltip="Configure provider, model, tools, and generation",
        ),
        WorkbenchAction(
            id="attach-context",
            label="Attach context",
            tooltip="Stage Library or workspace context",
        ),
        WorkbenchAction(
            id="run-library-rag",
            label="Run Library RAG",
            tooltip="Search Library evidence before sending",
        ),
        WorkbenchAction(
            id="save-chatbook",
            label="Save Chatbook",
            tooltip="Save this run as a Chatbook",
            disabled=not can_save_chatbook,
        ),
        WorkbenchAction(
            id="send",
            label="Send",
            tooltip="Send composer draft",
            disabled=not send_available,
            primary=send_available,
        ),
        WorkbenchAction(
            id="stop",
            label="Stop",
            tooltip="Stop active generation",
            disabled=not can_stop,
        ),
        WorkbenchAction(
            id="help",
            label="Help",
            tooltip="Show visible Console actions and shortcuts",
        ),
    )
    modes = (
        WorkbenchMode(
            id="provider",
            label=control_state.provider_label,
            active=True,
            status=provider_status,
        ),
        WorkbenchMode(
            id="model",
            label=control_state.model_label,
            status=provider_status,
        ),
        WorkbenchMode(id="persona", label=control_state.persona_label),
        WorkbenchMode(id="rag", label=control_state.rag_label),
        WorkbenchMode(id="sources", label=control_state.sources_label),
        WorkbenchMode(id="tools", label=control_state.tools_label),
        WorkbenchMode(id="approvals", label=control_state.approvals_label),
    )

    # Note: the shared Workbench `recovery` banner is intentionally never
    # populated here. The empty-transcript setup card and the composer
    # blocked-reason now own first-run/provider-setup guidance; surfacing it
    # again here would duplicate that guidance in a second top-level banner
    # (see the Phase 2 spec, section 2).
    return WorkbenchState(
        route_id="chat",
        density=workbench_density,
        header=WorkbenchHeaderState(
            title="Console",
            subtitle="Chat, source handoffs, live runs, and control actions.",
            status="blocked" if blocker else "ready",
            density=workbench_density,
        ),
        modes=modes,
        actions=actions,
        panes=(
            WorkbenchPaneState(id="context", title="Context"),
            WorkbenchPaneState(id="transcript", title="Transcript"),
            WorkbenchPaneState(id="inspector", title="Inspector"),
            WorkbenchPaneState(id="composer", title="Composer"),
        ),
        recovery=None,
    )
