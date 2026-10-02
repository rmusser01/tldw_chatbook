"""Console adapters for shared Workbench UI state."""

from __future__ import annotations

from tldw_chatbook.Chat.console_display_state import ConsoleControlState
from tldw_chatbook.Chat.console_glyphs import GLYPH_HOOKS
from tldw_chatbook.UI.Workbench.workbench_state import (
    Density,
    WorkbenchAction,
    WorkbenchHeaderState,
    WorkbenchMode,
    WorkbenchPaneState,
    WorkbenchState,
)
from tldw_chatbook.Widgets.glyph_fallback import resolve_glyph


def build_console_workbench_state(
    *,
    control_state: ConsoleControlState,
    provider_blocker_copy: str = "",
    provider_action_label: str = "Open Settings",
    can_send: bool = False,
    can_stop: bool = False,
    density: str = "normal",
    run_active: bool = False,
    ephemeral: bool = False,
    hook_attention: int = 0,
    readiness_word: str = "",
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
        hook_attention: Enabled hooks or permission errors needing attention.
        readiness_word: The active chat's spec §5 readiness word
            (TASK-33005.3); the header badge shows it unless a run is active.
        density: Requested Workbench density, currently ``normal`` or ``compact``.
        ephemeral: Whether the active session is temporary. Retained for
            callers even though no top action reads it today: Save Chatbook
            (the last consumer) left this strip for the ☰ composer menu and
            the Inspector's Artifacts row, which carry their own
            temporary-chat block.

    Returns:
        Immutable shared Workbench state used by Console widgets.
    """
    blocker = provider_blocker_copy.strip()
    workbench_density: Density = "compact" if density == "compact" else "normal"
    provider_status = "blocked" if blocker else "ready"
    send_available = can_send and not blocker

    # Save Chatbook is deliberately absent here: as a top-strip button it
    # was disabled at rest for most sessions (no artifact yet), spending a
    # always-visible slot on an almost-always-inert control. The action
    # stays reachable from the ☰ composer menu and the Inspector's
    # Artifacts row, both of which carry availability copy.
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
            id="hooks",
            label=resolve_glyph(GLYPH_HOOKS)
            + (f" {hook_attention}" if hook_attention else ""),
            tooltip=f"Hook permissions: {hook_attention} need review"
            if hook_attention
            else "Review hook permissions",
        ),
        WorkbenchAction(
            id="attach-context",
            # TASK-32325: the label says what the button DOES (open the
            # rail); the tooltip carries the staging pointer. "Attach
            # context" claimed an attachment this control never made, and
            # with the rail already open by default the click looked like
            # a dead button.
            label="Context rail",
            tooltip=("Open the Console context rail; stage sources from Library"),
        ),
        WorkbenchAction(
            id="run-library-rag",
            label="Search Library",
            tooltip="Search Library evidence before sending",
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
        WorkbenchMode(id="assistant", label=control_state.assistant_label),
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
            subtitle="— Chat, source handoffs, live runs, and control actions.",
            # TASK-347: a live generation must not read "Ready". A run only
            # runs once past the blocker gate, so running takes precedence.
            status="running" if run_active else ("blocked" if blocker else "ready"),
            density=workbench_density,
            # TASK-33005.3: the badge is the status row's readiness word, so
            # the strip below keeps its width for the context/cost chip.
            status_label="" if run_active else readiness_word,
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
