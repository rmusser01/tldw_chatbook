"""Conversation Inspector: lazy context, usage, and historical capture readers.

Services retain snapshot, accounting, and export authority. This modal binds every
asynchronous result to the conversation and disclosure generation captured at entry.
Live automatic project instructions may appear only in the disposable Next Send
preview; whole-payload exports continue to scrub those messages.
"""

from __future__ import annotations

import asyncio
import copy
import json
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from loguru import logger
from rich.text import Text
from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.css.query import NoMatches
from textual.reactive import reactive
from textual.screen import ModalScreen
from textual.widgets import (
    Button,
    Checkbox,
    LoadingIndicator,
    Static,
    TabbedContent,
    TabPane,
    TextArea,
)
from textual.worker import Worker, WorkerState

from tldw_chatbook.Chat.console_chat_models import ConsoleContextSnapshot
from tldw_chatbook.Chat.console_cost_tracker import ConsoleCostRow, ConsoleCostRowTotals
from tldw_chatbook.Chat.console_display_state import ConsoleProjectInstructionState
from tldw_chatbook.Chat.console_ephemeral import blocked_reason
from tldw_chatbook.Chat.console_exchange_capture import (
    ExchangeCapture,
    history_elision_marker,
)
from tldw_chatbook.Chat.console_project_instructions import EPHEMERAL_ORIGIN_KEY
from tldw_chatbook.Chat.console_trace_projection import project_capture_for_viewer
from tldw_chatbook.Chat.provider_usage import ProviderUsage
from tldw_chatbook.LLM_Calls.pricing_catalog import get_pricing_catalog
from tldw_chatbook.Utils.log_sanitizer import content_fingerprint
from tldw_chatbook.Utils.path_validation import validate_path
from tldw_chatbook.Widgets.confirmation_dialog import ConfirmationDialog
from tldw_chatbook.Widgets.Console.console_capture_policy_dialog import (
    CapturePolicyBindings,
    ConsoleTracePrivacyDialog,
)
from tldw_chatbook.Widgets.Console.console_project_instructions import (
    ConsoleProjectInstructionContextPanel,
)
from tldw_chatbook.Widgets.modal_dismissal import SafeModalDismissMixin
from tldw_chatbook.Widgets.pausable_progress import PausableLoadingIndicator

from .console_inspector_detail_pane import ConsoleInspectorDetailPane
from .console_inspector_presentation import (
    InspectorSection,
    context_detail,
    context_sections,
    usage_items,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from tldw_chatbook.Chat.trace_export_profiles import TraceViewerProfile
else:
    TraceViewerProfile = Any

MODAL_ID = "console-inspector-modal"
CLOSE_BUTTON_ID = "console-inspector-close"
VIEWER_PROFILE_BUTTON_ID = "console-inspector-viewer-profile"
TAB_COSTS = "inspector-costs"
TAB_EXCHANGE = "inspector-exchange"
TAB_NEXT_SEND = "inspector-next-send"

# Next Send tab (task-10, ported from the retired standalone context
# modal): the same 1 MiB raw-JSON size guard that modal used before
# rendering the assembled next-send payload as a giant ``TextArea`` --
# past this size Save to File is offered instead of trying to render it
# inline.
SIZE_THRESHOLD_BYTES = 1 * 1024 * 1024

# Next Send tab (task-10 review finding 2): the snapshot-load worker's own
# ``run_worker`` group -- kept OUT of Textual's "default" group (which the
# Costs tab's ``_load_turn_captures`` and the Exchange tab's ``_load_
# exchange_turn`` both land in, since neither passes ``group=``) for two
# reasons. (a) ``exclusive=True`` cancels every OTHER worker in the SAME
# group -- left at "default", a Next Send Refresh/"r" would cancel an
# in-flight Costs/Exchange capture load whose row/turn index is already
# marked loaded (``_loaded_row_indices``/``_loaded_exchange_turn_
# indices``), permanently emptying that row with no retry short of
# reopening the modal. (b) ``on_worker_state_changed`` below filters on
# this same group, so a Costs/Exchange loader failure (which already
# handles its own error internally and never reaches ERROR state, but
# should not be trusted to stay that way forever) can't produce this tab's
# "Failed to refresh context." toast or clear ITS spinner.
_NEXT_SEND_WORKER_GROUP = "console-inspector-next-send"

_EXCHANGE_ADAPTER_BOUNDARY_CAVEAT = (
    "Captured where Console hands the request to the provider adapter, not "
    "at the raw HTTP layer -- provider-internal framing and injected "
    "prompt-cache markers are not visible here (llama.cpp is the exception: "
    "its capture is the literal wire payload)."
)


@dataclass(frozen=True)
class InspectorTurn:
    """One transcript row's identity, for mapping a Costs-tab drill-in back
    to the message its captures belong to.

    Attributes:
        message_id: Persisted (durable) message id, or ``""`` for a
            message that has never been persisted (an ephemeral session).
        native_message_id: The in-memory ``ConsoleChatMessage.id`` --
            always present, unlike ``message_id``. This is what
            ``exchanges_loader`` is actually called with (see the module
            docstring): it is a stable key regardless of persistence
            state, so the loader can check the native store first and
            fall back to a DB read using the SAME message.
        index: 0-based transcript position, matching the ``index`` on the
            ``ConsoleCostRow`` this turn corresponds to (both are built
            from the same transcript-ordered message list by the same
            caller -- see ``build_cost_rows``).
        role: The message's role (``"user"``, ``"assistant"``, ...).
        preview: A short (caller-truncated) preview of the message's
            content, for a future richer row label.
    """

    message_id: str
    native_message_id: str
    index: int
    role: str
    preview: str


ExchangesLoader = Callable[[str], Awaitable[list[tuple[ExchangeCapture, bool]]]]
SnapshotFactory = Callable[[], Awaitable[ConsoleContextSnapshot]]


class ConsoleConversationInspector(SafeModalDismissMixin, ModalScreen[None]):
    """Unified Console conversation inspector: Costs / Exchange / Next Send.

    Every input is precomputed by the caller (``chat_screen.py``) and
    handed in at construction -- this widget never reaches into the
    Console store or DB directly except through the injected
    ``exchanges_loader``/``snapshot_factory`` callables, mirroring the
    retired standalone cost modal's "already computed, just render it"
    shape for the Costs tab's rows/totals.
    """

    BUNDLED_CSS = """
    ConsoleConversationInspector { align: center middle; }
    #console-inspector-modal {
        width: 110; max-width: 95%; height: 42; max-height: 90%;
        border: tall gray; padding: 1 2;
    }
    #console-inspector-header { height: auto; }
    #console-inspector-policy-status { height: auto; color: $text-muted; }
    #console-inspector-tabs { height: 1fr; margin-top: 1; }
    #console-inspector-costs-rows { height: 1fr; }
    .console-inspector-cost-row { height: auto; }
    #console-inspector-costs-totals { height: auto; margin-top: 1; text-style: bold; }
    #console-inspector-exchange-caveat { height: auto; color: gray; margin-bottom: 1; }
    #console-inspector-capture-status { height: auto; color: yellow; }
    #console-inspector-exchange-turns { height: 1fr; }
    .console-inspector-exchange-turn { height: auto; }
    .console-inspector-exchange-call { height: auto; }
    .console-inspector-exchange-section { height: auto; }
    .console-inspector-exchange-message { height: auto; }
    .console-inspector-exchange-call-actions { height: auto; margin-bottom: 1; }
    #console-inspector-actions { height: auto; margin-top: 1; }
    /* Next Send tab (task-10, ported from the retired context modal).
       LY-13 (TASK-2154.23) compacted the OLD modal's own top-level frame
       to content when there was nothing to show yet; DROPPED here (task-10
       review finding 1) -- the outer modal frame is now shared with the
       Costs/Exchange tabs and stays a fixed height regardless of this
       pane's state, so there was nothing left for a pane-scoped "auto"
       height to compact (measured identically empty vs. populated). */
    #console-inspector-next-send-pane { height: 1fr; }
    #console-inspector-next-send-header { height: auto; }
    #console-inspector-next-send-warning { height: auto; color: yellow; }
    #console-inspector-next-send-loading { display: none; }
    #console-inspector-next-send-loading.loading { display: block; }
    #console-inspector-next-send-tabs { height: 1fr; }
    #console-inspector-next-send-actions { height: auto; }
    """

    # Deliberate divergence from a literal "dismiss" action name (this
    # modal has no dirty state to guard, so the OBSERVABLE behavior would
    # be identical either way): every other Console modal in this codebase
    # is built on SafeModalDismissMixin's request_safe_cancel (backdrop
    # click + one-shot cancellation), and Tests/UI/test_console_modal_
    # dismissal.py enforces that as an app-wide, AST-verified contract --
    # see this task's report for the full reasoning.
    BINDINGS: ClassVar[list[tuple[str, str, str]]] = [
        ("escape", "request_safe_cancel", "Close"),
        ("r", "refresh", "Refresh"),
        ("c", "capture_policy", "Capture"),
        ("v", "viewer_profile", "Trace view"),
    ]
    SAFE_MODAL_CONTENT = f"#{MODAL_ID}"

    # Next Send tab (task-18300, ported from the retired standalone context
    # modal): with a project-instruction recovery action available, letting
    # Textual's default AUTO_FOCUS land on whatever's first in DOM order can
    # skip right past it. ``_focus_initial_control`` (called from
    # ``on_mount``, after a snapshot refresh, and after a recovery decision;
    # I1 narrowed it to only act while the Next Send tab is active) picks
    # the single most relevant control instead, without shifting any
    # layout.
    #
    # M3: kept for fidelity with the retired modal this was ported from, but
    # this assignment is itself a no-op against installed Textual (8.2.8) --
    # it does NOT prevent Textual's own default auto-focus. ``Screen.
    # AUTO_FOCUS`` (this class's base, via ``ModalScreen``) is already
    # ``None``, so setting it here changes nothing; and ``None`` on a Screen
    # does not disable auto-focus outright, it defers to ``App.AUTO_FOCUS``
    # (``"*"``, Textual's own default), which still runs. Whatever
    # prevention exists is entirely ``_focus_initial_control``'s doing.
    AUTO_FOCUS = None

    # Next Send tab (task-10) reactives, ported from the retired standalone
    # context modal.
    # task-16843: a bare instance default (`reactive(ConsoleContextSnapshot(...))`)
    # installs the SAME snapshot object on every modal instance until
    # `_load_snapshot` reassigns it -- `frozen=True` on the dataclass only
    # blocks reassigning its `current_messages`/`next_send_payload` fields, not
    # mutating the list/dict those fields point to in place. A callable
    # default gives each instance its own snapshot (and its own empty
    # list/dict) instead.
    snapshot = reactive(
        lambda: ConsoleContextSnapshot(current_messages=[], next_send_payload={})
    )
    raw_json = reactive(False)
    # Named ``next_send_loading``, not ``loading`` -- ``Widget`` (this
    # class's own base, via ``ModalScreen``) already declares a built-in
    # ``loading`` reactive with unrelated semantics (a whole-widget loading
    # OVERLAY). Shadowing it with this pane's own bool collided with
    # Textual's internal ``loading`` reads (e.g. ``Screen.update_pointer_
    # shape``) walking the ancestor chain and invoking THIS reactive's
    # ``init=True`` watcher before/after this pane's own DOM subtree was
    # around to query -- a real ``NoMatches`` observed while porting this.
    next_send_loading = reactive(False)

    def __init__(
        self,
        *,
        conversation_title: str,
        target_profile_key: str,
        target_is_current: Callable[[], bool],
        rows: Sequence[ConsoleCostRow],
        totals: ConsoleCostRowTotals,
        turns: Sequence[InspectorTurn],
        exchanges_loader: ExchangesLoader,
        snapshot_factory: SnapshotFactory,
        token_estimate: int | None = None,
        estimate_factory: Callable[[], int | None] | None = None,
        payload_estimate: Callable[[ConsoleContextSnapshot], int | None] | None = None,
        in_progress: bool = False,
        ephemeral: bool = False,
        project_instruction_state: ConsoleProjectInstructionState | None = None,
        project_instruction_state_factory: Callable[
            [], Awaitable[ConsoleProjectInstructionState]
        ]
        | None = None,
        project_instruction_session_id: str | None = None,
        project_instruction_recovery: Callable[
            [str | None, str], Awaitable[ConsoleProjectInstructionState | None]
        ]
        | None = None,
        target_session_id: str | None = None,
        target_conversation_id: str | None = None,
        capture_revision_provider: Callable[[], int | None] | None = None,
        capture_policy_bindings: CapturePolicyBindings | None = None,
        initial_tab: str = TAB_NEXT_SEND,
        context_budget_provider: Callable[[], tuple[int | None, int | None]]
        | None = None,
    ) -> None:
        """Initialize the inspector.

        Args:
            rows: Precomputed per-message cost rows (``build_cost_rows``'s
                output) for the Costs tab.
            totals: Precomputed aggregate totals (``build_cost_rows_totals``'s
                output) for the same ``rows``.
            turns: One :class:`InspectorTurn` per contributing message,
                index-aligned with ``rows`` via ``InspectorTurn.index ==
                ConsoleCostRow.index`` (both transcript-ordered).
            exchanges_loader: Async, called with one turn's
                ``native_message_id`` on first expand of that turn's Costs
                row; returns ``(capture, abandoned)`` pairs (see the module
                docstring's loader-contract note).
            snapshot_factory: Async, builds the Next Send tab's context
                snapshot. Called once on mount (and again on Refresh/"r"
                while that tab is active) via ``_load_snapshot``.
            token_estimate: Precomputed token estimate for the Next Send
                tab's header, or ``None``.
            estimate_factory: Re-estimate callback for a Next Send refresh,
                or ``None``.
            payload_estimate: task-25836 -- optional callback computing the
                header count from a LOADED snapshot (the whole next-send
                request: system row, messages incl. the draft turn, tool
                schemas, staged evidence), preferred over
                ``estimate_factory`` once the snapshot arrives; ``None``
                keeps the draft-only ``estimate_factory`` contract exactly
                as before.
            in_progress: Whether a response is currently in flight (shows
                the Next Send tab's in-progress warning line and disables
                its Refresh button).
            ephemeral: Whether the active session is temporary (blocks
                Next Send's Save-to-file affordance via ``blocked_reason``).
            project_instruction_state: Metadata-only project-instruction
                display state (task-18300, ported from the retired
                standalone context modal). ``None`` (the default, and what
                the cost-chip entry point always passes) mounts no project
                instructions panel on the Next Send tab at all.
            project_instruction_state_factory: Async re-fetch of the above,
                called from ``_load_snapshot`` alongside ``snapshot_factory``
                on mount and on Refresh/"r".
            project_instruction_session_id: The session captured when this
                modal was opened, threaded through to
                ``ConsoleProjectInstructionContextPanel`` and echoed back on
                a ``RecoveryRequested`` event so recovery always targets the
                session that was active when the panel was built, not
                whatever is active by the time the user acts.
            project_instruction_recovery: Async handler for one explicit
                recovery decision (enable / choose folder / disable),
                returning the refreshed display state or ``None`` when the
                captured session or action is no longer valid.
            target_session_id: Immutable session identity captured when the
                Inspector opens.
            target_conversation_id: Immutable persisted-conversation identity
                captured when the Inspector opens.
            capture_revision_provider: Optional process-local revision reader
                used to fail closed when cached captures become stale.
            initial_tab: Which tab id starts active -- ``"inspector-costs"``
                from the cost chip, ``"inspector-next-send"`` from Ctrl+Shift+P.
        """
        from tldw_chatbook.Chat.trace_export_profiles import TraceViewerProfile

        super().__init__()
        self._conversation_title = conversation_title
        self._target_profile_key = target_profile_key
        self._target_is_current = target_is_current
        self._snapshot_generation = 0
        self._disclosure_generation = 0
        self._snapshot_ready = False
        self._snapshot_status = "Preview not prepared"
        self._target_invalid = False
        self._rows = list(rows)
        self._usage_items = usage_items(rows, turns)
        self._context_budget_provider = context_budget_provider
        self._context_budget = (None, None)
        self._detail_generation = 0
        self._trace_selections: dict[str, str] = {}
        self._trace_calls: dict[str, tuple[tuple[str, ExchangeCapture, bool], ...]] = {}
        self._selected_export_key: str | None = None
        self._totals = totals
        # Review finding M5 (and its own regression, closed by the final
        # re-review): `turns` is index-aligned with `rows` via
        # InspectorTurn.index == ConsoleCostRow.index, but the CALLER
        # (chat_screen.py's _build_console_inspector_cost_data) builds one
        # turn per transcript MESSAGE, while `build_cost_rows` skips
        # non-contributing ones -- e.g. a bare tool-result message.
        # Unfiltered, the Exchange tab showed a Collapsible for every
        # message, half of them permanently reading "No capture recorded
        # for this turn". A turn is kept when EITHER half holds:
        #   - it has a matching cost row (`build_cost_rows`'s own
        #     contributing set) -- de-clutters rows that never had
        #     anything to show, same set the Costs tab already renders; OR
        #   - it is an assistant turn, regardless of a cost row -- Stop
        #     pressed before the first token still marks and persists the
        #     message and still flushes a "stopped" capture
        #     (console_chat_controller.py), but `build_cost_rows` emits no
        #     row for it (blank content, no usage). M5's `and`-only
        #     predicate dropped that turn from the Exchange tab entirely,
        #     making "what did I send that hung?" unreachable -- exactly
        #     the capture this tab exists to surface. Never hide an
        #     assistant turn on cost-row absence alone.
        contributing_indices = {row.index for row in self._rows}
        self._turns_by_index = {
            turn.index: turn
            for turn in turns
            if turn.index in contributing_indices or turn.role == "assistant"
        }
        self._exchanges_loader = exchanges_loader
        self._snapshot_factory = snapshot_factory
        self._token_estimate = token_estimate
        self._estimate_factory = estimate_factory
        self._payload_estimate = payload_estimate
        self._in_progress = in_progress
        self._ephemeral = ephemeral
        self._project_instruction_state = project_instruction_state
        self._project_instruction_state_factory = project_instruction_state_factory
        self._project_instruction_session_id = project_instruction_session_id
        self._project_instruction_recovery = project_instruction_recovery
        self._target_session_id = target_session_id
        self._target_conversation_id = target_conversation_id
        self._capture_revision_provider = capture_revision_provider
        self._capture_policy_bindings = capture_policy_bindings
        # Production wiring always supplies capture-policy bindings. The Full
        # fallback preserves the isolated presentation harness contract only;
        # an unbound Inspector is not reachable from the app.
        self._viewer_profile = (
            TraceViewerProfile.FULL
            if capture_policy_bindings is None
            else TraceViewerProfile.SAFE
        )
        if capture_policy_bindings is not None:
            try:
                self._viewer_profile = TraceViewerProfile(
                    getattr(capture_policy_bindings.read(), "viewer_profile", "safe")
                )
            except Exception:  # noqa: BLE001 - injected authority/loaders fail closed
                self._viewer_profile = TraceViewerProfile.SAFE
        try:
            self._capture_revision_at_open = (
                capture_revision_provider()
                if capture_revision_provider is not None
                else None
            )
        except Exception:  # noqa: BLE001 - injected authority/loaders fail closed
            self._capture_revision_at_open = None
        self._initial_tab = initial_tab or TAB_NEXT_SEND
        self._exchange_capture_by_call_key: dict[str, tuple[ExchangeCapture, bool]] = {}
        self._save_blocked_reason = blocked_reason("save-context", ephemeral=ephemeral)

    def _capture_revision_is_current(self) -> bool:
        """Return whether cached captures still belong to the open revision."""
        if not self._target_authority_is_current():
            return False
        provider = self._capture_revision_provider
        if provider is None:
            return True
        try:
            current = provider()
        except Exception:  # noqa: BLE001 - injected authority/loaders fail closed
            current = None
        if current is None:
            self._invalidate_stale_captures()
            return False
        if self._capture_revision_at_open is None:
            # An Inspector may open while purge owns the quiescence lease. It
            # must fail closed during that interval, then adopt the first real
            # post-lease revision before it has loaded any capture bodies.
            self._capture_revision_at_open = current
            return True
        if current == self._capture_revision_at_open:
            return True
        self._invalidate_stale_captures()
        return False

    def _clear_trace_details(self) -> None:
        """Remove both rendered bodies and projected caches before disclosure changes."""
        self._disclosure_generation += 1
        self._trace_calls.clear()
        self._exchange_capture_by_call_key.clear()
        self._selected_export_key = None
        for pane_id in (
            "console-inspector-usage-detail",
            "console-inspector-exchange-detail",
        ):
            for pane in self.query(f"#{pane_id}"):
                pane.clear_detail("Capture view changed — select a turn again")
        for button in self.query(
            "#console-inspector-export-call, #console-inspector-show-exchanges"
        ):
            button.disabled = True
        for pane in self.query("#console-inspector-exchange-detail"):
            pane.set_sections(self._exchange_sections())

    def _invalidate_stale_captures(self) -> None:
        self._clear_trace_details()
        for status in self.query("#console-inspector-capture-status"):
            status.update("Stored captures changed · close and reopen Inspector")

    async def _invalidate_stale_exchange_mounts(self) -> None:
        """Retain the existing capture-policy purge callback contract."""
        self._invalidate_stale_captures()

    def compose(self) -> ComposeResult:
        """Keep navigation and actions outside the independently scrolling panes."""
        with Vertical(id=MODAL_ID):
            yield Static(
                Text(f"Conversation Inspector · {self._conversation_title}"),
                id="console-inspector-header",
            )
            yield Static(
                self._capture_policy_text(),
                id="console-inspector-policy-status",
                markup=False,
            )
            with TabbedContent(id="console-inspector-tabs", initial=self._initial_tab):
                with TabPane("Context", id=TAB_NEXT_SEND):  # noqa: SIM117 - mirrors the widget hierarchy
                    with Vertical(id="console-inspector-next-send-pane"):
                        yield Static(
                            "Preparing preview…",
                            id="console-inspector-next-send-header",
                            markup=False,
                        )
                        yield Static(
                            "", id="console-inspector-next-send-warning", markup=False
                        )
                        if self._project_instruction_state is not None:
                            with VerticalScroll(
                                id="console-inspector-project-recovery"
                            ):
                                yield ConsoleProjectInstructionContextPanel(
                                    self._project_instruction_state,
                                    session_id=self._project_instruction_session_id,
                                    id="console-context-project-instructions",
                                )
                        yield PausableLoadingIndicator(
                            id="console-inspector-next-send-loading"
                        )
                        yield ConsoleInspectorDetailPane(
                            id="console-inspector-context-detail"
                        )
                        with Horizontal(id="console-inspector-next-send-actions"):
                            yield Checkbox(
                                "Raw JSON",
                                id="console-inspector-next-send-raw",
                                compact=True,
                                classes="inspector-modal-control",
                            )
                            yield Button(
                                "Refresh",
                                id="console-inspector-next-send-refresh",
                                disabled=self._in_progress,
                                compact=True,
                                classes="inspector-modal-control",
                            )
                            yield Button(
                                "Copy payload",
                                id="console-inspector-next-send-copy",
                                compact=True,
                                classes="inspector-modal-control",
                            )
                            save = Button(
                                "Save payload",
                                id="console-inspector-next-send-save",
                                disabled=self._save_blocked_reason is not None,
                                compact=True,
                                classes="inspector-modal-control",
                            )
                            save.tooltip = (
                                self._save_blocked_reason
                                or "Export the full prepared payload; automatic project instructions are omitted."
                            )
                            yield save
                with TabPane("Usage & cost", id=TAB_COSTS):
                    yield Static(
                        self._format_totals(self._totals),
                        id="console-inspector-costs-totals",
                        markup=False,
                    )
                    yield Static(
                        "Transcript usage · captured call costs below are details, not additional spend",
                        classes="inspector-view-note",
                        markup=False,
                    )
                    yield ConsoleInspectorDetailPane(
                        tuple(
                            InspectorSection(f"usage:{ordinal}", "Turn", item.title)
                            for ordinal, item in enumerate(self._usage_items)
                        ),
                        id="console-inspector-usage-detail",
                        classes="inspector-usage-detail",
                    )
                    yield Button(
                        "Inspect calls in Exchange history",
                        id="console-inspector-show-exchanges",
                        compact=True,
                        classes="inspector-modal-control",
                        disabled=True,
                    )
                with TabPane("Exchange history", id=TAB_EXCHANGE):
                    yield Static(
                        "", id="console-inspector-capture-status", markup=False
                    )
                    caveat = Static(
                        "Provider-adapter capture; HTTP framing may differ. llama.cpp captures its sent payload.",
                        id="console-inspector-exchange-caveat",
                        markup=False,
                    )
                    caveat.tooltip = _EXCHANGE_ADAPTER_BOUNDARY_CAVEAT
                    yield caveat
                    yield ConsoleInspectorDetailPane(
                        self._exchange_sections(),
                        id="console-inspector-exchange-detail",
                    )
                    yield Button(
                        "Export selected call…",
                        id="console-inspector-export-call",
                        compact=True,
                        classes="inspector-modal-control",
                        disabled=True,
                    )
            with Horizontal(id="console-inspector-actions"):
                yield Button(
                    f"View: {self._viewer_profile.value.title()}",
                    id=VIEWER_PROFILE_BUTTON_ID,
                    compact=True,
                    classes="inspector-modal-control",
                )
                yield Button(
                    "Capture settings…",
                    id="console-inspector-capture-settings",
                    compact=True,
                    classes="inspector-modal-control",
                    disabled=self._capture_policy_bindings is None,
                )
                yield Button(
                    "Close", id=CLOSE_BUTTON_ID, variant="primary", compact=True
                )

    def _exchange_sections(self) -> tuple[InspectorSection, ...]:
        sections = []
        for turn in self._turns_by_index.values():
            sections.append(
                InspectorSection(
                    f"turn:{turn.index}", "Turn", f"{turn.index + 1} · {turn.role}"
                )
            )
            for key, capture, abandoned in self._trace_calls.get(
                turn.native_message_id, ()
            ):
                sections.append(
                    InspectorSection(
                        f"call:{key}",
                        "Call",
                        self._exchange_call_title(capture, abandoned),
                    )
                )
        return tuple(sections)

    @on(ConsoleInspectorDetailPane.SectionSelected)
    def _section_selected(
        self, event: ConsoleInspectorDetailPane.SectionSelected
    ) -> None:
        event.stop()
        if not self._target_authority_is_current():
            return
        pane = event.pane
        if pane.id == "console-inspector-context-detail":
            self._detail_generation += 1
            self.run_worker(
                self._show_context_detail(event.key, self._detail_generation),
                exclusive=True,
                group="inspector-context-detail",
            )
        else:
            self._trace_selections[pane.id] = event.key
            self.run_worker(
                self._show_trace_detail(pane.id, event.key),
                exclusive=True,
                group=pane.id,
            )

    async def _show_context_detail(self, key: str, generation: int) -> None:
        if not self._snapshot_ready:
            return
        snapshot = self.snapshot
        detail = await asyncio.to_thread(context_detail, snapshot, key)
        if (
            generation != self._detail_generation
            or snapshot is not self.snapshot
            or not self.is_mounted
            or not self._target_authority_is_current()
        ):
            return
        pane = self.query_one(
            "#console-inspector-context-detail", ConsoleInspectorDetailPane
        )
        section = next((item for item in pane.sections if item.key == key), None)
        if section is not None:
            pane.set_detail(
                key,
                f"{section.group} · {section.label}",
                detail.raw_json
                if self.raw_json and detail.raw_json is not None
                else detail.text,
            )

    async def _show_trace_detail(self, pane_id: str, key: str) -> None:
        if pane_id == "console-inspector-usage-detail":
            self.query_one("#console-inspector-show-exchanges", Button).disabled = True
        if not self._capture_revision_is_current():
            return
        generation = self._disclosure_generation
        pane = self.query_one(f"#{pane_id}", ConsoleInspectorDetailPane)
        self._selected_export_key = None
        self.query_one("#console-inspector-export-call", Button).disabled = True
        if key.startswith("call:"):
            call = self._exchange_capture_by_call_key.get(key[5:])
            if call is None:
                return
            capture, abandoned = call
            text = await asyncio.to_thread(self._call_detail_text, capture, abandoned)
            if (
                generation != self._disclosure_generation
                or not self.is_mounted
                or not self._capture_revision_is_current()
                or self._trace_selections.get(pane_id) != key
            ):
                return
            pane.set_detail(key, self._exchange_call_title(capture, abandoned), text)
            self._selected_export_key = key[5:]
            self.query_one("#console-inspector-export-call", Button).disabled = False
            return
        usage_item = (
            self._usage_items[int(key[6:])] if key.startswith("usage:") else None
        )
        turn = (
            self._turns_by_index.get(int(key[5:])) if key.startswith("turn:") else None
        )
        message_id = (
            usage_item.message_key
            if usage_item
            else (turn.native_message_id if turn else "")
        )
        prefix = self._usage_detail_text(usage_item.row) if usage_item else ""
        if not message_id:
            pane.set_detail(
                key,
                "Usage detail",
                prefix
                + "\nCapture detail unavailable: missing or ambiguous message identity.",
            )
            return
        pane.set_detail(key, "Loading captured calls…", prefix)
        try:
            if message_id not in self._trace_calls:
                captures = await self._exchanges_loader(message_id)
                if (
                    generation != self._disclosure_generation
                    or not self.is_mounted
                    or not self._capture_revision_is_current()
                ):
                    return
                calls = []
                for ordinal, (capture, abandoned) in enumerate(
                    sorted(captures, key=lambda pair: (pair[0].created_at, pair[0].seq))
                ):
                    projected = (
                        project_capture_for_viewer(capture, self._viewer_profile)
                        if self._capture_policy_bindings is not None
                        else capture
                    )
                    call_key = f"{message_id}:{ordinal}"
                    calls.append((call_key, projected, abandoned))
                    self._exchange_capture_by_call_key[call_key] = (
                        projected,
                        abandoned,
                    )
                self._trace_calls[message_id] = tuple(calls)
            if self._trace_selections.get(pane_id) != key:
                return
            calls = self._trace_calls[message_id]
            detail = "\n\n".join(
                self._exchange_call_title(capture, abandoned)
                + "\n"
                + self._call_cost_line(capture)
                for _, capture, abandoned in calls
            )
            pane.set_detail(
                key,
                "Usage & captured calls" if usage_item else "Captured calls",
                prefix
                + "\n\n"
                + (
                    detail
                    or "No capture recorded for this turn. Capture may have been off, or stored history may have been purged."
                ),
            )
            self.query_one(
                "#console-inspector-exchange-detail", ConsoleInspectorDetailPane
            ).set_sections(self._exchange_sections())
            if usage_item:
                self.query_one(
                    "#console-inspector-show-exchanges", Button
                ).disabled = not calls
        except Exception:  # noqa: BLE001 - injected authority/loaders fail closed
            if (
                generation == self._disclosure_generation
                and self.is_mounted
                and self._capture_revision_is_current()
            ):
                pane.set_detail(
                    key,
                    "Capture unavailable",
                    prefix
                    + "\n\nCould not load captures. Select this turn again to retry.",
                )

    @staticmethod
    def _usage_detail_text(row: ConsoleCostRow) -> str:
        cost = (
            "Unavailable (unpriced)" if row.cost_usd is None else f"${row.cost_usd:.4f}"
        )
        return (
            f"Model: {row.model or 'Unavailable'}\nBasis: {'Estimated' if row.estimated else 'Reported'}\nCost: {cost}\n\n"
            f"Uncached input: {row.uncached_input:,}\nCache read: {row.cache_read:,}\nCache write: {row.cache_write:,}\nOutput: {row.output:,}\n"
            f"Audio input (subset of input): {row.audio_input:,}\nAudio output: {row.audio_output:,}\nTranscription: {row.transcription_seconds:g} seconds"
        )

    @staticmethod
    def _messages_summary(capture: ExchangeCapture) -> str:
        messages = capture.request.get("messages_payload") or []
        marker = history_elision_marker(messages)
        if marker is not None:
            return f"Messages ({marker['original_rows']} sent; {marker['omitted_rows']} elided by capture policy)"
        return f"Messages ({len(messages) if isinstance(messages, list) else 0})"

    def _call_detail_text(self, capture: ExchangeCapture, abandoned: bool) -> str:
        response_heading = (
            "Locally synthesized fallback (not model output)"
            if capture.response and capture.response.get("synthetic_fallback")
            else "Response"
        )
        text = (
            self._exchange_call_title(capture, abandoned)
            + "\n"
            + self._call_cost_line(capture)
            + "\n\nRequest (adapter boundary) · "
            + self._messages_summary(capture)
            + "\n"
            + self._json_block(capture.request)
            + "\n\n"
            + response_heading
            + "\n"
            + self._json_block(capture.response)
            + "\n\nReported usage\n"
            + self._json_block(capture.usage_json)
            + "\n\nOmitted fields\n"
            + self._json_block(capture.omitted_keys)
        )
        if len(text.encode("utf-8")) > SIZE_THRESHOLD_BYTES:
            return "Call exceeds 1 MiB. Use Export selected call to inspect it with the existing disclosure safeguards."
        return text

    @on(Button.Pressed, "#console-inspector-show-exchanges")
    def _show_usage_exchanges(self, event: Button.Pressed) -> None:
        event.stop()
        key = self._trace_selections.get("console-inspector-usage-detail", "")
        if not key.startswith("usage:"):
            return
        item = self._usage_items[int(key[6:])]
        turn = next(
            (
                turn
                for turn in self._turns_by_index.values()
                if turn.native_message_id == item.message_key
            ),
            None,
        )
        if turn is not None:
            self.query_one(
                "#console-inspector-tabs", TabbedContent
            ).active = TAB_EXCHANGE
            self.query_one(
                "#console-inspector-exchange-detail", ConsoleInspectorDetailPane
            ).select(f"turn:{turn.index}")

    @on(Button.Pressed, "#console-inspector-export-call")
    def _export_selected_call(self, event: Button.Pressed) -> None:
        event.stop()
        if self._selected_export_key is not None:
            self._open_exchange_export(self._selected_export_key)

    @on(Button.Pressed, "#console-inspector-capture-settings")
    def _capture_settings(self, event: Button.Pressed) -> None:
        event.stop()
        self.action_capture_policy()

    def on_mount(self) -> None:
        if self._initial_tab == TAB_NEXT_SEND:
            self._request_snapshot()
        self.call_after_refresh(self._focus_initial_control)

    def _request_snapshot(self) -> None:
        if self.next_send_loading or not self._target_authority_is_current():
            return
        self.run_worker(
            self._load_snapshot,
            exclusive=True,
            group=_NEXT_SEND_WORKER_GROUP,
            name="load_snapshot",
        )

    @on(TabbedContent.TabActivated, "#console-inspector-tabs")
    def _view_entered(self, event: TabbedContent.TabActivated) -> None:
        if not self._target_authority_is_current():
            return
        if event.pane.id == TAB_NEXT_SEND and not self._snapshot_ready:
            self._request_snapshot()

    def on_unmount(self) -> None:
        self._snapshot_generation += 1
        self._disclosure_generation += 1
        self._detail_generation += 1

    def on_screen_resume(self) -> None:
        self._target_authority_is_current()

    def _target_authority_is_current(self) -> bool:
        try:
            valid = not self._target_invalid and self._target_is_current()
        except Exception:  # noqa: BLE001 - injected authority/loaders fail closed
            valid = False
        if valid:
            return True
        if not self._target_invalid:
            self._target_invalid = True
            self._snapshot_generation += 1
            self._disclosure_generation += 1
            self._snapshot_ready = False
            self._snapshot_status = (
                "Conversation unavailable — close and reopen Inspector"
            )
            self.snapshot = ConsoleContextSnapshot(
                current_messages=[], next_send_payload={}
            )
            self._invalidate_stale_captures()
            for body in self.query(TextArea):
                body.load_text("")
            for button in self.query(Button):
                if button.id != CLOSE_BUTTON_ID:
                    button.disabled = True
        return False

    def _capture_policy_text(self) -> str:
        bindings = self._capture_policy_bindings
        if bindings is None:
            return "Future exchange capture: unavailable"
        try:
            snapshot = bindings.read()
        except Exception:  # noqa: BLE001 - injected authority/loaders fail closed
            return "Future exchange capture: unavailable"
        effective_capture = getattr(snapshot, "effective_capture_enabled", None)
        if effective_capture is None:
            effective_capture = snapshot.enabled
        future = "On" if effective_capture else "Off"
        fidelity = snapshot.effective.detail.value.title()
        pii = "On" if getattr(snapshot, "pii_redaction_enabled", False) else "Off"
        viewer = self._viewer_profile.value.title()
        if snapshot.active_run_detail is not None:
            run_state = (
                f"Active run frozen at {snapshot.active_run_detail.value.title()}"
            )
        elif (
            getattr(snapshot, "next_capture_enabled", None) is not None
            or getattr(snapshot, "next_pii_redaction_enabled", None) is not None
        ):
            run_state = "Next eligible send has a Capture/PII override"
        elif snapshot.next_detail is not None:
            run_state = (
                f"Next eligible send: {snapshot.next_detail.value.title()} (armed)"
            )
        else:
            run_state = "No active run is frozen"
        return (
            f"“{snapshot.conversation_title}” · Capture {future} · "
            f"Fidelity {fidelity} · Creds filtered\n"
            f"PII {pii} · Viewer {viewer} · {run_state}"
        )

    @property
    def viewer_profile(self) -> TraceViewerProfile:
        """Return the current explicit local disclosure profile."""

        return self._viewer_profile

    async def action_viewer_profile(self) -> None:
        """Switch disclosure profile, confirming every Safe-to-Full change."""

        from tldw_chatbook.Chat.trace_export_profiles import TraceViewerProfile

        if self._viewer_profile is TraceViewerProfile.FULL:
            self._viewer_profile = TraceViewerProfile.SAFE
        else:
            confirmed = await self.app.push_screen_wait(
                ConfirmationDialog(
                    title="View Full trace content?",
                    message=(
                        "Full may reveal persisted prompts, tool arguments/results, "
                        "automatic instructions, local paths, and sensitive prose "
                        "that PII detectors missed. Credentials and frozen masks "
                        "remain blocked."
                    ),
                    confirm_label="View Full",
                    cancel_label="Keep Safe",
                )
            )
            if not confirmed:
                return
            self._viewer_profile = TraceViewerProfile.FULL
        await self._reset_view_projection()

    async def _reset_view_projection(self) -> None:
        self._clear_trace_details()
        self.query_one("#console-inspector-policy-status", Static).update(
            self._capture_policy_text()
        )
        self.query_one(
            f"#{VIEWER_PROFILE_BUTTON_ID}", Button
        ).label = f"View: {self._viewer_profile.value.title()}"

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        """Hide the contextual capture binding when no live target exists."""
        if action == "capture_policy":
            return self._capture_policy_bindings is not None
        return True

    def action_capture_policy(self) -> None:
        """Open policy controls for the immutable Inspector target."""
        if (
            self._capture_policy_bindings is None
            or not self._target_authority_is_current()
        ):
            return
        self.app.push_screen(ConsoleTracePrivacyDialog(self._capture_policy_bindings))

    def _focus_initial_control(self) -> None:
        """Focus a visible Context recovery action, Refresh, or Close.

        A snapshot may finish after the user changes views. Recheck the
        active view before moving focus so an asynchronous completion
        cannot interrupt usage or exchange inspection. Explicitly reveal
        the control even when it already had focus before a refresh.
        """
        try:
            next_send_active = (
                self.query_one("#console-inspector-tabs", TabbedContent).active
                == TAB_NEXT_SEND
            )
        except NoMatches:
            next_send_active = False
        if not next_send_active:
            return
        for selector in (
            ".console-project-instruction-recovery-action",
            "#console-inspector-next-send-refresh",
            f"#{CLOSE_BUTTON_ID}",
        ):
            controls = list(self.query(selector))
            available = [
                control
                for control in controls
                if control.can_focus and not getattr(control, "disabled", False)
            ]
            if available:
                available[0].focus()
                # ``immediate=True``: the default (``False``) DEFERS the
                # actual scroll to "after a screen refresh" -- i.e. one
                # more async round trip -- which left the target reachable
                # by `.region` (a layout-time property, already correct)
                # for a query, but not for real mouse-click hit-testing on
                # the smallest supported viewport (80x24) for at least one
                # more event-loop turn.
                # ``top=True``: without it, the minimal scroll needed to
                # bring the LAST line of a multi-row recovery panel's
                # bottom edge exactly level with this pane's own bottom
                # edge -- an exact boundary, not a comfortable margin --
                # measurably left it unclickable at 80x24 even though
                # ``.region`` reported it on-screen (a real, reproduced
                # off-by-one at the viewport edge, not a test artifact).
                # Scrolling to the TOP instead lands the target control
                # comfortably inside the viewport rather than pinned to
                # its very edge.
                available[0].scroll_visible(animate=False, immediate=True, top=True)
                return

    @on(ConsoleProjectInstructionContextPanel.RecoveryRequested)
    async def _recover_project_instructions(
        self, event: ConsoleProjectInstructionContextPanel.RecoveryRequested
    ) -> None:
        """Apply one explicit recovery decision and refresh the panel in
        place (task-18300, ported verbatim from the retired standalone
        context modal)."""
        if (
            self._project_instruction_recovery is None
            or not self._target_authority_is_current()
        ):
            return
        event.stop()
        state = await self._project_instruction_recovery(event.session_id, event.action)
        if state is None or not self._target_authority_is_current():
            return
        self._project_instruction_state = state
        self.query_one(
            "#console-context-project-instructions",
            ConsoleProjectInstructionContextPanel,
        ).sync_state(state)
        self.call_after_refresh(self._focus_initial_control)

    @staticmethod
    def _format_row(row: ConsoleCostRow) -> str:
        """Pure ``str`` render for one breakdown row (verbatim from the
        retired standalone cost modal's own ``_format_row`` -- task-8
        moved it here; task-10 retired that modal and its now-unused copy).

        task-2390: ``row.cost_usd`` already folds in any audio/
        transcription dollar contribution (see ``ConsoleCostRow``'s own
        docstring), so a realtime row's audio-token and transcription-
        duration usage is appended here as its own segment -- omitted
        entirely for a non-realtime row (all three fields 0) -- rather
        than left invisible inside that one total.
        """
        cost_text = "unpriced" if row.cost_usd is None else f"${row.cost_usd:.4f}"
        if row.estimated:
            cost_text = f"~{cost_text}"
        text = (
            f"[{row.index}] {row.role} ({row.model or 'unknown'}) -- "
            f"in:{row.uncached_input} cache_r:{row.cache_read} "
            f"cache_w:{row.cache_write} out:{row.output}"
        )
        if row.audio_input or row.audio_output:
            text += f" audio_in:{row.audio_input} audio_out:{row.audio_output}"
        if row.transcription_seconds:
            text += f" transcribe:{row.transcription_seconds:g}s"
        return f"{text} -- {cost_text}"

    @staticmethod
    def _format_totals(totals: ConsoleCostRowTotals) -> str:
        """Pure ``str`` render for the aggregate totals row (verbatim from
        the retired standalone cost modal's own ``_format_totals``)."""
        if totals.total_cost_usd is None:
            cost_text = "Cost unavailable (unpriced rows)"
        else:
            cost_text = f"${totals.total_cost_usd:.4f}"
            if totals.has_estimated_entries:
                cost_text = f"~{cost_text} (includes estimated rows)"
        return (
            f"Total -- {totals.total_tokens} tokens -- {cost_text} "
            f"({totals.row_count} rows)"
        )

    @staticmethod
    def _call_cost_line(capture: ExchangeCapture) -> str:
        """Price one captured call through the same catalog helper
        ``build_cost_rows`` uses (``PricingCatalog.cost_for_usage``).

        Returns ``"unpriced"`` when the call has no recorded usage
        (``usage_json`` is ``None``) or that usage's provider/model has no
        known rate -- never a fabricated figure.
        """
        usage = ProviderUsage.from_json(capture.usage_json)
        if usage is None:
            return "unpriced"
        breakdown = get_pricing_catalog().cost_for_usage(usage)
        if breakdown is None:
            return "unpriced"
        return f"${breakdown.total:.4f}"

    # -- Exchange tab (task-9) -----------------------------------------

    def _exchange_call_title(self, capture: ExchangeCapture, abandoned: bool) -> str:
        text = (
            f"call {capture.seq} [{capture.status}] {capture.model} -- "
            f"{self._call_cost_line(capture)} · "
            f"capture: {capture.capture_detail.value.title()}"
        )
        if abandoned:
            text += " [abandoned regeneration]"
        if capture.trace_provenance == "legacy_snapshot":
            text += " · legacy snapshot · chronology: recorded call only"
        elif capture.trace_provenance == "legacy_blob":
            text += " · legacy blob (normalization pending) · chronology: recorded call only"
        if capture.trace_uncertainty:
            text += " · uncertainty disclosed"
        return text

    @staticmethod
    def _reported_usage_line(usage: ProviderUsage) -> str:
        """The call's actual, provider-reported buckets -- deliberately NOT
        prefixed with "~"/"est." anywhere in this string (hard constraint
        4): unlike every per-piece estimate below it, these numbers are
        authoritative."""
        return (
            f"Reported usage -- in:{usage.uncached_input} "
            f"cache_r:{usage.cache_read} cache_w:{usage.cache_write} "
            f"out:{usage.output}"
        )

    @staticmethod
    def _json_block(obj: Any) -> str:
        """Same idiom as the retired standalone context modal's own
        ``_json_block``; task-10 also reuses this one @staticmethod for
        the Next Send tab's rendering rather than duplicating it."""
        return json.dumps(obj, indent=2, default=str)

    def _open_exchange_export(self, call_key: str) -> bool:
        # Keep the trajectory export family off the Chat first-paint import
        # closure. It is needed only after the user opens this disclosure
        # dialog, not while the conversation inspector module is imported.
        from tldw_chatbook.Widgets.Console.console_exchange_export_dialog import (
            ConsoleExchangeExportDialog,
        )

        """Open the governor for the exact loaded call and capture revision."""
        if not self._capture_revision_is_current():
            return False
        call = self._exchange_capture_by_call_key.get(call_key)
        if call is None:
            return False
        capture, _ = call
        expected = self._capture_revision_at_open
        source_revision = self._capture_revision_provider
        if expected is None or source_revision is None:
            expected = 0
        generation = self._disclosure_generation

        def provider() -> int:
            if (
                generation != self._disclosure_generation
                or not self._target_authority_is_current()
            ):
                raise ValueError("Inspector target or disclosure changed")
            current = source_revision() if source_revision is not None else 0
            if current is None:
                raise ValueError("Capture authority unavailable")
            return current

        self.app.push_screen(
            ConsoleExchangeExportDialog(
                capture,
                expected_capture_revision=expected,
                capture_revision_provider=provider,
            )
        )
        return True

    def _validated_export_destination(self, path: Path) -> Path | None:
        """Validate a Downloads-bound export path through the repo's
        centralized ``path_validation`` module before any write.

        The Next Send tab's ``_save_json`` action builds
        ``Path.home() / "Downloads" / filename`` itself; this
        confirms that resolved destination actually stays inside Downloads
        (Qodo PR #1883 finding 2 -- repo rule: file paths go through
        ``path_validation.py``) before ``mkdir``/``write_text`` run.

        M4 (corrected): this does NOT catch a symlinked Downloads
        directory -- ``validate_path`` resolves BOTH ``path`` and
        ``downloads_dir`` before comparing, so a Downloads symlink resolves
        transparently on both sides of the check and never trips the
        traversal guard. What it DOES catch: the destination FILE itself
        already existing as a symlink pointing outside Downloads (``path``
        resolves through it, ``downloads_dir`` does not, so
        ``relative_to`` fails), and a future change that makes the
        filename component non-literal (e.g. accepts ``..`` segments or an
        absolute override). Deliberately returns the ORIGINAL ``path`` argument on success (not
        ``validate_path``'s own return value, which is always a real
        ``pathlib.Path`` freshly re-resolved from it) so the object this
        modal's own ``Path`` name constructed keeps flowing to
        ``mkdir``/``write_text`` unchanged -- this module's tests
        legitimately swap that name for a redirecting/failing stand-in,
        and this check must stay a pure guard, not a silent path swap. On
        rejection, notifies with the failure class and the attempted path
        only -- never the raw exception body, matching this file's
        existing no-payload-in-error-surface contract -- and returns
        ``None`` so the caller does not write. ``redact_paths=True`` keeps
        this same restraint inside ``path_validation`` itself: without it,
        a rejection logs the full attempted/resolved paths at WARNING/
        ERROR (``path_validation.py``'s own log lines), which would put
        the user's home directory into the log for every rejected save. This
        modal's own log line below fingerprints the path, while its user-visible
        toast intentionally names the selected destination and excludes the raw
        exception body.
        """
        downloads_dir = Path.home() / "Downloads"
        try:
            validate_path(path, downloads_dir, redact_paths=True)
            return path
        except ValueError as exc:
            logger.error(
                "Rejected export destination path_sha256={} exception_type={}",
                content_fingerprint(str(path)),
                type(exc).__name__,
            )
            self.notify(f"Save failed ({type(exc).__name__}): {path}", severity="error")
            return None

    # -- Next Send tab (task-10, ported from the retired context modal) -

    def watch_snapshot(self) -> None:
        """Re-render the Next Send tab whenever ``snapshot`` is replaced.

        ``_load_snapshot`` swaps in a freshly-fetched
        ``ConsoleContextSnapshot`` (current messages + the exact next-send
        payload); this keeps the Current Context and Next Send Payload
        panes in sync with it without either caller having to remember to
        call ``_update_view`` itself.
        """
        self._detail_generation += 1
        self._update_view()

    def watch_raw_json(self) -> None:
        """Re-render the Next Send tab when the raw-JSON toggle flips.

        ``raw_json`` switches ``_build_next_send_widgets`` between its
        structured (collapsible-sections) rendering and a single raw
        ``TextArea`` dump of the payload; this is what makes that switch
        take effect immediately instead of only on the next unrelated
        reflow.
        """
        self._update_view()

    def watch_next_send_loading(self) -> None:
        """Show/hide the Next Send tab's spinner as a snapshot load runs.

        ``next_send_loading`` is deliberately NOT named ``loading`` (see
        the reactive's own declaration comment above) so it cannot collide
        with ``Widget``'s built-in whole-screen loading overlay; this
        watcher is what actually drives the tab-local
        ``LoadingIndicator``'s visibility from that separate flag.
        """
        try:
            loading = self.query_one(
                "#console-inspector-next-send-loading", LoadingIndicator
            )
        except NoMatches:
            return
        if self.next_send_loading:
            loading.add_class("loading")
        else:
            loading.remove_class("loading")

    def _update_view(self) -> None:
        # LY-13 (TASK-2154.23) compaction was DROPPED here (task-10 review
        # finding 1) -- see this class's BUNDLED_CSS comment for why. The
        # empty state still renders its own guidance copy below
        # (``_build_current_context_widgets``); it just no longer resizes
        # the pane's own container to match.
        if not self.is_mounted:
            return
        if self._project_instruction_state is not None:
            # task-18300, ported verbatim from the retired standalone
            # context modal: sync the panel's content-free authority/source
            # metadata every render, then -- only when the binding is
            # actually authoritative for THIS folder (``enabled`` and a
            # matching locator) -- layer in the disposable preview this
            # snapshot carries. A stale/mismatched binding must not show a
            # preview of content that would not actually be sent.
            project_panel = self.query_one(
                "#console-context-project-instructions",
                ConsoleProjectInstructionContextPanel,
            )
            project_panel.sync_state(self._project_instruction_state)
            if (
                self._project_instruction_state.enabled
                and self._project_instruction_state.locator_match == "match"
            ):
                project_panel.sync_preview(self.snapshot.project_instruction_preview)

        warning = self.query_one("#console-inspector-next-send-warning", Static)
        if self._in_progress:
            warning.update("A response is in progress; snapshot may change.")
        else:
            warning.update("")
        warning.display = self._in_progress

        header = self.query_one("#console-inspector-next-send-header", Static)
        model = self.snapshot.next_send_payload.get("model") or "Unavailable"
        estimate = (
            f"~{self._token_estimate:,}"
            if self._token_estimate is not None
            else "Unavailable"
        )
        budget, output = self._context_budget
        header_text = (
            f"Model: {model} · Prepared input: {estimate} tokens\n"
            f"Input budget: {budget if budget is not None else 'Unavailable'} · Output reservation: {output if output is not None else 'Unavailable'}\n"
            f"{self._snapshot_status}"
        )
        header.update(header_text)

        pane = self.query_one(
            "#console-inspector-context-detail", ConsoleInspectorDetailPane
        )
        previous = pane.selected_key
        pane.set_sections(context_sections(self.snapshot))
        if self._snapshot_ready:
            keys = {section.key for section in pane.sections}
            key = (
                previous
                if previous in keys
                else (
                    "preview:messages"
                    if "preview:messages" in keys
                    else "current:messages"
                )
            )
            pane.select(key, open_detail=pane.detail_open)
        else:
            pane.clear_detail(self._snapshot_status)

    def _format_next_send_text(self) -> str:
        return self._json_block(self.snapshot.next_send_payload)

    def _format_export_text(self) -> str:
        """Serialize a body-free payload for clipboard and filesystem export
        (task-18300, ported verbatim from the retired standalone context
        modal's own ``_format_export_text``).

        PRIVACY: ``_format_next_send_text`` above (the raw-JSON checkbox and
        the per-field Collapsible rendering) is this tab's own DISPOSABLE
        preview -- gone the moment this modal closes, never leaving the
        process. Copy JSON and Save to File are the opposite: they persist
        OUTSIDE this modal (the OS clipboard, a file under Downloads), so
        this is the one formatter ``_copy_json``/``_save_json`` below are
        allowed to call. Any message the ephemeral-injection path tagged
        ``EPHEMERAL_ORIGIN_KEY == "project_instructions"`` -- an
        automatically-assembled project-instruction body, never something
        the user typed -- is dropped before serializing; a
        ``project_instructions_export`` marker notes the omission so the
        exported JSON is self-describing about what's missing rather than
        silently short.
        """
        payload = copy.deepcopy(self.snapshot.next_send_payload)
        messages = payload.get("messages")
        if isinstance(messages, list):
            retained = [
                message
                for message in messages
                if not (
                    isinstance(message, dict)
                    and message.get(EPHEMERAL_ORIGIN_KEY) == "project_instructions"
                )
            ]
            if len(retained) != len(messages):
                payload["messages"] = retained
                payload["project_instructions_export"] = "automatic body omitted"
        return self._json_block(payload)

    # ``_json_block`` itself is not redefined here -- the Exchange tab
    # above already carries the identical idiom (``json.dumps(obj,
    # indent=2, default=str)``, its own docstring notes it mirrors the
    # retired standalone context modal's own ``_json_block``); this tab
    # reuses that one @staticmethod rather than duplicating it a second
    # time.

    async def _load_snapshot(self) -> None:
        if not self._target_authority_is_current():
            return
        self._snapshot_generation += 1
        generation = self._snapshot_generation
        self.next_send_loading = True
        self._snapshot_status = (
            "Refreshing preview…" if self._snapshot_ready else "Preparing preview…"
        )
        try:
            new_snapshot = await self._snapshot_factory()
            if (
                generation != self._snapshot_generation
                or not self.is_mounted
                or not self._target_authority_is_current()
            ):
                return
            # Project-instruction state refreshed BEFORE the snapshot
            # assignment below too (task-18300, same rule as the token
            # estimate's own comment right below): ``_update_view`` reads
            # ``self._project_instruction_state`` synchronously off
            # ``watch_snapshot``, so assigning the snapshot first would
            # sync the panel against the STALE state for one refresh.
            if self._project_instruction_state_factory is not None:
                self._project_instruction_state = (
                    await self._project_instruction_state_factory()
                )
            # Estimate refreshed BEFORE the snapshot assignment below
            # (task-10 review finding 6): ``self.snapshot = ...`` is a
            # reactive that triggers ``watch_snapshot`` -> ``_update_view``
            # SYNCHRONOUSLY, which reads ``self._token_estimate`` for the
            # header text -- assigning the snapshot first would render
            # with the PRIOR estimate, one refresh stale (a real bug in
            # the retired standalone context modal this was ported from).
            # task-25836: prefer the payload-based estimate (the whole
            # next-send request -- system row, messages incl. the draft
            # turn, tool schemas, staged evidence) over the draft-only
            # factory; fall back to the factory when the payload yields
            # nothing estimable (e.g. an assembly-error payload).
            if self._payload_estimate is not None:
                payload_tokens = self._payload_estimate(new_snapshot)
                if payload_tokens is not None:
                    self._token_estimate = payload_tokens
                elif self._estimate_factory is not None:
                    self._token_estimate = self._estimate_factory()
                else:
                    self._token_estimate = None
            elif self._estimate_factory is not None:
                self._token_estimate = self._estimate_factory()
            if (
                generation != self._snapshot_generation
                or not self._target_authority_is_current()
            ):
                return
            self._context_budget = (
                self._context_budget_provider()
                if self._context_budget_provider is not None
                else (None, None)
            )
            self._snapshot_ready = True
            self._snapshot_status = (
                "Prepared now · refresh after changing the draft or settings"
            )
            if self.snapshot == new_snapshot:
                self._update_view()
            else:
                self.snapshot = new_snapshot
            self.call_after_refresh(self._focus_initial_control)
        except Exception:  # noqa: BLE001 - injected authority/loaders fail closed
            if self._target_authority_is_current():
                self._snapshot_status = (
                    "Refresh failed · showing stale preview"
                    if self._snapshot_ready
                    else "Preview unavailable · Refresh to retry"
                )
                self._update_view()
                self.notify(
                    "Could not prepare context. Refresh to retry.", severity="error"
                )
        finally:
            self.next_send_loading = False

    def on_worker_state_changed(self, event: Worker.StateChanged) -> None:
        """Surface a Next Send snapshot-load failure and clear its spinner.

        Filtered to THIS pane's own worker group (task-10 review finding
        2a) -- this screen also runs the Costs tab's ``_load_turn_
        captures`` and the Exchange tab's ``_load_exchange_turn`` workers
        (both left in Textual's "default" group), and an unfiltered
        handler here would toast this tab's "Failed to refresh context."
        message -- and clear THIS tab's spinner -- for a failure that has
        nothing to do with the Next Send tab.
        """
        if event.worker.group != _NEXT_SEND_WORKER_GROUP:
            return
        if event.state == WorkerState.ERROR:
            self.next_send_loading = False
            self.notify("Failed to refresh context.", severity="error")

    @on(Checkbox.Changed, "#console-inspector-next-send-raw")
    def _toggle_raw(self, event: Checkbox.Changed) -> None:
        event.stop()
        self.raw_json = event.value

    @on(Button.Pressed, "#console-inspector-next-send-refresh")
    def _refresh_next_send(self, event: Button.Pressed) -> None:
        event.stop()
        if not self._in_progress:
            self._request_snapshot()

    @on(Button.Pressed, "#console-inspector-next-send-copy")
    def _copy_json(self, event: Button.Pressed) -> None:
        event.stop()
        if not self._target_authority_is_current() or not self._snapshot_ready:
            return
        # ``_format_export_text``, NOT ``_format_next_send_text`` (task-18300):
        # this leaves the modal for the OS clipboard, so any automatically
        # injected project-instruction message body must be scrubbed first --
        # see that method's own docstring for the privacy contract.
        text = self._format_export_text()
        try:
            import pyperclip

            pyperclip.copy(text)
            self.notify("JSON copied to clipboard.")
        except Exception as exc:  # noqa: BLE001 - disclose only sanitized failure categories
            # No exception text: ``text`` in this frame's locals is the
            # export-scrubbed next-send payload (system prompt, messages,
            # staged sources), and loguru's diagnose formatter would
            # otherwise annotate the failing source line's names (including
            # ``text``) with their values. type(exc).__name__ is enough to
            # diagnose a clipboard failure without echoing payload content
            # (hard constraint 2/3 -- the retired standalone context
            # modal's own ``_copy_json`` interpolated ``exc`` itself into
            # the log line; this is the same fix already applied to this
            # retired raw Exchange disclosure path).
            logger.warning(
                f"Failed to copy context JSON to clipboard: {type(exc).__name__}"
            )
            self.notify("Copy failed: pyperclip unavailable.", severity="warning")

    @on(Button.Pressed, "#console-inspector-next-send-save")
    def _save_json(self, event: Button.Pressed) -> None:
        event.stop()
        # M5: re-check here as defense in depth on a privacy contract, same
        # rationale as the retired raw Exchange save path -- this button's
        # own ``disabled=`` state
        # (set from ``self._save_blocked_reason`` in ``compose``) was
        # previously the ONLY enforcement of the ephemeral save-block for
        # THIS tab; a direct call bypassing the button (e.g. a future
        # caller) would still write to disk.
        if (
            self._save_blocked_reason is not None
            or not self._target_authority_is_current()
            or not self._snapshot_ready
        ):
            return
        # ``_format_export_text``, NOT ``_format_next_send_text`` (task-18300)
        # -- same privacy contract as ``_copy_json`` above, this text lands
        # on disk.
        text = self._format_export_text()
        filename = f"chatbook_context_{datetime.now().astimezone().strftime('%Y%m%d_%H%M%S')}.json"
        path = Path.home() / "Downloads" / filename
        validated_path = self._validated_export_destination(path)
        if validated_path is None:
            return
        path = validated_path
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
            self.notify(f"Saved to {path}")
        except OSError as exc:
            # No exception text, no traceback -- same rationale as this
            # retired Exchange save path: ``text`` in this frame is
            # the export-scrubbed next-send payload, and an OSError's own
            # str() can also embed the offending path. The log keeps only the
            # failure class and a path fingerprint. The user-facing toast
            # intentionally names the selected destination, but excludes the
            # raw exception body that the retired standalone context modal's
            # own ``_save_json`` put in ``notify()`` (hard constraint 3).
            logger.error(
                "Failed to save context snapshot path_sha256={} exception_type={}",
                content_fingerprint(str(path)),
                type(exc).__name__,
            )
            self.notify(f"Save failed ({type(exc).__name__}): {path}", severity="error")
        except Exception as exc:  # noqa: BLE001 - disclose only sanitized failure categories
            logger.error(
                "Unexpected error saving context snapshot path_sha256={} "
                "exception_type={}",
                content_fingerprint(str(path)),
                type(exc).__name__,
            )
            self.notify(f"Save failed ({type(exc).__name__}): {path}", severity="error")

    def action_refresh(self) -> None:
        """ "r" binding: refresh the ACTIVE tab, when it has a live-reload
        entry point.

        Only the Next Send tab does (``_load_snapshot``, ported from the
        retired standalone context modal's own "r" binding) -- the Costs
        tab's rows are precomputed by the caller (no live-recompute entry
        point exists here, matching the retired standalone cost modal's
        own shape) and the Exchange tab's captures are fetched once per
        turn on first expand, so there is nothing to refresh on either of
        those tabs.
        """
        try:
            tabs = self.query_one("#console-inspector-tabs", TabbedContent)
        except NoMatches:
            return
        if tabs.active == TAB_NEXT_SEND and not self._in_progress:
            self._request_snapshot()

    async def action_dismiss(self) -> None:
        """Defensive fallback for the built-in "dismiss" action name."""
        await self.request_safe_cancel(source="visible")

    @on(Button.Pressed, f"#{CLOSE_BUTTON_ID}")
    async def _close(self, event: Button.Pressed) -> None:
        event.stop()
        await self.request_safe_cancel(source="visible")
