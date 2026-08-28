"""Console Conversation Inspector modal (task-8: scaffold + Costs tab;
task-9: the Exchange tab; task-10: the Next Send tab, and retirement of
the two modals it replaces; task-18300: the Next Send tab's project-
instructions panel).

task-18300: ``dev`` added a metadata-only project-instructions section
(status, sources, warnings, one-click recovery) to the retired standalone
context modal while this branch had already replaced that modal with this
one. Ported here: the four ``project_instruction_*`` constructor kwargs
(all default ``None`` -- the cost-chip entry point never passes them),
mounting ``ConsoleProjectInstructionContextPanel`` on the Next Send pane
when a state is given, ``_focus_initial_control``/``AUTO_FOCUS = None``,
the ``RecoveryRequested`` handler, and -- the load-bearing part --
``_format_export_text``: Copy JSON and Save to File must never carry an
automatically-injected project-instruction message body (tagged
``EPHEMERAL_ORIGIN_KEY == "project_instructions"``); only the disposable
raw-JSON/Collapsible preview (``_format_next_send_text``) may show it.

Replaced the two standalone modals that used to live in this directory
(one opened from the cost chip, the other via Ctrl+Shift+P / the command
palette) with ONE modal that gains a tab per surface: Costs (task-8),
Exchange (per-call request/response detail with status badges, task-9),
Next Send (the assembled next-send payload, task-10). Both entry points in
``chat_screen.py`` push the SAME instance, differing only by which tab
starts active (``initial_tab``).

``_format_row``/``_format_totals`` (Costs tab) and the Next Send tab's
worker/reactive/render methods (``_load_snapshot``, ``watch_snapshot``,
``_build_current_context_widgets``, ``_build_next_send_widgets``, ...) are
ported VERBATIM from the two retired modals -- still pure formatters and
"already computed, just render it" rendering over caller-supplied data,
same house pattern the rest of this widget already follows.

Loader contract (task-8, carried forward for task-9): ``exchanges_loader``
takes ONE turn's ``InspectorTurn.native_message_id`` and returns
``list[tuple[ExchangeCapture, bool]]`` -- ``(capture, abandoned)`` pairs,
not bare captures, so task-9's Exchange tab can render the "abandoned"
badge without a second read of the same rows. Two notes for callers:

    - A capture's ``created_at``/``seq`` is the only reliable ordering --
      both ``get_message_exchanges`` (SQL ``ORDER BY run_tag, seq``) and
      the in-memory store's own merge (``sorted(..., key=lambda c:
      (c.run_tag, c.seq))``) order by ``run_tag`` STRING, which is not
      chronological across multiple runs on one message. This widget
      re-sorts every loader result by ``(created_at, seq)`` before
      rendering rather than trusting incoming order.
    - For an in-memory (not-yet-persisted / ephemeral-session) capture,
      whether its ``abandoned`` flag is accurate depends entirely on the
      caller's ``exchanges_loader`` -- this widget only ever renders
      whatever the loader hands it. ``chat_screen.py``'s own loader
      resolves it through ``ConsoleChatStore.abandoned_exchange_run_tags``
      (task-9); a caller with no equivalent bookkeeping should pass
      ``False`` for every native capture, same as before.

Exchange tab (task-9) lazy-mount chain: turn -> call -> section -> (for the
"Messages" section only) one more level, per-message. Each level's
Collapsible header is cheap and built eagerly the moment its PARENT
expands; only the deepest, potentially-large payload (a ``TextArea``) waits
for that specific node's own first expand. A 50-call agent turn therefore
mounts 50 call headers on turn-expand, but zero ``TextArea`` widgets until
a caller actually drills into one.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger
from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.content import Content
from textual.css.query import NoMatches
from textual.reactive import reactive
from textual.screen import ModalScreen
from textual.widget import Widget
from textual.widgets import (
    Button,
    Checkbox,
    Collapsible,
    Label,
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
from tldw_chatbook.Chat.console_exchange_capture import ExchangeCapture
from tldw_chatbook.Chat.console_project_instructions import EPHEMERAL_ORIGIN_KEY
from tldw_chatbook.Chat.provider_usage import ProviderUsage
from tldw_chatbook.LLM_Calls.pricing_catalog import get_pricing_catalog
from tldw_chatbook.Utils.path_validation import validate_path
from tldw_chatbook.Utils.token_counter import estimate_tokens
from tldw_chatbook.Widgets.modal_dismissal import SafeModalDismissMixin
from tldw_chatbook.Widgets.Console.console_project_instructions import (
    ConsoleProjectInstructionContextPanel,
)
from tldw_chatbook.Widgets.Console.console_capture_policy_dialog import (
    CapturePolicyBindings,
    ConsoleCapturePolicyDialog,
)

MODAL_ID = "console-inspector-modal"
CLOSE_BUTTON_ID = "console-inspector-close"
TAB_COSTS = "inspector-costs"
TAB_EXCHANGE = "inspector-exchange"
TAB_NEXT_SEND = "inspector-next-send"
_COST_ROW_ID_PREFIX = "console-inspector-cost-row-"

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

# Shared between the Costs tab's per-turn drill-in and the Exchange tab's
# turn-level load (task-9) -- identical wording for identical situations,
# defined once so the two never drift apart.
_NO_CAPTURES_MESSAGE = (
    "No capture recorded for this turn (recorded before capture existed, "
    "capture disabled, or capture failed)."
)
_LOAD_FAILURE_MESSAGE = (
    "Could not load captures for this turn -- expand again to retry."
)
# Review finding I2: the spec requires this caveat stated IN the UI (twice),
# not just the User Guide -- a user in this tab has no other in-surface
# signal that capture happens at the provider-adapter boundary, not the raw
# HTTP layer (so provider-internal framing and injected `cache_control`
# markers never appear here), and that llama.cpp is the one exception (its
# capture IS the literal wire payload).
_EXCHANGE_ADAPTER_BOUNDARY_CAVEAT = (
    "Captured where Console hands the request to the provider adapter, not "
    "at the raw HTTP layer -- provider-internal framing and injected "
    "prompt-cache markers are not visible here (llama.cpp is the exception: "
    "its capture is the literal wire payload)."
)

# Exchange tab (task-9) DOM id prefixes -- the four-level lazy chain (turn
# -> call -> section -> message) is dispatched by a single
# ``@on(Collapsible.Toggled)`` handler that switches on which prefix an
# expanding Collapsible's id starts with (mirrors the Costs tab's own
# ``_COST_ROW_ID_PREFIX`` filtering, just one handler per level instead of
# one flat namespace).
_EXCHANGE_TURN_ID_PREFIX = "console-inspector-exchange-turn-"
_EXCHANGE_CALL_ID_PREFIX = "console-inspector-exchange-call-"
_EXCHANGE_SECTION_ID_PREFIX = "console-inspector-exchange-section-"
_EXCHANGE_MESSAGE_ID_PREFIX = "console-inspector-exchange-message-"
_EXCHANGE_EXPORT_BUTTON_PREFIX = "console-inspector-exchange-export-"

# Section keys, in render order. "toolcalls" (the response's own tool
# calls) is built separately -- and OMITTED entirely when empty -- rather
# than living in this tuple, since every other section always renders.
_SECTION_SYSTEM = "system"
_SECTION_MESSAGES = "messages"
_SECTION_TOOLS = "tools"
_SECTION_RESPONSE = "response"
_SECTION_TOOL_CALLS = "toolcalls"
_SECTION_SAMPLING = "sampling"

# Request keys already surfaced elsewhere in the call's UI (system prompt,
# messages, tools sections, and the call title's model) -- excluded from
# the "Sampling & routing" section so it shows only the remaining scalar
# kwargs (temperature, max_tokens, seed, ...) without repeating the bulk
# content.
_SAMPLING_EXCLUDED_REQUEST_KEYS = frozenset(
    {"system_message", "messages_payload", "tools", "model"}
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
    BINDINGS = [
        ("escape", "request_safe_cancel", "Close"),
        ("r", "refresh", "Refresh"),
        ("c", "capture_policy", "Capture"),
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
        rows: Sequence[ConsoleCostRow],
        totals: ConsoleCostRowTotals,
        turns: Sequence[InspectorTurn],
        exchanges_loader: ExchangesLoader,
        snapshot_factory: SnapshotFactory,
        token_estimate: int | None = None,
        estimate_factory: Callable[[], int | None] | None = None,
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
        initial_tab: str = TAB_COSTS,
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
        super().__init__()
        self._rows = list(rows)
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
        try:
            self._capture_revision_at_open = (
                capture_revision_provider()
                if capture_revision_provider is not None
                else None
            )
        except Exception:
            self._capture_revision_at_open = None
        self._initial_tab = initial_tab or TAB_COSTS
        self._loaded_row_indices: set[int] = set()

        # Exchange tab (task-9) lazy-mount bookkeeping. Each level's
        # dedup set is independent of the Costs tab's ``_loaded_row_
        # indices`` -- expanding a turn in one tab must not be conflated
        # with the other, even though both ultimately call the same
        # ``exchanges_loader``.
        self._loaded_exchange_turn_indices: set[int] = set()
        self._loaded_exchange_call_keys: set[str] = set()
        self._loaded_exchange_section_ids: set[str] = set()
        self._loaded_exchange_message_ids: set[str] = set()
        # "{turn_index}-{call_ordinal}" -> that call's capture, populated
        # once its turn's async load resolves; every deeper level (section
        # bodies, per-message bodies, Copy/Save) reads from here rather
        # than re-fetching.
        self._exchange_capture_by_call_key: dict[str, ExchangeCapture] = {}
        self._exchange_message_by_id: dict[str, Any] = {}
        self._save_blocked_reason = blocked_reason("save-context", ephemeral=ephemeral)

    def _capture_revision_is_current(self) -> bool:
        """Return whether cached captures still belong to the open revision."""
        provider = self._capture_revision_provider
        if provider is None:
            return True
        try:
            current = provider()
        except Exception:
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

    def _invalidate_stale_captures(self) -> None:
        """Drop decoded and mounted capture bodies after revision change."""
        self._loaded_row_indices.clear()
        self._loaded_exchange_turn_indices.clear()
        self._loaded_exchange_call_keys.clear()
        self._loaded_exchange_section_ids.clear()
        self._loaded_exchange_message_ids.clear()
        self._exchange_capture_by_call_key.clear()
        self._exchange_message_by_id.clear()
        try:
            exchange_turns = self.query_one("#console-inspector-exchange-turns")
        except NoMatches:
            exchange_turns = None
        if exchange_turns is not None:
            for text_area in exchange_turns.query(TextArea):
                text_area.load_text("")
        try:
            self.query_one("#console-inspector-capture-status", Static).update(
                "Stored captures changed · Refresh required"
            )
        except NoMatches:
            pass

    async def _invalidate_stale_exchange_mounts(self) -> None:
        """Remove call nodes that may have mounted across a revision change."""
        self._invalidate_stale_captures()
        for call in list(self.query(".console-inspector-exchange-call")):
            await call.remove()

    def compose(self) -> ComposeResult:
        """Build the header, tabbed body, and shared Close action."""
        with Vertical(id=MODAL_ID):
            yield Static(
                "Conversation Inspector", id="console-inspector-header", markup=False
            )
            yield Static(
                self._capture_policy_text(),
                id="console-inspector-policy-status",
                markup=False,
            )
            with TabbedContent(id="console-inspector-tabs", initial=self._initial_tab):
                with TabPane("Costs", id=TAB_COSTS):
                    with VerticalScroll(id="console-inspector-costs-rows"):
                        yield from self._build_costs_widgets()
                    yield Static(
                        self._format_totals(self._totals),
                        id="console-inspector-costs-totals",
                        markup=False,
                    )
                with TabPane("Exchange", id=TAB_EXCHANGE):
                    yield Static(
                        "", id="console-inspector-capture-status", markup=False
                    )
                    yield Static(
                        _EXCHANGE_ADAPTER_BOUNDARY_CAVEAT,
                        id="console-inspector-exchange-caveat",
                        markup=False,
                    )
                    with VerticalScroll(id="console-inspector-exchange-turns"):
                        yield from self._build_exchange_turn_widgets()
                with TabPane("Next Send", id=TAB_NEXT_SEND):
                    with VerticalScroll(id="console-inspector-next-send-pane"):
                        yield Static(
                            "Chat Context",
                            id="console-inspector-next-send-header",
                            markup=False,
                        )
                        yield Static(
                            "",
                            id="console-inspector-next-send-warning",
                            markup=False,
                        )
                        if self._project_instruction_state is not None:
                            # task-18300: ported from the retired standalone
                            # context modal -- id kept as
                            # ``console-context-project-instructions``
                            # (unchanged from that modal) rather than
                            # adopting this pane's own ``console-inspector-
                            # next-send-*`` prefix, since it is not part of
                            # this pane's own frame -- it is the SAME
                            # metadata-only panel widget the retired modal
                            # mounted, unmodified.
                            yield ConsoleProjectInstructionContextPanel(
                                self._project_instruction_state,
                                session_id=self._project_instruction_session_id,
                                id="console-context-project-instructions",
                            )
                        yield LoadingIndicator(id="console-inspector-next-send-loading")

                        with TabbedContent(id="console-inspector-next-send-tabs"):
                            with TabPane(
                                "Current", id="console-inspector-next-send-current"
                            ):
                                yield Vertical(
                                    id="console-inspector-next-send-current-body"
                                )
                            with TabPane(
                                "Next Send", id="console-inspector-next-send-payload"
                            ):
                                yield Vertical(
                                    id="console-inspector-next-send-payload-body"
                                )

                        with Horizontal(id="console-inspector-next-send-actions"):
                            yield Checkbox(
                                "Raw JSON", id="console-inspector-next-send-raw"
                            )
                            yield Button(
                                "Refresh",
                                id="console-inspector-next-send-refresh",
                                disabled=self._in_progress,
                            )
                            yield Button(
                                "Copy JSON", id="console-inspector-next-send-copy"
                            )
                            next_send_save_button = Button(
                                "Save to File",
                                id="console-inspector-next-send-save",
                                disabled=self._save_blocked_reason is not None,
                            )
                            if self._save_blocked_reason is not None:
                                next_send_save_button.tooltip = self._save_blocked_reason
                            yield next_send_save_button
            with Horizontal(id="console-inspector-actions"):
                yield Button("Close", id=CLOSE_BUTTON_ID, variant="primary")

    def on_mount(self) -> None:
        """Kick off the Next Send tab's first snapshot load (ported from
        the retired standalone context modal's own ``on_mount``).

        Runs regardless of ``initial_tab`` -- ``TabbedContent`` mounts all
        three panes' widgets up front (the Costs/Exchange tabs already rely
        on this: their own lazy-loading is implemented on top of it, via
        empty Collapsible bodies populated on first expand), so the Next
        Send pane's widgets exist for ``_load_snapshot`` to query/update
        the moment this fires, whether or not the user ever switches to
        that tab.
        """
        self.run_worker(
            self._load_snapshot,
            exclusive=True,
            group=_NEXT_SEND_WORKER_GROUP,
            name="load_snapshot",
        )
        self.call_after_refresh(self._focus_initial_control)

    def _capture_policy_text(self) -> str:
        bindings = self._capture_policy_bindings
        if bindings is None:
            return "Future exchange capture: unavailable"
        try:
            snapshot = bindings.read()
        except Exception:
            return "Future exchange capture: unavailable"
        future = (
            "Off" if not snapshot.enabled else snapshot.effective.detail.value.title()
        )
        if snapshot.active_run_detail is not None:
            run_state = (
                f"Active run frozen at {snapshot.active_run_detail.value.title()}"
            )
        elif snapshot.next_detail is not None:
            run_state = (
                "Next eligible send: "
                f"{snapshot.next_detail.value.title()} (armed)"
            )
        else:
            run_state = "No active run is frozen"
        return (
            f"“{snapshot.conversation_title}” · Future capture: {future} · c Change…\n"
            f"{run_state}"
        )

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        """Hide the contextual capture binding when no live target exists."""
        if action == "capture_policy":
            return self._capture_policy_bindings is not None
        return True

    def action_capture_policy(self) -> None:
        """Open policy controls for the immutable Inspector target."""
        if self._capture_policy_bindings is None:
            return
        self.app.push_screen(ConsoleCapturePolicyDialog(self._capture_policy_bindings))

    def _focus_initial_control(self) -> None:
        """Focus the most relevant available Next Send action (task-18300,
        ported from the retired standalone context modal's own
        ``_focus_initial_control``, adapted to this widget's own button ids;
        I1 review finding narrowed it to the Next Send tab only -- see below).

        Tried in order: a project-instruction recovery action (the panel is
        only mounted -- and only ever offers one -- when
        ``project_instruction_state`` was passed at construction, which the
        cost-chip entry point never does), then the Next Send tab's own
        Refresh button, then the modal's shared Close button. The first
        available (focusable, not disabled) control wins.

        I1: this method is scheduled from ``on_mount`` AND from
        ``_load_snapshot``'s tail -- and the latter fires on EVERY tab,
        because the Next Send snapshot prefetch itself starts
        unconditionally in ``on_mount`` regardless of ``initial_tab``. Doing
        nothing at all when the Next Send tab is not the ACTIVE one (checked
        fresh, at call time, not at schedule time) is therefore load-bearing
        in two ways, not just one:

          - Opening the cost-chip entry point (``initial_tab=TAB_COSTS``)
            used to still fall through to focusing Close -- the one
            selector that was NOT gated on ``next_send_active`` -- making
            Enter dismiss the modal and breaking immediate arrow-key tab
            switching on open.
          - A real ``snapshot_factory`` is slow (resolves the provider,
            reads ``AGENTS.md`` off-thread): a user already drilled into a
            Costs-tab row could have focus silently yanked to Close the
            moment that background load completed, even though they never
            touched the Next Send tab at all.

        Close now shares the same gate as the recovery action and Refresh
        button -- all three live only inside ``#console-inspector-next-send-
        pane`` semantics (Close is DOM-outside that pane, in
        ``#console-inspector-actions``, but is meaningless to (re)focus
        unless a Next Send interaction is what triggered this call). Off the
        Next Send tab, this method now does nothing: Textual's own
        App-level ``AUTO_FOCUS="*"`` fallback (see ``AUTO_FOCUS = None``
        above) still lands first-DOM-order focus on initial mount, and
        nothing here re-steals it later.

        The remaining scroll-visibility rationale is unchanged from the
        ported original: ``#console-inspector-next-send-pane`` is a
        ``VerticalScroll`` that can be shorter than its own content at the
        smallest supported viewport, so ``.focus()`` alone does not
        reliably scroll a widget into view -- and a SECOND focus call on an
        already-focused control (this method runs again after every
        snapshot reload and after every recovery decision) is a no-op that
        skips it entirely. ``scroll_visible()`` is called explicitly so the
        newly-focused control is reachable regardless of when this runs.
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
        if self._project_instruction_recovery is None:
            return
        event.stop()
        state = await self._project_instruction_recovery(
            event.session_id, event.action
        )
        if state is None:
            return
        self._project_instruction_state = state
        self.query_one(
            "#console-context-project-instructions",
            ConsoleProjectInstructionContextPanel,
        ).sync_state(state)
        self.call_after_refresh(self._focus_initial_control)

    def _build_costs_widgets(self) -> list[Widget]:
        """Costs-tab row widgets: one lazily-drillable Collapsible per row.

        Each Collapsible starts with NO children -- its body is populated
        only on first expand (see ``_on_row_toggled``/``_load_turn_
        captures``), so a long transcript never mounts per-call detail
        widgets up front.
        """
        if not self._rows:
            return [Static("No priced or estimated messages yet.", markup=False)]
        return [
            Collapsible(
                # Unlike Static, Collapsible's title IS markup-parsed
                # (CollapsibleTitle.__init__ -> Content.from_text(label),
                # whose `markup` default is True) -- a model id containing
                # "[test]" would render mangled and one containing "[/]"
                # raises MarkupError inside compose(), taking the whole
                # modal down with it. Content.from_text(..., markup=False)
                # is the literal-text escape hatch; it accepts a Content
                # unmodified too, so this is safe even if a future edit
                # hands it one.
                title=Content.from_text(self._format_row(row), markup=False),
                collapsed=True,
                id=f"{_COST_ROW_ID_PREFIX}{row.index}",
                classes="console-inspector-cost-row",
            )
            for row in self._rows
        ]

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
            cost_text = "unpriced"
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

    @on(Collapsible.Toggled)
    def _on_row_toggled(self, event: Collapsible.Toggled) -> None:
        if not self._capture_revision_is_current():
            return
        collapsible = event.collapsible
        collapsible_id = collapsible.id or ""
        if not collapsible_id.startswith(_COST_ROW_ID_PREFIX):
            return
        if collapsible.collapsed:
            return
        try:
            row_index = int(collapsible_id[len(_COST_ROW_ID_PREFIX) :])
        except ValueError:
            return
        if row_index in self._loaded_row_indices:
            return
        turn = self._turns_by_index.get(row_index)
        if turn is None:
            return
        self._loaded_row_indices.add(row_index)
        self.run_worker(
            self._load_turn_captures(collapsible, turn), exclusive=False
        )

    async def _load_turn_captures(
        self, collapsible: Collapsible, turn: InspectorTurn
    ) -> None:
        """Fetch one turn's captures and mount them into its Collapsible.

        Re-sorts by ``(created_at, seq)`` -- never trusts the loader's own
        order (see the module docstring's ordering note).

        A loader failure (e.g. ``get_message_exchanges`` raising
        ``CharactersRAGDBError``) is NOT folded into the "no captures"
        empty state -- that would permanently misreport a transient DB
        error as "this turn was never captured", with no way to retry
        short of reopening the whole modal (``_loaded_row_indices`` gates
        re-fetching on re-expand). Instead it renders a distinct message
        and un-marks the row as loaded so collapsing/re-expanding tries
        again.
        """
        if not self._capture_revision_is_current():
            self._loaded_row_indices.discard(turn.index)
            return
        load_failed = False
        try:
            pairs = await self._exchanges_loader(turn.native_message_id)
        except Exception as exc:
            # No traceback: a failure inside ``_exchanges_loader`` can leave
            # a DECODED ``ExchangeCapture`` (full request/response payload)
            # sitting in one of that call's own frames -- e.g. mid-loop in
            # the DB-fallback path's row decoder -- and loguru's diagnose
            # formatter would annotate the failing source line's names with
            # their values across the WHOLE frame chain, not just this one.
            # type(exc).__name__ plus the turn's own identifiers is enough
            # to diagnose and retry (see the "expand again to retry" UX
            # below); capture content is not needed for that.
            logger.error(
                f"console_inspector: exchanges_loader failed for turn "
                f"{turn.index} ({turn.native_message_id}): {type(exc).__name__}"
            )
            pairs = []
            load_failed = True

        if not self._capture_revision_is_current():
            self._loaded_row_indices.discard(turn.index)
            return

        # Both mount-state checks precede the query/mount below -- querying
        # or mounting into a Collapsible (or its Contents) that has already
        # left the DOM (e.g. this worker outlived a modal dismiss) would be
        # at best wasted work and at worst an error.
        if not collapsible.is_mounted:
            return
        try:
            contents = collapsible.query_one(Collapsible.Contents)
        except NoMatches:
            return
        if not contents.is_mounted:
            return

        if load_failed:
            self._loaded_row_indices.discard(turn.index)
            await contents.mount(Static(_LOAD_FAILURE_MESSAGE, markup=False))
            return

        if not pairs:
            await contents.mount(Static(_NO_CAPTURES_MESSAGE, markup=False))
            return

        ordered = sorted(pairs, key=lambda pair: (pair[0].created_at, pair[0].seq))
        for capture, _abandoned in ordered:
            await contents.mount(
                Static(
                    f"call {capture.seq} [{capture.status}] {capture.model} "
                    f"-- {self._call_cost_line(capture)}",
                    markup=False,
                )
            )

    # -- Exchange tab (task-9) -----------------------------------------

    def _build_exchange_turn_widgets(self) -> list[Widget]:
        """Exchange-tab turn widgets: one lazily-drillable Collapsible per
        turn, transcript-ordered. Mirrors ``_build_costs_widgets``'s
        empty-body-until-first-expand shape, but for the Exchange tab's own
        set of DOM ids (kept fully separate from the Costs tab's rows so
        the two tabs never share loaded/mounted state)."""
        turns = sorted(self._turns_by_index.values(), key=lambda turn: turn.index)
        if not turns:
            return [Static("No turns to inspect yet.", markup=False)]
        return [
            Collapsible(
                title=Content.from_text(
                    self._exchange_turn_title(turn, call_count=None), markup=False
                ),
                collapsed=True,
                id=f"{_EXCHANGE_TURN_ID_PREFIX}{turn.index}",
                classes="console-inspector-exchange-turn",
            )
            for turn in turns
        ]

    @staticmethod
    def _exchange_turn_title(turn: InspectorTurn, call_count: int | None) -> str:
        text = f"[{turn.index}] {turn.role}"
        if call_count is not None:
            noun = "call" if call_count == 1 else "calls"
            text += f" -- {call_count} {noun}"
        return text

    def _exchange_call_title(self, capture: ExchangeCapture, abandoned: bool) -> str:
        text = (
            f"call {capture.seq} [{capture.status}] {capture.model} -- "
            f"{self._call_cost_line(capture)} · "
            f"capture: {capture.capture_detail.value.title()}"
        )
        if abandoned:
            text += " [abandoned regeneration]"
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
    def _response_text(capture: ExchangeCapture) -> str:
        content = capture.response.get("content") if capture.response else None
        return "" if content is None else str(content)

    @staticmethod
    def _json_block(obj: Any) -> str:
        """Same idiom as the retired standalone context modal's own
        ``_json_block``; task-10 also reuses this one @staticmethod for
        the Next Send tab's rendering rather than duplicating it."""
        return json.dumps(obj, indent=2, default=str)

    @staticmethod
    def _exchange_section_id(call_key: str, section: str) -> str:
        return f"{_EXCHANGE_SECTION_ID_PREFIX}{call_key}-{section}"

    def _build_system_prompt_section(
        self, capture: ExchangeCapture, call_key: str
    ) -> Collapsible:
        text = str(capture.request.get("system_message") or "")
        est = estimate_tokens(text, "", "")
        title = f"System prompt (~{est} tokens est.)"
        return Collapsible(
            title=Content.from_text(title, markup=False),
            collapsed=True,
            id=self._exchange_section_id(call_key, _SECTION_SYSTEM),
            classes="console-inspector-exchange-section",
        )

    def _build_messages_section(
        self, capture: ExchangeCapture, call_key: str
    ) -> Collapsible:
        messages = capture.request.get("messages_payload")
        count = len(messages) if isinstance(messages, list) else 0
        title = f"Messages ({count})"
        return Collapsible(
            title=Content.from_text(title, markup=False),
            collapsed=True,
            id=self._exchange_section_id(call_key, _SECTION_MESSAGES),
            classes="console-inspector-exchange-section",
        )

    def _build_tools_section(
        self, capture: ExchangeCapture, call_key: str
    ) -> Collapsible:
        tools = capture.request.get("tools")
        count = len(tools) if isinstance(tools, list) else 0
        title = f"Tools ({count})"
        return Collapsible(
            title=Content.from_text(title, markup=False),
            collapsed=True,
            id=self._exchange_section_id(call_key, _SECTION_TOOLS),
            classes="console-inspector-exchange-section",
        )

    def _build_response_section(
        self, capture: ExchangeCapture, call_key: str
    ) -> Collapsible:
        text = self._response_text(capture)
        # Review finding M3: `synthetic_fallback` marks a response the
        # gateway generated locally (NO_PROVIDER_CONTENT_COPY /
        # UNSUPPORTED_PROVIDER_RESPONSE_COPY) because the provider returned
        # nothing usable -- the inspector must never present that UI copy
        # as if it were the model's own answer.
        synthetic = bool(
            capture.response.get("synthetic_fallback") if capture.response else False
        )
        if synthetic:
            title = (
                "Response (locally synthesized fallback copy -- the "
                "provider returned no content)"
            )
        else:
            est = estimate_tokens(text, "", "")
            title = f"Response (~{est} tokens est."
            usage = ProviderUsage.from_json(capture.usage_json)
            if usage is not None:
                title += f" / reported out:{usage.output}"
            title += ")"
        return Collapsible(
            title=Content.from_text(title, markup=False),
            collapsed=True,
            id=self._exchange_section_id(call_key, _SECTION_RESPONSE),
            classes="console-inspector-exchange-section",
        )

    def _build_tool_calls_section(
        self, capture: ExchangeCapture, call_key: str
    ) -> Collapsible | None:
        tool_calls = capture.response.get("tool_calls") if capture.response else None
        if not isinstance(tool_calls, list) or not tool_calls:
            return None
        title = f"Tool calls ({len(tool_calls)})"
        return Collapsible(
            title=Content.from_text(title, markup=False),
            collapsed=True,
            id=self._exchange_section_id(call_key, _SECTION_TOOL_CALLS),
            classes="console-inspector-exchange-section",
        )

    def _build_sampling_section(
        self, capture: ExchangeCapture, call_key: str
    ) -> Collapsible:
        return Collapsible(
            title=Content.from_text("Sampling & routing", markup=False),
            collapsed=True,
            id=self._exchange_section_id(call_key, _SECTION_SAMPLING),
            classes="console-inspector-exchange-section",
        )

    @on(Collapsible.Toggled)
    def _on_exchange_toggled(self, event: Collapsible.Toggled) -> None:
        """Single dispatch point for all four Exchange-tab lazy levels.

        Filters on the expanding Collapsible's id PREFIX (turn / call /
        section / message) -- every id outside this tab's four namespaces
        (e.g. a Costs-tab row, handled by ``_on_row_toggled`` above) falls
        through untouched. Each level mounts exactly once per node id,
        tracked in its own dedup set -- and, symmetric with the turn
        level's own retry contract (``_load_exchange_turn`` discards its
        index on failure), the three inner (synchronous) levels only ADD to
        their dedup set once ``_mount_exchange_*_body`` reports success,
        rather than marking-then-maybe-failing: a node whose capture was
        not yet cached or whose Collapsible left the DOM mid-toggle is left
        un-loaded, so collapsing/re-expanding it tries again instead of
        staying permanently empty.
        """
        if not self._capture_revision_is_current():
            return
        collapsible = event.collapsible
        collapsible_id = collapsible.id or ""
        if collapsible.collapsed:
            return

        if collapsible_id.startswith(_EXCHANGE_TURN_ID_PREFIX):
            try:
                turn_index = int(collapsible_id[len(_EXCHANGE_TURN_ID_PREFIX) :])
            except ValueError:
                return
            if turn_index in self._loaded_exchange_turn_indices:
                return
            turn = self._turns_by_index.get(turn_index)
            if turn is None:
                return
            self._loaded_exchange_turn_indices.add(turn_index)
            self.run_worker(
                self._load_exchange_turn(collapsible, turn), exclusive=False
            )
            return

        if collapsible_id.startswith(_EXCHANGE_CALL_ID_PREFIX):
            call_key = collapsible_id[len(_EXCHANGE_CALL_ID_PREFIX) :]
            if call_key in self._loaded_exchange_call_keys:
                return
            if self._mount_exchange_call_body(collapsible, call_key):
                self._loaded_exchange_call_keys.add(call_key)
            return

        if collapsible_id.startswith(_EXCHANGE_SECTION_ID_PREFIX):
            if collapsible_id in self._loaded_exchange_section_ids:
                return
            remainder = collapsible_id[len(_EXCHANGE_SECTION_ID_PREFIX) :]
            call_key, _, section = remainder.rpartition("-")
            if self._mount_exchange_section_body(collapsible, call_key, section):
                self._loaded_exchange_section_ids.add(collapsible_id)
            return

        if collapsible_id.startswith(_EXCHANGE_MESSAGE_ID_PREFIX):
            if collapsible_id in self._loaded_exchange_message_ids:
                return
            if self._mount_exchange_message_body(collapsible):
                self._loaded_exchange_message_ids.add(collapsible_id)
            return

    async def _load_exchange_turn(
        self, collapsible: Collapsible, turn: InspectorTurn
    ) -> None:
        """Fetch one turn's captures and mount one Collapsible per call.

        Twin of ``_load_turn_captures`` (same re-sort, same failure/empty
        handling and messages) but building CALL-level Collapsibles instead
        of flat Static lines, and caching each capture by its call key so
        every deeper level (sections, messages, Copy/Save) can build
        synchronously off already-fetched data.
        """
        if not self._capture_revision_is_current():
            self._loaded_exchange_turn_indices.discard(turn.index)
            return
        load_failed = False
        try:
            pairs = await self._exchanges_loader(turn.native_message_id)
        except Exception as exc:
            # No traceback -- same rationale as ``_load_turn_captures``'s
            # twin handler above: a failure inside ``_exchanges_loader`` can
            # leave a decoded ``ExchangeCapture`` payload in one of that
            # call's own frames, and diagnose would dump it. Turn identity
            # (not capture content) is enough to diagnose and retry.
            logger.error(
                f"console_inspector: exchanges_loader failed for exchange "
                f"turn {turn.index} ({turn.native_message_id}): "
                f"{type(exc).__name__}"
            )
            pairs = []
            load_failed = True

        if not self._capture_revision_is_current():
            self._loaded_exchange_turn_indices.discard(turn.index)
            return

        if not collapsible.is_mounted:
            return
        try:
            contents = collapsible.query_one(Collapsible.Contents)
        except NoMatches:
            return
        if not contents.is_mounted:
            return

        if load_failed:
            self._loaded_exchange_turn_indices.discard(turn.index)
            if not self._capture_revision_is_current():
                await self._invalidate_stale_exchange_mounts()
                return
            await contents.mount(Static(_LOAD_FAILURE_MESSAGE, markup=False))
            if not self._capture_revision_is_current():
                await self._invalidate_stale_exchange_mounts()
            return

        if not pairs:
            if not self._capture_revision_is_current():
                await self._invalidate_stale_exchange_mounts()
                self._loaded_exchange_turn_indices.discard(turn.index)
                return
            await contents.mount(Static(_NO_CAPTURES_MESSAGE, markup=False))
            if not self._capture_revision_is_current():
                await self._invalidate_stale_exchange_mounts()
                self._loaded_exchange_turn_indices.discard(turn.index)
            return

        ordered = sorted(pairs, key=lambda pair: (pair[0].created_at, pair[0].seq))
        collapsible.title = Content.from_text(
            self._exchange_turn_title(turn, call_count=len(ordered)), markup=False
        )
        for call_ordinal, (capture, abandoned) in enumerate(ordered):
            if not self._capture_revision_is_current():
                await self._invalidate_stale_exchange_mounts()
                self._loaded_exchange_turn_indices.discard(turn.index)
                return
            call_key = f"{turn.index}-{call_ordinal}"
            self._exchange_capture_by_call_key[call_key] = capture
            await contents.mount(
                Collapsible(
                    title=Content.from_text(
                        self._exchange_call_title(capture, abandoned), markup=False
                    ),
                    collapsed=True,
                    id=f"{_EXCHANGE_CALL_ID_PREFIX}{call_key}",
                    classes="console-inspector-exchange-call",
                )
            )
            if not self._capture_revision_is_current():
                await self._invalidate_stale_exchange_mounts()
                self._loaded_exchange_turn_indices.discard(turn.index)
                return

    def _mount_exchange_call_body(
        self, collapsible: Collapsible, call_key: str
    ) -> bool:
        """Populate one call's Collapsible: the omitted-keys/reported-usage
        lines and the Copy/Save row mount immediately (cheap, always
        useful); the six section Collapsibles mount with EMPTY bodies --
        each one's own TextArea (or, for Messages, its per-message
        Collapsibles) waits for that section's own first expand.

        Returns whether it actually mounted anything -- ``False`` (capture
        not yet cached, or the Collapsible already left the DOM) tells the
        caller NOT to mark this call as loaded, so a later re-expand
        retries instead of leaving a permanently empty node (review
        finding 5)."""
        if not self._capture_revision_is_current():
            return False
        capture = self._exchange_capture_by_call_key.get(call_key)
        if capture is None:
            return False
        try:
            contents = collapsible.query_one(Collapsible.Contents)
        except NoMatches:
            return False

        if capture.omitted_keys:
            contents.mount(
                Static(
                    f"Omitted by capture policy: {', '.join(capture.omitted_keys)}",
                    markup=False,
                )
            )

        usage = ProviderUsage.from_json(capture.usage_json)
        if usage is not None:
            contents.mount(Static(self._reported_usage_line(usage), markup=False))

        contents.mount(
            Horizontal(
                Button(
                    "Export…",
                    id=f"{_EXCHANGE_EXPORT_BUTTON_PREFIX}{call_key}",
                ),
                classes="console-inspector-exchange-call-actions",
            )
        )

        contents.mount(self._build_system_prompt_section(capture, call_key))
        contents.mount(self._build_messages_section(capture, call_key))
        contents.mount(self._build_tools_section(capture, call_key))
        contents.mount(self._build_response_section(capture, call_key))
        tool_calls_section = self._build_tool_calls_section(capture, call_key)
        if tool_calls_section is not None:
            contents.mount(tool_calls_section)
        contents.mount(self._build_sampling_section(capture, call_key))
        return True

    def _mount_exchange_section_body(
        self, collapsible: Collapsible, call_key: str, section: str
    ) -> bool:
        """Returns whether it actually mounted the section's body --
        ``False`` leaves the section un-loaded so a later re-expand
        retries (review finding 5), same contract as
        ``_mount_exchange_call_body``."""
        if not self._capture_revision_is_current():
            return False
        capture = self._exchange_capture_by_call_key.get(call_key)
        if capture is None:
            return False
        try:
            contents = collapsible.query_one(Collapsible.Contents)
        except NoMatches:
            return False

        if section == _SECTION_SYSTEM:
            text = str(capture.request.get("system_message") or "")
            contents.mount(TextArea(text, read_only=True))
            return True

        if section == _SECTION_MESSAGES:
            messages = capture.request.get("messages_payload")
            if not isinstance(messages, list):
                messages = []
            for message_ordinal, message in enumerate(messages):
                role = (
                    message.get("role", "?") if isinstance(message, dict) else "?"
                )
                message_id = (
                    f"{_EXCHANGE_MESSAGE_ID_PREFIX}{call_key}-{message_ordinal}"
                )
                self._exchange_message_by_id[message_id] = message
                contents.mount(
                    Collapsible(
                        title=Content.from_text(
                            f"[{message_ordinal}] {role}", markup=False
                        ),
                        collapsed=True,
                        id=message_id,
                        classes="console-inspector-exchange-message",
                    )
                )
            return True

        if section == _SECTION_TOOLS:
            tools = capture.request.get("tools") or []
            contents.mount(TextArea(self._json_block(tools), read_only=True))
            return True

        if section == _SECTION_RESPONSE:
            contents.mount(
                TextArea(self._response_text(capture), read_only=True)
            )
            return True

        if section == _SECTION_TOOL_CALLS:
            tool_calls = (
                capture.response.get("tool_calls") if capture.response else None
            )
            contents.mount(
                TextArea(self._json_block(tool_calls or []), read_only=True)
            )
            return True

        if section == _SECTION_SAMPLING:
            sampling = {
                key: value
                for key, value in capture.request.items()
                if key not in _SAMPLING_EXCLUDED_REQUEST_KEYS
            }
            contents.mount(TextArea(self._json_block(sampling), read_only=True))
            return True

        # Unreachable in practice -- every section id this widget itself
        # generates is one of the six keys above -- but an unrecognized
        # section must still not be marked loaded.
        return False

    def _mount_exchange_message_body(self, collapsible: Collapsible) -> bool:
        """Same success/failure contract as the call/section levels above
        (review finding 5)."""
        if not self._capture_revision_is_current():
            return False
        message = self._exchange_message_by_id.get(collapsible.id or "")
        if message is None:
            return False
        try:
            contents = collapsible.query_one(Collapsible.Contents)
        except NoMatches:
            return False
        contents.mount(TextArea(self._json_block(message), read_only=True))
        return True

    @on(Button.Pressed)
    def _on_exchange_call_button(self, event: Button.Pressed) -> None:
        button_id = event.button.id or ""
        if button_id.startswith(_EXCHANGE_EXPORT_BUTTON_PREFIX):
            event.stop()
            call_key = button_id[len(_EXCHANGE_EXPORT_BUTTON_PREFIX) :]
            self._open_exchange_export(call_key)

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
        capture = self._exchange_capture_by_call_key.get(call_key)
        if capture is None:
            return False
        expected = self._capture_revision_at_open
        provider = self._capture_revision_provider
        if expected is None or provider is None:
            expected = 0

            def provider() -> int:
                return 0
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
        the user's home directory into the log for every rejected save --
        this modal's own log line right below already names the path
        deliberately, so the redaction only removes an UNINTENTIONAL
        second copy from a module this file does not otherwise control.
        """
        downloads_dir = Path.home() / "Downloads"
        try:
            validate_path(path, downloads_dir, redact_paths=True)
            return path
        except ValueError as exc:
            logger.error(
                f"Rejected export destination {path}: {type(exc).__name__}"
            )
            self.notify(
                f"Save failed ({type(exc).__name__}): {path}", severity="error"
            )
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
        loading = self.query_one(
            "#console-inspector-next-send-loading", LoadingIndicator
        )
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

        header = self.query_one("#console-inspector-next-send-header", Static)
        header_text = "Chat Context"
        if self._token_estimate is not None:
            header_text += f" (~{self._token_estimate} tokens)"
        header.update(header_text)

        current_container = self.query_one(
            "#console-inspector-next-send-current-body", Vertical
        )
        current_container.remove_children()
        for widget in self._build_current_context_widgets():
            current_container.mount(widget)

        next_container = self.query_one(
            "#console-inspector-next-send-payload-body", Vertical
        )
        next_container.remove_children()
        for widget in self._build_next_send_widgets():
            next_container.mount(widget)

    def _build_current_context_widgets(self) -> list[Widget]:
        if not self.snapshot.current_messages:
            # LY-13 (TASK-2154.23): guidance, not a void. Prefix kept so the
            # existing "No conversation context" pins still match.
            return [
                Label(
                    "No conversation context yet.\n"
                    "Messages you send and receive will appear here.\n"
                    "The Next Send tab shows the exact payload — model, "
                    "system prompt, staged sources — your next message ships with.",
                    markup=False,
                )
            ]
        return [
            Collapsible(
                TextArea(msg.content, read_only=True),
                # Content.from_text(..., markup=False): same guard as every
                # other Collapsible title in this file (hard constraint 1)
                # -- msg.role/msg.status are enum-derived today, but a
                # Collapsible title IS markup-parsed by default and this
                # file does not leave that to chance anywhere else.
                title=Content.from_text(
                    f"[{msg.role}] {msg.status}", markup=False
                ),
                collapsed=True,
            )
            for msg in self.snapshot.current_messages
        ]

    def _build_next_send_widgets(self) -> list[Widget]:
        widgets: list[Widget] = []
        payload = self.snapshot.next_send_payload
        text = self._format_next_send_text()

        if len(text.encode("utf-8")) > SIZE_THRESHOLD_BYTES:
            widgets.append(
                Label(
                    "Context exceeds 1 MiB. Use Save to File to view the "
                    "full payload.",
                    markup=False,
                )
            )
            return widgets

        if self.raw_json:
            widgets.append(TextArea(text, read_only=True))
            return widgets

        widgets.append(
            Collapsible(
                Label(str(payload.get("model", "unknown")), markup=False),
                title=Content.from_text("Model", markup=False),
                collapsed=False,
            )
        )

        widgets.append(
            Collapsible(
                TextArea(self._json_block(payload.get("system")), read_only=True),
                title=Content.from_text("System", markup=False),
                collapsed=True,
            )
        )

        message_widgets = []
        for i, msg in enumerate(payload.get("messages", [])):
            message_widgets.append(
                Collapsible(
                    TextArea(self._json_block(msg), read_only=True),
                    title=Content.from_text(f"Message {i}", markup=False),
                    collapsed=True,
                )
            )
        widgets.append(
            Collapsible(
                *message_widgets,
                title=Content.from_text("Messages", markup=False),
                collapsed=False,
            )
        )

        response_prefill = payload.get("response_prefill")
        if response_prefill:
            widgets.append(
                Collapsible(
                    Label(
                        "The reply will continue from this prefill; the agent "
                        "loop (tools/MCP) is skipped for this send.",
                        markup=False,
                    ),
                    TextArea(self._json_block(response_prefill), read_only=True),
                    title=Content.from_text("Response Prefill", markup=False),
                    collapsed=False,
                )
            )

        tools = payload.get("tools")
        if tools:
            widgets.append(
                Collapsible(
                    TextArea(self._json_block(tools), read_only=True),
                    title=Content.from_text("Tools", markup=False),
                    collapsed=True,
                )
            )

        staged = payload.get("staged_sources")
        if staged:
            widgets.append(
                Collapsible(
                    TextArea(self._json_block(staged), read_only=True),
                    title=Content.from_text("Staged Sources", markup=False),
                    collapsed=True,
                )
            )

        return widgets

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
        self.next_send_loading = True
        try:
            new_snapshot = await self._snapshot_factory()
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
            if self._estimate_factory is not None:
                self._token_estimate = self._estimate_factory()
            self.snapshot = new_snapshot
            self.call_after_refresh(self._focus_initial_control)
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
        self.run_worker(
            self._load_snapshot,
            exclusive=True,
            group=_NEXT_SEND_WORKER_GROUP,
            name="load_snapshot",
        )

    @on(Button.Pressed, "#console-inspector-next-send-copy")
    def _copy_json(self, event: Button.Pressed) -> None:
        event.stop()
        # ``_format_export_text``, NOT ``_format_next_send_text`` (task-18300):
        # this leaves the modal for the OS clipboard, so any automatically
        # injected project-instruction message body must be scrubbed first --
        # see that method's own docstring for the privacy contract.
        text = self._format_export_text()
        try:
            import pyperclip

            pyperclip.copy(text)
            self.notify("JSON copied to clipboard.")
        except Exception as exc:
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
        if self._save_blocked_reason is not None:
            return
        # ``_format_export_text``, NOT ``_format_next_send_text`` (task-18300)
        # -- same privacy contract as ``_copy_json`` above, this text lands
        # on disk.
        text = self._format_export_text()
        filename = f"chatbook_context_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
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
            # str() can also embed the offending path. type(exc).__name__
            # plus the path we attempted is enough to diagnose (permissions,
            # disk full, path too long) without echoing content into the
            # log OR the user-facing toast -- the retired standalone context
            # modal's own ``_save_json`` put the raw exception text in a
            # USER-VISIBLE notify() (hard constraint 3); this names the
            # failure class and path instead.
            logger.error(
                f"Failed to save context snapshot to {path}: {type(exc).__name__}"
            )
            self.notify(
                f"Save failed ({type(exc).__name__}): {path}", severity="error"
            )
        except Exception as exc:
            logger.error(
                f"Unexpected error saving context snapshot to {path}: "
                f"{type(exc).__name__}"
            )
            self.notify(
                f"Save failed ({type(exc).__name__}): {path}", severity="error"
            )

    def action_refresh(self) -> None:
        """"r" binding: refresh the ACTIVE tab, when it has a live-reload
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
        if tabs.active == TAB_NEXT_SEND:
            self.run_worker(
                self._load_snapshot,
                exclusive=True,
                group=_NEXT_SEND_WORKER_GROUP,
                name="load_snapshot",
            )

    async def action_dismiss(self) -> None:
        """Defensive fallback for the built-in "dismiss" action name."""
        await self.request_safe_cancel(source="visible")

    @on(Button.Pressed, f"#{CLOSE_BUTTON_ID}")
    async def _close(self, event: Button.Pressed) -> None:
        event.stop()
        await self.request_safe_cancel(source="visible")
