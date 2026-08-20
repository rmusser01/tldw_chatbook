"""Console Conversation Inspector modal (task-8: scaffold + Costs tab;
task-9: the Exchange tab).

Replaces the two standalone modals (``ConsoleCostModal``, opened from the
cost chip; ``ConsoleContextModal``, opened via Ctrl+Shift+P / the command
palette) with ONE modal that gains a tab per surface: Costs (task-8),
Exchange (this task: per-call request/response detail with status badges),
Next Send (task-10: the assembled next-send payload -- a placeholder Static
until then). Both entry points in ``chat_screen.py`` now push the SAME
instance, differing only by which tab starts active (``initial_tab``).

The two legacy modal files are left untouched until task-10 retires them --
``_format_row``/``_format_totals`` are copied here VERBATIM from
``ConsoleCostModal`` (still pure ``@staticmethod`` formatters over an
already-computed ``ConsoleCostRow``/``ConsoleCostRowTotals``, per that
module's own "already computed, just render it" house pattern).

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

import json
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger
from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.content import Content
from textual.css.query import NoMatches
from textual.screen import ModalScreen
from textual.widget import Widget
from textual.widgets import (
    Button,
    Collapsible,
    Static,
    TabbedContent,
    TabPane,
    TextArea,
)

from tldw_chatbook.Chat.console_chat_models import ConsoleContextSnapshot
from tldw_chatbook.Chat.console_cost_tracker import ConsoleCostRow, ConsoleCostRowTotals
from tldw_chatbook.Chat.console_ephemeral import blocked_reason
from tldw_chatbook.Chat.console_exchange_capture import ExchangeCapture
from tldw_chatbook.Chat.provider_usage import ProviderUsage
from tldw_chatbook.LLM_Calls.pricing_catalog import get_pricing_catalog
from tldw_chatbook.Utils.token_counter import estimate_tokens
from tldw_chatbook.Widgets.modal_dismissal import SafeModalDismissMixin

MODAL_ID = "console-inspector-modal"
CLOSE_BUTTON_ID = "console-inspector-close"
TAB_COSTS = "inspector-costs"
TAB_EXCHANGE = "inspector-exchange"
TAB_NEXT_SEND = "inspector-next-send"
_COST_ROW_ID_PREFIX = "console-inspector-cost-row-"

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
_EXCHANGE_COPY_BUTTON_PREFIX = "console-inspector-exchange-copy-"
_EXCHANGE_SAVE_BUTTON_PREFIX = "console-inspector-exchange-save-"

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
    ``exchanges_loader``/``snapshot_factory`` callables, mirroring
    ``ConsoleCostModal``'s "already computed, just render it" shape for
    the Costs tab's rows/totals.
    """

    DEFAULT_CSS = """
    ConsoleConversationInspector { align: center middle; }
    #console-inspector-modal {
        width: 110; max-width: 95%; height: 42; max-height: 90%;
        border: tall gray; padding: 1 2;
    }
    #console-inspector-header { height: auto; }
    #console-inspector-tabs { height: 1fr; margin-top: 1; }
    #console-inspector-costs-rows { height: 1fr; }
    .console-inspector-cost-row { height: auto; }
    #console-inspector-costs-totals { height: auto; margin-top: 1; text-style: bold; }
    #console-inspector-exchange-turns { height: 1fr; }
    .console-inspector-exchange-turn { height: auto; }
    .console-inspector-exchange-call { height: auto; }
    .console-inspector-exchange-section { height: auto; }
    .console-inspector-exchange-message { height: auto; }
    .console-inspector-exchange-call-actions { height: auto; margin-bottom: 1; }
    #console-inspector-actions { height: auto; margin-top: 1; }
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
    ]
    SAFE_MODAL_CONTENT = f"#{MODAL_ID}"

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
                snapshot. Accepted now (task-10 wires the Next Send tab to
                it) but not yet called by this scaffold.
            token_estimate: Precomputed token estimate for the Next Send
                tab's header, or ``None``.
            estimate_factory: Re-estimate callback for a Next Send refresh,
                or ``None``.
            in_progress: Whether a response is currently in flight (Next
                Send tab warning, task-10).
            ephemeral: Whether the active session is temporary (blocks
                Next Send's Save-to-file affordance, task-10).
            initial_tab: Which tab id starts active -- ``"inspector-costs"``
                from the cost chip, ``"inspector-next-send"`` from Ctrl+Shift+P.
        """
        super().__init__()
        self._rows = list(rows)
        self._totals = totals
        self._turns_by_index = {turn.index: turn for turn in turns}
        self._exchanges_loader = exchanges_loader
        self._snapshot_factory = snapshot_factory
        self._token_estimate = token_estimate
        self._estimate_factory = estimate_factory
        self._in_progress = in_progress
        self._ephemeral = ephemeral
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

    def compose(self) -> ComposeResult:
        """Build the header, tabbed body, and shared Close action."""
        with Vertical(id=MODAL_ID):
            yield Static(
                "Conversation Inspector", id="console-inspector-header", markup=False
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
                    with VerticalScroll(id="console-inspector-exchange-turns"):
                        yield from self._build_exchange_turn_widgets()
                with TabPane("Next Send", id=TAB_NEXT_SEND):
                    yield Static(
                        "Next Send preview is not available yet.",
                        id="console-inspector-next-send-placeholder",
                        markup=False,
                    )
            with Horizontal(id="console-inspector-actions"):
                yield Button("Close", id=CLOSE_BUTTON_ID, variant="primary")

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
        """Pure ``str`` render for one breakdown row (verbatim from
        ``ConsoleCostModal._format_row`` -- task-8 moves it here; the old
        modal keeps its own copy until task-10 retires it).

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
        ``ConsoleCostModal._format_totals``)."""
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
            f"{self._call_cost_line(capture)}"
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
        """Same idiom as ``ConsoleContextModal._json_block``."""
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
            await contents.mount(Static(_LOAD_FAILURE_MESSAGE, markup=False))
            return

        if not pairs:
            await contents.mount(Static(_NO_CAPTURES_MESSAGE, markup=False))
            return

        ordered = sorted(pairs, key=lambda pair: (pair[0].created_at, pair[0].seq))
        collapsible.title = Content.from_text(
            self._exchange_turn_title(turn, call_count=len(ordered)), markup=False
        )
        for call_ordinal, (capture, abandoned) in enumerate(ordered):
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

        save_button = Button(
            "Save to File",
            id=f"{_EXCHANGE_SAVE_BUTTON_PREFIX}{call_key}",
            disabled=self._save_blocked_reason is not None,
        )
        if self._save_blocked_reason is not None:
            save_button.tooltip = self._save_blocked_reason
        contents.mount(
            Horizontal(
                Button("Copy JSON", id=f"{_EXCHANGE_COPY_BUTTON_PREFIX}{call_key}"),
                save_button,
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
        if button_id.startswith(_EXCHANGE_COPY_BUTTON_PREFIX):
            event.stop()
            call_key = button_id[len(_EXCHANGE_COPY_BUTTON_PREFIX) :]
            self._copy_exchange_capture(call_key)
        elif button_id.startswith(_EXCHANGE_SAVE_BUTTON_PREFIX):
            event.stop()
            call_key = button_id[len(_EXCHANGE_SAVE_BUTTON_PREFIX) :]
            self._save_exchange_capture(call_key)

    def _copy_exchange_capture(self, call_key: str) -> None:
        """Verbatim idiom from ``ConsoleContextModal._copy_json`` (that
        sibling interpolates ``exc``'s own message text into its log line;
        this copy does NOT -- ``pyperclip.copy(text)`` failing (e.g. a
        codec error while encoding ``text``) can embed a fragment of the
        very payload ``text`` was built from inside ``str(exc)``, and this
        is the one call in this file review finding 1's "no capture
        content, no exception message body" rule would otherwise miss),
        applied to one call's ``ExchangeCapture`` instead of the whole
        snapshot."""
        capture = self._exchange_capture_by_call_key.get(call_key)
        if capture is None:
            return
        text = json.dumps(asdict(capture), indent=2, default=str)
        try:
            import pyperclip

            pyperclip.copy(text)
            self.notify("JSON copied to clipboard.")
        except Exception as exc:
            logger.warning(
                f"Failed to copy exchange capture JSON to clipboard: "
                f"{type(exc).__name__}"
            )
            self.notify("Copy failed: pyperclip unavailable.", severity="warning")

    def _save_exchange_capture(self, call_key: str) -> None:
        """Verbatim idiom from ``ConsoleContextModal._save_json``, applied
        to one call's ``ExchangeCapture``. This method writes UNCONDITIONALLY
        -- the only enforcement of ``self._save_blocked_reason`` is the
        Save button's own ``disabled=`` state (set in
        ``_mount_exchange_call_body``); a direct call here (e.g. a future
        caller that bypasses the button) still writes to disk."""
        capture = self._exchange_capture_by_call_key.get(call_key)
        if capture is None:
            return
        text = json.dumps(asdict(capture), indent=2, default=str)
        filename = f"chatbook_exchange_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        path = Path.home() / "Downloads" / filename
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
            self.notify(f"Saved to {path}")
        except OSError as exc:
            # No exception text, no traceback: this frame's locals include
            # `text` -- the FULL capture payload (system prompt, messages,
            # tool schemas, response) -- and loguru's diagnose formatter
            # would otherwise annotate the failing source line's names
            # (including `text`) with their values. type(exc).__name__ is
            # enough to diagnose an OSError (permissions, disk full, path
            # too long) without echoing content; an OSError's own str() can
            # also embed the offending path/filename, which is fine, but we
            # skip it here for the same reason app.py's own file-write
            # error handlers do (see app.py's three "No traceback" comments).
            logger.error(
                f"Failed to save exchange capture to {path}: {type(exc).__name__}"
            )
            self.notify(f"Save failed: {exc}", severity="error")
        except Exception as exc:
            logger.error(
                f"Unexpected error saving exchange capture to {path}: "
                f"{type(exc).__name__}"
            )
            self.notify(f"Save failed: {exc}", severity="error")

    def action_refresh(self) -> None:
        """"r" binding, wired for parity with ``ConsoleContextModal``.

        Nothing to refresh yet in this scaffold: the Costs tab's rows are
        precomputed by the caller (no live-recompute entry point exists
        here, matching ``ConsoleCostModal``'s own shape) and the Next Send
        tab is a placeholder until task-10 wires ``snapshot_factory`` to
        an actual reload.
        """
        return

    async def action_dismiss(self) -> None:
        """Defensive fallback for the built-in "dismiss" action name."""
        await self.request_safe_cancel(source="visible")

    @on(Button.Pressed, f"#{CLOSE_BUTTON_ID}")
    async def _close(self, event: Button.Pressed) -> None:
        event.stop()
        await self.request_safe_cancel(source="visible")
