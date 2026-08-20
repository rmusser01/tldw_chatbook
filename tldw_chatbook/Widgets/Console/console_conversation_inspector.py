"""Console Conversation Inspector modal (task-8: scaffold + Costs tab).

Replaces the two standalone modals (``ConsoleCostModal``, opened from the
cost chip; ``ConsoleContextModal``, opened via Ctrl+Shift+P / the command
palette) with ONE modal that gains a tab per surface: Costs (this task),
Exchange (task-9: per-call request/response detail with status badges),
Next Send (task-10: the assembled next-send payload -- a placeholder Static
in THIS task). Both entry points in ``chat_screen.py`` now push the SAME
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
    - For an in-memory (not-yet-persisted / ephemeral-session) capture
      there is currently no reliable per-capture ``abandoned`` signal on
      ``ConsoleChatMessage`` itself (the store's own bookkeeping for this
      -- ``_abandoned_exchange_run_tags`` -- is a private, attach-time-only
      set with no public accessor); callers built against the native store
      path should pass ``False`` until that plumbing exists, matching
      ``chat_screen.py``'s own loader in this task.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass

from loguru import logger
from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.content import Content
from textual.css.query import NoMatches
from textual.screen import ModalScreen
from textual.widget import Widget
from textual.widgets import Button, Collapsible, Static, TabbedContent, TabPane

from tldw_chatbook.Chat.console_chat_models import ConsoleContextSnapshot
from tldw_chatbook.Chat.console_cost_tracker import ConsoleCostRow, ConsoleCostRowTotals
from tldw_chatbook.Chat.console_exchange_capture import ExchangeCapture
from tldw_chatbook.Chat.provider_usage import ProviderUsage
from tldw_chatbook.LLM_Calls.pricing_catalog import get_pricing_catalog
from tldw_chatbook.Widgets.modal_dismissal import SafeModalDismissMixin

MODAL_ID = "console-inspector-modal"
CLOSE_BUTTON_ID = "console-inspector-close"
TAB_COSTS = "inspector-costs"
TAB_EXCHANGE = "inspector-exchange"
TAB_NEXT_SEND = "inspector-next-send"
_COST_ROW_ID_PREFIX = "console-inspector-cost-row-"


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
                    yield Static(
                        "Exchange detail view is not available yet.",
                        id="console-inspector-exchange-placeholder",
                        markup=False,
                    )
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
        except Exception:
            logger.opt(exception=True).warning(
                "console_inspector: exchanges_loader failed for turn {}",
                turn.native_message_id,
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
            await contents.mount(
                Static(
                    "Could not load captures for this turn -- expand again "
                    "to retry.",
                    markup=False,
                )
            )
            return

        if not pairs:
            await contents.mount(
                Static(
                    "No capture recorded for this turn (recorded before "
                    "capture existed, capture disabled, or capture failed).",
                    markup=False,
                )
            )
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
