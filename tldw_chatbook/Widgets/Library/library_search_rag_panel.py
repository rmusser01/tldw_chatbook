"""Textual widget for Library-native Search/RAG."""

from __future__ import annotations

import asyncio
from decimal import Decimal

from loguru import logger
from rich.markup import escape as escape_markup

from textual import on, work
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.css.query import NoMatches
from textual.widgets import Button, Collapsible, Static
from textual.widget import Widget

from ...Chat.cost_display import build_provenance_line
from ...LLM_Calls.pricing_catalog import get_pricing_catalog
from ...Library.library_rag_answer_service import (
    ANSWER_STATUS_ABSTAINED,
    ANSWER_STATUS_FAILED,
    ANSWER_STATUS_NO_EVIDENCE,
    ANSWER_STATUS_READY,
    LibraryRagAnswer,
)
from ...Library.library_rag_state import (
    LIBRARY_RAG_SCOPE_TOGGLE_SOURCE_TYPES,
    LibraryRagPanelState,
    LibraryRagQueryState,
    LibraryRagResultRow,
    LibraryRagScopeState,
    LibraryRagSourceOption,
    library_rag_answer_display_text,
    library_rag_empty_state_quiet_copy,
    library_rag_paid_mode_notice,
    library_rag_score_suffix,
    library_rag_scope_summary,
    searching_status_line,
)
from ...Library.library_rechunk_service import (
    RECHUNK_SLOT,
    RECHUNK_WORKER_GROUP,
    acquire_bulk_rag_slot,
    bulk_rag_slot_in_flight,
    format_rechunk_summary,
    release_bulk_rag_slot,
)
from .library_rail import SelectAllOnFocusingClickInput


from tldw_chatbook.Widgets.Library.library_canvas_sync import (
    PostRecomposeCallback,
)


class LibrarySearchRagPanel(PostRecomposeCallback, VerticalScroll):
    """Display the source scope, query controls, and evidence results."""

    #: Stable id for the legacy-chunk report line (task-12, spec §10.1) --
    #: named here so Task 13's re-chunk control can find its sibling.
    LEGACY_CHUNK_REPORT_LINE_ID = "library-rag-legacy-chunk-line"

    #: Stable ids for task-13's re-chunk control + its summary line
    #: (spec §10.2-§10.3).
    RECHUNK_BUTTON_ID = "library-rag-rechunk-legacy"
    RECHUNK_SUMMARY_ID = "library-rag-rechunk-summary"

    def __init__(self, state: LibraryRagPanelState, **kwargs) -> None:
        super().__init__(**kwargs)
        self.state = state
        # task-12 (spec §10.1): cached copy of the legacy-chunk report line
        # fetched off the mount path. Compose re-reads this cache on every
        # rebuild (the ingest canvas's template-name cache pattern), so the
        # fetched line survives `sync_state` recomposes without re-querying.
        self._legacy_chunk_report: str = ""

    def sync_state(self, state: LibraryRagPanelState) -> None:
        """Rebuild only this mounted Search/RAG panel from ``state``.

        Args:
            state: Complete Search/RAG controls and results state to render.
        """
        self.state = state
        self.refresh(recompose=True)

    def on_show(self) -> None:
        """Fetch the legacy-chunk report once the canvas is actually visible.

        (task-12, spec §10.1) The report is sourced through the app's
        ``rag_admin_scope_service`` -- the ``rag.admin.observe.local``
        action -- scheduled OFF the mount path into a worker (mount-time DB
        populate is the documented "(0) count" trap; the ingest canvas's
        template picker established this exact shape). The Library screen
        remounts this canvas on every destination switch, so each visit
        re-queries: a re-chunk (Task 13) or new ingest is reflected the
        next time the user lands here.
        """
        self._request_legacy_chunk_report_refresh()

    def _request_legacy_chunk_report_refresh(self) -> None:
        """Schedule the legacy-chunk report fetch worker (once per show)."""
        try:
            self.run_worker(
                self._fetch_legacy_chunk_report(),
                group="library-rag-legacy-chunk-report",
                exclusive=True,
            )
        except Exception:
            # A worker-scheduling failure must never break the canvas.
            return

    async def _fetch_legacy_chunk_report(self) -> None:
        """Query the legacy-chunk report line via the scope service.

        TASK-21126: this is an ASYNC worker, so "off the mount path" is not
        the same as "off the event loop" — until that task the scope
        service evaluated the local backend's synchronous
        ``get_template_diagnostics`` (and with it the legacy-chunk census
        SELECT) inline here, freezing the UI for the duration. The census
        now runs on a worker thread inside
        ``RAGAdminScopeService._call_off_loop``; this coroutine only awaits
        it. Keep it that way: any new work added here runs on the loop.

        Consumes ONLY the payload's ``legacy_chunk_report`` field. The same
        payload's ``capability`` / ``missing_methods`` / ``fallback_enabled``
        are HARDCODED upstream (spec §11 item 4) and never render here --
        surfacing them would be a fabricated health claim. Degrades quietly
        on every failure shape (missing service, policy denial, store
        error): the line simply stays omitted, which is also its honest
        empty state (omit-when-empty, spec §10.1 -- a clean library shows
        nothing rather than a zero).
        """
        service = getattr(self.app, "rag_admin_scope_service", None)
        get_diagnostics = getattr(service, "get_template_diagnostics", None)
        if not callable(get_diagnostics):
            return
        try:
            payload = await get_diagnostics(mode="local")
        except Exception:
            return
        report = str((payload or {}).get("legacy_chunk_report") or "").strip()
        self._legacy_chunk_report = report
        self._apply_legacy_chunk_report(report)

    def _apply_legacy_chunk_report(self, report: str) -> None:
        """Show/hide the mounted report line in place (no remove/mount).

        Plain ``Static.update()`` + a ``display`` flip -- the same
        yield-free class of write the screen's snapshot syncers use, so
        this can never interleave with the panel's other refresh callers.

        task-13: the re-chunk control rides the report's visibility -- it
        is offered exactly when there is something older-engine to re-chunk
        (a fully stamped library shows neither).
        """
        try:
            line = self.query_one(
                f"#{self.LEGACY_CHUNK_REPORT_LINE_ID}", Static
            )
        except NoMatches:
            # Mid-recompose: the cache is set, so the rebuild renders it.
            return
        line.update(report)
        line.display = bool(report)
        try:
            button = self.query_one(f"#{self.RECHUNK_BUTTON_ID}", Button)
        except NoMatches:
            return
        # Never hide the control mid-run: an in-flight re-chunk keeps its
        # button mounted (disabled) even if this refresh lands an empty
        # report -- the summary line still has to surface.
        button.display = bool(report) or bulk_rag_slot_in_flight(RECHUNK_SLOT)

    def _legacy_chunk_report_line(self) -> Static:
        """Build the report line ``Static`` (always mounted, display-gated).

        Always mounted and shown/hidden via ``display`` rather than
        conditionally composed -- an async-fetched, instance-cached line
        must never depend on a remove/mount racing the panel's recompose
        cycle. ``display = False`` removes it from the layout entirely, so
        omit-when-empty holds visually: no line, no reserved row.
        """
        line = Static(
            self._legacy_chunk_report,
            id=self.LEGACY_CHUNK_REPORT_LINE_ID,
            classes="library-rag-quiet-line",
            # Service-built copy, but interpolated from DB state -- render
            # literally, matching the panel's other quiet lines.
            markup=False,
        )
        line.display = bool(self._legacy_chunk_report)
        return line

    def _rechunk_action_children(self) -> list[Widget]:
        """The re-chunk control + its summary row (task-13, spec §10.2).

        The control shares the report line's visibility (both derive from
        the cached report): it is offered exactly when older-engine items
        exist, so a fully stamped library shows neither. Always mounted and
        ``display``-gated -- the same never-remove/mount rule the report
        line follows, so a mid-run recompose cannot eat the summary.
        """
        shown = bool(self._legacy_chunk_report) or bulk_rag_slot_in_flight(
            RECHUNK_SLOT
        )
        button = Button(
            "Re-chunk older-engine items",
            id=self.RECHUNK_BUTTON_ID,
            classes="library-rag-recovery-action",
            tooltip=(
                "Re-chunk items persisted before the current chunking "
                "engine through the template-aware path, then re-index "
                "them. Runs cannot overlap a RAG index backfill."
            ),
        )
        button.display = shown
        summary = Static(
            "",
            id=self.RECHUNK_SUMMARY_ID,
            classes="library-rag-quiet-line",
            # Counts plus service-built notes -- literal, never markup.
            markup=False,
        )
        summary.styles.height = 1
        summary.display = False
        return [button, summary]

    @on(Button.Pressed, f"#{RECHUNK_BUTTON_ID}")
    def _handle_rechunk_legacy_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        self._trigger_rechunk_legacy()

    def _trigger_rechunk_legacy(self) -> None:
        """Guard, then launch the re-chunk worker (spec §10.3).

        The mutual in-flight guard with the Settings backfill lives in the
        shared slot registry -- a REFUSAL with a notice, never Textual
        worker cancellation (``exclusive=True`` CANCELS same-group workers
        on Textual 8.2.8; the task-228 lesson, deliberately not "fixed").
        """
        refusal = acquire_bulk_rag_slot(RECHUNK_SLOT)
        if refusal is not None:
            self.app.notify(refusal, severity="warning")
            return
        try:
            button = self.query_one(f"#{self.RECHUNK_BUTTON_ID}", Button)
        except NoMatches:
            pass
        else:
            button.disabled = True
        try:
            summary = self.query_one(f"#{self.RECHUNK_SUMMARY_ID}", Static)
        except NoMatches:
            pass
        else:
            summary.update("Re-chunking…")
            summary.display = True
        self._rechunk_legacy_worker()

    @work(thread=True, group=RECHUNK_WORKER_GROUP, exclusive=False)
    def _rechunk_legacy_worker(self) -> None:
        """The re-chunk worker (spec §10.2-§10.3), on its OWN group.

        ``exclusive=False`` is written out deliberately: this worker group
        must NEVER gain exclusive semantics -- Textual 8.2.8 cancels
        same-group workers, and the mutual exclusion with the backfill is
        the guard slot's job (a refusal notice), not cancellation's. The
        spec documents this as a measured deviation from CLAUDE.md gotcha
        9; do not "fix" it back.

        Thread worker (not async-on-the-loop): the per-item chunking and
        the chunk-row transaction are long synchronous stretches, exactly
        like the backfill worker's rationale. Services are pre-resolved
        OUTSIDE the transient ``asyncio.run`` loop (the #700-hardened
        pattern the backfill worker documents) so the shared RAG service
        is never constructed for the first time inside a loop that closes
        when this run finishes.
        """
        from ...RAG_Search.ingestion_indexing import (
            get_shared_rag_service,
            semantic_indexing_available,
        )
        from ...runtime_policy.types import PolicyDeniedError

        try:
            scope = getattr(self.app, "rag_admin_scope_service", None)
            launch = getattr(scope, "rechunk_legacy_media", None)
            if scope is None or not callable(launch):
                self.app.call_from_thread(
                    self.app.notify,
                    "Re-chunk could not start: the RAG admin service is "
                    "unavailable right now.",
                    severity="error",
                )
                return
            # §10.2.1: the whole re-index step is conditional on the
            # semantic index being enabled/present; the summary discloses
            # the skip. Pre-resolved here, before the transient loop.
            rag_service = None
            if semantic_indexing_available():
                rag_service = get_shared_rag_service()
            summary = asyncio.run(
                launch(mode="local", rag_service=rag_service)
            )
        except PolicyDeniedError as denied:
            self.app.call_from_thread(
                self.app.notify,
                f"Re-chunk was blocked by policy: {denied.user_message}",
                severity="error",
            )
            return
        except Exception as exc:
            logger.error(f"Legacy re-chunk worker crashed: {exc}")
            self.app.call_from_thread(
                self.app.notify, f"Re-chunk failed: {exc}", severity="error"
            )
            return
        finally:
            release_bulk_rag_slot(RECHUNK_SLOT)
            self.app.call_from_thread(self._finish_rechunk_run)
        line = format_rechunk_summary(summary)
        self.app.call_from_thread(self._apply_rechunk_summary, line)
        self.app.call_from_thread(
            self.app.notify, f"Re-chunk finished: {line}", severity="information"
        )

    def _apply_rechunk_summary(self, line: str) -> None:
        """Surface the run summary (main thread)."""
        try:
            summary = self.query_one(f"#{self.RECHUNK_SUMMARY_ID}", Static)
        except NoMatches:
            return
        summary.update(line)
        summary.display = bool(line)

    def _finish_rechunk_run(self) -> None:
        """Re-enable the control and refresh the (now lower) report count."""
        try:
            button = self.query_one(f"#{self.RECHUNK_BUTTON_ID}", Button)
        except NoMatches:
            pass
        else:
            button.disabled = False
            if not self._legacy_chunk_report and not bulk_rag_slot_in_flight(
                RECHUNK_SLOT
            ):
                button.display = False
        try:
            summary = self.query_one(f"#{self.RECHUNK_SUMMARY_ID}", Static)
        except NoMatches:
            pass
        else:
            # A failure path never lands a summary line -- retire the
            # in-flight placeholder so it cannot read as a stuck run.
            # (On success this runs BEFORE the summary lands, so a real
            # summary is never cleared.)
            if str(summary.renderable) == "Re-chunking…":
                summary.update("")
                summary.display = False
        # The report count dropped by however many items were re-chunked;
        # refresh it in place rather than waiting for the next remount.
        self._request_legacy_chunk_report_refresh()

    def compose(self) -> ComposeResult:
        # task-2859 item 7: drop the "Library " prefix (this canvas already
        # lives inside the Library destination) and match the rail row's
        # own spaced "Search / RAG" (library_shell_state.py) -- the canvas
        # used to say "Library Search/RAG", disagreeing with the rail on
        # both the prefix and the slash spacing. NOT the same string as
        # the cross-app "Library Search/RAG" evidence-provenance label
        # (``OWNER_LIBRARY_RAG``/``source=`` on staged Console evidence) --
        # that vocabulary is deliberately unchanged here.
        yield Static(
            "Search / RAG",
            id="library-rag-panel-title",
            classes="destination-section",
            markup=False,
        )
        with Vertical(
            id="library-rag-query-controls",
            classes=_query_region_classes(self.state),
        ):
            yield Button(
                _mode_toggle_label(self.state),
                id="library-rag-mode-toggle",
                tooltip=_mode_toggle_tooltip(self.state),
            )
            # LIB-17: this box is prefilled with the last-run query on every
            # rebuild -- SelectAllOnFocusingClickInput extends the rail
            # search box's own click-select-all fix here so a click-then-
            # type on a stale query replaces it instead of inserting at the
            # click position.
            yield SelectAllOnFocusingClickInput(
                value=self.state.query_state.query,
                placeholder="Ask or search Library sources",
                id="library-rag-query-input",
            )
            for child in library_rag_query_status_children(self.state):
                yield child
            yield Button(
                self.state.query_state.run_action.label,
                id=self.state.query_state.run_action.widget_id,
                disabled=not self.state.query_state.run_action.enabled,
                tooltip=self.state.query_state.run_action.tooltip,
            )

        with Vertical(
            id="library-rag-source-scope",
            classes=_scope_region_classes(self.state),
        ):
            yield Static(
                "Sources",
                id="library-rag-scope-heading",
                classes="destination-section",
            )
            yield Static(
                _scope_summary(self.state),
                id="library-rag-scope-summary",
            )
            for toggle in library_rag_scope_toggle_children(self.state):
                yield toggle
            # task-12 (spec §10.0/§10.1): the legacy-chunk report line --
            # "Chunked by an older engine: N items" -- lives HERE, on the
            # Library RAG surface ADR-003 names as the owner (not Settings).
            # Sits after the source toggles because it describes the state
            # of those sources' chunk data. Task 13's "Re-chunk older-engine
            # items" control joins it here (its own worker group + the
            # §10.3 mutual in-flight guard -- never this fetch's group).
            yield self._legacy_chunk_report_line()
            yield from self._rechunk_action_children()
            for child in library_rag_scope_recovery_children(self.state):
                yield child

        # PR-3 Task 3: the Answer region is its own sibling here, BETWEEN
        # source-scope and results -- never yielded inside the
        # `#library-rag-results` block below. That block's own teardown/
        # remount loop (`LibraryScreen._refresh_library_rag_results_widgets`)
        # only ever touches ITS OWN children (it skips
        # `LIBRARY_RAG_RESULTS_STATIC_WIDGET_IDS`, tears down the rest, and
        # remounts from `library_rag_results_body_children`); an answer
        # region mounted inside it would be destroyed on every results
        # refresh. `library_rag_answer_children` returns `[]` (nothing
        # yielded) outside rag mode and before any answer/in-flight state
        # exists, so the idle and keyword-mode canvases are unaffected.
        for child in library_rag_answer_children(self.state):
            yield child

        with Vertical(id="library-rag-results", classes="library-rag-region"):
            yield Static(
                results_heading_text(self.state),
                id="library-rag-results-heading",
                classes="destination-section",
            )
            for child in library_rag_results_body_children(self.state):
                yield child

        with Collapsible(
            title="Recent searches",
            collapsed=self.state.history_collapsed,
            id="library-rag-history",
        ):
            for child in library_rag_history_children(self.state):
                yield child


def _scope_summary(state: LibraryRagPanelState) -> str:
    """Return the source scope line for the main Search/RAG work lane."""
    return library_rag_scope_summary(state.scope)


def scope_toggle_label(option: LibraryRagSourceOption) -> str:
    """Return a toggle Button's visible label for one scope source option.

    Public (RAG-27 fix-review): also imported by the screen's snapshot-
    driven in-place refresh (`LibraryScreen._sync_library_rag_scope_toggle_and_run_gate_widgets`)
    so a background ingest's fresh counts can update each toggle's ``(N)``
    suffix without going through `library_rag_scope_toggle_children`'s
    full Button rebuild (a mount/remove sequence unsafe to run
    concurrently with the other refresh callers -- see that method's
    docstring).
    """
    marker = "✓" if option.selected else "○"
    return f"{marker} {option.label} ({option.count})"


def library_rag_scope_toggle_children(state: LibraryRagPanelState) -> list[Widget]:
    """Return one full-width toggle `Button` per real source type (B2).

    Shared by the panel's own `compose()` and the screen's incremental
    refresh so both build identical toggles from the same state. Only
    `LIBRARY_RAG_SCOPE_TOGGLE_SOURCE_TYPES` (notes/media/conversations) get
    a toggle -- workspaces/collections have no retrieval seam of their own.

    Args:
        state: Current Library Search/RAG panel display state.

    Returns:
        One toggle `Button` per real source type, disabled when that
    source's count is 0. Capture search remains inside Collections.
    """
    return [
        Button(
            scope_toggle_label(option),
            id=f"library-rag-scope-toggle-{option.source_type}",
            classes="library-rag-scope-toggle",
            disabled=not option.available,
            tooltip=f"Toggle {option.label} in the retrieval scope.",
        )
        for option in state.scope.options
        if option.source_type in LIBRARY_RAG_SCOPE_TOGGLE_SOURCE_TYPES
    ]


def library_rag_scope_shows_recovery(scope: LibraryRagScopeState) -> bool:
    """True when the scope region should render its recovery dump + Import media.

    Only the genuinely-empty-library case (no sources available at all)
    gets the full recovery presentation; a user deselecting every scope
    toggle with sources still available is covered by the query region's
    quiet line instead (A1/B2) -- an Import media button would not fix
    that case.
    """
    return bool(scope.recovery_copy) and not scope.has_available_sources


def library_rag_scope_recovery_children(state: LibraryRagPanelState) -> list[Widget]:
    """Return the scope region's no-sources gate line + Import media button, or none.

    Shared by `compose()` and the screen's incremental refresh. The gate
    copy is `LIBRARY_RAG_NO_SOURCES_GATE_COPY` -- one quiet muted line, not
    the retired Unavailable/Why/Next/Recovery/Owner dump -- rendered with
    the same quiet-line styling as the query gate lines.
    """
    if not library_rag_scope_shows_recovery(state.scope):
        return []
    return [
        Static(
            state.scope.recovery_copy,
            id="library-rag-scope-recovery",
            classes="library-rag-quiet-line",
        ),
        Button(
            "Open Import media",
            id="library-rag-open-import-export",
            classes="library-rag-recovery-action",
            tooltip="Open Library Import media to add sources.",
        ),
    ]


#: Caution copy for a `ready` answer whose `citation_status` is not
#: `validated` but which carries no recovery copy of its own -- a state the
#: shipping validator never produces, kept because nothing enforces that
#: cross-module invariant here. Says only what is actually known (the
#: citations did not validate; check the evidence), never that they were
#: "verified".
LIBRARY_RAG_ANSWER_UNVALIDATED_CAUTION = (
    "This answer's citations did not validate against the staged evidence. "
    "Check the evidence rows below before relying on it."
)

#: Fallback for a `ready` answer whose escaped display text comes back empty
#: (fix-review I2). `library_rag_answer_display_text` returns `""` for
#: unsafe input -- its own docstring names an embedded `<script>` block as
#: the example, and that is plausible model output here: the model is
#: quoting HTML-ingested library evidence back at us. Without this
#: fallback, a `ready`/`validated` answer rendered an EMPTY headline
#: `Static` under "Citations resolve to staged evidence." -- a silent
#: omission wearing the panel's own trust note. Rendered in the same
#: caution register as `LIBRARY_RAG_ANSWER_UNVALIDATED_CAUTION`, not as the
#: headline answer text, because there is nothing safe to show as the
#: headline.
LIBRARY_RAG_ANSWER_UNSAFE_TEXT_FALLBACK = (
    "The answer could not be displayed safely — see the evidence rows below."
)


def library_rag_answer_children(state: LibraryRagPanelState) -> list[Widget]:
    """Return the Answer region (`Vertical#library-rag-answer`), or none (PR-3 Task 3).

    Rag mode's headline: the RAG Answer worker's grounded-answer outcome
    (Task 1's `LibraryRagAnswer`), rendered as its own region -- see the
    mount-point comment in `compose()` for why this must never be yielded
    inside `#library-rag-results`.

    Nothing renders in keyword (search) mode -- Task 1's answer service is
    never invoked there, and this checks `state.query_state.mode` directly
    rather than only `state.answer`, so a stale answer left over from a
    mode flip can never leak through either. Nothing renders before any
    query has run in rag mode either (`state.answer is None` and
    `state.retrieval_status != "answering"`): the idle canvas has nothing
    to say yet.

    Visual hierarchy (design call, PR-3 Task 3): the answer IS the headline
    of rag mode, so a clean `ready` answer renders as plain, unmuted
    "headline" text (`#library-rag-answer-text`, a raised card, styled in
    CSS) -- but `abstained`/`no_evidence`/`failed` are NOT errors, and
    render in the same `.library-rag-quiet-line` register this panel
    already uses elsewhere (RAG-29/33) for "nothing went wrong, there's
    just nothing to show". `failed` additionally gets a one-line retry
    HINT, not a bespoke retry Button: the Run button already re-triggers
    retrieval + answer generation on the next press (wired in PR-3 Task 4),
    so a second button doing the same thing would be redundant, not
    additive -- worse than no button at all.

    Carried ruling (Task 1 review): `status == "ready"` never renders as a
    clean answer without branching on `citation_status` first.
    `uncited`/`unverified` (in fact, anything other than the literal
    `"validated"`) shows `citation_recovery` -- or
    `LIBRARY_RAG_ANSWER_UNVALIDATED_CAUTION` when that copy is somehow
    empty -- in a bordered callout ABOVE the answer text: at least as
    prominent as the answer, not a footnote below it, since a
    plausible-looking wrong answer is the single most dangerous failure
    mode this feature exists to prevent. `validated` gets
    a neutral one-line note instead -- and that note, like every other
    string this module builds, never calls it "verified":
    `build_answer_citation_validation` only checks that a citation label
    RESOLVES to a staged reference, never that the cited snippet actually
    supports the claim.

    Fix-review (I2): a `ready` answer whose escaped display text comes back
    empty (`library_rag_answer_display_text` returns `""` for unsafe input,
    e.g. an embedded `<script>` block quoted from HTML-ingested evidence)
    renders `LIBRARY_RAG_ANSWER_UNSAFE_TEXT_FALLBACK` in the caution
    register instead of an empty headline `Static` -- and the citation note
    is suppressed in that case, `clean` or not, since there is nothing safe
    shown for it to vouch for.

    Paid-moment footer (PR-3 Task 3): every settled status except
    `no_evidence` (the one path where no provider call was ever attempted)
    gets a trailing `#library-rag-answer-provenance` line naming what the
    call actually cost -- provider, model, and either a real dollar figure
    or an honest "pricing unknown"/no-usage statement, built by
    `_answer_provenance_line` from Task 1's shared
    `Chat.cost_display.build_provenance_line` and Task 2's
    `answer.provider`/`model`/`usage`. It renders for `failed` too whenever
    a real, billable response was already parsed before the failure (Task
    2's fix round) -- a call that cost money says so even though it did
    not produce a usable answer.

    Args:
        state: Current Library Search/RAG panel display state.

    Returns:
        `[]` outside rag mode or when there is nothing to show yet;
        otherwise a single-element list holding the answer region.
    """
    if state.query_state.mode != "rag":
        return []

    heading = Static(
        "Answer", id="library-rag-answer-heading", classes="destination-section"
    )

    if state.retrieval_status == "answering":
        # PR-3 Task 3: the one moment this region has something true to say
        # about cost BEFORE the outcome (and its token usage) is even
        # known -- which provider is about to be billed. Names it whenever
        # the screen resolved one (`state.in_flight_answer_provider`);
        # falls back to the pre-existing generic line when it did not
        # (every state built before this field existed, and the "answering"
        # override reached without ever resolving a provider -- unreachable
        # through the UI, since `_start_library_rag_answer` only raises
        # this status after `resolve_library_rag_answer_provider` already
        # returned one, but still a safe default for a direct `from_values`
        # call like this module's own gate16 tests use).
        in_flight_text = (
            f"Asking {state.in_flight_answer_provider}…"
            if state.in_flight_answer_provider
            else "Generating answer…"
        )
        return [
            Vertical(
                heading,
                Static(
                    in_flight_text,
                    id="library-rag-answer-status",
                    classes="library-rag-quiet-line",
                    # The provider name is config-sourced (`default_api_
                    # endpoint`), so this line interpolates a value the
                    # user controls -- render it literally rather than as
                    # Rich markup (PR-T2 review round 3, minor 1).
                    markup=False,
                ),
                id="library-rag-answer",
                classes="library-rag-region",
            )
        ]

    answer = state.answer
    if answer is None:
        return []

    body: list[Widget] = [heading]
    if answer.status == ANSWER_STATUS_READY:
        clean = answer.citation_status == "validated"
        if not clean:
            # `citation_status` and `citation_recovery` are set together by
            # `build_answer_citation_validation` -- but that invariant lives
            # in another module and nothing enforces it at the dataclass
            # level (PR-3 Task 3 review finding). Falling back to generic
            # caution copy, rather than skipping the callout when recovery
            # copy happens to be empty, keeps the branch's whole point
            # intact: an answer whose citations did not validate is NEVER
            # rendered as if they had.
            body.append(
                Static(
                    library_rag_answer_display_text(answer.citation_recovery)
                    or LIBRARY_RAG_ANSWER_UNVALIDATED_CAUTION,
                    id="library-rag-answer-caution",
                    classes="library-rag-callout is-caution",
                )
            )
        display_text = library_rag_answer_display_text(answer.text)
        if display_text.strip():
            body.append(
                Static(
                    display_text,
                    id="library-rag-answer-text",
                )
            )
            if clean:
                body.append(
                    Static(
                        "Citations resolve to staged evidence.",
                        id="library-rag-answer-citation-note",
                        classes="library-rag-quiet-line",
                    )
                )
        else:
            # Fix-review (I2): `library_rag_answer_display_text` returns
            # `""` for unsafe input -- a `ready` answer must never render
            # as a blank headline sitting under the "Citations resolve to
            # staged evidence." trust note (a silent omission presented as
            # trustworthy). This fallback ALWAYS wins over the citation
            # note, `clean` or not: there is nothing safe to show as the
            # answer either way.
            body.append(
                Static(
                    LIBRARY_RAG_ANSWER_UNSAFE_TEXT_FALLBACK,
                    id="library-rag-answer-unsafe",
                    classes="library-rag-callout is-caution",
                )
            )
    elif answer.status in (ANSWER_STATUS_ABSTAINED, ANSWER_STATUS_NO_EVIDENCE):
        body.append(
            Static(
                library_rag_answer_display_text(answer.text),
                id="library-rag-answer-text",
                classes="library-rag-quiet-line",
            )
        )
    elif answer.status == ANSWER_STATUS_FAILED:
        error_text = (
            library_rag_answer_display_text(answer.error)
            or "The answer could not be generated."
        )
        body.append(
            Static(
                f"Answer failed: {error_text}",
                id="library-rag-answer-error",
                classes="library-rag-quiet-line",
            )
        )
        body.append(
            Static(
                "Run the query again to retry.",
                id="library-rag-answer-retry-hint",
                classes="library-rag-quiet-line",
            )
        )
    else:
        # An answer status this module does not recognize -- nothing
        # well-formed to show here is safer than guessing at a
        # presentation for it.
        return []

    provenance_line = _answer_provenance_line(answer)
    if provenance_line is not None:
        body.append(
            Static(
                provenance_line,
                id="library-rag-answer-provenance",
                classes="library-rag-quiet-line",
                markup=False,
            )
        )

    return [Vertical(*body, id="library-rag-answer", classes="library-rag-region")]


def _answer_provenance_line(answer: LibraryRagAnswer) -> str | None:
    """The footer's provenance line for a settled answer, or `None` (PR-3 Task 3).

    Priced at RENDER time from `pricing_catalog.get_pricing_catalog()` --
    never stored on `answer` itself, so a pricing-config change is reflected
    on the very next render rather than freezing whatever rate was live
    when the answer landed.

    `None` (no footer at all) in exactly one case:

    * `answer.provider == ""` -- the no-evidence path, the ONLY path where
      no provider call was ever attempted (Task 2's own contract). A line
      naming an empty provider would be worse than no line.

    A second, narrower case is suppressed too:

    * `answer.model == "" and answer.usage is None` -- an exception fired
      before `generate_library_rag_answer`'s containment `try` ever reached
      `_invoke_chat` (a bundle-build failure, or the provider call itself
      raising, e.g. a realistic upstream 503): `provider` is still set (a
      plain function parameter, always safe -- Task 2's fix-review comment),
      but nothing else is known -- no model, no usage, nothing was ever
      spent. There is nothing true to report about cost here, so this
      renderer says nothing at all rather than a maximally minimal
      provider-only line.

    A blank `model` with real `usage` -- e.g. a provider payload that omits
    its own `"model"` key (`test_a_missing_model_key_yields_an_empty_model_
    without_raising`) -- is NOT suppressed: money was spent, so this footer
    must still say so. `build_provenance_line` itself renders a blank
    `model` gracefully (fix-review: Task 1's header now joins only the
    non-empty identifiers, so a blank model is OMITTED, never left as a
    dangling `" · "` with nothing after it) -- fixed at the shared-module
    source rather than papered over here, since any future caller of that
    function could hit the same upstream shape.

    Every other combination (model known, usage known, or both) renders --
    including a `failed` status whose usage survived a post-call
    processing failure (Task 2's fix round): a call that cost real money
    must say so even though it ultimately failed.
    """
    if not answer.provider or (not answer.model and answer.usage is None):
        return None

    cost: Decimal | None = None
    pricing_known = False
    if answer.usage is not None:
        breakdown = get_pricing_catalog().cost_for_usage(answer.usage)
        if breakdown is not None:
            cost = Decimal(str(breakdown.total))
            pricing_known = True

    return build_provenance_line(
        provider=answer.provider,
        model=answer.model,
        usage=answer.usage,
        cost=cost,
        pricing_known=pricing_known,
    )


def _query_blocked_is_quiet(query_state: LibraryRagQueryState) -> bool:
    """True when the run gate's blocker renders as a single quiet line (A1)."""
    return query_state.blocked_is_empty_query or query_state.blocked_is_no_scope


def library_rag_query_shows_full_recovery(query_state: LibraryRagQueryState) -> bool:
    """True when the query region should render the callout + recovery dump.

    Reserved for real failures (unsafe query, missing dependencies/index, no
    provider for RAG mode) -- the empty-query and no-scope gates render a
    single quiet line instead (A1), and the ready/searching states render
    neither.
    """
    return bool(query_state.recovery_copy) and not _query_blocked_is_quiet(query_state)


def library_rag_query_quiet_text(state: LibraryRagPanelState) -> str:
    """Return the text of the query region's single reserved quiet row.

    Extracted from `library_rag_query_status_children` (F1) so the screen's
    NO-`await` snapshot sync can refresh that one row with a plain
    `Static.update()` without rebuilding the callout block around it. Both
    callers derive from the SAME `state` the run gate is derived from, so
    the row can never disagree with the Run button beside it -- Task 4's
    collapse of "is a paid call ready" into one source of truth
    (`ready_answer_provider`) survives having two render sites.

    Returns:
        The quiet line's copy: a gate's quiet blocker, the ready `rag`
        mode's paid-mode notice, or `""` for every state that reserves the
        row without filling it.
    """
    query_state = state.query_state
    if query_state.blocked_is_empty_query:
        return "Enter a question or search query."
    if query_state.blocked_is_no_scope and state.scope.has_available_sources:
        return "Select at least one source."
    if query_state.ready_answer_provider:
        return library_rag_paid_mode_notice(query_state.ready_answer_provider)
    return ""


def library_rag_query_status_children(state: LibraryRagPanelState) -> list[Widget]:
    """Return the query region's status widgets (A1/A2).

    Shared by `compose()` and the screen's incremental refresh. The quiet
    gate line is ALWAYS returned -- with empty text in the searching state
    and a fixed one-row height so the Run button below it never shifts
    vertically when a gate's copy appears or disappears (2026-07 UAT: the
    button jumped ~2 rows on valid input, breaking muscle memory). The
    no-scope gate stays quiet-but-empty when the Library has no sources at
    all: the scope region's single no-sources gate line + "Open Import
    media" action own that state, so a second "Select at least one
    source." line would just re-stack guidance. Real failures (unsafe
    query, missing dependencies/index, no provider) additionally render
    the callout + recovery-copy block.

    The READY state is no longer always empty (PR-T2 Task 4): `rag` mode
    with a provider actually configured fills this same reserved row with
    `library_rag_paid_mode_notice`, naming the provider Run would bill --
    until this task, the ONLY provider-adjacent copy on this whole panel
    was the *blocked* branch's "Select a provider/model..." text, which
    vanishes the instant a provider IS configured. `search` mode's ready
    state is untouched -- it never calls a provider, so the row keeps its
    original empty-and-reserved behavior there. The row's copy comes from
    `library_rag_query_quiet_text`, which the screen's no-`await` snapshot
    sync also calls to update the mounted row in place (F1).

    Args:
        state: Current Library Search/RAG panel display state.

    Returns:
        The quiet-line `Static` (always), plus the callout + recovery
        widgets for full-recovery failures.
    """
    query_state = state.query_state
    quiet_line = Static(
        library_rag_query_quiet_text(state),
        id="library-rag-query-quiet-line",
        classes="library-rag-quiet-line",
        markup=False,
    )
    quiet_line.styles.height = 1
    children: list[Widget] = [quiet_line]
    if library_rag_query_shows_full_recovery(query_state):
        reason = query_state.run_action.disabled_reason
        children.extend(
            (
                Static(
                    f"Blocked | {reason}",
                    id="library-rag-query-blocked-callout",
                    classes="library-rag-callout is-blocked",
                ),
                Static(query_state.recovery_copy, id="library-rag-query-recovery"),
            )
        )
    return children


def _mode_toggle_label(state: LibraryRagPanelState) -> str:
    """Return the visible mode-toggle button label.

    task-14902: a KEPT one-press toggle (a genuine two-state mode flip
    that resets retrieval state -- a choice strip would add a press to
    the most common action for zero information). AC#1 is satisfied at
    the label instead: both modes render with the ``✓`` marker on the
    active one, so the full option space is on screen and one press IS a
    direct pick of the only other mode.
    """
    from ...Library.library_shell_state import library_toggle_label

    return library_toggle_label(
        "mode",
        ("Search", "RAG Answer"),
        0 if state.query_state.mode == "search" else 1,
    )


def _other_mode_label(state: LibraryRagPanelState) -> str:
    """Return the label of the mode a toggle press would switch TO.

    The cycle only ever has two states (`rag`/`search`), so the "other"
    mode is simply whichever one isn't current -- see `_mode_toggle_tooltip`.
    """
    return "Search" if state.query_state.mode == "rag" else "RAG Answer"


def _mode_toggle_tooltip(state: LibraryRagPanelState) -> str:
    """Return the mode-cycle button's tooltip, naming the next mode (RAG-39).

    A bare "Cycle Search/RAG mode." tooltip gives no hint how many modes
    exist or what a press does -- a two-state cycle looks identical to a
    five-state one. Naming the next mode makes the button's effect legible
    before the user presses it, and stays honest across a mode flip because
    it reads `state.query_state.mode` fresh on every build (recompose is
    the only path that rebuilds this button -- see the mode-toggle
    `Button.Pressed` handler in `library_screen.py`).

    PR-T2 Task 4: also names which side of the toggle spends money, in the
    tooltip's own compact register -- unconditionally, since "RAG Answer
    calls a provider" and "Search stays local" are properties of the MODE
    itself, true whether or not a provider happens to be configured right
    now (the quiet line's `library_rag_paid_mode_notice` is the one that
    additionally names the actual provider, and only once one is ready).
    """
    other = _other_mode_label(state)
    fact = "calls a paid provider" if other == "RAG Answer" else "stays local"
    return f"Cycle Search/RAG mode. Next: {other} — {fact}."


def results_heading_text(state: LibraryRagPanelState) -> str:
    """Return the Evidence region heading, surfacing top-k (A3).

    Public (Task 8): shared by the panel's own `compose()` and the screen's
    incremental refresh (`_refresh_library_rag_results_widgets`), mirroring
    every other body/heading builder in this module.

    "Per source" is only true for keyword mode: `_search_keyword` fans out
    one query per selected source and caps each independently at `top_k`.
    Rag mode's semantic leg is ONE store query (or one per allowlisted
    source type under an active scope, still merged by score) trimmed to a
    single `top_k` overall -- so the suffix is dropped there rather than
    making a claim that live UAT showed was false (RAG-29/scout item 3).
    """
    suffix = "" if state.query_state.mode == "rag" else " per source"
    return f"Evidence · top {state.query_state.top_k}{suffix}"


def library_rag_coverage_note_children(state: LibraryRagPanelState) -> list[Widget]:
    """Return the Evidence region's semantic coverage-note `Static`, or none.

    Shared by `compose()` and the screen's incremental refresh (folded into
    `library_rag_results_body_children` below, so both paths get it for
    free). Reuses the existing `library-rag-quiet-line` styling -- no new
    CSS. Empty (`[]`) whenever `state.coverage_note` has nothing to say
    (everything the query's semantic leg was asked to cover came back
    covered, and no result banded weak) -- see `library_rag_coverage_note`.
    """
    if not state.coverage_note:
        return []
    return [
        Static(
            state.coverage_note,
            id="library-rag-coverage-note",
            classes="library-rag-quiet-line",
        )
    ]


def library_rag_result_row_children(
    row: LibraryRagResultRow,
    index: int,
    selected_result_id: str,
) -> list[Widget]:
    """Return one evidence row as a single focusable card (C1/Task 12).

    Shared by the panel's own `compose()` and the screen's incremental DOM
    refresh (`_refresh_library_rag_results_widgets`) so both build identical
    rows from the same state.

    RAG-36 (live UAT, keyboard-only persona Sam): evidence rows used to be a
    flat list of sibling Statics plus a per-row `Horizontal` of buttons,
    mounted directly into the results container -- Tab only ever reached
    the buttons, with no row-level cursor and nothing indicating which row
    keyboard focus was "on". Every row's children are now wrapped in one
    `Vertical` card (`.library-rag-result-card`, `#library-rag-result-card-
    {index}`) that is itself a Tab stop; `LibraryScreen`'s Enter/`o`
    handlers resolve this card's index the same way the button handlers do
    (`_trailing_index` on the id) and call the exact same underlying
    selection/open methods -- no duplicated logic.

    Args:
        row: The evidence row to render.
        index: The row's position among the currently rendered results,
            used to build stable per-row widget ids.
        selected_result_id: The panel's currently selected result id, if any.

    Returns:
        A single-element list holding the row's card: title -> badges ->
        snippet -> citations (when present) -> an action row with Open
        first (primary emphasis, when the row is openable) then Select
        evidence.
    """
    selected = row.result_id == selected_result_id
    score = library_rag_score_suffix(
        row.score,
        score_kind=row.score_kind,
        vector_score=row.vector_score,
    )
    card_children: list[Widget] = [
        Static(
            f"{index + 1}. {row.title}{score}",
            id=f"library-rag-result-{index}",
            classes=(
                "library-rag-result-row is-selected"
                if selected
                else "library-rag-result-row"
            ),
        ),
        Static(
            row.row_badge_label,
            id=f"library-rag-result-badges-{index}",
            classes="library-rag-result-badges",
        ),
        Static(
            row.display_snippet,
            id=f"library-rag-result-snippet-{index}",
            classes="library-rag-result-snippet",
        ),
    ]
    if row.citation_labels:
        card_children.append(
            Static(
                f"Citations: {', '.join(row.citation_labels)}",
                id=f"library-rag-result-citations-{index}",
            )
        )
    actions: list[Widget] = []
    if row.can_open:
        actions.append(
            Button(
                "Open",
                id=f"library-rag-open-result-{index}",
                classes="library-rag-result-open console-action-primary",
                tooltip="Open this result's source in its Library editor/viewer.",
            )
        )
    actions.append(
        Button(
            "Selected evidence" if selected else "Select evidence",
            id=f"library-rag-select-result-{index}",
            classes="library-rag-result-action",
            tooltip="Select this evidence result for Console handoff.",
        )
    )
    card_children.append(Horizontal(*actions, classes="library-rag-result-actions"))
    card = Vertical(
        *card_children,
        id=f"library-rag-result-card-{index}",
        classes="library-rag-result-card",
    )
    # `Vertical.__init__` has no `can_focus` kwarg (only
    # `VerticalScroll`/`ScrollableContainer` accept it) -- set the instance
    # attribute directly, the same idiom already used elsewhere in this
    # screen (e.g. `left_rail.can_focus = True` in `library_screen.py`).
    card.can_focus = True
    return [card]


def library_rag_results_body_children(state: LibraryRagPanelState) -> list[Widget]:
    """Return the Evidence region's body widgets below the heading.

    Shared by `compose()` and the screen's incremental refresh
    (`_refresh_library_rag_results_widgets`) so both render identically:
    exactly one of evidence rows (plus a per-row Console handoff button on
    the selected row), the in-flight searching line, explicit retrieval
    recovery copy, or empty-state guidance, depending on retrieval status
    and result count.

    The `retrieval_status == "empty"` case (RAG-33/Task 11: a routine
    "your library has nothing matching this query" search) renders the
    quiet two-line `library_rag_empty_state_quiet_copy` instead of
    `state.recovery_copy`'s full Unavailable/Why/Next/Recovery/Owner
    dump -- that dump is reserved for real failures (`"blocked"`/
    `"failed"`: missing dependencies, empty index, provider unavailable,
    policy denial), which still render it verbatim because the user
    genuinely has to act on infrastructure there. Both branches keep
    `state.recovery_selector` as the rendered `Static`'s id, so existing
    selectors (`#library-rag-empty-state`, `#library-rag-service-error`)
    are unaffected.

    Args:
        state: Current Library Search/RAG panel display state.

    Returns:
        The widgets to mount directly below the Evidence heading.
    """
    # Task 8: the coverage note, when there is one, sits ahead of the row
    # list -- directly under the heading on every branch except the results
    # branch below, where task-2859's "N results for 'query'." headline
    # comes first. It is prepended to EVERY branch: `state.coverage_note`
    # used to be non-empty only alongside `state.results`, but a routing
    # disclosure (RAG-port P0: "this profile ran keyword-only", "no keyword
    # leg for the selected sources") survives the zero-row outcome, and zero
    # rows is exactly when it is most diagnostic -- the quiet no-match line
    # otherwise reads as a verdict on an index the search never queried.
    # Branches whose state has nothing to disclose prepend an empty list,
    # i.e. are unchanged. (The scope divert that used to be the second
    # example here retired with TASK-15020/B1: a scoped hybrid search now
    # runs hybrid, so nothing can emit that disclosure.)
    note_children: list[Widget] = list(library_rag_coverage_note_children(state))
    if state.results:
        # task-2859 item 10: "N results for 'query'" headline -- the
        # Evidence region used to jump straight from the mode/top-k
        # heading into the row cards with no line naming how many actually
        # landed or what query produced them.
        children: list[Widget] = []
        if state.results_count_line:
            # NOT markup=False: `results_count_line` is already
            # `escape_markup`-escaped (matching `coverage_note`/the empty-
            # state quiet copy below) -- disabling markup parsing here
            # would show the escape backslashes verbatim instead of
            # un-escaping them back to literal brackets.
            children.append(
                Static(
                    state.results_count_line,
                    id="library-rag-results-count-line",
                    classes="library-rag-quiet-line",
                )
            )
        # (rebase note) Reuse the already-computed `note_children` rather
        # than recomputing `library_rag_coverage_note_children(state)` --
        # dev's variable and the branch's headline are both real; only the
        # redundant recomputation was dropped.
        children.extend(note_children)
        for index, result in enumerate(state.results):
            children.extend(
                library_rag_result_row_children(result, index, state.selected_result_id)
            )
            if result.result_id == state.selected_result_id:
                children.append(
                    Button(
                        state.use_in_console_action.label,
                        id="library-rag-use-selected-in-console",
                        classes=(
                            "library-rag-console-action "
                            "library-rag-center-console-action"
                        ),
                        disabled=not state.use_in_console_action.enabled,
                        tooltip=state.use_in_console_action.tooltip,
                    )
                )
        return children
    if state.retrieval_status == "searching":
        return note_children + [
            Static(
                searching_status_line(state.scope.selected_source_types),
                id="library-rag-searching-line",
            )
        ]
    if state.recovery_copy and state.recovery_selector:
        if state.retrieval_status == "empty":
            return note_children + [
                Static(
                    # `state.searched_query`, NOT `state.query_state.query`
                    # (task-15 finding I3): the latter is live, not-yet-
                    # submitted input text that keeps moving after this
                    # "empty" outcome landed (in-panel edits, the rail
                    # search box, a scope toggle) -- this line must quote
                    # the query that actually produced the outcome it
                    # explains, not whatever is sitting in a box right now.
                    library_rag_empty_state_quiet_copy(
                        state.searched_query, state.scope
                    ),
                    id=state.recovery_selector,
                    classes="library-rag-quiet-line",
                )
            ]
        return note_children + [
            Static(state.recovery_copy, id=state.recovery_selector)
        ]
    if not state.scope.has_available_sources:
        # No Library sources at all: the scope region's single quiet gate
        # line + "Open Import media" action are the entire guidance for
        # this state -- repeating "No evidence yet"/"Add or import
        # sources…" here would re-stack the layered dump the quiet-gate
        # principle retired (2026-07 UAT).
        return note_children
    return note_children + [
        Static(
            "No evidence yet. Run Search/RAG to populate results.",
            id="library-rag-results-empty",
        ),
        Static(
            _evidence_empty_guidance(),
            id="library-rag-evidence-empty-guidance",
            classes="library-rag-empty-guidance",
        ),
    ]


def library_rag_history_children(state: LibraryRagPanelState) -> list[Widget]:
    """Return the `Recent searches` collapsible's child widgets (D1).

    Shared by the widget's own `compose` and the screen's incremental
    DOM refresh so both build identical rows from the same state.

    Args:
        state: Current Library Search/RAG panel display state.

    Returns:
        When history is empty, a single muted placeholder `Static`.
        Otherwise: a muted hint `Static` first, then one full-width
        `Button` per history entry (most recent first), then a
        `Clear history` `Button` last.
    """
    if not state.history:
        return [
            Static(
                "No recent searches.",
                id="library-rag-history-empty",
                classes="library-rag-history-empty",
            )
        ]
    children: list[Widget] = [
        Static(
            "Select an entry to run it again.",
            id="library-rag-history-hint",
            classes="library-rag-history-hint",
        )
    ]
    children.extend(
        Button(
            # Textual parses a plain string Button label as markup: an
            # unescaped stored entry like "docs [/archive] cleanup" raises
            # MarkupError at construction time -- and because history is
            # persisted before this rebuild, the crash would recur on every
            # Search-canvas entry after restart. Escaping mirrors the
            # `_sanitize_display_text(escape=True)` path result titles and
            # snippets already use.
            escape_markup(entry),
            id=f"library-rag-history-{index}",
            classes="library-rag-history-row",
            # RAG-38: history entries are bare strings -- no mode was
            # recorded when they ran -- so clicking one always re-runs
            # under the CURRENT mode, not necessarily the one it first ran
            # under. The tooltip says so honestly instead of implying an
            # exact replay, and stays truthful across a mode flip because
            # it reads `state.query_state.mode_label` fresh on every build.
            tooltip=(
                f"Re-runs under the current mode "
                f"({state.query_state.mode_label})."
            ),
        )
        for index, entry in enumerate(state.history)
    )
    children.append(
        Button(
            "Clear history",
            id="library-rag-history-clear",
            classes="library-rag-history-clear",
        )
    )
    return children


def _evidence_empty_guidance() -> str:
    """Return empty evidence workflow guidance."""
    return "Add or import sources, run a query, then select evidence for Console."


def _query_region_classes(state: LibraryRagPanelState) -> str:
    """Return query-region classes that reserve recovery height only when needed."""
    return (
        "library-rag-region has-recovery"
        if library_rag_query_shows_full_recovery(state.query_state)
        else "library-rag-region"
    )


def _scope_region_classes(state: LibraryRagPanelState) -> str:
    """Return source-scope classes that keep the ready state compact."""
    return (
        "library-rag-region has-recovery"
        if library_rag_scope_shows_recovery(state.scope)
        else "library-rag-region"
    )
