"""Media analysis orchestration shared by Reader, bulk selection and Import.

Moved bodies preserve their original names and control flow. Framework services
are live screen properties; sibling operations and shared shell fields are
named late-bound constructor ports. The app identity is fixed for this Library
screen's lifetime, so app_instance is the deliberate stable-identity exception.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from typing import Any

from loguru import logger
from textual.css.query import NoMatches, QueryError
from textual.widgets import Button, TextArea

from ...Library.ingest_analysis import (
    analysis_unavailable_reason,
    resolve_ingest_analysis_provider,
)
from ...Library.library_media_viewer_state import detail_analysis_text
from .canvas_sync import _sync_library_canvas

_ANALYZE_SELECTED_WORKER_GROUP = "library_media_analyze_selected"
_ANALYZE_ORIGIN_MEDIA = "media"
_ANALYZE_ORIGIN_IMPORT = "import"
_ANALYZE_ITEM_FAILED_REASON = "analysis did not persist"
_ANALYZE_AUTO_SKIP_REASON = "already analyzed"


class LibraryMediaAnalysisController:
    """Own analysis receipts and Reader analysis state, with explicit shell ports."""

    def __init__(
        self,
        *,
        screen: Any,
        app_instance: Any,
        build_library_media_state: Callable[..., Any],
        dispatch_library_media_analysis: Callable[..., Any],
        exit_library_media_select_mode: Callable[..., Any],
        library_canvas_projection_depth: Callable[..., Any],
        library_canvas_resync_pending: Callable[..., Any],
        set_library_canvas_resync_pending: Callable[[Any], None],
        library_media_analysis_provider_reason: Callable[..., Any],
        library_media_analyze_running: Callable[..., Any],
        set_library_media_analyze_running: Callable[[Any], None],
        library_media_backing_id: Callable[..., Any],
        library_media_bulk_delete_in_flight: Callable[..., Any],
        library_media_canvas_presentation: Callable[..., Any],
        library_media_detail: Callable[..., Any],
        library_media_select_mode: Callable[..., Any],
        refresh_library_media_detail: Callable[..., Any],
        run_library_service_call: Callable[..., Any],
        sanitize_media_field: Callable[..., Any],
        selected_media_id: Callable[..., Any],
        set_selected_media_id: Callable[[Any], None],
        sync_library_media_viewer_or_recompose: Callable[..., Any],
        update_library_ingest_dynamic_regions: Callable[..., Any],
    ) -> None:
        self._screen = screen
        self.app_instance = app_instance
        self._build_library_media_state = build_library_media_state
        self._get_dispatch_library_media_analysis = dispatch_library_media_analysis
        self._exit_library_media_select_mode = exit_library_media_select_mode
        self._get_library_canvas_projection_depth = library_canvas_projection_depth
        self._get_library_canvas_resync_pending = library_canvas_resync_pending
        self._set_library_canvas_resync_pending = set_library_canvas_resync_pending
        self._library_media_analysis_provider_reason = (
            library_media_analysis_provider_reason
        )
        self._get_library_media_analyze_running = library_media_analyze_running
        self._set_library_media_analyze_running = set_library_media_analyze_running
        self._library_media_backing_id = library_media_backing_id
        self._get_library_media_bulk_delete_in_flight = (
            library_media_bulk_delete_in_flight
        )
        self._library_media_canvas_presentation = library_media_canvas_presentation
        self._get_library_media_detail = library_media_detail
        self._get_library_media_select_mode = library_media_select_mode
        self._refresh_library_media_detail = refresh_library_media_detail
        self._run_library_service_call = run_library_service_call
        self._sanitize_media_field = sanitize_media_field
        self._get_selected_media_id = selected_media_id
        self._set_selected_media_id = set_selected_media_id
        self._sync_library_media_viewer_or_recompose = (
            sync_library_media_viewer_or_recompose
        )
        self._update_library_ingest_dynamic_regions = (
            update_library_ingest_dynamic_regions
        )
        self._library_media_analyze_total: int = 0
        self._library_media_analyze_done: int = 0
        self._library_media_analyze_failed_ids: tuple[str, ...] = ()
        self._library_media_analyze_choice: (
            tuple[tuple[str, ...], tuple[str, ...]] | None
        ) = None
        self._library_media_analyze_reason_cache: str | None = None
        self._library_media_analyze_origin: str = _ANALYZE_ORIGIN_MEDIA
        self._library_media_editing_analysis: bool = False
        self._library_media_generating_analysis: bool = False

    @property
    def app(self) -> Any:
        return self._screen.app

    @property
    def call_after_refresh(self) -> Any:
        return self._screen.call_after_refresh

    @property
    def is_running(self) -> Any:
        return self._screen.is_running

    @property
    def query_one(self) -> Any:
        return self._screen.query_one

    @property
    def refresh(self) -> Any:
        return self._screen.refresh

    @property
    def run_worker(self) -> Any:
        return self._screen.run_worker

    @property
    def _dispatch_library_media_analysis(self) -> Any:
        return self._get_dispatch_library_media_analysis()

    @property
    def _library_canvas_projection_depth(self) -> Any:
        return self._get_library_canvas_projection_depth()

    @property
    def _library_canvas_resync_pending(self) -> Any:
        return self._get_library_canvas_resync_pending()

    @_library_canvas_resync_pending.setter
    def _library_canvas_resync_pending(self, value: Any) -> None:
        self._set_library_canvas_resync_pending(value)

    @property
    def _library_media_analyze_running(self) -> Any:
        return self._get_library_media_analyze_running()

    @_library_media_analyze_running.setter
    def _library_media_analyze_running(self, value: Any) -> None:
        self._set_library_media_analyze_running(value)

    @property
    def _library_media_bulk_delete_in_flight(self) -> Any:
        return self._get_library_media_bulk_delete_in_flight()

    @property
    def _library_media_detail(self) -> Any:
        return self._get_library_media_detail()

    @property
    def _library_media_select_mode(self) -> Any:
        return self._get_library_media_select_mode()

    @property
    def _selected_media_id(self) -> Any:
        return self._get_selected_media_id()

    @_selected_media_id.setter
    def _selected_media_id(self, value: Any) -> None:
        self._set_selected_media_id(value)

    def _library_media_analyze_receipt_fields(self) -> dict[str, Any]:
        """Canvas inputs for the bulk-Analyze receipt (task-28007 AC#3/AC#4).

        (final review, I-2) An Import-origin run ("Analyze N skipped")
        drives the SAME screen-owned counters, so without this guard its
        progress rendered as a Media receipt on a canvas the user never
        started it from -- and that receipt's "Retry failed" re-ran those
        ids as a MEDIA run, leaving the Import rows still saying
        "analysis failed" and still counting in ``N``. The Import surface
        reports itself, per row, with its own disabled action for
        progress; the Media canvas shows nothing for that run.
        """
        if (
            getattr(self, "_library_media_analyze_origin", _ANALYZE_ORIGIN_MEDIA)
            == _ANALYZE_ORIGIN_IMPORT
        ):
            return {
                "analyze_receipt_total": 0,
                "analyze_receipt_done": 0,
                "analyze_receipt_failed": 0,
                "analyze_receipt_running": False,
                "analyze_choice_count": 0,
            }
        choice = self._library_media_analyze_choice
        return {
            # On the armed-choice path the total IS the pressed selection
            # ("N of M already analyzed"); no run has set a total yet.
            "analyze_receipt_total": (
                len(choice[0])
                if choice is not None
                else self._library_media_analyze_total
            ),
            "analyze_receipt_done": self._library_media_analyze_done,
            "analyze_receipt_failed": len(self._library_media_analyze_failed_ids),
            "analyze_receipt_running": self._library_media_analyze_running,
            "analyze_choice_count": (
                0 if choice is None else len(choice[0]) - len(choice[1])
            ),
        }

    def _library_media_analyze_reason(self) -> str:
        """Why bulk Analyze is off, memoised for one select-mode session.

        task-28007 AC#4: the bulk action wears the same sentence the
        Reader's Generate does (AC#5), but this one is read on EVERY media
        canvas sync, and ``resolve_ingest_analysis_provider`` is not free
        (Anthropic ``claude_subscription`` readiness shells out to the
        macOS keychain behind a 5s TTL). So it is resolved once per
        select-mode entry and dropped on exit -- the gesture itself
        re-resolves, so a provider configured mid-session is never refused
        on a stale memo.

        Returns:
            The resolver's reason while select mode is on and no provider
            is ready; "" otherwise.
        """
        if not self._library_media_select_mode:
            self._library_media_analyze_reason_cache = None
            return ""
        if self._library_media_analyze_reason_cache is None:
            self._library_media_analyze_reason_cache = (
                self._library_media_analysis_provider_reason()
            )
        return self._library_media_analyze_reason_cache

    def handle_library_media_analysis_edit(self, event: Button.Pressed) -> None:
        """Enter analysis edit mode for the open Library media viewer.

        Args:
            event: Button press event emitted by the analysis section's
                "Edit analysis" action.
        """
        event.stop()
        self._library_media_editing_analysis = True
        self._sync_library_media_viewer_or_recompose()

    def handle_library_media_analysis_cancel(self, event: Button.Pressed) -> None:
        """Discard in-progress analysis edits and return to the read-only view.

        Args:
            event: Button press event emitted by the analysis edit form's
                "Cancel" action.
        """
        event.stop()
        self._library_media_editing_analysis = False
        self._sync_library_media_viewer_or_recompose()

    def handle_library_media_analysis_save(self, event: Button.Pressed) -> None:
        """Read the analysis edit TextArea and hand the write off to a worker.

        Reads the edited analysis text directly (before any recompose
        removes the TextArea) and the current document content from the
        loaded detail -- ``save_analysis_version`` requires both, since it
        creates a new ``DocumentVersions`` row carrying the (unchanged)
        content alongside the edited analysis. Mirrors how
        ``handle_library_media_edit_save`` reads its form inputs
        synchronously before deferring to a worker.

        Args:
            event: Button press event emitted by the analysis edit form's
                "Save" action.
        """
        event.stop()
        media_id = self._selected_media_id
        if not media_id:
            self._library_media_editing_analysis = False
            self.refresh(recompose=True)
            return
        try:
            analysis_content = self.query_one(
                "#library-media-analysis-edit-text", TextArea
            ).text
        except (NoMatches, QueryError):
            self._library_media_editing_analysis = False
            self.refresh(recompose=True)
            return
        # Validate/sanitize the user-entered analysis at the UI boundary
        # before it reaches the persistence service.
        analysis_content = self._sanitize_media_field(
            analysis_content, max_length=100000
        )
        detail = (
            self._library_media_detail
            if isinstance(self._library_media_detail, Mapping)
            else {}
        )
        content = str(detail.get("content") or "")
        self.run_worker(
            self._save_library_media_analysis(
                media_id,
                content=content,
                analysis_content=analysis_content,
            )
        )

    async def _save_library_media_analysis(
        self,
        media_id: str,
        *,
        content: str,
        analysis_content: str,
        viewer_owned: bool = True,
    ) -> bool:
        """Persist an analysis edit as a new document version, then re-fetch detail.

        Guards against a missing ``save_analysis_version`` service or a
        failed write by logging the failure and surfacing a quiet notice,
        but always re-fetches detail afterwards so the viewer never shows a
        stale/half-applied edit. Analysis (re)generation via an LLM is
        explicitly out of scope -- this only persists caller-supplied text.

        Args:
            media_id: The Library media item id being edited.
            content: The current document content, sent unchanged alongside
                the edited analysis (``save_analysis_version`` requires it).
            analysis_content: The edited analysis text to persist.
            viewer_owned: False when the caller is the task-28007 bulk run
                rather than the Reader. A bulk item must not clear the
                Reader's editing flag, must not raise one toast per item
                (its receipt counts the failure), and must not re-fetch a
                detail nobody is reading.

        Returns:
            True when the analysis actually persisted. The bulk run counts
            a failed save as a failed item -- this used to be swallowed.
        """
        service = getattr(self.app_instance, "media_reading_scope_service", None)
        save_analysis_version = getattr(service, "save_analysis_version", None)
        service_media_id = self._library_media_backing_id(media_id)
        saved = False
        if callable(save_analysis_version):
            try:
                await self._run_library_service_call(
                    save_analysis_version,
                    mode="local",
                    media_id=service_media_id,
                    content=content,
                    analysis_content=analysis_content,
                    isolate_in_worker=True,
                )
                saved = True
            except Exception:
                logger.opt(exception=True).warning(
                    f"Failed to save Library media analysis for {media_id!r}."
                )
                if viewer_owned:
                    self._notify_library_media_analysis_warning(
                        "Could not save analysis changes; showing the latest "
                        "saved version."
                    )
        elif viewer_owned:
            self._notify_library_media_analysis_warning(
                "Analysis editing is unavailable."
            )
        if viewer_owned:
            self._library_media_editing_analysis = False
        if viewer_owned or media_id == self._selected_media_id:
            # A bulk item nobody is reading needs no detail re-fetch (that
            # call pulls the whole document and then discards it for any id
            # that is not the open selection); the OPEN item still refreshes
            # so the Reader never shows a stale analysis.
            await self._refresh_library_media_detail(media_id)
        return saved

    def _notify_library_media_analysis_warning(self, message: str) -> None:
        """Surface a quiet warning notice for a failed analysis-edit save.

        Args:
            message: Human-readable warning text to notify with.
        """
        notify = getattr(self.app_instance, "notify", None)
        if callable(notify):
            notify(message, severity="warning")

    def handle_library_media_analysis_generate(self, event: Button.Pressed) -> None:
        """Generate an analysis for the open item via the configured provider.

        task-28006: resolves the provider through the same seam the ingest
        path uses (``resolve_ingest_analysis_provider``), so the promise
        made here and the ingest receipt can never disagree. Not-ready
        surfaces the resolver's own honest reason instead of dispatching;
        ready flips the section into its "Generating analysis…" state and
        hands the call to a worker.

        Args:
            event: Button press event from the Analysis section's Generate
                action.
        """
        event.stop()
        if self._library_media_generating_analysis:
            return
        media_id = self._selected_media_id
        if not media_id:
            return
        resolution = resolve_ingest_analysis_provider(self.app_instance.app_config)
        reason = analysis_unavailable_reason(resolution)
        if reason:
            # Belt and braces behind the disabled control (AC#5): same
            # sentence the button already wears, so a click that somehow
            # lands (a stale mount, a keyboard route) never contradicts it.
            self._notify_library_media_analysis_warning(reason)
            return
        detail = (
            self._library_media_detail
            if isinstance(self._library_media_detail, Mapping)
            else {}
        )
        content = str(detail.get("content") or "")
        if not content.strip():
            self._notify_library_media_analysis_warning(
                "This item has no content to analyze."
            )
            return
        self._library_media_generating_analysis = True
        self._sync_library_media_viewer_or_recompose()
        self.run_worker(
            self._generate_library_media_analysis(
                media_id, content=content, resolution=resolution
            ),
            group="library_media_analysis_generate",
        )

    async def _generate_library_media_analysis(
        self,
        media_id: str,
        *,
        content: str,
        resolution: Any,
        viewer_owned: bool = True,
    ) -> bool:
        """Dispatch the analysis LLM call off-thread, then persist the result.

        Always clears the generating flag and re-fetches detail so the
        viewer never sticks in the progress state. On any failure the flag
        is cleared and a quiet warning is surfaced; nothing is persisted.

        Args:
            media_id: The open Library media item id.
            content: The document content to analyze.
            resolution: The ready ``IngestAnalysisResolution`` describing the
                provider, credential, and sampling parameters.
            viewer_owned: False when the task-28007 bulk run is the caller.
                A bulk item must not clear the Reader's "Generating
                analysis…" state (a concurrent Reader generation owns it),
                must not recompose the Reader, and must not toast per item.

        Returns:
            True when an analysis was produced and handed to the save seam.
            task-28007 AC#4: the bulk run counts failures from this, since
            a provider that returns nothing raises nothing.
        """
        try:
            analysis_text = await asyncio.to_thread(
                self._dispatch_library_media_analysis, content, resolution
            )
        except Exception:
            logger.opt(exception=True).warning(
                f"Analysis generation failed for {media_id!r}."
            )
            analysis_text = ""
        analysis_text = (analysis_text or "").strip()
        if viewer_owned:
            self._library_media_generating_analysis = False
        if not analysis_text:
            if viewer_owned:
                self._notify_library_media_analysis_warning(
                    "Analysis generation returned nothing; the item is unchanged."
                )
                self._sync_library_media_viewer_or_recompose()
            return False
        return await self._save_library_media_analysis(
            media_id,
            content=content,
            analysis_content=analysis_text,
            viewer_owned=viewer_owned,
        )

    def handle_library_media_analyze_selected(self, event: Button.Pressed) -> None:
        """Analyze every selected media item in one run (task-28007 AC#4).

        The selection is snapshotted from the RENDERED rows, not from
        ``RowSelection.ids``: that is a frozenset and carries no order,
        while the run has to follow the browse order the user is looking
        at. ("Review selected" reaches the same order the other way round
        -- it hands the unordered ids to ``_review_selected_worker`` and
        re-derives their order in there -- but a run that reports
        "Analyzing 3 of 40" as it goes needs the order before it starts.)

        Args:
            event: The Select-mode "Analyze" bulk-action button press.
        """
        event.stop()
        if self._library_media_bulk_delete_in_flight:
            return
        rows = self._build_library_media_state().rows
        self._start_library_media_analyze(
            tuple(row.media_id for row in rows if row.checked), overwrite=False
        )

    def handle_library_media_analyze_skip(self, event: Button.Pressed) -> None:
        """Run the armed choice over the un-analyzed items only (AC#3).

        Args:
            event: The receipt row's "Skip them" press.
        """
        event.stop()
        choice = self._library_media_analyze_choice
        if choice is None:
            return
        if not choice[1]:
            # Every selected item already had one: skipping them all
            # leaves nothing to run, so retire the choice rather than
            # leaving a dead row armed.
            self._clear_library_media_analyze_receipt()
            _sync_library_canvas(self, "media")
            return
        # Already partitioned, and these ids have no analysis by
        # construction -- ``overwrite=True`` skips a second read pass, it
        # does not overwrite anything.
        self._start_library_media_analyze(choice[1], overwrite=True)

    def handle_library_media_analyze_overwrite(self, event: Button.Pressed) -> None:
        """Run the armed choice over every selected item (AC#3's explicit yes).

        Args:
            event: The receipt row's "Overwrite" press.
        """
        event.stop()
        choice = self._library_media_analyze_choice
        if choice is None:
            return
        self._start_library_media_analyze(choice[0], overwrite=True)

    def handle_library_media_analyze_retry(self, event: Button.Pressed) -> None:
        """Re-run only the items the last run failed on (AC#4).

        Args:
            event: The receipt row's "Retry failed" press.
        """
        event.stop()
        failed = self._library_media_analyze_failed_ids
        if not failed:
            return
        self._start_library_media_analyze(failed, overwrite=True)

    def handle_library_media_analyze_receipt_dismiss(
        self, event: Button.Pressed
    ) -> None:
        """Clear the bulk-Analyze receipt (or its armed choice).

        Args:
            event: The receipt row's "Dismiss" press.
        """
        event.stop()
        self._clear_library_media_analyze_receipt()
        _sync_library_canvas(self, "media")

    def _clear_library_media_analyze_receipt(self) -> None:
        """Return every bulk-Analyze receipt field to its default."""
        self._library_media_analyze_total = 0
        self._library_media_analyze_done = 0
        self._library_media_analyze_failed_ids = ()
        self._library_media_analyze_choice = None

    def _start_library_media_analyze(
        self,
        media_ids: tuple[str, ...],
        *,
        overwrite: bool,
        on_item_done: Callable[[str, bool, str], None] | None = None,
    ) -> None:
        """Refuse, or claim the run and hand it to the one worker group.

        Shared by the bulk gesture and the receipt's own Skip/Overwrite/
        Retry actions, so all four obey the same one-run-at-a-time rule and
        the same provider gate. Also the entry point for a run over an
        ARBITRARY id set (task-28007 AC#1: an import run's analysis-skipped
        rows) -- there is no second loop; every caller shares this one.

        Args:
            media_ids: Ids to analyze, already in browse order.
            overwrite: Whether items that already carry an analysis are
                included. True SKIPS the AC#3 partition entirely, so a
                caller passing True owns that gate: only pass it for an id
                set the user has already chosen (Overwrite), or one already
                known to carry no analysis (Skip them, Retry failed).
            on_item_done: Optional per-item hook, ``(media_id, ok, reason)``,
                called after each item's outcome is counted in the loop
                below. Lets a caller outside the Media canvas (the Import
                queue) learn per-item outcomes without a second loop of its
                own. NOT called for an id the AC#3 partition pass diverts
                into the armed Skip/Overwrite choice -- that id ran through
                neither branch, so there is no outcome to report yet. Its
                presence also selects which surface ``on_unmount``'s
                interrupted-run notice points back at (``self.
                _library_media_analyze_origin``, fix round 1 I-3).
        """
        if self._library_media_analyze_running:
            notify = getattr(self.app_instance, "notify", None)
            if callable(notify):
                notify("Analysis already running", severity="warning")
            return
        if not media_ids:
            return
        # Belt and braces behind the disabled bulk action, and a fresh read
        # rather than the select-mode memo: a provider configured since
        # entry must not be refused (mirrors the Reader's Generate guard).
        resolution = resolve_ingest_analysis_provider(self.app_instance.app_config)
        reason = analysis_unavailable_reason(resolution)
        if reason:
            self._notify_library_media_analysis_warning(reason)
            return
        self._library_media_analyze_reason_cache = None
        self._clear_library_media_analyze_receipt()
        if self._library_media_select_mode:
            # task-31233's precedent, including its canvas sync: a bulk
            # action that runs leaves select mode, and without the repaint
            # the checkbox toolbar stays on screen over an already-cleared
            # selection until the worker's first sync -- a whole partition
            # pass (one DB read per selected id) later. A caller outside
            # select mode (the Import queue's run) never enters it, so this
            # block is a no-op there -- verified by test, not special-cased.
            self._exit_library_media_select_mode(announce_discard=False)
            _sync_library_canvas(self, "media")
        self._library_media_analyze_running = True
        # (fix round 1, I-3) The unmount notice needs to know where to send
        # the user back to -- derived from ``on_item_done``'s presence, the
        # same signal C-1's branch above already uses to distinguish the
        # two origins.
        self._library_media_analyze_origin = (
            _ANALYZE_ORIGIN_IMPORT
            if on_item_done is not None
            else _ANALYZE_ORIGIN_MEDIA
        )
        self.run_worker(
            self._analyze_library_media_selection(
                media_ids,
                resolution=resolution,
                overwrite=overwrite,
                on_item_done=on_item_done,
            ),
            group=_ANALYZE_SELECTED_WORKER_GROUP,
            exclusive=True,
            exit_on_error=False,
        )

    async def _fetch_library_media_analysis_detail(
        self, media_id: str, *, include_content: bool
    ) -> Mapping[str, Any] | None:
        """Fetch one item's detail off the event loop for the bulk run.

        Deliberately NOT ``_refresh_library_media_detail``: that one owns
        the Reader's session state, and a bulk run must not move the
        Reader's selection forty times.

        Args:
            media_id: The canonical media id.
            include_content: Whether the document body is needed (the
                analyzed/not-analyzed pass does not need it).

        Returns:
            The detail mapping, or None when the service is unavailable or
            returned something else.
        """
        service = getattr(self.app_instance, "media_reading_scope_service", None)
        get_media_item = getattr(service, "get_media_item", None)
        if not callable(get_media_item):
            return None
        detail = await self._run_library_service_call(
            get_media_item,
            mode="local",
            media_id=self._library_media_backing_id(media_id),
            include_content=include_content,
            include_versions=True,
            isolate_in_worker=True,
        )
        return detail if isinstance(detail, Mapping) else None

    async def _analyze_library_media_selection(
        self,
        media_ids: tuple[str, ...],
        *,
        resolution: Any,
        overwrite: bool,
        on_item_done: Callable[[str, bool, str], None] | None = None,
    ) -> None:
        """Analyze each id in turn, updating the receipt after every item.

        One worker for the whole run (AC#4). A per-item failure -- a raise
        OR a provider that returned nothing -- is counted and the run
        continues; nothing aborts it. When ``overwrite`` is False and any
        selected item already carries an analysis: a Select-mode caller
        (``on_item_done is None``) gets NOTHING run -- the Skip/Overwrite
        choice is armed in the receipt instead (AC#3). An Import-run caller
        (``on_item_done`` given) has no such card to show, so it auto-skips
        the already-analyzed ids instead and says so (fix round 1, C-1) --
        see the docstring on the branch below.

        Args:
            media_ids: Ids to analyze, in browse order.
            resolution: The ready resolution the gesture already checked.
            overwrite: Whether analyzed items are included.
            on_item_done: See ``_start_library_media_analyze``. Its mere
                presence also selects the C-1 auto-skip behavior above.
        """
        try:
            if not overwrite:
                unanalyzed = await self._library_media_unanalyzed_ids(media_ids)
                if len(unanalyzed) != len(media_ids):
                    if on_item_done is None:
                        self._library_media_analyze_choice = (media_ids, unanalyzed)
                        return
                    # (fix round 1, C-1) The Import queue has no
                    # Skip/Overwrite card to arm -- doing so here left an
                    # Import-started run with NOTHING visible: no
                    # receipts, no notice, the button just re-enabled with
                    # its count unchanged. Auto-skip the already-analyzed
                    # ids and say so instead; if that leaves nothing to
                    # run, say THAT and stop -- still no silent no-op.
                    # (Qodo review round, PR #2400 #3) An id dropped here
                    # never entered the loop below, so without an outcome
                    # of its own it stayed counted by "Analyze N skipped"
                    # forever -- every later press just re-discovered it
                    # already analyzed and reported nothing left to run.
                    # Record it as resolved through the SAME hook so it
                    # drops out of the count exactly like a generated one.
                    still_skipped = set(unanalyzed)
                    if on_item_done is not None:
                        for media_id in media_ids:
                            if media_id not in still_skipped:
                                on_item_done(media_id, True, _ANALYZE_AUTO_SKIP_REASON)
                    notify = getattr(self.app_instance, "notify", None)
                    if not unanalyzed:
                        if callable(notify):
                            notify("Nothing left to analyze")
                        return
                    if callable(notify):
                        already_analyzed = len(media_ids) - len(unanalyzed)
                        notify(f"{already_analyzed} already analyzed · skipped")
                media_ids = unanalyzed
            self._library_media_analyze_total = len(media_ids)
            _sync_library_canvas(self, "media", allow_screen_fallback=False)
            for media_id in media_ids:
                exc_reason = ""
                try:
                    persisted = await self._analyze_one_library_media_item(
                        media_id, resolution=resolution
                    )
                except Exception as exc:
                    persisted = False
                    # (fix round 1, I-1) A raised exception carries a real,
                    # specific reason -- capture it rather than falling
                    # through to the generic catch-all, so the receipt says
                    # something an import row's own "analysis failed:
                    # <reason>" line would.
                    exc_reason = str(exc)
                if persisted:
                    self._library_media_analyze_done += 1
                else:
                    self._library_media_analyze_failed_ids += (media_id,)
                if on_item_done is not None:
                    on_item_done(
                        media_id,
                        persisted,
                        (
                            ""
                            if persisted
                            else (exc_reason or _ANALYZE_ITEM_FAILED_REASON)
                        ),
                    )
                # Progress only: if the user has left the media canvas
                # mid-run, a missing canvas must NOT escalate to a
                # whole-screen recompose once per item on whatever screen
                # they moved to. The settling sync below uses the SAME
                # no-fallback rule (task-28007 Task 3, N2 -- this comment
                # used to claim it "keeps the default", which the fix-round-1
                # change to that sync's own ``allow_screen_fallback=False``
                # contradicted).
                _sync_library_canvas(self, "media", allow_screen_fallback=False)
        finally:
            self._library_media_analyze_running = False
            # Same no-fallback rule as the progress syncs: this also runs
            # on the cancellation path, i.e. while the screen is being
            # unmounted, where a whole-screen recompose is both useless and
            # unsafe. A canvas composed later reads these fields anyway.
            _sync_library_canvas(self, "media", allow_screen_fallback=False)
            if on_item_done is not None:
                # task-28007 AC#1/AC#2: an Import-run caller's LAST
                # ``on_item_done`` fires from inside the loop above, before
                # this ``finally`` clears the in-flight flag -- without
                # this, the Import queue's own action would repaint as
                # still-running one frame too early and stay disabled until
                # some unrelated later tick. Guarded on ``on_item_done`` so
                # the Media-canvas-only callers (Select mode) never pay for
                # an Import-canvas sync that is a no-op for them anyway
                # (``_update_library_ingest_dynamic_regions`` itself skips
                # work off the Import canvas). ``allow_screen_fallback=False``
                # (fix round 1, I-2): this also runs on the cancellation
                # path, same as the ``_sync_library_canvas`` call two lines
                # up -- without it, an unmount landing here (canvas already
                # torn down, ``_library_selected_row_id`` still pointing at
                # Import) falls into a whole-screen recompose on a dying
                # screen.
                self._update_library_ingest_dynamic_regions(allow_screen_fallback=False)

    async def _analyze_one_library_media_item(
        self, media_id: str, *, resolution: Any
    ) -> bool:
        """Load one item's content off the loop, then generate its analysis.

        Args:
            media_id: The canonical media id to analyze.
            resolution: The ready resolution shared by the whole run.

        Returns:
            True when an analysis was produced and persisted.
        """
        detail = await self._fetch_library_media_analysis_detail(
            media_id, include_content=True
        )
        content = str(detail.get("content") or "") if detail is not None else ""
        if not content.strip():
            return False
        return await self._generate_library_media_analysis(
            media_id, content=content, resolution=resolution, viewer_owned=False
        )

    async def _library_media_unanalyzed_ids(
        self, media_ids: tuple[str, ...]
    ) -> tuple[str, ...]:
        """The subset with no analysis on their newest version (AC#3).

        Read WITHOUT content: only the newest ``DocumentVersions`` row's
        analysis text decides this (``detail_analysis_text``, the same rule
        the Reader's Analysis tab uses), and pulling every document's body
        just to answer it would be the expensive way to ask. An unreadable
        item counts as un-analyzed: the run attempts it and reports its
        own failure rather than silently skipping it.

        Args:
            media_ids: The ids the gesture snapshotted.

        Returns:
            Those ids, in the same order, minus the already-analyzed ones.
        """
        unanalyzed: list[str] = []
        for media_id in media_ids:
            try:
                detail = await self._fetch_library_media_analysis_detail(
                    media_id, include_content=False
                )
            except Exception:
                detail = None
            if detail is None or not detail_analysis_text(detail):
                unanalyzed.append(media_id)
        return tuple(unanalyzed)
