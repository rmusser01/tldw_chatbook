"""Media Reader search, progress persistence and read-later orchestration.

Owns transient search/progress/memo state. No subtree is composed here: the
existing media viewer owns its pixels. Framework operations live-read the
screen; sibling operations are named late-binding ports. app_instance has
stable identity for this screen's lifetime and is the sole snapshot exception.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping
from functools import partial
from typing import Any

from loguru import logger
from textual.css.query import NoMatches, QueryError
from textual.widgets import Button, Input
from textual.worker import Worker

from ...Library.library_media_viewer_state import (
    analysis_find_unavailable_reason,
    build_library_media_viewer_state,
    detail_analysis_text,
    find_content_matches,
)
from ...Widgets.Library.library_media_content import LibraryMediaContentBody


class LibraryMediaReaderController:
    """Coordinate reading interactions without owning a DOM subtree."""

    def __init__(
        self,
        *,
        screen: Any,
        app_instance: Any,
        library_media_backing_id: Callable[..., Any],
        library_media_detail: Callable[..., Any],
        library_media_editing_analysis: Callable[..., Any],
        library_media_generating_analysis: Callable[..., Any],
        library_media_reader_session: Callable[..., Any],
        library_media_view: Callable[..., Any],
        mounted_library_media_viewer: Callable[..., Any],
        refresh_library_media_detail: Callable[..., Any],
        run_library_service_call: Callable[..., Any],
        selected_media_id: Callable[..., Any],
        sync_library_media_viewer_or_recompose: Callable[..., Any],
    ) -> None:
        self._screen = screen
        self.app_instance = app_instance
        self._library_media_backing_id = library_media_backing_id
        self._get_library_media_detail = library_media_detail
        self._get_library_media_editing_analysis = library_media_editing_analysis
        self._get_library_media_generating_analysis = library_media_generating_analysis
        self._get_library_media_reader_session = library_media_reader_session
        self._get_library_media_view = library_media_view
        self._mounted_library_media_viewer = mounted_library_media_viewer
        self._refresh_library_media_detail = refresh_library_media_detail
        self._run_library_service_call = run_library_service_call
        self._get_selected_media_id = selected_media_id
        self._sync_library_media_viewer_or_recompose = (
            sync_library_media_viewer_or_recompose
        )
        self._library_media_content_query: str = ""
        self._library_media_content_match_index: int = 0
        self._library_media_find_open: bool = False
        self._library_media_find_focus_pending: bool = False
        self._library_media_content_match_memo: (
            tuple[Any, str, tuple[int, ...], str] | None
        ) = None
        self._library_media_content_mode: str = "raw"
        self._library_media_read_scroll_by_id: dict[str, tuple[int, int]] = {}
        self._library_media_progress_pending_writes: dict[
            str, tuple[int | str, tuple[int, int]]
        ] = {}
        self._library_media_progress_inflight_write: (
            tuple[str, int | str, tuple[int, int]] | None
        ) = None
        self._library_media_progress_persisted_offsets: dict[str, tuple[int, int]] = {}
        self._library_media_progress_write_worker: Worker | None = None
        self._library_media_viewer_state_memo_detail: Any = object()
        self._library_media_viewer_state_memo_states: dict[
            tuple[str, str, str, bool], Any
        ] = {}

    @property
    def call_after_refresh(self) -> Any:
        return self._screen.call_after_refresh

    @property
    def is_attached(self) -> Any:
        return self._screen.is_attached

    @property
    def notify(self) -> Any:
        return self._screen.notify

    @property
    def query_one(self) -> Any:
        return self._screen.query_one

    @property
    def run_worker(self) -> Any:
        return self._screen.run_worker

    @property
    def _library_media_detail(self) -> Any:
        return self._get_library_media_detail()

    @property
    def _library_media_editing_analysis(self) -> Any:
        return self._get_library_media_editing_analysis()

    @property
    def _library_media_generating_analysis(self) -> Any:
        return self._get_library_media_generating_analysis()

    @property
    def _library_media_reader_session(self) -> Any:
        return self._get_library_media_reader_session()

    @property
    def _library_media_view(self) -> Any:
        return self._get_library_media_view()

    @property
    def _selected_media_id(self) -> Any:
        return self._get_selected_media_id()

    def _library_media_viewer_state_cached(
        self,
        detail: Mapping[str, Any] | None,
        *,
        arrival_note: str = "",
        backend: str = "local",
        canonical_id: str = "",
        force_raw: bool = False,
    ):
        """Memoized ``build_library_media_viewer_state`` per detail arrival.

        task-22208: the raw builder performs at least one O(document) string
        copy per call, and it used to run 2+ times per viewer sync (display
        state + the console-representation clause of the unchanged compare)
        on EVERY interaction -- traversal step, mode switch, More toggle,
        Escape. This memo bounds that to once per (detail arrival x build
        parameters), and because a no-change sync gets back the SAME state
        object, the sync's unchanged test can short-circuit on identity
        before ever falling back to the structural (content-memcmp) compare.

        Memo key and invalidation:
        * the detail OBJECT, by identity -- the detail is only replaced
          wholesale by ``_refresh_library_media_detail``'s settle (a fresh
          dict per fetch) or cleared to None, never mutated in place, so a
          new arrival (including an edit's refetch) always misses and
          rebuilds; a None detail memoizes the empty state the same way;
        * ``arrival_note`` / ``backend`` / ``canonical_id`` / ``force_raw``
          -- the remaining builder inputs; the per-detail entry dict is
          reset whenever the detail identity changes, so it holds at most
          the couple of parameter combinations live for one arrival.

        Known consequence, accepted by design: the "Updated: <age>" relative
        label freezes for the lifetime of one detail arrival (the raw
        builder stamps it from ``now`` per call). Recomputing it per sync is
        what the memo exists to stop -- and under the task-21116 compare a
        ticked-over age label would otherwise force a FULL document
        recompose just to repaint one metadata line.

        Args:
            detail: The loaded detail mapping, or None for the empty state.
            arrival_note: One-shot context line (see the raw builder).
            backend: Provenance backend displayed by Reader Info.
            canonical_id: Stable backend-qualified id override.
            force_raw: Force ``is_markdown`` False (external/server details
                render raw); folded into the memo so the replace also
                happens once per arrival.

        Returns:
            The memoized immutable viewer state.
        """
        if self._library_media_viewer_state_memo_detail is not detail:
            self._library_media_viewer_state_memo_detail = detail
            self._library_media_viewer_state_memo_states = {}
        key = (arrival_note, backend, canonical_id, force_raw)
        states = self._library_media_viewer_state_memo_states
        state = states.get(key)
        if state is None:
            state = build_library_media_viewer_state(
                detail,
                arrival_note=arrival_note,
                backend=backend,
                canonical_id=canonical_id,
            )
            if force_raw:
                state = dataclasses.replace(state, is_markdown=False)
            states[key] = state
        return state

    def handle_library_media_content_search_submitted(
        self, event: Input.Submitted
    ) -> None:
        """Set the in-content search query and jump to the first match.

        Submitted (rather than Changed) is used deliberately, so the query
        only takes effect on Enter. The mounted viewer coordinates its focused
        search controls and persistent body without rebuilding the screen.

        Args:
            event: Input submit event emitted by the content search box.
        """
        event.stop()
        # Strip once at the source so the status count, the body highlighting,
        # and prev/next navigation all search the exact same needle.
        submitted = event.value.strip()
        if submitted == self._library_media_content_query:
            # task-28011: re-pressing Enter on the same query walks to the
            # next match (find-bar convention) instead of no-opping.
            self._advance_library_media_content_match(1)
            return
        self._library_media_content_query = submitted
        self._library_media_content_match_index = 0
        matches = self._library_media_content_matches()
        viewer = self._mounted_library_media_viewer()
        if viewer is None:
            return
        try:
            viewer.sync_query_state(
                query=submitted,
                matches=matches,
                match_index=0,
            )
        except (NoMatches, QueryError):
            return
        self.call_after_refresh(self._focus_library_media_content_search_input)
        # Bring the first match into view after Rich text wrapping/layout settles.
        if matches:
            self.call_after_refresh(
                self._scroll_library_media_content_to_line, matches[0]
            )

    def _reset_library_media_search_on_mode_change(self, new_mode: str) -> None:
        """Drop the in-item search when the Reader actually changes tab.

        task-28026: each Reader tab searches its own corpus (the analysis
        text vs the transcript), so a real mode transition clears the active
        query, its match index, and the one-slot match memo rather than
        carrying one tab's highlights -- and a possibly out-of-range match
        index -- onto the other tab's text. A no-op re-press of the current
        mode leaves the search intact.

        Args:
            new_mode: The Reader mode being switched to.
        """
        if new_mode != self._library_media_reader_session.mode:
            self._close_library_media_find()
            self._library_media_content_match_memo = None

    def _capture_library_media_loaded_progress(self) -> None:
        """Snapshot and queue persistence of the local content body's offset.

        TASK-22210: this fires on every traversal step and mode switch, so
        the write itself is deduplicated (an offset already durable or
        already queued is skipped) and coalesced (one serial drainer keeps
        only each item's latest value) instead of spawning one SQLite
        writer per call.
        """
        session = self._library_media_reader_session
        loaded_id = session.loaded_id
        # task-28026: only the Read tab's content body carries transcript
        # reading progress. The Analysis tab reuses the same
        # #library-media-viewer-content id, so a capture taken in any other
        # mode would persist an unrelated scroll offset as transcript progress.
        if (
            session.external_detail
            or session.mode != "read"
            or loaded_id is None
            or not loaded_id.startswith("local:media:")
        ):
            return
        try:
            body = self.query_one(
                "#library-media-viewer-content", LibraryMediaContentBody
            )
        except (NoMatches, QueryError):
            return
        content = body.scroller
        offset = (int(content.scroll_x), int(content.scroll_y))
        self._library_media_read_scroll_by_id[loaded_id] = offset
        service = getattr(self.app_instance, "media_reading_scope_service", None)
        update_progress = getattr(service, "update_reading_progress", None)
        if not callable(update_progress) or session.loaded_backing_id is None:
            return
        self._queue_library_media_progress_write(
            loaded_id, session.loaded_backing_id, offset
        )

    def _library_media_progress_write_is_current(
        self, canonical_id: str, offset: tuple[int, int]
    ) -> bool:
        """True when ``offset`` already matches the newest known write intent.

        Precedence mirrors write recency: a queued value supersedes the
        in-flight one, which supersedes the last durably persisted one --
        so an equal older value never masks a newer pending write.
        """
        pending = self._library_media_progress_pending_writes.get(canonical_id)
        if pending is not None:
            return pending[1] == offset
        inflight = self._library_media_progress_inflight_write
        if inflight is not None and inflight[0] == canonical_id:
            return inflight[2] == offset
        return (
            self._library_media_progress_persisted_offsets.get(canonical_id) == offset
        )

    def _queue_library_media_progress_write(
        self, canonical_id: str, backing_id: int | str, offset: tuple[int, int]
    ) -> None:
        """Coalesce progress writes to the latest per-item value, one drainer.

        Last-write-wins coalescing, NOT ``exclusive=True``: cancelling an
        in-flight ``to_thread`` writer leaves the durable outcome unknown
        (the abandoned thread may still commit after its successor -- the
        task-1541 lesson). Keying pending values per item means a slow
        drain never drops a different item's final position either.
        """
        if self._library_media_progress_write_is_current(canonical_id, offset):
            return
        self._library_media_progress_pending_writes[canonical_id] = (
            backing_id,
            offset,
        )
        worker = self._library_media_progress_write_worker
        if not self.is_attached or (worker is not None and not worker.is_finished):
            return
        self._library_media_progress_write_worker = self.run_worker(
            self._drain_library_media_progress_writes(),
            group="library_media_reading_progress",
        )

    async def _drain_library_media_progress_writes(self) -> None:
        """Serialize progress writes while retaining each item's latest value."""
        while self._library_media_progress_pending_writes:
            canonical_id, (backing_id, offset) = next(
                iter(self._library_media_progress_pending_writes.items())
            )
            del self._library_media_progress_pending_writes[canonical_id]
            self._library_media_progress_inflight_write = (
                canonical_id,
                backing_id,
                offset,
            )
            try:
                persisted = await self._write_library_media_loaded_progress(
                    backing_id, offset
                )
            finally:
                self._library_media_progress_inflight_write = None
            if persisted:
                self._library_media_progress_persisted_offsets[canonical_id] = offset

    async def _write_library_media_loaded_progress(
        self, backing_id: int | str | None, offset: tuple[int, int]
    ) -> bool:
        """Persist a scroll snapshot already fenced to one loaded identity.

        Returns:
            True only when the service call completed; a failed write stays
            un-recorded so a later capture of the same offset retries it.
        """
        if backing_id is None:
            return False
        service = getattr(self.app_instance, "media_reading_scope_service", None)
        update_progress = getattr(service, "update_reading_progress", None)
        if not callable(update_progress):
            return False
        try:
            await self._run_library_service_call(
                update_progress,
                mode="local",
                media_id=backing_id,
                progress_data={"scroll_x": offset[0], "scroll_y": offset[1]},
                isolate_in_worker=True,
            )
        except Exception:
            logger.warning("Failed to persist Library media reading progress.")
            return False
        return True

    def _restore_library_media_loaded_progress(self, expected_id: str) -> None:
        """Restore only while the expected local identity still owns Reader."""
        session = self._library_media_reader_session
        if session.external_detail or session.loaded_id != expected_id:
            return
        offset = self._library_media_read_scroll_by_id.get(expected_id)
        if offset is None:
            return
        try:
            body = self.query_one(
                "#library-media-viewer-content", LibraryMediaContentBody
            )
        except (NoMatches, QueryError):
            return
        body.scroller.scroll_to(x=offset[0], y=offset[1], animate=False, force=True)

    def _close_library_media_find(self) -> None:
        """Reset the content Find bar: collapsed, no query, first match.

        task-31237: one seam for every reset path (item open, viewer exit,
        rail switch, delete, mode change, Escape) so a stale ``find_open``
        can never hold the bar open across a context the query reset
        already abandoned.
        """
        self._library_media_find_open = False
        self._library_media_content_query = ""
        self._library_media_content_match_index = 0

    def _library_media_find_unavailable_reason(self) -> str:
        """Why Find cannot open on the current Reader tab, or "" (Qodo on #2378)."""
        detail = self._library_media_detail
        analysis = detail_analysis_text(detail) if isinstance(detail, Mapping) else ""
        return analysis_find_unavailable_reason(
            mode=self._library_media_reader_session.mode,
            analysis=analysis,
            generating=self._library_media_generating_analysis,
            editing=self._library_media_editing_analysis,
        )

    def _consume_library_media_find_focus(self) -> bool:
        """Return and clear the one-shot Find-gesture focus token (task-31269)."""
        pending = self._library_media_find_focus_pending
        self._library_media_find_focus_pending = False
        return pending

    def handle_library_media_reader_find(self, event: Button.Pressed) -> None:
        """Open (or close) the Find bar for the tab being read.

        Args:
            event: The Find button press.
        """
        event.stop()
        if self._library_media_find_open:
            # task-31269 AC4: Find is a toggle -- a second press closes the
            # bar (live: it did nothing while the bar was open).
            self._close_library_media_find()
            self._sync_library_media_viewer_or_recompose()
            return
        reason = self._library_media_find_unavailable_reason()
        if reason:
            # Qodo on #2378: nothing to mount on this tab -- never arm
            # find_open silently (the button is already disabled with the
            # same reason; this guards the action path).
            self.notify(reason, severity="warning")
            return
        # task-31269: Find searches the tab you are reading. The Analysis
        # tab's bar is gated exactly like Read's now, so Find no longer
        # jumps Analysis -> Read (task-28026's transition predates the
        # collapsed bar). A same-mode reset is a no-op by design.
        self._reset_library_media_search_on_mode_change(
            self._library_media_reader_session.mode
        )
        # task-31237: the bar is collapsed until this gesture opens it; the
        # token below is what lets its mount take focus -- once.
        self._library_media_find_open = True
        self._library_media_find_focus_pending = True
        self._sync_library_media_viewer_or_recompose()
        self.call_after_refresh(self._focus_library_media_content_search_input)

    def _focus_library_media_content_search_input(self, *, _retries: int = 4) -> None:
        """Focus the mounted content search box after controls synchronize.

        task-31237: the input now MOUNTS during the viewer's recompose
        (the bar is collapsed until Find opens it), and screen-level
        ``call_after_refresh`` callbacks can flush before the widget-level
        recompose lands its children -- a short bounded re-defer chain
        covers the mount latency instead of silently losing the focus.
        """
        try:
            self.query_one("#library-media-content-search", Input).focus()
        except (NoMatches, QueryError):
            if _retries > 0:
                self.call_after_refresh(
                    partial(
                        self._focus_library_media_content_search_input,
                        _retries=_retries - 1,
                    )
                )

    async def handle_library_media_content_mode_rendered(
        self, event: Button.Pressed
    ) -> None:
        """Switch the open media item's Content section to the Rendered (Markdown) view.

        Args:
            event: Button press event emitted by the toggle strip's
                "Rendered" action.
        """
        event.stop()
        await self._set_library_media_content_mode("rendered")

    async def handle_library_media_content_mode_raw(
        self, event: Button.Pressed
    ) -> None:
        """Switch the open media item's Content section to the Raw text view.

        Args:
            event: Button press event emitted by the toggle strip's "Raw"
                action.
        """
        event.stop()
        await self._set_library_media_content_mode("raw")

    async def _set_library_media_content_mode(self, mode: str) -> None:
        """Shared Rendered/Raw toggle: no-op when already in ``mode``.

        The persistent body lazily mounts each view once; no media write is
        involved because the viewer has no editable body while browsing.

        Args:
            mode: ``"rendered"`` or ``"raw"``.
        """
        if (
            self._library_media_view != "viewer"
            or self._library_media_content_mode == mode
        ):
            return
        self._library_media_content_mode = mode
        viewer = self._mounted_library_media_viewer()
        if viewer is None:
            return
        try:
            await viewer.sync_mode(mode)
        except (NoMatches, QueryError):
            return

    def handle_library_media_content_search_next(self, event: Button.Pressed) -> None:
        """Advance to the next in-content search match and scroll it into view.

        Args:
            event: Button press event emitted by the "Next" search action.
        """
        event.stop()
        self._advance_library_media_content_match(1)

    def handle_library_media_content_search_prev(self, event: Button.Pressed) -> None:
        """Return to the previous in-content search match and scroll it into view.

        Args:
            event: Button press event emitted by the "Prev" search action.
        """
        event.stop()
        self._advance_library_media_content_match(-1)

    def _library_media_content_matches(self) -> tuple[int, ...]:
        """Return the open item's matching line indexes for the active query.

        task-22209: this used to run per Prev/Next click -- a full content
        copy out of ``build_library_media_viewer_state`` plus a full
        ``find_content_matches`` scan -- for a document and a query that
        had not changed since the previous click. The result now lives in a
        one-slot memo.

        Memo key, and why each part is in it:

        * the DETAIL OBJECT, by identity -- it is the document. A detail is
          only ever replaced wholesale (a settling fetch builds a fresh
          dict) or cleared to None, never mutated in place, so a new
          arrival always misses. Arrow-key traversal swaps the document
          while a submitted query stays live (only a row *press* blanks the
          query), so this component is load-bearing, not defensive.
        * the QUERY -- the same document answers differently per needle.

        ``match_index`` is deliberately NOT in the key: navigation moves
        the index over a match list that does not change.

        Returns:
            Ascending source-line indexes of the matching lines; empty when
            no item is open, the query is blank, or nothing matches.
        """
        detail = (
            self._library_media_detail
            if isinstance(self._library_media_detail, Mapping)
            else None
        )
        if detail is None:
            # Nothing to search -- and dropping the entry here means the
            # first lookup after a reader exit that cleared the detail
            # (rail switch, delete) releases the PREVIOUS document instead
            # of pinning its content behind the memo.
            self._library_media_content_match_memo = None
            return ()
        query = self._library_media_content_query
        # task-28026: the Analysis tab searches the analysis text; every other
        # mode searches the transcript. Mode is part of the memo key.
        mode = self._library_media_reader_session.mode
        memo = self._library_media_content_match_memo
        if (
            memo is not None
            and memo[0] is detail
            and memo[1] == query
            and memo[3] == mode
        ):
            return memo[2]
        # task-22208 memoizes the viewer state per detail arrival; going
        # through it keeps this memo's miss path from re-copying the whole
        # document that 22208 already built for this same arrival.
        viewer_state = self._library_media_viewer_state_cached(detail)
        corpus = viewer_state.analysis if mode == "analysis" else viewer_state.content
        matches = find_content_matches(corpus, query)
        self._library_media_content_match_memo = (detail, query, matches, mode)
        return matches

    def _advance_library_media_content_match(self, step: int) -> None:
        """Move the current content-search match index and scroll to it.

        No-ops when there is no open item or the query has no matches
        (the status line already reads "No matches" in that case).

        Args:
            step: ``1`` to move to the next match, ``-1`` for the previous
                one; wraps around the match count either direction.
        """
        matches = self._library_media_content_matches()
        if not matches:
            return
        self._library_media_content_match_index = (
            self._library_media_content_match_index + step
        ) % len(matches)
        line_index = matches[self._library_media_content_match_index]
        viewer = self._mounted_library_media_viewer()
        if viewer is None:
            return
        try:
            viewer.sync_match_index(
                matches=matches,
                match_index=self._library_media_content_match_index,
            )
        except (NoMatches, QueryError):
            return
        self.call_after_refresh(self._scroll_library_media_content_to_line, line_index)

    def _scroll_library_media_content_to_line(self, line_index: int) -> None:
        """Scroll the content region so the given source line is visible.

        When Raw is the ACTIVE mode, ``line_index`` (a SOURCE line index)
        is mapped to its virtual row through the Raw view's wrap index --
        scrolling to the source-line index directly, as if it were already
        a screen row, drifts once any line wraps. Otherwise (Rendered mode)
        this falls back to scrolling the active scroller's Y axis by that
        index directly.

        Gated on ``body.active_mode`` rather than ``body.raw_view is not
        None``: the latter is a LIFETIME accessor -- once Raw has been
        mounted once it stays mounted (and non-``None``) for the rest of
        the body's life -- so it kept routing scroll requests to the Raw
        view even after the user had switched back to Rendered, silently
        scrolling a hidden widget while the visible one never moved.

        Args:
            line_index: 0-based line index within the content text to
                reveal.
        """
        try:
            body = self.query_one(
                "#library-media-viewer-content", LibraryMediaContentBody
            )
        except (NoMatches, QueryError):
            return
        raw_view = body.raw_view
        if body.active_mode == "raw" and raw_view is not None:
            raw_view.scroll_to_source_line(line_index)
            return
        body.scroller.scroll_to(y=line_index, animate=False)

    def handle_library_media_read_later(self, event: Button.Pressed) -> None:
        """Toggle the open media item's read-it-later state via a worker.

        Reads the currently known saved state from ``_library_media_detail``
        (already reflecting ``is_read_it_later`` from the last fetch) to
        decide whether to save or remove, mirroring how
        ``handle_library_media_delete_confirm`` reads state synchronously
        before deferring to a worker.

        Args:
            event: Button press event emitted by the viewer's "Read it
                later" / "Remove from read-it-later" action.
        """
        event.stop()
        self._start_library_media_read_later_toggle()

    def _start_library_media_read_later_toggle(self) -> None:
        """Kick the read-it-later toggle worker for the open item (task-28027).

        Shared by the "Read later" button and the ``l`` accelerator so both
        read the last-fetched saved state and dispatch the same worker.
        """
        media_id = self._selected_media_id
        if not media_id:
            return
        detail = (
            self._library_media_detail
            if isinstance(self._library_media_detail, Mapping)
            else {}
        )
        currently_saved = bool(detail.get("is_read_it_later"))
        self.run_worker(
            self._toggle_library_media_read_later(
                media_id, currently_saved=currently_saved
            )
        )

    def action_library_media_read_later(self) -> None:
        """Keyboard 'l': toggle read-it-later for the open item (task-28027)."""
        self._start_library_media_read_later_toggle()

    async def _toggle_library_media_read_later(
        self, media_id: str, *, currently_saved: bool
    ) -> None:
        """Save or remove the read-it-later state, then re-fetch detail.

        Guards against a missing ``save_to_read_it_later``/
        ``remove_from_read_it_later`` service or a failed write by logging
        the failure and surfacing a quiet notice, but always re-fetches
        detail afterwards so the button's label never shows a stale state.

        Args:
            media_id: The Library media item id to toggle.
            currently_saved: Whether the item is currently saved for
                read-it-later (determines whether to save or remove).
        """
        service = getattr(self.app_instance, "media_reading_scope_service", None)
        method_name = (
            "remove_from_read_it_later" if currently_saved else "save_to_read_it_later"
        )
        method = getattr(service, method_name, None)
        service_media_id = self._library_media_backing_id(media_id)
        if callable(method):
            try:
                await self._run_library_service_call(
                    method,
                    mode="local",
                    media_id=service_media_id,
                    isolate_in_worker=True,
                )
            except Exception:
                logger.opt(exception=True).warning(
                    f"Failed to toggle Library media read-it-later state for {media_id!r}.",
                )
                self._notify_library_media_read_later_warning(
                    "Could not update read-it-later status."
                )
        else:
            self._notify_library_media_read_later_warning(
                "Read-it-later is unavailable."
            )
        await self._refresh_library_media_detail(media_id)

    def _notify_library_media_read_later_warning(self, message: str) -> None:
        """Surface a quiet warning notice for a failed read-it-later toggle.

        Args:
            message: Human-readable warning text to notify with.
        """
        notify = getattr(self.app_instance, "notify", None)
        if callable(notify):
            notify(message, severity="warning")
