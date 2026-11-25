"""Bind the Watchlists reader, briefing and transport to the live app owner."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from sqlite3 import Error as SQLiteError
from typing import TYPE_CHECKING

from textual.css.query import NoMatches
from textual.widgets import Button, Input, Static, TextArea

from ...Subscriptions.article_listening_lease import ListeningBusy
from ...Subscriptions.watchlist_normalizers import normalize_watchlist_item
from ...Widgets.confirmation_dialog import ConfirmationDialog
from .artifacts_pane import (
    ArtifactsPane,
    BriefListeningButton,
    BriefListeningRequested,
    BriefListeningTarget,
    CitationListenRequested,
)
from .content_pane import ArticleListenRequested
from .listening_transport import (
    CHECKING_OUTPUT,
    LOCAL_ONLY,
    ListeningTransport,
    ListeningTransportRequested,
    audio_host_notice,
    listening_reason,
    listening_service,
    output_issue,
    run_transport,
    update_listening_note,
)

if TYPE_CHECKING:
    from ..Screens.watchlists_collections_screen import WatchlistsCollectionsScreen
    from ...Subscriptions.article_listening_service import (
        ArticleListeningService,
        BriefListeningPreview,
        ListeningSnapshot,
    )

_NO_SERVICE = "Article listening is unavailable in this profile."
_FEED_TEXT = "Reads the saved feed text, which may be an excerpt."
_SAVED_TEXT = "Reads the saved article body."
_BRIEF_LABELS = {
    "play": "Play cited excerpts",
    "replace": "Play cited excerpts",
    "resume": "Resume cited excerpts",
    "resume_foreground": "Resume current article",
    "play_again": "Play again",
}


def _item_issue(item: dict | None, body: object) -> str:
    if item is None:
        return listening_reason("missing_item")
    if item.get("backend") != "local":
        return LOCAL_ONLY
    kind, fmt = item.get("content_kind"), item.get("content_format")
    if (kind, fmt) != (None, None) and (
        kind != "article" or fmt not in {"text", "markdown"}
    ):
        return (
            "This item has no supported article body. Change reports are unavailable."
        )
    if not isinstance(body, str) or not body or body.isspace():
        return listening_reason("empty_body")
    return ""


def _preview_copy(preview: BriefListeningPreview) -> str:
    text = f"{preview.playable_count} playable · {len(preview.unavailable_reasons)} unavailable"
    for reason in dict.fromkeys(preview.unavailable_reasons):
        text += " · " + listening_reason(reason)
    return text


def _preview_failure(error: Exception) -> str:
    if isinstance(error, SQLiteError):
        return listening_reason("article_audio_database_unavailable")
    return listening_reason(error.args[0] if error.args else None)


class WatchlistsListening:
    """Own the mounted projection and confirmation, with live app lookups."""

    def __init__(self, screen: WatchlistsCollectionsScreen) -> None:
        self.screen = screen
        self._output_owner: ArticleListeningService | None = None
        self._issue = _NO_SERVICE
        self._preview_key = None
        self._preview: BriefListeningPreview | None = None
        self._preview_issue = ""
        self._preview_ticket: object | None = None
        self._context = 0
        self._confirming = False
        self.reader_issue = listening_reason("missing_item")
        self.citation_issues: tuple[int, dict[int, str]] = (0, {})

    def invalidate_brief(self) -> None:
        self._context += 1
        self._preview_key = None
        self._preview = None
        self._preview_issue = ""
        self._preview_ticket = None

    def _active(self) -> bool:
        return self.screen.is_attached and self.screen.app.screen is self.screen

    @property
    def service(self) -> ArticleListeningService | None:
        return listening_service(self.screen.app_instance)

    def _local_issue(self, snapshot: ListeningSnapshot | None) -> str:
        if self.screen.runtime_backend != "local":
            return LOCAL_ONLY
        if snapshot is not None and (
            snapshot.phase == "unavailable"
            or snapshot.error == "local_listening_authority_required"
        ):
            return listening_reason(snapshot.error)
        return self._issue if self.service is self._output_owner else CHECKING_OUTPUT

    def _reader_issue(self, snapshot: ListeningSnapshot | None) -> str:
        if self.screen.active_section != "items":
            return "Open Read to listen to this article."
        issue = self._local_issue(snapshot)
        if issue:
            return issue
        if self.screen._selected_content_item is None:
            return listening_reason("missing_item")
        return self.reader_issue

    @staticmethod
    async def prepare_reader_issue(item: dict | None, fetched: str | None) -> str:
        body = fetched if fetched is not None else (item or {}).get("content")
        return await asyncio.to_thread(_item_issue, item, body)

    @staticmethod
    def load_citations(db, citation_ids):
        rows = {
            item_id: normalize_watchlist_item("local", row)
            for item_id, row in db.get_subscription_items_by_ids(citation_ids).items()
        }
        return rows, {
            item_id: _item_issue(row, row.get("content"))
            for item_id, row in rows.items()
        }

    def reader_key_available(self) -> bool:
        service = self.service
        return (
            self._active()
            and not self.screen._reader_verb_blocked()
            and not self._reader_issue(service.snapshot() if service else None)
            and bool(self.screen.query("#watchlists-listen"))
        )

    def listen_reader(self, item_id: str, runtime_backend: str) -> None:
        service = self.service
        item = self.screen._selected_content_item
        if (
            not self._active()
            or service is None
            or runtime_backend != "local"
            or item is None
            or item_id != item.get("id")
            or self._reader_issue(service.snapshot())
        ):
            return
        generation = self.screen._item_detail_generation
        self._listen(
            service,
            item_id,
            lambda: (
                self.screen.active_section == "items"
                and self.screen._selected_content_item is item
                and self.screen._item_detail_generation == generation
            ),
        )

    def _listen(
        self,
        service: ArticleListeningService,
        item_id: str,
        view_is_current: Callable[[], bool],
    ) -> None:
        revision = service.snapshot().revision
        if revision is None:
            return

        async def start() -> None:
            if (
                self._active()
                and self.service is service
                and self.screen.runtime_backend == "local"
                and view_is_current()
            ):
                await service.listen(
                    item_id, runtime_backend="local", expected_revision=revision
                )
                if self._active() and self.service is service:
                    self._preview_key = None
                    self.refresh()

        self.screen.run_worker(start(), name="watchlists-listen-article")

    def _brief_id(self, pane: ArtifactsPane) -> int | None:
        brief = pane.selected_briefing or {}
        displayed_id = brief.get("id")
        if (
            self.screen.active_section == "artifacts"
            and type(displayed_id) is int
            and brief.get("status") == "complete"
            and brief.get("watchlist_id") == self.screen._briefing_watchlist_id()
        ):
            return displayed_id
        return None

    def _citation_item(self, key: str | None) -> dict | None:
        return (
            self.screen._citation_item_lookup.get(int(key))
            if key and key.isdecimal()
            else None
        )

    def _citation_issue(self, key: str | None) -> str:
        if self._citation_item(key) is None:
            return listening_reason("missing_item")
        lookup_id, issues = self.citation_issues
        if lookup_id != id(self.screen._citation_item_lookup):
            return "Checking saved citation…"
        return issues.get(int(key), listening_reason("missing_item"))

    def listen_citation(self, briefing_id: int, item_key: str) -> None:
        service = self.service
        if (
            not self._active()
            or service is None
            or self._local_issue(service.snapshot())
        ):
            return
        try:
            pane = self.screen.query_one("#watchlists-artifacts-pane", ArtifactsPane)
        except NoMatches:
            return
        if (
            self._brief_id(pane) != briefing_id
            or pane.highlighted_citation() != item_key
        ):
            return
        item = self._citation_item(item_key)
        if item is not None and not self._citation_issue(item_key):
            context = self._context
            self._listen(
                service,
                item["id"],
                lambda: (
                    self._context == context
                    and pane.is_attached
                    and self._brief_id(pane) == briefing_id
                    and pane.highlighted_citation() == item_key
                ),
            )

    def refresh(self) -> None:
        if not self._active():
            return
        service = self.service
        if service is not self._output_owner:
            self._output_owner = service
            self._issue = CHECKING_OUTPUT if service is not None else _NO_SERVICE
            self.invalidate_brief()
            if service is not None:
                self.screen.run_worker(
                    self._probe_output(service),
                    exclusive=True,
                    group="wl-listening-readiness",
                )
        snapshot = service.snapshot() if service is not None else None
        host = audio_host_notice(self.screen.app_instance)
        try:
            self.screen.query_one(ListeningTransport).project(
                snapshot, self._issue, host
            )
        except NoMatches:
            return
        self._refresh_reader(snapshot, host)
        self._refresh_brief(service, snapshot, host)

    async def _probe_output(self, service: ArticleListeningService) -> None:
        issue = await output_issue()
        if self.service is service and self.screen.is_attached:
            self._issue = issue
            self.refresh()

    def _refresh_reader(self, snapshot: ListeningSnapshot | None, host: str) -> None:
        try:
            button = self.screen.query_one("#watchlists-listen", Button)
            note = self.screen.query_one("#watchlists-listen-note", Static)
        except NoMatches:
            button = note = None
        if button is not None and note is not None:
            issue = self._reader_issue(snapshot)
            item = self.screen._selected_content_item or {}
            copy = issue or (
                _FEED_TEXT
                if item.get("source_type") in {"rss", "atom", "json_feed", "podcast"}
                else _SAVED_TEXT
            )
            button.disabled = bool(issue)
            update_listening_note(note, " ".join(filter(None, (copy, host))))
        self.sync_reader_hint()

    def sync_reader_hint(self) -> None:
        eligible = self.reader_key_available()
        registered = self.screen._footer_shortcut_registration
        if eligible and not (registered and registered[0] == "watchlists-listening"):
            self.screen.register_footer_shortcuts(
                source="watchlists-listening",
                shortcuts=(("l", "Listen"),),
            )
        elif not eligible and registered and registered[0] == "watchlists-listening":
            self.screen.clear_footer_shortcuts(source="watchlists-listening")

    def _refresh_brief(
        self,
        service: ArticleListeningService | None,
        snapshot: ListeningSnapshot | None,
        host: str,
    ) -> None:
        try:
            pane = self.screen.query_one("#watchlists-artifacts-pane", ArtifactsPane)
            actions = pane.query_one("#artifacts-listening")
            play = pane.query_one("#artifacts-play-cited", BriefListeningButton)
            note = pane.query_one("#artifacts-listening-note", Static)
        except NoMatches:
            return
        briefing_id = self._brief_id(pane)
        actions.set_class(briefing_id is None, "artifacts-listening-hidden")
        if briefing_id is None:
            play.target = None
            return
        issue = self._local_issue(snapshot)
        if not issue and service is not None and snapshot is not None:
            # While active, timing checkpoints can advance the opaque
            # revision. Re-read guidance on context/action changes and when
            # idle state is replaced. A click always obtains fresh guidance.
            key = self._guidance_key(service, briefing_id, snapshot)
            if key != self._preview_key:
                self._preview_key = key
                self._preview = None
                self._preview_issue = ""
                ticket = self._preview_ticket = object()
                self.screen.run_worker(
                    self._read_preview(service, briefing_id, ticket),
                    exclusive=True,
                    group="wl-listening-preview",
                )
        preview = self._preview if not issue else None
        label = _BRIEF_LABELS[preview.action] if preview else _BRIEF_LABELS["play"]
        recheck = not issue and bool(self._preview_issue)
        if recheck:
            label = "Recheck cited excerpts"
        if (
            preview is not None
            and snapshot is not None
            and preview.action in {"resume", "resume_foreground"}
            and snapshot.phase in {"preparing", "playing"}
        ):
            label = "Queue in progress"
        if play.label.plain != label:
            play.label = label
        available = bool(
            preview and snapshot and self._brief_action_available(preview, snapshot)
        )
        play.target = (
            BriefListeningTarget(
                service, briefing_id, self._context, snapshot.revision, preview
            )
            if (available or recheck) and service and snapshot and snapshot.revision
            else None
        )
        play.disabled = play.target is None
        copy = (
            issue
            or self._preview_issue
            or (_preview_copy(preview) if preview else "Checking cited excerpts…")
        )
        if preview and preview.action == "resume_foreground" and available:
            copy = (
                "Resumes the current article. The saved queue stays in place. " + copy
            )
        update_listening_note(note, " ".join(filter(None, (copy, host))))
        try:
            listen = pane.query_one("#artifacts-listen-citation", Button)
            reason = pane.query_one("#artifacts-citation-listen-note", Static)
        except NoMatches:
            return
        citation_issue = issue or self._citation_issue(pane.highlighted_citation())
        listen.disabled = bool(citation_issue)
        update_listening_note(reason, " ".join(filter(None, (citation_issue, host))))

    def _guidance_key(self, service, briefing_id, snapshot) -> tuple:
        return (
            self._context,
            service,
            briefing_id,
            snapshot.phase,
            snapshot.actions,
            snapshot.revision
            if snapshot.phase not in {"preparing", "playing"}
            else None,
        )

    def _publish_guidance(self, service, briefing_id, preview, issue="") -> None:
        self._preview_key = self._guidance_key(service, briefing_id, service.snapshot())
        self._preview_ticket = None
        self._preview, self._preview_issue = preview, issue
        self.refresh()

    async def _read_preview(self, service, briefing_id, ticket) -> None:
        try:
            preview = await service.preview_brief(briefing_id, runtime_backend="local")
            issue = ""
        except (ListeningBusy, ValueError, OSError, SQLiteError) as error:
            preview = None
            issue = _preview_failure(error)
        if ticket is self._preview_ticket and service is self.service:
            self._preview, self._preview_issue = preview, issue
            self.refresh()

    @staticmethod
    def _brief_action_available(
        preview: BriefListeningPreview, snapshot: ListeningSnapshot
    ) -> bool:
        if preview.action in {"play", "replace"}:
            return preview.playable_count > 0 and snapshot.phase != "retiring"
        action = (
            "resume"
            if preview.action in {"resume", "resume_foreground"}
            else "play_again"
        )
        return action in snapshot.actions

    def play_brief(self, event: BriefListeningRequested) -> None:
        if not self._active() or self._confirming:
            return
        self._confirming = True
        self.screen.run_worker(self._play_brief(event), name="wl-listen-brief")

    async def _play_brief(self, event: BriefListeningRequested) -> None:
        screen = self.screen
        target = event.target
        service, briefing_id = target.service, target.briefing_id
        try:
            pane = screen.query_one("#watchlists-artifacts-pane", ArtifactsPane)

            def current() -> bool:
                return (
                    self._active()
                    and self.service is service
                    and screen.runtime_backend == "local"
                    and self._context == target.context
                    and event.button.is_attached
                    and pane.is_attached
                    and self._brief_id(pane) == briefing_id
                )

            if not current() or self._local_issue(service.snapshot()):
                return
            preview = await service.preview_brief(briefing_id, runtime_backend="local")
            if not current():
                return
            if preview != target.preview:
                self._publish_guidance(service, briefing_id, preview)
                return
            if not self._brief_action_available(preview, service.snapshot()):
                return
            if preview.action in {"resume", "resume_foreground", "play_again"}:
                action = "play_again" if preview.action == "play_again" else "resume"
                await run_transport(
                    screen.app_instance,
                    service,
                    action,
                    target.revision,
                    view_is_current=current,
                )
                return
            replacing = preview.action == "replace"
            if replacing:
                if not await screen.app.push_screen_wait(
                    ConfirmationDialog(
                        title="Replace saved listening?",
                        message=_preview_copy(preview)
                        + ". This replaces the saved queue and its position.",
                        confirm_label="Replace and play",
                    )
                ):
                    return
                if not current():
                    return
                checked = await service.preview_brief(
                    briefing_id, runtime_backend="local"
                )
                if not current():
                    return
                if checked != target.preview:
                    self._publish_guidance(service, briefing_id, checked)
                    screen.notify(
                        "The cited excerpts changed. Review them and try again.",
                        severity="warning",
                    )
                    return
            await service.play_brief(
                briefing_id,
                runtime_backend="local",
                replace=replacing,
                expected_revision=target.revision,
            )
        except NoMatches:
            return
        except (ListeningBusy, ValueError, OSError, SQLiteError) as error:
            if current():
                self._publish_guidance(
                    service, briefing_id, None, _preview_failure(error)
                )
        finally:
            self._confirming = False
            self.refresh()

    def transport(self, event: ListeningTransportRequested) -> None:
        service = self.service
        if self._active() and service is not None:
            self.screen.app.run_worker(
                run_transport(
                    self.screen.app_instance,
                    service,
                    event.action,
                    event.revision,
                    view_is_current=self._active,
                ),
                name="watchlists-listening-transport",
            )


class WatchlistsListeningMixin:
    """Textual message and binding handlers for the Watchlists screen."""

    def on_article_listen_requested(self, event: ArticleListenRequested) -> None:
        event.stop()
        self._listening.listen_reader(event.item_id, event.runtime_backend)

    def on_citation_listen_requested(self, event: CitationListenRequested) -> None:
        event.stop()
        self._listening.listen_citation(event.briefing_id, event.item_id)

    def on_brief_listening_requested(self, event: BriefListeningRequested) -> None:
        event.stop()
        self._listening.play_brief(event)

    def on_listening_transport_requested(
        self, event: ListeningTransportRequested
    ) -> None:
        event.stop()
        self._listening.transport(event)

    def _reader_verb_blocked(self) -> bool:
        """Keep Read verbs scoped to Local Read outside editable text."""
        focused = self.focused
        if isinstance(focused, Input) or (
            isinstance(focused, TextArea) and not focused.read_only
        ):
            return True
        return self.active_section != "items" or self.runtime_backend != "local"

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        if action == "listen_article":
            return self._listening.reader_key_available()
        return super().check_action(action, parameters)

    def action_listen_article(self) -> None:
        if self._listening.reader_key_available():
            item = self._selected_content_item
            if item is not None and isinstance(item.get("id"), str):
                self._listening.listen_reader(item["id"], self.runtime_backend)
