"""Library-owned artifact browsing; storage and mutations keep their owners."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger
from textual.widgets import Input

from ...Library.library_artifacts_catalog import LibraryArtifactsCatalog
from ...Library.library_artifacts_state import ArtifactKey, ArtifactScope
from ...Utils.adaptive_reader_state import (
    AdaptiveReaderLayoutProfile,
    normalize_adaptive_reader_preferences,
    resolve_adaptive_reader_layout,
)

if TYPE_CHECKING:
    from ..Screens.library_screen import LibraryScreen

ARTIFACT_READER_PROFILE = AdaptiveReaderLayoutProfile(list_first_when_empty=True)


class LibraryArtifactsController:
    """Fence data by profile/scope/selection and presentation by active visit."""

    def __init__(self, screen: LibraryScreen) -> None:
        self.screen = screen
        self.scope = ArtifactScope()
        self.page = None
        self.selected = None
        self.detail = None
        self.error = ""
        self.detail_error = ""
        self.loading = False
        self.detail_loading = False
        self.busy = False
        self.mode = "preview"
        self.shell = None
        self.generation = 0
        self.detail_generation = 0
        self.presentation = 0
        self.disposed = False
        self.suspended = False
        self.search_timer = None
        self.reader_open = False
        self._views = {}
        self._restore_target = None
        self._restore_scroll = None
        self._profile = self.profile()
        self._preferences = self.preferences()
        self.layout = resolve_adaptive_reader_layout(
            160, self._preferences, ARTIFACT_READER_PROFILE, reader_has_item=False
        )

    @property
    def app_instance(self):
        return self.screen.app_instance

    def profile(self) -> tuple[int, ...]:
        return tuple(
            id(getattr(self.app_instance, name, None))
            for name in ("subscriptions_db", "chachanotes_db", "local_chatbook_service")
        )

    def catalog(self) -> LibraryArtifactsCatalog:
        return LibraryArtifactsCatalog(
            subscriptions_db=getattr(self.app_instance, "subscriptions_db", None),
            chachanotes_db=getattr(self.app_instance, "chachanotes_db", None),
            chatbook_service=getattr(self.app_instance, "local_chatbook_service", None),
        )

    def active(self) -> bool:
        return not self.disposed and self.screen._library_selected_row_id.startswith(
            "artifacts-"
        )

    def presentable(self, generation: int, profile: tuple[int, ...]) -> bool:
        return (
            self.active()
            and not self.suspended
            and self.presentation == generation
            and self.profile() == profile
            and self.screen.is_mounted
            and self.screen.app.screen is self.screen
        )

    def preferences(self):
        library = self.app_instance.app_config.get("library", {})
        shared = library.get("reader", {})
        destination = library.get("artifacts_reader", {})
        raw = {**shared, **destination}
        for key in ("library_open", "library_width", "custom_widths_enabled"):
            if key in shared:
                raw[key] = shared[key]
        return normalize_adaptive_reader_preferences(raw)

    def enter_view(self, view: str) -> None:
        """Keep each artifact view's applied scope, selected copy and scroll in memory."""
        if view == self.scope.view:
            return
        self.leave()
        self.generation += 1
        self.detail_generation += 1
        saved = self._views.get(view)
        self.scope = saved[0] if saved else ArtifactScope(view=view)
        self.page = self.detail = self.selected = None
        self.loading = self.detail_loading = False
        self.error = self.detail_error = ""
        self.reader_open = False
        self._restore_target = saved[1] if saved else None
        self._restore_scroll = saved[2:] if saved else None

    def capture_view(self) -> None:
        items_y = body_y = 0
        if self.shell is not None:
            items = list(self.shell.query("#library-artifacts-list"))
            body = list(self.shell.query("#library-artifacts-body"))
            if not items or not body:
                return  # Recompose/disposal already removed the old pane children.
            items_y, body_y = items[0].scroll_y, body[0].scroll_y
        self._views[self.scope.view] = (
            self.scope,
            self.selected,
            items_y,
            body_y,
            self.mode,
            self.reader_open,
        )

    def build_shell(self, state, preferences):
        from ...Widgets.Library.library_artifacts_reader_shell import (
            LibraryArtifactsReaderShell,
        )
        from ...Widgets.Library.library_rail import LibraryRail

        rail = LibraryRail(
            state,
            preferences,
            query=self.screen._library_rail_search_value(),
            search_placeholder=self.screen._library_rail_search_placeholder(),
            workspaces_body_factory=self.screen._compose_workspaces_rail_body,
            top_action_factory=self.screen._compose_library_rail_top_action,
            lifecycle=self.screen._library_lifecycle,
            onboarding_all_empty=self.screen._library_onboarding_all_empty,
            id="library-rail",
            classes="destination-workbench-pane",
        )
        return LibraryArtifactsReaderShell(rail, self)

    def attach(self, shell) -> None:
        self.shell = shell
        self.resize()
        self.sync()
        if self.page is None and not self.loading:
            if self._restore_target:
                target, self._restore_target = self._restore_target, None
                self.open_target(target)
            else:
                self.request_scope(self.scope)
        self._queue_scroll_restore()
        self.screen._artifacts_navigation.resume()

    def sync(self) -> None:
        if self.shell is not None and self.shell.is_mounted and self.active():
            self.shell.sync()

    def resize(self, *, priority=None, manual_reopen=None) -> None:
        shell = self.shell
        if not shell or not shell.is_mounted:
            return
        self._preferences = self.preferences()
        self.layout = resolve_adaptive_reader_layout(
            shell.size.width,
            self._preferences,
            ARTIFACT_READER_PROFILE,
            previous=self.layout,
            priority=priority,
            reader_has_item=self.reader_open,
        )
        shell.sync_layout(self.layout, manual_reopen=manual_reopen)

    def toggle_pane(self, pane: str) -> None:
        opening = not (
            self.layout.library_open if pane == "library" else self.layout.items_open
        )
        section = "reader" if pane == "library" else "artifacts_reader"
        key = f"{pane}_open"
        self.app_instance.app_config.setdefault("library", {}).setdefault(section, {})[
            key
        ] = opening
        self.resize(
            priority=pane if opening else None, manual_reopen=pane if opening else None
        )
        self.screen.run_worker(
            self._persist(section, key, opening), group="library-artifacts-preferences"
        )

    async def _persist(self, section, key, value) -> None:
        from ...config import save_setting_to_cli_config

        try:
            await asyncio.to_thread(
                save_setting_to_cli_config,
                "library",
                section,
                {**self.app_instance.app_config["library"][section], key: value},
            )
        except Exception:  # noqa: BLE001 - preserve the mounted reader on owner failure
            self.notify("Pane preference could not be saved.", "warning")

    def focus_reader(self) -> None:
        if not self.active() or self.selected is None:
            return
        self.reader_open = True
        # Drop a transient Items priority without changing the user's saved pane choice.
        self.layout = replace(self.layout, priority_pane=None)
        self.resize()
        generation, profile = self.presentation, self.profile()
        self.screen.call_after_refresh(
            self._focus, "#library-artifacts-body", generation, profile
        )

    def focus_items(self, *, search=False) -> None:
        self.reader_open = False
        self.resize(priority="items")
        generation, profile = self.presentation, self.profile()
        self.screen.call_after_refresh(
            self._focus,
            "#library-artifacts-search" if search else "#library-artifacts-list",
            generation,
            profile,
        )

    def _focus(self, selector, generation, profile) -> None:
        if (
            self.presentable(generation, profile)
            and self.shell
            and self.shell.is_mounted
        ):
            self.shell.query_one(selector).focus()

    def search(self, query: str) -> None:
        self.stop_timer()
        if query == self.scope.query:
            return
        self.presentation += 1
        self.search_timer = self.screen.set_timer(
            0.2, lambda: self.request_scope(replace(self.scope, query=query))
        )

    def stop_timer(self) -> None:
        if self.search_timer:
            self.search_timer.stop()
            self.search_timer = None

    def request_scope(self, scope: ArtifactScope) -> None:
        self.stop_timer()
        self.scope = scope
        self._read_page()

    def request_page(self, direction: str) -> None:
        if self.loading or self.error or not self.page:
            return
        boundary = None
        read_direction = "after"
        if direction in ("prev", "last"):
            read_direction = "before"
        if direction == "prev" and self.page.items:
            boundary = self.page.items[0].order_key
        elif direction == "next" and self.page.items:
            boundary = self.page.items[-1].order_key
        self._read_page(boundary=boundary, direction=read_direction)

    def open_target(self, key: ArtifactKey) -> None:
        self._read_page(target=key)

    def _read_page(
        self, *, boundary=None, direction="after", target=None, fallback_missing=False
    ) -> None:
        self.generation += 1
        self.presentation += 1
        self.detail_generation += 1
        generation, profile, scope = self.generation, self.profile(), self.scope
        self.loading = True
        self.detail_loading = False
        self.error = ""
        self.sync()
        catalog = self.catalog()
        self.screen.run_worker(
            self._load_page(
                catalog,
                generation,
                profile,
                scope,
                boundary,
                direction,
                target,
                fallback_missing,
            ),
            group="library-artifacts-page",
        )

    async def _load_page(
        self,
        catalog,
        generation,
        profile,
        scope,
        boundary,
        direction,
        target,
        fallback_missing=False,
    ) -> None:
        missing = False
        try:
            if target:
                page = await asyncio.to_thread(catalog.locate, scope, target)
                if page is None:
                    if fallback_missing:
                        target = None
                        page = await asyncio.to_thread(catalog.read_page, scope)
                    else:
                        missing = True
                        raise LookupError(
                            "Requested artifact is missing or outside this filter."
                        )
            else:
                page = await asyncio.to_thread(
                    catalog.read_page, scope, boundary=boundary, direction=direction
                )
                if not page.items and page.total and boundary is not None:
                    page = await asyncio.to_thread(
                        catalog.read_page,
                        scope,
                        direction="before" if direction == "after" else "after",
                    )
            error = ""
        except Exception as exc:  # noqa: BLE001 - storage/action failures must remain recoverable
            logger.warning("Library artifact read failed: {}", type(exc).__name__)
            page = None
            error = (
                str(exc)
                if isinstance(exc, LookupError)
                else "Artifacts unavailable. Retry, or open Reports → Kept for saved reports."
            )
        if (
            self.disposed
            or generation != self.generation
            or self.profile() != profile
            or scope != self.scope
        ):
            return
        navigation = self.screen._artifacts_navigation
        if target and not navigation.target_is_current(target, generation):
            return
        self.loading = False
        self.error = error
        if missing:
            self.target_unavailable()
            navigation.target_finished(target, generation, missing=True)
        elif error and target:
            navigation.target_failed(target, generation)
        if page is not None:
            self.page = page
            keys = {row.key for row in page.items}
            selected = (
                target
                if target in keys
                else self.selected
                if self.selected in keys
                else page.items[0].key
                if page.items
                else None
            )
            self.select(selected, refresh=True)
            if target:
                navigation.target_finished(target, generation)
        self.sync()

    def target_unavailable(self) -> None:
        """Clear an exact missing handoff so an earlier selection cannot impersonate it."""
        self.page = self.detail = self.selected = None
        self.loading = self.detail_loading = False
        self.detail_generation += 1
        self.error = "Requested artifact is missing or outside this filter. Retry to browse available copies."
        self.sync()

    def select(self, key: ArtifactKey | None, *, refresh=False) -> None:
        if key == self.selected and not refresh:
            return
        if key and (
            not self.page or not any(row.key == key for row in self.page.items)
        ):
            return
        self.presentation += 1
        self.detail_generation += 1
        self.selected = key
        self.detail = None
        self.detail_error = ""
        self.detail_loading = key is not None
        self.mode = "preview"
        self.sync()
        if key:
            row = next(row for row in self.page.items if row.key == key)
            identity = (
                self.profile(),
                self.scope,
                key,
                row.revision,
                self.detail_generation,
            )
            self.screen.run_worker(
                self._load_detail(self.catalog(), identity),
                group="library-artifacts-detail",
            )

    async def _load_detail(self, catalog, identity) -> None:
        profile, scope, key, revision, generation = identity
        try:
            detail = await asyncio.to_thread(catalog.read_detail, key)
            error = (
                "This copy is no longer available. Retry to refresh the list."
                if detail is None
                else ""
            )
            if detail is not None and detail.revision != revision:
                detail = None
                error = "This copy changed. Retry to read its latest version."
        except Exception:  # noqa: BLE001 - preserve the mounted reader on owner failure
            detail = None
            error = "This copy could not be read. Retry to refresh it."
        if self.disposed or (profile, scope, key, generation) != (
            self.profile(),
            self.scope,
            self.selected,
            self.detail_generation,
        ):
            return
        self.detail, self.detail_error, self.detail_loading = detail, error, False
        self.sync()
        self._queue_scroll_restore()

    def _queue_scroll_restore(self) -> None:
        if not self.detail or self._restore_scroll is None or not self.shell:
            return
        saved, self._restore_scroll = self._restore_scroll, None
        self.mode, self.reader_open = saved[2:]
        self.sync()
        self.resize()
        self.screen.run_worker(
            self._restore_view_scroll(
                saved, self.presentation, self.profile(), self.shell
            ),
            group="library-artifacts-scroll-restore",
            exclusive=True,
            exit_on_error=False,
        )

    async def _restore_view_scroll(self, saved, generation, profile, shell) -> None:
        # Markdown parses and mounts asynchronously; a layout refresh alone may
        # still see an empty body and clamp the remembered offset to zero.
        await shell.work.wait_for_body()
        self.screen.call_after_refresh(
            self._apply_view_scroll, saved, generation, profile, shell
        )

    def _apply_view_scroll(self, saved, generation, profile, shell) -> None:
        if (
            not self.presentable(generation, profile)
            or shell is not self.shell
            or not shell.is_mounted
            or self.mode != saved[2]
        ):
            return
        items_y, body_y = saved[:2]
        shell.query_one("#library-artifacts-list").scroll_to(y=items_y, animate=False)
        shell.query_one("#library-artifacts-body").scroll_to(y=body_y, animate=False)

    def suspend(self) -> None:
        self.suspended = True
        self.presentation += 1
        self.stop_timer()

    def resume(self) -> None:
        self.suspended = False
        if self.profile() != self._profile:
            self.screen._artifacts_navigation.invalidate()
            self._profile = self.profile()
            self.page = self.detail = self.selected = None
            self._views.clear()
            self.request_scope(self.scope)
        elif self.active():
            # Restart a query stopped while leaving; settled reads remain valid.
            if self.shell and self.shell.is_mounted:
                query = self.shell.query_one("#library-artifacts-search", Input).value
                if query != self.scope.query:
                    self.request_scope(replace(self.scope, query=query))
            self.sync()

    def leave(self) -> None:
        self.capture_view()
        if self.scope.view in self._views:
            self._restore_scroll = self._views[self.scope.view][2:]
        self.presentation += 1
        self.stop_timer()

    def dispose(self) -> None:
        self.disposed = True
        self.generation += 1
        self.detail_generation += 1
        self.presentation += 1
        self.stop_timer()

    def notify(self, message, severity="information") -> None:
        self.screen.notify(message, severity=severity)

    def action(self, name: str) -> None:
        if name == "retry" and self.screen._artifacts_navigation.retry():
            return
        if name in {"first", "prev", "next", "last"}:
            self.request_page(name)
        elif name == "retry" and self.detail_error and not self.error:
            self.open_target(self.selected) if self.selected else self.request_scope(
                self.scope
            )
        elif name in {"all", "kept", "sort", "retry"}:
            scope = self.scope
            if name in {"all", "kept"}:
                scope = replace(scope, kept_only=name == "kept")
            elif name == "sort":
                scope = replace(
                    scope, sort="title" if scope.sort == "newest" else "newest"
                )
            self.request_scope(scope)
        elif name == "back":
            self.focus_items()
        elif name in {"preview", "details"}:
            self.mode = name
            self.sync()
        elif name == "manage":
            from ..Navigation.main_navigation import NavigateToScreen

            self.screen.post_message(NavigateToScreen("chatbooks"))
        elif name == "share" and self.detail and self.detail.can_share:
            self.screen._artifacts_share_controller.open_dialog(self.selected)
        elif name == "watchlists":
            self.open_watchlists()
        elif name == "demo":
            service = getattr(self.app_instance, "daily_report_demo_service", None)
            if service:
                task = service.run_demo_detached()
                if task:
                    task.add_done_callback(
                        lambda _: (
                            self.request_scope(self.scope)
                            if not self.disposed and self.active()
                            else None
                        )
                    )
            else:
                self.notify(
                    "The report demo is unavailable in this runtime.", "warning"
                )
        elif (
            name in {"keep", "export", "scripts", "play", "source", "console"}
            and self.detail
            and not (self.loading or self.error or self.busy)
        ):
            self.busy = True
            self.sync()
            identity = (
                self.profile(),
                self.selected,
                self.detail.revision,
                self.presentation,
            )
            self.screen.run_worker(
                self._run_action(name, identity), group="library-artifacts-action"
            )

    def open_watchlists(self) -> None:
        from ...Constants import (
            WATCHLISTS_NAV_CONTEXT_BACKEND,
            WATCHLISTS_NAV_CONTEXT_BRIEFING_ID,
            WATCHLISTS_NAV_CONTEXT_SECTION,
        )
        from ..Navigation.main_navigation import NavigateToScreen

        context = {}
        if self.selected and self.selected.source == "live_report":
            context = {
                WATCHLISTS_NAV_CONTEXT_BACKEND: "local",
                WATCHLISTS_NAV_CONTEXT_SECTION: "artifacts",
                WATCHLISTS_NAV_CONTEXT_BRIEFING_ID: f"local:briefing:{self.selected.native_id}",
            }
        self.screen.post_message(NavigateToScreen("watchlists_collections", context))

    async def _run_action(self, name, identity) -> None:
        profile, key, revision, presentation = identity
        try:
            catalog = self.catalog()
            fresh = await asyncio.to_thread(catalog.read_detail, key)
            if not self.presentable(presentation, profile) or self.selected != key:
                return
            if fresh is None or fresh.revision != revision:
                self.detail = None
                self.detail_loading = False
                self.detail_error = (
                    "This copy is no longer available. Retry to refresh the list."
                    if fresh is None
                    else "This copy changed. Retry to read its latest version."
                )
                return
            if name == "keep" and fresh.can_keep:
                from ...Subscriptions.briefing_keep import keep_briefing

                result = await asyncio.to_thread(
                    keep_briefing,
                    catalog.subscriptions_db,
                    catalog.chachanotes_db,
                    key.native_id,
                    origin="manual",
                )
                self.notify(
                    f"Kept in Library · {result['scripts_added']} scripts added"
                )
                if self.presentable(presentation, profile) and self.selected == key:
                    # An explicit successful Keep selects its durable copy even if
                    # the Watchlist was renamed after the earlier snapshot.
                    self.scope = replace(self.scope, query="")
                    if self.shell and self.shell.is_mounted:
                        self.shell.query_one(
                            "#library-artifacts-search", Input
                        ).value = ""
                    self.open_target(ArtifactKey("kept_report", result["kept_id"]))
            elif name == "scripts" and key.source == "kept_report":
                from ..Watchlists_Modules.kept_briefings_modal import KeptBriefingsModal

                scope = self.scope
                self.screen.app.push_screen(
                    KeptBriefingsModal(
                        catalog.chachanotes_db, initial_kept_id=key.native_id
                    ),
                    callback=lambda _: self.screen.call_after_refresh(
                        self._scripts_closed, key, profile, scope
                    ),
                )
            elif name == "export" and fresh.can_export:
                await self._export_dialog(key, profile, presentation)
            elif (
                name == "source"
                and fresh.source_available
                and fresh.source_conversation_id
            ):
                self.app_instance.resume_console_conversation(
                    fresh.source_conversation_id
                )
            elif name == "console" and key.source == "chatbook":
                from ..Screens.artifacts_screen import ArtifactsScreen

                service = self.app_instance.local_chatbook_service
                record = await asyncio.to_thread(
                    lambda: service.artifact_read_snapshot().get_record(key.native_id)
                )
                if (
                    record
                    and self.presentable(presentation, profile)
                    and self.selected == key
                ):
                    launch = ArtifactsScreen._build_chatbook_console_launch(record)
                    if launch:
                        self.app_instance.open_console_for_live_work(**launch)
            elif name == "play" and fresh.can_play:
                await self._play(key, profile, presentation)
        except Exception as exc:  # noqa: BLE001 - storage/action failures must remain recoverable
            from ...Subscriptions.briefing_keep import KeepRefused

            self.notify(
                str(exc)
                if isinstance(exc, KeepRefused)
                else f"Could not {name} this report. Please retry.",
                "warning",
            )
            logger.warning(
                "Library artifact action {} failed: {}", name, type(exc).__name__
            )
        finally:
            self.busy = False
            self.sync()

    def _scripts_closed(self, key, profile, scope) -> None:
        # The modal's own suspension changes the visit token. Its exact source
        # still owns this read-only refresh when the same Library view returns.
        if (
            self.disposed
            or self.profile() != profile
            or self.scope != scope
            or self.selected != key
            or not self.active()
            or not self.screen.is_mounted
            or self.screen.app.screen is not self.screen
        ):
            return
        self.detail = None
        self.detail_error = ""
        self._read_page(target=key, fallback_missing=True)

    async def _export_dialog(self, key, profile, presentation) -> None:
        from ...Subscriptions.briefing_export import default_briefing_filename
        from ...Third_Party.textual_fspicker import FileSave

        db = getattr(
            self.app_instance,
            "subscriptions_db" if key.source == "live_report" else "chachanotes_db",
            None,
        )
        getter = (
            db.get_briefing if key.source == "live_report" else db.get_kept_briefing
        )
        report = await asyncio.to_thread(getter, key.native_id)
        if (
            not report
            or not self.presentable(presentation, profile)
            or key != self.selected
        ):
            return
        # The accepted picker result owns its captured copy across parent suspension.
        await self.screen.app.push_screen(
            FileSave(
                location=str(Path.home()),
                title="Export report as Markdown",
                default_file=default_briefing_filename(
                    report, watchlist_name=str(report.get("watchlist_name") or "Report")
                ),
            ),
            callback=lambda path: self._write_export(path, report, profile),
        )

    async def _write_export(self, path, report, profile) -> None:
        from ...Subscriptions.briefing_export import briefing_markdown_document
        from ...Utils.path_validation import validate_path_simple

        if not path or self.profile() != profile or self.disposed:
            return
        try:
            destination = validate_path_simple(Path(path), require_exists=False)
            await asyncio.to_thread(
                destination.write_text,
                briefing_markdown_document(report),
                encoding="utf-8",
            )
            self.notify(f"Report exported to {destination.name}")
        except Exception:  # noqa: BLE001 - preserve the mounted reader on owner failure
            self.notify(
                "Report could not be exported. Check the destination and retry.",
                "error",
            )

    async def _play(self, key, profile, presentation) -> None:
        from ...Subscriptions.briefing_audio import audio_file_path_is_safe
        from ...TTS.audio_player import play_audio_file
        from ...Utils.path_validation import validate_path_simple

        db = getattr(self.app_instance, "subscriptions_db", None)
        raw = await asyncio.to_thread(db.get_artifact_audio_path, key)
        if not self.presentable(presentation, profile) or key != self.selected:
            return
        if not raw or not audio_file_path_is_safe(raw):
            self.notify("This report's audio is no longer available.", "warning")
            return
        path = validate_path_simple(Path(raw), require_exists=True)
        play_audio_file(path)
