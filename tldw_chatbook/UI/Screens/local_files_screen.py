"""Native search and cited answers for one indexed Workspace folder."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
from threading import Event
from typing import TYPE_CHECKING

from rich.text import Text
from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, VerticalScroll
from textual.message import Message
from textual.widgets import Button, Input, OptionList, Select, Static
from textual.widgets.option_list import Option

from ..Navigation.base_app_screen import BaseAppScreen
from ..Navigation.main_navigation import NavigateToScreen

if TYPE_CHECKING:
    from ...Library.library_rag_answer_service import LibraryRagAnswer
    from ...Workspaces.project_index.file_answers import FileAnswerDestination
    from ...Workspaces.project_index.models import (
        BindingSearchResult,
        ProjectIndexHit,
        ProjectIndexKey,
        ProjectIndexStatus,
    )
    from ...Workspaces.project_index.service import ProjectIndexService
    from ...app import TldwCli


_SEARCHABLE_STATES = {"Ready", "Partial", "Building"}
_INDEX_ERRORS = (OSError, RuntimeError, ValueError, sqlite3.Error)
_STALE = "Files or index changed. Search again for current results."


@dataclass(frozen=True)
class _Folder:
    workspace_id: str
    workspace_name: str
    binding_id: str
    root: Path


@dataclass(frozen=True)
class _Request:
    folder: _Folder
    query: str = field(repr=False)
    mode: str
    destination: FileAnswerDestination | None = None
    stop: Event = field(default_factory=Event, compare=False, repr=False)


type _PublishedSearch = tuple[_Request, ProjectIndexService, BindingSearchResult]


def _reveal_file(path: Path) -> bool:
    """Ask the host file manager to select an already validated original."""
    if sys.platform == "win32":
        explorer = Path(os.environ.get("SystemRoot", "C:/Windows")) / "explorer.exe"
        subprocess.Popen([str(explorer), "/select,", str(path)])
    elif sys.platform == "darwin":
        subprocess.Popen(["/usr/bin/open", "-R", str(path)])
    elif sys.platform.startswith("linux") and (gdbus := shutil.which("gdbus")):
        return (
            subprocess.run(
                [
                    gdbus,
                    "call",
                    "--session",
                    "--dest",
                    "org.freedesktop.FileManager1",
                    "--object-path",
                    "/org/freedesktop/FileManager1",
                    "--method",
                    "org.freedesktop.FileManager1.ShowItems",
                    f"['{path.as_uri()}']",
                    "",
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=5,
                check=False,
            ).returncode
            == 0
        )
    else:
        return False
    return True


def _location(hit: ProjectIndexHit) -> str:
    if not hit.line_start:
        return hit.relative_path
    return f"{hit.relative_path} · lines {hit.line_start}–{hit.line_end}"


class LocalFilesScreen(BaseAppScreen):
    """Search one live folder binding without creating a Console chat."""

    class BindingChanged(Message):
        def __init__(self, binding_id: str) -> None:
            super().__init__()
            self.binding_id = binding_id

    def __init__(self, app_instance: TldwCli, **kwargs) -> None:
        super().__init__(app_instance, "local_files", **kwargs)
        self._folders: dict[str, _Folder] = {}
        self._destinations: dict[int, FileAnswerDestination] = {}
        self._request: _Request | None = None
        self._pending_confirmation: _Request | None = None
        self._action_serial = 0
        self._unsubscribe = None
        self._folder_state = ""
        self._published: _PublishedSearch | None = None
        self._citations: list[ProjectIndexHit] = []
        self._preview_hit: ProjectIndexHit | None = None

    def compose_content(self) -> ComposeResult:
        if self.app.is_web:
            yield Static("Local Files is available in the native terminal app.")
            return
        with VerticalScroll():
            yield Static("Local Files", classes="workbench-header-title")
            with Horizontal(classes="form-row"):
                yield Select(
                    [],
                    prompt="Choose a Workspace folder",
                    id="local-files-folder",
                    classes="form-select w-fill",
                    disabled=True,
                )
                yield Button(
                    "Folder settings", id="local-files-settings", classes="form-button"
                )
            yield Static(
                "Loading Workspace folders…", id="local-files-index", markup=False
            )
            with Horizontal(classes="form-row"):
                yield Input(
                    placeholder="File name, words, or question",
                    id="local-files-query",
                    classes="form-input w-fill",
                )
                yield Button(
                    "Search", id="local-files-search", variant="primary", disabled=True
                )
                yield Button(
                    "Ask", id="local-files-ask", classes="form-button", disabled=True
                )
            with Horizontal(classes="form-row"):
                yield Select(
                    [
                        ("Keyword", "keyword"),
                        ("Semantic", "semantic"),
                        ("Hybrid", "hybrid"),
                    ],
                    allow_blank=False,
                    value="keyword",
                    id="local-files-mode",
                    classes="form-select w-fill",
                )
                yield Select(
                    [],
                    prompt="Choose an answer destination",
                    id="local-files-destination",
                    classes="form-select",
                    disabled=True,
                )
            yield Static("", id="local-files-status", markup=False)
            yield OptionList(
                id="local-files-results", classes="h-auto local-files-sources"
            )
            yield Static("", id="local-files-answer", markup=False)
            yield OptionList(
                id="local-files-citations", classes="h-auto local-files-sources"
            )
            yield Static("", id="local-files-preview", markup=False)
            yield Button(
                "Reveal in file manager", id="local-files-reveal", disabled=True
            )

    def on_mount(self) -> None:
        if self.app.is_web:
            return
        registry = getattr(self.app_instance, "workspace_registry_service", None)
        if registry is not None:
            self._unsubscribe = registry.subscribe_project_index_changes(
                lambda binding_id: self.post_message(self.BindingChanged(binding_id))
            )
        for widget_id in ("results", "citations", "reveal"):
            self.query_one(f"#local-files-{widget_id}").display = False
        self._reload_folders()
        self.run_worker(self._load_destinations(), group="local-files-destinations")

    def on_unmount(self) -> None:
        self._supersede_request()
        if self._unsubscribe is not None:
            self._unsubscribe()

    def on_screen_resume(self) -> None:
        request, self._pending_confirmation = self._pending_confirmation, None
        if request is not None and self._is_current(request):
            self._confirm_answer(request)

    def _reload_folders(self) -> None:
        self.run_worker(
            self._load_folders(), group="local-files-folders", exclusive=True
        )

    def _list_folders(self) -> list[_Folder]:
        from ...Workspaces.project_index.runtime import (
            project_index_registry_connections,
        )

        registry = getattr(self.app_instance, "workspace_registry_service", None)
        if registry is None:
            return []
        with project_index_registry_connections(registry.db):
            return [
                _Folder(
                    workspace.workspace_id,
                    workspace.name,
                    binding.binding_id,
                    Path(binding.locator),
                )
                for workspace in registry.list_workspaces()
                for binding in registry.list_folder_bindings(workspace.workspace_id)
            ]

    async def _load_folders(self) -> None:
        select = self.query_one("#local-files-folder", Select)
        try:
            folders = await asyncio.to_thread(self._list_folders)
        except _INDEX_ERRORS:
            self._index_copy(
                "Workspace folders are unavailable. Check Folder settings."
            )
            return
        if not self.is_attached:
            return
        self._folders = {folder.binding_id: folder for folder in folders}
        current = select.value
        select.set_options(
            (Text(f"{folder.workspace_name} · {folder.root}"), folder.binding_id)
            for folder in folders
        )
        select.disabled = not folders
        if current in self._folders:
            select.value = current
        elif folders:
            self._index_copy("Choose a folder to check its keyword index.")
        else:
            self._index_copy(
                "Add a local folder and enable its keyword index in Folder settings."
            )

    def _list_destinations(self) -> tuple[FileAnswerDestination, ...]:
        from ...Workspaces.project_index.file_answers import list_answer_destinations

        return list_answer_destinations(lambda: self.app_instance.app_config)

    async def _load_destinations(self) -> None:
        select = self.query_one("#local-files-destination", Select)
        destinations = await asyncio.to_thread(self._list_destinations)
        if not self.is_attached:
            return
        self._destinations = dict(enumerate(destinations))
        select.set_options(
            (Text(destination.label), key)
            for key, destination in self._destinations.items()
        )
        select.disabled = not destinations
        if not destinations:
            select.prompt = (
                "Configure loopback llama.cpp, Ollama, or Moonshot in Settings"
            )

    def _folder(self) -> _Folder | None:
        return self._folders.get(self.query_one("#local-files-folder", Select).value)

    def _destination(self) -> FileAnswerDestination | None:
        return self._destinations.get(
            self.query_one("#local-files-destination", Select).value
        )

    def _mode(self) -> str:
        return str(self.query_one("#local-files-mode", Select).value)

    def _query(self) -> str:
        return self.query_one("#local-files-query", Input).value.strip()

    def _index_copy(self, message: str) -> None:
        self.query_one("#local-files-index", Static).update(Text(message))

    def _status(self, message: str) -> None:
        self.query_one("#local-files-status", Static).update(Text(message))

    def _set_controls(self) -> None:
        searchable = self._folder_state in _SEARCHABLE_STATES
        self.query_one("#local-files-search", Button).disabled = not searchable
        self.query_one("#local-files-ask", Button).disabled = (
            not searchable or self._destination() is None
        )

    def _clear_preview(self) -> None:
        self._action_serial += 1
        self._preview_hit = None
        self.query_one("#local-files-preview", Static).update("")
        reveal = self.query_one("#local-files-reveal", Button)
        reveal.display = False
        reveal.disabled = True

    def _supersede_request(self) -> None:
        if self._request is not None:
            self._request.stop.set()
            self._request = None
        self._pending_confirmation = None
        self._published = None

    def _clear_request(self) -> None:
        self._supersede_request()
        self._citations.clear()
        self._clear_preview()
        self.query_one("#local-files-answer", Static).update("")
        for widget_id in ("results", "citations"):
            widget = self.query_one(f"#local-files-{widget_id}", OptionList)
            widget.clear_options()
            widget.display = False
        self._status("")

    def on_local_files_screen_binding_changed(self, event: BindingChanged) -> None:
        folder = self._folder()
        if folder and folder.binding_id == event.binding_id:
            self._clear_request()
            self._folder_state = ""
            self._set_controls()
            self._index_copy("Checking changed folder…")
            self._check_folder(folder)
        self._reload_folders()

    @on(Select.Changed, "#local-files-folder")
    def _folder_changed(self, event: Select.Changed) -> None:
        if event.value != event.select.value:
            return
        self._clear_request()
        self._folder_state = ""
        self._set_controls()
        if folder := self._folder():
            self._index_copy("Checking keyword index…")
            self._check_folder(folder)

    @on(Input.Changed, "#local-files-query")
    @on(Select.Changed, "#local-files-destination")
    @on(Select.Changed, "#local-files-mode")
    def _request_changed(self) -> None:
        self._clear_request()
        self._set_controls()

    def _check_folder(self, folder: _Folder) -> None:
        self.run_worker(
            self._load_folder_status(folder), group="local-files-index", exclusive=True
        )

    def _folder_service(
        self, folder: _Folder
    ) -> tuple[ProjectIndexService, ProjectIndexKey]:
        from ...Workspaces.project_index.runtime import (
            get_project_index_service,
            project_index_key,
        )

        app = self.app_instance
        service = get_project_index_service(app)
        return service, project_index_key(
            app, folder.workspace_id, folder.binding_id, folder.root
        )

    def _read_folder_status(
        self, folder: _Folder
    ) -> tuple[ProjectIndexStatus, bool, str]:
        service, key = self._folder_service(folder)
        status, _, upgrade = service.management_snapshot(key)
        try:
            semantic_state = service.semantic_state(key).state
        except _INDEX_ERRORS:
            semantic_state = "Unavailable"
        return status, upgrade, semantic_state

    async def _load_folder_status(self, folder: _Folder) -> None:
        try:
            status, upgrade, semantic_state = await asyncio.to_thread(
                self._read_folder_status, folder
            )
        except _INDEX_ERRORS:
            state = ""
            copy = "Folder index unavailable. Check Folder settings."
        else:
            state = "" if upgrade else status.state
            if upgrade:
                copy = "Keyword index requires an upgrade in Folder settings."
            else:
                copy = (
                    f"Keyword index · {status.state} · {status.indexed:,} files"
                    f" · Semantic index {semantic_state}"
                )
                if state not in _SEARCHABLE_STATES:
                    copy += " · Manage the index in Folder settings"
        if self.is_attached and self._folder() == folder:
            self._folder_state = state
            self._index_copy(copy)
            self._set_controls()

    @on(Button.Pressed, "#local-files-settings")
    def _open_settings(self) -> None:
        context = {"category": "workspaces"}
        if folder := self._folder():
            context.update(
                workspace_id=folder.workspace_id, binding_id=folder.binding_id
            )
        self.app.post_message(NavigateToScreen("settings", context))

    @on(Input.Submitted, "#local-files-query")
    @on(Button.Pressed, "#local-files-search")
    def _search(self) -> None:
        self._start()

    @on(Button.Pressed, "#local-files-ask")
    def _ask(self) -> None:
        if destination := self._destination():
            self._start(destination)

    def _start(self, destination: FileAnswerDestination | None = None) -> None:
        folder, query = self._folder(), self._query()
        if folder is None or self._folder_state not in _SEARCHABLE_STATES:
            return
        if not query:
            self._status("Enter a file name, search terms, or a question.")
            return
        self._clear_request()
        request = self._request = _Request(folder, query, self._mode(), destination)
        self._status("Searching the selected folder…")
        self.run_worker(
            self._retrieve(request), group="local-files-search", exclusive=True
        )

    def _is_current(self, request: _Request) -> bool:
        return (
            self.is_attached
            and self._request is request
            and next(
                (
                    screen
                    for screen in reversed(self.app.screen_stack)
                    if isinstance(screen, BaseAppScreen)
                ),
                None,
            )
            is self
            and self._folder() == request.folder
            and self._query() == request.query
            and self._mode() == request.mode
            and (
                request.destination is None
                or self._destination() == request.destination
            )
        )

    def _search_folder(
        self, request: _Request
    ) -> tuple[ProjectIndexService, BindingSearchResult]:
        service, key = self._folder_service(request.folder)
        query = request.query
        if request.destination is not None:
            from ...Utils.fts5_match_forms import fts5_query_tokens, is_fts5_stopword

            query = (
                " ".join(
                    token
                    for token in fts5_query_tokens(query)
                    if not is_fts5_stopword(token)
                )
                or query
            )
        search = service.search_binding(
            key,
            request.query,
            mode=request.mode,
            keyword_query=query if query != request.query else None,
            stop=request.stop,
        )
        return service, search

    async def _retrieve(self, request: _Request) -> None:
        for delay in (0, 0.05, 0.15, 0.3):
            if delay:
                await asyncio.sleep(delay)
            if not self._is_current(request):
                return
            try:
                service, search = await asyncio.to_thread(self._search_folder, request)
            except _INDEX_ERRORS as exc:
                if exc.args == ("project_index_search_busy",):
                    continue
                if self._is_current(request):
                    self._status(
                        "Search unavailable. Check the folder and its index in Folder settings."
                    )
                return
            else:
                break
        else:
            if self._is_current(request):
                self._status("The folder index is busy. Try again.")
            return
        if not self._is_current(request):
            return
        self._published = request, service, search
        self._show_results(search, request.mode)
        if request.destination is None:
            return
        result = search.result
        if not any(hit.excerpt.strip() for hit in result.hits):
            if result.status == "ready" and not result.refresh_required:
                self._status(
                    "No file excerpts matched this question. Try words used in the files."
                )
            return
        if request.destination.external:
            self._confirm_answer(request)
        else:
            self._start_answer(request, approved=False)

    def _show_results(self, search: BindingSearchResult, mode: str) -> None:
        result = search.result
        if result.coverage is not None:
            self._folder_state = result.coverage.state
            self._index_copy(
                f"Keyword index · {result.coverage.state} · {result.coverage.indexed:,} files"
            )
            self._set_controls()
        if result.status == "unavailable":
            copy = (
                "Semantic search unavailable. Check the folder's semantic index or select Keyword."
                if mode == "semantic"
                else "Search unavailable. Check the index in Folder settings."
            )
            self._status(copy)
            return
        count = len(result.hits)
        copy = (
            ("1 match" if count == 1 else f"{count} matches")
            if count
            else "No matches."
        )
        if result.status == "partial":
            copy += " · Partial index coverage"
        if mode == "hybrid" and result.semantic_status != "ready":
            copy += " · Keyword results only. Semantic index unavailable."
        if result.refresh_required:
            copy += " · Files changed. Refresh in Folder settings."
        self._status(copy)
        results = self.query_one("#local-files-results", OptionList)
        results.add_options(
            Option(
                Text(
                    f"{_location(hit)}\n{' '.join(hit.excerpt.split())[:180]}".rstrip()
                )
            )
            for hit in result.hits
        )
        results.display = bool(count)

    def _confirm_answer(self, request: _Request) -> None:
        from ...Widgets.confirmation_dialog import ConfirmationDialog

        destination = request.destination
        if destination is None:
            return
        if self.app.screen is not self:
            self._pending_confirmation = request
            return

        def finish(approved: bool | None) -> None:
            if not self._is_current(request):
                return
            if approved:
                self._start_answer(request, approved=True)
            else:
                self._status("Answer canceled. No question or excerpts were sent.")

        self.app.push_screen(
            ConfirmationDialog(
                title="Send question and file excerpts?",
                message=f"Send this question and bounded excerpts from the matched files to "
                f"{destination.label} for this answer?",
                confirm_label="Send and answer",
            ),
            finish,
        )

    def _start_answer(self, request: _Request, *, approved: bool) -> None:
        self._status("Generating an answer from matching file excerpts…")
        self.run_worker(
            self._answer(request, approved=approved),
            group="local-files-answer",
            exclusive=True,
        )

    async def _answer(self, request: _Request, *, approved: bool) -> None:
        from ...Workspaces.project_index.file_answers import answer_files

        published, destination = self._published, request.destination
        if published is None or published[0] is not request or destination is None:
            return
        _, service, search = published
        try:
            answer = await answer_files(
                query=request.query,
                search=search,
                destination=destination,
                approved=approved,
                app_config=lambda: self.app_instance.app_config,
                validate=service.validate_binding_hits,
                is_current=lambda: self._is_current(request),
            )
        except _INDEX_ERRORS:
            if self._is_current(request):
                self._status(
                    "Answer unavailable. Check the destination or try another search."
                )
            return
        if self._is_current(request):
            self._show_answer(search, answer)

    def _show_answer(
        self, search: BindingSearchResult, answer: LibraryRagAnswer
    ) -> None:
        from ...Chat.answer_citations import build_answer_citation_validation

        self._status(answer.error or answer.citation_recovery or "Answer")
        if not answer.text:
            return
        self.query_one("#local-files-answer", Static).update(Text(answer.text))
        validation = build_answer_citation_validation(
            answer.text, answer.evidence_bundle
        )
        hits = search.result.hits
        if answer.evidence_bundle is None:
            return
        options = []
        for ref in answer.evidence_bundle.references:
            if ref.evidence_id not in validation.cited_evidence_ids:
                continue
            hit_index = ref.metadata.get("hit_index")
            if type(hit_index) is int and 0 <= hit_index < len(hits):
                self._citations.append(hits[hit_index])
                options.append(Option(Text(f"[{ref.evidence_id}] {ref.title}")))
        citations = self.query_one("#local-files-citations", OptionList)
        citations.add_options(options)
        citations.display = bool(options)

    @on(OptionList.OptionSelected, ".local-files-sources")
    def _preview(self, event: OptionList.OptionSelected) -> None:
        if self._published is None:
            return
        hits = (
            self._citations
            if event.option_list.id == "local-files-citations"
            else self._published[2].result.hits
        )
        index = event.option_index
        if (
            0 <= index < len(hits)
            and event.option_list.get_option_at_index(index) is event.option
        ):
            self._open_hit(hits[index])

    @on(Button.Pressed, "#local-files-reveal")
    def _reveal(self) -> None:
        if self._preview_hit is not None:
            self._open_hit(self._preview_hit, reveal=True)

    def _open_hit(self, hit: ProjectIndexHit, *, reveal: bool = False) -> None:
        if self._published is None:
            return
        if reveal:
            self._action_serial += 1
        else:
            self._clear_preview()
        self.run_worker(
            self._inspect_hit(self._published, hit, self._action_serial, reveal),
            group="local-files-preview",
            exclusive=True,
        )

    async def _inspect_hit(
        self,
        published: _PublishedSearch,
        hit: ProjectIndexHit,
        serial: int,
        reveal: bool,
    ) -> None:
        request, service, search = published
        preview = self.query_one("#local-files-preview", Static)
        button = self.query_one("#local-files-reveal", Button)
        try:
            paths = await asyncio.to_thread(
                service.validate_binding_hits, search, (hit,)
            )
        except _INDEX_ERRORS:
            paths = None
        if not self._is_current(request) or serial != self._action_serial:
            return
        if not paths:
            self._clear_request()
            self._status(_STALE)
            return
        if reveal:
            try:
                opened = await asyncio.to_thread(_reveal_file, paths[0])
            except (OSError, subprocess.SubprocessError):
                opened = False
            if (
                self._is_current(request)
                and serial == self._action_serial
                and not opened
            ):
                self._status("File manager unavailable on this host.")
            return
        preview.update(
            Text(
                f"{_location(hit)}\n{hit.excerpt or 'No text excerpt for this match.'}"
            )
        )
        self._preview_hit = hit
        button.display = True
        button.disabled = False
        preview.scroll_visible()
