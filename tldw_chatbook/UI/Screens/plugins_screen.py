"""Source-qualified plugin browsing on the existing app-owned service."""

from __future__ import annotations

import asyncio
from copy import copy
from pathlib import Path
from typing import ClassVar
from uuid import uuid4

from rich.text import Text
from textual import on
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Button, Input, OptionList, Select, Static
from textual.widgets.option_list import Option

from ..Navigation.base_app_screen import BaseAppScreen

PAGE_SIZE = 25


class PluginsScreen(BaseAppScreen):
    """View state is disposable; source and plugin authority stay app-owned."""

    BINDINGS: ClassVar[list[Binding]] = [
        Binding("1", "plugins_view('installed')", "Installed", show=False),
        Binding("2", "plugins_view('browse')", "Browse", show=False),
        Binding("3", "plugins_view('marketplaces')", "Marketplaces", show=False),
        Binding("escape", "plugins_back", "Back", show=False),
        Binding("slash", "plugins_search", "Search", show=False),
    ]

    def __init__(self, app_instance):
        from ..Plugins_Modules.browse_state import PluginBrowseState

        super().__init__(app_instance, "plugins")
        self.browse = PluginBrowseState()
        self.rows = ()
        self._loading = False
        self._action_busy = False
        self._scope_generation = None
        self._source_service = None
        self._restore_focus = False

    @property
    def sources(self):
        if self._source_service is None:
            self._source_service = self.app_instance.plugin_source_service
        return self._source_service

    def compose_content(self):
        with Vertical(id="plugins-shell"):
            yield Static(
                "Plugins | Inspect sources, installation and activation",
                classes="section-title",
                markup=False,
            )
            yield Static("Scope: loading…", id="plugins-scope", markup=False)
            with Horizontal(id="plugins-modes"):
                for key, label in (
                    ("installed", "1 Installed"),
                    ("browse", "2 Browse"),
                    ("marketplaces", "3 Marketplaces"),
                ):
                    yield Button(label, id=f"plugins-{key}")
            with Horizontal(id="plugins-entry-actions"):
                yield Button("Repository…", id="plugins-repository")
                yield Button("Import…", id="plugins-import")
                yield Button("Add source…", id="plugins-add-source")
            with Horizontal(id="plugins-management-actions"):
                yield Button("Authority…", id="plugins-authority")
                yield Button("History…", id="plugins-history")
                yield Button("Candidates…", id="plugins-candidates")
                yield Button("Reset authority…", id="plugins-reset-authority")
            yield Input(
                self.browse.query,
                placeholder="Search plugins or sources (/)",
                id="plugins-search",
                max_length=1024,
                classes="form-input",
            )
            yield Static("Loading cached metadata…", id="plugins-status", markup=False)
            with Horizontal(id="plugins-panes"):
                with Vertical(id="plugins-list-pane"):
                    yield OptionList(id="plugins-list")
                    with Horizontal(id="plugins-pager"):
                        yield Button("Previous", id="plugins-previous", disabled=True)
                        yield Static("Page 1", id="plugins-page", markup=False)
                        yield Button("Next", id="plugins-next", disabled=True)
                with Vertical(id="plugins-detail-pane"):
                    yield Button("Back to list", id="plugins-back")
                    with VerticalScroll(id="plugins-detail-scroll"):
                        yield Static(
                            "Select a plugin or source to inspect its details.",
                            id="plugins-detail",
                            markup=False,
                        )
                        yield Static("", id="plugins-readme", markup=False)
                    with Horizontal(id="plugins-detail-actions"):
                        yield Button(
                            "Review preview…",
                            id="plugins-review-preview",
                            disabled=True,
                        )
                        yield Button(
                            "Preview — select entry",
                            id="plugins-preview-entry",
                            disabled=True,
                        )
                        yield Button(
                            "Refresh — select source",
                            id="plugins-refresh",
                            disabled=True,
                        )
                        yield Button(
                            "README — select installed",
                            id="plugins-read-readme",
                            disabled=True,
                        )
                        yield Button("Manage…", id="plugins-manage", disabled=True)
            with VerticalScroll(id="plugins-entry-form"):
                yield Static(
                    "", id="plugins-form-title", classes="section-title", markup=False
                )
                yield Static(
                    "Choose an explicit location. No account or directory is connected automatically.",
                    markup=False,
                )
                yield Input(
                    self.browse.location,
                    placeholder="Repository URL or absolute local directory",
                    id="plugins-location",
                    max_length=4096,
                    classes="form-input",
                )
                yield Input(
                    self.browse.ref,
                    placeholder="Git ref (HEAD)",
                    id="plugins-ref",
                    max_length=1024,
                    classes="form-input",
                )
                yield Select(
                    (("Cursor", "cursor"), ("Codex", "codex")),
                    value=self.browse.import_app,
                    allow_blank=False,
                    id="plugins-import-app",
                )
                yield Select(
                    (
                        ("Detect catalog", "auto"),
                        ("Agent Plugins", ".agents/plugins/marketplace.json"),
                        ("Cursor", ".cursor-plugin/marketplace.json"),
                        ("Claude", ".claude-plugin/marketplace.json"),
                    ),
                    value=self.browse.catalog_path,
                    allow_blank=False,
                    id="plugins-catalog",
                )
                with Horizontal(classes="plugins-form-actions"):
                    yield Button("Preview", id="plugins-submit")
                    yield Button("Review install…", id="plugins-review-install")
                    yield Button("Back", id="plugins-form-back")
                yield Static("", id="plugins-form-result", markup=False)

    def on_mount(self):
        self.register_footer_shortcuts(
            source="plugins",
            shortcuts=(("1-3", "view"), ("/", "search"), ("Esc", "back")),
        )
        registry = getattr(self.app_instance, "workspace_registry_service", None)
        self._scope_generation = getattr(registry, "mutation_generation", None)
        self._layout()
        self._reload()
        self.set_interval(0.5, self._check_scope_generation)

    def on_resize(self):
        self._layout()

    def _layout(self):
        if not self.query("#plugins-panes"):
            return
        narrow = self.size.width < 100
        list_focused = getattr(self.focused, "id", None) == "plugins-list"
        self.set_class(narrow, "plugins-narrow")
        form = self.browse.entry_action is not None
        self.query_one("#plugins-entry-form").display = form
        self.query_one("#plugins-panes").display = not form
        self.query_one("#plugins-search").display = not form
        self.query_one("#plugins-list-pane").display = not (
            narrow and self.browse.details_open
        )
        self.query_one("#plugins-detail-pane").display = (
            not narrow or self.browse.details_open
        )
        self.query_one("#plugins-back").display = narrow
        if form:
            self._render_form()
        elif narrow and self.browse.details_open and list_focused:
            self.call_after_refresh(self.query_one("#plugins-back").focus)

    def _check_scope_generation(self):
        registry = getattr(self.app_instance, "workspace_registry_service", None)
        generation = getattr(registry, "mutation_generation", None)
        if generation != self._scope_generation:
            self._scope_generation = generation
            self._reload()

    async def _workspace(self):
        registry = getattr(self.app_instance, "workspace_registry_service", None)
        if registry is None:
            return None
        active = await asyncio.to_thread(registry.get_active_workspace)
        return active.workspace_id if active is not None else None

    def _token(self):
        state = self.browse
        row = self._selected()
        revision = (
            getattr(row, "revision_digest", None),
            getattr(row, "digest", None),
            getattr(row, "source_id", None),
        )
        return (
            state.request_generation,
            state.workspace_id,
            state.view,
            state.query,
            state.page,
            state.selected_source_id,
            state.selected_installation_id,
            state.selected_entry_id,
            state.entry_action,
            state.location,
            state.ref,
            state.import_app,
            state.catalog_path,
            revision,
        )

    async def _current(self, token):
        workspace = await self._workspace()
        return self.is_running and token == self._token() and workspace == token[1]

    def _cancel_load(self):
        self.workers.cancel_group(self, "plugins-load")
        self._load_generation = None
        self._loading = False
        self._update_actions()

    def _reload(self):
        self.browse.request_generation += 1
        self._load_generation = self.browse.request_generation
        self._loading = True
        self.run_worker(
            self._load(self.browse.request_generation),
            group="plugins-load",
            exclusive=True,
        )

    async def _load(self, generation):
        from ..Plugins_Modules.browse_state import accept_response
        from ..Plugins_Modules.details import display_text

        if generation != self._load_generation:
            return
        self._status("Loading cached metadata…")
        self._update_actions()
        try:
            workspace = await self._workspace()
            if not self.is_running or not accept_response(
                generation, self.browse.request_generation
            ):
                return
            self.browse.workspace_id = workspace
            self.query_one("#plugins-scope", Static).update(
                "Activation in: "
                + display_text(workspace or "Global default", limit=256)
                + " | Installation: this device"
            )
            token = self._token()
            state = self.browse
            if state.view == "installed":
                rows = await self.sources.plugins.management_snapshot(
                    workspace,
                    query=state.query,
                    offset=state.page * PAGE_SIZE,
                    limit=PAGE_SIZE + 1,
                )
            else:
                await self.sources.load_cached()
                if state.view == "browse":
                    rows = self.sources.search(
                        state.query, offset=state.page * PAGE_SIZE, limit=PAGE_SIZE + 1
                    )
                else:
                    rows = tuple(
                        row
                        for row in self.sources.list_sources()
                        if state.query.casefold() in str(row.locator).casefold()
                    )[state.page * PAGE_SIZE : (state.page + 1) * PAGE_SIZE + 1]
            if not await self._current(token):
                return
            self._publish(rows)
        except (OSError, ValueError, PermissionError):
            if self.is_running and accept_response(
                generation, self.browse.request_generation
            ):
                self.rows = ()
                self.query_one("#plugins-list", OptionList).clear_options()
                self._status(
                    "Plugin metadata unavailable. Unlock plugin authority or resolve recovery, then reopen this view."
                )
                self.query_one("#plugins-detail", Static).update(
                    "No current details. Cached display state cannot authorize an operation."
                )
        finally:
            if generation == self._load_generation:
                self._load_generation = None
                self._loading = False
                self._update_actions()

    def _identity(self, row):
        return (
            getattr(row, "installation_id", None)
            or getattr(row, "identity", None)
            or row.source_id
        )

    def _selected_id(self):
        state = self.browse
        return {
            "installed": state.selected_installation_id,
            "browse": state.selected_entry_id,
            "marketplaces": state.selected_source_id,
        }[state.view]

    def _selected(self):
        return next(
            (row for row in self.rows if self._identity(row) == self._selected_id()),
            None,
        )

    def _publish(self, rows):
        from ..Plugins_Modules.details import display_text

        self.rows = tuple(rows[:PAGE_SIZE])
        listing = self.query_one("#plugins-list", OptionList)
        listing.clear_options()
        for row in self.rows:
            name = (
                getattr(row, "alias", None)
                or getattr(row, "name", None)
                or (row.locator.url if row.locator else "Unavailable source")
            )
            source = getattr(row, "source_id", None)
            listing.add_option(
                Option(
                    Text(
                        display_text(name, limit=256)
                        + ("\nSource: " + source if source else "")
                    ),
                    id=self._identity(row),
                )
            )
        selected = self._selected_id()
        listing.highlighted = next(
            (
                index
                for index, row in enumerate(self.rows)
                if self._identity(row) == selected
            ),
            None,
        )
        self.query_one("#plugins-page", Static).update(f"Page {self.browse.page + 1}")
        self.query_one("#plugins-previous", Button).disabled = self.browse.page == 0
        self.query_one("#plugins-next", Button).disabled = len(rows) <= PAGE_SIZE
        if self.rows:
            sources = (
                self.sources.list_sources() if self.browse.view != "installed" else ()
            )
            stale = any(row.cached and row.refresh_error for row in sources)
            self._status(
                "Showing stale cached results. Select a marketplace and Refresh."
                if stale
                else f"{len(self.rows)} results · Select with arrows and Enter"
            )
        elif self.browse.query:
            self._status("No matches. Change or clear the search.")
        elif self.browse.view == "installed":
            self._status(
                "No plugins installed. Choose Repository, Import, or Add source above."
            )
        elif not self.sources.list_sources():
            self._status(
                "No marketplace sources. Add a source or import a selected Cursor/Codex root."
            )
        else:
            self._status(
                "No cached entries. Select Marketplaces to inspect status and refresh a source."
            )
        self._render_details()
        focus = self.query(f"#{self.browse.focus_id}")
        if focus and (self._restore_focus or self.focused is None):
            focus.first().focus()
        self._restore_focus = False

    def _status(self, message):
        if not self.is_running:
            return
        from ..Plugins_Modules.details import display_text

        self.query_one("#plugins-status", Static).update(display_text(message))

    def _render_details(self):
        from ..Plugins_Modules.details import (
            entry_details,
            installed_details,
            source_details,
        )

        row = self._selected()
        text = "Select a plugin or source to inspect its details."
        if row:
            text = {
                "installed": installed_details,
                "browse": entry_details,
                "marketplaces": source_details,
            }[self.browse.view](row)
        self.query_one("#plugins-detail", Static).update(text)
        self.query_one("#plugins-readme", Static).update("")
        self._update_actions()

    def _update_actions(self):
        if not self.is_running:
            return
        if not self.query("#plugins-preview-entry"):
            return
        selected = self._selected()
        for identity, view, label, unavailable in (
            (
                "plugins-preview-entry",
                "browse",
                "Preview entry",
                "Preview — select entry",
            ),
            (
                "plugins-refresh",
                "marketplaces",
                "Refresh source",
                "Refresh — select source",
            ),
            (
                "plugins-read-readme",
                "installed",
                "Read README",
                "README — select installed",
            ),
            ("plugins-manage", "installed", "Manage…", "Manage — select installed"),
        ):
            button = self.query_one(f"#{identity}", Button)
            button.display = self.browse.view == view
            button.disabled = (
                not selected
                or self._loading
                or self._action_busy
                or (self.browse.view == "browse" and selected.support != "supported")
            )
            button.label = (
                "Busy…"
                if self._action_busy
                else label if not button.disabled else unavailable
            )
            if (
                self.browse.view == "browse"
                and selected
                and selected.support != "supported"
            ):
                button.label = "Preview unavailable — unsupported"
        self.query_one("#plugins-submit", Button).disabled = self._action_busy
        self.query_one("#plugins-review-install", Button).disabled = (
            self.browse.preview is None
        )
        self.query_one("#plugins-review-preview", Button).disabled = (
            self.browse.preview is None
        )

    @on(OptionList.OptionSelected, "#plugins-list")
    def _select_row(self, event):
        row = next(
            (item for item in self.rows if self._identity(item) == event.option.id),
            None,
        )
        if row is None:
            return
        self.browse.request_generation += 1
        self._loading = False
        if self.browse.view == "installed":
            self.browse.selected_installation_id = row.installation_id
        else:
            self.browse.selected_source_id = row.source_id
            if self.browse.view == "browse":
                self.browse.selected_entry_id = row.identity
        self.browse.details_open = True
        self._render_details()
        self._layout()
        if self.size.width < 100:
            self.query_one("#plugins-back").focus()

    @on(Input.Changed, "#plugins-search")
    def _search_changed(self, event):
        if self.browse.query != event.value:
            self.browse.query = event.value
            self.browse.page = 0
            self._reload()

    def action_plugins_view(self, view):
        self.browse.view = view
        self.browse.page = 0
        self.browse.details_open = False
        self.browse.entry_action = None
        self._layout()
        self._reload()

    def action_plugins_search(self):
        self.query_one("#plugins-search").focus()

    def action_plugins_back(self):
        self.browse.request_generation += 1
        self.browse.entry_action = None
        self.browse.details_open = False
        self._layout()
        self.query_one("#plugins-list").focus()
        self._reload()

    @on(Button.Pressed)
    def _button(self, event):
        identity = event.button.id or ""
        if not identity.startswith("plugins-"):
            return
        event.stop()
        action = identity.removeprefix("plugins-")
        if action in {"installed", "browse", "marketplaces"}:
            self.action_plugins_view(action)
        elif action in {"back", "form-back"}:
            self.action_plugins_back()
        elif action in {"manage", "review-install", "review-preview"}:
            from ...Widgets.plugin_review_modal import PluginReviewModal

            row = self._selected() if action == "manage" else None
            producer = (
                self.browse.preview
                if action in {"review-install", "review-preview"}
                else None
            )
            if row is not None or producer is not None:
                self.app.push_screen(
                    PluginReviewModal(
                        self.sources,
                        workspace_id=(
                            self.browse.workspace_id
                            if row
                            else self.browse.preview_workspace_id
                        ),
                        installation_id=row.installation_id if row else None,
                        producer=producer,
                    )
                )
        elif action in {"authority", "history", "candidates", "reset-authority"}:
            self.run_worker(
                self._management_entry(action),
                exclusive=True,
                group="plugins-management-entry",
            )
        elif action in {"previous", "next"}:
            self.browse.page += -1 if action == "previous" else 1
            self._reload()
        elif action in {"repository", "import", "add-source"}:
            self.browse.request_generation += 1
            self._cancel_load()
            self._status("Choose a source for the preview.")
            self.browse.entry_action = action
            self._layout()
            self.query_one("#plugins-location").focus()
        elif (
            action in {"refresh", "preview-entry", "read-readme", "submit"}
            and not self._action_busy
        ):
            self._capture_form()
            self.run_worker(
                self._perform(
                    action,
                    self._token(),
                    self._selected(),
                    copy(self.browse),
                    self.browse.catalog_path,
                ),
                group="plugins-action",
                exclusive=True,
            )

    async def _management_entry(self, action):
        from ...Widgets.plugin_review_modal import (
            PluginAuthorityDialog,
            PluginAuthorityResetModal,
            PluginStatusModal,
        )

        try:
            if action == "reset-authority":
                self.app.push_screen(PluginAuthorityResetModal(self.sources.plugins))
            elif action == "authority":
                posture = await self.sources.plugins.authority_status()
                if posture in {"needs_setup", "locked"}:
                    self.app.push_screen(
                        PluginAuthorityDialog(
                            self.sources.plugins,
                            "setup" if posture == "needs_setup" else "unlock",
                        ),
                        lambda ready: self._reload() if ready else None,
                    )
                elif posture in {"ready", "ready_reduced"}:
                    await self.sources.plugins.recover()
                    self._status(
                        "Authority recovered; saved-data recovery remains separately reviewed."
                    )
                    self._reload()
                else:
                    self._status(
                        f"Authority: {posture}. Restore the protected backend or wait for the owning instance; no automatic reset or takeover."
                    )
            else:
                self.app.push_screen(
                    PluginStatusModal(self.sources, candidates=action == "candidates")
                )
        except (OSError, ValueError, RuntimeError):
            self._status(
                "Authority unavailable. Check the protected backend and retry."
            )

    @on(Input.Changed, "#plugins-location")
    @on(Input.Changed, "#plugins-ref")
    @on(Select.Changed, "#plugins-import-app")
    @on(Select.Changed, "#plugins-catalog")
    def _form_changed(self, event):
        if self.is_mounted and self.browse.entry_action:
            previous = (
                self.browse.location,
                self.browse.ref,
                self.browse.import_app,
                self.browse.catalog_path,
            )
            self._capture_form()
            current = (
                self.browse.location,
                self.browse.ref,
                self.browse.import_app,
                self.browse.catalog_path,
            )
            if previous == current:
                return
            self.browse.request_generation += 1
            self.browse.preview = None
            self.query_one("#plugins-form-result", Static).update("")

    def _capture_form(self):
        self.browse.location = self.query_one("#plugins-location", Input).value
        self.browse.ref = self.query_one("#plugins-ref", Input).value
        self.browse.import_app = str(
            self.query_one("#plugins-import-app", Select).value
        )
        self.browse.catalog_path = str(self.query_one("#plugins-catalog", Select).value)

    def _render_form(self):
        action = self.browse.entry_action
        self.query_one("#plugins-form-title", Static).update(
            {
                "repository": "Preview selected repository",
                "import": "Preview Cursor/Codex import",
                "add-source": "Add marketplace source",
            }[action]
        )
        self.query_one("#plugins-ref").display = action != "import"
        self.query_one("#plugins-import-app").display = action == "import"
        self.query_one("#plugins-catalog").display = action == "add-source"
        self.query_one("#plugins-submit", Button).label = (
            "Add source" if action == "add-source" else "Preview"
        )

    async def _perform(self, action, token, row, draft, catalog):
        from ..Plugins_Modules.details import display_text

        focused_id = getattr(self.focused, "id", None)
        self._action_busy = True
        self._update_actions()
        self._status("Working on the captured source and scope…")
        try:
            if not await self._current(token):
                self._status("Workspace or selection changed. Reload and select again.")
                return
            result = None
            if action == "refresh" and row:
                result = await self.sources.refresh(row.source_id)
            elif action == "read-readme" and row:
                text = await self.sources.plugins.management_readme(
                    row.installation_id, row.revision_digest, workspace_id=token[1]
                )
                if await self._current(token):
                    self._status("README loaded from the selected retained revision.")
                    self.query_one("#plugins-readme", Static).update(
                        "\nREADME · untrusted literal text\n"
                        + display_text(
                            text or "No README.md in this revision.", limit=65536
                        )
                    )
                return
            elif action == "preview-entry" and row:
                from ...Plugins.git_source import SourceLocator

                locator = (
                    SourceLocator.model_validate_json(row.package_locator)
                    if row.package_locator
                    else None
                )
                result = await self.sources.review_entry(
                    row.source_id,
                    row.identity,
                    selection=(),
                    workspace_id=token[1],
                    selected_locator=locator,
                )
            elif action == "submit":
                state = draft
                if state.entry_action == "import":
                    from ...Plugins.imports import inspect_app_import

                    result = await asyncio.to_thread(
                        inspect_app_import, state.import_app, Path(state.location)
                    )
                else:
                    from urllib.parse import urlencode

                    from ...Plugins.git_source import (
                        parse_source_locator,
                        select_local_source,
                    )

                    location, ref = state.location.strip(), state.ref.strip() or "HEAD"
                    locator = (
                        await asyncio.to_thread(
                            select_local_source, Path(location), ref=ref
                        )
                        if Path(location).is_absolute()
                        else parse_source_locator(
                            location
                            + (
                                "#" + urlencode({"ref": ref})
                                if "#" not in location
                                else ""
                            )
                        )
                    )
                    if state.entry_action == "add-source":
                        result = await self.sources.add(
                            locator,
                            catalog_path=None if catalog == "auto" else str(catalog),
                        )
                    else:
                        operation = "ui-" + uuid4().hex
                        destination = (
                            await self.sources.plugins.acquisition_destination(
                                operation
                            )
                        )
                        result = await self.sources.plugins.acquire_snapshot(
                            locator, operation, destination
                        )
            if not await self._current(token):
                self._status(
                    "Result belongs to an earlier selection or workspace. Select again to continue."
                )
                return
            if action == "refresh" or (
                action == "submit" and self.browse.entry_action == "add-source"
            ):
                self.browse.entry_action = None
                self.browse.view = "marketplaces"
                self._layout()
                self._reload()
            elif result is not None:
                self.browse.preview = result
                self.browse.preview_workspace_id = token[1]
                inspection = getattr(result, "inspection", None)
                if inspection:
                    summary = (
                        f"Preview captured: {len(inspection.inventory)} components. Not installed or enabled.\n"
                        + "\n".join(
                            display_text(key, limit=256) for key in inspection.inventory
                        )
                    )
                else:
                    summary = (
                        f"Import preview: {len(result.candidates)} candidates. No sources or packages imported.\nExcluded: "
                        + ", ".join(result.exclusions)
                    )
                if self.browse.entry_action:
                    self.query_one("#plugins-form-result", Static).update(summary)
                else:
                    self.query_one("#plugins-detail", Static).update(summary)
                self._status(
                    "Preview retained for this session. Installation requires separate review."
                )
        except (OSError, ValueError, PermissionError):
            if token == self._token():
                self._status(
                    "Operation unavailable or source changed. Check the selected location, unlock authority if needed, and preview again."
                )
        finally:
            self._action_busy = False
            self._update_actions()
            if self.focused is None and focused_id and token == self._token():
                target = self.query(f"#{focused_id}")
                if target and target.first().focusable:
                    target.first().focus()

    def save_state(self):
        if self.query("#plugins-location"):
            self._capture_form()
        focused = self.focused
        if focused and focused.id:
            self.browse.focus_id = focused.id
        return {"browse": copy(self.browse)}

    def restore_state(self, state):
        from ..Plugins_Modules.browse_state import PluginBrowseState

        saved = state.get("browse")
        if isinstance(saved, PluginBrowseState):
            self.browse = copy(saved)
            self._restore_focus = True
            self.browse.request_generation += 1
