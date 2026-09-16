"""Recoverable authoring at the canonical workflows destination (ADR-138)."""

import asyncio
import sqlite3

from textual import on
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Input, Static, TextArea

from tldw_chatbook.UI.Navigation.base_app_screen import BaseAppScreen
from tldw_chatbook.UI.Workflows_Modules.console_context import WorkflowConsoleContext
from tldw_chatbook.UI.Workflows_Modules.controller import (
    WorkflowsController,
    step_label,
    visible_pane_ids,
)
from tldw_chatbook.UI.Workflows_Modules.editor import WorkflowEditor
from tldw_chatbook.UI.Workflows_Modules.library import (
    ChoiceModal,
    PagedChoiceModal,
    StepChooser,
    WorkflowLibrary,
    compact_button,
)
from tldw_chatbook.UI.Workflows_Modules.navigator import WorkflowNavigator
from tldw_chatbook.UI.Workflows_Modules.reference_picker import ReferencePicker
from tldw_chatbook.Widgets.workbench_focus import (
    WorkbenchPaneTarget,
    focus_relative_workbench_pane,
)
from tldw_chatbook.Workflows.document_service import PAGE_SIZE
from tldw_chatbook.Workflows.models import (
    DraftConflict,
    DraftWriteFailed,
    InvalidDraft,
    Issue,
    RevisionConflict,
)


class WorkflowsScreen(BaseAppScreen):
    """Consume app.workflow_documents and app.workflow_drafts.

    Owners initialize lazily on entry. Run is unavailable in this authoring slice.
    """

    def __init__(self, app_instance, **kwargs):
        super().__init__(app_instance, "workflows", **kwargs)
        documents = getattr(app_instance, "workflow_documents", None)
        drafts = getattr(app_instance, "workflow_drafts", None)
        self.controller = (
            WorkflowsController(documents, drafts)
            if documents is not None and drafts is not None
            else None
        )
        self._release = None
        self._busy = False
        self._loaded = False
        self._error = ""
        self._notice = ""
        self._raw_only_reason = ""
        self._run_reason = "Run unavailable in this authoring release. Sequential v1; branching v2; parallel v3."
        self._issue_index = 0
        self._workflow_views = {}
        self._recovery_was_locked = False
        self._refresh_lock = asyncio.Lock()

    def compose_content(self):
        with Vertical(id="workflow-authoring"):
            yield Static(
                "Workflows · Local authoring", id="workflows-title", markup=False
            )
            yield Static(
                "Loading local definitions…"
                if self.controller
                else "Not ready · authoring owner unavailable",
                id="workflow-draft-status",
                markup=False,
            )
            with Horizontal(id="workflow-action-bar"):
                yield compact_button(
                    "Save revision", "workflow-save-revision", disabled=True
                )
                yield compact_button("Validate", "workflow-validate", disabled=True)
                yield compact_button("Run", "workflow-run", disabled=True)
                yield compact_button("Import", "workflow-import")
                yield compact_button("Export", "workflow-export", disabled=True)
                yield compact_button(
                    "More…", "workflow-more", disabled=self.controller is None
                )
                yield compact_button(
                    "Retry", "workflow-retry-save", disabled=self.controller is None
                )
            yield Static(self._run_reason, id="workflow-readiness", markup=False)
            if self.controller is None:
                yield Static(
                    "The application has not supplied its Workflows document and draft services. Authoring and execution are not ready. Return after application setup completes.",
                    id="workflows-unavailable",
                    markup=False,
                )
                yield WorkflowConsoleContext(self.app_instance)
                return
            with Horizontal(id="workflow-selectors"):
                yield compact_button("Workflow…", "workflow-library-selector")
                yield compact_button("Step / Overview…", "workflow-step-selector")
            with Horizontal(id="workflow-panes"):
                yield WorkflowLibrary(id="workflows-library")
                yield WorkflowNavigator(id="workflows-navigator")
                yield WorkflowEditor(self.controller.documents, id="workflows-editor")
            yield WorkflowConsoleContext(self.app_instance)

    def _saved_selection(self):
        selection = self.state_data.get("selection")
        owner = self.controller.drafts
        saved_version = self.state_data.get("owner_version")
        if (
            owner.current is not None
            and saved_version is not None
            and saved_version != owner.confirmation_version
        ):
            return None
        return tuple(selection) if selection else None

    async def on_mount(self):
        if self.controller is None:
            await self._initialize_authoring()
        if self.controller is None:
            return
        self._recovery_was_locked = self.controller.drafts.editing_locked
        self._release = self.controller.drafts.subscribe(self._owner_changed)
        try:
            await self.controller.load(self._saved_selection())
            self.controller.section = self.state_data.get("section", "overview")
            self._workflow_views = self.state_data.get("workflow_views", {})
            self.query_one(WorkflowEditor).views = self.state_data.get("views", {})
            self._loaded = True
            self.call_after_refresh(self._refresh_authoring)
        except Exception:  # noqa: BLE001 -- bounded UI recovery; never expose private store paths
            self._error = "Definitions could not be loaded. Retry without changing the selected identity."
            self.call_after_refresh(self._show_status)
        self.call_after_refresh(self._layout_panes)

    async def _initialize_authoring(self):
        ensure = getattr(self.app_instance, "ensure_workflow_authoring", None)
        if not callable(ensure):
            return
        try:
            await ensure()
            self.controller = WorkflowsController(
                self.app_instance.workflow_documents, self.app_instance.workflow_drafts
            )
            await self.recompose()
        except Exception:  # noqa: BLE001 -- bounded setup failure keeps Retry available
            self.query_one("#workflow-draft-status", Static).update(
                "Local authoring could not open. Retry to open the same store."
            )
            self.query_one("#workflow-retry-save", Button).disabled = False

    def on_unmount(self):
        if self._release:
            self._release()
            self._release = None

    def _owner_changed(self):
        self._show_status()
        locked = self.controller.drafts.editing_locked
        if (
            self._recovery_was_locked
            and not locked
            and self.is_mounted
            and self._loaded
        ):
            self.controller.validate()
            self._error = ""
            self.call_after_refresh(self._refresh_authoring)
        self._recovery_was_locked = locked

    def _show_status(self):
        if not self.is_mounted or self.controller is None:
            return
        controller = self.controller
        draft = controller.draft
        locked = controller.drafts.editing_locked
        stale = bool(
            draft
            and controller.head
            and draft.base_revision_id != controller.head.revision_id
        )
        state = controller.drafts.status
        if controller.inspection:
            state = "Inspecting saved revision · read-only"
        elif draft:
            if state in {
                "Saved locally",
                "Draft recovered",
                "Based on saved revision",
                "Revision saved",
            }:
                state = (
                    "Saved revision · draft unchanged"
                    if draft.raw_text == controller.drafts.base.raw_json
                    else "Draft stored · not a saved revision"
                )
            elif state.startswith("Pending"):
                state = "Draft pending · not stored yet"
            else:
                state = "Draft · " + state
        if stale and not locked and not controller.inspection:
            state += " · Older-base draft; More → Copy draft onto saved head"
        status = self._error or (
            state
            + (
                " · " + draft.error + "; forms show last valid structure"
                if draft and draft.error
                else ""
            )
        )
        self.query_one("#workflow-draft-status", Static).update(status)
        save = self.query_one("#workflow-save-revision", Button)
        label = (
            "Recover draft…" if stale and not controller.inspection else "Save revision"
        )
        if save.label.plain != label:
            save.label = label
            save.tooltip = label
            save.refresh(layout=True)
        self.query_one("#workflow-export", Button).disabled = not bool(
            controller.inspection or controller.head
        )
        save.disabled = (
            self._busy
            or locked
            or not draft
            or (bool(draft.error) and not stale)
            or bool(controller.inspection)
            or bool(self._raw_only_reason)
        )
        self.query_one("#workflow-validate", Button).disabled = locked or not draft
        for identifier in (
            "workflows-editor",
            "workflows-library",
            "workflows-navigator",
            "workflow-more",
            "workflow-library-selector",
            "workflow-step-selector",
        ):
            self.query_one("#" + identifier).disabled = locked
        self.query_one("#workflow-run", Button).disabled = True
        self.query_one("#workflow-add-step", Button).disabled = (
            locked or bool(self._raw_only_reason) or bool(controller.inspection)
        )
        self.query_one("#workflow-retry-save", Button).display = (
            not self._loaded or controller.drafts.status.startswith("Not saved")
        )
        readiness = self._notice or self._run_reason
        if controller.issues:
            readiness = (
                f"{len(controller.issues)} authoring issue(s) · Validate to inspect. "
                + readiness
            )
        self.query_one("#workflow-readiness", Static).update(readiness)

    async def _refresh_authoring(self, *, rebuild=True, focus=False):
        # Owner callbacks and control workers can arrive during awaited mounts.
        # Serialize only widget reconciliation, never storage or execution.
        async with self._refresh_lock:
            await self._render_authoring(rebuild=rebuild, focus=focus)

    async def _render_authoring(self, *, rebuild=True, focus=False):
        controller = self.controller
        if controller is None or not self.is_mounted:
            return
        self._notice = ""
        if rebuild:
            await self._show_library()
        draft = controller.draft
        self._raw_only_reason = ""
        if draft:
            try:
                document = controller.documents.project(draft.last_valid_json)
            except InvalidDraft as error:
                self._raw_only_reason = str(error)
                document = {"steps": [], "name": "Raw inspection only"}
                controller.section = "overview"
            if controller.section.startswith("step:") and not any(
                step["id"] == controller.section[5:] for step in document["steps"]
            ):
                controller.section = "overview"
            self.query_one("#workflows-title", Static).update(
                "Workflows / "
                + str(document.get("name", "Untitled workflow"))
                + " · Local"
            )
            self.query_one(WorkflowNavigator).show_document(
                document, controller.section, controller.issues, rebuild=rebuild
            )
            await self.query_one(WorkflowEditor).render_draft(
                draft,
                section=controller.section,
                read_only=bool(controller.inspection),
                rebuild=rebuild,
                focus=focus,
                field_edit=None
                if controller.inspection
                else controller.drafts.field_edit,
                raw_repair_required=not controller.inspection
                and controller.drafts.raw_repair_required,
                raw_only_reason=self._raw_only_reason or None,
            )
        elif rebuild:
            self.query_one("#workflow-editor-heading", Static).update(
                "No workflows yet · New workflow opens a named draft"
            )
        self._show_status()
        self.call_after_refresh(self._layout_panes)

    def on_resize(self):
        self.call_after_refresh(self._layout_panes)

    async def _show_library(self):
        controller = self.controller
        if self.is_mounted:
            self.query_one(WorkflowLibrary).show_rows(
                controller.library_rows,
                offset=controller.library_offset,
                has_next=controller.library_has_next,
            )

    @on(WorkflowLibrary.PageRequested)
    def library_page_requested(self, event: WorkflowLibrary.PageRequested) -> None:
        event.stop()
        self.run_worker(
            self._library_page(event.offset, event.query),
            group="workflow-library-page",
            exclusive=True,
        )

    async def _library_page(self, offset, query):
        try:
            await self.controller.load_library(offset, query)
            await self._show_library()
            if self._error.startswith("Unable to load the library page."):
                self._error = ""
                self._show_status()
        except (OSError, RuntimeError, ValueError, sqlite3.Error):
            self._error = "Unable to load the library page. Change the search or reopen Workflows to retry."
            self._show_status()

    def _layout_panes(self):
        if not self.is_mounted or self.controller is None:
            return
        self.set_class(self.size.height < 30, "workflow-short")
        visible = visible_pane_ids(self.query_one("#workflow-panes").content_size.width)
        focused = self.app.focused
        for pane_id, selector in (
            ("workflows-library", "workflow-library-selector"),
            ("workflows-navigator", "workflow-step-selector"),
        ):
            pane = self.query_one("#" + pane_id)
            hiding_focus = focused is pane or (
                focused is not None and pane in focused.ancestors
            )
            pane.display = pane_id in visible
            button = self.query_one("#" + selector)
            button.display = not pane.display
            if not pane.display and hiding_focus:
                button.focus()
        self.query_one("#workflow-selectors").display = len(visible) != 3

    def action_focus_next_workbench_pane(self):
        """Delegate consumed by the existing app-global F6; no local binding."""
        if not self.controller:
            return
        visible = visible_pane_ids(self.query_one("#workflow-panes").content_size.width)
        preferred = {
            "workflows-library": ("workflow-library-search",),
            "workflows-navigator": ("workflow-navigation-list",),
            "workflows-editor": ("workflow-editor-heading",),
        }
        target = focus_relative_workbench_pane(
            self,
            (WorkbenchPaneTarget(pane, preferred[pane]) for pane in visible),
            direction=1,
        )
        if target:
            target.scroll_visible(animate=False)

    async def flush_pending_work(self) -> bool:
        """Existing awaited navigation veto. Retry or explicit loss stays visible."""
        try:
            owner = getattr(self.app_instance, "_workflow_authoring", None)
            if owner is not None:
                await owner.flush()
            elif self.controller and self.controller.drafts.current:
                await self.controller.drafts.flush()
            return True
        except DraftWriteFailed:
            self._error = "Not saved locally. Retry, or More → Discard pending changes to leave without recovery of those changes."
            self._show_status()
            return False

    def _start(self, operation):
        if self._busy:
            operation.close()
            return
        self._busy = True
        self._notice = ""
        self._show_status()

        async def run():
            try:
                await operation
            except (ValueError, DraftWriteFailed) as error:
                self._error = str(error)
            except Exception:  # noqa: BLE001 -- preserve draft and show a content-free owner failure
                self._error = (
                    "Workflows operation failed. The current draft is retained."
                )
            finally:
                self._busy = False
                self._show_status()

        self.run_worker(run(), group="workflows-controls", exclusive=True)

    def _open_choice(self, modal, callback, opener=None):
        opener = opener or self.app.focused
        opener_id = opener.id if opener else None
        selection = opener.selection if isinstance(opener, TextArea) else None
        cursor = opener.cursor_position if isinstance(opener, Input) else None

        def restore(result):
            if not self.is_mounted:
                return
            target = (
                opener
                if opener and opener.is_attached
                else next(iter(self.query("#" + opener_id)), None)
                if opener_id
                else None
            )
            if (
                not target
                or target.disabled
                or not target.can_focus
                or not all(parent.display for parent in (target, *target.ancestors))
            ):
                target = (
                    self.query_one("#workflow-step-selector")
                    if self.query_one("#workflow-step-selector").display
                    else self.query_one("#workflow-editor-heading")
                )
            target.focus()
            target.scroll_visible(animate=False)
            if isinstance(target, TextArea) and selection:
                target.selection = selection
            elif isinstance(target, Input) and cursor is not None:
                target.cursor_position = cursor
            if result is not None:
                callback(result)

        self.app.push_screen(modal, restore)

    async def _select_section(self, section):
        await self.controller.select_section(section)
        self._error = ""
        await self._refresh_authoring(focus=True)

    @on(WorkflowNavigator.Selected)
    @on(WorkflowEditor.Selected)
    def select_section(self, event):
        event.stop()
        self._start(self._select_section(event.section))

    async def _select_workflow(self, workflow_id, revision_id):
        editor = self.query_one(WorkflowEditor)
        editor.capture_view()
        if self.controller.draft:
            self._workflow_views[self.controller.draft.workflow_id] = {
                "section": self.controller.section,
                "views": editor.views,
            }
        await self.controller.select_workflow(workflow_id, revision_id)
        self._error = ""
        previous = self._workflow_views.get(workflow_id, {})
        self.controller.section = previous.get("section", "overview")
        editor.views = previous.get("views", {})
        await self._refresh_authoring(focus=True)

    @on(WorkflowLibrary.Selected)
    def select_workflow(self, event):
        event.stop()
        self._start(self._select_workflow(event.workflow_id, event.revision_id))

    @on(WorkflowEditor.RawEdited)
    async def raw_changed(self, event):
        event.stop()
        current = self.controller.drafts.current
        if (
            self.controller.inspection
            or not current
            or event.identity != (current.workflow_id, current.base_revision_id)
        ):
            return
        try:
            self.controller.drafts.update(event.text)
            self.controller.validate()
            self._error = ""
            # Keep the raw TextArea itself; reconcile its sibling form schema,
            # values, summaries and navigation against the new projection.
            await self._refresh_authoring(rebuild=True)
        except (ValueError, DraftWriteFailed) as error:
            self._error = str(error)
            self._show_status()

    @on(WorkflowEditor.FieldEdited)
    async def field_changed(self, event):
        event.stop()
        current = self.controller.drafts.current
        if current is None or event.identity != (
            current.workflow_id,
            current.base_revision_id,
        ):
            return
        pointer = event.pointer
        if event.step_id is not None:
            steps = self.controller.documents.project(current.last_valid_json)["steps"]
            index = next(
                (i for i, step in enumerate(steps) if step["id"] == event.step_id), None
            )
            if index is None:
                return
            pointer = f"/steps/{index}/" + pointer.split("/", 3)[3]
        try:
            self.controller.edit_field(pointer, event.text, as_json=event.as_json)
            self._error = ""
            await self._refresh_authoring(rebuild=False)
        except (ValueError, DraftWriteFailed) as error:
            self._error = str(error)
            self._show_status()

    @on(WorkflowEditor.ReferenceRequested)
    def reference(self, event):
        event.stop()
        section = self.controller.section
        if not section.startswith("step:"):
            return
        draft = self.controller.draft
        identity = (draft.workflow_id, draft.base_revision_id)

        def insert(expression):
            if isinstance(event.opener, Input):
                event.opener.insert_text_at_cursor(expression)
                text = event.opener.value
            elif isinstance(event.opener, TextArea):
                selection = event.opener.selection
                event.opener.replace(expression, selection.start, selection.end)
                text = event.opener.text
            else:
                return
            self.post_message(
                WorkflowEditor.FieldEdited(
                    event.pointer, text, False, identity, section[5:]
                )
            )

        self._open_choice(
            ReferencePicker(
                self.controller.documents.project(draft.last_valid_json),
                section[5:],
                event.value_type,
            ),
            insert,
            event.opener,
        )

    async def _save(self):
        if (
            self.controller.head
            and self.controller.draft.base_revision_id
            != self.controller.head.revision_id
        ):
            await self._recovery_dialog()
            return
        try:
            await self.controller.drafts.save_revision()
        except RevisionConflict:
            await self.controller.refresh_head()
            raise
        await self.controller.refresh_library()
        self._error = ""
        await self._refresh_authoring()

    async def _retry(self):
        if not self._loaded:
            await self.controller.load(self._saved_selection())
            self._loaded = True
        elif self.controller.drafts.current:
            await self.controller.drafts.flush()
        self._error = ""
        await self._refresh_authoring()

    async def _create(self, name):
        owner = getattr(self.app_instance, "_workflow_authoring", None)
        if owner:
            revision = await owner.create(name)
            await self.controller.load((revision.workflow_id, revision.revision_id))
        else:
            await self.controller.create(name)
        self._error = ""
        await self._refresh_authoring(focus=True)

    async def _import_file(self):
        from tldw_chatbook.Widgets.enhanced_file_picker import EnhancedFileOpen

        owner = getattr(self.app_instance, "_workflow_authoring", None)
        if owner is None:
            raise DraftWriteFailed(
                "Local file exchange is unavailable; reopen Workflows."
            )
        selected = await self.app.push_screen_wait(
            EnhancedFileOpen(
                title="Import workflow JSON (up to 16 MiB)",
                filters=["*.json"],
                context="workflow_import",
                select_button="Import",
            )
        )
        if selected is None:
            return
        revision = await owner.import_file(selected)
        await self.controller.load((revision.workflow_id, revision.revision_id))
        self._error = ""
        await self._refresh_authoring(focus=True)
        self.notify(
            "Imported locally. Review prompts and opaque metadata before sharing.",
            severity="information",
        )

    async def _export_file(self):
        from tldw_chatbook.Widgets.enhanced_file_picker import EnhancedFileSave

        owner = getattr(self.app_instance, "_workflow_authoring", None)
        revision = self.controller.inspection or self.controller.head
        if owner is None or revision is None:
            raise DraftWriteFailed("Select a saved definition before exporting.")
        confirmed = await self.app.push_screen_wait(
            ChoiceModal(
                "Review before sharing",
                (("Export saved definition", "export"),),
                detail=(
                    "Only the selected saved revision is exported; pending edits stay local. "
                    "Prompts, inputs and opaque metadata may contain secrets. "
                    "Review Advanced JSON before sharing. Preservation is not a secrets check."
                ),
            )
        )
        if confirmed != "export":
            return
        selected = await self.app.push_screen_wait(
            EnhancedFileSave(
                title="Export saved workflow",
                default_filename="workflow.json",
                filters=["*.json"],
                context="workflow_export",
                select_button="Export",
            )
        )
        if selected is None:
            return
        if await asyncio.to_thread(selected.exists):
            replacement = await self.app.push_screen_wait(
                ChoiceModal(
                    "Replace selected JSON file?",
                    (("Replace with saved definition", "replace"),),
                    detail="The selected file already exists. Replace its contents with this saved revision? Cancel leaves it unchanged.",
                )
            )
            if replacement != "replace":
                return
        await owner.export_file(selected, revision)
        self._error = ""
        self.notify("Saved definition exported.", severity="information")

    def _step_chooser(self, *, before=False):
        step_id = (
            self.controller.section[5:]
            if self.controller.section.startswith("step:")
            else ""
        )

        async def insert(step_type):
            draft = self.controller.change_steps(
                "add", step_id, step_type=step_type, before=before
            )
            old_ids = {
                step["id"] for step in self.query_one(WorkflowEditor).document["steps"]
            }
            new = next(
                step
                for step in self.controller.documents.project(draft.last_valid_json)[
                    "steps"
                ]
                if step["id"] not in old_ids
            )
            self.controller.section = "step:" + new["id"]
            await self._refresh_authoring(focus=True)

        self._open_choice(StepChooser(), lambda value: self._start(insert(value)))

    def _structural_action(self, operation, **options):
        step_id = self.controller.section[5:]
        draft = self.controller.draft
        version = self.controller.drafts.confirmation_version
        try:
            preview = self.controller.documents.edit_steps(
                draft.raw_text, operation, step_id, **options
            )
        except ValueError as error:
            message = str(error)
            self._open_choice(
                ChoiceModal(
                    "Repair dependencies before changing steps",
                    (("Repair in Advanced JSON", "repair"),),
                    detail=message,
                ),
                lambda _: self.query_one(WorkflowEditor).show_issue(
                    Issue("", "repair", message)
                ),
            )
            return
        detail = f"{operation.capitalize()} selected step {step_id}? References and explicit success routes remain unchanged."
        if operation == "move":
            detail += "\nNew order: " + " → ".join(
                step["id"]
                for step in self.controller.documents.project(preview)["steps"]
            )
            dependencies = self.controller.documents.step_dependencies(preview)
            detail += "\n" + (
                "\n".join(
                    issue.pointer + ": " + issue.message for issue in dependencies
                )
                or "No earlier-step references or explicit success routes."
            )

        async def apply(_):
            if (
                self.controller.draft != draft
                or self.controller.drafts.confirmation_version != version
                or self.controller.section != "step:" + step_id
            ):
                raise DraftConflict(
                    "Draft or selection changed; preview the step change again"
                )
            self.controller.change_steps(operation, step_id, **options)
            if operation == "delete":
                self.controller.section = "overview"
            await self._refresh_authoring(focus=True)

        self._open_choice(
            ChoiceModal(
                "Confirm step change",
                ((operation.capitalize(), "confirm"),),
                detail=detail,
                scroll_detail=True,
            ),
            lambda value: self._start(apply(value)),
        )

    def _more(self):
        choices = [
            ("New workflow", "new"),
            ("Add step", "add"),
            ("Inspect saved revision", "versions"),
            ("Discard draft · return to base", "discard"),
            ("Next issue", "next-issue"),
            ("Previous issue", "previous-issue"),
        ]
        if self.controller.draft:
            choices += [
                ("Open local drafts…", "local-drafts"),
                ("Copy draft onto saved head…", "recover"),
            ]
        if self.controller.drafts.status.startswith(
            "Not saved"
        ) or self._error.startswith("Not saved"):
            choices.append(("Discard pending changes · last durable draft", "pending"))
        if self.controller.inspection:
            choices = [
                ("Return to current draft", "return"),
                ("Edit as new revision", "edit-history"),
            ]
        if self._raw_only_reason:
            choices = [
                choice
                for choice in choices
                if choice[1]
                not in {"add", "discard", "pending", "recover", "edit-history"}
            ]
        self._open_choice(
            ChoiceModal("Workflow actions", tuple(choices)), self._more_selected
        )

    def _more_selected(self, action):
        if self._raw_only_reason and action in {
            "add",
            "discard",
            "pending",
            "recover",
            "edit-history",
        }:
            return
        if action == "new":
            self._open_choice(
                ChoiceModal(
                    "New workflow",
                    (("Create named workflow", "create"),),
                    text_input=True,
                ),
                lambda name: self._start(self._create(name)),
            )
        elif action == "add":
            self._step_chooser()
        elif action == "local-drafts":
            self._start(self._local_drafts())
        elif action == "recover":
            self._start(self._recovery_dialog())
        elif action in ("discard", "pending"):

            async def discard():
                if action == "discard":
                    await self.controller.drafts.discard_draft()
                else:
                    await self.controller.drafts.discard_pending()
                self._error = ""
                self.controller.validate()
                await self._refresh_authoring()

            self._open_choice(
                ChoiceModal(
                    "Discard draft?"
                    if action == "discard"
                    else "Lose pending changes?",
                    (("Confirm discard", "yes"),),
                    detail="Return to the saved base revision."
                    if action == "discard"
                    else "Only the last durable buffer will remain. Pending changes cannot be recovered.",
                ),
                lambda _: self._start(discard()),
            )
        elif action == "versions" and self.controller.draft:
            workflow_id = self.controller.draft.workflow_id

            def history(offset, query):
                return tuple(
                    (
                        f"v{offset + i + 1} · {revision.revision_id}",
                        revision.revision_id,
                    )
                    for i, revision in enumerate(
                        self.controller.documents.list_revisions(
                            workflow_id, page_size=PAGE_SIZE + 1, offset=offset
                        )
                    )
                )

            self._open_choice(
                PagedChoiceModal(
                    "Inspect a saved revision",
                    history,
                    detail="Inspection flushes the draft and opens a separate read-only projection.",
                ),
                lambda rid: self._start(self._inspect(rid)),
            )
        elif action == "return":
            self.controller.inspection = None
            self._error = ""
            self.controller.validate()
            self._start(self._refresh_authoring(focus=True))
        elif action == "edit-history":
            self._start(self._edit_history())
        elif action in ("next-issue", "previous-issue"):
            self._start(self._activate_issue(1 if action == "next-issue" else -1))

    async def _local_drafts(self):
        await self.controller.drafts.flush()
        workflow_id = self.controller.drafts.current.workflow_id

        def local_drafts(offset, query):
            return tuple(
                (
                    ("Repair raw JSON" if draft.error else "Local draft")
                    + " · base "
                    + draft.base_revision_id,
                    draft.base_revision_id,
                )
                for draft in self.controller.documents.list_drafts(
                    workflow_id, page_size=PAGE_SIZE + 1, offset=offset
                )
            )

        self._open_choice(
            PagedChoiceModal(
                "Open a local draft",
                local_drafts,
                detail="These are durable local buffers, including older bases. Opening preserves every other draft; it does not merge or overwrite.",
            ),
            lambda rid: self._start(self._select_workflow(workflow_id, rid)),
        )

    async def _recovery_dialog(self):
        owner = self.controller.drafts
        source, version = owner.current, owner.confirmation_version
        if source is None or self.controller.inspection:
            return
        await self.controller.refresh_head()
        head = self.controller.head
        if owner.current != source or owner.confirmation_version != version:
            raise DraftConflict(
                "Draft or selection changed; open a new recovery confirmation"
            )
        if head is None or head.revision_id == source.base_revision_id:
            self._error = "This draft is already on the saved head. Use Save revision."
            return
        if source.error:
            self._open_choice(
                ChoiceModal(
                    "Repair raw JSON before copying",
                    (("Repair raw JSON", "repair"),),
                    detail="Your exact text and last valid structure stay on the original base. Repair the text, then confirm Copy draft onto saved head.",
                ),
                lambda _: self.query_one(WorkflowEditor).show_issue(
                    Issue("", "invalid_json", source.error)
                ),
            )
            return
        target = await asyncio.to_thread(
            self.controller.documents.get_draft, head.workflow_id, head.revision_id
        )
        if owner.current != source or owner.confirmation_version != version:
            raise DraftConflict(
                "Draft or selection changed; open a new recovery confirmation"
            )
        if target and (
            target.error
            or target.raw_text != head.raw_json
            or target.last_valid_json != head.raw_json
        ):
            self._open_choice(
                ChoiceModal(
                    "Saved head has a local draft",
                    (("Open head draft without overwriting", "open-head"),),
                    detail="Copy is blocked. Both drafts are retained. Open the head draft to inspect, save or explicitly discard its changes.",
                ),
                lambda _: self._start(
                    self._select_workflow(head.workflow_id, head.revision_id)
                ),
            )
            return
        self._open_choice(
            ChoiceModal(
                "Copy draft onto saved head?",
                (("Copy draft", "copy"),),
                detail=f"Source base: {source.base_revision_id}\nTarget head: {head.revision_id}\nCopies content, changes base identity; no merge. Keeps the original local draft and all saved revisions.",
            ),
            lambda _: self._start(self._recover(source, head, version)),
        )

    async def _recover(self, source, head, version):
        await self.controller.drafts.recover_to_head(
            source, head, confirmation_version=version
        )
        self._error = ""
        self.controller.validate()
        # The owner subscription refreshes on settlement, also when this caller
        # is cancelled. Do not run a second concurrent form rebuild here.

    async def _inspect(self, revision_id):
        await self.controller.inspect_revision(
            self.controller.draft.workflow_id, revision_id
        )
        self._error = ""
        self.controller.validate()
        await self._refresh_authoring(focus=True)

    async def _edit_history(self):
        await self.controller.edit_inspected_revision()
        self._error = ""
        self.controller.validate()
        await self._refresh_authoring(focus=True)

    def _repair_raw_dialog(self):
        owner = self.controller.drafts
        source, version = owner.current, owner.confirmation_version
        self._open_choice(
            ChoiceModal(
                "Accept repaired whole document?",
                (("Accept Advanced JSON as the whole document", "accept"),),
                detail="This validates all raw JSON, not just the former field. Inspect any sibling or metadata edits first. Cancel keeps the protected bytes and last-valid forms.",
            ),
            lambda _: self._start(
                owner.repair_raw(source, confirmation_version=version)
            ),
        )

    async def _activate_issue(self, direction=0):
        issues = self.controller.validate()
        self._notice = ""
        if not issues:
            self._notice = (
                "Structure valid locally. Run is unavailable in this authoring release."
            )
            self._show_status()
            return
        self._issue_index = (self._issue_index + direction) % len(issues)
        issue = issues[self._issue_index]
        if issue.pointer.startswith("/steps/"):
            part = issue.pointer.split("/")[2]
            index = int(part) if part.isdecimal() else -1
            document = self.controller.documents.project(
                self.controller.draft.last_valid_json
            )
            if 0 <= index < len(document["steps"]):
                await self.controller.select_section(
                    "step:" + document["steps"][index]["id"]
                )
                await self._refresh_authoring()
        elif issue.pointer.startswith("/inputs") or issue.pointer.startswith(
            "/metadata/tldw_workflow/input_schema"
        ):
            await self.controller.select_section("inputs")
            await self._refresh_authoring()
        elif issue.pointer.startswith("/metadata/tldw_workflow/requirements"):
            await self.controller.select_section("requirements")
            await self._refresh_authoring()
        self.query_one(WorkflowEditor).show_issue(issue)

    def save_state(self):
        if not self.controller or not self.controller.draft:
            return {}
        editor = self.query_one(WorkflowEditor)
        editor.capture_view()
        draft = self.controller.drafts.current
        return {
            "selection": (draft.workflow_id, draft.base_revision_id),
            "owner_version": self.controller.drafts.confirmation_version,
            "section": self.controller.section,
            "views": editor.views,
            "workflow_views": self._workflow_views,
        }

    @on(Button.Pressed)
    def workflow_button(self, event):
        if self.controller is None:
            if event.button.id == "workflow-retry-save":
                event.stop()
                self._start(self.on_mount())
            return
        identifier = event.button.id
        if not identifier or not identifier.startswith("workflow-"):
            return
        event.stop()
        actions = {
            "workflow-save-revision": self._save,
            "workflow-retry-save": self._retry,
            "workflow-validate": self._activate_issue,
            "workflow-import": self._import_file,
            "workflow-export": self._export_file,
        }
        if identifier in actions:
            self._start(actions[identifier]())
        elif identifier == "workflow-more":
            self._more()
        elif identifier == "workflow-repair-raw":
            self._repair_raw_dialog()
        elif identifier == "workflow-new":
            self._more_selected("new")
        elif identifier in ("workflow-add-step", "workflow-add-first"):
            self._step_chooser()
        elif identifier == "workflow-library-selector":

            def library(offset, query):
                page = self.controller.documents.list_workflow_summaries(
                    page_size=PAGE_SIZE + 1, offset=offset, query=query
                )
                return tuple(
                    (label + " · Local", wid + ":" + rid) for label, wid, rid in page
                )

            def chosen(wid):
                if wid == "__new_workflow__":
                    self._more_selected("new")
                else:
                    self._start(self._select_workflow(*wid.split(":")))

            self._open_choice(
                PagedChoiceModal(
                    "Workflow library",
                    library,
                    searchable=True,
                    new_workflow=True,
                ),
                chosen,
            )
        elif identifier == "workflow-step-selector":
            choices = [
                ("Overview", "overview"),
                ("Run inputs", "inputs"),
                ("Requirements", "requirements"),
                ("Versions", "versions"),
            ]
            choices += [
                (step_label(step) + " · " + step["id"], "step:" + step["id"])
                for step in self.query_one(WorkflowEditor).document["steps"]
            ]
            self._open_choice(
                ChoiceModal("Step / Workflow sections", tuple(choices)),
                lambda section: self._start(self._select_section(section)),
            )
        elif identifier == "workflow-insert":
            self._open_choice(
                ChoiceModal(
                    "Insert",
                    (
                        ("Add before", "before"),
                        ("Add after", "after"),
                        ("Duplicate selected step", "duplicate"),
                    ),
                ),
                lambda choice: (
                    self._structural_action("duplicate")
                    if choice == "duplicate"
                    else self._step_chooser(before=choice == "before")
                ),
            )
        elif identifier == "workflow-move":
            self._open_choice(
                ChoiceModal(
                    "Move", (("Move earlier", "earlier"), ("Move later", "later"))
                ),
                lambda choice: self._structural_action(
                    "move", offset=-1 if choice == "earlier" else 1
                ),
            )
        elif identifier == "workflow-delete":
            self._structural_action("delete")
        elif identifier == "workflow-versions":
            self._more_selected("versions")
        elif identifier == "workflow-return-draft":
            self._more_selected("return")
