"""Focused coordinator for the reviewed one-time Database Notes import."""

from __future__ import annotations

import asyncio
import difflib
import inspect
import threading
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

from tldw_chatbook.Library.library_note_import_state import (
    NoteImportPhase,
    NoteImportReviewEffect,
    NoteImportWorkflowSnapshot,
    add_selected_file,
    apply_import_progress,
    begin_checking,
    begin_importing,
    begin_retry,
    initial_note_import_snapshot,
    project_library_note_import_snapshot,
    request_import_cancellation,
    revisit_latest_receipt,
    select_folder,
    set_approved_plan,
    set_collision_rename,
    set_destination_input,
    set_review_page,
    settle_import,
    show_review,
)
from tldw_chatbook.Notes.note_import_executor import LocalNoteImportTarget
from tldw_chatbook.Notes.note_folder_models import (
    FolderValidationError,
    normalize_folder_name,
)
from tldw_chatbook.Notes.note_import_plan_models import (
    ImportAction,
    ImportBounds,
    RootCollisionChoice,
)


class LibraryNoteImportController:
    """Own one import workflow while keeping Textual and storage late-bound."""

    def __init__(
        self,
        *,
        bounds: ImportBounds,
        database: Callable[[], Any],
        folder_repository: Callable[[], Any],
        receipt_repository: Callable[[], Any],
        discover_import_sources: Callable[..., Any],
        parse_import_sources: Callable[..., Any],
        classify_import_batch: Callable[..., Any],
        analyze_root_collision: Callable[..., Any],
        resolve_root_collision: Callable[..., Any],
        confirm_uncertain_match: Callable[..., Any],
        apply_item_override: Callable[..., Any],
        approve_note_import_plan: Callable[..., Any],
        executor_factory: Callable[[Any, Any, Any], Any],
        publish_snapshot: Callable[[Any], None],
        refresh_after_settlement: Callable[[], Any],
        review_note_reader: Callable[[str], Any] | None = None,
    ) -> None:
        """Bind the narrow planning/execution seams used by this workflow."""
        if not isinstance(bounds, ImportBounds):
            raise TypeError("bounds must be ImportBounds.")
        dependencies = (
            folder_repository,
            database,
            receipt_repository,
            discover_import_sources,
            parse_import_sources,
            classify_import_batch,
            analyze_root_collision,
            resolve_root_collision,
            confirm_uncertain_match,
            apply_item_override,
            approve_note_import_plan,
            executor_factory,
            publish_snapshot,
            refresh_after_settlement,
        )
        if not all(callable(dependency) for dependency in dependencies):
            raise TypeError("Import controller dependencies must be callable.")

        self._bounds = bounds
        self._database = database
        self._folder_repository = folder_repository
        self._receipt_repository = receipt_repository
        self._discover = discover_import_sources
        self._parse = parse_import_sources
        self._classify = classify_import_batch
        self._analyze_collision = analyze_root_collision
        self._resolve_collision = resolve_root_collision
        self._confirm_match = confirm_uncertain_match
        self._apply_override = apply_item_override
        self._approve = approve_note_import_plan
        self._executor_factory = executor_factory
        self._publish_snapshot = publish_snapshot
        self._refresh_after_settlement = refresh_after_settlement
        self._review_note_reader = review_note_reader

        self._state = initial_note_import_snapshot()
        self._before_check: NoteImportWorkflowSnapshot | None = None
        self._check_task: asyncio.Task[Any] | None = None
        self._cancel_event: threading.Event | None = None
        self._existing_top_level_names: tuple[str, ...] = ()
        self._error_message = ""

    @property
    def snapshot(self) -> NoteImportWorkflowSnapshot:
        """Return the current frozen authority-bearing workflow snapshot."""
        return self._state

    @property
    def presentation_snapshot(self) -> Any:
        """Return the frozen redacted canvas projection."""
        projected = project_library_note_import_snapshot(self._state)
        return (
            replace(projected, status_line=self._error_message)
            if self._error_message
            else projected
        )

    def publish(self) -> None:
        """Publish the current redacted projection to the screen owner."""
        self._publish_snapshot(self.presentation_snapshot)

    def begin_selection(self) -> None:
        """Start a new selection while retaining the latest session receipt."""
        self._state = initial_note_import_snapshot(
            latest_receipt=self._state.latest_receipt
        )
        self._existing_top_level_names = ()
        self._error_message = ""
        self.publish()

    def accept_selected_path(
        self,
        path: Path,
        *,
        is_folder: bool | None = None,
    ) -> None:
        """Admit one picker result as a file or the exclusive folder source."""
        if not isinstance(path, Path):
            raise TypeError("path must be a Path.")
        folder = path.is_dir() if is_folder is None else is_folder
        if type(folder) is not bool:
            raise TypeError("is_folder must be a boolean when provided.")
        self._state = (
            select_folder(self._state, path)
            if folder
            else add_selected_file(self._state, path)
        )
        self._error_message = ""
        self.publish()

    def set_destination(self, value: str) -> None:
        """Retain exact destination input and its mutation-free validation."""
        self._state = set_destination_input(self._state, value)
        self._error_message = ""
        self.publish()

    async def check(self) -> None:
        """Build a read-only review in the exact approved planner sequence."""
        before = self._state
        self._error_message = ""
        self._before_check = before
        self._state = begin_checking(before)
        self.publish()
        self._check_task = asyncio.current_task()
        try:
            plan, names, effects = await asyncio.to_thread(self._plan_selection, before)
        except asyncio.CancelledError:
            self._state = before
            self.publish()
            raise
        except Exception:
            self._state = before
            self._error_message = (
                "Could not check these sources. Review the selection and try again."
            )
            self.publish()
            raise
        finally:
            self._check_task = None
            self._before_check = None
        self._existing_top_level_names = names
        self._state = show_review(self._state, plan, review_effects=effects)
        self.publish()

    def _plan_selection(
        self, selected: NoteImportWorkflowSnapshot
    ) -> tuple[Any, tuple[str, ...], tuple[NoteImportReviewEffect, ...]]:
        discovery = self._discover(selected.selected_paths, self._bounds)
        destination = (
            selected.destination_segments if selected.requires_destination else None
        )
        batch = self._parse(
            discovery,
            self._bounds,
            destination_folder_segments=destination,
        )
        preliminary = self._classify(batch, self._bounds)
        observations = self._receipt_repository().prior_observations_for_plan_read_only(
            preliminary
        )
        plan = self._classify(
            batch,
            self._bounds,
            prior_observations=observations,
        )
        names = self._top_level_folder_names()
        plan = self._analyze_collision(plan, names)
        return plan, names, self._build_review_effects(plan)

    def _build_review_effects(self, plan: Any) -> tuple[NoteImportReviewEffect, ...]:
        matched = tuple(item for item in plan.items if item.match is not None)
        if not matched:
            return ()
        reader = self._review_note_reader
        if reader is None:
            target = LocalNoteImportTarget(
                db=self._database(),
                folder_repository=self._folder_repository(),
            )

            def reader(note_id: str) -> Any:
                return target.read_note(note_id=note_id)

        effects: list[NoteImportReviewEffect] = []
        for item in matched:
            note = reader(item.match.note_id)
            if note is None:
                continue
            effects.append(
                NoteImportReviewEffect(
                    item_id=item.item_id,
                    target_title=note.title,
                    target_version=note.version,
                    content_diff=self._bounded_note_diff(note, item),
                )
            )
        return tuple(effects)

    @staticmethod
    def _bounded_note_diff(note: Any, item: Any) -> str:
        if not item.payloads:
            return ""
        payload = item.payloads[0]
        max_input_chars = 16_000
        max_output_chars = 1_600
        marker = "\n… Diff preview truncated."

        def bounded_document(title: str, content: str) -> tuple[str, bool]:
            prefix = "Title: "
            total = len(prefix) + len(title) + 2 + len(content)
            bounded = f"{prefix}{title[: max_input_chars - len(prefix)]}"
            if len(bounded) < max_input_chars:
                separator = "\n\n"[: max_input_chars - len(bounded)]
                bounded += separator
                bounded += content[: max_input_chars - len(bounded)]
            return bounded, total > max_input_chars

        before_text, before_truncated = bounded_document(note.title, note.content)
        after_text, after_truncated = bounded_document(payload.title, payload.content)
        truncated = before_truncated or after_truncated
        before = before_text.splitlines()
        after = after_text.splitlines()
        chunks: list[str] = []
        size = 0
        budget = max_output_chars - len(marker)
        for line in difflib.unified_diff(
            before,
            after,
            fromfile="Existing note",
            tofile="Imported source",
            n=2,
            lineterm="",
        ):
            chunk = line if not chunks else f"\n{line}"
            if size + len(chunk) > budget:
                truncated = True
                break
            chunks.append(chunk)
            size += len(chunk)
        preview = "".join(chunks)
        return f"{preview}{marker}" if truncated else preview

    def _top_level_folder_names(self) -> tuple[str, ...]:
        repository = self._folder_repository()
        names: list[str] = []
        offset = 0
        while True:
            page = repository.list_children(parent_id=None, limit=500, offset=offset)
            names.extend(folder.name for folder in page.folders)
            next_offset = page.next_folder_offset
            if next_offset is None:
                return tuple(names)
            offset = next_offset

    def set_collision_name(self, name: str) -> None:
        """Retain and validate an exact proposed collision rename."""
        if not isinstance(name, str):
            raise TypeError("name must be text.")
        error = self._collision_rename_error(name)
        self._state = set_collision_rename(self._state, name, error=error)
        self.publish()

    def _collision_rename_error(self, name: str) -> str:
        if not name or not name.strip():
            return "Enter a different folder name."
        if name != name.strip():
            return "Remove leading or trailing spaces from the folder name."
        if len(name) > 255:
            return "Use a folder name with 255 characters or fewer."
        try:
            normalized = normalize_folder_name(name)
        except FolderValidationError:
            return "Enter one valid folder name without path separators."
        existing = {
            normalize_folder_name(value).key for value in self._existing_top_level_names
        }
        if normalized.key in existing:
            return "That folder name already exists. Enter a different name."
        return ""

    def set_collision_choice(self, choice: str) -> None:
        """Apply one explicit folder-root collision resolution."""
        plan = self._require_review_plan()
        collision_choice = RootCollisionChoice(choice)
        renamed = (
            self._state.collision_rename_input
            if collision_choice is RootCollisionChoice.RENAMED_ROOT
            else None
        )
        if collision_choice is RootCollisionChoice.RENAMED_ROOT and (
            not renamed or self._state.collision_rename_error
        ):
            raise ValueError("Enter a valid different folder name before renaming.")
        updated = self._resolve_collision(
            plan,
            collision_choice,
            existing_top_level_names=self._existing_top_level_names,
            renamed_root=renamed,
        )
        self._replace_review_plan(updated)

    def confirm_uncertain(self, item_id: str) -> None:
        """Confirm one uncertain match through the injected planner transform."""
        self._replace_review_plan(
            self._confirm_match(self._require_review_plan(), item_id)
        )

    def set_item_action(self, item_id: str, action: str) -> None:
        """Apply one item action, choosing a valid update effect by default."""
        plan = self._require_review_plan()
        item = next((entry for entry in plan.items if entry.item_id == item_id), None)
        if item is None:
            raise ValueError("The review item is unavailable.")
        selected = ImportAction(action)
        replace_content = item.replace_content
        add_membership = item.add_membership
        if selected is ImportAction.UPDATE_EXISTING and not (
            replace_content or add_membership
        ):
            replace_content = True
        updated = self._apply_override(
            plan,
            item_id,
            selected,
            replace_content=replace_content,
            add_membership=add_membership,
        )
        self._replace_review_plan(updated)

    def set_item_choice(self, item_id: str, choice: str, enabled: bool) -> None:
        """Apply one independent update-content or folder-membership choice."""
        if type(enabled) is not bool:
            raise TypeError("enabled must be a boolean.")
        plan = self._require_review_plan()
        item = next((entry for entry in plan.items if entry.item_id == item_id), None)
        if item is None:
            raise ValueError("The review item is unavailable.")
        if choice not in {"replace_content", "add_membership"}:
            raise ValueError("The review choice is unavailable.")
        kwargs = {
            "replace_content": item.replace_content,
            "add_membership": item.add_membership,
        }
        kwargs[choice] = enabled
        updated = self._apply_override(
            plan,
            item_id,
            item.selected_action,
            **kwargs,
        )
        self._replace_review_plan(updated)

    def _require_review_plan(self) -> Any:
        if self._state.phase is not NoteImportPhase.REVIEW or self._state.plan is None:
            raise ValueError("Import review is not active.")
        return self._state.plan

    def _replace_review_plan(self, plan: Any) -> None:
        page_number = self._state.page.page_number
        effects = self._state.review_effects
        rename_input = self._state.collision_rename_input
        rename_error = self._state.collision_rename_error
        collision = plan.root_collision
        if collision is not None and collision.choice in {
            RootCollisionChoice.USE_EXISTING,
            RootCollisionChoice.UNIQUE_SIBLING,
        }:
            rename_input = ""
            rename_error = ""
        self._state = set_review_page(
            show_review(
                begin_checking(self._state),
                plan,
                review_effects=effects,
            ),
            page_number,
        )
        self._state = replace(
            self._state,
            collision_rename_input=rename_input,
            collision_rename_error=rename_error,
        )
        self.publish()

    def set_page(self, page_number: int) -> None:
        """Move the bounded review page."""
        self._state = set_review_page(self._state, page_number)
        self.publish()

    def admit_execution(self, *, retry: bool) -> asyncio.Task[None] | None:
        """Synchronously admit one execution before a UI worker is scheduled."""
        if self._state.phase is NoteImportPhase.IMPORTING:
            return None
        self._error_message = ""
        restore_state = self._state
        if retry:
            if not self._state.can_retry:
                return None
            self._state = begin_retry(self._state)
            approved = self._state.approved_plan
            if approved is None:
                self._state = restore_state
                return None
        else:
            if (
                self._state.phase is not NoteImportPhase.REVIEW
                or self._state.plan is None
            ):
                return None
            approved = self._approve(self._state.plan)
            self._state = begin_importing(set_approved_plan(self._state, approved))
        self._cancel_event = threading.Event()
        self.publish()
        return asyncio.create_task(
            self._execute_admitted(
                approved,
                restore_state=restore_state,
                retry=retry,
                cancel_event=self._cancel_event,
            )
        )

    async def approve_and_execute(self) -> None:
        """Admit and execute the exact current review once."""
        execution = self.admit_execution(retry=False)
        if execution is not None:
            await execution

    async def _execute_admitted(
        self,
        approved: Any,
        *,
        restore_state: NoteImportWorkflowSnapshot,
        retry: bool,
        cancel_event: threading.Event,
    ) -> None:
        loop = asyncio.get_running_loop()

        def progress_callback(progress: Any) -> None:
            loop.call_soon_threadsafe(self._apply_progress, progress)

        operation: asyncio.Task[Any] | None = None
        try:
            executor = self._new_executor()
            operation = asyncio.create_task(
                asyncio.to_thread(
                    executor.retry_failed,
                    approved,
                    cancel_event=cancel_event,
                    progress_callback=progress_callback,
                )
                if retry
                else executor.execute_async(
                    approved,
                    cancel_event=cancel_event,
                    progress_callback=progress_callback,
                )
            )
            try:
                receipt = await asyncio.shield(operation)
            except asyncio.CancelledError:
                cancel_event.set()
                try:
                    receipt = await operation
                except asyncio.CancelledError:
                    self._state = restore_state
                    self.publish()
                    return
        except Exception:
            self._state = restore_state
            self._error_message = (
                "Retry could not finish safely. Review the receipt and try again."
                if retry
                else "Import could not start or finish safely. Review the plan and try again."
            )
            self.publish()
            raise
        finally:
            if self._cancel_event is cancel_event:
                self._cancel_event = None
        self._state = settle_import(self._state, receipt)
        self.publish()
        await self._refresh()

    def _apply_progress(self, progress: Any) -> None:
        if self._state.phase is not NoteImportPhase.IMPORTING:
            return
        self._state = apply_import_progress(self._state, progress)
        self.publish()

    def cancel(self) -> None:
        """Request cooperative cancellation of checking or execution."""
        if self._state.phase is NoteImportPhase.CHECKING:
            task = self._check_task
            if task is not None:
                task.cancel()
            if self._before_check is not None:
                self._state = self._before_check
                self.publish()
            return
        if self._state.phase is not NoteImportPhase.IMPORTING:
            return
        if self._cancel_event is not None:
            self._cancel_event.set()
        self._state = request_import_cancellation(self._state)
        self.publish()

    async def retry_failed(self) -> None:
        """Retry only receipt-authorized work using retained exact authority."""
        execution = self.admit_execution(retry=True)
        if execution is not None:
            await execution

    def revisit_receipt(self) -> None:
        """Reopen the latest durable receipt retained in this app session."""
        self._state = revisit_latest_receipt(self._state)
        self.publish()

    async def _refresh(self) -> None:
        result = self._refresh_after_settlement()
        if inspect.isawaitable(result):
            await result

    def _new_executor(self) -> Any:
        """Build an executor from the current app-owned local authorities."""
        return self._executor_factory(
            self._database(),
            self._folder_repository(),
            self._receipt_repository(),
        )
