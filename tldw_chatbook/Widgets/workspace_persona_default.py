"""Shared native controls for future conversations' workspace Persona default."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, ClassVar

from rich.text import Text
from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Checkbox, Select, Static

from ..Workspaces.models import DEFAULT_WORKSPACE_ID, WorkspaceAssistantDefaults
from ..Workspaces.registry_service import (
    LocalWorkspaceRegistryService,
    WorkspaceRegistryServiceError,
)
from .modal_dismissal import SafeModalDismissMixin


@dataclass(frozen=True)
class WorkspacePersonaSelection:
    """Uncommitted form values retained through a parent recompose."""

    persona: str = "auto"
    memory_mode: str = "read_only"
    confirm_read_write: bool = False


class WorkspacePersonaPicker(Vertical):
    """Select None or a saved Persona without changing the registry."""

    DEFAULT_CSS = """
    WorkspacePersonaPicker { height: auto; }
    WorkspacePersonaPicker Static { height: auto; }
    WorkspacePersonaPicker Select { height: 3; }
    """

    def __init__(
        self,
        persona_service: Any,
        *,
        selection: WorkspacePersonaSelection,
        allow_auto: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._personas = persona_service
        self._selection = selection
        self._allow_auto = allow_auto

    def compose(self) -> ComposeResult:
        options = [("None", "none")]
        if self._allow_auto:
            options.insert(0, ("Create a workspace Agent", "auto"))
        unavailable = False
        try:
            records = self._personas.list_persona_profiles() if self._personas else []
        except Exception:  # noqa: BLE001 - None remains a usable choice
            records, unavailable = [], True
        for record in records or []:
            if not isinstance(record, Mapping) or record.get("deleted"):
                continue
            persona_id = str(record.get("id") or "")
            if persona_id and persona_id not in {"auto", "none"}:
                options.append(
                    (Text(str(record.get("name") or persona_id)), persona_id)
                )
        if self._selection.persona not in {value for _, value in options}:
            options.append(("Saved Persona unavailable", self._selection.persona))
        yield Static("Default Persona · future new conversations", markup=False)
        yield Select(
            options,
            value=self._selection.persona,
            allow_blank=False,
            id="workspace-default-persona",
            compact=True,
        )
        yield Static(
            "Existing, copied and moved conversations keep their Persona.", markup=False
        )
        if unavailable:
            yield Static(
                "Persona list unavailable. None is still available.", markup=False
            )
        yield Static("Persona memory", markup=False)
        yield Select(
            [("Read only", "read_only"), ("Read and write", "read_write")],
            value=self._selection.memory_mode,
            allow_blank=False,
            id="workspace-default-memory",
            compact=True,
        )
        yield Checkbox(
            "Confirm this Persona may write memory across sessions",
            self._selection.confirm_read_write,
            id="workspace-default-memory-confirm",
            compact=True,
        )

    @on(Select.Changed)
    def _choice_changed(self, event: Select.Changed) -> None:
        event.stop()
        current = self.selection()
        if (current.persona, current.memory_mode) != (
            self._selection.persona,
            self._selection.memory_mode,
        ):
            self.query_one("#workspace-default-memory-confirm", Checkbox).value = False
            self._selection = self.selection()
        self._sync_memory_controls()

    def on_mount(self) -> None:
        self._sync_memory_controls()

    def _sync_memory_controls(self) -> None:
        selected = self.query_one("#workspace-default-persona", Select).value
        self.query_one("#workspace-default-memory", Select).disabled = selected in {
            "auto",
            "none",
        }
        self.query_one("#workspace-default-memory-confirm", Checkbox).disabled = (
            selected in {"auto", "none"}
            or self.query_one("#workspace-default-memory", Select).value != "read_write"
        )

    def selection(self) -> WorkspacePersonaSelection:
        """Capture uncommitted values before any parent recomposition."""
        return WorkspacePersonaSelection(
            str(self.query_one("#workspace-default-persona", Select).value),
            str(self.query_one("#workspace-default-memory", Select).value),
            self.query_one("#workspace-default-memory-confirm", Checkbox).value,
        )

    def creation_kwargs(self, *, profile_id: str | None = None) -> dict[str, Any]:
        """Validate current identity and confirmation without changing any store."""
        choice = self.selection()
        if choice.persona == "auto" and self._allow_auto:
            return {}
        if choice.persona == "none":
            return {"assistant_defaults": None}
        try:
            record = self._personas.get_persona_profile(choice.persona)
        except Exception as exc:
            raise ValueError(
                "Selected Persona is unavailable. Choose another Persona or None."
            ) from exc
        if (
            not isinstance(record, Mapping)
            or record.get("deleted")
            or str(record.get("id")) != choice.persona
        ):
            raise ValueError(
                "Selected Persona is unavailable. Choose another Persona or None."
            )
        if choice.memory_mode == "read_write" and not choice.confirm_read_write:
            raise ValueError(
                "Confirm read and write memory before applying this default."
            )
        return {
            "assistant_defaults": WorkspaceAssistantDefaults(
                assistant_id=choice.persona,
                persona_memory_mode=choice.memory_mode,
                tool_policy_profile_id=profile_id,
            ),
            "confirm_read_write": choice.confirm_read_write,
        }


class WorkspacePersonaDefaultModal(SafeModalDismissMixin, ModalScreen[bool]):
    """Edit one explicit workspace's default; Cancel never mutates it."""

    DEFAULT_CSS = """
    WorkspacePersonaDefaultModal { align: center middle; }
    #workspace-default-dialog { width: 68; max-width: 95%; height: auto; max-height: 95%;
        background: $surface; border: tall $primary; padding: 1 2; }
    #workspace-default-form { height: auto; max-height: 17; }
    #workspace-default-actions { height: 3; align-horizontal: right; }
    #workspace-default-error { height: auto; color: $error; }
    """
    SAFE_MODAL_CONTENT = "#workspace-default-dialog"
    BINDINGS: ClassVar = [("escape", "request_safe_cancel", "Cancel")]

    def __init__(
        self,
        registry: LocalWorkspaceRegistryService,
        persona_service: Any,
        workspace_id: str,
    ) -> None:
        super().__init__()
        self._registry, self._personas, self._workspace_id = (
            registry,
            persona_service,
            workspace_id,
        )
        record = registry.get_workspace(workspace_id)
        defaults = record.assistant_defaults if record else None
        self._selection = WorkspacePersonaSelection(
            defaults.assistant_id if defaults else "none",
            defaults.persona_memory_mode if defaults else "read_only",
        )
        self._original_defaults = defaults

    def compose(self) -> ComposeResult:
        with Vertical(id="workspace-default-dialog"):
            yield Static("Workspace default Persona", classes="console-modal-header")
            with VerticalScroll(id="workspace-default-form"):
                yield WorkspacePersonaPicker(self._personas, selection=self._selection)
                yield Static("", id="workspace-default-error", markup=False)
            with Horizontal(id="workspace-default-actions"):
                yield Button("Cancel", id="workspace-default-cancel", compact=True)
                yield Button("Apply", id="workspace-default-apply", compact=True)

    @on(Button.Pressed, "#workspace-default-cancel")
    async def _cancel(self, event: Button.Pressed) -> None:
        event.stop()
        await self.request_safe_cancel(source="button")

    @on(Button.Pressed, "#workspace-default-apply")
    def _apply(self, event: Button.Pressed) -> None:
        event.stop()
        try:
            record = self._registry.get_workspace(self._workspace_id)
            if (
                record is None
                or record.archived
                or self._workspace_id == DEFAULT_WORKSPACE_ID
            ):
                raise ValueError("This workspace is unavailable for default changes.")
            if record.assistant_defaults != self._original_defaults:
                raise ValueError(
                    "This workspace default changed. Reopen to review its current value."
                )
            values = self.query_one(WorkspacePersonaPicker).creation_kwargs(
                profile_id=getattr(
                    self._original_defaults, "tool_policy_profile_id", None
                )
            )
            defaults = values["assistant_defaults"]
            if defaults is None:
                self._registry.clear_assistant_defaults(
                    self._workspace_id, expected_record=record
                )
            else:
                self._registry.set_assistant_defaults(
                    self._workspace_id,
                    defaults,
                    confirm_read_write=values.get("confirm_read_write", False),
                    expected_record=record,
                )
        except (ValueError, WorkspaceRegistryServiceError) as exc:
            self.query_one("#workspace-default-error", Static).update(str(exc))
            return
        self.dismiss_safe_once(True)
