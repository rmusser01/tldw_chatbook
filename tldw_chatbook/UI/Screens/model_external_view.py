"""Configured user-owned Parakeet sources in Lab Models."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from textual import on
from textual.app import ComposeResult
from textual.containers import Vertical, VerticalScroll
from textual.css.query import NoMatches
from textual.message import Message
from textual.widget import Widget
from textual.widgets import Button, Static

from tldw_chatbook.STT.parakeet_sources import ParakeetSourceKey
from tldw_chatbook.Utils.optional_deps import parakeet_onnx_deps_installed

if TYPE_CHECKING:
    from tldw_chatbook.STT.parakeet_sources import ParakeetSourceService


class ExternalModelView(Widget):
    """Render configured external roots and post lifecycle intents."""

    class _KeyMessage(Message):
        def __init__(self, key: ParakeetSourceKey) -> None:
            self.key = key
            super().__init__()

    class ChangeRequested(_KeyMessage):
        """Choose a replacement directory for one exact source."""

    class StopRequested(_KeyMessage):
        """Stop using one configured external source."""

    class CopyRequested(_KeyMessage):
        """Copy one verified source through the managed-store boundary."""

    class CancelRequested(Message):
        """Cancel the active operation when no configured row exists."""

    BUNDLED_CSS = """
    ExternalModelView {
        height: 100%;
    }

    ExternalModelView .external-model-list {
        height: 1fr;
    }

    ExternalModelView .external-model-row {
        height: auto;
        padding: 1;
        margin-bottom: 1;
        border: solid $surface-lighten-1;
    }

    ExternalModelView .external-model-title {
        text-style: bold;
    }

    ExternalModelView .external-model-status,
    ExternalModelView .external-model-path {
        height: auto;
        color: $text-muted;
    }

    ExternalModelView .external-model-actions {
        width: 100%;
        height: auto;
        margin-top: 1;
    }

    ExternalModelView .external-model-actions Button {
        width: 100%;
    }

    ExternalModelView #external-model-operation-status {
        height: auto;
        margin-bottom: 1;
    }

    ExternalModelView #external-model-operation-status.-error {
        color: $error;
    }

    ExternalModelView #external-model-cancel-operation {
        width: 100%;
        margin-bottom: 1;
    }
    """

    def __init__(
        self,
        source_service: ParakeetSourceService,
        *,
        runtime_ready: Callable[[], bool] | None = None,
        id: str | None = None,
    ) -> None:
        self.source_service = source_service
        self._runtime_ready = (
            parakeet_onnx_deps_installed if runtime_ready is None else runtime_ready
        )
        self._operation_status = ""
        self._operation_error = False
        self._operation_active = False
        super().__init__(id=id)

    def compose(self) -> ComposeResult:
        status = Static(
            self._operation_status,
            id="external-model-operation-status",
            markup=False,
        )
        status.display = bool(self._operation_status)
        status.set_class(self._operation_error, "-error")
        yield status

        records = tuple(
            (key, record)
            for key, record in self.source_service.records().items()
            if record.directory is not None
        )
        runtime_ready = self._runtime_ready() if records else False
        if self._operation_active and not records:
            yield Button(
                "Cancel operation",
                id="external-model-cancel-operation",
            )
        if not records:
            yield Static(
                "No external Parakeet sources are configured. Choose Use from disk "
                "on a curated Parakeet model.",
                markup=False,
            )
        with VerticalScroll(classes="external-model-list"):
            for key, record in records:
                yield self._row(
                    key,
                    record.directory,
                    runtime_ready=runtime_ready,
                    operation_active=self._operation_active,
                )

    @staticmethod
    def _row(
        key: ParakeetSourceKey,
        directory,
        *,
        runtime_ready: bool,
        operation_active: bool,
    ) -> Vertical:
        actions = []
        for action, label in (
            ("change", "Change…"),
            ("stop", "Cancel operation" if operation_active else "Stop using"),
            ("copy", "Copy into managed store…"),
        ):
            button = Button(
                label,
                id=f"external-model-{action}-{key.value}",
                classes=f"external-model-{action}",
                variant="primary" if action == "change" else "default",
            )
            button.source_key = key
            button.source_action = action
            actions.append(button)
        return Vertical(
            Static(
                f"{'Parakeet v2' if key.value.startswith('v2_') else 'Parakeet v3'} "
                f"· {key.precision.upper()}",
                classes="external-model-title",
                markup=False,
            ),
            Static(
                (
                    "External source · descriptor verified"
                    if runtime_ready
                    else "Runtime required"
                ),
                classes="external-model-status",
                markup=False,
            ),
            Static(str(directory), classes="external-model-path", markup=False),
            Vertical(*actions, classes="external-model-actions"),
            classes="external-model-row",
            id=f"external-model-row-{key.value}",
        )

    def reload(self) -> None:
        """Re-read the service's in-memory records and retain operation copy."""

        self.refresh(recompose=True)

    def apply_operation_status(
        self,
        text: str,
        *,
        error: bool = False,
        active: bool | None = None,
    ) -> None:
        """Show path-safe progress or recovery copy on this edit surface."""

        self._operation_status = text
        self._operation_error = error
        active_changed = active is not None and active != self._operation_active
        if active is not None:
            self._operation_active = active
        if active_changed:
            self.refresh(recompose=True)
            return
        try:
            status = self.query_one("#external-model-operation-status", Static)
        except NoMatches:
            return
        status.update(text)
        status.display = bool(text)
        status.set_class(error, "-error")

    @on(Button.Pressed, ".external-model-actions Button")
    def _action_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        key = getattr(event.button, "source_key", None)
        action = getattr(event.button, "source_action", None)
        if type(key) is not ParakeetSourceKey:
            return
        event_type = {
            "change": self.ChangeRequested,
            "stop": self.StopRequested,
            "copy": self.CopyRequested,
        }.get(action)
        if event_type is not None:
            self.post_message(event_type(key))

    @on(Button.Pressed, "#external-model-cancel-operation")
    def _cancel_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        self.post_message(self.CancelRequested())


__all__ = ["ExternalModelView"]
