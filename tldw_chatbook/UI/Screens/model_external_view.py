"""Configured user-owned Parakeet sources in Lab Models."""

from __future__ import annotations

from typing import TYPE_CHECKING

from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.widget import Widget
from textual.widgets import Button, Static

from tldw_chatbook.STT.parakeet_sources import ParakeetSourceKey

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

    DEFAULT_CSS = """
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
        height: 3;
        margin-top: 1;
    }

    ExternalModelView .external-model-actions Button {
        width: auto;
        margin-right: 1;
    }

    ExternalModelView #external-model-operation-status {
        height: auto;
        margin-bottom: 1;
    }

    ExternalModelView #external-model-operation-status.-error {
        color: $error;
    }
    """

    def __init__(
        self,
        source_service: ParakeetSourceService,
        *,
        id: str | None = None,
    ) -> None:
        self.source_service = source_service
        self._operation_status = ""
        self._operation_error = False
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
        if not records:
            yield Static(
                "No external Parakeet sources are configured. Choose Use from disk "
                "on a curated Parakeet model.",
                markup=False,
            )
        with VerticalScroll(classes="external-model-list"):
            for key, record in records:
                yield self._row(key, record.directory)

    @staticmethod
    def _row(key: ParakeetSourceKey, directory) -> Vertical:
        actions = []
        for action, label in (
            ("change", "Change…"),
            ("stop", "Stop using"),
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
                "External source · descriptor verified",
                classes="external-model-status",
                markup=False,
            ),
            Static(str(directory), classes="external-model-path", markup=False),
            Horizontal(*actions, classes="external-model-actions"),
            classes="external-model-row",
            id=f"external-model-row-{key.value}",
        )

    def reload(self) -> None:
        """Re-read the service's in-memory records and retain operation copy."""

        self.refresh(recompose=True)

    def apply_operation_status(self, text: str, *, error: bool = False) -> None:
        """Show path-safe progress or recovery copy on this edit surface."""

        self._operation_status = text
        self._operation_error = error
        self.refresh(recompose=True)

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


__all__ = ["ExternalModelView"]
