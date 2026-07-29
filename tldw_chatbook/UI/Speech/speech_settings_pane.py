"""TTS Settings in the Console grammar.

Two measured defects drive the layout.

`save-settings-btn` sat at **y=102 in a 26-row viewport** -- the primary
action of a settings screen, four screens below where you land, after every
provider block. Here the actions are a strip at the top, where they are
reachable whatever is expanded below.

And a collapsed group said nothing: eight identical closed boxes, so
answering "which providers are set up?" meant opening each in turn. Groups
now state their own state, and the ones that need attention -- configured or
half-configured -- start open. The spec's rule for this view.

Settings owns **persisted defaults**. It never reads Playground state; the
Playground's overrides are session-scoped and do not write back.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from textual.app import ComposeResult
from textual.containers import Vertical, VerticalScroll
from textual.widgets import Static

from ..Workbench.workbench_state import WorkbenchAction
from .speech_action_strip import SpeechActionStrip
from .speech_settings_group import SpeechSettingsGroup
from .speech_settings_model import (
    SETTINGS_PROVIDER_ORDER,
    configured_state,
)

#: The view's commands. Ids are the legacy ones and `SpeechActionStrip`
#: mounts them verbatim -- `CommandStrip` rewrites every id it is given, so
#: `save-settings-btn` would become `workbench-action-save-settings-btn`
#: while the handler still matches the bare id: a button that renders and
#: can never fire.
SETTINGS_ACTION_SPECS: tuple[WorkbenchAction, ...] = (
    WorkbenchAction(
        id="save-settings-btn",
        label="Save",
        tooltip="Persist these settings",
        primary=True,
    ),
    WorkbenchAction(
        id="import-blends-btn",
        label="Import blends",
        tooltip="Load voice blends from a file",
    ),
    WorkbenchAction(
        id="export-blends-btn",
        label="Export blends",
        tooltip="Save voice blends to a file",
    ),
    WorkbenchAction(
        id="add-voice-blend-btn",
        label="Add blend",
        tooltip="Create a voice blend",
    ),
)

#: States whose groups start expanded. `default` stays closed: a provider
#: nobody has touched is the one the user is least likely to want, and
#: opening all eight rebuilds the wall of forms this replaces.
OPEN_STATES = ("configured", "incomplete")


class SpeechSettingsPane(Vertical):
    """The TTS Settings body: actions, then one group per provider."""

    def __init__(
        self,
        *,
        values: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Create the pane.

        Args:
            values: Current value per setting id, used to fill the controls
                and to decide which groups open.
            kwargs: Forwarded to ``Vertical``.
        """
        classes = kwargs.pop("classes", "")
        super().__init__(classes=f"speech-settings-pane {classes}".strip(), **kwargs)
        self.values: dict[str, Any] = dict(values or {})

    def compose(self) -> ComposeResult:
        """Compose the view.

        Returns:
            A ``ComposeResult`` yielding the title, the action strip, and a
            scrollable list of provider groups. The strip is outside the
            scroll region, so Save stays on screen however far down the
            groups run.
        """
        yield Static("⚙️ TTS Settings", classes="speech-pane-title")
        yield SpeechActionStrip(
            SETTINGS_ACTION_SPECS, id="speech-settings-actions"
        )

        with VerticalScroll(id="speech-settings-groups"):
            for provider in SETTINGS_PROVIDER_ORDER:
                state = configured_state(provider, self.values)
                yield SpeechSettingsGroup(
                    provider=provider,
                    values=self.values,
                    collapsed=state not in OPEN_STATES,
                )
