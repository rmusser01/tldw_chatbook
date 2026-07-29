"""What Save posts must not change when the view is rebuilt.

`_save_settings` does not write config: it collects 47 values off the
controls and posts one `STTSSettingsSaveEvent`. Persistence lives in the
event handler, which this phase does not touch. So the exact thing to hold
fixed is the dict that crosses that boundary -- and a snapshot of it, taken
from the legacy widget before any code moved, is a stronger check than
diffing a written file: it compares every key and value rather than the
effect of some of them.

If a key disappears here, a setting silently stopped being persisted.
"""

from __future__ import annotations

import json
import pathlib

import pytest
from textual.app import App, ComposeResult

from tldw_chatbook.Event_Handlers.STTS_Events.stts_events import (
    STTSSettingsSaveEvent,
)

BASELINE = (
    pathlib.Path(__file__).parent / "fixtures" / "tts_settings_save_baseline.json"
)


class _CapturingHost(App[None]):
    """Records the save event instead of letting it reach persistence."""

    def __init__(self, widget_factory):
        super().__init__()
        self._factory = widget_factory
        self.saved: list[STTSSettingsSaveEvent] = []

    def compose(self) -> ComposeResult:
        yield self._factory()

    def post_message(self, message):
        if isinstance(message, STTSSettingsSaveEvent):
            self.saved.append(message)
            return True
        return super().post_message(message)

    def notify(self, *args, **kwargs):
        pass


async def _settings_posted_by(widget_factory) -> dict[str, str]:
    app = _CapturingHost(widget_factory)
    async with app.run_test(size=(200, 80)) as pilot:
        await pilot.pause()
        await pilot.pause()
        pane = app.query_one(widget_factory.target)
        pane._save_settings()
        await pilot.pause()
    assert app.saved, "Save posted no event at all"
    return {k: repr(v) for k, v in dict(app.saved[-1].settings).items()}


class _LegacyFactory:
    from tldw_chatbook.UI.STTS_Window import TTSSettingsWidget as target

    def __call__(self):
        return self.target()


@pytest.mark.asyncio
async def test_save_still_posts_every_baseline_key():
    """The guard for the rebuild: 47 keys, captured off the legacy widget.

    A missing key is a setting that silently stopped being saved -- the user
    changes it, presses Save, sees no error, and it does not stick.
    """
    baseline = json.loads(BASELINE.read_text())
    posted = await _settings_posted_by(_LegacyFactory())

    missing = sorted(set(baseline) - set(posted))
    assert not missing, f"no longer saved: {missing}"


@pytest.mark.asyncio
async def test_save_posts_the_same_values_as_the_baseline():
    """Not just the same keys -- the same values.

    A key that survives while its value changes is worse than one that
    vanishes: it writes something wrong and says nothing.
    """
    baseline = json.loads(BASELINE.read_text())
    posted = await _settings_posted_by(_LegacyFactory())

    changed = {
        key: (baseline[key], posted[key])
        for key in sorted(set(baseline) & set(posted))
        if baseline[key] != posted[key]
    }
    assert not changed, f"values changed: {changed}"
