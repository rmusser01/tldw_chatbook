# Speech Playground (Phase 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild the TTS Playground in the Console grammar so its 57 controls are reachable without scrolling past the fold, with comparison axes always visible and provider tuning knobs collapsed.

**Architecture:** A pure model module classifies every control and maps providers to their parameters, so the classification is testable without mounting a widget. Four presentation widgets consume it — axis chips, provider parameter group, result history, action strip — assembled by `SpeechPlaygroundPane` inside the existing Lab frame body. The legacy `TTSPlaygroundWidget` branch is removed from `watch_current_view` only in the final task, so every earlier task is revertible.

**Tech Stack:** Python ≥3.11, Textual 8.2.7, pytest + pytest-asyncio, app-tier TCSS in `css/features/_lab.tcss` (rebuilt via `css/build_css.py`).

## Global Constraints

- Spec: `Docs/superpowers/specs/2026-07-27-speech-console-redesign-design.md`. Where this plan and the spec disagree, the spec governs — raise it rather than choosing.
- **Every one of the 57 controls must remain reachable.** Enumerate by id against the legacy widget; do not judge by looking.
- The Playground owns **session-scoped overrides only**. It never writes persisted defaults except through an explicit save-as-default action.
- Responsive: side by side at ≥64 cells of pane width, stacked below. **Nothing dropped, nothing truncated at 80×24.**
- Assert rendered text with `widget.render_line(0).text`. `content_region.width` is not a truncation oracle — it reported 16 for a 15-character label that did not render.
- `1fr` children compress instead of overflowing. When stacking, they become `auto`/fixed or content is clipped rather than scrollable.
- Textual's `Button` has a default `min-width`; action strips need `min-width: 0` or the last actions fall off the edge.
- CSS goes in `css/features/_lab.tcss` and the bundle is rebuilt (`python tldw_chatbook/css/build_css.py`). Never hand-edit `tldw_cli_modular.tcss`.
- Mutation-check every new guard: break what it protects and confirm it fails.
- Run tests with the repo venv: `.venv/bin/python -m pytest`.

---

### Task 1: The control model

**Files:**
- Create: `tldw_chatbook/UI/Speech/speech_playground_model.py`
- Test: `Tests/UI/test_speech_playground_model.py`

**Interfaces:**
- Produces: `AXIS_CONTROLS: tuple[str, ...]`, `PROVIDER_PARAMS: dict[str, tuple[str, ...]]`, `AUDIO_PARAMS: tuple[str, ...]`, `params_for_provider(provider: str) -> tuple[str, ...]`, `ALL_PLAYGROUND_CONTROLS: frozenset[str]`

- [ ] **Step 1: Write the failing test**

```python
import pytest

from tldw_chatbook.UI.Speech.speech_playground_model import (
    ALL_PLAYGROUND_CONTROLS,
    AXIS_CONTROLS,
    PROVIDER_PARAMS,
    params_for_provider,
)


@pytest.mark.unit
def test_only_the_selected_providers_parameters_are_offered():
    """ElevenLabs' knobs must not appear while Chatterbox is selected.

    This is the whole point of the split: 26 provider params exist, but a
    user comparing voices should see only the ones that apply.
    """
    chatterbox = set(params_for_provider("chatterbox"))
    elevenlabs = set(params_for_provider("elevenlabs"))

    assert "tts-exaggeration-input" in chatterbox
    assert "tts-stability-input" not in chatterbox
    assert "tts-stability-input" in elevenlabs
    assert "tts-exaggeration-input" not in elevenlabs


@pytest.mark.unit
def test_audio_post_processing_applies_to_every_provider():
    """Normalisation is not provider-specific and must not vanish."""
    for provider in PROVIDER_PARAMS:
        assert "tts-normalize-audio-switch" in params_for_provider(provider)


@pytest.mark.unit
def test_an_unknown_provider_offers_only_the_shared_parameters():
    """A provider the model does not know must not crash the view."""
    assert set(params_for_provider("nonexistent")) == {
        "tts-preprocess-text-switch",
        "tts-normalize-audio-switch",
        "tts-target-db-input",
    }


@pytest.mark.unit
def test_axes_and_parameters_do_not_overlap():
    """A control is an axis or a knob, never both."""
    overlap = set(AXIS_CONTROLS) & {
        c for params in PROVIDER_PARAMS.values() for c in params
    }
    assert overlap == set()


@pytest.mark.unit
def test_every_known_control_is_classified():
    """No control may be silently unclassified -- that is how one goes missing."""
    classified = set(AXIS_CONTROLS) | {
        c for params in PROVIDER_PARAMS.values() for c in params
    }
    assert classified <= ALL_PLAYGROUND_CONTROLS
    unclassified = ALL_PLAYGROUND_CONTROLS - classified
    # Actions, status and player controls are deliberately outside the
    # axis/knob split; they are named here so the set is explicit.
    assert unclassified, "expected actions/status/player to be unclassified"
```

- [ ] **Step 2: Run it and watch it fail**

Run: `.venv/bin/python -m pytest Tests/UI/test_speech_playground_model.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'tldw_chatbook.UI.Speech.speech_playground_model'`

- [ ] **Step 3: Write the model**

```python
"""Which Playground controls are comparison axes and which are tuning knobs.

Pure data plus two lookups, so the classification is testable without
mounting a 5,900-line widget. The split is the spec's central rule: the
Playground exists to compare options, so the variables you compare stay
visible and the ones you set once per provider collapse.
"""

from __future__ import annotations

#: Changed constantly while comparing; always visible.
AXIS_CONTROLS: tuple[str, ...] = (
    "tts-provider-select",
    "tts-model-select",
    "tts-voice-select",
    "tts-language-select",
    "tts-format-select",
    "tts-speed-input",
)

#: Applied whatever the provider, so appended to every provider's group
#: rather than duplicated into each.
AUDIO_PARAMS: tuple[str, ...] = (
    "tts-preprocess-text-switch",
    "tts-normalize-audio-switch",
    "tts-target-db-input",
)

#: Provider -> its own tuning knobs, excluding AUDIO_PARAMS.
PROVIDER_PARAMS: dict[str, tuple[str, ...]] = {
    "elevenlabs": (
        "tts-stability-input",
        "tts-similarity-input",
        "tts-style-input",
        "tts-speaker-boost-switch",
    ),
    "chatterbox": (
        "tts-exaggeration-input",
        "tts-cfg-weight-input",
        "tts-temperature-input",
        "tts-num-candidates-input",
        "tts-validate-whisper-switch",
        "tts-random-seed-input",
    ),
    "higgs": (
        "tts-higgs-temperature-input",
        "tts-higgs-top-p-input",
        "tts-higgs-repetition-penalty-input",
        "tts-higgs-voice-cloning-switch",
        "tts-higgs-multi-speaker-switch",
        "tts-higgs-delimiter-input",
    ),
    "kokoro": ("tts-kokoro-use-onnx",),
    "audio_cpp": (),
    "openai": (),
    "alltalk": (),
}

#: Every id the legacy TTSPlaygroundWidget composed, so the rebuild can be
#: checked for completeness rather than trusted. Actions, status and player
#: controls are here too -- they are not part of the axis/knob split but
#: they must not go missing.
ALL_PLAYGROUND_CONTROLS: frozenset[str] = frozenset(
    AXIS_CONTROLS
    + AUDIO_PARAMS
    + tuple(c for params in PROVIDER_PARAMS.values() for c in params)
    + (
        "tts-text-input",
        # actions
        "tts-generate-btn",
        "tts-random-text-btn",
        "tts-clear-text-btn",
        "tts-refresh-catalog-btn",
        "audio-play-btn",
        "pause-audio-btn",
        "stop-audio-btn",
        "audio-export-btn",
        "reference-audio-btn",
        "clear-reference-audio-btn",
        "higgs-voice-upload-btn",
        "higgs-clear-voice-btn",
        # status
        "tts-provider-status",
        "tts-generation-log",
        "tts-audio-cpp-restrictions",
        "reference-audio-status",
        "higgs-voice-status",
        "audio-player-status",
        "generation-status-container",
        "generation-status-text",
        # player
        "audio-player-container",
        "audio-progress-bar",
        "audio-time-display",
        "generation-progress",
        # provider param containers
        "kokoro-settings",
        "kokoro-language-row",
        "elevenlabs-settings",
        "chatterbox-settings",
        "higgs-settings",
        "higgs-voice-upload-row",
    )
)


def params_for_provider(provider: str) -> tuple[str, ...]:
    """Return the tuning knobs to render for one provider.

    Args:
        provider: The selected provider key, e.g. ``"chatterbox"``.

    Returns:
        That provider's own parameters followed by the shared audio
        post-processing ones. An unknown provider yields only the shared
        set -- a provider the model has not been taught about must degrade
        to "no special knobs", never raise into compose().
    """
    return PROVIDER_PARAMS.get(provider, ()) + AUDIO_PARAMS
```

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/python -m pytest Tests/UI/test_speech_playground_model.py -q`
Expected: 5 passed

- [ ] **Step 5: Verify the inventory is complete against the real widget**

Run this and confirm it reports nothing missing:

```bash
.venv/bin/python - <<'EOF'
import re, pathlib
from tldw_chatbook.UI.Speech.speech_playground_model import ALL_PLAYGROUND_CONTROLS
src = pathlib.Path("tldw_chatbook/UI/STTS_Window.py").read_text().split("\n")
start = next(i for i,l in enumerate(src,1) if l.startswith("class TTSPlaygroundWidget"))
end = next(i for i,l in enumerate(src,1) if i>start and l.startswith("class "))
live = {m.group(1) for l in src[start-1:end-1]
        for m in re.finditer(r'id="([a-z0-9_-]+)"', l)}
print("in widget, not in model:", sorted(live - ALL_PLAYGROUND_CONTROLS))
print("in model, not in widget:", sorted(ALL_PLAYGROUND_CONTROLS - live))
EOF
```

Expected: both lists empty. If not, add the missing ids to `ALL_PLAYGROUND_CONTROLS` — a control absent from the model is a control the rebuild will silently drop.

- [ ] **Step 6: Commit**

```bash
git add tldw_chatbook/UI/Speech/speech_playground_model.py Tests/UI/test_speech_playground_model.py
git commit -m "feat(speech): classify Playground controls into axes and provider knobs"
```

---

### Task 2: Axis chip row

**Files:**
- Create: `tldw_chatbook/UI/Speech/speech_axis_row.py`
- Modify: `tldw_chatbook/css/features/_lab.tcss`
- Test: `Tests/UI/test_speech_axis_row.py`

**Interfaces:**
- Consumes: `AXIS_CONTROLS` from Task 1.
- Produces: `SpeechAxisRow(values: dict[str, str], defaults: dict[str, str])`, with `axis_chip_id(axis: str) -> str` and `is_override(axis: str) -> bool`.

An override must be *visible*, per the spec: the chip shows the effective value and states that it differs from the saved default.

- [ ] **Step 1: Write the failing test**

```python
import pytest
from textual.app import App, ComposeResult
from textual.widgets import Static

from tldw_chatbook.UI.Speech.speech_axis_row import SpeechAxisRow, axis_chip_id


class _Harness(App[None]):
    def __init__(self, values, defaults):
        super().__init__()
        self._values, self._defaults = values, defaults

    def compose(self) -> ComposeResult:
        yield SpeechAxisRow(values=self._values, defaults=self._defaults)


@pytest.mark.asyncio
async def test_an_overridden_axis_is_marked_and_a_matching_one_is_not():
    """A session override must be visible, not implied.

    The spec makes the Playground own session-scoped overrides that never
    write back. If an override looks identical to a saved default, the user
    cannot tell what they have changed.
    """
    app = _Harness(
        values={"tts-voice-select": "Nova", "tts-format-select": "mp3"},
        defaults={"tts-voice-select": "Server default", "tts-format-select": "mp3"},
    )
    async with app.run_test(size=(120, 10)) as pilot:
        await pilot.pause()
        row = app.query_one(SpeechAxisRow)
        voice = app.query_one(f"#{axis_chip_id('tts-voice-select')}", Static)
        fmt = app.query_one(f"#{axis_chip_id('tts-format-select')}", Static)

        assert row.is_override("tts-voice-select") is True
        assert row.is_override("tts-format-select") is False
        assert voice.has_class("speech-chip-override")
        assert not fmt.has_class("speech-chip-override")
        assert "Nova" in voice.render_line(0).text
```

- [ ] **Step 2: Run it and watch it fail**

Run: `.venv/bin/python -m pytest Tests/UI/test_speech_axis_row.py -q`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the widget**

```python
"""The always-visible comparison axes, as one chip row."""

from __future__ import annotations

from typing import Any

from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.widgets import Static

from .speech_playground_model import AXIS_CONTROLS

#: Chip label per axis, in the order they are compared.
AXIS_LABELS: dict[str, str] = {
    "tts-provider-select": "Provider",
    "tts-model-select": "Model",
    "tts-voice-select": "Voice",
    "tts-language-select": "Language",
    "tts-format-select": "Format",
    "tts-speed-input": "Speed",
}


def axis_chip_id(axis: str) -> str:
    """Return the stable chip id for one axis.

    Args:
        axis: The axis control id, e.g. ``"tts-voice-select"``.

    Returns:
        ``"speech-axis-<axis>"``.
    """
    return f"speech-axis-{axis}"


class SpeechAxisRow(Horizontal):
    """One row of `Label: value` chips for the comparison axes."""

    def __init__(
        self,
        *,
        values: dict[str, str],
        defaults: dict[str, str],
        **kwargs: Any,
    ) -> None:
        """Create the row.

        Args:
            values: Effective value per axis for this session.
            defaults: Persisted default per axis, for override detection.
            kwargs: Forwarded to ``Horizontal``.
        """
        classes = kwargs.pop("classes", "")
        super().__init__(classes=f"speech-chip-row {classes}".strip(), **kwargs)
        self.values = values
        self.defaults = defaults

    def is_override(self, axis: str) -> bool:
        """Report whether this axis differs from its persisted default.

        Args:
            axis: The axis control id.

        Returns:
            True when a default exists and the effective value differs.
        """
        if axis not in self.defaults:
            return False
        return self.values.get(axis) != self.defaults[axis]

    def compose(self) -> ComposeResult:
        """Yield one chip per axis, marking overrides."""
        for axis in AXIS_CONTROLS:
            label = AXIS_LABELS[axis]
            value = self.values.get(axis, "--")
            override = self.is_override(axis)
            chip = Static(
                f"{label}: {value}" + (" *" if override else ""),
                id=axis_chip_id(axis),
                classes="speech-chip" + (" speech-chip-override" if override else ""),
                markup=False,
            )
            chip.tooltip = (
                f"Session override. Saved default: {self.defaults.get(axis)}"
                if override
                else f"Matches the saved default ({self.defaults.get(axis, 'unset')})"
            )
            yield chip
```

- [ ] **Step 4: Add the override style**

Append to `tldw_chatbook/css/features/_lab.tcss`:

```css
/* An override is stated, not implied: the spec has the Playground owning
   session-scoped values that never write back, so the user must be able to
   see at a glance what they have changed. The asterisk carries it for
   anyone who cannot distinguish the colour. */
.speech-chip-override {
    color: $ds-status-warning;
    text-style: bold;
}
```

Then rebuild: `.venv/bin/python tldw_chatbook/css/build_css.py`

- [ ] **Step 5: Run the tests**

Run: `.venv/bin/python -m pytest Tests/UI/test_speech_axis_row.py -q`
Expected: 1 passed

- [ ] **Step 6: Mutation-check the override marking**

Temporarily make `is_override` return `False` always, re-run, confirm the test fails on `voice.has_class("speech-chip-override")`. Restore.

- [ ] **Step 7: Commit**

```bash
git add tldw_chatbook/UI/Speech/speech_axis_row.py Tests/UI/test_speech_axis_row.py tldw_chatbook/css/
git commit -m "feat(speech): axis chip row with visible session overrides"
```

---

### Task 3: Provider parameter group

**Files:**
- Create: `tldw_chatbook/UI/Speech/speech_param_group.py`
- Test: `Tests/UI/test_speech_param_group.py`

**Interfaces:**
- Consumes: `params_for_provider` from Task 1.
- Produces: `SpeechParamGroup(provider: str)`, collapsed by default, rendering only that provider's knobs.

- [ ] **Step 1: Write the failing test**

```python
import pytest
from textual.app import App, ComposeResult
from textual.widgets import Collapsible

from tldw_chatbook.UI.Speech.speech_param_group import SpeechParamGroup


class _Harness(App[None]):
    def __init__(self, provider):
        super().__init__()
        self._provider = provider

    def compose(self) -> ComposeResult:
        yield SpeechParamGroup(provider=self._provider)


@pytest.mark.asyncio
async def test_only_the_selected_providers_knobs_are_mounted():
    app = _Harness("chatterbox")
    async with app.run_test(size=(120, 20)) as pilot:
        await pilot.pause()
        assert app.query("#tts-exaggeration-input")
        assert not app.query("#tts-stability-input")


@pytest.mark.asyncio
async def test_the_group_starts_collapsed():
    """Knobs are set once per provider; they must not occupy the screen by
    default or the axis row loses its prominence."""
    app = _Harness("chatterbox")
    async with app.run_test(size=(120, 20)) as pilot:
        await pilot.pause()
        assert app.query_one(Collapsible).collapsed is True


@pytest.mark.asyncio
async def test_a_provider_with_no_knobs_still_renders_the_shared_ones():
    app = _Harness("openai")
    async with app.run_test(size=(120, 20)) as pilot:
        await pilot.pause()
        assert app.query("#tts-normalize-audio-switch")
```

- [ ] **Step 2: Run it and watch it fail**

Run: `.venv/bin/python -m pytest Tests/UI/test_speech_param_group.py -q`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the widget**

```python
"""Provider tuning knobs, collapsed and scoped to the selected provider."""

from __future__ import annotations

from typing import Any

from textual.app import ComposeResult
from textual.widgets import Collapsible, Input, Static, Switch

from .speech_playground_model import params_for_provider

#: Human label per parameter id.
PARAM_LABELS: dict[str, str] = {
    "tts-stability-input": "Stability",
    "tts-similarity-input": "Similarity",
    "tts-style-input": "Style",
    "tts-speaker-boost-switch": "Speaker boost",
    "tts-exaggeration-input": "Exaggeration",
    "tts-cfg-weight-input": "CFG weight",
    "tts-temperature-input": "Temperature",
    "tts-num-candidates-input": "Candidates",
    "tts-validate-whisper-switch": "Validate with Whisper",
    "tts-random-seed-input": "Seed",
    "tts-higgs-temperature-input": "Temperature",
    "tts-higgs-top-p-input": "Top-p",
    "tts-higgs-repetition-penalty-input": "Repetition penalty",
    "tts-higgs-voice-cloning-switch": "Voice cloning",
    "tts-higgs-multi-speaker-switch": "Multi-speaker",
    "tts-higgs-delimiter-input": "Delimiter",
    "tts-kokoro-use-onnx": "Use ONNX",
    "tts-preprocess-text-switch": "Preprocess text",
    "tts-normalize-audio-switch": "Normalize audio",
    "tts-target-db-input": "Target dB",
}


class SpeechParamGroup(Collapsible):
    """The selected provider's tuning knobs, collapsed by default."""

    def __init__(self, *, provider: str, **kwargs: Any) -> None:
        """Create the group.

        Args:
            provider: The selected provider key.
            kwargs: Forwarded to ``Collapsible``.
        """
        self.provider = provider
        kwargs.setdefault("title", f"{provider} parameters")
        kwargs.setdefault("collapsed", True)
        super().__init__(**kwargs)

    def compose(self) -> ComposeResult:
        """Yield one row per parameter this provider actually has."""
        for param in params_for_provider(self.provider):
            label = PARAM_LABELS.get(param, param)
            yield Static(label, classes="speech-param-label")
            if param.endswith("-switch") or param.endswith("-onnx"):
                yield Switch(id=param, classes="speech-param-control")
            else:
                yield Input(id=param, classes="speech-param-control")
```

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/python -m pytest Tests/UI/test_speech_param_group.py -q`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/UI/Speech/speech_param_group.py Tests/UI/test_speech_param_group.py
git commit -m "feat(speech): provider-scoped tuning knobs, collapsed by default"
```

---

### Task 4: Result history

**Files:**
- Create: `tldw_chatbook/UI/Speech/speech_result_history.py`
- Test: `Tests/UI/test_speech_result_history.py`

**Interfaces:**
- Produces: `SpeechTake` (frozen dataclass: `take_id`, `voice`, `fmt`, `duration_s`, `created_label`), `SpeechResultHistory(takes)` with `add_take(take)`.

- [ ] **Step 1: Write the failing test**

```python
import pytest
from textual.app import App, ComposeResult

from tldw_chatbook.UI.Speech.speech_result_history import (
    SpeechResultHistory,
    SpeechTake,
)


class _Harness(App[None]):
    def __init__(self, takes=()):
        super().__init__()
        self._takes = takes

    def compose(self) -> ComposeResult:
        yield SpeechResultHistory(takes=self._takes)


@pytest.mark.asyncio
async def test_empty_history_says_why_it_is_empty():
    app = _Harness()
    async with app.run_test(size=(120, 20)) as pilot:
        await pilot.pause()
        history = app.query_one(SpeechResultHistory)
        assert "Generate" in history.render_line(1).text or app.query(
            "#speech-history-empty"
        )


@pytest.mark.asyncio
async def test_newest_take_is_first_and_each_take_can_be_played():
    """Comparison is the point: every take keeps its own controls."""
    takes = (
        SpeechTake("t1", "Server default", "mp3", 12.0, "14:00"),
        SpeechTake("t2", "Nova", "wav", 4.0, "14:02"),
    )
    app = _Harness(takes)
    async with app.run_test(size=(120, 20)) as pilot:
        await pilot.pause()
        rows = list(app.query(".speech-take-row"))
        assert len(rows) == 2
        assert "Nova" in rows[0].render_line(0).text
        assert app.query("#speech-take-play-t1")
        assert app.query("#speech-take-play-t2")
```

- [ ] **Step 2: Run it and watch it fail**

Run: `.venv/bin/python -m pytest Tests/UI/test_speech_result_history.py -q`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the widget**

```python
"""Generated takes, newest first, so options can be compared.

A pane showing only the latest result asks the user to remember what the
previous one sounded like. The spec's purpose statement -- identify which
option works best -- requires comparing takes, so each keeps its own row and
its own Play/Export.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Static


@dataclass(frozen=True)
class SpeechTake:
    """One generated audio take.

    Attributes:
        take_id: Stable id, used to build per-row control ids.
        voice: Voice used.
        fmt: Audio format, e.g. ``"mp3"``.
        duration_s: Length in seconds.
        created_label: Short display time, e.g. ``"14:02"``.
    """

    take_id: str
    voice: str
    fmt: str
    duration_s: float
    created_label: str

    @property
    def summary(self) -> str:
        """Return the one-line description shown in the history row."""
        minutes, seconds = divmod(int(self.duration_s), 60)
        return (
            f"{self.created_label}  {self.voice} · {self.fmt} · "
            f"{minutes}:{seconds:02d}"
        )


class SpeechResultHistory(Vertical):
    """The list of takes generated this session, newest first."""

    def __init__(self, *, takes: Iterable[SpeechTake] = (), **kwargs: Any) -> None:
        """Create the history.

        Args:
            takes: Existing takes, oldest first; rendered newest first.
            kwargs: Forwarded to ``Vertical``.
        """
        classes = kwargs.pop("classes", "")
        super().__init__(classes=f"speech-history {classes}".strip(), **kwargs)
        self.takes = list(takes)

    def compose(self) -> ComposeResult:
        """Yield the section head then one row per take, newest first."""
        yield Static("Result", classes="speech-section-head")
        if not self.takes:
            yield Static(
                "No takes yet. Generate to synthesize the text above.",
                id="speech-history-empty",
                classes="speech-result-state",
                markup=False,
            )
            return
        for take in reversed(self.takes):
            with Horizontal(classes="speech-take-row"):
                yield Static(take.summary, markup=False)
                yield Button(
                    "Play",
                    id=f"speech-take-play-{take.take_id}",
                    classes="workbench-action",
                    compact=True,
                )
                yield Button(
                    "Export",
                    id=f"speech-take-export-{take.take_id}",
                    classes="workbench-action",
                    compact=True,
                )

    def add_take(self, take: SpeechTake) -> None:
        """Append a take and rebuild the list.

        Args:
            take: The newly generated take.
        """
        self.takes.append(take)
        self.refresh(recompose=True)
```

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/python -m pytest Tests/UI/test_speech_result_history.py -q`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/UI/Speech/speech_result_history.py Tests/UI/test_speech_result_history.py
git commit -m "feat(speech): result history so takes can be compared"
```

---

### Task 5: Assemble the pane and pin the responsive contract

**Files:**
- Modify: `tldw_chatbook/UI/Speech/speech_playground_pane.py`
- Modify: `tldw_chatbook/css/features/_lab.tcss`
- Test: `Tests/UI/test_speech_playground_pane.py`

**Interfaces:**
- Consumes: Tasks 1–4.
- Produces: `SpeechPlaygroundPane` composing axis row, text input, action strip, param group, result history, and the status line.

- [ ] **Step 1: Write the failing test**

```python
import pytest
from textual.widgets import Button, Static

from Tests.UI.test_screen_navigation import _build_test_app
from tldw_chatbook.UI.Screens.stts_screen import STTSScreen


async def _speech(app):
    screen = STTSScreen(app)
    await app.push_screen(screen)
    return screen


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(120, 40), (80, 24)])
async def test_nothing_is_truncated_or_below_the_fold_at_either_size(size):
    """The defect this phase exists to fix, asserted at both sizes.

    `Generate` used to render at y=60 in a 34-row viewport. Rendered text is
    the oracle -- `content_region.width` reported 16 for a 15-character
    label that did not render.
    """
    app = _build_test_app()
    async with app.run_test(size=size) as pilot:
        screen = await _speech(app)
        await pilot.pause()
        await pilot.pause()

        body = screen.query_one("#lab-body")
        generate = screen.query_one("#tts-generate-btn", Button)
        assert body.region.contains_region(generate.region), (
            f"Generate below the fold at {size}: y={generate.region.y}"
        )

        for widget in screen.query(".speech-chip").results(Static):
            text = str(widget.renderable)
            assert text in widget.render_line(0).text, f"truncated at {size}: {text!r}"


@pytest.mark.asyncio
async def test_the_pane_scrolls_rather_than_clipping_when_stacked():
    """`1fr` children compress instead of overflowing, which clips content
    that should scroll. Assert the pane is genuinely taller than its
    viewport when stacked."""
    app = _build_test_app()
    async with app.run_test(size=(80, 24)) as pilot:
        screen = await _speech(app)
        await pilot.pause()
        await pilot.pause()
        pane = screen.query_one("#speech-playground-pane")
        assert pane.has_class("speech-split-stacked")
        assert pane.virtual_size.height > pane.container_size.height
```

- [ ] **Step 2: Run it and watch it fail**

Run: `.venv/bin/python -m pytest Tests/UI/test_speech_playground_pane.py -q`
Expected: FAIL — `#tts-generate-btn` does not exist yet in the new pane.

- [ ] **Step 3: Rewrite the pane to use Tasks 1–4**

Replace `SpeechPlaygroundPane.compose` so it yields, in order: the title; the action strip (using `tts-generate-btn` and the other legacy ids so wiring in Task 6 is an id lookup, not a rename); `SpeechAxisRow`; the split with `TextArea` left and `SpeechResultHistory` right; `SpeechParamGroup`; the provider status line.

Keep the existing `on_resize`/`_sync_split_layout` stacking. Use the legacy control ids throughout — the rebuild is a re-siting, not a renaming, and Task 6 depends on that.

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/python -m pytest Tests/UI/test_speech_playground_pane.py -q`
Expected: 3 passed

- [ ] **Step 5: Look at it**

Capture at both sizes and open the images. A green suite has already missed a blank body, a dead collapse handle and three sparse layouts in this workstream.

```bash
.venv/bin/python - <<'EOF'
import asyncio, os
os.environ["TLDW_CONFIG_PATH"] = "/tmp/speech-plan.toml"
open("/tmp/speech-plan.toml","w").write('[general]\nusers_name="plan"\n\n[splash_screen]\nenabled=false\n')
from tldw_chatbook.app import TldwCli
from tldw_chatbook.UI.Navigation.main_navigation import NavigateToScreen
async def main():
    for w,h in ((120,40),(80,24)):
        app = TldwCli()
        async with app.run_test(size=(w,h)) as pilot:
            await pilot.pause(); await asyncio.sleep(4); await pilot.pause()
            app.post_message(NavigateToScreen("stts"))
            await pilot.pause(); await asyncio.sleep(3); await pilot.pause()
            app.save_screenshot(f"/tmp/speech-{w}x{h}.svg")
asyncio.run(main())
EOF
qlmanage -t -s 1600 -o /tmp /tmp/speech-120x40.svg /tmp/speech-80x24.svg
```

- [ ] **Step 6: Commit**

```bash
git add tldw_chatbook/UI/Speech/ Tests/UI/test_speech_playground_pane.py tldw_chatbook/css/
git commit -m "feat(speech): assemble the Playground pane and pin the responsive contract"
```

---

### Task 6 — REVISED after measurement

Two premises in the original Task 6 were wrong, both found by measuring the
rebuilt pane rather than reasoning about it.

**"Legacy ids make wiring an id lookup, not a translation table" — false for
actions.** `CommandStrip._button_id` rewrites every action id:
`tts-generate-btn` becomes `workbench-action-tts-generate-btn`. The legacy
handler compares `event.button.id == "tts-generate-btn"`, so it never
matches. Either the pane stops using `CommandStrip` for these, or the
handler is rewritten to strip the prefix. Prefer the former: the strip's
value was its Console-grammar styling, which plain Buttons can carry via
the same classes.

**"Assert all 57 controls are present" — impossible by construction.**
Provider parameters are provider-scoped, which is the whole point: with
audio.cpp selected, `chatterbox-settings` and Higgs' six knobs correctly do
not exist. A single set comparison can never pass. The check must iterate
providers and assert the union across them, with the always-present
controls asserted on every provider.

**Measured gap:** the rebuilt pane currently renders **10 of 57**. Missing
and genuinely needed regardless of provider: the audio player
(`audio-player-container`, `audio-progress-bar`, `audio-time-display`,
`audio-player-status`), generation status (`generation-status-container`,
`generation-status-text`, `generation-progress`, `tts-generation-log`),
provider status (`tts-provider-status`, `tts-audio-cpp-restrictions`),
reference audio (3 controls) and Higgs voice upload (4 controls).

So Task 6 splits:

- **6a — always-present controls.** Add player, generation status, provider
  status, reference-audio and voice-upload controls to the pane. No
  behaviour; just the surface, with the completeness check made
  provider-aware and passing.
- **6b — action identity.** Replace `CommandStrip` with plain Buttons
  carrying the exact legacy ids and the same `workbench-action` classes, so
  `event.button.id` matches what the handler expects.
- **6c — wiring.** Move the generate closure (`_generate_tts`,
  `_generation_readiness_error`, `_get_select_key`, `_is_valid_voice`,
  `_sync_generate_enabled` -- 5 methods, ~322 lines) onto a mixin the pane
  inherits. Because the pane uses the same control ids, their `query_one`
  calls resolve unchanged. Append a `SpeechTake` on completion.
- **6d — retire the legacy branch.** Deferred to task-1266, and not on
  test colour.

  6c makes Generate reach synthesis, but the rebuilt pane has no catalog:
  nothing populates the provider, model, voice, language or format options,
  so the axes render as empty selects and a generation attempt resolves
  nothing. Retiring the legacy branch now would ship a screen that looks
  complete and cannot synthesize.

  The catalog closure is **32 methods, ~771 lines** -- `_load_provider_catalog`
  and its worker, `_load_provider_voices` and its worker, `_apply_catalog`,
  the request-token staleness machinery, the provider status copy, and
  `_show_provider_specific_controls`. That is more than twice the generate
  closure and its own piece of work, not a step inside this one.

  Phase 1b since delivered all of it: catalog (717 lines), playback and
  export (672), and the last shared behaviour. The pane populates its axes
  live, synthesizes, and receives its results.

  Retirement is now blocked only on **coverage**, not behaviour. The legacy
  host's 43 tests exercise the shared mixins through it; swapping the
  harness to the rebuilt pane leaves 41 failing -- 27 because the tests
  query `TTSPlaygroundWidget` by type (mechanical), 14 that need
  classifying as pane defect or fixture difference. That is task-1266.

### Task 6 (original, superseded): Wire behaviour and retire the legacy playground

**Files:**
- Modify: `tldw_chatbook/UI/Speech/speech_playground_pane.py`
- Modify: `tldw_chatbook/UI/STTS_Window.py`
- Modify: `tldw_chatbook/UI/Screens/stts_screen.py`
- Test: `Tests/UI/test_speech_playground_wiring.py`

**Interfaces:**
- Consumes: everything above, plus the existing TTS service the legacy widget called.

This is the task that makes the pane real. Read how `TTSPlaygroundWidget` handles `tts-generate-btn` before writing anything: the handler, the worker it starts, and where it publishes the resulting audio. Re-site those calls; do not reimplement them.

- [ ] **Step 1: Write the failing test**

```python
import pytest

from Tests.UI.test_screen_navigation import _build_test_app
from tldw_chatbook.UI.Screens.stts_screen import STTSScreen


@pytest.mark.asyncio
async def test_generate_reaches_the_synthesis_path(monkeypatch):
    """Pressing Generate must call the same service the legacy widget did."""
    calls = []
    # Patch the synthesis entry point the legacy widget used; the exact
    # target is read from TTSPlaygroundWidget during implementation.
    ...


@pytest.mark.asyncio
async def test_a_completed_generation_appends_a_take():
    """The result must land in the history, or Generate has no visible
    consequence."""
    ...
```

Fill both in from what the legacy handler actually does — the plan deliberately does not guess the service signature.

- [ ] **Step 2: Confirm every control survived**

```bash
.venv/bin/python - <<'EOF'
import asyncio, os
os.environ["TLDW_CONFIG_PATH"] = "/tmp/speech-plan.toml"
from tldw_chatbook.app import TldwCli
from tldw_chatbook.UI.Navigation.main_navigation import NavigateToScreen
from tldw_chatbook.UI.Speech.speech_playground_model import ALL_PLAYGROUND_CONTROLS
async def main():
    app = TldwCli()
    async with app.run_test(size=(200, 60)) as pilot:
        await pilot.pause(); await asyncio.sleep(4); await pilot.pause()
        app.post_message(NavigateToScreen("stts"))
        await pilot.pause(); await asyncio.sleep(3); await pilot.pause()
        present = {w.id for w in app.screen.query("*") if w.id}
        missing = ALL_PLAYGROUND_CONTROLS - present
        print("MISSING:", sorted(missing))
asyncio.run(main())
EOF
```

Expected: `MISSING: []`. Controls inside the collapsed parameter group count as present — they are mounted, just collapsed. Anything genuinely absent is a dropped capability and blocks the task.

- [ ] **Step 3: Remove the legacy branch**

In `STTSWindow.watch_current_view`, delete the `playground` branch and the `TTSPlaygroundWidget` mount. Delete `_redesign_view` from `STTSScreen` and return `SpeechPlaygroundPane` for the playground unconditionally.

Do not delete `TTSPlaygroundWidget` itself yet — phase 6 owns the window's retirement, and the class may still be referenced by tests that later phases rewrite. Record it in the spec's retirement section instead.

- [ ] **Step 4: Full verification**

```bash
.venv/bin/python -m pytest Tests/UI/test_speech_playground_model.py \
  Tests/UI/test_speech_axis_row.py Tests/UI/test_speech_param_group.py \
  Tests/UI/test_speech_result_history.py Tests/UI/test_speech_playground_pane.py \
  Tests/UI/test_speech_playground_wiring.py Tests/UI/test_stts_capability_state.py \
  Tests/UI/test_lab_frame.py Tests/UI/test_lab_mode_strip.py \
  Tests/UI/test_destination_headers.py -q
.venv/bin/python tldw_chatbook/css/check_bundle_sync.py
```

Then run the full `Tests/UI` suite and classify every failure against a worktree pinned to the branch's parent. Pre-existing failures are reported, not inherited silently.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat(speech): wire the Playground pane and retire the legacy view branch"
```

---

## Self-review

**Spec coverage.** Grammar (Task 5), axis/knob split (Tasks 1–3), result history (Task 4), override visibility (Task 2), responsive contract (Task 5), all-controls-survive (Tasks 1 and 6). The spec's save-as-default action is **not** covered — it is named in the resolved-ownership section but has no task here, because it needs Settings to exist first. It belongs in phase 2 and is called out rather than silently dropped.

**Placeholders.** Task 6's tests are deliberately incomplete: the service signature is read from the legacy handler during implementation rather than guessed. Every other step carries real code.

**Type consistency.** `SpeechTake` fields are used identically in the dataclass, `summary`, and the test. `axis_chip_id` is used by both the widget and its test. `params_for_provider` returns a tuple everywhere.

**Risk.** Task 6 is the largest and the only one that can leave the screen non-functional. It is last, and every task before it is revertible without touching the legacy widget.
