# TTS Settings (Phase 2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild TTS Settings so its 79 controls are one row each instead of
four, Save is reachable without scrolling, and a provider's configured state
is visible before you expand it.

**Architecture:** A pure model classifies each provider's settings and says
whether that provider is configured, so the classification is testable
without mounting a 2,230-line widget. A `SpeechSettingsGroup` renders one
provider as a collapsible whose header carries its state; a
`SpeechSettingsPane` assembles the groups under a persistent action strip.
Settings keeps sole ownership of persisted defaults — it never reads
Playground state.

**Tech Stack:** Python ≥3.11, Textual 8.2.7, pytest + pytest-asyncio,
app-tier TCSS in `css/features/_lab.tcss` (rebuilt via `css/build_css.py`).

## What is actually wrong, measured

Driven on the running app and measured under `run_test`, not estimated:

| Measurement | Value |
| --- | --- |
| Rows per control | **4** — median 4, min 4, max 4, no variance |
| `save-settings-btn` position | **y=102** in a 26-row viewport (~4 screens down) |
| Controls | 64 mounted (79 ids incl. containers) |
| Collapsible groups | 8, of which **1** is expanded on arrival |
| `compose()` | 768 lines |

The 4-row cost is the whole defect in one number: a blank row, the input's
top border, the label+value row, the bottom border — spent on values like
`5.0` and `10000`. Horizontally the same inputs span ~200 cells for
three-character values.

Two consequences follow, and the plan targets them directly:

- **Save is unreachable.** 102 rows down, after every provider block. The
  primary action of a settings screen is four screens from where you land.
- **Collapsed state says nothing.** 55 of 64 controls sit behind disclosures
  with no indication of which providers are configured, which are at
  defaults, and which are incomplete. The spec's rule — "only the configured
  ones expanded" — cannot be honoured without first being able to answer
  that question.

## Global Constraints

- Spec: `Docs/superpowers/specs/2026-07-27-speech-console-redesign-design.md`.
  Where this plan and the spec disagree, the spec governs — raise it rather
  than choosing.
- **All 79 ids must remain reachable.** Enumerate by id; do not judge by
  looking. Provider-scoped ids need a provider-aware check, exactly as
  `test_speech_playground_completeness.py` does — a flat "all present"
  assertion cannot pass by construction.
- **Settings owns persisted defaults; it never reads Playground state.** One
  direction only.
- Save must be reachable without scrolling at 80×24 and at 200×60.
- Assert rendered text with the composited screen where a border or a parent
  can hide it. A widget's own `render_line` is in its own coordinate space
  and shows neither — that is how five axis selects shipped rendering only
  their top border.
- Tests that must see app-tier CSS have to run under the real app. A bare
  `App` harness never loads the bundle, so a bundle rule is invisible to it
  in both directions.
- `@on` inside a plain mixin is **never dispatched** — Textual registers
  decorated handlers in its metaclass, per class. Declare handlers on the
  host.
- `remove()` is deferred. Mounting a replacement immediately raises
  `DuplicateIds`; reconcile to current state in a `call_after_refresh`
  callback instead.
- CSS goes in `css/features/_lab.tcss` and the bundle is rebuilt
  (`python tldw_chatbook/css/build_css.py`). Never hand-edit
  `tldw_cli_modular.tcss`.
- Mutation-check every new guard: break what it protects and confirm it
  fails. A guard that passes on first write is unproven.
- Run tests with the repo venv: `.venv/bin/python -m pytest`.

---

### Task 1: The settings model

**Files:**
- Create: `tldw_chatbook/UI/Speech/speech_settings_model.py`
- Test: `Tests/UI/test_speech_settings_model.py`

**Interfaces:**
- Produces: `PROVIDER_SETTINGS: dict[str, tuple[str, ...]]`,
  `SETTINGS_PROVIDER_ORDER: tuple[str, ...]`,
  `settings_for_provider(provider: str) -> tuple[str, ...]`,
  `ALL_SETTINGS_CONTROLS: frozenset[str]`,
  `configured_state(provider: str, values: Mapping[str, object]) -> str`
  returning one of `"configured" | "default" | "incomplete"`.

The id inventory comes from the legacy widget, recovered the same way the
Playground's did — read it out of `TTSSettingsWidget.compose()` rather than
retyping it, then freeze it as the yardstick.

`configured_state` is what makes the spec's "only the configured ones
expanded" rule implementable, and what the collapsed header will state. A
provider is `"configured"` when it holds at least one non-default value,
`"incomplete"` when a required field is empty while others are set, and
`"default"` otherwise.

- [ ] **Step 1: Write the failing test**

```python
import pytest

from tldw_chatbook.UI.Speech.speech_settings_model import (
    PROVIDER_SETTINGS,
    configured_state,
    settings_for_provider,
)


@pytest.mark.unit
def test_a_provider_holding_a_custom_value_reads_as_configured():
    """This is what decides which groups open on arrival.

    Without it the spec's "only the configured ones expanded" rule cannot be
    implemented, and the user is back to eight identical closed boxes.
    """
    state = configured_state(
        "audio_cpp",
        {"audio-cpp-base-url-input": "http://192.168.1.5:9000"},
    )
    assert state == "configured"


@pytest.mark.unit
def test_a_provider_at_its_defaults_reads_as_default():
    assert configured_state("audio_cpp", {}) == "default"


@pytest.mark.unit
def test_a_half_filled_provider_reads_as_incomplete():
    """Half-configured is the state worth surfacing: it is the one that
    fails at generation time with nothing on screen having said so."""
    state = configured_state(
        "elevenlabs",
        {"elevenlabs-api-key-input": "", "elevenlabs-voice-id-input": "abc"},
    )
    assert state == "incomplete"


@pytest.mark.unit
def test_an_unknown_provider_yields_no_settings_rather_than_raising():
    """compose() calls this; raising would take the screen down."""
    assert settings_for_provider("nonexistent") == ()


@pytest.mark.unit
def test_no_setting_is_claimed_by_two_providers():
    """A shared id would be written twice and read back wrong."""
    seen: set[str] = set()
    for controls in PROVIDER_SETTINGS.values():
        assert not (seen & set(controls))
        seen |= set(controls)
```

- [ ] **Step 2: Run it and watch it fail**

Run: `.venv/bin/python -m pytest Tests/UI/test_speech_settings_model.py -q`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Recover the inventory from the legacy widget**

Do not retype 79 ids. Read them out of `TTSSettingsWidget`, grouped by the
prefix the spec already documents (chatterbox 17, higgs 16, audio.cpp 15,
kokoro 8, elevenlabs 7, defaults 5, alltalk 4, openai 3):

```bash
.venv/bin/python - <<'PY'
import ast, pathlib, re
src = pathlib.Path("tldw_chatbook/UI/STTS_Window.py").read_text()
cls = next(n for n in ast.parse(src).body
           if isinstance(n, ast.ClassDef) and n.name == "TTSSettingsWidget")
body = "\n".join(src.split("\n")[cls.lineno - 1:cls.end_lineno])
print(sorted(set(re.findall(r'id="([\w-]+)"', body))))
PY
```

- [ ] **Step 4: Write the model, then re-run to green**

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/UI/Speech/speech_settings_model.py \
        Tests/UI/test_speech_settings_model.py
git commit -m "feat(speech): classify TTS settings by provider and configured state"
```

---

### Task 2: One provider, one row per setting

**Files:**
- Create: `tldw_chatbook/UI/Speech/speech_settings_group.py`
- Modify: `tldw_chatbook/css/features/_lab.tcss`
- Test: `Tests/UI/test_speech_settings_group.py`

**Interfaces:**
- Consumes: `settings_for_provider`, `configured_state` (Task 1)
- Produces: `SpeechSettingsGroup(Collapsible)` with
  `provider: str`, `state: str`, and `SETTING_LABELS: dict[str, str]`

Build children by passing them to `Collapsible.__init__`. **Do not override
`compose()`** — subclassing `Collapsible` and overriding `compose()` replaces
its title row and the contents container it toggles, so the group renders
fully expanded while still reporting `collapsed is True`. That bug passed a
flag-only assertion in phase 1; assert the contents' rendered height instead.

The header states the provider and its state, so a closed group is still
informative:

```
▶ ElevenLabs · configured
▶ Kokoro · defaults
▶ Higgs · incomplete — API key missing
```

- [ ] **Step 1: Write the failing test**

```python
import pytest
from textual.app import App, ComposeResult
from textual.widgets import Collapsible

from tldw_chatbook.UI.Speech.speech_settings_group import SpeechSettingsGroup


class _Harness(App[None]):
    def __init__(self, provider, values):
        super().__init__()
        self._provider, self._values = provider, values

    def compose(self) -> ComposeResult:
        yield SpeechSettingsGroup(provider=self._provider, values=self._values)


@pytest.mark.asyncio
async def test_a_collapsed_group_still_states_its_provider_state():
    """Eight identical closed boxes tell the user nothing. The header has to
    carry the answer to "is this one set up?" without being opened."""
    app = _Harness("elevenlabs", {"elevenlabs-api-key-input": "sk-xxx"})
    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.pause()
        title = app.query_one(Collapsible).title
        assert "ElevenLabs" in title
        assert "configured" in title.lower()


@pytest.mark.asyncio
async def test_the_group_collapses_for_real():
    """Assert what renders, not the flag. Overriding Collapsible.compose()
    replaces the contents container it toggles, so the group renders open
    while reporting collapsed is True."""
    app = _Harness("elevenlabs", {})
    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.pause()
        group = app.query_one(Collapsible)
        assert group.collapsed is True
        control = app.query_one("#elevenlabs-api-key-input")
        assert not control.region.height, "collapsed group still renders"


@pytest.mark.asyncio
async def test_each_setting_costs_one_row_not_four():
    """The defect this phase exists to fix, asserted as a number.

    The legacy form spent 4 rows on every control -- measured median 4, min
    4, max 4 -- which is why Save sat at y=102.
    """
    app = _Harness("audio_cpp", {})
    async with app.run_test(size=(160, 60)) as pilot:
        await pilot.pause()
        app.query_one(Collapsible).collapsed = False
        await pilot.pause()
        await pilot.pause()
        rows = [w.region.y for w in app.query(".speech-setting-row") if w.region.height]
        gaps = [b - a for a, b in zip(sorted(rows), sorted(rows)[1:]) if b > a]
        assert gaps and max(gaps) == 1, f"settings still cost {max(gaps)} rows each"
```

- [ ] **Step 2: Run it and watch it fail**
- [ ] **Step 3: Write the group and its CSS, then re-run to green**
- [ ] **Step 4: Mutation-check the row-cost guard** — set the row height to 2
      in `_lab.tcss`, rebuild the bundle, confirm the test fails, revert.
- [ ] **Step 5: Commit**

---

### Task 3: The pane, with Save reachable

**Files:**
- Create: `tldw_chatbook/UI/Speech/speech_settings_pane.py`
- Modify: `tldw_chatbook/css/features/_lab.tcss`
- Test: `Tests/UI/test_speech_settings_pane.py`

**Interfaces:**
- Consumes: `SpeechSettingsGroup` (Task 2), the model (Task 1)
- Produces: `SpeechSettingsPane` mounting `#save-settings-btn` and one group
  per provider in `SETTINGS_PROVIDER_ORDER`

Actions go in a `SpeechActionStrip` at the top, reusing the Playground's.
**Do not use `CommandStrip`** — it rewrites every action id
(`save-settings-btn` becomes `workbench-action-save-settings-btn`), and the
legacy handler matches on the bare id, so the button renders and never
fires.

- [ ] **Step 1: Write the failing test**

```python
import pytest
from textual.widgets import Button

from Tests.UI.test_screen_navigation import _build_test_app
from tldw_chatbook.UI.Screens.stts_screen import STTSScreen


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(200, 60), (80, 24)])
async def test_save_is_reachable_without_scrolling(size):
    """The defect this phase exists to fix: Save measured at y=102."""
    app = _build_test_app()
    async with app.run_test(size=size) as pilot:
        screen = STTSScreen(app)
        await app.push_screen(screen)
        await pilot.pause()
        row = next(b for b in screen.query(Button)
                   if getattr(b, "lab_view_key", None) == "settings")
        row.press()
        for _ in range(6):
            await pilot.pause()

        body = screen.query_one("#lab-body")
        save = screen.query_one("#save-settings-btn", Button)
        assert body.region.contains_region(save.region), (
            f"Save below the fold at {size}: y={save.region.y}"
        )
```

- [ ] **Step 2: Run it and watch it fail** — it should report a y far below
      the fold, which is the current behaviour.
- [ ] **Step 3: Write the pane, then re-run to green**
- [ ] **Step 4: Commit**

---

### Task 4: Completeness, provider-aware

**Files:**
- Test: `Tests/UI/test_speech_settings_completeness.py`

Mirror `test_speech_playground_completeness.py`: assert the union across
providers covers every required id, and that the always-present surface is
present for each. A flat "all 79 mounted" assertion cannot pass — settings
are provider-scoped, which is the point.

Name any legacy container the rebuild deliberately drops in a
`REPLACED_CONTAINERS`-style set, so dropping one is a recorded decision and
dropping anything else still fails.

- [ ] **Step 1: Write the test**
- [ ] **Step 2: Run and fix whatever it names**
- [ ] **Step 3: Mutation-check** — remove one control from the pane, confirm
      the test names it, restore.
- [ ] **Step 4: Commit**

---

### Task 5: Wire, verify live, retire

**Measured before starting** (Tasks 1-4 are committed; this is what remains):

| | |
| --- | --- |
| Behaviour closure, excluding `compose()` | **21 methods, 1190 lines** |
| `_save_settings` | 311 lines — **writes user config** |
| `_set_initial_values` | 238 lines — fills the pane's `values` |
| Decorated methods | `_discover_audio_cpp` (`@work`), `_normalize_openai_base_url` (`@classmethod`) |
| Module names the tests patch | `get_cli_setting`, `get_tts_service` |
| Test files still patching `STTS_Window` | 1 |

`@work` survives a mixin move — Textual resolves it at call time. `@on` would
not, but this closure has none.

**Persistence is the risk here, and it is different in kind from phases 1-4.**
Everything so far has been layout: wrong, but recoverable by looking at it.
`_save_settings` writes the user's real config. Verify it against a scratch
`TLDW_CONFIG_PATH` profile and diff the written file before and after — do
not test it against the live config, and do not judge it by "no exception
was raised".

**Files:**
- Modify: `tldw_chatbook/UI/STTS_Window.py` (the `settings` view branch)
- Modify: `tldw_chatbook/UI/Screens/stts_screen.py` if the rail needs it
- Test: `Tests/UI/test_stts_settings_widget.py` (retarget)

Read how `TTSSettingsWidget` saves before writing anything: the handler, what
it writes, and where. Re-site those calls; do not reimplement persistence.

Expect the same two seams that bit phase 1:

- Tests patch `get_cli_setting` on the `STTS_Window` module. Moved code
  resolves it from its own module, so the patch silently detaches. Give the
  pane a `_cli_setting` hook and override it where the tests already patch.
- Patches may be **inline** in individual tests as well as in a shared
  fixture. Grep for both; porting only the fixture leaves tests hitting the
  real config.

- [ ] **Step 1: Retarget `test_stts_settings_widget.py`, run, fix**
- [ ] **Step 2: Run the whole speech suite**
- [ ] **Step 3: Drive the running app** — open Settings, confirm Save is on
      screen, confirm a configured provider opens and a default one does
      not, change a value and save, reopen and confirm it persisted
- [ ] **Step 4: Delete `TTSSettingsWidget` only once the above is green**
- [ ] **Step 5: Commit**

---

## Deferred, and why

**"Save as default" from the Playground.** The spec requires it — comparison
without a way to commit the winner is half a tool — and it needs Settings to
own a write path first. It lands after Task 5, not inside it.

## Self-review

- Spec coverage: the spec's Settings paragraph asks for "one collapsible
  block per provider, only the configured ones expanded". Task 1 makes
  "configured" answerable, Task 2 renders it, Task 3 orders it.
- Every task ends with a committed, testable deliverable.
- No task depends on a later one.
- The measurements in this plan are recorded values, not estimates: 4 rows
  per control, Save at y=102, 8 groups with 1 expanded, 768-line compose.
