# Console Setup Card (Phase 2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the Console's provider blocker banner and static first-run copy with a live "Get started" setup card whose steps reflect real provider/model state, collapsing to one quiet line after setup and disappearing forever after the first successful send (persisted flag).

**Architecture:** A new pure module (`console_onboarding_state.py`) derives the card state from the existing readiness single source (`build_console_settings_readiness`). The existing `ConsoleTranscriptEmptyPanel` (already row-managed by `ConsoleTranscript`) renders that state, preserving its widget ids and adaptive primary action. `chat_screen.py` keeps orchestration only: it builds the state in the existing `_sync_console_transcript_guidance()` hook, records the persisted `console.onboarding.first_send_completed` flag on the first accepted send (mirroring the rail-prefs persistence pattern), and drops the legacy recovery-strip banner.

**Tech Stack:** Python 3.11+, Textual, pytest + pytest-asyncio (pilot), existing `_build_test_app`/`ConsoleHarness` UI harness.

**Spec:** `Docs/superpowers/specs/2026-07-02-console-dual-audience-ux-design.md` §2 (First-run experience), §5 (single readiness source). Phase 1 merged as PR #576; this plan builds on post-merge `dev`.

## Global Constraints

- Run tests with the venv interpreter and isolated home:
  `env HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share .venv/bin/python -m pytest -q <target> --tb=short`
- The `timeout` shell command is not available in this environment.
- `tldw_chatbook/css/tldw_cli_modular.tcss` is **generated**: edit only `tldw_chatbook/css/components/_agentic_terminal.tcss`, run `./build_css.sh`, commit both.
- The five Console UI suites are fully green on dev since PR #576 — there is NO expected-failure baseline anymore; any failure you cause is yours.
- Exact copy (spec §2): card title `Get started`; step labels `Add an API key` / `Pick a model` / `Send your first message`; ready line `Ready — type a message to begin.`; post-first-send empty line `No messages yet.`; composer blocked reasons use the `Send blocked — <action> to continue` form.
- Step glyphs: `✓` done, `●` active (current step), `○` pending (spec §4 glyph language).
- Lifecycle rules (spec §2): full card while setup incomplete and transcript empty; one ready line when setup complete but user has never sent; quiet `No messages yet.` forever after the first accepted send — including new tabs and workspaces (the flag is global, not per-scope).
- Behavior changes mandated by spec (update tests, preserving intent): the top provider recovery strip is REMOVED (its action moves into the card's primary button); the ready state shows only the one line (no action-button row).
- Existing widget ids must survive: `console-transcript-empty-state`, `console-empty-title`, `console-empty-body`, `console-empty-action-row`, `console-empty-choose-model`, `console-empty-attach-context`, `console-empty-run-library-rag`. New ids: `console-setup-step-1`, `console-setup-step-2`, `console-setup-step-3`.
- **Stage only files you changed (`git add <specific paths>`). NEVER `git add -A`, `git add .`, or `git commit -a`.** Never touch `.claude/settings.local.json`.
- Commit messages end with: `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`
- Console screen changes require live screenshot QA + explicit user approval before merge (Task 7).
- Work on branch `claude/console-setup-card-phase2` cut from current `dev`.

## File Structure

- Create `tldw_chatbook/Chat/console_onboarding_state.py` — pure card-state contracts (no Textual imports).
- Modify `tldw_chatbook/Chat/console_display_state.py` — `build_console_disabled_reason` copy update only.
- Modify `tldw_chatbook/Widgets/Console/console_transcript.py` — `ConsoleTranscriptEmptyPanel` renders `ConsoleSetupCardState`.
- Modify `tldw_chatbook/UI/Screens/chat_screen.py` — state building in `_sync_console_transcript_guidance()`, first-send flag read/write, banner removal.
- Modify `tldw_chatbook/css/components/_agentic_terminal.tcss` (+ regenerate `tldw_cli_modular.tcss`).
- Tests: new `Tests/Chat/test_console_onboarding_state.py`; extend `Tests/Chat/test_console_display_state.py`, `Tests/UI/test_console_internals_decomposition.py`, `Tests/UI/test_console_native_chat_flow.py`.

---

### Task 1: Pure onboarding state module

**Files:**
- Create: `tldw_chatbook/Chat/console_onboarding_state.py`
- Test: `Tests/Chat/test_console_onboarding_state.py` (new)

**Interfaces:**
- Consumes: `ConsoleSettingsReadiness` (`tldw_chatbook/Chat/console_session_settings.py:174` — fields `label: str`, `detail: str`, `native_send_supported: bool`).
- Produces (Tasks 3–4 rely on these exact names):
  - `CONSOLE_SETUP_CARD_TITLE = "Get started"`
  - `CONSOLE_READY_EMPTY_COPY = "Ready — type a message to begin."`
  - `CONSOLE_QUIET_EMPTY_COPY = "No messages yet."`
  - `CONSOLE_SETUP_STEP_GLYPHS = {"done": "✓", "active": "●", "pending": "○"}`
  - `@dataclass(frozen=True) ConsoleSetupStep(state: str, label: str, detail: str = "")` with property `glyph -> str`
  - `@dataclass(frozen=True) ConsoleSetupCardState(mode: str, steps: tuple[ConsoleSetupStep, ...] = (), body_copy: str = "")` — `mode` is `"card" | "ready_line" | "quiet"`
  - `build_console_setup_card_state(*, readiness: ConsoleSettingsReadiness, provider_label: str, has_model: bool, first_send_completed: bool, has_messages: bool, guidance_dismissed: bool) -> ConsoleSetupCardState`
  - `coerce_console_first_send_completed(raw: Any) -> bool` (missing/invalid → `False`; accepts bool/int/str truthy forms like the rail-state `_coerce_bool`)

- [ ] **Step 1: Write the failing tests**

Create `Tests/Chat/test_console_onboarding_state.py`:

```python
"""Pure Console setup-card state contracts."""

from tldw_chatbook.Chat.console_onboarding_state import (
    CONSOLE_QUIET_EMPTY_COPY,
    CONSOLE_READY_EMPTY_COPY,
    CONSOLE_SETUP_CARD_TITLE,
    ConsoleSetupCardState,
    build_console_setup_card_state,
    coerce_console_first_send_completed,
)
from tldw_chatbook.Chat.console_session_settings import ConsoleSettingsReadiness


def _readiness(label: str, *, ready: bool = False, detail: str = "") -> ConsoleSettingsReadiness:
    return ConsoleSettingsReadiness(
        label=label,
        detail=detail,
        native_send_supported=ready,
    )


def _build(**overrides) -> ConsoleSetupCardState:
    defaults = dict(
        readiness=_readiness("Missing key"),
        provider_label="OpenAI",
        has_model=True,
        first_send_completed=False,
        has_messages=False,
        guidance_dismissed=False,
    )
    defaults.update(overrides)
    return build_console_setup_card_state(**defaults)


def test_missing_key_renders_card_with_api_key_step_active():
    state = _build()
    assert state.mode == "card"
    assert CONSOLE_SETUP_CARD_TITLE == "Get started"
    assert [step.state for step in state.steps] == ["active", "done", "pending"]
    assert state.steps[0].label == "Add an API key"
    assert state.steps[0].glyph == "●"
    assert state.steps[1].label == "Pick a model"
    assert state.steps[2].label == "Send your first message"
    assert state.steps[2].glyph == "○"


def test_endpoint_problems_relabel_step_one():
    assert _build(readiness=_readiness("Invalid URL")).steps[0].label == "Save the provider endpoint"
    assert _build(readiness=_readiness("Endpoint not saved")).steps[0].label == "Save the provider endpoint"
    assert _build(readiness=_readiness("Unknown")).steps[0].label == "Choose a supported provider"
    assert _build(readiness=_readiness("Pending")).steps[0].label == "Choose a send-capable provider"


def test_provider_ready_without_model_activates_model_step():
    state = _build(readiness=_readiness("Ready", ready=True), has_model=False)
    assert state.mode == "card"
    assert [step.state for step in state.steps] == ["done", "active", "pending"]
    assert state.steps[0].glyph == "✓"
    assert state.steps[0].detail == "OpenAI ready"


def test_setup_complete_collapses_to_ready_line():
    state = _build(readiness=_readiness("Ready", ready=True), has_model=True)
    assert state.mode == "ready_line"
    assert state.body_copy == CONSOLE_READY_EMPTY_COPY
    assert state.steps == ()


def test_first_send_completed_is_quiet_forever():
    state = _build(
        readiness=_readiness("Ready", ready=True),
        first_send_completed=True,
    )
    assert state.mode == "quiet"
    assert state.body_copy == CONSOLE_QUIET_EMPTY_COPY
    # Quiet wins even when setup is incomplete on a fresh scope.
    assert _build(first_send_completed=True).mode == "quiet"


def test_messages_present_is_quiet():
    assert _build(has_messages=True).mode == "quiet"


def test_dismissal_hides_ready_line_but_not_setup_card():
    ready = _build(
        readiness=_readiness("Ready", ready=True),
        guidance_dismissed=True,
    )
    assert ready.mode == "quiet"
    blocked = _build(guidance_dismissed=True)
    assert blocked.mode == "card"


def test_coerce_first_send_completed():
    assert coerce_console_first_send_completed(True) is True
    assert coerce_console_first_send_completed("true") is True
    assert coerce_console_first_send_completed(1) is True
    assert coerce_console_first_send_completed(None) is False
    assert coerce_console_first_send_completed("no") is False
    assert coerce_console_first_send_completed({}) is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `env HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share .venv/bin/python -m pytest -q Tests/Chat/test_console_onboarding_state.py --tb=short`

Expected: FAIL with `ModuleNotFoundError: No module named 'tldw_chatbook.Chat.console_onboarding_state'`.

- [ ] **Step 3: Implement the module**

Create `tldw_chatbook/Chat/console_onboarding_state.py`:

```python
"""Pure first-run setup-card state contracts for the native Console."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from tldw_chatbook.Chat.console_session_settings import ConsoleSettingsReadiness

CONSOLE_SETUP_CARD_TITLE = "Get started"
CONSOLE_READY_EMPTY_COPY = "Ready — type a message to begin."
CONSOLE_QUIET_EMPTY_COPY = "No messages yet."
CONSOLE_SETUP_STEP_GLYPHS = {"done": "✓", "active": "●", "pending": "○"}

_STEP_ONE_LABELS = {
    "Missing key": "Add an API key",
    "Invalid URL": "Save the provider endpoint",
    "Endpoint not saved": "Save the provider endpoint",
    "Unknown": "Choose a supported provider",
    "Pending": "Choose a send-capable provider",
}
_TRUE_STRINGS = {"true", "yes", "1", "on"}


@dataclass(frozen=True)
class ConsoleSetupStep:
    """One numbered step in the Console first-run setup card."""

    state: str
    label: str
    detail: str = ""

    @property
    def glyph(self) -> str:
        return CONSOLE_SETUP_STEP_GLYPHS.get(self.state, "○")


@dataclass(frozen=True)
class ConsoleSetupCardState:
    """Display state for the Console empty-transcript onboarding surface."""

    mode: str
    steps: tuple[ConsoleSetupStep, ...] = ()
    body_copy: str = ""


def coerce_console_first_send_completed(raw: Any) -> bool:
    """Normalize the persisted first-send flag; anything unrecognized is False."""
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, int):
        return raw != 0
    if isinstance(raw, str):
        return raw.strip().lower() in _TRUE_STRINGS
    return False


def build_console_setup_card_state(
    *,
    readiness: ConsoleSettingsReadiness,
    provider_label: str,
    has_model: bool,
    first_send_completed: bool,
    has_messages: bool,
    guidance_dismissed: bool,
) -> ConsoleSetupCardState:
    """Derive the onboarding surface state from the readiness single source.

    Args:
        readiness: Current settings readiness from
            ``build_console_settings_readiness``.
        provider_label: User-facing provider name for the done-step detail.
        has_model: Whether a model is selected for the session.
        first_send_completed: Persisted global flag; once True the card never
            returns, including new tabs and workspaces.
        has_messages: Whether the active transcript has any messages.
        guidance_dismissed: In-session dismissal (user started composing).

    Returns:
        Card state: full ``card`` while setup is incomplete, one ``ready_line``
        when setup is complete but nothing was ever sent, ``quiet`` otherwise.
    """
    if has_messages or first_send_completed:
        return ConsoleSetupCardState(mode="quiet", body_copy=CONSOLE_QUIET_EMPTY_COPY)

    provider_done = bool(readiness.native_send_supported)
    if provider_done and has_model:
        if guidance_dismissed:
            return ConsoleSetupCardState(mode="quiet", body_copy=CONSOLE_QUIET_EMPTY_COPY)
        return ConsoleSetupCardState(mode="ready_line", body_copy=CONSOLE_READY_EMPTY_COPY)

    provider_name = str(provider_label or "Provider").strip() or "Provider"
    step_one = ConsoleSetupStep(
        state="done" if provider_done else "active",
        label=(
            "Provider ready"
            if provider_done
            else _STEP_ONE_LABELS.get(readiness.label, "Finish provider setup")
        ),
        detail=f"{provider_name} ready" if provider_done else "",
    )
    step_two = ConsoleSetupStep(
        state="done" if has_model else ("active" if provider_done else "pending"),
        label="Pick a model",
    )
    step_three = ConsoleSetupStep(
        state="pending",
        label="Send your first message",
        detail="Type below, Enter to send",
    )
    return ConsoleSetupCardState(
        mode="card",
        steps=(step_one, step_two, step_three),
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Same command as Step 2. Expected: PASS (8 tests).

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Chat/console_onboarding_state.py Tests/Chat/test_console_onboarding_state.py
git commit -m "feat(console): add pure setup-card onboarding state"
```

---

### Task 2: Composer blocked-reason copy

**Files:**
- Modify: `tldw_chatbook/Chat/console_display_state.py` (`build_console_disabled_reason`, lines 55–94)
- Test: `Tests/Chat/test_console_display_state.py`

**Interfaces:**
- Consumes/Produces: `build_console_disabled_reason(*, action_id, has_draft, send_blocked, setup_blocked_reason="") -> str` — signature unchanged; only the setup-blocked strings change to the spec form. The empty-draft string `"Send disabled: type a message"` is NOT a setup blocker and stays as-is.

- [ ] **Step 1: Update the existing tests' expected strings (failing first)**

In `Tests/Chat/test_console_display_state.py`, find the tests covering `build_console_disabled_reason` (grep `Send disabled:`). Update every setup-blocked expectation to the new copy, and add one new case:

| Old expected | New expected |
|---|---|
| `Send disabled: choose a model` | `Send blocked — choose a model to continue` |
| `Send disabled: add API key` | `Send blocked — add an API key to continue` |
| `Send disabled: configure endpoint` | `Send blocked — configure the endpoint to continue` |
| `Send disabled: choose a provider` | `Send blocked — choose a provider to continue` |
| `Send disabled: finish provider setup` | `Send blocked — finish provider setup to continue` |
| `Send disabled: type a message` | unchanged |

- [ ] **Step 2: Run tests to verify they fail**

Run: `env HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share .venv/bin/python -m pytest -q Tests/Chat/test_console_display_state.py -k disabled_reason --tb=short`

Expected: FAIL on the updated strings.

- [ ] **Step 3: Update the implementation**

In `build_console_disabled_reason` (console_display_state.py:74–94), replace the setup-blocked returns:

```python
    if send_blocked and setup_reason:
        if "model" in setup_reason_lower:
            return "Send blocked — choose a model to continue"
        if "api key" in setup_reason_lower:
            return "Send blocked — add an API key to continue"
        if "endpoint" in setup_reason_lower:
            return "Send blocked — configure the endpoint to continue"
        if (
            "choose a provider" in setup_reason_lower
            or "missing provider" in setup_reason_lower
        ):
            return "Send blocked — choose a provider to continue"
        return "Send blocked — finish provider setup to continue"
    if not has_draft:
        return "Send disabled: type a message"
    return ""
```

- [ ] **Step 4: Run tests + downstream consumers**

```bash
env HOME=... .venv/bin/python -m pytest -q Tests/Chat/test_console_display_state.py --tb=short
env HOME=... .venv/bin/python -m pytest -q Tests/UI/test_console_internals_decomposition.py -k "disabled or blocked" --tb=short
```

Expected: display-state suite PASS. Any UI test asserting the old `Send disabled:` setup strings fails here — update those expectations to the new copy in the same commit (intent preserved: the reason is still surfaced inline).

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Chat/console_display_state.py Tests/Chat/test_console_display_state.py Tests/UI/test_console_internals_decomposition.py
git commit -m "feat(console): spec-form send-blocked reasons"
```

---

### Task 3: Setup card rendering in the empty panel

**Files:**
- Modify: `tldw_chatbook/Widgets/Console/console_transcript.py` (`ConsoleTranscriptEmptyPanel`, lines 221–291; `ConsoleTranscript.sync_empty_state`, lines 333–358)
- Test: `Tests/UI/test_console_rail_sections.py` (append widget-level tests; the file already mounts small Console widgets in a bare `App`)

**Interfaces:**
- Consumes: Task 1's `ConsoleSetupCardState`, `ConsoleSetupStep`, `CONSOLE_SETUP_CARD_TITLE`.
- Produces (Task 4 relies on): `ConsoleTranscriptEmptyPanel.__init__(card_state: ConsoleSetupCardState, *, provider_action_label: str, provider_action_tooltip: str)` and `sync_card_state(card_state, *, provider_action_label, provider_action_tooltip) -> None`; `ConsoleTranscript.sync_empty_state(...)` gains the same `card_state` keyword and forwards it. Step rows render as `Static` ids `console-setup-step-1..3`, classes `console-setup-step` plus `console-setup-step-done|active|pending`; each row text is `f"{index}. {step.glyph} {step.label}"` plus `f"  {step.detail}"` when detail is non-empty. Existing ids stay: title `console-empty-title` (text `Get started` in card mode; hidden via `display=none` in ready_line/quiet), body `console-empty-body` (renders `body_copy` in ready_line/quiet; hidden in card mode), action row `console-empty-action-row` (visible only in card mode).

- [ ] **Step 1: Write the failing widget tests**

Append to `Tests/UI/test_console_rail_sections.py`:

```python
from tldw_chatbook.Chat.console_onboarding_state import (
    CONSOLE_QUIET_EMPTY_COPY,
    CONSOLE_READY_EMPTY_COPY,
    ConsoleSetupCardState,
    ConsoleSetupStep,
)
from tldw_chatbook.Widgets.Console.console_transcript import ConsoleTranscriptEmptyPanel


def _card_state() -> ConsoleSetupCardState:
    return ConsoleSetupCardState(
        mode="card",
        steps=(
            ConsoleSetupStep(state="active", label="Add an API key"),
            ConsoleSetupStep(state="done", label="Pick a model"),
            ConsoleSetupStep(
                state="pending",
                label="Send your first message",
                detail="Type below, Enter to send",
            ),
        ),
    )


class _SetupPanelApp(App):
    def __init__(self, state: ConsoleSetupCardState) -> None:
        super().__init__()
        self._state = state

    def compose(self):
        yield ConsoleTranscriptEmptyPanel(
            self._state,
            provider_action_label="Configure API",
            provider_action_tooltip="Open provider settings.",
        )


@pytest.mark.asyncio
async def test_setup_panel_card_mode_renders_steps_and_actions():
    app = _SetupPanelApp(_card_state())
    async with app.run_test(size=(100, 30)):
        title = app.query_one("#console-empty-title", Static)
        assert "Get started" in str(getattr(title.renderable, "plain", title.renderable))
        step1 = app.query_one("#console-setup-step-1", Static)
        text1 = str(getattr(step1.renderable, "plain", step1.renderable))
        assert "1. ● Add an API key" in text1
        step2 = app.query_one("#console-setup-step-2", Static)
        assert "2. ✓ Pick a model" in str(getattr(step2.renderable, "plain", step2.renderable))
        step3 = app.query_one("#console-setup-step-3", Static)
        text3 = str(getattr(step3.renderable, "plain", step3.renderable))
        assert "3. ○ Send your first message" in text3
        assert "Type below, Enter to send" in text3
        assert app.query_one("#console-empty-action-row").styles.display != "none"
        assert app.query_one("#console-empty-body").styles.display == "none"


@pytest.mark.asyncio
async def test_setup_panel_ready_line_hides_steps_and_actions():
    app = _SetupPanelApp(
        ConsoleSetupCardState(mode="ready_line", body_copy=CONSOLE_READY_EMPTY_COPY)
    )
    async with app.run_test(size=(100, 30)):
        body = app.query_one("#console-empty-body", Static)
        assert CONSOLE_READY_EMPTY_COPY in str(getattr(body.renderable, "plain", body.renderable))
        assert not list(app.query("#console-setup-step-1"))
        assert app.query_one("#console-empty-action-row").styles.display == "none"
        assert app.query_one("#console-empty-title").styles.display == "none"


@pytest.mark.asyncio
async def test_setup_panel_quiet_mode_shows_only_quiet_copy():
    app = _SetupPanelApp(
        ConsoleSetupCardState(mode="quiet", body_copy=CONSOLE_QUIET_EMPTY_COPY)
    )
    async with app.run_test(size=(100, 30)):
        body = app.query_one("#console-empty-body", Static)
        assert CONSOLE_QUIET_EMPTY_COPY in str(getattr(body.renderable, "plain", body.renderable))
        assert not list(app.query("#console-setup-step-1"))
        assert app.query_one("#console-empty-action-row").styles.display == "none"


@pytest.mark.asyncio
async def test_setup_panel_sync_card_state_transitions_modes():
    app = _SetupPanelApp(_card_state())
    async with app.run_test(size=(100, 30)) as pilot:
        panel = app.query_one(ConsoleTranscriptEmptyPanel)
        panel.sync_card_state(
            ConsoleSetupCardState(mode="ready_line", body_copy=CONSOLE_READY_EMPTY_COPY),
            provider_action_label="Choose model",
            provider_action_tooltip="Pick a model.",
        )
        await pilot.pause()
        assert not list(app.query("#console-setup-step-1"))
        body = app.query_one("#console-empty-body", Static)
        assert CONSOLE_READY_EMPTY_COPY in str(getattr(body.renderable, "plain", body.renderable))
```

(Reuse the file's existing `App`/`Static`/`pytest` imports; add only what's missing.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `env HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share .venv/bin/python -m pytest -q Tests/UI/test_console_rail_sections.py -k setup_panel --tb=short`

Expected: FAIL — `ConsoleTranscriptEmptyPanel.__init__` does not accept a `ConsoleSetupCardState`.

- [ ] **Step 3: Rework the panel**

In `console_transcript.py`:

1. Import the Task 1 contracts:
```python
from tldw_chatbook.Chat.console_onboarding_state import (
    CONSOLE_SETUP_CARD_TITLE,
    ConsoleSetupCardState,
)
```
2. Change `ConsoleTranscriptEmptyPanel.__init__` to take `card_state: ConsoleSetupCardState` as the first positional argument (replacing the old `copy: str`), keeping the `provider_action_label`/`provider_action_tooltip` keywords. Store `self.card_state = card_state`.
3. Rework `compose()` to render all parts and gate visibility by mode (so `sync_card_state` can flip modes without remounting the panel container):
   - Title `Static(CONSOLE_SETUP_CARD_TITLE, id="console-empty-title", classes="console-transcript-empty-title")` — `display: none` unless `mode == "card"`.
   - Three step rows: for `index, step in enumerate(self.card_state.steps, start=1)` yield `Static(self._step_text(index, step), id=f"console-setup-step-{index}", classes=f"console-setup-step console-setup-step-{step.state}", markup=False)`; the whole step block is composed only when `mode == "card"` AND steps are non-empty — in other modes compose no step widgets at all (the tests assert absence). To support mode transitions, `sync_card_state` recomposes the panel (`self.refresh(recompose=True)`) rather than patching widgets — the panel is small, this matches `ConsoleWorkspaceContextTray.sync_state`'s established recompose pattern.
   - Body `Static(self.card_state.body_copy, id="console-empty-body", ...)` — `display: none` when `mode == "card"`.
   - Keep the existing action row and three buttons exactly as today (ids, labels, tooltips, adaptive primary action) but set the row's `display: none` unless `mode == "card"`.
4. Add:
```python
    @staticmethod
    def _step_text(index: int, step: ConsoleSetupStep) -> str:
        text = f"{index}. {step.glyph} {step.label}"
        if step.detail:
            text = f"{text}  {step.detail}"
        return text

    def sync_card_state(
        self,
        card_state: ConsoleSetupCardState,
        *,
        provider_action_label: str,
        provider_action_tooltip: str,
    ) -> None:
        """Refresh the onboarding surface from a new card state."""
        self.card_state = card_state
        self.provider_action_label = provider_action_label
        self.provider_action_tooltip = provider_action_tooltip
        self.refresh(recompose=True)
```
5. Update the old `sync_empty_state` on the panel: delete it (its callers move to `sync_card_state`). In `ConsoleTranscript.sync_empty_state` (line 333) and `_build_row_widget`/`_update_row_widget` (lines 604–631), thread a `card_state: ConsoleSetupCardState` through instead of the bare `copy` string: `ConsoleTranscript` stores `self._empty_card_state` (default: `ConsoleSetupCardState(mode="quiet", body_copy=CONSOLE_QUIET_EMPTY_COPY)` — import `CONSOLE_QUIET_EMPTY_COPY`), the empty row builds `ConsoleTranscriptEmptyPanel(self._empty_card_state, ...)`, and updates call `panel.sync_card_state(...)`. Keep the transcript's public `sync_empty_state` name but change its signature to `sync_empty_state(card_state, *, provider_action_label, provider_action_tooltip)`; fix its callers in this file. (Task 4 fixes the chat_screen caller.)

- [ ] **Step 4: Run tests to verify they pass, check fallout in-file**

```bash
env HOME=... .venv/bin/python -m pytest -q Tests/UI/test_console_rail_sections.py -k setup_panel --tb=short
env HOME=... .venv/bin/python -m pytest -q Tests/UI/test_console_workbench_contract.py --tb=short
```

Expected: new tests PASS. `test_console_workbench_contract.py` and other screen-level suites may fail until Task 4 rewires chat_screen (the screen still calls the old signature) — run them, record the failures, and fix any that are widget-internal now; screen-level signature fallout is completed in Task 4 (note it in the commit message).

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Widgets/Console/console_transcript.py Tests/UI/test_console_rail_sections.py
git commit -m "feat(console): empty panel renders live setup card state"
```

---

### Task 4: Screen wiring — card state, first-send flag, lifecycle

**Files:**
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py` — `_sync_console_transcript_guidance()` (~line 4185), `_submit_console_native_draft()` (~line 5880), compose path that builds the initial empty panel, plus a new flag read/write pair modeled on `_save_console_rail_preferences` (~line 3180)
- Test: `Tests/UI/test_console_native_chat_flow.py`

**Interfaces:**
- Consumes: Task 1 (`build_console_setup_card_state`, `coerce_console_first_send_completed`), Task 3 (`sync_empty_state(card_state, *, provider_action_label, provider_action_tooltip)`), existing `_active_console_settings_readiness()` (chat_screen.py:1363, returns `(settings, readiness)`), `save_setting_to_cli_config` pattern.
- Produces: screen methods `_console_first_send_completed() -> bool` (reads `app_config["console"]["onboarding"]["first_send_completed"]` through `coerce_console_first_send_completed`, with an instance cache `self._console_first_send_completed_cached: bool | None = None`), `_record_console_first_send()` (sets cache + in-memory app_config + background-persists via `@work(thread=True)` `_save_console_onboarding_flag`), and `_build_console_setup_card_state() -> ConsoleSetupCardState`.

- [ ] **Step 1: Write the failing pilot tests**

Append to `Tests/UI/test_console_native_chat_flow.py` (reuse its harness/fixtures; it already has helpers to force blocked/ready provider settings — mirror `test_console_empty_transcript_teaches_setup_and_start_paths` at line 2151 for setup):

```python
@pytest.mark.asyncio
async def test_console_blocked_empty_transcript_shows_setup_card_steps():
    app = _build_test_app()
    host = ConsoleHarness(app)
    async with host.run_test(size=(180, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-transcript-empty-state")
        text = _visible_text(console)
        assert "Get started" in text
        assert "Add an API key" in text
        assert "Pick a model" in text
        assert "Send your first message" in text
        # The legacy banner strip is gone (Task 5 removes compose; here assert not displayed).
        _assert_selector_hidden_or_absent(console, "#console-provider-recovery-strip")


@pytest.mark.asyncio
async def test_console_first_send_flag_switches_empty_state_to_quiet():
    app = _build_test_app()
    app.app_config.setdefault("console", {})["onboarding"] = {"first_send_completed": True}
    host = ConsoleHarness(app)
    async with host.run_test(size=(180, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-transcript-empty-state")
        text = _visible_text(console)
        assert "No messages yet." in text
        assert "Get started" not in text
        assert "Add an API key" not in text


@pytest.mark.asyncio
async def test_console_accepted_send_records_first_send_flag():
    # Reuse the ready-provider send harness used by
    # test_console_send_refreshes_workspace_conversation_rail_after_persistence:
    # same fixtures/gateway stub, then assert the flag.
    ...  # copy that test's arrange/act block verbatim up to the accepted send
    onboarding = app.app_config.get("console", {}).get("onboarding", {})
    assert onboarding.get("first_send_completed") is True
```

For the third test: locate `test_console_send_refreshes_workspace_conversation_rail_after_persistence`, copy its arrange/act body (build app, ready provider, type, send, await acceptance) verbatim, then append the flag assertions. Do not invent a new send harness.

- [ ] **Step 2: Run tests to verify they fail**

Run: `env HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share .venv/bin/python -m pytest -q Tests/UI/test_console_native_chat_flow.py -k "setup_card or first_send" --tb=short`

Expected: FAIL (card copy absent, flag never written).

- [ ] **Step 3: Implement the wiring**

In `chat_screen.py`:

1. Imports: `build_console_setup_card_state`, `coerce_console_first_send_completed`, `ConsoleSetupCardState`, `CONSOLE_QUIET_EMPTY_COPY` from `tldw_chatbook.Chat.console_onboarding_state`.
2. Flag read (near `_current_console_rail_state` helpers):
```python
    def _console_first_send_completed(self) -> bool:
        """Return the persisted global first-send flag (cached per screen)."""
        if self._console_first_send_completed_cached is None:
            app_config = getattr(self.app_instance, "app_config", None)
            raw = None
            if isinstance(app_config, dict):
                onboarding = app_config.get("console", {})
                if isinstance(onboarding, dict):
                    onboarding = onboarding.get("onboarding", {})
                raw = onboarding.get("first_send_completed") if isinstance(onboarding, dict) else None
            self._console_first_send_completed_cached = coerce_console_first_send_completed(raw)
        return self._console_first_send_completed_cached
```
   Initialize `self._console_first_send_completed_cached: bool | None = None` alongside the screen's other instance state (find `_console_guidance_dismissed` initialization and put it next to it).
3. Flag write:
```python
    def _record_console_first_send(self) -> None:
        """Persist the one-time global first-send flag and refresh guidance."""
        if self._console_first_send_completed():
            return
        self._console_first_send_completed_cached = True
        app_config = getattr(self.app_instance, "app_config", None)
        if isinstance(app_config, dict):
            app_config.setdefault("console", {}).setdefault("onboarding", {})[
                "first_send_completed"
            ] = True
        self._save_console_onboarding_flag()
        self._sync_console_transcript_guidance()

    @work(thread=True)
    def _save_console_onboarding_flag(self) -> None:
        """Persist the first-send flag without blocking the UI thread."""
        try:
            save_setting_to_cli_config("console.onboarding", "first_send_completed", True)
        except Exception as exc:
            logger.warning("Failed to persist Console onboarding flag: {}", exc)
```
4. In `_submit_console_native_draft` (~5880), after `result = await controller.submit_draft(draft)`: `if result.accepted: self._record_console_first_send()` (before the existing `_sync_native_console_chat_ui()` call).
5. Card-state builder:
```python
    def _build_console_setup_card_state(self) -> ConsoleSetupCardState:
        """Build the empty-transcript onboarding state from current readiness."""
        settings, readiness = self._active_console_settings_readiness()
        has_model = bool(
            _string_value(getattr(settings, "model", None))
            or _string_value(getattr(settings, "configured_model", None))
        )
        return build_console_setup_card_state(
            readiness=readiness,
            provider_label=str(getattr(settings, "provider", "") or "Provider"),
            has_model=has_model,
            first_send_completed=self._console_first_send_completed(),
            has_messages=self._active_console_transcript_has_messages(),
            guidance_dismissed=self._console_guidance_dismissed,
        )
```
   For `_active_console_transcript_has_messages()`: implement from the store — active session's `messages_for_session` non-empty (guard store None / no active session → False). If `_string_value` isn't importable here, inline `str(x or "").strip()`.
6. In `_sync_console_transcript_guidance()` (~4185): replace the copy-string plumbing that feeds `transcript.sync_empty_state(...)` (line ~4228) with the card state: `transcript.sync_empty_state(self._build_console_setup_card_state(), provider_action_label=..., provider_action_tooltip=...)` keeping the existing adaptive action label/tooltip source (`_console_empty_recovery_action_copy`). Update the compose-time construction of the initial empty panel the same way. The old `_console_empty_transcript_copy`/`_console_blocked_empty_transcript_copy` static adapters become unused once nothing calls them — delete them and their direct unit tests if any (grep first; if other callers remain, leave them and note it).
7. `_dismiss_console_guidance` stays as-is (it feeds `guidance_dismissed`; the pure builder now decides that dismissal only hides the ready line, never the setup card).

- [ ] **Step 4: Run the new tests, then the affected suites**

```bash
env HOME=... .venv/bin/python -m pytest -q Tests/UI/test_console_native_chat_flow.py -k "setup_card or first_send" --tb=short
env HOME=... .venv/bin/python -m pytest -q Tests/UI/test_console_native_chat_flow.py Tests/UI/test_console_internals_decomposition.py Tests/UI/test_console_workbench_contract.py --tb=short
```

Expected: new tests PASS. Update pre-existing expectations that assert the old copy (`Start Console`, `Type in Composer, attach sources...`, `Add an API key to enable Send`) to the new card/ready-line copy — each updated test must keep asserting its original concern (guidance exists, adapts to readiness, buttons route correctly). Tests asserting the ready-state action buttons are visible change to assert the ready line with no action row (spec-mandated, note in commit message).

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/UI/Screens/chat_screen.py Tests/UI/test_console_native_chat_flow.py Tests/UI/test_console_internals_decomposition.py Tests/UI/test_console_workbench_contract.py
git commit -m "feat(console): wire live setup card with persisted first-send lifecycle"
```

---

### Task 5: Remove the provider blocker banner

**Files:**
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py` — compose block for `#console-provider-recovery-strip` (~lines 4786–4828), `_configure_console_provider_recovery_strip` (~4262), `_configure_console_provider_settings_action` (~4288), `_console_provider_recovery_strip_visible` (~4100), `_console_provider_blocker_display_copy` (~4105), and the strip-refresh lines inside `_sync_console_transcript_guidance` (~4242–4253)
- Test: `Tests/UI/test_console_internals_decomposition.py` (tests at lines ~1763, ~1859, ~1905, ~1923, ~1943)

**Interfaces:**
- Consumes: Task 4's card (the strip's job is now done by the card + composer reason).
- Produces: no `#console-provider-recovery-strip`, `#console-provider-blocker`, or `#console-open-provider-settings` widgets anywhere. **KEEP** `handle_console_open_provider_settings` / `_open_console_provider_recovery` (~5999–6040), `_console_provider_recovery_action`, `_console_provider_recovery_field`, and `_console_provider_blocker_copy` — the card's primary action button (`console-empty-choose-model`) and the composer reason still use them. Only the strip UI and its two `_configure_*` helpers plus `_console_provider_recovery_strip_visible`/`_console_provider_blocker_display_copy` go.

- [ ] **Step 1: Update the strip tests (failing first)**

Rework the five tests: assertions that the strip/blocker/settings-button exist (even hidden) become assertions that the selectors are ABSENT (`not list(console.query("#console-provider-recovery-strip"))` etc.), and re-anchor each test's real concern:
- `test_console_empty_transcript_promotes_start_here_and_provider_recovery` → renamed `..._promotes_setup_card_over_banner`: card visible with steps, strip absent.
- `test_console_provider_blocker_exposes_open_settings_action` → the card's primary button (`#console-empty-choose-model`) carries the recovery label/tooltip in the blocked state.
- `test_console_provider_settings_action_posts_navigation_message` → same navigation assertion, driven by pressing `#console-empty-choose-model` in the blocked state.
- `test_console_provider_settings_action_hidden_when_provider_ready` → in ready_line mode the action row is hidden entirely.
- `test_console_choose_model_state_hides_redundant_recovery_strip` → strip absent unconditionally; keep its choose-model copy assertion against the card's step 2 / primary button.

- [ ] **Step 2: Run to verify failures**

Run: `env HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share .venv/bin/python -m pytest -q Tests/UI/test_console_internals_decomposition.py -k "provider or promotes" --tb=short`

Expected: FAIL — strip still composed.

- [ ] **Step 3: Remove the strip**

Delete the compose block yielding the strip, blocker Static, and settings Button (~4786–4828); delete `_configure_console_provider_recovery_strip`, `_configure_console_provider_settings_action`, `_console_provider_recovery_strip_visible`, `_console_provider_blocker_display_copy`; delete the strip-refresh section of `_sync_console_transcript_guidance` (the `query_one("#console-provider-recovery-strip")`/settings-button update lines ~4242–4253). Verify the card's primary button handler path still routes: pressing `#console-empty-choose-model` in a blocked state must reach `_open_console_provider_recovery()` — check the existing button dispatch for `console-empty-choose-model` and repoint it from `_open_console_settings(...)` to `_open_console_provider_recovery()` when `_console_provider_recovery_action()` targets a recovery field (mirror the deleted button's routing). Grep for any remaining references to the deleted names (including CSS selectors in `_agentic_terminal.tcss` — leave CSS rules in place for now; Task 6 prunes them).

- [ ] **Step 4: Run the reworked tests + adjacent suites**

```bash
env HOME=... .venv/bin/python -m pytest -q Tests/UI/test_console_internals_decomposition.py --tb=short
env HOME=... .venv/bin/python -m pytest -q Tests/UI/test_console_native_chat_flow.py -k "blocked or recovery" --tb=short
```

Expected: PASS (including the two blocked-send recovery-feedback tests at test_console_native_chat_flow.py:1005/1038, which assert transcript/system feedback, not the strip).

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/UI/Screens/chat_screen.py Tests/UI/test_console_internals_decomposition.py
git commit -m "feat(console): retire provider blocker banner in favor of setup card"
```

---

### Task 6: Setup-card CSS

**Files:**
- Modify: `tldw_chatbook/css/components/_agentic_terminal.tcss`
- Regenerate: `tldw_chatbook/css/tldw_cli_modular.tcss` via `./build_css.sh`
- Test: `Tests/UI/test_console_persistent_rails.py` (mirror `test_generated_console_stylesheet_includes_rail_section_rules`)

**Interfaces:**
- Consumes: classes from Task 3: `.console-setup-step`, `.console-setup-step-done`, `.console-setup-step-active`, `.console-setup-step-pending`.
- Produces: styled selectors in both files; stale `.console-provider-recovery-strip`/`.console-provider-blocker`/`.console-provider-settings-action` rules pruned.

- [ ] **Step 1: Failing stylesheet test**

Append to `Tests/UI/test_console_persistent_rails.py`:

```python
def test_generated_console_stylesheet_includes_setup_card_rules():
    root = Path(__file__).resolve().parents[2] / "tldw_chatbook" / "css"
    component_css = (root / "components" / "_agentic_terminal.tcss").read_text()
    generated_css = (root / "tldw_cli_modular.tcss").read_text()
    for selector in (
        ".console-setup-step",
        ".console-setup-step-done",
        ".console-setup-step-active",
        ".console-setup-step-pending",
    ):
        assert selector in component_css, selector
        assert selector in generated_css, selector
    for stale in (".console-provider-recovery-strip", ".console-provider-blocker"):
        assert stale not in component_css, stale
        assert stale not in generated_css, stale
```

- [ ] **Step 2: Run to verify failure**

Run: `env HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share .venv/bin/python -m pytest -q Tests/UI/test_console_persistent_rails.py -k setup_card_rules --tb=short`

Expected: FAIL — selectors absent / stale rules present.

- [ ] **Step 3: Add CSS, prune stale rules, rebuild**

In `_agentic_terminal.tcss`, next to the `.console-transcript-empty-*` rules (grep for them), add:

```css
.console-setup-step {
    height: 1;
    color: $ds-text-primary;
}

.console-setup-step-done {
    color: $ds-text-muted;
}

.console-setup-step-active {
    text-style: bold;
}

.console-setup-step-pending {
    color: $ds-text-muted;
}
```

(If the sibling empty-panel rules use different text tokens, match those tokens instead.) Delete the `.console-provider-recovery-strip`, `.console-provider-blocker`, and `.console-provider-settings-action` rule blocks (grep for each; also remove `console-provider-api-key-action` if present). Run `./build_css.sh`.

- [ ] **Step 4: Run to verify pass + visual suites**

```bash
env HOME=... .venv/bin/python -m pytest -q Tests/UI/test_console_persistent_rails.py --tb=short
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/css/components/_agentic_terminal.tcss tldw_chatbook/css/tldw_cli_modular.tcss Tests/UI/test_console_persistent_rails.py
git commit -m "style(console): setup card step styles, drop banner rules"
```

---

### Task 7: Verification, screenshot QA, approval gate

**Files:**
- Create: screenshots under `Docs/superpowers/qa/console-setup-card-2026-07/`

- [ ] **Step 1: Full affected test run**

```bash
env HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share .venv/bin/python -m pytest -q \
  Tests/Chat/test_console_onboarding_state.py Tests/Chat/test_console_display_state.py \
  Tests/UI/test_console_rail_sections.py Tests/UI/test_console_internals_decomposition.py \
  Tests/UI/test_console_native_chat_flow.py Tests/UI/test_console_workbench_contract.py \
  Tests/UI/test_console_persistent_rails.py --tb=short
```

Expected: ALL PASS (no baseline failures exist on dev anymore).

- [ ] **Step 2: Live screenshot QA**

Use the proven capture recipe (see memory + `Docs/superpowers/qa/console-rail-ia-2026-07/README.md` session notes): textual-serve with a FRESH isolated HOME, playwright **bundled chromium** (not channel="chrome"), `goto(..., wait_until="commit")`, wait for `.intro-dialog` hidden, route-abort all non-localhost requests, kill stale `tldw_chatbook` python processes first. Capture:
1. Fresh install, no key: full `Get started` card — steps `1. ● Add an API key / 2. ✓|● Pick a model / 3. ○ Send your first message`, no banner strip above the transcript, composer reason `Send blocked — add an API key to continue`.
2. Ready provider (local llama.cpp if available, else configure a key): one-line `Ready — type a message to begin.` with no action buttons.
3. After a real accepted send, open a NEW tab: empty transcript shows only `No messages yet.`
4. Relaunch (fresh app instance, same HOME): still `No messages yet.` (flag persisted; also show `console.onboarding` in the HOME's config.toml).

- [ ] **Step 3: User approval gate**

Present the screenshots for explicit approval before any merge. Do not merge without it.

- [ ] **Step 4: Commit QA artifacts**

```bash
git add Docs/superpowers/qa/console-setup-card-2026-07/
git commit -m "docs(console): setup card phase 2 QA evidence"
```
