# First-Run Reliability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make first-run setup keyboard-safe, resumable, failure-tolerant, network-scoped, and visually clear before adding the Voice step.

**Architecture:** Keep pure setup decisions in `first_run_setup_state.py`, screen orchestration in `FirstRunSetupWizard.py`, and startup recovery in a small dedicated modal. Resume data is a versioned non-secret draft that never enters active chat or TTS configuration. Provider selection uses one `OptionList` whose disabled group rows cannot be selected.

**Tech Stack:** Python 3.11+, Textual, pytest, pytest-asyncio, Rich, TOML-backed application settings.

## Global Constraints

- Baseline is `origin/dev` at `5414d811b8720c1c32c5813f96925a82c60c5f72`.
- Required setup steps never auto-skip; only optional steps may skip with an explicit summary entry.
- Resume persistence contains no API keys or other secrets.
- Drafts are checkpointed only after a successful step commit, never on each keystroke.
- Draft parsing is bounded to known steps/fields, JSON scalar values, 64 fields, and 16 KiB total serialized data.
- Setup drafts are not active application configuration.
- First-run discovery contacts only the selected provider.
- Verify dark and light themes at 120x40, 177x45, and the narrow supported size.
- Keep all controls and text within their containers.
- Logs and notifications contain no endpoint query, userinfo, credential, request text, or card text.

---

## File Structure

- Modify `tldw_chatbook/UI/Wizards/first_run_setup_state.py`: pure progress, draft, recovery, and failure-policy contracts.
- Modify `tldw_chatbook/UI/Wizards/FirstRunSetupWizard.py`: provider list, navigation, persistence calls, and recovery actions.
- Create `tldw_chatbook/UI/Wizards/first_run_recovery_dialog.py`: startup Resume/Start over/Later modal only.
- Modify `tldw_chatbook/app.py`: choose initial offer, recovery prompt, or Home fallback.
- Modify `tldw_chatbook/css/features/_wizards.tcss`: focus, progress, contrast, and responsive dimensions.
- Modify the focused tests listed in each task; do not create a second first-run framework.

### Task 1: Replace the crash-prone provider RadioSet

**Files:**
- Modify: `tldw_chatbook/UI/Wizards/FirstRunSetupWizard.py:24,255-735`
- Modify: `tests/Wizards/test_first_run_setup_wizard.py:320-835,2064-2088,2742-2848`

**Interfaces:**
- Produces: `ProviderChoiceOption(Option)` carrying `provider_key: str | None` and `Option.disabled=True` for headings.
- Produces: `ProviderStep.selected_provider_key` updated only by a selectable provider option.
- Preserves: `ProviderStep.commit() -> tuple[bool, str]` and existing provider discovery/probe injection.

- [ ] **Step 1: Write the failing keyboard and structure tests**

```python
async def test_provider_down_and_space_never_selects_group_heading():
    step = _provider_step()
    app = _StepHost(step)
    async with app.run_test(size=(120, 40)) as pilot:
        choices = step.query_one("#setup-provider-choice", OptionList)
        assert all(
            option.disabled
            for index in range(choices.option_count)
            if (option := choices.get_option_at_index(index)).id.startswith("group-")
        )
        await pilot.press("down", "space", "down", "space")
        assert step.selected_provider_key
        assert not step.selected_provider_key.startswith("group-")


async def test_provider_keyboard_walk_visits_only_provider_rows():
    step = _provider_step()
    app = _StepHost(step)
    async with app.run_test(size=(120, 40)) as pilot:
        choices = step.query_one("#setup-provider-choice", OptionList)
        for _ in range(choices.option_count + 3):
            await pilot.press("down")
            highlighted = choices.get_option_at_index(choices.highlighted)
            assert not highlighted.disabled
```

- [ ] **Step 2: Run the focused tests and confirm the current RadioSet fails**

Run: `.venv/bin/python -m pytest tests/Wizards/test_first_run_setup_wizard.py -k "provider_down_and_space or provider_keyboard_walk" -v`

Expected: FAIL because `#setup-provider-choice` is a `RadioSet` containing `Static` children, or because `OptionList` is absent.

- [ ] **Step 3: Implement one grouped OptionList**

```python
class ProviderChoiceOption(Option):
    def __init__(self, prompt: Text, *, option_id: str, provider_key: str | None):
        super().__init__(prompt, id=option_id, disabled=provider_key is None)
        self.provider_key = provider_key


def _provider_group_option_id(title: str) -> str:
    return "group-" + "-".join(title.casefold().split())


def _provider_options(entries: Sequence[ConsoleProviderCatalogEntry]) -> list[Option]:
    options: list[Option] = []
    for group_title, group in ProviderStep._grouped_sections(entries):
        options.append(
            ProviderChoiceOption(
                Text(group_title, style="bold"),
                option_id=_provider_group_option_id(group_title),
                provider_key=None,
            )
        )
        options.extend(
            ProviderChoiceOption(
                Text(entry.display_name),
                option_id=f"provider-{entry.readiness_key}",
                provider_key=entry.readiness_key,
            )
            for entry in group
        )
    return options
```

Update composition to yield `OptionList(*_provider_options(entries), id="setup-provider-choice")`. Handle `OptionList.OptionHighlighted` and `OptionList.OptionSelected`; ignore any option whose `provider_key is None`. Replace `RadioSet` queries in `preferred_focus()`, commit fallback, re-entry, and tests.

- [ ] **Step 4: Run the complete provider-step regression group**

Run: `.venv/bin/python -m pytest tests/Wizards/test_first_run_setup_wizard.py -k "provider or down_space" -v`

Expected: PASS, including real Down/Space traversal and existing key-input focus order.

- [ ] **Step 5: Commit the provider control correction**

```bash
git add tldw_chatbook/UI/Wizards/FirstRunSetupWizard.py tests/Wizards/test_first_run_setup_wizard.py
git commit -m "fix: make first-run provider selection keyboard safe"
```

### Task 2: Add isolated setup drafts and crash-loop-safe recovery

**Files:**
- Modify: `tldw_chatbook/UI/Wizards/first_run_setup_state.py:15-135,246-355`
- Modify: `tldw_chatbook/UI/Wizards/FirstRunSetupWizard.py:3927-4190`
- Create: `tldw_chatbook/UI/Wizards/first_run_recovery_dialog.py`
- Modify: `tldw_chatbook/app.py:9020-9060`
- Modify: `tests/Wizards/test_first_run_setup_state.py:66-123`
- Modify: `tests/UI/test_first_run_wizard_live_contract.py:137-255`

**Interfaces:**
- Produces: `SetupDraft(version: int, track: str, active_step_id: str, values: Mapping[str, Mapping[str, object]], resume_attempted: bool)`.
- Produces: `read_setup_draft(app_config) -> SetupDraft | None` that rejects unknown versions and secret-shaped keys.
- Produces: `build_setup_draft_mutation(draft: SetupDraft | None) -> tuple[dict[str, dict[str, object]], dict[str, tuple[str, ...]]]`; `None` returns exact `first_run` delete keys for `save_settings_to_cli_config(..., delete_keys=...)`.
- Produces: `setup_recovery_action(app_config, environ) -> Literal["offer", "prompt", "home", "none"]`.
- Consumes: the existing `save_settings_to_cli_config` merge path; the draft is never passed to chat/TTS resolvers.

- [ ] **Step 1: Write failing pure-state tests**

```python
def test_resume_draft_rejects_secret_keys():
    config = {"first_run": {"draft_version": 1, "draft_values": {"provider": {"api_key": "secret"}}}}
    assert read_setup_draft(config) is None


def test_uncleared_resume_attempt_routes_to_home():
    config = {
        "first_run": {
            "setup_started": True,
            "setup_completed": False,
            "draft_version": 1,
            "active_step_id": "model",
            "resume_attempted": True,
        }
    }
    assert setup_recovery_action(config, {}) == "home"


def test_setup_draft_is_not_owned_active_configuration():
    settings, delete_keys = build_setup_draft_mutation(
        SetupDraft(1, "quick", "provider", {"provider": {"provider_value": "openai"}}, False)
    )
    assert set(settings) == {"first_run"}
    assert delete_keys == {}


def test_setup_draft_rejects_oversized_or_unknown_fields():
    oversized = {"provider_value": "x" * 20_000}
    config = {"first_run": {"draft_version": 1, "draft_values": {"provider": oversized}}}
    assert read_setup_draft(config) is None
```

- [ ] **Step 2: Run the state tests and verify failure**

Run: `.venv/bin/python -m pytest tests/Wizards/test_first_run_setup_state.py -k "resume_draft or recovery_action or isolated" -v`

Expected: FAIL because the versioned draft and recovery decision do not exist.

- [ ] **Step 3: Implement bounded draft serialization and the recovery modal**

```python
SETUP_DRAFT_VERSION = 1
_SECRET_FIELD_TOKENS = frozenset({"api_key", "credential", "password", "token", "secret"})

@dataclass(frozen=True, slots=True)
class SetupDraft:
    version: int
    track: str
    active_step_id: str
    values: Mapping[str, Mapping[str, object]]
    resume_attempted: bool = False


def setup_recovery_action(app_config, environ):
    if should_offer_wizard(app_config, environ):
        return "offer"
    draft = read_setup_draft(app_config)
    if draft is None:
        return "none"
    return "home" if draft.resume_attempted else "prompt"
```

`SetupRecoveryDialog` returns exactly `"resume"`, `"start_over"`, or `"later"` and states that credentials are not retained and may need to be entered again. Resume writes `resume_attempted=True` before pushing the wizard; `FirstRunSetupWizard` clears it with `call_after_refresh` after the target step mounts. After each successful non-summary step commit, save one allowlisted, bounded, non-secret checkpoint before navigation; do not persist partial field edits. Successful setup completion deletes all draft keys in the same settings mutation that marks setup complete. Start over passes the exact draft-key tuple from `build_setup_draft_mutation(None)` to `save_settings_to_cli_config({}, delete_keys=...)`; it does not rewrite the whole `first_run` section. Later dismisses without mutation. Update app startup to use `setup_recovery_action` and never auto-push an uncleared attempt.

- [ ] **Step 4: Run state and live recovery tests**

Run: `.venv/bin/python -m pytest tests/Wizards/test_first_run_setup_state.py tests/UI/test_first_run_wizard_live_contract.py -k "resume or recovery or start_over or finish_later" -v`

Expected: PASS; a failed resume reaches Home on the next launch and active configuration remains unchanged.

- [ ] **Step 5: Commit recovery support**

```bash
git add tldw_chatbook/UI/Wizards/first_run_setup_state.py tldw_chatbook/UI/Wizards/FirstRunSetupWizard.py tldw_chatbook/UI/Wizards/first_run_recovery_dialog.py tldw_chatbook/app.py tests/Wizards/test_first_run_setup_state.py tests/UI/test_first_run_wizard_live_contract.py
git commit -m "feat: add resumable first-run setup recovery"
```

### Task 3: Stop auto-skipping required-step failures

**Files:**
- Modify: `tldw_chatbook/UI/Wizards/FirstRunSetupWizard.py:130-235,3650-4050`
- Modify: `tests/Wizards/test_first_run_setup_wizard.py:2857-3085`
- Modify: `tests/UI/test_first_run_wizard_live_contract.py:1045-1200`

**Interfaces:**
- Produces: `SetupStep.required: bool` from `WizardStepConfig`.
- Produces: `SetupStepFailure(step_id: str, required: bool, reason_code: str)` with bounded reason codes.
- Produces UI actions `setup-step-retry`, `setup-step-manual`, and `setup-step-later` for required failures.
- Preserves optional failure summary rows and existing wizard navigation worker fencing.

- [ ] **Step 1: Write failing required/optional failure tests**

```python
async def test_required_provider_compose_failure_never_advances():
    original = ProviderStep.compose_step
    ProviderStep.compose_step = _raising_compose_step
    try:
        wizard = _make_wizard()
        app = _HostApp(wizard)
        async with app.run_test(size=(120, 40)) as pilot:
            container = wizard.query_one(SetupWizardContainer)
            container.select_track(TRACK_QUICK)
            provider_index = next(
                index for index, step in enumerate(container.steps)
                if step.config and step.config.id == STEP_PROVIDER
            )
            container.show_step(provider_index)
            await pilot.pause()
            assert container.steps[container.current_step].config.id == STEP_PROVIDER
            assert wizard.query_one("#setup-step-retry", Button).display
            assert wizard.query_one("#setup-step-manual", Button).display
            assert wizard.query_one("#setup-step-later", Button).display
    finally:
        ProviderStep.compose_step = original


async def test_optional_step_failure_records_skip_and_advances():
    original = RagStep.compose_step
    RagStep.compose_step = _raising_compose_step
    try:
        wizard = _make_wizard()
        app = _HostApp(wizard)
        async with app.run_test(size=(120, 40)):
            container = wizard.query_one(SetupWizardContainer)
            container.select_track(TRACK_FULL)
            assert STEP_RAG not in container.active_ids
            assert STEP_RAG in container.skipped_step_reasons
    finally:
        RagStep.compose_step = original
```

Define `_raising_compose_step` in the test module as a generator-shaped helper
that raises `RuntimeError("boom")`, matching the existing
`TestComposeCrashPolicy` fixture pattern. Change that class's current
auto-skip expectations to the required/optional split above.

- [ ] **Step 2: Verify current generic compose fallback fails the contract**

Run: `.venv/bin/python -m pytest tests/Wizards/test_first_run_setup_wizard.py -k "required_provider_compose_failure or optional_step_failure" -v`

Expected: FAIL because the current wrapper marks every failed step for automatic skipping.

- [ ] **Step 3: Implement required-step recovery without re-running partial composition**

```python
def _failure_widgets(self, failure: SetupStepFailure) -> list[Widget]:
    if not failure.required:
        return [Static("This optional step could not be shown.", classes="setup-step-error")]
    return [
        Static("This required step could not be shown.", classes="setup-step-error"),
        Horizontal(
            Button("Retry", id="setup-step-retry", variant="primary"),
            Button("Use manual setup", id="setup-step-manual"),
            Button("Finish later", id="setup-step-later"),
            classes="setup-step-recovery-actions",
        ),
    ]
```

For Retry, reconstruct only the failed step and replace its mounted container. Manual setup exits to the matching Settings destination without marking setup complete. Finish later persists the non-secret draft and dismisses. Keep raw exceptions in logs by category only; summary uses bounded reason copy.

- [ ] **Step 4: Run compose, navigation, and live contract tests**

Run: `.venv/bin/python -m pytest tests/Wizards/test_first_run_setup_wizard.py tests/UI/test_first_run_wizard_live_contract.py -k "compose or failure or navigation or mashing" -v`

Expected: PASS; required failures cannot disappear through Next or automatic advancement.

- [ ] **Step 5: Commit required-step recovery**

```bash
git add tldw_chatbook/UI/Wizards/FirstRunSetupWizard.py tests/Wizards/test_first_run_setup_wizard.py tests/UI/test_first_run_wizard_live_contract.py
git commit -m "fix: keep required setup failures recoverable"
```

### Task 4: Scope setup discovery to the selected provider

**Files:**
- Modify: `tldw_chatbook/UI/Wizards/FirstRunSetupWizard.py:255-1045`
- Modify: `tests/Wizards/test_first_run_setup_wizard.py:806-1220,2742-2848`
- Modify: `tests/UI/test_product_maturity_phase1_first_run.py:179-325`

**Interfaces:**
- Produces: `ProviderStep._begin_selected_provider_discovery(provider_key: str, generation: int) -> None`.
- Produces: one cancellation generation shared by provider probe and model discovery.
- Consumes: existing injected `discover` and model scope service, always with the selected provider key.

- [ ] **Step 1: Write a failing network-call ledger test**

```python
async def test_first_run_contacts_only_selected_provider():
    selected_discovery = AsyncMock(return_value=())
    step = _provider_step(discover=selected_discovery)
    app = _StepHost(step)
    async with app.run_test(size=(120, 40)) as pilot:
        step.select_provider("ollama")
        await pilot.pause()
    assert selected_discovery.await_count >= 1
    assert {call.args[0] for call in selected_discovery.await_args_list} == {"ollama"}
    assert app.notifications == []
```

- [ ] **Step 2: Run and confirm unrelated refreshes are observable**

Run: `.venv/bin/python -m pytest tests/Wizards/test_first_run_setup_wizard.py tests/UI/test_product_maturity_phase1_first_run.py -k "contacts_only_selected or discovery" -v`

Expected: FAIL until discovery is keyed and stale workers are fenced.

- [ ] **Step 3: Implement selected-provider discovery and quiet background behavior**

```python
def _begin_selected_provider_discovery(self, provider_key: str) -> None:
    self.probe_generation += 1
    generation = self.probe_generation
    self.run_worker(
        self._discover_selected_provider(provider_key, generation),
        group="setup-provider-discovery",
        exclusive=True,
    )
```

Do not enumerate every provider from first-run code. Discard results unless both generation and selected provider still match. Render progress and failure inline in the selected provider step. Suppress global catalog success/failure toasts while setup owns the request; preserve explicit Test feedback.

- [ ] **Step 4: Run provider/model discovery and first-run launch tests**

Run: `.venv/bin/python -m pytest tests/Wizards/test_first_run_setup_wizard.py tests/UI/test_product_maturity_phase1_first_run.py -k "provider or model or discovery or clean_first_run" -v`

Expected: PASS with no unsolicited first-run notification.

- [ ] **Step 5: Commit network scoping**

```bash
git add tldw_chatbook/UI/Wizards/FirstRunSetupWizard.py tests/Wizards/test_first_run_setup_wizard.py tests/UI/test_product_maturity_phase1_first_run.py
git commit -m "fix: scope first-run discovery to the chosen provider"
```

### Task 5: Correct progress, contrast, and responsive layout

**Files:**
- Modify: `tldw_chatbook/UI/Wizards/first_run_setup_state.py:230-275`
- Modify: `tldw_chatbook/UI/Wizards/FirstRunSetupWizard.py:3107-3950`
- Modify: `tldw_chatbook/css/features/_wizards.tcss`
- Modify: `tests/Wizards/test_first_run_setup_state.py:124-146`
- Modify: `tests/Wizards/test_first_run_setup_wizard.py:2649-2665,3113-3160`
- Modify: `tests/UI/test_product_maturity_phase1_visual_audit.py:236-330`

**Interfaces:**
- Produces: `SetupProgressItem(step_id: str, title: str, state: Literal["active", "complete", "upcoming"])`.
- Produces: `build_setup_progress(active_ids: tuple[str, ...], current_index: int) -> tuple[SetupProgressItem, ...]`.
- Consumes: active track after conditional steps are resolved; no hard-coded total in UI copy.

- [ ] **Step 1: Write failing progress and token tests**

```python
def test_progress_states_derive_from_active_track():
    rows = build_setup_progress(("welcome", "provider", "model", "summary"), 1)
    assert [row.state for row in rows] == ["complete", "active", "upcoming", "upcoming"]


def test_wizard_css_defines_distinct_focus_and_active_tokens():
    css = Path("tldw_chatbook/css/features/_wizards.tcss").read_text()
    assert ".setup-progress-item.-active" in css
    assert ".setup-choice-list:focus" in css
    assert "min-height" in css
```

- [ ] **Step 2: Run state and visual tests before styling**

Run: `.venv/bin/python -m pytest tests/Wizards/test_first_run_setup_state.py tests/Wizards/test_first_run_setup_wizard.py tests/UI/test_product_maturity_phase1_visual_audit.py -k "progress or visual or terminal_size" -v`

Expected: FAIL for derived state/token assertions or snapshots.

- [ ] **Step 3: Implement derived progress and responsive style states**

```python
@dataclass(frozen=True, slots=True)
class SetupProgressItem:
    step_id: str
    title: str
    state: Literal["active", "complete", "upcoming"]


def build_setup_progress(active_ids, current_index):
    return tuple(
        SetupProgressItem(
            step_id=step_id,
            title=STEP_TITLES[step_id],
            state="complete" if index < current_index else "active" if index == current_index else "upcoming",
        )
        for index, step_id in enumerate(active_ids)
    )
```

Render state classes from this projection. In `_wizards.tcss`, strengthen focused and selected borders, use existing semantic foreground/background variables for secondary text, constrain the provider list and footer with stable heights, and reduce empty vertical padding. Do not add viewport-scaled font sizes or decorative cards.

- [ ] **Step 4: Run the full first-run test set**

Run: `.venv/bin/python -m pytest tests/Wizards tests/UI/test_first_run_wizard_live_contract.py tests/UI/test_product_maturity_phase1_first_run.py tests/UI/test_product_maturity_phase1_visual_audit.py -v`

Expected: PASS at all supported dimensions in dark and light themes.

- [ ] **Step 5: Commit first-run visual resilience**

```bash
git add tldw_chatbook/UI/Wizards/first_run_setup_state.py tldw_chatbook/UI/Wizards/FirstRunSetupWizard.py tldw_chatbook/css/features/_wizards.tcss tests/Wizards tests/UI/test_first_run_wizard_live_contract.py tests/UI/test_product_maturity_phase1_first_run.py tests/UI/test_product_maturity_phase1_visual_audit.py
git commit -m "fix: clarify first-run progress and visual states"
```

## Plan Verification

Run: `.venv/bin/python -m pytest tests/Wizards tests/UI/test_first_run_wizard_live_contract.py tests/UI/test_product_maturity_phase1_first_run.py tests/UI/test_product_maturity_phase1_visual_audit.py -v`

Manual checkpoint: start with a clean profile, use only the keyboard through Provider and Model, interrupt and resume once, simulate a failed resumed step, and confirm the Home fallback remains usable without partial settings becoming active.
