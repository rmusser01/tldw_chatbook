# Console Rail Label Setting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Keep horizontal collapsed Console rail labels as the default while allowing users to opt into the existing compact stacked presentation from Settings.

**Architecture:** Normalize one boolean in the canonical `[console]` configuration, read it when a fresh `ChatScreen` composes both rail handles, and expose it through the existing staged Console Behavior draft/save pipeline. Reuse the existing `ConsoleRailHandle(vertical=...)` implementation and Settings registries; do not add a Console-local toggle, a new persistence path, or mounted-screen mutation.

**Tech Stack:** Python 3.11+, Textual 8.x, TOML configuration, pytest/Pilot.

## Global Constraints

- Horizontal remains the compatibility default for missing, false, or malformed values.
- Horizontal left is `Context ▸` at width 13; horizontal right is `Inspector` at width 11.
- Stacked handles are width 3, stack upright `Context` / `Inspector` characters, and omit direction glyphs.
- One preference governs both collapsed Console rails; expanded rails and non-Console consumers remain unchanged.
- Settings uses staged category-wide Save/Revert; runtime config changes only after persistence succeeds.
- A successful Save applies when navigation constructs the next Console screen; no app restart or live mounted-Console mutation.
- The visible checkbox label is `Stack collapsed rail labels`; user-facing style names are `Horizontal` and `Stacked`.
- Stable config key: `console.stack_collapsed_rail_labels`; stable field ID: `settings-console-stack-collapsed-rail-labels`.
- ADR required: no. ADR path: N/A. Reason: additive presentation preference using existing config and UI boundaries.

---

### Task 1: Normalize and ship the configuration default

**Files:**
- Modify: `Tests/test_config_console_defaults.py`
- Modify: `tldw_chatbook/config.py`

**Interfaces:**
- Consumes: existing `coerce_bool_setting(value: object, default: bool) -> bool` and `CONFIG_TOML_CONTENT`.
- Produces: normalized `app_config["console"]["stack_collapsed_rail_labels"]: bool`, defaulting to `False`.

- [x] **Step 1: Write failing configuration tests**

Add literal expectations covering the shipped template, absent config, explicit booleans/string booleans, and malformed input:

```python
def test_console_rail_labels_ship_horizontal_by_default():
    assert (
        config_module.DEFAULT_CONFIG_FROM_TOML["console"]
        ["stack_collapsed_rail_labels"]
        is False
    )


@pytest.mark.parametrize(
    ("raw_value", "expected"),
    [("false", False), ("true", True), ('"false"', False), ('"true"', True), ('"sideways"', False)],
)
def test_load_settings_normalizes_console_rail_label_style(
    tmp_path, monkeypatch, raw_value, expected
):
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        f"[console]\nstack_collapsed_rail_labels = {raw_value}\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))

    settings = config_module.load_settings(force_reload=True)

    assert settings["console"]["stack_collapsed_rail_labels"] is expected
```

Extend `test_load_settings_exposes_console_defaults` with a literal `False` assertion for the absent-key path.

- [x] **Step 2: Run the tests and verify RED**

Run:

```bash
.venv/bin/python -m pytest Tests/test_config_console_defaults.py -q
```

Expected: the new key is absent or malformed input is not normalized to `False`.

- [x] **Step 3: Add minimal normalization and template entries**

In `load_settings`, beside the other Console boolean normalization:

```python
final_console_settings_cli["stack_collapsed_rail_labels"] = coerce_bool_setting(
    final_console_settings_cli.get("stack_collapsed_rail_labels", False),
    False,
)
```

In `CONFIG_TOML_CONTENT` under `[console]`:

```toml
stack_collapsed_rail_labels = false  # Use compact stacked labels on collapsed Console rails
```

- [x] **Step 4: Run the tests and verify GREEN**

Run the same command; expected: all tests pass.

- [x] **Step 5: Commit the configuration slice**

```bash
git add Tests/test_config_console_defaults.py tldw_chatbook/config.py
git commit -m "feat(console): configure collapsed rail label style"
```

---

### Task 2: Compose both Console rail modes from the runtime preference

**Files:**
- Modify: `Tests/UI/test_console_rail_handle.py`
- Modify: `Tests/UI/test_console_shell_regions.py` or create a focused preference harness in `Tests/UI/test_console_rail_handle.py`
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py`

**Interfaces:**
- Consumes: normalized `app_instance.app_config["console"]["stack_collapsed_rail_labels"]` and `ConsoleRailHandle(vertical: bool)`.
- Produces: `ChatScreen._stack_collapsed_rail_labels() -> bool` and preference-driven handle widths for both rails.

- [x] **Step 1: Add failing horizontal mounted-geometry tests**

Add a horizontal harness and assert user-visible output, geometry, and readable tooltips with hand-derived values:

```python
assert left._display_label() == "Context ▸"
assert right._display_label() == "Inspector"
assert left.region.width == 13
assert right.region.width == 11
assert left_button.tooltip == "Open Context rail"
assert right_button.tooltip == "Open Inspector rail"
```

The production mutation these assertions catch is leaving the current hardcoded `vertical=True` / width-3 path active when the preference is false.

- [x] **Step 2: Add failing ChatScreen preference tests**

Mount a fresh `ChatScreen` twice with literal app configs:

```python
@pytest.mark.parametrize(
    ("stacked", "left_width", "right_width"),
    [(False, 13, 11), (True, 3, 3)],
)
async def test_fresh_console_composes_saved_rail_label_style(
    stacked, left_width, right_width
):
    app = _build_test_app()
    app.app_config.setdefault("console", {})[
        "stack_collapsed_rail_labels"
    ] = stacked
    host = ConsoleHarness(app)
    async with host.run_test(size=(160, 45)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        left = console.query_one(
            "#console-context-rail-handle", ConsoleRailHandle
        )
        right = console.query_one(
            "#console-inspector-rail-handle", ConsoleRailHandle
        )
        assert left.region.width == left_width
        assert right.region.width == right_width
```

Also assert the expected rendered labels for both modes and that open/collapse behavior still hides the corresponding handle.

- [x] **Step 3: Run the tests and verify RED**

Run:

```bash
.venv/bin/python -m pytest Tests/UI/test_console_rail_handle.py Tests/UI/test_console_shell_regions.py -q
```

Expected: the false case still composes stacked width-3 handles.

- [x] **Step 4: Implement the minimal preference read and composition branch**

Add beside `_console_collapse_large_pastes_enabled`:

```python
def _stack_collapsed_rail_labels(self) -> bool:
    """Return whether fresh collapsed Console handles use stacked labels."""
    app_config = getattr(self.app_instance, "app_config", {}) or {}
    console_config = app_config.get("console", {})
    if not isinstance(console_config, dict):
        return False
    return coerce_bool_setting(
        console_config.get("stack_collapsed_rail_labels", False),
        False,
    )
```

Resolve once during composition:

```python
stack_rail_labels = self._stack_collapsed_rail_labels()
```

Pass `vertical=stack_rail_labels` to both handles and choose widths without changing any other rail state:

```python
left_handle_width = ConsoleRailHandle.VERTICAL_WIDTH if stack_rail_labels else 13
right_handle_width = ConsoleRailHandle.VERTICAL_WIDTH if stack_rail_labels else 11
```

- [x] **Step 5: Run the tests and verify GREEN**

Run the Task 2 command. Then mutation-check by temporarily forcing `stack_rail_labels = True`; confirm the horizontal test fails before restoring the production branch.

- [x] **Step 6: Commit the Console slice**

```bash
git add Tests/UI/test_console_rail_handle.py Tests/UI/test_console_shell_regions.py tldw_chatbook/UI/Screens/chat_screen.py
git commit -m "feat(console): honor collapsed rail label preference"
```

---

### Task 3: Add the staged, searchable Settings control

**Files:**
- Modify: `Tests/UI/test_settings_configuration_hub.py`
- Modify: `Tests/UI/test_settings_save_commit_models.py`
- Modify: `tldw_chatbook/UI/Screens/settings_screen.py`

**Interfaces:**
- Consumes: existing `SettingsDraft`, `_stage_console_default_value`, `SettingsConfigAdapter.save_sections`, field search/focus registries, and Console Behavior Save/Revert actions.
- Produces: checkbox `#settings-console-stack-collapsed-rail-labels`, status `#settings-console-rail-label-style-status`, staged payload `{"console": {"stack_collapsed_rail_labels": bool}}`, and success-only runtime updates.

- [x] **Step 1: Write failing mount, state-text, search, and keyboard tests**

Use the production Settings destination harness and assert:

```python
toggle = screen.query_one(
    "#settings-console-stack-collapsed-rail-labels", Checkbox
)
status = screen.query_one("#settings-console-rail-label-style-status", Static)
assert toggle.value is False
assert str(toggle.label) == "Stack collapsed rail labels"
assert "Saved style: Horizontal" in str(status.renderable)
```

Drive Space with the checkbox focused and assert `Selected style: Stacked — unsaved` appears while `app.app_config` remains false. Drive `/`, type `vertical`, press Enter, and assert the checkbox owns focus. Assert the focused guide includes Purpose, Consequences, Saved as, Applies, and the canonical key.

- [x] **Step 2: Write failing Save/Revert/failure tests**

Use the existing adapter seam, but assert real screen behavior and the exact boundary payload:

```python
assert saved == [{"console": {"stack_collapsed_rail_labels": True}}]
assert app.app_config["console"]["stack_collapsed_rail_labels"] is True
assert "Rail labels: Stacked" in _visible_text(screen)
```

Stage the rail preference plus one existing Console field, invoke Revert, and assert both drafts disappear and the status reports the actual restored style. Make `save_sections` return `False`; assert the checkbox stays staged, runtime config stays false, and the result says the draft was retained and active style stayed Horizontal.

- [x] **Step 3: Run the Settings tests and verify RED**

Run:

```bash
.venv/bin/python -m pytest Tests/UI/test_settings_configuration_hub.py Tests/UI/test_settings_save_commit_models.py -q
```

Expected: selector lookup fails because the field does not exist.

- [x] **Step 4: Wire the field through every Settings registry**

Add `stack_collapsed_rail_labels` to `CONSOLE_BEHAVIOR_CONSOLE_KEYS` and `CONSOLE_BEHAVIOR_SAVE_ORDER`. Add the field to:

- `FIELD_SEARCH_INDEX` with visible label plus aliases `rail handle stacked vertical context inspector`;
- Console Behavior focus IDs;
- `SettingsOwnershipRecord.owns_config_sections`;
- Console Behavior category summary text.

Add `_syncing_console_rail_label_style = False`, include the normalized loaded value in `_console_behavior_loaded_values`, and derive draft-aware state through `_console_behavior_value`.

- [x] **Step 5: Compose and stage the control**

Before Composer paste handling, render:

```python
yield Static("Rail presentation", classes="destination-section")
yield Checkbox(
    "Stack collapsed rail labels",
    value=bool(self._console_behavior_value("stack_collapsed_rail_labels")),
    id="settings-console-stack-collapsed-rail-labels",
    tooltip="Use narrower stacked Context and Inspector rail labels.",
)
yield Static(
    self._console_rail_label_style_status(),
    id="settings-console-rail-label-style-status",
    classes="settings-help-copy",
)
```

Handle `Checkbox.Changed`, ignore sync events, stage via `_stage_console_default_value`, update the status/result, and refresh category draft widgets. Extend `_sync_console_behavior_widgets` so Save/Revert restore the checkbox and status without generating a new draft.

- [x] **Step 6: Add focused guidance and feedback**

Add the exact four focused rows from the design. On successful Save and Revert, include the actual loaded style in the result; on failed Save, preserve the draft and state that the active style did not change. Keep existing generic substrings where incumbent tests depend on them, e.g. `Console behavior settings saved.` followed by the rail result.

- [x] **Step 7: Run the Settings tests and verify GREEN**

Run the Task 3 command. Mutation-check the allowlist by temporarily removing `stack_collapsed_rail_labels` from `CONSOLE_BEHAVIOR_CONSOLE_KEYS`; confirm the payload test fails, then restore it.

- [x] **Step 8: Commit the Settings slice**

```bash
git add Tests/UI/test_settings_configuration_hub.py Tests/UI/test_settings_save_commit_models.py tldw_chatbook/UI/Screens/settings_screen.py
git commit -m "feat(settings): add Console rail label preference"
```

---

### Task 4: Prove navigation lifecycle, document the setting, and close the task

**Files:**
- Modify: `Tests/UI/test_settings_configuration_hub.py` or create `Tests/UI/test_console_rail_label_setting.py`
- Modify: `Docs/User_Guide/settings.md`
- Modify: `Docs/User_Guide/console/chat-basics.md`
- Modify: `backlog/tasks/task-14650 - Make-Console-rail-label-style-configurable.md`

**Interfaces:**
- Consumes: successful Settings save updates `app.app_config`; each new `ChatScreen(app)` reads that object during composition.
- Produces: one end-to-end regression proving Save-to-fresh-Console behavior, user documentation, visual evidence, and complete Backlog notes.

- [x] **Step 1: Write the failing lifecycle integration test**

Drive the real boundary in one test: stage and save the checkbox through a mounted Settings screen, then mount a fresh `ChatScreen` with the same app object and assert both handles are stacked. Repeat the fresh Console assertion after a failed save and confirm it retains the previous horizontal style. Do not mutate `app_config` directly between Settings and Console assertions.

- [x] **Step 2: Run the lifecycle test and verify RED**

Run the exact node ID with `-v`; expected: before the final wiring, either the saved app config or fresh Console geometry is wrong.

- [x] **Step 3: Make the smallest integration correction, if needed**

Only correct the boundary identified by the failing test: success-only runtime update in Settings or fresh-screen config resolution in `ChatScreen`. Do not add event buses, live widget mutation, or another settings service.

- [x] **Step 4: Run the lifecycle test and verify GREEN**

Re-run the exact node ID and the targeted Task 1–3 suites.

- [x] **Step 5: Update both user guides**

Document:

- Horizontal is the default.
- Settings path: Settings > Console Behavior > Rail presentation.
- Stacked saves horizontal space by using three-column handles.
- Save/Revert operate on every unsaved Console Behavior edit.
- A successful Save is visible when returning to Console; no restart is required.

- [x] **Step 6: Run static and targeted verification**

```bash
git diff --check
.venv/bin/python -m pytest Tests/test_config_console_defaults.py Tests/UI/test_console_rail_handle.py Tests/UI/test_console_shell_regions.py Tests/UI/test_settings_configuration_hub.py Tests/UI/test_settings_save_commit_models.py -q
```

Run the repository linter/formatter commands used by the branch if configured. If TCSS was not changed, do not regenerate the CSS bundle.

- [x] **Step 7: Perform isolated Textual visual verification**

Launch with a scratch config/data profile, capture Settings plus horizontal and stacked Console renders, and verify compositor paint/geometry at supported widths. Never point the run at the real config or data directory. Record screenshot paths and commands in the Backlog Implementation Notes.

- [x] **Step 8: Run full regression verification**

```bash
.venv/bin/python -m pytest -q
```

If failures occur, compare the identical command and failure set against a clean `origin/dev` worktree before classifying regressions.

- [x] **Step 9: Complete Backlog and commit**

Check all acceptance criteria only after evidence is green. Add concise Implementation Notes covering approach, files, tests, visual verification, ADR decision, and deviations. Set TASK-14650 to Done through the Backlog CLI.

```bash
git add Docs/User_Guide/settings.md Docs/User_Guide/console/chat-basics.md \
  "backlog/tasks/task-14650 - Make-Console-rail-label-style-configurable.md" \
  Tests/UI/test_console_rail_label_setting.py
git commit -m "docs(console): document configurable rail labels"
```

- [x] **Step 10: Update the existing pull request**

Push `codex/vertical-console-rail-labels`, update PR #1429 against `dev` with the setting behavior and verification evidence, and confirm its checks start successfully.
