# Non-Obscuring Focus PR 1 Visual QA

Date: 2026-05-28
Scope: PR 1 foundation screenshots and verification notes

## Captures

| Evidence | Path | Result |
| --- | --- | --- |
| Console focused composer | `Docs/superpowers/qa/non-obscuring-focus-selection/visual-captures/pr1-console-focused-composer.png` | Composer uses a thin non-warning focus cue and keeps the input area readable. |
| Console focused action button | `Docs/superpowers/qa/non-obscuring-focus-selection/visual-captures/pr1-console-focused-action-button.png` | Compact transcript action focus stays visible without covering the label. |
| Library focused source action | `Docs/superpowers/qa/non-obscuring-focus-selection/visual-captures/pr1-library-focused-source-action.png` | Source action focus uses readable text plus underline instead of a label-covering fill. |
| Top nav active plus focused | `Docs/superpowers/qa/non-obscuring-focus-selection/visual-captures/pr1-top-nav-focused-active.png` | Active destination remains readable and focus adds an underline cue. |
| Settings selected-category reference | `Docs/superpowers/qa/non-obscuring-focus-selection/visual-captures/pr1-settings-selected-category-reference.png` | Settings keeps the existing selected-category reference pattern readable. |

## Visual Notes

- Textual-web was launched against a temporary config with the splash screen disabled so the captures start on the Console surface.
- The Console provider setup state leaves message-level actions unavailable, so the Console action capture uses the transcript tab action button. Source contracts cover `.console-transcript-action-button:focus` directly.
- The Library source-action capture verifies that the foreground remains readable on focus and that the focus cue does not cover the action label.
- The Settings capture is a reference guard. PR 1 does not redesign Settings selection; it prevents the new global rules from regressing that surface.

## Verification Commands

Focused source contracts:

```bash
PYTHONPATH=. .venv/bin/python -m pytest Tests/UI/test_non_obscuring_focus_contract.py Tests/UI/test_master_shell_design_system_contract.py::test_library_mode_chip_focus_keeps_active_label_readable -q
```

Result recorded during implementation: `8 passed`.

Focused PR 1 suite:

```bash
PYTHONPATH=. .venv/bin/python -m pytest Tests/UI/test_non_obscuring_focus_contract.py Tests/UI/test_focus_accessibility.py Tests/UI/test_master_shell_design_system_contract.py Tests/UI/test_console_persistent_rails.py Tests/UI/test_settings_configuration_hub.py -q
```

Latest result after visual QA: `127 passed, 1 warning`.

Broader visual-parity probe:

```bash
PYTHONPATH=. .venv/bin/python -m pytest Tests/UI/test_non_obscuring_focus_contract.py Tests/UI/test_focus_accessibility.py Tests/UI/test_master_shell_design_system_contract.py Tests/UI/test_console_persistent_rails.py Tests/UI/test_console_native_transcript.py Tests/UI/test_destination_visual_parity_correction.py -q
```

Result recorded during implementation after the initial PR 1 changes: `21 failed, 136 passed, 1 warning`. The failures were in broad destination visual-parity expectations outside this foundation slice. The focus token/theme failure surfaced by that run was addressed afterward; the remaining destination parity items are not treated as PR 1 scope.

## Deferred Scan

The residual heavy-outline scan is expected to keep returning deferred legacy overrides that are inventoried in `audit-inventory.md`, especially Chat, Embeddings, and legacy sidebars:

```bash
rg -n 'outline: heavy|text-style: reverse|background: \$ds-status-(warning|error|blocked|unsaved)|background: \$warning|background: \$error' tldw_chatbook/css tldw_chatbook/UI/Navigation/main_navigation.py Tests/UI
```

PR 1 fixes only the shared/global, Console, Library, Settings, and top-nav selectors called out in the source spec. Later screen-by-screen PRs should burn down the deferred rows instead of expanding this foundation slice.

Final hygiene:

```bash
git diff --check
```

Latest result after visual QA: passed with no whitespace errors.
