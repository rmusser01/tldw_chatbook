---
id: TASK-16502
title: Console model popover crashes at mount when session has no model
status: Done
assignee:
  - '@claude'
created_date: '2026-08-15 16:10'
updated_date: '2026-08-15 16:45'
labels:
  - console
  - bug
  - textual-upgrade
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
User-reported crash: `InvalidSelectValueError: Illegal select value False.` from `Select(id='console-popover-model')` during `_on_mount`.

Same trap TASK-565 documented and swept in settings_screen.py: on Textual 8.x the Select no-selection sentinel is `Select.NULL`; `Select.BLANK` no longer exists on the widget and silently resolves through the MRO to `Widget.BLANK: ClassVar[bool] = False` (an unrelated render-optimization flag), so no AttributeError fires. `console_model_popover.py` passes `value=Select.BLANK` (== `False`) to the model Select whenever the session has no model set (fresh install, or a provider without a configured default model). `Select.__init__` stores the value unvalidated and validation runs at mount, where `False` is not a legal value — reproducing the reported traceback byte-for-byte. The `_apply` handler's `model_value in (None, Select.BLANK)` membership carries the same dead sentinel (inert today because `ModelSearchPicker.value` is `str | None`, but wrong-intent).

A repo-wide audit of the remaining `Select.BLANK` usages is TASK-16503; this task fixes only the crashing popover site.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [x] #1 The Alt+M quick popover mounts without error for a session whose settings have no model, showing the blank model row
- [x] #2 The popover no longer references the nonexistent `Select.BLANK` sentinel
- [x] #3 A mounted regression test covers the no-model popover open and was verified RED before the fix
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add a mounted regression test in Tests/UI/test_console_context_controls.py pushing ConsoleModelPopover with `model=None`; verify RED with `InvalidSelectValueError: Illegal select value False.`
2. Replace both `Select.BLANK` references in console_model_popover.py with `Select.NULL`.
3. Verify the test GREEN; run the console context-controls suite.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Replaced both `Select.BLANK` references in console_model_popover.py with `Select.NULL` — the mount-time `value=` seed (the crashing site: on Textual 8.2.8, `Select.BLANK` resolves to `Widget.BLANK == False`, which `Select.__init__` stores unvalidated and mount-time validation rejects) and the `_apply` membership check (inert today since `ModelSearchPicker.value` is `str | None`, fixed for intent). A comment at the seed site records why NULL, mirroring the settings_screen.py precedent from TASK-565.

Mounted regression test in Tests/UI/test_console_context_controls.py pushes the popover with `model=None` through the existing `_ContextHarness` and asserts the model Select sits on `Select.NULL`. Verified RED with the byte-identical user traceback (`InvalidSelectValueError: Illegal select value False.` on `Select(id='console-popover-model')` in `_on_mount`) before the fix, GREEN after.

The repo-wide audit of the ~60 remaining `Select.BLANK` usages (including deliberate False-placeholder sites that must NOT be blind-renamed) is filed as TASK-16503, with a lessons-testing-evidence.md entry recording that the mechanism was board-documented for three weeks (TASK-565) while this crash sat live in another file.

Files: tldw_chatbook/Widgets/Console/console_model_popover.py, Tests/UI/test_console_context_controls.py, backlog/docs/lessons-testing-evidence.md.
<!-- SECTION:NOTES:END -->
