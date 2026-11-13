---
id: TASK-40712
title: "Pressing Review install… or Review preview… after an Import… preview closes the app"
status: Done
assignee:
- '@claude'
created_date: 2026-11-13 04:43
updated_date: 2026-11-13 05:15
labels:
- follow-up
- plugins
- bug
priority: high
references:
- tldw_chatbook/Widgets/plugin_review_modal.py
- tldw_chatbook/UI/Screens/plugins_screen.py
- tldw_chatbook/app_lifecycle.py
- Docs/User_Guide/plugins.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Filed in round 14 of the 2026-10-23 burn-down from the TASK-40332 docs stream, and confirmed by its reviewer. Checked on dev 26b2e677dd, 2026-11-12.

After an **Import…** preview, pressing **Review install…** or **Review preview…** opens the review modal. Its load worker reads `self.producer.inspection` (`Widgets/plugin_review_modal.py:823`), but an `ImportPreview` has no `inspection`. The `AttributeError` fails the worker, and a worker failure exits the app (`app_lifecycle.py:891`). The user loses the session over a button press. The Plugins User Guide page currently warns users away from it.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 After an Import… preview, Review install… and Review preview… either open a working review of the imported copy or are disabled with a stated reason; neither raises; a test drives both buttons
- [x] #2 A failure while the plugin review modal loads shows an error in the modal and leaves the app running; a test forces a load failure and pins it
- [x] #3 The Plugins User Guide page's warning about this crash is removed or replaced by the new behaviour
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Checked on fu/r14-plugins at 4af56eb459 (dev 26b2e677dd plus the docs stream), 2026-11-13. The premise held: `PluginReviewModal._load` reads `producer.inspection`, an `ImportPreview` has none, and the `AttributeError` escapes the load worker's `except (OSError, ValueError, RuntimeError)`.

1. Plugins screen: an Import… preview disables Review install… and Review preview…, says why on the buttons and in the status line, and a press that reaches the handler anyway opens nothing.
2. Review dialog: any other load failure is shown in the dialog, with the dialog left inert, instead of failing the worker.
3. Tests for both; replace the User Guide warning; lane the tests.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Summary.** An Import… preview cannot open a review any more, and a review dialog whose load fails says so and leaves Chatbook running.

- **Screen** (`UI/Screens/plugins_screen.py`):
  - `_is_import_preview()` checks for an `ImportPreview`.
  - After an Import… preview, both review buttons are disabled and read "No package to review".
  - The status line says "Import previews only list what the folder holds. To install from it, use Repository… on the plugin's folder or Add source… on its marketplace."
  - The press handler never hands an `ImportPreview` to the dialog.
  - The first label, "Review — import has no package", was cut to "Review — import has no" at 80×24 (a 26-cell button), so the label is shorter; the test asserts the painted text at 80×24 and 211×44.
- **Dialog** (`Widgets/plugin_review_modal.py`):
  - `_load` keeps its existing message for `OSError`, `ValueError` and `RuntimeError`.
  - Any other exception now shows "Review could not load (<type>). Nothing was changed. Close and open it again." It is logged by type only, never the message.
  - The dialog stays inert: `_loading` stays set and **Review action** is disabled. Without that, the action select's mount echo ran the input-changed handler, whose "Inputs changed…" replaced the message; the first green attempt failed on exactly that.
- **Docs.** `Docs/User_Guide/plugins.md` Quirks: the crash warning is replaced by the disabled buttons' text and the dialog's load-failure text.

**Red → green** (`-n 4`, load ~47):
- With the source change reverted, all three cases fail:
  - The button test fails on the status copy.
  - The load-failure test fails with `WorkerFailed: … AttributeError("'ImportPreview' object has no attribute 'inspection'")`. That is the crash: the worker failure exits the app.
- With the change, 3 passed.

**PR lane.** Two node ids appended at the tail of `scripts/ui_pr_gate_census.txt`; `MINIMUM_FILES` 2031 → 2033. Serial times with the lane flags:
- `test_import_preview_review_buttons_state_why_and_never_open` (2 sizes): 33 s at load 32→29.
- `test_review_dialog_load_failure_stays_in_the_dialog`: 18 s at load 29.

**Covering run.** `Tests/UI/test_plugins_browse.py` and `Tests/UI/test_plugin_review.py` at `-n 4`, load 53→39: 95 passed, 1 failed. The failure is `test_installed_manager_collects_all_51_exact_root_refs`, which hit pytest-timeout's 300 s while creating 51 saved-data roots under load. It does not open an Import… preview or fail a load, so neither change reaches it. Run alone at HEAD it passes in 187 s, at load 33→29.

**Files.** `tldw_chatbook/UI/Screens/plugins_screen.py`, `tldw_chatbook/Widgets/plugin_review_modal.py`, `Tests/UI/test_plugins_browse.py`, `Docs/User_Guide/plugins.md`, `scripts/ui_pr_gate_census.txt`, `scripts/check_ui_pr_gate_census.py`.
<!-- SECTION:NOTES:END -->

