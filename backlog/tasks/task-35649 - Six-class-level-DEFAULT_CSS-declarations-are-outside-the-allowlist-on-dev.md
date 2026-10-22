---
id: TASK-35649
title: Six class-level DEFAULT_CSS declarations are outside the allowlist on dev
status: Done
assignee:
- '@claude'
created_date: 2026-10-20 03:07
updated_date: 2026-10-22 20:24
dependencies: []
labels:
- follow-up
- css
- tests
priority: medium
references:
- Tests/UI/test_widget_css_consolidation.py
- tldw_chatbook/UI/Evals/skill_eval_detail.py
- tldw_chatbook/UI/Evals/skill_eval_panel.py
- tldw_chatbook/UI/LLM_Management/llamacpp_setup_view.py
- tldw_chatbook/UI/Wizards/first_run_voice_pickers.py
- tldw_chatbook/Widgets/Library/library_file_notes_workspace.py
- tldw_chatbook/Widgets/session_summary_dialog.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by the review of Phase 9's rebase onto dev 88fe3108e5 (2026-10-20). `Tests/UI/test_widget_css_consolidation.py::test_class_level_css_stays_within_the_allowlist` fails on plain dev with six class-level `DEFAULT_CSS` declarations it does not allow:
- `UI/Evals/skill_eval_detail.py::SkillEvalDetail`
- `UI/Evals/skill_eval_panel.py::SkillEvalPanel`
- `UI/LLM_Management/llamacpp_setup_view.py::LlamaCppSetupView`
- `UI/Wizards/first_run_voice_pickers.py::VoiceOptionPicker`
- `Widgets/Library/library_file_notes_workspace.py::FileNotesHistoryDialog`
- `Widgets/session_summary_dialog.py::SessionSummaryDialog`

The guard keeps widget CSS in the generated sheets, where the boot CSS budget and the parse cache can see it. The test runs in no PR lane, so dev's CI never reported it.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 test_class_level_css_stays_within_the_allowlist passes on dev
- [x] #2 Each of the six widgets' rules lives in the generated sheets (or its allowlist entry has a recorded reason), and preflight's stylesheet checks pass
- [x] #3 Each widget renders the same before and after
- [x] #4 The allowlist test runs in a PR lane, so the next unlisted declaration fails a PR
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Confirm the six offenders at base dev 28beeb9ed5.
2. Rename each class's DEFAULT_CSS to BUNDLED_CSS, the sanctioned route the test names. The build bakes in Textual's type scope and keeps the widget-defaults tier and its tie-breaker. Then regenerate the sheets with css/build_css.py.
3. Check that every surface each widget mounts on loads the widget-defaults sheets (TldwCli or ConsolidatedCSSApp). Allowlist a declaration, with its reason, only where it does not.
4. Render parity: a scratchpad harness mounts each widget under the production sheets at 211x44 and 235x52 and exports SVG screenshots, run at base and at head, and the results are compared byte for byte.
5. Gate the allowlist test in the UI Fast Lane: append its node id at the census tail and raise MINIMUM_FILES.
6. Run each widget's own tests, the consolidation and governance tests, Tests/CI and preflight.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
All six declarations move to `BUNDLED_CSS`, the route the test prescribes, and none needs an allowlist entry. Every production host loads the widget-defaults sheets:
- `SkillEvalPanel` and `SkillEvalDetail` mount on `EvalsScreen`.
- `LlamaCppSetupView` mounts in `LLMManagementWindow`.
- `VoiceOptionPicker` mounts in the first-run wizard's Voice step.
- `FileNotesHistoryDialog` is pushed by the Folder-files workspace.
- `SessionSummaryDialog` is pushed by `app_lifecycle` at quit.

All of these run inside `TldwCli`. `css/build_css.py` regenerated `widget_defaults_{self,scoped}.tcss`. The build bakes in Textual's type scope, so the rules stay scoped to their widget, and the scoped stream keeps its losing tie-breaker.

- **Two ratchets surfaced on the way.**
  1. With the six converted, the allowlist test's second assertion, hidden until then behind the first, failed on three stale entries. `UI/Tools_Settings_Window.py`'s two classes were deleted in TASK-33081, and `Widgets/Media/media_viewer_panel.py::DeleteConfirmDialog` was removed in 445ace1c1a. The test requires removals, so the entries are gone.
  2. The boot CSS budget then failed at 608,184 B against 608,090 (`test_boot_parsed_css_bytes_stay_within_budget`, ADR-097). Base is 605,727 B. Rewriting the two modals' moved blocks as one-line rules (formatting only, the same declarations) saved 267 B. Head measures 607,917 B: +2,190 B, with 173 B of headroom left. The next boot-CSS growth must shed or defer (see the proposed follow-up in the coordinator report).
- **AC#1.** The test is red at base 28beeb9ed5 ("6 class-level CSS declaration(s) outside the allowlist", the six above) and green at head (7.5 s call under `-n 4`).
- **AC#2.** The rules live in `widget_defaults_self.tcss` and `widget_defaults_scoped.tcss`. A rebuild reproduces the committed sheets byte for byte, and preflight's generated-stylesheets check passes.
- **AC#3.** A temporary scratchpad harness, never committed, mounted each widget, or pushed each dialog, in a `ConsolidatedCSSApp` with `TldwCli.CSS_PATH`. It used the production ids, and both pickers sat in the wizard's `Horizontal > Vertical` row. The llama.cpp Collapsibles were expanded, and the Input and TextArea cursor blink was off in both arms. Each was rendered at 211x44 and 235x52 through `run.sh` on a private profile, and `export_screenshot()` SVGs were compared. Base against head: 12 of 12 byte-identical, both before and after the formatting trim. Noise control: two base runs, identical. Sensitivity control: head sources with the old sheets (the widgets' CSS missing) changed all 12 SVGs, for example the padding gone and the dialogs no longer centred.
- **AC#4.** `Tests/UI/test_widget_css_consolidation.py::test_class_level_css_stays_within_the_allowlist` is appended at the census tail, after TASK-35648's node. It lands in shard 5 of 13, and `MINIMUM_FILES` went from 589 to 590. The node is static: an AST walk with no app mount, about 5-10 s with its private-profile child. The census checker passes, and `Tests/CI` (`-n 4`) reports 1025 passed, 1 skipped.
- **Covering runs (`-n 4`).** Twenty-four files ran: the CSS build, bundle-sync and governance tests, the consolidation tests, the two performance CSS ratchets, and each widget's own tests. That gave 282 failed and 1209 passed at head, against 282 failed and 1203 passed at base on the same command. 248 of the failures are the local `RecoveryRequired: raw_source_selection_changed`. The failure sets match, except the fixed allowlist test and three `test_library_file_notes_workspace.py` tests that fail intermittently at base too: four base reruns showed 1, 1, 3 and 2 failures of the same three. `Tests/Architecture` (serial): the module-size and timer-inventory failures are identical at base, and none of them names a touched file. The screen-size ratchet passes 4 of 4.

Files: the six widget modules (`DEFAULT_CSS` renamed to `BUNDLED_CSS`; `SessionSummaryDialog`'s docstring says BUNDLED_CSS), `css/widget_defaults_self.tcss` and `css/widget_defaults_scoped.tcss` (regenerated), `Tests/UI/test_widget_css_consolidation.py` (three stale entries removed), `scripts/ui_pr_gate_census.txt`, `scripts/check_ui_pr_gate_census.py`, `backlog/docs/lessons-textual.md` (the boot-budget trap).
<!-- SECTION:NOTES:END -->
