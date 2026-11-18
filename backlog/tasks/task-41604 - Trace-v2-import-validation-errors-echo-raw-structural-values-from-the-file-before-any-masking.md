---
id: TASK-41604
title: "Trace v2 import validation errors echo raw structural values from the file before any masking"
status: Done
assignee:
- '@claude'
updated_date: 2026-11-18 03:14
created_date: 2026-11-17 19:44
labels:
- follow-up
- security
- trace
priority: medium
references:
- tldw_chatbook/Chat/trajectory_import.py
- tldw_chatbook/Chat/trajectory_export.py
- tldw_chatbook/UI/Screens/trajectory_screen.py
dependencies:
- TASK-41133
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by the round-15 trace review (TASK-41133, 2026-11-17). The problem predates round 15. It covers structural fields only; message content is not echoed.

**Where values are echoed.** `_validate_v2` (`trajectory_import.py:290`) builds its `TrajectoryImportError` messages from values read straight out of the file:
- `version` (`:298`, `:307`)
- the manifest `profile` (`:315`)
- `privacy_inventory` keys (`:392`)
- a duplicate `event_id` (`:417`)
- unknown `usage` keys (`:495`)
- `field_provenance` keys (`:556`, `:562`)

All of these are raised before the credential check at `:939`. A value shaped like a credential reaches the user unmasked.

**Where the message goes.**
- The error reaches `app.notify(str(exc), ...)` (`trajectory_screen.py:2036`) with markup parsing on. A raw value containing Rich markup such as `[/]` can therefore also raise `MarkupError`. The CLAUDE.md convention for notify text that interpolates runtime values is `markup=False`.
- A file whose `format` is not `tldw-trace` falls to the v1 validator, which echoes `format` and `version` the same way (`trajectory_export.py:1476`, `:1482`, `:1493`).

**Related.** TASK-41603 covers v1 content masking; TASK-41510 and TASK-41133 cover the masking rules.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 No import validation error shows a raw value from the file unmasked: each echoed structural value is masked by the current credential rules, or replaced by its position (for example `events[3]`)
- [x] #2 The import failure notice renders a value containing Rich markup literally, without raising
- [x] #3 Tests feed a credential-shaped `profile`, `event_id` and `field_provenance` key, and a markup-bearing value, and assert the shown message; red before the fix
<!-- AC:END -->

## Implementation Plan

1. Base check (45b1949ca3): `_validate_v2` and the v1 validator interpolate file values (`version`, `profile`, `privacy_inventory` keys, `event_id`, `usage` keys, `field_provenance` keys, reference targets, decision keys, lineage relationships, v1 `format`/`version`) into `TrajectoryImportError`; the screen calls `notify(str(exc))` with markup on.
2. One helper in `trajectory_export` masks a file value with `_mask_credential_material`; every echo in both validators goes through it.
3. `trajectory_screen.action_open_trace`'s failure notice passes `markup=False`.
4. Tests, red at base: credential-shaped `profile`, duplicate `event_id`, `field_provenance` key (and the other echo sites, v1 `format`/`version` included) never reach the message; a Rich-markup value in the file renders literally in the failure toast without raising.

## Implementation Notes

Every value an import error quotes from the file is masked by the bundle's credential rules, and the failure notice renders literally.

- **Base (45b1949ca3).** As described, plus echo sites the description did not list: `dangling ... {target!r}`, the privacy-decision field and duplicate key, the `event ... field_provenance`/`lacks matching redaction provenance` pair, the lineage relationship, and `_event_record`'s event id.
- **AC#1 (masked, not positions).** `trajectory_export._masked(value)` is `_mask_credential_material(value)[0]`; every file value interpolated into a `TrajectoryImportError` in `_validate_v2`/`_event_record` and into the v1 validator's `format`/`version` errors goes through it, so a non-secret value still reads as written (`duplicate event_id 'evt-7'`) and a credential reads `[credential redacted]`. Fixed names (`events[3]`, required field names) were already safe. The v1 mapping error quotes an exception raised over the already-masked document (TASK-41603).
- **AC#2.** `TrajectoryScreen.action_open_trace` notifies with `markup=False`. `[/]` in the message raised `MarkupError` when the toast rendered; a masked value's own `[credential redacted]` would otherwise be read as a markup tag too.
- **User Guide.** `console/semantic-trace-capture.md`: the **Import failed** notice masks quoted values and shows brackets as plain text.
- **Tests.**
  - `Tests/Chat/test_trace_v2_collaboration.py::test_an_import_error_never_shows_a_credential_from_the_file` (13 cases: version, profile, privacy_inventory key, duplicate event_id, usage key, field_provenance key and its state mismatch, dangling reference, decision field, duplicate decision, lineage relationship, v1 format, v1 version): the message holds the expected text with `[credential redacted]` and no part of the secret. Not on a PR lane (proposed follow-up).
  - `Tests/UI/test_trajectory_import_ui.py::test_import_failure_shows_file_values_masked_and_literally`: a file whose `format` is `[/] sk-...` shows `got '[/] [credential redacted]'` in the rendered toast. Appended to `scripts/ui_pr_gate_census.txt` (no trajectory import UI file was censused); `MINIMUM_FILES` 2244. Local serial time with the UI lane's flags: 3.0-6.4 s (load 66-77).
- **Red evidence.** The 13 pure cases at base: 13 failed, each with the raw secret (`... got 'sk-Xk9m...'`), every one on its intended branch. The UI test with masking in place and markup on: `MarkupError: auto closing tag ('[/]') has nothing to close`.
- **Runs** (run.sh, load 41-92): `Tests/UI/test_trajectory_import_ui.py`, `Tests/Chat/test_trace_v2_collaboration.py`, `test_trajectory_import.py`, `test_trajectory_import_bounds.py`, `test_trajectory_export.py` (`-n 4`): 736 passed. Census tests (`Tests/CI/test_ui_lane_shard_replay.py`, `test_derived_artifacts_workflow.py`, `test_ui_lane_shard_count_statements.py`, `test_ui_pr_gate_census_checker.py`): 209 passed. `scripts/check_markup_interpolation.py`: OK. `scripts/preflight.sh`: all checks passed. Size and identity ratchets (`Tests/Architecture/test_module_size_ratchet.py`, `test_screen_size_ratchet.py`, `test_chat_code_identity_ratchet.py`, `test_code_identity_ratchet.py`, serial): 270 passed.
- Files: `tldw_chatbook/Chat/trajectory_import.py`, `tldw_chatbook/Chat/trajectory_export.py`, `tldw_chatbook/UI/Screens/trajectory_screen.py`, `Tests/Chat/test_trace_v2_collaboration.py`, `Tests/UI/test_trajectory_import_ui.py`, `scripts/ui_pr_gate_census.txt`, `scripts/check_ui_pr_gate_census.py`, `Docs/User_Guide/console/semantic-trace-capture.md`.
