---
id: TASK-38699
title: "library_skills_controller.py is 29 lines over its size-ratchet row on dev, and no lane runs that row"
status: Done
assignee:
- '@claude'
created_date: 2026-11-03 11:40
updated_date: '2026-11-04 01:14'
labels:
- follow-up
- tests
- library
priority: low
references:
- Tests/Architecture/test_library_modules_size_ratchet.py
- tldw_chatbook/UI/Library_Modules/library_skills_controller.py
- .github/workflows/derived-artifacts.yml
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in the round-5 wave of the 2026-10-23 burn-down. Checked on dev 9bbd5c2515: `Tests/Architecture/test_library_modules_size_ratchet.py` (serial) gives 1 failed and 50 passed, with `library_skills_controller.py grew to 3171 lines (budget 3142, +29)`.

The growth came from three commits, all with commit date 2026-10-31:
- a61b917479, "feat(plugins): add reviewed lifecycle management UI": +23 net;
- c450b0d351, "Refresh blocked Skills rows after explicit trust Retry": +3;
- c7b80bbf5a, "Preserve pending Skills filter when trust Retry completes": +3.

The row was last set in e001557e30 (2026-09-25).

The PR lane runs only seven of this file's nodes (derived-artifacts.yml:369-375): `test_every_controller_file_has_a_budget_row`, and the grow and slack nodes for the prompt-browse, prompts and media controllers. So the red never reached a required check. The owner's 2026-10-31 size waiver covers re-pinning, as #3351 did for 18 module-size rows (TASK-38640).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The library modules size ratchet passes on dev, through a dated re-pin of this row under the 2026-10-31 waiver or a reduction of the file
- [x] #2 Whether every row of this ratchet should run on the PR lane is decided and recorded, and the lane is changed if so
<!-- AC:END -->

## Implementation Plan

1. Re-pin `library_skills_controller.py`'s row to its measured 3171 lines, with a dated comment citing the owner's 2026-10-31 size waiver and the three commits that grew it.
2. Lane decision: run every row. The whole file takes about 1 s, imports nothing from the app, and its two sibling ratchets (`test_module_size_ratchet.py`, `test_screen_size_ratchet.py`) already run whole on the fast PR lane. Replace the seven node ids in `derived-artifacts.yml`'s fast PR contract step with the one whole-file target, and make the same change in `Tests/CI/test_ci_queue_pressure_contract.py`'s `FAST_LANE_TARGETS` pin.
3. Run the ratchet serially (Architecture), then `Tests/CI` (which parses the workflow and so validates the YAML), and preflight.

## Implementation Notes

The `library_skills_controller.py` row is re-pinned to its measured 3171 lines, 3142 -> 3171, under the owner's 2026-10-31 size waiver. The row's dated comment names the three commits that grew it. The file is not reduced.

Lane decision: run every row. The whole ratchet takes about 1 s, is pure (it reads line counts and imports nothing from the app), and its sibling ratchets `test_module_size_ratchet.py` and `test_screen_size_ratchet.py` already run whole on the fast PR lane. The seven node ids covered only the three controllers TASK-38544 changed, which is how the Skills overrun stayed red on dev unseen. `derived-artifacts.yml`'s "Run fast PR contract" step now lists the one whole file in place of those seven nodes, at the same position (+1/-7). `Tests/CI/test_ci_queue_pressure_contract.py`'s `FAST_LANE_TARGETS` changes the same way, and TASK-38544's comment there no longer claims the Skills overrun is out of scope.

Evidence (run.sh):
- Red at base, serial: 1 failed, 50 passed (`library_skills_controller.py grew to 3171 lines (budget 3142, +29)`).
- Green after: the library, module and screen ratchets together give 256 passed (serial).
- YAML: `yaml.safe_load` parses the workflow.
- `Tests/CI`, `-n 4`: 1187 passed, 1 skipped. This includes the exact-target and non-overlap pins that parse the workflow.

Files: `Tests/Architecture/test_library_modules_size_ratchet.py`, `.github/workflows/derived-artifacts.yml`, `Tests/CI/test_ci_queue_pressure_contract.py`.

