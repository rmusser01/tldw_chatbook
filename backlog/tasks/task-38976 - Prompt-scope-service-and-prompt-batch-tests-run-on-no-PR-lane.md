---
id: TASK-38976
title: "Prompt scope-service and prompt batch tests run on no PR lane"
status: To Do
created_date: 2026-11-06 01:42
labels:
- follow-up
- ci
- tests
- prompts
priority: low
references:
- Tests/Prompt_Management/test_prompt_scope_service.py
- Tests/Prompt_Management/test_prompt_batch_models.py
- Tests/Prompts_DB/test_prompts_db_batch_mutations.py
- .github/workflows/derived-artifacts.yml
dependencies:
- TASK-38749
- TASK-38756
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 8 of the 2026-10-23 burn-down; checked on dev 29592ecc9b on 2026-11-05, and against the census and workflows as they will be once the round-8 branches merge.

No PR lane runs these three files. They are not in `scripts/ui_pr_gate_census.txt` and no workflow targets them. The only Prompts_DB lane target is `test_prompt_export_mixins.py` (`derived-artifacts.yml:371`). The notes of TASK-38749 and TASK-38756 both record this gap.

Runs on dev at `-n 4`, 3 runs each, load 11.9-12.6:

| File | Result | Wall time |
|---|---|---|
| `Tests/Prompt_Management/test_prompt_scope_service.py` | 187 passed | 5 s |
| `Tests/Prompt_Management/test_prompt_batch_models.py` | 51 passed, 1 failed (the lazy-export pin) | 5-7 s |
| `Tests/Prompts_DB/test_prompts_db_batch_mutations.py` | 64 passed, 2 failed (the privacy pins) | 11-12 s |

TASK-38756 fixes those three reds, and TASK-38749 adds two scope-service tests. Both are Done on branches that merge first.

Serial at load 5.5-5.8 (the dreams-boundary step runs serially at `--timeout=180`), the files take 1.4 s, 2.5 s and 11.5 s: about 16 s in all.

TASK-38760 (To Do) asks for a lane for the Prompts DB legacy and MCP execution-log tests. One PR can lane both sets in the same step.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The three files run in a PR lane step other than the PR Fast Lane, with a step comment that says why
- [ ] #2 Before they are added, each file passes 3 of 3 runs in that step's configuration on dev, and the measured times are recorded (about 16 s serial in all, measured on dev)
- [ ] #3 The job they join still ends at least 5 minutes under its cap in its slowest measured run
<!-- AC:END -->
