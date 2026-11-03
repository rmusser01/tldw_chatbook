---
id: TASK-38699
title: "library_skills_controller.py is 29 lines over its size-ratchet row on dev, and no lane runs that row"
status: To Do
created_date: 2026-11-03 11:40
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

The PR lane runs only six of this file's nodes, for the prompt-browse, prompts and media controllers (derived-artifacts.yml:369-375). So the red never reached a required check. The owner's 2026-10-31 size waiver covers re-pinning, as #3351 did for 18 module-size rows (TASK-38640).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The library modules size ratchet passes on dev, through a dated re-pin of this row under the 2026-10-31 waiver or a reduction of the file
- [ ] #2 Whether every row of this ratchet should run on the PR lane is decided and recorded, and the lane is changed if so
<!-- AC:END -->
