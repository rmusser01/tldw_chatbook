---
id: TASK-39164
title: "Six Console edit-and-resend and regenerate-branching tests are red on dev under their own bootstrap_profile marker"
status: To Do
created_date: 2026-11-08 06:54
labels:
- follow-up
- tests
- console
priority: low
references:
- Tests/Chat/test_console_edit_resend.py
- Tests/Chat/test_console_regenerate_branching.py
dependencies:
- TASK-39045
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 12 of the 2026-10-23 burn-down. TASK-39045's notes (round-12 branch fu/r12-ctests) name three edit-and-resend reds with "causes not read", and the ctlr stream's covering runs found the regenerate ones. Checked on dev b590a15afd and on the round-12 stack 9091820bf2, 2026-11-08 (run.sh, `-n 4`, load 24-35). The same six nodes fail on both trees, plain and under a temporary marker. Both files carry `pytestmark = pytest.mark.bootstrap_profile`, so these failures are their own.

**`Tests/Chat/test_console_edit_resend.py`:**
- `test_edit_and_resend_forks_user_sibling_and_streams_reply`: `RuntimeError: Generation projection persistence is unavailable.`
- `test_edit_and_resend_skill_refusal_leaves_no_pending_node`: `TypeError: …_refuse() takes 1 positional argument but 2 were given`.
- `test_edit_and_resend_carries_anchor_attachments`: `assert False is True`.

**`Tests/Chat/test_console_regenerate_branching.py`:**
- `test_regenerate_persists_new_sibling_when_store_has_persistence`: `RuntimeError: Generation projection persistence is unavailable.`
- `test_regenerate_stream_failure_retains_failed_sibling_and_restores_anchor`: the restored anchor id differs (`assert '…' == '…'`).
- `test_regenerate_mid_conversation_failure_restores_selected_anchor_not_former_tail`: the selected-anchor list differs.

Neither file runs on a PR lane.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Each of the six nodes passes on dev, or is retired with the commit that changed the behaviour it pinned named; each red is shown first with the same command
- [ ] #2 Each is classified as test drift or a product defect; a product defect (for example a failed regenerate that restores the wrong anchor) gets a test that is red before its fix
- [ ] #3 Both files run on a PR lane (never the fast PR contract), with three timings recorded
<!-- AC:END -->
