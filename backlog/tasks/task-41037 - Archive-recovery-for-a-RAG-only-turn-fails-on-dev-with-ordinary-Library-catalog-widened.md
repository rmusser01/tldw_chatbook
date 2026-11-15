---
id: TASK-41037
title: "Archive recovery for a RAG-only turn fails on dev with 'ordinary Library catalog widened'"
status: To Do
created_date: 2026-11-15 02:03
labels:
- follow-up
- architecture-review-2026-10-25
- bug
- console
- tests
priority: medium
references:
- Tests/UI/test_console_archive_recovery_flow.py
- tldw_chatbook/Agents/ordinary_tree.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`Tests/UI/test_console_archive_recovery_flow.py::test_native_rag_only_lifecycle_matrix_without_index` fails all three params (`chat_only`, `workspace_only` and `chat_and_workspace`) on dev ecad404bd4 (2026-11-15). The recovered turn ends as "Agent run failed: unexpected provider error (ordinary Library catalog widened)". That message is the refusal `Agents/ordinary_tree.py:432` raises when the Library tool catalog grows during an ordinary turn.

The 2026-11-12 Phase 3 verification saw the same three nodes red on dev 4a9c4c1132, right after #3480 (TASK-39147) added that check. Every other node in the file passes.

If the product is wrong, a user who restores an archived chat or workspace and asks a RAG-only follow-up gets a failed run instead of an answer. The file runs on no PR lane.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The task notes say whether the refusal is correct for this flow or a product defect, and name the commit
- [ ] #2 All three params pass on dev; a fixed product defect is shown red first
- [ ] #3 The file runs on a PR lane, or the task notes say why not
<!-- AC:END -->
