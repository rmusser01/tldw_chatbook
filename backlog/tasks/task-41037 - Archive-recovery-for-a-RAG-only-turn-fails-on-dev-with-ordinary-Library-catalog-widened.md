---
id: TASK-41037
title: Archive recovery for a RAG-only turn fails on dev with 'ordinary Library catalog
  widened'
status: In Progress
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
assignee:
- '@codex'
updated_date: 2026-11-15 18:41
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`Tests/UI/test_console_archive_recovery_flow.py::test_native_rag_only_lifecycle_matrix_without_index` fails all three params (`chat_only`, `workspace_only` and `chat_and_workspace`) on dev ecad404bd4 (2026-11-15). The recovered turn ends as "Agent run failed: unexpected provider error (ordinary Library catalog widened)". That message is the refusal `Agents/ordinary_tree.py:432` raises when the Library tool catalog grows during an ordinary turn.

The 2026-11-12 Phase 3 verification saw the same three nodes red on dev 4a9c4c1132, right after #3480 (TASK-39147) added that check. Every other node in the file passes.

If the product is wrong, a user who restores an archived chat or workspace and asks a RAG-only follow-up gets a failed run instead of an answer. The file runs on no PR lane.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The task notes say whether the refusal is correct for this flow or a product defect, and name the commit
- [ ] #2 All three params pass on dev; a fixed product defect is shown red first
- [x] #3 The file runs on a PR lane, or the task notes say why not
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no. Existing ADR-166 and ordinary Console authority apply; correct the mismatched name without changing the authority boundary.
1. Reproduce the mounted RAG-only failure on clean dev and attribute the mismatch.
2. Capture the canonical RAG provider name in the ordinary Library maximum. Leave Direct, eligibility and widening guards unchanged.
3. Run the three mounted restore scopes and focused authority checks. Gate the three nodes on separate UI lanes rather than adding the full mounted file.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Product defect introduced in `11d71e5db57` (merged in PR #3480): ordinary-turn admission froze `search_rag`, while the actual RAG fallback advertises `search_library_rag`. The widening refusal was correct for a mismatched catalog; the captured maximum was wrong. Use the existing provider constant in the RAG-only branch; keep the strict check and Direct catalog intact. ADR-166 already specifies no-index recovery with Library access allowed.

On dev `cc6006fcf5`, the untouched mounted chat-only case failed with the exact catalog refusal before the fake gateway received a request. All three scopes passed with the patch (3 passed). The full mounted file stays out of the fast lane because of its size; existing coverage is retained and the three regression nodes are appended to separate shards (20, 21, 22 after rebase). Current patched tests also passed 39 mounted card/plateau/imported-result checks.

The PR #3162 historical failures were separate: the boot census rejected a recovered-image worker that started with no images, and broad Console prompt selectors exceeded the CSS limit. Dev already carries fixes (including `ab43a0303b`), and the two Perf Guard guards and Console sheet-size guard pass. The Derived Artifacts markup failures were in Kanban notifications and fixed by `4611fecb6a`; the current markup checker passes. Four first-send, custody, and preparing checks from the historical failures also pass. The seven custom Library factory checks fail both with the patch and with the original dev module, outside this archive regression.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
