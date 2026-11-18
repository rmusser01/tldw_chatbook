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
- [x] #4 The first-send budget and UI failures blocking PR #3571 are diagnosed and repaired without relaxing their behavioral assertions or performance limits
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no. Existing ADR-166 and ordinary Console authority apply; correct the mismatched name without changing the authority boundary. The CI repair preserves ADR-097's import limit and ADR-250's explicit service dependencies; the existing request identity contract remains available through its original imports.
1. Reproduce the mounted RAG-only failure on clean dev and attribute the mismatch.
2. Capture the canonical RAG provider name in the ordinary Library maximum. Leave Direct, eligibility and widening guards unchanged.
3. Run the three mounted restore scopes and focused authority checks. Gate the three nodes on separate UI lanes rather than adding the full mounted file.
4. Diagnose the current-head first-send warm-up overrun and the Watchlists/backup review UI failures. Follow ADR-097's import-limit/owner-exception process and ADR-250's RAG service ownership; import the optional service off the UI loop. Replace premature UI observations with bounded publication conditions, retaining hit-testing, real clicks and pending-owner checks. Run the affected guards, request independent review, then let required PR CI and the merge queue complete.
5. For the newly exposed UI26 stale readiness-header failure, trace the hand-called poll and checked display to identify whether readiness, proof expiry or a refused maintenance pass owns the failure. Repair only that demonstrated cause; retain failure when the poll refresh is removed and rerun both wording cases and the existing mid-poll case.
6. Apply the explicitly approved ADR-097 94→98 first-send warm-up exception with its ledger row, retaining the original snapshot and zero spare headroom. Verify the guard and final rebased diff before re-arming the merge queue.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Product defect introduced in `11d71e5db57` (merged in PR #3480): ordinary-turn admission froze `search_rag`, while the actual RAG fallback advertises `search_library_rag`. The widening refusal was correct for a mismatched catalog; the captured maximum was wrong. Use the existing provider constant in the RAG-only branch; keep the strict check and Direct catalog intact. ADR-166 already specifies no-index recovery with Library access allowed.

On dev `cc6006fcf5`, the untouched mounted chat-only case failed with the exact catalog refusal before the fake gateway received a request. All three scopes passed with the patch (3 passed). The full mounted file stays out of the fast lane because of its size; existing coverage is retained and the three regression nodes are appended to separate shards (10, 11, 12 after the latest November 18 rebase). Current patched tests also passed 39 mounted card/plateau/imported-result checks.

The PR #3162 historical failures were separate: the boot census rejected a recovered-image worker that started with no images, and broad Console prompt selectors exceeded the CSS limit. Dev already carries fixes (including `ab43a0303b`), and the two Perf Guard guards and Console sheet-size guard pass. The Derived Artifacts markup failures were in Kanban notifications and fixed by `4611fecb6a`; the current markup checker passes. Four first-send, custody, and preparing checks from the historical failures also pass. The seven custom Library factory checks fail both with the patch and with the original dev module, outside this archive regression.
The rebased PR #3571 head `f9e8ab6` exposed three further CI failures on 2026-11-17. TASK-36947.3's service extraction added `RAG_Search.scope_evidence` to the first-send warm-up, exceeding ADR-097's 94-module limit by one. Remove that preimport and import the service on a worker at the shared evidence wrappers. Move only the existing immutable request identity contract into the already-resident `Chat.rag_scope`, retaining its original exports. Snapshot services and explicit/fallback session identity before yielding; the service coroutine and all authorization still run on the owning loop. Three isolated cases were red on the cold UI import, then green while switching chats and editing the original session during loading. The budget now passes at 94/94 without a raised limit. The diagnostic inventory changes only the existing content-free warning's owner.

The other two failures were premature UI observations: the Watchlists strip had no published geometry, and backup list child removal had retired the review before its sibling approval button. Controlled layout/Unmount delays reproduce both original assertions. Bounded geometry and complete replacement-list waits pass those controls while preserving the compositor hit tests, actual clicks and pending-owner assertions; no Watchlists or backup production behavior changes. Validation: 163 connected first-send/citation/scope tests, 8 mounted checks (including all three archive scopes and staged-session ownership), 207 module-size checks, and the unchanged import budget pass. Independent requesting-code-review found no actionable issues. The derived-artifact preflight and F-rule checks pass. Full suite and additional live/model traffic were not run. On November 18, hosted CI passed all three repaired failures but exposed one more readiness-header test; the Derived Artifacts aggregate failed only because that UI lane failed. Rebased onto dev `908e9840aa`, preserving all dev census entries plus the three archive scopes (floor 2246). Diagnose the exact readiness poll exit before any further repair; retain the poll-disabled negative control and wording assertions. The stale-header control failed both wording cases before forcing a genuinely new checked display read and passed both afterward. Disabling the poll still fails both cases. The connected file passed 35 cases and exposed a dev helper-signature mismatch in the remaining caller; passing its expected-header callback repaired that case (1 passed). Independent review found no actionable issue. Fresh census measurements on archived, unmodified dev give 99 warm-up modules; this rebased branch gives 98 against the existing 94 limit. The rebased derived-artifact preflight and F-rule check pass. The owner explicitly approved the narrow 94→98 exception on November 18. Record it in ADR-097 alongside the constant change; preserve the original snapshot and zero spare headroom. A further rebase onto dev `5266c33936` retains all 2251 dev UI census entries plus the three archive scopes (floor 2254). Final verification on dev `5266c33936`: 12 budget/message checks pass at 98/98, all 7 selected readiness/archive mounted checks pass, and the full derived-artifact preflight passes. Independent final-range review found no actionable issues. Hosted CI and merge remain pending.

<!-- SECTION:IMPLEMENTATION_NOTES:END -->
