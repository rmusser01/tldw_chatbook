---
id: TASK-31932
title: Reconcile PR 2427 with latest dev and complete review gates
status: In Progress
assignee:
  - '@codex'
created_date: '2026-09-06 17:14'
updated_date: '2026-09-06 17:38'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Preserve the reviewed test repairs and newly landed dev behavior while making PR 2427 eligible for normal reviewed integration.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The PR contains latest dev with all reviewed behavior preserved and no unresolved rebase conflicts.
- [x] #2 Review-created Backlog collisions are renumbered with upstream identities and historical evidence paths preserved.
- [ ] #3 Affected complete-file tests and derived artifact checks pass without weakening contracts or raising screen size limits.
- [ ] #4 Qodo findings and required checks on the final revision are handled before normal merge.
- [x] #5 Newly landed character-navigation test fixtures finalize only their own resources, with all seven cases passing and no retained SQLite descriptors under native attribution.
- [x] #6 New dev boot-worker warning probes retain their owned Loguru sinks for the mounted observation window, and newly attributed worker/smoke fixtures finalize exact owned resources.
- [x] #7 Incremental agent-step persistence tests finalize their owned database and write real run logs under their test-owned root without retaining a process-global workspace database.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Preserve the pushed head and isolated worktree, rebase onto fetched dev, and reconcile overlapping owners without losing upstream behavior.
2. Census all task buckets and refs/worktrees; renumber only review-owned collisions, preserve upstream identities and evidence paths, and verify the exact mapping.
3. Run complete targeted files for rebased runtime contracts and architecture inventories. Reconcile stale owner/constructor assertions and tighten slack budgets to measured counts. Keep genuine size failures open; obtain design approval before further owner extraction.
4. Review diagnostic statement deltas before regenerating the manifest. Run CSS, profile-path, schema, index, Backlog, affected tests, and scoped static checks.
5. Publish with an exact force-with-lease, request normal review, address verified Qodo findings, and merge without bypass only after final-head review and required checks succeed.
6. Apply the existing exact-owner real-app fixture adapter to the three new character-navigation test constructors, preserving all behavioral assertions. Verify the complete file with native descriptor attribution and shared cleanup fault controls. This is test-only reuse of established lifecycle APIs, with no new ADR or ownership design required.
7. Extend the already reviewed Loguru capture-lifetime repair (TASK-32014) to newly landed boot-worker parameter cases, keeping all three state transitions and the unknown-worker positive control. Attribute and close only exact worker/smoke fixture-owned resources through existing test lifecycle adapters, then run both complete files and shared cleanup controls with native attribution.

ADR required: no for the rebase and test/derived-artifact reconciliation.
ADR path: N/A; existing DESIGN.md section 7 governs retained controller ownership.
Reason: these steps preserve established storage, runtime, and UI contracts. Any new architectural change requires a separately approved design.

8. User-approved 2026-09-07: move private Canvas actions and citation discovery helpers into the existing ConsoleMessageController; preserve screen event, UI, and worker hooks, current dev behavior, and unchanged size limits. Verify all affected complete files and callback-ownership contracts.
9. User-approved 2026-09-07: restore useful settings failure diagnostics using fixed operation/phase, exception type, and validated opaque identifiers only. Exclude raw exception text, drafts, credentials, and provider/model labels. Add fault-injection privacy and usefulness assertions before implementing; regenerate the diagnostic inventory only after reviewing statement changes.
10. Finalize the incremental agent-step fixture with its existing database close API and supply RunLogWriter with the test-owned database directory. This preserves real logging while avoiding unrelated process-global workspace lookup. Verify the complete file with native descriptor attribution. No new ADR: test-only ownership using established constructors and lifecycle methods.
11. The second dev rebase removed the previously reviewed Save-as-Note owner repair and regression. Restore its test first, verify the wrong-owner failure, and retain the established notes_user_id contract. This is a routine regression repair under the existing notes ownership boundary, not a new storage design; unrelated upstream feature removals remain outside this repair.
12. Repair the reproduced Canvas gateway partial-body read: collect at most the existing byte limit plus one through EOF before JSON decoding; preserve oversized/malformed request refusal and cancellation. Add deterministic split-stream/boundary controls, run the complete gateway and browser files, and obtain independent review. Existing ADR-121 governs the unchanged bounded request/confirmation boundary; no new ADR required for this transport bug fix.
13. Reconcile only the four upstream appearance-width assertions with exact conversation identity, full state/tooltip text, and the owning tray's wrapped/truncated display contract; preserve service, selection, persistence, and resume assertions. Address Qodo comment 3954688544 by using the existing database transaction manager for both recovery-test reads, keeping queries and assertions unchanged. These are test-only contract reconciliations under steps 3 and 5, not new runtime or ownership designs; no ADR required. Verify both complete affected files before publication.
14. User approved the remaining Console/Library existing-owner moves, narrow selectors and Watchlists-only lazy stylesheet after checkpoint c27723b623. Execute the CSS work through `Docs/superpowers/plans/2026-09-08-pr2427-css-gate-paydown.md` under existing ADR-097, preserving all current caps and appearance. Record each controller cluster's exact owner/binding plan before its move, following DESIGN.md section 7 and existing Library migration governance. Keep source moves, test-callsite cleanup and any runtime bug repair attributable; retain all behavioral assertions and obtain independent spec/correctness review before publication.
15. The complete affected-file runs exposed stale fixtures from upstream changes, independent of the owner moves. Reconcile the two Console overflow-menu controls with TASK-31759's two added note actions, retaining exact ordered labels, captured-message/dismissal checks and explicit keyboard traversal of every enabled entry. Reconcile only the Scheduling overflow test's viewport with TASK-31712's intentional padding reduction, preserving all painted-scroll assertions and the separate 235x52 lifecycle check; repair the tooltip's internal reminder nouns to the existing scheduled-task vocabulary under TASK-23106. Verify complete affected files. ADR required: no; these are test-fixture and copy reconciliations of established behavior, not new UX or runtime contracts.
16. Reconcile dev 603812300f after the completed Wave7 rebase. Preserve scoped repeated-fault detection, real Media-state resume cache reset, and Copy selection. Retarget the nine stale test references to the existing MediaState owner without changing assertions; reproduce and investigate the remaining painted/layout failures separately. No new ADR: existing ownership and behavior remain unchanged.
17. Remove repeated status-widget updates and identical action-button construction within the existing Character Repair dialog. Keep fresh widget lookup, exact button IDs/classes/order/nesting and all repair/cancel/worker behavior. Characterize those contracts first, run the complete repair and existing CSS-parity files, and retain the 502-line ceiling. No new ADR: local deduplication without ownership, UI, or service-contract changes.
18. Reconcile painted tests with TASK-31983's focus-only left border, TASK-31979's empty-Reader width allocation, and TASK-31980's destructive-action gap. Keep the real compositor, focus movement/selection invariants, full metadata and action labels, and exact gap measurement. Treat any genuinely clipped action label as a runtime defect, not a reason to weaken text assertions. No new ADR: preserve the three established visual contracts.
19. Complete existing Prompts/Skills private-delegator cleanup instead of the obsolete Reader extraction. The tracked-file census found no external executable/test/patch/dispatch consumers for the 22 names below; retain constructor lambdas and change only their inner receiver. Characterize replacement-owner late binding before removal, update exact architecture inventories, and verify complete affected files. Existing controller bodies and all numeric ceilings remain unchanged; tighten only the measured Screen line pin after cleanup. No new ADR: existing owners and callback contracts, under the Library decomposition recipe's private-cleanup whitelist.
20. Isolate the roleplay-writer weakref test from the unrelated ten-second Environment poll cadence using its existing interval constant. Diagnostics traced the retained Screen to the cancelled Textual sleep TimerHandle's async context; shortening only that cadence makes the unchanged assertion pass. Keep the real timer, assert it exists before departure and is stopped on departure, retain the writer/durable-repair assertions and sub-0.5-second unmount bound, and never mutate asyncio's timer heap or framework internals. Retarget two Message test calls to their explicit existing owner so the seam guard sees the same receiver. No new ADR: test-only isolation and receiver spelling; production scheduling and lifecycle behavior stay unchanged.
21. Restore the real route-CSS lifecycle in the Watchlists latest-active-run routing test using existing FullAppDestinationContext for that one case. The boot-only harness bypasses `_ensure_screen_owned_css`, clipping the button so the compositor hits its parent and the real click never reaches the handler. Preserve exact target/route/label/status assertions and real scroll/click, and explicitly assert the click hit. No production or stylesheet change; no new ADR, under the already approved Watchlists-only lazy stylesheet contract.
22. Correct the two Console runtime-ownership fixtures that stack a second ChatScreen over production startup: await the existing production-screen helper and assert one Console visit, retaining every event, detach, identity and trusted raw-draft provenance assertion. TASK-31520 deliberately suspends rather than unmounts ordinary Console navigation; these two unmount-specific tests must explicitly use the existing `_unmount_installed_console` helper after real navigation to exercise actual Textual removal and recreation. Retain separate default-route reuse coverage and every cancellation/restoration assertion; do not change production suspend semantics. Reconcile the Canvas watcher fixture with the existing store-owned shared controller while keeping gateway absence, watcher identity, disable latching and disposal checks. Root-cause traces and the three failing cases precede edits; verify the complete ownership file. No new ADR: fixture-only alignment with existing app/runtime ownership.
23. Reconcile the newly completed Library inventory (968 passed / 31 failed) by distinct cause. Pin TASK-31645's actual one-row Chunking Lab chrome in compact Notes totals and select the Import command by its stable action rather than tuple position. Preserve exact remaining geometry, focus visibility and six-row surplus growth. Retain the per-profile grip contracts from TASK-31633/31951/31952 using a genuinely collapsed viewport and exact profile-specific grip widths. Seed actual Media data for route tests that require a list, and investigate the RAG-open and Clear-path failures without changing their behavior contracts. Keep metadata slot positions separate from human-readable text normalization. No new ADR for test-only reconciliation of these existing contracts (ADR-118 governs the retained Lab entry); any demonstrated production defect requires a separately recorded minimal repair and regression evidence.
24. Repair the inherited Media deep-link selection mismatch at the existing MediaController builder: local viewer projection uses the existing canonical Reader selection, preserving current-page membership/fallback and ordinary list/external-detail behavior. Add cold/warm, non-first-item and production numeric-source-ID regressions first, preserving exact Items/Reader identity and detail/title assertions; then run affected complete files. No new ADR: correction within existing Media selection ownership, not a new state owner or service contract. Separately reuse ConsolidatedCSSApp in the ingest fixture: the read-only compositor probe found MainNavigationBar consuming 23 rows and the fold hint intercepting Clear at y44 because the plain App omitted production widget defaults. Pin real compact navigation geometry before the real click; retain input identity, focus and clearing assertions. No production CSS or ingest behavior change.

Step 19 exact Prompts inventory: `_library_prompt_editor_active`,
`_sync_library_prompt_selection`, `_sync_library_prompt_memberships`,
`_stop_library_prompts_search_debounce`, `_capture_library_prompts_filter_cursor`,
`_invalidate_library_prompt_detail_generation`, `_invalidate_library_prompt_history`,
`_sync_library_prompt_history_region`, `_update_library_prompt_status_static`,
`_notify_prompt_dirty_veto`, `_clear_library_prompt_delete_pending`,
`_library_prompt_write_worker_is_active`, `_sync_library_prompt_mutation_presentation`,
`_library_prompt_nearest_survivor_focus`, `_notify_library_prompt_delete_failure`,
`_refocus_library_prompt_delete_action`.
Skills inventory: `_library_skills_canvas_kwargs`, `_consume_library_skill_scroll_pending`,
`_library_skill_on_disk_path`, `_update_library_skill_warnings_static`,
`_set_library_skill_discard_enabled`, `_refresh_library_skill_script_grant`.
Keep the deferred mirror/focus callbacks and `_library_prompt_can_update_original`
because the census found identity-sensitive or tracked evidence consumers.

2026-09-08 checkpoint: Wave7 rebase completed at cc0a5bd537. The affected Library
group is 359 passed / 18 failed; ratchets are 43 passed / 5 failed. Verification
and final-head review remain open. Fresh refs/worktree census required one more
review-only renumber: Assistant harness TASK-32013 to TASK-32040; upstream Media
debt TASK-32013 is unchanged. Historical mappings below remain provenance.

The isolated recovery worktree is `.worktrees/pr2427-review-recovery`. Rebase onto fetched dev `3090013cfea4dbf6133ac43d024656e2eb3a2a56` completed locally at `4a74c5d7e02552a5351d59df7647ea8811526bab`; publication remains open (AC #1). The three review-only collisions were renumbered to TASK-32013/32014/32015, with 3,589 task records passing the identity guard. Agent persistence has 34 complete-file passes and no retained SQLite descriptors under the native observer. Verification and final-head review remain open; see the dated reconciliation report for exact evidence and remaining failures.
<!-- SECTION:PLAN:END -->
