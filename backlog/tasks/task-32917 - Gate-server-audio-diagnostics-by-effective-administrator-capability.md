---
id: TASK-32917
title: Gate server audio diagnostics by effective administrator capability
status: In Progress
assignee: []
created_date: 2026-09-23 20:15
updated_date: 2026-09-27 16:24
labels: []
dependencies: []
references:
- backlog/decisions/178-server-audio-diagnostic-admin-boundary.md
- https://github.com/rmusser01/tldw_server/pull/2968
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Match Chatbook connected-mode audio diagnostics to the server administrator authorization contract so ordinary users see an explicit admin-required outcome.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Passive STT health remains usable for ordinary connected users.
- [x] #2 A server 403 for either diagnostic becomes an explicit admin-required result.
- [x] #3 Targeted connected-mode tests cover admin and nonadmin outcomes.
- [x] #4 Warm STT health and streaming diagnostics honor the server effective administrator capability before dispatch.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-09-23-server-audio-diagnostic-admin-parity.md
Final latest-dev qualification: reproduce the existing from_config client-factory bootstrap failure on dev 74965f694; align that one real-config test with the existing bootstrap_profile fixture contract, keeping its assertions intact; rerun all four affected test modules without exclusions, security/style and artifact guards; obtain read-only review; push the approved task-ID repair and rebase with exact force-with-lease; require fresh checks and Qodo review before merge.
Qualify rebase onto c041b6d81 after SSH PR2838 merged. All ten prior PR patches remain identical with no source overlap. SSH's landed task occupies33009, so continue the previously approved task-renumber exception by moving this PR's provider record from33009 to the currently free33010 and updating ADR179's link, preserving all task content and recording provenance. Run all four affected modules without exclusions, official inventory/Backlog guards, style/compile/Bandit; publish with exact lease d7122c6122d91c6febd766110dcabc9800edce48, then wait for fresh required checks and Qodo.
<!-- SECTION:PLAN:END -->
## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
PR: https://github.com/rmusser01/tldw_chatbook/pull/2822. Targeted service/scope tests: 15 passed, 1 existing from_config bootstrap failure excluded; Ruff/compileall/diff checks passed.

Qodo wildcard review resolved by server can_run_audio_diagnostics capability. Revised targeted tests: 22 passed, 1 unrelated from_config bootstrap test deselected; Ruff/format/compileall/diff checks passed on touched small files. Older servers without this capability defer to protected diagnostic endpoints.

2026-09-25: rebased PR #2822 onto Chatbook dev 3b11bf972 after OmniVoice merge. All three commits unchanged by range-diff; 22 focused tests passed, one pre-existing from_config bootstrap test deselected; diff check clean. Final required derived-artifact CI must rerun on this head.

2026-09-25: prior PR Fast Lane failed in two unchanged MCP Workbench UI tests; both passed locally and their files were identical across old/new dev and PR. A failed-job rerun was queued when dev advanced to 461668df0 (ingestion-only merge). Rebased onto that dev; all four prior commits unchanged by range-diff, 22 focused audio/API tests passed with one pre-existing from_config bootstrap test deselected, diff check clean. New required CI will decide merge readiness.

2026-09-25: rebased PR #2822 onto Chatbook dev 50f6096db after the tier-2 P2 Notes/Library/Media merge. All five prior PR commits are patch-identical by range-diff. The 22 focused audio/API tests passed (one pre-existing from_config bootstrap failure deselected), and git diff --check is clean. Await the required latest-head derived-artifact gate.

2026-09-25: rebased PR #2822 onto Chatbook dev 1b61ee2c0 after the off-by-default Tamagotchi merge. All six prior PR commits are patch-identical by range-diff; 22 focused tests passed with one pre-existing from_config bootstrap failure deselected; git diff --check is clean. Latest-head required CI must rerun.

2026-09-26 CI repair: PR Fast Lane and UI Fast Lane passed; Derived Artifacts failed on diagnostic inventory summary drift. Rebased onto dev 0053a0a40; all seven prior patches are identical. Local red checker reports only aggregate owner_files 596->597 and task_494_calls 7539->7541. The committed pin already contains 597 unique owner rows totaling 7541 TASK-494 calls; owner rows, diagnostics digests, and sink topology show no drift. Regenerate with the official script and verify that only these stale summary fields change.

Inventory red/green complete: official --write changed only owner_files and task_494_calls; a fresh --diff checker passes with 597 owners, 7541 TASK-494 calls and unchanged 14 sink files. The 22 focused audio/API tests passed (one known unrelated bootstrap test deselected); git diff --check passes. No Python source changed, so prior touched-source Bandit remains applicable. Commit the generated JSON plus plan/task records and require latest-head CI before merging.

2026-09-26 integration: dev c4225b5d3 merged tier-2 P3c and re-pinned the inventory. Rebase conflict was only our superseded two summary counters. Kept the newer upstream inventory unchanged, including its source_readiness owner addition. The first seven PR patches are identical by range-diff; the eighth now retains only plan/task evidence. Official inventory checker passes with 598 owners and 7542 TASK-494 calls; 22 focused audio/API tests pass (one known bootstrap test deselected). No Python source changed and diff check passes. Require fresh CI on this current-dev head before merge.

2026-09-26 late integration: required Derived Artifacts passed on 85426dd9f, but dev advanced to a3001f0ee with theme UX and MCP character authoring. Rebase completed cleanly and all eight prior PR patches are identical by range-diff. Official inventory checker passes with 598 owners and 7543 TASK-494 calls, retaining the upstream pin unchanged. The 22 focused audio/API tests pass with the same known bootstrap test deselected; git diff --check passes. No Python source changed and prior security qualification remains applicable. Fresh required CI must qualify the new head before merge.

2026-09-27: all checks, including required Derived Artifacts, passed on `00c587ef51`, but dev advanced through tier-2 P2c and OmniVoice wizard merges to `38084b68c`. Rebased cleanly; all eight prior patches are identical by range-diff. The upstream generated inventory is preserved unchanged. The focused service/scope/capability run passed 21 cases with one known unrelated from_config bootstrap test deselected. No Python source changed; prior touched-source Bandit remains applicable. The official inventory checker passes with 597 owners, 7548 TASK-494 calls and 14 sink files. Diff check passes. Fresh required CI must qualify this latest-dev head before merge.

2026-09-27 second integration: all checks and required Derived Artifacts passed on 805c20ec5, but dev advanced through the tier-2 security merge to 906b6e253. Rebased cleanly; all eight prior patches are identical by range-diff. The upstream generated inventory remains unchanged; the official checker passes with 597 owners, 7543 TASK-494 calls and 14 sink files. The focused service/scope/capability run passed 21 cases with one known unrelated from_config bootstrap test deselected. Diff check passes. No PR Python source changed; prior touched-source Bandit remains applicable. Fresh required CI must qualify this latest-dev head before merge.
2026-09-27 integration approval: the requester explicitly approved the manual task-renumbering exception. TASK-32917 remains the older audio diagnostic task; the younger provider-engine task moves to free ID TASK-33009 and ADR-179's inbound task link is updated. Status, ownership, acceptance criteria and implementation notes of the provider task are preserved, with provenance recorded. Continue PR #2822 on latest dev and require fresh CI before merge.
Baseline evidence 2026-09-27: test_server_audio_services_service_from_config_returns_provider_backed_service fails unchanged on latest dev 74965f694 with RecoveryRequired(raw_source_selection_changed), via build_client -> TLS trust -> real config read after the global per-test config redirect. Tests/conftest.py explicitly supports a per-node bootstrap_profile marker for real config consumers. Qualify this test under that existing isolated bootstrap fixture rather than excluding it. Production configuration/recovery guards remain intact.
Final local qualification on dev 74965f694: approved TASK-32917/TASK-33009 collision repair preserves provider task content and updates ADR-179. All nine prior patches survive rebase identically. Inventory checker passes (600 owners, 1362 TASK-492, 56 TASK-31551, 7550 TASK-494, 14 sinks); task guard passes 4436 files. The unchanged from_config failure reproduced on dev; applying the existing bootstrap_profile marker to its real TLS/config consumer preserves all assertions and yields 26 passed across the four affected modules with no exclusions. Ruff lint/format on five small files, changed-module compileall, diff check and touched-source Bandit pass (zero findings). Both server dependencies are merged. Final-head remote CI/Qodo and merge confirmation remain pending.
2026-09-27 16:19 integration: all checks including required Derived Artifacts passed on d7122c612; Qodo reviewed that exact head with zero bugs/rule violations/cross-repo conflicts and zero unresolved threads, and independent review found no issues. Live dev advanced to c041b6d81 (SSH workspace bindings PR2838). Clean rebase preserves all ten patches identically; no overlapping paths. The landed SSH task claims33009, reproducing an ID collision with the provider record renamed in this PR. Continue the requester's approved manual-renumber exception by moving only that unmerged provider record to free33010, updating its ADR179 inbound link and appending provenance. Older audio32917 and landed SSH33009 remain intact.
SSH-base qualification complete on c041b6d81: all ten previously reviewed patches are identical after rebase, with no changed PR source/tests. Continued approved provider-task repair now uses TASK33010 because the landed SSH record claims33009; original provider content remains byte-identical except ID and appended provenance, and ADR179's decoded link resolves to the renamed record. 26 tests pass across all four targeted modules with no exclusions; task guard passes4437files; inventory reproduces603owners/1395TASK492/56TASK31551/7545TASK494/14sinks unchanged from upstream. Ruff lint/format on five small files, changed-module compileall, diff checks and Bandit pass with zero findings. Exact lease for publishing is d7122c6122d91c6febd766110dcabc9800edce48. Fresh final-head CI/Qodo and GitHub merge confirmation remain required.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Chatbook now consults the server effective audio-diagnostic capability before privileged connected-mode diagnostics, preserves passive health, supports older servers through endpoint authorization, and maps 403 to admin_required. PR #2822 carries the compatibility change.
<!-- SECTION:FINAL_SUMMARY:END -->
