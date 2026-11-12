---
id: TASK-34215
title: Console workspace or casual chats with bounded background starts
status: Done
assignee:
  - '@codex'
created_date: '2026-10-02 23:27'
updated_date: '2026-10-04 07:08'
labels:
  - console
  - agents
dependencies:
  - TASK-34214
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Let a primary Console agent create a fresh chat in its workspace or casual scope, prepare a durable draft or start one bounded background turn, and preserve user control and recoverable outcomes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 new_chat supports both destinations and draft/start modes with strict validation and compatible omitted arguments; fork_chat retains its existing contract.
- [x] #2 Fresh chats persist destination assistant and generation defaults, explicit instructions overrides, global/workspace ownership, fresh scratch/project state, and no source staged inputs or grants.
- [x] #3 Both approval caches separate mode and resolved destination under live source ownership; the mounted card shows complete prompts and overrides without focus changes.
- [x] #4 Version-2 drafts retain edits and explicit clears across activation/restart; legacy handoffs keep their decoding behavior and stale writes cannot restore old text.
- [x] #5 Authorized machine-origin starts use shared allowance and wake/start capacity, preserve provenance through durable checkpoint migration, and return truthful draft/not_started/started/review_required results.
- [x] #6 Manual Send wins while a start is prepared, source cancellation has the documented cutoff, paused preflight retains a draft, and duplicate/crash callbacks cannot repeat dispatch or consume newer input.
- [x] #7 Targeted tests, lint/format/token checks and real Console/provider verification cover both destinations/modes, recovery and view detachment before completion.
- [x] #8 Context Next Send can preview the Console request containing the new chat tools without executing creation or depending on undefined builder references.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR path: backlog/decisions/219-console-chat-destinations-and-bounded-starts.md
Reason: Implements the approved chat creation, machine-origin submission, durable draft and two-store recovery contract.

Implementation follows the approved Task 2 brief. Keep In Progress until independent review and final evidence closure.
Detailed plan: Docs/superpowers/plans/2026-10-02-console-chat-destinations-and-starts.md, Task 2.
Spec: Docs/superpowers/specs/2026-10-02-console-chat-destinations-and-starts-design.md.

1. Add failing validation, scope/grant and durable destination-default tests.
2. Extend new_chat schema, trusted preparation and both approval caches; preserve fork_chat.
3. Persist destination identity/settings and revision-fenced version-2 drafts through existing runtime/store owners.
4. Migrate conversation dispatch checkpoints for native start origin and exact-attempt receipts, consuming matching drafts atomically.
5. Add runtime-owned one-attempt starts sharing allowance and fleet capacity; audit every origin-dependent submission branch.
6. Arbitrate manual intent before busy guards, enforce acceptance/cancellation cutoff, and make view completion an observer.
7. Verify targeted SQLite/Chat/mounted-UI tests, lint/format/token checks, documentation and real Console/provider behavior before closure.
8. Repair the directly affected Context Next Send tool-aware preview builder and prove preview does not execute creation (controller ruling in execution record; AC8).
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Primary Console new_chat now supports same_workspace/casual destinations and draft/start modes, with same_workspace/draft defaults. Fresh chats persist destination assistant/model settings and explicit standing-instruction overrides, true global/workspace ownership, private scratch and editable revision-fenced v2 drafts. Source state and grants do not transfer; fork_chat retains its contract.

Both approval caches bind exact live source incarnation, destination and mode. Runtime-owned starts reserve shared automatic capacity/allowance, check required project decisions before acceptance, and require exact AgentRunsDB plus conversation receipts before provider dispatch or Started. Native machine provenance remains literal and cannot grant human/profile authority. Physical provider completion retains capacity through Stop; initial preparation and uncertain settlement remain owned and conservative. Prepared manual Send wins before busy admission; accepted targets survive source Stop. Historical launch facts persist, while consumed handoffs stop contributing live blocked activity.

AgentRunsDB v19 provides canonical allowance membership/native attempts (TASK34214); ChaChaNotes v74 preserves the new machine origin and exact receipt. Context Next Send uses explicit schema flags and previews without creating chats (AC8). Updated controller/store/bridge/runtime/checkpoint/metadata owners, approval card, composer/queue/workspace/switcher projections, migration/tests and the existing tool guide. ADR required:yes; governing ADR backlog/decisions/219-console-chat-destinations-and-bounded-starts.md.

Independent task reviews approved the foundation and integration after two scoped fix rounds. Targeted evidence includes foundation100, planned Chat700, final native-start/switcher group163, migration/metadata/receipt and mounted Send/Stop checks, scoped Ruff, new-file formatting, formatter ratchets and whitespace checks. These overlapping selections are not summed. Canonical reports, exact full logs, real Console frames/receipts and rulings:Docs/superpowers/qa/2026-10-02-console-chat-starts/README.md. Real isolated Console/local provider verification covers all four destination/mode combinations, source custody and grants, draft edit/clear/restart, refusal/no replay, target Stop, saved status and successful manual recovery.

Qualifications:two direct project-budget mock fixtures fail identically at the exact pre-fix baseline because their lambdas reject reasoning_replay; streaming local usage remains unconfirmed and conservatively charged, with non-streaming confirmed settlement separately verified. Existing aggregate FD/escape diagnostics remain qualified. No full sweep or OS power-loss test was claimed. Blank-provider destinations use the codec's normal ABSENT snapshot; configured settings stay immutable. New integration cases consolidate several planned owner scenarios in test_console_chat_start.py, with existing owner suites used for regression coverage. The original dirty checkout remains untouched.

Final whole-branch review found three Important issues and one Minor notice gap. The single fix commit 9cb68f456e enforces exact available/unarchived destination before creation and native acceptance, captures the approved destination across asynchronous preparation, rechecks current runtime/bridge/owner eligibility before cutoff, discloses later supplied-body session authority and explicit instructions overrides, and carries the canonical Persona fallback notice through ordinary target ownership. The sole scoped re-review closed all four findings with no new breakage. Existing ADR-211 applies; no new ADR/schema/lifecycle owner was needed.

Final amended-owner selection: 140 passed precedes the last one-line capture refinement; final 14 capture/admission checks cover that refinement. Postcommit seven-file lint, new-three-file formatting, five inherited-debt ratchets and whitespace checks all passed. The broader targeted owner run (905 passed / 4 failed) had three fixture/schema failures resolved by covering reruns; the remaining fork None-returning fake reproduces at exact FIX_BASE, with 6,615 blobs independently verified and full test/store functions matching overall BASE. Its inherited failure and FD growth of 258 are explicit qualifications, alongside the earlier two reasoning_replay mocks and streaming usage uncertainty; no all-suite/resource-cleanup/power-loss claim. Actual boot11 verified complete approval disclosure and durable casual draft. All app/PTY clients closed normally and real config hash stayed unchanged. Reports, exact compressed logs/probes/baselines, terminal receipts and all rulings are preserved in Docs/superpowers/qa/2026-10-02-console-chat-starts/README.md. Implementation and reviews complete; branch retained for the user integration decision.

QA packaging also exposed ignored readable log copies. All 45 manifest-listed copies are explicitly tracked and all 141 artifact paths were verified against Git blobs before scratch cleanup. The observed incident and repeatable check are recorded in backlog/docs/lessons-backlog-hygiene.md.

Canonical ADR identity updated on 2026-10-04: current governance is ADR-219 (backlog/decisions/219-console-chat-destinations-and-bounded-starts.md), mechanically renamed from ADR-211 after confirming the distinct earlier PR2918 claim. Historical notes, source revisions and QA retain their original ADR number claims; the decision and runtime policy are unchanged.
<!-- SECTION:NOTES:END -->

Current-dev publication identity: 34215 replaces unmerged 33805. The landed TASK33802 census keeps its identity; the design/foundation/feature chain moved together to preserve dependency ordering. Historical QA retains original identifiers and bytes. Current evidence: Docs/superpowers/qa/2026-10-03-console-chat-starts-dev-integration/README.md.

Current-dev integration note: the historical implementation notes above describe the reviewed source revision and its original schemas. TASK34215.2 qualifies the integrated branch with AgentRuns22 and ChaChaNotes76, current native continuation/maintenance owners and new QA. Historical verification artifacts retain their original schema names and bytes.
