---
id: TASK-33020
title: Qualify and document the complete agent archive recovery journey
status: In Progress
assignee:
- '@codex'
created_date: 2026-09-27 15:26
updated_date: 2026-10-31 18:49
labels: []
dependencies:
- TASK-33019
documentation:
- Docs/superpowers/plans/2026-09-27-agent-archive-recovery.md
- backlog/decisions/166-agent-assisted-archive-recovery.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Verify the integrated recovery journey through production paths and document its limits.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Targeted integrated tests prove lifecycle, permission, chronology, concurrency and cancellation requirements.
- [x] #2 A disposable-profile live app run demonstrates both entry points and original-chat opening without implicit sends.
- [x] #3 User guides, ADR links and scoped lint and UI governance evidence describe delivered behavior accurately.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR path: backlog/decisions/166-agent-assisted-archive-recovery.md
Reason: qualification and documentation of existing approved boundaries; no new architectural decision.
1. Read Task7 brief, implementation reports, qualification pointers and relevant evidence lessons; map every spec section to current production tests.
2. Close uncovered integrated lifecycle, provenance/destination, chronology, excluded-role and cancellation assertions with real stores and current native producers, preserving restore/Open distinction.
3. Run one final targeted integration selection exactly as brief; classify failures with existing per-node baseline proof or focused export comparisons; rerun only affected nodes after corrections.
4. Complete user guidance and factual QA report with commands, revision, limits, visual artifacts and scoped static/link checks.
5. Real-model walkthrough remains open while authorized local endpoint is unavailable; record attempted checks and leave AC2/task open instead of relabeling fake-model evidence.
6. Self-review and commit relevant qualification/tests/docs; independent review follows. No full sweep, push or merge.
7. Repair only the qualification-discovered chronology presentation omissions in archive_recovery_cards.py: retain both applicable dates independently of date-match basis and label unknown possible date matches. Spec section4 and existing AC1 bind this behavior; no new boundary/CSS. Revalidate affected mounted chronology/card tests and include the fix in Task7 review.
8. After the in-flight final selection settles, repair missing visible stable row ordinals and inactive-branch match provenance in the same row formatter under spec sections4/5. Add focused real-card/branch assertions and rerun affected nodes only; record the initial selection against its original tested bytes.
9. Repair demonstrated focused Enter refusal at all native recovery action Button sites using one local Enter/Space-owning Button pattern, preserving emission-bound confirmation identity and no screen/global binding change. Apply only cards and disposable archive_recovery adapter; focused Restore/Open RED and affected controls/compact-wide verification qualify final bytes.
10. Resolve Task7 review qualification findings with bounded exact-owner resource attribution/plateau evidence for retained runtime read/write and shutdown lifecycles, without a broad repeat, GC/threshold manipulation or aggregate attribution claims from small-run silence. Correct the contradictory Task4 baseline-proof paragraph. Live AC2 remains open independently.
11. Address the whole-branch review I1-I3 and M1-M3 in one fix wave under existing AC1/3: canonical target identity versus display title, recoverable receipt Refresh, app-owned closed/background-source completion notification, probe executor failure isolation, phased final hashes and migration EOF whitespace. Add focused real-store/native regressions, rerun affected tests only, update durable QA, then one scoped re-review. Preserve live AC2 and aggregate resource qualification limits; no full suite/push/merge.
12. Retry authorized localhost model qualification with actual disposable native Console and original stores. Repair the demonstrated same-screen original-ID resume handoff omission under existing AC2/spec section7 using a focused production navigation regression; revalidate only affected paths and independently review. Keep captured successor/refused retry outcomes, model discovery loops and prior aggregate resource limits explicit; supported Capture-Off qualification must be labeled as such. No full suite, push, merge or original-checkout edits.
13. Repair the demonstrated loss of earlier model-visible recovery result identities/order in a fresh ordinary follow-up under existing spec section5/AC2. Add only bounded model-observed selection metadata (set/ref/order/title/workspace label/valid scope; no excerpts/messages), require the original resolved destination to exactly match the currently admitted frozen destination and Library capability; exclude manual-only, expired, closed/wrong-session and imported results, preserve exact page/row identity and mark omissions/ambiguity. Do not replay arbitrary TOOL transcript rows or persist new authority. Establish production-path two-turn RED, verify affected authority/history/selection tests and independently review before any further live retry. Use existing deny-only hooks to block generic RAG before initialization in any later disposable no-index probe. Existing ADR-166 covers this direct corrective implementation; no new ADR.
2026-10-31 backlog integration: ADR required: no new ADR. Existing ADR166/097/228 and current portrait/binding authority ADRs govern. Rebase onto currentdev preserving native owners and exact installed predecessor schemas. Move new archive chronology migrations to the next unused versions (ChaChaNotes84-to85; Workspaces9-to10), preserve portrait regions and binding revision triggers, and qualify fresh/upgraded/recovered stores plus focused archive authority/confirmation/cancellation/Open paths. Resolve source conflicts against current moved owners, regenerate only reviewed diagnostics/CSS, record exact size waivers, and obtain full independent requesting-code-review. User explicitly waives hostedCI and size gates for oldest-first backlog integration. No new live endpoint requests; prior30/30 cap stays exhausted. No release or actions on excluded3267/3297.
<!-- SECTION:PLAN:END -->
## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Qualified the complete native archive-recovery journey under ADR166/147: local archive search, exact target/scope confirmation through result-button and conversational entry points, truthful partial outcomes, and explicit original-ID Open. The delivered UI preserves chronology/numbered provenance and keyboard controls. Final reviewed repairs preserve canonical target identity, recoverable Refresh and app-owned completion after source closure. Updated three Console guides and reproducible scoped QA/artifacts.

Live qualification found two concrete omissions and corrected them through existing seams. Same-screen Open now schedules the existing guarded resume consumer. Fresh ordinary follow-ups retain bounded previously model-observed selection metadata to the exact same admitted destination, under current frozen Library/native/Persona/origin/kill/lifetime gates. No excerpts/history, broad TOOL replay, new schema/dependency or persistent journal. Next Send display/estimate omission is explicitly accepted; actual outgoing fitting/counting is tested.

Both entry points are natively qualified in ordinary Capture-Off. Earlier Direct/RAG searches and manual chat-only/workspace-only/both scopes are supplemented by a final real local-Gemma RAG-only search→“Restore the second one.”→exact Linden/Atlas workspace-only confirmation→explicit Open. Only Atlas restored once; original IDs/message versions/parents/active leaves and both unsent drafts survived. Four HTTP200/SSE-complete requests, two completed primary runs, no new request/run from Open. Normal-profile fingerprints match, no index/embedding downloads/remote inference/server changes, native Quit completed and model slot idle. Independent source and factual evidence reviews approved spec/quality with no actionable findings.

Evidence remains phased: original22-file selection486passed/12failed/7warnings, with recorded baseline classification; review-fix80affected+7finalfocused passes; sharedOpen31affected passes; follow-up28distinct new cases qualified across phases and63affected passes/no warnings. Runtime/result-store/new-test static checks pass; controller191baseline diagnostic signatures/format debt unchanged with no inserted-block diagnostic/edit. Scoped whitespace and144URL-decoded local Markdown targets verified. Historical aggregate+729UNATTRIBUTED and Capture-On successor/recovery/provider-cause limits remain unqualified. Task completion covers the accepted bounded behavior, not a fully green suite or general merge readiness. No full sweep/push/PR/merge/original-checkout edits/worktree cleanup.

Core changed files: recovery service/contracts/coordinator/runtime/result store; Console owner/shared navigation seam and native adapter/cards; focused production-path DB/agent/Console/UI tests and migration; Console guides, plan and durable QA/artifacts. ADR required: existing backlog/decisions/166-agent-assisted-archive-recovery.md (linked with ADR147); no duplicate ADR. Final source correction cd068fadb3 on codex/agent-archive-recovery. Current evidence: Docs/superpowers/qa/console/2026-09-27-agent-archive-recovery-live.md#approved-follow-up-implementation-and-native-qualification. Historical automated evidence: Docs/superpowers/qa/console/2026-09-27-agent-archive-recovery.md.
<!-- SECTION:NOTES:END -->

## Filing provenance

Created with Backlog CLI in a temporary Git workspace using the project configuration, after the production CLI repeatedly scanned local branch history. Copied into this project with a unique ID above the refreshed all-ref/worktree ceiling and current local task IDs; dependencies and contents were verified.
