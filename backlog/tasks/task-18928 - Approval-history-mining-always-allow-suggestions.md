---
id: TASK-18928
title: 'Approval history mining: "Always allow" suggestions surface'
status: In Progress
assignee:
- '@Robert'
created_date: 2026-08-19 09:55
updated_date: 2026-10-31 20:00
labels:
- console
- agents
- approvals
- mcp
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Port of hermes-agent's `approvals suggest` (2026-08-19 hermes-release review). Chatbook users repeatedly pick "Approve once" for the same safe MCP tools. Mine the user's own approval history into proposed "Always allow" entries, surfaced for explicit review in Settings (MCP permissions area): the user accepts or dismisses each proposal; accepting writes a real Always-allow entry through the existing permission store. Nothing is ever auto-allowed — the feature only drafts proposals, and runtime approval behavior does not change until the user accepts one. Proposals are definition-fingerprint aware, consistent with the existing "(definition changed)" badge semantics, and high-risk tool classes are excluded from suggestion.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A Settings surface lists suggestions supported by repeated explicit approved decisions for the current tool definition; legacy rows without a matching fingerprint, lossy/sentinel identities, and automatic session-grant reuse do not count
- [x] #2 Accepting a suggestion writes an explicit Always-allow entry to the selected permission profile through the existing profile digest/revision CAS API; a changed selection, selected or inherited policy, or displayed definition is refused, including policy races at the final writer
- [x] #3 Dismissing a suggestion is remembered (not re-proposed) and dismissals are reviewable/clearable from the same surface
- [x] #4 Tools in the existing high-risk classes are never suggested; the exclusion set is shared with the approval card's high-risk badge so the two cannot drift
- [x] #5 Tests prove no runtime approval behavior changes merely because suggestions exist (until one is accepted), plus mining, fingerprint, and dismissal-persistence coverage
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Preserve the existing proposal-only Settings surface and dismissal store. Correct the mining premise: MCPToolProvider records the first explicit approve_session as approved; approved-session means automatic reuse and must not count.
2. Add targeted failing reducer, execution-log, service, and workbench integration regressions for repeated explicit approvals, missing/mismatched historical fingerprints, selected-profile acceptance, stale profile CAS, and definition drift after display.
3. Add optional SHA-256 definition_hash metadata to the existing bounded execution log and its sanitizer. The explicit approval execution path supplies the reviewed HubTool fingerprint. No new store or schema version; old rows remain readable but cannot support suggestions.
4. Carry the selected PermissionProfileContext and displayed definition hash through recommendation rendering and accept messages. Re-resolve current eligibility in that profile, reject changed definitions, and call the existing permission writer with the captured profile digest/revision. Keep legacy report APIs callable; unqualified acceptance fails closed.
5. Preserve recovery guards, single-writer ownership, dismissal behavior, canonical Settings host, and literal notification output. Refresh publication must not install recommendations for a superseded profile.
6. Extend reviewed authority to the full inherited policy chain and compare it under the existing permission-store mutation fence. Reject lossy/sentinel audit identities before they can qualify a distinct catalog tool; retain the existing metadata-only sanitizer. Cover both review findings with native sanitizer/reducer and final-writer regressions.
7. Run targeted red/green tests with normal pytest options, static checks on changed code, and obtain a fresh independent review before completion. No full test sweep or release actions.
ADR required: no new ADR
ADR path: backlog/decisions/032-local-agent-tool-permission-boundary.md (approval-evidence addendum); backlog/decisions/079-workspace-assistant-defaults.md; backlog/decisions/240-mcp-destination-retirement-tools-hub-in-settings.md
Reason: extend the existing metadata log and preserve established explicit-consent, definition, profile-CAS, and Settings-host boundaries; no parallel authority or log store.
<!-- SECTION:PLAN:END -->
## Deferral note (2026-10-11, dev `9f8533686c`)

DEFERRED from the t2-hermes porting slice for capacity. Two sequencing facts
for the next session:

1. **The approval UX surface is under active rework**: tldw_chatbook PR
   #3036 ("Clarify Console permission approvals and decision feedback",
   draft) is open against the approval-card/decision-feedback test family —
   a Settings suggestions surface that reads approval semantics should land
   after that settles, per the slice's collision rule.
2. **The data-source decision is still open**: verify where Approve
   once/session decisions are durably recorded today (the permission
   store's audit trail vs the approval card flow) before designing the
   mining layer; the task's plan step 1 anticipates adding minimal durable
   history if none exists — that schema decision plus the Settings surface
   (MCP permissions area) plus the high-risk exclusion set shared with the
   approval card's badge is a full session on its own.

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
The Settings Permissions surface proposes reviewed Always-allow entries and persists dismissals independently of permission policy. Explicit acceptance uses the existing permission writer; no suggestion changes runtime authorization on its own.

2026-10-31 independent-review repair:
- Corrected the original mining premise: the first explicit session grant records approved; approved-session is automatic reuse and is excluded.
- Added optional canonical SHA-256 definition_hash metadata to the existing bounded audit log and sanitizer. The explicit approval execution path supplies the reviewed HubTool hash; legacy adapters remain executable without producing qualified history. Legacy/mismatched historical rows cannot support suggestions.
- Captured the selected PermissionProfileContext and displayed fingerprint in each acceptance request. Report resolution uses that profile; acceptance rejects changed definitions and forwards captured digest/revision to the existing store CAS writer. Stale selection and stale refresh publication are rejected.
- Preserved guarded dismissal storage, high-risk exclusions, Settings hosting, and the literal markup=False success notification.
- Updated the user guide, architecture reference, and ADR-032 approval-evidence addendum (also governed by ADR-079 and ADR-240). Recorded the producer-token incident in lessons-testing-evidence.md.

Targeted evidence: reducer/log/service/provider run had 139 passes and 6 POSIX skips; its one changed-contract test was corrected and a final focused run passed 23 tests. All five native workbench acceptance cases passed: named-profile write, changed selection, changed policy, changed displayed definition even with new successor approvals, and a policy change at the final write boundary. Three existing execution-log tests fail identically with the original 1b36436 module: steady append parsing (495), rotated append parsing (440), and Windows newline conversion in the same-size replacement fixture (1433 vs 1432 bytes). Three suggestion rendering/real-keyboard cases also passed. The full markup-interpolation guard and changed-range formatting checks passed. No new Ruff findings versus that head; six pre-existing findings remain. Full suite not run. Status remains In Progress pending fresh independent review and final verification disposition.

2026-10-31 second-review repair:
- The rendered profile context now includes a digest of the exact selected/fallback policy chain used by resolution. The existing final store writer compares it under its mutation fence in addition to selected-profile digest/revision CAS. Named-profile acceptance without captured chain authority fails closed; default-only legacy callers retain their sufficient selected-profile CAS. Unrelated profile changes and dismissals do not invalidate review.
- Audit sanitization drops qualification fingerprints when server/tool identities are lossy. The reducer independently refuses noncanonical/sentinel identities, including older invalid rows carrying a hash. Provider approval semantics are unchanged. ADR-032 and testing lessons record both reproduced failures.
- Red evidence: 11 targeted service/sanitizer/reducer regressions and 2 native inherited-policy cases failed as expected. Green evidence: full service/reducer files 44 passed; all 7 native acceptance cases passed; final compatibility/log/store controls 21 passed with one existing lock-probe fixture failure. That failure reproduces identically with the original e6119782 permission_store module: _ContestedLockProbe lacks acquire, and recovery admission raises before first store construction completes. No unrelated repair made.
- Syntax, changed-range formatting, git diff --check, and markup guard passed. Ruff has no new findings against e6119782 (10 existing findings in the currently touched files). No full sweep, provider-semantic change, remote action, or release. Status remains In Progress for fresh independent review.
2026-10-31 final backlog integration:
Rebased the five reviewed commits onto dev4d6764b132edde35c8a32b5d2c92ab189416dd6d; product patches are unchanged. Both appended conflict sections were preserved, including the landed Plugins lifecycle test. Independent integration review approved5810c2b21d15527120e099eb254802ffdb077e41 with3 focused tests passing (provider approval provenance, Plugins lifecycle ownership, inherited-policy refusal at final writer), zero failures/errors/skips; receipt is sibling test-runtime/integration-review-42ffd9b2f8ac4932a35412d83e7652b6/pytest.xml. Earlier limitations remain.
The full diagnostic rebuild identified exactly7 missing warnings:2 in unified_control_plane_service,5 in mcp_workbench. Statement review against the merge base confirmed fixed text plus exception type only, with no user text, paths, credentials or new sink. Canonical regeneration changed only those2 owner rows (26/2f3b52d782c628867557 and50/bbfdcc1a9205181fbafe); all other rows and sink/path projections equal the base. UI census passes1174/floor1174. Under the explicit size waiver, affected owner pins exactly match6054/1680/7011 lines. HostedCI waived by the user; no full sweep or release work.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->