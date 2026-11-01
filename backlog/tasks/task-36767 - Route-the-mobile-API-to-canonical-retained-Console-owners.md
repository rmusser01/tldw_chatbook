---
id: TASK-36767
title: Route the mobile API to canonical retained Console owners
status: Done
created_date: 2026-10-25 17:39
dependencies:
- TASK-36761
- TASK-36762
- TASK-36763
- TASK-36764
- TASK-36766
labels:
- console
- web-server
- mobile
- adr-244
priority: high
documentation:
- Docs/superpowers/plans/2026-10-25-shared-console-mobile-supervision-implementation.md
- Docs/superpowers/specs/2026-10-25-shared-console-mobile-supervision-design.md
- backlog/decisions/244-shared-console-ownership-and-mobile-supervision.md
assignee:
- '@codex'
updated_date: 2026-11-01 10:44
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Replace the separate mobile engine with authenticated access to existing served Console chats and permit shared mobile writes only after verified legacy cutover.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The mobile service no longer constructs a headless controller/store/bridge or selects mobile-only databases; phone-created chats, saved history and live controls use the retained canonical owner.
- [x] #2 Authenticated API requests route exact epochs, sessions, runs and decisions through the shared transport, preserve CSRF/origin/device/TLS policies and reconcile outcomes without offline replay.
- [x] #3 Saved conversation ownership is reserved before hydration and registered on persistence; unavailable native or served owners are shown honestly and are not resumed by a read.
- [x] #4 Canonical mobile writes require verified cutover, including verified no-data sources, while existing Library reading and supported sign-in/device flows remain available.
- [x] #5 API and stream failures, subscriber pressure, revoked sessions and stale protocol versions preserve the retained runtime and authoritative outcomes.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: backlog/decisions/244-shared-console-ownership-and-mobile-supervision.md
Reason: Implements the approved canonical retained-owner HTTP adapter, native create/open identity handling and nonmigrating reader boundaries; clarify the existing ADR before code if needed, without duplicating it.

1. Read actual Task9/spec/ADR/source and enumerate service constructors, routes, fakes, canonical reader and command/persistence callers; add a failing constructor-retirement/concurrent-client regression on isolated canonical stores.
2. Replace MobileChatService’s private engine/DB/provider graph with the admitted retained host, native projections/commands/tickets/receipts and real canonical saved-history/Library read seams that never create, migrate, reconcile or hydrate on a read.
3. Complete exact native create/open ownership reservation and saved-identity publication, require verified Task8 cutover before canonical mobile mutations, and refuse obsolete mutation envelopes before effects.
4. Wire strict public protocol1 snapshot/ticket/command/query routes and bounded authenticated stream; preserve session/CSRF/origin/TLS/revocation policies, distinguish unavailable/unknown outcomes and prohibit reconnect replay.
5. Update connected callers/fixtures/docs; run named targeted Task9 modules plus exact changed native-reader/command/control nodes, preserve native runtime coverage and first failures, apply scoped static/self-review and commit explicit source/test/doc/plan paths locally. Root owns Backlog metadata and independent Task9 review; no full suite or external publication.

R147 before-code clarification (existing ADR244, no new ADR): explicit authenticated/CSRF-protected POST /api/v1/mobile/console/host with strict protocol_version1 returns the exact retained host only after verified cutover and actual native execution admission. This bootstrap precedes separate exact-target Create/Open ticket issuance; it creates no chat/ticket/provider request. Repeated explicit bootstrap reuses the host, GETs and reconnect remain nonstarting, and replacement refuses without reticket/replay. Add focused bootstrap admission/auth/version/no-effect/concurrency tests; amend existing ADR244 and Superpowers Task9 before route code.

R148 validation clarification (ADR required: no new ADR; existing ADR244): after separately successful genuine public producer and same-target adapter, run one fresh complete permanent public pytest node on committed final fixture to close its still-unexecuted composition. This supersedes only the controller one-producer ceiling; original budgets/guards/no-real-profile limits remain. Preserve first failures/source epochs and prior reports; no unchanged broad reruns or recovery continuation. Full original57c77→finalHEAD review follows actual complete-node verification.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Before route code R147 recorded in SDD: explicit authenticated host-bootstrap POST obtains exact retained child identity after cutover+actual native execution admission; no ticket/chat/provider effects, nonstarting reads/reconnect and no hidden reticket. Existing ADR244/Task9 amended before code; full Task9 originalBASE57c77 and existing ACs unchanged. Source-grounded alternative avoids a manufactured stale-target request or general registry reservation interface.
Implementation22b664bd66201d264016fe94e0ae9d1b615bcc15 complete; root read full report. Before initial independent review, R148 closes reported validation gap: final permanent public pytest node never completed after its fixture corrections. One additional fresh complete isolated public node is released, superseding only controller one-producer ceiling (not budgets/guards/user scope). Prior genuine producerf707 and same-target adapter0 remain qualified/preserved; no recovery continuation. All AC/Done held pending complete-node result, artifact/source gate and full original57c77→finalHEAD review.
Full original57c77→22b independent review found T9-F01–F08 (3P1+5P2), no Critical. All eight enter one fix round1 with original implementer. Root verified connected source; cohesive native hydration/readiness/cleanup/cancellation/Goal/direct publication proposal precedes contract changes, then focused native regressions plus bounded-list/successor-discovery fixes. Existing ADR244 implements these ownership requirements; clarify before code if needed. No full suite or repeated broad cutover campaign. R148 permanent public gap is closed. AC/Done/Tasks10–11 remain held; final whole-feature fix wave unused.
Fix round1 committed6a3b7f834a +3d175e3809, complete review range22b664bd66→3d175e3809 (both commits). Root verified full fix report/raw126P at6a3b +12P admission follow-up at3d175 (distinct epochs), exact26-path source hashes matching each commit, all26currentHEAD files/22native AST nowarnings/indexclean/rootTaskonlydirt, ALL55fix+175current artifact hashes (overlap not summed). Static0introduced vsFIX22/original57; legacy debt precisely qualified. Fresh scoped independent review /root/rereview_task9_canonical_fix1 active on all eight findings plus fix-introduced breakage, readonly/nohelpers/noruntime. AC/Done/Tasks10–11 held pending verdict. Whole-feature final combined fix wave unused; no new ruling149.
Fix1 scoped independent review: seven original findings addressed; T9-F04 shared adoption remains open and T9-FIX1-N01 introduces first-save lost-publication race, both Important/P1. Root read full report and verified source anchors. ONE combined fix round2 with original implementer: common native adoption/Character custody and atomic staged observer/producer handoff, meaningful new native regressions, targeted static/source epochs, local commit then one scoped rereview3d175→fix2HEAD. Existing ADR244 applies; no new policy/ruling or approval needed. Exact prior report/manifests archived. AC/Done/Tasks10–11 remain held; final whole-feature combined fix wave unused.
Fix2 implementation6ab2357490b3ebe918c875e5d3c9a7e70c6c100a ROOT evidence gate accepted: FULL fix2 report/raw49P71.58/exactwrapper/static read, all7currentHEAD files/4nativeAST0warnings/indexempty/rootTaskonlydirt, all7before==afterepochhashes match6abcommit, ALL23fix+197current artifact hashes verified (overlap notsum). Scoped180FIX2=current vs181original diagnostics,0introduced/formatterratchet0, no blanketcleanlint claim. ONEfresh readonly/nohelpers/noruntime scoped review /root/rereview_task9_canonical_fix2 active onF04+N01+newFIX2breakage only,3d175→6ab. Seven old findings remain closed. Task9AC/Done/Tasks10–11 held; final wholefeature combinedfixwave unused/no newRuling149.
Completed canonical MobileChatService adapter and retired private mobile engine/DB selection. Explicit authenticated bootstrap/tickets/native receipts preserve exact targets and no replay; canonical Library/history reads remain nonstarting/read-only; verified cutover gates writes. Shared store owns hydration/readiness/publication and physical cancellation; native create/fork and staged Save/firstSend use atomic handoff, Character adoption and original cleanup custody are safe. Original full57c77→22b review plus scoped fix1→3d175/fix2→6ab235 are clean; all eight original findings and fix-introduced N01 closed, no parked items. Final49P71.58 on source6ab, prior126P6a3b/12P3d175 and R148public1P22b are distinct qualified epochs. Root verified all changed source/AST/HEAD, artifact hashes and native close evidence. Static0introduced/formatterratchet0; inherited lint debt and exact preTask9 inventory2fail carried to Task11/final, not blanketgreen. ADR required:no new ADR; existing244 clarified/linked before contract code and updated source-anchored docs. BackupRecovery obsolete engine status corrected. No fullsuite, actualAppService/browser, physicalphoneVPN, push/PR/merge or real-profile migration claimed.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Canonical mobile API now observes/controls the retained Console owner over canonical storage; the mobile execution/DB island is removed. Native ownership, adoption, producer handoff and cancellation fixes passed targeted actual-App tests and independent original plus scoped reviews. Final source6ab2357490:49 targeted tests passed, no introduced static debt; physical phone/VPN and complete real-child/browser qualification remain assigned to Tasks10–11.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
