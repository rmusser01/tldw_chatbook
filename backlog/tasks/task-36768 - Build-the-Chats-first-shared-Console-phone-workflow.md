---
id: TASK-36768
title: Build the Chats-first shared Console phone workflow
status: Done
created_date: 2026-10-25 17:39
dependencies:
- TASK-36767
labels:
- console
- web-server
- mobile
- adr-244
priority: medium
documentation:
- Docs/superpowers/plans/2026-10-25-shared-console-mobile-supervision-implementation.md
- Docs/superpowers/specs/2026-10-25-shared-console-mobile-supervision-design.md
- backlog/decisions/244-shared-console-ownership-and-mobile-supervision.md
assignee:
- '@codex'
updated_date: 2026-11-01 13:13
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Give phone users a usable view of existing chats, decisions and individual agents with clear command targets and reconnect state.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The phone opens on shared Chats with workspace, activity and running/waiting/finished/unavailable state, then provides live transcript, expandable tool activity, decisions and agent details.
- [x] #2 Send, Steer, Redirect and Stop name the exact supported target; subagents expose guidance and Stop, and queued acceptance is distinguished from consumption or completed cancellation.
- [x] #3 Per-chat and per-worker drafts survive transient disconnects while the page remains open, newer edits survive late acknowledgements, and switching phone views never changes desktop selection.
- [x] #4 Disconnected or uncertain state disables mutation, reconnect refreshes authoritative state without replay, and expired authentication follows the existing sign-in flow.
- [x] #5 Existing plain assets, semantic browser CSS variables, visible focus and control states support phone touch, keyboard, contained long content and soft-keyboard viewport; no new framework, service worker or localStorage is added.
- [x] #6 Users can page finished-worker metadata and explicitly read bounded canonical result text without starting execution; historical rows never advertise live controls, and pending decisions are visible from published chat metadata.
- [x] #7 Transcript and receipt pages advance independently to a finite coherent completion without duplicated bodies or premature acknowledgement; stale content and repeated/two-reader continuations retain correct reset behavior.
- [x] #8 Exact live controls remain usable after fresh target confirmation while long/changing history loads; disconnect/reset invalidates controls until metadata is refreshed, and stale targets/drafts and duplicate replay remain guarded.
- [x] #9 Current admitted output remains visible during large/changing history backfill through a bounded, labelled, inert and target-scoped excerpt; coherent history replaces it without duplicate or stale output.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no new ADR
ADR path: backlog/decisions/244-shared-console-ownership-and-mobile-supervision.md
Reason: Implements approved phone supervision through existing native/canonical owners. Before-codeR149metadata/R150attention+workerreads/R151independentcompletion/R152targetconfirmation/R153boundedcurrentcontent clarify same contract; no schema/newowner/service/registry/authority.

1. Read task/spec/currentwire/ADR244/constitution+Impeccable Operate/craftfloor, contextonce; plain assets/Library/auth/manifest preserved.
2. LateACK RED; R149workspace/activity andR150pendingcounts/purefleet native metadata, no transcriptfanout/pruning observer.
3. R150readonly canonical headers20keyset+selectedresult<=2000cp via scoped native scalar/substrSQL+finiteworker, no full payloads; realSQLite/API scope/absence/privacy/no-write/paging.
4. R151validexisting EOF/keysetend cursorpair, both-nullACK, idempotenttwo-reader/retryboundedcache, metadatafirst; unequal3/2 bothways/savedEOF/staleReset native+browser.
5. R152fresh exactadmitted/published native target metadata confirmscontrols independentbulkbody; disconnect/reset invalidates, preserve receipts/duplicates/staledraft/exactStop safeguards and native revalidation.
6. R153realnative starvationRED then target latest_message activeassistant-or-eligible tail<=2000cp/8KiBinside64KiB before packing, no discoverybody/cache. Honest inert partialpreview until coherenthistory contains exactwindow; Unicode/stale guards, active>64K/toolordering/nonBMP/frameprogress/native4changingreads tests.
7. Remaining Chats/worker/decisions exactcontrols/historyreadonly/drafts/immutable clears/honestoutcomes; no reconnectbootstrap/reticket/replay/desktopselection. Incumbent responsiveCSS/focus/Back/status/safearea/inertcontent; unsupportedpending/Open terminal, no framework/SW/storage.
8. Targetedbrowser/page/API/projection/DB/node/static, bothengines320/390/landscape/shortheight/focus/longcontent; one finished capture/fixconfirmation+detectoronce/nohelpers, separate Task11realchild/physicalVPN. Selfreview/firstfailures/frozenepochs/engine+capturepacket/localexplicitcommits excludingrootbacklog; report+manifest then one rootcombinedreview. No fullsuite/deps/realprofile/cleanup/publication.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
R154 workflow ruling before recapture: confirmation36UI had35P1F (stale worker-row activation safelydisabled/no wrongmutation), so required nonvisual keyed-roster fix and green behavior precede ONE same20-state mechanical source-currency capture refresh. Explicit narrow exception to Impeccable two-polishing-round screenshot count for finalJS evidence; preserve first+confirmation packets/failure, validate newimages once only, no style/critique/detector/context/helper/captureloop. Existing approved code/isolated QA, no new userpermission or product/authority scope. Independent combined reviewer remains pending final report/source gate.
Implemented shared Chats-first phone supervision in local commits ff30c31f0ac092a5f3a4710ec453db36b10a7fb0 and 0560042ea684907f58b74880e5168d495c56c031, through existing canonical/native owners under ADR-244 (R149–153, no new schema/service/DB owner/authority). Exact main/worker commands, memory drafts and late-ACK revision clearing, no reconnect replay, supported decisions, bounded canonical worker history, independent snapshot/receipt completion, fresh admitted control confirmation and inert current-output excerpts are covered. Independent original combined spec/code/UI review and scoped FIX1 re-review completed: all four Important reconnect/saved-history findings addressed, no new Critical/Important breakage. Original native/API/browser final gate 142 passed and 36 capture tests; latest client-fix covering gate 107 passed/0 skips (48 real-browser cases across Chromium/WebKit plus59 page/API). Counts overlap. Full source/artifact/HEAD consistency independently verified. Static checks introduced no findings; eight unchanged Task10BASE full-Ruff findings remain qualified. Visual ship covers the20 original ff30-era captures; later functional-only fix includes browser focus/44px/containment assertions without relabelling captures. Minor F05 receipt-display identity ambiguity remains recorded for final feature triage. No full suite, real provider, actual retained-child browser journey or physical phone/VPN is claimed here; Task11 owns remaining integration qualification. ADR check: existing backlog/decisions/244-shared-console-ownership-and-mobile-supervision.md; relevant README semantic and control contracts updated.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Phone UI implementation and one combined functional correction round completed and independently re-reviewed. All nine ACs verified; 0 open Critical/Important findings. Latest targeted UI/page/API run107 passed with both engines, source-frozen and locally committed. Task11 real served-child/security/lifecycle qualification and physical-phone/VPN evidence remain separate; F05 minor and inherited static findings stay explicit.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
