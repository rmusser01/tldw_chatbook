---
id: TASK-35679
title: Move measured transcript memory presentation reads off the Console event loop
status: In Progress
assignee:
  - '@codex'
created_date: '2026-10-20 02:27'
updated_date: '2026-10-20 03:16'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Investigate slower Console attachment and input tails, remove the next measured synchronous native presentation read while preserving storage and branch-memory correctness, and retain all controlled measurements.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The transcript memory banner uses the existing finite presentation reader without main-thread native config or memory reads for qualified stock owners.
- [ ] #2 Memory selection, custom-reader compatibility, source and owner changes, cancellation and physical retirement retain their original safety behavior.
- [x] #3 Before and after native profiles retain all return/input/attachment timings, exact exits, source hashes and guard counts, including adverse results.
- [x] #4 Targeted verification and independent review are recorded; PR3166 is updated with auto-merge disabled.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no. ADR path: N/A. Reason: adopt the existing disposable context-presentation memo and finite native ownership under ADR126/230; no new cache, TTL, storage/schema, permission or service boundary. 1. Retain original tail samples and separate original callback/worker timing: transcript live-policy read568ms during leave, styling215ms before worst input, config/workspace waits on attach. 2. RED/GREEN actual transcript tests pin zero unused live-policy/native config entry and publication reopening its unchanged gate. 3. Reuse the already-wired finite context inputs for qualified original callbacks before the transcript gate, include effective-memory publication in that gate, and leave custom/partial callbacks synchronous. Do not await context completion or stall streaming. 4. Verify cold/held-worker streaming progress, memory banner parity, source/owner drift, original custom affinity and native cancellation/retirement. 5. Run serial preceding-head control/current ABBA with independent review, preserve adverse outcomes, update PR3166 with auto-merge off and remove exact owned fixtures.

### 2026-10-31 current-dev integration

ADR required: no.
ADR path: N/A; existing ADR-126 and ADR-230 apply.
Reason: rebase the reviewed display scheduling work onto current native admission, source ownership, config-lock and kernel callers without changing those contracts.

1. Rebase original PR head 34f88d6322ee2819d3e189e97eda5bc0e3f419f1 onto dev b1ca1b11fa44a8e0ff0a09dfbffe940072861ba4 in the isolated PR3166 worktree. Resolve overlapping implementation at current defining owners.
2. Preserve finite native retirement, custom-callback affinity, exact drafts and attachment authority, measured Current cost on unavailable tokenizers, conservative CSS matching and deferred hidden Home reads.
3. Run focused native Windows checks with genuine private config/temp roots and targeted static checks. Review any diagnostic inventory changes individually; do not weaken behavioral guards or timing bounds. Hosted CI and size limits are waived by the user's 2026-10-31 backlog instruction.
4. Record current-base results and unresolved historical performance/safety qualifications, then obtain fresh independent review before any root-agent publication or merge. AC2 and existing In Progress qualification remain open unless directly established.

### 2026-10-31 independent-review repairs

ADR required: no.
ADR path: N/A; existing ADR-126/230 apply.
Reason: restore the existing checked-read freshness deadline and qualify the existing cadence/tab-publication contracts without changing authority, TTLs or native retirement.

1. Reproduce the slow-tokenizer proof-age reset through the original native config reader and presentation entry. Preserve the checked read's original timestamp through tokenizer preparation; test expiry refusal and fresh-read recovery.
2. Preserve the already-landed TASK35645 and rename only the incoming native-attachment task to root-allocated TASK38267 with provenance and maintained reference updates.
3. Instrument the actual extra tail projection decision and publication source without replacing qualified production readers. Repair the responsible product or deterministic fixture contract, retaining refusal/replay and current timing bounds.
4. Qualify the three existing connected tab-publication paths with targeted functional controls; retain original failures and reconstructible tested-source patches. Run focused adjacent checks only, then obtain fresh independent review through root.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented qualified transcript reuse of existing finite context inputs with source-identity invalidation; original custom-reader cadence and storage checks remain. Product commit9220460203. 104 distinct targeted checks passed; three connected tab-publication controls remain unqualified (also fail at different stages on exact original sources), so AC2 and task status remain open. Serial ABBA and independent review recorded in MEMORY_RESULTS.md: no consistent overall speedup; median input177.02 to186.68ms, attachment1023.92 to1038.45ms; worst input203.79 to247.49ms, worst attachment1973.95 to1590.37ms. Original startup policy call479ms to zero updated; neither settled return trace calls it. PR3166 update prepared with auto-merge off. ADR required:no under existing ADR126/230; no storage/schema or authority change.

### 2026-10-31 integration notes

Rebased onto b1ca1b11 with current native source/owner/config-lock callers retained. Deferred Library preparation restores the existing app startup handoff; stale appearance display reuse now retains service identity; reconciliation diagnostics omit exception bodies. Current-dev fixture seams and reviewed diagnostic/size inventories were updated. Results and unresolved qualifications: `Docs/Development/evidence/pr3166-integration-2026-10-31/RESULTS.md`. Existing ADR-126/230 apply; AC2 and In Progress remain unchanged. Root coordinates fresh independent review and final current-base publication.
### 2026-10-31 independent-review handoff

Preserved the original authenticated config proof timestamp across optional tokenizer preparation; the actual native-proof regression failed before repair and the focused checked-display/tokenizer files passed 48 tests after it. Incoming native-attachment task 35645 was renumbered to 38267 with provenance, leaving landed 35645 intact. The exact six-case cadence selection passed with original-code decision traces; deterministic late-publication/refusal controls passed with the 51-case input suite. The historical 6-versus-5 tail failure remains unattributed, not fixed or attributed to baseline. All three connected tab-publication contracts remain unqualified on base b1ca1b11; the original tab tests are unchanged, and the bounded native-child trace identifies an original config reader returning a real proof after 10.641s beyond its ten-second setup bound. Root deferred their final qualification until current-base integration. See `Docs/Development/evidence/pr3166-review-2026-10-31/RESULTS.md` for raw failures, source patches, exact receipts and limits. ADR-126/230 apply; AC2 and In Progress remain unchanged.
### 2026-10-31 final integrated functional qualification

Rebased all 37 reviewed commits and the preview evidence commit onto actual PR3162 merge a0dbc1174c3dbd4d2732b0212786ec9e08bfc3b5 without conflicts, retaining its final archive runtime/card fixes. All three previously open connected tab phases now pass targeted author controls with original native readers/proofs, real Close/DOM, and exact physical retirement: after_close 28.04s, coalesced 31.40s, same_session_drain 34.27s. Fixture waits use existing completion events at unchanged bounds; the two Surface-lock phases explicitly finish the traced startup publication within their original 10s setup budget. No production change was needed for this final qualification. Raw REDs, exact tested-source patches, source-preservation audit and limits are in `Docs/Development/evidence/pr3166-final-integration-2026-10-31/RESULTS.md`. Historical cadence 6-versus-5 remains unattributed; preview static scope-census limits remain explicit. ADR-126/230 apply. AC2 and In Progress remain open pending root's fresh independent review; publication is still root-owned.

<!-- SECTION:NOTES:END -->
