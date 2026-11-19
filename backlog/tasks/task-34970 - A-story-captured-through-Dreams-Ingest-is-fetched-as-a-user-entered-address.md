---
id: TASK-34970
title: A story captured through Dreams Ingest is fetched as a user-entered address
status: In Progress
assignee: []
created_date: '2026-10-12 01:45'
updated_date: '2026-10-12 19:39'
labels:
  - dreams
  - security
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Dreams refuses to ingest a story whose address is not a public destination, and that check at the door is the only guard on any fetch of the capture. The capture service fetches every capture as an address the user typed, so the address's own host counts as trusted. That holds for the first fetch, which the service starts itself as soon as the capture is saved, and for every retry made later from the Library. A story address that was public at the door and resolves to a private address by the time of a fetch is then fetched, on the first attempt as on a retry. Closing it means recording where a capture's address came from on the capture item, which is a schema step on the Library collections store, a store covered by backup and recovery. Found in the review of PR 2921 and left out of that change on purpose.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A capture whose address came from a Dreams story is fetched with no trusted origin on every attempt: the first fetch started when it is saved, every retry, and every redirect hop of each
- [ ] #2 A capture whose address the user typed keeps today's behaviour, including private addresses the user chose
- [ ] #3 An existing collections store upgrades in place and its recovery catalog entries are updated in the same change
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Historical root preparation before the second author update:
Persist actual capture source authority through versioned Collections records and every fetch/redirect/Library Retry; conservative unknown migration, explicit user intranet positive. Update/amend existing ADR207/126 before code, exact predecessor rollback/reopen/future refusal and installed recovery catalogs/participants. No new ADR ID.

ADR required: yes, amend existing207 egress provenance and126 recovery before code.
Persist conservative actual source authority through CaptureSaveRequest and versioned Collections, the initial automatically started extraction, all retries and every redirect. Preserve explicit user intranet intent. Qualify populated predecessor upgrades, rollback/reopen/refusal and exact installed recovery catalogs/participants.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

Historical root integration preparation:
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

Source author follow-up admitted into the user-authorized complete PR2921 review/remediation. Historical source description/notes retained. Current integration plan targets frozen updated author6c074d6eed/v6; actual post-Buddy dev and fresh author lease remain required before implementation. No product change or completion claimed.

Current source update: author0c941ba2ec984d960ba83376aeace0e830452bbc, 113 nonmerge commits/88 feature paths after actual merged Goal9595facbbb. Independent source review preserves the real shared Toast fix and stronger reconcile/suspension work; remaining integrated proof and actual post-Buddy/latest-dev publication gate are pending. Prior source/plan/qualification claims remain historical. No product change or completion is claimed here.

PR2921 current private integration qualification (2026-10-12):
Persisted typed CaptureSaveRequest authority in Collections v5 and passed it through actual initial extraction, Library Retry and every redirect. Generic/serialized authority is UNKNOWN; explicit user input retains authorized intranet behavior. Populated exact v4 runtime/installed upgrade, injected rollback, reopen, refusal, capture/relocation and source-byte proof pass in final 33-case checkpoint. ADR-207/126 amendments govern the change. Existing shared DNS address-pinning limitation remains disclosed.
Full source-bound outcomes, original failures, native owners and tradeoffs: .superpowers/sdd/2026-10-11-pr2921-current-dev-integration/task-1-report.md; external pr-review-wave2/pr2921-current-integration. Local implementation is ready for independent review. Actual-base restack, complete independent review and hosted required gates remain pending; preserve In Progress and the existing acceptance checklist until the full DoD is met.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
<!-- SECTION:FINAL_SUMMARY:BEGIN -->
<!-- SECTION:FINAL_SUMMARY:END -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Renumbering provenance

Renumbered from task-34802 to TASK-34970 (2026-10-13): this ID collided
with a concurrently filed task after the branches merged; per the
2026-08-21 owner rule (TASK-19601) the older arrival kept the id.
