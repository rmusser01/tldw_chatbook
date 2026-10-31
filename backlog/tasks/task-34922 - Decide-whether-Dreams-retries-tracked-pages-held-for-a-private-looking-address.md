---
id: TASK-34922
title: Decide whether Dreams retries tracked pages held for a private-looking address
status: Done
created_date: 2026-10-12 23:18
labels:
- dreams
- decision
priority: low
updated_date: 2026-10-14 00:42
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
When Dreams is switched on, a tracked page whose address resolves to a private or cloud-metadata address is held and never retried by the daily cycle; only a name that failed to resolve is retried. Both are answers from DNS, and a hotel or airport portal, or a VPN resolver, can answer every name with a private address, so switching Dreams on on such a network holds every tracked page until the user switches Dreams off and on again or re-tracks each story. A retry is the same public-address check, so it cannot enable a private address; it costs one lookup per held source on every retry, serially and with no backoff, and the retry runs on every cycle run while Dreams is on (the daily schedule, a startup catch-up, and each run the user starts), not once a day. This needs the owner's decision. Found in the independent review of PR #2921 (round 5) and re-checked on dev 7b38b26c69 after the merge, with real stores.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The owner's decision on retrying private and metadata holds is recorded
- [x] #2 The behaviour and the user guide match the decision
- [x] #3 A test pins the decided behaviour
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Owner decision 2026-10-13: retry private and metadata holds on later cycles.
2. Read the merged hold code on dev (track_service.py: _LASTING_HOLDS, _hold_may_pass, _resume_has_work, retry_unfinished_resume, the resume pass) and its tests.
3. Tests first: a private and a metadata hold clear on a later cycle once the name resolves publicly; while it still resolves privately the page stays held and no request is sent to it; a scheme hold stays lasting; update the T19 test to the new rule (a start still makes no lookup, a resume pass asks again).
4. Make every hold except scheme retryable: _LASTING_HOLDS = {scheme}; update the docstrings that state the old rule.
5. Update the guide's hold paragraph in Docs/User_Guide/dreams.md.
6. Run Tests/Dreams and Tests/Subscriptions; notes, ACs, Done.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Owner decision 2026-10-13: retry private and metadata holds on later cycles.

Plan step 3's "a start still makes no lookup" is superseded (review of #3101, R2): a start makes no lookup unless it has other reconcile work, as the first bullet below says. The plan text is left as written.

- track_service.py: _LASTING_HOLDS is now {scheme} only. A private or metadata answer is a DNS answer like a failed lookup, so _hold_may_pass/_resume_has_work treat it as passing: retry_unfinished_resume (the daily cycle) and any resume pass ask again, while a start makes no lookup unless it has other reconcile work (the hold is settled on, so it alone never makes the boot check ask for a pass; a start that runs a pass for another watch's unfinished record looks every passing hold up too, one lookup per held source). A retry is the same _public_url_refusal check, so a page that still resolves privately stays held. Docstrings that stated the old rule were updated.
- Tests (Tests/Dreams/test_track_service.py): a private (10.0.0.5, 192.168.1.1) and a metadata (169.254.169.254) hold clear on a later daily cycle once the name resolves publicly; a page that still resolves privately or to metadata stays held, its record stays (reason, on), and when the source is then run through the watcher's real check path (launch_run + execute_run) the fetch-time policy refuses it, last_error carries the policy text and no HTTP request is sent (httpx Client/AsyncClient send patched to record and raise; review of #3101, R1: the earlier assertion ran after the retry alone, which fetches nothing, so it could not see a request); a retry that changes nothing rewrites no record (review of #3101, N1); a scheme hold (ftp://, written to the database directly, since no app path reaches it) is not asked about daily. The T19 test became test_a_held_watch_is_settled_and_asked_again_by_a_pass_not_by_a_start. Seen red before the fix: 6 failed, 1 passed (the scheme test, already lasting).
- Docs/User_Guide/dreams.md: the Public addresses only paragraph now says Dreams asks again by itself at its next daily cycle, on switch-on or re-track, not on every start, including a private-looking answer; a still-private page stays off and nothing is requested; only a non-web address is not asked about again by itself.
- Verified on branch fix/task-34922-34926-dreams-holds-and-legacy-sources off dev 729f99d0ed, 2026-10-13: test_track_service + test_track_reconcile + test_cycle_service 338 passed.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
