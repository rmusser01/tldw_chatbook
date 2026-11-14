---
id: TASK-40128
title: Handle the Sync v2 encryption policy during server enrollment
status: Done
assignee: []
created_date: 2026-11-11 11:51
labels:
- uat
- sync
- server
dependencies: []
priority: medium
updated_date: 2026-11-11 20:49
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Activating a server can switch Chatbook to the Server source and register the device, then leave Manual Sync blocked. The client sends `client_private_v1` for dataset enrollment. The companion server fixture accepted `server_trusted_v1` and returned HTTP 500 for `client_private_v1`.

### Reproduction

1. Run a disposable companion server with Sync v2 routes, a fresh SQLite sync store, and an authenticated test account. Configure its advertised Sync v2 encryption policy as `server_trusted_v1` with readiness enabled. The UAT used the companion server's production sync routes on loopback with fixture authentication.
2. On Chatbook local `origin/dev` `6bd97b112f` on Windows, open **Settings → Overview → Advanced / Diagnostics → Switch Source / Server** in a disposable profile.
3. Enter the fixture server root and token, then choose **Activate Server**.
4. Observe HTTP 200 for capabilities and device registration, followed by HTTP 500 with `sync_store_error` for `POST /api/v1/sync/datasets/enroll`.
5. Choose **Preview manual sync**. The panel reports **No Sync v2 server profile is configured**.
6. For a control, send two personal-dataset enrollment requests to the same fixture with the same `notes.note` domain and dry-run metadata. Change only `encryption_policy`. The `client_private_v1` request returns HTTP 500 and the `server_trusted_v1` request returns HTTP 200.

### Evidence

- [Activation sequence and controlled policy comparison](../../qa/uat-core-workflows-2026-11-11/sync-policy-control.txt)
- [Blocked Manual Sync preview](../../qa/uat-core-workflows-2026-11-11/sync-preview.txt)

At the tested revision, server activation calls `prepare_sync_v2_profile_mode()` without specifying a policy. The default in `Sync_Interop/sync_scope_service.py` is `client_private_v1`. This finding covers the loopback fixture. Deployed-server push and pull remain unverified.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [x] #1 Enrollment uses a policy supported by the server and approved for that sync profile. Any change in the data trust model requires an explicit user choice.
- [x] #2 When the client and server have no mutually usable policy, Settings explains the mismatch and keeps Manual Sync visibly unavailable.
- [x] #3 For a compatible, approved configuration, enrollment creates a sync profile and Preview manual sync becomes available.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: No
ADR path: ../decisions/008-sync-v2-client-m1-contract-alignment.md
Reason: Apply the existing server-trusted policy using explicit profile consent.

Cover capability and consent checks, preserve existing dataset trust and enrollment scope, surface unsupported policy in Settings, and rerun loopback activation and preview.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Settings requires explicit consent to enroll a server-readable dataset. Enrollment requires advertised and ready `server_trusted_v1` support, validates the returned dataset, and enables Preview without issuing content push or pull. Reactivation refreshes capabilities on the same account, device, and dataset while preserving trust, cursors, and error state. Compatible Notes Organization profiles retain their explicit preparation action and report when preparation is complete. The implementation follows [ADR-008](../decisions/008-sync-v2-client-m1-contract-alignment.md).

The focused enrollment suite passed seven cases, including the capability refresh regression. Three readiness and Settings pilot cases passed, along with focused control, keyring, and compact layout coverage. The adjacent TLS configuration failure reproduced on the unchanged base.

PR review tightened the Notes organization trust gate. A legacy retry identity carries no approval, and generic Run cannot dispatch setup for an enrollment-only profile. A failed enrollment keeps its reserved device ID for retry while leaving the profile unenrolled.

In the loopback rerun, capabilities, device registration, and dataset enrollment all returned HTTP 200. Preview showed an enrolled profile with no pending outgoing changes. The walkthrough restored Local source afterward. [Settings walkthrough](../../qa/uat-core-workflows-2026-11-11/sync-enrollment-final.txt) · [Loopback HTTP trace](../../qa/uat-core-workflows-2026-11-11/sync-enrollment-final-http.txt)
<!-- SECTION:NOTES:END -->
