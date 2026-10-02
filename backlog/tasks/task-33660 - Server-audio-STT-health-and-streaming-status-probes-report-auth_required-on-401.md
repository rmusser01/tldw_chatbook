---
id: TASK-33660
title: >-
  Server audio: STT health and streaming status probes report auth_required on
  401
status: Done
assignee:
  - '@codex'
created_date: '2026-10-01 23:30'
updated_date: '2026-10-02 18:15'
labels:
  - audio
  - server-parity
  - bug
dependencies: []
references:
  - tldw_chatbook/Audio_Services_Interop/server_audio_services_service.py
  - 'https://github.com/rmusser01/tldw_server/pull/3058'
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
tldw_server PR #3058 (merged 2026-10-01; tracked server-side as TASK-13416) now requires an authenticated user for `GET /api/v1/audio/transcriptions/health` and `GET /api/v1/audio/stream/status`. The server routes stay protected by the owner's decision.

Chatbook still lets a server-mode install run without an API token: the API client only sends credentials when it has them. Against an updated server, a tokenless install (or one whose token the server rejects) now gets HTTP 401 from both probes. The server audio service let the raw `AuthenticationError` escape, so the scope service and anything consuming it saw an unclassified exception rather than the typed refusal the same service already uses for its other server-owned denial (`admin_required`).

The two probes should report an explicit authentication-required state instead.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A server-mode STT health probe that the server answers with 401 raises a `PolicyDeniedError` with reason code `auth_required` and a user message that names authentication, both for the plain probe and for a warm-up request whose capability lookup is refused with 401.
- [x] #2 A server-mode audio streaming status probe that the server answers with 401 raises the same `auth_required` refusal.
- [x] #3 Existing refusals keep their meaning: 403 on a warm-up still reports `admin_required`, and other server errors still propagate unchanged.
- [x] #4 Tests cover the tokenless path for both probes at the API client (no credential header is sent; 401 maps to `AuthenticationError`), at the server audio service, and end to end through the scope service.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: N/A
Reason: routine bug fix that reuses the existing typed-refusal pattern (`PolicyDeniedError`) and the existing `auth_required` reason code; no new boundary or contract.

1. Write failing tests: client wire test (tokenless, 401) in `Tests/tldw_api/test_audio_client.py`; service mapping tests in `Tests/Audio_Services/test_server_audio_services_service.py`; scope end-to-end test with a real client on `httpx.MockTransport` in `Tests/Audio_Services/test_audio_services_scope_service.py`.
2. In `ServerAudioServicesService`, add an `_auth_required` refusal beside `_admin_required` and map `AuthenticationError` to it in `get_stt_health` (including the warm-up capability lookup) and `get_audio_streaming_status`.
3. Run the targeted audio test files and nearby suites.

2026-10-01 current-dev qualification: reuse existing PR #2954 and its server TASK-13416 cross-link, starting from dev 922440b93e83. Verify both probes through transport, service, scope and the existing destination recovery presenter. Reproduce typed-refusal tests failing with current-dev production source, then passing with the PR patch. Address Qodo findings with guaranteed client cleanup and streaming-status API documentation. Run targeted tests, Ruff, Bandit and diff checks; require fresh matching-head hosted review and gates for the updated PR. Existing ADR-178 (backlog/decisions/178-server-audio-diagnostic-admin-boundary.md) supplies the adjacent server-owned diagnostic refusal boundary; no new ADR or UI contract is needed.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Both probes now report `auth_required` instead of leaking a raw `AuthenticationError` when the server answers 401.

**Approach: map the 401 (option b), not skip tokenless probes (option a).** The API client already turns a 401 into `AuthenticationError`, and `ServerAudioServicesService` already turns server-owned denials into a typed `PolicyDeniedError` (`admin_required` for 403 on warm-up). A new `_auth_required` refusal beside it covers every way to reach the 401: no token at all, a bearer token set later, and a token the server rejects. A tokenless pre-check would only cover the first, and would also block older servers that still serve these probes without auth. The reason code reuses the existing `auth_required` vocabulary (`runtime_policy/server_context.py`, `Research_Workspace/server_adapter.py`).

**Scope.** Only `get_stt_health` (including the warm-up capability lookup, which a tokenless client also gets 401 from) and `get_audio_streaming_status` changed. The scope service needs no change: it already propagates `PolicyDeniedError` from the server service, the same way it surfaces `admin_required`. No screen calls these probes today; `app.audio_services_scope_service` is wired in `app_service_wiring.py` but has no UI consumer, so there is no user-visible copy to update.

**Tests** (written first; the service and scope tests failed on the old code with a raw `AuthenticationError`):
- `Tests/tldw_api/test_audio_client.py`: a tokenless client sends no `X-API-KEY`/`Authorization` to either probe, and a 401 raises `AuthenticationError`.
- `Tests/Audio_Services/test_server_audio_services_service.py`: 401 maps to `auth_required` for the plain STT probe, the warm-up capability lookup, and streaming status.
- `Tests/Audio_Services/test_audio_services_scope_service.py`: end to end, a real tokenless `TLDWAPIClient` on an `httpx.MockTransport` returning 401, through the scope service in server mode.

Existing `admin_required` (403) and pass-through (503) tests still pass, which covers AC#3.

**Verification (2026-10-01, branch `fix/tokenless-audio-probes-401` off `origin/dev` 31d4f9b764):** `pytest Tests/tldw_api/test_audio_client.py Tests/Audio_Services/ Tests/tldw_api/test_client_error_classification.py Tests/RuntimePolicy/test_unsupported_capabilities.py`: 76 passed. `ruff check` and `ruff format --diff` are clean on the touched files, and `./scripts/preflight.sh` passes. `Tests/UI/test_screen_navigation.py::test_app_wires_local_and_server_skills_services` fails locally with `RecoveryRequired: raw_source_selection_changed`. It fails the same way with the production change reverted, so it is local backup-recovery state, not this change.

**Modified files:** `tldw_chatbook/Audio_Services_Interop/server_audio_services_service.py` and the three test files above.

**2026-10-01 current-dev qualification (Codex):** Reused PR #2954 instead of creating a duplicate task/implementation. Replayed its existing patch onto verified current dev 922440b93e83 in a new managed worktree. Current dev already maps HTTP 401 to AuthenticationError and can classify that as server_auth_required, but neither probe has a screen/widget consumer; the app only wires the scope service. The existing PR supplies the missing typed-refusal boundary using PolicyDeniedError(auth_required), compatible with the existing destination recovery presenter. No new transport error layer or UI was added.

Resolved both original-head Qodo findings: document the streaming-status return/refusal contract and close both test HTTP clients in finally blocks. Parameterized the service/scope cases so each probe is independently exercised; scope coverage includes missing and rejected credentials and checks the existing Server sign-in required / Settings recovery copy. This presenter test is automated contract evidence, not a claim that a native screen currently calls either probe.

Negative control against unchanged current-dev production source: 7 expected AuthenticationError failures and 1 passing API-mapping control. Restoring the PR source: 85 passed in 4.37s across Tests/tldw_api/test_audio_client.py, Tests/Audio_Services/, Tests/tldw_api/test_client_error_classification.py, Tests/RuntimePolicy/test_unsupported_capabilities.py and Tests/tldw_api/test_client_redirect_credential_leak.py. Ruff lint and format, production-source Bandit (zero findings), diff checks and both Backlog guards pass. Independent read-only review found no actionable issues. Existing ADR-178 remains applicable; no new ADR is needed. Hosted review/gates for the forthcoming updated head remain pending, so this task is In Progress until qualification finishes.

Scope limits: no full local suite, paid provider request, real server request, microphone capture, extension setup or native permission workaround. The broader Buddy human-speech/audibility, native-interaction and historical reload-trigger work remains open. Prior UAT receipts retain their original attribution; their server checkout and the main checkouts were not edited.

**2026-10-01 integration qualification:** All hosted checks, including required Derived Artifacts, passed on 092ed65e1c30; matching-head Qodo reported zero bugs, rule violations and cross-repo conflicts, with both review threads resolved. Dev then advanced to ab4df9995954 through the config warm-path change. Rebase completed without conflicts or overlapping PR paths; both reviewed commits are patch-identical by range-diff. The same 85 targeted tests pass on the new base (2.83s); Ruff lint/format, production-source Bandit, diff checks and both Backlog guards pass (4759 tasks). Implementation acceptance criteria are complete and the task is Done. Fresh hosted review and gates must qualify the final integration head before normal merge; prior-head green results do not authorize bypassing those gates. Existing ADR-178 applies. This closes only the Chatbook audio-probe implementation, not broader Buddy UAT.

2026-10-02 current-dev and live qualification: rebased onto e92b01515f95. All three earlier patches are identical by range-diff, with no incoming audio/client changes; incoming conftest and dependency changes were checked. The focused scope passed 85 tests in 1.92s. Ruff lint/format, production-source Bandit (zero findings), diff checks and all derived-artifact preflight guards passed. The first Ruff attempt could not write the managed-worktree cache and is excluded; a writable private cache supplied the passing result. Pytest reported one managed-worktree cache warning and existing temporary-directory cleanup warnings; no full suite ran.

Against the real disposable server on source f21f1160dd0a (current backend dev df17c8ac3f), both STT health and streaming status returned HTTP 401 with each of missing and deliberately rejected API credentials. The actual production client, server service and scope converted all four refusals to auth_required, and the existing recovery presenter returned Server sign-in required with the Settings action. No HTTP transport was mocked. This is client/service/scope and presenter evidence; no native screen caller, microphone, voice, provider request or model warm-up was exercised. Receipt: /private/tmp/buddy-audio-live-uat-20261002/result.json. The two failed setup probes were blocked by the imported test bootstrap's socket guard; its documented numeric-loopback-only mode supplied the valid live result, retaining external socket denial. Both failed logs remain private and excluded. Only the owned API listener was stopped, its port closed, and the previous profile database remained unchanged.

Hosted green/Qodo-zero results retain source 6ed8f96c5851 attribution. Fresh matching-head gates/review are required after publishing this rebase. TASK-33660 remains Done for the implementation; wider Buddy human speech/audibility, native interaction and historical reload-trigger UAT remain open.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Both server audio probes return the existing auth_required refusal on HTTP 401, with independent missing/rejected-credential coverage and Settings recovery copy. Existing HTTP authentication mapping, server protection, 403 administrator handling and redirect safety are preserved. PR #2954 carries the implementation; 85 targeted tests pass after current-dev rebase. Broader Buddy UAT remains open.
<!-- SECTION:FINAL_SUMMARY:END -->
