---
id: TASK-33660
title: >-
  Server audio: STT health and streaming status probes report auth_required on
  401
status: Done
assignee: []
created_date: '2026-10-01 23:30'
updated_date: '2026-10-02 01:10'
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
<!-- SECTION:NOTES:END -->
