---
id: TASK-16476
title: Console server adoption must not clobber a configured endpoint
status: Done
assignee:
  - '@Robert'
created_date: '2026-08-15 15:10'
labels: []
dependencies: []
---
## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
User report follow-up (2026-08-15, llama.cpp endpoint/config drift): `_apply_detected_local_server` (`UI/Screens/chat_screen.py`) persists `api_settings.<provider>.api_url` and `chat_defaults.provider`/`model` when the user adopts a discovered loopback server from the setup card. When the provider already has a DIFFERENT user-configured endpoint, adoption silently overwrites it — e.g. a user's `http://127.0.0.1:8080` replaced by the discovery default (note the defaults disagree: discovery probes `http://127.0.0.1:8080` first while the endpoint fallback default is `http://127.0.0.1:9099`, `Chat/local_server_discovery.py` vs `Chat/console_session_settings.py`). Discovery is loopback-only, so it can never see a LAN llama.cpp box; the persisted endpoint is the only record of it. Adoption also drops the detected endpoint from the session itself for llama.cpp (`base_url=None` after the provider-key check), so the config write is currently the only thing that makes "Use detected ..." effective at all — the fix must keep adoption effective for the session while protecting the persisted endpoint.

ADR required: no — preserving user-authored config at an existing write seam; the adoption contract's provider/model writes are unchanged.
<!-- SECTION:DESCRIPTION:END -->

## Implementation Plan (the how)

1. In `_apply_detected_local_server`: skip the `api_url` config write when the target provider already has a different non-empty persisted endpoint (fill only when absent), notify which endpoint was kept
2. Apply the detected base_url to the adopted session settings (replace the llama.cpp base_url=None drop with the detected URL) so adoption stays effective without the config write
3. Keep chat_defaults.provider/model writes unchanged
4. Red test exists (`test_detected_server_adoption_keeps_configured_endpoint`); run setup-card/discovery suites

ADR required: no
ADR path: N/A
Reason: preserving user-authored config at an existing write seam; adoption contract otherwise unchanged

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Adopting a detected server never silently overwrites an existing non-empty persisted endpoint for the target provider with a different value; the config write either omits the endpoint key or keeps the configured value (fills only when absent)
- [x] #2 Adoption still applies the detected endpoint to the active session for immediate use (session settings carry the detected base URL as a user-source selection), so "Use detected ..." remains effective without the config write
- [x] #3 `chat_defaults.provider`/`model` writes are unchanged (the affordance's stated contract: "Sets provider to X at host:port")
- [x] #4 When adoption skips the endpoint write because a different endpoint is configured, the user is told (one-line notice naming the kept configured endpoint)
- [x] #5 Regression test `test_detected_server_adoption_keeps_configured_endpoint` (added red in `Tests/UI/test_console_provider_persistence_regressions.py`) passes; existing setup-card/discovery suites stay green
- [x] #6 Persist-failure path unchanged: session-only apply with the existing "could not persist" warning
<!-- AC:END -->
## Implementation Notes

- `_apply_detected_local_server` (`UI/Screens/chat_screen.py`): the `api_url` config write now fills only when the provider has no configured endpoint; an existing DIFFERENT endpoint is kept and a warning notify names the kept endpoint. The detected base URL is applied to the adopted session settings (replacing the old llama.cpp `base_url=None` drop) so "Use detected ..." stays effective without the config write. `chat_defaults.provider`/`model` writes unchanged; persist-failure path unchanged.
- Modified files: `UI/Screens/chat_screen.py` (+ `safe_endpoint_display` import), `Tests/UI/test_console_provider_persistence_regressions.py` (regression test: configured 8080 kept, session base_url = detected 9099).
