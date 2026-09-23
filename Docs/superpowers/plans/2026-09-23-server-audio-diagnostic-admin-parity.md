# Server audio diagnostic admin parity — TASK-32917

ADR required: yes
ADR path: backlog/decisions/178-server-audio-diagnostic-admin-boundary.md
Reason: This aligns a connected-client security and service contract with the server's administrator boundary.

## Stage 1: Contract tests
**Goal:** Demonstrate passive health, admin-only dispatch, and explicit denial through connected and scope services.
**Success Criteria:** Targeted tests fail on the current behavior.
**Tests:** `Tests/Audio_Services/test_server_audio_services_service.py`, `test_audio_services_scope_service.py`.
**Status:** Complete

## Stage 2: Service gate
**Goal:** Check the server's effective audio diagnostic capability using the same client and translate 403s.
**Success Criteria:** Explicitly denied calls never dispatch; admin and wildcard-authorized calls dispatch; passive health stays available.
**Tests:** The targeted tests from Stage 1.
**Status:** Complete

## Stage 3: Verify and publish
**Goal:** Verify the touched scope, document the contract, and open the Chatbook PR.
**Success Criteria:** Targeted tests and style checks pass; PR links the server review finding.
**Tests:** Targeted pytest, compile/style checks, `git diff --check`.
**Status:** Complete

The first pass checked only `user.role` and was rejected in Qodo review because the server also honors wildcard permission. The server's existing current-user capabilities route now exposes the effective diagnostic decision, and the Chatbook client reads that decision. Revised verification: 22 focused audio and API client tests passed; Ruff, format, compileall, and diff checks passed on touched small files. The large API client and lazy-export module retain pre-existing lint findings and were checked by targeted tests and compileall. An existing `from_config` test remains red in the isolated checkout with `RecoveryRequired(raw_source_selection_changed)` during configuration bootstrap, before the changed diagnostic methods are called.
