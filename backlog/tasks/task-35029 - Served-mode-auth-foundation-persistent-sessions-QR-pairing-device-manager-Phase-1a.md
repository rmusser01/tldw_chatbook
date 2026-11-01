---
id: TASK-35029
title: 'Served-mode auth foundation: persistent sessions, QR pairing, device manager
  (Phase 1a)'
status: Done
created_date: 2026-10-15 17:49
updated_date: 2026-10-15 19:34
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implements spec 2026-10-15 §3: persistent device sessions, /auth/devices, QR pairing, tldw-serve --pair
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Restarting the served process keeps phone sessions logged in
- [x] #2 QR pairing works from /auth/devices and --pair
- [x] #3 Logout and per-device/all-device revocation work
- [x] #4 Web_Server README incident response updated
- [x] #5 ADR-233 linked
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
ADR: backlog/decisions/233-served-web-session-persistence-and-device-pairing.md
Spec: Docs/superpowers/specs/2026-10-15-served-mode-auth-and-mobile-surface-design.md
Spike: Docs/superpowers/spikes/2026-10-15-mobile-engine-binding-spike.md (spec §4.3 gate — Binding 1 feasible as-is)

Approach (Tasks 1-8 of the Phase 1a plan, on `feat/served-auth-phase1a`):

- **Task 1** — implementation plan doc (`Docs/superpowers/plans/`, 29959d8527).
- **Task 2** — ADR-233 "served web session persistence and device pairing" (ce8f2d6dcd).
- **Task 3** — disk-backed session store (`tldw_chatbook/Web_Server/session_store.py`, 34ab850366) plus persistent wall-clock device sessions on login/remember-device (`tldw_chatbook/Canvas/web_auth.py`, `Web_Server/serve.py`, README incident-response refresh, 74a87c1714).
- **Task 4** — `/auth/devices` manager: list/rename, logout, per-device and all-device revocation (`Web_Server/serve.py`, `Tests/Web_Server/test_web_auth_integration.py` +5 tests, 1f7e50cfd2).
- **Task 5** — QR device pairing from `/auth/devices` plus the remember-device checkbox (`Canvas/web_auth.py`, `Tests/Canvas/test_web_auth.py`, 4fd71ddc1e).
- **Task 6** — `tldw-serve --pair` prints a one-use phone pairing QR (471c6fd42e).
- **Task 7** — spec §4.3 targeted refactor: `TOOL_DESCRIPTION_CAPTURE_CAP` moved out of the widget module into textual-free `tldw_chatbook/Chat/approval_constants.py`, removing one controller→widget import edge (`Tests/Chat/test_approval_constants.py`, ab6bc49759).
- **Task 8** — throwaway feasibility spike (probe at `/tmp/mobile_binding_spike.py`, not committed): constructed `ConsoleChatStore` + real `ConsoleAgentBridge` + `ConsoleChatController` App-free against a tmp SQLite DB with a duck-typed App context (`call_from_thread` + headless `UnifiedMCPControlPlaneService` + `LocalWorkspaceRegistryService`) and completed a full **agent-mode** round-trip — `fs_list` advertised, one approval round raised (`local:__local__`, reason "ask"), decided headlessly via `resolve_pending_approval(approve_once, round_id=…)`, executed against the bound folder, and the post-tool continuation persisted. Recommendation recorded for Phase 1b: **Binding 1 (parent-process MobileChatService) feasible as-is**, with conditions (native-tools-capable provider/model, workspace binding for fs_* tools, project instructions disabled for mobile v1, agent-capable fake gateway for tests). Transitive-import check: 98 textual modules still load via two non-widget edges (documented in the spike; data, not a blocker).

Verification: Phase 1a-targeted tests green (`Tests/Web_Server/test_web_auth_integration.py` +5 new pass; `test_session_store.py`, `Tests/Canvas/test_web_auth.py` 80 pass, `Tests/Chat/test_approval_constants.py` pass; 18 passed under the Phase 1a filter). The 11 failures in `test_web_auth_integration.py` are the documented pre-branch environmental baseline (identical failure set before/after — `Backup_Recovery` `RecoveryRequired` fence), as are the 209 in `Tests/Agents/test_local_tool_provider.py`.

No lessons-*.md entry: nothing generalizable beyond what the spike doc already records in place.
Final review fix wave: added Revoke-all route/page/JS/test (C1).
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
