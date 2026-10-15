---
id: TASK-35030
title: 'Served mobile companion surface at /m: chat, approvals, PWA (Phase 1b)'
status: Done
created_date: 2026-10-15 20:04
updated_date: 2026-10-15 23:25
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implements spec 2026-10-15 §4 per spike verdict (Binding 1): /m page, mobile API + WS, MobileChatService, PWA
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] `/m` serves a phone-first companion page (chat, approvals inbox, PWA manifest + icons) behind the served app's session auth, reachable via the login page's Phone view link
- [x] MobileChatService binds the parent process's Console engine per the spike: headless app context, approval capture via `set_pending_approval` and resolve via `resolve_pending_approval`, REVIEWED run hooks never fire, unreviewed/config-broken hooks refuse sends fail-closed
- [x] REST API under `/api/v1/mobile` (conversations list/new, message history with cursor, non-blocking send → 202, cancel, approvals list/decide) behind session + CSRF auth with the documented `{"error": ...}` shapes and text bounds
- [x] Stream WebSocket authenticates via the `csrf.<token>` + `chatbook-v1` subprotocols, delivers run frames in order, answers client pings, and closes 4401 on session revoke
- [x] `[web_server] mobile_surface` gate (default true): page, mobile assets/manifest, REST routes, and stream all 404 as unregistered when false
- [x] Desktop activity reaches the phone via completion-level polling (12 s + on focus); no service worker, no offline support
- [x] Default-workspace chats expose no `fs_*`/`git_*` folder tools; approval rounds mount only for the most recently activated session
- [x] `Web_Server/README.md` documents the surface, the config flag, and the v1 limits
- [x] Targeted suites green; regression failures name-identical to the documented environment baselines
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
ADR: backlog/decisions/234-served-mobile-companion-surface.md; spec `Docs/superpowers/specs/2026-10-15-served-mode-auth-and-mobile-surface-design.md` §4; spike `Docs/superpowers/spikes/2026-10-15-mobile-engine-binding-spike.md` (construction authority); plan `Docs/superpowers/plans/2026-10-15-served-mobile-phase1b.md`.

Approach, per task (all on branch `feat/served-mobile-phase1b`):

1. **ADR-234 + this task** (b7093cc5d3, fb7819db25) — decision record for the served mobile companion surface; Binding 1 (parent-process engine) per the spike verdict.
2. **Agent-capable provider gateway double** (ebd19eeebe) — a `ProviderToolCalls`-capable test double; the conftest `StreamingGateway` cannot drive agent mode (spike condition 5).
3. **MobileChatService** (a698376719) — `Web_Server/mobile_service.py`: lazy engine construction over `<user data dir>/mobile/` DBs, headless app context, approval capture/resolve, per-session send admission, exactly-once terminal events; no `ensure_run_hooks` wired; `ensure_hook_permissions` fails closed.
4. **REST API** (2966318065) — `Web_Server/mobile_api.py`: `/api/v1/mobile/*` behind `_web_auth_middleware`, CSRF on POSTs, non-blocking send → 202, `MobileSendRefused` → 409, text bounds 400/413.
5. **Stream WS** (aa8bb29baa) — same subprotocol challenge as the terminal WS; ordered run frames, ping/pong, 4401 on revoke, 1011 on engine-stream failure.
6. **/m page + PWA** (b495794484) — vanilla JS/CSS in `Web_Server/static/mobile/` (no CDN, no service worker), manifest + icons, login-page Phone view discovery link, `tldw-serve --pair` QR.
7. **Config + README + close-out** (this commit) — config gate was already covered by the Task 4/6 gate-off tests (`test_mobile_api.py::test_mobile_surface_false_leaves_the_routes_unregistered`, `test_mobile_page.py::test_mobile_surface_false_leaves_the_page_unregistered`), so no duplicate router-map assertion was added. README "Mobile companion surface" section + `mobile_surface` key in the config block.

**Three live-check fix rounds** (controller-run visual verification): (1) 7222054ec2 non-blocking send + pinned composer + header fixes; (2) c37f4e4ebf `[hidden]` display guard repairing mobile view switching; (3) ef9ec23fc8 header right-edge fit + visible divider. All three were verified visually by the controller on a live server.

**Task-7 verification fix (deviation from docs-only scope, flagged in the task-7 report):** the full-mobile-suite verification run exposed a pre-existing race from fix round 1 — `run_completed` was emitted (by the turn task inside the session lock, or by a pump tick) while the session still refused sends, and a pump tick in `submit_draft`'s return window silently baselined a fast run's content (zero deltas). Three of five `test_mobile_service.py` tests failed intermittently-to-consistently before the fix. Fixed in `mobile_service.py`: the owning turn releases its admission slot before announcing terminal (`_release_turn_slot`), the pump holds terminal announcements and treats slot-owned sessions as live for scanning while a turn owns the slot (`_turn_slot_owned`), and a first-observed terminal message delivers `message_done` with its content. 8/8 consecutive green runs after.

**Tests/evidence:** `test_mobile_service.py` 5, `test_mobile_api.py` 20, `test_mobile_page.py` 16, `test_session_store.py` 5, `test_serve_cli_pair.py` 3 — all passed; `Tests/Canvas/test_web_auth.py` 80 passed; `test_web_auth_integration.py` 11 failed/9 passed and `test_web_server_dependency_gate.py` 4 failed/2 passed — failure names diffed against the stashed pre-change tree: byte-identical (pre-existing `RecoveryRequired` environment baselines). Live round-trip evidence (controller-run, visual): full agent round trip send → approval → decision → streamed completion on /m across the three fix rounds.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
