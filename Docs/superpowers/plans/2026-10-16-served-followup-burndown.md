# Served-Mode Follow-up Burn-down Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superposing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close every deferred follow-up recorded across the served-mode PR review threads (#3127/#3128/#3130/#3138), headlined by mobile conversations surviving server restarts.

**Architecture:** Six small, independently-testable batches on one branch off `dev`. No new ADRs — all items are recorded follow-ups of ADR-233/234/235/236 surfaces; behavior deltas are within each ADR's recorded follow-up notes (rehydration explicitly recorded in ADR-234's amendment + README).

**Tech Stack:** existing only.

**Spec:** the review threads themselves (linked per task); ADRs 233-236.
Branch: `chore/served-followup-burndown` off `origin/dev`. Worktree: /Users/macbook-dev/Documents/GitHub/tldw-auth-phase1a (venv .venv).

## Global Constraints

- Every item below is a RECORDED follow-up from a posted review — do not expand scope beyond the listed fixes.
- TDD; targeted pytest only (`.venv/bin/pytest`); the standard environmental baselines are name-identical sets (11 integration + 4 dep-gate + 209 Agents + 3 gateway/spawn RecoveryRequired); green sets must stay green (Canvas web_auth 83, session_store 7, cli_pair 3, approval_constants 2, webauthn 7, mobile service 21 / api 27 / page 19+, control protocol 63, relay publisher 15, ownership pin 83).
- If any fix turns out to conflict with an ADR or need a design decision beyond the recorded follow-up, STOP that item and report it back (do not improvise architecture).
- One commit per task; message prefixes per repo convention.

---

### Task 1: Mobile conversation rehydration (the headline follow-up)

Recorded in: ADR-234 amendment + README v1-limits ("rehydration is a recorded follow-up"); #3128 review Important 2.

**Files:** Modify `tldw_chatbook/Web_Server/mobile_service.py`, `Web_Server/README.md`; Test `Tests/Web_Server/test_mobile_service.py`.

**Requirements:**
- On engine build (or first `sessions`/`session_messages`/`send` access), the service lists persisted conversations from the MOBILE chat DB (isolated data dir — every conversation in it is mobile's) via the store's persistence seam and exposes them in `sessions` (id = the persisted conversation/session identity the store's restore path uses, title, workspace_id if derivable, `has_folder_tools` by workspace) WITHOUT eagerly hydrating trees.
- Accessing an unknown session id (`session_messages`/`send`) attempts a lazy restore via `ConsoleChatStore.restore_persisted_session` — research its existing call sites (TUI resume flow) for the hydration-data builder (`ConsoleConversationHydrationData` / all_nodes construction) and reuse exactly that machinery; unknown-after-restore still raises/404.
- After a service restart over the same `data_dir`: `sessions` lists earlier mobile conversations; `session_messages` returns their history; a `send` into a restored session works (full engine round-trip on a restored session in the test).
- README: replace the per-server-run limitation bullet with the new behavior (keep the note that pre-existing rows were written under the same schema).
- Tests: restart scenario (create session + turn → shutdown → new service over same data_dir → sessions listed, messages readable, one send completes).

### Task 2: Mobile service efficiency (three #3128 minors)

**Files:** Modify `mobile_service.py`, `mobile_api.py`; tests both files'.

- **Fingerprint-gated polling:** `session_messages` and the 0.5 s pump use the store's O(1) `session_fingerprint` seam (research its exact API at Chat/console_chat_store.py — grep `def session_fingerprint`) to skip unchanged sessions instead of materializing full transcripts (`messages_for_session`) every tick/12 s poll. Behavior identical; add a test asserting a fingerprint-unchanged session is not re-materialized (spy/counter on the heavy call).
- **Pump scoping:** the pump scans only sessions with an active run OR a turn slot OR pending tracks (no forever-scan of idle sessions). Keep the docstring honest (fix the stale forever-scan wording).
- **Conversations pagination:** `GET /api/v1/mobile/conversations` gains `?page=`/`per_page` (default per_page 50, max 100, same envelope `{"items","page","has_more"}` as the reading routes; default behavior unchanged shape-wise — items + page + has_more added) — mobile sessions list is small, but the route contract should match the reading routes. Update the app.js caller to handle the envelope (it can ignore pagination fields).

### Task 3: Mobile client + stream polish (three #3128/2c minors)

**Files:** Modify `Web_Server/mobile_api.py`, `static/mobile/app.js`; tests.

- **Truncated-bubble resync:** on merge, when a known message id arrives with LONGER content than the rendered row, update the row (fixes mid-stream reconnect truncation). Test at the JS-logic level is not possible — pin via the existing JS-marker test pattern (fragment assertions) + a service-level test if the merge logic has a server-visible equivalent (it does not — JS-marker test only, documented).
- **1011 on service shutdown:** `handle_mobile_stream` closes with `CLOSE_ENGINE_ERROR` (1011) when the service is shut down / `_ensure_started` raises (currently a normal close). Test: existing shutdown-close test extended to assert the close code.
- **Surface-label fallback:** approval cards label any non-"desktop" surface explicitly ("phone" only when surface==="mobile"; unknown values render the raw value) — app.js one-liner + marker test.

### Task 4: Auth/API small-hardening batch (#3127 minors)

**Files:** Modify `Canvas/web_auth.py`, `Web_Server/serve.py`, `Web_Server/mobile_api.py`; tests.

- **Prune write batching:** `_revoke_session` no longer calls `_persist_all()` unconditionally inside prune/eviction loops — batch one write after the loop (add `_revoke_session(..., persist=True)` param or a context; grant/rename/revoke-API paths keep immediate writes). Test: N expired sessions pruned → exactly one store write (spy on save).
- **CSRF-denial pin:** one integration test asserting 401 on `POST /auth/devices/revoke` (or pair) WITHOUT the CSRF header with a valid session.
- **Config error key names:** the day-key parsing in `create_server` raises ValueError naming the offending key (extend the existing absolute<idle message family). Test: monkeypatched bad value → message contains the key name.
- **413 uniform shape:** `mobile_api._read_mobile_json`'s aiohttp-over-budget path answers the uniform `{"error":"body_too_large"}` 413 (catch HTTPRequestEntityTooLarge at the route boundary — follow whatever the 1b final-review sketch prescribed: `_error(413, "body_too_large")`). Test exists from 2a? no — that was webauthn-adjacent; add the mobile-api one.
- **--pair bind-order hint:** `_print_pair_qr` prints a one-line "starting server…" hint before the QR (scan-during-startup confusion, #3127 minor 8). Trivial test extension.

### Task 5: Passkey hardening batch (#3130 + 2a-final minors)

**Files:** Modify `Web_Server/webauthn_service.py`, `webauthn_store.py`, `serve.py`, `Tests/Web_Server/test_webauthn_service.py`, `test_web_auth_integration.py`.

- **Attestation body cap:** register/complete route accepts a larger body (32 KiB) than the 4 KiB device-JSON default (packed x5c chains) — parameterize the body guard; test with an oversized-but-valid-shaped body (builder from existing helpers).
- **clientData pre-check:** `login_complete` parses clientDataJSON once, checks challenge membership + origin + type against the state BEFORE the credential loop (O(1) rejection of wrong-origin/wrong-challenge attempts instead of per-credential ValueError loops); unknown credential ids still 401. Tests: wrong-origin assertion fails fast (no credential iteration — spy) and existing tests stay green.
- **sign_count type guard:** `webauthn_store.load()` skips records with non-int/bool sign_count (mirror session_store's type validation). Test: tampered file with "sign_count": "soon" → record skipped, login fails clean.
- **Challenge-cap eviction test:** pin the 64-cap oldest-first eviction (fill 65 challenges via login_begin with injectable factory, assert the first is gone).
- **skipif gate:** `test_webauthn_service.py` gains the `check_web_server_available` skipif marker (matches the integration file).
- **Scheme refusal (2a F3):** `_derive_rp_host` refuses non-https schemes at construction (http://ftp:// → PasskeysUnavailable) + test.

### Task 6: Relay + broker polish (#3138/2c minors)

**Files:** Modify `Web_Server/mobile_service.py`, `Web_Server/served_relay_publisher.py`, `Canvas/control_protocol.py`, `Web_Server/README.md`; tests.

- **(child_id, round_id) keying:** `_desktop_approvals` keyed by the tuple; cross-child round_id collisions can no longer shadow/steal (2c final-review optional hardening). Tests: two children same round_id → both listed, decide routes to the right one.
- **Direct resolver-mirror test:** a test exercising app.py's `_served_relay_decide` against a REAL controller instance (not the stub mirror) — the 2c review's "rename would silently degrade" gap.
- **Publish-return gating:** the served-child wrapper publishes `approval_raised` only when the original callback's return is truthy (mounted) — the T2 minor edge. Test: original returns False → no publish.
- **README clause:** drop the loose "the same lifecycle as the desktop approval card" equivalence (keep the operative sentences).
- **Broker aclose deadlock fix (real bug, verified by the 2c review):** `CanvasControlBroker.aclose()` revokes children (closing their writers) BEFORE `await server.wait_closed()` so blocked `_read_message` loops unblock — the E2E test's forced ordering becomes unnecessary. Tests: broker.aclose() completes without the client closing first (with a timeout).

---

## Self-Review

**Coverage:** every item traces to a posted review thread or an ADR-recorded follow-up — rehydration (ADR-234/README), efficiency+client+stream (#3128 minors 3-7), auth batch (#3127 minors 4-8), passkey batch (#3130 minors 1/3/5 + 2a F2/F3/F5/F9), relay batch (2c final minors + T2/T4 deferred). Deliberately dropped as not-worth-it (recorded here): 91-char import line; import-time HTML read (existing idiom); client_max_size note (documented); raw index.html public-but-inert (documented); task-file stale hash; empty Final Summary sections; fido2 INFO log level (cross-cutting, informational identifiers); invalid_title/workspace direct tests (isinstance twins); reconnect-truncated-bubble full JS test coverage (marker-pinned only). Type consistency: no cross-task interfaces — each batch is self-contained; Task 2's envelope fields match the reading routes' existing `{"items","page","has_more"}`.
