---
id: TASK-35031
title: Served WebAuthn passkey login (Phase 2a, amends ADR-233)
status: Done
created_date: 2026-10-16 01:51
updated_date: 2026-10-16 03:06
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Spec 2026-10-15 §3.5: fido2 ceremonies, credential file store, /auth/passkeys routes, login+devices UI
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] WebAuthn registration/login ceremonies and the credential store are unit-verified in-process, including sign_count clone detection (7-test suite green)
- [x] The four /auth/passkeys routes enforce the correct posture: register/remove behind session + CSRF, the two login paths public and origin-checked, all answering 503 when passkeys are unavailable, responses no-store
- [x] /auth/login and /auth/devices gate the passkey UI on service availability (hidden-by-default buttons revealed by passkeys.js in WebAuthn-capable browsers; domain-origin notice otherwise)
- [x] A completed passkey login mints a persistent device session via the ADR-233 machinery (login_device; device label suffixed "(passkey)")
- [x] README passkey documentation added and ADR-235/ADR-233 amended (dependency corrected to fido2>=2,<3 per Ruling 5; ADR-233 Consequences cross-linked)
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
ADR: backlog/decisions/235-served-webauthn-passkey-login.md (amends backlog/decisions/233-served-web-session-persistence-and-device-pairing.md). Spec: Docs/superpowers/specs/2026-10-15-served-mode-auth-and-mobile-surface-design.md §3.5.

Approach per task (all on feat/served-passkeys-phase2a, TDD where code was written):

- Task 1 — ADR-235 authored and the fido2 dependency added to pyproject.toml's web extra (commit 3ea09ef071). Files: backlog/decisions/235-served-webauthn-passkey-login.md, backlog task file, pyproject.toml.
- Task 2 — WebAuthnCredentialStore (webauthn_store.py: atomic mkstemp+os.replace writes, mode 0600, fail-safe load, session_store discipline) and WebAuthnService (webauthn_service.py: Fido2Server over the derived RP, in-memory digest-keyed one-use 60 s TTL challenge states capped at 64, register/login ceremonies, list/remove), plus the 7-case unit suite with an in-process software authenticator (commit 81838af985). Re-review round mandated discoverable credentials: register_begin now passes resident_key_requirement=REQUIRED + user_verification=PREFERRED so roaming authenticators create credentials the empty-allow-list login can offer (commit 59a1fe9731).
- Task 3 — four+/auth/passkeys routes in serve.py (register/begin, register/complete behind session+CSRF; login/begin, login/complete public and origin-checked via _public_auth_paths; remove behind session+CSRF; all 503 {"error":"passkeys_unavailable"} when the service is absent; every response no-store), the WebAuthManager.login_device seam in Canvas/web_auth.py minting a persistent session with no token check, static/passkeys.js (feature-detected buttons, base64url handling, inline errors), login/devices page sections, create_server wiring over get_user_data_dir()/webauthn_credentials.json catching PasskeysUnavailable, and 7 new integration tests (commit 805ba4dfc6).
- Task 4 — README passkey paragraph, ADR-235 dependency text corrected, ADR-233 amendment cross-link, this close-out.

Deviations (recorded, all verified in task reports):

- Dependency rename per Ruling 5: the plan's "python-fido2 (>=1.1,<2)" does not exist on the index; both pyproject.toml occurrences were amended to "fido2>=2,<3" (verified-installed 2.2.1) in Task 2's commit. ADR-235's Decision text was corrected to match in Task 4, with a Status-adjacent amendment note.
- Resident-key fix (re-review): registration requires discoverable credentials (ResidentKeyRequirement.REQUIRED), otherwise roaming authenticators create server-side credentials that the discoverable-only login can never offer.
- login_device addition: WebAuthManager gained login_device(client_ip, device_label) — the login_with_access_token-equivalent without the token check — so passkey login reuses the full ADR-233 session machinery (persistence, cookies, capacity) rather than a bespoke grant.
- Availability gate is service presence (self._webauthn_service is None), not the vestigial .available property; PasskeysUnavailable is raised at construction (so live instances always have a usable RP) and create_server catches it to store None.
- The plan's "integration 11" baseline counted 11; the file collects 20 baseline tests (18 functions + one parametrized x3) — the operative requirement (name-identical baseline, additions only) holds: 16 passed / 11 failed with a byte-identical failure set (pre-existing ADR-126 recovery-fence environmental failures, reproduced on clean HEAD and in the main checkout).

Targeted verification matrix (Task 4, docs-only tree): webauthn_service 7 passed; web_auth integration 16 passed / 11 failed (failure set identical to baseline); Canvas web_auth 80 passed; session_store 5 passed; serve_cli_pair 3 passed; approval_constants 2 passed; mobile service 7 / api 23 / page 16 passed; web-server dependency gate failure set identical to baseline (4, environmental). No new failures.

Live-check procedure (operator; real platform authenticator cannot be automated here — the ceremonies themselves are covered by the in-process software-authenticator tests; this check validates platform integration only):

1. Serve with an HTTPS public_url naming a real domain (direct TLS or trusted HTTPS proxy).
2. From an authenticated browser, open /auth/devices and register a real platform passkey (phone/laptop) via "Add a passkey".
3. Sign out.
4. On /auth/login, click "Use passkey" and complete the ceremony — expect a redirect to / with a working app session.
5. Restart the served process and sign in again via "Use passkey" — confirms the credential persisted in webauthn_credentials.json across the restart.
6. On /auth/devices, Remove the credential and confirm "Use passkey" now fails (unknown credential).

Lessons check: nothing from this close-out generalizes beyond the task; the Ruling-5 rename evidence lives in the sdd progress log and the ADR-235 amendment note, and the pre-existing environmental failures were already documented in Task 3's report.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
