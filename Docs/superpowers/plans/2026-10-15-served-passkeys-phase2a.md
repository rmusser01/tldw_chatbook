# Served Passkey Authentication (Phase 2a) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Passkey (WebAuthn) login for served mode on stable HTTPS-domain origins — register a passkey from an authenticated `/auth/devices`, then sign in from the phone with a biometric tap instead of typing anything; QR/token remain the fallback for every origin passkeys cannot serve.

**Architecture:** A `WebAuthnService` in `Web_Server/` wraps `fido2.server.Fido2Server` (RP ID derived from `public_url`'s host; discoverable credentials, "none" attestation, UV preferred) with in-memory one-use challenges (60 s TTL, mirroring the bootstrap-nonce pattern) and a file-backed credential store in the `web_sessions.json` idiom. Four thin routes in `serve.py` (`/auth/passkeys/register/{begin,complete}` authed + CSRF; `/auth/passkeys/login/{begin,complete}` public with origin checks). Login-complete mints a persistent device session through the existing `WebAuthManager` — no new admission surface. `static/chatbook/passkeys.js` drives `navigator.credentials` on the login and devices pages.

**Tech Stack:** `python-fido2 >=1.1,<2` (new, `web` extra only; pulls `cryptography` which is already a dependency), aiohttp (existing), vanilla JS.

**Spec:** `Docs/superpowers/specs/2026-10-15-served-mode-auth-and-mobile-surface-design.md` §3.5 (binding) + §3.4 posture
**ADR:** Task 1 creates `backlog/decisions/235-served-webauthn-passkey-login.md` (amends ADR-233 per spec §3.5: "Amends the same ADR when it lands").

## Global Constraints

- RP ID/RP name derived ONLY from the configured `public_url` host at server build time; **passkeys are refused (registration and login) when `public_url` is unset or its host is an IP literal or localhost** — the QR/token paths remain the fallback (spec §3.5; the WebAuthn spec forbids IP RP IDs). Surface the refusal as a clear "Passkeys need a domain origin" notice on `/auth/devices`, not a 500.
- Discoverable (resident) credentials, attestation `"none"`, user verification `"preferred"`; ceremonies validate origin, challenge, type, and RP ID hash via the fido2 library's `Fido2Server.register_complete`/`authenticate_complete`; **sign_count must be monotonic** (clone detection: equal-or-lower count on a credential that has produced a count > 0 rejects the assertion and logs the credential id).
- Challenges: 32-byte urlsafe secrets, one-use, 60 s TTL, capped outstanding (64), stored digest-keyed in memory like bootstrap nonces — never persisted, never logged.
- Credential store: `get_user_data_dir() / "webauthn_credentials.json"`, atomic tmp+rename writes, 0600, single-owner (served parent). Stored per credential: id (hex), COSE public key (base64 CBOR bytes), rp_id, sign_count, label, created_at/last_used_at (wall clock), aaguid. NO private key material ever server-side.
- Login-complete grants a **persistent** device session (spec §3.5: nothing to type, survives restarts) via the existing `_new_session`-level path with `device_label` from the User-Agent (`_device_label_from_user_agent`) and label suffix "(passkey)".
- Registration requires an authenticated session + CSRF header (middleware default for POSTs outside `_public_auth_paths`); login begin/complete ARE added to `_public_auth_paths` (origin-validated POSTs, like `/auth/login`).
- No secrets (challenges, keys, cookies) in logs; log credential-id prefixes and statuses only.
- Baselines (environmental, pre-existing): 11 `test_web_auth_integration.py` + 4 `test_web_server_dependency_gate.py` + 209 `Tests/Agents/test_local_tool_provider.py` `RecoveryRequired` failures — name-identical before/after; green sets must stay green (Canvas web_auth 80, session_store 5, cli_pair 3, approval_constants 2, mobile service 7 / api 23 / page 16).
- Worktree `/Users/macbook-dev/Documents/GitHub/tldw-auth-phase1a`, branch `feat/served-passkeys-phase2a` off `feat/served-mobile-phase1b` (stacked; PR retargets as ancestors land).
- Targeted pytest only (`.venv/bin/pytest`); no full sweeps without operator opt-in.

---

### Task 1: ADR-235, dependency, Backlog task

**Files:**
- Create: `backlog/decisions/235-served-webauthn-passkey-login.md`
- Modify: `pyproject.toml` (both dependency blocks that list `textual-serve` — web extra ~line 365 and all-tools ~line 479: add `"python-fido2>=1.1,<2"`)
- Create via CLI: Backlog task (binary `~/.local/bin/backlog-py`; commits the task markdown too)

**Interfaces:**
- Consumes: nothing.
- Produces: ADR path + task id for later commits; the fido2 import available after `pip install -e ".[web]"` in the worktree venv (controller installs).

- [ ] **Step 1: ADR-235** (amends ADR-233 — link it, don't duplicate):

```markdown
# ADR-235: Served WebAuthn passkey login (amends ADR-233)

Date: 2026-10-15
Status: Accepted (spec 2026-10-15 §3.5; ADR-233 remains authoritative for
session persistence, device pairing, and incident response)

## Context

ADR-233 ended the long-token pain with persistent sessions and QR pairing.
Spec §3.5 deferred passkeys to Phase 2 as the premium path for the stable
HTTPS-domain origin: biometric unlock, nothing to type, credentials that
survive restarts because only public keys are stored server-side.

## Decision

1. WebAuthn login via python-fido2 (>=1.1,<2, web extra only). Discoverable
   credentials, attestation "none", user verification "preferred".
2. RP ID derived from [web_server] public_url's host at server build time.
   Passkeys are refused when public_url is unset or its host is an IP
   literal/localhost (WebAuthn spec forbids IP RP IDs) — QR/token remain
   the fallback; /auth/devices shows a notice, not an error.
3. Credential store: user_data_dir()/webauthn_credentials.json (0600,
   atomic, single-owner) holding credential ids, COSE public keys, RP id,
   sign_count, labels, timestamps. No private key material server-side.
   Challenges are in-memory, one-use, 60 s TTL, capped.
4. Registration from an authenticated /auth/devices ("Add a passkey",
   CSRF-protected). Login from /auth/login ("Use passkey") grants a
   persistent device session via the existing ADR-233 session machinery.
5. sign_count monotonicity enforced (clone detection): a regression rejects
   the assertion.

## Consequences

- A stolen webauthn_credentials.json alone cannot authenticate (public
  keys only). A cloned credential is detected via sign_count regression.
- Losing the file logs every passkey out (QR/token still work).
- No username step: single-operator, discoverable credentials — the
  authenticator offers the matching credential for the RP ID.
- Conditional mediation (autofill UI) is deliberately NOT included;
  button-initiated ceremonies only. Follow-up if wanted.

## Alternatives considered

- OTP/TOTP apps: typing codes on a phone — the friction passkeys remove.
- mTLS client certificates: browser UX is poor on phones and management
  is heavier for a single operator.
```

- [ ] **Step 2: Backlog task** — `backlog-py task create "Served WebAuthn passkey login (Phase 2a, amends ADR-233)" -d "Spec 2026-10-15 §3.5: fido2 ceremonies, credential file store, /auth/passkeys routes, login+devices UI" -s "In Progress"`; link the ADR in notes.
- [ ] **Step 3: pyproject** — add `"python-fido2>=1.1,<2"` to BOTH textual-serve blocks; controller runs `.venv/bin/pip install -q -e ".[web]"` and verifies `import fido2; fido2.__version__`.
- [ ] **Step 4: Commit** — `docs(adr): ADR-235 served WebAuthn passkey login (amends ADR-233)` (ADR + task file + pyproject).

---

### Task 2: WebAuthn store + service

**Files:**
- Create: `tldw_chatbook/Web_Server/webauthn_store.py`, `tldw_chatbook/Web_Server/webauthn_service.py`
- Test: `Tests/Web_Server/test_webauthn_service.py`

**Interfaces:**
- Consumes: `fido2.server.Fido2Server`, `fido2.webauthn` types; `get_user_data_dir`; the ADR-233 clock discipline (wall clock, injectable).
- Produces:

```python
# webauthn_store.py — the web_sessions.json idiom, verbatim patterns from
# Web_Server/session_store.py (atomic tmp+rename, 0600, fail-safe load):
@dataclass(frozen=True)
class StoredCredential:
    credential_id_hex: str
    public_key_b64: str        # base64 of the COSE CBOR bytes
    rp_id: str
    sign_count: int
    label: str
    aaguid: str
    created_at: float
    last_used_at: float

class WebAuthnCredentialStore:
    def __init__(self, path: Path) -> None: ...
    def load(self) -> dict[str, StoredCredential]: ...        # {} missing/corrupt/non-object
    def save(self, creds: Mapping[str, StoredCredential]) -> None: ...

# webauthn_service.py:
class PasskeysUnavailable(Exception): ...   # raised when RP derivation fails (no public_url / IP host)

class WebAuthnService:
    def __init__(self, *, public_url: str | None, store: WebAuthnCredentialStore,
                 clock: Callable[[], float] = time.time,
                 challenge_factory: Callable[[], str] = lambda: secrets.token_urlsafe(32)) -> None: ...
    # rp_id/rp_name derived at construction from public_url (urlsplit().hostname;
    # refuse IP literals incl. v6 brackets, empty, ".invalid" style empties,
    # localhost) else raise PasskeysUnavailable.
    # Owns a fido2.server.Fido2Server({...}, rp_id=rp_id, attestation="none") equivalent
    # (Fido2Server's user_verification="preferred" default stands; verify_origin Strict).
    @property
    def available(self) -> bool: ...                      # RP derivation succeeded
    def register_begin(self, *, label: str) -> dict: ...  # one-use challenge stored; returns
    #   {"publicKey": credential_creation_options-as-JSONable, "challenge_id": digest-key}
    def register_complete(self, challenge_id: str, client_data_json: str,
                          attestation_object_b64: str) -> dict: ...
    #   fido2 register_complete(state, response) -> credential_id, cose public key;
    #   persists StoredCredential (label from begin, UA-supplied by caller); returns
    #   {"credential_id": hex-prefix (first 8), "label": label}
    def login_begin(self) -> dict: ...                    # allow_credentials=[] (discoverable)
    def login_complete(self, challenge_id: str, client_data_json: str,
                       authenticator_data_b64: str, signature_b64: str) -> StoredCredential: ...
    #   authenticate_complete(state, response); enforces sign_count monotonicity (regression
    #   -> AuthenticationError("authentication denied"), logs credential-id prefix, persists
    #   last_used_at + new count); unknown credential_id -> AuthenticationError
    def list_credentials(self) -> list[dict]: ...         # [{credential_id_prefix, label, created_at, last_used_at}]
    def remove_credential(self, credential_id_hex: str) -> bool: ...
```

- [ ] **Step 1: Failing tests** — the ceremony WITHOUT hardware, using the fido2 "none"-attestation construction (this is the test recipe, adapt names to the library's exact API at the pinned version):

```python
# Tests/Web_Server/test_webauthn_service.py — build both halves in-process.
from fido2 import cbor
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives import hashes

def _make_keypair_and_attestation(rp_id, challenge, credential_id):
    """Build a 'none'-fmt attestation object for a fresh EC P-256 key."""
    priv = ec.generate_private_key(ec.SECP256R1())
    pub_cose = cbor.encode(_cose_key_dict(priv.public_key()))   # kty=2, alg=-7, crv=1, x, y
    rp_id_hash = hashlib.sha256(rp_id.encode()).digest()
    flags = 0x45            # UP|UV|AT
    auth_data = rp_id_hash + struct.pack(">BIB", flags, 0, 0) + b"\x00"*16 \
        + struct.pack(">H", len(credential_id)) + credential_id + pub_cose
    att_obj = cbor.encode({"fmt": "none", "attStmt": {}, "authData": auth_data})
    return priv, att_obj

def _client_data_json(kind, challenge, origin):
    return json.dumps({"type": kind, "challenge": challenge, "origin": origin})
```

Test cases (each with a fake clock for TTL; service built with
`public_url="https://chatbook.example"`, `store=WebAuthnCredentialStore(tmp_path/"webauthn_credentials.json")`):
1. `test_registration_and_login_round_trip` — register_begin → build attestation (origin `https://chatbook.example`, the returned challenge) → register_complete persists one credential → login_begin → build assertion (auth_data with flags 0x05 UP|UV, count 1; signature over auth_data||sha256(client_data) with the private key) → login_complete returns the StoredCredential; store file exists, 0600, contains no private material (only b64 COSE + hex id).
2. `test_login_grants_nothing_directly_but_service_validates_origin` — assertion whose client_data origin mismatches → AuthenticationError.
3. `test_challenge_one_use_and_ttl` — reuse of a consumed challenge_id → AuthenticationError; expired (>60 s via fake clock) → AuthenticationError.
4. `test_sign_count_regression_rejects` — login twice with the same count (1, then 1) → second raises.
5. `test_ip_or_missing_public_url_refuses` — `WebAuthnService(public_url=None, ...)` and `public_url="https://10.0.0.5:8443"` raise/hold `PasskeysUnavailable` (`available is False`).
6. `test_remove_credential` — remove → login_complete for that id → AuthenticationError.
7. `test_store_fail_safe` — corrupt/non-object JSON → load() == {}.

- [ ] **Step 2: Verify failure** (module missing), **Step 3: Implement** both modules (store mirrors `session_store.py` byte-for-byte in discipline; service holds `self._states: dict[str, _ChallengeState]` digest-keyed with expiry + prune-on-use, cap 64, mirroring `WebAuthManager._bootstraps`), **Step 4: Verify pass**, **Step 5: Commit** — `feat(web-auth): WebAuthn credential store + ceremony service (ADR-235)`.

---

### Task 3: Routes, wiring, and browser UI

**Files:**
- Modify: `tldw_chatbook/Web_Server/serve.py` (routes + mixin ctor `webauthn_service` kwarg; `_public_auth_paths` += `/auth/passkeys/login/begin`, `/auth/passkeys/login/complete`; `create_server` builds `WebAuthnCredentialStore(get_user_data_dir()/"webauthn_credentials.json")` + service from `public_url`; `handle_login` GET page gains "Use passkey" button when service available; `handle_devices_page` gains "Add a passkey" + credential list + per-credential Remove)
- Create: `tldw_chatbook/Web_Server/static/passkeys.js` (served via the `/static/chatbook/{name}` allowlist)
- Test: extend `Tests/Web_Server/test_web_auth_integration.py`

**Interfaces:**
- Consumes: Task 2's `WebAuthnService` API verbatim; `WebAuthManager` persistent-session minting via `login_with_access_token`-equivalent — add `WebAuthManager.login_device(*, client_ip: str, device_label: str) -> SessionGrant` (persist=True path; thin `_new_session` wrapper, no token check) and use it in login-complete.
- Produces:

| Route | Auth | Body | Response |
| --- | --- | --- | --- |
| `POST /auth/passkeys/register/begin` | session + CSRF | `{"label"?}` | `200 {"publicKey": {...}, "challenge_id": "..."}` or `503 {"error":"passkeys_unavailable"}` |
| `POST /auth/passkeys/register/complete` | session + CSRF | `{"challenge_id","client_data_json","attestation_object_b64"}` | `200 {"ok":true,"credential_id":"ab12…","label":...}` |
| `POST /auth/passkeys/login/begin` | public (origin-checked) | — | `200 {"publicKey": {...}, "challenge_id"}` or `503 passkeys_unavailable` |
| `POST /auth/passkeys/login/complete` | public (origin-checked) | `{"challenge_id","client_data_json","authenticator_data_b64","signature_b64"}` | `303` to `/` + session cookie (persistent), or `401` |

All `Cache-Control: no-store`. Login-complete uses `manager.login_device(device_label=label+" (passkey)")` and `_set_session_cookie` exactly like the QR bootstrap handler. When `service.available is False`: begin routes answer `503 {"error":"passkeys_unavailable"}`; the devices page renders the "Passkeys need a domain origin (`public_url`)" notice instead of the button; the login page omits "Use passkey".

`passkeys.js` (loaded by both pages, `defer`): `registerPasskey()` — POST begin → `navigator.credentials.create({publicKey})` (base64url-decode the id/challenge fields per the WebAuthn dictionary rules) → POST complete → reload; `loginPasskey()` — POST begin → `navigator.credentials.get({publicKey})` → POST complete (form-encoded or JSON; on 303 → `location.assign("/")`) → on failure show "Passkey sign-in failed" inline, never alert(). Feature-detect `window.PublicKeyCredential` and hide buttons when absent.

- [ ] **Step 1: Failing tests** (integration idiom, config-free paths only): register-begin without session → 401; with session but no CSRF header → 401; register-begin under a `public_url="https://chatbook.example"` service → 200 + challenge_id + publicKey containing the rpId; same under IP public_url → 503 passkeys_unavailable; login-begin public → 200; login-complete with garbage → 401; **full browser-less round-trip**: drive the Task-2 in-process ceremony helpers against the SERVER's routes (register via the authed client with the built attestation; then a fresh client login via the built assertion → 303 + `Set-Cookie` + `WebSessionStore(...).load()` gains one persistent record); devices page shows "Add a passkey" (domain) vs the notice (IP); login page contains "Use passkey" only when available.
- [ ] **Step 2/3/4:** verify failure → implement (mirror handler shapes from the devices/mobile routes; JSON bodies via `_read_device_json`) → verify green + baselines (integration 11 name-identical; Canvas 80; mobile suites untouched-green).
- [ ] **Step 5: Commit** — `feat(web-auth): /auth/passkeys routes, login + devices UI (ADR-235)`.

---

### Task 4: Docs, close-out, live-check procedure

**Files:**
- Modify: `tldw_chatbook/Web_Server/README.md` (auth section: passkey paragraph — registration from /auth/devices, "Use passkey" on /auth/login, domain-origin requirement, what losing the credentials file means, sign_count clone detection; incident response: removing all passkeys = delete the file or per-credential Remove), `backlog/tasks/<TASK-ID> - ....md`, ADR-233 (add one line under Consequences: "Amended by ADR-235 (passkey login).")
- Test: none new (Task 3's set is the coverage); re-run the full targeted matrix.

- [ ] **Step 1: README + ADR-233 amendment line + task file close-out** (ACs checked, Implementation Notes, status Done via backlog-py).
- [ ] **Step 2: Targeted regression matrix** — webauthn_service suite + integration (baseline 11) + Canvas 80 + session_store 5 + cli_pair 3 + approval_constants 2 + mobile 7/23/16 + dep-gate 4 name-identical.
- [ ] **Step 3: Live-check procedure** — real-passkey verification cannot be automated here (no authenticator): document the 6-step operator procedure in the task file (serve with an HTTPS public_url, register from /auth/devices with the phone/laptop platform passkey, sign out, sign in via "Use passkey", restart the process and sign in again — persistence — then Remove the credential and confirm login now fails). Note that ceremonies are covered by the in-process tests; the live check validates platform integration only. Lessons check per DoD.
- [ ] **Step 4: Commit** — `docs(web-auth): passkey README + ADR-233 amendment + close-out (ADR-235)`.

---

## Self-Review

**Spec §3.5 coverage:** fido2 in web extra (T1); discoverable creds + RP-ID-from-public_url + IP refusal (T2 service + T3 503/notice/UI gating); registration from authed /auth/devices (T3); "Use passkey" on login (T3); persistent session on login-complete (T3 login_device + cookie via existing path); QR remains fallback — untouched (global constraint; no route changes to /auth/bootstrap); ADR amendment (T1 ADR-235 + T4 line in 233). Conditional-UI autofill deliberately excluded and recorded as a possible follow-up in ADR-235 — spec does not require it.

**Placeholder scan:** the two API-shape risks are anchored, not vague — (a) exact fido2 ≥1.1 method signatures (`register_begin/register_complete/authenticate_complete`, options-JSON encoding) must be read from the installed library at implementation time (Task 2 Step 3 says so; the test recipe's helper names are marked adaptable); (b) WebAuthn dictionary base64url conversions in passkeys.js follow the standard encoding rules and must round-trip the server's JSON — the round-trip test covers them server-side, the operator live check covers the browser side.

**Type consistency:** `StoredCredential`/`WebAuthnCredentialStore` (T2) consumed verbatim in T3's service ctor; `WebAuthnService` method names identical across T2/T3/the JS calls; `PasskeysUnavailable` surfaces as the literal `passkeys_unavailable` error code in both begin routes and the devices-page notice; `login_device` added once (T3 Interfaces) and used once.
