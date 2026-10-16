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
