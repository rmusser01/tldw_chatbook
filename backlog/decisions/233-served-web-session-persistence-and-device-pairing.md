# ADR-233: Served web session persistence and device pairing

Date: 2026-10-15
Status: Accepted (implements spec Docs/superpowers/specs/2026-10-15-served-mode-auth-and-mobile-surface-design.md §3)

## Context

Served mode admitted remote browsers with one shared long access token and
in-memory sessions: 30-minute idle / 8-hour absolute expiry, and every process
restart revoked every device. For a single operator on a phone this meant
re-typing the long token constantly.

## Decision

1. Sessions persist to `get_user_data_dir()/web_sessions.json` (atomic
   tmp+rename writes, mode 0600, cookie digests only — never raw cookie
   values). Defaults: 30-day sliding idle, 180-day absolute.
2. The auth clock basis switches from `time.monotonic` to `time.time()`
   (wall clock) so persisted timestamps survive restarts. Wall-clock steps
   at day-scale granularity are accepted.
3. Device pairing: the existing one-use bootstrap nonce gains a QR face —
   "Add a device" on `/auth/devices` and `tldw-serve --pair`. Nonce TTL for
   issuance is 120 s; exchange stays one-use and rate-limited.
4. `/auth/devices` lists devices (User-Agent-derived labels, editable),
   supports sign-out, per-device revoke, and revoke-all.
5. Token-form login defaults to persistent sessions; unchecking
   "remember this device" downgrades to the old short in-memory session.

## Consequences

- Stopping the process no longer revokes sessions. Incident response:
  Revoke all on /auth/devices, or stop the process AND delete
  web_sessions.json, or rotate the access token.
- A stolen cookie is long-lived. Mitigations: digest-only disk storage,
  0600 file, per-device revocation with last-seen visibility. No
  client-IP binding — Wi-Fi/cellular roaming would break sessions;
  accepted for a single-operator deployment.
- Loopback auto-login, login rate limiting, TLS/proxy fail-closed rules,
  SameSite=Strict + CSRF posture, and the session capacity cap are unchanged.
- Amended by ADR-235 (passkey login).

## Alternatives considered

- Passkeys/WebAuthn as primary: deferred to Phase 2 (LAN-by-IP origins
  cannot be WebAuthn RP IDs; QR covers every path).
- Network-layer trust (tailnet/proxy): rejected as primary — pushes setup
  burden, leaves the WebSocket session layer half-designed.
