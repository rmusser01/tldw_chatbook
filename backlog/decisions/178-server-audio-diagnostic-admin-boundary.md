# ADR-178: Server audio diagnostics follow server administrator identity

Status: Proposed
Date: 2026-09-23
Task: TASK-32917

## Context

The server allows passive STT health reads for ordinary users but requires an
administrator for `warm=true` and for the streaming diagnostic endpoint.
Chatbook's general audio policy actions do not encode that role distinction, so
connected users currently discover it only as a raw 403 after dispatch.

## Decision

The connected audio service checks the active server's current-user identity
immediately before either privileged diagnostic, using the same API client that
will make the diagnostic request. Only `user.role == "admin"` passes. Missing or
non-admin identity fails closed with an `admin_required` policy result. The
server remains the final authority: a 403 from either diagnostic is converted
to the same result, covering permission changes between the identity read and
dispatch. Passive health reads do not make an identity request.

The scope service delegates this gate to the connected service; it does not
cache administrator status or infer it from local mode, general audio policy,
or a prior session. Other audio operations keep their existing contracts.

## Alternatives

- Rely only on the server 403: safe for the server, but leaves a known
  capability exposed to non-admin users and gives them an opaque error.
- Cache an administrator boolean in Chatbook: risks using stale identity after
  a server/account change or role revocation.
- Add a new server capability endpoint: unnecessary for two existing routes
  whose current-user profile already exposes the authoritative role.
