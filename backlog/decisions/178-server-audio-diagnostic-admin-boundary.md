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

The server reports `can_run_audio_diagnostics` from the same administrator
guard used by the protected routes in `/users/me/capabilities`. This includes
both an admin role and the server's wildcard-permission bypass. The connected
audio service checks that decision immediately before either privileged
diagnostic, using the same API client that will make the request. An explicit
false result yields `admin_required` before dispatch. A 403 from the capability
read or the diagnostic is converted to the same result, covering permission
changes between the check and dispatch. Passive health reads do not make a
capability request.

Older servers may omit the new decision or the discovery route. Chatbook then
dispatches the diagnostic and lets the server's existing authorization decide;
any diagnostic 403 still becomes `admin_required`. Other discovery errors
propagate and do not dispatch the privileged request.

The scope service delegates this gate to the connected service; it does not
cache administrator status or infer it from local mode, general audio policy,
or a prior session. Other audio operations keep their existing contracts.

## Alternatives

- Rely only on the server 403: safe for the server, but leaves a known
  capability exposed to non-admin users and gives them an opaque error.
- Cache an administrator boolean in Chatbook: risks using stale identity after
  a server/account change or role revocation.
- Infer from the profile role: rejects a non-admin principal whose wildcard
  permission passes the server guard. The role alone is not authoritative.
