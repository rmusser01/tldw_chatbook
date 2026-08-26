# Windows Actor Pack Startup Cleanup Design

**Task:** TASK-21251

**Deferred platform work:** TASK-21252

**Governing ADR:** [ADR-074](../../../backlog/decisions/074-portable-actor-packs-and-local-persona-visual-runtime.md)

## Problem

`ActorPackImportService` runs a bounded staging sweep in its constructor. On
Windows, `secure_private_directory()` can create the staging directory but
returns `UNVERIFIED_PLATFORM` because the private-path layer does not yet verify
native ACLs. The sweep currently treats every result that is not
`verified_private` as a cleanup failure, converts that condition to
`actor_pack_import_cleanup_denied`, and aborts application startup.

The crash is deterministic on Windows and occurs before the user asks to import
an Actor Pack. At the same time, the existing sweep relies on POSIX
descriptor-relative traversal and deletion. Allowing it to continue against an
unverified Windows path would weaken ADR-074's authority boundary and could make
startup cleanup destructive.

## Goals

- Let the application finish booting when Actor Pack staging can be located but
  the platform cannot prove that it is private.
- Preserve fail-closed cleanup: no candidate may be enumerated, authenticated,
  modified, or deleted without verified staging authority.
- Preserve current authenticated cleanup behavior on supported POSIX platforms.
- Keep Windows Actor Pack import unsupported until its native filesystem
  security model is implemented and tested under TASK-21252.

## Non-goals

- Implement Windows ACL, ownership, handle-pinning, or reparse-point checks.
- Enable Actor Pack inspection, staging, activation, or publication on Windows.
- Catch or suppress unrelated Actor Pack initialization failures in `app.py`.
- Change Actor Pack archive validation or activation semantics.

## Decision

`sweep_staging()` will distinguish an unverified-but-usable private-path result
from an unusable staging path. When the result is usable but not
`verified_private`, the sweep returns `0` before calling `os.scandir()` or any
candidate helper. This makes unsupported-platform startup housekeeping a
non-destructive no-op.

All other error handling remains unchanged:

- verified private staging follows the existing bounded, authenticated cleanup
  flow;
- unusable staging or filesystem errors still become
  `actor_pack_import_cleanup_denied`;
- Actor Pack import operations retain their own fail-closed private-staging
  checks and therefore remain unavailable on Windows.

The constructor continues to invoke `sweep_staging()`. The behavior change is
localized to the sweep because that is the component which owns the cleanup
authority decision.

## Data and control flow

1. Application wiring constructs `ActorPackImportService`.
2. The constructor calls `sweep_staging()`.
3. `secure_private_directory()` classifies the staging root.
4. If the result is verified private, the current bounded cleanup proceeds.
5. If the result is usable but unverified, the sweep returns `0` without
   examining directory contents.
6. If the result is unusable or the classification raises, the stable cleanup
   error is preserved.

No new persistent data, configuration, or user-visible state is introduced.

## Error handling and observability

The startup no-op does not emit a new exception because the platform limitation
is expected and no cleanup was attempted. Existing stable Actor Pack error
categories remain unchanged. Import attempts continue to fail at the existing
security gate rather than pretending Windows import is supported.

TASK-21252 owns any future user-facing capability messaging and native Windows
diagnostics. This fix must not pre-empt that security design with a path-based
fallback.

## Testing

A regression test will arrange an existing staging candidate and make the
private-path classifier return an unverified-but-usable result. Constructing the
service must succeed, an explicit sweep must return `0`, and the candidate must
remain untouched. Spies must also prove that `os.scandir()` and candidate
authority helpers are never called on this branch. The test should fail against
the current implementation with `actor_pack_import_cleanup_denied`.

The existing authenticated startup-sweep test remains the supported-platform
regression and must continue to pass. Focused Actor Pack import tests and static
checks will verify that the guard does not affect archive inspection or cleanup
on supported platforms.

## Alternatives considered

### Catch the cleanup error in application wiring

Rejected. This would hide all cleanup initialization failures, including real
permission or filesystem errors on supported platforms, and would put Actor Pack
security policy into the application composition root.

### Treat Windows staging as verified and run the POSIX cleanup

Rejected. The cleanup depends on POSIX directory descriptors and no-follow
semantics. Windows ACL and reparse-point authority have not been proven, so this
would violate ADR-074.

### Implement native Windows Actor Pack support in this fix

Deferred to TASK-21252. Native ACL validation, reparse-point handling,
handle-relative cleanup, and Windows CI form a separate security-sensitive unit
of work. They are not required to restore application startup safely.

## ADR check

ADR required: no

ADR path: `backlog/decisions/074-portable-actor-packs-and-local-persona-visual-runtime.md`

Reason: This routine bug fix preserves ADR-074's existing fail-closed staging
and cleanup authority policy. Native Windows support will require the ADR review
and documentation captured by TASK-21252.
