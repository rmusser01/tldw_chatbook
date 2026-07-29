# ADR-029: Local Private Data Boundary

Status: Accepted (a proposed amendment awaiting this ADR's owner's sign-off is recorded below,
not yet in effect — TASK-1240)
Date: 2026-07-23
Related Tasks: [TASK-943](../tasks/task-943%20-%20Establish-private-path-boundary-and-harden-config-bootstrap.md), [TASK-489](../tasks/task-489%20-%20Apply-private-storage-boundary-to-every-SQLite-owner-and-backup.md), [TASK-490](../tasks/task-490%20-%20Harden-persistent-log-and-tool-cache-file-lifecycles.md), [TASK-491](../tasks/task-491%20-%20Make-config-persistence-use-one-effective-path-and-live-runtime-boundary.md), [TASK-492](../tasks/task-492%20-%20Remove-private-payloads-from-persistent-diagnostics-and-tool-history.md), [TASK-493](../tasks/task-493%20-%20Contain-legacy-Notes-sync-paths-and-preserve-file-modes.md), [TASK-494](../tasks/task-494%20-%20Complete-metadata-only-boundary-across-remaining-production-diagnostics.md)
Supersedes: N/A

## Decision

Chatbook treats local configuration, provider credentials, application
databases and their sidecars, private backups, persistent logs, LLM payloads,
tool payloads, and legacy synchronized Notes as private data.

On POSIX systems, Chatbook automatically narrows eligible existing private
files to owner-only access and creates replacement files with owner-only
access. Application-owned private directories use owner-only traversal.
Hardening must operate on a verified file descriptor, must not follow a
symlink, and must not change an object not owned by the current effective user.
An unsafe config or database fails closed. If a private log cannot be secured,
file logging is disabled rather than writing to an unsafe target.

Windows permission state is reported as unverified until a separately approved
ACL implementation exists. Chatbook must not translate `chmod` success into a
claim that a Windows discretionary ACL is private.

The config module is the sole persistence owner for `config.toml`. Every
create, save, delete, encryption, decryption, and password-change operation
uses the same effective path, honors `TLDW_CONFIG_PATH`, and writes atomically.
Runtime consumers resolve the current cached settings at their request
boundary instead of importing a mutable snapshot.

Persistent application logs are metadata-only with respect to user and model
content. Prompts, message bodies, provider request payloads, provider response
bodies, API keys or key fragments, tool argument values, and tool result values
are never written to normal or debug persistent logs. In-memory tool execution
history is bounded and payload-free; the immediate tool-call return contract is
unchanged.

Legacy Notes sync canonicalizes the selected root once. A user may select a
root through a symlink, but descendants that are symlinks, junctions, reparse
points, or otherwise resolve outside that canonical root are not read or
written. Existing file modes survive atomic replacement; new synchronized
files use owner-only access on POSIX.

## Amendment (2026-07-28, TASK-1240) — pending owner sign-off

**This is a proposed amendment, not an adopted one.** It is recorded here for this ADR's owner
to review and is not authoritative until that sign-off lands — the same reason TASK-1240 was
filed as a gap report rather than fixed unilaterally. The TASK-1240 branch already implements the
six events below in code, ahead of this sign-off; until sign-off, that implementation is not yet
an authorized exception to the Decision and Required Boundaries as currently written, and the
branch is not to be merged on the strength of this document alone.

"Metadata-only with respect to user and model content" is clarified to permit a fixed set of
operational events. Six are admitted: `app_started`, `app_stopping`,
`persistent_sink_installed`, `worker_failed`, `scheduler_configured`, and
`unhandled_exception`. They carry only fields from the existing schema plus
`component`, a code-side subsystem identifier.

The exclusion list is unchanged: no prompt, message body, provider request or
response payload, key fragment, tool argument value, or tool result value is
persistable. `exception_type` is a class name; exception messages remain excluded.

This restores the design's stated goal of keeping persistent diagnostics useful.
Before it, the sink admitted nothing at all, because `log_persistent_metadata()`
had no production callers and every ordinary log record was rejected.

Known residual gap, not resolved by this amendment: no test composes a real production emitter
with a real installed sink (see [TASK-1330](../tasks/task-1330%20-%20Prove-app_started-is-never-emitted-before-the-persistent-sink-installs.md)).
Full design and test rationale: [Design spec](../../Docs/superpowers/specs/2026-07-28-persistent-operational-diagnostics-design.md).

## Context

The verified audit reproduced several violations of the intended local privacy
boundary:

- first-run configuration and SQLite files were created as `0644`;
- full prompts and tool arguments were retained by INFO-level file logging;
- provider and summarization paths logged payloads, response bodies, and
  partial API keys;
- encryption controls wrote `DEFAULT_CONFIG_PATH` even when the active config
  came from `TLDW_CONFIG_PATH`;
- provider modules retained stale imported settings after a successful save;
- a file symlink inside a Notes root imported bytes from outside the root;
- a DB-to-disk Notes sync widened an existing `0600` file to `0644`.

These are cross-module security and privacy contracts. Individual call-site
patches would leave multiple persistence owners and allow the same failures to
return.

## Required Boundaries

- Permission hardening verifies object type, current-user ownership, and
  no-follow identity before changing mode.
- Application-owned config, data, and log directories may be narrowed to
  `0700`. A custom parent directory selected by the user is never recursively
  changed; only the explicitly selected private file is hardened.
- A new SQLite database is pre-created as `0600` before `sqlite3.connect`.
  WAL/SHM/journal and backup behavior is covered by behavioral tests.
- A rotating log handler opens every newly created generation as `0600`, and
  startup hardens eligible existing generations.
- Config writes hold the existing in-process serialization lock and use one
  atomic private-write path.
- Production code does not write `config.toml` directly outside the config
  persistence module and does not import the mutable module-level `settings`
  object.
- Log metadata may include provider, model, operation, status code, payload
  length, duration, retry count, tool name, argument names, and result type or
  size. It may not include private values.
- Notes containment is checked for reads and writes. A failure for one entry is
  reported and does not abort unrelated safe entries.
- The legacy Notes change does not implement the future file-backed Notes
  coordinator, authority, journal, or recovery design from ADR-021.

## Alternatives Considered

| Option | Why rejected |
| --- | --- |
| Patch only the reproduced call sites | Leaves duplicate persistence owners, log leaks in adjacent providers, and no regression boundary. |
| Warn about loose permissions without changing them | Continues exposing secrets and private databases after the application knows the posture is unsafe. |
| Set a process-wide `umask(077)` | Process-global state is unsafe after threads start and would unexpectedly change permissions for user-requested exports and other non-private files. |
| Force every synchronized note to `0600` | Overrides intentional user sharing. Existing modes should be preserved; only new Chatbook-created notes default private. |
| Follow in-root symlinks but reject outside-root targets | Retains alias, race, and duplicate-identity behavior in a security-sensitive legacy path. |
| Claim portable security through `Path.chmod` | POSIX mode bits do not establish a Windows ACL guarantee. |
| Replace config TOML with keyring/encrypted storage | Valuable but outside containment of the active exposure and requires a separate migration and UX decision. |

## Consequences

### Benefits

- Existing POSIX users receive automatic containment instead of documentation
  telling them to repair modes manually.
- Config encryption and provider reads agree on one active file.
- Persistent logs remain operationally useful without retaining user/model
  content.
- Legacy Notes sync no longer imports outside-root files or widens private file
  permissions.
- Guard tests make the privacy ownership boundaries difficult to bypass
  accidentally.

### Accepted Trade-offs

- Unsafe config/database targets can block the affected operation until the
  user repairs ownership or chooses a safe location.
- File logging may be unavailable when its target cannot be secured.
- Windows reports an unverified privacy posture until native ACL work is
  separately designed.
- Descendant symlinks and junctions are not supported by legacy Notes sync,
  including links whose current target is inside the root.
- Metadata-only logs provide less low-level provider debugging information.
  Raw payload diagnostics require a separately designed, explicit, ephemeral
  support workflow.

## Links

- [Design specification](../../Docs/superpowers/specs/2026-07-23-local-privacy-containment-design.md)
- [ADR-004: Settings Storage Defaults Restart Boundary](004-settings-storage-defaults-restart-boundary.md)
- [ADR-006: Provider-Aware Generation Settings](006-provider-aware-generation-settings.md)
- [ADR-012: Provider Credential Settings Boundary](012-provider-credential-settings-boundary.md)
- [ADR-021: File-Backed Notes Disk Authority and Recovery Replica](021-file-backed-notes-disk-authority-and-recovery.md)
