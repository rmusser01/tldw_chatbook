# Local Privacy Containment Design

Date: 2026-07-23
Status: User-approved design; pending written-spec review
ADR: [ADR-022](../../../backlog/decisions/022-local-private-data-boundary.md)
Backlog: [TASK-488](../../../backlog/tasks/task-488%20-%20Harden-private-artifact-permissions-and-diagnostics.md), [TASK-489](../../../backlog/tasks/task-489%20-%20Make-config-persistence-use-one-effective-path-boundary.md), [TASK-490](../../../backlog/tasks/task-490%20-%20Remove-private-payloads-from-persistent-logs-and-tool-history.md), [TASK-491](../../../backlog/tasks/task-491%20-%20Contain-legacy-Notes-sync-paths-and-preserve-file-modes.md)

## Summary

Chatbook will enforce one explicit boundary for local private data before
repairing the later lifecycle, evaluation, worker, and packaging findings.
Existing eligible config, database, backup, and log files are automatically
hardened on POSIX. New private artifacts are created owner-only. Configuration
persistence is consolidated behind the effective config path. Persistent logs
and tool history retain operational metadata but not user, model, credential,
or tool payload values. Legacy Notes sync rejects descendant links and
outside-root targets and preserves existing file permissions.

This tranche contains active exposures without attempting the larger
application-state decomposition or the future file-backed Notes architecture.

## Problem

The audit reproduced independent privacy failures that share two architectural
causes:

1. several modules create or replace private files without a common
   permission and ownership policy; and
2. several modules independently persist configuration or diagnostic content
   without one owner for path selection and redaction.

Concrete reproduced outcomes were:

- `config.toml` and new SQLite databases at `0644`;
- an INFO log containing a full sentinel user prompt;
- tool arguments and results retained in logs and unbounded process history;
- provider and summarization logs containing request payloads, response
  bodies, or API-key fragments;
- encryption reporting success while modifying an unrelated default config;
- providers continuing to use a stale settings dictionary after save;
- a Notes-root file symlink importing outside-root content;
- an existing private note changing from `0600` to `0644` after sync.

## Goals

- Automatically contain eligible existing private artifacts on POSIX.
- Create private artifacts without a world/group-readable interval.
- Refuse unsafe config/database operations rather than silently continuing.
- Disable file logging if its target cannot be secured.
- Make the effective config path the only `config.toml` persistence target.
- Ensure settings saved during a process are observed by later provider calls.
- Keep persistent logs useful without retaining private payload values.
- Bound tool execution history without changing immediate tool results.
- Keep legacy Notes reads and writes beneath one canonical selected root.
- Preserve user-selected modes on existing Notes and use `0600` for new
  synchronized Notes on POSIX.
- Report platform enforcement honestly, especially on Windows.

## Non-Goals

- Keyring migration, new encrypted credential storage, or secret rotation.
- A raw-payload support bundle or persistent diagnostic mode.
- Recursive permission changes outside Chatbook-owned directories.
- General-purpose repository secret scanning.
- Windows ACL implementation or a claim that POSIX mode bits secure Windows.
- The file-backed Notes projection, mutation journal, recovery database, or
  authority changes described by ADR-021.
- Data deletion/vector retention, schema migration, eval, worker, packaging,
  or application-state work; those remain later tranches.

## Terminology

| Term | Meaning |
| --- | --- |
| Private file | A config, credential-bearing backup, application database or sidecar, persistent log, or other artifact explicitly classified by this design. |
| Application-owned directory | A Chatbook-specific config, data, or log directory selected by Chatbook defaults, not an arbitrary parent chosen by a user. |
| Effective config path | `_get_effective_config_path()`, honoring `TLDW_CONFIG_PATH` before the default. |
| Eligible object | A verified regular file or directory owned by the current effective POSIX user and opened without following a link. |
| Metadata-only log | A record containing operation identity and measurements but no prompt, message, payload, response-body, credential, tool-argument value, or tool-result value. |
| Canonical Notes root | The resolved directory selected at sync start and used for every containment comparison during that pass. |

## Architecture

### Private-path primitive

A small utility owns private artifact creation, inspection, and POSIX
hardening. It returns a structured posture rather than a bare boolean so
callers can distinguish:

- enforced;
- already private;
- unsupported/unverified platform;
- wrong owner;
- link or non-regular object;
- permission operation failure.

On POSIX, inspection and hardening operate on a verified descriptor with
no-follow flags where the platform exposes them. The helper verifies the
descriptor's object type and effective-user ownership before changing mode.
It does not recursively traverse arbitrary directories.

Application-owned private directories use `0700`. Private regular files use
`0600`. For a custom config or database path, Chatbook hardens the selected
file but does not change its parent directory.

Callers choose the failure policy:

- config and database open/write paths fail closed on an unsafe or
  unenforceable POSIX target;
- persistent file logging is disabled on an unsafe target while terminal/UI
  logging continues;
- historical backups or log generations that cannot be changed produce a
  redacted diagnostic and are not rewritten or deleted.

On Windows, the helper returns `unverified_platform`. Creation continues with
the normal platform APIs, but diagnostics and Settings must not label the
result owner-only or ACL-secure.

### Database and sidecar lifecycle

Before the first `sqlite3.connect` for a new file database, the database path
is created exclusively with private mode and then passed to SQLite. Existing
databases are verified and hardened before connection. In-memory databases are
unchanged.

Tests cover the real journal modes used by the application. A private database
must produce private `-wal`, `-shm`, or journal files on supported POSIX
platforms. Backups created by Chatbook use the same private creation path.

This tranche does not centralize every database class. It adds the smallest
shared primitive and applies it at the common/base connection seams and the
sampled independent database implementations that bypass those seams.

### Persistent log lifecycle

The rotating file handler opens every new log generation through a private
opener instead of relying on the process umask. Startup verifies the active log
and eligible rotated generations. The application-owned log directory is
private.

Failure to secure the path prevents installation of the file handler. It does
not disable terminal, Rich, or in-app logging.

### Exclusive config persistence owner

The config module owns every `config.toml` mutation:

- first creation;
- batched setting save;
- setting deletion;
- encryption enable;
- encryption disable;
- encryption password change;
- shutdown encryption persistence;
- reset/default creation.

Every operation resolves the effective path once, holds the existing
in-process config lock for the read-modify-write cycle, and uses atomic private
replacement. App and UI modules call config APIs rather than opening the file.

Provider and UI consumers call the cached settings loader/accessor at a
request or render boundary. They do not import the mutable module-level
`settings` object. A guard test rejects new production imports of that object
and new direct `config.toml` writes outside the owner module.

Cross-process lost-update protection is not added in this tranche; it remains a
separate config concurrency task because it requires a portable locking
contract. Atomic replacement continues to prevent partial files.

### Metadata-only logging

For Chat, provider adapters, summarization, and shared tool execution:

- requests log provider, model, streaming state, message count, payload byte or
  character length, timeout, and retry count;
- responses log provider, status code, duration, streaming completion, and
  response length;
- errors log a sanitized category, status code, and exception type without a
  raw response body;
- tool calls log tool name, argument names, status, duration, and result type
  or size.

Raw prompts, messages, system prompts, request dictionaries, response
dictionaries/bodies, API keys or fragments, tool argument values, and tool
result values are excluded at every persistent log level, including DEBUG.

The executor's public result remains unchanged. Its internal history becomes a
bounded collection with a fixed initial limit of 100 records, matching the
existing `get_execution_history(limit=100)` API. Records contain identity,
timestamps, status, duration, argument names, cache status, and result
type/size only. `get_execution_history()` still returns a list copy.

This policy directly covers Chatbook-owned logging. Third-party libraries stay
at their existing warning-or-higher logging floors. If a representative
sentinel test proves that a library warning or exception still persists
private content, that library logger is filtered or disabled for the file
handler; the design does not claim a nonexistent global content-redaction
filter.

### Legacy Notes containment

At the start of a sync pass, the selected root is resolved once. Selecting the
root through a link is permitted because the resolved directory becomes the
explicit root. Descendant symlinks, directory links, Windows junctions/reparse
points, and non-regular files are rejected.

Every read and write verifies that:

1. the lexical candidate is beneath the selected root;
2. the candidate's parent chain contains no descendant links;
3. the resolved target remains beneath the canonical root; and
4. the opened object still matches the inspected object when the platform
   exposes stable descriptor identity.

POSIX reads use no-follow opens where available. Windows uses explicit
reparse-point detection where the supported Python runtime exposes file
attributes. An unverified entry is skipped with a per-file error.

Existing files retain their permission bits during atomic replacement. New
files use `0600` on POSIX. The generic atomic-write helper's global default is
not changed because it also serves non-private export paths.

## Repository Credential Hygiene

The exact observed development filenames `openai-api-key.txt` and
`moonshot-api-key.txt` are added to `.gitignore`. A repository test verifies
that these names remain ignored. The files are not opened, deleted, moved, or
claimed to contain valid credentials.

This is a narrow staging guard, not a general secret scanner. Already tracked
content and differently named files remain outside this control.

## Error Handling and Diagnostics

- Security errors never include file contents, credentials, prompts, response
  bodies, or tool payload values.
- Path diagnostics may include the selected path, expected posture, object
  type, platform support status, and remediation.
- A failed hardening attempt never deletes or replaces the target.
- One rejected Notes entry does not stop other safe entries from synchronizing.
- Settings privacy diagnostics distinguish `enforced`, `already_private`,
  `unsafe`, and `unverified_platform`.
- No success message is emitted before the permission/config operation and its
  postcondition check complete.

## Testing Strategy

All production changes follow red-green-refactor TDD.

### Permission tests

- first config and database creation are `0600` under a `0022` umask;
- existing `0644` eligible files become `0600`;
- application-owned directories become `0700`;
- symlink, non-regular, and wrong-owner simulations are not changed;
- SQLite WAL/SHM/journal files and backups are private;
- active and rotated logs are private after rollover;
- failure to secure a log disables its file handler;
- Windows simulation reports unverified rather than enforced.

### Config tests

- every encryption operation changes only `TLDW_CONFIG_PATH` when set;
- first creation, reset, delete, and shutdown paths use the same target;
- atomic replacements remain `0600`;
- settings saved after provider import are observed by the next provider call;
- a production-source guard rejects direct config writes and
  `from ...config import settings`.

### Logging and tool-history tests

Sentinel values are passed through representative:

- chat;
- cloud provider;
- local provider;
- summarization;
- tool success, failure, timeout, and cache paths.

Captured standard/loguru logs and the real rotating log must not contain any
sentinel. Metadata assertions prove the log remains useful. More than 100 tool
calls prove history bounding while immediate results remain unchanged.

### Notes tests

- outside-root file symlink;
- descendant directory symlink;
- in-root descendant link;
- Windows reparse/junction simulation;
- candidate replacement between inspection and open;
- safe regular file alongside a rejected entry;
- existing `0600`, `0640`, and `0644` mode preservation;
- new-file `0600` creation.

## Delivery Decomposition

1. TASK-488 introduces the private-path posture and applies it to private
   artifacts, databases, sidecars, backups, and rotating logs.
2. TASK-489 establishes one effective-path config persistence owner and live
   settings access.
3. TASK-490 removes private payload retention from logs and tool history.
4. TASK-491 contains legacy Notes paths and preserves file modes.

TASK-489, TASK-490, and TASK-491 depend on the TASK-488 private-path primitive
but are otherwise independently reviewable. Each task receives its own
implementation plan, focused tests, verification, and review gate.

## ADR Check

ADR required: yes

ADR path: `backlog/decisions/022-local-private-data-boundary.md`

Reason: This design establishes cross-module security, privacy, configuration
ownership, logging, filesystem containment, and platform-enforcement policy.
ADR-012 continues to own provider credential UX, while ADR-021 continues to own
the future file-backed Notes authority and recovery architecture.
