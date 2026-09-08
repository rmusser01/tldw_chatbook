# ADR-127: Fresh-install private data root recovery

Status: Accepted
Date: 2026-09-07
Related task: TASK-32011
Amends: [ADR-029](029-local-private-data-boundary.md) (default location selection only)

## Context

A fresh installation crashes during config import if an existing `.local` or
`share` ancestor is group/world-writable. The private-path guard correctly refuses
that namespace, but startup currently has only one default data location. The
reporter cannot be contacted to repair their machine. Reproduction with an
isolated HOME and `share` at `0775` reaches the reported `shared_writable_parent`
exception before any database is created.

## Decision

Keep `~/.local/share/tldw_cli` as the conventional data root. If securing it fails
specifically with `unsafe_parent: shared_writable_parent`, and no filesystem entry
exists at that root, automatically secure `~/.tldw_cli-data` as the default root.
Both roots remain profile containers: the existing sanitized user name is appended.

The home-level directory is reserved application storage and is secured by the
same descriptor-based `0700` directory guard. Existing files, SQLite databases,
and logs retain their `0600` lifecycle. No ancestor permissions are changed and
no group-membership assumptions or weaker private-path checks are introduced.

Directory presence persists the selection without new config fields or a second
mutable configuration owner. When only the fallback exists, keep using it even
if the conventional ancestors are repaired. When both roots exist, refuse the
ambiguous default; an explicit `[paths] data_dir` can select the intended root.
Never inspect/migrate database contents or silently hide an existing conventional
root. Existence probes use `lstat`, count dangling final symlinks as entries, and
propagate errors other than absence instead of treating inaccessible data as new.

Explicit data roots retain their existing validation and precedence, with no
fallback. Unsafe HOME/ancestors, symlinks, foreign ownership, unavailable guards,
and other I/O errors still fail closed. Config-file selection is unchanged; this
decision addresses the reported data-root failure, not all possible filesystem
misconfiguration. XDG_DATA_HOME behavior remains unchanged.

## Alternatives

- Allow writable ancestors: loses the guarantee that another local user cannot
  replace a directory naming private data.
- Automatically chmod `.local`, `share`, or HOME: changes user-owned sharing
  policy outside application storage.
- Infer a private group from its name or account enumeration: depends on mutable
  local/network account policy and cannot establish filesystem isolation.
- Use an ephemeral directory: would turn persistent conversations into temporary
  data and break restart behavior.
- Retry the conventional root on each launch: silently switches profiles after a
  permission repair. Fallback presence deliberately prevents that.

## Verification

Real fresh-process config imports and SQLite writes/reopens must succeed with
writable `.local`/`share`, retain data on restart and after permissions are
repaired, and leave ancestors untouched. Existing-root, ambiguous-root, explicit
override, symlink, unsafe-HOME, and non-permission-failure cases must still refuse.

Design and plan: [TASK-32011 design](../../Docs/superpowers/specs/2026-09-07-fresh-install-data-root-recovery.md).
