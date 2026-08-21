# TASK-19520: Skill Trust Material Permissions Design

**Status:** Approved for planning

**Date:** 2026-08-21

**Task:** TASK-19520

**Governing decision:** [ADR-009: Local Skill Trust Boundary](../../../backlog/decisions/009-local-skill-trust-boundary.md)

## Context

The local-skill trust store persists an authenticated manifest, encrypted approved-version snapshots, and a file-backed rollback-generation marker. These files are security-sensitive inputs to the trust boundary defined by ADR-009, but the current atomic-write path creates them with the process's default filesystem permissions. On a typical POSIX installation that can leave the material readable by other users on the same machine.

TASK-17963 moved these writes to writer-unique temporary paths before an atomic `replace`, but deliberately preserved the existing permission behavior. TASK-19520 closes that remaining gap without changing the trust format, cryptography, rollback protocol, or the permissions of ordinary local-skill files.

## Goals

- Create trust manifest, snapshot, and file-backed generation-marker files with owner-only POSIX permissions (`0o600`).
- Ensure a temporary trust file is never created with broader permissions and restricted only afterward.
- Keep trust-store and snapshot directories owner-only (`0o700`) before writing sensitive files into them.
- Tighten files and trust-owned directories created by earlier versions on their next trust-store write.
- Preserve the atomic replacement, writer-unique naming, containment validation, serialization, cleanup, and error-propagation behavior established by TASK-17963.
- Keep non-trust callers of the shared atomic-write module behaviorally unchanged.
- Continue to operate on Windows and other platforms where POSIX mode bits are unavailable or advisory.

## Non-goals

- Managing Windows ACLs. Windows continues to inherit ACLs from the containing directory; POSIX mode enforcement is skipped there.
- Protecting files from malicious code running as the same OS account. ADR-009 already excludes malicious active application code and active process compromise.
- Changing trust payload formats, keys, cryptography, marker policy, or storage locations.
- Retrofitting owner-only modes onto ordinary skill content or every caller of `Skills_Interop.atomic_write`.
- Eagerly mutating trust files during reads or application startup. Legacy files are tightened when the store next writes them.
- Adding durability changes such as file or directory `fsync`; atomic-write durability is unchanged.

## Relationship to OS Access Controls

This change does not supersede the operating system's access-control model; it selects stricter defaults within that model. It is useful when separate unprivileged POSIX users share a host and directory traversal or a permissive umask would otherwise make trust material readable or writable. It is mostly redundant when an existing parent directory already prevents access, but preserves the invariant if parent permissions later change.

The modes do not separate application users that share one OS identity, restrict another process running under the same UID, override root or administrator authority, replace filesystem encryption for offline access, or implement Windows and storage-specific ACL policy. On ACL-backed or network filesystems, the effective ACL and mount semantics remain authoritative. TASK-19520 is therefore low-cost defence in depth around ADR-009's existing trust root, not a new authentication or tenant-isolation boundary.

## Security Invariants

1. On POSIX, every temporary file used for trust material is created with no group or other permissions before any path-based content write can occur.
2. On POSIX, the temporary file has mode `0o600` before `Path.replace` publishes it at the final path.
3. The trust-store directory is mode `0o700` before manifest or marker creation, and the snapshots directory is mode `0o700` before snapshot creation.
4. Replacing a permissive legacy target publishes the restrictive temporary inode, so the rewritten target becomes `0o600` without a post-replace exposure window.
5. Permission setup failure on a POSIX platform fails the trust write and follows the existing temporary-file cleanup path; the code does not continue with a knowingly broad mode.
6. Callers that do not request secure creation retain the current `Path.write_text`/`Path.write_bytes` behavior and modes.

## Options Considered

### 1. Secure creation in the shared primitive, trust-store opt-in — selected

Extend the shared atomic-write boundary with an optional owner-only temporary-file creation step. When requested, the primitive exclusively pre-creates the writer-unique temporary path with `os.open` and the fixed mode `0o600`, normalizes its owner bits through the open descriptor on POSIX, closes it, then invokes the existing writer callback and atomic replacement. Because reopening an existing file for truncation preserves its mode, the current JSON/text/bytes callbacks and serialization remain intact. Only the trust store enables this option.

This centralizes the security-critical ordering beside the existing atomic replace and cleanup logic while avoiding permission changes for ordinary local-skill files.

### 2. Private secure writers in `skill_trust_store.py`

The trust store could duplicate exclusive creation and cleanup around both its JSON and bytes writers. This limits the shared API change but creates a second implementation of the same write lifecycle and makes it easier for the two trust write paths to drift.

### 3. Owner-only permissions for all atomic-write callers

Making `0o600` the default would be simple but would silently change user-visible skill files, imports, indexes, and future shared-helper callers. Those files are outside TASK-19520's trust-material scope and may intentionally follow the user's normal umask.

### Rejected: temporarily changing the process umask

`umask` is process-global. Temporarily changing it around a write is unsafe when background workers or concurrent app instances perform unrelated writes and still does not provide a narrowly owned enforcement point.

## Selected Design

### Shared atomic-write behavior

`Skills_Interop/atomic_write.py` will accept an optional, keyword-only `owner_only: bool = False` switch on `replace_atomically`. The default preserves current behavior. The switch always means mode `0o600`; it is deliberately not a generic caller-supplied mode because TASK-19520 needs one security invariant, not a new permission-policy API.

When `owner_only` is true, the helper will:

1. Exclusively create the supplied writer-unique temporary path with `os.open` using `O_WRONLY | O_CREAT | O_EXCL`, plus `O_CLOEXEC`, `O_BINARY`, and `O_NOFOLLOW` where those flags are available.
2. Pass `0o600` at creation. A process umask can only remove permissions at this step, never add group or other access.
3. On POSIX, call `os.fchmod` on the still-open descriptor to normalize the mode to exactly `0o600` before closing it. This preserves the no-broad-permissions invariant even under an unusual umask.
4. Close the descriptor, invoke the existing path-based writer callback, and atomically replace the target.
5. Record ownership only after `os.open` succeeds. Failures after successful creation use the existing `BaseException` cleanup path; an `EEXIST` failure before creation does not unlink the unexplained path.

`O_EXCL` deliberately refuses a stale or attacker-precreated temporary path instead of truncating or following it. The existing PID-and-thread writer-unique name makes legitimate collisions exceptional; a collision fails closed and leaves both the target and unexplained temporary path unchanged. The implementation will track whether this invocation created the temp path so the outer cleanup handler unlinks only a path it owns. Default-mode callers retain their current cleanup behavior because their callback remains the creator.

### Trust-store integration

`skill_trust_store._atomic_write_json` and `_atomic_write_bytes` will continue to:

- validate the target and temporary paths against the independently supplied trust base directory;
- preserve hidden writer-unique temporary names;
- preserve JSON ordering, indentation, encoding, and trailing-newline behavior; and
- propagate write failures.

Their calls to `replace_atomically` will opt into `owner_only=True`. This covers manifests, encrypted snapshots, marker writes, and manifest rollback bytes through the existing call graph.

### Directory permissions

`_ensure_trust_directory` will keep its existing path validation and symlink refusal, create the requested leaf directory with `mode=0o700`, and on POSIX normalize that trust-owned leaf to `0o700` before returning it to a writer. Normalizing on every write also tightens permissive trust and snapshots directories left by earlier versions.

`SkillTrustStore.save_snapshot` must explicitly secure `self.store_dir` before securing `self.snapshots_dir`. It cannot rely on `snapshots_dir.mkdir(parents=True)` because production bootstrap can save a snapshot before a manifest, and Python does not apply the leaf's requested mode to intermediate parents. Manifest and marker writes already secure their trust-store base through `_atomic_write_json`. Ancestors above `self.store_dir` remain unchanged; the two trust-owned boundaries are `store_dir` and `snapshots_dir`.

On non-POSIX platforms the directory and file writes follow the same atomic code path, but `fchmod`/`chmod` enforcement and mode-bit assertions are skipped. This avoids pretending Unix mode bits implement Windows ACL policy while preserving compatibility.

## Error Handling and Recovery

- Failure after this writer creates a POSIX temporary file aborts the write. The target remains unchanged and best-effort cleanup removes this writer's temporary path.
- An exclusive-create collision before ownership is established leaves the unexplained temporary path untouched and surfaces the error.
- Failure to restrict a POSIX trust-owned directory aborts before sensitive file creation.
- A stale temporary-path collision raises rather than truncating an unexplained file.
- Marker-save rollback continues to restore the previous manifest through `_atomic_write_bytes`; the restored file is therefore tightened to `0o600` as well.
- Reads remain non-mutating. A permissive legacy file can remain permissive until a trust mutation rewrites it; this is the acceptance criterion's selected migration point.

## Testing Strategy

Add focused tests under `Tests/Skills/` that run on real temporary filesystem paths:

- On POSIX, save a manifest and snapshot through `SkillTrustStore` and a marker through `FileSkillTrustGenerationMarkerStore`; assert final file modes are exactly `0o600` and trust/snapshot directory modes are exactly `0o700`.
- Seed legacy targets with permissive modes, rewrite them through the production APIs, and assert the resulting files and trust-owned directories are tightened.
- Intercept `Path.replace` and inspect the source path before delegating, proving the in-flight temporary file is already `0o600` before publication.
- Exercise both JSON and bytes secure-write paths so manifest rollback bytes are not covered only indirectly.
- Verify a secure-create or replace failure cleans up its temporary path and preserves the original exception.
- Pre-create the exact writer-unique temp path, assert an `EEXIST` failure leaves that unexplained file byte-identical, and distinguish it from cleanup after successful creation.
- Save a snapshot into a missing or permissive trust root before saving a manifest and assert both the trust root and `snapshots/` are `0o700`.
- Keep existing concurrency and serialization-preservation tests green.
- Gate POSIX mode assertions with `os.name == "posix"`; retain platform-neutral round-trip tests so Windows exercises the new optional path in CI without asserting advisory Unix bits.

The permission tests should be mutation-checked by temporarily disabling the secure pre-creation or directory normalization and confirming the relevant assertions fail.

## Documentation and Decision Record

ADR required: no

ADR path: `backlog/decisions/009-local-skill-trust-boundary.md`

Reason: TASK-19520 directly hardens the persistence boundary already selected by ADR-009. It does not change storage ownership, trust policy, cryptography, or a cross-module contract beyond an optional implementation detail on the existing atomic-write helper.

The task implementation plan and notes will link this design and ADR-009. No user-facing configuration or workflow changes are introduced.
