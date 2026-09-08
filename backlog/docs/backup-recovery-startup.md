# Recovery startup and storage participation

Contract: [ADR-126](../decisions/126-complete-local-backup-and-recovery.md).
Implementation slice: TASK-31988. This does not expose Complete backup or
installation replacement; owner capture, drain, journal, and activation qualification
remain required.

## Recovery-required startup

Supported application, module, web, MCP, RAG backfill, and TTS worker/model launchers
check `~/.config/tldw_cli/recovery-bootstrap/` before configuration fallback or runtime
imports. `TLDW_CONFIG_PATH` selects a configuration file but cannot relocate this
fixed check. The Chatterbox pathname worker resolves the package adjacent to its
script before this guard, including source launches without `PYTHONPATH`. The check
does not parse TOML and does not depend on a recovery catalog
or on the reachability of a custom operation control root.

`Recovery required: recovery_pending` means a local operation still fences the
selected scope. `recovery_scope_uncertain` means intact local evidence cannot prove
safe scope. Keep the fixed records and custom operation storage intact. Startup
never deletes unknown, damaged, or interrupted records and never guesses that an
unreachable custom root completed successfully. Journal-authorized reconciliation
and the dedicated recovery UI arrive in later implementation slices.

Other profiles may open only when their saved local config fingerprint, registered
namespace roots, physical aliases, and retained alias tokens establish disjointness.
A changed config cannot serve as proof of disjointness during pending recovery.
An admission registry write-intent or pending remap conservatively refuses startup;
this task does not reconcile or clear those records.

Ordinary no-conflict startup remains possible on a native-unqualified host/storage.
That does not qualify backup or replacement. Known or uncertain recovery evidence
is checked before native capability decisions. An intact positively disjoint profile
remains usable even when an unrelated pending record exists; each owner path must
still remain within that verified profile scope.

## Local producer APIs and records

`startup_permission(config_selector, bootstrap_root)` returns `(allowed, code)` and
only reads bounded local evidence. `require_startup_permission()` selects the fixed
root and emits a bounded launcher refusal. Bootstrap uses stdlib and the existing
private path reader, with strict field/type/version/duplicate-key validation; it
never imports app/config, Pydantic, or the runtime lease coordinator.

`register_pending(bootstrap_root, operation_id, namespaces, control_root, selectors)`
privately creates one version-1 record named `pending-<sha256(operation_id)>.json`.
Fields are `version`, `operation_id`, `namespaces`, `control_root`, and `selectors`.
Selectors must be locally readable regular files (broken TOML is acceptable), and
available registered roots are checked against control/bootstrap overlap. A scope
without a registered mapping cannot prove disjointness. The exclusive file creation,
full data barrier, and full directory barrier must finish before the caller publishes
anything. Interrupted partial records remain fences. There is no clear operation.

`admission_authority(bootstrap_root)` selects only `bootstrap_root/admission`, using
TASK-31987's qualified protocol-2 `Admission` without weakening its native evidence,
non-nesting rule, registry write intents, or remap reservations. The private sibling
`unbound-owner` represents the synthetic `bootstrap.unbound` enrollment namespace.
Custom per-operation journal roots are separate from this stable authority.

`bind_profile(bootstrap_root, config_selector, namespaces, authority_root)` accepts
only that fixed authority and already-registered namespaces, including coverage for
the config selector. It creates a version-1 `profile-<sha256(selector)>.json` containing
`version`, `selector`, `fingerprint` (SHA-256 of config bytes), `namespaces`, and the
sorted registered `roots`. The record is local producer authority, never imported
archive authority. Each record is capped at 1 MiB and the directory at 4,096 entries;
these local metadata limits are independent of archive resource budgets.

Enrollment takes bounded exclusive maintenance of the synthetic unbound namespace.
Live unenrolled clients produce `close_unenrolled_clients_and_restart` without
publishing a profile binding. Explicit mapping refresh and existing-binding replacement
require a later owner protocol; this API never overwrites an existing binding.

## Lifetime and downstream boundaries

`admit_startup()` acquires idempotent process enrollment before runtime/config imports
and retains it until process exit. `acquire_storage(path)` returns an idempotent
`StorageLease` retirement token. A dedicated thread holds one complete predeclared
namespace set while actual DB connections or private file writers remain alive.
Multiple same-process connections share that lease, and a worker-thread close can
retire its token. After an accepted connection subclass returns successfully from
`close()`, the seam invokes the native SQLite base close before releasing its lease;
a deferred or pooled close cannot leave a writable native handle outside admission.
Failed close retains the fence; GC retires a never-explicitly-closed connection
without retrying an already attempted custom close. Memory and explicitly
classified foreign read-only sources retain their exemptions (see the owner census).

Owner authorization is directional: a declared directory admits itself and its
physically contained descendants, never undeclared ancestors or symlink escapes.
A declared regular file admits the same physical file, including hardlink aliases.
The symmetric overlap check remains only a conflict-detection primitive.

Live owners may continue inside their original verified mapping after an ordinary
config save. New processes with a changed config fingerprint participate as unbound;
new out-of-scope paths cannot enlarge a live lease. No config edit silently remaps
storage. Scope-aware binding refresh belongs to the later configuration owner work.
Spawned processes enroll independently. Forking with live owner holds is refused at
new persistence admission (`forked_owner_restart_required`); inherited SQLite handles
are not a supported alternative to fresh-process ownership.

An ordinary private seam called inside a maintenance context refuses promptly with
`maintenance_requires_owner_capability`. Capture executors still need a narrow,
executor-issued capability covering the exact source and operation-private staging
paths, plus retirement of process bootstrap and owner holds before maintenance.
No public bypass boolean, nested admission, automatic rebuild, imported authority,
or automatic fence clearing is introduced here. Raw writers still classified as
unsupported in the owner census prevent a full-protection claim.

The AST entrypoint census in `Tests/Architecture/test_recovery_entrypoints.py` also
classifies development/vendored examples and the external read-only grep worker.
These are not silently treated as ordinary profile-persistence launchers.
