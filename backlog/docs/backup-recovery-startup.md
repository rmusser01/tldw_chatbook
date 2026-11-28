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

## Core capture integration (TASK-31989)

Established ordinary admission now uses `Admission.open_existing` and shared-lock
verification of the exact unbound marker/registry identity. It does not re-register
or recreate lost evidence. Only positively initial marker creation registers the
unbound namespace. A real disjoint bound process can therefore open/write its
private owner while another namespace is under capture; an unbound process must
wait behind the capture's `bootstrap.unbound` gate.

Recovery consumers must use the fixed-authority opaque maintenance session and
hold every selected source namespace plus `bootstrap.unbound`; independently
created Admission roots cannot authorize application capture. Capture scope checks
current local bindings and directional source/staging paths and retires native
SQLite resources before scope exit. See [core qualification](backup-recovery-core-owners.md).

## Canonical initialization before mobile consolidation

When a selected canonical profile has never initialized its required Chat and
Workspace databases, first open the ordinary application with `--config PATH`
pointing to that canonical profile. Close the app completely, including its owned
workers, then request a new consolidation preview. Keep older app instances closed
through consolidation. This is an explicit operator prerequisite; consolidation
does not perform a hidden startup or reuse a preview taken before initialization.

A complete native inventory is still required, including when legacy mobile input
is absent or empty. Missing required owners or other unsupported dependencies
refuse before migration preparation. Initialization does not waive those issues.
An optional absent Runs store can be created by its journaled native owner only
where the complete canonical capture supports that absence. The rollback copy
describes the post-initialization state captured before import, not the earlier
uninitialized profile. See [ADR-244](../decisions/244-shared-console-ownership-and-mobile-supervision.md).

Use valid configured private storage parents; ordinary startup does not promise to
repair missing or unsafe parent chains. At an explicitly selected consolidation
start, previously unbound source/target profiles may be enrolled through native
writer exclusion. Keep all older clients closed. Existing bindings are checked,
never refreshed automatically. If enrollment reveals additional config-sibling
data, start returns `mobile_fresh_review_required` before preparing the migration.
Request a fresh preview, review its complete scope, and explicitly start again.
Legitimate bindings already published are retained; refusal does not delete them.

Consolidation uses the installed private recovery work directory for its encrypted
copies. A recognized default service container is permitted only for the exact
live operation's journaled output; arbitrary output paths or damaged/unknown
service contents still refuse. Do not relocate pending copies to work around a
refusal. The original source, pending evidence and failed private copies remain
available for explicit recovery.

After a completed consolidation, use the application's installed settings writer
for ordinary settings such as the theme. The cutover observer accepts a legitimate
native config publication only while source, target stores, reviewed profile
cohort and native storage authority remain unchanged. Direct config edits,
storage redirects or newly joined candidates require review and may refuse;
the observer never repairs or refreshes bindings. Pending recovery remains bound
to its original reviewed configuration. A bounded first-call refusal is not
completion: retain the operation and explicitly finish it through recovery.

An explicit pre-cutover rollback restores the authenticated canonical snapshot
and retains displaced current objects. Restarting it requires the original
rollback password and unchanged native physical/configuration authority. It does
not import or overwrite current legacy source content, and its original source
fingerprint identifies the operation rather than a new semantic inspection.
A later re-import still requires a fresh review and the supported unchanged-source
successor conditions; rollback does not approve changed history or mark cutover.

## Bounded first-enrollment review

Mobile consolidation must complete the entire installed inventory before first
profile binding. Initial read-only discovery and later reviews under maintenance
share one materialization allowance, including fixed SQLite reads and retained
files. A resource refusal does not approve a partial inventory or initialize
missing source stores. Resolve the reported review problem and take a fresh
preview; do not redirect the pending operation or automatically retry it. The
native copy and SQLite limits remain in force independently. ADR244 R121 records
this boundary; populated and no-data qualification are reported separately.

## Mobile operation and native proof limits

Current production local mobile start or recovery has one fixed 4 GiB cumulative
representation allowance for the complete operator workflow. Its original native
retained-history/input/proof work still shares a separate 16 MiB sub-bound; those
charges also consume the operation allowance. This is neither a per-file limit nor
a memory-use promise. Generic trace/Notes and private native defaults remain
16 MiB. The complete ordinary cutover observer uses one fixed 64 MiB coordinator
allowance, with 512 bytes reserved at issuance (R146). SQLite/archive/selector
limits remain independently enforced. No option changes these bounds and no
automatic retry renews them.

On a fixed `mobile_materialization_limit` refusal, preserve source and recovery
files and review the actual available recovery action. Initial failure and a
later explicit same-operation recovery are reported separately. Required missing
or unreadable credential storage still refuses capture; an empty readable test
store is only test isolation. ADR244 R122 and R124 record the implemented accounting
boundary. Explicit human authorization subsequently covered R129/R130 code and
isolated tests. R130 now prices terminal representations before target effects
and preserves all actual late charges. The retained full private cold case still
refuses at its native 16 MiB limit; it is not claimed to have private-path parity.
The same Workspace durable-commit interruption finishes without replay through
the actual public service in a fresh process, preserving receipts, rows, exclusions,
holds, original captures and source bytes.


A reviewed retry after verified mobile rollback must use the unchanged source and
same physically restored destination under a new operation ID. Original claims,
locks and displaced data remain retained. Linked ancestor evidence is checked
again under the same live source lock; a failed check never permits a redirect or
claim replacement. After committed cutover, restoring the canonical copy uses a
new current-target review and a new encrypted safety copy, not the old import's
pre-cutover rollback action. Ciphertext and plaintext archive hashes have distinct
roles; a retained file listing alone is not unlocked restore authority. ADR244
R125 records these boundaries. Fresh rollback-to-reviewed-successor and full
canonical-copy/SafetyCopy restoration now pass. The ordinary observer accepts
only authenticated direct restoration, including subsequent native data and theme
saves; source-claim, activation, manifest, plan, inode, schema and pending tampering
remain refusing. Included unchanged packaged artwork passes the same full route
without profile rebinding or package writes. Notes absent/present-marker cold
startup and one scripted mounted manual turn retain imported execution holds.
These are scoped local proofs, not phone, real-provider, active-sync or hardware
qualification; the complete Task8 review remains separate.

## Historical R126–R134 qualification notes

The following entries preserve the limits, authorization state and failures at
those earlier checkpoints. Their statements about pending approval, 2 GiB public
capacity or unfinished restoration/Notes qualification are historical and are
superseded by the current status above. They are not current operating guidance.

During public consolidation's initial read-only input planning, the native
journal lock remains held until the captured readers physically retire. Status
queries that read that journal can wait; the service's current progress remains
available. The lock ends before target writes. This does not change the fixed
2GiB public allowance, original16MiB native proof bound or ordinary defaults
(ADR244 R126). A budget refusal remains explicit recovery-required evidence.

Public mobile receipt, row and recovery-history checks use the existing journal
lock only for their pure verification bodies, then release it before further
writes or owner retirement (ADR244 R127). All native checks and the fixed
2GiB/16MiB pair remain. Before-effect capacity-tail qualification is still open;
a successful verification frame does not establish that guarantee.


ADR244 R128 narrows the new lock lifetime to each of the two fixed native owner
transactions: after owner construction, through its SQL commit/rollback, and ending
before physical owner retirement and sidecar/owner journal publication. Pure
verification reuses only this exact current frame. The public2GiB/native16MiB
limits and every native check remain. Successful transaction qualification still
does not establish the outstanding before-write capacity-tail guarantee.


ADR244 R129 defines ordinary gate support for a direct native restoration of this
operation's canonical recovery copy. It authenticates the actual committed new
replacement generation and current physical/schema state; restored pre-import rows
do not falsely count as imported receipts. Source consumption stays immutable and
owner execution still requires its independent review. Other recovery-copy histories
are not automatically accepted. This contract awaits explicit human authorization after automatic review refused
the observer edit, followed by implementation and same-target data/config-write/
tamper qualification. Current ordinary admission still refuses that restored target.


The reviewed R130 contract corrects the earlier whole-future-import cost
requirement: it calls for terminal local representations to be priced before target
effects, while every later materialization still checks its actual allowance.
Its early-pricing implementation and proposed fixed 4 GiB public envelope are
blocked pending explicit human authorization. Current production remains 2 GiB,
with original native 16 MiB/ordinary 16 MiB and all validations unchanged. Late
refusal retains pending/quarantine until native retirement or process exit;
recovery is an explicit new same-operation call. Strong complete rollback
preflight remains unchanged.

The R131 implementation contract permits a later mobile canonical-copy replacement
to preserve unchanged referenced installed builtin assets, verified against both
the archive and current native inventory. Package assets are never write targets.
Changed package bytes/configuration or unsupported historical shapes refuse. Native
copy/digest accounting retains the ordinary16MiB bound and native resource limits;
this is not a new restoration authority or an increased public capacity. Runtime
qualification of this newly approved branch is still pending.

R131 rejection chronology: the initial copy/digest edit was rejected before
execution, then the human explicitly approved that exact code and isolated-test
scope. Its optional accounting/current-member controls now pass narrow native
component tests. R134's separate existing-profile namespace integration was
subsequently rejected, then separately received explicit human approval. Its native registration/refusal controls pass. An earlier public run stopped at
the original SQLite resource guard; a later fresh run completed public import
and recovery-copy preview. Replacement now refuses at ordinary16MiB namespace
composition before staging. No complete included-builtin replacement or Safety
Copy is claimed; R129/R130 remain blocked.

Notes qualification correction (R133): the earlier R132 capture-refusal claim
was based on a validator that the mobile capture path does not call. Do not
interpret it as a demonstrated unsupported profile. A fresh native control now
qualifies both captures with historical linked-note fields and an external private
file: the database metadata is captured, while the external file is not included
or activated. Source DBs and that external file stayed unchanged. Public import
and natural Notes marker/candidate visibility remain unqualified. Generic replacement semantic checks remain unchanged; no source
field stripping, invented legacy owner or automatic file activation is permitted.

R131 authorization update: the exact bundled-artwork code and isolated tests now
have explicit human approval. Implementation/qualification is in progress through
normal tool review; this does not yet change the documented runtime support. The
original rejected attempt remains recorded. R129/R130 remain separately blocked.

R134 implementation clarification: an already enrolled profile may reference unchanged shipped artwork without owning the application's package directory. Public consolidation acquires the actual reviewed native capture source namespaces before maintenance while keeping its existing profile binding unchanged. Mobile-copy recovery uses that same native source exclusion for readonly artwork, with complete archive/member/current-profile validation. This implementation is being qualified; the observed existing-bound refusal and component-only results are retained. No package modification, tool permission or capacity change is authorized.

R134 execution status: the initial automatic-review rejection preceded process creation; the exact registration/holding scope later received explicit human approval. Native implementation and targeted tests are in progress. Existing profile records, package contents and source bytes remained unchanged in the latest closed composition controls; their target facts also remained unchanged. Earlier whole-public runs legitimately changed canonical targets through import, so those epochs do not establish target-byte equality. Current public capacity remains2GiB; R129/R130 remain stopped.

R134 authorization update: the user explicitly approved the exact native backup-source namespace registration/holding code and isolated tests for already-bound profiles. The earlier automatic-review rejection remains recorded. Implementation proceeds through normal review with unchanged profile records, agent permissions, public2GiB/native16MiB limits and existing physical admission checks. R129/R130 remain awaiting separate authorization.
