# Fresh storage proof outside the coordinator: implementation plan

> **For agentic workers:** Use superpowers:executing-plans for the root-owned
> serial Native TDD sequence. This draft author performs Evidence-only work.

**Goal:** Independent issued-state observation and actual lease retirement can
complete while another original actor performs a fresh native storage proof.

**Architecture:** Reuse the existing repository proof-before-lock pattern.
Keep every native guard fresh, count acquisitions/leases at their existing
boundaries, and put only exact metadata validation/publication under the shared
coordinator. A live-hold fallback is a per-call captured dependency, never a
permission cache or an excuse to move shared table reads outside their lock.

**Tech Stack:** Actual CPython 3.12, real Windows native admission/RLock/SQLite,
local sys.monitoring START/RETURN/LINE controls; existing private-child runner.

**Spec:** TASK-34406 shared coordinator and original performance acceptance criteria;
existing Docs/superpowers/plans/2026-10-04-console-performance-fixes.md.

ADR required: yes, existing ADR126 amendment before production.
ADR path: backlog/decisions/126-complete-local-backup-and-recovery.md.
Reason: scope-proof publication crosses the shared storage ownership boundary.

## Global constraints

- No guard/callback replacement, weaker privacy/path/source check, Native cache,
  widened namespace, maintenance capability, timeout or performance-cap change.
- Preserve initial/full/final startup permission, all scope membership checks,
  native hold, accepted pause semantics and count-before-final observation.
- Preserve custom/unqualified native outcomes and exact startup readmission.
- No App or Native launch by this draft author; root owns serial qualification.
- Actual same-source evidence tables and epoch are read/written under _lock.

## Review focus

- A last independent token may retire while proof runs: unrelated close must
  finish without transferring the retired hold's continuation authority.
- A pause/cancellation may start during proof: new ordinary acquisition refuses;
  already counted exact operation work retains its established pause contract.
- Source/selector/root/path/actor changes during proof must not publish authority
  from an earlier selection, including a changed installed operation or lease.
- Another acquisition may install/retire/change a hold during proof: reread exact
  hold/names/readiness/error and the evidence identity/epoch under the coordinator.
- Native/body/close failures keep existing cleanup/error precedence and uncertain
  resource ownership; no automatic success, empty witness or unknown retirement.

## Source findings, not runtime attribution

Current storage_admission._acquire_storage calls _scope under _lock in its first
scope publication and final revalidation blocks. _scope reads _records,
_registry, _binding/fingerprint/root/path proofs, including a native registry
shared lock. Locked check() in that caller and _reuse_evidence may call
_Operation.check(path), so its internally separated native stat/resolve is still
inside the caller's outer reentrant coordinator. _repository_operation and
participants._core_getter already perform full native proof before their short
pure final coordinator fences.

Fleet stage 6 establishes a 10.141-second first get_run admission window, with
the raw connector only .062 seconds. Its selected events do not establish which
actor owns the shared coordinator or separate lock wait from native work. No
exclusive CPU, whole-Send attribution or optimization saving is claimed here.

### Task 1: Qualify actual causal controls before any production edit

**Create:** test_storage_coordinator_native_io_draft.py in EvidenceRoot.
**Test:** three private-child routes scope_before_count, scope_after_count and
operation_check. Root may run the Evidence file directly without managed copy.

- [ ] Run genuine Windows private-child source-current RED. The original opener
  must be reached under exact acquisition/record-reader or operation ancestry.
  The control suspends its unchanged body, never replaces it. Exact actor,
  installed operation/lease, root/selector/path, original code/globals/defaults,
  module origin/bytes, real RLock and original guard identities must qualify.
- [ ] While the native body is held, a second actual Thread performs a
  nonblocking coordinator acquisition, then invokes the original close on a
  separately admitted live token. Its actual close LINE reaches the coordinator.
  Require lock entry and original close completion before release. Native body
  release/join, seeded physical SQLite close and final zero census precede the
  causal assertion, so a genuine RED cannot abandon a native owner.
- [ ] Classify boundary/setup/source/cleanup failures separately from product
  RED. The scalar receipt is written before the expected causal assertions.
  Global monitoring is zero, events bounded, and local tool physically retired.

### Task 2: Narrow original proof/publication split after qualified RED

**Modify only after root authorization:** storage_admission.py checked admission
and reuse seams; add metadata helper only if it makes the existing pure fences
explicit. Do not change raw source, repository, guard or native helper APIs.

- [ ] For every currently locked attempt.check(path), obtain the unchanged full
  native operation proof outside _lock. Inside _lock recheck exact pending
  acquisition membership, PID/Thread/Task, cancel, original operation identity,
  installed owner/lease/path/hold and the accepted pause semantics. Preserve all
  selected/related-path guards; do not collapse independent proofs by memo.
- [ ] Split _scope's shared metadata dependencies from its original fresh I/O.
  Capture its exact current hold/names and startup pause-owned source/roots/
  authority identity under _lock; keep _records/_registry/_binding/fingerprint/
  effective-roots/contains proofs fresh outside. All mutable shared table reads
  stay synchronized. Native callbacks and refusal behavior remain installed.
- [ ] Record whether the unchanged live-hold continuation branch actually
  supplied mapping authority. Only that branch depends on a retained live hold:
  recheck the same current hold/names/live state before accepting its result.
  A fresh valid saved binding does not fail merely because unrelated count
  changed; a retired fallback hold cannot lend its previous mapping. Startup
  readmission retains its exact issued pause and authority checks.
- [ ] Reenter _lock after each proof. Recheck actual actor/source/current
  selection, hold/names/readiness/error and captured dependencies before creating
  or reusing a hold and publishing a token. Keep token/pending counting before
  the final fresh proof, and keep final fresh startup permission plus scope
  validation outside the coordinator with a final pure publication fence.
- [ ] Preserve _reuse_evidence's synchronized hold/evidence/path-table capture,
  counted token and per-call fresh native observation. Move its native attempt
  checks outside with exact operation/hold/evidence/epoch postproof fences; no
  evidence or scope decision crosses a later call/await/actor.

### Task 3: Refusal and compatibility evidence before acceptance

- [ ] Three causal controls GREEN with original bodies/source stable and zero
  final native counters. A last-token close cannot be replaced by a fake lease.
- [ ] Held completed-proof controls: actual pause/cancel; removed issued
  operation/lease/participant; changed owner/path; changed root/selector/source;
  retired/changed hold and changed evidence epoch/identity. No custody/result
  publication from obsolete proof. Valid independent count change still works.
- [ ] Target original repository coordinator, raw-source pause/provenance,
  storage evidence oracle, startup readmission and native close-retention cases.
  Run corresponding original POSIX checks without simulating Windows.
- [ ] Static/source review then one serialized unchanged original whole probe.
  No whole budget pass or speed claim follows from a held-control test alone.

## Alternatives rejected

Running I/O on a worker while retaining _lock still blocks retirement/main actors.
Private RLock release/restore obscures scope ownership and publication races.
Moving _scope as-is races _holds and pause/startup metadata. Caching resolved
permissions, removing checks or using faster unqualified path APIs weakens
freshness and source/privacy boundaries. A new per-profile mutex, precreated
store, broadened generic worker admission or increased budget is outside scope.

Root obtained actual native RED before amending ADR-126 and editing production.
Three final-source causal controls and six publication races now pass. Original
compatibility: 53 pass, 41 platform skips, one Windows fixture symlink privilege
failure (1314), independently source-qualified. Cross-platform matrix and whole
performance verification remain pending; no completion claim.
