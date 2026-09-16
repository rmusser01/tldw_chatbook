# Settlement admission-phase observation

Frozen test-only change to the existing `thread_diagnostics.py` and `test_thread_diagnostics.py`, under root-authorized TASK-32562. Product admission, permission, scheduler, deadlines and runner selection are unchanged. Existing snapshot-failure diagnostic baseline 4773628967 is preserved.

Before implementation, inspected `admission_diagnostics._observe`: it takes an observer lock per call, starts its own sampler, reads registry-root aggregates, and does not isolate timing failures from original errors. Reusing it would violate this narrower contract. No changes to that module or new modules were made.

The existing runtime-settlement observer now times the actual `_Acquisition.initializing` context entry/body/exit and the existing storage `_scope`, bootstrap `startup_permission`, and storage-imported `admission_authority` binding. Context delegation uses the original type's special methods, preserves the exact entry value, exit exception triple and suppression return, and never calls exit after failed entry. Original ordinary returns and BaseExceptions are retained. Optional clock/metadata failures cannot prevent a call, change its result, or replace an exception. Stop restores all four exact saved bindings; existing runtime bindings also remain restored.

No per-call writes, additional observer locks, tracing, root/path/local-variable reads or new sampler. Fixed-key slots bound registration to 32 observed thread lifetimes without a lock; each has at most eight active nested phases and six fixed counter rows. Overflow is explicitly truncated. At most eight type/code-only errors are retained and serialized. A separate admission summary is added at existing settlement writes, including one new stage-begin record to establish the drain's counter baseline; it stays available when the product storage snapshot cannot acquire its lock. Data includes only fixed phase labels, Python thread IDs, counts, elapsed/max seconds and existing bounded error metadata. Inclusive nested timings must not be summed as exclusive time.

Limitations: these are the first 32 observed thread lifetimes, not an unlimited map of all future threads. A truncated result cannot exclude an unseen worker. Concurrent counter snapshots are observational, not an atomic authority snapshot. Active initializing-body threads are candidate initializers, not a proven same-root leader relation. Generic permission/scope labels do not claim first/post checks or measure the unwrapped native hold-ready gap. Existing stacks and phase/counter differences should guide any subsequent narrower measurement.

## Verification

- RED: seven context/nested observation cases fail in 0.79s before implementation, `/private/tmp/uat-admission-phase-red.log`.
- First GREEN seven cases; full final **46 passed in 3.11s**, `/private/tmp/uat-admission-phase-final.log`. Scope includes all preceding snapshot/runtime helper tests plus six exact context outcomes (success, entry/body/exit errors, suppression, cancellation), nested phase restoration, no per-call writes, clock/error-metadata failure identity, 35 concurrent threads and recursion beyond eight frames, bounded errors/privacy, and admission metadata while product storage lock is unavailable.
- Genuine native initialization contention test reuses existing selected/private config fixture in a disposable child with the unchanged 20-second bound. Original `_Acquisition.initializing` holds the leader while a follower waits; observation identifies both thread phases. Release lets both complete real permission, authority identity and scope checks, with original attempt cleanup. No mocked admission decision or native context.
- Measured native overhead: `/private/tmp/uat-admission-phase-overhead.py` and `.log`, 120 real admitted acquisitions with non-null native lease assertions. Three interleaved 20-acquisition batches per mode: baseline median 55.43ms, observed 54.90ms. No diagnostic file existed after the calls, proving zero per-call writes. Difference is within noise; this does not prove zero cost or Windows overhead/performance acceptance.
- Ruff and diff check pass. Bandit baseline/current: B105 1/1; B101 test assertions 134/165. No new non-assert findings.

Frozen hashes in `/private/tmp/uat-admission-phase-hashes.json`:
- thread_diagnostics.py: ad12fe0e51b91bfb456cb0ace40f07619aa3dd8f51ebf2b8921ba81df6b9e045
- test_thread_diagnostics.py: ff100f2908402c4f245bbcfec818d7e3992cd0f81da265f54e35e5ed37349c1d

Independent implementation review requested; native Windows cause remains unproven until the existing mounted support selection produces this metadata. No commits/pushes or UAT fixture mutations by this author.
