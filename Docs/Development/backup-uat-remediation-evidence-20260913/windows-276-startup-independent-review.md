# Windows276 startup: independent timing and stack review

**The60s startup criterion still fails. This run does not reproduce cfb5's final writer-held publication stage.** The new timing sample records an earlier unfinished config-lock acquisition, but the final stack shows that worker idle and the main thread performing a Canvas policy admission. No corrective product change is proved.

Run34920997447/job104228893861, exact27603954403930eb1ee9551995aef784dd97b9c8. Full saved log SHA256 independently verified: `dde7249eae2e53262563d4c03fc6a38da136a681d2c3418ef7cad8817afb03c5`. No edits, app boots or test reruns.

Verified accounting:31non-UI passed10.07s; four separate primary full-app cases timed out under unchanged60s deadlines. Separate pinned dev4631 diagnostic passed49.03s with1warning; candidate timed out. All8profiles (4per source) have source_matches=true, no profiler/ConfigTimings errors, and no dropped active records. Source matches are not installed package receipts.

## New measurements, with their actual bounds

The five selected config code objects attach after collection. Counts apply to that observed interval, not imports before collection. `config_lock` running means its contextmanager has not yet reached the first yield; acquisition includes native admission plus lock setup, not only mutex waiting. Body-and-release includes native validation/cleanup. No active thread-CPU estimate exists.

At02:31:41.619 (candidate observer elapsed62.953s), one transaction and config_lock frame on native thread8620 are active for7.188033/7.187950s, both phase `running`. No publication or settings-rebuild entry has been observed in the candidate by that sample. This is evidence of pre-yield acquisition delay, not successful settings replacement, lock ownership, post-write reload or7.188s pure mutex contention.

The candidate records16 completed user_directory calls: median0.432424s wall/.421875s threadCPU; minimum0.397542s. Fifteen run on main1560 and one on1104. The slowest main call is4.599254s wall/.484375s ownCPU, and the other-thread call1.488126s/.78125s. The outlier's non-CPU portion is real elapsed time not charged to that thread; the record cannot distinguish scheduling/GIL delay, another lock, or native I/O. Most roughly0.4s completions also consume roughly0.4s CPU; this shows synchronous work but not a dispensable native check.

The dev diagnostic records one completed transaction0.723258s; its config-lock0.723122s divides into0.092964s acquisition and0.630157s body-and-release. It also observes publication/settings rebuild. Candidate's unfinished acquisition and dev's completed transaction are different extents; a ratio would not be a same-work performance measurement. Dev26user-directory entries versus candidate16 do not identify interchangeable callsites or arguments. ConfigTimings stores only16 latest completions, while counts/slowest retain separate finite summaries.

## Every final candidate thread

The final timeout dump begins02:31:45.433, approximately3.81s **after** the last timing sample. All11thread stacks were read:

- All7executor workers (asyncio0–6, including8620 and1104) are idle in the pool queue. Thus the earlier active transaction cannot be called the lock holder at this cutoff. Its terminal return/unwind outcome is absent from the saved samples; it may have completed or failed during that gap.
- MainThread1560 executes `_watch_canvas_policy:2992` → `_canvas_enabled:2845` → live Canvas getter/config bootstrap → checked config/raw operation → acquire_storage/_acquire_storage907 → admission_authority. It is not waiting in CompactModelBar nor shown in rail publication.
- ui-stall-persist8456 waits for queue data; storage monitor5540 waits on its Event; startup profiler1536 waits on its sampling Event. No reciprocal wait cycle is demonstrated.

The printed innermost pair is admission_authority line137 (`return _existing_admission_authority(...)`) followed by pathlib.exists/stat, which is not that source statement's direct call chain. Thread-stack formatting is not an atomic freeze of all frame lines; do not infer a missing marker, a particular exists check or a new code defect from that juxtaposition. The broader watcher→native admission path is consistent; precise leaf attribution is unavailable.

## Disposition and smallest additional evidence

No source-backed guard-preserving correction is established. The required inactive Canvas watcher must remain: existing ADR121 and mounted lifecycle regression explicitly require pre-preview external-disable observation. The prior same-view binding correction has separate native proof; these new records do not show it failed or prove the remaining60s criterion is satisfied.

If one further diagnostic is authorized, instrument the existing **config_lock acquisition observation**, rather than change production: when the sampler sees a running config-lock frame, include bounded filename/function/line-only stacks for that recorded native thread and main in the same record. This would distinguish `_settings_rebuild_lock`/FILE waits from native admission/qualification at the actually measured7s interval. Retain the selected transaction/config-lock return-or-unwind outcome in bounded diagnostic output (with failures preserving the original result) before a later process timeout can lose it. No argument/local/config contents, native probes, function replacement, new product waits or altered deadline are necessary. Such evidence is still observational and must acknowledge non-atomic frame sampling; it would close the specific timing-to-terminal-stack gap seen here.

Current results and pending new native passes remain distinct. A sample in a costly function is not proof that its checks can safely be removed. No cache, wider guarded lifetime, polling suppression, timeout increase or speculative UI edit is recommended.

Full log: `/private/tmp/uat-276-windows-startup-job.log`. Semantics reviewed in `/private/tmp/uat-startup-config-timing-review.md`. Extracted8profiles,16completion statistics and all11terminal stack summaries: `/private/tmp/uat-276-startup-independent-summary.json`.
