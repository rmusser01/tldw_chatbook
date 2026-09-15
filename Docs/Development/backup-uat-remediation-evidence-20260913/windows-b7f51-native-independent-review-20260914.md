# b7f51 Windows native-close audit

Exact run34932756260/job104264244610, revision `b7f51bdc20e0355dce8ac630e0c1ddbe4fcdb8f3`. Independently rehashed22 artifact-index entries and verified clean exact-revision source receipt. One installed receipt has2,867 files; all2,475 source-comparable Python files equal the recorded source and exact Git revision:2,443 Windows-CRLF equivalents and32 exact byte matches, zero other normalization/mismatches. Wheel receipt `529c64c6aa86bf1dddac478a9ad1a7afc17507d16e418739005c9f67d5454195`. This verifies retained receipts against Git, not a fresh remote filesystem. No repository edits, app boots, tests, or remote requests.

## Accounting

Native59PASS, no failures/errors/skips. Product44cases:41PASS,3FAIL, no skips or missing outcomes. Native summed case1.505s / JUnit session1.922s; product summed case1248.985s / session1250.693s. All three failures are parent `subprocess.TimeoutExpired`, not reported assertion failures:

| Case | Child limit | Recorded case time |
|---|---:|---:|
|`test_native_guidance_is_coherent_and_refreshes_after_real_settings_save`|45s|50.181s|
|`test_mounted_console_complete_capture_and_resumed_writes[settings]`|420s|462.575s|
|Same `[library]`|420s|420.295s|

Case times include fixture/process/cleanup work; do not reinterpret Settings462.575 as its configured child deadline. The matched6b201 artifact has43PASS, with Settings156.024s and Library113.166s. The new44th guidance case had no earlier selected Windows outcome. Those historical passes remain valid but do not pass the current revision.

## Guidance timeout

The exact b7 source constructs a real `TldwCli` and enters `app.run_test` before creating its focused Console/store and executing the guidance assertions. It uses the shared `_run` default45s. No guidance child phase/thread/error artifact survives in this artifact set. The timeout therefore does **not** establish whether app setup completed, any new assertion ran, a functional assertion failed, or cleanup was pending. The embedded command text is source, not execution evidence; its marker strings must not be counted as observed milestones. Keep this case failed/uncompleted. No arbitrary deadline or product fix is justified from its traceback alone.

## Settings: actual preview refusal, then no terminal child outcome

Entry records show constructor completion36.062s and `run_test` yield68.047s. Retained capture-review metadata proves actual `preview_backup → preview_capture → discover` execution after the Settings navigation path. It records two `preview_sqlite_changed` exceptions at `storage_admission.sqlite_target:1504`: memberWAL, phaseopened_state, changed fieldsmtime_ns/ctime_ns. The fresh opened descriptor metadata differs from the pre-open snapshot; this is a real guarded refusal, not hypothetical timing. No changed inode/size is reported; timestamps alone neither identify the writer nor prove contents unchanged.

Discovery records `db.chachanotes.primary` unavailable and seven dependent-owner references unavailable (chat.attachments, db.agent_runs, db.evals, notes.sync_bindings, persona.visual_identity_builtin, quiz.local, study.local). The single inventory comparison reports no scope delta; that is not evidence of a valid/complete inventory. This could not satisfy the test's `assert inventory.complete` once preview returns. The artifact does not establish whether the child reached that assertion or was still finishing preview/cleanup: its outer process was killed before reporting a primary child exception. There is no capture-start, wait-result, runtime-settlement, or archive-success evidence. Do not relabel this as a successful capture followed by a shutdown-only failure.

## Library: earlier boundary unresolved

Constructor completion35.938s and `run_test` yield68.110s are recorded. No preview/discovery log exists, and no runtime settlement, wait-failure, or archive-completion artifact exists. The script still has UI-ready/boot-drain waits, note creation, actual Library navigation/loading, F4 navigation and preview before capture; entry markers do not distinguish these. No exact last user action, successful capture, or failing drain can be established.

## Shared late activity — strong evidence, bounded attribution

All four Library main-thread snapshots execute `console_workspace_context._fit_height_to_content` at1092/1094, once via `_maybe_relabel_for_width:992`; Settings catches the same fit callback once and otherwise callback/idle processing. Late numeric CPU samples show:

| Route | Retained wall window | Main-thread CPU in window | Process CPU in window |
|---|---:|---:|---:|
|Settings|56.765s|56.453s|57.203s|
|Library|56.703s|56.406s|56.953s|

Thus this is sustained main-thread CPU consumption during the recorded window, not merely a slow sleeping child. Samples do not count individual fit invocations or prove the entire CPU cost belongs to that function. The tray source's deferred-fit branch can requeue itself when children have not obtained geometry, but these records contain no geometry/branch counts. A specific endless-layout cause still requires proving that predicate on the actual Windows path; no timer/guard removal follows automatically.

Native workers are progressing or waiting in known paths: Scheduler `_record_heartbeat → default_heartbeat_path → config/user-directory/native admission`; Settings TraceGC `current_graph_epoch` waits in native initializing, while Canvas and retention wait on config acquisition; Library retention waits in initializing and Canvas waits on config. There is no captured named lease owner, maintenance/drain-false event, or terminal `admission_timeout` in these three failures. Do not call Scheduler or TraceGC the proven retained blocker.

Slowest completed pause-request group is3.5175412s Settings (threadCPU0, processCPU3.5625s) and4.0812158s Library (threadCPU0.015625, processCPU4.078125s), each one namespace/root. These are inclusive observations amid main-thread work, not proof that token arithmetic consumes those seconds. Lower-level native work and thread scheduling are not separated here.

## Disposition

Preserve all three current failures. No tested evidence says the later63756 inspector/workbench changes fix them; neither those changes nor new runs are part of this artifact. There are two distinct actionable evidence boundaries: the actual Settings WAL pre-open race, whose writer remains unknown, and sustained UI callback work before mounted workflow completion. A finite geometry/requeue count on the already-identified tray callback would resolve its conditional feedback-loop hypothesis; the present log is insufficient to implement such a correction. Guidance needs its own actual execution boundary before treating its45s failure as a presentation regression. No deadline increase, retry, guard relaxation, cache, or success reclassification is recommended.

Verification outputs: `/private/tmp/uat-b7f51-native-independent-verify.log`, `...-blobs.log`. Bounded counts, exact failed JUnit paths, hashes, preview metadata and CPU/coordinate summaries: `/private/tmp/uat-b7f51-native-independent-summary.json`.
