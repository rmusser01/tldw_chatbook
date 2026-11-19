# K5 implementation and release qualification

Base: `7d79700e025310e43ab9537ee65bb1792cfe2ba1`, branch `codex/workspace-project-keyword-index`, managed worktree only. Original checkout, user data and Library remain untouched. Root owns task/plan/spec/ADR/QA metadata; this report and private receipts are not product commit paths. Current source is5d927a98ffc695c281465b09a8020d1db196490a. Covering/fix checks and all six original canonical lexical500ms gates PASS. Broader admitted common-query500ms still fails and is an explicitly accepted unmet deviation; all earlier failures remain below. Root/final review owns task completion. Earlier chronological receipts remain in `task-5-report-history.md`; pending statements there are historical, not current status.

## Final implementation and contracts

Settings has an optional index card for each exact local folder binding. Constructors/compose accept immutable workspace/binding/root inputs and perform no I/O. Bounded background operations derive the runtime key, read management state and execute Enable, Refresh, Pause, Resume, Disable or Clear with expected scope/job/revision. Mounted visit/request/binding fences reject late completions; durable CAS rejects queued Enable after Clear and old Clear after newer Enable. Building polls have at most one pending operation and stop outside Building. Counts remain discovered/indexed, storage approximate, and the timestamp is truthfully **Last scan activity**, not last successful completion.

Blank folder/suffix inputs mean all eligible UTF-8 text. Empty and partly empty saved tuples round-trip without widening/narrowing. Default all/all includes extensionless files and manifests. Mandatory processing/sensitive/binding exclusions remain. SSH and unavailable authority are truthful unavailable states. Console uses its existing OperationLane and selected binding generation; its action waits for modal retirement/dismissal before the existing Workspaces Settings category route. No new navigation, Library scope, semantic control, CSS/token or dependency was introduced.

`policy(scope)` is an admitted exact-scope read. `management_snapshot(key)` returns `(status, policy, requires_upgrade)` through a validated nonmigrating open. Old-schema state cannot appear query-ready; Settings can Clear the current v2 cache even when migration headroom is insufficient. Clear enters this cleanup path before revocation. Service stop/snapshot and runtime composition snapshot acquire their locks off-loop inside complete shielded close tasks; cancellation waits for actual retirement.

Finite index-specific registry scopes span writer/supervisor, direct service/key operations, the whole `fs_search_index` public invocation before its first guard through final result gates, controller preparation and UI reads. Newly opened handles require positive native closure AND registered-core retirement of the exact handle; borrowed caller handles retain caller ownership. Generic SQLite close behavior is unchanged. Actual swallowed/denied close remains counted and raises `ProjectIndexRetirementUnproven`. Writer/supervisor cannot release retirement flags/pins/lifetimes early. An outer provider registry lease may outlive the inner Capture/cache pin because its own counted lease independently blocks maintenance, and must close before successful return.

Index-only wire v2 supports bounded multi-directory units. Aggregate entries, directory attempts, raw iterator inspections and pending paths are each <=256; source/scratch <=4MiB, frames <=2MiB, reads <=2 (observed1). The worker descends new children without collecting a tree. Each unit receives fresh durable job/binding/authority/exclusions admission. Chunks and whole-unit digest prepare outside the final registry->store atomic publication. Entries publish before completion receipts; only complete directories prune. Production eligible seeds are fetched after previous publication, avoiding repeated short frontiers before commit. Singleton APIs remain compatible adapters.

Schema3 adds indexed durable attempt revision, replacing the tree-sized attempted set. Terminal incomplete directories retry once under explicit new job/Resume revision; continuations stay eligible. The same FTS indexes path plus text; valid empty/whitespace captures with no ordinary chunks get one zero-length path chunk. All search ranking, live identity/hash validation, current exclusions and disclosure limits remain.

File-backed003 preserves state and SQL-backfills path FTS. Migration alone acquires exact binding native capability outside SQL/fences, counts all namespaces/sidecars, reserves DELETE journal plus1MiB headroom and sets bounded native pages before atomic schema recheck/migration. Capacity/SQLITE_FULL refusal leaves validated v2 intact and Clear-able. Currentv3 skips this capability. Unproven failed-constructor close retains at most one `(store, capability)` per affected binding; later construction refuses and close reports unresolved. No authority cache, token broker, foreign-thread close or new owner framework.

Exact29 product/test/doc paths are in `k5-owned-paths.txt`. Architecture documentation covers ownership, batching/migration, nonmigrating management, blank=all and limits; Console user guide explains opt-in management and explicit Refresh/Resume. ADR240 and all scope amendments were recorded by root before implementation.

## Verification commands and outcomes

All commands run from the managed worktree with `.venv/Scripts/python.exe`, task-owned TEMP/TMP/TMPDIR, existing private subprocess isolation and blocked socket fixtures. Receipt names below are relative to this report directory unless explicitly prefixed. No full suite, dependency install, real provider call, push or merge.

Final covering command (result appended below):

```text
python -m pytest Tests/UI/test_workspace_project_index_settings.py Tests/UI/test_console_project_index_status.py Tests/Agents/test_project_index_tool.py Tests/Tools/test_project_index_scan_worker.py Tests/Workspaces/test_project_index_scanner.py Tests/Workspaces/test_project_index_service.py Tests/Workspaces/test_project_index_recovery.py Tests/DB/test_workspace_project_index_db.py -k "not symlink" -q --tb=short
```

`not symlink` mistakenly excluded our native index junction regression based on conflation with a separate earlier K4 symlink privilege failure. This rationale was incorrect. The exact omitted node `Tests/Workspaces/test_project_index_scanner.py::test_symlink_or_junction_escape_is_never_captured` was run on final committed source: `k5-final-junction.txt`, 1passed1warning2.18s. Windows mklink /J succeeded; scan and direct Capture refused the escape and outside content stayed intact. This closes the omission without repeating the suite. No POSIX claim.

Meaningful TDD and corrected focused gates:

| Receipt | Actual result / meaning |
|---|---|
| task-5-red.txt; task-5-console-red.txt |9 missing-card/status failures plus1 isolation setup error; corrected Console1 intended missing-action failure10.16s |
| k5-registry-retirement-red.txt |5 intended failures72.25s, actual writer/supervisor registry native ownership leak |
| k5-query-unwrapped-red4.txt |1 actual query-thread registry leak35.50s after removing a test wrapper that concealed it |
| k5-strict-scope-red.txt; k5-strict-native-green.txt |missing strict API RED;9pass74deselected173.12s, real held outer native close, swallowed/denied closure, borrowed handles, writer/supervisor retention, encrypted recovery |
| k5-preparation-native-green.txt |1pass17deselected15.59s, actual controller registry retirement |
| k5-batch-red.txt; k5-batch-scanner-green.txt |4 intended absent DTO import errors;3pass26deselected3.06s: tiny nested unit,300emptydirs,600widefiles |
| k5-path-keyword-red.txt |3 intended absent filename/path candidates4.74s |
| k5-migration-red.txt; k5-migration-initial-green.txt |5pass/1 missing-runtime-context failure;6pass22deselected11.29s, v2 backfill/state/capacity/cleanup/nativebusy/currentv3 |
| k5-batch-boundary-green.txt |13pass76deselected68.82s: next-unit remote Pause/Disable/Clear, actual filename/empty/whitespace native Capture, changed/deleted/excluded omission, closed wire bounds |
| k5-management-policy-red.txt |5 intended default/empty-roundtrip/missing-management failures50.05s |
| k5-close-migration-red.txt |1 actual held003 migration heartbeat freeze14.37s |
| k5-management-close-green.txt |held-migration close passes;5 UI initializer NameErrors retained as implementation failures |
| k5-management-policy-green2.txt |12pass92deselected104.93s, actual mounted v2 pressure->Clear without003 allocation, empty/partlyempty policy, extensionless/manifests, mandatory exclusions, shielded composition cancellation, corrected legacy/frontier fixtures |
| k5-ui-bounded-final.txt |7pass10deselected63.64s, queued stale Enable, unavailable/SSH/busy, Console themes/focus/binding fence/Settings route |
| k5-final-controller-preparation.txt |exact changed controller preparation node1pass25.90s; off-loop plus actual native drain |

The exact controller command is `python -m pytest Tests/Chat/test_console_workspace_folder_note_contract.py::test_index_scope_preparation_runs_off_loop_and_note_tracks_final_tools -q --tb=short`. Prior `-k project_index` did not select that node; the separate exact run closes that gap.

Adverse diagnostics remain visible. Initial UI fixtures had deferred mount waiting, Textual field initialization, DACL and narrow source encoding issues. Horizontal compact actions really clipped Clear (2of16columns); existing vertical compact classes fixed it. `k5-ui-operation-error.txt` records real operation_path_outside_scope from an inappropriate repository operation scope; finite connection ownership grants no substituted authority. Provider close red/red2/red3 include fixture class targeting corrections AND a real public `invoke()` integration bypass, corrected by routing through public `invoke_detailed`. Recovery red1-3 include config/owner fixture setup mistakes; red4 is the actual leak. `k5-batch-service-initial.txt`4pass3fail199.70s: old second-directory barriers no longer reached after batching;256root entries now force a second unit. `k5-native-migration-query.txt` passed actual failed-native-close retention; a subsequent exclusion fixture had wrong DTO shape. `k5-batch-core-covering.txt`77pass3fail3skip3deselected156.78s exposed stale downgrade/frontier fixtures, subsequently corrected and covered. None of these are silently represented as all-green.

## Mounted UI and actual app journeys

Final mounted real Tldw Settings children cover dark80x24 and light170x48; theme/current_theme.dark is forced/asserted after startup and again before pixels. Each action receives focus, scroll-into-view and painted bounds before Enter. The same Ready card/request remains stable after2.2s with zero pending polls; waiting is relevant-pane bounded, not global app workers (unrelated startup/recompose workers otherwise obscure completion). Saved Library configuration repr is unchanged. No callback/recompose loop or leaked card poll worker was observed.

Final images are `k5-settings-textual-dark-80.{svg,png}`, `k5-settings-textual-light-170.{svg,png}`, `k5-settings-clear-textual-dark-80.{svg,png}`, `k5-settings-clear-textual-light-170.{svg,png}`. `k5-final-mounted-render.json` pairs exact SVG and rendered PNG SHA256/time, using existing headless Edge with external requests aborted. All four pixels inspected; light palette, blank-all labels and full compact actions are correct. Clear screenshots are focus/completion-frame proof; prior action visibility may remain until repaint, so visual Off-controls are not claimed from those pixels. Mounted state/control assertions are separate. Prior requested-light/startup-overridden-dark image is retained as adverse setup evidence. Console dark/light80/170 pixels and stable binding/action assertions are separate retained receipts. No stylesheet changed, so bundle-sync rebuild was unnecessary.

Two sequential fresh headless real-app-factory journey processes use caller-owned `.superpowers/k5-scale/journey-profile`. `k5-app-journey-stage1.txt` is an initial missing-config-root collection failure. Corrected stage1-run2 passes1test64.69s:300files built/queried, held real Unit publicPause retires child38692 in2.7835543s, ResumeReady, second Pause retires35976 in2.5350615s, leavePaused. Stage2 passes1test58.81s: different PID16508 versus29996, same key/profile, explicit persisted Resume; modify/delete+Refresh gives299files, changed marker hits/deleted omitted; detach a separate actual production-admitted binding while Capturechild26968 is held, then unavailable/nohits and child/pipes retired. Clear original binding gives Off/indexed0. Both stage JSONs show all owners retired/zero searches. This supplements mounted UI, not interactive terminal or human approval-UI qualification.

Real recovery holds an actual Capture child after read while its admitted pin blocks maintenance preflight. Native release proves PID/pipes/pin retirement and successful local drain; query process exits completely. Parent performs actual encrypted rollback, real ActivationStore approval of exported OWNER_ID in each paired generation, then a fresh child refuses the old serialized key/scope. Valid original config exists before real plan fingerprinting; no manufactured identity or post-publication rewrite. This is staged query-native->fullchildexit->actualrollback->fresholdkeyrefusal, not a simultaneous live-service restore. Focused strict9GREEN already covers it; final raw recovery receipts are retained separately below.

## Scale, profile provenance and measured costs

`.superpowers/k5-scale` is freshly task-created/owned and outside pytest retention. Root explicitly recorded this ownership exception. Original20k nested corpus is retained unchanged. The first cache accidentally lived in a private pytest profile; it is a partial/adverse run, not completion. Public Pause selected exact original config/bootstrap/current scope/job;1pass13.57s, runner failed Ready after Paused at~1793s, actual parent/interpreter/scanner PIDs13780/23636/11172/29660/25440 gone. `original-run-{before-pause,pause-command,pids,retirement}.json` and original run log preserve it. No live DB/profile copy, corpus reconstruction or synthetic completion.

Two fresh stable-profile preflights pass16.27s/17.14s with identical config/bootstrap/DataRoot/binding/authority `5165c8b497644cfa3814c0f1f48275a9288826435642e0fff8cc36a6d945919c` and actual native retirement. TLDW_TEST_CONFIG_ROOT is selected before imports, bootstrap_profile avoids private tmp-path redirect, real get_user_data_dir feeds app_factory. DataRoot is `stable-profile/home/.local/share/tldw_cli/default_user`.

Matched12-file production generator has g000+12leafdirs:13subfolders,14root-inclusive receipts,25entries. All profiles use actual service/runtime/scanner/store/tool, empty cache and mandatory guards. Old polled/competing writer38.058s, pipeline42.891s; caller11status28.104s. Old passive idle pipeline32.594s/writer27.558s;28admitted17.349s,14admit3.569s,14commit3.617s,5frontier0.890s,14scanner0.083s;220storage22.950s,161core11.093s,58activation1.244s; caller3status4.931s. Those nested inclusive costs are not additive.

New same-shape passive pipeline7.438s/writer2.9024496s;1unit0.1228067s,2admitted1.2310574s,1admit0.2662798s,1commit0.2244164s,2eligible0.4202835s,1finalpending0.0896188s,prepare0.0126066s,disk0.0003338s;23storage2.1528554s,16core1.0507768s,6activation0.1209496s;caller3status5.0023007s. Ready12/manifest1/deepmarker/nativealltrue. `phase-profile-{wall,idle,batch-v3}/{phases,nested-pipeline}.json` retains raw costs. `k5-batch-phase-source.json` has exact source/load; the historical pipeline JSON's generic 'may compete' text is preserved and qualified by the actual idle source receipt. Source also gained strict cleanup/pathFTS; this is end-to-end source-qualified comparison, not isolated single-line attribution.

Both literal full builds are NEW builds under the preflighted stable profile, not Resume. Product/core source stayed frozen until BOTH physically retired. `full-build-source.json`, `wide-build-source.json`, and `load-events.jsonl` record exact file hashes/start and test/load intervals. No parent tests competed. Full-build timing is concurrent-load, separate from final idle latency.

| Raw metric | Nested | Wide |
|---|---:|---:|
| UTF8files / subfolders |20,000 /20,200 (200groups+20,000leaves) |20,000 /0 (one directory) |
| State / discovered / indexed / manifests |Ready /40,200 /20,000 /329 |Ready /20,000 /20,000 /79 |
| Pipeline seconds / files per second |705.188 /28.361231331 |448.312 /44.611788219 |
| Whole test seconds |735.81 |512.00 |
| Baseline parent RSS bytes |221,294,592 |221,380,608 |
| Sampled parent peak bytes |231,579,648 |230,461,440 |
| Sampled combined peak bytes |258,265,088 |256,032,768 |
| Extra combined RSS bytes |36,970,496 |34,652,160 |
| Whole-binding peak/final bytes |19,853,312 |11,186,176 |
| Max frame bytes |33,646 |43,566 |
| Max queued/source bytes |2,167,424 |2,172,160 |
| Active reads |1 |1 |
| Actual native child gone |13028 |34096 |

RSS/disk peaks are sampled every1s, not exact allocation peaks. Whole-binding accounting includes all namespaces/sidecars; final payload is DB plus zero-byte native lock, no residual sidecars. Raw `nested-*`/`wide-*` launch/progress/pipeline/service-retired JSONs prove deep final marker, exact Ready counts, worker/pipes/SQLite/registry/pin/native retirement and serviceclosed/zero searches. Wide progress448.296s precedes final pipeline448.312s; report uses final. Normal provider roots/guard are test-seeded fixtures, explicitly separate from actual runtime registry/admission/native Capture and separate production controller/root tests.

Keyword works with chromadb/sentence_transformers/faiss imports unavailable via sys.modules None; no package uninstall claim. Inherited socket guard BLOCKED with no socket attempts. No model/provider/network access. Literal corpus files are task-generated, never original user folders.

## Static checks, self-review and limits

`k5-final-ruff.txt` reports99 findings on large shared files; `k5-final-ruff-baseline.json` compares code/message/source-line counters to exactBASE: provider23,controller60,UIworkspace3,Settings13,0new. New/index/test files pass. Final formatter17files clean. `k5-format-source-equivalence.json` records two post-build format-only wraps in worker/recoverytest: exact before/after byte hashes differ, parsed AST hashes match. Old scale evidence therefore qualifies equivalent behavior, not byte-identical final source. `git diff --check` passes; CRLF conversion warnings are not whitespace errors.

Governance command `python -m pytest Tests/UI/test_design_token_governance.py Tests/UI/test_component_pattern_governance.py Tests/Chat/test_console_workspace_folder_note_contract.py -k "design or component or project_index" -q --tb=short` actually gives14failed12passed18deselected393.96s. Four Windows separator cases; one locale-default cp1252 source decode; seven pin-helper subprocess import failures; two unchanged style/size violations. `k5-governance-baseline.json` proves103 relevant inputs equalBASE after UTF8/LF normalization, newcard0stylewrites. This supports unchanged-source attribution, NOT exactBASE runtime reproduction or all-green governance. No unrelated governance/CSS fix or repeat sweep.

Self-review checked exact selected authority/CAS, final registry->store fences, aggregate worker limits, no tree-sized attempt memory, entries-before-receipts, live filename evidence, migration rollback/capacity/native lease, nonmigrating UI Clear, close cancellation/positive retirement, theme/focus and owned diff. No extra lifecycle owner, retained authority interval or dependency. Existing completed `_Run` history/exception retention M1 remains an earlier deferred shared-file concern for final whole-branch review; K5 does not silently broaden it. Windows qualification does not establish POSIX behavior, exact native memory maxima, real human approval UI or semantic indexing. Independent root review remains required before task Done.

## Source-complete review checkpoint

Committed exact29 owned product/test/doc paths at `0295b820b5687c26020561a5a83fa4d743d877b8`, BASE `7d79700e025310e43ab9537ee65bb1792cfe2ba1`. `k5-owned-paths.txt` is the exact staged/committed list, verified against git before commit. Root metadata/QA remain unstaged. Commit message: `feat(settings): manage optional workspace keyword indexes`.

Source is complete for independent read-only review. Final eight-file covering run is STILL ACTIVE, last observed72passing tests/38%, no failure shown; final count is not yet asserted. Final idle candidate/live validation measurement remains pending that run's retirement. Both literal full builds are complete with real native retirement; strict cleanup, mounted/default/Clear/close and app journeys have focused GREEN receipts as above. This checkpoint is not task-Done, final verification PASS, or permission to merge. No source fixes will be made before the complete review list and scoped review loop.


## Final covering finding: factory compatibility

While the final covering process is still active, retained `k5-final-shutdown-factory-failure.xml` records `test_real_app_shutdown_without_console_drains_index_writer` failing at runtime.py:313: a pre-existing app lifecycle test injects `Rig.service` with a plain function factory; the new runtime close calls `service._factory.assert_retired()` unconditionally. `await service.aclose()` completed before that AttributeError. Production `_AdmittedStoreFactory` supplies the method, but this is a new integration compatibility assumption, not a baseline-attributed failure. No fix has been applied; source remains0295b820b5 pending complete independent review/scoped loop. Final aggregate run result remains pending.

## Authorized fix round1 (BASE0295b820b5687c26020561a5a83fa4d743d877b8)

Independent complete review identified I1 invalid policy drafts blocking Clear/Disable, I2 app-shutdown fixture adaptation, and Minor M1 scanner annotations/docs (deferred by root). Only card, card tests and one service test changed. Production runtime/store/scanner/provider/backend bytes remain unchanged.

I1: `_pressed` now reads/validates drafts only for Enable/Apply. All other actions retain admitted saved `self.policy`; the same captured scope/job/revision is posted. New mounted cases use invalid suffix `py` and prefix `../outside`, verify Clear/Disable actual service effect and exact captured scope/job/policy, and retain Enable/Apply rejection. `k5-fix1-policy-red.txt` is4test setup failures72.94s from my unsupported Textual attribute selector, not intended RED. Corrected existing Button.name iteration yields intended `k5-fix1-policy-red2.txt`:2failed2passed19deselected81.06s, Clear/Disable dispatch0 vs1, Enable/Apply validation passes. Command: `python -m pytest Tests/UI/test_workspace_project_index_settings.py -k invalid_policy_draft -q --tb=short`. Complete post-fix card file is running separately; original covering's earlier UI subset is pre-fix evidence.

I2 classification is clarified by complete review: standalone ProjectIndexService supports callable-only factories, but direct mutation of private app._project_index_service is not a public runtime composition contract. The production composition always supplies _AdmittedStoreFactory. Only the existing lifecycle fixture is adapted with a callable delegating to Rig.store and an honest assert_retired which checks serviceclosed, real scanner/pipes, every DB/registry/native flag and stopped writer/supervisor. Proof is not reached while the real writer is held and is reached after complete actual app shutdown. Original no-Console and physical retirement assertions remain. Production unconditional factory proof is preserved, with no optional getattr/no-op fallback.

`k5-fix1-close-green.txt`:5passed1warning131.40s. Exact command:

```text
python -m pytest Tests/Workspaces/test_project_index_service.py::test_real_app_shutdown_without_console_drains_index_writer Tests/Workspaces/test_project_index_service.py::test_shutdown_retains_lifetime_through_native_db_closure_and_repeated_cancel Tests/Workspaces/test_project_index_service.py::test_close_stays_responsive_through_held_constructor_migration Tests/Workspaces/test_project_index_service.py::test_runtime_close_shields_composition_snapshot_and_actual_drain Tests/DB/test_workspace_project_index_db.py::test_failed_migration_native_close_retains_count_and_binding_capability -q --tb=short
```

The actual failed app node now passes76.84s; repeated cancellation/native DB close21.57s; held constructor migration responsiveness19.08s; composition shield9.60s; production failed-native-migration-close counted capability proof2.68s. This supersedes the original known I2 failure without a duplicate whole covering run. These tests overlap original covering/new UI tests; they are correctness receipts, not idle timing.

## Final original covering result

`k5-final-index-covering.txt` completed: **1failed181passed3skipped1deselected1warning2946.52s**. The sole failure is I2, superseded by the exact post-fix shutdown/close5GREEN above. The skipped scanner cases are Windows newline filenames, POSIX permission-bit behavior (Windows ACL-denial is not qualified), and Windows retained no-delete-share ancestor handles preventing rename. The one deselected native junction node separately passed on the final backend. Original UI cases preceded I1; the complete post-fix card file is their required superseding receipt. Backend/provider/runtime/scanner/store source remains unchanged; no repeat full covering or corpora.

Final actual staged recovery passed70.26s. Stable copies: `k5-final-recovery-query-retired.json`, `k5-final-recovery-query-child.log`, `k5-final-recovery-old-key-child.log`, `k5-final-recovery-old-key-refused`, `k5-final-recovery-retention.json`. Actual held CapturePID34208 gone; maintenance blocked while pin held, pin closes after native exit, pipes closed and drain succeeds. The retention JSON records exact original hashes, both durable generation requirements/owner approvals, and current final paired profile/association activation witness. The first paired witness was asserted by the real test before its approval; only final paired state persists after rollback. No private config/key/payload bytes are retained by this collector.

Fix1 committed `3820fe11572696f62d27291971f5c8d1dba0faec`, exact three authorized paths, FIX_BASE0295b820b5687c26020561a5a83fa4d743d877b8. Full post-fix card command `python -m pytest Tests/UI/test_workspace_project_index_settings.py -q --tb=short`: **23passed1warning576.91s**, `k5-fix1-ui-green.txt`;23original child XML/hashes in `k5-fix1-private-child-results.json`. Actual dark/light children114.74s/107.90s; pressureClear44.43s; invalid Clear19.81s/Disable16.74s and retained Enable11.69s/Apply12.92s validation. Scoped Ruff clean,3files already formatted,diffcheck clean (`k5-fix1-{ruff,format,diff-check}.txt`). `k5-fix1-source-equivalence.json` proves only3authorized files changed; other26files are byte-identical to source0295. Backend full-scale/current covering receipts remain applicable. Root metadata/QA remain unstaged.

Final visual rerender first attempt timed out in Edge screenshot after fonts loaded; that renderer failure is not a UI test failure or valid new pixel receipt. Original source0295 Settings pairs remain root-retained under QA/visuals/source-0295 with their historical hash manifest. A bounded viewport image-document render is being used to complete exact latest Console/post-fix Settings pairing before idle measurement.


## Final visual pairs and corrected idle qualification

All seven latest SVG/PNG pairs were rendered and inspected after the original covering and post-fix UI runs. `k5-fix1-mounted-render.json` records source qualifications, render UTC and exact per-image SVG/PNG SHA256; `k5-fix1-render.txt` reports rendered7 in3.27s. Console pairs are original covering/source0295 (unchanged Console bytes); Settings pairs are post-fix3820. Both dark80 and light170 palettes/blank=all copy/full action labels are visible. Clear-focus images prove focus/completion frames; the light image still shows previous action geometry before the next paint, so it is not proof of settled Off controls. Mounted state assertions separately establish Off. Requested-light/startup-overridden-dark remains explicitly historical adverse evidence. Root retains14 image files separately from generic text/binary receipts; old source0295 Settings images remain versioned in QA/visuals/source-0295. No product/CSS/layout edit accompanied rendering.

The first rare-only probe is preserved as `test_k5_idle_latency_borrowed.py`, `k5-idle-borrowed-fixture.txt`, scale `idle-borrowed-{fixture,source,retirement}.json`. It passed108.49s but a fixture binding lookup left a borrowed registry connection at public-call entry; it does not prove new outer-owner cost/retirement and is not final qualification. The corrected lookup uses the existing strict finite scope only around list_folder_bindings, ends before key/provider work, and never wraps public invoke in test ownership.

The next setup failed25.54s before candidate measurements because the diagnostic incorrectly required Capture exitcode0. Preserved bytes are `test_k5_idle_latency_returncode.py`, `k5-idle-returncode-setup.txt`, scale `idle-returncode-{setup,source}.json`. Scanner._worker finally unconditionally retires the process tree even after successful Capture/Close; nonzero exitcode1 can accompany valid protocol success and positive native retirement. The corrected proof retains the actual returncode and requires non-None plus child gone, retired (including watcher), pipes closed. This corrects a probe assumption, not product semantics or the latency gate.

Final query-only command, source3820, same retained physical completed caches, stable profile selected before imports:

```text
TLDW_TEST_CONFIG_ROOT=<worktree>/.superpowers/k5-scale/stable-profile
TEMP=TMP=TMPDIR=<taskdir>/native-temp
.venv/Scripts/python.exe -m pytest .superpowers/sdd/2026-10-20-workspace-project-keyword-index/test_k5_idle_latency.py -q --tb=short --timeout=900
```

`k5-final-idle-latency.txt`: **1failed1warning298.85s**, solely the final500ms p95 assertion. `final-idle-latency.json` retains all300 measured candidate samples (50 after5warmups per workload/shape;330total calls), exact100/1/0 counts, all40 public samples and42 native retirement proofs including authority warmups. `final-idle-source.json` records per-file source hashes/probe SHA/start/load. No concurrent K5 tests/builds/renderers; light read-only coordination and uncontrolled OS background may overlap.

| Cache/workload | Admitted p50 ms | Admitted p95 ms | Inner lexical p50 ms | Inner lexical p95 ms |
|---|---:|---:|---:|---:|
| Nested common |758.9345|1173.7593|370.1506|577.3054|
| Nested rare |528.7032|754.7411|188.1142|296.3043|
| Nested miss |479.4613|546.3212|168.2326|200.8069|
| Wide common |592.3079|697.7660|270.2827|302.6195|
| Wide rare |477.4943|1139.3945|171.8254|379.4548|
| Wide miss |443.9915|521.9079|157.7784|180.9103|

All six admitted p95 measurements FAIL the unchanged<=500ms target. Common is `generated file` over20,000 matching files/top100, rare `qualificationdeepmarker`/1, miss `absentqualificationzzzz`/0. Inner lexical remains diagnostic only; it is not substituted for the failed admitted boundary. No reconstruction, rescan, direct SQL, fabricated search authority or dropped admission/filters. Authority is captured from the unchanged normal provider warmup and retains the real code-issued guard/requesting floor; provider roots/guard fixture remains test-seeded as disclosed above.

The admitted clock includes fresh current registry fence acquisition, exact root/chain checks, guard, status/scope/state checks, current exclusions union, canonical lexical_candidates, and fence exit. Store open/read pin and final native/cache/registry close are outside this cached-candidate clock. Candidate scopes positively retire before the20 public calls per shape. Whole-tool clock includes all production guards, store/pin admission, lexical lookup, native Capture/live validation, final gates and outer registry close, without a test ownership wrapper.

| Cache/public20 | Lexical p50/p95 ms | Native Capture p50/p95 ms | Whole tool p50/p95 ms |
|---|---:|---:|---:|
| Nested |173.6766 /198.7187|245.7455 /280.0327|2178.1153 /2296.6473|
| Wide |162.9615 /257.0884|241.7983 /349.0586|2007.7522 /2469.3757|

All public calls return the fresh marker, enter with empty current-thread registry and return with registry retired, zero searches. All42 Capture PIDs gone/retired/pipes closed; service close leaves0searches. These correctness/native receipts pass despite latency failure.

## Failure-only phase diagnosis and clock alignment

`python -m pytest <taskdir>/test_k5_idle_diagnosis.py -q --tb=short --timeout=900` completes1passed1warning71.72s. That pass covers diagnostic count/authority/retirement assertions and is NOT a latency GREEN. Three warmups plus5 samples for each workload/cache use the same production boundary. Scale `idle-phase-diagnosis.json`, `idle-diagnosis-source.json`, and `idle-diagnosis-{nested,wide}.{txt,pstats}` preserve phases/profiler/source/load; original final failed samples stay unchanged.

Median wall phases (ms), measured on the worker performing the boundary:

| Cache/workload | Fence enter | Guard | Status | Merge | Lexical | Fence exit |
|---|---:|---:|---:|---:|---:|---:|
| Nested common |348.883|0.023|105.273|0.117|587.799|0.054|
| Nested rare |349.383|0.027|107.126|0.087|253.825|0.069|
| Nested miss |372.290|0.042|120.703|0.094|282.557|0.060|
| Wide common |422.105|0.035|114.580|0.110|699.880|0.138|
| Wide rare |467.676|0.049|150.222|0.187|381.153|0.247|
| Wide miss |353.532|0.035|107.092|0.149|249.450|0.062|

cProfile instrumentation adds overhead, so these are diagnostic phases, not replacement release measurements. The profile also includes asyncio entries; cumulative values overlap and must not be summed or advertised as perfect thread-isolated totals. Per shape24candidate calls show152 acquire_storage calls; sensitive_exclusions_under24 calls approximately6.239s nested/7.471s wide; load_status24 calls2.694s/3.278s. Current registry fence recomputes sensitive context, whose raw-input key calls admitted config.get_user_data_dir; per-row normalized exclusions contribute to common-match cost.

Exact source alignment: service._search_owned performs initial authority.guard BEFORE _search_store, then current fence/root/chain/load_status/scope+Ready-or-Partial/floor merge/lexical_candidates. The corrected probe moves that initial guard into the fence; it adds pure indexed==20000/Ready assertions, not extra status I/O. Guard cost is0.02-0.05ms, immaterial to observed misses; any future final measurement should align its placement exactly. lexical_candidates also performs its own production guard. The old K4 query probe timed Rig's in-memory RLock fence plus canonical lexical_candidates, with status setup outside the clock and a constructed Rig authority. Its308.5711ms common p95 therefore does not qualify this actual production registry/status boundary.

No optimization is implemented or claimed sufficient. A narrow index-only candidate/status operation could consolidate duplicated DB admission/transactions while preserving exact scope/state/exclusions/activation; per-query normalization could avoid repeated normalization inside the SQL filter before rank/LIMIT. However existing current sensitive-path resolution dominates the fence and cannot simply be cached. Measured costs do not establish that these small changes alone meet500ms. Any interface/authority change requires parent scope/ADR decision before code. Release gate remains failed and task is not Done.


## Fix round2 and current release boundary

Root explicitly amended the later expanded admitted500ms target as an accepted unmet deviation before code, retaining the original canonical decorated lexical500ms gate. See task-5-fix2-brief.md, ADR240 and the independent metric audit. This is a requirement deviation, not a claim the faithful earlier benchmark was wrong. No authority/admission or generic storage/config optimization was authorized.

Fix2 commit5d927a98ffc695c281465b09a8020d1db196490a changes only the DB lexical method and DB tests relative to3820. Per-query immutable exclusion normalization removes repeated per-row normalization; `_path_allowed`/live validation, current/requesting floors, policy, SQL rank/LIMIT, activation and native ownership remain unchanged. All other product AST is identical (`k5-fix2-source-scope.json`). Behavioral real-store baseline1passed7.70s, six scoped DB/provider nodes6passed75.58s, Ruff/format/diff clean. Exact commands/security equivalence/source/receipts are in task-5-fix2-report.md. Independent source rereview found0new findings. Literal scanner/build/memory/disk receipts remain valid for unchanged build paths; final query behavior is separately measured on the same retained caches.


## Final measured result

The same-cache final query command completed **1passed1warning256.56s**, call255.50s (`k5-fix2-idle-latency.txt`). Source remains `5d927a98ffc695c281465b09a8020d1db196490a`, with no changes after launch. `fix2-idle-source.json` records exact source/probe hashes and load. `fix2-idle-latency.json` retains300 measured samples plus explicit5warmups/workload (330candidate calls total), counts, authority/scope and all public/native proofs. No corpus/cache rebuild or copy.

| Cache/workload | Canonical p50 ms | Canonical p95 ms | Expanded admitted p50 ms | Expanded admitted p95 ms |
|---|---:|---:|---:|---:|
|nested/common|208.1393|222.3257|495.2744|534.4660|
|nested/rare|157.6595|172.1625|439.7766|469.0829|
|nested/miss|156.7933|181.9234|441.0551|493.7689|
|wide/common|212.3573|232.7950|502.7325|561.9087|
|wide/rare|157.2588|177.2331|444.5825|479.2464|
|wide/miss|157.7064|168.1995|443.0204|467.3531|

All six original canonical decorated lexical p95<=500ms gates PASS. Both expanded common cells still FAIL500ms (nested534.4660,wide561.9087); the other four expanded cells are below500ms in this final run. The accepted stronger-target deviation remains visible, with all earlier six-cell misses and nested/common canonical577.3054ms RED untouched. This is not a claim that expanded admission meets its stronger target, nor an exact causal speedup ratio across runs with different uncontrolled OS load.

Public20/cache instrumented end-to-end p50/p95: nested1994.4823/2703.4715ms; wide1909.2825/1957.3418ms. Native Capture p50/p95: nested240.5639/281.9572ms; wide229.7040/257.1561ms. Public lexical-stage p50/p95: nested162.5927/214.5572ms; wide156.0074/168.9987ms. Whole-tool values include probe psutil retirement proof/bookkeeping after Capture; Capture clock excludes that test proof but includes production worker/pipes retirement.

All40public calls returned the fresh marker, began with empty current-thread registry and retired it before return, without an outer test ownership wrapper. Fixture lookup/candidate scopes retired before public calls. All42native Capture children (including two authority warmups) are gone/retired/pipes closed; service closes with0searches. Exact expected100/1/0 candidate counts hold for all samples. `fix2-idle-process-retirement.json` separately records no remaining matching idle/scanner Python processes after completion. Correctness/scoped source-review results pass; root/final whole-branch review still owns acceptance of the expanded-target deviation and final task status.
