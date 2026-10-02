# Task 2 — stronger live Hold evidence and short coordinator sections

Status: safety candidate implemented; performance acceptance FAILED. Independent specification/concurrency/security and quality reviews remain pending. TASK-33560 remains In Progress. Task 3 is not implemented. No numerical-goal or Windows/Linux qualification is claimed.

Dispatch BASE: `b4980bc58782ec1f5ba8b142c504773fad00d0e1`; branch `codex/backup-admission-perf-20261001`; worktree `/private/tmp/chatbook-native-credentials-20260929`. Implementation/code commit: `65b8f7e4f90fa1e13704fc77e7ce6666f9c35796`, tree `a5ae009e3a9025222635b78c8c30e5dc55a594f2`. The final evidence-only follow-up is identified by this report’s Git history and the controller handoff; it does not change measured application code. The accepted reconciliation reports remain `/private/tmp/task33560-reconciliation-report.md` and `/private/tmp/task33560-expanded-dev-reconciliation.md`.

## Implementation and boundaries

The existing `_Hold`, `_Evidence`, pending/live/retiring sets and public APIs remain the ownership model. No new dependency, writer protocol, persistent cache, monitor, MCP behavior or materializer/capture cache was added.

- `_Evidence` stores bounded current file bytes and directory entry names alongside posture/change observations. Candidate metadata only rejects reuse; identical metadata cannot grant it. A warm borrower rereads the complete bytes under the native registry read barrier before comparing them with its accepted derivation. Evidence covers selector, profiles, registry, marker, qualification file, pending-directory names and actual native gate/lease files. The existing settlement window remains an additional eligibility condition, not authority.
- `_Hold.predecessors` owns bounded link-free `_DirectoryChain` resources, each retaining every root-to-leaf predecessor. Every use reads current native `fstat` posture and checks each current parent/name edge against its held child. Stale chains cannot grant admission; a complete current cold route decides legitimate data-path replacement outcomes. Trusted-link paths retain their existing full route. Windows evidence reuse remains disabled; native Windows validation remains in the existing platform layer. No POSIX substitute or native qualification claim was introduced.
- `Admission._current_hold` validates the current authority identity, independently opened registry lock, publication intent, current registry, all foreign/alias/absence group derivation, original native group, gate and retained lease identities and current security posture. New ordinary borrowers refuse a closed gate; an already installed descendant can finish its exact retained scope. Native registry waiting uses the existing cancellation poll interval and cancellation event. Concurrent borrowers never share a lock file description.
- The native registry parsed model is reused only after a current bounded read compares every byte. The stored model is private to the Hold; consumers receive deep copies. Full derivation evidence is also reused only after current byte/edge/native checks. Before/after native publication barriers remain separate.
- Warm tokens are counted before validation. Native and filesystem work happens outside the coordinator mutex, then actual Hold identity, cancellation, selection, scope and evidence epoch are rechecked. Cold initializer single flight remains. `_scope` records the actual incumbent whose continuing mapping it used; retirement/replacement of that incumbent cannot silently create a new owner from the old permission.
- `_Operation.check` now performs memory/provenance checks. The shared `_check_operation` reserves/checks state, observes path resolution and parent identity outside `_lock`, then rechecks state. Only actual locked callers in `participants.py` and storage were adapted. Startup retirement likewise snapshots its owner, observes records outside the lock and checks the same owner before retirement.
- Last-token close still fences the Hold and joins outside `_lock`. The Hold closes retained predecessor resources before exiting its original native context. An ambiguous close is never retried against a possibly reused descriptor. Its resource and native context stay in `_retiring_holds`, with an error, and drain remains false. The real native-exclusion test proves maintenance cannot acquire while this uncertainty remains.

## Actual caller map

`storage_admission._check_operation` is used by storage operation entry/restoration/acquisition/final validation, `participants._core_access` and `_core_getter`, and the existing raw/config/chat/dictionary/visual participant bridges. Source census of direct bridge calls found no enclosing coordinator-lock context outside the adapted storage/participants calls: raw lines 535/825; config 332/373; dictionary 174/285; chat 274/353/408 (353 is only inside an ExitStack); visual 236. No sibling per-owner validation copy was added.

`Admission._directory`, `_tokens`, `_groups` keep their cold default. Only `_current_hold` supplies the Hold-owned predecessor lookup. Existing admission/recovery/publication callers receive no pins or parsed generation. `_read` keeps its uncached default; only `_current_hold` supplies an equal-byte previous native registry. `native_files.pinned_directory` and `private_paths._open_verified_parent` are unchanged, including trusted-link limits and capture/publication routes.

## Control-authority removal ruling

At exact BASE, `test_admission_evidence_reuse._bootstrap_root_removed` removes the bootstrap tree while a startup lease remains live. `control_records.admission_authority:114` recreates the marker and a new Admission authority. BASE `_acquire_storage:1237–1257` then finds the old Hold by `(PID, lexical root)`, compares only namespace names and discards the newly constructed authority. For UNBOUND, this returned allowed while retaining locks in the detached original authority.

Controller confirmed this is the stale native authority forbidden by the approved strict contract. Both current oracle arms now refuse this lost/recreated control authority. The oracle test was not weakened or rewritten. Legitimate data-directory rename/recreate cases still use cold derivation and retain their actual allowed results. Reenrollment can occur only after positive old-Hold retirement; no simultaneous substitute authority or blanket data-path refusal was added. Controller ruling and diagnosis are also recorded through official TASK-33560 notes.

## Red and green receipts

All commands used the existing venv and `/private/tmp/backup-followup-check.py`, which establishes private profile/HOME/XDG selections, Null keyring and network guard before app imports. Outputs remain private. No full suite, dependency install, real profile, network request or deadline change was used. The joined wrapper `/private/tmp/task33560-joined-run.py` executes that same runner with `runpy` and records actual imported source files/hashes; its final version also records source hashes before/after the run.

| Evidence directory under `/private/tmp` | Scope and result |
| --- | --- |
| `backup-followup-check-dtrej5yv` | Initial red: pause/selector changes during warm observation + blocked scope vs unrelated real transaction: 3 collected, 3 expected failures. |
| `backup-followup-check-7ckxrvo0` | First green: same 3 tests, all passed. |
| `backup-followup-check-gucj9p74` | Initial red: no current complete control byte reads and replaced gate permits write: 2 collected, 2 expected failures. |
| `backup-followup-check-2n3ewydg` | Interim 43-case run: 7 failures, 1 skip. Preserved: refusal-order/native-control and fallback differences; not a green receipt. |
| `backup-followup-check-awf_cw4n` | Interim 82-case run: 11 failures, 1 skip. Three actual newly read gate/lease paths were absent from evidence coverage; fixed by adding those mandatory files, without removing trace assertions. Eight participant failures were independently reproduced on BASE. |
| `backup-followup-check-17__b_c3` | Exact-BASE source with added resource tests: 2 expected red failures (no retained predecessor resources). |
| `backup-followup-check-secxbtmf` | Same retained-borrower and uncertain-close/native-exclusion tests: 2 green. |
| `backup-followup-check-hlskgzh4` | 110 cases: 1 CharactersRAGDB cursor failure, 1 skip. Exact BASE reproduced this failure. |
| `backup-followup-check-hup3wom9` | 167 collected, 166 passed, 1 intentional skip, zero failures/errors. Evidence/related/bootstrap/admission/native-files/startup-readmission modules. |
| `backup-followup-check-522tcehm` | Exact BASE red: same-inode corrupt control bytes with change stamps held equal still permits dependent I/O. 1 expected failure. |
| `backup-followup-check-r2f5ef7v` | Exact BASE red: incumbent retires during cold scope derivation, allowing a new owner to inherit its continuing mapping. 1 expected failure. |
| `backup-followup-check-ooen53m5` | 165 collected, 155 passed, 9 BASE-reproduced failures, 1 skip. Includes equal-stamp byte and retained native lock tests. Actual import joins: `/private/tmp/task33560-current-imports.json`. |

Every listed runner summary records zero undrained network attempts. A mistyped related-path filename produced a zero-test collection command at `/private/tmp/backup-followup-check-kwgo2z3z` (exit 4); it was corrected to the real `test_related_path_admission.py`, not counted as coverage. All new reader output is redirected privately before exposing the bounded summary.

### Exact-BASE controls and existing failures

`/private/tmp/task33560-base-tests` is a git-archive extraction of BASE, with the existing venv symlink. Only the evidence test file was subsequently extended to run new red cases; BASE production modules were not edited. This is a focused test control, not a rerun or relabeling of the accepted performance baseline.

- `/private/tmp/backup-followup-check-r_ic77_7`: 32 participant cases, 8 failures matching the draft by exact test name.
- `/private/tmp/backup-followup-check-4d_q_4rn`: 5 core escaped-cursor cases, 1 matching CharactersRAGDB failure.
- `/private/tmp/backup-followup-check-76cefhuq`: 7 source-joined control cases, the same 3 selected failures; import/source hashes in `/private/tmp/task33560-base-imports.json`.

Exact repeated failures (no unrelated pooling test or implementation changes):

1. `test_repository_schema_connection_is_closed_on_return[EventStateRepository]` — empty observed `_get_connection` list.
2. Same case `[SyncStateRepository]` — same empty list.
3. `test_actual_worker_transaction_drains_after_commit_and_native_close[EventStateRepository]` — expected ProgrammingError after close not raised.
4. Same case `[SyncStateRepository]` — drain remains false.
5. `test_pre_authority_attempt_stays_counted_until_return_and_cannot_allocate_late[authority]` — drain remains false.
6. Same case `[startup]` — drain remains false.
7. `test_pause_cancels_actual_pending_native_acquisition` — drain remains false.
8. `test_repository_transaction_rolls_back_and_closes_native_file[EventStateRepository]` — expected ProgrammingError not raised.
9. `test_admitted_transaction_finishes_but_escaped_cursor_keeps_native_hold[CharactersRAGDB]` — `base_db._QuiescentSQLiteCursor.fetchone:626` raises closed-cursor ProgrammingError.

The current Event/Sync implementations pool transaction connections and have distinct schema-connection lifetimes. These controls establish matching failures on the exact imported BASE production source, not a broad assertion that the entire baseline is healthy.

## Scoped static validation

Paired Bandit on four touched production files: zero findings at BASE and draft (`/private/tmp/task33560-bandit-base.json`, `task33560-bandit-current.json`). Paired Ruff: 24 existing production findings at both sides, plus the same 2 existing test-file findings; no new finding (`task33560-ruff-base.json`, `task33560-ruff-base-test.json`, `task33560-ruff-current.json`). Existing findings include deferred annotation/import/style items; no new suppression was introduced beyond the fixture-import pattern used by existing tests. Changed function ranges were formatted with Ruff, preserving unrelated baseline formatting. Compilation and `git diff --check` passed. Final scoped runs and source stability are recorded below when complete.

## Hold interfaces for Task 3

These are internal custody/evidence fields, not transferable admission authority:

- Existing `key`, `names`, `count`, `ready`, `stop`, `error`, `thread`, `evidence`, `path_evidence` retain their roles.
- `predecessors`: bounded directory-path to `_DirectoryChain`; links are ineligible. Each chain retains ordered predecessor descriptors and `close_error`.
- `resources`: every created chain is registered here before opening, including candidates whose opening/retirement is uncertain. It is changed under `_lock` while borrowers are counted; native opening/checking/closing is outside that lock.
- `registry_evidence`: one private `(full current bytes, detached parsed native registry)` generation, published under `_lock` only after final warm validation. It cannot replace the next native lock, read, intent/group/security or byte-equality check.
- `native_context`: original native admission context. It is exited only after last-borrower resource retirement succeeds; uncertainty retains it and the Hold in `_retiring_holds`.
- `_Evidence.pins` references resources owned by the same live Hold. `_Evidence.candidate()` is rejection-only; `observe()` performs fresh edge/security and complete byte observations.

Task 3 must reuse the upstream 1 Hz cadence and these existing retirement/counting sets. This implementation adds no monitor wake, cancellation/join scheduling or MCP store cache. Task 3 still needs its own reviewed current-consumer applicability, initial/immediate relevant wake, and positive settlement tests.

## Limits and remaining acceptance

Native execution here is macOS only. Windows full derivation is preserved, but no Windows ACL/reparse/native qualification was executed. Linux is unmeasured. No idle reduction is measured (Task 3). The unchanged complete-boundary <0.5 ms, boot-open reduction >=80%, and idle reduction >=50% remain binding. Stronger current full-byte/native validation is mandatory even if the candidate misses performance goals.

The accepted prior baseline remains the independently checked historical840/currentf337 pair with probe SHA `34278facac896ecc0e4ed8a3319243d3501272e87692a858779c6b449a475428`; currentf337 complete boundary 1.159021 ms and boot reduction 84.44381079%. It is not relabeled as BASE b498, and later upstream boot/source changes are not inferred from it. The retained probe has no raw per-iteration samples; its seed/warm boot child overwrite and bounded source-stat/native-retirement receipt limitations remain. The coordinated fixed-probe results below measure this implementation and preserve those prior measurement identities.

### Final source-stable batch and process-order limitation

`/private/tmp/backup-followup-check-8mk9ut1_` collected 281: 261 passed, 19 failed, 1 intentional skip, no errors/network attempts. `/private/tmp/task33560-final-imports.json` proves the four touched production source hashes were identical before and after, and records actual imported source paths/hashes. The eight BASE-imported module hashes all join their exact BASE Git blobs in `/private/tmp/task33560-source-control-join.json`.

Per-module final batch: evidence 48 pass/1 skip (all 11 added cases pass); related paths 7 pass; bootstrap 31 pass; admission 45 pass; native files 35 pass; startup readmission continuity 4 pass; core participants 49 pass/11 fail; participant lifetimes 24 pass/8 fail; runtime startup handoff 7 pass; startup owner retirement 7 pass; pending readmission 4 pass.

Ten additional core drain failures in this ordering follow `test_bootstrap.test_failed_explicit_factory_close_retains_admission_without_gc_retry`. That existing test deliberately leaves its failed-close lease alive and proves native exclusion still exists after GC. Its process-owned state must remain visible to subsequent global drain. Exact-BASE minimal order control `/private/tmp/backup-followup-check-291ole3d` runs that test followed by the MediaDatabase escaped-cursor drain and independent nested-exception drain: 3 collected, bootstrap passes and both later drains fail identically. Source/import evidence is `/private/tmp/task33560-base-order-imports.json`. This is not fixed with a global reset or inferred positive retirement. Controller requested separate private process module verification, recorded below. The combined receipt remains a failure, not an all-green claim.

Separate-process core verification: `/private/tmp/backup-followup-check-__bguwof` collected 60, 59 passed, only the exact-BASE CharactersRAGDB escaped-cursor failure remained. The ten combined-order drain failures disappeared without changing tests or cleaning unknown native ownership in-process. Production before/after hashes and imports are in `/private/tmp/task33560-core-isolated-imports.json`.

Separate-process participant verification: `/private/tmp/backup-followup-check-zkndcltt` collected 32, 24 passed and the same 8 exact-BASE failures remained, with zero errors/skips/network attempts. `/private/tmp/task33560-participant-isolated-imports.json` records unchanged before/after production hashes and actual imports. The final evidence module contributed 48 passes/1 intentional skip, including all 11 added cases, and the other eight affected safety/startup modules all passed as enumerated above. The remaining 9 isolated failures are explicit acceptance limitations, not suppressed tests.

Final scoped Bandit scanned 3,884 BASE and 4,280 current production lines, with no scan errors and no findings on either side. Final Ruff has the same 26 combined existing production/test findings as BASE. Source compilation and whitespace validation passed; whole-file format checks retain pre-existing formatting differences, while changed function ranges were formatted.


## Fixed-probe result: numerical acceptance failed

The controller explicitly reserved a quiet window. Four controls ran sequentially using the unchanged retained probe SHA `34278facac896ecc0e4ed8a3319243d3501272e87692a858779c6b449a475428`: BASE and candidate, each with 100 transaction iterations and 3 measured warm boots, plus the unchanged seed/warm-up children. No concurrent apps, tests, source edits or unrelated tools ran during timing. All original child/native limits remained unchanged. No control was retried.

Immutable package: `/private/tmp/task33560-task2-paired-t7iddr01`.

| Source | Commit | Tree | Full source content SHA256 |
| --- | --- | --- | --- |
| BASE | `b4980bc58782ec1f5ba8b142c504773fad00d0e1` | `047781af74cb23f061f5c8d8aa36564c829215a6` | `f82a131876cdb809e6ce32d3c1ff8415acb294c43119f83dddf2e2473bd50d1a` |
| Candidate | `65b8f7e4f90fa1e13704fc77e7ce6666f9c35796` | `a5ae009e3a9025222635b78c8c30e5dc55a594f2` | `cc3163a22bfe95f1bbf6d2a4b61ccdc49418756bebdd73b535fbb19d4ccf3506` |

Sources are separate git-archive extractions frozen read-only. `sources.json` records their identities; `source-stat-before.json` and `source-stat-after.json` retain full file byte/stat ledgers. `command-summary.json` confirms all four controls completed, both source byte/stat ledgers stayed equal, executable/containment byte/stat identities stayed equal, and source digests still match their manifests. Frozen permission modes are distinct from writable checkout modes and are applied equally to the pair.

| Metric | BASE b498 | Candidate 65b8 |
| --- | ---: | ---: |
| Complete transaction boundary median, including entry and retirement, excluding measured SQL body | 1.6881875 ms | **3.7872705 ms** |
| Narrow admission median (not the acceptance metric) | 0.3499165 ms | 2.072104 ms |
| Transaction opens / 100 iterations | 0 | 1,400 |
| Three boot open counts | 19,934 / 19,730 / 19,702 | 25,336 / 25,375 / 25,401 |
| Boot-open median | 19,730 | **25,375** |
| Three boot seconds | 8.241719375 / 8.843579583 / 8.418132208 | 9.261091 / 8.988196209 / 8.976043041 |

The complete boundary misses the unchanged <0.5 ms target and regresses against this BASE. Boot opens increase **28.6112519007%** against this BASE. The Stage 4 >=80% boot-open reduction target uses retained historical840’s **126,252** opens, not BASE b498. Candidate 25,375 is an arithmetic **79.9013084941%** reduction against that retained historical result, also short of 80%. This is not a new matched historical840 run, does not relabel the accepted historical840/currentf337 controls, and does not confer cross-source or native-platform qualification.

Raw control receipts are `receipts/base-transaction.json`, `draft-transaction.json`, `base-boot.json`, `draft-boot.json`; bounded analysis is `metrics-summary.json`. All **14 child/supervisor phases** report exit 0, retired true, supervisor retired true, no outstanding Hold/lease/operation/acquisition/raw/retiring ownership, zero network attempts and zero foreign-source modules, with the exact expected source digest. Quiet-window completion was reported only after these checks. All app and supervisor work had settled before the controller resumed other work.

The stronger current barrier accounts for material observed work: the candidate opens 14 objects per measured warm transaction (current independently locked native authority/gate/lease observations plus complete control-file/directory observations), whereas BASE’s metadata-only warm result opens none. Pin retention removes repeated directory opens in the eligible native walks but does not yet amortize the complete safe boundary enough. The retained probe does not provide a finer per-check breakdown, so no unsupported attribution or speculative timing improvement is claimed. No mandatory current-byte/native/foreign/alias/security validation was removed to improve these results.

Further limits: boot entries are 47/45/45 for BASE and 45/45/46 for the candidate; retirements are 47/45/45 and 45/45/45 respectively, despite final zero ownership in every child. Preserve these observable scheduling/counter differences. There are no raw per-iteration timing samples. Earlier boot child-path overwriting in the fixed probe remains unchanged. Execution is macOS only; native ACL/Windows handle counters do not establish Windows coverage. Idle reduction was not measured. No repeat, deadline change, goal waiver or speculative performance patch followed the failed measurements.

Controller will review this exact safety candidate through independent SPEC/concurrency/security and QUALITY gates. The requested performance outcome is unfinished; Task 3 and overall TASK-33560 completion are not authorized by this report.

## Exact implementation scope

Application/test files in the measured code commit:

- `tldw_chatbook/Backup_Recovery/storage_admission.py`
- `tldw_chatbook/Backup_Recovery/admission.py`
- `tldw_chatbook/Backup_Recovery/native_files.py`
- `tldw_chatbook/Backup_Recovery/participants.py`
- `Tests/Backup_Recovery/test_admission_evidence_reuse.py`

Associated metadata: this report and `backlog/tasks/task-33560 - PERF-08-part-2-event-driven-maintenance-probe-warm-MCP-store-reads-raw-participant-stamp-caches.md`. No Task 3 source, dependency, deadline, old probe, TASK-33267 status or unrelated pooling assertion was changed.
