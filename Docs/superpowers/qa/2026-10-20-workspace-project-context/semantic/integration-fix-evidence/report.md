# Final scoped integration correction before the pending review verdict

Base:441bb9aa898e946d1d57d095ce2e64e3d95d53ab. This is the same sole implementer's narrow integration correction accepted by root under Task36423 plans17/18, recorded before changes. The existing scoped reviewer has not issued its final verdict. No new review seat or generic fix wave is introduced. ADR241 and the accepted plan were amended before the operation-ownership guard; ADR103 was clarified before the approved same-job target addition. ADR required: existing ADR241/103 amended; no new architecture, schema, backend, owner framework or dependency.

## Concrete gaps and resulting behavior

The historical hosted PR Fast Lane job120758047078 at d012 had5 failures,2959 passes and3 skips. The supplied original raw97443-byte log has SHA256 cb3933947d2c8575c5dfff760f9f0410b5ee5651463741c338997c7e7eb97221. Two exact workflow tests omitted the already-approved project-index-contract job/verdict; three ADR103 count statements failed, including an IndexError when the actual perf-inclusive runner count21 exceeded the word list. The contracts now name the existing job/verdict, the current statements say20 ordinary/21 including perf, and the exact count assertion accepts the actual numeric count or its known exact word through a safe slice. It still requires exactly one statement match and exact count equality. Hyphenated twenty-one is recognized. Historical dated counts keep their original meaning. No workflow job/dependency/cap/shard/aggregation change was made for accounting.

Nested issued carriers borrow the same actual registry operation, including when their folder binding keys differ. The earlier exact-carrier cleanup check could let outer carrierA close its parent while actual nested carrierB still rooted a failed factory close in that operation. The guard now checks every pending wrapper owned by the retiring carrier's actual parent operation, across binding keys. New-work admission retains the binding-key filter across operations; an independently admitted operation can still retire. The existing carrier's issuance/live-parent/physical-thread/native-child/own-failed-close guards remain in place. The production delta is confined to the existing service guard.

The new test obtains real code-issued outer/nested carriers from the actual service, _AdmittedStoreFactory and live native leases, and an actual EmbeddingFactory/cache record. It does not construct fake carriers/tokens/operations or spoof positive SQLite metadata. Both same-binding and different-binding nested cases must refuse outer close, retain the identical pending wrapper/factory and the actual live operation/native pins, permit a real keyword hit, then succeed on the same-owner factory close retry. It positively asserts both carrier pins closed, actual operation removed and done, and final service closure. Only the factory resource's close callback is deliberately failed once. Its RED-only cleanup closes actual owners after the defect has destroyed admission; no admission is reconstructed to make the assertion pass.

## Targeted RED and GREEN commands and literal outcomes

ROOT ALLFS WRITE HOLD covered fresh trust and each complete parent/private-child lifetime; no source, test, CI or document changed while any child was active. Both trust runs reported the exact private test paths trusted_directory/trusted_existing_directory. The unchanged existing bounded runner uses the original core venv, worktree cwd and established private-profile runner; no model is selected. Actual preflight is Python3.12.10 with SQLite3.49.1, sqlite_source_id873d4e274b4988d260ba8354a9718324a1c26187a4ab4c1cc0227c03d0f10e70 and required TEMP_STORE=1/MAX_PAGE_SIZE=65536/THREADSAFE=1 options. SentenceTransformers is absent. The preserved pre.json has exact argv/cwd/selectors/runtime hashes and the parent log has literal runtime output. This is structural current-source proof, not a historical per-import trace.

RED invocation:

```powershell
& ./.superpowers/sdd/2026-10-20-workspace-project-semantic-index/repair-sdd-test-trust.ps1 -RunName integration-fix-red-trust-01
C:/Python312/python.exe .superpowers/sdd/2026-10-20-workspace-project-semantic-index/snapshot-integration-fix.py red-pre
& ./.superpowers/sdd/2026-10-20-workspace-project-semantic-index/run-task4-hosted-fixture-targets.ps1 -RunName integration-fix-red-01 -Selectors @(
    'Tests/CI/test_derived_artifacts_workflow.py::test_workflow_has_one_fast_prerequisite_and_one_required_aggregator',
    'Tests/CI/test_derived_artifacts_workflow.py::test_required_aggregator_fails_when_either_lane_fails',
    'Tests/CI/test_ui_lane_shard_count_statements.py',
    'Tests/RAG/test_project_projection_scope.py::test_nested_pending_close_keeps_actual_shared_operation_and_keyword'
)
# After natural parent/private-child exit:
C:/Python312/python.exe .superpowers/sdd/2026-10-20-workspace-project-semantic-index/snapshot-integration-fix.py red-post
```

Natural parent session25043 exited1, literal `6 failed, 8 passed, 1 warning in 66.94s (0:01:06)`; parent XML14 tests/6 failures/0 errors/0 skips/time66.926. All five prior hosted CI failures reproduced. The new real nested regression failed for the intended reason: actual observations were [(True,False,False),(False,False,False)] instead of refused-and-retained for both binding variants. There were no setup errors. Original child XML1/1/0/0/time58.948; child log `1 failed, 1 warning in 58.95s`, call57.81s; outer test call64.27s. Actual RED resource cleanup completed and no matching Python child remained before edits. RED failures are retained as failures, never relabeled green.

GREEN invocation:

```powershell
& ./.superpowers/sdd/2026-10-20-workspace-project-semantic-index/repair-sdd-test-trust.ps1 -RunName integration-fix-green-trust-01
C:/Python312/python.exe .superpowers/sdd/2026-10-20-workspace-project-semantic-index/snapshot-integration-fix.py green-pre
& ./.superpowers/sdd/2026-10-20-workspace-project-semantic-index/run-task4-hosted-fixture-targets.ps1 -RunName integration-fix-green-01 -Selectors @(
    'Tests/CI/test_derived_artifacts_workflow.py::test_workflow_has_one_fast_prerequisite_and_one_required_aggregator',
    'Tests/CI/test_derived_artifacts_workflow.py::test_required_aggregator_fails_when_either_lane_fails',
    'Tests/CI/test_ui_lane_shard_count_statements.py',
    'Tests/CI/test_ci_queue_pressure_contract.py::test_project_index_required_lane_has_real_pinned_sqlite_and_bounded_targets',
    'Tests/RAG/test_project_projection_scope.py::test_nested_pending_close_keeps_actual_shared_operation_and_keyword',
    'Tests/RAG/test_project_projection_scope.py::test_peer_pending_close_does_not_poison_admitted_carrier_retirement',
    'Tests/RAG/test_project_projection_scope.py::test_failed_project_factory_close_retains_real_parent_until_retry',
    'Tests/RAG/test_project_projection_scope.py::test_issued_scope_uses_live_parent_and_rejects_copy_after_close'
)
# After natural parent/private-child exit:
C:/Python312/python.exe .superpowers/sdd/2026-10-20-workspace-project-semantic-index/snapshot-integration-fix.py green-post
```

Natural parent session71998 exited0, literal `19 passed, 1 warning in 185.96s (0:03:05)`; parent XML19/0/0/0/time185.962. This is2 workflow contracts,11 statement/accounting parameters,1 exact project-lane contract and5 private ownership cases. All selected cases executed with zero skips. Five copied original private child XMLs each report1 test/0 failures/0 errors/0 skips: nested55.718s; independent peers17.555s and17.653s; own failed-close retry31.083s; native-owner child35.690s. Outer calls are60.83s nested,22.85/22.95s peers,36.21s own retry and40.74s native owner.

Physical retirement is distinct from passing pytest: the exact passed tests assert actual factory close retries, native-child cancellation/join and finish, physical pin closure and operation retirement. The new test joins both keyword supervisors and asserts native_retired before semantic setup; its two nested cases retain actual operation/pins on refusal and retire them positively after retry. All five private child processes and the parent exited naturally. The post-run process observation contains remaining=[], including previously observed parent68308/77728 and child24428/46256; no integration-fix-green-01 Python process remained. No timeout, forced kill, blind repeat or widened bound occurred. ROOT FS HOLD was released only after those checks and source34 equality. Known pytest-asyncio setup deprecation and Pydantic Field deprecation are preserved. The six GREEN parent/child logs contain no unhandled-thread, executor failure or unretrieved-exception warning. No HTTP-provider/model effect is claimed by these model-free controls.

## Exact source, CI and static qualification

integration-fix-snapshots/{red-pre,red-post,green-pre,green-post} each preserve34 raw files and a SHA256/byte manifest: all30 prior final-core paths plus the two newly modified CI test files, Tests/private_profile.py and the unchanged targeted runner. RED pre/post match exactly; GREEN pre/post and current source match exactly. The same snapshot set includes the actual test bodies, DB/runtime/allocation/provider/carrier/service code, conftest and workflow. Equality establishes the stated source boundaries; it does not convert old441bb9 app/model or20k proofs into guard-delta proofs.

The approved Windows project-index job now adds exactly Tests/RAG/test_project_projection_scope.py::test_nested_pending_close_keeps_actual_shared_operation_and_keyword, changing16 selectors/44 cases to17 selectors/45 cases. Its exact selector tuple contract passes. The new RED outer call64.27s plus the prior314.19s lane supports a prospective roughly379s total, within the unchanged1200s job/180s per-test limits. The actual new GREEN nested outer call was60.83s. The new45-case lane was not replayed merely for hashes; final hosted new-head CI remains pending. Existing deps, Python/runtime assertions,13 UI shards and fail-closed required aggregation remain unchanged.

Static command: C:/Python312/python.exe .superpowers/sdd/2026-10-20-workspace-project-semantic-index/verify-integration-fix-static.py. Literal output: `SCOPED FORMAT/LINT/AST/DIFFCHECK PASS: 5 Python; 10 explicit changed paths; 12 literal command records` and `SOURCE34 RED/GREEN/current exact equality; green thread/executor/unretrieved warning scan clear`. The receipt retains every actual argv/stdout/stderr:10 modified-region read-only Ruff format checks, Ruff check5 changed Python with `All checks passed!`, AST5, and explicit10-path git diff --check exit0 with empty stdout/stderr. Earlier actual range formatting is retained separately in integration-fix-static.json. Self-review followed actual nested registry admission, issuer parent construction, cleanup caller and guard, and exact workflow accounting; production changes remain at the existing seam.

Current qualified changed source hashes:

- `.github/workflows/derived-artifacts.yml`: `6b70ba4136b8590c79ce961bd648102cc2ae161f4cd5d6beb95c8de88d07c05d` (76284 bytes).
- `Docs/Architecture/project-indexing.md`: `6c1ad4ddad3524ca66c83a7ab225c5e365363fa8a179f78ee2a797b5bca728a2` (24430 bytes).
- `Docs/superpowers/plans/2026-10-20-workspace-project-semantic-index.md`: `cc3b90355b1807189ddf27ef929e58c14e4f75725c4f306a2ea24ae08ed9009a` (49598 bytes).
- `Tests/CI/test_ci_queue_pressure_contract.py`: `8ce28b06683a742b9194e5a880a566d84a5f26d61fafe8cf0f28ae7820287ece` (77715 bytes).
- `Tests/CI/test_derived_artifacts_workflow.py`: `75d495c2a30814dbd9dd01983d504cda057d3c0f0dbea490029d817f75c2d3f0` (25850 bytes).
- `Tests/CI/test_ui_lane_shard_count_statements.py`: `7e0c9d6eb380f1a9480d0d824fde91fd15bb2336983cd7752c8e5e2bc7fb77ba` (3761 bytes).
- `Tests/RAG/test_project_projection_scope.py`: `94faf85bd07b14eb666b287efc9ccb23c40460a6d7f349510c5d1d1a31911b5d` (43865 bytes).
- `backlog/decisions/103-fast-pr-lane-and-required-gate-aggregation.md`: `5dabc9646552585a6dc59ee6cdcf1ea8e10741defc7003924ba9ea9df4ffbbc4` (24405 bytes).
- `backlog/decisions/241-workspace-project-context-and-folder-bound-index.md`: `545c0bea2d2fceb0e6e6151a590c5f73600bd15b133201280ad8fe92a7fc2f10` (74922 bytes).
- `tldw_chatbook/Workspaces/project_index/service.py`: `1ca01992481ee9ad063cfea09ae863b6a1e33ed97ba5ce8ab330d243e9fa2935` (101358 bytes).

## Evidence and limits

Separate semantic/integration-fix-evidence retains only the two new RED/GREEN run families, all four exact source34 snapshot sets, fresh trust, current static/retirement records, unchanged runner and snapshot/static/retention helpers, narrow delta diff, this immutable precommit report and the one supplied historical hosted failure log. Its manifest enumerates original bytes/SHA256 and the lossless ZIP members; scoped attributes preserve raw line endings. The live final-fix-wave-report.md receives this same new section and later commit verification; later live text is not represented as in the precommit archive. Existing committed final-fix-evidence and historical manifests/raw QA remain unchanged. Root Task36423, root semantic README and untracked final-reviews are excluded. No model/archive evidence is copied from root-owned review files.

The prior actual cached MiniLM app lifecycle pass is historical at441bb9 for this subsequent guard delta. No real-model,20k,Console or mounted Settings replay occurred. Older scale results retain their historical source and62-74-second query limitation. This correction qualifies the actual targeted ownership and CI paths on the stated Windows runtime; it makes no final hosted-head CI, general relevance/performance, cross-platform SQLite or new app-model claim. No install/download/shared-environment mutation/original-checkout edit/full suite/push/PR/rebase/merge/automation/task closure occurred. Commit only the explicit10 changed source/test/CI/docs plus this new evidence group, verify actual HEAD bytes, provide the delta to the same scoped reviewer, then HOLD.
