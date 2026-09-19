# Exact f10a upstream baseline comparison

The exact incoming-dev baseline PASSES both disputed tests: **2 passed in 2.06s**, exit 0. They must not be classified as upstream baseline failures. The merged run remains two genuine integration-test failures before the tested watcher/receipt actions.

Revision: `f10a3fa36471ade188a002dc119bf19db54c4d83`.
Read-only Git archive extracted to `/private/tmp/uat-dev-f10a-baseline-source-i0a0uk3o`; no repository files changed. The archive contains original incoming tests and fixtures, without authority retargeting overrides or guard changes. Same Python 3.12 virtual environment and normal per-test fixture behavior as the merged run. Only these nodes ran:

- `Tests/Notes/test_notes_sync_runtime.py::test_a_disk_edit_after_activation_surfaces_on_the_next_check`
- `Tests/Notes/test_notes_sync_runtime.py::test_write_receipts_name_path_title_effect_for_completed_operations`

The test module is byte-identical between incoming and merged (SHA-256 `7d4eccd09e2837ece1e87e4cf5848a9d686510f677ef02d25e26d3c05796adba`). Pytest ran via `pytest.main` from the archive with explicit archive PYTHONPATH, safe-path Python, `-q`, and a separate basetemp. A collection observer only asserted the origins of already-loaded modules; it did not import product modules early, modify fixtures, or change outcomes. All 95 loaded product modules resolve inside the archive. The existing Requests dependency warning remains.

Evidence:

- `/private/tmp/uat-dev-f10a-upstream-baseline.log`
- `/private/tmp/uat-dev-f10a-upstream-baseline-origins.json`
- `/private/tmp/uat-dev-f10a-upstream-baseline-receipt.json`
- Baseline fixtures: `/private/tmp/uat-dev-f10a-baseline-fixtures`
- Prior merged failures: `/private/tmp/uat-dev-f10a-upstream-independent.log`

## Semantic difference and limits

Both fixture versions select a bootstrap profile before collection, then change HOME, XDG directories and TLDW_CONFIG_PATH per case. The merged conftest's private-profile exceptions do not match either of these nodes, so their ordinary retargeting remains effective. Unlike incoming dev, the merged product retains source-bound configuration admission. `_participant_state` rejects a config source when its freshly selected binding differs from the registered selection (`raw_participants.py`, config branch, around line 127). The merged failure logs repeatedly show `raw_source_selection_changed` while sensitive-path qualification calls user-data/database configuration getters; the exact upstream run has no such diagnostics and completes startup sync.

This establishes a real compatibility difference between backup source-bound admission and these inherited per-test retargeting fixtures. It does NOT, by itself, prove every startup failure was caused by those caught diagnostic exceptions: the tests assert only that initial synchronization changed the note, and the exact declined root status is not recorded in that log. Do not relax admission, claim a malformed database, or attribute the failure to the new receipt method from this evidence.

The receipt test fails at its first post-start content assertion, before calling `write_receipts`. The watcher test also fails before its disk-edit/manual-check assertions. The new watcher setdefault behavior is identical to incoming dev, and the tested receipt-worker resolution is not entered at those failure points. Thus the final 36-case native maintenance approval stands for its measured scope, while these two broader functional checks remain unresolved integration evidence. Use qualified fresh-profile/installed verification to establish actual startup and receipt behavior; no blanket upstream-pass or full merge acceptance follows from the focused lifetime suite.

No app-booting suite, remote operation, cleanup of unrelated fixtures, or product/test modification was performed.


## Qualified merged verification (same unchanged test bodies)

Both exact upstream async test functions now PASS in separate fresh merged-product processes, with private HOME/XDG/TLDW_CONFIG_PATH selected before imports and held unchanged. Driver: `/private/tmp/uat-dev-f10a-qualified-upstream.py`; machine results: `/private/tmp/uat-dev-f10a-qualified-results.json` (per-child logs and complete source-origin receipts).

Each child installs the existing `Tests.network_guard` before importing the test module, calls the original function via `asyncio.run(function(tmp_path))`, and verifies zero blocked network attempts, no `Tests.conftest` import, no app import, and unchanged selected environment. No product patch, authority reset, cache clearing, guarded-getter override, or TLDW_TEST_MODE override is used. The existing synthetic Notes/folder doubles inside the original upstream test are retained unchanged; the filesystem, sync store, coordinator, startup path, production runtime adapter, and receipt code are real. Every loaded product module resolves under the current worktree; the unchanged test SHA is verified in each receipt. Each child retains a 30-second diagnostic ceiling and exits 0.

The watcher function proves that a real disk edit followed by manual Check still leaves the change available to the watcher's next comparison. The receipt function proves automatic startup synchronization, real completed-store receipt path/title/effect, invalid-limit refusal, and receipt access after normal root pause. Their original assertions and cleanup run without modification.

The exact setup difference is selection lifetime: ordinary pytest imports source-bound configuration under its bootstrap profile, then the shared autouse fixture selects another profile before these unqualified cases. The successful harness selects its disposable profile once before imports and does not apply the later retargeting fixture. No runtime authority was manually re-associated. Thus incoming dev passes under its original fixture, merged fails under the retargeted fixture, and merged passes the same bodies under a correctly sustained profile lifetime. This establishes meaningful functional acceptance of these two Notes routes without treating their earlier failures as upstream baseline-equivalent or weakening native guards.
