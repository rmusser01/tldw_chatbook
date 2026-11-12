# Final combined prompt mix-in fix report

Implementation commit: `d7744a78e9094ad4df08a6466fab6d4eb8f412ce`.
FIX_BASE: `b69e9114626e21292104240cc70e912485a90c54`.
Managed detached worktree: `/Users/macbook-dev/.codex/worktrees/prompt-mix-ins/tldw_chatbook`.
TASK-38544 remains **In Progress for the controller's single scoped re-review**.
I1–I4 and M1/M2/M3/M5 implemented. M4 explicitly deferred. No helpers, full suite, push, merge, original-checkout edits, startup fix or unrelated cleanup.

Task AC4–7 and final implementation plan were added via backlog-py before product changes; now checked via CLI with implementation notes. ADR-286 was amended before implementing the controller-approved historical handoff contract. This extends the existing immutable fork snapshot and restricted transaction contribution; it does not create another persistence engine or weaken fresh acceptance.

## Changes and boundaries

- **I1:** Native SQL copy reads coherent owned history, retains Retry predecessors, writes receipt rows/items, then publishes each assistant's exact source marker once. Same-row multi-Retry chains no longer repeatedly invoke fresh marker transition. Empty snapshots, deduplication, lineage and source-purge independence remain. Fresh `write_attempt` and `_mark_assistant` remain strict.
- **I2:** `MixinForkState`/`MixinForkAttempt` hold immutable exact bodies and body-free marker fragments. Historical predecessors outside the prefix have explicit nullable anchors; every nonnull destination anchor is validated against conversation and role. The fork fence captures and rechecks history, durable commit rechecks inside its caller transaction, and live publication installs verified count/status/marker before registering the session. Temporary forks retain their own history across source close, nested forks, original Retry and promotion. Historical contribution runs before fresh batches, so the latest Retry marker wins without restamping or duplicate active batches. Fork history has no replacement/result-revision/trace-call fields. Active preferences start empty. Partial/future/lost/unreadable markers remain unavailable, never silently legacy. Action labels use the derived unavailable classification; receipt ownership remains replay authority.
- **I3:** Fresh Regenerate, edit-and-resend and Continue retain the existing shared source/currentness witness across provider resolution, hooks and transformations and refuse before mutation/dispatch. Original Retry, accepted queue snapshots and externally supplied complete configuration keep their existing independent custody; no current library capture was added to those paths.
- **I4:** Ordinary server Browse checks snapshot capability before switching mode. Refusal preserves Browse/source and gives bounded guidance. Awaited mode completion verifies exact state and source identity; absent/failed/stale mounted views do not receive an editor handoff. Mounted local editor control passes.
- **M1:** Archive fixture UPDATEs sqlite_sequence, INSERTs only when absent, and asserts allocated ID971971971. SQLite's internal name column is not unique: old INSERT OR REPLACE duplicated v6's row and allocated ID1, falsely matching common log digits. All five original privacy assertions remain. This was a current-only fixture defect, not production privacy leakage or inherited-original failure.
- **M2:** Restored generation test saves the original scalar attempt ID before mutable-message changes and requires live restored count exactly1.
- **M3:** Expected unavailable/conflicting/changed mix-in Preview yields disable/remove/refresh guidance and bounded info logging; unexpected errors retain their existing diagnostic path. Actual current UI capture confirms the new copy.
- **M5:** ADR/spec/plan status and architecture document implemented behavior pending scoped re-review. Historical whole-journey live evidence remains qualified to its actual commit.

## Final verification commands and outputs

All pytest commands used `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python`, from the managed worktree unless explicitly baseline-qualified. No counts below are summed across overlapping runs.

The **exact executable final command** is preserved in `final-fix/mixins-final-covering-command.sh`; it runs these named nodes with `-q --tb=short --show-capture=no -p no:cacheprovider -p mixins_final_fd_audit --basetemp=/private/tmp/mixins-final-covering-complete` and `PYTHONPATH=/private/tmp` (the read-only FD audit plugin is also preserved):

```text
Tests/Chat/test_console_prompt_mixin_forks.py
Tests/DB/test_console_prompt_mixins.py::test_copy_same_row_retry_chain_publishes_exact_marker_once
Tests/DB/test_console_prompt_mixins.py::test_legacy_fork_ignores_unrelated_malformed_metadata
Tests/Chat/test_console_prompt_mixin_retry.py::test_current_choice_replay_refuses_source_rebind
Tests/Chat/test_console_prompt_mixin_retry.py::test_continue_refuses_source_rebind_during_hook
Tests/Chat/test_console_prompt_mixin_retry.py::test_unavailable_mixin_preview_has_bounded_recovery_copy
Tests/Chat/test_console_prompt_mixin_retry.py::test_retry_original_regenerate_current_and_clean_history
Tests/Chat/test_console_prompt_mixin_journey.py::test_local_queue_retry_current_regenerate_reopen
Tests/Chat/test_console_prompt_mixin_journey.py::test_selected_generation_never_keeps_another_receipts_count
Tests/UI/test_console_prompt_mixins.py::test_ordinary_browse_mixin_capability_refusal_and_local_open
Tests/UI/test_console_prompt_mixins.py::test_browse_mixin_open_awaited_transition_is_fenced
Tests/UI/test_console_prompt_mixins.py::test_held_save_preserves_newer_raw_editor_draft_and_can_save_again
Tests/Chatbooks/test_chatbook_prompt_round_trip.py::test_prompt_export_failure_is_atomic_bounded_and_private
Tests/Chat/test_console_chat_fork.py::test_fork_public_record_fields_are_exact
```

Output: **51 passed in25.78s**, exit0 (`mixins-final-covering-complete.log`). Product/source files were unchanged afterward and match the current actual live capture. The separate finalized new fork module run (`-q --tb=short --show-capture=no -p no:cacheprovider --basetemp=/private/tmp/mixins-final-fork-complete3`) was **14 passed in5.97s**.

Earlier affected native owner run used the same flags and `--basetemp=/private/tmp/mixins-final-native-owners`, with whole modules `test_console_prompt_mixin_forks.py`, `test_console_chat_fork.py`, `test_console_chat_fork_persistence.py`, `test_console_chat_store_atomic_promotion.py`, `test_console_transaction_contribution.py`, `test_console_prompt_mixin_admission.py`, `Tests/UI/test_console_prompt_mixins.py`, plus the source-rebind/Continue-hook/Preview nodes and five archive nodes listed above. Output: **475 passed,4 failed in168.16s**. Two failures were exact public field inventories (updated explicitly with the new historical field); one was the known sparse-policy node; one was held Save's Select child not yet mounted (test now waits for `#mixin-placement #label`). Corrected exact-node follow-up: **32 passed in11.96s**. Final51 includes the field inventories and held Save again. The wide owner run was not repeated.

**FD evidence:** that earlier 479-case run emitted `open file descriptors grew by 761 over the test session (start=14, end=775, limit=200) — possible fd leak`. Its aggregate cause is unqualified; it is not claimed pristine or wholly inherited. Own new fixtures now explicitly close sessions/runtime/DB and wrap the native trace harness cleanup. Read-only per-node observations in the final run: first new async test14→17 (+3), remaining13 new cases17→17 (zero each). This does not prove every descriptor globally closes or erase prior debt. Final51 emits no FD sentinel warning. No GC forcing, warning suppression, threshold override or unrelated global leak fix was introduced.

### Baseline qualifications

Exact sparse-policy node: `Tests/Chat/test_console_chat_store_atomic_promotion.py::test_promotion_persists_sparse_context_policy_inside_the_bundle`.
Exact held Save node: `Tests/UI/test_console_prompt_mixins.py::test_held_save_preserves_newer_raw_editor_draft_and_can_save_again`.

Both nodes run with `-p original_native_admission -q --tb=short --show-capture=no -p no:cacheprovider`, PYTHONPATH pointing to the preserved `original-attribution` plugin directory. Current worktree: **1 failed,1 passed in2.96s**, `/private/tmp/mixins-final-current-baseline-nodes.log`. Exact FIX_BASE archive cwd `/private/tmp/mixins-final-fixbase-source`: **1 failed,1 passed in7.69s**, `/private/tmp/mixins-final-fixbase-nodes.log`. Basetemps are the matching log stem without `.log`. Sparse policy fails with `AssertionError: postcommit context-policy write`; held Save passes both, so its aggregate timing failure is not called original-inherited.

Controller independently ran only the sparse-policy node against exact original `d83ca84cdbd8cb934ba747bcb7a2179b934904ed`, same legitimate bootstrap admission: **1 failed in1.12s**, same postcommit context-policy path. Exact argv/cwd/plugin and output are in `original-attribution/promotion-sparse-original-result.json` and `.log`. This now truly qualifies original causality, unlike the earlier Task3BASE-only assertion. No current rerun was needed after that original attribution arrived.

Archive privacy original5pass/current5fail audits are preserved byte-for-byte in the four controller-named files under `original-attribution`: `archive-privacy.log`, `archive-privacy-result.json`, `current-archive-privacy.log`, `current-archive-privacy-result.json`. Only these four and the later explicitly authorized two sparse-policy audit files were copied from the exact root audit directory; no sharedtmp globs.

### Retained red/green iterations

All named iteration logs are preserved in `final-fix/`, including failures; they are not extra green current results.

| Log suffix | Result / attribution |
| --- | --- |
| final-fix-red | SQL2 expected marker-conflict reds; six fixture API errors (`update_draft` corrected to existing `set_session_draft`) |
| final-fix-red2 | Six real current-choice source replacement reds: accepted/dispatched incorrectly |
| final-fix-green1 | SQL2 + source6 + stronger scalar generation1:9 passed6.92s |
| final-forks1 | 10 failed1 passed: stale returned message snapshots, invalid boundary fixture; Preview passed |
| final-forks2 | 2 failed10 passed7.04s: fixture conversation/session identity collision and local artifact identifier |
| final-owners1 | 2 failed129 passed106.30s: stale post-Retry assertion and malformed legacy metadata handling; fixed fail-closed marker classification |
| final-forks3 | 13 passed7.64s |
| final-barriers | 1 failed25 passed14.53s: test treated nested DB transaction as savepoint; corrected to actual parent rollback |
| final-native-owners | 475 passed4 failed; qualifications above |
| final-native-followup | 32 passed11.96s |
| final-covering | 1 failed34 passed20.89s: failed-assistant fixture changed a returned copy, so legacy Retry action absent |
| final-fork-complete | 9 failed5 passed8.38s: fixture attempted complete→failed terminal transition |
| final-fork-complete2 | 9 failed5 passed7.32s: pending message editing correctly refused; switched to actual stream chunk→failed owner lifecycle |
| final-fork-complete3 | 14 passed5.97s, finalized cleanup and lifecycle |
| final-covering-complete | 51 passed25.78s, final current source |

The very early raw red log includes full capture; later invocations used `--show-capture=no`. Full historical logs retain their exact bytes and synthetic fixture payloads. Reproduction commands for the final state, the native-owner affected set and baseline qualification are given above; initial exploratory fixture iterations are not represented as rerunnable final-state failures.

## Static/source integrity

Ran the preserved `mixins-final-static.py`: scoped changed-line formatting against FIX_BASE (new test whole-file), then per-file Ruff JSON diagnostic multiplicities `(code,message)` against exact `git show FIX_BASE:path` bytes. Output `{}` introduced; **427 baseline /427 current diagnostics**, no introduced per-file multiplicities. This is inherited lint debt, not a whole-file lint-clean claim. JSON details are in `mixins-final-lint.json`.

`git diff --check -- tldw_chatbook Tests Docs/Architecture Docs/superpowers/plans Docs/superpowers/specs backlog` exited0. Changed Python files parsed via `ast.parse`; staged implementation whitespace check passed before local commit. Raw historical QA, terminal cells and ANSI logs are intentionally not whitespace-normalized.

Unchanged original launcher blobs verified with `git hash-object`: `app.py`=`ff4a7bc951ab33a82e4a03ea94bac7eae67d7fde`; `app_collections_wiring.py`=`625b8c08d3ee56bee3559123d3315f8adfd02cec`. Existing module startup ordering defect is source-qualified, not fixed or falsely claimed live-reproduced on original. Git commit emitted existing gc/unreachable-object warnings; no repository maintenance was attempted.

## Narrow current actual UI check

Existing private profile/launcher and SGR click helper were reused. `final-fix/live-preview/launch.json` records exact launch, profile and actions. Current actual console-script app on a160×45 tmux PTY reopened the saved chat whose enabled Scene source was already removed in earlier qualification. Menu→Prompts→Mix-ins showed unavailable Scene; Preview next send rendered:

`Mix-ins unavailable. Disable or remove unavailable selections, or refresh and retry Preview.`

and `Mix-ins unavailable; draft kept.` Current plain/ANSI frames and log suffix are preserved. This narrow check used an empty current draft; retained-draft correctness is covered by native tests and historical Send evidence, not falsely claimed as a new UI draft-preservation experiment. No provider Send/responder/full replay was run.

The controller emitted bounded info without the old expected-error traceback. The lower Prompts DB owner still emits bounded `transaction: rolling back category=ValueError`; this is not called entirely error-free logging. Normal Ctrl+Q at19:42:29 logged app/chat unmount, MCP disconnect, cached client and timers closed. Both recorded app PID41013 and isolated tmux server are absent. `teardown.json` retains exact checks; detached launcher exit code was not retained, so no exit0 claim. Current77 source hashes are unchanged since launch and match implementation commit d7744a78. PTY text/ANSI are actual app cells, not native OS font/paint proof.

Historical whole-journey live77 remains untouched and is verified against `247c9eb19ddbf49e49d55c7c1851f72fed2616a2`; it is not evidence later code ran then. Current product-source manifest independently covers final bytes. Full original final-review.md is preserved unchanged under `slices/`.

## Self-review and residuals

Reviewed connected copy/read/write, source fence, fork capture/stage/register/promotion, transient replay and generation action paths. Checked exact selected marker (not chronology), detached predecessor ownership, strict fresh anchors, parent rollback, late source mutation refusal, immutable tuple validation, empty preferences, all-three proof omission, actual original Retry after source purge, latest Retry marker surviving promotion, originalRetry/queue exemption, and bounded source failure reporting. No unsafe legacy fallback or blanket marker overwrite introduced. Exact public field inventory remains explicit.

Residuals: original sparse context-policy failure; unqualified earlier aggregate FD warning; M4 retiring browser worker log deferred; unchanged module launcher defect; no external provider/cache-hit/latency/cost qualification; no OS font fidelity proof. Historical predecessor540/packaging1 and old live whole journey were not rerun. No full suite. Root owns the single scoped re-review; this implementation task stays In Progress.

Evidence-only commit follows this report. `evidence-manifest.json` excludes itself and covers retained exact artifacts; `product-source-manifest.json` covers current sources/tests. The verifier checks filesystem/index or committed revision, historical live77 against247c9eb, and current narrow live77 against d7744a78. Final commit IDs and verifier output are delivered in the handoff; no post-verification product change is intended.
