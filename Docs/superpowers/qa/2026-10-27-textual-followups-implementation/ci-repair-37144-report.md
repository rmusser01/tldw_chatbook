# TASK-37144 implementation report

Status: DONE (implementer scope; task remains In Progress pending root review/CI).
Commit: `a0457fe1c7d9ba26d2441662bf9d5ac49a342c3a` — `test: isolate Console typing census from elapsed draft pauses`.
Base: `e96b3507be115e9790b4d37beafe471f5785252d`.
Worktree: `C:/Users/GDesktop-1/.codex/worktrees/textual-settings-patterns/tldw_tui` for every command. No remote, ref, PR, review-request or subagent actions were performed.

The fixture now applies its existing storage-mode draft-timer hold to every typing census. The new regression uses real Console/Pilot dispatch, native Textual Timer handles and replacement/coalescing, and actual coordinator callbacks. Its clock advances 0.21 seconds after each real key; delivery occurs only for the still-owned native handle whose scheduled interval is due. Every retained timer is stopped and its task drained in finally. It asserts all 24 keys reached the composer and native timers were armed, retaining the positive readiness floor and unchanged three-builds-per-key bound.

Following root's coverage review, the final control seeds 400 messages and enables a default-false `draft_pause` observation. After counting the typing burst, that observation independently calls the real coordinator stop/refresh and drains native publications, returns pause-prefixed counters, and restores the typing counts. The test asserts positive paused readiness and zero history/snapshot/projected-row traversal in both phases, with the existing context-estimate maximum of one row. This closes the pure-memory pause coverage gap without adding a paused CPU ceiling. The existing storage pause/idle/revisit callback paths and ceilings remain equivalent to base.

Only these three paths are committed:
- `Tests/Performance/test_console_keystroke_work_census.py`
- `.github/workflows/perf-guard.yml` — exactly one target added to the existing keystroke step; all other settings retained.
- `backlog/tasks/task-37144 - Keep-Console-typing-census-independent-of-elapsed-draft-pause-timers.md` — bounded plan extension, implementation evidence, checked AC1/2, navigable completed-owner/ADR links; AC3 remains unchecked.

## Native TDD evidence

Interpreter for every command: `C:/Users/GDesktop-1/Working/Github/tldw_tui/.venv/Scripts/python.exe -X utf8`.
The existing runner supplies a separate native private profile/TEMP tree before imports and keeps the existing 180-second pytest timeout. The existing Pilot watchdog remains 120 seconds. No full suites were run; App batches were announced and run one at a time.

Actual commands (prefix each with that interpreter):

```text
.test-tmp/textual-37031-37034/run-targeted-stream.py ci-repair37144-red01 Tests/Performance/test_console_keystroke_work_census.py::test_typing_census_excludes_elapsed_draft_pause_work
.test-tmp/textual-37031-37034/run-targeted-stream.py ci-repair37144-green01 Tests/Performance/test_console_keystroke_work_census.py::test_typing_census_excludes_elapsed_draft_pause_work Tests/Performance/test_console_keystroke_work_census.py::test_typing_never_snapshots_the_transcript Tests/Performance/test_console_keystroke_work_census.py::test_keystroke_work_does_not_scale_with_transcript_length
.test-tmp/textual-37031-37034/run-targeted-stream.py ci-repair37144-red02 Tests/Performance/test_console_keystroke_work_census.py::test_typing_census_excludes_elapsed_draft_pause_work
.test-tmp/textual-37031-37034/run-targeted-stream.py ci-repair37144-green02 Tests/Performance/test_console_keystroke_work_census.py::test_typing_census_excludes_elapsed_draft_pause_work
```

RED01 ran before the fixture fix. RED02 ran the strengthened final control with only the hoisted timer hold temporarily placed back under `if storage_units`; its enclosing subprocess wrapper restored exact final bytes in finally after the child/runner exited. Both RED failures were the intended assertion `4.041666666666667 <= 3` (97 readiness builds across 24 keys), not setup, teardown or skipped-test failures. Both observed 24 real timers and 24 native callback deliveries during typing.

Actual child XML outcomes:

| Run | Actual child | Outcome | Child time |
| --- | --- | --- | --- |
| red01 | new control, initial empty transcript | Expected readiness-bound failure | 92.420 s |
| green01 | new control, initial empty transcript | Pass, 25 builds/24 keys, 24 native timers | 90.088 s |
| green01 | existing transcript-snapshot guard | Pass | 86.241 s |
| green01 | existing empty/400-message scaling guard | Pass, both arms 25 builds, zero history rows | 139.033 s |
| red02 | final 400-message control plus explicit pause | Expected readiness-bound failure; separate pause 3 builds, zero history/snapshot rows | 89.851 s |
| green02 | final 400-message control plus explicit pause | Pass: typing 25 builds, separate pause 3 builds | 93.563 s |

Every actual child had zero errors/skips; only RED01/02 had the single expected failure. GREEN02 observed 24 native timers, zero due callbacks during held typing, zero typing/pause snapshots and history/projected rows, and context-estimate maxima 0/1. Actual runtime-source assertions for App, ChatScreen and ConsoleDraftSpendRefresh all resolve this worktree. The wrapper reports agree with the children; no pass claim relies only on wrapper dots.

Actual child XML lives under each run directory at `pytest/<case-dir>/private-profile-process/pytest.xml`. Consolidated exact child names/properties/paths: `.test-tmp/textual-37031-37034/ci-repair37144-child-receipts.json`. Runner commands and parent receipts are in each run's `command.json`, `verification.json`, `result.xml` and `run.log`.

## Static qualification and source preservation

Command: interpreter above + `.test-tmp/textual-37031-37034/ci-repair37144-final-static-proof.py` (exit 0). Reproducible script and JSON receipt are retained at that path and adjacent `.json`.

The proof reconstructs the exact already-passed GREEN01 full source bytes and matches SHA256 `67a4e18ec403413d81b989e40fa6a6562692ecf930a4476b220f41183d6d4b59`. Partial evaluation of final `draft_pause=False` yields an executable helper AST identical to GREEN01, so the two existing passed guards were source-qualified after the observation was added instead of blindly repeated. Partial evaluation of `storage_units=True, draft_pause=False` matches original base. All 23 other existing functions and every module assignment/ceiling match base. No fresh local storage-unit run is claimed; its preserved native callbacks/ceilings are qualified by exact AST equivalence, with prior actual hosted storage evidence remaining root-owned.

The YAML proof removes the one new executable target line and requires exact equality with base; it also parses the final three executable pytest targets and original timeout. Production working-tree diff is empty, including no untracked production files. Production Git objects remain:
- Entire `tldw_chatbook` tree: `e4265ce072b5213a57d64359ebf0184d0dc761c4`.
- `Docs/security/production-diagnostic-inventory.json`: `283912e7f06df2c37cbb1547ba0e7f9dc3a590d6`.

Final working-file SHA256 values:

| Path | SHA256 |
| --- | --- |
| Tests/Performance/test_console_keystroke_work_census.py | b27168b66459d914167c87f4c6c42edf9e6d521e7b96e16f25676b3f7f9bc773 |
| .github/workflows/perf-guard.yml | a35826e44acffc0d7c9604b063f570a1120f43475036a73f160bcf55d2aba68c |
| tldw_chatbook/app.py | bd16a79f5b8e612b9a19bffc3b60a733dc55d3ed24fc38ac52ce4e6171b2c83d |
| tldw_chatbook/UI/Screens/chat_screen.py | be1c1db087f2c339a47b27e20b9d7db4d2eb8b77bc71446bd1025dec32f829ea |
| tldw_chatbook/UI/Console_Modules/console_spend_projection.py | 73c5fee2b2b1a7058c6489084b5385e162b142bd5bdba506738bcd9b8c76741a |

Commands `-m ruff check --no-cache Tests/Performance/test_console_keystroke_work_census.py` and `git diff --check -- <three owned paths>` pass. Full-file `ruff format --check` exits 1 for five inherited formatting regions. The proof compares baseline and final formatter edits and requires those exact same regions/content, with no new format debt; no unrelated formatting was applied. An initial cached Ruff invocation hit the sandbox cache restriction and was replaced by `--no-cache`. An intermediate proof assembly could not find Ruff distribution metadata; the final script uses its installed CLI version and exits 0. A standalone formatter snippet needed EOF normalization; the full-source comparison is the final formatting evidence.

Self-review: inspected the complete owned diff, actual native child XML including pause counters, exact workflow change, source hashes and task state before committing. No unresolved correctness concern found. No production behavior, dependency, caps, timeouts, shards, CSS, import allowances or diagnostic/source budgets changed. Root-owned uncommitted QA files were left untouched.

Local versions: Python 3.12.10, pytest 8.3.5, Textual 8.2.8, Ruff 0.4.5. Inherited warnings remain: unset pytest-asyncio fixture-loop scope, Pydantic `exclude_if` deprecation, pydub/audioop deprecation, and missing optional audio-backend startup notices. Output is not claimed pristine. Current hosted Python/pytest qualification is still required.

ADR required: no; N/A. This is bounded test-harness phase attribution under existing `backlog/decisions/097-boot-budget-ratchets.md` and `backlog/decisions/103-fast-pr-lane-and-required-gate-aggregation.md`. Task notes link completed TASK-36390/33260/33005.2 and distinguish first-send layout TASK-37000. Root owns final scoped Superpowers review, QA/census integration, push and every actual current-head hosted run. This report does not claim those gates passed.
