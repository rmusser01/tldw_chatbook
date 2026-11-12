# Task 33164 — Capture-On archive recovery correction

Status: **DONE_WITH_CONCERNS for independent review**. Backlog remains In Progress; root owns its task/plan/ledger. Baseline: `0f5e537fe33553daf9d947834f90d14315a77772`. Worktree: `/Users/macbook-dev/.codex/worktrees/agent-archive-recovery/tldw_chatbook`.

A captured native follow-up now reaches the original second result's exact confirmation and restores only that conversation after explicit `chat_only` approval. The controller reseals the authorized metadata rider with the existing `CURRENT_TURN_TEXT` descriptor before the bridge. The provider factory's exact-value equality and all runtime/service guards are unchanged.

## Change and ownership proof

Only `console_chat_controller.py` and the existing `test_console_archive_recovery_followup.py` change. At the existing authenticated rider seam, the controller retains the unmodified rows, computes the final rows, and reseals the admitted request before shared first-request fitting/counting.

The private reseal:
- requires one admitted active user row, unchanged history, and an existing text-only-change validator that retains roles, attachments and sidecars;
- compares unmodified provider rows against the exact admitted frozen values (native locator annotations are excluded by the same admission convention);
- accepts only a saved source or already typed current-user text transform, retaining that transform's one original saved input without nesting;
- uses the existing repository to prove the source revision belongs to the exact accepted native user message, source conversation and user role;
- uses immutable replacement for only the active row and its descriptor, retaining frozen policy, metadata, durability, tools and attachment ownership fields;
- refuses a failed proof through `TraceCallPersistenceError(reservation_status="not_established")`, before the bridge's first capture reservation.

No generic changed-row fallback, historical artifact replay/relabel, guard relaxation, new export, UI/schema/dependency/configuration, or Cancel/Discard production change was introduced. Metadata/destination/Library/native/Persona/kill/lifetime gates remain upstream and unchanged. ADR check: no new ADR; direct correction under ADR097's current-turn ownership contract and ADR166's approved archive metadata scope.

## Test evidence

All commands ran in the isolated worktree with the installed interpreter, `PYTHONPATH=.` and pytest cache disabled. External transport is replaced through `ConsoleProviderGateway(chat_api_call_fn=...)`. Native runtime/controller/SQLite/agent/factory/preparation/dispatch/settlement remain real. The fixture denies and records DNS resolution; the existing repository guard denies and records IPv4/IPv6 socket I/O. Final GREEN has no network-guard errors or fixture teardown errors.

Command prefix:

```sh
PYTHONPATH=. /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -p no:cacheprovider
```

| Phase | Arguments following prefix | Raw exit | Result / classification |
| --- | --- | --- | --- |
| red | `Tests/Chat/test_console_archive_recovery_followup.py::test_captured_native_turn_restores_original_second_row -q --basetemp=/private/tmp/archive-capture-task1-red-stores` | 1 | 2 failed, 2 errors, 6.98s. Invalid setup: fake transport used nonexistent `ProviderToolCalls.calls`; first run stopped before the intended successor. |
| red2 | Same node, `-q --basetemp=/private/tmp/archive-capture-task1-red2-stores` | 1 | 2 failed, 2 errors, 9.96s. Direct/RAG bodies establish intended `trace_turn_unavailable`; separate existing blocked-checkpoint teardown failure. |
| red3 | Same node, `-q --basetemp=/private/tmp/archive-capture-task1-red3-stores` | 1 | 2 failed, 2 errors, 8.26s. Same body failure; test-only Discard/custody attempt did not resolve pre-fix cleanup. |
| red4 | `'Tests/Chat/test_console_archive_recovery_followup.py::test_captured_native_turn_restores_original_second_row[True]' -q --basetemp=/private/tmp/archive-capture-task1-red4-stores` | 1 | 1 failed, 1 error, 6.09s. Exact existing Cancel refusal captured; no further equivalent pre-fix probes. |
| green1 | Full new node, `-q --basetemp=/private/tmp/archive-capture-task1-green1-stores` | 0 | 2 passed, 8.62s. Direct/RAG complete and close cleanly. |
| controls1 | `Tests/Chat/test_console_archive_recovery_followup.py -q -k 'rider_composes or reseal_refuses' --basetemp=/private/tmp/archive-capture-task1-controls1-stores` | 0 | 9 passed, 30 deselected, 27.08s. |
| final | `Tests/Chat/test_console_archive_recovery_followup.py -q --basetemp=/private/tmp/archive-capture-task1-final-stores` | 0 | **39 passed, 117.45s**. Affected-file final selection; no full suite or prior 31-Open/63-check wave repeat. |

[Byte-preserved raw logs and SHA256 manifest](2026-09-27-agent-archive-recovery-capture-on/manifest.json) include every phase above. [Final raw output](2026-09-27-agent-archive-recovery-capture-on/final.log) and [read-only final pins/hashes](2026-09-27-agent-archive-recovery-capture-on/verification.json) are retained. All 14 copied artifacts were compared byte-for-byte to their original `/private/tmp/archive-capture-task1-*` files.

The primary two-mode test uses the actual retained page metadata/ref/order to request row 2. It verifies the intended target is still archived before approval; only that target changes after exact `chat_only` confirmation; the other conversation/workspace and active source session remain unchanged. Reconstructed first follow-up payload equals the actual provider-prepared payload consumed by the fake adapter. Every follow-up call pins the same exact current-user saved revision. Saved text stays `Restore the second one`; metadata is absent from ordinary transcript rows. The next ordinary captured send also completes. Exact predecessor call rows and reconstructed requests remain equal before/after both successors.

Read-only final SQLite evidence records two flows, each with 3 search + 3 restore + 1 ordinary COMPLETE calls. Each turn has one exact user revision and one frozen policy; every pin's source message equals its call turn. Exact opaque identities are in `verification.json`.

Focused descriptor controls cover existing `CURRENT_TURN_TEXT` composition through controller/factory without nesting or changing source/policy/durability, plus unowned artifact, unknown UUID, foreign source, changed history, changed role, changed attachment shape, changed sidecar and missing current owner. Refusals create no trace call and leave prior capture evidence unchanged. Composition is a controlled already-typed input check; it does not qualify or repair the unrelated dictionary producer.

The final file also revalidates existing Capture-Off recovery, Blocked/frozen Blocked/legacy/native/provider/temporary/remote/Persona/ask/kill guards, destination mismatch/unknown destination, manual/missing-origin/imported/expiry/eviction/closed/source isolation, whole-page/ambiguity/omission bounds, and existing first-request fit/no-fit checks.

## Static evidence and self-review

Ruff executable: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/ruff`.

| Command | Exit | Result |
| --- | --- | --- |
| `ruff check --no-cache Tests/Chat/test_console_archive_recovery_followup.py` | 0 | All checks passed. |
| `ruff format --no-cache --check Tests/Chat/test_console_archive_recovery_followup.py` | 0 | Already formatted. |
| `ruff check --no-cache tldw_chatbook/Chat/console_chat_controller.py --output-format=json` | 1 | Existing 191 diagnostics. |
| `ruff check --no-cache /private/tmp/archive-capture-task1-baseline-controller.py --output-format=json` | 1 | Baseline 191 diagnostics; baseline bytes from `git show 0f5e537fe33553daf9d947834f90d14315a77772:tldw_chatbook/Chat/console_chat_controller.py`. |
| `ruff format --no-cache --check tldw_chatbook/Chat/console_chat_controller.py` | 1 | Whole file has existing formatting debt. |
| `ruff format --no-cache --config pyproject.toml --check /private/tmp/archive-capture-task1-baseline-controller.py` | 1 | Baseline also would be reformatted. |
| Exact `Counter((code, message))` comparison of baseline/current Ruff JSON | 0 | Identical multiset, 191 each; no added or removed diagnostic signature. |
| Formatter-output comparison restricted to new helper/insertion blocks | 0 | Zero pending formatting edits in changed blocks; unrelated controller formatting preserved. |
| `git diff --check` before raw-log staging | 0 | Source/authored report changes had no whitespace errors. |
| `git diff --cached --check` after raw-log staging | 2 | Raw RED logs retain 14 original trailing spaces; all findings are confined to the four exact paths below. |

Initial cache-enabled Ruff attempted a worktree cache write and reported permission denied. That preliminary command's individual raw exit was not retained (the shell continued to later commands); it is not represented as a successful check. All effective static checks above used `--no-cache`. Formatting was generated in temporary copies and the scoped changes applied through the normal reviewed apply tool, with no bypass/rejection.

Raw evidence bytes are preserved. The exact whitespace exclusions are `Docs/superpowers/qa/console/2026-09-27-agent-archive-recovery-capture-on/red.log`, `red2.log`, `red3.log`, and `red4.log` in that same directory. Scoped staged whitespace verification of source, tests and authored QA passes; the entire staged artifact diff is not whitespace-clean.

Self-review checked the complete source/test diff against all three ACs, exact-source/non-text/history failures, original transform composition, field preservation, pre-fitting insertion, upstream authority gates, confirmation/target effects and lifecycle cleanup. The trace factory/runtime/service files are unchanged, and their SHA256 values match the diagnosis baseline. No scope expansion or unresolved source defect was found in this correction.

Changed source SHA256:

- Controller: `795b8ebccff0360d62ac69bf2426dd23ef6f7bec07c21f26a30f63015cf9273d`.
- Existing follow-up test: `68ffa85f831cd95829379905903561dae696a7c252d8ac0f6e5ee693f64fe52a`.

## Preserved concerns and rulings

Root's ledger ruling accepts the intended Direct/RAG RED bodies while recording pre-fix checkpoint teardown separately. The exact existing Cancel API returned `Trace reservation status is unavailable.` for PAUSED / TRACE_CALL preparation `8b36a5b4-3a9f-4517-98e7-5bb6af8952b4`. Shutdown then hit `Resolve pending dispatch before changing conversation branches.` No production custody/branch guard was bypassed or fixed. All final GREEN fixtures settle and close cleanly.

The earlier read-only diagnosis remains at `/private/tmp/archive-capture-on-diagnosis.md`: its final isolated phase had 2 passed/1 teardown error; its preliminary invalid OpenAI transport setup attempted remote resolution and failed DNS. That historical incident is not a phase of this correction. This correction's documented fake adapter and DNS/socket guards never reached a real model/network endpoint.

Root's separate baseline dictionary control failed at line 162 (`alias second` vs `expanded second`); no dictionary producer change or rerun is included. Historical `unsupported_surface_change`, aggregate FD growth, provider errors and earlier live qualification remain unexplained by this task. Earlier 28/63/Capture-Off completion is preserved.

Whole-controller static debt prevents a claim of globally clean lint/format. Next Send pre-admission display/estimate still omits the late rider, as previously disclosed and approved. Independent review and any live Capture-On qualification remain root-owned; this automated fake-boundary GREEN is not a live walkthrough. No app launch, model request, download, normal-profile edit, full suite, subagent, push, PR or merge was performed.

Only corrective source/tests and this QA/evidence set are intended for commit. Root-owned task, plan and SDD ledger remain outside the commit; the workspace task report is a copy for handoff.

## Fix round 1 — I1 admitted authority lifetime

Base: `e8fa9b0b97811e46c7431689320622396555b219`. The [complete independent review](2026-09-27-agent-archive-recovery-capture-on/fix1/review.md) contains one Important finding, I1, with no Critical or new actionable Minor findings. Review SHA256: `ba68d4b214509df814efc7d6b3b4bb8a45466a06a6371a44a21e759fd7d9d238`. Root amended the task/plan before source work; they remain excluded.

I1 was verified against the real acquire/await/release flow. The existing bridge cleanup region now starts immediately after recovery admission and includes all rider generation/reseal work. Its existing cancellation/error handlers and exact-authority finally release are reused. The ownership helper, sanitized `not_established` exception, authority gates, provider factory and trace runtime/service are unchanged. No Cancel/Discard production logic was changed. The only subsequent source edits wrap imports/expressions to format the moved block.

Two focused native-custody cases use the existing real captured search, actual admit and release, native dispatch/controller/SQLite, documented fake external adapter, and inherited DNS/socket guards. Refusal supplies an unknown saved-revision descriptor to the real helper at the admitted seam. Cancellation holds that real helper on an explicit barrier, invokes the actual Stop API, and observes cleanup while the helper is still held. Each observes exactly one real release of the exact admitted pending authority, closure/removal/inertness, no run activation, no successor capture factory/transport/restore, unchanged archived targets and predecessor capture rows/reconstruction. Refusal remains a controlled blocked result with the content-free `Provider call trace persistence failed.` and `reservation_status="not_established"`.

The test finally releases the barrier, waits for the held reseal callable's explicit completion event, and gathers the native turn task. This is an application-work completion barrier, not a literal OS executor-thread join. It then uses the existing exact paused-preparation Cancel API if needed and requires no dispatch recovery owner. All final cases close without fixture teardown or network-guard errors. This narrow cleanup evidence does not repair or qualify the earlier unrelated blocked-checkpoint Cancel/Discard failure.

### New phases (old phases/hashes preserved)

The command prefix remains `PYTHONPATH=. /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -p no:cacheprovider`; working directory remains the isolated worktree. Each command redirected stdout/stderr to the matching `/private/tmp/archive-capture-task1-fix1-*.log`.

| Phase | Arguments after prefix | Raw exit | Result |
| --- | --- | --- | --- |
| fix1-red | `Tests/Chat/test_console_archive_recovery_followup.py::test_native_archive_reseal_releases_exact_admitted_authority -q --basetemp=/private/tmp/archive-capture-task1-fix1-red-stores` | 1 | 2 failed, 6.70s, no teardown errors. Refusal reaches exact pending-authority leak. Cancellation initially surfaces the pre-try Stop path's `Cannot mark a stopped message terminal.` before its custody assertion; not a passing cancellation control. |
| fix1-red2 | `'Tests/Chat/test_console_archive_recovery_followup.py::test_native_archive_reseal_releases_exact_admitted_authority[cancellation]' -q --basetemp=/private/tmp/archive-capture-task1-fix1-red2-stores` | 1 | 1 failed, 2.83s, no teardown errors. Test gathers the native outcome with exceptions retained so the exact authority leak is asserted first. No source change before this RED. |
| fix1-green | Full new custody node, `-q --basetemp=/private/tmp/archive-capture-task1-fix1-green-stores` | 0 | 2 passed, 4.93s. |
| fix1-final | `Tests/Chat/test_console_archive_recovery_followup.py::test_native_archive_reseal_releases_exact_admitted_authority Tests/Chat/test_console_archive_recovery_followup.py::test_captured_native_turn_restores_original_second_row Tests/Chat/test_console_archive_recovery_followup.py::test_archive_rider_composes_existing_transform_without_changing_its_source -q --basetemp=/private/tmp/archive-capture-task1-fix1-final-stores` | 0 | **5 passed, 21.62s**: custody refusal/cancellation, captured Direct/RAG positives, typed composition. |

[Fix-round manifest](2026-09-27-agent-archive-recovery-capture-on/fix1/manifest.json) preserves 15 new review/raw-output/JSON artifacts byte-for-byte. The original 39-pass final, original phases and source hashes are unchanged historical evidence. No full file/suite or older wave was repeated in this round.

### Fix-round static and final checks

Executable prefix for Ruff: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/ruff`.

| Command/check | Raw exit | Result |
| --- | --- | --- |
| `ruff check --no-cache Tests/Chat/test_console_archive_recovery_followup.py` (initial and final outputs retained) | 0 each | Pass. |
| `ruff format --no-cache --check Tests/Chat/test_console_archive_recovery_followup.py` | 1 initial / 0 final | One new long assertion was wrapped through normal apply; final formatted. |
| `ruff format --no-cache --diff Tests/Chat/test_console_archive_recovery_followup.py` | 1 | Printed only that assertion's formatting hunk; no formatter source write. |
| `ruff check --no-cache tldw_chatbook/Chat/console_chat_controller.py --output-format=json` | 1 each | Initial 192 diagnostics included one moved-import formatting signature; normal apply wrapped that import. Final 191 signatures exactly match the retained e8 baseline. |
| `ruff format --no-cache --check tldw_chatbook/Chat/console_chat_controller.py` | 1 | Existing whole-file debt, retained as a limitation. |
| Inline Python signature/formatter checks | 1 / 1 / 0 final | First caught the moved import; second retained two formatting differences, one in unchanged admission lines and one newly indented expression. The expression was wrapped, unchanged admission debt excluded from the changed-block scope. Final exact signature comparison and changed-block formatter comparison pass. |
| Inline Python AST ancestry/source hashes | 0 | Reseal await26619 is under cleanup try26573, whose finally releases at26886; zero pending formatter edits in modified protected block26573–26626. Exact details in `static-final.json`. |
| Inline Python read-only SQLite evidence | 1 / 1 / 0 final | First query used nonexistent `status` (actual column `state`); second included pytest's duplicate `current` symlink. Corrected read-only query excludes symlink and passes: each actual custody store has only 3 predecessor COMPLETE calls and both targets remain archived. No DB writes or test rerun. |
| `git diff --check -- tldw_chatbook/Chat/console_chat_controller.py Tests/Chat/test_console_archive_recovery_followup.py` | 0 | Source/test whitespace clean. |
| `git diff --cached --check` after new raw evidence staging | 2 | 34 original trailing-space lines, confined to `fix1/red.log` (30) and `fix1/red2.log` (4); raw bytes unchanged. Output: `/private/tmp/archive-capture-task1-fix1-whitespace.log`. |
| `git diff --cached --check -- . ':!Docs/superpowers/qa/console/2026-09-27-agent-archive-recovery-capture-on/fix1/red.log' ':!Docs/superpowers/qa/console/2026-09-27-agent-archive-recovery-capture-on/fix1/red2.log'` | 0 | All other staged source/test/authored QA/review/JSON/raw evidence paths clean. |

Root's sixth ruling explicitly preserves those two raw RED logs and the four original named RED logs. These six exact evidence paths are the only whitespace exclusions; the entire artifact diff is not whitespace-clean. No other path is waived.

The [static JSON](2026-09-27-agent-archive-recovery-capture-on/fix1/static-final.json) and [read-only store JSON](2026-09-27-agent-archive-recovery-capture-on/fix1/verification.json) retain exact identities and hashes. Final controller SHA256: `0652d3fe258d974c06680889f558a2a993de75ed8da7300cba970d471d932a7a`. Final test SHA256: `c2f69ace650aa9d2d8c2efce697dac223de54fb41f29963b166bd258cbcdffe5`. Final formatter-only source wraps followed the test run; no executable logic changed after the 5-pass phase.

Self-review read the entire I1 source/test diff, checked exact acquire/release ownership, sanitized controlled refusal, cancellation/held-callable cleanup, trace/transport absence and positive/composition preservation. No new ADR is needed: existing ADR097/166 contracts apply. No source apply rejection occurred. This round addresses I1 only; inherited controller static debt, raw-log whitespace, Next Send omission, historical diagnostic/Cancel/dictionary/FD/live limitations remain explicitly preserved. Scoped independent re-review remains pending.

## Final scoped review

[Independent fix review](2026-09-27-agent-archive-recovery-capture-on/fix1/fix-review.md) approved the I1 correction in `e8fa9b0b978`→`03caf0a6b4bd`: spec compliant, quality Approved, 0 Critical/Important/Minor actionable findings. The reviewer read the complete scoped diff and observed evidence without rerunning tests. Root read the complete report and independently verified the 15 new artifacts, 14 immutable earlier artifacts, exact final source/test hashes, and three unchanged capture boundary files.

TASK-33164 is complete on the local branch. Verification remains phase-specific: the earlier 39-pass selection qualifies `e8fa9b0b978`; the five covering checks qualify the cleanup fix's executable behavior, with the documented later formatting-only wraps. Whole-controller static debt, the six exact raw-log whitespace exceptions, and historical/live qualification limits above remain unchanged. No live Capture-On walkthrough or full sweep is claimed.

## Supplemental live qualification — 2026-09-28

The user authorized a bounded Capture-On native walkthrough. [Live evidence](2026-09-28-agent-archive-recovery-capture-live.md) qualifies RAG-only natural selection/confirmed restore/original-history Open. Direct exposed an unsupported constructor argument; the [one-line correction and15checks](2026-09-28-console-direct-library-factory.md) are independently approved. Fresh corrected Direct performs tool discovery and archive search, reaching the shared10-attempt cap before its restore follow-up. TASK33164 AC1 is reopened/In Progress; the original automated phases and reviews remain completed. No fresh Direct live restore claim.
