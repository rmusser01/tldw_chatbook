# TASK-37268 implementation report

Status: **DONE_WITH_CONCERNS** — implementation and scoped verification complete; one legacy pre-export startup control remains unqualified locally, and root-owned inventory/review/CI/closure remain.

Workspace: `C:/Users/GDesktop-1/.codex/worktrees/textual-settings-patterns/tldw_tui`.
Base supplied by root: `41e530c9f6a8936f85f5c09107907e22d27cd55d`.

## Scope and implementation

- `tldw_chatbook/UI/Library_Modules/library_file_export.py`: only `export_library_note_file` changes. It captures the rendered payload before admission, retains the validated target and original Notes operation token, and synchronously admits a native screen worker in `library-note-export`, `exclusive=False`. Both FileSave and explicit Replace callbacks return after admission. Only `write_export_text` runs through stdlib `asyncio.to_thread`. No-clobber `FileExistsError` rechecks and confirmation/error/success publication stay on-loop. The late metadata recheck is inside the recoverable failure boundary. Existing Notes operation admission, cancellation, writer/format/newline semantics and source services remain authoritative.
- New `Tests/UI/test_library_note_export_responsiveness.py`: native Library action, actual FileSave keyboard journey, actual Replace/Cancel controls and atomic writer; independent literal expected bytes; main/write thread identities and actual App F12 acknowledgment before release. Ordinary/held new and Replace, racing-file Cancel/Replace, failed collision recheck, failed atomic Replace/text retry, changed Notes region and retired screen controls. All holds release and all work drains inside the App lifetime, including failure paths.
- Root expanded ownership for exactly `Tests/UI/test_library_shell.py::test_library_shell_note_write_export_file_writes_expected_content`: preserve its existing content/notice/status assertions, but wait for the real completion status and native worker drain before reading the file. Its docstring now describes synchronous admission.
- Root expanded ownership for exactly `Tests/Library/test_library_file_export.py::test_note_export_to_a_new_file_succeeds_where_hard_links_are_unsupported`: reuse its adjacent `_prompt_worker_screen` native worker owner, await completion, preserve all hard-link refusal/content/notice/operation assertions. No product fallback for a SimpleNamespace lacking a worker API.
- `Docs/Architecture/library-shell.md`: document captured Notes write ownership, native scheduling, late collision/error behavior, source stand-ins and retirement limits.

No screen/controller changes were needed. No dependency, schema, service, shared fixture, global state owner, CSS, timeout, privacy guard, ACL, launcher, census, task metadata, Git mutation or subagent changes were made.

ADR required: no. ADR path: N/A; apply existing ADR-033/036/031/150/161 and the adjacent Prompt/native-worker pattern. The implementation preserves domain/service/storage boundaries. Source/newline fidelity remains with the shared writer and adjacent PR3096 ownership. No Git staging, commits, branch operations or rebases were performed.

## Test fidelity and limits

The App is the existing `LibraryHarness` with the real Library screen and native Textual message/worker/modal boundaries. Notes is the established seeded `StaticLibraryNotesScopeService` stand-in, **not a real Notes SQLite source**. Tests compare a deep copy of the complete seeded source records and independently specified output bytes. They do not prove remote behavior, a storage-latency distribution, or a complete OS terminal session.

Unrelated optional backend count services are disabled as in the audit. Native source workers must settle before export. If a source snapshot times out, the helper waits for the real Retry DOM (worker completion can precede its recompose), delivers that Button event and drains the real retry; no getter/body/guard is replaced and no timeout is increased.

The three-second dispatch watchdog is a cleanup bound, not a dispatch SLA. In the held controls the actual App handler acknowledges F12 and releases the writer. The correct result requires acknowledgment before release and a different writer/main thread. Region/retirement holds release in `finally` inside App lifetime; the underlying accepted atomic write is also drained after native screen retirement.

All App runs use the project Python `C:/Users/GDesktop-1/Working/Github/tldw_tui/.venv/Scripts/python.exe -X utf8` through `.test-tmp/textual-37266-37268/run-targeted.py`, escalated as authorized, with native-private profile/TEMP parents and the existing 180-second limits. One App batch at a time. No full suite was run, per the user/AGENTS instruction.

## RED and invalid attempts

Every run below is retained under `.test-tmp/textual-37266-37268/<label>/`, including parent command/log/XML/verification JSON. Exact private-child XML/logs/timelines were additionally archived under each run's `child-receipts/`, with `receipt-manifest.json` hashes. Do not count the invalid attempts as intended RED.

| Run | Classification | Actual outcome |
| --- | --- | --- |
| `task3-red-notes` | Invalid | Test-only removed screen ownership attribute; controller owns selected note ID. |
| `task3-red-notes-ready` | Invalid | Actual writer reached, but literal LF bytes disagreed with the current native Windows CRLF policy. Expected bytes now explicitly preserve the existing native policy, rather than changing product fidelity. |
| `task3-red-notes-native-newlines` | Invalid | Native source snapshot hit its existing five-second deadline; Retry had not recomposed when immediately queried. Helper now witnesses actual recovery DOM. |
| `task3-red-notes-recovery-ready` | Invalid intended assertion, useful observation | Actual writer/main thread `170616`, watchdog release, then App acknowledgment with `before_release=false`; exact output/source/operation checks reached. A later test-only focus assumption failed: established Notes focus is `library-note-body`, unlike Prompt's Export button. |
| `task3-red-notes-dispatch` | **Intended RED, unchanged product** | `Notes writer blocked App dispatch`, `[False] != [True]`. Writer/main thread `166688`; watchdog released before write completed, then App acknowledgment. Parent 1 failure/0 errors and actual child 1 failure/0 errors, no skip. |

Intended RED parent XML SHA-256: `b1c20ba5cd04d3accae8102dc597a35a9413cd0ae066580e6d7652c100af943c`.
Actual child XML SHA-256: `681aeb4760df83a64795cd6569ad40bdb09280e9fb2b27104b16bfd8ab1dcce3`.
Exact test source at intended RED: `.test-tmp/task3-owned-before/red-dispatch-test.py`.

## Source, static and diagnostic evidence

- `.test-tmp/task3-owned-before/sources.json` and raw snapshots preserve before sources. `.test-tmp/task3-base-proof.json` shows those sources equal the supplied Git base after CRLF normalization and records both raw Git-blob/workspace hashes. The first raw-byte comparison failed because the native checkout uses CRLF and Git blobs use LF; this is qualified rather than called byte identity.
- `.test-tmp/task3-owned-green/sources.json` pins all implementation/test sources before the new mounted batch. Documentation was subsequently clarified without changing runtime/test sources.
- `.test-tmp/task3-static-comparison.json`: the exporter and two existing test modules change only their authorized top-level functions; **every other top-level AST is equal** to the captured base. No unrelated string/Unicode mutation.
- Scoped Ruff `--no-cache` check passes on exporter, new regression and adapted Library helper module. They pass Ruff format check. Shell baseline and after JSONs are identical: three pre-existing E721 diagnostics; whole-file formatting debt predates this task. The three E721 locations are 531, 944 and 3755. `.test-tmp/task3-shell-function-format.json` verifies the isolated authorized function is formatted both before and after (one terminal newline normalization for the extraction). No broad shell formatting or pin increase.
- `.test-tmp/task3-focused-worker-dom-style.json`: existing guard collectors run on the actual changed exporter module: W001 synchronous worker targets 0, W002 post-await DOM 0, W003 roots/unresolved handoffs 0, ungrouped exclusive schedules 0, inline-style writes 0. This is a focused module check, not a claim that all package governance is green. An initial check adapter used the wrong private collector name `_violations` and failed its import before collecting; the corrected `_violations_in_tree` invocation produced the persisted clear result. No expensive unrelated full governance sweep was repeated.
- Exact owned Git diff was self-reviewed and `git diff --check` passed on tracked deliverables. Root owns staging/commits/reviews/closure.

Diagnostic inventory: **pending root-owned refresh; no pass claim**. The CLI drift check reported only `library_file_export.py`, count `4 -> 4`, digest `b4a748817136c4c4f8de -> c352738411d1464094ba`, no sink topology rows. `.test-tmp/task3-diagnostics.json` proves the same four diagnostic statements after mapping the single Notes nested owner `_write -> _fail`: same warning message, `note_id`, `type(exc).__name__`, level, multiplicity and no exception capture. No calls/warnings were removed or suppressed. Required artifact: `Docs/security/production-diagnostic-inventory.json`, owned by root after final source/rebase with full owner/sink conservation.

The inventory CLI emits drift to stderr. The initial stdout-only redirect is preserved as the empty `.test-tmp/task3-diagnostic-inventory.txt`; the surrounding PowerShell command returned zero because its final command was Get-Content, **not because inventory passed**. Tool output captured the drift branch (which returns one). A clearly labeled, separate transcription is `.test-tmp/task3-diagnostic-inventory-transcribed.txt`; it is not a raw receipt. Root will capture the fresh raw pre-refresh check correctly, per instruction; no redundant scan was run.

Inherited pytest-asyncio unset fixture-loop-scope, Pydantic `exclude_if`, audioop and optional audio backend warnings remain qualified. Legacy harness service-not-initialized log lines reflect the test stand-ins, not proof of actual Notes DB integration.

## Final runtime results and AC mapping

New mounted batch `task3-green-notes`: **10 passed**, parent and all ten actual isolated children each have zero failures/errors/skips. Parent session 1066.489 seconds (17m46s on this Windows host); parent XML SHA-256 `011551b4250ab2198444721bf161f95e3bc09582c47e6f98f2d519f9c4231ff8`. Raw parent/child receipts are archived under that run; `receipt-manifest.json` records all hashes. Runtime/test source hashes still match the exact pre-run pins.

Actual held evidence (`.test-tmp/task3-held-thread-evidence.json`): new destination writer/main `136352/171712`, Replace writer/main `171020/168112`; both App acknowledgments have `before_release=true` and precede actual write completion. Neither uses its watchdog release.

Full parent phases were requested with `--durations=0 --durations-min=0`. `.test-tmp/task3-green-timings.json` exports every new parent's actual inclusive XML time, displayed setup/call/teardown costs, actual child case-inclusive time, child session time and child XML hash/path. **Private-child split phases hidden by its default duration output are not zero**; case-inclusive and session totals are separately reported. Displayed two-decimal phase times have rounding; XML inclusive times are authoritative. These are test execution costs, not storage latency or dispatch percentiles.

| Case | Parent inclusive s | Parent setup/call/teardown s (displayed) | Child case inclusive s | Child session s | Child XML SHA-256 |
| --- | ---: | --- | ---: | ---: | --- |
| `dispatches_app_key_before_destination_release[new-ordinary]` | 96.901 | 1.28/95.32/0.31 | 33.092 | 71.665 | `d92895f3637b30ece90381e717b94a029190aca9d639121dd2912a1b40cf86d9` |
| `dispatches_app_key_before_destination_release[new-held-writer]` | 117.745 | 0.20/117.33/0.22 | 57.445 | 104.321 | `017d2f6429d8ba369428463071a224833f739157796cf822f2e71f2f9f388224` |
| `dispatches_app_key_before_destination_release[replace-ordinary]` | 110.152 | 0.01/109.96/0.18 | 43.451 | 89.921 | `1da7e7e60bd159c0d439894ac5e619cfaef988de0cbd8ad5e565db6506032363` |
| `dispatches_app_key_before_destination_release[replace-held-writer]` | 102.002 | 0.01/101.58/0.42 | 46.592 | 80.597 | `79ff89d619c78ce2cd1089fad62173f3824d792bf3f8fd93f2d94aa186424a69` |
| `late_collision_keeps_precious_file_until_answer[cancel]` | 104.050 | 0.02/103.79/0.24 | 36.519 | 82.397 | `48feeeaf98342a561756cd3dab7e9f7fa02668ade8b88f5f3054866c6065a425` |
| `late_collision_keeps_precious_file_until_answer[replace]` | 112.417 | 0.01/111.76/0.64 | 57.829 | 94.000 | `a06f99d85da60993affb9cc57509b18b8b6784af41782050623503e11f9b29e0` |
| `late_collision_keeps_precious_file_until_answer[recheck-error]` | 98.161 | 0.01/97.39/0.76 | 48.450 | 79.983 | `6e17bd4796b64a667720e4d43c3d1d6a1c23a95bbd548e142d81b4b7d909d6e4` |
| `accepted_write_preserves_changed_result_owner[context-region]` | 80.769 | 0.01/80.51/0.25 | 31.204 | 59.804 | `1bc7156a7252659873da58df044897b0347df9563dd48b14e8b29451df2918a6` |
| `accepted_write_preserves_changed_result_owner[retired-screen]` | 98.240 | 0.01/97.98/0.25 | 47.172 | 84.081 | `b98db2bd0ce555c848d8b6eaa7af17c2043a0c38bea5e1e1f19620975b2a5335` |
| `failed_replace_keeps_file_and_allows_text_retry` | 92.060 | 0.01/91.68/0.37 | 41.518 | 75.395 | `97cafe7829e62aae755825dad56c78d5a97345f5cd09382d5af8e8024d2e51e7` |

### Existing compatibility controls

`task3-existing-controls`: **18 passed, 1 failed, 0 errors, 0 skipped**, 341.28 seconds. Parent XML SHA-256 `d6e18d44a2eef96f0c6533c2b1233088f2dcfc68c218c4c1fea6acb4569880e7`. Command, complete log/XML and receipt manifest remain in that run directory.

Passed existing controls:

- Native compact precious-file Cancel; confirmed Replace and remembered next export folder.
- Both narrowly adapted existing Notes controls (ordinary write completion and all two hard-link refusal variants), with every original assertion retained; invalid-path rejection remains synchronous and passed.
- Neighboring Prompt result after another screen; both Prompt hard-link refusal variants; Prompt render failure/retry and late metadata failure/retry.
- Actual atomic failed Replace preservation/no temporary leak, no-clobber refusal, Replace exactly once, and four existing Notes Markdown/text/default-title/timestamp builder controls.

The sole failing case was `Tests/UI/test_library_export_replace_confirm.py::test_note_export_onto_an_existing_file_asks_escape_keeps_it_replace_writes`, **before export**, at line 241 `_open_note_editor` -> shared `_wait_for_selector('.library-notes-row')`: source snapshot hit its existing five-second deadline, and the row never mounted within the existing 30-second helper bound. No picker/write/Replace/Escape export logic was reached. This is not an export responsiveness failure or a path assertion result. New product work changes only a function body that was never called on that failing path; shared source/readiness owners and their ASTs are untouched. It is excluded as a usable local Escape/Replace control rather than counted green or claimed a fully reproduced baseline export outcome.

Root authorized one exact retry, with no body/helper/getter/timeout/source changes. `task3-existing-escape-retry` again failed at the same pre-export editor/source deadline: **1 failure, 0 errors, 0 skips**, 144.36 seconds; XML SHA-256 `1e5a1da8ed800a4832359500bec75680f7c7175922f475de8e793ca3cf8bfa33`. Raw receipt and manifest retained. No repeated broad batch or additional retry followed. The anticipated Windows path-substring assertion was **never reached** and is not classified as an observed failure.

Legacy profile replay is qualified: `.test-tmp/task3-controls/task3_bootstrap_controls.py` adds only the supported `bootstrap_profile` marker for exact selected existing shell success/path-rejection and Prompt result controls, skipping private wrappers/already-marked tests. Bodies, getters, guards and shared fixtures are unchanged. `.test-tmp/task3-controls/selectors.json` and `plugin-proof.json` record exact selections and plugin/JSON hashes. The invocation sets PYTHONPATH to that task-local directory, then runs the unchanged launcher with `-p task3_bootstrap_controls` and exact selectors. Existing Replace/Cancel modules already carry their own bootstrap marker. These legacy controls mount directly under the launcher's native-private profile; they are not falsely described as ten new-style isolated children. The 10 new controls have actual child XML, inspected separately above.

### AC and deliverable mapping

| Requirement | Evidence / limitation |
| --- | --- |
| AC1: unrelated App dispatch before held writer release | Both new and explicit Replace held controls pass; exact different thread IDs, actual App acknowledgment before release/completion, no watchdog, persisted native child timelines. |
| AC2: one correct captured export and source preserved | Ordinary/held new/Replace cases each assert literal full bytes, exactly one atomic publication and one success receipt, same captured Notes operation token and complete seeded source preservation. Additional region/retirement controls preserve the same captured bytes/source. Source service is a stand-in, not real Notes SQLite. |
| AC3: existing outcomes retained | New real collision Cancel/Replace/recheck-error, failed atomic Replace and successful text retry, region/focus and screen-retirement cases all pass; 18 existing helper/native controls pass. The older Escape/Replace control is explicitly locally unqualified after two pre-export startup failures; root's final CI must retain that qualification until actual gate evidence. |
| AC4: mounted private-profile ordinary and held journeys | Four dispatch paths and six additional native outcome cases pass, all 10 parents and actual children0F/0E/0skip; actual picker/Replace/writer, source readiness/recovery witnesses and in-App finally release/drain. |
| Owned product/test/docs | Five final source files and SHA-256 pins/snapshots in `.test-tmp/task3-owned-final/sources.json`; all runtime/test hashes exactly match the successful new batch. Only doc heading/error wording was clarified after its pre-run pin. |
| Scoped static/self-review |3 clean Ruff/format modules; shell's three E721 findings unchanged at 531/944/3755; authorized shell function formatting preserved; other AST owners unchanged; focused native worker/DOM/wait/style collectors clear; final tracked diff whitespace and all five file trailing-whitespace checks pass. |

### Root handoff / remaining work

- Review exactly the five owned source files, including the two narrowly expanded existing-test owners. All unrelated top-level ASTs remain equal to base.
- Root owns independent spec/quality/whole-branch reviews and any resulting requests; no leaf agents were spawned.
- Root must capture the raw diagnostic drift/return code properly, update `Docs/security/production-diagnostic-inventory.json` only after final source/rebase, prove full owner/sink conservation, and obtain a passing inventory check. Leaf makes no inventory-pass claim.
- Root owns complete UI census/gate enrollment and capacity verification. New Windows batch cost was 1066.489 seconds; all new parent phases and true child-inclusive/session costs/hashes are exported, without treating hidden phases as zero. No timing-only rerun occurred.
- Root retains the legacy pre-export startup qualification and inherited warnings/static debt, and owns actual hosted CI, rebase, Git commits, open PR and Backlog/task closure. This report does not mark the Backlog task Done or claim final CI green.

