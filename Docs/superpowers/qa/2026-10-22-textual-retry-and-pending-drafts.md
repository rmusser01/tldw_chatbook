# Textual retry and pending-draft fixes

Implements TASK-36360 through TASK-36363 from the [post-3207 audit](2026-10-22-textual-book-after-pr3207.md). The book is reference material; the four Backlog acceptance contracts govern implementation. This is a bounded correction of existing UI owners, without schema, provider, dependency, privacy/admission or service policy changes.

Plan: [2026-10-22-textual-retry-and-pending-drafts](../plans/2026-10-22-textual-retry-and-pending-drafts.md).

## Implemented behavior and evidence

| Task | Behavior | Targeted evidence |
| --- | --- | --- |
| 36360 | A final quiz submission combines earlier completed answers with the current answer without mutating pending answers before success. A definite failed submission can be retried or edited without duplicate grading. | 5 mounted Start/Submit/Load cases; real QuizScopeService, LocalQuizService and SQLite. RED: 3 intended failures, 2 ordinary controls pass. GREEN: 36 new/adjacent service and draft tests pass. |
| 36361 | Appearance, Storage and Console Behavior advance the saved baseline only for the submitted snapshot and retain newer raw controls, including queued changes, without replacing editors or focus. | 19 mounted private-TOML cases pass; 4 final real-writer/format checks pass; 29 adjacent checks pass. Failure controls inject a definite OSError into the actual raw writer and verify unchanged file bytes before retry. |
| 36362 | Successful Save closes only an unchanged form; newer raw values remain dirty. A retained create adopts the durable row ID, locks owner identity and updates that row on the next Save. Later confirmed discard returns the earlier saved/queued result to refresh the workbench. | All 20 new outcomes covered: 18 initial GREEN passes plus 2 corrected quit-dialog readiness checks. All 30 adjacent form/quit checks pass. Real SchedulingService/SQLite, including failed writes, queued outbox, held/failed postcommit reads and same-row retry. |
| 36363 | A delayed file read publishes only if its current editor, live text and working-source snapshots still match and the owner remains attached. Later edits/replace/clear remain authoritative and discarded content has no success notice. | Qualified RED: 3 intended failures/1 unchanged control pass. Final GREEN:16pass, including all4new held-read cases,7existing full-source/import cases and5chapter-freshness cases. Actual Import/picker/read, retained source/Generate and original file-byte checks; no synthesis/network. |

The Settings tests cover same/different-field edits, queued input, returning to the old baseline, unchanged and normalized submissions, genuine writer failure, next Save and confirmed Revert. Storage remains restart-only and does not reconnect live handles. The narrow completion helper is lazy and same-owner; the Settings size ratchet is lowered from 35,668 to its measured 35,623 lines.

## Evidence qualifications

Only targeted local tests were run. No full local suite or end-to-end server/provider/synthesis claim is made. Each run uses a unique native private test parent with the project Python 3.12 runtime; existing profile/privacy admission is retained. Raw commands, logs, XML, selected-baseline proofs and source hashes are retained locally under `.test-tmp/task-36360-36363/`.

The broader legacy quiz screen run produced 37 passes and 15 startup failures before quiz construction (`load_settings` → raw participant → `RecoveryRequired(raw_source_selection_changed)`). An exact-base subprocess selection and current representative both reproduce that refusal. The Settings adjacent run produced 29 passes and one equivalent pre-Settings startup refusal; its exact-base subprocess selection also reproduces it. Those qualifications establish startup failure provenance, not passing legacy screen journeys. The legacy quiz-screen file and the refused Settings hub spy are absent from the required UI census.

Initial queued Settings test setup used an Input insertion API that did not replace a selection. Those three setup failures are excluded; actual Input.replace then produced the three intended RED assertions against unchanged production source. One incomplete final-fault preparation run and a baseline wrapper encoding failure are also excluded. Final real-writer evidence is `settings-final-fault-04`, and all nineteen GREEN cases are retained in `settings-green-all-02` with an AST/source proof for the final formatting/failure-control refinement.

Recurring Question RED retained qualified raw-input/write witnesses; initial keyboard/focus/checkbox setup mistakes are excluded. The first GREEN had two dialog-readiness failures; waiting for the actual mounted cancel control fixes only that test, and those two cases pass separately. All other functions and production source remained unchanged. Explicit whitespace-free parameter IDs preserve parameter data and the normalized module AST; twenty collected selectors map to their passing receipts.

Audiobook clear uses the public production TextArea delete-left action after actual Pilot selection; edit/replacement use actual Pilot printable input while the read is held. This is bound-action dispatch, not raw Pilot Backspace, because that binding would bubble through the held parent. Thirteen optional legacy speech setup errors reproduce in a fresh parent before owner actions; an exact-base/current representative pair reproduces the same raw-source refusal. Those two optional files are absent from the census. Existing five E721 owner diagnostics and one formatter hunk remain identical to baseline; test formatting is clean.

Inherited Pydantic and pytest-asyncio deprecations remain visible. A broader legacy quiz run also emitted audioop/executor warnings that were not individually baseline-qualified. No blanket warning filter or admission relaxation was introduced.

The shared worker guard found two post-await title/help lookups in the Recurring Question finisher. Two strengthened existing metadata-read cases remove that real presentation chrome during a held SQLite write: RED2 raises WorkerFailed/NoMatches; the guarded fix passes four focused metadata/admission/rollback cases. The full worker-contract check passes with2,782 unchanged input hashes. The new metadata-refresh diagnostic was reviewed statement-by-statement: it interpolates only routing owner and durable local row ID, introduces no sink, and its inventory pin changes only that owner from2calls to3.

## Decisions and scope

No new ADR: existing [ADR-011](../../../backlog/decisions/011-chatbook-workbench-ui-system.md) and [ADR-033 application session state](../../../backlog/decisions/033-application-session-state-ownership.md) govern the bounded owner state. Settings also retains [ADR-033 commit models](../../../backlog/decisions/033-settings-commit-models-three-honestly-labeled.md), [ADR-004 restart boundary](../../../backlog/decisions/004-settings-storage-defaults-restart-boundary.md) and [ADR-161 component patterns](../../../backlog/decisions/161-component-pattern-library.md). Scheduling retains [ADR-018](../../../backlog/decisions/018-local-server-hybrid-scheduled-tasks.md); proposed editor ADRs are context only. Existing task/ADR links remain in each original Backlog contract.

This does not add uncertain-commit/server-wide idempotency, global draft/lifetime state, other audiobook import routes, or new persistence/outbox semantics. Held completions exercise concurrency deterministically; no frequency or natural disk-latency estimate follows from them.

## Integration and review

All four independent task reviews are Approved. The scoped Recurring Question guard re-review finds the worker issue addressed with no new Critical/Important breakage; the controller separately reviewed the exact new diagnostic, verified the two-line owner pin change and ran the final inventory check successfully. Targeted integration reports106passes; source-wide census, worker, profile-owned-path, markup and Backlog identity/parse guards pass. Final diagnostic-pin verification passes (691owners,7732TASK-494calls,16sinkfiles). Whole-branch review of the rebased tree and hosted current-head CI are delivery gates; their latest results are recorded in the PR. Census checker formatting retains seven identical baseline hunks, while its Ruff check and changed floor/comments introduce no new diagnostics. All48new concrete nodes are enrolled, raising the census floor603→651. The original603selectors retain their positions; only the new48-node tail is balanced using previous green run39840153541 and worst retained native node timings. Estimated maximum pytest work is17.4minutes across13shards; this is a planning estimate, not hosted job wall time or a performance benchmark. Shard count, timeouts, caps and admission remain unchanged. Superpowers requesting-code-review replaces Qodo as requested.

## Rebase after PR3208

Rebased onto dev31bb928ab8 after the provider-copy, shared UI-test-helper and idle-heap changes. The only final-commit conflict is the UI census append/floor: retain all604 dev selectors in their exact positions and all48 new concrete nodes, for652 total. Rotating only this PR's48-node tail by one keeps47 nodes in their previously verified shards; the held-read automation case moves from shard6 to shard2 (prior hosted call2.36s; prior complete jobs900s/908s respectively). The incoming idle-heap file keeps its dev slot. Thirteen shards and all budgets/caps remain unchanged. The earlier603→651 counts and source receipts describe the initial implementation snapshot, not this combined census; current combined validation belongs to the new head's checks and PR body. No production fix owner changes were needed.

Fresh combined census/replay/shard-count and both Settings size-ratchet tests:96passed, zero failures/errors/skips (14.685s). The combined diagnostic inventory verifies692owners,7734TASK-494calls and16sinkfiles. These targeted local checks do not substitute for the rebased head's required hosted CI.

## Rebase after PR3218

Rebased onto dev23bd46cbcd after the Settings model-id validation and Tools & MCP labels landed. Production code merges without conflict. The sole conflict is the Settings size ratchet: retain the incoming -5-line reduction and this PR's -45-line delegation, pinning the combined module at its actual35,618 lines with no slack. Earlier size counts are historical receipts, not this combined module's count. The four regression bodies and acknowledgement helper remain unchanged; the UI census remains652 across13shards with all budgets unchanged.

Fresh targeted incoming model-id/Overview/Tools badge integration plus both Settings size guards:16passed, zero failures/errors/skips (239.653s). The native command and XML are retained under `.test-tmp/task-36360-36363/rebase-3219-after3218-integration-01/`. Independent Superpowers review and the new pushed-head hosted checks remain delivery gates recorded in the PR. No new ADR or behavior change is introduced by this conflict resolution.

## Rebase after PR3210

Dev28d7ab2c9a adds the Environment controller file after the604 inherited selectors. The combined census preserves all605 dev entries in their exact order and all48 new nodes, for653 total. Rotating only the new tail retains47 prior shards; the different-storage case moves from shard7 to3. Its prior hosted call was45.36s; prior complete jobs for shards7/3 were1027s/1052s. Adding that call to the old shard3 duration gives1097.36s, a planning estimate rather than a measured new-head result. Thirteen shards and all caps remain unchanged. The checker now has one653floor assignment; earlier652counts are historical.

Both Textual verification lessons retain their complete text in `backlog/docs/lessons-textual-ui-verification.md`, linked from the audit. The shared testing-lessons document matches dev. No production fix or regression-body change is introduced. Current combined native roots/controller/census checks, independent review and new-head hosted CI are delivery gates reported in the PR.


## Integration after PRs #3215, #3216, #3187 and #3217

Rebased onto dev `52b3207ff1cfecdd057272de0312b9e10fde363f`. Only census/floor conflicts required resolution: retain the exact616-selector dev prefix plus all48 new cases (664 total), with a single floor of664. Rotate only the PR tail two places right;46cases retain their previous shard, held-read moves2→6 and different-storage moves3→7. All13shards and caps are unchanged. Parent job times plus previous-head call durations estimate a maximum1,108.86s; this is a projection, not a new-head run.

Previous head `91a9d20b99` passed Derived Artifacts39915260015 and Perf Guard39915260016; all48 new cases were verified exactly once across13 successful UI jobs. Its qualified local integration result remains286passed/one inherited Windows path-separator assertion failure. Current native census/CSS/size guards, inventory, independent review, feedback and hosted CI are delivery gates reported in the PR. Production fixes and regression bodies are unchanged by this conflict resolution.
