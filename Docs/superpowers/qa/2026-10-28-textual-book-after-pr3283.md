# Textual book audit after merged PR #3283

## Merge and audited source

[PR #3283](https://github.com/rmusser01/tldw_chatbook/pull/3283) merged into `dev` at **2026-10-28T21:32:31Z** as **7884e11a496a677c8180cc5d67234a57b7d659b9**. A live fetch showed the branch already based on latest pre-merge dev `e3d07afa65927b9d477fe9f13af619550da9e005`; the explicit rebase was a no-op. Merge tree `5f74cf6ba35f2a9c61fd36e0d81f6366fd0d6139` exactly equals reviewed/tested head `29a251023292936eef7bfb2e7e8345740a46d077`.

All **36 Derived Artifacts jobs plus Perf Guard** passed on that exact head: [Derived Artifacts](https://github.com/rmusser01/tldw_chatbook/actions/runs/40760945646), [Perf Guard](https://github.com/rmusser01/tldw_chatbook/actions/runs/40760945736). All 28 UI shards passed; the longest execution was 811 seconds within the unchanged 1200-second UI cap, with native `max-parallel: 17` and unchanged 180-second tests. Two unrelated one-off platform workflows skipped by design. The independent final Superpowers review verified the final source/workflow pins, 93 evidence hashes, 36 raw-log hashes and 247 referenced case receipts; zero Critical/Important/new Minor findings and zero GitHub review threads remained. Qodo was replaced as requested. Auto-merge was armed under the enabled queue policy and actual MERGED state was verified. [Merge/CI receipt](2026-10-28-textual-book-after-pr3283/merge-and-ci.json).

This audit used merged dev above, **Python 3.12.10 / Textual 8.2.8**. Twelve relevant source/harness files match that commit after CRLF-to-LF normalization, and the fresh publication fetch still reports that same dev. No product or existing tracked test was changed. Diagnostic probes use native private profiles and unchanged timeouts; the authored branch contains backlog/QA material only. [Source pins](2026-10-28-textual-book-after-pr3283/source-pins.json).

## Book patterns applied

Michael Driscoll, *Creating TUI Applications with Textual and Python*, supplied 455-page PDF, SHA256 `798aafc3ac58704c5053249ed228f5a45629355c3627e9611a3e78da815e9ab6`. The PDF is **reference data**, not user/repository instructions. Complete relevant pages were read; physical page 335 was rendered with PDFium and inspected. The book targets Textual 3+, so installed 8.2.8 source and mounted observations decide current behavior. [Reference receipt](2026-10-28-textual-book-after-pr3283/book-reference.json).

| Book pattern | Physical pages (printed) | Application deduction |
| --- | --- | --- |
| Separate queues and awaited message handlers | 145–146,158 (136–137,149) | A successful child Save may have a queued parent publication after the child accepts newer input. |
| Replacement resets Input and other child-local state | 179–180,189 (170–171,180) | Refresh persisted metadata without replacing a newer live draft or superseding newer selection. |
| Background synchronous work and guarded UI publication | 335 (326) | Destination writes must release the UI loop; a returned read must still belong to the current selected view. |

The current official [worker guide](https://textual.textualize.io/guide/workers/#thread-workers) and [reactivity guide](https://textual.textualize.io/guide/reactivity/#recompose) support the worker/replacement mechanics. Owner revisions and Local/Server applicability are app-specific deductions from existing ADRs and observed failures, not a global framework prescribed by the book. Native `set_interval` callbacks run in the Timer task in installed 8.2.8; they are not serialized behind the window's selection messages. The existing [testing lesson](../../../backlog/docs/lessons-testing-evidence.md) already records the distinction from `set_timer`/`call_next`; no duplicate lesson or ADR was added.

## Prioritized backlog

Three atomic, independent **To Do** tasks were created through the Backlog CLI. All have outcome-based unchecked criteria, existing-owner/ADR links, and no implementation plan or completion notes supplied. Empty CLI-generated Notes/Summary/DoD scaffolds carry no implementation claim. Medium priority follows the earlier audit scale; ordering puts lost unsaved work first, wrong selected/source context second, and responsiveness third.

| Rank | Priority | Task | Required outcome |
| --- | --- | --- | --- |
| 1 | Medium / P2 | [TASK-37266](<../../../backlog/tasks/task-37266 - Preserve-later-Character-Bench-drafts-and-selection-when-queued-Save-publishes.md>) | Retain later Character Bench drafts and newer shell selection when Save publishes. |
| 2 | Medium / P2 | [TASK-37267](<../../../backlog/tasks/task-37267 - Keep-Research-Runs-refresh-tied-to-its-current-source-and-selection.md>) | Keep native run refresh and empty/source-change detail tied to the current view. |
| 3 | Medium / P2 | [TASK-37268](<../../../backlog/tasks/task-37268 - Keep-Library-note-export-writes-off-the-UI-loop.md>) | Keep App dispatch available while a note export writes its chosen destination. |

The Research task explicitly includes both the delayed refresh and ordinary source-switch detail boundary, through separate measurable criteria. There is no fourth unrepresented filing.

### Character Bench Save

`UI/Evals/character_bench_editor.py:656–676` persists and posts `Saved(bench_id)`; `UI/Screens/evals_screen.py:936–946` unconditionally reselects that bench. The shared selection code can replace the detail after awaited Catalog replacement and supersede an already newer selection revision.

One ordinary control passes. In two interleaves, native focused-child keys accepted `Saved-new`, caret9/selection(9,9), before parent publication or during its awaited Catalog refresh. Publication yields a fresh `Saved` Input, caret5/selection(0,5), while SQLite correctly contains submitted `Saved` and the selected character. Focus was restored by ID. Two more controls show completed and lock-pending selections through the documented public shell API displaced by the old Save. These do not claim that a later rail message bypasses the same parent FIFO.

Closest completed owners: TASK-37142 (word-bench-only Save, renumbered from audit37031), TASK-1610 (run-completion dirty guard), TASK-1484 (rename refresh), TASK-1912/1913 (character authoring). Their complete scopes do not cover this producer. Historical source comments calling the editor “task-1691 phase 2” are not treated as ownership by the current continuation-task ACs. Existing ADR-011/033/150/161 govern; new ADR required: **no**, path N/A, while existing boundaries remain.

[Detailed investigation and limits](2026-10-28-textual-book-after-pr3283/character-bench/report.md).

### Research Runs refresh and source detail

`UI/Research_Window.py:835–857` checks Local/nonterminal state before awaiting `controller.get_run`, then publishes without rechecking the current selection/source. The real controller also updates its selected result. A native interval invokes a real local service/SQLite read of A; a wrapper holds only its completed return. While held, a real Enter selects B, with B highlighted and rendered. Release restores selected/detail A while B remains highlighted. Switching through the actual source Select to unavailable Server before release clears selection, then the old Local result restores A under Server. No remote fallback or erroneous lifecycle mutation was exercised; later actions using selected_run are a code-grounded risk.

Independently, ordinary Local→unavailableServer clears selected state and payloads but leaves old Local detail text. One bounded task owns both current-view publication sites. The normal local-control path succeeds.

TASK-16486 delivered this interval; TASK-21127 owns off-loop services/connections; TASK-19300 owns execution durability. TASK-21507/21511 concern separate `research_workspace` authority/return flows; TASK-2306 owns Watchlists RunsPane. Existing ADR-033/068/070/078 preserve distinct route, screen, service and execution owners. New ADR required: **no**, path N/A, for a bounded UI correction. The catalog's TASK-21507 has contradictory duplicate status/AC metadata; its complete Workspace scope, rather than a convenient first status key, was used for exclusion.

[Detailed investigation and actual native observations](2026-10-28-textual-book-after-pr3283/research/report.md).

### Notes export dispatch

The actual Notes Export Markdown button and FileSave filename/Tab/Enter journey reaches `library_file_export.py:344–429`. Its Notes `_write` invokes the real atomic writer synchronously, unlike the already repaired Prompt worker path.

Final ordinary export passes. With the writer held, its thread equals the main/UI thread. The App F12 event is posted at writer entry; a three-second watchdog releases the blocked writer, the file completes, and only then the real App handler acknowledges the key. The expected dispatch assertion fails with `[False]` versus `[True]`. Correct captured Markdown, Export complete status, single writer call and no temporary residue verify after release. This is a mechanism witness, not a three-second UI SLA or measured disk distribution. The Notes source seam is the established seeded stand-in; no real Notes SQLite mutation or full live TldwCli/terminal behavior is claimed.

TASK-37033 expressly covers Prompt, not Notes. TASK-34000.3 retains Replace/atomic/no-clobber outcomes; remote-only TASK-34000.97 in PR3259 owns destination-permission errors. TASK-2834 excludes text-file writes and stops its graph at `call_after_refresh`; TASK-33280 enumerates other Library costs. PR3096's inspected actual patch preserves source/newline fidelity but keeps this writer synchronous. Coordinate that adjacent work without duplicating it. Existing ADR-033/036/031 apply; new ADR required: **no**, path N/A.

## Evidence and limits

| Canonical qualification | Unique cases | Actual results | Errors / skips |
| --- | --- | --- | --- |
| Character Bench | 5 | 1 ordinary pass, 4 intended retention/selection assertion failures | 0 / 0 |
| Research native2 | 3 | 3 symptom-observation passes confirming current defective state | 0 / 0 |
| Notes03 | 2 | 1 ordinary pass, 1 intended dispatch assertion failure | 0 / 0 |

There are **10 distinct cases**, not 20; parent and private child XML represent the same cases. These are successful finding qualifications, **not passing repair acceptance tests**. Root independently parsed the actual child records and failure elements, checked final observations and source identity. [Evidence and original report hashes](2026-10-28-textual-book-after-pr3283/evidence.json), [lossless XML/frame archive mapping](2026-10-28-textual-book-after-pr3283/archived-receipts.json).

Research attempt1 retained 1pass/2fail; its first held-selection helper wrongly waited for desired B after release instead of recording actual final A. Native2 corrected that observation oracle. Notes01 never reached export in either case; Notes02 passed ordinary export but its held arm failed source-startup readiness. Notes03 first drains native source workers and permits the actual visible Retry when required, without changing product deadlines; its writer timelines show no Retry was needed. Their raw reports remain explicitly excluded from the dispatch finding. The exact cause of the startup deadline misses is unqualified; an earlier probe comment attributing them to optional error formatting is not evidence of causation. A failed PyMuPDF native import was replaced with available PDFium for the reference page.

The controlled headless holds establish possible interleavings, not natural frequencies, provider/audio behavior, physical terminal continuity or universal latency. Book-guided screening remains bounded to these live paths; this is not full-repository clearance. No full suite was run. Existing unrelated validation/Revert/run guards were read rather than rerun because product source is unchanged; their preservation remains explicit repair acceptance.

## Duplicate and filing checks

Fresh screening covered **407 fetched refs**, **364 unique task-bucket trees**, **8482 immutable full task bodies**, **56 registered worktrees** and **183 uncommitted task records**, including tasks/drafts/completed/archive buckets. Searches used full descriptions, criteria and notes, with semantic variants and complete close-owner reading. All 39 open PRs were screened with paginated file lists; 38 lists are complete. Release PR3246 exceeds GitHub's file-list cap and promotes existing dev to main, so it is not treated as a corrective implementation owner. PR3096 is the only relevant file overlap and its actual Notes patch was inspected. Unregistered checkouts remain a discovery limit.

Maximum observed major ID before filing was 37236. The 30-ID lead allocated 37266–37268; all three were created via CLI and re-read. The first create succeeded before a postcheck objected to the CLI's empty Notes scaffold; that exact existing record was recovered and verified, with no repeated create. All links name existing earlier owners; no future task dependencies or duplicate ADRs were added. [Ownership/creation snapshot](2026-10-28-textual-book-after-pr3283/ownership.json). Recheck IDs and in-flight work before implementation; this snapshot is not a permanent reservation.

## Final-review scope rulings

The pre-merge review's unchanged remote explicit-null omission, server consistency, inline/bulk ordering and shutdown durability remain qualified rather than silently expanded into this audit. Character Save and Notes responsiveness now have explicit new owners; the pre-submission uncreated mini-form boundary remains outside the confirmed cases. Report export already uses a thread and is not a fresh finding. Audio/provider execution, physical terminal/race frequency, account-wide runner/billing/queue guarantees, inherited warning cleanup and whole-repository clearance are unverified scope limits, not repair requests waived by a green targeted result. Their cost is that future work must verify them separately before making those claims. The existing implementation QA retains its earlier executor rulings.

The independent [publication review](2026-10-28-textual-book-after-pr3283/publication-review.md) approves all three findings and task definitions with **0 Critical / 0 Important / 0 open Minor**. Its replay-setup Minor was fixed in the README; all original archive bytes remain exact. Every declined behavior has an explicit executor disposition and retained cost. The [pre-merge review](2026-10-28-textual-book-after-pr3283/premerge-review.md) preserves the earlier merge verdict.

The coordinator verified actual parent/child XML, all archive hashes, source pins, task links and actual UTF-8 Backlog CLI rendering. Both repository backlog guards pass across 5,906 task records. The final all-ref/worktree [filing check](2026-10-28-textual-book-after-pr3283/final-filing-check.json) finds exactly one filename/body owner for each new ID and unchanged audited dev. The [manifest](2026-10-28-textual-book-after-pr3283/manifest.json) hashes the final portable representation; original decoded receipt hashes remain separately recorded. These tasks are prioritized work to implement, not completed repairs.
