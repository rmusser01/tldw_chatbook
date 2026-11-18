# Whole-branch review: TASK-37266–37268

## Scope and verdict

**Implementation approved; final delivery remains gated on current-head hosted CI and task closure.** No new Critical, Important, or Minor defect was found. Two inherited Minor categories remain explicitly deferred below. This is not a green-PR or merge-readiness certification.

Reviewed base `3efe8476cebf70eceafa605d1abed76a4aad4e0e` through product head `16b74ab4e2c1c3a2983e2f70d7609f1f295f11e4`, plus the supplied evidence/metadata supplement at `7c2b8f5aab066035ed9d06f9719a93d184f8e5c3`: completed diagnostic receipt, refreshed integration archive/map, updated QA qualification, and first two Task4 checklist items. Root reports PR #3290 open against dev with hosted checks queued; I did not independently inspect GitHub and claim no hosted success.

The full review package's SHA-256 matches `final-review-index.json`. Its 128 indexed paths comprise 29 text deltas, 81 conserved earlier audit artifacts, and 18 implementation evidence/review artifacts. I reviewed all changed product/test/CI/documentation scope using the complete package's indexed reading aid and structured evidence inspection. The 84-file carried comparison consists of 81 unchanged audit artifacts and the three intentionally updated task files. Conserved audit artifacts were treated as historical evidence, not re-audited as new product changes.

Read the complete three task specifications, implementation plan, QA report and ledger rulings, the requesting-code-review template, task review reports, and relevant named owner context. Review actions were read-only apart from this requested report. No tests, application runs, Git/index mutations, or subagents were dispatched.

## Strengths

- **Character Save protects actual editing context.** `tldw_chatbook/UI/Evals/character_bench_editor.py:287`, `:419`, and `:700` combine raw values/card selection with an editor-local synchronous monotonic revision. Successful value-equal Saves, text roundtrips, selection/caret movement and focus changes revoke older applicability. `tldw_chatbook/UI/Screens/evals_screen.py:540` retains the existing selection revision for Save refresh; `:600` checks current selection, originating editor and current form after the awaited rail replacement. Persistence and Catalog/Inspector refresh remain separate from later unsaved draft retention. The repair addresses the initial review's ABA finding without a new global owner.
- **Research preserves the current view's authority.** `tldw_chatbook/UI/Research_Window.py:450`, `:775`, and `:853` clear detail immediately on source departure and reject timer returns after source/selection revision changes. Accepted refreshes preserve payload objects; terminal/Server skip behavior remains intact. Direct selected-run writers and action-ID consumers were inspected: native transitions use the setter, while actions use window selection. Preserving the controller's fetched-result cache is justified by those consumers.
- **Notes uses the existing writer and operation contract.** `tldw_chatbook/UI/Library_Modules/library_file_export.py:377` captures rendered content and the resolved target; `:400` offloads only the writer. Late no-clobber collision rechecks remain inside the recoverable error boundary. `:434` admits a grouped non-exclusive native worker, while the unchanged operation owner at `library_notes_controller.py:1649` refuses duplicate admission and `:1689` prevents completion status from moving into a different active region. The validated screen caller and shared confirmation seam retain their existing responsibility. Native retirement cancels UI publication while an already-running thread may finish.
- **Regression assertions exercise the repaired boundaries.** The new tests preserve real SQLite/native child input and region awaits for Character; real Local service/SQLite returns and native timers/Select/Enter for Research; and real FileSave/Replace/atomic-writer dispatch for Notes. Expectations include exact widget identity/context, action target, independent output bytes, precious-file preservation, source preservation, failure/retry, and drained holds. The two existing Notes tests retain their assertions and adapt only synchronous-completion assumptions.
- **CI enrollment is complete and bounded.** `scripts/ui_pr_gate_census.txt:1` retains all 1,140 inherited selections exactly once and adds 43 concrete behavior cases plus the unchanged canonical-class guard. All 1,184 selections are unique; the three new files have split-file contracts in `Tests/CI/test_ui_pr_gate_census_checker.py:501`. `.github/workflows/derived-artifacts.yml:638` increases only matrix membership to 32 while retaining max-parallel 17, serial execution, minimal dependencies, 20-minute jobs and 180-second tests. Required aggregation still depends on the matrix and fails on non-success (`:937`, `:985`). ADR-103 records the measured policy amendment and its qualifications.
- **Claims remain appropriately bounded.** Task records remain In Progress with unchecked ACs pending final closure. Architecture descriptions match the final behavior. The QA distinguishes parent/child evidence, invalid runs, local platform limits, service stand-ins, inherited failures and forecasts. TASK-37330 separately records the pre-existing Research artifact caller mismatch without claiming that the timer fixture proves artifact-action success.

## Findings

### Critical — 0

None found in the reviewed change.

### Important — 0

None found in the reviewed change. Outstanding hosted/closure gates below are mandatory unfinished verification, not waived defects or fabricated implementation findings.

### Minor — 0 new; 2 inherited categories retained

1. **Warning noise remains disclosed and deferred.** `Docs/superpowers/qa/2026-10-28-textual-followups-implementation/task1-local-receipts.tar.gz` (actual child XML, line 1), corresponding task2/task3 archives, and `progress.md:85`, `:165`, `:213` retain pytest-asyncio configuration, Pydantic, audioop, optional-backend and inherited splash warning output. These reduce diagnostic clarity but do not invalidate the targeted passing behavior assertions. Disposition: retain the qualification; repair in configuration/dependency/splash ownership when undertaken. Do not suppress warnings or broaden this bug-fix branch to clear them.

2. **Existing lint/format debt remains bounded.** `Tests/UI/test_library_shell.py:531`, `:944`, `:3755` retain the three E721 diagnostics; its unchanged bulk formatting and inherited formatting regions in `Tests/CI/test_derived_artifacts_workflow.py` and `Tests/CI/test_ci_queue_pressure_contract.py` are not whole-file clean. Scoped new files/changed function and CI lint have passing receipts. The AST conservation proof shows only the authorized shell function changed. Disposition: defer unrelated cleanup, preserve exact qualifications, and never present scoped checks as whole-file clearance.

## Evidence and source confidence

- Independently verified the full package hash, each of the four portable archive hashes, every archive member's byte count/hash, and refreshed integration members after the supplement. Member counts are 219 Character, 87 Research, 99 Notes and 35 integration. No extracted receipt was executed.
- Parsed actual XML rather than relying on parent totals. Character's final repair GREEN has **31 parent passes and 26 actual child passes**; its repair RED has six actual child failures at editor/context retention, with no error/skip substituted for the defect. Research's intended RED has three actual failures (old A replaces B, old running A replaces newer paused A, stale detail after source clearing); GREEN has **seven parent/seven child passes**. Notes' intended dispatch RED fails after the App key arrives only after watchdog release; GREEN has **ten parent/ten child passes**.
- Inspected actual Notes GREEN timeline JSON. Both held new-file and Replace cases show distinct writer/main thread IDs, App acknowledgment with `before_release: true`, acknowledgment before completion, and no watchdog release. This supports responsiveness and ordering, not a latency SLA.
- Final runtime product pins match current sources after accounting for checkout newline normalization. Research and Notes test/doc pins also match. Character's final executed regression source differs only by the explicit whitespace-free description parameter ID; its parameter values and body remain unchanged. The collection map records the final IDs and historical aliases. Existing passing task reviews were corroborating evidence, not substitutes for inspecting product/tests/raw outcomes.
- Read the Textual 8.2.8 watcher implementation excerpts and verified their installed source hashes: synchronous callbacks execute inline; `Screen.focused` changes synchronously; closing watcher owners are pruned before callback invocation. This supports the Character fence and retirement behavior at the tested framework version.
- Integration XML records five intended census RED failures; then 228 passes/34 failures; then 32 passes/two failures; then the frozen exact two-case retry passes. The latest-outcome map contains 262 distinct passing observations, not one all-green batch. Nine separate closure controls pass. The mixed-checkpoint run remains invalid evidence and is not erased.
- Independently recomputed selector conservation and the capacity sum/round-robin maximum from the stored complete cost map and final order: **19,142.875769 seconds total**, **599.342269 seconds maximum at 32 shards**, **999.243609-second forecast**. Candidate 31's ideal lower bound is 1,020.138940 seconds, already above the 1,020-second target. Final census ordering equals the replay order, and source hashes match after LF normalization. The inherited input identifies exactly 28 UI logs and excludes unrelated lanes. These are modeled costs; hosted capacity remains unproven here.
- Independently compared the retained timer data: baseline/current each have 120 semantically identical sites and 75 identical roots. Worker group/mutation findings are unchanged; named external guard inputs match current source after newline normalization. The comparison retains the ProjectIndexBindingCard `_poll` inventory gap after path normalization. This supports no new Research obligation; it does not turn the 46-pass/eight-failure static batch green.
- The supplemented `diagnostic-inventory.json` retains actual command arguments, stdout/stderr and exits **1/0/0/0** for before/statements/write/postcheck. The statement check reports four calls retained with one moved/reindented warning and no added/removed statement. Final postcheck verifies 701 owners and 16 sink files. The changed code retains sanitized arguments and severity; the inventory delta is the intended exporter digest only. The completed raw postcheck closes the earlier task-review diagnostic gate.
- The artifact ownership receipt records 412 refs, 366 task trees and 55 registered worktrees plus controller variants and source hashes. Its bounded discovery qualification and linked earlier owners support separate TASK-37330 tracking, not universal knowledge of every checkout or future change.

## Deferred rulings and remaining gates

| Item | Disposition |
| --- | --- |
| Targeted local verification and native-private profiles | Accepted. AGENTS.md forbids an unrequested full sweep; privacy guards/timeouts were not relaxed. No full-suite conclusion is inferred. |
| Existing ADRs for three bounded repairs | Accepted. Owners, schema, dependencies, routes and runtime boundaries remain unchanged. Shared capacity is covered by amended ADR-103. |
| Research controller fetched-result cache | Accepted. Current window/action consumers do not derive selection from that cache; no mirror-consistency guarantee is claimed. |
| Research artifact action mismatch | Deferred to the concrete medium TASK-37330. The changed timer path neither creates nor repairs it; real scope payload preparation only qualifies retention. |
| Research static inventory failures | Deferred as conserved baseline limitations for this scoped change, including non-Windows inventory gaps. Any actual required hosted failure still requires repair. |
| Warnings and three E721/format debt | Deferred as the two inherited Minor categories above. No new cleanup obligation or suppression. |
| Diagnostic inventory | Closed by the supplied successful raw final postcheck and preserved four-call/sink ownership. |
| Old Notes Escape/Replace case | **Open hosted gate.** Both local failures stop at source readiness before export. The actual XML says the Notes row never mounted and source loading timed out. It is neither passing nor baseline-proven. Root must inspect the exact enrolled case's successful hosted result. |
| Canonical-class guard | **Open hosted gate.** Conserved inputs support Windows path attribution. Newly enrolling the unchanged guard is appropriate; root must inspect its actual Linux pass. |
| 32-shard capacity and required aggregation | Source/model review approved; **actual final-head hosted application jobs and required aggregate must pass** under unchanged limits. Forecast is not hosted proof. |
| Final reviewed open PR | Root reports PR #3290 created. **Current-head CI, feedback resolution and final task CLI/AC/plan hygiene remain open.** Keep the PR open; no merge authorization was supplied. |

## Declined to judge

- A live remote Research server and remote mutation/rollback behavior: no such execution was supplied; Server controls use explicit async doubles and this branch does not add remote guarantees.
- Successful Research artifact-button loading: the known unchanged caller contract defect is separately tracked; prepared real payload data does not qualify the action.
- Real Notes SQLite source mutation: the source is the established seeded service stand-in. The export writer receives captured text; this evidence proves stand-in source preservation and actual destination bytes only.
- Arbitrary concurrent programmatic source restoration/catalog refreshes outside the native source/selection journeys: unchanged owners and task scope concern the timer's publication after supported view transitions, not a general Research concurrency rewrite.
- Rollback or cancellation of an already-started destination write after screen retirement: neither implemented nor promised. The tested contract is completed accepted I/O with retired UI publication suppressed.
- Natural race frequency, user-visible storage-latency distribution, full terminal interaction and an SLA: controlled native holds establish possibility and repaired ordering, not those measurements.
- Whole-repository warning/static debt and unrelated formatting repair: conserved inputs and scoped evidence do not justify expanding these three bounded fixes; required hosted failures remain actionable.
- Re-auditing the conserved 81-file earlier audit packet as new behavior: its indexed blob conservation makes it historical background; new implementation claims and receipt artifacts were reviewed separately.
- Current-head hosted passes, platform-wide capability, queue latency and final green-open-PR completion: unavailable to this read-only review at report time and explicitly left as root-owned gates rather than presumed success.

## Assessment

**Ready to continue final hosted qualification: Yes. Ready to declare the requested work complete or ready to merge: No, pending the explicit gates above.**

The implementation is small relative to the ownership and timing problems it repairs, and independent inspection of product code, native assertion evidence, selector conservation and capacity data supports the stated behavior. No source fix is requested by this review. Finish current-head CI, inspect the two specifically unqualified cases, address any real failures/feedback, then complete the backlog and plan records without merging the PR.
