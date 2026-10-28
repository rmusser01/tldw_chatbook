### Spec Compliance

- ❌ Issues found: AC3's safe write-failure feedback/retry boundary is incomplete for a late no-clobber collision. The new retry-confirmation call can raise an I/O exception out of its native worker instead of reporting the export failure (`tldw_chatbook/UI/Library_Modules/library_file_export.py:478`; Important I1 below).
- ✅ AC1/AC2/AC4 are implemented: a synchronous native-worker handoff owns a captured Markdown string and destination; the actual mounted private-profile tests cover ordinary/held New/Replace writes, real App key dispatch, canonical content, unchanged SQLite rows and worker drain (`library_file_export.py:457`, `:469`, `:500`; `Tests/UI/test_library_prompt_export_responsiveness.py:40`, `:117`, `:139`, `:153`).
- ✅ Other inspected AC3 outcomes retain their contracts: explicit Replace and late-race Cancel/Replace (`test_library_prompt_export_responsiveness.py:53`, `:113`, `:171`), safe render failure/retry (`Tests/Library/test_library_file_export.py:1122`), unchanged write-failure/retry (`Tests/UI/test_library_prompt_export_journeys.py:244`) and scoped navigation receipts (`test_library_prompt_export_journeys.py:292`). All brief-listed files have corresponding hunks. The extra existing helper/journey adaptations match the bounded approvals at `.superpowers/sdd/2026-10-27-textual-37031-37034/progress.md:60`, `:66`, `:68`.
- ⚠️ Cannot verify from this task package: root's final combined-byte six-case run and hosted current-head CI. Earlier race/fidelity/retry receipts precede the narrowly qualified render-catch change; the supplied AST qualification and current hashes support reuse, but do not constitute a new combined runtime result. Root must retain its separate final gates. No full-suite or pristine-output claim is made.

### Strengths

- `library_file_export.py:469` sends only the filesystem writer through `asyncio.to_thread`; the modal callback returns synchronously through `_start_write`, preventing Textual from awaiting the write on its message pump. Receipt and confirmation work resumes on the UI loop.
- `library_file_export.py:457` captures an immutable rendered string before dispatch, while `:503` uses a non-exclusive named worker group. Independent calls retain separate closures and do not cancel each other's writes. The unchanged caller already captures the live editor fields and Prompt ID (`library_prompts_controller.py:3586`, `:3615`; `library_screen.py:28486`).
- `test_library_prompt_export_responsiveness.py:76` observes the actual App event handler, and the held writer only proceeds after that handler acknowledges the key or its unchanged watchdog expires. The final held receipts show acknowledgement before release and physical completion, with no watchdog events. This is a causal ordering test, not merely a thread-ID or heartbeat assertion.
- `Tests/Library/test_library_file_export.py:1122` adds a real malformed JSON serialization failure and successful retry. The existing helper test adaptations keep their substantive assertions; the recorded AST assertion qualification supports that reading.
- `Docs/Architecture/library-shell.md:38` documents the ownership change without adding dependencies, changing shared export helpers, altering visual tokens, or introducing an export framework. The task record retains In Progress status for controller gates.

### Issues

#### Critical (Must Fix)

- None.

#### Important (Should Fix)

- **I1 — Catch failure of the late-collision confirmation recheck.** `tldw_chatbook/UI/Library_Modules/library_file_export.py:478`: `_write` catches a `FileExistsError` from the no-clobber writer, then calls `request_export_write` inside that `except` clause. That helper performs another filesystem existence check (`:316`, implemented by `destination.is_symlink() or destination.exists()` at `:178`). If the parent becomes inaccessible, or that metadata probe otherwise raises `OSError`, the exception is outside the surrounding try's protection and escapes `_write`. `screen.run_worker` leaves `exit_on_error=True`, so Textual handles the escaped exception as a fatal worker failure rather than the established sanitized export error. This is a new second probe on the collision path: the original `request_export_write` caught the writer's `FileExistsError` and called `_ask` directly (`:321`). Include the recheck in the recoverable export error boundary, report the exception category through the captured notifier, and add a focused late-collision/recheck-failure/retry control. Merely suppressing worker exceptions would lose the required feedback. Evidence is the inspected production control flow and installed Textual `dom.py:502` / `worker.py:382`; no existing retained test exercises this particular recheck failure.

#### Minor (Nice to Have)

- **M1 — Retained warning noise is inherited debt.** `.test-tmp/textual-37031-37034/task3-green05/run.log:3`, `:8`, `:22`: the final passing run still emits Pydantic `Field(exclude_if)`, `audioop`, and unset pytest-asyncio loop-scope deprecations. These are disclosed and not introduced by this diff; record/resolve them with their dependency/test-configuration owners rather than describing this output as pristine or broadening this task's product patch. This is not a task approval blocker.

### Checks and Evidence

- Reviewed immutable package `review-8a91bd5d85..6d6ba5b29f.diff`, base `8a91bd5d85cdc7ab50190592d4a4439201d20759`, head `6d6ba5b29f52a449c37abaedc114bcae081842c0`. The initial tool response truncated the middle; recovered only the omitted diff section. No Git commands, source edits, test reruns or subagents. Only this requested review artifact was written.
- Independently parsed retained XML, including actual private-child XML rather than trusting wrappers: `task3-red02` has ordinary child pass and held child fail at dispatch ordering, zero child errors/skips; `task3-green03` has six passing children, while its parent has the reported four failures; `task3-green04` has 102 passes/3 failures and zero errors/skips; `task3-green05` has four passing parents and four passing actual children, zero failures/errors/skips.
- Independently inspected the final `dispatch-timeline.json` files. New-held orders key acknowledgement at 522645.031 before completion at 522645.062; Replace-held acknowledges at 522763.015 before completion at 522763.031. Each records one acknowledgement with `before_release=true`, no watchdog release, and one writer invocation. Equal-resolution release timestamps do not undermine the explicit logical order.
- Independently parsed `task3-baseline01`, `task3-baseline02`, `task3-baseline03`: the twelve Windows helper failures, two receipt-format failures, inventory failure and preimport setup failure are present in original-source receipts, with zero errors/skips. Baseline03 also preserves the intended held-write dispatch failure. These are qualified baseline limitations, not passing tests. The current diff neither weakens these assertions nor changes allowances.
- Read source-restoration receipt `task3-source-qualification-03.json`, render-catch AST receipt `task3-capture-guard-qualification.json`, and assertion preservation receipt `task3-test-assertion-qualification.json`. Recomputed all four final production/test SHA256 values; each matches `task3-final-source-hashes.json` and the implementer report.
- **Focused outside-diff check — collision semantics and fatal worker errors:** read only `resolve_export_destination`, `export_destination_exists`, `request_export_write`, and installed Textual worker default/error handling. This establishes I1; shared helper source itself is unchanged.
- **Focused outside-diff check — capture and scoped publication:** read `_export_library_prompt`, `_write_library_prompt_export_file`, and `_report_library_prompt_action_result` at `library_prompts_controller.py:3567`, `library_screen.py:28447`, and `library_prompts_controller.py:2839`. Payload/identity capture precedes dispatch; current editor/screen identity gates inline status, with otherwise global notification. No broader controller crawl.
- **Focused outside-diff check — real service and failure coverage:** read `_prompt_host` at `Tests/UI/test_library_export_replace_confirm.py:111` and the unchanged write-failure/retry body at `Tests/UI/test_library_prompt_export_journeys.py:244`. The host wires the real Prompt scope service/DB; retry tests fail the actual destination seam and then exercise the real picker again.
- No dedicated concurrent-export runtime receipt is supplied; independent immutable closures and non-exclusive workers are verified statically. The held-test success path waits for native workers before closing SQLite, then retires the gate in `finally` (`test_library_prompt_export_responsiveness.py:139`, `:149`).

### Assessment

**Task quality: Needs fixes.**

**Counts: Critical 0 / Important 1 / Minor 1 (inherited warning debt).**

**Reasoning:** The scheduling repair and causal responsiveness evidence are sound, and the patch preserves the existing export ownership with a small product change. The new late-collision metadata recheck needs the same safe failure boundary as the write before AC3 can be approved; final combined-source/hosted gates remain separately owned by root.
