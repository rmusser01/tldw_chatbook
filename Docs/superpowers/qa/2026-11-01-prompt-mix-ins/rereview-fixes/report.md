# PR3346 second scoped review fixes

Fix base: `1a58f0af2cbc5129f9c8f01595d67c330a7defda`; target dev remains `2acbcd42f3e7b07fd122f5f6dc6b22e7385f10c5`. This report qualifies only the R1/R2 follow-up. The original review and scoped re-review remain preserved. TASK38544 stays In Progress for parent review/publication; AC13/14 and the implementation plan preceded product edits. ADR286 now describes the final acceptance/layout boundary.

## R1 — accepted placement observations

The existing capture-currentness witness now retains its captured preference revision. Current manual preparation, current queue admission and current-choice replay pass that provenance to the existing preference service only after native/ephemeral/queue/receipt acceptance. Received Send uses its original input witness and commits the captured draft before publishing accepted layout. No frozen configuration acquires observation authority.

The service reconciles accepted placement/order only when that revision still matches current preferences; a changed layout advances the intent revision. Name presentation remains Workbench-owned. Last-used revisions remain independent accepted facts, without a revision bump when layout is unchanged. Publication reads no source, does not re-add a removed selection and skips unavailable preferences. Newer user edits win. Original Retry, old externally supplied complete configurations and queued drain cannot roll current layout back; current queue observations publish once at admission. Existing acceptance-phase checks permit the accepted request to finish while old pre-acceptance witnesses and Workbench intents are correctly stale.

Native successive sends now retain the observed sequence: B moves After and is accepted; then A moves After and joins B's group end, yielding `second / body B / body A`. The same assertion passes after real SQLite flush/reopen and fresh native Send. Received manual and queue paths, stale layout intent, refusal/provider/native transaction failure, current Regenerate/Edit/Continue, newer user selection, exact Retry and frozen queue controls cover the owner boundaries.

## R2 — stale artifact completion

ArtifactSelected remains a worker so nested confirmation/Cancel can run. Artifact open retains mode generation, source and detail ownership across awaits. Manual-target capture stays local until that ownership is still current, and both Prompt and selected Recipe paths refuse stale completion before editor/state publication. Own authorized mode transitions happen after this check, so they do not cancel themselves. Source changes, navigation and unmount invalidate an old open without losing a newer dirty mix-in editor or publishing its old manual target.

Actual event-route controls hold the injected async target callback, navigate with the real Mix-ins/Inspect controls, enter new Instructions, then release the callback. Both artifact types preserve that work. Source-change and actual removal controls, normal Prompt/Recipe/legacy open, saved Recipe chooser navigation and real nested confirmation/Cancel controls pass.

## Verification and iterations

Exact pytest argv, exit status, elapsed time and output are in the matching JSON records. All tests used the repository Python3.12 environment, no pytest cache provider, `PYTHONDONTWRITEBYTECODE=1` and distinct task-owned basetemp directories. Counts overlap; they are not one suite total.

- `red`: 2 failed, demonstrating accepted placement loss and stale manual-target publication before product changes.
- `green-initial`: 7 passed, 3 failed. Same-session order, both dirty/source controls and nested confirmations passed. Reopen and two removal assertions required investigation.
- `controls1`: 5 passed, 3 failed. Hook/provider/native commit refusal then acceptance, stale Workbench intent and usage persistence/racing selection passed. Reopen/removal setup issues remained.
- `reopen-diagnostic` and `reopen-error-detail`: each 1 failed. Read-only Python monitoring captured the actual original native error: `Durable Console Library policy no longer matches acceptance.` The reused resend restore helper hydrates messages/preferences but omits asynchronous Library policy hydration required before a new Send. The final test calls production `hydrate_session_library_policy`; no transaction or policy check is bypassed. Textual's `is_mounted` remains set after removal; removal is instead verified through screen-stack absence and the real unmount generation change. Both diagnostic outputs remain preserved.
- `controls2`: **6 passed** — corrected real reopen/Send, both actual removal controls, received manual/queue acceptance and newer-user-intent preservation.
- `replay-controls`: **22 passed** — old supplied manual/queue configurations, original Retry, one-time current queue observation, later selection preservation, existing unavailable-choice/source fences, first native/ephemeral acceptance and current routes.
- `final-controls`: **18 passed, 1 failed** — all mounted and normal artifact/source controls plus Regenerate/Continue passed; the new Edit test misspelled the existing API as `edit_and_resend`.
- `final-edit-control`: **3 passed** — corrected exact Edit node and both received admission controls after restricting observation authority to actual received intents.

The three affected preflight guards pass: Textual worker contract, markup interpolation and persistent diagnostic inventory. No census, inventory or suppression update was needed. SQL/index/recovery gates and their prior approved evidence are unchanged. Per-file Ruff against fix base: **200 baseline / 200 current, zero introduced findings**. AST duplicate-method inventory: **zero new shadows across 9 touched Python files**, excluding property setters/deleters. No newly failing formatter file; inherited formatter debt remains unchanged. `git diff --check` passes.

## Provenance and limits

Current manifests hash this source/governance and new evidence. Historical actual-terminal port capture at5dac and earlier mixed-ancestry captures at247/d774 remain separately qualified; they were not rerun or relabeled as current. The independent depth-one clone at source commit `a443d71c5ceca74feca6d2fc5af1d6e29c682ccc` verified all756 current records against HEAD, with reachable commit count1, no object alternates and all three historical source objects absent. Each historical source check explicitly reported UNAVAILABLE. Exact commands/results are retained in `fresh-clone-proof.json`; this proves published-byte provenance, not additional behavior coverage. New tests are native controller/SQLite and mounted Textual journeys, not a new actual-terminal whole journey or external provider/cache qualification. No full suite, unrelated baseline repair, cloud metrics, push, PR mutation or merge performed by this agent.
