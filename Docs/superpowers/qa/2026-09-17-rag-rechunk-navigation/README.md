# Re-chunk navigation continuity — TASK-32719

2026-09-17 UTC, `feat/component-pattern-library`, based on `4cc9819805`.

Replacing the initiating Search/RAG panel lost running feedback and completion
notices. Re-chunk now has one app-session UI owner and an app-owned thread worker;
panels read its current state and subscribe only while mounted. The latest receipt
survives navigation, and returning before completion stays disabled with progress.
Starting another run replaces the receipt. Failure clears progress and releases
admission for retry, including failure to schedule the worker.

ADR required: yes.
[ADR-164](../../../../backlog/decisions/164-rechunk-run-lifetime.md) records the
operation lifetime and alternatives. The existing local scope service, runtime
policy, shared Re-chunk/backfill slot and worker group remain unchanged. Completion
publishes state and releases admission together on the UI thread. No durable job
history, restart resumption, new dependency, visual token or stylesheet is added.

## Automated evidence

- [Initial red run](navigation-red.txt): **4 failed in 38.39s**. Canvas and
  whole-screen replacement both return an enabled action during active work;
  finishing while away fails to surface the expected completion notice.
- [Targeted run](targeted-tests.txt): **96 passed in 110.27s**, without exclusions
  or a full suite. Twenty-one new cases cover sixteen theme/size/route/return-time
  combinations, four policy/backend failures while away followed by retry, and
  worker-scheduling failure followed by retry. In-flight returns also check
  duplicate Re-chunk and backfill refusal without another service call.
- Existing cases cover the previous feedback/focus fixes, real SQLite re-chunk
  behavior, census responsiveness/unmount, conditional forced reindex/cache
  behavior, policy admission and actual Settings backfill trigger refusal.
- [Static checks](static-checks.json): new owner/test/runner lint and formatting
  pass, as does the new panel block's formatting. The
  [panel lint comparison](lint-comparison.json) drops from five existing
  diagnostics to four, with no additions.
- [Independent review](review.json): scheduling-failure coverage was requested
  and confirmed present; no unresolved code/test/runner finding remains.

## Native evidence

The [runner](native_check.py) boots real TldwCli in a private native tmux terminal
and validates the private config/data paths before app imports. It seeds one
empty skipped source and one new valid source per cell, with actual legacy chunk
rows and workspace memberships. A timing gate delays the real scope method, then
delegates to the actual local re-chunk service. Semantic indexing is explicitly
disabled. No provider or retrieval calls are made.

All four final theme/size cells [pass](result.json), totaling four actual
re-chunk runs. Each starts with keyboard Tab/Enter, visits Notes and returns while
the worker is held, then leaves for Console through Ctrl+2. The worker finishes
there; Ctrl+3 and the Search/RAG rail row return to a complete receipt and a
refreshed legacy count. Keyboard scrolling reveals the full wrapped receipt.
Each run migrates one valid item and skips the empty item, taking the census from
two to one. Four items receive the current engine stamp and replacement chunks.
The skipped chunks and all five source records are unchanged
([before](source-before.json), [after](source-after.json)).

The [initial native harness attempt](initial-harness-failure.json) passed the wide
cell, then tried to focus a Search/RAG row hidden by Notes' compact adaptive
navigation pane. That wait also expired the artificial worker gate. The runner
was corrected to activate the existing visible Nav grip first; no production
change was made for this harness issue. The fresh final run passed all four cells.

All eight final captures were rendered and inspected in one batch. They show
readable running feedback after the Notes return and complete counts/notes after
completion in Console, including the compact wrapped re-index disclosure.
Captures wait for natural notification expiry. [Hashes](capture-hashes.json)
record raw and stored bytes; only trailing SVG source-line whitespace is normalized.

| Theme / size | Returned while running | Returned after completion |
| --- | --- | --- |
| Dark / 170×48 | [Capture](textual-dark-170-running.svg) | [Capture](textual-dark-170-completed.svg) |
| Dark / 80×24 | [Capture](textual-dark-80-running.svg) | [Capture](textual-dark-80-completed.svg) |
| Light / 170×48 | [Capture](textual-light-170-running.svg) | [Capture](textual-light-170-completed.svg) |
| Light / 80×24 | [Capture](textual-light-80-running.svg) | [Capture](textual-light-80-completed.svg) |

[Lifecycle checks](lifecycle.json) verify app-run return, exit 0 and PID absence
before closing the owned terminal. All ten private databases pass read-only
`quick_check`; no conversation messages were added. Default config, UI state and
runtime policy fingerprints are unchanged. Logs show normal app stopping with no
error/critical or unhandled-exception lines, and both faulthandler logs are empty.
[ID checks](id-check.json) find this task and ADR's sole owners across 305 refs and
27 worktrees.

The guide documents app-session continuity and retry behavior; the testing lesson
distinguishes child recomposition from actual navigation. Native semantic reindexing
and process-exit resumption are outside qualification. No full suite, push or merge
was performed. Next bounded review: Search/RAG recovery links and return navigation.

## Reproduce the targeted run

```sh
.venv/bin/python -m pytest \
  Tests/UI/test_library_rag_rechunk_navigation.py \
  Tests/UI/test_library_rag_rechunk_feedback.py \
  Tests/UI/test_library_rag_rechunk_action.py \
  Tests/UI/test_library_rag_legacy_chunk_report.py \
  Tests/UI/test_library_rag_legacy_chunk_report_real_backend.py \
  Tests/Library/test_library_rechunk_service.py \
  Tests/RuntimePolicy/test_rechunk_policy_pin.py \
  Tests/UI/test_library_rag_mode_scope_keyboard.py \
  -q --tb=short --show-capture=no
```
