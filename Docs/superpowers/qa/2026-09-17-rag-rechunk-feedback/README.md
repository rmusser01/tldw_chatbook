# Re-chunk progress and completion feedback — TASK-32718

2026-09-17 UTC, `feat/component-pattern-library`, based on `79a7a270d6`.

Mode or source changes rebuilt the Re-chunk children with an enabled button and
no progress or receipt. The compact receipt also clipped its re-index disclosure
to one row. The panel now retains its own running state and summary and rebuilds
children from them; the summary uses normal Static auto height. The shared
Re-chunk/backfill refusal guard, policy admission and service behavior are unchanged.
Another panel retains the existing admission/refusal behavior; this change does
not promise receipt persistence across navigation or restart.

ADR required: no. This routine repair implements existing
[ADR-078](../../../../backlog/decisions/078-chunking-template-convergence.md),
[ADR-031](../../../../backlog/decisions/031-tui-keybinding-and-footer-hint-conventions.md)
and [ADR-150](../../../../backlog/decisions/150-design-token-system-and-design-language.md).
No new visual tokens, stylesheet rules or storage/service contracts are needed.

## Automated evidence

- [Red evidence](feedback-red.txt): **7 failed, 1 passed** before the fix. Four
  theme/size journeys expose lost disabled/progress state, two completed-receipt
  rebuilds lose the receipt, and the compact unrebuilt receipt clips its note.
- [Targeted run](targeted-tests.txt): **75 passed in 69.87s**. Eleven new cases
  cover keyboard start, active mode/scope changes, completed receipt retention
  and wrapping, plus missing-service, policy-denied and worker-error recovery
  followed by successful retry. Existing cases cover real SQLite re-chunking,
  census responsiveness/unmount, forced reindex/cache behavior, policy pinning,
  backfill exclusion and the previous mode/scope focus repair.
- Two existing CSS checks still read only the former monolithic sheet. The
  [baseline probe](css-baseline.txt) reproduces both failures with starting panel
  source. [Its plugin](baseline_source_plugin.py) substitutes only that source
  read from `79a7a270d6`; this is not a full baseline checkout. Both tests now use
  the existing `app_css_text()` helper to include the generated Library sheet,
  preserving every state/token/class assertion. No stylesheet changed.
- [Static checks](static-checks.json): new test/runner/plugin lint and formatting
  pass, as do changed production/test ranges. A broader formatting probe includes
  an unchanged wrapped expression; it is recorded and left untouched.
  [Lint comparison](lint-comparison.json) has no added diagnostics (five existing
  production diagnostics and two in the existing test file).
- [Independent review](review.json): no actionable production, test or runner
  finding. No full suite or exclusions were used.

## Native evidence

The [runner](native_check.py) boots real TldwCli with a private profile and native
tmux terminal. It seeds five actual Media records with unstamped chunk rows and
workspace memberships. One source is empty and remains skipped; each cell adds
one valid source. A timing gate holds the real scope method before delegation,
allowing mode/scope changes during work without replacing the real re-chunk
service. Semantic indexing is disabled in this profile. No provider or retrieval
request is made; the dummy provider readiness only permits changing to RAG mode.

All four final theme/size cells [pass](result.json). Tab reaches Re-chunk from the
mode control and Enter starts it. Changing mode and Media selection retains
disabled/progress state without another run. Each actual run re-chunks one item,
skips the empty item, and lowers the legacy census from two to one. Switching mode
after completion preserves the full receipt. Keyboard scrolling reveals the
progress and wrapped receipt; compositor text checks the complete disclosure.
Four valid items receive the current engine stamp and replacement chunks; the
skipped item's chunks and all five source records remain unchanged
([before](source-before.json), [after](source-after.json)).

The initial native run also passed all behavior/lifecycle checks
([result](initial-run-result.json), [lifecycle](initial-run-lifecycle.json)).
Inspection showed prior completion toasts obscuring compact controls. The runner
now waits for natural toast expiry. One fresh confirmation run produced the eight
final captures below; all were rendered and inspected, with unobscured progress,
counts and complete compact re-index disclosure. No production change followed
the first visual inspection. [Capture hashes](capture-hashes.json) record raw and
stored bytes; only trailing SVG source-line whitespace was normalized.

| Theme / size | Running | Completed |
| --- | --- | --- |
| Dark / 170×48 | [Capture](textual-dark-170-running.svg) | [Capture](textual-dark-170-completed.svg) |
| Dark / 80×24 | [Capture](textual-dark-80-running.svg) | [Capture](textual-dark-80-completed.svg) |
| Light / 170×48 | [Capture](textual-light-170-running.svg) | [Capture](textual-light-170-completed.svg) |
| Light / 80×24 | [Capture](textual-light-80-running.svg) | [Capture](textual-light-80-completed.svg) |

[Lifecycle checks](lifecycle.json) verify app-run return, exit 0 and PID absence
before closing the owned terminal. Ten private databases pass read-only
`quick_check`; no conversation messages were added. Default config, UI state and
runtime policy fingerprints remain unchanged. Normal app stopping is logged,
without error/critical or unhandled-exception lines; both faulthandler logs are
empty. [Task-ID checks](task-id-check.json) find this task's sole owner across
306 refs and 27 worktrees.

The guide documents keyboard access and same-panel persistence. The testing
lesson records why display gating does not preserve children across parent
recomposition, and why full Static text is insufficient evidence of readable
notes. Actual semantic reindexing, remote providers and cross-destination receipt
persistence are outside this native qualification. Next bounded review:
navigating away from and returning to Search/RAG while Re-chunk is active.
No full suite, push or merge was performed.

## Reproduce the targeted run

```sh
.venv/bin/python -m pytest \
  Tests/UI/test_library_rag_rechunk_feedback.py \
  Tests/UI/test_library_rag_rechunk_action.py \
  Tests/UI/test_library_rag_legacy_chunk_report.py \
  Tests/UI/test_library_rag_legacy_chunk_report_real_backend.py \
  Tests/Library/test_library_rechunk_service.py \
  Tests/RuntimePolicy/test_rechunk_policy_pin.py \
  Tests/UI/test_library_rag_mode_scope_keyboard.py \
  -q --tb=short --show-capture=no
```
