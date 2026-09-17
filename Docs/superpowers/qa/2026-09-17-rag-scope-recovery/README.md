# Search/RAG scope recovery — TASK-2377

2026-09-17 UTC (September 16 Pacific), on `feat/component-pattern-library`,
based on `7718676e8f`. Recovery visibility now follows the currently mounted
scope container across navigation, whole-screen composition and panel recomposition.

## Defect and repair

The old boolean cache outlived the widgets it described. Starting with source
count A, leaving Search, receiving count B, returning to Search, then receiving
count A again left the B recovery UI mounted. The equal cached boolean suppressed
the necessary update. Both directions reproduced: a false empty-Library warning
beside available sources, and a missing recovery action after sources disappeared.

The cache now pairs a weak reference to the scope container with the requested
recovery visibility. A replacement container cannot reuse the previous container's
change gate. Weak references avoid keeping detached widget trees alive. Eager
cache writes still coalesce repeated snapshots before the mirror starts; both
the mirror and full refresh record the container/state actually rendered under
the existing lock. The docstring names all five current full-refresh callers.

No layout, tokens, provider behavior, persistence or source authority changed.
No new ADR is required; existing
[ADR-003](../../../../backlog/decisions/003-settings-library-rag-defaults.md)
and [ADR-150](../../../../backlog/decisions/150-design-token-system-and-design-language.md)
govern the unchanged boundary and design language.

## Automated evidence

- [Red](regression-red.txt): all six mounted cases failed the final recovery
  assertion before repair. The matrix covers both count directions through
  navigation, screen recompose and panel `sync_state`/recompose.
- [Initial green](regression-green.txt): those six cases and the existing timeout
  and steady-state recovery checks pass (8 tests).
- [Targeted gate](final-targeted-tests.txt): **268 passed in 138.14s**, with no
  exclusions. Includes the full Search/RAG gate16 file, query-gate race/paint
  checks, Library RAG state and the focused recovery tests.
- [Stronger steady-state assertions](steady-state-tests.txt): **6 passed** after
  adding direct repeated sync calls, checking scope-child identity, and verifying
  with a wrapped spy that no mirror worker is scheduled in either state. Identical
  snapshot payloads return before the recovery cache, so the direct calls are
  needed to prove that this cache itself avoids rebuilding unchanged widgets.
  These cases overlap the gate above; counts must not be summed.
- [Lint comparison](lint-comparison.json): no new diagnostics in existing files
  (controller 2, screen 205, state 0). The new test and native runner pass lint
  and formatting; the state file and changed production ranges pass formatting.
  [Independent review](review.json) identified a ready-state no-op scheduling
  assertion gap; the wrapped spy closes it, and final review has no remaining
  findings. Diff whitespace checks pass.

Final gate command:

```sh
.venv/bin/python -m pytest \
  Tests/UI/test_library_rag_scope_recovery.py \
  Tests/UI/test_library_rag_query_gate_race.py \
  Tests/UI/test_product_maturity_gate16_library_search_rag.py \
  Tests/Library/test_library_rag_state.py \
  Tests/UI/test_library_shell.py::test_library_rag_source_snapshot_timeout_then_real_snapshot_clears_recovery \
  Tests/UI/test_library_shell.py::test_library_rag_scope_recovery_steady_state_snapshot_causes_no_churn \
  -q --tb=short --show-capture=no
```

## Native evidence

[Runner](native_check.py), [final result](result.json),
[lifecycle](lifecycle.json) and [capture hashes](capture-hashes.json).
Final profile: `/private/tmp/tldw-2377-run-003`. Real TldwCli uses LinuxDriver,
an owned tmux TTY with both streams attached, an exclusive instance lock, and a
primed terminal capability probe. Navigation and recovery actions use focused
controls followed by Enter; query text/focus and source snapshots are set
programmatically.

Each of the four theme/size cells runs both source-count directions around the
actual Search → Notes → Search route. The runner verifies that the cache still
names the old scope before applying the critical snapshot, so an intervening
query refresh cannot conceal the defect. Query focus/text survive the snapshot,
Run agrees with source availability, and repeated snapshots retain the children.
Each ready cell submits a real local keyword search and verifies the exact seeded
Media ID/title; a service wrapper records four completed real calls without
altering arguments or outcomes. Each empty cell focuses and activates Import media.

| Theme/size | Local result | Query/empty state | Focused recovery action |
| --- | --- | --- | --- |
| Dark 170×48 | [View](textual-dark-170-ready.svg) | [View](textual-dark-170-empty.svg) | [View](textual-dark-170-recovery-action.svg) |
| Dark 80×24 | [View](textual-dark-80-ready.svg) | [View](textual-dark-80-empty.svg) | [View](textual-dark-80-recovery-action.svg) |
| Light 170×48 | [View](textual-light-170-ready.svg) | [View](textual-light-170-empty.svg) | [View](textual-light-170-recovery-action.svg) |
| Light 80×24 | [View](textual-light-80-ready.svg) | [View](textual-light-80-empty.svg) | [View](textual-light-80-recovery-action.svg) |

The first successful run's eight overview captures were rendered and inspected;
the confirmation pass adds four inspected, focused-action captures. At 80 columns
the message sits below the query viewport, then wraps readably above the focused
recovery button after it scrolls into view. Compositor assertions require both
the message and action to be painted before activation.

[Run-001](run-001.json) stopped when the runner attempted to focus hidden Search
navigation inside compact Notes. It exited normally with status 1; it is not
matrix qualification. Opening the real Notes Library grip corrected the runner.
[Run-002](run-002.json) passed the full matrix. Run-003 adds recovery-action paint
and captures post-search focus geometry; no further product change was needed.

All ten private databases pass read-only SQLite `quick_check`; source before/after
records match and no conversation messages were created. Default config/UI/runtime
fingerprints match across all attempts. The final app log has no error/critical
entries, faulthandler is empty, `app.run` returned, shell status is 0, and exact
PID absence was verified before closing the owned terminal.

Limits: source counts are deliberately injected around a real persistent Media
record to control timing; this does not qualify database deletion or source-snapshot
producers. The real searches are local keyword searches, not semantic retrieval,
provider generation or remote access. No full suite, push or merge was performed.

The captures expose a separate keyboard issue, now TASK-32707: result reveal
scrolls the still-focused query outside its viewport (y=-19 at 80×24), while the
footer still says typing in field. This is the next bounded review; the scope
cache repair does not claim to fix result-arrival focus.
