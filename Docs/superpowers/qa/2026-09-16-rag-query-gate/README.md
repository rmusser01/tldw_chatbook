# Library Run/disclosure coherence — 2026-09-16

TASK-2530 keeps the reserved query disclosure mounted. Snapshot sync and full
refresh now update it and Run together without yielding; an older refresh only
rebuilds conditional recovery widgets, so it cannot overwrite the latest pair.
If the disclosure widget is missing, Run stays disabled.

Native inspection also found that the old sentence clipped before its provider
name at 80 columns. The ready copy is now `To {provider}: question + evidence`.
The mode toggle still explains that Search stays local. The one-line reservation
and Run position are unchanged; no stylesheet or token value changed.

ADR required: no new ADR. This repairs existing
[ADR-003](../../../../backlog/decisions/003-settings-library-rag-defaults.md)
and [ADR-150](../../../../backlog/decisions/150-design-token-system-and-design-language.md)
behavior without changing provider authority, billing, or retrieval boundaries.
The task's original suggestion to disable Run after teardown was replaced by a
retained, coherent pair: a final write alone would leave the intermediate gap.

## Automated evidence

The scheduling regression [failed in all four cases](regression-red.txt) before
repair and [passed afterward](regression-green.txt): source arrival, source loss,
provider replacement, and missing disclosure. The barrier holds an actual
conditional widget removal, then invokes the real synchronous snapshot updater.
Assertions cover both the suspended and settled full refresh.

The later compositor regression [failed in four compact cells](compact-red.txt)
with the old copy; wide cells passed. Final coverage checks the complete painted
notice, reserved height, and stable Run position in both modes and themes at
170×50 and 80×24, using OpenAI and the longer `local_transformers` name.

The [final targeted run](final-targeted-tests.txt) passed **275 tests in 147.45s**:
query race/paint, keystroke preservation, Search/RAG gates, display-state models,
token governance, and generated-CSS consistency. No full suite was run. Two
unrelated tests were deselected: the previously tracked evidence-heading failure
(TASK-15390), and focused-card Open with `source_id="media-1"` (follow-up under
TASK-4111). The latter [also fails with the unchanged HEAD controller](baseline-failures.txt).

The initial [broader run](targeted-tests.txt) passed 258 and failed two. Its other
failure was an unmounted widget's inline-height assertion; token CSS supplies that
height after mounting. That assertion also fails at baseline and is now replaced
by the stronger mounted geometry checks described above. Counts overlap between
runs and must not be summed.

New test/QA files pass Ruff and formatting. [Baseline comparison](static-baseline.json)
shows no added diagnostics in touched existing files; controller/model counts
remain 2/7 and existing test-file counts remain 1/5. Changed production ranges are
formatted. Two independent review passes found no actionable issue, including
after the compact-copy change; see [review](review.json).

## Native evidence and limits

[Run-004](result.json) used real TldwCli, LinuxDriver, a TTY-backed console, an
exclusive private-profile lock, and a primed terminal capability probe. Actual
navigation opens Explore all tools, Search/RAG, RAG Answer, and pastes the query.
The scheduling check then injects **synthetic display states** at the panel-state
seam and suspends a conditional removal under the real panel lock. It invokes
the production snapshot updater and checks the full painted disclosure during
and after that yield. Query focus/value survive; removing scope disables Run.

These are native widget/scheduling checks, **not provider readiness, retrieval,
generation, or a paid request**. No Run or Send action is activated. Synthetic
panel counts intentionally differ from the real empty Library rail. Navigation
uses programmatic focus followed by Enter; complete Tab traversal and mouse
operation are not qualified here.

All eight final SVGs were rendered and inspected. The full notice is visible in
both compact cells, and the disabled gate keeps Run in place. [Inspection](inspection.json)
records this scope. [Capture hashes](capture-hashes.json) distinguish raw and
stored whitespace-normalized SVGs.

| Theme/size | Ready | Disabled |
| --- | --- | --- |
| Dark 170×48 | [Ready](textual-dark-170-ready.svg) | [Disabled](textual-dark-170-blocked.svg) |
| Dark 80×24 | [Ready](textual-dark-80-ready.svg) | [Disabled](textual-dark-80-blocked.svg) |
| Light 170×48 | [Ready](textual-light-170-ready.svg) | [Disabled](textual-light-170-blocked.svg) |
| Light 80×24 | [Ready](textual-light-80-ready.svg) | [Disabled](textual-light-80-blocked.svg) |

[Persistence](persistence.json) confirms ten healthy private SQLite databases,
zero messages, unchanged default config/UI/runtime hashes across all attempts,
and no error/critical/traceback log lines or faulthandler bytes. [Lifecycle](lifecycle.json)
confirms app return, shell exit 0, and independent PID absence before closing only
the owned terminal. This checks selected default files, not a whole-home diff.

[Attempt history](native-attempts.json) retains two runner navigation failures;
those launches also redirected terminal stderr and do not qualify native paint.
Run-003 qualified the scheduling fix but exposed [compact clipping](compact-before.svg).
Run-004 adds exact compositor-text assertions and closes that gap. A first
read-only SQLite inspection used system Python and could not open a database;
the project's interpreter completed all ten checks.

To reproduce, run the prior [profile preparer](../2026-09-16-handoff-excerpts/prepare_profile.py)
with the project interpreter and a fresh absolute temporary path. Start an owned
tmux session at 170×48, set PYTHONPATH to this checkout, and run:

```text
.venv/bin/python Docs/superpowers/qa/2026-09-16-rag-query-gate/native_check.py PROFILE SOCKET SESSION
```

Keep stdout/stderr attached to the terminal. Record shell exit, verify PID absence,
then close only that session. Final raw profile: `/private/tmp/tldw-2530-run-004`.
The unrelated context-metadata drift (deprecated PRODUCT register, stale design
metadata, missing buildPath) remains outside this repair.

Next bounded review: TASK-4111 result opening, then the remaining Search/RAG journey.

Follow-up, 2026-09-17: [TASK-15390](../2026-09-17-rag-evidence-heading/README.md)
is closed. The heading exclusion above relied on the open historical task; its
original ordering failure was already fixed and did not reproduce on this branch.
The follow-up isolates a separate saved-dev profile-storage dependency and runs
the complete gate16 file without exclusions. Historical run counts above are
unchanged.
