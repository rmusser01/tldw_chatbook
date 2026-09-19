# Saved Console sidebar startup — TASK-32702

Baseline: `df121a4169` on `feat/component-pattern-library`.

Fixed the caught `_sidebar_state_save_timer` AttributeError observed during
TASK-32700's fresh-process Import checks. `ChatScreen.__init__` loaded saved state
before creating the timer, dirty flag and persistence-worker fields. Assigning
the reactive sidebar state invokes its watcher immediately, including when the
saved mapping is empty. The loader caught that watcher failure as a load error.
The constructor now creates those existing fields before loading state, so it
also retains any timer/dirty state established during restoration.

No persistence schema, debounce duration, worker, quit-flush, styling or rail
preference behavior changed. ADR required: no; this is a routine initialization
bug fix within the existing contract.

## Automated evidence

- [Red receipt](red.json): both new empty/populated saved-profile cases fail on
  the unchanged source and log the missing-timer error.
- [Final targeted run](sidebar-tests-final.txt): **10 passed in 15.90s** across
  `test_chat_screen_sidebar_state_debounce.py` and `test_chat_screen_ui_state_path.py`.
  These cover real construction, restored expansion/search/section values,
  debounced writing, immediate-quit flushing, in-flight writes, reset and profile
  path isolation. The added test runs real Textual reactivity and file persistence.
- The [broader selection](targeted-tests.txt) has ten passes and two failures in
  `test_chat_screen_suspend.py`. Both fail identically on the
  [unchanged baseline](baseline-suspend-tests.txt): its fake fleet lacks
  `_console_wake_user_priority`. They bypass the constructor repaired here.
  No full suite ran; these baseline failures remain outside this task.
- [Static comparison](static.json): no new Ruff diagnostics; the screen has the
  same 212 inherited findings, with embedded line offsets normalized for comparison.
  Fatal checks and formatting of the edited range pass. The test and native runner
  are lint/format clean; whole-screen formatting debt remains unchanged.
- [Independent review](review.json): no actionable findings.

## Native save and restart

[Save](save-result.json) and [restart](restart-result.json) are separate native
`TldwCli`/`LinuxDriver` processes using one validated private profile and the same
recorded application/runner hashes. A seeded `ui_state.toml` contains:

```toml
[sidebar]
search_query = "saved search"
last_active_section = "notes"

[sidebar.collapsible_states]
notes = true
chat = false
```

The first process restores all fields, changes `restart-check` to true through
the real reactive persistence signal, and requests normal Ctrl+Q without a
debounce wait. The second process restores that added value and the original
fields. This verifies the stored sidebar-state contract; the fixture keys do not
claim to exercise visible Collapsible gestures or the separate Console rail
preference system. Exact immediate-teardown behavior is pinned by the mounted
automated regression; terminal delivery timing is not a hard real-time guarantee.

[Lifecycle](lifecycle.json) records normal `app.run()` return, shell exit 0 and
separate process-absence checks for both processes. Only the owned terminal was
closed afterward. [Persistence](persistence.json) records ten healthy private
databases, zero conversations/messages, unchanged default-profile hashes and
the final saved values. The app log contains no ERROR/CRITICAL lines or traceback
headers, and the private faulthandler log is empty. No provider request was sent.

The [final Console capture](console-restart.svg) was inspected once at 170×48 in
textual-dark; [inspection](inspection.json) records its hashes and limited claim.
This change has no visual styling scope and does not constitute another complete
Console layout review.

## Reproduce

Prepare a new private profile with `data/`, `data/db/` and `config/` directories,
all configured data/database paths beneath it, catalog refresh disabled and the
seed above. The helper reuses the fail-closed
[profile guard](../2026-09-16-ingest-lifecycle/native_check.py). Set `PYTHONPATH`
to this checkout's absolute root and use its `.venv/bin/python` in an owned tmux
session sized 170×48. Run these as separate processes in order, waiting for exit
and recording its code between phases:

```text
.venv/bin/python Docs/superpowers/qa/2026-09-16-sidebar-startup/native_check.py PROFILE SOCKET SESSION save
.venv/bin/python Docs/superpowers/qa/2026-09-16-sidebar-startup/native_check.py PROFILE SOCKET SESSION restart
```

The helper validates isolation before importing the app, primes the terminal
capability probe, verifies its application import path and requires exclusive
profile ownership. Each phase refuses an existing output directory. Preserve
independent exit, log and profile checks alongside the runner results.
