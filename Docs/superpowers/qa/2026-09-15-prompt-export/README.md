# Prompt Copy and Markdown export — TASK-32629

Copy and Export results now appear in the existing Prompt status area. The
status scrolls into view without moving focus or rebuilding editor fields.
This repairs stacked notifications that covered Copy and the next FileSave
Save button in the native 80×24 journey.

Export feedback uses an application notification when the original Prompt
editor is no longer active. It does not replace another Prompt's status.
The captured export content, source identity and mutation guards remain.
Existing ADRs 040, 086, 150 and 161 govern the repair; no new token, stylesheet,
storage contract or dependency was introduced.

The renderer's docstring also now matches the existing parser: empty author
and prompt lanes normalize to `None`, empty details remain an empty string,
and multiline lanes retain their interior blank lines. No rendering or parsing
logic changed.

## Automated evidence

[verification.txt](verification.txt) records **104 passing targeted checks**:
19 new journeys, 42 existing Copy/Export cases, 17 parser round trips,
13 design-token/bundle checks and 13 controller wiring checks.
The new journey file uses real SQLite and the production stylesheet harness:

- Legacy, structured Prompt and structured Recipe Copy/Export round trips at
  170×48 and 80×24, each in dark and light themes.
- Actual More actions, filename entry, Tab/Enter Save, Escape cancellation,
  readable focus return, and unchanged source row and live TextArea identity.
- Missing/throwing clipboard adapters, failed file write and successful retry.
  Feedback must paint inline without creating Prompt notifications.
- A delayed export result after another Prompt, the Browse list, or another
  screen becomes active; its feedback must preserve the current editor status.

Before repair, the compact failure regression failed because Copy feedback did
not reach the inline status. Self-review added a separate inactive-screen
regression: the route alone still looked active after the Library was covered.
The reporter now checks that Library is also the current screen.

The neighboring Copy/Export selection retains artifact admission, unsupported
metadata, compatibility structure, live unsaved content, path validation and
mutation guards. The parser/renderer round-trip file and design-token/bundle
checks also pass. No full repository test sweep was run.

[static.json](static.json) records unchanged Ruff diagnostics against
`9ff4839ce0`: 16 in the existing controller, 205 in the screen, 24 in the
neighboring canvas test file, and one in the renderer. Changed functions are
formatted; the new test file and retained native probe pass lint and formatting.
The existing debt was not reformatted across these large files.

## Native evidence

The probe uses the actual `TldwCli` with `LinuxDriver` in an owned tmux session,
exclusive profile ownership, and a fresh private copy of synthetic review data.
All ten configured database paths, the database base directory and data root
were checked within that private profile. No personal profile or provider
request was used. [native_check.py](native_check.py) is the executed probe;
profile creation and tmux ownership are external prerequisites.

The real Copy adapter emitted OSC52 whose decoded payload matched the Markdown
and file output. The owned tmux server had `set-clipboard off`, so this verifies
the terminal clipboard handoff without claiming operating-system clipboard
delivery. Clipboard unavailability is deliberately injected; the export error
is a real `FileNotFoundError` from a missing subdirectory. Retry writes a real
file through the actual picker. [exported-sample.md](exported-sample.md) retains
the multiline synthetic content.

The final run-004 (PID 41434) measured journeys cover 170×48 dark and 80×24 light. Failure, success
and the next picker were rendered and visually inspected:

| State | Wide | Compact |
| --- | --- | --- |
| Write failure and focused Export | [170×48](failure-170.svg) | [80×24](failure-80.svg) |
| Save focused in picker | [170×48](picker-170.svg) | [80×24](picker-80.svg) |
| Successful export and focus return | [170×48](success-170.svg) | [80×24](success-80.svg) |

The status, focused Export and picker Save controls remain readable. Long paths
and the scrolled compact filename show only the portion that fits their fields.
The final run records no deferred action-paint waits caused by Prompt toasts.

[result.json](result.json), [persistence.json](persistence.json), and
[native-log-review.json](native-log-review.json) retain measured sizes, normal
compact Ctrl+Q, fresh exit 0 and observed shell return, plus a read-only check
that the saved source retains version 1 and its original body.

Existing Console sidebar, optional-audio and worker-status notices remain
in the fresh log; the two export failures are intentional.
This qualifies the local Prompt journey, not clean startup or a provider call.

The earlier run-001 failed on a toast-obscured compact Copy control. Run-002
waited for Copy/Export toast expiry, then independently failed on the obscured
Save button; its [before capture](before-picker-80.svg) shows the cause. It
needed explicit dialog dismissal before normal quit and returned exit 1.
Neither failed run is counted as final verification.

## Remaining work

History, Collections and Use in Console are the next complete Prompt flows to
review. Skills, Collections, ingestion and other destinations remain in the
broader feature audit. Integration into `dev` remains pending.
