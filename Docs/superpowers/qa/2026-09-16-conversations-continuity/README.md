# Library Conversations continuity — TASK-32701

Baseline: `0f7cb59e41` on `feat/component-pattern-library`.

Resizing a conversation reader no longer lets Notes restore focus into Library
navigation and discard an unsubmitted Find draft. Explicitly reopening Items
restores its last focused field with the same selection. Found messages remain
visible in short terminals: the reader can scroll, its transcript has a
token-backed minimum height, and the settled reveal scrolls the transcript
viewport before the row. A newer focus choice cancels that delayed reveal.

This implements the existing [adaptive-reader contract](../../../../backlog/decisions/086-library-adaptive-reader-shell.md)
and [design language](../../../../backlog/docs/design-language.md). No new ADR is
required; ADR-031, ADR-150 and ADR-161 also apply. Token values, conversation
authority, persistence and application structure are unchanged.

## Native evidence

[native_check.py](native_check.py) runs the real `TldwCli` with its terminal
driver, a fresh private profile and two disposable conversations created through
the real database API. It uses mounted controls and keyboard actions to open
Alpha, edit list/Find drafts, submit Find, switch Read/Info, and recover from an
empty filter. Resizes cover 80×24, 170×48 and 170×24 in both Textual themes.
Library and Items may collapse to preserve the reader; the filter journey
explicitly reopens Items through its grip.

[Run-004](native-result.json) passed all journeys, including empty-filter
recovery at both compact and wide sizes. Ten SVG captures were inspected in
one batch: focused query borders and selection, compact Clear filter, and the
focused message body are readable. Wide captures retain the three-pane layout.
[Inspection](inspection.json) records the exact capture hashes and limits.
The reader's top actions can scroll above the viewport when revealing a message;
the selected message remains above the footer.

[Persistence](persistence.json) records ten healthy private databases, exactly
two fixture conversations and 26 unchanged message bodies, and unchanged hashes
for the three checked default-profile configuration files. No unhandled handler
exception was logged. [Lifecycle](lifecycle.json) records normal app return,
shell exit 0, absent app PID, and closing only the owned terminal after exit.

Earlier functional runs remain in ignored scratch. Run-001 reproduced the
nested-viewport visibility defect before its repair. Run-002 began with the
terminal still compact and could not focus the hidden Alpha row; the runner
now explicitly establishes its starting size. Run-003 passed the initial
journeys; run-004 added compact empty-result coverage. These functional runs
preceded the single visual inspection batch.

## Automated verification and review

[Verification](verification.json) records 158 passing neighboring checks plus
the final 12 focused checks: ten new continuity cases and two updated keyboard
tests. The original eight continuity tests produced six failures on baseline
and eight passes with the repair. Removing the current-focus guard caused both
Find/race cases to fail; restoring it passed. The held-callback regression moves
focus to Info before releasing old resize work, then verifies visible Info and
unchanged outer scroll.

Two unchanged closeout tests still fail on baseline and this change: the route
cycle compares a media selector with a widget ID, and Notes paging assumes
visible Items at 80×24. They remain outside this Conversations repair. The
initial neighbor run also exposed two obsolete keyboard expectations: explicit
Library reopen now correctly restores its remembered descendant before F6, and
`Widget.focus()` must settle before inspecting Escape hints. The corrected
tests assert those contracts and pass. No full suite ran.

The new tests and native runner are lint-clean. Four checked files pass Ruff
formatting. [Static comparison](static-comparison.json) records 215 inherited
Ruff findings with no new findings across the five edited existing Python
files; three retain pre-existing whole-file formatting debt. Generated CSS
freshness and design/component governance passed. [Independent review](review.json)
found no production defects and confirmed the final evidence and keyboard
expectations. [Task-ID collision check](id-collision-check.json) found this task
only in its owning worktree across 267 refs and 27 worktrees.

## Reproduction and limits

Prepare a fresh profile with `data/`, `data/db/` and `config/`; every configured
data/database path must resolve beneath it. Disable catalog refresh and use
disposable local data. The runner validates the profile before importing the
app, sets private config/XDG paths and a null keyring, requires exclusive access,
and establishes 170×48 before navigation. Run it in an owned tmux terminal:
`native_check.py PROFILE SOCKET SESSION`. Retain its unique exit receipt and
verify the recorded PID is gone before closing the terminal.

These checks qualify list filtering, Read/Info, Find and resize continuity.
They do not qualify Archive/Restore, Export, Resume, provider calls, every
transcript shape or arbitrary terminal sizes. The separate TASK-32700 real
Import lifecycle remains blocked by host semaphore allocation (errno 28).
