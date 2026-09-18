# Conversation workspace link receipt — TASK-32388

Baseline: `33079ca5a2` on `feat/component-pattern-library`.

A retained conversation link receipt and its Undo now appear only when the
workspace ID they name is active and the conversation remains eligible there.
Both initial composition and retained reader updates apply the same projection.
Undo rechecks that condition when handling an event; an old press after a
workspace change, missing/unavailable registry context or stale list has no
membership effect. Returning to the original workspace with the same loaded
conversation restores its receipt, and Undo removes only that original link.

This repairs the existing ephemeral receipt contract under
[ADR-005](../../../../backlog/decisions/005-console-workspace-server-readiness.md),
[ADR-150](../../../../backlog/decisions/150-design-token-system-and-design-language.md)
and [ADR-161](../../../../backlog/decisions/161-component-pattern-library.md).
No new ADR is required.

## Native and persistence evidence

[native_check.py](native_check.py) runs the real `TldwCli` terminal driver with a
fresh private profile, one real saved conversation/message and two explicit
workspaces. Mounted Link and Undo controls are activated with Enter. Workspace
changes use the real registry API while Library remains mounted; retained sync
is exercised at 170×48 and recomposition at 80×24, in both Textual themes.
This does not qualify navigation through the Console workspace switcher.

[Native result](native-result.json) records all four journeys passing: receipt
and Undo disappear in Beta, return in Alpha, and Undo leaves only Beta's link.
All eight captures were inspected in one batch. The named receipt and focused
Undo remain readable at both sizes. Compact mode collapses Navigation and Items;
with Undo focused, the message body extends below the viewport. This pass
qualifies the receipt/action, not compact transcript reveal.
[Inspection](inspection.json) records native and stored SVG hashes; only
whitespace-only SVG source lines were normalized.

[Persistence](persistence.json) confirms ten healthy private databases, exactly
one unchanged fixture conversation/message, only Beta's remaining membership,
and Alpha still active. Three checked default-profile configuration hashes
are unchanged. The private log contains no errors, tracebacks or unhandled
exceptions. [Lifecycle](lifecycle.json) records normal app return, shell exit 0,
PID absence and closing only the owned terminal after the app exited.

## Automated checks and review

[Verification](verification.json) records 144 distinct targeted passing tests:
112 neighboring checks followed by a final 59-check run (27 overlap), covering
the added stale-list case and design/component/bundle governance. Five receipt
cases failed against baseline production code. A separate guard mutation made
the stale Undo case delete the original link; restoring the guard passed.
Three syntax warnings came from source parsed by the existing governance test.
No full suite ran.

The duplicate-name scenario proposed in the initial plan was removed because
workspace names must be unique. Its invalid-fixture failures are excluded from
the red evidence. The native run preceded a final narrowing from broad exception
handling to `WorkspaceRegistryServiceError`; the final mounted unavailable
context test verifies that documented registry error contract.

[Static comparison](static-comparison.json) records no new lint findings across
the three edited existing Python files (212 inherited findings and inherited
whole-file formatting debt). The native runner passes Ruff lint and format;
the two changed test functions pass formatting independently.
[Independent review](review.json) found no actionable production regressions,
including after the stale-list test and typed error catch were added.

## Reproduction and limits

Prepare a fresh profile with `data/`, `data/db/` and `config/`; every configured
data/database path must resolve beneath it. Disable catalog refresh and use
only disposable local data. The runner validates paths before importing the
app, sets private config/XDG paths and a null keyring, primes the terminal probe,
requires exclusive profile access and establishes its initial 170×48 size.
Run in an owned tmux terminal: `native_check.py PROFILE SOCKET SESSION`.
Keep normal TTY streams, record a unique shell exit receipt and verify the
recorded app PID is absent before closing the terminal.

These checks cover retained workspace link receipts and exact-target Undo.
They do not qualify workspace-switcher navigation, archive/restore, export,
provider calls or the separately blocked TASK-32700 Import lifecycle.
