# TASK-32783 — Tool Profile Bind handoff

Bind preserves the active workspace's unsaved Persona/memory fields while
staging the chosen Tool Profile. It opens the visible Persona picker, or Apply
when a Persona is already staged, with guidance beside that control. The
Default workspace receives visible create/selection guidance. No defaults are
saved by Bind itself; existing memory and imported-profile confirmations apply.

The handoff waits for pane replacement and respects newer focus, category,
workspace and dialog ownership. Independent review reproduced a delayed valid
callback after layout: queued focus made the reveal see the old search anchor.
Synchronous guarded focus fixes that ordering. The new regression and the
reviewer's same probe both pass; no remaining introduced blocker was found.

## Targeted evidence

91 distinct targeted cases pass; no full suite or provider requests were run.

| Scope | Cases | Evidence |
| --- | --- | --- |
| Draft retention, visible continuation, memory preservation, delayed valid/stale handoffs, theme/size cases | 14 final-source | [Focused](focused.txt) |
| Existing Tool Profiles, assistant defaults/journeys, token/component governance | 77 before the final one-line focus correction | [Regressions](regressions.txt) |
| Existing Bind handlers rechecked after that correction | 2 repeated from the 77 above | [Final existing Bind](existing-bind-final.txt) |

[Red evidence](red-summary.txt) records the original six failures and the later
review regression. Scoped Ruff introduces no diagnostics (Settings 114→114;
existing tests 3→3); new tests and runner are clean. Changed methods and new files
pass formatting. Backlog and diagnostic inventory guards pass. The inventory
check required the project Python: system Python 3.9 cannot parse the repository's
Python 3.12 generic syntax. No production change was made for that tooling error.

## Native visual qualification

The real app uses LinuxDriver and both TTY streams in a fresh private profile,
with real Persona, workspace registry and tool-pack services. Each cell imports
an actual service-exported pack, checks Default recovery, stages in an explicit
workspace, selects a Persona, returns through Tool Profiles and checks that Bind
retains the draft. It cancels first-bind review without saving, then explicitly
confirms a fresh review and verifies exact Persona/profile/read-only defaults.
Fixture preparation creates the source profile and workspace rows; service and
dialog behavior are not replaced.

| Theme / viewport | Default recovery | Persona continuation | Preserved draft / Apply | Exact first-bind review |
| --- | --- | --- | --- | --- |
| Dark 80×24 | [View](textual-dark-80x24-recovery.svg) | [View](textual-dark-80x24-persona.svg) | [View](textual-dark-80x24-draft.svg) | [View](textual-dark-80x24-review.svg) |
| Dark 170×48 | [View](textual-dark-170x48-recovery.svg) | [View](textual-dark-170x48-persona.svg) | [View](textual-dark-170x48-draft.svg) | [View](textual-dark-170x48-review.svg) |
| Light 80×24 | [View](textual-light-80x24-recovery.svg) | [View](textual-light-80x24-persona.svg) | [View](textual-light-80x24-draft.svg) | [View](textual-light-80x24-review.svg) |
| Light 170×48 | [View](textual-light-170x48-recovery.svg) | [View](textual-light-170x48-persona.svg) | [View](textual-light-170x48-draft.svg) | [View](textual-light-170x48-review.svg) |

All sixteen final captures were rendered and visually inspected. Focus and the
full guidance are painted at both sizes and in both themes. Review remains
scrollable while its Cancel/Confirm controls stay visible.

[Final native result](native-result.json), [capture hashes](capture-manifest.json)
and [lifecycle receipt](lifecycle.json) pin final source and runner. Normal Ctrl+Q
exit returned 0; the app PID was absent before terminal closure, the instance
lock was reacquired, all eleven private DBs passed integrity checks,
conversations/messages stayed empty, and default-profile fingerprints were
unchanged. There were no error or faulthandler logs.

Run001 stopped on a duplicate fixture workspace name, then exited cleanly;
[its receipt](failed-run001-lifecycle.json) is not passing journey evidence.
Run002 completed four cells on the earlier queued-focus implementation and is
[historical only](pre-final-run002-native-result.json). Run003 is the final-source
qualification above. Both earlier profiles were checked before terminal closure.

ADR-107, ADR-079 and ADR-150 govern this bounded repair. The existing deferred
focus lesson now records this additional incident. The [review ledger](../../reports/2026-09-18-tool-profiles-review.md)
retains the separate MCP Edit geometry gap and concurrent-workflow scope. Parent
PR CI also reports the prior compact action selector exceeding its structural
CSS ratchet by one; that pre-existing change is a separate follow-up.
