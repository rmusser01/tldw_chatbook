# MCP tool-switch labels — TASK-32822

Compact Servers panes clipped non-master tool names and their on/off states.
The existing full-width, auto-height wrapping rule now covers every secondary
action in the tool-gate group, including the master. It still covers the Tools
master control. No tokens, settings handlers, permission policy or stylesheet
registrations change. ADR required: no; this applies ADR-150 and ADR-161.

## Targeted verification

The [baseline matrix](red-results.json) has four intended compact geometry
failures and four wide passes. The [qualified case ledger](qualified-cases.json)
records **37 distinct passing cases**: eight new full-label/keyboard/disabled
layout cases, four existing master-control layout cases, seven gate rendering/
routing/state cases, three workbench persistence/focus cases and fifteen token,
bundle, byte and selector guards. Each case ran serially in an isolated profile.
No full suite was run. The [selected cases](selected-cases.txt) and individual
logs retain the exact coverage and fixture diagnostics.

CSS bytes are **584,112 / 608,090**; selector candidates remain **274 / 274**.
No limits or exceptions were increased. All [seven preflight guards](preflight.txt)
pass. The layout scan reports no findings; the new test and native runner pass
Ruff and formatting. Independent review is recorded alongside this receipt.

## Native visual and lifecycle evidence

The real TldwCli runs with a private profile and LinuxDriver in a native terminal.
Normal MCP navigation and keyboard Servers/built-in selection lead to the gate
group. Direct focus of the first gate is followed by real Tab traversal of all
eleven controls. Every focused control fits its clip and paints its whole label.
Deep research is then directly focused and activated with Enter: the real private
setting saves on and back off in each cell, with focus restored after rebuild.
Only the requested tools setting changes; permission profiles remain unchanged.

All [eight final captures](GALLERY.md) were rendered and inspected, covering
Deep research on and Ask user focus after reversal at 80×24 and 170×48 in both
themes. The compact Deep research label wraps with its complete state; Ask user
and the master remain readable. The [native receipt](native/result.json) records
four passing cells and twelve source hashes matching the final source.

The [lifecycle receipt](lifecycle.json) confirms normal App.run return, exit 0,
absent PID, released profile lock, ten healthy private SQLite databases, zero
conversations/messages, unchanged default config/UI state/runtime policy, and
no error or faulthandler output. The terminal exits with its wrapper.

The [first native attempt](failed-native-001/result.json) failed before mode
entry because the runner queried the Workbench for a button owned by its screen.
Only that harness selector was corrected. Its [clean shutdown receipt](failed-lifecycle-001.json)
is retained; it contributes no final visual qualification. The first final-run
lifecycle probe used bare python3/SQLite 3.51.0 and could not open six databases
read-only. The same check with the app's .venv Python 3.12/SQLite 3.49.1 passed all
ten; no database or production changes were made to obtain that result.

The allocation receipt confirms this task's sole ownership across local refs
and worktrees. Wider Servers lifecycles, concurrent gate saves, connected servers
and tool execution remain outside this bounded layout repair. PR2707 remains
draft/unmerged and requires its own visual review and merge approval.
