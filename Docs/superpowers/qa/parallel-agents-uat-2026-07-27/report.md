# Parallel-agents fleet UAT — 2026-07-27

Live user-acceptance test of the parallel-agents stack at dev tip `c0f68ad1c`
(post #975/#976/#1030/#1041). Real TUI in tmux (235×52, SGR click injection),
scratch profile `pa_uat`, llama.cpp at `:9099` (gemma-4-26B), two seeded
workspaces (Red/Blue) each with one read-only bound folder containing a
codeword file. Captures under `captures/`.

## Scenario results

| # | Scenario | Result |
|---|----------|--------|
| S1 | Viewed-session agent run: two approval rounds (own-file read, cross-workspace attempt), Approve-all→Submit each, correct answer ("crimson"), zero toasts for viewed completion | PASS |
| S7 | Workspace confinement: own-workspace read OK; cross-workspace read `outside every allowed root` | PASS |
| S2 | Background park: Blue run backgrounded → verbatim toast `Agent in Blue Chat (Blue) needs approval.`, ◆ on tab, collapsed group header `Blue ◆` aggregate, Agent rail section auto-opens | PASS (but see F1) |
| S3 | Sticky Agent-section collapse (task-915): collapsed while fleet busy; stayed collapsed through a new viewed run's payload changes; persisted preference honored after quiet | PASS |
| S4 | Visit-mounts-parked-card; ◆ persists until decision; Approve-all→Submit resolves | PASS (run itself ended in the loop guard, F3) |
| S5 | Background completion: verbatim toast `Agent in Red Chat (Red) finished.`, ✓ marker, clear-on-visit, correct answer ("nosmirc") | PASS |
| S6 | Settings ▸ Console Behavior cap row: edit 3→1, draft "unsaved"→Save→"saved"; cap honored on next send with honest refusal `1 agents already running (Red Chat). Wait for one to finish or interrupt it.` | PASS |
| S8 | Top-level section aggregate (task-912 AC#1) | BLOCKED by F4 — section headers not user-collapsible |

## Findings

### F1 — Important (UX): fleet summary line lives below the rail's scroll fold
`#console-agent-fleet-summary` renders with correct text and a fully-displayed
ancestor chain, but sits at the BOTTOM of the Agent rail section, below the
viewed session's status and step bullets. Whenever the rail content above it
(session browser + agent step details — i.e., after any agent run) fills the
viewport, the line is off-screen; found live only by wheel-scrolling deep into
the rail (captures/s2up.txt shows the section ending at the fold with the line
absent). Headless proof: with a done viewed session + parked background
session, the widget's region is `y=48` in a 44-row viewport — display chain
all-True, so `test_fleet_summary_line_is_reachable_on_the_live_rendered_surface`
passes while the user sees nothing. The spec's "at a glance" intent is carried
today only by the tab/sidebar ◆ markers. Suggested: render the fleet line at
the TOP of the Agent section (above status/steps) or pin it outside the
scrollable flow; strengthen the test to assert viewport intersection
(`region.y < viewport.height`), not just the display chain.

### F2 — Important (correctness): duplicate park toast for an already-parked round
With Blue parked (toast already shown minutes earlier), the completion of the
VIEWED session's (Red's) run re-fired `Agent in Blue Chat (Blue) needs
approval.` for the same still-parked round. The once-per-card guard does not
survive whatever re-marshal/re-park the viewed-run-completion sync performs.
Reproduction sketch: park B; run+complete a run in viewed A; observe second
toast at A's completion. Needs a repro test + guard keyed on card identity
across re-derives.

### F3 — Observation (no action): agent loop guard fires on model loops
Blue's second run ended with `Agent run stuck: loop detected: read_file
repeated in a 1-cycle (3x).` The guard worked as designed (run terminated,
viewed, no stray toast); the loop itself is model behavior (gemma re-calling
the same tool). No app defect.

### F4 — Important (UX): top-level browser section headers are dead affordances
Starred/Workspaces/Chats headers render ▾/▸ carets but do not respond to
clicks — three attempts (caret column exact, caret+1, label) left the
Workspaces section expanded. Group rows (e.g. Blue) toggle fine. Consequences:
(a) a misleading affordance; (b) task-912's section-level marker aggregation
(collapsed-section ◆) is unreachable through live UI interaction — it can only
manifest via persisted default collapse (e.g. empty Chats). Either wire section
headers for click-toggle or drop the caret glyph from non-interactive headers.

### F5 — Design-visibility (backlog): screen navigation silently kills the fleet
Navigating Console → Settings (to edit the cap) and back destroyed the parked
Blue run: on_unmount tears down the controller (denies pending rounds — the
AA-1052-documented instance lifecycle), and the return builds a fresh
controller with no markers, no toast, no record of what died. From the user's
seat: background agents do not survive leaving the Console screen, and nothing
says so. Deliberate architecture, but the visibility gap deserves either a
confirm-on-navigate-with-busy-fleet, a returning notice ("2 runs were
cancelled when you left"), or at minimum documentation in the user guide.

## Verdict
The shipped core loop — parallel runs, confinement, parking, round-identity
approvals, markers, toasts, sticky collapse, cap row — behaves as specified on
the real surface. Four findings filed as tasks (F1/F2/F4/F5); none block
day-to-day use, F2 is the only correctness-class defect and it is cosmetic-
duplicative rather than approval-unsafe (no auto-approve, no wrong-round
resolution observed anywhere in the session).
