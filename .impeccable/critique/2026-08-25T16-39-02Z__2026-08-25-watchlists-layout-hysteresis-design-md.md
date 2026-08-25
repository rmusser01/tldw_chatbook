---
target: Watchlists layout hysteresis design spec
total_score: 32
max_score: 40
na_heuristics: 
p0_count: 0
p1_count: 4
timestamp: 2026-08-25T16-39-02Z
slug: 2026-08-25-watchlists-layout-hysteresis-design-md
---
# Watchlists Layout Hysteresis Design Critique

## Design Health Score

| # | Heuristic | Score | Key issue |
|---|---|---:|---|
| 1 | Visibility of System Status | 3 | Collapsed grips do not expose manual, responsive, or Article Focus provenance. |
| 2 | Match System / Real World | 4 | Collapse early and reopen with breathing room matches stable resizable tools. |
| 3 | User Control and Freedom | 3 | Explicit actions bypass hysteresis, but priority lifetime needs definition. |
| 4 | Consistency and Standards | 3 | Matches Library's four-column policy, but three-pane ordering must be explicit. |
| 5 | Error Prevention | 3 | Prevents flap and persistence errors; initialization still needs a precise contract. |
| 6 | Recognition Rather Than Recall | 3 | Grips preserve reachability, but collapse cause remains implicit. |
| 7 | Flexibility and Efficiency | 4 | Keyboard and grip actions remain immediate; no debounce latency. |
| 8 | Aesthetic and Minimalist Design | 4 | Protects Reader calm without adding visible controls. |
| 9 | Error Recovery | 3 | Rollback is preserved, but no-op request bookkeeping needs definition. |
| 10 | Help and Documentation | 2 | The spec is strong; the surface does not explain temporary collapse. |
| **Total** | | **32/40** | **Good; UX direction sound, state contract needs hardening.** |

## Design Specificity Verdict

The design is strongly authored for Chatbook. Terminal-cell boundaries, the permanent
Reader, named collapse priorities, ASCII grips, Article Focus, persisted preferences,
and asynchronous pane mounts make it product-specific rather than generic responsive
layout prose.

The deterministic scan returned zero findings. That is expected for a Markdown design
spec and does not prove the described behavior exists. Code inspection confirmed the
implementation gaps the task names: Watchlists has no previous-state resolver input,
uses child workbench width before screen width, sends equal-layout requests, and does
not distinguish passive from explicit recomputations.

Browser overlays were not applicable because the target is a design document rather
than a rendered route; no server was started.

## Overall Impression

The asymmetric four-column dead band is the correct interaction. The main opportunity
is to turn the prose into an executable transition contract so the controller cannot
interpret the same width differently after focus, tab, rollback, or initialization
transitions.

## What's Working

- Passive geometry changes are correctly separated from explicit user intent.
- The threshold behavior is concrete: collapse below `T`, reopen at `T + 4`, and stay
  open while shrinking back to `T`.
- Equal-layout suppression protects both visual stability and the asynchronous mount
  path without adding debounce latency.

## Priority Issues

### P1 — Multi-pane reopening order is not executable yet

The spec names reverse reopening but does not define how failed reopen candidates alter
the width calculation for later candidates. Define deterministic evaluation in reverse
collapse order and include the all-open Read thresholds: Feed Items 95, Navigation 119,
Inspector 149.

### P1 — Recompute cause must be explicit

One controller method serves mount, resize, manual toggle, Article Focus, rollback, and
section changes. A boolean default is easy to misuse. Give every call an explicit cause
or explicit previous-state argument; only passive resize may consume responsive history.

### P1 — Temporary priority needs a lifecycle contract

Define set, replace, park, clear, and rollback behavior across Article Focus and section
changes. Otherwise a tab round-trip can change the pane arrangement at the same width.

### P1 — First settled width needs an operational definition

Name the exact Textual width property. Zero or fallback widths must neither seed
hysteresis history nor issue workbench requests. Specify what happens if a positive
width later transiently returns to zero.

### P2 — No-op suppression and rollback bookkeeping interact

If an explicit preference action produces no DOM transition, it must not attach rollback
state to a stale request token. Compare against controller-desired state, issue no token
for an equal layout, and create apply rollback state only for a real request.

## Persona Red Flags

- **Power user:** priority behavior that changes after a tab round-trip breaks muscle
  memory even when the layout looks stable.
- **Keyboard-only reader:** boundary tests must prove focus does not move on no-op widths
  and is handed to the grip only on a real collapse.
- **First-time user:** the same grip can mean manual, responsive, or Article Focus
  collapse. Perfectly predictable restoration is essential while provenance remains
  intentionally invisible.

## Minor Observations

- Treat Library parity as behavioral parity; importing a Library constant would create
  an unnecessary Watchlists-to-Library dependency.
- Add one management-tab example and an explicit-open-inside-the-band test.
- Compare no-op results to controller-desired state while an async request is in flight,
  not only to the last rendered workbench state.

## Questions to Consider

- Should temporary pane priority be preserved and parked across Article Focus and tabs,
  or deliberately cleared at a named boundary?
- Should first settled mean the first positive screen allocation, the first Resize after
  mount, or the first acknowledged workbench layout?
