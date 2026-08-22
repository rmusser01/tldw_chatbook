---
target: Console bounded Context and Inspector section scrolling spec
total_score: 24
max_score: 40
na_heuristics:
p0_count: 1
p1_count: 2
timestamp: 2026-08-21T08-41-37Z
slug: 8-21-console-bounded-rail-section-scroll-design-md
---
# Console bounded rail sections — pre-implementation critique

## Design Health Score

| # | Heuristic | Score | Key issue |
|---|---|---:|---|
| 1 | Visibility of System Status | 2/4 | Overflow is signaled, but zero-height Context bodies and two indistinguishable scroll layers are not. |
| 2 | Match System / Real World | 3/4 | Literal scroll language works; `Other` and nested scroll ownership remain system-centric. |
| 3 | User Control and Freedom | 2/4 | Boundary handoff is strong, but the shortest supported state is impossible and unfunded sections lack recovery. |
| 4 | Consistency and Standards | 3/4 | Shared mechanics are coherent; the 80x24 promise contradicts protected chrome. |
| 5 | Error Prevention | 2/4 | Deterministic allocation prevents corruption, but stale reconciliation and ambiguous scroll ownership can misdirect input. |
| 6 | Recognition Rather Than Recall | 2/4 | Headers and hints help, but users must infer which layer moves and why an open section is empty. |
| 7 | Flexibility and Efficiency | 3/4 | Line/page/Home/End behavior is strong; long Inspector navigation remains serial. |
| 8 | Aesthetic and Minimalist Design | 3/4 | Natural heights reduce waste; repeated hints and a possible `Other` bucket weaken hierarchy. |
| 9 | Error Recognition and Recovery | 2/4 | Focus recovery is planned, but constrained layout and disappearing descendants lack user-facing recovery. |
| 10 | Help and Documentation | 2/4 | A guide update is planned; in-context help does not distinguish local from outer scrolling. |
| **Total** |  | **24/40** | **Promising, but revise the short-height and interaction contracts before implementation.** |

## Design Specificity Verdict

The structure is strongly authored for Chatbook rather than category-interchangeable.
Its Context allocator, Inspector ownership map, staged Sources ordering, stable row IDs,
run state, exact terminal geometry, and focus/persistence rules are specific to this
product. The weaker layer is moment-to-moment interaction language: both scroll owners
use the same generic hint, an open zero-height section can resemble a failed load, and
`Other` exposes an implementation fallback as user-facing information architecture.

The deterministic layout scan returned zero findings for all three source targets:
`left_rail.py`, `right_rail.py`, and `console_run_inspector.py` (exit 0, JSON `[]`).
Those are not false positives, but the detector cannot model Textual compositor
geometry. A production-CSS compositor probe found the blocking 80x24 contradiction and
confirmed the dynamic reconciliation seams the static scan cannot see.

No browser overlay is available: Console is a native Textual terminal compositor, not
an HTML DOM. Production-CSS Textual regions and scroll metrics were the visual/runtime
fallback signal.

## Overall Impression

This is a mature, unusually explicit design. Exact 20/21 accounting, stable hint rows,
source authority, focus recovery, and pointer boundary handoff are all well conceived.
The biggest opportunity is to make the nested-scrolling model as understandable as the
geometry is rigorous. One short-height promise is impossible as written, and the
current wording does not fully answer the user's two questions: “what will move?” and
“why is this open section empty?”

## What's Working

1. **Honest geometry:** 0/1/20/21 physical-row behavior, wrapped-row counting, stable
   hint slots, and no overlaying content form a testable contract.
2. **Product semantics are protected:** Sources remains first, Scope remains compact,
   existing IDs/actions retain meaning, and Context/Inspector retain different outer
   behaviors for good reasons.
3. **The core mechanics fit Textual:** native line/page/Home/End bindings and pointer
   event bubbling already support the intended keyboard scrolling and boundary handoff.

## Priority Issues

### [P0] The 80x24 all-header promise is physically impossible

The production Context body is ten rows high at 80x24. Seven current direct headers
consume fourteen rows before any content or hint allocation; only the first three fit.
Zero-height bodies therefore cannot preserve every header without outer scrolling.

**Fix:** use an explicit short-height fallback. The strongest option is to allow outer
Context scrolling only when headers physically cannot fit, while keeping the expanded
header-first allocator at normal heights. At 120x30 and other constrained-but-feasible
states, prioritizing the most recently activated Context section avoids an “open but
empty” result without changing persisted open preferences.

### [P1] Local and outer hints do not identify the scroll owner

Both a section and the Inspector rail can simultaneously show `▼ more — scroll`, yet
the next wheel or arrow gesture can move different layers. First-time users must learn
by trial; regular users can still move the wrong viewport after focus changes.

**Fix:** keep the requested local copy, but differentiate the outer cue or name the
next hidden section. Pair it with a dimensionally stable, non-color-only active state
across the focused viewport and its header.

### [P1] Live content changes need an explicit reconciliation contract

Textual descendant Mount, Unmount, and Resize events do not bubble. Sources, Changed
Files, Agent, Character, Run groups, Session Settings, and live-work cards can all
change through independent sync/recompose paths without resizing the bounded ancestor.
Allocations, focusability, and local/outer hints could become stale.

**Fix:** require every named mutation owner to request the shared body's coalesced
post-refresh reconciliation; section reconciliation must also invalidate the outer
Inspector hint. Use Textual's native pointer bubbling instead of custom event reposting.

### [P2] Focus traversal through scrollboxes containing controls is underspecified

Overflowing viewports become focus stops while Sources, Changed Files, Agent, and
live-work bodies can contain inputs and buttons. The spec does not define Tab order,
reverse traversal, descendant auto-reveal, or recovery when a focused descendant
disappears.

**Fix:** specify `viewport → interactive descendants in DOM order → next section`,
reverse it for Shift+Tab, fully reveal focused descendants, and define nearest-control
then header/collapse recovery for every shrink/recompose path.

### [P2] Inspector macro-navigation and `Other` weaken the information architecture

Twenty-line caps stop one section from becoming infinite, but up to fifteen direct
boundaries can still create a long serial document. A generic `Other` heading makes the
important Review Changes action harder to predict and normalizes future ownership gaps.

**Fix:** retain a semantic `Changes` boundary for Review Changes, fail development tests
for unknown rows/actions rather than exposing them as `Other`, and add a lightweight
next/previous-section path or make the outer hint name the next hidden section.

## Persona Red Flags

### First-time technical user

- Identical local and outer hints require experimentation to learn the active owner.
- An expanded header with a zero-height body resembles a compositor failure.
- Every overflowing viewport adds a focus stop without explaining its keyboard role.

### First-time non-technical user

- “Scroll” does not say whether to scroll the section or the whole Inspector.
- An open-but-empty Context section provides no plain-language cause or recovery.
- `Other`, `RAG/source`, and Source Readiness rely on internal product vocabulary.

### Regular technical power user

- A long Inspector remains serial without section-to-section navigation.
- Focus recovery is incomplete when a live update removes an interactive descendant.
- Stale reconciliation could leave the visible hint inconsistent with actual content.

### Regular non-technical user

- Reopening a section at a preserved mid-content offset lacks a “partway through” cue.
- Identical hint language undermines the predictability of the new pattern.
- `Other` is not a stable category and could accumulate unrelated actions over time.

## Minor Observations

- A blank one-row hint slot at scroll end is the right anti-layout-shift behavior, but
  production-CSS tests should ensure it reads as spacing rather than missing content.
- The conversation tray already performs deferred fitting and nearest-scroll-parent
  restoration; tests must prove the new local bounded body becomes the intended parent.
- Sources has three cap seams, not two: inline composition, CSS, and sync-time
  reapplication. Session Settings likewise has both a CSS minimum and inline maximum.
- The static detector's clean output should be recorded as “clean but insufficient for
  native runtime layout,” not as proof that the design is verified.
