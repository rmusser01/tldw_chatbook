---
id: TASK-25887
title: 'Console Inspector: the blocked-state guidance is below the fold at 128x40'
status: To Do
assignee: []
created_date: '2026-08-31 21:35'
labels:
  - console
  - ux
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
When no provider is configured, the Inspector's entire job is to say so and tell the user what to do. At 128x40 with the Inspector open, the three rows that carry that message -- Setup, Blocked impact, and Next action: Set up provider -- all render below the rail's visible bottom. The user sees only 'Run recipe' and 'Live work: No active work', neither of which mentions that sending is blocked. This is what test_console_workbench_standard_width_inspector_snapshot has been failing on; the test is correct and the app is wrong.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 At 128x40 with the provider unconfigured and the Inspector open, the user can see that send is blocked and what to do about it, without scrolling
- [ ] #2 test_console_workbench_standard_width_inspector_snapshot passes against the app rather than by weakening its assertion
- [ ] #3 The chosen behaviour is pinned by a test that asserts on VISIBLE geometry, not just DOM presence
<!-- AC:END -->

## Evidence

Probed from inside pytest (a standalone script does not reproduce it -- the
`Tests/UI` conftest fixtures are what make `#console-inspector-rail-open`
hittable). Harness identical to
`test_console_workbench_standard_width_inspector_snapshot`: 128x40, onboarding
complete, Inspector opened by click.

```
RAIL=(94, 6, 34, 28)  visible bottom y=34
BODY=(96, 27, 30, 60)          <-- 60 rows of content in a 28-row rail

  y= 28 vis      console-inspector-run-heading      'Run'
  y= 29 vis      console-inspector-run-recipe       'Run recipe: OpenAI /'
  y= 33 vis      console-inspector-live-work        'Live work: No active work'
  y= 34 CLIPPED  console-inspector-setup            'Setup: Provider configuration'
  y= 36 CLIPPED  console-inspector-blocked-impact   'Blocked impact: Send is'
  y= 41 CLIPPED  console-inspector-next-action      'Next action: Set up provider'
  y= 42 CLIPPED  console-inspector-provider         'Provider: blocked - openai is'
```

Every row is mounted with the right text. The three that matter are one row
past the fold. What the user actually sees is "Run recipe" and "Live work: No
active work" -- and *nothing that says sending is blocked*.

## Why this was mistaken for test drift

The failing assertion is `assert "Blocked impact" in normalized_svg`, and an
SVG screenshot contains only what is painted. Two earlier readings of this were
wrong and are recorded so the next person does not repeat them:

1. "The row was renamed or removed by #2220." It was not -- it is mounted with
   its original id and copy.
2. "The assertion is stale: it wants a blocked row from a healthy fixture."
   Also wrong. The fixture IS provider-blocked -- the same test asserts
   `next_action.render_line(0) == "Next action: Set up provider"` and that
   assertion passes, because it reads the widget directly rather than the
   painted screen.

That contrast is the useful part: a DOM assertion and a screenshot assertion on
the same row disagree, and the disagreement IS the bug. The DOM one passes and
hides it; the screenshot one fails and finds it.

## Scope note

Bisected to `c2f64f690` (#2220), which is also where TASK-25715's header-padding
finding came from. The Inspector body renders 60 rows into 28. Deciding which
rows earn the visible band -- or whether a blocked run should scroll its
guidance into view -- is an Inspect rail design call, not a mechanical fix, so
this is filed rather than patched.

## The obvious fix does not work — checked before proposing it

`ROW_GROUPS` order drives display order (`console_inspector_ownership.py`, the
`for owner, _heading_id, labels in ROW_GROUPS: for label in labels:` pass), and
it lists the Run group as:

```
"Run recipe", "Live work", "Setup", "Send blocked",
"Recovery action", "Blocked impact", "Next action", "Provider"
```

Two things make "just move the blocked rows to the front" look right:

- `chat_screen.py` already expresses that intent -- it builds the blocked rows
  and does `rows=setup_rows + inspector_state.rows`, i.e. **prepend**. The
  ownership pass then re-sorts by the tuple above and silently discards it.
- The tuple order is NOT a #2220 decision. `git show c2f64f690` touches only the
  Source Readiness group (`Sources` -> `Retrieval`); the Run order dates from
  `e472110e8` (2026-08-21) and carries no comment defending it.

**It still does not fix this.** Measured from the probe, the Run group gets
**6 visible rows** (y=28..33): one heading plus five. The blocked-state content
needs nine — `Setup` wraps to 2, `Blocked impact` to 5, plus `Next action` and
`Provider`. Reordering would surface `Setup` and part of `Blocked impact` and
push `Next action: Set up provider` -- the only row that tells the user what to
DO -- from y=41 to roughly y=36. Still clipped.

So no permutation of these rows fits. The fix has to reduce what the blocked
state renders (shorter copy, or one combined row), or scroll the guidance into
view, or give the blocked state a compact summary that collapses the detail.
That is why this stays a design call rather than a patch.
