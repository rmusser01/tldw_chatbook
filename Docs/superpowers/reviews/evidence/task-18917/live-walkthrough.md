# TASK-18917 live walkthrough evidence

Date: 2026-08-29

## Production-shaped mounted matrix

Command:

```text
../../.venv/bin/python -m pytest Tests/UI/test_library_adaptive_reader_closeout.py -q -k 'notes_branch_paging_is_contained_focusable_and_collapsible_in_production_shell'
```

Result: `4 passed, 7 deselected`.

The mounted harness uses the real `LibraryScreen`, adaptive-reader shell,
Notes canvas, and the repository stylesheet hierarchy. At 160×50, 120×35,
100×30, and 80×24 it exercised long identifying titles, independent root and
expanded-folder folder/note pagers, branch-local scroll preservation, a gated
one-shot failure, stable Retry focus, focus on the first appended note,
Library/Items collapse, compositor text, and horizontal containment. The
unchanged closeout cases continue to cover Conversations, Media, Prompts, and
Skills destination identities and reader behavior.

### RED and correction evidence

- At 160×50 the real compositor initially allocated the Items owner at
  `x=36 width=40`, but the Notes canvas/list painted at `x=38 width=40`, two
  cells beyond the Items content edge. Constraining the Notes canvas/list to
  `1fr`, `min-width: 0`, and `max-width: 100%` produced `x=38 width=36`, fully
  inside the Items edge at column 76.
- Explicitly closing Items was immediately undone by Notes list-priority
  reconciliation. Automatic list priority is now gated by the requested
  Items-open preference. Mounted regressions close Items, reconcile, resize,
  open Library, and close Library again; requested and effective Items state
  remain closed throughout.
- At 100×30 and 80×24, opening Library evicted Items as intended, but closing
  Library could leave both navigation panes closed: the explicit Items
  priority fit at the breakpoint and was then rejected by stale hysteresis.
  The explicit Notes interaction may ignore only the previous hysteresis
  state; the current-allocation guard remains unconditional. A transient
  allocation regression proves an explicit close cannot resolve against
  stale geometry.
- Prompt and Skills Items recomposes could leave the old same-ID filter focused
  after its child was detached. Their shared production canvas lifecycle now
  clears focus before teardown and restores the current same-ID child after
  recompose, while preserving a newer mounted focus choice. The cross-reader
  route cycle presses F6 without a test-side focus repair and verifies the focus
  owner is mounted.
- At 80×24 the identifying long title is intentionally ellipsized by the
  compositor; its identifying text remains painted and contained. This was a
  test-expectation correction, not a production defect.

## Isolated real-repository walkthrough

Command:

```text
../../.venv/bin/python -m pytest Tests/Live/test_library_notes_tree_paging_live.py -q -s
```

Result: `1 passed`.

The test creates a `tmp_path` ChaChaNotes SQLite database and drives
`LocalNoteFolderRepository` through `NotesScopeService` in the mounted
production-shaped Notes reader. It seeds 25 roots, 25 Unfiled notes, one
parent with 25 children and 45 visible placements, a deep managed descendant,
a duplicate manual/managed placement, and a managed placement shadowed at its
ancestor. Repository assertions confirm exact totals of 25 root folders, 25
children, 25 Unfiled placements, 45 primary placements, and one visible
deepest placement.

The repository pages are collected directly before mounting. Both duplicate
membership IDs are asserted independently; the shadowed managed ancestor is
absent from all three Primary pages; and its exact descendant membership is
the sole Deepest placement. The Pilot then loads root and child continuations,
expands Primary, gates a one-shot failure, proves the disabled loading pager
retains focus, observes the exact Retry control and focus, retries to 40 loaded
placements, locates a placement in the 20–39 middle page, and observes the
exact `Notes 21–40 of 45  Load earlier` control. It locates one duplicate by
exact manual membership ID, then collapses Primary and locates the exact
Deepest placement, proving all four ancestors expand and the exact descendant
membership receives focus.

The final isolated run emitted:

```text
duplicate_memberships=df0be05a-3a99-4521-bd8c-0cd1428213f2,c57010af-8ce2-42f0-93ba-b82b61676f75
duplicate_focus=df0be05a-3a99-4521-bd8c-0cd1428213f2
deepest_focus=cdc2c1da-b11c-4be1-a7dd-02e792e65373
retry=Couldn’t load more · Retry (control id equals focus id)
earlier=Notes 21–40 of 45 Load earlier
```

Exact final control/compositor and geometry observations were:

| Terminal | Items width | Work width | Observation |
|---|---:|---:|---|
| 160×50 | 40 | 77 | title control `Shadowed managed ancestor`; compositor `Shadowed managed`; pager control `Notes 1–20 of 25  Load more notes`; compositor `Notes 1–20 of 25` |
| 120×35 | 56 | 50 | same exact title/pager controls and identifying compositor text |
| 100×30 | 42 | 48 | same exact title/pager controls and identifying compositor text |
| 80×24 | 32 | 38 | same exact controls; identifying text is intentionally clipped but contained |

All four compositors painted the Notes/Library navigation context, the Notes
list stayed within the Items owner, and no horizontal overflow was observed.

## Targeted and static closeout

The exact Task 8 feature/cross-reader command completed with `697 passed in
218.57s (0:03:38)`.
This includes the Notes models, real repository, scope service,
pure paging and tree state, mounted navigator/canvas/shell/closeout coverage,
and the isolated live walkthrough; it is not the full repository suite.

`tldw_chatbook/css/build_css.py` now emits the stable header `Generated:
deterministic`; a two-clock regression proves identical sources produce
byte-identical app bundles. After staging the intended generated baseline, the
exact prescribed build followed by `git diff --exit-code --` across all five
generated sheets exited 0 without normalizing or ignoring any line. Ruff on
the changed Python files, `git diff --check`, and the requested `compileall`
over Notes, Library, Library widgets, and `library_screen.py` also exited 0.
