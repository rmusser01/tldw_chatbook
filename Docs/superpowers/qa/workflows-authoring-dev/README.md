# Workflows authoring visual evidence

TASK-32601, 2026-09-14. Capture script reuses the reviewed Textual/Rich export
method and runs the new Workflows screen through the real TldwCli navigation and
authoring lifecycle. The existing app test factory substitutes unrelated service
startup; the workflow database, services, screen, widgets and CSS are real.
The five-step file-to-note definition is synthetic test content.

```sh
PYTHONPATH=. DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib \
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python \
  Docs/superpowers/qa/workflows-authoring-dev/capture.py
```

Run from the authoring worktree. Use only the test sandbox established before app
imports, never the installed user profile. Splash is disabled in that disposable
config; unrelated model-catalog refresh is substituted, and the test network
guard is installed. Wait only for workflow control workers, not all periodic app
workers. Do not overlap this capture with another app-boot test process.

Outputs are in `.impeccable/review/workflows-authoring-dev/`: wide overview,
and selected-step frames at 160x48, 110x36 and 60x20. Each retains the raw Textual
SVG, an equivalent Menlo SVG, and PNG. The capture asserts exact non-stylesheet
geometry/content equality between the SVGs.

## Capture validity

Final capture exited 0; all four PNGs opened and verified nonblank/correctly named.
The layouts show three panes, two panes, then an editor with labeled selectors. Run is disabled.
This is visual evidence, not complete behavioral qualification.

- Installed font: Menlo, /System/Library/Fonts/Menlo.ttc, fixed-pitch spacing 100.
- Actual Cairo advances at 20px: iiii and WWWW both 48.1640625, normal and bold.
- Disabled Run contrast: 7.2544:1 at every size.
- Search workflows contrast: 6.7679:1 at the wide size.

The shared factory logs its intentionally absent ChaChaNotesDB. The interpreter's
existing RequestsDependencyWarning also appears. These are recorded as harness/
environment noise, not hidden or treated as clean output.

Selected-step exports also assert that both the Prompt label and its populated
reference value appear in the raw Textual SVG, before font conversion. The
fixture's literal `{{ prepare.text }}` is synthetic, not a provider response.

## Independent review

The fresh UI review found two material issues: ambiguous “Saved locally” status,
and a focused Prompt field whose label was above the 60x20 viewport. One batched
fix clarified saved revision versus stored draft and brought the label into view.
Its recapture revealed a blank field: the real app's inherited vertical padding
consumed the compact TextArea's content area. A second scoped fix removed that
compact vertical padding. A real-app regression verifies label, populated value
and focus together.

The final same-reviewer verdict was **disposition: ship**, with both original
findings resolved and no visible regression from those fixes across the four
frames. This is a fix-list verdict, not a second whole-surface audit. There were
three capture batches total: initial, first correction, second correction. The
output paths retain only the final reviewed frames.

Behavioral/static test evidence is separate in TASK-32601 and the implementation
report. No HTML design detector is applicable to this Textual surface. Global
DESIGN.md, PRODUCT.md and stale Impeccable metadata are not rewritten.

## Built surface

The authoring editor retains the established Operate workbench: terminal
monospace, compact labeled controls, semantic focus and readable disabled Run
with an explicit unavailability explanation.

Layout follows available content width: at 132 columns or more, a 28-column
library, 27-column navigator and flexible editor; at 96–131, navigator/editor
plus “Workflow…”; below 96, editor plus “Workflow…” and “Step / Overview…”.
The final 160×48 overview shows an ordered selectable step list; selected-step
frames at 160×48, 110×36 and 60×20 show the continuous form. Below 30 rows,
unexpanded form text areas use compact sizing without vertical padding.
The 60×20 Prompt capture shows label, value, caret and focus together.

Inputs, Action, Outputs, Execution and Advanced fold independently. Action
starts open; missing required Inputs open automatically. Per-selection fold,
scroll and focus state is retained. Current source also keeps the already-focused
raw JSON field visible during reconciliation and serializes widget removal/mounting
with native `asyncio.Lock`. Those interaction corrections postdate the visual
verdict and preserve the layout.

Shared visual values come from `tldw_chatbook/css/core/_variables.tcss`: `$ds-*`
spacing, control sizing, surfaces, emphasis, focus and disabled states.
`tldw_chatbook/css/features/_workflows.tcss` owns the feature's pane/field geometry.

The pinned status distinguishes “Saved revision · draft unchanged”, “Draft
stored · not a saved revision” and “Draft pending · not stored yet”. Saving a
revision remains an explicit action. The reviewer's ship disposition covers
only the status and 60×20 Prompt fixes.
