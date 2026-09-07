---
id: TASK-21595
title: >-
  Sweep timer and tick paths for Static update calls that lay out by default
status: Done
assignee: []
created_date: '2026-08-23'
updated_date: '2026-08-25'
labels:
  - performance
  - ui
  - textual
priority: medium
---
## Description

TASK-21501 found that the Console composer's cursor-blink tick called `Static.update(renderable)`
without `layout=False`, and since Textual's `Static.update` signature is
`(content, *, layout: bool = True)`, every blink armed a **full screen reflow** — 396
`Widget.arrange` calls per 6 ticks, about 1.9 reflows per second on an idle focused composer. The
method's own docstring said it must not do that.

That is unlikely to be the only place. Any `.update(` on a timer, tick, watcher or animation path
carries the same default.

## Acceptance Criteria

- [x] Every `.update(` call reachable from a `set_interval`, `set_timer`, animation or high-frequency watcher is enumerated
- [x] Each is either given `layout=False` with a stated reason the rendered size cannot change, or documented as legitimately needing layout
- [x] Each `layout=False` is justified by a geometry-equivalence check across the states the path can produce — not by inspection; TASK-21501's mutation showed `outer_size` alone is insufficient, since painted rows changed 1 → 2 while `outer_size` stayed constant
- [x] Layout-pass counts are measured before and after against a measured idle floor, not asserted as zero
- [x] A lint or guard test prevents a new timer-path `.update(` from defaulting to `layout=True`

## Evidence

From TASK-21501, measured with counters around `Screen._refresh_layout`, `Compositor.reflow`,
`Widget.arrange` and `Static._layout_updates`, over six draft shapes:

| per 6 blink ticks | before | after |
|---|---|---|
| `Screen._refresh_layout` | 6 | 0 |
| `Widget.arrange` calls | 396 | 0 |
| time in `_refresh_layout` | 3.1-6.5 ms/tick | 0 |

## Implementation Plan

1. Build a *census*, not a grep: parse the whole package with `ast`, collect every
   repeating-clock root (`set_interval`, plus `set_timer` callbacks that re-arm
   themselves), follow the intra-package call graph out from each, and collect every
   reachable `.update(` that does not pass `layout=`.
2. Prove the census can see: assert it finds the known roots (including the TASK-21692
   blink clock and at least one self-rearming one-shot) before trusting a green run.
3. Classify every site found. Fix only where the rendered size provably cannot change;
   document the rest with a stated kind of exemption.
4. Justify each `layout=False` by a geometry-equivalence A/B against the real layout
   engine (paint X with `layout=True`, scrub, paint X with `layout=False`, compare),
   not by reading the stylesheet.
5. Measure layout-pass counts before/after against a measured idle floor, arms
   interleaved in one process.
6. Ship the census as a guard test so a new timer-path `.update(` cannot default to
   `layout=True` unnoticed.
7. Mutation-test every new assertion.

## Implementation Notes

**Result: two material sites, both animation-rate, both fixed. 52 further sites
enumerated and classified; none of them wanted fixing.**

### The census

`Tests/Architecture/test_timer_path_static_update_inventory.py` rebuilds the census on
every run. It parses all 1,889 package modules, finds **35 repeating-clock roots** — 32
`set_interval` call sites plus 3 `set_timer` callbacks that re-arm *themselves* (an
interval spelled as a chain of one-shots; a `set_interval`-only sweep misses these, and
one of the two bugs found lives behind exactly that shape) — walks the call graph six
hops out from each, and collects every reachable `.update(` call. That yields **54
candidate sites** across 20 modules.

The census was checked for blindness against the timer callbacks it found *no* sites in
(`console_setup_modal._tick`, `console_background_effect._advance_frame`,
`main_navigation._update_overflow_hints`, `base_tamagotchi`): each was read, and each
genuinely repaints via `refresh(layout=False)`, a change-signature gate, or a reactive.

### Fixed (2)

| site | clock | before | after |
|---|---|---|---|
| `SplashScreen._update_animation` | `set_interval(animation_speed)`, 0.01–0.1 s across the shipped cards (**10–100 fps**) | 20 `_refresh_layout`, 20 `Compositor.reflow`, 40 `Widget.arrange` per 20 ticks | 0 / 0 / 0 |
| `PersonaBuddyWidget._paint_frame` | self-rearming `advance_frame` at the visual's frame duration (>= 10 fps), plus the 0.1 s poll | 20 / 20 / 20 per 20 ticks | 0 / 0 / 0 |

Measured under `ConsolidatedCSSApp` at 120x40, 20 ticks x 6 **interleaved** repeats in one
process (medians), idle floor 0 in both arms:

| | splash before | splash after | buddy before | buddy after |
|---|---|---|---|---|
| `Screen._refresh_layout` | 20 | **0** | 20 | **0** |
| `Compositor.reflow` | 20 | **0** | 20 | **0** |
| `Widget.arrange` | 40 | **0** | 20 | **0** |
| time inside `_refresh_layout` | 49.2 ms | **0** | 12.7 ms | **0** |
| process CPU over the window | 102.6 ms | 95.6 ms | 30.5 ms | 25.4 ms |

The work is gone, not deferred: the measurement window includes two event-loop settles
per tick, so relocated layout would still have been counted.

At the splash's shipped cadence that is 10–100 whole-screen reflows per second removed
from **startup**, the one moment the app is already contended (module imports, DB opens).

### Why `layout=False` is safe at each — proven, not inspected

Both surfaces are container-pinned, so content cannot size them:

* `#splash-display` is `width: 100%; height: 100%` in both sheets that select it
  (`features/_splash.tcss`, `components/_settings_splash_theme.tcss`).
* `#persona-buddy-frame` is pinned twice over — `BUNDLED_CSS` says `100%/100%`, and
  `_apply_geometry` *also* assigns `frame.styles.width/height = "100%"` **inline**, which
  beats every sheet.

TASK-21692 showed that reading the stylesheet is not enough (there, painted rows changed
1 -> 2 while `outer_size` stayed constant). So each is pinned by a geometry-equivalence
**A/B against the real layout engine** instead
(`Tests/UI/test_timer_path_layout_cost.py::assert_geometry_is_content_independent`): for
each content shape, paint it with `layout=True` and record a witness, *scrub* the widget
with a deliberately different shape (without which the `layout=False` arm would inherit
the first arm's geometry and pass vacuously), then paint the same content with
`layout=False` and compare. The witness carries `outer_size`, `container_size`,
`content_size`, `region`, `scrollable_content_region`, every sibling's region and size,
and the painted per-row cell widths. Eight content shapes each: empty, one row, exactly
at the box width, one past it, a full box, **taller than the box**, CJK double-width, and
a Rich `Text`.

### Classified, not fixed (52)

Every remaining site carries an entry in `CLASSIFIED_SITES` with one of four stated
kinds, and the guard rejects a reason that does not name its kind:

* **NEEDS-LAYOUT (39)** — the box really is content-sized, so `layout=False` would leave
  stale geometry on screen. That is a behaviour change, not an optimisation. Examples:
  `.console-video-preview-frame` is `height: auto` and the `Pixels` row count tracks the
  scaled image; the Console status chips and the Send disabled-reason strip are
  `width: auto` (the label length *is* the width) and assign `styles.width`/`display` on
  the same call; `.console-model-section-value` is `text-wrap: wrap` with `max-height: 3`.
* **NOT-A-WIDGET (6)** — `dict.update` / `set.update` / `ProgressBar.update(progress=)`,
  which are syntactically indistinguishable from `Static.update` in an AST.
* **NOT-PER-TICK (3)** — reached only through an equality gate, so the tick does not
  repaint (e.g. `_sync_console_rail_system_line`, gated on
  `_console_rail_system_line_last` since TASK-251; `_display_static_fallback`, which runs
  on the error edge that also stops the animation timer).
* **UNREACHABLE (6)** — `Widgets/loading_states.py`, `Widgets/status_dashboard.py` and
  `Widgets/detailed_progress.py` have **no importer anywhere in the repo**, production or
  tests. Their timer repaints cost nothing because nothing mounts them. Not retired here;
  noted as an out-of-scope finding.

Two Console tick paths that looked like candidates are *already* handled and show up in
the census as explicit: the TASK-21692 blink (`_render_visible_draft_only`) and the
setup-modal snow field (`_render_flakes(layout=False)`, TASK-21134).

### The guard

Four tests. The load-bearing one asserts every clock-reachable `.update(` is classified —
either it passes `layout=` explicitly, or `CLASSIFIED_SITES` says why it need not. A new
timer-path repaint lands there unclassified and fails, which is the only signal this cost
ever gets: nothing else in the suite counts layout operations. The other three keep the
guard honest — the census must still find its known roots (a silently-broken AST walker
would make everything else green), classifications must still name a real call, and a
reason must state its kind rather than being prose.

### Mutation results (every new assertion, per test)

| mutation | expected red | result |
|---|---|---|
| splash `layout=False` removed | cost test | RED: "matrix: 6 animation ticks cost 6 extra screen layout passes (idle floor 0)" (x3 cards) |
| buddy `layout=False` removed (all 4 calls) | cost test | RED: "6 pet frame repaints cost 6 extra screen layout passes (idle floor 0)" |
| `_update_animation` stops painting | cost test's paint half | RED: "the animation stopped painting new frames" (x3) |
| `_paint_frame` returns early | cost test's paint half | RED: "_paint_frame no longer repaints the frame surface" |
| `#splash-display` forced `height: auto` | geometry A/B | RED on all 8 shapes |
| buddy inline pin -> `auto` | geometry A/B | RED on all 8 shapes |
| splash `layout=False` removed | census guard | RED, names `SplashScreen._update_animation:437` |
| buddy `layout=False` removed | census guard | RED, names `PersonaBuddyWidget._paint_frame:[757,762,769,771]` |
| `_clock_roots` returns `[]` | census-can-see test | RED: "census collapsed to 0 clock roots" |
| fabricated allowlist entry | staleness test | RED, names the fabricated key |
| reason without a kind prefix | kinds test | RED, names the entry |

**Two mutants survived first and were findings about the tests, not passes:**

1. The buddy geometry A/B survived making `#persona-buddy-frame` content-sized, because
   the harness was a plain `App` with `CSS_PATH = tldw_cli_modular.tcss` — and that
   bundle contains **zero** `persona-buddy-frame` rules; they live in the generated
   `widget_defaults_self.tcss`, which only `ConsolidatedCSSApp` registers. The widget was
   mounting unstyled, so the test was measuring nothing. Both harnesses now inherit
   `ConsolidatedCSSApp`. (Even then the CSS mutation still survived — the *real* pin is
   the inline `frame.styles.height = "100%"` in `_apply_geometry`, which is what the
   mutation finally had to target.)
2. The census guard survived removing the splash fix, because its key was
   `(file, receiver)` and `_display_static_fallback` writes to a local also named
   `display` — one allowlist entry was covering two distinct call sites. The key is now
   `(file, enclosing Class.method, receiver)`.

### Not touched

No timer lifecycle changed anywhere — both edits add one keyword argument to a
`Static.update` call. Creation, pause/resume, `stop()`, `on_unmount` and quit paths are
byte-identical, so there is no shutdown surface to walk.

### Test counts

* New: `Tests/UI/test_timer_path_layout_cost.py` **7 passed**;
  `Tests/Architecture/test_timer_path_static_update_inventory.py` **4 passed**.
* Touched-module regression: `Tests/Persona_Buddy/`, `Tests/Persona_Visual/`, and the
  five splash/startup suites — **754 passed**.
* `Tests/UI/test_persona_buddy_widget.py`, `test_persona_buddy_app_mount.py`,
  `test_library_file_notes_workspace.py` — **229 passed, 2 failed**. Both failures are
  `test_high_stakes_file_notes_states_are_legible_in_shipped_themes[size0/size1]` in a
  module this branch does not touch; reproduced identically on pristine dev `7f38cb6ef`
  in a separate worktree (`'Save failed' paints at 3.33:1, below 4.5:1`,
  `assert 3.3290074521434456 >= 4.5`).
* `./scripts/preflight.sh` green (all five checks).

### Out-of-scope findings

* `Widgets/loading_states.py`, `Widgets/status_dashboard.py`,
  `Widgets/detailed_progress.py`, `Widgets/activity_log.py` have no importer anywhere in
  the repo — four dead widget modules, each carrying its own `set_interval`. Candidates
  for retirement.
* `ConsoleVideoPreview._update_progress` repaints a `height: 1` progress line once per
  decoded video frame from `_show_frame`. It is a worker-driven path rather than a timer,
  so it is outside this census's definition, and its layout request coalesces with
  `_update_frame`'s (which legitimately lays out) in the same loop turn — so the gain
  would be nil. Noted rather than changed.

### Modified / added files

* `tldw_chatbook/Widgets/splash_screen.py`
* `tldw_chatbook/Widgets/Persona_Widgets/persona_buddy_widget.py`
* `Tests/Architecture/test_timer_path_static_update_inventory.py` (new)
* `Tests/UI/test_timer_path_layout_cost.py` (new)
* `backlog/docs/lessons-textual.md`
