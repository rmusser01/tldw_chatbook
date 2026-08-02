# Screen Decomposition — Design

**Status:** draft for review — 2026-08-02, revision 2
**Scope:** `chat_screen.py`, `settings_screen.py`, `library_screen.py`

## Purpose

Three screens carry 52,079 lines between them, each dominated by a single class with
500–620 methods. They are the hardest files in the codebase to change safely, and they
are where this project's recurring geometry defects keep appearing. This design defines
how they come apart.

The immediate trigger is navigation work — jump mode (task-1951) and border key
hints (task-1950) both need to reason about a screen's regions, which is
impractical while a screen *is* one object with 567 methods. But the decomposition is
worth doing on its own terms.

## The measured problem

| | `chat_screen.py` | `settings_screen.py` | `library_screen.py` |
|---|---|---|---|
| File lines | 20,338 | 15,922 | 15,819 |
| Dominant class | `ChatScreen` | `SettingsScreen` | `LibraryScreen` |
| Class lines | 18,779 (92%) | 14,329 (90%) | 14,973 (95%) |
| Methods | **567** | **623** | **508** |
| `__init__` attributes | 27 | 33 | 19 |
| Distinct `self.*` names | 605 | 547 | 439 |
| Largest method | `compose_content` 681 | `_render_provider_detail` 530 | `compose_content` 381 |

Two facts shape everything below.

**The state surface is small.** Nineteen to thirty-three instance attributes across
classes of five hundred–plus methods. Of `ChatScreen`'s 605 distinct `self.*` names,
roughly 578 are *methods calling other methods*. This is a call-graph problem, not a
shared-mutable-state swamp — which is why extraction is feasible rather than hopeless.

**The bloat has three different flavours**, and a single technique will not address all
of them:

- `chat_screen.py` — event dispatch and orchestration. `on_button_pressed` is 368 lines.
- `settings_screen.py` — per-category rendering. `_render_provider_detail` (530),
  `_render_detail_pane` (470), `_render_library_rag_editor_fields` (368).
- `library_screen.py` — composition and data snapshots. `compose_content` (381),
  `_list_local_source_snapshot` (263).

## What has already been tried, and why the screens still grew

Extraction has happened at scale: 27 modules under `Chat/console_*.py` (25,313 lines)
and 38 under `Widgets/Console/` (17,693 lines) — roughly 43,000 lines already pulled out
of the Console's orbit.

It worked, and it did not shrink the Screen. What got extracted was **logic and leaf
widgets**. What stayed is **Textual glue**: event handlers, action methods, and
sync/refresh orchestration. A screen that delegates its logic still owns every wire.

The lesson this design takes from that: extracting *what a feature computes* is not
enough. The unit that has to move is *what a feature owns* — its region of the DOM, its
events, and its state, together.

## The design: two collaborator kinds, one rule for choosing

**A region widget owns pixels. A controller does not.**

That is the whole rule. It is decidable for every cluster in all three screens, and both
kinds already exist in this codebase — `Widgets/Console/` holds 58 classes, 38 of which
define their own `compose()` and 32 of which handle their own `on_*` events. This design
applies an established pattern at a larger grain; it does not introduce architecture.

**The existence proof is the Evals screen.** `evals_screen.py` is 2,513 lines — the one
healthy screen among the five largest — precisely because its regions live in
`UI/Evals/` as widgets (`library_rail.py`, `inspector.py`, the editors) and its
read-side state lives in a view model (`evals_state.py`). Its screen keeps layout,
routing, and cross-region coordination, and nothing else. This design is "make the
other three screens shaped like the Evals screen", stated as rules.

### Where the code lives

Each screen gets a package mirroring `UI/Evals/` and the existing `UI/*_Modules/`
convention: `UI/Console_Modules/`, `UI/Settings_Modules/`, `UI/Library_Modules/`.
Region widgets and controllers both live there — one home per screen, so a reader finds
a screen's collaborators in one place. Existing leaf widgets in `Widgets/Console/` stay
where they are: they are reusable parts, and a region is a one-place-only composition
of them. The screen module itself keeps its current path — imports across the repo
reference it, and moving it is churn with no decoupling value.

### Region widgets

A region widget is a compound `Widget` that:

- composes its own subtree, so `compose_content` becomes a layout skeleton that yields
  regions rather than 681 lines of inline construction;
- handles its own events with `@on(...)`, so the within-region share of
  `on_button_pressed`'s 368-line dispatch evaporates rather than being relocated; what
  cannot evaporate — a press whose effect crosses regions — becomes a typed message the
  screen handles, visible in the type system instead of buried in an if-chain;
- adopts `RecomposeCaptureGuard` if it ever recomposes — the house convention (seven
  existing subclasses) for not orphaning mouse capture across its own teardown;
- posts messages upward for anything the screen must coordinate, rather than being
  called downward;
- owns the CSS for its own subtree in `css/features/`.

The screen keeps layout, cross-region coordination, and the Textual lifecycle.

### Controllers

A controller is a plain object that owns state and behaviour with no region of its own.
It holds the state it is responsible for, takes a screen handle for the framework
services it genuinely needs (`query_one`, `run_worker`, `call_after_refresh`), and is
constructed once in `__init__`.

`self.app_instance` is `ChatScreen`'s single most-referenced name — 317 references,
for database handles, config, and notifications. Collaborators do not traverse to it:
whatever a controller needs from the app is passed at construction, named, so a
controller's dependencies are its signature. A region widget receives data and posts
messages; where it genuinely needs an app service, the screen passes that service in,
never `app_instance` wholesale.

Workers a controller starts run under a group named for that controller, so
`exclusive=True` scopes to its own work rather than to whichever collaborator last
used the screen's node.

A controller that finds itself calling `query_one` for more than a handful of well-known
ids is a region widget wearing the wrong hat. That is the review signal.

### Why not controllers alone

Controllers alone were the first proposal and are rejected. They leave `compose_content`
and `on_button_pressed` standing, because a controller with a screen handle still needs
the screen to compose and to receive the button event. They do nothing for
`settings_screen.py`, whose bloat is rendering. And a controller that reaches back
through `query_one` has the same coupling as before, now hidden behind a facade —
visible coupling is better than concealed coupling.

### Why not region widgets alone

Non-visual orchestration — session lifecycle, workspace context, agent run bookkeeping —
would have to become widgets owning no DOM. That is a worse fit than a plain object, and
it would put state behind a mount lifecycle that does not need one.

### Bindings and focus

Screen-level `BINDINGS` keep working throughout, because ids and action methods stay on
the screen until the behaviour they trigger moves. When a region takes ownership of a
behaviour, its binding moves onto the region widget, where Textual resolves it while
focus is inside the region; a chord that must work regardless of focus stays on the
screen and delegates. Nothing here waits on config-driven keybindings (task-1952) — but
every binding that moves lands in one obvious place, the region, so task-1952 has named
homes to bind into and jump mode (task-1951) can treat "the regions of the current
screen" as its target list.

## Decomposition targets

These are the clusters the analysis found, with the rule applied. Counts are method
lines within the dominant class; they size the work, they are not extraction targets to
hit exactly.

### `chat_screen.py`

The DOM already names its regions — `console-shell` contains `console-left-rail`,
`console-main-column`, `console-context-rail`, `console-inspector-rail`,
`console-control-bar`, `console-mode-bar` — but `compose_content` builds them inline,
yielding twelve raw `Static(...)` calls. The regions exist conceptually and have no code
of their own. That is the gap.

| Cluster | Lines / methods | Kind |
|---|---|---|
| rail + inspector | 958 / 42 | region widgets (left, context, inspector rails) |
| composer | 377 / 18 | region widget |
| workspace | 1,382 / 40 | controller |
| session | 916 / 31 | controller |
| message | 1,027 / 23 | controller |
| dictation | 742 / 20 | controller |
| agent | 612 / 15 | controller + a region widget for its rail section |
| character | 708 / 14 | controller + a region widget for its rail section |
| image / attachment | 1,031 / 27 | controller |

The `sync` cluster — 1,347 lines across 31 methods, the largest verb group in the class
— splits along the same rule: a sync that repaints one region becomes that region's own
refresh; a sync that coordinates several regions stays on the screen. That residue is
why the success criterion below is ownership rather than a line count: the screen
legitimately keeps cross-region coordination, and it will not end up small.

### `settings_screen.py`

Bloat is concentrated in per-category detail rendering. Each category's detail pane
becomes a region widget; the screen keeps category selection. The save path is the
interface that decides whether this screen fits the pattern: today
`action_settings_save_category` (517 lines) reads every field back out of the rendered
pane by id. After decomposition each pane owns "collect what I am showing" — returning
its edited values as one mapping — and a save controller owns validation and
persistence. A pane that cannot say what it holds without the screen reaching into it
is drawn at the wrong boundary; that is the review signal for this screen.

### `library_screen.py`

`compose_content` (381) splits along the same region lines as the Console.
`_list_local_source_snapshot` (263) and its siblings are data access with no region and
become a controller.

## Migration safety

This is the section that matters most, because this refactor walks directly through this
project's worst-defect territory. Moving a region changes DOM structure and CSS
selectors, and the recurring failures here have all been geometry: a control pushed out
of reach three separate times; a bare `Container` defaulting to `height: 1fr` and
starving its sibling; pane scroll theming left on the wrong node while its contract test
passed against stale rules; a checkbox no user could toggle while 867 tests passed.

Rules, all non-negotiable:

1. **One region per change.** A change moves exactly one region or extracts exactly one
   controller. No batch moves.
2. **Ids are preserved verbatim.** A region widget composes the same ids in the same
   nesting. If an id must change, that is its own change with its own review, never a
   passenger on an extraction.
3. **Painted-geometry assertions before and after.** Every extraction carries a test
   asserting the moved region's controls are hit-testable —
   `screen.get_widget_at(*control.region.center)` resolves to the control — at 160x45
   **and** 235x52. Where such a test already exists for the region, it must pass
   unchanged; where none exists, it is written *before* the move, against the current
   code, so it is proven to pass before it is relied upon.
4. **CSS moves with its region**, into `css/features/`, and the bundle is regenerated via
   `build_css.py`. The bundle is never hand-edited.
5. **Behaviour changes are forbidden in an extraction.** An extraction that also fixes a
   bug is two changes; do the fix separately, before or after, so a regression has one
   candidate cause.
6. **A characterisation test precedes any extraction whose behaviour is not already
   covered.** Extracting untested behaviour is how silent regressions ship.

## Testing

- Existing tests that drive the screen through the DOM — `pilot` clicks, key presses,
  id queries — must pass unchanged. They are the regression net; one that needs editing
  is a signal the extraction changed behaviour.
- Existing tests that reach into private methods will break when the method moves. That
  is mechanical, not behavioural: retarget the call and keep the assertion
  byte-for-byte. An assertion that has to *change* is a finding about the extraction,
  never a test to accommodate.
- Each new region widget gets its own test file, mounting it in a minimal host app and
  driving it through `pilot` — real clicks and keypresses, asserting persisted results
  rather than widget state.
- Each new controller gets unit tests with no Textual mount.
- The geometry assertions in rule 3 above.
- After each screen completes, its full suite runs serially (`-p no:randomly`), because
  this repo's parallel runs produce cross-test interference that is not a branch signal.

## Order and delivery

One spec — this document — covering the pattern and validating it against all three
screens. Then **one implementation plan per screen**, each landing independently:

1. **Console** (`chat_screen.py`) — first, because the navigation work that motivated
   this targets it, and because its regions are already named in the DOM.
2. **Settings** (`settings_screen.py`) — second; its per-category panes are the cleanest
   test of whether the region-widget contract generalises beyond the Console's layout.
3. **Library** (`library_screen.py`) — third.

A screen's plan is not written until the previous screen's is merged, so each plan can
carry what the last one learned.

Success for a screen is that its dominant class no longer holds regions or feature state
— not a line-count target. A line count is a proxy that invites splitting a file without
decoupling anything.

## Not in scope

- **Mixins.** Splitting a class body across files by feature was considered and rejected:
  it reduces file size while leaving every collaborator able to see all 605 `self.*`
  names, so nothing becomes independently testable. It is filing, not decoupling.
- **Behaviour and UX changes.** This design moves code. Jump mode, border key hints, and
  every other Bagels-derived idea are separate work that becomes easier afterwards.
- **The other 44 screens.** Three screens account for 61% of `UI/Screens/`. The rest are
  small enough not to need this, and the convention this establishes is documented in
  `DESIGN.md` for whichever grows into it next.
- **A rewrite.** Every step preserves behaviour and ids. There is no point at which the
  Console is rebuilt.
