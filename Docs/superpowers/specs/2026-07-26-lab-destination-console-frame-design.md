# Lab Destination Console-Style Frame Design

Date: 2026-07-26
Status: Draft (pending spec review)
Scope: The Lab destination's **shell and information architecture only**. Each mode's interior
is left to its own follow-up spec.

## Summary

The Lab destination seats three screens — Models (`llm`), Speech (`stts`), and Evals (`evals`) —
each a thin `BaseAppScreen` wrapper around a pre-Console legacy window that ignores the design
system. This spec gives the three a shared, Console-styled frame: one destination header, one mode
strip, a collapsible catalog rail, a body, and a collapsible inspector.

The frame is a **base class the three screens inherit**, not a merge of the three screens. Routes,
`TAB_*` constants, command-palette entries, and route-inventory ownership are untouched.

Each mode's legacy sidebar is lifted out of its window into the frame's rail. Every legacy body
stays exactly where it is; rebuilding those interiors is explicitly out of scope.

## Live verification

The screens were driven in the real app before this spec was written (`.claude/skills/verify`,
tmux at 235×52, scratch `TLDW_CONFIG_PATH`). Source reading alone had produced two wrong claims and
missed one whole component. What is actually on screen today:

- **Models** — bordered 5-row destination header; mode strip; a ~19-column sidebar titled
  "LLM Options" with nine rows (Llama.cpp, Llamafile, Ollama, vLLM, ONNX, Transformers, MLX-LM,
  Local Models, Download Models), one blank row between each; body shows the llama.cpp form.
- **Speech** — same header; a ~29-column sidebar titled "Speech Menu" with three rows, then a
  rule, then "Additional Features:" with three more (Voice Cloning, Speech Recognition, Audio
  Effects), then a **dependency-status panel** reporting missing optional deps. That panel renders
  clipped: its bottom border never closes and "whisper" breaks mid-word.
- **Evals** — header and mode strip render, then **nothing at all** down to the footer. Rows 10–51
  are blank.
- **The active mode's label is invisible on every Lab screen.** See the prerequisite below.

Two corrections this forced: the destination header is a **5-row bordered block**, not the single
row assumed from `$ds-destination-header-height`; and Models — not Speech — is the expensive mount.

Measured mount cost (`run_test`, 120×40, three runs):

| Widget | Mount (ms) |
|---|---|
| baseline (one `Static`) | 47.4 / 47.2 / 54.8 |
| `STTSWindow` (Speech) | 360.8 / 371.1 / 459.1 |
| `LLMManagementWindow` (Models) | 487.9 / 522.9 / **786.6** |

`LLMManagementWindow.compose` builds all nine `llm-view-*` scrolls eagerly and merely toggles their
display; `STTSWindow` mounts exactly one content widget at a time. The bigger file is the cheaper
mount. This cost is pre-existing — every visit to Models already pays it.

**Console's frame, verified.** Console renders a *blocking* setup modal (`chat_screen.py:6602` —
"keep the workbench inert") that covers the entire destination until
`provider_done = readiness.native_send_supported` (`console_onboarding_state.py:188`). Clearing it
needs a provider key from `NATIVE_CONSOLE_PROVIDER_KEYS`, which is exactly
`{"llama_cpp", "local_llamacpp"}` — **not** `"llama.cpp"`. With `provider = "llama_cpp"` pointed at
a live `llama-server`, the real frame renders:

```
DestinationHeader          Console / subtitle / [Ready]
CONTROL BAR row 1          Provider: Llama_cpp | Model: … | Assistant: General |
                           RAG: off | Sources: 0 staged | Tools: 0 ready | Approvals: 0 pending
CONTROL BAR row 2          New tab  Settings  Attach context  Run Library RAG  Save Chatbook  Help
workspace grid  [ left rail ~34 cols ][ main ][ Inspector handle ]
  left rail: title row "Console context  ◀"  then  Session ▼ / Starred ▼ /
             Workspaces ▼ / Chats ▼ / Context ▼
composer                   separate bordered block below the grid
footer                     F6 next pane | Shift+F6 previous pane | F1 help | …
```

Four things this corrected in the frame design below: Lab gains a **status row** (Console's
defining density element, which Lab had no equivalent of); the rail gains an **in-rail title and
collapse button**; the rail width is now chosen *against* Console's observed ~34 rather than
invented; and the frame **registers footer shortcuts**, which was omitted entirely.

Screenshots (SVG, 200×50) for Models, Speech, Evals, Console-blocked, and Console-frame were
captured via `App.save_screenshot` under `run_test`. The app has no in-app screenshot command, so a
driver script is required — worth keeping for future UI specs.

## Prerequisite — PR0: the invisible active-mode label

Independent of this redesign and shipping first.

The bundle carries a global, unscoped rule:

```
tldw_cli_modular.tcss:5356
.is-active {
    border: round $ds-action-focus;
    text-style: bold;
}
```

`LabModeStrip` neutralizes it in `DEFAULT_CSS` (`lab_mode_strip.py:61-73`), but **app-tier CSS beats
`DEFAULT_CSS` regardless of specificity**. The chip therefore gets a round border, becomes a 3-row
box inside a height-1 strip, and only its top border row survives:

```
row 9:   ' Modes:    Models    Speech  ╭─────────╮'     <- "Evals" is nowhere
row 10:  ''                                             <- not wrapped below
```

MCP has the structurally identical strip and renders correctly because its equivalent rule is
declared **app-tier** with higher specificity than `.is-active`:

```
tldw_cli_modular.tcss:5855
#mcp-mode-strip Button.mcp-mode-chip { ... border: none; }    /* (1,1,1) beats (0,1,0) */
```

**Fix:** PR0 creates `features/_lab.tcss` and registers it in `CSS_MODULES` (PR2 later extends the
same module), mirroring `.personas-mode-chip.is-active` at bundle 5361:

```css
#lab-mode-strip Button.lab-mode-chip { border: none; }

#lab-mode-strip .lab-mode-chip.is-active {
    border: none;
    background: $ds-focus-bg;
    color: $ds-focus-fg;
    text-style: bold underline;
}
```

**Test:** assert the active chip's *rendered label*. A test asserting the `is-active` class is
applied passes today, with the label invisible.

### Sweep of other `.is-active` consumers

| Consumer | Neutralizing rule | Verdict |
|---|---|---|
| `.nav-button.is-active` | none — the border **is** the intended affordance | OK; nav bar is 3 rows, verified live |
| `.lab-mode-chip.is-active` | none | **BROKEN**, verified live |
| `.mcp-mode-chip.is-active` | bundle 5855 | OK, verified live |
| `.personas-mode-chip.is-active` | bundle 5361 | OK |
| `ListItem.personas-library-row.is-active` | bundle 5389 | OK |
| `#mcp-hub-rail Button.mcp-rail-row.is-active` | bundle 5906 | OK |
| MCP audit-mode buttons | none — **deliberate**, documented at `mcp_audit_mode.py:209-212` | Intended |
| `.workbench-mode.is-active` | bundle 6292 | OK |
| `.library-collection-row` + `is-active` | **none anywhere** — only `library_collections_panel.py:153` and an `@on` at `library_screen.py:13326` | **RESOLVED — not the Lab defect.** Investigated live (see below) |

### `.library-collection-row` — investigated, no fix

`LibraryCollectionsPanel` was mounted with the production bundle (`App(CSS_PATH=...)`, size
`(120, 40)`), with three collections and the middle one `selected=True`. The first measurement
pass (superseded below) read `widget.styles.border` and `widget.size`. That was a mistake:
`Widget.size` is the **content** box (inside any border), not the widget's full on-screen extent,
so it reads `(16, 1)` for every row regardless of whether that row also carries a border — it
cannot distinguish a 1-row button from a 3-row bordered one. Re-measured with `widget.region`
(the full outer rectangle, border included) after two `pilot.pause()` calls past the initial
render:

```
'Alpha - 1 item'    is-active=False  border=['','','','']                      region=Region(x=0, y=2, width=16, height=1)
'Bravo - 1 item'    is-active=True   border=['round','round','round','round']  region=Region(x=0, y=3, width=18, height=3)
'Charlie - 1 item'  is-active=False  border=['','','','']                      region=Region(x=0, y=6, width=18, height=1)
```

`region.height` shows what `size` could not: the selected row is genuinely 3 rows tall on screen,
its siblings 1. Reading the actual compositor output (`screen._compositor.render_strips()`)
confirms the label survives the extra rows — `"Bravo"` is present in the rendered strip text.

The rendered result is **not** the Lab defect. Lab's mode strip is height-constrained to one
row, so the border's box clipped the label away entirely. The collections list has vertical room,
so the bordered row simply grows to three rows and its label stays fully readable — it reads as a
deliberate selection box.

The one real observation is that a selected row is three rows tall while its siblings are one, so
the list reflows as selection moves. That is a cosmetic inconsistency, not a defect, and changing
it would alter Library's appearance to fix nothing. **No change made.** If Library's own design
work later wants uniform row heights, this is where to start.

## Goals

- One Console-styled frame shared by all three Lab modes, at Console-level information density.
- Each mode's catalog moves into a collapsible rail; bodies keep working untouched.
- Keyboard access to modes that does not thrash the screen stack.
- Zero change to routes, `TAB_*` constants, palette entries, or route-inventory ownership.

## Non-goals

- Rebuilding the Models or Speech interiors. Their bodies are lifted, not rewritten.
- Any Evals interior work — that is the Evals rebuild's PR3.
- Generalizing a shared collapsible workbench (see Prior art).
- Changing which screens belong to Lab. Membership is settled: Models, Speech, Evals.

## Prior art: why the container is not shared

`DestinationWorkbench` (`Widgets/destination_workbench.py`) is a fixed `Horizontal` of equal-width
panes with no collapse. The Watchlists rebuild declined to reuse it and built
`WatchlistsWorkbench`, saying collapse "graduates into the shared widget" once a second consumer
appears.

That container now exists and is in review (`feat/watchlists-phase-b-workbench`), and it is **not
reusable here**: it is bound to a five-member `Region` enum whose `CENTRE_REGIONS` is a three-deep
vertical stack, with `solo()` semantics. Lab needs three regions, no stacked centre, no solo.
Generalizing it would mean re-typing an enum-bound state machine on a branch that has already been
through two review rounds.

Lab therefore builds its own small container and takes **knowledge, not code**:

1. **Never name a reactive `layout` on a Widget subclass.** `Widget.layout` is an existing
   unsettable Textual property the compositor calls `.arrange()` on every pass; a same-named
   reactive makes Textual call `.arrange()` on the domain object and crash. Lab's is `rail_layout`.
2. **Keep the collapse state machine pure and separate from the widget**, so it is testable without
   mounting anything.
3. **Persist collapse to config**, not to screen state.

Once both containers have shipped and settled, a later spec can generalize against two real
consumers instead of one speculative one.

## Architecture

New base `LabScreen(BaseAppScreen)` in `tldw_chatbook/UI/Screens/lab_frame.py`. `LLMScreen`,
`STTSScreen`, and `EvalsScreen` inherit it.

```
DestinationHeader          5 rows   bordered block: title / subtitle / status
LabStatusRow               1 row    mode-supplied status chips  (Console parity)
LabModeStrip               1 row    Models | Speech | Evals
LabWorkbench               1fr      Lab-private, three regions
 ┌──────┬──────────────┬───────────────────────┬────────────┬──────┐
 │handle│  LEFT RAIL   │        BODY           │ INSPECTOR  │handle│
 │ w=13 │  title + ◀   │        bench          │ collapsed  │ w=11 │
 │      │  catalog     │        1fr            │ by default │      │
 │      │  width 26    │                       │            │      │
 └──────┴──────────────┴───────────────────────┴────────────┴──────┘
```

Console pairs its status row with a second **action** row. Lab does not copy that: the mode strip
already occupies that slot, and Lab's per-mode actions live in the body today. One status row is
the parity worth taking; a second action row would be chrome without a consumer.

`$ds-lab-rail-width: 26` against Console's observed ~34: Console's rail holds conversation titles,
Lab's holds fixed short labels whose longest is `Speech Recognition` at 18 characters.

### Hooks

| Hook | Default | Purpose |
|---|---|---|
| `lab_header_state()` | abstract | `WorkbenchHeaderState` for the header |
| `lab_status_chips()` | `()` — row not rendered | status chips for this mode |
| `compose_lab_rail()` | empty rail | the catalog |
| `compose_lab_body()` | abstract | the bench |
| `compose_lab_inspector()` | honest empty-state panel | the output |
| `lab_footer_shortcuts()` | frame defaults | registered via `register_footer_shortcuts` |

The frame also renders the rail's own title row and collapse button, mirroring Console's
`Console context ◀`; modes supply only the title string.

The frame owns collapse state, handle visibility, `$ds-*` tokens, framed-region borders, the
density class, and the width contract. It owns nothing about any mode's content.

### Width contract

```
rail 26  +  body >= 63  +  collapsed inspector handle 11   =  100   guaranteed
rail 26  +  body 44     +  expanded inspector 30           =  100   NOT guaranteed
```

Both rails open at 100 columns is explicitly not a guarantee — Console scopes its own contract to
the collapsed handle the same way (`chat_screen.py:7195-7196`). `$ds-lab-rail-width: 26` is sized to
the longest rail label, `Speech Recognition` (18 chars), plus padding and frame border;
`$ds-lab-inspector-width: 30` is the expanded inspector.

### `on_mount` hazard

`STTSScreen.on_mount` (`stts_screen.py:50`) overrides `BaseAppScreen.on_mount` **without calling
`super()`**. The frame therefore puts no required setup in `on_mount`; it exposes an explicit hook
invoked from one place. `STTSScreen`'s override is additionally fixed to call `super()` as part of
this work.

### Bindings

Textual 8.2.7 sets `_inherit_bindings = True`; a subclass's `BINDINGS` **merges** with the base
rather than replacing it (verified by direct probe). `EvalsScreen`'s bare-digit `1..6` card
bindings and the frame's `[` / `]` coexist without conflict.

## Mode switching

### Preview, then commit

`_create_navigation_screen` (`app.py:5508-5530`) mandates a **fresh screen instance for every
navigation**; caching one caused an exception-free full-UI freeze root-caused 2026-07-11. Immediate
bracket cycling would therefore construct and throw away a ~360–790 ms screen per keypress.

So brackets do not navigate. **`[` / `]` move focus to the adjacent mode chip**, and nothing else:

- preview is Textual's native `:focus` styling, already in the bundle (`Button:focus`, line 954)
- `Enter` is ordinary `Button` activation on the focused chip, which already posts
  `NavigateToScreen` via `lab_mode_strip.py:101-108`
- `Tab` is ordinary focus movement away from the strip. `Escape` is deliberately **not** given a
  strip meaning: `EvalsScreen` already binds it to `action_evals_back` (`evals_screen.py:31`), and
  a competing strip binding would shadow that on one of the three modes only.

Cycling Models → Evals builds **zero** intermediate screens. No in-flight guard, no debounce timer,
no race with the fresh-screen rule, and no new state machine, CSS class, or cancel path — focus is
the preview.

Two consequences worth stating plainly. Brackets are printable keys, so text inputs consume them
first; they act only from button or list focus, exactly as Personas documents at
`personas_screen.py:237-239`. And `Enter` on the *already active* chip is a deliberate no-op —
`_handle_mode_chip` returns without posting when the route matches.

An earlier draft of this spec had `[` / `]` set a separate preview state committed by a
screen-level `Enter` binding. That does not work: if `[` fired at all, focus was on a button or
list, so `Enter` is consumed by that focused widget and activates it instead of committing. The
preview would have been enterable but not committable.

### Lazy body mount

The frame composes header, mode strip, and rail, then constructs and mounts the body from
`call_after_refresh` so first paint is not blocked. This improves *all* navigation into Models, not
only cycling — the ~0.5–0.8 s moves after first paint rather than before it. It does not reduce
total work.

**Deferred construction breaks every caller that assumes the body already exists**, and there are
five today:

| Site | Method | Failure without a fix |
|---|---|---|
| `stts_screen.py:55` | `on_mount` | `self.stts_window or self.query_one(STTSWindow)` — attribute is `None` and the query raises `NoMatches`, on **every** visit to Speech |
| `stts_screen.py:67` | `on_screen_suspend` | same pattern |
| `stts_screen.py:77` | `on_screen_resume` | same pattern |
| `evals_screen.py:58` | `action_evals_back` (Escape) | bare `query_one(EvalsWindowV3)` before the body mounts |
| `evals_screen.py:69` | `action_evals_open` (digits `1..6`) | same |

The frame therefore fires an explicit **`on_lab_body_ready()`** hook once the body is mounted.
Screen initialization that touches the body moves into that hook, and the two Evals actions guard
on the body being present rather than assuming it. This is the same hazard as the `on_mount`
override above and is fixed the same way: no required work runs where the body may not exist yet.

## Rail contents and lift seams

### Models

Nine rows lifted from `.llm-nav-button`, in two sections replacing the current flat "LLM Options"
stack:

```
Local servers                Models
  Llama.cpp                    Local Models
  Llamafile                    Download Models
  Ollama
  vLLM
  ONNX
  Transformers
  MLX-LM
```

`handle_nav_button` (`LLM_Management_Window.py:963`) sets the `active_view` reactive and
`watch_active_view` (`:982`) does the work. Once the buttons are rail siblings, `Button.Pressed`
bubbles to the **screen**, not the window, so the `@on` never fires. `LLMScreen` catches the rail
press and sets `window.active_view`; the reactive and its watcher are otherwise unchanged.

**The trap is silent.** `watch_active_view` also calls `self.query(".llm-nav-button")` to move the
`-active` class. With the buttons gone, `query()` returns an **empty set rather than raising** — the
body still switches correctly while selection highlighting silently stops tracking. The rail owns
its own active styling, driven by the screen.

### Speech

Four modes and one action, per the IA decision below, plus the capability panel:

```
Modes                        Actions
  Playground                   Voice Cloning ↗   (pushes its own screen)
  Settings
  AudioBook                  [ dependency status panel ]
  Speech Recognition
```

Three hazards, all distinct:

1. **It crashes rather than degrades.** `watch_current_view` (`STTS_Window.py:5007-5013`) calls
   `self.query_one("#view-playground-btn", Button)`. `query_one` **raises `NoMatches`** — so unlike
   Models' quiet no-op, Speech throws on every view switch until retargeted.
2. **`on_button_pressed` is dual-purpose and the second purpose is load-bearing.** Beyond the
   sidebar branches, its `else` (`:5036-5044`) manually forwards unhandled presses into the active
   content widget. Stripping the sidebar branches carelessly takes the delegation with them,
   breaking buttons *inside* the playground, settings, and audiobook widgets.
3. **The capability panel becomes a status chip, not a rail region.** `#speech-capability-status` is
   a status block, not a mode or an action, and today it renders **clipped** — the screenshot shows
   it cut off mid-list at `local_tts,` with its bottom border never drawn. Relocating it into the
   frame's status row (`lab_status_chips()`) fixes the clipping by removing the constraint that
   caused it, rather than patching the box: a one-line chip such as `Local speech: deps missing`
   with the full detail on hover, matching how Console renders `Tools: 0 ready`. This is the
   concrete payoff of the Console status-row parity noted above.

**IA decisions:**

- **`view-effects-btn` is removed.** It is `disabled=True` and notifies "Audio Effects coming
  soon!" — dead chrome, confirmed on screen.
- **`view-voice-cloning-btn` moves from a mode to an action.** It calls
  `app.push_screen(VoiceCloningWindow())`, leaving the Lab frame entirely; it is marked as leaving
  rather than presented as a sibling of the four in-frame modes. Re-hosting it inside the frame is
  interior work and out of scope.

Both are user-visible removals and belong in release notes.

### Evals

Empty rail with the honest empty state.

**Evals renders nothing today because `EvalsWindowV3` composes a `Screen` as a child widget.**
`evals_window_v3.py:51-52` does `self.current_screen = EvalNavigationScreen(...)` then
`yield self.current_screen`, and `EvalNavigationScreen` is a `Screen` subclass
(`eval_nav_screen.py:40`; MRO `[EvalNavigationScreen, Screen, Generic, Widget, DOMNode]`). Screens
belong on the app's screen stack, not in the widget tree. Every entry in `_create_screen` is
likewise a `Screen`, and `go_back` / `reset_to_home` `self.mount(...)` them the same way.

So there is genuinely no parity to preserve, and **PR3 must rebuild rather than adapt** — the whole
`EvalsWindowV3` navigation model is invalid, not merely unstyled. **PR3 fills `compose_lab_rail()`
and `compose_lab_body()` rather than authoring its own frame**; this must be agreed before PR3
starts or the two designs collide at merge.

The card hub itself holds **no** database references, so this is not a data problem. Separately,
implementers on this branch will hit
`Evals_DB.SchemaError: Database version 4 is newer than supported version 3` — the concurrent
word-bench branch left a v4 DB at the user-scoped `evals.db` (`eval_orchestrator.py:94`). That is
pre-existing, unrelated to the blank render, and must not be mistaken for a regression introduced
here.

### Selection styling

Legacy sidebars signal selection with Textual's `variant="primary"`. The rail uses the `is-active`
class treatment, so selection reads consistently with the mode strip — and, per PR0, with an
app-tier rule rather than `DEFAULT_CSS`.

## Rail widget promotion

`ConsoleRailHandle` and `ConsoleRailSectionHeader` already have **six** consumers — `chat_screen`,
`home_screen`, `library_screen`, `personas_screen`, `Widgets/Home/home_rail`,
`Widgets/Library/library_rail` — while living in a Console-private namespace. The handle also
imports `CONSOLE_RAIL_INSPECTOR_LABEL` from `tldw_chatbook.Chat.console_rail_state` (a widget
reaching into the Chat layer) and hard-codes Console's badge vocabulary (`"1 approval"` → `"1 appr"`,
`"artifact"` → `"art"`). `ConsoleRailSectionHeader` is nearly generic — its only coupling is two
glyph constants.

```
Widgets/destination_rail.py              NEW, pure
    DestinationRailHandle                no Chat imports; caller supplies
    DestinationRailSectionHeader         tooltips and display strings

Widgets/Console/console_rail_handle.py   STAYS
    ConsoleRailHandle(DestinationRailHandle)
        applies Console's abbreviations; keeps the Chat-layer import
```

**All six existing consumers change nothing, and the CSS bundle sees zero diff.** The TCSS contains
no type selectors for these widgets (verified) — only class selectors — so new Python type names are
invisible to CSS, and the `.console-rail-*` class names are deliberately kept. Renaming those classes
is a deferred cleanup, not part of this work.

## CSS, tokens, persistence

### `LabRailStore`

Rail collapse is a preference, not data, so it goes to config.

- Section `lab`, key `collapsed_rails`, flat. **Not** because dotted sections fail — that bug is
  fixed (`config.py:3905-3918` walks the nested tree; a `[lab.rails]` write/read round-trip was
  verified end to end) — but because flat is simpler and matches the sibling store.
- **Default sentinel is `None`, never `[]`.** `get_cli_setting` returns `default` only when the key
  is absent (verified), so `None` distinguishes "never set" from "user explicitly expanded
  everything". Collapsing that distinction re-imposes the first-run default forever.
- **Persisted:** rail collapse only. Not the active mode (that is the route), not the selected view
  within a mode.
- **First run:** left rail open, inspector collapsed — the rail is the mode's primary navigation and
  earns its width; the inspector is an empty state and should not spend columns saying so.

### CSS

New `features/_lab.tcss`, appended to `CSS_MODULES` in `build_css.py` — an ordered list, so it lands
after `components/` and before `utilities/` (which override anything). The bundle
`tldw_cli_modular.tcss` is generated and rebuilt at boot; regenerate via `build_css`, never
hand-edit.

- Colors, borders, and status treatments go in the **app-tier bundle, not `DEFAULT_CSS`** — the
  bundle outranks `DEFAULT_CSS` regardless of specificity. PR0 exists because this rule was broken.
- `LabWorkbench.DEFAULT_CSS` carries **only structural guards** (height/min-height) so harness apps
  that do not load the bundle still lay out, mirroring `lab_mode_strip.py:46-74`.

### Tokens

`$ds-lab-rail-width: 26` and `$ds-lab-inspector-width: 30` in `core/_variables.tcss`, alongside the
existing `$ds-library-source-browser-width` precedent.

## Testing

Every test below is **mutation-checked**: revert the fix, confirm the test goes red. Two of the
three seams fail in ways a naive test sails past.

### Lift seams

| Test | Catches |
|---|---|
| exactly one rail row carries `is-active`, and it is the row just clicked | Models' silent highlight death — a test asserting "clicking Ollama shows the Ollama view" passes while it is broken |
| switching across all four Speech views raises nothing | the `query_one` → `NoMatches` crash |
| a button owned by the mounted content widget is still handled after the sidebar leaves | the load-bearing delegation fallback |
| the capability status renders as a one-line chip with full detail on hover | the clipped panel seen live |
| a mode returning no status chips renders no status row at all | dead chrome on modes without status |
| the footer carries this destination's shortcuts | the omitted `register_footer_shortcuts` wiring |

### Frame contract

Mirroring `test_console_workbench_contract.py`:

| Test | Proves |
|---|---|
| geometry at width 100: rail 26 + collapsed handle 11 + body ≥ 63 | the width contract |
| rail collapse round-trips config **and survives a mode switch** | why collapse lives in config, not `save_state` |
| `[` / `]` move focus to the adjacent chip and navigate nothing; `Enter` on a focused chip commits | zero intermediate mounts |
| `on_lab_body_ready()` fires after the body mounts, and Speech's init runs there | the five deferred-body call sites |
| Escape and digits `1..6` on Evals are safe before the body mounts | `action_evals_back` / `action_evals_open` raising `NoMatches` |
| `EvalsScreen` has bare digits `1..6` **and** `[` / `]` | binding merge across the MRO |
| the active mode chip's **rendered label** is present | PR0's bug, which a class-only assertion misses |
| empty inspector renders the honest empty state | not a blank box |

### No-regression

Must pass untouched through **every** PR: `test_destination_shells.py` (its `SCREEN_BY_ROUTE`
suite) · `test_workbench_route_inventory.py` · `test_command_palette_shell_routes.py` ·
`test_evals_screen_shell.py`

`test_lab_mode_strip.py` must pass untouched through **PR0 and PR1**. PR2 deliberately changes the
strip's keyboard behavior, so that suite is legitimately *extended* there — do not treat its
needing changes in PR2 as a regression signal, and do not weaken it to keep it green.

And critically `test_console_rail_sections.py` plus the Console/Personas/Home/Library rail tests.
**Those passing with zero edits is the proof that subclassing beat migrating.** If they need
changes, the promotion approach failed and we should know immediately.

### Live verification

Unit tests are exactly where geometry contracts lie. Closing evidence is driving the real TUI at
**100 columns** and at default size, walking all three modes and both rails, with a capture per
mode.

## Risks

1. **Models' ~0.5–0.8 s mount is pre-existing and not fixed here.** Lazy mounting defers it past
   first paint; it does not remove it. The real fix is making `LLMManagementWindow` stop composing
   all nine views eagerly — interior work, out of scope, worth filing.
2. **`STTSScreen.on_mount` gaining `super()`** is a real behavior change, though the base only logs
   today.
3. **Evals PR3 coordination** must be agreed before PR3 starts.
4. **Speech's two IA removals are user-visible.**
5. **`.library-collection-row` is an unresolved suspect** from the PR0 sweep, deliberately not fixed
   blind.

## Sequencing

**Every PR must leave the app in a shippable state.** An earlier draft split the frame from the
rail lifts — PR2 landed the frame on all three screens, PR3 lifted Models' sidebar, PR4 lifted
Speech's. That intermediate state is not shippable: after PR2 the frame renders an empty left rail
(first run opens it) while each legacy sidebar is still alive inside its body, so Models and Speech
would each show **two navigation columns side by side** — worse than today.

Each screen's sidebar lift is therefore folded into that screen's own adoption PR. A screen adopts
the frame and fills its rail in the same change, or it does not adopt yet.

| PR | Contents | Shippable on its own |
|---|---|---|
| **PR0** ✅ | creates `features/_lab.tcss`, registers it in `CSS_MODULES`, regenerates the checked-in bundle via `build_css`; app-tier `.lab-mode-chip.is-active` rule + focus-disambiguation guard + rendered-label test | yes — the active mode label becomes visible |
| **PR1** ✅ | `Widgets/destination_rail.py` pure base; `ConsoleRailHandle` becomes a subclass; zero consumer edits, zero bundle content diff | yes — pure refactor, no visual change |
| **PR2** | `LabScreen` frame, `LabWorkbench`, `LabRailLayout`, `LabRailStore`, status row, focus-based mode keys, lazy body mount + `on_lab_body_ready()`, footer shortcuts — **and Models adopts it, lifting its nine-row sidebar into the rail** | yes — Models is fully good; Speech and Evals unchanged |
| **PR3** | Speech adopts the frame, lifts its sidebar, and moves the capability panel into a status chip; the two IA removals land here | yes |
| **PR4** | Evals adopts the frame with an empty rail and the honest empty state | yes |

PR0 and PR1 shipped together as PR #940. PR2's tasks should be written against PR1's realised API
rather than an anticipated one.

Note that PR2 carries the frame *and* one screen's lift, making it the largest of the five. If it
proves too big in planning, the split to reach for is by frame region (frame + rail, then status
row + inspector), never frame-without-rail — that is the unshippable seam this section exists to
prevent.
