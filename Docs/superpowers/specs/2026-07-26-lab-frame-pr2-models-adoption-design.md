# Lab Frame PR2 — Frame plus Models Adoption Design

Date: 2026-07-26
Status: Draft (pending spec review)
Parent: [Lab destination Console-style frame](2026-07-26-lab-destination-console-frame-design.md)
Depends on: PR #940 (PR0 + PR1), merged to `dev`

## Summary

PR2 builds the shared `LabScreen` frame and has **Models adopt it in the same change**, lifting its
nine-row sidebar into the frame's rail. Speech and Evals are untouched and adopt in PR3 and PR4.

The parent spec's sequencing rule governs: a screen adopts the frame and fills its rail in one
change, or it does not adopt yet. Landing the frame with an empty rail beside Models' surviving
legacy sidebar would show two navigation columns — worse than today.

## What PR0/PR1 landed, verbatim

PR2 builds on the API as it actually shipped, not as it was anticipated:

```python
# tldw_chatbook/Widgets/destination_rail.py
class DestinationRailHandle(Vertical):
    def __init__(self, *, label, badge="", button_id, badge_id, side,
                 open_tooltip=None, **kwargs) -> None
    @property
    def open_tooltip(self) -> str      # resolved on READ, tracks label
    def sync_state(self, label, badge) -> None
    def _display_label(self) -> str    # documented override seam
    def _display_badge(self) -> str    # documented override seam

class DestinationRailSectionHeader(Horizontal):
    def __init__(self, title, *, section_id, open, **kwargs) -> None
    def sync_open(self, open: bool) -> None

RAIL_SECTION_TOGGLE_PREFIX = "console-rail-section-toggle-"
GLYPH_EXPANDED = "▾"
GLYPH_COLLAPSED = "▸"
```

`open_tooltip` became a read-resolved property during PR #940 review: it had been captured in
`__init__`, so `sync_state` renaming a rail left the tooltip naming the previous one. Lab uses the
derived path, so this matters here.

`features/_lab.tcss` exists and is registered in `CSS_MODULES`. PR2 **extends that same module**; it
does not add another.

## File structure

Screen classes live in `UI/Screens/`; `*_Modules/` packages hold panes and helpers (8 such packages
exist: `MCP_Modules`, `Watchlists_Modules`, `Persona_Modules`, …). Lab follows that split.

| File | Responsibility | Rough size |
|---|---|---|
| `UI/Screens/lab_frame.py` | `LabScreen(BaseAppScreen)` — the shared base | ~200 |
| `UI/Lab_Modules/lab_workbench.py` | `LabWorkbench` container: rail \| body \| inspector | ~150 |
| `UI/Lab_Modules/lab_rail_layout.py` | `LabRailLayout` — pure collapse state, no widget | ~60 |
| `UI/Lab_Modules/lab_rail_store.py` | config load/save of collapse state | ~70 |
| `UI/Lab_Modules/lab_server_status.py` | pure reader over the six app `Popen` handles | ~90 |
| `UI/Screens/llm_screen.py` | modify — adopts the frame, supplies rail/body/status | |
| `UI/LLM_Management_Window.py` | modify — remove nav buttons, trim the orphaned watcher block | |
| `css/features/_lab.tcss` | extend — rail, workbench, status row, rail-row `is-active` | |

## Frame anatomy

```
DestinationHeader        5 rows   title / subtitle / status
LabStatusRow             1 row    mode-supplied chips; NOT RENDERED when a mode supplies none
LabModeStrip             1 row    Models | Speech | Evals
LabWorkbench             1fr
 [handle 13][ RAIL 26 ][ BODY 1fr ][ INSPECTOR 30 ][handle 11]
```

### Hooks

| Hook | Shape | Called |
|---|---|---|
| `lab_header_state()` | returns `WorkbenchHeaderState` | compose |
| `lab_status_chips()` | **re-callable**, returns `tuple[LabStatusChip, ...]` | compose **and** every refresh |
| `compose_lab_rail()` | generator | compose |
| `build_lab_body()` | **factory returning a Widget** | after first paint |
| `compose_lab_inspector()` | generator | compose |
| `LAB_FOOTER_SHORTCUTS` (class constant) | `((key, label), …)` | mount |
| `on_lab_body_ready()` | callback | after the body mounts |

`build_lab_body` is a factory rather than a `ComposeResult` generator for two reasons: the body is
mounted after first paint, and Watchlists established that widget *instances* do not survive
`recompose=True` while factories do. The rail and inspector stay inline generators — they are cheap
and not deferred.

A chip is a two-field frozen dataclass declared beside the frame:

```python
@dataclass(frozen=True)
class LabStatusChip:
    chip_id: str   # stable DOM id suffix; identifies the Static across refreshes
    text: str      # rendered copy, e.g. "Servers: 2 running"
```

`lab_status_chips()` must be safe to call repeatedly. The frame builds one `Static` per `chip_id` on
compose and **mutates it via `.update(text)`** on refresh, matching chips by `chip_id`. It never
recomposes the status row: recomposing on a timer churns widgets and can steal focus. A chip set
whose `chip_id`s change between calls is a programming error — the frame logs and ignores unknown
ids rather than silently mounting new widgets on a 2-second timer.

**As built, this is a class constant `LAB_FOOTER_SHORTCUTS`, not a hook method.** No Lab mode has
mode-specific shortcuts, and an overridable hook nobody overrides is dead API; a mode that later
needs its own overrides the constant, which is the same extension point with less machinery.

The constant holds `((key, label), …)` pairs for
`BaseAppScreen.register_footer_shortcuts`. The frame supplies the mode-navigation defaults
(`[` / `]` cycle, `Enter` commit); Models adds nothing of its own in PR2, so it does not override
the hook.

`LabScreen.__init__` keeps `BaseAppScreen`'s `(app_instance, screen_name)` signature and needs no
extra parameter: all three Lab screens already pass their route as `screen_name` (`"llm"`, `"stts"`,
`"evals"`), so the frame derives the mode strip's `active_route` from `self.screen_name`.

### A hook hazard PR2 creates for PR3

The frame depends on `on_screen_resume` for its modal-pop refresh. `STTSScreen.on_screen_resume`
(`stts_screen.py:72`) **overrides without calling `super()`** — so when Speech adopts the frame in
PR3, it will silently kill that refresh. This is the same defect class the parent spec records for
`on_mount`, on a second hook, and it is recorded here because PR2 is what introduces the dependency.

**As built, the frame did NOT achieve this.** `LLMScreen` defines `on_screen_resume` as a public
override (`llm_screen.py`), so the hazard is live rather than designed around: when Speech adopts in
PR3 and keeps its own `super()`-less `on_screen_resume`, the refresh dies silently.

PR3 must therefore do two things as part of adoption: fix `STTSScreen`'s two `super()`-less
overrides (`on_screen_suspend` at `:61`, `on_screen_resume` at `:72`), **and** move the frame's
resume work behind a private method the subclasses call, so the next adopter cannot repeat it.

Also note the refresh this protects is close to redundant: Textual does not pause a screen's timers
on suspend, so the 2-second poll keeps running under a modal anyway.

### Carried constraints

- `LabRailLayout` is a frozen dataclass, separate from the widget, so collapse logic is testable
  without mounting anything.
- The reactive is named **`rail_layout`, never `layout`** — `Widget.layout` is an unsettable Textual
  property the compositor calls `.arrange()` on every pass; shadowing it crashes the compositor.
- Rail collapse persists to **config**, not `save_state`: `_create_navigation_screen`
  (`app.py:5508-5530`) mandates a fresh screen instance per navigation, so screen-scoped state does
  not survive a mode switch.
- Width contract: rail 26 + body ≥ 63 + collapsed inspector handle 11 fits 100 columns. Both rails
  open at 100 columns is explicitly not guaranteed, matching Console.

## Models adoption

### The rail

Nine rows lifted from `.llm-nav-button`, in two sections replacing the flat "LLM Options" stack:

```
Local servers              Models
  Llama.cpp                  Local Models
  Llamafile                  Download Models
  Ollama
  vLLM
  ONNX
  Transformers
  MLX-LM
```

Each row carries its view key as an **attribute** — `button.lab_view_key = "llama-cpp"` — rather than
encoding it in the id. `Button` has no `__slots__` (verified), and this mirrors
`library_collections_panel.py:156`'s `button.collection_id = …`. It avoids string surgery on ids.

The nine keys are exactly `LLMManagementWindow.view_mapping`'s keys (`:245-255`).

**The two section headings are static labels, not collapsible sections.** PR1's
`DestinationRailSectionHeader` is available and Console uses it, but seven rows and two rows do not
need collapsing, and making them collapsible immediately raises whether that state persists —
which, given a fresh screen per navigation, would mean extending `LabRailStore` for no user benefit.
A later mode with a long rail can adopt the section header then.

### The seam

`LLMManagementWindow.active_view` (`reactive("llama-cpp", recompose=False)`, `:239`) stays the single
source of truth. `watch_active_view` (`:982`) swaps view visibility by toggling `.llm-view.-active`;
that half is untouched.

```
rail press ──→ LLMScreen sets window.active_view = key
                        │
window.active_view ─────┼──→ watch_active_view      → body view visibility (unchanged)
                        └──→ LLMScreen._sync_rail_active → rail is-active styling
```

**The screen watches the reactive; it does not style on press.**
`DOMNode.watch(obj, attribute_name, callback, init=True)` (verified signature) works across widgets
and is already used at `evals_window_v3.py:59`. Two consequences:

- `LLMManagementWindow.on_mount` sets `active_view = "llama-cpp"` itself (`:269`). A press-only
  handler would leave the rail unhighlighted on arrival; watching covers both origins.
- `init=True` fires the callback immediately on registration, so the rail seeds itself. No separate
  seeding step.

The watch is registered in `on_lab_body_ready()`, since the window does not exist before then.

### The silent trap

Once the buttons are rail siblings, `Button.Pressed` bubbles to the **screen**, not the window, so
the window's `@on(Button.Pressed, ".llm-nav-button")` never fires. And `watch_active_view`'s
`self.query(".llm-nav-button")` returns an **empty set rather than raising** — the body still
switches correctly while selection highlighting silently dies. A test asserting "clicking Ollama
shows the Ollama view" passes straight through this.

### Trimming the orphaned block

`watch_active_view`'s nav-button lines are dead only because this change moved those buttons out,
and its `query_one(f"#nav-{new_view}")` sits in a `try/except QueryError` that **logs a warning on
every view switch**. PR2 deletes those lines and leaves the view-visibility loop and
`_populate_help_text` alone — verified self-sufficient.

No code outside `LLM_Management_Window.py` references the `nav-*` ids or `.llm-nav-button`, except
dead CSS in `Constants.py:966-976`, which is left in place (see Non-goals).

## Status row and inspector

Both read the six `app.*_server_process` handles (`app.py:3582-3587`) through the pure
`lab_server_status.py`, using the codebase's existing liveness idiom
`proc and proc.poll() is None` (`llm_management_events.py:1028-1030`, `mlx_lm.py:169`).

```
STATUS ROW   Servers: 2 running

INSPECTOR    Running servers
               ● llama.cpp    running
               ● ollama       running
               ○ llamafile    stopped
               ○ vLLM         stopped
               ○ ONNX         stopped
               ○ MLX-LM       stopped
```

**No `View:` chip.** The rail already highlights the active view a few columns away; Console's chips
earn their place by surfacing what is not otherwise visible.

**No port numbers.** Ports live in each view's port `Input`, not on the `Popen` handle. Reading them
would couple the inspector to view internals for little gain; running/stopped is the useful part.

### Refresh

```
on_lab_body_ready()   initial paint
set_interval(2.0)     every subsequent change — launches, stops, and crashes alike
on_screen_resume      a modal popping back over a live Lab screen
```

**There is deliberately no refresh-on-press trigger.** Pressing `llamacpp-start-server-button` does
not synchronously create the process — the event handler assigns `app.llamacpp_server_process` from
an async worker afterwards, so a press-triggered refresh reads pre-launch state and renders
"stopped". That is worse than a short lag. The timer catches the transition within 2 seconds.

`on_screen_resume` is nearly redundant for navigation, since `switch_screen` posts `ScreenResume`
(verified) to a screen that is already fresh. It earns its place only for modal push/pop over a live
Lab screen.

The timer dies with the screen: navigation unmounts the outgoing instance. It keeps ticking while a
modal sits over the screen — six non-blocking `waitpid` calls, no I/O, no database.

The refresh entry point is a **directly callable method** and the interval is a named constant, so
tests drive the refresh and assert on the `Static`s rather than sleeping on wall-clock.

## CSS

`features/_lab.tcss` gains rail, workbench, status-row, and rail-row rules.

**The global `.is-active` rule reaches the rail rows, and it breaks them.** `tldw_cli_modular.tcss`
declares an unscoped `.is-active { border: round $ds-action-focus; }`, and app-tier CSS beats widget
`DEFAULT_CSS` regardless of specificity — the defect PR0 fixed for the mode chips. Measured on rail
rows:

| Row height | Result with `is-active` |
|---|---|
| `height: 1` (a dense rail) | `border='round'`, **`region.height == 2`** — a half-bordered artifact that displaces its neighbours |
| `height: auto` | `border='round'`, `region.height == 3` — grows, as Library's collection rows do |

A dense rail sets `height: 1`, so without an app-tier `.lab-rail-row.is-active { border: none; … }`
rule the rail is visibly broken the moment anything is selected. This is the third consumer of that
global rule to need explicit neutralizing; PR0's sweep predicted it.

**Why `is-active` at all, rather than dodging the global rule?** Using the legacy `-active` name
would avoid it for free. It is still the wrong choice: `is-active` is the established rail-row
convention — `#mcp-hub-rail Button.mcp-rail-row.is-active` (bundle `:6314`) is the same widget shape
that needed the same neutralizing rule — and `.workbench-mode`, `.personas-mode-chip`, and
`ListItem.personas-library-row` all follow it. Diverging would buy one avoided CSS block and cost
the design system's only selection convention.

**Do not unify the two active-class names in this screen.** The `llm-view-*` bodies are shown and
hidden by `.llm-view.-active`, which `watch_active_view` toggles; the rail rows use `is-active`.
They look like an inconsistency worth tidying and are not: renaming the views' class breaks
visibility, and renaming the rail's opts out of the design system and its `is-active` styling.

All colours and borders go app-tier, never in widget `DEFAULT_CSS`. The bundle is regenerated via
`build_css.py` and never hand-edited.

## Testing

Every test is **mutation-checked**: revert the fix, confirm red, restore.

### The silent-failure pair

| Test | Catches |
|---|---|
| exactly one rail row carries `is-active`, and it is the row just pressed | the silent highlight death — the obvious body-switch test passes without it |
| the rail highlights correctly **on arrival, before any press** | a press-only implementation, which `on_mount`'s own `active_view` assignment defeats |

### Frame contract

| Test | Proves |
|---|---|
| geometry at width 100: rail 26 + collapsed handle 11 + body ≥ 63 | the width contract |
| rail collapse round-trips config **and survives a mode switch** | why collapse lives in config, not `save_state` |
| `[` / `]` move focus to the adjacent chip and navigate nothing; `Enter` commits | zero intermediate screen mounts |
| a mode supplying no chips renders **no status row at all** (driven by a test-double screen, since Models always supplies one) | the empty-chips path, which has no real consumer until PR3/PR4 and would otherwise ship unexercised |
| the body is **absent at first paint**, present after the deferred mount | the lazy-mount claim itself — without this the frame could mount inline and every other test still passes |
| `on_lab_body_ready()` fires after the body mounts | the deferred-body call sites |
| selected rail row has no border **and** all nine rows are `region.height == 1` | the global `.is-active` rule; a border-only assertion misses a height regression |
| the frame wires `active_route` into the mode strip correctly | frame-level wiring — the existing strip suite mounts it standalone and would not notice |
| `lab_server_status` against a fake carrying six attributes | chips and inspector, with no subprocesses and no mounting |
| refresh mutates the `Static`s rather than recomposing the row | timer churn and focus theft |

### No-regression

`test_lab_mode_strip.py` · `test_destination_rail.py` · `test_destination_shells.py` ·
`test_workbench_route_inventory.py` · `test_command_palette_shell_routes.py` ·
`test_evals_screen_shell.py` · `test_stts_capability_state.py`

Known pre-existing failures on `dev` that must stay unchanged:
`test_console_persistent_rails.py::test_generated_console_stylesheet_includes_rail_rules` (1), and
`test_library_shell.py`'s 3 deterministic failures plus 2 documented CPU-contention flakes. **Gate on
failure names, never a raw count.**

### Live verification

Drive the real TUI at **100 columns** and at default size: all three Lab modes, both rails collapsed
and expanded, and a server started and stopped to watch the chip and inspector track it. Screenshot
per mode. Unit tests assert geometry; only the running app confirms it.

## Risks

1. **`lab_frame.py` is the single-file risk** — header, status row, mode strip, workbench, collapse,
   lazy mount, and footer in one class. If it passes ~250 lines during implementation, split the
   lazy-mount / `on_lab_body_ready` machinery into its own module; it is the least coupled piece.
2. **Models' 488–787 ms mount is deferred, not removed.** `LLMManagementWindow.compose` builds all
   nine `llm-view-*` scrolls eagerly and merely toggles their display. Lazy mounting moves that past
   first paint. The real fix belongs with whoever rebuilds Models' interior.
3. **Transitional asymmetry, deliberate.** Only Models inherits `LabScreen` in PR2; Speech and Evals
   keep mounting `LabModeStrip` themselves under `BaseAppScreen`. Both paths work — the strip is a
   shared widget — but bracket-key mode switching exists only on Models until PR3/PR4.
4. **The 2 s timer is the only mechanism** for server-state change, so Start shows up to 2 s later.
   Accepted: the alternative reported pre-launch state as "stopped".

## Known gaps, recorded not solved

**Below 100 columns is unspecified.** Console carries density classes; Lab inherits none, and this
spec's width contract only guarantees the 100-column case with the inspector collapsed. Narrower
terminals will render *something* — the rails have `min-width` — but nothing here defines what.
Worth a pass once three modes exist and there is a real layout to degrade.

**Returning to Models always lands on Llama.cpp.** `LLMManagementWindow.on_mount` sets
`active_view = "llama-cpp"` unconditionally (`:269`), and screens are rebuilt per navigation, so the
last-viewed provider is never restored. The frame could persist it alongside rail collapse, but the
reset lives in the window, and changing it is interior behaviour this PR does not touch.

## Non-goals

- No Speech or Evals adoption — PR3 and PR4.
- No rebuild of any `llm-view-*` body. The nine views are lifted around, not rewritten.
- No port numbers in the inspector.
- No removal of the orphaned `.llm-nav-pane .llm-nav-button` CSS in `Constants.py:966-976`.
- No change to `LLMManagementWindow` beyond removing the nav buttons from `compose` and trimming the
  orphaned block in `watch_active_view`.
