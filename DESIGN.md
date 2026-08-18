---
name: tldw_chatbook
description: Local-first agentic knowledge console with a terminal-native, cyberpunk-cozy product system.
colors:
  canvas: "$background"
  panel: "$panel"
  surface: "$surface"
  raised-surface: "$surface"
  field-surface: "$surface-darken-1"
  text-primary: "$text"
  text-muted: "$text-muted"
  text-disabled: "$text-disabled"
  action-primary: "$primary"
  action-secondary: "$secondary"
  focus-accent: "$accent"
  ready-success: "$success"
  warning-amber: "$warning"
  blocked-error: "$error"
  grid-line: "$surface-lighten-1"
typography:
  display:
    fontFamily: "terminal emulator monospace"
    fontSize: "1 terminal cell"
    fontWeight: 700
    lineHeight: 1
    letterSpacing: "normal"
  headline:
    fontFamily: "terminal emulator monospace"
    fontSize: "1 terminal cell"
    fontWeight: 700
    lineHeight: 1
    letterSpacing: "normal"
  title:
    fontFamily: "terminal emulator monospace"
    fontSize: "1 terminal cell"
    fontWeight: 700
    lineHeight: 1
    letterSpacing: "normal"
  body:
    fontFamily: "terminal emulator monospace"
    fontSize: "1 terminal cell"
    fontWeight: 400
    lineHeight: 1
    letterSpacing: "normal"
  label:
    fontFamily: "terminal emulator monospace"
    fontSize: "1 terminal cell"
    fontWeight: 700
    lineHeight: 1
    letterSpacing: "normal"
rounded:
  none: "none"
  terminal-round: "round"
  terminal-tall: "tall"
  terminal-heavy: "heavy"
spacing:
  cell-0: "0 cells"
  cell-1: "1 cell"
  cell-2: "2 cells"
  cell-3: "3 cells"
  cell-5: "5 cells"
components:
  button-primary:
    backgroundColor: "{colors.action-primary}"
    textColor: "{colors.text-primary}"
    rounded: "{rounded.none}"
    padding: "0 1 cell"
    height: "3 cells"
  button-primary-focus:
    backgroundColor: "{colors.action-primary}"
    textColor: "{colors.text-primary}"
    rounded: "{rounded.none}"
    padding: "0 1 cell"
    height: "3 cells"
  field-input:
    backgroundColor: "{colors.field-surface}"
    textColor: "{colors.text-primary}"
    rounded: "{rounded.terminal-round}"
    padding: "0 1 cell"
    height: "3 cells"
  destination-header:
    backgroundColor: "{colors.panel}"
    textColor: "{colors.text-primary}"
    rounded: "{rounded.terminal-tall}"
    padding: "1 2 cells"
  panel:
    backgroundColor: "{colors.panel}"
    textColor: "{colors.text-primary}"
    rounded: "{rounded.terminal-round}"
    padding: "1 2 cells"
  status-badge:
    backgroundColor: "{colors.raised-surface}"
    textColor: "{colors.text-primary}"
    rounded: "{rounded.none}"
    padding: "0 1 cell"
    height: "1 cell"
---

# Design System: tldw_chatbook

## 1. Overview

**Creative North Star: "The Neon Workbench"**

This system is a dense terminal workbench for controlled agentic work: cyberpunk in atmosphere, efficient in layout, effective in state exposure, and cozy enough to keep users inside long-running workflows. The interface should feel like a trusted local control room, not a decorative command-line costume. Color and borders exist to reveal state, focus, authority, and recovery.

The visual model is themeable Textual UI. Semantic variables such as `$background`, `$panel`, `$surface`, `$primary`, `$accent`, `$success`, `$warning`, and `$error` are the source of truth, with `ds-*` aliases documenting product-level roles. Future work should preserve the compact screen grammar: global destination navigation, destination header, local mode bar, primary list or queue, main workspace, optional inspector, and footer status.

Reject generic chatbot surfaces, study-only framing, SaaS dashboard tropes, marketing-card layouts, vague "AI assistant" language, hidden recovery states, and interfaces that require log reading to understand status. Console is the live work surface; other destinations prepare, inspect, organize, configure, or hand off work.

**Key Characteristics:**

- Terminal-native density with readable labels.
- Themeable semantic tokens, not one hardcoded palette.
- Status and source authority visible before action.
- Flat by default, structured by borders, panels, and tonal layers.
- Keyboard-first focus with no layout shifts on hover or focus.

## 2. Colors

The palette is semantic and restrained: dark terminal surfaces by default, bright action and state colors used only when they explain control, authority, readiness, or recovery.

### Primary

- **Signal Primary** (`$primary`): primary actions, active controls, strong selected rows, and execution affordances.
- **Focus Phosphor** (`$accent`): focus outlines, active structure, compact section emphasis, and selected input borders.

### Secondary

- **Secondary Circuit** (`$secondary`): secondary action roles and source-role differentiation when `$primary` would overstate importance.

### Tertiary

- **Workspace Glow** (`$ds-authority-workspace`, aliased to `$accent`): workspace-scoped authority, contextual source roles, and staged handoff state.

### Neutral

- **Deep Canvas** (`$background`): root terminal canvas and major unused space.
- **Console Panel** (`$panel`): headers, footers, primary panel backgrounds, and stable control surfaces.
- **Raised Surface** (`$surface`, `$surface-lighten-1`): cards, inputs, collapsible headers, toolbars, and list rows.
- **Grid Line** (`$surface-lighten-1`, `$surface-lighten-2`): panel borders, dividers, table lines, and structural separators.
- **Readable Text** (`$text`): default foreground.
- **Dim Telemetry** (`$text-muted`): metadata, footer hints, secondary help, and inactive controls.
- **Disabled Ghost** (`$text-disabled`): disabled action labels only.

### Named Rules

**The Semantic First Rule.** Never choose a color because it looks cyberpunk. Choose the token that names the state: focus, ready, running, warning, approval required, blocked, error, workspace, server, local, dry-run, synced, or conflict.

**The Rare Neon Rule.** Bright accent color is earned by state or action. It must not become decoration, background wash, or page mood.

**The Legible Disabled Rule (TASK-1801).** A disabled control's label must render at **at least 3:1 against its own background**, measured in a running terminal — not inferred from token names. Disabled is a state to *read*, not a state to *guess at*: several surfaces communicate a restriction only through the disabled control's own label ("writes a file to disk — not available in a temporary chat"), so an illegible disabled label silently voids that explanation.

Two dimmers compound here and neither is visible in the stylesheet:

1. the theme sets `text-disabled: auto 38%` — alpha over the panel, ~3.4:1 on its own;
2. Textual's `Button:disabled` adds `text-style: bold dim` *and* `color: auto 50%` (`textual/widgets/_button.py`), roughly halving the result again.

Together these put **all 58 shipped themes below 3:1**, including `high_contrast_yellow_black`. Measured live, the composer menu's disabled rows rendered at **1.05:1 and 1.25:1**.

Two consequences for anyone touching disabled styling:

- **`text-style: none` does not clear Textual's `dim`.** Verified by measuring the running app; a rule that relies on it will still render at roughly half the colour it declares. State the colour bright enough to survive the halving.
- **Widget `DEFAULT_CSS` cannot override it.** Both a screen's `DEFAULT_CSS` and `Button`'s sit in the same tier, where `Button` wins for a `Button`. Disabled overrides belong in the app stylesheet.

A third layer exists on some surfaces: the shared Workbench action bar also carries `.is-disabled { opacity: 0.55 }`, stacking **three** dimmers and measuring 1.45:1. Note the class — that bar uses `is-disabled`, not `console-action-disabled`; editing the wrong one measures no change at all.

Measured results after applying this rule:

| surface | before | after |
| --- | --- | --- |
| composer menu rows | 1.25:1 | 4.80:1 |
| Workbench action bar | 1.45:1 | 6.74:1 |

Disabled must still read as clearly dimmer than enabled — enabled controls measure 10.6:1 to 12.6:1, so these targets leave the states obviously distinct.

## 3. Typography

**Display Font:** terminal emulator monospace
**Body Font:** terminal emulator monospace
**Label/Mono Font:** terminal emulator monospace

**Character:** The type system is intentionally mono-forward because Textual renders inside terminal cells. Hierarchy comes from weight, casing restraint, labels, borders, and region placement, not from display fonts or large type.

### Hierarchy

- **Display** (bold, 1 terminal cell, line-height 1): rare destination title use inside headers and splash-adjacent surfaces.
- **Headline** (bold, 1 terminal cell, line-height 1): destination headers, panel titles, and selected work summaries.
- **Title** (bold, 1 terminal cell, line-height 1): section titles, collapsible headers, list group labels, and modal titles.
- **Body** (regular, 1 terminal cell, line-height 1): transcript text, field help, descriptions, and list content. Prose should wrap before it becomes difficult to scan, usually around 65 to 75 characters when a region is prose-heavy.
- **Label** (bold, 1 terminal cell, normal letter spacing): badges, status labels, authority chips, shortcuts, and button labels.

### Named Rules

**The Cell Discipline Rule.** Do not simulate web typography inside Textual. Use one-cell rhythm, bold weight, concise copy, and region hierarchy.

**The Label Before Color Rule.** Every important colored state needs readable text. Color supports recognition; labels carry meaning.

## 4. Elevation

This system does not use shadow elevation as a primary depth cue. Depth is conveyed through Textual layers: `$background` canvas, `$panel` containers, `$surface` controls, `round` or `tall` borders, compact dividers, and visible focus outlines. Lift is structural, not atmospheric.

### Named Rules

**The Flat Control Room Rule.** Surfaces are flat at rest. Use tonal layering, borders, and labeled state instead of drop shadows, blur, or glass effects.

**The Border Has A Job Rule.** Borders define regions, focus, approval, recovery, or source authority. Decorative borders are forbidden.

## 5. Components

### Buttons

- **Shape:** borderless rectangular Textual buttons by default (`border: none`), with fixed terminal-cell height for stable layout.
- **Primary:** `$primary` background, `$text` foreground, usually `height: 3` and `padding: 0 1`.
- **Hover / Focus:** hover changes background only; focus uses `outline: heavy $accent`. Hover and focus must not change dimensions.
- **Secondary / Ghost / Tertiary:** use `$surface-darken-1`, `$surface`, or `$surface-lighten-2` backgrounds with `$text-muted` for lower priority actions.

### Chips

- **Style:** one-line badges and source-role chips use compact `padding: 0 1`, `$ds-surface-raised` backgrounds, and readable labels.
- **State:** selected, active, ready, warning, blocked, local, server, workspace, dry-run, and source-role chips must use semantic status or authority tokens.

### Cards / Containers

- **Corner Style:** Textual `round` borders for panels and inspectors; `tall` borders only for high-level destination headers.
- **Background:** `$panel` for durable regions, `$surface` or `$boost` for active working surfaces.
- **Shadow Strategy:** no shadows. Use tonal layer changes and borders.
- **Border:** `$ds-grid-line` for normal panels; `$accent`, `$warning`, `$error`, or `$success` only when the panel state requires it.
- **Internal Padding:** `1 2` for destination headers and panels, `1` for inspectors and dense inner regions.

### Inputs / Fields

- **Style:** `height: 3`, `width: 100%`, solid `$primary` border or round `$surface-lighten-1` border, with `padding: 0 1`.
- **Focus:** border or outline shifts to `$accent`; background may use `$accent 10%` for focus visibility.
- **Error / Disabled:** errors use `$error` border plus `$error 10%` background. Disabled controls use lowered opacity, `$surface-darken-1`, and `$text-disabled`.

### Dense-form control convention (one-row fields)

Dense workbench forms (Settings and its widgets) cannot afford full
borders: a Textual border costs a row above and below, tripling a
one-row field. The Console composer joined the convention in
task-17651 — it renders as a one-row bar (growing to four with the
draft) whose left edge recolors for the has-draft state and flips to
the thick focus accent, and the Console workbench frame now closes at
the workspace grid's single bottom border. The convention for these
fields (task-1586):

- **Rest:** every editable field carries a one-column left edge
  (`border-left: solid $ds-control-edge`). The edge's *presence* is the
  carrier — a structural marker separating controls from prose — so
  color is reinforcement, never the sole signal. Muted at rest.
- **Focus:** the edge flips to `thick $ds-action-focus` and the
  background swaps to `$ds-focus-bg` (the task-345 focus surface) with
  bold text. Three concurrent signals: edge weight, background, weight.
- **Toggles and switches:** always paired with a text-state word
  ("On"/"Off", "Enabled"/"Disabled") — the word is the state.
- **Inert actions:** disabled buttons carry a text annotation for *why*
  ("— no changes"), never dimming alone.
- **No underline on fields:** underlined placeholders read as links
  (task-185).
- **Persistence badge:** the State bar leads with the category's save
  model in the same position everywhere ("Draft — save with s" /
  "Applies immediately" / "Auto-saved" / "Managed in editor" /
  "Per-item Save/Reset" / "Validate, then Save" / "Read-only here") —
  five save models coexist on the Settings screen, and the badge is what
  keeps their differing footer keys from reading as inconsistency
  (task-1717).
- **Fold indicator:** scrollable inspector columns reserve a bottom row
  ("▼ more — scroll…") shown only while content overflows, so a
  mid-sentence clip is never the only signal that more exists
  (task-1623).
- **Pinned contract row:** the State bar (badge + scope) is pinned
  between the pane title and the scrollable body, never inside it — the
  save contract must not scroll away while the user is acting on the
  category (task-1716).
- **Field-level search:** "/" indexes field labels as well as category
  names and owned config keys; a field hit echoes "Category › Field" and
  Enter lands focus ON the field, which also fires its guidance
  (task-1715).
- **Inert destructive actions** carry their reason in the label
  ("Delete — built-in") and are disabled, never merely red (task-1643).
- **Voice carriers (task-1625):** the three pane titles are the one
  Focus Phosphor accent on the screen; the State bar sits on the focus
  steel at low alpha; toasts carry a full severity-tinted round border,
  never a side stripe.

### Navigation

- **Style:** top navigation and tab links are compact, theme-aware, and text-labeled. Home and Console remain reachable at supported widths.
- **Default:** `$panel` background and `$text-muted` labels.
- **Hover:** `$panel-lighten-1` background and `$text`.
- **Active:** `$accent` background or `$accent` text with bold weight, depending on whether the nav is button-based or link-based.
- **Overflow:** use command palette or explicit compact hints, never hidden mystery navigation.

### Destination Header

The destination header is a product contract, not decoration. It carries title, one-line purpose, readiness, authority, primary action, and blocked recovery when needed. It uses `$ds-surface-panel`, `border: tall $ds-action-focus`, `padding: 1 2`, and bold text.

### Recovery Callout

Recovery callouts name owner, problem, impact, and next action. They use `$warning` or `$error` state tokens with a tinted background, but the text must remain explicit enough to work without color.

### Console Transcript

Console transcript messages use compact role/body grammar and full-width terminal rules. Unselected messages stay quiet; selected messages reveal contextual actions. Tool, approval, recovery, stopped, and failed turns follow the same flow instead of becoming separate visual languages.

## 6. Do's and Don'ts

### Do:

- **Do** use Textual semantic tokens and `ds-*` aliases for new product UI.
- **Do** keep Console as the live work surface; destinations prepare, inspect, organize, configure, or hand off work.
- **Do** expose local, server, workspace, remote-only, dry-run, syncing, synced, conflict, ready, blocked, approval required, and unavailable states as readable labels.
- **Do** preserve keyboard focus with `outline: heavy $accent` or the theme-aware equivalent.
- **Do** keep hover and focus states dimensionally stable.
- **Do** use skeleton states, explicit empty states, and recovery callouts instead of silent disabled controls.
- **Do** use compact panels and inspectors when they help scan dense work.

### Don't:

- **Don't** make Chatbook feel like a generic chatbot, a study-only app, a file manager, or a decorative terminal skin.
- **Don't** use SaaS dashboard tropes, marketing-card layouts, vague "AI assistant" language, hidden recovery states, or interfaces that require reading logs to understand status.
- **Don't** collapse Personas, Skills, MCP, ACP, Schedules, Workflows, Library, Study, and Workspaces into one undifferentiated "agents" bucket.
- **Don't** turn every destination into a live agent console.
- **Don't** use side-stripe accent borders on cards, list items, callouts, or alerts. If a colored state is needed, use a full border, background tint, status badge, or explicit icon/label.
- **Don't** use gradient text, decorative glass blur, bouncy motion, or full-saturation accents on inactive states.
- **Don't** hide why an action is disabled. Give the recovery path in the surface, tooltip, inspector, or command palette.

## 7. Screen decomposition

This system also governs how a large screen's *code* is shaped, not just how it
renders — because the recurring geometry defects this file's other sections
guard against (a control pushed out of reach, a bare `Container` starving its
sibling, theming left on the wrong node) keep originating in screens that own
too much undifferentiated DOM and behaviour inside one class. A screen with
regions is a screen where those defects have somewhere narrower to hide.

### Named Rules

**The One Rule.** A region widget owns pixels. A controller does not. That
is the whole test, decidable per cluster: if it composes its own subtree and
handles its own `on_*` events, it is a region widget; if it owns state and
behaviour with no DOM of its own, it is a controller.

**The One Home Rule.** A screen's collaborators — region widgets and
controllers alike — live in one package next to the screen:
`UI/<Screen>_Modules/` (for example `UI/Console_Modules/`,
`UI/Settings_Modules/`, `UI/Library_Modules/`). Existing reusable leaf
widgets stay where they already are (`Widgets/Console/` and siblings); a
region is a one-place composition of them, not a relocation of them.

**The Six Migration Rules.** Every extraction — moving one region or one
controller out of a screen — follows six non-negotiable rules, stated in
full in `Docs/superpowers/specs/2026-08-02-screen-decomposition-design.md`
under "Migration safety." They are not restated here; read them there
before touching a screen's DOM.

**Existence proofs.** Two, so far: the Evals screen (`evals_screen.py`,
2,513 lines, the one healthy screen among the five largest — its regions
live in `UI/Evals/`: `library_rail.py`, `inspector.py`, the editors, with
read-side state in `evals_state.py`), and Console wave 1 (`UI/Console_Modules/`:
the shared frame helper, `ConsoleLeftRail`, `ConsoleInspectorRail`, and
`ConsoleDictationController`).

**Naming a controller's dependencies.** A controller's constructor is its
dependency list — see `ConsoleDictationController.__init__`
(`tldw_chatbook/UI/Console_Modules/dictation.py`) as the canonical example.
Its docstring settles three kinds of binding, one rule per kind:

1. Framework services the controller genuinely needs (`query_one`,
   `run_worker`, `post_message`, `set_timer`, `set_interval`, `is_mounted`)
   live-read from the screen through a `@property` on every access, never
   snapshotted — a value captured once at construction goes stale the
   instant a caller or test replaces the attribute on the screen instance
   afterward.
2. Everything else the controller depends on that is not its own state is a
   named, keyword-only constructor parameter, passed as a late-binding
   callable — a lambda closing over the screen at *call* time, not a bound
   method frozen at construction. This is what "a controller's dependencies
   are its signature" means in practice: discoverable by reading the
   constructor, not by reading every property on the class.
3. `app_instance` may be stored as a plain attribute — the one snapshot
   exception — only where its identity is stable for the controller's life
   and that stability is justified in the docstring, the way a screen
   method's identity is not. This is a per-dependency exception, never a
   default; a controller that snapshots something whose identity *can*
   change under it is repeating the staleness bug wave 1 found and fixed
   twice before landing on this rule.

**A region widget never stores a removable child-widget instance.** When a
screen hands a region widget a piece of its content that the screen may
later remove and replace OUTSIDE that region's own `compose()` — by
`query_one`-ing straight into the region's DOM, the way
`_apply_console_live_work_card_swap` and
`_render_character_avatar_into_section` do for `ConsoleInspectorRail`'s
live-work card and `ConsoleLeftRail`'s character avatar — the region must
not store that content as a bare widget INSTANCE and re-yield it from
`compose()`. Nothing keeps the stored reference in sync with the screen's
own remove/remount, so it is safe only as long as nothing ever calls
`compose()` again; the moment a `reactive(..., recompose=True)` is added to
that region, its `compose()` re-yields a widget Textual has already
removed from the DOM — reappearing stale content (e.g. the pre-swap
source-readiness card) instead of raising. Pass a zero-arg builder
callable instead (preferred — the same late-binding shape as a
controller's constructor dependencies above; see `ConsoleInspectorRail`'s
`live_work_card_builder` and `ConsoleLeftRail`'s
`character_avatar_widget_builder` in `UI/Console_Modules/right_rail.py` /
`left_rail.py`) so every `compose()` call mounts a fresh instance built
from current state, or have the region re-query its own DOM instead of
caching a reference at all.

**A region owns its behaviour, not its children's API — so querying a child
widget by id is not a boundary violation.** Wave 3 extracted
`ConsoleTranscriptRegion` and left the screen reaching the transcript's DOM two
different ways, which looked like an unfinished migration. It is not, and the
rule is worth stating so nobody "finishes" it. The region defines exactly three
public behaviours — `capture_reading_state`, `restore_reading_state`,
`note_follow_intent` — all of them about the *viewport*: scroll position and
follow intent, which belong to the region because the region is what scrolls.
Three screen methods reach the region for those. The other eight reach past it
with `query_one("#console-native-transcript", ConsoleTranscript)` because they
need the transcript widget's *own* API — appending a message, clearing a
selection — which the region does not own and should not proxy.

Routing those eight through the region would mean eight pass-through getters
whose whole body is `return self._transcript_or_none()`. That converts an owner
into a façade, adds a layer with no invariant behind it, and buys nothing: in
Textual, `query_one` is transparent across compound-widget boundaries by design,
so an id lookup that crosses into a region is idiomatic rather than a leak. The
test is **ownership, not reachability** — ask "whose invariant is this?", and
only route through the region when the answer is the region's.

**Controller-to-controller traffic goes through named callables the screen
wires at construction — never a back-door through screen attributes.** Wave 2
put three controllers on one screen, and clusters genuinely reference each
other: sessions live inside workspace context, and dictation drives the
hands-free loop. The temptation is for one controller to reach the other by
reading a screen attribute (`self._screen._session`, or a live property that
proxies to it), which reintroduces exactly the coupling the extraction removed
and hides it behind indirection. Instead the screen owns the wiring: it builds
both controllers, then passes each the specific callables the other exposes
(`ConsoleSessionController`'s five seam callables into
`ConsoleWorkspaceController`, and vice versa; the six that replaced
`ConsoleDictationController`'s hands-free reach-backs). Two consequences worth
stating because both cost a review round to establish. First, the lambdas must
resolve the sibling at CALL time (`lambda: self._session.x()`), never capture
it at construction, or whichever controller is built second is `None` at
wiring time. Second, if a proxy property is write-only, its getter must raise
`RuntimeError` and NOT `AttributeError` — `hasattr()` and
`getattr(obj, name, default)` swallow `AttributeError` specifically, so a
defensive read would silently take the default forever with nothing raised
anywhere.

**A proxy property standing in for a plain attribute must be read-WRITE.**
The rule above governs the write-only case; wave 3 found its mirror image and
paid for it. Moving state into a controller turns what was
`self._console_original_attempt_previews = {}` in `__init__` into a `@property`
on the screen that forwards to the controller — and a `@property` with only a
getter makes the name **assignment-hostile**, where the plain attribute it
replaced accepted writes from anywhere. Wave 3 shipped one getter-only proxy
and turned **41 tests red in a file the branch never touched**, all of them
`AttributeError: property '…' of 'ChatScreen' object has no setter`. Two
things follow. First, check the *baseline* shape before writing a proxy: if it
was a plain assignable attribute, the proxy needs a setter, and "nothing writes
it today" is a coverage claim rather than a contract — write the setter anyway.
Second, the setter must write THROUGH to the controller (`self._message.x = v`),
never to a shadow attribute the controller never reads; that variant is worse
than the crash, because the tests go green over a dead write. An `ast` audit
comparing every new property against the baseline attribute's shape catches
both, and is cheap enough to run every wave.

**Controller construction lives in `UI/Console_Modules/wiring.py`, not in the
screen's `__init__`.** Wiring six controllers with named callable dependencies
is verbose by design — the verbosity IS the dependency list — and by wave 4 it
had grown `ChatScreen.__init__` to 782 lines, 411 of them construction. Those
statements now live in `build_console_controllers(screen)`, which the screen
calls at the point the first construction occupied. Two rules come with it.
First, **the constructions move verbatim**: every named keyword argument stays
character for character, because a controller's dependencies are its signature
and collapsing them into a per-controller dependency object would hide exactly
what the binding rule above exists to expose. Second, **the call site's position
matters** — the screen sets ~250 attributes around it, and the late-binding
lambdas read them — so a new controller is added to `wiring.py`, never back into
`__init__`.

**An import an extraction leaves behind may be load-bearing for tests.** When
code moves out of a module, the imports it used often become code-unused, and
deleting them looks like tidying. It is not always: tests patch symbols through
whatever module namespace they can reach, and
`monkeypatch.setattr(chat_screen_module, "ConsoleDictationController", ...)`
keeps a `chat_screen`-level import alive with **no reference an import-grep can
find**. Wave 4 deleted five such imports and turned 28 tests red across five
files. Before removing an import an extraction orphaned, grep the test suite for
the SYMBOL as an attribute of the module you are editing, not just for the name.
An import kept for this reason carries `# noqa: F401` and a comment saying so,
because otherwise the next `ruff --fix` harvests it.

**New Console code goes in `UI/Console_Modules/`, and a ratchet enforces it.**
The decomposition's first two waves extracted ~4,900 lines out of
`chat_screen.py` and the file still ended up *larger* than when the work
started, because ~5,500 lines of concurrent feature work landed in it over the
same window. Extraction cannot outrun growth, so the screen's size is now a
ceiling: `Tests/Architecture/test_screen_size_ratchet.py` holds a line and
method budget that may only ever be lowered. A wave lowers it in the same PR
that earns the reduction; a feature that would raise it belongs in a module
instead. The test's failure message names the module directory and this
section, because the moment it fires is the moment someone needs to know where
else to put the code — not a link to chase later.
