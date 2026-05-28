---
name: tldw_chatbook
description: Local-first agentic knowledge console with a terminal-native, cyberpunk-cozy product system.
colors:
  canvas: "$background"
  panel: "$panel"
  surface: "$surface"
  raised-surface: "$surface-lighten-1"
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
