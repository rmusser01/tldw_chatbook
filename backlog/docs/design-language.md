# Design Language — tldw_chatbook UI Constitution

Status: governing document (ADR-150, TASK-32475)
Audience: humans and agents building or modifying any screen, widget, or
stylesheet in this repo.

> Humans define the design language. Agents speak the design language.
> Prompts are temporary; this document is memory.

When you design a new screen or modify UI, **compose from the tokens and
rules below**. Do not invent visual values per screen. If the language
cannot express what you need, extend the language first (see "Adding to the
language"), then compose.

## 1. Where things live

| Concern | Location |
|---|---|
| Design tokens (`$ds-*`) | `tldw_chatbook/css/core/_variables.tcss` |
| Textual theme palettes (raw hex) | `tldw_chatbook/css/Themes/themes.py` |
| Stylesheet modules | `tldw_chatbook/css/{core,layout,components,features,utilities}/` |
| Build manifest (module order) | `tldw_chatbook/css/build_css.py` (`CSS_MODULES`) |
| Generated bundle (never edit) | `tldw_chatbook/css/tldw_cli_modular.tcss` |
| Governance tests | `Tests/UI/test_design_token_governance.py`, `test_css_bundle_sync_guard.py` |

Textual TCSS substitutes variables **per token**: tokens are scalar,
single-value definitions (`$ds-space-1: 1;`, `$ds-duration-fast: 0.2s;`).
Multi-value composite tokens are not used (ADR-150).

## 2. Token catalog

### 2.1 Color and semantic meaning (existing layer)

- **Surfaces:** `$ds-surface-panel`, `$ds-surface-raised`,
  `$ds-surface-inspector`, `$ds-grid-line`, `$ds-column-line`,
  `$ds-surface-sunken` (recessed resting surface for **enabled** chrome —
  inactive toggles, sunken strips; never a disabled state)
- **Text:** `$ds-text-primary`, `$ds-text-muted`, `$ds-text-disabled`
- **Focus (non-obscuring focus contract):** `$ds-focus-bg`, `$ds-focus-fg`,
  `$ds-focus-accent`, `$ds-action-focus`, `$ds-input-focus-border`,
  `$ds-input-focus-bg`, `$ds-input-focus-accent`, `$ds-control-edge`
- **Status:** `$ds-status-ready | -running | -info | -warning |
  -approval-required | -blocked | -error | -error-readable | -paused |
  -unsaved | -recovered`
- **Authority / source roles:** `$ds-authority-*`, `$ds-source-role-*`

Never use raw hex in stylesheets; theme variables (`$primary`, `$surface`,
`$text-muted`, …) and `$ds-*` aliases are the only legal colors. Error text
that must be **read** uses `$ds-status-error-readable`, not `$error`
(a11y, TASK-2230).

### 2.2 Spacing scale

`$ds-space-0 | -1 | -2 | -3` (0–3 cells/rows). The TUI's physical unit is
one cell; the scale is deliberately tiny. Prefer the semantic aliases, which
make intent visible:

| Token | Use for |
|---|---|
| `$ds-space-inline` | gap between siblings in a horizontal row (buttons, chips) |
| `$ds-space-stack` | gap between stacked fields/rows in a form |
| `$ds-space-section` | gap between major sections; title ↔ content separation |
| `$ds-space-inset` | padding inside a panel, card, or bordered box |

### 2.3 Control sizing

- `$ds-control-height: 3` — standard action control (buttons, selects,
  single-line inputs)
- `$ds-control-height-compact: 1` — tab bars, status lines, chip rows
- `$ds-textarea-min-height: 5` — multi-line text inputs

### 2.4 Motion

`$ds-duration-fast: 0.2s` (hover feedback), `$ds-duration-medium: 0.3s`
(panel/visibility changes), `$ds-duration-slow: 0.5s` (large layout shifts
only). No other durations in new UI.

### 2.5 Opacity and the disabled state

`$ds-opacity-dim: 70%` is the only opacity token, for de-emphasized but
readable secondary content. **Opacity is not the disabled mechanism** for
action controls: stacking `opacity: 50%` on already-dim text composited to a
measured 1.34:1 and made disabled controls look absent (TASK-25822). The
disabled state is `$ds-disabled-bg` + `$ds-text-disabled-readable` at full
opacity; hover on disabled keeps the disabled surface.

### 2.6 Typography emphasis

TUI typography is weight/style, not size. `$ds-text-strong: bold`,
`$ds-text-emphasis: italic`. The focus marker `bold underline` is part of
the focus contract and stays as-is.

### 2.7 Component states

Every interactive control must define, at minimum:

- **rest:** default surface; editable fields carry the one-column left
  `$ds-control-edge` (task-1586 convention)
- **hover:** `$ds-hover-bg` / `$ds-hover-fg` — must NOT reuse the focus
  surface (hover is not focus)
- **focus:** `$ds-focus-bg` / `$ds-focus-fg` / `bold underline` — the
  non-obscuring focus contract: focus must be visibly distinct from rest
  AND must not obscure content. Never alias the focus surface back to a
  resting surface (this nullified focus twice: TASK-345, task-1586)
- **disabled:** `$ds-disabled-bg` + `$ds-text-disabled-readable` at full
  opacity (never opacity-stacked — see §2.5); hover on disabled keeps the
  disabled surface

### 2.8 Layout laws

- **Standard shell sidebar:** `$ds-sidebar-width: 25%`,
  `$ds-sidebar-min-width: 35`, `$ds-sidebar-max-width: 80`, docked left,
  collapsible to zero width (never a sliver).
- **Feature geometry stays local** until a second consumer appears; then it
  promotes to `_variables.tcss` (e.g. `$ds-console-composer-height`).
- Pane borders use `$ds-grid-line` / `$ds-column-line`, never raw hex.

## 3. Interaction rules (binding on all screens)

1. **Focus is sacred.** Every keyboard-reachable element must show visible,
   non-obscuring focus per §2.7. An invisible focus indicator is a critical
   defect (a Tab+Enter once activated "Save as…" invisibly).
2. **Selects are three shapes.** A `Select` may draw its own border, draw it
   on `SelectCurrent`, or be borderless-compact. Style only what every shape
   has — its colors — unless you have measured the geometry (TASK-2300).
3. **Left borders cost a column, never a row** — prefer them over
   top/bottom edges in dense one-row forms.
4. **Readable beats decorative.** Status hues that fail contrast on dark
   canvases use the `-readable` variant for text.
5. **Footer hints advertise only implemented actions**; screens do not bind
   terminal-convention keys (see `backlog/decisions/031-*.md`).

## 4. How to describe a new screen (the recipe)

When asking an agent (or planning yourself) for new UI, describe it in the
language:

1. **Layout law:** which shell pattern (sidebar screen, three-pane
   workbench, modal dialog, compact strip) and which layout tokens apply.
2. **Component inventory:** which existing components/classes
   (`.form-input`, `.form-section-title`, `.status-area`, `Button`,
   …) fill each region. New components reuse state tokens from §2.7.
3. **Semantic meaning:** which status/authority/source-role tokens each
   element carries.
4. **States:** confirm rest/hover/focus/disabled are all specified.
5. **Exceptions:** anything the tokens cannot express → add a token first.

A good brief sounds like: *"Standard sidebar shell (`$ds-sidebar-*`), a
stacked form (`$ds-space-stack`) of `.form-input` fields, actions row at
`$ds-space-section` with default Button states, status line using
`$ds-status-unsaved` when dirty."* — no colors, no pixels, no "modern".

## 5. Adding to the language

1. Add the token to `core/_variables.tcss` in the right section, scalar
   value, `$ds-<category>-<name>[-<variant>]` naming. Aliases point at
   scale tokens, never at raw literals.
2. If it is a **rule** rather than a value, add it to this document.
3. Rebuild the bundle: `python tldw_chatbook/css/build_css.py` (the app also
   rebuilds on boot when sources are newer; the committed bundle must
   reproduce — `test_css_bundle_sync_guard.py` enforces this).
4. Run `pytest Tests/UI/test_design_token_governance.py Tests/UI/`.
5. Changing an **existing token's value** is a visual-breaking change:
   expect to verify live (`backlog/docs/lessons-live-verification.md`).

## 6. What is deliberately NOT tokenized yet

- The ~7,000 legacy dimension literals in feature sheets — migrate
  opportunistically when touching a file; new sheets must be token-clean.
  **Migration reference:** `features/_chat.tcss` (TASK-32480) is the
  canonical end-to-end example of the pattern — tokenize what maps 1:1,
  keep feature-specific geometry literal with its comments, invent nothing
  in the feature sheet.
- Python-side `styles.*` mutations (~250 call sites) — new code assigns
  token-backed classes instead of ad-hoc literals.
- Multi-value composite tokens (e.g. `border: round $border` as one token)
  — blocked on verified Textual substitution semantics (ADR-150).
- Feature-scoped geometry tokens already in `_variables.tcss` (`$ds-home-*`,
  `$ds-library-*`, …) — promote to layout laws when reused.
