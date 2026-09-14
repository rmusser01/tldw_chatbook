# Component-Pattern Catalog

Date: 2026-09-13
Governance: ADR-161 (`backlog/decisions/161-component-pattern-library.md`) — the
component-pattern layer of the design system
Design: `Docs/superpowers/specs/2026-09-13-component-pattern-library-design.md`
(§3.2 catalog, §3.3 registry)
Under: ADR-150 and the design-language constitution (`backlog/docs/design-language.md`)
— the constitution stays the laws; this catalog is the reference.

This is the catalog of canonical component patterns: the CSS-class vocabulary
screens compose from, plus the structure and state contracts around it. Screens
keep hand-composing `classes="..."` strings (ADR-161 rejected Python builders);
what changes is that there is now one winning vocabulary per family, documented
here and enforced mechanically.

## How to use this catalog

- Build UI by composing the classes below into `classes="..."` strings; do not
  invent sibling conventions (`settings-input-label`, `field-label`, …). If a
  pattern is missing, extend the family's owning sheet — see ADR-150 for the
  token rules any new rule must obey.
- Check `tldw_chatbook/css/patterns.json` (the machine-readable registry) before
  adding a class definition: a `Canonical` class may be *defined* only in its
  owning sheet; compound/scoped selectors using it (`.compact .form-label`) are
  legal composition anywhere.
- Every visual value in a pattern's rules traces to a `$ds-*` token
  (`tldw_chatbook/css/core/_variables.tcss`). Never introduce literals.

## Lifecycle statuses

- **Canonical** — the winning pattern. Single-definition rule applies: a rule
  whose selector is exactly the class (plus state pseudo-selectors) may exist
  only in the owning sheet.
- **Deprecated** — still rendered, but renamed; carries `renames_to` and a
  usage-count ratchet in the registry that may only decrease.
- **Draft** — the pattern is real and in use, but its canonical owning sheet
  does not exist yet; governance does not enforce single-definition until it
  flips to Canonical.

## The registry (`tldw_chatbook/css/patterns.json`)

The registry is the machine-readable twin of this doc and the single source the
governance tests read (`Tests/UI/test_component_pattern_governance.py`, Task 6).
Schema:

```json
{
  "version": 1,
  "families": {
    "<family>": {
      "owning_sheet": "components/_forms.tcss",
      "widget_contract": "optional free-text (e.g. ListView/DataTable type selectors)",
      "classes": { "<class>": { "status": "canonical" } }
    }
  },
  "deprecated": { "<class>": { "renamed_to": "<class>", "ceiling": 0 } }
}
```

Notes on the schema:

- `owning_sheet` paths are relative to `tldw_chatbook/css/` and name a
  `CSS_MODULES` sheet (see `tldw_chatbook/css/build_css.py`).
- A class may carry its own optional `owning_sheet` override when it physically
  lives in a different sheet than the rest of its family. Today exactly one
  exists: `nav-button` sits in `components/_navigation.tcss` (its states and the
  clip-ghost override), while the `sidebar-*` classes stay with the family sheet
  `layout/_sidebars.tcss`. JSON has no comments — this paragraph is the
  documentation for that key.
- The doc-sync governance test requires every class in the registry to appear in
  this document; keep the two in lockstep (the quick-reference table below is
  the sync surface).

### Registry quick reference

| Family | Owning sheet | Classes (status) |
|---|---|---|
| forms | `components/_forms.tcss` | form-label, form-row, form-col, form-input, form-textarea, form-select, form-checkbox, form-button, form-section-title, form-section-collapsible, form-actions (all Canonical); settings-input-label (Canonical variant of form-label, owning sheet `features/_settings.tcss`, bundle-resident via the settings split's pinned token); settings-compact-input (Canonical variant of form-input), settings-detail-row (Canonical variant of form-row) — both task 9; settings-input-row, settings-select-row, settings-status-row, settings-compact-select, settings-focus-card (Draft — Settings-screen-only, landed in features/_settings.tcss with the task-10 carve) |
| buttons | `components/_buttons.tcss` | action-button, button-group, button-group-left, button-group-center, button-group-right, sidebar-toggle (all Canonical) |
| lists | `components/_lists.tcss` | none — widget contract (ListView/DataTable/OptionList cursor-hover-selected type selectors) |
| dialogs | `components/_dialogs.tcss` | dialog-title, dialog-buttons (all Canonical) |
| status | `components/_status.tcss` | status-label, status-area, status-item (all Canonical) |
| navigation | `layout/_sidebars.tcss` | sidebar, sidebar-button, sidebar-header, sidebar-listview, sidebar-section-collapsible (Canonical); nav-button (Canonical, owning sheet `components/_navigation.tcss`) |
| sections | `components/_sections.tcss` | section-title, section-header, subsection-title (all Canonical) |
| messages | `components/_messages.tcss` | message-header, message-text, message-actions (all Canonical) |
| ds_primitives | `components/_ds_primitives.tcss` | ds-panel, ds-toolbar, ds-field-row, ds-info-callout, ds-approval-card, ds-destination-header (all Canonical; extracted from the agentic monolith in task 9) |
| sizing | `utilities/_helpers.tcss` | w-auto, w-full, w-fill, w-0, h-auto, h-full, h-fill, h-0, h-1, h-2, h-3, p-0, m-0, border-none (all Canonical; the tokenized replacements for runtime `.styles.*` literals, task 12) |

## Entry template

Every family entry has this fixed shape (spec §3.2):

- **Purpose** and **when-not-to-use** (one paragraph each)
- **Structure**: a minimal `compose()` snippet showing the container/class skeleton
- **Class inventory**: each class, its role, required rest/hover/focus/disabled
  states (state contract per constitution §2.7)
- **Tokens consumed** (`$ds-*` links; never literals)
- **Python idiom** where one exists
- **Lifecycle status**: Canonical, Deprecated (with rename target + ratchet
  count), or Draft

---

## Forms & fields

**Purpose.** Dense settings-style forms and dialogs: label + control pairs,
rows and columns, textareas, selects, checkboxes, collapsible advanced
sections, and a trailing action row. This is the default way to lay out any
group of inputs.

**When-not-to-use.** Console transcript rows — that is the messages family's
grammar (speaker accents, tool-call/result variants). One-off prompts with a
single field and no labels don't need the form skeleton; compose the widget
directly with the Button type contract.

**Structure.**

```python
def compose_form(self):
    yield Label("Model", classes="form-label")
    yield Input(placeholder="gpt-4o", classes="form-input")
    with Container(classes="form-row"):
        yield Label("Temperature", classes="form-label")
        yield Input(classes="form-input")
    yield TextArea(classes="form-textarea")
    yield Select([], classes="form-select")
    with Collapsible(title="Advanced", classes="form-section-collapsible"): ...
    with Container(classes="form-actions button-group button-group-right"):
        yield Button("Save", classes="form-button")
```

**Class inventory.**

- `form-label` — field label. No interactive states.
- `settings-input-label` — documented **variant** of `form-label` (ADR-161
  task 7, probe-measured): the settings grid-row label column — fixed
  24-col width (min 12), height 1, `$ds-surface-panel` background,
  `$ds-text-primary`; not density-equivalent to `form-label`, kept as a
  variant rather than force-merged. Owning sheet
  `components/_agentic_terminal.tcss`; scoped compounds (RAG card width 20,
  imagegen backend rows width 12, stacked rows width 100%) compose on top.
- `form-row` — horizontal field-grouping container; stack with `$ds-space-stack`.
- `settings-detail-row` — documented **variant** of `form-row` (ADR-161 task 9,
  probe-measured): the read-only detail line — `$ds-surface-panel` fill,
  `$ds-text-primary`, one-row minimum, `$ds-space-inline` padding. Consumed on
  two production surfaces (Settings workbench + the STTS speech settings pane),
  so it lives in this owning sheet (boot bundle) rather than the lazily-loaded
  settings screen sheet. `#settings-impact-pane .settings-detail-row` stays
  with the Settings surface.
- `form-col` — equal-width (`1fr`) column inside a `form-row`; edge columns drop
  their outer `$ds-space-inline` padding.
- `form-input` — single-line `Input`. Rest edge per §2.7 (the compact-field
  `$ds-control-edge` left-edge convention); `:focus` recolours via
  `$ds-input-focus-border` / `$ds-input-focus-accent` / `$ds-input-focus-bg`;
  `.error` modifier marks validation failure.
- `settings-compact-input` — documented **variant** of `form-input` (ADR-161
  task 9, probe-measured): the one-row editable field — height
  `$ds-control-height-compact`, `$ds-width-fill` share of the row, the
  task-1586 one-column left edge at rest flipping to a thick
  `$ds-action-focus` edge on focus, no underline (task-185). Its border
  suppression is what lets the single row paint text (TASK-13154.1); do NOT
  merge into `form-input`. Also composed by the pattern gallery, so it lives
  in this owning sheet (boot bundle). The `.settings-invalid-input` states
  (error tint, stronger at 28% while focused — task-1369) live with it;
  scoped compounds (`#settings-web-search-panel …`,
  `.settings-speech-path-control …`) stay with the Settings surface.
- `form-textarea` — multi-line `TextArea`, `min-height: $ds-textarea-min-height`;
  same `:focus` contract as `form-input`.
- `form-select` — `Select` wrapper sized to `$ds-control-height`; focus states
  come from the shared `Select:focus` recolour in the owning sheet.
- `form-checkbox` — checkbox/toggle margin slot; the `ToggleButton:focus`
  contract (non-obscuring recolour of `.toggle--label`) is pinned in the owning
  sheet and by `Tests/UI/test_non_obscuring_focus_contract.py`.
- `form-button` — `Button` inside form action rows; hover/focus/disabled states
  are the Button type contract from the buttons family.
- `form-section-title` — section heading with rule, `$ds-text-strong`.
- `form-section-collapsible` — `Collapsible` section wrapper for advanced
  options.
- `form-actions` — trailing action row; compose with
  `button-group button-group-right` for right-aligned actions.

**States.** §2.7 contract via the `form-input` rest edge (`$ds-control-edge`)
and the shared hover/focus/disabled vocabulary: `$ds-hover-*`, `$ds-focus-*`,
`$ds-disabled-*`.

**Tokens consumed.** `$ds-space-stack`, `$ds-space-inline`,
`$ds-space-section`, `$ds-space-inset`, `$ds-control-height`,
`$ds-textarea-min-height`; typography `$ds-text-strong`, `$ds-text-emphasis`;
text `$ds-text-muted`, `$ds-text-primary`; surfaces `$ds-surface-raised`; focus
`$ds-input-focus-border`, `$ds-input-focus-accent`, `$ds-input-focus-bg`,
`$ds-focus-bg`, `$ds-focus-fg`, `$ds-focus-accent`, `$ds-action-focus`.

**Python idiom.** `Widgets/form_components.py` builders exist (2 importers) —
optional, not canonical. Hand-compose.

**Lifecycle.** All Canonical (owning sheet `components/_forms.tcss`), plus
the `settings-input-label` variant (Canonical, owning sheet
`features/_settings.tcss` — task 10), the `settings-compact-input` variant of
`form-input` and the `settings-detail-row` variant of `form-row` (Canonical,
this owning sheet — task 9). The Settings-screen-only field classes
(`settings-input-row`, `settings-select-row`, `settings-status-row`,
`settings-compact-select`, `settings-focus-card`) are Draft: probe-measured
genuinely different from the forms family but consumed only by the Settings
workbench, so they landed in `features/_settings.tcss` with the task-10
carve (that sheet is the settings split's source; the pure rules ride the
lazily-loaded settings screen sheet exactly as before).

**Consolidation record (ADR-161 task 9, 2026-09-13).** The remaining
settings field vocabulary was probed against the forms family on the
production cascade (computed styles + height-1 geometry, Task 7
methodology): every class diverges materially (3–11 property diffs), so
nothing was renamed or deprecated — all are variants, not duplicates.
`settings-compact-input` and `settings-detail-row` moved tokenized into this
owning sheet because each has a second surface beyond Settings (the pattern
gallery; the STTS speech settings pane). The move fixes a latent
visit-order trap the split documented for `settings-input-label`: those two
definitions rode the lazy settings screen sheet, so the STTS pane and the
gallery demo rendered them unstyled until Settings was first visited.
Bundle-only probes after the move match the full-stack target exactly. The
five Settings-only siblings stay put (Draft) with their consumers;
`settings-compact-select`'s `> SelectCurrent`/`> SelectOverlay` shape rules
stay with it.

**Consolidation record (ADR-161 task 7, 2026-09-13).** The competing label
conventions were renamed onto `form-label` and their definitions deleted:
`field-label` (probe-identical computed styles), `form-field-label` (had no
CSS anywhere — gained the winner's styling), `settings-label` (color-only
divergence on the deprecated legacy Tools & Settings window — intended
consistency change: `$primary` → `$text-muted`, spacing total unchanged),
`setting-label` (dead CSS, zero compose sites — deleted). All four are
Deprecated in the registry: `setting-label`, `settings-label` and
`form-field-label` at ceiling 0; `field-label` at ceiling 1 — the single
remaining count is the prose "Title Case field-label convention" comment in
`UI/Screens/settings_screen.py` (about label wording, not the CSS class).

---

## Buttons & action rows

**Purpose.** Action rows and grouped buttons, plus the app-wide `Button` type
contract (hover/focus/disabled) that every button in every family inherits.

**When-not-to-use.** Sidebar navigation destinations use `sidebar-button` /
`nav-button` (navigation family); message action strips use `.message-actions`
(messages family). `sidebar-toggle` is only for the equal-width collapsing
toggle strips beside a docked sidebar.

**Structure.**

```python
def compose_actions(self):
    with Container(classes="button-group button-group-right"):
        yield Button("Cancel", id="cancel")
        yield Button("Save", id="save", variant="primary", classes="form-button")
    yield Button("Notes", classes="sidebar-toggle")
```

**Class inventory.**

- `action-button` — a `Button` in a trailing/toolbar action row;
  `$ds-space-inline` right margin, `min-width: 12`. `.primary` modifier
  recolours to `$primary`/`$text`; `.primary.large` bumps to `min-width: 20`,
  height 3, `$ds-text-strong`. Hover/focus/disabled come from the Button type
  contract. Promoted from `features/_evaluation_unified.tcss` (ADR-161
  task 4; cascade-neutral — it was the sole surviving bare definition).
- `button-group` — full-width group container; margins from
  `$ds-space-section`.
- `button-group-left` / `button-group-center` / `button-group-right` —
  alignment modifiers for the group. No states of their own.
- `sidebar-toggle` — equal-width (`1fr`) strip button at `$ds-control-height`;
  enabled but recessed chrome — resting surface is `$ds-surface-sunken` (NOT
  the disabled token), hover lifts to `$ds-surface-raised`.
- Button type contract (owning sheet): `:hover` → `$ds-hover-bg` +
  `$ds-hover-fg` + `$ds-text-strong`; `:focus` → `$ds-focus-bg` +
  `$ds-focus-fg` + bold underline, outline suppressed; `:disabled` →
  `$ds-disabled-bg` + `$ds-text-disabled-readable` at full opacity
  (TASK-25822: opacity-on-dim-text read as absent, not unavailable).

**Tokens consumed.** `$ds-hover-bg`, `$ds-hover-fg`, `$ds-focus-bg`,
`$ds-focus-fg`, `$ds-text-strong`, `$ds-disabled-bg`,
`$ds-text-disabled-readable`, `$ds-control-height`, `$ds-surface-sunken`,
`$ds-surface-raised`, `$ds-text-primary`, `$ds-space-section`,
`$ds-space-inline`, `$ds-space-0`.

**Python idiom.** none — hand-compose class strings. (`create_button_group`
in `Widgets/form_components.py` exists but is optional, not canonical.)

**Lifecycle.** All Canonical. The `button-group*` definitions moved from
`components/_forms.tcss` and `.action-button` from
`features/_evaluation_unified.tcss` into the owning sheet
`components/_buttons.tcss` (ADR-161 task 4; no rename, so no Deprecated
entry).

---

## Lists & tables

**Purpose.** Consistent cursor / hover / selected rendering for `ListView`,
`DataTable`, `OptionList`, `SelectionList`, and `Tree` everywhere, without any
per-screen classes. This family is a **widget contract**, not a class
vocabulary: the owning sheet styles the widgets' own internal parts
(`ListItem:hover`, `.-highlight`, `.datatable--cursor/--hover/--selected`,
`.option-list--option-highlighted`, `.tree--cursor`, …) via type selectors.

**When-not-to-use.** Do not invent list-item classes for what the contract
already styles — a screen needing different row chrome composes a scoped
compound selector in its own feature sheet (composition is always legal).
Custom item rendering inside a ListView still uses the contract for states.

**Structure.** There are no classes to compose — instantiate the widget and
the contract applies:

```python
def compose_list(self):
    yield ListView(*items)      # ListItem hover/-highlight from the contract
    yield DataTable()           # cursor/header focus cues from the contract
    yield Select(options)       # OptionList overlay contract
```

**Class inventory.** None (no public classes). The contract additionally pins
the non-obscuring focus behaviour for focused tables/trees/compact selects
(TASK-1160 / TASK-2300: recolour cells the widget already draws; never an
outline that overwrites content or its click metadata).

**Tokens consumed.** `$ds-focus-bg`, `$ds-focus-fg`, `$ds-grid-line`,
`$ds-input-focus-bg`.

**Python idiom.** none — hand-compose widgets; the contract applies
automatically.

**Lifecycle.** Canonical as a contract (owning sheet
`components/_lists.tcss`); registry `"classes"` is empty by design.

---

## Dialogs & modals

**Purpose.** Any modal: subclass `SafeModalDismissMixin` +
`ModalScreen[None]`, compose a frame body, a centered title, optional subtitle
and message, and a trailing action row.

**When-not-to-use.** Non-modal popovers (inline reveals, dropdowns, the
pattern gallery). Those are compositions on the current screen, not modals.

**Structure.**

```python
class RenameDialog(SafeModalDismissMixin, ModalScreen[None]):
    def compose(self) -> ComposeResult:
        with Container():  # the dialog frame body
            yield Label("Rename", classes="dialog-title")
            yield Input(classes="form-input")
            with Container(classes="dialog-buttons button-group button-group-right"):
                yield Button("Cancel", id="cancel")
                yield Button("Rename", id="confirm", variant="primary")
```

**Class inventory.**

- `dialog-title` — centered bold heading (`$ds-text-strong`,
  `$ds-space-stack` bottom margin). Canonical definition in the owning sheet
  since ADR-161 task 4 (promoted from `features/_media.tcss`;
  `ConfirmationDialog.DEFAULT_CSS`'s latent copy deleted with the move).
- `dialog-subtitle` — muted secondary line (still defined in
  `features/_media.tcss`; promotes in a later task).
- `dialog-message` — body copy above the buttons (still defined in
  `features/_media.tcss`; promotes in a later task).
- `dialog-buttons` — horizontal action row at height 3, `align: center
  middle`; each `Button` takes `$ds-space-inline` side margins and
  `min-width: 12`. Compose with `button-group button-group-right` for the
  standard right-aligned layout.

**States.** Per §2.7 — the dialog itself has no states; the buttons in
the row follow the Button type contract (`$ds-hover-*` / `$ds-focus-*` /
`$ds-disabled-*`).

**Tokens consumed.** `$ds-text-strong`, `$ds-space-stack`,
`$ds-space-inline`, `$ds-space-0`; action states come from the buttons
family.

**Python idiom.** `ConfirmationDialog` (`Widgets/confirmation_dialog.py`, 36
importers) for confirm flows; `SafeModalDismissMixin`
(`Widgets/modal_dismissal.py`, 58 of 66 modals) for every modal's
escape/click dismissal contract.

**Lifecycle.** Canonical family (owning sheet `components/_dialogs.tcss`;
`dialog-title`/`dialog-buttons` promoted and registered in ADR-161 task 4 —
the move is cascade-neutral: every competing rule is id- or widget-scoped
and wins by specificity). `dialog-subtitle`/`dialog-message` join the
registry when they promote.

---

## Status, empty & loading states

**Purpose.** Status readouts: a muted label over a read-only output surface
(`status-label` + `status-area`), plus empty and loading states documented as
composition recipes (the dead `empty_state`/`loading_states` widgets were
deleted per ADR-161 §3.8 — compose `Label`/`Loading` with these classes
instead).

**When-not-to-use.** Transient notifications use Textual's `notify()` — there
is no toast class to compose. Flow-embedded callouts and approval prompts are
ds-primitives (`ds-info-callout`, `ds-approval-card`), not status areas.

**Structure.**

```python
def compose_status(self):
    yield Label("Status:", classes="status-label")
    yield TextArea("", id="ingest-status", read_only=True, classes="status-area")
```

**Class inventory.**

- `status-label` — muted caption above the area (`$ds-text-muted`).
- `status-area` — read-only output surface: `$ds-surface-raised` fill, rounded
  border, `$ds-space-inset` padding.
- `status-item` — per-line status entry (wizard progress lists). States:
  `.completed` (`$success`), `.active` (`$ds-focus-bg`/`$ds-focus-fg`, bold
  underline), `.error` (`$error`). The Import wizard's `.warning` state is a
  documented feature-local scoped compound (`ImportProgressStep
  .status-item.warning` in `features/_wizards.tcss`, task-19734 — an import
  that skipped everything is not green), not part of the canonical set.

**States.** Read-only surfaces — no hover/focus/disabled contract of their
own. Semantic colouring maps to the `$ds-status-*` set (readable error text
uses `$ds-status-error-readable`, per task-2230).

**Tokens consumed.** Today: `$ds-text-muted`, `$ds-surface-raised`,
`$ds-space-inset`, `$ds-space-stack`, `$ds-space-section`, `$ds-space-0`,
`$ds-space-inline`, `$ds-width-full`; state accents `$ds-focus-bg` /
`$ds-focus-fg` on `.status-item.active`. Semantic mapping:
the `$ds-status-*` family (`$ds-status-ready`, `$ds-status-running`,
`$ds-status-info`, `$ds-status-warning`, `$ds-status-error`,
`$ds-status-error-readable`, …).

**Python idiom.** `create_status_area` in `Widgets/form_components.py` —
optional; hand-composing the two widgets is equally canonical.

**Lifecycle.** Canonical family (owning sheet `components/_status.tcss`,
filled by ADR-161 task 8a).

**Consolidation record (ADR-161 task 8a, 2026-09-13).** `.status-label` /
`.status-area` moved from `components/_forms.tcss` (byte-identical winners,
tokenized on arrival; `width: 100%` → `$ds-width-full`, the new shared token
for full-width promoted components). `.status-item` promoted bare from the
two declaration-identical scoped sets in `features/_wizards.tcss`
(`ProgressStep` / `ImportProgressStep`); computed-style probes in both step
contexts were identical before and after (margin `0 0 1 2` via
`$ds-space-0 $ds-space-0 $ds-space-stack $ds-space-2`, muted base,
`$success`/focus/`$error` states), height-1 geometry captured in both runs.
The `ImportProgressStep .status-item.warning` state stayed feature-scoped.

---

## Navigation & sidebars

**Purpose.** Docked sidebars and destination navigation: the standard shell
sidebar with header, lists, and collapsible sections, and the main navigation
strip's destination buttons.

**When-not-to-use.** Main-content panels and inspectors are sections /
ds-primitives territory; toolbar strips inside content use `ds-toolbar`. A
sidebar that is really a settings form should compose forms-family classes
inside a `sidebar` container.

**Structure.**

```python
def compose_sidebar(self):
    with Container(classes="sidebar"):
        yield Label("Conversations", classes="sidebar-header")
        yield ListView(id="conv-list", classes="sidebar-listview")
        with Collapsible(title="Recent", classes="sidebar-section-collapsible"):
            yield ListView()
        yield Button("New", classes="sidebar-button")

# Main navigation strip (UI/Navigation/main_navigation.py):
# NavigationButton(...) renders with the nav-button contract
```

**Class inventory.**

- `sidebar` — docked-left container; geometry is the ADR-150 layout law
  (`$ds-sidebar-width` / `$ds-sidebar-min-width` / `$ds-sidebar-max-width`);
  `.collapsed` modifier hides it.
- `sidebar-header` — 3-row header band above the sidebar content.
- `sidebar-button` — full-width menu button inside a sidebar.
- `sidebar-listview` — bounded list surface inside a sidebar.
- `sidebar-section-collapsible` — grouped collapsible section;
  `:focus-within` recolours via `$ds-focus-accent` / `$ds-focus-bg`.
- `nav-button` — destination button in the main navigation strip; owning
  sheet `components/_navigation.tcss` (per-class override — the
  `.nav-button-clip-ghost` scroll-edge override lives there; base geometry
  and resting chrome come from `MainNavigationBar`'s scoped `BUNDLED_CSS`
  rules, `.nav-button` / `:hover` / `.is-active`, which migrate per the
  BUNDLED_CSS rule). The TASK-16811 `NavigationButton.active`/`:focus`
  type-selector rules that sat in the owning sheet were deleted as dead in
  ADR-161 task 8b, for two independent reasons (fix round 1 corrected the
  rationale): the surviving `UI/Navigation/main_navigation.py`
  NavigationButton (same CSS type name as the deleted base_components
  widget those rules came with) never receives an `active` class (only
  `is-active`), so `.active`/`.active:focus` matched nothing; and the
  `:focus` rule's declarations were value-identical to the app-tier
  `Button:focus` / `Button:hover:focus` contract in
  `components/_buttons.tcss`, which matches every NavigationButton —
  deleting a same-value duplicate cannot change computed styles
  (probe-verified computed-identical on the mounted bar). Do NOT restore
  the rules on a "scoped rules out-rank the type selector" belief: tier
  beats specificity in Textual — app-CSS (`CSS_PATH`) sources beat
  widget-default-tier sources regardless of specificity, so the bar's
  lifted-BUNDLED_CSS scoped rules can never neutralize an app-tier type
  selector (experimentally confirmed by re-inserting the rule's shape with
  a red background at app tier: it paints).

**States.** §2.7 for the interactive classes (buttons take the Button type
contract; `sidebar-section-collapsible` carries the `:focus-within` recolour).

**Tokens consumed.** `$ds-sidebar-width`, `$ds-sidebar-min-width`,
`$ds-sidebar-max-width`, `$ds-space-inline`, `$ds-focus-accent`,
`$ds-focus-bg`, `$ds-focus-fg`.

**Python idiom.** none — hand-compose class strings.

**Lifecycle.** All Canonical (family sheet `layout/_sidebars.tcss`;
`nav-button` at `components/_navigation.tcss` via the per-class override).

---

## Sections & containers

**Purpose.** Titled sections and subsections inside screens and panels:
`section-header` / `section-title` to open a major region, `subsection-title`
to divide it.

**When-not-to-use.** Form-internal section headings use
`form-section-title` (forms family); sidebar grouping uses
`sidebar-section-collapsible` (navigation family).

**Structure.**

```python
def compose_sections(self):
    yield Label("Sources", classes="section-header")
    yield DataTable()
    yield Label("Remote indexes", classes="subsection-title")
    yield ListView()
```

**Class inventory.**

- `section-title` — major region heading.
- `section-header` — major region heading, underlined variant.
- `subsection-title` — secondary heading within a section.

`section-title` and `subsection-title` each have exactly one bare definition
now, in the owning sheet `components/_sections.tcss` (ADR-161 task 4
dissolved the `features/_tools-settings.tcss` ×2 and
`features/_evaluation_unified.tcss` duplicates; the winners are the
tools-settings pair by usage weight, tokenized on arrival). `.section-header`'s
canonical definition also lives there, moved from
`components/_shared_components.tcss` — two later-in-cascade copies remain as
documented follow-up, not part of the family: `features/_chat.tcss`'s bare
bold-underline variant (still wins text-style/margins app-wide) and
`components/stats_screen.css`'s screen-sheet copy (still supplies its
`border-left` and, now that it follows the owning sheet in the manifest,
`padding-left: $ds-space-1`).

**States.** Static text — no interactive states.

**Tokens consumed.** `$ds-text-strong`, `$primary`/`$secondary` theme
colours, `$ds-space-section`, `$ds-space-stack`, `$ds-space-1`,
`$ds-space-0`.

**Python idiom.** none — hand-compose class strings.

**Lifecycle.** All Canonical (owning sheet `components/_sections.tcss`,
created by ADR-161 task 4).

---

## Chat & message rendering

**Purpose.** Transcript grammar: `ChatMessage` bubbles with speaker variants
(`-user`, `-ai`, `-tool-call`, `-tool-result`), a one-row header, scrollable
message text, and an action strip that reveals on completion.

**When-not-to-use.** Settings forms and dialogs have their own families; a
static log readout is a status area, not a transcript.

**Structure.** Screens mount the widget; the skeleton is its compose contract
(`Widgets/Chat_Widgets/chat_message.py`):

```python
# ChatMessage(classes="-user")   # variant: -user / -ai / -tool-call / -tool-result
# └─ Vertical                    # the bubble surface; variant classes recolour it
#    ├─ Label(role, classes="message-header")
#    ├─ Markdown(text, classes="message-text")
#    └─ Horizontal(classes="message-actions")   # Buttons follow the Button contract
```

**Class inventory.**

- `ChatMessage` type contract — full-width auto-height bubble; variant classes
  set speaker/tool surfaces (`-user`, `-ai`, `-tool-call`, `-tool-result`).
- `message-header` — one-row speaker/tool header band.
- `message-text` — scrollable body text (`.tts-generating` modifier marks
  in-flight speech).
- `message-actions` — bottom action strip; `.-generating` hides it until
  generation completes. Buttons take the Button type contract
  (`$ds-focus-bg` / `$ds-focus-fg` on focus); the strip's own Button rest/
  hover and the `.delete-button` destructive hover are pinned in the owning
  sheet.

**States.** §2.7 for the action-strip buttons; the bubbles themselves are
static containers.

**Tokens consumed.** `$ds-focus-bg`, `$ds-focus-fg`, `$ds-surface-raised`
(action-strip button states in the owning sheet). Speaker accents:
`$ds-chat-user-accent` and the `$ds-chat-user-accent-*` /
`$ds-chat-assistant-accent-*` dark/light polarity pairs from
`core/_variables.tcss` (consumed by the console/chat screens); inline
roleplay semantics use the `$ds-chat-rp-*` pairs.

**Python idiom.** none — hand-compose class strings.

**Lifecycle.** Canonical family (owning sheet `components/_messages.tcss`).

**Consolidation record (ADR-161 task 8c, 2026-09-13).** The shared grammar
was deduplicated INTO the owning sheet, which already carried the
app-tier winners: `ChatMessage`'s entire `DEFAULT_CSS` (every rule an
identical or strict-subset copy of bundle rules — app CSS outranks
`DEFAULT_CSS` regardless of specificity) was deleted, as were
`ChatMessageEnhanced`'s copies of `.message-header`/`.message-text`/
`.message-actions` (+ strip Button hovers and the delete-button hover,
which moved here). Probe-verified computed-identical on the mounted
widgets (user/ai/tool-call bubbles, headers, text, strips). Three
findings recorded rather than forced:
1. `.message-text Markdown` (both widgets' DEFAULT_CSS) was **dead** —
   the Markdown widget *is* `.message-text`; nothing nests one inside
   it. Deleted, not moved.
2. `.message-text.tts-generating` **diverges**: the owning sheet's rule
   (thick accent border + opacity 0.8) wins the border; ChatMessageEnhanced's
   widget-local copy still contributes `padding-left: 1` on top. Kept as
   a documented widget-local variant, not merged.
3. Console transcript rows are **not** shared vocabulary: every console
   transcript class is `console-*`-prefixed feature-local chrome
   (`console-transcript-message-*`, `console-message-header`, …) in
   `features/_console{,_panels}.tcss` since the task-10 carve — they stay
   feature-local.
Also deleted as dead: `features/_chat.tcss`'s `.chat-message.user` /
`.chat-message.assistant` (zero compose sites). Known dead artifact left
for a later sweep: `Constants.css_content` — an unreferenced legacy
CSS string carrying old copies of several family classes; it is never
registered as a stylesheet, so it cannot affect rendering or governance.

---

## ds-primitives

**Purpose.** The token-native primitive containers that grew up inside the
agentic-terminal monolith and now form the base layer other families compose
from: panels, toolbars, field rows, callouts, approval cards, and the
destination header.

**When-not-to-use.** When a family pattern already covers the need — fields
belong to forms, modals to dialogs, status readouts to status. Feature-specific
chrome stays feature-scoped rather than growing this family.

**Structure.**

```python
def compose_panel(self):
    with Container(classes="ds-panel"):
        with Container(classes="ds-toolbar"):
            yield Button("Refresh")
        with Container(classes="ds-field-row"):
            yield Label("Name", classes="form-label")
            yield Input(classes="form-input")
    with Container(classes="ds-info-callout"):
        yield Static("Server is off — path now local-only.")
    with Container(classes="ds-approval-card"):
        yield Static("Approve fs_write on config.toml?")
```

**Class inventory.**

- `ds-panel` — bordered panel surface (`$ds-surface-panel` fill,
  `$ds-grid-line` border).
- `ds-toolbar` — one-row toolbar strip on `$ds-surface-raised`.
- `ds-field-row` — label-above-value field container (vertical; min-height 2).
- `ds-info-callout` — informational callout, `$ds-status-info` border + tint.
- `ds-approval-card` — approval prompt card, `$ds-status-approval-required`
  thick border + tint.
- `ds-destination-header` — tall-accented destination header
  (`$ds-action-focus` border, `$ds-surface-panel` fill).

Defined in `components/_ds_primitives.tcss` (the bundle's FIRST component
sheet — the atomic layer) since ADR-161 task 9, extracted tokenized from
`components/_agentic_terminal.tcss`; their `.density-compact` /
`.density-comfortable` variants ride with the family. ADR-161 task 10
moved the six leftover shell primitives in tokenized as well
(`.ds-inspector`, `.ds-status-badge`, `.ds-recovery-callout`,
`.ds-source-role`, `.ds-event-row`, `.ds-shortcut-bar`, with their density
companions) — three are composed app-wide, the components home; they are
not registry classes until a catalog pass documents them. Rest-state geometry
lives in tokens (`$ds-panel-min-height`, `$ds-destination-header-min-height`
and its density variants, `$ds-field-row-min-height`, `$ds-toolbar-min-height`).

**States.** Static containers — interactive states come from whatever
controls are composed inside them (buttons → Button contract).

**Tokens consumed.** `$ds-surface-panel`, `$ds-surface-raised`,
`$ds-text-primary`, `$ds-grid-line`, `$ds-action-focus`, `$ds-status-info`,
`$ds-status-approval-required`, `$ds-text-strong`, `$ds-space-0/1/2`, and the
family's `$ds-*-min-height` geometry tokens.

**Python idiom.** none — hand-compose class strings.

**Lifecycle.** All Canonical since ADR-161 task 9 (2026-09-13): the owning
sheet exists, the single-definition rule applies, and the move was
probe-verified computed-style-neutral (generic + density + height-1
geometry contexts) with byte-identical SVG A/B on Home, Console, Settings
and Library.

## Sizing & box-model utilities

**Purpose.** The one-dimensional companions to the semantic component
classes: the tokenized replacement for the legacy runtime
`.styles.<dimension> = <literal>` assignments (ADR-161 task 12, spec 3.10
Python side). Compose them in `classes="..."` instead of assigning dimensions
on widget instances at runtime.

**When-not-to-use.** When a semantic class already encodes the geometry
(e.g. `action-button` carries `$ds-control-height`, `status-area` carries
`$ds-width-full`) — reach for the utility only for the bare dimension itself.
Runtime-computed geometry (measured widths, adaptive row counts) is NOT a
class: those sites keep a runtime assignment through `set_styles(...)`, the
inline-precedence escape hatch, which the `.styles` ratchet deliberately
still allows — the ratchet targets constant literals, not computed layout.
A site that must override an id-scoped rule also uses `set_styles(...)`
(utilities win equal-specificity ties as the bundle's last sheet, but never
beat id selectors where inline styles used to).

**Class inventory.**

- `w-auto` / `h-auto` — auto sizing (`$ds-width-auto` / `$ds-height-auto`).
- `w-full` / `h-full` — 100% fill (`$ds-width-full` / `$ds-height-full`).
- `w-fill` / `h-fill` — one grid share (`$ds-width-fill` / `$ds-height-fill`).
- `w-0` / `h-0` — collapsed axis (`$ds-size-0`).
- `h-1` / `h-2` / `h-3` — fixed row heights (`$ds-size-1/2/3`).
- `p-0` / `m-0` — zero padding / margin (`$ds-space-0`).
- `border-none` — border suppression (keyword `none`; the legacy
  `("none", "transparent")` tuple in disguise).

Defined in `utilities/_helpers.tcss` (the bundle's LAST sheet, so the
utilities win equal-specificity ties). Not every `$ds-size-N` step is a
class: `w-`/`h-N` classes exist only where the Python migration consumed
them; grow the set one entry at a time as new consumers appear.

**States.** None — single-purpose dimension rules.

**Tokens consumed.** `$ds-size-0/1/2/3`, `$ds-space-0`, `$ds-width-auto`,
`$ds-width-full`, `$ds-width-fill`, `$ds-height-auto`, `$ds-height-full`,
`$ds-height-fill`.

**Python idiom.** `Widget(..., classes="status-area h-auto")` — or
`widget.add_class("h-1")` / `widget.remove_class("h-1")` for state toggles.

**Lifecycle.** All Canonical since ADR-161 task 12 (2026-09-13); the classes
replaced 574 ratcheted runtime assignments across the UI layer.
