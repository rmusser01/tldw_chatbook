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
  "deprecated": { "<class>": { "renames_to": "<class>", "ceiling": 0 } }
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
| forms | `components/_forms.tcss` | form-label, form-row, form-col, form-input, form-textarea, form-select, form-checkbox, form-button, form-section-title, form-section-collapsible, form-actions (all Canonical) |
| buttons | `components/_buttons.tcss` | button-group, button-group-left, button-group-center, button-group-right, sidebar-toggle (all Canonical) |
| lists | `components/_lists.tcss` | none — widget contract (ListView/DataTable/OptionList cursor-hover-selected type selectors) |
| dialogs | `components/_dialogs.tcss` | none yet — vocabulary promotes in with the consolidation task |
| status | `components/_status.tcss` | none yet — vocabulary promotes in with the consolidation task |
| navigation | `layout/_sidebars.tcss` | sidebar, sidebar-button, sidebar-header, sidebar-listview, sidebar-section-collapsible (Canonical); nav-button (Canonical, owning sheet `components/_navigation.tcss`) |
| sections | `components/_sections.tcss` | section-title, section-header, subsection-title (all Canonical) |
| messages | `components/_messages.tcss` | none yet — vocabulary promotes in with the consolidation task |
| ds_primitives | `components/_ds_primitives.tcss` | ds-panel, ds-toolbar, ds-field-row, ds-info-callout, ds-approval-card, ds-destination-header (all DRAFT) |

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
- `form-row` — horizontal field-grouping container; stack with `$ds-space-stack`.
- `form-col` — equal-width (`1fr`) column inside a `form-row`; edge columns drop
  their outer `$ds-space-inline` padding.
- `form-input` — single-line `Input`. Rest edge per §2.7 (the compact-field
  `$ds-control-edge` left-edge convention); `:focus` recolours via
  `$ds-input-focus-border` / `$ds-input-focus-accent` / `$ds-input-focus-bg`;
  `.error` modifier marks validation failure.
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

**Lifecycle.** All Canonical (owning sheet `components/_forms.tcss`).

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
`$ds-surface-raised`, `$ds-text-primary`, `$ds-space-section`.

**Python idiom.** none — hand-compose class strings. (`create_button_group`
in `Widgets/form_components.py` exists but is optional, not canonical.)

**Lifecycle.** All Canonical. Note: the `button-group*` definitions currently
sit in `components/_forms.tcss`; they move to the owning sheet
`components/_buttons.tcss` during consolidation (no rename, so no Deprecated
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

**Class inventory.** (Registry classes land here when the vocabulary promotes
into `components/_dialogs.tcss`; today the definitions live in
`features/_media.tcss` and `ConfirmationDialog.DEFAULT_CSS`.)

- `dialog-title` — centered bold heading.
- `dialog-subtitle` — muted secondary line.
- `dialog-message` — body copy above the buttons.
- `dialog-buttons` — horizontal action row; compose with
  `button-group button-group-right` for the standard right-aligned layout.

**States.** Per §2.7 — the dialog itself has no states; the buttons in the
row follow the Button type contract (`$ds-hover-*` / `$ds-focus-*` /
`$ds-disabled-*`).

**Tokens consumed.** None directly today — the vocabulary is structural
(layout, alignment, `text-style: bold`). On promotion the margins tokenize
onto `$ds-space-*`; action states come from the buttons family.

**Python idiom.** `ConfirmationDialog` (`Widgets/confirmation_dialog.py`, 36
importers) for confirm flows; `SafeModalDismissMixin`
(`Widgets/modal_dismissal.py`, 58 of 66 modals) for every modal's
escape/click dismissal contract.

**Lifecycle.** Canonical family (owning sheet `components/_dialogs.tcss`);
registry `"classes"` fills when the promotion lands.

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

**Class inventory.** (Registry classes land here when the vocabulary promotes
into `components/_status.tcss`; today the definitions live in
`components/_forms.tcss`.)

- `status-label` — muted caption above the area (`$ds-text-muted`).
- `status-area` — read-only output surface: `$ds-surface-raised` fill, rounded
  border, `$ds-space-inset` padding.
- `status-item` — per-line status entry; exists today only scoped
  (`ProgressStep .status-item` in `features/_wizards.tcss`) and joins the
  family on promotion.

**States.** Read-only surfaces — no hover/focus/disabled contract of their
own. Semantic colouring maps to the `$ds-status-*` set (readable error text
uses `$ds-status-error-readable`, per task-2230).

**Tokens consumed.** Today: `$ds-text-muted`, `$ds-surface-raised`,
`$ds-space-inset`, `$ds-space-stack`, `$ds-space-section`. Semantic mapping:
the `$ds-status-*` family (`$ds-status-ready`, `$ds-status-running`,
`$ds-status-info`, `$ds-status-warning`, `$ds-status-error`,
`$ds-status-error-readable`, …).

**Python idiom.** `create_status_area` in `Widgets/form_components.py` —
optional; hand-composing the two widgets is equally canonical.

**Lifecycle.** Canonical family (owning sheet `components/_status.tcss`,
currently a header-only stub; the consolidation task fills it and registers
the classes).

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
  sheet `components/_navigation.tcss` (per-class override — the states and
  the `.nav-button-clip-ghost` scroll-edge override live there; base geometry
  comes from `MainNavigationBar.DEFAULT_CSS` and migrates during
  consolidation per the BUNDLED_CSS rule). `NavigationButton.active` /
  `:focus` recolour via `$ds-focus-bg` / `$ds-focus-fg` with bold underline.

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

Today these have four competing definitions across feature sheets
(`features/_chat.tcss`, `features/_evaluation_unified.tcss`,
`features/_tools-settings.tcss` ×2, `components/_shared_components.tcss`);
the consolidation task dissolves them into one definition each in the owning
sheet `components/_sections.tcss` (which is why the registry already points
there).

**States.** Static text — no interactive states.

**Tokens consumed.** The tokenized exemplar definition (`features/_chat.tcss`)
uses `$ds-space-stack` and `$ds-text-strong`; the losing definitions' raw
literals tokenize on promotion (margins → `$ds-space-*`, colours → text/surface
tokens).

**Python idiom.** none — hand-compose class strings.

**Lifecycle.** All Canonical (owning sheet `components/_sections.tcss`,
created by the consolidation task).

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

**Class inventory.** (Registry classes land here when the shared grammar
promotes into `components/_messages.tcss`.)

- `ChatMessage` type contract — full-width auto-height bubble; variant classes
  set speaker/tool surfaces.
- `message-header` — one-row speaker/tool header band.
- `message-text` — scrollable body text (`.tts-generating` modifier marks
  in-flight speech).
- `message-actions` — bottom action strip; `.-generating` hides it until
  generation completes. Buttons take the Button type contract
  (`$ds-focus-bg` / `$ds-focus-fg` on focus).

**States.** §2.7 for the action-strip buttons; the bubbles themselves are
static containers.

**Tokens consumed.** `$ds-focus-bg`, `$ds-focus-fg`, `$ds-surface-raised`
(action-strip button states in the owning sheet). Speaker accents:
`$ds-chat-user-accent` and the `$ds-chat-user-accent-*` /
`$ds-chat-assistant-accent-*` dark/light polarity pairs from
`core/_variables.tcss` (consumed by the console/chat screens); inline
roleplay semantics use the `$ds-chat-rp-*` pairs.

**Python idiom.** none — hand-compose class strings.

**Lifecycle.** Canonical family (owning sheet `components/_messages.tcss`);
registry `"classes"` fills when the shared grammar is carved out.

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

All currently defined in `components/_agentic_terminal.tcss`; the monolith
carve-up moves them (tokenized) into `components/_ds_primitives.tcss`.

**States.** Static containers — interactive states come from whatever
controls are composed inside them (buttons → Button contract).

**Tokens consumed.** `$ds-surface-panel`, `$ds-surface-raised`,
`$ds-text-primary`, `$ds-grid-line`, `$ds-action-focus`, `$ds-status-info`,
`$ds-status-warning`, `$ds-status-approval-required`.

**Python idiom.** none — hand-compose class strings.

**Lifecycle.** All DRAFT — flips to Canonical in the monolith carve-up
(Task 9), when the owning sheet `components/_ds_primitives.tcss` exists and
the single-definition rule can safely apply.
