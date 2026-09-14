# Component-Pattern Library Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Completion record:** `Docs/superpowers/reports/2026-09-14-component-pattern-library-closeout.md` records final evidence and justified deviations from the original recipes below.

**Goal:** Turn ADR-150's token system into a full design system: a governed catalog of canonical component patterns, one winning CSS vocabulary per family, the `_agentic_terminal.tcss` monolith carved up (absorbing TASK-24451), and every legacy literal — sheet dimensions and Python `styles.*` — migrated to tokens with zero-literal end state.

**Architecture:** CSS-class catalog + mechanical governance (no Python builders). A registry (`css/patterns.json`) is the single machine-readable source for governance tests, the catalog doc, and a snapshot-tested pattern gallery. Consolidation proceeds measure → rename → verify, per family, one PR per task. Moves stay within the bundle/per-screen track split; moved and migrated rules arrive fully tokenized.

**Tech Stack:** Textual 8.2.8 (exact pin), pytest + existing pilot helpers (`Tests/textual_test_utils.py`), the repo's CSS build pipeline (`tldw_chatbook/css/build_css.py`), Backlog.md CLI.

**Spec:** `Docs/superpowers/specs/2026-09-13-component-pattern-library-design.md` — the plan argues from the spec; read both. Spec § references below are to that file.

## Global Constraints

- Never edit `tldw_chatbook/css/tldw_cli_modular.tcss` directly. After any `.tcss` source change: `python tldw_chatbook/css/build_css.py` (AGENTS.md hard rule).
- New/renamed sheets are post-ADR-150: no raw hex, no raw numeric `padding`/`margin` (spec §3.5).
- No terminal-convention keybindings on screens (ctrl+c/v/x/s/d/z/a/r/w, ctrl+p/q/f1/f6); the gallery registers via command palette only (ADR-031).
- Targeted tests only (tests touching modified files/functionality); full sweeps require explicit user opt-in (AGENTS.md).
- Live app checks use a scratch profile: `TLDW_CONFIG_PATH=<scratch>/config.toml`, stderr left attached to the pane, absolute `.venv/bin/python` path (`backlog/docs/lessons-live-verification.md`).
- Textual is exact-pinned (`8.2.8`); never bump it in this project.
- Bundle byte ratchet: check `git log --oneline -1 -- Tests/` / run the byte-ratchet test before large sheet changes and coordinate with TASK-31500 (spec §3.9).
- Every task ends with its own commit (conventional style). One task = one PR.
- Sheet membership changes require editing `CSS_MODULES` in `tldw_chatbook/css/build_css.py` (order matters: components before features; `_ds_primitives` first among new component sheets).

---

### Task 1: ADR-161, backlog filing, linkage

**Files:**
- Create: `backlog/decisions/161-component-pattern-library.md`
- Modify: `backlog/docs/design-language.md` (add pointer in §4), `AGENTS.md` (Design Language section)
- Backlog: parent task + subtasks; TASK-24451 supersession note

**Interfaces:**
- Produces: parent task ID referenced by every later task's commit notes; ADR-161 path cited in `design-language.md` and the catalog doc created in Task 3.

- [ ] **Step 1: Write ADR-161** — create `backlog/decisions/161-component-pattern-library.md` with this content (status Accepted, date 2026-09-13, related spec path):

```markdown
# ADR-161: Component-pattern layer of the design system

Status: Accepted
Date: 2026-09-13
Related: ADR-150, Docs/superpowers/specs/2026-09-13-component-pattern-library-design.md

## Decision

The design system gains a component-pattern layer above ADR-150's tokens:

1. Form factor is a CSS-class catalog, not Python builders. Patterns are documented
   classes + structure + state contracts; screens keep hand-composing class strings.
   Evidence: two prior builder libraries died of disuse (0 and 2 importers) while the
   raw class vocabulary reached 300+ uses. Winning Python idioms (SafeModalDismissMixin,
   ConfirmationDialog) are documented as part of their pattern, not replaced.
2. `tldw_chatbook/css/patterns.json` is the machine-readable registry (families,
   owning sheets, class lifecycle). Governance tests enforce: single definition per
   Canonical class (bare-selector rules only in the owning sheet; compound/scoped
   selectors are legal composition), deprecated-name and literal ratchets,
   doc/gallery sync.
3. Carve-up rule: classes promote to sheets matching consumption scope. Moves stay
   within the bundle vs per-screen-source track (Textual per-source $variable scope;
   TASK-15993/16811). New per-screen blocks carry no local $ds-* fallbacks.
4. ADR-150 §6's "migrate opportunistically" policy is AMENDED: the ~6,250 legacy
   dimension literals and ~580 Python `styles.*` visual assignments migrate as
   committed, ratcheted, project-owned work. End state: zero raw literals in sheets
   (recurring values → shared tokens; feature geometry → `$ds-<feature>-*` tokens)
   and zero ad-hoc Python visual assignments.

## Alternatives considered

- Python builder revival — contradicts 170 screens of practice; strands existing UI.
- Opportunistic literal migration (status quo per ADR-150 §6) — unowned and
  unbounded; rejected by the project owner for consistency and long-term stability.
- One big-bang migration pass — unreviewable; ADR-150 already rejected this shape.

## Consequences

- `_variables.tcss` grows feature-scoped `$ds-<feature>-*` tokens (append-mostly).
- Bundle bytes grow from tokenization; sequencing coordinates with TASK-31500.
- `_agentic_terminal.tcss` is dissolved (absorbs TASK-24451); ≤2,000
  comment-stripped lines per sheet enforced on completion.
```

- [ ] **Step 2: File the backlog tasks**

```bash
backlog task create "Component-pattern library: catalog, governance, monolith carve-up, zero-literal migration" -d "Implements Docs/superpowers/specs/2026-09-13-component-pattern-library-design.md (ADR-161). Parent task; one subtask per plan task." --ac "9-family catalog + registry live,Governance tests green and ratchets pinned,Monolith dissolved under 2,000-line ceiling,TASK-24451 superseded and closed,Zero dimension literals in sheets,Zero Python ad-hoc visual style assignments,Gallery snapshots dark+light green"
```

Note the printed task ID (`task-<N>`), then create subtasks (repeat for Tasks 2–13 of this plan, editing title/AC to match):

```bash
backlog task create "Delete dead CSS/widget code" -p task-<N> --ac "Files gone,Grep clean of dynamic refs,Bundle rebuilt smaller"
backlog task create "Registry + catalog doc" -p task-<N> --ac "patterns.json loads,Governance-test schema contract holds,Catalog exemplar entries complete"
# ... one per plan Task 2..13
```

Mark supersession:

```bash
backlog task edit 24451 --notes "Superseded by task-<N> (component-pattern library; ADR-161). The monolith split lands as plan Tasks 9-10. Close when the size-ceiling test ships."
```

- [ ] **Step 3: Add linkage** — in `backlog/docs/design-language.md` §4, append one line after the recipe intro: `The reusable component vocabulary itself is cataloged in backlog/docs/component-patterns.md (ADR-161); compose from the catalog first.` In `AGENTS.md`'s Design Language section, add the same path to the required-reading line.
- [ ] **Step 4: Commit** — `git add backlog/decisions/161-component-pattern-library.md backlog/docs/design-language.md AGENTS.md && git commit -m "docs(adr): ADR-161 component-pattern layer; ADR-150 s6 migration policy amended"`

---

### Task 2: Dead code deletion

**Files:**
- Delete: `tldw_chatbook/css/components/_unified_sidebar.tcss`, `tldw_chatbook/Widgets/base_components.py`, `tldw_chatbook/Widgets/empty_state.py`, `tldw_chatbook/Widgets/loading_states.py`, `tldw_chatbook/Widgets/toast_notification.py`, `tldw_chatbook/Widgets/custom_list_items.py`, `tldw_chatbook/Widgets/enhanced_sidebar.py`, `tldw_chatbook/Widgets/detailed_progress.py`
- Modify (if they exist): `tldw_chatbook/Widgets/__init__.py` re-exports of deleted names

**Interfaces:** none consumed; produces byte headroom the gallery (Task 5) spends.

- [ ] **Step 1: Verify zero live references, including dynamic**

```bash
for m in base_components empty_state loading_states toast_notification custom_list_items enhanced_sidebar detailed_progress; do
  grep -rn "$m" tldw_chatbook/ --include="*.py" | grep -v "Widgets/$m.py" | grep -v "^\s*#"; done
grep -rn "_unified_sidebar" tldw_chatbook/ Tests/
grep -rn -E "importlib|__import__" tldw_chatbook/Widgets/ | grep -iE "base_components|empty_state|loading|toast|custom_list|enhanced_sidebar|detailed"
```

Expected: no hits (comment lines excluded). Any hit → that file is a consumer; surface it before deleting.

- [ ] **Step 2: Delete files, fix `__init__.py`, rebuild, verify**

```bash
git rm tldw_chatbook/css/components/_unified_sidebar.tcss tldw_chatbook/Widgets/base_components.py tldw_chatbook/Widgets/empty_state.py tldw_chatbook/Widgets/loading_states.py tldw_chatbook/Widgets/toast_notification.py tldw_chatbook/Widgets/custom_list_items.py tldw_chatbook/Widgets/enhanced_sidebar.py tldw_chatbook/Widgets/detailed_progress.py
python tldw_chatbook/css/build_css.py
python -m pytest Tests/UI/test_design_token_governance.py Tests/UI/test_css_bundle_sync_guard.py -q
wc -c tldw_chatbook/css/tldw_cli_modular.tcss   # record: must be ≤ previous
```

- [ ] **Step 3: Commit** — `git commit -am "chore(css): delete dead builder/widget modules and orphaned _unified_sidebar sheet"`

---

### Task 3: Registry + catalog doc

**Files:**
- Create: `tldw_chatbook/css/patterns.json`, `backlog/docs/component-patterns.md`

**Interfaces:**
- Produces: `patterns.json` schema (consumed verbatim by governance tests in Task 6):

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

- [ ] **Step 1: Write `patterns.json`** with the nine families at their CURRENT winners (owning sheets that exist today; new sheets from Tasks 4/9 update this file):

```json
{
  "version": 1,
  "families": {
    "forms":        { "owning_sheet": "components/_forms.tcss", "classes": {
        "form-label": {"status": "canonical"}, "form-row": {"status": "canonical"},
        "form-col": {"status": "canonical"}, "form-input": {"status": "canonical"},
        "form-textarea": {"status": "canonical"}, "form-select": {"status": "canonical"},
        "form-checkbox": {"status": "canonical"}, "form-button": {"status": "canonical"},
        "form-section-title": {"status": "canonical"}, "form-section-collapsible": {"status": "canonical"},
        "form-actions": {"status": "canonical"} } },
    "buttons":      { "owning_sheet": "components/_buttons.tcss", "classes": {
        "button-group": {"status": "canonical"}, "button-group-left": {"status": "canonical"},
        "button-group-center": {"status": "canonical"}, "button-group-right": {"status": "canonical"},
        "sidebar-toggle": {"status": "canonical"} } },
    "lists":        { "owning_sheet": "components/_lists.tcss",
        "widget_contract": "ListView/DataTable/OptionList cursor-hover-selected type selectors; no public classes", "classes": {} },
    "dialogs":      { "owning_sheet": "components/_dialogs.tcss", "classes": {} },
    "status":       { "owning_sheet": "components/_status.tcss", "classes": {} },
    "navigation":   { "owning_sheet": "layout/_sidebars.tcss", "classes": {
        "sidebar": {"status": "canonical"}, "sidebar-button": {"status": "canonical"},
        "sidebar-header": {"status": "canonical"}, "sidebar-listview": {"status": "canonical"},
        "sidebar-section-collapsible": {"status": "canonical"}, "nav-button": {"status": "canonical"} } },
    "sections":     { "owning_sheet": "components/_sections.tcss", "classes": {
        "section-title": {"status": "canonical"}, "section-header": {"status": "canonical"},
        "subsection-title": {"status": "canonical"} } },
    "messages":     { "owning_sheet": "components/_messages.tcss", "classes": {} },
    "ds_primitives": { "owning_sheet": "components/_ds_primitives.tcss", "classes": {
        "ds-panel": {"status": "canonical"}, "ds-toolbar": {"status": "canonical"},
        "ds-field-row": {"status": "canonical"}, "ds-info-callout": {"status": "canonical"},
        "ds-approval-card": {"status": "canonical"}, "ds-destination-header": {"status": "canonical"} } }
  },
  "deprecated": {}
}
```

- [ ] **Step 2: Write `backlog/docs/component-patterns.md`** — header citing ADR-161 + the spec, the fixed entry template from spec §3.2 (Purpose / When-not-to-use / Structure / Class inventory / Tokens / Python idiom / Lifecycle), then entries. Write these two entries in FULL as the exemplars:

**forms & fields:** purpose = dense settings-style forms and dialogs; structure snippet:

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

states = §2.7 contract via `.form-input` rest edge (`$ds-control-edge`), `$ds-hover-*`, `$ds-focus-*`, `$ds-disabled-*`; tokens = `$ds-space-stack/-inline/-section/-inset`, `$ds-control-height`, `$ds-textarea-min-height`; when-not-to-use = console transcript rows (messages family) and one-off prompts. Python idiom = `Widgets/form_components.py` builders exist (2 importers) — optional, not canonical.

**dialogs & modals:** purpose = any modal; structure = `class X(SafeModalDismissMixin, ModalScreen[None])` with `Container .dialog-frame`-style body, `Label(..., classes="dialog-title")`, actions row `classes="dialog-buttons button-group button-group-right"`; Python idiom = `ConfirmationDialog` (36 importers) for confirm flows; states per §2.7; when-not-to-use = non-modal popovers.

Then stub-free entries for the other seven families populated from the inventory data (buttons, lists-contract, status, navigation, sections, messages, ds-primitives) using the same template with the class lists from `patterns.json` above.

- [ ] **Step 3: Sanity check + commit** — `python -c "import json; d=json.load(open('tldw_chatbook/css/patterns.json')); assert set(d['families']) == {'forms','buttons','lists','dialogs','status','navigation','sections','messages','ds_primitives'}"` then `git add ... && git commit -m "feat(design): component-pattern registry and catalog doc (ADR-161)"`

---

### Task 4: CSS-only dedupe and promotions (no renames)

**Files:**
- Create: `tldw_chatbook/css/components/_sections.tcss`
- Modify: `components/_forms.tcss` (remove duplicate `.form-input` at one of lines 43/306 — diff first, keep the token-clean one; move `.button-group*` rules to `_buttons.tcss`), `components/_dialogs.tcss` (add `.dialog-title`, `.dialog-buttons`), `features/_media.tcss` (delete its `.dialog-title`/`.dialog-buttons` definitions), `features/_evaluation_unified.tcss` (delete `.action-button`, re-added in `_buttons.tcss`), `Widgets/confirmation_dialog.py` (delete the `.dialog-title` rule from its class-level CSS string — attr is `BUNDLED_CSS` or `DEFAULT_CSS`; check the file), `tldw_chatbook/css/build_css.py` (insert `components/_sections.tcss` into `CSS_MODULES` after `components/_forms.tcss`), `tldw_chatbook/css/patterns.json` (dialogs gains `.dialog-title`/`.dialog-buttons`; buttons gains `.action-button` + `.button-group*`)

**Interfaces:** consumes `patterns.json` schema (Task 3); produces the canonical single definitions the gallery (Task 5) renders.

- [ ] **Step 1: Diff competing definitions and pick winners**

```bash
for c in section-title section-header form-input dialog-title; do
  grep -rn "^\.$c\b\|^\.$c " tldw_chatbook/css/{core,layout,components,features,utilities}/*.tcss; done
```

Winner rule (spec §3.6): the definition used by most screens; ties → the one already in a components sheet; both must then be tokenized to the winner's values. For `.section-title` the expected winner is `components/_shared_components.tcss`'s `.section-header` companion set — create `_sections.tcss` containing the winning `.section-title`, `.section-header`, `.subsection-title` (tokenize on arrival: colors→`$ds-*`, padding/margin→`$ds-space-*`), delete the other three definitions (`features/_tools-settings.tcss` ×2 — one is line ~96 — and `features/_evaluation_unified.tcss` ~120).

- [ ] **Step 2: Apply moves, rebuild, verify**

```bash
python tldw_chatbook/css/build_css.py
python -m pytest Tests/UI/test_design_token_governance.py Tests/UI/test_css_bundle_sync_guard.py -q
python -m pytest Tests/UI/ -k "media or eval or settings" -q     # screens whose visuals may shift
wc -c tldw_chatbook/css/tldw_cli_modular.tcss                    # must be < previous (dedupe is net-negative)
```

- [ ] **Step 3: Live-verify screens that changed** (spec §3.6: dedupe winners are visual changes): tmux recipe on tools-settings + evaluation screens; scratch `TLDW_CONFIG_PATH`; confirm section headers still render hierarchy (title ↔ content separation) in both dark and light themes.
- [ ] **Step 4: Update registry + catalog entries** (mark moved classes canonical under new owners) and **commit**: `git commit -m "refactor(css): single definitions for section/dialog/button vocabulary; promotions to owning sheets"`

---

### Task 5: Pattern gallery + snapshot tests

**Files:**
- Create: `tldw_chatbook/Widgets/pattern_gallery.py`, `tldw_chatbook/css/features/_pattern_gallery.tcss`, `Tests/UI/test_pattern_gallery_snapshots.py`, `Tests/UI/snapshots/pattern_gallery/dark.svg`, `Tests/UI/snapshots/pattern_gallery/light.svg`
- Modify: `tldw_chatbook/css/build_css.py` (`features/_pattern_gallery.tcss` appended to `CSS_MODULES`), `tldw_chatbook/app.py` (import + register provider next to `ConsoleCommandProvider` at line ~590 import block and the provider list at ~7573)

**Interfaces:**
- Produces: `PatternGalleryScreen` (import path `tldw_chatbook.Widgets.pattern_gallery`); gallery source file path is consumed by the governance sync test (Task 6) as the constant `GALLERY_SOURCE`.

- [ ] **Step 1: Write the gallery screen** (`tldw_chatbook/Widgets/pattern_gallery.py`):

```python
"""Design-system pattern gallery (ADR-161). Command-palette entry only."""
from __future__ import annotations
from textual.app import ComposeResult
from textual.containers import Container, Vertical, Horizontal
from textual.widgets import Button, Checkbox, Input, Label, ListView, ListItem, Select, Static, TextArea
from textual.binding import Binding
from textual.screen import Screen


class PatternGalleryScreen(Screen):
    BINDINGS = [Binding("escape", "app.pop_screen", "Back")]
    CSS_HIGHLIGHT = tuple()  # styles live in features/_pattern_gallery.tcss

    def compose(self) -> ComposeResult:
        with Vertical(id="pg-root"):
            yield Label("Pattern Gallery — canonical component families (ADR-161)", classes="section-title")
            # forms
            yield Label("Forms & fields", classes="section-header")
            with Container(classes="form-col"):
                yield Label("Model", classes="form-label")
                yield Input(placeholder="gpt-4o", classes="form-input")
                yield Label("System prompt", classes="form-label")
                yield TextArea(classes="form-textarea")
                yield Select([("option-a", "a"), ("option-b", "b")], classes="form-select")
                yield Checkbox("Stream response", classes="form-checkbox")
            # buttons
            yield Label("Buttons & action rows", classes="section-header")
            with Container(classes="button-group button-group-left"):
                yield Button("Primary", classes="form-button")
                yield Button("Disabled", disabled=True)
            # lists
            yield Label("Lists & tables", classes="section-header")
            yield ListView(
                ListItem(Label("ready — succeeded row")),
                ListItem(Label("running — active row")),
                ListItem(Label("blocked — failed row")),
            )
            # dialogs (structure preview, not a pushed modal)
            yield Label("Dialogs & modals", classes="section-header")
            with Container(classes="pg-dialog-frame"):
                yield Label("Confirm deletion", classes="dialog-title")
                yield Static("This cannot be undone.", classes="help-text")
                with Container(classes="dialog-buttons button-group button-group-right"):
                    yield Button("Cancel")
                    yield Button("Delete")
            # status
            yield Label("Status, empty & loading", classes="section-header")
            with Container(classes="status-area"):
                yield Label("ready", classes="status-label")
                yield Label("running", classes="status-label")
                yield Label("approval-required", classes="status-label")
            # navigation
            yield Label("Navigation & sidebars", classes="section-header")
            with Container(classes="sidebar"):
                yield Label("Sidebar header", classes="sidebar-header")
                yield Button("Nav item", classes="sidebar-button")
            # messages
            yield Label("Chat & message rendering", classes="section-header")
            yield Static("user: speaks with $ds-chat-user-accent", classes="pg-msg-user")
            yield Static("assistant: speaks with $ds-chat-assistant-accent", classes="pg-msg-assistant")
            # ds-primitives
            yield Label("ds-primitives", classes="section-header")
            with Container(classes="ds-toolbar"):
                yield Button("Tool A")
                yield Button("Tool B")
            with Container(classes="ds-panel"):
                yield Label("Panel content", classes="ds-field-row")
```

`features/_pattern_gallery.tcss` (token-clean; only layout glue — patterns come from their owning sheets):

```css
/* Pattern gallery glue (ADR-161). No family styling here — the gallery must
   render the canonical sheets' output, not its own. */
#pg-root { padding: $ds-space-inset; }
.pg-dialog-frame { border: round $ds-grid-line; padding: $ds-space-inset; }
.pg-msg-user { color: $ds-chat-user-accent; }
.pg-msg-assistant { color: $ds-chat-assistant-accent; }
```

- [ ] **Step 2: Register the palette provider** — in `app.py` import block (~line 590) add `from .Widgets.pattern_gallery import PatternGalleryScreen  # noqa: E402` and a provider mirroring `LibraryIngestProvider`'s shape (new class in `pattern_gallery.py`):

```python
from textual.command import Hit, Hits, Provider

class PatternGalleryProvider(Provider):
    COMMANDS = (("Design System: Pattern Gallery", "open_pattern_gallery",
                 "Browse every canonical component pattern live"),)

    async def discover(self) -> Hits:
        for text, _id, help_text in self.COMMANDS:
            yield Hit(1.0, text, lambda: self.screen.app.push_screen(PatternGalleryScreen()), help=help_text)

    async def search(self, query: str) -> Hits:
        matcher = self.matcher(query)
        for text, _id, help_text in self.COMMANDS:
            if (score := matcher.match(text)) > 0:
                yield Hit(score, matcher.highlight(text),
                          lambda: self.screen.app.push_screen(PatternGalleryScreen()), help=help_text)
```

Add `PatternGalleryProvider` to the provider list where `ConsoleCommandProvider` appears (~line 7573).
- [ ] **Step 3: Write the snapshot test** (`Tests/UI/test_pattern_gallery_snapshots.py`):

```python
"""Gallery SVG snapshots (spec 3.4): fixed size, textual-dark + textual-light."""
from __future__ import annotations
import os, re
from pathlib import Path
import pytest
from textual.app import App
from tldw_chatbook.Widgets.pattern_gallery import PatternGalleryScreen

FIXTURES = Path(__file__).parent / "snapshots" / "pattern_gallery"

def _normalize(svg: str) -> str:
    svg = re.sub(r"<\?xml[^>]*\?>", "", svg)
    svg = re.sub(r"<svg[^>]*>", "<svg>", svg)
    svg = re.sub(r'font-family="[^"]*"', 'font-family="X"', svg)
    return svg.strip()

class _GalleryApp(App):
    def __init__(self, theme: str) -> None:
        super().__init__()
        self._theme = theme
    def on_mount(self) -> None:
        self.theme = self._theme
    def compose(self):
        yield PatternGalleryScreen()

@pytest.mark.parametrize("theme", ["textual-dark", "textual-light"])
@pytest.mark.asyncio
async def test_gallery_snapshot(theme: str) -> None:
    app = _GalleryApp(theme)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        svg = _normalize(app.export_screenshot(simplify=True))
    fixture = FIXTURES / f"{theme.removeprefix('textual-')}.svg"
    if os.environ.get("UPDATE_SNAPSHOTS"):
        fixture.parent.mkdir(parents=True, exist_ok=True)
        fixture.write_text(svg, encoding="utf-8")
    assert svg == fixture.read_text(encoding="utf-8").strip(), (
        f"Pattern rendering changed under {theme}. If intended: UPDATE_SNAPSHOTS=1 "
        "and review the diff; spec 5.9 requires token-value changes to be live-verified.")
```

- [ ] **Step 4: Generate fixtures and verify** — `UPDATE_SNAPSHOTS=1 python -m pytest Tests/UI/test_pattern_gallery_snapshots.py -q` then rerun without the env var (PASS twice), and manually inspect both SVGs render every family.
- [ ] **Step 5: Commit** — `git commit -m "feat(design): pattern gallery screen with dark/light SVG snapshots (ADR-161)"`

---

### Task 6: Governance tests + ratchet pinning

**Files:**
- Create: `Tests/UI/test_component_pattern_governance.py`, `Tests/UI/pin_pattern_ratchets.py`, `Tests/UI/pattern_ratchet_baseline.json`

**Interfaces:**
- Consumes: `patterns.json` schema (Task 3), `GALLERY_SOURCE` = `tldw_chatbook/Widgets/pattern_gallery.py`, catalog doc path `backlog/docs/component-patterns.md`, `CSS_MODULES` from `tldw_chatbook.css.build_css`, `iter_blocks`/`WIDGET_ATTR`/`SCREEN_ATTR` from `tldw_chatbook.css.widget_css`.

- [ ] **Step 1: Write the pin script** (`Tests/UI/pin_pattern_ratchets.py`):

```python
"""Pin current literal/style counts as the ratchet baseline (spec 3.7/3.10)."""
from __future__ import annotations
import json, re
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
CSS = ROOT / "tldw_chatbook/css"
PKG = ROOT / "tldw_chatbook"
_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)
_DIM = re.compile(r"\b(?:padding|margin|width|height)(?:-(?:top|right|bottom|left))?\s*:\s*[0-9]")
_PYSTYLE = re.compile(r"\.styles\.(?:background|color|border\w*|width|height|padding\w*|margin\w*|opacity\w*)\s*=\s*[^=]")

def sheets() -> list[Path]:
    out = []
    for sub in ("core", "layout", "components", "features", "utilities"):
        out += sorted((CSS / sub).glob("*.tcss"))
    return out

baseline = {"dimensions": {}, "python_styles": {}}
for s in sheets():
    n = len(_DIM.findall(_COMMENT.sub("", s.read_text(encoding="utf-8"))))
    if n:
        baseline["dimensions"][str(s.relative_to(CSS))] = n
for p in sorted(PKG.rglob("*.py")):
    n = len(_PYSTYLE.findall(p.read_text(encoding="utf-8", errors="ignore")))
    if n:
        baseline["python_styles"][str(p.relative_to(PKG))] = n
Path(__file__).parent.joinpath("pattern_ratchet_baseline.json").write_text(json.dumps(baseline, indent=2, sort_keys=True))
print(json.dumps({k: sum(v.values()) for k, v in baseline.items()}))
```

Run it, record the totals in the task notes, commit the baseline.

- [ ] **Step 2: Write the governance test** (`Tests/UI/test_component_pattern_governance.py`):

```python
"""Component-pattern governance (ADR-161, spec 3.7)."""
from __future__ import annotations
import json, re
from pathlib import Path
import tldw_chatbook
from tldw_chatbook.css.build_css import CSS_MODULES
from tldw_chatbook.css.widget_css import iter_blocks, WIDGET_ATTR, SCREEN_ATTR

ROOT = Path(__file__).resolve().parents[2]
CSS = ROOT / "tldw_chatbook/css"
PKG = Path(tldw_chatbook.__file__).parent
REGISTRY = json.loads((CSS / "patterns.json").read_text(encoding="utf-8"))
CATALOG = ROOT / "backlog/docs/component-patterns.md"
GALLERY = PKG / "Widgets/pattern_gallery.py"
BASELINE = json.loads((Path(__file__).parent / "pattern_ratchet_baseline.json").read_text())

_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)
_RULE = re.compile(r"([^{}]+)\{([^{}]*)\}")
_DIM = re.compile(r"\b(?:padding|margin|width|height)(?:-(?:top|right|bottom|left))?\s*:\s*[0-9]")
_PYSTYLE = re.compile(r"\.styles\.(?:background|color|border\w*|width|height|padding\w*|margin\w*|opacity\w*)\s*=\s*[^=]")

def _sheet_sources() -> dict[str, str]:
    out = {}
    for sub in ("core", "layout", "components", "features", "utilities"):
        for p in sorted((CSS / sub).glob("*.tcss")):
            out[str(p.relative_to(CSS))] = _COMMENT.sub("", p.read_text(encoding="utf-8"))
    return out

def _bundled_sources() -> dict[str, str]:
    out = {}
    for attr in (WIDGET_ATTR, SCREEN_ATTR):
        for block in iter_blocks(PKG, attr):
            key = f"{block.module}:{getattr(block, 'cls', block)}:{attr}"
            out[key] = _COMMENT.sub("", block.css)
    return out

def _bare_definitions(css: str) -> set[str]:
    """Classes defined by a bare selector (one class token + pseudo suffixes)."""
    found = set()
    for m in _RULE.finditer(css):
        for part in m.group(1).split(","):
            sel = part.strip()
            if re.fullmatch(r"\.[A-Za-z_][\w-]*(?::[a-zA-Z-]+)*", sel):
                found.add(sel.split(":")[0][1:])
    return found

def _canonical() -> dict[str, str]:
    out = {}
    for fam, spec_ in REGISTRY["families"].items():
        for cls in spec_["classes"]:
            if spec_["classes"][cls]["status"] == "canonical":
                out[cls] = spec_["owning_sheet"]
    return out

def test_canonical_classes_defined_only_in_owning_sheet():
    canonical = _canonical()
    sources = {**{k: v for k, v in _sheet_sources().items()}, **_bundled_sources()}
    offenders = {}
    for cls, owner in canonical.items():
        for src, css in sources.items():
            if cls in _bare_definitions(css) and src != owner:
                offenders.setdefault(cls, []).append(src)
    assert not offenders, f"Canonical classes redefined outside their owning sheet: {offenders}"

def test_deprecated_names_ratchet_down():
    for cls, meta in REGISTRY["deprecated"].items():
        count = 0
        for p in PKG.rglob("*.py"):
            count += len(re.findall(rf"\b{re.escape(cls)}\b", p.read_text(encoding="utf-8", errors="ignore")))
        assert count <= meta["ceiling"], f"{cls}: {count} uses > ceiling {meta['ceiling']}"

def test_catalog_and_gallery_sync():
    doc = CATALOG.read_text(encoding="utf-8")
    gallery = GALLERY.read_text(encoding="utf-8")
    for cls in _canonical():
        assert cls in doc, f"{cls} missing from catalog doc"
        assert cls in gallery, f"{cls} missing from pattern gallery"

def test_dimension_literal_ratchet():
    for sheet, css in _sheet_sources().items():
        count = len(_DIM.findall(css))
        assert count <= BASELINE["dimensions"].get(sheet, 0), f"{sheet}: {count} > {BASELINE['dimensions'].get(sheet, 0)}"

def test_python_style_ratchet():
    for p in sorted(PKG.rglob("*.py")):
        rel = str(p.relative_to(PKG))
        count = len(_PYSTYLE.findall(p.read_text(encoding="utf-8", errors="ignore")))
        assert count <= BASELINE["python_styles"].get(rel, 0), f"{rel}: {count} > baseline"
```

Note: `iter_blocks` yields `BundledBlock` objects — check `tldw_chatbook/css/widget_css.py:121` for exact attribute names (`module`, class name, `css` text) and adjust the `getattr` in `_bundled_sources` to the real fields before running.

- [ ] **Step 3: Fallback-ban manifest** — scan current per-screen blocks: `python -c "from pathlib import Path; import tldw_chatbook; from tldw_chatbook.css.widget_css import iter_blocks, SCREEN_ATTR; pkg=Path(tldw_chatbook.__file__).parent; [print(b.module) for b in iter_blocks(pkg, SCREEN_ATTR) if __import__('re').search(r'\\\$ds-[a-z0-9-]+\s*:', b.css)]"` → paste the printed module list into the test as `FALLBACK_ALLOWED: frozenset({...})` and add:

```python
def test_no_new_local_ds_fallbacks():
    for key, css in _bundled_sources().items():
        if not key.endswith(SCREEN_ATTR):
            continue
        module = key.split(":")[0]
        has_fallback = re.search(r"\$ds-[a-z0-9-]+\s*:", css)
        if has_fallback:
            assert module in FALLBACK_ALLOWED, f"{module} adds a local $ds-* fallback (TASK-15993 trap)"
```

- [ ] **Step 4: Run all governance tests green** — `python -m pytest Tests/UI/test_component_pattern_governance.py -q` (the sync test passes because Task 5's gallery references every canonical class in `patterns.json`; if a family has zero canonical classes — lists, dialogs pre-Task-4, status, messages — it contributes nothing).
- [ ] **Step 5: Commit** — `git commit -m "test(design): component-pattern governance — single-definition, ratchets, sync checks (ADR-161)"`

---

### Task 7: Label consolidation (the rename exemplar)

**Files:**
- Modify: `components/_forms.tcss`, `components/_agentic_terminal.tcss`, `layout/_sidebars.tcss`, `components/_unified_sidebar`-successor sheets holding `.setting-label`/`.field-label`, ~60 Python files under `tldw_chatbook/UI/` + `Widgets/`, matching files under `Tests/`, `tldw_chatbook/css/patterns.json`, `backlog/docs/component-patterns.md`

**Interfaces:**
- Produces: the rename procedure reused verbatim in Tasks 8–9, and deprecated registry entries (`settings-input-label`, `setting-label`, `settings-label`, `field-label`, `form-field-label` → `form-label`) with ceilings pinned at their post-rename counts.

- [ ] **Step 1: Measure equivalence** — for each loser class vs `.form-label`, print the declaration diff:

```bash
python - <<'EOF'
import re
from pathlib import Path
css = "\n".join(p.read_text() for p in Path("tldw_chatbook/css").rglob("*.tcss"))
for name in ("form-label", "settings-input-label", "setting-label", "settings-label", "field-label", "form-field-label"):
    for m in re.finditer(rf"^[^\n]*\.{re.escape(name)}\b[^{{]*\{{([^}}]*)\}}", css, re.M):
        print(f"--- .{name} ---\n{m.group(1).strip()}")
EOF
```

Decision rule (spec §3.6): declaration-identical (modulo tokens) → rename to `form-label`; differs only in density/size → rename to `form-label` AND add `density-compact` (existing modifier) where the compact look must persist; differs semantically → keep as a documented variant class in the forms family and record it canonical in the registry instead. **Geometry check**: render each variant once in a height-1 row in a scratch pilot run and confirm the label text paints (compositor text non-empty) — a "equivalent" pair that fails this is NOT equivalent (TASK-13154.1).

- [ ] **Step 2: Scripted rename** (word-boundary, Python + tcss + Tests atomically):

```bash
python - <<'EOF'
import re
from pathlib import Path
MAP = {"settings-input-label": "form-label", "setting-label": "form-label",
       "settings-label": "form-label", "field-label": "form-label",
       "form-field-label": "form-label"}
targets = list(Path("tldw_chatbook").rglob("*.py")) + list(Path("tldw_chatbook/css").rglob("*.tcss")) + list(Path("Tests").rglob("*.py"))
for p in targets:
    text = original = p.read_text(encoding="utf-8", errors="ignore")
    for old, new in MAP.items():
        text = re.sub(rf"\b{re.escape(old)}\b", new, text)
    if text != original:
        p.write_text(text, encoding="utf-8")
        print("edited", p)
EOF
```

- [ ] **Step 3: CSS-side consolidation** — delete losing definitions (now unreferenced), ensure `_forms.tcss` holds the single `.form-label` (plus documented compact variant if Step 1 kept one); rebuild; registry: move the losers into `"deprecated"` with `ceiling` = their remaining count in `tldw_chatbook/` (expect 0; any stragglers in comments are fine — the ratchet counts code only if you keep `\b` word matches outside comments an issue, adjust ceiling to measured).
- [ ] **Step 4: Verify per spec §3.6** — governance suite green; `grep -rn "settings-input-label\|setting-label\|field-label" tldw_chatbook/` returns only comment/prose hits; targeted tests: `python -m pytest Tests/UI/ -k "settings or console or sidebar or ingest" -q`; gallery snapshots still green; SVG A/B on the top-3 screens by old-name occurrence (find them: `grep -rc "settings-input-label" tldw_chatbook/UI/ | sort -t: -k2 -rn | head -3` — run BEFORE the rename and note the files).
- [ ] **Step 5: Commit** — `git commit -m "refactor(design): single form-label vocabulary across screens (ratcheted)"`

---

### Task 8: Remaining family consolidations

**Files:** per family — owning sheets from `patterns.json`, their current definition sites, Python/Tests references.

**Interfaces:** consumes the Task 7 rename procedure verbatim.

One sub-batch per family, each its own commit; all are renames or sheet moves with no visual redesign:

- [ ] **8a status**: move `.status-area`/`.status-label` rules from `components/_forms.tcss` and `.status-item` from its feature sites into `components/_status.tcss` (tokenized); registry gains them canonical; no Python renames (same class names).
- [ ] **8b navigation**: single `.sidebar-button` definition (survivor: `layout/_sidebars.tcss`); tokenize on arrival; registry check.
- [ ] **8c messages**: extract the classes shared by Chat window + Console transcripts (speaker accents, tool-call/result wrappers — identify via `grep -rn "chat-user\|chat-assistant\|tool-call\|tool-result" tldw_chatbook/css/ | head -30`) into `components/_messages.tcss`; feature-specific transcript classes stay in their feature sheets; registry lists the shared set canonical.
- [ ] **8d verification per batch**: rebuild → governance + bundle-sync green → targeted feature tests → gallery snapshot green (gallery already renders these families).

---

### Task 9: Monolith carve-up, batch 1 — ds-primitives + settings vocabulary

**Files:**
- Create: `tldw_chatbook/css/components/_ds_primitives.tcss`
- Modify: `components/_agentic_terminal.tcss` (remove extracted blocks), `components/_forms.tcss` (documented compact field variants, if Step 1 says so), `build_css.py` (`components/_ds_primitives.tcss` FIRST among component sheets), `patterns.json`, catalog doc

**Interfaces:** produces the ds-primitives canonical home consumed by the gallery sync test (gallery already references these classes — Task 5).

- [ ] **Step 1: Extract** the ds-primitives block (`.ds-panel`, `.ds-toolbar`, `.ds-field-row`, `.ds-info-callout`, `.ds-approval-card`, `.ds-destination-header`, their `:hover`/state and `.density-*` variants — located around `components/_agentic_terminal.tcss:30-112`) into the new sheet, tokenizing on arrival (colors → `$ds-*`; padding/margin → `$ds-space-*`; feature geometry → new `$ds-<feature>-*` tokens in `core/_variables.tcss`, e.g. `$ds-toolbar-height`).
- [ ] **Step 2: settings-* decision** — run the Task 7 measurement harness for `.settings-input-row`, `.settings-input-label`-family survivors, `.settings-compact-input`, `.settings-detail-row`, `.settings-status-row` vs the forms family. Expected outcome per inventory: they are the compact field variant → move the winners into `components/_forms.tcss` as documented variants (registry canonical), rename Python references via the Task 7 script, deprecate old names with ceilings.
- [ ] **Step 3: Rebuild + verify** — `python tldw_chatbook/css/build_css.py`; governance + sync-guard green; `python -m pytest Tests/UI/ -k "console or settings or library or home" -q`; gallery snapshots green; SVG A/B on Console, Settings, Library, Home (the monolith's consumers — intra-bundle order changed for these rules).
- [ ] **Step 4: Commit** — `git commit -m "refactor(css): extract ds-primitives and settings field vocabulary from the agentic monolith"`

---

### Task 10: Monolith carve-up, batch 2 — console/library/destination + ceiling test + TASK-24451 closure

**Files:**
- Create: `tldw_chatbook/css/features/_console.tcss`, `features/_library.tcss`, `layout/_destination.tcss`, `features/_home.tcss` (as needed by what remains)
- Modify: `components/_agentic_terminal.tcss` (dissolved to Console chrome only), `layout/_panes.tcss` (its `.destination-section` definition moves out), `build_css.py` (new sheets: `layout/_destination.tcss` in the layout group; feature sheets in the features group), `Tests/UI/test_component_pattern_governance.py` (add the ceiling test), `patterns.json`

- [ ] **Step 1: Move by prefix** — `console-*` (~200 classes) → `features/_console.tcss`; `library-*` → `features/_library.tcss`; `home-*` → `features/_home.tcss`; `destination-*` (both the `_panes.tcss` and monolith definitions, deduped to one) → `layout/_destination.tcss`. Every moved rule tokenized on arrival (spec §3.5). Stay within the bundle track — the per-screen `screen_agentic_*.tcss` sources are NOT touched.
- [ ] **Step 2: Add the size-ceiling test** (ships here, per spec §3.5):

```python
def test_sheet_size_ceiling():
    """No CSS_MODULES sheet exceeds 2,000 comment-stripped lines (TASK-24451 close-out)."""
    for rel in CSS_MODULES:
        text = _COMMENT.sub("", (CSS / rel).read_text(encoding="utf-8"))
        lines = [ln for ln in text.splitlines() if ln.strip()]
        assert len(lines) <= 2000, f"{rel}: {len(lines)} active lines > 2000"
```

- [ ] **Step 3: Verify + byte accounting** — rebuild; governance (incl. ceiling) + sync-guard green; `python -m pytest Tests/UI/ -k "console or library or destination or home" -q`; gallery snapshots; record `wc -c` of the bundle before/after in the task notes (moves are byte-neutral, tokenization slightly positive).
- [ ] **Step 4: Close TASK-24451** — `backlog task edit 24451 -s Done --notes "Superseded by task-<parent>: monolith dissolved, ceiling test shipped (plan Task 10)."` (verify AC of 24451 are met or moved to the parent first).
- [ ] **Step 5: Commit** — `git commit -m "refactor(css): dissolve _agentic_terminal grab-bag into console/library/home/destination sheets; 2,000-line ceiling (closes TASK-24451)"`

---

### Task 11: In-place literal migration (zero-literal end state)

**Files:** every remaining sheet with `dimensions` entries in `Tests/UI/pattern_ratchet_baseline.json`, plus `core/_variables.tcss` (new tokens).

**Interfaces:** consumes the Task 6 ratchet (each sheet's count must hit 0) and the `_chat.tcss` exemplar pattern.

- [ ] **Step 0: Byte-ratchet coordination** — locate and run the byte-ratchet test (`grep -rn "806" Tests/ --include="*.py" | head -3"`). If the bundle is still over limit and TASK-31500 hasn't landed pay-down, STOP and sequence with its owner (spec §3.9): pair each migration PR with a pay-down chunk or wait.
- [ ] **Step 1: Exemplar pass on `features/_wizards.tcss`** (1,497 lines, representative):
  1. Inventory literals: `python -c "import re;print(len(re.findall(r'\b(?:padding|margin|width|height)(?:-(?:top|right|bottom|left))?\s*:\s*[0-9]', open('tldw_chatbook/css/features/_wizards.tcss').read())))"`
  2. Map each: recurring value (appears in 2+ sheets — check with `grep -rn "height: 3" tldw_chatbook/css/`) → shared token (`$ds-control-height`, `$ds-space-*`, new `$ds-*` for genuinely new recurring values like `width: 1fr` → `$ds-width-fill: 1fr` if introduced); feature-specific → new `$ds-wizards-<name>` token in `core/_variables.tcss`.
  3. Preserve every comment; add one header line citing ADR-161 and the migration date (exemplar: `features/_chat.tcss` header).
  4. Rebuild; ratchet for this sheet must read 0; governance + sync-guard green; `python -m pytest Tests/UI/ -k "wizard" -q`; SVG A/B on the wizard screens.
  5. Commit: `git commit -m "refactor(css): migrate _wizards.tcss to zero literals (ADR-161)"`
- [ ] **Step 2: Remaining sheets, one commit each**, in descending literal count (find order via the baseline JSON). Per sheet: the five sub-steps above, with `-k "<feature>"` targeted tests. The sheet set is the baseline's `dimensions` keys minus `_chat.tcss` (already clean) minus sheets zeroed by Tasks 9–10 moves. Track progress by re-running the pin script with a dry variant (compare live counts vs baseline; never re-pin upward).
- [ ] **Step 3: Floor check** — `python -m pytest Tests/UI/test_component_pattern_governance.py::test_dimension_literal_ratchet -q` with a temporary assertion flip: every baseline entry's live count is 0. When true, simplify the test to `assert not _DIM_matches_anywhere` (hard zero) and drop the baseline dimensions section.

---

### Task 12: Python `styles.*` cleanup

**Files:** every entry in `pattern_ratchet_baseline.json` → `python_styles` (batch by widget family).

**Interfaces:** consumes catalog classes (Tasks 3–9); produces the zero floor for `test_python_style_ratchet`.

- [ ] **Step 1: Batch recipe** (per file, one commit per coherent batch — e.g. "console widgets", "ingestion forms"):
  1. List sites: `grep -nE "\.styles\.(background|color|border\w*|width|height|padding\w*|margin\w*|opacity\w*) *=" <file>`
  2. For each: replace the runtime assignment with a class from the catalog (e.g. `w.styles.background = "red"` → `w.add_class("status-label")` + a `$ds-status-*`-backed class; `w.styles.display`/layout-only assignments are NOT in scope — the ratchet regex only counts visual properties).
  3. If no class expresses it: add the class to the owning sheet (tokenized) + registry + catalog + gallery FIRST (spec §5 "Adding to the language"), then use it.
  4. Verify: targeted tests for the file; ratchet count decreases; commit `git commit -m "refactor(ui): replace ad-hoc styles.* with token-backed classes in <area> (ADR-161)"`.
- [ ] **Step 2: Floor check** — same flip-to-hard-zero as Task 11 Step 3 for `test_python_style_ratchet`.

---

### Task 13: Close-out — constitution rewrite, hygiene, DoD

**Files:**
- Modify: `backlog/docs/design-language.md` (§6), parent task file, `Tests/UI/test_design_token_governance.py` (empty the now-zero `_HEX_GRANDFATHERED` if the last hex landed)

- [ ] **Step 1: Rewrite constitution §6** — replace the "What is deliberately NOT tokenized yet" section with:

```markdown
## 6. Literal policy (ADR-161)

There are no legacy literals: every color and dimension value in every sheet is a
`$ds-*` token (shared, or feature-scoped `$ds-<feature>-*` awaiting a second
consumer). Python sets visual values only through token-backed classes. The
governance tests (`test_component_pattern_governance.py`) hold the floor at zero;
a new value enters only by first becoming a token (see "Adding to the language").
Multi-value composite tokens remain deferred until Textual substitution
semantics are verified (ADR-150).
```

- [ ] **Step 2: Final verification sweep** (targeted, per constraints): full governance file + token governance + bundle sync + gallery snapshots, all green; `grep -rn '#[0-9a-fA-F]\{6\}' tldw_chatbook/css/{core,layout,components,features,utilities}/*.tcss` outside `_variables.tcss` comments = 0.
- [ ] **Step 3: Lessons check** — per AGENTS.md DoD #9, add a `lessons-*.md` entry ONLY if this project produced a generalizable trap (candidate: the two-track move rule if a move bites; do not invent one otherwise).
- [ ] **Step 4: Task hygiene** — parent task: all AC checkboxes `- [x]`, Implementation Notes (approach, deviations, files), status Done via `backlog task edit <id> -s Done`. Verify every subtask likewise Done or consciously cancelled with notes.

---

## Self-Review (completed)

**Spec coverage:** §3.1 → Tasks 3/5 form factor; §3.2 catalog → Task 3; §3.3 registry → Task 3; §3.4 gallery → Task 5; §3.5 homes/carve-up/two-track/ceiling/tokenization-on-move → Tasks 4/9/10; §3.6 consolidation pipeline → Tasks 4/7/8; §3.7 governance (checks 1–7) → Task 6 (+10 for the ceiling); §3.8 dead code → Task 2; §3.9 byte sequencing → Task 11 Step 0 + per-task `wc -c` checks; §3.10 in-place migration + Python cleanup → Tasks 11/12; §5 task structure/ADR-161 → Task 1; §6 testing strategy → spread per task. Gaps: none found.

**Placeholder scan:** Task 8c requires a grep to enumerate the shared message classes before editing — that grep IS the step (the class list is discovered data, not withheld design). Task 9 Step 2's variant-vs-rename decision is a measured decision rule, not a TBD. No "similar to Task N" without the procedure being restated where it is consumed (Tasks 8–9 restate the verification steps).

**Type consistency:** `patterns.json` keys (`families`/`deprecated`/`ceiling`/`renames_to`) match between Task 3 schema and Task 6 test code; `iter_blocks(PKG, WIDGET_ATTR|SCREEN_ATTR)` matches `widget_css.py:803`; `CSS_MODULES` matches `build_css.py:195`; gallery import path `tldw_chatbook.Widgets.pattern_gallery` consistent across Tasks 5/6.
