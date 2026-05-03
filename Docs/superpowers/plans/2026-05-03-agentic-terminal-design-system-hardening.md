# Agentic Terminal Design System Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Establish the visual design-system foundation first, then harden the current Unified Shell and agentic terminal implementation so it remains usable, honest, and extensible as the app migrates from legacy tabs to Chat-first agentic workflows.

**Architecture:** Treat the visual design system as the first source of truth, not as polish after code. Create and review a Textual-aware visual system contract before runtime implementation, then add contract tests around the seams that are most likely to regress: primary top navigation, explicit overflow discoverability, command-palette naming, product-model visibility, footer shortcut state, semantic design tokens, and shell chrome ownership. Implement only small code changes behind those tests, then update the design docs to match verified behavior.

**Tech Stack:** Python 3.11+, Textual, pytest, TCSS modules, `tldw_chatbook/css/build_css.py`, existing `TldwCli` command-palette providers.

---

## Scope

This plan starts from current `origin/dev`, where Unified Shell Phase 1 already exists. It first creates the visual design-system foundation that future code work must follow. It does not recreate the New_UI concept screens and does not convert every legacy screen to the new design system in one pass.

The work hardens the existing shell contract:

- Visual design-system foundations are established before runtime hardening.
- Top bar remains the global primary destination navigation.
- Home and Console stay first-class, always reachable destinations.
- Compact destination labels, especially `W+C`, must expose full meaning through tooltip/help/palette text.
- Workspaces, Personas, Flashcards, and Quizzes stay visible in the product model even if some remain legacy/direct routes.
- The bottom bar becomes a context shortcut/status surface driven by an explicit source of truth, not by scraping arbitrary widgets.
- Design-system tokens stay Textual-compatible and implementation-safe.
- Shell chrome migration gets guardrails before larger screen rewrites.

## Non-Goals

- Do not redesign screen content for Library, Personas, Workspaces, flashcards, or quizzes in this plan; do verify they remain findable in labels, tooltips, docs, or command-palette help.
- Do not remove legacy routes or `ALL_TABS`.
- Do not replace the command palette.
- Do not move every `BaseAppScreen` to app-owned chrome in one large patch.
- Do not edit untracked root checkout files or depend on untracked `Docs/Design/New_UI/*.png`.
- Do not start runtime implementation tasks until Phase 0 visual design-system artifacts are committed and reviewed.

## Preconditions

- Work from `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/codex-agentic-design-system-review-fixes`.
- Confirm the branch is based on current `origin/dev`.
- Use the repo root virtualenv when running focused tests from this worktree:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest --version
```

- If the virtualenv is unavailable, stop and set up dependencies before changing code.
- Use `apply_patch` for manual edits.
- Use TDD for every runtime change: failing test, minimal implementation, focused verification, commit.
- Complete and commit Phase 0 before editing runtime Python or TCSS files. Phase 0 is design/doc work, not runtime implementation.

## File Structure

Modify:

- `Docs/Design/master-shell-design-system-contract.md` - verified contract updates and links to the visual system foundation.
- `tldw_chatbook/UI/Navigation/shell_destinations.py` - canonical shell destination metadata; add full-label and nav-priority semantics here if needed.
- `tldw_chatbook/UI/Navigation/main_navigation.py` - renders primary top navigation from destination metadata.
- `tldw_chatbook/app.py` - command-palette route labels/help; keep shell vocabulary aligned with navigation metadata.
- `tldw_chatbook/Widgets/AppFooterStatus.py` - bottom shortcut/status bar; preserve existing word, token, and DB status updates.
- `tldw_chatbook/css/core/_variables.tcss` - Textual-compatible semantic design token aliases.
- `tldw_chatbook/css/components/_agentic_terminal.tcss` - shared design-system classes and state styles.
- `tldw_chatbook/css/tldw_cli_modular.tcss` - generated stylesheet after TCSS module changes.
- `tldw_chatbook/css/Themes/themes.py` - `agentic_terminal` theme semantic variable values.
- `Docs/Design/master-shell-route-inventory.md` - destination metadata and route vocabulary updates only if behavior changes.

Create:

- `Docs/superpowers/specs/2026-05-03-agentic-terminal-visual-design-system-design.md` - visual source of truth for the agentic terminal design language.
- `Docs/Design/agentic-terminal-visual-system.md` - implementation-facing visual contract with token, component, state, and mockup guidance.
- `tldw_chatbook/UI/Navigation/shortcut_context.py` - small typed model for footer shortcut context, if the footer needs a reusable source-of-truth object.
- `Tests/UI/test_app_footer_shortcut_context.py` - focused tests for footer shortcut context lifecycle.
- `Tests/UI/test_shell_chrome_contract.py` - guardrail tests for shell chrome ownership and duplicate navigation.
- `Tests/UI/test_shell_product_model_visibility.py` - guardrail tests for overflow hints and visible product-model vocabulary.

Extend:

- `Tests/UI/test_master_shell_navigation.py` - top-nav capacity and destination metadata tests.
- `Tests/UI/test_command_palette_shell_routes.py` - command-palette full-label/help contract.
- `Tests/UI/test_command_palette_providers.py` - renamed display-label regressions where relevant.
- `Tests/UI/test_master_shell_design_system_contract.py` - semantic token and generated stylesheet contract tests.

## Phase 0: Visual Design-System Foundation

This phase must complete before Task 0 or any runtime implementation. It turns the existing screenshots/concepts/specs into an implementation-aware visual language that can be tested and coded in Textual without guessing.

**Files:**
- Create: `Docs/superpowers/specs/2026-05-03-agentic-terminal-visual-design-system-design.md`
- Create: `Docs/Design/agentic-terminal-visual-system.md`
- Modify: `Docs/Design/master-shell-design-system-contract.md`
- Modify: `Docs/superpowers/plans/2026-05-03-agentic-terminal-design-system-hardening.md`

- [x] **Step 1: Create the visual design-system spec**

Create `Docs/superpowers/specs/2026-05-03-agentic-terminal-visual-design-system-design.md`.

The spec must include these sections:

```markdown
# Agentic Terminal Visual Design System

## Purpose

Define the reusable visual language for the Chat-first agentic shell before runtime implementation begins.

## Product Model

- Console is the primary agentic control surface.
- Top bar is primary destination navigation.
- Workspaces are global context, not a buried note feature.
- Personas shape behavior and identity.
- Flashcards and quizzes remain visible Study modules.
- Library, Search, Media, Notes, Artifacts, Handoffs, MCP, ACP, Skills, Schedules, and Workflows remain visible in the product model.

## Visual Principles

- Command-line credible, not decorative terminal cosplay.
- Dense enough for power users, legible enough for first-time users.
- Status and authority are visible without reading logs.
- Recovery is part of the interface, not hidden in errors.

## Screen Grammar

- Top navigation: primary destinations only.
- Destination header: local title, purpose, readiness, source authority, and primary actions.
- Main work area: task-specific panels.
- Inspector/sidebar: context, provenance, artifacts, or settings.
- Footer: current shortcut context and low-noise status.

## Token System

Document semantic roles for surfaces, text, actions, focus, status, authority, source roles, borders, and density.

## Component Anatomy

Document top nav, destination header, panel, toolbar, status badge, recovery callout, source role chip, approval card, event row, field row, inspector, and shortcut bar.

## State Language

Document ready, running, paused, blocked, unavailable, approval required, unsaved, stale, conflict, recovered, local, server, workspace, remote-only, and dry-run.

## Reference Screens

Provide ASCII/reference mockups only where they clarify reusable layout rules:

- Home dashboard shell.
- Console agentic control surface.
- Library or Workspace/source context screen.
- Study or Personas screen showing flashcards/quizzes/persona visibility.

## Textual Implementation Constraints

- Use hyphenated TCSS variables and `Theme.variables` keys.
- Avoid dotted token names in TCSS.
- Require fallback glyphs and readable text labels.
- Design for compact terminal widths.
- Preserve keyboard-first workflows.
```

- [x] **Step 2: Create the implementation-facing visual contract**

Create `Docs/Design/agentic-terminal-visual-system.md`.

This file is the bridge from design language to implementation and must include:

- Canonical semantic token names using Textual-safe hyphenated names.
- Component class mapping to `.ds-*` classes.
- State class mapping to `.is-*`, `.source-*`, and `.needs-*` classes.
- Density rules for compact and comfortable terminal layouts.
- ASCII mockups or layout diagrams for the 2-4 reference screens from the spec.
- A "Do Not Implement Yet" section for screen-specific redesigns that are not part of this hardening plan.

- [x] **Step 3: Link the visual contract from the master shell contract**

Update `Docs/Design/master-shell-design-system-contract.md` so it names `Docs/Design/agentic-terminal-visual-system.md` as the visual source of truth.

Add a short gate:

```markdown
Runtime shell implementation must not introduce new visual patterns, token names, or state treatments that are not represented in `Docs/Design/agentic-terminal-visual-system.md`.
```

- [x] **Step 4: Review the visual foundation before runtime work**

Review the new visual foundation against these checks:

- Does it explain what a first-time user sees first?
- Does it preserve command-palette and keyboard speed for power users?
- Does it keep Console primary without hiding Workspaces, Personas, Flashcards, Quizzes, Library, Search, Media, Notes, Artifacts, and Handoffs?
- Does it map visual ideas to Textual-compatible tokens/classes?
- Does it avoid relying on screenshots as literal 1:1 implementation requirements?

- [x] **Step 5: Verify docs-only quality**

Run:

```bash
git diff --check
rg -n "Docs/Design/agentic-terminal-visual-system.md|agentic-terminal-visual-design-system" Docs/Design/master-shell-design-system-contract.md Docs/superpowers/plans/2026-05-03-agentic-terminal-design-system-hardening.md
```

Expected:

- No whitespace errors.
- The master shell contract and this plan both reference the visual system artifacts.

- [x] **Step 6: Commit**

```bash
git add Docs/superpowers/specs/2026-05-03-agentic-terminal-visual-design-system-design.md Docs/Design/agentic-terminal-visual-system.md Docs/Design/master-shell-design-system-contract.md Docs/superpowers/plans/2026-05-03-agentic-terminal-design-system-hardening.md
git commit -m "Add agentic terminal visual design system foundation"
```

## Task 0: Verify Product-Model Visibility And Overflow Hints

**Files:**
- Create: `Tests/UI/test_shell_product_model_visibility.py`
- Modify: `tldw_chatbook/UI/Navigation/shell_destinations.py`
- Modify: `tldw_chatbook/UI/Navigation/main_navigation.py`
- Modify: `Docs/Design/master-shell-route-inventory.md`

- [x] **Step 1: Write failing visibility and overflow tests**

Create `Tests/UI/test_shell_product_model_visibility.py`.

```python
import pytest
from textual.app import App
from textual.widgets import Static

from tldw_chatbook.Constants import TAB_STUDY
from tldw_chatbook.UI.Navigation.main_navigation import MainNavigationBar
from tldw_chatbook.UI.Navigation.shell_destinations import get_shell_destination
from tldw_chatbook.app import TabNavigationProvider


def test_library_destination_keeps_workspaces_visible():
    library = get_shell_destination("library")

    assert "Workspaces" in library.purpose
    assert "Workspaces" in library.tooltip


def test_study_modules_remain_discoverable_as_legacy_direct_route():
    help_text = TabNavigationProvider.TAB_HELP_TEXT[TAB_STUDY].lower()

    assert "flashcards" in help_text
    assert "quizzes" in help_text


@pytest.mark.asyncio
async def test_navigation_exposes_explicit_overflow_hint():
    class TestApp(App):
        def compose(self):
            yield MainNavigationBar(active="home")

    app = TestApp()

    async with app.run_test(size=(60, 20)) as pilot:
        await pilot.pause(0.1)
        overflow = app.query_one("#nav-overflow-hint", Static)

    assert "More" in str(overflow.renderable)
    assert "Ctrl+P" in str(overflow.renderable)
```

- [x] **Step 2: Run tests to verify they fail for the intended reasons**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/UI/test_shell_product_model_visibility.py --tb=short
```

Expected:

- Fails because Library metadata does not mention Workspaces.
- Fails because `#nav-overflow-hint` does not exist.

- [x] **Step 3: Add minimal product-model vocabulary**

Update the Library destination metadata, not screen content:

```python
ShellDestination(
    "library",
    "Library",
    "library",
    "Workspaces, source material, imports, notes, media, conversations, and Search/RAG.",
    "Browse Workspaces, imports, notes, media, search, and source material.",
    ("notes", "media", "ingest", "search", "conversation"),
)
```

Update `Docs/Design/master-shell-route-inventory.md` so Library compatibility explicitly includes Workspaces.

- [x] **Step 4: Add a non-invasive overflow hint**

In `MainNavigationBar.compose()`, keep all existing nav buttons and add an explicit overflow/discovery hint after the primary nav row:

```python
yield Static("More: Ctrl+P", id="nav-overflow-hint", classes="nav-overflow-hint")
```

In `MainNavigationBar.DEFAULT_CSS`, add styling that does not hide Home or Console:

```css
.nav-overflow-hint {
    dock: right;
    width: auto;
    padding: 0 1;
    color: $text-muted;
}
```

Do not build a responsive overflow menu in this task. The purpose is to remove silent hidden navigation by making command-palette discovery explicit.

- [x] **Step 5: Verify focused tests**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/UI/test_shell_product_model_visibility.py Tests/UI/test_master_shell_navigation.py --tb=short
```

Expected: PASS.

- [x] **Step 6: Commit**

```bash
git add Tests/UI/test_shell_product_model_visibility.py tldw_chatbook/UI/Navigation/shell_destinations.py tldw_chatbook/UI/Navigation/main_navigation.py Docs/Design/master-shell-route-inventory.md
git commit -m "Harden shell product visibility"
```

## Task 1: Lock Navigation Metadata And Compact Label Semantics

**Files:**
- Modify: `tldw_chatbook/UI/Navigation/shell_destinations.py`
- Modify: `tldw_chatbook/UI/Navigation/main_navigation.py`
- Test: `Tests/UI/test_master_shell_navigation.py`

- [x] **Step 1: Write metadata tests for full labels and priority**

Add tests that fail until destination metadata exposes both compact and full meaning.

```python
def test_compact_navigation_labels_preserve_full_meaning():
    from tldw_chatbook.UI.Navigation.shell_destinations import get_shell_destination

    wc = get_shell_destination("watchlists_collections")

    assert wc.label == "W+C"
    assert wc.full_label == "Watchlists+Collections"
    assert "Watchlists+Collections" in wc.tooltip
    assert wc.navigation_priority < get_shell_destination("settings").navigation_priority
```

Also add a mounted test that keeps Home and Console first in the rendered top nav:

```python
@pytest.mark.asyncio
async def test_home_and_console_remain_first_primary_destinations():
    class TestApp(App):
        def compose(self):
            yield MainNavigationBar(active="home")

    app = TestApp()

    async with app.run_test(size=(60, 20)) as pilot:
        await pilot.pause(0.1)
        buttons = list(app.query(".nav-button"))

    assert [(button.id, str(button.label).strip()) for button in buttons[:2]] == [
        ("nav-home", "Home"),
        ("nav-console", "Console"),
    ]
```

- [x] **Step 2: Run the failing navigation tests**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/UI/test_master_shell_navigation.py --tb=short
```

Expected:

- Fails because `ShellDestination.full_label` and `navigation_priority` do not exist.

- [x] **Step 3: Add minimal destination metadata**

Update `ShellDestination` with explicit optional metadata while preserving current constructor compatibility.

```python
@dataclass(frozen=True)
class ShellDestination:
    destination_id: str
    label: str
    primary_route: str
    purpose: str
    tooltip: str
    legacy_routes: tuple[str, ...] = ()
    full_label: str | None = None
    navigation_priority: int = 50

    @property
    def accessible_label(self) -> str:
        return self.full_label or self.label
```

Set priority and full label only where needed:

```python
ShellDestination(
    "watchlists_collections",
    "W+C",
    "watchlists_collections",
    "Monitored sources and curated reading/content collections.",
    "Open Watchlists+Collections for monitored sources and curated collections.",
    ("subscriptions", "subscription"),
    full_label="Watchlists+Collections",
    navigation_priority=40,
)
```

Use lower priority numbers for the most essential destinations:

- Home: `10`
- Console: `20`
- Library: `30`
- Watchlists+Collections: `40`
- Everything else: default `50` unless the test needs a documented distinction.

- [x] **Step 4: Render tooltip/help from accessible label**

In `MainNavigationBar.compose()`, keep the visible label unchanged but ensure compact labels expose full meaning.

```python
button = Button(
    destination.label,
    id=f"nav-{destination.destination_id}",
    classes="nav-button",
    tooltip=destination.tooltip,
)
button.tooltip = destination.tooltip
```

Do not add additional overflow behavior in this task. Task 0 owns the minimal explicit `More: Ctrl+P` overflow hint; this task only locks compact-label metadata.

- [x] **Step 5: Rerun focused navigation tests**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/UI/test_master_shell_navigation.py --tb=short
```

Expected: PASS.

- [x] **Step 6: Commit**

```bash
git add tldw_chatbook/UI/Navigation/shell_destinations.py tldw_chatbook/UI/Navigation/main_navigation.py Tests/UI/test_master_shell_navigation.py
git commit -m "Harden shell navigation metadata"
```

## Task 2: Align Command Palette With Full Destination Vocabulary

**Files:**
- Modify: `tldw_chatbook/app.py`
- Test: `Tests/UI/test_command_palette_shell_routes.py`
- Test: `Tests/UI/test_command_palette_providers.py`

- [ ] **Step 1: Write failing palette tests for compact labels**

Add a test that verifies the palette can be found with the full `Watchlists+Collections` term even though the top nav label stays `W+C`.
Add `TAB_WATCHLISTS_COLLECTIONS` to the imports in the test file that owns this assertion.

```python
def test_tab_navigation_provider_exposes_full_label_for_compact_destinations(tab_provider):
    command_text, tab_id, help_text = tab_provider._tab_command(TAB_WATCHLISTS_COLLECTIONS)

    assert tab_id == TAB_WATCHLISTS_COLLECTIONS
    assert "W+C" in command_text
    assert "Watchlists+Collections" in command_text
    assert "Watchlists+Collections" in help_text
```

Extend the async provider tests to search the full label and help-keyword aliases:

```python
@pytest.mark.asyncio
async def test_search_finds_watchlists_collections_by_full_label(tab_provider):
    hits = []
    async for hit in tab_provider.search("Watchlists+Collections"):
        hits.append(hit)

    assert any("Watchlists+Collections" in hit.text for hit in hits)
    assert any("Watchlists+Collections" in (hit.help or "") for hit in hits)


@pytest.mark.asyncio
async def test_search_matches_destination_help_keywords(tab_provider):
    cases = [
        ("Workspaces", "Library"),
        ("flashcards", "Study"),
        ("quizzes", "Study"),
    ]

    for query, expected_label in cases:
        hits = []
        async for hit in tab_provider.search(query):
            hits.append(hit)

        assert any(expected_label in hit.text for hit in hits), query
```

- [ ] **Step 2: Run failing palette tests**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q \
  Tests/UI/test_command_palette_shell_routes.py \
  Tests/UI/test_command_palette_providers.py::TestTabNavigationProvider::test_search_uses_current_display_labels_for_renamed_tabs \
  Tests/UI/test_command_palette_providers.py::TestTabNavigationProvider::test_search_finds_watchlists_collections_by_full_label \
  Tests/UI/test_command_palette_providers.py::TestTabNavigationProvider::test_search_matches_destination_help_keywords \
  --tb=short
```

Expected:

- Fails until help/search text includes the full destination vocabulary and provider search scores help text.

- [ ] **Step 3: Generate palette labels/help from shell metadata where possible**

Keep the existing `TabNavigationProvider` structure, but stop duplicating shell destination meaning for primary destinations.

Add small helpers in `TabNavigationProvider`:

```python
@classmethod
def _shell_destination_for_tab(cls, tab_id: str):
    from .UI.Navigation.shell_destinations import get_shell_destination, resolve_shell_route

    resolved = resolve_shell_route(cls.route_for_tab(tab_id))

    try:
        return get_shell_destination(resolved.destination_id)
    except KeyError:
        return None

@classmethod
def _shell_command_label(cls, tab_id: str, visible_label: str) -> str:
    destination = cls._shell_destination_for_tab(tab_id)
    if destination is None or destination.accessible_label == visible_label:
        return visible_label
    return f"{visible_label} ({destination.accessible_label})"

@classmethod
def _shell_help_text(cls, tab_id: str) -> str | None:
    destination = cls._shell_destination_for_tab(tab_id)
    if destination is None:
        return None
    return f"Open {destination.accessible_label} for {destination.purpose}"
```

Then update `_tab_command()` so the matcher searches the full label for compact destinations:

```python
def _tab_command(self, tab_id: str) -> tuple[str, str, str]:
    label = get_tab_display_label(tab_id)
    command_label = self._shell_command_label(tab_id, label)
    help_text = self._shell_help_text(tab_id) or self.TAB_HELP_TEXT.get(tab_id, f"Switch to {label}")
    return f"Tab Navigation: Switch to {command_label}", tab_id, help_text
```

This route-resolution step is required so `TAB_CHAT`/`chat` resolves to the `console` destination and legacy aliases such as `tools_settings` resolve to their shell owner. If a legacy direct route does not resolve to a shell destination, leave its existing direct-route help untouched.

Update `search()` so power users can find routes by help text as well as command label:

```python
for command_text, tab_id, help_text in tab_commands:
    score = max(matcher.match(command_text), matcher.match(help_text))
    if score > 0:
        yield Hit(
            score,
            matcher.highlight(command_text),
            partial(self.switch_tab, tab_id),
            help=help_text,
        )
```

- [ ] **Step 4: Preserve legacy direct commands**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/UI/test_command_palette_shell_routes.py --tb=short
```

Expected:

- `TAB_TOOLS_SETTINGS` still routes to `mcp`.
- Legacy direct commands remain available.
- Settings and MCP remain separate shell commands.

- [ ] **Step 5: Rerun provider-focused tests**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/UI/test_command_palette_shell_routes.py Tests/UI/test_command_palette_providers.py --tb=short
```

Expected: PASS, or only pre-existing unrelated skips.

- [ ] **Step 6: Commit**

```bash
git add tldw_chatbook/app.py Tests/UI/test_command_palette_shell_routes.py Tests/UI/test_command_palette_providers.py
git commit -m "Align command palette shell vocabulary"
```

## Task 3: Add Footer Shortcut Context Source Of Truth

**Files:**
- Create: `tldw_chatbook/UI/Navigation/shortcut_context.py`
- Modify: `tldw_chatbook/Widgets/AppFooterStatus.py`
- Test: `Tests/UI/test_app_footer_shortcut_context.py`

- [ ] **Step 1: Write failing footer context tests**

Create `Tests/UI/test_app_footer_shortcut_context.py`.

```python
import pytest
from textual.app import App

from tldw_chatbook.UI.Navigation.shortcut_context import ShortcutAction, ShortcutContext
from tldw_chatbook.Widgets.AppFooterStatus import AppFooterStatus


@pytest.mark.asyncio
async def test_footer_uses_global_shortcuts_by_default():
    class TestApp(App):
        def compose(self):
            yield AppFooterStatus(id="footer")

    app = TestApp()

    async with app.run_test(size=(100, 12)) as pilot:
        await pilot.pause(0.1)
        footer = app.query_one("#footer", AppFooterStatus)

        assert "Ctrl+Q quit" in footer.shortcut_text
        assert "Ctrl+P palette" in footer.shortcut_text


@pytest.mark.asyncio
async def test_footer_replaces_stale_context_shortcuts():
    class TestApp(App):
        def compose(self):
            yield AppFooterStatus(id="footer")

    app = TestApp()

    async with app.run_test(size=(100, 12)) as pilot:
        await pilot.pause(0.1)
        footer = app.query_one("#footer", AppFooterStatus)
        footer.set_shortcut_context(
            ShortcutContext(
                source="console",
                actions=(ShortcutAction("Ctrl+Enter", "send"),),
            )
        )
        footer.set_shortcut_context(
            ShortcutContext(
                source="library",
                actions=(ShortcutAction("Ctrl+F", "search"),),
            )
        )

        assert "Ctrl+F search" in footer.shortcut_text
        assert "Ctrl+Enter send" not in footer.shortcut_text
```

- [ ] **Step 2: Run failing footer tests**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/UI/test_app_footer_shortcut_context.py --tb=short
```

Expected:

- Fails because `shortcut_context.py`, `ShortcutAction`, `ShortcutContext`, `set_shortcut_context()`, and `shortcut_text` do not exist.

- [ ] **Step 3: Add a small typed context model**

Create `tldw_chatbook/UI/Navigation/shortcut_context.py`.

```python
"""Shortcut context model for the global footer."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ShortcutAction:
    key: str
    label: str
    available: bool = True

    def render(self) -> str:
        suffix = "" if self.available else " unavailable"
        return f"{self.key} {self.label}{suffix}"


@dataclass(frozen=True)
class ShortcutContext:
    source: str
    actions: tuple[ShortcutAction, ...]

    def render(self) -> str:
        visible = [action.render() for action in self.actions]
        return " | ".join(visible)
```

- [ ] **Step 4: Update footer without breaking existing status updates**

In `AppFooterStatus`, replace the hard-coded `_key_quit` responsibility with a shortcut display while preserving IDs expected by CSS where practical.

```python
DEFAULT_SHORTCUT_TEXT = "Ctrl+Q quit | Ctrl+P palette"

class AppFooterStatus(Widget):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._shortcut_text = self.DEFAULT_SHORTCUT_TEXT
        self._shortcut_display = Static(self._shortcut_text, id="footer-key-quit")
        ...

    @property
    def shortcut_text(self) -> str:
        return self._shortcut_text

    def _set_shortcut_text(self, text: str) -> None:
        self._shortcut_text = text
        self._shortcut_display.update(text)

    def set_shortcut_context(self, context: ShortcutContext) -> None:
        text = context.render() or self.DEFAULT_SHORTCUT_TEXT
        self._set_shortcut_text(text)

    def clear_shortcut_context(self) -> None:
        self._set_shortcut_text(self.DEFAULT_SHORTCUT_TEXT)
```

Keep the existing `#footer-key-quit` id so `_widgets.tcss` continues to style the footer shortcut area. Only edit `_widgets.tcss` and regenerate `tldw_cli_modular.tcss` if this id-preserving approach fails in focused tests. Do not remove word count, token count, or DB status behavior.

- [ ] **Step 5: Verify focused footer tests**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/UI/test_app_footer_shortcut_context.py --tb=short
```

Expected: PASS.

- [ ] **Step 6: Verify existing footer update users still work syntactically**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/UI/test_master_shell_design_system_contract.py Tests/UI/test_chat_window_enhanced.py --tb=short
```

Expected:

- Design-system contract still passes.
- Chat window tests do not fail because footer update methods changed.

- [ ] **Step 7: Commit**

```bash
git add tldw_chatbook/UI/Navigation/shortcut_context.py tldw_chatbook/Widgets/AppFooterStatus.py Tests/UI/test_app_footer_shortcut_context.py
git commit -m "Add footer shortcut context model"
```

## Task 4: Harden Semantic Token And Generated CSS Contracts

**Files:**
- Modify: `Tests/UI/test_master_shell_design_system_contract.py`
- Modify: `tldw_chatbook/css/core/_variables.tcss`
- Modify: `tldw_chatbook/css/components/_agentic_terminal.tcss`
- Modify: `tldw_chatbook/css/Themes/themes.py`
- Regenerate: `tldw_chatbook/css/tldw_cli_modular.tcss`

- [ ] **Step 1: Write failing tests for token naming and theme parity**

Extend `Tests/UI/test_master_shell_design_system_contract.py`:

```python
def test_design_system_tokens_use_textual_safe_names():
    source_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (
            Path("tldw_chatbook/css/core/_variables.tcss"),
            Path("tldw_chatbook/css/components/_agentic_terminal.tcss"),
        )
    )

    assert "ds." not in source_text
    assert "$ds-" in source_text


def test_agentic_terminal_theme_variables_cover_required_tokens():
    themes_text = THEMES_PY.read_text(encoding="utf-8")
    for token_name in REQUIRED_SEMANTIC_TOKENS:
        assert f'"{token_name}"' in themes_text


def test_generated_stylesheet_preserves_textual_safe_tokens():
    loaded_text = LOADED_TCSS.read_text(encoding="utf-8")

    assert "ds." not in loaded_text
    for token_name in REQUIRED_SEMANTIC_TOKENS:
        assert f"${token_name}" in loaded_text
```

- [ ] **Step 2: Run failing contract tests**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/UI/test_master_shell_design_system_contract.py --tb=short
```

Expected:

- Fails if any required token is missing from `agentic_terminal_theme.variables` or TCSS sources.

- [ ] **Step 3: Add only missing semantic aliases**

Use hyphenated TCSS variables only. Do not introduce dotted token names.

Add missing aliases in `tldw_chatbook/css/core/_variables.tcss` only when a class actually uses or needs them, for example:

```css
$ds-surface-panel: $panel;
$ds-surface-field: $boost;
$ds-surface-overlay: $surface;
$ds-surface-divider: $primary-background-lighten-2;
$ds-text-primary: $text;
$ds-text-secondary: $text-muted;
$ds-action-focus: $accent;
$ds-status-ready: $success;
$ds-status-warning: $warning;
$ds-status-error: $error;
$ds-authority-local: $success;
$ds-source-role-evidence: $accent;
```

Mirror required token keys in `agentic_terminal_theme.variables`.

- [ ] **Step 4: Regenerate modular CSS after TCSS edits**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python tldw_chatbook/css/build_css.py
```

Expected:

- `tldw_chatbook/css/tldw_cli_modular.tcss` is updated.
- Output lists `components/_agentic_terminal.tcss` as processed.

- [ ] **Step 5: Verify design-system contract**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/UI/test_master_shell_design_system_contract.py --tb=short
git diff --check
```

Expected: PASS and no whitespace errors.

- [ ] **Step 6: Commit**

```bash
git add Tests/UI/test_master_shell_design_system_contract.py tldw_chatbook/css/core/_variables.tcss tldw_chatbook/css/components/_agentic_terminal.tcss tldw_chatbook/css/Themes/themes.py tldw_chatbook/css/tldw_cli_modular.tcss
git commit -m "Harden agentic terminal token contract"
```

## Task 5: Add Shell Chrome Guardrails Before Any Larger Migration

**Files:**
- Create: `Tests/UI/test_shell_chrome_contract.py`
- Modify only if needed: `tldw_chatbook/UI/Navigation/base_app_screen.py`
- Modify only if needed: `tldw_chatbook/app.py`

- [ ] **Step 1: Write guardrail tests for duplicate nav and local context leakage**

Create `Tests/UI/test_shell_chrome_contract.py`.

```python
import pytest
from textual.app import App

from tldw_chatbook.UI.Navigation.base_app_screen import BaseAppScreen
from tldw_chatbook.UI.Navigation.main_navigation import MainNavigationBar


@pytest.mark.asyncio
async def test_base_app_screen_mounts_exactly_one_navigation_bar():
    class TestScreen(BaseAppScreen):
        def __init__(self, app_instance):
            super().__init__(app_instance, "home")

    class HostApp(App):
        async def on_mount(self):
            await self.push_screen(TestScreen(self))

    app = HostApp()

    async with app.run_test(size=(100, 20)) as pilot:
        await pilot.pause(0.1)
        assert len(list(pilot.app.query(MainNavigationBar))) == 1


def test_navigation_contract_keeps_context_out_of_top_nav():
    from tldw_chatbook.UI.Navigation.shell_destinations import SHELL_DESTINATION_ORDER

    forbidden_local_terms = {"approval required", "selected source", "unsaved changes", "provider unavailable"}
    joined = " ".join(
        f"{destination.label} {destination.tooltip} {destination.purpose}".lower()
        for destination in SHELL_DESTINATION_ORDER
    )

    for term in forbidden_local_terms:
        assert term not in joined
```

This test intentionally uses a minimal host app instead of `TldwCli()` so it does not touch local SQLite user data.

- [ ] **Step 2: Run guardrail tests**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/UI/test_shell_chrome_contract.py --tb=short
```

Expected:

- PASS if the current per-screen nav mount is clean.
- If it fails because test setup is invalid, fix only the test harness.
- If it fails because duplicate nav is real, stop and inspect before changing runtime code.

- [ ] **Step 3: Avoid shell-owned chrome migration unless tests prove a defect**

If duplicate navigation is not present, do not move nav ownership in this PR. Document that larger app-owned shell chrome remains a later migration.

If duplicate navigation is present:

- Add the smallest guard to prevent double mount.
- Do not rewrite all screens.
- Add a follow-up Backlog task or design note for full app-owned chrome migration.

- [ ] **Step 4: Commit**

```bash
git add Tests/UI/test_shell_chrome_contract.py tldw_chatbook/UI/Navigation/base_app_screen.py tldw_chatbook/app.py
git commit -m "Add shell chrome contract guardrails"
```

If no runtime files changed, omit them from `git add`.

## Task 6: Update Design Docs To Match Verified Behavior

**Files:**
- Modify: `Docs/Design/master-shell-design-system-contract.md`
- Modify: `Docs/Design/agentic-terminal-visual-system.md`
- Modify: `Docs/Design/master-shell-route-inventory.md`
- Modify: `Docs/superpowers/specs/2026-05-02-agentic-terminal-design-system-design.md`

- [ ] **Step 1: Update only verified contract sections**

Document:

- `ShellDestination.full_label` or equivalent full-label metadata.
- Top nav visible label vs accessible/help label rule.
- Footer shortcut source-of-truth API.
- Token naming contract: hyphenated TCSS variables, `Theme.variables` keys, no dotted TCSS names.
- Shell chrome guardrail test coverage and the deferred full app-owned chrome migration.
- Any verified runtime deviation from Phase 0 visual system rules.

- [ ] **Step 2: Do not claim screen-level redesign completion**

Keep wording explicit:

- Home and Console shell contract is hardened.
- Library, Workspaces, Personas, Flashcards, Quizzes, Search, Media, Notes, and Handoffs remain destination-level product model commitments unless separately verified.
- Concept screenshots remain inspiration, not 1:1 implementation requirements.

- [ ] **Step 3: Run docs/code whitespace check**

Run:

```bash
git diff --check
```

Expected: no whitespace errors.

- [ ] **Step 4: Commit**

```bash
git add Docs/Design/master-shell-design-system-contract.md Docs/Design/agentic-terminal-visual-system.md Docs/Design/master-shell-route-inventory.md Docs/superpowers/specs/2026-05-02-agentic-terminal-design-system-design.md
git commit -m "Document verified shell design-system contracts"
```

## Final Verification

Run from `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/codex-agentic-design-system-review-fixes`:

Phase 0 docs gate:

```bash
test -f Docs/superpowers/specs/2026-05-03-agentic-terminal-visual-design-system-design.md
test -f Docs/Design/agentic-terminal-visual-system.md
rg -n "Agentic Terminal Visual Design System|Screen Grammar|Token System|Component Anatomy|State Language|Reference Screens" Docs/superpowers/specs/2026-05-03-agentic-terminal-visual-design-system-design.md
rg -n "Docs/Design/agentic-terminal-visual-system.md|Runtime shell implementation must not introduce" Docs/Design/master-shell-design-system-contract.md
```

Runtime and contract verification:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q \
  Tests/UI/test_master_shell_navigation.py \
  Tests/UI/test_shell_product_model_visibility.py \
  Tests/UI/test_command_palette_shell_routes.py \
  Tests/UI/test_command_palette_providers.py \
  Tests/UI/test_app_footer_shortcut_context.py \
  Tests/UI/test_master_shell_design_system_contract.py \
  Tests/UI/test_shell_chrome_contract.py \
  --tb=short

git diff --check
git status --short --branch
```

Expected:

- Phase 0 visual design-system docs exist and are linked from the shell contract.
- Focused tests pass, or only pre-existing unrelated skips are reported.
- `git diff --check` reports no whitespace errors.
- Branch is ahead of `origin/dev` by the planned commits.

Manual UX smoke, if the app can launch in the current environment:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m tldw_chatbook.app
```

Walk through:

- Home is the first visible destination and Console is second.
- Narrow the terminal enough that not every destination comfortably fits; confirm `More: Ctrl+P` or equivalent overflow guidance remains visible.
- Open the command palette and verify `Watchlists+Collections`, `Workspaces`, `flashcards`, and `quizzes` are findable through labels or help text.
- Navigate Home -> Console -> Library -> Personas -> W+C -> Settings using keyboard actions where practical.
- Confirm footer shortcut text changes only when a screen/workflow supplies an explicit context and does not leave stale Console shortcuts after leaving Console.
- Record any environment limitation separately from product behavior, especially local SQLite access failures.

## Review Checklist

- [ ] Phase 0 visual design-system spec exists and is committed before runtime implementation.
- [ ] Implementation-facing visual contract maps visual rules to Textual-safe tokens, classes, states, and density rules.
- [ ] Reference mockups clarify reusable layout grammar without becoming literal 1:1 screen mandates.
- [ ] Top navigation still renders Home and Console first.
- [ ] Top navigation exposes explicit overflow/discovery guidance instead of silently relying on horizontal scroll.
- [ ] Workspaces, Flashcards, and Quizzes remain discoverable through destination metadata, command-palette help, or route inventory.
- [ ] `W+C` remains compact in the nav but exposes `Watchlists+Collections` through tooltip/help/palette text.
- [ ] Command palette preserves all legacy direct routes.
- [ ] Footer shortcuts update from an explicit context object and clear stale actions.
- [ ] Word count, token count, and DB footer updates still work.
- [ ] TCSS uses hyphenated token names only.
- [ ] `agentic_terminal_theme.variables` covers required design-system tokens.
- [ ] Generated stylesheet contains the updated design-system classes and tokens.
- [ ] Shell chrome tests prevent duplicate global navigation and context leakage.
- [ ] Docs describe only behavior verified by tests or direct code inspection.

## Stop Conditions

- Stop if Phase 0 visual design-system artifacts are missing, unreviewed, or contradict the current shell route inventory.
- Stop if focused `TldwCli()` tests fail before assertions due local SQLite sandbox access; treat that as an environment limitation and prefer smaller widget/provider tests for this plan.
- Stop if implementing footer context requires changing event handlers across multiple feature screens; create a follow-up task instead.
- Stop if explicit nav overflow requires a custom responsive widget larger than one PR; this plan only adds the minimal `More: Ctrl+P` discovery hint and guardrails for a later overflow affordance.
- Stop if command-palette changes would remove legacy route discoverability.
