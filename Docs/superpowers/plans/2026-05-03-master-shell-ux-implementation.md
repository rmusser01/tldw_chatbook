# Master Shell UX Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the approved master shell UX so `Home` becomes the dashboard/front door, user-facing `Chat` becomes `Console`, and every live agent/run flow converges on Console while preserving route compatibility.

**Architecture:** Gate implementation on the agentic terminal design-system contract, then add a central shell-destination model and migrate navigation, startup, Home, Console language, and destination containers in small PR-sized slices. Existing route IDs and wrapped legacy screens remain stable during migration; new labels and wrappers provide the user-facing IA without forcing a broad rewrite. Home owns dashboard/readiness/next-action state, while Console owns live sessions and all launch/follow flows.

**Tech Stack:** Python 3.11+, Textual, modular TCSS, pytest, existing `TldwCli` screen-router, current `BaseAppScreen` and `MainNavigationBar` shell, existing local/server scope services.

---

## Source Material

- Spec: `Docs/superpowers/specs/2026-05-02-new-user-first-run-shell-ux-design.md`
- Design system: `Docs/superpowers/specs/2026-05-02-agentic-terminal-design-system-design.md`
  - As of plan writing, this file exists on branch `codex/agentic-terminal-design-system` at commit `626db080`.
  - This plan depends on the implemented design-system foundation, not only the design-system spec document. Before shell work starts, the implementation branch must contain `_agentic_terminal.tcss`, build-order wiring, generated `tldw_cli_modular.tcss`, semantic tokens, and the `agentic_terminal` theme.
- Existing shell: `tldw_chatbook/UI/Navigation/base_app_screen.py`
- Existing navigation: `tldw_chatbook/UI/Navigation/main_navigation.py`
- Existing route resolver: `tldw_chatbook/app.py`
- Existing route tests: `Tests/UI/test_screen_navigation.py`
- Existing label tests: `Tests/UI/test_navigation_label_language.py`
- Existing Chat screen: `tldw_chatbook/UI/Screens/chat_screen.py`
- Existing Chat window: `tldw_chatbook/UI/Chat_Window_Enhanced.py`

## Current Constraints

- Do not delete legacy route IDs.
- Do not rename internal `chat` route in the first pass.
- Do not copy `Docs/Design/New_UI/*.png` literally; treat those as directional concept artifacts only.
- Keep `Docs/Design/New_UI/` untracked unless the user explicitly asks to include those assets.
- Do not invent screen-local visual language for new shell surfaces. Use the design-system semantic component classes, state classes, density classes, and testing hooks from `Docs/superpowers/specs/2026-05-02-agentic-terminal-design-system-design.md`.
- The design-system spec and implementation are being developed in parallel. Task 0 is a hard prerequisite; do not proceed to Task 1 with only the spec document present.
- Honor the design-system migration order: token/theme foundation first, shared component classes second, shell/navigation third, destination-by-destination adoption after Home and Console.
- The app currently has local docs commits ahead of `origin/dev`; implementation should happen in a clean worktree from current `origin/dev` plus the spec/plan commits or after those docs are merged.
- App-level `TldwCli()` tests may fail in some environments from local SQLite path restrictions; use `HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share` when running UI tests.

## Design System Integration Rules

The master-shell implementation must consume the agentic terminal design system rather than creating a second style layer.

- New Home, Console, destination-wrapper, status, recovery, approval, staged-source, and shortcut UI should use the shared design-system classes when present: `.ds-destination-header`, `.ds-panel`, `.ds-inspector`, `.ds-status-badge`, `.ds-recovery-callout`, `.ds-source-role`, `.ds-approval-card`, `.ds-event-row`, `.ds-field-row`, `.ds-toolbar`, and `.ds-shortcut-bar`.
- New state styling should use the shared state classes: `.is-active`, `.is-disabled`, `.is-blocked`, `.is-running`, `.is-paused`, `.is-unsaved`, `.is-stale`, `.is-conflict`, `.needs-approval`, `.source-local`, `.source-server`, `.source-workspace`, `.source-remote-only`, and `.source-dry-run`.
- New screens should support `.density-compact` and `.density-comfortable` without separate widget implementations.
- Status must never be color-only. Tests should assert readable labels such as `Ready`, `Blocked`, `Approval required`, `Paused`, `Running`, `Unavailable`, or `Recovered`.
- Home and Console are the first adoption surfaces because the design system identifies them as the shell drivers. Destination wrappers can start with correct structure and testing hooks, then adopt richer visual treatment incrementally.
- The route inventory must track both product ownership and design-system ownership: top navigation stays global, destination context stays in page headers, local inspectors remain destination-owned, and the bottom bar owns current-context shortcuts/status.

## File Structure

### New Files

- `Docs/Design/master-shell-route-inventory.md`
  - Human-readable inventory of current routes, wrappers, labels, shortcuts, command-palette entries, and target master-shell destination.

- `Docs/Design/master-shell-design-system-contract.md`
  - Implementation checklist mapping the parallel design-system spec onto master-shell screens, components, state classes, density modes, and testing hooks.

- `tldw_chatbook/css/components/_agentic_terminal.tcss`
  - Design-system component classes for `.ds-*` shell primitives and density/state-aware styling.

- `tldw_chatbook/UI/Navigation/shell_destinations.py`
  - Single source of truth for user-facing shell destinations, labels, tooltips, route aliases, grouping, and compatibility metadata.

- `tldw_chatbook/Home/dashboard_state.py`
  - Pure, side-effect-free Home dashboard state and next-best-action selection. No Textual imports.

- `tldw_chatbook/Home/__init__.py`
  - Exports Home dashboard models/helpers.

- `tldw_chatbook/UI/Screens/home_screen.py`
  - Textual Home dashboard screen, backed by `dashboard_state.py`.

- `tldw_chatbook/UI/Screens/library_screen.py`
  - Library destination wrapper for Notes, Media, Conversations, Import/Export, and Search/RAG entry.

- `tldw_chatbook/UI/Screens/library_conversations_screen.py`
  - Library-owned saved conversation browsing/source-access stub used until the legacy CCP screen is split.

- `tldw_chatbook/UI/Screens/artifacts_screen.py`
  - Artifacts destination wrapper for Chatbooks and generated/portable output entry.

- `tldw_chatbook/UI/Screens/personas_screen.py`
  - Personas destination wrapper for character/persona/prompt/dictionary/lore entry.

- `tldw_chatbook/UI/Screens/watchlists_collections_screen.py`
  - Watchlists+Collections destination wrapper with `Watchlists` and `Collections` sections.

- `tldw_chatbook/UI/Screens/schedules_screen.py`
  - Schedules destination wrapper. Owns "when" only.

- `tldw_chatbook/UI/Screens/workflows_screen.py`
  - Workflows destination wrapper. Owns "what" only.

- `tldw_chatbook/UI/Screens/mcp_screen.py`
  - MCP destination wrapper around existing unified MCP panels/services.

- `tldw_chatbook/UI/Screens/acp_screen.py`
  - ACP destination wrapper. Initially an honest empty/capability state if no ACP runtime exists locally yet.

- `tldw_chatbook/UI/Screens/skills_screen.py`
  - Skills destination wrapper around existing `Skills_Interop` services and local skill directory.

- `tldw_chatbook/UI/Screens/settings_screen.py`
  - Settings destination wrapper for global preferences, appearance, accounts/auth, storage, and app-level behavior.

- `Tests/UI/test_shell_destinations.py`
  - Unit tests for destination model, labels, aliases, and legacy route compatibility.

- `Tests/UI/test_master_shell_design_system_contract.py`
  - Pure contract tests for required design-system class names, readable status labels, and destination-wrapper testing hooks. These tests should not assert raw colors.

- `Tests/Home/test_dashboard_state.py`
  - Unit tests for Home state and deterministic next-best-action selection.

- `Tests/UI/test_home_screen.py`
  - Mounted Textual tests for Home dashboard sections and actions.

- `Tests/UI/test_master_shell_navigation.py`
  - Mounted Textual tests for top-level navigation order, labels, tooltips, active state, and old-route compatibility.

- `Tests/UI/test_command_palette_shell_routes.py`
  - Unit tests for command-palette shell vocabulary and route emission for Settings, MCP, Console, and legacy aliases.

- `Tests/UI/test_destination_shells.py`
  - Mounted Textual tests for destination wrapper purpose lines, primary actions, and section links.

- `Tests/UI/test_console_live_work_handoffs.py`
  - Mounted tests for Launch/Follow/Resume/Open-in-Console actions and pending Console launch rendering.

### Modified Files

- `tldw_chatbook/Constants.py`
  - Add master shell route constants where needed. Keep legacy tab constants stable.
  - Update user-facing labels through compatibility helpers rather than deleting old IDs.

- `tldw_chatbook/css/build_css.py`
  - Include the agentic terminal component module in the generated CSS build order.

- `tldw_chatbook/css/tldw_cli_modular.tcss`
  - Generated stylesheet actually loaded by `TldwCli.CSS_PATH`; must contain design-system classes/tokens after CSS build.

- `tldw_chatbook/UI/Navigation/main_navigation.py`
  - Render from `shell_destinations.py` instead of local `NAV_GROUPS`.
  - Keep button IDs stable enough for tests and route compatibility.

- `tldw_chatbook/UI/Navigation/base_app_screen.py`
  - Continue to mount one global nav. Optionally add a shared screen-header helper only if it reduces duplication.

- `tldw_chatbook/app.py`
  - Add Home and new destination wrappers to `_resolve_screen_navigation_target()`.
  - Set first-run startup to Home when appropriate.
  - Keep legacy route aliases.
  - Update command-palette `TabNavigationProvider` to use shell destinations and Console language.

- `tldw_chatbook/UI/Tab_Bar.py`
  - Preserve legacy compatibility if still used by tests or fallback paths.

- `tldw_chatbook/UI/Tab_Dropdown.py`
  - Preserve shared label compatibility.

- `tldw_chatbook/UI/Tab_Links.py`
  - Preserve shared label compatibility.

- `tldw_chatbook/UI/Screens/chat_screen.py`
  - User-facing title/copy becomes Console. Internal class may remain `ChatScreen`.

- `tldw_chatbook/UI/Chat_Window_Enhanced.py`
  - User-facing empty-state/orientation copy becomes Console. Preserve behavior.

- `Tests/UI/test_screen_navigation.py`
  - Extend route coverage to new master shell destinations and legacy aliases.

- `Tests/UI/test_navigation_label_language.py`
  - Update expected labels and ensure old tab widgets use compatibility labels.

- `Tests/UI/test_chat_first_run_orientation.py`
  - Update Chat copy expectations to Console where relevant.

- `Docs/Development/chat-first-shell-migration.md`
  - Add a short note that the user-facing destination is now Console while internal `chat` route remains compatible.

## Shared Verification Commands

Use these commands throughout unless a task gives a narrower command:

```bash
env HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/UI/test_shell_destinations.py Tests/Home/test_dashboard_state.py Tests/UI/test_master_shell_navigation.py --tb=short
```

```bash
env HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/UI/test_master_shell_design_system_contract.py --tb=short
```

```bash
env HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/UI/test_screen_navigation.py Tests/UI/test_navigation_label_language.py --tb=short
```

```bash
git diff --check
```

Run `git diff --check` immediately before every task commit, even when the focused task section does not repeat it.

## Task 0: Land Agentic Terminal Design-System Foundation

**Files:**
- Create: `Docs/superpowers/specs/2026-05-02-agentic-terminal-design-system-design.md` if the spec commit is not already present
- Create: `tldw_chatbook/css/components/_agentic_terminal.tcss`
- Modify: `tldw_chatbook/css/build_css.py`
- Modify: `tldw_chatbook/css/main.tcss`
- Modify: `tldw_chatbook/css/tldw_cli_modular.tcss`
- Modify: `tldw_chatbook/css/Themes/themes.py`
- Test: `Tests/UI/test_master_shell_design_system_contract.py` after Task 1 creates the contract regression

This is a prerequisite gate, not optional shell work. Complete it by merging/rebasing the implemented design-system branch or by landing a separate design-system PR before any navigation, Home, Console, or destination-wrapper code changes.

- [ ] **Step 1: Verify the design-system branch includes implementation, not docs only**

Run:

```bash
git show --stat codex/agentic-terminal-design-system -- Docs/superpowers/specs/2026-05-02-agentic-terminal-design-system-design.md tldw_chatbook/css/components/_agentic_terminal.tcss tldw_chatbook/css/build_css.py tldw_chatbook/css/main.tcss tldw_chatbook/css/tldw_cli_modular.tcss tldw_chatbook/css/Themes/themes.py
```

Expected: the branch or chosen implementation source includes the TCSS module, build wiring, generated loaded stylesheet, and theme changes. If it only includes the spec document, stop and implement the design-system foundation as its own PR before continuing.

- [ ] **Step 2: Merge or rebase the implemented foundation into the shell implementation worktree**

Use the repo's normal branch hygiene. Do not hand-copy the design-system spec or create a second style layer inside shell screens.

- [ ] **Step 3: Rebuild the generated stylesheet**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python tldw_chatbook/css/build_css.py
```

Expected: `tldw_chatbook/css/tldw_cli_modular.tcss` contains the `components/_agentic_terminal.tcss` section and the same semantic classes/tokens that Task 1 tests assert.

- [ ] **Step 4: Stop condition**

If `_agentic_terminal.tcss`, `agentic_terminal` theme entries, or generated loaded CSS are still missing, stop. Do not start shell navigation implementation.

## Task 1: Inventory Current Routes, Design-System Contract, And Compatibility Surface

**Files:**
- Create: `Docs/Design/master-shell-route-inventory.md`
- Create: `Docs/Design/master-shell-design-system-contract.md`
- Create: `Tests/UI/test_master_shell_design_system_contract.py`
- Modify: `Tests/UI/test_screen_navigation.py`
- Test: `Tests/UI/test_screen_navigation.py`, `Tests/UI/test_master_shell_design_system_contract.py`

- [ ] **Step 1: Write the design-system implementation contract**

Add `Docs/Design/master-shell-design-system-contract.md` with this structure:

```markdown
# Master Shell Design System Contract

Date: 2026-05-03

## Purpose

This contract maps the agentic terminal design system onto the master-shell implementation slices.

## Source Documents

- `Docs/superpowers/specs/2026-05-02-new-user-first-run-shell-ux-design.md`
- `Docs/superpowers/specs/2026-05-02-agentic-terminal-design-system-design.md`

## Required Shared Classes

| Class | Purpose | First master-shell use |
| --- | --- | --- |
| `.ds-destination-header` | Page title, purpose, local scope/status, primary action | Home and all wrappers |
| `.ds-panel` | Bordered purposeful content region | Home dashboard and wrappers |
| `.ds-inspector` | Selected item detail, readiness, permissions, recovery | Console and wrappers |
| `.ds-status-badge` | Readable semantic status label | Home, Console, wrappers |
| `.ds-recovery-callout` | Owner/problem/next action recovery copy | Home, Console, wrappers |
| `.ds-source-role` | context/evidence/editable-target/output-seed chips | Console staged context |
| `.ds-approval-card` | Approve/reject decision surface | Console, Home summary |
| `.ds-event-row` | Tool/run/audit event row | Console, Home active work |
| `.ds-field-row` | Label/control/help/validation row | Settings, builders |
| `.ds-toolbar` | Local action group | Destination wrappers |
| `.ds-shortcut-bar` | Current-context shortcuts/status | Shell bottom bar |

## Required State Classes

`.is-active`, `.is-disabled`, `.is-blocked`, `.is-running`, `.is-paused`, `.is-unsaved`, `.is-stale`, `.is-conflict`, `.needs-approval`, `.source-local`, `.source-server`, `.source-workspace`, `.source-remote-only`, `.source-dry-run`

## Density Contract

All new shell surfaces must support `.density-compact` and `.density-comfortable` without separate widget implementations.

## Required Readable Status Labels

`Ready`, `Running`, `Paused`, `Blocked`, `Unavailable`, `Approval required`, `Unsaved`, `Recovered`

## Testing Rules

- Assert stable IDs or classes for primary actions, status badges, source authority, source roles, approval cards, shortcut bars, next-best actions, and open/follow-in-Console controls.
- Assert readable status text. Do not assert raw color values.
- Assert destination context appears inside page headers, not the global top navigation.

## Stop Conditions

- Stop before shell implementation if the design-system spec is unavailable on the implementation branch.
- Stop before shell implementation if the shared design-system TCSS classes, density classes, state classes, and `agentic_terminal` theme are missing and need a separate design-system PR.
- Stop before shell implementation if the generated stylesheet loaded by `TldwCli.CSS_PATH` does not contain the design-system classes and semantic tokens.
```

- [ ] **Step 2: Add design-system contract regression**

Create `Tests/UI/test_master_shell_design_system_contract.py`:

```python
from pathlib import Path


REQUIRED_DESIGN_SYSTEM_CLASSES = {
    "ds-destination-header",
    "ds-panel",
    "ds-inspector",
    "ds-status-badge",
    "ds-recovery-callout",
    "ds-source-role",
    "ds-approval-card",
    "ds-event-row",
    "ds-field-row",
    "ds-toolbar",
    "ds-shortcut-bar",
}

REQUIRED_STATE_CLASSES = {
    "is-active",
    "is-disabled",
    "is-blocked",
    "is-running",
    "is-paused",
    "is-unsaved",
    "is-stale",
    "is-conflict",
    "needs-approval",
    "source-local",
    "source-server",
    "source-workspace",
    "source-remote-only",
    "source-dry-run",
}

READABLE_STATUS_LABELS = {
    "Ready",
    "Running",
    "Paused",
    "Blocked",
    "Unavailable",
    "Approval required",
    "Unsaved",
    "Recovered",
}

DESIGN_SYSTEM_TCSS = Path("tldw_chatbook/css/components/_agentic_terminal.tcss")
DESIGN_SYSTEM_SPEC = Path("Docs/superpowers/specs/2026-05-02-agentic-terminal-design-system-design.md")
MAIN_TCSS = Path("tldw_chatbook/css/main.tcss")
LOADED_TCSS = Path("tldw_chatbook/css/tldw_cli_modular.tcss")
BUILD_CSS_PY = Path("tldw_chatbook/css/build_css.py")
APP_PY = Path("tldw_chatbook/app.py")
THEMES_PY = Path("tldw_chatbook/css/Themes/themes.py")
CLASS_TEXT_FILES = [
    DESIGN_SYSTEM_TCSS,
    Path("tldw_chatbook/css/utilities/_states.tcss"),
]
SEMANTIC_TOKEN_FILES = [
    Path("tldw_chatbook/css/core/_variables.tcss"),
    DESIGN_SYSTEM_TCSS,
]

REQUIRED_SEMANTIC_TOKENS = {
    "ds-surface-panel",
    "ds-text-primary",
    "ds-action-focus",
    "ds-status-ready",
    "ds-status-warning",
    "ds-status-error",
    "ds-authority-local",
    "ds-source-role-evidence",
}


def test_master_shell_design_system_class_contract_is_documented():
    contract = "Docs/Design/master-shell-design-system-contract.md"
    text = Path(contract).read_text(encoding="utf-8")
    for class_name in REQUIRED_DESIGN_SYSTEM_CLASSES | REQUIRED_STATE_CLASSES:
        assert f".{class_name}" in text


def test_status_contract_requires_readable_labels():
    contract = Path("Docs/Design/master-shell-design-system-contract.md").read_text(encoding="utf-8")
    for label in READABLE_STATUS_LABELS:
        assert label in contract


def test_agentic_terminal_design_system_spec_is_present():
    assert DESIGN_SYSTEM_SPEC.exists()


def test_agentic_terminal_tcss_module_is_implemented_and_imported():
    assert DESIGN_SYSTEM_TCSS.exists()
    class_text = "\n".join(path.read_text(encoding="utf-8") for path in CLASS_TEXT_FILES if path.exists())
    main_text = MAIN_TCSS.read_text(encoding="utf-8")
    build_text = BUILD_CSS_PY.read_text(encoding="utf-8")

    assert '@import "./components/_agentic_terminal.tcss";' in main_text
    assert '"components/_agentic_terminal.tcss"' in build_text
    for class_name in REQUIRED_DESIGN_SYSTEM_CLASSES | REQUIRED_STATE_CLASSES:
        assert f".{class_name}" in class_text
    assert ".density-compact" in class_text
    assert ".density-comfortable" in class_text


def test_loaded_stylesheet_contains_agentic_terminal_contract():
    loaded_text = LOADED_TCSS.read_text(encoding="utf-8")
    app_text = APP_PY.read_text(encoding="utf-8")

    assert "tldw_cli_modular.tcss" in app_text
    assert "components/_agentic_terminal.tcss" in loaded_text
    for class_name in REQUIRED_DESIGN_SYSTEM_CLASSES | REQUIRED_STATE_CLASSES:
        assert f".{class_name}" in loaded_text
    for token_name in REQUIRED_SEMANTIC_TOKENS:
        assert token_name in loaded_text


def test_agentic_terminal_semantic_tokens_and_theme_exist():
    token_text = "\n".join(path.read_text(encoding="utf-8") for path in SEMANTIC_TOKEN_FILES if path.exists())
    for token_name in REQUIRED_SEMANTIC_TOKENS:
        assert token_name in token_text

    themes_text = THEMES_PY.read_text(encoding="utf-8")
    assert "agentic_terminal" in themes_text
```

- [ ] **Step 3: Write the inventory checklist document**

Add `Docs/Design/master-shell-route-inventory.md` with this structure:

```markdown
# Master Shell Route Inventory

Date: 2026-05-03

## Purpose

This inventory maps current routes and UI surfaces onto the approved master shell IA.

## Destination Map

| Master destination | Legacy routes | Existing screen/wrapper | Current user-facing label | Compatibility requirement |
| --- | --- | --- | --- | --- |
| Home | `home` | new | Home | New default for first-run users |
| Console | `chat` | `ChatScreen` | Chat | User-facing label becomes Console; route remains `chat` |
| Library | `notes`, `media`, `ingest`, `search`, `conversation`, conversation browsing | multiple | Notes/Media/Ingest/Search | New wrapper links to source routes; `conversation` means saved conversation browsing/source access |
| Artifacts | `chatbooks` | `ChatbooksScreen` | Chatbooks | New wrapper owns Chatbooks and outputs |
| Personas | `ccp`, character/persona/prompt/lore subviews | `ConversationScreen` | Library | User-facing label becomes Personas for behavior/identity management |
| Watchlists+Collections | `subscriptions` plus collections services | `SubscriptionScreen` | Subscriptions | New wrapper has Watchlists and Collections sections |
| Schedules | schedule surfaces | existing scheduler surfaces | mixed | New wrapper owns when-runs |
| Workflows | workflow surfaces | future/existing workflow code | mixed | New wrapper owns what-runs |
| MCP | `tools_settings`, tools/MCP settings | `ToolsSettingsScreen` and MCP panels | legacy Settings/tools label | New wrapper owns MCP/tool capability control; `tools_settings` becomes an MCP alias |
| ACP | ACP surfaces | new | ACP | New wrapper with honest unavailable state if needed |
| Skills | skills services | new | Skills | New wrapper around local/server skills |
| Settings | `settings`, `customize` | existing screens | Settings/Customize | Settings owns global app preferences only; do not route global preferences through `tools_settings` |
```

Add sections for:

```markdown
## Shortcut And Command Palette Inventory

| Shortcut/command | Current target | Master destination | Keep/change |
| --- | --- | --- | --- |

## Import/Export Boundary

Library import/export means source import/export.
Artifacts import/export means bundle/output import/export.

## Design-System Boundary

Top navigation is global primary destination navigation only.
Destination context, source authority, readiness, and recovery belong inside destination headers or local panels.
Status labels, source authority, approvals, staged source roles, recovery callouts, and shortcuts use the agentic terminal design-system contract.

## Open Questions For Implementation

- None. Add only implementation-time discoveries here.
```

- [ ] **Step 4: Add route inventory regression**

In `Tests/UI/test_screen_navigation.py`, add a pure test that documents all current `PRIMARY_ROUTE_IDS` plus the new master-shell aliases expected by the plan.

```python
def test_master_shell_route_inventory_has_known_legacy_routes():
    expected_legacy_routes = {
        "chat",
        "notes",
        "media",
        "ingest",
        "search",
        "study",
        "ccp",
        "conversation",
        "chatbooks",
        "subscriptions",
        "tools_settings",
        "llm",
        "stts",
        "evals",
        "logs",
        "stats",
        "coding",
        "customize",
    }

    app = _build_test_app()
    unresolved = []
    for route in expected_legacy_routes:
        _screen_name, _tab_id, screen_class = app._resolve_screen_navigation_target(route)
        if screen_class is None:
            unresolved.append(route)

    assert unresolved == []
```

Expected before implementation: may fail for any currently unsupported legacy route. If it fails, decide whether the route should be documented as intentionally unsupported or mapped.

- [ ] **Step 5: Run inventory and design-system contract regressions**

Run:

```bash
env HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/UI/test_screen_navigation.py::test_master_shell_route_inventory_has_known_legacy_routes Tests/UI/test_master_shell_design_system_contract.py --tb=short
```

Expected: FAIL until missing legacy aliases are mapped or inventory expectation is corrected and the real design-system TCSS/theme implementation has been merged. Do not proceed to Task 2 while `test_agentic_terminal_tcss_module_is_implemented_and_imported` or `test_agentic_terminal_semantic_tokens_and_theme_exist` is failing.

If the design-system branch is merged but `test_loaded_stylesheet_contains_agentic_terminal_contract` fails, rebuild generated CSS before continuing:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python tldw_chatbook/css/build_css.py
```

Expected after rebuild: `tldw_chatbook/css/tldw_cli_modular.tcss` contains the `components/_agentic_terminal.tcss` section and the same `.ds-*`, `.is-*`, `.density-*`, and semantic token strings tested above.

- [ ] **Step 6: Update only the inventory or aliases needed for a passing baseline**

If a route is intentionally not currently routable, move it out of `expected_legacy_routes` and into the inventory doc as a non-route command/surface. Do not add new UX behavior in this task.

- [ ] **Step 7: Run focused screen navigation and contract tests**

Run:

```bash
env HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/UI/test_screen_navigation.py Tests/UI/test_master_shell_design_system_contract.py --tb=short
```

Expected: PASS.

- [ ] **Step 8: Commit**

Stage the planning docs, contract tests, and any design-system source or generated stylesheet files touched by this preflight/rebuild. Omit only paths that do not exist in the current slice because the design-system branch has not been merged yet.

```bash
git add Docs/Design/master-shell-route-inventory.md Docs/Design/master-shell-design-system-contract.md Tests/UI/test_screen_navigation.py Tests/UI/test_master_shell_design_system_contract.py tldw_chatbook/css/components/_agentic_terminal.tcss tldw_chatbook/css/build_css.py tldw_chatbook/css/main.tcss tldw_chatbook/css/tldw_cli_modular.tcss tldw_chatbook/css/Themes/themes.py
git commit -m "test: document master shell planning contracts"
```

## Task 2: Add Central Shell Destination Model

**Files:**
- Create: `tldw_chatbook/UI/Navigation/shell_destinations.py`
- Create: `Tests/UI/test_shell_destinations.py`
- Modify: `tldw_chatbook/Constants.py`
- Test: `Tests/UI/test_shell_destinations.py`, `Tests/UI/test_navigation_label_language.py`

- [ ] **Step 1: Write failing destination model tests**

Create `Tests/UI/test_shell_destinations.py`:

```python
from tldw_chatbook.UI.Navigation.shell_destinations import (
    SHELL_DESTINATION_ORDER,
    get_shell_destination,
    resolve_shell_route,
)


def test_master_shell_destination_order_matches_spec():
    assert [destination.label for destination in SHELL_DESTINATION_ORDER] == [
        "Home",
        "Console",
        "Library",
        "Artifacts",
        "Personas",
        "Watchlists+Collections",
        "Schedules",
        "Workflows",
        "MCP",
        "ACP",
        "Skills",
        "Settings",
    ]


def test_legacy_routes_resolve_to_master_destinations():
    expectations = {
        "chat": ("console", "chat"),
        "home": ("home", "home"),
        "notes": ("library", "notes"),
        "media": ("library", "media"),
        "ingest": ("library", "ingest"),
        "search": ("library", "search"),
        "chatbooks": ("artifacts", "chatbooks"),
        "ccp": ("personas", "ccp"),
        "conversation": ("library", "conversation"),
        "conversations_characters_prompts": ("personas", "ccp"),
        "subscriptions": ("watchlists_collections", "subscriptions"),
        "tools_settings": ("mcp", "tools_settings"),
        "settings": ("settings", "settings"),
    }

    for route, expected in expectations.items():
        resolved = resolve_shell_route(route)
        assert (resolved.destination_id, resolved.canonical_route) == expected


def test_each_shell_destination_has_recovery_tooltip_copy():
    for destination in SHELL_DESTINATION_ORDER:
        assert destination.tooltip
        assert destination.purpose
```

Expected: FAIL because `shell_destinations.py` does not exist.

- [ ] **Step 2: Run failing test**

Run:

```bash
env HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/UI/test_shell_destinations.py --tb=short
```

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `shell_destinations.py`**

Create `tldw_chatbook/UI/Navigation/shell_destinations.py`:

```python
"""Master shell destination metadata and route compatibility helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class ShellDestination:
    destination_id: str
    label: str
    primary_route: str
    purpose: str
    tooltip: str
    legacy_routes: tuple[str, ...] = ()


@dataclass(frozen=True)
class ResolvedShellRoute:
    destination_id: str
    canonical_route: str
    requested_route: str


SHELL_DESTINATION_ORDER: tuple[ShellDestination, ...] = (
    ShellDestination("home", "Home", "home", "Dashboard, notifications, status, and next actions.", "Open dashboard, notifications, and active work."),
    ShellDestination("console", "Console", "chat", "Live agent conversations, approvals, tools, RAG, and runs.", "Open the live agent Console.", ("chat",)),
    ShellDestination("library", "Library", "library", "Source material, imports, notes, media, conversations, and Search/RAG.", "Browse, import, search, and query source material.", ("notes", "media", "ingest", "search", "conversation")),
    ShellDestination("artifacts", "Artifacts", "artifacts", "Generated outputs, bundles, reports, datasets, and Chatbooks.", "Browse generated and portable outputs.", ("chatbooks",)),
    ShellDestination("personas", "Personas", "personas", "Characters, personas, prompts, dictionaries, and behavior profiles.", "Manage behavior profiles and persona context.", ("ccp", "conversations_characters_prompts", "characters", "prompts")),
    ShellDestination("watchlists_collections", "Watchlists+Collections", "watchlists_collections", "Monitored sources and curated reading/content collections.", "Monitor feeds and curate collections.", ("subscriptions", "subscription")),
    ShellDestination("schedules", "Schedules", "schedules", "When jobs, watchlists, and workflows run.", "Manage run timing, triggers, and recovery."),
    ShellDestination("workflows", "Workflows", "workflows", "Reusable procedures, recipes, dry-runs, and outputs.", "Build and launch repeatable agent workflows."),
    ShellDestination("mcp", "MCP", "mcp", "MCP servers, tools, permissions, auth, and audit.", "Configure tool and server capability plumbing.", ("tools_settings",)),
    ShellDestination("acp", "ACP", "acp", "Agent Client Protocol agents, sessions, runtimes, diffs, and terminals.", "Manage ACP agents and sessions."),
    ShellDestination("skills", "Skills", "skills", "Agent Skills packs, discovery, validation, and attachments.", "Browse, import, validate, and attach skills."),
    ShellDestination("settings", "Settings", "settings", "Global app preferences, appearance, accounts, and storage.", "Configure application preferences.", ("customize",)),
)

_BY_DESTINATION_ID: Mapping[str, ShellDestination] = {
    destination.destination_id: destination for destination in SHELL_DESTINATION_ORDER
}

_ROUTE_MAP: dict[str, ResolvedShellRoute] = {}
_ROUTABLE_LEGACY_ROUTES = {
    "chat",
    "notes",
    "media",
    "ingest",
    "search",
    "conversation",
    "chatbooks",
    "ccp",
    "subscriptions",
    "tools_settings",
    "customize",
}
_CANONICAL_ROUTE_OVERRIDES = {
    "conversations_characters_prompts": "ccp",
    "characters": "ccp",
    "prompts": "ccp",
    "subscription": "subscriptions",
}

for destination in SHELL_DESTINATION_ORDER:
    _ROUTE_MAP[destination.primary_route] = ResolvedShellRoute(
        destination.destination_id,
        destination.primary_route,
        destination.primary_route,
    )
    _ROUTE_MAP[destination.destination_id] = ResolvedShellRoute(
        destination.destination_id,
        destination.primary_route,
        destination.destination_id,
    )
    for legacy_route in destination.legacy_routes:
        canonical_route = _CANONICAL_ROUTE_OVERRIDES.get(
            legacy_route,
            legacy_route if legacy_route in _ROUTABLE_LEGACY_ROUTES else destination.primary_route,
        )
        _ROUTE_MAP[legacy_route] = ResolvedShellRoute(
            destination.destination_id,
            canonical_route,
            legacy_route,
        )


def get_shell_destination(destination_id: str) -> ShellDestination:
    return _BY_DESTINATION_ID[destination_id]


def resolve_shell_route(route: str) -> ResolvedShellRoute:
    return _ROUTE_MAP.get(route, ResolvedShellRoute(route, route, route))
```

Adjust the canonical route list if tests reveal a mismatch with existing app routes.

- [ ] **Step 4: Update constants without deleting old constants**

In `tldw_chatbook/Constants.py`, keep all existing `TAB_*` constants. Add only new constants needed by the shell:

```python
TAB_HOME = "home"
TAB_LIBRARY = "library"
TAB_ARTIFACTS = "artifacts"
TAB_PERSONAS = "personas"
TAB_WATCHLISTS_COLLECTIONS = "watchlists_collections"
TAB_SCHEDULES = "schedules"
TAB_WORKFLOWS = "workflows"
TAB_MCP = "mcp"
TAB_ACP = "acp"
TAB_SKILLS = "skills"
TAB_SETTINGS = "settings"
```

Do not add these to `ALL_TABS` until the app route resolver can mount them.

- [ ] **Step 5: Run tests**

Run:

```bash
env HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/UI/test_shell_destinations.py Tests/UI/test_navigation_label_language.py --tb=short
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add tldw_chatbook/UI/Navigation/shell_destinations.py tldw_chatbook/Constants.py Tests/UI/test_shell_destinations.py
git commit -m "feat: add master shell destination model"
```

## Task 3: Add Home Route And Startup Compatibility

**Files:**
- Create: `tldw_chatbook/UI/Screens/home_screen.py`
- Modify: `tldw_chatbook/Constants.py`
- Modify: `tldw_chatbook/app.py`
- Modify: `Tests/UI/test_screen_navigation.py`
- Test: `Tests/UI/test_screen_navigation.py`

- [ ] **Step 1: Write failing route tests**

Add to `Tests/UI/test_screen_navigation.py`:

```python
def test_home_route_resolves_to_home_screen():
    app = _build_test_app()

    screen_name, current_tab, screen_class = app._resolve_screen_navigation_target("home")

    assert screen_name == "home"
    assert current_tab == "home"
    assert screen_class.__name__ == "HomeScreen"


def test_first_run_initial_route_defaults_to_home():
    app = _build_test_app()
    app.app_config["_first_run"] = True
    app._initial_tab_value = "chat"

    assert app._resolve_initial_shell_route() == "home"


@pytest.mark.parametrize("configured_route", ["home", "library", "settings", "notes"])
def test_returning_user_initial_route_preserves_configured_default(configured_route):
    app = _build_test_app()
    app.app_config["_first_run"] = False
    app._initial_tab_value = configured_route

    assert app._resolve_initial_shell_route() == configured_route


def test_startup_route_validation_accepts_shell_and_legacy_defaults():
    app = _build_test_app()

    for route in ["home", "library", "settings", "notes"]:
        assert app._normalize_initial_tab_from_config(route) == route


def test_startup_route_validation_rejects_unknown_default():
    app = _build_test_app()

    assert app._normalize_initial_tab_from_config("definitely-not-a-route") == "chat"
```

Expected: FAIL because `HomeScreen`, `_resolve_initial_shell_route()`, and `_normalize_initial_tab_from_config()` do not exist.

- [ ] **Step 2: Run failing tests**

Run:

```bash
env HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/UI/test_screen_navigation.py::test_home_route_resolves_to_home_screen Tests/UI/test_screen_navigation.py::test_first_run_initial_route_defaults_to_home Tests/UI/test_screen_navigation.py::test_returning_user_initial_route_preserves_configured_default Tests/UI/test_screen_navigation.py::test_startup_route_validation_accepts_shell_and_legacy_defaults Tests/UI/test_screen_navigation.py::test_startup_route_validation_rejects_unknown_default --tb=short
```

Expected: FAIL.

- [ ] **Step 3: Add minimal HomeScreen**

Create `tldw_chatbook/UI/Screens/home_screen.py`:

```python
"""Home dashboard screen for the master shell."""

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Static

from ..Navigation.base_app_screen import BaseAppScreen


class HomeScreen(BaseAppScreen):
    """Dashboard, notifications, readiness, and next-best action surface."""

    def __init__(self, app_instance, **kwargs):
        super().__init__(app_instance, "home", **kwargs)

    def compose_content(self) -> ComposeResult:
        with Vertical(id="home-dashboard"):
            yield Static("Home", id="home-title", classes="ds-destination-header")
            yield Static(
                "Dashboard, notifications, status, active work, and next actions.",
                id="home-purpose",
                classes="destination-purpose",
            )
```

This screen should remain structurally minimal in this task, but it must use the design-system destination-header class from the beginning so later Home dashboard work does not introduce a second header pattern.

- [ ] **Step 4: Wire Home in app route resolver**

In `tldw_chatbook/Constants.py`, add `TAB_HOME` to the master shell constants and include it in the routable tab set once the Home route exists. Do not change `TAB_CHAT = "chat"`.

In `tldw_chatbook/app.py`, import `HomeScreen` near other screen imports and update `_resolve_screen_navigation_target()`:

```python
from .UI.Screens.home_screen import HomeScreen
```

Add:

```python
screen_map = {
    "home": HomeScreen,
    "chat": ChatScreen,
    "ingest": MediaIngestScreen,
    "coding": CodingScreen,
    "conversation": ConversationScreen,
    "ccp": ConversationScreen,
    "media": MediaScreen,
    "notes": NotesScreen,
    "search": SearchScreen,
    "evals": EvalsScreen,
    "tools_settings": ToolsSettingsScreen,  # Temporary legacy mapping; Task 3A replaces this with MCPScreen.
    "llm": LLMScreen,
    "customize": CustomizeScreen,
    "logs": LogsScreen,
    "stats": StatsScreen,
    "stts": STTSScreen,
    "study": StudyScreen,
    "writing": WritingScreen,
    "research": ResearchScreen,
    "chatbooks": ChatbooksScreen,
    "subscription": SubscriptionScreen,
    "subscriptions": SubscriptionScreen,
}
```

Keep the fallback to `ChatScreen` for unknown or invalid routes so legacy startup failures remain recoverable.

Add startup resolution:

```python
def _valid_startup_route_ids(self) -> set[str]:
    """Routes allowed in default_tab before every shell route joins ALL_TABS."""
    from .UI.Navigation.shell_destinations import SHELL_DESTINATION_ORDER

    shell_routes = {destination.primary_route for destination in SHELL_DESTINATION_ORDER}
    legacy_aliases = {"conversation", "llm", "subscription", "subscriptions", "tools_settings"}
    return set(ALL_TABS) | shell_routes | legacy_aliases


def _normalize_initial_tab_from_config(self, configured_route: str | None) -> str:
    candidate = configured_route or TAB_CHAT
    if candidate in self._valid_startup_route_ids():
        return candidate
    logging.warning("Default tab '%s' from config not valid. Falling back to '%s'.", candidate, TAB_CHAT)
    return TAB_CHAT


def _resolve_initial_shell_route(self) -> str:
    """Return the startup route without losing returning-user preference."""
    if self.app_config.get("_first_run", False):
        return "home"
    return getattr(self, "_initial_tab_value", "chat")
```

Update the existing `__init__` default-tab validation path so it does not reject new shell routes before `_resolve_initial_shell_route()` runs:

```python
initial_tab_from_config = get_cli_setting("general", "default_tab", TAB_CHAT)
self._initial_tab_value = self._normalize_initial_tab_from_config(initial_tab_from_config)
```

Update `_push_initial_screen()` to call this helper:

```python
initial_tab = self._resolve_initial_shell_route()
```

This satisfies the spec requirement that first-run users land on Home while returning users keep their configured/default screen. It also prevents configured shell defaults such as `home`, `library`, or `settings` from being rejected only because they are not legacy `ALL_TABS` entries yet.

- [ ] **Step 5: Run focused tests**

Run:

```bash
env HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/UI/test_screen_navigation.py::test_home_route_resolves_to_home_screen Tests/UI/test_screen_navigation.py::test_first_run_initial_route_defaults_to_home Tests/UI/test_screen_navigation.py::test_returning_user_initial_route_preserves_configured_default Tests/UI/test_screen_navigation.py::test_startup_route_validation_accepts_shell_and_legacy_defaults Tests/UI/test_screen_navigation.py::test_startup_route_validation_rejects_unknown_default --tb=short
```

Expected: PASS. Do not assert `#nav-home` or `#nav-console` in this task; master-shell navigation is Task 4 so this PR slice remains independently shippable.

- [ ] **Step 6: Commit**

```bash
git add tldw_chatbook/UI/Screens/home_screen.py tldw_chatbook/Constants.py tldw_chatbook/app.py Tests/UI/test_screen_navigation.py
git commit -m "feat: add Home screen route"
```

## Task 3A: Add Minimal Destination Route Stubs Before Global Nav Exposure

**Files:**
- Create: `tldw_chatbook/UI/Screens/library_screen.py`
- Create: `tldw_chatbook/UI/Screens/library_conversations_screen.py`
- Create: `tldw_chatbook/UI/Screens/artifacts_screen.py`
- Create: `tldw_chatbook/UI/Screens/personas_screen.py`
- Create: `tldw_chatbook/UI/Screens/watchlists_collections_screen.py`
- Create: `tldw_chatbook/UI/Screens/schedules_screen.py`
- Create: `tldw_chatbook/UI/Screens/workflows_screen.py`
- Create: `tldw_chatbook/UI/Screens/mcp_screen.py`
- Create: `tldw_chatbook/UI/Screens/acp_screen.py`
- Create: `tldw_chatbook/UI/Screens/skills_screen.py`
- Create: `tldw_chatbook/UI/Screens/settings_screen.py`
- Modify: `tldw_chatbook/app.py`
- Modify: `Tests/UI/test_screen_navigation.py`
- Test: `Tests/UI/test_screen_navigation.py`

- [ ] **Step 1: Write failing all-destination route test**

Add to `Tests/UI/test_screen_navigation.py`:

```python
def test_all_master_shell_primary_routes_resolve_before_nav_exposure():
    app = _build_test_app()
    expected_routes = {
        "home",
        "chat",
        "library",
        "conversation",
        "artifacts",
        "personas",
        "watchlists_collections",
        "schedules",
        "workflows",
        "mcp",
        "acp",
        "skills",
        "settings",
    }

    unresolved = []
    for route in expected_routes:
        _screen_name, _tab_id, screen_class = app._resolve_screen_navigation_target(route)
        if screen_class is None:
            unresolved.append(route)

    assert unresolved == []


def test_conversation_route_uses_library_conversation_context():
    app = _build_test_app()

    screen_name, current_tab, screen_class = app._resolve_screen_navigation_target("conversation")

    assert screen_name == "conversation"
    assert current_tab == "conversation"
    assert screen_class.__name__ == "LibraryConversationsScreen"


def test_legacy_tools_settings_route_uses_mcp_context():
    app = _build_test_app()

    screen_name, current_tab, screen_class = app._resolve_screen_navigation_target("tools_settings")

    assert screen_name == "tools_settings"
    assert current_tab == "mcp"
    assert screen_class.__name__ == "MCPScreen"
```

Expected: FAIL until every primary shell route has at least a minimal screen.

- [ ] **Step 2: Add minimal reusable wrapper pattern**

Each stub should extend `BaseAppScreen`, use `.ds-destination-header` and `.ds-panel`, include an honest purpose line, and avoid pretending unavailable runtime features exist. Use only static route-safe links here. Later tasks enrich these wrappers.

Minimal structure:

```python
class LibraryScreen(BaseAppScreen):
    def __init__(self, app_instance, **kwargs):
        super().__init__(app_instance, "library", **kwargs)

    def compose_content(self) -> ComposeResult:
        with Vertical(id="library-shell"):
            yield Static("Library", id="library-title", classes="ds-destination-header")
            yield Static(
                "Source material, conversations, imports/exports, and Search/RAG.",
                id="library-purpose",
                classes="destination-purpose",
            )
            with Vertical(id="library-sections", classes="ds-panel"):
                yield Static("Notes | Media | Conversations | Import/Export | Search/RAG")
```

Apply the same minimal pattern to LibraryConversations, Artifacts, Personas, Watchlists+Collections, Schedules, Workflows, MCP, ACP, Skills, and Settings. `LibraryConversationsScreen` should pass `"library"` to `BaseAppScreen` so the Library nav destination remains active while the `conversation` route remains routable.

- [ ] **Step 3: Wire all primary shell routes**

In `_resolve_screen_navigation_target()`, map every primary shell route to the new minimal wrapper classes:

```python
"library": LibraryScreen,
"conversation": LibraryConversationsScreen,
"artifacts": ArtifactsScreen,
"personas": PersonasScreen,
"watchlists_collections": WatchlistsCollectionsScreen,
"schedules": SchedulesScreen,
"workflows": WorkflowsScreen,
"mcp": MCPScreen,
"tools_settings": MCPScreen,
"acp": ACPScreen,
"skills": SkillsScreen,
"settings": SettingsScreen,
```

Before adding the screen map entries, update the resolver aliases:

- remove any `screen_aliases["conversation"] = "ccp"` entry so `conversation` can resolve directly to `LibraryConversationsScreen`
- remove any `tab_aliases["conversation"] = TAB_CCP` entry and set conversation current-tab semantics to `"conversation"`
- set `tab_aliases["tools_settings"] = "mcp"` if a tab alias map is still used

Keep all other legacy routes mapped to their existing screens.
Replace any earlier temporary `"conversation": ConversationScreen` mapping from Task 3 with `"conversation": LibraryConversationsScreen` in this task so saved conversation browsing is Library-owned before global navigation is exposed.
Replace any earlier temporary `"tools_settings": ToolsSettingsScreen` mapping with `"tools_settings": MCPScreen` in this task so legacy tool-setting navigation opens the MCP/tool-control shell instead of broad global settings.

- [ ] **Step 4: Run focused route test**

Run:

```bash
env HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/UI/test_screen_navigation.py::test_all_master_shell_primary_routes_resolve_before_nav_exposure Tests/UI/test_screen_navigation.py::test_conversation_route_uses_library_conversation_context Tests/UI/test_screen_navigation.py::test_legacy_tools_settings_route_uses_mcp_context --tb=short
```

Expected: PASS.

- [ ] **Step 5: Run whitespace check**

Run:

```bash
git diff --check
```

Expected: no output.

- [ ] **Step 6: Commit**

```bash
git add tldw_chatbook/UI/Screens/library_screen.py tldw_chatbook/UI/Screens/library_conversations_screen.py tldw_chatbook/UI/Screens/artifacts_screen.py tldw_chatbook/UI/Screens/personas_screen.py tldw_chatbook/UI/Screens/watchlists_collections_screen.py tldw_chatbook/UI/Screens/schedules_screen.py tldw_chatbook/UI/Screens/workflows_screen.py tldw_chatbook/UI/Screens/mcp_screen.py tldw_chatbook/UI/Screens/acp_screen.py tldw_chatbook/UI/Screens/skills_screen.py tldw_chatbook/UI/Screens/settings_screen.py tldw_chatbook/app.py Tests/UI/test_screen_navigation.py
git commit -m "feat: add master shell route stubs"
```

## Task 4: Render Master Shell Navigation From Destination Model

**Files:**
- Modify: `tldw_chatbook/UI/Navigation/main_navigation.py`
- Modify: `tldw_chatbook/UI/Navigation/base_app_screen.py`
- Modify: `tldw_chatbook/app.py`
- Create: `Tests/UI/test_master_shell_navigation.py`
- Modify: `Tests/UI/test_screen_navigation.py`
- Modify: `Tests/UI/test_navigation_label_language.py`
- Test: `Tests/UI/test_master_shell_navigation.py`, `Tests/UI/test_screen_navigation.py`, `Tests/UI/test_navigation_label_language.py`

- [ ] **Step 1: Write failing navigation order test**

Create `Tests/UI/test_master_shell_navigation.py`:

```python
from unittest.mock import Mock

import pytest
from textual.app import App
from textual.widgets import Button

from tldw_chatbook.UI.Navigation.main_navigation import MainNavigationBar


@pytest.mark.asyncio
async def test_master_shell_navigation_order_and_labels():
    class TestApp(App):
        def compose(self):
            yield MainNavigationBar(active="home")

    app = TestApp()

    async with app.run_test(size=(180, 20)) as pilot:
        await pilot.pause(0.1)

        actual = [(button.id, str(button.label).strip()) for button in app.query(".nav-button")]

    assert actual == [
        ("nav-home", "Home"),
        ("nav-console", "Console"),
        ("nav-library", "Library"),
        ("nav-artifacts", "Artifacts"),
        ("nav-personas", "Personas"),
        ("nav-watchlists_collections", "Watchlists+Collections"),
        ("nav-schedules", "Schedules"),
        ("nav-workflows", "Workflows"),
        ("nav-mcp", "MCP"),
        ("nav-acp", "ACP"),
        ("nav-skills", "Skills"),
        ("nav-settings", "Settings"),
    ]
```

Expected: FAIL because current nav renders legacy labels.

- [ ] **Step 2: Write route-emission test**

Add:

```python
@pytest.mark.asyncio
async def test_master_shell_navigation_routes_to_primary_route():
    events = []

    class TestApp(App):
        def compose(self):
            yield MainNavigationBar(active="home")

        def on_mount(self):
            self.query_one("#nav-console", Button).press()

        def on_navigate_to_screen(self, message):
            events.append(message.screen_name)

    app = TestApp()

    async with app.run_test(size=(180, 20)) as pilot:
        await pilot.pause(0.1)

    assert events == ["chat"]
```

Expected: FAIL until nav maps `Console` to current `chat` route.

- [ ] **Step 3: Write all-visible-destinations route safety test**

Add:

```python
@pytest.mark.asyncio
async def test_every_visible_master_shell_nav_destination_resolves():
    from tldw_chatbook.UI.Navigation.shell_destinations import SHELL_DESTINATION_ORDER
    from Tests.UI.test_screen_navigation import _build_test_app

    app = _build_test_app()

    for destination in SHELL_DESTINATION_ORDER:
        _screen_name, _tab_id, screen_class = app._resolve_screen_navigation_target(destination.primary_route)
        assert screen_class is not None, destination.primary_route
```

Expected: PASS only because Task 3A mounted minimal wrappers for every primary shell route.

- [ ] **Step 4: Run failing tests**

Run:

```bash
env HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/UI/test_master_shell_navigation.py --tb=short
```

Expected: FAIL.

- [ ] **Step 5: Update MainNavigationBar to use `SHELL_DESTINATION_ORDER`**

Replace local `NAV_GROUPS` and `NAV_TOOLTIPS` usage in `tldw_chatbook/UI/Navigation/main_navigation.py` with destination metadata:

```python
from .shell_destinations import SHELL_DESTINATION_ORDER, get_shell_destination
```

In `compose()`:

```python
for i, destination in enumerate(SHELL_DESTINATION_ORDER):
    if i > 0:
        yield Static("·", classes="nav-separator")

    button = Button(
        destination.label,
        id=f"nav-{destination.destination_id}",
        classes="nav-button",
        tooltip=destination.tooltip,
    )
    if destination.destination_id == self.active_screen:
        button.add_class("is-active")
    yield button
```

In `handle_navigation()`:

```python
destination_id = button_id.replace("nav-", "")
destination = get_shell_destination(destination_id)
screen_name = destination.primary_route
```

If `active_screen` is a legacy route such as `chat`, normalize it through `resolve_shell_route()` before comparing active state. Use `.is-active` for the selected destination and avoid introducing a new `.active` state class.

- [ ] **Step 6: Update screen active names**

Each new destination wrapper should pass the master destination id to `BaseAppScreen`; for legacy screens still mounted directly, normalize active nav through `resolve_shell_route(self.screen_name).destination_id`.

Prefer this change inside `BaseAppScreen.compose()`:

```python
from .shell_destinations import resolve_shell_route

active_destination = resolve_shell_route(self.screen_name).destination_id
yield MainNavigationBar(active=active_destination)
```

- [ ] **Step 7: Run focused nav tests**

Run:

```bash
env HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/UI/test_master_shell_navigation.py Tests/UI/test_screen_navigation.py Tests/UI/test_navigation_label_language.py --tb=short
```

Expected: PASS after expected old-label tests are updated.

- [ ] **Step 8: Run whitespace check**

Run:

```bash
git diff --check
```

Expected: no output.

- [ ] **Step 9: Commit**

```bash
git add tldw_chatbook/UI/Navigation/main_navigation.py tldw_chatbook/UI/Navigation/base_app_screen.py Tests/UI/test_master_shell_navigation.py Tests/UI/test_screen_navigation.py Tests/UI/test_navigation_label_language.py
git commit -m "feat: render master shell navigation"
```

## Task 5: Add Pure Home Dashboard State And Next-Best Action Logic

**Files:**
- Create: `tldw_chatbook/Home/__init__.py`
- Create: `tldw_chatbook/Home/dashboard_state.py`
- Create: `Tests/Home/test_dashboard_state.py`
- Test: `Tests/Home/test_dashboard_state.py`

- [ ] **Step 1: Write failing Home state tests**

Create `Tests/Home/test_dashboard_state.py`:

```python
from tldw_chatbook.Home.dashboard_state import (
    HomeDashboardInput,
    choose_next_best_action,
    summarize_home_dashboard,
)


def test_next_best_action_prioritizes_blockers():
    state = HomeDashboardInput(
        model_ready=False,
        pending_approval_count=2,
        active_run_count=1,
        has_library_content=True,
    )

    action = choose_next_best_action(state)

    assert action.action_id == "fix_model_setup"
    assert action.label == "Set up Console model"


def test_next_best_action_prioritizes_pending_approval_after_readiness():
    state = HomeDashboardInput(
        model_ready=True,
        pending_approval_count=2,
        active_run_count=1,
        has_library_content=True,
    )

    action = choose_next_best_action(state)

    assert action.action_id == "review_approvals"
    assert action.target_route == "chat"


def test_dashboard_summary_exposes_required_sections():
    dashboard = summarize_home_dashboard(
        HomeDashboardInput(
            model_ready=True,
            pending_approval_count=0,
            active_run_count=0,
            has_library_content=False,
        )
    )

    assert [section.section_id for section in dashboard.sections] == [
        "status",
        "attention",
        "active_work",
        "next_best_action",
        "recent_work",
    ]


def test_dashboard_summary_exposes_lightweight_control_ids_for_core_states():
    dashboard = summarize_home_dashboard(
        HomeDashboardInput(
            model_ready=True,
            pending_approval_count=1,
            running_run_count=1,
            paused_run_count=1,
            failed_run_count=1,
            failed_schedule_count=1,
            active_run_count=3,
            has_library_content=True,
        )
    )

    control_ids = {control.control_id for control in dashboard.controls}
    assert {
        "home-approve",
        "home-reject",
        "home-pause",
        "home-resume",
        "home-retry",
        "home-open-details",
        "home-open-in-console",
    }.issubset(control_ids)
```

Expected: FAIL because module does not exist.

- [ ] **Step 2: Run failing tests**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/Home/test_dashboard_state.py --tb=short
```

Expected: FAIL.

- [ ] **Step 3: Implement pure state module**

Create `tldw_chatbook/Home/dashboard_state.py`:

```python
"""Pure Home dashboard state and next-best-action selection."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HomeDashboardInput:
    model_ready: bool = False
    mcp_ready: bool = True
    acp_ready: bool = True
    rag_ready: bool = False
    pending_approval_count: int = 0
    active_run_count: int = 0
    running_run_count: int = 0
    paused_run_count: int = 0
    failed_run_count: int = 0
    failed_schedule_count: int = 0
    has_library_content: bool = False
    has_recent_work: bool = False
    active_detail_route: str = "chat"


@dataclass(frozen=True)
class HomeAction:
    action_id: str
    label: str
    target_route: str
    reason: str


@dataclass(frozen=True)
class HomeSection:
    section_id: str
    title: str
    lines: tuple[str, ...]


@dataclass(frozen=True)
class HomeControl:
    control_id: str
    label: str
    target_route: str
    applies_to: str


@dataclass(frozen=True)
class HomeDashboard:
    next_action: HomeAction
    sections: tuple[HomeSection, ...]
    controls: tuple[HomeControl, ...]


def choose_next_best_action(state: HomeDashboardInput) -> HomeAction:
    if not state.model_ready:
        return HomeAction("fix_model_setup", "Set up Console model", "llm", "Console needs a working model before live AI tasks.")
    if state.pending_approval_count:
        return HomeAction("review_approvals", "Review pending approvals", "chat", "Agent work is waiting for a decision.")
    if state.failed_schedule_count:
        return HomeAction("recover_schedules", "Review failed schedules", "schedules", "Scheduled work needs recovery.")
    if state.active_run_count:
        return HomeAction("resume_active_work", "Resume active work", "chat", "Live work is already running.")
    if not state.has_library_content:
        return HomeAction("import_sources", "Import Library sources", "library", "Library content makes Console and RAG more useful.")
    if state.rag_ready:
        return HomeAction("search_library", "Search your Library", "library", "Search/RAG is ready over saved content.")
    return HomeAction("start_console", "Start in Console", "chat", "Console is ready for a task.")


def build_home_controls(state: HomeDashboardInput) -> tuple[HomeControl, ...]:
    controls: list[HomeControl] = []
    if state.pending_approval_count:
        controls.extend(
            (
                HomeControl("home-approve", "Approve", "chat", "approval"),
                HomeControl("home-reject", "Reject", "chat", "approval"),
            )
        )
    if state.running_run_count or state.active_run_count:
        controls.append(HomeControl("home-pause", "Pause", "chat", "running_work"))
    if state.paused_run_count:
        controls.append(HomeControl("home-resume", "Resume", "chat", "paused_work"))
    if state.failed_run_count or state.failed_schedule_count:
        controls.append(HomeControl("home-retry", "Retry", "schedules", "failed_work"))
    if (
        state.pending_approval_count
        or state.active_run_count
        or state.running_run_count
        or state.paused_run_count
        or state.failed_run_count
        or state.failed_schedule_count
    ):
        controls.extend(
            (
                HomeControl("home-open-details", "Open details", state.active_detail_route, "work_details"),
                HomeControl("home-open-in-console", "Open in Console", "chat", "console"),
            )
        )
    return tuple(controls)


def summarize_home_dashboard(state: HomeDashboardInput) -> HomeDashboard:
    next_action = choose_next_best_action(state)
    approval_label = "Approval required" if state.pending_approval_count else "Ready"
    return HomeDashboard(
        next_action=next_action,
        sections=(
            HomeSection("status", "Status", (f"Model: {'Ready' if state.model_ready else 'Blocked'}",)),
            HomeSection("attention", "Attention", (approval_label, f"Pending approvals: {state.pending_approval_count}")),
            HomeSection("active_work", "Active Work", (f"Running: {state.running_run_count}", f"Paused: {state.paused_run_count}", f"Failed: {state.failed_run_count}")),
            HomeSection("next_best_action", "Next Best Action", (next_action.label, next_action.reason)),
            HomeSection("recent_work", "Recent Work", ("Recent work available" if state.has_recent_work else "No recent work yet",)),
        ),
        controls=build_home_controls(state),
    )
```

- [ ] **Step 4: Export Home helpers**

Create `tldw_chatbook/Home/__init__.py`:

```python
from .dashboard_state import (
    HomeAction,
    HomeControl,
    HomeDashboard,
    HomeDashboardInput,
    HomeSection,
    build_home_controls,
    choose_next_best_action,
    summarize_home_dashboard,
)

__all__ = [
    "HomeAction",
    "HomeControl",
    "HomeDashboard",
    "HomeDashboardInput",
    "HomeSection",
    "build_home_controls",
    "choose_next_best_action",
    "summarize_home_dashboard",
]
```

- [ ] **Step 5: Run tests**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/Home/test_dashboard_state.py --tb=short
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add tldw_chatbook/Home Tests/Home
git commit -m "feat: add Home dashboard state model"
```

## Task 6: Render Home Dashboard Sections And Lightweight Controls

**Files:**
- Modify: `tldw_chatbook/UI/Screens/home_screen.py`
- Create: `Tests/UI/test_home_screen.py`
- Test: `Tests/UI/test_home_screen.py`, `Tests/Home/test_dashboard_state.py`

- [ ] **Step 1: Write failing mounted Home tests**

Create `Tests/UI/test_home_screen.py`:

```python
from unittest.mock import Mock

import pytest

from tldw_chatbook.Chat.chat_handoff_models import ChatHandoffPayload
from tldw_chatbook.Home.dashboard_state import HomeDashboardInput
from tldw_chatbook.UI.Navigation.main_navigation import NavigateToScreen
from Tests.UI.test_screen_navigation import _build_test_app


@pytest.mark.asyncio
async def test_home_screen_shows_dashboard_sections():
    app = _build_test_app()

    async with app.run_test(size=(160, 40)) as pilot:
        await app.handle_screen_navigation(NavigateToScreen("home"))
        await pilot.pause(0.1)

        assert app.query_one("#home-title").has_class("ds-destination-header")
        assert app.query_one("#home-next-best-action").has_class("ds-panel")
        for selector in [
            "#home-status",
            "#home-attention",
            "#home-active-work",
            "#home-next-best-action",
            "#home-recent-work",
        ]:
            assert app.query_one(selector)


@pytest.mark.asyncio
async def test_home_primary_action_opens_target_route():
    app = _build_test_app()
    seen = []

    original_handler = app.handle_screen_navigation

    async def capture(message):
        seen.append(message.screen_name)
        await original_handler(message)

    app.handle_screen_navigation = capture

    async with app.run_test(size=(160, 40)) as pilot:
        await app.handle_screen_navigation(NavigateToScreen("home"))
        await pilot.pause(0.1)
        await pilot.click("#home-primary-action")
        await pilot.pause(0.1)

    assert seen[-1] in {"chat", "llm", "library", "schedules"}


@pytest.mark.asyncio
async def test_home_screen_shows_lightweight_agent_and_schedule_controls():
    app = _build_test_app()
    app._home_dashboard_test_input = HomeDashboardInput(
        model_ready=True,
        pending_approval_count=1,
        running_run_count=1,
        paused_run_count=1,
        failed_run_count=1,
        failed_schedule_count=1,
        active_run_count=3,
        has_library_content=True,
    )

    async with app.run_test(size=(160, 40)) as pilot:
        await app.handle_screen_navigation(NavigateToScreen("home"))
        await pilot.pause(0.1)

        for selector in [
            "#home-approve",
            "#home-reject",
            "#home-pause",
            "#home-resume",
            "#home-retry",
            "#home-open-details",
            "#home-open-in-console",
        ]:
            assert app.query_one(selector).has_class("ds-toolbar")


@pytest.mark.asyncio
async def test_home_control_clicks_call_available_runtime_hooks():
    app = _build_test_app()
    app._home_dashboard_test_input = HomeDashboardInput(
        model_ready=True,
        pending_approval_count=1,
        running_run_count=1,
        paused_run_count=1,
        failed_run_count=1,
        failed_schedule_count=1,
        active_run_count=3,
        has_library_content=True,
    )
    app.approve_active_home_item = Mock()
    app.reject_active_home_item = Mock()
    app.pause_active_home_item = Mock()
    app.resume_active_home_item = Mock()
    app.retry_active_home_item = Mock()

    async with app.run_test(size=(160, 40)) as pilot:
        await app.handle_screen_navigation(NavigateToScreen("home"))
        await pilot.pause(0.1)
        for selector in [
            "#home-approve",
            "#home-reject",
            "#home-pause",
            "#home-resume",
            "#home-retry",
        ]:
            await pilot.click(selector)
            await pilot.pause(0.1)

    app.approve_active_home_item.assert_called_once()
    app.reject_active_home_item.assert_called_once()
    app.pause_active_home_item.assert_called_once()
    app.resume_active_home_item.assert_called_once()
    app.retry_active_home_item.assert_called_once()


@pytest.mark.asyncio
async def test_home_detail_controls_route_to_owner_and_console():
    app = _build_test_app()
    app._home_dashboard_test_input = HomeDashboardInput(
        model_ready=True,
        pending_approval_count=1,
        active_run_count=1,
        has_library_content=True,
        active_detail_route="workflows",
    )
    seen = []

    original_handler = app.handle_screen_navigation

    async def capture(message):
        seen.append(message.screen_name)
        await original_handler(message)

    app.handle_screen_navigation = capture

    async with app.run_test(size=(160, 40)) as pilot:
        await app.handle_screen_navigation(NavigateToScreen("home"))
        await pilot.pause(0.1)
        await pilot.click("#home-open-details")
        await pilot.pause(0.1)
        await app.handle_screen_navigation(NavigateToScreen("home"))
        await pilot.pause(0.1)
        await pilot.click("#home-open-in-console")
        await pilot.pause(0.1)

    assert "workflows" in seen
    assert "chat" in seen


@pytest.mark.asyncio
async def test_pending_chat_handoff_does_not_create_live_work_controls():
    app = _build_test_app()
    app.pending_chat_handoff = ChatHandoffPayload(
        source="library",
        item_type="note",
        title="Research note",
        body="Context to stage in Console.",
    )

    async with app.run_test(size=(160, 40)) as pilot:
        await app.handle_screen_navigation(NavigateToScreen("home"))
        await pilot.pause(0.1)

        assert len(app.query("#home-pause")) == 0
        assert len(app.query("#home-resume")) == 0
        assert len(app.query("#home-retry")) == 0
        assert len(app.query("#home-open-in-console")) == 0
```

Expected: FAIL because sections and button do not exist.

- [ ] **Step 2: Run failing tests**

Run:

```bash
env HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/UI/test_home_screen.py --tb=short
```

Expected: FAIL.

- [ ] **Step 3: Add Home dashboard rendering**

In `tldw_chatbook/UI/Screens/home_screen.py`, derive a `HomeDashboardInput` from app attributes conservatively:

```python
def _build_dashboard_input(self) -> HomeDashboardInput:
    test_override = getattr(self.app_instance, "_home_dashboard_test_input", None)
    if test_override is not None:
        return test_override

    providers = getattr(self.app_instance, "providers_models", {}) or {}
    model_ready = bool(providers)
    pending_launch = getattr(self.app_instance, "pending_console_launch", None)
    live_work_active = isinstance(pending_launch, dict) and bool(pending_launch)
    active_detail_route = "chat"
    if isinstance(pending_launch, dict):
        source_route = pending_launch.get("source")
        if source_route in {"watchlists_collections", "schedules", "workflows", "acp"}:
            active_detail_route = source_route
    return HomeDashboardInput(
        model_ready=model_ready,
        pending_approval_count=0,
        active_run_count=1 if live_work_active else 0,
        running_run_count=1 if live_work_active else 0,
        has_library_content=False,
        has_recent_work=bool(getattr(self.app_instance, "_screen_states", {})),
        active_detail_route=active_detail_route,
    )
```

Do not count `pending_chat_handoff` as active or running work. `pending_chat_handoff` is staged context for Console; Home may surface it later as recent/staged context, but it must not create pause/resume/retry/open-live controls.

Render each section with stable IDs:

```python
for section in dashboard.sections:
    section_id = section.section_id.replace("_", "-")
    yield Static(section.title, id=f"home-{section_id}", classes="ds-panel")
    yield Static("\\n".join(section.lines), id=f"home-{section_id}-body")
```

Render primary action as a `Button`:

```python
yield Button(dashboard.next_action.label, id="home-primary-action")
```

Render lightweight controls with stable IDs. These controls may initially dispatch to existing screens or app hooks; do not perform irreversible approve/reject/pause/resume/retry behavior without an existing runtime method to call.

```python
for control in dashboard.controls:
    yield Button(control.label, id=control.control_id, classes="ds-toolbar")
```

Use readable status labels in section copy (`Ready`, `Blocked`, `Approval required`, `Running`, `Paused`, `Unavailable`) rather than color-only glyphs.

Handle the primary action and lightweight controls through one button handler. Controls call named app hooks when available; otherwise they route to the safe destination carried by the control:

```python
HOME_CONTROL_METHODS = {
    "home-approve": "approve_active_home_item",
    "home-reject": "reject_active_home_item",
    "home-pause": "pause_active_home_item",
    "home-resume": "resume_active_home_item",
    "home-retry": "retry_active_home_item",
}

@on(Button.Pressed)
def handle_home_button(self, event: Button.Pressed) -> None:
    button_id = event.button.id
    if button_id == "home-primary-action":
        self.post_message(NavigateToScreen(self._current_dashboard.next_action.target_route))
        return

    control = next((item for item in self._current_dashboard.controls if item.control_id == button_id), None)
    if control is None:
        return

    method_name = HOME_CONTROL_METHODS.get(control.control_id)
    method = getattr(self.app_instance, method_name, None) if method_name else None
    if callable(method):
        method()
    else:
        self.post_message(NavigateToScreen(control.target_route))
```

Use the exact `@on` import pattern already used in nearby screens.

- [ ] **Step 4: Run focused tests**

Run:

```bash
env HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/UI/test_home_screen.py Tests/Home/test_dashboard_state.py --tb=short
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/UI/Screens/home_screen.py Tests/UI/test_home_screen.py
git commit -m "feat: render Home dashboard"
```

## Task 7: Reframe Chat As Console In User-Facing Shell Copy

**Files:**
- Modify: `tldw_chatbook/Constants.py`
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py`
- Modify: `tldw_chatbook/UI/Chat_Window_Enhanced.py`
- Modify: `tldw_chatbook/app.py`
- Modify: `Tests/UI/test_navigation_label_language.py`
- Modify: `Tests/UI/test_chat_first_run_orientation.py`
- Modify: `Tests/UI/test_command_palette_providers.py`
- Modify: `Tests/UI/test_command_palette_basic.py`
- Create: `Tests/UI/test_command_palette_shell_routes.py`
- Test: `Tests/UI/test_navigation_label_language.py`, `Tests/UI/test_chat_first_run_orientation.py`, `Tests/UI/test_chat_screen_state.py`, `Tests/UI/test_command_palette_shell_routes.py`, `Tests/UI/test_command_palette_providers.py`, `Tests/UI/test_command_palette_basic.py`

- [ ] **Step 1: Write failing Console label tests**

Update or add tests:

```python
def test_legacy_chat_route_uses_console_user_label() -> None:
    from tldw_chatbook.Constants import TAB_CHAT, get_tab_display_label

    assert TAB_CHAT == "chat"
    assert get_tab_display_label(TAB_CHAT) == "Console"


def test_tools_settings_label_is_mcp_not_global_settings() -> None:
    from tldw_chatbook.Constants import TAB_MCP, TAB_SETTINGS, TAB_TOOLS_SETTINGS, get_tab_display_label

    assert get_tab_display_label(TAB_MCP) == "MCP"
    assert get_tab_display_label(TAB_TOOLS_SETTINGS) == "MCP"
    assert get_tab_display_label(TAB_SETTINGS) == "Settings"
```

Add a mounted assertion in `Tests/UI/test_chat_first_run_orientation.py` that the first-run text contains `Console` and does not describe the destination as merely a generic chat tab.

Create `Tests/UI/test_command_palette_shell_routes.py`:

```python
from tldw_chatbook.Constants import (
    ALL_TABS,
    TAB_CHATBOOKS,
    TAB_CCP,
    TAB_CODING,
    TAB_CUSTOMIZE,
    TAB_EVALS,
    TAB_INGEST,
    TAB_LLM,
    TAB_LOGS,
    TAB_MEDIA,
    TAB_MCP,
    TAB_NOTES,
    TAB_RESEARCH,
    TAB_SEARCH,
    TAB_SETTINGS,
    TAB_STATS,
    TAB_STTS,
    TAB_STUDY,
    TAB_SUBSCRIPTIONS,
    TAB_TOOLS_SETTINGS,
    TAB_WRITING,
)
from tldw_chatbook.app import TabNavigationProvider


def test_tab_navigation_provider_routes_settings_and_mcp_separately():
    assert TabNavigationProvider.route_for_tab(TAB_SETTINGS) == "settings"
    assert TabNavigationProvider.route_for_tab(TAB_MCP) == "mcp"
    assert TabNavigationProvider.route_for_tab(TAB_TOOLS_SETTINGS) == "mcp"
    assert TabNavigationProvider.route_for_tab("llm") == TAB_LLM


def test_tab_navigation_provider_includes_settings_and_mcp_shell_commands():
    tab_ids = TabNavigationProvider.navigation_tab_ids()

    assert TAB_SETTINGS in tab_ids
    assert TAB_MCP in tab_ids
    assert TAB_TOOLS_SETTINGS not in tab_ids


def test_tab_navigation_provider_preserves_all_legacy_direct_commands():
    primary_shell_ids = set(TabNavigationProvider.navigation_tab_ids())
    legacy_tab_ids = set(ALL_TABS) - primary_shell_ids

    command_tab_ids = set(TabNavigationProvider.command_palette_tab_ids())

    assert legacy_tab_ids.issubset(command_tab_ids)
    representative_direct_tabs = [
        TAB_CCP,
        TAB_LLM,
        TAB_NOTES,
        TAB_MEDIA,
        TAB_SEARCH,
        TAB_INGEST,
        TAB_SUBSCRIPTIONS,
        TAB_CHATBOOKS,
        TAB_STTS,
        TAB_EVALS,
        TAB_STUDY,
        TAB_CODING,
        TAB_LOGS,
        TAB_STATS,
        TAB_CUSTOMIZE,
        TAB_WRITING,
        TAB_RESEARCH,
    ]

    for tab_id in representative_direct_tabs:
        assert TabNavigationProvider.route_for_tab(tab_id) == tab_id
    assert TabNavigationProvider.route_for_tab(TAB_TOOLS_SETTINGS) == "mcp"
    assert TabNavigationProvider.route_for_tab("llm") == TAB_LLM


def test_tab_navigation_provider_copy_uses_shell_vocabulary():
    assert "global preferences" in TabNavigationProvider.TAB_HELP_TEXT[TAB_SETTINGS]
    assert "MCP" in TabNavigationProvider.TAB_HELP_TEXT[TAB_MCP]
    assert "MCP" in TabNavigationProvider.TAB_HELP_TEXT[TAB_TOOLS_SETTINGS]
```

Update existing command-palette tests so they match the migration rather than protecting old behavior:

```python
@pytest.mark.asyncio
async def test_search_shows_shell_and_legacy_direct_commands(tab_provider):
    hits = [hit async for hit in tab_provider.search("tab")]

    assert len(hits) == len(TabNavigationProvider.command_palette_tab_ids())
    tab_texts = [hit.text for hit in hits]
    assert any("Console" in text for text in tab_texts)
    assert any("Library" in text for text in tab_texts)
    assert any("Settings" in text for text in tab_texts)
    assert any("Models" in text for text in tab_texts)
    assert any("Speech" in text for text in tab_texts)
    assert any("Coding" in text for text in tab_texts)


def test_switch_tab_posts_navigation_message(tab_provider):
    tab_provider.switch_tab(TAB_NOTES)

    tab_provider.app.post_message.assert_called_once()
    message = tab_provider.app.post_message.call_args.args[0]
    assert message.screen_name == TabNavigationProvider.route_for_tab(TAB_NOTES)
    tab_provider.app.notify.assert_called_once()


def test_all_command_palette_tabs_are_navigable(tab_provider):
    for tab_id in TabNavigationProvider.command_palette_tab_ids():
        tab_provider.app.post_message.reset_mock()
        tab_provider.switch_tab(tab_id)
        message = tab_provider.app.post_message.call_args.args[0]
        assert message.screen_name == TabNavigationProvider.route_for_tab(tab_id)
```

In `Tests/UI/test_command_palette_providers.py`, replace stale assertions that `search("tab")` returns `len(ALL_TABS)` or that `switch_tab()` mutates `app.current_tab` directly. In `Tests/UI/test_command_palette_basic.py`, assert the provider exposes `route_for_tab`, `navigation_tab_ids`, `legacy_command_tab_ids`, and `command_palette_tab_ids`, and that `switch_tab()` can post a navigation message without requiring direct tab mutation.

- [ ] **Step 2: Run failing tests**

Run:

```bash
env HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/UI/test_navigation_label_language.py Tests/UI/test_chat_first_run_orientation.py Tests/UI/test_command_palette_shell_routes.py Tests/UI/test_command_palette_providers.py Tests/UI/test_command_palette_basic.py --tb=short
```

Expected: FAIL until copy changes.

- [ ] **Step 3: Update shared labels**

In `tldw_chatbook/Constants.py`:

```python
TAB_DISPLAY_LABELS = {
    TAB_CHAT: "Console",
    TAB_CCP: "Personas",
    TAB_NOTES: "Notes",
    TAB_MEDIA: "Media",
    TAB_SEARCH: "Search",
    TAB_INGEST: "Import",
    TAB_EVALS: "Evals",
    TAB_LLM: "Models",
    TAB_TOOLS_SETTINGS: "MCP",
    TAB_STATS: "Stats",
    TAB_LOGS: "Logs",
    TAB_CODING: "Coding",
    TAB_STTS: "Speech",
    TAB_STUDY: "Study",
    TAB_WRITING: "Writing",
    TAB_RESEARCH: "Research",
    TAB_SUBSCRIPTIONS: "Watchlists+Collections",
    TAB_CHATBOOKS: "Chatbooks",
    TAB_MCP: "MCP",
    TAB_SETTINGS: "Settings",
    TAB_CUSTOMIZE: "Customize",
}
```

Do not change `TAB_CHAT = "chat"`.

- [ ] **Step 4: Update command palette shell routing and help text**

In `tldw_chatbook/app.py`, update `TabNavigationProvider.TAB_HELP_TEXT`:

```python
TAB_CHAT: "Switch to Console for live agent work, chat, RAG, approvals, and tool runs",
TAB_MCP: "Switch to MCP tools, servers, permissions, auth, and audit",
TAB_TOOLS_SETTINGS: "Switch to MCP tools, servers, permissions, auth, and audit",
TAB_SETTINGS: "Switch to Settings for global preferences, appearance, accounts, and storage",
```

Add a routing helper and use `NavigateToScreen(...)` instead of only setting `current_tab`:

```python
@classmethod
def navigation_tab_ids(cls) -> tuple[str, ...]:
    """Primary shell destinations used by top navigation."""
    return (
        TAB_HOME,
        TAB_CHAT,
        TAB_LIBRARY,
        TAB_ARTIFACTS,
        TAB_PERSONAS,
        TAB_WATCHLISTS_COLLECTIONS,
        TAB_SCHEDULES,
        TAB_WORKFLOWS,
        TAB_MCP,
        TAB_ACP,
        TAB_SKILLS,
        TAB_SETTINGS,
    )

@classmethod
def legacy_command_tab_ids(cls) -> tuple[str, ...]:
    """Direct legacy commands kept for power-user command palette speed."""
    primary_shell_ids = set(cls.navigation_tab_ids())
    legacy_ids = tuple(
        tab_id for tab_id in ALL_TABS
        if tab_id not in primary_shell_ids
    )
    return legacy_ids

@classmethod
def command_palette_tab_ids(cls) -> tuple[str, ...]:
    """Primary shell commands first, followed by searchable legacy direct routes."""
    return cls.navigation_tab_ids() + tuple(
        tab_id for tab_id in cls.legacy_command_tab_ids()
        if tab_id not in cls.navigation_tab_ids()
    )

@staticmethod
def route_for_tab(tab_id: str) -> str:
    if tab_id == TAB_SETTINGS:
        return "settings"
    if tab_id in {TAB_MCP, TAB_TOOLS_SETTINGS}:
        return "mcp"
    if tab_id == TAB_CHAT:
        return "chat"
    if tab_id == "llm":
        return TAB_LLM
    return tab_id

def switch_tab(self, tab_id: str) -> None:
    route = self.route_for_tab(tab_id)
    self.app.post_message(NavigateToScreen(route))
    self.app.notify(f"Switched to {get_tab_display_label(tab_id)}", severity="information")
```

Update `search()` and `discover()` to build commands from `command_palette_tab_ids()`, not raw `ALL_TABS` and not only `navigation_tab_ids()`. Primary shell destinations should sort first and use the new IA vocabulary; every existing direct command from `ALL_TABS` that is not a primary shell destination must remain searchable, including Notes, Media, Search, Ingest, CCP/Personas legacy, Subscriptions, Chatbooks, Models, Speech, Evals, Study, Coding, Logs, Stats, Customize, Writing, and Research. Command execution still goes through `NavigateToScreen(route_for_tab(tab_id))` so shell destinations and route aliases stay compatible.

- [ ] **Step 5: Update Chat screen/window visible copy**

Replace user-facing text only:

- `Chat` title -> `Console`
- `Chat handoff` -> `Console handoff` only if the phrase is visible to users
- `Use in Chat` can remain during migration if changing it would be a larger cross-surface flow; document that it will become `Use in Console` in a later slice.

Do not rename classes or files.

- [ ] **Step 6: Run focused tests**

Run:

```bash
env HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/UI/test_navigation_label_language.py Tests/UI/test_chat_first_run_orientation.py Tests/UI/test_chat_screen_state.py Tests/UI/test_command_palette_shell_routes.py Tests/UI/test_command_palette_providers.py Tests/UI/test_command_palette_basic.py --tb=short
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add tldw_chatbook/Constants.py tldw_chatbook/app.py tldw_chatbook/UI/Screens/chat_screen.py tldw_chatbook/UI/Chat_Window_Enhanced.py Tests/UI/test_navigation_label_language.py Tests/UI/test_chat_first_run_orientation.py Tests/UI/test_command_palette_shell_routes.py Tests/UI/test_command_palette_providers.py Tests/UI/test_command_palette_basic.py
git commit -m "feat: reframe Chat as Console"
```

## Task 8: Add Library, Artifacts, And Personas Destination Wrappers

**Files:**
- Modify: `tldw_chatbook/UI/Screens/library_screen.py`
- Modify: `tldw_chatbook/UI/Screens/artifacts_screen.py`
- Modify: `tldw_chatbook/UI/Screens/personas_screen.py`
- Modify: `tldw_chatbook/app.py`
- Create: `Tests/UI/test_destination_shells.py`
- Test: `Tests/UI/test_destination_shells.py`, `Tests/UI/test_screen_navigation.py`

- [ ] **Step 1: Write failing destination wrapper tests**

Create `Tests/UI/test_destination_shells.py`:

```python
import pytest

from Tests.UI.test_screen_navigation import _build_test_app
from tldw_chatbook.UI.Navigation.main_navigation import NavigateToScreen


@pytest.mark.parametrize(
    ("route", "title_id", "purpose_text"),
    [
        ("library", "#library-title", "source material"),
        ("artifacts", "#artifacts-title", "generated"),
        ("personas", "#personas-title", "behavior"),
    ],
)
@pytest.mark.asyncio
async def test_primary_destination_wrappers_mount(route, title_id, purpose_text):
    app = _build_test_app()

    async with app.run_test(size=(160, 40)) as pilot:
        await app.handle_screen_navigation(NavigateToScreen(route))
        await pilot.pause(0.1)

        title = app.query_one(title_id)
        assert title
        assert title.has_class("ds-destination-header")
        assert purpose_text in str(app.screen.query_one(".destination-purpose").renderable).lower()
        assert app.screen.query_one(".ds-panel")


@pytest.mark.asyncio
async def test_library_exposes_source_sections_and_import_export_boundary():
    app = _build_test_app()

    async with app.run_test(size=(160, 40)) as pilot:
        await app.handle_screen_navigation(NavigateToScreen("library"))
        await pilot.pause(0.1)

        for selector in [
            "#library-open-notes",
            "#library-open-media",
            "#library-open-conversations",
            "#library-open-import-export",
            "#library-open-search",
        ]:
            assert app.query_one(selector)
```

Expected: FAIL because the minimal wrappers from Task 3A do not yet expose the required Library source sections/actions.

- [ ] **Step 2: Run failing tests**

Run:

```bash
env HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/UI/test_destination_shells.py --tb=short
```

Expected: FAIL.

- [ ] **Step 3: Implement minimal wrapper pattern**

Each wrapper should extend `BaseAppScreen`, pass the master destination id, render a title, purpose line, primary action, section list, and compatibility actions that route to existing screens. Use `.ds-destination-header` for the title/header region and `.ds-panel` for grouped section links; do not add screen-local panel/header classes unless they wrap the design-system classes.

Library must explicitly expose `Notes`, `Media`, `Conversations`, `Import/Export`, and `Search/RAG`. In this plan, `conversation` means saved conversation browsing/source access under Library. Persona/character/prompt behavior remains under Personas.

Example for `library_screen.py`:

```python
"""Library destination wrapper."""

from textual import on
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Button, Static

from ..Navigation.base_app_screen import BaseAppScreen
from ..Navigation.main_navigation import NavigateToScreen


class LibraryScreen(BaseAppScreen):
    def __init__(self, app_instance, **kwargs):
        super().__init__(app_instance, "library", **kwargs)

    def compose_content(self) -> ComposeResult:
        with Vertical(id="library-shell"):
            yield Static("Library", id="library-title", classes="ds-destination-header")
            yield Static(
                "Source material, notes, media, conversations, imports/exports, and Search/RAG.",
                id="library-purpose",
                classes="destination-purpose",
            )
            with Vertical(id="library-sections", classes="ds-panel"):
                yield Button("Open Notes", id="library-open-notes")
                yield Button("Open Media", id="library-open-media")
                yield Button("Open Conversations", id="library-open-conversations")
                yield Button("Import/Export Sources", id="library-open-import-export")
                yield Button("Search/RAG", id="library-open-search")

    @on(Button.Pressed, "#library-open-notes")
    def open_notes(self) -> None:
        self.post_message(NavigateToScreen("notes"))

    @on(Button.Pressed, "#library-open-conversations")
    def open_conversations(self) -> None:
        self.post_message(NavigateToScreen("conversation"))
```

Repeat the pattern:

- Artifacts: title, purpose, `Open Chatbooks`, `Generated outputs coming from server/local output services`.
- Personas: title, purpose, `Open Personas`, routing to `ccp`; do not use Personas for Library conversation browsing.

- [ ] **Step 4: Verify routes remain wired**

Task 3A already mapped these primary shell routes. Verify `_resolve_screen_navigation_target()` still includes:

```python
"library": LibraryScreen,
"artifacts": ArtifactsScreen,
"personas": PersonasScreen,
```

Keep legacy routes `notes`, `media`, `ingest`, `search`, `conversation`, `chatbooks`, and `ccp` mapped to existing screens. `conversation` remains Library-owned saved conversation browsing; `ccp` remains Personas-owned behavior/profile management until the combined legacy screen is split.

- [ ] **Step 5: Run focused tests**

Run:

```bash
env HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/UI/test_destination_shells.py Tests/UI/test_screen_navigation.py --tb=short
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add tldw_chatbook/UI/Screens/library_screen.py tldw_chatbook/UI/Screens/artifacts_screen.py tldw_chatbook/UI/Screens/personas_screen.py tldw_chatbook/app.py Tests/UI/test_destination_shells.py
git commit -m "feat: add Library Artifacts Personas shells"
```

## Task 9: Add Watchlists+Collections, Schedules, And Workflows Wrappers

**Files:**
- Modify: `tldw_chatbook/UI/Screens/watchlists_collections_screen.py`
- Modify: `tldw_chatbook/UI/Screens/schedules_screen.py`
- Modify: `tldw_chatbook/UI/Screens/workflows_screen.py`
- Modify: `tldw_chatbook/app.py`
- Modify: `Tests/UI/test_destination_shells.py`
- Test: `Tests/UI/test_destination_shells.py`, `Tests/UI/test_subscription_window_watchlists.py`

- [ ] **Step 1: Add failing wrapper tests**

Extend `Tests/UI/test_destination_shells.py`:

```python
@pytest.mark.parametrize(
    ("route", "expected_sections"),
    [
        ("watchlists_collections", ["Watchlists", "Collections"]),
        ("schedules", ["Next Run", "Paused", "Failed"]),
        ("workflows", ["Recipes", "Dry Run", "Launch in Console"]),
    ],
)
@pytest.mark.asyncio
async def test_automation_destination_wrappers_explain_ownership(route, expected_sections):
    app = _build_test_app()

    async with app.run_test(size=(180, 40)) as pilot:
        await app.handle_screen_navigation(NavigateToScreen(route))
        await pilot.pause(0.1)

        visible_text = " ".join(str(widget.renderable) for widget in app.screen.query("Static") if hasattr(widget, "renderable"))
        for section in expected_sections:
            assert section in visible_text
```

Expected: FAIL.

- [ ] **Step 2: Run failing test**

Run:

```bash
env HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/UI/test_destination_shells.py::test_automation_destination_wrappers_explain_ownership --tb=short
```

Expected: FAIL.

- [ ] **Step 3: Implement Watchlists+Collections wrapper**

The wrapper must explicitly preserve server parity boundaries:

```python
yield Static("Watchlists", classes="destination-section")
yield Static("Monitored sources, filters, jobs, runs, outputs, templates, alerts, telemetry, retry/backoff.")
yield Static("Collections", classes="destination-section")
yield Static("Reading/content items, highlights, saved searches, archive state, note links, templates, feeds, import/export.")
yield Button("Open current Watchlists", id="wc-open-watchlists")
```

Button routes can initially open the existing `subscriptions` screen.

- [ ] **Step 4: Implement Schedules wrapper**

Render:

- purpose: "Schedules own when things run."
- sections: `Next Run`, `Paused`, `Failed`, `Retry`, `Open in Console`
- honest empty state if no scheduler data is available.

- [ ] **Step 5: Implement Workflows wrapper**

Render:

- purpose: "Workflows own what procedure runs."
- sections: `Recipes`, `Inputs`, `Steps`, `Dry Run`, `Approvals`, `Outputs`, `Launch in Console`
- honest empty state if no workflow service is wired yet.

- [ ] **Step 6: Verify routes remain wired**

Task 3A already mapped these routes. Verify `_resolve_screen_navigation_target()` still includes:

```python
"watchlists_collections": WatchlistsCollectionsScreen,
"schedules": SchedulesScreen,
"workflows": WorkflowsScreen,
```

- [ ] **Step 7: Run focused tests**

Run:

```bash
env HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/UI/test_destination_shells.py Tests/UI/test_subscription_window_watchlists.py --tb=short
```

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add tldw_chatbook/UI/Screens/watchlists_collections_screen.py tldw_chatbook/UI/Screens/schedules_screen.py tldw_chatbook/UI/Screens/workflows_screen.py tldw_chatbook/app.py Tests/UI/test_destination_shells.py
git commit -m "feat: add automation destination shells"
```

## Task 10: Add MCP, ACP, Skills, And Settings Wrappers

**Files:**
- Modify: `tldw_chatbook/UI/Screens/mcp_screen.py`
- Modify: `tldw_chatbook/UI/Screens/acp_screen.py`
- Modify: `tldw_chatbook/UI/Screens/skills_screen.py`
- Modify: `tldw_chatbook/UI/Screens/settings_screen.py`
- Modify: `tldw_chatbook/app.py`
- Modify: `Tests/UI/test_destination_shells.py`
- Test: `Tests/UI/test_destination_shells.py`, `Tests/UI/test_unified_mcp_panel.py`

- [ ] **Step 1: Add failing wrapper tests**

Extend `Tests/UI/test_destination_shells.py`:

```python
@pytest.mark.parametrize(
    ("route", "expected_text"),
    [
        ("mcp", "tools and servers"),
        ("acp", "Agent Client Protocol"),
        ("skills", "SKILL.md"),
        ("settings", "global preferences"),
    ],
)
@pytest.mark.asyncio
async def test_protocol_and_settings_wrappers_have_distinct_boundaries(route, expected_text):
    app = _build_test_app()

    async with app.run_test(size=(180, 40)) as pilot:
        await app.handle_screen_navigation(NavigateToScreen(route))
        await pilot.pause(0.1)

        visible_text = " ".join(str(widget.renderable) for widget in app.screen.query("Static") if hasattr(widget, "renderable"))
        assert expected_text in visible_text


@pytest.mark.asyncio
async def test_legacy_tools_settings_route_opens_mcp_not_global_settings():
    app = _build_test_app()

    async with app.run_test(size=(180, 40)) as pilot:
        await app.handle_screen_navigation(NavigateToScreen("tools_settings"))
        await pilot.pause(0.1)

        visible_text = " ".join(str(widget.renderable) for widget in app.screen.query("Static") if hasattr(widget, "renderable"))
        assert "tools and servers" in visible_text
        assert "global preferences" not in visible_text
```

Expected: FAIL.

- [ ] **Step 2: Run failing test**

Run:

```bash
env HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/UI/test_destination_shells.py::test_protocol_and_settings_wrappers_have_distinct_boundaries Tests/UI/test_destination_shells.py::test_legacy_tools_settings_route_opens_mcp_not_global_settings --tb=short
```

Expected: FAIL.

- [ ] **Step 3: Implement MCP wrapper**

MCP screen should mount or deep-link only MCP/tool-control UI by default, not the broad legacy `ToolsSettingsScreen`:

- purpose: "MCP owns tools and servers."
- sections: Servers, Tools, Permissions, Auth, Audit, Test Tool.
- primary action: `Open MCP Management`.
- route or mount existing `UnifiedMCPPanel` if safe.
- if reusing legacy code is unavoidable, deep-link to an MCP-only subview and hide or exclude broad General, Configuration, Appearance, or global preference panels.
- `NavigateToScreen("tools_settings")` must resolve to `MCPScreen` or an MCP-only route, not `ToolsSettingsScreen`.

- [ ] **Step 4: Implement ACP wrapper**

ACP may start as an honest capability state:

- purpose: "ACP owns Agent Client Protocol agents, sessions, runtimes, diffs, and terminals."
- sections: Installed agents, Sessions, Resume, Diffs, Terminal/Shell.
- if no ACP service exists, show: "ACP runtime is not configured yet. Install or configure an ACP-compatible agent before launch."
- primary action can be disabled with recovery text until real service exists.

Do not fake ACP functionality.

- [ ] **Step 5: Implement Skills wrapper**

Skills should use existing `local_skills_service` where possible:

- purpose: "Skills owns Agent Skills packs."
- sections: Installed, Discover/Import, Validate, Scripts, References, Assets, Attachments.
- show local skills directory: `get_user_data_dir() / "skills"` if available from app service.
- primary action: `Browse Skills` or `Import Skill`.

- [ ] **Step 6: Implement Settings wrapper**

Create or update `tldw_chatbook/UI/Screens/settings_screen.py` as a real wrapper. Do not reuse `ToolsSettingsScreen` as the Settings route. Settings must present global preferences, appearance, accounts/auth, storage, and app-level behavior. It may link to legacy `customize` and to explicitly settings-like subviews after they are separated, but it must not rely on `tools_settings` as the global settings destination because `tools_settings` is reserved for MCP/tool-control compatibility.

- [ ] **Step 7: Verify routes remain wired**

Task 3A already mapped these routes. Verify `_resolve_screen_navigation_target()` still includes:

```python
"mcp": MCPScreen,
"tools_settings": MCPScreen,
"acp": ACPScreen,
"skills": SkillsScreen,
"settings": SettingsScreen,
```

Keep `customize` as an appearance/customization legacy route. Keep `tools_settings` as a legacy alias into `MCPScreen`, not as a route to broad global settings.

- [ ] **Step 8: Run focused tests**

Run:

```bash
env HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/UI/test_destination_shells.py Tests/UI/test_unified_mcp_panel.py --tb=short
```

Expected: PASS.

- [ ] **Step 9: Commit**

```bash
git add tldw_chatbook/UI/Screens/mcp_screen.py tldw_chatbook/UI/Screens/acp_screen.py tldw_chatbook/UI/Screens/skills_screen.py tldw_chatbook/UI/Screens/settings_screen.py tldw_chatbook/app.py Tests/UI/test_destination_shells.py
git commit -m "feat: add protocol and skills destination shells"
```

## Task 11: Add Console Live-Work Follow And Staged Context Handoff Boundaries

**Files:**
- Modify: `tldw_chatbook/app.py`
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py`
- Modify: `tldw_chatbook/UI/Screens/library_screen.py`
- Modify: `tldw_chatbook/UI/Screens/artifacts_screen.py`
- Modify: `tldw_chatbook/UI/Screens/personas_screen.py`
- Modify: `tldw_chatbook/UI/Screens/watchlists_collections_screen.py`
- Modify: `tldw_chatbook/UI/Screens/schedules_screen.py`
- Modify: `tldw_chatbook/UI/Screens/workflows_screen.py`
- Modify: `tldw_chatbook/UI/Screens/acp_screen.py`
- Modify: `tldw_chatbook/UI/Screens/skills_screen.py`
- Create: `Tests/UI/test_console_live_work_handoffs.py`
- Test: `Tests/UI/test_console_live_work_handoffs.py`, `Tests/UI/test_chat_first_handoffs.py`

- [ ] **Step 1: Write failing tests for live work convergence**

Create `Tests/UI/test_console_live_work_handoffs.py`:

```python
from unittest.mock import Mock

import pytest

from Tests.UI.test_screen_navigation import _build_test_app
from tldw_chatbook.Chat.chat_handoff_models import ChatHandoffPayload
from tldw_chatbook.UI.Navigation.main_navigation import NavigateToScreen


def test_app_exposes_open_console_for_live_work_helper():
    app = _build_test_app()

    assert hasattr(app, "open_console_for_live_work")


@pytest.mark.asyncio
async def test_open_console_for_live_work_routes_to_chat_route():
    app = _build_test_app()
    seen = []

    async def fake_handler(message):
        seen.append(message.screen_name)

    app.post_message = lambda message: seen.append(getattr(message, "screen_name", None))

    app.open_console_for_live_work(source="workflows", title="Daily digest")

    assert seen == ["chat"]


@pytest.mark.parametrize(
    ("route", "button_id"),
    [
        ("watchlists_collections", "watchlists-follow-in-console"),
        ("schedules", "schedules-follow-in-console"),
        ("workflows", "workflows-launch-in-console"),
        ("acp", "acp-follow-in-console"),
    ],
)
@pytest.mark.asyncio
async def test_destination_live_work_actions_open_console(route, button_id):
    app = _build_test_app()

    async with app.run_test(size=(180, 40)) as pilot:
        await app.handle_screen_navigation(NavigateToScreen(route))
        await pilot.pause(0.1)
        await pilot.click(f"#{button_id}")
        await pilot.pause(0.1)

    assert app.current_tab == "chat"
    assert app.pending_console_launch["source"] == route


@pytest.mark.parametrize(
    ("route", "button_id"),
    [
        ("library", "library-use-in-console"),
        ("artifacts", "artifacts-use-in-console"),
        ("personas", "personas-attach-to-console"),
        ("skills", "skills-attach-to-console"),
    ],
)
@pytest.mark.asyncio
async def test_staged_context_actions_use_chat_handoff_not_live_launch(route, button_id):
    app = _build_test_app()
    app.open_chat_with_handoff = Mock()
    app.open_console_for_live_work = Mock()

    async with app.run_test(size=(180, 40)) as pilot:
        await app.handle_screen_navigation(NavigateToScreen(route))
        await pilot.pause(0.1)
        await pilot.click(f"#{button_id}")
        await pilot.pause(0.1)

    app.open_chat_with_handoff.assert_called_once()
    payload = app.open_chat_with_handoff.call_args.args[0]
    assert isinstance(payload, ChatHandoffPayload)
    app.open_console_for_live_work.assert_not_called()
    assert getattr(app, "pending_console_launch", None) in (None, {})


@pytest.mark.asyncio
async def test_console_renders_pending_launch_context():
    app = _build_test_app()
    app.pending_console_launch = {
        "source": "workflows",
        "title": "Daily digest",
        "payload": {},
    }

    async with app.run_test(size=(180, 40)) as pilot:
        await app.handle_screen_navigation(NavigateToScreen("chat"))
        await pilot.pause(0.1)

        assert app.query_one("#console-pending-launch-card")
```

Expected: FAIL because helper, wrapper buttons, and Console pending-launch rendering do not exist.

- [ ] **Step 2: Run failing tests**

Run:

```bash
env HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/UI/test_console_live_work_handoffs.py --tb=short
```

Expected: FAIL.

- [ ] **Step 3: Implement app helper**

In `tldw_chatbook/app.py`:

```python
def open_console_for_live_work(self, *, source: str, title: str, payload: dict | None = None) -> None:
    """Open Console for live work launched from another destination."""
    self.pending_console_launch = {
        "source": source,
        "title": title,
        "payload": payload or {},
    }
    self.post_message(NavigateToScreen("chat"))
```

Keep this separate from `open_chat_with_handoff()`. The existing `open_chat_with_handoff(ChatHandoffPayload)` seam remains the only path for staged source, artifact, persona, and skill context.

- [ ] **Step 4: Preserve staged context handoffs for source/artifact/persona/skill actions**

For actions named `Use in Console` or `Attach to Console` from Library, Artifacts, Personas, and Skills, build a `ChatHandoffPayload` and call the existing `open_chat_with_handoff(payload)` seam. Do not set `pending_console_launch` for these actions; they stage context for a chat session rather than follow a live run.

Use stable IDs:

- `library-use-in-console`
- `artifacts-use-in-console`
- `personas-attach-to-console`
- `skills-attach-to-console`

- [ ] **Step 5: Add wrapper buttons to use live helper only for live work**

In destination wrappers, use `open_console_for_live_work()` for actions named:

- Launch in Console
- Follow in Console
- Resume in Console
- Open live run in Console

Do not use it for static configuration, source staging, artifacts, personas, skills, or history views. Initial live-run wrappers using this helper are Watchlists+Collections, Schedules, Workflows, and ACP.

- [ ] **Step 6: Render pending launch context in Console**

In `tldw_chatbook/UI/Screens/chat_screen.py`, detect `app.pending_console_launch` during composition or mount and render a compact `#console-pending-launch-card` that names the source and title. The card should use `.ds-panel` and a readable source/status label. Do not start a live run automatically in this slice.

- [ ] **Step 7: Run focused tests**

Run:

```bash
env HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/UI/test_console_live_work_handoffs.py Tests/UI/test_chat_first_handoffs.py --tb=short
```

Expected: PASS.

- [ ] **Step 8: Run whitespace check**

Run:

```bash
git diff --check
```

Expected: no output.

- [ ] **Step 9: Commit**

```bash
git add tldw_chatbook/app.py tldw_chatbook/UI/Screens/chat_screen.py tldw_chatbook/UI/Screens/library_screen.py tldw_chatbook/UI/Screens/artifacts_screen.py tldw_chatbook/UI/Screens/personas_screen.py tldw_chatbook/UI/Screens/watchlists_collections_screen.py tldw_chatbook/UI/Screens/schedules_screen.py tldw_chatbook/UI/Screens/workflows_screen.py tldw_chatbook/UI/Screens/acp_screen.py tldw_chatbook/UI/Screens/skills_screen.py Tests/UI/test_console_live_work_handoffs.py
git commit -m "feat: add Console live work launch helper"
```

## Task 12: Update Documentation And Run UX Smoke Verification

**Files:**
- Modify: `Docs/Development/chat-first-shell-migration.md`
- Modify: `Docs/superpowers/plans/2026-05-01-ux-audit-remediation.md`
- Modify: `Docs/Design/master-shell-route-inventory.md`
- Modify: `Tests/UI/test_shell_destinations.py`
- Modify: `Tests/UI/test_screen_navigation.py`
- Modify: `Tests/UI/test_master_shell_design_system_contract.py` if closeout adjusts the design-system status contract
- Modify: `Tests/UI/test_command_palette_providers.py`
- Modify: `Tests/UI/test_command_palette_basic.py`
- Modify: `Tests/UI/test_chat_first_handoffs.py`
- Modify: `Tests/UI/test_console_live_work_handoffs.py`
- Modify: `Tests/UI/test_command_palette_shell_routes.py`
- Test: `Tests/UI/test_ux_audit_smoke.py`, focused shell tests

- [ ] **Step 1: Update migration docs**

In `Docs/Development/chat-first-shell-migration.md`, add:

```markdown
## Master Shell Update

The user-facing `Chat` destination is now `Console`. The internal `chat` route remains stable during migration. Live agent work should open or follow in Console; Library, Artifacts, Personas, Watchlists+Collections, Schedules, Workflows, MCP, ACP, Skills, and Settings prepare, configure, inspect, or preserve work.
```

- [ ] **Step 2: Update UX remediation plan status**

In `Docs/superpowers/plans/2026-05-01-ux-audit-remediation.md`, add a short note that the tooltip/remediation stream led to the master shell UX spec and this implementation plan. Do not rewrite the old remediation plan.

- [ ] **Step 3: Update route inventory with final implementation state**

Ensure `Docs/Design/master-shell-route-inventory.md` includes final routes and remaining deferred surfaces. Ensure `Docs/Design/master-shell-design-system-contract.md` reflects any class, state, density, or testing-hook adjustments made during implementation.

- [ ] **Step 4: Add final mounted-route and status-language inventory test**

Add a final test in `Tests/UI/test_shell_destinations.py` or `Tests/UI/test_screen_navigation.py`:

```python
def test_every_shell_destination_has_readable_purpose_and_mounted_route():
    from tldw_chatbook.UI.Navigation.shell_destinations import SHELL_DESTINATION_ORDER

    app = _build_test_app()
    for destination in SHELL_DESTINATION_ORDER:
        assert destination.purpose
        assert destination.tooltip
        _screen_name, _tab_id, screen_class = app._resolve_screen_navigation_target(destination.primary_route)
        assert screen_class is not None, destination.primary_route
```

Also keep `Tests/UI/test_master_shell_design_system_contract.py::test_status_contract_requires_readable_labels` in the final focused suite so status remains text-labeled rather than color-only.

- [ ] **Step 5: Run focused shell test suite**

Run:

```bash
env HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/UI/test_master_shell_design_system_contract.py Tests/UI/test_shell_destinations.py Tests/Home/test_dashboard_state.py Tests/UI/test_home_screen.py Tests/UI/test_master_shell_navigation.py Tests/UI/test_destination_shells.py Tests/UI/test_screen_navigation.py Tests/UI/test_navigation_label_language.py Tests/UI/test_command_palette_shell_routes.py Tests/UI/test_command_palette_providers.py Tests/UI/test_command_palette_basic.py Tests/UI/test_console_live_work_handoffs.py Tests/UI/test_chat_first_handoffs.py Tests/UI/test_ux_audit_smoke.py --tb=short
```

Expected: PASS or document any environment-only SQLite access failures with exact traceback.

- [ ] **Step 6: Run whitespace check**

Run:

```bash
git diff --check
```

Expected: no output.

- [ ] **Step 7: Commit docs and smoke closeout**

```bash
git add Docs/Development/chat-first-shell-migration.md Docs/superpowers/plans/2026-05-01-ux-audit-remediation.md Docs/Design/master-shell-route-inventory.md Docs/Design/master-shell-design-system-contract.md Tests/UI/test_shell_destinations.py Tests/UI/test_screen_navigation.py Tests/UI/test_master_shell_design_system_contract.py Tests/UI/test_command_palette_shell_routes.py Tests/UI/test_command_palette_providers.py Tests/UI/test_command_palette_basic.py Tests/UI/test_console_live_work_handoffs.py Tests/UI/test_chat_first_handoffs.py
git commit -m "docs: close out master shell UX migration plan"
```

## PR Slicing Recommendation

Do not ship this as one large PR. Use these slices:

1. Design-system implementation preflight, contract, route inventory, and shell destination model.
2. Home route/startup and minimal route stubs for every primary destination.
3. Master shell navigation once every visible destination resolves.
4. Home dashboard state and UI.
5. Console user-facing rename.
6. Library, Artifacts, Personas wrapper enrichment.
7. Watchlists+Collections, Schedules, Workflows wrapper enrichment.
8. MCP, ACP, Skills, Settings wrapper enrichment.
9. Console live-work launch/follow helper.
10. Documentation and UX smoke closeout.

Each slice should:

- add failing tests first
- implement the smallest safe behavior
- run focused tests
- run `git diff --check`
- commit before moving to the next slice

## Execution Notes

- Use a fresh worktree per PR slice if the main branch is dirty or behind `origin/dev`.
- Prefer preserving old route IDs and adding user-facing labels over renaming internal modules.
- Do not do broad CSS/theming rewrites in this master-shell plan. Consume the agentic terminal design-system tokens/classes once they land; if they are absent, stop and merge/rebase the design-system work first.
- When adding wrappers, start with honest empty states and links to existing screens. Do not fake unavailable ACP or workflow runtime behavior.
- Preserve power-user shortcuts. If a shortcut target changes, add a compatibility command or clear deprecation path.
- Keep `Use in Chat` copy migration separate unless a task explicitly changes it to `Use in Console`; this avoids mixing broad copy changes with route/shell behavior.
