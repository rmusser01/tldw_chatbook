# Tools Hub in Settings — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Retire the MCP shell destination and host its four-mode workbench (Servers / Tools / Permissions / Audit) as one "Tools & MCP" category inside the Settings screen.

**Architecture:** The existing `MCPWorkbench` (`UI/MCP_Modules/mcp_workbench.py`, ~6.9k lines) is re-hosted, not rewritten. A new pane module `UI/Screens/settings_tools_hub.py` ports the 347-line `mcp_screen.py` shell's responsibilities (mode chips, footer hints, reload worker, deep-link context) into a Settings category pane. Destination retirement follows the ADR-172 template exactly: compatibility hotkey, route aliases, palette aliases, default nav context.

**Tech Stack:** Python ≥3.12, Textual 8.x (≥8.0.0,<9), pytest.

**Spec:** `Docs/superpowers/specs/2026-10-15-tools-hub-in-settings-design.md` (approved 2026-10-16 — read it before starting; the plan argues from it).

## Global Constraints

- The workbench internals are frozen: no edits inside `UI/MCP_Modules/*`, the permission store, or ToolPackService.
- Keep existing DOM ids `#mcp-mode-strip`, `mcp-mode-servers/tools/permissions/audit` chip ids, and `#mcp-hub-workbench` so the app-bundle CSS (`css/features/_agentic_terminal.tcss` `#mcp-mode-strip` rules) and ported tests keep working.
- No new hex literals or ad-hoc `$ds-*` tokens (design-token governance, ADR-150; `Tests/UI/test_design_token_governance.py` fails otherwise). The ported chip CSS uses `$surface`/`$text`/`$text-muted` exactly as `mcp_screen.py` does today.
- Keybinding conventions (gotcha #11): never bind terminal-convention keys; screen actions use single-letter htop-style bindings; footer hints only advertise implemented actions.
- Testing: targeted runs only unless the user opts into a full sweep (AGENTS.md). No `timeout` command in this environment.
- Backlog hygiene: the task file gets an Implementation Plan before code and Implementation Notes after; status moves via `backlog task edit <id> -s …`.
- Commit style: `feat(ui): …` / `test(ui): …` with the backlog task id in parens, matching recent history.

---

### Task 1: Governance first — backlog task + ADR-237

**Files:**
- Create: `backlog/decisions/237-mcp-destination-retirement-tools-hub-in-settings.md`
- Create: `backlog/tasks/task-<id> - Retire-MCP-destination-into-Settings-Tools-and-MCP-hub.md` (created via `backlog task create`, not by hand)

**Interfaces:**
- Consumes: the approved spec.
- Produces: ADR path + backlog task id referenced by every later commit message.

- [ ] **Step 1: Create the backlog task**

```bash
cd /Users/macbook-dev/Documents/GitHub/tldw_chatbook
backlog task create "Retire MCP destination into Settings Tools-and-MCP hub" \
  -d "The MCP destination is protocol jargon and a whole top-level destination for tool config. Fold all four workbench modes into one Settings category per the approved spec." \
  --ac "Settings has a Tools & MCP category hosting the four-mode MCPWorkbench,mcp/tools_settings routes alias to Settings with category context,ctrl+9 preserved as compatibility deep-link,mcp-defaults read-only category deleted,Tool Profiles handoff lands on the pane in Permissions mode,targeted tests green" \
  -s "In Progress"
backlog task list --plain | head -5   # note the assigned id
```

- [ ] **Step 2: Add the implementation plan to the task**

```bash
backlog task edit <id> --plan "1. Pane module + TOOLS category registration (destination still live)\n2. Keybindings a/r/t/1-4 + per-mode footer hints\n3. Deep links + Tool Profiles handoff\n4. Destination retirement + route aliases + compat hotkey\n5. Test migration sweep\n6. Docs + verification"
```

- [ ] **Step 3: Write ADR-237**

Create `backlog/decisions/237-mcp-destination-retirement-tools-hub-in-settings.md` following the house ADR shape (see `backlog/decisions/172-library-artifacts-browse-and-navigation.md`):

```markdown
# ADR-237: Retire the MCP destination; Settings hosts the Tools & MCP hub

Status: Accepted
Date: 2026-10-16
Related Task: task-<id> (fill in from Step 1)
Spec: Docs/superpowers/specs/2026-10-15-tools-hub-in-settings-design.md
Amends: ADR-015 (destination set 14 -> 13; fold map gains mcp/tools_settings -> Settings)
Supersedes: the `mcp-defaults` Settings domain contract's "server operations stay
in MCP" stance (the contract object is deleted with the category)
Related: ADR-172 (retirement template reused), ADR-148 (workbench IA preserved
unchanged), ADR-031/ADR-152 (shortcut lockstep for removing ctrl+9)

## Decision

Retire the MCP shell destination. Settings gains one "Tools & MCP" category
(`SettingsCategoryId.TOOLS = "tools"`) that hosts the existing four-mode
`MCPWorkbench` intact inside a new pane module
(`UI/Screens/settings_tools_hub.py`); `UI/Screens/mcp_screen.py` is deleted
after its shell logic (mode chips, per-mode footer hints, exclusive reload
worker, bounded deep-link context) is ported to the pane.

Route and hotkey compatibility follow the ADR-172 template: `mcp` and
`tools_settings` resolve to Settings (screen-registry alias + shell route
map), carry default context `{"category": "tools"}`, and ctrl+9 survives as
`MCP_COMPATIBILITY_SHORTCUT` deep-linking Settings -> Tools & MCP. The
Settings palette command gains "tools"/"mcp" alias terms.

Settings keybindings follow the screen's own per-category dispatch
precedent: `1-4` (mode switch) are new self-guarded bindings; `a`/`r`/`t`
gain a TOOLS branch (add server / force reload / test tool) while every
other category keeps today's behavior.

## Context

The destination is protocol jargon behind a whole top-level slot; its own
copy says most people never need it. Users looking for tools in Settings
found only Tool Profiles (lifecycle) and a read-only MCP Defaults pointer.
The prior in-code contract warned against "flattening tools into Settings" —
that objection was about flattening tool config into form categories; this
decision re-homes the workbench intact and deletes the pointer card instead.

## Alternatives Considered

| Option | Why rejected |
| --- | --- |
| Config/operations split (Settings takes permissions only) | Splits "where do I manage tools?" across two surfaces and dissects the workbench |
| Hub-and-pointer (link better, change nothing) | That is the status quo; judged not easy enough |
| Keep the destination, add a Settings alias | Two homes for one surface; the jargon destination remains |

## Consequences

- `SHELL_DESTINATION_ORDER` drops to 13 (14 pre-retirement); ctrl+9 leaves
  `SHELL_DESTINATION_SHORTCUTS` in lockstep (ADR-031/152).
- ADR-148's rail IA, recovery review, and audit drill-down are unchanged;
  only the host screen changes.
- The Tool Profiles "edit policy" handoff becomes an internal Settings
  category switch with the same bounded context.
- Startup configs with `default_tab = "mcp"` still land on Settings -> Tools
  & MCP via the alias + default context.

## Links

- Spec: Docs/superpowers/specs/2026-10-15-tools-hub-in-settings-design.md
```

- [ ] **Step 4: Link the ADR and spec from the task, then commit**

```bash
backlog task edit <id> --notes "ADR-237 written; spec linked"
git add backlog/decisions/237-mcp-destination-retirement-tools-hub-in-settings.md \
  "backlog/tasks/task-<id> - Retire-MCP-destination-into-Settings-Tools-and-MCP-hub.md" \
  Docs/superpowers/specs/2026-10-15-tools-hub-in-settings-design.md
git commit -m "docs: ADR-237 + spec for Tools hub in Settings (task-<id>)"
```

---

### Task 2: TOOLS category id + ToolsHubPanel pane + Settings registration

The destination stays live this task; both surfaces coexist so the category lands first.

**Files:**
- Create: `tldw_chatbook/UI/Screens/settings_tools_hub.py`
- Modify: `tldw_chatbook/UI/Screens/settings_config_models.py` (`SettingsCategoryId`)
- Modify: `tldw_chatbook/UI/Screens/settings_screen.py` (summary entry, domain-contract deletion, render branch, ownership record, save/restore state)
- Test: `Tests/UI/test_settings_tools_hub.py` (new)

**Interfaces:**
- Consumes: `MCPWorkbench(app_instance, **kwargs)`, `MCPWorkbench.set_initial_view_state(dict | None)`, `.get_view_state() -> dict`, `.set_mode(str)`, `.active_mode -> str`, `.reload(force: bool) -> Awaitable`, `.open_add_server_form() -> Awaitable`, `.open_test_for_selected_tool() -> Awaitable`, `MCPWorkbench.ModeChanged` (message, `.mode` attr), `MCP_HUB_MODES: dict[str, dict[str, str]]` (keys `label`, `button_id`) — all from `tldw_chatbook/UI/MCP_Modules/mcp_workbench.py`; `DestinationModeStrip` from `tldw_chatbook/Widgets/destination_workbench.py`.
- Produces: `ToolsHubPanel` (constructor `ToolsHubPanel(app_instance, initial_view_state: dict | None = None, **kwargs)`), with `activate_mode(mode: str) -> None`, `add_server() -> None`, `refresh_now() -> None`, `test_selected_tool() -> None`, `get_view_state() -> dict[str, object] | None`, `active_mode -> str`, `handle_runtime_backend_change(runtime_backend: str) -> None`, and message `ToolsHubPanel.ModeChanged(mode: str)`; `TOOLS_HUB_MODE_SHORTCUTS: dict[str, tuple[tuple[str, str], ...]]`; `SettingsCategoryId.TOOLS`. Later tasks call these names.

- [ ] **Step 1: Write the failing test**

Create `Tests/UI/test_settings_tools_hub.py`. Model the harness on `Tests/UI/test_destination_shells.py` (see its `DestinationHarness` and how the mcp cases mount `MCPScreen` without full services — `MCPWorkbench` mounts fine against a stub app).

```python
"""Settings hosts the MCP workbench as the Tools & MCP category."""

import pytest

from Tests.UI.test_destination_shells import (
    ConsolidatedCSSApp,
    _active_destination_screen,
    _open_settings_category,
)
from tldw_chatbook.UI.MCP_Modules.mcp_workbench import MCPWorkbench
from tldw_chatbook.UI.Screens.settings_config_models import SettingsCategoryId
from tldw_chatbook.UI.Screens.settings_tools_hub import ToolsHubPanel


class _PanelHarnessApp(ConsolidatedCSSApp):
    """Mounts one ToolsHubPanel the way DestinationHarness mounts screens."""

    def __init__(self, initial_view_state=None):
        super().__init__()
        self.app_instance = type("AppInstanceStub", (), {"unified_mcp_service": None})()
        self._initial_view_state = initial_view_state

    def compose(self):
        yield ToolsHubPanel(
            self.app_instance,
            initial_view_state=self._initial_view_state,
            id="settings-tools-hub",
        )


def test_tools_category_id_exists_and_mcp_defaults_retired():
    assert SettingsCategoryId.TOOLS.value == "tools"
    with pytest.raises(AttributeError):
        _ = SettingsCategoryId.MCP_DEFAULTS


async def test_panel_mounts_workbench_and_mode_chips():
    host = _PanelHarnessApp()
    async with host.run_test() as pilot:
        panel = host.query_one("#settings-tools-hub", ToolsHubPanel)
        workbench = panel.query_one(MCPWorkbench)
        assert workbench.active_mode == "servers"
        chips = panel.query("#mcp-mode-strip Button")
        assert [c.id for c in chips] == [
            "mcp-mode-servers",
            "mcp-mode-tools",
            "mcp-mode-permissions",
            "mcp-mode-audit",
        ]
        await pilot.click("#mcp-mode-permissions")
        assert workbench.active_mode == "permissions"
        assert "is-active" in panel.query_one("#mcp-mode-permissions").classes


async def test_panel_view_state_round_trip():
    host = _PanelHarnessApp(initial_view_state={"mode": "audit"})
    async with host.run_test():
        panel = host.query_one("#settings-tools-hub", ToolsHubPanel)
        workbench = panel.query_one(MCPWorkbench)
        assert workbench.active_mode == "audit"
        state = panel.get_view_state()
        assert isinstance(state, dict)
```

(If `ConsolidatedCSSApp`, `_active_destination_screen`, or `_open_settings_category` are not importable from `test_destination_shells.py`, import them from wherever `Tests/UI/test_settings_tool_profiles.py` gets them — same helpers, one canonical home.)

- [ ] **Step 2: Run it to verify it fails**

```bash
python -m pytest Tests/UI/test_settings_tools_hub.py -x -q
```

Expected: FAIL — `ModuleNotFoundError: No module named 'tldw_chatbook.UI.Screens.settings_tools_hub'` (and the `SettingsCategoryId.TOOLS` assertion after the import is added).

- [ ] **Step 3: Add the enum value and write the pane module**

In `settings_config_models.py`, add `TOOLS = "tools"` to `SettingsCategoryId` (after `TOOL_PROFILES`) and delete the `MCP_DEFAULTS = "mcp-defaults"` line.

Create `tldw_chatbook/UI/Screens/settings_tools_hub.py` — a port of `UI/Screens/mcp_screen.py` with the screen-specific parts (BINDINGS, footer registration, `on_screen_resume/suspend`, `apply_navigation_context`, `save_state`/`restore_state` on the screen) replaced by pane equivalents:

```python
"""Settings Tools & MCP category pane: mode strip + the MCP workbench.

Port of the retired MCP destination shell (`UI/Screens/mcp_screen.py`): the
same mode chips, per-mode shortcut sets, exclusive reload worker group, and
bounded initial view state -- hosted inside the Settings screen instead of a
standalone destination. DOM ids (#mcp-mode-strip, mcp-mode-* chips,
#mcp-hub-workbench) are kept so the app-bundle CSS and ported tests are
unchanged.
"""

from typing import Any

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.message import Message
from textual.widgets import Button, Static

from ...Widgets.destination_workbench import DestinationModeStrip
from ..MCP_Modules.mcp_workbench import MCP_HUB_MODES, MCPWorkbench

_MODE_BY_BUTTON_ID = {spec["button_id"]: mode for mode, spec in MCP_HUB_MODES.items()}

_MODE_TOOLTIPS = {
    "servers": "Servers mode: view MCP servers and their readiness.",
    "tools": "Tools mode: browse and test scoped MCP tools.",
    "permissions": (
        "Permissions mode: set Allow / Ask / Off per tool. Space cycles the selected row."
    ),
    "audit": "Audit mode: review MCP tool execution history and drill into a call's detail.",
}

# F-055: the hint set is PER-MODE -- a key is only advertised where it
# actually works. Mirrors the retired screen's MCP_MODE_SHORTCUTS.
_COMMON_SHORTCUTS: tuple[tuple[str, str], ...] = (
    ("1-4", "mode"),
    ("a", "add server"),
    ("r", "refresh"),
)
TOOLS_HUB_MODE_SHORTCUTS: dict[str, tuple[tuple[str, str], ...]] = {
    "servers": _COMMON_SHORTCUTS,
    "tools": _COMMON_SHORTCUTS + (("t", "test tool"),),
    "permissions": _COMMON_SHORTCUTS + (("space", "cycle permission"),),
    "audit": _COMMON_SHORTCUTS,
}

# Shared reload-worker identity so a manual refresh, a runtime-triggered
# reload, and repeats of either serialize against each other.
_RELOAD_WORKER_GROUP = "settings-tools-hub-reload"


class ToolsHubPanel(Vertical):
    """Hosts the MCP workbench for the Settings Tools & MCP category."""

    DEFAULT_CSS = """
    Button.mcp-mode-chip {
        width: auto;
        /* min-width kept in lockstep with the higher-specificity app-bundle
        rule (#mcp-mode-strip Button.mcp-mode-chip, _agentic_terminal.tcss);
        divergence here would silently lose to the bundle. */
        min-width: 10;
        height: 1;
        min-height: 1;
        padding: 0 1;
        border: none;
    }
    .mcp-mode-chip.is-active {
        border: none;
        text-style: bold underline;
    }
    /* A1: focus must not impersonate the active-mode indicator (bold
    underline). Uses the raw $surface/$text tokens deliberately -- the
    $ds-focus-* aliases are only defined once the app-wide tcss bundle is
    loaded, and harness-mounted tests set no CSS_PATH (same reasoning as the
    retired mcp_screen.py copy). */
    .mcp-mode-chip:focus,
    .mcp-mode-chip:hover:focus {
        background: $surface;
        color: $text;
        text-style: bold;
        outline: none;
    }
    .mcp-mode-chip.is-active:focus,
    .mcp-mode-chip.is-active:hover:focus {
        background: $surface;
        color: $text-muted;
        text-style: bold underline;
        outline: none;
    }
    """

    class ModeChanged(Message):
        """The workbench switched mode; Settings re-renders footer hints."""

        def __init__(self, mode: str) -> None:
            super().__init__()
            self.mode = mode

    def __init__(
        self,
        app_instance: Any,
        initial_view_state: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._app_instance = app_instance
        self._initial_view_state = initial_view_state
        self.workbench: MCPWorkbench | None = None

    def compose(self) -> ComposeResult:
        yield Static(
            "MCP (Model Context Protocol) lets chatbook use external "
            "tools — most people never need to change anything here.",
            id="tools-hub-purpose",
            classes="destination-purpose",
        )
        with DestinationModeStrip(
            id="mcp-mode-strip", classes="destination-mode-strip"
        ):
            for mode, spec in MCP_HUB_MODES.items():
                chip = Button(
                    spec["label"],
                    id=spec["button_id"],
                    classes="mcp-mode-chip console-action-subdued",
                    compact=True,
                    tooltip=_MODE_TOOLTIPS.get(mode, spec["label"]),
                )
                chip.set_class(mode == "servers", "is-active")
                yield chip
        self.workbench = MCPWorkbench(self._app_instance, id="mcp-hub-workbench")
        self.workbench.set_initial_view_state(self._initial_view_state)
        yield self.workbench

    @property
    def active_mode(self) -> str:
        return self.workbench.active_mode if self.workbench else "servers"

    def get_view_state(self) -> dict[str, Any] | None:
        return self.workbench.get_view_state() if self.workbench else None

    def set_initial_view_state(self, state: dict[str, Any] | None) -> None:
        self._initial_view_state = state
        if self.workbench is not None:
            self.workbench.set_initial_view_state(state)

    def _sync_mode_chips(self, active_mode: str) -> None:
        for candidate, spec in MCP_HUB_MODES.items():
            chips = list(self.query(f"#{spec['button_id']}"))
            if chips:
                chips[0].set_class(candidate == active_mode, "is-active")

    def activate_mode(self, mode: str) -> None:
        if self.workbench is None:
            return
        self.workbench.set_mode(mode)
        self._sync_mode_chips(self.workbench.active_mode)

    def add_server(self) -> None:
        """Switch to Servers mode and open the Add-server form (worker-mounted)."""
        if self.workbench is None:
            return
        self.activate_mode("servers")
        self.run_worker(
            self.workbench.open_add_server_form(),
            name="settings-tools-hub-add-server",
            group="settings-tools-hub-add-server",
            exclusive=True,
        )

    def refresh_now(self) -> None:
        """Force-reload the workbench (bypasses the revisit cache, TASK-34965)."""
        if self.workbench is None:
            return
        self.run_worker(
            self.workbench.reload(force=True),
            name="settings-tools-hub-manual-refresh",
            group=_RELOAD_WORKER_GROUP,
            exclusive=True,
        )

    def test_selected_tool(self) -> None:
        """Open the Test Tool panel for the inspector's selected tool."""
        if self.workbench is None:
            return
        self.run_worker(
            self.workbench.open_test_for_selected_tool(),
            name="settings-tools-hub-test-tool",
            group="settings-tools-hub-test-tool",
            exclusive=True,
        )

    def handle_runtime_backend_change(self, runtime_backend: str) -> None:
        """A backend change must fully reload (force, TASK-34965)."""
        _ = runtime_backend
        if self.workbench:
            self.run_worker(
                self.workbench.reload(force=True),
                name="settings-tools-hub-runtime-refresh",
                group=_RELOAD_WORKER_GROUP,
                exclusive=True,
            )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        mode = _MODE_BY_BUTTON_ID.get(event.button.id or "")
        if mode is None:
            return
        event.stop()
        self.activate_mode(mode)

    def on_mcp_workbench_mode_changed(self, event: MCPWorkbench.ModeChanged) -> None:
        event.stop()
        self._sync_mode_chips(event.mode)
        # F-055: Settings re-registers per-mode footer hints off this message.
        self.post_message(self.ModeChanged(event.mode))
```

- [ ] **Step 4: Run the pane tests to verify they pass**

```bash
python -m pytest Tests/UI/test_settings_tools_hub.py -x -q
```

Expected: 3 PASS. Helper imports: `test_settings_tool_profiles.py` shows where `_build_test_app`, `DestinationHarness`, `_open_settings_category`, and `_active_destination_screen` live — copy that import block if the names resolve from a different module than `test_destination_shells.py`.

- [ ] **Step 5: Register the category in SettingsScreen**

In `tldw_chatbook/UI/Screens/settings_screen.py`:

1. Import the pane next to the other sibling-module imports (~line 349 area):

```python
from .settings_tools_hub import TOOLS_HUB_MODE_SHORTCUTS, ToolsHubPanel
```

2. In `_category_summaries()` (~line 5435), insert directly after the `TOOL_PROFILES` entry:

```python
            SettingsCategorySummary(
                SettingsCategoryId.TOOLS,
                "Tools & MCP",
                "Agent tools, MCP servers, permissions, and audit.",
                "Immediate actions",
            ),
```

(`"Immediate actions"` mirrors the Workspaces status verb at ~line 5497.)

3. Delete the `MCP_DEFAULTS` `SettingsCategorySummary` entry (~line 5584) and the `MCP_DEFAULTS` `SettingsDomainCategoryContract` in `SETTINGS_DOMAIN_CATEGORY_CONTRACTS` (~line 2129).

4. `grep -n "MCP_DEFAULTS" tldw_chatbook/UI/Screens/settings_screen.py` and clean every remaining hit — the two category listings (~5676 and ~6562 regions pre-edit) are membership tuples (guided/testable/domain lists) that must drop the entry, not keep a dead name. After this step the grep must return nothing.

5. In the category-pane compose chain, next to the `TOOL_PROFILES` branch (~line 23753):

```python
        elif category is SettingsCategoryId.TOOLS:
            yield ToolsHubPanel(
                self.app_instance,
                initial_view_state=self._tools_hub_initial_view_state,
                id="settings-tools-hub",
            )
```

6. Add an ownership record beside the `TOOL_PROFILES` record (~line 6961 — find `SettingsOwnershipRecord(\n                category=SettingsCategoryId.TOOL_PROFILES,`):

```python
            SettingsOwnershipRecord(
                category=SettingsCategoryId.TOOLS,
                owns_config_sections=(
                    "MCP permission-store Tool policy",
                    "[mcp] enable/expose controls",
                ),
                reads_runtime_state_from=(
                    "Unified MCP panel",
                    "MCP configured server target store",
                ),
                writes_allowed=True,
                runtime_owner="MCP workbench (immediate reviewed actions)",
                boundary_copy=(
                    "Tools & MCP owns servers, tools, permissions, and audit "
                    "in place; Tool Profiles next door owns profile lifecycle."
                ),
                recovery_copy=(
                    "Re-open the category or press r to force a workbench reload."
                ),
            ),
```

7. Pane lookup + view-state plumbing on `SettingsScreen` (add near the other pane helpers; used again by Tasks 3-5):

```python
    def _tools_hub_pane(self) -> ToolsHubPanel | None:
        try:
            return self.query_one("#settings-tools-hub", ToolsHubPanel)
        except QueryError:
            return None
```

In `save_state` (~line 4346), after the `active_category` line:

```python
        tools_pane = self._tools_hub_pane()
        if tools_pane is not None:
            state["mcp_hub_view_state"] = tools_pane.get_view_state()
```

In `restore_state` (~line 4403), after the `active_category` restore:

```python
        mcp_view_state = state.get("mcp_hub_view_state")
        if isinstance(mcp_view_state, dict):
            self._tools_hub_initial_view_state = mcp_view_state
        tools_pane = self._tools_hub_pane()
        if tools_pane is not None:
            tools_pane.set_initial_view_state(
                self._tools_hub_initial_view_state
            )
```

And initialize the attribute in `SettingsScreen.__init__` (~line 3941):

```python
        self._tools_hub_initial_view_state: dict[str, object] | None = None
```

- [ ] **Step 6: Write the Settings-level category test, run everything**

Add to `Tests/UI/test_settings_tools_hub.py`, mirroring `test_canonical_settings_category_loads_app_owned_listing_off_thread` in `Tests/UI/test_settings_tool_profiles.py` (same `_build_test_app` + `DestinationHarness(app, "settings")` + `_open_settings_category` shape):

```python
@pytest.mark.asyncio
async def test_settings_registers_tools_category_and_mounts_pane(request):
    app = _build_test_app()
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(120, 35)) as pilot:
        settings = _active_destination_screen(host)
        summaries = {s.category: s.title for s in settings._category_summaries()}
        assert summaries.get(SettingsCategoryId.TOOLS) == "Tools & MCP"
        assert SettingsCategoryId.MCP_DEFAULTS not in summaries
        await _open_settings_category(pilot, "#settings-category-tools")
        await pilot.pause()

        assert settings.active_category == SettingsCategoryId.TOOLS.value
        assert settings.query_one("#settings-tools-hub", ToolsHubPanel)
        assert settings.query_one(MCPWorkbench)
```

(The sidebar category button id is `#settings-category-{category.value}` → `#settings-category-tools`; the `@pytest.mark.asyncio` + `request` fixture markers copy the tool-profiles file's conventions.)

```bash
python -m pytest Tests/UI/test_settings_tools_hub.py Tests/UI/test_settings_tool_profiles.py -q
```

Expected: PASS (tool-profiles run guards against registry regressions).

- [ ] **Step 7: Commit**

```bash
git add tldw_chatbook/UI/Screens/settings_tools_hub.py \
  tldw_chatbook/UI/Screens/settings_config_models.py \
  tldw_chatbook/UI/Screens/settings_screen.py \
  Tests/UI/test_settings_tools_hub.py
git commit -m "feat(ui): Tools & MCP category hosts the MCP workbench in Settings (task-<id>)"
```

---

### Task 3: Keybindings + per-mode footer hints + F1 help

**Files:**
- Modify: `tldw_chatbook/UI/Screens/settings_screen.py` (BINDINGS, three actions, `_category_footer_shortcuts`, footer registration on category switch and mode change)
- Test: `Tests/UI/test_settings_tools_hub.py`

**Interfaces:**
- Consumes: `ToolsHubPanel.activate_mode/add_server/refresh_now/test_selected_tool/active_mode`, `ToolsHubPanel.ModeChanged`, `TOOLS_HUB_MODE_SHORTCUTS` (Task 2), `BaseAppScreen.register_footer_shortcuts(source=..., shortcuts=...)` / `clear_footer_shortcuts(source=...)`.
- Produces: `SettingsScreen.action_settings_tools_mode(mode: str) -> None`; footer-hint source id `"settings-tools"` (Task 5's retired-screen port references it).

- [ ] **Step 1: Write the failing tests**

Append to `Tests/UI/test_settings_tools_hub.py`:

```python
async def test_mode_keys_and_verb_keys_dispatch_only_in_tools_category():
    async with app.run_test() as pilot:
        settings = app.screen
        # Outside TOOLS: the a/r/t actions keep their existing no-op/toast
        # behavior and never touch a pane (none mounted).
        settings.action_settings_test_category()  # must not raise
        settings._select_category(SettingsCategoryId.TOOLS.value)
        await pilot.pause()
        pane = settings.query_one("#settings-tools-hub", ToolsHubPanel)
        called: list[str] = []
        pane.activate_mode = lambda mode: called.append(mode)  # type: ignore[assignment]
        pane.test_selected_tool = lambda: called.append("test")  # type: ignore[assignment]
        settings.action_settings_tools_mode("audit")
        assert called == ["audit"]
        settings.action_settings_test_category()
        assert called == ["audit", "test"]


async def test_footer_shortcuts_follow_active_mode():
    async with app.run_test() as pilot:
        settings = app.screen
        settings._select_category(SettingsCategoryId.TOOLS.value)
        await pilot.pause()
        base = settings._category_footer_shortcuts(SettingsCategoryId.TOOLS)
        assert ("1-4", "mode") in base
        settings.query_one("#settings-tools-hub", ToolsHubPanel).activate_mode("tools")
        await pilot.pause()
        in_tools_mode = settings._category_footer_shortcuts(SettingsCategoryId.TOOLS)
        assert ("t", "test tool") in in_tools_mode
```

- [ ] **Step 2: Run to verify failure**

```bash
python -m pytest Tests/UI/test_settings_tools_hub.py -q
```

Expected: FAIL — `SettingsScreen` has no `action_settings_tools_mode`, and `_category_footer_shortcuts(TOOLS)` returns `()`.

- [ ] **Step 3: Implement**

In `settings_screen.py`:

1. BINDINGS (~line 3656) — append after the personal-context entries (mode keys are new; `a`/`r`/`t` already bound):

```python
        # Tools & MCP (ADR-237): mode keys self-guard to the TOOLS category,
        # mirroring the RAG accelerator self-guard pattern above.
        ("1", "settings_tools_mode('servers')", "Tools: Servers"),
        ("2", "settings_tools_mode('tools')", "Tools: Tools"),
        ("3", "settings_tools_mode('permissions')", "Tools: Permissions"),
        ("4", "settings_tools_mode('audit')", "Tools: Audit"),
```

(Textual accepts action-call strings in the tuple form — the retired `MCPScreen` used `Binding("1", "mcp_mode('servers')", ...)`.)

2. New action (next to `action_settings_rag_set_active`, ~line 35171):

```python
    def action_settings_tools_mode(self, mode: str) -> None:
        """1-4 -- switch Tools & MCP mode; no-op outside the TOOLS category."""
        if self.active_category != SettingsCategoryId.TOOLS.value:
            return
        pane = self._tools_hub_pane()
        if pane is not None:
            pane.activate_mode(mode)
```

3. TOOLS branches at the top of the three verb actions (self-guard shape copied from `action_settings_rag_set_active`'s docstring rule):

In `action_settings_save_category` (~33845):

```python
        if self.active_category == SettingsCategoryId.TOOLS.value:
            # Tools & MCP has no draft state; `s` stays unadvertised there
            # (footer hints below) and answers with guidance like other
            # non-guided categories.
            self.app.notify(
                "Tools & MCP applies changes immediately — nothing to save."
            )
            return
```

In `action_settings_revert_category` (~34748), before the existing body:

```python
        if self.active_category == SettingsCategoryId.TOOLS.value:
            pane = self._tools_hub_pane()
            if pane is not None:
                pane.refresh_now()  # `r` = force reload in this category
            return
```

In `action_settings_test_category` (~35022), before the existing body:

```python
        if self.active_category == SettingsCategoryId.TOOLS.value:
            pane = self._tools_hub_pane()
            if pane is not None:
                pane.test_selected_tool()  # `t` = test tool in this category
            return
```

And in `action_settings_rag_set_active` (~35171), before the existing body:

```python
        if self.active_category == SettingsCategoryId.TOOLS.value:
            # `a` = add server in this category (per-category dispatch, the
            # same pattern s/r/t use).
            pane = self._tools_hub_pane()
            if pane is not None:
                pane.add_server()
            return
```

4. Footer hints in `_category_footer_shortcuts` (~3695), before `return tuple(shortcuts)`:

```python
        if category is SettingsCategoryId.TOOLS:
            pane = self._tools_hub_pane()
            mode = pane.active_mode if pane is not None else "servers"
            shortcuts.extend(TOOLS_HUB_MODE_SHORTCUTS.get(mode, ()))
```

5. Register/clear the hint set with the screen's footer when entering/leaving the category and on mode change (the retired screen did this in `on_mount`/`on_screen_resume`/`on_mcp_workbench_mode_changed`). Add to `SettingsScreen`:

```python
    def _refresh_tools_footer_shortcuts(self) -> None:
        """Register the active Tools-mode hint set; clear it otherwise."""
        if self.active_category != SettingsCategoryId.TOOLS.value:
            self.clear_footer_shortcuts(source="settings-tools")
            return
        pane = self._tools_hub_pane()
        mode = pane.active_mode if pane is not None else "servers"
        self.register_footer_shortcuts(
            source="settings-tools",
            shortcuts=TOOLS_HUB_MODE_SHORTCUTS.get(mode, ()),
        )

    def on_tools_hub_panel_mode_changed(
        self, event: ToolsHubPanel.ModeChanged
    ) -> None:
        event.stop()
        self._refresh_tools_footer_shortcuts()
```

Call `self._refresh_tools_footer_shortcuts()` at the end of `_select_category` (~26638) — it self-clears for every non-TOOLS category, so one call site covers enter and leave.

- [ ] **Step 4: Run the tests**

```bash
python -m pytest Tests/UI/test_settings_tools_hub.py Tests/UI/test_screen_footer_hints.py -q
```

Expected: PASS (footer-hints file guards the "only advertised keys work" rule).

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/UI/Screens/settings_screen.py Tests/UI/test_settings_tools_hub.py
git commit -m "feat(ui): Tools & MCP per-category keys and per-mode footer hints (task-<id>)"
```

---

### Task 4: Deep links — navigation context + Tool Profiles handoff

**Files:**
- Modify: `tldw_chatbook/UI/Screens/settings_screen.py` (`apply_navigation_context` TOOLS branch, handoff handler)
- Test: `Tests/UI/test_settings_tools_hub.py`, `Tests/UI/test_mcp_permissions_handoff.py`

**Interfaces:**
- Consumes: Task 2's `ToolsHubPanel.set_initial_view_state`; the bounded-context validation ported verbatim from `UI/Screens/mcp_screen.py::apply_navigation_context` (lines 161-194 pre-deletion); `_after_category_panes(callback)` (~line 4822).
- Produces: Settings context contract `{"category": "tools", "mode": str, "tool_policy_profile_id": str, "profile_revision": int | None, "profile_policy_digest": str}` — Task 5's route map and compat hotkey produce exactly `{"category": "tools"}` and `{"category": "tools", "mode": ...}`.

- [ ] **Step 1: Write the failing tests**

Append to `Tests/UI/test_settings_tools_hub.py`:

```python
def _valid_profile_context() -> dict[str, object]:
    return {
        "category": "tools",
        "mode": "permissions",
        "tool_policy_profile_id": "research",
        "profile_revision": 3,
        "profile_policy_digest": "a" * 64,
    }


async def test_apply_navigation_context_accepts_bounded_tools_context():
    async with app.run_test() as pilot:
        settings = app.screen
        settings.apply_navigation_context(_valid_profile_context())
        await pilot.pause()
        assert settings.active_category == SettingsCategoryId.TOOLS.value
        pane = settings.query_one("#settings-tools-hub", ToolsHubPanel)
        assert pane.workbench.active_mode == "permissions"


async def test_apply_navigation_context_rejects_malformed_tools_context():
    async with app.run_test() as pilot:
        settings = app.screen
        settings._select_category(SettingsCategoryId.APPEARANCE.value)
        bad = _valid_profile_context()
        bad["profile_policy_digest"] = "zz"  # not 64 hex chars
        settings.apply_navigation_context(bad)
        await pilot.pause()
        assert settings.active_category != SettingsCategoryId.TOOLS.value
```

- [ ] **Step 2: Run to verify failure**

```bash
python -m pytest Tests/UI/test_settings_tools_hub.py -q
```

Expected: FAIL — the TOOLS branch doesn't exist; `apply_navigation_context` ignores unknown-but-valid category handling only via generic `_select_category` and won't set the mode/profile state (first test fails on `active_mode`).

- [ ] **Step 3: Implement the TOOLS branch in `apply_navigation_context` (~line 25187)**

Insert before the `PROVIDERS_MODELS` fall-through (mirroring the SPEECH_TTS branch shape):

```python
        if category_value == SettingsCategoryId.TOOLS.value:
            allowed_keys = {
                "category",
                "mode",
                "tool_policy_profile_id",
                "profile_revision",
                "profile_policy_digest",
            }
            if not set(context).issubset(allowed_keys):
                logger.debug("Ignoring unsafe Tools Settings navigation context")
                return
            view_state: dict[str, object] | None = None
            mode = context.get("mode")
            if mode is not None:
                if not isinstance(mode, str) or mode not in MCP_HUB_MODES:
                    logger.debug("Ignoring invalid Tools Settings mode: %r", mode)
                    return
                view_state = {"mode": mode}
            profile_id = context.get("tool_policy_profile_id")
            if profile_id is not None:
                revision = context.get("profile_revision")
                digest = context.get("profile_policy_digest")
                # Same bounds the retired MCPScreen.apply_navigation_context
                # enforced: non-empty id <=128 chars, revision None or >=1,
                # digest exactly 64 hex chars.
                if (
                    type(profile_id) is not str
                    or not profile_id
                    or len(profile_id) > 128
                    or (
                        revision is not None
                        and (type(revision) is not int or revision < 1)
                    )
                    or (
                        type(digest) is not str
                        or len(digest) != 64
                        or any(
                            character not in "0123456789abcdef"
                            for character in digest
                        )
                    )
                ):
                    logger.debug(
                        "Ignoring malformed Tools profile deep-link context"
                    )
                    return
                view_state = {
                    "mode": "permissions",
                    "tool_policy_profile_id": profile_id,
                    "profile_revision": revision,
                    "profile_policy_digest": digest,
                }
            self._tools_hub_initial_view_state = (
                dict(view_state) if view_state is not None else None
            )
            self._clear_navigation_provider_context()
            self._select_category(category_value, restore_focus=True)
            pane = self._tools_hub_pane()
            if pane is not None:
                pane.set_initial_view_state(self._tools_hub_initial_view_state)
            return
```

Add `MCP_HUB_MODES` to the existing `from .settings_tools_hub import …`? No — import it from the workbench module alongside the pane import at the top of the file:

```python
from ..MCP_Modules.mcp_workbench import MCP_HUB_MODES
```

- [ ] **Step 4: Rewire the Tool Profiles handoff**

In the `ToolProfilesPanel.EditPolicyRequested` handler (~line 6270), replace the `self.post_message(NavigateToScreen("mcp", {…}))` call with:

```python
        self.post_message(
            NavigateToScreen(
                "settings",
                {
                    "category": "tools",
                    "mode": "permissions",
                    "tool_policy_profile_id": event.profile_id,
                    "profile_revision": event.revision,
                    "profile_policy_digest": event.policy_digest,
                },
            )
        )
```

- [ ] **Step 5: Update and run the handoff test**

`Tests/UI/test_mcp_permissions_handoff.py` currently builds on `from tldw_chatbook.UI.Screens.mcp_screen import MCPScreen` and asserts routing to the MCP screen. Update it: drop the `MCPScreen` import; the test's real-app flow (Tool Profiles → Edit policy) should now assert it lands on the Settings screen with `#settings-tools-hub` mounted, `MCPPermissionsMode` visible, and the target profile selected — the existing `_visible`/`_cursor_visible` assertions stay, applied to the matrix inside Settings. Keep both theme and size parametrizations.

```bash
python -m pytest Tests/UI/test_settings_tools_hub.py Tests/UI/test_mcp_permissions_handoff.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add tldw_chatbook/UI/Screens/settings_screen.py \
  Tests/UI/test_settings_tools_hub.py \
  Tests/UI/test_mcp_permissions_handoff.py
git commit -m "feat(ui): Tools & MCP deep links; Tool Profiles handoff goes internal (task-<id>)"
```

---

### Task 5: Runtime-backend forwarding (small, folds the retired screen's last job)

**Files:**
- Modify: `tldw_chatbook/UI/Screens/settings_screen.py`
- Test: `Tests/UI/test_settings_tools_hub.py`

**Interfaces:**
- Consumes: `ToolsHubPanel.handle_runtime_backend_change(str)` (Task 2); the app calls `active_screen.handle_runtime_backend_changed(resolved_backend)` after runtime commits (`tldw_chatbook/app.py` ~line 2038).
- Produces: none (terminal behavior).

- [ ] **Step 1: Failing test**

```python
async def test_runtime_backend_change_forces_pane_reload():
    async with app.run_test() as pilot:
        settings = app.screen
        settings._select_category(SettingsCategoryId.TOOLS.value)
        await pilot.pause()
        pane = settings.query_one("#settings-tools-hub", ToolsHubPanel)
        reloaded: list[bool] = []
        pane.handle_runtime_backend_change = lambda backend: reloaded.append(True)  # type: ignore[assignment]
        await settings.handle_runtime_backend_changed("local")
        assert reloaded == [True]
```

- [ ] **Step 2: Verify failure**

```bash
python -m pytest Tests/UI/test_settings_tools_hub.py::test_runtime_backend_change_forces_pane_reload -q
```

Expected: FAIL — `SettingsScreen` has no `handle_runtime_backend_changed`.

- [ ] **Step 3: Implement on `SettingsScreen`**

```python
    async def handle_runtime_backend_changed(self, runtime_backend: str) -> None:
        """Forward runtime-backend changes to the Tools & MCP pane (ADR-237).

        Args:
            runtime_backend: Newly active runtime backend identifier.
        """
        pane = self._tools_hub_pane()
        if pane is not None:
            pane.handle_runtime_backend_change(runtime_backend)
```

- [ ] **Step 4: Verify pass + commit**

```bash
python -m pytest Tests/UI/test_settings_tools_hub.py -q
git add tldw_chatbook/UI/Screens/settings_screen.py Tests/UI/test_settings_tools_hub.py
git commit -m "feat(ui): Settings forwards runtime backend changes to Tools & MCP (task-<id>)"
```

---

### Task 6: Retire the destination — routes, hotkey, palette; delete mcp_screen.py; test sweep

Atomic by necessity: deleting `mcp_screen.py` breaks every importing test, so the retirement and the import sweep land together.

**Files:**
- Modify: `tldw_chatbook/UI/Navigation/shell_destinations.py`, `tldw_chatbook/UI/Navigation/screen_registry.py`, `tldw_chatbook/app.py`, `tldw_chatbook/app_lifecycle.py`
- Delete: `tldw_chatbook/UI/Screens/mcp_screen.py`
- Test: `Tests/UI/test_screen_navigation.py`, `Tests/UI/test_destination_shells.py`, `Tests/ProductionApp/test_retired_destination_root_state.py`, `Tests/ProductionApp/test_reactive_ownership_maturity.py`, `Tests/Performance/test_ui_latency_guardrails.py`, `Tests/UI/test_latest_dev_core_app_usability_smoke.py`, `Tests/UI/test_destination_visual_parity_correction.py`, `Tests/UI/test_mount_io_off_pump.py`, `Tests/UI/test_widget_css_consolidation.py`, `Tests/UI/test_screen_footer_hints.py`, `Tests/UI/test_product_maturity_phase6_recovery_docs.py`, `Tests/Backup_Recovery/test_mcp_recovery_catalog.py`, `Tests/Backup_Recovery/test_mcp_recovery_publication.py`, `Tests/UI/test_mcp_workbench.py`

**Interfaces:**
- Consumes: Task 2-5 Settings surface; `ARTIFACTS_COMPATIBILITY_SHORTCUT` pattern (`shell_destinations.py` ~217, `app.py` ~948, `app_lifecycle.py:1845`).
- Produces: `MCP_COMPATIBILITY_SHORTCUT = "ctrl+9"`; `action_settings_tools_hub()` on the app; route aliases `"mcp" -> "settings"` and `"tools_settings" -> "settings"`.

- [ ] **Step 1: Write the failing route/shortcut tests**

In `Tests/UI/test_screen_navigation.py`, replace `test_legacy_tools_settings_route_uses_mcp_context` (~line 570) with:

```python
def test_legacy_mcp_routes_alias_to_settings_tools():
    app = _build_test_app()

    for route in ("mcp", "tools_settings"):
        screen_name, current_tab, screen_class = (
            app._resolve_screen_navigation_target(route)
        )
        assert screen_name == "settings"
        assert current_tab == "settings"
        assert screen_class.__name__ == "SettingsScreen"


def test_mcp_shell_routes_resolve_to_settings_destination():
    from tldw_chatbook.UI.Navigation.shell_destinations import resolve_shell_route

    for route in ("mcp", "tools_settings"):
        resolved = resolve_shell_route(route)
        assert resolved.destination_id == "settings"
        assert resolved.canonical_route == "settings"


def test_ctrl9_is_tools_compatibility_binding():
    from tldw_chatbook.UI.Navigation.shell_destinations import (
        SHELL_DESTINATION_SHORTCUTS,
        MCP_COMPATIBILITY_SHORTCUT,
    )

    assert MCP_COMPATIBILITY_SHORTCUT == "ctrl+9"
    assert "mcp" not in SHELL_DESTINATION_SHORTCUTS
    assert "ctrl+9" not in set(SHELL_DESTINATION_SHORTCUTS.values())
```

(Use the file's existing `_build_test_app` import — the old test at ~line 570 calls `app._resolve_screen_navigation_target(route)`; keep that exact call. Add `@pytest.mark.asyncio` to the async tests below if the file isn't in pytest-asyncio auto mode.)

- [ ] **Step 2: Verify failure**

```bash
python -m pytest Tests/UI/test_screen_navigation.py -k "mcp or tools_settings" -q
```

Expected: FAIL.

- [ ] **Step 3: Retire the destination**

1. `shell_destinations.py` — delete the `ShellDestination("mcp", …)` block (~lines 114-121) and the `"mcp": "ctrl+9"` entry (~line 206). Add next to `ARTIFACTS_COMPATIBILITY_SHORTCUT` (~line 217):

```python
# ADR-237: retained hotkey for the retired MCP destination, now a
# Settings > Tools & MCP deep-link outside permanent navigation.
MCP_COMPATIBILITY_SHORTCUT = "ctrl+9"
```

Add `"mcp"` to `_ROUTABLE_LEGACY_ROUTES` (keep `"tools_settings"`, already present ~line 231), and after the artifacts line (~line 284):

```python
_ROUTE_MAP["mcp"] = ResolvedShellRoute("settings", "settings", "mcp")
_ROUTE_MAP["tools_settings"] = ResolvedShellRoute(
    "settings", "settings", "tools_settings"
)
```

Extend the Settings destination's `palette_aliases` (the `research` destination shows the field) with `("tools", "mcp", "tool settings")`.

2. `screen_registry.py` — delete the `"mcp"` (~154) and `"tools_settings"` (~178) `ScreenRoute` entries; add to `_SCREEN_ALIASES` (~230):

```python
    # ADR-237: the MCP destination retired into Settings > Tools & MCP.
    "mcp": "settings",
    "tools_settings": "settings",
```

3. `app.py` — import `MCP_COMPATIBILITY_SHORTCUT` next to `ARTIFACTS_COMPATIBILITY_SHORTCUT` (~line 329) and add a binding beside the artifacts one (~line 948):

```python
        # ADR-237: preserve muscle memory after MCP folds into Settings.
        Binding(
            MCP_COMPATIBILITY_SHORTCUT,
            "settings_tools_hub",
            "Settings Tools & MCP",
            show=False,
        ),
```

Add to `_LEGACY_ROUTE_LIBRARY_NAV_CONTEXT` (~line 2473, where `"customize": {"category": "theme"}` already lives):

```python
        "mcp": {"category": "tools"},
        "tools_settings": {"category": "tools"},
```

4. `app_lifecycle.py` — next to `action_library_artifacts` (~line 1845):

```python
    def action_settings_tools_hub(self) -> None:
        """Preserve Ctrl+9 as a Tools & MCP route outside the shell strip."""
        self.post_message(NavigateToScreen("mcp"))
```

(The bare legacy route rides the alias + default context from items 2-3, exactly like ctrl+6 → `TAB_ARTIFACTS`.)

- [ ] **Step 4: Verify the route tests pass, then delete the screen**

```bash
python -m pytest Tests/UI/test_screen_navigation.py -q
git rm tldw_chatbook/UI/Screens/mcp_screen.py
```

- [ ] **Step 5: Test migration sweep**

Run each file, fix the `MCPScreen` import/route fallout, guided by:

- `Tests/UI/test_destination_shells.py` — the mcp block (`DestinationHarness(app, "mcp")` cases ~2561-2967): the harness will now mount `SettingsScreen` via the alias; port each case to select the TOOLS category and assert against `#settings-tools-hub` / `MCPWorkbench` instead of `MCPScreen` shell markers (`#mcp-title` is gone; the purpose line is `#tools-hub-purpose`). Mode-chip and mode-key assertions map 1:1 to the pane.
- `Tests/ProductionApp/test_retired_destination_root_state.py` — replace the `MCPScreen` import and mcp root-state case with alias assertions (`resolve "mcp" → settings`); `tools_settings_active_view` stays in `REMOVED_ROOT_NAMES`.
- `Tests/UI/test_mcp_workbench.py` — mounts `MCPWorkbench` directly; only imports/paths change if it referenced `mcp_screen` (it should not — verify).
- The rest (`test_reactive_ownership_maturity`, `test_ui_latency_guardrails`, `test_latest_dev_core_app_usability_smoke`, `test_destination_visual_parity_correction`, `test_mount_io_off_pump`, `test_widget_css_consolidation`, `test_screen_footer_hints`, `test_product_maturity_phase6_recovery_docs`, `test_mcp_recovery_catalog`, `test_mcp_recovery_publication`) — mechanical: swap `MCPScreen` mounts for Settings-with-TOOLS-category or `ToolsHubPanel` mounts; keep each test's assertions otherwise identical.

```bash
python -m pytest Tests/UI/test_destination_shells.py \
  Tests/UI/test_screen_navigation.py \
  Tests/UI/test_screen_footer_hints.py \
  Tests/UI/test_mcp_workbench.py \
  Tests/UI/test_settings_tools_hub.py \
  Tests/UI/test_mcp_permissions_handoff.py \
  Tests/ProductionApp/test_retired_destination_root_state.py \
  Tests/ProductionApp/test_reactive_ownership_maturity.py \
  Tests/Performance/test_ui_latency_guardrails.py \
  Tests/UI/test_latest_dev_core_app_usability_smoke.py \
  Tests/UI/test_destination_visual_parity_correction.py \
  Tests/UI/test_mount_io_off_pump.py \
  Tests/UI/test_widget_css_consolidation.py \
  Tests/UI/test_product_maturity_phase6_recovery_docs.py \
  Tests/Backup_Recovery/test_mcp_recovery_catalog.py \
  Tests/Backup_Recovery/test_mcp_recovery_publication.py -q
```

Expected: PASS. Also run the token-governance guard since CSS moved into the pane:

```bash
python -m pytest Tests/UI/test_design_token_governance.py -q
```

Expected: PASS (no new hex literals; tokens only).

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "feat(ui): retire MCP destination into Settings Tools & MCP; ctrl+9 compat deep-link (task-<id>)"
```

---

### Task 7: Docs, live verification, backlog hygiene

**Files:**
- Modify: `AGENTS.md` (Special Systems > Tool Calling mention of the MCP screen), `Docs/Architecture/` tools/MCP doc if it names the destination (check `Docs/Architecture/README.md` index), `backlog/tasks/task-<id> …`
- Consumes: everything above.

- [ ] **Step 1: Update docs**

- `AGENTS.md` Special Systems > Tool Calling: replace any "MCP destination" wording with "Settings > Tools & MCP (ADR-237)".
- Check `Docs/Architecture/` for the MCP/tools subsystem doc; update its "surface" section to name the Settings category; leave the workbench internals description untouched.
- `backlog/decisions/015-shell-destination-ia.md`: append an `Amended by: ADR-237` line under Status, matching how ADR-210 amended it.

- [ ] **Step 2: Live verification (per `backlog/docs/lessons-live-verification.md`)**

```bash
python3 -m tldw_chatbook.app
```

Verify by hand: F4 → "Tools & MCP" category renders the four modes; each mode chip switches; `1-4`, `a`, `r`, `t` work in the category and keep their old meaning elsewhere; ctrl+9 lands on Settings → Tools & MCP; command palette search "tools" and "mcp" finds Settings; Tool Profiles → Edit policy lands on the permissions matrix with the profile selected; the nav strip shows 13 destinations with no MCP entry and no shortcut gaps.

- [ ] **Step 3: Close the backlog task**

Tick every AC checkbox in the task file, add Implementation Notes (approach, decisions, modified files — include the spec/ADR links), then:

```bash
backlog task edit <id> -s Done --notes "Implemented per ADR-237; see Implementation Notes"
```

- [ ] **Step 4: Commit**

```bash
git add AGENTS.md Docs/Architecture backlog/decisions/015-shell-destination-ia.md "backlog/tasks/task-<id> - Retire-MCP-destination-into-Settings-Tools-and-MCP-hub.md"
git commit -m "docs: Tools & MCP in Settings documentation closeout (task-<id>)"
```

---

## Self-review notes

- Spec coverage: §1 navigation → Task 6; §2 category+pane → Task 2; §3 keybindings → Task 3; §4 deep links → Tasks 4+6 (compat hotkey produces the `{"category": "tools"}` context via the legacy-route map); §5 naming → Task 2 summary entry + Task 6 palette aliases; §6 non-goals → Global Constraints; §7 ADR → Task 1; §8 testing → per-task tests + Task 6 sweep; §9 rollout slicing → task order (pane first, retirement last); Risks table → mitigations embedded in tasks.
- The spec's "port `handle_runtime_backend_changed`" is Task 5; the spec's startup-config compatibility (`default_tab = "mcp"`) is covered by Task 6's alias + default-context entries (startup path reads the same map at `app.py` ~4354).
- Known drift risk: line numbers in this plan were verified against the working tree on 2026-10-16; if earlier tasks shift lines, the anchor greps (`grep -n "SettingsCategoryId.TOOL_PROFILES"` etc.) in each step are authoritative, not the numbers.
