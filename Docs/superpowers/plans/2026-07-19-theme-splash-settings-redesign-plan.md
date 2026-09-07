# Theme & Splash Settings Redesign — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move the Theme Editor and Splash Screen into the Settings screen as first-class sidebar categories, retire the standalone Customize screen, and fix animated splash previews.

**Tech Stack:** Python 3.11+, Textual ≥3.3.0, pytest, existing tldw_chatbook settings/configuration hub code.

---

## Phase 0 — Audit all references to retiring widgets/routes

### Task 0: Audit all references to retiring widgets/routes

**Files:** all Python files

- [x] **Step 1: Run grep**

```bash
grep -rn "Theme_Editor_Window\|ThemeEditorView\|SplashScreenViewer\|CustomizeScreen\|CustomizeWindow\|TAB_CUSTOMIZE\|customize-window" tldw_chatbook Tests --include="*.py"
```

- [x] **Step 2: Record every hit**

See "Reference Audit Results" below.

- [x] **Step 3: Commit audit note**

```bash
git add docs/superpowers/plans/2026-07-19-theme-splash-settings-redesign-plan.md
git commit -m "docs(plan): record Customize/theme/splash reference audit"
```

### Reference Audit Results

Grep command run from the worktree root:

```bash
grep -rn "Theme_Editor_Window\|ThemeEditorView\|SplashScreenViewer\|CustomizeScreen\|CustomizeWindow\|TAB_CUSTOMIZE\|customize-window" tldw_chatbook Tests --include="*.py"
```

#### Retiring UI files

Files that define the standalone Customize screen/window or the widget classes that currently live inside it.

- `tldw_chatbook/UI/Customize_Window.py`
  - `23` — `class CustomizeWindow(Container):`
  - `34` — `CustomizeWindow {` (CSS block)
  - `214` — `from .Theme_Editor_Window import ThemeEditorView`
  - `215` — `theme_editor = ThemeEditorView()`
  - `241` — `from ..Widgets.splash_screen_viewer import SplashScreenViewer`
  - `242` — `splash_viewer = SplashScreenViewer()`
- `tldw_chatbook/UI/Screens/customize_screen.py`
  - `10` — `from ..Customize_Window import CustomizeWindow`
  - `16` — `class CustomizeScreen(BaseAppScreen):`
  - `27` — `self.customize_window = CustomizeWindow(self.app_instance, classes="window")`
  - `43` — docstring: `"Forward button events to the CustomizeWindow handler."`
- `tldw_chatbook/UI/Screens/__init__.py`
  - `20` — `"CustomizeScreen": ".customize_screen",` (lazy loader map)
  - `33` — `'CustomizeScreen',` (public export list)
- `tldw_chatbook/UI/Theme_Editor_Window.py`
  - `1` — module header `# tldw_chatbook/UI/Theme_Editor_Window.py`
  - `44` — `class ThemeEditorView(VerticalScroll):`
  - `48` — `ThemeEditorView {` (CSS block)
- `tldw_chatbook/Widgets/splash_screen_viewer.py`
  - `98` — `class SplashScreenViewer(Container):`

#### Settings screen references

Files where the Settings screen currently embeds or links out to the retiring Customize/theme/splash pieces.

- `tldw_chatbook/UI/Tools_Settings_Window.py`
  - `39` — `from .Theme_Editor_Window import ThemeEditorView`
  - `2572` — `yield ThemeEditorView()`
  - `4305` — docstring: `"Lazily load the SplashScreenViewer when the gallery tab is first accessed."`
  - `4317` — `from ..Widgets.splash_screen_viewer import SplashScreenViewer`
  - `4318` — `viewer = SplashScreenViewer(classes="embedded-splash-viewer")`
  - `4322` — `"SplashScreenViewer loaded successfully"`
  - `4325` — `"Failed to load SplashScreenViewer: {e}"`
- `tldw_chatbook/UI/Screens/settings_screen.py`
  - `831` — docstring mentions "Theme, density, and visual defaults shared with the app shell."
  - `1072` — `"general.default_theme"`
  - `1073` — `"general.palette_theme_limit"`
  - `1079` — `reads_runtime_state_from=("app theme", "Customize destination")`
  - `1081` — `runtime_owner="Settings persisted defaults; Customize full theme editor"`
  - `1083-1084` — comments about Settings vs. Customize ownership
  - `1922-1925` — theme/palette theme limit setting key resolution
  - `1938-1950` — theme-related anchor IDs and synced settings
  - `1978-2006` — `_appearance_theme_options()` and theme/density/font summary
  - `3155-3165` — `_appearance_theme_summary()`
  - `5019-5030` — theme field help text
  - `5063` — help text: "Configure global visual defaults without replacing Customize."
  - `5146-5149` — help text referencing opening Customize
  - `6253-6270` — appearance/theme controls and "Open Customize" button
  - `6690-6710` — destination section and "Open Customize" button
  - `6768-6770` — another "Open Customize" button
  - `7006-7007` — theme fields in the synced-fields list
  - `7067` — `self.post_message(NavigateToScreen("customize"))`
  - `7074` — `handle_appearance_theme_changed`
  - `7077-7082` — `handle_appearance_palette_theme_limit_changed`
  - `8571` — applies `default_theme` to `self.app_instance.theme`
  - `8841-8848` — updates theme/palette limit controls from saved values

#### Navigation/route references

Files that register or route the `customize` destination.

- `tldw_chatbook/UI/Navigation/screen_registry.py`
  - `83` — `"customize": ScreenRoute("customize", "customize", "tldw_chatbook.UI.Screens.customize_screen", "CustomizeScreen")`
- `tldw_chatbook/UI/Navigation/shell_destinations.py`
  - `130` — `("customize", "logs", "stats")`
  - `155` — `"customize",`
- `tldw_chatbook/UI/Workbench/route_inventory.py`
  - `48` — `"customize": "settings",`

#### Constants/app references

Core app constants, imports, bindings, and CSS selectors tied to the Customize route/window.

- `tldw_chatbook/Constants.py`
  - `33` — `TAB_CUSTOMIZE = "customize"`
  - `58` — `ALL_TABS` list includes `TAB_CUSTOMIZE`
  - `78` — `TAB_CUSTOMIZE: "Customize"` (tab label map)
- `tldw_chatbook/app.py`
  - `84` — `from tldw_chatbook.Constants import ..., TAB_CUSTOMIZE, ...`
  - `234` — `from .UI.Customize_Window import CustomizeWindow`
  - `548` — `TAB_CUSTOMIZE: "Switch to appearance customization"`
  - `2226` — CSS id list includes `"customize-window"`
  - `4342` — event-handler map entry: `TAB_CUSTOMIZE: {},  # Customize handles its own events`

#### Test references

Tests that import or assert behavior around the Customize tab/window.

- `Tests/UI/test_command_palette_shell_routes.py`
  - `6` — `TAB_CUSTOMIZE,`
  - `107` — `TAB_CUSTOMIZE,`
- `Tests/UI/test_non_obscuring_focus_contract.py`
  - `1261` — `from tldw_chatbook.UI.Customize_Window import CustomizeWindow`
  - `1263` — `css_block(CustomizeWindow.DEFAULT_CSS, ".customize-nav-button.active-nav")`
