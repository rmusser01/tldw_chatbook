---
id: TASK-36523
title: css/main.tcss imports two deleted sheets and CSS-MODULAR.md describes a CSS build the app no longer has
status: To Do
created_date: 2026-10-24 00:26
labels:
- follow-up
- css
- docs
- cleanup
priority: low
references:
- tldw_chatbook/css/main.tcss
- tldw_chatbook/css/CSS-MODULAR.md
- tldw_chatbook/css/build_css.py
- Tests/UI/test_master_shell_design_system_contract.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by TASK-36382 (PR #3227, 2026-10-23), checked on dev 5b5920b8e2.

`css/main.tcss` imports two sheets that are no longer in the tree:
- `features/_tools-settings.tcss` (:50), deleted with ToolsSettingsWindow (TASK-33081/33244);
- `features/_evaluation.tcss` (:53).

The app does not load `main.tcss`. Its stylesheets come from `build_css.py`'s own manifest, which already records the `_tools-settings` removal (build_css.py:299). The only reader of `main.tcss` is `test_master_shell_design_system_contract.py:103-106`, which checks one `@import` line. So the file looks like the stylesheet entry point but is not one.

`css/CSS-MODULAR.md` tells developers the opposite:
- It shows `main.tcss` as the "Main entry point" (:24) and says "Updated app.py to use main.tcss" (:257).
- Its "Automatic CSS Building" section (:402-415) says that `app.py` checks module timestamps at startup, rebuilds with `build_css.py`, and falls back to legacy CSS from Constants. `app.py` only reads paths from `build_css` (`screen_css_paths`, `widget_defaults_sources`). There is no startup rebuild.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 No .tcss file in the tree imports a sheet that does not exist, and main.tcss is either removed together with its test reference or marked as not loaded by the app
- [ ] #2 CSS-MODULAR.md describes how the app's CSS is built and loaded today, including how the bundle is regenerated and checked (preflight)
<!-- AC:END -->
