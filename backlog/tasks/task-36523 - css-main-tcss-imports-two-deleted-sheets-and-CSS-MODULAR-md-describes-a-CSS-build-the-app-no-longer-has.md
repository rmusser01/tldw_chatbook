---
id: TASK-36523
title: css/main.tcss imports two deleted sheets and CSS-MODULAR.md describes a CSS build the app no longer has
status: In Progress
assignee:
- '@claude'
updated_date: 2026-11-03 08:05
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

Correction (2026-11-03): there is a startup rebuild. It lives in `tldw_chatbook/app_entry.py`, moved from `app.py` by TASK-33011. On source checkouts, `main_cli_runner`, `_run_module_main` and `get_app` call `_generated_css_is_stale()` and run `build_css.py` when it reports stale sheets.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 No .tcss file in the tree imports a sheet that does not exist, and main.tcss is either removed together with its test reference or marked as not loaded by the app
- [x] #2 CSS-MODULAR.md describes how the app's CSS is built and loaded today, including how the bundle is regenerated and checked (preflight)
- [x] #3 The CSS build's own messages and helper scripts describe the current build: no stale app.py instruction, no timestamp claim the bundle header no longer carries, and no helper that runs an interpreter below the project's Python floor
- [ ] #4 No stylesheet under tldw_chatbook/css is left that no build input or loader reads
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Confirm what the app loads: `TldwCli.CSS_PATH`, `_get_default_css`, the screen/route/modal split sheets, the source-tree boot rebuild in `app_entry.py`, and that nothing but one contract-test assertion reads `main.tcss` (Textual CSS has no `@import`).
2. Add a guard test that every `@import` in a `.tcss` file under the package resolves; show it red against the two missing sheets.
3. Delete `main.tcss` (a hand-kept copy of `CSS_MODULES` that already lacks most modules) and its test reference; the guard goes green.
4. Rewrite `CSS-MODULAR.md` to describe today's build and load path, the boot rebuild, and how the bundle is regenerated and checked (`check_bundle_sync.py`, preflight, CI).
5. Run the covering tests and preflight.
6. (Coordinator follow-up, AC #3.) Move the guard into `Tests/UI/test_css_staleness_manifest.py`, a whole-file UI census entry, so a PR lane runs it, and show it red there with a temporary bad `@import`.
7. Replace `build_css.py` `main()`'s stale "Update app.py" instructions and the docstrings' stale claims: the timestamp claim in `check_bundle_sync.py`, and the build's list of generated files.
8. `build_css.sh` is referenced by `Docs/Development` pages, so keep it. Make it run an interpreter at or above the 3.12 floor (`$PYTHON`, then the repo `.venv`), as `CSS-MODULAR.md` says, and mention it there.
9. (Review round.) Fix the review's four minor issues: the comment-stripping and cascade-order wording in `CSS-MODULAR.md`; the drift message, split enumeration and "exact" wording in `check_bundle_sync.py`; the stale `widget_css.py` docstring (and `_get_default_css`'s); `build_css.sh`'s `cd` before a relative `$PYTHON`.
10. Replace the guard with a stronger one: no `.tcss` under the package contains `@import`. Show it red with a temporary file.
11. AC #4: sweep the stylesheets under `css/` against every loader and build input. Delete the ones nothing reads, or stop and report any that something still references.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Deleted `css/main.tcss` and its one test reference, and rewrote `CSS-MODULAR.md` as a description of the current build. Then moved the guard test into a PR lane, and updated the build's own messages and `build_css.sh` to match the current build.

- **What loads.** `TldwCli.CSS_PATH` loads `screen_css_scoped.tcss`, the bundle and `screen_css_self.tcss`. `_get_default_css` adds the two widget-defaults sheets. The split sheets load through `LibraryScreen`/`SettingsScreen` `CSS_PATH`, `_SCREEN_OWNED_ROUTE_CSS` and modal `CSS_PATH`s. Nothing reads `main.tcss`, and Textual cannot: a probe parse of `@import "./x.tcss";` raises `TokenError`. `main.tcss` was a hand-kept copy of `CSS_MODULES` that already lacked most modules, so it was deleted rather than repaired.
- **Correction to the description.** There is a startup rebuild. It moved from `app.py` to `app_entry.py` (TASK-33011): on source checkouts, `main_cli_runner`, `_run_module_main` and `get_app` call `_generated_css_is_stale()` and run `build_css.py` when it reports stale sheets. The new doc describes it. What the old doc got wrong was the location, the mtime-only rule and the fallback to CSS in Constants (removed).
- **Guard.** `test_no_stylesheet_imports_a_missing_sheet` checks that every `@import` in a package `.tcss` file resolves. It was first added to `Tests/UI/test_master_shell_design_system_contract.py`, where it went red before the deletion on `['tldw_chatbook/css/main.tcss: ./features/_tools-settings.tcss', 'tldw_chatbook/css/main.tcss: ./features/_evaluation.tcss']` and green after it (with the edited `test_agentic_terminal_tcss_module_is_implemented_and_imported`, 2 passed, under a temporary `@private_profile_test` marker for the local RecoveryRequired trap). That file is in no PR lane, so the guard moved to `Tests/UI/test_css_staleness_manifest.py`, a whole-file entry in `scripts/ui_pr_gate_census.txt` (census unchanged). In its new home it went red with a temporary `css/_ci_red_probe.tcss` importing a missing sheet (`['css/_ci_red_probe.tcss: ./features/_does_not_exist.tcss']`) and green once that file was removed. The contract file's change is only the `main.tcss` removal. After review it was replaced by `test_no_stylesheet_uses_import`, which fails on any line starting with `@import` in a package `.tcss` file. Textual rejects every `@import`, so this guard is shorter and catches strictly more. It went red with a temporary sheet importing the existing `core/_variables.tcss` (`['css/_ci_red_probe.tcss']`), a case the old guard would have passed, and green once that sheet was removed.
- **Build text (AC #3).** `build_css.py` `main()` no longer says "Update app.py to use 'tldw_cli_modular.tcss'"; it says to commit the regenerated sheets and run preflight. Its module docstring now lists the split sheets as well. `check_bundle_sync.py`'s docstrings no longer claim a `Generated:` timestamp (the header has read `Generated: deterministic` since 38ab1fc57e) or that only two sheets are checked. No test pinned the old strings, and no behaviour changed.
- **`build_css.sh` (AC #3).** It is still referenced by three `Docs/Development` pages and historical plans, so it is kept. It ran a bare `python3`, which is the system 3.9 on macOS. It now uses `$PYTHON` when set, else the first of `.venv/bin/python`, `python3`, `python3.14`, `python3.13` and `python3.12` that meets the 3.12 floor, and it works from any directory. Checked: `PYTHON=/usr/bin/python3` (3.9.6) is refused with exit 1; the default chain rebuilds with no change to the committed sheets. `CSS-MODULAR.md` documents it.
- **Review round (4 minor issues).** `CSS-MODULAR.md` now says that only the bundle and the widget-defaults and screen sheets strip comments, since the split sheets keep theirs, and that the `CSS_MODULES` order is only broadly core-to-utilities, since `components/_settings_personal_context.tcss` sits before `features/_settings.tcss` for the Settings split. In `check_bundle_sync.py`, the drift and success messages now name each sheet's own sources (the `.tcss` modules for split sheets, the `BUNDLED_CSS`/`BUNDLED_SCREEN_CSS` declarations for the others). It no longer lists three of the ten split sheets, and it says the comparison is exact apart from the blanked `Generated:` line. The `widget_css.py` docstring and `_get_default_css`'s docstring now describe `widget_defaults_{self,scoped}.tcss` and `screen_css_{scoped,self}.tcss` as they are loaded today; these are docstring-only changes, and `app.py`'s line count is unchanged. `build_css.sh` no longer runs `cd`, so a relative `$PYTHON` resolves from the caller's directory. Checked: the repro `cd tldw_chatbook && PYTHON=../../../../.venv/bin/python ../build_css.sh` fails with the old script (exit 1) and builds with the new one. The new script also built when called through a script path and a `$PYTHON` path that both contain spaces, and still refused a 3.9 `$PYTHON`.
- **AC #4 not met: stopped as instructed.** The sweep compared every tracked `.tcss`/`.css` under `css/` with `CSS_MODULES`, the split modules and sheets, the four boot sheets (`CSS_PATH` and `_get_default_css`), and any path string under `tldw_chatbook/` and `Tests/` (`_SCREEN_OWNED_ROUTE_CSS` and screen/modal `CSS_PATH`s name only generated split sheets). Three sheets are outside the build. `Themes/theme_tester.tcss` is loaded by the standalone `ThemeDemoApp` (`css/Themes/theme_tester.py` `CSS_PATH`), so a loader does read it. `features/_new_ingest.tcss` and `features/_chatbooks_improved.tcss` are loaded by nothing in the app, but tests read them as fixtures. `_new_ingest.tcss` is read by `Tests/UI/test_non_obscuring_focus_contract.py` (`NEW_INGEST`; the `SOURCE_ONLY_CSS_MODULES` not-in-bundle contract; `test_new_ingest_focus_overrides_defer_to_shared_contracts`; `test_new_ingest_drop_zone_active_state_is_readable`) and listed in `Tests/UI/test_design_token_governance.py` `_LEGACY_SHEETS`. `_chatbooks_improved.tcss` is read by `test_non_obscuring_focus_contract.py` (`CHATBOOKS_IMPROVED`, `test_chatbooks_search_input_focus_uses_stable_thin_contracts`), listed in `_LEGACY_SHEETS`, and named in a comment in `UI/Chatbooks_Window_Improved.py:332`. Nothing was deleted. Deleting them needs a decision on those tests (PR #3342 also edits `test_non_obscuring_focus_contract.py`).
- **Checks.** `./scripts/preflight.sh` passed after every change. `Tests/Architecture/test_module_size_ratchet.py` (serial) gave 201 passed. Final covering run with `-n 4` (load average about 26): `test_css_staleness_manifest`, `test_css_bundle_sync_guard`, `test_css_build_integrity`, `test_css_generated_comments`, `test_widget_css_consolidation`, `test_component_pattern_governance`, `test_design_token_governance` and `test_notes_unsaved_indicator_removed` gave 357 passed and 2 failed. The same command at dev base `66155632ed` gave 356 passed and the same 2 failed, with the same messages. One is `test_app_boots_after_removal` (RecoveryRequired `raw_source_selection_changed`). The other is `test_widget_css_consolidation.py::test_base_class_blocks_precede_their_subclasses` (`[scoped] BaseAppScreen is a base of KanbanBoardScreen but is emitted after it`), which no PR lane runs. Earlier rounds: `test_boot_css_byte_budget` and `test_non_obscuring_focus_contract` gave 1 passed and 98 setup RecoveryRequired errors, identical at dev.
- **Files.** `tldw_chatbook/css/main.tcss` (deleted), `tldw_chatbook/css/CSS-MODULAR.md`, `tldw_chatbook/css/build_css.py`, `tldw_chatbook/css/check_bundle_sync.py`, `tldw_chatbook/css/widget_css.py`, `tldw_chatbook/app.py` (docstring), `build_css.sh`, `Tests/UI/test_master_shell_design_system_contract.py`, `Tests/UI/test_css_staleness_manifest.py`.
<!-- SECTION:NOTES:END -->

