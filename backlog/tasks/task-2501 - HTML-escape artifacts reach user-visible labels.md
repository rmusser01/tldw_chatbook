---
id: TASK-2501
title: HTML-escape artifacts reach user-visible labels
status: To Do
assignee: []
created_date: '2026-08-04 21:52'
labels:
  - console
  - ui
  - polish
dependencies: []
priority: low
---

## Description

`tldw_chatbook/Chat/console_display_state.py`'s `_safe_display_text` helper (~line 73) runs every title/source/status/snippet value it touches through `html.escape`. That escaped text reaches at least two rendered surfaces:

- The staged-evidence strip's per-row labels (`build_console_staged_evidence_strip_state`, ~lines 699-712; rendered by `console_staged_evidence_strip.py`) and the Inspector's "Sources" tray summary and evidence rows (`ConsoleStagedContextState.from_live_work`, ~lines 295-327 and 552-557; rendered by `console_staged_context.py`).
- The Console Settings modal's "Sources" line (`console_settings_modal.py::_sources_label`, ~line 1411, feeding the `Static` at ~line 542) — this reads `ConsoleSettingsContextEstimate.staged_context_summary`, which `chat_screen.py` (~line 4199) sets directly from the same `ConsoleStagedContextState.summary` the tray already escaped.

Every one of these destinations is a `Static` composed with `markup=False` (`console_staged_evidence_strip.py` ~lines 61/70/83/91; `console_staged_context.py` ~lines 84/98/104; `console_settings_modal.py` ~line 545) — a markup-free sink that would render a raw `&` correctly on its own. Because the text was already HTML-escaped upstream, a title containing `&` renders literally as `&amp;` instead. This was live-verified 2026-08-04 (PR-T1 live check, `live-check-report.md` S1): a media item titled `notes with spaces & émojis 🚀` rendered as `notes with spaces &amp; émojis 🚀` in the strip.

This is not a one-off bug in a single widget: `_safe_display_text` is the module-wide convention in `console_display_state.py` (per PR-T1's I1 fix, `2c5600041`, which deliberately extended the *same* escaping to a previously-unescaped field for consistency with this file's siblings). Fixing one call site would just leave the artifact in the others and re-diverge the convention PR-T1 just unified.

## Acceptance Criteria

- [ ] Titles/labels containing `&` (and other HTML-special characters) render literally, not as HTML entities, in the staged-evidence strip, the Inspector "Sources" tray, and the Console Settings modal's "Sources" line
- [ ] The fix is applied module-wide in `console_display_state.py` (e.g. `_safe_display_text` stops HTML-escaping when every consumer is a markup-free sink), not patched one call site at a time
- [ ] The existing Rich-markup-injection protection these call sites also provide (e.g. a title containing `[/]` or `[bold]`) is unchanged — this task fixes the redundant HTML-escaping artifact, not the markup-safety behavior PR-T1's I1 fix (`2c5600041`) shipped
- [ ] A test pins a `&`-containing title rendering literally on at least the strip and the Settings modal's "Sources" line
