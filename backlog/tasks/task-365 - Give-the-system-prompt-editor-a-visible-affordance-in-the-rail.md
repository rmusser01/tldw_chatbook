---
id: TASK-365
title: Give the system-prompt editor a visible affordance in the rail
status: Done
assignee:
  - '@claude'
created_date: '2026-07-20 14:21'
updated_date: '2026-07-23 07:40'
labels: [console, ux]
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The rail line 'System: none' is styled identically to the static Provider/Model/Temperature lines (cell-attrs: same bg, no underline/bold/inverse), yet clicking it opens the session 'Edit system prompt' modal (textarea, Name, Save to Library, Clear/Cancel/Apply, 'Applies to this session.'). The Console Settings modal opened by 'Configure' — where a user would look for everything model-related — has no system-prompt field, and no other visible entry point exists on the screen.

**Repro:** 1. Look at rail Model section: 'System: none' renders as inert text (styling identical to labels above). 2. Click it -> 'Edit system prompt' modal opens. 3. Open Configure modal -> no system prompt control anywhere in it.

**Verifier note:** Code-verified and uncovered by any ledger item or backlog task. #console-rail-system-line is a plain Static with no interactive styling (only color rules at _agentic_terminal.tcss:2428-2435, no hover/underline, no tooltip assigned at compose, chat_screen.py:7264-7281) yet a screen-level on_click at chat_screen.py:11282-11285 opens the system-prompt editor; grep confirms ConsoleSettingsModal contains zero system-prompt controls. One correction to the claim of 'only door': a /system composer command also opens the editor (commit 'feat(console): /system + system prompt modal + rail preview') — but that path is equally undiscoverable, so the P2 discoverability grade stands.

**Source:** Console UX expert review 2026-07-20 (finding j5-system-prompt-hidden-door; P2, verdict NEW, high confidence) — full report `Docs/superpowers/qa/console-ux-expert-review-2026-07-17/README.md`, app under review at origin/dev cad9e271d, J5 settings journey. Evidence: `j5-21-rail-model-section.png`, `j5-22-system-none-click.png`, `j5-23-modal-open.png` (regression/P1 captures committed alongside the report; the full 339-capture set is on the review machine).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 An interactive row must look interactive (button styling like the adjacent 'Configure', or a chevron/link treatment), and/or the session settings modal should include or link to the system prompt. Otherwise users conclude the system prompt cannot be changed
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Took the chevron/link-treatment branch of the AC. The clickable
`#console-rail-system-line` (a `Static` with a screen-level `on_click` that opens
the system-prompt editor) now renders with a trailing `▸` affordance — the same
glyph the rail already uses for its other interactive controls — so it visibly
reads as the one actionable row in the Model section instead of inert label text
like the Provider/Model lines above it. Appended in the screen wrapper
`_console_rail_system_line_state` (constant `CONSOLE_RAIL_SYSTEM_EDIT_AFFORDANCE`)
so the pure `build_console_rail_system_line` helper's contract is untouched. The
line is `height: 1` so the extra glyph clips rather than wrapping to a hidden row.

The affordance lives in the text (not only CSS) because the Console pilot harness
loads widget DEFAULT_CSS but not the built `tldw_cli_modular.tcss`, so a
CSS-only treatment would not be regression-testable.

Verified in `Tests/UI/test_console_system_prompt.py`: new
`test_console_rail_system_line_shows_interactive_affordance` asserts the raw line
ends with ` ▸`; the existing content assertions stay focused via a
`_rail_system_line_text` helper that strips the affordance. 24 tests green.
<!-- SECTION:NOTES:END -->
