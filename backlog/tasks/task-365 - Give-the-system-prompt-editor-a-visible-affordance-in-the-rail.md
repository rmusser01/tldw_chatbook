---
id: TASK-365
title: Give the system-prompt editor a visible affordance in the rail
status: To Do
assignee: []
created_date: '2026-07-20 14:21'
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
- [ ] #1 An interactive row must look interactive (button styling like the adjacent 'Configure', or a chevron/link treatment), and/or the session settings modal should include or link to the system prompt. Otherwise users conclude the system prompt cannot be changed
<!-- AC:END -->
