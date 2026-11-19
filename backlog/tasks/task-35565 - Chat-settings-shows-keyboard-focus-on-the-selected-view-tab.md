---
id: TASK-35565
title: Chat settings shows keyboard focus on the selected view tab
status: To Do
created_date: 2026-10-18 17:42
dependencies: []
labels:
- model-config-redesign
- follow-up
- console
priority: low
references:
- tldw_chatbook/css/features/_console_panels.tcss
- Tests/UI/test_console_settings_geometry.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in Phase 9's live capture phase (2026-10-18) and confirmed in triage to predate Phase 9: it reproduces at the phase base 111f8e48fc.

When the selected view tab in Chat settings (for example "● Model and generation") holds keyboard focus, it paints exactly as it does at rest, so focus cannot be seen. The rule `.console-settings-view-tab.console-settings-view-active` (specificity 0,2,0) beats `Button:focus` (0,1,1). Its `text-style: bold` drops the focus underline, and its colours already match the focus colours. Moving focus to another tab shows the underline there.

Live evidence: Phase 9 capture 01g, and the triage's base capture 01g-base.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The selected view tab shows a visible focus style while it holds focus, and the unfocused selected tab does not, at 211x44 and 235x52
- [ ] #2 The ADR-097 boot CSS budget still holds
- [ ] #3 A test reads the painted style and pins the difference
<!-- AC:END -->
