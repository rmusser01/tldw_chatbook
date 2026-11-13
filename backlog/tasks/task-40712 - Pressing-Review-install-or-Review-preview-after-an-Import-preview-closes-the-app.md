---
id: TASK-40712
title: "Pressing Review install… or Review preview… after an Import… preview closes the app"
status: To Do
assignee: []
created_date: 2026-11-13 04:43
labels:
- follow-up
- plugins
- bug
priority: high
references:
- tldw_chatbook/Widgets/plugin_review_modal.py
- tldw_chatbook/UI/Screens/plugins_screen.py
- tldw_chatbook/app_lifecycle.py
- Docs/User_Guide/plugins.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Filed in round 14 of the 2026-10-23 burn-down from the TASK-40332 docs stream, and confirmed by its reviewer. Checked on dev 26b2e677dd, 2026-11-12.

After an **Import…** preview, pressing **Review install…** or **Review preview…** opens the review modal. Its load worker reads `self.producer.inspection` (`Widgets/plugin_review_modal.py:823`), but an `ImportPreview` has no `inspection`. The `AttributeError` fails the worker, and a worker failure exits the app (`app_lifecycle.py:891`). The user loses the session over a button press. The Plugins User Guide page currently warns users away from it.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 After an Import… preview, Review install… and Review preview… either open a working review of the imported copy or are disabled with a stated reason; neither raises; a test drives both buttons
- [ ] #2 A failure while the plugin review modal loads shows an error in the modal and leaves the app running; a test forces a load failure and pins it
- [ ] #3 The Plugins User Guide page's warning about this crash is removed or replaced by the new behaviour
<!-- AC:END -->
