---
id: TASK-41604
title: "Trace v2 import validation errors echo raw structural values from the file before any masking"
status: To Do
assignee: []
created_date: 2026-11-17 19:44
labels:
- follow-up
- security
- trace
priority: medium
references:
- tldw_chatbook/Chat/trajectory_import.py
- tldw_chatbook/Chat/trajectory_export.py
- tldw_chatbook/UI/Screens/trajectory_screen.py
dependencies:
- TASK-41133
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by the round-15 trace review (TASK-41133, 2026-11-17). The problem predates round 15. It covers structural fields only; message content is not echoed.

**Where values are echoed.** `_validate_v2` (`trajectory_import.py:290`) builds its `TrajectoryImportError` messages from values read straight out of the file:
- `version` (`:298`, `:307`)
- the manifest `profile` (`:315`)
- `privacy_inventory` keys (`:392`)
- a duplicate `event_id` (`:417`)
- unknown `usage` keys (`:495`)
- `field_provenance` keys (`:556`, `:562`)

All of these are raised before the credential check at `:939`. A value shaped like a credential reaches the user unmasked.

**Where the message goes.**
- The error reaches `app.notify(str(exc), ...)` (`trajectory_screen.py:2036`) with markup parsing on. A raw value containing Rich markup such as `[/]` can therefore also raise `MarkupError`. The CLAUDE.md convention for notify text that interpolates runtime values is `markup=False`.
- A file whose `format` is not `tldw-trace` falls to the v1 validator, which echoes `format` and `version` the same way (`trajectory_export.py:1476`, `:1482`, `:1493`).

**Related.** TASK-41603 covers v1 content masking; TASK-41510 and TASK-41133 cover the masking rules.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 No import validation error shows a raw value from the file unmasked: each echoed structural value is masked by the current credential rules, or replaced by its position (for example `events[3]`)
- [ ] #2 The import failure notice renders a value containing Rich markup literally, without raising
- [ ] #3 Tests feed a credential-shaped `profile`, `event_id` and `field_provenance` key, and a markup-bearing value, and assert the shown message; red before the fix
<!-- AC:END -->
