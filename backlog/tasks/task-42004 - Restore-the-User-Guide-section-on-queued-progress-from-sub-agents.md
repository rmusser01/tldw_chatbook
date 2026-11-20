---
id: TASK-42004
title: "Restore the User Guide section on queued progress from sub-agents"
status: To Do
assignee: []
created_date: 2026-11-19 19:39
updated_date: 2026-11-19 19:39
labels:
- follow-up
- docs
- console
priority: low
references:
- Docs/User_Guide/console/agent-runs-and-tools.md
- tldw_chatbook/Agents/fleet_message_tools.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by review r17dc-rev1 (round 17). Squash 6a6242ce3a removed "Queued progress from sub-agents" (`report_to_supervisor`, Agent → Progress: N queued, Discard selected), added by 827b622638. The feature still ships (`Agents/fleet_message_tools.py`, the `console_agent_progress_modal.py` Queued progress view). The newer "Queued child progress" section describes the same queue in its saved-chat form.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The section is restored and reconciled with "Queued child progress", every sentence checked against current code and cited in the notes
<!-- AC:END -->
