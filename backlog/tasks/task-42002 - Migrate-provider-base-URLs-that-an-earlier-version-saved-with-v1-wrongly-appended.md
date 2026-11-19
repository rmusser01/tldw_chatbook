---
id: TASK-42002
title: "Migrate provider base URLs that an earlier version saved with /v1 wrongly appended"
status: To Do
assignee: []
created_date: 2026-11-19 19:39
updated_date: 2026-11-19 19:39
labels:
- follow-up
- providers
- settings
priority: medium
references:
- tldw_chatbook/Provider_Contract/provider_endpoint_contract.py
- tldw_chatbook/config.py
- Docs/User_Guide/settings.md
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by TASK-41704 (round 17). Before TASK-41704, Settings appended `/v1` to a first-party cloud base with no `v1`, so configs may hold known-broken values such as Google `.../v1beta/v1` or `https://api.cohere.com/v1`; TASK-41704 only stops new saves. Owner decision, 2026-11-19: "Migrate on load": when the app loads one of these known-broken shapes for a first-party provider, it rewrites it to the documented base. This changes those configs' project-instruction keys once.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Each known-broken shape TASK-41704's old rule produced is rewritten to the provider's documented base when config loads, and the rewrite is saved once
- [ ] #2 A value that is not one of those shapes, including any gateway or custom base, is never rewritten
- [ ] #3 The rewrite is logged without the URL's credentials or query, and the notes list the shapes covered
- [ ] #4 Tests cover each shape and a non-matching control, red before the change
<!-- AC:END -->
