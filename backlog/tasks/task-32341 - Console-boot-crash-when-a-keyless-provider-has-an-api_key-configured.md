---
id: TASK-32341
title: Console boot crash when a keyless provider has an api_key configured
status: To Do
assignee: []
created_date: '2026-09-10 19:09'
labels:
  - console
  - provider-readiness
  - bug
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The Console readiness validator raises 'Console credential source conflicts with its facet' when the credential facet is not_required but a stored key exists, and the app exits before rendering anything. Reproduced with [api_settings.custom] api_key on the keyless custom provider; introduced with the Console conversation-settings redesign (939dee8dc2, 2026-09-04). Keyless OpenAI-compatible servers commonly still accept a bearer token, so this configuration is legitimate. Source: live approval-card / MCP-permissions UX review 2026-09-10 against dev 3315241674 (snapshot .impeccable/critique/2026-09-10T17-31-53Z__hatbook-widgets-chat-widgets-chat-approval-card-py.md).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The app starts and Console reaches Ready when a keyless provider has an api_key configured.
- [ ] #2 Readiness reports the credential facet and its source consistently for keyless providers with and without a stored key, without raising.
- [ ] #3 A regression test covers a keyless provider with a stored key.
<!-- AC:END -->

## Renumbering provenance

<!-- SECTION:PROVENANCE:BEGIN -->
This task was filed as TASK-32272 on 2026-09-10 at 12:19 PT (19:19 UTC) in the
approval-card / MCP-permissions UX review wave (PR #2574), which makes it the
OLDER arrival against the TASK-32272 that `dev` minted later the same day
("Library Notes select mode shows two selection counters that disagree").

It renumbered to TASK-32341 anyway. The bare 2026-08-21 owner rule of TASK-19601
quoted in `scripts/check_backlog_task_ids.py` (older arrival keeps the id) is
superseded by the 2026-09-08 refinement recorded in
`backlog/docs/lessons-backlog-hygiene.md`: **landed-keeps-id trumps
older-keeps-id -- a task already on origin/dev never renumbers; the unmerged
side moves regardless of timestamps, because renumbering landed ids breaks
external references.** The dev-side TASK-32272 is merged; this task was cited
only from its own unmerged plan. `dev` applied the same refinement earlier on
2026-09-10 when it renumbered its archive-lifecycle task to TASK-32300.

Renumbered 2026-09-10. Commit messages on the wave branches
`approval-wave-a/b/c` written before this date that cite `task-32272` refer to
THIS task; the dev-side TASK-32272 keeps the id.
<!-- SECTION:PROVENANCE:END -->
