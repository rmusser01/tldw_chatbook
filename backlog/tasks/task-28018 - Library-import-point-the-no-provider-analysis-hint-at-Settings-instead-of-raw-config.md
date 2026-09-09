---
id: TASK-28018
title: >-
  Library import - point the no-provider analysis hint at Settings instead of
  raw config
status: To Do
assignee: []
created_date: '2026-09-02 04:11'
labels:
  - library
  - media-ux
dependencies: []
references:
  - >-
    .impeccable/critique/2026-09-02T04-00-36Z__tldw-chatbook-ui-screens-library-screen-py.md
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The pre-Start hint reads: Set provider under [analysis_defaults] in your config - instructing hand-editing TOML in an app with a Settings screen. First-timers will not find it. Point at the concrete in-app Settings location, ideally with a direct navigation affordance.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The hint names the in-app Settings location for configuring the analysis provider
- [ ] #2 Wording is verified against the actual Settings IA
<!-- AC:END -->
