---
id: TASK-186
title: Console/Library micro-polish batch from core-loop UAT 2026-07
status: To Do
assignee: []
created_date: '2026-07-12 02:48'
labels:
  - ux
  - polish
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Small defects from the 2026-07-11 UAT (evidence in Docs/superpowers/qa/core-loop-uat-2026-07): Model rail line reads 'llama_cpp / ' with a trailing slash and no model name while the top chip shows it; two left-rail sections are both titled Context; the footer shows Palette Menu twice; Enter in a valid ingest path field does not start the ingest; the media summary action is labeled 'Open in Media manager' but opens the in-Library viewer; save-as-note produces generic 'Console message' titles with no source context; splash animation logs NameError ESCAPED_OPEN_BRACKET every run; legacy selector errors (#chat-api-provider, #app-log-display) log on every boot.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Model rail shows the selected model or omits the empty slash,Rail section titles are unique,Footer shows the palette hint once,Enter submits a valid ingest form,Media open action label matches its in-Library destination,Saved-message notes carry conversation title/date context,Splash NameError and recurring legacy-selector errors are fixed or silenced
<!-- AC:END -->
