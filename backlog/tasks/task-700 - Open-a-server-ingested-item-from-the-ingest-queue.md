---
id: TASK-700
title: Open a server-ingested item from the ingest queue
status: To Do
assignee: []
created_date: '2026-07-26 13:58'
labels:
  - library
  - ingest
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
A server ingest finishes with its content in the server's library, not this machine's, so the queue row's 'Open in Library' stays withheld. The server does report the id of the row it created, so the item is addressable -- there is just no affordance that opens it in the server-scoped Library view. Users who import on the server currently have no route from a finished job to the thing it produced.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A finished server-origin job offers a way to view the item it created,The action opens the item in a server-scoped view rather than looking for a local row,A job whose server result carries no usable id offers no such action
<!-- AC:END -->
