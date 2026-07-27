---
id: TASK-1120
title: >-
  Selecting an item shows "Type: source" and offers source actions
status: To Do
assignee: []
created_date: '2026-07-28 10:30'
labels:
  - watchlists
  - bug
  - ui
  - uat
priority: high
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Clicking a row in the **Items** table selects it, and the Inspector names it correctly — but classifies it as a source:

```
Selected: Lightsail object storage concerns - Part 2
Type: source
           Preview
          Check now
```

`Preview` and `Check now` are *source* actions. The item actions the Inspector is built to offer — `Mark reviewed`, `Ingest`, `Ignore` — never appear, so an item cannot be acted on at all.

Observed with real scraped content on `origin/dev` `79152bbb6`: 10 items fetched from `https://summitroute.com/blog/feed.xml`, clean profile.

`InspectorPane._entity_type` decides this, and the entity reaching it evidently carries the shape of a source rather than an item. Worth checking what `ItemSelected` puts on the wire versus what `SourceSelected` does, and whether the Items table's rows are being routed through the sources selection path — the Sources table's own selection defect (task-1105) suggests these tables share more wiring than is obvious.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Selecting an item reports `Type: item`
- [ ] #2 The Inspector offers `Mark reviewed`, `Ingest` and `Ignore` for a selected item
- [ ] #3 Those actions change the item's status, verified against the database
- [ ] #4 Source, run, rule and notification selections still report their own types
- [ ] #5 A test selects an item and asserts the reported type and offered actions, proven to fail against current code
<!-- AC:END -->
