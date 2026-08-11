---
id: TASK-14910
title: >-
  Server-mode ingest forecast still claims 0-byte files will fail without
  knowing
status: To Do
assignee: []
created_date: '2026-08-11 01:59'
labels:
  - library
  - ingest
  - server
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found while closing task-14827 (the server-mode forecast/receipt divergence for refused files), and deliberately left out of that task's scope.

build_ingest_forecast counts every 0-byte staged file as a certain failure on BOTH backends. On the local path that is verified: run_parse_job refuses an empty source before any write, and the local governance test asserts it. On the server path nothing verifies it -- _submit_server_ingest_job builds kwargs for the empty file and sends it, so the outcome belongs to the server, which this process cannot inspect (the same reason the forecast refuses to claim anything about server tooling).

So the server forecast makes exactly one claim it has not earned. Either the client should refuse to send a 0-byte file and fail it locally with the reason it already knows -- making the claim true by construction on both backends -- or the server forecast should stop counting empty files. task-14827's server governance test deliberately holds no empty file for this reason, and says so, which is why the gap is written down here rather than papered over.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A 0-byte file staged for a SERVER import has one outcome the forecast and the receipt agree on
- [ ] #2 The server governance test in Tests/integration/test_library_ingest_flow.py covers a 0-byte file without any stubbed server behaviour deciding its fate
<!-- AC:END -->
