---
id: TASK-18908
title: Fix --serve --port being ignored by the python -m entry point
status: Done
assignee:
  - '@Robert'
created_date: '2026-08-19 22:50'
updated_date: '2026-08-19 23:50'
labels:
  - serve
  - bug
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
python -m tldw_chatbook.app --serve --port N silently ignores the port: the __main__ block parses args (including the new --focus flag from task-18812) but never routes them into run_web_server, so serve mode always binds the config default (8000). The tldw-serve console-script entrypoint handles host/port/title/debug correctly. Phone/server setups over --serve need a stable custom port.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 python -m tldw_chatbook.app --serve --port 8765 serves on 8765 (verified live via HTTP 200)
- [x] #2 --host/--web-title/--debug also route through on the __main__ path
- [x] #3 argparse behavior of both entrypoints stays identical (--help exits 0, bad flags exit 2, --focus still routes)
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Fixed via PR #1834 (merged d-series): __main__ serve branch now routes --host/--port/--web-title/--debug into run_web_server, mirroring main_cli_runner. Verified live: --serve --port 8765 binds 8765 (HTTP 200, log 'Starting web server at http://localhost:8765'); argparse exits unchanged (--help 0, bad flag 2). Tests: Tests/UI/test_serve_main_args.py (parser flags + runpy-driven routing with run_web_server patched). CI: 2 shards red on the pre-existing dev breakage documented in PR #1829's provenance comment; this PR's suites green. ADR: not required (bug fix, mechanical).
<!-- SECTION:NOTES:END -->
