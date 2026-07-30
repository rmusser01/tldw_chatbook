---
id: TASK-1459
title: >-
  Spike: shared CSS parse cache across UI test app mounts
status: To Do
assignee: []
created_date: '2026-07-30 08:55'
labels:
  - testing
  - performance
  - spike
priority: high
dependencies: [task-1457]
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Every `TldwCli` mount in tests re-parses the 14,616-line `css/tldw_cli_modular.tcss` (~1,500+ parses/run counting harness apps that set the same CSS_PATH). Textual 8.2.7 caches parses per-Stylesheet-instance only (`Stylesheet._parse_rules`, LRUCache(64)), and its cache key omits the variables token set (safe per-instance, unsafe globally). A test-only session cache could remove nearly all of that cost — but sharing parsed RuleSets across App instances assumes post-parse immutability, which is unproven. This is a spike with a go/no-go gate, not a committed win.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

- [ ] Measured single-mount cost split (CSS parse vs rest) recorded before any caching
- [ ] Cache key includes a fingerprint of the stylesheet's variable tokens in addition to Textual's per-instance key fields
- [ ] Full Tests/UI canary run with the cache on: junit outcome diff vs baseline is EMPTY (any diff = fall back to deepcopy-on-hit variant or no-go)
- [ ] Env-var escape hatch disables the cache; nightly serial CI run stays cache-off for one cycle
- [ ] Go/no-go decision + measurements recorded in Implementation Notes
