---
id: TASK-1459
title: >-
  Spike: shared CSS parse cache across UI test app mounts
status: Done
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

- [x] Measured single-mount cost split: parse = 0.12-0.15s of a 0.37-0.54s mount (22-37%), 14 unique blobs re-parsed per app instance
- [x] Cache key includes a fingerprint of the stylesheet's variable tokens in addition to Textual's per-instance key fields
- [x] Full Tests/UI canary + OFF-vs-OFF control: the empty-diff bar is unachievable on this machine (32 flips between two IDENTICAL cache-off runs); cache-attributable diff = none (isolation A/B identical on/off; ON run mid-pack between the two OFF runs) — deviation recorded in notes
- [x] Env-var escape hatch (TLDW_TEST_CSS_CACHE=0); nightly cache-off cycle deferred to the task-1465 CI rework
- [x] Go/no-go decision + measurements recorded in Implementation Notes

## Implementation Plan

1. Probe the mount cost split (as a pytest test — config isolation)
2. Study Stylesheet._parse_rules / set_variables invalidation on installed Textual 8.2.7
3. Global cache in Tests/UI/css_cache.py, key = Textual's per-instance key + a variables fingerprint; lazy install from the root conftest only when textual is already imported
4. Canary Tests/UI ON vs OFF; on non-empty diff, measure the OFF-vs-OFF noise floor before concluding

## Implementation Notes

**GO.** Deterministic per-mount win: 0.385s -> ~0.26s warm (~35%, matching the
measured parse share); 14 cached entries serve every subsequent app instance.

**Gate deviation, recorded:** the AC demanded an EMPTY canary outcome diff. The
ON-vs-OFF canary showed 12 regressed/0 recovered — but a control pair of two
IDENTICAL cache-off runs showed 28 regressed/4 recovered against each other:
this machine's flip noise floor (rotating flaky families + evening load
growth: OFF runs took 13:37 then 23:12 for the same tests) is ~3x the observed
canary diff, and the ON run sits between the two OFF runs on both failures and
wall time. Cache-attributability was separately falsified by an isolation A/B
of all 12 flagged tests: identical outcomes with cache on and off. The shipped
gate is therefore "cache diff bounded by the measured noise floor + no
deterministic cache-attributable failure", not the unfalsifiable empty diff.

Design: process-global dict in front of Textual's per-instance LRUCache(64);
key extends Textual's (css, read_from, is_default_rules, tie_breaker, scope)
with tuple(sorted(self._variables.items())) — per-instance safety relies on
set_variables() clearing the instance cache, a guarantee that does not span
instances. Shared RuleSet lists mirror Textual's own intra-instance aliasing.
Install is lazy (root-conftest autouse fixture checks sys.modules for
textual.css.stylesheet) so non-UI sessions pay nothing; TLDW_TEST_CSS_CACHE=0
disables. Added: Tests/UI/css_cache.py. Modified: Tests/conftest.py.
