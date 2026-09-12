---
id: TASK-591
title: Clean up or wire dead Subscriptions/security.py validator
status: Done
assignee:
  - '@zcode'
created_date: '2026-07-23 12:00'
labels: [subscriptions, cleanup]
dependencies: [task-328]
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`Subscriptions/security.py` `SecurityValidator.validate_feed_url`/`sanitize_item` and `SSRFProtector` are now policy-correct (delegate to egress) but have ZERO live callers. Either wire `sanitize_item` into the item-ingestion path (post-fetch item-URL validation) or DELETE the dead code. Live fetch-layer SSRF protection is already in place via `guarded_fetch_httpx_async` from TASK-328.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [x] Either wire `sanitize_item` into the active item-ingestion path with live tests OR delete `SecurityValidator` and `SSRFProtector` as dead code
- [x] No live functionality regresses
<!-- AC:END -->

## Implementation Plan

1. Re-verify the July claim on current dev: enumerate every reference to `SecurityValidator`/`SSRFProtector` (production, tests, docs) and each symbol in `Subscriptions/security.py`.
2. Choose the AC branch: wire `sanitize_item` into ingestion (new surface) vs delete dead code. Prefer delete if zero production invocations.
3. Delete the dead classes + their exception types + dead plumbing (`monitoring_engine` param, `__init__` re-exports, defusedxml import), keeping live symbols (`SecurityError`, `CredentialEncryptor`, `InputValidator`).
4. Update the tests that pinned the dead surface; regenerate the diagnostic inventory with the sanctioned script after reviewing the delta.

ADR required: no
ADR path: N/A
Reason: Dead-code removal; the live SSRF boundary (egress guard) is unchanged and already governed by existing decisions.

## Implementation Notes

Chose the DELETE branch. Verified on current dev: `SecurityValidator` and `SSRFProtector` have zero production invocations -- `monitoring_engine` imported the former only as a never-passed, never-read constructor param (`self.security_validator` assigned once, never used); every other reference was tests or docs. Live-fetch SSRF protection stays where TASK-328 put it: `Utils/egress.guarded_fetch_*`.

Deleted: `SecurityValidator`, `SSRFProtector`, `SSRFError`, `XXEError` (unreferenced), the defusedxml import block (served only `validate_xml_content`), the dead `monitoring_engine` param (all `FeedMonitor()` callers are argless), and the two `__init__` re-exports. Kept: `SecurityError` (live: `watchlist_failure`), `CredentialEncryptor` (live: `Web_Server/artifact_share_manifest`), `InputValidator` (not named by this task's AC; noted as a separate dead-code candidate).

Tests: deleted `test_subscription_egress_wiring.py` (pinned the delegation of the deleted class). Rewired `test_subscription_security_config_contract.py`: removed the validator AST helper, its special-cased scan path, and three validator-bound tests; added `test_subscription_security_declares_no_scheme_policy` asserting neither the classes nor scheme-policy tables can return. `test_watchlist_failure.py` swaps `SSRFError` for its live parent `SecurityError` (same classification case). Diagnostic inventory regenerated via `scripts/check_persistent_diagnostic_inventory.py --write` after reviewing the delta: security.py 4 -> 2 diagnostic calls -- both removed calls lived inside deleted dead code (`sanitize_item`'s warning, the defusedxml fallback warning).

Verification: imports of `tldw_chatbook.Subscriptions` and `artifact_share_manifest` OK; targeted runs 120 passed (contract + watchlist failure + off-loop); inventory gate 69 passed; full `Tests/Subscriptions/` + egress allowlist + inventory run shows the only new-vs-dev failure was the inventory pin, green after regeneration (19 other failures reproduce identically on clean dev -- flaky/env set). Ruff deltas all <= 0 (security.py improved by 4).

Modified: `tldw_chatbook/Subscriptions/security.py`, `monitoring_engine.py`, `__init__.py`, `Tests/Subscriptions/test_subscription_security_config_contract.py`, `Tests/Subscriptions/test_watchlist_failure.py`, `Docs/security/production-diagnostic-inventory.json`. Deleted: `Tests/Subscriptions/test_subscription_egress_wiring.py`.