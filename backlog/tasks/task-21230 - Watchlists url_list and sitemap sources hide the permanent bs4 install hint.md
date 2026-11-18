---
id: TASK-21230
title: >-
  Watchlists url_list and sitemap sources hide the permanent bs4 install hint
status: Done
assignee: []
created_date: '2026-08-23'
labels:
  - bug
  - watchlists
  - subscriptions
  - error-handling
dependencies: []
priority: medium
---

## Description

Source: close-out of the 2026-08-22 holistic performance review burn-down (queue TASK-21100
– TASK-21134). Found by the TASK-21104 adversarial review and its implementation pass.
Base evidence doc: `Docs/Design/2026-08-22-holistic-perf-review.md`.

TASK-21104 made `beautifulsoup4` a guarded import so a base install can still boot, and the
guard reports a missing bs4 with an install hint. In
`Subscriptions/local_watchlists_service.py` the task-1394 per-URL privacy suppression used by
the `url_list` and `sitemap` loops (`_check_url_isolated`) reduces ANY exception — including a
**permanent** `ImportError` — to a type-only DEBUG line plus a `DISPOSITION_ERROR` carrying
`reason=None`. A user whose watchlist uses a `url_list` or `sitemap` source therefore sees an
unexplained error with no reason text and never learns to install the extra. The single-`url`
arm, which does not go through the suppression, surfaces the hint verbatim. The hint string is
a module constant containing no user data, so surfacing it does not conflict with the privacy
rule the suppression exists for.

Second, pre-existing hygiene in the same package, found during that pass and re-confirmed on
dev `b2b1e2e0d`: `Subscriptions/monitoring_engine.py` imports `from loguru import logger`
twice (lines 27 and 37).

## Acceptance Criteria

- [ ] A `url_list` or `sitemap` watchlist source that fails because `beautifulsoup4` is not installed surfaces the same install hint a single-`url` source surfaces
- [ ] The `DISPOSITION_ERROR` produced for that case carries a non-null reason
- [ ] Transient and network-shaped per-URL failures still carry no URL, no exception message and no user data — the task-1394 suppression is unchanged for them
- [ ] `Subscriptions/monitoring_engine.py` imports `logger` exactly once
- [ ] A test fails if a permanent ImportError is re-suppressed to `reason=None` on the isolated arm

## Implementation Plan

1. Premise-verify the suppression path and the single-`url` arm against dev
   `a3c47c6c72` (line cites drifted; symbols stable).
2. In `_check_url_isolated`, distinguish a PERMANENT ImportError whose message
   is exactly the TASK-21104 module-constant hint: carry the constant as the
   disposition's `reason`; every other exception keeps `reason=None`.
3. Register the new `(DISPOSITION_ERROR, _BS4_INSTALL_HINT)` pair in
   `_disposition_count_keys` (counts into the same "error" counter -- without
   this the new pair KeyErrors `_disposition_counts` mid-run).
4. Surface the shared permanent reason verbatim as the all-error run's
   `error_msg`/`log_text`, ahead of the failure-classification aggregate.
5. Remove the duplicate `from loguru import logger` in `monitoring_engine.py`.
6. Tests: executor-level drive (monkeypatched URLMonitor, no app-config read)
   for the hint case, the transient case, and the logger-count pin.

## Implementation Notes (2026-10-11, dev `a3c47c6c72`)

- **Premise confirmed**: the isolated arm reduced the bs4 ImportError to
  `reason=None`; `monitoring_engine.py` imported `logger` twice.
- **Fix** in `Subscriptions/local_watchlists_service.py`: `_check_url_isolated`
  now sets `reason` to `_BS4_INSTALL_HINT` only when the raised ImportError's
  message equals that module constant (identity-matched, never a free-form
  exception message, so the task-1394 privacy rule is untouched for
  transient/network failures); `_disposition_count_keys` registers the new
  pair into the "error" counter; the all-error path surfaces the shared
  permanent reason verbatim as `error_msg`/`log_text` -- the same text the
  single-`url` arm surfaces through `record_run_failure`. Duplicate logger
  import removed from `Subscriptions/monitoring_engine.py`.
- **Tests** (all green): `test_url_list_surfaces_bs4_install_hint_on_permanent_import_error`
  (reds if the ImportError is re-suppressed to reason=None -- without the fix
  `error_msg` falls to the classification aggregate/legacy message);
  `test_url_list_transient_failures_still_carry_no_reason` (privacy pin:
  no exception text or query string anywhere user-visible; unit pin that the
  transient disposition pair stays reason-less); `test_monitoring_engine_imports_logger_exactly_once`.
  Tests drive `_default_run_executor`/`_check_url_isolated` directly because
  the `create_source` DB-write path reads app config and fails closed with
  the documented local shared-profile `RecoveryRequired` caveat (Tests/UI
  lesson; the pre-existing task-1394 flow tests are locally-red for the same
  reason -- verified identical at HEAD by checkout swap -- and pass in CI).
