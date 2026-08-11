---
id: TASK-15463
title: Watchlists backend: one SubscriptionsDB instance and scheduler work off the event loop
status: To Do
assignee: []
created_date: '2026-08-11 12:05'
labels:
  - perf
  - watchlists
  - db
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Probe-verified in the audit: `LocalWatchlistsService._db()` returns `self.db_factory()` (`Subscriptions/local_watchlists_service.py:320-321`; wired `app.py:5984-5988`), so nearly every service method constructs a fresh `SubscriptionsDB` — a ~52-statement `executescript` plus migration probes per call. Measured: 3.4 ms per reconstruction on an EMPTY DB vs 0.04 ms for the same query on a held connection (~85×); first construction 35 ms; a Watchlists screen refresh fires 5+ loads. The class is already thread-safe (`threading.local`, `DB/Subscriptions_DB.py:1133-1135`). Separately, the scheduler's due checks run sync sqlite bookkeeping (`local_watchlists_service.py:802-897`) and full XML/JSON feed parsing (`ET.fromstring`, `Subscriptions/monitoring_engine.py:851-864`) inline on the event loop — enabled by default, firing on any tab (the fetch itself is async httpx and fine; queue bookkeeping is already to_thread).

Fix direction: build one SubscriptionsDB at wiring time and have the factory return the cached instance (keep a factory seam for tests); move the check handler's DB work and feed parse to a thread. Stability constraint: run accounting/receipts have task-2305 history — preserve run records and due-detection semantics exactly. Evidence and method: Docs/Design/2026-08-11-input-latency-audit.md (audit of dev 82b595049; all file:line cites verified there).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Exactly one SubscriptionsDB is constructed per app session in production wiring (evidence); tests keep an injectable factory
- [ ] #2 No synchronous sqlite or feed parsing runs on the event loop during a due check (evidence)
- [ ] #3 Scheduler behavior — due detection, run records, item upserts, briefings — unchanged (existing surface green)
<!-- AC:END -->
