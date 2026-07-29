# Briefing subsystem: revive-or-retire audit

**Date:** 2026-07-27
**Baseline:** `origin/dev` @ `9a9e9fea5`
**Question:** Watchlists spec #2 (briefings + 2-speaker podcasts + scheduled delivery) assumed
it would build on the existing briefing machinery in `Subscriptions/`. Is that machinery live,
and can it be revived?

**Answer: retire it.** The briefing machinery is bolted to a *retired scheduler*, not to the one
that actually runs. Reviving the briefing path would mean reviving the dead scheduler with it.

## Method

Static greps were run first and gave a misleading answer twice, so the conclusion here rests on a
runtime import trace: import `tldw_chatbook.app` under a throwaway `TLDW_CONFIG_PATH` profile and
inspect `sys.modules`. That distinguishes *a module that is imported* from *a class that is
constructed* — the distinction the greps kept blurring.

Two of this programme's own prior claims were wrong and are corrected below.

## Findings

### 1. There are two parallel watchlist-checking implementations

| | Retired | Live |
|---|---|---|
| Entry point | `Subscriptions/textual_scheduler_worker.py` | `Scheduling/scheduler/handlers/watchlist_check_handler.py` |
| Size | 492 LOC | 159 LOC |
| Monitors via | `Subscriptions/website_monitor.py` | `Subscriptions/monitoring_engine.py` (`FeedMonitor`, `URLMonitor`) |
| In `sys.modules` after `import tldw_chatbook.app` | **no** | **yes** |
| Briefing step | yes | **none** |
| Shadow mode | no | yes |

The live handler is stateless, shadow-mode aware, and delegates to `monitoring_engine`. It has no
briefing or aggregation step at all. The briefing machinery hangs entirely off the retired side.

### 2. `BriefingGenerator` is imported but never constructed

`Subscriptions/briefing_generator` **is** in `sys.modules` at startup — but only because
`Subscriptions/__init__.py:65` eagerly imports it inside a `try:` block. That is a module-level
side effect, not a call path: it costs import time on every launch and executes nothing.

The class has exactly one construction site, `textual_scheduler_worker.py:101`:

```python
self.briefing_generator = BriefingGenerator(db) if BRIEFING_AVAILABLE else None
```

`textual_scheduler_worker` is **not** in `sys.modules` after a full app import. So
`BriefingGenerator` is never instantiated in production, and `generate_briefing` is never called.

### 3. The UI controllers that would drive it target a class that no longer exists

`UI/Subscription_Modules/` has **zero** production importers — the only importer anywhere is
`Tests/Subscriptions/test_notifications_inbox_controller.py`. The package is absent from
`sys.modules` after app import.

`SubscriptionBackendController` (the only thing that imports `textual_scheduler_worker`) drives
`self.window`, a `SubscriptionWindow`. **`class SubscriptionWindow` does not exist anywhere in the
codebase.** The window was deleted; its controllers were not.

### 4. Reachability of each module

Sizes are the retirement estimate.

| Module | LOC | Reached by | Status |
|---|---|---|---|
| `briefing_templates.py` | 813 | nothing | dead |
| `briefing_generator.py` | 838 | `Subscriptions/__init__` (import only), `textual_scheduler_worker` | dead |
| `recursive_summarizer.py` | 635 | `aggregation_engine` only | dead (transitive) |
| `aggregation_engine.py` | 612 | `briefing_templates` (dead), `textual_scheduler_worker` (dead) | dead |
| `rss_feed_generator.py` | 605 | `website_monitor` only | dead (transitive) |
| `export_manager.py` | 591 | nothing | dead |
| `distribution_manager.py` | 536 | nothing | dead |
| `textual_scheduler_worker.py` | 492 | `subscription_backend_controller` (dead) | dead |
| `website_monitor.py` | — | `textual_scheduler_worker` only | dead (transitive) |

`monitoring_engine.py` is **live** and stays — it is what the working scheduler calls.

Only one test touches any of this: `Tests/Subscriptions/test_scheduler_deprecation.py`, which
asserts the deprecation warning fires. Nothing tests the behaviour.

### 5. ADR-019's dual-run safety net does not exist in the shipped app

ADR-019 (Accepted, 2026-07-19) describes a shadow-mode dual-run: the old scheduler stays "the
execution authority by default", the new handler shadows it, and a runtime toggle allows "instant
rollback to the old scheduler". None of that is true of the code on `origin/dev`:

- The old scheduler is **unreachable** (finding 3), so it cannot be the execution authority and the
  documented rollback — "set the flag false and `SubscriptionScheduler` resumes authoritative
  execution" — cannot happen. Setting the flag false leaves *no* executor.
- `[scheduling] watchlist_checks_enabled` ships **`false`** (`config.py:2296`, and the `app.py:4666`
  default). With it false, `app.py:4674` never constructs the handler, `"watchlist_job"` is never
  registered in the `handlers` dict, and `watchlist_projection` is passed to `SchedulerLoop` as
  `None`. The new path is off entirely — not even shadowing.
- `watchlist_checks_shadow` ships **`true`**. So even after enabling checks, the handler fetches
  and then discards: `record_check_result` is skipped and `URLMonitor` is built with
  `persist_snapshots=False` (`watchlist_check_handler.py:55-59, 107`).

**Net effect: nothing checks watchlists on a schedule.** Getting working scheduled checks requires
setting *both* flags, and neither is reachable from the Watchlists UI. Manual "Check now" works —
that is the `launch_run` path fixed on 2026-07-27 — but there is no automatic checking at all.

To be precise about scope: the Watchlists UI does not currently expose `check_frequency` either
(it appears only in `local_watchlists_service.py:665`), so no user is setting a schedule that
silently fails. The accurate statement is that **automatic checking is unimplemented end to end**,
not that it is broken.

This matters for spec #2 more than the briefing question does. Spec #2's headline is *scheduled*
delivery; there is no working schedule to deliver on yet.

## Corrections to earlier claims

- **"~4,600 LOC with zero importers outside the package"** — wrong on the mechanism. There *are*
  importers; they form a closed cycle (`textual_scheduler_worker` ↔ `subscription_backend_controller`)
  with no external entry. A naive grep sees four importers and concludes the code is live. That is
  presumably how it survived this long.
- **"`watchlists_collections_screen.py:38` imports `notifications_inbox_controller`"** — wrong.
  The Watchlists screen does not import from `Subscription_Modules` at all.

## Consequence for spec #2

The spec has a prerequisite it did not know about. Ordering:

1. **Make scheduled checks actually run** (finding 5) — promote the handler out of shadow, decide
   the default, and give the UI a way to set a check frequency. Without this there is nothing to
   attach a briefing to.
2. **Retire the island** (findings 1-4) so the briefing work is not built beside a second, dead
   implementation of the same thing.
3. **Then** build briefing/podcast generation as a new handler alongside `watchlist_check_handler`
   in `Scheduling/scheduler/handlers/`, on the live seam.

Two things make step 3 cheaper than it looks:

- The scrape/fetch pipeline underneath actually works as of 2026-07-27 (PR #989 line), so a
  briefing generator would be fed real content for the first time.
- `TTS/audiobook_generator.py` already does `multi_voice` with per-character voices and segment
  concatenation — most of the 2-speaker podcast path exists and is live.

## Reproducing

```bash
SCRATCH=$(mktemp -d)
printf '[general]\nusers_name = "audit_probe"\n' > "$SCRATCH/config.toml"
TLDW_CONFIG_PATH="$SCRATCH/config.toml" .venv/bin/python -c "
import sys, tldw_chatbook.app
print(sorted(m for m in sys.modules if 'briefing' in m or 'scheduler' in m or 'Subscription_Modules' in m))"
rm -rf ~/.local/share/tldw_cli/audit_probe "$SCRATCH"
```

Expected: `briefing_generator` present, `textual_scheduler_worker` absent,
`Subscription_Modules` absent, `Scheduling.scheduler.handlers.watchlist_check_handler` present.
