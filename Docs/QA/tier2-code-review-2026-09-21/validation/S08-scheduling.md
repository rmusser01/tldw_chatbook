# S08 — `Subscriptions/` + `Scheduling/` validation

Zero commits touched `tldw_chatbook/Subscriptions/` or `tldw_chatbook/Scheduling/` between the
review commit `3722a85748` and `HEAD`. All 9 findings checked directly against current code.
`qa/tier2-code-review-2026-09-21/phase4-verification.md` independently re-traced the P0 (its
"lead re-verification" pass) and reaches the same conclusion — cited below as corroboration, not
as a substitute for direct proof.

## 1. P0 [D1] — a sync while Schedules shows "This device" mirrors server-owned reminders/definitions under `owner_id="local"`, arming double execution
- Verdict: CONFIRMED
- Site now: `Scheduling/services/sync_engine.py:265-267` (`pull`: `target_owner = owner_id if
  owner_id is not None else self.owner_id`); UI trigger
  `UI/Screens/scheduling/schedules_workbench.py:5395-5408` (`action_sync_now` gates on
  `_server_available` reachability only, then `_run_sync` at `:5399-5409` reads `owner_id =
  service.owner_id` and calls `service.sync_now(owner_id)` unconditionally)
- Proof: read `scheduling_service.py:337-338` — `self.owner_id = runtime_source` (a
  UI-togglable view, not the connected server's identity). Read `scheduling_service.py:1162-1180`
  `_active_server_owner_id()` — its own docstring states the exact rule the finding relies on:
  "`self.owner_id` is a UI-togglable VIEW ... it cannot stand in for 'which server this session is
  connected to'" — and this helper is never threaded into `pull`/`sync_now`. Read
  `schedules_workbench.py:4870-4887 _on_owner_local`/`_set_owner` confirms the view can be flipped
  to `"local"` while the server stays connected. `phase4-verification.md:392-421` independently
  re-traced the same chain (pull → `_apply_pulled_reminders(conn, target_owner, …)` → queue
  `is_server_scoped_owner` keys on the `server:` prefix, which the pull never applies) and reached
  the same CONFIRMED verdict.
- Note: severity is correctly P0 — ADR-077 (`backlog/decisions/077-server-offloaded-scheduled-
  agent-tasks.md`, Status: Accepted) explicitly names and rejects exactly this shape ("Both sides
  execute, dedupe at delivery ... for agent work this is double execution with nondeterministic
  ordering, and dedupe after the fact cannot un-run side effects" — `:107`), and the consequence
  (a second unattended LLM run per `recurring_question` occurrence) is real user cost, not cosmetic.

## 2. P1 [D1] — watchlist `sitemap` sources seed `trusted_origins` from the discovered URL itself
- Verdict: CONFIRMED
- Site now: `Subscriptions/local_watchlists_service.py:2171` (sitemap arm of
  `_default_run_executor` loops `_urls_for_sitemap()` results into `_check_url_guarded`) →
  `:2421-2470` (`_check_url_isolated`: `monitor.check_url({**subscription_config, "source": url,
  "type": "url"})`, no explicit `trusted_origins`) → `Subscriptions/monitoring_engine.py:1729
  (url = subscription["source"])` → `:1766` (`trusted_origins=origin_set(url)`) — the trust set is
  derived from the very URL being fetched.
- Proof: `python -c "from tldw_chatbook.Utils.egress import evaluate_url_policy, origin_set; ..."`
  → `evaluate_url_policy('http://127.0.0.1:8080/admin', trusted_origins=origin_set(same_url))` →
  `allowed=True reason=ok`; the same URL with `trusted_origins=frozenset()` → `allowed=False
  reason=private`. Exact match to the review's repro.

## 3. P2 [D1] — `v4_to_v5.rollback`/`v5_to_v6.rollback` each stamp a schema version one lower than they leave behind
- Verdict: CONFIRMED
- Site now: `Scheduling/db/migrations/v4_to_v5.py:64-72` (docstring "returning `db` to schema
  version 4"; guard `if current_version >= 4`; `INSERT ... VALUES (3,)`), `v5_to_v6.py:69-77`
  (docstring "...to schema version 5"; guard `>= 5`; `INSERT ... VALUES (4,)`)
- Proof: direct read of both, plus siblings for contrast — `v3_to_v4.py:176` (rollback from v4 to
  v3, correctly stamps `(3,)`) and `v6_to_v7.py:131` (rollback from v7 to v6, correctly stamps
  `(6,)`) both do the arithmetic right, confirming the two flagged files are typos, not a
  convention. `grep -n "v4_to_v5|v5_to_v6" Tests/Scheduling/**` → zero hits — neither `rollback()`
  is tested.

## 4. P2 [D1/D4] — `scheduler_heartbeat.write_heartbeat` claims durability/atomicity, has no `fsync`
- Verdict: CONFIRMED
- Site now: `Scheduling/scheduler_heartbeat.py:2` (module docstring: "TASK-26025: durable
  scheduler liveness heartbeat. A small atomic JSON file..."), `:94-118` (`mkstemp` →
  `fdopen.write` → `os.replace`, no `flush`/`fsync`)
- Proof: `grep -n "fsync" tldw_chatbook/Scheduling/scheduler_heartbeat.py` → zero hits.
  `backlog/tasks/task-32808.5` (Done) Implementation Notes list its census by module name; neither
  `Scheduling/scheduler_heartbeat.py` nor `Subscriptions/briefing_export.py` appears — confirms
  "census incomplete," matching the S07 voiceprint finding's same root cause.

## 5. P2 [D4] — `_effective_max_tokens`/`_invoke_chat` are three constants-only copies
- Verdict: CONFIRMED
- Site now: `Subscriptions/briefing_service.py:570/594`, `Subscriptions/briefing_cast.py:477/496`,
  `Library/library_rag_answer_service.py:481/506`
- Proof: `grep -n "_effective_max_tokens|_invoke_chat" <the three files>` confirms all six
  function definitions exist at the cited lines; `briefing_cast.py:480/506` carry explicit
  "Copies `briefing_service.…`'s exact shape (not imported...)" comments acknowledging the
  duplication.

## 6. P2 [D4] — `Scheduling/` writes a non-canonical UTC timestamp shape invisible to the ADR-173 guard
- Verdict: CONFIRMED
- Site now: `Scheduling/db/scheduled_tasks_db.py:713-722` (`_to_utc_iso` → `.isoformat()`,
  microsecond + `+00:00` shape)
- Proof: `python scripts/check_timestamp_writers.py` → `"0 datetime.utcnow() site(s), 0 naive
  datetime.now().isoformat() occurrence(s) ... OK"` while `_to_utc_iso` is live and non-canonical.
  `grep -rln "Utils\.timestamps" tldw_chatbook/Scheduling tldw_chatbook/Subscriptions` → exactly
  one hit (`Subscriptions/site_config_manager.py`), vs `Utils/timestamps.py`'s 28 repo-wide
  importers — confirms "one adopter in this whole slice."

## 7. P3 [D3] — `baseline_manager._get_latest_baseline` computes and discards a decompression
- Verdict: CONFIRMED
- Site now: `Subscriptions/baseline_manager.py:663-668` — `zlib.decompress(row["raw_html"]).
  decode("utf-8")` and the `except Exception: row["raw_html"]` fallback are both bare, unassigned
  expression statements; `ContentBaseline(...)` below is constructed with no `compressed_content=`
  argument (defaults to `None`)
- Proof: direct read confirms both dead statements. `grep -n "from .*baseline_manager import|
  import baseline_manager" tldw_chatbook/ Tests/` → zero hits (module has no importers, matching
  `task-1360` Done / "Decide the fate of baseline_manager").

## 8. P3 [D3] — `write_heartbeat`'s cleanup path opens `/dev/null` and immediately closes it
- Verdict: CONFIRMED
- Site now: `Scheduling/scheduler_heartbeat.py:114-118` — `except Exception: with
  open(os.devnull, "w"): pass` immediately before the real `os.unlink(tmp)`
- Proof: read confirms the no-op, no comment explaining it. Subsumed by the fix in Finding 4.

## 9. P3 [D2] — the scheduler's reminder handler does synchronous dispatch/DB work on the event loop
- Verdict: CONFIRMED
- Site now: `Scheduling/scheduler/handlers/reminder_handler.py:33-47` (`async def handle`, calls
  `self.dispatch_service.dispatch(...)` with no `await`/offload)
- Proof: `grep -n "def dispatch"
  tldw_chatbook/Notifications/notification_dispatch_service.py` confirms it's a plain `def`
  (sync). `Scheduling/scheduler/loop.py:797-799` awaits `handler(task)` inline with no
  `to_thread`. Matches the review's own "inferred, not measured" confidence — I did not read
  `dispatch`'s full body either, only confirmed it is sync and uncalled-via-offload.

TOTALS: confirmed=9 fixed=0 wrong=0 demoted=0 promoted=0
