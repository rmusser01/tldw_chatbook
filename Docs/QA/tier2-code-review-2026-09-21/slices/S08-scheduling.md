# S08 — `Subscriptions/` + `Scheduling/`

**Coverage:** files read in full: 13 | sampled: 25 | mechanical only: 49 (of 87).
Full: `scheduler_heartbeat.py`, `scheduler/queue.py`, `scheduler/loop.py`, `schedule_compute.py`, `recovery.py`,
`services/watchlist_projection.py`, five of the six `db/migrations/v*.py`, `Subscriptions/db_offload.py`.
Sampled: 25 (targeted regions, 100–600 lines each). Mechanical only: 49 — pattern rows plus cross-cutting greps
(`trusted_origins`, `strftime`, `os.replace`, `threading.Lock`, `owner_id`).

## Findings

### P0 [D1] — A sync while the Schedules view is on "This device" mirrors server-owned reminders **and** automation definitions under `owner_id="local"`, arming them for local execution alongside the server's own — the double execution ADR-077 exists to prevent
- Where: `Scheduling/services/sync_engine.py:265-273` (`pull`:
  `target_owner = owner_id if owner_id is not None else self.owner_id`), `:291-346` (`_pull_reminders` →
  `_apply_pulled_reminders(conn, target_owner, …)`), `:1354-1391` (`_pull_definitions` →
  `upsert_automation_definitions_from_server(owner_id, …)`); `Scheduling/db/scheduled_tasks_db.py:625-683` and
  `:2802+` (both store the caller's `owner_id` verbatim); owner origin
  `Scheduling/services/scheduling_service.py:337-341` + `app.py:10865-10868` (`runtime_source="local"`); the only
  mutator is the workbench toggle `UI/Screens/scheduling/schedules_workbench.py:4870-4893`; the sync trigger
  `:5355-5393`/`:5395-5408` gates on server **reachability** only and passes `service.owner_id` straight through.
- Evidence: **reproduced by the reviewer against a real `ScheduledTasksDB`; re-traced by the lead** — see
  `phase4-verification.md` "S08-P0".
  ```
  SyncEngine(db, FakeClient(), "local").pull()      # FakeClient returns one server reminder
  -> mirrored rows: [('17bbfde9', 'local', 'srv-1', True, '2020-01-01T09:00:00+00:00')]
  -> PriorityQueue(db).load(); pop_due(now)
     ARMED LOCALLY / due now: [('Server-owned daily digest', 'local')]
  # and with one server automation definition (family=recurring_question, lifecycle=configured):
  -> db.list_armable_automation_definitions("local")
     ARMABLE LOCALLY: [('Nightly research brief', 'local', 'srv-def-1')]
  ```
- Why it matters: ADR-077 (`backlog/decisions/077-server-offloaded-scheduled-agent-tasks.md`) decision 1 is
  "execution follows ownership", and its **rejected** alternative is named as "Both sides execute, dedupe at
  delivery — *the current de-facto shape for reminders* … for agent work this is double execution with
  nondeterministic ordering, and dedupe after the fact cannot un-run side effects." Every guard built for this
  (`is_server_scoped_owner` at the queue seam, `_still_armable`'s dispatch-time re-read,
  `list_armable_automation_definitions`' owner filter, `run_reminder_now`'s refusal) keys on the `server:` prefix —
  and **the pull is the one writer that never puts the prefix on.** Result: duplicate reminder notifications, and
  for `recurring_question` definitions a second unattended LLM run per occurrence at the user's expense.
- Recommended correction: the pull's *storage* owner must come from the connected server identity, not the UI view
  toggle. `SchedulingService._active_server_owner_id()` (`scheduling_service.py:1162-1180`) already computes
  `f"server:{app.active_server_id}"` for exactly this reason, **and its docstring already states the rule**
  ("`self.owner_id` is a UI-togglable VIEW … it cannot stand in for 'which server this session is connected to'").
  Thread that value into `sync_now`/`pull` as the storage owner (keeping `self.owner_id` as the view filter), or
  refuse `sync_now` outright when the active owner is not server-scoped.
- Size: M · ADR: no (ADR-077 already decides it; this is an unclosed implementation gap) · Confidence: **verified**
- Pinning test: none. `Tests/Scheduling/test_sync_engine.py` constructs `SyncEngine(..., owner_id="server:1")` in
  **every** pull test (lines 22, 45, 67, 82, 95, 110, 128, 144, 159, …); the only `owner_id="local"` case (line 56)
  passes `server_client=None`, so it never pulls. `Tests/Scheduling/test_owner_filter.py` pins the *queue-side*
  filter, not the pull-side owner assignment.
- Already covered: none.

### P1 [D1] — Watchlists `sitemap` sources fetch sitemap-**discovered** URLs with `trusted_origins` seeded from those same discovered URLs, so a `<loc>` naming a private/loopback address is fetched
- Where: `Subscriptions/local_watchlists_service.py:2502-2540` (`_urls_for_sitemap` returns `<loc>` text from the
  fetched document) → `:2161-2177` (`_default_run_executor`'s `sitemap` arm loops those into `_check_url_guarded`)
  → `:2421-2470` (`_check_url_isolated` → `monitor.check_url({**config, "source": url, "type": "url"})`) →
  `Subscriptions/monitoring_engine.py:1759-1768` (`_fetch_url_content`:
  `guarded_fetch_httpx_async(url, …, trusted_origins=origin_set(url))`).
- Evidence:
  ```
  evaluate_url_policy("http://127.0.0.1:8080/admin", trusted_origins=origin_set(same)) -> allowed=True  reason=ok
  evaluate_url_policy("http://127.0.0.1:8080/admin", trusted_origins=frozenset())      -> allowed=False reason=private
  evaluate_url_policy("http://192.168.1.1/",         trusted_origins=origin_set(same)) -> allowed=True  reason=ok
  ```
  Metadata IPs stay blocked (`egress._post_resolution` checks `metadata` **before** consulting `trusted_origins`),
  so the exposure is private/loopback/LAN, not 169.254.169.254.
- Why it matters: `Utils/egress.py`'s module contract is "Shared pipeline code must NEVER auto-trust its own input
  URL", and `config.py`'s `[web_security]` block names the exact input: "content-derived URLs (redirects,
  **sitemap/crawl discoveries**, feed items) must resolve to public IPs".
  `Tests/Web_Scraping/test_sitemap_crawl_trusted_origins.py` fixed four latent (caller-less) copies of this and
  explicitly cleared the Watchlists path — but it checked only the *sitemap document's own* fetch
  (`_urls_for_sitemap`, correctly provenance-correct) and never the per-`<loc>` fetches downstream. **The shipped
  `sitemap` source type is the one live caller that test file names.**
- Recommended correction: `_check_url_guarded`/`_check_url_isolated` should thread an explicit `trusted_origins`
  down (seeded from the *subscription's* `source`, not the discovered URL), and `URLMonitor._fetch_url_content`
  should take it as a fail-closed keyword defaulting to `frozenset()` — the shape
  `Article_Extractor_Lib.scrape_article` already uses. `_urls_for_sitemap`'s own `origin_set(source)` stays
  (user-configured), as do the `url`/`url_list`/`feed`/`api` arms'.
- Size: M · ADR: no · Confidence: verified (policy behaviour reproduced; call chain traced, not executed end to end)
- Pinning test: none for `Subscriptions/`. The one adjacent test states the *opposite* requirement as a rule.
- Already covered: none (TASK-32806 `.1`–`.8` have no egress-trust item).

### P2 [D1] — `v4_to_v5.rollback` and `v5_to_v6.rollback` each stamp a schema version one lower than the schema they leave behind
- Where: `Scheduling/db/migrations/v4_to_v5.py:64-73` (docstring "returning `db` to schema version 4"; guard
  `if current_version >= 4`; `INSERT … VALUES (3)`) and `v5_to_v6.py:69-78` (docstring "…to schema version 5";
  guard `>= 5`; `INSERT … VALUES (4)`). `v3_to_v4.py:175-176` and `v6_to_v7.py:128-131` get it right, which is what
  makes these two visible as typos rather than a convention.
- Evidence:
  ```
  start version: 7
  after v5_to_v6.rollback: 4   (docstring says 5)
  after v4_to_v5.rollback: 3   (docstring says 4)
  v4 tables still present: {'automation_results', 'automation_runs'}
  ```
- Why it matters: the recorded version and the physical schema disagree — the half-applied state the migration
  chain's structural detection is meant to make unobservable. `Scheduling/recovery.py`'s `_ScheduledTasksAdapter`
  validates a restore candidate with `version_query="SELECT MAX(version) FROM schema_version"` against
  `versions=(7,)`, so the stamp is not cosmetic. Forward re-migration self-heals (each step re-detects structurally),
  which is why this is P2.
- Size: S · Confidence: verified
- Pinning test: **none — both `rollback()`s are entirely untested** (`rg -n "v4_to_v5|v5_to_v6" Tests/` returns only
  unrelated Workspace/Media/Library migrations), while `v4` and `v7` each have a round-trip assertion.

### P2 [D1/D4] — `scheduler_heartbeat.write_heartbeat` describes itself as durable and atomic, hand-rolls temp+rename, and does not `fsync` — TASK-32808.5's census never reached it
- Where: `Scheduling/scheduler_heartbeat.py:1-9` (module docstring: *"durable scheduler liveness heartbeat. A small
  **atomic** JSON file…"*), `:92-124` (`mkstemp` → `fdopen.write` → `os.replace`, no `flush`/`fsync`).
- Evidence: TASK-32808.5's Implementation Notes enumerate the census by name — `RAG_Search/config_profiles`,
  `emergency_stop`, `Agents/local_tool_provider`, `Chat/trajectory_export`, `Tools/local_tool_impls`,
  `UI/Console_Modules/video`, `Utils/tls_trust`, `MCP/permission_store` — and **neither
  `Scheduling/scheduler_heartbeat.py` nor `Subscriptions/briefing_export.py` appears**. The shared helper does fsync
  (`Utils/atomic_file_ops.py:98-103`).
- Why it matters: atomicity and durability differ. On an OS/power crash the heartbeat can come back truncated;
  `read_heartbeat` (`:68-73`) maps any `ValueError` to `None`, and `classify_scheduler_liveness` maps `None` to
  `"never_started"`. The surface then reports "Scheduler: not started" for a scheduler that had been ticking — the
  precise confusion TASK-26025 was written to remove.
- Recommended correction: route through `Utils/atomic_file_ops.atomic_write_text` (the call site already swallows
  every exception). *(Lead's note: the helper still lacks a parent-directory fsync and the Darwin `F_FULLFSYNC`
  barrier — see the repo-wide finding in `report.md`. Adopting it here is an improvement, not a complete fix.)*
- Size: S · Confidence: verified (census-membership verified against the task file; fsync absence verified by read)
- Already covered: TASK-32808.5 (Done) — **this shows that task's census was incomplete.**
- **Retired sibling:** `Subscriptions/briefing_export.py:626-640` is the *other* uncensused hand-roll, but it **does**
  fsync before `os.replace` and has a documented `O_NOFOLLOW`/`fchmod` reason to stay hand-rolled — a justified keeper.

### P2 [D4] — `_effective_max_tokens` + `_invoke_chat` are three copies differing only in three module constants; the TASK-21515 DeepSeek fix and its Qodo #7/#8 follow-up each had to land three times
- Where: `Subscriptions/briefing_service.py:570-593`/`:594-625`; `Subscriptions/briefing_cast.py:477-495`/`:496-525`;
  `Library/library_rag_answer_service.py:481-504`/`:506-537`.
- Evidence: token-identical except `BRIEFING_*` vs `CAST_*` vs `ANSWER_*` (`{REASONING_,}MAX_TOKENS`, `TEMPERATURE`).
  Two of the three carry a "Copies `briefing_service.…`'s exact shape (not imported: that function is private to its
  own module)" note — the duplication is acknowledged, its cost is not.
- Why it matters: the reasoning-budget rule ("the DeepSeek handler's `max_tokens` is the whole reasoning-inclusive
  budget, so a reasoning-typed default returns an empty completion") is a **provider** fact, not a caller fact.
  Three copies means three chances to miss it; a fourth non-streaming single-shot caller will get an empty
  completion on DeepSeek by default.
- Recommended correction: one `Chat/` helper taking `(endpoint, model, *, max_tokens, reasoning_max_tokens)` plus
  one `invoke_single_shot_chat(...)`; the three call sites pass their own constants. Canonical home: beside
  `chat_api_call`'s own seam in `Chat/` (all three already import `model_capabilities` symbols from there).
- Size: M · Confidence: verified
- Already covered: TASK-32808.9 would cover the byte-identical `_error_text`; **the `_invoke_chat`/
  `_effective_max_tokens` pair is a behaviour-carrying copy, not a verbatim shim, and is named nowhere.**
  *(Cross-slice: S05 independently found `_error_text` has DRIFTED — Library caps at exactly `ERROR_CHAR_CAP`, both
  Subscriptions copies emit `ERROR_CHAR_CAP + 6`.)*

### P2 [D4] — `Scheduling/` writes a timestamp shape ADR-173 does not sanction, and the ADR-173 guard is structurally blind to it
- Where: `Scheduling/db/scheduled_tasks_db.py:713-750` (`_to_utc_iso` → `datetime.isoformat()`, i.e. microsecond
  `+00:00`), used by ~30 write sites in that file; `:3302-3318` (`_serialize_result_fields` stores server-supplied
  timestamp *strings* verbatim). Four hand-rolled readers: `services/watchlist_projection.py:11`,
  `services/briefing_projection.py:69`, `services/scheduling_service.py:1386`,
  `scheduler/handlers/automation_handler.py:61`, plus `scheduled_tasks_db._parse_utc_iso:1237` and
  `schedule_compute._naive_as_utc:35`.
- Evidence:
  ```
  $ .venv/bin/python scripts/check_timestamp_writers.py
  timestamp writers: 0 datetime.utcnow() site(s), 0 naive datetime.now().isoformat() occurrence(s) (0 pinned). OK
  ScheduledTasksDB._to_utc_iso(now)  -> 2026-09-22T04:28:01.580877+00:00 | is_canonical_utc? False
  Utils.timestamps.utc_now_iso()     -> 2026-09-22T04:28:01.580Z
  ```
  `rg -ln "Utils\.timestamps" tldw_chatbook/Scheduling tldw_chatbook/Subscriptions` → **one hit total**
  (`Subscriptions/site_config_manager.py`); the helper has 28 importers repo-wide.
- Why it matters: **this column set has already paid for the drift.** `list_automation_results` (`:2596-2604`) and
  the v6→v7 dedupe (`v6_to_v7.py:49-66`) both had to wrap `strftime('%Y-%m-%dT%H:%M:%f', …)` around
  `created_at`/`updated_at` because locally-written (`+00:00`, microseconds) and server-mirrored (`Z`, verbatim)
  rows are lexically incomparable — and the v7 comment states the stake: "picking the wrong 'newest' here is not a
  display bug, it permanently discards the real newest row."
- Recommended correction: (a) point `_to_utc_iso` at `Utils.timestamps.to_utc_iso` for new writes; (b) normalize
  server-supplied timestamp *strings* in `_serialize_result_fields` and `_apply_pulled_reminders`; (c) keep the
  read-side `strftime()` normalizations (existing rows are not rewritten); (d) extend
  `check_timestamp_writers.py` to flag `.isoformat()` on an aware UTC datetime reaching a DB parameter.
- Size: M · Confidence: verified
- Pinning test: `Tests/Scheduling/test_schedule_compute.py:58 test_slot_string_is_canonical_utc_iso` asserts the
  `+00:00` shape — **a false friend**: the name uses ADR-173's word, the assertion predates it and pins the
  non-canonical shape. It would have to be updated deliberately.
- Already covered: TASK-32803.5 (Done) — **its census, by construction, could never see this writer.**
  *(Fifth independent sighting of a `check_timestamp_writers.py` blind spot; see S01, S05, S06, S11.)*

### P3 [D3] — `baseline_manager._get_latest_baseline` computes and discards a decompression, and its `except` body is a bare expression
- Where: `Subscriptions/baseline_manager.py:663-668` —
  `try: zlib.decompress(row["raw_html"]).decode("utf-8") / except Exception: row["raw_html"]`. **Both statements are
  dead** (results discarded). `ContentBaseline.compressed_content` therefore stays `None` on every round-trip, and
  `_store_baseline` (`:696`) writes `baseline.compressed_content or b""` — a re-store of a read-back baseline would
  blank `raw_html`.
- Why it matters: a data-loss shape, defused only by the module having **zero importers**
  (`rg -n "from .*baseline_manager import|import baseline_manager"` → nothing). TASK-1360 ("Decide the fate of
  baseline_manager", Done) resolved to leave it; this is one more argument for deleting rather than leaving an
  attractive-looking donor module whose docstrings advertise a richer vocabulary than the live path.
- Size: S · Confidence: verified

### P3 [D3] — `write_heartbeat`'s cleanup path opens `/dev/null` and immediately closes it
- Where: `Scheduling/scheduler_heartbeat.py:114-116` — `except Exception: with open(os.devnull, "w"): pass` before
  the real `os.unlink(tmp)`. A no-op with no comment. Deleted for free by the P2 adoption above. · Size: S

### P3 [D2] — the scheduler's own reminder handler does synchronous notification/DB work on the event loop
- Where: `Scheduling/scheduler/handlers/reminder_handler.py:34-48` — `async def handle` calls
  `self.dispatch_service.dispatch(...)` (sync, persists an inbox notification) with no offload; `SchedulerLoop`
  awaits it inline (`loop.py:797/799`).
- Why it matters: the loop is otherwise scrupulous — `_offload` wraps every DB call, `_emergency_stopped` and
  `_record_heartbeat` were explicitly moved off-loop by TASK-31507, `_run_preflight` offloads after Qodo #6 — so
  this is the one remaining inline sync path, and it blocks the heartbeat write in the same `tick`.
- Size: S · Confidence: inferred (not measured; `dispatch` not read in full)
- Already covered: report as an instance for TASK-32804.12 (To Do).

## Candidate triage
**CONFIRMED:** `os_replace_no_atomic` `scheduler_heartbeat.py:117` (P2 — missed by 32808.5's census, not deliberately
excluded); `token_est_len_div4` `token_manager.py:113` (already TASK-32808.10); `inline_truncate`
`github_scraper.py:419` and `automation_execution._bounded:86` (already TASK-32808.3).
**RETIRED:** `os_replace_no_atomic` `briefing_export.py:640` — does `flush`+`fsync` before `os.replace` with
`O_EXCL`/`O_NOFOLLOW`/`fchmod` and a documented reason; justified keeper under 32808.5 AC#3.
`strftime` `v6_to_v7.py:40,56,58` + `scheduled_tasks_db.py:2534,2602` (`%Y-%m-%dT%H:%M:%f`) — **retired as a bug,
confirmed as a symptom**: SQLite's `%f` is `SS.SSS` (verified:
`strftime('%Y-%m-%dT%H:%M:%f','2026-09-21T12:00:06.5Z')` → `2026-09-21T12:00:06.500`), so seconds are not missing
and offsets/`Z` normalize correctly. It is the compensation for the P2 mixed-shape column, not a defect.
`strftime` ×6 elsewhere — display-only, explicitly localized via `.astimezone()` with a naive→UTC guard first.
`fetchall_no_limit` — 6 migration `PRAGMA table_info` rows (bounded by column count); `scheduled_tasks_db.py`
sync/queue paths that need the whole armable set (`list_armable_automation_definitions` is separately capped with a
truncation warning). `fetchall_dynamic_sql` `briefing_selection.py:250/331` — interpolated fragments are
module-level literals; every caller value is bound. `lock_and_execute` ×2 — one guards `_db_instance` construction
(double-checked, documented), the other is per-domain in-memory rate-limit state. `mutable_class_attr`
`web_scraping_pipelines.py:347` — a class-level type registry mutated only by `register_pipeline`.
`except_exception_pass` ×7 and `except_exception_return` ×10 — each a diagnostics/classification path with an
explicit `noqa: BLE001` and a stated reason. `function_body_import` ×57 — ADR-097 boot-ratchet comments or
optional-dep/cycle breaks. `try_import_guard` ×25 — `defusedxml`-with-fallback is the house shape. `legacy_markers`
×32 — comments documenting deleted producers/consumers, `git grep`-verified.
`_maybe_await` — no call site in this slice passes a sync argument that does blocking I/O.
**XXE:** `pytest Tests/Subscriptions/test_watchlist_opml_entity_expansion.py -q` → **14 passed**. `Subscriptions/`
is fully hardened, no `_KNOWN_UNHARDENED` entry lives in this slice, and no new unregistered stdlib parse exists.
*One blind spot worth the lead's note:* the census early-returns on **any** `defusedxml` import in a module, so a
module that imports it once and parses with stdlib `ET` elsewhere is invisible — no instance in this slice.
**Egress:** all 17 `httpx.AsyncClient` sites route through `guarded_fetch_httpx_async`; the scrapers correctly pass
`trusted_origins=frozenset()`. The one defect is the *provenance of the trust seed* on the sitemap path (P1), not a
missing guard.
**UNVERIFIED:** `fetchall_no_limit` in `watchlist_bundle_service.py` ×6, `site_config_manager.py:464/623`,
`local_watchlists_service.py:1844/2859/3002/3030` — bounded by user-curated set sizes, not measured.

## D4 observations for repo-wide Phase 3
1. `_effective_max_tokens` + `_invoke_chat` (3 copies, constants-only difference) — home `Chat/`. Drift latent, not
   yet realized: all three agree only because the DeepSeek fix was applied three times by hand.
2. `_error_text` (3 byte-identical) — documented duplication; feeds TASK-32808.9. **S05 found it has drifted.**
3. `_sanitize_text`/`_string_value`/`_nonempty`/`_valid_text`/… — **8 copies** of "stripped non-blank str or None",
   one being `Scheduling/automation_execution.py:94`. Zero drift. Home: `Utils/`.
4. **`validate` + `capture` across 11 `*/recovery.py` modules** — **not** the abstract-method case in the lead's
   addendum: these are copy-pasted bodies differing only in a string owner-tag literal and a `version_query`. A
   `_SQLiteDeclaration` base implementation parameterised by two class attributes would subsume all eleven.
   *(Lead's note: S25 independently reached the same cluster from the `Backup_Recovery` side and ruled it Protocol
   conformance — the two readings differ and Phase 3 resolves them; see `report.md`.)*
5. **`Utils/timestamps.py` (28 importers) has one adopter in this whole slice.** `Scheduling/` has zero and rolls one
   writer plus six readers. The "helper exists, ignored" sub-case where the ignoring has already produced a
   data-correctness workaround (P2 above).
6. `_parse_iso_timestamp` ×2 in `Scheduling/services/` — byte-identical with a comment explaining why they were not
   merged; both now supersedable by `Utils.timestamps.parse_utc`.
7. `_require_beautifulsoup` ×3 — byte-identical optional-dep guards; one copy is in the dead `baseline_manager`, so
   a 2-copy cluster after TASK-32807 lands.

## Left UNVERIFIED
| Claim | Why | Command |
|---|---|---|
| The P1 sitemap SSRF is reachable end to end at runtime (not just call-chain + policy repro) | needs a live `URLMonitor.check_url` against a fake sitemap server; app must not be run | `pytest Tests/Subscriptions/test_local_watchlists_service.py -q` after adding a case that serves a sitemap containing `<loc>http://127.0.0.1:PORT/…</loc>` and asserts `EgressBlockedError` — currently it would pass the fetch |
| The P0 double-fire is *observable* as two user-visible notifications (i.e. the server actually fires the mirrored row concurrently) | requires a real tldw_server; the client side is proven, the server side is asserted by ADR-077 §1/§2 | point the app at a live server, create a reminder server-side, toggle Schedules to "This device", press `s`, wait past `next_run_at`, compare the inbox count against the server's run ledger |
| Whether `_schema_is_current()`'s fast path can leave `schema_version` holding two rows in a way `get_schema_version()` — `SELECT version … LIMIT 1` with **no `ORDER BY`**, unlike every migration's `MAX(version)` — reads wrongly | traced every version 0–7; each self-corrects within one chain run, and a persistent two-row state needs a hand-dropped table | `for v in 0..7: seed a db at v; run ScheduledTasksDB(path); assert COUNT(*) FROM schema_version == 1` |
| The `fetchall_no_limit` rows in `watchlist_bundle_service` / `local_watchlists_service` / `site_config_manager` are bounded in practice | not measured; all user-curated sets | `sqlite3 ~/.local/share/tldw_cli/*/tldw_subscriptions.db "select count(*) from watchlist_sources; select count(*) from subscriptions; select count(*) from site_configs;"` |
| `NotificationDispatchService.dispatch` actually does blocking SQLite work (P3) | `Notifications/` is outside this slice | `rg -n "def dispatch" -A 40 tldw_chatbook/Notifications/notification_dispatch_service.py` |
