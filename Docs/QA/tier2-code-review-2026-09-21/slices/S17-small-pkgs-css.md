# S17 — twelve small packages + the Python under `css/`

**Coverage:** files read in full: 8 | sampled: 24 | mechanical only: 48 (of 80).
`css/widget_css.py` (1,014 lines) was read **docstring-only** — flagged in UNVERIFIED.

## Findings

### P1 [D1] — An expired/invalid server token leaves the notification observer in a permanent silent retry loop; no path in `Notifications/` classifies a 401 or offers re-auth
- Where: `Notifications/event_observer.py:233-243`; `server_notification_events.py:146-150`; **root cause**
  `tldw_api/client.py:1668-1682`; consumer `UI/Screens/scheduling/schedules_workbench.py:970-1004`.
- Evidence: `issubclass(httpx.ResponseNotRead, ValueError)` → **False** (MRO is
  `ResponseNotRead → StreamError → RuntimeError`). Reproduced: `raise_for_status()` on an **unread** streaming 401 →
  `e.response.json()` raises `ResponseNotRead`, which `except ValueError: pass` at `client.py:1676` does **not**
  catch, so the `AuthenticationError` at `:1679` is never constructed.
  `rg -n "AuthenticationError|401|reauth|unauthor" tldw_chatbook/Notifications/` → **no match. Zero 401 handling in
  the package.** Propagation traced to `except Exception:` at `event_observer.py:233`.
  `_NOTIFICATION_OBSERVER_MAX_RECONNECTS = 5`, `_RESTART_DELAY = 5.0`: five exponential-backoff reconnects, then
  re-raise; `observe()` records `status="error", reason=type(exc).__name__` = **`"ResponseNotRead"`**; the workbench
  logs **one** `warning` and every identical repeat at `debug`, then restarts after 5 s — **forever**.
  `rg -n "record_observer_status" tldw_chatbook/` → written in one place, **read by no UI.**
- Why it matters: an expired token is retried like a dropped socket indefinitely; **the user sees notifications
  silently stop with no prompt to re-authenticate**, and the recorded reason is a transport artefact that names
  nothing about auth.
- Recommended correction: **two parts, both needed.** (a) `await e.response.aread()` before `.json()` and widen the
  guard to `except (ValueError, httpx.ResponseNotRead)` — that alone makes `AuthenticationError` reachable
  (**this is S06's fix**). (b) In `event_observer.py`, add a non-retryable classifier arm *above* the generic one so
  an auth failure exits the loop with a distinct result instead of consuming the reconnect budget.
- Size: M · Confidence: **verified** (httpx behaviour reproduced; chain read end to end; constants greped)
- Already covered: **no.** This is the `Notifications/` half of S06's `tldw_api` finding and **needs a fix on both
  sides.**

### P1 [D1] — `Metrics/metrics_logger.timeit` times coroutine *creation*, not execution: 8 async call sites record ~0 s and `status="success"` for every failure
- Where: `Metrics/metrics_logger.py:89-124` — **the `wrapper` is `def`, not `async def`.** Async call sites:
  `RAG_Search/reranker.py:496,694,828,1180`; `parallel_processor.py:144,289,427`;
  `simplified/enhanced_rag_service.py:69`; `simplified/rag_service.py:3042`.
- Evidence: **reproduced.**
  ```
  @timeit("slow_async") async def slow(): await asyncio.sleep(0.25); return "done"
    -> metric slow_async: value=0.000000 status=success
  @timeit("boom_async") async def boom(): await asyncio.sleep(0.05); raise RuntimeError("real failure")
    -> metric boom_async: value=0.000004 status=success    # the exception still propagates to the caller
  ```
- Why it matters: **every `reranker_*`, `batch_*`, `enhanced_rag_indexing_document` and `rag_chunking_operation`
  histogram is a constant ~0, and every failure on those paths is labelled `success`.** A metric that is actively
  misleading is worse than a missing one; any perf work consulting these reads noise.
- Recommended correction: branch on `inspect.iscoroutinefunction(func)` and return an `async def` wrapper that
  `await`s inside the `try`. ~12 lines, no call-site change. · Size: S · Confidence: **verified (reproduced)**

### P1 [D2] — The server-notification observer executes 15.2 SQLite statements and 2 `BEGIN IMMEDIATE` write transactions **per event** synchronously on the Textual event loop; half of those write transactions are empty
- Where: `Notifications/event_observer.py:168-212` (four plain sync store calls in an `async def` body) +
  `server_notification_events.py:133,152`. Worker: `schedules_workbench.py:909-914` `run_worker(...)` — **a
  coroutine worker.**
- Evidence: instrumented `sqlite3.Connection.set_trace_callback` over a 5-event run against a real file-backed
  `EventStateRepository`:
  ```
  handled: 5 | statements: 76          -> 15.2 statements / event
  BEGIN IMMEDIATE: 10   COMMIT: 10     -> 2 write-lock transactions / event
  15 SELECT 1 FROM event_dedupe_records …   (3 probes per event)
   5 INSERT INTO event_dedupe_records …     <- only 5, not 10
  ```
  `acknowledge_event` already inserts the dedupe row; `EventObserver.run:209` then calls `remember_event`, which
  opens a **second `BEGIN IMMEDIATE`**, re-probes the same key, finds it present, **and commits nothing — 5 of the
  10 `BEGIN IMMEDIATE`s are pure WAL write-lock acquisitions with zero writes.**
- Why it matters: every server notification stalls the UI thread for a full read-modify-write round trip plus an
  empty second write-lock acquisition, and `BEGIN IMMEDIATE` contends with the dispatch-worker threads writing the
  same store.
- Recommended correction: (a) delete the redundant `remember_event` on the `ADVANCED` branch; (b) route the four
  store calls through an injectable `offload` defaulting to `asyncio.to_thread`.
- Size: S for (a), M for (b) · Confidence: **verified (measured)**

### P2 [D2] — `NotificationDispatchService.dispatch` does three blocking SQLite operations and is awaited inline by the scheduler, which is documented as a coroutine worker
- Where: `Notifications/notification_dispatch_service.py:22-83` (sync `def`); caller
  `Scheduling/scheduler/handlers/reminder_handler.py:41`; same shape at `briefing_handler.py:486,563` and
  `automation_handler.py:639`.
- Evidence: `app.py:17117-17130` — `run_worker(self.scheduler_loop.run(), …)` **with the comment "A COROUTINE
  worker, never `thread=True`"**. `dispatch()` does `get_settings()` + `insert_notification()` + a follow-up
  `get_notification()` — three round trips; **first-ever dispatch additionally triggers `_ensure_schema` →
  `connect_private_sqlite` → `executescript` CREATE TABLE, all on the loop.**
  **Contradiction:** `client_notifications_db.py:53-57` states *"the inbox is read from the UI thread and written
  from **dispatch worker threads**"* — **the scheduler path is neither.**
- Size: S · Confidence: verified
- **This settles S08's open question: yes, `dispatch` does blocking SQLite, and yes, it is awaited inline with no
  offload.**

### P2 [D1] — `ensure_readable_text_hues` is applied to all 70 shipped themes and every user theme, but never to Textual's own `BUILTIN_THEMES`; `textual-light` ships `$ds-value-fg` at 2.8:1
- Where: guard at `css/Themes/themes.py:83-126`, applied at `:56` and `:1929-1931`. **Gap:**
  `textual.app.App.__init__` registers `BUILTIN_THEMES` itself, and `app.py:1147-1160` offers `"textual-dark"`,
  `"textual-light"` plus everything in `self.app.available_themes`.
- Evidence: measured with the module's own `_contrast_ratio`/`_AA_RATIO`:
  ```
  textual-light:     text-accent  2.80      catppuccin-latte:  text-accent  3.97
  catppuccin-frappe: text-primary 4.31      atom-one-dark:     text-primary 4.22, text-accent 3.38
  atom-one-light:    text-accent  3.65
  shipped ALL_THEMES failures after guard: 0 of 70
  ```
  Consumers are live: `$ds-value-fg: $text-accent` in `screen_feature_scheduling.tcss:47-50` and
  `screen_feature_evals.tcss:47-50`.
- Why it matters: **this is the same defect `ensure_readable_text_hues` was written for, still live on 5 themes the
  switcher offers — including the one light theme named in the app's own hard-coded list.**
- Recommended correction: run the guard over `self.available_themes.values()` after `App.__init__` registers the
  built-ins; it is already idempotent. Extend `test_theme_contrast.py`'s parametrize to include `BUILTIN_THEMES`.
- Size: S · Confidence: **verified (measured)**
- Pinning test: `test_resolved_readable_tokens_clear_aa_on_every_theme` **states the requirement, but its corpus
  excludes exactly the failing themes.**
- Already covered: **TASK-31429 shipped the guard; the guard's coverage set was never the registered-theme set.**

### P2 [D4] — `_coerce_bool` is re-rolled 3× and drifts from the canonical `config.coerce_bool_setting` on 4 of 13 inputs; the drifted values persist to rail state
- `Home/home_rail_state.py:24-35`, `Chat/console_rail_state.py:335-346`, `Library/library_rail_state.py:53-64` —
  byte-identical to each other. Canonical: `config.py:1201-1218` (**26 importing files**).
  ```
  input      _coerce_bool(F)   coerce_bool_setting(F)
  ' yes '    True              False
  2          True              False
  't'        False             True
  'y'        False             True     (9 other cases agree)
  ```
- Why it matters: **a hand-edited config value means different things on Home vs Console vs Library, and the
  resolved boolean is persisted back into rail state — the drift reaches storage.**
- Recommended correction: delete all three and the six retyped string sets; import the canonical helper. **Pick the
  deliberate semantics first** — `" yes "` and non-zero `int` currently work in the rails and would stop; `"t"`/`"y"`
  currently don't and would start. **That choice is the whole content of the change.**
- Size: S · Confidence: **verified (differential run)**
- Already covered: **TASK-32808.4 is marked Done and these three non-adopters survived it.**
  *(Lead's note: this is a different group from the 11 `_coerce_bool` copies the lead ruled on — those diverge from
  `Utils.coerce_bool_flag`; these three diverge from `config.coerce_bool_setting`, the narrower gate helper.)*

### P2 [D3] — `tldw_chatbook/Coding/` is a one-file dead package that cannot even be imported
- 358 lines, **no `__init__.py`**. AST census over every non-`Coding/` file → **zero importers**; `rg` → only QA
  JSON and one backlog doc row. `python -c "import tldw_chatbook.Coding.code_mapper"` →
  **`ModuleNotFoundError: No module named 'diskcache'`**, and `grep -c diskcache pyproject.toml` → **0**.
  It nonetheless **ships** (listed in the wheel-identity manifest). Its `SimpleIO.tool_output/warning/error` use bare
  `print()`, so reviving it as-is would write to stdout under a Textual TUI. · Size: S · Confidence: verified
- Already covered: **`Coding/` is in no TASK-32807 bucket.**

### P3 [D4] — `Chatbooks/conflict_resolver.py:204` writes naive **local** time into merged note *content*, two lines above a correct `utc_now_iso()` call at `:210`
- The timestamp is baked into the user's note content, so it is **permanent and unconvertible**; two users in
  different zones merging the same chatbook produce different note bodies. · Size: S
- Already covered: **TASK-32803.5 (Done) missed it.**

### P3 [D4] — `conflict_resolver.py:145,147` append `"..."` unconditionally, claiming truncation that did not happen
- A 3-character description renders as `abc...` **in the conflict prompt the user resolves against.** Two of
  TASK-32808.3's 52 re-rolls — **noted because these two carry the actual defect, not just the duplication.** · Size: S

### P3 [D4] — `Chatbooks/server_chatbook_service.py:29` re-rolls `Utils/timestamps.utc_now_iso` output-identically
- Both produced `2026-09-22T05:04:06.235Z`; `is_canonical_utc()` → True for both. **Zero drift — pure duplication.** · S

### P3 [D2] — `Stats/user_statistics.py:604` compiles a 7-range emoji regex inside the function body
- Immediately after a `LIMIT 5000` `fetchall()`. Small — the surrounding work dominates — and it runs on a thread
  worker, so off-loop. Hoist to a module constant. · Size: S

### P3 [D2] — `_count_matching_presentations` issues one `SELECT` per event key inside an `IMMEDIATE` transaction
- `Notifications/event_state_repository.py:1936-1949`, called from `clear_server_profile_state` with
  `event_keys` = **every event row for the profile**. **The write lock is held for the whole scan** — so logout,
  credential removal and profile deletion hold it for O(rows) round trips. One `SELECT COUNT(*) … WHERE event_key
  IN (SELECT …)` replaces it. · Size: S

### P3 [D3] — `state/__init__.py` eagerly imports 4 production-dead modules; one live leaf import drags them onto the boot path
- Only `state/ui_state.py` has a production consumer (`chat_screen.py:533`, 3 of ~15 fields used); `app_state.py`,
  `chat_state.py`, `navigation_state.py`, `notes_state.py` (431 lines) have **zero**.
  `Tests/Performance/boot_budget_snapshots/ui_ready_modules.txt:1021-1026` lists all six at ui-ready — **the leaf
  import triggers the package `__init__`.** Exactly the shape the ADR-097 ratchet exists to police.
- Recommended correction: the PEP 562 `__getattr__` idiom — **`Notifications/__init__.py:55` is a sibling in this
  same slice doing precisely that.** · Size: S
- Pinning test: `test_legacy_state_exports_remain_serialization_compatible` asserts the exports stay importable, so
  the lazy form must keep `__all__` working; **deleting them is out of scope.**

### P3 [D3] — 41 modules import six underscore-private helpers across a package boundary from `Backup_Recovery.participants`
- `rg -l "Backup_Recovery.participants import" tldw_chatbook/ | wc -l` → **41**; `participants.py` has **no
  `__all__`**. These are the de-facto public DB-participation API of the codebase, named as if private.
  · Size: M *(Same cluster S25 measured at ~30 underscore names / 20 modules for `_core_access` specifically.)*

### P3 [D1] — `Home/active_work_adapter._unread_notification_count` swallows every exception and reports "0 unread"
- `:487-489`. A genuine inbox DB failure is indistinguishable from an empty inbox on Home, with no log line.
  Also: the off-loop offload at `:470-481` is gated on a **five-way exact `type(...) is` identity check**, so any
  subclass or wrapper silently falls back to the inline sync branch. · Size: S

### P3 [D1] — `css/build_css` never fsyncs the parent directory after `os.replace`
- `:75-103` (`_atomic_write_text` fsyncs the file, no dir fsync) and `main()`'s publish loop at `:1235-1238` (**raw
  `os.replace`, no fsync at all**). Contrast `Tool_Packs/receipt_store.py:755-761`, which does both. Low stakes —
  the artefact regenerates — but the same discipline gap as the shared helper. · Size: S

## Candidate triage
**RETIRED:** `except_exception_pass` ×6 (`Otel_Metrics` nested cleanup inside an arm ending in a bare `raise`;
`Terminal` best-effort teardown). `except_exception_return` ×11 (`Tool_Packs/service.py` deliberate fail-closed
categorisation; `binding.py:459` fails toward *more* confirmation; `client_notifications_db.py:253` documented;
`themes.py:105` the documented ANSI/translucent escape, verified to fire only for `ansi-*`).
`fetchall_no_limit` ×8 (`GROUP BY DATE(...)` bounded to ≤31 rows; a 4-key settings table; a
`WHERE conversation_id = ?` preflight). `fetchall_dynamic_sql` `chatbook_creator.py:902` (fixed column list, all
values parameterised). `os_replace_no_atomic` `Tool_Packs/receipt_store.py:756` — **the strongest fsync discipline
in the slice (file + dir)**; `chatbook_creator.py:2236` fsyncs the file immediately before. `raw_mkdir` ×8 (the
importer side — the trust boundary — uses `secure_private_directory`). `tempfile_no_secure` ×9 (all
`mkstemp`/`NamedTemporaryFile(dir=…)` with `O_EXCL`/`0o600`, or build-time temp dirs). `id_keyed_dict`
`Tool_Packs/binding.py:235` — **the value holds a strong ref and `confirm()` re-checks identity; id reuse cannot
produce a false match.** `lock_and_execute` ×2 — **exemplary**: per-`threading.get_ident()` held connections under a
re-entrant lock, `BEGIN IMMEDIATE` with a documented `BUSY_SNAPSHOT` rationale, and a depth guard so `close()`
cannot pull a connection out from under live work. `sys_path_mutation` ×2 (guarded, documented bare-script path).
`try_import_guard` ×7. `legacy_markers` ×21. `function_body_import` ×56 (task-285 schema deferral, intra-package
cycle breaks, cold paths; `Metrics/metrics.py:268,287` documented as deliberate for import-order + testability).
`__getattr__` `Notifications/__init__.py:55` — **the correct lazy idiom, and the model recommended for `state/`.**
**`_maybe_await` — the shape is present (`notifications_scope_service.py:255,269,298,322,341,360,374` eagerly
evaluate a sync `ClientNotificationsService` method that goes straight to SQLite) but `rg` over `UI/` and `Widgets/`
finds NO live event-loop caller.** Reported as a negative with the re-check command. **The genuine
loop-blocking instance in this slice is not a `_maybe_await` one — it is `EventObserver.run`'s four direct sync
store calls (P1, measured).**

## D4 observations for repo-wide Phase 3
1. **The canonical atomic-write helper is *weaker* than at least one local re-roll — adopting it would be a
   regression.** `Utils/atomic_file_ops.py` has three writers and **zero** parent-directory fsyncs;
   `Tool_Packs/receipt_store.py:755-761` fsyncs the file **and** `_fsync_directory(self.root)` (and `:492-493` does
   root + root.parent). **Before TASK-32808.5's remaining adopters are converted, the helper needs the dir fsync
   added — otherwise every conversion silently downgrades durability.**
   *(Fourth independent confirmation; see S25, S14, S15/S16.)*
2. **`Sharing/` and `Outputs/` are a near-verbatim clone pair of the scope-service scaffold, not two
   implementations.** The 40-line preamble diffs **only** on the enum type name and three error strings. **The
   shared part is ~45 of ~360 lines per scope service**, × the 40-odd files in the cluster.
3. **`_maybe_await` D1: real shape, no reachable instance here** — with the command to re-check.
4. **Contesting one line of the lead's `_get_connection` ruling** — see the lead's correction in
   `phase4-verification.md`. Two of nine overrides call `connect_private_sqlite` **directly**;
   **anyone auditing by grepping for `super()._get_connection()` gets a false negative on exactly the two stores
   that had a documented reason to differ.**
5. **`_stream_fileno` — 3 copies** (`app.py:2393`, `Tools/raw_cli_executor.py:538`,
   `Terminal/posix_backend.py:155`), two byte-identical. Zero drift. Home: `Utils/`.
6. **`_coerce_bool` — 3 copies + 6 retyped string sets, with drift, against a 26-importer canonical helper**, and
   **TASK-32808.4 is marked Done.** The Phase 3 census should treat "Done" adoption tasks as needing a re-sweep.

## Left UNVERIFIED
| Claim | Why | Command |
|---|---|---|
| The `ResponseNotRead` 401 reaches `EventObserver`'s `except Exception` at runtime (traced by reading; the httpx half **is** reproduced) | app must not be run; no fault-injecting SSE transport exists in `Tests/Notifications/` | write a transport whose `stream()` raises `httpx.ResponseNotRead`, assert `_record_status` got `reason="ResponseNotRead"` and the loop restarted |
| The 15.2-statement cost is what a *real* server burst produces (measured against a synthetic 5-event transport + a real repo) | needs a live server | same `set_trace_callback` instrumentation against `ServerNotificationEventTransport` with a real feed |
| `textual-light`'s 2.8:1 is visible on the Scheduling/Evals screens (theme value and tcss consumer proved **separately**, not composited) | booting regenerates the CSS bundle and would corrupt this worktree | in a scratch profile, switch to `textual-light`, open Scheduling, sample `$ds-value-fg` against `$surface` |
| `css/widget_css.py` (1,014 lines) has no generator/checker divergence of its own | **read the 80-line module docstring only** | `pytest Tests/UI/test_widget_css_consolidation.py -q`, then read `iter_blocks`/`render_stylesheets` for any **write** to `css_dir` (a builder mutating its input would make `check_bundle_sync`'s call order significant) |
| `check_bundle_sync.main()` handles `build_screen_owned_sheets`' cross-split `AssertionError` | it catches only `FileNotFoundError` and `ValueError`; no overlapping split constructed | add an overlapping selector to two `SCREEN_OWNED_SPLITS` sheets in a scratch copy — expect a raw traceback instead of an `::error::` annotation (**still exit 1, so preflight stays correct; only the message degrades**) |
| `Home/dashboard_state.py` (1,500) and `Terminal/*` (~4,500) hold no further D1s | mechanical scan only (locks, threads, subprocess, timers, `run_worker`, dotted config — all clean); bodies not read | `pytest Tests/Terminal/ Tests/Home/ -q` plus a read of `io_actors.py`'s four `Lock()` sites for a lock-ordering cycle |
