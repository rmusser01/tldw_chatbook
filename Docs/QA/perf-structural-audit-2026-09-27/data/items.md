

# m-console-profile

## summary
Runtime measurements of the Console at origin/dev 840ed2ca58 (Python 3.12.11, Textual 8.2.8, Apple M5 Max with 18 cores, 211x44 viewport). The main checkout was never touched and the audit tree's git status is still clean.

**Headline.** One cost dominates every hot Console path: a warm `load_settings()` call still pays the full ADR-126 storage-admission handshake. That handshake is about 340 to 490 `openat` calls walking every path component, plus pinned-directory and registry reads, on every call.
- TASK-32804.1 fixed only `get_cli_setting`, which now costs 0.001 ms warm.
- `load_settings()` still costs 10.5 to 15.5 ms per call in isolation and 15.8 to 21.6 ms in the running app.
- `ChatScreen._provider_readiness_app_config` calls it:
  - 4 times a second at idle;
  - about 1.2 times per keystroke;
  - 320 to 360 times per first send;
  - plus some calls per visit.
- Share of main-thread busy time (sampled, main thread only):
  - about 100% at idle;
  - about 80% while typing;
  - about 65 to 70% during a send.

**Hot paths measured.**
- **Idle Console.** Main-thread CPU is 7.7 to 9.7% of a core; the whole process uses 12 to 15%.
- **Typing.** Echo latency is fast (median 2.7 to 4.1 ms), but post-echo work costs 21 ms per key (sampled) and 30 to 37 ms per key net in steady repetitions. On top of that, each typing pause fires a debounced settings-summary rebuild that stalls the loop for 250 to 415 ms.
- **First send.** Enter to provider dispatch takes 6.5 to 11.8 s wall across 6 runs, with the main thread about 85% busy. Where the time goes:
  - an admission storm in `_sync_native_console_chat_ui`, which has no derivation scope;
  - character-context publishes rebuilding the inspector state;
  - a cold RAG `ConfigProfileManager` built on the event loop (794 ms);
  - on the worker thread, sensitive-path resolution fanning out to about 17 guarded config accessors, twice per run (3.6 s wall).
- **Warm Console visit.** Composer is ready in about 200 ms wall (about 185 ms CPU).
  - The screen is now REUSED (TASK-31520, `reusable=True`), so the standing "re-mints 559 widgets" finding is stale.
  - 63% of the remaining visit cost is Textual's resume restyle of all 533 nodes (593 `Stylesheet.apply` calls).

**Environment caveats.**
1. The machine was heavily shared. `uptime` load averages were 5.95 at start and 18.13 at end; per run they were 14–25 (run3 25.8→22.7, send1 22.5→25.4, send2 21.2→15.8, cprof1 14.2→12.4, raw1 14.9→17.9, census 10.9→12.3). A prior run2 by this same agent label, about an hour earlier, ran at 16–35.
   - Wall-clock numbers are inflated, so I report main-thread `thread_time` CPU and deterministic call counts wherever possible.
   - About half of process time was system time (open() syscalls), which contends in the kernel under load.
2. **Path depth.** Admission cost grows linearly with path depth. A fit over three depths gives about 2.0 ms plus 0.71 ms per path component.
   - The probe profile sits at depth 12 to 14; a real `~/.config/tldw_cli` sits at depth 4, where the fit gives about 4.8 ms per call.
   - So admission-dominated probe milliseconds should be scaled by about 0.3 to 0.46 for production on this CPU. Call counts need no scaling.
3. **cProfile on 3.12** uses `sys.monitoring`, so it merges all threads and mis-nests async frames. Use its call counts, not its cumulative times. Attribution comes from a main-thread stack sampler (1 ms) plus a worker-thread sampler.
4. **The stock `Tests/Performance/run_console_mount_profile.py` is stale.** It fails with "profile condition did not settle" because the Console route is reusable now: `on_mount` no longer fires on each visit, so its hydration hook never runs. I used my own driver built on the harness's helpers (`_configure_isolated_profile`, `app_factory._build_test_app`, `_configure_native_ready_console`), with a class-patched fake `ConsoleProviderGateway` (60 chunks, 5 ms apart), the test network guard, and the null keyring.
5. **The three-turn harness was not run.** It is a campaign runner that creates git worktrees and needs a real endpoint.
6. **Only the first send could be measured.** Repeat sends were not measurable: `prompt_queue.dispatch` returned SENT (254 to 311 ms on the loop), but no message was appended and the provider was never called. The queue was DRAINING/RELEASED and the run status "completed". This may be a probe artifact or a real dropped-second-send bug; it is unverified and outside perf scope, but worth a look.
7. **Keystroke census pytest: 3 passed, 1 failed.** `test_keystroke_work_does_not_scale_with_transcript_length` fails on `context_estimate_max_rows` 0 vs 1: the test's strict equality is tighter than its own ≤1 bound, and this is not an O(N) regression. The census does not count config admissions, so it passes while every key pays one.

**Structural note: fix once where every caller routes through.**
- An unguarded warm-hit fast path in `load_settings` should collapse the per-key and send costs by most of their admission share, and shrink the idle cost.
- Adding `_console_derivation_scope()` to the remaining sync passes (`_sync_native_console_chat_ui`, `_sync_console_settings_summary`, the character-context presentation) caps each of them at one admission per pass.
- Only 3 derivation scopes exist in all of `chat_screen.py`.

## clean areas
- get_cli_setting warm path (TASK-32804.1 fix holds: 0.001 ms/call, 0 admissions, measured)
- Keystroke echo latency: median 2.7-4.1 ms, max 3-44 ms over 20 single keys per run (the key itself reaches the composer fast; the cost is post-echo)
- Transcript-length scaling on the keystroke path: census shows 0 messages_for_session, 0 snapshot/spend/cost/context rows per key at 400 messages (TASK-24300 fix holds)
- Template-default builds per key = 0 (TASK-24301 memo holds)
- Composer draft-edit sync IS inside a derivation scope (chat_screen.py:21965) - the memo works; the remaining per-key cost is the single warm load_settings admission it lets through
- Console screen reuse (TASK-31520): warm visits construct 0 widgets (Widget.__init__ = 0, mount = 0 during the visit) - the standing 're-mints 559 widgets' cost is gone at this revision
- _default_registry_factory (Tools/workspace_file_roots.py:447) is process-cached - its 142 ms Workspace_DB build is first-run only
- ui-stall-watchdog thread (Utils/ui_responsiveness.py:307) sleeps in time.sleep - appears as a busy Python leaf to the sampler but costs nothing
- Streaming chunk path itself: 60 chunks at 5 ms spacing finished streaming 250-900 ms after provider dispatch; no per-chunk admission hot spot was visible in the samples (the send cost is all before the provider call)

## census
### Isolation used for every Python run

- `S=.../scratchpad/audit/scratch/console-cpu-profile`
- `cd /Users/macbook-dev/Documents/GitHub/tldw-perf-audit`
- Environment:
  - `HOME`, `XDG_CONFIG_HOME`, `XDG_DATA_HOME`, `XDG_CACHE_HOME`, `TLDW_CONFIG_PATH` all point under `$S`.
  - `TLDW_TEST_MODE=1`, `HF_HUB_OFFLINE=1`, `PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring`, `TMPDIR=$S/tmp`.
  - `PYTHONPATH` is the audit tree.
- Interpreter: `.venv/bin/python` (3.12.11).
- The driver also installs `Tests.network_guard`, and the harness's `_configure_isolated_profile` re-points the profile to `$PROBE_PROFILE_ROOT` (under `$S`).

### Commands and run counts

| # | Command | Runs / samples | Load (before→after) |
|---|---|---|---|
| A | `python $S/bench_depth.py` with `PROBE_PROFILE_ROOT` = `$S/d`, `$S/d/a/b/c`, `$S/d/a/b/c/e/f/g/h` | 3 depths; 100 warm calls each, after 5 warm-ups | 15.3 |
| B | `python $S/wrap.py Tests/Performance/run_console_mount_profile.py --iterations 8` (stock harness under the network guard) | 1; **FAILED**: RuntimeError "profile condition did not settle" | 14.1→23.4 |
| C | `python -m pytest Tests/Performance/test_console_keystroke_work_census.py -p no:cacheprovider -p no:xdist -q -rA` with `PYTHONDONTWRITEBYTECODE=1` | 4 tests: **3 pass, 1 fail** (140 s) | 10.9→12.3 |
| D | `$S/run2.sh run3 PROBE_PROFILE_ROOT=$S/p3 MOUNT_ITERS=5 MOUNT_PROF_ITERS=2 KEYS=30 SENDS=3` (sampler) | 7 visits, 5×30 keys, 20 echo probes, 3 sends | 25.8→22.7 |
| E | `$S/run2.sh send1` and `send2` with `PHASES=send SENDS=3 SEND_VIA_PILOT=1` (send2 adds a trace of the send path) | 2×3 sends (only the first reaches the provider) | 22.5→25.4; 21.2→15.8 |
| F | `$S/run2.sh cprof1 CPROFILE=1 MOUNT_ITERS=2 MOUNT_PROF_ITERS=1 KEYS=30 SENDS=1` | cProfile `.pstats` for idle, visit, typing, send | 14.2→12.4 |
| G | `$S/run2.sh raw1 DUMP_RAW=1 MOUNT_ITERS=2 MOUNT_PROF_ITERS=1 KEYS=30 SENDS=1` | raw main-thread and worker stacks for subtree analysis | 14.9→17.9 |
| (H) | run2: the same driver (v1), by an earlier instance of this agent | 7 visits, 5×30 keys, 3 sends | 16.4→16.1 (avg over 15 min ~20) |

### Measured table (main-thread `thread_time` CPU unless marked wall)

| Metric | Values across runs | Attribution |
|---|---|---|
| Warm `load_settings()`, isolated (A) | depth 12: **10.53 ms**; depth 15: 12.77 ms; depth 19: 15.51 ms. Per call: 338 / 404 / 492 `_native_open`, 22 verified-parent walks, 17 pinned-dir walks, 1 admission | Fit: 2.0 + 0.71 ms × depth, so **≈4.8 ms** at the real `~/.config/tldw_cli` depth (4) |
| Warm `get_cli_setting()` (A) | 0.001 ms, 0 admissions | TASK-32804.1 fast path works |
| Warm `load_settings()`, in-app (D, G, H) | 21.6 / 15.8 / 17.7 ms | Same 338 opens per call |
| Idle Console, main CPU (5 s windows) | 9.74% (D), 8.11% (G), 7.66% (F), 7.84% (H) | 100% of main busy is `_poll_console_credential_readiness` at 4 Hz: 20 `load_settings` admissions / 5 s |
| Idle Console, process CPU | 14.8 / 13.2 / 12.1 / 12.7% | Workers: trace maintenance `run_batch` ≈176 ms wall / 5 s; 4 private-sqlite helper `fork_exec` / 5 s (F) |
| Warm Console visit, composer ready (wall) | D: 199, 190, 206, 371, 208, 209, 222 ms; G: 167, 205, 199 ms (median ≈205) | — |
| Warm Console visit, CPU to quiet | D: 181–362 ms (median 190); G: 157–196 ms | Of 195.8 ms sampled (G): Textual resume `update_node_styles` **123 ms** (593 `Stylesheet.apply`); `_reconcile_console_after_attach` → readiness admission 37.6 ms; `on_screen_resume` → unseen-marks transaction 15.9 ms |
| Typing, net CPU/key (30 keys, 50 ms gaps) | first rep 46–62 ms; steady reps 30–37 ms (D, F, G), 19–20 ms (H); burst 16–32 ms | Sampled (G): **21.4 ms/key** in `_handle_console_composer_draft_edit` → `_active_console_provider_model_display_uncached` → session stale-default → `load_settings` |
| Per-key call counts (D) | 1.23 admissions, 7.17 `_provider_readiness_app_config`, 1.0 `_build_console_control_state`, 4 `Stylesheet.apply`, 12.6 `query_one`, 7.1 `refresh`, 1 `_refresh_layout` | Deterministic |
| Echo latency, median / max (20 keys) | 4.13 / 43.8 (D); 3.22 / 6.5 (G); 2.73 / 3.3 (F); 3.05 / 9.9 (H) | Fast |
| Debounced spend refresh (0.2 s after the last edit) | `_sync_console_settings_summary`: 414.6 ms (H, sampled), 369 ms (F), 250 ms (E) per fire | No derivation scope: 13 readiness builds, 47 provider selections, 128 `admission_authority` per fire |
| First send, Enter→provider (wall) | 6966 (H), 11794 (D), 10779, 6542 (E), 10692 (F, cProfile), 8652 (G) ms | Main thread about 85% busy (G: 11.7 s of 13.9 s) |
| First send, main / process CPU | 5.2–8.0 s / 9.1–14.2 s | — |
| First send, main-thread counts (E) | 322 `load_settings` bodies, 402 admissions, 453 `acquire_storage`, 4,963 verified-parent walks, **76,820** `_native_open`, 708 `_provider_readiness_app_config`, 11 `_sync_native_console_chat_ui` | Deterministic |
| First send, main-thread breakdown (G, sampled) | `_sync_native_console_chat_ui` 7.1 s; `_poll_console_credential_readiness` 1.4 s; Enter dispatch 1.32 s; `_submit_draft_body` 1.0 s | Inside the UI sync: character-context refresh → inspector rebuild 3.0 s; inspector state 1.2 s; transcript sync 1.3 s. Inside dispatch: `ConfigProfileManager.__init__` **794 ms** + RAG simplified import 73 ms |
| First send, worker threads (G) | `run_reply` 4.56 s | `run_log.bind` → `is_within` → `resolve_sensitive_context` **3.6 s** (×2 per run; about 17 guarded accessors each; 63 `get_user_data_dir` per send in F) |
| Warm Enter dispatch (sends 2 and 3) | 311 and 254 ms on the loop | Readiness admission plus turn-context build |

### cProfile top cumulative (F; all threads merged; about 2–5× overhead)

- **Typing** (30 keys): `config_participants.operation` 1169 ms / 120 calls; `_provider_readiness_app_config` 1144 / 267; `acquire_storage` 956 / 128; `_ensure_active_console_session_settings` 891 / 121; `_handle_console_composer_draft_edit` 607 / 30; `spend refresh` 452 / 1; `_sync_console_settings_summary` 369 / 1.
- **Send**: `acquire_storage` 8892 / 865; `bootstrap.pinned_directory` 7887 / 18,476; `startup_permission` 6940 / 1,815; `_open_verified_parent` 5126 / 9,762; `_provider_readiness_app_config` 5113 / 802; `native_open` 4135 / 152,611; `get_user_data_dir` 3194 / 63; `ConfigProfileManager.__init__` 1003 / 1.
- **Idle** (5 s): `_poll_console_credential_readiness` 559 / 20; `trace_maintenance.run_batch` 243 / 4; `HelperLease.start` 165 / 4 (`fork_exec` ×4).
- **Visit**: `acquire_storage` 902 / 81; `_reconcile_console_after_attach` 881 / 3; `textual_css_fastpath.apply` 464 / 797; `Screen._on_screen_resume` 354 / 1.


# m-db-bench

## summary
I measured the main DB hot paths on scratch databases at realistic volume, with no sqlite_stat1 (production state, asserted). Every Python run used the isolation env from the task (scratch HOME/XDG/TLDW_CONFIG_PATH, TLDW_TEST_MODE=1, PYTHONPATH=audit tree). The app was never booted. The audit tree is unchanged (git status clean).

Three separate regressions dominate. Each one landed after the last perf review.

(1) Every top-level `db.transaction()` on 12+ DB owners now runs Backup_Recovery storage admission (`_core_operation`, then `acquire_storage`). That costs about 245 `open()` syscalls and 6–15 ms per transaction. A raw `BEGIN;SELECT 1;COMMIT` costs 1.4 µs. TASK-31502 measured 23 µs per transaction block on 09-04. This landed in b5251e9a6e (09-16).

(2) Every private SQLite connection open on macOS/Linux spawns a `python -I -S` helper subprocess (61a49de2e0, 09-07). That is 73 ms per open (60 min), against 0.06 ms raw and 0.55–0.63 ms in TASK-21127/21131.

(3) Some call sites close the connection after every call (`list_and_close` and `run_owned_db_call`), so they pay (2) on every call. `scope.list_conversations` is 107–115 ms with one helper spawn per call; with a held connection it is 11 ms. The 1 Hz legacy-trace tick is 79 ms and spawned a helper every tick in the probe.

Separately, the TASK-278 content-search rewrite (Done) left a correlated `EXISTS(... messages_fts MATCH ...)`, which re-runs the full-text query once per candidate conversation. Console browser and switcher-History searches take about 3 s for a rare term and about 70 s (count query alone) for a common term. The old LIKE shape took 154 ms, and an uncorrelated `IN (SELECT ... MATCH)` takes 3.7 ms or 91 ms.

The per-conversation paths are healthy. Opening a conversation (tree/messages), keyword batching and message counts are index-driven and sub-ms to 11 ms, even for a 2,000-message thread. The media v9 partial indexes keep browse/list at about 1 ms of SQL. Most of the remaining listing cost is the admission tax.

**Hot paths in scope:** Library snapshot (notes, media, conversations lists and counts), Console conversation browser (list and search), session-switcher History search, conversation open (`get_conversation_tree`), per-send persist (`add_message`), Library media browse and search, Library RAG keyword seam, and the agent/MCP `library_*` tools.

**Environment caveats:** the machine was heavily loaded by other sessions. `uptime` read load averages 18.46/24.09/24.04 before, 26.10 during the first bench, 12.2–25.1 across later runs, and 25.10/20.04/22.49 at the end. Absolute milliseconds are probably inflated 1.5–3x, with visible run-to-run drift (transaction cost 6.9 ms in one run vs 15.2 ms in another; `search_conversations_by_content('python')` 263 vs 485 ms). The ratios and the syscall and spawn counts are stable.

Page cache was warm for the median columns; the 'first' column is the closest thing to cold. I could not purge the OS cache without sudo, so true cold numbers are higher.

The admission and helper costs were measured with an unbound recovery-bootstrap in the scratch HOME. A real enrolled profile may do more registry work per admission (estimated, not measured).

Seed scripts and the benchmark harness (which traces the executed SQL and runs EXPLAIN on it) are in /private/tmp/claude-501/-Users-macbook-dev-Documents-GitHub-tldw-chatbook/15c5e43c-3dea-41a4-87ab-3b132387df58/scratchpad/audit/scratch/db-bench/. The seeded DBs are kept there as `dbs/chacha_seed.db` (523 MB) and `dbs/media_seed.db` (419 MB). Raw results are in `results_chacha.json`, `results_media.json` and `plans_chacha.txt`.

## clean areas
- ChatConversationService.get_conversation_tree (TASK-22206 shape): 0.32 ms for 42 msgs, 11.1 ms for a 2,000-msg conversation. One idx_msgs_conv_ts-driven query and no N+1.
- CharactersRAGDB.get_messages_for_conversation: 0.22–0.61 ms per 100-row page (ASC or DESC, idx_msgs_conv_ts); 11 ms for all 2,000 rows.
- get_library_conversation_messages SQL: all four statements are index-driven, and offset 1980 costs the same as offset 0. The ~7 ms total is almost entirely admission.
- count_messages_for_conversations(75 ids) 2.4 ms and get_keywords_for_conversations(75 ids) 0.33 ms: correct batch shapes, no N+1.
- Conversation list page query: ORDER BY last_modified DESC, id DESC is served by idx_conversations_archive with no temp B-tree for ORDER BY. list_all_active_conversations is about 3.5 ms per 1,000-row page. The rail-prune loop runs in a @work(thread=True).
- search_conversations_by_title 0.19 ms, search_notes (FTS) 0.4–5.5 ms, search_messages_by_content scoped to one conversation 2–130 ms, list_keywords/search_keywords under 0.2 ms.
- list_notes(100) 0.4 ms (idx_notes_last_modified ordered scan). It returns full content, but at 5k notes that is cheap.
- Media v9 partial indexes: the browse listing with no query uses idx_media_active_recent; sort by title and filter by type are covered; the count uses a covering index. SQL is about 1 ms, and the rest of the 6–10 ms is admission.
- Media fetch_all_keywords 0.24 ms, fetch_keywords_for_media_batch(50) 3.3 ms, fetch_media_for_keywords 1.5 ms, get_media_by_id 0.03 ms, list_media_items(50) about 0.3 ms per call.
- Per-send write SQL itself is modest. Message insert with all triggers (FTS, sync_log JSON, search-dirty) is about 0.8 ms per row in bulk; add_message's 9 ms is admission-dominated.
- Quiescent-cursor per-statement tax is 2.45 µs vs 0.61 µs raw. That matches TASK-31502's own numbers, so nothing new there.
- locate_conversation_page: 25 ms (window function over all active rows). Acceptable at 3k conversations on a rare path; recorded in the census only.

## census
### Setup (all runs used the isolation recipe)
```
S=/private/tmp/claude-501/-Users-macbook-dev-Documents-GitHub-tldw-chatbook/15c5e43c-3dea-41a4-87ab-3b132387df58/scratchpad/audit/scratch/db-bench
E="env HOME=$S/home XDG_CONFIG_HOME=$S/cfg XDG_DATA_HOME=$S/data XDG_CACHE_HOME=$S/cache TLDW_CONFIG_PATH=$S/cfg/config.toml TLDW_TEST_MODE=1 PYTHONPATH=/Users/macbook-dev/Documents/GitHub/tldw-perf-audit /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -u"
cd /Users/macbook-dev/Documents/GitHub/tldw-perf-audit
$E $S/seed_chacha.py $S/dbs/chacha.db      # CharactersRAGDB(explicit path); raw SQL inside db.transaction(); all triggers fire
$E $S/seed_media.py  $S/dbs/media.db       # MediaDatabase(explicit path); raw SQL + manual media_fts/keyword_fts rows (the code's own pattern)
$E $S/bench_chacha.py $S/dbs/chacha.db     # harness.py: 1 traced first run + 7 timed runs (5 for multi-second cases), median; EXPLAIN QUERY PLAN on every traced statement
$E $S/bench_media.py  $S/dbs/media.db
$E $S/probe_convsearch.py | probe_extra.py | probe_connect.py | probe_scope.py | prof_txn.py | prof_open.py  (on a copy)
$E -c "import asyncio,runpy,sys; sys.argv=['x','<db>']; asyncio.run(runpy.run_path('$S/probe_owned.py')['main']())"
```
**Volume:**
- ChaChaNotes (523 MB): 3,000 conversations (15% workspace-scoped, 20% character, 5% archived, 2% deleted) and 149,981 messages (5 conversations × 2,000 messages, the rest lognormal; ~470 chars avg). Also 5,000 notes (5% at 2–6k words), 300 keywords, 4,437 conversation keywords, 10,062 note keywords, 161,332 sync_log rows.
- Media (419 MB): 5,000 media rows (18.4 KB avg content, 3% trash, 2% deleted), 400 keywords, 12,551 media keywords, 33,395 MediaChunks plus 33,395 UnvectorizedMediaChunks.
- **sqlite_stat1 absent in both (asserted).**
- Seeding the 150k messages through all triggers took 123.6 s (~0.82 ms per message).

**Search terms:**
- rare: 'bribriexlo' — 647 message hits, 402 matching global conversations
- common: 'python' — 70,333 message hits, 2,413 conversations
- no-hit: 'zzqqxx'

### Per-statement / per-transaction / per-connection overhead (per_call_us: 200 warm-up calls, then the median of 5 samples of n calls; n=3000, or n=60 for transactions)
| operation | median | min |
|---|---|---|
| raw sqlite3 `SELECT 1` | 0.61 µs | 0.57 µs |
| ChaChaNotes quiescent `conn.execute('SELECT 1')` | 2.45 µs | 2.37 µs |
| `db.get_connection()` (warm) | 9.5 µs | 9.4 µs |
| `db.execute_query('SELECT 1')` (ChaChaNotes) / (Media) | 16.1 µs / 8.2 µs | 15.7 / 7.5 µs |
| raw `BEGIN; SELECT 1; COMMIT` | 1.37 µs | 1.36 µs |
| **`with db.transaction(): SELECT 1` (ChaChaNotes)** | **6,870 µs** (run 1 at load ~26: 15,241) | 6,573 µs |
| nested `db.transaction()` ×2 | 7,388 µs | 6,293 µs |
| **`with db.transaction()` (Media)** | **6,137 µs** | 5,187 µs |
| cProfile, 50 transactions | 12,250 `posix.open` = **245 open() per transaction**; ~100% of time under `_core_operation → acquire_storage` | |
| **`connect_private_sqlite` open+close** (median of 9) | **72.7 ms** | 60.3 ms |
| raw `sqlite3.connect` open+close | 0.06 ms | 0.05 ms |
| `python -I -S -c pass` (the helper's interpreter floor, median of 7) | 20.1 ms | 17.2 ms |
| fresh-thread `CharactersRAGDB.get_connection()` (median of 7) | 99.4 ms | 76.9 ms |
| raw connect + first query (schema parse, 656 objects) | 5.9 ms | 5.7 ms |
| `CharactersRAGDB(existing 523 MB)` constructor, 3 runs | 214 / 92 / 108 ms, then 141 / 70 / 76 ms | |
| `MediaDatabase(existing 419 MB)`, 3 runs | 95 / 52 / 46 ms | |
| fresh schema create: CharactersRAGDB / MediaDatabase | 1.29–2.36 s / 1.77 s | |
| admission contention, 1 thread × 20 transactions vs 4 threads × 20 | 27.2 ms per transaction (incl. connection open) vs 1,319 ms wall = 16.5 ms per transaction throughput (only 1.65x) | |

### ChaChaNotes methods (ms; median of 7 unless marked; 'first' = first traced call; 'adm' = storage admissions per call)
| method (UI caller) | median | min | first | stmts | adm | plan notes (no stat1) |
|---|---|---|---|---|---|---|
| svc.list_conversations(scope=all, 20) — Library snapshot | 8.28 | 7.36 | 45.4 | 4 | 1 | idx_conversations_archive, ordered; temp B-trees only for GROUP BY and the keyword ORDER BY |
| svc.list_conversations(global, 75, generic) — Console browser | 15.73 | 15.15 | 125.4 | 4 | 1 | same |
| svc.list_conversations(workspace, 75) | 13.65 | 11.96 | 50.1 | 4 | 1 | uses idx_conversations_archive, not the workspace index |
| svc.list_conversations(all, offset 2800) | 7.51 | 6.37 | 15.3 | 4 | 1 | |
| **svc.list_conversations(global, query=rare)** (5 runs) | **2,899** | 2,786 | 3,261 | 4 | 1 | **CORRELATED SCALAR SUBQUERY → SCAN fts VIRTUAL TABLE per conversation**, plus temp B-tree |
| svc.list_conversations(global, query=no-hit) | 200.5 | 193.6 | 260.4 | 2 | 1 | CORRELATED |
| svc.list_conversations(workspace, query=rare) (5 runs) | 107.7 | 101.9 | 125.2 | 4 | 1 | CORRELATED |
| **switcher History query_terms=(rare,)** (5 runs) | **3,110** | 3,001 | 3,029 | 4 | 1 | CORRELATED (per-term EXISTS) |
| switcher History query_terms=(no-hit,) | 234.6 | 227.4 | 305.4 | 2 | 1 | CORRELATED |
| COUNT only with the correlated FTS filter: rare / **'python'** / no-hit | 2,734 / **69,757** / 115 (n=1 each) | | | | | |
| same COUNT, uncorrelated `id IN (SELECT conversation_id … rowid IN (SELECT rowid FROM messages_fts MATCH ?))` (median of 5) | **3.7 / 91.4 / 1.5** | 3.6 / 88.3 / 1.3 | | | | LIST SUBQUERY + bloom filter |
| same COUNT, pre-TASK-278 correlated `content LIKE` (median of 3) | 154.5 | 143.2 | | | | |
| db.locate_conversation_page(recent / oldest) | 25.4 / 25.4 | 23.1 | 25.3 | 1 | 1 | MATERIALIZE all rows + 2 temp B-trees + automatic index |
| db.get_all_conversation_ids() | 1.43 | 1.31 | 1.53 | 1 | 0 | temp B-tree (ORDER BY id) |
| db.list_all_active_conversations(1000) × 3 pages | 10.46 | 9.75 | 10.8 | 3 | 0 | ordered index |
| db.search_conversations_by_title(rare) | 0.19 | 0.15 | 1.34 | 2 | 0 | FTS |
| db.search_conversations_by_content(rare / 'python') — Library RAG keyword seam | 5.4 / **263.3** (rerun 485) | 4.8 / 246 | 32.8 / **2,035** | 1 | 0 | 3 temp B-trees (GROUP BY, DISTINCT, ORDER BY); rewrite: python 485 → 191 |
| db.search_messages_by_content(rare, med / 'python', heavy) | 2.26 / 130.1 | 2.1 / 91.9 | | 1 | 0 | FTS then rowid lookup |
| db.list_library_conversations_page(20) — agent tool | 4.14 | 3.73 | 8.43 | 3 | 1 | |
| **db.search_library_conversations_page(rare / 'python') — agent tool** | **272.4 / 309.7** | 261.7 / 303.6 | 422.7 | 3 | 1 | CORRELATED `messages.content LIKE '%q%'` per conversation, in both COUNT and page (hit_2) |
| svc.get_conversation_tree(42 / 2,000 msgs) | 0.32 / 11.14 | 0.31 / 10.68 | | 4 | 0 | idx_msgs_conv_ts |
| db.get_messages_for_conversation(med 100 / heavy 100 / heavy 100 DESC / heavy 10k) | 0.22 / 0.48 / 0.61 / 11.14 | | | 1 | 0 | idx_msgs_conv_ts |
| db.get_library_conversation_messages(heavy, offset 0 / 1980) | 7.36 / 6.97 | 5.9 / 6.5 | | 4 | 1 | all indexed; about 0.5 ms of SQL |
| db.count_messages_for_conversations(75) | 2.39 | 2.28 | 5.88 | 1 | 0 | |
| db.get_keywords_for_conversations(75) | 0.33 | 0.30 | | 1 | 0 | |
| svc.effective_active_leaf(heavy) — fork path | 13.49 | 13.04 | | 2 | 1 | |
| db.list_notes(100) — Library snapshot | 0.40 | 0.37 | 11.26 | 1 | 0 | ordered index scan |
| **db.count_notes()** — Library rail | 3.56 | 2.92 | **172.6** | 1 | 0 | **SCAN notes** (`deleted` sits after `content`, so overflow pages are read); with an index on notes(deleted): COVERING INDEX, 4.96 → 0.26 ms |
| db.search_notes(rare / 'python') | 0.40 / 5.52 | | | 1–2 | 0 | FTS |
| db.list_library_notes_page(20) — agent tool | 8.24 | 7.14 | 10.1 | 8 | 1 | count does SCAN notes |
| db.search_library_notes_page(rare / 'python') — agent tool | 52.1 / 48.2 | 51.0 / 46.8 | 97.4 | 8 | 1 | SCAN notes with `content LIKE`, evaluated twice |
| db.list_keywords(100) / search_keywords | 0.17 / 0.12 | | | | 0 | |

### Media methods (median of 7; 5 where marked)
| method (UI caller) | median | min | first | stmts | adm | plan notes |
|---|---|---|---|---|---|---|
| svc.list_media_items(50) — Library snapshot (2 calls per timing) | 0.62 | 0.61 | 11.2 | 4 | 0 | |
| svc.list_media_items(50, include_keywords) | 3.86 | 3.66 | | 3 | 0 | |
| svc.search_media(browse, no query, 50) | 7.40 | 5.60 | 6.1 | 2 | 1 | idx_media_active_recent; correlated DocumentVersions has_analysis per row (indexed) |
| svc.search_media(browse, offset 2400 / title_asc / type=video) | 9.62 / 6.64 / 6.23 | | 176.7 / 14.7 / 8.2 | 2 | 1 | covered by the v9 indexes |
| svc.search_media(browse search, match_reasons) rare / 'python' / no-hit (5 runs) | 31.4 / 42.7 / 14.6 | 30.7 / 37.5 / 13.4 | **173 / 482** / 12.9 | 3 | **2** | FTS id-list + LIKE on candidates; keyword-reason probe runs in its own transaction |
| db.search_media_db(rare, relevance) (5 runs) | 30.0 | 28.9 | | 2 | 1 | temp B-trees for DISTINCT and ORDER BY |
| db.fetch_all_keywords / fetch_keywords_for_media_batch(50) / fetch_media_for_keywords | 0.24 / 3.26 / 1.47 | | | 1 | 0 | |
| svc.list_media_trash(50) — Library Trash | 10.36 | 8.22 | 39.7 | 2 | 0 | idx_media_deleted + table lookups for is_trash (after 18 KB content) + temp B-tree on trash_date |
| db.get_media_by_id | 0.03 | | | 1 | 0 | |
| db.get_distinct_media_types | 6.21 | 4.48 | | 1 | 1 | covering index; the cost is admission |
| db.get_all_active_media_ids | 5.79 | 5.13 | | 1 | 0 | idx_media_deleted + is_trash table lookups |
| db.list_library_media_page(20) — agent tool | 8.07 | 7.16 | | 3 | 1 | |
| db.search_library_media_page(rare / 'python') — agent tool (5 runs) | **143.0 / 70.6** | 141.2 / 69.5 | 184 | 3 | 1 | `content LIKE '%q%'` over about 92 MB of Media.content |

### Connection-churn and background probes
| probe | result |
|---|---|
| `scope.list_conversations(mode=local)` via ChatConversationScopeService (`list_and_close`) — Library snapshot / Console browser (median of 7) | **107.4 / 115.4 ms** (min 95.7 / 98.1); **7 helper spawns in 7 calls** |
| same query on a held connection (`svc.list_conversations`) | 11.0 ms (min 8.1) |
| 1 Hz tick body `await run_owned_db_call(db, LegacyTraceMaintenance.run_batch)`, steady state (9 ticks) | **79.4 ms per tick** (min 75.5); **9 helper spawns in 9 ticks**; child CPU **43.3 ms per tick** |
| `run_batch()` on a held connection (median of 9) | 8.18 ms (min 5.96), which is all admission |
| `db.add_message` (one streaming persist; median of 7) | 9.03 ms (min 8.5) |

Load averages: 18.46–26.10 at the start, 12.2–25.1 during runs, 25.10/20.04/22.49 at the end.


# m-idle-boot

## summary
Runtime probe: boot timeline and the idle background tax on audit tree 840ed2ca58. All measurements are warm boots of a scratch profile, headless Pilot `run_test(size=(211,44))` with Console as the default screen, plus one real-terminal boot in tmux. The isolation env was always on, and the real profile was verified untouched (find -newer marker returned nothing; config.toml md5 unchanged). Audit tree git status is clean.

ENVIRONMENT CAVEAT: the machine was heavily oversubscribed by other agents. It is an Apple M5 Max with 18 cores, and `uptime` load averages were 41→33, 29→50, 51→35, 36→27, 25→36, 23→19, 11→10, 10→15 and 22→22 across runs. Wall-clock numbers and the syscall-heavy costs scale about 2–3x with load. run8 and run9 (load 10–15) are the cleanest, so ranges are quoted as low-load to high-load.

BOOT: 7.5–9.1 s to `_ui_ready` at load 10–15, split as import 0.9–1.1 s, TldwCli() 2.4–3.1 s and mount→ready 4.1–4.9 s. At load 25–50 it is 17–20 s. A fresh first boot took 23.4 s. The first 5–12 s after `_ui_ready` are NOT idle: the process runs at 90–115% CPU, the loop is 30–75% busy, and there are 13–16 loop blocks of 82–900 ms.

Boot critical path: about 190k open() + 6.4k listdir + 5.4k flock syscalls before `_ui_ready` (158k of the opens on MainThread), plus 32 forked private-SQLite helper Python processes (12 of them synchronous on MainThread). The dominant cost is the Backup_Recovery storage admission. Every warm load_settings / get_runtime_config_snapshot / get_user_data_dir call re-walks the admission registry: 11 / 11 / 33 ms CPU per call at load 10, and 21 / 26 / 63–76 ms at load 22–35. That is 647 / 647 / 1,723 open() per call. The TASK-32804.1 warm fastpath covers only get_cli_setting, which measured 0.001 ms, so that fix holds.

IDLE TAX (steady state, 15–77 s after ready): 10.8–17.8% of one core in-process, plus 3.1–3.9% in reaped helper child processes. The loop thread uses 7–12% of a core. The loop runs 74–118 iterations/s with 53–300 ms/s busy, sees 330–590 involuntary context switches/s, and issues about 3,400 open() syscalls/s (about 2,600/s on the UI thread). A real-terminal run confirmed it: 11.3% of a core with Console visible and 0.44 helper spawns/s. With a modal covering Console it dropped to 4.7%, because the credential poll is gated on is_current.

Idle costs, ranked:
1. The 4 Hz Console credential poll: 9–33 ms CPU per tick, one load_settings admission each. In situ it measured 19–86 ms per tick, which means a loop stall every 250 ms. Known task TASK-32804.3; the "~8 ms" claim holds only at low load.
2. Legacy trace maintenance (about 1 Hz): each tick opens a fresh ChaChaNotes connection on a rotating executor thread. On macOS/Linux that forks a `python -I -S private_sqlite_helper_entry.py` helper, about 45 ms of child CPU each. Known task TASK-31501; this is new evidence.
3. The backup-maintenance monitor polls a 36-open filesystem probe at 10 Hz through to_thread. It is the largest source of loop wakeups. New finding.
4. The canvas-policy watcher runs at 4 Hz with no Canvas preview open.
5. Three 2 Hz widget timers, the 1 Hz heartbeat, and a 10 Hz watchdog thread. These are cheap per tick.

On the send path, every composer send calls `send_refusal_copy` → `default_emergency_stop_path()` → get_user_data_dir() on the loop. That measured 33 ms CPU at load 10 and 63 ms CPU / 96 ms wall at load 22, with 1,724 open() per send.

Structural note: nearly every finding reduces to one root cause, the per-call Backup_Recovery admission (config_participants.guarded / raw_participants._scope / storage_admission.acquire_storage) plus a helper spawn per new private-SQLite connection. One amortized, generation-checked admission and one retained helper or verdict cache would remove most of the boot syscall storm and the idle tax.

Nothing leaks at idle: no timers, workers or threads were created in the idle window, RSS was flat at 343.5→343.7 MB, and the Textual repaint and blink timers were paused. Headless mode skips terminal segment rendering, so headless render cost is a lower bound; the tmux run covers the real driver. TLDW_TEST_MODE gates only one chunking shim in production code, so the headless idle numbers are representative. The PYTHON_KEYRING_BACKEND null backend and HTTP(S)_PROXY=127.0.0.1:9 were added for isolation, and that proxy may change the local-server-discovery network timings.

## clean areas
- get_cli_setting warm fastpath (TASK-32804.1): measured 0.001 ms/call over 500 calls, 0 file opens; the fix holds
- get_canvas_execution_enabled / load_cli_config_and_ensure_existence warm: 0.001-0.004 ms/call, no admission
- Textual internal timers at idle: Animator, Screen._on_timer_update (both screens), 7 Input cursor-blink timers and the ConsoleComposerBar blink timer are all PAUSED; 0 idle ticks, so there are no idle repaints
- ChatScreen._poll_console_environment (10 s, chat_screen.py:16943): 0.003 ms/call, 2-3 ticks per window; fine
- TldwCli._record_ui_heartbeat (1 Hz, app.py:11943): ~0.05 ms/tick; fine (its watchdog thread is noted in F12)
- No leaks in a long idle: 0 timers, 0 workers and 0 threads created during the 20-30 s idle windows; RSS 343.5->343.7 MB, 346.5->346.7 MB, 344.0->344.1 MB
- Workers live at idle: only the scheduling coroutine worker ('run','scheduling'); the boot worker fleet matches the Tests/Performance/test_boot_worker_census.py allowlist (no unlisted names seen)
- SchedulerLoop 30 s poll (Scheduling/constants.py:15) does its DB work through _offload; only the estop path resolution is on-loop (F9)
- DBStatusManager periodic update is 120 s (db_status_manager.py:172); media cleanup / change-review retention are 86400 s; not idle costs
- Logging at idle: app log grew ~104 KB across 13 boots; stderr DEBUG volume negligible; not an idle cost
- ui-stall-persist thread blocks on queue.get (0 wakeups); chatbook-storage-admission thread blocks on Event.wait (0 wakeups)
- Parallel service-init pool (4 ThreadPoolExecutor threads): notes/media/prompts/providers 0.1-1.0 s each, overlapped off the main thread; not the critical path
- Screen pre-import thread (tldw-screen-preimport): 1.56 s wall / 0.60 s CPU, off-loop after ready; contends for the GIL during the post-ready tail but is by design

## census
### Probe setup (all runs)
Scratch dir `S=/private/tmp/claude-501/-Users-macbook-dev-Documents-GitHub-tldw-chatbook/15c5e43c-3dea-41a4-87ab-3b132387df58/scratchpad/audit/scratch/boot-idle-probe`.
- **Profile** `P=$S/pdone`, `cfg/config.toml` = `[general] users_name="perfaudit"`, `[splash_screen] enabled=false`, `[first_run] setup_completed=true`. All runs are warm re-boots of this profile, except smoke, which was that profile's first boot.
- **Launcher** `$S/run.sh`: `cd /Users/macbook-dev/Documents/GitHub/tldw-perf-audit && env -i PATH=/usr/bin:/bin:/usr/sbin:/sbin TERM=xterm-256color HOME=$P/home USERPROFILE=$P/home XDG_CONFIG_HOME=$P/cfg XDG_DATA_HOME=$P/data XDG_CACHE_HOME=$P/cache TLDW_CONFIG_PATH=$P/cfg/config.toml TLDW_TEST_MODE=1 PYTHONPATH=<audit tree> PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring HF_HUB_OFFLINE=1 HTTP_PROXY=HTTPS_PROXY=ALL_PROXY=http://127.0.0.1:9 PROBE_SCRATCH=$S PROBE_OUT=<json> /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python $S/probe.py`.
- **Guard:** probe.py exits unless every profile env var is under $S and keyring is null.
- **What probe.py does:** it patches the Textual WorkerManager._new_worker, Thread.start, Timer.__init__/_tick, BaseEventLoop._run_once, KqueueSelector.select and Handle._run. It then boots `TldwCli().run_test(size=(211,44))`, polls `_ui_ready`, keeps a 1 Hz CPU timeline across the settle window and measures the idle window. Idle measurements are getrusage SELF/CHILDREN, `time.thread_time` on the loop, psutil per-thread CPU and `top -l N -s 1 -pid`. Optional extras: a sys.addaudithook I/O census (`PROBE_AUDIT=count|site`), cProfile (`PROBE_MODE=profile`; process-wide on 3.12) plus a 20 Hz stack sampler, timing wrappers on TldwCli/ChatScreen methods (`PROBE_TIME_METHODS=1`), and post-idle micro-benchmarks run on the loop thread.
- **Real-terminal run:** `tmux -L perfaudit new-session -d -x 211 -y 44 "$S/app_env.sh"` (same env, `python -m tldw_chatbook.app`), then `top -l 61 -s 1 -pid` and a psutil child-process census. Two windows, 60 s and 45 s. Quit with C-q, then `kill-server`.
- **Run count:** 13 in-process boots (smoke, run1–run12, with run10/11 as short attribution runs) plus 1 real-terminal boot. `uptime` was recorded before and after each run in `$S/out/*.uptime`.

### Boot phases (seconds from probe start; import is measured after the probe's patches)
| run | mode | load before→after | import app | TldwCli() | mount→_ui_ready | total to _ui_ready |
|---|---|---|---|---|---|---|
| smoke | fresh profile, first boot | 18→– | 2.33 | 11.70 | 9.33 | 23.45 |
| run1 | plain | 41→33 | 2.15 | 6.38 | 10.02 | 18.67 |
| run2 | plain | 29→50 | 1.86 | 8.53 | 9.67 | 20.16 |
| run5 | plain+audit count | 25→36 | 3.26 | 4.59 | 9.31 | 17.27 |
| run7 | method timing | 23→19 | 1.49 | 3.96 | 4.46 | 9.99 |
| run8 | plain+audit count | 11→10 | 0.93 | 2.40 | 4.10 | 7.49 |
| run9 | plain | 10→15 | 1.08 | 3.13 | 4.86 | 9.14 |

The app's own STARTUP TIMING SUMMARY reports `Total initialization time` of 4.6–8.5 s, but basic_init + parallel_init are only 1.2–1.8 s. Roughly 60–75% of constructor time is untracked.

**Boot I/O before `_ui_ready`** (run10, audit): 189,995 open(), 6,421 listdir, 5,415 flock, 52 chmod, 30 sqlite3.connect, 32 subprocess.Popen. Of the Popens, 31 are private-SQLite helper forks (12 of them on MainThread) and 1 is croniter's `file -b`. The following tail up to 15 s adds 68,803 open() and 21 helper forks.

**Post-ready tail** (run9): the 1 Hz timeline shows proc 109/115/102/90% CPU and loop 51/30/75/74% for the first 4 s, then settles to about 9%. Loop blocks of 191, 136, 385, 333, 440, 267, 236, 708, 298, 82, 116, 278 and 288 ms were recorded.

**MainThread boot costs** (run7 method timing, wall / CPU ms):

| method | calls | wall ms | CPU ms |
|---|---|---|---|
| `_wire_watchlists_and_notifications_services` | 1 | 1,488 | 1,238 |
| `ChatScreen._provider_readiness_app_config` | 189 | 1,206 | 916 |
| `_build_console_inspector_state` | 4 | 918 | 733 |
| `_build_console_provider_selection` | 36 | 664 | 445 |
| `_active_console_provider_model_display` | 8 | 511 | – |
| `_current_console_rail_state` | 3 | 501 | – |
| `_active_console_settings_readiness` | 14 | 458 | – |
| `_wire_character_persona_services` | – | 288 | – |
| `_wire_chat_conversation_services` | – | 189 | – |
| `_wire_workspace_registry_services` | – | 171 | – |
| `_setup_logging` | – | 157 | – |
| `_wire_evaluation_services` | – | 144 | – |
| `_build_chatbook_db_paths` | – | 140 | – |
| `_wire_server_context_provider` | – | 138 | – |
| `_wire_library_collections_services` | – | 134 | – |
| `_construct_notes_sync_runtime_owner` | – | 126 | – |
| `_init_model_catalog_disk_store` | – | 121 | – |

In the first 10 s after ready: `_provider_readiness_app_config` 785 calls / 2,510 ms, `_build_console_inspector_state` 19 calls / 1,176 ms, `_current_console_rail_state` 16 calls / 1,161 ms, `_build_console_provider_selection` 150 calls / 715 ms.

### Idle tax (steady state, ChatScreen current, no input)
"other thr" = other threads, as % of one core. "children" = reaped helper-process CPU, as % of one core. "cred poll" is measured in situ.

| run | load | window | proc %core | loop %core | other thr | children | loop iter/s | loop busy ms/s | nivcsw/s | cred poll ms/tick |
|---|---|---|---|---|---|---|---|---|---|---|
| run1 | 41→33 | 20 s | 17.8 | 12.1 | 5.7 | n/m | 110.7 | 135 | 589 | 31.0 (80 ticks) |
| run2 | 29→50 | 20 s | 13.8 | 9.2 | 4.6 | n/m | 73.4 | 302 | 383 | 86.5 (63 ticks; skipped ticks) |
| run5 | 25→36 | 30 s (+47–77 s) | 14.7 | 10.0 | 4.7 | 3.1 | 106.8 | 182 | 482 | 42.4 (116) |
| run9 | 10→15 | 20 s | 10.8 | 7.0 | 3.9 | 3.1 | 118.5 | 83 | 389 | 19.3 (78) |
| run4 (cProfile) | 37→27 | 20 s | 19.9 | 13.1 | 6.8 | 3.9 | 104 | 220 | 1022 | – |
| tmux real driver, Console visible | 13 | 45 s | 11.3 (top 7–25%/s) | – | – | 0.44 spawns/s | – | – | 562 ctx/s | – |
| tmux, consent modal over Console | 16 | 60 s | 4.7 (top ~4%/s) | – | – | 18 spawns/60 s | – | – | – | poll gated off |

**Idle I/O** (run5 audit, 30 s): 101,228 open(), of which MainThread 77,467 (2,582/s) and executor threads 792/s. Also 3,038 listdir, 3,165 flock, 21 helper forks (0.70/s) and 20 sqlite3.connect.

**Idle loop handles** (run9, 20 s): to_thread 544, asyncio.sleep completions 509, monitor_app 382, self-pipe wakeups 291, canvas-policy watch 158, credential timer 78 (1,509 ms), 3 × 0.5 s timers 40 each, trace maintenance 37, heartbeat 20.

**Timers alive at idle end** (identical in all runs):

| timer | interval | site | cost per tick |
|---|---|---|---|
| credential poll | 0.25 s | chat_screen.py:16947 | 9–86 ms |
| `_update_overflow_hints` | 0.5 s | main_navigation.py:540 | – |
| `_sync_progress_counts` | 0.5 s | console_character_context.py:131 | – |
| `_sync_progress_count` | 0.5 s | left_rail.py:627 | – |
| heartbeat | 1 s | app.py:11943 | – |
| environment poll | 10 s | chat_screen.py:16943 | – |
| DB sizes | 120 s | db_status_manager.py:172 | – |
| one-shot | 65 s | – | – |
| media/change-review retention | 86400 s | – | – |

The three 0.5 s timers cost 4–18 µs each per call. Eleven timers were paused and produced 0 ticks.

**Non-timer idle loops:**
- `monitor_app`: 10 Hz, runtime_maintenance.py:689.
- `_watch_canvas_policy`: 4 Hz, console_runtime.py:3030.
- legacy trace maintenance: about 1 Hz, console_runtime.py:3248.
- scheduler: 30 s.

**Threads at idle (12–15):** MainThread, chatbook-storage-admission (Event.wait), ui-stall-watchdog (sleep(0.1) loop), ui-stall-persist (queue.get), and asyncio_0..9 (default executor, idle between rotating to_thread polls).

### Micro-benchmarks (post-idle, loop thread, CPU ms per call; load ≈10 run8 | ≈22 run12 | ≈35 run4)
| call | CPU ms (load 10 / 22 / 35) | syscalls per call |
|---|---|---|
| config.load_settings() warm | 11.0 / 20.8 / 21.9 | 647 open, 21 listdir, 19 flock |
| config.get_runtime_config_snapshot() | 10.9 / 20.5 / 25.8 | 647 open |
| config.get_user_data_dir() | 33.4 / 62.9 / 76.1 | 1,723 open, 51 listdir, 52 flock, 1 chmod |
| emergency_stop.default_emergency_stop_path() | – / 61.7 / – | 1,723 open |
| prompt_history.default_prompt_history_path() | 26.7 / 63.4 / – | 1,723 open |
| ConsoleChatController.send_refusal_copy (per send) | – / 62.6 (96.2 wall) / – | 1,724 open |
| storage_admission.acquire_storage()+close | 3.3 / 7.6 / – | 265 open |
| recovery_review._ordinary_operation (@unqualified LLM call) | 4.2 / 10.3 / – | 346 open |
| ChatScreen._poll_console_credential_readiness | 9.0 / 19.9 / 23.3 (33.0 in run3) | 647 open |
| storage._local_pause_requested (10 Hz poll) | 0.61 / 1.50 / 1.42 | 36 open, 2 flock |
| get_cli_setting / get_canvas_execution_enabled | 0.001 / 0.002–0.004 | 0 |
| croniter cold import (standalone, 3 runs) | 24.8 ms, incl. platform.architecture() `file -b` fork 7.5 ms | 1 Popen |


# m-ratchets

## summary
Runtime probe of the perf ratchets on pristine dev 840ed2ca58. Everything ran in the isolated audit tree, 38 Python runs in total.

RATCHET STATE
- The screen pre-import payload guard is RED: 554/500 modules, 409,566/378,740 LOC, and the library route alone is 125,111/123,319 LOC. No open task covers it. It went red silently because perf-guard.yml does not run it.
- Four ratchets are near-red:
  - boot CSS bytes 607,640/608,090 (450 B, 0.07% headroom)
  - CSS bare-type rules 273/274 in the guard, 274/274 on a real TldwCli boot
  - ui-ready census 1031/1033. One of 6 warm boots measured 1033.
  - import weight 681/686
- Green: keystroke census (all zeros), boot worker census, CSS source count (25 at boot, 40 after a tour, cliff at 64), destination-tour and seeded-Library budgets.

HEADLINE (new, measured beyond what the guards see)
The backup-recovery admission handshake (ADR-126, merged around 09-16) now dominates event-loop time. TASK-32804.1 only gave get_cli_setting a warm fast path (1 µs, verified). Two functions still pay the full handshake on every call:
- load_settings(): 9-10 ms per call even on a cache hit, 381 descriptor opens.
- get_user_data_dir(): 29-33 ms per call, 1,020 opens.

I wrapped config_participants.operation to count calls on the main thread (the event loop). Counts per phase (loads 14-25):
- boot to _ui_ready: 104 calls, 3.5-4.4 s. That is about 70% of a 4.1-6.3 s construct-to-ready. get_user_data_dir alone is 40 calls and 1.5-2.1 s, all from the _wire_* builders in TldwCli.__init__.
- first 1.5 s after ready: 133-134 calls, 2.9-4.1 s.
- idle Console: 20 calls per 5 s (the 0.25 s credential poll), 15-41 ms per tick, 6-16% of the loop.
- typing on a new chat: 1.1-2.9 calls per key, 26-73 ms of loop time per key.
- every Console visit: 56 calls, 1.1-1.3 s.

A cProfile of one warm boot shows 207k open() and 205k close() syscalls and 572 acquire_storage calls. Warm TTI (process start to _ui_ready, 6 boots, load 14-25) was 7.7-9.9 s wall and 8.1-9.3 s process CPU. The 09-04 review measured 2.84 s at load 7-9; that comparison is not paired.

SECOND HEADLINE: GC
No gc.freeze or GC threshold policy exists anywhere in the package. Full collections take 122-266 ms on the boot heap (564-570k tracked objects) and 326-422 ms after an 8-screen tour (1.38-1.63M objects). Automatic gen-2 collections fired 12 times in 24 screen visits, at 272-871 ms each, and 7 times during boot at 54-174 ms each. The pre-import pass causes 2 gen-2 pauses of 59/69 ms (none with gc disabled).

The keystroke census cannot see config admissions. It reports 0 while the same keys run 27-69 guarded load_settings calls. That is how F1/F2 landed without any red perf signal.

ENVIRONMENT CAVEATS
- The machine had 18 cores, 17 users and heavy concurrent load. uptime before the suite (22:59) was load 22.22/24.31/24.11; after the suite (23:13) 36.55/36.46/32.88; 11.7 to 53 during probes; 26.69 at the end (23:39). Absolute ms are inflated about 1.5-2x. Call and syscall counts, module/byte/rule counts and the isolated micro-benchmarks are the robust numbers.
- Suite: 816 tests, 13m45s, 32 failed / 784 passed. Only 1 failure is a real ratchet breach (pre-import). The others:
  - 1 stale meta-test (expects 6 CSS sources, 5 exist).
  - About 14 hit RecoveryRequired('raw_source_selection_changed'), raised at the module-scope APP_CONFIG = load_settings() at app.py:1110. This is the known harness baseline from the per-test env redirect. It blinds test_ui_ready_before_nonessential_startup_services_finish, the TTS/STTS lazy-init guards and the footer-token-timer guard.
  - About 15 three-turn-profile harness tests need a git object (eb8225a32f) or a worktree that the detached audit tree does not have. Environmental.
  - 1 RAG CLI host-isolation assertion: the CLI writes recovery-bootstrap admission gate files under HOME.
- run_console_mount_profile.py is broken on dev: 'profile condition did not settle', 2/2 attempts. So there is no mount-profile number; Console visit cost comes from my tour probes instead.
- Isolation: every probe ran with scratch HOME/XDG_*/TLDW_CONFIG_PATH and TLDW_TEST_MODE=1. Early on I ran one pytest --collect-only without my env vars; the root conftest had already redirected HOME/XDG/TLDW_CONFIG_PATH before collection. The real ~/.config/tldw_cli/config.toml mtime is unchanged (Sep 26 07:01). The audit tree `git status` is clean after all runs.

Scratch artifacts are in /private/tmp/claude-501/-Users-macbook-dev-Documents-GitHub-tldw-chatbook/15c5e43c-3dea-41a4-87ab-3b132387df58/scratchpad/audit/scratch/perf-ratchet/:
- suite output: perf.txt; child logs under childlogs/
- probe scripts: probe_*.py
- probe logs: config_hot_*.log, config_tour.log, config_typing*.log, tour_gc.log, boot_profile.txt, uiready_probe.txt, preimport_probe.txt

SUGGESTED PR GROUPING
- PR-A (F1): unguarded warm-hit fast path for load_settings. Also reorder the memo check in session.py:3874, and add config-admission counting to the keystroke census.
- PR-B (F2): memoize get_user_data_dir per config generation, or resolve it once in __init__.
- PR-C (F9): amortize storage-admission directory pins. Depends on the ADR-126 owners.
- PR-D (F3, needs an ADR per TASK-31966): gc.freeze after ready and after the pre-import pass, plus threshold tuning.
- PR-E (F4): defer the 72 new pre-import modules and add the guard to perf-guard.yml.
- PR-F (F5+F6): ScreenOwnedSplit for research, settings-theme and lab CSS, which pays down both CSS ratchets. Move the modal wide-tier rules out of the Console module.
- PR-G (F7+F8): residency paydown and ledger backfill; repair the blinded guards and the mount profiler.

## clean areas
- Console keystroke derivation census: messages_for_session/snapshot/spend/cost/context rows all 0, settings_readiness_builds 0/key (budget 3), template_default_builds 0/key (budget 0), census identical at 0 vs 400 messages (junit properties of test_keystroke_work_does_not_scale_with_transcript_length) -- the TASK-24300/24301 fixes hold
- get_cli_setting warm read: 0.001 ms median, 0 descriptor opens (2 runs x 40 calls) -- TASK-32804.1's _warm_config_cache_hit fastpath (config.py:6481) is effective for this entry point
- Boot worker/thread census (test_boot_worker_census.py): all started workers and threads within ALLOWED_BOOT_WORKERS/ALLOWED_BOOT_THREADS, expected sentinels present
- Textual CSS parse-cache cliff: 25 CSS sources at boot, 40 after an 8-destination tour (soft limit 56, LRU cliff 64) -- 16 sources of headroom
- UI latency guardrails: destination tour and seeded-Library open both pass the 10 s budgets (all probe arrivals 0.67-6.8 s under load 22-40)
- CSS fastpath equivalence tests (identical computed styles for every node; ancestor-class runtime follow) pass
- Heavy-import eliminations: torch/transformers/legacy feature windows not loaded by `import tldw_chatbook.app`; total sys.modules 1,772 vs 2,200 tripwire; app import 1.40-1.63 s CPU (8.0 s hang tripwire)
- Speculative voice latency gates and RAG citation provenance benchmark gates (non-harness cases) pass
- Pre-import pass per-route cost for routes other than library/ccp/schedules/stts/settings is small (<=40 ms CPU each)

## census
### Commands (all from `/Users/macbook-dev/Documents/GitHub/tldw-perf-audit`)

All commands used the isolation env:

```
env HOME=$P/home XDG_CONFIG_HOME=$P/cfg XDG_DATA_HOME=$P/data XDG_CACHE_HOME=$P/cache TLDW_CONFIG_PATH=$P/cfg/config.toml TLDW_TEST_MODE=1 PYTHONPATH=<audit tree> /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python ...
```

`$P` is a per-probe subdir of `$S=.../scratchpad/audit/scratch/perf-ratchet`.

| # | Command | Runs |
|---|---|---|
| 1 | `-m pytest Tests/Performance -q -p no:cacheprovider -p no:randomly --timeout=900 \| tee $S/perf.txt`. I ran without `-x` because it is a superset of the `-x` run. 816 tests, 13m45s, 32 failed / 784 passed. | 1 |
| 2 | `$S/probe_css_diff.py`: the guard's own census helpers, diffed vs `boot_budget_snapshots/boot_css_bytes.json` | 1 |
| 3 | `$S/probe_css_parse.py`: fresh `textual.css.Stylesheet` parse of the bundle, with and without 3 non-boot modules | 7 reps + 15 interleaved reps |
| 4 | `$S/probe_preimport.py 3`: fresh interpreter, same walk as the preimport census, plus wall and CPU per route | 3 |
| 5 | Pre-import pass under `-X importtime`, then a gc-callback on/off A/B | 1 + 1 + 1 |
| 6 | `$S/probe_uiready.py`: the guard's `_CENSUS_SCRIPT`, plus TTI and process CPU at `_ui_ready` | 1 fresh + 6 warm boots |
| 7 | `$S/probe_boot_profile.py`: cProfile of one warm boot to `_ui_ready` | 1 (+1 discarded: thread_time timer invalid) |
| 8 | `$S/probe_tour.py 2` and `$S/probe_tour_gc.py 3`: guardrail tour, CSS source/rule census, `gc.collect()` timing, automatic GC callbacks | 1 boot × 2 rounds; 1 boot × 3 rounds |
| 9 | `$S/probe_config_read.py`: warm-profile micro-benchmark, 40 calls per function | 2 |
| 10 | `$S/probe_config_hot.py`, `probe_config_tour.py`, `probe_config_typing.py`: wrap `Backup_Recovery.config_participants.operation`, count outermost main-thread calls per phase and caller | 3 + 1 + 2 (one on a fresh profile) |
| 11 | `Tests/Performance/run_console_mount_profile.py --iterations 4`, then `--iterations 1` | 2, both FAILED "profile condition did not settle" |

### Ratchet / guard state (pristine dev 840ed2ca58)

| Guard | Measured | Budget | Headroom | State |
|---|---|---|---|---|
| boot import weight (`tldw_chatbook.*` after `import app`) | 681 (suite + probe, identical) | 686 | 5 (0.7%) | green, near-red. Snapshot drift +16/−4. |
| total `sys.modules` after import | 1,772 | 2,200 | 428 | green |
| `import tldw_chatbook.app` | 1.40–1.63 s CPU / 1.61–2.62 s wall (3 runs) | 8.0 s tripwire | — | green |
| ui-ready module census (warm) | 1031 ×6, 1033 ×1 (7 warm boots incl. suite) | 1033 | 2 → 0 | at the edge. Fresh first boot: 1075. |
| boot-parsed CSS bytes | 607,640 B | 608,090 B | 450 B (0.07%) | near-red. +24,207 B since the 09-18 pin (583,433). |
| CSS fastpath bare-type rules | 273 (guard, 160×45) / 274 (TldwCli, 170×48) / 307 after tour | 274 | 1 / 0 | near-red. Button=176–177. |
| CSS sources (parse-cache cliff) | 25 boot / 40 after tour | soft 56 / cliff 64 | 16 | green |
| screen pre-import payload: modules | **554** | 500 | **−54** | **RED** |
| screen pre-import payload: LOC | **409,566** | 378,740 | **−30,826** | **RED** |
| screen pre-import payload: library route LOC | **125,111** | 123,319 | **−1,792** | **RED** |
| boot worker/thread census | all within allowlist | allowlist | — | green |
| keystroke work census | all counters 0 at 0 and 400 msgs; readiness 0/key; template 0/key | 3/key, 0/key | full | green, but blind to config admissions |
| console mount profile (`run_console_mount_profile.py`) | runner crashes | — | — | BROKEN 2/2 |
| ui latency guardrails: tour, CSS count, seeded Library | pass (tour child 58 s; library pass) | 10 s per destination | wide | green |
| ratchet meta `test_snapshots_are_real_not_hollow` | `assert 5 == 6` | — | — | stale test, RED |
| startup-perf `test_ui_ready_before_nonessential_startup_services_finish`, `tts/stts_handler_initializes_on_first_use`, `test_booted_app_arms_no_token_timer...` | `RecoveryRequired: raw_source_selection_changed` (app.py:1110) | — | — | harness-red (known baseline), guards blind |

### Probe measurements

| Metric | Value | Runs / load |
|---|---|---|
| warm TTI, process start → `_ui_ready` (census boot, 120×40, preimport off) | 7.70, 8.28, 8.39, 8.54, 9.36, 9.89 s wall; 8.09–9.32 s process CPU | 6 warm; load 14–25 |
| fresh-profile first boot TTI | 12.8 s | 1 |
| `load_settings()` cache hit (isolated) | 9.0 / 9.9 ms median; p90 11.8–12.6 ms; **381 descriptor opens per call** | 2 × 40 calls; load ~12 |
| `get_user_data_dir()` (isolated) | 29.0 / 32.7 ms median; **1,020 opens per call** | 2 × 40 calls |
| `get_cli_setting()` (isolated) | 0.001 ms, 0 opens | 2 × 40 × 2 keys |
| guarded ops on main thread, boot → ready | 104 ops = 3.5–4.4 s (get_user_data_dir 40 = 1.5–2.1 s; load_settings 61 = 1.2–1.6 s) | 5 runs |
| guarded ops, first 1.5 s after ready | 133–134 ops = 2.9–4.1 s | 5 runs |
| guarded ops, idle Console 5 s | 20 load_settings = 308 / 366 / 400 / 818 ms (1 per 0.25 s tick) | 4 runs |
| guarded ops, 24 keys on a new chat, credential poll stopped | 27 ops / 633 ms (fresh profile); 69 ops / 1,765 ms (warm) | 2 runs |
| guarded ops per Console visit | 56 ops = 1.09 / 1.34 s | 2 visits |
| guarded ops per Personas/LLM/Settings/Schedules visit | 1–2 ops, 39–185 ms | 1 |
| cProfile warm boot (all threads, profiler-inflated) | posix.open 207,141 calls; close 205,537; `acquire_storage` 572 calls 10.3 s cumulative; `_open_directory_component` 109,864 calls 7.0 s; `qualified_for` 1,144 calls 4.5 s | 1 |
| pre-import pass CPU (GIL-holding) | 900 / 776 / 713 ms; library 229–293, ccp 125–168, schedules 78–84, stts 60–73, settings 46–64 ms | 3 |
| pre-import pass, gc on vs off | 781 vs 631 ms CPU; 2 gen-2 pauses of 59.2 and 69.3 ms | 1 + 1 |
| full `gc.collect()` at boot heap | 122–129 / 174–266 ms (564–570k tracked objects) | 2 boots × 3 |
| full `gc.collect()` after tour | 326–363 / 375–422 ms (1.38M / 1.62M objects) | 2 boots × 3 |
| automatic gen-2 GCs during 24 visits | 12 pauses: 272, 870, 603, 646, 854, 772, 546, 721, 352, 382, 454, 467 ms | 1 (load 22–40) |
| automatic gen-2 GCs during boot | 7 pauses, 54–174 ms; gen-1 max 145 ms (627 gen-1 collections) | 1 |
| CSS bundle cold parse (422,385 B) | median 316 ms (min 230); minus research, settings-theme and lab (−58,136 B): median 287 ms, so **−30 ms per full parse** | 15 interleaved reps; load ~50 |
| screen arrival, tour probe (arrive ms; three probe runs, three rounds in the last) | Chat 1,889–4,230; Home 675–1,249; Library 679–4,494; Personas 1,166–4,110; Schedules 1,756–4,270; MCP 2,194–6,786 (slowest everywhere); LLM 716–2,267; Settings 905–2,490 | 3 runs; load 22–40 |

Environment: uptime was 22:59, load 22.22/24.31/24.11 before the suite; 23:13, load 36.55/36.46/32.88 after it; 23:39, load 26.69 at the end. The machine has 18 cores and 17 users were logged in.


# m-screen-tour

## summary
Runtime measurement of what each screen switch costs, on origin/dev 840ed2ca58 in the audit tree. The app was booted headless in the scratch profile, with the real profile left untouched (~/.config/tldw_cli/config.toml mtime unchanged) and the audit tree left clean (git status empty). The tour posted NavigateToScreen for library → settings → personas → mcp → notes (alias) → home → chat (Console). It ran 3 rounds in each of 3 separate processes: round 1 is the cold pass (n=3), rounds 2–3 are warm (n=6). Per switch it recorded wall time, process CPU, event-loop-thread CPU (thread_time on the loop thread), the longest single loop stall (a 2 ms heartbeat), DOM size, widgets built, query_one calls, sqlite connects (loop vs thread), GC pauses and RSS. Follow-up probes attributed the costs: a recorder of storage-admission call sites, a main-thread stack sampler, cProfile, leak-rate probes, a detached-widget census, and a gc.freeze experiment.

Main result: nearly all the remaining visit cost is synchronous storage-admission work (hundreds of directory open() calls per call) on the event loop, plus memory held after visits that inflates GC pauses.

1. **MCP (P0):** every visit makes 165–240 storage admissions on the loop, through MCP store reads that go through mcp_source_participants → recovery_activation.selected_path. That is 1.1–2.6 s of loop CPU and single stalls of 0.44–0.88 s.
2. **Console revisit (P0):** Console is now a reused instance and builds 0 widgets on a revisit, yet a revisit still costs 0.84–1.42 s of loop CPU. 116–127 of that comes from warm load_settings() calls, each 12–14 ms because the config-participants guard does the admission work even on a cache hit. The TASK-32804.1 fastpath covers only get_cli_setting. get_user_data_dir() costs 42–51 ms per warm call.
3. **Leaks:**
   - Settings (P0): every visit retains its SettingsScreen through a Textual Signal subscription it never unsubscribes. That is 10/10 instances, about +71k objects and +10.7 MB per visit.
   - Personas (P1): Textual's thread workers keep a reference to the last worker on each pool thread, which pins departed screens. 6/10 were retained, about +69k objects and +14 MB per visit; the ceiling is the pool size.
   - Home (P1): the reused Home screen fully recomposes on every visit, even after the user has left. The discarded widget trees stay pinned by its query_one cache, about +26 widgets per visit.
4. **GC (P1):** a full gen2 collection takes 133–151 ms on the 660k-object boot heap. Automatic gen2 collections landed inside about 1 in 3–4 switches, at 181–367 ms, and grow with the leaks. gc.freeze() after boot measured 0.0–0.1 ms. That is a global GC policy change, so TASK-31966 requires an ADR first.
5. **Screens that work well:** Library, Notes-alias and Home warm visits cost about 18–35 ms of loop CPU. Screen reuse (TASK-24452) works for Library.

**Environment caveats:** Apple M5 Max, 18 cores, Python 3.12.11, Textual 8.2.8, size 235x52. The machine was heavily loaded by other sessions' pytest and tldw_server processes. uptime load average was 5.86 before the first run, rose to 13.96–26.05 during the runs, and was 8.80 after the last. Wall times are therefore inflated, roughly 1.5–2×. Loop-thread CPU and the admission, widget and query_one counts are the robust figures.

**Method caveats:** cProfile in this interpreter recorded worker-thread frames too: ensure_actor_pack_recovery, which runs via asyncio.to_thread, showed up in its output. So loop attribution comes from a main-thread sampler and a per-call recorder that notes which thread each admission ran on. The keyring was forced to the null backend (PYTHON_KEYRING_BACKEND) as extra isolation. The first Console visit happens during boot, so the Console rows are the first and later returns to the boot instance, not a true cold visit.

## clean areas
- Library warm visit and the Notes route alias: reused instance, 0 widgets built, 26-36 ms loop CPU per visit (the TASK-24452 reuse pattern works here)
- Home warm arrival itself: 20 ms (the cost is the post-arrival recompose, finding F5)
- Library cold visit: 57 widgets and 2-3 loop admissions (10-17 ms); the remaining ~0.5 s loop CPU is mount and CSS for a first build; no finding beyond F9/F11
- MCPScreen instances are freed after navigation (0/10 retained)
- Screen stack stays at 2 on every switch; reusable instances stay installed as designed (chat, home, library)
- No sqlite3.connect on the loop thread during warm visits; loop-side connects happened only on cold Home (notifications DB, P3)
- Screen preimport thread finishes before the tour (waited explicitly); boot to _ui_ready 3.7-4.5 s under load
- get_cli_setting warm path: 0.002 ms/call measured, so the TASK-32804.1 fastpath is effective for that entry point

## census
### Screen switch / visit cost tour (origin/dev 840ed2ca58, audit tree, isolated profile)

**Setup**
```
S=/private/tmp/claude-501/-Users-macbook-dev-Documents-GitHub-tldw-chatbook/15c5e43c-3dea-41a4-87ab-3b132387df58/scratchpad/audit/scratch/switch-tour
mkdir -p $S/{home,cfg,data,cache,out}
printf '[general]\nusers_name = "perfaudit"\n[first_run]\nsetup_completed = true\n[splash_screen]\nenabled = false\n' > $S/cfg/config.toml
cd /Users/macbook-dev/Documents/GitHub/tldw-perf-audit && env HOME=$S/home XDG_CONFIG_HOME=$S/cfg XDG_DATA_HOME=$S/data XDG_CACHE_HOME=$S/cache \
  TLDW_CONFIG_PATH=$S/cfg/config.toml TLDW_TEST_MODE=1 PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring \
  PYTHONPATH=/Users/macbook-dev/Documents/GitHub/tldw-perf-audit \
  TOUR_ROUNDS=3 TOUR_OUT=$S/out/t{A,B,C}.json /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python $S/tour.py
```
- The `first_run` section is extra relative to the brief's recipe. It matches the canonical census config and keeps the wizard off.
- Run 0 (warmup.json, 1 round) was a discarded profile-warm boot.
- Runs A, B and C were timing runs, 3 rounds each.
- Extra probes:
  - `TOUR_PROFILE=...`: cProfile run.
  - `TOUR_ADMIT=1` (tour2.py, depth 3/5): storage-admission call-site recorder.
  - `TOUR_SAMPLE=... TOUR_SWITCH=0.0005`: main-thread stack sampler.
  - `leakrate.py` (LEAK_ROUTE=settings|personas|mcp, N=10), `leakrate2.py` (type-growth histogram), `orphans.py` / `orphans2.py` (detached-widget census and root chains), `leakchain*.py` (referrer chains), `bench_inapp.py` (warm config reads), `gcfreeze.py`.
- Size 235x52. Screen preimport was ON (the default outside pytest) and the tour waited for it to finish. uptime load average: 5.86 before, 12–26 during, 8.80 after, on 18 cores.

**Per-visit cost.** Median of 3 processes, range in parentheses. Cold is round 1 (n=3); warm is rounds 2–3 (n=6). loop-CPU is thread_time on the event-loop thread. "settle" is the last DOM change, widget construction or loop stall over 30 ms before 0.6 s of quiet.

| dest | pass | arrive ms | settle ms | settle loop-CPU ms | settle proc-CPU ms | max loop stall ms | DOM | widgets built | query_one | sqlite conn loop/thread | GC ms (worst gen2) | instance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| library | cold (3) | 399 (292-618) | 970 (845-1166) | 587 (503-613) | 1012 | 253 (207-442) | 57 | 63 | 109 | 0/15 | 182 (181) | new |
| settings | cold (3) | 708 (624-724) | 1132 (1010-1406) | 732 (667-989) | 887 | 308 (287-493) | 144 | 151 | 64 | 0/7 | 239 (231) | new |
| personas | cold (3) | 328 (265-408) | 1494 (1298-1561) | 842 (752-943) | 1320 | 238 (219-243) | 239 | 243 | 636 | 0/14 | 248 (224) | new |
| mcp | cold (3) | 124 (105-131) | 1501 (1114-1791) | 1444 (1079-1457) | 1474 | 574 (488-690) | 152 | 164 | 176 | 0/0 | 25 (223) | new |
| notes→library | cold (3) | 30 (24-33) | 30 | 28 (23-31) | 31 | 27 | 57 | 0 | 33 | 0/11 | 4 | reused |
| home | cold (3) | 249 (229-419) | 657 (512-819) | 391 (202-507) | 540 | 262 (159-291) | 55 | 109 | 71 | 2/8 | 236 (266) | new |
| chat (return to boot instance) | r1 (3) | 156 (111-167) | 1226 (979-1479) | 889 (790-1094) | 1157 | 478 (436-484) | 537 | 0 | 594 | 0/7 | 12 | reused |
| library | warm (6) | 34 (30-36) | 34 | 31 (26-34) | 35 | 30 | 57 | 0 | 29 | 0/13 | 4 | reused |
| settings | warm (6) | 204 (164-261) | 578 (477-751) | 256 (217-378) | 383 | 99 (79-125) | 144 | 151 | 63 | 0/6 | 14 | new |
| personas | warm (6) | 377 (278-577) | 1260 (1058-1527) | 824 (690-997) | 1189 | 328 (271-379) | 239 | 243 | 577 | 0/9 | 316 (352) | new |
| mcp | warm (6) | 127 (113-160) | 2292 (1458-2777) | 2203 (1420-2618) | 2244 | 684 (438-876) | 176 | 189 | 228 | 0/1 | 31 (296) | new |
| notes→library | warm (6) | 31 (27-73) | 31 | 31 (25-49) | 31 | 28 | 57 | 0 | 34 | 0/11 | 6 | reused |
| home | warm (6) | 20 (20-27) | 431 (292-681) | 129 (96-442) | 415 | 31 (22-343) | 55 | 54 (post-arrival recompose) | 42 | 0/8 | 11 (316) | reused |
| chat | warm (6) | 151 (128-164) | 1426 (1061-1859) | 1170 (837-1422) | 1411 | 661 (466-696) | 537 | 0 | 562 | 0/4 | 13 | reused |

Boot: `_ui_ready` 3.68–4.55 s; the app import took 0.94–1.07 s. The heap held 661k tracked objects and 341–345 MB RSS after boot settle; after 21 switches it held 1.34M objects and 489–494 MB.

**Storage admissions per visit** (TOUR_ADMIT recorder, warm round; one admission is about 600–1600 directory open() calls)

| dest | admissions total | on loop | loop wall ms | top loop site |
|---|---|---|---|---|
| mcp | 243 | 240 | 1570 | mcp_workbench `_tool_policy_inventory` ×95, `get_kill_switch` ×40, `effective_tool_states` ×34, `local_external_catalog`/`get_external_servers` ×42, `load_context` ×18 (all via `recovery_activation.selected_path`) |
| chat | 153 | 127 | 948 | `chat_screen.py:7700` load_settings via `_build_console_provider_selection_uncached` ×42, wiring.py:1488 ×38, `_active_console_settings_readiness_uncached` ×14, `_persisted_chat_defaults` ×10 |
| settings | 16 | 3 | 18 | `authoring.customized_count` ×2 (compose), `list_console_unseen_marks` ×1 (nav bar mount) |
| library / notes | 26 / 25 | 2 | 10-15 | `get_active_workspace` via `_library_onboarding_admission_key` |
| home | 18-22 | 1-5 | 16-303 (cold 2 connects 259 ms) | nav bar `recompute_console_attention`; notifications DB (cold) |
| personas | 34-51 | 1-7 | 5-51 (cold: `managed_service` → get_user_data_dir ×4) | |

cProfile of a Console revisit: 43,962 posix.open calls (1.1 s self time) and 152 acquire_storage calls. An MCP visit: 80,067 posix.open calls (2.0 s self time) and 254 acquire_storage calls.

**Warm config reads inside the booted app** (bench_inapp.py, 100 calls × 3 reps)

| call | ms/call | open() per call |
|---|---|---|
| `load_settings()` | 11.9–13.9 | 597 |
| `get_cli_setting()` | 0.002 | 0 |
| `get_user_data_dir()` | 43–51 | 1590 |

**Retention per visit** (leakrate.py: N visits of X, each followed by Home, then gc.collect)

| X | retained X instances | objects Δ | RSS Δ | full gen2 ms (base → after) |
|---|---|---|---|---|
| settings, N=10 | 10/10 | +708k | +107 MB | 151 → 329 |
| personas, N=10 | 6/10 | +690k | +143 MB | 131 → 335 |
| mcp, N=10 | 0/10 | +407k (from the Home half) | +77 MB | 148 → 324 |

orphans.py, 8 cycles:

| cycle | detached-but-alive widgets | notes |
|---|---|---|
| Library ↔ Home | +207 | 112 NavigationButton, 8 MainNavigationBar |
| Library ↔ Chat | +1 | |

The type histogram over visits 4→12 of MCP ↔ Home shows +96k `textual.cache.FIFOCache`, +13.7k `Strip`, +232 widget cores.

**GC:** thresholds (700, 10, 10). Full gen2 took 133–142 ms at boot settle, and 0.0–0.1 ms after `gc.freeze()` (657,429 objects frozen).


# pat-algorithmic

## summary
Cross-cutting algorithmic sweep of the whole tldw_chatbook package (2,541 .py files, 1.83M LOC) at audit tree 840ed2ca58. Method: untruncated rg census plus three AST detectors run on source only: (1) anti-patterns inside loops, (2) inner full-rescans nested in loops, (3) accumulate-and-rescan loops. The detectors produced about 1,300 candidate sites. I triaged every candidate on a data-scaled path by reading the code and tracing a real caller, and measured each reported cost with an isolated micro-benchmark (HOME/XDG/TLDW_CONFIG_PATH set to scratch, TLDW_TEST_MODE=1, audit-tree PYTHONPATH).

Overall health: the Console streaming core is sound. The store folds chunks into a list buffer and materializes it once per 0.2 s tick, the transcript row planner is windowed and linear, turn grouping and browser merges use sets or dicts, and the history trimmer binary-searches over memoized token estimates. The quadratic and whole-scan costs sit on paths that previous perf reviews did not cover:
- The Console model picker does O(M²) work per keystroke. That is ~10 ms at OpenRouter's measured 456 models, 38 ms at 1,000, and 90 ms at 2,000, against a raised cap of 4,096.
- World-info/lorebook matching runs one regex per key over ~9 KB of scan text on every send. That is 62 ms at 300 entries and 219 ms at 1,000, against 5–15 ms with a substring prefilter.
- The meeting sink rewrites and re-serializes the whole JSONL transcript per segment, which is O(n²). A 1,500-segment meeting costs 10.5 s of CPU live and 4.7 s at Stop.
- The Notes-sync identity fallback is O(off-path bindings × files) of sha256 work on the UI event loop: 229 ms at 200/3k and 2 s at 500/10k.
- The Console Terminal re-projects every cell of every session per frame: 11–20 ms per session per frame.
- The agent StreamGate re-scans the whole buffer per chunk: 267 ms per 50 KB reply and 843 ms per 100 KB.
- The fallback in-memory vector store does O(N) list lookups per add plus a per-row Python similarity loop: 5.2 s to index 10k documents and 161 ms per search, against 1.4 ms vectorized.
- Offset synthesis in chunking is O(chunks × words × run) when chunk text does not match the source words: 13 s for a 100k-word document.
- The Personas preview re-parses the full reply's markup on every chunk.
- The Notes editor counts words over the whole body on every keystroke, with a regex scan that is 4–5× slower than `split()`.
- The hosted-provider SSE decoder walks every byte in a Python loop: 38 ms per 1 MB stream against 0.6 ms for `splitlines()`.
- Visual and text compaction do a descending linear search that re-prepares, re-renders and re-tokenizes each candidate prefix, and the text-compaction planner runs on the event loop.

The known thinking-delta cost is confirmed not to be in ThinkingCapture: it measured 8–10 µs per delta at 100 KB. I found no new evidence on it, so it is not reported.

Structural notes:
- Several streaming consumers outside Console do eager per-chunk whole-text work instead of per-frame or list-buffer work (Personas preview, agent StreamGate, the hosted SSE decoder).
- Several planners use descending one-step searches where the quantity being searched is monotone, so a binary search or prefix sums would do.
- Membership dedupe on lists (`x not in list`) survives in catalog-sized model lists.
- Dead code seen during the sweep, which costs nothing because nothing imports it: `Subscriptions/baseline_manager.py`, `TTS/base_backends.stream_with_chunks`, and `chat_message(_enhanced).update_message_chunk` (0 callers).
- Adjacent memory note, outside this category: `NotesSyncFileSnapshot` retains `text` and `raw_bytes` for every vault file in the per-root observation-reuse cache.

## clean areas
- tldw_chatbook/Chat/console_chat_store.py append_stream_chunk/_fold_stream_buffer_without_persistence: list buffer, one join per 0.2 s tick (verified linear); _message_or_raise is dict-based; _recompute_active_path is O(path) per real append (2 per send), tool markers O(1)
- tldw_chatbook/Widgets/Console/console_transcript.py _transcript_rows/_flat_transcript_rows: windowed (pruned/hidden ids), linear; _console_markdown_body_ends_in_open_fence is O(body) per 0.2 s tick, self-documented and bounded; tool-diff display text cached per row
- tldw_chatbook/Chat/console_turn_grouping.py group_console_transcript_messages: single linear pass
- tldw_chatbook/UI/Console_Modules/workspace.py _merge_console_browser_rows and Workspaces/conversation_browser_state.py: set-based dedupe, linear build
- tldw_chatbook/Chat/console_history_budget.py bound_messages_to_window: binary search over memoized estimate_tokens (TASK-18602 cache), not O(n^2)
- tldw_chatbook/Chat/console_thinking_capture.py observe_thinking_delta: 8-10 us/delta measured at 100 KB (the known 0.2 ms/delta is elsewhere, no new evidence)
- tldw_chatbook/Chat/reply_sentence_sequencer.py and Audio/voice_phrase_sequencer.py: += buffers drained per sentence or capped by _MAX_* limits
- tldw_chatbook/Chat/llamacpp_think_filter.py: per-char probe bounded by tag length
- tldw_chatbook/Widgets/Console/console_side_chat_modal.py: per-chunk Static.update(markup=False) of a text capped at 100 KB; render coalesced per frame
- tldw_chatbook/Chat/console_prepared_request.py rewrite/build_console_request: inner scans are per-unit, not whole-transcript; system-row pop(0) is tiny
- tldw_chatbook/Agents/agent_runtime.py _detect_cycle (bounded deque), stream_prefix_verdict (lstrip returns self, no copy)
- tldw_chatbook/Utils/egress.py (bytearray accumulators), Tools/web_tool_impls.py (sniff join runs once)
- tldw_chatbook/Audio/dictation_service*.py (joins reset each cadence), Audio/voice_transcription.py (frame sum bounded by chunk frames), Audio/meeting_capture.py pcm_window (bounded ring)
- tldw_chatbook/LLM_Calls/hosted_chat_streaming.py _consume_line joins (per line, bounded); world_info/chat-dictionary list membership small
- tldw_chatbook/RAG_Search/simplified/citations.py, rag_service keyword spans, fusion/reranker: small k per document
- tldw_chatbook/Web_Scraping crawlers (list.pop(0) BFS bounded by max_pages, network-bound, visited is a set)
- tldw_chatbook/Terminal/screen_model.py scrollback (deque popleft), posix_backend bytearray trims
- tldw_chatbook/Widgets/Console/console_composer_bar.py undo stack eviction (bounded depth), wrap loop bounded by width
- tldw_chatbook/Widgets/Library/library_file_notes_workspace.py _build_folder_index (linear), _editor_changed (no whole-text work)
- tldw_chatbook/UI/Library_Modules/library_note_import_controller.py _bounded_note_diff (16 KB input cap)
- tldw_chatbook/Widgets/Persona_Widgets/personas_dictionary_tryit.py word_diff (button press, small samples)
- tldw_chatbook/DB/ChaChaNotes_DB.py _library_organization_for_notes (per-note lists), search_* matched_fields sorts (tiny)
- tldw_chatbook/Chat/trajectory.py derive/_strong_components (per-group sorts, O(E log E))
- tldw_chatbook/Workspaces/change_turn_tracker.py discover_baseline (queue of roots, small), Widgets/Console/console_workspace_files_modal.py (paged)
- tldw_chatbook/Chunking/engine/strategies/json_xml.py, propositions.py, sentences.py (re-sums and pop(0) only on flush or at bounded prefixes)
- Dead or unimported, so no cost: Subscriptions/baseline_manager.py (char-level SequenceMatcher), TTS/base_backends.py stream_with_chunks (O(n^2) bytearray slicing, 0 callers), Widgets/Chat_Widgets/chat_message(_enhanced).py update_message_chunk (0 callers)

## census
| Pattern (tldw_chatbook/, untruncated) | Sites | Files | Inside a loop (AST) | Hot and reportable after triage |
|---|---|---|---|---|
| `.pop(0)` | 49 | 37 | 31 | 1 (vector_store LRU, F7); the rest are bounded queues, BFS over small root sets, or crawler BFS |
| `.insert(0, …)` | 30 | 26 | 6 | 0 |
| `copy.deepcopy(` | 355 | 75 | 60 | 0 new (per-send payload deepcopies in hosted_provider_engine/moonshot/zai/qwencloud copy node trees, not strings; P3, not filed) |
| `sorted(` | 1168 | 452 | 264 (mostly `for x in sorted(...)` headers) | 0 |
| `.sort(` | 125 | 85 | 8 | 0 (diarization alignment per-segment sort folded into F16) |
| `"".join` / `''.join` | 337 | 222 | 145 | 0 re-join-growing-buffer-per-iteration on hot paths |
| `self.attr += <str/bytes>` (non-numeric), per-call | 38 triaged of 117 attr-iadd | 74 | – | 2 on per-chunk paths (F6 StreamGate `_buf`, F9 Personas `_partial_text`); 2 dead (`update_message_chunk`) |
| local `bytes +=` accumulators (`= b""` then `+=`) | 3 | 1 (audiobook_generator) | 2 | F14 (P3) |
| `x in <local list>` inside loop | 71 | 63 | 71 | F1 (model picker, per keystroke); F2 sibling (`match not in matched`, small) |
| list `.index()` in loop | 9 | 8 | 9 | F7 |
| list `.remove()` in loop | 55 (mostly widget.remove()) | 40 | 55 | F7 |
| difflib / SequenceMatcher | 14 files | 14 | – | 0 hot (baseline_manager is dead; monitoring_engine fixed by TASK-16839; others bounded or button-press) |
| AST inner full-rescan nested in a loop (Chat, DB, Library, Notes, Console, Agents, RAG, Character_Chat, Workspaces) | 346 | – | 346 | F4 (notes-sync identity fallback); rest are per-row sub-collections |
| AST accumulate-and-rescan loops (whole tree) | 328 | – | 328 | 0 hot (all flush-time joins or bounded chunks) |
| Descending linear `for n in range(avail, 0, -1)` planners | 3 | 3 | – | F12 (visual compaction), F13 (text compaction); third bounded (think-filter tag) |
| Per-char Python scanning of stream bytes | 1 | hosted_chat_streaming.py | – | F11 |
| Whole-document work per keystroke | 3 surfaces checked (Library note body, file-notes editor, composer) | – | – | F10 (note word count); file-notes editor clean |

Measured (isolated venv, Python 3.12):
- F1 picker: 10.0 / 37.7 / 90.4 ms per keystroke at M = 456 / 1000 / 2000 models.
- F2 world-info: 61.5 / 218.8 ms per send at 300 / 1000 entries (×3 keys, 9.4 KB scan text), against 5.5 / 15.3 ms with a substring prefilter.
- F3 meeting sink: 1,500 segments → 10.46 s live CPU and 4.68 s Stop-pass; 3,000 → 55.2 s and 26.8 s.
- F4 notes sync: 13 / 229 / 1,961 ms at (10, 200, 500) off-path bindings × (3k, 3k, 10k) files, against 1.8–6.3 ms for an index built once.
- F5 terminal: 11.1 ms (120×40) and 20.2 ms (200×50) per snapshot.
- F6 StreamGate: 24.6 / 266.8 / 842.8 ms per 20 / 50 / 100 KB reply (70% of it in `str.find`).
- F7 in-memory vector store: 474 ms / 2.7 s / 5.2 s to index 2k / 5k / 10k; 161 ms per search against 1.4 ms vectorized.
- F8 offset synthesis: 38 ms when chunks match against 13.3 s when they do not (100k words, 625 chunks).
- F9 Personas preview markup: 112 / 472 / 2,336 ms total for 2 / 4 / 8 KB replies.
- F10 note word count: 1.05 / 3.75 / 18.1 ms per keystroke at 50 / 200 / 1,000 KB (`split()` is 0.22 / 0.75 / 4.36).
- F11 SSE decoder: 38.2 ms per 1 MB against 0.6 ms for `splitlines()`.
- F12 visual compaction: 953 ms pre-render search at 100 units, plus about 8 ms per rendered page.


# pat-caching

## summary
Scope: a caching, memoization and repeated-parse census across all of tldw_chatbook (2,541 .py files, 1.83M lines, audit tree 840ed2ca58). Method: an AST census of every deepcopy, lru_cache/cache, module-level and instance dict cache, re.* call, json/toml parse, inspect.signature, MarkdownIt and ast.parse site. Each hot-looking site was then read and traced to at least one real caller. Micro-benchmarks ran isolated: scratch HOME/XDG/TLDW_CONFIG_PATH, PYTHONPATH pinned to the audit tree, and no import of config.py, app.py or MCP/server.py, because those write the profile or take storage admission at import.

Overall health: good. The known-dangerous shapes were already fixed with good patterns:
- The token estimator memo is keyed on (model, provider, len, hash) and bounded at 4,096 entries.
- ModelCapabilities has a per-pair cache.
- The Console registry display reads use a mutation-generation-keyed cache. This is the template to copy.
- The transcript signature cache is pruned on set_messages, and the background-effect frame cache is fixed.
- The web fetch, search and robots caches are bounded with TTL and earliest-expiry eviction.
- The theme catalog colour cache is bounded.
- The keyring read cache has a TTL.
- The research _accepted_parameters helper is keyed on class, not bound method.
- All 21 lru_cache/cache sites are bounded and appropriate.
- 273 distinct literal regex patterns are used inside functions, which fits under CPython's 512-entry re cache, so none of them recompile.

The remaining misses are structural: pure, deterministic derivations recomputed on every call.
1. Shipped and built-in themes run the AA hue-pinning colour-system generation at import. This costs 24.7 ms before first paint, measured.
2. The built-in MCP manifest is rebuilt by reading and parsing MCP/server.py three times per call. It is reached at least twice per agent send on the event loop, which is new evidence beyond the core review's Hub-screen framing.
3. The Console transcript re-parses the selected assistant reply with a fresh MarkdownIt 2-3 times per row-plan pass, only to find Canvas fences.
4. Lore and dictionary inputs are re-read from the DB, deep-frozen, thawed and re-processed on every send, with dynamic per-keyword regexes and no revision cache.
The rest is P3 hygiene: double log redaction, a per-SSE-chunk deepcopy chain, a settings reader that deep-copies the whole config for one scalar, a write-only dependency 'cache', and a duplicate lru_cache on a method.

Proposed PR groups:
- **PR-A, memoize deterministic derivations:** F2 (cache the server.py AST/manifest and definition_hash) and F3 (fence pre-check, a module MarkdownIt and a per-message memo). F9 and F10 ride along.
- **PR-B, boot:** F1 (lazy or baked theme pinning).
- **PR-C, lore/dictionary revision cache:** F4, reusing the _ConsoleRegistryDisplayReads generation pattern.
- **PR-D, logging:** F5 (redact once per record).
- **PR-E, hygiene:** F6, F7, F8.

## clean areas
- Utils/token_counter.py estimate_tokens: bounded memo (4096, clear-on-full) keyed (model,provider,len,hash); custom-tokenizer gate only on miss
- model_capabilities.py get_model_capabilities: per-(provider,model) dict cache on a process singleton; models.dev gap-fill uses its own memory cache (LLM_Provider_Catalog/models_dev_catalog.py)
- config.py get_cli_setting/load_settings: cache-hit fastpaths (no deepcopy on read); deepcopies are confined to write/rebuild paths
- Tools/web_tool_impls.py _fetch_cache/_robots_cache/_search_cache: bounded + TTL + locked eviction
- css/Themes/theme_catalog.py _COLOURS_CACHE: repr-keyed, bounded 1024
- UI/Console_Modules/workspace.py _ConsoleRegistryDisplayReads: mutation-generation-keyed cache (good template)
- Widgets/Console/console_transcript.py: _message_signature_cache pruned in set_messages; _body_wrap_table lru(4); diff display text memoized per row
- Widgets/Console/console_background_effect.py: per-frame grid cache (task-261 fix holds)
- Chat/console_provider_gateway.py _reasoning_metadata_cache (size-bounded) and _context_window_target_memo (clear >32)
- UI/Console_Modules/character.py expression spec cache (bounded); retrieval scope caches keyed per conversation
- Agents/agent_service.py _tool_protocol_cache (keyed on tool tuple, cleared on reset)
- Chat/stream_stall_watchdog.py _SESSION_TRACKERS (bounded); Library/library_browse_location _GENERATIONS
- Chat/console_chat_controller._apply_chat_dictionaries: final user message only, offloaded via asyncio.to_thread
- RAG_Search/pipeline_loader.py: singleton loader, TOML read once
- UI/Screens/trajectory_screen.py _poll_revision: revision-gated, rebuild in thread worker
- All 21 lru_cache/functools.cache sites (bounded; no fresh-object keys; research _accepted_parameters keyed on class)
- In-function literal regex: 393 calls / 273 distinct patterns < CPython 512-entry re cache (no thrash)
- Module-level re.compile (649 literal): 66.5 ms cold total across the WHOLE tree, 35 ms of it in lazily-imported Chunking/engine/strategies/code.py
- inspect.signature (34 sites): measured 8 us/call (bound method), 28 us for chat_api_call bind_partial per send -- not worth fixing
- Library snapshot clone (UI/Library_Modules/library_snapshot_cache.py): measured 2.1 ms per deepcopy of a ~300-record snapshot, twice per visit
- Console build_context_snapshot / _presented_message_snapshots deepcopies: inspector-open only; measured ~6.8 ms per 400 messages (plus captures)
- Agents/agent_stream.py fence gate: self._buf += chunk is O(n) per chunk (attribute, no in-place concat) but off-loop and ~ms total per reply
- ChatScreen._load_sidebar_state toml.load per construction: measured 0.14 ms parse (covered by TASK-1320 for the I/O placement)
- RAG_Search/simplified/embeddings_wrapper.EmbeddingsWrapper._cache unbounded (cache_size never enforced) but legacy test wrapper with no production caller
- Media_Creation/image_generation_service._generation_cache: written nowhere (dead attribute)
- Chunking engine LRUCache copy_on_access + template_runtime provenance deepcopies: ingest path, off-loop, per-chunk small dicts
- Widgets/Console/character_expression_avatar.py 30 fps: frames prepared once in thread; mosaic per frame change only while visible

## census
### Census (audit tree 840ed2ca58, AST over 2,541 files; counts untruncated)

| Pattern | Sites | Where it matters | Verdict |
|---|---|---|---|
| `copy.deepcopy` calls | **354** (AST; 355 by `rg -c`) in **75 files** | UI 70, Chat 51, LLM_Calls 51, TTS 36, config.py 33, Widgets 26, Event_Handlers 16, Agents 13, Chunking 11, Library 10, MCP 7, other 30 | Only per-chunk hot sites: 8 (hosted_chat `_filtered_*` ×5 plus wrapper `__next__` in moonshot, zai and hosted_provider_engine), **15.3 µs/chunk measured** (F6). Per-send: console_turn_context `__post_init__`/`_freeze` (12), MCP `_normalized_schema` and library descriptors (F2), prompt-transform freeze (F4). Per tool round: project_instruction_runtime (6) (F8). Everything else is on save, inspector, settings or ingest paths |
| `@lru_cache` / `@cache` / `functools.cache(...)` / `cached_property` | **21** (19 lru_cache + 1 `@cache` + 1 `functools.cache()` call) + 1 cached_property | maxsize 1..4096 | All bounded. 3 are on methods: `model_capabilities.is_vision_capable` (retains self, duplicates a dict cache, stale after `add_model_capability`: F10); `research_scope_service._accepted_parameters` (staticmethod keyed by class: fine); `windows_files` (process singleton, noqa). None are keyed on fresh objects |
| Module-level mutable dicts (`_x = {}`/dict()/OrderedDict()/defaultdict) | **66** | 7 named cache/memo: web_tool_impls ×3 (bounded+TTL), theme_catalog `_COLOURS_CACHE` (1024), ingest_capabilities `_INSTALLED_PROBE_CACHE`, token_counter `_ESTIMATE_CACHE` (4096), RepoMap (Third_Party) | No unbounded growth found. The id()-keyed dicts (trace service, visual identity, provider_setup) are capability registries with explicit pops, not caches |
| Instance dict caches named cache/memo | **58** | reviewed all | Bounded or invalidated except `EmbeddingsWrapper._cache` (legacy, no production caller) and `ImageGenerationService._generation_cache` (never written) |
| `re.compile` at module/class scope | **668** (649 literal) | cold compile of all literals = **66.5 ms total** (measured); `Chunking/engine/strategies/code.py` 35 ms (lazy import), `Subscriptions/monitoring_engine.py` 5.8 ms, all others ≤2 ms | Not a boot issue except via whichever modules are on the boot path |
| `re.*` inside functions | **511** (393 literal, 273 distinct; 118 dynamic) | dynamic hot-ish: `world_info_processor._keyword_in_text`, `Chat_Dictionary_Lib` whole-word patterns (per entry per send) | Literal patterns are served from the re cache (<512 distinct). Dynamic per-keyword patterns are folded into F4 |
| `json.loads` / `json.dumps` / `json.load` / `json.dump` in functions | **695 / 700 / 68 / 45** | top files: ChaChaNotes_DB 37, Evals_DB 32, console_chat_store 29, LLM_API_Calls 28 | Mostly row codecs. Hot repeated serialization: `definition_hash` json.dumps(sort_keys)+sha256 per tool, twice per agent send (F2) |
| TOML parses | **50** (tomllib 36, toml 14) | boot: `DEFAULT_CONFIG_FROM_TOML` 102 KB, **6.4 ms** once; ChatScreen sidebar 0.14 ms/visit | No per-keystroke or per-tick TOML parse found |
| `ast.parse` at runtime | **13** | `MCP/server.py:135` ×3 per manifest build, **2.40 ms each** (F2) | The only hot one |
| `MarkdownIt(...)` built per call | **3** | `console_message_actions.py:80` per transcript plan (F3), `console_speech_text.py:153` per speak (cold) | F3 |
| `inspect.signature` | **34** | 8 µs/call (bound method) measured | Not worth a finding |
| `optional_deps.check_dependency` | **71** calls | writes `DEPENDENCIES_AVAILABLE` but never reads it; a failed import costs 33 µs each time | F9 |
| Token-counting implementations | **21** `def count_tokens/estimate_tokens*` | canonical `Utils/token_counter.estimate_tokens` is memoized | Duplication is structural, not hot |


# pat-concurrency

## summary
Cross-cutting sweep of threads, event loops, executors and locks over all of tldw_chatbook at 840ed2ca58. Every thread, loop and executor creation site was counted with an AST pass (not a truncated grep), and every hot candidate was traced to a real caller. I ran safe micro-benchmarks under the scratch isolation env (HOME/XDG/TLDW_CONFIG_PATH in scratch, TLDW_TEST_MODE=1).

The most important result is a new cost amplifier. Since ADR-125 (2026-09-07), every new thread-local SQLite connection goes through connect_private_sqlite. That spawns a `python -I -S private_sqlite_helper_entry.py` subprocess (~60 ms) and adds a storage-admission handshake (~16 ms, ~257 posix.open). Measured: a ChaChaNotes open costs 75–91 ms, against 0.018 ms for a warm query. So every "fresh thread" or "connection per operation" pattern in the codebase now costs ~75 ms per occurrence. Before ADR-125, TASK-31504 measured a connect at ~0.44 ms.

Worse, two registries keep strong references to thread-local handles: the ChaChaNotes quiescence registry (base_db.py:157) and the Backup_Recovery participants registry (participants.py:378/550, a plain dict covering 11 repository types). So any per-call thread that exits while holding a handle leaks it permanently. Measured leak: +1 registered connection, +2 fds, and ~1.7 MB RSS per ChaChaNotes handle. Media DB leaks the same way (+2 fds and +1 live StorageLease per thread). The Notes sync executor docstring documents exactly this failure and fixes it locally with a per-thread persistent loop (TASK-23027). Other sites still have it.

Hot paths this hits on default config (agent_runtime=True, exchange_capture=True, trace_normalized_writes=True):
- **Console send, agent path.** Each run creates a new _ModelCallLifeline thread and a new event loop. As a result:
  - Each trace write (4 per model call) is an owned operation on a thread with no handle, so each one opens and closes a connection (measured 75.8 ms vs 4.8–8.5 ms warm) and closes with `PRAGMA wal_checkpoint(TRUNCATE)`.
  - Each run gets a brand-new httpx.AsyncClient: 33.7 ms to build with verify=True, then a cold TCP/TLS handshake to the provider.
- **Agent tool calls.** Every call runs on a new bare thread. DB-touching tools leak their handle and pay ~75 ms each.
- **Agent MCP tool calls.** These execute on the Textual loop, and their sync execution-log append costs 60–73 ms (9 admission handshakes, ~2,400 opens).

Secondary structural issues:
- The shared default executor (min(32, cpu+4): 12 slots on an 8-core laptop) carries whole agent runs, including approval waits that can be indefinite, and model installs, alongside ~1,000 short `asyncio.to_thread` offloads.
- File tools spawn a Python worker subprocess per call (~55 ms). Raw-shell commands spawn a spawn-context process per command (~150 ms).
- An app-wide RLock is held across connection open and BEGIN IMMEDIATE in the trace factory, which the plain-send path takes on the UI loop.

Minor hygiene: offloads that still block on `future.result()`, 10–20 Hz idle watchdog and fleet polls, fresh event loops per call in the RAG circuit breaker, and non-exclusive workers on screen resume.

Overall: the thread and loop plumbing is careful about correctness (bounded joins, abandon semantics, cancellation). The speed problem is lifetime granularity. Threads, loops, HTTP clients and DB handles are scoped per call or per run, where the underlying resources (helper-validated connections, TLS pools) are now expensive to create. The single highest-leverage fix is one connection owner per long-lived thread plus one app-lifetime model-call loop. Candidate PR groups:
- **(A)** Hold one ChaChaNotes handle per lifeline or settlement thread, and stop checkpointing on every close (F1, F8).
- **(B)** Close or reuse every core-repository handle opened by tool threads and asyncio.run bridges (F2, part of F12).
- **(C)** App-lifetime model-call loop plus a shared SSLContext (F3).
- **(D)** Amortise storage admission per connection or scope, and move MCP execution recording off the loop (F4, F5; tagged TASK-32804.1).
- **(E)** Dedicated executors for long-lived jobs, plus persistent tool worker processes (F6, F7).
- **(F)** Hygiene sweep (F9–F13).

## clean areas
- app.py screen pre-import threads (_schedule_initial_screen_preimport/_schedule_screen_preimport): paced by yield ratio, core-count throttle and navigation park; not first-paint GIL contention
- app.py App.__init__ 4-worker parallel init pool (pre-loop, boot-once, correctly timed)
- RAG_Search/ingestion_indexing.py indexer: one persistent thread and loop reused across batches (only the per-batch DB close is a churn site, listed under F1)
- Chat/console_voice_worker.py: one persistent voice loop thread
- Chat/console_chat_store.py _stream_persistence_executor: single persistent thread without owned-op churn
- Chat/console_generate_image.py LLM-context executor: bounded single worker with saturation refusal (good pattern)
- Web_Scraping/WebSearch_APIs.py and Writing/Research scope services: lazily built module-level executors (TASK-3220 fix holds)
- Notes/notes_sync_executor.run_worker_coroutine: per-thread persistent loop, the template to reuse for asyncio.run bridges
- Agents/execution_capacity.py ledger lock: metadata only, no I/O under the lock
- DB/transaction_observer.py global RLock: O(1) critical sections, no I/O
- Metrics/metrics.py registry lock (double-checked fast path) and Utils/token_counter.py estimate-cache lock: O(1)
- Tools/web_tool_impls.py fetch/robots/search cache locks and rate limiter (sleep not under a lock)
- Agents/agent_service.py _call_with_timeout join-slicing (worker.join(0.5) is not a busy-wait)
- Textual thread workers (run_worker(thread=True)/@work(thread=True)): pooled on the default executor and cancelled with their node
- TTS AsyncAudioPlayer (offloads play to its executor), speech playback progress loop (bounded by playback, exits on NoMatches)
- Change-review finalization/consent worker pools (fixed-size, long-lived)
- Terminal backend reader threads (per session, long-lived); Terminal 50 Hz monitor already tracked as TASK-31503
- Audio capture queues (unbounded but real-time producers with buffer-limit guards): recording_service, dictation, streaming_sink, diarizer

## census
## Thread / loop / executor / lock creation census

Source: AST pass over tldw_chatbook/ (untruncated). Paths are relative to the audit tree. Sites were classified by hand after tracing.

| Primitive | Call sites | Files | Lifetime classification / notable sites |
|---|---|---|---|
| `threading.Thread()` | 87 | 52 | About 38 per-call/per-op, ~34 long-lived service, ~10 shutdown/teardown, ~5 per-hold/retire. **Per-call hot:** `Agents/agent_service.py:1995` (1 per tool call), `:5818` (1 per fleet child); `Chat/console_agent_bridge.py:2446` (1 per run + 1 per child, each with its own loop); `Tools/workspace_tool_executor.py:857/878/957/1094` (4–7 per file-tool call, plus a subprocess); `Tools/raw_cli_executor.py:887/892/973/1247` (per command, plus a spawn process); `Agents/agent_worktree_git.py:94` and `Tools/git_tool_impls.py:267` (2 per git op); `Chunking/.../ebook_chapters.py:158/300/340` (1 per regex, ReDoS guard); `TTS/audio_player.py:421` (per playback); `app.py:4776` (per STT job); `Chat/console_chat_controller.py:15113` (per approval round); `LLM_Calls/anthropic_subscription.py:144` (every 5 s while polled); `Backup_Recovery/storage_admission.py:498` (per admission hold when no lease is live) |
| `asyncio.run()` | 66 | 34 | All run in worker threads. Each call builds a fresh loop plus a fresh default executor; inner `to_thread` threads die at loop exit and leak their DB handles. Hot sites: `Agents/tool_catalog.py:1593` (per builtin tool call), `Agents/library_rag_tool_provider.py:395` (per call), `UI/Console_Modules/retrieval.py:90` (per session switch: to_thread → asyncio.run → run_in_executor), `Chat/console_agent_bridge.py:5954/6211/6309...` (per run / $skill), `UI/Screens/chat_screen.py:20477/20512` (per attachment), `Agents/run_hooks.py:439` (per hook). Template fix exists: `Notes/notes_sync_executor.py:95` |
| `asyncio.new_event_loop()` | 9 | 9 | Long-lived per thread: indexer, voice worker, notes-sync, voice child. Per run: lifeline (`console_agent_bridge.py:2445`). Per call: `RAG_Search/simplified/circuit_breaker.py:243` (each embedding batch), `health_check.py:425`. Never closed: `Web_Scraping/Article_Extractor_Lib.py:230` (thread-local loop per pool thread) |
| `loop.run_until_complete` | 8 | 6 | Same owners as above |
| `ThreadPoolExecutor()` | 24 | 19 | 17 long-lived or lazy singletons. 7 per call (`with ThreadPoolExecutor(1)` + blocking `.result()`): `simple_cache.py:489/777/888`, `local_media_reading_service.py:4486`, `Backup_Recovery/staging.py:1055`, `publication.py:2426`, `app.py:8021` (boot, fine) |
| `ProcessPoolExecutor()` | 1 | 1 | `RAG_Search/parallel_processor.py:185`, per call (cold) |
| multiprocessing `Process`/`Pool` (spawn) | 5 | 5 | `raw_cli_executor.py:1184` (per command), STT executor, ingest parse pool, parakeet worker |
| `subprocess.Popen()` | 28 | 23 | Includes the **private-SQLite helper per connection open** (`DB/private_sqlite_process.py:366`) and the workspace worker per file op (`workspace_tool_executor.py:284`) |
| `asyncio.to_thread()` | 980 | 213 | Shared default executor, min(32, cpu+4) = 22 here, 12 on an 8-core laptop. Also carries whole agent runs (`console_chat_controller.py:6607`) and model installs (`llm_screen.py:2224`) |
| `loop.run_in_executor()` | 49 | 16 | 34 target the default executor (`None`) |
| `run_worker(coroutine)` / `run_worker(thread=True)` | 833 / 70 | 130 / 36 | On-loop / pooled. 66 of 903 lack both `exclusive` and `group`; 31 of those are in `on_*`/`action_*` handlers |
| `@work(thread=True)` / `@work(async)` | 128 / 55 | 25 / 25 | Pooled / on-loop |
| `app.call_from_thread()` | 321 | 56 | Blocking handoff from worker to UI |
| `run_coroutine_threadsafe()` | 11 | 8 | Per agent model call onto the lifeline; **per MCP tool call onto the Textual loop** (`Agents/mcp_tool_provider.py:1344`) |
| `threading.Lock/RLock/Condition/Event` | 233 / 138 / 19 / 125 | 146 / 108 / 17 / 67 | 79 module-level (global) locks. Held across I/O: `Chat/console_trace_runtime.py:92` (RLock across connection open + BEGIN IMMEDIATE), `app.py:775` (known, bounded), `anthropic_subscription.py:52` (keychain subprocess) |
| `asyncio.Lock/Event/Semaphore/Queue` | 110 / 45 / 12 / 9 | 74 / 33 / 11 / 9 | Not reviewed individually |
| `queue.Queue()` | 24 | 18 | 9 unbounded: audio capture ×6, `ingestion_indexing.py:965`, `git_tool_impls.py:258`, `remote_worker_bundle.py:3276` |
| `threading.Timer()` | 4 | 4 | `UI/Console_Modules/raw_cli.py:605`: one Timer thread per repaint window while output streams |
| `time.sleep()` | 56 | 43 | Poll loops: fleet waits at 20 Hz (`agent_service.py:3799/3991/6447`), UI watchdog at 10 Hz, admission hold wait at 100 Hz (`storage_admission.py:936`), git reader at 100 Hz |
| `await asyncio.sleep(≤0.1)` in a loop | 83 | ~40 | On the UI loop: `MCP/unified_control_plane_service.py:3628` (200 Hz during a tool test), `llm_screen.py:606/2303` (100 Hz while a modal is open), `console_chat_controller.py:5593` |
| `set_interval()` | 62 | 51 | Covered by the polling sweep |
| Owned-op DB connection sites (`operation_owned_connection` / `run_owned_db_call` / `_run_owned_chat_db_operation`) | 52 | ~16 | Each opens and closes a handle when the thread has none: ~75–90 ms, including the helper subprocess and a `wal_checkpoint(TRUNCATE)` on close |

### Measured primitives (this machine, isolated env)

| Operation | Cost |
|---|---|
| Thread spawn + join | 0.21 ms |
| `asyncio.run(noop)` | 0.67 ms |
| `httpx.AsyncClient(verify=True)` | 33.7 ms (3.3 ms with a shared SSLContext) |
| New ChaChaNotes connection | 75–91 ms (helper spawn ~60 ms, admission ~16 ms, ~257 `posix.open`); a warm query is 0.018 ms |
| Owned trace write on a thread with no handle | 75.8 ms (4.8–8.5 ms with a handle) |
| One leaked ChaChaNotes handle | +2 fds, ~1.7 MB RSS |
| Media DB bare-thread open | 55 ms, +2 fds, +1 live StorageLease |
| ChaChaNotes `transaction()` admission | ~4.8 ms, 245 opens |
| `MCPExecutionLog.append` | 60–73 ms median (p90 ~106 ms), ~2,400 opens, 9 admissions |
| Spawn-child import of `raw_cli_executor` | 112 ms |
| Workspace worker interpreter + import | ~55 ms |


# pat-config-reads

## summary
Scope: config, settings and credential reads on hot paths across all of tldw_chatbook, audited at dev 840ed2ca58. The TASK-32804.1 fastpath works. A warm get_cli_setting costs 1.4 µs (dotted form 1.8 µs) and load_cli_config_and_ensure_existence costs 1.3 µs, measured with 10k timeit calls in an isolated profile. The 403 get_cli_setting call sites are therefore not a cost any more.

The finding is structural: the fastpath covered only one of the 16 functions wrapped by `@_config_participants.guarded`. The others still run the full ADR-126 storage-admission handshake on every outermost call, even on a warm cache: load_settings, get_runtime_config_snapshot, get_user_data_dir (and so all 16 get_*_db_path) and get_model_cache_dir. The handshake is `operation()` → `raw._scope`, which re-walks about 39 (load_settings) and about 104 (get_user_data_dir) verified-parent chains with fresh `openat` calls. The cost grows with path depth; the per-call measurements below are at the audit's 13-level scratch path:
- **load_settings:** 609–726 os.open and about 28k Python calls, 17–28 ms. At a real profile depth that is about 297 opens, roughly 8 ms. TASK-32804.3's author measured the same 8 ms independently.
- **get_user_data_dir:** 1,620–1,932 opens and about 126k Python calls, 67–174 ms. About 788 opens and 20–30 ms at a real profile depth.
- **Nested reads:** reads inside an outer `operation()` are free (0 extra opens measured). The whole cost is paid per outermost call.

The same keystone cost lands on four hot paths:
- **Console loop (F1):** the 4 Hz credential poll at idle, every composer keystroke, the 5 Hz run tick (whose explicit `operation(config)` wrapper pays it even for warm reads), each send, and Settings compose.
- **Boot (F2):** the TldwCli.__init__ _wire_* chain. An isolated boot to _ui_ready issued 263,299 os.open calls and 797 config admission operations, including 59 get_user_data_dir and 212 load_settings calls. get_user_data_dir alone accounts for about 100k of those opens.
- **Agent tools and @-references (F3):** resolve_sensitive_context calls get_user_data_dir 19 times. That makes every agent file/git/patch tool call and every @-reference token cost about 0.3–0.9 s. Measured at scratch depth: ReadFileTool on a 2-byte file took 873 ms and 32.9k opens; expanding 3 @-references took 2.7 s and 94k opens.
- **Message actions (F4):** retry, regenerate, edit, summarize, restore and queue dispatch each pay one get_user_data_dir on the loop.

Everything else checked is healthy. Provider-readiness math is cheap (get_provider_readiness about 10 µs, build_console_settings_readiness about 0.6 ms). TASK-24454 and TASK-32804.3 name readiness recompute, but the real cost is the load_settings admission that feeds it. Keyring reads are TTL-cached. No TOML or JSON config parse sits on a per-keystroke or per-tick path.

Suggested PR grouping:
- **PR-A (low risk, largest win):** add lock-free warm fastpaths to load_settings and get_runtime_config_snapshot, mirroring `_warm_config_cache_hit`. Drop the explicit `operation(config)` from `ChatScreen._run_console_config_sync` on warm reads, keeping the storage-pause deferral. This closes TASK-32804.3 AC#1 without touching its revision gate, and the leftover of TASK-32804.1 AC#2.
- **PR-B (low risk):** make resolve_sensitive_context resolve the user-data directory once instead of 19 times. Thread one context through local_tool_impls.read_file and console_references, and make `_resolve_sandbox_config` compute its default lazily.
- **PR-C (medium risk, needs the private_paths/ADR-126 owner):** memoize get_user_data_dir per config generation with a single lstat identity re-check. This removes the boot tax and the F4/F5 costs.
- **PR-D (P3 hygiene):** resolve the scheduler's paths once at construction, and replace the per-request snapshot deepcopy in the LLM handlers.

Measurement caveats: numbers come from isolated probes in a sandboxed HOME (config-reads scratch dir); the machine load average was 20–38 during the runs. Treat the ms figures as ranges; the open counts are deterministic.

## clean areas
- get_cli_setting / load_cli_config_and_ensure_existence warm path: 1.4 us / 1.3 us per call measured (TASK-32804.1 _warm_config_cache_hit is live on dev 840ed2ca58); 403 AST call sites are not a cost; the throttled _external_edit_detected stat is fine
- Provider readiness math: get_provider_readiness ~10 us/call measured; build_console_settings_readiness ~0.6 ms (boot profile) -- the per-keystroke readiness cost named in TASK-24454 is the load_settings admission that fetches its input, not the readiness logic
- Settings screen provider readiness (settings_screen.py:13700 _provider_readiness_app_config reads in-memory app.app_config; all 9 get_provider_readiness sites pass background_credentials=True)
- Home readiness: fresh load_settings runs in the content-snapshot thread; compose uses in-memory config (TASK-31805)
- Keyring: server credential store 5 s TTL (TASK-32922), media-gen keyring_get 10 s TTL + single-flight (TASK-32924/32926), Claude-subscription keychain 5 s TTL with a background snapshot for UI, citation fingerprint key loaded once per service build, Canvas web token only for remote serve, Backup_Recovery/credentials per-record store is a cold backup path
- Streaming per-chunk paths (Widgets/Console/console_transcript.py, Agents/agent_stream.py, Chat/console_agent_bridge.py deltas, UI/Console_Modules/hands_free.py/realtime.py deltas, console_provider_gateway): no guarded config reads per chunk
- TOML parses (51 sites): all on cold user actions (Settings validate/revert, theme editor, backup/restore, image/video-gen raw section on panel compose); Library reader-pref read uses asyncio.to_thread(read_cli_config_serialized)
- JSON reads (68 sites): no per-keystroke/per-tick config JSON re-read found; speech voice-blend JSON reads are user-action paths
- Library/RAG settings keystroke profile reload is cached (TASK-32804.2 Done); RAG ConfigProfileManager and PromptHistory path are process singletons
- deepcopy of whole config objects: none on hot paths except inside get_runtime_config_snapshot (0.6 ms for the 73 KB settings dict)
- Console command popup, guidance dismissal, tool-count helpers on the keystroke/tick path: no config I/O
- Personal-context service memoized (TASK-32370 open for its budget); scheduler heartbeat write is offloaded; get_model_cache_dir/get_cli_log_file_path are cold; chatbook importer get_user_data_dir is guarded to once per import; Tools_Settings_Window force_reload paths are deprecated/nav-unreachable user actions

## census
**Warm-cost probes** (isolated profile: HOME/XDG/TLDW_CONFIG_PATH under scratch/config-reads, TLDW_TEST_MODE=1, audit tree on PYTHONPATH; machine load avg 20-38; opens via sys.addaudithook)

| Call (warm) | os.open / call | Py fn calls | Time @ scratch depth 13 | Est. at real ~/ depth |
|---|---|---|---|---|
| get_cli_setting (flat / dotted) | 0 | - | 1.4 us / 1.8 us (10k timeit) | same |
| load_cli_config_and_ensure_existence | 0 | - | 1.3 us | same |
| load_settings (guarded) | 609-726 (+39/dir level) | ~28k | 17-28 ms | ~297 opens, ~8 ms (TASK-32804.3 measured ~8 ms) |
| get_runtime_config_snapshot | 609 + deepcopy 0.6 ms | - | 27-48 ms | ~9 ms |
| operation(config) + N nested load_settings | 609 total, nested reads = 0 extra | - | same as one call | ~8 ms per outermost scope |
| get_user_data_dir / any get_*_db_path | 1,620-1,932 (+104/level) | ~126k | 67-174 ms | ~788 opens, 20-30 ms |
| resolve_sensitive_context (19 x get_user_data_dir) | 30,780 | - | 0.6-2.8 s | ~0.3-0.6 s |
| ReadFileTool.execute (2-byte file) | 32,899 | - | 873 ms | ~0.35-0.5 s |
| expand_references, 3 @refs (send path) | 94,212 | - | 2,723 ms | ~1-1.3 s |
| get_provider_readiness (math only) | 0 | - | 10 us | same |

**Boot to _ui_ready** (another audit agent's isolated cProfile, scratch/boot-import/boot.prof): 263,299 os.open; 797 config_participants.operation entries; 940 raw._scope entries; 59 get_user_data_dir; 212 load_settings; 425 ChatScreen._provider_readiness_app_config calls, including at least 36 that reached a guarded load_settings.

**Call-site census** (untruncated. Line counts are rg -c lines and include docstrings; AST counts are executable calls and exclude the generated Tools/remote_worker_bundle.py)

| Pattern | rg -c lines / files | AST calls / files | Top modules | Hot-path sites |
|---|---|---|---|---|
| get_cli_setting( | 440 / 100 | 403 / 94 | Summarization_General_Lib 48, app 34, console_chat_controller 22, config 22, transcription_service 21 | none costly (1.4 us) |
| load_cli_config_and_ensure_existence( | 56 / 21 | 50 / 18 | config 17, Tools_Settings_Window 12 | none costly; force_reload only on user actions |
| load_settings( | 109 / 38 | 61 / 26 | Local_Summarization_Lib 16, config 13, settings_screen 9, app 8, home 7, chat_screen 6 | chat_screen.py:7700 (via _provider_readiness_app_config: 18 sites in chat_screen, 9 in session.py, 2 in wiring.py), console_runtime.py:505, media_viewer_panel.py:1522/1594, get_cli_providers_and_models (9 calls) |
| get_runtime_config_snapshot( | 37 / 24 | 35 / 22 | LLM_API_Calls_Local 10, settings_screen 4 | about 20 per-request LLM handler sites (worker threads); settings_screen.py:21607 compose |
| get_user_data_dir( | 203 / 60 | 136 / 55 | app 25, config 25, personas_screen 10 | TldwCli.__init__ _wire_* chain (16 direct + 22 get_*_db_path in app.py); emergency_stop.py:98 via send_refusal_copy; file_operation_tools.py:64; scheduler loop.py:599 |
| get_*_db_path( | 88 lines | 59 / 22 | app 22 | boot; sensitive_paths._sensitive_db_paths (13 per context) |
| resolve_sensitive_context / is_sensitive_path without ctx | - | 17 + 9 | Tools/*, git/patch tools | every agent fs/git/patch tool call; console_references.py:307 per @token |
| explicit operation(config) in UI | 1 | 1 | chat_screen.py:21774 | every control-bar sync plus the 0.2 s run tick |
| keyring.* | 53 / 14 | - | server_credentials, link_key_custody | all TTL-cached or cold |
| toml/tomllib load(s) | 51 / 30 | - | Backup_Recovery, config | ChatScreen.__init__ _load_sidebar_state (raw._scope + toml.load per Console visit, unmeasured); the rest are cold |
| json.load( | 68 / 42 | - | speech_settings_mixin 6 | none per-tick |
| get_provider_readiness( | 31 / 18 | 30 / 17 | settings_screen 9 | cheap (10 us) |
| build_console_settings_readiness( | 11 / 6 | 10 / 5 | console_settings_modal 5 | cheap (~0.6 ms) |

**Guarded (handshake-paying) functions** in config.py, from the `@_config_participants.guarded` decorator lines: 1862 load_settings, 1927 _load_settings_uncached, 6318, 6699 _load_cli_config_bootstrap (now behind the warm fastpath), 6782, 6802, 6841, 6966, 7080 get_runtime_config_snapshot, 7169, 7179, 7190, 7276, 7288, 9433 get_user_data_dir, 9808 get_model_cache_dir. Only _load_cli_config_bootstrap has a warm bypass.


# pat-db-from-ui

## summary
Cross-cutting sweep of synchronous DB access reachable on the Textual event loop. Scope: UI (376 files), Widgets (307), Event_Handlers (40), Home (4), app.py, plus the Console, canvas, Persona_Buddy and Research_Workspace controllers. The audit tree is pinned at origin/dev 840ed2ca58.

Method: an AST sweep found 1,438 DB-ish call sites. 1,293 of them are neither awaited nor inside a thread function. 917 of those are in-memory ConsoleChatStore accessors. I triaged the other 341 by hand, plus 273 sites that pass a DB handle as an argument. I also swept all 280 Input/TextArea.Changed handlers, all 55 set_interval callbacks, and every scope service's `_maybe_await` seam. Suspect sites were traced to real callers. Six safe probes against scratch DBs gave the measured numbers.

Overall health: the Console send path, the Library screen and the Watchlists screen are well hardened. They use to_thread or run_db_off_loop behind an is_memory_db gate, and several verified-fine patterns hold. No timer callback does sqlite on the tick, and no keystroke handler runs an undebounced synchronous DB search.

Headline finding (new, measured, a regression after the 09-04 review): since commit 61a49de2e0 (2026-09-07), every file-backed `connect_private_sqlite` spawns a Python helper subprocess. It costs 42-78 ms, against 0.07-0.18 ms for a raw connect. Opening a new ChaChaNotes connection on a fresh thread measured 50-83 ms, against 0.13 ms warm. The UI's standard offload wrapper, `run_finite_local_worker`, closes the connection it opened after every call. It is reached by 88 `_run_library_service_call` sites (75 of which also create a new event loop per call with `asyncio.run`), 32 Notes folder operations, 21 media-reader calls and the Home counts. So most Library, Notes and media-reader calls now pay a helper spawn each. That makes TASK-24457's per-visit connection cost roughly 70 times what it assumed.

Other loop-blocking findings:
- **Evals:** composing the rail runs every EvalsDB read inside compose on the loop, measured at 52-98 ms. Datasets are re-read 3 times per compose even though the docstrings claim no new read. A json_extract scan covers the whole eval_results table. load_grid takes another 21-56 ms.
- **Console:** the turn-terminal persist and settle transaction runs on the loop once per send. The 9 KB update primitive alone measured 6-13 ms. TASK-22205 left this as residue.
- **Settings ▸ Agents:** it constructs AgentRunsDB inside compose, measured at 53-76 ms per visit.
- **Scope services:** `_maybe_await(sync_local())` runs local sqlite on the loop in the Study and Quiz scope services (including every reviewed flashcard), the persona scope service (67 sites) and the notifications scope service. Writing's `_ThreadOffloadedBackend` is the one-place fix template already in the repo.
- **Smaller loop-blocking paths:** Change Review runs git subprocesses and DB reads on the loop for each j/k file switch. A meeting transcript is parsed twice on each open. The Chatbook wizard runs 7 queries in its on_mount.
- **Minor:** Home computes a cold cache inside compose, and Buddy speech reads a persona row at 1 Hz.

Structural notes: `EvaluationScopeService` is built at boot but has no UI consumer. `ChatbookCreationWindow` and 3 CharactersRAGDB constructions in `Tools_Settings_Window` sit on a deprecated path. `_rehydrate_console_message_*` has no production caller.

## clean areas
- Console send-path prompt appliers (chat_screen.py:14509/14533): the controller offloads them with asyncio.to_thread (console_chat_controller.py:23287, 23397)
- Console durable-turn commit and pre-dispatch CAS: _run_durable_db_call puts them on to_thread (TASK-22205)
- Event_Handlers/Chat_Events/chat_rag_events.py: resolve_scope_for_session and _current_local_evidence_ids use to_thread behind an is_memory_db gate
- UI/Console_Modules/retrieval.py and workspace.py scope read/write: gated to_thread
- Console copy/save as markdown (chat_screen.py:7106/7130): to_thread
- Console raw CLI _execute (raw_cli.py:281-288): thread worker
- Console trajectory build and 0.5 s revision poll: thread workers
- Console character picker options and visual-identity resolution: passed to worker threads
- Console dictionary attach/list (chat_events_console_dictionaries): passed to to_thread
- Library ingest preflight duplicate annotation: runs inside the worker-thread preflight
- Library review-set liveness (TASK-32804.4 memo), export counts/preview (to_thread unless memory DB), paginated browse (verified-fine list)
- Watchlists/Collections screen: briefing settings reads and writes via to_thread; LocalWatchlistsService uses run_db_off_loop at 49 sites
- Writing: _ThreadOffloadedBackend (writing_scope_service.py:113) plus writing_controller._call. This is the template for the pattern fix
- Stats screen: @work(thread=True)
- Personas screen: 123 offloads; CCP character handler functions passed to to_thread (only small one-shot persona-scope handlers remain; see F5)
- Home flashcards, eval and read-later counts: @work(thread=True) at home_screen.py:380; content seam via to_thread
- Study dashboard fallback DB reads (task-15471): to_thread
- All 55 set_interval callbacks in scope: none does sqlite or file I/O on the tick; heavy work goes to run_worker or to_thread (39 have no I/O at all)
- All 280 Input/TextArea.Changed handlers: DB-backed searches are debounced and/or threaded (character-context search via run_owned_db_call, prompt/scope/tag pickers via workers, file-notes search via to_thread, Workflows PagedChoiceModal via to_thread)
- Event_Handlers/note_ingest_events.py: import worker via to_thread; eval_db_operations.py has no importer (dead)
- app.py timers: change-review retention and media cleanup via to_thread
- Scheduling workbench timers: screen-scoped (liveness reads a small heartbeat file only); DB reads are one-shot actions, P3
- Tamagotchi: opt-in, JSON file storage, not sqlite

## census
### A. Sweep totals (untruncated)
| Measure | Count |
|---|---|
| Files in scope | UI 376 · Widgets 307 · Event_Handlers 40 · Home 4 · app.py · 4 controllers |
| AST DB-ish call sites | 1,438 |
| … not awaited and not inside a thread function | 1,293 |
| … of which in-memory ConsoleChatStore accessors | 917 |
| … remaining, triaged by hand | 341 (+273 sites that pass a DB handle as an argument) |
| Offload primitives in scope | asyncio.to_thread 673 (110 files) · thread=True 223 (59 files) · run_in_executor 24 (4 files) |
| Input/TextArea.Changed handlers | 280: every DB-backed one is debounced and/or threaded |
| set_interval sites with method callbacks | 55: 39 have no I/O at all; 0 do direct sqlite on the tick |
| DB constructors in UI code | 7: ChatbookCreationWindow 2, Tools_Settings_Window 3 (deprecated), settings_agents_panel 1 (on the loop), eval_db_operations 1 (unused) |
| `_run_library_service_call` sites | 88, of which 75 use `isolate_in_worker=True` (runs `asyncio.run` per call) |
| `run_finite_local_worker` references | 14; it closes the connection it opened after every call |

### B. Per-site census (thread? = runs off the event loop)
| Site | DB method | Context | Trigger / frequency | Thread? |
|---|---|---|---|---|
| Chat/console_chat_controller.py:27275 | store.mark_message_complete → update_message_content + settle transaction (console_chat_store.py:16928/16949) | async stream coroutine | every send (terminal write) | **no** |
| Chat/console_chat_controller.py:12069 | commit_durable_turn | async | every send | yes |
| UI/Evals/library_rail.py:470-471, 605-606, 792-793 | list_datasets ×3, list_runs(500), run_group_cell_failure_counts, list_tasks ×2 | compose | Evals visit, and every rail-dirty select | **no** (52–98 ms) |
| UI/Evals/results_grid.py:508 | load_grid (drains all results, json.loads) | compose | run-group select | **no** (21–56 ms) |
| UI/Screens/evals_screen.py:896 | list_runs plus 2× list_tasks (via *_by_id) | select() | every Evals selection | **no** |
| UI/Evals/inspector.py:316/363 | load_bench, get_model | compose | every selection | **no** |
| Widgets/settings_agents_panel.py:143/181/382 | AgentRunsDB() + list_agent_definitions | __init__ in compose, async on_mount | Settings ▸ Agents visit (closed on unmount) | **no** (53–76 ms) |
| UI/Study_Modules/flashcards_handler.py:1013/1124 | StudyScopeService → update_flashcard_review, get_flashcard, get_due_flashcards | async handler | every reviewed card | **no** |
| UI/Study_Modules/flashcards_handler.py:585/649 | list_decks(100), list_flashcards(100) | async | Study visit, deck select | **no** |
| UI/Screens/personas_screen.py:15576/15583, 14864 | persona scope get/update/delete_persona_profile (local) | async handler | one-shot edits | **no** |
| UI/Screens/change_review_screen.py:1634/1957/2015 | change_snapshots_for_conversation + `git diff --name-status` per root | call_after_refresh, Select.Changed | open, turn switch | **no** |
| UI/Screens/change_review_screen.py:3539/4252 | notes_for_run + `git diff` subprocess | _focus_leaf (j/k/click) | every newly focused file | **no** |
| UI/Screens/library_screen.py:32824 and Widgets/Library/library_media_canvas.py:1950 | get_media_by_id (SELECT *) + meeting.json + full transcript.jsonl parse | viewer-state build, compose | every meeting-media open (×2) | **no** |
| UI/Screens/library_screen.py:16361 | can_rename_meeting_speakers (primary-key read + stat) | handler | every media selection | no (cheap) |
| UI/Wizards/ChatbookCreationWizard.py:236-373 | 7 queries incl. get_all_prompts, list_notes with content | SmartContentTree.on_mount | wizard open | **no** |
| UI/ChatbookCreationWindow.py:206/273 | new CharactersRAGDB + PromptsDatabase + 5 lists | on_mount | deprecated Tools path | **no** |
| Home/active_work_adapter.py:618 (+707, 480) | list_home_run_snapshot(20), list_queue(100) | compose_content, _sync_home_triage | Home visit or row click once the 3 s cache is stale | **no** (cold path) |
| UI/Navigation/buddy_speech.py:99 | get_persona_profile | 1 Hz asyncio task | while Buddy speech is on for a persona | **no** |
| UI/Screens/library_screen.py:13022/13024 (88 sites) | any sync scope call | to_thread(run_finite_local_worker) (+asyncio.run) | every Library list or open | yes, but a new connection and helper spawn per call |
| Notes/notes_scope_service.py:491 (32 sites) | folder repository operations | same wrapper | Library ▸ Notes folder operations | yes, plus a helper spawn |
| Media/media_reading_scope_service.py:195 (21 sites) | LocalMediaReadingService leaf calls | same wrapper | media reader | yes, plus a helper spawn |
| Event_Handlers/Chat_Events/chat_rag_events.py:1041-1065, 1298-1320 | scope and evidence reads | async, gated on memory DB | per retrieval | yes |
| UI/Console_Modules/retrieval.py:307/316; workspace.py:5763/5777 | read/write_conversation_scope, workspace scope | async, gated | scope picker | yes |
| UI/Console_Modules/workspace.py:5419/5621/5669/7942 (~12 sites) | registry get_active_workspace / get_workspace | handlers, async | menu or picker open | no (point reads, P3) |
| UI/Screens/scheduling/schedules_workbench.py:4412-4483, 4590, 4926, 5013 | count/list results, sync state, conflicts | actions, handlers | mark-read, owner switch | no (bounded, P3) |
| app.py:15027 | reconcile_stale_automation_runs | App.on_mount | once per boot | no (1 UPDATE) |
| Subscriptions/local_watchlists_service.py (49 sites) | get_new_items etc. | async | Watchlists | yes (run_db_off_loop) |
| Writing_Interop/writing_scope_service.py:113 | all local writing operations | _ThreadOffloadedBackend | every Writing call | yes (template) |
| UI/Screens/stats_screen.py:178 | stats queries | @work(thread=True) | Stats visit | yes |

### C. Scope-service `_maybe_await(sync_local())` census (server-only branches excluded)
| Scope service | direct local-capable sites | resolving to a synchronous local method | offload |
|---|---|---|---|
| MediaReadingScopeService | 94 | 83 by name (51 local methods touch the DB) | partial: 21 via _call_local_leaf |
| StudyScopeService | 39 | 27 | none |
| QuizScopeService | 6 | 6 | none |
| CharacterPersonaScopeService | 67 via _invoke_backend_method | local service has 79 sync methods, 0 to_thread | none (TASK-32804.12) |
| NotificationsScopeService | 6 | 6 | none |
| EvaluationScopeService | 21 | 21 | none (no UI consumer) |
| WritingScopeService | 28 | 28 | **all** (_ThreadOffloadedBackend) |
| WatchlistScopeService | 37 | 0 | local service is async + run_db_off_loop |

### D. Measured costs (safe probes on scratch DBs, load average 10–31)
| Operation | Cost |
|---|---|
| raw sqlite3.connect | 0.07–0.18 ms |
| connect_private_sqlite on an existing file (spawns the helper subprocess) | 42.7 ms median at load 10 · 78.3 ms at load 31 |
| ChaChaNotes first query on a new thread / warm | 49.5–50.8 ms (load 10), 57–83 ms (load 25–30) / 0.13 ms |
| AgentRunsDB re-open + list_agent_definitions + close | 52.9 ms (load 10) · 76.2 ms (load 31) |
| update_message, 9 KB (FTS + sync_log triggers) | 8.2 ms median (6.1–13.1) |
| Evals rail-compose reads at 8k / 30k eval_results | 51.8 / 98.3 ms |
| run_group_cell_failure_counts at 8k / 30k | 14.5–24.6 / 41.8–62.6 ms |
| load_grid, one 4×50 group | 21–56 ms |


# pat-db-schema

## summary
DB schema / index / query-plan sweep over every SQLite owner under tldw_chatbook (105 files carry CREATE TABLE text; 312 CREATE INDEX statements outside recovery/; 305 index rows in scripts/index_plan_pin_census.tsv, 87 plan-pinned and 218 pre-convention). I built isolated scratch DBs with the real constructors (ChaChaNotes v73 with 300 conversations, 9.1k messages and 2k notes; plus 6k-note/37 MB, 8.3k-conversation, 15k-lorebook-entry and 8k-keyword variants; Media 600 docs plus a 400 x 60 KB variant; Prompts, Subscriptions, AgentRuns, Workspace, Collections, Evals). I captured real method SQL with the sqlite3 trace callback and ran EXPLAIN QUERY PLAN with sqlite_stat1 absent. I also ran a static EQP pass over about 870 literal SELECTs.

**Main result: the expensive part is the connection and transaction layer, not missing indexes.**
1. Every private SQLite connect launches a fresh Python helper subprocess to vet the file: about 75 ms wall and 72 ms CPU, with at most 4 at once.
2. The 'owned connection' wrappers close the handle after every operation. On an executor thread that holds no handle, each Console durable write, trace write, maintenance tick or fleet-wake poll therefore pays a helper launch. Measured 15/15 launches on unprimed threads, 0/15 on primed ones.
3. Every outermost `transaction()` on every participating DB runs a Backup_Recovery storage admission. It does 16 root-anchored directory walks, 7 registry.json reads and a flock, under a process-global RLock. Measured 9.3 ms against 0.043 ms stubbed, and it fully serialises transactions across threads and DBs (0.96x of ideal on 4 threads).
4. At idle, the legacy trace maintenance loop pays all of this once a second (TASK-31501, now with a much larger cost).

**Query-plan level:** the TASK-278 'fix' made `search_conversations_page` evaluate a full `messages_fts` MATCH once per candidate conversation. It measured 1124 ms against 7 ms for the uncorrelated IN form (and 15.7 ms for the pre-278 LIKE form). This runs on Ctrl+K History search, Library ▸ Conversations search and the Console browser search. Without sqlite_stat1, low-selectivity single-column indexes (10 on `deleted` alone, plus `enabled`) pull the planner into inverted joins: media and prompt keyword batch 113.6 ms vs 0.28 ms; lorebook entries 7.1 ms vs 0.5 ms. Media `sync_log` is never pruned and stores the full content twice per ingest (45.9 MB sync_log for 22.5 MB of content). No DB sets cache_size, mmap_size or temp_store. Plan shapes for the primary list and read paths are healthy (see clean_areas).

**Suggested PR groups:**
- (A) Connection layer: cache the helper's verdict or keep one helper alive, and give each DB a single held-connection worker instead of closing the handle after every operation.
- (B) Admission once per connection or per generation, not per transaction, and never under the global lock during file I/O.
- (C) Query-plan fixes: correlated FTS, unary-plus or index pruning, title and deleted indexes, julianday normalisation, message index pruning. Each gets a stat1-absent EQP pin.
- (D) sync_log retention for the media DB.
- (E) PRAGMA tuning.

## clean areas
- ChaChaNotes list_all_active_conversations and Library/Console browser pages (no query): SEARCH idx_conversations_archive, index-ordered, no temp sort (plan-pinned)
- Conversation message reads: get_messages_for_conversation, get_message_tree_rows_for_conversation, get_root_messages_for_conversation, get_library_conversation_messages all use idx_msgs_conv_ts or idx_messages_conversation_id_id (0.1-1 ms for 30 msgs)
- Batched id-list reads: get_attachments_for_messages, get_generation_metadata_for_messages, get_message_images_by_ids, get_conversation_archive_states, unread_ids_for, semantic-revision provider projection. All are chunked IN-lists on PK/unique indexes (the scanner's 'loop execute' hits here are chunk loops, not N+1)
- FTS searches: search_messages_by_content, search_conversations_by_title, search_notes, search_keywords, search_character_cards are FTS-driven with rowid joins
- ChaChaNotes keyword batches (get_keywords_for_notes_batch, get_keywords_for_conversations, Library notes/conversation keyword joins) are driven from the link-table autoindex; ChaChaNotes keywords has no deleted-only index, so no inversion
- Media DB list/sort pages (search_media_db library_summary, get_paginated_files, list_library_media_page) use the TASK-21126 partial indexes idx_media_active_*; media FTS search is about 13 ms at 22 MB of content; the FTS-AND-LIKE re-verify is bounded by the hit set
- Single-id media keyword lookups (fetch_keywords_for_media_batch([id]), get keywords for one media) plan correctly: mk first
- Collections capture item pages, console trace indexes, canvas, note_links, automatic_work reservations: plan-pinned tests exist and plans are index searches
- Held-connection DBs (Chunking_Lab_DB, Workflows_DB, Library_Collections_DB, Workspace_DB, AgentRuns_DB, NotesDeviceStateStore) set WAL+NORMAL once per connection, with no per-transaction schema census
- Chatbook importer wraps each conversation in one transaction(immediate=True), so nested add_message calls do not pay admission per row
- Subscriptions watchlist/source count queries are one grouped covering-index scan per visit; the find_duplicate_items full scan has zero callers (dead code)
- AgentRuns orphan reconcile (TASK-32804.10) is batched and off the compose path; change_notes run_id scans exist but the table is tiny
- PARSE_DECLTYPES timestamp converters: measured +40% on a 5000-row fetch (17.4 to 24.3 ms), about 1.4 us per row, P3-level, not filed
- FTS5 '_config' reload statements seen in traces are per-statement internal checks on data_version, negligible
- Stats screen (user_statistics.py) runs about 15 full message-table scans, but inside a thread worker on a cold, user-initiated path (P3, not filed)
- Tamagotchi SQLite storage opens a connection per operation but is dormant (no import site)

## census
### 1. Hot queries vs index coverage (EXPLAIN QUERY PLAN, `sqlite_stat1` absent, scratch DBs built by the real constructors)

| # | Query (method) | DB | Caller path / frequency | Plan without stats | Coverage | Measured |
|---|---|---|---|---|---|---|
| 1 | `list_all_active_conversations` | CC | Console / Library lists | SEARCH idx_conversations_archive, index-ordered | ✅ covered | 0.3 ms (300 convs) |
| 2 | `search_conversations_page(query=None, scope/char/state/workspace)` | CC | Console browser, Library, Personas | SEARCH idx_conversations_archive, then filter scope/client/char/workspace; COUNT walks every active conv | ⚠️ partial | about 7 ms, of which about 6 ms is admission |
| 3 | `search_conversations_page(query)` and `query_terms` (also `locate_conversation_page`) | CC | Ctrl+K History (debounced typing), Library ▸ Conversations submit, Console browser search | CORRELATED SCALAR SUBQUERY → full `messages_fts` MATCH per conversation | ❌ **O(C×H)** | **1124 ms** vs 7.2 ms with IN-subquery; 0-hit term 49.6 vs 0.3 ms; switcher path 679 ms |
| 4 | `get_conversations_for_character` | CC | Personas conversations, Character_Chat_Lib | picks idx_conversations_archive over idx_conv_char, plus TEMP B-TREE (julianday) | ❌ misplanned | 2.18 vs 0.43 ms (8.3k convs) |
| 5 | `get_conversation_by_id` | CC | everywhere | PK autoindex | ✅ | 0.09 ms |
| 6 | message tree, root and paged reads | CC | Console restore, Library reader | SEARCH idx_msgs_conv_ts | ✅ | 0.8 ms for 31 rows |
| 7 | `get_latest_message_for_conversation` | CC | lists | idx_msgs_conv_ts + temp for id tiebreak | ✅ | 0.3 ms |
| 8 | `count_messages_for_conversations` | CC | lists | idx_msgs_conversation + temp GROUP BY | ✅ | 2.8 ms (20 convs) |
| 9 | `get_messages_for_conversations_batch` | CC | batch export | window over all msgs of N convs, including image_data, then rn filter | ⚠️ no per-conv limit pushdown | 4.9 ms |
| 10 | `search_library_conversations_page` | CC | agent/MCP Library tool | correlated `m.content LIKE '%q%'` per conv, plus FTS list subqueries; ×3 statements | ❌ full corpus LIKE | 8.9 ms per statement per 3.6 MB (0-hit), linear |
| 11 | `list_notes` / `list_library_notes_page` page | CC | Library, agent | SCAN via idx_notes_last_modified, filter deleted | ✅ (active) | 0.07 ms |
| 12 | `count_notes`, Library COUNT, `list_deleted_notes` | CC | Library visit evidence, Trash | SCAN notes (no `deleted` index) | ⚠️ | 2.0 ms → 0.24 ms with (deleted,last_modified); Trash page 2.76 → 0 ms (6k notes / 37 MB) |
| 13 | `page_note_placements` unfiled, newest/oldest | CC | Library ▸ Notes tree slices | SCAN n + NOT EXISTS + TEMP sort on `julianday(last_modified)`, selecting full content | ❌ function sort | 24.3 ms vs 0.1 ms with plain `last_modified DESC` (6k notes) |
| 14 | `search_library_notes_page(q)` | CC | agent/MCP Library tool | SCAN notes `content LIKE` + FTS, ×2 | ❌ LIKE where FTS exists | 29.5 ms (2k notes) |
| 15 | `get_note_by_title` / `get_conversation_by_name` | CC | chatbook import, per item plus unique-name loops | SCAN notes / SCAN conversations | ❌ no title index | 2.38 → 0.61 ms per call with index (6k notes) |
| 16 | lorebook `get_world_book_entries` | CC | per RP send (world-info capture), Personas ▸ Lore N+1 counts | SEARCH idx_world_book_entries_**enabled** + temp sort | ❌ misplanned | 7.07 vs 0.52 ms per book (15k entries) |
| 17 | media keyword batch (`_library_keywords_for_media`, module `fetch_keywords_for_media_batch`, method with >1 id) | Media | RAG media search (keyword filter), scope picker, agent Library media pages | SEARCH k USING idx_keywords_**deleted**, probe mk per keyword | ❌ inverted join | **113.6 vs 0.28 ms** (8k keywords, 50-id page) |
| 18 | prompt keyword batch (Prompts_DB.py:3443) | Prompts | agent Library prompt pages | idx_promptkeywordstable_deleted drives the join | ❌ same shape | (same shape as #17) |
| 19 | `search_media_db` list/sort/type/keyword | Media | Library ▸ Media | idx_media_active_recent / type / title (TASK-21126) | ✅ | 12-16 ms (mostly admission-free reads) |
| 20 | `search_media_db(q)` | Media | Library ▸ Media search | FTS + per-hit `content LIKE` re-verify | ✅ bounded | 13 ms at 22 MB |
| 21 | Subscriptions `find_duplicate_items` | Subs | none (dead) | SCAN subscription_items (OR defeats index) | ❌ but dead | n/a |

### 2. Cost of the connection and transaction layer (the dominant DB cost)

| Operation | Measured (scratch) | Notes |
|---|---|---|
| `with db.transaction(): SELECT 1` (CC) | **9.34 ms median**, 245 `open()` | 0.043 ms with `_core_operation` stubbed (217×). At a real `~/.config` depth: about 117 opens (16 root walks × 8 fewer components) |
| `add_message` (CC) | 13.8 ms median (populate p95 59 ms) | 1.98 ms with admission stubbed (86% admission) |
| `add_media_with_keywords` | 8.0 ms median | same admission seam |
| 4 threads × 40 txns across CC + Media | 0.96× of ideal 4× | global `storage_admission._lock` serialises every DB |
| fresh private connect (`connect_private_sqlite`) | 75 ms median wall; 28.6 ms parent + 42.9 ms child CPU | `subprocess.Popen(python -I -S helper)` per connect; 4-way cap |
| `run_owned_db_call` on unprimed executor thread | 78.9 ms, 15/15 helper spawns | 10.3 ms and 0/15 when the pool thread already holds a handle |
| `CharactersRAGDB` reopen of an up-to-date DB | 158 ms (113 ms connect) | per construction |
| raw SQL insert of one message | 0.915 ms; 0.818 without 4 redundant indexes; 0.43 ms without the sync_log trigger | trigger payload 1.54× content |

### 3. PRAGMA settings per DB (grep of connection setup)

| DB | journal | synchronous | FK | busy | cache_size / mmap / temp_store |
|---|---|---|---|---|---|
| ChaChaNotes | WAL | NORMAL | ON | connect timeout 15 s | none |
| Media, Prompts | WAL | NORMAL | ON | default | none |
| Subscriptions | WAL | NORMAL | ON | BUSY_TIMEOUT_MS | none |
| AgentRuns | WAL | NORMAL | ON | 5000 | none |
| Workspace, Library_Collections, Evals, RAG_Indexing, Ingest_Jobs, Notes device state, Scheduling | WAL | NORMAL | ON | default | none |
| Chunking_Lab | WAL | FULL (deliberate) | ON | 5 s | none |
| Backup validation | n/a | n/a | n/a | n/a | cache_size = -2048, temp_store = MEMORY (the only place) |

Measured effect of cache_size = 64 MB / mmap = 256 MB / temp_store = MEMORY on notes_big: LIKE scan 21.3 → 14.9 ms; julianday sort 11.1 → 3.4 ms; FTS subquery 4.6 → 4.4 ms.

### 4. Static EQP sweep (literal SELECTs bound with NULL params)

| DB | OK | SCAN | TEMP-only | unparseable fragment |
|---|---|---|---|---|
| chachanotes (18 modules) | 412 | 111 | 42 | 113 |
| media | 72 | 5 | 9 | 46 |
| prompts | 37 | 16 | 5 | 5 |
| subs | 33 | 29 | 6 | 10 |
| agentruns | 33 | 12 | 3 | 7 |
| workspace | 3 | 5 | 0 | 0 |
| collections | 2 | 2 | 0 | 0 |
| evals | 12 | 4 | 3 | 1 |

Most SCANs are sqlite_master or schema_version probes, tiny tables, maintenance GC CTEs, the Stats screen, or fragments. The ones that matter are triaged in section 1.

**Schema size (scratch):** ChaChaNotes has 182 tables (FTS shadows included), 289 indexes and 184 triggers; the `messages` table carries 11 indexes, of which idx_msgs_conversation and idx_messages_variant_of are prefix-redundant and idx_msgs_ranking and idx_messages_role have no reader. Largest table in ChaChaNotes is `sync_log`: 13.8 MB against 6.2 MB messages and 4.1 MB notes. Media `sync_log` is 45.9 MB against 22.9 MB Media and 22.9 MB DocumentVersions.

**Loops containing `execute` (AST):** 288 loops across tldw_chatbook; the ChaChaNotes, Media and Chat hits reviewed are chunked IN-lists, single-transaction batches or migrations. No hot N+1 found apart from the lorebook per-book entries (#16) and the per-row FTS probes in Prompts (4 per result row).


# pat-god-modules

## summary
Structural slice, run over the whole of tldw_chatbook (2,541 files, 1,825,311 lines) at audit tree 840ed2ca58. Import measurements used isolated subprocesses with HOME, XDG_* and TLDW_CONFIG_PATH pointed at scratch, plus TLDW_TEST_MODE=1. The app was never constructed.

**What the god modules actually cost.**
- **File size itself is nearly free at import.** Unmarshalling the 1.5 MB .pyc of library_screen takes 3.3 ms. The cost comes from three places:
  - what a module imports at module scope,
  - what TldwCli.__init__ constructs before first paint,
  - the model classes those imports define.
- **Boot heap.** `import tldw_chatbook.app` loads 681 own modules and 1,772 total, in about 1.2–1.5 s warm. Adding the Chat screen closure brings the total to about 1.6 s.
- **Model definitions are a third of pre-paint import.** App plus Chat define 1,605 dataclasses, which take about 540 ms with GC off. Another 264 pydantic models take about 128 ms.
- **app.py pulls modules Chat never needs.** It is a 228-import composition root: 404 of its 470 module-scope names are used only inside function bodies. It pulls 367 own modules beyond Chat's own closure, costing 450–540 ms measured. That includes the TTS stack, File-Notes git, Home, MCP control plane, Actor Packs, Kanban and Evals.
- **Five feature DBs are opened in TldwCli.__init__, before the loop starts.** Each costs 58–75 ms, because every POSIX `connect_private_sqlite` spawns a Python helper subprocess (47 ms against 0.09 ms for a plain connect).

**Hot paths identified.**
- Boot / first paint: F1, F2 and F6.
- Every private-SQLite connection open (F3), which feeds boot, Library visits and RAG searches.
- The first server-backed call in server mode, which imports tldw_api/client.py on the loop (F4, 575–854 ms, 1,257 pydantic models).
- Automatic gen-2 GC over a 250k-object boot heap: about 62 ms per collection, and about 0 ms after `gc.freeze()` (F5, supports TASK-31966).

**Mitigations already in place (verified).**
- The screen registry is lazy.
- A background pre-importer thread imports all 23 routed screens (1,104 ms, 1,514 own modules) off the loop. So the size of library, settings and personas does not tax boot.
- Settings has already moved to region-scoped recompose.
- Library whole-screen recompose is still about 22 live sites, already tracked by TASK-281/22888/21243.

**Governance.** The size ratchets are not a PR gate: they run only on main pushes and in nightly. 13 rows are currently over budget on dev; console_chat_controller alone is 1,135 lines over. settings_screen (33k lines, the #2 module) still has no row.

**Duplication.** The verbatim-duplication mass is mostly `Tools/remote_worker_bundle.py`, a deliberate copy of the tool implementations for remote workers on Python 3.10 (no local runtime cost), plus the legacy Evals loaders. The only duplicated families that pay at runtime are:
- the scope-service `_maybe_await` scaffold (66 definitions, 873 call sites; known, TASK-32898/ADR-177);
- the per-DB connection-helper family: 20 `_get_connection` copies, and 51 files calling `connect_private_sqlite` directly, of which about 27 open per call. This family is why F3 multiplies.

**Proposed PR groups, ordered by speed payoff.**
1. **PR-A, boot DBs lazy (F1):** LibraryCollections, Workspace, Subscriptions, ScheduledTasks and Evals become lazy lock-guarded properties, following the ClientNotificationsDB / TASK-21105 / rag_admin shape. About 340 ms off every boot.
2. **PR-B, composition-root diet (F2, pairs with TASK-33011):**
   - split the Message classes out of tts_events and stts_events;
   - make file_notes_session_owner, tts_service, home adapter, kanban, MCP control plane, actor-pack coordinator and evaluation services lazy;
   - lower MAX_TLDW_MODULE_COUNT in the same PRs.
   About 290 ms deferrable.
3. **PR-C, themes (F6):** generate the colour system for the active theme only. About 25–30 ms.
4. **PR-D, private-SQLite helper amortization (F3):** needs an ADR and security review. Use a persistent retained helper or a per-inode prepare memo, then hold connections in the per-search RAG legs and the Personal Context per-operation opens.
5. **PR-E, tldw_api first-call (F4):** pre-import the client module on the pre-import thread when a server is configured, and thread the first `build_client`. Later, split TLDWAPIClient into per-namespace delegates.
6. **PR-F, GC policy (F5):** needs an ADR per TASK-31966. Call `gc.freeze()` after `_ui_ready` and after the pre-import pass.
7. **PR-G, ratchet governance (F7):** add the three size-ratchet files to the PR fast lane, re-pin the 13 breached rows, and add a settings_screen row.

## clean areas
- Screen registry + background screen pre-importer (app.py:16971 `_preimport_screens`): importing all 23 routed screen modules after app costs ~1,104 ms / 1,514 tldw modules but runs on a paced daemon thread -- god screens (library 491-521 ms incremental, personas 405-427, settings 352-364) are NOT on the boot path. Verified by fresh-process import measurements.
- Package __init__ eager imports on the boot path (Chat, Actor_Packs, STT, Home, Media, RAG_Admin, LLM_Provider_Catalog, Evaluations_Interop, UI.Navigation, Canvas): a what-if meta-path probe replacing each __init__ with a PEP 562 lazy stub saved 0-2 modules each -- the submodules are imported by other boot code anyway. Not worth changing.
- God-module FILE SIZE itself: marshal.loads of the largest .pyc files is 0.2-3.3 ms (library_screen 1,524 KiB -> 3.3 ms). Size is not the import cost; module-scope imports/work are.
- `-X importtime` self-time spikes on leaf Chat modules (console_library_policy 26 ms, provider_readiness 31 ms, console_live_work 21 ms, console_session_settings 33.6 ms) are GC pauses, not module work: with gc disabled they measure 3.6 / 1.3 / 2.7 / 5.6 ms. Do not chase them.
- config.py embedded TOML template: 102 KB / 2,226 lines parsed by tomllib at import in 4.3 ms; deepcopy 0.38 ms. Not a cost (config's real import cost is admit_startup/installation_client_id/load_settings, outside this slice).
- Settings screen: already migrated off screen-level recompose=True (task-15475; region.refresh(recompose=True) at settings_screen.py:16700/17871). No whole-screen recompose sites remain.
- FirstRunSetupWizard: 14 self.refresh(recompose=True) sites, all on state transitions of a one-time flow (install result, activation, selection commit); InstallProgressed path updates in place and recomposes only on NoMatches fallback. Cold path.
- Verbatim-duplicate mass (dup_census.py re-run): dominated by Tools/remote_worker_bundle.py (deliberate remote-worker 3.10-floor bundle mirroring git_tool_impls/patch_tool_impls/local_tool_impls/path_validation/sensitive_paths -- no local runtime cost) and legacy Evals dataset_loader vs eval_runner copies. No runtime-paying verbatim dup found beyond the scope-service scaffold (known TASK-32898) and the connection-helper family (feeds F3).
- UI/Tools_Settings_Window.py (6,928 lines, DEPRECATED) + UI/Screens/tools_settings_screen.py: reachable only via the unused `ToolsSettingsScreen` lazy export in UI/Screens/__init__.py; never loaded at boot, first paint or by the pre-importer -- dead weight with zero speed cost (retirement owned by TASK-32807.4).
- Known-dead legacy modules from the tier-2 legacy-reachability table (Widgets/NewIngest, Tamagotchi widget, Confluence, MediaWindow_v2/media_screen, SiteConfigSettings, dictation_service, etc.): none are resident at boot or first paint (checked against the fresh-process module sets). Only Evals eval_orchestrator/eval_runner/task_loader (via app wiring, see F1) and Kanban_Interop (known TASK-21107/21239) load at boot.
- tldw_chatbook/__init__.py: one eager import (tiktoken_runtime arm) -- fine.
- Token-estimator duplication: 14 estimate/count_tokens defs across Chunking/Subscriptions/TTS/Character_Chat/Utils -- distinct purposes, none on a shared hot path. Not a finding.
- Console transcript reconciler, streaming persistence, Library pagination, browse-search debounce, subscriptions scheduler template, screen-registry laziness -- verified-fine list from CONTEXT.md respected; not re-examined.

## census
### C1. Package size distribution (AST census, `scratch/structural/census.py`)
2,541 .py files, 1,825,311 lines. Modules ≥10k lines: **15** · 5k–10k: **23** · 2k–5k: **124** · 1k–2k: **264**.

### C2. Top 40 modules by lines (resident-at = first place the module is loaded in a fresh isolated process)
| # | module | lines | top-level imports | defs | classes | resident at |
|---|---|---:|---:|---:|---:|---|
| 1 | UI/Screens/library_screen.py | 35,902 | 147 | 1386 | 1 | screen pre-import thread |
| 2 | UI/Screens/settings_screen.py | 33,091 | 121 | 1186 | 19 | screen pre-import thread |
| 3 | Chat/console_chat_controller.py | 30,502 | 108 | 650 | 34 | first paint (Chat) |
| 4 | UI/Screens/chat_screen.py | 25,406 | 160 | 842 | 7 | first paint (Chat) |
| 5 | DB/ChaChaNotes_DB.py | 24,424 | 28 | 449 | 10 | boot (import app) |
| 6 | Chat/console_chat_store.py | 22,545 | 67 | 627 | 44 | first paint (Chat) |
| 7 | app.py | 19,682 | 228 | 591 | 17 | boot |
| 8 | tldw_api/client.py | 16,689 | 67 | 1222 | 3 | on demand (first build_client) |
| 9 | UI/Screens/personas_screen.py | 16,533 | 118 | 504 | 20 | screen pre-import thread |
| 10 | UI/Screens/watchlists_collections_screen.py | 14,330 | 74 | 410 | 7 | screen pre-import thread |
| 11 | Notes/file_notes_git_service.py | 11,527 | 20 | 303 | 43 | **boot** (app.py:420) |
| 12 | UI/Wizards/FirstRunSetupWizard.py | 10,860 | 46 | 462 | 31 | on demand |
| 13 | Chat/console_agent_bridge.py | 10,847 | 66 | 224 | 27 | Console mount |
| 14 | DB/Client_Media_DB_v2.py | 10,309 | 23 | 133 | 6 | boot (via config) |
| 15 | config.py | 10,129 | 38 | 189 | 19 | boot |
| 16 | Widgets/Library/library_file_notes_workspace.py | 8,856 | 45 | 315 | 18 | on demand |
| 17 | Agents/agent_service.py | 8,777 | 41 | 165 | 7 | first paint |
| 18 | Widgets/Console/console_transcript.py | 8,452 | 52 | 314 | 17 | first paint |
| 19 | UI/Console_Modules/workspace.py | 7,949 | 38 | 285 | 8 | first paint |
| 20 | Widgets/Console/console_settings_modal.py | 7,816 | 44 | 270 | 10 | on demand |
| 21 | Chat/console_provider_gateway.py | 7,483 | 63 | 208 | 25 | first paint |
| 22 | UI/Tools_Settings_Window.py | 6,928 | 33 | 131 | 3 | never (dead) |
| 23 | Media/local_media_reading_service.py | 6,888 | 17 | 254 | 1 | boot |
| 24 | DB/Subscriptions_DB.py | 6,801 | 21 | 134 | 8 | boot |
| 25 | UI/MCP_Modules/mcp_workbench.py | 6,774 | 42 | 193 | 4 | screen pre-import |
| 26 | Widgets/Console/console_composer_bar.py | 6,587 | 30 | 201 | 14 | first paint |
| 27 | Chat/console_trace_service.py | 6,561 | 25 | 174 | 25 | on demand |
| 28 | UI/Library_Modules/library_notes_controller.py | 6,366 | 42 | 326 | 1 | on demand |
| 29 | TTS/profile_repository.py | 6,304 | 37 | 193 | 10 | on demand |
| 30 | Widgets/Settings_Widgets/speech_tts_settings_panel.py | 6,093 | 43 | 177 | 10 | screen pre-import |
| 31 | UI/Console_Modules/session.py | 5,667 | 53 | 196 | 6 | first paint |
| 32 | MCP/unified_control_plane_service.py | 5,600 | 29 | 110 | 4 | **boot** (pinned by test_app_import_weight) |
| 33 | UI/Screens/scheduling/schedules_workbench.py | 5,492 | 38 | 203 | 2 | screen pre-import |
| 34 | Notes/notes_sync_executor.py | 5,467 | 18 | 185 | 11 | on demand |
| 35 | Agents/local_tool_provider.py | 5,246 | 34 | 96 | 11 | on demand |
| 36 | DB/Prompts_DB.py | 5,231 | 22 | 106 | 7 | boot |
| 37 | UI/Library_Modules/library_prompts_controller.py | 5,199 | 33 | 191 | 1 | on demand |
| 38 | UI/Screens/llm_screen.py | 5,180 | 51 | 187 | 3 | screen pre-import |
| 39 | UI/Screens/change_review_screen.py | 4,967 | 22 | 172 | 7 | on demand |
| 40 | Chat/console_runtime.py | 4,964 | 25 | 174 | 13 | boot |

.pyc unmarshal cost (measured): library_screen 1,524 KiB → 3.3 ms; settings 2.7 ms; controller 2.3 ms; chat_screen 2.1 ms; everything else ≤2 ms. **File size alone is not the import cost.**

### C3. Top 30 classes by lines
| # | class | location | lines | methods |
|---|---|---|---:|---:|
| 1 | LibraryScreen | UI/Screens/library_screen.py:996 | 34,782 | 1334 |
| 2 | SettingsScreen | UI/Screens/settings_screen.py:2852 | 30,240 | 1076 |
| 3 | ConsoleChatController | Chat/console_chat_controller.py:4563 | 25,940 | 455 |
| 4 | ChatScreen | UI/Screens/chat_screen.py:1767 | 23,631 | 760 |
| 5 | CharactersRAGDB | DB/ChaChaNotes_DB.py:717 | 23,476 | 413 |
| 6 | ConsoleChatStore | Chat/console_chat_store.py:1633 | 20,913 | 560 |
| 7 | TLDWAPIClient | tldw_api/client.py:1207 | 15,478 | 1218 |
| 8 | PersonasScreen | UI/Screens/personas_screen.py:1074 | 15,460 | 462 |
| 9 | WatchlistsCollectionsScreen | UI/Screens/watchlists_collections_screen.py:636 | 13,695 | 392 |
| 10 | TldwCli | app.py:7488 | 12,139 | 359 |
| 11 | MediaDatabase | DB/Client_Media_DB_v2.py:263 | 8,771 | 94 |
| 12 | LibraryFileNotesWorkspace | Widgets/Library/library_file_notes_workspace.py:813 | 8,044 | 276 |
| 13 | ConsoleWorkspaceController | UI/Console_Modules/workspace.py:565 | 7,385 | 226 |
| 14 | FileNotesGitService | Notes/file_notes_git_service.py:2890 | 7,086 | 144 |
| 15 | ConsoleSettingsModal | Widgets/Console/console_settings_modal.py:1022 | 6,795 | 247 |
| 16 | ToolsSettingsWindow | UI/Tools_Settings_Window.py:195 | 6,729 | 127 |
| 17 | AgentService | Agents/agent_service.py:2059 | 6,719 | 54 |
| 18 | LocalMediaReadingService | Media/local_media_reading_service.py:176 | 6,713 | 253 |
| 19 | SubscriptionsDB | DB/Subscriptions_DB.py:451 | 6,348 | 121 |
| 20 | ConsoleComposerBar | Widgets/Console/console_composer_bar.py:402 | 6,186 | 191 |
| 21 | MCPWorkbench | UI/MCP_Modules/mcp_workbench.py:618 | 6,157 | 165 |
| 22 | LibraryNotesController | UI/Library_Modules/library_notes_controller.py:635 | 5,706 | 323 |
| 23 | ConsoleAgentBridge | Chat/console_agent_bridge.py:5166 | 5,507 | 98 |
| 24 | ConsoleTranscript | Widgets/Console/console_transcript.py:2969 | 5,484 | 195 |
| 25 | SpeechTTSSettingsPanel | Widgets/Settings_Widgets/speech_tts_settings_panel.py:709 | 5,385 | 143 |
| 26 | UnifiedMCPControlPlaneService | MCP/unified_control_plane_service.py:290 | 5,311 | 92 |
| 27 | SchedulesWorkbench | UI/Screens/scheduling/schedules_workbench.py:388 | 5,105 | 163 |
| 28 | ConsoleSessionController | UI/Console_Modules/session.py:738 | 4,930 | 160 |
| 29 | TTSProfileRepository | TTS/profile_repository.py:1425 | 4,880 | 118 |
| 30 | LLMScreen | UI/Screens/llm_screen.py:324 | 4,857 | 173 |

### C4. Size-ratchet status on dev 840ed2ca58 (budget vs `len(read_text().splitlines())`)
| ratchet row | budget | measured | delta |
|---|---:|---:|---:|
| Chat/console_chat_controller.py | 29,367 | 30,502 | **+1,135** |
| UI/Wizards/FirstRunSetupWizard.py | 10,404 | 10,860 | **+456** |
| Chat/console_chat_store.py | 22,344 | 22,545 | +201 |
| UI/Screens/chat_screen.py (screen ratchet) | 25,218 | 25,406 | +188 |
| Widgets/Console/console_transcript.py | 8,353 | 8,452 | +99 |
| UI/Screens/personas_screen.py | 16,436 | 16,533 | +97 |
| UI/Screens/library_screen.py (screen ratchet) | 35,855 | 35,902 | +47 |
| UI/MCP_Modules/mcp_workbench.py | 6,760 | 6,774 | +14 |
| UI/Library_Modules/library_skills_controller.py | 3,142 | 3,154 | +12 |
| Widgets/Console/console_settings_modal.py | 7,807 | 7,816 | +9 |
| UI/Screens/watchlists_collections_screen.py | 14,324 | 14,330 | +6 |
| tldw_api/client.py | 16,687 | 16,689 | +2 |
| UI/Screens/settings_screen.py | **no row** | 33,091 | — (task-1378/31202) |
In total, 13 rows are over budget; app.py, TTS files, llm_screen and change_review sit exactly at their pins. No PR workflow runs `Tests/Architecture/*size_ratchet*` (test.yml is main-push-only, and derived-artifacts fast lanes don't list them).

### C5. Import-cost census (fresh isolated interpreter, warm .pyc, machine under load from parallel agents)
| measurement | result |
|---|---|
| `import tldw_chatbook.app` | 1.2–1.7 s; 1,772 modules, **681 tldw** |
| of which config.py module body (self) | 115 ms gc-off / 142 ms gc-on |
| of which app.py module body (self) | 104 ms gc-off |
| GC during boot import | 483 collections, 100–204 ms |
| app.py module-scope names | 470 imported; **404 used only inside function bodies**; 2 never referenced |
| TldwCli.__init__ reachable wiring | 25 methods, ~125 service/DB constructions |
| modules app loads beyond Chat's own closure | **367 tldw / 407 total, +488–535 ms**; 538 ms summed self (gc off) |
| chat_screen incremental after app (first paint) | 365–378 ms, 279 tldw modules (Chat pkg 147 ms self, Widgets 84, UI 44, Agents 29) |
| per-screen incremental after app | library 491–521 ms / 354 · personas 405–427 / 266 · settings 352–364 / 295 · stts 122–139 / 63 · llm 49–53 · evals 36–40 · watchlists 38–44 · mcp 34–36 · schedules 16–17 · home 4–5 |
| all 23 routed screens after app | 1,104 ms → 1,514 tldw modules (off-loop pre-importer) |
| `tldw_api.client` after app+chat | **575–854 ms, 56 modules, 1,257 pydantic models**; not resident after all 23 screens |
| dataclasses defined by app+Chat import | **1,605 → 537–590 ms** (gc off); app alone 1,041 → ~388 ms |
| pydantic models at app import | 264 tldw (293 total) → ~128 ms; top: kanban_schemas 76 (21.5 ms, TASK-21107), Backup_Recovery.journal 41 (17 ms), citation_source_locators 21, citation_trace_models 15 |
| dataclass decorator microbench (6 fields) | plain 163 µs · slots 178 µs · **frozen 310 µs** · frozen+slots 317 µs; 1,515 frozen+slots declarations in tree |
| full gen-2 GC after app+chat import | 250,482 tracked objects → **62.3 ms**; after `gc.freeze()` → **0.0 ms** |

### C6. App-only import subtrees (attributed to app.py's direct imports; gc-off self-time sum)
| app.py import | app-only self ms | modules | pre-paint need |
|---|---:|---:|---|
| Notes.file_notes_git_service (app.py:420) | 50.3 | 6 | no (Library File Notes) |
| TTS.TTS_Generation (+adapter_bootstrap) | 44.5 | 10 | no |
| css.Themes.themes (app.py:187) | 34.2 | 2 | only the active theme |
| Event_Handlers.TTS_Events.tts_events (app.py:403) | 30.2 | 8 | only the Message classes (for @on) |
| Home.active_work_adapter | 29.4 | 22 | no (Home not initial) |
| Kanban_Interop.server_kanban_service | 28.9 | 2 | no (TASK-21107/21239) |
| MCP.unified_control_plane_service + local_control_service | 36.6 | 9 | no (but pinned by a test) |
| Widgets.Settings_Widgets.speech_tts_panel_types | 17.9 | 8 | no |
| Actor_Packs.persona_coordinator | 16.2 | 15 | no |
| Widgets.splash_screen | 12.3 | 7 | yes if splash enabled |
| TTS.playground_types + audio_cpp_artifact_dependencies + adapter_types | 28.2 | 12 | no |
| Event_Handlers.STTS_Events.stts_events | 8.9 | 3 | only the Message classes |
| Evals.eval_orchestrator | 5.4 | 9 | no |
| **total attributable** | **~446** | | ~290 ms deferrable |

### C7. Private-SQLite connection census
| measurement | result |
|---|---|
| `connect_private_sqlite('db.evals', path)` ×12 (macOS) | **47.4 ms median (45–151)**, **12/12 spawned a `python -I -S private_sqlite_helper_entry.py` child** |
| plain `sqlite3.connect`+close | 0.09 ms |
| helper admission | 4 transient slots per process (HELPER_ADMISSION) |
| files calling connect_private_sqlite | 51 (94 call sites); 27 with no held-connection pattern (mostly recovery/cold); `_get_connection` re-implemented 20×, `_connect` 7×, `_held_connection` 6× |
| boot DB constructors (reopen, i.e. steady-state boot) | LibraryCollectionsDB 70.5 ms · WorkspaceDB 75.4 · SubscriptionsDB 74.8 · ScheduledTasksDB 58.1 · EvalsDB 60.6 (1 helper spawn each) → **~340 ms serial in TldwCli.__init__**; ClientNotificationsDB (TASK-21105 lazy shape) 0.1 ms / 0 spawns |
| first-ever run | LibraryCollectionsDB 591 ms/2 spawns; ScheduledTasksDB 550 ms/8 spawns; WorkspaceDB 201 ms |

### C8. Duplication census (qa/core-code-review-2026-09-17/candidates/dup_census.py re-run → scratch/structural/dup/)
dup_by_name 1,981 rows · dup_verbatim 381 · dup_shape 825.
- Runtime-paying families:
  - scope-service scaffold: `_maybe_await` 66 definitions, 873 call sites, `_enforce_policy` 43 verbatim (known TASK-32898 / ADR-177);
  - private-SQLite connection helpers (C7).
- Largest verbatim masses with no local runtime cost:
  - `Tools/remote_worker_bundle.py` vs `git_tool_impls` / `patch_tool_impls` / `local_tool_impls` / `path_validation` / `sensitive_paths`. This is a deliberate bundle for remote workers on Python 3.10; the biggest pairs are git_diff 128 lines, run_git 104, _denylist_pathspecs 95 and execute_pinned_operation 94.
  - `Evals/dataset_loader` vs `eval_runner` (legacy).

### C9. Dead or legacy code resident at boot
| module | resident | via |
|---|---|---|
| Kanban_Interop.* (5 modules) + tldw_api.kanban_schemas | boot | app.py:574 (known TASK-21107/21239) |
| Evals.eval_orchestrator / eval_runner / task_loader | boot | app.py:746 → `_wire_evaluation_services` builds EvaluationOrchestrator + EvalsDB (see F1) |
| Tools_Settings_Window, NewIngest, Tamagotchi widget, Confluence, MediaWindow_v2, SiteConfigSettings, dictation_service, route_inventory | not resident (checked) | — |


# pat-io-on-loop

## summary
Whole-tree sweep (2,541 .py files) for NON-DB blocking work on the Textual event loop. The work combined an AST reachability scan with manual tracing and isolated micro-benchmarks. The scan followed loop contexts (on_*/@on/compose/on_mount/watch_*/action_*/timer callbacks/async def) into blocking primitives: same-module depth 3, plus cross-module resolution of imported helpers. Benchmarks ran with HOME/XDG/TLDW_CONFIG_PATH in scratch, TLDW_TEST_MODE=1, a null keyring backend and PYTHONPATH pinned to the audit tree.

Overall the offload discipline is good: 996 asyncio.to_thread, 235 thread=True and 50 run_in_executor sites. Keyring reads are all TTL-cached or thread-bounded (TASK-32921/32922/32926). Sync HTTP runs in workers. The send-path git @-references use to_thread. Audio playback uses an executor.

The dominant NEW finding is not a raw open()/subprocess call. It is the Backup_Recovery storage-admission layer. Every @content_call method, every acquire_storage() and every un-scoped get_user_data_dir() re-derives admission evidence from disk on each call: about 480–2,000 verified-parent open() syscalls and about 12–15 ms per acquire (measured). TASK-32804.1 fixed this only for warm get_cli_setting, by bypassing it. The same tax still lands on the loop in three hot places:
- (1) Per send, capture_skill_context_maximum runs a trust status plus a fingerprint digest for every installed user skill, synchronously inside _admit_console_turn_to_runtime. Measured: ~79 ms with 1 skill and 313–665 ms with 5 skills.
- (2) On every Console mount/resume, a coroutine (not thread) worker awaits LocalSkillsService.get_context. Measured: ~58 ms even with zero user skills.
- (3) get_user_data_dir() is uncached at ~47 ms warm (5 admissions, 1,591 opens). It has 203 call sites, 29 of them reachable from TldwCli.__init__; 14 DB-path helpers alone cost 728 ms in isolation. It is also hit from a 5 s Schedules timer and 23 other UI loop contexts.

Secondary findings:
- webbrowser.open/App.open_url runs synchronously on the loop; on macOS it waits on osascript (≥73 ms measured).
- A structural 'async def facade over sync file I/O' pattern in the local Skills/Chatbooks services (86 @content_call methods) makes awaiting them on the loop look safe.
- A latent unbounded base64-audio JSON history.

Structural note: the fix point for the first three findings is shared (amortize admission in storage_admission, or memoize get_user_data_dir the way 32804.1 memoized config). The per-site fix is to move the skill work off the loop via to_thread, as SkillsScopeService.get_library_user_content_evidence already does. Measurement caveat: the isolated HOME is 14 path components deep and _open_verified_parent cost is linear in depth, so a real ~6-component HOME is plausibly ~40–50% of the per-call numbers. The per-skill and per-call multipliers still hold.

## clean areas
- Keyring on the loop: all 35 call sites (9 files) are either TTL/single-flight cached (runtime_policy/server_credentials.py TASK-32922, Skills_Interop/skill_trust_store.py _cached_keyring_read TASK-32921, Media_Generation/config_machinery.py keyring_get TASK-32926), thread+timeout bounded (Audio/voiceprint.py:295-320), lazily built off boot (app.py:8847 _resolve_server_credential_store), or default-off (Chat/citation_trace_identity.py, canonical_writes_enabled=False)
- LLM_Calls/anthropic_subscription.py: Keychain `security` subprocess is behind a background thread + 5 s TTL for UI readiness (_SubscriptionReadinessCache)
- Chat/console_chat_controller.py:10106-10121 @-reference expansion incl. `git diff` subprocess runs via asyncio.to_thread
- Subprocess sites (60 in 48 files): all non-UI callers run in worker threads or use asyncio.create_subprocess_exec (Utils/install_clipboard.py, Notes/file_notes_git_service.py AsyncGitProcessRunner, STTS _convert_audio_format); only 3 UI Popen launches remain (open-folder/open-video), fork/exec-only, P3
- TTS/audio_player.py: the macOS 100 ms sleep in SimpleAudioPlayer.play runs inside AsyncAudioPlayer's executor, not on the loop
- Sync HTTP (requests/httpx.Client, 38 sites): none reachable on the loop. LLM_Management_Window Ollama 3 s poll is a gated, widget-owned worker (task-15211/22220); settings_image_gen_defaults probes and academic_providers run in workers (academic_providers builds a client per search, P3 network hygiene only)
- time.sleep (66 sites/46 files): all in worker threads or subprocess supervisors except the lock-contention polls inside config._default_data_root_lock (config.py:9280) and Backup_Recovery/admission.py:217/646, which are reached via get_user_data_dir/acquire_storage (see findings)
- Widgets/Console/console_transcript.py:5680 _append_paint_log file append: env-gated debug only (TLDW_TRANSCRIPT_PAINT_LOG)
- UI/Screens/library_screen.py:26388 prompt import: file reads already via asyncio.to_thread; only a single-folder iterdir on the loop (P3)
- UI/Screens/watchlists_collections_screen.py:14165 opens the browser in a thread worker (the good pattern)
- UI/Screens/trajectory_screen.py:728 revision poll: rebuild runs in a thread worker
- Console dictation availability probe (Chat/console_voice_input.probe): find_spec only, cheap
- UI/Console_Modules/image.py:729 and Widgets/Library/library_media_image_preview.py: header-only PIL reads / bounded thumbnails
- Tools/workspace_file_roots.py registry: process-cached, no per-call path resolution
- Chat/prompt_history.py: data-dir resolved once per runtime
- Skills per-keystroke `$` popup uses cached _console_skill_candidates (no per-key service call)

## census
| Site / pattern (untruncated `rg -c` totals) | Operation | Loop-reachable context found | Frequency |
|---|---|---|---|
| `@content_call(` 86 methods / 6 files (Skills, SkillTrust, Chatbooks) + `acquire_storage(` 88 / 45 files | storage admission: ~12–15 ms and ~480–2,000 `posix.open` per acquire (measured) | Console send capture, Console mount/resume, Evals on_mount, Console save-as-Chatbook | per send / per visit / per click |
| `Chat/console_chat_controller.py:1300` `capture_skill_context_maximum` | per-skill trust status + fingerprint digest (admission + dir walk + sha256) | sync inside `_admit_console_turn_to_runtime` (`UI/Console_Modules/wiring.py:256`), called from async `_stage_normal_chain` (`prompt_queue.py:996`) | **every send**; 79 ms/1 skill, 313–665 ms/5 skills (measured) |
| `UI/Console_Modules/skill.py:57` → `LocalSkillsService.get_context` | same as above, minus the digest | `run_worker(coro)` at `chat_screen.py:16935` (mount) and `:23831` (resume) | every Console visit; 58 ms with 0 user skills (measured) |
| `get_user_data_dir()` 203 calls / 60 files | 5 admissions + data-root lock + `secure_private_directory`; ~47 ms warm (measured) | 23 UI loop contexts (AST scan), incl. the Schedules 5 s timer (`schedules_workbench.py:858/1142`), Evals on_mount, briefing play, `/stop`, project-skills timer; 29 resolutions reachable from `TldwCli.__init__` | boot + timers + clicks |
| `get_*_db_path()` 14 helpers (`config.py:9505-9777`) | wrap `get_user_data_dir` | boot wiring; RAG service per search (worker) | 728 ms for 14 warm calls (measured) |
| `webbrowser.open(` / `App.open_url(`: 8 sites | `osascript` popen + wait on macOS; `xdg-settings` subprocess on first Linux call | 5 webbrowser + 2 open_url sync on the loop (Console link click, Settings About, Library ×2, change-review PR, env row); 1 threaded (watchlists) | per link click; ≥73 ms (measured osascript no-op) |
| `subprocess.run/Popen/check_*` 60 / 48 files | subprocess | only 3 UI `Popen` launches on the loop (`ChatbookExportManagementWindow.py:1172`, `ChatbookCreationWizard.py:1099`, `chat_screen.py:19804`) | rare click, fork/exec cost only (P3) |
| `keyring.*` 35 / 9 files | OS credential store | none uncached on the loop | – |
| `time.sleep(` 66 / 46 files | sleep | only contention polls in `config.py:9280` and `admission.py:217/646` (reached via the admission path) | contention only |
| `requests.*`/`httpx.Client(` 19+19 sites | sync HTTP | none on the loop | – |
| `.read_text(` 121/58, `.read_bytes(` 52/33, `open(` 607/228, `json.load(` 68/42, `tomllib.load` 37/25 | file I/O | on the loop: skills index/SKILL.md (above); theme-editor TOML scan (known TASK-33078); speech blend JSON in `speech_settings_mixin.py:853/536` (small, P3); media-canvas compose parses meeting `transcript.jsonl` (`Library/meeting_speaker_rename.py:81`, 5.4 ms per 1 h meeting); Settings image/video-gen panels re-parse on-disk config.toml in compose (`settings_image_gen_defaults.py:279`), P3 | per compose / category open |
| `shutil.which(` (in 112 shutil calls) | PATH scan | `LLM_Management_Window.py:859/900/957`, `settings_video_gen_panel` compose ×3, `video_player_screen.on_mount`, `ssh_available` | per view open (sub-ms to a few ms, P3) |
| Offload primitives: `asyncio.to_thread` 996/216, `thread=True` 235/69, `run_in_executor` 50/17 | – | – | discipline is widespread |


# pat-logging

## summary
Logging sweep over all of tldw_chatbook (2,541 .py files, 8,914 log call sites found by AST). Individual call sites are mostly clean. Streaming, keystroke, timer and render paths carry almost no logging, and the few f-strings that interpolate large values sit on cold or off-loop paths. The cost comes from how the pipeline is set up.

(1) Logging_Config.py forwards every loguru record to stdlib through a sink at level="TRACE". loguru's min_level early return therefore never fires. Every logger.debug pays about 7-8 us instead of 0.15 us. Every opt(lazy=True) guard is defeated, including TASK-275's per-SQL guard (12.7 us instead of 0.46 us). Every dropped opt(exception=True).debug formats a full backtrace, about 50 us instead of 1 us. The app.py "Disable debug logging for performance" only changes stdlib levels, so it has no effect on loguru.

(2) Every INFO and higher record is redacted three times. The rotating file handler's shouldRollover re-runs the redacting formatter, emit runs it again, and the Logs-buffer handler redacts a third time. Each pass is about 66 us for a 140-character line. The record is then written and flushed synchronously on whichever thread emitted it, the event loop included. The measured total is about 340 us per INFO record, or about 233 us before the Logs buffer attaches. A real steady-state boot capture emits 72 records before the first screen mounts, which is about 17 ms. A first-run boot emits 201, about 47 ms or more. Each navigation emits 3 records on the loop, about 1 ms.

(3) INFO volume is high where it should be DEBUG. The DB layer logs one INFO per row write, which dominates bulk import. App.on_worker_state_changed logs a WARNING on every transition of App-owned workers outside a short allowlist.

A proposed single-pass pipeline was measured at 108 us per INFO record (down from 339) and 0.14 us per dropped debug. Moving the file and buffer handlers behind a QueueHandler would remove the rest from the loop.

Suggested PR grouping:
- PR-A, logging pipeline: F1, F2, F9, and optionally a QueueHandler. It is small and centralised in Logging_Config.py, log_sanitizer.py and app.py; it needs redaction regression tests.
- PR-B, log volume: F3, F4, F6, F7. Mostly level demotions.
- PR-C, handler and buffer hygiene: F5, F8.

Out-of-scope observation for the algorithmic lane: Chunking/engine/chunker.py:1406-1409 runs a per-character Python loop (unicodedata.category) over the whole input on every chunk_text call.

## clean areas
- Modern Console streaming path: LLM_Calls/hosted_chat_streaming.py, hosted_chat.py, legacy_line_stream.py, qwencloud_streaming.py, llamacpp_bounded.py, Agents/agent_stream.py, Agents/native_tools.py have 0 debug/info calls; hosted_provider_engine.py has 1 and console_provider_gateway.py has 3, none per chunk
- Legacy LLM_API_Calls stream generators: per-line logging exists only in the Cohere (per-event debug) and HuggingFace generators; the rest is warning-on-malformed-line only
- Chat/console_chat_controller.py (23 debug, all in exception or rare paths), console_chat_store.py (3), console_turn_grouping and console_runtime loops are exception-only
- Per-keystroke handlers: AST scan of on_key, Input.Changed, TextArea.Changed, watch_*, render* and resize handlers found logging only in ChatbookCreationWizard (F6) and exception branches
- set_interval callbacks: AST scan found no non-exception logging except the daily perform_media_cleanup
- config.get_cli_setting and load_settings cache-hit paths are log-free
- Metrics/metrics_logger._log_metric is gated off by env (TLDW_METRICS_LOGGING); Metrics/metrics.py uses stdlib logging.debug, which is cheap when disabled
- 38 stdlib-logging modules (Chat_Functions payload loops, Chat_Dictionary_Lib per-entry, Client_Media_DB_v2 transactions): stdlib isEnabledFor drops them at about 0.19 us
- DB/sql_logging.preview_params bounds previews at 200 chars; the eager-BLOB stringify from the 2026-07-16 A1 finding stays fixed (only laziness regressed, see F2)
- TextualHandler returns early when devtools are not connected (only format cost, WARNING+ only)
- Logs screen route is not reusable, so its live append path is cleared on unmount
- Expensive-interpolation sweep (json.dumps, pformat, model_dump, query_one or stat inside log f-strings): only WebSearch_APIs:772 (F7), video_processing:960 (truncated, cold) and audio_processing:660 (one stat, cold)
- Utils/optional_deps loop logs are exception-only; UI/Console_Modules/workspace.py per-row debug calls are exception-only

## census
### Log call-site census (AST over tldw_chatbook/, origin/dev 840ed2ca58)

| metric | count |
|---|---|
| total log call sites | 8,914 (warning 2,738 · error 2,240 · debug 1,924 · info 1,855 · exception 126 · critical 15 · success 10 · trace 6) |
| `debug(f"...")` f-string sites (AST, any logger name) | 831 in 175 files (multi-line regex over common logger names: 766 in 160) |
| `info(f"...")` f-string sites | 1,221 |
| `logger.opt(lazy=True)` call sites | 5, all defeated by the TRACE sink |
| `opt(exception=True).debug(` sites | 186 in 48 files (top: realtime.py 21, watchlists_collections_screen 20, workspace.py 11, app.py 11, personas_screen 10) |
| `isEnabledFor`/level guards | 4 |
| log calls inside loops | 825 (debug/info in loops: 299; INFO in loops: top chatbook_importer 23, WebSearch_APIs 10, Prompts_Interop 9) |
| modules importing loguru `logger` / using stdlib `getLogger` | 569 / 38 |

### Top 25 modules by `logger.debug(f"...")` (AST)
| module | debug f-str | all debug | all info |
|---|---|---|---|
| DB/ChaChaNotes_DB.py | 51 | 57 | 129 |
| Local_Ingestion/transcription_service.py | 39 | 61 | 154 |
| DB/Client_Media_DB_v2.py | 37 | 62 | 75 |
| LLM_Calls/LLM_API_Calls.py | 29 | 48 | 4 |
| LLM_Calls/Summarization_General_Lib.py | 23 | 138 | 32 |
| Character_Chat/Chat_Dictionary_Lib.py | 22 | 25 | 4 |
| LLM_Calls/Local_Summarization_Lib.py | 21 | 124 | 23 |
| app.py | 19 | 50 | 83 |
| Character_Chat/Character_Chat_Lib.py | 16 | 32 | 21 |
| Utils/optional_deps.py | 15 | 17 | 31 |
| UI/Wizards/ChatbookCreationWizard.py | 14 | 17 | 1 |
| RAG_Search/ingestion_indexing.py | 14 | 15 | 5 |
| Local_Ingestion/PDF_Processing_Lib.py | 13 | 14 | 19 |
| Embeddings/Embeddings_Lib.py | 12 | 16 | 15 |
| Chunking/engine/strategies/tokens.py | 12 | 15 | 4 |
| UI/Speech/speech_playback_mixin.py | 11 | 31 | 8 |
| TTS/audio_player.py | 11 | 16 | 6 |
| RAG_Search/simplified/simple_cache.py | 11 | 11 | 5 |
| RAG_Search/simplified/embeddings_wrapper.py | 11 | 13 | 20 |
| UI/Wizards/BaseWizard.py | 10 | 10 | 11 |
| LLM_Calls/LLM_API_Calls_Local.py | 10 | 13 | 12 |
| Chunking/engine/chunker.py | 10 | 23 | 4 |
| Chunking/engine/base.py | 9 | 10 | 1 |
| Chat/Chat_Functions.py | 8 | 25 | 14 |
| Web_Scraping/WebSearch_APIs.py | 8 | 15 | 37 |

### Measured per-event cost (isolated micro-benchmarks replicating the shipped sink and handler config, Py3.12, loguru 0.7.3)
| event | shipped | if loguru sink level = INFO |
|---|---|---|
| `logger.debug("const")` (dropped by stdlib) | 7.0-8.4 us | 0.14-0.16 us |
| `execute_query` lazy SQL debug (TASK-275 guard) | 12.7 us | 0.46 us |
| `opt(exception=True).debug`, 5/25-frame traceback | 46.6 / 56.9 us | 0.8 / 1.8 us |
| `opt(lazy=True).debug(repr(40-msg payload))` | 56.9 us (lazy evaluated anyway) | about 0 |
| `logger.info(140-char line)`: file (rotating, redacting) + Logs buffer | 339 us (3 sanitizer passes, counted) | proposed 1-pass: 108 us |
| same, file handler only (pre-Logs-buffer / splash disabled) | 233 us | |
| `logger.info(2000+ char line)` | 2,128 us | |
| `redact_log_line` one pass, 140 chars | 66 us (12 credential regexes 18 us, `redact_user_paths` 15 us incl. `Path.home()` 3.8 us) | |

### Real volume (captured app log `Docs/superpowers/qa/console-custom-endpoints-uat-2026-09-13/captures/13-app-log-crash-context.txt`)
Counts are INFO+ records. Steady-state boot: 72 records before the first screen mounts, 78 to "startup complete". First-run boot: 201. Screen navigation: 3 INFO. Settings save: 5 INFO.

### Isolated DB probe (scratch profile, real `CharactersRAGDB`)
Fresh init: 75-81 INFO and 103-111 DEBUG. Reopen: 2 INFO and 8 DEBUG. Each `add_message`: 1 INFO and 2 DEBUG. Each `get_messages_for_conversation`: 1 DEBUG.


# pat-memory

## summary
Scope: a memory growth, retention and leak sweep across all 2,541 .py files under tldw_chatbook in the audit tree (origin/dev 840ed2ca58). I used three AST censuses. One covered module-level and class-level mutable containers, lru_cache and deque. One found instance attributes that are only ever added to. One grouped the mixins of the reusable Home, Chat and Library screens. I also ran rg sweeps for whole-file reads, base64, queues, RichLog buffers, timers and id()-keyed caches. I triaged every app-lifetime owner the censuses surfaced and ran 5 isolated micro-benchmarks. Each used HOME, XDG and TLDW_CONFIG_PATH pointed at the scratch dir, and any database was a scratch file.

Overall health: memory management on the long-lived Console owners is careful. ConsoleChatStore purges about 40 per-session maps when a session closes. The agent bridge prunes its live slots each turn. The token-estimate, stall-watchdog, read-ledger, image-render, log and terminal-scrollback buffers are all bounded. The worst problems here are per-write and per-call costs that create large temporary allocations, not slow leaks.

Headline finding (F1, new, measured): every ChaChaNotes connection installs sqlite3 set_trace_callback only to spot BEGIN, COMMIT and ROLLBACK. CPython then expands the bound SQL, hex-encoding any BLOB, for the statement and again for every trigger and FTS sub-step.
- Inserting a message with a 3 MiB image through db.add_message took 1,203 ms versus 9 ms for a text-only message.
- On a plain connection with 5 triggers, the same insert took 1,095 ms with a no-op trace callback and 0.7 ms without one.
- The Console durable turn commit holds BEGIN IMMEDIATE for that long for each image sent. Every BLOB table on ChaChaNotes pays this tax, including attachments, avatars, trace artifacts and exchanges.

Other major items:
- F2: the normalized trace (on by default) re-projects every history message on each provider call, base64-encoding every image in history. Measured 27 ms and 55 MiB peak versus 4.9 ms and 1 MiB when the history has no images.
- F3: opening a conversation loads every image BLOB of every branch and keeps it for the whole session.
- F4: the web_fetch cache is bounded by entry count but not by bytes.
- F5: parakeet-mlx, the default speech-to-text engine on macOS, decodes the whole audio file as float64 before chunking.

Correction to an earlier hypothesis: exchange-capture retention (F6) is off by default. It only applies when trace_legacy_writes is enabled, so I rated it P3.

Structural note: TASK-24452 made the Home, Chat and Library screens reusable, so their per-session maps now live as long as the app. ChatScreen's maps are part of F7.

Out of scope (for other agents): importing console_exchange_capture writes recovery-bootstrap files under HOME at import time. console_transcript.py imports PIL at module scope on the Chat path.

## clean areas
- tldw_chatbook/Chat/console_chat_store.py: _purge_claimed_session_runtime_state pops ~40 per-session/per-message maps on close; _stream_chunks_by_message popped at every terminal path; _session_mru filtered to live ids
- tldw_chatbook/Chat/console_agent_bridge.py: _live slots pruned per turn (_prune_live_run_slots), _historical_cache popped per conversation, buddy_tool_sequences is run-local
- tldw_chatbook/Widgets/Console/console_transcript.py _message_signature_cache pruned; Chat/console_image_view.py ConsoleImageRenderCache LRU (IMAGE_CACHE_MAX_ENTRIES)
- tldw_chatbook/UI/Library_Modules/library_media_controller.py preview image cache bounded by LIBRARY_MEDIA_PREVIEW_CACHE_LIMIT
- tldw_chatbook/Utils/token_counter.py _ESTIMATE_CACHE (hash-keyed, clear-on-overflow 4096, pins no text)
- tldw_chatbook/Chat/stream_stall_watchdog.py _SESSION_TRACKERS capped 512
- tldw_chatbook/Agents/fs_read_ledger.py (512/run cap); LocalToolProvider is per-run in Console
- tldw_chatbook/Terminal/screen_model.py scrollback bounded 5,000 lines / 4 MiB; Terminal/session_manager.py pops closed sessions
- Logging: app._log_records deque(maxlen=10000), Logs_Window deque + RichLog(max_lines); RichLogHandler asyncio.Queue never attached in master shell (#app-log-display absent)
- UI/LLM_Management_Window.py RichLogs lack max_lines but server stdout/stderr go to DEVNULL (server_lifecycle.py:644) so volume is tiny
- RAG_Search/simplified/simple_cache.py (entries + max_memory_mb), Embeddings_Lib OrderedDict cache, Utils/text_wrap_index.py segment cache capped
- Audio/meeting_capture.py meters (600 s horizon) and PCM rings trimmed; Audio/duplex_transport.py and acoustic_isolation.py histories use maxlen
- Subscriptions/site_config_manager.py RateLimiter trims timestamps per check
- Size-guarded whole reads: Tools/local_tool_impls.py fs_read (MAX_READ_FILE_BYTES), Chat/console_references.py (max_bytes), chat_image_events.py (max_image_bytes), personas avatar (PERSONAS_AVATAR_MAX_BYTES)
- Backup_Recovery/visual_identity_participants.py _issued released on cancel (forget_cancelled) and publication cleanup
- app.py _reusable_screen_instances bounded to 3 routes by design (TASK-24452)
- TTS/TTS_Generation.py task/response sets discard via done-callbacks (AST false positives)
- Class-level mutables: TTSEventHandler._request_cooldown bounded (MAX_COOLDOWN_ENTRIES + cleanup), AgentRunsDB._swept_paths intentional, TokenChunkingStrategy._failed_tokenizers intentional
- Chat/console_trace_service.py id()-keyed capability maps pruned via _prune_unreferenced_parents
- Dead but harmless: Notifications/notification_presentation.py store, TTS CostTracker, Subscriptions TokenBudgetTracker, Media_Creation _generation_cache (no prod constructors or writers)

## census
| Pattern (untruncated counts, audit tree) | Count | Triage |
|---|---|---|
| Module-level empty mutable containers | 134 (108 mutated inside functions: 27 weak, 81 strong) | Strong ones checked: most are bounded registries or lock maps. Unbounded/risky: web_tool_impls `_fetch_cache` (F4). Metrics `_metrics_registry` has label-cardinality growth (F12). |
| Class-level mutable attributes (not BINDINGS) | 18 | None harmful; all bounded or intentionally shared. |
| `lru_cache`/`cache` with maxsize=None | 1 (a no-arg function) | fine |
| `lru_cache` on instance methods (retains self) | 2 (ModelCapabilities.is_vision_capable is a singleton; Windows-only `_Native`) | fine |
| `deque()` without maxlen / with maxlen | 48 / 31 | All unbounded ones checked: most are trimmed by hand. One real bug: the emote feed loses maxlen after a restore (F9). |
| Instance attributes grow-only (AST heuristic, excluding Third_Party) | 353 candidates | Many false positives (done-callback discards, getattr-by-name purges, per-call builders). Real on app-lifetime owners: ConsoleChatController 14 maps, ChatScreen 6, retrieval/canvas controllers (F7), reranker (F8), trace `_surface_ref_cache` (F11). |
| Instance `*cache*` dicts | 49 | 1 unbounded on a long-lived owner (reranker, F8). The others evict or are screen-scoped. |
| `RichLog(` / with max_lines | 18 / 2 | Unbounded ones are low-volume |
| `.read_bytes()` / `.read_text(` / `.read()` | 51 (32 files) / 122 / 82 | User-facing ones are size-guarded. Unbounded decode: parakeet `sf.read` (F5). |
| `b64encode(` / `b64decode(` | 60 / 67 | Hot one: semantic-revision envelopes (F2) |
| Unbounded `asyncio.Queue()` / `queue.Queue()` | 6 / 16 | Producer-consumer pairs; the logging queue is never attached |
| `set_interval(` / `set_timer(` | 71 / 126 | No timer lambdas capturing large payloads found. 3 are app-owned timers created from widgets. |
| weakref containers and refs | 126 | Good pattern, widely used |
| `set_trace_callback` in prod | 1 (base_db.py:738, on every ChaChaNotes connection) | F1 |
| Measured: trace-callback tax, 5 triggers | 3 MiB blob: 0.7 ms -> 1,095 ms. Text 50 KB: 0.019 -> 1.53 ms. Text 500 KB: 0.12 -> 15.5 ms | F1 |
| Measured: `db.add_message` | Text 9.4 ms; with 3 MiB image 1,203 ms (cProfile: 1.17 s in cursor.execute C code) | F1 |
| Measured: `project_semantic_revision_provider_messages` for 100 revisions | Text-only 4.9 ms / 1.0 MiB peak; 5 x 3 MiB images 27 ms / 55 MiB peak | F2 |
| Measured: exchange capture retained per provider call | No tools: 4.7 KB + 10.3 KB blob. 30 tool schemas: 56 KB + 26 KB blob | F6 (legacy path, default off) |


# pat-network

## summary
HTTP client lifecycle sweep over all of tldw_chatbook at 840ed2ca58. An AST census found 87 construction or bare-call sites: 33 httpx.AsyncClient, 15 httpx.Client, 13 create_default_session, 7 requests.Session, 6 build_httpx_async_client, 5 build_httpx_client, 4 bare requests.get/post, 3 aiohttp.ClientSession and 1 AsyncHTTPTransport. Only about 12 are long-lived pooled clients. Timeouts are in good shape: TASK-19830's DefaultTimeoutSession covers every requests session, the AST found 0 timeout-less bare requests calls, and the 3 unbounded reads are deliberate and bounded elsewhere.

The problems are about client lifetime, not timeouts. Two costs are measured here and apply across the whole codebase.

(1) Per-send transport churn. Every hosted LLM call builds a fresh requests.Session and closes it after the stream. So nothing reuses a connection across sends, agent turns, summarization chunks or deep-search relevance calls. On loopback, a fresh session per POST costs 15.2 ms against 0.9 ms reused. Most of that 14 ms is urllib3 reloading certifi for each new connection. A real provider adds DNS plus 2-3 RTTs of TCP/TLS on top (F1).

(2) SSL context rebuild per client. httpx rebuilds an SSLContext from certifi on every client construction (12-24 ms, versus 0.09 ms with a cached context), because tls_trust.httpx_verify() returns a bare True. About 15 sites build clients on the Textual event loop. These include the boot-time model-catalog refresh right after the initial screen push, every watchlist feed/URL check (the scheduler is a coroutine worker on the app loop), and the Settings, Console and wizard probes (F2, F3).

Separately, the Research window's Ask Follow-up handler runs a blocking non-streaming chat_api_call inside an async def on the loop. It freezes the whole UI for the full LLM round-trip (F4). Its sibling _default_gap_fn was already fixed for exactly this.

Model lists and catalogs are cached properly: a 24 h disk store, a TTL memo on context windows, and a 30-min robots cache. The one exception is llama.cpp's per-send /health or /v1/models pre-probe (F5). No hot path reads a streamed response fully into memory: guarded fetches are capped by max_bytes and model downloads stream to disk. MCP has no HTTP client (it is stdio only).

Structural notes:
- Subscriptions/scrapers plus web_scraping_pipelines (3,559 LOC, 10 per-call client sites) is dead code, and a repo test already says so.
- Article_Extractor_Lib.scrape_article is currently inert because load_and_log_configs() returns {} and then raises KeyError. Once repaired, it will launch Chromium per URL (F9).

Suggested PR grouping:
- PR-A: tls_trust cached SSLContext plus routing the raw httpx.AsyncClient sites through build_httpx_* (F2, and it cuts F3/F6/F7 construction cost).
- PR-B: a shared pooled HTTPAdapter in egress.create_default_session (F1).
- PR-C: watchlist per-run client (F3).
- PR-D: offload the research follow-up call (F4).
- PR-E: hygiene (F5-F9).

## clean areas
- tldw_api/client.py:1312 -- one lazily-built AsyncClient per TLDWAPIClient, and runtime_policy/bootstrap.LegacyConfigServerClientProvider caches the client (server-mode Interop calls are pooled)
- Chat/console_provider_gateway.py app-loop client -- plain llama.cpp sends and readiness probes reuse one pooled AsyncClient per loop (_active_http_client), with explicit Timeout
- TTS/base_backends.py:121 -- API TTS backends are cached per backend id by TTS_Backends manager (TTS_Backends.py:177), so the client is pooled
- Embeddings/Embeddings_Lib.py:530 -- one requests.Session per embedder closure, reused across batches, timeout=30 per post
- LLM_Management/snapshot_client.py:88, Utils/github_api_client.py:179 (per-loop cache), Evals/word_bench/capture_client.py:206, Media_Creation/swarmui_client.py:142, Confluence auth session -- long-lived clients
- Model catalog caching -- LLM_Provider_Catalog disk store (24h stale_after_hours) plus discovery_cache; Console context-window memo with TTL and background refresh; web_fetch robots cache (30-min TTL, negative caching)
- Timeouts -- the AST sweep found 0 timeout-less bare requests.* calls; every create_default_session gets a (10,30) default; httpx clients without timeout= fall back to the finite 5 s default; the 3 unbounded reads (audio_cpp request client read=None, llamacpp_bounded timeout=None under an asyncio deadline, SSE read re-armed per chunk) are deliberate
- Utils/egress.py guarded_fetch_* -- responses capped by max_bytes, async DNS via loop.getaddrinfo, redirects re-validated per hop
- Model/installer downloads (kokoro.py:194, parakeet_v2_installer.py:151, diarizer_engine_onnx.py, Model_Artifacts/acquisition.py per-batch client) stream to disk with size caps
- Off-loop sync HTTP correctly threaded -- settings_image_gen_defaults probe (@work(thread=True)), Library ingest_preflight (@work(thread=True)), stream_resolve (asyncio.to_thread), research academic lanes (to_thread), web_deep_search DNS/robots (dedicated executor), relevance chat_api_call (to_thread), Research _default_gap_fn (_offload_pipeline_call), evals/library analysis chat_api_call (worker threads), run_webhooks (own thread + Runner)
- MCP/ and Agents/mcp_tool_provider.py -- the MCP client is stdio-only with no HTTP client
- *_Interop packages -- only Research_Interop/academic_providers.py and Skills_Interop/skill_remote_fetch.py touch httpx directly; the rest go through the cached TLDWAPIClient
- Streaming parsers (hosted_chat_streaming iter_content(8192), iter_lines on requests streams) -- no whole-response buffering or quadratic accumulation; the deepcopy of the payload per attempt in owned_json_post measured 0.51 ms for a 200-message plus 30-tool payload (negligible)
- Console readiness and llama.cpp local discovery (discover_local_servers) share one client across candidate probes

## census
### HTTP client construction census (AST sweep of tldw_chatbook/, untruncated)

**87 sites in total:**

| Constructor | Sites |
|---|---|
| `httpx.AsyncClient` | 33 |
| `httpx.Client` | 15 |
| `create_default_session` | 13 |
| `requests.Session` | 7 |
| `build_httpx_async_client` | 6 |
| `build_httpx_client` | 5 |
| bare `requests.post` / `requests.get` | 3 / 1 |
| `aiohttp.ClientSession` | 3 |
| `httpx.AsyncHTTPTransport` | 1 |

**Grep counts:**
- `rg -c`: 19 `httpx.Client(` in 8 files, 35 `httpx.AsyncClient(` in 24 files, 8 `requests.Session(` in 6 files.
- Factory callers: `create_default_session` 18 in 8 files, `build_httpx_async_client` 7 in 6, `build_httpx_client` 6 in 4, `guarded_fetch_requests` 10 in 7, `guarded_fetch_httpx_async` 22 in 10.

**Measured (py3.12, httpx 0.28.1, requests 2.32.5, urllib3 2.6.3):**
- New `httpx.AsyncClient` with verify=True: 12.8–24 ms (max 65 ms). `httpx.Client`: 16 ms. `AsyncHTTPTransport`: 11.5 ms.
- With `verify=<cached SSLContext>`: 0.09–0.14 ms.
- urllib3 context plus certifi load, paid for every new HTTPS connection: 14.3 ms.
- Loopback TLS POST: fresh Session 15.2 ms vs reused Session 0.9 ms.
- `load_verify_locations` releases the GIL (heartbeat gap 1.7 ms).

| Area / sites | Client | Pooled? | Timeout | Runs on | Path / frequency |
|---|---|---|---|---|---|
| LLM_Calls/hosted_chat.py:710 (groq, openrouter, deepseek, mistral, moonshot, zai, databricks, together, fireworks, cerebras, custom-hosted) | create_default_session | **NO** (per call, closed with the stream) | explicit + (10,30) default | to_thread worker | **per send / agent turn** |
| LLM_API_Calls.py:855, 950 (openai), 1750 (anthropic), 2718 (cohere), 3559 (google), 4299 bare post + 4386 (huggingface); qwencloud.py:1176 | create_default_session / requests.post | **NO** | yes | to_thread worker | **per send** |
| Summarization_General_Lib.py:1039, Local_Summarization_Lib.py:90 | create_default_session | **NO** | yes | worker | per chunk |
| LLM_API_Calls_Local.py:282, 1093 (local openai-compatible, kobold) | create_default_session | NO (usually localhost http) | yes | worker | per send |
| LLM_API_Calls.py:485 get_openai_embeddings | create_default_session | NO | yes | - | DEAD (0 callers) |
| Chat/console_provider_gateway.py:3600 | build_httpx_async_client, one per event loop | yes on app loop; **per run** on agent lifeline loops | explicit | app loop / lifeline thread | llama.cpp sends, probes, agent runs |
| LLM_Provider_Catalog/openai_compatible_model_discovery.py:879 | build_httpx_async_client | NO | yes | **APP LOOP** | boot refresh (per stale provider, 24 h), Settings Discover |
| Chat/local_server_discovery.py:529, 580 | build_httpx_async_client | per probe (shared across candidates) | yes | **APP LOOP** | Console setup card (once per screen), settings-modal probe |
| settings_endpoint_probe.py:526, 634; server_switch_modal.py:225; FirstRunSetupWizard.py:951 (httpx 5 s default); first_run_voice_step_state.py:273; vllm_connection.py:614 | raw httpx.AsyncClient | NO | yes / default | **APP LOOP** | user-initiated probes |
| Subscriptions/monitoring_engine.py:1053, 1759; local_watchlists_service.py:2527, 2598 | raw httpx.AsyncClient | **NO** (per feed, per URL) | 30 s | **APP LOOP** (scheduler coroutine worker) | every due check / Check Now, ×N URLs |
| LLM_Calls/llamacpp_bounded.py:385, 390 | AsyncHTTPTransport + AsyncClient, keepalive 0 | NO (deliberate) | None, bounded by asyncio deadline | **APP LOOP** (WorkflowSession) | per workflow model step |
| Subscriptions/scrapers/* (10 sites) | httpx.AsyncClient | NO | 30 s | - | DEAD (0 importers) |
| Research_Interop/academic_providers.py ×9 | httpx.Client() | NO | per-request timeout= | to_thread | per research query × up to 10 lanes |
| Image_Generation/http_client.py:120 (fetch_json:173, fetch_image_bytes:334, image_format_utils:248) | build_httpx_client | **NO** | 120 s | worker | **per poll (1–2 s)** during generation |
| Tools/web_tool_impls.py:877 (lazy), 929, 1835 | build_httpx_client | per call; pooled within a crawl | yes | agent thread / executor | per tool call; :929 per scraped result, even on robots-cache hit |
| Web_Scraping/WebSearch_APIs.py:2784, 3152, 3832 (+3201 bare post) | requests.Session | within one search | yes | worker | per web search |
| Utils/egress.py:1062 guarded_fetch_requests with session=None (callers: audio_processing:296, local_media_reading_service:4518, Article_Extractor 327/1144/1227) | requests.Session | NO | 30 s | worker | per URL fetch |
| Web_Scraping/Article_Scraper/crawler.py:201, 381 | aiohttp.ClientSession | within a crawl | per request | - | crawl |
| Media_Playback/stream_resolve.py:125, 150 | httpx.Client ×2 per resolve | NO | yes | to_thread | /stream-video |
| settings_image_gen_defaults.py:801 | httpx.Client | NO | yes | @work(thread) | user probe |
| Library/ingest_preflight.py:221 | urllib opener | NO | yes | @work(thread) | per preflight |
| Long-lived: tldw_api/client.py:1312, TTS/base_backends.py:121, audio_cpp.py:421/440, snapshot_client.py:88, github_api_client.py:179, word_bench capture_client.py:206, swarmui_client.py:142, Embeddings_Lib.py:530, confluence_auth.py:52, OCR_Backends.py:830 | mixed | **YES** | yes (audio_cpp read=None by design) | mixed | - |
| Cold one-shots: web_article_ingestion.py:83, transcription_service.py:3211, ollama_model_mgmt.py:110 (local), kokoro.py:194, diarizer_engine_onnx.py:441, acquisition.py:1391/2199, skill_remote_fetch.py:261, run_webhooks.py:397 (own thread) | mixed | per op | yes | threads | rare |

**Bottom line:**
- About 12 sites are long-lived and pooled.
- 13 are per-call sessions on the hosted-LLM path.
- About 16 build a client or transport on the Textual event loop.
- 11 are dead code.
- 0 have timeouts that can wedge a worker indefinitely.


# pat-recompose

## summary
Scope: a census of recompose, remount and over-mount patterns across all of tldw_chatbook (Third_Party excluded). It is AST-exact, so comment and docstring mentions are not counted. Every notable site was triaged by trigger and subtree size, and the worst were traced to a real caller. I ran bare-Textual calibration probes plus four targeted probes of real widgets and screens, all with isolated HOME/XDG/TLDW_CONFIG_PATH. Nothing was written to the audit tree (git status shows 0 changes).

Overall health. The big historical sources of recompose storms are handled or tracked. Library whole-screen recompose is TASK-281, held by a ratchet pinned at 63. The Console trays are equality- or structure-guarded, and the transcript is windowed and incremental. Watchlists search keystrokes are fixed (TASK-15460/15461), as is the Speech panel (card-scoped). The Console pickers are debounced and capped.

What remains is a second tier that recomposes on SELECTION or per VISIT rather than per character typed:
1. The Home screen whole-screen recomposes on every visit after its chatbook snapshot worker finishes. Measured: 68 ms CPU / ~150 ms wall, against 3 ms for the targeted `_sync_home_triage()` that already exists.
2. Watchlists rebuilds the Notifications pane on every cursor keystroke (measured 35 ms CPU / ~75 ms wall at 100 rows).
3. Watchlists rebuilds the Inspector (and the Content pane for items) on every Sources/Runs/Items/Notifications selection, including j/k article navigation (measured ~12 ms CPU for Inspector, ~42 ms for Content, ~90 ms wall combined).
4. Console character search recomposes its section, including the focused Input, twice per keystroke, with no debounce.

The structural gap is virtualization. No user-data list uses OptionList or Tree; every one mounts one Button or Static per row. Calibration: a compact Button costs 1.6–2.2 ms to mount on the harness (1000 rows = 2.0 s), while an OptionList of the same 200–1000 rows is flat at 81–89 ms. Unbounded or large lists that pay this:
- Workspace Files tree: full remount on every expand/collapse/page/filter, with up to 10,000 entries per directory and 500 filter hits.
- /rewind modal: one Button per user prompt in the session, with no cap.
- Evals rail: up to 500 rows × 3 lists, plus 4 sync DB queries re-run inside compose on every section toggle.

Cross-cutting pattern: 31 loops do serial `await container.mount(w)`. Measured at 1.35× / 1.94× / 3.42× slower than `mount_all` at 20 / 100 / 300 items. The worst is `console_turn_file_card`, which runs on every card mount (so on every Console visit) and can issue up to ~250 serial awaits for one expanded diff.

Dead code that still carries these patterns: ArtifactsScreen (route aliased to library, 8 whole-screen recompose sites), SkillsScreen (aliased), ActivityLogWidget (only its tests use it; a 60 s timer rebuilds up to 1000 rows), EnhancedStatusWidget and ChatMessageEnhanced (never constructed; the latter is still imported lazily on the TTS-complete path). Already-known items are cited in the census rather than re-filed: TASK-24452, 26834, 31506, 22660, 281/22888/21243/31509/31583, and 32810.10.

## clean areas
- Console trays (workspace_details, retrieval_scope_row, staged_context, staged_evidence_strip, settings_summary, right_rail ConsoleSelectedTurnActivity, dispatch_recovery, run_inspector, inspector_section, mcp_rail, home_rail/home_canvas): all equality- or structure-guarded, with in-place patch paths; recompose only on a real structural change
- ConsoleWorkspaceContextTray: conversation browser capped (75 results / 12 per group); recompose skipped via DOM-proof _can_skip_recompose; width relabel has hysteresis
- Console transcript: windowed (TASK-1365) and incremental reconciler (verified-fine list); ConsoleMessageHeader.sync_header patches in place, recomposing only when the speech-action slot appears or disappears
- Console picker modals (character, prompt, reaction, style, scope, session switcher): debounced and capped (e.g. CHARACTER_PICKER_MAX_RESULTS=40)
- Model install progress (llm_screen/_model_install_progressed -> CuratedView/RemoteView/InstalledView.apply_progress/set_install_state, FirstRunSetupWizard._install_progressed): targeted update_progress; recompose only in the NoMatches fallback
- Lab frame refresh_lab_status: in-place Static updates with an equality guard (2 s poll)
- Speech TTS settings panel: dropdown changes use card-scoped _replace_card_bodies; the remaining whole-panel recomposes are save/revert/restore gestures only
- Personas library pane: debounced search + paged ListView; personas transcript preview capped at 200
- Library conversation reader: keyed reconcile, 20-message pages
- Emoji picker grid: 0.3 s debounce, capped at 180; FirstRunSetupWizard model radio list capped at 20 (_PICKER_MODEL_LIMIT)
- Watchlists tree source rows: cached per widget instance (_source_cache); a region recompose re-queries only a fresh tree
- Settings screen: category switches are region-scoped since TASK-15475; _sync_overview_sync_widgets recomposes two tiny regions
- MCP servers mode: detail/toolbar/gate rebuilds fire on RowSelected (Enter/click), not cursor moves; about 9 gate rows
- Workflows editor: rebuild=False in-place path used for typing
- Library canvases/controllers: covered by open TASK-281/22888/21243/31509/31583 and the recompose ratchet (pin 63); not re-filed
- Chat/Console whole-screen recompose via wiring.refresh_screen and handle_console_unstage_evidence: fallback-only (runs only when targeted _sync_pending_launch_surfaces fails)

## census
### A. Pattern totals: tldw_chatbook/ (Third_Party excluded), AST-exact

Raw text `rg --count 'recompose=True'` finds 393 mentions in 95 files, comments included.

| Pattern | calls | files | in UI/ | in Widgets/ |
|---|---|---|---|---|
| `refresh(recompose=True)` | 191 | 63 | 143 | 48 |
| `reactive/var(..., recompose=True)` | 43 | 14 | 35 | 8 |
| direct `.recompose()` (incl. `super().recompose()`) | 38 | 24 | 19 | 19 |
| `remove_children()` | 123 | 67 | 57 | 64 |
| `.mount()` | 354 | 98 | 194 | 134 |
| `.mount_all()` | 36 | 22 | 20 | 16 |
| `ListItem()` | 37 | 20 | 6 | 29 |
| `Markdown()` widget or renderable | 17 | 16 | 7 | 8 |
| compose loops `for…: yield Widget` | 265 | – | – | – |
| per-item mount loops | 47 | – | – | – |
| ListView append loops | 10 | – | – | – |

Of the 47 per-item mount loops, **31 do serial `await mount`**.

### B. Per-file recompose leaders

Columns: rc = `refresh(recompose=True)` calls, rx = recompose reactives, dr = direct recompose.

| file | rc | rx | dr | trigger / subtree | verdict |
|---|---|---|---|---|---|
| UI/Screens/library_screen.py | 24 | 0 | 6 | whole screen | known TASK-281 (ratchet 63) |
| UI/Screens/model_installed_view.py | 20 | 0 | 0 | import, delete, activate gestures | P3; progress path is targeted |
| UI/Library_Modules/library_prompts_controller.py | 17 | 0 | 0 | whole Library screen, per gesture | known (core review D3) |
| UI/Wizards/FirstRunSetupWizard.py | 14 | 0 | 0 | speech step, per gesture | clean-ish |
| UI/Watchlists_Modules/artifacts_pane.py | 2 | 10 | 0 | region, per load | ok |
| Widgets/Settings_Widgets/speech_tts_settings_panel.py | 3 | 0 | 6 | post-save, revert | P3 |
| UI/Screens/artifacts_screen.py | 8 | 0 | 0 | whole screen, per visit | DEAD (route aliased to library) |
| UI/Screens/settings_screen.py | 2 | 0 | 4 | panes and regions | ok |
| UI/Watchlists_Modules/inspector_pane.py | 0 | 6 | 0 | every entity selection | **F3** |
| Widgets/Settings_Widgets/personal_context_panel.py | 0 | 6 | 0 | every record click (Selects included) | **F10** |
| model_curated_view / kept_briefings_modal / briefing_preset_modal / library_export_controller | 5 each | – | – | gestures | P3 / known |
| UI/Watchlists_Modules/watchlist_tree.py | 0 | 5 | 0 | expand or tag click | ok |
| Widgets/Library/library_notes_canvas.py | 4 | 0 | 1 | breakpoint flips only | ok |
| UI/Watchlists_Modules/notifications_pane.py | 0 | 2 | 0 | **every cursor key** | **F2** |
| UI/Watchlists_Modules/content_pane.py | 0 | 1 | 0 | every article open (j/k) | **F3** |
| items / runs / rules / sources panes | 0 | 1–3 | – | per page load or form toggle | ok |
| UI/Evals/library_rail.py | 1 | 0 | 0 | section toggle; DB in compose | **F7** |
| UI/Screens/home_screen.py | 1 | 0 | 0 | **every Home visit** | **F1** |
| Widgets/Console/console_character_context.py | 0 | 0 | 1 | **2× per keystroke** | **F4** |
| UI/Screens/workflows_screen.py | 0 | 0 | 1 | every visit (double compose) | F13 |
| Console widgets (workspace_context 2, project_instructions 2, transcript 3, run_inspector 1, inspector_section 1, staged_* 2, settings_summary 1, retrieval_scope_row 1) | – | – | – | guarded | ok; sync_preview is known TASK-32810.10 |

### C. remove_children + mount rebuild leaders

| file | remove_children | mount | mount_all | verdict |
|---|---|---|---|---|
| UI/Screens/backup_restore_screen.py | 8 | 11 | 0 | cold path; serial awaits |
| UI/MCP_Modules/mcp_servers_mode.py | 8 | 3 | 4 | per server Enter |
| UI/Workflows_Modules/editor.py | 4 | 14 | 0 | in-place path exists |
| Widgets/Console/console_transcript.py | 5 | 11 | 0 | incremental; verified fine |
| Widgets/Persona_Widgets/personas_character_editor_widget.py | 6 | 7 | 0 | per selection, small |
| UI/MCP_Modules/mcp_inspector.py | 5 | 2 | 6 | per selection, small |
| Widgets/Console/console_prompts_modal.py | 4 | 11 | 0 | mode switch |
| Widgets/Console/console_workspace_files_modal.py | 2 | 7 | 2 | **F5**, whole tree per click |
| UI/Screens/stats_screen.py | 1 | 18 | 0 | capped |

### D. User-data lists that mount one widget per row

| list | site | bound | trigger | verdict |
|---|---|---|---|---|
| Workspace Files tree and filter | console_workspace_files_modal.py:774 | 10,000 per directory; 500 filter hits | every expand, collapse, page, filter | **F5** |
| /rewind prompts | console_rewind_modal.py:178 | unbounded | /rewind | **F6** |
| Evals rail datasets, tasks, runs | Evals/library_rail.py:751 | 500 each | section toggle, mutation | **F7** |
| Watchlists notifications | notifications_pane.py:88 | 100 | per cursor key | **F2** |
| Console turn-file rows and hunks | console_turn_file_card.py:447/688 | unbounded files; 50 hunks | every card mount | **F8** |
| Settings workspaces | settings_screen.py:20557 | unbounded, with N+1 DB and stat | category render | **F9** |
| Personal-context records | personal_context_panel.py:414 | unbounded | every record click | F10 |
| Console character rows | console_character_context.py:394/446 | 4×5 or 8 | 2× per keystroke | **F4** |
| Chatbooks cards | Chatbooks_Window_Improved.py:565 | unbounded zips | debounced search | P3 |
| Audiobook conversation and note pickers | conversation_selection_dialog.py:220 | 100 conversations × 6 widgets | modal open | P3 |
| Console conversation browser | console_workspace_context.py:1848 | 75 | state change | capped, ok |
| Personas library | personas_library_pane.py:444 | paged, debounced | – | ok |
| Personas transcript preview | – | 200 | – | ok |
| Library conversation reader | – | page 20 | – | ok |
| Emoji grid | – | 180 | debounced | ok |
| Wizard model list | – | 20 | – | ok |
| Session tabs | – | – | – | known TASK-26834 |

### E. Calibration

Probe harness: bare Textual 8.2.8 App at 211×50. Each number covers remove, mount and settle.

- **Compact Button:** 50 rows 101 ms, 200 rows 318 ms, 500 rows 1092 ms, 1000 rows 1998 ms. That is 1.6–2.2 ms per row.
- **Static:** 0.6–1.2 ms per row.
- **OptionList with the same rows:** 55–89 ms, roughly flat from 200 to 1000 rows.
- **Serial `await mount` vs one `mount_all`:**
  - 20 items: 79 vs 59 ms
  - 100 items: 263 vs 136 ms
  - 300 items: 1160 vs 339 ms


# pat-startup-imports

## summary
Boot / first-paint import graph audit of origin/dev 840ed2ca58. Measured in an isolated scratch profile (HOME, XDG_*, TLDW_CONFIG_PATH and TLDW_TEST_MODE all pointed at scratch, keyring set to the null backend, PYTHONPYCACHEPREFIX in scratch). Two caveats on the numbers. First, the machine had a load average of 15 to 48 on 18 cores because of other agents, so wall times run about 1.5 to 2x high; I prefer CPU time and syscall counts. Second, the scratch path is 13 components deep versus about 5 for a real ~/.config, so the admission open() counts run about 2.5x high.

Headline: the import graph is not the biggest boot cost. The Backup_Recovery storage-admission layer is. Admission syscalls dominate TldwCli() construction, mount and idle: 82k open() calls on the main thread in the constructor, 70k on the event loop during mount, and about 4.9k per second on the main thread at idle. Every loop stall over 100 ms in the first 5 s after _ui_ready was sampled inside bootstrap.pinned_directory (the worst was 1,361 ms). Warm load_settings() still costs 647 opens (62 ms) and get_user_data_dir() 1,723 opens (214 ms) per call. TASK-32804.1 fixed only get_cli_setting.

Phases: `import tldw_chatbook.app` costs 1.3 to 1.45 s CPU, loads 681 tldw / 1,772 total modules, and runs module-scope I/O (admit_startup, two config loads, and a `file` subprocess spawned by croniter). The Chat first-paint leg adds about 0.44 s CPU and 279 tldw modules; the Widgets.Console package __init__ accounts for most of it. Every private SQLite connection spawns a Python helper process: 11 on the main thread before first paint, 5 of them on the event loop. Boot GC runs 4 gen2 collections during mount (517 ms, max 179 ms), and nothing in the tree calls gc.freeze or tunes thresholds.

Clean, deferrable import wins, each measured:
- tldw_profile_core pulled in for one constant (~23 ms in-app).
- The TTS/STTS stack is forced eager by @on Message classes and a tts_service built in __init__ (43 modules, ~97 ms).
- Readable-hue pinning for all 94 themes at import (34 ms).
- `requests` pulled in by 7 boot modules (99 modules, ~25 to 34 ms).
- The Console gateway builds its httpx client and SSL context eagerly at mount (40 to 45 ms). The TLS factory rebuilds an SSLContext for every client (~12 ms each).
- croniter spawns `file` at import (13 to 19 ms).
- Cryptodome with cffi/pycparser (16 to 27 ms).
- The Git half of File Notes is built in __init__ (~30 ms).
- Modal screens eagerly imported by the Widgets.Console __init__.

Structural:
- app.py is a god module: about 150 module-scope feature imports, because __init__ wires every service. The lazy meeting_session_owner property is the in-tree precedent for the fix.
- `import tldw_chatbook.config` alone costs about 400 ms (85 modules). Any Chat.* import costs 0.66 to 0.9 s through the eager Chat/__init__.
- Launching with `python -m tldw_chatbook.app` makes every spawn child re-import the full app. tldw-serve launches that way, so each raw CLI command pays it.

Ratchets: boot-import 681/686 (headroom 5); ui-ready 1,031/1,033 (headroom 2). The pre-import payload ratchet is red on dev (554/500 modules, 409,566/378,740 LOC) and perf-guard.yml does not run it.

Disclosure: my first 5 importtime runs, before I set PYTHONPYCACHEPREFIX, wrote gitignored __pycache__/*.pyc under the audit tree. `git status` there is still clean, and all later runs used the scratch cache prefix.

## clean areas
- tldw_chatbook.MCP.server import path: 192-393 ms, 37 tldw modules (lean)
- Spawn worker target modules are kept lean: Tools/raw_cli_executor (137 ms, 23 mods, no config), Local_Ingestion/ingest_parse_worker (13 ms, 6 mods), STT/executor (35 ms, 12 mods)
- Utils/optional_deps.embeddings_rag_deps_installed (chat_screen right-rail readiness) is find_spec-only; numpy is NOT imported at first paint (blame hook false positive checked)
- HEAVY_MODULES guard holds: torch/transformers/nltk/scipy/sklearn/pandas/docling absent at import and at _ui_ready
- Screen registry is lazy; UI.Screens.* other than chat are absent at _ui_ready with TLDW_SCREEN_PREIMPORT=0
- get_cli_setting warm path is fast (0.002 ms/call, 0 opens) after TASK-32804.1
- Chat.provider_setup_persistence's 27-30 ms importtime self is GC noise (6 ms isolated); not a finding
- Qualification evidence re-parse (0.021 ms) and ctypes CDLL per call (0.015 ms) are micro costs; only the directory walks matter
- textual.widgets.Markdown / markdown_it are needed by console_transcript at first paint anyway; app.py's own import of them is not a deferral target
- RAG_Search.simplified / Chunking / Personal_Context stay off the first-paint window (census ABSENT lists hold)
- pydantic plugin entry-point scan is a one-time ~8 ms; not worth disabling
- regex compiles at import: 455 cache misses spread thinly (largest mdurl 10.7 ms third-party, log_sanitizer 3.1 ms); hygiene only
- Local_Ingestion lazy __init__, Skills_Interop/Sync_Interop/TTS PEP 562 facades exist (but app.py resolves their names eagerly at module scope)

## census
### A. Top 40 by cumulative time: `python -X importtime -c "import tldw_chatbook.app"`
Median of 10 warm runs (it_11..it_20). importtime inflates times and includes GC; plain warm import is 1.3-2.4 s wall and 1.29-1.46 s CPU.

| # | module (tc = tldw_chatbook) | cum ms | self ms | first importer |
|---|---|---|---|---|
| 1 | tc.app | 2517.8 | 189.6 | - |
| 2 | tc.Chat (package init) | 577.2 | 0.3 | tc.Chat.chat_conversation_scope_service |
| 3 | tc.Chat.server_chat_conversation_service | 521.0 | 0.3 | tc.Chat |
| 4 | tc.runtime_policy.bootstrap | 520.6 | 1.1 | tc.Chat.server_chat_conversation_service |
| 5 | tc.Utils.tls_trust | 519.0 | 0.3 | tc.runtime_policy.bootstrap |
| 6 | tc.config | 415.2 | 193.1 (112 without GC) | tc.Utils.tls_trust |
| 7 | tc.Chat.chat_conversation_scope_service | 253.6 | 0.2 | tc.app |
| 8 | tc.Chat.console_runtime | 137.7 | 5.0 | tc.app |
| 9 | tc.Backup_Recovery.storage_admission | 133.4 | 1.0 | tc.app (line 3) |
| 10 | tc.Home | 133.3 | 0.2 | tc.Home.active_work_adapter |
| 11 | textual.widget | 120.9 | 1.8 | tc.app |
| 12 | textual | 98.0 | 0.4 | textual.widget |
| 13 | textual._on | 95.0 | 0.2 | textual |
| 14 | textual.css.model | 93.9 | 1.3 | textual._on |
| 15 | tc.Backup_Recovery.admission | 91.4 | 5.4 | tc.Backup_Recovery.storage_admission |
| 16 | tc.DB.Client_Media_DB_v2 | 84.5 | 2.1 | tc.config (config.py:70) |
| 17 | tc.Home.active_work_adapter | 80.5 | 0.8 | tc.app |
| 18 | tc.STT.persistence | 73.7 | 1.0 | tc.DB.Client_Media_DB_v2 |
| 19 | textual.css._help_renderables | 63.2 | 0.2 | textual.css.model |
| 20 | rich.console | 63.0 | 4.5 | textual.css._help_renderables |
| 21 | tc.Notes.file_notes_git_service | 62.8 | 16.5 | tc.app:420 |
| 22 | tc.Chat.citation_artifact_ownership | 59.8 | 4.6 | tc.app |
| 23 | requests | 54.3 | 0.5 | tc.Utils.tls_trust:27 |
| 24 | tc.Skills_Interop.skill_trust_service | 50.4 | 1.1 | tc.app:632 (facade) |
| 25 | tc.Home.dashboard_state | 49.9 | 6.3 | tc.Home |
| 26 | tc.Skills_Interop.skill_trust_crypto | 47.9 | 0.7 | skill_trust_service |
| 27 | tc.Chat.console_settings_defaults | 47.8 | 4.6 | tc.app |
| 28 | textual.app | 47.0 | 3.8 | tc.app |
| 29 | tc.Chat.console_display_state | 46.0 | 8.8 | tc.Chat.console_runtime |
| 30 | tc.Kanban_Interop.server_kanban_service | 44.5 | 43.9 | tc.app:574 (known TASK-21107/21239) |
| 31 | tc.Library.library_local_rag_search_service | 43.4 | 0.6 | tc.app:230 |
| 32 | tc.css.Themes.themes | 42.4 | 42.3 | tc.app:187 |
| 33 | tc.Sync_Interop.sync_readiness | 42.1 | 1.1 | tc.app:647 |
| 34 | tldw_profile_core | 40.6 | 0.4 | tc.Sync_Interop.sync_readiness:8 |
| 35 | tc.Library.library_rag_service | 40.5 | 0.9 | library_local_rag_search_service |
| 36 | tc.Backup_Recovery.journal | 39.2 | 30.5 | tc.config (installation_client_id) |
| 37 | tc.Library.library_rag_state | 38.5 | 4.4 | tc.Library.library_rag_service |
| 38 | httpx | 37.0 | 0.4 | tc.Utils.tls_trust:26 |
| 39 | markdown_it | 36.8 | 0.2 | tc.app:101 (textual Markdown) |
| 40 | markdown_it.main | 36.5 | 0.5 | markdown_it |

### B. Boot phases
Headless run_test at 211x44, warm profile, `TLDW_SCREEN_PREIMPORT=0`. Syscalls counted with an audit hook.

| phase | tldw mods (total) | wall | main-thread open() | loop/thread listdir | helper Popen (main/thr) | notes |
|---|---|---|---|---|---|---|
| import | 681 (1,772) | 1.3-2.4 s (CPU 1.29-1.46 s) | 4,033 non-import (+1,827 .pyc) | 348 | 0 (+1 `file` from croniter) | 61 ctypes dlopen; 3 gen2 GC = 71.5 ms |
| TldwCli() | 711 | 5.7-6.1 s (cProfile CPU 4.4 s) | 82,295 (+10,121 thr) | 2,721 | 6 / 3 | `_wire_watchlists_and_notifications_services` alone 1.95 s (cProfile) |
| mount → _ui_ready | 1,031 (2,285) | 7.6-8.2 s | 69,790 ON LOOP (+15,895 thr) | 2,319 | 5 on loop / 14 | 4 gen2 = 517 ms, max 178.6 ms |
| idle, 10 s window (8 s after ready) | - | - | 48,791 (4.9k/s) | 1,585 | 0 / 13 | ~7 load_settings/s from readiness paths |
| first 12 s after ready | - | - | - | - | - | 9 loop stalls >100 ms (732, 431, 116, 197, 342, 1361, 175, 496, 437 ms), all sampled in admission walks |

### C. Warm per-call cost of config reads (50 calls each)

| call | ms/call | opens/call | listdir/call |
|---|---|---|---|
| get_cli_setting | 0.002 | 0 | 0 |
| load_settings (cache hit) | 61.7 | 647 | 21 |
| get_user_data_dir | 214.5 | 1,723 | 51 |

get_user_data_dir is called 59 times per boot, and there are 275 config_participants.wrapped calls per boot.

### D. Third-party marginal import cost
Measured after a baseline of textual, pydantic, httpx, rich and loguru; median of 5 runs.

| package | ms | leg / importer |
|---|---|---|
| tldw_profile_core | 70.0 standalone (23.3 self-sum in app) | import / Sync_Interop.sync_readiness:8 |
| requests (+chardet 43 mods, charset_normalizer, urllib3) | 25.4 | import / tls_trust + 6 other modules |
| httpcore + anyio + h11 | 21.7 (first AsyncClient 39-44 ms) | mount / console_provider_gateway:2750 |
| Cryptodome.Cipher.AES (+cffi, pycparser) | 16.5 | import / skill_trust_crypto:13, Sync_Interop/crypto:11 |
| croniter (+dateutil; spawns `file -b python`) | 13.0 | import / scheduled_tasks_db:19 |
| PIL.Image | 8.3 | mount / console_transcript:15, console_image_view:22 |
| cryptography aead | 8.0 | import / Subscriptions/security:27 |
| chardet | 7.0 | via requests |
| keyring (+jaraco) | 6.7 | ctor / Media_Generation/config_machinery:26 (TASK-21109) |
| yaml | 6.7 | import / Client_Media_DB_v2:47 (+5 more) |
| prometheus_client | 6.6 | import / Metrics/metrics:36 (try/except, eager) |
| emoji | 6.1 (emoji_picker module 9.2) | ctor / Backup_Recovery/raw_participants:239 |
| regex | 4.1 | import / input_validation:16 |
| psutil | 3.7 | import / metrics_logger:9 |
| tokenizers | 3.7 | import / custom_tokenizers:26 |
| aiofiles | 1.8 | mount / Media_Creation/__init__ via style picker |

### E. Import-phase self-time by family
importtime with GC disabled, median of 4 runs; all 1,680 modules sum to 1,508 ms.

| family | modules | ms |
|---|---|---|
| TTS/STTS | 43 | 96.8 |
| textual | 139 | 77.2 |
| Backup_Recovery | 32 | 71.4 |
| Chat citations | 11 | 52.6 |
| Scheduling + croniter + dateutil | 36 | 38.8 |
| Notes file-notes git | 5 | 36.4 |
| requests family | 99 | 33.8 |
| Kanban (known) | 3 | 32.1 |
| pydantic | 42 | 31.2 |
| Cryptodome + cffi + pycparser | 32 | 27.0 |
| Library | 23 | 26.5 |
| tldw_profile_core | 9 | 23.3 |
| prometheus + psutil | 29 | 14.0 |

### F. Objects built at import time

| construct | count | cost |
|---|---|---|
| dataclasses | 1,041 | 220-380 µs each ≈ 0.25-0.4 s CPU |
| pydantic models | 293 | journal 41, kanban_schemas 76, input_validation 27, profile_core ~27 |
| regex cache misses | 455 | - |
| enums | 287 | - |
| gc.freeze / set_threshold / gc.disable in tldw_chatbook | 0 files | - |

### G. Package `__init__` files on the ready path
104 packages in the ready set; 29 have a PEP 562 `__getattr__`.

| package | eager imports | notes |
|---|---|---|
| Widgets.Console | 34 | ~9 of them modals |
| Internal_Prompts | 10 | known resident, cheap |
| Actor_Packs | 8 | |
| STT | 6 | |
| Chat | 6 | includes dead ServerChatLoopService |
| Media_Creation | 3 | swarmui client and image service for one template import |

### H. Non-app entry imports

| import | cost | modules |
|---|---|---|
| tldw_chatbook.config | 396-451 ms | 85 tldw / 446 total |
| tldw_chatbook.Chat.console_chat_models | 664-898 ms | 796 total |
| tldw_chatbook.UI.Screens.chat_screen, marginal after app | 432-495 ms CPU | +279 tldw / +347 total |

### I. Ratchets (run via pytest on this tree)

| guard | measured / limit | status |
|---|---|---|
| boot-import-weight | 681 / 686 | headroom 5; snapshot drift +16/-4 |
| ui-ready-census | 1031 / 1033 | headroom 2 |
| MAX_MODULE_COUNT | 1,772 / 2,200 | pass |
| preimport payload modules | 554 / 500 | RED (+72 new modules vs snapshot) |
| preimport payload LOC | 409,566 / 378,740 | RED |
| per-route LOC: library | 125,111 / 123,319 | RED |

The preimport budget test is not run by perf-guard.yml. With pre-import on, the pass adds 680 modules post-ready and p99 loop lag rises to 28-122 ms, against 14-20 ms with it off.


# pat-timers

## summary
Whole-tree census of periodic activity in tldw_chatbook at 840ed2ca58. I used untruncated AST scans plus grep, and read every set_interval site. The census found: 62 real set_interval call sites (71 textual) in 52 files, plus realtime.py's `_set_interval` alias and two factories in wiring.py; 119 set_timer calls (126 textual); 739 call_later/call_after_refresh calls with no self-re-arm; 239 while-loops that sleep or wait; 19 queue.get(timeout)/select loops; about 84 threading.Thread constructions; no watchdog/inotify library. Most set_timer calls are debounces. The self-rescheduling set_timer chains are all bounded.

Idle cost on the default screen (Console visible, no optional features) is roughly 45 wakeups/s. Contributors: UI heartbeat 1 Hz; ui-stall watchdog and drain threads at 10 Hz each; canvas-policy watcher at 4 Hz with a to_thread hop; legacy trace maintenance at 1 Hz (known TASK-31501); credential poll at 4 Hz; the left-rail and character-context progress polls at 2 Hz each; and a 2 Hz nav-bar tick on each of the 3 retained screens. The CPU cost is dominated by the 4 Hz credential/readiness poll (about 8 ms per tick, TASK-32804.3). That pattern is duplicated on 4 other surfaces because the subscription cache publishes no expiry signal.

The worst new costs are data-scaled or feature-scoped:
(1) The Library Folder-files workspace re-walks the whole notes folder every 1.5 s with pathlib. Measured 41–49 ms per poll at 500 files and 175–500 ms at 5,000 files, against 27 ms for a scandir walk. It also pays about 25 ms of fixed backup-admission overhead (about 1,200 open() calls) per reconcile.
(2) The Console 0.2 s run tick snapshots the whole transcript twice per tick; the recovery reverse-scan walks to the start in the common no-recovery case. Measured 2.2 ms per 400-message snapshot, so about 2% of a core during any run at 400 messages.
(3) Each open terminal session runs a 200 Hz runtime poll and a 100 Hz input-flush poll, on top of the known 50 Hz monitor.
(4) The auto-wake cancel probe does BEGIN IMMEDIATE write transactions at up to 20 Hz.
(5) MCP workbench, voice TTS bridge (1 kHz on the UI loop) and Buddy polling are feature-scoped taxes.

Structural notes:
- Hidden-screen timers: Textual 8 does not pause widget timers on screen suspend. Reusable Console/Library/Home screens therefore keep every widget timer alive while hidden. Some are gated on is_current/is_active; left_rail and character_context are not.
- Good in-tree templates exist and should be reused: PollingNotesSyncWatcher backoff (1→10 s), pausable_progress clock pausing, the change_review_finalization/RAG-indexer blocking-get-plus-sentinel workers, persona_buddy_widget per-frame one-shot timers, and backup_restore_screen's suspend/resume pause.
- There are 38 copy-pasted maintenance_drain implementations (27 sleep-poll).

Measured probes ran only under the scratch-isolated HOME/XDG/TLDW_CONFIG_PATH. The real config mtime was confirmed unchanged.

## clean areas
- Logging_Config.py:113 RichLog queue processor: event-driven await queue.get(), sleep(1) only on error
- Subscriptions scheduler thread=True periodic sync (verified-fine template)
- Scheduling/scheduler/loop.py:446 30 s tick; heartbeat write offloaded (TASK-31507); queue reload every 60 ticks
- Notes/notes_sync_watcher.py:76-134 adaptive backoff 1->10 s with interruptible sleep (template)
- Utils/db_status_manager.py:172 120 s DB-size stat via asyncio.to_thread, change-gated log
- app.py:19014/19049 24 h media cleanup and change-review retention, offloaded
- app.py:16589 boot worker reconcile 2 s, self-stops when gate drains
- app.py:6681 remote ingest poll 5 s, needs-scoped and self-exiting
- app.py:3989/4063 ingest progress drain and pool monitor (blocking connection.wait; idle pool retirement at app.py:5575)
- RAG_Search/ingestion_indexing.py:1131 indexer thread: blocking get + _STOP sentinel
- Workspaces/change_review_finalization.py:714 fs workers: blocking get + sentinel (template)
- UI/Navigation/main_navigation.py:540 overflow tick gated on screen.is_active + geometry signature
- Widgets/pausable_progress.py ProgressBar/LoadingIndicator clocks paused while hidden, with architecture guard (all 8 LoadingIndicator sites use it)
- Widgets/Console/console_background_effect.py:178 opt-in, gated on screen.is_active
- Widgets/Console/console_setup_modal.py ConsoleSetupBackdrop now static (08-22 9.9% fix held)
- UI/LLM_Management_Window.py:617 Ollama probe gated on active screen, worker-owned, exclusive
- UI/Screens/backup_restore_screen.py:504 poll paused on suspend
- Widgets/Console/console_composer_bar.py:3713 cursor blink paused unless focused
- Console cost-TTL / transcript-sync / fleet-survivor timers stop on suspend and self-stop (chat_screen.py:23654-23658)
- Widgets/Console/console_prompt_queue_modal.py:188 revision-gated
- Chat_Widgets approval/question card deadline 1 Hz, scoped to pending card
- Meetings screen 0.2 s tick early-returns without a session; meeting watchdog scoped to live meeting
- Console hands-free/realtime/dictation 10 Hz ticks torn down in on_screen_suspend
- Workspace change review poll (workspace_change_review.py:103) gated on current + PREPARING state, reads off-loop
- set_timer: 119 sites, almost all debounces/one-shots; self-rescheduling chains bounded (chat_screen.py:16864, model_installed_view.py:826/982, console_session_surface.py:797, library_screen.py:10597 deadline-bounded)
- call_later/call_after_refresh: 739 sites, no self-re-arming chains
- No watchdog/inotify file-watcher library; threading.Timer only used as one-shot watchdogs (4 sites)
- Web_Server/serve.py:938 websocket expiry sleeps until expiry (good pattern)

## census
**Method.** AST scans cover set_interval, set_timer, call_later/call_after_refresh (with self-re-arm detection), every `while` loop containing sleep/wait/wait_for, and `.get(timeout=)`/select loops. Every set_interval site and every long-lived loop was then read by hand.

**Counts (untruncated):**
- set_interval: 62 real calls (71 textual) in 52 files. Also `self._set_interval` at realtime.py:346, and the factories at wiring.py:1390/1788.
- set_timer: 119 calls (126 textual) in 58 files.
- call_later + call_after_refresh: 739 calls.
- asyncio.sleep: 199 in 101 files. time.sleep: 67 in 47 files. `.wait(`: 280 in 120 files.
- while-loops with sleep/wait: 239. `.get(timeout)`/select loops: 19.
- threading.Thread: 84 in 49 files. loop.call_later: 2. File-watcher libraries: 0.

### A. App-level, always on once started

| Site | Cadence | Per tick | Idle / hidden / unused | Stops | Where it runs |
|---|---|---|---|---|---|
| app.py:11943 UI heartbeat | 1 Hz | drift calc | always | quit | loop |
| Utils/ui_responsiveness.py:311 ui-stall-watchdog | 10 Hz (time.sleep 0.1) | reads a timestamp | always | close() | thread |
| Utils/ui_responsiveness.py:123 ui-stall-persist | 10 Hz (get timeout 0.1) | none when empty | forever after the first diagnostic | close() | thread |
| Chat/console_runtime.py:1115→3033 canvas policy watcher | 4 Hz + to_thread hop | 5 µs config read (measured) | always, for every Canvas-enabled user (the default) | only when Canvas is disabled | loop + pool |
| Chat/console_runtime.py:3288 legacy trace maintenance | 1 Hz | write-lock txn | always | dispose | loop (**TASK-31501**) |
| Utils/db_status_manager.py:172 | 1/120 Hz | ~15 stats | always | quit | loop → thread |
| app.py:19014/19049 media cleanup, retention | 1/24 h | DB cleanup | always | quit | loop → thread |
| app.py:16589 boot reconcile | 0.5 Hz | slot check | startup only | self-stops | loop |
| Scheduling/scheduler/loop.py:446 | 1/30 Hz | due scan + heartbeat file write | always | stop() | loop (write off-loop) |
| Notes/notes_sync_watcher.py:127 | 1→10 s backoff | discover walk per root | when sync is configured | stop() | loop → thread |
| UI/Navigation/buddy_management.py:933 | 2 Hz | sessions tuple, set_scope | once Buddy is configured | **never** | loop |
| Workspaces/change_review_consent.py:387 | 2 threads × 20 Hz | none when empty | forever after first use | dispose | threads |
| Workspaces/change_review_finalization.py:773 publisher | ~40 Hz (get 0.025) | none when empty | forever after first use | stop | thread |

### B. Console screen (reusable; kept suspended when hidden)

| Site | Cadence | Per tick | Hidden | Stops on suspend? |
|---|---|---|---|---|
| chat_screen.py:16947 credential poll | 4 Hz | readiness rebuild, ~8 ms (**TASK-32804.3**) | gated (is_current) | no |
| chat_screen.py:16943 environment poll | 0.1 Hz | no-op while the rail is closed | gated | no |
| chat_screen.py:18947 transcript sync | 5 Hz while any run is in flight | whole-UI sync + 2 full transcript snapshots | n/a | yes |
| chat_screen.py:11969 cost TTL | 0.1 Hz while WARM | chip repaint | n/a | yes |
| left_rail.py:627 progress | 2 Hz | query_one + progress state | **not gated** | no |
| console_character_context.py:131 | 2 Hz | O(buttons×sessions) count rebuild | **not gated** | no |
| main_navigation.py:540 (one per retained screen, 3) | 2 Hz each | signature check | gated | no |
| chat_screen.py:23890 watchlist-operation follower | 0.5 Hz while an operation is active | to_thread JSON round-trip per operation | runs while hidden | no |
| console_assistant_turn.py:237 / console_transcript.py:1544 | 10 Hz per running tool/raw-CLI row | layout=False label update | no gate | when the row settles |
| character_expression_avatar.py:142 | 30 Hz (animated expression) | hit-test + frame_at | visibility-gated | on finish or unmount |

### C. Other screens and widgets (scoped)

- **Library file notes**, library_file_notes_workspace.py:2376: 1.5 s. Runs a full folder walk plus replica fetches. Gated by TASK-22219. See F1.
- **MCP workbench**, mcp_workbench.py:986: 4 Hz. Unconditional Static.update plus a stat. Not gated. See F6.
- **Readiness polls at 4 Hz**, all gated on current: settings_screen.py:4180, personas_screen.py:1949, FirstRunSetupWizard.py:1559, console_settings_modal.py:2637.
- **Schedules**, schedules_workbench.py:852/858: every 60 s and every 5 s. The 5 s tick reads the heartbeat file on the loop.
- **Short modal polls:**
  - skills modal (skills_screen.py:241): 5 Hz
  - notes recovery dialog: 4 Hz
  - agent progress modal: 2 Hz
  - Buddy conversation modal: 5 Hz; Buddy workspace modal: 1 Hz; Buddy speech controls: 2 Hz
  - session switcher: 5 Hz (**TASK-31506**)
- **Buddy widget**, persona_buddy_widget.py:284: 10 Hz snapshot poll, gated on the current view.
- **Research_Window.py:814**: 0.5 Hz. Re-renders detail unconditionally while a run is live.
- **Lab**, llm_screen.py:5008: 0.5 Hz, in-memory.
- **Video player / preview** (0.25 s / 0.5 s), **speech playground** (0.2 Hz), **audio troubleshooting** (10 Hz), **status dashboard / snapshot manager** (1 Hz), **splash**, **Tamagotchi** (unused at boot), **fspicker listing**: all scoped.

### D. Long-lived loops and threads (feature-scoped)

| Site | Cadence | Notes |
|---|---|---|
| Terminal/session_manager.py:1337 `_run_runtime` | **200 Hz** (wait 0.005) | per open terminal session; non-blocking read poll |
| Terminal/posix_backend.py:975 `_flush_pending_input` | **100 Hz** (wait 0.01) | per session |
| Terminal/posix_backend.py:968 monitor | 50 Hz | **TASK-31503** |
| Agents/agent_service.py:3980 / 6402 fleet settle and wait | 20 Hz | calls automatic_work.should_cancel → BEGIN IMMEDIATE |
| Chat/console_fleet_wake.py:680 | 4 Hz | run_owned_db_call → new connection per call |
| Chat/console_voice_tts_bridge.py:259/344/418/430 | **1 kHz** | voice TTS, on the UI loop |
| Audio/voice_process_entry.py:1147/1155; native_duplex_stream.py:267/276 | 20 / 100 / 1000 Hz | voice child spin-waits |
| Chat/console_voice_worker.py:239; console_voice_process.py:967 | 20 Hz each | voice session |
| Audio/streaming_sink.py:1342/1131 | 100 Hz / 20 Hz | TTS drain and backpressure |
| UI/Navigation/buddy_speech.py:76 | 1 Hz | full transcript snapshot per tick |
| UI/Screens/watchlists_collections_screen.py:10882 | 10 Hz | DB poll while a briefing generates |
| Subscriptions/feed_server.py:601 | 20 Hz | serve_forever(poll_interval=0.05), opt-in |
| Research_Interop/local_research_engine.py:386; Library/collections_capture_service.py:372 | 1/30 Hz | lease heartbeats while a run is active |
| 38× `maintenance_drain` (27 poll at 10–20 ms) | backup windows only | duplicated implementation |
| ~60 other bounded waits (e.g. image adapters 1–2 s, git/process reapers 10 ms, lock acquires 10–50 ms) | short-lived | bounded by deadline |

**Measured wake costs (bare Event.wait loop, this machine):**

| Timeout | Wakes/s | Core used |
|---|---|---|
| 5 ms | 162 | 0.46% |
| 10 ms | 82 | 0.24% |
| 20 ms | 42 | 0.13% |
| 100 ms | 9 | 0.04% |


# slice-1

## summary
Slice #1 "Agents#1" (35 files, 30,439 lines; dominated by agent_service.py 8.8k, local_tool_provider.py 5.2k, agent_runtime.py 3.2k). Hot paths identified: (a) every Console send runs AgentService.run_turn -> _run_one on the agent worker thread (bridge run_reply), with the model call submitted to a per-turn lifeline loop, not the Textual loop; (b) per model call (_make_call_model/call_model plus the bridge's async_worker_guard); (c) per tool call (_call_with_timeout spawns a daemon thread wrapped in worker_guard, then provider invoke); (d) per send on the MAIN loop: MCPToolProvider.compose_catalog and LocalToolProvider construction, both called from the controller's _compose_agent_request_providers; (e) the Chat first-paint import leg, where 24 Agents modules load.

Overall: the pure loop logic (runtime, stream gate, history projection, native tools, fleet coordinator) is cheap. The cost sits in the security and durability seams that the agent layer calls over and over. The biggest one:
- RunLogWriter.bind runs on every send and calls is_within twice without a pre-resolved context. Each call re-resolves the deliberately uncached sensitive-path set: 13 DB-path accessors, each calling get_user_data_dir with a file lock and storage admission.
- Measured in an isolated profile: run_turn with an instant fake model took 2.26 s with the run log on and 0.24 s with it off. is_within costs 969 ms without a context and 0.39 ms with one.
- The same uncached resolution recurs in LocalToolProvider's fs ledger helpers: fs_read measured 2.1-3.9 s end to end.

Second, recovery/storage admission (~1,200 os.open per acquisition) is paid again at every guard nesting level, per model call, per tool call (a fresh thread reacquires every time), per AgentRunsDB statement, and per permission-store read. Four or more of those reads happen on the event loop per send during provider composition.

Smaller issues:
- An O(N^2) schema-fit probe on every first request.
- The MCP invoke lock is held across human approval waits and network I/O.
- agent_service/agent_runtime sit on the Chat first-paint leg only for a 3-line helper, two constants and a pure parser.
- Minor quadratic stream buffering and 20 Hz sleep-polling.

Caveat on magnitudes: the scratch profile paths are about twice as deep (15 components) as a default install (~6), and the admission cost scales with path components. Absolute numbers are therefore upper bounds; realistic values are roughly 40-60% of measured.

Out-of-slice note for the parent to dedupe:
- Chat/console_trace_custom_pii.py showed about 60 ms of self import time on the chat leg.
- Agents/run_log.py appends cost about 18 ms per record (two acquire_storage calls each). Both belong to other slices.

## clean areas
- tldw_chatbook/Agents/agent_models.py: pure dataclasses and constants. Its ~8-10 ms import is dataclass processing (19 classes) and matters only through the first-paint finding F7.
- tldw_chatbook/Agents/native_tools.py: pure, with no config or I/O.
- tldw_chatbook/Agents/history_projection.py and canvas_tool_provider.build_canvas_runtime_guidance: per-model-call history projection measured at 0.06 ms (native) to 0.43 ms (fence, ~1 MB history). Fine.
- tldw_chatbook/Agents/agent_runtime.py loop body: _truncate_tool_result, _append_tool_result and _detect_cycle are bounded. No per-step deepcopy or whole-transcript serialization. _tool_protocol_cache repr key costs 0.23 ms and render_tool_protocol 2 ms for 80 schemas.
- tldw_chatbook/Agents/fleet_coordinator.py: handle copies are small. Retention is capped at 5 transcripts x 200k chars and pruned identities at 256. The event list is pruned by prune_terminal on every coordinator fetch.
- tldw_chatbook/Agents/fleet_messages.py and fleet_message_tools.py: in-memory, bounded.
- tldw_chatbook/Agents/execution_capacity.py, automatic_work_budget.py, automatic_work_runtime.py, human_input_wait.py: lock-guarded in-memory bookkeeping. Config reads go through the TASK-32804.1 fastpath.
- tldw_chatbook/Agents/approval_provenance.py, persona_policy.py, ask_user_questions.py, bulk_reader_corpus.py, agent_presets.py, model_retry.py, fallback_chain.py, fs_read_ledger.py (per-run bounded LRU): pure or cold.
- tldw_chatbook/Agents/local_tool_provider.py construction and list_catalog: measured 0.28 ms and 0.017 ms. _default_specs' three get_cli_setting reads are cheap after the fastpath. It is only the invoke-time ledger helpers (F2) that are expensive.
- tldw_chatbook/Agents/builtin_tool_gate.py enumerators (all_tool_gates, _off_tool_gate_status, tool_gate_breadcrumb): already fixed to one config load by TASK-32804.7 (Done). Lazy import of local_tool_provider costs ~10 ms once.
- tldw_chatbook/Agents/agent_routing.py: config read once per run or spawn.
- tldw_chatbook/Agents/profile_tool_provider.py, library_tool_provider.py, agent_lesson_promotion.py, agent_worktree.py, agent_worktree_recovery.py: tool-thread or cold paths (manual recovery, per-spawn worktree). Linear scans are bounded by small data.
- tldw_chatbook/Agents/fleet_coordinator._retain_locked: json.dumps and deepcopy under the lock are bounded by the 200k-char cap.
- tldw_chatbook/Agents/agent_service.py _safe_agent_step_record: step results are capped at 2,000 chars upstream (agent_runtime.py:3164), so the per-step contains_local_path scan is ~0.25 ms, not the 7 ms measured on a 29 KB result.

## census
| Operation (isolated scratch profile, paths ~2x deeper than default) | Measured | Frequency |
|---|---|---|
| AgentService.run_turn, instant fake model, 0 tools: run log on / off | 2,258 ms / 239 ms | every send |
| RunLogWriter.bind (2x is_within, no context) | 1,617-1,695 ms | every primary run |
| resolve_sensitive_context() | 740 ms | per uncached call |
| is_within(): no context / pre-resolved context | 969 ms / 0.39 ms | per call |
| config.get_user_data_dir() | 45 ms | x13 per sensitive context |
| LocalToolProvider.invoke fs_read (2-line file) | 2.1-3.9 s (1.6 s in _record_fs_read_observation) | per tool call |
| worker_guard admission (fresh thread) | 27-33 ms median, ~1,290 os.open | per tool call and per model call |
| activation.execution(): outer / nested same thread | 38 ms / 13 ms | per guard level (3 per run) |
| MCPPermissionStore.get_kill_switch (file present, outer lease held) | 23-31 ms, 1,190 os.open | >=4 on main loop per send; 2-4 per tool call |
| AgentRunsDB statement (insert or metadata read) | 3.5-8 ms, 245 os.open | per step / lifecycle event |
| AgentRunsDB calls per run: 0 / 1 / 3 tool rounds | 16 / 24 / 34 calls (109 / 197-220 / 225 ms) | per send |
| probe_initial_catalog, N=40 / N=80 schemas (warm-cold) | 14-45 ms / 54-173 ms vs ~1 ms single measure | per send (first request) |
| Marginal import on Chat leg: agent_service / agent_runtime | 14-21 ms / 5-10 ms | boot (first paint) |
| StreamGate total for 20K / 100K chars (4-char chunks) | 17.5 ms / 506 ms | per streamed turn |


# slice-10

## summary
Slice #10 (Chat#4, 12 files, 31,676 lines) is dominated by tldw_chatbook/Chat/console_chat_store.py (22,545 lines; ConsoleChatStore alone has about 560 methods). The rest is the models, compaction, context repository, context window, hydration, activation and actions, markdown export, and the two slash-command helpers.

Hot paths traced into the slice:
- Per chunk: append_stream_chunk, from the controller's direct-provider loop, the agent bridge and realtime.
- Per 0.2 s run tick and per UI resync: messages_for_session (known, TASK-24300). The tick also calls context_control_inputs for the memory banner (chat_screen.py:18358), fork_eligibility for the selected row (chat_screen.py:18289) and the library-activity projection token.
- Per send: the durable commit is already offloaded (TASK-22205). mark_message_complete, stopped and failed are not; they run on the app loop through asyncio.create_task.
- Per conversation open: restore_persisted_session. The bulk reads were moved off the loop; the publication step was not.
- Per transcript selection change: library_activity_snapshot and fork_eligibility.
- Per compaction commit: the lineage walk.

Overall health: the in-memory algorithms are mostly sound.
- Stream folding is incremental (TASK-259), and chunk persistence is deferred to a single-slot executor (task-33081).
- Active-path walks are O(depth). The trace-settlement executor is bounded at 64 with async capacity waits.
- Resume uses one batched attachment fetch and one batched generation-metadata fetch.

The costs concentrate at synchronous DB seams reached from the event loop. The biggest new, measured structural finding: every ChaChaNotes transaction() now pays a Backup_Recovery storage-admission handshake. Measured in a scratch profile, that is about 245 open() calls, 5 JSON decodes and about 4.9 ms per empty transaction, against 26.6 µs for execute_query. It multiplies every slice DB site that runs on the loop:
- terminal persistence, about 6 of 6-9 ms per send;
- fork eligibility, 5.2 ms per tick or selection;
- the per-tick memory-banner read, 6.2 ms;
- resume publication, 23-33 ms of 34-45 ms;
- the library-activity read, about 8 ms.

The fixes group into:
- (a) amortise the admission (DB/Backup_Recovery owners, sibling of TASK-32804.1);
- (b) gate or memoise the per-tick controller projections (memory banner, fork eligibility);
- (c) fix the O(T×R) library-activity projection;
- (d) move terminal persistence and resume publication off the loop, mirroring TASK-22205 and prepare_console_session_data;
- (e) P3 hygiene: the double digest, the deepcopy of frozen exchanges, the discarded per-chunk snapshot, residual inline-persist reads and the resume double read.

Import self-time of the slice modules on the Chat first-paint leg is about 35-45 ms (store 14-21 ms, models 10-11, repository 6-9, compaction 6-8). This is the god-module and dataclass cost already tracked by TASK-32809.2 and TASK-32809.3.

All probes ran with isolated HOME, XDG_* and TLDW_CONFIG_PATH under the scratch dir, file DBs in scratch, and TLDW_TEST_MODE=1. Scripts are in /private/tmp/claude-501/-Users-macbook-dev-Documents-GitHub-tldw-chatbook/15c5e43c-3dea-41a4-87ab-3b132387df58/scratchpad/audit/scratch/slice10/.

## clean areas
- tldw_chatbook/Chat/console_command_grammar.py: pure tokenizer and registry dict lookup, no I/O
- tldw_chatbook/Chat/console_command_suggestions.py: module-level precompiled regexes, per-keystroke suggestions capped at max_results=200, pure
- tldw_chatbook/Chat/console_context_window.py: bounded 64-entry OrderedDict LRU with TTL and single-flight Future, streamed probe capped at 256 KiB, send path reads cache only and refreshes in background (task-33081). The key property re-hashes per access, which is negligible
- tldw_chatbook/Chat/console_context_policy.py: pure validation and merge, no I/O
- tldw_chatbook/Chat/console_conversation_actions.py: pure menu model
- tldw_chatbook/Chat/console_conversation_activation.py: async orchestration with shields and serialization, no I/O of its own
- tldw_chatbook/Chat/console_conversation_markdown.py: cold export path, regex precompiled
- tldw_chatbook/Chat/console_conversation_hydration.py: explicit-stack O(N) tree flatten (TASK-22206), one batched attachment fetch, prepare_console_session_data runs its reads via asyncio.to_thread (only the double read noted as P3)
- console_chat_store.py streaming core: append_stream_chunk has no per-chunk sqlite, _fold_stream_buffer_without_persistence collapses the buffer (TASK-259), messages_for_session defers the pending-row INSERT to a single-slot executor (task-33081)
- console_chat_store.py tree maintenance: _recompute_active_path and _with_tool_markers are O(path+markers) per mutation, active_path_message_ids is O(depth), _ingest_full_tree and _chain_legacy_flat_roots are linear
- console_chat_store.py provider-trace settlement: single-worker executor bounded at _MAX_PROVIDER_TRACE_SETTLEMENT_WORK=64, capacity waits are asyncio.Event (no busy-wait)
- console_chat_store.py settings persistence drain: writes go through asyncio.to_thread (lines 9066 and 9176)
- console_chat_store.py exchange blob cache: compressed once per (run_tag, seq, status) key (TASK-19325), pruned per message; only the retention after terminal is flagged
- console_chat_store.py accessors presentation_context, dispatch_recovery_*, session_mru_ids, payload/display revisions, variant_sets_for_conversation: O(1) or O(sessions)
- console_chat_store.py is a single app-runtime-owned instance (console_runtime.py:3174), and its executors are shut down in end_app_runtime
- console_chat_store.py capture-policy, voice-promotion, ephemeral-promotion and fork-staging regions: cold paths (settings apply, fork, promotion), not audited further for per-interaction cost
- tldw_chatbook/Chat/console_context_compaction.py planning (complete_durable_units, plan_compaction, select_effective_memory): linear over the lineage; provider calls go through asyncio.wait_for plus the gateway (to_thread)
- tldw_chatbook/Chat/console_context_repository.py SQL is parameterized, listing APIs validate limit/offset pages, and the memory/selection tables are small per conversation (the per-transaction admission and lineage walk are flagged separately)

## census
| probe (isolated scratch profile, file DB, this Mac, idle) | result |
|---|---|
| empty `with db.transaction(): pass` (ChaChaNotes) | 4.9 ms/op; ~245 posix.open + 253 fstat + 5 JSON decodes + 10 listdir per txn |
| `db.execute_query("SELECT 1")` (no txn wrapper) | 26.6 µs/op |
| `db.get_conversation_active_leaf` (txn read) | 4.3 ms/op |
| `store.mark_message_complete` (plain turn) | 53 SQL stmts, 6.0-8.0 ms flat from turn 1 to 200; ~6 ms of it is one storage admission |
| `store.fork_eligibility(selected)` (durable session, 100 turns) | 5.2 ms/call, 97% in one txn read |
| `restore_persisted_session` publication (400 msgs) | 34-45 ms on loop, 3 txns = 23-33 ms admission, 15 stmts |
| `prepare_console_session_data` (off-loop, warm) | 92 ms at 400 msgs |
| `library_activity_snapshot` end-to-end | 9.6 ms at 50 turns (100 traj rows), 20.8 ms at 200 turns (400 rows) |
| project_library_activity T×R loop only | 0.7 ms (T50/R100), 10.3 ms (T200/R400), 25.4 ms (T200/R1000), 63.3 ms (T400/R1200) |
| per-tick memory-banner pieces (200 turns) | messages_for_session 1.9 ms + get_message_versions 0.84 ms + load_applicable_branch_memory 6.2 ms (+ snapshot build, estimated) |
| `_load_persisted_branch_state` (400-msg lineage) | 1,202 statements, 10.5 ms inside BEGIN IMMEDIATE (excluding admission) |
| prefix_digest + _persisted_prefix_digest (400 msgs x 1 KB / 4 KB) | 3.2 / 9.2 ms (a single digest is 1.8 / 4.6 ms) |
| deepcopy(message) with exchanges (30 tool schemas) | 0.21 ms (1 exchange), 2.2 ms (10), 5.7 ms (25) |
| dataclasses.replace(ConsoleChatMessage) | 4.0 µs |
| get_messages_for_conversation with vs without image BLOBs (400 msgs, 20 x 1 MiB) | 6.9 vs 7.2 ms (warm page cache, no difference) |
| import self-time (Chat first-paint leg) | store 14-21 ms, models 10-11 ms, repository 6-9 ms, compaction 6-8 ms, others under 2.3 ms |


# slice-11

## summary
Slice #11 "Chat#5" has 39 files and about 25k lines. It is mostly pure Console projection, state and repository code. Hot paths traced:
- Per send: console_prepared_request (window, serialize, account), console_history_budget, and the console_dispatch_repository insert/CAS/settle transactions.
- Per 0.2 s streaming tick: the cost chip via console_cost_tracker, and the transcript row plan, which calls console_message_actions for the selected message.
- Per keystroke or control-bar sync: the console_display_state control/inspector/evidence builders and the prompt-queue shelf.
- Per conversation visit: restore calls ConsoleDispatchRepository.reconcile_for_session on the loop; image rows use ConsoleImageRenderCache.
- Per boot: console_launch_wake discovery, plus the module imports of 32 of the 39 slice modules, which load before the Console first paint.

Overall health is good. Most of the obvious traps are already handled:
- Token estimates are memoized (TaskEstimate cache plus the token_counter memo).
- The evidence-bundle parse is cached on the launch object, and the runtime hands back the exact same object, so the cache really hits.
- Library-policy DB work runs through asyncio.to_thread with an owned connection.
- Fleet wake uses run_owned_db_call.
- The prompt-queue snapshots are revisioned and reuse entries.
- Paste-attach config reads are now lru_cached (the core-review P1 is fixed).
- The prepared-request multi-pass counting was already measured at 3.9 ms warm.

Six findings survive verification, four of them measured:
1. Image pixel thumbnails are built lazily on the event loop after the off-loop prepare: 58–75 ms measured for 16 images when a conversation with images opens.
2. assistant_canvas_html_blocks builds a fresh MarkdownIt and parses the whole selected message 2–3 times per transcript refresh. That is 1.5–5.1 ms each, and refreshes run at 5 Hz while streaming and on every selection move.
3. Launch-wake discovery opens a private SQLite connection and full-scans agent_runs synchronously on the loop at every boot, and scans twice when results are pending. Cost is 2–49 ms depending on how many runs are stored.
4. PIL is imported at module scope by console_image_view, the first of 7 PIL importers on the Console first-paint path. About 10–12 ms, measured.
5. Evidence formatting is not cached (P3).
6. Tiny unbounded per-session snapshot caches in the prompt queue (P3).

Structural note: at import time, the slice's frozen/slotted dataclass and pydantic class construction costs about 40–60 ms of module-body time. console_display_state alone is 5.3 ms and console_dispatch_checkpoint is 4.2–5.0 ms. About 14 ms of that is cold-feature modules: image gen/edit, video, hands-free, exchange capture, message actions.

Files not loaded at Console first paint (already lazy): console_interrupt_rounds, console_environment_state, console_exchange_export, console_help, console_launch_wake, console_persona_assignment.

## clean areas
- tldw_chatbook/Chat/console_cost_tracker.py: TokenEstimateCache is keyed by row id and every hit is checked against the row text, so it can't serve a stale number. The chip reuses its settled snapshot while the display revision is unchanged. fingerprint_payload runs only when the payload revision changes while idle, and measured 1.36 ms for 301 rows / 602 KB. build_cost_rows goes through the memoized token_counter.
- tldw_chatbook/Chat/console_prepared_request.py: about 9 wire-count passes per send, but the token_counter memo (TASK-18602) absorbs them. The core review measured 3.9 ms warm at 43k tokens, so this is verified fine and not re-reported.
- tldw_chatbook/Chat/console_history_budget.py: prune and retire passes are linear, and the binary-search window uses the memoized counter. No per-chunk use.
- tldw_chatbook/Chat/console_dispatch_repository.py: every write is one db.transaction(immediate=True) per turn transition, and there is no per-chunk SQL. reconcile_for_session on restore measured 2.2 ms median on a 400-message active path with no checkpoint, in-memory DB. read_for_session is used only by tests.
- tldw_chatbook/Chat/console_dispatch_checkpoint.py: small strict JSON parses, once per turn (only its import-time cost is noted, under F4).
- tldw_chatbook/Chat/console_library_policy_coordinator.py and console_library_policy_repository.py: file-backed DB calls go through asyncio.to_thread plus operation_owned_connection. This is the good pattern.
- tldw_chatbook/Chat/console_prompt_queue.py and console_prompt_queue_coordinator.py: revisioned snapshot cache with per-entry reuse, bounded tombstones, and owner-thread asserts (only the tiny leak in F6).
- tldw_chatbook/Chat/console_fleet_wake.py: event-driven through call_soon_threadsafe. DB reads go through run_owned_db_call. The recover() 20 Hz poll is bounded by _ui_ready at boot.
- tldw_chatbook/Chat/console_fleet_attention.py: announces are rare, and the unseen-id listing is cached by revision in UI/Console_Modules/fleet.py.
- tldw_chatbook/Chat/console_interrupt_rounds.py: the 1 s event.wait poll runs on the agent worker thread, and only while a decision is pending.
- tldw_chatbook/Chat/console_library_activity_buffer.py: bounded to 256 events per session and 64 per batch, and the persist callback runs outside the lock.
- tldw_chatbook/Chat/console_generate_image.py and console_generate_video.py: callers offload through asyncio.to_thread, and there is one shared single-worker executor that is never re-created per call.
- tldw_chatbook/Chat/console_paste_attach.py: _supported_patterns is lru_cached, so the core-review 98 ms P1 is fixed.
- tldw_chatbook/Chat/console_display_state.py: evidence_bundle_from_launch is cached on launch identity, and the runtime returns the exact staged launch object, so it hits. ConsoleControlState and ConsoleInspectorState builders are pure and cheap. middle_elide_path and diff-feedback rendering are linear or byte-capped.
- tldw_chatbook/Chat/console_exchange_capture.py and console_exchange_export.py: capture is opt-in, and SAFE mode keeps only 8 tail rows. Export is a cold path.
- tldw_chatbook/Chat/console_live_work.py: deepcopy happens only when a launch is staged.
- Pure helpers with no hot-path cost: console_environment_state.py, console_onboarding_state.py, console_expression_state.py, console_ephemeral.py, console_glyphs.py, console_help.py, console_prefill.py, console_provider_endpoints.py, console_library_destination.py (networks precompiled at module scope), console_endpoint_provenance.py, console_project_instructions.py, console_library_policy.py.
- Cold paths: console_generation_settings_metadata.py (conversation open), console_persona_assignment.py (settings modal), console_hands_free.py (pure state machine; its 10 Hz tick lives in the UI module and runs only while hands-free is active), console_image_edit_operations.py (stdlib only).

## census
| module (slice #11) | self import ms, `-X importtime` of chat_screen (noisy, includes GC) | isolated body ms (GC off, deps preloaded) | before first paint? |
|---|---|---|---|
| console_display_state | 7.1 / 12.0 | 5.3 | yes (app import, via console_runtime) |
| console_dispatch_checkpoint | 6.4 / 28.1 (GC spike) | 4.2–5.0 | yes (app import) |
| console_message_actions | 4.4 | 2.0–2.6 | yes (chat_screen) |
| console_library_policy | 2.8–4.3 | – | yes |
| console_prompt_queue | 4.0 | – | yes |
| console_generation_settings_metadata (pydantic) | 3.5 | – | yes |
| console_prepared_request | 3.4 | – | yes |
| console_image_edit_operations | 1.7–2.9 | – | yes (app.py:133, module scope) |
| console_generate_image | 2.6 | – | yes |
| console_live_work | 2.3–2.7 | – | yes |
| console_hands_free | 2.3 | – | yes |
| console_history_budget | 2.1 | 1.3–2.0 | yes |
| console_exchange_capture | 2.0 | – | yes |
| console_cost_tracker | 2.0 | – | yes |
| console_image_view (+PIL.Image) | 0.9 self / 9.5 cumulative | PIL.Image adds 10.1–11.8 over a textual baseline | yes, first PIL importer |
| 32 of 39 slice modules loaded at chat_screen import | ~64 ms summed self (noisy) | – | – |

Launch-wake discovery query (plain-sqlite3 replica of the agent_runs schema and indexes, new read-only connection each run, warm cache). Query plan: `SCAN child USING INDEX idx_agent_runs_conversation; SEARCH parent USING PK`
| rows | DB size | median ms |
|---|---|---|
| 5,000 | 3.6 MB | 2.30 |
| 20,000 | 14.5 MB | 7.92 |
| 10,000 (2.8 KB rows) | 41 MB | 13.3 |
| 30,000 (2.8 KB rows) | 124 MB | 48.9 |


# slice-12

## summary
Slice #12 (Chat#6, 34 files, ~30.3k lines) is mostly the Console send pipeline: `console_provider_gateway.py` (7.5k lines; handles resolve_for_send, stream_chat, capture and trace verification), `console_runtime.py` (5k lines; the app-owned runtime with its config accessors, timers and maintenance loops), plus session settings, trace, voice and speech helpers.

Hot paths traced:
- **Per send:** `resolve_for_send`, Capture-On trace verification and exchange capture, trace-boundary writes, and a readiness probe.
- **Per streamed chunk:** the llama.cpp SSE loop, `record_exchange_content`, `observe_response`, and ThinkingCapture.
- **Idle / always-on:** the Canvas policy watcher at 4 Hz from App.on_mount, and legacy trace maintenance at 1 Hz (already tracked as TASK-31501).
- **First Console use:** gateway construction and `ensure_chat_store` recovery.
- **Modal opens:** the model popover and settings modal call `resolve_for_send`.

Overall health: the generic provider path does its blocking work well. Provider iteration, readiness, dictionaries, world-info, @-references and auxiliary calls all run on threads. Metadata refreshes are backgrounded, and streaming is not quadratic in the consumer.

Two P1 findings dominate:
1. **Credential sanitizing repeated per call.** Every Capture-On provider call (the default for manual sends) runs CredentialSanitizer 24-27 times over the whole transcript. Measured: 67-97 ms for 42-60 KB on the llama.cpp branch, where it runs directly on the event loop. On the generic path it took 311 ms for 222 KB in the worker thread, holding the GIL and delaying the first token.
2. **Warm `load_settings()` still pays the admission handshake.** The runtime-owned config accessors (`_provider_config_for_app` and friends) call `load_settings()` on every read. The TASK-32804.1 fastpath only covered `get_cli_setting`, so each call still pays about 8 ms in a real profile (15-29 ms measured here, ~600 file opens). Every send pays this 4 or more times. The Inspector's Next-Send snapshot pays it once per transcript message.

P2 findings:
- Network probes on non-send modal paths.
- An eager SSL/httpx client built on the UI loop.
- The always-on 4 Hz thread-hopping Canvas poll.
- Synchronous trace-boundary SQLite write transactions on the loop.
- A startup recovery scan with no index on `state`, which grows with every provider call.

Structural notes:
- The same sanitize policy is re-applied in three layers (gateway capture, exchange capture, and `verify_provider_request_shadow`). Each layer re-walks the full, immutable history on every call. Caching sanitized results per semantic revision would turn the per-send cost from O(transcript) into O(new message).
- The llama.cpp branch of `stream_chat` does its verification and capture inline on the loop, while the generic branch offloads the same work to a thread.
- The gateway god module imports console_prepared_request, which pulls in Agents.agent_runtime and agent_models (about 40 ms inside the chat_screen import). This mostly overlaps the Console first-paint import set, so it is noted here rather than filed.

## clean areas
- tldw_chatbook/Chat/console_references.py: git diff subprocess and file reads run via asyncio.to_thread in the controller's send path (console_chat_controller.py:10112-10120)
- tldw_chatbook/Chat/console_side_chat.py: reply accumulated in list plus join and capped; stream goes through gateway
- tldw_chatbook/Chat/console_send_diagnostics.py: bounded to 64 events, no file I/O on caller, monitor closed off-loop
- tldw_chatbook/Chat/console_scratch_space.py: one mkdtemp per session, cleanup on a worker thread
- tldw_chatbook/Chat/console_settings_defaults.py, console_settings_apply.py, console_settings_durability.py: apply/refresh called via asyncio.to_thread (chat_screen.py:5780, 5949-5957); preview uses shallow copies
- tldw_chatbook/Chat/console_session_settings.py: build_console_settings_readiness measured 22-32 us/call; default settings 29 us; build_console_provider_options 0.55 ms (modal/popover only); context estimate is incremental (history_used_tokens)
- tldw_chatbook/Chat/console_provider_support.py: resolve_console_provider_identity ~5 us, supported_generation_fields ~14 us (micro, unmemoized but not hot enough to matter)
- tldw_chatbook/Chat/console_switcher_state.py: linear aggregation; 5 Hz recompute already tracked by TASK-31506
- tldw_chatbook/Chat/console_rail_state.py: small tuple scans, precompiled regex
- tldw_chatbook/Chat/console_skill_resolver.py: find_embedded_mentions 2.7 ms on a 46 KB draft, once per send
- tldw_chatbook/Chat/console_speech_text.py: MarkdownIt constructed per completed reply (~160 us), TTS path only
- tldw_chatbook/Chat/console_speech.py, console_speech_preferences.py, console_save_targets.py, console_tool_activity.py, console_trace_chunk_rows.py, console_trace_errors.py, console_session_endpoint_policy.py: pure, small
- tldw_chatbook/Chat/console_roleplay_identity.py, console_roleplay_metadata.py: resolve_console_message_presentation 1.8 us/row
- tldw_chatbook/Chat/console_speculative_voice.py, console_speculative_voice_session.py, console_realtime_loop.py: lazily imported (hands_free import_module); diagnostics persisted only at first-stage events; get_cli_setting on warm fastpath
- tldw_chatbook/Chat/console_raw_cli.py: executor built lazily on first execute(); pydantic model definitions ~ms at boot only
- tldw_chatbook/Chat/console_thinking_history.py: prepare-time only
- tldw_chatbook/Chat/console_trace_legacy.py: batched per-conversation match index; runs inside maintenance via run_owned_db_call (cadence tracked by TASK-31501)
- tldw_chatbook/Chat/console_provider_gateway.py generic path: chat_api_call iteration, normalization and dispatch-start commit run in asyncio.to_thread worker; complete_auxiliary offloads sync adapter; reasoning/context-window metadata refresh are background tasks with TTL and in-flight de-dup
- tldw_chatbook/Chat/console_runtime.py: chat-dictionary and world-info appliers are awaited via asyncio.to_thread by the controller; receipt reads threaded; legacy maintenance cadence already TASK-31501

## census
| Probe (isolated scratch profile, no network) | Result |
|---|---|
| Capture-On llama.cpp send, 20 rows / 42 KB transcript | 27 CredentialSanitizer calls, 67-69 ms before adapter entry, on the calling (event-loop) thread |
| Capture-On llama.cpp send, 28 rows / 60 KB (100-message history trimmed by the default window) | 27 calls, 96 ms |
| Capture-On generic (openai) send, 100 rows / 222 KB | 24 calls, 311-316 ms before chat_api_call, in the worker thread (holds the GIL) |
| One CredentialSanitizer pass over a 225 KB, 100-message payload | 55 ms (about 15 regex subs per string) |
| Warm `load_settings()` / `_provider_config_for_app` | 17-29 ms per call, 597 posix.open (12-component scratch path; the core review measured about 8 ms on a real profile) |
| 40 x `_global_user_display_name_for_app` (per-message path in `_presented_message_snapshots`) | 603 ms |
| `load_settings()` nested inside `operation(config)` | 0.05 ms |
| `get_canvas_execution_enabled()` / Canvas watcher tick (to_thread + task) | 2 us / 99 us CPU, 277 us wall; runs 4 per second forever |
| `httpx.AsyncClient()` construction (SSL context) | 76 ms first, then 11-37 ms |
| `recover_open_calls` query shape, in-memory, no `state` index | 5 ms at 10k calls, 31 ms at 50k calls |
| llama per-chunk overhead (2 JSON parses + chunk sanitize + envelope sizing) | about 20 us per chunk |
| ThinkingCapture per delta | 4 us at 5 KB, rising to 11 us at 244 KB; 16k deltas total 87 ms |


# slice-13

## summary
Slice 13 (Chat#7, 31 files / 31.8k lines) is dominated by the Capture-On semantic trace ledger (console_trace_*). Capture is on by default ([console] exchange_capture defaults True, normalized_writes_enabled True, and production wires _LazyTraceBoundaryFactory), so every manual send to a durable conversation runs this code, and every agent tool-loop iteration runs it again.

Hot paths identified:
(1) Pre-dispatch trace reservation: ConsoleProviderGateway.stream_chat -> _reserve_trace_call -> ConsoleTraceBoundaryFactory.__call__. It runs synchronously on the Textual loop (turn task is asyncio.create_task from UI/Console_Modules/wiring.py:284 -> console_runtime.accept_turn), inside BEGIN IMMEDIATE.
(2) _verify_trace_shadow -> verify_provider_request_shadow -> CredentialSanitizer. It runs in the provider worker thread for generic providers and directly on the loop for the llama.cpp route.
(3) mark_dispatch_started / bind_and_mark_dispatch, run in the worker thread.
(4) mark_response_started runs on the loop at the first chunk. Settlement runs at stream end.
(5) Background: the 1 Hz legacy maintenance loop plus a full-ledger GC every 60 s after any epoch change.
(6) Startup: recover_open_calls inside ensure_chat_store on first Console mount.

I measured an isolated real-SQLite driver that mirrors Tests/Benchmarks/test_console_trace_growth.py, with an instant fake adapter and 1.5 KB messages. Pure trace overhead per send grows from about 108 ms at 10 messages to 378-505 ms at 200 messages. The loop was starved about 308 ms per send at 200 messages (heartbeat; contiguous max 55 ms). The biggest levers:
- The credential sanitizer re-scans the whole transcript 5-7 times per send (one pass is 62.5 ms at 300 KB; a memoized repeat pass is 0.33 ms).
- A fresh hardened ChaChaNotes connection, including a private-sqlite helper subprocess spawn, is opened and closed per trace write on worker threads (60-73 ms each).
- On-loop O(N) reservation work with 3 point queries per prior revision (9 -> 23-41 ms).
- console_trace_calls has no run_id index and no open-state index. With 50k calls, get_run_origin runs a full SCAN (11.7 ms) and startup recovery takes 94 ms. Candidate indexes, verified without sqlite_stat1: 0.007 ms and 0.04 ms.
- Trace GC re-marks the entire reachable ledger (about 7-10 us per row) in BEGIN IMMEDIATE every minute after any send.

Cross-slice discovery (reported as F9): every outermost ChaChaNotes transaction pays about 5 ms for Backup_Recovery storage admission (245 open() calls per transaction). This multiplies every trace write and the 1 Hz maintenance tick (now measured at 6-9 ms CPU per tick; TASK-31501 new evidence).

Known tasks re-evidenced:
- TASK-31501: tick cost.
- TASK-31505: a new site. The custom-PII worker is spawned per PRIOR artifact inside the on-loop prefix match.
- TASK-22504: voice_input import is 4 ms self; no new evidence.

Structural note: correctness-grade verification (re-deriving and re-sanitizing the full provider surface on every call) is done with O(transcript) work per send. The architecture needs append-only incremental verification: cache per-revision sanitized bytes and compare the delta only. Otherwise every long agent session pays seconds per tool call.

Suggested PR groups:
- A, schema: F4+F5 indexes with plan pins.
- B, sanitizer: F1 memo, plus skipping the redundant passes in final_values, plus moving llama verify off the loop.
- C, connections: F3 dedicated trace-writer connection, plus F12.
- D, reservation: F2 off-loop and batched owner/revision lookups.
- E, GC: F6 gate on orphaning mutations and chunk the transactions; fold in the TASK-31501 backoff.
- F, turn context: F10 _freeze fast path.
- G, P3 hygiene: F11, F13-F15.
- F9 belongs to the DB/Backup_Recovery owner.

## clean areas
- tldw_chatbook/Chat/console_trace_projection.py: read_calls runs via asyncio.to_thread (the inspector loader is threaded); project_capture_for_viewer runs per opened call only (cold, user-initiated)
- tldw_chatbook/Chat/console_trace_native_reader.py: inspector reader runs off-loop; iter_message_call_lineage does an N+1 get_call per lineage row, but it is off-loop and user-initiated (P3, not filed)
- tldw_chatbook/Chat/console_trace_metrics.py: lock-guarded counters, trivial
- tldw_chatbook/Chat/console_trace_models.py: module-scope re.compile plus validators; cheap
- tldw_chatbook/Chat/console_trace_provenance.py: module scope is constants and census tuples only; admit_message_provenance's per-message ensure_current_revision loop is covered under F2
- tldw_chatbook/Chat/console_trace_regex_worker.py: covered by known TASK-31505 (new site reported in F8)
- tldw_chatbook/Chat/console_trace_maintenance.py PhysicalTraceCompactor: VACUUM is gated by size, freelist, idle thresholds and a dispatch pause; fine
- tldw_chatbook/Chat/console_transaction_contribution.py: thin validated cursor wrapper; clean
- tldw_chatbook/Chat/console_turn_grouping.py: pure O(n) grouping; only minor recompute in thinking_activity_id (F14)
- tldw_chatbook/Chat/console_turn_preparation.py: pure state machine; only _execution_context_with_attempt re-freeze (covered by F10)
- tldw_chatbook/Chat/console_voice_input.py: probe() measured 0.5 ms; capture work spawned off-loop; warm-up on a daemon thread; config reads cache-backed; import is 4 ms self (boot-leg deferral already tracked by TASK-22504)
- tldw_chatbook/Chat/console_voice_process.py: spawn and monitor bounded to active voice sessions; 20 Hz poll only while a voice process lives; source_identity cost noted in F15
- tldw_chatbook/Chat/console_voice_process_effects.py: 1 ms sleep loop only while superseded attempts hold outstanding credit (bounded)
- tldw_chatbook/Chat/console_voice_attempts.py, console_voice_supervisor.py, console_voice_preflight.py, console_voice_eligibility.py, console_voice_controls.py: no loop-blocking or DB work found
- tldw_chatbook/Chat/console_voice_trace_gateway.py, console_voice_trace_promotion.py: per-voice-turn validation and JSON canonicalization, bounded; opt-in qualified path
- tldw_chatbook/Chat/console_voice_promotion.py: census-pinned absent at _ui_ready; 28 ms self-import (18 slotted dataclasses) paid once on first voice use (not filed)
- tldw_chatbook/Chat/console_voice_settings.py: speculative_voice_qualified hashing noted in F16; otherwise clean
- tldw_chatbook/Chat/console_visual_evaluation.py and console_visual_benchmark.py: dead at runtime (tracked by task-19571); no runtime cost

## census
| conversation size (1.5 KB user msgs, instant fake adapter) | stream_chat trace overhead | factory reservation (ON LOOP) | _verify_trace_shadow (worker; ON LOOP for llama.cpp) | mark_dispatch_started (worker, incl. fresh conn) | mark_response_started (ON LOOP) | loop starvation per send (heartbeat sum / max gap) |
|---|---|---|---|---|---|---|
| 10 msgs | 103-137 ms | 7.4-10.2 ms | 15.5-16.1 ms | 67-92 ms | 5.4-13 ms | 52 ms / 31 ms |
| 50 msgs | 179-241 ms | 10.7-13.5 ms | 75-97 ms | 76-111 ms | 5.5-16.5 ms | 126 ms / 42 ms |
| 100 msgs | 254-329 ms | 15-20 ms | 167-178 ms | 79-113 ms | 6-7.6 ms | 186 ms / 50 ms |
| 200 msgs | 378-505 ms | 23-28 ms (41 ms profiled) | 309-414 ms | 69-105 ms | 6.6-12.7 ms | 308 ms / 55 ms |

Micro-benchmarks, all measured in isolated scratch:

| item | measured |
|---|---|
| One CredentialSanitizer pass over 200 x 1.5 KB | 62.5 ms (memoized repeat: 0.33 ms) |
| Operation-owned trace write on a reused worker thread | 60-73 ms/op, vs 5-8 ms with a held connection |
| Any outermost ChaChaNotes transaction (storage admission) | about 5 ms, 245 open() calls |
| Legacy maintenance steady tick | 6-9 ms CPU |
| Trace GC collect | 15 ms @ 700 rows -> 40-64 ms @ 5.6k rows |
| Calls @ 50k rows, no stat1: next_seq | 5.6 ms -> 0.007 ms with (run_id, call_sequence) index |
| Calls @ 50k rows, no stat1: get_run_origin | 11.7 ms -> 0.007 ms with the same index |
| Calls @ 50k rows, no stat1: recover_open_calls | 94 ms -> 0.04 ms with partial open-state index |
| _freeze of 3 x 300 world-book entries + 200 dictionary entries | 9 ms per freeze (x2 per send) |


# slice-14

## summary
Slice #14 (Chat#8): 40 files, ~21.4k lines, all read or skimmed. I traced the hot paths from their callers.

Hot paths found:
- **Keystroke and send path:** provider readiness; Console library-activity projection (message selection and every send); boot imports of about 16 slice modules through console_runtime and console_settings_defaults.
- **Per streamed chunk:** llamacpp_think_filter, stream_stall_watchdog, reply_sentence_sequencer, thinking_blocks.canonical_json_text_bytes.
- **Per send or per tool persist:** thinking_blocks, provider_continuation, message_metadata and provider_usage validators; conversation_send_refusal; prompt_history.append.
- **Hands-free voice:** the TTS PCM bridge on the app loop.
- **Console attention recompute:** conversation_local_marks_service.
- **Modal or first-run:** scope picker listers, local server discovery.

Overall health: the per-chunk streaming helpers are clean. They are bounded and O(chunk). Readiness itself is cheap at about 12 µs per call, so the keystroke concern sits with its callers (TASK-24454 / TASK-32804.3).

Main problems, in order:
1. **F1 (P1).** `library_activity_snapshot` runs on the UI loop on every message-selection keypress (j/k) and on every send. Its cost grows as turns × rows. It does a synchronous fetch of all trajectory rows and strictly re-decodes each library row at 464 µs apiece. Measured 99 ms at 100 turns with library use. The 2026-09-17 review judged it "not per-tick" but missed the selection path.
2. **F2 (P1).** The hands-free TTS bridge busy-polls at 1 kHz with `asyncio.sleep(0.001)` on the Textual loop for the whole of spoken playback, even though `CreditWindow` already has an event-driven waiter.
3. **F3 (P2).** Provider-continuation checkpoints are fully re-parsed about 7 times and rewritten whole on every tool-call transition. That is O(n²) over an agent run, measured at 21.7 ms per pass at 128 calls.
4. **F4 (P2).** The scope-picker tag vocabulary still runs up to 200 `SELECT n.*` queries just to count rows. The notes lister fetches 500 full notes twice per refresh.

The rest are P3 hygiene: path scan on full tool outputs, httpx TLS client per localhost probe, LIKE-prefix scan with no LIMIT, full prompt-history rewrite plus fsync per send past the cap, a boot import of the 1,855-line setup-persistence module for one helper, test-only dead code carrying poll loops, and an unbounded ScopeCache.

Structural note: several validators in this slice (continuation, library activity, thinking) re-validate trusted canonical data at every layer: runtime, store, DB. Each "boundary" pays a full parse, so the cost multiplies with the number of layers rather than the size of the change.

## clean areas
- tldw_chatbook/Chat/provider_readiness.py - get_provider_readiness measured 4-12 us/call, verdict 4-9 us; the keystroke cost is caller-side (TASK-24454, TASK-32804.3)
- tldw_chatbook/Chat/thinking_blocks.py - canonical dump of a 255KB envelope 1.6 ms, DB-side parse+dump 2.3 ms; runs twice per turn; per-delta canonical_json_text_bytes is O(delta)
- tldw_chatbook/Chat/llamacpp_think_filter.py - streaming splitter is O(chunk) with a bounded tag-suffix buffer
- tldw_chatbook/Chat/reply_sentence_sequencer.py - content buffer force-split at 200 chars, so the boundary rescan per feed is bounded; the queue is tiny
- tldw_chatbook/Chat/stream_stall_watchdog.py - Py3.12 wait_for uses a timeout context (no task per item); session tracker registry is capped at 512
- tldw_chatbook/Chat/local_reasoning.py - per-send template sha256 only; gateway caches template metadata and refreshes it in the background
- tldw_chatbook/Chat/provider_usage.py, usage_recorder.py, cost_display.py, sampling_params.py, provider_catalog.py, provider_failures.py, trace_export_profiles.py - pure, cheap helpers
- tldw_chatbook/Chat/message_metadata.py - from_json/to_json per row are cheap; is_empty builds a default instance (~us)
- tldw_chatbook/Chat/custom_endpoint_registry.py - entry_for re-validates all entries but measured 41 us with 5 endpoints
- tldw_chatbook/Chat/provider_endpoint_contract.py, provider_test_evidence.py - bounded validators
- tldw_chatbook/Chat/provider_setup_persistence.py runtime - deepcopy of the whole config only on first-run-wizard save (cold)
- tldw_chatbook/Chat/permission_summary_service.py - config projection plus one readiness call
- tldw_chatbook/Chat/rag_scope.py - read/write run off-loop via asyncio.to_thread in UI/Console_Modules/retrieval.py (apart from the unbounded ScopeCache, F11)
- tldw_chatbook/Chat/conversation_archive_actions.py - storage_call uses asyncio.to_thread; per-send archive-state read is off-loop
- tldw_chatbook/Chat/conversation_local_marks_service.py - list_marked_conversation_ids is cached with generation-safe invalidation; unread_ids_for is chunked and threaded (except the LIKE scan, F7)
- tldw_chatbook/Chat/server_chat_conversation_service.py, server_chat_loop_service.py - thin async wrappers; the client is cached by the provider (runtime_policy build_client)
- tldw_chatbook/Chat/console_worktree_recovery.py - DB reads in asyncio.to_thread (opens a fresh AgentRunsDB per page, but it is a cold path)
- tldw_chatbook/Chat/console_workspace_actions.py - static menu data
- tldw_chatbook/Chat/trajectory.py derive_trajectory - roughly linear; only the trajectory screen and review paths use it (cold)
- tldw_chatbook/Chat/trajectory_export.py, trajectory_import.py - file I/O via asyncio.to_thread; json round-trip copies only on export (cold); correctly kept off the first-paint leg
- tldw_chatbook/Chat/voice_phrase_sequencer.py, library_preparation.py - small facade and projection
- tldw_chatbook/Chat/prompt_history.py complete() ghost scan - already tracked by TASK-21120

## census
| Probe (isolated env, audit tree 840ed2ca58) | Measured |
|---|---|
| 1 ms `asyncio.sleep` poll loop, bare asyncio | ~840 wakeups/s, 2.4% of a core per poller |
| `decode_library_activity_event` (8 source refs) | 464 us/row |
| `encode_library_activity_event` | 430 us/event |
| library_activity_snapshot projection only, 50 / 100 / 200 turns (10 rows/turn, no activity rows) | 4.3 / 13.0 / 52.7 ms |
| same projection, 25 / 50 / 100 turns with 2 library searches per turn | 22.9 / 46.0 / 99.0 ms |
| continuation dump+parse, 8 / 32 / 64 / 128 calls (90KB / 381KB / 770KB / 1.5MB) | 1.5 / 5.2 / 10.7 / 21.7 ms per pass |
| `transition_provider_call`, 64 / 128 calls | 7.4 / 21.8 ms |
| `contains_local_path`, 487KB prose / 509KB JSON | 15.2 / 43.3 ms |
| `redact_local_paths`, 509KB JSON | 112 ms |
| `httpx.AsyncClient()` first / warm / `verify=False` | 59.3 / 11.2 / 0.4 ms |
| `get_provider_readiness`, openai / llama_cpp / missing key | 12.3 / 4.4 / 8.9 us |
| `entry_for`, 5 custom endpoints | 41 us |
| thinking envelope 255KB: dump / DB validate | 1.6 / 2.3 ms |
| boot import self-time of slice modules on the `import tldw_chatbook.app` path | ~22 ms total (message_metadata 3.3, provider_setup_persistence 3.3, provider_test_evidence 2.9, thinking_blocks 2.2, provider_continuation 1.8, custom_endpoint_registry 1.6, ...) |
| `EXPLAIN QUERY PLAN` for the marks `LIKE 'console_unseen:%'` query | SCAN USING COVERING INDEX plus TEMP B-TREE (a range predicate gives SEARCH) |


# slice-15

## summary
Slice #15 Chunking (61 files, ~23k lines). It stays off the boot path: no Chunking module is in Tests/Performance/boot_budget_snapshots/ui_ready_modules.txt, so TASK-21102 still holds. Entry points traced: (a) improved_chunking_process (Chunk_Lib shim). Every local ingest calls it: PDF, OCR, plaintext/document via local_file_ingestion._chunk_text_for_ingest (default method 'sentences'), audio/video, the Library re-chunk (thread worker plus asyncio.run), summarization, and template apply. It runs in the parse pool or worker threads, not on the loop. (b) The engine Chunker, including the hierarchical path used by RAG parent/child. (c) template_runtime._execute_report, used by Lab preview children and admin apply_template. (d) Lab: LabCoordinator/AutosaveWriter/LocalPreviewRunner are app-owned; the Lab screen runs lab_state transitions through asyncio.to_thread. (e) First-use imports on the UI thread: app._top_up_ingest_parse_pool imports template_runtime (16-21 ms, 75 modules, once), and the first Console agent turn with direct Library tools imports chunking_interop_library (~22 ms, 76 modules, once).

Health: the vendored engine strategies scale linearly. I measured them at 1x and 2x size (words, sentences, paragraphs, tokens, fixed_size, json, xml, code, propositions). The heavy costs sit in the glue around the engine. (1) Offset synthesis in the shim is O(N*S) and becomes cubic-ish on its fallback path. It dominates or entirely makes up ingest chunking time: 3.9 s for 513 KB with plaintext defaults, 26 s for 945 KB of JSON, 20 s for 616 KB with fixed_size. (2) The config shim rebuilds a ConfigParser on every call, and it is called per line and per chunk. (3) Semantic chunking calls sklearn per sentence pair. (4) The structure_aware range check is quadratic. (5) Lab execution double-scans the text per chunk. (6) Input sanitation loops per character in Python. (7) Lab previews pay a fresh ~0.5 s interpreter per candidate. (8) Lab sample typing reruns a ~28 ms whole-sample transition per keystroke with no coalescing.

Structural: Chunking/__init__ eagerly imports Chunk_Lib, and engine/strategies/__init__ imports every strategy. So importing any light submodule (_template_conversion, lab_models, interop for the constant AUTO_SENTINEL) loads ~70 modules including langdetect, which is imported only for a dead flag.

No open backlog task covers any of these. TASK-21102, TASK-163 and TASK-31645 are Done and address different scope. All timings were measured in an isolated scratch profile (HOME/XDG/TLDW_CONFIG_PATH in scratch, TLDW_TEST_MODE=1, PYTHONPATH=audit tree).

Suggested PR grouping:
- PR-A (shim offsets): F1 and F5. Use the engine's chunk_text_with_metadata offsets and make the fallback linear.
- PR-B (engine glue): F3, F6 and F4. Memoize the config parser, use a regex sanitizer, bisect the processed ranges.
- PR-C: F2, semantic vectorization.
- PR-D (Lab): F7, F8 and F9.
- PR-E (hygiene): F10 through F14.

## clean areas
- engine/strategies/{words,sentences,paragraphs,tokens,fixed_size,json_xml,code,code_ast,propositions}.py: scale linearly (measured at 1x/2x input, ratio 0.9-2.4x); the only per-chunk tax is the shared config read in F3
- engine/process_text/{pipeline,dispatch,metadata,options,preparation,models}.py: linear per-chunk metadata finalization, one md5 per chunk
- engine/auto_planner.py, engine/constants.py, error_policy.py, llm_context.py, option_utils.py, exceptions.py: pure, module-level regexes, cheap
- engine/templates.py TemplateProcessor ops (add_overlap/merge_small/filter_empty/format_chunks): linear; TemplateProcessor() construction is cheap
- engine/splitters/*: trivial
- auto_selection.resolve_auto: 0.55 ms per item measured (list_templates + validate 9 templates), fine even per ingest job on the UI thread
- chunking_interop_library.py CRUD: parameterized single-row queries; the full-scan helpers get_documents_using_template (LIKE) and get_template_statistics (json_extract GROUP BY) have zero production callers
- _template_conversion.py: deepcopy only during the one-time v6->v7 media DB migration
- lab_autosave.AutosaveWriter: good pattern (single-thread executor, 0.3 s debounce / 1 s max wait, all hashing and SQLite off-loop)
- lab_runner.LocalPreviewRunner supervision: asyncio.to_thread + selectors, bounded 60 s wall time, correct reaping; the only cost is the per-candidate cold interpreter (F7)
- lab_comparison.py / lab_recovery.py / lab_preflight.py: called via asyncio.to_thread from the Lab screen and results region; current_local_runtime is cheap
- Chunk_Lib module-scope get_internal_prompt: <0.1 ms
- engine Chunker LRU cache via the shim: never hits (fresh engine per improved_chunking_process), but the measured overhead is negligible (sha256 plus deepcopy of a str list)
- Dead but never imported, so zero runtime cost: engine/utils/metrics.py, engine/strategies/ebook_chapters_patch.py, engine/multilingual.py, _shims/prompt_loader.py alias
- Legacy token_chunker.py / language_chunkers.py: lazy per-instance tokenizer/Tagger construction, no production callers besides rolling_summarize's count_tokens
- Boot/first paint: no Chunking module in ui_ready_modules.txt / boot_import_modules.txt (only the stdlib-only tldw_chatbook.chunking_engine_version)
- Library re-chunk: rechunk_one_item is async but runs under asyncio.run inside a thread=True worker (library_rechunk_run.py) or the MCP to_thread bridge, so it does not block the loop

## census
| Measurement (isolated scratch profile) | Result |
|---|---|
| improved_chunking_process, words 400/200, 1.3 MB | 241 ms total, 127 ms in _synthesize_flat_offsets |
| paragraphs 500/200 (plaintext/web_article default), 513 KB | 3.9 s total, ~all in offset synthesis |
| sentences 500/200, 513 KB / sentences 1500/100 (document default), 513 KB | 0.94 s / 2.6 s, ~all in offset synthesis |
| fixed_size, 616 KB | 19.6 s (20.3 s synthesis) |
| json 945 KB / xml 425 KB | 26.5 s / 4.6 s (synthesis) |
| semantic, 5k / 10k sentences | 1.9 s / 4.0 s; per-pair sklearn ~1.5 s vs vectorized 1.2 ms |
| hierarchical_flat, 32k lines, with / without 3 boundary rules | 4.75 s / 0.39 s |
| load_comprehensive_config / _get_chunking_bool / safe_search per call | 19 us / 25 us / 31 us (raw re.search 0.08 us) |
| structure_aware at 2.5k / 5k / 10k elements | 64 / 219 / 887 ms |
| _execute_report find pair on a 1.9 MB Lab sample | 0.84 s (fixed_size) / 1.1 s (sentences) |
| _sanitize_input per-char loop vs regex, 2 MB | 64 ms vs 4.5 ms |
| Lab child interpreter import | ~0.5 s wall, 859 modules |
| lab_state.replace_sample with 1.9 MB sample | ~28 ms per keystroke; whole-session canonical_json 67 ms |
| ebook chapter detection fork vs thread (490 MB RSS) | +10-20 ms per call |
| rolling_summarize detail=0, 320 KB | 1.7 s CPU before a single LLM call |


# slice-16

## summary
Slice #16 (DB#1): ChaChaNotes_DB.py (24,424 lines), AgentRuns_DB.py (3,379), Chunking_Lab_DB.py (654). Every file was covered. I measured the hot paths with isolated micro-benchmarks on file-backed scratch DBs (HOME, XDG_* and TLDW_CONFIG_PATH redirected into the scratch dir; the app was never booted).

Hot paths in scope:
1. **Per-transaction cost.** `TransactionContextManager.__enter__` (ChaChaNotes, 184 `transaction()` sites) and AgentRuns `connection()`/`transaction()` (49 sites, reads included) both call `_core_operation`.
2. **Connection acquisition.** `_get_thread_connection` → `connect_private_sqlite`, combined with the repo-wide "to_thread then `close_connection()`" pattern.
3. **Conversation search filter.** Used by the Console session switcher History search, the Console workspace search and Library locate.
4. **Library note/conversation search.**
5. **Boot and launch sweeps.** Schema init (cheap: 0.1–0.4 ms), the messages_fts backfill probe, and the AgentRuns orphan reconcile.
6. **Message and character reads** on tables whose image BLOB sits in the middle of the row.

Overall health: the query-level code is mostly careful. It uses id-chunked batch reads, a no-N+1 tree read, lazy DEBUG logging, a cached column set, paged Library lists and an O(1) `append_steps`.

Two regressions dominate, and both landed after the 2026-09-04 perf review:
- **(a) Storage admission on every transaction.** Backup_Recovery storage admission, wired into ChaChaNotes and AgentRuns transactions on 2026-09-16 (b5251e9a6e, TASK-32628), makes every outermost transaction, and every AgentRuns read, cost about 4–6 ms instead of about 0.02 ms. That is roughly 245 file opens per transaction.
- **(b) Per-open helper process.** Every SQLite open now starts a helper subprocess (the ADR-125 design), about 50 ms each. The app closes and reopens connections after almost every worker call, so it pays this over and over.

Separately, TASK-278 (Done) turned the conversation search into a correlated FTS scan. It takes seconds at a few thousand conversations. Several earlier optimisations quietly do nothing:
- TASK-15474's "image-free" projections still read through every image, because the BLOB column comes before the columns the query reads.
- TASK-32804.10 removed the N+1 from the orphan reconcile, but it still scans and parses the whole history on every launch.

Structural notes:
- ChaChaNotes_DB is a god class (one class, about 440 methods). Its import self-time is only about 4 ms, so the cost is the 184 separate transaction sites, each paying admission, rather than import time.
- `:memory:` databases skip both the admission and the helper process. The test suite is mostly in-memory, so it cannot see findings F1, F3 or F4.
- AgentRuns_DB's module-scope import of agent_models pulls in the eager Chat package whenever the first repository is accessed.

## clean areas
- ChaChaNotes _initialize_schema steady state: when the DB is already current, the repair, CREATE INDEX IF NOT EXISTS and trigger checks cost 0.12 ms (measured). The migration chain runs only on upgrade.
- ChaChaNotes execute_query/execute_many logging: lazy opt(lazy=True) with preview_params. Metrics calls are no-ops when disabled (about 1 µs).
- _messages_table_columns: cached per instance (a PRAGMA once).
- Id-chunked batch readers (get_attachments_for_messages, get_generation_metadata_for_messages, get_message_versions_by_ids, get_message_images_by_ids, get_note_version_states, get_conversation_archive_states, get_conversations_metadata_by_ids, count_messages_for_conversations): single transaction, 500-id chunks, no N+1.
- get_message_tree_rows_for_conversation (TASK-22206): one indexed query per conversation (see F6 for the BLOB-layout caveat).
- list_library_notes_page / list_library_conversations_page / get_library_conversation_messages: paged, substr() previews, batched keyword fetch. The duplicate keyword query in _library_organization_for_notes is trivial.
- search_conversations_by_title / search_conversations_by_content / search_messages_by_content / character FTS paging: uncorrelated FTS joins with LIMIT.
- sync_log retention triggers: indexed on sync_log(entity, entity_id). messages_au is correctly column-scoped (TASK-21128).
- set_conversations_archived / replace_keywords_for_conversation / append_message_exchanges_local / upsert_trajectory_rows: per-row statements inside a single transaction with small N (not per-row commits).
- Flashcards/quizzes/kept briefings/artifact windows: cold paths, bounded tables.
- AgentRuns_DB: append_steps is a pure indexed INSERT (task-18601), metadata-only readers skip steps, _batch_hydrate_steps is chunked, _initialize_schema costs 0.36 ms on a current DB (measured).
- Chunking_Lab_DB: owned by a dedicated writer thread, one held connection, bounded payload caps. A GC transaction after each save plus synchronous=FULL means two fsync'd transactions per checkpoint; this is deliberate recovery durability on a cold tool (P3 at most, not filed).
- get_messages_for_conversations_batch: no callers outside the module (dead, not hot).

## census
| Measurement (scratch profile, file DB, macOS, Py3.12) | Real arm | Baseline arm | Note |
|---|---|---|---|
| Empty `with db.transaction()` (ChaChaNotes) | 5.67 ms | 0.024 ms (`_core_operation` stubbed) | 6 alternating A/B rounds; 09-04 review measured 18–23 µs |
| 1 turn = 2× add_message | 13.6 ms | 2.65 ms | same A/B |
| AgentRunsDB.get_run_metadata / append_steps | 3.84 / 4.08 ms | 0.015 / 0.058 ms | admission on read path too |
| get_connection()+SELECT 1 | 13.0 µs | raw sqlite 2.0 µs | 09-04: 1.94 µs |
| New connection on a worker thread | 49–54 ms | — | helper subprocess spawn per open |
| close_connection TRUNCATE checkpoint | 0.6 / 5.9 / 14.2 ms | — | WAL 0 / 4 / 16 MB |
| CharactersRAGDB() on current DB | 67–78 ms | schema init alone 0.12 ms | helper spawn + admission |
| AgentRunsDB() on existing DB | 69 ms | schema init alone 0.36 ms | Settings ▸ Agents compose path |
| search_conversations_page('python', scope all), 3000 convs / 45k msgs | 2,010 ms | — | correlated FTS EXISTS |
| COUNT with filter, 1000 convs, 'python' / 'w17' prefix | 760 ms / 10,296 ms | 3.4 ms / 42 ms uncorrelated IN(...) | same result counts |
| search_library_conversations_page, 15.5 MB messages | 95–102 ms | LIKE branch alone 37 ms | LIKE runs for count and page |
| list_character_cards(200) / count, 500 KB avatars | 17.9 / 17.4 ms | 0.89 / 0.14 ms (no avatars) | image column precedes read columns |
| Tree read, 20 × 2 MB images, image_data not selected | 5.7–6.3 ms | — | overflow-chain walk |
| backfill_messages_fts no-work boot probe, 45k msgs | 390 ms first / 11–12 ms warm | — | runs under BEGIN IMMEDIATE |
| reconcile_orphaned_runs, 2000 runs × 50 steps | 860 ms, +103 MB peak | — | every process launch |


# slice-17

## summary
Slice #17 (DB#2, 16 files, 31,033 lines): Client_Media_DB_v2, Subscriptions_DB, Prompts_DB, Evals_DB, base_db, RAG_Indexing_DB, automatic_work, Workspace_DB, VisualIdentity_DB, Library_Collections_DB, Library_Ingest_Jobs_DB, agent_worktrees, chachanotes_fts_backfill, Workflows_DB, canvas_payload_validation, and __init__. All benchmarks ran in an isolated scratch profile (HOME, XDG_* and TLDW_CONFIG_PATH under the scratchpad, TLDW_TEST_MODE=1, loguru sinks removed) using the venv interpreter.

The query code itself is mostly healthy. Library media pagination and FTS plans are pinned. Prompt search no longer builds an IN-list (TASK-32804.10 holds). RAG indexing uses executemany. Subscriptions reader search is FTS-first with a cached completeness check and chunked IN lookups. Readers on the Subscriptions, Watchlists and Personas screens run through asyncio.to_thread.

The dominant costs are in the wrapper layer that every store in this slice now sits behind. All of it arrived with the backup/recovery program on 2026-09-16 (b5251e9a6e), after the last holistic perf review (09-04), which is why no task covers it:

(1) Storage admission on every top-level transaction/connection. Each `@_core_transaction` block (Workspace, Media, Prompts, Library Collections, Library Ingest Jobs, Subscriptions, Evals, RAG Indexing, and also ChaChaNotes through ChaChaNotes_DB.py:24221 and AgentRuns) runs the full acquire_storage path. That path re-reads the recovery-bootstrap records and registry from disk: about 245 open(), 253 fstat, 70 lstat, 10 listdir and 7 flock per block. It costs 4.6–7.8 ms (measured; roughly 3.5–4.5 ms at a real ~/.config depth), against 0.5 µs for the query. The work sits inside a per-root 'initializing' section, so every DB operation in the process is serialized, capped near 200 ops/s. With 3 background DB threads, a UI-thread operation's median rises to 16.8 ms, with p95 42 ms and max 75 ms.

(2) base_db.run_owned_db_call and operation_owned_connection close the worker handle after each call. The next call re-opens it through connect_private_sqlite, and each open spawns a `python -I -S` helper subprocess: 54 ms per one-statement call. The Console character-context scope check makes 2 such calls on every 0.2 s transcript tick during active runs, measured at 137 ms wall and about 127 ms CPU per tick.

(3) base_db.register_semantic_mutation_guard installs a SQLite trace callback on every ChaChaNotes connection. CPython builds sqlite3_expanded_sql while holding the GIL, once per statement and again for each trigger program. A 1 MB BLOB insert costs 26 ms with no triggers and 131 ms with 2 triggers. A worker thread inserting a 3 MB BLOB stalls the event loop for 144 ms.

Smaller costs: _repository_types() rebuilt 3 times per get_connection (execute_query costs 10 µs against 0.5 µs raw); 3 uncached WorkspaceDB reads on the event loop per send; an N+1 in search_prompts; and per-item transactions for bulk trash/delete.

Cross-slice notes for dedupe: the root causes of F1 and F4 live in Backup_Recovery (slice 4, cold tier, so its auditor may not treat them as hot). The helper subprocess in F2 lives in DB/private_sqlite*.py (slice 18). ChaChaNotes itself belongs to another slice.

Structural note, not filed (already in the core review, report.md:2578): the held-connection template (_held_connection, connection, transaction, close) is copy-pasted across 8 stores in this slice. Each getter calls _core_access 2–3 times, so any fix to the per-call wrapper cost has to reach every copy unless the template moves into base_db.

## clean areas
- tldw_chatbook/DB/__init__.py: one trivial import (sqlite_datetime_fix adapters); no boot cost of note
- tldw_chatbook/DB/canvas_payload_validation.py: a pure deterministic SQLite function (utf-8 check + sha256) that runs only on Canvas revision writes
- tldw_chatbook/DB/Workflows_DB.py: one held connection plus an RLock; migrations only at construction; no hot caller
- tldw_chatbook/DB/agent_worktrees.py: small keyed queries on AgentRunsDB; runs only on agent worktree lifecycle events
- tldw_chatbook/DB/automatic_work.py: synchronous=FULL admission transactions are deliberate, only for unattended chains, and on agent worker threads; _snapshot scans only one chain's reservations
- tldw_chatbook/DB/RAG_Indexing_DB.py: mark_items_indexed batches with executemany in one transaction; used only by background ingestion indexing (pays F1 per block)
- tldw_chatbook/DB/chachanotes_fts_backfill.py: thread worker with paced, bounded, interruptible chunks (100 ms pause); never on the event loop
- tldw_chatbook/DB/Subscriptions_DB.py: reader search is FTS-first with a cached completeness check; get_subscription_items_by_ids chunks its IN lists; Watchlists screen readers run via asyncio.to_thread (N small reads per hop now each pay F1)
- tldw_chatbook/DB/Evals_DB.py: less frequent screen; the legacy EvaluationOrchestrator.store_result path is not wired to the UI; each connection() pays F1
- tldw_chatbook/DB/Library_Collections_DB.py: constructed once per app (app.py:9764/9811 cache it); idempotent schema DDL runs once; the docstring claim that single-statement reads 'cost nothing extra' is now false because of F1
- tldw_chatbook/DB/Client_Media_DB_v2.py search_media_db / list_library_media_page / get_paginated_media_list: count and page in one transaction, FTS-first CROSS JOIN count, plans pinned without sqlite_stat1 (verified-fine list); _persist_chunks does per-chunk INSERTs inside one transaction on ingest worker threads (minor)
- tldw_chatbook/DB/Client_Media_DB_v2.py process_chunks / get_all_active_media_for_embedding / get_all_content_from_database / get_unprocessed_media: no production callers (dead code, not on any hot path)
- tldw_chatbook/DB/Prompts_DB.py list_prompts / browse_prompts / list_library_prompts_page: bounded pages, count and page in one transaction, batch keyword fetch
- tldw_chatbook/DB/VisualIdentity_DB.py: Personas-screen reads go through to_thread; flagged loops are comprehensions (false positive)
- tldw_chatbook/DB/Library_Ingest_Jobs_DB.py: startup restore runs off-thread (app.py:2944); per-tick local parse progress does not persist (persist=False); state-transition upserts are UI-thread by design and pay F1

## census
| Probe (isolated scratch profile, macOS, warm, loguru sinks removed) | Result |
|---|---|
| raw sqlite3 `SELECT 1` | 0.48 µs |
| `WorkspaceDB._held_connection().execute(SELECT 1)` (_core_getter + _core_access only) | 7.7 µs |
| `MediaDatabase.execute_query(SELECT 1)` | 10.1 µs (3× `_repository_types()` per call) |
| `with WorkspaceDB.connection()` / `.transaction()` + SELECT 1 | 4.6 / 5.0 ms |
| `with MediaDatabase.transaction()` + SELECT 1 | 4.8 ms |
| `with PromptsDatabase.transaction()` + SELECT 1 | 5.4–5.9 ms |
| `MediaDatabase.list_library_media_page(20,0)` on an EMPTY DB | 5.0 ms |
| `LocalWorkspaceRegistryService.get_workspace` / `read_change_review_consent` | 5.7 / 5.1 ms (depth 10); 8.1 / 8.8 ms (depth 17); about 0.27 ms per extra path component |
| ChaChaNotes `get_character_conversation_search_revision` (1-row txn) | 7.8 ms |
| syscalls per top-level `WorkspaceDB.connection()` | 245 open, 253 fstat, 70 lstat, 36 stat, 10 listdir, 7 flock |
| UI-thread `WorkspaceDB.connection()` idle vs with 3 background DB threads | median 4.15 → 16.8 ms, p95 6.9 → 42.3 ms, max 22 → 75 ms; process total about 200 ops/s |
| DB constructor, reopening an existing file (Workspace/Media/Prompts/LibColl/Evals/Subs) | 42–51 ms each (includes the private-sqlite helper subprocess spawn) |
| `SubscriptionsDB(':memory:')` (per watchlist preview) | 5.2 ms |
| `run_owned_db_call(WorkspaceDB or CharactersRAGDB, 1 SELECT)` | 54 ms per call |
| same op via plain `asyncio.to_thread` with the connection held | 7.0 ms |
| character scope check (2× run_owned_db_call metadata) per Console sync tick | 137 ms wall, 72 ms child CPU + 55 ms self CPU; with held connections: 20 ms |
| trace-guarded insert, 20 KB / 200 KB text | 0.049 / 0.474 ms (guard off: 0.007 / 0.044) |
| trace-guarded insert, 1 MB BLOB, 0 / 2 triggers | 26.5 / 131 ms (guard off: 0.19 ms); 1 / 5 trace calls per insert, each receiving 2,000,039 chars |
| event-loop worst lateness while a worker thread inserts 3 MB BLOBs | guard off 0.2 ms, guard on 144 ms |
| `PromptsDatabase.search_prompts('dragon')`, 25/page, 300 prompts | 2.68 ms (N+1 keyword part 0.32 ms vs 0.17 ms batched) |


# slice-18

## summary
Slice #18 (DB#3, 17 files, 11,495 lines). The hot surfaces are (1) the private SQLite connection seam in DB/private_sqlite.py and private_sqlite_process.py, which every file-backed open in the app goes through (connect_private_sqlite has about 50 importing modules), and (2) the Console Character section's search and scope repository in DB/character_conversation_search.py. The recovery_* modules, sql_validation, sql_logging, transaction_observer, sqlite_datetime_fix and fts_backfill_pacing are cold or cheap.

Headline, measured: since commit 61a49de2e0 (2026-09-07, "preserve SQLite locks across private opens"), every POSIX file-backed open spawns a fresh `python -I -S` helper process, runs a prepare exchange, then does a close handshake and reaps the child.
- Open+close costs a median 42.8 ms end to end, against 0.096 ms for a raw sqlite3 open: about 450x.
- On macOS the fork_exec path (close_fds=True) holds the GIL, so every other thread stalls, including the Textual event loop: max 22–26 ms at a 600 MB heap and 45 ms at 1 GB, against a 1.4 ms idle baseline.
- No open perf task records this. The 08-29 and 09-04 perf reviews predate it. The design spec's own benchmark showed a repository open going from 6.4 ms to 142 ms, but no task records it. As a result TASK-24457 (8 opens per Library visit) and TASK-31501 (1 Hz maintenance) both understate their real cost.

Several patterns multiply this per-open cost on hot async paths:
- run_owned_db_call / operation_owned_connection close the thread-local handle after each call, and ChaChaNotes close runs PRAGMA wal_checkpoint(TRUNCATE).
- The Console character search runs a full DB pipeline on every Input.Changed, with no debounce and no cancellation. Each keystroke runs 5 thread hops, a BEGIN IMMEDIATE write, and full-transcript projection of each candidate.
- The launch-wake discovery does a private open synchronously on the event loop right after first paint.
- The v66 triggers bump a global revision on every message write in any conversation. That revision is the fingerprint the 5 Hz run-time transcript poll re-reads through 2 thread-hop DB reads per tick, so every persisted message forces a full character-section refresh.
- Keyword-index maintenance holds the ChaChaNotes write lock for O(corpus) work in one transaction, and rewrites source_revision on every document row.

Structural note: the fixes group naturally into three PRs.
- PR A (connection seam): stat-only fast path or posix_spawn, plus connection reuse / no TRUNCATE on close. The fast path needs an owner decision, because the 2026-09-07 spec rejected a persistent pool and a fast path that has no helper fallback.
- PR B (Console character search): debounce plus cancellation, remove the keystroke-path ensure, chunk maintenance, drop the full-transcript revalidation.
- PR C (revision and polling): gate the triggers to character conversations and take the scope refresh off the 5 Hz poll. Also move launch-wake off the loop.

## clean areas
- tldw_chatbook/DB/transaction_observer.py - process-local dict + RLock per managed transaction, bounded, O(1); fine
- tldw_chatbook/DB/sql_logging.py - lazy, bounded, BLOB-safe previews (the good pattern)
- tldw_chatbook/DB/sql_validation.py + sql_identifier_core.py - precompiled regex, set lookups, warnings only on failure; get_safe_order_by_clause re-validating a static profile is trivial
- tldw_chatbook/DB/sqlite_datetime_fix.py - converters measured at 0.15 us per DATETIME value (5000 rows x 2 cols: 4.57 vs 3.11 ms); negligible
- tldw_chatbook/DB/fts_backfill_pacing.py - correct chunk-pause/abort-slice template (should be reused by keyword maintenance, see F5)
- tldw_chatbook/DB/recovery_core.py, recovery_core_schema.py (271 KB), recovery_operations.py (324 KB) - backup/restore-only paths, NOT on the ui_ready import path (Tests/Performance/boot_budget_snapshots/ui_ready_modules.txt); recovery_sqlite.py (87 lines) is on boot path but trivial
- tldw_chatbook/DB/private_sqlite_protocol.py - bounded JSON framing, cheap
- tldw_chatbook/DB/private_sqlite_helper.py / private_sqlite_helper_entry.py - child-side logic is ~1 ms (in-process prepare_batch measured 1.03 ms); the cost is the spawn, covered by F1
- tldw_chatbook/DB/private_sqlite.py backup/copy/restore/profile-migration functions - cold user-initiated backup paths; _backup_pages(pages=1) measured 222 ms vs 230 ms for pages=-1 on a 103 MB DB, so step size is not a speed problem
- tldw_chatbook/DB/private_sqlite.py module import - 1.6 ms self, owner registry is a static MappingProxy; not a boot cost
- tldw_chatbook/DB/private_sqlite.py in-process _prepare_* duplicate (lines ~765-1246) - Windows-only on shipped platforms; a known structural dup (qa/core-code-review-2026-09-17/slices/DB-rest.md P2), not a POSIX speed cost
- tldw_chatbook/DB/character_conversation_search.py browse paths (recent_groups, page_for_character, unavailable_page, repair_candidates) - bounded LIMIT/keyset; _recent_resolved_summaries GROUP BY limited to assistant_kind='character' rows via idx_conversations_assistant_identity

## census
| Measurement (macOS arm64, Py 3.12 venv, isolated scratch env) | Result |
|---|---|
| HelperLease prepare+close on scratch DB, 20 runs | min 38.5 / med 44.6 / max 59.2 ms |
| Breakdown (15 runs) | prepare exchange incl. child startup 37.3 ms, close handshake+reap 9.0 ms |
| End-to-end connect_private_sqlite("db.base", scratch)+close, 12 runs | med 42.8 ms (first 206 ms incl. imports) |
| Raw sqlite3.connect+select+close | 0.096 ms |
| In-process private_sqlite_files.prepare_batch | 1.03 ms |
| Sibling-thread stall while a worker spawns (fork_exec, close_fds=True) | idle baseline max 1.4 ms; 300 MB heap p99 7.6 / max 12.7 ms; 600 MB p99 11-13.7 / max 22-26 ms; 1 GB p99 22 / max 45 ms |
| Same at 600 MB with close_fds=False (posix_spawn on macOS) | p99 1.4 / max 7.3 ms |
| SelectedBranchEligibilityProjector._project (in-memory, 600-char msgs) | 50 msgs 0.15 ms; 500 msgs 1.25 ms; 2000 msgs 5.74 ms |
| sqlite datetime converters | 0.15 us/value |
| sqlite backup 103 MB pages=1 / 256 / -1 | 222 / 189 / 230 ms |


# slice-19

## summary
Slice #19 Evals (53 files, 21.8k lines, cold tier). Entry points: (1) BOOT: app.py:746 imports Evals.eval_orchestrator at module scope, and TldwCli.__init__ (app.py:8287 -> 10175) constructs EvaluationOrchestrator eagerly before first paint. That pulls eval_orchestrator, eval_runner, task_loader, eval_errors, config_loader, configuration_validator, concurrency_manager and DB.Evals_DB into the boot graph: 5.3 ms of imports measured in the dev venv. (2) Evals screen: UI/Screens/evals_screen.py and UI/Evals/* call word_bench, character_probe and skill_eval storage and runners. The runs are coroutine workers (run_worker without thread=True), so all synchronous EvalsDB calls in them run on the event loop. (3) Home open-run counts (thread worker). (4) Backup_Recovery maintenance binding. No timers, threads or polling live in the slice. The one busy-poll (_maintenance_drain, 20 ms) only runs during maintenance with active legacy runs, and nothing can start those.

Measured headline costs:
- F1: the HF `datasets` import is try/except-guarded but not lazy, and it sits on the boot path. It costs nothing in the dev venv, but costs ≥0.44 s (pandas alone) on any install that has datasets. The nemo and all-tools extras pull it in through nemo-toolkit.
- F2: the eager orchestrator construction costs about 100–148 ms of pre-paint boot, for a cold feature.
- F3: word-bench cells (5.2 ms each) and character-probe conversations (300 rows = 1.75 s freeze) are persisted row by row on the event loop. The skill-eval sibling was already moved to asyncio.to_thread.
- F4: the run-group pivot copies every run's full snapshot into its config_overrides, then decodes every copy and json_extract-scans every stored cell. run_groups() costs 65 ms and preflight_for_bench 68 ms, each at least once per Evals click.
- F5: load_grid costs 98 ms against 29 ms for a lean read.
- F6 (cross-cutting, owned by Backup_Recovery): every EvalsDB method pays ~3–8 ms of storage admission (245 os.open calls) for 0.003 ms of SQL. The same @_core_transaction wrapper covers 16 repositories, so the Backup_Recovery/DB slice should confirm the wider scope.

Structural: about 7.5k lines of the legacy run stack (eval_runner, specialized_runners, dataset_loader/validator, base_runner, ui_integration, ab_testing, exporters) are dead. TASK-32904 already covers them and they are not re-reported here, except that eval_runner and task_loader still ride the boot import graph (F1). All probes ran against the audit tree with HOME/XDG/TLDW_CONFIG_PATH pointed into a scratch dir. The real ~/.local/share/tldw_cli/default_user/evals.db mtime is unchanged (Aug 14).

## clean areas
- tldw_chatbook/Evals/__init__.py -- PEP 562 lazy facade, correct
- tldw_chatbook/Evals/word_bench/capture_client.py -- one pooled httpx.AsyncClient per target per run, 120 s timeout, fully async; no loop blocking
- tldw_chatbook/Evals/word_bench/analysis.py -- pure math over top-k<=20; spread() is O(T^2*k) with tiny T; results_grid does not recompose on lens changes
- tldw_chatbook/Evals/word_bench/normalizer.py, word_bench/models.py -- module-scope regex, small per-cell work
- tldw_chatbook/Evals/character_probe/runner.py -- every provider call goes through asyncio.to_thread, semaphore-bounded gather; no DB access
- tldw_chatbook/Evals/character_probe/{prompt,tags,targets,probe_format,models,cards}.py -- pure or small; snapshot_cards issues one ChaCha query per card (small N, folded into F3)
- tldw_chatbook/Evals/skill_eval/{runner,judge,simulation}.py -- provider calls through asyncio.to_thread; persistence already offloaded in evals_screen._persist (tier-2 S18)
- tldw_chatbook/Evals/skill_eval/{scoring,static_analyzer,prompts,models}.py -- pure, module-scope regex
- tldw_chatbook/Evals/skill_eval/storage.py -- iter_artifacts pages at 100 with OFFSET (runs have ~66 artifacts, fine); list_skill_eval_benches has zero callers
- tldw_chatbook/Evals/eval_errors.py -- ErrorHandler history bounded at 100
- tldw_chatbook/Evals/concurrency_manager.py, steering.py, research_report_scorer.py -- trivial
- tldw_chatbook/Evals/recovery.py -- only function-local imports from Backup_Recovery; no module-scope work beyond constant tuples
- tldw_chatbook/Evals/eval_templates/* -- only consumer is the orphan Widgets/template_selector.py (function-local import); not on any live path
- Dead legacy run stack (eval_runner run path, specialized_runners, dataset_loader, dataset_validator, base_runner, ui_integration, ab_testing with eager scipy, exporters, orchestrator run_evaluation/quick_eval) -- unreachable, covered by TASK-32904; not re-reported except the boot-import leg in F1

## census
| Measurement (isolated scratch profile, audit tree 840ed2ca58) | Result |
|---|---|
| import tldw_chatbook.Evals.eval_orchestrator chain at boot (datasets absent) | 5.3 ms cumulative (-X importtime) |
| import pandas cold (lower bound for `import datasets`) | 439 ms |
| EvaluationOrchestrator(client_id="tldw_cli_app") standalone | 148 ms first (schema create), 96-106 ms warm file |
|  of which EvalsDB open (helper spawn + schema) / eval_config.yaml load | ~55 ms / ~9.5 ms |
| EvalsDB.get_task(missing id) vs raw sqlite same query | 7.84 ms vs 0.0027 ms (245 os.open per call) |
| word_bench save_cell (store_result) mean over 24,000 cells | 5.17 ms/cell |
| character_probe save_conversations, 300 conversations | 1,748 ms (5.83 ms/row) |
| Synthetic history: 20 run groups x 4 targets x 300 snippets (80 runs, 24k cells) | avg config_overrides 113 KB/run |
| Evals_DB.list_runs(limit=500) vs 5 needed columns | 36.1 ms vs 2.1 ms |
| run_group_cell_failure_counts() | 39.6 ms |
| EvalsViewModel.run_groups() / preflight_for_bench() | 65.0 ms / 68.0 ms |
| load_grid(one 300x4 group) vs lean 2-column read | 97.8 ms vs 29.4 ms |


# slice-2

## summary
Slice #2 (Agents#2, 18 files, 11,990 lines) was read in full. Almost none of this code runs on the Textual event loop. It runs on the agent worker thread, per-call daemon threads and to_thread workers. So the costs here show up as extra time-to-first-token and extra seconds per agent step, not as UI freezes. The only slice code that runs on the loop is BuiltinToolProvider construction in async _run_agent_reply (console_chat_controller.py:28199), measured at 15-45 µs, and the in-memory SessionTodoStore. The Console run-log UI consumers (availability probe, modal paging) all use thread workers.

Hot paths found:
(1) Per send / per run: build_first_request_schema_plan calls probe_initial_catalog. It renders and tokenizes every cumulative prefix of the tool catalog, so the cost is O(N²). Measured 61-329 ms cold and 10-103 ms warm for 60-100 tools, against 3-6.5 ms for one full-set measurement. It also runs again for every sub-agent run.
(2) Per agent step: every run-log record (1 + 2×tool calls per step) takes the Backup_Recovery storage admission twice and reopens the segment file. Measured about 12 ms per append, against 0.05 ms for the raw write.
(3) Per tool batch in project-bound runs: InstructionActivationLedger.prepare holds the shared ledger lock while it re-walks the target's directory chain and re-reads absent AGENTS.md candidates on every batch, with no memo. Locally that is 2.7-4.7 ms and 521-1221 lstat calls per batch. On SSH-remote bindings it is 4×depth executor ops per batch plus 4 ops per send (ping + 3 fs_read). Op counts were measured with a fake executor. Each op spawns an ssh client and a remote `python3 -I -c` process.
(4) When [hooks] are configured, PreToolUse hooks run serially, one call at a time.
(5) Per-tool-result post-processing does per-character and per-leaf Python work.

Structural notes for other slice owners:
(a) Backup_Recovery.storage_admission.acquire_storage costs about 6 ms per call (145 fd-verified opens and repeated bootstrap control-record reads). It has 88 call sites repo-wide, including DB/private_sqlite, so it is a cross-cutting tax beyond run_log.
(b) Importing tool_catalog cold pulls tldw_chatbook.Chat/__init__ → server_chat_conversation_service → runtime_policy.bootstrap → tls_trust → config (about 810 ms cumulative cold). That chain is already boot-resident via app.py and console_chat_controller, so it is not a unique cost of this slice. The Chat package owner should check whether Chat/__init__ needs to import eagerly.
(c) BuiltinToolProvider's instruction_root parameter is never passed anywhere, so path_targets always returns (). This is dead code, not a performance issue.

No finding here is already covered by an open backlog task. I grepped backlog/tasks for probe_initial_catalog, InstructionChainPayloadState, RemoteInstructionIO, resolve_targets and acquire_storage: nothing open matched. The only perf-adjacent hit was TASK-33009 (SSH remote workspace bindings, Done), and it does not cover this cost.

## clean areas
- tldw_chatbook/Agents/run_log_format.py: pure linear codec; iter_records is single-pass with anchor resync
- tldw_chatbook/Agents/run_log_paging.py: bounded pages (max_records/max_content/max_scan budgets), scandir per segment step; its only UI consumer (console_run_log_modal / Console_Modules/agent.py) loads pages in thread=True workers
- tldw_chatbook/Agents/run_log_search.py: load_records re-reads the whole log on every search_run_log/run_log_stats/run_log_slice call, but measured 7.9 ms load + 10.9 ms contains-search for a 12 MB / 1482-record log; outputs bounded; per-call re.compile is on the model-supplied pattern (can't hoist); P3 at most
- tldw_chatbook/Agents/session_todo_store.py: in-memory, hard-capped at 50 items; defensive dict copies are O(50)
- tldw_chatbook/Agents/run_context.py, run_tool_policy.py, tool_arg_coercion.py, tool_refusals.py: ContextVar lookups / O(1) cap counter / bounded recursive coercion / constant
- tldw_chatbook/Agents/raw_shell_tool_provider.py: per-invoke pydantic validation + stamp dict ops; progress sink is a passthrough; imported lazily (TYPE_CHECKING in controller)
- tldw_chatbook/Agents/run_webhooks.py: opt-in, bounded 32-slot queue, idle-retiring single thread, SSRF check; httpx.AsyncClient per delivery is acceptable at per-run-end frequency
- tldw_chatbook/Agents/run_log_eviction.py: off by default (run_log_evict_enabled=False); pure wrapper over console_history_budget
- tldw_chatbook/Agents/recovery.py: cold backup-discovery path only (Backup_Recovery destinations/restore_plan)
- tldw_chatbook/Agents/tool_catalog.py: BuiltinToolProvider() measured 15-45 µs, including the per-send construction on the loop at console_chat_controller.py:28199; lazy gateable-tool imports are one-time (note_management_tools 9.8 ms); the per-run ToolCatalogRegistry snapshot cache is correct; find()/list_catalog cheap; _coerce_arguments' per-call load_schema is a dict lookup for every provider checked
- tldw_chatbook/Agents/tool_catalog.py startup-import: its heavy deps (library_rag_tool_provider → Library RAG chain) are already boot-resident via app.py:230 and console_chat_controller; canvas/fleet/coercion imports are properly deferred
- tldw_chatbook/Agents/project_instruction_resolver.py local resolve_startup: all three controller call sites (console_chat_controller.py ~21807/21906/27917) run it via asyncio.to_thread; about 1 ms of lstat work
- tldw_chatbook/Agents/run_hooks.py: engine is only built when [hooks] are configured (console_runtime.py:3803); fire_async uses to_thread; config provider re-parses an in-memory dict (cheap); notify is bounded
- tldw_chatbook/Agents/virtual_cli_provider.py: construction + stamp handling are cheap; only result sanitization is flagged (F6)

## census
| Measurement (isolated profile, Python 3.12, audit tree 840ed2ca58) | Result |
|---|---|
| probe_initial_catalog, 60 tools, native JSON / fence protocol (cold estimate cache) | 61 ms / 147 ms |
| probe_initial_catalog, 100 tools, native / fence (cold) | 172 ms / 329 ms |
| probe_initial_catalog, 100 tools, native / fence (warm memo) | 23 ms / 103 ms |
| single catalog_schema_tokens of the full 100-tool set | 2.9 ms / 6.5 ms |
| RunLogWriter.append (2 KB record), startup admission held | median 12.1 ms, p90 16.8 ms |
| acquire_storage alone | median 6.5 ms (~145 native opens) |
| raw open+append+flush of same bytes | 0.05 ms |
| resolve_targets local depth-5, 0 / 3 exclusions | 2.65 ms, 521 lstat / 4.65 ms, 1221 lstat (every batch) |
| resolve_targets remote depth-3 (fake executor) | 12 ops (3 stat_path + 9 fs_read) per batch, identical on batch 2 and 3 |
| resolve_startup remote (fake executor) | 4 ops per send (ping + 3 fs_read) |
| InstructionChainPayloadState.capture, 60/200/600 msgs + 60 schemas | 0.77 / 1.13 / 2.24 ms per tool batch |
| wrap_review 5-call batch with trivial /usr/bin/true PreToolUse hook | 25.1 ms serial (single fire 4.7 ms) |
| _sanitize_result 1 MB git-diff-like output | ~50 ms (32 KB: 1.6 ms) |
| redact_root_locator, 5000-match dict, local / remote root | 7.7 ms / 9.1 ms |
| asyncio.run per call vs reused Runner | 0.42 ms vs 0.08 ms |
| BuiltinToolProvider() construction | 15-45 µs |


# slice-20

## summary
Slice #20 (Event_Handlers, 40 files, 16,907 lines). The runtime-hot code is in TTS_Events/tts_events.py (4,455 lines; Console Speak, auto-speak, hands-free per-sentence speech and spoken-feedback acks), STTS_Events/stts_events.py (2,991; Speech Playground, settings save, audiobook), Chat_Events/chat_rag_events.py (2,055; loaded at Chat first paint through UI/Console_Modules/retrieval.py, and runs per send when Library-RAG evidence is staged), chat_events_console_dictionaries.py (inspector attach/detach), chat_image_events.py (per attachment) and worker_handlers/ (every app-owned Worker.StateChanged). app.py imports tts_events, stts_events, worker_handlers and worker_events at module scope, so they sit on the boot path. The LLM_Management_Events package loads only when the Models/Lab screen is visited.

Overall health is better than most hot slices. Nearly all SQLite and file work is already moved off the loop with asyncio.to_thread or `@work(thread=True)`. The file-write paths in tts_events batch at 64 KB through one offload seam. The RAG scope reads are batched with json_each (no N+1) and cached per app.

The one real loop-blocker is the legacy TTS file-playback branch. It calls SimpleAudioPlayer.play() and stop() synchronously in an async handler. That means stop() of the previous clip, then time.sleep(0.1) on macOS afplay, then Popen (measured 11 ms median, 27 ms max from a 600 MB-heap process), so each spoken-feedback ack stalls the UI for more than 110 ms on macOS. The file's own D7 comments admit this and moved only 2 of the 5 call sites off the loop.

The second-largest cost is in app.py's TTS progress and complete handlers, which this slice's TTS events trigger. Each event does two full-DOM query() walks for ChatMessage and ChatMessageEnhanced widgets that nothing in the codebase creates any more (measured 1.86 ms per pair at 560 widgets, 3–4 pairs per utterance). The first TTS event also lazily imports chat_message_enhanced (measured 10–14 ms warm, 46 ms cold, 32 extra modules).

Structural notes:
- About 1,900 lines of this slice are dead in production: app_lifecycle, tab_events, eval_db_operations, ingest_status_helper, ingest_events, Chat_Events/chat_messages, dictation_integration_events, Media_Creation_Events, most of note_ingest_events, and two empty llamacpp/llamafile modules. Another ~700 lines of legacy RAG entry points in chat_rag_events have no production callers and ride the Chat first-paint import.
- app.py needs only Message classes from tts_events and stts_events for its @on decorators, yet imports the full handler modules. Moving the messages into small modules frees boot time and about 8–10 modules of UI-ready census headroom (TASK-23155, TASK-31816 and TASK-32644 report zero headroom).

Suggested PR grouping:
- (A) TTS playback off-loop: F1, plus the F7 dedupe.
- (B) Delete the dead TTS widget walks and the lazy import: F2.
- (C) Split the TTS/STTS message classes into light modules: F3.
- (D) Playground I/O offload and streaming: F4.
- (E) Dead-code removal: F6, together with the F8 and F9 hygiene items.
- (F) Staged-evidence scope reuse: F5.

## clean areas
- tldw_chatbook/Event_Handlers/LLM_Management_Events/server_lifecycle.py: run_server_subprocess keeps one thread per server with output sent to DEVNULL, so there is no per-line call_from_thread storm. stop_server_process uses asyncio.to_thread(terminate_process_bounded). Shutdown's stop_all_server_processes is called through to_thread in app.on_unmount.
- tldw_chatbook/Event_Handlers/LLM_Management_Events/llm_management_events.py, llm_management_events_ollama.py, llm_management_events_vllm.py, llm_management_events_onnx.py, llm_management_events_mlx_lm.py, llm_management_events_transformers.py: every launch runs as run_worker(thread=True), and every HTTP, model-scan (rglob) and socket probe runs through asyncio.to_thread or a worker thread. The module-scope huggingface_hub import measured about 2 ms (the package is lazy).
- tldw_chatbook/Event_Handlers/LLM_Management_Events/gguf_source_modes.py: pure data projection with no I/O.
- tldw_chatbook/Event_Handlers/Chat_Events/chat_rag_events.py scope resolution (resolve_scope_for_session, _existing_ids_sync, _current_*_evidence_ids_sync): SQLite reads are offloaded with to_thread (the sync path is taken only for in-memory test DBs), use single batched json_each queries (no N+1), and are cached per app by ScopeCache for display.
- tldw_chatbook/Event_Handlers/Chat_Events/chat_events_console_dictionaries.py: its sync DB readers are called only through asyncio.to_thread from ChatScreen workers. There is a bounded N+1 (load_chat_dictionary per attached id) on the modal-open path, which is negligible.
- tldw_chatbook/Event_Handlers/Chat_Events/chat_image_events.py: every production caller (Utils/file_handlers.py via attachment_core; the chat_screen clipboard path) runs it off the loop. See F9 for the asyncio.run-per-call pattern.
- tldw_chatbook/Event_Handlers/TTS_Events/tts_events.py generation path (_generate_tts, _stream_response_via_sink, _play_utterance_legacy_artifact, _run_owned_file_playback): artifact create, append and delete go through _run_blocking_tts_io (to_thread) in 64 KB batches. sink.open is offloaded. WAV collection is capped at 16 MiB. stop_live_sink is non-joining and loop-safe.
- tldw_chatbook/Event_Handlers/STTS_Events/stts_events.py settings save: the config write runs inside the service publication (off the loop). Whole-config deepcopy measured 0.64 ms per copy (about 5 per save), which is negligible. The ffmpeg conversion uses asyncio.create_subprocess_exec.
- tldw_chatbook/Event_Handlers/worker_events.py: thin adapter. Chat_Functions is already loaded at boot through Library RAG, so the marginal cost is 0.3 ms.
- tldw_chatbook/Event_Handlers/media_events.py, Audio_Events/*.py, ingest_utils.py: modules that only define Message classes or constants; no runtime cost.

## census
| file | lines | prod status | hot-path role | verdict |
|---|---|---|---|---|
| TTS_Events/tts_events.py | 4455 | live, boot import (app.py:403) | per utterance: Speak, auto-speak, hands-free, spoken feedback | F1 (P1), F2 producer, F3, F7 |
| STTS_Events/stts_events.py | 2991 | live, boot import (app.py:412) | Speech Playground, settings save | F3, F4 |
| Chat_Events/chat_rag_events.py | 2055 | live, Chat first paint (retrieval.py:31) | per send with staged evidence | F5; ~700 dead lines (F6) |
| LLM_Management_Events/* (9 files) | 3739 | live on Models/Lab visit | button handlers | clean (threaded) |
| note_ingest_events.py | 693 | only _import_template_files is live (backup, threaded) | none | dead remainder (F6) |
| Chat_Events/chat_messages.py | 445 | dead | none | F6 |
| Chat_Events/chat_image_events.py | 363 | live | per attachment | F9 |
| eval_db_operations.py | 336 | dead | none | F6 |
| media_events.py | 267 | live (messages only) | none | clean |
| Media_Creation_Events/ | 250 | dead | none | F6 |
| worker_handlers/ | 322 | live, boot | every app-owned Worker.StateChanged | F8 |
| chat_events_console_dictionaries.py | 163 | live (chat_screen) | inspector modal | clean |
| notes_events.py | 141 | live (lazy) | Library notes template list | F10 |
| app_lifecycle / tab_events / ingest_status_helper / ingest_events | 343 | dead | none | F6 |
| Audio_Events/dictation_integration_events.py | 106 | dead | none | F6 |
| Audio_Events (recording, dictation), ingest_utils, worker_events | 225 | live | none | clean |


# slice-21

## summary
Slice #21 LLM_Calls (26 files, 25,669 lines), audited at origin/dev 840ed2ca58.

Hot paths:
(1) `chat_api_call` calls a handler, which returns a stream. The Console gateway's `asyncio.to_thread` worker then iterates it one chunk at a time. This covers every hosted, local and ADR-179 engine provider.
(2) The Console gateway's async llama.cpp path (resolve, reachability probe, stream, complete) runs ON the event loop.
(3) Per-request setup: config snapshot, a new `requests.Session`, payload build.
(4) The Console setup-card local-server discovery, voice realtime connect, and the opt-in boot model-catalog refresh.

Main result: the dominant cost in this slice is the Backup/Recovery provider guard in `recovery_review.py`, added in TASK-32628 (b5251e9a6e). It wraps 47 callables in this slice plus 10 outside it. Every guarded call builds an `_Operation`: a storage-admission lease plus a witness-history read. Measured in an isolated scratch profile shaped like the user's real one (`recovery-bootstrap` holding only `admission` and `unbound-owner`), that costs about 6.5 ms, and a guarded call costs 8 ms in total. Every `_using()` then re-runs `check()`, which is about 76 `posix.open` calls and 1.3 to 3 ms.

The guard sets three costs:
- **Streamed chunks (F1).** The `_OpenAIStream` wrapper re-checks on EVERY streamed chunk, measured at 1.29 ms per SSE record. The real decode, validate and normalize pipeline is about 20 µs per record, so the guard is roughly 60x the work it guards. It caps throughput at about 775 chunks/s and burns about 1.3 s of CPU per 1000-chunk reply.
- **The event loop (F2).** The async wrappers do the same work synchronously on the Textual loop: about 21 to 27 ms per llama.cpp send, and 75 ms for an 8-probe discovery.
- **Misplaced decorators (F7).** Two decorators sit on helpers rather than on the provider handlers.

No open task tracks the guard's cost. The core review only saw `acquire_storage` behind config reads.

Second theme: `get_runtime_config_snapshot()` and `load_settings()` still cost about 15 ms each per call. They pay the admission handshake plus a deepcopy of the whole config, because TASK-32804.1's fastpath only covers `get_cli_setting` (measured at 1 µs). Every provider request calls one of them to read a single `api_settings` table (F3).

Third theme: every hosted request builds a new `requests.Session`. There is no keep-alive, so each send and each agent tool step pays a new TCP and TLS handshake (F4). Fixing this interacts with ADR-062's rule that the stream owns its response and session.

Structure: `LLM_API_Calls.py` (4,704 lines) and the two summarization libraries (5,100 lines) are god modules with a copy of the transport per provider. Import cost is modest: `LLM_API_Calls` adds 12 ms on top of config, with no heavy third-party imports. The summarization libraries (74 ms, pulling in langdetect and textual) are imported lazily. The slice is otherwise healthy: the SSE decode, validation, pricing and payload-copy costs are all measured in microseconds to low milliseconds.

The realtime outbound audio queue is an unbounded `asyncio.Queue`. That is minor memory growth, bounded by session length, and was not filed.

Suggested PR grouping:
- **PR-A:** F1 + F2 + F7, the guard memo, an off-loop admission step, and the misplaced decorators. This is the biggest win and needs sign-off from the recovery owner.
- **PR-B:** F3, a cheap config-section read (extends TASK-32804.1).
- **PR-C:** F4, a pooled session with an ADR-062 amendment.
- **PR-D:** the P3s (F5, F6, F8, F9, F10).

## clean areas
- tldw_chatbook/LLM_Calls/hosted_chat_streaming.py: SSERecordDecoder costs about 10 µs per event and 37 ns per byte (measured). The per-character Python loop is acceptable at streaming rates; only the chunk_size choice is flagged (F9).
- tldw_chatbook/LLM_Calls/hosted_chat.py: the HostedChatStream strict JSON parse, validation and filtering cost about 20 µs per event end to end (measured, 1000-event synthetic stream). owned_json_post's deepcopy of the payload takes under 1 ms for a 1 MB, 300-message, 40-tool payload (measured).
- tldw_chatbook/LLM_Calls/hosted_chat.py: ProviderPayloadValidators.normalize_call_batch re-validates every historical tool call per send. It measured 1.7 ms per send at 100 tool calls with 6 KB arguments each, so it is not worth changing.
- tldw_chatbook/LLM_Calls/pricing_catalog.py: a lazy process singleton. get_pricing measured 0.2 to 2 µs, including the unknown-model models.dev gap-fill; the catalog builds in 1 ms.
- tldw_chatbook/LLM_Calls/anthropic_subscription.py: UI readiness is off-loop (background thread with a TTL memo, no join on the loop). home_screen calls it with background=False only through asyncio.to_thread. Only the 5 s TTL respawn is flagged (F5).
- tldw_chatbook/Chat/Chat_Functions.py: engine handler registration (_LazyHostedHandler) is lazy, so hosted_provider_engine is imported on first use.
- tldw_chatbook/LLM_Calls/Summarization_General_Lib.py and Local_Summarization_Lib.py: never imported at module scope by UI code (every importer is function-local), so the 74 ms import cost (langdetect, textual, platformdirs) is not on the boot path. The str += accumulation in their stream loops relies on CPython's in-place concatenation and is linear.
- tldw_chatbook/LLM_Calls/LLM_API_Calls.py: import costs about 12 ms incremental over config, with 19 of the app's own modules and no heavy third-party packages (measured). Metrics calls go to loguru, and time.sleep retry backoffs run only on worker threads, capped at 60 s.
- tldw_chatbook/LLM_Calls/qwencloud.py and qwencloud_streaming.py: the per-event canonical digest and shape checks are bounded by _MAX_TRACKED_SEQUENCES. Payload deepcopies are small. Only the per-request session is flagged (F4).
- tldw_chatbook/LLM_Calls/qwencloud_url.py, realtime/protocol.py, realtime/__init__.py: realtime/__init__ is lazy, and websockets is resolved via require_dependency at connect time. Per-frame base64 and JSON handling in realtime/openai_session.py is inherent to the protocol.
- tldw_chatbook/LLM_Calls/groq.py, deepseek.py, mistral.py, openrouter.py: thin adapters over hosted_chat_request plus LegacyLineStream. Their only costs are the shared ones in F1, F3, F4 and F6.
- tldw_chatbook/LLM_Calls/moonshot.py and zai.py: per-send resolution and payload normalization are cheap apart from F1, F3 and F6. Readiness passes app_config explicitly, so it does not take the snapshot.

## census
| Guard site class | Count | Where it runs | Measured cost |
|---|---|---|---|
| `@_provider_recovery.unqualified` in LLM_Calls | 47 (+1 `openai_call`) | Worker thread (sync handlers) and loop (realtime connect, 2 sites) | 8.1 ms per outer call; nested calls add 2 × `check()` (1.3–3 ms each) |
| `@unqualified` outside the slice | 8 (gateway 5, local_server_discovery 2, catalog 1) + `catalog_call` 1 + `discovery_call` 1 | ON the event loop (12 async decorated sites in total) | 12.6 ms for the resolve shape, 8.5 ms for stream entry, 75 ms for 8 gathered probes |
| `_OpenAIStream` per yielded chunk | Every streamed reply from every chat_api_call handler | Gateway worker thread | 1.29 ms per record (303 `_history` calls for 302 records), about 76 `posix.open` per check |
| `get_runtime_config_snapshot()` / `load_settings()` per request | 14 handler sites; vLLM and custom_openai summarizers call it 3× per call | Worker thread | 14.9 ms / 15.3 ms (`get_cli_setting` = 0.001 ms) |
| New `requests.Session` per request | 13 sites | Worker thread | New TCP+TLS handshake per send (network cost estimated) |


# slice-22

## summary
Slice #22 (Library#1, 37 files, ~31.9k lines) is mostly pure state/projection modules plus the Collections-capture persistence stack, the ingest job registry, the Library keyword/RAG search service and the ingest preflight/capabilities helpers. I traced the hot paths through their UI and app callers, which sit outside the slice.

1. **Library ▸ Import registry listener (per job transition). This is the dominant problem.** Every registry mutation synchronously rebuilds the whole ingest canvas state twice and deep-copies the whole job queue three times, then recomposes an unwindowed queue panel. A folder import is therefore O(N²) on the UI thread. Measured with the real registry and state builder: the synchronous submit loop takes 333 ms at 100 files, 2.8 s at 300 and 33 s at the 1000-file scan limit. That excludes widget recompose and persistence. TASK-32804.5 marks its AC#1 as met ("1,000-file folder import does not block the UI"), but its benchmark only modelled the research listener, not this one. The root cause is the registry's copy-on-read `jobs()`: 16.5 ms per call at 1000 jobs, 72% of it in `copy.deepcopy`. Five other callers pay it too, including a second registry listener (Parakeet), the writer's claim-per-job, `check_action` and the single-job lookup.
2. **Boot.** Collections-capture wiring runs about 40 ms of sync SQLite and filesystem work on the event loop, from a `set_timer` 0.1 s after mount (measured). `app.py` also imports `library_local_rag_search_service` at module scope: 26.8 ms cumulative, mostly `Chat_Functions`, which is first imported here.
3. **Per-interaction.**
   - Library keyword search runs the notes and prompts seams' sync SQLite on the loop. The media and conversations seams already use threads.
   - A folder submit walks the folder again on the UI thread with 2–3 stat calls per entry.
   - Note-import review re-sorts and re-paginates the whole plan on every rename keystroke: 14 ms at 5000 items.
4. **Off-loop but wasteful.**
   - The Collections browse page selects `item.*`, which pulls full article text and HTML for 20 rows only to discard it.
   - The Export preview sums `LENGTH(CAST(content AS BLOB))` over the whole media corpus. Measured: 54 ms against 3.7 ms with `octet_length`, warm, on 400 MB.

Structural notes:
- `Library/__init__` now costs 5.1 ms and 10 modules after config (measured), so TASK-22503's package-level cost is largely mitigated. The expensive leaves are the ones `app.py` imports directly.
- `build_prompts_list_state` has no production callers; only tests use it.
- The good patterns found were all confirmed: `ExportProgressThrottle`, the memoised `find_spec` probe, the capture service's `asyncio.to_thread` `_call`, the threaded preflight, and the capped Media page (20 rows) and Notes list (100 rows).

Suggested PR grouping:
- **PR-A (Import canvas and registry: F1, F2, F9).** Copy-on-write registry with a cached visible tuple and a dict index; a coalesced listener that reuses the state it just built; a windowed queue; batched delete.
- **PR-B (off-loop Library I/O: F3, F4, F5).** Thread the capture wiring, reuse the preflight file list with `os.scandir`, and thread the notes/prompts seams.
- **PR-C (query shape: F6, F7, F13).** Explicit summary columns, `octet_length`, and `COUNT` queries instead of fetching id lists.
- **PR-D (hygiene: F8, F10, F11, F12).** Memoise `_page`, lazy-import `chat_api_call`, memoise the provider gate, and narrow the lock.

## clean areas
- tldw_chatbook/Library/library_media_state.py: page-bounded (LIBRARY_MEDIA_BROWSE_PAGE_SIZE=20); build_library_media_browse_state and validate/freeze are O(page) per build; the legacy build_library_media_state is only called with () by the controller
- tldw_chatbook/Library/library_media_viewer_state.py: markdown sniff capped at MAX_MARKDOWN_SNIFF_CHARS=32000 and MAX_MARKDOWN_SNIFF_LINES=200 (task-2858), compiled regexes at module scope
- tldw_chatbook/Library/library_media_reader_state.py: pure reducers, no I/O
- tldw_chatbook/Library/library_notes_state.py: list capped by LIBRARY_SOURCE_PAGE_SIZES['notes']=100; validate_database_note_draft runs per save only
- tldw_chatbook/Library/library_notes_session.py: mutate() is an O(1)-ish string-equality check per keystroke; saves go through an asyncio.shield-coalesced driver
- tldw_chatbook/Library/library_notes_tree_state.py + library_notes_tree_paging.py: projections are slice-bounded; row()/reconcile linear scans run over the visible rows only
- tldw_chatbook/Library/library_prompts_state.py: build_prompt_editor_state (reached per editor keystroke via _library_prompt_text_fields_match_state) measured at 25 us median; browse pages bounded; build_prompts_list_state is dead code (test-only callers)
- tldw_chatbook/Library/library_conversations_state.py: page-bounded
- tldw_chatbook/Library/library_conversation_reader_state.py: find runs on Input.Submitted only (library_conversation_reader_controller.py:882); page settles are O(page)
- tldw_chatbook/Library/collections_capture_service.py: every repository call is offloaded via asyncio.to_thread(run_finite_local_worker) (_call:211-218); extraction heartbeat every 30 s; tasks tracked and cancellable
- tldw_chatbook/Library/collections_offline_store.py: all calls threaded through the capture service; copies are bounded at 50 MB
- tldw_chatbook/Library/collections_legacy_recovery.py: bounded pages; batched streaming export
- tldw_chatbook/Library/collections_capture_repository.py (except list_page's item.*): tag fetch is batched with IN (not N+1); FTS used for search; indexed sort paths
- tldw_chatbook/Library/library_collections_service.py: LIMIT-bounded queries; LIKE search only over small user-collection tables on the MCP/tool cold path
- tldw_chatbook/Library/ingest_capabilities.py: find_spec probe memoised in _INSTALLED_PROBE_CACHE; generic_option_default is a schema lookup
- tldw_chatbook/Library/ingest_preflight.py analyze_path: runs in @work(thread=True) (library_screen.py:28661-28672); URL probe has a 5 s timeout
- tldw_chatbook/Library/ingest_analysis.py: pure resolution; per-tick use noted in F1
- tldw_chatbook/Library/export_progress.py: ExportProgressThrottle is the good pattern (0.1 s min interval)
- tldw_chatbook/Library/library_export_state.py: pure form state
- tldw_chatbook/Library/library_fts_query.py, library_pager_state.py, library_expand_policy.py, library_content_evidence.py, ingest_types.py, collections_capture_models.py: pure, bounded helpers
- tldw_chatbook/Library/library_artifacts_catalog.py + library_artifacts_state.py: merge/sort bounded by ARTIFACT_PAGE_SIZE windows
- tldw_chatbook/Library/library_notes_lasting_sync_state.py: O(plan) review rebuild per selection; acceptable for its frequency
- tldw_chatbook/Library/__init__.py: PEP 562 lazy capture exports; measured 5.1 ms / 10 new modules after config (TASK-22503's package-level cost largely mitigated)
- tldw_chatbook/Library/library_rag_answer_service.py: answer generation offloaded; provider gate noted in F11

## census



# slice-23

## summary
Slice 23 (Library#2, 20 files, ~13.3k lines) is mostly pure display-state builders and agent-tool services. Their own CPU cost is small: LibraryRagPanelState.from_values takes 86 us with 20 rows, build_library_shell_state and the rail, review-set and skills state helpers take microseconds, and the server-ingest poller is gated and stops on its own.

The real costs come from impure seams these modules sit on, and they surface on per-keystroke paths.

**Hot paths traced**
- **Library Search/RAG query box.** Every Input.Changed rebuilds the panel state, and that call includes a synchronous provider-readiness gate that costs about 10 ms (F1).
- **Media Reader `]`/`[` with an active review set.** Each key press opens 1 read transaction and 3 write transactions, re-reads the whole set, and invalidates the TASK-32804.4 memo, so the next render loads the set again (F3).
- **Server-mode Collections reader.** Every capture operation first makes a docs-info HTTP round trip (F4).
- **First visit to Library or Settings.** Both pull in the whole chunking engine only to use a threading slot guard (F5).

**Root cause (F2).** Every explicit transaction on a Backup_Recovery-registered core DB pays a storage-admission handshake of about 245 `open()` syscalls, because `participants.py` `_core_operation` re-verifies directories per transaction. Measured costs for an empty transaction:

| DB | Cost |
|---|---|
| LibraryCollectionsDB | 3.7-5.6 ms |
| MediaDatabase | 4.8 ms |
| CharactersRAGDB | 4.8 ms |
| PromptsDatabase | 4.7 ms |

`execute_query` autocommit reads cost 0.01 ms. The cost scales at about 0.34 ms per path component, so a default `~/.local/share` profile is estimated at about 2-2.5 ms. `load_settings()` shows the same thing at 8.85 ms warm: TASK-32804.1 only fast-pathed `get_cli_setting`. F1 and F3 are largely this tax repeated on the event loop. TASK-31502's measured 23 us per transaction predates the admission layer. The DB-chacha review folded `_core_operation` into TASK-31502 without measuring it.

**Known items not re-filed**
- TASK-22503: Library/__init__ eagerly imports local_library_tool_service and library_tool_contract at boot, about 5.9 ms cumulative.
- TASK-31508: list_review_sets issues 2N+1 statements. Re-measured at 18 ms for 20 sets x 500 items; it runs off the loop.

**Minor (P3), agent worker threads only, but GIL-bound**
- The chunk-span scan is O(nodes x chunks).
- Message-page fitting re-serializes the payload quadratically.
- `asyncio.run` builds a new event loop for every prompts/skills backend call.

All probes ran with an isolated HOME and XDG dirs, TLDW_CONFIG_PATH and TLDW_TEST_MODE=1 under the scratch dir, with temporary DBs. No user profile, keychain or network was touched.

## clean areas
- tldw_chatbook/Library/library_shell_state.py: pure, builds about 20 frozen dataclasses per call (microseconds). Boot self-time is 2.3 ms, which is ordinary dataclass creation.
- tldw_chatbook/Library/library_rail_state.py: pure preference and lifecycle coercion
- tldw_chatbook/Library/row_selection.py: pure set operations. The frozenset copy from .ids is only passed once per action, never tested in a loop.
- tldw_chatbook/Library/library_structural_wait.py: pure. The screen arms one-shot set_timer calls only, stopped on end.
- tldw_chatbook/Library/review_set_state.py: pure, O(n log n) over at most 500 items
- tldw_chatbook/Library/library_rag_score_kinds.py: pure math
- tldw_chatbook/Library/library_skills_state.py: pure. The fingerprint (json plus sha256) costs microseconds per page request. yaml runs only on editor open and save.
- tldw_chatbook/Library/library_rag_state.py builders: from_values takes 86 us with 20 rows. from_result takes 80 us per row, once per outcome. display_snippet takes 33 us per row per results rebuild. library_rag_profile_top_k takes 2 us. Module regexes are precompiled.
- tldw_chatbook/Library/library_rag_service.py: async seam. The real backend (library_local_rag_search_service) offloads with asyncio.to_thread.
- tldw_chatbook/Library/server_ingest_reconcile.py and server_ingest_status.py: the poller in app.py is async, stops when nothing is pending, and update_progress skips unchanged progress before persisting.
- tldw_chatbook/Library/web_clip_request.py: pure kwargs building
- tldw_chatbook/Library/meeting_speaker_rename.py: can_rename is memoized per id (0.02 ms). The legend parse is memoized per detail arrival (2.7 ms for a 1500-segment transcript on a miss). Rename runs on a thread worker.
- tldw_chatbook/Library/library_rechunk_service.py runtime: the batch runs in run_worker(thread=True) and chunk replacement uses executemany in one transaction. Only its import shape is a finding (F5).
- tldw_chatbook/Library/library_tool_contract.py: descriptor table is built once at import (0.75 ms). fit_page_payload deepcopy plus sizing takes 1.6 ms for 100 items on the agent worker.
- tldw_chatbook/Library/local_library_tool_service.py list/search: bounded pages with no N+1. It runs on the agent worker thread or MCP to_thread, never on the UI loop.
- tldw_chatbook/Library/server_collections_capture_service.py mapping code: pure, apart from the docs-info round trip (F4).

## census
| probe (isolated scratch HOME, ~10 path components) | measured |
|---|---|
| empty LibraryCollectionsDB.read_transaction | 3.7-5.6 ms (~245 posix.open per transaction); 6.1 ms at 17 components |
| empty MediaDatabase / CharactersRAGDB / PromptsDatabase transaction() | 4.8 / 4.8 / 4.7 ms |
| MediaDatabase / CharactersRAGDB execute_query SELECT 1 (autocommit) | 0.01 ms |
| config.load_settings() warm | 8.85 ms (get_cli_setting warm: 0.001 ms) |
| library_rag_answer_provider_gate() (default config) | 10.0 ms per call |
| LibraryRagPanelState.from_values (20 rows) | 0.086 ms |
| review-set walk step (get_active + mark + cursor + refresh), 50 / 500 items | 13.8 / 19.8 ms |
| get_active_review_set, 50 / 500 items | 3.2 / 4.3 ms |
| incremental import of library_rechunk_service (config/Chat/MediaDB already loaded) | 25.6 ms |
| _chunk_span_for: 500 nodes x 5000 chunks / last 200-node page | 66.8 / 36.7 ms |
| _fit_message_page 50 x 8000 chars | 27.9 ms |


# slice-24

## summary
Slice #24 Local_Ingestion (18 files, ~20k lines), audited at 840ed2ca58. None of this slice's code blocks the Textual event loop.

Four ways into the slice:
1. **Boot.** app.py imports local_file_ingestion, ingest_parse_progress, ingest_parse_worker and stt_batch_routing at module scope. The Library and LLM screens and the first-run wizard also import parakeet_v2_artifact at module scope. Measured total is about 9 ms (local_file_ingestion 5.2 ms, parakeet_v2_artifact 3.1 ms), because the heavy per-format processors are already deferred through the `_ensure_*` placeholders and the PEP 562 `__init__`.
2. **Pure helpers called from the UI or the queue.** classify_ingest_source, detect_file_type, is_http_url, canonicalize_url, resolve_batch_stt_route and parakeet_reference are pure and take microseconds.
3. **Library ingest.** run_parse_job runs parse_local_file_for_ingest inside a spawn Pool worker. persist_parsed_media runs on the writer thread. Progress travels through a bounded queue (64) and a 0.25 s coalescer. This is a sound design.
4. **Voice, attachments and collections capture.**
   - Voice: TranscriptionService is imported and constructed on every Console Mic press, via asyncio.to_thread → app._create_console_dictation_service. Each captured segment then runs transcribe_buffer on the dictation processing thread.
   - Console attachments: Utils/file_handlers runs parse_local_file_for_ingest in the UI process via asyncio.to_thread.
   - Collections capture: extract_article_for_ingest also runs through to_thread.

The real costs are latency and CPU off the loop:
- **Model reloads (P1).** All model caches are per instance: `_LegacyTranscriptionBackend._model_cache`, `_parakeet_mlx_model`, lightning/qwen/nemo, and the diarization embedding model. The Console builds a fresh TranscriptionService on every Mic press, and the Library faster-whisper path builds one per file. So the speech model reloads on every press and on every audio/video job. This is documented in console_voice_input.py but no task covers it.
- **Heavy module import (P2).** transcription_service eagerly imports a dead scipy.io.wavfile, plus faster_whisper, soundfile, requests and nemo. Measured cost is 157–196 ms warm and 474 ms cold. Blocking scipy alone brings it to about 100–120 ms; blocking scipy and faster_whisper brings it to 31–40 ms.
- **Redundant work on the ingest path:**
  - double ffmpeg transcode through a 192k MP3 (measured about 10x slower than one pass);
  - three yt-dlp extract_info calls per video URL;
  - chardet run over the whole file (measured about 1.2 s/MB);
  - speaker-count estimation using 9 spectral fits plus silhouette (measured 4.6 s at n=2400, versus 0.18 s for one fit);
  - docling chosen by "auto" for plain .docx (measured 3.7–6.2 s import per worker).

Structural notes:
- transcription_service.py is a 4.5k-line god module whose import pulls in every provider's stack.
- The `_store_in_database` paths in audio_processing and video_processing are unreachable in production (media_db is always None).
- `_find_ffmpeg` is implemented three times.
- The per-file TranscriptionService pattern is the single root cause of the reload findings.

No DB N+1 was found. The one persist-seam UPDATE uses idx_unvectorized_media_chunks_media_id, and the second transaction only runs when a template is used.

## clean areas
- tldw_chatbook/Local_Ingestion/__init__.py: PEP 562 lazy re-exports; this is the template pattern
- tldw_chatbook/Local_Ingestion/analysis_gate.py: pure predicate
- tldw_chatbook/Local_Ingestion/ingest_parse_progress.py: bounded queue (64), non-blocking put_nowait, per-job coalescer flushed every 0.25 s, work is O(jobs)
- tldw_chatbook/Local_Ingestion/ingest_parse_worker.py: module scope imports stdlib only; the error-chain walk is capped at 3 and guarded against cycles; the spawn pool is the correct off-process design
- tldw_chatbook/Local_Ingestion/stt_batch_routing.py: pure routing, 0.4 ms import
- tldw_chatbook/Local_Ingestion/parakeet_v2_artifact.py: 3.1 ms import on the Library/LLM screen paths; UI-called helpers (parakeet_reference, descriptor builders) are pure; acquisition and httpx are imported only inside provisioning functions; curated_registry caches the descriptors once per process
- tldw_chatbook/Local_Ingestion/local_file_ingestion.py boot surface: 5.2 ms import with the lazy _ensure_* processor loaders; classify_ingest_source, detect_file_type, is_http_url and canonicalize_url are pure and reached from Library preflight, web-clip and server-request code in microseconds; read_ingest_file_bytes is bounded by fstat before a capped read; persist_parsed_media issues one add_media_with_keywords plus an optional single UPDATE transaction on the writer thread, and the chunk UPDATE is indexed on media_id
- tldw_chatbook/Local_Ingestion/web_article_ingestion.py fetch: streamed, size-capped, 30 s timeout, egress-guarded; it runs in the parse worker or via to_thread (collections capture)
- tldw_chatbook/Local_Ingestion/PDF_Processing_Lib.py: OCR, docling and analysis imports are lazy; the gc.collect and sleep cleanup only runs for bytes input, which the Library path never uses
- tldw_chatbook/Local_Ingestion/Book_Ingestion_Lib.py: EPUB archive limits enforced before parse; per-item BeautifulSoup parsing is linear (html.parser is slower than lxml, which is a minor issue); only in the worker
- tldw_chatbook/Local_Ingestion/OCR_Backends.py availability flags: find_spec probes; heavy backends are initialized lazily (the one exception, tesseract, is listed in the findings)
- tldw_chatbook/Local_Ingestion/transcription_service.py faster-whisper file loop: progress callbacks throttled to at least 1% or 5 s; torch/transformers already lazy via _ensure_torch_import
- Metrics calls (log_counter/log_histogram with file-path labels) across the slice: metrics are disabled by default and return early; no in-memory accumulation

## census
| Import (isolated env, `python -X importtime` or perf_counter; config pre-imported) | Cumulative cost | Where it is paid |
|---|---|---|
| boot-path Local_Ingestion modules (local_file_ingestion + ingest_parse_worker + stt_batch_routing + parakeet_v2_artifact) | ~9 ms | app boot (fine) |
| transcription_service (all deps) | 157-196 ms warm / 474 ms cold | first Console Mic press per process; every spawned audio/video parse worker |
| transcription_service with scipy blocked | 99-120 ms | (dead `from scipy.io import wavfile` = ~60-80 ms warm, 213 ms cold) |
| transcription_service with scipy+faster_whisper blocked | 31-40 ms | target after the lazy-import fix |
| diarization_service | 2144 ms (torch 770 ms + sklearn 1369 ms) | "availability probes" really import |
| docling (import + construct DocumentConverter) | 3678-6231 ms + 12 ms | Document "auto" when docling is installed |
| PDF_Processing_Lib | 335 ms (pymupdf4llm 255 ms) | needed (default parser) |
| Document_Processing_Lib | 203 ms (pptx 92 ms + openpyxl 59 ms are never used by the Library path) | worker |
| audio_processing | 196 ms (Chat_Functions 103 ms) | worker |
| cv2 (Image_Processing_Lib) | 40 ms warm / 391 ms cold | worker, used only by extract_features, which Library forces off |


# slice-25

## summary
Slice #25 MCP (35 files, 27,020 lines, all under tldw_chatbook/MCP). The package matters for speed mostly through the Console, not the MCP Hub. Every Console send and every agent MCP tool call resolve MCP policy and catalog state synchronously on the Textual loop. Each MCP JSON store read also pays the ADR-126 storage-admission handshake: about 1,150 open() calls and 11-23 ms per read, measured on a warm process with startup admission held. This is the same shape TASK-32804.1 fixed for config reads, but none of the MCP stores (permission, local, context, target, execution log) got a warm cache.

Hot paths found and measured:
(1) Send path. `capture_mcp_definition_maximum` runs synchronously in the ui_submit/launch_chain path: about 78 ms. `_compose_mcp_provider`/`compose_catalog` then runs on the main loop in `_run_agent_reply`: about 131 ms. That is about 210 ms of loop stall per default agent send, with only the 32 built-in tools and no external servers. Across the send the permission store is read about 7 times, the local store 3 times, and server.py is parsed with `ast.parse` 6 times.
(2) Tool-call path. `execute_hub_tool`, submitted to the main loop per agent MCP tool call, costs about 130 ms: nested admission guards 40 ms, N+1 governance store loads 29 ms, a full local-store rewrite for runtime activity 20-49 ms, and the execution-log append 40 ms.
(3) Boot. `TldwCli.__init__` builds LocalMCPStore, UnifiedMCPContextStore, ConfiguredServerTargetStore and the control plane (which loads its context) before first paint: about 58 ms, on top of the TASK-31510 upsert.
(4) Server mode. Every `build_client()` re-reads mcp_server_targets.json, about 10 ms per call.

Structural: the MCP package's own module-scope import work at boot is modest (about 15 ms of MCP self-time; most of the rest is shared with Library/runtime_policy). But server.py is on the boot import chain (app → local_control_service → local_runtime_delegate → server) and imports mcp_unified.gateway at module scope only to set an availability flag. The standalone stdio side (gateway_runtime, TldwMCPServer, local_server_tools) is off the TUI path and mostly uses to_thread correctly.

The Hub-side costs (load_section, run_action, the server service's 9 round-trips) are already tracked in TASK-32804.12; the findings here add measured per-read costs and new send and tool-call scope.

Measurement caveat: numbers come from an isolated scratch profile on an M-series Mac, on a 12-component path. The verified-directory walk opens each path component, so a real ~/.local/share/tldw_cli profile should be roughly 0.8x these numbers, and slower hardware (the Fedora ThinkPad) higher. No real profile was touched. The pre-existing, git-ignored __pycache__ in the audit tree was reused, and `git status` in the audit tree is clean.

Cross-slice note for the UI owner: UI/MCP_Modules/mcp_workbench.py:986 runs `set_interval(0.25, _sync_local_config_save_status)`, which queries canvases and stats the config file 4 times a second while the Hub is mounted. Its per-tick cost was not measured here.

## clean areas
- tldw_chatbook/MCP/__init__.py -- the TYPE_CHECKING-only facade is already lazy
- tldw_chatbook/MCP/__main__.py -- standalone entry, off the TUI path
- tldw_chatbook/MCP/tool_naming.py -- pure stdlib; dedupe_names memoizes the next suffix, so it is linear
- tldw_chatbook/MCP/spawn_guard.py, mcp_import.py -- regexes precompiled at module scope, deferred import, cold paths
- tldw_chatbook/MCP/character_authoring_models.py -- pydantic models imported only inside the character-write functions
- tldw_chatbook/MCP/redaction.py -- regexes precompiled; the heavy result redaction runs on the agent worker thread, not the loop
- tldw_chatbook/MCP/readiness.py -- pure derivation logic, no I/O
- tldw_chatbook/MCP/permission_prompt_reducer.py -- pure; reached only from the /fewer-permission-prompts slash command
- tldw_chatbook/MCP/local_config_saves.py -- writes go through asyncio.to_thread behind a lock, and state reads are O(1) dict lookups
- tldw_chatbook/MCP/hub_tool_catalog.py -- cheap per call; the deepcopy of each schema matters only for very large external catalogs (folded into F1)
- tldw_chatbook/MCP/execution_log.py -- the TASK-21134 identity memo works (no re-scrub); the remaining cost is the admission guard (F2/F3)
- tldw_chatbook/MCP/gateway_runtime.py, server.py TldwMCPServer, local_server_tools.py -- standalone stdio server process; local tools and resources already dispatch via asyncio.to_thread
- tldw_chatbook/MCP/server_request_handlers.py, hub_test_execution.py -- sampling handler and Hub Test Tool, cold paths
- tldw_chatbook/MCP/activation.py, recovery.py -- thin wrappers; the cost sits in Backup_Recovery admission (F2)
- tldw_chatbook/MCP/unified_control_models.py -- dataclasses only (about 2.9 ms import self-time)
- tldw_chatbook/MCP/client.py -- asyncio subprocess streams with bounded line sizes; no sync I/O on the loop apart from the P3 notes in F11/F14
- tldw_chatbook/MCP/server_unified_service.py -- the 9-round-trip access-context re-resolution is already covered by TASK-32804.12
- tldw_chatbook/MCP/permission_store.py resolve_effective_state -- definition_hash is computed only for tools with an explicit tool-level entry; effective_tool_states batches into one load

## census
| Measured op (isolated scratch profile, warm, startup admission held) | median ms | where it runs |
|---|---|---|
| raw json.loads of the permission file | 0.02 | n/a (baseline) |
| MCPPermissionStore.get_kill_switch() (one guarded load, ~1,155 open() calls) | 22.8 | loop (send x3-5), worker (per local/MCP tool call) |
| LocalMCPStore.load() (8 KB / 959 KB file) | 11.4 / 21.9 | loop (send x3) |
| LocalMCPStore.record_runtime_activity() (8 KB / 959 KB) | 18-20 / 49.1 | loop (per built-in MCP tool call) |
| MCPExecutionLog.append() | 40.5 | loop (per MCP tool call) |
| activation.execution(control plane) enter+exit / nested with local service | 29.9 / 40.0 | loop (every guarded MCP call) |
| local governance check (tool.execute) | 29.1 | loop (per built-in MCP tool call) |
| describe_local_mcp_capabilities() (server.py read + ast.parse x3) | 5.8-8.5 | loop (send x2; Hub Advanced x3) |
| capture_mcp_definition_maximum(app) | 77.8 | loop, ui_submit |
| _compose_mcp_provider equivalent (pre-check + compose_catalog) | 131.0 | loop, _run_agent_reply |
| Boot: LocalMCPStore() + UnifiedMCPContextStore() + ConfiguredServerTargetStore() + control-plane ctor | 16 + 17 + 10 + 15 = ~58 | TldwCli.__init__, pre-paint |
| ConfiguredServerTargetStore.get_target() | 10.0 | loop (every server-mode build_client) |
| gate_tool_test / record_tool_decision | 18.9 / 48.0 | agent worker thread |


# slice-26

## summary
Slice #26 (Media, 5 files, 13,233 lines): local_media_reading_service.py (6,888), media_reading_scope_service.py (3,773), server_media_reading_service.py (2,030), media_reading_normalizers.py (502), __init__.py (40). All three services are built once in TldwCli.__init__ (app.py:8109-8126). Hot entry points: the Library screen (browse pages, reader open, highlights, reading progress, bulk delete/restore, "Review these"), the Console scope picker (Chat/scope_picker_listers.py -> search_media), the Library keyword-RAG seam, Home's content snapshot (list_media_items), Research Workspace source lists (get_media_detail fan-out), agent library tools (chunks/navigation, on the agent worker thread) and the Schedules workbench (list_reading_digest_outputs).

Overall: the Library is disciplined. Every Library seam call passes isolate_in_worker=True, and I found no Library path that runs sqlite on the event loop. The trouble is what each off-loop call costs.

Headline, measured: every threaded Media seam call opens a fresh SQLite connection. run_finite_local_worker closes the worker thread's connection after each call. On POSIX, each open launches an exec'd helper process (private_sqlite.prepare_in_helper) and runs storage admission. That comes to about 47 ms per call, against 0.07 ms when a pool-thread connection is reused. A Library media open makes 4 seam calls and costs about 210 ms. A 20-row browse page costs about 58 ms, of which the query itself is under 5 ms.

Second, also measured: every MediaDatabase.transaction() pays about 3.9 ms of Backup_Recovery storage-admission filesystem checks (about 245 posix.open calls), against 0.01 ms for a raw transaction. The slice's 47 per-call CREATE-TABLE-IF-NOT-EXISTS bootstraps turn plain reads such as list_highlights and list_reading_digest_outputs into about 3.7 ms admission-gated transactions. The tier-2 review measured the same bootstrap at 0.051 ms because it used raw sqlite.

Third: the Console scope picker enumerates up to 5,000 full media rows on every refresh, including every prev/next page click. Each refresh does N+1 read-it-later SELECTs and about 16.5 ms of row normalisation on the loop, for a result that is only a list of ids. Other items: the non-summary search_media re-fetches offset+limit rows per page and runs one SELECT per row; the item-detail seam reads full bodies and every version body even when include_content=False, so the Library's deliberate "read WITHOUT content" bulk pass silently fails; and there are smaller P3s.

Structural notes: the local and scope classes form a 10.6k-line pair on the boot import path through an eager package __init__. The marginal boot cost is only about 3.4 ms, because the dependencies are already loaded. A large part of the surface has no production caller outside Media/ and Tests: ingestion sources, reading import/export, digest schedules, archives, file artifacts, annotations, document insights and ingest jobs. The already-known finding that 117 _maybe_await sites run unthreaded (TASK-32804.12) is mostly latent in production because the Library isolates every call. Beware its recommended fix, routing all 117 through _call_local_leaf: until F1 is fixed, that would add about 47 ms of connection churn to each call.

## clean areas
- tldw_chatbook/Media/media_reading_normalizers.py -- pure dict projections, O(1) per row, no I/O, no regex; per-row cost is only the 5000-row picker volume (F2)
- tldw_chatbook/Media/__init__.py -- eager, but measured marginal boot import is 3.4 ms because STT.persistence/runtime_policy/Library deps are already imported earlier on the app.py path (see P3 F10)
- MediaReadingScopeService._call_local_leaf predicate (media_reading_scope_service.py:147-197) -- correctly positive-confirms local/not-coroutine/not-memory before threading; the problem is only the per-call finite-worker connection retirement (F1)
- Library browse summary path: search_media(library_summary=True) -> search_media_db lean 5-column projection with SQL-side OFFSET, one match_reasons SELECT per page only for the browse caller
- list_library_media_trash / list_media_trash / get_paginated_files (Home) -- bounded LIMIT/OFFSET, lean projections
- _build_local_media_list_response -- keywords fetched with one batched fetch_keywords_for_media_batch call, not N+1
- list_highlights / annotations DB queries -- indexed by item_id/media_id (idx_local_reading_highlights_item_id); only the per-call schema bootstrap is costly (F5)
- Library progress writes (library_screen.py _queue_library_media_progress_write) -- coalesced last-write-wins single drainer, isolated off-loop
- Every Library-screen and library_media_controller seam call passes isolate_in_worker=True (verified all get_media_item/list_highlights/create/delete_highlight/check/download/restore/delete/progress/search sites) -- no on-loop sqlite from Library
- library_local_rag_search_service._search_media -- bounded top_k, threaded via _call_local_leaf
- ServerMediaReadingService method bodies -- thin awaits on a cached TLDWAPIClient (client object is cached; only the per-call context resolution is suspect, F7)
- LocalMediaReadingService.__init__ -- trivial attribute assignment, no I/O at boot
- _chunk_text -- ChunkingService imported lazily (task-21102), off the boot path

## census
| Measurement (scratch DB, isolated env, Python 3.12, loguru sinks removed) | Value |
|---|---|
| direct get_reading_progress on warm main-thread connection | 0.01 ms |
| scope.get_reading_progress via _call_local_leaf (to_thread + run_finite_local_worker) | 48.05 ms |
| asyncio.to_thread reusing pool-thread connection (no finite-worker close) | 0.07 ms |
| open+close one MediaDatabase connection on a thread (cProfile: prepare_in_helper subprocess ~36 ms + acquire_storage ~9 ms) | 47.5 ms |
| Library media-open seam sequence (get_media_item + list_highlights + get_reading_progress + check_media_file, isolate_in_worker replica) | 210.6 ms |
| one 20-row Library browse page (library_summary) via scope | 57.8 ms |
| empty `with db.transaction()` (MediaDatabase) | 3.91-4.24 ms |
| raw BEGIN/INSERT/COMMIT on the same connection | 0.01 ms |
| upsert_reading_progress / save_media_to_read_it_later | 3.81 / 3.82 ms |
| _ensure_local_reading_aux_schema / _ensure_local_ingestion_schema per call | 3.71 / 4.44 ms |
| same 13 DDL statements on same conn without transaction | 0.014 ms |
| list_highlights / list_reading_digest_outputs(limit=1) | 3.85 / 3.31 ms |
| svc.search_media non-summary limit=500 vs search_media_db alone | 10.48 vs 4.97 ms |
| svc.search_media limit=2000 vs search_media_db alone | 30.77 vs 10.36 ms |
| 500x get_media_read_it_later_state vs one IN(500) batch | 4.39 vs 0.09 ms |
| search_media(limit=20) at offset 0 / 1000 / 1980 | 3.86 / 7.73 / 11.29 ms |
| scope.search_media(limit=5000) end-to-end (picker list_ids shape) | 141-162 ms |
| of which loop-side normalize of 5000 rows | 16.5 ms |
| id-only SELECT of 5000 ids | 1.77 ms |
| get_all_document_versions 5x540 KB include_content True vs False | 0.98 vs 0.49 ms |
| get_library_media_chunks (2000 chunks, context=2) vs windowed SELECT | 4.65 vs 0.02 ms |
| marginal boot import of tldw_chatbook.Media inside `import tldw_chatbook.app` | 3.4 ms |


# slice-27

## summary
Slice #27 "Notes#1" (14 files, about 30.8k lines under tldw_chatbook/Notes/). I found four hot surfaces:

(a) Boot. app.py:419-421 imports Notes_Library, note_folder_repository and the whole File Notes Git chain at module scope, and app.py:7933 builds the session owner eagerly in TldwCli.__init__.

(b) Library > Notes > Folder Files. library_file_notes_workspace drives FileNotesService and FileNotesReplica through asyncio.to_thread. It runs a full scan() on first open (the Library screen is rebuilt on every visit, so this happens every visit), a full scan() after every create/move/delete/restore, and a 1.5 s reconcile() poll while the workspace is visible.

(c) The File Notes Git panel. FileNotesGitService coroutines run as tasks on the Textual event loop, and each child is started with asyncio.create_subprocess_exec.

(d) Database-notes tree paging and search. note_folder_repository is reached through NotesScopeService._run_folder_repository, which uses to_thread, so it is off the loop.

Headline: the File Notes replica deletes FTS rows by filtering on UNINDEXED columns. That is a full FTS scan per upsert. scan() also upserts every file on every run, with no unchanged-file skip. Together this makes scan quadratic. Measured: 300 files 106 ms, 1000 files 524 ms, 2000 files 1.8 s first scan and 3.1 s on rescan. The user waits on this when opening Folder Files and after each file action.

The Git chain adds about 34 ms of warm boot import (106 dataclasses) for a Library sub-mode.

Git status refresh has two costs. It parses the entire repository index on the UI loop, with O(index x endpoints) overlap work on top: about 110 ms at 20k tracked files and about 500 ms at 80k. It also starts 9 sequential git children, and each start forks on the loop thread: 2.6 ms at baseline RSS, 8.4 ms at +300 MB, 16.9 ms at +800 MB.

The idle reconcile tick costs 60-85 ms per 1.5 s at 2000 files. About 65% of that is pathlib relative_to. The tick runs in a thread but holds the GIL and the service operation lock.

Smaller items: tree search does a double full-CTE evaluation (342 ms off-loop for a common term at 20k notes); oversized files are read whole and stored three times in the replica; about 450 lines of dead tree-batch SQL; per-call DDL for note links; an N+1 keyword fetch in sync finalisation; cold-path sync filesystem work and 100 Hz retry loops in the commit/push paths.

Structural notes:
- TASK-21132 (open) describes an upward closure CTE that no longer exists. `_load_managed_folder_rows` is now seeded from `requested_roots`, so TASK-21132 looks stale and can be closed on evidence.
- TASK-22219 (Done) gated the reconcile poll on visibility, but the per-tick cost itself was never addressed.
- The 2026-08-22 holistic review measured `Notes.file_notes_git_service` at 39.5 ms self, but no task was ever filed for it.

Fix grouping for PRs:
- PR-A: replica FTS keyed by rowid, plus a scan that skips unchanged files, batches its transaction, and uses reconcile() after actions.
- PR-B: lazy File Notes Git owner/service, following the `meeting_session_owner` precedent.
- PR-C: Git status diet. Scope ls-files with pathspecs or parse in to_thread, use a regex-based oid check and prefix-set overlap, and merge the rev-parse/config probes.
- PR-D: reconcile walk with os.scandir and string-relative paths, one lstat per entry.
- PR-E: hygiene (dead tree-batch code, link-schema DDL, keyword N+1, search COUNT OVER()).

## clean areas
- tldw_chatbook/Notes/__init__.py -- empty, no eager submodule imports
- tldw_chatbook/Notes/agent_lessons.py -- build_agent_lessons_runtime_guidance (called once per agent send in Agents/agent_service.py:2568/2986) is a pure string build over disclosed schema names; credential regexes are module-level compiled; initialize_agent_lessons_folder runs once from _wire_notes_sync_services with an early 'already_seeded' return
- tldw_chatbook/Notes/file_notes_conflict_compare.py -- inputs capped (200k chars / 10k lines in, 120k chars / 2k lines out); only reached from the conflict-compare dialog
- tldw_chatbook/Notes/note_folder_models.py -- frozen dataclasses and pure validators, no I/O
- tldw_chatbook/Notes/file_notes_git_commit.py and file_notes_git_push.py -- module-level compiled regexes and pure parsers/validators used only on user-triggered commit/push; no loops over repo-sized data
- tldw_chatbook/Notes/file_notes_session_owner.py -- snapshot()+coalesce_session_changes measured 0.04/0.11/0.44 ms at 300/1000/3000 recorded changes, so the unbounded _changes history is negligible; _maintenance_drain's 10 ms poll is deadline-bounded and only runs during storage maintenance
- tldw_chatbook/Notes/git_process_containment.py POSIX path -- async subprocess with bounded waits; the Windows pipe reader/writer threads per child exist only on the owned-process-tree (push) path
- tldw_chatbook/Notes/Notes_Library.py read seams (list_notes, count_notes, list_library_notes, search_library_notes, get_library_note_text, get_note_version_states) -- thin delegates whose callers run in worker threads (notes_scope_service to_thread, agent service thread); metric calls are no-ops unless TLDW_METRICS_LOGGING is set (cached env gate); marginal boot import of Notes_Library+folder repo/models measured 5.2-5.7 ms
- tldw_chatbook/Notes/note_folder_repository.py paging/locate/mutation-context -- all off-loop via NotesScopeService._run_folder_repository (asyncio.to_thread); page size 20; _load_managed_folder_rows now anchors on requested roots (TASK-21132's mechanism is gone); carrying the note content column in tree pages measured as negligible (6.1 vs 5.6 ms at 5k notes)
- FileNotesGitService construction (build_file_notes_session_owner) itself costs 0.17 ms; only the module import is expensive
- file_notes_git_network.py -- push-only path; SSH ControlMaster/ControlPersist explicitly disabled so process-group settle waits cannot be pinned by a mux master

## census
| Probe (safe, isolated, scratch dir) | Result |
|---|---|
| FTS5 `DELETE FROM files_fts WHERE root=? AND relative_path=?` (UNINDEXED cols) | plan = `SCAN files_fts VIRTUAL TABLE`; per upsert 0.46 ms @200 rows, 1.85 ms @1000, 5.2 ms @3000 |
| FileNotesService.scan (in-memory replica, ~3 KB .md files) | 300 files 106 ms, 1000 files 524 ms, 2000 files 1839 ms first / 3141 ms rescan-unchanged; without replica 53/207/392 ms (linear) |
| FileNotesService.reconcile idle tick (2000 files) | 60-85 ms per tick; _walk_candidates 67 ms, ~65% in pathlib.relative_to; list_active_files 2.7 ms |
| Warm import after textual/asyncio preloaded | file_notes_session_owner(+commit,push) 16-18 ms; rest of git chain (git_service, git_network, containment) 16-17 ms; total ~34 ms |
| parse_index_entries_z + allowed_paths overlap (20 endpoints) | 2k entries 5+6 ms; 20k 56+52 ms; 80k 270+224 ms (on the event loop in the status cycle) |
| asyncio.create_subprocess_exec('git --version') on-loop spawn | 2.6 ms (base RSS), 8.4 ms (+300 MB), 16.9 ms (+800 MB); 9 spawns per status refresh |
| search_note_tree_placements CTE (count + page) | 5k notes: rare 5.7 ms / common 68.5 ms; 20k: rare 27 ms / common 342 ms (off-loop) |
| Unfiled tree page (COUNT + ORDER BY title LIMIT 20) | 20k notes 23 + 25 ms off-loop; content column adds ~0-3 ms |


# slice-28

## summary
Slice #28 (Notes#2, 20 files, 28.6k lines) splits into two hot surfaces. (A) NotesScopeService is the app-wide notes facade. It is imported at boot (app.py:422) and called from Console (Save-as-Note, the scope picker, per-send auto-retrieve), Home (per visit), Library (every notes/tree call) and Collections. (B) The lasting-sync stack (store, executor, reconciler, authority, coordinator, filesystem, legacy, conflicts, models) is driven on the Textual event loop by notes_sync_runtime (outside this slice) on every watcher hint, Sync now, boot reconcile and conflict resolution. The Import-once pipeline (discovery, parsers, planner, executor, receipts) runs in worker threads.

Overall health: the code is carefully bounded and well-indexed, and earlier perf work (TASK-21101, 21112, 21129, 23027) holds at the SQL level. That work is now undercut by two cross-cutting costs that landed in September:

- **Per-transaction storage admission.** Backup-recovery admission (`@_core_transaction` on NotesDeviceStateStore.transaction, TASK-32628) runs for every transaction. It costs 3.4–3.7 ms per call, against 0.006 ms for raw SQLite. Measured through the executor, the same UPDATE_NOTE sync action takes ~210–240 ms of loop time with admission, against ~5.7 ms of loop-side store work with admission stubbed out. TASK-23027 last measured ~6.6 ms per note, so this is a regression.
- **Per-call connection churn.** Finite workers close their connection after every call, and every fresh private SQLite open now spawns a helper subprocess (private_sqlite.prepare_in_helper, since 61a49de2e0). Together these make each isolated notes read cost ~53–59 ms, against ~1 ms on a held connection.

Other findings are loop-side and algorithmic:
- Sync planning runs O(N) on the loop: ~30 ms at 1k bindings, ~140 ms at 5k.
- A moved or deleted file triggers an O(missing×discovered) sha256 rescan: 149 ms for 200 of 2000 files.
- Several local scope-service methods run synchronous SQLite on the loop.
- Import progress rebuilds the full validated receipt every 25 items, which is quadratic: 12.6 s at 5k items.
- The executor reloads the full recovery blob about 6 times per action.
- The organization repository's collision fallback does a full table scan on the common path.

Structural note: NotesScopeService is async-shaped but synchronous inside for the local scope. Every caller has to choose between blocking the loop and paying for finite-worker isolation (thread, asyncio.run, connection open, then close with a TRUNCATE checkpoint). One guard inside the service, using a held pool connection, would fix both. All measurements were taken in an isolated scratch profile under a 12-component path. Admission cost scales with path depth, so expect roughly half the admission figures on a real ~7-component profile path. The helper-subprocess figures do not depend on path depth.

## clean areas
- tldw_chatbook/Notes/notes_sync_legacy.py: legacy snapshot is LIMIT-bounded (LEGACY_SNAPSHOT_EVIDENCE_LIMIT, post TASK-21112) and runs once at migration through the runtime's _maintenance_offload thread
- tldw_chatbook/Notes/notes_sync_coordinator.py: portalocker NON_BLOCKING leases; RootLease.authoritative costs 4 stat calls per check, negligible
- tldw_chatbook/Notes/notes_sync_conflicts.py: pure helpers; the difflib output is bounded by _bound_diff_output
- tldw_chatbook/Notes/notes_device_state_schema.py: the schema census now runs once per held connection (TASK-21101 holds); indexes cover the binding/operation/recovery predicates
- tldw_chatbook/Notes/notes_device_state_store.py SQL shape: narrow projections (active_binding_note_ids, has_binding_for_note_or_path, candidate_binding_ids) are index-served; the per-call cost is admission, not SQL (see F1)
- tldw_chatbook/Notes/note_import_discovery.py: scandir with dir_fd walk and bounded entries; folder_is_obsidian_vault is a single stat; all callers are threaded (to_thread in the import controller and runtime)
- tldw_chatbook/Notes/note_import_planner.py: dict-keyed classification, no N^2; runs in to_thread(_plan_selection). apply_item_override is an O(n) plan rebuild per click, minor
- tldw_chatbook/Notes/note_import_parsers.py: runs in a thread; minor: YAML frontmatter is scanned and then loaded (two passes) with a pure-Python SafeLoader subclass
- tldw_chatbook/Notes/note_import_execution_models.py: the plan digest is computed once and cached on ApprovedNoteImportPlan
- tldw_chatbook/Notes/note_import_plan_models.py: O(n) __post_init__ validation, acceptable
- tldw_chatbook/Notes/note_import_executor.py: whole execute runs via asyncio.to_thread; per-effect transactions are the durability design (per-transaction admission cost is under F1, progress cost under F6)
- tldw_chatbook/Notes/notes_sync_authority.py: thin adapter, always reached on worker threads via run_worker_coroutine
- tldw_chatbook/Notes/notes_sync_filesystem.py: POSIX observe runs in threads. WindowsNotesSyncObservationFilesystem.observe re-reads and re-hashes the whole root per call (the executor calls it ~5 times per action), but it is not wired into production (zero non-test references)
- tldw_chatbook/Notes/note_import_windows_fs.py: Windows-only logic; its only cost on POSIX is being imported through notes_sync_filesystem.py:14 (counted in F9)
- tldw_chatbook/Notes/notes_sync_models.py: regexes precompiled; validate_notes_sync_opaque_id builds 2 PureWindowsPath objects per call (1.8 us), counted in F3
- NotesScopeService._sync_local_note_keywords: per-keyword queries bounded by tag count, inside one transaction
- Logging: zero eager f-string logger calls in the whole slice

## census
| Probe (isolated scratch profile, Py3.12) | Measured |
|---|---|
| NotesDeviceStateStore.get_root/get_binding warm, startup lease held | 3.4-3.7 ms/call (raw sqlite BEGIN/SELECT/COMMIT 0.006 ms); ~245 open() per call at 12-component path |
| Executor UPDATE_NOTE, per action | 61 store calls on loop thread; 210-240 ms wall; worst stall 26-38 ms. Admission stubbed: 5.7 ms loop-side store, worst stall 1.6-8 ms |
| Executor UPDATE_NOTE with 5 MB note (admission stubbed) | 31 MB of recovery BLOB read on loop; 37 ms loop-side; worst stall 25 ms |
| _run_folder_repository(db.count_notes) finite worker vs held to_thread | 53.4 ms vs 1.0 ms |
| scope.count_notes / Home-style list_notes vs held conn | 57 ms / 59 ms vs 0.95 / 0.42 ms |
| Fresh ChaChaNotes connection open (cProfile) | ~52 ms (helper subprocess ~35-44 ms, acquire_storage ~7 ms) |
| BindingObservation build + plan_reconciliation | N=1000: 18.8 + 10.7 ms; N=5000: 89 + 50 ms (token alone 4.6 / 23 ms) |
| Missing-binding identity rescan | 200 missing x 2000 files: 149 ms contiguous |
| aggregate_receipt (per 25-item progress batch) | 14.3 ms at 1k items (0.6 s/import); 63 ms at 5k (12.6 s/import) |
| search_notes local (4000-note DB) on calling thread | 3.6-5.5 ms |
| notes_sync_runtime incremental import in App.on_mount | 28-32 ms, 18 modules |


# slice-29

## summary
Slice #29 "Notes#3" (7 files, 6,990 lines). Nearly all of the cost sits in notes_sync_runtime.py, the app-owned lasting-sync runtime. It runs on the Textual loop: app.on_mount does create_task(owner.start()).

Hot paths, traced end to end:
(a) The automatic reconcile pass. schedule_hint -> _run_hint -> _reconcile -> _fresh_authority -> adapter.observe_root + plan_reconciliation. It runs once per active root at boot, on every watcher hint (1 s cadence while files change), and on every Chatbook autosave of a bound note (2 s debounce) via note_session_port._signal_lasting_sync -> note_changed. TASK-32633 plans to wire 17 more write paths into this.
(b) The review flows. A manual Check runs check_root, then conflict_labels, then binding_labels, and each one re-observes the whole folder. Setup review / activation / apply_reviewed / compare_conflict each re-observe and re-plan.
(c) The receipts and history views: write_receipts, resolution_history, active_conflict_receipts.
(d) The per-file descriptor-anchored reads in sync_paths.
(e) The File Notes 1.5 s poll, which re-runs recovery_review.require_pairing.

Overall health: all I/O is correctly offloaded. Store calls go through _maintenance_offload, file reads through to_thread, the watcher scan runs in a thread, and TASK-23027's observation reuse works. What remains is:
- pure-Python CPU that scales with vault size and runs on the loop;
- redundant recomputation of the same plan and token;
- duplicate full-root passes.

The loop-side costs:
- ~21 ms to build BindingObservation, which carries pathlib-heavy validators, plus ~10 ms per plan_reconciliation at 1,000 bindings. The plan runs 2-4 times per call.
- An O(K×N) rename/delete identity scan: 389 ms at 1,000 files.
- A PyYAML frontmatter lift at 269 µs/file for Obsidian vaults: a ~270 ms stall twice during setup/activation.
- Every sync-written file echoes back through the watcher as a second full pass.

The 1,000-file-per-root discovery cap (max_files=1_000) bounds all of these.

Structural note: every public runtime method independently takes a fresh full-root observation (the "prove the review is still current" pattern), and every save or hint re-observes the whole root. So per-interaction cost is O(root size), not O(changed items). The cheapest structural lever is to compute the plan once per observation and move the pure assembly (bindings loop, unclaimed loop, plan, frontmatter lift) into the thread hop that observe_root already makes.

Known and not re-filed: runtime construction before first paint (TASK-21247); abandoned-setup start gate (TASK-21240); idle watcher backoff (TASK-21112, Done); per-action executor stall, measured at 2.4 ms (TASK-21129, Done). TASK-23027 (Done) recorded the ~26 ms "loop-side build/plan tail" as a residual but no open task tracks it, so F2 reports it as new.

## clean areas
- tldw_chatbook/Notes/notes_sync_watcher.py: the scan runs in asyncio.to_thread, with exponential backoff plus jitter up to 10 s, an interruptible sleep, and stop on shutdown/maintenance. Idle cost is ~17 ms per scan at 1,000 files (measured) × ~8 scans/min, about 0.2% of a core. Already covered by TASK-21112.
- tldw_chatbook/Notes/template_store.py: only reached through NOTE_TEMPLATES in Event_Handlers/notes_events.py, which is imported lazily on first template use and then cached at module level. merge_templates is a cold write path.
- tldw_chatbook/Notes/recovery.py: backup/recovery owner adapters. Every import inside is function-local, and it is only reached from Backup_Recovery/DB recovery code (cold path).
- tldw_chatbook/Notes/server_notes_workspace_service.py: a thin async wrapper. tldw_api schema imports are deferred. RuntimeServerContextProvider caches the client, and the keyring read is cached with a TTL (TASK-32922). load_workspace_context's 4 sequential awaits have no production caller (tests only). filter_workspace_notes does a client-side substring match over at most MAX_RESEARCH_SELECTION_IDS rows, which is minor.
- notes_sync_runtime.py store access: every NotesDeviceStateStore call from the owner goes through _maintenance_offload (a to_thread hop). note_file_location is offloaded. The _bundles cap is 8, and the observation-reuse cache is bounded at 32 MiB per root and revalidated per item. The watcher only starts once a lease exists. _load_roots does one get_root per root, and root counts are tiny.
- notes_sync_runtime.folder_is_sync_root: a sync Path.resolve() on the loop per Import-once folder pick, looping over a handful of roots. Negligible.
- notes_sync_runtime._maintenance_drain: a 10 ms poll, but only during a backup-maintenance drain bounded by its deadline.
- sync_paths.py fsyncs and double parent-identity verification: durability and TOCTOU guarantees, deliberately not flagged.
- notes_sync_conflicts.build_conflict_comparison (called on the loop from compare_conflict): unified_diff at the 10k-line cap measured 18 ms, so it is fine.
- recovery_review.py import weight: imported lazily (notes_recovery_dialog is loaded inside a function; file_notes_service imports it under TYPE_CHECKING or inside functions).

## census
| probe (Python 3.12, this Mac, isolated env, audit tree 840ed2ca58) | N | measured |
|---|---|---|
| BindingObservation construction (observe_root bindings loop, on loop) | 1000 | 20.6 ms |
| plan_reconciliation (on loop) | 1000 | 10.3 ms per call |
| _observation_token (on loop) | 1000 | 4.2 ms per call |
| rename/delete identity scan (observe_root L1008-1015, on loop), K missing paths × N files | 100 / 300 / 1000 | 4 / 34 / 389 ms |
| _lifted_note_metadata, Obsidian frontmatter (on loop) | 1 file / 1000 | 269 µs / 269 ms |
| discover_import_sources + _discovery_signature (watcher, thread) | 1000 files | 17 ms |
| cold observe: per-file to_thread vs one batched to_thread | 1000 files | 241-325 ms vs 131-156 ms |
| PinnedSyncRoot.read_bytes | 1000 files | 211 µs per file |
| xattr list + ACL probe: CDLL-per-call vs cached libc | 1 fd | 25.9 µs vs 1.4 µs |
| unified_diff at the comparison cap | 10k lines | 18 ms (fine) |


# slice-3

## summary
Slice #3 Audio (36 files, 27,859 lines). Almost none of it loads at boot. Only streaming_sink (~2 ms, via tts_events at module scope) loads before first paint, and voice_process_types (~2.7 ms) loads after ui_ready. recording_service, numpy, sounddevice and torch never load at boot. The costs this slice leaks into interactive paths:

(1) Lab > Speech > Dictation calls the blocking LazyLiveDictationService start/stop directly on the event loop. The first Start measures about 0.45 s warm and 1.3 s cold. Stop can block for seconds while it joins the processing thread and transcribes the final segment.

(2) Console realtime voice builds the mic recorder with default retain_audio=True and no cap. Every captured frame stays in memory for the whole session (48 KB/s, ~173 MB/h).

(3) Two synchronous config writes run on the loop: update_privacy_settings (3 writes, 174 ms measured) and meetings apply_device_choice.

(4) Blocking PortAudio calls run on the UI thread in the realtime path: first-entry recorder import/construct (116 ms measured import), a sink open on every reply, and abort/close on barge-in and TTS stop.

(5) The first Lab > Speech visit imports numpy (~23 ms), and dictation_service_lazy is the only reason.

(6) Meetings with live diarization are O(n^2): every speaker refinement repaints the whole RichLog (37 ms at 600 lines, measured) and rewrites all of transcript.jsonl. The Stop-pass relabel does this once per changed segment.

The speculative duplex voice pipeline (qualification-gated, TASK-23175) runs mostly in a child process and has a sound IPC design (coalesced scheduling, credit windows). It still carries P3 costs: 1-5 ms sleep polling (1.75% of a core per 1 ms poll, measured), pure-Python per-sample DSP (resampler at ~30 ms per second of audio on the owner loop, correlation at ~20 ms per window), and a per-frame rebuild of the rolling window (2.8% of a core).

Structural: Audio/dictation_service.py (the legacy LiveDictationService, 652 lines) and Widgets/voice_input_widget.py have no production importers and duplicate the lazy service. recording_service imports both capture backends plus webrtcvad at module scope.

Suggested PR groups:
- A: move dictation/realtime/sink blocking calls off the loop (F1, F4, F3).
- B: memory fixes (F2 plus F8).
- C: meetings O(n^2) (F6, F7).
- D: lazy imports and dead code (F5, F12, F13).
- E: speculative-voice polling and DSP vectorisation (F9, F10, F11).

## clean areas
- tldw_chatbook/Audio/__init__.py: PEP 562 lazy exports; importing the package pulls in no backend
- tldw_chatbook/Audio/streaming_sink.py core: lock-guarded O(1) leftover-offset _take_locked, bounded 60 s buffer, notify-thread handoff, lazy sounddevice. Only the sync open/stop-on-loop use (F4) and the 10 ms drain poll (F10) are flagged. Boot import is ~2 ms, judged fine
- tldw_chatbook/Audio/wav_writer.py: buffered append-only writer, header patched on close
- tldw_chatbook/Audio/meeting_owner.py: prepare/start/stop/enroll/learn/recover all run from @work(thread=True) workers in meetings_screen; the watchdog runs only while a meeting is active. Only apply_device_choice (F3) runs on the loop
- tldw_chatbook/Audio/meeting_capture.py: PCM rings capped at 60 s, tap buffer capped at 1 s, numpy handle cached. EnergyRing._slice copies 6k-entry deques per partial (~0.3 ms, processing thread): negligible
- tldw_chatbook/Audio/diarizer_local.py: one persistent worker subprocess with the model resident, bounded lock waits, non-blocking pin() from the UI thread, per-seq reply matching
- tldw_chatbook/Audio/diarizer_cluster.py, diarizer_engine_onnx.py, diarizer_engine_speechbrain.py: work bounded by max_speakers; numpy/torch/sherpa load only inside the worker subprocess
- tldw_chatbook/Audio/system_audio_tap.py: probe and swiftc compile run in the prepare worker; reader/stderr threads use buffered reads
- tldw_chatbook/Audio/voiceprint.py: store built and voiceprint loaded off the UI thread; keyring read is bounded
- tldw_chatbook/Audio/voice_process_io.py and voice_process_protocol.py: coalesced owner-loop scheduling, snapshot-bounded drain batches, credit windows, no body copies on write
- tldw_chatbook/Audio/parakeet_voice_worker.py and voice_transcription.py: persistent spawn worker; STT calls go through executor/to_thread
- tldw_chatbook/Audio/voice_turn_coordinator.py, voice_phrase_sequencer.py, voice_metrics.py, duplex_contracts.py, voice_process_types.py, aec_backend.py: bounded deques and char caps, no per-event whole-history walks found
- tldw_chatbook/Audio/realtime_mic_tap.py buffering logic: bounded pre-ready buffer, per-thread in-flight tracking; tap.stop already runs via asyncio.to_thread. Only the recorder kwargs (F2) and UI-thread construction (F4) are flagged
- Console dictation path (UI/Console_Modules/dictation.py ConsoleStreamingDictationSession) runs the blocking start/stop halves through asyncio.to_thread. This is the template F1 should copy
- Widgets/audio_troubleshooting_dialog.py: device enumeration already runs via run_worker(thread=True)

## census
| Probe (isolated env, audit tree 840ed2ca58) | Result |
|---|---|
| import Audio.recording_service (numpy+sounddevice Pa_Initialize+webrtcvad) | 116 ms warm, 500 ms cold |
| import sounddevice alone (Pa_Initialize) | ~80 ms |
| import numpy | 23-29 ms |
| import Local_Ingestion.transcription_service (first dictation Start) | 213 ms warm, 1038 ms cold |
| update_privacy_settings = 3x save_setting_to_cli_config | 174 ms median (161-205) |
| Lab>Speech stts_screen marginal import after app | 85-118 ms; numpy 23 ms of it, sole importer dictation_service_lazy |
| RichLog clear + rewrite 600 lines (wrap, 211 cols) | 37 ms median |
| iter_normalized_pcm_frames, 1 s of 24 kHz PCM | 31.6 ms (48 kHz passthrough 5.7 ms) |
| asyncio.sleep poll loop, 1 ms / 10 ms / 50 ms | 1.75% / 0.38% / 0.08% of a core |
| TranscriptEngine.append_admitted_frame, full 4 s window | 0.28 ms per 10 ms frame (2.8% core) |
| acoustic-isolation lag scan shape, 51 lags x 6000 samples | 20 ms pure-Python vs 0.14 ms numpy |
| voice source_identity() (53 files, 1.69 MB sha256) | 2.7 ms warm, 18-40 ms first |
| boot importtime: Audio.streaming_sink / voice_process_types | 1.7-2.5 ms at boot / 2.7-5.6 ms post-ready |


# slice-30

## summary
Slice #30 Personal_Context (21 files, 12,286 lines). Hot paths traced from outside the slice:
(1) Console agent send. The run awaits `ConsoleChatController._personal_context_service()` (to_thread, bounded at 10 s), then `_compose_profile_tool_provider` (to_thread), then `ProfileToolProvider.list_catalog()` in the run_reply thread (bridge compose plus the registry cache build, so at least twice), then `ProfileContextService.build_snapshot()` (run_reply thread). All of this happens before the provider is called, so it adds straight to time-to-first-token.
(2) Settings > My Profile: load, and a reload after every mutation. This runs in thread workers.
(3) Profile interview. The launch prep runs synchronously ON THE EVENT LOOP; the coordinator's start/answer/finish run in threads.
(4) First-link to the home server. `_run_personal_context_link` is `run_worker(coro)`, so its sync repository and keyring calls run on the loop.
(5) The sync outbox dispatcher, which runs only when linked.

Overall health: the design is correct but very expensive. `PersonalContextRepository._connection()` opens a fresh hardened `connect_private_sqlite` for every repository statement group unless the caller is inside `read_operation()`. TASK-31504 added `read_operation()` for exactly one caller (`_compose_profile_tool_provider`, confirmed at 1 connect). Every other caller is unwrapped: ProfileToolProvider `_live_scope`/`list_catalog`/`invoke`, `settings_snapshot`, the interview coordinator, the interview launch, `first_link_snapshot`, and the outbox dispatch.

The cost of each connect also rose about 100x since TASK-31504 was measured (0.44 ms). Since 2026-09-07 every `connect_private_sqlite` on POSIX spawns a fresh `python -I -S private_sqlite_helper_entry.py` subprocess plus an admission thread. I measured 44-50 ms per connect with cProfile, in isolation.

The repository list methods are also N+1 (they re-fetch each row they already hold, and run a per-row quarantine probe). Together these give measured costs of:
- 0.5-1.3 s per profile-tool catalog listing or invocation
- 16 s for a My Profile load (50 records, 200 proposal receipts, 3 workspaces)
- about 5.5 s per interview answer
- 1.3 s frozen on the event loop when an interview is launched

Secondary costs:
- Every agent view decrypts the whole export snapshot, including proposal receipts that are never pruned, at least 5 times per send.
- A trusted-directory path walk from `/` runs on every statement inside an operation.
- A DISABLED profile still pays about 88 ms per send, because only ABSENT is negatively cached.
- The first agent send of every process reads the OS keychain even for users who never created a profile. On the very first send it also creates a key and the DB; this is the 3.7 s / wedged cost TASK-32344 bounded but did not remove.

Fix groups for PRs:
- PR-A (low risk, largest win): wrap every public read entry point in `read_operation()` (profile tool provider, `settings_snapshot`, coordinator actions, `first_link_snapshot`, dispatcher) and negative-cache the DISABLED state.
- PR-B: de-N+1 the repository list methods and add a view snapshot that skips proposals.
- PR-C: move interview launch prep and link planning off the loop, and drop the keychain probe.
- PR-D: stop keychain and DB creation for absent profiles.
- PR-E (cross-cutting, DB owner): reuse or pool the private-SQLite helper instead of one process per connect.
- PR-F (hygiene): lazy package `__init__`, proposal receipt compaction, outbox write amplification.

All measurements come from isolated scripts under scratch/slice30. They used a scratch HOME, XDG_* and TLDW_CONFIG_PATH, the InMemoryProfileKeyProtector, no network, and the real SQLite and helper path.

## clean areas
- tldw_chatbook/Personal_Context/crypto.py: per-object AES-GCM envelope with 2 AESGCM constructions per op; decrypt measured at about 27-37 us per object including pydantic validation. Fine per object; volume is the problem (see F6).
- tldw_chatbook/Personal_Context/runtime_policy.py, repository_models.py, paths.py: tiny, no runtime cost.
- tldw_chatbook/Personal_Context/sync_outbox.py: thin wrapper; its cost is the unwrapped repository calls it forwards (covered in F1 fix scope and F13).
- tldw_chatbook/Personal_Context/interview_diff.py: pure O(n) dict passes over the batch and existing records.
- tldw_chatbook/Personal_Context/interview_provider.py: FixedQuestionProvider is O(1). ConfiguredModelQuestionProvider uses the shared chat_api_call from the coordinator thread (no client per call on the loop).
- tldw_chatbook/Personal_Context/reconciliation.py: pure planning. The nested workspace loop is O(local_ws x remote_ws x remote_records), bounded by small workspace counts, and runs once per link.
- tldw_chatbook/Personal_Context/export_service.py: one transactional snapshot per export, run in a thread worker from the Settings panel.
- tldw_chatbook/Personal_Context/context_service.py: the serialize loop re-renders JSON per record but the 12 KB byte gate short-circuits tokenization. Measured cold at 5.8 ms for 50 records and 13.7 ms for 200 (plus a one-time 118 ms tiktoken load); adds at most 30 estimate-cache entries per build. Not worth a finding.
- tldw_chatbook/Personal_Context/key_protector.py: scrypt (N=2^14) only on the passphrase protector paths; keyring use is on the paths flagged in F4/F8.
- tldw_chatbook/Personal_Context/bootstrap.py: fine apart from the absent-profile keychain issue in F8.
- Chat/console_chat_controller.py `_personal_context_service` (10 s budget plus in-flight guard) and `_compose_profile_tool_provider`: correctly off-loop. The TASK-31504 single-connection claim holds (measured 1 connect).
- Absent-profile negative cache (service.status absent signature): holds, with 0 connects per send after the first.
- Settings personal_context_panel load/mutation/export and ProfileInterviewScreen coordinator actions: all run in thread workers (not loop-blocking; latency only).
- Boot/first-paint: Personal_Context is imported lazily everywhere (TYPE_CHECKING or function-local imports); nothing is on the boot path.

## census
| Operation (isolated, real SQLite + private-SQLite helper) | Profile shape R/P/W | Wall time | Hardened connects | Notes |
|---|---|---|---|---|
| `connect_private_sqlite` (repo._connect) | - | 44-50 ms each | 1 | cProfile: `HelperLease.start` runs `subprocess.Popen([python,-I,-S,helper])` plus an admission thread join |
| `storage_signature()` / `verify_trusted_directory` | - | 0.14 / 0.13 ms | 0 | runs once per `_connection()` inside an operation |
| `_compose_profile_tool_provider` replica (wrapped) | 50/200/3 | 70 ms | 1 | 77 path walks, 532 decrypts |
| `ProfileContextService.build_snapshot` | 50/200/3 | 58 ms | 1 | 263 decrypts per view |
| `ProfileToolProvider.list_catalog()` | 20/0/2 | 857 ms | 21 | unwrapped; runs at least twice per send |
| `ProfileToolProvider._live_scope()` | 20/0/2 | 520 ms | 13 | runs on every list_catalog and invoke |
| `ProfileToolProvider.invoke(profile_search)` | 20/0/2 | 1,297 ms | 30 | runs per profile tool call |
| DISABLED profile: compose + build_snapshot | 0/0/0 | 43.7 + 43.5 ms | 1 + 1 | cost for any user who created a profile but left agent use off |
| `svc.settings_snapshot()` (Settings > My Profile load) | 50/200/3 | 15,975 ms | 346 | the same call inside `read_operation()` takes 162 ms |
| `svc._list_profile_proposals()` | 50/200/3 | 9,250 ms | 204 | write txn plus a per-row `_is_quarantined` |
| `repo.list_records()` (N+1) | 50/200/3 | 4,519 ms | 102 | 62 ms inside an operation (still 103 path walks) |
| `repo.list_scopes()` | 50/200/3 | 441 ms | 10 | - |
| Interview launch replica (event loop) | 50/0/3 | 1,310 ms | 27 | excludes the 4 keychain calls of the draft probe |
| `coordinator.start` / `answer` / `finish` (thread) | 50/0/3 | 5,780 / 5,190-5,241 / 5,349 ms | 117-118 each | - |
| Personal_Context package import (after pydantic/crypto/config) | - | ~17.4 ms | - | from -X importtime; tldw_profile_core accounts for 17.4 ms |


# slice-31

## summary
Slice #31 Prompt_Management (23 files, 9,867 lines). Hot paths: (1) Console Prompts modal (open, page change, 200 ms-debounced search), the `/prompt` and `/system` picker filter (200 ms debounce), Draft Shelf CRUD, save-to-Library and record-usage. All of these reach `PromptScopeService` from coroutine workers or handlers ON the event loop. (2) Library ▸ Prompts. Its controller correctly isolates scope-service calls in `asyncio.to_thread` (`isolate_in_worker=True`), but import parsing still runs on the loop, and the editor rebuilds its full state on every keystroke. (3) Console auto-retrieval's prompts keyword seam, which runs per send. (4) Boot: `app.py` imports four names from the PEP 562 lazy package facade at module scope, so 16 of the 23 slice modules are resident at boot.

Structural diagnosis: `PromptScopeService` is an async facade over synchronous SQLite. It has 32 `_maybe_await(sync_call())` seams and only one `to_thread` (`count_prompts`). Every caller must therefore either block the loop (Console) or spin a fresh `asyncio.run` loop in a worker thread per call (Library). The right fix site is the scope service itself.

Cross-slice discovery: every `PromptsDatabase.transaction()` pays the Backup_Recovery storage-admission handshake. That is about 4.2 ms and about 245 `open()` syscalls per transaction in an isolated profile, against 0.1 ms of actual SQL. It dominates list, draft and collection costs, and the same decorator wraps 26 transaction methods in 16 stores. TASK-32804.1 amortised this only for config reads.

Also found:
- a quadratic legacy-prompt decomposer on the Library editor's per-keystroke path;
- a pure-Python YAML loader running on the loop during import;
- dead boot wiring: `PromptChatbookScopeService` and the `server_prompt_service` instance have no production readers, and an unused frontmatter probe logs a WARNING every boot;
- duplicated Local/Server service classes and normalizers.

All measurements come from an isolated scratch profile using the pinned audit tree. Nothing touched the real profile.

## clean areas
- prompt_artifact_models.py / prompt_artifact_codec.py / prompt_block_compiler.py: per-selection/per-save decode+compile; measured normalize_prompt_record on 25 structured (6 KB definition) records = 0.5 ms total; XML collision regex is served by re's internal cache
- prompt_normalizers.py: PromptScopeService._normalize_prompt_list normalizes each item twice (normalize_prompt_list then _normalize_prompt_record), measured negligible (0.03-0.5 ms per 25 rows); history-page normalizer is per history open, bounded page
- prompt_source_capabilities.py: canonical_json_utf8_size json.dumps once per structured save; local_prompt_capabilities is a cheap frozen dataclass
- prompt_batch_models.py, prompt_restore_errors.py, prompt_chatbook_record.py, prompt_markdown_export.py, server_prompt_adapter.py: O(1)/O(fields) per record or per explicit export action
- prompt_improvement_service.py / prompt_improvement_prompts.py / prompt_improvement_models.py / prompt_preservation.py: one-shot user action; estimate_tokens is memoized (TASK-18602); module-level regexes precompiled
- PromptScopeService.count_prompts: correctly offloaded via asyncio.to_thread(run_finite_local_worker)
- PromptScopeService.get_capabilities: server capabilities cached on success; local is a constant
- ServerPromptService._require_client -> runtime_policy build_client caches the TLDWAPIClient (pooled httpx client); per-call get_active_context/credential lookup belongs to the runtime_policy slice
- Library prompts controller (outside slice): every scope-service call uses _run_library_service_call(isolate_in_worker=True) -> asyncio.to_thread; collections/memberships also isolated
- Console prompt picker / modal: debounce (200 ms) + search tokens + bounded 25-row pages; this is the GOOD debounce pattern apart from the on-loop DB work
- LocalPromptService._ensure_collection_schema: executescript DDL + commit per collection call measured 0.02 ms (cheap, hygiene only)
- PromptsDatabase.search_prompts N+1 keyword fetch: measured 0.09 ms raw for 25 rows (not material); DB-level p.* + unindexed-sort cost belongs to the DB slice
- Prompts_Interop markdown/json parsers: linear regex passes, one prompt per file; import_prompts_from_files has zero production callers
- local_prompt_service.LocalPromptService consumers (agent library tools via console_runtime.py:703, MCP server) run on agent/MCP threads, not the UI loop

## census
| Probe (isolated profile, audit tree 840ed2ca58) | 200 prompts | 2000 prompts |
|---|---|---|
| scope.search_prompts local limit=25 (Console modal/picker path, prefix "Summ") | 2.9 ms | 26-28 ms |
| scope.list_prompts local per_page=10 (modal open/page) | 3.6 ms | 7.0 ms |
| scope.list_prompt_drafts (Draft Shelf, empty table) | 3.7 ms | 6.7 ms |
| scope.list_prompt_collections limit=100 | 4.4 ms | 6.5 ms |
| raw SQL for list_prompts (COUNT + page) | - | 0.1 ms |
| empty `with db.transaction(): SELECT 1` | - | 4.19 ms, ~245 os.open |
| db.execute_query SELECT 1 (no admission) | - | 0.01 ms |
| build_prompt_editor_state (Library editor, per keystroke) 4K/14K/56K/140K legacy | 0.04 / 0.56 / 6.6 / 34 ms | |
| decompose_legacy_lanes x2 lanes 14K/56K/140K/280K | 0.6 / 6.2 / 37 / 144 ms | |
| YAML import 0.94 MB (200 prompts) safe_load_all vs CSafeLoader vs json | 213 / 15 / 0.6 ms | |
| compile_prompt_variables x2 lanes 4K/20K/100K chars | 0.9 / 5.1 / 24 ms | |
| slice import increment at boot (16/23 modules resident after `import tldw_chatbook.app`) | 7.5 ms | |


# slice-32

## summary
Slice #32 RAG_Search (45 files, 27,351 lines). The RAG code is mostly light at boot: `RAG_Search/__init__` is PEP 562 lazy, and `RAG_Search.simplified` stays at 0 modules at _ui_ready. On the Console first-paint leg, RAG_Search modules cost only about 6 ms. Hot paths identified: (1) Library Search/RAG and the Console Library-search modal. Both run `@work(exclusive=True, group="library_rag_search") async def _execute_library_rag_search` (library_screen.py:35095), which is a coroutine worker on the event loop. It calls LibraryLocalRagSearchService, which does `await rag_service.search(...)` on EnhancedRAGServiceV2. (2) Every Console send: the turn context reads `library_rag_profile_top_k()`. (3) Post-ingest indexing on the `rag-ingestion-indexer` daemon thread, plus the Settings Backfill thread worker. (4) Optional LLM reranking.

The dominant cost is not the retrieval itself. It is the ADR-126 recovery-admission layer that this slice stacks 4–5 deep on every call: `projection_lifetime.async_operation` + `service_query`/`store_query` + `activation_async_guarded`. In the isolated env (HOME depth 10), a raw Chroma query costs 0.91 ms. The guarded `store.search` costs 138 ms and `search_with_citations` costs 245 ms. A full V2 semantic search costs about 450 ms wall with mock embeddings. Cache hits cost the same as misses, because the cache sits inside the guards. A 1 ms loop heartbeat saw 160–340 ms single stalls and about 400 ms of total loop blocking per search.

Root causes in-slice:
- `generation._scope` re-resolves `get_media/chachanotes/prompts_db_path()` on every scope, at about 33 ms each (they route through the guarded `get_user_data_dir`, which TASK-32804.1's `get_cli_setting` fastpath did not cover).
- The nested service_query/store_query layers re-run the whole scope instead of reusing it.
- `async_guarded`/`async_operation` spawn a new Task per call. This defeats the identity-keyed lease reuse, so every nested guarded call re-enters `execution_scope` for every source: about 70 per search and about 71 per indexed document.

The same mechanism causes:
- a 37 ms/doc indexing overhead (guarded Chroma delete 27.6 ms vs 0.16 ms raw);
- UI heartbeat jank (p99 24 ms, max 130 ms) while ingest indexing runs in the background thread;
- about 5 ms of loop time per reranker LLM call;
- in the restored-backup state, a full-corpus reconciliation four times per search.

Separately:
- The first Console send pays about 420 ms on the loop. That covers importing the eager `simplified/__init__` tree (186 modules including numpy and Chunking) and building the profile manager (12 × `RAGConfig()` at about 31 ms each, via `default_chroma_persist_directory` → `get_user_data_dir`).
- Scoped semantic search re-embeds the query and re-runs the whole guarded search once per source type.
- The keyword leg SELECTs full media bodies and regex-scans each whole document on the loop (30–38 ms per 1.1 MB row).

Magnitude caveat: path-admission cost scales with directory depth (measured about 5.6 ms per path component per path getter). A typical depth-2 HOME should see roughly 25–30% of the admission-bound numbers, which still puts a Library search at about 100+ ms on the loop. The machine was also shared with other audit agents during measurement.

## clean areas
- RAG_Search/__init__.py -- PEP 562 lazy facade; ~0.6 ms incl. admit_startup(); keeps Chunking/simplified off boot (TASK-21102/21731 guarantee holds at _ui_ready)
- RAG_Search/semantic_availability.py -- service construction and stats via asyncio.to_thread; app-level cache with generation staleness; no per-search construction (Library _resolve_rag_runtime reuses app._rag_service)
- RAG_Search/local_citation_capture.py -- imported by console_display_state at first paint (~0.8 ms); formatting operates on bounded staged evidence only
- RAG_Search/fusion.py, search_modes.py, pipeline_types.py, simplified/data_models.py, simplified/citations.py -- pure functions over top_k-sized inputs; module-scope regex only
- RAG_Search/chunking_service.py, enhanced_chunking_service.py, parent_child_adapter.py -- thin delegates to Chunk_Lib; run on ingestion worker threads / rag executor
- RAG_Search/simplified/collection_fingerprint.py, collection_indexes.py -- Settings callers (fetch_index_status, activate/save) all run in thread workers
- RAG_Search/simplified/search_service.py -- MCP path; shared service via to_thread; keyword content batch-fetched (no N+1); runs in MCP server process, not the TUI loop
- RAG_Search/simplified/simple_cache.py async get/put/prune -- TASK-32811.6 accounting fix is on dev; _deep_getsizeof cost verified fine by core review (sync-wrapper hygiene noted separately)
- RAG_Search/ingestion_indexing.py IngestionIndexer threading model -- one daemon thread, one event loop for the thread lifetime, blocking queue.get (no polling); Settings backfill is @work(thread=True); only the per-doc admission overhead is flagged
- RAG_Search/simplified/vector_store.py semantic query offload -- TASK-32804.9's asyncio.to_thread for the Chroma query is present on dev
- RAG_Search/simplified/rag_service.py _hybrid_search -- legs gathered concurrently; FTS sub-legs run SQL in run_in_executor; FTS queries are LIMITed and id-scoped via json_each (no per-id placeholders)
- RAG_Search/pipeline_loader.py, pipeline_builder_simple.py, pipeline_functions_simple.py -- legacy perform_* path has no production caller (TASK-32807.2 In Progress); its first-paint import leg via chat_rag_events is ~1 ms, not worth a separate finding
- RAG_Search/parallel_processor.py, simplified/health_check.py, simplified/enhanced_rag_service.py *_with_parents paths, config_profiles experiments -- dead/unreached in production (TASK-32807.2 / core review D3)
- RAG_Search/simplified/circuit_breaker.py -- threading.Lock-based; call_sync's per-call new_event_loop has no production caller
- RAG_Search/eval/ (gating, metrics, regression) and backfill.py -- test harness / CLI only, not on any UI path
- No set_interval/set_timer/polling, run_worker, reactive, or query_one anywhere in the slice; model_recovery maintenance drain polls at 10 ms only during backup maintenance
- Embeddings/Embeddings_Lib.py's module-scope import of RAG_Search.model_recovery is light (pydantic + Backup_Recovery), and Embeddings_Lib is not on the boot path

## census
| Measurement (isolated env, HOME depth 10, mock embeddings; 1 run each, median of 10-15) | Value |
|---|---|
| raw `collection.query` (60 chunks) | 0.91 ms |
| guarded `ChromaVectorStore.search` (1 store_query) | 137.8 ms |
| guarded `search_with_citations` (2 nested store_query) | 244.7 ms |
| `EnhancedRAGServiceV2.search` semantic / hybrid, Chroma, cache miss | 449 / 565 ms |
| V2 search, in-memory store: miss vs cache hit | 315 vs 312 ms |
| loop heartbeat during one search: longest stall / total blocked | 162-344 ms / ~400 ms |
| per semantic search: query_scope / get_*_db_path / acquire_storage / execution_scope / open() | 4 / 12 / 83 / 70 / 28,719 |
| `get_media_db_path()` warm | 33-43 ms (depth 10), 118 ms (depth 20) -> ~5.6 ms per path component |
| `generation.query_scope(svc)` alone | 111-127 ms |
| `activation.execution()` 3 sources / 3-nested | 12 / 18.8 ms |
| first `library_rag_profile_top_k()` (first Console send) | 421 ms, +186 modules (numpy, Chunking) |
|   of which import `simplified.active_config` / first `get_profile_manager()` | 73 / 326 ms |
| `RAGConfig()` default construction | 30.7 ms (12 per profile-manager build) |
| `require_local_embedding` per query embedding (HF id, no witnesses) | 11.0 ms |
| `_keyword_citation_spans` per 1.1 MB row | 29.8-37.9 ms |
| `index_batch_optimized` overhead per doc (50 docs) | 37.1 ms/doc; ~71 execution_scope/doc |
| guarded `delete_document` vs raw Chroma `delete(where)` | 27.6 vs 0.16 ms |
| UI heartbeat during background indexing (40 docs) | p99 3.2 -> 23.9 ms, max 4.9 -> 129.9 ms, ticks 1161 -> 622 |
| pointwise rerank of 20 rows, zero-cost stub LLM | 101 ms (~5 ms admission per LLM call) |
| `recovery._rows` canonicalization | 0.14 ms per 384-dim chunk |
| second `MediaDatabase(...)` on existing DB (first keyword search pool) | 42-50 ms |


# slice-33

## summary
Slice #33 (ROOT#1, 9 files, ~23.4k lines; app.py is 19,682 of them). I read every file and read the hot paths in app.py in full. Hot paths in scope: (a) the boot chain `cli.main_cli_runner` -> `app_entry.main_cli_runner` -> `TldwCli.__init__` -> `on_mount` -> `_push_initial_screen` -> `_post_mount_setup` -> the deferred timers; (b) the `NavigateToScreen` worker and FIFO-lock navigation, with reusable screens and screen-owned CSS; (c) the App-level `Worker.StateChanged` hook. That message has bubble=False, so the hook sees only App-owned workers. (d) The TTS event handlers. (e) The Library ingest queue mixin: the claim/top-up/progress/write loop on the UI thread, the spawn parse pool and its drain and monitor threads. (f) The logging pipeline: the loguru -> stdlib forwarder, the redacting file sink and the Logs-screen buffer handler. (g) The command-palette providers.

Overall health: the navigation, deferred-startup and shutdown machinery is careful. It uses to_thread or thread workers, bounded waits, a staggered boot fleet and paced pre-imports. The real costs sit in four places:
1. The ingest registry is deep-copied whole on the UI thread, per file, in two places. TASK-32804.5 fixed a sibling site; this is O(N²) and the registry has no bound within a session.
2. The boot-time CSS staleness check measures 80–134 ms per source-tree boot, against the ~0.3 ms its docstring claims.
3. The loguru forwarding sink is installed at TRACE. That makes every suppressed debug call cost ~50x more, and it silently defeats the DB layer's `opt(lazy=True)` guards on every statement.
4. The general parse pool is never retired. Its workers and a 20 Hz poll thread stay alive after the first import.

Smaller items: a launch-wake SQLite self-join over the unbounded agent_runs table runs on the loop right after `_ui_ready`; unhandled App workers emit WARNING spam through a pipeline that redacts each record twice; a 10 Hz watchdog thread runs always-on; there are dead TTS widget queries and a dead media-type prefetch; eager wiring of 74 server services and 41 scope services in `__init__`.

Structural note: app.py still has 228 module-scope imports, 31 of them *_Interop packages. `TldwCli.__init__` builds ~115 service facades before the loop exists; this is covered by TASK-33011 and the ADR-097 census. The census measures in-process `run_test` boots, so it cannot see entry-point-only boot work: the CSS staleness walk, and the textual_image/PIL import in `warm_up_image_protocol`, about 11 ms measured.

Items already known and not re-filed:
- `self.theme` set in `on_mount` forces a full bundle reparse (TASK-22505).
- `console_voice_input` import and `find_spec` probes in `on_mount` (TASK-22504).
- Folder-submit fsync per job (TASK-32804.5).
- `get_personal_context_service` bootstrap on the loop (TASK-32370).
- Standing costs: Console re-mint (TASK-24452) and `query_one` counts per switch (TASK-24455).

Probes: all ran against the audit tree with an isolated HOME/XDG/TLDW_CONFIG_PATH and TLDW_TEST_MODE, or against pure stdlib/synthetic replicas. Nothing wrote to either checkout: the CSS manifest save was stubbed out.

## clean areas
- tldw_chatbook/__init__.py: arms the tiktoken assets lazily and sets env vars; imports loguru only if it is already loaded
- tldw_chatbook/__main__.py and tldw_chatbook/cli.py: every heavy import is function-local
- tldw_chatbook/chunking_engine_version.py: a constant only
- tldw_chatbook/Constants.py: pure constants, no imports, no module-scope work
- app.py query_one override: the default screen holds only #screen-container, so the NoMatches fallback costs microseconds
- app.py navigation (_dispatch_screen_navigation worker, FIFO asyncio.Lock, overlay dismissal, reusable installed screens, _ensure_screen_owned_css first-visit only): clean apart from the standing TASK-24452/24455 costs
- app.py command-palette providers: static lists; TabNavigationProvider measured at 0.12-0.29 ms per keystroke because Textual's Matcher caches
- app.py on_app_focus and WideViewportTierMixin.on_resize: set_class is a no-op unless the width crosses the threshold
- app.py staggered boot-worker fleet: the 2 s reconcile timer stops itself once the gate drains
- app.py deferred startup tasks (citation reconcile, legacy citation migration, subscription interrupt reconcile, media cleanup, change-review retention): all go through asyncio.to_thread with bounded batches
- app.py ingest progress path: the drain thread coalesces at 0.25 s and updates are persist=False; _restore_ingest_jobs runs as a thread worker; the research startup sweeps use to_thread
- app.py remote ingest poll: an async worker that awaits network I/O and exits when drained
- app.py screen pre-import threads: paced, finite, check shutdown, park during navigation
- app.py TTS profile repository open: an async task; project-skills discovery and the env-key flag write are thread workers
- app.py quit/shutdown: joins are bounded and kept off the loop (except the afplay psutil scan noted in F13)
- app.py persona buddy: disabled by default, so the passive property is a cheap preferences parse
- app.py UI heartbeat set_interval at 1 Hz: the per-tick work is trivial (see F7 for the watchdog thread)
- app_destinations.py: lazy module; the Personal Context bootstrap is covered by TASK-32370
- Logging_Config.py: RichLogHandler is not installed in the master shell; PrivateRotatingFileHandler hardening runs only on open or rollover; enable_crash_forensics runs once
- app_entry.py: early logging and config loads are cache-backed; argparse and the emoji check are cheap

## census
| probe (isolated, measured) | result |
|---|---|
| synthetic ingest job snapshot (`_copy_job` replica, 5 deepcopies/job) | 1.19 ms @100 jobs, 6.02 ms @500, 11.73 ms @1000 |
| `_generated_css_is_stale` (real code, manifest save stubbed) | 81-134 ms per call (fresh result) |
|  of which: manifest 198 keys via validate_path+stat | 19-55 ms (lexical join+stat: 1.3-1.5 ms) |
|  of which: walk of 2,519 .py | walk only 7-8 ms; +stat 14-15 ms; +relative_to 49-52 ms; relative_to+stat 61-66 ms |
| loguru suppressed debug call, forward sink at TRACE vs INFO | 6.0-6.4 us vs 0.12 us |
| lazy SQL debug + indexed SELECT (execute_query shape), TRACE vs INFO sink | 7.7-8.2 us vs 1.7 us per statement |
| redact_log_line per line | 44 us |
| unhandled worker transition (debug + WARNING through the file and buffer sinks) | 105 us |
| launch-wake discovery query (synthetic agent_runs, warm cache) | 6.5-10.7 ms @5k runs/24 MB; 23.5-28 ms @20k runs/98 MB |
| psutil.process_iter over ~990 processes (quit path) | 35-55 ms |
| textual_image.widget import (warm_up_image_protocol, no tty) | 10-12 ms, loads PIL |
| chat_message_enhanced import: cold vs after textual_image/PIL | 52-58 ms vs 1.2-1.8 ms |


# slice-34

## summary
Slice #34 (ROOT#2) is config.py (10,129 lines), emergency_stop.py, model_capabilities.py and provider_registry.py. Everything the app does reads config through config.py, so the slice's hot paths sit outside it: the Console keystroke derivation pass, the 4 Hz credential poll, the per-send emergency-stop gate, boot (config import plus TldwCli.__init__), Settings saves, and agent file/git/patch tool calls.

The dominant problem is structural. The ADR-126 admission wrapper `@_config_participants.guarded` was added on 2026-09-16 (b5251e9a6e). On every call it runs a full storage-admission handshake: both config RLocks, acquire_storage, and roughly 500 to 1,600 posix.open calls. TASK-32804.1 moved only `get_cli_setting` / `load_cli_config_and_ensure_existence` ahead of that wrapper (a warm read now costs 1.3 µs). The other guarded readers still pay the handshake on every warm call. Measured in an isolated scratch profile (deep HOME path; real-profile cost is roughly 0.5x, per TASK-32804.1's 4.8 ms):

- `load_settings()`: 10.1 ms
- `get_user_data_dir()`: 34 ms, 1,592 opens per call, no memo
- `get_runtime_config_snapshot()`: 14 ms
- `get_api_key()` and `get_cli_providers_and_models()`: 11 ms each, because they call load_settings

Because the handshake takes the same RLocks that every config write holds for its whole ~60 ms transaction, a warm load_settings on the loop blocks while a worker saves or resolves tool context. Measured: median 36 ms and max 65 ms during concurrent saves; p95 66 ms and max 81 ms during concurrent agent tool-context resolution.

The resulting costs:
- The 4 Hz Console credential poll is an always-on idle tax of about 2-4% of a core.
- Every printable Console keystroke pays one load_settings handshake.
- Every send runs get_user_data_dir on the loop via the emergency-stop gate (28 ms measured).
- TldwCli.__init__ statically reaches about 26 get_user_data_dir calls before first paint, roughly 0.4 to 0.9 s.
- Every agent file/git/patch tool call spends 462 ms resolving the sensitive-path context (19 get_user_data_dir calls).
- A Settings save costs 62 ms: three TOML parses, whole-config deepcopies, and a 43 ms full rebuild, more than half of which is get_user_data_dir again.
- With config encryption on, each save re-derives scrypt keys twice per encrypted value, at 27 ms each.

Suggested PR grouping:
- PR-A: warm fast paths for load_settings and get_runtime_config_snapshot, following the TASK-32804.1 precedent (F1, F3). Low risk; removes the idle tax, the keystroke cost and the lock contention.
- PR-B: memoize get_user_data_dir per config identity with a cheap lstat revalidation, and resolve the emergency-stop path and sensitive-path context once (F2). Largest boot and per-send win; needs a security review against ADR-127.
- PR-C: trim the write and rebuild path and the module-scope boot reads (F4, F6).
- PR-D: session memo for decrypted config values (F5).
- PR-E: hygiene (F7 god-module split, F8 lru_cache on a method).

provider_registry.py is clean. model_capabilities.py and emergency_stop.py are clean apart from F8 and F2's call site.

## clean areas
- tldw_chatbook/provider_registry.py: stdlib-only frozen-dataclass data module; module-scope work is a few tuple/dict comprehensions over 28 records; no I/O, no heavy imports
- tldw_chatbook/model_capabilities.py: family regexes precompiled at module scope; per-(provider,model) _capability_cache; lazy global instance; config read via the fast load_cli_config_and_ensure_existence path; models.dev gap-fill is off by default, memory-only and never fetches. Only F8 remains (plus resolve_deepseek_effective_model's snapshot use on cold briefing/RAG-answer paths, folded into F3)
- tldw_chatbook/emergency_stop.py: small JSON sentinel with an atomic write; the read is a FileNotFoundError fast path (~µs). Its only cost is default_emergency_stop_path -> get_user_data_dir (F2)
- config.py get_cli_setting / load_cli_config_and_ensure_existence warm path (_warm_config_cache_hit, TASK-32804.1): 1.3 µs per call measured
- config.py runtime_capture_policy(): cached per generation and rollout env, 1.3 µs; current_config_identity() 1.1 µs; _get_effective_config_path() lru-cached, 1.0 µs
- config.py get_console_ssh_settings 7 µs, get_media_ingestion_defaults 1.5 µs, get_detected_api_providers 9 µs, get_canvas_config_policy 0.7 µs (Web_Server only), load_openai_mappings 0.03 ms
- config.py _external_edit_detected: stat throttled to 1 Hz, and own writes re-stamp the file (_install_bootstrap_cache_from_raw), so there are no spurious reloads
- config.py _load_cli_config_bootstrap lock-free fast path (TASK-21124) is sound; migrate_config_file_if_needed is a free no-op; validate_config_keys/difflib runs only from the Settings advanced editor
- config.py get_notes_sync_watcher_intervals / get_notes_sync_recovery_capacity_bytes / load_console_library_migration_seed: pure in-memory reads
- config.py provider_settings_for_key / normalize_provider_config_key: cheap per call; per-keystroke volume already tracked by TASK-24454

## census
| call (warm, isolated scratch profile, deep HOME) | ms/call | note |
|---|---|---|
| get_cli_setting / load_cli_config_and_ensure_existence | 0.0013 | TASK-32804.1 fast path |
| load_settings() | 10.1 | guarded handshake on every warm hit |
| get_user_data_dir() | 34.2 | 1,592 posix.open/call; get_*_db_path 28-32 |
| get_model_cache_dir() | 41.6 | guarded + nested |
| get_runtime_config_snapshot() | 14.0 | handshake + 2 RLocks + whole-config deepcopy (0.53 ms) |
| get_atomic_config_snapshot() | 23.0 | write lock + raw parse + merge |
| get_api_key('openai') / get_cli_providers_and_models() | 11.5 / 11.4 | via load_settings |
| load_settings(force_reload=True) (rebuild) | 42.7 | ~35 ms of it is get_user_data_dir |
| save_setting_to_cli_config | 62.5 | 3 TOML parses + deepcopies + rebuild |
| resolve_sensitive_context() (per agent tool call) | 461.6 | 19 x get_user_data_dir |
| emergency-stop send gate | 28.4 | get_user_data_dir + read |
| warm load_settings with concurrent saves (other thread) | median 36.3 / max 65.0 | RLock contention |
| warm load_settings with concurrent tool-context resolve | p95 66.4 / max 81.2 | RLock contention |
| tomllib.loads(CONFIG_TOML_CONTENT) | 4.4 | every import |
| scrypt N=16384 (per encrypted value) | 27.3 | no derived-key memo |
| import tldw_chatbook.config (-X importtime) | self 72 / cum 265 | cum includes DB modules app imports anyway |


# slice-35

## summary
Slice #35 Scheduling: tldw_chatbook/Scheduling, 41 files and about 17k lines. I read all of loop.py, queue.py, the four handlers, heartbeat, projections, the service/sync/DB hot paths and the migrations, and I skimmed the pure helper modules.

HOT PATHS
- Boot: app.py:487-500 imports the whole Scheduling core eagerly. App.__init__ builds ScheduledTasksDB at app.py:10469.
- on_mount (before _ui_ready): runs reconcile_stale_automation_runs synchronously and create_task(recover_inflight_transfers()) at app.py:15027/15045.
- Scheduler: SchedulerLoop.run() is a coroutine worker on the app loop, started after _ui_ready at app.py:16260. It ticks every 30 s.
- Schedules workbench: on_mount calls three synchronous DB reads, then a coroutine-worker load_tasks. Every reminder save/toggle/delete runs as a coroutine worker that awaits SchedulingService.
- Sync: sync_now and the results-pull workers are coroutine workers. ConflictsTab._resolve is synchronous.

DOMINANT ISSUE (measured)
- ScheduledTasksDB opens a fresh private-SQLite connection for every operation (66 sites).
- On POSIX every open goes through DB/private_sqlite.py prepare_in_helper, which spawns a Python helper subprocess. cProfile puts 38 of 45 ms per open there.
- Result: about 45-51 ms per open against 0.8 ms for a raw sqlite3.connect. Every DB method costs about 45-60 ms whatever the query.
- Many callers run on the event loop, so this becomes real freezes:
  - about 218 ms per Schedules visit;
  - 100-160 ms per reminder save;
  - about 175 ms of event-loop stall on every boot, plus 44 ms in __init__;
  - 649 ms first-run construction.
- The helper-spawn-per-open cost is cross-cutting and lives in the DB slice. Any store that opens a connection per operation pays it; TASK-24457 (Library) is the same class. The in-repo fix template is the held per-thread connection from TASK-21131 (EventStateRepository / ClientNotificationsDB).

SECOND ISSUE: the async facade over synchronous DB calls
- SchedulingService's reminder/list/transfer paths make 52 direct self.db calls; SyncEngine makes 80. None are wrapped in asyncio.to_thread, while the automation-definition paths in the same file already are.
- Everything reached from coroutine workers therefore blocks the loop.

SMALLER STRUCTURAL NOTES
- execution_scope is nested redundantly per tick (loop stall of about 12 ms every 30 s).
- Every queue reload retires the SubscriptionsDB worker connection and runs a TRUNCATE checkpoint.
- The heartbeat pays a full F_FULLFSYNC barrier every 30 s.
- automation_results has no retention and is sorted by an expression, so every read scans the whole table.
- db/__init__.py imports schema, which imports migrations.v0_to_v1. That defeats the warm-boot 'skip migration imports' intent, but costs only about 0.3 ms, so no finding was filed.

Overall: the loop and handler design is good (offload helper, spawn-not-await handlers, bounded timeouts). The cost sits in the DB connection model and in the reminder/sync service layer not following the offload discipline its sibling paths already use. Measurements were taken with isolated HOME/XDG/TLDW_CONFIG_PATH in the scratch dir; see the census.

## clean areas
- tldw_chatbook/Scheduling/scheduler/queue.py - in-memory sorted list, pop_due is O(due) with tiny n; load() runs via SchedulerLoop._offload (thread). list.pop(0) is negligible at queue sizes seen
- tldw_chatbook/Scheduling/scheduler/loop.py tick core - emergency-stop read and heartbeat write are offloaded (TASK-31507 fix verified in code); dispatch DB writes go through _offload; sync preflights offloaded; reload wake-up is event-driven (call_soon_threadsafe), not polled
- tldw_chatbook/Scheduling/scheduler/handlers/automation_handler.py - every DB write via asyncio.to_thread, execution spawned as a task (never awaited in the tick), claim guard bounded, heavy automation_execution import lazy
- tldw_chatbook/Scheduling/scheduler/handlers/watchlist_check_handler.py - all SubscriptionsDB access through run_db_off_loop; monitors lazily constructed
- tldw_chatbook/Scheduling/scheduler/handlers/reminder_handler.py - NotificationDispatchService.dispatch uses ClientNotificationsDB's held thread-local connection (cheap on loop); fires only per reminder
- tldw_chatbook/Scheduling/scheduler/handlers/briefing_handler.py generation path - spawned task, _default_preset_id/_watchlist_name/keep_briefing all to_thread (only the incident writes in F6 are not)
- tldw_chatbook/Scheduling/services/server_client.py - async, per-call asyncio.wait_for timeouts, bounded exponential backoff, capabilities cache; no HTTP client constructed per request (delegates to the notifications service)
- tldw_chatbook/Scheduling/services/scheduling_service.py automation-definition authoring/lifecycle/resolve/review/run-now paths - consistent asyncio.to_thread discipline
- tldw_chatbook/Scheduling/services/watchlist_projection.py, briefing_projection.py - pure row->model mapping, only called off-loop from PriorityQueue.load (workbench passes include_projections=False)
- tldw_chatbook/Scheduling/schedule_compute.py, schedule_input_parsing.py, schedule_vocabulary.py, automation_preview.py, automation_validation.py, recurring_question_scope.py, slot_keys.py, task_incidents.py (module-scope precompiled regexes), constants.py, models.py - pure functions; per-keystroke form use (parse_forgiving_datetime, croniter.is_valid) is microsecond-level
- tldw_chatbook/Scheduling/automation_execution.py / automation_health.py - lazily imported; measured marginal import after app boot 0.7 ms (+2 modules), get_cli_setting fast path, per-row health is cheap
- tldw_chatbook/Scheduling/events.py - Message classes only, lazily imported by app (_post_reminder_dispatched)
- tldw_chatbook/Scheduling/recovery.py - backup/recovery adapters, cold path only (lazy import from Backup_Recovery.sqlite_validation)
- tldw_chatbook/Scheduling/scheduler_heartbeat.py read side - read_heartbeat 0.025 ms; workbench's 5 s liveness refresh is negligible
- tldw_chatbook/Scheduling/db/scheduled_tasks_db.py sync-apply bodies (_apply_pulled_reminders, upsert_*_from_server, _detect_server_deletions_conn) - one transaction per batch/page, indexed per-row lookups (UNIQUE(owner_id, server_id)); cost issue is only the connection model and loop placement (F1/F3)
- automation_runs retention (create_automation_run prunes to 200/definition) and scheduled_task_runs prune - bounded tables

## census
| Measurement (isolated scratch profile, macOS, py3.12) | Median |
|---|---|
| ScheduledTasksDB._get_connection + close | 45-51 ms (38 ms in prepare_in_helper subprocess) |
| raw sqlite3.connect + 2 PRAGMAs | 0.8 ms |
| get_reminder_task / list_reminder_tasks | 59 / 57 ms |
| ScheduledTasksDB() warm / cold (first run) | 42-44 ms (1 open) / 649-1058 ms (8 opens) |
| on_mount reconcile_stale_automation_runs | 47 ms |
| recover_inflight_transfers (sync part on loop) | 95-127 ms |
| Schedules visit on-loop reads (sync_state+conflicts+unread count+list_tasks) | 218 ms |
| SchedulingService list_tasks / create / update / delete (local) | 50 / 102 / 158 / 101 ms |
| mark_reminder_dispatched (get+update) | 96 ms |
| execution_scope enter/exit | 6.1 ms |
| write_heartbeat (fsync+F_FULLFSYNC) | 8.0 ms |
| croniter get_next | ~10 us/iter |
| inbox query @20k results (SCAN + TEMP B-TREE) raw / via DB | 62 / 100 ms |
| unread-ids load @13k unread | 199 ms |
| Scheduling-only boot imports: module self-time / croniter+dateutil / models | 12.1 / 6.6 / 5.7 ms |


# slice-36

## summary
Slice #36 is tldw_chatbook/Subscriptions (46 files, 24,277 lines). It is the Watchlists and briefings business layer. Its hot entry points are:
- Boot: app.py imports it at module scope and TldwCli.__init__ runs _wire_watchlists_and_notifications_services.
- The Watchlists screen: every visit runs the overview (4 sequential service reads) plus the Reader page and the tree, and each item action (mark read, star, status) is one service call.
- Scheduled and manual checks: the SchedulerLoop is a coroutine worker on the app loop (app.py:16253). LocalWatchlistsService.execute_run, FeedMonitor, URLMonitor and wait_for_terminal_run all run on that loop, and only explicit to_thread or run_db_off_loop hops leave it.
- Home (list_home_run_snapshot, threaded), OPML import/export, briefing, script and audio generation (coroutine workers), and the daily-report demo.

Overall: the CPU-heavy diff, extraction and feed-parse work was already moved off the loop (TASK-15463, 16839, 19562) and the briefing modules are disciplined. The dominant costs now come from the layers underneath the slice.

1. Connection churn (P1). db_offload.run_db_off_loop was changed by TASK-31993.5 (Done) to retire the worker thread's connection after every call. On macOS and Linux, every private-SQLite connect spawns a `python -I -S` helper subprocess. Result: every watchlists DB op costs 45.6 ms instead of 3.7 ms (measured). This silently undoes TASK-15463's held-connection win.
   - The Watchlists overview pays about 180 ms of avoidable latency per visit.
   - A scheduled feed check spawns about 15 interpreters.
2. Per-operation admission tax (P1, cross-slice). Every SubscriptionsDB method pays about 3.7 ms of Backup_Recovery storage admission (245 open() syscalls) against a 7 µs query.
3. Pre-paint boot cost (P1). SubscriptionsDB construction plus the startup boundary capture cost about 60 ms before first paint.

Second-tier findings (P2):
- An 8.6 ms httpx AsyncClient/SSL context is built on the loop per fetch.
- Briefing audio stitching (pydub decode, quadratic append, re-decode) runs on the loop.
- Feed ETag/Last-Modified is read but never persisted, so conditional GET never fires. Combined with an unconditional upsert, each unchanged item causes about 22 row changes per check.
- Sitemap XML and API JSON are parsed on the loop, and sitemaps have no URL cap.
- The OPML seams do sync sqlite on the loop plus an N+1 import.

Structural notes:
- About 5.5k lines (22%) of the slice are never imported: scrapers/*, web_scraping_pipelines.py, site_config_manager.py, baseline_manager.py (TASK-1360), token_manager.py. They have zero runtime cost and are candidates for deletion, not a speed fix.
- The slice's incremental boot import is only 13.9 ms (measured).

All measurements ran on an isolated scratch profile (macOS arm64, py3.12, Textual 8.2.8). The admission-tax magnitude depends on the profile path depth. The helper-spawn cost is structural on non-Windows.

Suggested PR grouping:
- (A) db_offload owner-thread and hop consolidation: F1, F9.
- (B) boot deferral: F3.
- (C) network client reuse, conditional GET and upsert churn: F4, F6, F7.
- (D) loop hygiene: F5, F8, F10, F11, F12.
- (E) cross-slice admission cache: F2, owned by Backup_Recovery.

## clean areas
- tldw_chatbook/Subscriptions/__init__.py: PEP 562 lazy facade; monitoring_engine/security load only on demand
- tldw_chatbook/Subscriptions/briefing_service.py, briefing_cast.py: DB work grouped into single asyncio.to_thread hops per stage; sync chat providers hop via to_thread; claim registries O(1)
- tldw_chatbook/Subscriptions/briefing_selection.py: SQL LIMIT/COUNT/MAX bounded window queries, predicate reuse instead of enumerated ids (datetime() predicate non-sargable only on first-briefing window: cold, one-shot)
- tldw_chatbook/Subscriptions/briefing_keep.py, briefing_export.py, briefing_feed.py, daily_reports_view.py: invoked from thread workers or to_thread by all traced callers
- tldw_chatbook/Subscriptions/fts_backfill.py: thread worker, paced, abortable; lock-shape risk already tracked in TASK-21233
- tldw_chatbook/Subscriptions/watchlists_operation_coordinator.py: Semaphore(4) bound, per-receipt dicts pruned on terminal, terminal checks via to_thread
- tldw_chatbook/Subscriptions/startup_reconcile.py reconcile_interrupted_subscription_work: runs threaded (coordinator.reconcile_startup to_thread); only the pre-paint capture is flagged
- tldw_chatbook/Subscriptions/monitoring_engine.py CPU paths: feed parse, HTML extraction, segmentation/diff all hopped to threads with segment-once sharing (TASK-16839); snapshot retention bounded at 3/url with indexed lookup
- tldw_chatbook/Subscriptions/server_watchlists_service.py: client provider caches the TLDWAPIClient (no client-per-request)
- tldw_chatbook/Subscriptions/watchlist_content_alert_service.py: per-scope haystack cache already present
- tldw_chatbook/Subscriptions/watchlist_normalizers.py, watchlist_failure.py, item_dates.py, noise_defaults.py (lru_cache), watchlist_rule_matching.py, item_persist.py validation: pure, cheap per row
- tldw_chatbook/Subscriptions/html_text.py: all regexes compiled at module scope (only body_snippet input size flagged P3)
- tldw_chatbook/Subscriptions/watchlist_item_page.py: deepcopy measured 0.26 ms per 50-item page, negligible
- tldw_chatbook/Subscriptions/watchlist_scope_service.py: thin async delegation (only OPML import/export flagged)
- tldw_chatbook/Subscriptions/briefing_voices.py, recovery.py, db_offload.py in-memory branch: clean
- tldw_chatbook/Subscriptions/watchlist_preview_service.py: builds SubscriptionsDB(':memory:') on the loop (6.5 ms measured), user-triggered only, not filed
- Dead, never imported, zero runtime cost: scrapers/* (3,059 lines), web_scraping_pipelines.py, site_config_manager.py (Backup_Recovery only probes sys.modules), baseline_manager.py (TASK-1360), token_manager.py

## census
| Measurement (isolated scratch profile, macOS arm64, py3.12) | Result |
|---|---|
| `run_db_off_loop(db, db.get_subscription, 1)` median / p90 | 45.6 / 50.0 ms |
| `asyncio.to_thread(db.get_subscription, 1)` (cached per-thread conn) | 3.7 ms |
| 5 concurrent: run_db_off_loop vs to_thread | 101.6 vs 19.0 ms |
| SubscriptionsDB connection open (spawns `python -I -S` helper) | 42.6 ms |
| SubscriptionsDB.close() (incl. wal_checkpoint TRUNCATE) | 0.35 ms |
| get_subscription on same thread vs raw `select 1` | 3.74 ms vs 0.007 ms (245 posix.open per call) |
| SubscriptionsDB construct: cold / warm | 161 / ~45 ms |
| capture_prior_process_boundary (4 txns) | 14.8 ms |
| httpx.AsyncClient() construct+close, verify=True / False | 8.57 / 0.23 ms |
| sitemap defusedxml parse+walk, 5k / 50k URLs (0.7 / 7.1 MB) | 17 / 174 ms |
| API json.loads 4.8 MB | 5 ms |
| pydub-style `combined = combined + seg` concat, 20/40/80 turns x 15 s | 20 / 66 / 280 ms (b''.join: ~1-2 ms) |
| WatchlistFilterService.evaluate, 10 items x 1 MB page x 5 keyword filters (+1 regex) | 29 ms (65 ms) |
| Re-upsert of 50 unchanged items | 11.9 ms, 1134 row changes (vs 3.2 ms existence probe) |
| body_snippet on 2000-char HTML preview | 0.19 ms/row (9.4 ms per 50-row page) |
| SubscriptionsDB(':memory:') construct (preview) | 6.5 ms |
| Slice incremental boot import (app.py's 6 Subscriptions imports) | 13.9 ms |
| run_db_off_loop call sites | 61 (49 local_watchlists_service, 5 monitoring_engine, 4 check handler, 3 screen) |


# slice-37

## summary
I audited the whole Sync_Interop slice: 40 files, about 15.4k lines, at dev 840ed2ca58. The slice is mostly cold code, but a few parts are hot and they are expensive.

What calls into the slice:
- **Boot.** app.py imports six sync services at module scope, and Workspaces.display_state (on the chat_screen first-paint path) imports sync_readiness and sync_promotion_state.
- **Settings > Overview.** The sync rows refresh on every mount, resume and category switch. They run on a thread worker.
- **Server-mode users only.** When the active runtime source is a server, `_wire_notes_sync_services` wires NotesOrganizationSyncService. It runs from a 5 s post-ready timer, on server switch, and on first use through the deferred facade. After that, every Notes folder operation, every lasting-sync reconcile (one per file change), every conversation-keyword edit and every note-keyword save calls `resolve_profile_scope` or `_profile`, both backed by the sync state DB.
- **Manual Sync button, adoption review and server activation.** These are async `@work` coroutines with no thread, and they run the sync services on the event loop.

There are no timers, polling loops or threads anywhere in the slice, and no `to_thread` either: every async sync method does its SQLite work inline.

The dominant structural cost is in SyncStateRepository. It opens a new private SQLite connection for every method call, and on macOS/Linux each open spawns the private_sqlite helper process (a fresh `python -I -S`). I measured this: about 46 ms per repository call, against 0.045 ms on a held connection. First use in a session costs about 91 ms, and creating the file for the first time costs 318 ms. Write methods commit and then read back on a second connection, so each write pays twice. This one pattern multiplies every other finding: Settings sync-safety rows cost 209 ms (local) or 320 ms (server) per refresh, and each server-mode Notes folder operation stalls the loop by about 47 ms.

Other findings:
- The legacy Notes organization inventory re-hashes the whole source for every single intent it commits. That is O(n²), and it runs on the event loop. The digest step alone takes 11.7 ms per iteration at 3,000 items, about 35 s in total. A `ponytail:` comment in the code already acknowledges it.
- sync_readiness pulls the tldw_profile_core pydantic tree onto the boot path just to read one integer constant: 18.4 ms in `-X importtime`. The ui_ready census only governs `tldw_chatbook.*` modules, so it cannot see this.
- The full Sync v2 service graph (24 modules, 10.5 ms measured) is imported and built at boot for every user. SyncRestoreService in that graph has no production caller, and sync_once cannot run in production because `sync_v2_local_store` is never assigned (TASK-1602).
- The outbox read helpers load and JSON-parse the whole outbox (dispatched rows are never pruned) and filter in Python, with N+1 calls per item. This is latent until TASK-1602 lands; today only the Personal Context first link reaches it.

Suggested PR grouping:
- **(A) Sync state store:** hold one connection per thread (F1) and push outbox filters into SQL, with pruning (F7).
- **(B) Notes organization scope:** resolve and memoize the profile scope off the loop and outside write transactions (F3, F8), and replace the double whole-table link scan with a targeted diff (F9).
- **(C) Manual sync off the loop:** move the blocking work to threads (F4) and give the inventory a single snapshot with batched commits (F2).
- **(D) Boot:** import tldw_profile_core lazily and build the sync graph lazily (F5, F6).
- **(E) Hygiene:** remove duplicate helpers and dead modules (F10).

No open backlog task covers the connection churn, the inventory, or the scope-resolution findings.

## clean areas
- sync_readiness.build_sync_readiness_report / sync_promotion_state.build_sync_promotion_state: pure dataclass builders called from Workspaces/display_state._workspace_sync_label on Console renders; cheap (only the module-scope import is a problem, F5)
- No polling: grep of the slice finds no set_interval/set_timer/threading/while-True+sleep; the only while-True loops are pagination loops bounded by server has_more
- chat_outbox_producer.py + ChaChaNotes read_committed_chat_sync_intent: ConsoleChatStore.sync_v2_chat_producer is never wired in production (no `sync_v2_chat_producer=`/ChatSyncV2OutboxProducer( outside tests), so no per-send or per-chunk sync cost
- notes_outbox_producer via NotesScopeService._enqueue_local_note_upsert: production callers never pass sync_v2_profile (only Workflows passes None), so no per-note-save outbox cost today
- hashing.canonical_payload_hash (per-message sha256 over compact JSON): cheap, called at most a few times per turn
- validation.py: linear per-envelope checks
- crypto.py: AES-GCM per envelope; scrypt only in wrap/unwrap recovery (SyncKeyRecoveryService has no production consumer); the Cryptodome import is already paid by Skills_Interop at boot
- envelope_builder.py / envelope_applier.py / domain_adapters/*: per-envelope work proportional to input; tldw_api schema imports already deferred (task-285)
- ServerSyncService._require_client: client cached by RuntimeServerContextProvider.build_client (no client per request)
- Settings sync rows: computed in a thread worker (_refresh_sync_rows) and applied in one call_from_thread hop, a good pattern; the only waste is the connection churn in F1
- personal_context_adapter.py / personal_context_dispatcher construction: lazily imported (census-forbidden pre-ready); per-record HMAC plus canonical JSON is proportional
- recovery.py: import-only SQLite declarations, no runtime I/O
- Sync_Interop/__init__.py is a PEP 562 lazy facade (defeated by app.py's module-scope import, see F6); domain_adapters/__init__ is eager but tiny (<2 ms)
- sync_mirror_report.py, conflict_review.py (logic), sync_state.py: small pure helpers
- notes_mirror.py, notes_m1_flow.py, notes_local_store.py, sync_profile_status_state.py, key_recovery_service.py: no production consumers (dead or test-only), not imported at boot
- sync_state_repository summary helpers (_sync_v2_outbox_summary, _sync_v2_identity_summary) already use GROUP BY counts; get_sync_v2_profile_summary has no production caller
- Conflict-report LIKE-prefix scans (case-insensitive LIKE cannot use idx_sync_conflict_scope) and the `(? IS NULL OR col = ?)` filters in list_identity_mappings/list_mirror_reports defeat indexes, but those tables are effectively empty in production (record_identity_mapping has 0 callers); not filed

## census
| Measurement (isolated scratch profile, macOS, py3.12) | Result |
|---|---|
| SyncStateRepository.get_latest_mirror_report / list_sync_v2_profile_states (per call, warm) | 45.7 / 47.3 ms |
| Same query on a held sqlite3 connection | 0.045 ms |
| connect_private_sqlite alone (helper subprocess spawn) | 41.0 ms |
| First repository call in a new session (existing file: schema ensure + query) | 91.3 ms |
| First-ever schema init (new file) | 318 ms |
| SyncScopeService.list_write_sync_promotion_states (Settings Overview rows), local / server | 208.7 / 319.5 ms |
| Inventory snapshot digest only, S=500 / S=3000 items (per iteration; x S iterations) | 1.5 ms (0.8 s total) / 11.7 ms (35 s total) |
| tldw_profile_core import inside `import tldw_chatbook.app` (-X importtime cumulative) | 18.4 ms |
| tldw_profile_core standalone after pydantic | 31-37 ms |
| Rest of Sync_Interop boot graph (24 modules, profile_core pre-imported) | 10.4-10.9 ms |


# slice-38

## summary
Slice #38 TTS#1 (80 files, ~59.7k lines) checked against origin/dev 840ed2ca58. I skimmed every file, read the hot code, traced callers outside the slice, and ran benchmarks in an isolated profile (all writes went to the scratch dir).

Hot paths in this slice:
1. **Boot.** app.py imports the TTS core at module scope and builds `TTSService` inside `TldwCli.__init__`, before first paint.
2. **First TTS use per session.** The legacy backend manager imports all seven backends.
3. **Every local-backend utterance.** Kokoro, Chatterbox, Higgs and OmniVoice each call `AudioService.convert_audio` once per clip.
4. **Play button.** `SimpleAudioPlayer.play` is called from UI handlers.
5. **Studio/Playground generate.** Request admission calls the studio preferences loader.
6. **Briefing audio stitch.**
7. **Streaming PCM frame decode** for voice mode.

Overall health: the core is well built. Profile-repository SQLite, clone materialization, model loads, audio.cpp launch preparation, downloads and ONNX inference all run off the loop (dedicated executor, `to_thread`, or per-stream worker threads). The only background timer is the audio.cpp health check, which runs only while a managed audio.cpp server is running.

The costs leak at the edges:
- **Boot:** about 45–64 ms of TTS-only imports plus about 35 ms to build the service before first paint, most of that in one `get_user_data_dir` call.
- **One blanket import:** the first TTS use loads every backend, including Higgs, which imports torch. That freezes the UI for 0.55–2.2 s even for OpenAI-only users.
- **`async def` functions doing blocking work:** pydub/ffmpeg encode per utterance, a fixed 100 ms `time.sleep` in Play on macOS, a forced full config re-parse (~35 ms) per Studio generate, and quadratic pydub concatenation in the briefing stitch.

Structural notes:
- TTS modules are pulled at module scope by app.py, `tts_events`, `stts_events` and `settings_speech_tts`. Fixing boot cost needs a small split of the TTS event message classes out of the handler modules. This also adds ui_ready census headroom (TASK-23155/31816/32644).
- `profile_repository.py` (6,304 lines) and `profile_service.py` (3,678) are god modules, but `profile_repository` is already lazy (about 10 ms warm when first used). `profile_service` is imported at boot only because of app.py:303.
- `profile_source.py` checks the exact composition edge (`sys._getframe` comparisons against `build_default_tts_service` and `get_user_data_dir`). That rules out the easy fix of passing a pre-resolved data dir. Deferring construction is the viable one.
- Audiobook generation is unreachable today: the handler calls a nonexistent `get_cost_estimate` and fails before `generate_audiobook` runs, so its quadratic and blocking code is latent (P3).

Suggested PR groups:
- **PR-A, TTS boot diet (F1):** lazy `tts_service`, TYPE_CHECKING-only types in app.py, TTS message classes moved to a light module, lazy recipe registry, project-before-deepcopy in `legacy_bridge`.
- **PR-B, lazy legacy backend registry (F2):** import only the requested backend; move torch/torchaudio/librosa into the functions that use them; also defer the `kokoro_onnx` and numpy imports.
- **PR-C, move audio work off the loop (F3, F4, F6, plus m4b `subprocess.run`):** `to_thread` inside `convert_audio`, `play_audio_file` and `concat_wav_segments`; drop the fixed sleep; join-based stitch; header-derived duration.
- **PR-D, Studio admission without a disk re-parse (F5).**
- **PR-E, single-pass PCM normalization (F8).**
- **Before anyone wires audiobook (F7):** bytearray/join accumulation and `to_thread`.

## clean areas
- tldw_chatbook/TTS/__init__.py: PEP 562 lazy export facade (good pattern; admit_startup short-circuits after first call)
- tldw_chatbook/TTS/profile_repository.py: all SQLite work goes through a single-worker ThreadPoolExecutor (_submit_operation -> executor.submit); per-op authority revalidation and check_repository_source run on the worker thread; the module is lazily imported on first profile use (~10 ms warm)
- tldw_chatbook/TTS/profile_schema.py, profile_validation.py (quick_check + full-row validation at open) and profile_migration_{candidate,journal,namespace,native,publication,recovery}.py: open-time and migration-time only, on the repository worker thread; TASK-21130 already streams reference BLOBs
- tldw_chatbook/TTS/profile_reference_materialization.py, profile_reference_storage.py, profile_reference_audio.py, profile_reference_types.py: native file I/O via asyncio.to_thread workers; WAV parsing bounded
- tldw_chatbook/TTS/profile_service.py observe_availability: page-bounded; per-clone-profile audio_cpp_guided_dependency_snapshot is in-memory (AudioCppSettingsConfig.from_mapping measured at ~21 us)
- tldw_chatbook/TTS/adapters/audio_cpp.py: pooled httpx clients per adapter generation, explicit timeouts, cached catalog, bounded response reads, cleanup via to_thread
- tldw_chatbook/TTS/audio_cpp_supervisor.py: health scheduler only while a managed server runs (default 10 s interval, gated on state); bounded diagnostics deque; process spawn is async; startup health poll is bounded by a deadline
- tldw_chatbook/TTS/audio_cpp_guided_launch.py, audio_cpp_package_scanner.py: binary validation, port selection, artifact creation and package scans all run in asyncio.to_thread or a settings worker thread
- tldw_chatbook/TTS/audio_cpp_config.py, audio_cpp_guided_config.py, audio_cpp_managed_config.py, audio_cpp_contract.py, audio_cpp_artifact_catalog.py, audio_cpp_artifact_dependencies.py: pure validation and projection, µs-level per call; import cost counted in F1
- tldw_chatbook/TTS/adapter_registry.py: per-acquire _freeze_configuration of a small provider projection (µs-level); reconfigure deep copies only on the settings path
- tldw_chatbook/TTS/legacy_bridge.py per-chunk path (_LegacyOperation.__anext__ with an uncontended asyncio.Lock and bytes(chunk)): lightweight
- tldw_chatbook/TTS/TTS_Generation.py orchestration: no per-chunk heavy work; catalog and voice loops only retry on generation change
- tldw_chatbook/TTS/backends model loading: kokoro (_run_native_work), chatterbox (_initialize_sync via to_thread), higgs (load_model_thread), omnivoice (to_thread ONNX sessions); inference is off-loop; kokoro ONNX uses a per-stream worker thread with its own loop (deliberate isolation, ~1 ms overhead)
- tldw_chatbook/TTS/backends/alltalk.py, elevenlabs.py, openai.py: bytearray buffering, pooled/timeout-bounded httpx; elevenlabs 1 KiB aiter_bytes chunking is minor
- tldw_chatbook/TTS/backends/chatterbox_process.py and chatterbox subprocess protocol: JSON lines over pipes, bounded buffers (base64 decode of the final clip on the loop is a few ms)
- tldw_chatbook/TTS/omnivoice_prompt.py, omnivoice_sampler.py, kokoro_pytorch.py, kokoro_languages.py: numpy/ONNX work runs inside worker threads
- tldw_chatbook/TTS/playback_capability.py: adapt_console_speech_format measured at 52-58 us per call (find_spec plus shutil.which), acceptable per request
- tldw_chatbook/TTS/backends/voice_manager_base.py, chatterbox_voice_manager.py, higgs_voice_manager.py, omnivoice_voice_manager.py: small JSON profile files; per-call reads are sub-ms (package-import residue noted in F2)
- tldw_chatbook/TTS/loose_voice_lifetime.py: whole-file reference copy bounded at 100 MiB, cold voice-import path
- tldw_chatbook/TTS/character_request_resolver.py, default_profile_request_resolver.py: go through the repository executor; no loop I/O
- tldw_chatbook/TTS/_async_lifecycle.py, audio_limits.py, audio_schemas.py, base_backends.py, legacy_catalogs.py, legacy_request_builder.py, openai_compatible_config.py, pcm_playback.py, playground_types.py, preferences.py, profile_errors.py, profile_portability.py, profile_source.py, profile_sqlite_policy.py, profile_sqlite_proof.py, omnivoice_artifact_catalog.py, migrations/v0..v4: pure data/validation; no hot-path cost beyond F1 import time

## census
| Probe (isolated HOME/XDG/TLDW_CONFIG_PATH, warm .pyc, M-series Mac) | Result |
|---|---|
| TTS modules loaded by `import tldw_chatbook.app` | 38 (incl. profile_service, TTS_Generation, audio_cpp_supervisor/recipes) |
| TTS-only import cost (ablation: non-TTS deps preloaded) | 45-52 ms (38 modules, 5 runs); 55-64 ms (31-module variant, 5 runs) |
| Top self times under ablation | audio_cpp_recipes 4.8 ms, adapter_types 3.7, audio_cpp_supervisor 2.9, profile_service 2.9, audio_cpp_guided_config 2.8, profile_types 2.5, effective_settings 2.4 ms |
| `build_default_tts_service(cfg)` | 30-44 ms per call; 2.4-2.8 ms without `get_user_data_dir` (which alone is 28-37 ms, uncached) |
| `BackendRegistry.ensure_builtins()` (first legacy TTS use) | 554-640 ms warm (3 runs), 2,224 ms first run; +887 modules; `backends.higgs` alone 528 ms (torch) |
| `import kokoro_onnx` (on loop at first Kokoro init) | 41-77 ms |
| ffmpeg mp3 encode as pydub.export runs it (temp wav, subprocess, read back) | 32 ms / 5 s clip, 60 ms / 20 s, 131 ms / 60 s (first call 133 ms) |
| `StudioTTSPreferenceStore.load(migrate=False)` (studio admission loader) | median 34.9 ms (33-43, 10 runs, default 100 KB config) |
| Pairwise bytes concat (pydub `AudioSegment.__add__` pattern) vs `b"".join` | 20x15 s: 17 vs 0.8 ms; 60x10 s: 83 vs 1.6 ms; 150x10 s: 605 vs 4.0 ms |
| `bytes +=` per chunk (audiobook) 20 MB | 8 KiB chunks: 1,696 ms vs bytearray 1.6 ms; 64 KiB: 210 vs 1.3 ms |
| `iter_normalized_pcm_frames` raw PCM, 10 s clip | 24 kHz 240 us/frame, 22.05 kHz 261, 44.1 kHz 349, 48 kHz 43; one-shot normalize is 130-150 ms per 10 s (about 2x cheaper) |


# slice-39

## summary
Slice #39 (TTS#2, 13 files, 9,072 lines) is mostly cold, security-heavy domain code. Nearly all file and SQLite I/O is already off the loop: voice-bundle file work goes through asyncio.to_thread workers, the profile repository runs on a single-thread executor, and the UI loads Studio preferences with to_thread. The slice touches five hot paths:
(1) the boot import leg. profile_types, studio_preferences, request_admission, sample_audio_validation and provider_ids cost about 7.5 ms self, pulled in by app.py's module-scope TTS imports (TTS_Generation is 62-65 ms cumulative standalone) and by console_chat_store's CharacterRef.
(2) the per-Studio-Generate admission path. This is the main finding: a forced config re-read and TOML re-parse runs on the event loop for every Speech Studio Generate click, measured at 56 ms median.
(3) audiobook text preprocessing. langdetect runs once per sentence on the loop and is auto-enabled for ElevenLabs, measured at 1.28 ms per sentence, about 0.5 s per 400-sentence chapter.
(4) the profile-repository open proof. It runs PRAGMA quick_check twice per open (once in the helper subprocess, once on the live connection). Off-loop, but it scales with clone-reference blob bytes.
(5) voice-bundle export. The bundle is encoded and hashed on the loop (P3).
No slice module has timers or always-on threads. The recovery.py adapters run only in backup and restore flows.

Two things I saw outside the slice, for other agents to check:
(a) a warm load_settings() / get_runtime_config_snapshot() cost about 14 ms per call in my isolated probe. cProfile attributes it to the Backup_Recovery config_participants guard (storage_admission acquire, about 1,900 posix.open calls per forced reload). The deep scratch-profile path may inflate this, so the config-reads and Backup_Recovery owners should re-measure it.
(b) audiobook_generator.py:621 accumulates each segment's audio with `audio_data += chunk` (quadratic bytes concatenation) on the same audiobook path as F2.
All probes ran with HOME, XDG_* and TLDW_CONFIG_PATH redirected to scratch/s39. The real ~/.config/tldw_cli/config.toml mtime was unchanged afterwards (Sep 26 07:01).

## clean areas
- tldw_chatbook/TTS/profile_store_lock.py: acquire() sleeps and polls at 50 ms, bounded to 5 s, but only on the profile repository's single-worker ThreadPoolExecutor at open and restore time (profile_repository.py:1840/1964/3417). Never on the loop.
- tldw_chatbook/TTS/provider_ids.py: a constant tuple (0.1 ms import).
- tldw_chatbook/TTS/voice_blend_paths.py: path helpers plus one atomic private JSON write. It imports only config and private_paths, which are already loaded at boot. The UI's reads of the small blend JSON belong to other slices.
- tldw_chatbook/TTS/recovery.py: Backup_Recovery owner adapters (discover/validate/capture), reached only from backup and restore flows. Hashing each WAV blob in validate is intended verification. The voice-root list is deduplicated before the tree walk.
- tldw_chatbook/TTS/voice_bundle_codec.py: bounded (40 MiB archive, 33 MiB uncompressed), linear ZIP layout parsing, chunked decompression. inspect and commit decode it inside asyncio.to_thread workers.
- tldw_chatbook/TTS/voice_bundle_service.py: every copy, fingerprint and publish runs in asyncio.to_thread (_run_worker / _run_worker_settled). The session map is capped at 4 with a 600 s TTL. The commit path's triple source re-read and hash is a deliberate TOCTOU guard and runs off-loop. The only exception is export encoding on the loop (F6). _maintenance_drain's 10 ms poll runs only during maintenance and is deadline-bounded.
- tldw_chatbook/TTS/studio_preferences.py parsing and serialization: linear and bounded. All UI load/save/reset calls already use asyncio.to_thread (STTS_Window.py:1763, speech_settings_pane.py:631/1318/1378/1415). Only the admission-coordinator loader is on the loop (F1).
- tldw_chatbook/TTS/text_processing.py: patterns are precompiled at class level, string-pattern re.sub calls hit re's cache, and chunking is linear. The normalizer cost for chat-sized input is negligible.
- tldw_chatbook/TTS/profile_types.py runtime validation: O(len) per field. Profile options must be empty by contract (_validate_provider_contract), so the canonical-JSON size check is trivial.
- tldw_chatbook/TTS/request_admission.py, apart from F1: per-request work is O(1). _WriterPreferredGate creates one task per release, which is micro.
- tldw_chatbook/TTS/sample_audio_validation.py WAV path: wav_has_complete_frames measured 0.13 ms for 8 MB. The file-based validate_playable_audio_file called by profile_service already runs via asyncio.to_thread (speech_profile_mixin.py:451).
- tldw_chatbook/TTS/profile_validation.py: row decode and metadata digest are linear. length(wav_bytes) in the digest query does not load blob content (measured 0.07 ms for 174 MB).

## census
| probe (isolated scratch profile, py3.12) | result |
|---|---|
| StudioTTSPreferenceStore.load(migrate=False) (forced reload, 102 KB default config) | 55.7 ms median / 73.9 ms max |
| get_runtime_config_snapshot(force_reload=True) | 56.2 ms median / 86.8 ms max |
| get_runtime_config_snapshot() (non-forced) | 14.8 ms median |
| load_settings() warm (cache hit) | 14.0 ms (out of slice: config_participants guard) |
| deepcopy(settings) | 0.5 ms |
| AdvancedTextProcessor SSML path, 400 sentences / 29 KB | 664 ms first call, 512 ms warm (1.28 ms/sentence) |
| AdvancedTextProcessor plain format_for_tts, 29 KB | 10.2 ms |
| import av (PyAV) | 60.6 ms |
| wav_has_complete_frames 8 MB | 0.13 ms |
| sha256 32 MB / 2.9 MB | 10.4 ms / 0.9 ms |
| PRAGMA quick_check 58 MB / 174 MB of blobs | 9.5 ms / 28-32 ms |
| boot import self: profile_types / studio_preferences / request_admission / sample_audio_validation(cum) / provider_ids | 3.3-3.8 / 1.5 / 0.8-1.1 / 1.7 / 0.1 ms |
| TTS_Generation cumulative (standalone) | 62-65 ms |


# slice-4

## summary
Slice #4 (Backup_Recovery, 80 files, ~43k lines) is not cold. Its admission and participant primitives sit under nearly every hot path in the app:
- every SQLite get_connection() and outermost transaction of ~20 installed repositories (ChaChaNotes, Media, Prompts, AgentRuns, Workspace and the rest)
- every private SQLite connect and every private-file helper
- every guarded config derivation (load_settings, get_user_data_dir)
- every LLM provider call (LLM_Calls/recovery_review)
- every Skills/Chatbooks call (local_content_lifetime.call)
- every Agents/MCP/RAG guarded entry (admission_runtime / execution_scope)
- an always-on 10 Hz maintenance monitor
- boot imports and boot-time execution_allowed calls

The single structural cause is that ADR-126 authority is re-derived from disk on every call, and no process-local generation cache exists. storage_admission.acquire_storage() re-reads the bootstrap records, the registry.json under flock and the qualification JSON. It also re-runs ctypes native_identity and walks every pinned directory from '/'. That is about 245 open() calls and 5.8 ms per call in the audit sandbox, estimated 2.5–4 ms on a real, shallower profile. This happens even though the process already holds a live startup _Hold for the same (pid, root).

Every consumer multiplies this cost:
- a ChaChaNotes outermost transaction costs 7.6 ms against about 1 µs for raw SQLite
- a warm load_settings() costs 14.6 ms and get_user_data_dir() 33.8 ms; about 18 of the latter run in TldwCli.__init__
- the Skills get_context admission costs about 20 ms on the event loop per Console visit
- each boot/scheduler execution_allowed costs about 8.5 ms on the loop
- all admissions serialize through one 10 ms-polled section (4-thread p95 61 ms)

Separately, the 10 Hz pause monitor burns about 1.6% of a core forever. Inside a transaction, each get_connection re-resolves the DB path twice (97–228 µs). _repository_types() rebuilds a 21-entry type map with 13 imports on every statement. The cold backup and restore flows re-hash whole archives 9–13 times. Backup preview classification is O(n·d²).

Highest-leverage PR: an admission fast path in storage_admission. When a ready, live hold exists and a 3–4 stat evidence fingerprint is unchanged, mint the lease from that hold and skip admission_authority. This also collapses F2, F4, F5 and F6 and the TASK-31501 tick cost.

Numbers are from the isolated sandbox. HOME there is 13 path components deep against about 5 on a real profile, and the walk-heavy costs scale with depth; the per-finding real-profile estimates allow for this. No app boot was performed.

## clean areas
- archive_models.py, limits.py, models.py, plan_records.py, profile_catalog.py, profile_paths.py (pure selectors), native_platform.py, native_files.py, effective_roots.py, space.py, service_storage.py: small helpers; they cost something only through the admission callers already reported
- group_selection.py, data_groups.py, restore_groups.py, preserved_groups.py, destinations.py: cold preview/restore planning, bounded work apart from classify_entries (F11)
- recovery_service.py: work is ThreadPoolExecutor-backed. current() is in-memory, so the backup screen's 5 Hz _refresh_status poll is cheap (and it pauses on suspend). start_recovery's synchronous status() read on the loop is a one-click P3 and was not filed
- recovered_media.py / recovered_media_messages.py: the Console image lookup is batched (200), runs via asyncio.to_thread and returns early when the catalog is absent. resolve_message_image (guarded get_user_data_dir plus an SQLite open per widget) is reachable only from ChatMessageEnhanced, which production never constructs
- runtime_producer_lifetime.py: an RLock plus a dict per call is cheap. generated_media_lifetime.py adds one acquire_storage per file path (covered by F1)
- age_worker.py / crypto.py: streaming 64 KiB pipes; one subprocess per transform by design (only the redundant helper self-test is flagged, F12)
- capture.py / capture_service.py / storage_admission.copy_capture_file: streaming 1 MiB copies with bounded memory (only the extra hash pass is noted, F10)
- sqlite_validation.py: PRAGMA quick_check only on restore candidates (cold, expected)
- journal.py, publication.py, replacement.py, later_rollback.py, staging.py, isolated_restore.py, launcher.py, recovery_copies.py, recovery_files.py, recovery_restart.py, inert_extraction.py: cold restore/rollback flows; the only issues found are repeated whole-archive hashing (F10) and the journal import on the boot path (F9)
- credentials.py / rollback_credentials.py / credential_policies.py: keyring reads only during backup/restore (cold)
- config_adapter.py, rag_inventory.py, rag_definition_participant.py, rag_projection_validation.py, rag_projection_lifetime.py, rag_indexing.py, owner_registry.py: cold backup adapters
- dictionary_file_participants.py, dictionary_source_job.py, settings_file_participants.py, async_file_participants.py, mcp_source_participants.py, persona_visual_participants.py, visual_identity_participants.py, chat_source_participants.py, raw_participants.py, config_participants.py: no independent hot loops. Their per-operation cost is the acquire_storage handshake (F1/F2). The lock.acquire(timeout=0.05) waits are cancellation checks, not spins
- admission.py: the 10 ms flock poll loops run only while a maintenance gate is contended (cold). Its per-call cost is through pause_requested (F3) and admission_authority (F1)
- unsaved_editors.py, runtime_maintenance.py (everything except monitor_app's idle loop), participants.drain / _retire_current_thread_caches: run only during a backup pause
- profile_open.py: acknowledge_mounted is a no-op unless a recovered profile is selected, and it uses to_thread
- activation.py / generation_witnesses.py / control_records.py / bootstrap.py / qualification.py: pure readers, costly only through the callers reported in F1/F6/F13
- __init__.py (docstring only), __main__.py: no eager submodule imports

## census
All figures were measured in the isolated sandbox: HOME 13 path components deep (against about 5 on a real profile), with an unbound profile. Walk-heavy costs scale with path depth, so real-profile costs are estimated at about 40–70% of these.

| Primitive / consumer | Sandbox cost | Notes |
|---|---|---|
| `storage_admission.acquire_storage()+close` | 5.75 ms, 245 `open()` | 2× startup_permission, admission_authority, qualified_for, 2× _scope |
| ChaChaNotes outermost `with db.transaction()` + SELECT 1 | 7.59 ms (raw sqlite BEGIN/SELECT/COMMIT 0.001 ms) | 1 acquire_storage, 6 `_repository_types()` |
| `execute_query("SELECT 1")` outside txn | 12.1 µs (6.3 µs participant tax; 4.35 µs from rebuilding `_repository_types`) | 3 `_repository_types()` + 3 global RLock |
| `execute_query` inside txn | 97–127 µs (raw cursor 1.5 µs) | 2× `_Operation.check` → `Path.resolve()` + stat |
| nested `transaction()` inside txn | 228 µs | 4× path checks |
| warm `load_settings()` | 14.6 ms | 2 acquire_storage |
| warm `get_user_data_dir()` | 33.8 ms | 5 acquire_storage; ~18 call sites reachable from `TldwCli.__init__` |
| `get_runtime_config_snapshot()` / `get_model_cache_dir()` | 11.1 / 31.8 ms | 2 / 6 acquire_storage |
| warm `get_cli_setting()` | 0.001 ms | already fixed by TASK-32804.1 |
| `activation.execution_allowed(owner, path)` | 8.5 ms | |
| `local_content_lifetime.operation(5 paths)` (Skills `@content_call`) | 19.7 ms (1 path: 4.7 ms) | |
| `LLM_Calls.recovery_review._Operation` per provider call | 3.9 ms | plus load_settings 14.6 ms |
| `_local_pause_requested()` probe | 1.25 ms wall / 1.18 ms CPU, 33 `open()` | |
| `monitor_app`-style loop at 10 Hz | 1.55–1.61% of a core (bare sleep loop 0.07%) | |
| `acquire_storage` under 4 threads | p50 17.3 / p95 61 / max 83 ms (single thread p50 7.0) | 3.46× wall for 4× the work |
| `installation_client_id()` at config import | 24.1 ms (journal.py 17.5 ms self import, archive_models 4.1 ms) | |
| first `_repository_types()` in a fresh process | 211 ms, 487 modules imported | |
| `crypto.helper_capability()` | 56.7 ms | spawns `python -I age_worker.py info` |
| `inventory.classify_entries` | 2.5k items 0.29 s; 7.6k 1.10 s; 23k 4.15 s | 2.25M `Path()` constructions at 23k |
| `ctypes.CDLL(None)` + argtypes | 25 µs per call | `native_identity` 21.6 µs; `_qualified_identity` 36.8 µs |


# slice-40

## summary
Slice #40 "Tools" (36 files, ~29.7k lines). Almost all of this code runs on agent worker threads or in subprocesses, not on the Textual event loop. The hot paths are therefore per agent tool call (Console fs_*/git_*/fs_patch through LocalToolProvider -> WorkspaceToolExecutor, plus the builtin read_file/list_directory/write_file/glob/grep), per raw-CLI command (user `!cmd` and model raw_cli), per web_fetch/deep-search URL, and per send when the draft has @-references. The on-loop callers I traced are safe: chat_screen's watchlists receipt poll uses to_thread, runs every 2 s and stops when idle; @-reference expansion and the preview-snapshot builders run in to_thread. The two small exceptions are the quit-time SSH import (F8) and eager module imports (F9).

Main finding (F1): the whole slice assumes one resolve_sensitive_context() per tool call is cheap. Its docstrings say "~11 config accessors". That assumption no longer holds. Each call runs config.get_user_data_dir() 19 times with no memoization. Since #2495 (2026-09-07) and TASK-32628 (2026-09-16), each of those calls takes a cross-process file lock (portalocker with a 50 ms sleep retry) and runs about 1,600 open() calls. Measured in an isolated profile: 36 ms per get_user_data_dir and 540 ms per resolve. One end-to-end fs_read of a 2-line file took 700-750 ms. With two concurrent tool resolves running, a third get_user_data_dir caller hits p90 355 ms. That third caller can be any of the 56 get_user_data_dir call sites in UI/, Widgets/, app.py and Chat/. The config/Utils slice should own the root fix in get_user_data_dir; the per-run snapshot belongs in Tools.

Other issues: fs_grep scans the whole tree with no early exit (6-9 s on the repo for max_results=20). Every raw-CLI command spawns a child that re-imports about 95 ms of modules, and re-runs the whole app module when the app was started with `python -m tldw_chatbook.app`. Every fs/git call spawns a fresh `-I -m` interpreter (about 42 ms), while the builtin grep worker's lean launch pattern costs about 15 ms. The raw-CLI output sanitizer walks every character in Python at 54 ms/MB and keeps going after both output caps are full. Smaller P3 items: web_fetch builds a new httpx client per call, the Watchlists result fit is O(n^2), quit imports the SSH stack on the loop, and TASK-23112's raw-CLI import deferral does not hold on the Console path.

Structural notes: remote_worker_bundle.py is a generated flat copy of local_tool_impls, git_tool_impls, patch_tool_impls and sensitive_paths, so the F2 fix must be regenerated into it. There are two grep implementations: the builtin GrepFiles has early-break and killable batches, the Console fs_grep has neither. Startup import is fine: in an import-time trace the Tools modules add only a few ms to chat_screen's import, because the heavy Scheduling and Subscriptions dependencies are already imported by app.py.

## clean areas
- tldw_chatbook/Tools/__init__.py - PEP 562 lazy re-exports; only tool_executor (~0.8 ms) is eager
- tldw_chatbook/Tools/tool_executor.py - calculator bounded before the result is built; no hot-path cost
- tldw_chatbook/Tools/remote_binding_status.py - pure in-memory cache with no I/O; UI chip reads only hit the cache
- tldw_chatbook/Tools/remote_root_types.py, remote_sensitive_paths.py - pure data/types
- tldw_chatbook/Tools/remote_workspace_transport.py - ControlMaster reused per host, control dir resolved once under a lock, per-host semaphore, no retries
- tldw_chatbook/Tools/remote_workspace_executor.py - _bundle_payload lru_cached; `ssh -G` runs only on the identity-capture path; recovery probe debounced
- tldw_chatbook/Tools/remote_binding_locator.py - `ssh -G` only at binding add/capture, bounded timeout
- tldw_chatbook/Tools/build_remote_worker_bundle.py - build-time tool; runtime only imports expected_bundle_stamp
- tldw_chatbook/Tools/remote_worker_bundle.py - runs on the remote host only (inherits the F2 grep pattern via regeneration)
- tldw_chatbook/Tools/virtual_cli_impls.py - argparse parsers built once at module scope; dispatch goes through the pinned executor
- tldw_chatbook/Tools/document_expansion_tool.py - bounded fetch, include_image_data=False, lazy DB accessors
- tldw_chatbook/Tools/note_management_tools.py - per-user DB cached (task-692 fix still in place)
- tldw_chatbook/Tools/rag_search_tool.py, web_search_tool.py - lazy, not on the boot path
- tldw_chatbook/Tools/character_tool_service.py - per-char json.dumps in _page_end is bounded (0.4 ms/field card view, <=12k chars per page)
- tldw_chatbook/Tools/watchlists_command_service.py - handlers run on agent worker threads only; memberships batched
- tldw_chatbook/Tools/workspace_file_roots.py - registry memoized with a per-thread WorkspaceDB connection; per-call cost is 2 small queries plus a few stats (sub-ms); workspace_context_note is per run
- tldw_chatbook/Tools/workspace_tool_protocol.py, workspace_wire_decode.py, workspace_root_pin.py, workspace_tool_dispatch.py, workspace_tool_worker.py, worker_watchdog.py - worker-side, bounded, one-shot
- tldw_chatbook/Tools/_grep_worker.py - lean `-S -P` script launch, streamed line reads, early break at max_matches
- tldw_chatbook/Tools/patch_tool_impls.py - resolves the sensitive context once per multi-file apply (cost comes from F1, not from this file)
- tldw_chatbook/Tools/code_audit_tool.py, file_operation_hooks.py - dead code but never imported in production, so no runtime cost
- Polling: no set_interval/set_timer in the slice; raw-CLI 50 ms queue polls run only while a command executes
- UI-side callers traced: chat_screen._poll_console_watchlists_operations (to_thread, 2 s, stops when idle), console_references expansion (to_thread), preview snapshot builders (to_thread), settings SSH probe (thread worker)

## census
| Probe (isolated profile, audit tree @840ed2ca58, Py3.12) | Result |
|---|---|
| config.get_user_data_dir() steady state (15-component data path) | 36 ms/call, ~1,590 posix.open per call |
| resolve_sensitive_context() | 540 ms/call (19 x get_user_data_dir; _sensitive_db_paths alone 475 ms) |
| WorkspaceToolExecutor.execute('fs_read') on a 2-line file | 693-754 ms end-to-end (_call_context accounts for most of it) |
| get_user_data_dir latency with 2 concurrent tool resolves | p50 63 ms, p90 355 ms, max 368 ms |
| `python -c pass` / import workspace_tool_worker (per fs/git call spawn) | 20 ms / 42 ms |
| spawn Process, stdlib target / raw_cli_executor target | 25 ms / 113-128 ms |
| _grep_relative_files 'TODO', max_results=20, whole repo (26k files) | files mode 9.05 s, content mode 5.81 s |
| _grep_relative_files 'import', max_results=5, tldw_chatbook/ | 1.04 s (_relative_target_is_safe 45% of profile) |
| _StreamSanitizer / _OutputAccumulator after both caps full | 54 ms per MB |
| httpx.Client() vs Client(verify=cached ctx) | 10.2 ms vs 0.58 ms |
| import remote_workspace_transport (incremental after config) | ~34 ms |


# slice-41

## summary
Slice #41 is the 27 top-level modules directly under tldw_chatbook/UI/ (26,297 lines). Despite the "hot" label, most of it is secondary screens: Logs, LLM/Models, Speech Lab (STTS + dictation + voice cloning + profile library), Study, Research, Writing and Chatbooks. The only code on the boot and Console hot path is five small modules: stable_command_palette, console_command_provider and image_gen_command_provider (imported by app.py at module scope), plus character_display_text and destination_recovery (pulled in at boot via Chat.console_display_state and Console widgets). Every screen module in the slice is otherwise lazy: the screen registry loads it, and the post-paint background thread pre-imports it with pacing. So module-scope imports in those windows cost only GIL contention after first paint.

Measured headline items:
1. **wcwidth on the boot path (F1).** character_display_text imports wcwidth 0.8.2 only for a `wcwidth(ch) < 0` test. That test is fully covered by its own `unicodedata.category in {Cc, Cf, Cs}` check: 0 of 1,114,112 code points differ. The import costs 8–14 ms warm on the pre-paint `import tldw_chatbook.app` path.
2. **Chatbooks constructor blocks the loop (F2).** Every Chatbooks visit (and every "Manage exports" open) calls `get_private_chatbooks_dir()` in a constructor on the event loop. That calls `config.get_user_data_dir()`, which is not memoised and does roughly 1,000–2,000 `open()` syscalls through storage-admission and private_paths verification. Measured 35.6 ms per call. TASK-1320 moved the Chatbooks scan into a thread but missed this constructor. The root cause is cross-cutting: 203 call sites of `get_user_data_dir()` app-wide. The config / Backup_Recovery slice owners should look at it.

Everything else is P3: per-record synchronous Logs rendering (73 µs per record, measured), the quadratic dictation transcript redraw, eager construction of the LLM screen's panes (7–21 ms per visit), the Speech Lab mount spin-wait, the unbounded Research event log, the Chatbooks card remounts, and file I/O on the loop in voice cloning and audiobook.

The structural item: TASK-32807.4 deletes the dead Tools_Settings_Window (6,928 lines), but that orphans three more modules (1,554 lines) that the TASK-32807.7 census does not list. The whole tree has zero runtime cost because nothing imports ToolsSettingsScreen at runtime.

Correctness side-finds (out of efficiency scope, but they double work and need tasks):
- **Dictation Start does start-then-stop.** STTSWindow.on_button_pressed (STTS_Window.py:2506) re-dispatches every unhandled Button.Pressed to `content.children[0].on_button_pressed`. ImprovedDictationWindow.on_button_pressed never calls `event.stop()`, so it runs twice per press. Start Dictation therefore starts, and on the second dispatch sees `is_dictating` and stops. Voice cloning's async handler is re-called but never awaited.
- **Widgets mutated from worker threads.** PersistentLogHandler.emit (app.py:11767) calls LogsWindow.append_record on whichever thread logged. The dictation callbacks (dictation_service_lazy.py:1627) call TextArea.load_text and query_one from the DictationProcessor thread.

Isolation note: every probe ran with scratch HOME, XDG_* and TLDW_CONFIG_PATH, plus TLDW_TEST_MODE=1. A LogsWindow unmount wrote its filter state to the scratch config.toml only. The early probes, run before I set PYTHONPYCACHEPREFIX, left gitignored `__pycache__` directories in the audit tree; no tracked file was touched.

## clean areas
- tldw_chatbook/UI/__init__.py: a single Static.renderable shim; trivial at boot
- tldw_chatbook/UI/stable_command_palette.py: one query_one per palette nav key; 0.5 ms import
- tldw_chatbook/UI/console_command_provider.py: static tuple; 0.5 ms import
- tldw_chatbook/UI/image_gen_command_provider.py: demo screen and PIL imported lazily inside search()
- tldw_chatbook/UI/focus_ownership.py: pure helper
- tldw_chatbook/UI/tools_settings_messages.py: one Message class
- tldw_chatbook/UI/tts_profile_recovery.py: static projection table
- tldw_chatbook/UI/server_chatbook_service_lease.py: reuses app.server_chatbook_service, so no per-request client in normal runs
- tldw_chatbook/UI/destination_recovery.py: pure dataclass/copy helpers; 2 ms boot import
- tldw_chatbook/UI/stts_playground_catalog.py: pure catalog projection, no I/O
- tldw_chatbook/UI/character_display_text.py runtime cost: 17 us per 180-char label (measured); only the boot import is a finding (F1)
- tldw_chatbook/UI/stts_profile_library.py: DataTable paged at 50 rows, 0.25 s search debounce, service calls async, export via asyncio.to_thread, maintenance drain is bounded
- tldw_chatbook/UI/Writing_Window.py: every controller call goes through asyncio.to_thread (writing_controller.py:43)
- tldw_chatbook/UI/Study_Window.py: only per-view-switch remount of small form widgets; DB work is in Study_Modules (other slice)
- tldw_chatbook/UI/Research_Window.py: run list capped at limit=25; scope service runs its local backend on a dedicated executor; 2 s interval returns immediately unless a local non-terminal run is selected, and dies with the non-reusable screen
- tldw_chatbook/UI/LLM_Management_Window.py: Ollama probe is async, 0.25 s capped, exclusive worker, gated on screen.is_active; managed-GGUF inventory runs on a thread worker
- tldw_chatbook/UI/Logs_Window.py: filter text debounced 0.2 s, regex compile cached, render capped at 1000 lines, level counts incremental, config persist debounced and off-loop
- tldw_chatbook/UI/Chatbooks_Window_Improved.py and ChatbookExportManagementWindow.py: directory scan and manifest preview already use asyncio.to_thread (TASK-1320/15471); search debounced
- tldw_chatbook/UI/STTS_Window.py: Studio preference load via to_thread; audiobook chapter detection on a debounced @work(thread=True) worker; VoiceCloningWindow imported lazily
- tldw_chatbook/UI/Voice_Cloning_Window.py: profile listing and profile creation already use asyncio.to_thread
- tldw_chatbook/UI/Tools_Settings_Window.py, Sharing_Panel.py, Outputs_Panel.py, ChatbookCreationWindow.py: never imported at runtime (only via the unused lazy Screens.ToolsSettingsScreen export), so zero runtime cost; see F10

## census
| module | lines | runtime reach | boot import? |
|---|---:|---|---|
| Tools_Settings_Window.py | 6928 | dead (only Screens.ToolsSettingsScreen lazy export, never accessed) | no |
| stts_profile_library.py | 3726 | Speech Lab ▸ Profiles; personas_screen imports at module scope | no (background pre-import) |
| STTS_Window.py | 2570 | Speech Lab (stts route) | no |
| LLM_Management_Window.py | 2403 | llm route | no |
| ChatbookExportManagementWindow.py | 1213 | Chatbooks ▸ Manage exports modal | no |
| Study_Window.py | 1163 | study route | no |
| Dictation_Window_Improved.py | 1051 | Speech Lab ▸ Dictation (imported eagerly by STTS_Window) | no |
| Research_Window.py | 1010 | research route | no |
| Voice_Cloning_Window.py | 870 | Speech Lab ▸ Voice cloning (lazy) | no |
| Chatbooks_Window_Improved.py | 823 | chatbooks route | no |
| Logs_Window.py | 740 | logs route (+ MAX_LOG_RECORDS read at boot, function-local) | no |
| Outputs_Panel.py / Sharing_Panel.py / ChatbookCreationWindow.py | 594/563/397 | dead (orphans of Tools_Settings_Window) | no |
| stts_playground_catalog.py | 490 | Speech playground | no |
| ChatbookTemplatesWindow.py | 418 | Chatbooks ▸ templates modal | no |
| destination_recovery.py | 416 | many screens | yes (2.0 ms) |
| Writing_Window.py | 322 | writing route | no |
| character_display_text.py | 149 | Console display state / switcher / pickers | yes (8–14 ms incl. wcwidth) |
| console_command_provider.py / image_gen_command_provider.py / stable_command_palette.py | 148/30/17 | app.py module scope | yes (<0.6 ms each) |
| server_chatbook_service_lease.py / tts_profile_recovery.py / focus_ownership.py / tools_settings_messages.py / __init__.py | 80/71/60/40/5 | helpers | __init__ only |


# slice-42

## summary
Slice #42 (UI/Console_Modules part 1, 32 files, about 30k lines) is made of controller and region modules that ChatScreen builds and calls from almost every Console path. The dominant hot path is the Console sync tick, `ChatScreen._sync_native_console_chat_ui` (chat_screen.py:18645). It runs every 0.2 s from `_poll_transcript` for as long as any run is active, which covers all streaming. It runs up to 10 Hz from the realtime voice tick when the transcript is dirty, and back-to-back while raw-CLI output streams. It also runs on every session switch, send, or resync, and 0.2 s after every typing pause via ConsoleDraftSpendRefresh. On each call it fans out into this slice with no gates: message.reconcile_console_speech_context, image._reconcile_h3_*, retrieval warm/summaries (guarded), character._refresh_active_character_avatar_if_scope_changed, character_context.refresh_if_scope_changed, left_rail.sync_model_recovery, agent._console_agent_section_payload, library_activity.sync_projection, the conversation-browser build's agent._console_subagent_counts_for_rows, and image._build_console_image_specs.

Five legs of that tick do real work on every tick even though their inputs rarely change:
- F2: a whole-screen relayout from `Static.update`, plus the rail allocation-reconcile query storm.
- F3: an AgentRunsDB COUNT on the event loop.
- F4: 2 to 4 private-SQLite round trips for the Character fingerprint, awaited before the transcript is painted.
- F6: a copy of the whole transcript for the avatar.
- F8: a reload of the recovered-media catalog.

The structural fix is to split the tick into a streaming fast path (transcript plus the activity line) and a rails slow path driven by dirty flags or revisions. That split removes most of this cost as a group.

The other hot entry points:
- **First Console compose (P0, F1):** it builds the agent runtime synchronously before first paint.
- **Per-keystroke paths:** the Character search (F5) is undebounced, recomposes, and makes 5 DB round trips. The composer's price availability check (F12) is cheap and bounded.
- **Background timers:** a 2 Hz left-rail progress poll (F11), 1 Hz dictation, and 0.1 s voice ticks that only run during sessions.

One cross-cutting multiplier sits outside the slice but was measured through it (F10). Every ChaChaNotes and AgentRuns transaction pays the storage-admission handshake: about 245 open() calls and 3 to 4.4 ms per tiny transaction. `run_owned_db_call` also closes any per-thread connection it opened, so a call can pay a fresh helper-process connect of about 50 ms. This is what turns 'one small SELECT per tick' into 10 to 60 ms.

Well built in this slice: environment polling (10 s, gated, threaded, TTL), retrieval summaries (scope-guarded, to_thread), prompt-queue and dispatch-recovery regions (equality-guarded), the right-rail scroll and geometry split, run-log probing (thread worker plus cache), avatar prerender off the loop, and raw CLI running on a worker thread with a bounded preview.

## clean areas
- UI/Console_Modules/environment.py - 10 s poll gated on rail-open/root, git/gh on thread workers, net TTL + backoff; landing is cheap
- UI/Console_Modules/retrieval.py - dictionary/world-book summaries scope-change guarded and run via asyncio.to_thread; scope read/write off-loop (asyncio.run-per-call helper runs inside to_thread only on scope change - negligible)
- UI/Console_Modules/dispatch_recovery.py - recompose only when the immutable projection changes
- UI/Console_Modules/prompt_queue.py - ConsolePromptQueueRegion.sync_presentation key-guarded; controller is body-free projections
- UI/Console_Modules/library_activity.py - token-guarded projection (active_path tuple per tick is O(n) but trivial)
- UI/Console_Modules/dictation.py - 1 Hz elapsed timer only while recording; blocking native calls via to_thread; maintenance 20 ms sleep-poll only during maintenance pause
- UI/Console_Modules/hands_free.py, realtime.py - 0.1 s ticks exist only during voice sessions; tap.stop/settle off-loop (note realtime drives full native sync up to 10 Hz, amplifying F2/F3/F4/F6)
- UI/Console_Modules/raw_cli.py - command runs in a thread worker; 32 KB bounded preview; projection coalesced (minor threading.Timer churn, F14)
- UI/Console_Modules/archive.py - storage_call off-loop, resume/archive per click
- UI/Console_Modules/review_selection.py - annotations/trajectory reads via to_thread / thread worker
- UI/Console_Modules/right_rail.py - outer reconcile coalesced, pure-scroll path split from geometry (TASK-21117), fold hint painted with layout=False; ConsoleSelectedTurnActivity recompose equality-guarded (minor focus-path redundancy F13)
- UI/Console_Modules/agent.py run-log probe - thread worker + cache + retry_at; historical_snapshot cached per conversation by the bridge
- UI/Console_Modules/fleet.py - unseen ids cached by revision; survivor tick only while a drain is owed
- UI/Console_Modules/image.py - H3 edit completions reconcile is O(completions); generation/edit prep and remote fetch off-loop; save paths via to_thread (F8 is the one per-tick issue)
- UI/Console_Modules/message.py - message actions are per-click; save-image fetch/write off-loop; rehydrate helpers are dead code with no production caller (no runtime cost)
- UI/Console_Modules/send_price.py - expensive tooltip derived on demand only (TASK-23018); keystroke path is availability only (F12 minor)
- UI/Console_Modules/console_spend_projection.py, provider_continuation_recovery.py, rail_section_layout.py, research_command.py, reaction_preview.py, conversation_token_preparation.py, capture_policy_bindings.py, library_policy.py, frame.py, character_avatar_layout.py, __init__.py - pure projections / thin bindings / off-loop helpers; no hot-path cost found
- UI/Console_Modules/character.py - avatar resolution and card fetches via to_thread; request-key guard prevents repaint churn (F6 is the per-tick transcript copy)
- Slice module-scope imports: no heavy third-party imports; dictation.py is merely the first edge into the eager Widgets/Console package (678 ms cumulative), which the Console needs at first paint anyway - owned by the Widgets slice / TASK-22213/22504

## census
| Leg of the 0.2 s Console sync tick (`_sync_native_console_chat_ui`) that lands in this slice | Gate | Cost per tick | Basis |
|---|---|---|---|
| character_context.refresh_if_scope_changed (F4) | none (not gated on Character section or character presence) | 2 `run_owned_db_call` = 4 admission-taxed txns; 9.9 ms/call with a warm thread connection, 61 ms/call on a thread without one. Awaited before the transcript paint | measured |
| left_rail.sync_model_recovery + allocation reconcile (F2) | none | 1 whole-screen `_refresh_layout` per tick; +8 ms CPU/tick (rail + 300 filler widgets) to +20 ms CPU/tick (700 fillers) | measured in a harness |
| agent._console_subagent_counts_for_rows (F3) | "run active", which forces a refresh every tick | 1 AgentRunsDB COUNT query on the loop: 9.9 ms | measured |
| agent._console_agent_section_lines, drilled in (F7) | only while drilled into a sub-agent | `get_run` with full step hydration on the loop: 3.9 ms (100 steps) | measured |
| character._current_request (F6) | react_character_expressions (default True) | full transcript replace-copy: 4.1 us/message (0.8 ms at 200 messages, 4 ms at 1000) | measured |
| image._recovered_console_image_specs (F8) | only "no task in flight", ignores an unchanged selection | worker thread: `exists()` when no catalog; helper connect + schema validation + O(n) query (~50-80 ms) when the catalog exists | estimated |
| retrieval / library_activity / image H3 / speech reconcile | revision or scope guarded | <0.5 ms | estimated |
| Once per app session: first Console compose builds the agent bridge (F1) | none | AgentRunsDB open 77-117 ms + ChangeTurnTracker 57-70 ms + bridge import 24 ms, all on the loop before first paint | measured components |


# slice-43

## summary
Slice #43 (UI/Console_Modules#2: session.py 5.7k, workspace.py 7.9k, wiring.py 2.3k, video.py 1.5k, terminal.py, skill.py, transcript.py, status_row.py, worktree.py, trace_call_recovery.py) is the Console's controller layer. Every file sits on the Chat/Console boot import leg (wiring imports all controllers, 37 ms cumulative). The hot paths I found and traced:
(a) The native sync pass `ChatScreen._sync_native_console_chat_ui`, which runs on a 0.2 s interval (chat_screen.py:18947) for every streamed run and on every other sync. From it, `_sync_native_console_transcript` calls `video._build_video_card_specs` and `workspace._build_console_workspace_context_state`.
(b) The per-send path. `prompt_queue._stage_normal_chain`, an async def on the event loop, calls `wiring._admit_console_turn_to_runtime`, which calls `session._build_console_turn_execution_context` synchronously. The same builder is reached from the queue path, the send-price tooltip, and the cost chip.
(c) Per keystroke into the Terminal workspace: `terminal.ConsoleTerminalController.send_key`.
(d) Per Console visit/resume: the skill-candidate refresh and the registry reconcile.
(e) Session-tab and workspace switches.
Overall health is better than a module this size suggests. The workspace rail already has a generation-keyed registry read cache, one build per tick, a memoised active-workspace lookup, a debounced token-cancelled search, a TTL persisted-rows cache refreshed by a worker, and off-loop folder availability. Most session DB work already goes through asyncio.to_thread.
The remaining costs cluster in four places:
1. MEASURED: the generated-video card path runs the Backup_Recovery storage-admission machinery on the event loop for every video message on every tick. That is 19 ms per video message, or 215 ms when a recovered-media catalog exists. The sibling image path was already moved to a worker; video never was.
2. MEASURED: the Terminal controller snapshots every session's whole screen on every keystroke and every output frame. That costs 3.6 ms at 80x24 and 17 ms at 211x44, per session per snapshot.
3. The full turn-configuration capture runs synchronously on the loop per send. It includes a skill catalog trust scan whose per-skill directory hashing TASK-32921 left in place, MCP inventory hashing, and several DB reads. The cost chip also re-runs this whole capture at 5 Hz while streaming in any conversation with attachments, even though that path only reads provider_selection and capabilities from it. A prior review had retired this path as off the tick, but it missed the attachment branch.
4. Registry reads and writes still run on the loop during workspace-tree paging and tab/workspace switches.
Structural notes: workspace.py (7,949 lines) and session.py (5,667 lines) are god controllers loaded at first paint. Between them they hold 12 methods with zero production references (3 are referenced nowhere at all). The slice also has three independent memo mechanisms keyed on registry mutation_generation. Terminal/pyte is correctly deferred behind `_DeferredConsoleTerminalController`; importtime confirms it stays off the boot path.
Out-of-slice evidence for the orchestrator: every `connect_private_sqlite` spawns a helper subprocess. I measured about 49 ms per connection, in DB/private_sqlite.py:1377 `prepare_in_helper`. Every `generated_media_lifetime.participant.operation` costs about 7-10 ms (roughly 245 `os.open` calls in `storage_admission.acquire_storage`). Any hot-path caller of either inherits that cost.

## clean areas
- tldw_chatbook/UI/Console_Modules/status_row.py: pure config reads plus one guarded move_child; the collapse persist runs on a worker thread
- tldw_chatbook/UI/Console_Modules/worktree.py and trace_call_recovery.py: thin adapters; recovery listing is awaited and the dialog is pushed on demand
- tldw_chatbook/UI/Console_Modules/transcript.py: ConsoleChangeReviewProjection caches marker blocks keyed on (conversation, publication revision); inject_resume_agent_markers returns early when there are no blocks; reading-state capture/restore is O(1)
- tldw_chatbook/UI/Console_Modules/skill.py command handling (/skills list, blocked-match, pending install/script decisions); only the mount/resume refresh path is flagged
- workspace.py _ConsoleRegistryDisplayReads (generation-keyed registry display cache) and ConsoleTickWorkspaceBuilds (one build per tick): verified working
- workspace.py _resolve_console_active_workspace_id / _current_console_workspace_context: memoised, read-only on the keystroke path
- workspace.py rail search (transition_browser_search / transition_workspace_tree_search): 0.2 s debounce timer plus token cancellation (the GOOD pattern)
- workspace.py persisted-rows TTL cache with exclusive worker refresh; _manual_unread_for_rows batches misses into a to_thread worker
- workspace.py folder availability (_request_workspace_files_availability_refresh) runs off-loop and is coalesced; display_state no longer stats disk on the loop
- workspace.py workspace switcher/rename/archive/restore/star/appearance/delete/state writes all go through storage_call or asyncio.to_thread; the workspace-files visit resolution runs off-loop
- workspace.py _fetch_workspace_rows / _persisted_console_browser_rows: list_conversations offloaded, bounded by CONSOLE_CONVERSATION_BROWSER_RESULT_LIMIT
- session.py _default_console_session_settings cross-pass memo, _ensure_active_console_session_settings per-pass memo, pristine-defaults (session, config) memo, and title lookups gated on creation (TASK-26839/32804.6)
- session.py _sync_console_session_draft derives blank defaults unconditionally, but I measured 27 us/call, so not worth fixing
- session.py fork (_run_fork_io), promote-temporary, activity acknowledgement, manual-read tokens, and visual-identity picker/preview: all to_thread or single-flight off-thread
- session.py project-instruction display refresh: authority resolved via to_thread with in-flight plus TTL dedupe
- session.py store.sessions() scans: a list copy of a handful of open tabs; cheap
- wiring.py _DeferredConsoleTerminalController keeps Terminal/pyte off the boot import (confirmed with -X importtime); raw-CLI projection is coalesced and the runtime throttles repaints with a threading.Timer
- wiring.py build_console_controllers: about 23 controllers built from late-binding lambdas with no eager I/O (per-visit re-mint cost is TASK-24452)
- video.py generate/save-copy/regenerate/stream/play paths run via asyncio.to_thread with shielded finalisation

## census
| Probe (isolated scratch profile, Py3.12 / Textual 8.2.8) | Result |
|---|---|
| `TerminalScreenModel.snapshot()` 80x24 | 3.6 ms/call |
| `TerminalScreenModel.snapshot()` 211x44 | 17.2 ms/call |
| snapshot equality (fingerprint compare) | 0.01-0.03 ms |
| `VideoStore.resolve_state` with no recovered catalog | 19.4 ms per video message (cProfile: 2x `acquire_storage`, about 245 `posix.open` each) |
| `VideoStore.resolve_state` with a recovered-media catalog | 215 ms per video message |
| `RecoveredMedia(root)` + `resolve_reference` | 143.8 ms; `resolve_reference` alone 49.5 ms (`connect_private_sqlite` spawns a helper subprocess) |
| `get_character_card_by_id` with a 300 KB image blob | 0.156 ms vs 0.044 ms for a name-only SELECT |
| `blank_console_session_settings(cfg)` | 27 us (not a finding) |
| `-X importtime` chat_screen: wiring cumulative | 37.4 ms; slice-owned modal imports about 5.4 ms (switcher 2.8, skills-import 1.3, reaction picker 0.7, video capacity 0.6) |
| Methods with zero production references | workspace.py 7 (3 referenced nowhere, tests included), session.py 5 |


# slice-44

## summary
Slice #44 UI/Evals (13 files, 9,329 lines). The Evals screen is a Lab route and is not reusable, so each visit builds a new EvalsScreen. The screen imports every widget in this slice. The hot paths are:
(1) visit: on_mount -> LabScreen._populate_regions -> LibraryRail.compose, which runs before first paint;
(2) each rail click: select() -> run_worker(coroutine) _swap_selection_regions, which rebuilds the body and inspector (and the rail when rail_dirty) by calling the widgets' compose();
(3) rail section toggle, Save/Revert/Run-complete/create/import (all rebuild the rail);
(4) arrow keys in ResultsGrid;
(5) bench runs, which do per-cell persistence.

Every EvalsDB read on these paths is synchronous on the event loop. No widget or view-model read goes through a thread. The view model (EvalsViewModel) has no memo, so each selection repeats the same reads 2-3 times. The worst of them, run_groups(), JSON-decodes up to 500 run snapshots and then full-scans eval_results, running json_extract over every logprobs blob.

Measured in isolated scratch DBs:
- Run-group click: 33 ms (light, 500 cells), 206 ms (moderate, 12k cells), 830 ms (heavy, 60k cells).
- Bench click: 52 / 133 / 456 ms.
- Rail DB reads per visit or rebuild: 33 / 134 / 405 ms, plus 230-924 ms to remount up to 1,500 uncapped Button rows.
- Section collapse is a whole-rail recompose: 247 / 670 ms.
- Dataset click: SnippetEditor mounts 4 widgets per snippet, 279 ms at 200 snippets and 1.34 s at 1,000.

A cross-slice amplifier sits under all of this. Every EvalsDB `with self.connection()` goes through Backup_Recovery storage admission (`_core_transaction` -> `acquire_storage`), which costs about 3.9-4.7 ms and 245 open() syscalls per operation, against 0.006 ms for the raw sqlite execute. So even a tiny install pays about 3-4 ms per DB call; a bench click makes 16 calls. The same wrapper is on ChaChaNotes `transaction()`, Media, Prompts and 6 other stores.

There are no timers or polling loops in the slice apart from CardPicker's 0.2 s search debounce. Nothing is on the boot import path.

Structural themes, in suggested PR groups:
- PR-A: move the selection view state to a single threaded, memoised read per swap (F1, F4, F8, F6).
- PR-B: patch the rail in place instead of recomposing it (F2, F7, F12-rail).
- PR-C: virtualise or cap the per-row widget lists (SnippetEditor, CardPicker, ClassicTaskDetail) (F5, F11).
- PR-D: hygiene (F9, F10, F13, F14).
- F3 belongs to the Backup_Recovery/DB owner and should be deduped against the DB-slice audits.

Scratch probes are in /private/tmp/claude-501/-Users-macbook-dev-Documents-GitHub-tldw-chatbook/15c5e43c-3dea-41a4-87ab-3b132387df58/scratchpad/audit/scratch/slice44/ (bench.py, rail.py, mount.py, grid.py, apply.py, prof.py, prof2.py). All ran with HOME, XDG_* and TLDW_CONFIG_PATH pointed into that scratch directory; only scratch paths were written.

## clean areas
- tldw_chatbook/UI/Evals/notify_mixin.py -- trivial helper, no cost
- tldw_chatbook/UI/Evals/skill_eval_panel.py -- DB-free widget; all updates are targeted query_one/Select mutations, estimate recompute is pure arithmetic
- tldw_chatbook/UI/Evals/skill_eval_launch.py -- chat callable is sync but the skill-eval runner dispatches it via asyncio.to_thread; readiness checked once per distinct provider; heavy imports (Chat_Functions) already resident; LocalSkillsService listing is small index-JSON read (only P3-level loop work)
- tldw_chatbook/UI/Evals/card_picker.py search path -- 0.2 s debounced (TASK-15476), rebuilds only the rows container, toggling a row mutates one label (the mount size itself is F11)
- tldw_chatbook/UI/Evals/inspector.py EvalsCellInspector.show_cell -- targeted Static.update per focus change, no recompose, no DB
- tldw_chatbook/UI/Evals/results_grid.py lens/baseline/sort/export -- mutate the mounted DataTable (virtualised) and Statics directly; no DB re-read; export writes via FileSave callback (cold path)
- tldw_chatbook/UI/Evals/bench_editor.py Add/Remove/prompt-mode -- targeted #evals-bench-targets-section rebuild, never a whole-editor recompose (only the N+1 in F13 remains)
- tldw_chatbook/UI/Evals/snippet_editor.py import read -- bounded (2 MiB) and on a thread worker; parsers use module-level compiled regexes
- tldw_chatbook/UI/Evals/library_rail.py apply_selection (rail-click re-mark) -- measured 0.6 ms @180 rows, 1.1 ms @650, 8.1 ms @1500; acceptable
- tldw_chatbook/UI/Evals/sample_bench.py HTTP -- WordBenchCaptureClient holds one pooled httpx.AsyncClient per target and is closed after the run; progress callbacks do targeted button-label updates only
- Polling: no set_interval/while-True/sleep loops anywhere in the slice (CardPicker debounce timer only); EvalsScreen._feed_skill_eval_panel_if_unfed retry loop is bounded
- Startup-import: UI/Evals is imported only by evals_screen.py (lazy ScreenRoute); Backup_Recovery/runtime_maintenance.py and unsaved_editors.py reference UI.Evals by string/sys.modules lookup, not import
- Logging: only exception-path logger calls in the slice; no hot-path f-string logging
- tldw_chatbook/UI/Evals/character_bench_editor.py ProbeSetDetail/_probe_listing_widget -- probes rendered as ONE Static, not a widget per probe

## census
| Measurement (isolated scratch DB, Textual 8.2.8, 211x44 headless) | light (5 tasks, 500 cells) | moderate (20 tasks, 60 runs, 12k cells, 54 MB) | heavy (300 runs, 60k cells, 268 MB) |
|---|---|---|---|
| EvalsDB.get_task (admission-dominated) vs raw conn.execute | 3.9-4.7 ms vs 0.006 ms (245 os.open/call) | same | same |
| run_group_cell_failure_counts() | 6.9 ms | 74 ms | 345 ms |
| list_runs(limit=500) | 4.3 ms | 7.7 ms | 48 ms |
| LibraryRail compose DB reads (8 calls) | 33 ms | 134 ms | 405 ms |
| Bench selection DB reads (16 calls) | 52 ms | 133 ms | 456 ms |
| Run-group selection DB reads (body+inspector) | 33 ms | 206 ms | 830 ms |
| load_grid (one 400-cell group) | 14 ms | 24 ms | 26 ms |
| LibraryRail mount+paint incl. reads | - | 238 ms (70 buttons) | 647 ms (190 buttons) |
| Rail section toggle (whole-rail recompose) | - | 237-263 ms | 658-683 ms |
| LibraryRail mount, no DB, 180 / 650 / 1500 rows | 230 / 370 / 924 ms | | |
| SnippetEditor mount+paint, 50 / 200 / 1000 snippets | 113 / 279 / 1338 ms | | |
| ResultsGrid per-arrow-key probe scan, 200 / 1000 snippets x 5 probes | 0.5 / 2.8 ms | | |
| save_cell during seeding (per cell, incl. admission) | ~4.3 ms | | |
| json.loads of one 2.6 MB inline dataset | 9 ms | | |


# slice-45

## summary
Slice #45 (UI/Library_Modules#1, 31 files, 31,556 lines) holds the non-visual controllers behind the Library screen. Most of the screen's event-loop work runs through them. They are reached from @on handlers and actions on the reusable LibraryScreen, from the ingest-registry listener (fired synchronously per job submit and marshalled per job transition), and from per-keystroke editor/filter handlers. Service I/O is well disciplined: nearly every DB and file read goes through _run_library_service_call(isolate_in_worker=True) or asyncio.to_thread, uses generation fences, and runs in exclusive workers. There is no sync sqlite on the loop in this slice. The real costs are CPU and render work on the loop.

(1) The ingest-registry listener (_handle_library_ingest_registry_changed) deep-copies the whole job queue via registry.jobs() on every notification. That is always once, and four times plus two full state builds when the Ingest canvas is showing. A folder submit (per-file listener fire) is therefore O(n²) again. Measured: 0.6 s at 500 files and 2.3 s at 1,000 files for the always-on copy alone; 4.1 s at 500 files on the Ingest canvas. This re-opens what TASK-32804.5 fixed for the app-level listener.

(2) The ingest queue panel recomposes every job row on every job transition. The queue is uncapped: up to 500 persisted jobs plus the session, and DONE rows never collapse. Measured on a synthetic tree: 100 rows ≈ 0.21 s and 500 rows ≈ 0.9–1.3 s per recompose.

(3) Every Notes-editor keystroke runs the ungated _apply_library_notes_stage_visibility, which does 6–7 whole-screen bool(query(...)) DOM walks. Measured ≈ 22–24 ms per keystroke on a synthetic 362-widget screen. TASK-23151's signature gate exists but only the resize path uses it.

(4) Every Collections interaction (row pick, scope, page, filter, favorite, and so on) runs two whole-screen recomposes. The helper binds refresh to Screen.refresh and is called from 41 sites.

(5) The Library conversation reader eagerly loads the whole transcript 20 messages at a time. After each page it re-updates every mounted row and mounts rows one by one. Measured ≈ 2.75 s of loop work for a 500-message conversation.

Smaller issues:
- A full ingest state build (queue deep copy) runs per path keystroke.
- The browse controllers (media, prompts, trash) do two full-canvas recomposes per request, one for loading and one for the result, driven by a 0.12 s filter debounce.
- Export, media-highlight and ingest actions still take whole-screen recomposes.
- The full source snapshot is re-fetched per completed import job, and exclusive-cancel does not stop the thread work.
- Note-import group actions are O(page × plan).
- Row toggles run whole-screen class queries.

Structural note: the controllers are verbatim moves out of the 35.9k-line library_screen.py and bind framework services (refresh, query, query_one) to the SCREEN. So controller-local refresh(recompose=True) and self.query() calls silently become whole-screen operations. That root pattern sits behind findings 3, 4 and 9.

## clean areas
- tldw_chatbook/UI/Library_Modules/library_artifacts_controller.py: search debounced at 0.2 s, all catalog reads go through _run_library_service_call, the config save runs in to_thread, and sync() patches the OptionList in place (no recompose)
- tldw_chatbook/UI/Library_Modules/library_artifacts_navigation.py: in-memory handoff store only
- tldw_chatbook/UI/Library_Modules/library_artifacts_share_controller.py: every service touch uses asyncio.to_thread
- tldw_chatbook/UI/Library_Modules/library_character_repair_controller.py: modal whose repair and refresh run in asyncio.to_thread
- tldw_chatbook/UI/Library_Modules/library_collections_capture_controller.py: headless and generation-fenced; CollectionsCaptureScopeService._call offloads via to_thread
- tldw_chatbook/UI/Library_Modules/library_collections_saved_search_controller.py
- tldw_chatbook/UI/Library_Modules/library_conversations_controller.py: 20-row pages, offloaded service calls, exclusive page worker
- tldw_chatbook/UI/Library_Modules/library_conversation_recovery.py: annotate dedupes workspace lookups and runs in to_thread
- tldw_chatbook/UI/Library_Modules/library_navigation_controller.py
- tldw_chatbook/UI/Library_Modules/library_inspection_admission.py: one-shot navigation path; its recompose is a single admitted navigation
- tldw_chatbook/UI/Library_Modules/library_notes_sync_controller.py: async runtime API; snapshot() is in-memory; the only disk touch is one is_dir stat per setup check
- tldw_chatbook/UI/Library_Modules/library_note_import_controller.py: planning and execution run in to_thread; progress is published per batch, not per item (group action is minor, see F11)
- tldw_chatbook/UI/Library_Modules/library_media_trash_browse_controller.py and library_prompt_browse_controller.py: fenced exclusive workers with isolated service calls (double sync noted in F7)
- tldw_chatbook/UI/Library_Modules/library_media_browse_controller.py: I/O offloaded; the facet and page fences are correct (double recompose in F7)
- tldw_chatbook/UI/Library_Modules/library_browse_route_swap.py: the targeted Media<->Notes swap replaced the measured 124 ms whole-screen recompose
- tldw_chatbook/UI/Library_Modules/canvas_sync.py: canvas-scoped Tier-2 sync with the rail patched via apply_selection (targeted); remaining fallbacks are known under TASK-281/TASK-22888
- State dataclasses: library_{conversations,collections,export,ingest,media,notes}_state.py and library_notes_work_session.py are plain dataclasses with no module-scope I/O; the media preview image cache is bounded (LIBRARY_MEDIA_PREVIEW_CACHE_LIMIT)
- tldw_chatbook/UI/Library_Modules/__init__.py: eager imports only of prompt modules that are already on the lazily imported Library screen graph
- Notes controller I/O legs: backlinks and note-location lookups run in to_thread/_maintenance_offload, autosave is debounced, the delete/trash/undo service calls are offloaded, and the ingest preflight is @work(thread=True)
- Media controller: filter search is debounced (0.12 s), detail loads are selection-settled, and image decode runs in to_thread

## census
| Probe (isolated env, audit tree, Textual 8.2.8) | Result |
|---|---|
| registry.jobs() deep copy, 100 / 500 jobs | 0.45 ms / 2.40 ms |
| build_library_ingest_state, 100 / 500 jobs | 0.82 ms / 3.93 ms |
| Folder submit with the always-on listener shape (1× jobs()), 500 / 1,000 files | 609 ms / 2,297 ms |
| Folder submit on the Ingest canvas (4× jobs() + 2× state build), 500 files | 4,084 ms |
| bool(screen.query(6-selector)), 362-widget screen | 8.8 ms |
| bool(screen.query('.cls')) / bool(query('#id')) / list(query(Type)), 362 widgets | 2.0 / 1.35 / 0.9 ms |
| Queue-panel recompose, 100 rows (400 widgets) / 500 rows (2,000 widgets) | 211 ms / 935–1,310 ms |
| Reader pattern (per-row mount + re-update all rows per 20-message page), 500 messages | 2,754 ms total (mounts 874 ms, re-updates 165 ms, rest frames) |
| One full re-update of 500 reader rows plus a frame | 81 ms |
| Note word-count regex, 20k / 150k words | 1.46 ms / 11.4 ms |


# slice-46

## summary
Slice #46 (UI/Library_Modules#2, 20 files, 17,470 lines): the Prompts, Skills and Search/RAG controllers plus their state, history, collections, import and helper modules. All of it is Library-only; the three big controllers are imported lazily from LibraryScreen.__init__, so none of it sits on the boot path.

Service and database I/O is disciplined. Almost every service call goes through `_run_library_service_call(..., isolate_in_worker=True)` or `asyncio.to_thread`, including prompt detail, save and delete, history, memberships, skills browse, save and delete, trust posture, script grant and skill import file reads.

The real costs are on the render side, plus one config-read cost that the slice hits on every keystroke:

- **Search/RAG query keystroke (F1, P1).** Each keystroke in the query box runs the full panel-status refresh: 8 whole-screen `self.query('#id')` scans, layout-forcing Static updates, widgets built and thrown away, recovery and callout blocks removed and remounted, and a rebuild of the provider gate. The gate calls `config.load_settings()`.
- **Warm `load_settings()` is slow (F3, P1, app-wide).** A warm call still costs 11–15 ms and about 596 file opens, because the admission handshake wraps the cache-hit check. TASK-32804.1 fixed `get_cli_setting` (1 µs) but not `load_settings`, which has 103 call sites.
- **Synchronous credential read (F2, P1, conditional).** The RAG gate reads credentials synchronously. For Anthropic in subscription mode on macOS this runs a Keychain `security` subprocess on the event loop at most every 5 s, with a 5 s timeout.
- **Full panel rebuilds (F4, P1).** Selecting an evidence card, starting a search and applying its outcome each rebuild every result card and history row, one awaited remove/mount at a time. That is about 145 ms in a synthetic app of the same shape.
- **Prompts delete/undo (F5, P2).** One click stacks a whole-screen recompose on top of 3–4 canvas recomposes. A prior review reported the 16 whole-screen sites; no backlog task was filed.
- **Browse loading frame (F6, P2).** Before every Prompts or Skills fetch the controller paints a "loading" state, which is a full canvas recompose, then recomposes again when the result lands. The Prompts list canvas `sync_state` always recomposes.
- **Skills editor keystrokes (F7, P2).** Each keystroke does whole-screen DOM scans and, in the Name field, forces a relayout. The Prompts editor avoids both by acting only on the clean-to-dirty transition.
- **P3 items.** The skill tool catalog is rebuilt on every skill open, on the event loop. The source snapshot is deep-copied 2–3 times per visit. The unavailable-character pager recomposes the whole screen.

Structural note: the controllers are verbatim-moved god classes (5.2k, 3.2k and 1.9k lines). They reach the screen through about 100 forwarding properties, but that indirection costs almost nothing. The cost comes from the refresh granularity they inherited: whole-panel and whole-screen rebuilds, and `query()` truth tests where a cached `query_one` would do.

## clean areas
- UI/Library_Modules/prompt_history.py: every count, page and restore call is isolate_in_worker; page size 10; state is immutable and only repaints through sync_view
- UI/Library_Modules/prompt_history_region.py: recompose is scoped to the region, sync_state is change-gated and only the view model is compared. Rows accumulate with 'Load older' (O(loaded rows) per publish), but the path is rare
- UI/Library_Modules/prompt_collections.py: all catalog, membership and apply calls run off the loop. Minor: _hydrate_membership_labels pages the whole catalog 100 at a time to resolve uncached labels, and the label cache lives on the screen instance (rebuilt per visit). One extra query for most users
- UI/Library_Modules/prompt_collection_manager_modal.py: recomposes are modal-scoped (small subtree); catalog loads run in workers
- UI/Library_Modules/library_skills_builtin_controller.py: config persist uses asyncio.to_thread; service calls are isolate_in_worker
- UI/Library_Modules/library_skill_import_controller.py: file read_bytes/read_text and inspect_skill_directory use to_thread; only one is_dir, an iterdir and about 2N+1 lstat run on the loop per explicit import (negligible)
- UI/Library_Modules/library_skills_browse_controller.py: exclusive worker group, isolate_in_worker, bounded page-fetch retries (apart from the loading-paint issue in F6)
- UI/Library_Modules/note_session_port.py: load and save are isolate_in_worker (two sequential hops per load; minor)
- UI/Library_Modules/screen_helpers.py: the keyring-backed principal read is documented off-loop; its only loop caller passes resolve_principal=False
- UI/Library_Modules/screen_constants.py, screen_support_types.py, library_prompts_state.py, library_skills_state.py, library_rag_search_state.py: pure data and constants, no module-scope I/O
- UI/Library_Modules/skill_import_choice_modal.py: a small OptionList modal
- Prompts editor per-keystroke dirty tracking (library_prompts_controller.py:2456-2517): cached query_one plus build_prompt_editor_state, measured at 0.02 ms; the meta/history/discard repaint only runs on the clean-to-dirty transition
- Prompts search-as-you-type debounce (library_prompts_controller.py:1363-1419): a 250 ms timer with a token-superseded dispatch, which is the good pattern (apart from the double recompose in F6)
- Skills trust posture and script-grant loads: to_thread; the posture repaint takes the header-only early return in LibrarySkillsListCanvas.sync_state, so an unchanged posture costs nothing
- Skill and prompt detail load/save/delete paths: all offloaded through _run_library_service_call(isolate_in_worker=True)
- Search history persistence: goes through the @work(thread=True) _save_library_search_history

## census
| Probe (isolated, scratch HOME / TLDW_CONFIG_PATH) | Result |
|---|---|
| warm `config.load_settings()` | median 15.1 ms, min 10.9, max 26.8 (100 calls); cProfile: ~596 `posix.open` per call via `Backup_Recovery/config_participants.py:400 wrapped` |
| warm `config.get_cli_setting()` (TASK-32804.1 fastpath) | 0.001 ms |
| `library_rag_answer_provider_gate()` (openai default, not subscription) | median 18.1 ms, max 43.8 ms |
| `library_rag_profile_top_k()` | 0.003 ms |
| `build_prompt_editor_state` (legacy prompt, 2.3 KB + 1.5 KB) | 0.02 ms |
| Textual 8.2.8, 506-widget screen: `bool(screen.query('#id'))` | 1.44 ms (~2.85 µs per node) |
| same: `screen.query_one('#id')` (cached) | 0.0002 ms |
| same: `Static.update(layout=True)` + one refresh cycle vs `layout=False` | 47–49 ms vs 39.6 ms (~7–10 ms reflow) |
| remove + remount 15 RAG-style cards (8 widgets each) + 10 history rows, sequential awaits | 145 ms median (idle cycle 24 ms); batched remove_children/mount_all 98 ms |
| `copy.deepcopy` of a representative Library source snapshot (100/50/50 records + prompts + skills) | 1.33 ms median, 2.2 max |
| `BuiltinToolProvider().list_catalog()` + LocalToolProvider (all gates + local tools on) | first call 110 ms, warm 0.2–0.4 ms; gates off 0.01 ms; cold `import tool_catalog` 430–540 ms |
| spawn `/usr/bin/security help` (lower bound for the Keychain read spawn) | 4.7 ms median |


# slice-47

## summary
Slice #47 (UI/MCP_Modules, 13 files, ~18k lines) is the MCP Hub screen: MCPScreen -> MCPWorkbench (rail + 4 mode canvases + inspector). The only importer outside the slice is UI/Screens/mcp_screen.py, and the screen pre-importer loads it off the event loop, so there is no boot-path cost. Its hot paths are:
- **Visit:** on_mount schedules reload() -> _sync_children(). On warm visits, _apply_view_state then runs a second full _sync_children.
- **Polling:** a 0.25 s set_interval projects save status.
- **Selection:** rail and servers-table clicks run a full _sync_children.
- **Row highlight:** DataTableClickSelectMixin turns every arrow key in the Tools, Permissions and Audit tables into a selection.
- **Filter typing:** each filter keystroke rebuilds a whole table.
- **Space press:** each press in the Permissions matrix triggers a standalone _sync_permissions_mode.
- **Actions:** lifecycle actions and config saves each run one or two full passes.

Health: writes are well handled. Permission setters, the kill switch, config/gate saves, the import-file read, the test preview mint/run and the workspace-root and master saves are all offloaded (asyncio.to_thread or owner.submit). The deferred canvas mount (task-2901) and the deferred initial load (TASK-1320) are good.

The remaining cost is reads plus rebuilds:
- **Guarded store reads.** Every interaction funnels into synchronous calls to the five guarded MCP stores on the loop: permission, execution log, local, context and target stores. Each call costs 11-28 ms, measured, and is dominated by the storage-admission handshake: a hold-thread spawn and join plus about 1,190 open() calls per call.
- **Full rebuilds.** Every _sync_children pass unconditionally rebuilds all five DataTables, including those on hidden canvases, and remounts the callouts, toolbar, toggles and inspector actions.

Structurally, _sync_children is one monolithic pipeline that about 15 callers reuse regardless of what changed. TASK-32804.7 (Done) removed only the config-read part of this cost. The permission-store, execution-log and target-store reads, the full-table rebuilds and the 4 Hz relayout are not covered by any open task. TASK-32804.12 covers the service side of the synchronous control-plane calls generically.

Suggested PR groups:
- **(A) Cheap UI wins, low risk:** F5 (4 Hz poll dedup and gating) and F6 (filter debounce, precomputed cells, fixed-height rows).
- **(B) Offload the workbench data phase:** F1, F2, F3's data half and F9. Move the permission/log/catalog reads into one asyncio.to_thread hop per interaction and dedupe reads within it. MCPToolProvider already calls the same store methods from worker threads, which is the thread-safety precedent.
- **(C) Scope the render phase:** F3's render half and F4. Render hidden canvases lazily on mode switch, run scoped syncs per interaction, and skip the redundant restore pass.
- **(D) Prewarm the first-visit imports:** F7.
- **(E) Root-cause admission amortisation for the MCP store route:** F8, in Backup_Recovery, coordinated with TASK-32804.1.

Measurement scripts are in /private/tmp/claude-501/-Users-macbook-dev-Documents-GitHub-tldw-chatbook/15c5e43c-3dea-41a4-87ab-3b132387df58/scratchpad/audit/scratch/s47/: bench_perm.py, bench_pass.py, bench_hub_tools.py, bench_hub_first.py, bench_poll.py, bench_tables.py, bench_audit.py and bench_filter_fix.py. Every run was isolated (HOME, XDG_* and TLDW_CONFIG_PATH under the scratch dir, TLDW_TEST_MODE=1) and headless where Textual was involved.

## clean areas
- tldw_chatbook/UI/MCP_Modules/__init__.py: empty package, no eager imports
- tldw_chatbook/UI/MCP_Modules/mcp_local_master_button.py: trivial Button subclass
- tldw_chatbook/UI/MCP_Modules/mcp_recovery_review.py: static confirmation dialog, rare path
- tldw_chatbook/UI/MCP_Modules/mcp_schema_form.py: parse_schema is a pure dict walk costing microseconds; the form is built only when Test Tool opens
- tldw_chatbook/UI/MCP_Modules/mcp_profile_form.py: its per-keystroke watch does 2 query_one calls plus a disabled/tooltip set, which is negligible; the import preview runs on button press and parse errors are bounded
- tldw_chatbook/UI/MCP_Modules/mcp_server_mutations.py: plain form, work runs on submit only
- mcp_rail.py sync_state: rows are updated in place when the structure is unchanged, and a recompose happens only when the source, scope or server-key list changes (mount-echo guards are intact)
- mcp_inspector.py Advanced runner: opt-in (hidden by default); preference writes use asyncio.to_thread and section loads run in exclusive workers
- Writes across mcp_workbench.py are all offloaded (asyncio.to_thread or MCPLocalConfigSaves.submit): the permission setters via _call_profile_scoped_off_loop, set_kill_switch, _save_builtin_flag/_save_tool_gate, the workspace root and master switch, the import-file size check and read, and the test preview mint, revoke and execute
- Deferred canvases (_mount_deferred_canvases, task-2901) and the deferred initial load (on_mount -> call_after_refresh -> worker, TASK-1320) are the right pattern
- Tool-test 'active' poll (_TOOL_TEST_ACTIVE_POLL_SECONDS=0.3): runs only while a prior test of the same tool is running, is gated on _test_panel_is_current and exits when that test ends
- Logging in the slice is on failure paths only; no debug/info f-strings on hot paths
- Memory: the audit window is capped at 200 of at most 1,000 on-disk rows, and the _in_flight and reclaim-task sets are pruned; no unbounded growth found
- Module imports: mcp_workbench's MCP.* imports are already loaded at boot (app constructs UnifiedMCPControlPlaneService), and the screen module itself is pre-imported off-loop by the screen pre-importer
- Resize handlers (_reflow_table, on_resize refit) rebuild only when the measured width budget changes

## census
| Operation (isolated, headless where Textual) | Median cost |
|---|---|
| MCPPermissionStore.read_profile_inventory_snapshot + per-profile digests (5 KB / 63 KB / 281 KB file) | 11.8 / 13.2 / 17.6 ms |
| MCPPermissionStore.load() (behind effective_tool_states, get_kill_switch, list_tool_arg_rules, gate_tool_test) | 16.6-18.1 ms |
| MCPExecutionLog.read_recent(200) with 400 records on disk | 27.8 ms |
| LocalMCPStore.load() (local_external_catalog and get_catalog_bundle each call it) | 13.3 ms |
| UnifiedMCPContextStore.load() (load_context) | 12.3 ms |
| cProfile of 20 permission-store loads | 23,800 posix.open calls (about 1,190 per call) and 20 thread.join calls in storage_admission.close |
| local_hub_tools-equivalent build (2 providers + definition_hash): steady / first per process | 1.06 ms / 91.6 ms (97 modules lazily imported) |
| MCPToolsMode filter keystroke (_apply_filter + idle dimension pass), 40 / 120 tools | 17.5 / 26.5-37.3 ms |
| Same, 120 tools, with height=1 instead of auto-height | 15.1 ms (vs 26.5 ms in the same run) |
| MCPPermissionsMode.update_matrix, 47 / 127 rows | 16.6 / 29.7 ms |
| MCPToolsMode.update_states full rebuild (runs on every Space press), 40 / 120 tools | 18.5 / 42.0 ms |
| MCPAuditMode.update_entries(200), visible / hidden | 20.3 / 12.7 ms |
| 4 Hz _sync_local_config_save_status poll: idle CPU with / without, 118-widget workbench | 1.13-1.19% / 0.03% of a core; 40 full-screen relayouts per 10 s (21 ms) vs 0 |
| Derived: store I/O per Space press / per row highlight / per reload() | about 66 ms / about 28 ms / about 122 ms |


# slice-48

## summary
Slice #48 (UI/Screens#1, 7 files, 10,140 lines). Live entry points: ChangeReviewScreen (4,967 lines), a pushed overlay opened from the Console Inspect rail, the turn-file-card "Review" button and the Environment "Review & commit" row. The same module's AgentRunsChangeReviewProvider is also built on the event loop every time a ConsoleTurnFileCard mounts. ChangeRevertConfirmModal is used by Console turn Undo-All. ACPScreen is a live route. BackupRestoreScreen is opened from app.action_backup_restore and the recovery launcher; it is cold. artifact_share_dialog is used by Library's share controller. ArtifactsScreen is dead as a destination: `_SCREEN_ALIASES["artifacts"] = "library"`, and its only production consumer imports it for one classmethod.

All the real cost is in ChangeReviewScreen. Its snapshot-turn path is synchronous on the event loop by design; the `_load_current_mode` docstring says "The snapshot path keeps its existing synchronous posture unchanged". No backlog task tracks this.
- Opening the screen, switching turns, and reloading after every revert each spawn at least 3 git subprocesses on the loop. Each one goes through `acquire_storage` storage admission, which reads registry.json under a lock twice per call; that part is outside this slice, but it multiplies the cost. The same paths also hydrate the full step log of the run once per snapshot row.
- Every j/k or tree click spawns one git diff on the loop, plus a full table scan of change_notes.
- Every up/down arrow in the diff pane rebuilds and re-lays out the entire diff Static, up to 2,000 lines. Measured at +57 ms per keypress at the cap.
- Current mode builds an uncapped tree, and the commit modal mounts one Checkbox per changed file.
- Snapshot diffs are content-addressed and immutable, yet nothing caches them. The diff memo holds only one leaf.

Everything else is healthy. BackupRestoreScreen does all service work in thread workers, and its 5 Hz poll already pauses on suspend and skips unchanged updates. ACPScreen does cheap in-memory work. ArtifactsScreen's recompose storms (2-4 whole-screen recomposes per visit or resume) and its N+1 kept-report lookups are unreachable, so they are hygiene only. Its real issue is structural: 1,712 lines of dead code that a Library action imports for a static builder.

Structural note: change_review_screen.py mixes the provider (data and git layer), the screen, and three modals. The per-leaf git and DB reads should move into one token-guarded thread worker, the same pattern `_load_current_mode` already uses in this file.

## clean areas
- tldw_chatbook/UI/Screens/__init__.py -- PEP 562 lazy exports, no eager screen imports
- tldw_chatbook/UI/Screens/backup_restore_state.py -- pure label helper
- tldw_chatbook/UI/Screens/acp_screen.py -- compose_content does only in-memory state and Popen.poll() via manager.snapshot(); launch/stop are @work(thread=True); recompose only after an explicit launch/stop on a ~40-widget screen
- tldw_chatbook/UI/Screens/artifact_share_dialog.py -- simple modal, one SelectionList (line-virtualized), handlers only read widget values
- tldw_chatbook/UI/Screens/backup_restore_screen.py service calls -- every preview/summary/list/media read runs in @work(thread=True) workers with revision guards; start_* handlers only schedule onto the service executor; list pages are small (media paged at 20)
- tldw_chatbook/UI/Screens/backup_restore_screen.py 5 Hz poller -- RecoveryService.current() is an in-memory dict read under a lock; the poller already pauses on suspend and compares text before Static.update (tier-2 S19 fix verified)
- tldw_chatbook/UI/Screens/change_review_screen.py current-mode status/detection/commit/push/PR -- all correctly off-loop in thread workers with token guards (_dispatch_git_detection, _load_current_mode, _dispatch_commit*, _dispatch_push, _dispatch_pr)
- tldw_chatbook/UI/Screens/change_review_screen.py revert/undo-all/preflight and note save/delete -- use asyncio.to_thread correctly
- tldw_chatbook/UI/Screens/change_review_screen.py ChangeRevertConfirmModal / ChangeGitPushModal -- trivial compose, no I/O
- change_review_screen.py logging -- loguru {} placeholders (lazy), paths fingerprinted, no large payloads
- artifacts_screen.py share flow -- the Web_Server chain is lazily imported (TYPE_CHECKING plus call-site imports); listing runs in a thread worker

## census
| Measurement (this Mac, Textual 8.2.8, Py 3.12; scripts in scratch/s48) | Result |
|---|---|
| `git diff -M --name-status -z A B` (710-file range) | median 34.0 ms |
| `git diff -M --numstat -z A B` (710-file range) | median 168.1 ms (scales with files; a few-file turn is ~15-25 ms) |
| `git diff -M A B -- one/file.py` | median 18.8 ms |
| `git cat-file -e sha^{commit}` (bare spawn floor) | median 15.3 ms |
| Diff pane cursor move: rebuild Text + Static.update + refresh, 200 / 1000 / 2000 lines | +10 / +29 / +57 ms incremental per keypress (pilot baseline 21.8 ms subtracted; Text build alone 0.3 / 1.6 / 3.3 ms) |
| Tree population, 1k / 5k / 20k leaves (build + first refresh) | ~21 / ~83 / ~380 ms incremental (after subtracting the ~22 ms pilot baseline) |
| `SELECT * FROM change_notes WHERE run_id=? ORDER BY id` query plan with the shipped partial index | SCAN change_notes (the partial index is used only when `AND delivered_at IS NULL` is added) |


# slice-49

## summary
Slice #49 (UI/Screens#2, 11 files, ~32.8k lines) is dominated by chat_screen.py: 25,406 lines, one ChatScreen class of about 23.6k lines and 772 methods. Its hot paths are (a) the 0.2 s transcript-sync tick (`_sync_native_console_chat_ui`, 5 Hz for any active run), (b) the 0.25 s credential poll, which runs for the life of the reusable Console, (c) the per-keystroke on_key / DraftChanged / Input.Changed chain, and (d) per-visit on_screen_resume.

Most of the historical cost here was already found and fixed: the keystroke path is gated and memoized, click-outside dismissal uses registries, sidebar persistence is debounced off the loop, and the transcript reconciler, cost chip and context estimate are revision- or TTL-cached. Findings in that territory are therefore residual rather than new classes of problem.

What is new in this pass:
1. `_apply_console_settings_summary_state` still runs two uncached full-DOM `self.query()` walks on every settings-summary sync. Measured at about 2.2 ms each on a 1,000-node DOM (about 4.5 ms per call), and it is on the 0.2 s tick.
2. `_build_video_card_specs` does sync storage admission, stat and resolve work per video row on every tick. When a recovered-media catalog exists it also constructs a full `RecoveredMedia` (migrate, validate, recover) plus a second SQLite connection, all on the event loop.
3. The per-tick full transcript snapshot plus O(n) fingerprint and signature passes, measured at 7 ms per tick at 1,000 messages and 28 ms at 3,000.
4. `_ensure_console_chat_controller()` is a side-effecting getter. Every call rebinds 15 view-hook slots and reruns `_sync_console_chat_core_state`. It runs at least 3 times per tick and is threaded into the Console controllers as their accessor.
5. The recovered-image lookup is re-dispatched on every sync instead of only when its selection changes.

The credential poll (TASK-32804.3, In Progress) is still a P0-grade idle tax: about 8 ms per tick at 4 Hz is roughly 3 % of a core whenever Console is visible.

Other screens:
- Home re-renders the whole screen with `refresh(recompose=True)` on every visit, after the chatbook-snapshot worker finishes, bypassing its own targeted `_sync_home_triage`. Its 3 s active-work TTL means rail clicks usually miss the cache and run 3 sync DB queries on the loop (residual of task-282).
- Evals runs many sync SQLite reads per selection swap: 500-row list queries run repeatedly, and lookups by id linearly scan a freshly fetched 500-row list. `ResultsGrid.compose` also drains every result page with JSON decode on the loop.
- Chunking Lab is almost entirely `to_thread` and fine, apart from undebounced whole-sample edits on each keystroke.

Structural notes:
- The chat_screen import closure is 1,657 modules and about 0.86 s cold (measured). It is pre-warmed behind the splash (task-21110), and the Widgets.Console package `__init__` still eagerly loads about 19 ms of modals that TASK-22213 deliberately left.
- Dead legacy sidebar-state code: duplicate `on_button_pressed` and duplicate `_restore_collapsible_states` definitions, so the first copy of each is shadowed and dead (about 200 lines), and handlers for `#chat-expand-all` / `#chat-collapse-all` / `#chat-reset-settings`, ids no code composes.

## clean areas
- chat_screen.py on_key / _handle_console_composer_draft_edit / _on_console_composer_draft_changed: equality-gated workbench push, derivation-scope memo, popup sync gated on draft text (residual known as TASK-24454 / TASK-21120)
- chat_screen.py on_mouse_down / on_click / on_mouse_up / on_paste: ancestor walk first, then registry-based menu lookup with early return (TASK-21119); no full-DOM queries
- chat_screen.py sidebar-state persistence: debounced 0.5 s and written off the loop through _FileJob; suspend flush serialized
- chat_screen.py _prune_console_rail_preferences: @work(thread=True), one-shot latch
- chat_screen.py _poll_console_environment: 10 s, gated on active screen and open rail (TASK-31628 covers lifecycle)
- chat_screen.py transcript sync timer: self-stopping on _console_transcript_poll_needed(), stopped in on_screen_suspend, re-armed only when runs are in flight
- chat_screen.py cost chip / context estimate: ADR-190 revision keys plus 1 s streaming TTL; the cost TTL timer only runs while the cache is WARM
- chat_screen.py _poll_console_watchlists_operations: asyncio.to_thread reads, 2 s cadence, exits when no operation is active or the screen is unmounted
- chat_screen.py _sync_console_citation_count_discovery: incremental signature diff, DB reads via asyncio.to_thread
- chat_screen.py whole-screen refresh(recompose=True): fallback-only (line 16043)
- chat_screen.py export (aiofiles) and subprocess.Popen open-file: user-action only
- home_screen.py _refresh_home_content_snapshot / _refresh_home_active_work_cache / _save_home_rail_preferences: to_thread or thread workers
- lab_frame.py: body mounted from call_after_refresh; rail toggles use apply_rail_layout, not recompose
- lab_mode_strip.py, chat_screen_state.py, chatbooks_screen.py, destination_recovery.py, library_conversations_screen.py: trivial, nothing on hot paths
- image_gen_demo_screen.py: generation in a @work(thread=True) worker; the post-result PIL decode on the loop is dev-demo only
- chunking_lab_screen.py file, recovery, media and catalog I/O: all asyncio.to_thread; autosave coalesced (lab_autosave.AutosaveWriter)
- evals_screen.py run workers (bench, character, skill-eval): provider calls to_thread; selection swaps serialized by lock and revision, never cancelled mid-teardown

## census
| Probe (isolated env, Textual 8.2.8, py3.12) | Result |
|---|---|
| `screen.query('#id .cls')` + `.first()` on ~1,000-node DOM | 2.225 ms/call (query_one '#id' cached: 0.0003 ms) |
| `dataclasses.replace` snapshot of N ConsoleChatMessage (messages_for_session shape) | 1.33 ms @200, 6.61 ms @1000, 17.94 ms @3000 |
| transcript fingerprint + tuple compare + citation/identity/id-set passes | ~0.5 ms @1000, ~10.5 ms @3000 |
| total per-tick transcript O(n) passes (5 Hz while streaming) | 7.1 ms @1000 (3.5 % core), 28.4 ms @3000 (14 % core) |
| `import tldw_chatbook.UI.Screens.chat_screen` cold (isolated) | 0.86-0.96 s cumulative, 1,657 modules in closure |
| Widgets.Console package-init modals (citation_sources, edit_message, fork_chat, prompts, rename, save_as, workspace_switcher) | ~18.7 ms cumulative import |
| Code counts in chat_screen.py | 142 query_one, 16 self.query(), 80 run_worker, 25 set_timer, 4 set_interval, 31 _ensure_console_chat_controller() call sites (+22 in Console_Modules) |


# slice-5

## summary
Slice #5 Canvas (18 files, 13.9k lines under tldw_chatbook/Canvas/). Canvas is lazily wired, but its costs land in five places. (1) Boot: config.py imports Canvas.limits, and the package __init__ then eagerly pulls Canvas.models, about 6.5 ms of boot. (2) First Console paint: ConsoleRuntime.ensure_chat_store is reached from ChatScreen.compose_content. It imports console_canvas_controller + repository + profiles and builds CanvasService and the ProfileSnapshot, which reads and SHA-256s about 1.8 MB of packaged JS. That is 14-20 ms measured, even if Canvas is never used. (3) Always-on: a 4 Hz asyncio.to_thread policy watcher runs from app mount for the whole process (a test pins this), and served mode's parent runs a duplicate one. (4) Per Console interaction: the message action row re-parses the whole selected message with a fresh MarkdownIt to find Canvas fences. That happens 2-3x per transcript refresh, per selection keystroke, and at 5 Hz while streaming. Measured 9 ms/refresh at 10 KB and 34 ms at 40 KB. This is the hottest finding. (5) Per Canvas interaction: the native aiohttp gateway runs on Textual's own loop and calls a synchronous authority (via _maybe_await). So every /api/state, /render, /api/plan and /api/navigate request, and every 'Open in Canvas' click, runs several full active-path validations on the loop (IN over up to 4096 ids, about 2.5 us per id measured), plus metadata scans, full-source reads and, on import, a BEGIN IMMEDIATE write. Compilation is correctly moved to an executor, but render plans for immutable revisions are recompiled on every frame load. The compiler's diagnostic source locator is quadratic and takes about 50% of compile time on large documents (measured). gateway.py mixes aiohttp with plain wire dataclasses, so the first open pays 76-160 ms of aiohttp import on the loop, and served-child boot pays it before first paint. Structural notes: Canvas/staging.py CanvasStagingStore is used only by tests, and the production staging is Chat/console_canvas_controller.py. The controller holds a threading.RLock across SQLite calls that both tool workers and UI-loop paths take. The Settings > Privacy Canvas card still resolves the served-auth policy (getaddrinfo + a possible keyring call) on the loop, which is a gap left by Done TASK-33081. No existing open task covers any Canvas-runtime perf item. Fixes group naturally into 3 PRs: (a) transcript fence-parse memo; (b) Canvas import/boot laziness (lazy __init__, split the aiohttp-free gateway types, lazy controller/snapshot, deduplicated asset bytes); (c) gateway/authority off-loop plus plan cache, locator bisect, static-asset caching, watcher backoff and long-poll.

## clean areas
- tldw_chatbook/Canvas/limits.py - pure bounded validators; iterative JSON-depth walk; CanvasLimits() is 1.7 us (measured)
- tldw_chatbook/Canvas/capabilities.py - capability store bounded by max_active; O(records) prune/revoke on a small map
- tldw_chatbook/Canvas/compilation.py - correct run_in_executor offload with non-blocking 2-slot admission and shielded cancellation
- tldw_chatbook/Canvas/archive.py - Chatbook export/import only (cold); per-revision source streaming is intentional
- tldw_chatbook/Canvas/authoring.py, tldw_chatbook/Canvas/guide.py - small package-resource reads at tool-call time only
- tldw_chatbook/Canvas/control_protocol.py - length-prefixed JSON frames with bounded payload validation; served-mode IPC only, no polling of its own
- tldw_chatbook/Canvas/web_auth.py WebAuthManager - bounded OrderedDict/deque session and rate-limit state; per-request work is O(1)
- tldw_chatbook/Canvas/repository.py read queries - indexed (idx_canvas_documents_conversation, idx_canvas_revisions_canvas_sequence/origin/parent); row counts capped by the 10 canvases x 100 revisions limits
- tldw_chatbook/Canvas/service.py list/resolve logic - linear over at most 1000 metadata rows; its only cost is repeated _validate_scope (see F4)
- tldw_chatbook/Canvas/models.py per-instance validation - microsecond-level (its cost is the import, see F7)
- tldw_chatbook/Canvas/profiles.py + runtime_assets.py - one-shot per process (cost is where it runs, see F5)
- gateway.py _render_plan_wire + json.dumps - 2.4 ms for a 444 KB/805-element plan (measured); acceptable per frame load
- Widgets/Console/console_transcript.py canvas_card_presentations - cheap metadata projection per message
- NativeConsoleCanvasAuthority bounded maps - _parsed_block_imports (512), _publication_receipts (256), _browser_targets (64)

## census
| Measurement (isolated env, audit tree 840ed2ca58, py3.12) | Result |
|---|---|
| assistant_canvas_html_blocks / action_groups per call, 2 KB / 10 KB / 40 KB message | 1.2 / 4.4 / 16.9 ms; per refresh (x2) 2.8 / 8.9 / 33.8 ms; per selection (x3) 4.1 / 13.3 / 50.7 ms |
| MarkdownIt('commonmark') construction alone | 0.18 ms |
| compile_canvas_document 5.6 KB / 90 KB / 444 KB | 8.6 / 26 / 107 ms; _SourceLocator share 0.2 / 11 / 57 ms |
| compile, 250 KB doc with 240 KB leading script + 560 elements | 92-97 ms (locator-dominated) |
| CanvasService._validate_scope-like IN query, path 100 / 500 / 2000 / 4096 | 0.17 / 1.2 / 5.0 / 10.5 ms |
| dataclasses.replace(ConsoleChatMessage) (canvas_active_path projection, per message) | 4.1 us |
| import Canvas.gateway after app import (aiohttp not loaded) | 76-160 ms (aiohttp 52-122 ms) |
| import Canvas.compiler after app import | 20 ms |
| import console_canvas_controller (+repository, profiles, compilation) after chat_screen | 11.7 ms |
| load_application_profile_snapshot cold / warm | 7.9 / 1.9-2.8 ms |
| Canvas.models self import time on boot path (via Canvas/__init__) | 6.0-6.5 ms |
| 4 Hz to_thread watcher pattern, idle CPU | 0.12 % core (idle baseline 0.001 %) |
| socket.getaddrinfo('localhost') cold / warm | 1.8 / 0.1 ms |


# slice-50

## summary
Slice #50 is one file, tldw_chatbook/UI/Screens/library_screen.py: 35,902 lines, 1.67 MB, one LibraryScreen class with 1,334 methods (587 are delegators of 3 lines or fewer), and an __init__ of 1,855 lines that holds 415 accessor lambdas. The screen route has been reusable since TASK-31521, so __init__ runs once per app run. That probably makes TASK-24457 ("8 fresh connections per visit") stale.

Many earlier passes have already tuned this file: the on_resize signature gate (TASK-23025), per-click canvas-scoped sync (TASK-21116), the review-set snapshot memo (TASK-32804.4), the content-match memo, the ingest-options memo on the app, and the source snapshot running off the loop with a deadline. There is no polling; the 6 one-shot timers are all stopped on suspend.

Hot paths traced:
- rail-row / destination switch
- open item from landing or Search
- Import canvas: submit plus the registry listener on every job transition
- Search/RAG query box, per keystroke
- footer model, rebuilt on every screen refresh and focus change
- per-visit resume: snapshot refresh plus entry reconcile
- first import of the module

New findings:
1. P0: the ingest registry listener does O(queue) work per notification and remounts the whole queue panel. A folder submit is therefore O(N²): measured 103 ms at 100 files and 1.06 s at 300. Each later job transition remounts every row: 200 ms at 100 rows, 650 ms at 300. Up to 500 jobs persist across restarts. TASK-32804.5 fixed only the app-level listener.
2. P0: every destination switch outside ordinary↔ordinary and Media↔Notes, and every cross-route open, still does a whole-screen `await self.recompose()`. The Phase-C design record measured 124 ms and 177 mounts for this.
3. P1: each keystroke in the Library RAG query box rebuilds the provider gate, which calls `config.load_settings()`. A warm call measured 6–7.5 ms because the storage-admission handshake still runs on every call. TASK-32804.1 only fast-pathed `get_cli_setting`.
4. P2: the `refresh()` override rebuilds the whole footer shortcut model on every plain refresh, including uncached `focus_chain` walks.
5. P2: a module-scope import of `prompt_variables_dialog` pulls in the eager Widgets/Console package: +308 modules and about 380 ms, measured by A/B.
6. P3: throwaway canvas builds per reconcile, `SELECT *` note bodies plus a deepcopy per visit, and `asyncio.run` per isolated service call.

Structural note: the god module's own import self-time is 23.5 ms (measured). The size ratchet was already reported (prior review, P2), so it is not re-filed. Housekeeping: my early probes ran before I set PYTHONDONTWRITEBYTECODE and may have refreshed gitignored __pycache__ files in the audit tree; `git status` is clean. All other scratch output is under scratchpad/audit/scratch/slice50/.

## clean areas
- polling: no set_interval / while-sleep loops; the 6 set_timer sites are one-shot and all are stopped in on_screen_suspend (library_screen.py:9205-9259)
- on_resize (8124): gated on a cheap layout signature before any query work (TASK-23025)
- on_key (8749): O(1) except the '/' branch; typed characters are consumed by Inputs before they bubble
- Local source snapshot (_list_local_source_snapshot 13232): gathered off-loop, asyncio.wait_for deadline, exclusive worker group; only P3 residue (F7)
- Media detail / highlights / progress fetch (_refresh_library_media_detail 19078): offloaded via _run_library_service_call
- Ingest preflight + duplicate probe (28593/28661): @work(thread=True)
- Export count/export workers (19625/20022): thread=True, exclusive
- Config persistence (rail prefs, search history, ingest options, reader prefs, editor modes): @work(thread=True) or asyncio.to_thread
- Ingest options config read: memoised on the App, keyed on current_config_identity (library_screen.py:658-689)
- Review-set reads: memoised _active_review_set_snapshot (31154, TASK-32804.4)
- Media content-match memo (33283); media analysis-reason resolution gated to the analysis tab (media controller 3939)
- LLM analysis generation: asyncio.to_thread (33616)
- LibraryRail.sync_state patches rows in place when the section shape is unchanged
- save_state (9780): lightweight scalars/frozen dataclasses only
- logging: 1 eager f-string debug, on a worker thread; no payload dumps
- network: no httpx/requests in this file
- db-query: no direct sqlite/execute in this file; all DB access goes through scope services
- Module-level tail: only 2 staticmethod rebinds (35817/35827)

## census
| Probe (isolated env, audit tree 840ed2ca58) | Result |
|---|---|
| `LibraryIngestJobRegistry.jobs()` at 100 / 300 / 500 jobs | 0.49 / 1.55 / 2.44 ms |
| Per-notify Library ingest listener state work (jobs() x2 + build_library_ingest_state + counts) at 100 / 300 / 500 jobs | 2.16 / 5.83 / 11.05 ms |
| Folder submit of N files with that listener attached, N=100 / 300 | 103 ms / 1,057 ms (quadratic) |
| Textual 8.2.8 recompose+settle of a queue-shaped panel (Static + Horizontal[2 Buttons] per row), 20 / 100 / 300 rows | 63 / 200 / 649 ms |
| `config.load_settings()` warm cache hit | 6.2-7.5 ms (~560 posix.open per call via storage admission) |
| `get_cli_setting()` warm (post TASK-32804.1) | 0.001 ms |
| `library_rag_answer_provider_gate()` warm | 6.5 ms median, 8.5 ms p95 |
| `Screen.focus_chain` at ~100 / 200 / 400 widgets | 0.15 / 0.31 / 0.73 ms |
| `import library_screen` after base_app_screen | 760 ms, 866 new modules |
| Same, with Widgets/Console `__init__` bypassed | 358-402 ms, 558 modules, console_chat_controller not loaded |
| library_screen module self import time (-X importtime) | 23.5 ms |
| `copy.deepcopy` of a representative source snapshot (100 notes / 50 media / 20 convs / 40 skills) | 0.54 ms median |
| `asyncio.run` per isolated worker call | 0.31 ms median, 1.37 ms p95 |


# slice-51

## summary
Slice #51 (UI/Screens#5, 17 files, ~31k lines) is dominated by personas_screen.py (16.5k lines, the Roleplay workbench) and llm_screen.py plus the model_*_view panes (Models/Lab). The slice is mostly careful. Nearly every DB, file and keyring call in personas_screen goes through asyncio.to_thread or _drain_to_thread, the meetings and models workers are thread=True, the Lab 2 s poll is equality-guarded, and the library search debounces at 0.2 s. The problems left are about how much work the UI does for each interaction, not raw sync I/O.

Hot paths traced:
- Roleplay library search, paging and sorting. Each re-render clears and remounts every ListItem: about 200 widgets per 50-row page (measured 112-185 ms). The Dictionaries and Lore modes are not paged at all.
- The Roleplay route is not in the reusable-screen opt-in (TASK-24452), so every visit re-composes the screen and re-runs the initial load. That load includes a redundant read of up to 1000 characters plus a Select options build for a widget this screen does not have, the remount, and the auto-select fan-out.
- Selecting one character reads the same SELECT * character_cards row, image BLOB included, 5 times across serial thread hops.
- Lore mode counts entries with an N+1 query that fetches every entry, and repeats it on every debounced search and every entry edit.
- The Roleplay conversation search has no debounce.
- vLLM setup: every keystroke posts DraftChanged, which calls get_api_key and then load_settings(). load_settings still pays the storage-admission handshake on a warm cache hit (measured 13 ms, about 600 open() calls, because of @_config_participants.guarded at config.py:1861). The TASK-32804.1 warm fastpath only covers get_cli_setting, and there are 103 load_settings() call sites repo-wide.
- Meetings with live diarization: every relabelled segment clears and rewrites the whole RichLog. That is O(n^2) over a meeting (measured 41 ms per repaint at 720 lines, 5.6 s cumulative over 360 segments), and the Stop pass fires one full repaint per changed segment.
- Model installs send progress to the UI once per 1 MiB chunk with no throttle. Pre-verify hashing can produce 1000+ UI events per second.
- The Remote model variant filter remounts every variant row on every keystroke.

Structural notes:
- personas_screen and llm_screen are god modules, but they load lazily and the app's deferred screen pre-importer warms them. personas_screen's first import is +165 ms and 111 modules measured on top of app+chat_screen, and it runs off the loop.
- Boot import path: clean for this slice. All importers of slice modules are the registry, lazy imports or function-local imports.

## clean areas
- tldw_chatbook/UI/Screens/logs_screen.py - thin wrapper; load_from_app on mount only
- tldw_chatbook/UI/Screens/mcp_screen.py - thin wrapper; reload/add/test delegated to workbench workers (their internals belong to the MCP slice)
- tldw_chatbook/UI/Screens/research_screen.py - thin wrapper
- tldw_chatbook/UI/Screens/media_runtime_state.py - dataclass, no importers (dead but zero runtime cost)
- tldw_chatbook/UI/Screens/notes_scope_models.py - dataclasses only
- tldw_chatbook/UI/Screens/model_catalog_consent.py - lazily imported by app.py at 15254 only when consent is required
- tldw_chatbook/UI/Screens/model_browser_state.py and model_memory_presenter.py - pure render-state helpers, regexes compiled at module scope
- tldw_chatbook/UI/Screens/provider_model_resolution.py - pure orchestration; the catalog merge cost it awaits belongs to LLM_Provider_Catalog (another slice)
- tldw_chatbook/UI/Screens/profile_interview_screen.py - every coordinator operation runs as a thread worker
- tldw_chatbook/UI/Screens/model_curated_view.py - catalog load in a thread worker; progress applied in place (recompose only as a fallback)
- tldw_chatbook/UI/Screens/model_external_view.py - recompose only on user reload or operation status
- tldw_chatbook/UI/Screens/model_installed_view.py - inventory and legacy os.walk scan run off-thread and bounded; GGUF import progress already throttled to one event per 64 MiB; recomposes only on user lifecycle actions
- llm_screen.py LAB_SERVER_POLL_SECONDS=2.0 poll - read_server_rows is a cheap process poll and LabScreen.refresh_lab_status is equality-guarded
- llm_screen.py _probe_local_server - already async (TASK-15473)
- personas_screen.py _poll_console_handoff_readiness 4 Hz poll - gated on is_active; readiness compute measured at 23.7 us per tick (build_default_console_session_settings + get_provider_readiness); stopped on unmount
- personas_screen.py library search - 0.2 s debounce with timer cancel (the good pattern); the remount behind it is covered in F1
- personas_screen.py DB and file I/O - consistently asyncio.to_thread / _drain_to_thread (character page, count cache, lore/dictionary ops, imports and exports, visual identity, actor packs)
- meetings_screen.py prepare/start/stop/enroll/offer/voiceprint workers - all thread=True; keyring stays off the UI thread
- first-visit import of personas_screen (+165 ms, 111 modules measured) - warmed off-loop by app.py's deferred screen pre-importer

## census
| Measurement (isolated profile, Textual 8.2.8, M-series) | Result |
|---|---|
| ListView clear+extend of 50 rows, each ListItem(Vertical(Static,Static)) = 200 widgets, plus paint, minimal app at 211x44 | 112-185 ms (6 runs) |
| Same with 200 rows (800 widgets) | 352 ms |
| RichLog(wrap=True) clear + rewrite of 180 / 360 / 720 lines | 10.9 / 28.4 / 40.6 ms |
| Cumulative full repaints over a 360-segment meeting | 5.58 s |
| load_settings() warm, cache hit | 12.8-13.0 ms/call (~600 posix.open) |
| get_api_key("vllm") warm | 12.6-13.5 ms/call |
| get_cli_setting warm (TASK-32804.1 fastpath) | 0.002 ms/call |
| mosaic_from_image(1024x1024 RGBA, 24x10, cover) | 6.4 ms (RGB: 4.0 ms) |
| PNG decode, 1.5 MiB (ConsoleImageRenderCache.prepare, off-thread) | 15.5 ms |
| 1000 Rich Text options + dict copies (refresh_character_list) | 4.1-4.8 ms |
| Personas console readiness compute per poll tick | 23.7 us |
| First import of personas_screen / llm_screen / meetings_screen after app+chat_screen | +165 ms, 111 modules / +42 ms / +5 ms |


# slice-52

## summary
Slice #52 (UI/Screens#6, 28 files, 21,343 lines): the Schedules workbench (scheduling/*, 13.6k lines), the Research workspace screen, and 13 Settings helper modules.

The biggest problem is in Schedules. SchedulesWorkbench is a non-reusable route, so it is rebuilt on every visit. Across it I counted 19 call sites that run synchronous `service.db.*` reads or writes directly on the event loop, plus more through the `async def` reminder methods on SchedulingService. Those methods look async but call sqlite synchronously inline.

Normally this pattern would be a small cost. Here it is severe because of the root cause I measured, which sits outside this slice. ScheduledTasksDB opens a fresh connection for every operation. Each open goes through `connect_private_sqlite`, which on POSIX starts a new child process (`python -I -S private_sqlite_helper_entry.py`) to prepare the file. In an isolated scratch profile, one ScheduledTasksDB read cost 46–51 ms median. A plain sqlite connect plus query cost 0.63 ms. So each DB call on the loop is a ~46 ms freeze.

Hot paths and what they cost:
- **Cursor move onto a reminder row:** 3 reads, 132 ms median (measured). The same cost repeats on every filter debounce, chip switch and queue reload.
- **Visit:** `on_mount` blocks for ~145 ms before first paint. Adding `list_tasks`, the row-0 detail and the probe callback brings it to about 385 ms of loop blocking per visit.
- **Each sync completion:** ~145 ms.
- **Results overlay:** ~140 ms when it opens and after every r/d/o keystroke.
- **Reminder mutations** (toggle, delete, edit): about 90–185 ms, plus a ~190 ms reload.
- **Mark-all-read:** runs 2 connection opens per result in sequence, so 200 unread results take about 19 s of wall time.

The definition-row path already follows the right pattern: a worker, `asyncio.to_thread`, and a stale-selection guard. The reminder path should copy it.

The single highest-leverage fix is a held per-thread connection in ScheduledTasksDB (the store template the codebase already documents). Measured, that would take every one of these costs down about 50x.

The rest of the slice is healthy:
- **Research workspace screen:** it offloads everything with `asyncio.to_thread`, gates layout work, and pages its reads. One minor issue: sources are fetched twice per visit.
- **Settings helpers:** these are mostly pure builders and measured cheap. RAG validation takes 0.05 ms and loading RAG defaults 0.014 ms, so TASK-32804.2 and TASK-32804.1 hold here. Two P3 hygiene items remain: the Image Gen panel re-parses the 102 KB config.toml (3.9 ms) on every compose, and an unused per-key rewrite helper is still in place.

Structural notes:
- `schedules_workbench.py` is a 5,492-line god module. Its route loads lazily, so it adds nothing at boot.
- SchedulingService's async facade over synchronous DB calls is the main layering trap. Callers reasonably assume `await service.x()` does not block the loop.

None of this is covered by an open task. TASK-1320, the umbrella for mount I/O, does not list Schedules.

## clean areas
- tldw_chatbook/UI/Screens/research_workspace_screen.py: every store and adapter call goes through asyncio.to_thread or the to_thread-heavy LocalResearchWorkspaceAdapter (50 offloads). _apply_pane_layout is gated before its query_one work (TASK-23025). Source and note pages are bounded (limit 25 and 20). Workers are grouped and exclusive with exit_on_error=False. Only issue: the double sources refresh (F9).
- tldw_chatbook/UI/Screens/scheduling/definition_detail.py: set_definition only paints the data it is given and does no I/O. The Sources mini-editor mounts only when the user starts an edit.
- tldw_chatbook/UI/Screens/scheduling/task_detail.py: TaskDetail and TaskInspector only paint data (about 15 cached query_one calls per set_task). The problem is the caller, not this module (F1).
- tldw_chatbook/UI/Screens/scheduling/unified_rows.py: build, filter and sort are pure O(n), and the filter runs behind a 200 ms debounce. Measured: 500 reminders convert in 2.4 ms.
- tldw_chatbook/UI/Screens/scheduling/forms/automation_definition_form.py, reminder_form.py, new_task_choice_modal.py: preview and save run only on button press, in grouped exclusive workers. The timezone options come from a curated list; available_timezones() is never called.
- tldw_chatbook/UI/Screens/scheduling/definition_audit_view.py: fills itself from a worker at mount. tldw_chatbook/UI/Screens/scheduling/sync_status_widget.py and workbench_host_screen.py: trivial.
- Schedules timers: the 60 s next-run ticker and the 5 s liveness ticker pause in on_screen_suspend and skip when the screen is covered. The tick re-render skips the DB for an unchanged selection (from_tick guard). The liveness tick reads only a small heartbeat JSON file (sub-ms).
- Schedules SSE notification observer: scoped to the workbench, stopped by an Event in on_unmount, with flat backoff and debounced single-flight results pulls.
- Schedules definition-row detail: the three count reads run in asyncio.to_thread inside a grouped exclusive worker with a stale-selection guard. This is the template F1 should copy.
- tldw_chatbook/UI/Screens/settings_rag_profile_adapter.py and settings_library_rag_defaults.py: measured hard_config_errors at 0.046 ms, validate_library_rag_defaults at 0.048 ms and load_rag_defaults_from_active_profile at 0.014 ms, so the TASK-32804.1 and TASK-32804.2 fixes hold. Profile file operations run in thread workers.
- tldw_chatbook/UI/Screens/settings_config_adapter.py: load() deepcopies the whole config, measured at 0.35 ms, which is fine. The advanced-config editor (settings_advanced_config.py) sends every read, validate and replace through asyncio.to_thread.
- tldw_chatbook/UI/Screens/settings_endpoint_probe.py: runs only when the user clicks Test, uses a short timeout, and opens a short-lived AsyncClient per probe. That is acceptable for a click-triggered action.
- tldw_chatbook/UI/Screens/settings_privacy_security.py, settings_network_defaults.py, settings_context_memory.py, settings_appearance_defaults.py, settings_config_models.py, settings_rag_definition_actions.py, settings_provider_view_model.py: pure builders and dataclasses. The keyring read is already off-thread (TASK-32926). Provider picker filtering is pure CPU over roughly 50 entries.

## census
| Trigger (Schedules workbench) | Site | Sync ScheduledTasksDB opens on loop | Loop cost |
|---|---|---|---|
| Cursor move onto a reminder row | schedules_workbench.py:1850 -> 2044-2045 | 3 (4 if to_server_failed) | 132 ms median (measured) |
| Filter debounce / chip switch / every load_tasks render | _render_table -> _update_detail_for_index | 3 | ~132 ms |
| Visit on_mount (before first paint) | :835, :838, :839 | 3 | ~145 ms (get_sync_state 50.7 + count_unread 47.2 measured + get_conflicts) |
| Visit load_tasks worker (coroutine) | :1333 service.list_tasks | 1 + pydantic rows | 57 ms at 500 rows (measured) |
| Reachability probe done | :4741 | 1 | ~50 ms |
| SyncCompleted / SyncFailed | :4962-4965, :4982-4985 | 3 each | ~145 ms |
| Results overlay open, and each r/d/o | :2466-2479, results_tab.py:699-701 | 3 | ~140 ms + decode of up to 200 rows |
| Mark all read (rail or 'a') | :4412-4418, results_tab.py:619 | 2 on loop + 2 per result off loop | ~95 ms + ~95 ms per result wall time |
| Toggle / delete / edit / duplicate reminder | :2280, :2342, :2990, :3102, :4558, :5150, :5349 (service facade) | 2-4 + reload | ~90-185 ms + ~190 ms reload |
| Acknowledge incident | :3015 + re-render | 1 + 3 | ~180 ms |
| Conflicts overlay push | :2548 | 1 | ~46 ms |
| Per-open cost baseline | DB/private_sqlite.py:1878 prepare_in_helper -> private_sqlite_process.py:367 Popen | n/a | 38.8 ms connect vs 0.63 ms raw sqlite connect+query (measured) |


# slice-53

## summary
Slice #53 is one file: tldw_chatbook/UI/Screens/settings_screen.py (33,091 lines; SettingsScreen alone is about 30k lines and 1,076 methods). It is the F4 Settings destination.

Hot paths:
1. A fresh screen visit. The route is not reusable, so every visit builds a new screen.
2. A category switch: _select_category -> watch_active_category -> run_worker(_swap_category_panes) -> two SettingsRegion.recompose() calls.
3. Input.Changed keystrokes. There are 71 handlers; they stage a draft and then run a widget-refresh cascade.
4. DescendantFocus / Tab moves.
5. A 4 Hz subscription-readiness timer.

Method: I mounted SettingsScreen headless in a bare Textual App that loads the real CSS bundle, at 211x44, on an isolated scratch profile. I measured main-loop busy time by timing asyncio Handle._run, measured GC with gc.callbacks, and used cProfile only to attribute cost. I checked that the real profile was untouched before and after.

Overall result: steady-state config reads are healthy now. get_cli_setting takes about 1.5 µs warm, and the TASK-32804.2 RAG cache holds: a Library/RAG keystroke costs about 10 ms of loop time.

The dominant costs are structural:
- **Category switches rebuild both panes from scratch.** The heavy categories cost 215-900 ms of loop time per switch. Textual stylesheet.apply is 55-65% of that. About half of each heavy pane's widgets sit in display:none subtrees. Every mounted control also posts a Changed echo, which runs the full staging cascade.
- **GC pauses follow the churn.** A gen2 GC of 120-360 ms lands every 1-2 switches, and the app never calls gc.freeze.
- **The screen is rebuilt on every visit.** A fresh Settings visit costs 356-513 ms of loop time.
- **Keystrokes are heavy.**
  - Every keystroke writes identical text into Statics with layout=True, which forces about 2 full layout passes per key on 300-400-widget panes.
  - The Model field re-derives the whole model-profile block on every key.
  - Two instant-persist inputs (permission summary, model-catalog stale-hours) rewrite config.toml on every keystroke. That doubles keystroke latency to about 140-150 ms.

Secondary issues:
- load_settings() and get_runtime_config_snapshot() still pay the admission handshake warm, about 9 ms each; the TASK-32804.1 fast path covers only get_cli_setting.
- Providers & Models Save writes config synchronously on the loop, about 108 ms.
- The Workspaces compose does N+1 sqlite queries plus stat() calls on the loop.
- The Image Gen panel scans the template directory on the loop.
- The Speech panel restyles its whole subtree twice on mount because it reads width 0.
- The module import pulls in the 1.38 MB console_chat_controller just for one enum.

Measurement caveats: the bare host has fewer live objects than TldwCli, so real gen2 pauses are likely longer. The scratch HOME has 13 path components, so the admission costs here are roughly 3x what a real ~/.config profile would show.

## clean areas
- _poll_subscription_readiness 4 Hz timer (settings_screen.py:4180/4245): measured 2-3 ms total loop-busy per 5 s idle on Overview/Providers/Appearance; it is gated on an (category, provider, status) observation tuple and stopped in on_unmount; the Settings screen is unmounted on navigate-away (non-reusable route)
- Library/RAG per-keystroke path: the TASK-32804.2 loaded-defaults cache holds; measured about 10 ms loop-busy per keystroke (lowest of all categories)
- get_cli_setting warm read: 1.5 us (TASK-32804.1 fast path effective); SettingsConfigAdapter().load() is 0.34 ms (a deepcopy only)
- All @work(thread=True) writers, probes and loaders are off-loop: _refresh_sync_rows, _rag_index_status_worker, _rag_backfill_worker (its asyncio.run runs inside the thread worker), _image_gen/_video_gen_panel_load_worker (TASK-32926), _persist_model_catalog_section_values, _persist_console_toggle / permission-summary drain workers (single-writer coalescing), and the storage/appearance/console-behavior save workers
- _perform_runtime_source_switch now uses asyncio.to_thread (TASK-32804.12 progress confirmed)
- _load_speech_tts_default_profile_choices and _apply_console_exchange_capture: they await async services; runtime_capture_policy is cached per config generation
- Tool Profiles: the listing is fetched off-thread, and _sync_tool_profile_operations is revision-gated (cheap on the 4 Hz tick)
- Category search per keystroke: static summaries, _ownership_by_category is cached, and the cost is about 17-25 ms loop-busy on 143 widgets (minor)
- on_key / on_resize / _sync_responsive_workbench: cheap; _sync_responsive_workbench only acts on real width changes
- SettingsURLInput.render_line: mirrors Textual's Input and only adds the autolink break under textual-web
- _get_internal_prompts_customized_count: memoized per screen instance (paid once per fresh visit, about 15 ms profiled; see the reuse finding)
- Module-level RAG adapter seams (activate_profile, fetch_index_status, ...): lazy imports per ADR-097, no import-time work
- _fold_long_tokens regexes: string patterns served by Python's re cache; not hot

## census
| category (211x44, bare host) | widgets | loop-busy r1 / r2 (ms) | max single handle (ms) | gen2 GC seen |
|---|---|---|---|---|
| fresh Settings visit (push) | 143 | 356 / 513 | 180-195 | 88 ms |
| providers-models | 328 | 477 / 330 | 38-302 | 239 ms |
| console-behavior | 411 | 544 / 650 | 70-285 | 190 ms |
| speech-tts | 351 | 681 / 570 | 103-297 | 120-278 ms |
| image_generation | 353 | 603 / 442 | 76-297 | 230-296 ms |
| library-rag | 280 | 899 / 324 | 28-620 | 329-356 ms |
| appearance | 219 | 301 / 215 | 25-172 | 133 ms |
| storage | 149 | 105 / 126 | 18-305 | 299 ms |
| network / personas / skills / mcp-defaults etc. | 94-110 | 37-60 | 7-12 | none |
| keystroke: model field / console temperature / appearance / storage / library-rag | - | 31-37 / 30-49 / 23 / 13-20 / 10 per key | - | - |
| Tab focus move: network(108) / storage(149) / appearance(219) / library-rag(280) / providers(328) / console-behavior(411) | - | 25-33 / 31-37 / 40 / 42 / 49-58 / 61-65 | - | - |
| instant-persist keystroke wall: permission-summary model / stale-hours | - | 139 / 150 median (245 max); 1 config write per key | - | - |


# slice-54

## summary
Slice #54 (UI/Screens#8, 16 files, ~26.8k lines). Hot entry points traced: WatchlistsCollectionsScreen (a lazily routed tab, not reusable; on_mount starts 5 loaders; the Reader j/k path; tree clicks; about 20 write verbs), TrajectoryScreen (a modal pushed from the Console via review_selection -> wiring._present_console_trajectory; it polls at 2 Hz and has live search), VideoPlayerScreen (a modal from Console_Modules/video.py; a 24 fps frame pump), and the STTS, Study, Workflows, Stats and Writing tabs (lazy routes, pre-imported in a background thread after first paint by app._preimport_heavy_screens). The settings_* modules are data/validation helpers imported by settings_screen. settings_speech_tts is also on the BOOT import path (app.py -> Event_Handlers/STTS_Events/stts_events.py and speech_tts_panel_types).

Overall health: the Watchlists screen has had heavy prior tuning. TASK-2200, 15461, 15464, 15778 and 19562 left debounced search and count refreshes, a serial surface-refresh queue, DB reads off the loop via run_db_off_loop/to_thread, and layout persistence on a thread. The worst residue is structural, not polling:
- Every tree-data load remounts the whole left-rail WatchlistTree, one Button per node or source with no virtualization. That includes the counts-only reload 0.6 s after each unread item is opened.
- The tree write flows run synchronous SubscriptionsDB writes inside coroutine workers.

The new, measured hot costs sit in two modals:
- The Trajectory ledger re-renders synchronously on every search keystroke, including a full-trace json.dumps/lower() match pass. Measured 8-45 ms per keystroke at 2k records. Live refreshes also do full rebuilds on the loop, measured 20.5 ms.
- The video player converts full-resolution frames and builds renderables on the UI thread at 24 fps. The kitty path costs about 24 ms per frame, which is about 58% of the loop. ffmpeg has no scale filter, so 1080p frames are 6.2 MB each.

Smaller items: the Speech screen chain drags numpy into every session through the background pre-import (the "lazy" dictation service imports numpy at module scope). The Study dashboard's 'fixed' offload misses the primary service path. There are repeated uncached scoped_source_rows SQL calls on the loop, a Video Gen panel that re-parses config.toml on every compose, and one-shot config writes on the loop.

Nothing in this slice is P0. There are 3 P1s (trajectory search, video frame render, watchlists rail rebuild). No open task covers them. TASK-21134 (Done) throttled only the trajectory brush drag, not search or live refresh.

## clean areas
- tldw_chatbook/UI/Screens/settings_search_index.py -- static label table built once at settings_screen import; per-keystroke Settings search over it is a few hundred str.lower() calls (sub-ms)
- tldw_chatbook/UI/Screens/settings_storage_defaults.py -- pure path validation; a handful of stat() calls on Settings actions only
- tldw_chatbook/UI/Screens/settings_web_search.py -- save/test/probe go through asyncio.to_thread; __init__ uses the cached config read (only discard() force-reloads, noted as P3)
- tldw_chatbook/UI/Screens/settings_speech_tts.py -- pure validators/state; deepcopies are of small per-field dicts; only cost is its ~6 ms boot-path import (P3 finding)
- tldw_chatbook/UI/Screens/stats_screen.py -- @work(thread=True) load, 21 bounded queries (LIMIT 1000/5000) off-loop, set_reactive-batched single rebuild per load
- tldw_chatbook/UI/Screens/stts_screen.py -- compose-time dependency probe is find_spec only (measured 0.2-1.1 ms for 9 modules); rail highlight is a small query; the only issue is the transitive numpy import (P3 finding)
- tldw_chatbook/UI/Screens/study_scope_models.py -- tiny dataclasses, 1 ms import
- tldw_chatbook/UI/Screens/tools_settings_screen.py -- DEPRECATED, route aliased to MCP; only lazily referenced from UI/Screens/__init__ map, never loaded in production
- tldw_chatbook/UI/Screens/writing_screen.py -- thin wrapper around WritingWindow
- tldw_chatbook/UI/Screens/workflows_screen.py -- controller load/select/library paging/draft persistence all via asyncio.to_thread with debounced draft flush; per-keystroke field edits are small in-memory JSON projections (P3 note only)
- tldw_chatbook/UI/Screens/video_player_screen.py status timer (4 Hz, stopped in _invalidate_current/on_unmount) and pipeline cleanup on a thread worker
- tldw_chatbook/UI/Screens/trajectory_screen.py revision poll (2 Hz in-memory dict read via ConsoleChatStore.get_payload_revision, stopped on unmount), import via asyncio.to_thread, initial >5000-record render on a thread
- tldw_chatbook/UI/Screens/watchlists_collections_screen.py: layout persistence (thread worker + generation lock), items search debounce 0.3 s, tree counts debounce 0.6 s, run-progress tick fingerprint gating, lazy item-content fetch, serial call_next surface-refresh drain, briefing/preset/settings DB ops via asyncio.to_thread, webbrowser.open on @work(thread=True), local item/list/run/rule reads via run_db_off_loop in LocalWatchlistsService (normalization measured 0.7 us/row)
- tldw_chatbook/UI/Screens/skills_screen.py SkillsScreen list/trust calls use asyncio.to_thread (module is nav-dead; structural P3 only)

## census
| Measurement (isolated env, Python 3.12 / Textual 8.2.8) | Result |
|---|---|
| TrajectoryScreen._render_ledger per search query, 2,000 records, 8 KiB tool results | 8.1 / 45.3 / 12.3 / 12.6 / 14.6 / 9.6 ms (q = l, lo, lor, zzz, zz, empty) |
| TrajectoryScreen._matching_records (all records), q=zzz vs empty | 9.35 ms vs 0.19 ms |
| TrajectoryScreen._apply_live_snapshot, 2,000 records | 20.5 ms |
| Static.update + layout + paint, 8 / 64 / 256 KiB text (includes ~30 ms of pilot.pause overhead) | ~30-40 / ~45 / ~95 ms |
| Video frame on the UI thread, 1080p: kitty (TGP construct + render) | 24.4 ms/frame on a realistic frame, 42.7 ms on noise |
| Video frame on the UI thread, 1080p: halfcell (frombytes + copy + thumbnail + Pixels + rich render) | 3.7-3.9 ms/frame |
| Video frame on the UI thread, 1080p: ascii | 1.5 ms/frame |
| Remove + mount + paint of N compact Buttons in a rail (bare app, 211x44), N = 50 / 200 / 400 | ~55 / 100-147 / 243-267 ms |
| stts_screen first import after Console loaded | 90-102 ms (numpy 20-32 ms of it) |
| numpy import | 32 ms, +10.8 MB maxrss |
| Pre-import sweep: first route to load numpy | stts_screen |
| settings_speech_tts cumulative import on the app boot path | 6.3 ms |
| tomllib.load of the default 102 KB config.toml | 3.2-3.5 ms |
| shutil.which x3 | 0.45 ms |
| find_spec probe x9 (Speech rail) | 0.2-1.1 ms |
| normalize_watchlist_item | 0.7 us/row |
| First import, other slice screens (after Console loaded): watchlists / workflows / trajectory / study | 14.5-17.9 / 6.6-7.3 / 5.3-5.5 / 3.1-3.7 ms |


# slice-55

## summary
Slice #55 UI/Speech: 19 files, 14,655 lines. It is the whole Speech tab (the Playground, Studio settings, Voice Blends and the audio.cpp runtime card). Its host is UI/STTS_Window.py, which sits outside the slice and is reached through the lazy, non-reusable `stts` route.

Hot paths found:
1. **Tab visit and view switch.** Every Speech visit, and every return to Playground from another Speech view, re-mints SpeechPlaygroundPane (161 widgets). The host calls `remove_children` and then mounts a fresh pane.
2. **Discovery on mount.** Each mount starts catalog and voice discovery through `@work` async workers. These are NOT threads, so their sync work runs on the event loop.
3. **Keystrokes.** The TTS text box and the speed Input run handlers on every keystroke.
4. **audio.cpp poll.** A 5 s poll runs while audio.cpp is selected, and audio.cpp is the default provider.
5. **Playback loop.** A 10 Hz progress loop runs for the whole of playback.

Measured findings:
- **Config deep copy on the loop.** `get_runtime_config_snapshot()` deep-copies the whole settings tree on the event loop, 12–26 ms per call. It runs once per Playground mount from the host, and from the catalog mixin when its revision guard passes. The consumer only reads three sections.
- **Mount cost.** A Playground mount plus settle took 146–230 ms headless. The always-composed audio.cpp runtime card costs 10–22 ms of that even when another provider is selected.
- **Voice-profile scans.** Directory and JSON scans run synchronously on the loop inside `_apply_catalog`: 8–11 ms per call for Chatterbox/Higgs (33–41 ms first call), 1.4–1.8 ms for OmniVoice at 50 profiles. `_apply_catalog` runs at least twice per catalog load and on most select changes.

Code-read findings:
- Play waits a fixed 200 ms, and Pause/Resume a fixed 100 ms, before acting.
- The poll and progress loops repaint and relayout every tick even when nothing changed.
- Keystrokes and each catalog apply fan out into redundant axis-row and tooltip repaints.
- Small blend-JSON reads and the audio export copy run on the loop.
- About 1.8k lines of dead legacy settings-form code remain: speech_settings_group.py is used only by tests, and most of SpeechSettingsMixin is unreachable.

No open backlog task covers any of this; the only open task that mentions the slice is ruff debt, TASK-27006.

Outside the slice, for the owner of UI/STTS_Window.py: the first Speech visit imports numpy through Dictation_Window_Improved → Audio/dictation_service_lazy (+87 ms, 98 modules). The playground pane's own first import adds 40–60 ms (65 modules, including all of textual_fspicker, stts_profile_library and Persona widgets). This is first-visit only, because the route is lazy.

Aside: `_catalog_test_fingerprint` compares a registry-wide settings generation (0 until a save) with a per-slot revision (starts at 1). That comparison makes the catalog-evidence path, and its snapshot copy, rare at startup. So F1's per-mount cost is dominated by the host site.

Overall health: mostly good. The pane has workers with generation fences, off-thread canonicalization and Studio load/save, bounded stores, paused hidden progress clocks, and timers that stop on unmount. The remaining costs fall into three groups: (a) whole-tree remount per view switch, (b) sync work inside async (non-thread) workers, and (c) unconditional repaints on timers and keystrokes.

Suggested PR groups:
- **PR-A (low risk):** F1, F4 and F9 — remove the config deep copies, move voice/blend and export I/O off the loop and cache it.
- **PR-B (medium risk):** F2 and F3 — keep the Playground mounted across Speech views, lazy-mount the audio.cpp card, and pass the resolved provider into the constructor.
- **PR-C (low risk):** F5, F6 and F8 — change-gated repaints for the poll, keystroke and playback paths.
- **PR-D (low–medium risk):** F7 — delete the sleep barriers.
- **PR-E (low risk):** F10 — delete the dead legacy code.

## clean areas
- speech_settings_pane.py SpeechSettingsPane: Studio preferences load/save/reset all via asyncio.to_thread in workers; per-keystroke _input_changed -> _sync_source_copy/_sync_dirty_state is cheap (~10 query_one, dataclass build); responsive layout set_class is idempotent
- speech_profile_mixin.py: evidence recording via asyncio.to_thread, profile save delegated to handler through awaited service, exclusive named workers
- speech_playground_pane.py clone-reference validation: 0.3 s debounce via cancelled asyncio tasks, canonicalize_reference_wav on asyncio.to_thread, revision/context fences, joined on unmount
- speech_playground_pane.py _cli_setting override: axis/model/voice/format/speed seeds served from in-memory Studio/global snapshots (get_cli_setting fallback measured 0.016 ms)
- speech_runtime_status.py: SpeechTTSRuntimeStatusStore bounded by provider and provider×model keys; projections are pure in-memory
- speech_settings_contracts.py / speech_settings_model.py / speech_playground_model.py: constant tables only; boot-path import (via app.py -> stts_events -> settings_speech_tts) measured ~2.1 ms self for contracts + 1.2 ms runtime_status
- speech_param_group.py: provider-scoped rows passed to Collapsible, no recompose
- speech_clone_setup.py, speech_action_strip.py, speech_effects_pane.py: small, in-place updates, no timers or I/O
- audio_cpp_runtime_card.py diagnostics: RichLog max_lines=200 and rewritten only when the diagnostic tuple changes
- Hidden progress bars use PausableProgressBar (TASK-23022), so the hidden generation/playback bars arm no idle clocks
- Timer/worker lifecycle: stts route is non-reusable, so the 5 s poll and pane workers stop on unmount; catalog/voice/playback worker groups are exclusive and cancelled in on_unmount
- speech_synthesis_mixin.py _generate_tts: per-press query_one fan-out only, then post_message; no I/O
- AsyncAudioPlayer calls are all run_in_executor (no sync audio I/O on the loop)
- speech_playground_pane.py _sync_truthful_status_rows / status projections: in-memory, event-driven (not timer-driven)

## census
| Measurement (isolated scratch profile, Textual 8.2.8, headless run_test 235x52) | Result |
|---|---|
| get_runtime_config_snapshot() per call (default ~100 KB config, ~73 KB settings JSON) | 11.8 / 25.8 / 16.4 ms |
| load_global_speech_tts_state(snapshot.values) | 0.12–0.30 ms |
| get_cli_setting (cache hit) | 0.016 ms |
| SpeechPlaygroundPane mount+settle (catalog stubbed) | 146–230 ms, 158–161 widgets |
| baseline: Vertical of 158 plain Statics, same harness | 56–115 ms (median 62) |
| SpeechAxisRow alone | 61–123 ms (median 85) |
| AudioCppRuntimeCard alone | 35–77 ms (median 51), 28 widgets |
| Pane A/B with vs without runtime card (provider=kokoro) | 176.9 vs 154.6 ms; 175.2 vs 165.5 ms (median) |
| _chatterbox_profile_choices (dir present, empty store) | 33–41 ms first, 8–10 ms steady |
| ChatterboxVoiceManager ctor / list_profiles | 5.7–7.4 ms / 5.0–6.0 ms |
| _omnivoice_profile_choices, 5 / 50 profiles | 0.15 / 1.4–1.8 ms steady (first 11 / 5 ms) |
| _kokoro_blend_choices (absent file) | 0.01 ms |
| First-visit incremental import: speech_playground_pane after app | 39–60 ms, 65 modules |
| First-visit incremental import: STTS_Window after pane (numpy via Dictation) | 87 ms, 98 modules |


# slice-56

## summary
Slice #56 UI/Watchlists_Modules (27 files, 14,680 lines). Every module is imported only lazily, through the screen registry's watchlists_collections route (plus one function-local import in library_artifacts_controller), so none of it is on the boot path.

Hot paths I traced from WatchlistsCollectionsScreen:
(a) The Read loop. j/k, a click or space fires ItemSelected, which sets ContentPane.item (recompose), pushes the item to InspectorPane (recompose), and runs mark-read-on-open. That mark-read calls _request_tree_counts_refresh, and 0.6 s later _load_tree_data fires, which unconditionally rebuilds the whole left rail (WatchlistTree) and the centre header.
(b) Reader page arrival (ArticleListPane._rebuild_rows). Fires on tree-scope clicks, filter changes, the 0.3 s-debounced search reload and Next/Prev.
(c) Per-keystroke search in the Reader and Sources panes. This is now cheap: about 5 ms in the Reader, and about 2 ms sync plus the table repaint in Sources.
(d) Tree expand/collapse/tag clicks, which are recompose=True on the whole tree.
(e) Artifacts: settings Selects, script selection and briefing reloads.
(f) Notifications row selection.
(g) Per-visit compose. The route is not reusable, so the screen is rebuilt on every visit.
(h) RunsPane.run_poll. This is the only timer in the slice: 1 Hz, bounded to 60 ticks, gated on selection and running status, and it does its DB read off-loop.

Overall health is good. The slice has already had a lot of perf work: task-15460/15461/15779/16852/2200/15464/15776 removed the per-keystroke teardowns and the screen-level recomposes. All DB access goes through the backend controller to scope_service to LocalWatchlistsService with run_db_off_loop, or through asyncio.to_thread in the modals.

What remains is one structural pattern: 'recompose as an update mechanism' on four widgets whose data changes often.
- WatchlistTree has no in-place count patch, so every count refresh remounts 55–430 widgets (57–345 ms measured). It also runs a synchronous source-row query per expanded watchlist inside compose.
- ContentPane and InspectorPane recompose on every article open. That costs about 30–45 ms more than an in-place update, plus 24 ms when the Inspector is open.
- ArtifactsPane recomposes the whole pane (94 ms) for scalar settings the Select already shows.
- NotificationsPane rebuilds its 100-row table on every highlight (49 ms).

Two further findings:
- An uncached secure-path walk (get_user_data_dir via briefing_audio_dir, about 46 ms per call, roughly 1,850 open() syscalls) runs inside ScriptDetailRegion.compose. The root cause is cross-cutting (get_user_data_dir has no cache).
- A synchronous Home active-work adapter fan-out runs in compose_content on every visit, because its cache TTL is 3 s.

Minor items: deepcopy of all cached reader pages on every Next page, overview counts computed by fetching 100 full item rows, a dead ItemsPane class, and small selection-set copies in SourcesPane.

All timings came from headless Textual run_test probes in an isolated scratch profile (HOME, XDG_* and TLDW_CONFIG_PATH under scratch). The pilot.pause baseline of about 21 ms was subtracted. Scripts are in scratchpad/audit/scratch/s56/.

## clean areas
- runs_pane.py: run_poll is bounded (60 x 1 s), exits on deselect/non-running, posts RunProgressTick whose screen handler does one off-loop get_run and a fingerprint compare; selection/highlight/items/logs all patched in place
- rules_pane.py: selection is not recompose; only form open/close and data arrival recompose (small pane)
- sources_pane.py per-keystroke search/filters: plain reactives re-populating DataTable in place (measured 1.8 ms sync per keystroke with 100 sources, ~17 ms repaint); selection highlight patched via update_cell
- article_list.py per-keystroke search: visibility toggle over mounted rows with no-op guards (measured ~5 ms over baseline); screen reload is 0.3 s debounced; row repaints are single-row
- watchlists_backend_controller.py: all list/get calls route to async scope/local services that use run_db_off_loop (verified LocalWatchlistsService.list_sources/list_items/list_reader_items_page/get_run); no sync DB on loop
- briefing_preset_modal.py / kept_briefings_modal.py: DB via asyncio.to_thread, cast LLM call via to_thread inside briefing_cast; recompose confined to modal (cold path)
- bulk_sources_modal.py, opml_dialogs.py, snapshot_view_modal.py: cold modal paths; parsing only on submit/open
- overview_pane.py: pane-scoped recompose of ~10 widgets on data arrival only (screen patches it via watch_overview_data, no screen recompose)
- region_layout.py (pure), region_layout_store.py (load uses cached get_cli_setting; save runs in run_worker(thread=True); write-on-load only for one-time migration)
- pane_grip.py, watchlists_tab_strip.py, humane_time.py (thin re-export; parse is fromisoformat), table_selection.py (row_with_id O(n) on <=100 rows is fine)
- watchlists_workbench.py: layout transitions mount/remove only changed side bodies under a lock; no polling
- No set_interval, threads, subprocess, keyring, httpx clients or eager heavy imports anywhere in the slice; slice is off the boot import path (lazy screen registry)
- Scheduling.services function-local import in SourcesPane.source_next_check_text is a sys.modules hit (app.py already imports it at boot) -- no cost

## census



# slice-57

## summary
Slice #57 UI/Wizards: 9 files, 17,225 lines. FirstRunSetupWizard.py alone is 10,860 lines (the god-module point is already known: tier-2 S21 / TASK-32809.2). Three entry points:
- **Boot:** app.py imports first_run_setup_state on every start to run pure predicates. The only cost is +3.3 ms, because the package __init__ drags in BaseWizard.
- **First-run wizard:** pushed at first boot (the first screen a new user sees), and again on rerun from Settings, the command palette or recovery resume.
- **Chatbook wizards:** Chatbooks tab, Create and Import.

All numbers below come from isolated probes: scratch HOME/XDG/TLDW_CONFIG_PATH, a headless run_test at 211x44 with the real 422 KB CSS bundle, and network discovery stubbed.

**Main costs, most expensive first:**
1. **Theme preview on every arrow key (F3).** The Appearance step's radio group selects whatever is highlighted, and each selection sets app.theme. That forces a full stylesheet reparse and restyle: about 420 ms per key measured, and 0.9–1.2 s in the real app per TASK-33075.
2. **Chatbook export/import run on the event loop (F4).** They call the synchronous ChatbookCreator/ChatbookImporter inside run_worker(coroutine), which is not a thread. The UI freezes for the whole export or import, and preview unpacks the entire archive just to read manifest.json.
3. **All 11 wizard steps built up front (F1).** The wizard mounts 297 widgets at open. Measured 250–366 ms until mounted and 475–629 ms until idle, mostly styling cost. Only Welcome (9 widgets) is visible, and the default Quick track never shows 5 steps (136 widgets).
4. **Config file parsed on every arrow key in the provider list (F2).** Each arrow key reads, locks and parses the full ~100 KB config.toml on the event loop (17 ms per call; select_provider mean 20.5 ms). With encryption on it also decrypts every stored key (about 21 ms each), and the discovery worker adds two more load_settings reads.
5. **Speech step rebuilds itself for every state change (F6).** One rebuild costs 137–203 ms, and a language change costs about 320 ms.

**Smaller items:**
- Chatbook Create does 7 synchronous database reads on the event loop when the wizard opens, pulling full note and briefing bodies (F5).
- Every step's on_show runs twice (F7): show_step calls it directly and Textual's Show event calls it again.
- About 51 ms of imports on the first-run first paint (F12).
- Two full config rewrites per Next (F11).
- The known 4 Hz readiness poll now runs from wizard open (F9).
- INFO-level logging on every keystroke in the Chatbook wizard (F10).

**Correctness bug (not a speed issue):** arrowing onto Cerebras, Fireworks, Together or custom_hosted in the provider list makes select_provider → read_provider_secret_presence → canonical_provider_key raise ValueError('Provider is not supported.') inside an event handler. Reproduced in the harness; in the real app this would likely crash it. It may relate to TASK-32919 (which covers the same four providers in the Console settings modal); it should be routed separately.

**Proposed PR groups:**
- **(a) Theme preview:** debounce the preview (F3).
- **(b) Chatbook wizards off the event loop:** F4 + F5 + F10.
- **(c) Lazy step bodies:** F1 + F7 + F12, plus the F9 timer arm/stop.
- **(d) Provider list:** debounce selection and move config snapshots off the event loop (F2).
- **(e) Speech step:** limit rebuilds to the status panel (F6).
- **(f) Config writes:** one write per Next (F11).

## clean areas
- tldw_chatbook/UI/Wizards/first_run_setup_state.py: pure logic; boot-time predicates (setup_recovery_action, should_offer_wizard, any_provider_configured) cost microseconds; build_summary_rows is pure. The only cost is the import tail covered by F8
- tldw_chatbook/UI/Wizards/first_run_speech_step_state.py: pure; routing policy and registry are built once at module scope and cached
- tldw_chatbook/UI/Wizards/first_run_voice_step_state.py: sample synthesis is async httpx with an explicit timeout, called from a worker
- tldw_chatbook/UI/Wizards/first_run_recovery_dialog.py: trivial modal, no I/O
- FirstRunSetupWizard ProviderStep per-keystroke API-key and endpoint handlers: helpers measured at µs level (resolve_provider_endpoint 7 µs, get_provider_readiness(anthropic) 11 µs, read_provider_secret_presence 1.4 µs); the cost is a dozen query_one calls, well under 1 ms
- SetupWizardContainer.commit_config and persist_setup_checkpoint: writes run in the executor; _mirror_into_app_config is an in-memory merge
- SummaryStep._render_rows: config reload, managed-artifact existence check and onnx runtime probe all run through run_in_executor; the re-render is guarded by is_running
- ProtectKeysStep: scrypt encryption runs through run_in_executor
- Speech and Voice install, preflight, provision, activate and delete paths: @work(thread=True) with asyncio.run inside the worker thread; install progress updates ModelInstallProgress in place with no rebuild
- SetupWizardProgress.set_items: equality-guarded rebuild of about 50 widgets, once per step change
- Optional-dependency probes (embeddings_rag_deps_installed 0.13 ms warm, parakeet_onnx_deps_installed 0.02 ms): find_spec only
- _probe_first_run_provider_connection: one httpx client per user-pressed Test, deliberately scoped so the secret is not retained; not a pooling problem
- ChatbookImportWizard file selection, conflict and options steps, and ChatbookCreationWizard basic-info and preview steps: small forms with no DB or file I/O beyond one-off path resolution
- BaseWizard navigation (show_step, update_progress): a few query_one calls per step change

## census
| Measurement (isolated probe, audit tree 840ed2ca58) | Value |
|---|---|
| Boot: marginal import of first_run_setup_state after app import | +3.3 ms (BaseWizard 1.0 ms via package __init__) |
| First wizard open: import FirstRunSetupWizard module | +29.5 ms, +42 modules |
| First wizard open: ToolsStep compose imports Agents.tool_catalog | +21.4 ms, +14 modules |
| Wizard widgets mounted at open (all 11 steps) | 297 (welcome 9, provider 26, model 8, voice 45, rag 18, speech 46, tools 43, notes 4, appearance 25, protect 5, summary 15) |
| Wizard push → mounted / → idle (warm CSS, 211x44) | 251–366 ms / 475–629 ms |
| get_atomic_config_snapshot (102 KB config, no encryption) | 15.9–17.3 ms/call |
| load_settings warm / get_cli_providers_and_models | 6.0–6.9 ms / 6.6 ms |
| ProviderStep select_provider per arrow key | mean 20.5 ms, max 29.7 ms (6 keys, 6 snapshots = 103.9 ms) |
| scrypt N=16384 r=8 (per encrypted config value) | 21.5 ms |
| AppearanceStep theme arrow key (press + settle) | 411–431 ms |
| SpeechSetupStep single rebuild + settle | 137–203 ms; language change about 322 ms (2 rebuilds) |
| save_settings_to_cli_config (worker) | 43 ms steady; 187–239 ms first write at wizard open |
| on_show calls per step show (ModelStep / VoiceSetupStep) | own on_show 2x, WizardStep.on_show 3x |


# slice-58

## summary
Slice #58 Utils (63 files, ~25k lines). 41 of these modules load at `import tldw_chatbook.app` and 9 more by `_ui_ready`. The hot entry points are: `paths.get_user_data_dir` (a thin wrapper around config.get_user_data_dir, called from 203 sites; every DB-path accessor goes through it), `private_paths` (the no-follow directory walk behind every private open), `sensitive_paths` (the agent file-tool denylist), `input_validation` (on the composer keystroke path through the send-price check), `log_sanitizer` (applied to every INFO+ record), `token_counter`/`tiktoken_runtime` (token estimation), `ui_responsiveness` (always-on watchdog), `tls_trust`/`egress` (HTTP client factories) and `textual_css_fastpath` (styling every node).

Headline: **get_user_data_dir has no memo.** Each call re-runs storage admission, the default-root lock and `secure_private_directory`. A profile showed about 57 walks from the filesystem root and about 1,600 `open()` syscalls per call. Measured cost is 27–35 ms per call at a 15-component scratch path and 48 ms at 21 components, about 2.2 ms per component, so roughly 14 ms at the default `/Users/<u>/.local/share/tldw_cli/default_user`. The faster path (`verified_user_data_directory`) only applies inside an admitted raw operation, so ordinary callers never get it. Every `get_*_db_path` accessor pays the same cost (`get_chachanotes_db_path` measured 26 ms). The TldwCli.__init__ region calls get_user_data_dir 20 times and DB accessors 16 times before first paint, and Console `compose_content` calls it on every visit through `default_prompt_history_path` (whose docstring says 'IO-free').

`sensitive_paths.resolve_sensitive_context` multiplies this cost by about 20: 607 ms measured per resolution. `fs_read` resolves it twice, and each Console @-reference resolves it once more during send.

Other findings:
- The composer keystroke path runs an O(n) per-character Python loop over the draft (3.1 ms per call at 100K chars).
- Each INFO+ log record is redacted three times, because stdlib RotatingFileHandler.shouldRollover formats the record once more.
- scrypt runs per encrypted config value on every config reload or write (22 ms per value).
- tiktoken cold-loads take 60/89 ms on the first caller, which can be the event loop.
- A process-wide fd-protection lock is held across whole Chatterbox generations.
- ui_responsiveness wakes 20 times per second while idle.
- Each httpx client built through the tls_trust factories re-creates an SSL context (9.2 ms vs 0.08 ms with a shared one).

Cross-slice notes for other agents:
- The root of F1 is in config.py:9434 and Backup_Recovery (raw_participants._scope, storage_admission). TASK-32562's Windows profiling already recorded '15 get_user_data_dir calls 15.94s' but nobody has optimised it.
- Logging_Config forwards every loguru record at TRACE (about 5.8 us per dropped debug call).
- The durable-write primitives cost 4.5–8 ms median and up to 45 ms (atomic_private_write_text / atomic_write_text; two F_FULLFSYNC barriers); callers outside this slice must stay off the event loop.

Overall the rest of the slice is healthy: text_wrap_index, textual_css_fastpath, fts5 helpers and mosaic prerendering are already optimised or off the loop.

## clean areas
- tldw_chatbook/Utils/textual_css_fastpath.py - already-optimised ordered-candidate/ancestor-rejection path; the per-apply _ancestor_names set build is about 2% of a 0.38 ms apply, not worth changing
- tldw_chatbook/Utils/text_wrap_index.py - ASCII short-line fast path plus a bounded segment cache and prefix sums
- tldw_chatbook/Utils/fts5_match_forms.py, timestamps.py, datetime_codec.py, text.py - pure per-query/per-value string work
- tldw_chatbook/Utils/Emoji_Handling.py - result cached in a module global
- tldw_chatbook/Utils/boot_worker_policy.py, app_shutdown.py - pure bookkeeping and Event.wait, no polling
- tldw_chatbook/Utils/db_status_manager.py - 120 s interval, stats run via asyncio.to_thread, INFO line only on change
- tldw_chatbook/Utils/adaptive_reader_state.py, library_rail_width.py, console_background_effects.py, reasoning_config.py, sensitive_config_keys.py - pure normalisers
- tldw_chatbook/Utils/mosaic_render.py - avatar callers prerender off the loop (character_avatar_layout.prerender_character_avatar)
- tldw_chatbook/Utils/text_selection_crash_guard.py - one try/except around App.on_event, negligible
- tldw_chatbook/Utils/local_stt_providers.py - find_spec probes at 0.35 ms per probe, run once per mount and per dictation press
- tldw_chatbook/Utils/markdown_parsing.py - linear callout rewrite; check_dependency per widget construction costs about 6 us
- tldw_chatbook/Utils/persistent_diagnostics.py - PersistentDiagnosticFilter (Path.resolve per record) is no longer installed on live handlers; persist_event is schema formatting only
- tldw_chatbook/Utils/github_api_client.py - one pooled AsyncClient per loop, with timeouts; cold path
- tldw_chatbook/Utils/egress.py policy checks - config reads use the get_cli_setting fast path (about 1 us); sync-DNS variants are reached only from worker threads
- tldw_chatbook/Utils/file_handlers.py - attachment processing is dispatched via asyncio.to_thread (plus asyncio.run, one loop per attachment, minor)
- tldw_chatbook/Utils/file_extraction.py - its quadratic filename-hint scan (text[:start].split per code block) is only reachable from the retired ChatMessage/ChatMessageEnhanced widgets, which have no live constructions
- tldw_chatbook/Utils/Splash.py, Splash_Strings.py, about_text.py, NotificationHelper.py, log_widget_manager.py, splash_animations.py, platform_files.py, filesystem_identity.py, instance_lock.py, startup_errors.py, startup_logging.py, secure_temp_files.py, note_importers.py, doctor.py, install_clipboard.py, db_upgrade_notice.py, ui_responsiveness_artifacts.py, widget_helpers.py, sensitive_llm_logging.py - cold or trivial
- tldw_chatbook/Utils/windows_files.py - Windows-only, skimmed but not audited deeply
- tldw_chatbook/Utils/tiktoken_runtime.py import - light (inspect is already paid by Textual/pydantic); cost moves to first use (see F6)

## census
| Measurement (isolated env, Py3.12, macOS) | Result |
|---|---|
| get_user_data_dir(), 15-component scratch path | 27.5 ms min / 35.1 ms median (23.8 ms avg in an earlier run) |
| get_user_data_dir(), 21-component path | 36.6 ms min / 48.4 ms median, so about 2.2 ms per component and about 14 ms estimated at the default 6-component path |
| get_chachanotes_db_path() | 26.1 ms per call |
| Profile per get_user_data_dir call | about 57 _open_verified_parent walks, 5 acquire_storage calls, about 1,591 posix.open calls |
| _open_verified_parent, 15 components | 108.6 us (about 7 us per component) |
| resolve_sensitive_context() | 607 ms; is_sensitive_path(no ctx) 442 ms; with ctx 0.15 ms |
| validate_console_draft (keystroke path) | 0.006 ms at 200 chars, 0.054 ms at 2K, 0.577 ms at 20K, 3.12 ms at 100K; a str.translate replacement takes 0.072 ms at 100K |
| redact_log_line | 35–44 us per 120–156-char line; 593 us per 2,000-char key=value line |
| ConfigEncryption.decrypt_value (scrypt N=16384) | 22.2 ms; decrypt_config with 8 keys 179 ms |
| First estimate_tokens (tiktoken cl100k / o200k) | 60.3 ms / 88.6 ms; warm 0.026 ms |
| httpx.AsyncClient() with verify=True vs a shared SSLContext | 9.15 ms vs 0.083 ms |
| atomic_write_text 4KB / atomic_private_write_text 4KB / plain fsync+replace | 8.06 (max 23.5) / 4.53 (max 45.1) / 0.17 ms median |
| ui_responsiveness watchdog + drain idle | 0.033% of one core, 20 wakeups/s |
| Boot import: tokenizers / metadata.version('tiktoken') / input_validation self | 2.6–3.1 ms / 1.3–1.75 ms / 5.2 ms |


# slice-59

## summary
Slice #59 (tldw_chatbook/Widgets top level: 58 files, ~23.6k lines) is mostly shared chrome and modal widgets. The hot-path pieces are AppFooterStatus (every screen), destination_rail, glyph_fallback, status_line, modal_dismissal (80 importers), recompose_capture_guard, pausable_progress, prune_safe_select, workbench_focus, compact_model_bar (Console control bar), model_search_picker (Console model popover and settings modal), enhanced_file_picker (36 construct sites) and splash_screen (boot). Most of it is healthy. The good patterns are PausableProgressBar, the picker search debounce, file-picker persistence batched onto a thread worker, the theme scan on a thread worker, diff prepare() off-thread, DetailValueRow in-place edits, and the rail-handle recompose gated on change.

Main problems found:
1. **(P1, root cause mostly outside the slice)** TASK-32804.1's warm fastpath fixed get_cli_setting only. `load_settings()` is still @guarded and costs 7.3–8.6 ms warm (about 600 posix.open calls per call). `get_user_data_dir()` costs 22 ms warm. There are 109 and 203 call sites respectively; this slice reaches them from compact_model_bar and project_skills_import_modal.
2. **(P1)** A hidden legacy CompactModelBar is minted on every Console visit only so old selectors keep resolving. It pays about 14 ms of load_settings, two Selects with overlays, and a mount-window Select.Changed burst.
3. **(P2)** The model_search_picker filter is O(n²), with per-item validation recomputed 1–2x per keystroke and no debounce: 2.3 ms at 350 models, 7.7 ms at 1000, 69.5 ms at 4096, per call.
4. **(P2)** EmojiGrid remounts up to 180 Buttons per populate, removing the old ones one at a time: 148–198 ms per open or filter settle.
5. **(P2)** ConversationSelectionDialog mounts 724 widgets for 100 conversations: 348 ms to open and 72 ms per filter keystroke.
6. **(P2)** The file picker's on_mount forces BookmarksManager's first use, so task-261's "lazy" fix still does 5 stats plus a synchronous config rewrite on the loop (58 ms) the first time each context opens. Every Ctrl+D is another 50 ms synchronous write.
7. **(P2, extends TASK-33078)** Edit/Clone of a saved theme runs the full themes-folder scan on the UI thread. The editor constructor also pays a 6.45 ms writing-scope handshake.
8. **(P2)** The boot splash imports all 87 effect modules to use one: 10–12 ms warm, 292 ms cold.
9. **(P3)** Smaller items: the Settings splash preview animates forever at 10–100 Hz; the llama.cpp snapshot manager runs a 1 Hz idle layout-refresh tick; the Agents panel does its sqlite work and a ~50 ms config write on the event loop.

Structural note: ten top-level widget modules (about 3.9k lines) have no production importer and cost nothing at runtime: activity_log, document_generation_modal, feedback_dialog, file_extraction_dialog, file_picker_dialog, status_dashboard, template_selector, tool_message_widgets, voice_input_widget and status_widget (plus Event_Handlers/ingest_status_helper.py). Also dead: the EmojiPickerScreen class, AppFooterStatus.update_word_count/update_token_count/update_db_sizes_display, and CompactModelBar.sync_from_sidebar (its only caller has no callers). CLAUDE.md still lists tool_message_widgets as a "Key Widget". Deleting them is hygiene only, not a speed win.

All measurements ran in an isolated scratch profile (HOME, XDG_* and TLDW_CONFIG_PATH under the scratchpad, TLDW_TEST_MODE=1, PYTHONPATH pointing at the audit tree). The audit tree is unmodified (git status clean).

## clean areas
- tldw_chatbook/Widgets/AppFooterStatus.py: per-screen footer. The responsive ladder is O(k^2) over about 10 hint actions, only on resize or context change. _build_tamagotchi does one get_cli_setting per compose (measured 1 us); the pet is opt-in.
- tldw_chatbook/Widgets/modal_dismissal.py: per-click query_one plus query(Footer) over a small modal DOM; focus restore runs only on dismiss.
- tldw_chatbook/Widgets/recompose_capture_guard.py: cheap capture check around recompose only.
- tldw_chatbook/Widgets/glyph_fallback.py: identity fast path when ASCII mode is off.
- tldw_chatbook/Widgets/status_line.py: single query_one plus update.
- tldw_chatbook/Widgets/destination_rail.py: DestinationRailHandle.sync_state recomposes a 1-2 child subtree only when label or badge changed; sync_open updates in place.
- tldw_chatbook/Widgets/destination_workbench.py, workbench_focus.py, select_values.py, detach_safe_text_area.py, reader_scroll.py, prune_safe_select.py (2 hasattr per construct): clean.
- tldw_chatbook/Widgets/pausable_progress.py: the GOOD pattern. Hidden progress clocks are paused, not ticking.
- tldw_chatbook/Widgets/detail_value_row.py: in-place value and editor swaps, no recompose.
- tldw_chatbook/Widgets/diff_widgets.py: DiffView.prepare() runs off-thread (console_transcript caller). textual_diff_view adds about 1 ms incremental import on top of already-loaded textual.
- tldw_chatbook/Widgets/theme_preview.py: paint() is 7 query_one plus kwargs set_styles (setattr path, no CSS parse) per colour keystroke. Cheap.
- tldw_chatbook/Widgets/settings_theme_picker.py: folder scan in a thread=True exclusive worker; colour systems cached (theme_catalog._COLOURS_CACHE); theme switches reuse the last listing.
- tldw_chatbook/Widgets/settings_splash_screen_viewer.py: config persistence via @work(thread=True).
- tldw_chatbook/Widgets/settings_internal_prompts_panel.py / settings_internal_prompts_editor_modal.py: persistence via asyncio.to_thread; override_state reads the cached get_cli_setting.
- tldw_chatbook/Widgets/settings_advanced_config_panel.py / settings_web_search_panel.py: all I/O through workers. Per-keystroke work is a few 100 KB string joins and compares, under 1 ms.
- tldw_chatbook/Widgets/settings_image_gen_panel.py / settings_video_gen_panel.py: keyring-backed config loaded off-thread by the screen. Compose parses config.toml with tomllib (measured 3.1 ms for 102 KB). Minor.
- tldw_chatbook/Widgets/enhanced_file_picker.py search and persistence: search is debounced 0.2 s, listing is progressive (FileRecord-backed rows, no per-row stat), and recent/last-dir persistence is coalesced into one thread worker (task-15470). Only the bookmarks path is flagged.
- tldw_chatbook/Widgets/emoji_picker.py data load: lazy and module-cached (_load_emojis measured 10.6 ms once; emoji package import about 20-37 ms on first appearance-picker open, lazily imported from workspace.py).
- tldw_chatbook/Widgets/llamacpp_snapshot_manager.py refresh path: asyncio.to_thread for preference I/O, refresh on entry only. Only the 1 Hz elapsed tick is flagged.
- tldw_chatbook/Widgets/audio_troubleshooting_dialog.py: device enumeration in a thread worker; the 10 Hz meter runs only while testing and is gated on screen.is_active.
- tldw_chatbook/Widgets/password_dialog.py, confirmation_dialog.py, delete_confirmation_dialog.py, cancel_confirmation_dialog.py, backup_group_selector.py, voice_blend_dialog.py, voice_profile_dialog.py, form_components.py, pattern_gallery.py (dev-only gallery): no hot costs.
- tldw_chatbook/Widgets/workspace_create_modal.py / workspace_persona_default.py: small-modal recomposes on explicit add/remove/page actions. Sync .SKILLS discovery on add-folder is a documented, accepted trade-off.
- tldw_chatbook/Widgets/project_skills_import_modal.py: loose-file reads via asyncio.to_thread. Only the get_user_data_dir calls are flagged (under F1).
- Boot imports from this slice (confirmation_dialog about 1 ms, glyph_fallback, AppFooterStatus, splash_screen about 1.3 ms self time, card_definitions 0.25 ms): cheap. Their heavy cumulative cost is config/UI packages that app.py loads anyway.
- Dead top-level widget modules, zero runtime cost (no production importer, verified by module- and class-name grep): activity_log, document_generation_modal, feedback_dialog, file_extraction_dialog, file_picker_dialog, status_dashboard, template_selector, tool_message_widgets, voice_input_widget, status_widget (+ Event_Handlers/ingest_status_helper.py). Also the EmojiPickerScreen class and the AppFooterStatus word/token/db-size updaters.

## census
| Measurement (isolated scratch profile, Textual 8.2.8, py3.12) | Result |
|---|---|
| warm `get_cli_setting` | 0.001 ms |
| warm `load_settings()` | 7.3–8.6 ms (~597 posix.open/call via config_participants.guarded → operation → acquire_storage) |
| warm `get_cli_providers_and_models()` | 7.1–7.2 ms |
| warm `get_user_data_dir()` | 22.4 ms |
| `SettingsThemeEditor()` constructor | 6.45 ms |
| `save_setting_to_cli_config` (102 KB config) | 50–52 ms |
| first-use `BookmarksManager.get_bookmarks()` (defaults + write) | 58.4 ms; second use 0.01 ms |
| `EmojiGrid.populate_grid` 180 emojis (replace 180) | 148–198 ms; batched remove/mount alternative 113–118 ms |
| `EmojiGrid.populate_grid` 12 emojis after 180 | 49 ms (dominated by per-child remove) |
| `ConversationSelectionDialog` 100 conversations | open 348 ms (724 widgets); filter keystroke 72 ms |
| model picker `_catalog_model_ids` equivalent | n=350: 2.3 ms; n=1000: 7.7 ms; n=4096: 69.5 ms (normalize 36.9 ms + O(n²) dedupe) |
| splash `load_all_effects()` | warm 10.4–12.0 ms (87 effect modules, 93 new sys.modules); cold (fresh pycache) 292.5 ms |
| tomllib parse of 102 KB config | 3.1 ms |
| incremental `import textual_diff_view` | 0.8 ms |


# slice-6

## summary
Slice #6 (Character_Chat, 32 files, 24,074 lines) audited at 840ed2ca58. The slice runs on these hot paths:
- **Per send, on the event loop.** The UI submit path (`wiring.py:256 _admit_console_turn_to_runtime` -> `session.py:3499 _build_console_turn_execution_context`, plus controller `queue_prompt`/`edit_queued_prompt`/`_submit_draft_body`) calls `capture_prompt_transform_inputs`. That reads dictionaries and world books. In persona sessions it also calls `resolve_turn_persona_policy_rules` -> `get_persona_profile`.
- **Per send, on worker threads.** The dictionary and world-info appliers, and world-info/dictionary matching.
- **Per avatar state change or emote event, on worker threads.** `resolve_visual_identity`.
- **1 Hz Buddy speech poll** (persona binding, loop).
- **Personas persona tab.** Scope-service calls run inline on the loop.
- **Boot.** `TldwCli.__init__` builds the persona and dictionary services and runs the builtin seeding. Slice modules also cost about 22-24 ms of self import time.

Overall the slice's own algorithms are mostly sound. The emote stream parser, the mood classifier and the Samira preflight were already optimised (TASK-22227/21111). Personas lore UI, Console world-book summary/picker, ChatDictionaryScopeService, expression-set IO and buddy conversion all offload correctly.

The dominant cost does not come from the slice's own logic. Slice code reaches the Backup_Recovery storage-admission handshake constantly: `ChaChaNotes transaction()` -> `_core_operation` -> `acquire_storage`, `@_chat_sources.guarded` on 98+39 service methods, `@visual_lifetime.db_guard`/`reader_guard`. Each admission is about 7 ms, roughly 245 `open()` syscalls in the isolated profile. A read-only `transaction()` costs 12.5 ms against 0.012 ms for `get_connection().execute`. This was introduced by b5251e9a6e (TASK-32628, 2026-09-16), after TASK-31502 measured 23 us per transaction. It is therefore a regression of about 500x. F0 records the cross-cutting root, which is outside the slice and probably also reported by the DB/structural agents. F1-F3, F6, F7 and F9 are the slice-local levers:
- use `execute_query` for reads;
- stop guarding in-memory accessors;
- construct services lazily;
- skip redundant resolves.

Two findings scale with user data and are independent of admission: the unbounded dictionary version-history sidecar (F4) and the O(keys x text) lore/dictionary matching (F5).

**Caveat:** all timings were taken in the scratch profile, whose data path is 14 components deep. The admission walk is per path component, so a real `~/.local/share` profile (about 6 components) likely pays roughly half the admission share.

**Structural:** `Character_Chat_Lib.py` carries about 1.3k lines (17 functions) with no references in `tldw_chatbook` outside the module (F11).

## clean areas
- tldw_chatbook/Character_Chat/emote_directives.py: CharacterEmoteStreamParser is run-based and bounded per chunk; already optimised by TASK-22227. Only its import of visual_identity is flagged (F8).
- tldw_chatbook/Character_Chat/character_mood.py: once per completed character turn, about 2.3 ms at 16k chars, documented and bounded.
- tldw_chatbook/Character_Chat/chat_dictionary_scope_service.py: _call_backend threads file-backed local calls via _DictionaryJob/asyncio.to_thread (the TASK-15469 fix); the good pattern.
- tldw_chatbook/Character_Chat/world_info_resolver.py summarize_active_world_books: called via asyncio.to_thread and gated on scope change (retrieval.py:530-561). Only the entry hydration is noted (F12).
- Personas Lore UI (personas_screen.py _lore_manager call sites 4373..8252): every WorldBookManager call is wrapped in asyncio.to_thread.
- Console world-book and dictionary attach/detach pickers (chat_screen.py:14633/14700, chat_events_console_dictionaries.py): run via asyncio.to_thread.
- tldw_chatbook/Character_Chat/expression_set_io.py: PIL at module scope, but only imported lazily or under TYPE_CHECKING; resolve/build are called via asyncio.to_thread (personas_screen.py:13532/13629).
- tldw_chatbook/Character_Chat/buddy_conversion.py: convert/publish/suggest are all called via asyncio.to_thread or drain_thread.
- tldw_chatbook/Character_Chat/persona_list_paging.py: in-memory filter/sort over 100 profiles or fewer.
- tldw_chatbook/Character_Chat/expression_generation.py: manifest is lru_cache(maxsize=1).
- tldw_chatbook/Character_Chat/character_generation.py + character_generation_controller.py: async gateway calls, no sync network on the loop.
- tldw_chatbook/Character_Chat/server_character_persona_service.py + server_chat_dictionary_service.py: thin async wrappers; the client comes from a provider that caches it (runtime_policy/bootstrap.py:251-254).
- tldw_chatbook/Character_Chat/character_card_formats.py, world_book_import.py, artwork_attribution.py, character_avatar.py, world_info_diagnostics.py, character_events.py: cold import/export or data-only paths, not on the boot import graph.
- tldw_chatbook/Character_Chat/character_conversation_navigation.py: thin facade and dataclasses; its queries live in DB/character_conversation_search.py (outside the slice). Only its import cost is noted (F8).
- tldw_chatbook/Character_Chat/Chat_Dictionary_Lib.py list_chat_dictionaries: uses json_array_length without parsing entries; the raw-cursor reads avoid the transaction() admission tax.
- tldw_chatbook/Character_Chat/visual_identity.py _find_builtin_samira_card: already a JSON1 query (TASK-21111); PIL is lazy (TASK-22217).
- tldw_chatbook/Character_Chat/__init__.py and ccv3_parser.py: empty; no eager package imports.
- Console avatar image work (UI/Console_Modules/character.py): all resolves and decodes are off-loop via asyncio.to_thread. The cost is repeated work on the worker (F6), not loop blocking.

## census
| Measurement (isolated scratch profile, path depth 14; Python 3.12, no loguru sinks) | median |
|---|---|
| ChaChaNotes `db.transaction()` + `SELECT 1` | 12.5 ms (p95 21.2) |
| `get_connection().execute("SELECT 1")` / `execute_query` | 0.012 / 0.015 ms |
| `capture_prompt_transform_inputs`, no lore attached (on loop per send) | 7.0 ms (p95 16.7) |
| `_collect_active_world_books` 0 / 1x20 / 3x200 / 5x500 books x entries | 6.6 / 8.3 / 10.9 / 28.8 ms |
| `collect_active_chatdict_entries` 3x50 / 3x200 / 5x500 | 0.5 / 1.9 / 38.3 ms |
| `LocalCharacterPersonaService.get_persona_profile` (in-memory lookup) | 22.3 ms (3x acquire_storage) |
| `list_persona_profiles` / `update_persona_profile` | 22.8 / 28.5 ms |
| `LocalChatDictionaryService.list_dictionaries` | 19.7 ms |
| `build_persona_service` + `build_dictionary_service` (warm; App.__init__) | 58-78 + 49-55 ms |
| `seed_builtin_content`, already seeded (boot) | 12.6 ms |
| `resolve_visual_identity`, Samira idle/thinking/speaking | 13.9 / 15.3 / 15.1 ms |
| `_inspect_image_bytes` 1024^2 WebP / 1254^2 PNG portrait | 8.0 / 19.6 ms |
| WorldInfoProcessor build + process_messages, ~5k-char scan, 100/400/1200/5000 keys | 2.9 / 11.4 / 45.5 / 200 ms |
| same, 1200 keys: fresh compiles / precompiled searches / token-set lookup | 9.9 / 33.2 / 0.2 ms |
| `process_user_input` 150 / 600 / 2500 dictionary entries | 3.3 / 14.0 / 77.3 ms |
| Dictionary history sidecar, 200-entry dict at 10 / 100 / 300 edits: size, dump per edit, boot load | 1.3 / 13.1 / 39.3 MB; 23 / 211 / 746 ms; 3 / 32 / 139 ms |
| Slice module self import time on boot path (boot-import run 3 / chat path) | 24.4 / 22.0 ms |


# slice-60

## summary
Slice #60 (Widgets/Console#1, 55 files, 31,619 lines) is dominated by ConsoleComposerBar (6,587 lines), which is the per-keystroke hot path: ChatScreen.on_key calls handle_console_key, then insert_text, _sync_hidden_input, _refresh_visible_draft and sync_action_state, and posts DraftChanged, which makes the screen resync (a second sync_action_state). The other hot paths are:
- the transcript's per-turn activity reconcile into console_assistant_turn;
- always-mounted rail widgets with their own timers (character context, expression avatar);
- first-paint construction of every Console widget;
- a set of modals (Prompts, Conversation Inspector, image viewer).

I ran a safe headless probe: a minimal App with the real CSS bundle and 500 filler widgets, not TldwCli, with isolated HOME/XDG. It measured every printable key at 8 synchronous stylesheet.update_nodes (3.25 ms), 6 refresh(layout=True) and 1 full Screen._refresh_layout. Three independent causes remove all of it (handler 3.74 to 0.27 ms, layouts 1.0 to 0.0 per key):
- **Regression.** The ADR-161 token-class migration (ae9093a714, 2026-09-14) turned no-op inline style writes into remove_class/add_class pairs. Textual applies these synchronously, including on widgets that are not yet mounted. That cost is paid per keystroke in _sync_send_disabled_reason, and at every mount/recompose: composer mount went from 29 to 95 update_nodes, about 45 ms extra with the real CSS. The pattern appears 227 times across 25 files, and 107 of those are in this slice.
- **Hidden mirror input.** The hidden compatibility Input's virtual_size reactive re-arms a whole-screen layout on every key.
- **Visible draft.** The visible draft's Static.update defaults to layout=True even though its geometry is inline-pinned. TASK-24453 closed on the claim that this layout 'cannot be removed'; the probe shows it can.

Secondary, data-scaled costs:
- whole-draft grapheme wrap run 2x per key and 3-4x per arrow key (quadratic for long unbroken tokens: 26.5 ms/key at 20 KB);
- unconditional ConsoleActivityDisclosure re-sync of every activity in a turn, per thinking delta;
- trace-capture sanitization on the loop in the Inspector (59 ms per 200 KB capture);
- Prompts-modal local SQLite search on the loop via _maybe_await;
- image-viewer mosaic on the loop (25 ms).

Idle/background: Console is a reusable route, so it is suspended and never unmounted. Widget-level timers (character-context 2 Hz poll, avatar 30 Hz, activity elapsed 10 Hz) are never quiesced by on_screen_suspend, but their per-tick cost is small.

Overall health: the composer, bounded sections, inspector sections, background effect and modals are generally well-guarded (signatures, memo keys, to_thread, debounce, caps). The biggest structural lever is a single no-op-when-unchanged token-class helper, applied repo-wide. It fixes both the keystroke regression and a first-paint and recompose style-apply storm that feeds TASK-24452.

Suggested PR grouping:
- **(A)** Token-class helper plus composer keystroke guards (F1, F3).
- **(B)** Composer per-key layout removal (F2), then wrap memo and windowing (F4).
- **(C)** Activity disclosure equality short-circuit, with a per-activity sync gate in console_transcript (F5).
- **(D)** Off-loop fixes for the Inspector projection, Prompts local search and image-viewer mosaic (F6, F7, F8).
- **(E)** Console widget timer quiescence on suspend (F9, F10).
- **(F)** Modal import deferral (F11), coordinated with the _ui_ready census tasks.

## clean areas
- console_background_effect.py: frame computed once per (serial,w,h) (task-261), timer tick early-outs when screen not active; opt-in effect
- console_composer_bar.py cursor blink: memoized per blink phase, Static.update(layout=False), early-out when screen inactive (TASK-21692/22218/22219 held)
- console_composer_bar.py _apply_draft_height/_sync_collapsed_presentation/_sync_raw_cli_state/_sync_improvement_recovery: signature-guarded (task-24453 held)
- console_composer_bar.py send-price: keystroke path asks only the cheap availability seam; tooltip derivation only on pointer-over-Send (TASK-23018 held)
- console_composer_bar.py undo history: coalesced per typed run, depth + char-budget bounded
- console_inspector_section.py: equality-guarded, in-place row patching; recompose only on structural change
- console_bounded_section.py: coalesced call_after_refresh reconcile, equality-guarded geometry writes (query_one volume already owned by TASK-24455)
- console_inspector_detail_pane.py: OptionList-backed (virtualized) section list
- console_conversation_inspector.py next-send: snapshot/detail text via asyncio.to_thread; payload token estimate measured 0.2 ms for 400 msgs; call detail text via to_thread
- console_model_popover.py: Select.set_options measured 0.9 ms (400 opts) / 1.9 ms (4096 opts) - not a keystroke problem
- console_control_bar.py / console_context_controls.py / console_provider_picker.py / console_retrieval_scope_row.py: state-equality gated or row-scoped recompose
- console_prompt_queue_modal.py: 5 Hz poll but revision-gated, list bounded by MAX_CONSOLE_QUEUE_ENTRIES
- console_agent_progress_modal.py: 2 Hz poll only while modal open (minor unconditional count/body repaint per tick)
- console_character_picker_modal.py / console_prompt_picker_modal.py / console_reaction_picker_modal.py / console_appearance_picker_modal.py: debounced filters; results capped (character CHARACTER_PICKER_MAX_RESULTS, EmojiGrid MAX_DISPLAY=180)
- console_citation_sources_modal.py, console_agent_history_modal.py, console_exchange_export_dialog.py, console_endpoint_template_modal.py, console_project_instructions.py: DB/file work already via asyncio.to_thread
- console_assistant_turn.py ConsoleToolPreview._rewrap: preview capped at 160 chars (DEFAULT_CONSOLE_TOOL_RESULT_DISPLAY_CHARS); measured ~0.3 ms even at 60 lines
- console_auto_speak_consent.py: per-message-completed subscription only; destination resolution async; TTS import deferred
- console_command_popup.py: idempotent re-show, OptionList rows bounded to suggestions (query_one is Textual-cached)
- console_capture_policy_dialog.py, console_fork_chat_modal.py, console_edit_message_modal.py, console_rename_session_modal.py, console_feedback_comment_modal.py, console_generate_image_modal.py, console_composer_menu_modal.py, console_message_more_menu.py, console_conversation_action_menu.py, console_rag_settings_modal.py, console_library_search_modal.py, console_library_access_modal.py, console_prompt_draft_editor.py, console_prompt_draft_save_dialog.py, console_prompt_improve_view.py, console_prompt_comparison_modal.py, console_prompts_state.py, console_prompts_browse.py: cold/bounded UI, nothing material
- console_inspector_ownership.py, console_inspector_presentation.py, console_canvas_card.py, console_generation_card.py, console_activity_outcome_notice.py, console_agent_steering_bar.py, console_rail_handle.py: presentation-only (construction-time class dances counted under F3)

## census
| Measurement (safe headless probe: minimal App with the real 422 KB CSS bundle, 500 filler Statics, isolated HOME) | Baseline | Reason-strip gated | + hidden mirror off + visible Static layout=False |
|---|---|---|---|
| composer handler per key (insert_text + one screen resync) | 3.74 ms | 0.32 ms | 0.27 ms |
| stylesheet.update_nodes per key | 8.0 (3.25 ms) | 0 | 0 |
| Screen._refresh_layout per key | 1.00 (6.3 ms @500 fillers; ~11.5 ms on real Console per 08-29 review) | 1.00 | 0.00 |
| composer mount: update_nodes | 95 (65.6-68.0 ms style work, wall ~148 ms) | pre-mount class updates suppressed: 29 (18.5-21.2 ms, wall ~99-113 ms) | |

| Draft size | row_count + renderable wrap per key |
|---|---|
| prose 2K / 8K / 20K chars | 0.26 / 0.88 / 2.14 ms |
| code 20K chars | 2.71 ms |
| no-space blob 2K / 8K / 20K | 0.79 / 5.84 / 26.54 ms |

| Other measured | |
|---|---|
| 30 unchanged ConsoleActivityDisclosure.sync_activity | 5.71 ms handler + 1 forced layout (22.6 ms in probe) |
| sanitize_capture_value 25 KB / 201 KB request | 7.9 / 58.9 ms |
| mosaic_from_image full-screen 227x46 | 24.7 ms |
| avatar mosaic per frame 24x12 / 40x20 | 0.49 / 1.33 ms (thread) |

| Pattern census | Slice | Repo |
|---|---|---|
| ADR-161 `remove_class(*(name for name in ...))` + add_class dance | 107 sites / 12 files (composer 39) | 227 / 25 files |
| Widget-level set_interval in Console tree not quiesced by ChatScreen.on_screen_suspend | character_context 0.5 s, expression avatar 1/30 s, activity header 0.1 s | - |


# slice-61

## summary
Slice #61 (Widgets/Console#2, 35 files, 31,719 lines). The hot paths here are: (a) the Console's 0.2 s sync poll. While a run is active, chat_screen.py:18947 calls ConsoleTranscript.set_messages every tick, then refresh_messages → _reconcile_rows whenever the refresh key changes, which is every tick while streaming. (b) j/k and click selection in the transcript: select_message → refresh_messages. (c) Session-tab strip sync (sync_sessions, run every tick; it rebuilds when the tab set changes). (d) Opening and resizing the Settings modal.

Overall the transcript's data layer is healthy. The per-message signature cache works, get_cli_setting is read once per pass, fence appends are throttled, the prune check is coalesced, drag selection is memoized, and there is no loop I/O. The dominant cost is in the style engine, not the Python.

**Top finding (P0).** `_sync_message_classes` removes all 11 managed row classes and re-adds the wanted ones on every sync_message. Textual's add_class/remove_class restyle the node's whole subtree synchronously. Under the default `role_accents` transcript style, the assistant row carries `console-transcript-message-role-assistant`, a managed class, so every streaming tick restyles the reply's entire Markdown subtree twice. So does any tool-status change in the turn.
- Measured with the production 422 KB bundle and the app's own CSS fastpath installed: 9 ms per tick for a 10-block reply, 30 ms for 40 blocks, 125 ms for 120 blocks, against 0.8–1.6 ms with a diff-based sync.

**The same disease, structurally.** A codemod left a "remove every `h-*`/`w-*` class, then add one" idiom at 227 sites in 25 files (74 in this slice).
- Each site restyles twice for nothing when the class is already present, and doubles style work when a widget is built (0.25 → 0.55 ms per Button).
- It is ~160 ms of a ~550 ms Settings-modal open (250 extra restyles per open).
- It also inflates tab-strip rebuilds. Those rebuilds also await one remove/mount per child: 51 ms against 30 ms when batched.

**Smaller findings:**
- Selection toggles restyle the Markdown subtree legitimately (32–110 ms per keypress on long replies).
- Whole-transcript O(n) walks run every tick regardless of the mounted window (1.8 ms at 2,000 messages).
- The Markdown-row signature computes the unused plain renderer.
- Throwaway widgets are built only to compare signatures.
- PIL, rich_pixels and textual_diff_view are imported at module scope on the Console first-paint leg (one of at least 4 PIL chains).
- The terminal viewport keeps a snapshot for every terminal session ever opened, with no pruning.

Known and not re-filed:
- The 5 Hz switcher projection poll (TASK-31506). No new evidence.

**Suggested PR grouping:**
1. F1 + F2: transcript class sync and selection restyle.
2. F5 codemod, which also carries F4 and most of F3.
3. F6 + F7 + F8: transcript per-tick hygiene.
4. F9 with the other PIL chains.

Bench scripts are in scratchpad/audit/scratch/s61/: bench_sel_fp.py, bench_tabs2_fp.py, bench_settings3_fp.py, bench_setmsgs.py, bench_classes.py.

## clean areas
- tldw_chatbook/Widgets/Console/console_status_chips.py — every sync_* (state/run/scope/cost/temporary) is equality-guarded; poll ticks are free
- tldw_chatbook/Widgets/Console/console_settings_summary.py, console_staged_context.py, console_staged_evidence_strip.py — equality-guarded; they recompose only their own small subtree on a real change
- tldw_chatbook/Widgets/Console/console_run_inspector.py — TASK-259 updates rows in place; it recomposes only on a structural change
- tldw_chatbook/Widgets/Console/console_send_authority_summary.py — sync_state equality-guarded; set_compact guarded
- tldw_chatbook/Widgets/Console/console_task_panel.py — event-driven (todo hook), not polled
- tldw_chatbook/Widgets/Console/console_session_switcher_modal.py — query is debounced and loaders are awaited; the 5 Hz active-projection poll is already TASK-31506 (no new evidence); result list capped at 50
- tldw_chatbook/Widgets/Console/console_scope_picker_modal.py — debounced filter and tag search, listers use asyncio.to_thread, page-limited rows
- tldw_chatbook/Widgets/Console/console_style_picker_modal.py — debounced; get_all_templates is process-cached
- tldw_chatbook/Widgets/Console/console_turn_file_card.py — DB/diff/note reads and writes all go through asyncio.to_thread; the resize relabel is local
- tldw_chatbook/Widgets/Console/console_video_preview.py — decode runs in a thread worker with call_from_thread backpressure; the 0.5 s offscreen timer exists only while playing
- tldw_chatbook/Widgets/Console/console_terminal_workspace.py rendering — _render_lines measured 0.36 ms per 50-line frame (only the _states retention is flagged)
- tldw_chatbook/Widgets/Console/console_selection.py, console_workbench_state.py, console_terminal_messages.py — pure logic, no Textual cost
- tldw_chatbook/Widgets/Console/console_selection_menu.py — the clamp/measure chain is bounded by shrink-once guards
- tldw_chatbook/Widgets/Console/console_speech_controls.py, console_voice_preview.py, console_video_card.py — cheap guarded updates
- Modals console_save_as_modal.py, console_save_markdown_modal.py, console_summarize_preview_modal.py, console_video_capacity_modal.py, console_terminal_session_modal.py, console_rewind_modal.py, console_review_notes_modal.py (writes awaited off-loop), console_system_prompt_modal.py, console_side_chat_modal.py (async stream, request-id fence), console_run_log_modal.py (thread worker), console_setup_modal.py, console_workspace_action_menu.py — cold paths, no loop-blocking I/O found
- tldw_chatbook/Widgets/Console/console_settings_modal.py — the 0.25 s _poll_subscription_readiness interval costs 0.10 ms per tick; resolve_entry_credential does no keyring access
- tldw_chatbook/Widgets/Console/console_transcript.py — verified fine: the TASK-259 per-message signature cache hits; get_cli_setting is hoisted per pass (task-32804.6); fence-append throttle (TASK-15456); the prune check is coalesced through call_after_refresh; drag offsets are memoized (TASK-21114, lru_cache(4)); message_more_menus_on_screen is a registry walk, not a DOM query; sync_jump_indicator is guarded; debug/info logging only on cold prune/hydration paths; the tool DiffView prepares in asyncio.to_thread; raw-CLI 10 Hz elapsed timer only while a command runs (update(layout=False), matches its 0.1 s display)

## census
| Pattern: `X.remove_class(*(name for name in X.classes if name.startswith("h-"/"w-")))` then `add_class(...)` | sites |
|---|---|
| Repo-wide | 227 in 25 files |
| In this slice | 74: console_settings_modal 22, console_session_surface 17, console_speech_controls 10, console_settings_summary 6, console_session_switcher_modal 6, console_status_chips 4, console_run_inspector 4, console_transcript 2, console_send_authority_summary 2, console_staged_context 1 |
| Largest outside the slice | console_composer_bar 39, chat_screen 27, console_workspace_context 18, console_inspector_section 16 |

Measured costs. Setup: Textual 8.2.8, headless 211x44, production `css/tldw_cli_modular.tcss`, the app's `install_stylesheet_fastpath()` installed, 30 tool rows in the turn.

| Scenario | as-is | with a diffed class sync / batched rebuild |
|---|---|---|
| Streaming tick `refresh_messages`, 10-block reply | 9.3 ms | 0.8 ms |
| Streaming tick, 40-block reply | 30.5 ms (max 120) | 1.0 ms |
| Streaming tick, 120-block reply | 124.7 ms | 1.6 ms |
| Tool-status-change tick, 40 / 120 blocks | 34.9 / 149.4 ms | 4.9 / 4.3 ms |
| Selection toggle (j/k) onto or off the reply, 40 / 120 blocks | 67.9 / 208.6 ms | 31.6 / 109.6 ms (remainder is the legitimate 'selected' restyle) |
| New tab: `sync_sessions` rebuild, 3→4-8 tabs / 12→13-17 tabs | 20.2 / 51.2 ms | 13.7 / 29.7 ms |
| Settings modal open (271 nodes) | ~547 ms, 350 class-triggered restyles | ~387 ms, 100 restyles, with the layout syncs stubbed |
| Button construction (restyle churn after construct vs `classes=` at construction) | 0.55 ms | 0.25 ms |
| `set_messages` per tick, 400 / 2,000 messages (runs every 0.2 s tick, before the refresh-key gate) | 0.26 / 1.83 ms | — |


# slice-62

## summary
Slice #62 (Widgets/Console#3, 9 files, 6,512 lines) was audited against pinned tree 840ed2ca58. Every file was read in full.

HOT PATH: ChatScreen._sync_console_workspace_context has 45 call sites. It also runs from the 5 Hz transcript poll while a run is active (chat_screen.py:18766 inside the set_interval(0.2) body). From there, ConsoleLeftRail.sync_workspace_context (left_rail.py:1920) pushes one ConsoleWorkspaceContextState into four widgets in this slice: two ConsoleWorkspaceContextTray projections (content="workspace" and "conversations"), ConsoleWorkspaceDetailsTray, and ConsoleWorkspaceTree.sync_projection. When no run is active, every call pushes even if nothing changed. During a run, only changed states are pushed.

The skip paths are now cheap. I measured _mounted_signatures at 0.12-0.31 ms, whole-state equality at 6-12 us, and the tree's projection memo returns before any per-row work. The remaining cost is on the change paths:
- The Conversations tray rebuilds its whole browser whenever any conversation_browser field changes. That includes tooltip-only fields such as the relative-age label, which is rebuilt from datetime.now on every build and so changes at minute boundaries during runs (F1, P1). Cost is 30-60 ms CPU headless.
- The Details tray recomposes whenever any part of the state changes, including fields it never renders (F2).

The Workspace Files modal does all of its filesystem I/O correctly, off-loop through coalesced asyncio.to_thread lanes. Its rendering is the problem: it re-mounts one Button per entry on every expand, collapse, Load more, filter result or exclusion toggle, and does this twice per expand. Measured 105 ms for 200 entries up to 985 ms for 2,000 entries (F3). It shows file previews of up to 200k characters in a single Static, measured at 240 ms to render and 155-195 ms per resize (F4). It also re-lays itself out on every Console sync because the attention line is re-pushed even when unchanged (F5, P3).

HOW I MEASURED: headless runs in an isolated profile, with the real widgets and a synthetic state. The app stylesheet was not loaded, so every number is a floor. The real app adds per-node selector-matching cost, which TASK-26834 cause 2 already documents.

STRUCTURAL NOTES:
- Widgets/Console/__init__.py is outside this slice. It eagerly imports about 40 Console modules, including console_workspace_switcher_modal. That makes the function-local lazy imports of the switcher in workspace.py:4929 and archive.py pointless. The switcher module itself is light, so the cost belongs to the package-init owner's slice.
- Importing any slice module pulls in tldw_chatbook.config, whose module scope runs load_settings. In my isolated run this wrote a config.toml into the scratch HOME. This is config.py's concern, not this slice's.
- console_workspace_context.py still carries several wrap/marker helpers the current row path no longer calls: wrap_console_conversation_title, _marker_prefixed_name_lines, _row_marker and _conversation_row_secondary. This is hygiene only, with no speed cost. I could not count references because the Bash classifier was unavailable at the end.

Scratch benchmarks: /private/tmp/claude-501/-Users-macbook-dev-Documents-GitHub-tldw-chatbook/15c5e43c-3dea-41a4-87ab-3b132387df58/scratchpad/audit/scratch/slice62/bench_tray.py and bench_files.py.

## clean areas
- tldw_chatbook/Widgets/Console/console_workspace_tree.py: the keyed incremental Tree sync keeps node identity. A value/identity memo skips unchanged 5 Hz pushes (TASK-22202). _update_tooltip is memoized, and Textual TreeNode add/remove/invalidate are O(1) or O(k) list operations. No recompose and no per-row widgets. Clean.
- tldw_chatbook/Widgets/Console/console_workspace_context.py skip path: _can_skip_recompose is well bounded. The DOM signature walk measured 0.12 ms at 47 widgets and 0.31 ms at 101 widgets; whole-state equality measured 6-12 us. The workspace-mode tray's TASK-26836 read set works. _fit_height_to_content is one deferred pass, with a coalesced, unsubscribing layout-signal retry. The row cap is 12, adaptive to about half the rail.
- tldw_chatbook/Widgets/Console/console_workspace_files_modal.py I/O layer: every list, filter, read and set_exclusion call runs through asyncio.to_thread via _OperationLane, with latest-wins coalescing and generation fences. Filter input is debounced (150 ms). Worker progress is coalesced into a single _FilterProgressReady message under a lock. The teardown joins in-flight work.
- tldw_chatbook/Widgets/Console/console_workspace_switcher_modal.py: compose is memory-only. workspace_persona_label_suffix reads assistant_defaults from the passed record, and get_persona_profile is a scan of an in-memory list (TASK-32804.6 moved the registry reads off the loop). The Show-archived toggle flips row.display without recomposing. Rename and receipt modals are trivial.
- tldw_chatbook/Widgets/Console/prompt_variables_dialog.py: capped at 64 variables. It rebuilds rows only on the System-lane checkbox toggle. compile_prompt_variables is a pure single-pass lexer, and the module it imports is stdlib-only, which keeps it cheap on the Console import leg.
- tldw_chatbook/Widgets/Console/trace_export_dialog.py: preflight, build and write all go through asyncio.to_thread, with a stale-profile guard. The file picker is imported function-locally, and the module itself is loaded lazily (trajectory_screen.py:1993). Only a trivial path stat runs on the loop.
- tldw_chatbook/Widgets/Console/trace_export_profile_ui.py: a constants-only leaf that deliberately keeps the export engine off the Chat first-paint leg. Clean.
- tldw_chatbook/Widgets/Console/conversation_row_presentation.py: pure, cheap helpers. Clean.

## census
| Measurement (headless Textual 8.2.8, 235x52, isolated profile, no app stylesheet = floor) | CPU median (max) | Notes |
|---|---|---|
| Conversations tray sync, unchanged state (skip path) | 1.6 ms (2.1) | includes state build + pilot overhead |
| Conversations tray, tooltip-only age-label delta, 12 rows / 47 widgets | 32.3 ms (85.0) | full refresh(recompose=True) |
| Conversations tray, tooltip-only age-label delta, 30 rows / 101 widgets | 62.4 ms (128.5) | full refresh(recompose=True) |
| Details tray, browser-only (unrendered) delta | 8.8 ms (12.4) vs 1.3 ms skip | ~7.5 ms net wasted per delta |
| _mounted_signatures walk | 0.12 ms (47 w) / 0.31 ms (101 w) | fine |
| ConsoleWorkspaceContextState == (12/30 rows) | 6 us / 12 us | fine |
| Files modal _render_tree, 200 entries | 109-126 ms | remove_children + mount_all |
| Files modal _render_tree, 800 entries (3 dirs expanded) | 318-324 ms | |
| Files modal _render_tree, 2000 entries (10 Load-more merges) | 878-987 ms | cap is 10,000 |
| Files modal _render_tree, 500 filter matches | 309-337 ms | FILTER_RESULT_LIMIT=500 |
| Files modal _render_viewer, 196k-char Static | 243-257 ms | files <=200k chars shown whole |
| Terminal resize with 196k preview mounted | 155-196 ms | re-wrap of whole Static |
| Unchanged attention push with 800-row tree mounted | ~16 ms vs ~5-12 ms baseline | Static.update always layout=True |


# slice-63

## summary
Slice #63 (Widgets/Library#1, 28 files, 29.5k lines; audit tree 840ed2ca58). These are the Library screen's widget layer: Folder Files workspace and Session Git panel (13.2k lines between them), Ingest canvas and queue, Conversations list and reader, Media list, viewer, raw view, image preview and trash, Collections reader, Landing, Export, and the Note-import and Add-from-files canvases.

Hot paths I traced and measured headless, with the real CSS bundle and isolated HOME/XDG/TLDW_CONFIG_PATH:
- **Folder Files runtime.** A 1.5 s `set_interval` poll, per-keystroke editor handlers and autosave. Measurements: reconcile costs 51-64 ms per tick for 1,000 files and ~300 ms for 5,000. Even for 30 files it costs 58 ms, and ~80% of that is Backup_Recovery storage admission (~1,000 dir-fd `open()` per reconcile). The loop does 9-12 ms of re-render plus 2 screen layouts per tick when nothing changed. The poll keeps running while the workspace is hidden behind Database Notes. Each editor keystroke costs 6.1 ms of synchronous work plus one screen layout; the control run shows 0 layouts.
- **Ingest.** Every job transition recomposes the whole uncapped queue panel: ~230 ms at 200 jobs and ~560 ms at 500 (1,076 widgets). The Library registry listener rebuilds the full state once per `submit`, making a folder submit O(N²) on the loop: 1.4 s at 500 files, 5.8 s at 1,000. TASK-32804.5 fixed only the app-level listener.
- **Conversations.** A normal row click recomposes the whole list canvas: ~140 ms at the 20-row page, ~235 ms at 50. The reader's progressive loader re-updates every mounted message on every page or continuation sync. Measured 7.1 s and 24.7k `Static.update` calls for 1,000 messages, then 166-292 ms for every later no-change sync, which includes each select-mode checkbox click.

Cross-cutting: this app's 3,612-rule stylesheet makes pre-mount `add_class`/`set_class` cost 0.34 ms each (0.001 ms with `update=False`), and `Button(compact=True)` costs 0.99 ms. That is a big part of why every recompose in this slice is expensive.

Already known and not re-filed: Media Trash re-measure (TASK-31509), Note-import review-row recompose (TASK-32804.11 AC#4), media reader Markdown per block (TASK-22660), whole-screen recomposes (TASK-281/22888), 8 connections per visit (TASK-24457).

Verified fixed since the 09-17 core review:
- `_fetch_chunk_templates` now runs off-loop through `_call_off_loop` (task-32804.12).
- `_configured_sync_folder` config reads now cost 3 µs per pair (TASK-32804.1 fastpath).
- The git-panel existence probe was fixed (PR #2739).

Structural: `library_file_notes_workspace.py` (8,856 lines, one class) is where most of the hot-path residue lives. It re-derives its whole control surface (`_update_controls`: ~40 `query_one`, 4 DOM walks, git-panel `set_mutating`) on every keystroke, poll tick and state change, instead of patching only what changed.

## clean areas
- tldw_chatbook/Widgets/Library/library_canvas_sync.py: post-recompose callback plumbing and the `library_row_button` helper, with no runtime cost of its own
- tldw_chatbook/Widgets/Library/library_media_raw_view.py: virtualized with WrapIndex, debounced reindex, timer stopped on unmount (good pattern)
- tldw_chatbook/Widgets/Library/library_media_content.py: lazy one-time mode mounts, memoized match-line scan
- tldw_chatbook/Widgets/Library/library_media_image_preview.py: decode runs in asyncio.to_thread (library_media_controller.py:2582); mosaic memoized (task-22208)
- tldw_chatbook/Widgets/Library/library_media_viewer.py: loading, query and match state patched in place (task-22207/22500); the remaining recomposes are deliberate toggles or renames. Markdown-per-block is known as TASK-22660
- tldw_chatbook/Widgets/Library/library_media_canvas.py: selection patched in place through apply_reader_state; full sync_state recompose only on infrequent actions (bulk delete, receipts); row-geometry messages are deduped
- tldw_chatbook/Widgets/Library/library_ingest_canvas.py LibraryIngestCanvas form: option edits patched in place (sync_option_group); the chunk-template fetch is now off-loop (rag_admin_scope_service._call_off_loop, task-32804.12)
- tldw_chatbook/Widgets/Library/library_artifacts_widgets.py and library_artifacts_reader_shell.py: search debounced 0.2 s in the controller, detail reads off-thread (P3 non-exclusive worker aside)
- tldw_chatbook/Widgets/Library/library_collections_capture_reader.py: bounded pages, scoped recompose of the saved-search rail only
- tldw_chatbook/Widgets/Library/library_adaptive_reader_shell.py and library_browse_reader_shell.py: resize-driven messages only
- tldw_chatbook/Widgets/Library/library_notes_add_from_files_canvas.py: setup and review edits have in-place paths; full recompose only on phase changes
- tldw_chatbook/Widgets/Library/library_note_import_canvas.py: destination and collision inputs patched in place (the review-row recompose is known as TASK-32804.11 #4)
- tldw_chatbook/Widgets/Library/library_export_canvas.py: sync_state recompose has no production caller (dead)
- tldw_chatbook/Widgets/Library/library_media_trash_canvas.py: only the known TASK-31509 re-measure
- tldw_chatbook/Widgets/Library/library_file_notes_git_panel.py: rows are session-scoped (small) and rendered in a worker; the arrow-key probe was fixed in #2739 (its per-keystroke residue comes through the workspace's set_mutating, reported in F5)
- tldw_chatbook/Widgets/Library/library_choice_strip.py, library_character_return.py, library_emergency_return.py, library_file_notes_events.py, library_note_work_pane.py, library_note_folder_dialog.py: small, cold or modal
- Folder Files _initialize, scan, open and save I/O all run in asyncio.to_thread (good); `_configured_sync_folder` now costs 0.003 ms per pair of reads (TASK-32804.1 fastpath)

## census
| Measurement (headless run_test, real CSS bundle, isolated profile) | Result |
|---|---|
| FileNotesService.reconcile, nothing changed, 30 / 1,000 / 5,000 files | 58 / 51-64 / 269-320 ms per call (worker thread, GIL-held) |
| Share of the 30-file reconcile spent in `_replica_execution_scope` (storage admission) | ~80% (5 acquire_storage, ~1,000 dir-fd open() per reconcile) |
| Folder Files poll tick, loop side, idle / with active search | 9.1 ms + 2 layouts (5.5 ms) / 12.1 ms + 2 layouts (19.3 ms) |
| Folder Files editor keystroke: handler body, then screen layouts | 6.1 ms sync + 1 layout per key (control: 0 layouts); +25 ms wall per key vs stubbed |
| Autosave then next poll tick | full navigator `_rebuild_tree` 5/5 cycles |
| Ingest queue panel recompose, mixed rows, 50 / 200 / 500 jobs | 68-114 / 217-250 / 527-574 ms (131 / 446 / 1,076 widgets) |
| Folder submit: Library listener state rebuilds, 100 / 500 / 1,000 files | 56 ms / 1.43 s / 5.78 s (O(N^2)) |
| Conversation reader progressive load, 100 / 400 / 1,000 messages | 177 ms / 922 ms / 7.07 s; Static.update 221 / 3,881 / 24,701 |
| Conversation reader no-change sync_state, 100 / 400 / 1,000 messages | 76 / 166 / 292 ms |
| Conversations list row click (sync_state recompose), 20 / 50 rows | ~180-226 / 239-382 ms wall; idle baseline 45 ms, so ~140 / ~235 ms net |
| Pre-mount add_class with the 3,612-rule sheet vs update=False | 0.343 ms vs 0.001 ms; Button() 0.52 ms, Button(compact=True) 0.99 ms |
| Session change log snapshot+coalesce, 100 / 1,000 / 3,000 changes | 0.05 / 0.14 / 0.34 ms per call |
| get_cli_setting pair (the core review's _configured_sync_folder P2) | 0.003 ms (now fine) |


# slice-64

## summary
Slice #64 (Widgets/Library#2, 12 files, ~11.8k lines). Hot paths traced from outside the slice:
(a) Every Notes editor keystroke: `handle_library_note_{body,title,keywords}_changed` → `_apply_library_note_presentation_state` → `LibraryNotesCanvas.apply_session_state` → `apply_compact_presentation` + `update_note_chrome_facts`.
(b) Every caret move in the note body: `@on(TextArea.SelectionChanged)` → `update_note_chrome_facts`.
(c) Every Notes sync through `canvas_sync._sync_library_canvas(screen, "notes")`, which has about 50 call sites. These include note-row press, note-detail landing, folder expand/collapse, select-mode toggles and rename autosave. Each one calls `LibraryNoteWorkPane.sync_state` and then `LibraryNotesCanvas.sync_state`.
(d) Prompt Basic-editor keystrokes through `_basic_prompt_text_changed`.
(e) Library rail syncs and Search/RAG panel visits.

Overall the slice is disciplined:
- Static writes on the typing path are diff-checked.
- Id `query_one` calls hit Textual 8's per-node id cache, so about 85 lookups per keystroke cost under 0.1 ms.
- Backlinks are capped at 50, tree rows page at 20 per branch, and RAG history is capped at 10.
- The census runs off the event loop.
- The work panes already short-circuit unchanged syncs.

Two structural problems dominate.
1. `LibraryNotesCanvas.sync_state` (and the list canvases for Prompts, Skills and Search/RAG) always ends in `refresh(recompose=True)`. There is no check for unchanged inputs and no in-place selection path. As a result:
   - Each note open recomposes the Items list twice.
   - Any list interaction while a note is open (unless an editor field has focus) rebuilds the whole editor.
2. The editor compose mounts the full Markdown preview of the body even though Preview is hidden by default. Measured cost is 22 ms at 2 KB, 75 ms at 10 KB and 405 ms at 35 KB (697 hidden block widgets), paid on every note open and on every editor rebuild.

Each keystroke and caret move also forces a full-screen reflow, from two sites that Textual cannot de-duplicate:
- `styles.max_width = None` always calls `refresh(layout=True)`.
- `Static.update()` defaults to `layout=True` on a Static whose size is fixed.
Measured reflow cost is 1.2 ms with 356 widgets and 2.8 ms with 656.

Lesser items:
- Prompt Basic typing reloads the hidden Advanced card's TextArea on every keystroke and runs its preview sync twice.
- The Search/RAG legacy-chunk census runs twice per visit.
- The rail's in-place sync forces layout even for identical text.
- A few DOMQuery truth-tests remain.
- The Skills tool picker rebuilds its options on every keystroke.

All measurements come from isolated pure-Textual 8.2.8 headless probes with default CSS and no tldw imports. The Library's real CSS is heavier, so real costs are likely higher. Suggested PR grouping:
- PR-A: Notes canvas sync short-circuit + in-place selection + editor delegation (F1), bundled with the lazy hidden preview (F2) because they share the editor rebuild path.
- PR-B: layout-free per-keystroke writes (F3, F7).
- PR-C: list-canvas equality guards for Prompts/Skills/RAG (F4).
- PR-D: prompt editor per-keystroke hygiene (F5), the double census (F6), the remaining DOMQuery truth-tests (F8) and the tool picker (F9).

## clean areas
- tldw_chatbook/Widgets/Library/library_notes_canvas.py apply_session_state: Static writes are diff-checked (_static_text), Input/TextArea writes skip focused fields, the hidden-preview re-render while typing is gated (TASK-32804.11 AC#2 holds), and the ~85 id query_one calls per keystroke hit Textual's per-node id cache (<0.1 ms total)
- tldw_chatbook/Widgets/Library/library_notes_canvas.py apply_compact_presentation / sync_state: the bool(query('#id')) sites named by TASK-32804.11 AC#1 are converted to query_one+NoMatches; the remaining self.query() calls in sync_state are only truth-tested on the import/lasting_add same-mode branch
- tldw_chatbook/Widgets/Library/library_notes_canvas.py list compose: tree rows page at 20 per branch (LIBRARY_NOTES_TREE_PAGE_SIZE), backlinks capped at 50, trash page 20, tiebreak labels O(rows), on_resize/apply_pane_width recompose only when a width decision flips
- tldw_chatbook/Widgets/Library/library_rail.py: route switches use in-place apply_selection, LibraryDetailsRow.on_resize is guarded against identical re-hangs, fold-cue writes are guarded, the width contract short-circuits on an equal contract, and LibraryRailRowButton label refits are reactive no-ops when unchanged
- tldw_chatbook/Widgets/Library/library_prompt_work_pane.py and library_skill_work_pane.py: sync_state short-circuits unchanged kwargs, the template for F4
- tldw_chatbook/Widgets/Library/library_search_rag_panel.py: result cards bounded by top-k, history capped at LIBRARY_SEARCH_HISTORY_LIMIT=10, the census is offloaded via asyncio.to_thread (TASK-21126), the pricing catalog is a cached singleton, and the per-edit recovery log is deduped
- tldw_chatbook/Widgets/Library/library_rechunk_run.py: the rare heavy op runs on a thread worker (asyncio.run inside that thread is acceptable at this frequency); session-owned with slot exclusion
- tldw_chatbook/Widgets/Library/library_notes_sync_roots_canvas.py: paged roots, recompose only on an explicit snapshot sync, no I/O
- tldw_chatbook/Widgets/Library/notes_recovery_dialog.py: 4 Hz poll only while the modal is open, in-memory predicate, the timer handle is stopped on finish/stale (TASK-32800.5)
- tldw_chatbook/Widgets/Library/library_review_set_picker.py and prompt_delete_confirmation_modal.py: dumb modals with precomputed rows and a bounded preview, no I/O
- tldw_chatbook/Widgets/Library/library_skills_canvas.py LibrarySkillsTrustHeader.sync_state: equality guard before recompose; list sync has a header_only in-place path

## census
| Probe (pure Textual 8.2.8, headless 211x44, default CSS) | Result |
|---|---|
| screen._refresh_layout (full reflow), 356 widgets | 1.21 ms median |
| screen._refresh_layout, 656 widgets | 2.81 ms median |
| `styles.max_width = None` sets `_layout_required` | 10/10 calls (vs 1/10 for an unchanged scalar) |
| Hidden `Markdown.update`, 2 KB note | 22.0 ms wall, 40 blocks |
| Hidden `Markdown.update`, 10 KB note | 74.5 ms wall, 200 blocks |
| Hidden `Markdown.update`, 35 KB note | 404.9 ms wall, 697 blocks |
| Hidden `TextArea.load_text`, 2 / 10 / 35 KB | 0.10 / 0.38 / 1.34 ms median |
| `TextArea.text` join, 35 KB | 3.7 us |
| Canvas recompose+settle, 20 / 60 / 120 list rows (46 / 86 / 146 widgets) | 51.6 / 80.0 / 113.2 ms median |


# slice-65

## summary
Slice 65 (Widgets/Persona_Widgets, 37 files, 15.7k lines). It covers the Personas workbench widgets (library, inspector, preview, character/profile editors, dictionary/lore detail, pickers) and the Persona Buddy surfaces (the floating PersonaBuddyWidget plus the conversation, workspace and management modals). All of them are reached lazily, either through the screen registry (personas_screen imports them) or through UI/Navigation/buddy_* openers. The package __init__ is light and nothing in the slice is on the boot path.

Hot paths I traced:
- Personas library search, page, sort and mode switch go through PersonasLibraryPane.update_rows.
- The inspector's conversation search posts one ConversationSearchChanged per keystroke.
- The preview pane's streamed Test Reply calls append_reply_chunk once per chunk.
- The Buddy has a 10 Hz snapshot poll and is re-mounted on every screen switch.
- The Buddy conversation modal re-projects every 0.2 s; the workspace inbox refreshes every 1 s; speech controls refresh every 0.5 s.
- The character and profile editors run dirty-tracking on every keystroke.

Overall health: I/O discipline is good. Every DB, decode and conversion step in the Buddy and import flows goes through asyncio.to_thread, drain_thread or a thread worker. The known Buddy fixes are present and correct: TASK-21122's paint gate, geometry debounce and retry backoff, and TASK-21595's layout=False frame updates. Library search and the pickers are debounced.

The main structural cost is widget-per-row lists (ListView of ListItem+Static) that are cleared and fully rebuilt on each interaction. In a bare Textual app I measured about 49 ms to rebuild 20 rows and about 170 ms for 200 rows, against about 5 ms and 24 ms for an OptionList. That cost runs per keystroke in the inspector conversation search, which has no debounce and also opens one BEGIN IMMEDIATE write transaction per keystroke on a thread that cannot be cancelled. It also runs per search, page or sort in the library pane, and on open and per filter in five near-identical picker modals.

Secondary issues:
- The Buddy conversation modal's 5 Hz tick copies the entire session transcript (measured 3.4 ms per tick for 1000 messages) and re-claims decision views twice per tick.
- Streamed preview replies re-parse the whole growing line on every chunk. This is quadratic: measured about 0.95 s of event-loop time for an 8 KB reply.
- Each screen switch with the Buddy enabled re-runs the full visual resolve and PIL decode, because view_generation is part of the resolution key and no prepared frames are cached.
- The Buddy's 10 Hz poll survives even though the controller already pushes change notifications.

Suggested PR groups:
- **(A) Personas list rendering:** OptionList or recycled rows for the library pane, inspector conversations and one shared picker (F2, F7, render half of F1).
- **(B) Conversation keyword search:** debounce it, and call ensure_keyword_index once per selection instead of per keystroke (F1).
- **(C) Buddy modal projection ticks:** bounded transcript tail, revision gate, suspend gate, compare-before-update (F3, F8, same pattern as TASK-31506).
- **(D) Buddy view lifecycle:** timer set to the next lease expiry instead of the 10 Hz poll, reuse the accepted visual across views, frame-prep LRU (F4, F5).
- **(E) Preview streaming render:** throttle updates and style the line once at finalize (F6).
- **(F) Hygiene:** editor snapshot short-circuit, inspector action-state coalescing, lazy import of the Actor Pack review dialog (F9, F10, F11).

## clean areas
- tldw_chatbook/Widgets/Persona_Widgets/__init__.py, personas_messages.py, personas_pane_messages.py and personas_state.py: pure Message and state dataclasses with light imports (textual.message only). They do not drag heavy modules onto the boot path.
- persona_buddy_widget.py paint path: the TASK-21122 paint-authority gate, 250 ms geometry-persist debounce and capped exponential retry backoff are present and correct, and TASK-21595 layout=False is used on every frame update. Only the residual poll (F4) and the per-view re-resolve (F5) remain.
- buddy_management_modal.py: artwork listing, preview rendering, paging and apply all run through asyncio.to_thread callbacks in UI/Navigation/buddy_management.py. There is no sync I/O on the loop.
- buddy_character_review.py and petdex_import_review.py: conversion, preview decode, source checks and publish all go through asyncio.to_thread or drain_thread and are fenced by generation counters. The per-keystroke _sync_controls only reads widget state.
- actor_pack_import_review.py: one-shot modal. The PIL portrait decode in compose is small and runs once per review. Only its module-scope import is noted (F11).
- personas_character_card_widget.py and persona_profile_card_widget.py: bounded sanitize and display toggles, no remounts.
- personas_character_dictionaries.py, personas_character_world_books.py and personas_policy_rules_editor.py: small DataTable or ListView lists rebuilt only on discrete actions.
- personas_dictionary_detail.py and personas_dictionary_validation.py: validation is linear. The finding-to-pattern lookup is O(findings x entries) but dictionaries are small; Changed handlers only compare a small settings dict.
- personas_lore_detail.py, personas_lore_tryit.py and personas_dictionary_tryit.py: button-driven. The difflib word_diff runs on a user-typed sample per Run press, which is acceptable.
- personas_character_tts_widget.py and character_tts_portability_dialogs.py: state-push rendering only, no I/O.
- personas_visual_identity_pack_widget.py and personas_persona_visual_pack_widget.py: the per-keystroke filter is in-memory over pack assets. Preview decode is delegated to exclusive screen workers that use asyncio.to_thread.
- personas_conversation_transcript_widget.py: capped at 200 messages, one Static each. Measured about 50 ms per open in a bare app; a single joined Static measured no cheaper (about 70 ms), so I did not file it.
- personas_inspector_pane.py conversation paging: append-only with tail replacement, which is the good pattern. Only the un-debounced search path (F1) is a problem.
- Library-pane search and all five picker filters are debounced (PERSONAS_SEARCH_DEBOUNCE_SECONDS or SEARCH_DEBOUNCE_SECONDS=0.2). The debounce is correct; the cost is the row widgets (F2, F7).
- Logging: no eager f-string logger calls on hot paths in the slice. No deepcopy, no re.compile inside functions (all regexes are module-level).

## census



# slice-66

## summary
Slice #66 (Widgets/Settings_Widgets, 10 files, 11,261 lines) at dev 840ed2ca58. All measurements come from isolated probes: scratch HOME/XDG/TLDW_CONFIG_PATH and TLDW_TEST_MODE=1. Each probe mounts the real SpeechTTSSettingsPanel in a Textual 8.2.8 run_test app at 211x44 and uses A/B monkeypatches. Scripts are in scratch/s66/probe_tts*.py.

Hot paths in scope:
(1) The Speech & TTS draft-field keystroke path: Input/Select/Switch.Changed -> handle_draft_field_changed -> _announce_draft_state -> has_unsaved_changes, draft_snapshot, _refresh_status_rows, then DraftModified -> settings shell _update_draft_status_widgets.
(2) Speech category visit, which rebuilds the panel every time (the settings category swap re-mints pane regions).
(3) Post-action whole-panel recomposes after Save, Revert, Restore defaults and package add/remove.
(4) My Profile (PersonalContextSettingsPanel) record, scope, editor and interview-mode interactions, all driven by recompose=True reactives.
(5) Workspaces-row rebuilds that construct WorkspaceChangeReviewPanel, plus its 0.5 s preparing poll.

Headline: the Speech panel's per-keystroke cost is not the Python compute (~1.3-2.4 ms). It is 13-14 unconditional Static.update()/Button.label/refresh(layout=True) calls per keystroke. They force a relayout of the 255-395-widget panel: +21 ms (openai) / +24 ms (audio_cpp) per keystroke, measured. Equality-guarding the updates brings keystroke latency back to the plain-Input baseline, also measured.

The same announce path fires 30x during every Speech visit (audio_cpp) as freshly composed controls post their initial Changed events. That is ~49 ms of compute plus 30 DraftModified round trips into the shell. The visit itself mints 395 widgets, 187 of them never displayed (collapsed or mode-hidden). Seven whole-panel recompose sites survive TASK-15475's card-scoped refactor, at ~180-280 ms each.

Overall health: good discipline on I/O. Every DB and service mutation in the Personal Context panel/modals and Tool Profiles runs in thread workers. The server-switch test is async httpx with egress check and timeout. speech_tts_panel_types keeps the 6k-line panel off the app import path (TASK-21108 holds). The remaining costs are render and recompute patterns.

Out-of-slice observations to route elsewhere (not filed here):
(a) config.get_user_data_dir() measured ~21 ms per call when not inside a participants operation (interprocess data-root lock plus secure-directory verification). Any event-loop caller pays it; reached here via ShadowRepoService() at settings_screen.py:20711.
(b) Workspaces/change_review_consent.py:387 _worker_loop: 2 daemon threads poll queue.get(timeout=0.05) forever once first scheduled, ~40 wakeups/s idle.
(c) settings_screen.py:9056 _update_draft_status_widgets (the DraftModified handler) also does unguarded Static.update and Button.label per keystroke, so the in-app keystroke cost is at least the panel-only number measured here.

## clean areas
- Widgets/Settings_Widgets/__init__.py: trivial docstring-only package init, no eager submodule imports
- Widgets/Settings_Widgets/speech_tts_panel_types.py: pure dataclasses and validators, the only module on the app boot path (app.py:380). No I/O, and its import stays light per TASK-21108. Its validated-copy cost (~0.25 ms) only matters through the per-keystroke announce path (F2)
- Widgets/Settings_Widgets/server_switch_modal.py: connection test is an async @work(exclusive) with an egress policy check, a 5 s httpx timeout and follow_redirects=False. The per-test AsyncClient is a one-shot user action. settings_screen imports it function-locally
- Widgets/Settings_Widgets/personal_context_link_modal.py: pure presentation. Decision clicks update one Static plus the Approve button in place, with no recompose and no I/O
- Widgets/Settings_Widgets/personal_context_review_modal.py: proposal resolve and interview commit/rewrite/cleanup all run in thread workers via call_from_thread. The row-Apply `_revision` recompose rebuilds a small modal (P3 hygiene at most, not filed)
- Widgets/Settings_Widgets/tool_profiles_panel.py: apply_listing is equality-guarded and serialized by a lock. The listing is fetched in a thread worker by settings_screen. The per-profile widget count is small
- Widgets/Settings_Widgets/tool_pack_import_review.py: four review modals with a fixed widget count and pure string formatting. The only overlap is a duplicated _plain_text helper shared with tool_profiles_panel (no speed cost)
- personal_context_panel.py I/O: settings_snapshot load, every mutation and every export run with thread=True workers, generation-fenced. No sync DB or crypto work on the loop
- speech_tts_settings_panel.py construction: __init__ measured 0.34 ms. Config is read from the shell's snapshot (TASK-32804.8 holds). speech_local_dependency_availability(refresh=True) (9 find_spec calls) is negligible
- speech_tts_settings_panel.py card-scoped dropdown rebuilds (_replace_card_bodies for provider/default/model/voice policy changes) are in place per TASK-15475. Package scans use scan_audio_cpp_package_root_async with group cancellation
- speech_tts_settings_panel.py request_save: the config write runs sync on an explicit Save press. Prior review judged this the repo-wide Save pattern and did not flag it; not re-filed

## census
| Measurement (isolated run_test, 211x44, Textual 8.2.8) | Value |
|---|---|
| Keystroke into draft Input, openai panel (255 widgets): full / announce no-op / Static-update equality guard | 116.4 / 95.6 / 95.5 ms median (pilot-inclusive; delta = +21 ms) |
| Keystroke into audio_cpp base URL (395 widgets): full / guard (Static + handoff guard) | 120.2 / 95.5 ms (delta = +24 ms) |
| Layout refreshes per keystroke: full vs guarded | 14 vs 1 (openai); 17 vs 1 (audio_cpp) |
| _announce_draft_state in-call compute | 1.3-2.4 ms (has_unsaved 0.23, draft_snapshot 0.75, _refresh_status_rows 0.31) |
| _announce_draft_state calls during one Speech visit mount | 30 (audio_cpp) / 14 (openai), 49.3 / 19.9 ms compute |
| Speech panel mount+settle (announce no-op, minus ~50 ms pilot) | ~265 ms audio_cpp / ~180 ms openai |
| Widgets minted / not displayed (audio_cpp) | 395 / 187 |
| Whole-panel recompose (openai) vs 2-card rebuild | ~180 ms vs ~76 ms (bare pilot.pause 25.8 ms subtracted) |
| ShadowRepoService().available / get_user_data_dir() / shutil.which('git') | 25.1 / 21.5 / 0.019 ms |


# slice-67

## summary
Slice #67 Workspaces (21 files, ~14.2k lines) at 840ed2ca58. Hot paths: (1) LocalWorkspaceRegistryService, the SQLite registry, which the Library workspace rail/reader, Console context builds, per-send turn capture, Settings and the file inspector all call. (2) display_state builders that run on the loop: build_library_workspace_depth_state and build_console_workspace_state. (3) Pure Console rail projections (conversation_browser_state, workspace_tree_state). (4) Agent Change Review. Per turn it does a B and an E shadow-git snapshot on a fixed worker pool. B gates the first mutating tool dispatch behind a fixed 3 s wait. (5) Console Inspect-rail Environment polling: a local tier every 10 s on a worker thread, plus a gh tier with a 60 s TTL.

Main result: every registry method opens its own WorkspaceDB connection()/transaction() scope. Each scope pays the Backup_Recovery storage-admission handshake: ~245 verified-directory posix.open calls, 2.84 ms per read measured, while the SQL itself takes ~25 us. TASK-32804.1 removed this cost for config reads only. Wherever a loop calls the registry per record, that cost multiplies into real stalls. The worst case is the Library workspace-depth build: one get_item_memberships per visible record (~170), each a full table SCAN with no index, run on the event loop after every rail-row press, snapshot change and link/unlink. It measured 484 ms per build. Wrapping the loop in one outer scope brings it to 28 ms, and a batched query takes ~0.05 ms.

Change Review snapshots do O(index entries x depth) stat walks: 1.09 s per snapshot on a 24k-file tree, versus 78 ms memoized. They also repeat the tree walk three times per turn, re-spawn 7 git config pins, and re-acquire admission for every git command. All of this sits on the B critical path of the 3 s tool-dispatch gate.

The Environment local tier spawns ~10 git processes (~0.5 s CPU) every 10 s while the Inspect rail is open. `git status --porcelain=v2 --branch` would replace most of them.

Smaller items: 3 Change Review threads that poll forever at ~80 wakeups/s after first use; an unbounded handoff-row list that the Console Details tray recomposes on any unrelated context change; metadata scrubbing that re-normalizes constants per key; git gc on every launch; per-path git spawns in revert preflight; a whole-tree re-walk per file-filter keystroke; and dead helpers in display_state.

The pure projection code is healthy (<0.3 ms per build). The package __init__ is a proper PEP 562 lazy facade. Most blocking file and git work is already threaded. The previous generation-keyed Console registry memo (TASK-21118/22201) works as intended.

## clean areas
- tldw_chatbook/Workspaces/__init__.py - PEP 562 lazy export facade; resolves on first attribute access only
- tldw_chatbook/Workspaces/conversation_browser_state.py - pure; measured 0.263 ms per build for 75 rows (dedupe, sort, age labels); run on the 0.2 s run tick only once per tick (TASK-22201 dedupe)
- tldw_chatbook/Workspaces/workspace_tree_state.py - pure; measured 0.176 ms per build for 225 rows / 5 workspaces
- tldw_chatbook/Workspaces/conversation_attention.py - pure, tiny
- tldw_chatbook/Workspaces/eligibility.py - pure
- tldw_chatbook/Workspaces/assistant_defaults.py - pure; persona_policy/permission_store imports are function-local
- tldw_chatbook/Workspaces/agent_provisioning.py - lazily imported on a post-ready timer; backfill is flag-gated and runs once per boot
- tldw_chatbook/Workspaces/recovery.py - trivial adapter, no git process
- tldw_chatbook/Workspaces/models.py - dataclass validation is cheap except scrub_secret_metadata at large exclusion lists (see F8)
- tldw_chatbook/Workspaces/environment_status.py gh tier - worker thread, 5 s timeout, 60 s TTL keyed on (root, branch), gated on rail open
- tldw_chatbook/Workspaces/git_workspace.py commit/push/diff/untracked_preview - bounded reads (untracked_preview caps at max_lines*400 bytes); callers in change_review_screen run off-thread
- tldw_chatbook/Workspaces/file_inspector.py list_directory/read_file - 200-entry pages, 10k scan cap, 8-file LRU page cache, 8 continuation tokens; modal runs via asyncio.to_thread with a latest-only lane and generation cancellation
- tldw_chatbook/Workspaces/change_review_finalization.py fs workers - fixed 2-thread pool, blocking get() with _STOP sentinel (only the publisher polls, see F7)
- tldw_chatbook/Workspaces/change_turn_tracker.py - production uses the coordinator's fixed workers; begin_turn's thread-per-turn is fallback-only
- tldw_chatbook/Workspaces/change_bounds.py knobs - change_review_setting re-measured at ~1.5 us/call via get_cli_setting fastpath (the first-call cost is only the config import)
- tldw_chatbook/Workspaces/registry_service.py keystroke path - Console memoizes active-workspace + display reads by mutation_generation (TASK-21118/22201) and serves them without SQL while the registry is unchanged; get_workspace_scope is read via asyncio.to_thread
- tldw_chatbook/Workspaces/change_retention.py - runs via asyncio.to_thread; only the always-gc policy is flagged (F9)

## census
| Measurement (isolated scratch env, audit tree 840ed2ca58) | Result |
|---|---|
| WorkspaceDB registry point read (get_item_memberships), per call | 2.84 ms (cProfile: 245 posix.open per read via participants._core_transaction -> acquire_storage) |
| 170 raw sqlite point queries, same table, no admission | 4.27 ms total (~25 us each); plan = SCAN workspace_memberships |
| build_library_workspace_depth_state, 170 records (100 notes/50 media/20 convs) | 484 ms (50 memberships), 467 ms (500), 492 ms (3000) |
| same build inside one outer db.connection() scope | 28.3 ms |
| batched IN query, 100 ids | 0.05 ms |
| build_console_workspace_state uncached | 8.6 ms (M=50), 10.8 ms (M=500), 22.2 ms (M=3000; 3000 handoff rows) |
| _nested_owner loop over 24,212 index entries (avg depth 3.12) | 1,091 ms; per-directory memo 78 ms |
| scan_root on audit tree (24,214 files) | 168.5 ms; os.scandir variant 69.1 ms |
| git status --porcelain=v1 -z -uall (audit tree) | ~0.06 s wall, ~0.25 s CPU |
| git diff --numstat -z HEAD | ~0.03 s wall, ~0.22 s CPU |
| 5 small git probes (rev-parse/symbolic-ref/remote -v/...) | 66 ms wall |
| BacklogTaskScanner.scan (4,478 task files, 950 entries) | cold 191 ms, warm 30.8 ms |
| 3 idle Change Review poll loops (0.025 s + 2x0.05 s timeouts) | 0.137% core, ~80 wakeups/s |
| WorkspaceRuntimeBinding construct with 200 exclusions | 2.43 ms |
| get_cli_setting warm | 1.3-1.5 us (fastpath confirmed) |


# slice-68

## summary
Slice #68 (tldw_api#1, 34 files, 31,847 lines): the package `__init__` lazy facade, the 16,689-line `TLDWAPIClient` (`client.py`), `MCPUnifiedClient`, and about 30 pydantic schema modules.

Hot paths identified:
- The only boot-path members are `tldw_api/__init__` (0.34 ms), `exceptions` (0.13 ms) and `notes_workspace_limits` (0.05 ms), which are clean, plus `kanban_schemas` (about 16 ms). `kanban_schemas` is already known as TASK-21107.
- `client.py` is correctly kept out of the local-mode boot closure (TASK-285). The code where the user actually pays for this slice is the first `TLDWAPIClient` construction. That construction is `build_runtime_api_client` doing `from tldw_chatbook.tldw_api import TLDWAPIClient`, which imports `client.py` and 51 of its 52 schema modules at module scope (72 modules, about 1,300 pydantic classes). Measured at 396 to 437 ms warm, isolated, with pydantic, httpx and kanban already loaded.
- In server mode this import runs on the Textual event loop. It happens 0.1 s after the first frame, through a `set_timer` callback (the Collections capture authority activation), and again through the async `handle_runtime_backend_changed` path. This is new and not covered by an open task (F1). The structural root is that `client.py` needs 544 schema names at runtime (382 more are annotation-only), so a TYPE_CHECKING move alone cannot fix it. Pydantic `defer_build=True` alone halves the cost (measured 396 to 193 ms). Lazy model resolution through the existing PEP 562 facade would remove most of it (F2).
- Per request, the client is a thin, well-shaped async transport:
  - one pooled `httpx.AsyncClient` per instance, with a connect-timeout ceiling;
  - SSE and NDJSON streaming via `aiter_lines`, with list-join buffering rather than `str +=`;
  - one shared error translator;
  - an O(n) endpoint guard;
  - no sleeps, threads, loops-per-call or eager logging.

The remaining costs are architectural:
- Each instance builds its own httpx pool and SSLContext (measured 8 to 10 ms per construction). 29 `from_config` services each get a private provider, so the pool is fragmented about 30 ways and every first call pays a fresh TLS handshake (F3).
- Every binary download is fully buffered in memory by `_binary_request`, and the chatbook export download then writes the whole blob synchronously from a button handler (F4).

Overall health is good. The package-level laziness is working, and the one real cost is the server-mode first-construction import, which has two low-risk fixes (warm the import on a thread, or `defer_build`) and one larger structural one (lazy model resolution).

## clean areas
- tldw_chatbook/tldw_api/__init__.py: PEP 562 lazy facade, measured 0.34 ms import; `__getattr__` caches on the module; `__dir__` is only used by dir()
- tldw_chatbook/tldw_api/exceptions.py and notes_workspace_limits.py (boot path): stdlib-only, 0.13 ms and 0.05 ms
- tldw_chatbook/tldw_api/client.py request primitives (_request, _headers_request, _stream_request, _sse_request, _stream_sse_request):
- one pooled AsyncClient per instance, reused across calls
- connect timeout capped at 15 s
- SSE data lines buffered in a list and joined once per event
- NDJSON parsed per line
- no str+= accumulation, no sleeps or polling, no per-call thread or loop
- one shared error translator (_raise_api_error_from)
- redirect refusal closes the response
- tldw_chatbook/tldw_api/client.py per-call helpers:
- _reject_unsafe_endpoint -> validate_request_endpoint_path does O(len) string checks with no regex
- params dict comprehensions are trivial
- only one logger call (a warning on a bad NDJSON line, with lazy {} formatting)
- tldw_chatbook/tldw_api/mcp_unified_client.py: thin async wrapper over root_client._request; no loops, polling or client construction; stream_governance_events has zero consumers (cold)
- Schema modules in slice (account_security, audio, audiobook, auth_user, character_persona, chat_conversation, chat_dictionary, chat_documents, chat_grammar, chat_loop, claims, collections_feeds, companion, connectors, data_tables, evaluations, feedback, flashcards, kanban, llm_provider, mcp_governance, mcp_unified, media_reading, meetings, notes_workspace, notifications_reminders, ocr_vlm, outputs, personalization):
- no module-scope I/O, no re.compile, no heavy module-scope computation
- validators are light
- their only cost is pydantic class construction at import (see F2)
- media_reading_schemas lazy `STT.persistence` import inside the provenance validators: free in-app, because STT.persistence is already in the ui_ready closure.
- The validate -> dump -> load round trip per ServerMediaListItem is intentional (canonical, size-bounded) and costs tens of microseconds per item; judged not worth a finding.
- character_persona_schemas at personas_screen/ccp_persona_handler module scope: loaded by the background screen pre-import thread (preimport_payload.json), not on the loop
- kanban_schemas on the boot path (about 16 ms warm, measured): already tracked by TASK-21107 and TASK-21239; not re-reported

## census



# slice-69

## summary
Slice #69 (tldw_api#2) is 25 pydantic request/response schema modules plus tldw_api/utils.py, 6,798 lines and about 516 model/enum classes. It contains no per-interaction code: no timers, no DB access, and no widgets. Its validators are trivial O(n) checks, and regexes are compiled at module scope. Its one real cost is import time. Pydantic 2.12 builds a core schema for every class when the module is imported, and none of the 1,266 BaseModel subclasses in tldw_api use defer_build (0 uses repo-wide). The PEP 562 facade and the task-285 guard test keep these modules off `import tldw_chatbook.app`. The cost is instead paid in one synchronous block the first time TLDWAPIClient is resolved, because client.py:32-1066 eagerly imports about 54 schema modules. I measured that block at 402-445 ms with warm .pyc files, with pydantic, httpx and loguru already loaded; this slice's share is 146-169 ms. For users whose runtime source is "server", that happens on the event loop on every boot, 0.1 s after first paint: set_timer calls _deferred_wire_collections_capture_services, which leads to RuntimeServerContextProvider.build_client and then build_runtime_api_client, which does a function-local `from tldw_chatbook.tldw_api import TLDWAPIClient`. The same happens on the first switch to the server source. The guard test runs with no server configured, so it cannot see this path.

Setting defer_build=True cuts the slice import from 146-169 ms to 65-68 ms and the full client import from 402-445 ms to 177-203 ms. The first validation of a model then pays a one-time build of about 1 ms, measured for the nested SyncV2PushRequest. Structural notes: study_extensions_schemas.py is dead, with 14 classes that all duplicate other modules. Local-mode code imports pydantic schema modules only to reach pure helpers and constants, costing 3-6 ms once per process. Probes ran with isolated HOME, XDG_*, TLDW_CONFIG_PATH and TLDW_TEST_MODE, with PYTHONDONTWRITEBYTECODE set, and imported only tldw_api modules. Scratch files are under scratchpad/audit/scratch/s69/.

## clean areas
- tldw_chatbook/tldw_api/utils.py: prepare_files_for_httpx passes open file handles to httpx, which streams them, so there are no whole-file reads. model_to_form_data is a single O(fields) pass. The eager f-string logging is on cold upload paths only.
- tldw_chatbook/tldw_api/sync_schemas.py validators (SyncV2Envelope alias/private-payload checks, SyncV2PushRequest dataset-id loop, capability map sanitizers capped by _MAX_CAPABILITY_* constants): O(n) per request, no deepcopy, no json round-trips.
- tldw_chatbook/tldw_api/skills_schemas.py: SKILL_NAME_PATTERN and SUPPORTING_FILE_NAME_PATTERN are compiled at module scope. Validators are linear. The per-file content.encode() used for byte counting is capped at 5 MB per file and 25 MB total, on a cold skill create/update path.
- rag_admin_schemas.py, research_search_schemas.py, writing_manuscript_schemas.py, watchlists_schemas.py, slides_schemas.py, prompt_studio_schemas.py, web_clipper_schemas.py: validators are trivial (dict copy, set difference over tiny literal sets, alias lookup).
- Across all 25 slice modules: no module-scope I/O, TypeAdapter, model_rebuild, model_json_schema, create_model, lru_cache, or costly config (validate_assignment, revalidate_instances, json_schema_extra) was found.
- prompt_chatbook, quizzes, research_runs, scheduled_tasks_automation, schemas, server_runtime, sharing, storage, study_suggestions, text2sql, tools, translation, user_governance, user_keys, voice_assistant: plain field declarations. Their only cost is the class-construction cost covered by F2.
- Boot path: Tests/Utils/test_tldw_api_schema_deferral.py confirms `import tldw_chatbook.app` loads no slice module. Only kanban_schemas is allowlisted, and that is already TASK-21107.

## census
Per-module import cost (cumulative ms, from `python -X importtime` of tldw_api.client with warm .pyc and pydantic/httpx/loguru preloaded, dev 840ed2ca58):

| module | classes | import ms |
|---|---|---|
| sync_schemas | 47 | 32.5 (17.6 self + 14.9 tldw_profile_core) |
| writing_manuscript_schemas | 55 | 14.6 |
| watchlists_schemas | 65 | 13.8 |
| schemas | 20 | 12.0 |
| slides_schemas | 32 | 10.4 |
| prompt_studio_schemas | 35 | 8.1 |
| prompt_chatbook_schemas | 21 | 6.3 |
| rag_admin_schemas | 26 | 6.0 |
| research_search_schemas | 20 | 5.2 |
| scheduled_tasks_automation_schemas | 14 | 5.0 |
| sharing_schemas | 24 | 4.5 |
| web_clipper_schemas | 13 | 4.4 |
| voice_assistant_schemas | 21 | 4.3 |
| quizzes_schemas | 13 | 4.2 |
| research_runs_schemas | 16 | 3.5 |
| storage_schemas | 17 | 3.1 |
| skills_schemas | 11 | 2.8 |
| server_runtime_schemas | 14 | 2.8 |
| user_keys_schemas | 13 | 2.6 |
| user_governance_schemas | 8 | 2.6 |
| study_suggestions_schemas | 8 | 1.8 |
| text2sql_schemas | 3 | 0.7 |
| tools_schemas | 4 | 0.7 |
| translation_schemas | 2 | 0.5 |
| study_extensions_schemas | 14 | never imported (dead) |

Whole-slice import in a fresh process, 3 runs each: 168.8 / 146.0 / 151.7 ms without defer_build, and 67.6 / 65.3 / 67.9 ms with `BaseModel.model_config['defer_build']=True`. Full `tldw_api.client` import: 401.9 / 442.8 / 444.7 ms without, 193.9 / 177.0 / 203.0 ms with. First validation of SyncV2PushRequest (nested): 0.03 ms without defer, 1.0 ms with it. Later validations take 0.006 ms either way.


# slice-7

## summary
Slice #7 "Chat#1" (28 files, 24,743 lines) at pinned origin/dev 840ed2ca58. Hot entry points traced: (a) the per-send Console write path. The async controller (_run_direct_provider_reply, _attach_stream_usage) calls store.mark_message_complete, set_message_usage and the trajectory flush synchronously on the loop, and each lands in chat_persistence_service.update_message_content or write_trajectory_rows. (b) The Console send payload build (attachment_core.image_url_part), called 2-3 times per send. (c) Per-visit conversation reads through chat_conversation_scope_service: Library conversations page, Console resume tree, Home snapshot. (d) The citation subsystem (14 modules, about 15.5k lines). It is imported at boot by app.py and at first paint by chat_screen and console_chat_store, although canonical citation writes are OFF by default. It is also used per Console visit by the citation-count discovery worker. (e) Chat_Functions.chat_api_call, once per send in a worker; a light dispatcher that is fine at runtime.

The biggest costs are storage-admission and connection overheads, not SQL. I measured all of them in an isolated scratch profile. Every depth-0 CharactersRAGDB.transaction() pays a Backup_Recovery storage-admission handshake of about 7 ms (about 245 open() syscalls). A bare BEGIN/COMMIT costs about 0 ms, so a Console message write costs 8-11 ms, and 2-3 of those run on the event loop per send. Every new SQLite connection spawns a Python helper subprocess, about 65 ms. The chat scope service retires its worker connection after every call, so each Library conversations page, Console resume and Home snapshot pays that again: 56 ms vs 7 ms per list call. Chat-source guarded reads cost about 37 ms vs 0.8 ms raw. The citation pydantic cluster adds 45-60 ms to boot and first paint, for a feature that is off by default; pydantic defer_build alone halves it.

Other per-send costs: history images are base64-encoded again and the fingerprint json-digests the multi-MB data URLs, about 70 ms per send for 5x1.5 MB of history. The citation-count discovery performs one 8 ms transaction per assistant message (827 ms per 100 messages on each visit) for a result that is always UNVERIFIABLE while writes are disabled.

Path-depth caveat: the scratch HOME is about twice as deep as a production profile path, so the admission numbers could be roughly half in production. The helper-subprocess cost does not depend on path depth. Several root causes sit just outside the slice (DB/ChaChaNotes_DB.py TransactionContextManager, DB/private_sqlite.py helper, Backup_Recovery/storage_admission.py). They are reported here because they multiply this slice's hot paths and I measured them from this slice's callers; the DB/Backup_Recovery slice agent may double-report them.

Structural notes: Chat/__init__.py is an eager package init. Chat_Functions is a 2.5k-line god module whose provider table eagerly imports every provider module, and 5 boot-loaded modules import it. The 3,859-line chat_persistence_service is fully synchronous and most of its callers are on the loop; only 4 sites go through the controller's _run_durable_db_call offload. save_history has zero production callers (dead code).

## clean areas
- tldw_chatbook/Chat/chat_conversation_service.py list_conversations / locate_conversation_page / _normalize_conversation_rows: batched message counts and keywords (no N+1); get_conversation_appearances is one batched SELECT; get_conversation_tree is one query plus an iterative build (TASK-22206 fix holds)
- tldw_chatbook/Chat/chat_conversation_scope_service.py list_conversations/locate/get_conversation_tree correctly use asyncio.to_thread (task-283). Only the per-call connection retirement is a problem (F2)
- tldw_chatbook/Chat/Chat_Functions.py chat_api_call: single-pass dispatch; log/metric calls once per send are negligible; runs in a worker via console_provider_gateway._chat_api_call; the legacy chat() path (CCP/media analysis) is cold
- tldw_chatbook/Chat/chat_persistence_service.py commit_durable_turn / promote paths: offloaded by the controller via _run_durable_db_call (TASK-22205); get_message_versions is batched (TASK-32804.12 fix holds)
- tldw_chatbook/Chat/citation_trace_repository.py create_local_trace_builder and the whole CitationTraceBuilder/seal path: gated on local_citation_writes_ready, so zero runtime cost with the default config; the keyring provider is never touched when writes are disabled
- tldw_chatbook/Chat/citation_artifact_ownership.py reconcile_pending: gated on writes_enabled and run via asyncio.to_thread in app._reconcile_citation_artifact_ownership
- tldw_chatbook/Chat/citation_payload_lifecycle.py: TypeAdapter(...) constructed per call (lines 253, 559, 563, 1238), but only on the collect/revoke GC path, which requires canonical writes. Cold, noted only
- tldw_chatbook/Chat/citation_repair.py, answer_citations.py, citation_trace_models.py marker regexes: module-level compiled regexes; called once per RAG answer, not per chunk
- tldw_chatbook/Chat/citation_source_locators.py, citation_evidence_models.py, citation_trace_identity.py, citation_trace_adapters.py, citation_provenance_runtime.py, citation_service_factory.py: pure models/transforms; runtime cost gated. Only the import cost matters (F3)
- tldw_chatbook/Chat/character_expression_playback.py: decode is serialized and runs off-thread (the avatar widget uses asyncio.to_thread for decode and mosaic render). PreparedExpression.frame_at rebuilds a <=512-entry boundaries list per tick, a microsecond-scale micro-cost
- tldw_chatbook/Chat/console_activity_receipts.py while-True loop: a bounded cursor-paging loop with a non-advance guard, not a busy loop
- tldw_chatbook/Chat/chat_handoff_messages.py, chat_handoff_models.py, chat_models.py, chat_loop_scope_service.py (server passthrough), Chat_Deps.py, assistant_generation_state.py: no hot-path cost
- tldw_chatbook/Chat/attachment_core.py config readers: the TASK-32804.12 load_chat_images_config single-fetch fix holds

## census
| Measurement (isolated scratch profile, Py3.12, median) | Cost |
|---|---|
| CharactersRAGDB.transaction() empty, depth 0 | 7.0-7.2 ms (~245 posix.open) |
| bare sqlite3 BEGIN IMMEDIATE/SELECT/COMMIT | ~0.0 ms |
| ChatPersistenceService.update_message_content (citation repo wired, writes off) | 10.85 ms |
| same, no citation repo | 8.08 ms |
| ChatPersistenceService.create_message | 8.42 ms |
| get_message_by_id / _without_blob (no txn) | 0.045-0.048 ms |
| new-thread CharactersRAGDB.get_connection (helper subprocess) | 64-71 ms |
| list_conversations on pooled thread, close after each call (scope-service pattern) | 56.5 ms |
| list_conversations on pooled thread, connection kept | 6.9 ms |
| guarded ChatConversationService.get_messages_with_context(limit=100) (bound, production wiring) | 37.3 ms |
| guarded get_citations (no citations exist) | 36.9 ms |
| raw db.get_messages_for_conversation(limit=100) | 0.78 ms |
| CitationLegacyMigrationService.get_journal (guarded) | 23.8 ms |
| citation count lookups, 100 assistant msgs, writes disabled | 827 ms (8.3 ms/msg) |
| active_owner_candidate_message_ids batched (100 ids) | 7.4 ms |
| citation cluster incremental import (5 runs) | 45-61 ms |
| same with pydantic defer_build | 22.5 ms |
| payload build, 5x1.5 MB history images (image_url_part) | 8.5 ms |
| fingerprint_payload over that payload | 22.6 ms |
| base64 data URL: 1 / 5 / 10 MB | 1.1 / 5.7 / 11.4 ms |
| ConsoleActivityReceiptService.publish_ordinary / hydrate | 9.1 / 15.8 ms |


# slice-70

## summary
Slice #70 (HOT-MISC-1, 70 files, ~32k lines) audited at 840ed2ca58. I read every file at least briefly and read the hot code in full. Where timing was possible I measured with isolated micro-benchmarks: HOME, XDG_* and TLDW_CONFIG_PATH were all pointed at scratch, and nothing touched the real profile.

Hot paths in scope:
(1) MainNavigationBar. It is composed by every BaseAppScreen, so it mounts on every visit to a non-reusable screen, on every whole-screen recompose, and when the overflow menu opens.
(2) The Terminal keystroke and output path. The UI side is Console_Modules/terminal.py; the model side is session_manager, screen_model and io_actors, run by per-session runtime, input and monitor threads.
(3) Server-mode `RuntimeServerContextProvider.build_client()` / `get_active_context()`. There are about 52 service call sites, and many run inside async methods on the loop.
(4) The Buddy app-level polls: speech at 1 Hz and scope reconcile at 2 Hz.
(5) The Schedules SSE EventObserver, which runs as a loop coroutine.
(6) The vLLM form and Prompt block editor keystroke handlers.

Largest in-slice costs:
- **Terminal snapshot (F1).** `TerminalScreenModel.snapshot()` re-projects every cell of every session, and 58% of that time is in `_safe_style`. Measured cost is 12.5–19.4 ms per session at 200–235 columns. It runs on the UI thread at least twice per terminal keystroke, because `send_key` calls `_selected_session_id` which builds a full `view_state`. During output it runs once per frame.
- **Terminal output throughput (F2).** The same projection runs for every scrolled line during feed, which caps output at 0.19 MB/s (measured) while holding the GIL and model_lock.
- **Server-mode target read (F5).** In server mode, every server API call re-reads mcp_server_targets.json through the hardened participant reader, at 7.6 ms per call (measured).

Cross-slice discovery (F6): every core-repository `connection()`/`transaction()` now calls `storage_admission.acquire_storage` on each operation. That costs about 245 `open()` syscalls and 4–7 ms, against 0.02 ms for a raw query, measured on ChaChaNotes and ClientNotificationsDB. The admission layer landed on 2026-09-16 (TASK-32628). TASK-31502 measured a per-transaction block at 23 µs on 2026-09-04, so this is roughly a 200× regression on the hottest layer. F7 and F8 are in-slice call sites that pay it on the event loop.

Structural notes:
- The terminal's visible-refresh coalescing (`refresh_requested` / `acknowledge_visible_refresh`) is dead code. As a result, every parser turn, including turns for hidden sessions, makes a blocking `call_from_thread`.
- The per-session runtime and input threads poll at 200 Hz and 100 Hz on top of the known 50 Hz monitor (TASK-31503).
- The nav bar's 2 Hz interval keeps firing on the suspended reusable Home, Console and Library screens, because Textual does not pause timers on suspend.

Overall health: good discipline elsewhere. Writing_Modules offloads sync backends with `to_thread`. The vLLM profile, preflight and probe paths are off-loop. The screen registry is lazy. The Navigation stores are in-memory. The Prompt editor's pure state rebuild is cheap (measured).

## clean areas
- tldw_chatbook/UI/Navigation/screen_registry.py - lazy import_module per route (verified-fine list); only home/chat/library are reusable
- tldw_chatbook/UI/Navigation/screen_state_store.py - in-memory dict, shallow copies only
- tldw_chatbook/UI/Navigation/pending_handoff_store.py - in-memory slots; deepcopy only on stage/claim (rare)
- tldw_chatbook/UI/Navigation/shell_destinations.py, shortcut_context.py, nav_overflow_menu.py (except the attention recompute in F7), __init__.py - static tables/models
- tldw_chatbook/UI/Navigation/character_conversation_navigation.py, _character_conversation_wire.py, conversation_settings_navigation.py, vllm_handoff.py - validation models and modals, no I/O
- tldw_chatbook/UI/Navigation/audio_cpp_model_handoff.py - service/lease I/O via asyncio.to_thread
- tldw_chatbook/UI/Navigation/buddy_management.py - modal/apply I/O is off-loop via to_thread (sequential per-session to_thread hops in _target_choices are a minor P3, modal-open only); only the 2 Hz timer is flagged (F9)
- tldw_chatbook/UI/Navigation/buddy_conversation.py - record reads via to_thread
- tldw_chatbook/UI/Navigation/base_app_screen.py - recompose guards are cheap; its cost is the nav bar it composes (F7/F10)
- tldw_chatbook/Terminal/__init__.py, backend.py, contracts.py, launch.py, posix_launcher.py - lazy/pure; pyte and the backend are loaded only on first session; create_session runs via to_thread; cleanup polling runs on the executor, not the loop
- tldw_chatbook/Terminal/protocol_gate.py - per-byte Python state machine at 127 ns/byte (measured); small next to F2
- tldw_chatbook/runtime_policy/engine.py, enforcement.py, types.py, source_state.py, recovery.py, unsupported_capabilities.py, domain_edge_contracts.py, server_parity_models.py, server_parity_state.py, server_event_scope.py - PolicyEngine.evaluate is microseconds; registry import self-time about 2.5 ms at boot (acceptable)
- tldw_chatbook/runtime_policy/bootstrap.py - LegacyConfigServerClientProvider caches its client; commit_state writes only on source/probe changes
- tldw_chatbook/runtime_policy/server_credentials.py - keyring reads cached (TASK-32922); see the TTL caveat in F5
- tldw_chatbook/runtime_policy/server_capabilities.py - refresh() has no production caller (dead path, not a cost)
- tldw_chatbook/Notifications/client_notifications_db.py, client_notifications_service.py, notification_dispatch_service.py - queries are indexed and LIMITed; only the cross-slice per-op admission tax applies (F6)
- tldw_chatbook/Notifications/__init__.py - PEP 562 lazy exports
- tldw_chatbook/Notifications/server_notifications_service.py, notifications_scope_service.py, server_notifications_scope_service.py, notification_presentation.py, recovery.py, local_event_producer.py, event_cursor_store.py - thin wrappers; the per-call cost is build_client (F5)
- tldw_chatbook/UI/LLM_Management/vllm_setup.py, vllm_profiles.py, vllm_connection.py - preflight, profile repository and probes run via asyncio.to_thread or async httpx with bounded timeouts; the per-probe AsyncClient is user-initiated only
- tldw_chatbook/Widgets/Prompts/prompt_block_editor_state.py - update_block per keystroke measured at 0.031 ms (12 blocks x 2 KB) and 0.176 ms (24 blocks x 8 KB)
- tldw_chatbook/Widgets/Prompts/prompt_block_editor.py - per-keystroke _sync_card plus _sync_footer (~8 query_one, guarded set_options/tooltip); the out-of-slice library preview load_text mirror is not counted here
- tldw_chatbook/UI/Writing_Modules/writing_controller.py - the model pattern: sync backends dispatched with asyncio.to_thread
- tldw_chatbook/Widgets/Evals/__init__.py, UI/LLM_Management/__init__.py, Widgets/Prompts/__init__.py, UI/Writing_Modules/__init__.py - trivial

## census
| probe (isolated, scratch profile) | result |
|---|---|
| TerminalScreenModel.snapshot 80x24 (2k scrollback) | 3.30 ms |
| snapshot 200x50 blank screen | 12.53 ms |
| snapshot 200x50 full / 235x50 full | 14.83 / 19.37 ms |
| snapshot hotspot | _safe_style 118,310 calls per 10 snapshots = 58% of time |
| TerminalScreenModel.feed 1.15 MB of ls-style output @200 cols | 5.97 s (0.19 MB/s); 80% in _retain_scrollback_line/_project_line |
| TerminalProtocolGate.feed | 127 ns/byte |
| idle runtime-bridge poll replica (5 ms) / input-flush poll (10 ms) | 0.34% / 0.16% of a core per session (lower bound) |
| ConfiguredServerTargetStore.get_target (per build_client) | 7.59 ms, 556 open() calls per call |
| ChaChaNotes empty read transaction() | 6.67 ms (vs execute_query SELECT 1: 0.02 ms) |
| list_console_unseen_marks, 0 rows / 2000 rows | 4.52 / 3.89 ms (admission-dominated; plan: SCAN covering idx + TEMP B-TREE) |
| ClientNotificationsDB.list_notifications(limit=20) / get_settings | 6.78 / 6.68 ms |
| prompt block update_block per keystroke | 0.031-0.176 ms |

Note: the scratch profile path has 14 components against about 6 for ~/.local/share/tldw_cli/default_user. Private-path walks scale with depth, so the default-profile cost is estimated at roughly 40-60% of the admission-dominated rows above.


# slice-71

## summary
Slice #71 (85 files, ~32k LOC). Hot paths found: (a) Console task-surface cards (approval/question/skill/worktree/chat-create/watchlists-op), which are event-driven with round-identity guards and are clean; (b) the app-level TTS progress/complete handlers, which still import and query the dead legacy ChatMessage/ChatMessageEnhanced widgets on every event (F4); (c) the Home active-work adapter, which is a sync API that cold-computes 3 seam reads when its 3 s TTL cache is stale. It is called on the event loop by Schedules and Watchlists, not only by Home (F5). (d) Per-keystroke paths: Research sources search is P1, measured at ~32 ms per keystroke with no debounce and no unchanged-text short-circuit (F1). The Workflows raw-JSON editor rebuilds two OptionLists on every keystroke (F7). The Chatbook wizard SmartContentTree filter posts one NodeExpanded per node and hides nothing, measured at ~40 ms per keystroke (F3). Smaller per-keystroke costs: review editor, chapter title, quick notes. (e) TrajectoryTimeline.render has an O(agent_boundaries x records) lookup, measured at 72 ms per render at 3k records, and it re-renders on drag, zoom, pan and 0.5 s live-follow snapshots (F2). (f) Opt-in Tamagotchi does 30 s JSON read/backup/write on the loop.

Verified cheap:
- The Personas 4 Hz handoff-readiness poll (personas_preview_controller.console_handoff_readiness) measured 26 us per tick with an in-memory config. It is not a TASK-32804.3-style problem.
- The Web_Server share start/stop paths are threaded.
- The CCP character/persona handlers already use to_thread or thread workers.
- The Persona conversations controller is paginated and off-thread.

Structural notes:
- Dead code in the slice that is never imported in production: Widgets/Media (4 files, ~3.1k LOC), Chat_Widgets/chat_shell_bar.py, chat_handoff_card.py and Tamagotchi/examples. config_search_widget is reachable only from the deprecated Tools_Settings_Window.
- ChatMessage is dead yet imported at boot (app.py:502).
- Home.dashboard_state claims shell_destinations is a leaf, but importing it runs UI/Navigation/__init__, which pulls the whole app graph (~936 modules). There is no boot delta today.

Scratch probes are in scratchpad/audit/scratch/slice71/ and all ran under full HOME/XDG/TLDW_CONFIG_PATH isolation. The headless render numbers include a ~22-32 ms pilot.pause baseline, which was subtracted in the costs quoted.

## clean areas
- tldw_chatbook/Widgets/Chat_Widgets/chat_approval_card.py: set_batch has an unchanged-round guard, the 1 Hz deadline timer runs only while armed, and arg summarization is ~1.5 ms even for 1 MB args
- tldw_chatbook/Widgets/Chat_Widgets/chat_task_cards.py: sync_state is event-driven (per approval, question or skill round) and each card guards its own round identity; the watchlists-op follow poll is 2 s, off-thread and stops when there are no active ops
- tldw_chatbook/Widgets/Chat_Widgets/{chat_question_card,skill_install_confirm_card,skill_script_confirm_card,worktree_confirm_card,worktree_recovery_dialog,chat_create_confirm_card,watchlists_operation_card,chat_resume_panel}.py: small and lazily mounted, with 1 Hz timers only while a question is pending
- tldw_chatbook/Widgets/Chat_Widgets/{chat_shell_bar,chat_handoff_card}.py: dead (no production importers), so no runtime cost
- tldw_chatbook/Widgets/Media/* (media_viewer_panel, media_list_panel, media_search_panel, media_navigation_panel): dead package with no importers, so no runtime cost; delete candidate
- tldw_chatbook/UI/CCP_Modules/*: character list and loads use asyncio.to_thread or thread workers with generation guards; the eager __init__ adds nothing beyond what personas_screen already imports
- tldw_chatbook/UI/Persona_Modules/personas_preview_controller.py: the 4 Hz handoff-readiness poll body measured 26 us/tick; provider_readout measured 41 us
- tldw_chatbook/UI/Persona_Modules/personas_conversations_controller.py: paginated with a sentinel row, thread workers and to_thread link re-fence
- tldw_chatbook/UI/Persona_Modules/{buddy_conversion,personas_preview_coordinator}.py: I/O runs through to_thread
- tldw_chatbook/Home/dashboard_state.py and home_rail_state.py: pure state over bounded item lists (only the import-layering note, F12)
- tldw_chatbook/Web_Server/artifact_share.py, artifact_share_server.py, artifact_share_manifest.py: start/stop runs in thread workers or to_thread and the server is a subprocess; the textual_serve availability check costs ~2 ms
- tldw_chatbook/Web_Server/serve.py: separate web-serve process; the 4 Hz canvas-policy poll hits the warm config cache fast path; the patched JS is cached by mtime
- tldw_chatbook/UI/Workflows_Modules/{controller,library,navigator,reference_picker,console_context}.py: DB reads are to_thread, the page search is debounced, console_context is a thread worker, and draft persistence is debounced (only the per-keystroke rebuilds in F7/F8)
- tldw_chatbook/UI/Research_Workspace_Modules/{source_receipt,add_source_modal,source_inspector,header_region,mode_bar,pane_handle,workspace_menu,chat_region,studio_region,overlay_conflict_modal,quick_note_modals}.py: slot pools, awaited callbacks, modals
- tldw_chatbook/UI/Widgets/{table_click_select,trace_filter_bar}.py and the lazy PEP 562 __init__.py: cheap
- tldw_chatbook/Widgets/TTS/character_voice_widget.py: no recompose, and regex detection is on explicit action only
- tldw_chatbook/Widgets/Tamagotchi/__init__.py: already lazy; the whole package is off by default ([tamagotchi] enabled=false)
- tldw_chatbook/UI/Subscription_Modules/__init__.py

## census
| probe (isolated env, audit tree 840ed2ca58) | result |
|---|---|
| Research ResearchSourceList.sync_page(25 rows), headless 211x60 | sync 5.9-6.1 ms; with relayout/paint 58-60 ms vs 22-32 ms pause baseline (+~26 ms); unchanged content costs the same (59 ms) |
| TrajectoryTimeline 4-lane render, 200-col plot | n=200/55 agent boundaries 0.9 ms; n=1000/204: 7.0 ms; n=3000/970: 71.9 ms (71.2 ms is lane 3 alone) |
| SmartContentTree._apply_filters, 700 items | 3.6 ms sync + ~42 ms drain/render per keystroke; tree shows 707 lines with both '' and 'zzz' filters, so it hides nothing |
| app.query(ChatMessage) on a 540-widget screen | 1.15 ms per query (2 per TTSProgressEvent) |
| first import of chat_message_enhanced (PIL + rich_pixels + textual_image) | 13-15 ms warm cache |
| Personas console_handoff_readiness tick (4 Hz) | 26 us |
| Home.dashboard_state standalone import | 594 ms cold, 936 modules (via UI.Navigation __init__ → ACP_Interop → Chat → config) |
| chapter word counts, 30 chapters / 120k words | 1.9 ms per title keystroke (plus a full DataTable rebuild) |


# slice-72

## summary
Slice #72 (57 files, ~11.1k lines). Most of it is cheap, or never imported in production.

**Live hot paths in the slice**
- Home rail and canvas: the Home screen at boot and on every triage sync.
- Workbench header and strips: DestinationHeader.sync_state runs on every Console keystroke.
- Study flashcards and quizzes controllers: the Study screen, deck and quiz changes, and every review rating.
- Chunking Lab regions: fed on every keystroke in the sample, editor and save dialog.
- ModelArtifacts progress: every install-progress event.
- Lab rail store: rail toggle clicks on the LLM, Speech and Evals screens.
- RAGSearch/search_handoff: Console library-RAG turns and handoffs.
- Writing widgets: project, outline and scene selection.
- NoteSelectionDialog: Speech Studio's "import from notes".

**Overall health**
Home and Workbench are already incremental and equality-gated (task-282, task-15452), so they are clean.

**Findings (6 at P2, 3 at P3, no P0 or P1)**
- **Mounting.** Three sites append up to 100 ListItems one `await ListView.append` at a time (Study flashcards, Study quizzes, Writing projects). Measured 176-185 ms against 36-65 ms for a single `extend`.
- **Sync SQLite on the event loop.** Local-mode Study calls run SQLite on the loop through `run_worker(coro)` plus `_maybe_await`. A rating costs 7.6 ms for its write transaction plus 1 ms for the next-card query. Listing 100 cards costs 3.8 ms. This is known under TASK-32898; the per-rating write is new evidence.
- **Chunking Lab re-rendering.** ResultsRegion clears and repopulates both 100-row DataTables 4-5 times per edit batch, even when nothing changed. Measured about 3.5 ms sync plus about 12 ms extra repaint, per keystroke once results exist.
- **NoteSelectionDialog.** Mounts about 7 widgets per note. At 100 notes that is 712 widgets and 279 ms to push. Select-all is O(N²) in `query_one`: about 10k lookups, 140 ms.
- **Install progress floods.** Install progress posts one UI message per MiB with no coalescing, at 241 µs per event. The 1 MiB pre-verify hash step can emit 1000-2000 events per second, which ties up about 25-50 % of the event loop (estimated from the per-event cost).
- **Lab rail toggle writes config on the loop.** It rewrites config.toml synchronously. The Home rail already does the same save in a `@work(thread=True)` worker; the Lab rail does not.
- **P3 items:**
  - The Writing detail panel reloads the scene body into its TextArea 3 times per selection.
  - The Study shell refreshes its summary widgets 3-5 times per action with no check for unchanged values.
  - search_handoff runs the text validator on every field several times per result, and builds one evidence bundle that is thrown away.

**Structural notes (no runtime cost, so not filed as findings)**
- These have no production importers and only tests import them: Widgets/NewIngest (5 files, about 1.2k lines), Widgets/Coding_Widgets/repo_tree_widgets.py (726 lines), Workbench's WorkbenchFrame/WorkbenchPane/StateBlock, and state/{app,chat,notes,navigation}_state. The state package is a lazy facade (PEP 562), so only ui_state loads.
- flashcards_handler and quizzes_handler each carry a full copy of about 10 scope helpers. That duplication belongs to the scope-service scaffold task (TASK-32898 / TASK-32808.6).

**Outside the slice but worth routing**
- `get_due_flashcards` orders by `datetime(f.next_review)`, which no index can serve. Belongs to the DB slice.
- Server-mode `list_decks` pages through the entire deck collection. Belongs to the Study_Interop slice.
- llm_screen's own progress `deliver` has the same unthrottled shape as `make_progress_callback`.

All benchmarks ran in isolated scratch HOME/XDG/TLDW_CONFIG_PATH. `save_setting_to_cli_config` and `apply_settings_mutation_to_cli_config` were not run.

## clean areas
- tldw_chatbook/Widgets/Home/home_canvas.py + home_rail.py: sync_state already patches in place, gated on equality; rows capped by HOME_RECENT_WORK_LIMIT. Clean.
- tldw_chatbook/UI/Workbench/workbench_widgets.py: DestinationHeader/CommandStrip/ModeStrip/RecoveryCallout are equality-gated (task-15452) and sort only when out of order. WorkbenchPane/StateBlock/WorkbenchFrame are ungated but have no production consumer. Clean.
- tldw_chatbook/UI/Workbench/__init__.py (PEP 562 lazy), workbench_state.py (re.sub per id construction, negligible), focus.py, help.py: clean
- tldw_chatbook/UI/Lab_Modules/lab_speech_status.py: 9 find_spec probes per Speech visit, measured 1.0 ms cold / 0.4 ms warm. Clean.
- tldw_chatbook/UI/Lab_Modules/lab_server_status.py, lab_rail_layout.py, lab_workbench.py (display toggles only): clean
- tldw_chatbook/UI/Chunking_Lab_Modules/sample_region.py + editor_region.py: TextArea.text join on each keystroke measured 0.09 ms at 2 MB, negligible. File reads are bounded and run via asyncio.to_thread in the screen.
- tldw_chatbook/UI/Chunking_Lab_Modules/dialogs.py: TemplateDialog filters in memory on each keystroke over a small template list. Clean apart from on_edit feeding F3.
- tldw_chatbook/UI/Chunking_Lab_Modules/results_region.py: _prepare and inspect run off the loop (asyncio.to_thread) and tables are paged at 100. Only the redundant re-render is flagged (F3).
- tldw_chatbook/Widgets/ModelArtifacts/{activation_controls,install_modal,plan_panel,runtime_choice_modal,local_gguf_import}.py: consent/intent only, no I/O. Eager __init__ is harmless because app.py already imports Model_Artifacts at boot (app.py:384).
- tldw_chatbook/UI/Research_Modules/research_controller.py + bundle_rendering.py: ResearchScopeService already sends local backend calls to a dedicated thread (_run_on_backend_thread); this is the template the Study controllers should follow. Clean.
- tldw_chatbook/Widgets/Writing/writing_outline_tree.py: two walks of the structure plus a Tree rebuild per project load. Tree is virtualized. Clean.
- tldw_chatbook/Widgets/Note_Widgets/note_creation_modal.py: one query_one per keystroke to clear the error text. Negligible.
- tldw_chatbook/Widgets/Study/study_dashboard.py + quiz_session_widget.py: small fixed widget set. Only the ungated updates are noted (F8).
- tldw_chatbook/state/*: lazy facade (PEP 562); only ui_state.UIState (a plain dataclass) is used, by chat_screen. Clean.
- tldw_chatbook/UI/Views/RAGSearch/__init__.py: docstring only
- tldw_chatbook/Widgets/NewIngest/* and tldw_chatbook/Widgets/Coding_Widgets/repo_tree_widgets.py: no production references (rg count 0 outside their own directories), imported only by tests. No runtime cost; deletion candidates, not perf findings.

## census
| Measurement (isolated scratch profile, Textual 8.2.8, Py3.12) | Result |
|---|---|
| ListView: 100 x `await append` vs 1 x `extend` | 176-185 ms vs 36-65 ms |
| NoteSelectionDialog push with 100 notes | 279 ms, 712 widgets |
| NoteSelectionDialog select_all (100 notes) | 140 ms, 101 update_selection_count calls (~10.1k query_one) |
| Chunking Lab: 5 DataTable rebuilds x 100 rows (one edit batch) | 3.4-4.9 ms sync + ~12 ms extra settle vs no-rebuild baseline |
| ModelInstallProgress x2: per InstallProgressed event | 241 us/event (1000 events = 241 ms loop) |
| LocalStudyService.submit_flashcard_review (3000-card deck, WAL file DB) | 7.56 ms median |
| LocalStudyService.get_next_review_candidate | 0.95 ms median |
| LocalStudyService.list_flashcards(limit=100) | 3.76 ms median |
| TextArea.text = 104 KB body: 1 vs 3 assignments | 19 ms vs 36-37 ms sync |
| build_library_rag_evidence_bundle 10 x 4000-char / 20 x 4000-char | 2.04 ms / 4.30 ms |
| sanitize_string(4000 chars) | 105 us/call |
| tomllib.loads(102 KB default config) / toml.dumps | 3.8 ms / 1.0 ms |
| speech find_spec probes (9 modules) | 1.0 ms cold / 0.41 ms warm |
| TextArea Document.text join at 2 MB | 0.09 ms |


# slice-73

## summary
Slice #73 (cold tier, 174 files, ~60k lines). The slice does little work at idle and has no always-on timers or threads. Its real costs leak into three hot paths: every Console send, boot/first paint, and Speech/TTS settings saves.

(1) Every send, and every Console mount/resume, calls `capture_skill_context_maximum` / `get_context` synchronously on the event loop. For each installed skill that path runs `status_for_skill` plus `current_fingerprint_digest`, and each call wraps itself in Backup_Recovery storage admission: about 9 `acquire_storage` and 12 `operation` scopes per skill, roughly 2,600 `open()` syscalls. Measured in an isolated profile: 219 ms for 5 skills and 833 ms for 20 skills (no trust manifest); 311 ms for 5, 1.22 s for 20 and 3.29 s for 50 (trust bootstrapped). The actual file scan is 0.24 ms per skill. TASK-32921 (Done) fixed only the keyring part of this path and explicitly left the rest.

(2) The boot splash is on for 7 s by default. At first mount it imports all 87 effect modules (about 55 ms, warm) to use one. 70 of the 87 effects emit per-cell Rich markup, which `Static.update` parses synchronously on every frame. At 235x52, digital_rain, sound_bars and spotlight_reveal cost 150-275 ms per 50 ms tick, and zen_garden about 53 ms; digital_rain, sound_bars, spotlight_reveal and zen_garden are in the shipped active_cards list. So about 11% of boots saturate the loop for the whole splash, while the app is also trying to pre-import screens in parallel.

(3) The STT and Model_Artifacts package `__init__`s import eagerly, and `app.py` imports STT, dispatch and store modules at module scope. Together they add about 20-28 ms to boot.

(4) Every managed-artifact lease (`acquire_installed_root` / `acquire_dependencies`) re-computes a full SHA-256 of payloads that are 1-19 GB. This happens on Speech/TTS Save, Revert, Restore defaults and Discard-on-leave, and on TTS profile create/update/assign/delete. It is threaded, but it takes seconds to tens of seconds. An install also hashes the payload two or three times.

Cold-path items: deep-search citation fuzzy matching (1.3 s per unmatched quote; the SequenceMatcher cache is defeated and each quote is matched twice); the web-search relevance loop is fully serial with deliberate sleeps; DuckDuckGo pagination opens a new connection per page; Evaluations_Interop services are built at boot with no consumers; Article_Extractor launches a browser per URL and keeps per-thread event loops.

Out-of-scope correctness notes worth filing separately:
- `Article_Extractor_Lib.load_and_log_configs()` returns `{}`, so `fetch_html` raises `KeyError('web_scraper')` before any Playwright launch. Article URL scraping appears broken.
- `Web_Scraping/Article_Scraper/__init__.py` is empty, so `generic_scraper`'s import of `Scraper` always fails and Article_Scraper is dead code.
- TASK-31001 (broken splash cards) is still open.

Structurally, the dominant lever outside this slice is the per-call cost of Backup_Recovery admission, about 3.7 ms per `acquire_storage`. TASK-32562's log saw the same thing (500 path admissions took 1.14 s). Any per-item wrapper multiplies it.

## clean areas
- tldw_chatbook/STT/executor.py, executor_worker.py, executor_process_tree.py, coordinator.py, dispatch_coordinator.py: model work runs in a spawned worker process; the reader thread blocks on recv (no polling); the 10 ms process-tree polls run only during bounded teardown; one handoff thread per terminal dictation event (negligible); the executor is built lazily (app._ensure_local_stt_executor)
- tldw_chatbook/STT/parakeet_onnx.py, transcribe_cpp.py: ffmpeg subprocesses run inside the worker process with timeouts; audio buffers are bounded by MAX_BUFFER_AUDIO_BYTES
- tldw_chatbook/STT/parakeet_external.py: verification cache keyed on stat (ino/mtime_ns). This is the pattern Model_Artifacts should reuse
- tldw_chatbook/STT/parakeet_sources.py, registry.py, routing.py, legacy_bridge.py, persistence.py: cold, pure logic (import cost covered in F4); the per-reuse re-verify of the Parakeet VAD dependency touches only about 2 MB
- tldw_chatbook/Model_Artifacts/machine_memory_probe.py: @work(thread=True), runs once per session (llm_screen._run_machine_memory_probe)
- tldw_chatbook/Model_Artifacts/acquisition.py, fetch.py, remote_huggingface.py: one AsyncClient reused per artifact, streamed in 1 MiB chunks, driven via asyncio.run inside @work(thread=True) (llm_screen); only the post-download re-hash is an issue (F6)
- tldw_chatbook/Model_Artifacts/leases.py: NON_BLOCKING portalocker with a 50 ms backoff only under contention
- tldw_chatbook/Model_Artifacts/service.py list_installed / disk_usage: stat and JSON only, no hashing; reconcile() full-hashes, but it is an explicit repair action in a thread worker
- tldw_chatbook/Model_Artifacts/gguf_admission.py, _deferred_gguf_managed_import.py, curated_registry.py, recovery.py, maintenance.py: bounded header reads, cold
- tldw_chatbook/Skills_Interop/__init__.py: PEP 562 lazy exports
- tldw_chatbook/Skills_Interop/project_skills_prompt.py + project_skills_discovery.py: startup discovery runs in a @thread worker (app._discover_project_skills_for_startup)
- tldw_chatbook/Skills_Interop/skill_trust_crypto.py: scrypt KDF is reached only through asyncio.to_thread (library_screen._call_library_skill_trust_service)
- tldw_chatbook/Skills_Interop/skill_script_runner.py: runs through asyncio.to_thread; its 50 Hz wait poll runs only while a script is running (P3, not filed)
- tldw_chatbook/Skills_Interop/skill_remote_fetch.py, server_skills_service.py, skill_package_inspection.py, atomic_write.py, recovery*.py: cold
- tldw_chatbook/Web_Scraping/search_backend_settings.py: cheap static catalog
- tldw_chatbook/Web_Scraping/Article_Extractor_Lib.py imports: playwright, trafilatura and pandas are deferred behind find_spec; Tools/__init__ makes WebSearchTool lazy
- tldw_chatbook/Web_Scraping/WebSearch_APIs.py: LLM calls run in asyncio.to_thread with wait_for timeouts; imported lazily by web_tool_impls and the research engine; config reads are cache-backed
- tldw_chatbook/Web_Scraping/Confluence/*, cookie_scraping/cookie_cloner.py, Article_Scraper/*: unreachable from the app (dead code, no runtime cost)
- tldw_chatbook/Evaluations_Interop/*: thin wrappers with under 1 ms import self-time; LocalEvaluationsService.list_runs has a 1+2N query pattern (_enrich_run: get_model + get_run_metrics per run), but it has no production caller
- tldw_chatbook/Utils/Splash_Screens/card_definitions.py: get_all_card_definitions() takes 0.02 ms; the splash timer is gated on screen.is_active and stopped on close, repaints with layout=False, and skips identical frames
- tldw_chatbook/Config_Files/create_custom_template.py: standalone CLI script, not imported by the app; tldw_chatbook/assets/__init__.py is empty

## census
Splash effects at 235x52, measured with isolated Python 3.12 / Textual 8.2.8. Cost is effect.update() plus Content.from_markup, which Static.update runs synchronously on the loop. Render/compositor cost is excluded. A * marks cards in the shipped config.py active_cards (35 cards).

| card | interval | update+parse / frame | loop share |
|---|---|---|---|
| * digital_rain | 50 ms | 275 ms | 551% |
| raindrops_pond | 50 ms | 271 ms | 542% |
| data_stream | 20 ms | 108 ms | 540% |
| maze_generator | 10 ms | 54 ms | 536% |
| * sound_bars | 50 ms | 219 ms | 438% |
| * spotlight_reveal | 50 ms | 155 ms | 310% |
| binary_matrix | 50 ms | 72 ms | 144% |
| doom_fire | 50 ms | 57 ms | 113% |
| * zen_garden | 50 ms | 53 ms | 107% |
| * ant_colony | 100 ms | 51 ms (82 ms at 211x44) | 51-82% |
| * game_of_life | 100 ms | 40-47 ms | 40-47% |
| * train_journey | 100 ms | 24 ms | 24% |
| * matrix | 50 ms | 3.3 ms | 7% |

Fix probe: the same 235x52 grid parsed from per-cell markup took 233 ms; built as spans (Content with Span runs) it took 2.3 ms. load_all_effects() imports 87 modules in 54-58 ms (warm).

Per-send skill context capture (`capture_skill_context_maximum`) against installed skills of 5 files / about 20 KB each, isolated scratch profile:

| skills | no trust manifest | trust bootstrapped |
|---|---|---|
| 1 | 32 ms | n/a |
| 5 | 219 ms | 311 ms |
| 20 | 833 ms | 1,224 ms |
| 50 | n/a | 3,292 ms |

Raw `scan_skill_directory` is 0.24 ms per skill; one `acquire_storage` plus close is 3.68 ms. cProfile over 3 captures of 10 skills: 270 `acquire_storage` calls, 360 `operation` scopes, 79,830 `posix.open` calls; 73% of the time is in `acquire_storage`.


# slice-74

## summary
Slice #74 (COLD-MISC-2, 115 files, ~59.9k lines). All reads were in the audit tree at 840ed2ca58. Every probe ran under the isolated scratch HOME/XDG/TLDW_CONFIG_PATH.

Where this cold slice costs time on hot paths:
(1) Boot. app.py imports and constructs every Actor Pack, Writing, Research, Research Workspace and Chatbooks service in TldwCli.__init__. Measured cost is about 50 ms before first paint: 32–39 ms of marginal imports, plus about 15 ms of work in the constructors.
(2) The event loop. The local deep-research engine runs as run_worker(coro), so all of its synchronous SQLite persistence runs on the UI loop. TASK-21127 declined to move it off the loop because it measured 1.6 ms per run. That premise no longer holds: each call now costs about 3 ms, and an 8 MB evidence-pool save blocks the loop for 32–55 ms.
(3) Chatbooks. The import and create wizards run the whole operation on the loop. preview_chatbook extracts the entire archive just to read manifest.json: 1.0 s for 300 members, 3.2 s for 1000.
(4) Research Workspace Sources pane. It fetches every media item's full detail row, including content, one item at a time, and fetches them twice per refresh. Measured about 2.4 s per 25-source refresh, and about 4–5 s per reorder keypress.

A cross-cutting multiplier sits under most of this. The Backup_Recovery storage-admission layer adds about 3 ms and about 245 open() syscalls to every trivial LocalWritingService or LocalResearchService operation. secure_private_directory costs about 5 ms per call. Worker-thread media reads reconnect every time: about 33 ms each to spawn a private-sqlite helper, because run_finite_local_worker retires the connection after each call. That overhead turns every N+1 in this slice into seconds.

Structurally the slice is in good shape. Most features already use asyncio.to_thread, single-thread backend executors, lazy registries and PEP 562 facades. The exceptions are Actor_Packs/__init__, which is eager, and Research_Workspace.contracts, which is pulled onto the boot path for a single enum.

## clean areas
- Image_Generation/*: the adapter registry is lazy (dotted paths). get_image_generation_config is cached behind an RLock. Console /generate-image runs prepare/run_generation via asyncio.to_thread. ComfyUI and FAL poll with sleeps only on worker threads. The per-request httpx clients are acceptable at this frequency.
- Media_Playback/*: preview_policy and availability are pure or find_spec only. AvFrameSource decodes on a Textual worker thread. probe_file, ffmpeg and ffplay run on thread workers (video_player_screen). stream_resolve runs under asyncio.to_thread. Only shutil.which x2 runs on the loop (negligible).
- Tool_Packs/*: this is a good template. ToolPackService composition is deferred until first feature use and runs off-thread (app.py _compose_tool_pack_service_off_thread). Controllers use asyncio.to_thread. The receipt store is bounded by max_total_bytes. Tool_Packs is not on the boot or ui_ready module set.
- Actor_Packs controller.py and import_controller.py: export/import run via asyncio.create_task(asyncio.to_thread(...)). Recovery and the staging sweep moved to deferred startup threads (TASK-21106, TASK-22216 hold). The only issue is the boot import, which is in F1.
- Research_Workspace source_association.py and source_readiness.py: every store call goes through asyncio.to_thread. The startup resume is bounded (limit=50). No polling.
- Writing_Interop and Research_Interop scope services: a single-thread backend executor keeps SQLite off the loop (TASK-21125/21127). _accepted_parameters is lru_cached on the class, not on bound methods. stream_run_events is correctly rerouted to the offloaded list_run_events.
- UI/Research_Window 2 s auto-refresh: gated on a selected LOCAL run that is not in a terminal state, and routed through the offloaded scope service.
- Chatbooks import DB phase: one BEGIN IMMEDIATE per conversation, with nested adds joining it. Archive members are copied in 64 KB chunks. Registry parse is cheap: measured _load_registry 3.3 ms, list_home_artifact_snapshot 6.6 ms and artifact_read_snapshot 8.9 ms for 1000 records / 4.6 MB, all off-loop. ChatbookArtifactSnapshot memoizes rows per scope.
- Persona_Visual first-visit import: after PIL is loaded (it is already loaded by ui_ready), importing authoring/importer/publication/runtime/assets costs about 6.5 ms warm. Not worth deferring.
- Persona_Visual builtin_pixel_migu: runs once per session on the startup/readiness worker.
- Research_Interop academic lanes run concurrently via asyncio.gather plus to_thread. Retry sleeps run on worker threads only.
- Research_Interop research_scope_service maintenance drain: the 10 ms sleep loop only runs during backup/restore quiescence and is bounded by a deadline.

## census
| Probe (isolated env, warm pyc) | Result |
|---|---|
| Marginal import of the 41 slice modules at `import tldw_chatbook.app` (all non-slice deps pre-imported, 3 runs) | 31.7–39.0 ms (Actor_Packs 15.9, Research_Workspace 11.4 [contracts 5.9], Chatbooks 6.7, Research_Interop ~2.9, Writing ~1.6) |
| TldwCli.__init__-time slice imports (after app import) | Persona_Visual.repository 7.7–8.2 ms warm (20.8 ms cold); RW local_adapter+quick_notes 2.0–2.1; server_adapter 0.5–0.6 |
| secure_private_directory steady state (ResearchPasteStagingStore.__init__) | 4.5–5.8 ms/call, about 295 posix.open per call |
| LocalWritingService get_project / list_projects | 3.14 / 2.85 ms (245 posix.open per call) |
| LocalResearchService get_run / holds_lease / update_run_progress / record_run_event | 3.03 / 3.30 / 3.26 / 3.40 ms |
| LocalResearchService launch_run, first use | 133 ms fresh DB, 331–359 ms cold process, 69–95 ms warm app |
| Engine evidence pool at the 8.3 MB cap | _bounded_evidence 10–11 ms + save_artifact 21.5–45 ms; get_artifact read-back alone 10.5 ms |
| ChatbookImporter.preview_chatbook (full extraction) | 50 members 213 ms; 300 members 989 ms; 1000 members 3.22 s; 300 members + 32 MB media 1.03 s; preflight only 6–19 ms |
| MediaReadingScopeService.get_media_detail x25 (200 KB content each) | gathered 551–626 ms; serial 1.86–1.89 s; one worker-thread call 34.7 ms, of which 33 ms is connect_private_sqlite/helper start |


# slice-75

## summary
Slice #75 (COLD-MISC-3, 164 files, ~60k lines) audited at origin/dev 840ed2ca58. Most of this slice is cold feature backends, and they are mostly well-behaved: Persona_Buddy, LLM_Management, Workflows, Petdex and RAG_Admin push file and DB work through asyncio.to_thread or thread workers, keep caches bounded and load PIL/torch lazily. The real costs this slice adds to hot paths come from three places.

(1) Boot import and construct legs. app.py imports css.Themes.themes at module scope, and that module pins AA text colours on all 91 themes at import time (about 20 ms measured, P1). app.py also eagerly imports and constructs 13 interop families. Five of them (Claims, Prompt_Studio, MCP_Governance, Auth_Account, Audio_Services) have zero runtime consumers, the same shape as known TASK-21239. The CLI entry (app_entry) imports OpenTelemetry at module scope, and for every embeddings_rag user it fails part-way through, wasting about 11 ms. Neither the ui_ready census nor the import-weight guard can see this, because both measure `import tldw_chatbook.app` or a Pilot boot, not the real CLI entry.

(2) The model-catalog startup refresh right after first paint. It runs as an async worker on the event loop and does an F_FULLFSYNC durable save on every launch, even when nothing changed (21-33 ms measured). It also builds an httpx AsyncClient for each stale provider on the loop, about 9-10 ms each.

(3) Settings and Study click paths that still do synchronous config writes or SQLite on the loop. The theme "Use" config write measured 46-52 ms per switch. The Study local backend still reads and writes SQLite on the loop because the task-15471 offload only covers a fallback branch that never runs in production.

Known items re-confirmed and not re-filed:
- Kanban_Interop.server_kanban_service still costs 18-20 ms of warm self import time (TASK-21107), and Kanban still has no consumer (TASK-21239).
- Generated-video retention still runs in TldwCli.__init__ (TASK-21109, app.py:7869).
- TieAwareStylesheet reparses are covered by TASK-22505.
- The Internal_Prompts residency is documented at 1-2.4 ms.

Structural note: the pattern of app.py wiring roughly 40 interop scope services at import and construct time goes beyond this slice (Writing, Research, Sharing, ...). The whole family should be audited for consumers, and each one should move to lazy first-use construction.

## clean areas
- tldw_chatbook/Persona_Buddy/* - controller runs blocking work via asyncio.to_thread with cancellation-safe drain loops (not busy-waits); rendering.py loads PIL/rich_pixels lazily behind lru_cache and bounds frames; console_adapter is cheap lock-guarded bookkeeping; library.py listing is paginated (LIMIT <=100); controller import is deferred (TASK-21103)
- tldw_chatbook/LLM_Management/* - snapshot_service._file routes store I/O through asyncio.to_thread; the readiness retry loop is bounded (10 x 0.5 s); no timers at boot
- tldw_chatbook/Workflows/* - draft_session/authoring/session do DB and file work via asyncio.to_thread; the draft debounce uses loop.call_later; local_steps' asyncio.run only runs on worker threads (documented to raise on a loop thread); per-keystroke draft validation is bounded by text admission limits
- tldw_chatbook/Petdex/* - network fetch, zip/folder reads and conversion run off the loop via review.drain_thread/to_thread; reads are byte-capped
- tldw_chatbook/RAG_Admin/* - rag_admin_scope_service offloads thread-safe local backends with to_thread (task-32804.12)
- tldw_chatbook/Internal_Prompts/* - resolver is a cached get_cli_setting plus str work; ui_ready residency is documented and measured at 1-2.4 ms
- tldw_chatbook/UX_Interop/* - PEP 562 lazy facade, pure contract builders
- tldw_chatbook/Stats/user_statistics.py - runs in a thread worker (stats_screen uses call_from_thread); content scans are LIMIT-bounded (50/1000/5000); the remaining O(messages) aggregates (AVG(LENGTH), GROUP BY DATE) run off-loop once per Stats visit
- tldw_chatbook/Third_Party/textual_fspicker/* - threaded os.scandir, off-loop sort/filter via to_thread, batched option projection with sleep(0) yields (only the always-on 33 Hz timer is flagged)
- tldw_chatbook/Third_Party/aider/* - zero consumers and never imported: dead code, but it costs nothing at runtime
- tldw_chatbook/css/build_css.py, css/widget_css.py, css/check_bundle_sync.py, css/Themes/theme_tester.py - the builder and rglob scans only run in dev or rebuild; boot reads just the two small widget_defaults sheets once
- tldw_chatbook/css/tie_aware_stylesheet.py - per-add_source overhead is one dict lookup; reparse cost is known TASK-22505
- tldw_chatbook/Media_Creation/* - the aiohttp session is reused per client; generation runs async; not on the boot import leg
- tldw_chatbook/Media_Generation/* - keyring import is about 3 ms marginal; the keyring read was taken off startup by TASK-21111(b)
- tldw_chatbook/Video_Generation/video_store.py - covered by known TASK-21109; the store policy read is secrets-free
- tldw_chatbook/Kanban_Interop/* - known TASK-21107 (re-measured 18-20 ms warm self import) and TASK-21239 (zero consumers); no new evidence
- tldw_chatbook/Embeddings/Embeddings_Lib.py - torch/transformers/numpy are lazy (_ensure_*); only the global lock is flagged
- tldw_chatbook/Metrics/metrics_logger.py - per-call cost when disabled is a cached env check plus a dict copy (sub-microsecond); fine even on the ChaChaNotes per-statement sites
- tldw_chatbook/LLM_Provider_Catalog/model_discovery_cache.py, model_discovery_merge.py, models_dev_catalog.py, model_catalog_settings.py - bounded caches; models.dev is opt-in (default off) and loaded once; Console model-option resolution only runs when the settings modal opens
- tldw_chatbook/Study_Interop/*_normalizers.py, server_*_service.py, quiz/study scope pagination - server paging is page-sized and the client is cached by the provider
- tldw_chatbook/Server_Runtime_Interop/* - thin async client wrappers consumed by runtime_policy.server_capabilities; no timers

## census
| Measurement (isolated scratch profile, Py3.12 / Textual 8.2.8, warm pyc) | Result |
|---|---|
| `import tldw_chatbook.app` total self time (3 runs) | 940-1227 ms |
| Slice packages, summed self import time | 54.7 / 58.9 / 77.4 ms |
| css.Themes.themes self import | 24.1-24.7 ms (generate() x91 = 19 ms; ensure x70 = 14.9 ms) |
| Kanban_Interop.server_kanban_service self import (known TASK-21107) | 18.2-20.3 ms |
| LLM_Provider_Catalog package cumulative | 5.2-6.7 ms |
| 5 dead interop families (Claims/Prompt_Studio/MCP_Gov/Auth/Audio), self | about 3-4 ms warm, about 14 ms on first (pyc-compile) run |
| Otel_Metrics marginal import after app (opentelemetry-api/sdk present, exporter absent) | 11.1-11.8 ms, then OTEL_AVAILABLE=False |
| atomic_write_bytes (F_FULLFSYNC), 30 KB / 110 KB | median 23.1 / 20.8 ms, max 27.7 / 33.0 ms |
| apply_settings_mutation_to_cli_config (102 KB config) | 45.8-52.0 ms |
| httpx client construction (create_client / build_httpx_async_client) | 8.0-8.5 / 8.7-9.7 ms |
| ModelCatalogDiskStore.load_into: 8 providers (789 models) / 20 providers (2806) / near-cap (7200) | 1.4 / 7.2 / 11.7 ms |


# slice-76

## summary
Slice #76 (COLD-MISC-4, 64 files, ~10.1k lines) is almost entirely thin, policy-gated REST shims (Server*Service + *ScopeService pairs) plus two local JSON stores (Feedback, Chat Grammars), the ACP runtime process manager and the Ollama management client. None of the slice code has timers, threads, sqlite, polling or per-keystroke paths, and every Ollama call goes through asyncio.to_thread. The slice's real cost is structural and lands on the boot path. app.py imports 16 of these packages at module scope (app.py:551-700) and TldwCli.__init__ (via _wire_watchlists_and_notifications_services, app.py:8121) constructs 16 server/scope pairs plus two local JSON stores. A full grep of attribute names shows none of these 18 services has a single consumer outside the wiring. Measured in an isolated profile: the 54 slice modules cost 9.0 to 12.8 ms self import time (3 runs), and each local store's __init__ `_load()` goes through the raw-participant storage-admission machinery at 4 to 6.5 ms steady state (about 290 open() syscalls each, 18 ms cold). That is roughly 20 ms per launch before first paint for unreachable features (F1). This is the same shape as TASK-254 (RAG_Admin, Done) and TASK-21239 (Kanban, open) but a new, larger scope. Two latent costs would become hot as soon as a UI is wired. (F2) The local store mutations are `async def` with fully sync bodies: storage admission, two nested deepcopies of the whole record list, a whole-file indented JSON rewrite and an fsync. Measured 27 ms per submit at 1 record, 48 ms at 1,000 and 72 ms at 2,000. Soft-deleted rows are never purged. (F3) Every from_config server service owns a private LegacyConfigServerClientProvider, which means a private TLDWAPIClient and httpx pool. Its first request builds a new SSLContext on the event loop (measured 10 to 11 ms per httpx.AsyncClient(verify=True), 47 ms the first time) plus a fresh TLS handshake, and those clients are never closed. That is 16 sites in this slice and 29 in app.py. Minor: the dead duplicate packages Sharing/, Outputs/ and WebClipper/ (zero production imports) and ~20 copies of identical shim boilerplate (F4). ACP start_session always blocks its thread worker for the full 2 s startup timeout even on success (F5). Suggested PR grouping: PR-A retires or lazy-wires the 18 zero-consumer services, defers the local-store `_load` to first use and deletes the Sharing/Outputs/WebClipper duplicates (F1+F4). PR-B moves the from_config sites onto the shared server_context_provider, or caches one SSLContext in tls_trust (F3, app-wide). PR-C fixes the local JSON store write path only if a consumer is added (F2).

## clean areas
- All 16 *ScopeService modules (Sharing_Interop, Meetings_Interop, Voice_Assistant_Interop, Outputs_Interop, Feedback_Interop, Chat_Grammars_Interop, External_Connectors_Interop, Collections_Interop, Personalization_Interop, Companion_Interop, User_Governance_Interop, Web_Scraping_Interop, Web_Clipper_Interop, Tools_Interop, Text2SQL_Interop, Translation_Interop): pure dict normalization and policy routing, no I/O, no loops over unbounded data, no module-scope work beyond an Enum and constant lists. The double _dump/_normalize (server service, then scope) is micro.
- All 16 Server*Service modules: construction is lazy (0.077 ms measured for all 16 pairs); tldw_api schema imports are already deferred function-locally (task-285); no sync HTTP.
- Meetings_Interop stream_meeting_session_events: a proper async generator with no buffering.
- Collections_Interop local path: calls LocalWatchlistsService methods that are themselves async (Subscriptions/local_watchlists_service.py:538/837/1186), so there is no hidden sync sqlite via _maybe_await.
- Local_Inference/ollama_model_mgmt.py: not on the boot import path (only llm_management_events_ollama imports it); all 9 callers wrap it in asyncio.to_thread; the streaming callback is a no-op; the requests.Session per call against a local http Ollama is micro-hygiene only.
- ACP_Interop/runtime_session.py and the ACP_Interop __init__: light (the console_live_work dependency is stdlib-only); snapshot() on the Console sync tick is a Popen.poll() plus a small dict, which is fine; start/stop run in @work(thread=True) workers (acp_screen.py:451/493), so they do not block the loop.
- Boot import accounting: Backup_Recovery.raw_participants (imported by local_feedback_service and local_chat_grammars_service) is already on the boot path via config/storage_admission, so it adds no import cost here; `requests` is also pulled in elsewhere.

## census
| Measurement (isolated profile, Py3.12, audit tree 840ed2ca58) | Value |
|---|---|
| Slice modules imported by `import tldw_chatbook.app` | 54 |
| Sum of their self import time (3 warm runs) | 9.0 / 12.8 / 9.1 ms |
| 16 Server*Service.from_config + ScopeService constructions | 0.077 ms total |
| LocalFeedbackService.__init__, missing file (steady state / cold first) | 4.4-6.5 ms / 18.1 ms |
| LocalChatGrammarsService.__init__, missing file | 3.8-6.6 ms |
| open() syscalls in one store `_load()` (cProfile) | 293 |
| LocalFeedbackService.submit_feedback at 1 / 100 / 1000 / 2000 records | 27 / 30 / 48 / 72 ms (file 1.4 MiB at 2000) |
| LocalChatGrammarsService.create_grammar at 1 / 100 / 300 records | 26 / 21 / 29 ms |
| httpx.AsyncClient(verify=True) construction (first / steady) | 47 ms / 10-11 ms |
| Slice server services wired via from_config (private client provider each) | 16 (29 in app.py overall) |
| Consumers outside app.py wiring for the 18 slice services | 0 |


# slice-8

## summary
Slice #8 (Chat#2, 7 files, 13,542 lines) is almost entirely tldw_chatbook/Chat/console_agent_bridge.py (10,847 lines). The other six files are small and mostly clean.

Hot paths traced:
(a) The 0.2 s Console transcript poll (chat_screen._start_console_transcript_sync_timer). It runs while any run is active. It calls the rail payload (_console_agent_section_payload -> live_snapshot / historical_snapshot / fleet_snapshot / subagent_run) and ConsoleChangeReviewProjection.project -> bridge.change_review_marker_messages.
(b) Conversation resume, via the async open_console_workspace_conversation -> _inject_resume_agent_markers -> bridge.resume_marker_messages. It runs synchronously on the loop.
(c) Console compose_content (before first paint) -> _console_agent_section_lines -> _ensure_console_agent_bridge.
(d) Per-send run_reply, on a worker thread via asyncio.to_thread, plus the per-chunk streaming in _StreamingModelAdapter, on a per-run lifeline loop thread.
(e) The agent-turn finalize (async, on the loop) -> record_run_assistant_message.

Overall health: the live in-memory paths are well bounded. That covers _LiveStepFeed tail, 1 Hz live-usage publish throttle, lock-guarded fleet snapshots, off-loop run-log paging and full-log probe, lru_cached refusal table, and token-count memo. The systemic problem is the AgentRunsDB-derived read models:
- Change-review markers, resume markers, the historical rail snapshot and the drill-in record are all built on the event loop.
- Each uses SELECT * plus full step hydration (JSON-decoding every step payload of every run) when it needs only a few columns or the last 5 steps.
- Three independent derivations run on the same resume.
- The cost is amplified by a newly observed per-operation Backup_Recovery storage-admission tax: about 245 open() syscalls and 5-9 ms on every AgentRunsDB connection() and every ChaChaNotes transaction(). This lives outside the slice, but it sets the floor under every bridge DB call.

Measured in an isolated scratch profile: one conversation open costs about 73 ms of loop time at 30 agent turns, about 186 ms at 100 turns and about 410 ms at 300 turns, summing resume + change-review + historical.

Separately, every agent send and every fleet child builds a fresh event loop. That loop gets a fresh httpx client, which is closed at run end, so provider connections are never reused across turns.

Suggested PR groups:
- PR-A: a metadata-first agent-history read model, computed once off the loop and shared (F1, F2, F4, F5, F9, F12).
- PR-B: cheap sanitizer and per-step helpers (F3, F11).
- PR-C: a persistent model-call loop / connection reuse (F6).
- PR-D: the storage-admission per-op tax, for the Backup_Recovery owner (F7).
- PR-E: move AgentRunsDB construction and the anchor write off the loop (F8, F10; extends TASK-32804.10).

## clean areas
- tldw_chatbook/Chat/console_appearance.py: pure validate/parse helpers. The batch json parse runs in chat_conversation_service (other slice). Clean.
- tldw_chatbook/Chat/console_auto_speak.py: pure policy, no I/O. Clean.
- tldw_chatbook/Chat/console_auxiliary_routing.py: pure dataclass replace. Clean.
- tldw_chatbook/Chat/console_assistant_defaults.py: runs only on new-session creation in a non-global workspace that has defaults (1-2 service reads). Cold path, acceptable.
- tldw_chatbook/Chat/console_capture_policy_repository.py: the write callers use asyncio.to_thread (controller :5812, bare-thread reconcile :5914) and reads run on store hydration. Parameterized, single-row. Clean at slice level. read() opens a deferred transaction for one SELECT, which is harmless.
- tldw_chatbook/Chat/console_canvas_controller.py: in-memory staging under an RLock. Closed stages are bounded by _MAX_RETAINED_CLOSED_RUNS=256. The per-mutation scans over _runs are small and canvas mutations are rare. Clean.
- console_agent_bridge per-chunk streaming (_chat_call_impl._consume): runs on the per-run lifeline thread, not the UI loop. _emit_live_usage is O(1) per chunk, the usage publish is throttled to 1 Hz, and usage_snapshot is a small dict copy. Clean.
- console_agent_bridge live rail reads (live_snapshot, live_run_snapshot, fleet_snapshot, _subagent_summaries_from_fleet, _prune_settled_fleet_survivors): in-memory and bounded by live children. Clean even at 3-4 calls per tick.
- console_agent_bridge _LiveStepFeed: bounded deque tail plus count (TASK-18604 fix holds). Clean.
- console_agent_bridge run-log reads (run_log_available, load_run_log_page, resolve_run_log_target): called only from thread workers in Console_Modules/agent.py (:1425 run_worker thread=True). Clean.
- console_agent_bridge preview builders (build_project_instruction_preview_request etc.): dispatched with asyncio.to_thread by the controller (:21999). Clean.
- console_agent_bridge import cost: only +15 ms and +17 modules beyond chat_screen (measured). Bridge constructor 0.2 ms cold. _refusal_statuses is lru_cached and lazy. Not a startup problem.
- Token counting in build_console_first_request_plan / _fenced_project_instruction_payload_fits: repeated full counts are memoized per message by Utils/token_counter._ESTIMATE_CACHE. Verified fine.
- subagent_counts badge path: batched and TTL-gated in Console_Modules/agent.py (Finding A). Clean.

## census
| Measurement (isolated scratch profile, Py3.12 + Textual 8.2.8, macOS arm64; AgentRunsDB with N primary runs x S steps, M sub-agents x 2S steps, tool results of R chars) | 30x10, 5 subs, R=2k | 100x20, 20 subs, R=3k (3.5 MB) | 300x30, 60 subs, R=4k (22.8 MB) |
|---|---|---|---|
| resume_marker_messages (on loop at conversation open) | 44 ms | 92 ms | 255 ms |
| change_review_marker_messages (on loop per revision bump / switch) | 15 ms | 41 ms | 96 ms |
| _derive_historical_snapshot (on loop at rail cache miss) | 13.5 ms | 53 ms | 58 ms |
| subagent_run = get_run (on loop per 0.2 s drill tick) | 4.7 ms | 15.9 ms | 5.0 ms |
| list_runs(all kinds) raw | - | 19 ms | 74 ms |
| list_runs(primary) raw vs latest_primary_run_metadata | - | 18 vs 9.9 ms | 45 vs 9.7 ms |
| AgentRunsDB connection()+SELECT 1 | 5.7 ms (245 open(), 253 fstat, 65 lstat per op); raw SQL 0.056 ms | | |
| ChaChaNotes transaction()+SELECT 1 | 5.4 ms (245 open() per op) | | |
| safe_intermediate_thinking_summary (200 chars) | 31 us; regex check + str.translate equivalent 1.5 us | | |
| _console_tool_result_display_cap per call | 2.3 us (x6000 per 300-turn resume = ~14 ms) | | |
| drill-in steps join over 1000 steps | 4.4 ms per tick | | |
| AgentRunsDB ctor + reconcile_on_init | 70 ms fresh DB / 303 ms on 22.8 MB DB | | |
| console_agent_bridge import beyond chat_screen | +15 ms, +17 modules | | |


# slice-9

## summary
Slice #9 is effectively one god module, tldw_chatbook/Chat/console_chat_controller.py (30,502 lines, 1.38 MB, 455 methods on ConsoleChatController), plus the 857-line console_chat_fork.py. The controller has no widgets. Its costs land on the event loop through five hot paths, all traced to real callers:

(1) A turn-configuration snapshot is built synchronously on the loop for every manual send (prompt_queue._stage_normal_chain -> wiring._admit_console_turn_to_runtime -> session._build_console_turn_execution_context), every queued prompt, every retry, regenerate, continue, edit, summarize and fleet wake. It runs the module-level capture_* helpers that live in this slice.
(2) The submit -> accept -> stream pipeline.
(3) The 0.2 s Console transcript poll (5 Hz while any run is active), which calls controller.context_control_inputs and run_marker_for on every tick.
(4) Send gating: send_refusal_copy runs on every send and every message action.
(5) First paint, via the controller -> store -> fork import leg.

The biggest item is the skills catalog capture. It re-audits every local skill's trust on every send, with manifest HMAC, directory scans and Backup_Recovery storage admission for each skill. Measured in the scratch profile: 0.2 s for 5 skills and 0.75 to 1.15 s for 20, whether trust is locked or unlocked. It is 0.1 ms with no trust service.

A common root cause runs through findings F1, F2, F4 and F5. Each guarded read (operation(config), get_user_data_dir, the skill-trust _execution_scope, MCP permission-store load) re-reads recovery control records and re-opens every path component through private_paths. That costs roughly 600 to 2,600 open() calls per call, so reads that should cost microseconds take 12 to 48 ms. All ms figures were measured in a scratch profile whose path is about 14 components deep. A real profile is about 6 deep, so expect roughly 40 to 50% of the measured number; the O(n) and per-call structure does not change.

Other hot-path items:
- The Capture-On trace-provenance transaction (Capture-On is the default) is still a BEGIN IMMEDIATE on the loop, even though TASK-22205 moved its two sibling transactions off the loop.
- The memory banner re-derives durable snapshots, branch memory and the global context policy on every tick.
- Each send re-materialises the whole transcript about 8 to 10 times.
- Structurally: the module-size ratchet row (29,367 lines) is already exceeded by 1,135 lines by static count at 840ed2ca58, and the controller's own unmarshal is about 13 to 16 ms warm on the first-paint path.

Verified healthy: the direct-stream per-chunk loop, durable commit and CAS offload, applier offloads, agent-run threading and token counting (see clean_areas).

## clean areas
- console_chat_controller.py _run_direct_provider_reply per-chunk loop (27052-27180): only store.append_stream_chunk + ThinkingCapture.observe; task-33081 incremental thinking append verified (no per-delta full envelope validation, replace_message_thinking(validated=True)); no per-chunk sqlite/logging/fingerprinting
- Durable per-send commit and checkpoint CAS (_accept_durable_turn 12069, resume_durable_postcommit 12302/12475) correctly offloaded via _run_durable_db_call/asyncio.to_thread with :memory: fallback
- _apply_chat_dictionaries/_apply_world_info (23240-23440): applier work offloaded via asyncio.to_thread (the inputs they consume are captured on-loop -- see F2)
- _record_prompt_history -> PromptHistory.append (thread-scoped write), agent run (_run_agent_reply) runs via asyncio.to_thread; approval/skill/chat-create/question/worktree interrupt waits poll event.wait(1.0) on worker threads, not the loop
- _durable_context_snapshots version reads: task-32804.12 batching verified present (get_message_versions_by_ids, 0.69 ms for 400 ids) -- the N+1 from the 2026-09-17 core review is fixed; remaining per-tick cost is caller frequency (F4)
- Token counting inside prepare_chat_request: count_console_messages_tokens measured 0.07 ms (64K tokens) / 0.28 ms (260K tokens) -- not a hotspot despite multiple prepare() passes per send
- get_cli_setting sites in the controller (29): fastpath measured 9 reads = 0.01 ms total; runtime_capture_policy() is generation-cached
- Controller __init__ (4643-5480): cheap construction (repositories, registries, locks); no I/O beyond per-restored-session queue hydration
- run_state/run_state_for/activity_for/streaming_session_id/_set_run_state/_advance_lifecycle_revision: dict/lock operations, cheap per call
- build_context_snapshot / _presented_message_snapshots / _redact_secrets / _replace_image_data_with_placeholders: preview-modal-only (inspector open/refresh); deepcopy of 400 messages measured 5.9 ms -- acceptable for a modal
- provider_messages_for_next_send_estimate: reached only from ConsoleSendPriceController.presentation_for_draft (Send hover), not the keystroke path (TASK-23018 gate verified)
- capture_run_admitted_workspace_roots / _validate_project_instruction_binding lstat chains: ~20 syscalls per binding, sub-ms
- console_chat_fork.py fingerprint/validation functions: fork-action (cold) path only; PIL decode per selected generated image is acceptable there (import-time cost reported separately as F7)
- prepare_library_for_turn/_capture_rag_context: delegate to awaited providers (other slices)
- Maintenance/purge capture-policy mutation threads (5943/6144/15113): cold settings/approval paths; ownership issues already in core review 2026-09-17

## census
| probe (scratch profile, isolated env, py3.12) | measured |
|---|---|
| capture_skill_context_maximum, trust unlocked, 5 / 20 / 40 skills | 196-261 ms / 0.9-1.15 s / 1.8-2.0 s per call |
| capture_skill_context_maximum, trust locked, 5 / 20 skills | 221 ms / 749 ms |
| capture_skill_context_maximum, no trust service, 20 skills | 0.2 ms |
| MCPPermissionStore.load() / get_kill_switch() (file absent) | 35.6 / 34.9 ms |
| describe_local_mcp_capabilities() (rebuilt per call, 32 tools) | 8.7 ms |
| with operation(config) (Backup_Recovery scope) alone | 12.3 ms (~600 open() per call) |
| 9x get_cli_setting inside it | 0.01 ms |
| get_user_data_dir() via default_emergency_stop_path() | 34.8-48.3 ms (~1,590 open() per call) |
| emergency_stop_state(cached path) | 0.009 ms |
| trace admission txn (BEGIN IMMEDIATE + ensure_current_revision x N + ensure_policy + commit), N=50/200/400 | 9.6 / 14.5 / 15.9 ms median (max 69 ms) |
| load_applicable_branch_memory (400-row lineage) | 4.96 ms |
| get_message_versions_by_ids(400) | 0.69 ms |
| store.messages_for_session-equivalent snapshot copy (400 msgs, 43-field dataclass replace) | 2.28 ms per call |
| fingerprint_payload 200 text turns / 4x1.5 MB images | 1.8 ms / 19.3 ms (+7.0 ms base64) |
| persisted_attachment_digest per 1.5 MB image | 0.51 ms |
| PIL.Image import (resident at Console first paint via console_chat_fork + 6 others) | 9-11 ms warm |
| console_chat_controller self import time | 13-16 ms warm, 61 ms cold |
| run-marker pass O(S^2) activity() calls, S=5/15/30 | 0.03 / 0.23 / 1.03 ms |
