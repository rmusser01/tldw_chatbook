# Structural efficiency & speed audit — tldw_chatbook

- **Pin:** `origin/dev` **840ed2ca58** (2026-09-27)
- **Scope:** all 2,541 Python files and 1,825,311 lines under `tldw_chatbook/`
- **Method:** 95 finder agents plus adversarial verifiers, in one workflow run of 533 agent starts across two sessions (123 hit the account usage limit and were retried on resume):
  - **76 code slices.** Hot code (UI, Chat, DB, runtime) in ~32k-line slices; cold feature backends in ~60k-line slices.
  - **14 cross-cutting pattern sweeps**, each over the whole tree.
  - **5 runtime measurement probes**, all in an isolated scratch profile:
    - the perf-guard suite;
    - DB micro-benchmarks on a synthetic 3k-conversation / 150k-message DB;
    - Console mount, typing and send CPU profiles;
    - boot plus idle tax;
    - an 8-destination screen-switch tour.
- **Result:**
  - **1,043 raw findings → 920 confirmed**, 22 refuted, 11 already fully tracked, 90 unverified. Dedup gives **905 unique issues**: **42 P0, 194 P1, 335 P2, 334 P3**.
  - They group into **30 PRs in 7 waves** (§4).
  - Every issue, by PR: [`appendix-issues-by-pr.md`](appendix-issues-by-pr.md).
  - Raw data and tooling: [`data/`](data/).

---

## 1. Executive summary

**The app has not slowed down in many places at once. It has slowed down in two places that everything else calls.** Both landed after the last holistic perf review (2026-09-04), and together they multiply the cost of almost every other interaction:

1. **Per-call Backup_Recovery storage admission (ADR-126, `b5251e9a6e`, 09-16).**
   - Every warm `load_settings()`, `get_runtime_config_snapshot()`, `get_user_data_dir()` / `get_*_db_path()` call pays a full admission handshake. So does every outermost `transaction()` on ~12 DB owners, and every guarded MCP, persona or dictionary store read.
   - The handshake re-walks directory chains from `/` with one `open()` per path component: about **245 opens per DB transaction** (4–15 ms), **~650 per warm `load_settings`** (9–20 ms) and **~1,000–1,700 per `get_user_data_dir`** (29–53 ms).
   - TASK-32804.1 (09-19) fixed only `get_cli_setting` (now 1 µs). The other 15 guarded entry points were left paying.
   - Measured: **~207k `open()` syscalls to reach `_ui_ready`**; 104 admissions on the loop during boot (about 70% of construct-to-ready); **56 per Console visit** (1.1–1.3 s of loop CPU); **1–3 per keystroke** (26–73 ms/key); **4 per second at idle** from the credential poll (Console idle at 7.7–9.7% of a core on the loop thread).
2. **A Python helper subprocess per private-SQLite connection (ADR-125, `61a49de2e0`, 09-07), multiplied by close-per-call wrappers.**
   - Every file-backed `connect_private_sqlite` on macOS/Linux spawns `python -I -S private_sqlite_helper_entry.py`: **~45–75 ms per open**, against 0.06 ms raw and 0.55 ms before 09-07.
   - The standard off-loop wrappers close the handle after every call: `run_owned_db_call`, `operation_owned_connection`, `run_finite_local_worker`, `list_and_close`, `run_db_off_loop`, the scope services and `ScheduledTasksDB`. So a helper spawn is paid **per operation** on:
     - Library, Notes, media reader, Watchlists and Schedules calls;
     - every agent trace write;
     - the 1 Hz legacy trace-maintenance tick, forever.
   - TASK-24457's "8 connections per Library visit" is now ~70× more expensive than when it was filed.

Four further systemic problems sit on top of those two:

3. **No GC policy, plus screen leaks.**
   - Nothing calls `gc.freeze()` or tunes thresholds. Automatic gen-2 collections take **130–871 ms** and land in about 1 of every 2–4 screen switches (heap 0.57M → 1.6M objects after a tour).
   - The leaks feed those pauses. Every Settings visit leaks its whole `SettingsScreen` (~10.7 MB, via a Textual `Signal` subscription never dropped). Personas screens are pinned by a thread-worker ContextVar (~14 MB each). Home pins discarded trees.
   - `gc.freeze()` after boot measured **~0 ms**.
4. **Event-loop discipline gaps on the send, tick and keystroke paths.**
   - The **first send takes 6.5–11.8 s before the provider call**. Causes:
     - ~400 admissions from sync passes with no derivation scope;
     - a per-skill trust audit (40–90 ms per installed skill);
     - the MCP catalog resolved twice (~210 ms);
     - a cold RAG profile manager built on the loop (794 ms);
     - sensitive-path resolution at ~17 guarded config reads each (3.6 s wall).
   - Each typing pause fires a **250–415 ms** loop stall.
   - Streaming restyles whole Markdown subtrees every tick.
5. **An eager boot composition root.**
   - `TldwCli.__init__` constructs ~15 services and opens **5 feature DBs** before first paint (~340 ms for the DBs alone; each open pays the helper spawn).
   - `app.py` pulls **367 app-only modules** (~450–540 ms) onto the pre-paint import path.
   - The screen pre-import payload ratchet is **red** (554/500 modules) and was not caught, because `perf-guard.yml` doesn't run it.
6. **The perf guards are blind to all of the above.**
   - The keystroke census reported **0** work per key while each key ran 27–69 guarded `load_settings` calls; it counts derivations, not admissions or helper spawns.
   - The Console mount profiler is broken on dev: the route is reusable now and the hook never fires.
   - ~14 startup guards are masked by a `RecoveryRequired` harness baseline.
   - That is how both regressions shipped without a red signal.

**Fix order that follows from this:**
- **Wave 0:** guards and five independent quick wins.
- **Wave 1:** the keystones (admission, connection lifecycle, GC).
- **Re-measure**, then **waves 2–6**. Once Wave 1 lands, many downstream P1s shrink by an order of magnitude; the core-review lesson "re-measure against the config fast path before touching the rest" applies again.
- **Minimum high-impact set:** PERF-01 → PERF-11 (11 PRs). Measured, it removes the dominant cost from boot, idle, typing, send, visits and agent tool calls.

### Headline numbers

Measured on this pin. The machine carried load averages of 10–50 from other sessions, so absolute ms are inflated ~1.5–3×. Syscall, call, spawn and object counts are the robust figures (§7).

| Path | Now | Dominant cause | Fixed by |
|---|---|---|---|
| Warm TTI (process start → `_ui_ready`) | 7.5–9.9 s (09-04: 2.84 s, unpaired) | ~190–207k `open()`s, 32 helper spawns (12 on main thread), eager `__init__` | PERF-06/07/08/09/16/17 |
| First 5–12 s after ready | loop 30–75% busy, 13–16 blocks of 82–900 ms | post-ready Console derivation storm + admissions | PERF-06, PERF-14 |
| Idle Console | 10.8–17.8% of a core in-process (+3–4% in reaped helpers), ~3,400 `open()`/s | 4 Hz credential poll (1 admission/tick), 1 Hz trace maintenance (helper spawn/tick), 10 Hz backup-maintenance probe | PERF-06, PERF-10, PERF-08 |
| Typing on Console | +21–37 ms/key post-echo; 250–415 ms stall per typing pause | 1–3 admissions/key; debounced spend refresh outside a derivation scope | PERF-06, PERF-14 |
| First send (Enter → provider dispatch) | 6.5–11.8 s | ~400 admissions, skill trust audit, MCP catalog ×2, RAG cold build, sensitive paths | PERF-06/07, PERF-12, PERF-15 |
| Warm Console visit | 0.84–1.42 s loop CPU (0 widgets built) | ~120 warm `load_settings` calls + 533-node resume restyle | PERF-06, PERF-14 |
| MCP visit | 1.1–2.6 s loop CPU, 165–240 admissions | guarded MCP store reads on the loop | PERF-08, PERF-23 |
| Conversation search (Ctrl+K History, Library, browser) | ~3 s rare term, **~70 s** common term | correlated `EXISTS(… messages_fts MATCH …)` per candidate | PERF-02 |
| Insert a message with a 3 MiB image | **1.2 s** (9 ms text-only) | `set_trace_callback` makes SQLite render every BLOB param as hex, per statement and trigger | PERF-04 |
| Library op (scope service, media seam, notes read) | ~47–115 ms per call | helper spawn per call (close-per-call) | PERF-09 |
| GC pauses | 130–871 ms, about every 2nd–4th switch | no `gc.freeze`, leaks | PERF-05, PERF-11 |
| Conversation text search, DB only | 1,124 ms vs 7 ms uncorrelated | (same as above) | PERF-02 |

---

## 2. Method, briefly

- **Finders.** Each slice finder got a shared context file with:
  - the audit rules;
  - the severity rubric (P0 = user-visible ≳100 ms freeze on a common path, an always-on tax, or unbounded growth);
  - the 81 open perf-flavoured tasks;
  - the prior reviews' "verified fine, do not fix" list;
  - the Textual traps (`run_worker(coro)` is not a thread, and so on).

  Each finder traced at least one real caller per finding.
- **Verification.** Every finding then went to an independent verifier told to refute it. The verifier re-read the cited line, traced callers across the repo (no truncated greps), looked for missed offloads and caches, matched open tasks, and re-rated severity. Verifiers downgraded ~100 severities, upgraded ~15, refuted 22 and marked 11 as fully tracked.
- **Measurement.** The five probes and most pattern sweeps ran safe micro-benchmarks under a scratch `HOME` / `XDG_*` / `TLDW_CONFIG_PATH` with `TLDW_TEST_MODE=1`. The user's real profile was verified untouched afterwards: `config.toml` hash `fd1f1351…` and mtime Sep 26 are unchanged.
- **Dedup.** Findings in the same PR group were clustered by location and title similarity: 1,021 live findings → 905 issues. The keystone problems were reported independently by 5–10 agents with consistent numbers. The `n` column in the appendix is that corroboration count.

---

## 3. Root causes and the bad practices behind them

Ranked by leverage. Each names the pattern to stop doing, with the census numbers from the pattern sweeps.

### R1. Security and integrity checks re-derived from disk on every call (no amortization)
- **Where:** `Backup_Recovery/config_participants.py`, `participants.py`, `storage_admission.py:835`, `bootstrap.py:33`, `Utils/private_paths.py`.
- **What it does:** every outermost guarded call re-walks up to ~104 verified-parent chains with fresh `openat`s, reads `registry.json` ~7 times, and takes a flock. All of this runs under a process-wide `_Acquisition.initializing` section with a 10 ms poll-wait, which caps DB operations at ~200/s process-wide (4 threads get 0.96× of one).
- **Scaling:** nested calls are free, so the cost is paid per outermost call. It is linear in path depth: about 2.0 ms + 0.71 ms per path component.
- **Bad practice:** putting a per-call guard around a pure in-memory cache read. TASK-32804.1 recognised this ("the handshake protects the file/build lifecycle, not a pure in-memory cache read") but applied it to 1 of 16 guarded functions.

### R2. Expensive connection setup combined with connection-per-operation lifecycles
- ADR-125 made a new connection cost a helper process, which is correct for its lock-safety goal.
- About 37 `to_thread(...) then close_connection()` sites, the owned-connection wrappers, and 88 `_run_library_service_call` sites (75 of which also create a **new event loop per call** with `asyncio.run`) still assume connects are free.
- **Leak:** two registries hold strong references to thread-local handles, the quiescence registry (`base_db.py:157`) and the participants registry (`participants.py:378/550`). Any per-call thread that exits while holding one leaks it: +1 connection, +2 fds, ~1.7 MB RSS per ChaChaNotes handle.
- **Bad practice:** threads and connections as disposable per-call resources.

### R3. No memory or GC policy for a long-lived, widget-heavy process
- No `gc.freeze()` and no thresholds, with a 0.57–1.6M-object heap.
- Three leak mechanisms feed it:
  - Textual `Signal` subscriptions: `WeakKeyDictionary` values are bound methods that pin their keys.
  - The thread-worker `active_worker` ContextVar is never reset on pool threads.
  - A `query_one` cache on a reused screen pins its discarded trees.

### R4. Sync work behind async-looking seams on the event loop
- **Offload discipline is broadly good:** 996 `asyncio.to_thread`, 235 `thread=True` and 50 `run_in_executor` sites. No timer callback does sqlite on the tick, and no keystroke handler runs an undebounced DB search except the three noted in PERF-14/22.
- **The misses cluster around:**
  - "async def facades over sync I/O": the local Skills and Chatbooks services (86 `@content_call`s), `SchedulingService`, `AudioService.convert_audio`, `recovery_review` async provider-guard wrappers;
  - coroutine workers that assume they are threads (`LocalSkillsService.get_context` on mount and resume);
  - Console sync passes that don't open a `_console_derivation_scope()` (only 3 exist in `chat_screen.py`).

### R5. The composition root builds everything before first paint
- `TldwCli.__init__` wires ~15 services, 5 feature DBs, Evals, Watchlists, MCP control plane, persona/dictionary services, TTS and File-Notes-git, plus 18 interop services **with zero consumers**.
- `app.py` is a 228-import root in which **404 of 470 module-scope names are used only inside functions**.
- Module-scope work compounds it: AA hue-pinning for 91–94 themes (25–35 ms), `tldw_profile_core` imported for one integer, `requests` pulled by 7 boot modules, the citation pydantic cluster (45–60 ms), and `MCP/server.py` importing the gateway just to set a flag.
- Model definitions are about a third of pre-paint import: 1,605 dataclasses (~540 ms) and 264 pydantic models (~128 ms).

### R6. Whole-subtree rebuilds where a targeted update exists, and no virtualization
- **Recompose on selection or visit, not per keystroke:**
  - Home re-composes on every visit (68 ms, against 3 ms for the targeted `_sync_home_triage()` that already exists);
  - Watchlists rebuilds panes per cursor key;
  - Library destination switches `await self.recompose()`;
  - the Settings category switch re-mints both panes (215–900 ms);
  - streaming re-adds managed classes on every tick.
- **Virtualization:** **no user-data list uses `OptionList`/`Tree`**. Every one mounts a Button or Static per row: 1.6–2.2 ms per row, so 1,000 rows take 2.0 s, against a flat ~85 ms for an `OptionList`.

### R7. Transport churn
- Every hosted LLM call builds and discards a `requests.Session`: 15.2 ms against 0.9 ms reused on loopback, plus a real TCP/TLS handshake per send.
- Every httpx client rebuilds an `SSLContext` from certifi (12–24 ms); ~15 sites do it on the loop.
- Every agent run builds a new thread, a new event loop and a new `AsyncClient` (33.7 ms plus a cold handshake).
- 87 client construction sites, only ~12 of them long-lived. Timeouts are in good shape.

### R8. The logging pipeline defeats its own level filter
- `Logging_Config.py` forwards every loguru record to stdlib at level `TRACE`. So every `logger.debug` pays ~7–8 µs instead of 0.15 µs, every `opt(lazy=True)` guard is defeated (including TASK-275's), and a dropped `opt(exception=True).debug` formats a full traceback (~50 µs).
- Every INFO+ record is redacted **three times** (~340 µs) and flushed synchronously on the emitting thread, the loop included.

### R9. Guards that measure the wrong unit
- The census counts derivations, not admissions, helper spawns, `open()`s or GC pauses.
- The pre-import payload guard is not in `perf-guard.yml`.
- The mount profiler is broken by route reuse.
- Guards run under a harness that raises `RecoveryRequired` at `app.py:1110`.
- Several ratchets sit at 0–2 units of headroom: boot CSS bytes 607,640/608,090; bare-type rules 273–274/274; ui-ready 1031–1033/1033; import weight 681/686.

**Verified fine; do not "fix" these** (re-confirmed on this pin):
- The streaming core. The store folds chunks into a list buffer and materializes once per 0.2 s tick; the transcript planner is windowed and linear.
- The token-estimate memo (bounded, keyed on hash).
- The ModelCapabilities per-pair cache.
- The Console registry-display generation cache, which is the template to copy.
- All 21 `lru_cache` sites.
- Regex use: 273 literal patterns fit CPython's 512-entry cache.
- Keyring reads (TTL-cached, TASK-32921/22/26).
- Web fetch/search caches.
- Media v9 partial indexes (browse at ~1 ms SQL).
- Per-conversation open, messages and keyword paths.
- Library screen reuse.
- Boot worker census.
- Timeouts everywhere.

---

## 4. The PR plan

**Conventions** (from this repo's standing rules):
- one PR per work stream;
- branch off `origin/dev` in a worktree;
- rebase (not merge-in) before merge, and run `./scripts/preflight.sh`;
- every PR states its **before/after gate number**, measured with the PERF-01 guards on a quiet machine, and adds a guard or test that fails if the regression returns;
- P2/P3 items listed for a PR are **in scope when they touch the same files**; the rest can ride a later polish PR.

**File hot-spots to serialize:**

| File | PRs touching it |
|---|---|
| `config.py` | 06, 07 |
| `Backup_Recovery/*` | 06, 07, 08 |
| `DB/base_db.py`, `DB/private_sqlite.py` | 04, 09, 10 |
| `UI/Screens/chat_screen.py` | 06, 13, 14 |
| `Chat/console_chat_controller.py` | 12, 15 |
| `app.py` | 03, 11, 16, 17 |

### Wave overview

| Wave | PRs | Theme | Depends on | Size |
|---|---|---|---|---|
| **0: guards + quick wins** (parallel) | 01–05 | Make the regressions visible; five small, independent, low-risk fixes | none | S each |
| **1: keystones** | 06–11 | Admission cost, connection lifecycle, trace tick, GC | 01 (for gates); 08 and 11 need owner decisions | M–L |
| *re-measure* | | Re-run the 5 probes; re-rank waves 2–6 (many P1s shrink 5–50×) | Wave 1 | |
| **2: Console** | 12–15 | Send path, idle/tick/render, typing/first paint, agent runtime | 06 (06/07/09 strongly preferred) | M–L |
| **3: boot** | 16–19 | `__init__` diet, import diet, server-mode client, CSS | 07, 09 | M |
| **4: network** | 20 | Pooled sessions, cached SSLContext | none | M |
| **5: screens** (parallel) | 21–25 | Library, Settings/Personas, MCP, Watchlists/Schedules, others | per task: 06 / 08 / 09 (each task's `dependencies` names the keystones its screen needs); all follow the post-Wave-1 re-measure | M–L |
| **6: feature/data-scaled** (parallel) | 26–30 | Terminal, Notes sync / Personal Context, DB hygiene, memory/algorithms, cold sweep | 09 | S–M |

Per-PR issue counts (unique issues after dedup):

| PR | Task | Title | P0 | P1 | P2 | P3 |
|---|---|---|---|---|---|---|
| PERF-01 | TASK-33260 | Perf guards that see admissions, spawns, pre-import | 0 | 0 | 4 | 4 |
| PERF-02 | TASK-33261 | Conversation search FTS regression | 1 | 0 | 2 | 1 |
| PERF-03 | TASK-33262 | Logging pipeline | 0 | 0 | 4 | 9 |
| PERF-04 | TASK-33263 | Trace-callback BLOB expansion | 0 | 1 | 0 | 0 |
| PERF-05 | TASK-33264 | Screen leaks | 1 | 2 | 0 | 0 |
| PERF-06 | TASK-33265 | Config warm-hit fast paths + derivation scopes | 6 | 3 | 4 | 2 |
| PERF-07 | TASK-33266 | Memoize user-data dir / DB paths / sensitive paths | 4 | 7 | 3 | 0 |
| PERF-08 | TASK-33267 | Amortize storage admission (ADR-126 amendment) | 5 | 19 | 12 | 5 |
| PERF-09 | TASK-33268 | Private-SQLite connection lifecycle | 5 | 9 | 3 | 3 |
| PERF-10 | TASK-33269 | Legacy trace maintenance | 4 | 1 | 0 | 0 |
| PERF-11 | TASK-33270 | GC policy | 1 | 3 | 0 | 0 |
| PERF-12 | TASK-33271 | Console send path | 2 | 15 | 17 | 12 |
| PERF-13 | TASK-33272 | Console idle / tick / render | 2 | 11 | 36 | 40 |
| PERF-14 | TASK-33273 | Console typing / first paint / resume | 1 | 7 | 10 | 6 |
| PERF-15 | TASK-33274 | Agent runtime | 3 | 14 | 13 | 13 |
| PERF-16 | TASK-33275 | `TldwCli.__init__` diet + dead boot work | 0 | 12 | 6 | 8 |
| PERF-17 | TASK-33276 | Boot import diet + pre-import paydown | 0 | 5 | 18 | 21 |
| PERF-18 | TASK-33277 | Server-mode client off the loop | 1 | 1 | 4 | 2 |
| PERF-19 | TASK-33278 | Boot CSS paydown | 0 | 2 | 0 | 0 |
| PERF-20 | TASK-33279 | HTTP client reuse | 0 | 3 | 5 | 12 |
| PERF-21 | TASK-33280 | Library screen | 3 | 22 | 24 | 29 |
| PERF-22 | TASK-33281 | Settings + Personas | 0 | 4 | 27 | 10 |
| PERF-23 | TASK-33282 | MCP workbench | 0 | 7 | 2 | 5 |
| PERF-24 | TASK-33283 | Watchlists + Schedules | 2 | 7 | 9 | 23 |
| PERF-25 | TASK-33284 | Other screens | 1 | 14 | 39 | 45 |
| PERF-26 | TASK-33285 | Terminal | 0 | 2 | 3 | 0 |
| PERF-27 | TASK-33286 | Notes sync + Personal Context | 0 | 11 | 23 | 10 |
| PERF-28 | TASK-33287 | DB query hygiene | 0 | 0 | 13 | 11 |
| PERF-29 | TASK-33288 | Memory growth + data-scaled algorithms | 0 | 4 | 24 | 20 |
| PERF-30 | TASK-33289 | Cold-feature / hygiene sweep | 0 | 8 | 30 | 43 |

---

### Wave 0: guards and quick wins

#### PERF-01 (TASK-33260): Perf guards that can see what shipped
- **Why first:** the two keystone regressions shipped green, so every later PR needs a guard that sees its unit.
- **Scope:**
  - count **config admissions, storage admissions, private-SQLite helper spawns and `open()`s** in the keystroke census and a new idle/visit census (budget 0 per key, 0 per idle tick);
  - wire `test_screen_preimport_payload_budget.py` into `perf-guard.yml`, which is **red at 554/500** (pay it down in PERF-17 or re-pin with a task);
  - fix `run_console_mount_profile.py` for the reusable Console route;
  - fix the stale CSS meta-test (expects 6 sources, 5 exist);
  - un-blind the ~14 guards masked by `RecoveryRequired('raw_source_selection_changed')` from module-scope `APP_CONFIG = load_settings()` (`app.py:1110`);
  - make size ratchets a PR gate (TASK-32809.1).
- **Tasks:** TASK-32644, TASK-23155/31816, TASK-32809.1.
- **Gate:** the new census fails on today's dev (26–73 ms, 1–3 admissions per key) and passes after PERF-06.

#### PERF-02 (TASK-33261): Conversation search: undo the correlated FTS `EXISTS` (P0, a TASK-278 regression)
- **Problem:** `DB/ChaChaNotes_DB.py:11759` `search_conversations_page` evaluates a full `messages_fts MATCH` once **per candidate conversation**.
- **Cost:** 1,124 ms against **7 ms** for the uncorrelated form in the DB bench, and ~3 s (rare term) or **~70 s** (common term; count query alone) on the 150k-message probe.
- **Fix:** `conversations.id IN (SELECT m.conversation_id FROM messages_fts JOIN messages m ... WHERE messages_fts MATCH ?)`, keeping title/id matching.
- **Test:** add a plan pin captured with `sqlite_stat1` absent, per CLAUDE.md. Keep the hidden-column MATCH form; the bare-alias form was the 07-16 trap.
- **Reaches:** Ctrl+K History search, Library ▸ Conversations search, Console browser search. Also fold in the Personas inspector's per-keystroke conversation search.
- **Size:** S.

#### PERF-03 (TASK-33262): Logging pipeline
- **Changes:**
  - give the loguru→stdlib sink the effective minimum level, not `TRACE`, so dropped debug calls cost 0.15 µs instead of 7–8 µs and `opt(lazy=True)` works again (TASK-275);
  - redact once per record, not three times (`shouldRollover` + `emit` + Logs buffer);
  - move file and buffer handlers behind a `QueueHandler`;
  - demote per-row INFO in the DB layer and the chatbook importer to DEBUG;
  - stop the WARNING-per-transition for app-owned workers (TASK-31806).
- **Measured:** 339 → 108 µs per INFO record, 0.14 µs per dropped debug.
- **Size:** S. Centralised in `Logging_Config.py`, `Utils/log_sanitizer.py` and `app.py`.
- **Tests:** redaction regressions.

#### PERF-04 (TASK-33263): ChaChaNotes trace callback expands every BLOB (P1)
- **Problem:** `DB/base_db.py:738` installs `set_trace_callback` only to spot `BEGIN`/`COMMIT`/`ROLLBACK`. CPython therefore renders the expanded SQL, hex-encoding every BLOB, for the statement and every trigger/FTS sub-step.
- **Cost:** a 3 MiB image message takes **1,203 ms against 9 ms**, and the durable-turn commit holds `BEGIN IMMEDIATE` that long.
- **Fix:** detect transaction boundaries without a trace callback, e.g. `connection.in_transaction` checks around the execute wrapper, or the authorizer's `SQLITE_TRANSACTION` action code (already installed). Preserve the semantic-mutation guard's fail-closed behaviour.
- **Size:** S.
- **Test:** image-insert timing pin plus the existing guard tests.

#### PERF-05 (TASK-33264): Screen leaks
- **Changes:**
  - **Settings (P0):** unsubscribe `theme_changed_signal` on unmount (`settings_screen.py:4187`), since each visit retains the whole screen (+71k objects, +10.7 MB);
  - **Personas (P1):** reset or avoid the thread-worker `active_worker` ContextVar pin (`ccp_character_handler.py:470`), which retains 6 of 10 screens at ~14 MB each;
  - **Home (P1):** stop the per-visit whole-screen recompose after the chatbook-snapshot worker (use `_sync_home_triage()`, 3 ms against 68 ms) and drop the `query_one` cache pin.
- **Gate:** retained-screen count after 10 visits = 0. Add a leak test that fails if any future screen subscribes to an app signal without unsubscribing.
- **Size:** S.

### Wave 1: keystones (owner decisions flagged)

#### PERF-06 (TASK-33265): Config warm-hit fast paths + Console derivation scopes (P0 ×6)
- **Changes:**
  - extend TASK-32804.1's `_warm_config_cache_hit` precedent: an unguarded warm-hit path for `load_settings`, `get_runtime_config_snapshot` (which also deep-copies the whole config on every call) and `get_model_cache_dir`;
  - remove the explicit `operation(config)` wrapper from the 5 Hz run tick;
  - add `_console_derivation_scope()` to `_sync_native_console_chat_ui`, `_sync_console_settings_summary` and the character-context presentation (one admission per pass instead of hundreds);
  - reorder the memo check at `session.py:3874`;
  - stop instant-persist Settings inputs calling `load_settings` per keystroke.
- **Closes or advances:** TASK-32804.1 (reopen: the fast path covered 1 of 16 guarded functions), TASK-32804.3 (4 Hz poll), TASK-24454, TASK-2902 (part).
- **Expected (measured shares):** idle Console loop from 7.7–9.7% to ~1% of a core; typing admissions from 1–3 per key to 0; most of the first-send admission storm and of the 1.1–1.3 s per Console visit removed.
- **Risk:** low–medium, since the precedent exists. Keep the external-edit stat check and a generation guard.
- **Size:** M.

#### PERF-07 (TASK-33266): Memoize `get_user_data_dir`, DB paths and the sensitive-path context per config generation (P0 ×4)
- **Problem:** `get_user_data_dir()` (`config.py:9433`, 203 call sites, 29 on the boot path) costs 29–53 ms and ~1,000–1,700 opens per call. It runs 5 admission scopes, the data-root file lock and a `secure_private_directory` walk. It is uncached.
- **Fan-out:**
  - `resolve_sensitive_context` calls it **19–20 times** (607 ms per resolution);
  - every agent file/git/patch tool call and every `@`-reference resolves it at least once;
  - `RunLogWriter.bind` resolves it twice per send;
  - the emergency-stop path resolves it per send and per 30 s scheduler tick;
  - `RAGConfig()` resolves the Chroma dir through it.
- **Fix:** resolve once per config generation and data-dir setting, then re-verify the **whole verified chain** cheaply instead of re-walking it:
  - compare every ancestor's current `lstat` identity (`st_dev`/`st_ino`, mode, owner) with the identity pinned at first verification. That is O(depth) stat calls, with no opens, registry reads or locks.
  - where a caller can take a directory fd, prefer `openat` relative to the held, verified fd, so a later path swap cannot redirect it.
  - a **leaf-only check is not sufficient**: re-permissioning or swapping an ancestor while the leaf is unchanged must still be refused, as the current walk refuses it.

  Memoize `resolve_sensitive_context` on the same key. Cache `default_emergency_stop_path()`.
- **Owner decision D2:** is a generation-keyed memo with a per-call **full-chain** identity re-check (every verified ancestor, not just the leaf) an acceptable replacement for the per-call chain walk under ADR-029/ADR-126?
- **Tasks:** TASK-1320.
- **Size:** M.

#### PERF-08 (TASK-33267): Amortize Backup_Recovery storage admission (ADR-126 amendment) (P0 ×5, P1 ×19)
- **The biggest lever and the riskiest change:**
  - reuse verified directory pins and admission evidence within a generation, re-checking identity through held descriptors (`fstat` of pinned dir fds) instead of re-walking `/` with fresh `openat`s per component;
  - take admission per connection or per generation for DB owners instead of per outermost `transaction()` (~245 opens each on ChaChaNotes, AgentRuns, Prompts, Evals, Subscriptions, Workspace registry, Notes device state);
  - replace the process-wide `_Acquisition.initializing` 10 ms poll-wait with a condition, so unrelated DBs don't serialize;
  - add a warm read cache for the 5 guarded MCP JSON stores (~240 admissions per MCP visit);
  - drop `@_chat_sources.guarded` from pure in-memory persona and dictionary reads;
  - run the provider recovery guard once per stream, not per chunk;
  - move async provider-guard wrappers off the loop;
  - make the run-log append take admission once;
  - replace the backup-maintenance monitor's **10 Hz** 36-open probe with an event or 1 Hz;
  - load `journal.py`'s ~40 pydantic models lazily.
- **Owner decision D1:** amend ADR-126 to permit generation-scoped admission evidence. Security invariant to preserve: a moved, replaced or re-permissioned directory, **whether the admitted directory or any verified ancestor**, is detected before any read or write that depends on it.
  - `fstat` on the held fds catches re-permissioning of the pinned inodes.
  - Catching a path-level swap also needs a per-component `lstat` identity comparison, or access that goes through the held fds (`openat`).
- **Tasks:** TASK-32860, TASK-31502 (quiescence tax, related), TASK-24457.
- **Gate:** `open()`s to `_ui_ready` (today ~190–207k); `open()`/s at idle (today ~3,400); per-transaction cost (today 4–15 ms, raw 1.4 µs).
- **Size:** L. Split into 08a (evidence reuse in `storage_admission`/`bootstrap`) and 08b (per-owner call-site changes) if review needs it.

#### PERF-09 (TASK-33268): Private-SQLite connection lifecycle (P0 ×5)
- **Principle:** keep ADR-125's helper exactly as designed, but **open far fewer connections**.
- **Changes:**
  - replace close-per-call wrappers with long-lived per-thread handles on a small dedicated DB executor per owner, following the TASK-23027 Notes-sync executor template. The wrappers are `run_owned_db_call`, `operation_owned_connection`, `run_finite_local_worker`, the scope services' `list_and_close`, the `Media` seam, `NotesScopeService`, `run_db_off_loop`, `SyncStateRepository`, receipts ledger reads, and `ScheduledTasksDB`'s per-operation connections (**P0: every scheduling call ~45–50 ms**);
  - stop the blocking `wal_checkpoint(TRUNCATE)` on every close;
  - fix the handle leak: weak references in the quiescence and participants registries, or deregistration on thread exit;
  - move `console_launch_wake` and the character-context fingerprint round trips off the loop;
  - stop the 75 `asyncio.run`-per-call Library service hops from minting a loop each time.
- **Owner question Q3:** ADR-125 allows batching a DB's fixed sidecar inventory per helper operation. Should validation be done once per DB per process lifetime (with helper leases) rather than per connection? This is optional, and a further ~10× on first-visit costs.
- **Gate:** helper spawns per Library visit, per idle minute (target 0) and per send.
- **Tasks:** TASK-24457 (re-scope: its cost is now ~70× larger).
- **Size:** M–L.

#### PERF-10 (TASK-33269): Legacy trace maintenance (TASK-31501, with new evidence) (P0 ×4)
- **Problem:** the ~1 Hz forever tick now costs 6–9 ms CPU plus, on an unconnected executor thread, a **helper spawn per tick** (79 ms wall, 45 ms child CPU). Trace GC re-marks the whole reachable ledger inside `BEGIN IMMEDIATE` every 60 s after any send.
- **Fix:**
  - park the loop once `logical_complete` and wake it from the exchange writer;
  - move the completion check into a read-only transaction;
  - run on one connected thread;
  - make GC incremental (a mark generation or dirty set).
- **Gate:** idle wakeups from this loop drop to 0; idle `open()`/s.
- **Size:** S–M.

#### PERF-11 (TASK-33270): GC policy (TASK-31966, needs an ADR) (P0)
- **Problem:** gen-2 pauses of 130–871 ms land about every 2nd–4th screen switch; 4 gen-2 collections (517 ms) run during mount.
- **Fix:** `gc.freeze()` after `_ui_ready` and after the screen pre-import pass, plus a documented threshold policy (e.g. `gc.set_threshold` tuned for a long-lived TUI heap). Optionally, an idle-time `gc.collect(1)` so collections land when the user isn't interacting.
- **Measured:** `gc.freeze()` ≈ 0.0–0.1 ms, and it removes the scan of the boot heap.
- **Owner decision D3:** TASK-31966 requires an ADR for a global GC policy.
- **Order:** after PERF-05 (leaks), so frozen objects aren't leaked ones.
- **Size:** S code, plus an ADR.

### Wave 2: Console (after re-measuring)

#### PERF-12 (TASK-33271): Console send path: one off-loop turn snapshot (P0 ×2, P1 ×15)
- **Consolidate** the two per-send turn-snapshot builders (`console_chat_controller.py:1300`/`:1335`, `session.py:3499`) into one, computed off the loop. It covers:
  - **skill trust and fingerprint:** cache by manifest stat identity, 40–90 ms per installed skill today;
  - **MCP catalog + permissions:** once per send, not twice (~210 ms), with `compose_catalog`'s two permission-store loads merged;
  - **workspace registry:** read once (3 uncached WorkspaceDB blocks today);
  - **world books and dictionaries:** read off the loop, without N+1;
  - **lazy RAG `ConfigProfileManager`:** don't touch `rag_defaults.top_k` when RAG is unused (794 ms cold build);
  - **manifest parse cache:** `MCP/server.py` is parsed 3× per call.
- **Also:**
  - world-info matching gets a substring prefilter plus compiled regexes: 62 ms at 300 entries and 219 ms at 1,000, against 5–15 ms;
  - the credential sanitizer runs once per message, memoized by content hash (today 5–7 whole-transcript passes per send, 24–27 per Capture-On provider call);
  - terminal persistence (a ~53-statement write transaction) and trace reservation (O(conversation) inside `BEGIN IMMEDIATE`) move off the loop;
  - `library_activity_snapshot` becomes linear.
- **Tasks:** TASK-32804.12, TASK-22205, TASK-31505.
- **Gate:** Enter → provider dispatch, today 6.5–11.8 s.
- **Size:** L.

#### PERF-13 (TASK-33272): Console idle, tick and streaming render (P0 ×2, P1 ×11)
- **Changes:**
  - **Credential/readiness polls:** make the 4 Hz poll event-driven, with the subscription cache publishing an expiry signal. The same poll pattern is duplicated on 5 surfaces.
  - **0.2 s tick:** stop re-running `context_control_inputs` (durable snapshots, branch-memory SQL, a recovery config scope) every tick; memoize on transcript revision.
  - **Message rows (P0):** message-row class sync removes and re-adds managed classes every sync, doubly restyling the whole Markdown subtree each streaming tick (`console_transcript.py:1451`). Diff the class sets instead. Selection restyles the whole subtree too (TASK-26834).
  - **Rail and cards:** `left_rail.sync_model_recovery` forces a screen relayout every tick through an unconditional `Static.update` (TASK-21244); cache video card specs per message.
  - **Tray and chip:** the conversations tray rebuilds for tooltip-only age deltas; the cost chip re-captures the turn config at 5 Hz.
  - **Transcript actions:** use one MarkdownIt parse per reply, not 2–3 per row-plan pass.
  - **Run tick:** stop the whole-transcript snapshot taken twice per 0.2 s run tick.
- **Tasks:** TASK-32804.3, TASK-21244, TASK-26834, TASK-24300, TASK-22213.
- **Gate:** idle Console CPU and loop wakeups/s (today ~45/s); per-tick ms at 400 messages.
- **Size:** M–L.

#### PERF-14 (TASK-33273): Console typing, first paint and resume (P0, P1 ×7)
- **Changes:**
  - **typing pause (P0):** the debounced spend refresh stalls the loop 250–415 ms; move it inside a derivation scope and off the loop;
  - **composer layout:** the per-keystroke whole-screen layout has three removable triggers (TASK-21120);
  - **class dance:** the ADR-161 class-dance does synchronous stylesheet applies on unmounted widgets (composer mount 29 → 95 `update_node` calls);
  - **Character search:** debounce it and stop the double recompose; it makes 5 private-SQLite round trips per key;
  - **first paint:** construct `AgentRunsDB`, `ChangeTurnTracker` and the bridge off the loop (TASK-32804.10); move the launch-wake scan off the loop; build Canvas machinery only when Canvas is used (a 1.8 MB hashed profile snapshot today);
  - **resume:** move `LocalSkillsService.get_context` into a thread worker (~58 ms with zero skills);
  - **post-ready storm:** 5–12 s at 30–75% loop (TASK-2902);
  - **resume restyle:** investigate the 533-node restyle, 63% of a warm visit (upstream Textual cost; the lever is fewer nodes).
- **Stale standing finding:** TASK-24452 ("Console re-mints 559 widgets per visit") is **stale**. Console is a reused route since TASK-31520; update or close it.
- **Size:** M.

#### PERF-15 (TASK-33274): Agent runtime (P0 ×3, P1 ×14)
- **Changes:**
  - **worker model:** use a persistent agent worker thread or executor with one event loop and one pooled `AsyncClient`, instead of a new `_ModelCallLifeline` thread, a new loop and a new client per send (33.7 ms SSL plus a cold handshake). Tool calls today run on a **new bare thread each, leaking the DB handles they open** (P0).
  - **trace writes:** run them on a connected thread; they open and close a fresh connection per operation today (P0).
  - **MCP tool calls:** move them off the loop; each does ~130 ms of admission, governance and audit I/O, plus a 60–70 ms sync execution-log append.
  - **catalog probe:** fix `probe_initial_catalog`, which is O(N²) over cumulative prefixes.
  - **hydration:** conversation open runs three full-hydration derivations of the same history; the rail snapshot hydrates all runs twice per turn; the change-review projection hydrates everything to read two columns.
  - **misc:** the StreamGate re-scans its buffer per chunk (843 ms per 100 KB reply); nested `AGENTS.md` is re-walked per tool batch; `fs_grep` keeps scanning after `max_results`; the profile tool provider re-reads outside `read_operation` (13–30 connects per call).
- **Tasks:** TASK-231, TASK-22213.
- **Size:** L. May split into 15a (threads, loops, clients, trace writes) and 15b (hydration and algorithms).

### Wave 3: boot and first paint

#### PERF-16 (TASK-33275): `TldwCli.__init__` diet + dead boot work (P1 ×12)
- **Defer to lazy properties or the post-`_ui_ready` tier** (`Utils/boot_worker_policy.py`):
  - the 5 feature DBs (~340 ms);
  - `EvaluationOrchestrator` (Evals DB, YAML, TaskLoader);
  - Watchlists wiring (~60 ms of SQLite);
  - MCP stores and control plane (~58 ms, TASK-31510);
  - local persona and dictionary services (110–130 ms);
  - collections-capture wiring (~40 ms at mount + 0.1 s);
  - boot scheduling maintenance (full-table scans on the loop);
  - the notes-sync runtime owner (TASK-21247);
  - `TTSService`;
  - the File-Notes-git chain;
  - cold feature services (Actor Packs, Research, Writing, Chatbooks, Persona_Visual).
- **Delete:**
  - 18 interop services with zero consumers;
  - the dead `RichLogHandler` path (TASK-186);
  - the boot media-type prefetch with no consumer;
  - dead duplicate packages (`Sharing/`, `Outputs/`, `WebClipper/`);
  - the ~2,600 lines of dead `Event_Handlers` code on the Chat import (TASK-32807.3).
- **Owner decision D5:** delete the zero-consumer interop services, or lazy-load them?
- **Gate:** `TldwCli()` construct time (today 2.4–3.1 s at load 10); main-thread helper spawns before first paint (today 12).
- **Size:** M.

#### PERF-17 (TASK-33276): Boot import diet + pre-import ratchet paydown (P1 ×5)
- **Import fixes:**
  - TTS/STTS stack (43 modules, ~97 ms), forced by `@on` Message classes;
  - `tldw_profile_core` for one constant (~23–27 ms);
  - `requests` via 7 boot modules (~25–34 ms);
  - citation pydantic cluster (45–60 ms);
  - theme AA hue pinning for all 91–94 themes at import (20–35 ms): pin only the active theme, lazily;
  - `MCP/server.py` → gateway import;
  - HF `datasets` via the eager Evals orchestrator (TASK-32904);
  - the splash importing all 87 effects (~55 ms);
  - `Widgets.Console` package `__init__`;
  - entry-point image-protocol warm-up (PIL on every boot);
  - `app.py`'s function-body-only names (404 of 470) moved into functions (TASK-33011).
- **Ratchets:** pay the pre-import payload ratchet back under 500 modules (72 new modules since it was set; the library route alone is 125k/123k LOC) and pin it in `perf-guard.yml` via PERF-01.
- **Size:** M.

#### PERF-18 (TASK-33277): Server-mode `TLDWAPIClient` off the loop (P0)
- **Problem:** the first client build imports a 1,257-model schema surface (575–854 ms) on the event loop, 0.1 s after first paint.
- **Fix:**
  - build or pre-warm it in a thread;
  - `defer_build=True` on the ~516 slice classes;
  - share one pooled client instead of ~30 private pools and SSLContexts;
  - stop re-reading `mcp_server_targets.json` per `build_client()`;
  - stream binary downloads.
- **Tasks:** TASK-285.
- **Size:** M.

#### PERF-19 (TASK-33278): Boot CSS paydown
- **Problem:** boot CSS sits at 607,640/608,090 B (450 B headroom) and the bare-type-subject ratchet at 273–274/274.
- **Fix:** give research, settings-theme and lab CSS a `ScreenOwnedSplit`; move modal wide-tier rules out of the Console module. The boot-time CSS staleness check costs 80–134 ms per source-tree boot, against a documented ~0.3 ms (TASK-18910).
- **Size:** S–M.

### Wave 4: network

#### PERF-20 (TASK-33279): HTTP client reuse (P1 ×3)
- **Changes:**
  - a per-provider pooled `requests.Session` in `LLM_Calls/hosted_chat.py:710`: 15.2 ms against 0.9 ms reused on loopback, plus the real TCP/TLS handshake per send and per agent step;
  - a cached `ssl.SSLContext` from `Utils/tls_trust.httpx_verify()` instead of `True` (12–24 ms per client, ~15 loop sites);
  - watchlist feeds on a pooled client;
  - Research "Ask Follow-up" off the loop: it runs a blocking non-streaming `chat_api_call` inside `async def` (P1, easy; its sibling `_default_gap_fn` was already fixed);
  - `webbrowser.open` off the loop (≥73 ms osascript on macOS);
  - llama.cpp per-send probe TTL;
  - image/video generation polling on a reused client.
- **Size:** M.

### Wave 5: screens (parallelizable)

- **PERF-21 (TASK-33280) Library** (P0 ×3, P1 ×22). The largest screen PR; split by canvas if needed.
  - Most destination switches still `await self.recompose()` on the whole screen (TASK-281).
  - The ingest registry listener does O(queue) work and remounts the whole queue panel per notification, so a folder submit is O(N²), with deep-copies 3× per mutation (TASK-32804.5, TASK-31583).
  - The workspace-depth build issues ~170 admission-wrapped full-scan registry reads on the loop (P0).
  - Folder Files re-walks the whole notes folder every **1.5 s** with pathlib: 41–49 ms at 500 files, 175–500 ms at 5k. It also keeps polling while hidden. Fix: scandir, change detection, visibility gating.
  - Search/RAG keystrokes run 8 whole-screen DOM scans plus a provider gate.
  - Notes-editor keystrokes run 6–7 DOM walks (TASK-32804.11).
  - Notes canvas `sync_state` always recomposes; the hidden Markdown preview is mounted on every open.
  - The conversation reader's progressive load is O(N²).
  - Collections run two whole-screen recomposes per interaction (41 sites).
  - Review-set `]`/`[` costs 4 transactions (TASK-32804.4).
- **PERF-22 (TASK-33281) Settings + Personas** (P1 ×4, P2 ×27).
  - Make both routes reusable (owner decision D4, TASK-24452 follow-through): Settings is 356–513 ms loop-busy per visit, Personas 0.7–1.0 s.
  - The Settings category switch re-mints both panes (215–900 ms) plus a mount-echo `Changed` storm; mount the category pane targeted instead.
  - Personas and Roleplay lists remount every row per debounced keystroke, page or sort. Use `OptionList` virtualization.
  - Debounce the inspector search, which runs a `BEGIN IMMEDIATE` per key.
- **PERF-23 (TASK-33282) MCP workbench** (P1 ×7).
  - Split the monolithic `_sync_children`, which re-reads stores and rebuilds 5 DataTables per interaction (TASK-32804.7).
  - Stop running two full passes per warm visit.
  - Arrow keys do guarded store reads plus an inspector remount.
  - Space in the permissions matrix costs ~100 ms.
  - Filter inputs rebuild the DataTable per key with no debounce.
  - The 4 Hz save poll does an unconditional `Static.update`, causing a relayout 4×/s.
- **PERF-24 (TASK-33283) Watchlists + Schedules** (P0 ×2, P1 ×7).
  - **Schedules:**
    - a cursor move runs 3 sync `ScheduledTasksDB` reads on the loop (132 ms, P0);
    - each visit runs ~145 ms in `on_mount` before first paint plus ~385 ms per visit;
    - `SchedulingService` and `SyncEngine` are sync sqlite behind `async def`;
    - the results overlay runs 3 queries on the loop;
    - mark-all-read is N+1 with 2 connection opens per result.
  - **Watchlists:**
    - the tree remounts per count refresh (one Button per node);
    - the inspector and content panes recompose per selection, including j/k;
    - the notifications pane rebuilds per cursor key.
- **PERF-25 (TASK-33284) Other screens** (P0, P1 ×14).
  - Evals re-pivots all runs and full-scans `eval_results` on the loop per visit, and the SnippetEditor has no cap.
  - Change Review runs git subprocesses and step-log reads on the loop, and re-lays out a 2,000-line diff per arrow key.
  - Speech Playground re-mints 161 widgets per visit and deep-copies the whole settings tree on mount.
  - The video player converts full-resolution frames on the UI thread at 24 fps.
  - Research sources search has no debounce.
  - Splash effects emit per-cell Rich markup that `Static.update` parses every frame.
  - Dictation start/stop runs on the loop.
  - The warm Console resume restyle, if not handled in PERF-14.

### Wave 6: feature-scoped and data-scaled (parallelizable)

- **PERF-26 (TASK-33285) Terminal** (P1 ×2).
  - Each frame re-projects every cell of every session: 11–20 ms per session per frame, twice per keystroke.
  - The output parse caps at 0.19 MB/s under the GIL.
  - Per-session polls run at 200 Hz (runtime) and 100 Hz (input flush), on top of the 50 Hz ownership scan (TASK-31503).
  - Fix: dirty-line tracking and event-driven wakeups.
- **PERF-27 (TASK-33286) Notes sync + Personal Context** (P1 ×11).
  - **Notes sync:**
    - the identity fallback does O(bindings × files) sha256 work on the loop (229 ms at 200/3k, 2 s at 500/10k);
    - `observe_root` and planning run on the loop;
    - the organization inventory is O(n²);
    - the replica's FTS delete by UNINDEXED columns scans the whole index per upsert.
  - **Personal Context:**
    - the Settings ▸ My Profile load runs without `read_operation`: **346 hardened connects, ~16 s**;
    - each interview answer costs ~117 connects plus keychain calls (~5.5 s);
    - the repository list is N+1;
    - ABSENT and DISABLED profiles still pay connects and a keychain hit per send (TASK-31504, TASK-32370, TASK-33081).
- **PERF-28 (TASK-33287) DB query hygiene** (P2 ×13).
  - The remaining N+1s and unbounded `fetchall`s.
  - Low-selectivity single-column indexes (10 on `deleted` alone) that the stat-less planner picks.
  - The startup trace-recovery scan of the `calls` table.
  - The launch-wake `agent_runs` full scan per boot.
  - The media `sync_log`, which is never pruned and logs full content twice per ingest.
  - The review-set 2N+1 listing (TASK-31508).
  - Follow the CLAUDE.md plan-pin rule for any new index.
- **PERF-29 (TASK-33288) Memory growth + data-scaled algorithms** (P1 ×4).
  - **Unbounded growth:**
    - the realtime mic tap keeps every frame (~173 MB/h, unbounded);
    - the dictionary version-history sidecar rewrites the whole file per edit.
  - **Whole-conversation work:**
    - opening a conversation loads every image BLOB of every branch;
    - the normalized trace re-base64s every history image per provider call (27 ms / 55 MiB peak).
  - **Quadratic or cubic algorithms:**
    - the meeting sink rewrites JSONL per segment (O(n²), 10.5 s CPU for a 1,500-segment meeting);
    - the Console model picker is O(M²) per keystroke (90 ms at 2,000 models, against a 4,096 cap);
    - chunking offset synthesis is O(N·S), cubic on fallback (13 s for 100k words);
    - the fallback vector store does O(N) list lookups (5.2 s to index 10k docs).
  - **Repeated work:**
    - speech models are cached per `TranscriptionService` instance, so **every Mic press reloads the model**;
    - parakeet decodes whole files as float64.
  - **Unbounded caches and logs:**
    - web_fetch cache (bounded by count, not bytes);
    - chunking SecurityLogger;
    - reranker cache (TASK-32810.11).
- **PERF-30 (TASK-33289) Cold-feature and hygiene sweep** (P1 ×8, P2 ×30, P3 ×43; 24 of these unverified). TTS, Audio, STT, Evals, Chunking, Backup_Recovery and interop P2/P3 items, including:
  - the legacy TTS request importing all 7 backends on the loop (torch via higgs);
  - `AudioService.convert_audio` doing sync decode/encode inside `async def`;
  - the model-catalog refresh fsyncing on the loop every launch;
  - the Environment tier spawning ~10 git processes every 10 s (TASK-31628);
  - the web-search relevance loop being serial with deliberate sleeps.
  - Lowest priority. Re-verify the unverified items first (§7).

---

## 5. Owner decisions needed

Backlog: PERF-01..PERF-30 are filed as **TASK-33260..TASK-33289** (PERF-NN = TASK-(33259+NN)), labelled `perf-audit-2026-09`, with dependencies following the waves.


| # | Decision | Blocks | Recommendation |
|---|---|---|---|
| **D1** | Amend ADR-126 to allow generation-scoped admission evidence (re-check identity via held descriptors plus per-component identity comparison, not a re-walk from `/` per call; admission per connection or generation instead of per transaction) | PERF-08 | Yes. It is the single largest lever, and the invariant (detect a moved, replaced or re-permissioned admitted directory or ancestor before use) can be kept with held fds plus per-component `lstat` identity checks, or `openat` through the held fds. |
| **D2** | Accept a per-generation memo of `get_user_data_dir` / DB paths / sensitive-path context with a per-call full-chain identity re-check (every verified ancestor) | PERF-07 | Yes, provided the re-check covers every ancestor; a leaf-only check would miss ancestor re-permissioning. It mirrors TASK-32804.1's accepted reasoning. |
| **D3** | GC policy ADR (`gc.freeze` after ready and pre-import; thresholds; optional idle collect) | PERF-11 | Yes, after PERF-05 lands the leak fixes. |
| **D4** | Make the Settings and Personas routes reusable (the TASK-24452 owner call, open since 08-29) | PERF-22 | Yes. Console and Library reuse already work. |
| **D5** | Delete the 18 zero-consumer interop services and the dead duplicate packages, rather than lazy-load them | PERF-16 | Delete; lazy-load only what has a planned consumer. |
| **Q3** | ADR-125: validate once per DB per process (helper leases) rather than per connection? | optional PERF-09 follow-up | Only if PERF-09's reuse leaves first-visit costs visible. |

---

## 6. Existing tasks this plan absorbs

71 open or recent tasks overlap findings. Mapping is in `data/pr_stats.json` (`known`). The ones to act on:

- **Reopen or extend:**
  - **TASK-32804.1** (fast path covered 1 of 16 guarded entry points) → PERF-06/07;
  - **TASK-278** (Done, but its rewrite is the P0 search regression) → PERF-02.
- **Re-scope with new cost evidence:**
  - **TASK-24457**: connections are now ~70× more expensive → PERF-09;
  - **TASK-31501**: the tick now spawns a helper → PERF-10;
  - **TASK-31502**: the quiescence tax is now dwarfed by admission → PERF-08/28.
- **Stale; update or close:**
  - **TASK-24452**: Console no longer re-mints; the remaining cost is the resume restyle → PERF-14; the Settings and Personas reuse part → PERF-22.
- **Absorbed:**

  | PR | Tasks |
  |---|---|
  | PERF-01 | TASK-32644, TASK-23155, TASK-31816, TASK-32809.1 |
  | PERF-03 | TASK-275, TASK-31806 |
  | PERF-06 | TASK-32804.3, TASK-24454, TASK-2902 |
  | PERF-11 | TASK-31966 |
  | PERF-12 | TASK-32804.12, TASK-22205, TASK-31505 |
  | PERF-13 | TASK-21244, TASK-26834, TASK-24300, TASK-22213 |
  | PERF-14 | TASK-21120, TASK-32804.10 |
  | PERF-16 | TASK-31510, TASK-21247, TASK-186, TASK-32807.3, TASK-33011 |
  | PERF-17 | TASK-32904, TASK-1378 |
  | PERF-18 | TASK-285 |
  | PERF-19 | TASK-18910 |
  | PERF-21 | TASK-281, TASK-32804.4, TASK-32804.5, TASK-32804.11, TASK-31583 |
  | PERF-23 | TASK-32804.7 |
  | PERF-26 | TASK-31503 |
  | PERF-27 | TASK-31504, TASK-32370, TASK-33081 |
  | PERF-28 | TASK-31508, TASK-21593 |
  | PERF-29 | TASK-605, TASK-32810.11 |
  | PERF-30 | TASK-31628 |

---

## 7. Coverage, confidence and gaps

- **Coverage:**
  - all 95 finder items completed: 76 slices covering 100% of files, 14 sweeps, 5 probes;
  - 953 of 1,043 findings were adversarially verified;
  - **90 remain unverified**, almost all in cold slices: Audio 13, COLD-MISC-3 11, Chunking/TTS/Backup_Recovery/COLD-MISC-1/-2/HOT-MISC-2 8 each, Evals 7, TTS#2 6, COLD-MISC-4 5. The verifier hit the account usage limit twice, and the resume ended with the session. They are marked `unverified` in the appendix. Most land in PERF-29/30. Re-verify before implementing.
- **Second-lens pass did not complete.** The planned second measure/trace pass on confirmed P0/P1s never finished (usage limit). Mitigations:
  - most P0s already carry **measured** costs from the finder or probe that found them;
  - the keystones (R1, R2, GC) were measured **independently by 5–10 agents** with consistent numbers;
  - still, every PR's first step should re-measure its items against the PERF-01 guards before changing code.
- **Refute rate is low:** 22 of 1,043 (2%), while severity changes were common (~115). Treat P2/P3 severities as approximate.
- **Load inflation.** Every probe ran with load averages of 10–50 from other sessions, so absolute ms are inflated ~1.5–3×. Syscall, spawn and call counts are load-independent.
- **Path depth.** The probe profiles sat at path depth 12–14 against ~4–5 for a real `~/.config`, so **admission milliseconds scale by ~0.3–0.46 for a real profile**; the fit is ~2.0 ms + 0.71 ms per component. Call counts do not scale. Even at real depth, a warm `load_settings` is ~5–8 ms and `get_user_data_dir` ~20–30 ms per call.
- **TTI comparison.** 7.5–9.9 s against the 09-04 review's 2.84 s is **not a paired comparison**: different load and different path depth. The regression direction is certain; its exact size is not.
- **Probe artifacts lost.** The scratchpad (probe scripts, seeded DBs, logs) was wiped when the session restarted. Numbers survive in `data/findings.json` and `data/items.md`; the scripts would need rewriting to re-run the same probes.
- **Possible functional bug, outside perf scope.** In the Console send probe, repeat sends dispatched (SENT) but appended nothing and never reached the provider (queue `DRAINING/RELEASED`). It may be a probe artifact. Worth a separate look.

## 8. Files

- `report.md`: this report.
- `appendix-issues-by-pr.md`: all 905 unique issues by PR (severity, status, location, tasks, corroboration, measured).
- `data/findings.json`: all 1,043 findings with verifier verdicts, reasons, costs and fix sketches.
- `data/issues.json` (generated, not committed): the deduplicated issues with their PR assignment. Recreate it with `classify.py` → `dedupe.py` → `plan_map.py`; the full rebuild order is in `data/extract_journal.py`.
- `data/items.md`: every agent's summary, clean-areas list and census tables (the timer census, recompose census, DB index-coverage census, HTTP-client census, import-time top-40 and so on).
- `data/pr_stats.json`: per-PR counts and the task ↔ PR map.
- `data/extract_journal.py`, `classify.py`, `dedupe.py`, `plan_map.py`: rebuild everything from the workflow journal.
