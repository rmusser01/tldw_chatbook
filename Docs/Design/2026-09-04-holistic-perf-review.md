# Holistic performance review — 2026-09-04

Seventh holistic performance review. Pin: dev `f51fcaf204` (1,119 commits since
the 2026-08-30 review's pin `0ef6f3fd4e`). Baseline arm for paired
measurements: `b62407e258` (2026-09-01, the merge of PR #2281 — i.e. *after*
the previous review's CSS wins landed, so deltas measure the last ~3.5 days of
feature work, not the previous review's own improvements).

Commissioned with the same prompt as the 2026-08-22/24/27/29/30 reviews, with
one new datum: **users report the app has recently slowed down.**

Tasks filed: **31500–31511.**

## Method and its limits

- All probes ran against detached worktrees of the two pins, isolated config
  (`TLDW_CONFIG_PATH` + redirected `HOME`/`XDG_*`, splash disabled,
  `first_run.setup_completed = true`, `TLDW_TEST_MODE=1`,
  `TLDW_SCREEN_PREIMPORT=0` — the same environment as
  `Tests/Performance/test_ui_ready_module_census.py`).
- **The machine carried load average 7–9 with nine concurrent pytest processes
  from other sessions for the whole review.** Absolute wall numbers are
  inflated roughly 1.5–2× against prior reviews' quiet-machine numbers and are
  NOT comparable across review dates. Every regression verdict below therefore
  comes from interleaved same-day paired arms (ABBA/BAAB), and per-switch
  process-CPU is reported alongside wall time because it attributes better
  under load.
- Boot runs: cold subprocess per run, profile pre-warmed once per arm (the
  first boot on a fresh profile pays one-time schema creation — 08-29 trap).
- A first probe run measured `_ui_ready` at ~9.3 s and was **discarded**: the
  scratch config had not disabled the splash screen or marked first-run setup
  complete. Probe configs must match the canonical census config exactly
  before any number is comparable.

## 0. Ratchet state on pristine dev (ADR-097)

| guard | state | value |
|---|---|---|
| import weight | GREEN | 637/660 (headroom 23) |
| ui-ready module census | GREEN | 966/972 (headroom 6) |
| boot worker census | GREEN | pass |
| CSS fastpath (incl. bare-type rule ratchet) | GREEN | pass |
| **boot-parsed CSS bytes** | **RED** | **821,753 / 806,000** |

The CSS byte ratchet is red on pristine dev for the fourth review running.
PR #2281 paid it down to 780,368 B on 2026-09-01; dev added **+41,385 B of
first-paint CSS in ~4 days**. Culprits (from the guard's own diff):
`features/_scheduling.tcss` 5,994 → 16,165 (+10,171, nearly tripled),
`components/_agentic_terminal.tcss` +5,342 (still growing despite the 25812
split), `screen_agentic_console.tcss` +4,530, `core/_variables.tcss` +1,558,
plus ~19 new widget-default segments (tool-pack import modals ≈4.7 KB,
profile interview, settings personal context, library media canvas, review-set
picker, scheduling views). → **TASK-31500** (paydown). The consumption
pattern ADR-097 exists to stop is still running; the byte guard is still not
in `perf-guard.yml` (task-24459), so breaches keep landing silently.

## 1. Regression verdicts (paired arms, current vs 2026-09-01)

| metric | current `f51fcaf204` | baseline `b62407e258` | verdict |
|---|---|---|---|
| cold boot → `_ui_ready` (4 runs/arm) | mean 2.84 s | mean 2.68 s | no regression (Δ within within-arm spread of ±0.5 s) |
| import of `tldw_chatbook.app` | 0.95–1.5 s | 0.93–1.1 s | no regression |
| switch tour, Chat ping-pong CPU (4 visits) | mean ≈924 ms | mean ≈983 ms | no regression |
| switch tour, Library ping-pong CPU | mean ≈469 ms | mean ≈555 ms | no regression |
| typing, CPU/key on empty Console (40 keys) | 32.6 / 36.6 ms | 28.3 / 56.3 ms | no regression (high variance under load) |
| idle CPU on Console, 15 s | 46–61 ms (0.3–0.4 % core) | 46–62 ms | no regression |
| ChaChaNotes per-statement overhead | **1.94 µs** | **0.82 µs** | **+137 % — regressed** (§3) |
| ChaChaNotes per-`transaction()` block | **23.3 µs** | **18.2 µs** | **+29 % — regressed** (§3) |

**The interactive hot paths measured by prior reviews did not regress in this
window.** What DID change is background/ambient behavior (§2–§4) and
per-statement DB overhead (§3) — the kind of cost that shows up as "the app
feels slower lately" on battery, on slower disks, and under concurrent
activity, without moving any single foreground metric.

## 2. NEW: a 1 Hz write-lock poll on the main DB, forever (TASK-31501)

`Chat/console_runtime.py:994-1152` (`_schedule_legacy_trace_maintenance`) —
armed for every real profile whenever the Console chat store binds a DB with
a `transaction` attribute (`console_runtime.py:913`, `:989`). The loop:

- calls `asyncio.to_thread(maintenance.run_batch)` then sleeps 1.0 s in every
  steady-state branch → **~1 wake/second for the life of the process**;
- `run_batch` (`Chat/console_trace_maintenance.py:735`) opens
  `self.db.transaction(immediate=True)` — an **immediate transaction acquires
  the write lock** — and runs 3 SELECTs *even when the migration is
  `logical_complete` and there is nothing to do* (the completion check lives
  inside the transaction);
- every 60 s it additionally runs `TraceGarbageCollector.current_graph_epoch`
  and, on epoch change, GC + physical compaction;
- there is **no idle gate, no visibility gate, and no config off-switch**
  (`legacy_normalization_enabled = callable(getattr(db, "transaction", None))`).

This silently breaks the "zero SQL at idle" property the 08-27 review's
notes-sync fix established, wakes the CPU every second (battery), churns a
thread-pool hop per second, and acquires the same write lock user sends
contend for. Introduced by the semantic-trace-ledger wave (commit
`6004b02038` and siblings); no task records the 1 Hz cadence as a decision.

Fix directions (either preserves behavior): event-driven wake — the exchange
writer signals when a new `message_exchanges` row lands, the loop parks until
then once `logical_complete`; or a steady-state backoff (1 s while provider
recently active, 30–60 s otherwise) with the completion check moved to a
read-only transaction.

## 3. NEW: every ChaChaNotes statement now pays a shared-lock tax (TASK-31502)

`DB/base_db.py:35-410` adds `SQLiteConnectionQuiescenceRegistry` (one
process-wide `Condition(RLock())` per DB file) and wraps the ChaChaNotes
connection so **every** `execute`/`executemany`/`executescript` runs
`begin_use()`/`end_use()` — lock acquire, release, and `notify_all()` — around
every single statement (`ChaChaNotes_DB.py:3339`, `:22895`), plus a second
pair around each `transaction()` block and a third around connection
acquisition. It exists so the trace-compaction VACUUM (§2's 60 s pass) can
quiesce all connections — a maintenance event that is rare to nonexistent in
a normal session — but the bookkeeping is unconditional.

Measured (file-backed DB, warm, 20k statements, reproduced twice per arm):
per-`SELECT 1` through `get_connection()` **0.82 → 1.94 µs (+137 %)**; per
single-statement `transaction()` block **18.2 → 23.3 µs (+29 %)**; raw
sqlite3 floor 0.51 µs both arms. Honest sizing: ~1.1 µs × even 10,000
statements ≈ 11 ms — **not** the user-visible slowdown by itself, but it is
an unconditional tax on the hottest layer of the app, it grows with
cross-thread contention (`notify_all` per statement on a shared condition),
and it multiplies with §2 (the 1 Hz loop's statements all pay it too).

Fix direction: an uncontended fast path — e.g. check a relaxed "quiesce
requested" flag before touching the condition variable, taking the full lock
protocol only while a quiesce is pending or in progress.

## 4. NEW: Terminal sessions scan the whole process table at 50 Hz (TASK-31503)

`Terminal/posix_backend.py:53` (`_PROCESS_POLL_SECONDS = 0.01`), `:968`
(`_monitor_owned_processes` wakes every 20 ms), `:1128-1200`
(`_default_scan_locked`): every tick enumerates `psutil.pids()` and calls
`os.getsid` + `os.getpgid` **for every PID on the system**, plus
`psutil.Process(...).create_time()` for candidates.

Measured on this machine (663 processes): **0.30 ms CPU per scan → ~1.5 % of
a core per open terminal session, continuously**, ~30k syscalls/s, even at an
idle shell prompt. Two terminal tabs ≈ 3 %, and each session runs its own
monitor thread. The whole `Terminal/` package is new in this window — a user
who adopted the feature gets a permanent background tax that would read
exactly as "the app got slower lately."

Fix direction: drop the monitor cadence to 250–500 ms (ownership bookkeeping
does not need 50 Hz), and/or scan adaptively (fast only briefly after spawn
activity on the PTY).

## 5. NEW: Personal Context pays hardened connects per send, even unconfigured (TASK-31504)

`Personal_Context/repository.py:419` opens a fresh owner-checked connection
per repository method (25 `closing(self._connect())` sites); each connect
walks the directory path from `/` with `dir_fd`/`O_NOFOLLOW` ownership checks
(`Utils/private_paths.py:1282`), prepares 3 sidecar files, and issues extra
PRAGMAs. `Chat/console_chat_controller.py:2769`
(`_compose_profile_tool_provider`, on the default agent send path) calls
`authorized_context_view` **twice** (consistency check) plus `list_scopes`,
`get_scope_authority`, `get_manifest` — and an unconfigured profile still
pays `status()`'s connects before failing closed; nothing caches the
negative result between sends.

Measured: connect ≈ 0.44 ms; `is_destroyed` ≈ 0.43 ms; `get_manifest` ≈
0.44 ms; **unconfigured per-send floor ≈ 1.75 ms** (off-thread). Small
today — but for a *configured* profile the same path is 6+ connects plus
**two full decrypts of the export snapshot per send**, which scales with
profile size. Fix: cache the locked/unconfigured status, build the view once
per send, reuse one connection across a logical operation.

## 6. NEW: opt-in custom-PII redaction spawns a subprocess on the event loop, inside a held write transaction (TASK-31505)

`Chat/console_trace_regex_worker.py:126-225` (`run_custom_pii_batch`) runs
`subprocess.Popen([sys.executable, "-I", ...])` and blocks on
`process.communicate(...)` (deadline 500 ms). Called (when
`pii_redaction_enabled` with a *custom* ruleset):

- from `console_provider_gateway.py:1133` via the synchronous
  `_complete_scoped_exchange` in the streaming `finally:` — **on the event
  loop**, once per completed/stopped exchange;
- from `console_chat_controller.py:8587` inside
  `_build_durable_trace_request`'s per-saved-row loop **while
  `transaction(immediate=True)` is held** (`:8559`) — a fresh interpreter
  spawn per message row with the write lock held, on the event loop.

Off by default (`RuntimeCapturePolicy.pii_redaction_enabled = False`), but
when enabled this freezes the UI on every send. Fix: pre-compute redactions
before the transaction opens, and move the spawn+wait off the loop. (The
worker-subprocess *isolation* is presumably deliberate for `-I` sandboxing —
keep it, relocate the wait.)

## 7. Smaller new items

- **Session-switcher modal (Ctrl+K) polls at 5 Hz** while open
  (`Widgets/Console/console_session_switcher_modal.py:55`, `:253`), and each
  tick recomputes the full active-session projection — iterating
  `store.sessions()` and calling `activity_for` **twice** per session
  (`UI/Console_Modules/workspace.py:2563-2774`) — unconditionally, just to
  compute a change fingerprint. O(open sessions) × 5/s; fleet users pay most.
  → **TASK-31506**.
- **Scheduler tick does sync file I/O on the event loop** every poll interval
  (30 s), for every user, always: heartbeat write (mkdir + mkstemp + write +
  rename; `Scheduling/scheduler/loop.py:380`, `scheduler_heartbeat.py:94`)
  and emergency-stop `read_text` (`loop.py:411`). Sub-ms on a healthy SSD;
  still the wrong thread — `asyncio.to_thread` both. → **TASK-31507**.
- **Review-set listing is an N+1**: `Library/review_set_service.py:174-198`
  issues 2N+1 statements (re-fetching each header row it just enumerated) and
  an unbounded `SELECT * FROM review_set_items` per set, to render a picker
  that needs a name and a progress summary. Off-loop, but scales with
  accumulated sets × items. → **TASK-31508**.
- **Media Trash canvas re-measures on every recompose**:
  `Widgets/Library/library_media_trash_canvas.py:167-244` chains
  `call_after_refresh(_measure_after_layout)` from mount, resize AND
  recompose; each pass is 3+ `query_one` + per-child geometry + a possible
  second pass — and the screen builds a fresh canvas instance per state
  transition, multiplying with the known per-visit fresh-screen cost.
  → **TASK-31509**.
- **`TldwCli.__init__` does sync JSON read (and possibly write) before first
  paint** for legacy MCP server-target sync
  (`app.py:8442` → `MCP/server_target_store.py:300-342`) — only for users
  with a legacy `tldw_api` base_url configured; small, but it is on the
  pre-paint critical path. → **TASK-31510**.
- **Agent-run webhooks spawn a thread + a fresh asyncio loop per delivery**
  and `load_settings()` from disk per run completion
  (`Agents/run_webhooks.py:~257`, `agent_service.py:3264`). Opt-in, low
  frequency. → **TASK-31511**.

## 8. Standing findings re-confirmed (not refiled)

- **Console visits cost ~0.7–1.25 s of CPU each and warm visits are no
  cheaper** (this review's tour, both arms) — the fresh-screen-instance
  architecture (task-24452, owner call open since 08-29) is still the single
  largest interactive lever in the app.
- **Typing on an empty Console costs ~30 ms CPU/key under load**, and the
  profile shows why it scales badly: 40 keystrokes drove ~41,000
  message-pump iterations — every keystroke wakes every mounted widget's
  pump for an idle/refresh check, so per-key cost is proportional to mounted
  widget count (~1,000 pumps with Console up). Same in both arms (not a
  recent regression). Reducing mounted-widget count (24452 and Console
  decomposition) is the lever; `textual._callback.count_parameters` also
  recomputed `inspect.signature` 40,892× in the window (upstream cache
  misses on bound methods — a possible upstream delegation like the 08-29
  fastpath, sized ~0.27 s/40 keys ≈ 7 ms/key here, load-inflated).
- Idle is otherwise still clean (0.3–0.4 % core in-harness; the §2 loop's
  1 Hz work sits mostly in the to_thread hop and DB layer).

## 9. Verified healthy / not filed

- Boot, switch, typing, idle: no regression vs 2026-09-01 (§1 table).
- Console transcript/streaming append path: untouched in this window
  (verified by lane sweep + git log).
- Collections capture/offline store, media browse/trash controllers, ingest
  pipeline: clean patterns (batched INs, to_thread, bounded reconcile;
  ingest untouched).
- Sync_Interop/MCP/Tool_Packs boot surface: module-scope imports are
  constants-only; heavy work correctly deferred (several ADR-097 "perf:"
  commits in-window did this proactively).
- Composer per-keystroke path: only a constant-work width helper added.
- Scheduling workbench timers: screen-scoped, stop on unmount.

## Appendix: probe inventory

Probes live in the session scratchpad (not committed): cold-boot milestones
(subprocess, 4 runs/arm), destination tour (8 destinations × cold/warm +
Chat↔Library ping-pong ×4, arrival/settled/CPU per switch), typing (40 keys,
CPU/key + cProfile attribution), idle (15 s CPU), ChaChaNotes statement/txn
micro-bench (20k/2k iterations, ×2 per arm), Personal Context op timings
(50 reps), terminal scan micro-bench (50 scans). Lane sweeps: five read-only
subagent lanes over `git diff 0ef6f3fd4e..f51fcaf204` (Chat; Console
widgets; Personal_Context/Tool_Packs/Sync_Interop/MCP; Library;
Scheduling/Agents/DB/app/Utils/Terminal), each reporting file:line findings
that were then re-verified in source and, where impactful, measured before
filing.
