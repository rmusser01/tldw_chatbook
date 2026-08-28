# Holistic performance review — 2026-08-27

**Pin:** dev `c6218918d1`. **Question:** the app "slowed down recently and users are complaining" —
general sluggishness, input/click lag, slow startup and screen switches.

**Scope: delta.** Two prior reviews already ran — [2026-08-22](2026-08-22-holistic-perf-review.md)
(pin `35d4bf3a1`, 30 fixes shipped) and [2026-08-24](2026-08-24-holistic-perf-review.md) (pin
`a71e62e4b`, 29 fixes shipped, burn-down merged 08-25/26). This review deliberately re-measured
rather than re-read, and excludes both reviews' verified-clean lists.

**A scoping correction, recorded because it shaped the briefs.** This review was commissioned
against the 08-22 pin — 675 commits. The 08-24 review already covered most of that window,
including three files named in the lane briefs by name (`notes_sync_executor.py`,
`ChaChaNotes_DB.py` v49→v50, `Actor_Packs/importer.py`). The genuinely un-reviewed window is
**`a71e62e4b..c6218918d1`, 178 commits**. One lane caught this mid-flight and re-ranked its
findings against the correct baseline.

---

## The finding that explains the complaint

**Every printable keystroke re-derives the entire next request to price a tooltip.**
`1d59f96def` (#2114, merged 2026-08-26) added `UI/Console_Modules/send_price.py`;
`Widgets/Console/console_composer_bar.py:1737-1741` calls it from `sync_action_state`, reached from
`insert_text` on every key once the draft is non-empty.

Timed **inside `ChatScreen.on_key`** (not through `pilot.press()`, which inflates), interleaved ×3:

| session | pricer on | pricer off | |
|---|---|---|---|
| empty | 0.897 ms | 0.405 ms | 2.2× |
| 400 messages | **3.564 ms** | 0.419 ms | **8.5×** |
| 400 msgs, session not flagged `has_user_work` | **8.0 ms** | 0.110 ms | **73×** |

cProfile: **1,200 message copies per keystroke** — `dataclasses.replace` on a 32-field dataclass,
~65% of the call — from three full-session snapshot passes.

**The cache is structurally incapable of hitting.** `TokenEstimateCache(max_entries=1)` keys on
`(model, provider, tuple(rows))` and the last row *is the draft*, so the signature changes every
character. Measured: **20 calls, 0 hits, 20 misses.**

At this review's ×3–5 constrained-hardware multiplier: **11–40 ms between a key press and its echo**,
scaling with conversation length. Four days old. Filed as **TASK-23018**.

---

## What else is real

### Screen rebuilds got more expensive; screens are still never cached

| | 08-24 base | tip |
|---|---|---|
| LibraryScreen widgets per visit | 67 | **157 (+134%)** |
| ResearchWorkspaceScreen (new, F10) | — | **693 per visit** |
| Library DOM queries per resize frame | 31.6 | **84.2** |

**544 of the Research screen's 691 mounted widgets (79%) sit inside `display=False` subtrees** —
constructed, mounted and CSS-matched on every visit, never painted. Cause is eager slot pools:
`Research_Workspace_Modules/source_list.py:196` composes `MAX_VISIBLE_SOURCE_ROWS = 25` slots × ~13
widgets **on an empty profile**, plus 20 receipt slots. One whole-screen recompose: **1.73 s**.

**A negative finding that overturns the obvious reading:** whole-screen `recompose=True` sites went
**down**, not up — `library_screen.py` 126 → 96, package 420 → 388. The +6,340 lines did not add
recompose sites; they made each existing one **2.3× more expensive**.

### Idle burn is fixed for existing users — and terrible for new ones

11 of 14 destinations idle at **0.03–0.26% of a core**, ~7× better than the 1.8% the 08-22 review
recorded. **Idle cost does not explain the complaint for a configured user.** Two exceptions:

- **The setup-modal snow costs 4.33% of a core vs 0.096% with the tick neutralised (45×)** — the
  screen every *new* user lands on. TASK-21134's two fixes are both intact and are not the issue:
  `_tick` itself is 30 ms/15 s, while `Screen._on_timer_update` is **555–712 ms/15 s (15–19 ms per
  repaint)** because the backdrop is a full-viewport `Static` that dirties all 483 widgets. Measured
  identical against the 08-24 pin, so **pre-existing, not a delta**. On constrained hardware:
  **13–30% of a core, permanently, for a decoration, before the user types anything.**
- **Four invisible `ProgressBar`s are 88% of the Lab screen's idle CPU** — 0.616% → 0.076% (8×) with
  the framework clocks neutralised. **960 of 1018 timer fires in 15 s change zero pixels.**
  `ProgressBar(total=None)` makes Textual's `Bar` indeterminate → a **15 Hz `set_interval` forever**,
  and `display = False` does not stop it: Textual gates only the *repaint* on `is_on_screen`, never
  the timer. Six live instances found (Lab ×4, Library, Personas at 16 Hz), 13 more `ProgressBar(`
  and 6 more `LoadingIndicator()` sites exist.

### Boot accretion

Closure **657 modules against a 660 budget**. **15 of the 20 new modules come from one import
statement**: `Library/library_ingest_jobs.py:77` reaches a stdlib-only validator *through*
`Research_Workspace/__init__.py`, which eagerly re-exports the tree — including `server_adapter`,
which imports a 26-model pydantic module for **one integer constant**
(`MAX_WORKSPACE_SOURCE_OWNER_ROWS = 10_100`). Cost: **≈48 ms and 8 modules** for a validator that
costs 6 ms standalone. Time-to-interactive regressed **~65 ms** against an A/A control establishing a
30 ms noise floor.

### Storage, not latency

**Exchange capture stores the whole conversation on every send, permanently.** `messages_payload` is
on the capture allowlist, so the blob grows 2.8 KB at turn 1 → **145.4 KB at turn 200**, totalling
**15.40 MB for one 200-turn conversation**. Default **on**. No retention path: the only purge is
user-invoked and filtered to `capture_detail = 'full'`, and nothing hard-deletes conversations, so
the `ON DELETE CASCADE` never fires.

### Notes sync

`to_thread(lambda: asyncio.run(coro))` **9 times per synced note** (684 µs each vs 0.1 µs for
`await`). `observe_root` re-reads every file and re-selects every note on every sync — at N=1000 with
**nothing changed**: **350–370 ms, 1,000 SELECTs, 1,017 file opens, worst loop stall 36–48 ms** — and
it runs at boot. ~1 abandoned SQLite connection per note (1,010 opened, 0 closed).

---

## The structural finding, which outranks any individual fix

**The guards are being consumed faster than they are built.**

| guard | budget | now | headroom |
|---|---|---|---|
| boot import weight | 660 modules | 657 | **3 (0.5%)** |
| `_ui_ready` census | 970 | ~950 | ~20 — **family assertion already RED** |
| boot CSS bytes | 860,000 | 842,236 | **17,764 (2.1%)** |
| pre-import payload | 500 / 380k LOC | 481 / 368,814 | **19 (3.8%) / 11,186 (2.9%)** |

Every budget was pinned on 2026-08-25 "just above reality". Two days later all four are within
2–4% of breach; at observed merge rates each breaches within a day or two.

**The sharpest instance:** the guards forbidding `Chat.trajectory_export` on the first-paint path
were written 2026-08-25 and **breached within ~24 hours by `c6218918d1`, the current tip** — which
touched neither guard file. `chat_screen.py:52-57` carries a comment explicitly forbidding this;
#2126 routed around it through a file the comment does not name. Three module-scope edges now reach
it, each importing a **three-member enum** that drags 1,463 LOC. **Fixing one edge buys nothing.**

**And the instruments have blind spots that hid the two largest idle clocks.** The timer census
(`Tests/Architecture/test_timer_path_static_update_inventory.py`) is **green while blind to both**:

- It matches `set_interval` as an exact callee name. `UI/Console_Modules/realtime.py:345` now spells
  a 10 Hz clock `self._set_interval(...)` through a constructor-injected callable — it **silently
  left the census**, and the root count did not move (35 → 35) because another root arrived and the
  two cancelled.
- It parses only `tldw_chatbook/**.py`, and **no package file assigns `auto_refresh`** — the 15 Hz
  `ProgressBar` clocks live inside `textual/dom.py`. Structurally invisible.

The CSS allowlist ratchet is also **red again on pristine tip**, with two new defectors — the
identical failure TASK-22212 fixed 48 hours earlier.

---

## Verified clean — do not re-fix

- **First interaction is byte-identical** to the 08-24 pin: 19 keystrokes → **0 SQL statements, 21
  layout passes**, on both arms. 21118's memo, 21692/22218's blink gates and 22203's rail gate hold.
- **Boot determinism improved**: mount-phase SQL **255 → 90** statements; `sqlite3.connect`, `open()`
  and `Thread.start()` counts identical.
- **The migration chain is healthy**: v46→v51 on a **334 MB / 100k-message** profile takes **547.5 ms**
  in one transaction; v49→v50 is 5.44 ms and idempotent. TASK-21100's deferral held.
- **Store census clean**: 15/15 stores read back **WAL + `synchronous=NORMAL`** live; zero DELETE/FULL.
- **`Chunking` and `RAG_Search.simplified` remain 0 at `_ui_ready`** — TASK-21731's guarantee holds.
- **No new repeating clocks on any Console surface**; `TldwCli.__init__` did not grow (8 lines of
  attribute init); `on_mount` got *smaller*.
- **The CSS parse-cache cliff is NOT the cause.** 47 stylesheet sources at both ends of the window,
  unchanged across 653 commits; warm parse 0.1–0.8 ms; **zero parse calls on warm screen switches**;
  17 slots of headroom against the `LRUCache(64)`. The mechanism was proved real by forcing it — at
  78 sources parse goes to **154 ms and never recovers** — but the app is not near it. This was the
  reviewer's leading hypothesis and it is disproved.
- **The 5 red `test_rag_citation_provenance_benchmark.py` tests are a stale test double**, not a
  production defect: the rejection is TASK-22030's own message, i.e. that fix working correctly
  against a hand-rolled persistence stub.

## Corrections made during this review

1. **The scope was wrong** — see the note at the top. A second review had already covered most of the
   commissioned window.
2. **The CSS cliff hypothesis was disproved**, above.
3. **A lane refuted another lane's "clean" verdict** on exchange capture, which had been built from a
   hand-made input rather than the real caller's kwargs. The finding stands at 15.40 MB.
4. **cProfile misattributed 460 ms** to a boot worker that an un-profiled `Handle._run` stall census
   prices at ≤7 ms. Reported as *not a finding*.
5. **A 150 ms in-migration import was nearly filed**, then found to be already resident in the app's
   closure — real marginal cost ~0. Not filed.
6. **Two `_timers` entries that looked like hot clocks** are one-shots Textual leaves in the list
   after firing. Reading `_timers` requires `_repeat`, not `_interval`.

## Method notes worth keeping

- **Time inside the handler**, not through `pilot.press()+pause()`, which is harness-inflated.
- **Interleave A/B arms.** This machine runs other sessions' tests concurrently; a non-interleaved
  comparison in a prior review inverted its own result.
- **The test suite installs a process-global CSS parse cache** that silently hides every parse
  measurement — set `TLDW_TEST_CSS_CACHE=0`.
- **The 08-22 pin cannot boot the test harness at all** (the disarmed-guard defect task-21106 later
  fixed), so A/B against it requires the earliest bootable post-pin commit.
- `-p no:randomly` is a **no-op** here — pytest-randomly is not installed.
