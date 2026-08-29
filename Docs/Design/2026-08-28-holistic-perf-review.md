# Holistic performance review, 2026-08-28

Fifth full perf review (after `2026-08-22`, `2026-08-24`, `2026-08-27`, and the
earlier input-latency audit), commissioned with the same prompt as its
predecessors: users report the app has slowed down again.

**Review pin:** dev `3a3383123e`. **Burn-down branch:** cut from dev
`4da99a8849` (two documentation-only commits later; verified with
`git diff --stat`, no code between them).

Seven findings. **Six fixed in this cycle** (TASK-24300 … TASK-24305);
**TASK-24306's fix was implemented, found to be a validation regression, and
reverted** — see below. The finding stands; the approach was wrong.

---

## Headline

Typing one character into the Console composer rebuilt the entire
provider-readiness and session-settings object graph, discovered the result was
identical to the previous keystroke's, and threw it away. The equality gate that
follows (`_push_console_control_state_if_changed`) skips the DOM write but not
the compute — the derivation has already run by the time the gate is consulted.

On top of that, one leg of the derivation answered "does this session have any
messages?" by deep-snapshotting the entire transcript, so the cost grew linearly
with conversation length.

| per keystroke, 400-message conversation | before | after |
|---|---|---|
| `messages_for_session` calls | 3.27 | **0** |
| message snapshots allocated | 1,310 | **0** |
| `build_default_console_session_settings` calls | 3.25 | **0** |
| `build_console_settings_readiness` calls | 4.35 | **2.35** |
| draft-edit handler, interleaved A/B (3 rounds) | 16.61 / 11.74 / 10.97 ms | **0.727 / 0.734 / 0.706 ms** |

The fixed arm's variance collapsed along with the mean, because the term that
scaled with the transcript is gone rather than merely smaller.

---

## Method, and a correction to it

Everything was measured against detached worktrees with their own uv venvs.
Every probe asserts `tldw_chatbook.__file__` resolves inside the worktree it
means to measure — that assertion caught one cross-worktree run during this
cycle, exactly the editable-install trap the previous review recorded.

**Absolute `pilot.press` latency is quoted nowhere in this document.** Textual's
Pilot posts one callback per mounted widget per `pause()` — 81,862 dispatch
lookups for 40 keystrokes on a 488-widget screen — so press timings track widget
count, not app work.

**The correction this cycle produced, which the first draft of the review got
wrong.** The initial figures (6.51 ms/key empty, 39.70 ms/key at 400 messages)
were taken while the full `Tests/Performance` suite ran in the background. The
same unchanged code measured 1.31 and 11.27 ms/key on a quiet machine, and
later 3.80 then 6.75 ms/key for identical input twenty minutes apart. Load
average on this machine sits at 5–10 from concurrent agent sessions. **The
mechanism was real and the absolute numbers were not.** Call counts —
`messages_for_session` at exactly 3.27/key, `normalize_provider_config_key` at
exactly 250.5/key — reproduced on every run at every load level, and are what
the guards now pin. Recorded in `lessons-testing-evidence.md`.

---

## Findings

### 1. TASK-24301 — the provider graph is re-derived on every keystroke and screen entry

`_build_console_control_state` reads provider/model/settings, active session,
character, assistant identity, library policy, staged source count, tool and
approval counts. **It does not read the draft text at all**, which is why the
equality gate always finds the state unchanged; the code's own comment says the
Workbench state "does not move" with the draft.

A single *warm* return to the Chat screen cost 159.8 ms of app-side Python and
ran **18,938** `normalize_provider_config_key` calls, with
`_build_console_provider_selection_uncached` at 117.7 ms cumulative. The Library
screen's warm entry, for contrast, is 18.0 ms.

Fixed in two separable layers: the per-pass memo (`_console_derivation_scope`,
task-15452) widened to cover the session-settings leg, and a cross-pass memo for
the template defaults keyed on the config object's **identity** against a
retained reference.

**Deliberately not extended to `build_console_settings_readiness`**, which reads
`os.environ` for credentials. Caching readiness against a stale snapshot is
precisely the task-177 regression — a provider configured in Settings stayed
blocked until restart — and it was not worth re-introducing for the remaining
milliseconds.

### 2. TASK-24300 — emptiness checks deep-copied the whole transcript

`messages_for_session` materialises every stream buffer and deep-snapshots every
message. Four call sites used it as a predicate; one is on the keystroke path.

**The precondition was verified rather than assumed.** The guard above the hot
site short-circuits on `session.has_user_work`, and appending messages does *not*
set that flag — only renaming a session, replacing its settings, or persisting a
non-empty draft do. Typing does not set it either: the draft lives on the
composer widget, and `store.session_draft` stayed empty through a 10-key burst.
So this fired for resumed conversations and sessions on untouched defaults, not
for every user on every keystroke.

Fixed with O(1) `has_messages`/`message_count` and a lazy
`iter_messages_newest_first` for the two reverse scans.

### 3. TASK-24302 — a `partial()` wrapper blinded the boot-worker census

Red on pristine dev. The census recorded `kwargs.get("name") or getattr(work,
"__name__", "")`, and a `functools.partial` has no `__name__`.

The important part is not the one red worker. An AST census found **34
partial-wrapped `run_worker` sites in this repository, 22 passing no explicit
name** — every one of them would have collapsed into the same anonymous
`('', group)` identity, so the reviewed allowlist could not tell them apart. The
census now unwraps `partial.func` recursively; the boot-leg site is also named
explicitly.

### 4. TASK-24303 — nine modules crossed onto the first-paint leg

The ui-ready ratchet measured **972/970 and was non-deterministic — three of four
consecutive runs red**, because boot work lands concurrently with `_ui_ready`.

**Deferring in one importer made a different guard worse.** Deferring the shell
and virtual-CLI providers in `Chat/console_chat_controller` alone took the
pre-import payload from 379,497 to 381,696 LOC and turned that ratchet red.
Bisecting the edit rather than theorising showed why: the modules stopped being
resident before the registry walk and started being *charged to it*. The cost
had relocated, not gone. `UI/MCP_Modules/mcp_workbench` imports the same two
providers at module scope, and both had to defer.

Final: ui-ready **964/970 (headroom 6), deterministic across four runs**;
pre-import payload **378,930/380,000 (headroom 1,070, up from 503)**. No ratchet
constant was raised (ADR-097).

### 5. TASK-24304 — the config path was resolved 1,132 times per screen entry

`_get_effective_config_path` ran `expanduser` + `abspath` + `normpath` on every
call. The environment read stays live in the caller; the normalisation behind it
is memoised on **both** `TLDW_CONFIG_PATH` and `HOME` — `expanduser` reads `HOME`
at call time, so an override containing `~` resolves differently when `HOME`
moves, and tests move `HOME` routinely.

Measured on a warm Chat entry: **1,132 lookups → 1,132 cache hits → 0 path
normalisations.**

### 6. TASK-24305 — tiktoken was imported on the first line of every cold start

`tldw_chatbook/__init__.py` called `install_tiktoken_runtime()` at package
import, costing 19.6–29.1 ms of `import tiktoken.load` whether or not the
session tokenised. `Utils/token_counter.py` also imported tiktoken at module
scope, so deferring only the shim would have bought nothing.

**This was invisible to every guard in the repository**: all four boot ratchets
count `tldw_chatbook.*` modules only, so third-party import weight is outside
every budget. Profiling was the only way to find it. Whether that surface
deserves its own budget arm is an open owner call.

The wall-clock saving did not separate from noise on a loaded machine; the
honest evidence is the module's absence from the closure, verified in a
subprocess.

### 7. TASK-24306 — first run decodes 31 WebP images before first paint — **REVERTED**

Two errors in one finding, worth recording in full.

**The filed premise was wrong**, and an anti-vacuity assertion caught it. The
finding said "31 animated WebP frames". The bundled pack is 31 *still* WebP
files; PIL routes even single-frame WebP through `WebPAnimDecoder`, which is
what the profile's 31 `get_next` calls actually showed.

**Then the fix was wrong.** `_inspect_image_bytes` computed a duration
unconditionally and discarded it for still images on the next line, so guarding
it on `is_animated` looked free. It is not: **`_image_duration_ms` is the only
caller of `image.load()` on that path, and `Image.open()` reads the header
without decoding the payload.** With the guard in, `_inspect_image_bytes`
accepts bytes whose real decode raises `OSError` — verified directly with
interior payload corruption at intact container header and length:

| bytes | `_inspect_image_bytes` | real decode |
|---|---|---|
| intact | accepted | OK |
| mid-payload corrupted | **ACCEPTED** | FAILS (`OSError`) |
| late-payload corrupted | **ACCEPTED** | FAILS (`OSError`) |

Truncation *is* still caught by `Image.open`, which is why this was not obvious.
The repository's own
`test_complete_validation_rechecks_cumulative_actual_decoded_work` went red;
the word "actual" in its name is load-bearing. **The new test written for the
fix passed** — it pinned the optimisation without checking the invariant the
decode was upholding, which is the more useful half of the lesson.

The measured saving was real (first-run `__init__` 0.5028 / 0.4975 / 0.5015 /
0.5025 s → 0.2888 / 0.2765 / 0.2824 / 0.2916 s, four interleaved pairs) and came
*from* removing that validation, so no version of this approach keeps both.

**What the next attempt should do instead:** move the built-in pack seeding off
the first-paint critical path rather than weakening what it checks.
`deferred_actor_pack_recovery` is precedent (task-21106); `seed_builtin_content`
is currently called synchronously from `get_chachanotes_db_lazy`, and decoupling
it is the real work. An in-code comment now records why the guard must not be
"optimised" again without keeping a decode.

---

## Verified clean

Regressions looked for and not found — where the previous three cycles' fixes
are still load-bearing:

| surface | measurement |
|---|---|
| SQL on the keystroke path | 0 statements / 40 keys |
| idle app-side work on Console | ~0.8 ms across 6 s (caret blink + nav hints only) |
| streaming chunk handling | 0.25 ms/chunk empty, 0.57 ms/chunk at 200 messages |
| Library warm screen entry | 18.0 ms app-side, 97 widgets |
| steady-state `TldwCli.__init__` | 0.13–0.18 s |

Broad regression A/B against a pristine merge-base worktree, same subset both
sides: **15 failed / 404 passed on each, identical name sets** — every failure
pre-existing on dev.

## Guard status

`Tests/Performance` on pristine dev `3a3383123e`: **2 failed, 773 passed**.
On the branch: **781 passed, 0 failed** (both dev reds fixed; seven new guards).

| ratchet | pristine dev | branch |
|---|---|---|
| ui-ready module census | **972/970 RED** (3 of 4 runs) | 964/970, headroom 6 |
| boot worker allowlist | **RED** (anonymous worker) | green |
| pre-import payload (LOC) | 379,497/380,000, headroom 503 | 378,930, headroom 1,070 |
| pre-import payload (modules) | 491/500 | 489/500 |
| boot import weight | 651/660, headroom 9 | unchanged |
| boot CSS bytes | 857,246/860,000 | unchanged |

Every new guard was mutation-tested: the fix was reverted, the guard observed
red with a diagnostic that names the defect, and the fix restored.

---

## Open items for the owner

1. **Third-party boot import weight is unguarded.** All four ratchets count
   `tldw_chatbook.*` only. TASK-24305 was reachable only by profiling.
2. **`boot-import-weight` sits at 651/660 with headroom 9**, unchanged by this
   cycle and consumed steadily (646 → 651 in one day before it).
3. **`css/components/_agentic_terminal.tcss` remains ~42% of the boot CSS
   bundle**, carried over from the 08-27 review.
4. **Not confirmed, worth a look.** A warm Library entry shows 520
   `_open_directory_component`, 32 `_prepare_artifact` and 8 `get_connection`
   calls with high cumulative (I/O) time but negligible self-time. That is
   either harmless syscall accounting or synchronous filesystem work on the
   switch path; it needs a dedicated measurement, which this cycle did not run.
5. **Pre-existing dev red, untouched:**
   `Tests/Chat/test_console_agent_swap.py::test_run_error_via_regenerate_
   preserves_original_answer_and_status` fails identically on pristine dev.
