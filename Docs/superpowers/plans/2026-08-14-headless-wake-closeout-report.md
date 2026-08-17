# Close-out report — headless wake (task-15860, plan Tasks 7 + 8 + the live pass)

- Branch `feat/task-15860-closeout`, worktree `.worktrees/headless-closeout`.
- Base `origin/dev` `524194c15`; dev moved to `f71e625d1` during the work.
  Every measurement below is on `524194c15` plus this branch's own commits.
- Predecessors, all merged: `…-task-0-report.md`, `…-task-1-report.md`,
  `…-lifetime-report.md`, `…-viewless-report.md`, `…-continuity-report.md`,
  `…-fires-report.md`, `…-approval-report.md`, `…-launch-report.md`.

**One sentence:** the feature works — a supervisor genuinely wakes with no
Console screen mounted, on another screen, in another session, and at the
next launch after a restart — and the live pass found one real bug that
tests could not see: **a headless approval card mounts empty and cannot be
answered until you click that session's tab**, which also stalls every
other conversation's owed wake behind it.

---

## 1. Part A — the four AC#3 invariants, proven together on current dev

New suite: `Tests/UI/test_console_headless_invariants_gate.py` (4 tests,
real app, real on-disk ChaChaNotes + `agent_runs`, the real navigation
API, production wake chain, no gate monkeypatched).

| Invariant | Verdict | How it was proven |
|---|---|---|
| **No USER transcript row in a headless wake, on the DB ROWS** | **PASS** | Already pinned for the nav-away path (`test_console_headless_wake_fires.py`) and the launch path (`test_console_launch_wake.py`). This gate adds the un-pinned third path — a wake WITHHELD by the kill switch and then RELEASED through `retry_soon` — and asserts the persisted senders are exactly `['user','assistant','system','assistant']` with `count('user') == 1`, plus the store's user rows unchanged. Confirmed live four separate times against a real provider (§3). |
| **Exactly-once across a restart mid-commit, via the ledger** | **PASS on the bound; the strong claim was FALSE** | Process one dies *inside* the window `_deliver` opens (the ledger stamp raises — durable state byte-identical to a kill there: rows committed, ledger unstamped, ◈ set); process two is a real launch; process three another. **Measured: six rows, not four** — the same child result announced twice. Process two's own stamp commits, so a third launch adds nothing. `_deliver`'s own comment predicts this ("a lost stamp risks one re-announce at a later claim, never a lost result"); the User Guide claimed the stronger thing and is corrected. The test asserts the BOUND (≤1 re-announce, row shape, no USER row, third launch inert), never the number, so closing the window later is not a test failure. **Reproduced live by accident** (§3.4). |
| **`autowake_enabled = false` silences the headless fire point and loses nothing durable** | **PASS** | Existing tests assert the registry, mark and ledger; this one asserts the **conversation's persisted rows do not move** while OFF, then flips the switch ON on the same live coordinator and watches the rows go 2 → 4. Mutation-tested: neutering `if not autowake_enabled(): return` in `_attempt` makes it fail. Confirmed live across a restart (§3.5). |
| **Deliveries stay serialized app-wide (one `_delivering` per runtime)** | **PASS** | Two tests. (a) structural: one runtime, one controller, one coordinator object across a real screen REPLACEMENT taken mid-delivery, with the in-flight flag not reset by the fresh screen. (b) behavioural: a SECOND conversation whose own session is idle cannot start a wake turn while one is in flight — the only gate that can enforce that. Mutation-tested: neutering `if self._delivering is not None: return` kills it. **Observed in the wild** (§3.3): a blocked round held every other conversation's wake until it was denied. |

### The two harness traps this cost (both recorded in the file and in `lessons-testing-evidence.md`)

1. The seeding send already streamed a payload, so every payload assertion
   must count from a baseline, not from zero. The first draft failed on
   this, reading a harness artefact as a product failure.
2. **Two successive drafts of the serialization test passed and survived
   the mutation.** Draft 1 used one conversation, so the per-session busy
   gate refused the second attempt several lines before `_delivering` was
   ever read — a test of gate 1 wearing gate N's name. Draft 2 used two
   conversations and still survived, because the provider double's stall
   belongs to the GATEWAY: with the guard removed the second turn genuinely
   started and then parked at the same stall, streaming nothing, so
   "refused" and "started then blocked identically" were the same
   measurement. Counting *entries into the readiness probe* separates them,
   and the mutation then kills the test immediately.

---

## 2. Part B — documentation

- `Docs/User_Guide/console/agent-runs-and-tools.md`
  - exactly-once paragraph: the mid-commit re-announce is stated plainly.
  - headless-approval paragraph: the live bug (task-17500) is stated with
    the workaround, and the app-wide stall it causes.
  - kill switch: the TOML tier needs a **restart** (config is read once at
    startup; only the env var is live). Found by mis-testing it live.
  - marker precedence: after a woken turn finishes off-view the tab shows
    `✓`, not `◈`.
  - a "Verified against dev @ 524194c15 — 2026-08-17" stamp listing exactly
    the four scenarios driven live and naming the two things the pass could
    not confirm.
- Spec `2026-08-08-supervisor-agent-fleet-design.md`: §7's "honest
  architectural limit as built" paragraph carries its superseding note; the
  follow-up row for task-15860 closes. (The plan called that §10; the row
  lives in §11.)
- `backlog/docs/lessons-live-verification.md`: the import-provenance trap;
  the P1 finding (a DB append is invisible both to a live Console and to the
  next mount).
- `backlog/docs/lessons-testing-evidence.md`: the unreachable-guard mutation
  trap; measure-then-assert.
- `backlog/tasks/task-17500`: the approval-card bug, filed with the
  reproduction, the control that makes it headless-specific, and the seams
  the evidence points at.

---

## 3. Part C — the live pass

**Rig.** tmux, 220×58, `.venv/bin/python -m tldw_chatbook.app`, a scratch
profile with its own `HOME`, `XDG_*`, `TLDW_CONFIG_PATH` **and `[paths]
data_dir`** (the config path alone does not move the databases — see the
profile-isolation lesson). Real Anthropic `claude-sonnet-5`. Isolation
verified afterwards: the real `~/.config/tldw_cli/config.toml` is
byte-identical (same SHA before and after), every database the run created
is under the scratch `data_dir`, and `git diff | grep -c sk-ant` is 0.
Ground truth for every claim is the app's own ChaChaNotes and
`agent_runs.db`, read with `sqlite3` — the pane is corroboration, never the
finding.

*Harness note worth keeping: piping the app's stdout (`| tee`) or
redirecting its stderr makes Textual render at 80×24 or not render at all.
The app needs a real tty on both.*

### 3.1 A child finishes while you are in another Console session — **PASS**

```
=== TOAST at t=10s ===
│ Background sub-agent finished in “Use spawn_subagent to  │
```

Session B ("Chat 2") displayed throughout. Rows 10 → 12 in session A's
conversation: `system` (machine-origin notice) + `assistant`. The reply
used the child's result — *"The background essay-writing sub-agent has
completed. The full 1500-word essay is available in the run log…"*. The
◈ `fleet_unseen` mark was SET on the unwatched conversation
(`49ab353d|fleet_unseen|…20:29:33`). No USER row.

### 3.2 A child finishes while you are on Library — **PASS**

Console genuinely unmounted (screen `Library | Local` for the whole
window). Rows 6 → 8 at t≈45s with nothing mounted; persisted senders
`user, assistant, system, assistant, user, assistant, system, assistant`.
The ◈ mark was set at delivery and **cleared only on return** (the marks
table went from one row to empty once the conversation was viewed).
Returning showed the delivered turn already in the transcript, including
the supervisor's own in-wake tool markers:

```
│  ⚙ search_run_log → record 000005 [tool_call/search_run_log]
│                     ▼ reply ready — jump to latest
```

The `◈` glyph itself was seen in the tab bar for a still-owed
conversation: `◈ Use spawn_subagent… ✕`.

### 3.3 A headless approval — **FAIL (the toast half passes; the card does not)**

The announcement works, on Library, with Console closed:

```
│ Agent in Two instructions. FIRST: us... needs approval   │
│ to use a tool. Open Console to review -- nothing runs    │
```

Opening Console shows `◆` on the session tab and `Approvals: 1 pending`
in the status bar — and a card that is **visible and empty**:

```
│ █▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀█ │
│ █                                                       █ │
│ █  Approval required                                    █ │
│ █                                                       █ │
│ █▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄█ │
```

No tool name, no arguments, no decision controls. Stable across 20s.
Reproduced on **both** headless paths — the nav-away wake and the
launch wake.

Three controls make it a real, headless-specific defect rather than a
capture artefact:

- **A mounted round renders fully.** Asking for the same tool with Console
  open gives `Built-in · write_file (high risk)`, the arguments, and
  `Approve all  Submit  Deny all`.
- **A session switch repairs it.** Clicking that conversation's tab
  re-derives the card through `switch_session` and it comes back complete
  and correct — `{"path":"paper-essay.txt","content":"# The History of
  Papermaking: Fr…}` with `Approve once ▼ / Deny` and the bulk buttons.
  So the payload is intact and the round is answerable; only the
  open-Console path mounts a body-less card.
- **Denying it released work that was stuck behind it.** A second
  conversation's completion had been sitting undelivered with its ◈ mark
  set; the instant the blocked round was submitted as Deny, that
  conversation went from 4 rows to 6. That is the app-wide serialization
  invariant working exactly as designed — and it turns one unanswerable
  card into an app-wide stall.

Filed as **task-17500** (high). This falsifies the shipped sentence "the
card is waiting, already mounted, the moment you open Console" and the
approval landing's own AC; both are corrected.

### 3.4 Restart: an owed wake delivered at launch — **PASS**

Quit with a completion recorded and undelivered (◈ set,
`wake_delivered_at` NULL). Relaunched with `default_tab = "library"`, so
no `ChatScreen` is ever constructed:

```
t=4s  rows=2 consoleComposer=0 screen=''
t=8s  rows=3 consoleComposer=0 screen=' Library | Local'
t=12s rows=4 consoleComposer=0 screen=' Library | Local'
```

`user, assistant(OFFTEST), system(notice), assistant("Got the essay from
the sub-agent — it's a complete ~900…")`; ledger stamped
`24ca3e7c … 2026-08-17T20:49:53`; ◈ mark survived (launch delivery has no
view). A **second relaunch watched for 60 s added nothing** (rows stayed
at 4).

An earlier, accidental run of the same shape is the live reproduction of
§1's mid-commit finding: quitting while a wake turn sat blocked on the
unanswerable approval left the ledger unstamped, and the next launch wrote
a **second** identical notice into that conversation (rows 15 → 16).
Exactly one duplicate, exactly as the test measured.

### 3.5 `autowake_enabled = false` — **PASS**

With the switch off (set in `config.toml` and **restarted** — see below),
a child settled: the ◈ mark was written, the completion toast fired, the
ledger row stayed unstamped, and **the conversation stayed at 2 rows for
25 s after the mark appeared**. Turning it on and relaunching delivered
exactly what OFF had recorded (§3.4 is that same run).

*Found the hard way:* editing `autowake_enabled` in `config.toml` does
**not** affect a running process — `_setting` falls through to
`get_cli_setting`, which reads the config cached at startup. Only
`TLDW_AGENTS_AUTOWAKE_ENABLED` is live, which is what the unit tests use.
My first attempt at this scenario flipped the file and watched a wake
deliver anyway; that was my harness, not the product. Documented on the
page.

### 3.6 The turn-activity line (PR #1762) — **PARTIAL PASS**

The line is live and its clock advances. Measured across two real turns at
~0.4 s sampling:

```
Generating…
Thinking… · 1s
Thinking… · 2s
…
Thinking… · 18s
```

Monotonic, 1 s → 18 s, and `Generating…` appears before the first token.

**What I could not confirm live: the `⚙ <tool> · Ns` state.** Not for want
of trying (a 45 s zero-sleep capture burst caught nothing). The app's own
Trajectory view explains it: every tool call in the session started and
finished within the same second —

```
 5  tool_call   ↳ spawn_subagent -> started 283cf4cb…   13:42:10   13:42:10
 9  tool_call   ↳ write_file -> ⚙ write_file → tool call denied…  13:44:01   13:44:01
```

— and the one long wait available, an approval, is **not** part of the
tool step, so the line reads `Thinking… · Ns` while the app is in fact
waiting on the user. That is arguably the more interesting observation:
during an approval wait the activity line does not name the tool that is
waiting. Recorded as an observation, not filed — it is a pre-existing
property of when the tool step is recorded, not something this arc
changed. The `⚙` state itself is covered by
`Tests/UI/test_console_turn_activity_line.py` with a held-in-flight tool.

---

## 4. Gate

Read counts, cwd = this worktree, parent-repo `.venv/bin/pytest`,
`-p no:randomly`.

| Run | Result |
|---|---|
| Baseline, 11 arc suites (UI + Chat), `524194c15` | **103 passed** in 158 s |
| Baseline, `Tests/Agents` + `test_console_headless_wake_invariants.py` + `test_console_fleet_wake.py` | **1492 passed** in 55 s |
| Import provenance | **1 passed** — `PROBE imported tldw_chatbook from: …/.worktrees/headless-closeout/tldw_chatbook/__init__.py` |
| New invariant gate alone | **4 passed** |
| Final combined gate (all arc suites + the new one + `Tests/Agents`) | see §5 |

No full-suite run was attempted: the machine carries foreign pytest runs
and a 281-file population run was measured at ~200 minutes by the previous
agent. That is a stated limit of this gate, not implied coverage.

Production tree was mutated twice (both `_attempt` guards) and restored by
Edit each time, with `git diff --name-only tldw_chatbook/ | wc -l` reading
`0` after each restore. The live run regenerated the CSS bundle's
timestamp line; that too was restored by Edit, not by checkout.

---

## 5. task-15860's four ACs

1. **A finished background sub-agent wakes its supervisor with no Console
   screen mounted, under the same gate, caps and approval floor** — MET.
   Driven live twice (§3.1 nav-away-to-Library, §3.4 launch with Console
   never opened) and pinned by `test_console_headless_wake_fires.py`,
   `test_console_launch_wake.py`, and the caps/floor tests in
   `Tests/Chat/test_console_headless_wake_invariants.py`.
2. **No regression to screen-scoped semantics** — MET. Owned by the
   lifetime and approval landings and re-run green here; leaving Console
   still cancels the visit's turns and denies its parked rounds (observed
   live: the round left unanswered was denied on leaving).
3. **Every wake invariant holds headless** — MET, with one honest residue:
   no USER row (three paths), exactly-once (bounded at one re-announce
   inside the commit window, documented), kill switch, app-wide
   serialization. §1.
4. **The User Guide's honest-limits paragraph rewritten** — MET, plus the
   spec's superseding note and its follow-up row.

Marked **Done**. One bug is open against the feature, filed separately as
task-17500; it is a defect in a shipped behaviour, not an unmet AC.
