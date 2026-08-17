# Launch-wake report — delivering what was owed while the app was closed (task-15860, plan Task 6)

- Branch `feat/task-15860-launch-wake`, worktree
  `.worktrees/headless-approval/.worktrees/headless-launch`.
- **Merge-base `ed49499b8`** (contains all seven merged headless-wake
  landings: ownership, lifetime, viewless, continuity, wake-fires, the
  cross-suite sync fix, and headless approval). Every baseline below was
  measured either on the untouched branch before any production edit, or
  in a detached worktree at that commit
  (`.worktrees/headless-launch-base`) — never against `dev`'s tip.
  (`dev` moved to `feea06193` during this task; `ed49499b8` is still the
  merge-base, confirmed by `git merge-base`.)
- Predecessors: `…-fires-report.md` (§9's "launch / first-boot wake is
  untouched" — this report executes and closes it), `…-approval-report.md`,
  `…-viewless-report.md`, `…-continuity-report.md`.

**One sentence:** a background sub-agent that finished while the app was
closed now wakes its supervisor at the next launch — mark-gated, behind
the existing `[agents] autowake_enabled`, and costing an install that has
never run one exactly one indexed read and zero constructed objects.

---

## 1. What was actually missing, executed rather than read

The fires landing removed the "no Console mounted" limit *inside a
process that had opened Console once*. `ConsoleRuntime.ensure_chat_
controller` / `ensure_agent_bridge` are lazy and their only callers were
`ChatScreen` and its Console modules, so with Console never opened there
is no controller, no `ConsoleFleetWakeCoordinator` and no fan-out
registration. That report labelled it an inference. `Tests/UI/
test_probe_launch_wake.py` executes it:

```
PROBE P1 deferred-startup reached: True
PROBE P1 screen stack: ['Screen', 'LibraryScreen']
PROBE P2 chat_controller: None
PROBE P2 chat_store: None
PROBE P2 agent_bridge: None
PROBE P2 ui_ready: True
```

So the app *does* reach `_schedule_deferred_startup_work` on a launch that
never opens Console — there was a hook site, and nothing used it.

---

## 2. The design, and the two things it is NOT

**The owner's ruling, implemented literally:** at startup do ONE cheap
indexed read; deliver only for a conversation that already carries a
`FLEET_UNSEEN` mark **and** an owed `agent_runs` ledger row. Behind
`[agents] autowake_enabled`; **no** separate `autowake_on_launch` switch.

`tldw_chatbook/Chat/console_launch_wake.py`:

| Function | What it costs |
|---|---|
| `marked_conversations_at_launch(app)` | the kill switch, then ONE `conversation_local_marks` listing. Nothing else. |
| `deliver_launch_wakes(app, marked)` | builds the runtime **viewlessly**, seeds through the same `seed_from_marks` the Console mount claim uses, hydrates one session per owed conversation, and hands off. |

`app.py._schedule_launch_wake()` hangs it on the existing deferred
startup work, after the first interactive frame.

It is **not** a second wake path. Everything after the hand-off is the
already-shipped machinery, unchanged: the kill switch again at
`_attempt`'s fire point, `send_refusal_copy` (run state, queue ownership,
`max_parallel_runs`), user-wins-ties, one delivery at a time app-wide,
the `wake_delivered_at` ledger for exactly-once, and the viewless
`wake_conversation_in_view` that KEEPS the ◈ mark on a delivery nobody
watched.

It is **not** ledger-seeded. See §5.

---

## 3. The red, and its before/after

`Tests/UI/test_console_launch_wake.py::
test_a_launch_delivers_a_wake_owed_from_a_previous_process`

Two genuinely separate `TldwCli` processes over one `tmp_path`:

- **process one** mounts a real `ChatScreen`, sends once through
  `submit_draft` (so the conversation really persists), writes a terminal
  survivor run into the real sibling `agent_runs.db`, and sets the ◈ mark;
- **process two** boots with `default_tab="library"`, so the real routing
  never constructs a `ChatScreen` at all — asserted as a precondition in
  every test in the file, not assumed.

**BEFORE** (at `ed49499b8`, zero bytes of this branch, the finished file
copied into the detached baseline worktree — measured twice: **6 failed,
3 passed** against the first draft and **8 failed, 3 passed** against the
final ten-test file):

```
FAILED …::test_a_launch_delivers_a_wake_owed_from_a_previous_process
E   AssertionError: a background sub-agent finished before this process
    started, its ◈ mark and its owed ledger row both survived -- and the
    launch delivered no wake turn at all
FAILED …::test_a_second_launch_does_not_re_announce_a_delivered_wake
FAILED …::test_a_launch_into_console_delivers_without_stealing_the_active_tab
FAILED …::test_a_launch_with_no_marks_constructs_nothing_and_reads_once
E   AssertionError: a launch with no marks must cost exactly one indexed
    mark listing; got []
FAILED …::test_the_startup_cost_pin_is_not_vacuous
FAILED …::test_the_kill_switch_silences_the_launch_fire_point_and_loses_nothing
FAILED …::test_an_unresolvable_ephemeral_mark_is_cleared_at_launch
FAILED …::test_a_launch_hydrates_only_the_conversations_that_are_owed
8 failed, 3 passed, 3 warnings in 138.82s
```

The file COLLECTS cleanly at the merge-base — every symbol it imports
exists there — so each red is the behaviour failing, not an `ImportError`
(the trap `lessons-testing-evidence.md`'s TASK-16838 entry names; observed
again here for the *hydration* file, which cannot collect at base because
its subject does not exist, and is therefore reported separately rather
than folded into a red count).

**AFTER: 10 passed.**

The three that pass at the merge-base are named rather than hidden: the
phantom-wake test and the nothing-owed test pass because at the baseline
*nothing happens at all*, and the provenance probe. Both are still worth
having — they are the pins that must stay green, not reds.

The chain the red asserts, not merely the send:

1. exactly ONE wake turn reaches the provider, its payload's trailing
   entry `role="user"` carrying the machine marking and the child's
   result — **and the payload carries the conversation's real prior
   history**, which is what makes hydration load-bearing rather than
   decorative (a supervisor woken with an empty session spends money to
   say nothing);
2. the machine-origin SYSTEM notice (exactly one `metadata.origin ==
   "agent_wake"` row) and the assistant reply are in the app-owned store
   *and* in ChaChaNotes — persisted senders exactly
   `["user", "assistant", "system", "assistant"]`;
3. `agent_runs.wake_delivered_at` is stamped;
4. **no USER row** anywhere — the store's only USER row is process one's,
   and the DB has exactly one `user` sender;
5. the **◈ mark SURVIVES**, because a runtime with no view reports the
   conversation as not watched.

---

## 4. The startup-cost pin, and its mutation

`test_a_launch_with_no_marks_constructs_nothing_and_reads_once` — the pin
that protects every user who never touches the fleet. Four independent
observations, because "nothing was constructed" is exactly the claim a
weak test states and never checks:

1. the marks service saw **exactly one** `list_marked_conversation_ids`
   call, for `fleet_unseen`;
2. the runtime holds no store, no gateway, no bridge, no controller;
3. **no `agent_runs.db` file exists on disk** — constructing the bridge
   opens (and creates) it, so the filesystem is an observer production
   code cannot lie to;
4. no `deferred_launch_wake` task was ever created.

Its control, `test_the_startup_cost_pin_is_not_vacuous`, runs the same
four probes WITH a mark and watches every one flip. Without it, a launch
hook that never ran at all would satisfy the pin perfectly.

"Indexed" is verified, not asserted:
`ChaChaNotes_DB.py`'s `idx_conversation_local_marks_type` is
`(mark_type, updated_at DESC, conversation_id)`, which covers
`list_marked_conversation_ids`' `WHERE mark_type = ? ORDER BY updated_at
DESC, conversation_id ASC LIMIT ?` completely.

**Mutations against the pin:**

| # | Mutation | Result |
|---|---|---|
| M3 | `app.py`'s `if not marked: return` → `if False:` | **1 failed** — the pin, on observation 4 (`a launch with no marks scheduled the wake task: ['deferred_launch_wake', …]`). The construction observations did **not** fire, because `deliver_launch_wakes` has its own `if not marked: return 0`. Defence in depth, and worth recording: one guard's removal is not observable through the other. |
| M3b | **both** guards removed | **2 failed** — the pin on observation 1 (`got ['fleet_unseen', 'fleet_unseen']`; `seed_from_marks` does its own listing) **and** the phantom test on `an unmarked owed row still built the whole Console runtime at launch` |

---

## 5. The phantom-wake case stays green

`test_a_crash_killed_child_swept_to_error_wakes_nobody_at_launch`.

A child left `running` by a crash is swept to `error` by the next
`AgentRunsDB.__init__` (modelled the way `Tests/DB/test_agent_runs_db.py`
models a restart — discarding the per-process `_swept_paths` entry). That
makes it terminal, undelivered and therefore **owed by the ledger's own
definition**. It carries no mark, because nothing ever settled it through
the fan-out.

The test's two harness preconditions are what stop it passing for the
wrong reason:

```
the reconcile sweep must have marked the crash-killed child as error   -> ok
the ledger must consider this orphan OWED                              -> [child_id]
a crash-killed child leaves no ◈ mark                                  -> ()
```

and then: nothing reaches the provider, **and** `console_runtime.
chat_controller is None` — the launch did not even build the runtime.
That second assertion is what makes it a pin on the mark gate rather than
on the send gate.

**Result: green on this branch, and green at the merge-base** (it is a
"must not fire" property, so it passes on both sides — stated plainly).
M3b is what proves it is not vacuous.

---

## 6. Hydration — extracted, with a characterization pin

**Decision: EXTRACTED, as a pure refactor, in its own commit, before the
launch path used it** (`ccd1fd8f9`).

The plan flagged that this policy is screen-bound and asked for a
decision from the code as it now stands. As it stands,
`ChatScreen._resume_console_workspace_conversation`
(`UI/Console_Modules/workspace.py`) interleaves the session-producing
policy with a screen's own work, and the session-producing half turns out
to read only two app members — `chat_conversation_scope_service` and
`chachanotes_db`. Duplicating it for the launch path would have put a
second, worse resume policy in the codebase, which is what the plan
warned against; extracting it is a ~110-line move.

Moved to `tldw_chatbook/Chat/console_conversation_hydration.py`:

| Moved | Why it is policy, not plumbing |
|---|---|
| `console_messages_from_conversation_tree` | ALL branches (not the latest), parenthood from tree NESTING, empty rows dropped but transparent to their children, plus the one batched attachment fetch for positions ≥ 1 |
| `load_console_conversation_tree` | the `depth_cap=10_000, root_limit=10_000` caps: the service defaults are 50, and a truncated tree makes a wrong provider payload |
| `hydrate_console_session` | workspace/title resolution, the durable active-leaf pointer, the runtime-backend/assistant/character field discipline, `restore_persisted_session`, the roleplay overlay |
| `apply_resume_settings_overrides` | the two things the CONVERSATION ROW contributes to settings — saved system prompt (verbatim, only blank collapses) and pinned prefill |

`default_console_session_settings` joined `console_session_settings.py` so
the llama.cpp base-url rule that always accompanied
`build_default_console_session_settings` at the screen's call site has one
home too.

The screen keeps every view concern: draft snapshot, both failure toasts,
resume-marker overlay, character-name/label resolution, retrieval-scope
warm, repaint, focus. `_console_messages_from_conversation_tree` keeps its
name and signature — **eight test files call it directly**.

**The one difference between the callers, named rather than hidden:** the
screen's base settings are inherited from the currently ACTIVE session; a
launch has no active session, so it starts from the config defaults the
screen itself falls back to. That is inherent to having no view.

**The characterization pin** —
`Tests/Chat/test_console_conversation_hydration.py`, two tests:

- `test_the_screen_tree_walk_still_flattens_every_branch` pins the
  screen's public seam on a deliberately awkward fixture (two branches off
  one root, a truly-empty node mid-branch whose child must re-parent
  through it): `[("m1", None), ("m2", "m1"), ("m4", "m2"), ("m5", "m1")]`;
- `test_a_launch_hydrated_session_matches_a_screen_resumed_one` is the
  equivalence test the plan asked for: for one fixture conversation, the
  session a launch hydrates headlessly and the session the screen resumes
  agree on the whole message tree with its parenthood, on ten identity /
  roleplay fields, and on the **full settings snapshot**.

An equivalence test between two callers of the same function is weak by
construction — a mutation inside the shared function breaks both sides
equally. So the test also carries absolute assertions (the system prompt
restored verbatim including its leading spaces and trailing newline; the
roleplay `user_name_override`; different session uuids so the two stores
are provably distinct). **M11** (the roleplay overlay dropped) dies on the
absolute assertion, not on the comparison — measured.

**Evidence the refactor is behaviour-preserving:** the five resume-owning
suites are **286 passed** on this branch and **286 passed** at the
merge-base in the detached baseline worktree, same invocation.

---

## 7. Mutations run and killed

Every mutation applied by **Edit** and restored by **Edit**; after the
last one `grep -rn "MUTATION-" tldw_chatbook/ Tests/` returns **0** and
`git diff` against the commits is empty.

| # | Mutation | Killed by | Result |
|---|---|---|---|
| M1 | the whole change absent (the untouched merge-base, the new file copied in) | the six behavioural tests | **6 failed, 3 passed** (§3) |
| M2 | `marked_conversations_at_launch` ignores `autowake_enabled` | the kill-switch test **only** | 1 failed, 8 passed |
| M3 | `app.py`'s empty-marks guard removed | the startup-cost pin only | 1 failed, 8 passed |
| M3b | **both** empty-marks guards removed | the pin **and** the phantom test | 2 failed |
| M4 | `_conversation_exists_locally` → always `True` | the ephemeral-mark test **only** | 1 failed, 8 passed |
| M5 | `_conversation_exists_locally` → always `False` | 5 tests (e2e, second-launch, pin control, kill switch, owed-only) | 5 failed, 4 passed |
| M7 | the per-conversation `has_pending` guard removed | `test_a_launch_hydrates_only_the_conversations_that_are_owed` only | 1 failed, 8 passed **after the fix below** |
| M8 | `wake.retry_soon()` never called | 5 tests | 5 failed, 4 passed |
| M9 | `controller.app = app` dropped | the e2e **after the fix below** | 1 failed, 8 passed |
| M10 | `wake.wire(app=app)` dropped | 6 tests | 6 failed, 3 passed |
| M11 | `hydrate_console_session` drops the roleplay overlay | the equivalence test's absolute assertion | 1 failed, 1 passed |
| M12 | the tree walk follows only the last child | the characterization test **and** three of `test_console_resume_active_path`'s own | 4 failed, 27 passed |
| M13 | the active-session restore removed | `test_a_launch_into_console_delivers_without_stealing_the_active_tab`, on exactly the tab-stealing assertion | 1 failed, 9 passed |

### The two that survived, and what they taught

**M7 survived the first draft**, and investigating it found a real gap
rather than a nuisance. With a single unowed mark the launch returns at
`seed_from_marks() == 0` *before* the per-conversation loop, so
`has_pending` was unreachable and owned nothing at all — the suite could
not tell the difference between "we checked each conversation" and "we
never got that far". Fix:
`test_a_launch_hydrates_only_the_conversations_that_are_owed` gives the
launch TWO marks, one owed, and asserts the unowed one is **never
hydrated** — because otherwise the user opens Console to an unexplained
tab for a conversation nothing happened in.
`test_a_mark_with_nothing_owed_is_left_alone_at_launch` gained the same
"no session was hydrated" half; it previously asserted only that the mark
survived.

**M9 would have survived**: nothing in the suite touched
`controller.app`. It is not a slot the viewless machinery covers — it is
deliberately excluded from `CONSOLE_VIEW_HOOK_SLOTS` because it is the
APP — and it is what carries a headless approval round's app-wide toast
and its `call_from_thread` bridge. The e2e now asserts it, **labelled as
a wiring assertion**: the consequence itself is owned by
`Tests/UI/test_console_headless_approval.py`, and driving a risk-tagged
tool through a launch wake was out of this slice's budget.

M10's kill list is the positive result worth reading: dropping
`wake.wire(app=app)` takes out six tests including the ephemeral-mark one,
because `seed_from_marks` reads the marks service off the wired app — so
the stale-mark clearing depends on the same wiring the delivery does.

---

## 8. The stale / ephemeral mark — verdict

**The leak is real, and both halves are now executed.**

- `Tests/UI/test_probe_launch_wake.py::test_probe_p3a…` drives the
  production source: `ConsoleChatController._agent_conversation_id`
  returns `session.persisted_conversation_id or session_id`, so an
  **unsaved** session's fleet work is keyed by a uuid.
  Measured: `persisted_conversation_id=None`, keyed id
  `23ddaee4-9394-4f91-9561-006cc2daba92`, `get_conversation_by_id` →
  `None`.
- `…::test_probe_p3…` drives the durability half: a second app over the
  same on-disk DB still lists that mark, and it still resolves to nothing.

**What this branch does about it:** a launch that owes such a conversation
a wake it can never deliver **clears the mark** rather than retrying it on
every boot for the life of the install
(`test_an_unresolvable_ephemeral_mark_is_cleared_at_launch`). Three
deliberate narrowings, each pinned:

1. the authority is the **local DB** (`get_conversation_by_id is None`),
   not a failed tree load — a tree load can fail for reasons that have
   nothing to do with the row existing (a scope service in server mode, a
   transient error), and clearing a live user's badge on one of those
   would be a real loss. A DB that cannot answer reads as "exists", so
   uncertainty keeps the mark;
2. only for a mark that is **otherwise owed**, i.e. one the launch was
   about to act on;
3. the owed `agent_runs` rows are **left unstamped** — they are stamped by
   delivery, never by give-up — and with the mark gone nothing indexes
   them, so nothing re-announces them. Asserted.

One consequence worth naming because it was not designed for and is
correct anyway: `get_conversation_by_id` filters `deleted = 0`, so a
**soft-deleted** conversation reads as absent too. A ◈ badge left behind
by a conversation the user deleted is therefore cleared at the next launch
by the same path — and it could never have been resolved either.

**Deliberately not touched:** a mark with **nothing owed** is left alone
even if its conversation is missing. With nothing owed, a mark is the
*delivered-but-unseen* badge task-15971 exists to show, and this slice has
no way to tell "an ephemeral chat whose results were all delivered" from
"a badge the user has not looked at yet" without inventing new badge
policy. `test_a_mark_with_nothing_owed_is_left_alone_at_launch` pins the
current behaviour. The residual cost is one indexed `undelivered_wake_runs`
read per launch per such mark.

---

## 9. Kill switch

`test_the_kill_switch_silences_the_launch_fire_point_and_loses_nothing`,
driven through the real env-var seam (`TLDW_AGENTS_AUTOWAKE_ENABLED`),
across three processes over one durable state:

- **OFF**: nothing reaches the provider, `console_runtime.chat_controller
  is None` (the switch is read *before* the mark listing, so an install
  with auto-wake off pays nothing at all), the ◈ mark survives, and
  `wake_delivered_at` is still NULL;
- **ON, next launch**: the same owed completion is delivered, and its
  notice carries the child's result.

M2 (the switch dropped from the launch read) kills exactly this test and
nothing else. Note that `seed_from_marks` honours the switch too, so the
kill is not from the delivery itself but from the runtime being
constructed — which is why that assertion is in the test.

---

## 10. Gate

Runner both sides: `.venv/bin/pytest <paths> -p no:randomly
-p no:cacheprovider -q --no-header -rf`, cwd = the worktree.
`Tests/test_probe_import_provenance.py` is in every row — the venv's
editable install resolves `tldw_chatbook` to a FOREIGN worktree and loses
only by `sys.meta_path` ordering. **Every count below was READ off a
summary line.**

| Gate | Baseline @ merge-base `ed49499b8` | Branch | Delta |
|---|---|---|---|
| **The specified battery** — `test_console_headless_wake_fires` (1), `test_console_headless_approval` (14), `test_console_sync_outlives_screen` (5), `test_console_store_continuity` (4), `test_console_viewless_hooks` (12), `test_console_runtime_lifetime` (14), `test_console_runtime_ownership` (7), `test_screen_residency` (7), `test_console_headless_wake_invariants` (13), the 16-file wake glob (109), probe (1) | **187 passed, 0 failed** (216.4s) | see §10.1 | — |
| **The new suites** — `test_console_launch_wake` + `test_console_conversation_hydration` + probe | **6 failed, 3 passed** (launch file only; the hydration file cannot exist there — its subject does not) | **11 passed** | +6 |
| **The resume-owning suites** — `test_console_resume_active_path`, `test_console_workspace_controller`, `test_console_session_settings`, `test_console_generation_store`, `test_console_video_message`, probe | **286 passed, 0 failed** (133.5s) | **286 passed, 0 failed** (128.8s) | **0** |
| `Tests/Agents/` + probe | not measured — a green final cannot hide a regression | see §10.1 | — |
| **The whole Console population in ONE process** — every `Tests/UI/test_console_*.py` + `test_screen_residency` + probe | see §10.1 | see §10.1 | — |

### 10.1 Final counts

*(filled in below from the completed runs)*

---

## 11. Deliberately not done

- **The invariant gate (plan Task 7)** and **the docs sweep (Task 8)** are
  out of scope. Only the User Guide sentence about *this* behaviour was
  written (§12), because shipping a false one is not acceptable for even
  one merge.
- **The page's "Verified against" stamp is untouched.** This slice
  verified two paragraphs, not the page.
- **The spec's own superseding note** (`…supervisor-agent-fleet-design.md`
  §7's "honest architectural limit as built") is Task 8's.
- **A launch wake's approval round was not driven end to end.** The app
  handle it needs is asserted (§7, M9); the behaviour is owned by the
  approval landing's suite. Naming it rather than implying coverage.
- **The launch-hydrated session becomes the store's ACTIVE session**,
  because at launch there is no other. A user who opens Console after a
  launch wake therefore lands in the woken conversation rather than a
  fresh chat. That is arguably right — the ◈ points there and money was
  just spent there — but it is a UX change nobody has ruled on, and it is
  named here rather than buried.

## 12. The User Guide sentence

`Docs/User_Guide/console/agent-runs-and-tools.md` said:

> One honest limit remains. **Nothing wakes while the app is not running**
> — completions are recorded durably (mark + ledger) and claimed the next
> time the app starts and Console opens; a wake at launch, before Console
> has ever been opened in that session, is follow-up work (task-15860).

It now says:

> **A wake you were owed is delivered at the next launch, without opening
> Console.** Nothing runs while the app is closed — a completion that
> lands then is recorded durably (the `◈` mark plus the ledger) and waits.
> At the next start, once the app is up and interactive, any conversation
> that still carries a `◈` mark *and* still owes a result has its
> supervisor woken there and then: the conversation is reopened in the
> background, the turn runs, and you find it already in the transcript
> with its `◈` still lit when you open Console. Nothing else is woken —
> never a conversation without a mark, and never one whose results were
> already delivered — and the whole thing is off when `[agents]
> autowake_enabled` is off (there is no separate launch switch). If you
> have never run a background sub-agent, launch does exactly what it did
> before: one indexed check that finds nothing.
>
> One case cannot be delivered and is cleaned up instead: sub-agent work
> started in a **temporary (unsaved) chat** belongs to a session that does
> not survive the app, so there is no conversation left to wake. Its `◈`
> mark is cleared at the next launch rather than left pointing at nothing.
> Save the chat before starting long background work you want to come back
> to.

## 13. Concerns

1. **A launch now spends money before the user has done anything.** That
   is the point of the task and it is triple-gated (kill switch, ◈ mark,
   owed ledger row), but it is a real behavioural change: opening the app
   can now start a paid turn with no window on it. The ◈ mark is the only
   signal until the user opens Console.
2. **The woken conversation becomes the active session** (§11). Worth an
   owner eyeball before this reaches users.
3. **The launch fires once, at startup.** A completion that lands while
   the app is running but Console has never been opened in that process
   still has no fire point: the fan-out consumer that records it lives on
   the agent bridge, which nothing built. That is not a regression (it was
   unreachable before too — with no Console there is no bridge to run a
   sub-agent from), but it means "wake at launch" is exactly what it says.
4. **The stale-mark clearing is a durable delete.** It is narrowed three
   ways (§8) and uncertainty always keeps the mark, but it is the only
   write in this slice that destroys user-visible state.
5. **`AgentRunsDB`'s reconcile sweep now runs earlier** for an install
   with marks: at launch rather than at the first Console open. Same
   sweep, same per-process guard, just sooner. Named because it changes
   *when* a crash-killed child's row turns `error`.
6. **The equivalence pin is structurally weak** (§6) and leans on its
   absolute assertions. If someone later changes the shared hydration
   function, the comparison will not catch it; M11 is the evidence that
   the absolute half does.
