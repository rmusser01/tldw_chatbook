# Headless-approval report — surface it, and let the user answer it (task-15860, plan Task 5)

- Branch `feat/task-15860-headless-approval`, worktree
  `.worktrees/headless-approval`.
- **Merge-base `e112798f1`** (contains all six merged headless-wake
  landings: ownership, lifetime, viewless, continuity, wake-fires, and
  the cross-suite sync fix). Every baseline below was measured either on
  the untouched branch before any production edit, or in a throwaway
  detached worktree at that commit (`.worktrees/headless-approval-base`)
  — never against `dev`'s tip.
- Predecessors: `…-task-0-report.md` (P4), `…-viewless-report.md`,
  `…-fires-report.md` (§9's flagged, unexecuted inference — this report
  executes it).

**One sentence:** a risk-tagged tool in a headless wake turn used to be
refused in **1.01 s** with the user never told; it now raises an app-wide
toast at the moment it arms, keeps its badge, waits, and has its card
already mounted when the user opens Console — with the deadline, the
approval floor, and every deny path untouched.

---

## 1. Step 1 — the measured behaviour, and why the plan's number was stale

The plan and the seam map both carry P4's figure: 120.43 s to a
fail-closed `timeout`. **That is no longer what happens, for two
independent reasons, and the fires landing's §9 flagged one of them as an
unexecuted inference. Executed here:**

`Tests/UI/test_probe_headless_approval_behaviour.py` — a real `ChatScreen`
mounted and seeded, a wake turn held in flight at the provider readiness
probe, Console left through the **real navigation API**
(`NavigateToScreen("library")` plus the real "Leave Console?" dialog
answered by pressing Leave), then a risk-floored built-in row
(`server_key "agent:builtin"`, `reason "risk_floored"` — the shape
`build_tool_review_hook` emits) armed from a plain child thread:

```
config: _DEFAULT_MCP_APPROVAL_TIMEOUT_SECONDS = 0.0
config: effective [mcp] approval_timeout_seconds = 0.0
wake turn parked at the readiness probe: True
Console unmounted: True
runtime outlived the screen: True
visit Event set (_shutdown_requested): True
controller._disposed: False
controller.app wired: True
set_pending_approval wired: False
park_pending_approval wired: False
MEASURED decisions = {'builtin__write_file': 'deny'}
MEASURED wall time to verdict = 1.01s
registry: round still registered after the verdict = False
registry: parked payload retained = False
```

**1.01 s to `deny`. Not 120 s, and not `timeout`.** Two changes landed
under P4's number:

1. **ADR-067** (`8403c12e6`, merged since Task 0) dropped
   `_DEFAULT_MCP_APPROVAL_TIMEOUT_SECONDS` from `120.0` to **`0.0`** —
   `<= 0` now means *no deadline at all* — and added a no-`app` guard
   that denies on the spot. P4's controller had no `app`, so P4's exact
   rig would today return in microseconds rather than 120 s.
2. **The lifetime landing** made `_shutdown_requested` per-VISIT, and
   `request_mcp_approvals` binds it at ARM time
   (`_bind_visit_cancel_signal`). While Console is detached that Event is
   already **set**, so `_is_session_cancelled` denied the round at the
   first 1.0 s poll.

The last two lines of the probe matter as much as the number: after the
verdict the round is gone from `_pending_approval_rounds` *and* its
payload is popped. So the viewless landing's slot note — "a round armed
headless is still registered and still claimable at the next mount" — was
true only for about one second.

**What the user actually experienced, therefore, was not 120 s of
silence. It was that a headless wake could not use ANY risk-tagged tool:
every one was auto-refused within a second, and nothing ever said so.**
Fail-closed, and unusable.

---

## 2. The red, and its before/after

`Tests/UI/test_console_headless_approval.py` — 14 tests. Seven were red
at the merge-base, each for its own reason:

| Test | RED at `e112798f1` |
|---|---|
| `…toasts_app_wide_and_is_resolvable` (e2e, real navigation) | `nothing surfaced app-wide; the user is on LibraryScreen and sees: ''` |
| `…announces_through_the_app_not_the_screen` | `a round armed with no view announced nothing app-wide` |
| `…mounts_a_round_armed_while_detached` | `attaching a view left the armed round invisible` |
| `…two_headless_rounds_share_one_payload_slot…` | no announcements at all (`[]`) |
| `…denies_a_round_armed_while_detached_too` | `the headless round self-denied instead of waiting` |
| `…a_configured_deadline_still_expires…` | `{'write_file': 'deny'}` at ~1 s, not `'timeout'` at 2 s |
| `…the_risk_floor_still_raises_a_card…` | the floored round arms, but is never announced |

**AFTER: 14 passed.**

The e2e is the whole chain through production: real DBs, real
`ChatScreen`, real navigation both ways, the round armed on a plain
worker thread, the toast read off the **mounted `Toast` widget on the
Library screen**, the card read off the **mounted `.approval-row`
widgets**, and the verdict delivered by **pressing the card's fast
Approve button** — never by calling `resolve_pending_approval` directly.

Two harness facts that a weaker test would have hidden, both measured:

- **Textual's `run_test` defaults `notifications=False`**, which makes
  `Screen._extend_compose` skip the `ToastRack` entirely. A toast can
  never mount under the default. A test asserting on `app._notifications`
  would have passed while proving nothing reached a screen; the e2e
  passes `notifications=True` and asserts on the rendered widget.
- **`Toast` is a `Static` that never calls `update()`**, so its
  `renderable` is empty. The first version of the toast-reading helper
  reported "no toast" for a toast that was on screen. It reads
  `Toast.render()` now.

---

## 3. What now surfaces, and through which seam

| Signal | Seam | Notes |
|---|---|---|
| **Toast**, app-wide | `ConsoleChatController._announce_detached_approval` → `App.notify(..., severity="warning")` | `App.notify` is documented thread-safe (it posts a message), so no `call_from_thread` marshal — and it renders on **whatever screen the user is on**, which is the only seam that can reach them. The screen-owned seam (`ChatScreen._park_console_approval`) is `None` while detached, by the viewless landing's design. |
| **Badge** (`NEEDS_APPROVAL`) | `add_pending_round` | Needed **no change**: `request_mcp_approvals` already called it unconditionally, before consulting any UI hook. Pinned rather than built. |
| **The card, at attach** | `ConsoleRuntime.remount_pending_approval` → `ConsoleChatController.remount_pending_approval_for_active_session` | Re-derives from `_parked_approval_payloads` — the same single source of truth `switch_session` uses, not a second copy. |

The message: `Agent in <session title> needs approval to use a tool. Open
Console to review -- nothing runs until you answer.` The title is
`escape_markup`-ed (`App.notify` renders content markup by default). An
app double without `notify` is silently skipped, and a raising/
incompatible `notify` is logged — **a missing toast must never become a
missing approval.**

The detached branch is chosen by `_approval_view_is_detached()`, which
reads the two card **seams** (`set_pending_approval` /
`park_pending_approval` both `None`) rather than `ConsoleRuntime.view`.
Deliberate: those two slots are what an announcement would travel
through, `detach_view` clears them together, and asking the runtime
instead would be wrong in exactly the case this exists for — a controller
whose seams are unwired for any other reason still surfaces nothing while
claiming a view.

---

## 4. Making the round survivable — the one semantic change

The surfacing alone is useless if the round dies in a second. The change
that makes "resolvable by opening Console" true is in
`_bind_visit_cancel_signal`:

```python
if self._disposed or not self._shutdown_requested.is_set():
    return self._shutdown_requested
event = self._headless_visit_cancel
if event is None or event.is_set():
    event = threading.Event()
    self._headless_visit_cancel = event
return event
```

**This is the same category error the wake-fires landing fixed one layer
up.** `_shutdown_requested` answers "did the visit that armed this round
end?". A round armed while the runtime is DETACHED was not armed during
any visit, so answering to that Event answers a different question —
exactly as `_attempt` was reading "a visit ended" as "the app is
exiting". The deferred Event stands in for the visit that has not
happened yet.

**Nothing about the fail-closed posture moved.** A headless round is
still ended by:

- **`leave_console()`** — set by `_cancel_headless_rounds()`. By then the
  user has had a Console visit in which to answer it and navigated away
  instead, which is precisely the case AC#2's rule is about.
- **`begin_shutdown()`** (app exit) — same call. A DISPOSED controller is
  excluded from the deferred binding entirely, so an app-exit round keeps
  the permanently-set Event and denies at once.
- **this run's own cancel event** — untouched.
- **a configured positive `[mcp] approval_timeout_seconds`** — untouched,
  and pinned: a 2 s deadline still expires a detached round in under 6 s.

And no path returns an approval without a human: pinned across the
deadline and app-exit paths by
`test_no_headless_path_returns_an_approval_without_a_human`.

**The deadline is not paused or extended while detached.** That was
plan Task 5's explicit instruction and it is honoured literally —
`timeout_seconds` is read and applied exactly as before.

---

## 5. The attach-time card, and the ordering bug found doing it

`attach_view` re-derives the card **only on a NEW claim**
(`previous is not view`), not on every `_ensure_console_chat_controller()`
call. That gate is load-bearing, and mutation M5 proved it: removing it
fails the **e2e**, because re-pushing the payload rebuilds the card's rows
and the fast-Approve press then lands on a superseded generation — i.e.
the user clicks Approve and nothing happens. Exactly the "discard a
half-made decision" hazard, caught by a user-visible consequence rather
than by an assertion about call counts.

That alone was not enough, and the reason was measured, not guessed:

> `ChatScreen._restore_native_console_state` does a plain
> `self._task_resume_state = TaskResumeState.from_dict(payload.get(
> "task_resume_state"))`, and `_complete_screen_navigation` runs
> `restore_state` on the incoming screen — whose `_ensure_console_chat_
> store` call is what triggers `attach_view` in the first place. So the
> attach-time mount ran, set the card, and was **overwritten microseconds
> later by the snapshot**.

`ChatScreen.restore_state` now re-derives from the controller *after* the
snapshot lands. That is the continuity landing's own principle applied
one slot further: the app-owned controller is the only source of truth
for what is armed, and a view snapshot of it is a second one.

Scoped deliberately: it **mounts** an armed round, it does not **clear** a
stale one. A snapshot carrying a `pending_approval` for a round that has
since resolved still restores a dead card — clicking it resolves nothing,
because `resolve_pending_approval` fails closed on the missing round id.
That is a pre-existing defect on the same path, I reproduced no red for
it, and the discipline is reproduce-red-before-fixing, so it is named in
§12 rather than quietly swept up.

---

## 6. The two-round verdict: FILED, not fixed — and asserted against

`test_two_headless_rounds_share_one_payload_slot_and_only_one_mounts`
executes what happens with two rounds armed for one session while
detached:

- both rounds are independently **registered** (verdicts and the badge
  are round-keyed);
- both are independently **announced** — a sibling round never silences a
  new one;
- **only the LAST-armed round has a payload to mount**, because
  `_parked_approval_payloads` is a single slot per session
  (`request_mcp_approvals`' own `finally` documents this as an "accepted
  scope limitation");
- both still resolve correctly by round id, and the older one keeps the
  badge lit until it does.

**This task does not fix it.** Per-round payload storage changes the
card's mount/park contract for every caller, mounted rounds included,
which is a wider change than making the headless case surfaceable. It
stays filed as **task-15661**, is now a measured fact with a test that
will fail the day someone fixes it, and is stated in the User Guide
rather than hidden.

---

## 7. Safety pins

Each is a test in the new file; each was run, and §8's mutations show
which ones bite.

| Pin | Test |
|---|---|
| Leaving Console still denies a round armed DURING the visit (AC#2) | `test_leaving_console_still_denies_a_round_armed_during_the_visit` |
| A headless round is deferred, never immortal: attach-then-leave denies it | `test_leaving_console_denies_a_round_armed_while_detached_too` |
| App exit denies a detached round | `test_app_exit_denies_a_round_armed_while_detached` |
| A configured deadline still expires a detached round on schedule | `test_a_configured_deadline_still_expires_a_headless_round` |
| No automatic path ever returns an approval | `test_no_headless_path_returns_an_approval_without_a_human` |
| A wake delivery cannot resolve a pending card | `test_a_wake_delivery_cannot_resolve_a_pending_headless_round` |
| The approval FLOOR still applies headless — a real risk-tagged tool asks | `test_the_risk_floor_still_raises_a_card_in_a_headless_turn` |
| A mounted round does not get a second, app-level toast | `test_a_round_armed_with_a_view_attached_does_not_double_announce` |
| Attach with nothing armed mounts nothing | `test_attaching_a_view_with_no_armed_round_mounts_nothing` |

The floor pin drives the **real** `ReadFileTool` (`risk_tags
("reads",)`) through the **real** `BuiltinToolGate` and the **real**
`build_tool_review_hook`, wired to a detached controller exactly as
`_run_agent_reply` wires it
(`functools.partial(self.request_mcp_approvals, session_id=...)`), on a
worker thread. The hook composition never consults the submission origin,
so "the floor still applies in a woken turn" is exactly this.

---

## 8. Mutations run, killed, and the one that survived

| # | Mutation | Result |
|---|---|---|
| M1 | `_bind_visit_cancel_signal` always returns `_shutdown_requested` (the pre-fix code) | **KILLED** — 3 tests: e2e, `…detached_too`, `…configured_deadline…` |
| M2 | `_cancel_headless_rounds` sets nothing | **KILLED** — 3: `…detached_too`, `…app_exit…`, `…without_a_human` |
| M3 | drop the app-wide announcement | **KILLED** — 4: e2e, `…announces_through_the_app…`, two-round, floor |
| M4 | `_approval_view_is_detached()` always True | **KILLED** — `…does_not_double_announce`; also HANGS `Tests/UI/test_console_mcp_approval.py::test_request_mcp_approvals_zero_timeout_keeps_round_armed_for_late_decision` (no card ever mounts, nothing can answer) |
| M5 | remove the `claimed` gate on the attach remount | **KILLED** — the e2e: the rebuilt rows make the user's Approve press a no-op |
| M6 | drop the `restore_state` re-derive | **KILLED** — the e2e |
| M7 | mount even when nothing is armed | **KILLED** — `…with_no_armed_round_mounts_nothing` |
| M8 | stop dropping `_headless_visit_cancel` after setting it | **SURVIVED** — see below |
| M9 | remove BOTH `_headless_visit_cancel` defences (the drop *and* the `is_set()` guard) | **KILLED** — exactly the new property test |
| M10 | `_is_session_cancelled` ignores the visit Event entirely (fail-open) | **KILLED** — 5 tests in the new file |

### M8's lesson, and what investigating it found

M8 survived the whole file. Investigated rather than patched: the drop is
**redundant** with `_bind_visit_cancel_signal`'s own `event.is_set()`
guard — a round arming after a leave gets a fresh Event either way — so
neither line is individually killable. An equivalent mutation.

What that redundancy hides is a real property, so the fix was to pin the
property rather than the line:
`test_a_second_headless_round_after_a_leave_is_not_born_denied`. Removing
both defences (M9) makes a second headless round inherit a pre-fired
Event and self-deny in ~1 s — silently restoring exactly the behaviour
this task removed — and M9 kills that test and nothing else.

### The second finding: two AC#2 pins were vacuous

M10 (delete the visit-cancel check outright, fail-open for every
session-scoped round) killed 5 tests in the new file — and left
`Tests/Chat/test_console_runtime_lifetime.py` **14/14 green in 0.98 s**,
which is less than one poll interval. Its two approval-round pins —
`test_leaving_console_still_denies_a_parked_approval_round` and
`test_a_round_from_the_previous_visit_is_not_resurrected` — build the
controller with `app is None`, and **ADR-067 added a no-`app` guard that
denies every name on the spot**, so those rounds stopped reaching the
poll loop when that ADR landed. They were passing on the guard's verdict.

That matters here specifically: they are the pins for the exact signal
this task re-binds. Both now wire a `call_from_thread` app; the file runs
in 2.81 s, and M10 re-applied fails **both** of them.

### A third, smaller one: a precondition that observed the wrong write

M3 made `…mounts_a_round_armed_while_detached` fail deterministically —
a "kill" with nothing to do with the mutation. Traced:
`request_mcp_approvals` registers the round *before* retaining its
payload, with `_resolve_mcp_approval_timeout_seconds()` in between — a
`get_cli_setting` read that, on a cold per-test config dir, **creates the
config file**. The precondition waited on the registration alone and
returned inside that file-I/O window. The mount depends on the payload,
so that is what it observes now.

---

## 9. Gate

Every count below was read from the run's own summary line. The
import-provenance probe (`Tests/test_probe_import_provenance.py`) is in
every invocation — it is the only thing proving the code under test is
this worktree rather than the venv's foreign editable install.

### 9.1 Targeted gate

**A gate-hygiene correction worth recording:** the first baseline
invocation passed both `Tests/Agents/` and `Tests/Agents/test_agent_runs_
wake_ledger.py`. pytest collapsed the directory argument against the
more specific file and collected **283 tests instead of 1,733** — and
reported a perfectly healthy "282 passed". Reading the count is not
enough; the count has to be *checked against what you meant to run*.
Never pass a directory and a file inside it in the same invocation.

| Run | Baseline (untouched branch, `e112798f1` + the probe file only) | Final (branch) |
|---|---|---|
| provenance probe + `test_console_mcp_approval` + `test_console_headless_wake_fires` + `test_console_sync_outlives_screen` + `test_console_store_continuity` + `test_console_viewless_hooks` + `test_console_runtime_lifetime` + `test_console_runtime_ownership` + `test_screen_residency` + `Tests/Agents/` (1,458) + the 12 wake suites + the 5 skill-confirm suites | **1732 passed, 1 skipped** (267.1s) | **1747 passed, 1 skipped** (352.4s) |

**Zero failures on either side.** The +15 is exactly accounted for: the
14 tests of the new file plus the step-1 probe (the final run adds both;
the skip is `test_probe_p4_headless_approval_cost`, which needs
`--run-slow` and by design burns a full deadline).

### 9.2 The whole Console population, one process

Because per-file runs cannot see app-lifetime bugs (the fires landing's
§11 found a production self-kill exactly that way), the whole Console
population ran in ONE process on both sides: every
`Tests/UI/test_console_*.py` + `test_screen_residency` + the provenance
probe. Both sides ran concurrently on the same machine (plus three
foreign `pytest` processes from other sessions throughout), so the
comparison below is of failure *sets*, not durations.

| | Branch (`f7ef432b5`) | Baseline (`.worktrees/headless-approval-base` @ `e112798f1`) |
|---|---|---|
| Files | 165 | 164 (no `test_console_headless_approval.py`) |
| Collected | 3,404 | 3,390 |
| Result | **27 failed, 3377 passed** (3562.97s) | **23 failed, 3367 passed** (3553.41s) |

`comm -13` over the sorted node-id lists is **empty**: every one of the
baseline's 23 failures is also on the branch, unchanged. The arithmetic
closes exactly — +14 collected (the new file), +10 passed, +4 failed.

**All 14 of the new file's tests passed inside the population.**

### The 4 branch-unique failures are an artefact of MY editing, not a regression

All four are
`Tests/UI/test_console_prompts_controller.py::test_screen_keeps_a_real_
delegation_for_every_outside_caller[...]`, which does
`inspect.getsource(getattr(ChatScreen, name))` and asserts the body is a
thin forwarder. Its failure text gives the game away — it read the wrong
method entirely:

```
>       assert "self._prompts." + name in body
E       assert ('self._prompts.' + '_console_command_apply_system') in
        '    def _insert_prompt_text_into_composer(self, text: str, *, replace: bool) -> bool:\n ...'
```

The parametrisation has six names. **The two that passed are defined at
`chat_screen.py:5330` and `:6264`; the four that failed are at `:16684`,
`:16790`, `:16794` and `:17198`** — i.e. exactly those *after* line
14903, which is where commit `a3f04b280` (a comment-only correction, net
+7 lines) landed **while the hour-long run was in flight**. Each method's
`co_firstlineno` was fixed at import, `inspect.findsource` calls
`linecache.checkcache` and re-reads the changed file, and every method
below the edit reports source that is 7 lines off.

Re-run on a stable tree, same branch, same commit:
**`Tests/UI/test_console_prompts_controller.py` → 37 passed.**

**Lesson, recorded because it cost an investigation:** do not commit to a
file the running suite imports. An hour-long single-process run reads
source off disk lazily (`inspect`, `linecache`, tracebacks), so an edit
mid-run manufactures failures that look like yours and are not — and the
tell is a line-number-ordered split in the parametrisation, not the
assertion text.

---

## 10. Deliberately not done

- **The two skill confirms are untouched, and are now inconsistent with
  the approval card.** `request_skill_install_confirm` /
  `request_skill_script_confirm` fail closed *immediately* when their
  seam is `None` (the viewless landing's documented contract), so in a
  headless turn a skill install is refused in silence while a tool
  approval now waits and asks. That is a defensible difference — a skill
  install is a much larger grant — but it is a difference, not a
  design, and nobody has ruled on it. Named here rather than quietly
  extended.
- **The single-payload slot** (task-15661) — §6.
- **Launch / first-boot wake** (plan Task 6), **the invariant gate**
  (Task 7) and **the docs sweep** (Task 8) are out of scope; only the
  User Guide sentence about *this* behaviour was written (§11).
- **The page's "Verified against" stamp is not touched.** This slice
  verified one paragraph, not the page; claiming a whole-page
  verification I did not do would be worse than a stale stamp.

---

## 11. The User Guide sentence

`Docs/User_Guide/console/agent-runs-and-tools.md` said:

> And **a headless wake that needs an approval card parks it**: the card
> is retained and is claimable when you next open Console, but nothing
> surfaces it while you are away, and if nobody answers it the request
> denies itself at the `[mcp] approval_timeout_seconds` deadline
> (default 120s).

Three claims, all wrong before this branch: the default is `0`, not
120 s; the request was refused in ~1 s by the ended visit's Event, not by
any deadline; and nothing was retained after that. It now says:

> **A headless wake that needs approval asks you, wherever you are.**
> When a woken turn reaches a tool that requires your approval, a toast
> names it on whatever screen you're on ("Agent in “…” needs approval to
> use a tool. Open Console to review — nothing runs until you answer."),
> the session picks up its usual approval badge, and the card is waiting,
> already mounted, the moment you open Console. The tool does not run
> until you answer it. Nothing auto-approves: navigating away from
> Console again denies the request (the same rule as any card you leave
> unanswered), and so does quitting the app. If you have set a positive
> `[mcp] approval_timeout_seconds`, it still expires the request on
> schedule — being away does not buy the request extra time. The shipped
> default is `0`, which means no deadline: the request waits for you. One
> limitation: if a woken turn arms two approval rounds for the same
> conversation, only the most recent one has a card to mount; the older
> one still has to be answered, and until it is, it keeps the badge lit
> (task-15661).

The lead-in above it ("Two honest limits remain") became "One honest
limit remains" — the launch-wake limit is still real and still Task 6's.

---

## 12. Concerns

1. **A headless approval round can now wait indefinitely at the shipped
   default.** That is ADR-067's accepted posture for the mounted case
   ("prompts should not expire on users who step away") applied to the
   case where the user has *definitely* stepped away — and the toast is
   what makes it fair. But it does mean an agent worker thread can be
   parked on a human decision for as long as the app runs. Quitting or
   re-visiting-and-leaving Console both end it; nothing else does.
2. **Skill confirms diverge from tool approvals headlessly** (§10). Worth
   an owner ruling before someone discovers it as a bug.
3. **The e2e depends on `run_test(notifications=True)`.** Any future test
   asserting on toasts must pass it; the default silently removes the
   `ToastRack`. Recorded in the test's own comment because it is
   invisible otherwise.
4. **`_approval_view_is_detached()` is a seam predicate, not a runtime
   predicate.** A controller that legitimately has one seam wired and not
   the other would take the mounted path and surface nothing. No such
   caller exists today (the slots are cleared together), and the
   alternative — asking `ConsoleRuntime.view` — is wrong in the case that
   matters. Stated so the next reader does not have to re-derive it.
5. **A stale restored card is a pre-existing defect on the path I
   touched, left alone.** `ScreenStateStore`'s `task_resume_state`
   snapshot can carry a `pending_approval` for a round that resolved
   after the snapshot was taken (e.g. denied by the very
   `leave_console()` that preceded it), and `restore_state` mounts it. The
   card is dead: `resolve_pending_approval` fails closed on the missing
   round id, so clicking it does nothing at all. The one-line fix would be
   to clear when nothing is armed — but no red was reproduced for it and
   it is outside this task's ACs, so it is reported rather than bundled.
6. **The parallel population runs shared a machine.** Branch and baseline
   Console populations ran concurrently, so both carry the same
   contention. The comparison that matters is the failure *set*, which is
   what §9.2 compares; absolute durations are not comparable to the fires
   landing's.
