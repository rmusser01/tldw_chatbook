# Task 0 report — four execution probes (task-15860, headless wake)

**No production code was written.** This is evidence only.

- Branch/base: detached worktree at `origin/dev` `7a6e8f804`
  (`.worktrees/headless-probe`).
- Probes: `Tests/UI/test_probe_headless_wake_p1_continuity.py`,
  `Tests/UI/test_probe_headless_wake_p2_p3_p4.py`,
  `Tests/test_probe_import_provenance.py`.
- Runner: `.venv/bin/pytest <files> -p no:randomly -q -s`, cwd = the
  worktree. **Every probe was executed twice; every verdict below was
  identical on both runs.** Run 1: `7 passed in 144.64s`. Run 2:
  `7 passed in 141.31s`. The extended P3 (P3b) was executed after run 2:
  `4 passed in 130.71s`.

## Trap found before any probe ran

The venv's editable install resolves `tldw_chatbook` to a **foreign
worktree**:
`.venv/lib/python3.12/site-packages/__editable___tldw_chatbook_0_1_8_0_finder.py`
maps `tldw_chatbook` → `.worktrees/task-2512-mcp-unified/tldw_chatbook`.
Every result in this report would have been about someone else's branch if
that had won. It does not: setuptools' editable finder is *appended* to
`sys.meta_path`, so the stdlib `PathFinder` (searching the rootdir pytest
prepends, because `Tests/` is a package) resolves first.

Proven, not assumed — `Tests/test_probe_import_provenance.py`:

```
PROBE imported tldw_chatbook from: .../.worktrees/headless-probe/tldw_chatbook/__init__.py
PROBE worktree root:               .../.worktrees/headless-probe
1 passed
```

---

## P1 — the off-ramp. **VERDICT: the plan's reading is CONFIRMED. Design A survives.**

**Question.** Console history continuity across a nav-away is claimed to be
an in-memory `ScreenStateStore` snapshot, not the DB. If DB rows written
while Console is down DO surface correctly at remount, designs B/C get far
cheaper and A may be unjustified.

**Method (all through production paths).** Real `TldwCli`
(`Tests/UI/app_factory._build_test_app`), real on-disk `CharactersRAGDB`,
real `ChatScreen` pushed and mounted, one real `submit_draft` to persist the
conversation, then **the real navigation API**
(`app.handle_screen_navigation(NavigateToScreen("library"))` — real
`save_state`, real unmount, real `controller.shutdown()`), then a SYSTEM row
and an ASSISTANT row appended through the production
`ChatPersistenceService.create_message`, then `NavigateToScreen("chat")`
back.

**Executed output (run 2; run 1 identical modulo uuids):**

```
pre-nav DB rows: user:'first user message', assistant:'assistant one'
pre-nav active_leaf=14292047-...
after nav-away screen=LibraryScreen
headless appends: system=302206ed-... assistant=b820dae3-...
active_leaf after DB-only appends=14292047-... (unchanged=True)
post-return in-memory transcript: [('user', 'first user message'), ('assistant', 'assistant one')]
(a) headless rows in restored transcript:  system=False assistant=False
(a-dom) headless rows in rendered widgets: system=False assistant=False
post-return session persisted_conversation_id=0bce270a-...   <- same conversation
(b) provider payload: [('system', 'You are a capable assistant with optiona'),
                       ('user', 'first user message'),
                       ('assistant', 'assistant one'),
                       ('user', 'second user message')]
(b) headless rows in provider payload: system=False assistant=False
(c) final DB rows (id | sender | parent | content):
      3bc60661 | user      | None     | 'first user message'
      14292047 | assistant | 3bc60661 | 'assistant one'
      302206ed | system    | 14292047 | 'HEADLESS-SYSTEM-ROW-P1'
      b820dae3 | assistant | 302206ed | 'HEADLESS-ASSISTANT-ROW-P1'
      22a62346 | user      | 14292047 | 'second user message'     <- FORK
      76b0462f | assistant | 22a62346 | 'assistant one'
(c) final active_leaf points at: assistant 'assistant one'   (the forked branch)
(c) next persisted append -> FORKS away from the headless rows: True
```

**Answers.**

- **(a) Do the rows render?** No. Absent from the restored in-memory
  transcript and from the rendered widget tree. The restored screen is a
  different `ChatScreen` instance (asserted) on the same persisted
  conversation id (asserted), and it shows two rows where the DB has four.
- **(b) Are they in the next send's provider payload?** No. The payload is
  `system-prompt / user / assistant / user` — built entirely from the
  restored in-memory store.
- **(c) Active leaf and forking.** The DB-only append left
  `conversations.active_leaf_message_id` untouched (still the pre-nav
  assistant). The next persisted append parented itself to that pre-nav
  assistant, **forking the tree**: the headless `system → assistant` chain
  and the new `user → assistant` chain are siblings, and the leaf pointer
  then moved onto the new branch. The headless rows are stranded on a dead
  branch that no Console surface will ever show.

**Two objections, closed by execution.**

1. *"A real headless writer would also maintain the active-leaf pointer."*
   Variant probe: same sequence plus
   `db.set_conversation_active_leaf(conversation_id, headless_assistant_id)`.
   Result unchanged — `system=False assistant=False` in both the transcript
   and the payload, the next append still forked, and the pointer was then
   overwritten (`final active_leaf still the headless assistant: False`).
   The snapshot restore never consults the pointer.
2. *"Maybe nothing reads DB appends, mounted or not."* Control probe: the
   same out-of-band append **with Console still mounted** is also invisible
   (`live store: False`, `next payload: False`). The in-memory store is the
   source of truth for both the transcript and the payload in every case.
   This makes the finding stronger, not weaker: a DB-only headless delivery
   path is invisible unconditionally.

**Consequence.** Designs B and C (throwaway headless controller / DB-only
delivery) are rejected on evidence. The recommendation for design A stands,
and the DECISIONS.md premise "(1) … DB-only headless writes are invisible +
divergent at remount" is now executed rather than read.

---

## P2 — post-unmount fan-out on current dev

**Question.** PR 3a-2 Task 1 proved the *attention* consumer fires
post-teardown; the wake coordinator was added later. After `on_unmount`,
does `ConsoleFleetWakeCoordinator.on_fleet_drained` still record, and is its
captured loop still alive?

**Method.** Real screen + real nav away (shutdown NOT suppressed), then the
drain fired through **the bridge's own `FleetDrainFanout` from a plain
thread** — the production path, not the coordinator method directly.

```
runs_db present pre-unmount: True
captured loop pre-unmount is the app loop: True
fan-out registrations pre-unmount:  ['fleet-attention', 'usage-reattach', 'fleet-wake']
post-unmount _shutdown_requested set: True
P2(loop) captured loop post-unmount: is_none=False is_closed=False is_app_loop=True is_running=True
fan-out registrations post-unmount: ['fleet-attention', 'usage-reattach', 'fleet-wake']
P2(registry) recorded post-unmount (via the bridge fan-out): has_pending=True pending_ids=('3b1edef9-...',)
P2(delivery) wake turns reaching the provider after unmount: 0
P2(delivery) delivering_conversation_id=None
P2(ledger)   wake_delivered_at=None
```

**Answer: YES and YES.** The registration survives, the child-thread intake
records into the pending registry, and the captured loop is the app loop —
open and running. Delivery is refused, and the refusal is attributable to
exactly one line: `_attempt`'s `_shutdown_requested` gate
(`console_fleet_wake.py:460-462`). Nothing about the signal path dies with
the screen.

---

## P3 — how much already works if the controller merely survives

**Question.** Keep the controller alive artificially (skip `shutdown()`),
unmount the screen, settle a survivor. Does the wake reach `submit_draft`,
does the turn complete, and which screen-wired hook slots are touched with
no view?

**Method.** Identical rig; `ConsoleChatController.shutdown` monkeypatched to
a no-op for the navigation only. The screen genuinely unmounts (asserted:
`chat not in app.screen_stack`). Every controller attribute bound to a
`ChatScreen` method was wrapped with a recorder *before* the navigation, so
"touched" is measured, not inferred.

```
shutdown() was suppressed for this probe
post-unmount _shutdown_requested set: False
wake loop still the app loop: True
P3(delivery) wake turns reaching the provider after unmount: 1
P3(delivery) payload tail role='user' carries the child's result=True
P3(turn) wake_delivered_at stamped (turn accepted+committed): True
P3(transcript) rows in the (orphaned) store:
    [('user','first user message'), ('assistant','wake reply'),
     ('system','[Background sub-agent comple'), ('assistant','wake reply')]
P3(hooks) screen-bound controller slots (15 incl. delivery_ui_hook):
    _chat_dictionary_applier, _global_user_display_name, _library_provider_factory,
    _rag_capture_provider, _world_info_applier, notify_run_failure, notify_run_outcome,
    on_submission_accepted, park_pending_approval, set_pending_approval,
    set_pending_skill_install, set_pending_skill_script, wake_conversation_in_view,
    wake_user_priority_probe   (+ wake.delivery_ui_hook)
P3(hooks) TOUCHED during the wake turn:
    _chat_dictionary_applier, _world_info_applier, wake.delivery_ui_hook,
    wake_conversation_in_view, wake_user_priority_probe
P3(hooks) NOT touched:
    _global_user_display_name, _library_provider_factory, _rag_capture_provider,
    notify_run_failure, notify_run_outcome, on_submission_accepted,
    park_pending_approval, set_pending_approval, set_pending_skill_install,
    set_pending_skill_script
```

**Answer.** With the controller merely surviving, **the entire wake turn
already works headlessly**: `submit_draft` was reached, the model payload's
trailing `user` entry carried the child's `agent_runs.result`, the
transcript gained a machine-origin SYSTEM row and no USER row, and the
durable ledger stamped `wake_delivered_at` — which `_deliver` writes only
after real acceptance. Five of fifteen hook slots are touched; three are
the wake's own probes.

**Sizing.** The work is not "build headless wake". It is: stop destroying
the runtime, move continuity, and rebind five hooks.

**Hazard found, not inferred.** All five touched slots were still bound to
methods of a **dead** `ChatScreen`, and none raised. A silent wrong answer
from `wake_conversation_in_view` decides whether the `◈` unseen mark
survives (task-15971's entire mechanism), and one from
`wake_user_priority_probe` decides whether the user wins a tie. Detach must
rebind these, not leave them dangling.

### P3b — the composition of P1 and P3 (the decision-relevant fact)

Same run, extended: after the surviving-controller wake completed, the
controller was really shut down and Console was re-entered through the real
navigation API.

```
P3b(db) rows persisted by the surviving-controller wake:
    [('user','first user message'), ('assistant','wake reply'),
     ('system','[Background sub-agent comple'), ('assistant','wake reply')]
P3b(remount) transcript the user sees on returning:
    [('user','first user message'), ('assistant','wake reply')]
P3b(remount) the wake notice survives the return: False
```

A wake turn that genuinely ran, spent money, and stamped the ledger is
**invisible** to the user on return, because the `ScreenStateStore` snapshot
was taken before it. This is the concrete cost of keeping continuity
screen-owned, and it is why design A's structural half (plan Task 3) is
required rather than cosmetic.

---

## P4 — headless approval cost

**Question.** With no UI wired, a risk-tagged tool in a wake turn: actual
wall time to denial, and the effective `[mcp] approval_timeout_seconds`.

**Method.** A `ConsoleChatController` with **no `app`, no
`set_pending_approval`, no `park_pending_approval`** (P3 established those
slots are untouched by a viewless turn, so this is the real headless shape).
The pending row is shaped as a risk-floored built-in — `server_key
"agent:builtin"`, `reason "risk_floored"` — the shape
`build_tool_review_hook` emits for a risk-tagged tool. A wake turn runs on
the same controller and therefore through the same
`request_mcp_approvals` gate.

```
controller.app wired: False
set_pending_approval wired: False
park_pending_approval wired: False
P4(config)   effective [mcp] approval_timeout_seconds = 120.0  (shipped default constant 120.0)
P4(measured) decisions={'builtin__write_file': 'timeout'}
P4(measured) wall time to verdict = 120.43s      (run 1: 120.47s)
control      injected 0.05s deadline -> elapsed 1.00s   (poll granularity is 1.0s)
```

**Answer.** **120.43 s** of real wall time, then the fail-closed `timeout`
verdict, with nothing surfaced to the user at any point. The effective
timeout is the shipped default `120.0`
(`console_chat_controller.py:259`; resolved through
`get_cli_setting("mcp", "approval_timeout_seconds", …)` at `:4325`, so a
config override moves it). The control at a 0.05 s injected deadline
returned in 1.00 s, confirming the cost IS the configured deadline plus the
1.0 s poll granularity (`console_chat_controller.py:262`) and nothing else.

Note on wording: the verdict word is `"timeout"`, not `"deny"`
(`console_chat_controller.py:4051-4053`); the refusal and its audit record
land downstream in the tool providers as `"denied-timeout"`. Fail-closed
either way.

---

## Verdict

**Design A survives P1.** The off-ramp was not taken: DB rows written while
Console is unmounted do not surface, are not sent, and fork the conversation
tree — and that holds even when the headless writer maintains the durable
active-leaf pointer, and even when Console is still mounted. Designs B and C
are rejected on evidence.

P2 and P3 shrink the remaining work considerably: the signal path, the
registry, the loop and the whole wake turn already survive teardown. What
does not survive is the runtime object and the continuity mechanism. P3b
shows those two are one problem, not two.

P4 prices the one deliberately-deferred safety behaviour at 120 s of silence
per risk-tagged call, which the User Guide must state rather than the user
discover.
