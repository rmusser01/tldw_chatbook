# Supervisor Fleet PR 3a-2 — Auto-Wake + Cross-Conversation Notification

Spec: `Docs/superpowers/specs/2026-08-08-supervisor-agent-fleet-design.md` §3
invariant 5 (CORRECTED 2026-08-11 — auto-wake is required), §7 Notifications
(cross-conversation by requirement). PR 3a-1 (merged, `d5445a4c1`) lets a child
outlive its turn; this PR makes a finished child (a) wake its supervisor so it
can act on the result, and (b) reach the user wherever they are.

## The architectural truth this plan is built on

The seam survey (2026-08-13) established: the Console bridge and controller are
**per-screen** (`UI/Console_Modules/agent.py:484-519`; screens never cached,
`app.py:7695`; `ChatScreen.on_unmount` → `controller.shutdown()`,
`chat_screen.py:14535-14550`). "Cross-conversation" today means cross-session-
WITHIN-Console. A wake firing while the user is on Library has no controller to
fire into.

**Ruling for this PR:** build the entire chain — hardened completion signal,
durable unseen-completion mark, app-wide indicator, wake machinery — such that
the wake fires **immediately whenever a Console controller exists** (the normal
hop-between-sessions case) and is **staged durably and claimed at next Console
mount** when one does not. The remaining gap — a supervisor that acts with no
Console mounted at all ("headless wake") — requires moving bridge+controller
ownership above the screen; every artifact of this PR is substrate that version
needs too, the only delta is where the wake runs. That gap is **named in the PR
body and filed as a follow-up, not papered over.** If the owner wants headless
now, this PR is still the right first half.

## Global constraints

- Env rules as 3a-1 (no `git stash`; Edit-based restores; pytest-only python;
  never touch `~/.config/tldw_cli`; no worktrees under /private/tmp; regenerate
  the CSS bundle, never hand-edit).
- **Reproduce red before fixing; mutation-test every new test.** Six confident
  readings have been refuted by execution in this programme.
- **A report's "already handled / out of scope" is an untested claim** unless a
  test proves it (lessons-testing-evidence.md, 3a-1 entry).
- A wake notice is **never user input and never approval**: it must not be
  rendered as a USER row, must not clear the composer, must not satisfy any
  approval gate, and risk-tagged tools stay floored to `ask` in a woken turn.
- Spend on wake turns is bounded by existing containment (§5) + the existing
  `max_parallel_runs` cap; `autowake_enabled` (default ON) is the kill switch,
  following `subagents_outlive_turn`'s recorded reasoning (`agent_service.py:115-125`).
- Tests assert rendered geometry, not DOM presence, for any UI surface.

## Verified seam map — cite these, do not re-derive

| Seam | Where | Fact |
|---|---|---|
| Last-child-done signal | `console_agent_bridge.py:3281-3325` `_child_run_scope`, wired `:2989` | Fires on the CHILD's thread at scope exit; sole consumer `_close_post_turn_change_window`; second consumer attaches cleanly at `:3323-3325` |
| Ordering hazard | `agent_service.py:1819-1822` vs `:1846-1852` | Scope exits BEFORE `fleet.finish()` — coordinator still says `running` at signal time. DB is terminal on the happy path (`_persist` inside scope, `:2773`) but NOT on the raise path (`set_status` in `run_child`'s finally, `:1878-1884`) |
| Scope identity | `:2989` partial | Carries only `conversation_id` + adapter; needs `session_id` threaded |
| SYSTEM rows dropped from payloads | `console_chat_controller.py:10519-10522` | Only USER/ASSISTANT reach the model; mirrored `:5636,:5661,:5824` |
| System-prompt fold | `:10242-10277` + `console_agent_bridge.py:333-357` | task-1531 precedent for folding extra text |
| Evidence prefix | `:7442-7486` | One-shot marked block on final USER message |
| `turn_bundle_block` | bridge `:2311`, applied `:3022-3035`, passed `:9582` | Per-turn extra block, agent path only, never mutates caller's list |
| Send boundary | `submit_draft` `:2313`; **appends a USER row** | The invariant-5 collision; origin enum `console_chat_models.py:41-45` (MANUAL/QUEUED) is the designed extension point; queue's authorization-token pattern `:2356-2362`, `_submit_queued_entry` `:7530-7546` |
| Send gating | `is_send_allowed` `console_chat_models.py:350-358`; `_active_run_rejection` `:10906`; `max_parallel_runs` default 3 `console_chat_models.py:178` | A wake must wait-and-retry; `ConsolePromptQueueCoordinator._drain_waiting` `:366-445` is the pattern |
| Origin metadata | `MessageMetadata` `message_metadata.py:50-104`; `messages.metadata_json` (no schema change needed) | `from_json` DROPS unknown keys — marking is not visible to older builds (accepted, local-only display) |
| App-wide surfaces | toast: `_notify_console_run_outcome` `chat_screen.py:21580-21615` → `app.notify` (renders on ANY screen); app-object slot precedent `_console_fleet_teardown_notice` `:14430-14455,:14546-14550`; nav badge: **none exists** (`main_navigation.py`) | Toast is the only working cross-screen surface today |
| Cross-session markers | `ConsoleRunMarker` `console_chat_models.py:96-122`, stamped `:10726-10733` for non-active sessions | Session tabs + sidebar rows already show per-conversation outcome glyphs |
| Deep link | `PendingHandoffStore`/`HandoffChannel` `pending_handoff_store.py:48-59`; `open_chat_with_handoff` `app.py:5458-5484` | No channel carries a conversation id into Console yet; pattern established |
| Durable no-DDL bit | `conversation_local_marks` `ChaChaNotes_DB.py:938-947`; allowlist `conversation_local_marks_service.py:35` (`STARRED` only) | Adding a mark type = one allowlist line; local-only, never synced |
| Restart sweep | `reconcile_orphaned_runs` `AgentRuns_DB.py:714-790`, lazy once-per-process at first bridge construction | Wake does not survive restart (invariant 3: no new persistence machinery); the MARK does |
| UI clock hazard | `chat_screen.py:15855-15866` + `console_prompt_queue_coordinator.py:176-181` | The 0.2s poll SELF-STOPS when only survivors run (a survivor occupies no slot) — root cause of task-15664; any new indicator relying on the poll is dead exactly when needed |
| Usage re-attach | `_watch_post_turn_usage` `:8664-8697`, consumer `:8698-8757`, chip fold `chat_screen.py:8508-8515` | Poll-only today; idempotence pre-certified in its docstring; task-15660/15667's fix is the second consumer of the same signal |
| Delivery IOU | `agent_service.py:2038-2044` | "collecting a foreign child's RESULT into this turn's history is delivery, which is PR 3a-2's job" |

## Absorbed follow-ups
**15660** (usage re-attach on last-child-done), **15667** (spend missing from
row/exports — same fix, close as satisfied or duplicate), **15664** (elapsed
does not tick — its AC#2 "no timer repaint when nothing is live" is a design
constraint on this PR's own tick). **15661** only if wake work touches approval
parking; otherwise flag the widened window in its task file.

---

### Task 1: Establish screen-teardown truth by execution, and pin the signal's ordering

No fix ships in this task; it produces the facts Tasks 4-5 are built on, plus
one missing guard.

- [ ] By EXECUTION (headless harness or live): when `ChatScreen` unmounts with
  a survivor running — does the child thread keep running? Does its DB terminal
  write land? Does its store append land in the persisted conversation? What
  does the task-1143 teardown notice count post-3a-1? Do the controller's
  shutdown cancel-events reach the child (does it see cancellation and stop)?
  Record the answers in the ledger; they decide whether Task 4's durable mark
  is written from a path that survives teardown or must be written earlier.
- [ ] Pin the scope-exit ordering with a test (survey hazard 8: nothing names
  `_child_run_scope`/`_live_child_counts` today): last child's scope exit fires
  the consumer exactly once, after the DB row is terminal on the happy path;
  and a test documenting the raise path's later `set_status` (the wake must
  read the DB and tolerate not-yet-terminal on that path).
- [ ] Gate + commit.

### Task 2: Harden and identity the completion signal

- [ ] Thread `session_id` (and run identity) through the `_child_run_scope`
  partial (`:2989`).
- [ ] Give 3a-2 a signal that fires **after** the row is terminal on BOTH
  paths: either move/duplicate the hook after `run_child`'s finally
  (`agent_service.py:1878-1884`) or have the consumer read the DB with a
  bounded settle. Decide from Task 1's evidence; do not disturb the change-
  window consumer's existing ordering (its behaviour is pinned by
  `test_opening_a_window_whose_last_child_already_left_closes_it_at_once`).
- [ ] Fan-out seam: one signal, N consumers (change window, usage re-attach,
  wake scheduler, indicator), each isolated — one consumer raising must not
  starve the others. Mutation-test that isolation.
- [ ] Gate + commit.

### Task 3: Usage re-attach (absorbs 15660 + 15667)

- [ ] On last-child-done: re-attach the originating assistant message's usage
  (recompute-all + replace — idempotence already pinned by the 6b test);
  message row and exports now include survivor spend; the chip's unattributed
  line falls to zero (15660 AC#4); re-attach twice = same stored total.
- [ ] 15667's ACs re-verified against this; close or dup it honestly.
- [ ] Gate + commit.

### Task 4: Durable completion mark + app-wide indicator

- [ ] New `conversation_local_marks` mark type (allowlist + service APIs; no
  DDL) set on child-terminal from a path Task 1 proved survives teardown;
  cleared on delivery (Task 5) or on the user viewing the conversation.
- [ ] App-wide toast on child completion via the app object (thread-hop
  through `call_from_thread`; `_notify_console_run_outcome` is the copy/shape
  precedent) — fires regardless of current screen; action/copy names the
  conversation; deep-link handoff channel (new `HandoffChannel` member)
  opens Console at that conversation.
- [ ] Console-side: run-marker glyphs (`CONSOLE_RUN_MARKER_GLYPHS`) driven for
  fleet completions on non-active sessions; a session-tab/badge surface for
  "unseen completion" backed by the durable mark (so it survives restart).
- [ ] **The tick:** a timer that runs while survivors are live and stops when
  none are (15664 AC#2 verbatim), fixing the elapsed display AND making every
  surface above actually update — the 0.2s poll self-stops in exactly this
  state today (survey hazard 1). Geometry-asserted tests; a clock-advance test
  that fails when frozen (15664 AC#3).
- [ ] Gate + commit.

### Task 5: The wake itself

- [ ] New `ConsoleSubmissionOrigin` member (e.g. `AGENT_WAKE`) with a
  coordinator-issued authorization (queue-token precedent `:2356-2362`):
  **no USER transcript row** — the notice reaches the model via
  `turn_bundle_block` (or system-fold; decide against the code, record why),
  and the transcript shows a SYSTEM-class row marked as machine-origin
  (`MessageMetadata` origin field), never clearing the composer.
- [ ] Wake content: the finished child(ren)'s results from the DB, fenced,
  explicitly labelled "background sub-agent completion — not user input; not
  approval for anything". Coalescing: one wake bundles ALL undelivered
  completions for that conversation; the durable mark is the undelivered bit;
  no double-delivery, no lost completion (mutation-test both).
- [ ] Scheduling: fire when `is_send_allowed` + not queue-controlled + under
  `max_parallel_runs`; otherwise wait-and-retry (`_drain_waiting` pattern). If
  no controller exists (Console unmounted), the mark stages the wake; next
  Console mount claims it (teardown-notice/handoff precedent). Wake turns for
  a conversation the user is actively typing in must not steal the slot ahead
  of the user's own send — user wins ties.
- [ ] Config: `autowake_enabled`, default ON, `_setting` + `_coerce_*` shape,
  commented line in `config.py:2749`'s block; kill switch honoured at both
  fire points (immediate and mount-claim).
- [ ] Safety: a woken turn grants no new authority — approval gates and risk
  floors unchanged (test: a risk-tagged tool in a woken turn still raises a
  card); wake chains bounded only by containment — assert a wake turn is a
  normal turn under every existing cap.
- [ ] Gate + commit.

### Task 6: Spec, docs, paperwork

- [ ] Fix the spec's internal contradictions (survey hazard 10): §11 lines
  ~491/498 and §10's PR map still carry the pre-correction rulings.
- [ ] User Guide: auto-wake behaviour, the notice's not-user-input marking,
  the kill switch, the cross-screen indicator, the honest headless limit.
- [ ] File the 3a-2 subtask under programme task-13154 (only 13154.1 exists);
  file the headless-wake follow-up; update 15660/15664/15667 per Tasks 3-4;
  ID hygiene: sweep all remotes + ghost-check, leapfrog, re-verify at merge.
- [ ] Gate + commit.

### Task 7: Battery + live verification

- [x] Battery vs pristine dev at this branch's own base; READ counts.
  (2026-08-13: Agents+Chat+agent_runs_db 5808 passed / 3 failed — all three
  pre-existing, two re-verified failing on a pristine worktree of
  `61f6ae575` itself; UI cluster 139 passed; tick/wake-wiring/handoff/
  discoverability 97 passed + the known `[size0]` red; collect-only 41,817
  / 0 errors.)
- [x] Live (lessons-live-verification.md binds; scratch config; Anthropic
  key): (1) child finishes while the user is in ANOTHER Console session —
  toast + marker + wake turn fires there, transcript shows the marked notice,
  supervisor's reply references the child's result; (2) child finishes while
  the user is on Library — toast appears there, returning to Console claims
  the staged wake; (3) `autowake_enabled=false` — no wake, indicator still
  works; (4) restart with a finished-but-undelivered child — mark survives,
  reconcile marks orphans `error`, no phantom wake, mount-claim delivers the
  terminal result; (5) user typing during a due wake — user's send wins.
  (2026-08-13: all five driven live against real Anthropic on an isolated
  scratch profile — see the sdd task-7 report for panes, DB stamps, and the
  three UI-layer findings filed as follow-ups.)
  *(Scenario 2 corrected 2026-08-14, wake-integrity arc, per the
  coordinator's design ruling for task-15971: a Console screen that is
  mounted but not being LOOKED AT — covered by a pushed screen, or with
  another session tab in front — delivers the wake OFF-VIEW immediately.
  That is the intended behavior: the user learns of it via the settle
  toast plus the `FLEET_UNSEEN` `◈` mark, which an off-view delivery
  leaves SET until the conversation is viewed.
  **Re-corrected the same day (task-16300):** the first version of this
  note justified the ruling with "a Chat screen can stay RESIDENT on
  nav-away — a navigation issued while a pushed screen sits above Chat
  pops the modal, not Chat". That was a screen-stack leak, not a nav
  model, and it is fixed — navigation reduces the stack to its content
  screen before switching, so leaving Console unmounts it. Scenario 2 as
  originally verified (stage on nav-away, claim at next mount) is
  therefore the contract again for nav-away, alongside restart / first
  boot; the off-view delivery ruling is untouched, since it never
  depended on residency.)*
- [x] Ledger + PR body: the headless limit stated plainly. (Ledger written;
  PR body is the coordinator's step — the headless limit text is staged in
  the task-7 report.)

## Deliberately NOT in this PR (→ 3b / follow-up)
- Headless wake (supervisor acts with no Console mounted) — requires moving
  bridge+controller ownership above the screen; this PR builds everything else
  it needs. Filed, named in the PR body.
- `send_to_agent` + mailboxes; finished-agent continuation (3b, steering).
- Nav-bar badge infrastructure (no mechanism exists today; the toast + mark +
  session glyphs deliver the requirement; a persistent nav badge is UI-polish
  scope for phase 4).
