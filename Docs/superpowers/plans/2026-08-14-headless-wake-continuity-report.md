# Continuity report — the app-owned store owns Console history (task-15860)

Plan Task 3, the structural half of design A. Owner call #3 answered
**PROCEED** (`.superpowers/sdd/2026-08-14-headless-wake/DECISIONS.md`,
"OWNER ANSWERED 2026-08-14 (3)"), on the staging condition that the pure
ownership move landed separately first — which it did (PR #1648), followed
by the lifetime landing and the viewless-hook landing.

- Branch: `feat/task-15860-store-continuity`, worktree
  `.worktrees/headless-continuity`.
- **Merge-base: `31b0ef6a1`** (contains all four merged headless-wake
  landings). Every baseline below was measured at that commit, on the
  untouched tree — not against dev's tip.
- Predecessors: `…-task-0-report.md` (the executed probes),
  `…-task-1-report.md`, `…-lifetime-report.md`, `…-viewless-report.md`.

**One sentence:** Console message history stops travelling through
`ScreenStateStore`; the app-owned `ConsoleRuntime`'s `ConsoleChatStore` is
the only place it lives, so a turn that ran while Console was unmounted is
there when the user comes back.

---

## 1. The red, reproduced before anything was changed

Task 0's P3b, turned into a regression test and executed on the untouched
merge-base:

```
FAILED Tests/UI/test_console_store_continuity.py::
       test_a_wake_that_ran_while_console_was_unmounted_is_in_the_transcript
E   AssertionError: the wake notice is missing from the transcript the user
    sees on returning (P3b). Transcript was
    [('user', 'first user message'), ('assistant', 'assistant one')];
    the DB has 4 rows.
E   assert '[Background sub-agent completion — automated notice]'
        in 'first user message\nassistant one'
```

**Two of four rows**, exactly as P3b measured — and this time with no
artificial survival and no monkeypatched gate.

### How a genuinely headless wake turn is produced without touching the gate

`_attempt`'s `_shutdown_requested` refusal is the wake-fires-headless
slice and is untouched here, byte for byte. So the test does not make a
wake fire while detached; it uses the path production already has:

1. the survivor's settle fires through the real fan-out
   (`on_fleet_drained` from a plain thread) while Console is MOUNTED;
2. the provider's readiness probe (`resolve_for_send`) stalls — the
   everyday shape of a cold llama.cpp probe;
3. the user navigates away and answers the real "Leave Console?"
   `ConfirmationDialog` (a busy fleet raises it) by pressing **Leave**;
4. the probe is released. Both wake rows — the machine-origin SYSTEM
   notice and the assistant reply — are appended and persisted with **no
   Console mounted**, because `submit_draft` appends the notice at the
   acceptance point, *after* `resolve_for_send` returns, and because
   `leave_console` deliberately does not cancel an in-flight `AGENT_WAKE`
   turn (the owner ruling the lifetime landing implemented).

The ledger stamp (`agent_runs.wake_delivered_at`) and the four persisted
ChaChaNotes rows are asserted as harness preconditions, so the test can
never pass by the wake silently not happening.

**After the change: green.** The returning user sees all four rows, in the
store and in the rendered widget tree.

---

## 2. What still travels through `ScreenStateStore`, and what no longer does

`ChatScreen.save_state` still publishes a `native_console_state` payload.
Its contents changed:

| Key | Before | Now |
|---|---|---|
| `sessions` | every session, via `_console_session_to_state` | **gone** |
| `messages_by_session` | every message, via `_serialize_console_message` | **gone** |
| `active_session_id` | store's active session | **gone** |
| `version` | `NATIVE_CONSOLE_STATE_VERSION` | unchanged |
| `task_resume_state` | screen's `TaskResumeState` | unchanged |
| `image_view_modes` | screen's per-message image view overrides | unchanged |
| `library_rag_source_types` | Console-local RAG source scope | unchanged |
| `pending_console_launch` | staged live-work launch | unchanged |
| `console_evidence_sent_notice` | "evidence sent" memory | unchanged |

Everything left is **screen-instance** state that dies with the screen and
has nowhere else to live. Nothing left is conversation content.

`_restore_native_console_state` no longer calls
`ConsoleChatStore.restore_state`, no longer rebuilds sessions or messages,
and no longer rehydrates image bytes / attachments / generation metadata —
the live objects never lost them.

Two details that are easy to get wrong and are deliberate:

- **The restore still TOUCHES the runtime**, and that is load-bearing:
  `_complete_screen_navigation` restores the INCOMING screen before
  `switch_screen` unmounts the outgoing one, and this is where the incoming
  view claims the runtime (`ensure_console_runtime(app, view=self)` →
  `attach_view`) in time for the outgoing screen's later `detach_view` to
  find a different claimant and do nothing. Measured (M3/M3b below): the
  claim is made by ANY read of a runtime-backed handle, not by
  `_ensure_console_chat_store` specifically — remove the contact entirely
  and the continuity test goes red.
- **The composer→store draft flush stays** and is now the one place the
  view's uncommitted draft is written back into the object that outlives
  it.

### Bonus: five losses the snapshot round trip was causing

`ConsoleChatStore.restore_state` clears far more than it restores, and the
payload had no slot for any of it. Not calling it stops these, for free:

- the message **tree** (`_nodes_by_session`, `_children_by_parent`) —
  `_ingest_linear_messages` re-parented the flat list as a single chain, so
  every sibling/variant branch (swipe history) was collapsed on every tab
  switch;
- the local **active-leaf cursor** (`_active_leaf_by_session`), reset to
  "last row of the linear path";
- the **`/rewind` context summary** (`_context_summary_by_session`), reset
  to `(None, None)`;
- per-session **speech preferences** and the **one-shot prefill**, neither
  of which `_console_session_to_state` carried;
- the session's **`rag_scope_holder`**, replaced by a fresh empty holder.

### The pending-attachment stash: kept, with one new line

The stash lives on the APP object, never in the snapshot (bytes are
forbidden there), so it was never a second source of truth for message
state. It is kept — it is the only thing that knows an H3 image edit
completed while Console was away — but
`_adopt_console_pending_attachments` now calls
`store.clear_pending_attachments(session_id)` before re-staging. Without
it, a surviving store plus a re-staging adopt would **double** every staged
attachment on every navigation, up to the cap.

---

## 3. The four-way agreement test, and how it is made hard to fool

`test_transcript_payload_db_and_active_leaf_all_agree` appends through the
runtime (the headless wake turn above) and then asserts the four surfaces
P1 found disagreeing:

1. **Transcript** — four rows in the returning screen's store.
2. **Persisted rows** — `[m.persisted_message_id for m in transcript]`
   equals the DB's row ids **in order**. Matching on the DURABLE id, not on
   content, is what stops a look-alike rebuild passing: the first version of
   this assertion compared `m.id` and failed even after the fix, because the
   store mints its own in-memory ids — which is exactly the class of
   coincidence the test exists to reject.
3. **No fork** — every DB row's `parent_message_id` is checked against its
   predecessor's id, one by one, and the first row must be a root. P1's
   failure was a FORK, not a missing row, and a set-comparison would have
   missed it.
4. **Active leaf** — `conversations.active_leaf_message_id` equals the last
   row of that chain.
5. **Provider payload** — every USER/ASSISTANT row the user can see is in
   the next send's payload, and the payload's ordering of them matches the
   transcript's (index positions must be sorted).
6. **The other direction** — the machine-origin SYSTEM notice must **not**
   be in the payload. Console SYSTEM rows are UI chrome
   (`_provider_message_payloads` keeps only USER/ASSISTANT), so "agreement"
   cannot be bought by narrating everything to the model. This half was
   added after the first run: the test initially demanded the notice be in
   the payload, and the code was right and the test was wrong.
7. **The next append extends the chain** — a follow-up send is made, and
   its persisted row must parent to the previous leaf, with the leaf then
   moving to it. This is the exact assertion P1 saw fail.

Also asserted: the row count is 4 (not "at least"), every visible row
carries a non-null `persisted_message_id`, and the P3b test separately
checks the rendered widget tree, not only the store.

---

## 4. The freeze-incident gate: a soak that asserts INTERACTIVITY

The mechanism changed here is the one hardened after the 2026-07-11 freeze:
re-mounting a torn-down screen left child pumps permanently stopped while
widgets still reported `mounted=True`, the compositor kept presenting a
stale frame, and every click was hit-tested into the dead tree. **That
freeze was exception-free**, so a soak asserting "nothing raised" would
have passed straight through it.

`test_rapid_route_switching_leaves_console_interactive` does ten real
route switches (chat ⇄ library ⇄ settings, through
`app.post_message(NavigateToScreen(...))` and the real confirm dialog) and
then asserts the app still RESPONDS:

- **(a) a real keypress reaches a live widget**: `pilot.press("z", "q")`
  after focusing the composer must change `composer.draft_text()` and end
  with the typed characters;
- **(b) a real click is hit-tested into a live tree**: focus is moved to
  the transcript surface, then `pilot.click("#console-native-composer")`
  must move focus back;
- **(c) the transcript repaints**: the pre-churn message must still be
  present in the rendered widget text, and a turn sent afterwards **through
  the screen's own Enter path** (not `submit_draft` — only the screen
  dispatch arms the transcript poll, which is the thing being tested) must
  appear in the rendered text;
- **(d) nothing accumulated**: one runtime, one controller, one store, and
  the screen's handle is the same object as the runtime's.

Result: **passes at the merge-base and passes after the change** — as a
regression guard should. Its teeth are shown by mutation M5 below (the
repaint assertion dies when the transcript sync is skipped), not by it
being red today.

---

## 5. Session identity across a navigation

`test_session_identity_survives_a_navigation_for_an_unsaved_chat` uses a
conversation with **no `persisted_conversation_id` at all** — the case no
DB re-read could ever recover, and the reason today's mount-claim can
deliver a staged wake into an unsaved chat. Two sessions are open, a draft
is typed into the composer with real keys, then Console is left and
re-entered through the real navigation API.

Asserted: the id list is identical and in order, the active session is
unchanged, the typed draft survives, the second session keeps its title —
and `store.sessions()[0] is first`, i.e. the returning Console gets **the
same session object**, not a rebuilt look-alike.

That last assertion is the one that was red at the merge-base:

```
E   AssertionError: the returning Console got a REBUILT session object for
    an unsaved conversation -- a staged wake holding the old reference
    would target a dead id
E   assert ConsoleChatSession(...id='62f188d8-…', …todo_store=<…0x1311f6f60>)
        is ConsoleChatSession(...id='62f188d8-…', …todo_store=<…0x1269146b0>)
```

The **id** was preserved by the old snapshot (`_console_session_to_state`
carried it), which is why staged wakes worked; the object was not, and
everything hanging off it — `SessionTodoStore`, `rag_scope_holder`,
`speech_preferences`, `pending_attachments`, `one_shot_prefill` — was
silently replaced. Now both survive.

---

## 6. Mutations run and killed

Every restore was Edit-based; `grep -rn "MUTATION-" tldw_chatbook/` is
clean and `git diff` shows only the intended changes.

| # | Mutation | Expected to die | Actually died |
|---|---|---|---|
| M1 | the whole change reverted (i.e. the untouched merge-base) | the three continuity reds | `..._is_in_the_transcript`, `..._all_agree`, `..._session_identity...` — 3 failed / 1 passed ✅ |
| M2 | `_adopt_console_pending_attachments` drops the new `store.clear_pending_attachments(session_id)` | a pending-attachment duplication test | **8** tests in `test_console_pending_attachment_stash.py` ✅ |
| M3 | `_restore_native_console_state` reads `self._console_chat_store` instead of calling `_ensure_console_chat_store()` | the superseded-screen claim test | **nothing — SURVIVED** (see below) |
| M3b | the restore touches the runtime not at all (non-constructing read through the app) | the claim ordering | `test_a_wake_that_ran_while_console_was_unmounted_is_in_the_transcript` ✅ |
| M4 | `ConsoleChatStore._persist_active_leaf` never writes | the four-way test's active-leaf leg | `test_transcript_payload_db_and_active_leaf_all_agree` ✅ |
| M5 | `_sync_native_console_transcript` never calls `transcript.refresh_messages()` | the soak's repaint assertion | **nothing — SURVIVED** (see below), then 2 after the fix ✅ |

**M3 survived, and the docstring was wrong.** The claim is not made by
`_ensure_console_chat_store` specifically: `_console_chat_store` is itself
a runtime-backed property, so ANY read of it calls
`ensure_console_runtime(app, view=self)` and claims. M3b — a
non-constructing read straight off `app.console_runtime` — isolates the
claim properly and dies. The docstring now states the measured version
(runtime CONTACT is what matters, not the spelling) instead of the
plausible one.

**M5 survived, and it caught a test passing for the wrong reason** — the
whole point of the discipline. `_rendered_text` was reading
`ConsoleTranscript._messages`, the widget's MODEL, which `set_messages`
assigns before a single row is built; so "the wake notice is RENDERED"
and "the transcript repaints" were both really "the data arrived". The
helper now walks `_row_widgets`, what `_reconcile_rows` actually mounts.
Re-run under the same mutation: **2 failed** (P3b's render assertion and
the soak's repaint assertion). Restored: 4 passed. Both a test bug and,
had it shipped, a soak with no teeth on the one property the
freeze incident is about.

---

## 7. Gate — baseline (merge-base `31b0ef6a1`) vs final

Runner: `.venv/bin/pytest <paths> -p no:randomly -q --no-header -rf`,
cwd = the worktree, `Tests/test_probe_import_provenance.py` in every gate
(the venv's editable install resolves `tldw_chatbook` to a FOREIGN
worktree and loses only by `sys.meta_path` ordering). Every count below
was READ off a summary line, never inferred.

| Gate | Baseline @ merge-base `31b0ef6a1` | Final @ branch | Delta |
|---|---|---|---|
| **`Tests/Chat/` + probe** | 14 failed, 5587 passed, 66 skipped (1134.74s) | **14 failed, 5587 passed, 66 skipped** (1012.66s) | **0** — the same fourteen names, in the same files |
| **Gate battery** — `test_console_viewless_hooks`, `test_console_runtime_lifetime`, `test_console_runtime_ownership`, `test_screen_residency`, `Tests/Agents/`, 13 fleet+wake files, `test_console_mcp_approval`, 9 session/workspace/browser files, the new continuity suite, probe | its 6 failures re-measured node-by-node at the merge-base: **6 failed, 0 passed** | **6 failed, 2023 passed** | **0** — all six reproduce with this branch absent |
| **Snapshot/round-trip battery** — `test_console_native_chat_flow`, `test_console_pending_attachment_stash`, `test_console_live_work_handoffs`, `test_console_rag_settings_modal`, `test_console_skill_install_confirm`, `test_console_composer_menu`, `test_console_scope_row`, 3× `Tests/ProductionApp/`, `test_screen_navigation`, `Tests/State/test_screen_state_store`, `test_application_state_ownership`, probe | **7 failed, 753 passed** (902.25s) | 25 failed / 735 passed BEFORE the test updates; after them, per-file: 19 passed (`-k` the round-trip family), 22 passed (pending-attachment stash), 3 passed (`test_chat_root_state_removal`) | 19 of the 25 were mine and are fixed; the other 6 are merge-base reds (see below) |
| **New continuity suite** (`Tests/UI/test_console_store_continuity.py`) | did not exist; on the untouched production tree: **3 failed, 1 passed** | **4 passed** | +3 |

**The six merge-base reds inside the gate battery**, each re-run at
`31b0ef6a1` with zero bytes of this branch and each failing there
(`6 failed` in 13.58s):
`test_console_session_settings::test_mounted_console_unmount_times_out_hung_refresh_and_repairs_on_resume`,
`test_console_workspace_controller` ×2,
`test_console_workspace_context_rail` ×3.

**The six merge-base reds inside the snapshot battery** (from that
battery's own merge-base run): `test_ctrl_k_opens_session_switcher_and_
activates_native_session`, `test_switcher_rename_choice_chains_to_rename_
modal`, `test_chat_composition_retirement` ×3 (all three
`AttributeError: 'ChatScreen' object has no attribute
'_ensure_console_video_store'`), and
`test_application_state_ownership::test_runtime_source_state_store_
references_are_confined_to_owner_modules`. A seventh, `test_console_
accepted_send_records_first_send_flag`, failed at the MERGE-BASE and
passed on the branch — flaky under load, in the branch's favour, and not
claimed as a fix.

**Honest gap:** the snapshot battery's full re-run after the test updates
did not complete — it sat at 66% for over an hour under five-way CPU
contention and was killed. Its per-file re-runs (above) are the evidence
that stands, and every file in it was already green WITH the production
change in place during the 25-failure run, so the only files whose state
changed afterwards are the three that were re-run individually.

---

## 8. Deliberately not done

- **`_attempt`'s `_shutdown_requested` gate is untouched**, byte for byte.
  A wake still does not FIRE while Console is detached; this task makes a
  wake that DID run visible on return. Firing is the wake-fires-headless
  slice.
- **The approval clock (Task 5) and launch wake (Task 6)** are untouched.
- **The now-dead serializers are kept.** `_serialize_console_message`,
  `_restore_console_message`, the three `_rehydrate_console_message_*`
  delegators and `_console_session_to_state`/`_console_session_from_state`
  have **no production caller left**. They are still exercised by ~45
  tests, which now describe a path the app never takes. Retiring them is
  wide and mechanical, and was deliberately kept out of the riskiest
  landing of the arc: **task-16520**. Each one's docstring now says so.
- **Runtime-identity scoping of Console history** is unchanged — see
  concerns.

## 9. Concerns

1. **~45 tests now exercise code with no production caller.** The
   per-session and per-message (de)serializers survive this landing; the
   tests that pinned them were retargeted at a documented test-local round
   trip rather than deleted, because their subject (legacy-payload
   tolerance, character-provenance narrowing) is still live code. Until
   **task-16520** retires both, that coverage describes a path the app
   never takes. Deliberate, and the alternative — a ~45-test deletion
   inside the riskiest landing of the arc — was worse.

2. **Runtime-identity scoping of Console history is now visibly
   vestigial.** `ScreenStateStore.restore` drops a snapshot whose
   `RuntimeIdentity` no longer matches (a local↔server switch), which used
   to mean "Console starts fresh after a server switch". It has not meant
   that since the lifetime landing: `ConsoleRuntime` has no identity check
   at all (`ensure_chat_store` returns the store it already holds), and
   nothing clears it on a switch — so the sessions survived the identity
   change anyway, snapshot or no snapshot. This landing does not change
   that behaviour by a byte; it just removes the code that looked like it
   was enforcing it. Whether Console history SHOULD reset on a runtime
   switch is an owner question, and it is now the only place that decision
   could live.

3. **Same-target navigation** still carries the lifetime landing's
   recorded consequence (the outgoing screen's streaming turn is no longer
   cancelled). Unchanged here, and still worth the owner eyeball that
   report asked for.

4. **Machine contention made wall-clock numbers unreliable** and produced
   at least two failures at BOTH the merge-base and the branch
   (`test_ctrl_k_opens_session_switcher_and_activates_native_session`,
   `test_switcher_rename_choice_chains_to_rename_modal` — both drive a
   modal behind `pilot.pause(0.2)`). Four to six foreign `pytest`
   processes ran on this machine throughout; a 15-minute suite took 15
   minutes at the merge-base and 15 minutes on the branch, but a
   single-file run varied by 3×. Counts were read, never inferred.

5. **`Tests/UI/` was NOT run in full.** Under the contention above it was
   not affordable. What WAS run: every UI file the change's blast radius
   names (the whole snapshot/round-trip surface, the runtime/lifetime/
   viewless/residency trio, all thirteen fleet+wake files, the MCP
   approval file, nine session/workspace/browser files, both ProductionApp
   route files, and the new continuity suite) plus `Tests/Chat/` in full.
   Saying so plainly rather than implying coverage that was not obtained.

6. **The soak is a pytest test, not `run_workbench_soak.py`.** It copies
   that script's shape (route churn, then probe the app) but asserts
   interactivity instead of writing responsiveness artifacts, so it can
   live in this task's gate and fail a PR. The artifact-writing soak is
   untouched.
