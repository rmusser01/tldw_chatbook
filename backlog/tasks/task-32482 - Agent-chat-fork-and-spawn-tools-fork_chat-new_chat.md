---
id: TASK-32482
title: Agent chat fork and spawn tools (fork_chat / new_chat)
status: In Progress
assignee: []
created_date: '2026-09-12 01:25'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Let a Console agent prepare parallel-workstream chats for the user: fork_chat copies the current conversation's active path into a new chat (first-ever writer of the parent_conversation_id / forked_from_message_id lineage columns); new_chat creates a fresh chat. Both accept title / opening_prompt / instructions, require an explicit user approval per call by default (Allow / Allow-for-session / Deny, fail-closed, session-scoped remember), land the opening prompt as a composer draft the user sends, and open the chat in the background with a toast. Spec: Docs/superpowers/specs/2026-09-11-agent-chat-fork-spawn-design.md; ADR: backlog/decisions/150-agent-chat-fork-and-spawn.md; Plan: Docs/superpowers/plans/2026-09-11-agent-chat-fork-spawn.md; follow-ups: sub-agents (task-32480) and preset/provider args (post ADR-147 integration PR).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Agent can fork the current chat into a new one via a confirmed tool call (verbatim active-path copy with lineage columns set)
- [x] #2 Agent can create a fresh chat with title and instructions via a confirmed tool call
- [x] #3 Every creation requires explicit user approval by default with per-tool session-scoped remember and fail-closed behavior when no UI or on cancel or timeout
- [x] #4 Opening prompt lands as a composer draft the user sends and survives app restart via conversation metadata
- [x] #5 New chats appear in the same workspace without switching the active session and a toast announces them
- [x] #6 Forks of character-bound chats refuse agent instructions and ephemeral or empty sources return clear tool errors
- [x] #7 Tool schemas are advertised to primary agents only while sub-agent runs are unchanged
- [x] #8 Workspace listing shows created chats under whichever mechanism governs it
- [x] #9 Tests cover fork helper semantics (remap, lineage, atomicity, uncapped copy) and confirm rounds (allow, deny, remember, fail-closed, park)
- [x] #10 User docs updated for both tools
<!-- AC:END -->

## Implementation Notes

### Live verification — PASSED (2026-09-12, post-dev-rebase)

Full walkthrough on the rebased branch, real terminal (tmux), real local
llama.cpp endpoint (Gemma-4-26B), isolated scratch profile:

- Agent called fork_chat itself (schema advertised, in-loop dispatch).
- Confirm card rendered with full enrichment: proposed title, "Copies 2
  messages from '<source title>'", requesting run id, full opening-prompt body.
- Allow executed the creation in the BACKGROUND: source session stayed
  active; "UAT fork" appeared as a tab and in the conversations/workspace
  listing; the model received the success JSON and reported the fork.
- Verbatim copy landed (lineage columns set, console_agent_handoff draft
  persisted); the draft appeared in the fork's composer, cursor in place.
- Divergence: sending in the fork left the source's message count unchanged.
- Restart: the fork chat and its copied message persisted.

Live defects found and fixed during the UAT (each with regression tests):
1. Plan flags rode a preview call instead of the real run_reply request plan.
2. LoopDeps chat-create population was lost in the rebase (dispatch fell to
   the allow-list: "Tool not permitted").
3. The copy failed on the in-flight EMPTY assistant placeholder row the
   submit path echoes before the first token (add_message refuses
   contentless re-inserts mid-turn) -- copy now skips contentless scaffold
   rows and the fork leaf falls back to the last copied message.
4. A resolved card lingered and re-appeared on session switch: the unified
   decision projection's early returns bypassed our standalone card
   registry (teardown now pushes the head/None directly; session
   reprojection runs our remount before delegating).

Environment notes for future runs: the first send on a fresh scratch profile
intercepts on the project-instruction folder modal (disable with `d`), and
`LLAMA_CPP_API_KEY` must be set even for a keyless local endpoint. Sends can
stall in this headless-ish environment's personal-context bootstrap (dev's
own CONSOLE_PRE_PROVIDER_SETUP_BUDGET comment documents the traced Keychain
case); retrying the turn proceeds.

### Rebase onto dev (2026-09-12)

Rebased onto origin/dev after the provider-routing PR (#2635) and six later PRs
merged. Notable integrations: dev gained native fork-lineage support in
`ChatPersistenceService.create_conversation` (our Task-1 passthrough is
superseded — same kwarg names, executor unchanged); `restore_persisted_session`
gained `activate` natively (our Task-8 deferred handoff-clear grafts onto it);
the controller's confirm machinery moved to an interrupt-host (our chat-create
rounds stay standalone, swept via a direct `_revoke_chat_create_rounds` hook in
`revoke_approval_rounds_for_run`); ChatTaskCards moved to a routing-table sync
(our card rides `_routes`). Dev-baseline fixes carried in our branch: the
RUNTIME_TOOL_NAMES exhaustive sets, the LoopDeps positional pin test
(`replace_disclosed_names` slot), and the view-hook ownership inventory
(additions only). Known pre-existing dev failures NOT caused by this branch:
six `test_console_runtime_ownership` fleet-wake tests (`delivering_session_id`
vs `delivering_session_ids` rename, tests stale on origin/dev);
`test_confirm_payload_carries_timeout_and_request_id` flakes under combined-run
load, passes isolated.

### Live verification — attempted 2026-09-12, BLOCKED at the base send path (remains OPEN)

Attempted per the plan's Step 4 checklist, in an isolated worktree launch (scratch
TLDW_CONFIG_PATH + scratch data_dir copied from the real config for provider
credentials; import provenance verified against the feature worktree's own venv;
no real-profile mutations; all scratch material with credentials deleted after).

Findings — every blocker is in the BASE send path, not in fork_chat/new_chat:

1. First send on a fresh scratch profile is intercepted by the project-instruction
   folder-chooser modal ("No eligible folders / no_eligible_binding"); while it is
   up the run state parks at "Validating provider." behind the modal. Dismissing
   with `d` (Disable) unblocks that send.
2. The composer-side provider gate blocks sends with the generic
   "Send blocked — finish provider setup" copy when `LLAMA_CPP_API_KEY` (the env
   var named by `[api_settings.llama_cpp].api_key_env_var`) is unset, even for a
   localhost keyless llama.cpp endpoint. Setting a dummy value clears the gate.
3. After both, a send still parks at the "Preparing..." turn-acceptance state with
   no HTTP connection to the model server and no run row created. The identical
   behavior reproduces on the BASE commit 91caf08be2 (zero of this task's
   changes) under the same scratch profile — the two pre-existing failing tests
   at base (`test_confirm_callback_absent/present_from_bridge_when_no_ui_sink_wired`,
   `visible_copy='binding_unavailable'`) are the same send path.

Conclusion: the live walkthrough cannot proceed until the base Console send path
works in a fresh-profile environment (the in-flight provider-routing PR reworks
exactly this region; a concurrent smoke session on that tree shows sends working
there). Re-run the plan's Step 4 checklist on the merged result — or on this
branch with the routing WIP applied — before flipping this task to Done. The
automated surface remains green (568 passed / 2 documented pre-existing base
failures), including real-SQLite bridge→controller→store integration and a real
Textual-pilot click round-trip through the confirmation card.



**Approach.** Two runtime tools (`runtime:fork_chat` / `runtime:new_chat`, ADR-150)
advertised to primary agents only, executed as confirmed tool calls: a
worker-thread blocking confirm round (Allow / Allow for this session / Deny)
arms a `ChatCreateConfirmCard`; an allow runs the executor (fork = verbatim
active-path copy with lineage, new_chat = fresh conversation in the session's
workspace scope); the UI-thread completion lands the created chat as a
non-activated background session with the opening prompt as a one-shot
composer draft. Verification: targeted automated suites green (see below);
live verification remains open (see the caveat at the end).

**Decisions.**

- **No DB migration.** The pre-existing `parent_conversation_id` /
  `forked_from_message_id` conversation columns (both FK-enforced) get their
  first-ever writers here; lineage rides a passthrough in
  `Chat/chat_persistence_service.create_conversation`. No schema file was
  touched by any feature commit.
- **Session-scoped remember.** "Allow for this session" grants per tool —
  run-local memo in the bridge closures plus the controller's
  `_chat_create_session_grants` for cross-run remembers inside one Console
  session. Nothing is remembered past the session (no "Always allow").
- **Denial guard after two.** Two denials of the same tool in one run
  terminal-disable that tool for the rest of the run; the model is told once.
- **One-shot draft rehydration.** The `console_agent_handoff` conversation
  metadata key (`draft` / `created_via` / `source_run_id`) rehydrates the
  opening prompt into the composer on restore; the key is consumed when
  that session FIRST becomes active (final-review semantics — see the
  fix-wave paragraph), so an unopened draft survives an app restart and a
  chat opened once is never re-filled; the completion path re-applies the
  same draft idempotently.
- **Worktree isolation from routing WIP.** Implemented on the isolated
  `.worktrees/agent-chat-fork-spawn` worktree, clear of the uncommitted
  provider-routing WIP (TASK-32477); the only shared-file signature changes
  (`create_conversation` / `restore_persisted_session`) are in files that PR
  does not touch.
- **Deferred follow-ups.** Sub-agent access to the tools (TASK-32480) and
  preset/provider argument integration (post ADR-147 routing PR).

**Files touched (grouped by task).**

- Task 3 (names + schemas): `tldw_chatbook/Agents/agent_models.py`,
  `tldw_chatbook/Agents/tool_catalog.py`,
  `Tests/Agents/test_agent_chat_create_tools.py`.
- Task 4 (service seam): `tldw_chatbook/Agents/agent_runtime.py`,
  `tldw_chatbook/Agents/agent_service.py`,
  `Tests/Agents/test_agent_runtime.py`,
  `Tests/Agents/test_agent_models.py`,
  `Tests/Agents/test_install_skill_runtime_tool.py` (RUNTIME_TOOL_NAMES
  exhaustive-set updates).
- Tasks 2/5 fork primitive + lineage passthrough:
  `tldw_chatbook/Chat/chat_conversation_service.py`
  (`copy_conversation_active_path`),
  `tldw_chatbook/Chat/chat_persistence_service.py`,
  `Tests/Chat/test_chat_conversation_service.py`,
  `Tests/Chat/test_chat_persistence_service.py`.
- Tasks 5/6 (confirm rounds + card): 
  `tldw_chatbook/Chat/console_chat_controller.py` (rounds, session grants,
  revoke-on-cancel, parking),
  `tldw_chatbook/Widgets/Chat_Widgets/chat_create_confirm_card.py`,
  `tldw_chatbook/Widgets/Chat_Widgets/chat_task_cards.py`,
  `tldw_chatbook/UI/Screens/chat_screen_state.py`,
  `tldw_chatbook/UI/Console_Modules/skill.py`,
  `tldw_chatbook/Chat/console_runtime.py` (`set_pending_chat_create` slot),
  `tldw_chatbook/UI/Screens/chat_screen.py` (card mount/decision wiring),
  `Tests/Chat/test_console_chat_create_confirm.py`,
  `Tests/Chat/test_chat_create_confirm_card.py`.
- Task 7 (executor + completion):
  `tldw_chatbook/Chat/console_agent_bridge.py` (closures, denial guard,
  outcome-contract mapping),
  `tldw_chatbook/Chat/console_chat_controller.py`
  (`execute_agent_chat_create` + `complete_agent_chat_create` slot),
  `tldw_chatbook/UI/Screens/chat_screen.py` (`_complete_agent_chat_create`),
  `Tests/Chat/test_console_chat_create_integration.py`.
- Task 8 (background session + rehydration):
  `tldw_chatbook/Chat/console_chat_store.py` (non-activating restore,
  one-shot handoff draft), `tldw_chatbook/UI/Screens/chat_screen.py`
  (final wiring), `Tests/Chat/test_console_chat_store.py`.
- Task 9 (docs + closure): `Docs/User_Guide/console/agent-runs-and-tools.md`
  (new "Chat creation tools" section + stamp),
  `backlog/docs/lessons-console-wiring.md` (new),
  this file. `Docs/User_Guide/console.md` was checked and deliberately not
  modified — it does not enumerate agent tools, so no workstream blurb was
  added there.

**Final-review fix wave (post-61bf4a8a3f).** The whole-branch review's
three Important findings, fixed in one wave: (1) **payload enrichment** —
`request_chat_create_confirm` now enriches the card payload before arming
(`_enrich_chat_create_confirm_payload` in `console_chat_controller.py`):
`fork_source_title`/`fork_message_count` (count = ACTIVE-PATH length, the
exact ancestry `copy_conversation_active_path` copies), a default title
("Fork of <source>" / "New Chat") when the agent omitted one, and run-id
attribution; the card renders the fork line only when those keys are
present and shows "Requested by agent run <id>"; all enrichment is
best-effort and degrades without blocking the round. (2) **Activation-time
handoff consumption** — `restore_persisted_session` no longer clears the
persisted key at restore; `console_chat_store` records a pending clear and
performs the best-effort optimistic-locked clear when that session FIRST
becomes active (`_consume_pending_agent_handoff_clear` off
`_activate_session`; an activating restore re-consumes directly), so a
never-opened draft survives restarts (AC #4) and a once-opened chat is
never re-filled. (3) **Orphan guard** — `execute_agent_chat_create`
returns `empty_history` BEFORE `create_conversation` when the fork source
tree has no message nodes, and a post-create failure best-effort
soft-deletes the created row (`_discard_chat_create_orphan`) so no
unexplained row surfaces in the workspace listing. Also: revoke-docstring
registry count fixed (three), ADR-150's primitive name updated to the
shipped `copy_conversation_active_path`, and the user guide's fork-summary,
restart-durability, and caps wording aligned (titles truncate at 120; only
`opening_prompt`/`instructions` error when oversize).

**AC caveat / open item — live verification.** ACs #5 and #8 are left
unchecked, and #4 carries a caveat, because their UI halves are code-reviewed
and unit/bridge-tested but not UI-tested in a running app: the toast copy and
the workspace listing's rendered appearance (#5, #8) are only reachable live,
and #4's draft-in-composer is covered at store/bridge level
(`test_restore_rehydrates_handoff_draft_once`,
`test_execute_new_chat_creates_conversation_and_completion`,
`test_restore_persisted_session_activate_false_keeps_current`) rather than
visually. The live checklist in
`Docs/superpowers/plans/2026-09-11-agent-chat-fork-spawn.md` step 4 (propose
two workstreams, allow, verify card bodies, toast, workspace listing, drafts,
send-in-fork isolation, restart persistence of both chats and unopened
drafts) remains OPEN for the human per
`backlog/docs/lessons-live-verification.md`; task status is therefore left
In Progress, not Done. Automated verification (task 9): the 10-file targeted
list is green except the two known pre-existing failures in
`Tests/Chat/test_console_skill_script_confirm.py`
(`test_confirm_callback_absent_from_bridge_when_no_ui_sink_wired`,
`test_confirm_callback_present_when_no_ui_sink_wired`), present at the branch
base before this feature's commits.
