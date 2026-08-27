# Roleplay Resume Prior Character Chats Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: use
> `superpowers:subagent-driven-development` for same-session execution or
> `superpowers:executing-plans` for a separate execution session. Apply
> `superpowers:test-driven-development` to every behavior change and
> `superpowers:verification-before-completion` before claiming completion.

**Goal:** Let a user open a saved local character conversation from Roleplay's
read-only preview and resume that authoritative conversation as the final active
Console chat, without converting it into draft context or RAG scope.

**Architecture:** Roleplay remains a bounded discovery/preview surface and posts
one validated conversation ID through normal screen navigation. The freshly
mounted Console settles its older pending intents in the established order, then
uses the existing workspace opener and canonical hydration path. Versioned
conversation metadata supplies the historical character-name snapshot required to
reconstruct prompt behavior; a focused runtime rollback seam makes pre-commit
hydration and presentation atomic without touching durable data.

**Tech stack:** Python 3.11+, Textual 8.x, asyncio, dataclasses, SQLite-backed Chat
services, pytest/pytest-asyncio, modular TCSS plus generated consolidated TCSS.

**Spec:**
`Docs/superpowers/specs/2026-08-26-roleplay-resume-prior-character-chat-design.md`

**Backlog task:**
`backlog/tasks/task-22988 - Resume-prior-character-chats-from-Roleplay.md`

## ADR check

ADR required: yes

ADR path: `backlog/decisions/046-roleplay-chat-display-identity-and-template-provenance.md`

Reason: resuming must define a durable metadata-version contract and which
historical character identity is authoritative for prompt reconstruction. Amend the
existing provenance ADR instead of creating a duplicate. ADR-026 continues to own
Console conversation entry, and ADR-033 continues to own application session state.

The ADR amendment is the first change in Task 1, before implementation code.

## Global constraints

- Work only in the clean worktree
  `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/roleplay-resume-chat`
  on branch `codex/roleplay-resume-chat`.
- Preserve unrelated user changes and do not modify the original worktree.
- Use `../../.venv/bin/python` from this worktree.
- Run targeted tests only. The repository requires separate user approval before a
  full test sweep.
- Keep **Resume chat**, **Send transcript to Console draft**, and **Open in
  Library** as separate operations. Resume carries no transcript, title, character
  record, handoff body, or RAG scope.
- Do not add a new persistence table, schema migration, result object, navigation
  service, Roleplay-side hydrator, or second Console resume implementation.
- `True` means resume committed, `False` means the durable conversation is missing,
  and `None` means a transient failure was already reported. Cancellation rolls
  back and propagates as `asyncio.CancelledError`.
- A live matching Console session is never deleted by rollback. A new restored
  runtime session is committed only after its full Console presentation succeeds.

## File map

### Governance and task state

- Modify
  `backlog/decisions/046-roleplay-chat-display-identity-and-template-provenance.md`
  to record metadata v2 and historical character-name authority.
- Modify
  `backlog/tasks/task-22988 - Resume-prior-character-chats-from-Roleplay.md`
  to add this implementation plan, then completion evidence and notes only after
  verification.
- Keep the approved spec synchronized if implementation review exposes a genuine
  contract correction.

### Roleplay context and persistence

- Modify `tldw_chatbook/Chat/console_roleplay_metadata.py` for backwards-readable
  v1 and v2 metadata and v2 writes.
- Modify `tldw_chatbook/Chat/chat_persistence_service.py` and
  `tldw_chatbook/Chat/console_chat_store.py` to persist the character-name snapshot
  at every existing roleplay-context write.
- Modify `tldw_chatbook/Chat/console_conversation_hydration.py` to restore identity
  solely from saved conversation state and support delayed activation.

### Console resume and navigation

- Modify `tldw_chatbook/Chat/console_chat_store.py` to share exact runtime-session
  cleanup and expose guarded rollback for a newly restored session.
- Modify `tldw_chatbook/UI/Console_Modules/workspace.py` to make the canonical opener
  tri-state, active-match-first, atomic, and snapshot-driven.
- Modify `tldw_chatbook/UI/Console_Modules/wiring.py` and
  `tldw_chatbook/UI/Screens/chat_screen.py` to remove the current-card name lookup
  seam and add ordered pre-mount resume navigation.
- Modify `tldw_chatbook/Constants.py` for the dedicated ID-only navigation key.

### Roleplay UI

- Modify
  `tldw_chatbook/UI/Persona_Modules/personas_conversations_controller.py` for
  distinct preview failure and Resume navigation state.
- Modify `tldw_chatbook/UI/Screens/personas_screen.py` for the three-row action
  hierarchy, handlers, focus order, preview gate, and compact layout.
- Modify
  `tldw_chatbook/Widgets/Persona_Widgets/personas_inspector_pane.py` so card-level
  actions remain hidden throughout preview recomputations and restore on Back.
- Modify
  `tldw_chatbook/Widgets/Persona_Widgets/personas_conversation_transcript_widget.py`
  for the persistent preview note and distinct error state.
- Modify the appropriate source rule in
  `tldw_chatbook/css/components/_workbench.tcss`, then regenerate
  `tldw_chatbook/css/widget_defaults_scoped.tcss` and
  `tldw_chatbook/css/widget_defaults_self.tcss` with the repository builder.

### Targeted tests

- Modify `Tests/Chat/test_console_roleplay_metadata.py`.
- Modify `Tests/Chat/test_chat_persistence_service.py`.
- Modify `Tests/Chat/test_console_chat_store.py`.
- Modify `Tests/Chat/test_console_conversation_hydration.py`.
- Modify `Tests/UI/test_console_resume_active_path.py`.
- Modify `Tests/UI/test_console_workspace_controller.py`.
- Modify `Tests/UI/test_console_native_chat_flow.py` only where current-card lookup
  expectations become obsolete.
- Add `Tests/UI/test_console_roleplay_resume_navigation.py` for the pre-mount and
  ordered-startup contract.
- Modify `Tests/UI/test_personas_workbench.py` for Roleplay behavior, keyboard flow,
  production CSS, and 80x24 containment.

---

## Task 1: Amend ADR-046 and evolve roleplay metadata to v2

**Files:**

- Modify:
  `backlog/decisions/046-roleplay-chat-display-identity-and-template-provenance.md`
- Modify: `tldw_chatbook/Chat/console_roleplay_metadata.py`
- Modify: `tldw_chatbook/Chat/chat_persistence_service.py`
- Modify: `tldw_chatbook/Chat/console_chat_store.py`
- Test: `Tests/Chat/test_console_roleplay_metadata.py`
- Test: `Tests/Chat/test_chat_persistence_service.py`
- Test: `Tests/Chat/test_console_chat_store.py`

### Step 1: Record the authority decision before implementation

Add a dated amendment to ADR-046 stating:

- `console_roleplay_context` v2 adds optional `character_name_snapshot`.
- New writes use v2; readers accept v1 and v2; versions greater than v2 remain
  fail-closed and block merge writes.
- The snapshot is the character name that owned the resolved prompt/template
  projection when the conversation was saved.
- A v1 conversation has no historical-name authority. Resume must not fetch or
  backfill the current character card name.
- Saved resolved `system_prompt` remains authoritative when provenance or the
  historical name is absent.
- The data remains in the existing merge-safe metadata object; no schema migration
  is introduced.

Link TASK-22988 and the approved design from the ADR.

### Step 2: Write failing metadata contract tests

In `Tests/Chat/test_console_roleplay_metadata.py`, update the former v2-future cases
to v3 and add tests proving:

- v1 reads the existing user/template fields and returns
  `character_name_snapshot is None`.
- v2 reads all three owned fields exactly.
- v2 merge preserves unrelated siblings and writes only non-empty owned fields.
- clearing all owned fields removes the owned object.
- v3 reads as empty and a v3 merge raises `RoleplayContextVersionError`.
- an invalid v2 snapshot degrades only the snapshot to `None`; otherwise-valid user
  and template provenance remains available.
- merge refuses non-text, blank, multiline/control-containing, sanitizer-changing,
  or over-180-character snapshots instead of persisting a different name.

Use an expected v2 shape such as:

```python
{
    "console_roleplay_context": {
        "version": 2,
        "user_name_override": "Captain Rowan",
        "character_system_template": "Speak with {{user}} as {{char}}.",
        "character_name_snapshot": "Alraune",
    }
}
```

Run the red test:

```bash
../../.venv/bin/python -m pytest Tests/Chat/test_console_roleplay_metadata.py -v
```

Expected: new v2 assertions fail because production still treats v2 as future and
the dataclass has no snapshot field.

### Step 3: Implement the smallest v1/v2 parser and v2 writer

In `console_roleplay_metadata.py`:

- Set `ROLEPLAY_CONTEXT_VERSION = 2`.
- Add `character_name_snapshot: str | None = None` to
  `ConsoleRoleplayContext`.
- Accept only exact integer versions `1` and `2`; keep booleans and future versions
  untrusted.
- Parse v1 without inventing a snapshot.
- Validate the v2 snapshot with one private helper. Require exact `str`/`None`,
  strip outer whitespace, require non-blank, cap at
  `CHARACTER_SPEAKER_LABEL_MAX_CHARACTERS`, and require
  `sanitize_character_display_label(value, max_characters=180) == value` so
  persistence never silently changes prompt identity.
- On read, catch only snapshot validation failure and set that field to `None` while
  retaining otherwise-valid owned fields. On merge, let validation raise
  `ValueError` so unsafe new state cannot be written.
- Write v2 for all non-empty contexts and preserve outer sibling keys.

The core data contract should remain a single dataclass:

```python
@dataclass(frozen=True, slots=True)
class ConsoleRoleplayContext:
    user_name_override: str | None = None
    character_system_template: str | None = None
    character_name_snapshot: str | None = None
```

Do not create a per-version class hierarchy.

### Step 4: Write failing persistence propagation tests

In `Tests/Chat/test_chat_persistence_service.py`, extend the optimistic retry and
sibling-preservation tests to pass `character_name_snapshot="Alraune"` and assert
the stored v2 payload retains it across a conflict retry.

In `Tests/Chat/test_console_chat_store.py`, extend the existing roleplay context
write tests to assert every construction site passes the live session's
`character_name` snapshot:

- the projection snapshot write;
- `_persist_roleplay_context` for an already-persisted session;
- first conversation creation/persistence metadata.

Also assert a generic/non-character session does not invent a snapshot.

Run the red tests:

```bash
../../.venv/bin/python -m pytest \
  Tests/Chat/test_chat_persistence_service.py \
  Tests/Chat/test_console_chat_store.py \
  -k 'roleplay_context or roleplay_projection or first_persist' -v
```

### Step 5: Thread the snapshot through existing persistence seams

Add `character_name_snapshot: str | None` to:

```python
ChatPersistenceService.update_conversation_roleplay_context
ConsoleConversationStore.update_conversation_roleplay_context
```

Update every concrete implementation, protocol, fake with an explicit signature,
and each `ConsoleRoleplayContext` construction. Source the value from
`session.character_name` only when `session.assistant_kind == "character"`; pass
`None` otherwise. Keep the current optimistic retry and sibling merge unchanged.

Use `rg` to prove no call site was missed:

```bash
rg -n "update_conversation_roleplay_context\(|ConsoleRoleplayContext\(" \
  tldw_chatbook Tests
```

### Step 6: Verify Task 1 and commit

```bash
../../.venv/bin/python -m pytest \
  Tests/Chat/test_console_roleplay_metadata.py \
  Tests/Chat/test_chat_persistence_service.py \
  Tests/Chat/test_console_chat_store.py \
  -k 'roleplay or first_persist' -v
git diff --check
git add \
  backlog/decisions/046-roleplay-chat-display-identity-and-template-provenance.md \
  tldw_chatbook/Chat/console_roleplay_metadata.py \
  tldw_chatbook/Chat/chat_persistence_service.py \
  tldw_chatbook/Chat/console_chat_store.py \
  Tests/Chat/test_console_roleplay_metadata.py \
  Tests/Chat/test_chat_persistence_service.py \
  Tests/Chat/test_console_chat_store.py
git commit -m "feat(console): preserve historical roleplay identity"
```

---

## Task 2: Restore historical identity through canonical hydration

**Files:**

- Modify: `tldw_chatbook/Chat/console_conversation_hydration.py`
- Modify: `tldw_chatbook/UI/Console_Modules/workspace.py`
- Modify: `tldw_chatbook/UI/Console_Modules/wiring.py`
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py`
- Test: `Tests/Chat/test_console_conversation_hydration.py`
- Test: `Tests/UI/test_console_resume_active_path.py`
- Test: `Tests/UI/test_console_native_chat_flow.py`
- Test: `Tests/UI/test_console_workspace_controller.py`

### Step 1: Write failing hydration authority tests

Add tests showing that canonical hydration:

- restores a v2 local character session with
  `session.character_name == "Alraune"` and
  `session.settings.character_label == "Alraune"`;
- expands future trusted template projections with the saved name, not a renamed
  current card;
- leaves both fields empty for a v1 conversation with no snapshot, while retaining
  its saved resolved system prompt and v1 user/template fields;
- restores generic sessions without character identity;
- produces equivalent state from the launch-wake caller and the screen caller.

Update `Tests/UI/test_console_resume_active_path.py`'s guarded durable roleplay test
so v2 is valid and v3 is future. Remove assertions that expect a current character
card lookup to rename the restored session.

Run the red tests:

```bash
../../.venv/bin/python -m pytest \
  Tests/Chat/test_console_conversation_hydration.py \
  Tests/UI/test_console_resume_active_path.py \
  -k 'roleplay or character_name or hydration' -v
```

### Step 2: Move saved-name projection into hydration

In `hydrate_console_session`:

1. Parse `conversation["metadata"]` before `restore_persisted_session`.
2. Use `roleplay_context.character_name_snapshot` only when the trusted restored
   `assistant_kind == "character"`.
3. Replace resumed settings' `character_label` with that snapshot or `""`.
4. Pass `character_name=character_name` to `restore_persisted_session`.
5. Keep `session.user_display_name_override` and
   `session.character_system_template` sourced from the same parsed context.

Add `activate: bool = True` to `hydrate_console_session`; keep launch wake behavior
unchanged through the default and make the workspace path pass `activate=False` in
Task 3.

### Step 3: Remove the mutable-card lookup from the resume boundary

Delete the `resolve_resumed_character_name` dependency from:

- `ConsoleWorkspaceController.__init__` and its property shim;
- `UI/Console_Modules/wiring.py`;
- direct-construction fixtures in `Tests/UI/test_console_workspace_controller.py`.

Remove the workspace block that overwrites `session.character_name` from the current
card. Remove `ChatScreen._resolve_resumed_character_name` only after `rg` proves no
other behavior uses it. Current-card lookup may still supply avatar/display assets
through existing character ID paths; it must not alter saved prompt behavior.

### Step 4: Verify Task 2 and commit

```bash
../../.venv/bin/python -m pytest \
  Tests/Chat/test_console_conversation_hydration.py \
  Tests/UI/test_console_resume_active_path.py \
  Tests/UI/test_console_native_chat_flow.py \
  Tests/UI/test_console_workspace_controller.py \
  -k 'resume or roleplay or character_name or hydration' -v
rg -n "resolve_resumed_character_name" tldw_chatbook Tests
git diff --check
git add \
  tldw_chatbook/Chat/console_conversation_hydration.py \
  tldw_chatbook/UI/Console_Modules/workspace.py \
  tldw_chatbook/UI/Console_Modules/wiring.py \
  tldw_chatbook/UI/Screens/chat_screen.py \
  Tests/Chat/test_console_conversation_hydration.py \
  Tests/UI/test_console_resume_active_path.py \
  Tests/UI/test_console_native_chat_flow.py \
  Tests/UI/test_console_workspace_controller.py
git commit -m "refactor(console): hydrate saved character identity"
```

Expected `rg`: no matches. If a non-resume consumer exists, retain that consumer but
remove it from the resume call chain and document the exception in the task notes.

---

## Task 3: Make restored-session activation atomic

**Files:**

- Modify: `tldw_chatbook/Chat/console_chat_store.py`
- Modify: `tldw_chatbook/Chat/console_conversation_hydration.py`
- Modify: `tldw_chatbook/UI/Console_Modules/workspace.py`
- Test: `Tests/Chat/test_console_chat_store.py`
- Test: `Tests/Chat/test_console_conversation_hydration.py`
- Test: `Tests/UI/test_console_workspace_controller.py`

### Step 1: Write failing exact-rollback tests

In `Tests/Chat/test_console_chat_store.py`, construct a restored branched session
with off-path nodes and per-message/session auxiliary state. Assert a new method:

```python
store.rollback_restored_session(
    session.id,
    expected_session=session,
    prior_active_session_id=prior.id,
)
```

- removes the exact restored session, all owned messages (including off-path nodes),
  indices, stream/persistence/variant/retry/speech/projection/emote state,
  preparation/recovery state, workspace projection, and library holder;
- restores the prior active session when it still exists;
- succeeds without mutating the durable conversation;
- refuses when the ID now points to a different object;
- never removes a pre-existing live matching session.

Add regression coverage proving `close_session` still chooses the same neighboring
session and clears the same runtime state after cleanup extraction.

Run the red test:

```bash
../../.venv/bin/python -m pytest Tests/Chat/test_console_chat_store.py \
  -k 'rollback_restored or close_session' -v
```

### Step 2: Extract one exact runtime purge helper

Extract the state-deletion block currently embedded in `close_session` into one
private store helper such as `_purge_session_runtime_state(session_id)`. Reuse it
from:

- `close_session` after its user-facing recovery/preparation guards;
- `rollback_created_pristine_session` after its pristine identity checks;
- new `rollback_restored_session` after exact object-identity validation.

Keep session-selection policy outside the purge helper. `close_session` retains its
neighbor-selection behavior; rollback restores only the captured prior active ID if
it still exists.

### Step 3: Write failing hydration and workspace atomicity tests

Add tests for ordinary exceptions and `asyncio.CancelledError` at each boundary:

- policy hydration/reconciliation after the runtime session is created;
- marker overlay;
- effective-scope resolution;
- final transcript/UI synchronization;
- switching to an already-live session whose UI synchronization fails.

For every case assert:

- prior active session and draft/settings remain exact;
- a newly restored runtime target is absent after failure;
- an already-live target remains present;
- no durable delete/update occurs;
- ordinary transient errors return `None` after one approved notification;
- cancellation performs the same rollback and then propagates.

Run the red tests:

```bash
../../.venv/bin/python -m pytest \
  Tests/Chat/test_console_conversation_hydration.py \
  Tests/UI/test_console_workspace_controller.py \
  -k 'rollback or cancellation or atomic or resume' -v
```

### Step 4: Harden hydration and the canonical opener

In `hydrate_console_session`, capture `prior_active_session_id` before creation. If
any operation after `restore_persisted_session` raises, call
`rollback_restored_session` with the exact returned session, then re-raise. Switch
the new session only when `activate=True`.

In `ConsoleWorkspaceController`:

- Change `open_console_workspace_conversation` to return `bool | None` and return the
  canonical outcome on every path.
- Make `_console_session_id_for_workspace_conversation` select the active matching
  session first, then the first other match in creation order.
- Capture the prior active session before any live switch or new hydration.
- Call `hydrate_console_session` with `activate=False`.
- Apply marker overlay, workspace/scope state, native sync, browser refresh, retry
  poke, and composer focus before returning `True`.
- If post-hydration presentation fails, roll back only the exact newly restored
  session, best-effort resynchronize the prior active session, show the approved
  durable error, and return `None`.
- If switching an existing session fails during sync, switch back to the prior
  active session but do not delete either live session.
- On missing tree, retain `False`, mark browser rows broken where applicable, and use
  the approved ID-only copy:

```text
Couldn't resume this saved conversation: it was deleted or couldn't be read.
Your previous Console chat is still active.
```

- Catch `asyncio.CancelledError` separately, perform exact rollback/reswitch, and
  re-raise. Do not convert cancellation to `None`.

Do not call `close_session` from rollback; its user close policy is intentionally
broader than this transaction abort.

### Step 5: Verify Task 3 and commit

```bash
../../.venv/bin/python -m pytest \
  Tests/Chat/test_console_chat_store.py \
  Tests/Chat/test_console_conversation_hydration.py \
  Tests/UI/test_console_workspace_controller.py \
  -k 'resume or rollback or close_session or cancellation or atomic' -v
git diff --check
git add \
  tldw_chatbook/Chat/console_chat_store.py \
  tldw_chatbook/Chat/console_conversation_hydration.py \
  tldw_chatbook/UI/Console_Modules/workspace.py \
  Tests/Chat/test_console_chat_store.py \
  Tests/Chat/test_console_conversation_hydration.py \
  Tests/UI/test_console_workspace_controller.py
git commit -m "fix(console): make saved chat resume atomic"
```

---

## Task 4: Add ID-only navigation and ordered Console startup

**Files:**

- Modify: `tldw_chatbook/Constants.py`
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py`
- Add: `Tests/UI/test_console_roleplay_resume_navigation.py`
- Test: `Tests/UI/test_console_workspace_controller.py`

### Step 1: Write failing navigation-context tests

In the new focused test module, prove `ChatScreen.apply_navigation_context`:

- accepts `{CONSOLE_NAV_CONTEXT_RESUME_LOCAL_CONVERSATION_ID: "  abc  "}` and
  captures `"abc"`;
- ignores non-strings, blanks, and values longer than 256 characters;
- performs no widget query, store creation, hydration, session switch, sync, or
  focus before mount;
- keeps a class-level `None` default for fixtures built with `__new__`.

Use the exact constant:

```python
CONSOLE_NAV_CONTEXT_RESUME_LOCAL_CONVERSATION_ID = (
    "resume_local_conversation_id"
)
```

Run the red test:

```bash
../../.venv/bin/python -m pytest \
  Tests/UI/test_console_roleplay_resume_navigation.py -v
```

### Step 2: Write failing ordered-mount tests

Use spies that append to a call log and cover this exact sequence:

```text
first-chat intent (existing synchronous step)
chat handoff
roleplay repair
prompt insert
provider intent
fleet completion
resume selected conversation
final Console presentation/focus (owned by opener)
```

Assert:

- the five older session-switching consumers are not also scheduled by independent
  0.15-second timers when Resume is pending;
- the mount's first `on_screen_resume` does not duplicate those consumers;
- ordinary non-conflicting startup timers/workers still run;
- initial native sync and focus do not paint an intermediate session;
- a transiently released earlier handoff remains pending in its original channel,
  but the explicit resume still runs last;
- cancellation stops the ordered branch and propagates to its worker;
- the opener receives only the normalized conversation ID.

### Step 3: Implement capture-only context handling

Add class defaults near `_console_mount_visit_refreshed`:

```python
_pending_resume_local_conversation_id: str | None = None
_resume_navigation_startup_in_progress: bool = False
```

Implement `apply_navigation_context(context: Mapping[str, object]) -> None` as a
synchronous validator/capture method only. Do not access DOM or Console runtime
objects there.

### Step 4: Implement one ordered startup branch

When a pending resume ID exists, `on_mount` must schedule one async worker after the
existing DOM settle hedge. That method consumes the ID once and sequentially:

1. awaits `_consume_pending_chat_handoff()`;
2. calls `_consume_pending_console_roleplay_repair()`;
3. awaits `_consume_pending_console_prompt_insert()`;
4. calls `consume_pending_console_provider_intent()`;
5. calls fleet completion and awaits it only when the returned value is awaitable;
6. awaits `_workspace.open_console_workspace_conversation(target)` last.

Suppress the five parallel mount timers, the initial native transcript projection,
and initial focus restoration only for this ordered branch. Gate the mount-generated
`on_screen_resume` from scheduling duplicate prompt/provider/fleet/repair/focus
work while `_resume_navigation_startup_in_progress` is true. Leave unrelated
collapsible, dictation, image, skill, survivor, and task-resume work unchanged.

Clear the in-progress flag in `finally`; do not retry or clone the resume request.
The canonical opener owns success/failure focus and notifications.

### Step 5: Verify Task 4 and commit

```bash
../../.venv/bin/python -m pytest \
  Tests/UI/test_console_roleplay_resume_navigation.py \
  Tests/UI/test_console_workspace_controller.py \
  -k 'navigation or ordering or resume or pending' -v
git diff --check
git add \
  tldw_chatbook/Constants.py \
  tldw_chatbook/UI/Screens/chat_screen.py \
  Tests/UI/test_console_roleplay_resume_navigation.py \
  Tests/UI/test_console_workspace_controller.py
git commit -m "feat(console): resume saved chat after pending intents"
```

---

## Task 5: Build the Roleplay preview Resume interaction

**Files:**

- Modify:
  `tldw_chatbook/UI/Persona_Modules/personas_conversations_controller.py`
- Modify: `tldw_chatbook/UI/Screens/personas_screen.py`
- Modify:
  `tldw_chatbook/Widgets/Persona_Widgets/personas_inspector_pane.py`
- Modify:
  `tldw_chatbook/Widgets/Persona_Widgets/personas_conversation_transcript_widget.py`
- Modify: `tldw_chatbook/css/components/_workbench.tcss`
- Regenerate: `tldw_chatbook/css/widget_defaults_scoped.tcss`
- Regenerate: `tldw_chatbook/css/widget_defaults_self.tcss`
- Test: `Tests/UI/test_personas_workbench.py`

### Step 1: Write failing preview-state and action-hierarchy tests

Extend `TestConversationsPanel` and the production-CSS harness to assert:

- selecting a row immediately opens the read-only loading preview;
- the preview note is always visible outside the scroll region and says exactly
  `Preview shows up to 200 messages. Resume opens the saved chat in Console.`;
- empty content says `No messages to display.` while a fetch exception says exactly
  `Couldn't load this preview. You can still resume the saved chat.`;
- Resume remains enabled in loading, empty, and load-failure states for a valid row;
- Send transcript refuses loading/failure and never stages an empty failure as
  context;
- all inspector/card-level chat, export, Buddy, and delete actions are hidden during
  preview and restored on Back;
- hidden card actions cannot still fire through Ctrl+Enter and disappear from footer
  hints while preview is open;
- action structure is exactly three vertical rows:
  1. full-width primary **Resume chat**;
  2. full-width secondary **Send transcript to Console draft**;
  3. equal subdued **Back to card** and **Open in Library** buttons;
- no conversation preview load completion steals focus from the conversations list;
- F6 lands on Resume and Tab traversal is
  Resume → Send transcript → Back → Open in Library → transcript scroll;
- at 80x24 and standard width, every action is contained and the transcript scroll
  retains positive usable height under production consolidated CSS;
- the disabled busy button's rendered foreground/background contrast meets the
  repository's disabled-control floor.

Update the old focus test that currently expects transcript-scroll focus after load.

Run the red tests:

```bash
../../.venv/bin/python -m pytest Tests/UI/test_personas_workbench.py \
  -k 'conversation or workbench_focus or compact or contrast' -v
```

### Step 2: Add a persistent inspector visibility gate

In `PersonasInspectorPane` add retained state and a public setter:

```python
_card_actions_visible: bool = True

def set_card_actions_visible(self, visible: bool) -> None:
    self._card_actions_visible = bool(visible)
    self._apply_action_state()
```

Change `_apply_action_state` so `#personas-inspector-actions` displays only when
both a selection exists and `_card_actions_visible` is true. This is intentionally a
retained gate: later selection/readiness syncs must not re-show the card actions
over a transcript preview.

In `PersonasScreen._show_center`, call the setter with
`visible_id != _CONVERSATION_VIEW_ID` before returning. Add
`_conversation_preview_is_open()` and make `_console_action_allowed()` false in
preview so keyboard and footer behavior share the same rule.

### Step 3: Build the explicit three-row layout

In `PersonasScreen.compose`, replace the current `Horizontal` action container with
a `Vertical` containing:

```python
yield Button(
    "Resume chat",
    id="personas-conversation-resume",
    classes="console-action-primary",
)
yield Button(
    "Send transcript to Console draft",
    id="personas-conversation-continue-console",
    classes="console-action-secondary",
)
with Horizontal(id="personas-conversation-navigation-actions"):
    yield Button(
        "Back to card",
        id="personas-conversation-back",
        classes="console-action-subdued",
    )
    yield Button(
        "Open in Library",
        id="personas-conversation-open-library",
        classes="console-action-subdued",
    )
```

Give the container an explicit nine-line height: three lines per row. Make Resume
and transcript-draft full width; make the bottom buttons `1fr` each with no overflow.
Retain DOM order as the Tab order.

Prepend `personas-conversation-resume` to the center work-area's preferred F6 focus
targets. Its hidden ancestor makes it unavailable outside preview.

### Step 4: Separate loading, loaded, and failed preview state

In `PersonasConversationsController`:

- add `_failed_conversation_id` and clear it on each new open;
- have the worker report a distinct failure continuation rather than converting an
  exception to `history=[]`;
- leave `_loaded_conversation_id` unset on failure;
- call `PersonasConversationTranscriptWidget.show_error()` for the matching current
  selection only;
- keep Resume keyed to `_open_conversation_id`, not transcript load state;
- keep bounded draft staging keyed to `_loaded_conversation_id` and the existing
  6000-character body/truncation fields;
- remove `call_after_refresh(_focus_conversation_transcript)` from successful load.

In the transcript widget, compose the fixed preview note between the title and
`VerticalScroll`; add `show_error()` that replaces only the scroll contents.

### Step 5: Add the Roleplay Resume handler and single-flight state

Add `resume_in_console()` to the controller and wire
`#personas-conversation-resume` in the screen. It must:

1. capture and normalize the currently open conversation ID into a local variable;
2. verify that ID still exists in `_conversation_rows`, otherwise stay in Roleplay
   and notify `This conversation is no longer available. Refresh conversations and
   try again.`;
3. ignore a repeated press for the same in-flight target;
4. disable the button and change its label to `Opening Console…`;
5. post only:

```python
NavigateToScreen(
    TAB_CHAT,
    {CONSOLE_NAV_CONTEXT_RESUME_LOCAL_CONVERSATION_ID: target_id},
)
```

6. schedule a short fallback that restores the exact target's button/guard only if
   the same Roleplay screen is still mounted and the same target remains open.

Do not call `_stage_handoff`, copy the transcript, resolve the title, or attach RAG
metadata in this path.

### Step 6: Add readable busy styling and regenerate CSS

Add a narrowly scoped app-tier rule in `_workbench.tcss` for
`#personas-conversation-resume:disabled` that neutralizes the generic 50% opacity
stack while keeping the control visibly disabled. Use existing `$ds-*`/surface
tokens and the repository's established app-tier disabled idiom; do not change all
disabled buttons.

Regenerate both consolidated files from the source builder:

```bash
../../.venv/bin/python -m tldw_chatbook.css.build_css
```

Never hand-edit the generated TCSS mirrors.

### Step 7: Verify Task 5 and commit

```bash
../../.venv/bin/python -m pytest Tests/UI/test_personas_workbench.py \
  -k 'conversation or workbench_focus or compact or contrast' -v
../../.venv/bin/python -m tldw_chatbook.css.build_css
git diff --check
git add \
  tldw_chatbook/UI/Persona_Modules/personas_conversations_controller.py \
  tldw_chatbook/UI/Screens/personas_screen.py \
  tldw_chatbook/Widgets/Persona_Widgets/personas_inspector_pane.py \
  tldw_chatbook/Widgets/Persona_Widgets/personas_conversation_transcript_widget.py \
  tldw_chatbook/css/components/_workbench.tcss \
  tldw_chatbook/css/widget_defaults_scoped.tcss \
  tldw_chatbook/css/widget_defaults_self.tcss \
  Tests/UI/test_personas_workbench.py
git commit -m "feat(roleplay): resume saved character chats"
```

---

## Task 6: Joined verification, self-review, and backlog completion

**Files:**

- Modify:
  `backlog/tasks/task-22988 - Resume-prior-character-chats-from-Roleplay.md`
- Modify lessons documentation only if implementation exposes a genuinely reusable
  incident-backed lesson.

### Step 1: Run the complete targeted feature gate

```bash
../../.venv/bin/python -m pytest \
  Tests/Chat/test_console_roleplay_metadata.py \
  Tests/Chat/test_chat_persistence_service.py \
  Tests/Chat/test_console_chat_store.py \
  Tests/Chat/test_console_conversation_hydration.py \
  Tests/UI/test_console_resume_active_path.py \
  Tests/UI/test_console_workspace_controller.py \
  Tests/UI/test_console_native_chat_flow.py \
  Tests/UI/test_console_roleplay_resume_navigation.py \
  Tests/UI/test_personas_workbench.py \
  -v
../../.venv/bin/python -m tldw_chatbook.css.build_css
git diff --check
```

Do not run the full repository suite without asking the user first.

### Step 2: Inspect the final diff against every acceptance criterion

```bash
git status --short
git diff --stat origin/dev...HEAD
git diff origin/dev...HEAD -- \
  tldw_chatbook/Chat \
  tldw_chatbook/UI \
  tldw_chatbook/Widgets \
  Tests/Chat \
  Tests/UI \
  backlog/decisions/046-roleplay-chat-display-identity-and-template-provenance.md
```

Self-review specifically for:

- ID-only navigation with no title/transcript/RAG leakage;
- no Roleplay duplicate of Console hydration;
- active matching session precedence and preserved live draft/settings;
- v1 no-guess behavior and v3 fail-closed behavior;
- exact-object runtime rollback and cancellation propagation;
- no intermediate mount sync/focus that can override the final target;
- preview action hiding that survives state recomputation;
- 80x24 containment and readable busy state under generated production CSS;
- unchanged bounded draft handoff and Library navigation.

If review finds a defect, add a failing targeted regression test before the fix and
rerun the affected task gate plus the complete targeted feature gate.

### Step 3: Complete backlog hygiene only after green evidence

Using Backlog CLI:

1. Check all twelve acceptance criteria.
2. Add concise implementation notes listing the approach, core files, ADR-046
   amendment, tri-state/cancellation behavior, targeted commands, and exact results.
3. Record any plan deviation.
4. Add an incident-backed lesson only if one actually occurred.
5. Set TASK-22988 to Done only when every Definition-of-Done item is satisfied.

Example command shape; fill the notes with the actual observed evidence:

```bash
backlog task edit 22988 --notes "Implemented the approved ID-only Roleplay-to-Console resume flow; amended ADR-046; verified the recorded targeted suites and generated CSS."
backlog task edit 22988 -s Done
```

Do not mark Done if any acceptance criterion, targeted test, static check, generated
CSS verification, documentation update, or self-review item remains incomplete.

### Step 4: Commit completion documentation

```bash
git add \
  'backlog/tasks/task-22988 - Resume-prior-character-chats-from-Roleplay.md' \
  backlog/docs
git commit -m "docs(backlog): complete roleplay chat resume"
git status --short --branch
```

If no lessons file changed, stage only the task file. The final status must show no
uncommitted feature changes.
