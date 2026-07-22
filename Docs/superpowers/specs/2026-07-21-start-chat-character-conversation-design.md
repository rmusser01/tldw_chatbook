# TASK-427 — Start Chat opens a real character conversation in native Console

- **Date:** 2026-07-21
- **Task:** TASK-427 (P0, from the RP/character-card UX review `Docs/superpowers/qa/rp-ux-review-2026-07-21/report.md`)
- **Branch base:** origin/dev (schema v22)

## Problem

"Start Chat" / "Open in Console" from the Roleplay workbench does not open a live character conversation in the native Console. The handoff degrades to invisible "staged context" (`chat_screen.py:_stage_handoff_as_console_live_work`): blank transcript, no greeting, no character identity, composer prefilled with a meta-instruction (`Respond as {name}.`). Observed live: sending that prefilled text routes through the agent/sub-agent harness, which replies *"Please provide the previous conversation and the character details you'd like me to use."* — the staged context never reaches the model. The RP core loop is a dead end.

## Goal (acceptance criteria)

1. Start Chat on a character opens Console with the character greeting visible as the **first assistant message**.
2. The created conversation is **titled/labelled with the character name** and retains that identity **across app restarts**.
3. The first user send **replies in character** (card system prompt + definition applied) without the user re-describing the character.
4. Character sends **do not route through the agent/sub-agent harness** unless the user has explicitly enabled an agent profile.

## Scope

**In scope:** the inspector **Start Chat** button (`#personas-start-chat`) and the **Attach** button when it carries `intent="start_chat"` — the path that carries character identity in handoff metadata.

**Out of scope (deferred to existing follow-up work):**
- Character-scoped **dictionaries and world-info** on the native send path (the `_active_console_dictionary_scope_ids` / applier `char_data=None` seam — explicitly deferred in PR #664 / P2g). Only the card **system prompt** applies here.
- The preview pane's **"Open in Console"** (continue-an-existing-transcript) flow — genuinely different "continue this transcript" semantics.

**No ChaChaNotes migration.** The `conversations` table already has `character_id INTEGER REFERENCES character_cards(id)` (`ChaChaNotes_DB.py:263-280`, schema v22), and `ChatPersistenceService.create_conversation` already accepts `character_id`/`character_name`/`assistant_kind`/`conversation_title`/`system_prompt` (`chat_persistence_service.py:44-98`). The native session-creation path simply never passes them.

## Design

### 1. Session identity (in-memory)

Add to `ConsoleChatSession` (`console_chat_store.py:148-160`):
- `character_id: int | None = None`
- `character_name: str | None = None`
- `agent_runtime_enabled: bool | None = None` — per-session override of the global agent-runtime default (`None` = inherit global; `False` = force plain provider).

The card system prompt is carried by the existing `ConsoleSessionSettings.system_prompt` (`console_session_settings.py:172`); `character_label` (`:166`) carries the display name.

### 2. Persist identity

In `persist_session_if_needed` (`console_chat_store.py:948-984`): when `session.character_id is not None`, call `create_conversation` with `character_id=session.character_id`, `character_name=session.character_name`, `assistant_kind="character"`, `assistant_id=str(session.character_id)` (instead of the hardcoded `assistant_kind="generic", assistant_id="console"`). The adapter derives a "Chat with {name}" title and writes `conversations.character_id`. `system_prompt` is already passed from settings.

### 3. Native character branch (handoff consumer)

In `ChatScreen._consume_pending_chat_handoff` (`chat_screen.py:8741-8779`), before the native staged-context degradation (`tab_container is None` → `_stage_handoff_as_console_live_work`), branch when the payload is a character Start-Chat:

- Detect via `_character_session_identity_from_handoff(payload)` (`chat_screen.py:442-475`) returning a non-`None` `(character_id, character_name, assistant_id)` (it already reads `metadata["intent"]=="start_chat"` + `metadata["selected_kind"]=="character"` and extracts the numeric id).
- **Re-fetch the card by id off-thread** (`run_worker(..., exit_on_error=False)`; recurring app-crash guard for off-thread reads) via `Character_Chat_Lib.load_character_and_image` / `db.get_character_card_by_id` (`ChaChaNotes_DB.py:4326`) to obtain `first_message` and the system-prompt/definition fields.
- **Build the effective system prompt** from the card (`system_prompt` plus definition fields), mirroring the Personas preview's `system_prompt()` construction so the Console character behaves like the preview.
- **Create a dedicated new session** (`store.create_session`, NOT `ensure_session` — a character chat must not pollute the active conversation), title `Chat with {name}`, set `settings.system_prompt` from the card, `character_label = name`, `character_id`/`character_name` on the session, and `agent_runtime_enabled = False`.
- **Seed the greeting** as the first message: `store.append_message(session_id, role=ConsoleMessageRole.ASSISTANT, content=<placeholder-substituted first_message>, persist=True)`. Placeholder substitution via `replace_placeholders(first_message, name, user_name)` (same helper the preview uses).
- Switch the Console to that session and focus the composer.
- Clear `app_instance.pending_chat_handoff`.

**Fallback (no regression):** if `_character_session_identity_from_handoff` returns `None`, or the card fetch fails / has no id, fall through to the existing `_stage_handoff_as_console_live_work(payload)` path unchanged.

### 4. Card system prompt on send (AC #3)

No new send-path code: the controller already emits `session.settings.system_prompt` as the leading system message via `_leading_system_message` / `_provider_messages_for_session` (`console_chat_controller.py:2598-2629`) on every send. Setting `settings.system_prompt` at session creation (step 3) satisfies AC #3.

### 5. Plain provider path (AC #4)

The agent-vs-plain decision is at `console_chat_controller.py:2040-2062`:
```
if self._agent_runtime_enabled and self._agent_bridge is not None and not prefill:
    return await self._run_agent_reply(...)
```
Change the gate to consult a **per-session** override: the effective agent-runtime is `session.agent_runtime_enabled` when set, else the global `self._agent_runtime_enabled`. Character sessions set the override to `False`, so they always take the plain-provider branch (`provider_gateway.stream_chat`) regardless of the global default. A user who explicitly enables the agent for that session flips the override — satisfying "unless the user has explicitly enabled an agent profile."

### 6. Restart identity (AC #2)

`_resume_console_workspace_conversation` (`chat_screen.py:3387-3481`) currently restores title/workspace/messages/settings but not `character_id`. Extend it to read `conversation.get("character_id")` (and character name) back onto the restored `ConsoleChatSession` so the identity survives restart. The title and the seeded greeting message already persist via the conversation row + messages.

## Data flow

```
Start Chat (inspector)
  → _attach_selection_to_console(intent="start_chat")
  → _stage_handoff(payload{metadata: selected_kind=character, selected_record_id=<id>, selected_name, intent})
  → app.open_chat_with_handoff(payload)  [gated on chat_defaults.enable_tabs — pre-existing, unchanged]
  → NavigateToScreen(Console)
  → _consume_pending_chat_handoff
      → [character branch] resolve id → fetch card (worker) → build system prompt
      → store.create_session(identity, system_prompt, agent_off)
      → store.append_message(ASSISTANT, greeting, persist)
      → switch session, focus composer
First user send
  → plain provider path (agent override off) with card system prompt as leading system message
  → in-character reply
Restart
  → _resume_console_workspace_conversation reads character_id back → identity restored; title + greeting persisted
```

## Testing

- **Store** (`Tests/Chat/test_console_chat_store.py`, `FakePersistence`): `persist_session_if_needed` on a character-bound session passes `character_id`/`character_name`/`assistant_kind="character"` to `create_conversation`; a non-character session still passes `generic`/`console` (no regression).
- **Controller** (`Tests/Chat/test_console_chat_controller.py`, `CapturingGateway`): a character session (agent override `False`) with an agent bridge present still takes the **plain provider** path (agent bridge not invoked) and prepends the card system prompt as the leading system message; a normal session with the global agent default unchanged still routes to the agent path.
- **Handoff / screen** (`Tests/UI/test_chat_first_handoffs.py` or a new native-branch test): a native-Console handoff with `intent=start_chat` + character metadata creates a **dedicated** character-bound session, seeds the greeting as the **first ASSISTANT message**, and titles it "Chat with {name}"; an unresolvable/id-less payload **falls back** to the staged-context path.
- **Live-verify** (real TUI, scratch profile + local OpenAI-compat mock on :9099): Start Chat → greeting renders as first assistant message → send → in-character reply via plain provider → restart the app → conversation still titled/identified with the character.

## Risks / mitigations

- **Off-thread card fetch** → `run_worker(exit_on_error=False)` (uncaught worker exceptions kill the app — recurring lesson).
- **Regression risk on the handoff path** → the character branch is additive and guarded; any failure to resolve identity falls through to the exact current staged-context behavior.
- **`enable_tabs` gate** (`app.py:3476`): Start Chat already no-ops today when a user disabled tabs. Pre-existing, out of scope; noted, not changed.
- **Agent bridge is `None` for in-memory DBs** — the per-session override must be consulted *in addition to* the existing `_agent_bridge is not None` check; character sessions force plain regardless.

## Non-goals

- Character dictionaries / world-info on native send (deferred follow-up).
- Preview "Open in Console" transcript continuation.
- Changing the `enable_tabs` gate.
- Any schema change.
