# TASK-427 — Start Chat opens a real character conversation in native Console

- **Date:** 2026-07-21
- **Task:** TASK-427 (P0, from the RP/character-card UX review `Docs/superpowers/qa/rp-ux-review-2026-07-21/report.md`)
- **Branch base:** origin/dev (schema v22)
- **Revision:** 2 — incorporates an adversarial design review (greeting-must-be-display-only-to-provider; message-owning agent gate; explicit UI-sync step for AC#3; workspace scoping; inline card fetch).

## Problem

"Start Chat" / "Open in Console" from the Roleplay workbench does not open a live character conversation in the native Console. The handoff degrades to invisible "staged context" (`chat_screen.py:_stage_handoff_as_console_live_work`): blank transcript, no greeting, no character identity, composer prefilled with a meta-instruction (`Respond as {name}.`). Observed live: sending that prefilled text routes through the agent/sub-agent harness, which replies *"Please provide the previous conversation and the character details you'd like me to use."* — the staged context never reaches the model. The RP core loop is a dead end.

## Goal (acceptance criteria)

1. Start Chat on a character opens Console with the character greeting visible as the **first assistant message**.
2. The created conversation is **titled/labelled with the character name** and retains that identity **across app restarts**.
3. The first user send **replies in character** (card system prompt + definition applied) without the user re-describing the character.
4. Character sends **do not route through the agent/sub-agent harness** unless the user has explicitly enabled an agent profile.

## Scope

**In scope:** the inspector **Start Chat** button (`#personas-start-chat`) and the **Attach** button when it carries `intent="start_chat"` — the path that carries character identity in handoff metadata (`_character_session_identity_from_handoff` fires only for `intent=="start_chat"` + `selected_kind=="character"`).

**Out of scope (deferred):**
- Character-scoped **dictionaries and world-info** on the native send path (deferred in PR #664 / P2g). Only the card **system prompt** applies here.
- The preview pane's **"Open in Console"** transcript-continuation flow.
- A **per-session "use the agent with this character" opt-in** UI. None exists today; character sessions are plain-provider. A future opt-in can layer on the gate described in §5.
- Persona (user-profile) Start Chat (`selected_kind=="persona_profile"`): falls through to the existing staged-context behavior, unchanged.

**No ChaChaNotes migration.** The `conversations` table already has `character_id INTEGER REFERENCES character_cards(id)` (`ChaChaNotes_DB.py:263-280`, schema v22), and `ChatPersistenceService.create_conversation` already accepts `character_id`/`character_name`/`assistant_kind`/`conversation_title`/`system_prompt` (`chat_persistence_service.py:44-98`). The native session-creation path simply never passes them.

## Design

### 1. Session identity (in-memory)

Add to `ConsoleChatSession` (`console_chat_store.py:148-160`):
- `character_id: int | None = None`
- `character_name: str | None = None`

`character_id` is the single source of truth for "this is a character session": it drives persistence (§2), the plain-provider gate (§5), and greeting handling (§7). The card system prompt rides the existing `ConsoleSessionSettings.system_prompt`; `character_label` (`:166`) carries the display name for the rail.

No `agent_runtime_enabled` field is added — the plain-provider decision keys on `character_id` (§5), which already persists, so it survives restart for free (this is why the review's I2 "agent override lost on restart" cannot occur here).

### 2. Persist identity

In `persist_session_if_needed` (`console_chat_store.py:948-984`): when `session.character_id is not None`, call `create_conversation` with `character_id=session.character_id`, `character_name=session.character_name`, `assistant_kind="character"`, `assistant_id=str(session.character_id)` (instead of the hardcoded `assistant_kind="generic", assistant_id="console"`). `system_prompt` is already passed from settings. The adapter derives a "Chat with {name}" title (preserved on resume — see §6). A non-character session keeps `generic`/`console` unchanged.

`session.character_id` is set **before** the first `persist=True` message (§3 orders create → set identity → seed), and `persist_session_if_needed` guards on `persisted_conversation_id`, so `create_conversation` runs exactly once, with character fields present.

### 3. Native character branch (handoff consumer)

In `ChatScreen._consume_pending_chat_handoff` (`chat_screen.py:8741-8779`), before the native staged-context degradation (`tab_container is None` → `_stage_handoff_as_console_live_work`), add a character branch. **All work happens inline within the existing guarded (`_handoff_consumption_in_progress`) `try/finally` body** — no detached worker — so the re-entrancy guard and the `pending_chat_handoff` clear stay correct and a failure can still fall back:

1. Detect via `_character_session_identity_from_handoff(payload)` (`chat_screen.py:442-475`) returning non-`None` `(character_id, character_name, assistant_id)`.
2. **Fetch the card by id off the event loop**: `card = await asyncio.to_thread(db.get_character_card_by_id, character_id)` (`ChaChaNotes_DB.py:4326`; the DB uses `threading.local` connections with `check_same_thread=False`, so a threaded read is safe). Wrap in `try/except`. Use `get_character_card_by_id`, **not** `load_character_and_image` (the latter decodes a PIL image the caller must `.close()` and runs placeholder substitution over the system-prompt fields — both unwanted here).
3. **On any failure** (identity `None`, card missing, exception): fall through to `_stage_handoff_as_console_live_work(payload)` unchanged, then clear `pending_chat_handoff`. No regression.
4. **Build the effective system prompt** exactly like the preview (`personas_preview_controller.py:189-194`): `"\n".join` of the non-empty `[system_prompt, personality, description, scenario]` card fields, fallback `"Stay in character."`. Apply `replace_placeholders` **only** to the greeting (`first_message`), not to the system prompt (matches the preview).
5. **Display name** comes from the fetched `card["name"]` (metadata `selected_name` can be blank → "Chat with " / empty label). Use the metadata id only for routing.
6. **Create a dedicated new session** via `store.create_session(...)` (NOT `ensure_session` — a character chat must not pollute the active conversation), with `workspace_id=CONSOLE_GLOBAL_WORKSPACE_ID` (character chats are global `ccp_character` discovery, never workspace work — this avoids the `create_conversation` `ValueError` path for workspace scopes and keeps the conversation visible in the character's conversation browser, which excludes `scope_type=="workspace"`). Set `title="Chat with {name}"`, `settings.system_prompt` = built prompt, `character_label=name`, `character_id`/`character_name` on the session.
7. **Seed the greeting** as the first message: `store.append_message(session_id, role=ConsoleMessageRole.ASSISTANT, content=<substituted first_message>, persist=True)`, wrapped in `try/except` (a persist failure must not crash the consumer). Substitution: `replace_placeholders(first_message, name, "User")`.
8. **Make it the visible, synced session (see §4)**, then clear `pending_chat_handoff`.

### 4. Make the session visible AND apply the card prompt on the first send (AC#3)

This is the step the first design version got wrong. The send path emits `self.system_prompt` (the **controller** attribute), not `session.settings.system_prompt` (`_leading_system_message`, `console_chat_controller.py:2611`). `self.system_prompt` is only refreshed from the active session when `update_provider_selection` runs, which happens inside `_sync_native_console_chat_ui()` → `_sync_console_chat_core_state()`.

Because `store.create_session` **pre-activates** the new session (`console_chat_store.py:234`), the shared `_activate_native_console_session` (`chat_screen.py:1354-1367`) early-outs on `if active_session_id != session_id:` and **skips the sync**. So the handoff consumer must, after create + seed, call **`await self._sync_native_console_chat_ui()` directly** (renders the transcript incl. the greeting, refreshes the session list, and runs `update_provider_selection` so `controller.system_prompt` = the card prompt for the first send), then `self._focus_console_composer_if_needed(force=True)`.

No other send-path change is needed for AC#3: with `controller.system_prompt` set, `_leading_system_message` prepends the card prompt on submit/retry/regenerate/continue.

### 5. Plain provider path (AC#4)

The agent-vs-plain gate is `console_chat_controller.py:2051`:
```
if self._agent_runtime_enabled and self._agent_bridge is not None and not prefill:
    return await self._run_agent_reply(...)
```
Change it to consult the **session that owns the message being streamed**, derived race-free from the message id (reading `self.store.active_session_id` at the gate is racy — a session switch is not blocked during a run's async setup):
```
owner_id = self.store.session_id_for_message(assistant_message_id)
owner = next((s for s in self.store.sessions() if s.id == owner_id), None)
force_plain = owner is not None and owner.character_id is not None
if self._agent_runtime_enabled and self._agent_bridge is not None and not prefill and not force_plain:
    ... agent ...
```
(`session_id_for_message` at `console_chat_store.py:690`; the `next(... sessions())` lookup mirrors the established pattern at `console_chat_controller.py:1766/1862`.) A character session (`character_id` set) therefore always takes the plain-provider branch (`provider_gateway.stream_chat`), regardless of the global `[console] agent_runtime` default, and **across restarts** (character_id persists and is restored in §6). A future per-session "use agent" opt-in would relax `force_plain`.

### 6. Restart identity (AC#2)

`_resume_console_workspace_conversation` (`chat_screen.py:3387-3481`) restores title/workspace/messages/settings but not `character_id`. Extend it to set `session.character_id = conversation.get("character_id")` on the restored `ConsoleChatSession` (the value round-trips via `normalize_conversation_row`, `chat_conversation_service.py:223`). This restores both identity **and** the plain-provider behavior (§5 keys on `character_id`). The title ("Chat with {name}") and the seeded greeting message already persist and restore; `system_prompt` restores via `_console_session_settings_for_resume` (`chat_screen.py:3370-3384`), so AC#3 also holds across restart. `character_name` is not stored as a column; the rail label after restart derives from the persisted title (acceptable — no name-dependent send logic).

### 7. Greeting is display-only to the provider

Seeding the greeting as a normal ASSISTANT message makes it eligible for the provider payload (`_provider_message_payloads` includes USER and ASSISTANT, `console_chat_controller.py:2645+`), producing `[system, assistant(greeting), user]`. **Anthropic rejects a leading assistant turn (400)** — `LLM_API_Calls.py:1093` only checks that *some* user message exists and never reorders — and strict Gemini alternation likewise. The preview avoids this by keeping the greeting pane-only (`self.history` empty; sends `[system]+[user]`).

Fix: in `_provider_message_payloads`, **skip assistant messages that precede the first payload-eligible user message**. This:
- makes the first character send `[system, user]` (valid for all providers);
- is a **no-op** for normal user-first sessions;
- matches the preview (greeting not in model context — decided deliberately over a synthetic-user-prepend, for preview parity; the tradeoff is the model does not see its own opening line on turn 1, mitigated by the card system prompt).

**Edge — regenerate/continue on the seeded greeting:** the greeting is a `complete` assistant message, so `regenerate_message`/continue would target it and (with the drop) build a `[system]`-only payload → provider error. Guard both paths: when the target assistant message has **no preceding payload-eligible user turn**, deny the action with a gentle status ("Nothing to regenerate before the character's opening line.") rather than send an invalid payload.

## Data flow

```
Start Chat (inspector)
  → _attach_selection_to_console(intent="start_chat")
  → _stage_handoff(payload{metadata: selected_kind=character, selected_record_id=<id>, selected_name, intent})
  → app.open_chat_with_handoff(payload)   [gated on chat_defaults.enable_tabs=true by default — unchanged]
  → NavigateToScreen(Console)
  → _consume_pending_chat_handoff  [guarded body]
      → [character branch] resolve id → await to_thread(get_character_card_by_id) → build 4-part system prompt
      → store.create_session(global workspace, title "Chat with {name}", system_prompt, character_id/name)
      → store.append_message(ASSISTANT, substituted greeting, persist=True)   [try/except]
      → await _sync_native_console_chat_ui()   → update_provider_selection → controller.system_prompt = card prompt
      → focus composer; clear pending_chat_handoff
      → [on any failure] _stage_handoff_as_console_live_work(payload)  (unchanged fallback)
First user send
  → gate keys on owner.character_id → plain provider path
  → _provider_message_payloads drops the leading greeting → [system(card prompt), user]
  → in-character reply
Restart
  → _resume_console_workspace_conversation sets session.character_id from the row
  → identity + plain-provider behavior restored; title + greeting + system_prompt persisted
```

## Testing

- **Store** (`Tests/Chat/test_console_chat_store.py`, `FakePersistence`): `persist_session_if_needed` on a character-bound session passes `character_id`/`character_name`/`assistant_kind="character"`/`assistant_id` to `create_conversation`; a non-character session still passes `generic`/`console`.
- **Controller — agent gate** (`Tests/Chat/test_console_chat_controller.py`): with an agent bridge present and global `agent_runtime` enabled, a send in a **character** session (character_id set) takes the **plain** path (agent bridge not invoked); a normal session still routes to the agent. Assert via the message-owning session, not the active session.
- **Controller — greeting exclusion** (`CapturingGateway`): a session whose first message is a seeded ASSISTANT greeting sends `[system, user]` on the first user turn (no leading assistant); the card system prompt is the leading system message. Regenerate/continue targeting the greeting is denied (no invalid payload).
- **Handoff / screen** (`Tests/UI/test_chat_first_handoffs.py` or a new native-branch test): a native-Console handoff with `intent=start_chat` + character metadata creates a **dedicated** character session in the **global** workspace, seeds the greeting as the **first ASSISTANT message**, titles it "Chat with {name}", and calls `_sync_native_console_chat_ui`; an unresolvable/id-less or fetch-failing payload **falls back** to the staged-context path with no exception.
- **Restart** (resume path test or store): a resumed conversation row carrying `character_id` restores `session.character_id`, so its sends stay on the plain path.
- **Live-verify** (real TUI, scratch profile + local OpenAI-compat mock on :9099): Start Chat → greeting renders as first assistant message → send → in-character reply via plain provider → restart the app → conversation still titled/identified with the character and still plain-path.

## Risks / mitigations

- **Assistant-first provider 400** → §7 drops the leading greeting from the payload (verified against the Anthropic handler).
- **AC#3 silent failure on first send** → §4 calls `_sync_native_console_chat_ui()` directly (the pre-activation early-out would otherwise skip the prompt sync).
- **Wrong workspace / crash on seed** → §3/§6 force `CONSOLE_GLOBAL_WORKSPACE_ID`; the greeting-seed persist is `try/except`.
- **Racy agent gate** → §5 keys on the message-owning session, not the active session.
- **Re-entrancy / lost fallback** → §3 keeps the card fetch inline (`asyncio.to_thread`) within the guarded body; failure falls through to the staged-context path.
- **Regression on the handoff path** → the character branch is additive and guarded; every failure falls through to today's exact behavior.
- **`enable_tabs` gate** (`app.py:3476`, default `true`): Start Chat already no-ops when a user disabled tabs. Pre-existing, out of scope; noted, not changed.

## Non-goals

- Character dictionaries / world-info on native send (deferred follow-up).
- Preview "Open in Console" transcript continuation.
- Per-session "use the agent with this character" opt-in UI.
- Changing the `enable_tabs` gate or any schema.
