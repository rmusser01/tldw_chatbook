# Roleplay — Native Console send-path chat-dictionary application

**Status:** Implemented.

**Program:** Roleplay (Personas) redesign. Follow-up filed from P1g (#661).

## Why

The native Console send path applies **no** chat dictionaries. `collect_active_chatdict_entries` has exactly one caller — `Event_Handlers/Chat_Events/chat_events.py:981`, the **legacy** sidebar send. `ConsoleChatController.submit_draft` (and the agent bridge / provider gateway) never apply dictionaries. So dictionaries authored in Roleplay and attached via the P1g Console inspector show as "in play" but do **not** shape the actual model call on the native Console — the primary chat surface. The written `metadata.active_dictionaries` is correct and *does* apply if the conversation is opened in the legacy chat surface, but not in the Console itself.

This cycle wires the native Console send to apply the same shared-resolver union the P1g summary shows, so **"shown = applied"** holds end-to-end on the surface users actually chat in.

## Scope

**In scope:** apply **conversation** chat dictionaries (`metadata.active_dictionaries`) on every native Console send path (fresh submit, retry, continue, regenerate), covering both the provider and agent-bridge branches. Substitution is **ephemeral** — the provider payload for that turn is transformed; the persisted transcript keeps the raw text (matching the legacy send and the existing skill-substitution behavior).

**Deferred (explicitly out of scope):**
- **Character** dictionaries on the native send. Native sessions carry only a free-text `character_label` (`console_session_settings.py:162`), never a character card/id, so `char_data` is always `None` and no character dicts resolve. Character-dict application on native sends requires native sessions to first *carry* a real character — its own later cycle. The applier keeps a `char_data` parameter (always `None` now) so that cycle plugs in without a rename.
- **Per-send diagnostics** (which entries fired / near-misses). Application is silent; the P1g inspector "What's in play" remains the sole surfacing. A "Console dictionary diagnostics" surface can be a later cycle.
- No new user-facing enable/disable toggle. Parity with the legacy send, which applies whenever entries exist.

## Ground truths (verified at dev `7b227601`)

- **Single seam.** `ConsoleChatController` builds `provider_messages` (from stored raw messages) then runs `_apply_skill_substitution(provider_messages)` at **4 sites**: `submit_draft` (:255), `retry_message` (:528), `continue_from_message` (:561), `regenerate_message` (:604). Each site has `session_id` (or `session.id`) in scope, checks `refuse`, then calls `_stream_assistant_response`.
- **Both branches covered pre-split.** `_stream_assistant_response` dispatches to `_run_agent_reply` (agent runtime) or the provider stream. The agent path splits the leading system message off `provider_messages` and forwards the rest as `agent_messages=` to `run_reply` (:994-1035). So a transform applied to `provider_messages` *before* `_stream_assistant_response` reaches both branches.
- **Message content shape.** `_provider_message_payloads` emits a user turn's `content` as a **str** for text-only turns (:1327) and as a **parts list** (`{"type":"text","text":…}` + image parts) when the turn carries budgeted images (:1305); an over-budget image-only turn becomes a str placeholder (:1324).
- **Controller has no db and no character.** Constructor deps include `store`, `provider_gateway`, `_skills_service`, `_agent_bridge` — no `chachanotes_db`. `session.persisted_conversation_id` is the DB conversation id (None for an unsaved session). `_agent_conversation_id()` returns `persisted_conversation_id or session_id` — the fallback is **not** a DB conversation and must not be used for dictionary lookup.
- **Applier primitives.** `Chat_Dictionary_Lib.collect_active_chatdict_entries(db, conversation_id, char_data) -> List[ChatDictionary]` (never-raise union) + `process_user_input(text, entries: List[ChatDictionary], max_tokens, strategy) -> str` (never-raise text substitution). The legacy send uses `max_tokens=500`, `strategy="sorted_evenly"`, and applies them on a `thread=True` worker (`chat_events.py:1258`).
- **Native sends run on the UI event loop.** `_submit_console_native_draft` (and the retry/continue/regenerate handlers) are dispatched via `run_worker(<coroutine>, …)` — an **async** worker, NOT `thread=True` (`chat_screen.py:7983`), so it runs on the app's event loop. Any synchronous DB/CPU work in the send path therefore blocks the UI loop and must be offloaded (`asyncio.to_thread`). `collect_active_chatdict_entries` (DB read) + `process_user_input` (regex matching) are synchronous.
- **Single construction site.** `ChatScreen` builds the controller once at `chat_screen.py:2369` (injecting `skills_service=`, `agent_bridge=`, …). `self.app_instance.chachanotes_db` is the db handle.
- `COMMAND_PREFIX = "/"` (`console_command_grammar.py:25`).

## Architecture

Mirror the established skill-substitution pattern: a post-assembly, pre-stream transform on `provider_messages`, driven by an injected callable.

### Components

1. **`Chat_Dictionary_Lib.apply_active_chatdicts_to_text(db, conversation_id, char_data, text, *, max_tokens, strategy) -> str`** (new). Never-raise convenience: `entries = collect_active_chatdict_entries(db, conversation_id, char_data)`; if no entries, return `text` unchanged; else `return process_user_input(text, entries, max_tokens=max_tokens, strategy=strategy)`. Any exception → return the original `text`. `char_data` is accepted (always `None` from the native caller) for the deferred character cycle.

2. **`ConsoleChatController`** gains:
   - a constructor param `chat_dictionary_applier: Callable[[str | None, str], str] | None = None` (a bound `(conversation_id, text) -> substituted_text`), stored as `self._chat_dictionary_applier`. Presence is the enable gate (mirrors `_skills_service`).
   - `async _apply_chat_dictionaries(provider_messages, session_id) -> list[dict]` (new): returns `provider_messages` unchanged when no applier; resolves `conversation_id = <active session>.persisted_conversation_id` (None → unchanged); finds the final `role == "user"` message; **skips** if that message's pre-substitution content is a str starting with `COMMAND_PREFIX` (a skill command — leave the skill mechanism untouched); otherwise substitutes each substitutable text via `await asyncio.to_thread(self._chat_dictionary_applier, conversation_id, text)` — **offloaded** so the synchronous DB read + regex matching never block the UI event loop (native sends are async workers on that loop):
     - **str content** → substitute the whole string;
     - **parts list** → substitute each `{"type": "text"}` part's `"text"` (image parts untouched);
     - writes the result into a **new** message dict (never mutates the stored message or the input dict) and returns a new list with only that message replaced. The message-shape logic (find final user message, command check, extract text, rebuild) is cheap and stays on the loop; only the applier call is offloaded.
   - a call `provider_messages = await self._apply_chat_dictionaries(provider_messages, session_id)` added at each of the 4 sites, immediately after the `if refuse is not None: return …` skill-substitution guard.

3. **`ChatScreen`** gains `_console_chat_dictionary_applier(conversation_id, text) -> str`: resolves `db = getattr(self.app_instance, "chachanotes_db", None)` **at call time**; returns `text` if `db is None` or `not conversation_id`; else `cdl.apply_active_chatdicts_to_text(db, conversation_id, None, text, max_tokens=_CHATDICT_MAX_TOKENS, strategy=_CHATDICT_STRATEGY)`. It is passed as `chat_dictionary_applier=self._console_chat_dictionary_applier` at the controller construction site. Constants `_CHATDICT_MAX_TOKENS = 500`, `_CHATDICT_STRATEGY = "sorted_evenly"` (legacy parity).

### Data flow (per send)

```
stored messages (raw)
  → _provider_messages_for_session / _through_message   (assembly)
  → _apply_skill_substitution                            (existing; may refuse → abort)
  → await _apply_chat_dictionaries(provider_messages, session_id)   (NEW, ephemeral, applier offloaded via asyncio.to_thread)
        · conv_id = session.persisted_conversation_id  (None → no-op)
        · final user message: skip if "/command", else substitute str / text-parts
  → _stream_assistant_response
        ├─ provider branch: stream_chat(resolution, provider_messages)
        └─ agent branch:    run_reply(agent_messages = provider_messages[1:])
```

The stored transcript is never touched; only the ephemeral payload for this turn changes.

### Ordering / interaction

- Dictionaries apply **after** skill substitution (the `/command` is already consumed/rendered), and are **skipped** entirely when the final user message is a skill command — so the two mechanisms never cross-contaminate.
- On retry/regenerate/continue, `provider_messages` are rebuilt from the raw stored messages and re-substituted with the **currently** attached dictionaries. This is intentional (apply what is currently "in play"), idempotent given unchanged attachments, and consistent with the inspector summary.
- On **continue** from an assistant message, `_ensure_user_continuation_instruction` (which runs before skill-substitution) appends a synthesized `CONSOLE_CONTINUE_INSTRUCTION` user message, so the "final user message" the transform targets is that synthesized instruction (there is no new user prose on a continue). This is the same message skill-substitution targets on continue, so the two transforms stay consistent; substituting into the ephemeral continuation instruction is acceptable (payload-only, never persisted). No special-casing of continue.

### Error handling

- `apply_active_chatdicts_to_text` and `_console_chat_dictionary_applier` never raise; any failure returns the raw text so a send always proceeds.
- `_apply_chat_dictionaries` catches any unexpected `Exception` (from the offloaded applier call or shape handling) and returns `provider_messages` unchanged — a dictionary problem must never break a send. It does **not** swallow `asyncio.CancelledError` (a Stop mid-send must still cancel the run). It never mutates stored messages or the input list/dicts; it builds fresh copies.
- Unsaved session (`persisted_conversation_id is None`), missing db, or no attached dicts → the payload is returned unchanged.

## Testing

- **Lib (`apply_active_chatdicts_to_text`), real DB:** a conversation with an attached dict whose pattern matches → substituted; no conversation / no attached dicts / conversation with a hostile (malformed) dict row → raw text returned; never raises.
- **Controller (`_apply_chat_dictionaries`), fake applier:** final user message substituted (str content); text part of a **parts-list** content substituted while image parts are untouched; a `"/command"` final message left unchanged; earlier (non-final) user messages untouched; no applier / no `persisted_conversation_id` → unchanged; the stored `ConsoleChatStore` messages are unchanged after the transform; the applier is invoked off the event loop (e.g. a fake applier that records its thread differs from the loop thread, or simply that a synchronous-blocking fake applier does not stall the run); parametrized so the transform is exercised for `submit`/`retry`/`continue`/`regenerate` structure.
- **Integration (load-bearing), real `ChatScreen` + real DB + real controller:** attach a conversation dict via the P1g seam, submit a native draft whose text matches the dict pattern, and assert — through a fake `provider_gateway.stream_chat` capturing the payload — that the model received the **substituted** text while the stored transcript kept the **raw** text. Include a second assertion (or a sibling test) that the **agent** branch's `agent_messages` also carries the substituted text (both branches derive from the same post-transform `provider_messages`).

## Task shape

One plan, ~4 tasks, no decomposition:
1. `apply_active_chatdicts_to_text` lib function + real-DB tests.
2. `ConsoleChatController` applier param + `_apply_chat_dictionaries` + 4 call sites + controller tests (fake applier).
3. `ChatScreen` applier method + constants + injection at the construction site + load-bearing integration test.
4. Full gate (Console + Character_Chat suites + import smoke) + flip spec status.

## Acceptance criteria

- [ ] A native Console send whose active session has an attached conversation dictionary sends the **substituted** text to the model (provider and agent branches), while the persisted transcript keeps the raw text.
- [ ] Substitution applies on all four send paths (submit, retry, continue, regenerate).
- [ ] Text part of an image+text (parts-list) user turn is substituted; image parts are untouched.
- [ ] A `/command` final user message is not dict-substituted.
- [ ] Unsaved session / no attached dicts / missing db / applier error → the send proceeds with unchanged text (never raises).
- [ ] Character dictionaries are not applied (deferred), and no per-send diagnostic surface is added.
- [ ] Full gate suite green; `import tldw_chatbook.app` OK.
