# Start Chat → Real Character Conversation (TASK-427) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the Roleplay workbench's "Start Chat" open a real character conversation in the native Console — greeting as the first assistant message, character identity persisted, card system prompt applied on send, and the plain provider path (not the agent harness).

**Architecture:** Wire character identity through the existing native-Console session/persistence machinery (no schema change — `conversations.character_id` already exists). A new character branch in the handoff consumer fetches the card, builds the effective system prompt like the Personas preview, creates a dedicated global-workspace session, seeds the greeting as a persisted assistant message, and directly syncs the Console UI. The greeting is excluded from provider payloads (display-only, like the preview) so strict providers accept the first send. The plain-provider gate keys on `character_id`.

**Tech Stack:** Python 3.11+, Textual, pytest / pytest-asyncio, SQLite (ChaChaNotes_DB).

## Global Constraints

- **Run tests via the repo venv, from the worktree root:** `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest ...` (system python lacks deps).
- **No ChaChaNotes migration.** Schema stays v22; `conversations.character_id` already exists.
- **Design source of truth:** `Docs/superpowers/specs/2026-07-21-start-chat-character-conversation-design.md` (rev 2).
- **Commit message trailer:** end each commit body with `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.
- **`character_id` is the single "this is a character session" signal** — it drives persistence, the plain-provider gate, and restart behavior. No separate agent-override field.
- Preview parity: effective system prompt = newline-join of non-empty `[system_prompt, personality, description, scenario]` card fields, fallback `"Stay in character."`; `replace_placeholders` applies ONLY to the greeting (`first_message`), with user name `"User"`.

---

## Task 1: Character identity on the session + character-aware persist

**Files:**
- Modify: `tldw_chatbook/Chat/console_chat_store.py` (dataclass `ConsoleChatSession` ~148-160; `persist_session_if_needed` ~948-984)
- Test: `Tests/Chat/test_console_chat_store.py`

**Interfaces:**
- Produces: `ConsoleChatSession.character_id: int | None`, `ConsoleChatSession.character_name: str | None`. When `character_id` is set on a session, `persist_session_if_needed` calls `create_conversation(character_id=..., character_name=..., assistant_kind="character", assistant_id=str(character_id), ...)`.

- [ ] **Step 1: Write the failing test**

Add to `Tests/Chat/test_console_chat_store.py` (near the existing `test_persist_session_if_needed_passes_system_prompt`). `FakePersistence.create_conversation(**kwargs)` already records calls in `self.created` / returns an id — inspect its recorded kwargs (match the existing fixture's attribute; the existing system-prompt test shows how it captures kwargs).

```python
def test_persist_session_if_needed_passes_character_identity(self):
    store, fake = self._store_with_fake_persistence()  # mirror the existing helper
    session = store.create_session(title="Chat with Elara")
    session.character_id = 7
    session.character_name = "Elara"
    conv_id = store.persist_session_if_needed(session.id)
    assert conv_id is not None
    kwargs = fake.last_create_kwargs  # same accessor the system-prompt test uses
    assert kwargs["character_id"] == 7
    assert kwargs["character_name"] == "Elara"
    assert kwargs["assistant_kind"] == "character"
    assert kwargs["assistant_id"] == "7"

def test_persist_session_if_needed_non_character_stays_generic(self):
    store, fake = self._store_with_fake_persistence()
    session = store.create_session(title="Chat 1")
    store.persist_session_if_needed(session.id)
    kwargs = fake.last_create_kwargs
    assert kwargs["assistant_kind"] == "generic"
    assert kwargs["assistant_id"] == "console"
    assert kwargs.get("character_id") is None
```

If `FakePersistence` does not already expose `last_create_kwargs`, add it: in its `create_conversation(self, **kwargs)` set `self.last_create_kwargs = kwargs` before returning the id. If the existing helper to build a store+fake has a different name, reuse it verbatim (grep `def _store_with_fake` / how `test_persist_session_if_needed_passes_system_prompt` builds its store).

- [ ] **Step 2: Run test to verify it fails**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Chat/test_console_chat_store.py -k character_identity -v`
Expected: FAIL — `ConsoleChatSession` has no `character_id` attribute (AttributeError) or the kwargs assertions fail.

- [ ] **Step 3: Add the dataclass fields**

In `console_chat_store.py`, in `ConsoleChatSession` (after `one_shot_prefill: str | None = None`):

```python
    one_shot_prefill: str | None = None
    #: When set, this is a character-bound session: it persists with the
    #: character's id, forces the plain-provider path, and restores as a
    #: character session (task-427). ``None`` = a normal Console session.
    character_id: int | None = None
    character_name: str | None = None
```

- [ ] **Step 4: Pass identity through `persist_session_if_needed`**

In `persist_session_if_needed`, replace the `create_conversation(...)` call (currently hardcoding `assistant_kind="generic", assistant_id="console"`) with identity-aware kwargs:

```python
        scope_type, persisted_workspace_id = self._persistence_scope(session)
        if session.character_id is not None:
            identity_kwargs = {
                "assistant_kind": "character",
                "assistant_id": str(session.character_id),
                "character_id": session.character_id,
                "character_name": session.character_name,
            }
        else:
            identity_kwargs = {
                "assistant_kind": "generic",
                "assistant_id": "console",
            }
        session.persisted_conversation_id = self.persistence.create_conversation(
            conversation_title=session.title,
            workspace_id=persisted_workspace_id,
            scope_type=scope_type,
            system_prompt=session.settings.system_prompt
            if session.settings is not None
            else None,
            **identity_kwargs,
        )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Chat/test_console_chat_store.py -v`
Expected: PASS (new tests + all existing store tests green).

- [ ] **Step 6: Commit**

```bash
git add tldw_chatbook/Chat/console_chat_store.py Tests/Chat/test_console_chat_store.py
git commit -m "feat(console): character identity on session + character-aware persist (task-427)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Task 2: Greeting is display-only to the provider + regenerate/continue guard

**Files:**
- Modify: `tldw_chatbook/Chat/console_chat_controller.py` (`_provider_message_payloads` ~2645; `regenerate_message` ~1161; `continue_from_message` ~1105)
- Test: `Tests/Chat/test_console_chat_controller.py`

**Interfaces:**
- Produces: `_provider_message_payloads` drops assistant messages that precede the first payload-eligible user message. `regenerate_message`/`continue_from_message` return a blocked `ConsoleSubmitResult` when the resulting payload has no user turn.

- [ ] **Step 1: Write the failing tests**

Add to `Tests/Chat/test_console_chat_controller.py` (reuse `CapturingGateway` and the existing controller-build helper — grep how `test_submit_draft_prepends_system_prompt_message` builds its controller + store and drives a send). The greeting is a persisted ASSISTANT message seeded before any user turn.

```python
async def test_leading_greeting_excluded_from_provider_payload(self):
    controller, store, gateway = self._build_controller(CapturingGateway())
    session = store.create_session(title="Chat with Elara")
    controller.switch_session(session.id)
    store.append_message(session.id, role=ConsoleMessageRole.ASSISTANT,
                         content="Greetings, traveler.", persist=False)
    store.set_session_draft(session.id, "Hi")
    await controller.submit_draft(session.id)
    sent = gateway.captured_messages  # list[dict] of the outbound payload
    roles = [m["role"] for m in sent]
    # No leading assistant: first non-system role is user.
    assert "assistant" not in roles[: roles.index("user")] if "user" in roles else True
    assert roles[0] in ("system", "user")
    # The greeting text is not in the outbound payload.
    assert all("Greetings, traveler." not in (m.get("content") or "") for m in sent)

async def test_regenerate_on_leading_greeting_is_blocked(self):
    controller, store, gateway = self._build_controller(CapturingGateway())
    session = store.create_session(title="Chat with Elara")
    controller.switch_session(session.id)
    greeting = store.append_message(session.id, role=ConsoleMessageRole.ASSISTANT,
                                    content="Greetings.", persist=False)
    result = await controller.regenerate_message(greeting.id)
    assert result.ok is False
    assert not gateway.captured_messages  # nothing sent
```

Match the real `ConsoleSubmitResult` success attribute name (grep its definition — it may be `.ok`, `.success`, or positional). Match `CapturingGateway`'s captured-messages attribute name exactly.

- [ ] **Step 2: Run tests to verify they fail**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Chat/test_console_chat_controller.py -k "leading_greeting or regenerate_on_leading" -v`
Expected: FAIL — greeting text appears in the payload / regenerate is not blocked.

- [ ] **Step 3: Drop leading assistant messages in the payload builder**

In `_provider_message_payloads`, in the `for message in session_messages:` loop (~2678), add a leading-assistant skip. Introduce `seen_user = False` before the loop and skip pre-user assistants:

```python
        payloads: list[dict[str, Any]] = []
        seen_user = False
        for message in session_messages:
            if message.role not in {
                ConsoleMessageRole.USER,
                ConsoleMessageRole.ASSISTANT,
            }:
                continue
            if skip_failed and message.status == "failed":
                continue
            # A seeded character greeting is a display-only assistant turn:
            # keep it out of the provider payload so strict providers (Anthropic,
            # Gemini) never see an assistant-first message array (task-427).
            if not seen_user and message.role is ConsoleMessageRole.ASSISTANT:
                continue
            if message.role is ConsoleMessageRole.USER:
                seen_user = True
```
(The existing body — text extraction / image budgeting / payload append — stays after this, unchanged.)

- [ ] **Step 4: Add the no-user-turn guard to regenerate and continue**

Add a helper near `_ensure_user_continuation_instruction`:

```python
    @staticmethod
    def _has_user_turn(provider_messages: list[dict[str, Any]]) -> bool:
        return any(
            m.get("role") == ConsoleMessageRole.USER.value for m in provider_messages
        )
```

In `regenerate_message`, right after `self._ensure_user_continuation_instruction(provider_messages)` (~1196):

```python
        self._ensure_user_continuation_instruction(provider_messages)
        if not self._has_user_turn(provider_messages):
            return self._block(
                session_id,
                "Nothing to regenerate before the character's opening line.",
            )
```

In `continue_from_message`, right after its `self._ensure_user_continuation_instruction(provider_messages)` (~1137):

```python
        self._ensure_user_continuation_instruction(provider_messages)
        if not self._has_user_turn(provider_messages):
            return self._block(
                session_id,
                "Nothing to continue before the character's opening line.",
            )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Chat/test_console_chat_controller.py -v`
Expected: PASS (new tests + all existing controller tests green — the leading-drop must be a no-op for normal user-first sessions).

- [ ] **Step 6: Commit**

```bash
git add tldw_chatbook/Chat/console_chat_controller.py Tests/Chat/test_console_chat_controller.py
git commit -m "feat(console): greeting display-only in provider payload + regen/continue guard (task-427)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Task 3: Plain-provider gate keyed on the message-owning character session

**Files:**
- Modify: `tldw_chatbook/Chat/console_chat_controller.py` (`_stream_assistant_response` gate ~2051)
- Test: `Tests/Chat/test_console_chat_controller.py`

**Interfaces:**
- Consumes: `ConsoleChatSession.character_id` (Task 1).
- Produces: character sessions (`character_id` set) always take the plain-provider branch, even when the global agent runtime is enabled and an agent bridge is present.

- [ ] **Step 1: Write the failing test**

Add to `Tests/Chat/test_console_chat_controller.py`. Build the controller **with** an agent bridge and global agent-runtime enabled, then assert a character session does NOT invoke the bridge. Define a minimal spy bridge:

```python
class _SpyAgentBridge:
    def __init__(self):
        self.calls = 0
    async def run_reply(self, *args, **kwargs):
        self.calls += 1
        raise AssertionError("agent bridge should not be called for a character session")

async def test_character_session_forces_plain_provider(self):
    bridge = _SpyAgentBridge()
    controller, store, gateway = self._build_controller(
        CapturingGateway(), agent_bridge=bridge, agent_runtime_enabled=True
    )
    session = store.create_session(title="Chat with Elara")
    session.character_id = 7
    controller.switch_session(session.id)
    store.set_session_draft(session.id, "Hi")
    result = await controller.submit_draft(session.id)
    assert bridge.calls == 0
    assert gateway.captured_messages  # plain provider path ran

async def test_normal_session_still_uses_agent_when_enabled(self):
    bridge = _RecordingAgentBridge()  # returns a benign reply, records the call
    controller, store, gateway = self._build_controller(
        CapturingGateway(), agent_bridge=bridge, agent_runtime_enabled=True
    )
    session = store.create_session(title="Chat 1")
    controller.switch_session(session.id)
    store.set_session_draft(session.id, "Hi")
    await controller.submit_draft(session.id)
    assert bridge.calls == 1
```

If `self._build_controller` does not accept `agent_bridge`/`agent_runtime_enabled`, extend it to pass them into the controller constructor / the `set_agent_runtime(enabled, bridge)` setter (grep the controller's constructor + `console_chat_controller.py:570` for how the owner wires them). Define `_RecordingAgentBridge` with an async `run_reply` that returns whatever the real `_run_agent_reply` expects the bridge to return (grep `_run_agent_reply` ~2240 for the outcome shape) — or, if that is heavy, assert the split differently: keep `_SpyAgentBridge` for the character case and, for the normal case, assert that with `character_id=None` the gate condition `self._agent_runtime_enabled and self._agent_bridge is not None and not prefill` is True by unit-testing a small extracted predicate (see Step 3).

- [ ] **Step 2: Run test to verify it fails**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Chat/test_console_chat_controller.py -k "character_session_forces_plain" -v`
Expected: FAIL — the agent bridge IS called (AssertionError from the spy) because the gate ignores `character_id`.

- [ ] **Step 3: Key the gate on the message-owning session**

In `_stream_assistant_response`, replace the gate (~2051):

```python
        owner_id = self.store.session_id_for_message(assistant_message_id)
        owner = next(
            (s for s in self.store.sessions() if s.id == owner_id), None
        )
        force_plain = owner is not None and owner.character_id is not None
        if (
            self._agent_runtime_enabled
            and self._agent_bridge is not None
            and not prefill
            and not force_plain
        ):
            return await self._run_agent_reply(
                resolution=resolution,
                provider_messages=provider_messages,
                assistant_message_id=assistant_message_id,
                prepare_retry=prepare_retry,
                variant_mode=variant_mode,
            )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Chat/test_console_chat_controller.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Chat/console_chat_controller.py Tests/Chat/test_console_chat_controller.py
git commit -m "feat(console): plain-provider gate for character sessions (task-427)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Task 4: Native character branch in the handoff consumer

**Files:**
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py` (`_consume_pending_chat_handoff` ~8741; add helper `_start_character_console_session`)
- Test: `Tests/UI/test_chat_first_handoffs.py`

**Interfaces:**
- Consumes: Task 1 (`session.character_id`/`character_name`), Task 2 (greeting exclusion), Task 3 (plain gate). Existing: `_character_session_identity_from_handoff` (`chat_screen.py:442`); `store.create_session`, `store.append_message`; `self._sync_native_console_chat_ui`, `self._focus_console_composer_if_needed`.
- Produces: a dedicated character-bound Console session with the greeting seeded, on the native handoff branch, with a guarded fallback to `_stage_handoff_as_console_live_work`.

- [ ] **Step 1: Write the failing test**

Add to `Tests/UI/test_chat_first_handoffs.py` (reuse its `ChatScreen` harness + a fake character DB that returns a card for `get_character_card_by_id`; mirror `stub_characters` from `Tests/UI/test_personas_character_attach.py`). Assert on the native branch (no legacy tab container):

```python
async def test_native_start_chat_builds_character_session_with_greeting(self, ...):
    # app_instance.chachanotes_db.get_character_card_by_id(7) -> card dict with
    # name/first_message/system_prompt/personality/description/scenario.
    screen = ...  # mounted native Console ChatScreen (no legacy tab container)
    screen.app_instance.pending_chat_handoff = _character_start_chat_payload(
        character_id=7, name="Elara",
        first_message="Greetings, {{user}}.",
    )
    await screen._consume_pending_chat_handoff()
    store = screen._ensure_console_chat_store()
    session = store._sessions[store.active_session_id]
    assert session.character_id == 7
    assert session.character_name == "Elara"
    assert session.title == "Chat with Elara"
    assert session.workspace_id == CONSOLE_GLOBAL_WORKSPACE_ID
    msgs = store.messages_for_session(session.id)
    assert msgs[0].role is ConsoleMessageRole.ASSISTANT
    assert msgs[0].content == "Greetings, User."   # {{user}} substituted
    assert "Stay in character." not in (session.settings.system_prompt or "")  # real prompt used
    assert screen.app_instance.pending_chat_handoff is None

async def test_native_start_chat_falls_back_when_card_fetch_fails(self, ...):
    screen = ...
    screen.app_instance.chachanotes_db.get_character_card_by_id = \
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("boom"))
    screen.app_instance.pending_chat_handoff = _character_start_chat_payload(
        character_id=7, name="Elara", first_message="Hi")
    await screen._consume_pending_chat_handoff()  # must not raise
    # Fell back to staged context; no character session created.
    store = screen._ensure_console_chat_store()
    active = store._sessions.get(store.active_session_id) if store.active_session_id else None
    assert active is None or active.character_id is None
    assert screen.app_instance.pending_chat_handoff is None
```

`_character_start_chat_payload(...)` builds a dict shaped like a `ChatHandoffPayload` with `metadata={"intent": "start_chat", "selected_kind": "character", "selected_record_id": str(character_id), "selected_name": name}` and `source="personas"`. Model it on the payloads used by the existing `test_chat_screen_start_chat_handoff_binds_character_session_identity` (`:236`).

- [ ] **Step 2: Run test to verify it fails**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_chat_first_handoffs.py -k native_start_chat -v`
Expected: FAIL — the native branch stages context instead of creating a character session (no `character_id`, no greeting).

- [ ] **Step 3: Add the character-session builder helper**

In `chat_screen.py`, add imports if missing near the top-of-file imports: `import asyncio` (check it is already imported) and `from ...Character_Chat.Character_Chat_Lib import replace_placeholders` (grep — add only if absent). Add the method to `ChatScreen`:

```python
    async def _start_character_console_session(self, payload) -> bool:
        """Build a dedicated character conversation from a Start-Chat handoff.

        Returns True when a character session was created; False to let the
        caller fall back to the staged-context path (task-427).
        """
        identity = _character_session_identity_from_handoff(payload)
        if identity is None:
            return False
        character_id, _name_hint, _assistant_id = identity
        db = getattr(self.app_instance, "chachanotes_db", None)
        if db is None:
            return False
        try:
            card = await asyncio.to_thread(db.get_character_card_by_id, character_id)
        except Exception:
            logger.opt(exception=True).warning(
                "Start Chat: character card fetch failed; staging context instead."
            )
            return False
        if not card:
            return False

        name = str(card.get("name") or _name_hint or "").strip() or "Character"
        parts = [
            str(card.get(key) or "").strip()
            for key in ("system_prompt", "personality", "description", "scenario")
        ]
        system_prompt = "\n".join(p for p in parts if p) or "Stay in character."
        greeting = replace_placeholders(str(card.get("first_message") or ""), name, "User")

        store = self._ensure_console_chat_store()
        settings = replace(
            self._default_console_session_settings(),
            system_prompt=system_prompt,
            character_label=name,
        )
        session = store.create_session(
            title=f"Chat with {name}",
            workspace_id=CONSOLE_GLOBAL_WORKSPACE_ID,
            settings=settings,
        )
        session.character_id = int(character_id)
        session.character_name = name
        if greeting:
            try:
                store.append_message(
                    session.id,
                    role=ConsoleMessageRole.ASSISTANT,
                    content=greeting,
                    persist=True,
                )
            except Exception:
                logger.opt(exception=True).warning(
                    "Start Chat: greeting seed/persist failed; continuing."
                )
        await self._sync_native_console_chat_ui()
        self._focus_console_composer_if_needed(force=True)
        return True
```

Verify `replace` (from `dataclasses`) and `_default_console_session_settings` exist in this module (grep — both are used elsewhere in `chat_screen.py`). `ConsoleMessageRole` and `CONSOLE_GLOBAL_WORKSPACE_ID` are already imported (`chat_screen.py:96` and workspace const import).

- [ ] **Step 4: Call the helper from the native branch**

In `_consume_pending_chat_handoff`, inside the `if tab_container is None:` block (~8756), try the character path first:

```python
            tab_container = self._get_tab_container()
            if tab_container is None:
                if await self._start_character_console_session(payload):
                    self.app_instance.pending_chat_handoff = None
                    return
                # Not a character Start-Chat (or fetch failed): stage context.
                self._stage_handoff_as_console_live_work(payload)
                self.app_instance.pending_chat_handoff = None
                return
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_chat_first_handoffs.py -v`
Expected: PASS (new native-branch tests + existing legacy-path handoff tests green).

- [ ] **Step 6: Commit**

```bash
git add tldw_chatbook/UI/Screens/chat_screen.py Tests/UI/test_chat_first_handoffs.py
git commit -m "feat(console): Start Chat builds a real character session on the native handoff branch (task-427)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Task 5: Restore character identity on resume

**Files:**
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py` (`_resume_console_workspace_conversation` ~3466-3472)
- Test: `Tests/UI/test_chat_first_handoffs.py` or the existing resume-path test module (grep for `_resume_console_workspace_conversation` in `Tests/`)

**Interfaces:**
- Consumes: the persisted `conversations.character_id` (round-trips via `normalize_conversation_row`).
- Produces: a resumed session with `session.character_id` set, so its sends stay on the plain path (Task 3) after restart.

- [ ] **Step 1: Write the failing test**

Add a test that resumes a conversation whose row carries `character_id=7` and asserts the restored session is character-bound. Reuse the resume-path harness (grep an existing `_resume_console_workspace_conversation` test; if none, drive it via a fake `conversation` dict that includes `"character_id": 7`).

```python
async def test_resume_restores_character_identity(self, ...):
    screen = ...
    # conversation row / tree fake with character_id=7 and title "Chat with Elara"
    await screen._resume_console_workspace_conversation(target=..., conversation={..., "character_id": 7, "title": "Chat with Elara"}, tree=...)
    store = screen._ensure_console_chat_store()
    session = store._sessions[store.active_session_id]
    assert session.character_id == 7
```

Match the real signature/args of `_resume_console_workspace_conversation` (read ~3387-3410 for its parameters; adapt the call).

- [ ] **Step 2: Run test to verify it fails**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_chat_first_handoffs.py -k resume_restores_character -v`
Expected: FAIL — `session.character_id` is `None`.

- [ ] **Step 3: Set `character_id` on the restored session**

In `_resume_console_workspace_conversation`, right after `session = store.restore_persisted_session(...)` (~3472):

```python
        raw_character_id = conversation.get("character_id")
        if raw_character_id is not None:
            try:
                session.character_id = int(raw_character_id)
            except (TypeError, ValueError):
                session.character_id = None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_chat_first_handoffs.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/UI/Screens/chat_screen.py Tests/UI/test_chat_first_handoffs.py
git commit -m "feat(console): restore character identity on resume so sends stay plain (task-427)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Task 6: Full-suite regression + live verification

**Files:** none (verification only).

- [ ] **Step 1: Run the affected suites**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Chat/test_console_chat_store.py Tests/Chat/test_console_chat_controller.py Tests/UI/test_chat_first_handoffs.py -q`
Expected: all green.

- [ ] **Step 2: Live-verify in the real TUI**

Using the `verify` recipe: scratch `TLDW_CONFIG_PATH` profile with a reachable provider (local OpenAI-compat mock on :9099 if no real server), import a character card, click inspector **Start Chat**. Confirm:
- the greeting renders as the first assistant message in the Console transcript;
- the session/tab is titled "Chat with {name}";
- sending a message returns an **in-character** reply (not the agent sub-agent "please provide the character details" dead-end);
- quitting and relaunching the app shows the conversation still titled/identified with the character, and a further send still replies in character on the plain path.

- [ ] **Step 3: Mark ACs + notes on the task**

```bash
backlog task edit 427 --check-ac 1 --check-ac 2 --check-ac 3 --check-ac 4
backlog task edit 427 --notes "<implementation summary>"
```

---

## Self-Review

- **Spec coverage:** §1 identity→Task 1; §2 persist→Task 1; §3 native branch→Task 4; §4 UI sync for first-send system prompt→Task 4 (`_sync_native_console_chat_ui`); §5 plain gate→Task 3; §6 restart→Task 5; §7 greeting display-only + regen/continue guard→Task 2. All covered.
- **Placeholder scan:** test bodies note where to match existing fixture accessor names (`last_create_kwargs`, `captured_messages`, `ConsoleSubmitResult.ok`) — these are "match the real name" instructions, not TBDs; the executing agent greps the one true name. No functional placeholders.
- **Type consistency:** `character_id: int|None` and `character_name: str|None` are used identically across Tasks 1/3/4/5; `_start_character_console_session(payload) -> bool` and the `force_plain` gate reference only fields defined in Task 1.
- **Ordering:** 1 (fields+persist) → 2 (payload) → 3 (gate) → 4 (integration) → 5 (restart) → 6 (verify). Task 4 depends on 1/2/3; Task 5 depends on 1.
