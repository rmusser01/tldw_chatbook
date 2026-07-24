# Native Console send-path chat-dictionary application — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Apply the conversation chat dictionaries the P1g Console inspector shows to the actual native Console send, so "shown = applied" holds on the surface users chat in.

**Architecture:** Mirror the existing skill-substitution transform in `ConsoleChatController`: a new `async _apply_chat_dictionaries(provider_messages, session_id)` runs right after `_apply_skill_substitution` at all four send sites, offloads the synchronous dictionary work to a thread, and rewrites only the ephemeral provider payload (the persisted transcript keeps the raw text). `ChatScreen` injects a db-bound applier callable; the substitution primitive lives in `Chat_Dictionary_Lib`.

**Tech Stack:** Python 3.11+, Textual, `Chat_Dictionary_Lib` (`collect_active_chatdict_entries` + `process_user_input`), `ConsoleChatController`, pytest.

**Spec:** `Docs/superpowers/specs/2026-07-16-roleplay-native-console-dict-apply-design.md` (committed `f08ea7c3`). Branch `claude/roleplay-native-console-dict-apply` off dev `7b227601`.

## Global Constraints

- **Conversation dictionaries only.** `char_data` is always `None` from the native caller (native sessions carry no character card). Character-dict application is deferred; keep the `char_data` param so the later cycle plugs in.
- **Ephemeral.** Only the provider payload for the turn is transformed; the persisted `ConsoleChatStore` transcript keeps the raw user text. Never mutate a stored message or an input dict/list — build fresh copies.
- **Silent.** No per-send diagnostics, no new user-facing enable/disable toggle. Presence of the injected applier is the only gate.
- **Never break a send.** The applier and the transform must not raise: any failure returns the text/payload unchanged. The transform must **not** swallow `asyncio.CancelledError` (Stop mid-send must still cancel).
- **Off the UI event loop.** Native sends run as async workers on the app's event loop (`run_worker(<coroutine>, …)`, `chat_screen.py:7983`). The synchronous DB read + regex matching must be offloaded via `asyncio.to_thread`.
- **Constants (legacy parity):** `max_tokens = 500`, `strategy = "sorted_evenly"`.
- **Roles/keys:** final `role == "user"` message only; `ConsoleMessageRole.USER.value == "user"`; `COMMAND_PREFIX == "/"`; message content is either a `str` or a parts list `[{"type": "text", "text": …}, <image parts>]`.
- **Test env** (prefix every pytest run):
  `HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest <targets> -q -p no:cacheprovider -o addopts="" --timeout=300 --timeout-method=thread`
- **Staging:** stage only each task's own files. Never `git add -A`; never stage anything under `.superpowers/`.

---

## File structure

- `tldw_chatbook/Character_Chat/Chat_Dictionary_Lib.py` — add `apply_active_chatdicts_to_text` (the never-raise substitution primitive). (Task 1)
- `tldw_chatbook/Chat/console_chat_controller.py` — add the applier constructor param + `_apply_chat_dictionaries` + 4 call-site additions. (Task 2)
- `tldw_chatbook/UI/Screens/chat_screen.py` — add the two constants, `_console_chat_dictionary_applier`, and the injection at the single controller construction site. (Task 3)
- `Tests/Character_Chat/test_apply_active_chatdicts_to_text.py` — new (Task 1)
- `Tests/Chat/test_console_dictionary_application.py` — new (Task 2)
- `Tests/UI/test_console_dictionary_send_integration.py` — new (Task 3)

---

## Task 1: `apply_active_chatdicts_to_text` substitution primitive

**Files:**
- Modify: `tldw_chatbook/Character_Chat/Chat_Dictionary_Lib.py` (add one module-level function; place it directly after `collect_active_chatdict_entries`)
- Test: `Tests/Character_Chat/test_apply_active_chatdicts_to_text.py` (create)

**Interfaces:**
- Consumes: `collect_active_chatdict_entries(db, conversation_id, char_data) -> List[ChatDictionary]`; `process_user_input(user_input: str, entries: List[ChatDictionary], max_tokens: int, strategy: str) -> str`; both already in this module.
- Produces: `apply_active_chatdicts_to_text(db, conversation_id, char_data, text, *, max_tokens=500, strategy="sorted_evenly") -> str` — never raises; returns `text` unchanged when `text` is not a str, no entries apply, or anything fails.

- [ ] **Step 1: Write the failing test.** Create `Tests/Character_Chat/test_apply_active_chatdicts_to_text.py`:

```python
import pytest

from tldw_chatbook.Character_Chat import Chat_Dictionary_Lib as cdl
from tldw_chatbook.Character_Chat.local_chat_dictionary_service import LocalChatDictionaryService
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB


@pytest.fixture
def db(tmp_path):
    database = CharactersRAGDB(tmp_path / "apply_chatdicts.db", "test-client")
    yield database
    database.close_connection()


def _attach_matching_dict(db, conv_id, name="Slang", key="Warden", content="grim jailer"):
    dict_id = cdl.save_chat_dictionary(
        db, name, entries=[cdl.ChatDictionary(key=key, content=content)]
    )
    assert dict_id is not None
    LocalChatDictionaryService(db).attach_to_conversation(dict_id, conv_id)
    return dict_id


def test_applies_attached_conversation_dictionary(db):
    conv_id = db.add_conversation({"title": "Attach"})
    _attach_matching_dict(db, conv_id)
    out = cdl.apply_active_chatdicts_to_text(
        db, conv_id, None, "The Warden nods.", max_tokens=500, strategy="sorted_evenly"
    )
    assert out == "The grim jailer nods."


def test_no_conversation_returns_text_unchanged(db):
    out = cdl.apply_active_chatdicts_to_text(
        db, None, None, "The Warden nods.", max_tokens=500, strategy="sorted_evenly"
    )
    assert out == "The Warden nods."


def test_conversation_without_dicts_returns_text_unchanged(db):
    conv_id = db.add_conversation({"title": "Empty"})
    out = cdl.apply_active_chatdicts_to_text(
        db, conv_id, None, "The Warden nods.", max_tokens=500, strategy="sorted_evenly"
    )
    assert out == "The Warden nods."


def test_non_string_text_returned_unchanged(db):
    conv_id = db.add_conversation({"title": "T"})
    _attach_matching_dict(db, conv_id)
    sentinel = ["not", "a", "string"]
    assert cdl.apply_active_chatdicts_to_text(db, conv_id, None, sentinel) is sentinel


def test_never_raises_when_collect_fails(db, monkeypatch):
    conv_id = db.add_conversation({"title": "T"})
    _attach_matching_dict(db, conv_id)

    def _boom(*a, **k):
        raise RuntimeError("collect exploded")

    monkeypatch.setattr(cdl, "collect_active_chatdict_entries", _boom)
    out = cdl.apply_active_chatdicts_to_text(db, conv_id, None, "The Warden nods.")
    assert out == "The Warden nods."
```

- [ ] **Step 2: Run — expect FAIL** (`AttributeError: module ... has no attribute 'apply_active_chatdicts_to_text'`).

Run: `... -m pytest Tests/Character_Chat/test_apply_active_chatdicts_to_text.py -q`

- [ ] **Step 3: Implement.** In `tldw_chatbook/Character_Chat/Chat_Dictionary_Lib.py`, directly after the `collect_active_chatdict_entries` function, add:

```python
def apply_active_chatdicts_to_text(
    db: "CharactersRAGDB",
    conversation_id: Optional[str],
    char_data: Optional[Dict[str, Any]],
    text: str,
    *,
    max_tokens: int = 500,
    strategy: str = "sorted_evenly",
) -> str:
    """Apply the active chat dictionaries to ``text`` for a send (never raises).

    Collects the dictionary union that applies to ``conversation_id`` (+
    ``char_data``, always ``None`` for the native Console today) and runs the
    standard ``process_user_input`` substitution. Returns ``text`` unchanged
    when it is not a string, no dictionaries apply, or anything fails -- a
    dictionary problem must never break a chat send.
    """
    if not isinstance(text, str):
        return text
    try:
        entries = collect_active_chatdict_entries(db, conversation_id, char_data)
        if not entries:
            return text
        return process_user_input(text, entries, max_tokens=max_tokens, strategy=strategy)
    except Exception:
        logger.opt(exception=True).warning(
            "apply_active_chatdicts_to_text failed; returning text unmodified."
        )
        return text
```

(This module already imports `from loguru import logger` at line 16, plus `Optional`, `Dict`, `Any`, and `CharactersRAGDB` under `TYPE_CHECKING` — no new imports needed.)

- [ ] **Step 4: Run — PASS.**

Run: `... -m pytest Tests/Character_Chat/test_apply_active_chatdicts_to_text.py -q`
Expected: 5 passed.

- [ ] **Step 5: Commit.**

```bash
git add tldw_chatbook/Character_Chat/Chat_Dictionary_Lib.py Tests/Character_Chat/test_apply_active_chatdicts_to_text.py
git commit -m "feat(chat-dictionaries): apply_active_chatdicts_to_text send-path primitive (never-raise)"
```

---

## Task 2: `ConsoleChatController._apply_chat_dictionaries` + call sites

**Files:**
- Modify: `tldw_chatbook/Chat/console_chat_controller.py` (constructor param + `self._chat_dictionary_applier`; new `_apply_chat_dictionaries`; 4 call-site additions at `submit_draft`, `retry_message`, `continue_from_message`, `regenerate_message`)
- Test: `Tests/Chat/test_console_dictionary_application.py` (create)

**Interfaces:**
- Consumes: injected `chat_dictionary_applier: Callable[[str | None, str], str] | None` (Task 3 supplies the real one); `self.store.sessions()` (each session has `.id` and `.persisted_conversation_id`); `ConsoleMessageRole.USER.value`, `COMMAND_PREFIX`, `asyncio` (all already imported).
- Produces: `async _apply_chat_dictionaries(provider_messages: list[dict], session_id: str) -> list[dict]`; a `chat_dictionary_applier=None` keyword-only constructor param.

- [ ] **Step 1: Write the failing tests.** Create `Tests/Chat/test_console_dictionary_application.py`:

```python
import threading

import pytest

from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole


class _Gateway:
    async def resolve_for_send(self, _selection):
        class _R:
            ready = True
            visible_copy = ""
        return _R()

    async def stream_chat(self, _resolution, _messages):
        if False:
            yield ""


def _controller(applier):
    store = ConsoleChatStore()
    controller = ConsoleChatController(
        store=store,
        provider_gateway=_Gateway(),
        provider="llama_cpp",
        model="test-model",
        chat_dictionary_applier=applier,
    )
    return controller, store


def _session_with_conv(store, conv_id="conv-1"):
    session = store.create_session(title="t")
    session.persisted_conversation_id = conv_id
    return session


def _warden(conv_id, text):
    return text.replace("Warden", "grim jailer")


@pytest.mark.asyncio
async def test_substitutes_final_user_string(warden_applier=_warden):
    controller, store = _controller(warden_applier)
    session = _session_with_conv(store)
    messages = [
        {"role": "system", "content": "sys"},
        {"role": ConsoleMessageRole.USER.value, "content": "The Warden nods."},
    ]
    out = await controller._apply_chat_dictionaries(messages, session.id)
    assert out[-1]["content"] == "The grim jailer nods."
    # Input list/dicts untouched (fresh copies only).
    assert messages[-1]["content"] == "The Warden nods."


@pytest.mark.asyncio
async def test_substitutes_text_part_of_parts_list_leaving_images():
    controller, store = _controller(_warden)
    session = _session_with_conv(store)
    image_part = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}
    messages = [
        {
            "role": ConsoleMessageRole.USER.value,
            "content": [{"type": "text", "text": "The Warden nods."}, image_part],
        }
    ]
    out = await controller._apply_chat_dictionaries(messages, session.id)
    parts = out[-1]["content"]
    assert parts[0] == {"type": "text", "text": "The grim jailer nods."}
    assert parts[1] is image_part


@pytest.mark.asyncio
async def test_skips_skill_command_message():
    controller, store = _controller(_warden)
    session = _session_with_conv(store)
    messages = [{"role": ConsoleMessageRole.USER.value, "content": "/Warden do a thing"}]
    out = await controller._apply_chat_dictionaries(messages, session.id)
    assert out[-1]["content"] == "/Warden do a thing"


@pytest.mark.asyncio
async def test_only_final_user_message_substituted():
    controller, store = _controller(_warden)
    session = _session_with_conv(store)
    messages = [
        {"role": ConsoleMessageRole.USER.value, "content": "The Warden earlier."},
        {"role": "assistant", "content": "ok"},
        {"role": ConsoleMessageRole.USER.value, "content": "The Warden now."},
    ]
    out = await controller._apply_chat_dictionaries(messages, session.id)
    assert out[0]["content"] == "The Warden earlier."
    assert out[-1]["content"] == "The grim jailer now."


@pytest.mark.asyncio
async def test_no_applier_returns_input_unchanged():
    controller, store = _controller(None)
    session = _session_with_conv(store)
    messages = [{"role": ConsoleMessageRole.USER.value, "content": "The Warden nods."}]
    out = await controller._apply_chat_dictionaries(messages, session.id)
    assert out[-1]["content"] == "The Warden nods."


@pytest.mark.asyncio
async def test_unsaved_session_returns_input_unchanged():
    controller, store = _controller(_warden)
    session = store.create_session(title="t")  # persisted_conversation_id stays None
    messages = [{"role": ConsoleMessageRole.USER.value, "content": "The Warden nods."}]
    out = await controller._apply_chat_dictionaries(messages, session.id)
    assert out[-1]["content"] == "The Warden nods."


@pytest.mark.asyncio
async def test_applier_runs_off_the_event_loop():
    loop_thread = threading.get_ident()
    seen = {}

    def _recording_applier(conv_id, text):
        seen["thread"] = threading.get_ident()
        return text.replace("Warden", "grim jailer")

    controller, store = _controller(_recording_applier)
    session = _session_with_conv(store)
    messages = [{"role": ConsoleMessageRole.USER.value, "content": "The Warden nods."}]
    out = await controller._apply_chat_dictionaries(messages, session.id)
    assert out[-1]["content"] == "The grim jailer nods."
    assert seen["thread"] != loop_thread  # offloaded via asyncio.to_thread


@pytest.mark.asyncio
async def test_applier_exception_returns_input_unchanged():
    def _boom(conv_id, text):
        raise RuntimeError("applier exploded")

    controller, store = _controller(_boom)
    session = _session_with_conv(store)
    messages = [{"role": ConsoleMessageRole.USER.value, "content": "The Warden nods."}]
    out = await controller._apply_chat_dictionaries(messages, session.id)
    assert out[-1]["content"] == "The Warden nods."
```

(`ConsoleMessageRole` lives in `tldw_chatbook.Chat.console_chat_models` — the same module the controller imports it from at `console_chat_controller.py:14-18`.)

- [ ] **Step 2: Run — expect FAIL.** The constructor rejects `chat_dictionary_applier` (TypeError) and/or `_apply_chat_dictionaries` is missing (AttributeError).

Run: `... -m pytest Tests/Chat/test_console_dictionary_application.py -q`

- [ ] **Step 3a: Add the constructor param.** In `console_chat_controller.py`, in `__init__`'s keyword-only block, add after `skill_substitution_enabled: bool = True,`:

```python
        chat_dictionary_applier: "Callable[[str | None, str], str] | None" = None,
```

and in the body, after `self._skill_substitution_enabled = skill_substitution_enabled` (near the other `self._…` assignments), add:

```python
        self._chat_dictionary_applier = chat_dictionary_applier
```

- [ ] **Step 3b: Add the transform.** Add this method to `ConsoleChatController` (place it directly after `_apply_skill_substitution`):

```python
    async def _apply_chat_dictionaries(
        self, provider_messages: list[dict[str, Any]], session_id: str
    ) -> list[dict[str, Any]]:
        """Apply the active conversation chat dictionaries to the final user
        message of the ephemeral provider payload (never the stored transcript).

        Mirrors `_apply_skill_substitution` (final `role == "user"` message
        only, one rule for fresh sends AND retry/continue/regenerate). The
        synchronous DB read + regex substitution are offloaded via
        `asyncio.to_thread` because native sends run as async workers on the UI
        event loop. Skill commands are left untouched. Any failure returns the
        payload unchanged so a dictionary problem can never break a send;
        `asyncio.CancelledError` is re-raised so a mid-send Stop still cancels.
        """
        applier = self._chat_dictionary_applier
        if applier is None:
            return provider_messages

        session = next((s for s in self.store.sessions() if s.id == session_id), None)
        conversation_id = session.persisted_conversation_id if session is not None else None
        if not conversation_id:
            return provider_messages

        final_index: int | None = None
        for index in range(len(provider_messages) - 1, -1, -1):
            if provider_messages[index].get("role") == ConsoleMessageRole.USER.value:
                final_index = index
                break
        if final_index is None:
            return provider_messages

        message = provider_messages[final_index]
        content = message.get("content")
        if isinstance(content, str) and content.startswith(COMMAND_PREFIX):
            return provider_messages

        try:
            if isinstance(content, str):
                new_content: Any = await asyncio.to_thread(applier, conversation_id, content)
                if new_content == content:
                    return provider_messages
            elif isinstance(content, list):
                new_parts: list[Any] = []
                changed = False
                for part in content:
                    if (
                        isinstance(part, dict)
                        and part.get("type") == "text"
                        and isinstance(part.get("text"), str)
                    ):
                        new_text = await asyncio.to_thread(applier, conversation_id, part["text"])
                        if new_text != part["text"]:
                            changed = True
                            new_parts.append({**part, "text": new_text})
                            continue
                    new_parts.append(part)
                if not changed:
                    return provider_messages
                new_content = new_parts
            else:
                return provider_messages
        except asyncio.CancelledError:
            raise
        except Exception:
            return provider_messages

        new_messages = list(provider_messages)
        new_messages[final_index] = {**message, "content": new_content}
        return new_messages
```

- [ ] **Step 3c: Add the 4 call sites.** In each send method, immediately after the existing `if refuse is not None: return self._block(...)` (or `return …`) that follows `_apply_skill_substitution`, add the transform. Use the local session variable that already exists at each site:

  - `submit_draft` (after the skill-sub refuse check, before `self._notify_submission_accepted()`):
    ```python
        provider_messages = await self._apply_chat_dictionaries(provider_messages, session.id)
    ```
  - `retry_message` (after `if refuse is not None: return self._block(session_id, refuse)`):
    ```python
        provider_messages = await self._apply_chat_dictionaries(provider_messages, session_id)
    ```
  - `continue_from_message` (after its skill-sub refuse check, before appending the assistant message):
    ```python
        provider_messages = await self._apply_chat_dictionaries(provider_messages, session_id)
    ```
  - `regenerate_message` (after its skill-sub refuse check):
    ```python
        provider_messages = await self._apply_chat_dictionaries(provider_messages, session_id)
    ```

- [ ] **Step 4: Run — PASS.**

Run: `... -m pytest Tests/Chat/test_console_dictionary_application.py -q`
Expected: 8 passed.

Then the controller regression:
`... -m pytest Tests/Chat/test_console_chat_controller.py Tests/Chat/test_console_skill_substitution.py Tests/Chat/test_console_variant_stream.py Tests/Chat/test_console_agent_swap.py -q`
Expected: all pass.

- [ ] **Step 5: Commit.**

```bash
git add tldw_chatbook/Chat/console_chat_controller.py Tests/Chat/test_console_dictionary_application.py
git commit -m "feat(console): apply conversation chat dictionaries to native send payload (all send paths)"
```

---

## Task 3: `ChatScreen` applier injection + load-bearing integration test

**Files:**
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py` (two module-level constants; `_console_chat_dictionary_applier` method; `chat_dictionary_applier=` kwarg at the construction site in `_ensure_console_chat_controller`, ~`:2393`)
- Test: `Tests/UI/test_console_dictionary_send_integration.py` (create)

**Interfaces:**
- Consumes: Task 1 `cdl.apply_active_chatdicts_to_text`; Task 2 `chat_dictionary_applier` constructor param; `self.app_instance.chachanotes_db`.
- Produces: `ChatScreen._console_chat_dictionary_applier(conversation_id, text) -> str` (a sync callable passed to the controller).

- [ ] **Step 1: Write the failing integration test.** Create `Tests/UI/test_console_dictionary_send_integration.py`:

```python
import pytest

from Tests.UI.test_destination_shells import _build_test_app, _wait_for_selector
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import ConsoleHarness
from tldw_chatbook.Character_Chat import Chat_Dictionary_Lib as cdl
from tldw_chatbook.Character_Chat.chat_dictionary_scope_service import ChatDictionaryScopeService
from tldw_chatbook.Character_Chat.local_chat_dictionary_service import LocalChatDictionaryService
from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB


@pytest.fixture
def dictionary_db(tmp_path):
    db = CharactersRAGDB(tmp_path / "console_dict_send.db", "test-client")
    yield db
    db.close_connection()


def _active_native_session(console):
    store = console._ensure_console_chat_store()
    return next(s for s in store.sessions() if s.id == store.active_session_id)


class _CapturingGateway:
    """Records the provider_messages the send would transmit; yields one chunk."""

    def __init__(self):
        self.captured = None

    async def resolve_for_send(self, _selection):
        class _R:
            ready = True
            visible_copy = ""
        return _R()

    async def stream_chat(self, _resolution, provider_messages):
        self.captured = [dict(m) for m in provider_messages]
        yield "ok"


def _final_user_content(messages):
    for message in reversed(messages):
        if message.get("role") == ConsoleMessageRole.USER.value:
            return message.get("content")
    return None


@pytest.mark.asyncio
async def test_native_send_applies_conversation_dictionary_provider_branch(dictionary_db):
    app = _build_test_app()
    app.chachanotes_db = dictionary_db
    app.chat_dictionary_scope_service = ChatDictionaryScopeService(
        local_service=LocalChatDictionaryService(dictionary_db), server_service=None
    )

    conv_id = dictionary_db.add_conversation({"title": "Send flow"})
    dict_id = cdl.save_chat_dictionary(
        dictionary_db, "Slang", entries=[cdl.ChatDictionary(key="Warden", content="grim jailer")]
    )
    LocalChatDictionaryService(dictionary_db).attach_to_conversation(dict_id, conv_id)

    async with ConsoleHarness(app).run_test(size=(180, 48)) as pilot:
        screen = pilot.app.screen_stack[-1]
        await _wait_for_selector(screen, pilot, "#console-native-composer")
        _active_native_session(screen).persisted_conversation_id = conv_id

        controller = screen._ensure_console_chat_controller()
        gateway = _CapturingGateway()
        controller.provider_gateway = gateway
        controller._agent_runtime_enabled = False  # force the provider branch

        result = await controller.submit_draft("The Warden nods.")
        assert result.accepted

        # Model received the SUBSTITUTED text...
        assert _final_user_content(gateway.captured) == "The grim jailer nods."
        # ...while the persisted transcript keeps the RAW text.
        store = screen._ensure_console_chat_store()
        stored = [m for m in store.messages_for_session(_active_native_session(screen).id)
                  if m.role is ConsoleMessageRole.USER]
        assert stored[-1].content == "The Warden nods."


@pytest.mark.asyncio
async def test_native_send_applies_conversation_dictionary_agent_branch(dictionary_db):
    app = _build_test_app()
    app.chachanotes_db = dictionary_db
    app.chat_dictionary_scope_service = ChatDictionaryScopeService(
        local_service=LocalChatDictionaryService(dictionary_db), server_service=None
    )

    conv_id = dictionary_db.add_conversation({"title": "Agent send"})
    dict_id = cdl.save_chat_dictionary(
        dictionary_db, "Slang", entries=[cdl.ChatDictionary(key="Warden", content="grim jailer")]
    )
    LocalChatDictionaryService(dictionary_db).attach_to_conversation(dict_id, conv_id)

    async with ConsoleHarness(app).run_test(size=(180, 48)) as pilot:
        screen = pilot.app.screen_stack[-1]
        await _wait_for_selector(screen, pilot, "#console-native-composer")
        _active_native_session(screen).persisted_conversation_id = conv_id

        controller = screen._ensure_console_chat_controller()
        gateway = _CapturingGateway()
        controller.provider_gateway = gateway

        captured = {}

        def _fake_run_reply(*, agent_messages, assistant_message_id, **kwargs):
            captured["agent_messages"] = [dict(m) for m in agent_messages]
            from tldw_chatbook.Chat.console_chat_controller import ConsoleSubmitResult
            store = screen._ensure_console_chat_store()
            store.append_stream_chunk(assistant_message_id, "ok")
            return ConsoleSubmitResult(True, False, "ok")

        class _Bridge:
            run_reply = staticmethod(_fake_run_reply)

        controller._agent_bridge = _Bridge()
        controller._agent_runtime_enabled = True

        result = await controller.submit_draft("The Warden nods.")
        assert result.accepted
        assert _final_user_content(captured["agent_messages"]) == "The grim jailer nods."
```

Note: if `_fake_run_reply`'s signature or `ConsoleSubmitResult` construction does not match the real `ConsoleAgentBridge.run_reply` contract, mirror the fake used in `Tests/Chat/test_console_agent_swap.py` (the canonical agent-bridge double) instead — the load-bearing assertion is only that `agent_messages` carries the substituted text.

- [ ] **Step 2: Run — expect FAIL.** With no applier injected, the model receives the raw text: `assert _final_user_content(...) == "The grim jailer nods."` fails (it equals `"The Warden nods."`).

Run: `... -m pytest Tests/UI/test_console_dictionary_send_integration.py -q`

- [ ] **Step 3a: Add constants.** Near the other module-level constants at the top of `chat_screen.py` (after imports), add:

```python
_CHATDICT_MAX_TOKENS = 500
_CHATDICT_STRATEGY = "sorted_evenly"
```

- [ ] **Step 3b: Add the applier method.** Add to `ChatScreen` (near `_dictionary_scope_service` / the other P1g dictionary helpers):

```python
    def _console_chat_dictionary_applier(self, conversation_id: str | None, text: str) -> str:
        """Bound applier handed to the native Console controller: apply the
        active CONVERSATION chat dictionaries to a send's text (never raises).

        Resolves the db lazily (at call time), so a controller built before the
        db is ready still works. Conversation-only: ``char_data`` is ``None``
        (native sessions carry no character card yet).
        """
        db = getattr(self.app_instance, "chachanotes_db", None)
        if db is None or not conversation_id or not isinstance(text, str):
            return text
        from ...Character_Chat import Chat_Dictionary_Lib as cdl
        return cdl.apply_active_chatdicts_to_text(
            db,
            conversation_id,
            None,
            text,
            max_tokens=_CHATDICT_MAX_TOKENS,
            strategy=_CHATDICT_STRATEGY,
        )
```

- [ ] **Step 3c: Inject at the construction site.** In `_ensure_console_chat_controller` (`chat_screen.py`, the `ConsoleChatController(...)` call ~`:2369`-`:2393`), add after `skills_service=getattr(self.app_instance, "skills_scope_service", None),`:

```python
                chat_dictionary_applier=self._console_chat_dictionary_applier,
```

- [ ] **Step 4: Run — PASS.**

Run: `... -m pytest Tests/UI/test_console_dictionary_send_integration.py -q`
Expected: 2 passed.

- [ ] **Step 5: Commit.**

```bash
git add tldw_chatbook/UI/Screens/chat_screen.py Tests/UI/test_console_dictionary_send_integration.py
git commit -m "feat(console): wire native send to apply conversation dictionaries (ChatScreen applier)"
```

---

## Task 4: Full gate + spec status

**Files:**
- Modify: `Docs/superpowers/specs/2026-07-16-roleplay-native-console-dict-apply-design.md` (status line)

- [ ] **Step 1: Full gate.**

```
HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
  Tests/Character_Chat/ Tests/Chat/test_chat_functions.py \
  Tests/Chat/test_console_chat_controller.py Tests/Chat/test_console_skill_substitution.py \
  Tests/Chat/test_console_variant_stream.py Tests/Chat/test_console_agent_swap.py \
  Tests/Chat/test_console_dictionary_application.py \
  Tests/Character_Chat/test_apply_active_chatdicts_to_text.py \
  Tests/UI/test_console_dictionary_send_integration.py Tests/UI/test_console_native_chat_flow.py \
  -q -p no:cacheprovider -o addopts="" --timeout=300 --timeout-method=thread
```

Expected: all pass (record exact counts). Then the import smoke:

```
HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -c "import tldw_chatbook.app; print('IMPORT OK')"
```

- [ ] **Step 2: Flip spec status** — `**Status:** Design approved (brainstorm), pending spec review.` → `**Status:** Implemented.`

- [ ] **Step 3: Commit.**

```bash
git add Docs/superpowers/specs/2026-07-16-roleplay-native-console-dict-apply-design.md
git commit -m "docs(roleplay): mark native Console dict-application spec implemented"
```

---

## Notes for the executor

- **Load-bearing tests** (do not accept a fake substitute): Task 1 real-DB substitution (attach a matching dict → text is substituted; no conv / no dicts / collect-raises → raw); Task 2 the off-loop assertion (`asyncio.to_thread`) and the ephemeral/no-mutation assertions; Task 3 the real-seam integration (a real `ChatScreen` + real DB + real controller, provider branch AND agent branch both receive the substituted text while the stored transcript keeps the raw text). These pin the two things the feature exists for: "shown = applied" on the native send, and "the transcript is never rewritten."
- **Do not block the UI loop.** The transform must offload the applier via `asyncio.to_thread`; it must never call the synchronous applier directly on the event loop. (Native sends are async workers on that loop — `chat_screen.py:7983`.)
- **Never break a send.** Both the lib primitive and the controller transform swallow ordinary exceptions and return the input unchanged; the controller transform must re-raise `asyncio.CancelledError`.
- **Ephemeral only.** Assert (Task 3) that the stored `ConsoleChatStore` user message keeps the raw text after a send — the transform must never touch stored messages or mutate its input list/dicts.
- **Scope.** Conversation dictionaries only (`char_data=None`); no character dicts, no diagnostics, no new toggle. Do not add config surfaces — use the `500` / `"sorted_evenly"` constants.
```
