"""TASK-34435: one collection per console turn on the prompt-transform seam.

Every console send needs the conversation's world books and chat-dictionary
entries twice conceptually -- once when the turn's frozen
``prompt_transform_inputs`` snapshot is captured, and once when the
dictionary/world-info appliers transform the final provider payload. The
turn must pay each *collection* exactly once: the appliers consume the
captured bundle instead of re-fetching it from the store.

The spies count the collection entrypoints that both sides share
(``_collect_active_world_books`` for books, ``collect_active_chatdict_entries``
for dictionaries) across one production-wired send (the runtime's
frozen-input appliers, not test stubs), plus the underlying world-book fetch
for a deeper count. A regression back to double collection shows 2x here.
"""

import functools
import json
from types import SimpleNamespace

import pytest

import tldw_chatbook.Character_Chat.Chat_Dictionary_Lib as cdl_module
import tldw_chatbook.Character_Chat.world_info_resolver as resolver_module
from tldw_chatbook.Character_Chat.Chat_Dictionary_Lib import (
    collect_active_chatdict_entries,
)
from tldw_chatbook.Character_Chat.local_chat_dictionary_service import (
    LocalChatDictionaryService,
)
from tldw_chatbook.Character_Chat.world_book_manager import WorldBookManager
from tldw_chatbook.Chat.console_chat_controller import (
    ConsoleChatController,
    capture_prompt_transform_inputs,
)
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Chat.console_dispatch_checkpoint import (
    ConsoleEgressClass,
    ConsoleResolvedDestination,
)
from tldw_chatbook.Chat.console_project_instructions import (
    ProjectInstructionControlState,
)
from tldw_chatbook.Chat.console_runtime import (
    _apply_chat_dictionaries_for_app,
    _apply_world_info_for_app,
)
from tldw_chatbook.Chat.console_session_settings import ConsoleSessionSettings
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

# The send path resolves provider/config snapshots in-body; keep the
# collection-time profile (see Tests/conftest.py) instead of the per-test
# sandbox, as the hook-admission send tests do.
pytestmark = pytest.mark.bootstrap_profile


class RecordingGateway:
    """Ready streaming gateway that records the final provider payload."""

    def __init__(self):
        self.messages_seen = None

    async def resolve_for_send(self, _selection):
        return SimpleNamespace(
            ready=True,
            provider="llama_cpp",
            model="test-model",
            base_url="http://127.0.0.1:9099",
            visible_copy="",
            resolved_destination=ConsoleResolvedDestination(
                provider="llama_cpp",
                model="test-model",
                endpoint_identity="http://127.0.0.1:9099",
                egress_class=ConsoleEgressClass.ON_DEVICE,
            ),
        )

    async def stream_chat(self, _resolution, messages, **_kwargs):
        self.messages_seen = messages
        yield "ok"


@pytest.fixture
def db(tmp_path):
    db = CharactersRAGDB(tmp_path / "single-collection.db", "test-client")
    yield db
    db.close_connection()


@pytest.fixture(autouse=True)
def _isolated_resolver_cache():
    """The ADR-221 resolver cache is module-level; keep tests hermetic."""
    resolver_module._clear_world_info_cache()
    yield
    resolver_module._clear_world_info_cache()


def _attach_world_book(db, conv_id):
    books = WorldBookManager(db)
    book_id = books.create_world_book("Lore")
    books.create_world_book_entry(
        book_id, keys=["dragon"], content="Dragons hoard turn snapshots."
    )
    books.associate_world_book_with_conversation(conv_id, book_id)


def _attach_dictionary(db, conv_id):
    service = LocalChatDictionaryService(db)
    dict_id = service.create_dictionary(
        {
            "name": "Abbrev",
            "entries": [{"pattern": "WI", "replacement": "world info"}],
        }
    )["id"]
    conv = db.get_conversation_by_id(conv_id)
    metadata = json.loads(conv.get("metadata") or "{}")
    metadata["active_dictionaries"] = [dict_id]
    db.update_conversation(
        conv_id, {"metadata": json.dumps(metadata)}, expected_version=conv["version"]
    )


def _armed_controller(db, gateway):
    """The production applier wiring ``ConsoleRuntime.ensure_chat_controller``
    binds (console_runtime.py): the runtime's frozen-input appliers over the
    app-owned db. The controller resolves its own turn snapshots
    (``resolve_runtime_turn_configuration_snapshot``), the viewless path."""
    app = SimpleNamespace(chachanotes_db=db, app_config={})
    from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService

    store = ConsoleChatStore(persistence=ChatPersistenceService(db))
    controller = ConsoleChatController(
        store=store,
        provider_gateway=gateway,
        chat_dictionary_applier=functools.partial(
            _apply_chat_dictionaries_for_app, app
        ),
        world_info_applier=functools.partial(_apply_world_info_for_app, app),
        # Hermetic hook authority: no review pending, no hooks configured.
        hook_permissions_accessor=lambda: SimpleNamespace(
            snapshot=lambda: SimpleNamespace(blocked_reason=None)
        ),
    )
    # The runtime assigns the app to the controller in set_chat_controller
    # (console_runtime.py: ``value.app = self._app``); mirror that wiring.
    controller.app = app
    # No workspace scope: a workspace-scoped durable turn would require the
    # workspace registry, which this hermetic rig does not own.
    session = store.ensure_session()
    session.project_instruction_state = ProjectInstructionControlState.legacy_disabled()
    if session.settings is None:
        session.settings = ConsoleSessionSettings(provider="llama_cpp")
    return controller, session


@pytest.mark.asyncio
async def test_one_console_send_collects_world_books_and_dictionaries_once(
    db, monkeypatch
):
    controller, session = _armed_controller(db, RecordingGateway())
    gateway = controller.provider_gateway

    # Turn 1 creates the durable conversation (and its Library policy row);
    # only the COUNTED turn 2 carries the world book and dictionary.
    first = await controller.submit_draft("hello")
    assert first.accepted, first.visible_copy
    conv_id = session.persisted_conversation_id
    assert conv_id, "turn 1 must have persisted the conversation"
    _attach_world_book(db, conv_id)
    _attach_dictionary(db, conv_id)

    counts = {"books": 0, "book_fetch": 0, "dicts": 0, "captures": 0}

    original_collect_books = resolver_module._collect_active_world_books
    original_collect_dicts = collect_active_chatdict_entries
    original_book_fetch = WorldBookManager.get_world_books_for_conversation
    original_capture = capture_prompt_transform_inputs

    def spy_collect_books(*args, **kwargs):
        counts["books"] += 1
        return original_collect_books(*args, **kwargs)

    def spy_collect_dicts(*args, **kwargs):
        counts["dicts"] += 1
        return original_collect_dicts(*args, **kwargs)

    def spy_book_fetch(*args, **kwargs):
        counts["book_fetch"] += 1
        return original_book_fetch(*args, **kwargs)

    def spy_capture(*args, **kwargs):
        counts["captures"] += 1
        return original_capture(*args, **kwargs)

    monkeypatch.setattr(
        resolver_module, "_collect_active_world_books", spy_collect_books
    )
    monkeypatch.setattr(
        cdl_module, "collect_active_chatdict_entries", spy_collect_dicts
    )
    monkeypatch.setattr(
        WorldBookManager, "get_world_books_for_conversation", spy_book_fetch
    )
    monkeypatch.setattr(
        "tldw_chatbook.Chat.console_chat_controller.capture_prompt_transform_inputs",
        spy_capture,
    )

    result = await controller.submit_draft("WI: a dragon appears")
    assert result.accepted, result.visible_copy

    # Behavior parity: the dictionary substitution and world-info injection
    # both landed in the ephemeral provider payload.
    final_user = gateway.messages_seen[-1]
    assert final_user["role"] == "user"
    assert "world info" in final_user["content"]
    assert "Dragons hoard turn snapshots." in final_user["content"]
    assert "WI: a dragon appears" not in final_user["content"]

    # One send, one collection per side (TASK-34435).
    assert counts["dicts"] == 1, f"dictionary collection ran {counts['dicts']}x"
    assert counts["books"] == 1, f"world-book collection ran {counts['books']}x"
    assert counts["book_fetch"] == 1, (
        f"underlying world-book fetch ran {counts['book_fetch']}x"
    )
    assert counts["captures"] >= 1


@pytest.mark.asyncio
async def test_mid_turn_book_edit_applies_the_captured_snapshot(db):
    """Per-turn consistency: the apply side never re-reads the store.

    The turn's frozen ``prompt_transform_inputs`` is the authoritative
    snapshot. If a world book is edited between the capture and the apply
    (here: inside the applier call itself), the injected content must come
    from the CAPTURED entry, not the edited row -- the same per-turn
    consistency the pre-frozen-inputs behavior had.
    """
    controller, session = _armed_controller(db, RecordingGateway())
    gateway = controller.provider_gateway
    first = await controller.submit_draft("hello")
    assert first.accepted, first.visible_copy
    conv_id = session.persisted_conversation_id
    _attach_world_book(db, conv_id)

    real_applier = controller._world_info_applier
    edited = {"done": False}

    def editing_applier(conv, text, history, *rest):
        # Mutate the stored entry AFTER the capture, BEFORE the apply.
        if not edited["done"]:
            edited["done"] = True
            WorldBookManager(db).update_world_book_entry(1, content="EDITED MID-TURN")
        return real_applier(conv, text, history, *rest)

    controller._world_info_applier = editing_applier

    result = await controller.submit_draft("a dragon appears")
    assert result.accepted, result.visible_copy

    final_user = gateway.messages_seen[-1]
    assert "Dragons hoard turn snapshots." in final_user["content"]
    assert "EDITED MID-TURN" not in final_user["content"]


def test_console_screen_level_applier_copies_stay_removed():
    """TASK-34667 absence pin: the screen-level dictionary/world-info
    applier copies must not come back.

    ``ChatScreen._console_chat_dictionary_applier`` and
    ``ChatScreen._console_world_info_applier`` were passed into
    ``ConsoleRuntime.ensure_chat_controller``, which unconditionally
    overwrote both via its runtime-owned ``kwargs.update`` -- dead on every
    path, and already drifted: the screen world-info copy called
    ``world_info_resolver.apply_world_info_to_message`` with a 3-argument
    signature the live call site (which passes ``frozen_inputs`` as a
    fourth argument) would have rejected outright. Dead always-overwritten
    copies are the ``_library_provider_for_app`` precedent recorded in
    console_runtime.py: a stale copy silently shadowed the live
    binding and cost 24 Library tools behind one swallowed warning. The
    runtime-owned ``_apply_*_for_app`` seams this file pins above are the
    only live applier wiring; a screen-level copy invites editing the
    wrong seam with zero effect.
    """
    from tldw_chatbook.UI.Screens import chat_screen as chat_screen_module

    assert not hasattr(
        chat_screen_module.ChatScreen, "_console_chat_dictionary_applier"
    )
    assert not hasattr(chat_screen_module.ChatScreen, "_console_world_info_applier")
    # The bounds constants existed only to feed the dead dictionary copy.
    assert not hasattr(chat_screen_module, "_CHATDICT_MAX_TOKENS")
    assert not hasattr(chat_screen_module, "_CHATDICT_STRATEGY")
