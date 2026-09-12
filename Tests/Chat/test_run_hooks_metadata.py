"""Hook configuration and machine-origin transcript persistence contracts."""

import tomllib

import pytest

from tldw_chatbook import config
from tldw_chatbook.Agents.run_hooks import HookSpec, load_hooks_config
from tldw_chatbook.Chat.chat_conversation_service import ChatConversationService
from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Chat.console_conversation_hydration import (
    console_messages_from_conversation_tree,
)
from tldw_chatbook.Chat.message_metadata import MESSAGE_ORIGIN_HOOK, MessageMetadata
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB


def test_default_config_template_configures_no_hook_commands():
    parsed = tomllib.loads(config.CONFIG_TOML_CONTENT)

    assert parsed["hooks"]["enabled"] is True
    assert load_hooks_config(parsed).hooks == ()


def test_real_config_loader_preserves_hooks_across_setting_save(tmp_path, monkeypatch):
    path = tmp_path / "config.toml"
    path.write_text(
        "[hooks]\nenabled = false\n"
        '[[hooks.hook]]\nevent = "PreToolUse"\nmatcher = "fs_*"\n'
        'command = ["/custom/guard", "argument with spaces", "$literal"]\n'
        "timeout_s = 2.5\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(path))
    expected = HookSpec(
        "PreToolUse",
        ("/custom/guard", "argument with spaces", "$literal"),
        matcher="fs_*",
        timeout_s=2.5,
    )

    loaded = load_hooks_config(config.load_settings(force_reload=True))
    assert loaded.enabled is False
    assert loaded.hooks == (expected,)

    assert config.save_setting_to_cli_config("hooks", "enabled", True)
    reloaded = load_hooks_config(config.load_settings(force_reload=True))
    assert reloaded.enabled is True
    assert reloaded.hooks == (expected,)
    assert tomllib.loads(path.read_text(encoding="utf-8"))["hooks"]["hook"][0][
        "command"
    ] == list(expected.command)


@pytest.mark.integration
@pytest.mark.parametrize(
    "content", ["Additional context", "Send blocked by hook: guard refused"]
)
def test_hook_origin_system_row_survives_database_rehydration(tmp_path, content):
    db = CharactersRAGDB(tmp_path / "hooks.sqlite", "run-hooks-test")
    try:
        persistence = ChatPersistenceService(db)
        store = ConsoleChatStore(persistence=persistence)
        session = store.create_session(title="Hook metadata")
        message = store.append_message(
            session.id,
            role=ConsoleMessageRole.SYSTEM,
            content=content,
            persist=True,
            metadata=MessageMetadata(origin=MESSAGE_ORIGIN_HOOK),
        )
        assert message.persisted_message_id is not None
        row = db.get_message_by_id(message.persisted_message_id)
        assert MessageMetadata.from_json(row["metadata_json"]).origin == "hook"

        tree = ChatConversationService(db).get_conversation_tree(
            session.persisted_conversation_id, root_limit=100, depth_cap=100
        )
        nodes = console_messages_from_conversation_tree(tree, db=db)
        restored = ConsoleChatStore(persistence=ChatPersistenceService(db))
        restored_session = restored.restore_persisted_session(
            title="Restored hook metadata",
            workspace_id=None,
            persisted_conversation_id=session.persisted_conversation_id,
            all_nodes=nodes,
            active_leaf_persisted_id=message.persisted_message_id,
        )
        messages = restored.messages_for_session(restored_session.id)
        assert len(messages) == 1
        assert messages[0].content == content
        assert messages[0].role is ConsoleMessageRole.SYSTEM
        assert messages[0].metadata.origin == MESSAGE_ORIGIN_HOOK
    finally:
        db.close_connection()
