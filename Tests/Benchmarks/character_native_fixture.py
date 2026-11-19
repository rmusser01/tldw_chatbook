"""Small native walkthrough fixture; preparation is not native qualification."""

from __future__ import annotations

import asyncio
import json
import platform
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from Tests.Benchmarks.character_qualification_fixture import (
    FIXTURE_VERSION,
    _copy_checkpointed,
    file_digest,
    owned_descriptors,
    verify_disposable_source,
)


def build_native(root: Path) -> dict[str, Any]:
    """Prepare ordinary, unavailable and empty cards without using real data."""
    if (root / ".qualification-owned").read_text() != FIXTURE_VERSION:
        raise ValueError("Unreserved native fixture root")
    path = root / "native.sqlite"
    if path.exists() or path.is_symlink():
        raise FileExistsError("Never overwrite a native fixture")
    from tldw_chatbook.Character_Chat.character_conversation_navigation import (
        CharacterConversationNavigationService,
    )
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

    receipt: dict[str, Any] = {
        "status": "failed",
        "fixture_version": FIXTURE_VERSION,
        "cards": {},
        "limitations": "Synthetic preparation only; no native keyboard, Windows, participant or latency result.",
    }
    database = None
    try:
        database = CharactersRAGDB(path, client_id="native-qualification")
        authority = database.get_local_authority_id()
        epoch = datetime(2026, 1, 1, tzinfo=UTC)
        ordinal = 0
        for name in (
            "Amber",
            "Indigo",
            "Cedar",
            "Copper",
            "Unavailable",
            "Empty Atlas",
        ):
            card_id = database.add_character_card(
                {
                    "name": f"SYNTHETIC QA {name}",
                    "description": "Offline disposable card for navigation qualification.",
                }
            )
            receipt["cards"][name] = card_id
            count = 0 if name == "Empty Atlas" else 2 if name == "Unavailable" else 7
            for number in range(1, count + 1):
                conversation_id = f"native-{name.lower()}-{number:02d}"
                stamp = (epoch + timedelta(minutes=ordinal)).isoformat()
                with database.transaction(immediate=True) as connection:
                    database.add_conversation(
                        {
                            "id": conversation_id,
                            "character_id": card_id,
                            "assistant_kind": "character",
                            "assistant_id": str(card_id),
                            "assistant_authority_id": authority,
                            "title": f"QA {name} beacon {number:02d}",
                        }
                    )
                    database.add_message(
                        {
                            "id": f"{conversation_id}-user",
                            "conversation_id": conversation_id,
                            "sender": "user",
                            "role": "user",
                            "content": f"NATIVE_MARKER_{name.upper()}_{number:02d}",
                            "timestamp": stamp,
                        }
                    )
                    assistant_id = f"{conversation_id}-assistant"
                    database.add_message(
                        {
                            "id": assistant_id,
                            "parent_message_id": f"{conversation_id}-user",
                            "conversation_id": conversation_id,
                            "sender": "assistant",
                            "role": "assistant",
                            "content": f"Synthetic reply for {conversation_id}. No provider was called.",
                            "timestamp": (
                                epoch + timedelta(minutes=ordinal, seconds=1)
                            ).isoformat(),
                        }
                    )
                    database.set_conversation_active_leaf(conversation_id, assistant_id)
                    connection.execute(
                        "UPDATE conversations SET created_at = ?, last_modified = ? WHERE id = ?",
                        (stamp, stamp, conversation_id),
                    )
                    ordinal += 1
            if name == "Unavailable":
                database.soft_delete_character_card(card_id, expected_version=1)
        service = CharacterConversationNavigationService(database)
        if service.ensure_keyword_index().value != "ready":
            raise RuntimeError("Native Keyword generation is not ready")
        connection = database.get_connection()
        receipt["counts"] = [
            connection.execute(query).fetchone()[0]
            for query in (
                "SELECT COUNT(*) FROM conversations",
                "SELECT COUNT(*) FROM messages",
                "SELECT COUNT(*) FROM character_conversation_search_documents",
            )
        ]
        if (
            receipt["counts"] != [30, 60, 28]
            or connection.execute("PRAGMA quick_check").fetchone()[0] != "ok"
        ):
            raise RuntimeError("Native fixture integrity/counts differ")
        receipt["status"] = "prepared-not-qualified"
        receipt["initial_open_conversations"] = ["native-amber-07", "native-indigo-07"]
    except BaseException as error:
        receipt["exception_type"] = type(error).__name__
        raise
    finally:
        if database is not None:
            database.close()
            receipt["registered_handles_after_cleanup"] = (
                database.registered_connection_count()
            )
        if path.exists():
            receipt["corpus_digest"] = file_digest(path)
        (root / "native-receipt.json").write_text(
            json.dumps(receipt, sort_keys=True, indent=2)
        )
    return receipt


def launch_native(
    root: Path, corpus: Path, receipt_path: Path, *, expected_head: str
) -> None:
    """Run in a manually opened terminal; preload two real exact saved chats."""
    verify_disposable_source(corpus, receipt_path)
    receipt = json.loads(receipt_path.read_text())
    if (
        receipt.get("status") != "prepared-not-qualified"
        or receipt.get("head") != expected_head
        or receipt.get("fixture_version") != FIXTURE_VERSION
        or receipt.get("corpus_digest") != file_digest(corpus)
    ):
        raise ValueError(
            "Native launch requires the frozen verified preparation receipt"
        )
    from tldw_chatbook import config
    from tldw_chatbook.app import TldwCli
    from tldw_chatbook.UI.Screens.chat_screen import ChatScreen

    destination = Path(config.get_chachanotes_db_path()).resolve()
    if not destination.is_relative_to(root):
        raise ValueError("Mutable application database escaped the disposable profile")
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    _copy_checkpointed(corpus, destination)

    class SyntheticNativeApp(TldwCli):
        def on_mount(self) -> None:
            super().on_mount()
            self.run_worker(
                self.prepare_tabs, name="synthetic-native-preparation", exclusive=True
            )

        async def prepare_tabs(self) -> None:
            async with asyncio.timeout(60):
                while not (
                    getattr(self, "_ui_ready", False)
                    and isinstance(self.screen, ChatScreen)
                ):
                    await asyncio.sleep(0.01)
                chat = self.screen
                for conversation_id in receipt["initial_open_conversations"]:
                    if not await chat._workspace._resume_console_workspace_conversation(
                        conversation_id
                    ):
                        raise RuntimeError("Synthetic initial tab failed to open")
                store = chat._ensure_console_chat_store()
                open_ids = [
                    session.persisted_conversation_id
                    for session in store.sessions()
                    if session.persisted_conversation_id
                ]
                if sorted(open_ids) != sorted(receipt["initial_open_conversations"]):
                    raise RuntimeError(
                        "Synthetic initial tabs do not match exact identities"
                    )
                current = next(
                    session.persisted_conversation_id
                    for session in store.sessions()
                    if session.id == store.active_session_id
                )
                (root / "native-startup.json").write_text(
                    json.dumps(
                        {
                            "status": "prepared-not-qualified",
                            "head": expected_head,
                            "source_corpus_digest": receipt["corpus_digest"],
                            "open_conversations": open_ids,
                            "current_conversation": current,
                            "host": platform.platform(),
                            "cells": list(self.size),
                        },
                        indent=2,
                    )
                )

    app = SyntheticNativeApp()
    try:
        app.run()
    finally:
        (root / "native-return.json").write_text(
            json.dumps(
                {
                    "status": "returned-not-qualified",
                    "head": expected_head,
                    "return_code": app.return_code,
                    "source_unchanged": file_digest(corpus) == receipt["corpus_digest"],
                    "owned_database_descriptors": owned_descriptors(root),
                    "limitations": "App.run return is not proof of native input, normal quit, retirement, Windows or participant success.",
                },
                indent=2,
            )
        )
