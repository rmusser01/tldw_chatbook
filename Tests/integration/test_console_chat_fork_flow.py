"""Provider-free production journey for Console chat forks (TASK-23088)."""

from __future__ import annotations

import json
from html import unescape
from dataclasses import replace
from io import BytesIO
from pathlib import Path

import pytest
from loguru import logger
from PIL import Image as PILImage
from textual.css.query import NoMatches
from textual.pilot import OutOfBounds
from textual.widgets import Button, Input

from Tests.Chat.test_citation_trace_repository import (
    TEST_FINGERPRINT_CODEC,
    _identity,
    _sealed_write,
)
from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_destination_shells import _wait_for_selector
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
)
from tldw_chatbook.Chat.chat_conversation_service import ChatConversationService
from tldw_chatbook.Chat.citation_provenance_runtime import (
    CitationProvenanceRuntimePolicy,
)
from tldw_chatbook.Chat.citation_trace_repository import CitationTraceRepository
from tldw_chatbook.Chat.console_chat_controller import (
    ConsoleRunState,
    ConsoleRunStatus,
    fingerprint_canonical_locator,
    resolve_project_instruction_binding,
)
from tldw_chatbook.Chat.console_chat_models import (
    ConsoleCitationNoticeCode,
    ConsoleCitationPhase,
    ConsoleCitationPresentation,
    ConsoleDispatchRecoveryKind,
    ConsoleDispatchRecoveryState,
    ConsoleMessageRole,
    GenerationVariantMeta,
    MessageAttachment,
)
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Chat.console_context_policy import (
    ConsoleContextPolicyOverrides,
    ContextCompactionMode,
)
from tldw_chatbook.Chat.console_conversation_hydration import (
    apply_resume_settings_overrides,
    console_messages_from_conversation_tree,
)
from tldw_chatbook.Chat.console_library_policy import (
    ConsoleAssistantLibraryAccess,
    ConsoleAutoRetrieve,
    ConsoleLibraryPolicyCandidate,
)
from tldw_chatbook.Chat.console_project_instructions import (
    ProjectInstructionControlState,
)
from tldw_chatbook.Chat.console_speech_preferences import ConsoleSpeechPreferences
from tldw_chatbook.Chat.console_session_settings import ConsoleSessionSettings
from tldw_chatbook.Chat.rag_scope import RagScope, ScopeItem
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.Video_Generation.video_metadata import VideoGenerationMetadata
from tldw_chatbook.Widgets.Console.console_fork_chat_modal import ConsoleForkChatModal
from tldw_chatbook.Widgets.Console.console_transcript import ConsoleTranscript


async def _settle(pilot, predicate, *, label: str, attempts: int = 400) -> None:
    for _ in range(attempts):
        if predicate():
            await pilot.pause()
            return
        await pilot.pause(0.01)
    raise AssertionError(f"Timed out waiting for {label}")


async def _click_visible(console, pilot, selector: str) -> None:
    """Deliver one real hit-tested click after production layout settles."""

    for _ in range(40):
        try:
            widget = console.query_one(selector)
            widget.scroll_visible(animate=False, force=True)
            await pilot.pause()
            await pilot.click(selector)
            return
        except (NoMatches, OutOfBounds):
            await pilot.pause(0.05)
    raise AssertionError(f"Could not click visible selector: {selector}")


async def _press_session_tab(console, pilot, session_id: str) -> None:
    """Route a session-tab press through the mounted production handler."""

    tab = console.query_one(f"#console-session-tab-{session_id}", Button)
    tab.press()
    await pilot.pause()


def _png_bytes(color: tuple[int, int, int]) -> bytes:
    output = BytesIO()
    PILImage.new("RGB", (2, 2), color).save(output, format="PNG")
    return output.getvalue()


def _source_rows(db: CharactersRAGDB, conversation_id: str) -> tuple[object, ...]:
    messages = tuple(
        dict(row) for row in db.get_messages_for_conversation(conversation_id, limit=50)
    )
    message_ids = tuple(str(row["id"]) for row in messages)
    return (
        dict(db.get_conversation_by_id(conversation_id)),
        messages,
        db.get_attachments_for_messages(message_ids),
        db.get_generation_metadata_for_messages(message_ids),
        db.get_conversation_console_project_context(conversation_id),
        db.get_conversation_active_leaf(conversation_id),
    )


def _source_live(store: ConsoleChatStore, session_id: str) -> tuple[object, ...]:
    session = next(item for item in store.sessions() if item.id == session_id)
    return (
        session.title,
        session.workspace_id,
        session.persisted_conversation_id,
        session.settings,
        session.context_policy_overrides,
        session.project_instruction_state,
        tuple(store.active_path_message_ids(session_id)),
        tuple(store.messages_for_session(session_id)),
        store.payload_revision(session_id),
    )


def _conversation_count(db: CharactersRAGDB) -> int:
    return int(
        db.get_connection()
        .execute(
            "SELECT COUNT(*) FROM conversations WHERE deleted = 0 AND client_id = ?",
            (db.client_id,),
        )
        .fetchone()[0]
    )


def _restore_conversation(
    store: ConsoleChatStore,
    db: CharactersRAGDB,
    conversation_id: str,
) -> tuple[object, tuple[object, ...]]:
    row = db.get_conversation_by_id(conversation_id)
    tree = ChatConversationService(db).get_conversation_tree(conversation_id)
    messages = tuple(console_messages_from_conversation_tree(tree, db=db))
    settings = apply_resume_settings_overrides(
        ConsoleSessionSettings(provider="openai", model="gpt-fork-fixture"),
        row,
    )
    session = store.restore_persisted_session(
        title=row["title"],
        workspace_id=row["workspace_id"],
        persisted_conversation_id=conversation_id,
        all_nodes=messages,
        active_leaf_persisted_id=db.get_conversation_active_leaf(conversation_id),
        settings=settings,
        activate=False,
    )
    return session, messages


@pytest.mark.asyncio
async def test_console_chat_fork_complete_provider_free_journey(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fork durable and temporary chats through the mounted production path."""

    isolated_profile = tmp_path / "isolated-profile"
    config_home = isolated_profile / "config"
    data_home = isolated_profile / "data"
    cache_home = isolated_profile / "cache"
    for path in (isolated_profile, config_home, data_home, cache_home):
        path.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HOME", str(isolated_profile))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(config_home))
    monkeypatch.setenv("XDG_DATA_HOME", str(data_home))
    monkeypatch.setenv("XDG_CACHE_HOME", str(cache_home))
    db_path = data_home / "chatbook.sqlite"
    project_root = tmp_path / "bound-project"
    project_root.mkdir()
    instruction_body = "PROJECT_INSTRUCTION_BODY_MUST_NOT_LEAK"
    (project_root / "AGENTS.md").write_text(instruction_body, encoding="utf-8")
    attachment_sentinel = b"ATTACHMENT_BYTES_MUST_NOT_LEAK"
    attachment_bytes = _png_bytes((32, 64, 96)) + attachment_sentinel
    approval_round = "source-approval-round"
    permission_decision = "approve_session_SECRET_DECISION"
    recovery_body = "Source-only recovery"
    run_body = "Source run active"
    provider_secret = "sk-provider-secret-must-not-leak"

    db = CharactersRAGDB(db_path, client_id="fork-flow")
    citation_repository = CitationTraceRepository(
        db,
        policy=CitationProvenanceRuntimePolicy(canonical_writes_enabled=True),
        identity_context=_identity(db),
        fingerprint_codec=TEST_FINGERPRINT_CODEC,
    )
    app = _build_test_app(config_overrides={"API": {"openai_api_key": provider_secret}})
    app.chachanotes_db = db
    app.citation_trace_repository = citation_repository
    workspace_id = "workspace-fork-flow"
    app.workspace_registry_service.create_workspace(
        workspace_id=workspace_id,
        name="Fork flow workspace",
        description="Isolated integration fixture",
    )
    binding = app.workspace_registry_service.add_folder_binding(
        workspace_id,
        project_root,
        allow_write=True,
    )
    project_state = ProjectInstructionControlState(
        project_instructions_enabled=True,
        working_folder_binding_id=binding.binding_id,
        working_folder_locator_fingerprint=fingerprint_canonical_locator(
            str(project_root.resolve())
        ),
        project_instruction_notice_key="source-notice-must-clear",
    )

    durable_ids: list[str] = []
    temporary_promoted_id: str | None = None
    log_lines: list[str] = []
    sink = logger.add(lambda record: log_lines.append(str(record)), level="DEBUG")
    host = ConsoleHarness(app)

    try:
        async with host.run_test(size=(120, 35)) as pilot:
            console = host.screen_stack[-1]
            await _wait_for_selector(console, pilot, "#console-native-transcript")
            store = console._ensure_console_chat_store()
            settings = replace(
                console._session._default_console_session_settings(),
                provider="openai",
                model="gpt-fork-fixture",
                system_prompt="Cite sources exactly.",
                pinned_prefill="source-only prefill",
            )
            source = store.create_session(
                title="Research source",
                workspace_id=workspace_id,
                settings=settings,
                runtime_backend="local",
                assistant_kind="generic",
                assistant_id="console",
                project_instruction_state=project_state,
            )
            source.user_display_name_override = "Rowan"
            source.speech_preferences = ConsoleSpeechPreferences(auto_speak=True)
            source.context_policy_overrides = ConsoleContextPolicyOverrides(
                compaction_mode=ContextCompactionMode.OFF
            )
            source.rag_scope_holder.set(
                RagScope(
                    items=(ScopeItem("note", "fork-evidence"),),
                    updated_at="2026-08-27T00:00:00Z",
                )
            )
            store.stage_session_library_policy(
                source.id,
                ConsoleLibraryPolicyCandidate(
                    auto_retrieve=ConsoleAutoRetrieve.NEVER,
                    assistant_access=ConsoleAssistantLibraryAccess.BLOCKED,
                ),
            )
            store.append_message(
                source.id,
                role=ConsoleMessageRole.USER,
                content="Question with an attachment",
                attachments=(
                    MessageAttachment(
                        attachment_bytes,
                        "image/png",
                        "evidence.png",
                        0,
                    ),
                ),
                persist=True,
            )
            cited_answer = store.append_message(
                source.id,
                role=ConsoleMessageRole.ASSISTANT,
                content="Answer [S1].",
                persist=True,
            )
            user_boundary = store.append_message(
                source.id,
                role=ConsoleMessageRole.USER,
                content="Middle user boundary",
                persist=True,
            )
            image_boundary = store.append_generation_message(
                source.id,
                content="[image] selected diagram",
                variants=(
                    (
                        _png_bytes((255, 0, 0)),
                        "image/png",
                        GenerationVariantMeta(
                            prompt="selected diagram",
                            negative_prompt="",
                            backend="openai",
                            model="image-fixture",
                            seed=7,
                            style=None,
                            params={"size": "small"},
                        ),
                    ),
                ),
                persist=True,
            )
            later_user = store.append_message(
                source.id,
                role=ConsoleMessageRole.USER,
                content="Later turn excluded from both middle forks",
                persist=True,
            )
            later_video = store.append_video_message(
                source.id,
                video_metadata=VideoGenerationMetadata(
                    name="source-video-store-key",
                    prompt="animate selected diagram",
                    backend="minimax",
                    model="video-fixture",
                    source_image_message_id=image_boundary.id,
                ),
                persist=True,
            )
            off_path = store.create_sibling(
                image_boundary.id,
                role=ConsoleMessageRole.ASSISTANT,
                content="Off-path sibling",
                persist=True,
            )
            store.set_active_leaf(source.id, later_video.id)
            source_conversation_id = source.persisted_conversation_id
            assert source_conversation_id is not None

            cited_row = db.get_message_by_id(cited_answer.persisted_message_id)
            prepared = citation_repository.prepare_write(_sealed_write())
            with db.transaction() as cursor:
                citation_repository.write_prepared(
                    cursor,
                    prepared,
                    message_id=cited_answer.persisted_message_id,
                    message_revision=cited_row["version"],
                    message_body=cited_answer.content,
                )
            store.set_citation_presentation(
                cited_answer.id,
                ConsoleCitationPresentation(
                    phase=ConsoleCitationPhase.SELECTED,
                    notice_code=ConsoleCitationNoticeCode.REPAIRED,
                ),
            )

            controller = console._ensure_console_chat_controller()
            controller.add_pending_round(source.id, approval_round)
            store._dispatch_recoveries_by_session[source.id] = (  # noqa: SLF001
                ConsoleDispatchRecoveryState(
                    kind=ConsoleDispatchRecoveryKind.QUARANTINED,
                    assistant_message_id=later_video.id,
                    conversation_id=source_conversation_id,
                    visible_copy=recovery_body,
                    actions=(),
                )
            )
            source.scratch_authority = "source-only-scratch"  # type: ignore[attr-defined]
            source.mcp_grants = {permission_decision}  # type: ignore[attr-defined]
            source.local_tool_grants = {permission_decision}  # type: ignore[attr-defined]
            runtime = console._console_runtime()
            source_scratch = runtime.scratch_spaces.snapshot(source.id)
            with runtime.scratch_spaces.lease(source_scratch) as leased:
                assert leased == source_scratch.root

            await console._sync_native_console_chat_ui()
            source_db_before = _source_rows(db, source_conversation_id)
            source_live_before = _source_live(store, source.id)

            # Pointer-select the USER boundary, then exercise the selected-row
            # keyboard action and edit the focused default title.
            await _click_visible(console, pilot, f"#console-message-{user_boundary.id}")
            transcript = console.query_one(
                "#console-native-transcript", ConsoleTranscript
            )
            assert transcript.selected_message_id == user_boundary.id
            await pilot.press("f")
            await _settle(
                pilot,
                lambda: isinstance(host.screen_stack[-1], ConsoleForkChatModal),
                label="USER fork modal",
            )
            user_modal = host.screen_stack[-1]
            assert isinstance(user_modal, ConsoleForkChatModal)
            painted_wide = unescape(host.export_screenshot(simplify=True)).replace(
                "\N{NO-BREAK SPACE}", " "
            )
            assert "Fork chat" in painted_wide
            user_request = console._session._active_fork_request
            assert user_request is not None
            user_fork_session_id = user_request.fork_session_id
            user_fork_conversation_id = user_request.fork_conversation_id
            assert user_fork_conversation_id is not None
            user_modal.query_one(
                "#console-fork-chat-title", Input
            ).value = "Focused user-boundary fork"
            controller._set_run_state(  # noqa: SLF001
                ConsoleRunState(ConsoleRunStatus.STREAMING, run_body),
                session_id=source.id,
            )
            await pilot.press("enter")
            await _settle(
                pilot,
                lambda: (
                    host.screen_stack[-1] is console
                    and store.active_session_id == user_fork_session_id
                ),
                label="renamed USER fork activation",
            )
            durable_ids.append(user_fork_conversation_id)

            user_fork = next(
                s for s in store.sessions() if s.id == user_fork_session_id
            )
            fork_scratch = runtime.scratch_spaces.snapshot(user_fork.id)
            assert fork_scratch.root != source_scratch.root
            assert fork_scratch.token != source_scratch.token
            assert fork_scratch.identity != source_scratch.identity
            with runtime.scratch_spaces.lease(fork_scratch) as leased:
                assert leased == fork_scratch.root
            assert controller.has_pending_approval_round(source.id)
            assert not controller.has_pending_approval_round(user_fork.id)
            assert (
                controller.run_state_for(source.id).status is ConsoleRunStatus.STREAMING
            )
            assert (
                controller.run_state_for(user_fork.id).status is ConsoleRunStatus.IDLE
            )
            assert source.id in store._dispatch_recoveries_by_session  # noqa: SLF001
            assert user_fork.id not in store._dispatch_recoveries_by_session  # noqa: SLF001
            assert not hasattr(user_fork, "scratch_authority")
            assert not hasattr(user_fork, "mcp_grants")
            assert not hasattr(user_fork, "local_tool_grants")
            assert (
                resolve_project_instruction_binding(
                    user_fork, app.workspace_registry_service
                ).root
                == project_root.resolve()
            )

            # Switch back through the real session tab, then use the pointer
            # Fork button on the selected ASSISTANT generated-image boundary.
            await _press_session_tab(console, pilot, source.id)
            await _settle(
                pilot,
                lambda: store.active_session_id == source.id,
                label="source tab activation",
            )
            await _click_visible(
                console, pilot, f"#console-message-{image_boundary.id}"
            )
            assert transcript.selected_message_id == image_boundary.id
            await _wait_for_selector(
                console,
                pilot,
                f"#console-message-action-fork-{image_boundary.id}",
            )
            await _click_visible(
                console,
                pilot,
                f"#console-message-action-fork-{image_boundary.id}",
            )
            await _settle(
                pilot,
                lambda: isinstance(host.screen_stack[-1], ConsoleForkChatModal),
                label="ASSISTANT fork modal",
            )
            assistant_request = console._session._active_fork_request
            assert assistant_request is not None
            assistant_fork_session_id = assistant_request.fork_session_id
            assistant_fork_conversation_id = assistant_request.fork_conversation_id
            assert assistant_fork_conversation_id is not None
            await pilot.press("enter")
            await _settle(
                pilot,
                lambda: (
                    host.screen_stack[-1] is console
                    and store.active_session_id == assistant_fork_session_id
                ),
                label="ASSISTANT fork activation",
            )
            durable_ids.append(assistant_fork_conversation_id)

            await _press_session_tab(console, pilot, user_fork_session_id)
            await _press_session_tab(console, pilot, source.id)
            await _press_session_tab(console, pilot, assistant_fork_session_id)
            assert _source_rows(db, source_conversation_id) == source_db_before
            assert _source_live(store, source.id) == source_live_before

            user_rows = db.get_messages_for_conversation(user_fork_conversation_id)
            assistant_rows = db.get_messages_for_conversation(
                assistant_fork_conversation_id
            )
            assert [row["content"] for row in user_rows] == [
                "Question with an attachment",
                "Answer [S1].",
                "Middle user boundary",
            ]
            assert [row["content"] for row in assistant_rows] == [
                "Question with an attachment",
                "Answer [S1].",
                "Middle user boundary",
                "[image] selected diagram",
            ]
            assert user_rows[0]["image_data"] == attachment_bytes
            assert assistant_rows[0]["image_data"] == attachment_bytes
            assert assistant_rows[-1]["image_data"] == _png_bytes((255, 0, 0))
            assistant_generation = db.get_generation_metadata_for_messages(
                (assistant_rows[-1]["id"],)
            )[assistant_rows[-1]["id"]]
            assert [item["prompt"] for item in assistant_generation] == [
                "selected diagram"
            ]
            assert [item["seed"] for item in assistant_generation] == [7]
            assert all(
                row["content"] not in {later_user.content, off_path.content}
                for row in (*user_rows, *assistant_rows)
            )
            for fork_id, boundary in (
                (user_fork_conversation_id, user_boundary),
                (assistant_fork_conversation_id, image_boundary),
            ):
                fork_row = db.get_conversation_by_id(fork_id)
                assert (
                    fork_row["root_id"]
                    == db.get_conversation_by_id(source_conversation_id)["root_id"]
                )
                assert fork_row["parent_conversation_id"] == source_conversation_id
                assert (
                    fork_row["forked_from_message_id"] == boundary.persisted_message_id
                )
                assert (
                    db.get_conversation_active_leaf(fork_id)
                    == (db.get_messages_for_conversation(fork_id)[-1]["id"])
                )

            # Exercise an independent temporary journey at the narrow viewport.
            await pilot.resize_terminal(80, 24)
            temporary = store.create_session(
                title="Temporary source",
                workspace_id=workspace_id,
                settings=settings,
                project_instruction_state=project_state,
                ephemeral=True,
            )
            temporary_user = store.append_message(
                temporary.id,
                role=ConsoleMessageRole.USER,
                content="Temporary citation marker [S1]",
            )
            temporary_image = store.append_message(
                temporary.id,
                role=ConsoleMessageRole.ASSISTANT,
                content="Temporary source image",
                attachments=(
                    MessageAttachment(
                        _png_bytes((0, 0, 255)),
                        "image/png",
                        "temporary.png",
                        0,
                    ),
                ),
            )
            temporary_video = store.append_video_message(
                temporary.id,
                video_metadata=VideoGenerationMetadata(
                    name="temporary-video-store-key",
                    prompt="animate temporary image",
                    backend="minimax",
                    model="video-fixture",
                    source_image_message_id=temporary_image.id,
                ),
            )
            await console._session._activate_native_console_session(temporary.id)
            await console._sync_native_console_chat_ui()
            durable_count_before = _conversation_count(db)
            await _click_visible(
                console, pilot, f"#console-message-{temporary_video.id}"
            )
            assert transcript.selected_message_id == temporary_video.id
            await _wait_for_selector(
                console,
                pilot,
                f"#console-message-action-fork-{temporary_video.id}",
            )
            await _click_visible(
                console,
                pilot,
                f"#console-message-action-fork-{temporary_video.id}",
            )
            await _settle(
                pilot,
                lambda: isinstance(host.screen_stack[-1], ConsoleForkChatModal),
                label="temporary fork modal",
            )
            temporary_request = console._session._active_fork_request
            assert temporary_request is not None
            painted_narrow = unescape(host.export_screenshot(simplify=True)).replace(
                "\N{NO-BREAK SPACE}", " "
            )
            assert "Fork chat" in painted_narrow
            temporary_fork_session_id = temporary_request.fork_session_id
            await pilot.press("enter")
            await _settle(
                pilot,
                lambda: (
                    host.screen_stack[-1] is console
                    and store.active_session_id == temporary_fork_session_id
                ),
                label="temporary fork activation",
            )
            assert _conversation_count(db) == durable_count_before
            temporary_fork = next(
                item
                for item in store.sessions()
                if item.id == temporary_fork_session_id
            )
            assert temporary_fork.ephemeral
            assert temporary_fork.persisted_conversation_id is None
            assert temporary.persisted_conversation_id is None
            assert [
                message.content
                for message in store.messages_for_session(temporary_fork_session_id)
            ][0] == temporary_user.content
            assert store.messages_for_session(temporary_fork_session_id)[
                -1
            ].video_metadata.is_unavailable_tombstone

            await console._session._promote_console_temporary_session()
            await _settle(
                pilot,
                lambda: temporary_fork.persisted_conversation_id is not None,
                label="temporary fork promotion",
            )
            temporary_promoted_id = temporary_fork.persisted_conversation_id
            assert temporary_promoted_id is not None
            promoted = db.get_conversation_by_id(temporary_promoted_id)
            assert promoted["root_id"] == temporary_promoted_id
            assert promoted["parent_conversation_id"] is None
            assert promoted["forked_from_message_id"] is None
            assert temporary.persisted_conversation_id is None

            view_snapshot = json.dumps(
                console._serialize_native_console_state(),
                default=str,
                sort_keys=True,
            )
            notifications = "\n".join(str(item) for item in app._notifications)
            diagnostics = "\n".join(log_lines)
            for private in (
                instruction_body,
                str(project_root.resolve()),
                str(source_scratch.root),
                attachment_sentinel.decode(),
                approval_round,
                permission_decision,
                recovery_body,
                run_body,
                provider_secret,
            ):
                assert private not in view_snapshot
                assert private not in notifications
                assert private not in diagnostics

        # A restarted app and store read the committed rows rather than the
        # previous screen's in-memory sessions.
        logger.remove(sink)
        db.close()
        restarted_db = CharactersRAGDB(db_path, client_id="fork-flow")
        restarted_repository = CitationTraceRepository(
            restarted_db,
            policy=CitationProvenanceRuntimePolicy(canonical_writes_enabled=True),
            identity_context=_identity(restarted_db),
            fingerprint_codec=TEST_FINGERPRINT_CODEC,
        )
        restarted_app = _build_test_app()
        restarted_app.chachanotes_db = restarted_db
        restarted_app.citation_trace_repository = restarted_repository
        restarted_host = ConsoleHarness(restarted_app)
        try:
            async with restarted_host.run_test(size=(120, 35)) as pilot:
                restarted_console = restarted_host.screen_stack[-1]
                await _wait_for_selector(
                    restarted_console, pilot, "#console-native-transcript"
                )
                restarted_store = restarted_console._ensure_console_chat_store()
                restored = [
                    _restore_conversation(
                        restarted_store, restarted_db, conversation_id
                    )
                    for conversation_id in (*durable_ids, temporary_promoted_id)
                ]
                await restarted_console._sync_native_console_chat_ui()
                for session, _messages in restored:
                    await _press_session_tab(restarted_console, pilot, session.id)
                    await _settle(
                        pilot,
                        lambda session_id=session.id: (
                            restarted_store.active_session_id == session_id
                        ),
                        label=f"reloaded tab {session.id}",
                    )

                user_messages = restored[0][1]
                assistant_messages = restored[1][1]
                temporary_messages = restored[2][1]
                assert [message.content for message in user_messages] == [
                    "Question with an attachment",
                    "Answer [S1].",
                    "Middle user boundary",
                ]
                assert [message.content for message in assistant_messages] == [
                    "Question with an attachment",
                    "Answer [S1].",
                    "Middle user boundary",
                    "[image] selected diagram",
                ]
                assert temporary_messages[0].content == "Temporary citation marker [S1]"
                assert temporary_messages[-1].video_metadata.is_unavailable_tombstone
                assert not hasattr(temporary_messages[-1].video_metadata, "path")
                assert temporary_messages[-1].video_metadata.name.startswith(
                    "forked-video-"
                )

                for conversation_id, (_session, messages) in zip(
                    durable_ids, restored[:2]
                ):
                    owner_rows = (
                        restarted_db.get_connection()
                        .execute(
                            "SELECT message_id, trace_id FROM rag_message_trace_owners "
                            "WHERE message_id IN ({})".format(
                                ",".join("?" for _ in messages)
                            ),
                            tuple(message.persisted_message_id for message in messages),
                        )
                        .fetchall()
                    )
                    assert owner_rows
                    assert all(row["trace_id"] == "trace-1" for row in owner_rows)
                temporary_owner_count = (
                    restarted_db.get_connection()
                    .execute(
                        "SELECT COUNT(*) FROM rag_message_trace_owners "
                        "WHERE message_id IN ({})".format(
                            ",".join("?" for _ in temporary_messages)
                        ),
                        tuple(
                            message.persisted_message_id
                            for message in temporary_messages
                        ),
                    )
                    .fetchone()[0]
                )
                assert temporary_owner_count == 0

                for session, messages in restored:
                    path = restarted_store.active_path_message_ids(session.id)
                    assert len(path) == len(messages)
                    assert (
                        restarted_store.get_message(path[-1]).content
                        == messages[-1].content
                    )
        finally:
            restarted_db.close()
    finally:
        if sink in getattr(logger, "_core", object()).handlers:
            logger.remove(sink)
        try:
            db.close()
        except Exception:
            pass
