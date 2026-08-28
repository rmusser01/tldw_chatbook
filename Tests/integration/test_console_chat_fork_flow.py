"""Provider-free production journey for Console chat forks (TASK-23088)."""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from html import unescape
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
    hydrate_console_session,
)
from tldw_chatbook.Chat.console_dispatch_checkpoint import (
    ConsoleDispatchCheckpoint,
    ConsoleDispatchCheckpointState,
    ConsoleDispatchReconstructability,
    ConsoleEgressClass,
    ConsoleLibraryItemScopeSnapshot,
    ConsoleProviderIntent,
    ConsoleResolvedDestination,
    ConsoleTurnLibraryAuthority,
)
from tldw_chatbook.Chat.console_library_policy import (
    ConsoleAssistantLibraryAccess,
    ConsoleAutoRetrieve,
    ConsoleLibraryPolicyCandidate,
    ConsoleLibraryPolicySnapshot,
)
from tldw_chatbook.Chat.console_project_instructions import (
    ProjectInstructionControlState,
)
from tldw_chatbook.Chat.console_speech_preferences import ConsoleSpeechPreferences
from tldw_chatbook.Chat.console_session_settings import ConsoleSessionSettings
from tldw_chatbook.Chat.rag_scope import RagScope, ScopeItem
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
from tldw_chatbook.Agents.local_tool_provider import LOCAL_SERVER_KEY
from tldw_chatbook.Agents.mcp_tool_provider import MCPPendingCall
from tldw_chatbook.MCP.permission_store import (
    BUILTIN_TOOL_SERVER_KEY,
    resolve_effective_state_by_key,
)
from tldw_chatbook.Video_Generation.video_metadata import VideoGenerationMetadata
from tldw_chatbook.Widgets.Console.console_fork_chat_modal import ConsoleForkChatModal
from tldw_chatbook.Widgets.Console.console_transcript import ConsoleTranscript
from tldw_chatbook.Workspaces.registry_service import LocalWorkspaceRegistryService


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


def _rows(
    db: CharactersRAGDB,
    sql: str,
    params: tuple[object, ...],
) -> tuple[dict[str, object], ...]:
    return tuple(dict(row) for row in db.get_connection().execute(sql, params))


def _source_rows(db: CharactersRAGDB, conversation_id: str) -> tuple[object, ...]:
    messages = tuple(
        dict(row) for row in db.get_messages_for_conversation(conversation_id, limit=50)
    )
    message_ids = tuple(str(row["id"]) for row in messages)
    placeholders = ",".join("?" for _ in message_ids) or "NULL"
    return (
        dict(db.get_conversation_by_id(conversation_id)),
        messages,
        db.get_attachments_for_messages(message_ids),
        db.get_generation_metadata_for_messages(message_ids),
        db.get_conversation_console_project_context(conversation_id),
        db.get_conversation_active_leaf(conversation_id),
        _rows(
            db,
            "SELECT * FROM console_conversation_context_policy "
            "WHERE conversation_id = ?",
            (conversation_id,),
        ),
        _rows(
            db,
            "SELECT * FROM console_conversation_library_policy "
            "WHERE conversation_id = ?",
            (conversation_id,),
        ),
        _rows(
            db,
            "SELECT message_id, message_revision, trace_id, state "
            "FROM rag_message_trace_owners "
            f"WHERE message_id IN ({placeholders}) ORDER BY message_id",
            message_ids,
        ),
        _rows(
            db,
            "SELECT message_id, conversation_id, seq, event_kind, payload_json "
            "FROM message_trajectory_metadata WHERE conversation_id = ? "
            "ORDER BY seq",
            (conversation_id,),
        ),
    )


def _source_live(
    store: ConsoleChatStore,
    controller: object,
    scratch_spaces: object,
    session_id: str,
) -> tuple[object, ...]:
    session = next(item for item in store.sessions() if item.id == session_id)
    scratch = scratch_spaces.snapshot(session_id)
    return (
        session.title,
        session.workspace_id,
        session.persisted_conversation_id,
        session.settings,
        session.context_policy_overrides,
        store.session_library_policy_candidate(session_id),
        session.rag_scope_holder.scope,
        session.runtime_backend,
        session.assistant_kind,
        session.assistant_id,
        session.assistant_authority_id,
        session.persona_memory_mode,
        session.character_id,
        session.character_name,
        session.user_display_name_override,
        session.character_system_template,
        session.speech_preferences,
        session.project_instruction_state,
        controller.has_pending_approval_round(session_id),
        controller.run_state_for(session_id),
        store.dispatch_recovery_for_session(session_id),
        scratch,
        scratch_spaces.is_live(scratch),
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


async def _restore_conversation(
    store: ConsoleChatStore,
    app: object,
    db: CharactersRAGDB,
    conversation_id: str,
) -> tuple[object, tuple[object, ...]]:
    row = db.get_conversation_by_id(conversation_id)
    tree = ChatConversationService(db).get_conversation_tree(conversation_id)
    settings = apply_resume_settings_overrides(
        ConsoleSessionSettings(provider="openai", model="gpt-fork-fixture"),
        row,
    )
    session = await hydrate_console_session(
        app=app,
        store=store,
        conversation_id=conversation_id,
        tree=tree,
        settings=settings,
    )
    messages = tuple(store.messages_for_session(session.id))
    return session, messages


def _checkpoint(
    db: CharactersRAGDB,
    *,
    conversation_id: str,
    user_message_id: str,
    assistant_message_id: str,
    attempt_id: str,
) -> ConsoleDispatchCheckpoint:
    user = db.get_message_by_id(user_message_id)
    assistant = db.get_message_by_id(assistant_message_id)
    policy = ConsoleLibraryPolicySnapshot(
        auto_retrieve=ConsoleAutoRetrieve.NEVER,
        assistant_access=ConsoleAssistantLibraryAccess.BLOCKED,
        policy_revision=1,
        source="durable",
    )
    return ConsoleDispatchCheckpoint(
        assistant_message_id=assistant_message_id,
        user_message_id=user_message_id,
        conversation_id=conversation_id,
        preparation_id=f"preparation-{attempt_id}",
        attempt_id=attempt_id,
        state=ConsoleDispatchCheckpointState.ACCEPTED,
        checkpoint_revision=1,
        user_message_version=int(user["version"]),
        assistant_message_version=int(assistant["version"]),
        origin="manual",
        queue_entry_id=None,
        frozen_authority=ConsoleTurnLibraryAuthority(
            policy=policy,
            direct_library_tools=False,
            source_types=("notes",),
            scope_snapshot=ConsoleLibraryItemScopeSnapshot(
                note_ids=("fork-evidence",),
                media_ids=(),
                conversations_allowed=False,
            ),
            provider_intent=ConsoleProviderIntent(
                provider="openai",
                model="gpt-fork-fixture",
                endpoint=None,
            ),
            attempt_id=attempt_id,
        ),
        resolved_destination=ConsoleResolvedDestination(
            provider="openai",
            model="gpt-fork-fixture",
            endpoint_identity="fixture-endpoint",
            egress_class=ConsoleEgressClass.UNKNOWN,
        ),
        reconstructability=ConsoleDispatchReconstructability(
            attachments_reconstructable=True,
            evidence_reconstructable=True,
            prefill_reconstructable=True,
            opaque_reference=None,
        ),
    )


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
    approval_secret = "APPROVAL_ARGUMENT_MUST_NOT_LEAK"
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
    permission_service = app.unified_mcp_service
    permission_store = permission_service.permission_store
    assert permission_store is not None
    permission_store.set_tool_state(LOCAL_SERVER_KEY, "fs_read", "ask")
    permission_store.set_tool_state(BUILTIN_TOOL_SERVER_KEY, "calculator", "deny")
    permission_service.approve_for_session(LOCAL_SERVER_KEY, "fs_read")
    permission_owner_before = (
        permission_store.load(),
        permission_service.is_session_approved(LOCAL_SERVER_KEY, "fs_read"),
        resolve_effective_state_by_key(
            permission_store.load(), LOCAL_SERVER_KEY, "fs_read"
        ),
        resolve_effective_state_by_key(
            permission_store.load(), BUILTIN_TOOL_SERVER_KEY, "calculator"
        ),
    )
    assert permission_owner_before[1] is True
    assert permission_owner_before[2].state == "ask"
    assert permission_owner_before[3].state == "deny"
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
                            prompt="unselected default diagram",
                            negative_prompt="",
                            backend="openai",
                            model="image-fixture",
                            seed=6,
                            style=None,
                            params={"size": "small"},
                        ),
                    ),
                    (
                        _png_bytes((0, 255, 0)),
                        "image/png",
                        GenerationVariantMeta(
                            prompt="selected non-default diagram",
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
            source_recovery = store.publish_durable_dispatch_checkpoint(
                source.id,
                _checkpoint(
                    db,
                    conversation_id=source_conversation_id,
                    user_message_id=later_user.persisted_message_id,
                    assistant_message_id=later_video.persisted_message_id,
                    attempt_id="source-recovery",
                ),
                in_flight=True,
            )
            store.mark_dispatch_recovery_needed(source.id, later_video.id)
            source_recovery_after_mark = store.dispatch_recovery_for_session(source.id)
            assert source_recovery_after_mark is not None
            source_recovery_copy = source_recovery_after_mark.visible_copy
            assert source_recovery.assistant_message_id == later_video.id
            controller._set_run_state(  # noqa: SLF001
                ConsoleRunState(ConsoleRunStatus.STREAMING, run_body),
                session_id=source.id,
            )
            runtime = console._console_runtime()
            source_scratch = runtime.scratch_spaces.snapshot(source.id)
            source_lease = runtime.scratch_spaces.lease(source_scratch)
            assert source_lease.__enter__() == source_scratch.root

            await console._sync_native_console_chat_ui()
            await _click_visible(console, pilot, f"#console-message-{user_boundary.id}")
            transcript = console.query_one(
                "#console-native-transcript", ConsoleTranscript
            )
            assert transcript.selected_message_id == user_boundary.id

            # ConsoleHarness mounts the production screen under the harness app;
            # the production controller therefore marshals through that live
            # Textual owner here, just as it marshals through TldwCli in-process.
            controller.app = host
            controller.mcp_approval_timeout_seconds = lambda: 30.0
            approval_task = asyncio.create_task(
                asyncio.to_thread(
                    controller.request_mcp_approvals,
                    [
                        MCPPendingCall(
                            llm_name="mcp__fixture__read",
                            server_key=LOCAL_SERVER_KEY,
                            tool_name="fs_read",
                            server_label="Local tools",
                            arguments={"opaque": approval_secret},
                            reason=approval_secret,
                        )
                    ],
                    session_id=source.id,
                )
            )
            await _settle(
                pilot,
                lambda: controller.has_pending_approval_round(source.id),
                label="real source approval round",
            )

            source_db_before = _source_rows(db, source_conversation_id)
            source_live_before = _source_live(
                store, controller, runtime.scratch_spaces, source.id
            )

            # Pointer-select the USER boundary, then exercise the selected-row
            # keyboard action and edit the focused default title.
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
            title_input = user_modal.query_one("#console-fork-chat-title", Input)
            assert title_input.has_focus
            # The production modal selects the incumbent default on mount. Typing is
            # therefore the real keyboard replacement path; no test-side value write.
            assert title_input.selection == (0, len(title_input.value))
            await pilot.press(*"Focused user-boundary fork")
            assert title_input.value == "Focused user-boundary fork"
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
            assert (
                _source_live(store, controller, runtime.scratch_spaces, source.id)
                == source_live_before
            )
            assert _source_rows(db, source_conversation_id) == source_db_before
            assert permission_owner_before == (
                permission_store.load(),
                permission_service.is_session_approved(LOCAL_SERVER_KEY, "fs_read"),
                resolve_effective_state_by_key(
                    permission_store.load(), LOCAL_SERVER_KEY, "fs_read"
                ),
                resolve_effective_state_by_key(
                    permission_store.load(), BUILTIN_TOOL_SERVER_KEY, "calculator"
                ),
            )
            assert controller.has_pending_approval_round(source.id)
            assert not controller.has_pending_approval_round(user_fork.id)
            assert (
                controller.run_state_for(source.id).status is ConsoleRunStatus.STREAMING
            )
            assert (
                controller.run_state_for(user_fork.id).status is ConsoleRunStatus.IDLE
            )
            assert store.dispatch_recovery_for_session(source.id) is not None
            assert store.dispatch_recovery_for_session(user_fork.id) is None
            assert (
                user_fork.project_instruction_state.project_instruction_notice_key
                is None
            )
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
            approval_payload = console._task_resume_state.pending_approval
            assert approval_payload is not None
            controller.resolve_pending_approval(
                {"mcp__fixture__read": "deny"},
                round_id=approval_payload["round_id"],
            )
            assert await asyncio.wait_for(approval_task, timeout=2.0) == {
                "mcp__fixture__read": "deny"
            }
            source_lease.__exit__(None, None, None)
            assert not controller.has_pending_approval_round(source.id)
            source_live_after_approval = _source_live(
                store, controller, runtime.scratch_spaces, source.id
            )
            await _click_visible(
                console, pilot, f"#console-message-{image_boundary.id}"
            )
            assert transcript.selected_message_id == image_boundary.id
            await _wait_for_selector(
                console,
                pilot,
                f"#console-message-action-variant-next-{image_boundary.id}",
            )
            variant_next = console.query_one(
                f"#console-message-action-variant-next-{image_boundary.id}", Button
            )
            assert not variant_next.disabled
            variant_next.press()
            await pilot.pause()
            await _settle(
                pilot,
                lambda: (
                    console._console_generation_browse().get(image_boundary.id) == 1
                ),
                label="non-default generated-image selection",
            )
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
            assert (
                _source_live(store, controller, runtime.scratch_spaces, source.id)
                == source_live_after_approval
            )

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
            assert assistant_rows[-1]["image_data"] == _png_bytes((0, 255, 0))
            # A forked single image is canonical position zero (`image_data`),
            # so there must be no stale extra-variant attachment sidecar.
            assert db.get_attachments_for_messages((assistant_rows[-1]["id"],)) == {}
            assert _png_bytes((255, 0, 0)) not in {
                row["image_data"] for row in assistant_rows
            }
            assistant_generation = db.get_generation_metadata_for_messages(
                (assistant_rows[-1]["id"],)
            )[assistant_rows[-1]["id"]]
            assert [item["prompt"] for item in assistant_generation] == [
                "selected non-default diagram"
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
            source_checkpoint = source_recovery.checkpoint
            assert source_checkpoint is not None
            store.register_ephemeral_dispatch_recovery(
                temporary.id,
                user_message_id=temporary_user.id,
                assistant_message_id=temporary_video.id,
                preparation_id="preparation-temporary-recovery",
                attempt_id="temporary-recovery",
                checkpoint_state=ConsoleDispatchCheckpointState.ACCEPTED,
                origin="manual",
                queue_entry_id=None,
                frozen_authority=source_checkpoint.frozen_authority,
                resolved_destination=source_checkpoint.resolved_destination,
                reconstructability=source_checkpoint.reconstructability,
                runtime_active=True,
            )
            store.mark_dispatch_recovery_needed(temporary.id, temporary_video.id)
            controller._set_run_state(  # noqa: SLF001
                ConsoleRunState(ConsoleRunStatus.STREAMING, "Temporary source run"),
                session_id=temporary.id,
            )
            temporary_scratch = runtime.scratch_spaces.snapshot(temporary.id)
            temporary_lease = runtime.scratch_spaces.lease(temporary_scratch)
            assert temporary_lease.__enter__() == temporary_scratch.root
            await console._session._activate_native_console_session(temporary.id)
            await console._sync_native_console_chat_ui()
            temporary_live_before = _source_live(
                store, controller, runtime.scratch_spaces, temporary.id
            )
            durable_count_before = _conversation_count(db)
            temporary_transcript = console.query_one(ConsoleTranscript)
            temporary_transcript.select_message(temporary_video.id)
            await pilot.pause()
            assert temporary_transcript.selected_message_id == temporary_video.id
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
            assert (
                _source_live(store, controller, runtime.scratch_spaces, temporary.id)
                == temporary_live_before
            )
            assert store.dispatch_recovery_for_session(temporary_fork.id) is None
            assert (
                controller.run_state_for(temporary_fork.id).status
                is ConsoleRunStatus.IDLE
            )
            temporary_fork_scratch = runtime.scratch_spaces.snapshot(temporary_fork.id)
            assert temporary_fork_scratch.root != temporary_scratch.root
            assert temporary_fork_scratch.token != temporary_scratch.token
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
            assert (
                _source_live(store, controller, runtime.scratch_spaces, temporary.id)
                == temporary_live_before
            )
            assert (
                temporary_fork.project_instruction_state.project_instruction_notice_key
                is None
            )
            temporary_lease.__exit__(None, None, None)

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
                source_scratch.token,
                str(temporary_scratch.root),
                temporary_scratch.token,
                str(permission_store.path),
                attachment_sentinel.decode(),
                approval_secret,
                source_recovery_copy,
                run_body,
                "Temporary source run",
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
                restored = []
                for conversation_id in (
                    source_conversation_id,
                    *durable_ids,
                    temporary_promoted_id,
                ):
                    restored.append(
                        await _restore_conversation(
                            restarted_store,
                            restarted_app,
                            restarted_db,
                            conversation_id,
                        )
                    )
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

                source_messages = restored[0][1]
                user_messages = restored[1][1]
                assistant_messages = restored[2][1]
                temporary_messages = restored[3][1]
                assert [message.content for message in source_messages] == [
                    "Question with an attachment",
                    "Answer [S1].",
                    "Middle user boundary",
                    "[image] selected diagram",
                    "Later turn excluded from both middle forks",
                    later_video.content,
                ]
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
                assert assistant_messages[-1].image_data == _png_bytes((0, 255, 0))
                assert len(assistant_messages[-1].attachments) == 1
                assert [
                    item.prompt for item in assistant_messages[-1].generation_metadata
                ] == ["selected non-default diagram"]
                assert temporary_messages[0].content == "Temporary citation marker [S1]"
                assert temporary_messages[-1].video_metadata.is_unavailable_tombstone
                assert not hasattr(temporary_messages[-1].video_metadata, "path")
                assert temporary_messages[-1].video_metadata.name.startswith(
                    "forked-video-"
                )

                restored_source = restored[0][0]
                assert restored_source.user_display_name_override == "Rowan"
                assert restored_source.speech_preferences == ConsoleSpeechPreferences(
                    auto_speak=True
                )
                source_identity_after_restart = ChatConversationService(
                    restarted_db
                ).get_conversation_metadata(source_conversation_id)
                assert source_identity_after_restart is not None
                assert source_identity_after_restart["assistant_kind"] is None
                assert source_identity_after_restart["assistant_id"] == "console"
                assert (
                    restored_source.runtime_backend,
                    restored_source.assistant_kind,
                    restored_source.assistant_id,
                    restored_source.assistant_authority_id,
                    restored_source.persona_memory_mode,
                    restored_source.character_id,
                ) == ("local", None, None, None, None, None)
                restored_user_boundary = next(
                    message
                    for message in source_messages
                    if message.content == "Middle user boundary"
                )
                restored_source_eligibility = restarted_store.fork_eligibility(
                    restored_user_boundary.id
                )
                assert restored_source_eligibility.eligible is True, (
                    restored_source_eligibility.reason
                )
                assert restarted_store.session_library_policy_candidate(
                    restored_source.id
                ) == ConsoleLibraryPolicyCandidate(
                    auto_retrieve=ConsoleAutoRetrieve.NEVER,
                    assistant_access=ConsoleAssistantLibraryAccess.BLOCKED,
                )
                assert (
                    _source_rows(restarted_db, source_conversation_id)
                    == source_db_before
                )

                promoted_session = restored[3][0]
                assert promoted_session.settings.pinned_prefill is None
                assert (
                    promoted_session.project_instruction_state.project_instruction_notice_key
                    is None
                )
                assert promoted_session.project_instruction_state.project_instructions_enabled
                assert (
                    promoted_session.project_instruction_state.working_folder_binding_id
                    == binding.binding_id
                )
                assert (
                    promoted_session.project_instruction_state.working_folder_locator_fingerprint
                    == project_state.working_folder_locator_fingerprint
                )
                reloaded_workspace_db = WorkspaceDB(
                    app.local_workspace_db.db_path,
                    client_id=app.local_workspace_db.client_id,
                )
                try:
                    reloaded_workspace_registry = LocalWorkspaceRegistryService(
                        reloaded_workspace_db
                    )
                    promoted_binding = resolve_project_instruction_binding(
                        promoted_session,
                        reloaded_workspace_registry,
                    )
                    assert promoted_binding is not None
                    assert promoted_binding.binding.binding_id == binding.binding_id
                    assert promoted_binding.root == project_root.resolve()
                    assert (
                        promoted_binding.locator_fingerprint
                        == project_state.working_folder_locator_fingerprint
                    )
                finally:
                    reloaded_workspace_db.close()

                for conversation_id, (_session, messages) in zip(
                    durable_ids, restored[1:3]
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
