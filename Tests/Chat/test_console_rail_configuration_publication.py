"""Real rail writes must not silently replace a selected backend's sources."""

from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
import sys

import pytest

from Tests.private_profile import private_profile_test


@private_profile_test
@pytest.mark.asyncio
async def test_external_original_rail_save_invalidates_backend_selection(request):
    """Exercise the actual writer and reader, in separate Python processes."""
    from Tests.Chat.test_console_execution_profile import _settled_profile
    from tldw_chatbook import config
    from tldw_chatbook.Chat.console_rail_state import build_console_rail_preference_key
    from tldw_chatbook.Chat.console_execution_profile import (
        open_console_backend_profile,
    )

    path, directory, *_ = _settled_profile()
    profile = open_console_backend_profile(
        expected_config_path=path, expected_data_directory=directory,
    )
    original_cache = config._CONFIG_CACHE
    original_identity = config.current_config_identity()
    key = build_console_rail_preference_key(layout_scope="global").value
    try:
        assert profile.selection_is_current()
        # This is the unchanged public writer called by the original rail
        # worker. The separate process cannot mutate the reader's Python cache.
        writer = await asyncio.to_thread(
            subprocess.run,
            [sys.executable, "-c", (
                "import sys; from pathlib import Path; "
                "from tldw_chatbook import config; "
                "from tldw_chatbook.Chat.console_rail_state import "
                "build_console_rail_preference_key; "
                "assert config.get_cli_config_path() == Path(sys.argv[1]); "
                "assert config.get_user_data_dir() == Path(sys.argv[2]); "
                "assert config.save_setting_to_cli_config("
                "'console.rail_state', "
                "build_console_rail_preference_key(layout_scope='global').value, "
                "{'workspace_open': True})"
            ), str(path), str(directory)],
            env={**os.environ, "TLDW_CONFIG_PATH": str(path)},
            capture_output=True, check=False,
        )
        assert writer.returncode == 0, "Original external rail save failed"
        assert config._CONFIG_CACHE is original_cache
        assert config.current_config_identity() == original_identity
        assert profile.selection_is_current()

        # Respect the original external-edit sampling interval. No cache flag,
        # loader, file-stamp check or selection predicate is patched.
        await asyncio.sleep(1.05)
        settings = config.load_settings()
        assert settings["console"]["rail_state"][key]["workspace_open"] is True
        assert config._CONFIG_CACHE is not original_cache
        assert not profile.selection_is_current()
    finally:
        await profile.aclose()


async def _external_rail_save(path, directory, key, value, *, extra=False, marker=False):
    result = await asyncio.to_thread(
        subprocess.run,
        [sys.executable, "-c", (
            "import sys,json,tomllib; from pathlib import Path; "
            "from tldw_chatbook import config; "
            "assert config.get_cli_config_path() == Path(sys.argv[1]); "
            "assert config.get_user_data_dir() == Path(sys.argv[2]); "
            "assert config.save_setting_to_cli_config("
            "'console.rail_state', sys.argv[3], json.loads(sys.argv[4])); "
            "assert sys.argv[5] == 'False' or config.save_setting_to_cli_config("
            "'console.rail_state', 'unrequested-record', {'left_open': False}); "
            "assert sys.argv[6] == 'False' or config.replace_cli_config({"
            "**tomllib.loads(config.read_cli_config_serialized()), '_first_run': True})"
        ), str(path), str(directory), key, json.dumps(value), str(extra), str(marker)],
        env={**os.environ, "TLDW_CONFIG_PATH": str(path)},
        capture_output=True, check=False,
    )
    assert result.returncode == 0, "Original external rail save failed"


def _rail_comparison_note(backend, publication):
    """Report allowlisted product keys and shapes, never values or paths."""
    from copy import deepcopy

    command, _profile, raw, normalized, _facts, _permission_capture = backend._rail_refresh
    update = command.rail_update

    def changed(value):
        result = deepcopy(value)
        result.setdefault("console", {}).setdefault("rail_state", {})[update["key"]] = dict(update["value"])
        return result

    original_expected = changed(normalized)
    complete_expected = deepcopy(original_expected)
    mirror = complete_expected.get("COMPREHENSIVE_CONFIG_RAW")
    if type(mirror) is dict:
        complete_expected["COMPREHENSIVE_CONFIG_RAW"] = changed(mirror)

    # Literal product schema names only. Unknown/user-defined names are counted
    # together; neither their spelling nor any setting value is emitted.
    allowed = frozenset("""
        API_COMPONENT_ROOT APP_DATABASE_CONFIG APP_MODE_STR APP_RAG_SEARCH_CONFIG
        APP_TTS_CONFIG COMPREHENSIVE_CONFIG_RAW DATABASE_URL LOG_LEVEL OPENAI_API_KEY
        PROJECT_ROOT Prompts SINGLE_USER_API_KEY SINGLE_USER_FIXED_ID SINGLE_USER_MODE
        STT_settings SearchEngines SearchSettings USERS_DB_CONFIGURED USERS_NAME acp
        agents analysis_defaults anthropic_api aphrodite_api api_endpoints api_settings
        appearance auto_save briefings_feed_server caching canvas character_defaults
        chat chat_defaults chat_dictionaries chunking chunking_config cohere_api
        config_schema_version confluence console console_ssh custom_openai_api
        custom_openai_api_2 database deepseek_api diarization dictation elevenlabs_api
        embedding_config first_run general github groq_api hooks huggingface_api
        image_generation kobold_api legacy_api_key_providers library llama_cpp_api
        llamacpp_snapshots llm_api_settings llm_management local_ingestion logging mcp
        media_cleanup media_processing meetings metrics mistral_api model_capabilities
        model_catalog network notes ollama_api ooba_api openai_api openrouter_api _first_run
        output_path permission_summary persona_buddy processing_choice prompts providers
        rag rag_citations rag_search scheduling search_engine_specific_settings
        search_engines_keys search_settings_general session_summary skills splash_screen
        subscriptions sync system_preferences tabby_api tamagotchi tldw_api tools
        transcription tts_settings video_generation vllm_api web_scraper_settings
        web_security web_server webfetch webhooks rail_state rail_layout_scope
        agent_open workspace_open context_open left_open right_open width left_width
        right_width enabled model models api_key api_url base_url default_model
    """.split())
    missing = object()

    def shape(value):
        names = {dict: "dict", list: "list", tuple: "tuple", str: "str", bool: "bool",
                 int: "int", float: "float", type(None): "none"}
        return {
            "type": "missing" if value is missing else names.get(type(value), "other"),
            "items": len(value) if type(value) in (dict, list, tuple) else None,
        }

    def differences(previous, current, path, depth=0):
        if previous == current:
            return []
        row = {
            "key": path, "before": shape(previous), "after": shape(current),
            "direction": "added" if previous is missing else "removed" if current is missing else "changed",
        }
        if type(previous) is not dict or type(current) is not dict or depth == 4:
            return [row]
        changed_keys = {item for item in previous.keys() | current.keys()
                        if previous.get(item, missing) != current.get(item, missing)}
        row["changed_count"] = len(changed_keys)
        known = changed_keys & allowed
        row["unknown_count"] = len(changed_keys - known)
        result = [row]
        for item in sorted(known):
            result.extend(differences(previous.get(item, missing), current.get(item, missing),
                                      f"{path}.{item}", depth + 1))
        return result

    return json.dumps({
        "raw_matches_requested": changed(raw) == publication.selection_cache,
        "normalized_matches_single_projection": original_expected == publication.sources.configuration,
        "normalized_has_original_raw_mirror": type(mirror) is dict,
        "normalized_matches_both_projections": complete_expected == publication.sources.configuration,
        "differences": (
            differences(changed(raw), publication.selection_cache, "raw")
            + differences(complete_expected, publication.sources.configuration, "normalized")
        )[:100],
    }, sort_keys=True)


@private_profile_test
@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", [
    "accepted", "accepted_unmounted", "partial", "extra", "persisted_marker", "refresh_failure",
])
async def test_exact_rail_publication_reassociates_only_after_retired_refresh(request, outcome):
    import inspect
    from contextlib import nullcontext
    from dataclasses import replace

    from Tests.Chat.test_console_control_trace import _observe
    from Tests.Chat.test_console_execution_profile import _settled_profile
    from Tests.Chat.test_console_execution_process import _SentFrames
    from tldw_chatbook import config
    from tldw_chatbook.Chat.console_execution_backend import ConsoleExecutionBackend
    from tldw_chatbook.Chat.console_execution_profile import open_console_backend_profile
    from tldw_chatbook.Chat.console_execution_protocol import ConsoleWireMessage, WireIdentity
    from tldw_chatbook.Chat.console_execution_worker import _BackendService
    from tldw_chatbook.Chat.console_domain_display import ConsoleDisplayFull
    from tldw_chatbook.Chat.console_display_phases import ConsoleDisplayPhaseRefused
    from tldw_chatbook.Chat.console_session_settings import ConsoleSessionSettings
    from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired
    from tldw_chatbook.Backup_Recovery.mcp_source_participants import source_receipt
    from tldw_chatbook.MCP.local_permission_operation import LocalPermissionOperation
    from tldw_chatbook.Tool_Packs.binding import profile_policy_digest
    from tldw_chatbook.Chat.console_rail_state import (
        build_console_rail_preference_key, serialize_console_rail_stored_preferences,
    )

    path, directory, *_ = _settled_profile()
    # Provisioning above can create the first file and annotate that creator's
    # cache with _first_run. The real child opens an already-created profile
    # after App setup. Read that existing file through the original loader
    # before capturing this fixture's backend authority, just as the child does.
    config.load_settings(force_reload=True)
    assert "_first_run" not in config._CONFIG_CACHE
    profile = open_console_backend_profile(expected_config_path=path, expected_data_directory=directory)
    backend = ConsoleExecutionBackend(profile, profile_epoch="rail-profile", service_epoch="rail-service")
    channel = _SentFrames()
    key = build_console_rail_preference_key(layout_scope="global").value
    value = serialize_console_rail_stored_preferences({"workspace_open": True})
    original_cleanup = profile._cleanup
    sequence_offset = 0
    full_reads = []
    accepted = outcome in {"accepted", "accepted_unmounted"}

    def command(sequence, operation, payload, revision=0):
        return ConsoleWireMessage(
            WireIdentity("rail-profile", "rail-service", sequence + sequence_offset,
                         f"rail-{sequence}", source_revision=revision),
            "command", operation, payload,
        )

    def replies(sequence):
        return [row for row in channel.messages if row.identity.request_id == f"rail-{sequence}"]

    def fail_reload(_frame):
        raise RuntimeError("Original configuration publication failed")

    async def permission_action(operation):
        sequence = backend._sequence + 1
        request_id = f"rail-permission-{sequence}"
        finished = service.accept(ConsoleWireMessage(
            WireIdentity(
                "rail-profile", "rail-service", sequence, request_id,
                source_revision=backend.controls.source_revision,
            ),
            "command", "PermissionAction", {"operation": operation},
        ))
        assert finished is not None
        _, receipt = backend.adapter._receipts[request_id]
        await finished
        messages = [row for row in channel.messages if row.identity.request_id == request_id]
        assert [row.kind for row in messages] == ["received", "applied", "retired"]
        assert messages[-1].payload["retired"] is True
        assert await asyncio.shield(receipt.retired) is True
        return receipt.applied.result()

    async def visible_projection(action):
        sequence = backend._sequence + 1
        receipt = backend.dispatch(ConsoleWireMessage(
            WireIdentity(
                "rail-profile", "rail-service", sequence, f"rail-view-{sequence}",
                session.id, session.incarnation_id, backend.controls.source_revision,
            ),
            "command", "ReadProjection",
            {"known_display_revision": None,
             "view_checkpoint": {"attachment_id": "rail-view", "visit": 1, "action": action}},
        ))
        try:
            with _observe(ConsoleDisplayFull.run.__code__, sys.monitoring.events.PY_START,
                          lambda frame: full_reads.append(frame.f_locals["self"])):
                return await asyncio.shield(receipt.applied)
        finally:
            assert await asyncio.shield(receipt.retired) is True

    try:
        backend.start()
        service = _BackendService(backend, channel)
        display = backend.runtime._execution_service.prepare_display_owner()
        producers = display.character, display.context, display.appearance
        assert display._graph_bound  # Original open_local_controls binds before any FULL.
        if accepted:
            permissions = backend.runtime._execution_service.ensure_permission_operations()
            permission_service = permissions._control_plane
            permission_store = permissions._store
            permission_publication = permissions._publication
            permission_path_lock = permissions._path_lock
            permission_grants = permissions._grants
            permission_grant_lock = permissions._grant_lock
            permission_receipt = permissions._source_receipt
            permission_snapshot = await permission_action({"kind": "ReadPermissionSnapshot", "strict": True})
            digest = profile_policy_digest(permission_snapshot.payload["profiles"]["default"])
            # Use the original producer: this test does not manufacture a grant
            # by populating the retained set or treating a display read as one.
            with LocalPermissionOperation(permission_store, permissions._require_current) as grant:
                permission_service.approve_for_session(
                    "agent:builtin", "Calculator", expected_profile_digest=digest,
                )
            assert grant.retired
            assert permission_grants == {("default", "agent:builtin", "Calculator")}
        if outcome == "accepted":
            session = backend.store.create_session(
                title="Rail source transition", assistant_kind="generic",
                settings=ConsoleSessionSettings(provider="openai", model=None, source="user"),
            )
            await visible_projection("mount")
            assert len(full_reads) == 1
            original_full = full_reads[0]
            assert original_full.effect.completed.result() is True
            assert original_full.physically_retired
        sequence_offset = backend._sequence
        await service.accept(command(1, "FenceSource", {
            "source": "configuration", "rail_update": {"key": key, "value": value},
        }))
        assert [row.kind for row in replies(1)] == ["received", "applied", "retired"]
        assert replies(1)[-1].payload["retired"] is True
        original_sources = backend.domain.sources
        before = profile._selection_cache
        actual = dict(value)
        if outcome == "partial":
            actual.pop("agent_open")
        await _external_rail_save(
            path, directory, key, actual,
            extra=outcome == "extra", marker=outcome == "persisted_marker",
        )
        assert profile._selection_cache is config._CONFIG_CACHE is before

        observation = (
            _observe(inspect.unwrap(config._publish_runtime_config_unlocked).__code__,
                     sys.monitoring.events.PY_START, fail_reload)
            if outcome == "refresh_failure" else nullcontext()
        )
        with observation:
            await service.accept(command(2, "RefreshConfiguration", {"fence_request_id": "rail-1"}, 1))
        if outcome == "refresh_failure":
            assert [row.kind for row in replies(2)] == ["received", "failed", "retired"]
            assert replies(2)[-1].payload["retired"] is False
            assert not backend.controls._publications
        else:
            assert [row.kind for row in replies(2)] == ["received", "applied", "retired"]
            assert replies(2)[-1].payload["retired"] is True
            publication = backend.adapter._result("rail-2", ("RefreshConfiguration",))
            assert publication.selection_cache is config._CONFIG_CACHE is not before
            with pytest.raises(RuntimeError, match="source association changed"):
                backend.require_sources()
            with pytest.raises(RuntimeError, match="not issued"):
                backend._require_rail_publication(replace(publication))
            if outcome == "persisted_marker":
                # An actual persisted marker remains an unrelated write. The
                # verifier must not strip it just because initial provisioning
                # uses the same name for a transient bootstrap annotation.
                assert "_first_run" not in before
                assert publication.selection_cache["_first_run"] is True
                with pytest.raises(RuntimeError, match="another change"):
                    backend._require_rail_publication(publication)
            if accepted:
                # The original normalizer publishes separate Console and raw
                # mirror tables. Exact requested values belong in both; an
                # incomplete mirror or any unrelated key still must refuse.
                from copy import deepcopy

                note = _rail_comparison_note(backend, publication)
                predicates = json.loads(note)
                predicates.pop("differences")
                assert predicates == {
                    "raw_matches_requested": True,
                    "normalized_matches_single_projection": False,
                    "normalized_has_original_raw_mirror": True,
                    "normalized_matches_both_projections": True,
                }, note
                settings = publication.sources.configuration
                mirror = settings["COMPREHENSIVE_CONFIG_RAW"]
                try:
                    settings["COMPREHENSIVE_CONFIG_RAW"] = deepcopy(mirror)
                    settings["COMPREHENSIVE_CONFIG_RAW"]["console"]["rail_state"][key].pop("agent_open")
                    with pytest.raises(RuntimeError, match="another change"):
                        backend._require_rail_publication(publication)
                    settings["COMPREHENSIVE_CONFIG_RAW"] = deepcopy(mirror)
                    settings["COMPREHENSIVE_CONFIG_RAW"]["console"]["rail_state"]["unrequested-record"] = {}
                    with pytest.raises(RuntimeError, match="another change"):
                        backend._require_rail_publication(publication)
                finally:
                    settings["COMPREHENSIVE_CONFIG_RAW"] = mirror
                # Source transition controls use the actual issued publication
                # and retained FULL. These are structural refusal probes; no
                # synthetic native success or replacement producer is supplied.
                with pytest.raises(RuntimeError, match="not issued"):
                    display.apply_configuration_publication(replace(publication))
                with pytest.raises(RuntimeError, match="control graph changed"):
                    await asyncio.to_thread(display.apply_configuration_publication, publication)
                if outcome == "accepted":
                    display._full = original_full
                    try:
                        with pytest.raises(ConsoleDisplayPhaseRefused, match="still owns work"):
                            display.apply_configuration_publication(publication)
                    finally:
                        display._full = None
                async with display.context.lock:
                    with pytest.raises(ConsoleDisplayPhaseRefused, match="still owns work"):
                        display.apply_configuration_publication(publication)
                display.apply_configuration_publication = lambda _publication: None
                try:
                    with pytest.raises(ConsoleDisplayPhaseRefused, match="defining body changed"):
                        backend.domain.apply_configuration_publication(publication)
                finally:
                    del display.apply_configuration_publication
                assert display.sources is original_sources
            applied = service.accept(command(3, "ApplySourceRevision", {"publication_request_id": "rail-2"}, 1))
            if applied is not None:
                await applied
            if accepted:
                actual = [row.kind for row in replies(3)]
                if actual != ["received", "applied", "retired"]:
                    pytest.fail("Original rail publication refused: " + _rail_comparison_note(backend, publication))
                assert replies(3)[-1].payload["retired"] is True
                backend.require_sources()
                assert backend.profile is profile and profile._cleanup is original_cleanup
                assert backend.domain.sources.configuration is profile.configuration
                assert profile.configuration["console"]["rail_state"][key] == value
                assert backend.controls._intake_open
                assert display is backend.runtime._execution_service.prepare_display_owner()
                assert producers == (display.character, display.context, display.appearance)
                assert display.sources is backend.domain.sources is not original_sources
                assert display.context.key is display.context.value is None
                assert display.context.at == 0.0
                assert display._character_reuse_checkpoint is None
                assert backend.runtime._execution_service.ensure_permission_operations() is permissions
                assert permissions._control_plane is backend.domain.sources.control_plane is permission_service
                assert permissions._store is permission_service._permission_store is permission_store
                assert permissions._publication is permission_store._permission_publication is permission_publication
                assert permissions._path_lock is permission_store._path_lock is permission_path_lock
                assert permissions._grants is permission_service._session_approvals is permission_grants
                assert permissions._grant_lock is permission_service._session_approvals_lock is permission_grant_lock
                assert permission_grants == {("default", "agent:builtin", "Calculator")}
                current_permission_receipt = permissions._source_receipt
                assert current_permission_receipt == source_receipt(permission_store)
                assert current_permission_receipt != permission_receipt
                permissions._source_receipt = permission_receipt
                try:
                    with pytest.raises(RecoveryRequired, match="permission_operation_source_changed"):
                        permissions._require_current()
                finally:
                    permissions._source_receipt = current_permission_receipt
                permissions._grant_lock = object()
                try:
                    with pytest.raises(RecoveryRequired, match="permission_operation_source_changed"):
                        permissions._require_current()
                finally:
                    permissions._grant_lock = permission_grant_lock
                permission_reads = []
                with _observe(LocalPermissionOperation.__enter__.__code__, sys.monitoring.events.PY_START,
                              lambda frame: permission_reads.append(frame.f_locals["self"])):
                    current_snapshot = await permission_action({"kind": "ReadPermissionSnapshot", "strict": True})
                assert current_snapshot.payload == permission_snapshot.payload
                assert len(permission_reads) == 1
                permission_read = permission_reads[0]
                assert permission_read.source is permission_store
                assert permission_read.observations["acquisition"] and permission_read.observations["raw"]
                assert permission_read.retired and permission_read.retirement_error is None
                assert not permissions._reads and not permissions._unresolved
                assert await permission_action({"kind": "ListSessionApprovals", "profile_id": "default"}) == [
                    ("agent:builtin", "Calculator")
                ]
                if outcome == "accepted":
                    with pytest.raises(ConsoleDisplayPhaseRefused):
                        original_full.require_current()
                    projection = await visible_projection("visible")
                    assert projection.source_revision == 1
                    assert len(full_reads) == 2
                    assert full_reads[-1].sources is backend.domain.sources
                    assert full_reads[-1].effect.completed.result() is True
                    assert full_reads[-1].physically_retired
                else:
                    assert not full_reads
                    assert backend.presentation.view_lifecycle._current is None
                assert display._full is None and not display._preparation_reads
            else:
                assert [row.kind for row in replies(3)] == ["failed"]
        if not accepted:
            assert backend.domain.sources is original_sources
            assert backend.runtime._execution_service._display_owner.sources is original_sources
            assert profile._selection_cache is before
            assert not backend.controls._intake_open
        assert not backend.controls.admitted_effects
    finally:
        await backend.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("retired", [True, False])
async def test_rail_receipt_failure_joins_independent_retirement(retired):
    """Transport control: failed Applied never retires the actual operation."""
    from types import SimpleNamespace

    from tldw_chatbook.Chat.console_execution_process import ConsoleChannelLost
    from tldw_chatbook.Chat.console_execution_rail_preferences import _settle

    loop = asyncio.get_running_loop()
    receipt = SimpleNamespace(**{name: loop.create_future() for name in ("received", "applied", "retired")})
    receipt.received.set_result(None)
    failure = RuntimeError("Original rail operation failed")
    task = asyncio.create_task(_settle(receipt))
    await asyncio.sleep(0)
    receipt.applied.set_exception(failure)
    await asyncio.sleep(0)
    assert not task.done()
    receipt.retired.set_result(retired)
    with pytest.raises(RuntimeError) as caught:
        await task
    if retired:
        assert caught.value is failure
    else:
        assert type(caught.value) is ConsoleChannelLost


@pytest.mark.asyncio
async def test_rail_receipt_repeated_cancellation_retains_admitted_operation():
    from types import SimpleNamespace

    from tldw_chatbook.Chat.console_execution_rail_preferences import _settle

    loop = asyncio.get_running_loop()
    receipt = SimpleNamespace(**{name: loop.create_future() for name in ("received", "applied", "retired")})
    task = asyncio.create_task(_settle(receipt))
    await asyncio.sleep(0)
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done() and not receipt.received.cancelled()
    receipt.received.set_result(None)
    receipt.applied.set_result({"source_revision": 1})
    await asyncio.sleep(0)
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done() and not receipt.retired.cancelled()
    receipt.retired.set_result(True)
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_rail_admission_refusal_does_not_invent_retirement():
    from types import SimpleNamespace

    from tldw_chatbook.Chat.console_execution_process import ConsoleRemoteFailure
    from tldw_chatbook.Chat.console_execution_rail_preferences import _settle

    loop = asyncio.get_running_loop()
    receipt = SimpleNamespace(**{name: loop.create_future() for name in ("received", "applied", "retired")})
    failure = ConsoleRemoteFailure("command_failed", "admission")
    receipt.received.set_exception(failure)
    receipt.applied.set_exception(failure)
    with pytest.raises(ConsoleRemoteFailure) as caught:
        await _settle(receipt)
    assert caught.value is failure
    assert receipt.applied.exception() is failure
    assert not receipt.retired.done()


@pytest.mark.asyncio
@pytest.mark.parametrize("owner", ["_project_modal", "_project_modal_transition"])
async def test_rail_save_refuses_preexisting_modal_before_any_command_or_write(owner):
    """Transport refusal only; no fake storage or successful native receipt."""
    from types import SimpleNamespace

    from tldw_chatbook.Chat.console_execution_process import ConsoleChannelLost, ConsoleExecutionClient
    from tldw_chatbook.Chat.console_execution_rail_preferences import publish_rail_preference

    app = SimpleNamespace(console_runtime=None)
    client = ConsoleExecutionClient(app)
    app.console_execution_client = client
    client._loop = asyncio.get_running_loop()
    setattr(client, owner, object())
    with pytest.raises(ConsoleChannelLost, match="unavailable"):
        await publish_rail_preference(SimpleNamespace(app_instance=app), "scope", {"workspace_open": True})
    assert not client._requests and not client._workspace_operations
    assert getattr(client, "_rail_publication", None) is None
    assert client._channel_error is None and client._source_revision == 0


@private_profile_test
@pytest.mark.asyncio
async def test_original_parent_rail_worker_publishes_to_same_contained_backend(request):
    """Exercise original writer, actual child replies and separate native proof."""
    import threading
    import time
    from contextlib import ExitStack

    from Tests.Chat.console_execution_mount_wait import wait_for_selected_composer
    from Tests.Chat.test_console_control_trace import _observe
    from Tests.Chat.test_console_execution_backend_process import _ObservedStderr
    from tldw_chatbook import config
    from tldw_chatbook.app import TldwCli
    from tldw_chatbook.Chat import console_execution_rail_preferences as rail
    from tldw_chatbook.Chat.console_execution_process import ConsoleExecutionClient
    from tldw_chatbook.Chat.console_rail_state import (
        build_console_rail_preference_key, coerce_console_rail_preferences,
        serialize_console_rail_stored_preferences,
    )
    from tldw_chatbook.Constants import TAB_CHAT, TAB_HOME
    from tldw_chatbook.UI.Navigation.main_navigation import NavigateToScreen
    from tldw_chatbook.UI.Screens.chat_screen import ChatScreen

    assert config.save_settings_to_cli_config({
        "first_run": {"setup_completed": True},
        "splash_screen": {"enabled": False},
        "general": {"default_tab": "home"},
        "chat_defaults": {"provider": "llama_cpp", "model": "local-model"},
        "api_settings.llama_cpp": {
            "api_url": "http://127.0.0.1:9099", "model": "local-model",
        },
    })
    app = TldwCli(console_execution="process")
    observed = []
    stderr_tail = bytearray()

    def observe_child(frame):
        caller = frame.f_back
        if (caller.f_code is ConsoleExecutionClient._launch.__code__
            and caller.f_locals.get("self") is app.console_execution_client):
            process = frame.f_locals["self"]
            process.stderr = _ObservedStderr(process.stderr, stderr_tail)

    def report_backend_failure_locations():
        # The existing worker emits these safe frame locations to its original
        # stderr sink. The diagnostic can finish draining after app shutdown.
        for match in re.finditer(
            rb"Console backend \w+ \w+ failed \(\w+\); frames=[\w: <>.-]+",
            bytes(stderr_tail),
        ):
            print(match.group().decode("ascii"))

    request.addfinalizer(report_backend_failure_locations)

    def writer_entered(frame):
        observed.append(frame.f_locals["scope"])

    observers = ExitStack()
    request.addfinalizer(observers.close)
    observers.enter_context(_observe(
        subprocess.Popen.__init__.__code__, sys.monitoring.events.PY_RETURN, observe_child,
    ))
    async with app.run_test(size=(120, 40)):
        await app._initial_screen_setup_task
        await app._console_execution_start_task
        client = app.console_execution_client
        assert app.console_runtime is None and client._view is None
        original_process, original_tree = client._process, client._tree
        # Schedule the same threaded preference worker while its real Chat
        # owns the gesture. Textual cancels workers launched on a removed screen.
        await app.handle_screen_navigation(NavigateToScreen(TAB_CHAT))
        async with asyncio.timeout(30):
            await wait_for_selected_composer(app)
        screen = app.screen
        assert type(screen) is ChatScreen
        assert client._view is screen and screen.app is app
        original_revision = client._source_revision
        key = build_console_rail_preference_key(layout_scope="global").value
        prior = screen._console_rail_state_config().get(key)
        requested_open = not coerce_console_rail_preferences(prior).workspace_open
        with _observe(rail._write_rail_preference.__code__, sys.monitoring.events.PY_START, writer_entered):
            before = set(app.workers)
            # The original setter updates the resident preference before its
            # worker deliberately reads the latest value under the write lock.
            screen._set_console_rail_preference(
                section_updates={"workspace": requested_open}, _reconcile=False,
            )
            value = serialize_console_rail_stored_preferences(
                app.app_config["console"]["rail_state"][key],
            )
            assert value["workspace_open"] is requested_open
            worker, = (
                item for item in app.workers
                if item not in before and item.node is screen
                and item.name == "_save_console_rail_preferences"
            )
            await worker.wait()
        assert len(observed) == 1
        scope, = observed
        assert scope.task.done() and scope.task.result() is True
        assert scope.writer.done() and scope.writer.result() is True
        assert scope.writer_retired and rail._writer_retired(scope)
        assert scope.writer_state.thread is not threading.current_thread()
        assert scope.writer_state.source is config
        assert dict(scope.value) == value
        assert all(rail._positive(receipt) for receipt in (scope.fence, scope.refresh, scope.applied))
        assert client._source_revision == original_revision + 1 and client._intake_open
        assert getattr(client, "_rail_publication", None) is None
        assert client._process is original_process and client._tree is original_tree
        assert config.load_settings()["console"]["rail_state"][key] == value

        await app.handle_screen_navigation(NavigateToScreen(TAB_HOME))
        async with asyncio.timeout(30):
            while client._view is not None:
                await asyncio.sleep(0.01)
        assert app.current_tab == TAB_HOME
        # A fresh permission command checks the same source after leaving Chat.
        receipt = client.submit("PermissionAction", {
            "operation": {"kind": "ReadPermissionSnapshot", "strict": True},
        })
        await receipt.received
        await receipt.applied
        assert await receipt.retired is True
    # run_test teardown owns the original App drain. This bounded join observes
    # that retained shutdown; it must not start a replacement cleanup owner.
    shutdown = client._shutdown_task
    assert shutdown is not None and app._console_runtime_shutdown_task.done()
    assert app._console_runtime_shutdown_task.result() is None
    assert await client.shutdown(deadline=time.monotonic() + 30) is True
    assert client._shutdown_task is shutdown and client.physically_closed
