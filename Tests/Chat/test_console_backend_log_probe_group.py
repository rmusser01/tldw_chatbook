"""The original metadata probe shares only its own finite AgentRuns handle."""

import asyncio
import sqlite3
import sys
import threading
from types import CodeType, MethodType

import pytest

from Tests.private_profile import private_profile_test


@private_profile_test
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "outcome",
    ["available", "missing", "read_error", "uncertain", "changed_dispatch", "revoked"],
)
async def test_readiness_log_probe_groups_original_metadata(request, outcome):
    from Tests.Chat.test_console_control_trace import _observe
    from Tests.Chat.test_console_execution_profile import _settled_profile
    from tldw_chatbook.Chat.console_backend_readiness import ConsoleBackendReadiness
    from tldw_chatbook.Chat.console_execution_backend import ConsoleExecutionBackend
    from tldw_chatbook.Chat.console_execution_profile import (
        open_console_backend_profile,
    )
    from tldw_chatbook.Chat.console_execution_protocol import (
        ConsoleWireMessage,
        WireIdentity,
    )
    from tldw_chatbook.Chat.console_execution_retirement import _ReadResources
    from tldw_chatbook.Chat.console_session_settings import ConsoleSessionSettings
    from tldw_chatbook.DB.base_db import run_owned_db_call

    path, directory, *_ = _settled_profile()
    profile = open_console_backend_profile(
        expected_config_path=path,
        expected_data_directory=directory,
    )
    backend = ConsoleExecutionBackend(
        profile, profile_epoch="profile", service_epoch="log-group"
    )
    read_code = next(
        item
        for item in ConsoleBackendReadiness.warm.__code__.co_consts
        if type(item) is CodeType and item.co_name == "read"
    )
    probe_code = next(
        item
        for item in read_code.co_consts
        if type(item) is CodeType and item.co_name == "log_probe"
    )
    seen, closed, file_entries, overrides = [], [], [], []
    injected = False
    main_thread = threading.get_ident()

    def in_probe(frame):
        while frame is not None:
            if frame.f_code is probe_code:
                return True
            frame = frame.f_back
        return False

    def metadata_return(frame):
        if frame.f_locals.get("self") is bridge and in_probe(frame):
            connection = getattr(database._thread_local, "conn", None)
            if outcome == "changed_dispatch":
                assert connection is None  # Original fallback closes each read.
                return
            assert connection is not None
            assert threading.get_ident() != main_thread
            seen.append(connection)

    def owning_entry(frame):
        nonlocal injected
        if frame.f_locals.get("self") is not bridge or not in_probe(frame):
            return
        if outcome != "changed_dispatch":
            seen.append(database._thread_local.conn)
        if outcome == "read_error" and not injected:
            injected = True
            raise OSError("original optional metadata read failed")

    def authority_entry(frame):
        if frame.f_locals.get("self") is not bridge or not in_probe(frame):
            return
        from tldw_chatbook.Backup_Recovery import storage_admission
        from tldw_chatbook.DB import base_db

        assert getattr(database._thread_local, "conn", None) is None
        assert id(database) not in getattr(base_db._OPERATION_OWNERS, "ids", ())
        operation = getattr(storage_admission._operation_local, "operation", None)
        assert operation is None or operation.participant.repository() is not database
        file_entries.append(True)
        if outcome == "revoked":
            bridge.forget_session_file_authority(session.id)

    def close_entry(frame):
        if frame.f_locals.get("self") is database and in_probe(frame):
            connection = getattr(database._thread_local, "conn", None)
            if connection is not None:
                closed.append(connection)

    def confirmation(frame):
        observation = frame.f_locals["self"]
        if (
            outcome == "uncertain"
            and closed
            and observation.read.creator is backend.presentation.readiness
        ):
            # Keep real cleanup, but make its original receipt explicitly
            # uncertain. Optional availability must not manufacture retirement.
            observation.invalid = True

    try:
        backend.start()
        bridge, database = backend.runtime._agent_bridge, backend.runtime._agent_runs_db
        conversation = await run_owned_db_call(
            profile.storage.chachanotes,
            profile.storage.chachanotes.add_conversation,
            {"title": "Original log metadata"},
        )
        run_id = await run_owned_db_call(
            database,
            database.create_run,
            conversation_id=conversation,
            agent_kind="primary",
        )
        session = backend.store.create_session(
            title="Log metadata",
            assistant_kind="generic",
            settings=ConsoleSessionSettings(
                provider="openai", model=None, source="user"
            ),
        )
        backend.store.publish_first_persisted_conversation(session.id, conversation)
        if outcome in {"available", "revoked"}:
            from functools import partial
            from tldw_chatbook.Agents.run_log import RunLogWriter

            scratch = backend.controller._scratch_spaces.snapshot(session.id)
            lease = partial(backend.controller._scratch_spaces.lease, scratch)
            writer = RunLogWriter(root=scratch.root, access_scope=lease)
            try:
                writer.bind(run_id)
                assert writer.append(
                    run_id=run_id, kind="primary", type="model", content="Complete"
                )
            finally:
                writer.close()
            bridge._remember_run_log_authority(
                run_id,
                scratch.root,
                session_id=session.id,
                access_scope=lease,
                conversation_id=conversation,
            )
        if outcome == "changed_dispatch":
            original = type(bridge).latest_primary_run_id

            def custom_latest(self, conversation_id):
                overrides.append(True)
                return original(self, conversation_id)

            bridge.latest_primary_run_id = MethodType(custom_latest, bridge)
        # This original main-thread handle is independent of the read worker.
        independent = database._held_connection()
        with (
            _observe(
                type(bridge).resolve_run_log_target.__code__,
                sys.monitoring.events.PY_RETURN,
                metadata_return,
            ),
            _observe(
                type(bridge)._run_log_authority_for.__code__,
                sys.monitoring.events.PY_START,
                authority_entry,
            ),
            _observe(
                type(bridge)._owning_run_id_for_log.__code__,
                sys.monitoring.events.PY_START,
                owning_entry,
            ),
            _observe(
                type(database).close.__code__,
                sys.monitoring.events.PY_START,
                close_entry,
            ),
            _observe(
                _ReadResources.confirmed.__code__,
                sys.monitoring.events.PY_START,
                confirmation,
            ),
        ):
            receipt = backend.dispatch(
                ConsoleWireMessage(
                    WireIdentity(
                        "profile",
                        "log-group",
                        1,
                        "log-probe",
                        session.id,
                        session.incarnation_id,
                        backend.controls.source_revision,
                    ),
                    "command",
                    "ReadProjection",
                    {},
                )
            )
            try:
                if outcome == "uncertain":
                    with pytest.raises(RuntimeError, match="retirement"):
                        await asyncio.shield(receipt.applied)
                else:
                    await asyncio.shield(receipt.applied)
            finally:
                assert await asyncio.shield(receipt.retired) is (outcome != "uncertain")
        assert closed
        if outcome == "changed_dispatch":
            assert overrides and len(closed) == 2 and closed[0] is not closed[1]
        else:
            assert len(seen) == 2
            assert all(connection is seen[0] for connection in seen)
            assert closed == [seen[0]]
        if outcome == "read_error":
            assert injected and not file_entries
        else:
            assert file_entries  # Every file callback entered after DB cleanup.
        for connection in closed:
            with pytest.raises(sqlite3.ProgrammingError):
                sqlite3.Connection.in_transaction.__get__(connection)
        assert database._thread_local.conn is independent
        assert independent.execute("SELECT 1").fetchone()[0] == 1
        if outcome != "uncertain":
            assert backend.presentation.readiness.agent_log_available is (
                outcome == "available"
            )
    finally:
        if outcome == "changed_dispatch" and "bridge" in locals():
            vars(bridge).pop("latest_primary_run_id", None)
        if outcome == "uncertain" and closed:
            with pytest.raises(RuntimeError):
                await backend.aclose()
        else:
            await backend.aclose()
