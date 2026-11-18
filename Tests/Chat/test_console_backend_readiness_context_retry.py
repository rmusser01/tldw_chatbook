"""Retry only an actual readiness read's positively settled display drift."""

import asyncio
import copy
import sys
import threading
from dataclasses import replace
from types import CodeType

import pytest

from Tests.private_profile import private_profile_test


@private_profile_test
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "change", [
        "once", "continuous", "source", "hidden", "uncertain",
        "agent_once", "agent_continuous", "persisted", "settings",
        "bridge", "binding", "settings_unrevisioned", "agent_uncertain",
        "persisted_hidden", "persisted_bridge", "persisted_uncertain",
    ]
)
async def test_readiness_context_retry_requires_original_settlement(request, change):
    from Tests.Chat.test_console_control_trace import _observe
    from Tests.Chat.test_console_execution_profile import _settled_profile
    from tldw_chatbook import config
    from tldw_chatbook.Chat.console_backend_readiness import (
        ConsoleBackendReadiness,
        ConsoleBackendReadinessChanged,
    )
    from tldw_chatbook.Chat.console_control_dispatch import ConsoleLocalControls
    from tldw_chatbook.Chat.console_domain_display import ConsoleDomainDisplayOwner
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
    from tldw_chatbook.Chat.console_agent_bridge import AgentLiveSnapshot
    from tldw_chatbook.DB.base_db import run_owned_db_call

    path, directory, *_ = _settled_profile()
    profile = open_console_backend_profile(
        expected_config_path=path,
        expected_data_directory=directory,
    )
    backend = ConsoleExecutionBackend(
        profile, profile_epoch="profile", service_epoch="context-retry"
    )
    original_sources = None
    original_bridge = None
    effects, reads, full_reads, marker_effects = [], [], [], []
    changed = 0
    publication_changed = False
    old_marker_selection = None
    old_marker_refused = False
    initial_at = None
    read_code = next(
        item
        for item in ConsoleBackendReadiness.warm.__code__.co_consts
        if type(item) is CodeType and item.co_name == "read"
    )
    loop = asyncio.get_running_loop()

    def read_entry(frame):
        if frame.f_locals.get("self") is not backend.presentation.readiness:
            return
        if reads:
            assert all(
                effect.completed.done() and effect.completed.result() is True
                for effect in effects
            )
            if len(reads) < 2 or not change.startswith("persisted"):
                assert backend.presentation.readiness.at == initial_at
        reads.append(True)

    def effect_entry(frame):
        nonlocal publication_changed, old_marker_refused
        effect = frame.f_locals.get("effect")
        if effect is not None and effect.owner is backend.presentation.readiness:
            effects.append(effect)
        if effect is not None and effect.kind == "transcript_marker_read":
            marker_effects.append(effect)
        if (
            effect is None or effect.owner is not backend.presentation.readiness
            or len(reads) != 2 or old_marker_selection is None or old_marker_refused
        ):
            return
        # Observe the actual effect's return: both readiness reads have now
        # positively retired, while the old marker selection still refuses.
        assert all(item.completed.result() is True for item in effects)
        source = backend.controls._settings_command_tasks[asyncio.current_task()]
        with pytest.raises(RuntimeError, match="^Console marker publication changed$"):
            backend.presentation.project_transcript(source.identity, old_marker_selection, ())
        old_marker_refused = True
        if change == "persisted_hidden":
            backend.presentation.view_lifecycle.admit("readiness-visit", 1, "hidden")
        elif change == "persisted_bridge":
            successor = copy.copy(original_bridge)
            backend.runtime._agent_bridge = backend.controller._agent_bridge = successor
        else:
            return
        publication_changed = True

    def config_entry(frame):
        nonlocal changed
        if frame.f_back.f_code is not read_code or (
            changed and change not in {"continuous", "agent_continuous"}
        ):
            return
        settled = threading.Event()
        errors = []

        def mutate():
            nonlocal changed, old_marker_selection
            try:
                if change in {"agent_once", "agent_continuous", "agent_uncertain"}:
                    original_bridge.begin_setup_phase(conversation_id)
                    if change != "agent_continuous":
                        original_bridge._publish_live(
                            conversation_id, run_id, AgentLiveSnapshot(status="running"),
                            primary=True,
                        )
                elif change.startswith("persisted"):
                    marker = backend.presentation._marker_projection
                    old_marker_selection = marker, marker._selection()
                    backend.store.publish_first_persisted_conversation(session.id, conversation_id)
                elif change == "settings":
                    backend.store.replace_session_settings(
                        session.id, replace(session.settings, temperature=0.5),
                    )
                elif change == "settings_unrevisioned":
                    session.settings = replace(session.settings, temperature=0.5)
                elif change == "bridge":
                    successor = copy.copy(original_bridge)
                    backend.runtime._agent_bridge = successor
                    backend.controller._agent_bridge = successor
                elif change == "binding":
                    backend.store.rebind_persisted_conversation(session.id, other_conversation_id)
                else:
                    backend.store._bump_display_projection_revision(session.id)
                changed += 1
                if change == "source":
                    backend.domain.sources = replace(original_sources)
                elif change == "hidden":
                    backend.presentation.view_lifecycle.admit(
                        "readiness-visit", 1, "hidden"
                    )
            except BaseException as error:
                errors.append(error)
            finally:
                settled.set()

        loop.call_soon_threadsafe(mutate)
        assert settled.wait(10), "Owner-loop context mutation did not run"
        if errors:
            raise errors[0]

    def negative_observation(frame):
        observation = frame.f_locals["self"]
        if (
            change in {"uncertain", "agent_uncertain"}
            and observation.read.creator is backend.presentation.readiness
        ):
            # Preserve the original negative receipt path; never substitute a
            # positive result or change the original confirmed() implementation.
            observation.invalid = True
        elif (
            change == "persisted_uncertain"
            and observation.read.creator is backend.presentation._marker_projection
        ):
            observation.invalid = True

    try:
        backend.start()
        original_sources = backend.domain.sources
        original_bridge = backend.runtime.agent_bridge
        session = backend.store.create_session(
            title="Readiness context",
            assistant_kind="generic",
            settings=ConsoleSessionSettings(
                provider="openai", model=None, source="user"
            ),
        )
        readiness = backend.presentation.readiness
        if change in {"agent_once", "agent_continuous", "agent_uncertain", "binding"} or change.startswith("persisted"):
            database = profile.storage.chachanotes
            conversation_id = await run_owned_db_call(
                database, database.add_conversation, {"title": "Readiness publication"},
            )
            assert conversation_id
            if not change.startswith("persisted"):
                backend.store.publish_first_persisted_conversation(session.id, conversation_id)
            if change == "binding":
                other_conversation_id = await run_owned_db_call(
                    database, database.add_conversation, {"title": "Other readiness publication"},
                )
                assert other_conversation_id
            if change in {"agent_once", "agent_uncertain"}:
                runs = backend.runtime._agent_runs_db
                run_id = await run_owned_db_call(
                    runs, runs.create_run, conversation_id=conversation_id, agent_kind="primary",
                )
        initial_at = readiness.at
        with (
            _observe(read_code, sys.monitoring.events.PY_START, read_entry),
            _observe(
                config.load_settings.__code__,
                sys.monitoring.events.PY_START,
                config_entry,
            ),
            _observe(
                ConsoleLocalControls._finish_effect.__code__,
                sys.monitoring.events.PY_RETURN,
                effect_entry,
            ),
            _observe(
                ConsoleDomainDisplayOwner.read.__code__,
                sys.monitoring.events.PY_START,
                lambda _frame: full_reads.append(True),
            ),
            _observe(
                _ReadResources.confirmed.__code__,
                sys.monitoring.events.PY_START,
                negative_observation,
            ),
        ):
            receipt = backend.dispatch(
                ConsoleWireMessage(
                    WireIdentity(
                        "profile",
                        "context-retry",
                        1,
                        "context-read",
                        session.id,
                        session.incarnation_id,
                        backend.controls.source_revision,
                    ),
                    "command",
                    "ReadProjection",
                    {},
                )
            )
            if change in {"once", "agent_once", "persisted", "settings"}:
                await asyncio.shield(receipt.applied)
                assert readiness.context_key == readiness._context_key(session)
                assert readiness.agent_key == readiness._agent_key(session)
                assert readiness.key == readiness._key(session, session.settings, backend.controls.source_revision)
                assert readiness._context_facts is not None
                # A real saved-marker read can consume the existing readiness
                # TTL; its final warm then performs one ordinary fresh read.
                assert len(reads) in ({2, 3} if change == "persisted" else {2})
                if change == "persisted":
                    assert old_marker_refused
                    assert len(marker_effects) == 1
                    assert marker_effects[0].completed.result() is True
                    assert marker_effects[0].physically_retired is True
            elif change.startswith("persisted_"):
                with pytest.raises(Exception) as caught:
                    await asyncio.shield(receipt.applied)
                assert type(caught.value) is not ConsoleBackendReadinessChanged
                assert len(reads) == 2
                assert old_marker_refused
                assert len(marker_effects) == (1 if change == "persisted_uncertain" else 0)
                if change != "persisted_uncertain":
                    assert publication_changed
            elif change in {"continuous", "agent_continuous"}:
                with pytest.raises(ConsoleBackendReadinessChanged) as caught:
                    await asyncio.shield(receipt.applied)
                assert readiness.require_retired_context_change(caught.value) is True
                assert len(reads) == 2
                assert readiness.at == initial_at and readiness._context_facts is None
            else:
                with pytest.raises(Exception) as caught:
                    await asyncio.shield(receipt.applied)
                assert type(caught.value) is not ConsoleBackendReadinessChanged
                assert len(reads) == 1
                assert readiness.at == initial_at and readiness._context_facts is None
            if change in {"uncertain", "agent_uncertain", "persisted_uncertain"}:
                assert await asyncio.shield(receipt.retired) is False
                assert (marker_effects if change == "persisted_uncertain" else effects)[-1].completed.result() is False
                assert not backend.controls._intake_open
            else:
                assert await asyncio.shield(receipt.retired) is True
                assert all(effect.completed.result() is True for effect in effects)
            assert len(full_reads) == 1
            assert not readiness.pending and not readiness.reads
    finally:
        if original_sources is not None:
            backend.domain.sources = original_sources
        if original_bridge is not None:
            backend.runtime._agent_bridge = original_bridge
            backend.controller._agent_bridge = original_bridge
        # A deliberately uncertain original effect must remain negative; close
        # still performs its original physical cleanup and must not certify it.
        if change in {"uncertain", "agent_uncertain", "persisted_uncertain"} and effects and effects[-1].completed.done():
            with pytest.raises(RuntimeError):
                await backend.aclose()
        else:
            await backend.aclose()
