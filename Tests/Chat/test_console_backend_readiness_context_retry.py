"""Retry only an actual readiness read's positively settled display drift."""

import asyncio
import sys
import threading
from dataclasses import replace
from types import CodeType

import pytest

from Tests.private_profile import private_profile_test


@private_profile_test
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "change", ["once", "continuous", "source", "hidden", "uncertain"]
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

    path, directory, *_ = _settled_profile()
    profile = open_console_backend_profile(
        expected_config_path=path,
        expected_data_directory=directory,
    )
    backend = ConsoleExecutionBackend(
        profile, profile_epoch="profile", service_epoch="context-retry"
    )
    original_sources = None
    effects, reads, full_reads = [], [], []
    changed = 0
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
            assert backend.presentation.readiness.at == initial_at
        reads.append(True)

    def effect_entry(frame):
        effect = frame.f_locals.get("effect")
        if effect is not None and effect.owner is backend.presentation.readiness:
            effects.append(effect)

    def config_entry(frame):
        nonlocal changed
        if frame.f_back.f_code is not read_code or (changed and change != "continuous"):
            return
        settled = threading.Event()
        errors = []

        def mutate():
            nonlocal changed
            try:
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
            change == "uncertain"
            and observation.read.creator is backend.presentation.readiness
        ):
            # Preserve the original negative receipt path; never substitute a
            # positive result or change the original confirmed() implementation.
            observation.invalid = True

    try:
        backend.start()
        original_sources = backend.domain.sources
        session = backend.store.create_session(
            title="Readiness context",
            assistant_kind="generic",
            settings=ConsoleSessionSettings(
                provider="openai", model=None, source="user"
            ),
        )
        readiness = backend.presentation.readiness
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
                sys.monitoring.events.PY_START,
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
            if change == "once":
                await asyncio.shield(receipt.applied)
                assert readiness.context_key == readiness._context_key(session)
                assert readiness._context_facts is not None
                assert len(reads) == 2
            elif change == "continuous":
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
            if change == "uncertain":
                assert await asyncio.shield(receipt.retired) is False
                assert effects[-1].completed.result() is False
                assert not backend.controls._intake_open
            else:
                assert await asyncio.shield(receipt.retired) is True
                assert all(effect.completed.result() is True for effect in effects)
            assert len(full_reads) == 1
            assert not readiness.pending and not readiness.reads
    finally:
        if original_sources is not None:
            backend.domain.sources = original_sources
        # A deliberately uncertain original effect must remain negative; close
        # still performs its original physical cleanup and must not certify it.
        if change == "uncertain" and effects and effects[-1].completed.done():
            with pytest.raises(RuntimeError):
                await backend.aclose()
        else:
            await backend.aclose()
