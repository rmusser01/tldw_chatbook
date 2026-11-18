"""Actual retired readiness inputs survive display-only invalidation."""

import asyncio
import sys
import time
from types import CodeType, SimpleNamespace

import pytest

from Tests.private_profile import private_profile_test


@private_profile_test
@pytest.mark.asyncio
async def test_backend_readiness_reuses_native_facts_without_renewing_age(
    request,
    monkeypatch,
):
    from Tests.Chat.test_console_control_trace import _observe
    from Tests.Chat.test_console_execution_profile import _settled_profile
    from tldw_chatbook import config
    from tldw_chatbook.Chat import console_backend_readiness as readiness_source
    from tldw_chatbook.Chat.console_execution_backend import ConsoleExecutionBackend
    from tldw_chatbook.Chat.console_execution_profile import (
        open_console_backend_profile,
    )
    from tldw_chatbook.Chat.console_execution_protocol import (
        ConsoleWireMessage,
        WireIdentity,
    )
    from tldw_chatbook.Chat.console_session_settings import ConsoleSessionSettings
    from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
    from tldw_chatbook.Chat.provider_usage import ProviderUsage
    from tldw_chatbook.UI.Console_Modules import pricing_display

    path, directory, *_ = _settled_profile()
    profile = open_console_backend_profile(
        expected_config_path=path,
        expected_data_directory=directory,
    )
    backend = ConsoleExecutionBackend(
        profile,
        profile_epoch="profile",
        service_epoch="readiness-reuse",
    )
    # Control only this display owner's clock to test its existing one-second
    # boundary deterministically. Native work, source checks and retirement
    # remain real; these are work-count controls, not latency measurements.
    clock = [time.monotonic()]
    monkeypatch.setattr(
        readiness_source, "time", SimpleNamespace(monotonic=lambda: clock[0])
    )
    read_code = next(
        item
        for item in readiness_source.ConsoleBackendReadiness.warm.__code__.co_consts
        if type(item) is CodeType and item.co_name == "read"
    )
    native_reads = []
    fail_config = [False]
    fail_pricing = [False]
    pricing_reads = []

    def pricing_entry(_frame):
        pricing_reads.append(True)
        if fail_pricing[0]:
            fail_pricing[0] = False
            raise OSError("optional price read failed")

    def config_entry(frame):
        if fail_config[0] and frame.f_back.f_code is read_code:
            fail_config[0] = False
            raise OSError("readiness configuration read failed")

    try:
        backend.start()
        session = backend.store.create_session(
            title="Display invalidation",
            assistant_kind="generic",
            settings=ConsoleSessionSettings(
                provider="openai", model=None, source="user"
            ),
        )
        readiness = backend.presentation.readiness

        async def projection():
            sequence = backend._sequence + 1
            receipt = backend.dispatch(
                ConsoleWireMessage(
                    WireIdentity(
                        "profile",
                        "readiness-reuse",
                        sequence,
                        f"projection-{sequence}",
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
                return await asyncio.shield(receipt.applied)
            finally:
                assert await asyncio.shield(receipt.retired) is True

        with (
            _observe(
                read_code,
                sys.monitoring.events.PY_START,
                lambda _frame: native_reads.append(True),
            ),
            _observe(
                config.load_settings.__code__,
                sys.monitoring.events.PY_START,
                config_entry,
            ),
            _observe(pricing_display.prepare_pricing_display.__code__,
                     sys.monitoring.events.PY_START, pricing_entry),
        ):
            await projection()
            assert len(native_reads) == 1
            facts, at = readiness._context_facts, readiness.at
            summary = readiness.context_summary
            selection = readiness.send_selection
            assert facts is not None
            assert len(pricing_reads) == 1
            assert not readiness.cost.unavailable

            backend.store._bump_display_projection_revision(session.id)
            backend.store.set_session_one_shot_prefill(session.id, "Actual prefill")
            await projection()
            assert len(native_reads) == 1
            assert readiness._context_facts is facts and readiness.at == at
            assert readiness.context_summary is not summary
            assert readiness.context_key == readiness._context_key(session)
            assert readiness.send_selection is not selection
            assert not readiness.send_selection["project_bindings_eligible"]
            assert len(pricing_reads) == 1

            # Late usage changes the display revision, not request payload.
            # Actual pricing must update without another config/catalog read.
            message = backend.store.append_message(
                session.id, role=ConsoleMessageRole.ASSISTANT, content="Paid answer",
            )
            usage = ProviderUsage(
                uncached_input=1_000_000, output=1_000_000,
                provider="anthropic", model="claude-sonnet-4-6",
            )
            backend.store.set_message_usage(message.id, usage)
            payload_revision = backend.store.payload_revision(session.id)
            await projection()
            assert len(native_reads) == 1 and len(pricing_reads) == 1
            assert readiness.cost.facts.snapshot.total_usd == 18.0
            assert readiness.context_summary.cost is not None
            backend.store.set_message_usage(message.id, ProviderUsage(
                uncached_input=2_000_000, output=2_000_000,
                provider="anthropic", model="claude-sonnet-4-6",
            ))
            await projection()
            assert backend.store.payload_revision(session.id) == payload_revision
            assert readiness.cost.facts.snapshot.total_usd == 36.0
            assert len(native_reads) == 1 and len(pricing_reads) == 1

            clock[0] += 1.0
            await projection()
            assert len(native_reads) == 2
            assert readiness.at == clock[0] and readiness._context_facts is not facts

            # A failed new native read cannot publish reusable facts, even
            # though its original resource owners positively retire.
            clock[0] += 1.0
            fail_config[0] = True
            with pytest.raises(OSError, match="configuration read failed"):
                await projection()
            assert len(native_reads) == 3
            assert readiness._context_facts is None
            assert not readiness.pending and not readiness.reads
            await projection()
            assert len(native_reads) == 4 and readiness._context_facts is not None

            previous_cost = readiness.context_summary.cost
            clock[0] += 1.0
            fail_pricing[0] = True
            await projection()  # Real successful Applied + Retired despite optional failure.
            assert readiness.cost.unavailable
            assert readiness.context_summary.cost == previous_cost
            reads = len(native_reads), len(pricing_reads)
            backend.store._bump_display_projection_revision(session.id)
            await projection()
            assert (len(native_reads), len(pricing_reads)) == reads
            assert readiness.context_summary.cost == previous_cost
            clock[0] += 1.0
            await projection()
            assert not readiness.cost.unavailable
            assert readiness.cost.facts.snapshot.total_usd == 36.0
    finally:
        await backend.aclose()
