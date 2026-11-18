"""Real backend display reads retain the original character cache boundary."""

import asyncio
import json
import sys
import time
from types import SimpleNamespace

import pytest

from Tests.private_profile import private_profile_test


@private_profile_test
@pytest.mark.asyncio
async def test_backend_visible_character_reuses_only_current_retired_observation(
    request,
    monkeypatch,
):
    from Tests.Chat.test_console_control_trace import _observe
    from Tests.Chat.test_console_execution_profile import _settled_profile
    from tldw_chatbook.Chat.console_backend_view_lifecycle import (
        ConsoleBackendViewLifecycle,
    )
    from tldw_chatbook.Chat.console_control_dispatch import ConsoleLocalControls
    from tldw_chatbook.Chat.console_display_phases import ConsoleDisplayPhysicalOwner
    from tldw_chatbook.Chat.console_execution_backend import ConsoleExecutionBackend
    from tldw_chatbook.Chat.console_execution_profile import (
        open_console_backend_profile,
    )
    from tldw_chatbook.Chat.console_execution_protocol import (
        ConsoleWireMessage,
        WireIdentity,
    )
    from tldw_chatbook.Chat.console_session_settings import ConsoleSessionSettings
    from tldw_chatbook.UI.Console_Modules import character_context

    path, directory, *_ = _settled_profile()
    profile = open_console_backend_profile(
        expected_config_path=path,
        expected_data_directory=directory,
    )
    backend = ConsoleExecutionBackend(
        profile,
        profile_epoch="profile",
        service_epoch="character-reuse",
    )
    # Only the original character owner's clock is controlled. Every command,
    # native reader and resource retirement is real; this measures work counts.
    clock = [time.monotonic()]
    monkeypatch.setattr(
        character_context,
        "time",
        SimpleNamespace(monotonic=lambda: clock[0]),
    )
    phases, completed, counts = [], [], []

    def phase_returned(frame):
        full, phase = frame.f_locals["self"], frame.f_locals["phase"]
        if full.owner.runtime is backend.runtime:
            assert phase.joined and phase.retirement.confirmed
            phases.append(frame.f_locals["read"].__qualname__)

    def full_returned(frame):
        full = frame.f_locals["full"]
        if full.owner.runtime is backend.runtime:
            completed.append(full)

    def checkpoint_admitted(frame):
        lifecycle = frame.f_locals["self"]
        if lifecycle is backend.presentation.view_lifecycle:
            # Admission revokes reuse before its lifecycle task can run.
            assert lifecycle._display_prepared is None

    try:
        backend.start()
        session = backend.store.create_session(
            title="Character reuse",
            assistant_kind="generic",
            settings=ConsoleSessionSettings(
                provider="openai", model=None, source="user"
            ),
        )

        async def projection(label, action=None):
            before = len(phases)
            sequence = backend._sequence + 1
            receipt = backend.dispatch(
                ConsoleWireMessage(
                    WireIdentity(
                        "profile",
                        "character-reuse",
                        sequence,
                        f"projection-{sequence}",
                        session.id,
                        session.incarnation_id,
                        backend.controls.source_revision,
                    ),
                    "command",
                    "ReadProjection",
                    {}
                    if action is None
                    else {
                        "view_checkpoint": {
                            "attachment_id": "actual-client",
                            "visit": 1,
                            "action": action,
                        }
                    },
                )
            )
            try:
                await asyncio.shield(receipt.applied)
            finally:
                assert await asyncio.shield(receipt.retired) is True
            full = completed[-1]
            assert full.stamp is None and full.selection.command_source is not None
            assert full.logical.retired.done() and full.logical.retired.result() is None
            assert full.effect.completed.result() is True and full.physically_retired
            count = sum(
                name.startswith("ConsoleCharacterDisplayPhases.")
                for name in phases[before:]
            )
            counts.append((label, count))
            return count

        with (
            _observe(
                ConsoleBackendViewLifecycle.admit.__code__,
                sys.monitoring.events.PY_RETURN,
                checkpoint_admitted,
            ),
            _observe(
                ConsoleDisplayPhysicalOwner.native_phase.__code__,
                sys.monitoring.events.PY_RETURN,
                phase_returned,
            ),
            _observe(
                ConsoleLocalControls._finish_display_effect.__code__,
                sys.monitoring.events.PY_RETURN,
                full_returned,
            ),
        ):
            assert await projection("headless-cold") == 5
            assert await projection("headless-repeat") == 2
            assert await projection("mount", "mount") == 2
            owner = backend.runtime._execution_service._display_owner
            at = owner.character._presentation_scope_at
            assert await projection("visible-repeat") == 0
            assert owner.character._presentation_scope_at == at
            assert await projection("hidden", "hidden") == 2
            assert await projection("hidden-repeat") == 2
            assert await projection("visible-again", "visible") == 2
            assert await projection("visible-again-repeat") == 0
            clock[0] += character_context._CHARACTER_PRESENTATION_TTL_SECONDS
            assert await projection("expired") == 2
            backend.store.append_message(
                session.id, role="user", content="Changed payload"
            )
            assert await projection("payload-changed") == 2
            assert await projection("payload-repeat") == 0
            assert backend.runtime.view is None and backend.controller.app is None
    finally:
        request.node.user_properties.append(
            ("character_native_phase_counts", json.dumps(counts))
        )
        await backend.aclose()
