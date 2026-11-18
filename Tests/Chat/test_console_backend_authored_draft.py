"""Original authored draft application precedes a current visible wake poke."""

import asyncio
import sys

import pytest

from Tests.private_profile import private_profile_test


@private_profile_test
@pytest.mark.asyncio
async def test_backend_empty_authored_draft_retains_handoff_order_and_view_currentness(
    request,
):
    from Tests.Chat.test_console_control_trace import _observe
    from Tests.Chat.test_console_execution_profile import _settled_profile
    from tldw_chatbook.Chat.console_backend_presentation import (
        ConsoleBackendPresentation,
    )
    from tldw_chatbook.Chat.console_execution_backend import ConsoleExecutionBackend
    from tldw_chatbook.Chat.console_execution_profile import (
        open_console_backend_profile,
    )
    from tldw_chatbook.Chat.console_execution_protocol import (
        ConsoleWireMessage,
        WireIdentity,
    )
    from tldw_chatbook.Chat.console_fleet_wake import ConsoleFleetWakeCoordinator
    from tldw_chatbook.Chat.console_received_intent import (
        ConsoleReceivedIntentInputMixin,
    )
    from tldw_chatbook.Chat.console_session_settings import ConsoleSessionSettings

    path, directory, *_ = _settled_profile()
    profile = open_console_backend_profile(
        expected_config_path=path,
        expected_data_directory=directory,
    )
    backend = ConsoleExecutionBackend(
        profile, profile_epoch="profile", service_epoch="draft"
    )
    events = []
    try:
        backend.start()
        session = backend.store.create_session(
            title="Authored draft",
            assistant_kind="generic",
            settings=ConsoleSessionSettings(
                provider="openai", model=None, source="user"
            ),
        )

        def message(operation, payload):
            sequence = backend._sequence + 1
            return ConsoleWireMessage(
                WireIdentity(
                    "profile",
                    "draft",
                    sequence,
                    f"draft-{sequence}",
                    session.id,
                    session.incarnation_id,
                    backend.controls.source_revision,
                ),
                "command",
                operation,
                payload,
            )

        mounted = backend.dispatch(
            message(
                "ReadProjection",
                {
                    "known_display_revision": None,
                    "view_checkpoint": {
                        "attachment_id": "actual-client",
                        "visit": 1,
                        "action": "mount",
                    },
                },
            )
        )
        try:
            await asyncio.shield(mounted.applied)
        finally:
            assert await asyncio.shield(mounted.retired) is True
        assert (
            backend.presentation._capture_composition(session)["draft_spend_run_active"]
            is False
        )

        def update(draft, *, expected=None):
            receipt = backend.dispatch(
                message(
                    "UpdateInputs",
                    {
                        "draft": draft,
                        "authored_token": [1, backend._sequence + 1],
                        "expected_settings_revision": expected,
                        "expected_settings": None,
                        "expected_attachment_ids": [],
                        "remove_attachment_ids": [],
                        "prefill": None,
                    },
                )
            )
            assert receipt.applied.done()
            assert receipt.applied.result() == session.draft_revision
            return receipt

        def handoff_returned(frame):
            if frame.f_locals["self"] is backend.store:
                events.append(("handoff", frame.f_locals["draft"]))

        def wake_started(frame):
            if frame.f_locals["self"] is backend.controller._fleet_wake:
                assert session.draft == ""
                assert events[-1] == ("handoff", "")
                events.append(("wake", ""))

        update("First draft")
        with (
            _observe(
                ConsoleReceivedIntentInputMixin._publish_session_draft_handoff.__code__,
                sys.monitoring.events.PY_RETURN,
                handoff_returned,
            ),
            _observe(
                ConsoleFleetWakeCoordinator.retry_soon.__code__,
                sys.monitoring.events.PY_START,
                wake_started,
            ),
        ):
            update("")
            assert events == [("handoff", ""), ("wake", "")]
            update(
                ""
            )  # A new authored token with unchanged text emits no compatibility event.
            assert events == [("handoff", ""), ("wake", ""), ("handoff", "")]
            update("Second draft")
            before = list(events)
            with pytest.raises(RuntimeError, match="input revision changed"):
                update("", expected=session.settings_revision + 1)
            assert events == before and session.draft == "Second draft"

            hidden = []

            def revoke_after_capture(frame):
                if frame.f_locals["self"] is backend.presentation:
                    # Original admission invalidates the captured token before
                    # the real input command applies; no producer is replaced.
                    hidden.append(
                        backend.presentation.view_lifecycle.admit(
                            "actual-client",
                            1,
                            "hidden",
                        )
                    )

            with _observe(
                ConsoleBackendPresentation.capture_display_reuse_checkpoint.__code__,
                sys.monitoring.events.PY_RETURN,
                revoke_after_capture,
            ):
                update("")
            assert events == before + [("handoff", "")]
            assert len(hidden) == 1 and session.draft == ""
            await backend.presentation.view_lifecycle.prepare(hidden[0])
            update("Hidden draft")
            before = list(events)
            update("")
            assert events == before + [("handoff", "")]
        assert backend.runtime.view is None and backend.controller.app is None
    finally:
        await backend.aclose()
