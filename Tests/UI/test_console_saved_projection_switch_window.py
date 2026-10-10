"""Original runtime draft CAS/projection preserves switch-window typing.

This isolates the publication boundary, not physical saving. Real commit outcome
and worker retirement are covered by the saved-acceptance integration controls.
"""

import pytest

from Tests.UI.test_console_approval_compact_layout import _wait_for_reconciled_console
from Tests.UI.test_console_native_chat_flow import _select_llamacpp_console
from Tests.UI.test_destination_shells import _build_test_app, _wait_for_selector
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
)
from tldw_chatbook.Chat.console_turn_context import ConsoleTurnCustodyRequest
from tldw_chatbook.Widgets.Console import ConsoleComposerBar


@pytest.mark.bootstrap_profile
@pytest.mark.asyncio
async def test_saved_fallback_projection_preserves_new_tab_settle_window_typing():
    app = _build_test_app()
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = "test-model"
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        await _wait_for_reconciled_console(console, pilot)
        _select_llamacpp_console(console)
        store = console._ensure_console_chat_store()
        runtime = console._console_runtime()
        controller = console._ensure_console_chat_controller()
        session_a = store.ensure_session(title="Chat A")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("accepted A draft")
        console._session._sync_console_session_draft()
        stash = composer.capture_draft_for_send()
        inputs = store.session_input_snapshot(session_a.id)
        assert stash is not None and stash.text == inputs.draft
        request = ConsoleTurnCustodyRequest(
            turn_id="saved-switch-projection",
            session_id=session_a.id,
            draft=inputs.draft,
            configuration=controller.resolve_turn_configuration_snapshot(session_a.id),
            _pressed_inputs=inputs,
            _pressed_stash=stash,
            _pressed_attachment_generation=runtime._attached_generation,
        )
        claim = store.claim_received_turn(
            session_a.id,
            request.turn_id,
            draft_revision=inputs.draft_revision,
            _allow_draft_change=True,
        )
        assert claim is not None
        record = None
        try:
            record = runtime._register_custody(
                request, store=store, received_claim=claim
            )
            # The existing switch test's coalescing guard holds the real swap.
            console._console_sync_in_progress = True
            try:
                await console._session._create_native_console_session_from_active_context()
                session_b_id = store.active_session_id
                assert session_b_id != session_a.id
                assert console._console_visible_draft_session_id == session_a.id
                switch = console._session._console_draft_switch_snapshot
                assert switch is not None and switch[0] == session_a.id
                assert switch[1] == inputs.draft
                composer.insert_text("B-only suffix")
                assert composer.draft_text() == inputs.draft + "B-only suffix"
                # The original observer attributes this pending suffix to B,
                # so A's exact CAS legitimately still succeeds.
                current = store.session_input_snapshot(session_a.id)
                assert current.draft == inputs.draft
                assert current.draft_revision == inputs.draft_revision
                assert store.commit_session_input_draft(inputs)
                assert store.session_draft(session_a.id) == ""
                runtime._project_received_input(record, draft_committed=True)
            finally:
                console._console_sync_in_progress = False
            console._session._sync_console_session_draft()
            assert console._console_visible_draft_session_id == session_b_id
            assert composer.draft_text() == "B-only suffix"
            assert store.session_draft(session_b_id) == "B-only suffix"
            assert store.session_draft(session_a.id) == ""
        finally:
            console._console_sync_in_progress = False
            if record is not None:
                runtime._release_custody(record.turn_id)
            else:
                store.release_received_turn(claim)
            await runtime.dispose()
        assert store.received_turn_for_session(session_a.id) is None
        assert not runtime.has_custodied_turns(session_a.id)
