"""TASK-339: keystrokes typed during the session-switch settle window.

`_sync_console_session_draft` used to save the LIVE composer text to the old
session and reload the new session's stored draft whenever the deferred swap
finally ran — keystrokes typed after the switch keypress were misattributed
to the old session and wiped/reordered in the composer (UX review finding
j2-post-switch-typing-reordered; the mangled text was sent and persisted).

The settle window is simulated deterministically: the coalescing guard is
held so the activation's inline sync defers, typing happens in the gap, and
the swap then runs exactly as a later sync pass would.
"""

import pytest
from textual.events import Key

from Tests.UI.test_console_native_chat_flow import _select_llamacpp_console
from Tests.UI.test_destination_shells import _build_test_app, _wait_for_selector
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
)
from tldw_chatbook.Widgets.Console import ConsoleComposerBar
from tldw_chatbook.Widgets.Console.console_composer_bar import (
    classify_console_raw_draft,
)


async def _settle_window_swap(console, pilot, *, type_during_window: str):
    """Switch to a fresh session while the sync is held, typing in the gap."""
    store = console._ensure_console_chat_store()
    session_a = store.ensure_session(title="Chat A")
    composer = console.query_one("#console-native-composer", ConsoleComposerBar)
    composer.focus()
    await pilot.pause()

    composer.load_draft("old draft")
    console._session._sync_console_session_draft()  # settle tracker onto A

    # Hold the coalescing guard: the new-tab path's inline sync defers, so
    # the draft swap runs only on a LATER pass — the real settle window.
    console._console_sync_in_progress = True
    try:
        await console._session._create_native_console_session_from_active_context()
        if type_during_window:
            composer.insert_text(type_during_window)
    finally:
        console._console_sync_in_progress = False

    session_b_id = store.active_session_id
    assert session_b_id != session_a.id
    console._session._sync_console_session_draft()
    return store, session_a, session_b_id, composer


@pytest.mark.asyncio
async def test_console_switch_carries_settle_window_typing_to_new_session():
    app = _build_test_app()
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = "test-model"
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        await _select_llamacpp_console(console)

        store, session_a, session_b_id, composer = await _settle_window_swap(
            console, pilot, type_during_window="typed in window"
        )

        # The keystrokes belong to the NEW session, in order…
        assert composer.draft_text() == "typed in window"
        # …and the OLD session keeps only what it actually had at the
        # switch keypress (no misattributed keystrokes).
        assert store.session_draft(session_a.id) == "old draft"


@pytest.mark.asyncio
async def test_console_switch_without_typing_swaps_drafts_exactly_as_before():
    app = _build_test_app()
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = "test-model"
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        await _select_llamacpp_console(console)

        store, session_a, session_b_id, composer = await _settle_window_swap(
            console, pilot, type_during_window=""
        )

        assert composer.draft_text() == ""
        assert store.session_draft(session_a.id) == "old draft"


@pytest.mark.asyncio
async def test_raw_refusal_during_switch_settle_preserves_both_session_drafts():
    app = _build_test_app()
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = "test-model"
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        await _select_llamacpp_console(console)

        store = console._ensure_console_chat_store()
        session_a = store.ensure_session(title="Chat A")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.focus()
        await pilot.pause()
        console._session._sync_console_session_draft()

        assert composer.handle_console_key(Key("exclamation_mark", "!")) is True
        assert composer.handle_console_key(Key("space", " ")) is True
        composer.insert_pasted_text("pwd")
        raw_stash = composer.stash_draft_for_send()
        assert raw_stash is not None

        console._console_sync_in_progress = True
        try:
            await console._session._create_native_console_session_from_active_context()
            composer.insert_text("b-suffix")
        finally:
            console._console_sync_in_progress = False

        session_b_id = store.active_session_id
        assert session_b_id != session_a.id
        assert console._console_visible_draft_session_id == session_a.id

        console._raw_cli._refuse(session_a.id, raw_stash, "test refusal")
        assert composer.draft_text() == "b-suffix"
        assert store.session_draft(session_a.id) == ""

        console._session._sync_console_session_draft()
        assert composer.draft_text() == "b-suffix"

        console._session._capture_console_draft_switch_snapshot()
        store.switch_session(session_a.id)
        console._session._sync_console_session_draft()

        restored = composer.stash_draft_for_send()
        assert store.session_draft(session_b_id) == "b-suffix"
        assert restored is not None
        assert restored.raw_cli_prefix_typed is True
        assert restored.has_paste is True
        assert restored.segments == raw_stash.segments
        classified = classify_console_raw_draft(restored)
        assert classified.kind == "raw"
        assert classified.text == "pwd"


@pytest.mark.asyncio
@pytest.mark.bootstrap_profile
@pytest.mark.parametrize(
    ("old_state", "suffix"),
    [
        ("closed", ""),
        ("closed", " B-only suffix"),
        ("revised", " B-only suffix"),
        ("live", " B-only suffix"),
    ],
    ids=["closed", "closed-suffix", "live-revised", "live-undo"],
)
async def test_old_draft_snapshot_after_switch_keeps_exact_session_owners(
    old_state, suffix
):
    from Tests.UI.test_console_approval_compact_layout import (
        _wait_for_reconciled_console,
    )

    app = _build_test_app()
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = "test-model"
    host = ConsoleHarness(app)
    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        await _wait_for_reconciled_console(console, pilot)
        store = console._ensure_console_chat_store()
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        old = store.ensure_session(title="Old visible chat")
        composer.load_draft("")
        if old_state == "live":
            # Ordinary character edits coalesce as one user typing run. Two
            # multi-character inserts create separate undo entries instead.
            for character in "old draft":
                composer.insert_text(character)
            assert len(composer.export_undo_history()[0]) == 1
        else:
            composer.insert_text("old draft")
        console._session._sync_console_session_draft()
        assert console._session._console_visible_draft_revision is not None
        assert old.id not in console._session._console_undo_histories
        other = store.create_session(title="Unrelated owner", activate=False)
        store.set_session_draft(other.id, "unrelated draft")
        other_revision = store.session_input_snapshot(other.id).draft_revision
        target = store.create_session(title="Existing target", activate=False)
        store.set_session_draft(target.id, "target draft")
        console._session._capture_console_draft_switch_snapshot()
        if old_state == "closed":
            # Public domain close changes the store before its view projection.
            # This test exercises draft binding; close confirmation has its own
            # real tab-strip controls and is not replaced here.
            store.close_session(old.id)
        elif old_state == "revised":
            store.set_session_draft(old.id, "newer resident draft")
        store.switch_session(target.id)
        if suffix:
            if old_state == "live":
                for character in suffix:
                    composer.insert_text(character)
                assert len(composer.export_undo_history()[0]) == 1
            else:
                composer.insert_text(suffix)
        assert composer.draft_text() == "old draft" + suffix
        console._session._sync_console_session_draft()
        assert store.active_session_id == target.id
        assert console._session._console_visible_draft_session_id == target.id
        assert composer.draft_text() == "target draft" + suffix
        assert store.session_draft(target.id) == "target draft" + suffix
        assert store.session_draft(other.id) == "unrelated draft"
        assert store.session_input_snapshot(other.id).draft_revision == other_revision
        assert console._session._console_draft_switch_snapshot is None
        if old_state == "closed":
            assert {row.id for row in store.sessions()} == {target.id, other.id}
            assert old.id not in console._session._console_undo_histories
            with pytest.raises(KeyError):
                store.session_input_snapshot(old.id)
        elif old_state == "revised":
            assert store.session_draft(old.id) == "newer resident draft"
            assert old.id not in console._session._console_undo_histories
        else:
            assert store.session_draft(old.id) == "old draft"
            console._session._capture_console_draft_switch_snapshot()
            store.switch_session(old.id)
            console._session._sync_console_session_draft()
            assert composer.draft_text() == "old draft"
            console._console_composer_undo()
            assert composer.draft_text() == ""
            assert store.session_draft(old.id) == ""
            assert store.session_draft(target.id) == "target draft" + suffix
