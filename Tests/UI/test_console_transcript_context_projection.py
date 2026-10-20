"""Transcript painting shares finite context display work without awaiting it."""

import asyncio
import sys
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from Tests.UI.test_console_shared_context_policy import _actual_calls, _screen
from tldw_chatbook.Chat import console_context_policy
from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
from tldw_chatbook.Chat.console_context_compaction import (
    EffectiveMemoryKind,
    EffectiveMemoryResult,
    LegacyMemorySnapshot,
)
from tldw_chatbook.UI.Console_Modules import console_spend_projection as spend
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen
from tldw_chatbook.Widgets.Console.console_transcript import (
    derive_console_memory_banner_presentation,
)

pytestmark = pytest.mark.bootstrap_profile


class BannerPainted(Exception):
    pass


def _transcript_screen(screen, store, controller):
    paints, fetches = [], []

    def paint(value):
        paints.append(value)
        raise BannerPainted

    transcript = Mock()
    transcript.set_memory_banner_presentation.side_effect = paint
    screen.query_one = lambda *_args: transcript
    screen._console_transcript_region_or_none = lambda: None
    screen._change_review_projection = SimpleNamespace(
        project=lambda messages: messages, revision_key=lambda: ("review", 1)
    )

    def messages():
        fetches.append(1)
        return store.messages_for_session(store.active_session_id)

    screen._message = SimpleNamespace(
        _native_console_messages=messages,
        sync_selected_fork_eligibility=lambda _: (None, None),
    )
    screen._console_chat_controller = controller
    screen._ensure_console_chat_controller = lambda: controller
    screen._ensure_console_chat_store = lambda: store
    screen._console_original_attempt_previews = {}
    screen._pending_console_swipe_selection = None
    screen._show_model_thinking = lambda: False
    screen._console_presentation_context = lambda: None
    screen._sync_console_citation_count_discovery = lambda _: None
    screen._sync_console_rate_chip = lambda _: None
    screen._console_citation_counts = {}
    screen._console_annotation_previews = {}
    screen._library_activity = SimpleNamespace(sync_transcript=lambda _: {})
    screen._review_selection = SimpleNamespace(
        _sync_console_annotation_discovery=lambda _: None,
        _console_change_review_provider=lambda: None,
    )
    screen._agent = SimpleNamespace(
        console_turn_activity=lambda: None,
        console_turn_activity_abandon_action=lambda: None,
    )
    screen._context_spend = SimpleNamespace(
        _context_inputs=lambda owner,
        session_id: spend.ConsoleContextReadSnapshot.for_screen(
            screen, max_age=1.0
        ).inputs(owner, session_id)
    )
    return paints, fetches


async def _render(screen, *, force=False):
    try:
        await ChatScreen._sync_native_console_transcript(screen, force=force)
    except BannerPainted:
        pass


@pytest.mark.asyncio
async def test_original_transcript_does_not_read_unused_live_global_policy(
    tmp_path, monkeypatch, record_property
):
    database, store, controller, screen, tasks = _screen(tmp_path)
    paints, _ = _transcript_screen(screen, store, controller)
    monkeypatch.setattr(console_context_policy, "_LAST_GLOBAL_OVERRIDES", None)
    try:
        with _actual_calls(controller) as calls:
            await _render(screen)
        record_property("direct_policy_calls", calls["live_policy"])
        record_property("direct_config_entries", calls["config"])
        assert calls["live_policy"] == 0, "transcript read unused global policy"
        assert calls["config"] == 0, "transcript entered native config on UI loop"
        assert paints == [None]
    finally:
        await asyncio.gather(*tasks, return_exceptions=True)
        database.close()


@pytest.mark.asyncio
async def test_context_publication_reopens_unchanged_transcript_gate(tmp_path):
    database, store, controller, screen, tasks = _screen(tmp_path)
    paints, fetches = _transcript_screen(screen, store, controller)
    try:
        await _render(screen)
        assert len(fetches) == 1
        assert tasks, "cold transcript did not request finite display preparation"
        await asyncio.gather(*tasks)
        # The existing context owner first warms its checked config, then
        # schedules memory preparation from the next original display read.
        await _render(screen)
        await asyncio.gather(*tasks)
        await _render(screen)
        assert len(fetches) == 2, "published memory inputs stayed behind tick gate"
        await _render(screen)
        assert len(fetches) == 2, "settled memory inputs should keep the tick gate"
        assert paints == [None, None]
    finally:
        await asyncio.gather(*tasks, return_exceptions=True)
        database.close()


@pytest.mark.asyncio
async def test_streaming_transcript_keeps_painting_during_original_native_context_read(
    tmp_path,
):
    database, store, controller, screen, tasks = _screen(tmp_path)
    paints, fetches = _transcript_screen(screen, store, controller)
    entered, release = threading.Event(), threading.Event()
    code = controller._context_repository.has_active_memory_selection.__func__.__code__
    main = threading.get_ident()
    previous, previous_thread = sys.getprofile(), threading.getprofile()

    def observe(frame, event, _argument):
        if frame.f_code is code and event == "return" and threading.get_ident() != main:
            entered.set()
            assert release.wait(10), "test did not release original memory probe"

    try:
        projection = spend.ConsoleReadinessConfigProjection.for_screen(screen)
        assert await projection.warm()
        threading.setprofile_all_threads(observe)
        await _render(screen)
        assert await asyncio.to_thread(entered.wait, 5)
        assert controller._preparation_reads, "original finite owner was not retained"
        for content in ("first chunk", "second chunk"):
            store.append_message(
                store.active_session_id,
                role=ConsoleMessageRole.ASSISTANT,
                content=content,
            )
            await asyncio.wait_for(_render(screen), 2)
        assert len(fetches) == 3
        assert paints == [None, None, None]
        assert not release.is_set()
    finally:
        release.set()
        await asyncio.gather(*tasks, return_exceptions=True)
        threading.setprofile_all_threads(previous_thread)
        sys.setprofile(previous)
        assert not controller._preparation_reads
        database.close()


@pytest.mark.asyncio
async def test_transcript_banner_matches_original_branch_validated_memory(tmp_path):
    database, store, controller, screen, tasks = _screen(tmp_path)
    paints, _ = _transcript_screen(screen, store, controller)
    try:
        boundary = store.append_message(
            store.active_session_id,
            role=ConsoleMessageRole.ASSISTANT,
            content="Earlier answer",
            persist=True,
        )
        store.append_message(
            store.active_session_id,
            role=ConsoleMessageRole.USER,
            content="Follow-up",
            persist=True,
        )
        store.set_session_context_summary(
            store.active_session_id, "Private legacy summary", boundary.id
        )
        snapshot = spend.ConsoleContextReadSnapshot.for_screen(screen, max_age=1.0)
        assert await snapshot.warm(controller, store.active_session_id)
        expected = derive_console_memory_banner_presentation(
            controller.context_control_inputs(store.active_session_id)[2],
            store.messages_for_session(store.active_session_id),
        )
        assert expected is not None
        await _render(screen)
        assert paints == [expected]
    finally:
        await asyncio.gather(*tasks, return_exceptions=True)
        database.close()


@pytest.mark.asyncio
async def test_custom_live_reader_keeps_original_post_gate_cadence_and_thread(
    tmp_path,
    monkeypatch,
):
    database, store, controller, screen, tasks = _screen(tmp_path)
    paints, fetches = _transcript_screen(screen, store, controller)
    calls = []

    def custom(session_id):
        calls.append((threading.get_ident(), session_id))
        return None, None, EffectiveMemoryResult(EffectiveMemoryKind.RAW)

    monkeypatch.setattr(controller, "context_control_inputs", custom)
    try:
        await _render(screen)
        await _render(screen)
        assert calls == [(threading.get_ident(), store.active_session_id)]
        assert len(fetches) == 1 and paints == [None]
        assert not tasks
    finally:
        await asyncio.gather(*tasks, return_exceptions=True)
        database.close()


@pytest.mark.asyncio
async def test_custom_presentation_reader_is_not_newly_used_by_transcript(
    tmp_path,
    monkeypatch,
):
    database, store, controller, screen, tasks = _screen(tmp_path)
    _transcript_screen(screen, store, controller)

    async def custom(*_args, **_kwargs):
        pytest.fail("transcript newly entered a custom settings presentation reader")

    monkeypatch.setattr(controller, "context_control_presentation_inputs", custom)
    try:
        with _actual_calls(controller) as calls:
            await _render(screen)
            await _render(screen)
        assert calls["live_policy"] == 1
        assert not tasks
    finally:
        await asyncio.gather(*tasks, return_exceptions=True)
        database.close()


@pytest.mark.asyncio
async def test_malformed_shared_inputs_remain_optional_display_failure(tmp_path):
    database, store, controller, screen, tasks = _screen(tmp_path)
    paints, _ = _transcript_screen(screen, store, controller)
    screen._context_spend._context_inputs = lambda *_: (None,)
    try:
        with _actual_calls(controller) as calls:
            await _render(screen)
        assert paints == [None]
        assert calls["live_policy"] == 0
        assert not tasks
    finally:
        await asyncio.gather(*tasks, return_exceptions=True)
        database.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "custom_method", ["context_control_inputs", "context_control_presentation_inputs"]
)
async def test_restored_stock_reader_never_reuses_custom_memory_within_ttl(
    tmp_path,
    monkeypatch,
    custom_method,
):
    database, store, controller, screen, tasks = _screen(tmp_path)
    paints, _ = _transcript_screen(screen, store, controller)
    row = store.messages_for_session(store.active_session_id)[0]
    custom_memory = EffectiveMemoryResult(
        EffectiveMemoryKind.LEGACY_PREFIX,
        legacy=LegacyMemorySnapshot(
            store.ensure_session().persisted_conversation_id,
            "Custom display summary",
            row.persisted_message_id,
        ),
    )
    custom_value = None, None, custom_memory

    def custom_sync(_session_id):
        return custom_value

    async def custom_async(_session_id, **_kwargs):
        return custom_value

    try:
        snapshot = spend.ConsoleContextReadSnapshot.for_screen(screen, max_age=1.0)
        with monkeypatch.context() as patch:
            patch.setattr(
                controller,
                custom_method,
                custom_async
                if custom_method.endswith("presentation_inputs")
                else custom_sync,
            )
            assert await snapshot.warm(controller, store.active_session_id)
            assert snapshot.value == custom_value
        await _render(screen)
        assert paints == [None], "restored stock reader reused a custom banner"
    finally:
        await asyncio.gather(*tasks, return_exceptions=True)
        database.close()
