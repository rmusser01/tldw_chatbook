"""The opt-in probe must measure refreshed paint, not pending widget state."""

import asyncio

import pytest

from Tests.Benchmarks.console_character_switcher_latency import (
    PaintWindow,
    _accepted_modal_closed,
    _inspect_copy,
    _keyword_evidence,
    _observe_commit_waiter,
)


def test_busy_latency_requires_a_current_refresh_and_uses_posted_time():
    screen = object()
    window = PaintWindow("search", "query", 1_000_000_000, screen=screen)
    # Merely beginning a pending operation cannot report an instantaneous paint.
    assert window.busy_ns is None
    window.observe("Searching local chats…", 1_010_000_000, screen=object())
    assert window.busy_ns is None
    window.observe("Previous result", 1_020_000_000, screen=screen)
    assert window.busy_ns is None
    window.observe("Searching local chats…", 1_095_000_000, screen=screen)
    window.observe("Searching local chats…", 1_120_000_000, screen=screen)
    assert window.summary()["busy_ms"] == 95.0
    assert window.busy_text == "Searching local chats…"


@pytest.mark.parametrize(
    "paint_ns,gap_ns,expected",
    [
        (None, 5_000_000, "No busy frame"),
        (100_100_000, 5_000_000, "Busy paint"),
        (90_000_000, 50_100_000, "Event-loop"),
    ],
)
def test_missing_or_over_budget_observation_cannot_pass(paint_ns, gap_ns, expected):
    screen = object()
    window = PaintWindow("activation", "query", 0, screen=screen)
    if paint_ns is not None:
        window.observe("Opening…", paint_ns, screen=screen)
    window.gaps_ns.append(gap_ns)
    assert any(expected in failure for failure in window.failures())


def test_finishing_paint_and_terminal_gap_are_kept_without_changing_thresholds():
    screen = object()
    window = PaintWindow("activation", "query", 0, screen=screen)
    window.observe("Finishing…", 100_000_000, screen=screen)
    window.gaps_ns.extend([5_000_000, 50_000_000])
    assert window.failures() == []
    assert window.summary()["event_loop_max_gap_ms"] == 50.0
    assert window.summary()["busy_ms"] == 100.0


def test_copy_integrity_uses_production_schema_functions_without_closing_observer(
    tmp_path,
):
    from Tests.conftest import _close_database_instance
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

    path = tmp_path / "copied.sqlite"
    observer = CharactersRAGDB(path, client_id="latency-probe-test")
    connection = observer.get_connection()
    try:
        assert _inspect_copy(path) == (0, 0, 0)
        assert connection.execute("SELECT 1").fetchone()[0] == 1
        assert observer.registered_connection_count() == 1
    finally:
        _close_database_instance(observer)


def test_keyword_evidence_reports_stale_generation_without_maintaining_it(tmp_path):
    from Tests.conftest import _close_database_instance
    from tldw_chatbook.Character_Chat.character_conversation_navigation import (
        CharacterConversationNavigationService,
    )
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

    database = CharactersRAGDB(tmp_path / "status.sqlite", client_id="latency-status")
    try:
        service = CharacterConversationNavigationService(database)
        assert str(service.ensure_keyword_index()) == "ready"
        ready = _keyword_evidence(database)
        database.add_character_card({"name": "New unrelated startup card"})
        stale = _keyword_evidence(database)
        assert ready["status"] == "ready"
        assert stale["status"] == "absent"
        assert stale["revision"] == ready["revision"] + 1
        assert stale["generations"] == ready["generations"]
        assert stale["counts"] == [0, 0, 0]
        assert stale["dirty_conversations"] == 0
    finally:
        _close_database_instance(database)


@pytest.mark.parametrize("prior_failure", [None, RuntimeError("operation failed")])
def test_terminal_receipt_cannot_pass_changed_corpus(prior_failure):
    import json

    from Tests.Benchmarks.console_character_switcher_latency import _finalize_evidence

    evidence = {"status": "passed", "failures": [], "corpus_sha256_before": "before"}
    failure = _finalize_evidence(evidence, prior_failure, "different")
    receipt = json.loads(json.dumps(evidence))
    assert receipt["status"] == "failed"
    assert receipt["source_unchanged"] is False
    assert "Controller corpus changed" in receipt["failures"]
    assert receipt["error"] == f"{type(failure).__name__}: {failure}"
    if prior_failure is not None:
        assert failure is prior_failure
    else:
        assert isinstance(failure, AssertionError)


@pytest.mark.asyncio
async def test_modal_settlement_uses_real_pop_and_registry_not_lifetime_flag():
    from textual.app import App
    from textual.screen import ModalScreen, Screen

    app = App()
    async with asyncio.timeout(5), app.run_test() as pilot:
        chat = app.screen
        modal = ModalScreen()
        await app.push_screen(modal)
        await pilot.pause()
        assert modal.is_mounted and app.is_mounted(modal)
        assert not _accepted_modal_closed(app, modal, chat)
        removed = app.pop_screen()
        # Stack transfer is synchronous; actual unregister is deferred.
        assert app.screen is chat and modal not in app.screen_stack
        assert app.is_mounted(modal)
        assert not _accepted_modal_closed(app, modal, chat)
        await removed
        assert modal.is_mounted  # Textual's lifetime flag intentionally stays true.
        assert not app.is_mounted(modal)
        assert _accepted_modal_closed(app, modal, chat)
        await app.push_screen(Screen())
        assert not _accepted_modal_closed(app, modal, chat)
        await app.pop_screen()


@pytest.mark.asyncio
@pytest.mark.parametrize("owner", [None, object()], ids=["ordinary", "owned"])
async def test_commit_observer_forwards_exact_owner_and_waits_for_acknowledgement(
    owner,
):
    request, extra, tag = object(), object(), object()
    window = PaintWindow("activation", "query", 0)
    entered, release = asyncio.Event(), asyncio.Event()

    async def original(actual_request, actual_extra, *, complete_presentation, marker):
        assert actual_request is request and actual_extra is extra
        assert complete_presentation is owner and marker is tag
        entered.set()
        await release.wait()

    observed = _observe_commit_waiter(original, lambda: window)
    task = asyncio.create_task(
        observed(request, extra, complete_presentation=owner, marker=tag)
    )
    try:
        await asyncio.wait_for(entered.wait(), 1)
        assert not task.done() and not window.calls
        release.set()
        await asyncio.wait_for(task, 1)
        assert len(window.calls) == 1
        assert window.calls[0]["kind"] == "commit_started"
    finally:
        release.set()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [RuntimeError("rejected"), asyncio.CancelledError()])
async def test_commit_observer_records_failure_without_acknowledging_or_replacing_it(
    error,
):
    window = PaintWindow("activation", "query", 0)

    async def original(request):
        raise error

    observed = _observe_commit_waiter(original, lambda: window)
    with pytest.raises(type(error)) as caught:
        await observed(object())
    assert caught.value is error
    assert len(window.calls) == 1
    assert window.calls[0]["kind"] == "commit_waiter_error"
    assert window.calls[0]["exception_type"] == type(error).__name__
