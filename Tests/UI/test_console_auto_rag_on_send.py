"""Console opt-in RAG auto-retrieve on send (TASK-406 / TASK-3170 task 8).

The send path grows ONE new hook, ``ConsoleRetrievalController._maybe_auto_retrieve_for_send``,
called from the consume-on-send seam (``_capture_console_staged_rag``) so an
auto-retrieved bundle is staged and consumed by the SAME send and renders in
the staged-evidence strip exactly like a manual chip run.

Every clause of the hook's gate is pinned by its own test here, in gate order:
toggle -> plain-text send -> already-staged -> empty scope -> retrieval. The
tests are deliberately split so a dropped clause reds exactly one of them.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from Tests.fixtures.required_doubles import exploding_double
from Tests.UI.test_console_dictionary_send_integration import (
    _CapturingGateway,
    _final_user_content,
)
from Tests.UI.test_destination_shells import _build_test_app, _wait_for_selector
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
)
from tldw_chatbook.Chat.console_chat_models import ConsoleProviderSelection
from tldw_chatbook.Chat.console_live_work import ConsoleLiveWorkLaunch
from tldw_chatbook.Chat.rag_scope import EffectiveScope, scope_empty_notice
from tldw_chatbook.Chat.console_turn_context import ConsoleTurnConfigurationSnapshot
from tldw_chatbook.config import load_settings, save_setting_to_cli_config
from tldw_chatbook.Event_Handlers.Chat_Events.chat_rag_events import (
    LocalRagContextResult,
)
from tldw_chatbook.Library.library_rag_service import (
    LibraryRagSearchOutcome,
    RETRIEVAL_FAILED_WHY,
)
from tldw_chatbook.Library.library_rag_state import LibraryRagResultRow
from tldw_chatbook.UI.Screens import chat_screen as chat_screen_module
from tldw_chatbook.UI.Console_Modules import retrieval as retrieval_module
from tldw_chatbook.UI.Console_Modules.retrieval import ConsoleRetrievalController

STRIP_ID = "#console-staged-evidence-strip"


@pytest.fixture(autouse=True)
def _reset_auto_retrieve_toggle():
    """Leave the isolated config exactly as this suite found it.

    The toggle is a persisted `[chat_defaults]` key; sibling suites
    (`test_console_rag_settings_modal.py`) assert it reads False at rest, so
    a test here that flips it on must not leak that across the session.
    """
    yield
    save_setting_to_cli_config("chat_defaults", "rag_auto_retrieve_on_send", False)


def _enable_auto_retrieve() -> None:
    save_setting_to_cli_config("chat_defaults", "rag_auto_retrieve_on_send", True)


def _use_disk_config(app):
    """Source the app's config snapshot from disk, exactly as `app.py` does.

    `_build_test_app` hands `TldwCli` a synthetic three-key `app_config`
    with no `[chat_defaults]` at all, and production correctly refuses to
    "refresh" a snapshot that never came from `load_settings()`
    (`ChatScreen._console_config_snapshot_is_disk_loaded`). That was
    harmless while `_maybe_auto_retrieve_for_send` read the toggle live via
    `get_cli_setting`; task-14803 (commit 5be9e6a04) moved the read onto the
    frozen per-turn `ConsoleTurnExecutionContext`, which
    `ConsoleSessionController._build_console_turn_execution_context` builds
    from that app config -- so on THIS harness, and only on this harness,
    `_enable_auto_retrieve()` stopped reaching the code under test. The
    shipping app assigns `self.app_config = load_settings()` (app.py), whose
    result does carry the persisted toggle; do the same here so these two
    mounted tests exercise the real read path instead of a snapshot no real
    user has.
    """
    app.app_config = load_settings()
    return app


class _RecordingRagService:
    """Records each `search()` and returns preconfigured rows."""

    def __init__(self, rows=(), *, delay: float = 0.0) -> None:
        self.rows = tuple(rows)
        self.delay = delay
        self.calls: list[dict] = []

    async def search(self, query, source_types, mode, **kwargs):
        self.calls.append(
            {
                "query": query,
                "source_types": tuple(source_types),
                "mode": mode,
                **kwargs,
            }
        )
        if self.delay:
            await asyncio.sleep(self.delay)
        return {"runtime_backend": "local-test", "results": list(self.rows)}


def _rows(count: int) -> tuple[LibraryRagResultRow, ...]:
    return tuple(
        LibraryRagResultRow.from_result(
            {
                "source_id": f"media-{index}",
                "chunk_id": f"chunk-{index}",
                "title": f"Source {index}",
                "content": f"Body {index}",
                "score": 1.0 - index / 10,
                "runtime_backend": "local",
                "source_type": "media",
            }
        )
        for index in range(1, count + 1)
    )


def _staged_launch() -> ConsoleLiveWorkLaunch:
    return ConsoleLiveWorkLaunch.from_values(
        source="Library Search/RAG",
        title="Manually staged",
        payload={"query": "manual", "evidence_bundle": {"bundle_id": "manual"}},
        status="staged",
    )


def _auto_rag_screen(*, service=None, staged=None, has_pending: bool = False):
    """A screen-state stand-in wired to the real retrieval controller."""
    notices: list[tuple[str, str]] = []

    def _notify(message, severity="information", **_kwargs):
        notices.append((str(message), severity))

    app = SimpleNamespace(
        library_rag_search_service=service,
        pending_handoffs=SimpleNamespace(has_pending=lambda _channel: has_pending),
        notify=_notify,
        _rag_service=None,
    )
    screen = SimpleNamespace(
        app_instance=app,
        is_mounted=True,
        notices=notices,
        _pending_console_launch_context=staged,
        _pending_console_launch_auto_open_inspector=False,
        _console_evidence_sent_notice=None,
        _console_library_rag_source_types=("media",),
        _sync_console_pending_launch_surfaces=lambda: True,
    )
    screen._has_staged_console_evidence = lambda: bool(
        screen._pending_console_launch_context is not None
        or app.pending_handoffs.has_pending(None)
    )
    screen._retrieval = ConsoleRetrievalController(
        app_instance=app,
        active_native_session=lambda: None,
        current_conversation_id=lambda: None,
        clear_evidence_sent_notice=lambda: setattr(
            screen, "_console_evidence_sent_notice", None
        ),
        consume_pending_launch=lambda: screen._pending_console_launch_context,
        release_consumed_launch=lambda *_args: None,
        is_mounted=lambda: screen.is_mounted,
        sync_retrieval_scope_row=lambda: None,
        sync_control_bar=lambda: None,
        request_control_bar_sync=lambda: None,
        dictionary_scope_service=lambda: None,
        set_library_rag_source_scope=lambda value: setattr(
            screen, "_console_library_rag_source_types", tuple(value)
        ),
        set_library_rag_query=lambda _value: None,
        run_library_rag_action=lambda: None,
        library_rag_source_scope=(
            lambda: tuple(screen._console_library_rag_source_types)
        ),
        library_rag_top_k=lambda: (
            chat_screen_module._console_library_rag_profile_top_k()
        ),
        pending_launch=lambda: screen._pending_console_launch_context,
        set_pending_launch=lambda launch: setattr(
            screen, "_pending_console_launch_context", launch
        ),
        set_pending_auto_open=lambda value: setattr(
            screen, "_pending_console_launch_auto_open_inspector", value
        ),
        set_evidence_sent_notice=lambda value: setattr(
            screen, "_console_evidence_sent_notice", value
        ),
        sync_pending_launch_surfaces=screen._sync_console_pending_launch_surfaces,
        refresh_screen=lambda: None,
        has_staged_evidence=screen._has_staged_console_evidence,
    )
    return screen


def _patch_scope(monkeypatch, state: str = "unscoped", cause=None) -> None:
    monkeypatch.setattr(
        retrieval_module,
        "resolve_effective_scope_for_chat",
        AsyncMock(return_value=EffectiveScope(state=state, allowlist={}, cause=cause)),
    )


# --------------------------------------------------------------------------
# Gate clause 1: the config toggle
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_toggle_off_means_no_retrieval_call(monkeypatch):
    """Default OFF: not one retrieval call, not one staged launch."""
    _patch_scope(monkeypatch)
    service = _RecordingRagService(_rows(2))
    screen = _auto_rag_screen(service=service)

    await screen._retrieval._maybe_auto_retrieve_for_send("what changed in the notes")

    assert service.calls == []
    assert screen._pending_console_launch_context is None
    assert screen.notices == []


@pytest.mark.asyncio
async def test_toggle_default_is_off_at_the_read_site(monkeypatch):
    """Pin the code-level fallback, not just the shipped config value.

    `[chat_defaults] rag_auto_retrieve_on_send = false` ships in the default
    config, so the test above passes even if this read site's own default
    were flipped to True -- the key is always present. A user config written
    before this key existed has no such row, and for them the literal here
    is the only thing standing between "off by default" and silent spend.
    """
    seen: dict[tuple[str, str], object] = {}

    def _recording_get_cli_setting(section, key, default=None, **_kwargs):
        seen[(section, key)] = default
        return default

    monkeypatch.setattr(retrieval_module, "get_cli_setting", _recording_get_cli_setting)
    _patch_scope(monkeypatch)
    service = _RecordingRagService(_rows(2))
    screen = _auto_rag_screen(service=service)

    await screen._retrieval._maybe_auto_retrieve_for_send("what changed in the notes")

    assert seen[("chat_defaults", "rag_auto_retrieve_on_send")] is False
    assert service.calls == []


# --------------------------------------------------------------------------
# Gate clause 2: plain-text sends only
# --------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("draft", ["/prompt summarize", "$research notes", "  /rewind"])
async def test_slash_command_send_never_retrieves(monkeypatch, draft):
    """Slash commands and `$skill` invocations are not questions to retrieve for."""
    _enable_auto_retrieve()
    _patch_scope(monkeypatch)
    service = _RecordingRagService(_rows(2))
    screen = _auto_rag_screen(service=service)

    await screen._retrieval._maybe_auto_retrieve_for_send(draft)

    assert service.calls == []
    assert screen._pending_console_launch_context is None


@pytest.mark.asyncio
async def test_plain_text_send_does_retrieve(monkeypatch):
    """Control for the clause above: ordinary prose still retrieves."""
    _enable_auto_retrieve()
    _patch_scope(monkeypatch)
    service = _RecordingRagService(_rows(2))
    screen = _auto_rag_screen(service=service)

    await screen._retrieval._maybe_auto_retrieve_for_send("what changed in the notes")

    assert len(service.calls) == 1
    assert service.calls[0]["query"] == "what changed in the notes"
    assert service.calls[0]["mode"] == "rag"


# --------------------------------------------------------------------------
# Gate clause 3: manual staging always wins
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_already_staged_evidence_skips_auto_retrieve(monkeypatch):
    """No double spend: an existing staged bundle is left exactly as it was."""
    _enable_auto_retrieve()
    _patch_scope(monkeypatch)
    service = _RecordingRagService(_rows(2))
    staged = _staged_launch()
    screen = _auto_rag_screen(service=service, staged=staged)

    await screen._retrieval._maybe_auto_retrieve_for_send("what changed in the notes")

    assert service.calls == []
    assert screen._pending_console_launch_context is staged


@pytest.mark.asyncio
async def test_unclaimed_handoff_also_counts_as_already_staged(monkeypatch):
    """A handoff the very next consume will claim is staging too.

    `_consume_pending_console_launch` claims an unclaimed CONSOLE_LIVE_WORK
    entry on the same send, so retrieving here would spend a search whose
    result the claim immediately supersedes.
    """
    _enable_auto_retrieve()
    _patch_scope(monkeypatch)
    service = _RecordingRagService(_rows(2))
    screen = _auto_rag_screen(service=service, has_pending=True)

    await screen._retrieval._maybe_auto_retrieve_for_send("what changed in the notes")

    assert service.calls == []
    assert screen._pending_console_launch_context is None


# --------------------------------------------------------------------------
# Gate clause 4: EMPTY scope short-circuit
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_empty_scope_short_circuits_with_shared_copy(monkeypatch):
    """An EMPTY effective scope never reaches the backend, and says so once."""
    _enable_auto_retrieve()
    _patch_scope(monkeypatch, state="empty", cause="deleted-items")
    service = _RecordingRagService(_rows(2))
    screen = _auto_rag_screen(service=service)

    await screen._retrieval._maybe_auto_retrieve_for_send("what changed in the notes")

    assert service.calls == []
    assert screen._pending_console_launch_context is None
    assert screen.notices == [
        (
            scope_empty_notice("deleted-items"),
            "warning",
        )
    ]


# --------------------------------------------------------------------------
# The retrieval itself
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_query_is_length_capped_before_it_reaches_retrieval(monkeypatch):
    _enable_auto_retrieve()
    _patch_scope(monkeypatch)
    service = _RecordingRagService(_rows(1))
    screen = _auto_rag_screen(service=service)

    await screen._retrieval._maybe_auto_retrieve_for_send("a" * 9_000)

    assert len(service.calls) == 1
    assert len(service.calls[0]["query"]) == retrieval_module.AUTO_RAG_QUERY_MAX_CHARS


@pytest.mark.asyncio
async def test_retrieval_requests_the_active_profile_top_k(monkeypatch):
    """Not a hardcoded 5: the request carries the active profile's top_k."""
    _enable_auto_retrieve()
    _patch_scope(monkeypatch)
    monkeypatch.setattr(
        chat_screen_module, "_console_library_rag_profile_top_k", lambda: 7
    )
    service = _RecordingRagService(_rows(1))
    screen = _auto_rag_screen(service=service)

    await screen._retrieval._maybe_auto_retrieve_for_send("what changed in the notes")

    assert service.calls[0]["top_k"] == 7


def test_profile_top_k_reads_the_active_rag_config(monkeypatch):
    """The helper's source of truth is the resolved active profile.

    TASK-15020/B3 moved the read onto `resolve_active_rag_top_k` -- the
    depth-only resolution shared with the Library window, which reads the
    same profile without building (and torch-importing) the whole config.
    So this patches THAT function, which is what the helper now actually
    reads; `Tests/RAG/test_active_config_resolution.py` pins that the two
    resolutions report the same number.
    """
    from tldw_chatbook.RAG_Search.simplified import active_config

    monkeypatch.setattr(active_config, "resolve_active_rag_top_k", lambda: 11)

    assert chat_screen_module._console_library_rag_profile_top_k() == 11


@pytest.mark.asyncio
async def test_in_flight_placeholder_is_staged_while_retrieval_runs(monkeypatch):
    """The strip must show "Retrieving..." for the WHOLE retrieval window.

    Review finding: every other assertion about the placeholder is a
    *clear* assertion (`... is None` afterwards), which stays true when the
    placeholder is never staged at all -- so deleting the only in-flight
    signal the user gets during the <=5s window left the whole suite green.
    This observes the staged launch from INSIDE the retrieval call, which is
    the only moment the claim is about.
    """
    _enable_auto_retrieve()
    _patch_scope(monkeypatch)
    screen = _auto_rag_screen()
    observed: list = []

    class _StagingObservingRagService:
        async def search(self, query, source_types, mode, **kwargs):
            observed.append(
                (
                    screen._pending_console_launch_context,
                    screen._has_staged_console_evidence(),
                )
            )
            return {"runtime_backend": "local-test", "results": list(_rows(2))}

    screen.app_instance.library_rag_search_service = _StagingObservingRagService()

    await screen._retrieval._maybe_auto_retrieve_for_send("what changed in the notes")

    assert len(observed) == 1
    in_flight, was_staged = observed[0]
    assert in_flight is not None, "no in-flight staging while retrieval ran"
    assert was_staged is True
    assert in_flight.status == "searching"
    assert in_flight.payload["query"] == "what changed in the notes"
    assert in_flight.recovery == retrieval_module.CONSOLE_AUTO_RAG_SEARCHING_COPY
    # ...and the results replaced it rather than piling up a second card.
    settled = screen._pending_console_launch_context
    assert settled is not in_flight
    assert settled.status == "staged"


@pytest.mark.asyncio
async def test_zero_results_never_leaves_a_blocking_launch_staged(monkeypatch):
    """An auto-retrieve that matched nothing must stage nothing.

    `_console_send_blocked_reason` BLOCKS any send while a RAG-sourced
    launch with zero available evidence is staged -- so staging the manual
    run's "no results" card here would have made one empty auto-retrieve
    lock the composer for every later send.
    """
    _enable_auto_retrieve()
    _patch_scope(monkeypatch)
    service = _RecordingRagService(())
    screen = _auto_rag_screen(service=service)

    await screen._retrieval._maybe_auto_retrieve_for_send("what changed in the notes")

    assert len(service.calls) == 1
    assert screen._pending_console_launch_context is None


@pytest.mark.asyncio
async def test_timeout_sends_without_evidence_and_notifies(monkeypatch):
    """A slow backend never holds the send: quiet notice, nothing staged."""
    _enable_auto_retrieve()
    _patch_scope(monkeypatch)
    monkeypatch.setattr(retrieval_module, "AUTO_RAG_TIMEOUT_SECONDS", 0.01)
    service = _RecordingRagService(_rows(2), delay=5.0)
    screen = _auto_rag_screen(service=service)

    await screen._retrieval._maybe_auto_retrieve_for_send("what changed in the notes")

    assert screen._pending_console_launch_context is None
    assert len(screen.notices) == 1
    message, severity = screen.notices[0]
    assert severity == "warning"
    # No cached runtime on the app -> the timeout is first-run model load,
    # and the copy must say so rather than blaming retrieval.
    assert message == retrieval_module.CONSOLE_AUTO_RAG_INITIALIZING_NOTICE


@pytest.mark.asyncio
async def test_timeout_with_a_live_runtime_reports_failure_not_initializing(
    monkeypatch,
):
    """With a warm runtime the same timeout is an honest retrieval failure."""
    _enable_auto_retrieve()
    _patch_scope(monkeypatch)
    monkeypatch.setattr(retrieval_module, "AUTO_RAG_TIMEOUT_SECONDS", 0.01)
    service = _RecordingRagService(_rows(2), delay=5.0)
    screen = _auto_rag_screen(service=service)
    screen.app_instance._rag_service = SimpleNamespace(search=lambda *a, **k: None)

    await screen._retrieval._maybe_auto_retrieve_for_send("what changed in the notes")

    assert screen.notices == [
        (retrieval_module.CONSOLE_AUTO_RAG_FAILED_NOTICE, "warning")
    ]
    assert RETRIEVAL_FAILED_WHY in retrieval_module.CONSOLE_AUTO_RAG_FAILED_NOTICE


@pytest.mark.asyncio
async def test_retrieval_exception_sends_without_evidence(monkeypatch):
    """A raising retrieval seam never escapes into the send."""
    _enable_auto_retrieve()
    _patch_scope(monkeypatch)
    monkeypatch.setattr(
        retrieval_module,
        "run_library_rag_search",
        exploding_double(
            RuntimeError("backend exploded"),
            reason="the raising retrieval seam must actually be reached",
        ),
    )
    screen = _auto_rag_screen(service=_RecordingRagService(_rows(2)))

    await screen._retrieval._maybe_auto_retrieve_for_send("what changed in the notes")

    assert screen._pending_console_launch_context is None
    assert screen.notices == [
        (retrieval_module.CONSOLE_AUTO_RAG_FAILED_NOTICE, "warning")
    ]


@pytest.mark.asyncio
async def test_failed_outcome_is_reported_not_silently_swallowed(monkeypatch):
    """`run_library_rag_search` converts backend errors into a failed OUTCOME."""
    _enable_auto_retrieve()
    _patch_scope(monkeypatch)
    monkeypatch.setattr(
        retrieval_module,
        "run_library_rag_search",
        AsyncMock(return_value=LibraryRagSearchOutcome(status="failed")),
    )
    screen = _auto_rag_screen(service=_RecordingRagService(()))

    await screen._retrieval._maybe_auto_retrieve_for_send("what changed in the notes")

    assert screen._pending_console_launch_context is None
    assert screen.notices == [
        (retrieval_module.CONSOLE_AUTO_RAG_FAILED_NOTICE, "warning")
    ]


# --------------------------------------------------------------------------
# Happy path, end to end through the real send
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_happy_path_stages_then_send_consumes(monkeypatch):
    """Auto-retrieval stages a bundle the SAME send consumes and prompts."""
    _enable_auto_retrieve()
    _patch_scope(monkeypatch)
    app = _use_disk_config(_build_test_app())
    service = _RecordingRagService(_rows(2))
    app.library_rag_search_service = service
    context = "[S1] MEDIA — Source 1\nBody 1"
    capture = AsyncMock(
        return_value=LocalRagContextResult(context=context, citation_builder=None)
    )
    monkeypatch.setattr(
        retrieval_module, "capture_console_staged_evidence_for_chat", capture
    )

    async with ConsoleHarness(app).run_test(size=(180, 48)) as pilot:
        screen = pilot.app.screen_stack[-1]
        await _wait_for_selector(screen, pilot, "#console-native-composer")
        assert screen._pending_console_launch_context is None

        controller = screen._ensure_console_chat_controller()
        gateway = _CapturingGateway()
        controller.provider_gateway = gateway
        controller._agent_runtime_enabled = False

        result = await controller.submit_draft("what changed in the notes")
        await pilot.pause()

        assert result.accepted is True
        # Retrieval ran once, for this draft, in profile-driven RAG mode.
        assert len(service.calls) == 1
        assert service.calls[0]["query"] == "what changed in the notes"
        assert service.calls[0]["mode"] == "rag"
        # The auto-staged launch is what the capture received...
        auto_launch = capture.await_args_list[0].args[1]
        assert auto_launch is not None
        assert auto_launch.source == "Library Search/RAG"
        assert auto_launch.payload["evidence_bundle"]["references"]
        # ...and this send consumed it.
        assert screen._pending_console_launch_context is None
        assert _final_user_content(gateway.captured) == (
            f"Evidence: {context}\n\n---\n\nwhat changed in the notes"
        )
        strip_text = " ".join(
            str(node.renderable) for node in screen.query(f"{STRIP_ID} Static")
        )
        assert "Evidence sent with this message" in strip_text


@pytest.mark.asyncio
async def test_send_proceeds_when_auto_retrieve_fails(monkeypatch):
    """Retrieval failure is never a send blocker.

    The exploding double is registered (task-15270): this test was green for
    two months while retrieval never fired at all, asserting only that an
    ordinary send works (task-15210). A never-called double is now a failure
    on its own, so the same silent degradation cannot recur.
    """
    _enable_auto_retrieve()
    _patch_scope(monkeypatch)
    app = _use_disk_config(_build_test_app())
    # Both guards are wanted here, and they cover different gaps:
    # ``exploding_double`` registers itself so a never-called failure
    # double fails the test even if nobody remembers to assert it
    # (task-15270), while the explicit ``await_count`` below states the
    # claim at the assertion site (task-15210). Bound to a name so the
    # registered double IS the one asserted on.
    exploding_search = exploding_double(
        RuntimeError("backend exploded"),
        reason="the send must be shown surviving a retrieval failure",
    )
    monkeypatch.setattr(retrieval_module, "run_library_rag_search", exploding_search)

    async with ConsoleHarness(app).run_test(size=(180, 48)) as pilot:
        screen = pilot.app.screen_stack[-1]
        await _wait_for_selector(screen, pilot, "#console-native-composer")
        controller = screen._ensure_console_chat_controller()
        gateway = _CapturingGateway()
        controller.provider_gateway = gateway
        controller._agent_runtime_enabled = False

        result = await controller.submit_draft("what changed in the notes")
        await pilot.pause()

        # Retrieval was actually attempted. Without this the test passes for
        # the wrong reason whenever the toggle fails to reach the hook -- it
        # did exactly that between task-14803 and task-15210, silently
        # asserting only that an ordinary send works.
        assert exploding_search.await_count == 1
        assert result.accepted is True
        assert screen._pending_console_launch_context is None
        assert _final_user_content(gateway.captured) == "what changed in the notes"


def test_send_kind_classification_is_shared_not_duplicated():
    """`_is_plain_text_send` reads the grammar's own prefixes."""
    from tldw_chatbook.Chat.console_command_grammar import COMMAND_PREFIX
    from tldw_chatbook.Chat.console_skill_resolver import MENTION_SIGIL

    assert retrieval_module.is_plain_text_send("a plain question") is True
    assert retrieval_module.is_plain_text_send(f"{COMMAND_PREFIX}prompt x") is False
    assert retrieval_module.is_plain_text_send(f"{MENTION_SIGIL}skill x") is False
    assert retrieval_module.is_plain_text_send("   ") is False
    assert retrieval_module.is_plain_text_send(None) is False


@pytest.mark.asyncio
async def test_capture_seam_calls_the_hook_before_consuming(monkeypatch):
    """The hook must run at the consume-on-send seam, ahead of the consume.

    Placing it anywhere later would stage a bundle the CURRENT send can no
    longer pick up.
    """
    order: list[str] = []
    seen_contexts: list = []
    turn_context = ConsoleTurnConfigurationSnapshot.capture(
        session_id="s1",
        provider_selection=ConsoleProviderSelection(provider="test-provider"),
    )

    async def _hook(_draft, turn_context=None):
        # Production signature since task-14803 (commit 5be9e6a04), which
        # threads the turn's frozen context to the hook. The one-argument
        # stub this replaced made the seam's call raise `TypeError`, which
        # the seam's deliberate `except Exception` swallowed -- so the hook
        # never ran and only the ordering assertion below noticed.
        seen_contexts.append(turn_context)
        order.append("auto-retrieve")

    def _consume():
        order.append("consume")
        return None

    screen = _auto_rag_screen()
    screen._retrieval._clear_evidence_sent_notice = lambda: order.append("clear-notice")
    screen._retrieval._maybe_auto_retrieve_for_send = _hook
    screen._retrieval._consume_pending_launch = _consume
    monkeypatch.setattr(
        retrieval_module,
        "capture_console_staged_evidence_for_chat",
        AsyncMock(
            return_value=LocalRagContextResult(context=None, citation_builder=None)
        ),
    )

    await screen._retrieval._capture_console_staged_rag("question", turn_context)

    assert order == ["clear-notice", "auto-retrieve", "consume"]
    # ...and the hook retrieves for THIS turn's captured configuration, not
    # for whatever the live settings happen to say by the time it runs.
    assert seen_contexts == [turn_context]


@pytest.mark.asyncio
async def test_capture_seam_contains_an_exploding_auto_retrieve(monkeypatch):
    """A raising hook must not cost the send its manually staged evidence.

    `_capture_rag_context` converts any provider exception into
    `context=None`, so an escape here would drop the bundle the consume was
    about to hand the model -- an optional convenience taking out the
    deliberate one.
    """
    launch = _staged_launch()
    released: list[tuple] = []
    context = "[S1] MEDIA — Source 1\nBody 1"

    async def _explode(_draft, turn_context=None):
        raise RuntimeError("auto-retrieve exploded")

    screen = _auto_rag_screen()
    screen._retrieval._maybe_auto_retrieve_for_send = _explode
    screen._retrieval._consume_pending_launch = lambda: launch
    screen._retrieval._release_consumed_launch = lambda *args: released.append(args)
    monkeypatch.setattr(
        retrieval_module,
        "capture_console_staged_evidence_for_chat",
        AsyncMock(
            return_value=LocalRagContextResult(context=context, citation_builder=None)
        ),
    )

    result = await screen._retrieval._capture_console_staged_rag("question")

    assert result.context == context
    assert len(released) == 1
