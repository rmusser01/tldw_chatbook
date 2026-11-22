"""TASK-33621.5/.6: a provider error, then the automatic retry's trace fails.

The send goes out once, the provider answers HTTP 429, and the agent's
in-place retry then cannot reserve its trace call, so it pauses on the
"Trace capture blocked" card before anything else is sent. Every test mounts
the real ``ChatScreen`` with its real controller, store, agent loop, gateway,
provider adapter and durable trace runtime on a file-backed database. Only
the HTTP response is fake, and only the retry's reservation is made to fail,
in one of the two ways it fails in the app:

* ``unknown``: SQLite refuses the reservation transaction ("database is
  locked", reproduced live by holding a write lock across the retry). The
  outcome of the reservation is unknown to the gateway.
* ``refused``: the reservation transaction itself refuses the retry
  (``trace_tool_chain_unavailable``, the refusal the review recorded) and
  rolls back, so nothing was reserved.
"""

from __future__ import annotations

import sqlite3
from types import SimpleNamespace

import pytest
from textual.widgets import Button, Static

from Tests.UI.test_console_blocked_send_recovery import (
    _assistant_texts,
    _frame_rows,
    _mounted_console,
    _send,
    _settle_workers,
    _until,
)
from Tests.UI.test_console_provider_error_copy import (
    HTTP_REPLY,
    answer_http,
    real_provider_adapter,
)
from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole, ConsoleRunStatus
from tldw_chatbook.Chat.console_trace_errors import TraceCallPersistenceError
from tldw_chatbook.Chat.console_trace_repository import ConsoleTraceRepository
from tldw_chatbook.UI.Console_Modules.provider_continuation_recovery import (
    TraceCallRecoveryCallout,
)
from tldw_chatbook.Widgets.Console.console_send_authority_summary import (
    project_console_send_authority,
)

pytestmark = pytest.mark.bootstrap_profile

DRAFT = "Name one color"


def fail_the_retry_reservation(h, monkeypatch, kind: str) -> list[object]:
    """Make the second trace reservation (the agent's retry) fail as ``kind``."""
    gateway = h.controller.provider_gateway
    factory = gateway._trace_call_boundary_factory
    routes: list[object] = []
    refusing = SimpleNamespace(active=False)
    original_lookup = ConsoleTraceRepository.get_call_by_logical_identity

    def lookup(self, *args, **kwargs):
        # Inside the reservation transaction: the retry finds no previous
        # call, which the factory refuses as ``trace_tool_chain_unavailable``.
        return None if refusing.active else original_lookup(self, *args, **kwargs)

    monkeypatch.setattr(ConsoleTraceRepository, "get_call_by_logical_identity", lookup)

    def flaky(request, resolution, route):
        routes.append(route)
        if len(routes) != 2:
            return factory(request, resolution, route)
        if kind == "unknown":
            raise sqlite3.OperationalError("database is locked")
        if kind == "dispatch_unknown":
            # A reservation whose dispatch read-back failed: the request may
            # have gone out, so this one is the genuine unknown delivery.
            raise TraceCallPersistenceError(
                boundary=SimpleNamespace(dispatch_outcome="unknown")
            )
        refusing.active = True
        try:
            return factory(request, resolution, route)
        finally:
            refusing.active = False

    monkeypatch.setattr(gateway, "_trace_call_boundary_factory", flaky)
    return routes


async def paused_after_429(h, monkeypatch, kind: str, posts: list[str]):
    real_provider_adapter(h, monkeypatch)
    answer_http(monkeypatch, [429, 200], posts)
    routes = fail_the_retry_reservation(h, monkeypatch, kind)
    await _send(h, DRAFT)
    callout = h.console.query_one(TraceCallRecoveryCallout)
    await _until(
        h.pilot,
        lambda: callout.display,
        timeout=30.0,
        what=f"no recovery card: {h.controller.run_state}",
    )
    assert len(posts) == 1, posts  # the 429; the retry was never sent
    assert len(routes) == 2, routes
    return callout


def visible_text(h) -> str:
    return " ".join(" ".join(_frame_rows(h.host)).split())


async def press(h, callout, action: str) -> None:
    button = callout.query_one(f"#console-trace-{action}", Button)
    button.focus()
    await h.pilot.pause(0.2)
    await h.pilot.press("enter")
    await _settle_workers(h)
    await h.console._sync_native_console_chat_ui()
    await h.pilot.pause(0.3)


def header_label(h) -> str:
    return h.console._build_console_workbench_state(
        h.console._build_console_control_state(None)
    ).header.status_label


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["unknown", "refused"])
async def test_the_card_leads_with_the_provider_error_it_follows(
    tmp_path, monkeypatch, kind
):
    """.5 AC#1/#2/#4: status and meaning first, the trace problem second."""
    async with _mounted_console(tmp_path, monkeypatch, (160, 52)) as h:
        posts: list[str] = []
        callout = await paused_after_429(h, monkeypatch, kind, posts)

        text = visible_text(h)
        assert "provider was not contacted" not in text
        assert "OpenAI returned HTTP 429 (rate limit exceeded)" in text
        assert "Impact: OpenAI was contacted" in text
        assert "the automatic retry was not sent" in text
        retry = callout.query_one("#console-trace-retry", Button)
        assert retry.display and str(retry.label) == "Retry"
        assert retry.variant == "primary"

@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["unknown", "refused"])
async def test_send_without_capture_dispatches_the_turn(tmp_path, monkeypatch, kind):
    """.6 AC#2/#3: no false unknown-delivery panel; the turn is answered."""
    async with _mounted_console(tmp_path, monkeypatch, (160, 52)) as h:
        posts: list[str] = []
        callout = await paused_after_429(h, monkeypatch, kind, posts)

        await press(h, callout, "send-without")
        await _until(
            h.pilot,
            lambda: h.controller.run_state.status is ConsoleRunStatus.COMPLETED,
            timeout=30.0,
            what=f"the send did not complete: {h.controller.run_state}",
        )
        await h.console._sync_native_console_chat_ui()
        await h.pilot.pause(0.3)

        assert len(posts) == 2, posts
        assert _assistant_texts(h) == [HTTP_REPLY]
        assert h.store.dispatch_recovery_for_presentation(h.session.id) is None
        assert "delivery status is unknown" not in visible_text(h)


@pytest.mark.asyncio
async def test_retry_reruns_capture_when_nothing_was_reserved(tmp_path, monkeypatch):
    """.6 AC#2: Retry capture re-runs the captured send and it is answered."""
    async with _mounted_console(tmp_path, monkeypatch, (160, 52)) as h:
        posts: list[str] = []
        callout = await paused_after_429(h, monkeypatch, "refused", posts)

        await press(h, callout, "retry")
        await _until(
            h.pilot,
            lambda: h.controller.run_state.status is ConsoleRunStatus.COMPLETED,
            timeout=30.0,
            what=f"the retry did not complete: {h.controller.run_state}",
        )
        await h.console._sync_native_console_chat_ui()
        await h.pilot.pause(0.3)

        assert len(posts) == 2, posts
        assert _assistant_texts(h) == [HTTP_REPLY]
        assert "delivery status is unknown" not in visible_text(h)
        with h.database.transaction() as cursor:
            states = [
                row[0]
                for row in cursor.execute(
                    "SELECT state FROM console_trace_calls ORDER BY created_at"
                ).fetchall()
            ]
        assert states == ["error", "complete"], states


@pytest.mark.asyncio
async def test_retry_reshows_the_capture_error_when_the_reservation_is_unknown(
    tmp_path, monkeypatch
):
    """.6 AC#2: an unconfirmed reservation keeps the card, never 'unknown delivery'."""
    async with _mounted_console(tmp_path, monkeypatch, (160, 52)) as h:
        posts: list[str] = []
        callout = await paused_after_429(h, monkeypatch, "unknown", posts)

        await press(h, callout, "retry")

        assert len(posts) == 1, posts
        assert callout.display
        status = str(callout.query_one("#console-trace-status", Static).render())
        assert "could not be confirmed" in status, status
        assert h.store.dispatch_recovery_for_presentation(h.session.id) is None
        assert "delivery status is unknown" not in visible_text(h)
        assert header_label(h) == "Blocked"
        # Review follow-up: Capture-On Retry cannot succeed from here, so
        # the action the status line names leads instead.
        retry = callout.query_one("#console-trace-retry", Button)
        send_without = callout.query_one("#console-trace-send-without", Button)
        assert retry.variant != "primary"
        assert send_without.variant == "primary"

        await press(h, callout, "send-without")
        await _until(
            h.pilot,
            lambda: h.controller.run_state.status is ConsoleRunStatus.COMPLETED,
            timeout=30.0,
            what=f"the send did not complete: {h.controller.run_state}",
        )
        assert len(posts) == 2, posts

@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["unknown", "refused"])
async def test_cancel_send_discards_the_turn_and_returns_the_draft(
    tmp_path, monkeypatch, kind
):
    """.6 AC#1: Cancel works even when the reservation status is unknown."""
    async with _mounted_console(tmp_path, monkeypatch, (160, 52)) as h:
        posts: list[str] = []
        callout = await paused_after_429(h, monkeypatch, kind, posts)
        h.console._console_composer_or_none().load_draft("")

        await press(h, callout, "cancel")

        assert not callout.display
        assert "Trace reservation status is unavailable" not in visible_text(h)
        assert h.console._console_composer_or_none().draft_text() == DRAFT
        assert len(posts) == 1
        assert h.controller.trace_call_recovery_preparation() is None
        assert h.store.dispatch_recovery_for_presentation(h.session.id) is None
        assert h.controller.run_state.status is ConsoleRunStatus.STOPPED
        assert header_label(h) != "Blocked"

@pytest.mark.asyncio
async def test_an_unknown_delivery_reads_as_local_and_blocks_the_header(
    tmp_path, monkeypatch
):
    """.6 AC#3/#4: a genuine unknown delivery after a 429.

    The retry's dispatch read-back failed, so it may have gone out: this is
    the one case the duplicate-request warning is for. The session is local,
    so the panel never says "on the source device", and the header chip
    says Blocked with the composer, rail and Inspect handle. (Its Retry
    anyway's busy state is pinned in ``test_console_dispatch_recovery.py``;
    pressing it is the next test.)
    """
    async with _mounted_console(tmp_path, monkeypatch, (160, 52)) as h:
        posts: list[str] = []
        real_provider_adapter(h, monkeypatch)
        answer_http(monkeypatch, [429, 200], posts)
        fail_the_retry_reservation(h, monkeypatch, "dispatch_unknown")
        await _send(h, DRAFT)
        await _until(
            h.pilot,
            lambda: h.store.dispatch_recovery_for_presentation(h.session.id)
            is not None,
            timeout=30.0,
            what=f"no unknown-delivery owner: {h.controller.run_state}",
        )
        await h.console._sync_native_console_chat_ui()
        await h.pilot.pause(0.3)

        text = visible_text(h)
        assert "Response delivery status is unknown." in text
        assert "may send a duplicate request" in text
        assert "on the source device" not in text
        assert "Send blocked — resolve response recovery first" in text
        assert header_label(h) == "Blocked"
        assert project_console_send_authority(
            h.console._build_console_inspector_state(None)
        ).run.startswith("Blocked")


@pytest.mark.asyncio
async def test_retry_anyway_redispatches_a_genuine_unknown_delivery(
    tmp_path, monkeypatch
):
    """.6 AC#5 / .70 AC#1, AC#3: Retry anyway after a 429 and an unknown delivery.

    The composer send staged no evidence, so its request can be rebuilt
    exactly and Retry anyway is offered enabled. Pressing it re-dispatches
    the same turn once and the reply arrives.
    """
    async with _mounted_console(tmp_path, monkeypatch, (160, 52)) as h:
        posts: list[str] = []
        real_provider_adapter(h, monkeypatch)
        answer_http(monkeypatch, [429, 200], posts)
        fail_the_retry_reservation(h, monkeypatch, "dispatch_unknown")
        await _send(h, DRAFT)
        await _until(
            h.pilot,
            lambda: (
                h.store.dispatch_recovery_for_presentation(h.session.id) is not None
            ),
            timeout=30.0,
            what=f"no unknown-delivery owner: {h.controller.run_state}",
        )
        await h.console._sync_native_console_chat_ui()
        await h.pilot.pause(0.3)
        assert len(posts) == 1, posts  # only the 429 went out

        retry = h.console.query_one("#console-dispatch-recovery-retry_anyway", Button)
        assert retry.display and not retry.disabled, visible_text(h)
        assert "cannot be reconstructed exactly" not in visible_text(h)
        retry.focus()
        await h.pilot.pause(0.2)
        await h.pilot.press("enter")
        await _until(
            h.pilot,
            lambda: h.controller.run_state.status is ConsoleRunStatus.COMPLETED,
            timeout=30.0,
            what=f"Retry anyway did not complete: {h.controller.run_state}",
        )
        await h.console._sync_native_console_chat_ui()
        await h.pilot.pause(0.3)

        assert len(posts) == 2, posts
        assert _assistant_texts(h) == [HTTP_REPLY]
        assert h.store.dispatch_recovery_for_presentation(h.session.id) is None
        assert "delivery status is unknown" not in visible_text(h)


def select_a_mixin(h, monkeypatch) -> None:
    """Select one literal mix-in through the app's real mix-in service."""
    from Tests.Chat.test_console_prompt_mixin_admission import A, Scope
    from tldw_chatbook.Chat.console_prompt_mixins import (
        MixinPreferences,
        MixinSelection,
    )

    scope = Scope()
    scope.release.set()
    scope.records[A.record_uuid]["prompt_definition"]["body"] = "body A"
    monkeypatch.setattr(h.controller.app, "prompt_scope_service", scope, raising=False)
    service = h.controller.prompt_mixin_service()
    service.replace_preferences(
        h.session.id,
        expected_revision=service.preferences(h.session.id).revision,
        value=MixinPreferences(entries=(MixinSelection(A, True, "A", "before"),)),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mixin", "regenerate"),
    [(False, True), (True, True), (True, False)],
    ids=["no-mixin-regenerate", "mixin-regenerate", "mixin-next-turn"],
)
async def test_a_recovered_retry_keeps_regenerate_and_the_next_turn_captured(
    tmp_path, monkeypatch, mixin, regenerate
):
    """.6 AC#2: a turn recovered by Retry stays usable with capture on.

    Retry answers in a second run of the turn's one mix-in attempt, which
    keeps its first (failed) trace call. With a mix-in selected, the next
    turn's trace admission must still prove the recovered answer: through
    the attempt binding after Regenerate (an operational replacement), and
    through the turn's surface lineage either way.
    """
    async with _mounted_console(tmp_path, monkeypatch, (160, 52)) as h:
        if mixin:
            select_a_mixin(h, monkeypatch)
            await h.pilot.pause(0.3)
        posts: list[str] = []
        callout = await paused_after_429(h, monkeypatch, "refused", posts)
        await press(h, callout, "retry")
        await _until(
            h.pilot,
            lambda: h.controller.run_state.status is ConsoleRunStatus.COMPLETED,
            timeout=30.0,
            what=f"the retry did not complete: {h.controller.run_state}",
        )
        if regenerate:
            reply = [
                m
                for m in h.store.messages_for_session(h.session.id)
                if m.role is ConsoleMessageRole.ASSISTANT
            ][-1]
            assert (await h.controller.regenerate_message(reply.id)).accepted
        await _send(h, "Name another color")
        await _until(
            h.pilot,
            lambda: h.controller.run_state.status is ConsoleRunStatus.COMPLETED,
            timeout=30.0,
            what=f"the next turn did not complete: {h.controller.run_state}",
        )

        # 429, Retry, [Regenerate,] next turn: the next turn was sent.
        assert len(posts) == (4 if regenerate else 3), posts
        with h.database.transaction() as cursor:
            states = [
                row[0]
                for row in cursor.execute(
                    "SELECT state FROM console_trace_calls ORDER BY created_at"
                ).fetchall()
            ]
            attempts = cursor.execute(
                "SELECT replaces_attempt_id IS NOT NULL, expected_item_count "
                "FROM console_prompt_mixin_attempts ORDER BY rowid"
            ).fetchall()
        # The next turn was captured: its call is the third, complete.
        assert states == ["error", "complete", "complete"], states
        assert sum(row[0] for row in attempts) == int(regenerate)
        assert {row[1] for row in attempts} == {int(mixin)}, attempts


# The run-1 first call's state, forged after the Retry (each with its boundary
# kept, so only the lineage rule's state clause can refuse it).
FORGED_FIRST_CALL = {
    "left_open": (
        "state='reserved', outcome=NULL, dispatch_started_at=NULL, "
        "response_started_at=NULL, settled_at=NULL"
    ),
    "delivery_unknown": "state='dispatch_unknown', outcome=NULL",
    "answered": (
        "state='complete', outcome='complete', response_started_at=settled_at"
    ),
}


@pytest.mark.asyncio
@pytest.mark.parametrize("forged", list(FORGED_FIRST_CALL))
async def test_a_recovered_retry_is_not_proved_unless_its_earlier_run_all_failed(
    tmp_path, monkeypatch, forged
):
    """Negative controls for the lineage rule above: only an all-error run counts.

    The recovered turn's row may name its earlier run only when every call
    of that run ended in error. If the run's first call was left open
    (reserved, never settled), ended with its delivery unknown, or was
    answered, the next captured turn must not prove the recovered answer,
    so it stops on the trace card. The call keeps its call boundary, so the
    rule's state clause (not its missing-boundary clause) is what refuses it.
    """
    async with _mounted_console(tmp_path, monkeypatch, (160, 52)) as h:
        select_a_mixin(h, monkeypatch)
        await h.pilot.pause(0.3)
        posts: list[str] = []
        callout = await paused_after_429(h, monkeypatch, "refused", posts)
        await press(h, callout, "retry")
        await _until(
            h.pilot,
            lambda: h.controller.run_state.status is ConsoleRunStatus.COMPLETED,
            timeout=30.0,
            what=f"the retry did not complete: {h.controller.run_state}",
        )
        with h.database.transaction() as cursor:
            first = cursor.execute(
                "SELECT call_id FROM console_trace_calls WHERE state='error'"
            ).fetchone()[0]
            for guard in ("lifecycle", "terminal", "immutable", "set_once"):
                cursor.execute(f"DROP TRIGGER console_trace_calls_{guard}_guard")
            cursor.execute(
                f"UPDATE console_trace_calls SET {FORGED_FIRST_CALL[forged]} "
                "WHERE call_id=?",
                (first,),
            )
        await _send(h, "Name another color")
        await _until(
            h.pilot,
            lambda: h.controller.run_state.status is ConsoleRunStatus.BLOCKED,
            timeout=30.0,
            what=f"the next turn was not stopped: {h.controller.run_state}",
        )
        assert len(posts) == 2, posts  # the next turn was not sent
