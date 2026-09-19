"""Behavioral replacements for the retired nested-expander Inspector UI."""

import asyncio
from dataclasses import replace

import pytest
from textual.widgets import (
    Button,
    Checkbox,
    Static,
    TabbedContent,
    TextArea,
)

from Tests.UI.test_console_conversation_inspector import (
    InspectorHarness,
    _capture,
    _capture_policy_bindings_for_inspector,
    _default_kwargs,
    _row,
    _turn,
    _wait_until,
)
from tldw_chatbook.Chat.console_chat_models import (
    ConsoleChatMessage,
    ConsoleContextSnapshot,
    ConsoleMessageRole,
)
from tldw_chatbook.Chat.console_cost_tracker import build_cost_rows_totals
from tldw_chatbook.Chat.trace_export_profiles import TraceViewerProfile
from tldw_chatbook.Widgets.Console.console_conversation_inspector import (
    SIZE_THRESHOLD_BYTES,
    TAB_COSTS,
    TAB_EXCHANGE,
    TAB_NEXT_SEND,
)
from tldw_chatbook.Widgets.Console.console_exchange_export_dialog import (
    ConsoleExchangeExportDialog,
)
from tldw_chatbook.Widgets.Console.console_inspector_detail_pane import (
    ConsoleInspectorDetailPane,
)


def pane(modal, view):
    return modal.query_one(
        f"#console-inspector-{view}-detail", ConsoleInspectorDetailPane
    )


async def choose(pilot, modal, view, key, expected):
    reader = pane(modal, view)
    reader.select(key)
    await _wait_until(pilot, lambda: expected in reader.query_one(TextArea).text)
    return reader.query_one(TextArea).text


@pytest.mark.asyncio
async def test_usage_loads_selected_identity_only_and_preserves_totals():
    calls = []

    async def loader(message_id):
        calls.append(message_id)
        return []

    rows = [_row(), replace(_row(1), cost_usd=None, estimated=True)]
    app = InspectorHarness(
        **_default_kwargs(
            rows=rows,
            totals=build_cost_rows_totals(rows),
            turns=[_turn(1, native_message_id="second"), _turn()],
            exchanges_loader=loader,
        )
    )
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        assert calls == []
        modal = app.screen
        totals = str(
            modal.query_one("#console-inspector-costs-totals", Static).render()
        )
        assert "unavailable" in totals
        text = await choose(pilot, modal, "usage", "usage:1", "No capture recorded")
        assert calls == ["second"]
        assert "Estimated" in text and "unpriced" in text
        assert (
            str(modal.query_one("#console-inspector-costs-totals", Static).render())
            == totals
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("rows", [[], [replace(_row(), cost_usd=0)]])
async def test_empty_and_zero_cost_are_distinct(rows):
    app = InspectorHarness(
        **_default_kwargs(rows=rows, totals=build_cost_rows_totals(rows))
    )
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        if rows:
            text = await choose(
                pilot, app.screen, "usage", "usage:0", "No capture recorded"
            )
            assert "$0.0000" in text
        else:
            assert pane(app.screen, "usage").sections == ()
            assert "0 rows" in str(
                app.screen.query_one("#console-inspector-costs-totals", Static).render()
            )


@pytest.mark.asyncio
async def test_capture_failure_keeps_usage_and_can_retry():
    calls = 0

    async def loader(_):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ValueError("private exception body")
        return []

    app = InspectorHarness(**_default_kwargs(exchanges_loader=loader))
    async with app.run_test(size=(120, 40)) as pilot:
        text = await choose(
            pilot, app.screen, "usage", "usage:0", "Could not load captures"
        )
        assert "$0.0010" in text and "private exception body" not in text
        await choose(pilot, app.screen, "usage", "usage:0", "No capture recorded")
        assert calls == 2


@pytest.mark.asyncio
async def test_exchange_calls_are_chronological_lazy_literal_and_export_governed():
    later = _capture(
        "aaa",
        1,
        "2026-09-02",
        "m[/]",
        request={
            "system_message": "private prompt",
            "messages_payload": [{"role": "user", "content": "hello"}],
            "tools": [{"name": "clock"}],
            "temp": 0.7,
        },
        response={"content": "world"},
        omitted_keys=("api_key",),
    )
    earlier = _capture(
        "zzz", 0, "2026-09-01", "early", response={"content": "early answer"}
    )

    async def loader(_):
        return [(later, True), (earlier, False)]

    app = InspectorHarness(
        **_default_kwargs(exchanges_loader=loader, initial_tab=TAB_EXCHANGE)
    )
    async with app.run_test(size=(120, 40)) as pilot:
        modal = app.screen
        assert not modal._trace_calls
        await choose(pilot, modal, "exchange", "turn:0", "abandoned")
        sections = pane(modal, "exchange").sections
        assert [row.key for row in sections] == ["turn:0", "call:n1:0", "call:n1:1"]
        assert "early" in sections[1].label
        assert all("private prompt" not in row.label for row in sections)
        text = await choose(pilot, modal, "exchange", "call:n1:1", "private prompt")
        assert all(
            value in text
            for value in ("hello", "clock", "world", "temp", "api_key", "m[/]")
        )
        assert modal._open_exchange_export("n1:1")
        await pilot.pause()
        assert isinstance(app.screen, ConsoleExchangeExportDialog)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status,abandoned",
    [("complete", False), ("stopped", False), ("error", False), ("complete", True)],
)
async def test_call_status_is_explicit(status, abandoned):
    cap = replace(_capture("r", 0, "t", "m"), status=status)

    async def loader(_):
        return [(cap, abandoned)]

    app = InspectorHarness(
        **_default_kwargs(exchanges_loader=loader, initial_tab=TAB_EXCHANGE)
    )
    async with app.run_test(size=(120, 40)) as pilot:
        text = await choose(
            pilot,
            app.screen,
            "exchange",
            "turn:0",
            "abandoned" if abandoned else status,
        )
        assert ("abandoned" if abandoned else status) in text


@pytest.mark.asyncio
async def test_synthetic_response_is_not_claimed_as_model_output():
    cap = replace(
        _capture(
            "r",
            0,
            "t",
            "m",
            response={"content": "local fallback", "synthetic_fallback": True},
        ),
        trace_provenance="synthetic_fallback",
    )

    async def loader(_):
        return [(cap, False)]

    app = InspectorHarness(
        **_default_kwargs(exchanges_loader=loader, initial_tab=TAB_EXCHANGE)
    )
    async with app.run_test(size=(120, 40)) as pilot:
        await choose(pilot, app.screen, "exchange", "turn:0", "m")
        text = await choose(
            pilot, app.screen, "exchange", "call:n1:0", "Locally synthesized"
        )
        assert "not model output" in text


@pytest.mark.asyncio
@pytest.mark.parametrize("invalidate", ["revision", "target", "dismiss", "disclosure"])
async def test_delayed_capture_cannot_restore_invalid_content(invalidate):
    started, release = asyncio.Event(), asyncio.Event()
    authority = {"valid": True, "revision": 1}

    async def loader(_):
        started.set()
        await release.wait()
        return [
            (
                _capture(
                    "r", 0, "t", "secret model", response={"content": "secret body"}
                ),
                False,
            )
        ]

    app = InspectorHarness(
        **_default_kwargs(
            exchanges_loader=loader,
            initial_tab=TAB_EXCHANGE,
            target_is_current=lambda: authority["valid"],
            capture_revision_provider=lambda: authority["revision"],
        )
    )
    async with app.run_test(size=(120, 40)) as pilot:
        modal = app.screen
        pane(modal, "exchange").select("turn:0")
        await started.wait()
        if invalidate == "revision":
            authority["revision"] = 2
        elif invalidate == "target":
            authority["valid"] = False
        elif invalidate == "dismiss":
            await pilot.press("escape")
        else:
            await modal.action_viewer_profile()
        release.set()
        await pilot.pause()
        assert not modal._exchange_capture_by_call_key
        assert all("secret" not in reader.text for reader in modal.query(TextArea))


@pytest.mark.asyncio
async def test_full_to_safe_clears_both_trace_views_and_export():
    cap = _capture(
        "r",
        0,
        "t",
        "m",
        request={"system_message": "private prompt"},
        response={"content": "private result"},
    )

    async def loader(_):
        return [(cap, False)]

    app = InspectorHarness(
        **_default_kwargs(
            exchanges_loader=loader,
            capture_policy_bindings=_capture_policy_bindings_for_inspector(),
        )
    )
    async with app.run_test(size=(120, 40)) as pilot:
        modal = app.screen
        modal._viewer_profile = TraceViewerProfile.FULL
        await choose(pilot, modal, "usage", "usage:0", "m")
        modal.query_one(TabbedContent).active = TAB_EXCHANGE
        await choose(pilot, modal, "exchange", "turn:0", "m")
        await choose(pilot, modal, "exchange", "call:n1:0", "private prompt")
        await modal.action_viewer_profile()
        assert modal.viewer_profile is TraceViewerProfile.SAFE
        assert not modal._exchange_capture_by_call_key
        assert pane(modal, "usage").query_one(TextArea).text == ""
        assert pane(modal, "exchange").query_one(TextArea).text == ""
        assert modal.query_one("#console-inspector-export-call", Button).disabled


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(80, 24), (120, 40), (160, 48)])
async def test_context_sections_raw_selection_budget_and_reachable_close(size):
    snapshot = ConsoleContextSnapshot(
        current_messages=[
            ConsoleChatMessage(role=ConsoleMessageRole.USER, content="retained message")
        ],
        next_send_payload={
            "model": "m[/]",
            "system": "private instructions",
            "messages": [{"role": "user", "content": "unsent draft"}],
            "tools": [{"name": "clock"}],
            "response_prefill": "prefilled response",
        },
    )

    async def factory():
        return snapshot

    app = InspectorHarness(
        **_default_kwargs(
            snapshot_factory=factory,
            initial_tab=TAB_NEXT_SEND,
            payload_estimate=lambda _: 42,
            context_budget_provider=lambda: (8000, 1000),
        )
    )
    async with app.run_test(size=size) as pilot:
        modal = app.screen
        await _wait_until(pilot, lambda: modal._snapshot_ready)
        keys = {section.key for section in pane(modal, "context").sections}
        assert {
            "current:messages",
            "preview:messages",
            "preview:system",
            "preview:tools",
            "preview:response_prefill",
        } <= keys
        assert all(
            "private instructions" not in section.label
            for section in pane(modal, "context").sections
        )
        text = await choose(
            pilot, modal, "context", "preview:system", "private instructions"
        )
        assert text == "private instructions"
        modal.query_one(Checkbox).value = True
        await _wait_until(
            pilot,
            lambda: (
                '"private instructions"'
                == pane(modal, "context").query_one(TextArea).text
            ),
        )
        header = str(
            modal.query_one("#console-inspector-next-send-header", Static).render()
        )
        assert all(
            item in header for item in ("m[/]", "~42", "8000", "1000", "Prepared now")
        )
        assert await pilot.click("#console-inspector-close")
        await pilot.pause()
        assert app.screen is not modal


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid", ["target", "dismiss"])
async def test_delayed_context_cannot_restore_after_target_loss(invalid):
    started, release = asyncio.Event(), asyncio.Event()
    valid = True

    async def factory():
        started.set()
        await release.wait()
        return ConsoleContextSnapshot(
            current_messages=[], next_send_payload={"system": "private body"}
        )

    app = InspectorHarness(
        **_default_kwargs(
            snapshot_factory=factory,
            initial_tab=TAB_NEXT_SEND,
            target_is_current=lambda: valid,
        )
    )
    async with app.run_test(size=(120, 40)) as pilot:
        modal = app.screen
        await started.wait()
        if invalid == "target":
            valid = False
        else:
            await pilot.press("escape")
        release.set()
        await pilot.pause()
        assert not modal._snapshot_ready
        assert "private body" not in str(modal.snapshot)


@pytest.mark.asyncio
async def test_failed_refresh_preserves_stale_preview_and_busy_refresh_is_disabled():
    calls = 0

    async def factory():
        nonlocal calls
        calls += 1
        if calls > 1:
            raise ValueError("secret exception")
        return ConsoleContextSnapshot(
            current_messages=[], next_send_payload={"system": "previous preview"}
        )

    app = InspectorHarness(
        **_default_kwargs(snapshot_factory=factory, initial_tab=TAB_NEXT_SEND)
    )
    async with app.run_test(size=(120, 40)) as pilot:
        modal = app.screen
        await _wait_until(pilot, lambda: modal._snapshot_ready)
        await choose(pilot, modal, "context", "preview:system", "previous preview")
        modal.action_refresh()
        await _wait_until(pilot, lambda: "stale" in modal._snapshot_status)
        assert "previous preview" in str(modal.snapshot)
        modal._in_progress = True
        modal.action_refresh()
        await pilot.pause()
        assert calls == 2


@pytest.mark.asyncio
async def test_large_section_has_bounded_reader_and_payload_export_guidance():
    async def factory():
        return ConsoleContextSnapshot(
            current_messages=[],
            next_send_payload={"system": "x" * (SIZE_THRESHOLD_BYTES + 1)},
        )

    app = InspectorHarness(
        **_default_kwargs(snapshot_factory=factory, initial_tab=TAB_NEXT_SEND)
    )
    async with app.run_test(size=(120, 40)) as pilot:
        await _wait_until(pilot, lambda: app.screen._snapshot_ready)
        text = await choose(
            pilot, app.screen, "context", "preview:system", "exceeds 1 MiB"
        )
        assert len(text) < 200
        assert "payload" in text


@pytest.mark.asyncio
@pytest.mark.parametrize("invalidate", ["target", "disclosure"])
async def test_open_export_rechecks_parent_authority(invalidate):
    valid = True
    cap = _capture("r", 0, "t", "m", response={"content": "captured response"})

    async def loader(_):
        return [(cap, False)]

    app = InspectorHarness(
        **_default_kwargs(
            exchanges_loader=loader,
            initial_tab=TAB_EXCHANGE,
            target_is_current=lambda: valid,
        )
    )
    async with app.run_test(size=(120, 40)) as pilot:
        modal = app.screen
        await choose(pilot, modal, "exchange", "turn:0", "m")
        await choose(pilot, modal, "exchange", "call:n1:0", "captured response")
        assert modal._open_exchange_export("n1:0")
        await pilot.pause()
        export = app.screen
        assert export._capture_revision_provider() == export._expected_capture_revision
        if invalidate == "target":
            valid = False
        else:
            await modal.action_viewer_profile()
        assert not export._revision_is_current()


@pytest.mark.asyncio
async def test_existing_purge_callback_clears_both_views():
    app = InspectorHarness(**_default_kwargs())
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        modal = app.screen
        pane(modal, "usage").set_detail("usage:0", "Usage", "private historical body")
        await modal._invalidate_stale_exchange_mounts()
        assert pane(modal, "usage").query_one(TextArea).text == ""


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(80, 24), (120, 40), (160, 48)])
@pytest.mark.parametrize(
    "view,tab,key",
    [("usage", TAB_COSTS, "usage:0"), ("exchange", TAB_EXCHANGE, "turn:0")],
)
async def test_historical_readers_stay_within_modal_at_supported_sizes(
    size, view, tab, key
):
    async def loader(_):
        return [
            (
                _capture(
                    "run",
                    1,
                    "2026-09-18T00:00:00Z",
                    "example",
                    request={
                        "messages_payload": [
                            {"role": "user", "content": "long request " * 200}
                        ]
                    },
                ),
                False,
            )
        ]

    app = InspectorHarness(**_default_kwargs(initial_tab=tab, exchanges_loader=loader))
    async with app.run_test(size=size) as pilot:
        modal = app.screen
        detail = pane(modal, view)
        detail.select(key)
        await _wait_until(pilot, lambda: bool(modal._trace_calls))
        if view == "exchange":
            detail.select(
                next(
                    section.key
                    for section in detail.sections
                    if section.key.startswith("call:")
                )
            )
        await pilot.pause()
        reader = detail.query_one(TextArea)
        frame = modal.query_one("#console-inspector-modal")
        assert frame.region.contains_region(detail.region)
        assert detail.region.contains_region(reader.region)
        assert frame.region.contains_region(
            modal.query_one("#console-inspector-close").region
        )
        assert reader.is_on_screen
