"""Approval choices never confuse a displayed default with deliberate review."""

from html import unescape

import pytest
from Tests.UI.consolidated_css import APP_STYLESHEETS
from tldw_chatbook.Chat.approval_presentation import (
    ApprovalAuthority,
    ApprovalBatchView,
    ApprovalRowView,
)
from tldw_chatbook.UI.Console_Modules.approval_controls import ApprovalDraft
from textual.widgets import Button
from Tests.UI.test_console_mcp_approval import _CardHarnessApp, _sample_calls
from tldw_chatbook.Widgets.Chat_Widgets.chat_approval_card import ChatApprovalCard
from textual.widgets import Select
from Tests.UI.test_console_mcp_approval import _raw_shell_call

pytestmark = pytest.mark.bootstrap_profile


def view(raw=False):
    owner = ApprovalAuthority(
        "raw_shell" if raw else "mcp",
        "default",
        "Default",
        "Console",
        "console_chat" if raw else "profile",
        "call",
        "Settings",
    )
    row = ApprovalRowView(
        "one",
        2,
        "Tool",
        (),
        owner,
        ("approve_once", "approve_session", "deny"),
        raw,
        "",
        (),
    )
    return ApprovalBatchView("round", "chat", "run", 1, (row,), 2, not raw, True)


def test_programmatic_default_does_not_complete_raw_review():
    draft = ApprovalDraft(view(True))
    assert not draft.can_apply()
    assert draft.stage("one", "deny", deliberate=False)
    assert not draft.can_apply()
    assert draft.submit_map() == {}


def test_reselecting_raw_default_deny_is_deliberate():
    draft = ApprovalDraft(view(True))
    assert draft.stage("one", "deny", deliberate=True)
    assert draft.can_apply()
    assert draft.submit_map() == {"one": "deny"}


def test_draft_rejects_unavailable_scope_without_changing_choice():
    draft = ApprovalDraft(view())
    assert not draft.stage("one", "always_allow", deliberate=True)
    assert draft.submit_map() == {"one": "approve_once"}
    assert "2" in draft.summary()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "action,decision", [("approve", "approve_once"), ("deny", "deny")]
)
async def test_single_more_options_is_noncommitting_and_allow_once_is_one_gesture(
    action, decision
):
    app = _CardHarnessApp()
    async with app.run_test(size=(80, 24)) as pilot:
        card = app.query_one(ChatApprovalCard)
        card.set_batch(_sample_calls()[:1], timeout_seconds=0, round_id="single")
        await pilot.pause()
        more = card.query_one(".approval-more-options", Button)
        await pilot.click(more)
        await pilot.pause()
        assert app.decided == []
        await pilot.press("escape")
        await pilot.click(card.query_one(f".approval-row-fast-{action}", Button))
        await pilot.pause()
        assert app.decided == [{"mcp__srv_a__search": decision}]
        assert app.decided_round_ids == ["single"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "action,decision", [("approve", "approve_once"), ("deny", "deny")]
)
async def test_bulk_commits_complete_counted_batch_immediately(action, decision):
    app = _CardHarnessApp()
    async with app.run_test(size=(120, 40)) as pilot:
        card = app.query_one(ChatApprovalCard)
        card.set_batch(_sample_calls(), timeout_seconds=0, round_id="bulk")
        await pilot.pause()
        button = card.query_one(f"#approval-{action}-all", Button)
        assert "3" in str(button.label)
        await pilot.click(button)
        await pilot.pause()
        assert app.decided == [
            {"mcp__srv_a__search": decision, "mcp__srv_b__write": decision}
        ]


@pytest.mark.asyncio
async def test_alt_a_enter_cannot_commit():
    app = _CardHarnessApp()
    async with app.run_test(size=(80, 24)) as pilot:
        card = app.query_one(ChatApprovalCard)
        card.set_batch(_sample_calls()[:1], timeout_seconds=0, round_id="neutral")
        await pilot.pause()
        card.focus_first_decision()
        await pilot.press("enter")
        await pilot.pause()
        assert app.focused.id == "approval-request-summary"
        assert app.decided == []


@pytest.mark.asyncio
async def test_stale_more_options_cannot_change_replacement():
    app = _CardHarnessApp()
    async with app.run_test() as pilot:
        card = app.query_one(ChatApprovalCard)
        card.set_batch(_sample_calls()[:1], timeout_seconds=0, round_id="old")
        await pilot.pause()
        card.query_one(".approval-more-options", Button).press()
        card.set_batch(_sample_calls()[:1], timeout_seconds=0, round_id="new")
        await pilot.pause()
        assert not card.has_class("approval-options-open")
        assert not card.query_one(".approval-row-decision", Select).display
        assert app.decided == []


@pytest.mark.asyncio
async def test_raw_same_default_deny_selection_completes_mixed_review():
    app = _CardHarnessApp()
    async with app.run_test(size=(120, 40)) as pilot:
        card = app.query_one(ChatApprovalCard)
        card.set_batch(
            [*_sample_calls()[:1], _raw_shell_call("echo approved")],
            timeout_seconds=0,
            round_id="raw",
        )
        await pilot.pause()
        card.query_one("#approval-submit", Button).press()
        await pilot.pause()
        assert app.decided == []
        await pilot.click(card.query_one("#approval-more-options-batch", Button))
        raw = list(card.query(".approval-row-decision"))[1]
        raw.focus()
        await pilot.press("enter", "enter")
        await pilot.pause()
        card.query_one("#approval-submit", Button).press()
        await pilot.pause()
        assert app.decided == [
            {"mcp__srv_a__search": "approve_once", "raw-call-1": "deny"}
        ]


@pytest.mark.asyncio
async def test_more_options_scope_choice_requires_explicit_apply():
    app = _CardHarnessApp()
    async with app.run_test(size=(120, 40)) as pilot:
        card = app.query_one(ChatApprovalCard)
        call = _sample_calls()[0]
        card.set_batch([call], timeout_seconds=0, round_id="scope")
        await pilot.pause()
        await pilot.click(card.query_one(".approval-more-options", Button))
        select = card.query_one(".approval-row-decision", Select)
        select.focus()
        await pilot.press("enter", "down", "enter")
        await pilot.pause()
        assert select.value == "approve_session"
        assert app.decided == []
        await pilot.click(card.query_one("#approval-submit", Button))
        await pilot.pause()
        assert app.decided == [{"mcp__srv_a__search": "approve_session"}]
        assert app.decided_round_ids == ["scope"]


@pytest.mark.asyncio
async def test_captured_replacement_revision_rejects_queued_decision():
    from dataclasses import replace

    app = _CardHarnessApp()
    async with app.run_test() as pilot:
        card = app.query_one(ChatApprovalCard)
        first = view()
        calls = [
            {
                "llm_name": "one",
                "call_id": "one",
                "tool_name": "read_file",
                "arguments": {"path": "preview"},
            }
        ] * 2
        card.set_batch(
            calls,
            timeout_seconds=0,
            round_id="round",
            view=first,
            presentation_revision=1,
        )
        await pilot.pause()
        card.query_one("#approval-approve-all", Button).press()
        card.set_batch(
            calls,
            timeout_seconds=0,
            round_id="round",
            view=replace(first, revision=2),
            presentation_revision=2,
        )
        await pilot.pause()
        assert app.decided == []
        assert not card.query_one("#approval-submit", Button).disabled


@pytest.mark.asyncio
async def test_bulk_once_cannot_fall_back_to_temporary_scope():
    app = _CardHarnessApp()
    async with app.run_test() as pilot:
        card = app.query_one(ChatApprovalCard)
        calls = _sample_calls()
        calls[0]["options"] = ["approve_session", "deny"]
        card.set_batch(calls, timeout_seconds=0, round_id="narrowed")
        await pilot.pause()
        button = card.query_one("#approval-approve-all", Button)
        assert not button.disabled and str(button.label) == "Review individually"
        button.press()
        await pilot.pause()
        assert app.decided == []


class _PaintedCardHarness(_CardHarnessApp):
    """Load shipping source sheets for painted presentation assertions."""

    CSS_PATH = list(APP_STYLESHEETS)


def _owner_payload(*, count=1, warning=False, args=None, stamp_domain="call"):
    from dataclasses import replace
    from tldw_chatbook.Agents.mcp_tool_provider import MCPPendingCall
    from tldw_chatbook.Chat.approval_presentation import profile_authority
    from tldw_chatbook.Chat.console_chat_controller import _build_approval_payload

    pending = [
        MCPPendingCall(
            llm_name=f"tool-{index}",
            server_key="external:review",
            tool_name="read_file",
            server_label="External",
            arguments=args or {"path": f"target-{index}.md"},
            reason="ask",
            options=(
                "approve_once",
                "approve_session",
                "allow_matching",
                "always_allow",
                "deny",
            ),
            call_id=f"key-{index}",
            path_precheck_failed=warning,
            presentation_authority=profile_authority(
                "mcp", "Writer", "Captured location", stamp_domain
            ),
        )
        for index in range(min(count, 2))
    ]
    payload = _build_approval_payload("review-round", "chat", "run", pending, 0, None)
    if count > 2:
        payload["view"] = replace(
            payload["view"],
            call_count=count,
            rows=tuple(
                replace(row, call_count=count // 2) for row in payload["view"].rows
            ),
        )
    return payload


def _set_payload(card, payload, *, round_id="review-round"):
    from dataclasses import replace

    card.set_batch(
        payload["calls"],
        timeout_seconds=0,
        round_id=round_id,
        view=replace(payload["view"], round_id=round_id),
        presentation_revision=payload["presentation_revision"],
    )


def _assert_painted(app, widget):
    assert app.screen.region.contains_region(widget.region), (
        widget.region,
        widget.parent.region,
        widget.parent.styles.layout,
        app.query_one(ChatApprovalCard).classes,
    )
    hit, _ = app.screen.get_widget_at(*widget.region.center)
    assert hit is widget


@pytest.mark.asyncio
async def test_captured_warning_survives_reused_row_and_is_painted():
    from textual.widgets import Static

    app = _PaintedCardHarness()
    async with app.run_test(size=(120, 40)) as pilot:
        card = app.query_one(ChatApprovalCard)
        payload = _owner_payload(warning=True)
        _set_payload(card, payload)
        await pilot.pause()
        for round_id in ("review-round", "replacement"):
            if round_id == "replacement":
                _set_payload(card, payload, round_id=round_id)
                await pilot.pause()
            header = card.query_one(".approval-row-header", Static)
            assert "will fail even if approved" in str(header.content).lower()
            _assert_painted(app, header)


@pytest.mark.asyncio
async def test_captured_header_redacts_secrets_and_preserves_long_targets():
    from textual.widgets import Static

    secret = "sk-test-" + "A" * 32
    app = _PaintedCardHarness()
    async with app.run_test(size=(120, 40)) as pilot:
        card = app.query_one(ChatApprovalCard)
        _set_payload(
            card, _owner_payload(args={"api_key": "private-key-value", "note": secret})
        )
        await pilot.pause()
        text = str(card.query_one(".approval-row-header", Static).content)
        assert "private-key-value" not in text and secret not in text
        prefix = "/" + "directory/" * 40
        payload = _owner_payload(count=2)
        from dataclasses import replace

        payload["view"] = replace(
            payload["view"],
            rows=tuple(
                replace(row, targets=(prefix + suffix,))
                for row, suffix in zip(payload["view"].rows, ("one.md", "two.md"))
            ),
        )
        _set_payload(card, payload, round_id="long-targets")
        await pilot.pause()
        headers = list(card.query(".approval-row-header"))
        assert prefix + "one.md" in str(headers[0].content)
        assert prefix + "two.md" in str(headers[1].content)


@pytest.mark.asyncio
async def test_single_commit_names_selected_scope_and_escape_returns_to_opener():
    app = _PaintedCardHarness()
    async with app.run_test(size=(120, 40)) as pilot:
        card = app.query_one(ChatApprovalCard)
        _set_payload(card, _owner_payload())
        await pilot.pause()
        more = card.query_one(".approval-more-options", Button)
        await pilot.click(more)
        select = card.query_one(Select)
        select.value = "approve_session"
        await pilot.pause()
        submit = card.query_one("#approval-submit", Button)
        assert "Until Chatbook exits" in str(submit.label)
        _assert_painted(app, submit)
        await pilot.press("escape")
        assert app.focused is more
        assert app.decided == []
        await pilot.click(more)
        select.value = "always_allow"
        await pilot.pause()
        assert "Remember this tool" in str(submit.label)
        await pilot.click(submit)
        await pilot.pause()
        assert app.decided == [{"key-0": "always_allow"}]
        assert app.decided_round_ids == ["review-round"]


@pytest.mark.asyncio
async def test_mixed_commit_paints_complete_count_and_scope_summary():
    from textual.widgets import Static

    app = _PaintedCardHarness()
    async with app.run_test(size=(120, 40)) as pilot:
        card = app.query_one(ChatApprovalCard)
        _set_payload(card, _owner_payload(count=2))
        await pilot.pause()
        await pilot.click(card.query_one("#approval-more-options-batch", Button))
        await pilot.pause()
        selects = list(card.query(Select))
        selects[0].value = "approve_session"
        selects[1].value = "deny"
        await pilot.pause()
        summary = card.query_one("#approval-count-scope-summary", Static)
        assert summary.display
        text = str(summary.content)
        assert (
            "Allow 1" in text
            and "Deny 1" in text
            and "Until Chatbook exits" in text
            and "Writer" in text
        )
        _assert_painted(app, summary)
        await pilot.click(card.query_one("#approval-submit", Button))
        await pilot.pause()
        assert app.decided == [{"key-0": "approve_session", "key-1": "deny"}]
        assert app.decided_round_ids == ["review-round"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "action,decision", [("approve", "approve_once"), ("deny", "deny")]
)
async def test_narrow_counted_bulk_actions_remain_distinct_and_exact(action, decision):
    from textual.widgets import Static

    count = 12345678901234567890
    app = _PaintedCardHarness()
    async with app.run_test(size=(40, 40)) as pilot:
        card = app.query_one(ChatApprovalCard)
        _set_payload(card, _owner_payload(count=count))
        await pilot.pause()
        allow = card.query_one("#approval-approve-all", Button)
        deny = card.query_one("#approval-deny-all", Button)
        assert "Allow" in str(allow.label) and "once" in str(allow.label)
        assert "Deny" in str(deny.label) and str(allow.label) != str(deny.label)
        summary = card.query_one("#approval-count-scope-summary", Static)
        assert str(count) in str(summary.content)
        for widget in (allow, deny, summary):
            _assert_painted(app, widget)
        await pilot.click(allow if action == "approve" else deny)
        await pilot.pause()
        assert app.decided == [{"key-0": decision, "key-1": decision}]
        assert app.decided_round_ids == ["review-round"]


def test_draft_summary_binds_each_tool_to_its_selected_scope():
    from dataclasses import replace

    first = replace(
        view().rows[0], verdict_key="search", call_count=1, action_label="Search"
    )
    second = replace(first, verdict_key="write", action_label="Write")
    draft = ApprovalDraft(replace(view(), rows=(first, second)))
    draft.stage("search", "approve_session", deliberate=True)
    draft.stage("write", "approve_once", deliberate=True)
    text = draft.summary()
    assert "Search: Until Chatbook exits" in text
    assert "Write: Allow this call once" in text
    assert "Allow 2" in text and "Deny 0" in text


@pytest.mark.asyncio
async def test_many_tool_scope_summary_keeps_commit_painted_at_compact_size():
    """Full per-tool scope prose must not push Apply outside the viewport."""
    from tldw_chatbook.Agents.mcp_tool_provider import MCPPendingCall
    from tldw_chatbook.Chat.approval_presentation import profile_authority
    from tldw_chatbook.Chat.console_chat_controller import _build_approval_payload
    from textual.widgets import Static

    pending = [
        MCPPendingCall(
            llm_name=f"tool-{index}",
            server_key="external:review",
            tool_name=f"tool-{index}",
            server_label="External",
            arguments={"path": f"target-{index}.md"},
            reason="ask",
            options=("approve_once", "approve_session", "deny"),
            call_id=f"key-{index}",
            presentation_authority=profile_authority(
                "mcp", "Writer", "Captured location", "call"
            ),
        )
        for index in range(10)
    ]
    payload = _build_approval_payload("review-round", "chat", "run", pending, 0, None)
    app = _PaintedCardHarness()
    async with app.run_test(size=(80, 24)) as pilot:
        card = app.query_one(ChatApprovalCard)
        _set_payload(card, payload)
        await pilot.pause()
        await pilot.click(card.query_one("#approval-more-options-batch", Button))
        await pilot.pause()
        for select in card.query(Select):
            select.value = "approve_session"
        await pilot.pause()
        summary = card.query_one("#approval-count-scope-summary", Static)
        assert "Allow 10" in str(summary.content)
        assert "Deny 0" in str(summary.content)
        assert "Until Chatbook exits" in str(summary.content)
        submit = card.query_one("#approval-submit", Button)
        _assert_painted(app, submit)
        headers = list(card.query(".approval-row-header"))
        assert len(headers) == 10
        for index, header in enumerate(headers):
            assert f"tool-{index}" in str(header.content)
            assert "Writer" in str(header.content)
        scopes = list(card.query(".approval-row-scope"))
        assert len(scopes) == 10
        assert all("Until Chatbook exits" in str(scope.content) for scope in scopes)
        await pilot.click(submit)
        await pilot.pause()
        assert app.decided == [
            {f"key-{index}": "approve_session" for index in range(10)}
        ]
        assert app.decided_round_ids == ["review-round"]


@pytest.mark.asyncio
async def test_batch_broader_scopes_require_disclosure_and_survive_escape():
    from textual.widgets import Static

    app = _PaintedCardHarness()
    async with app.run_test(size=(80, 24)) as pilot:
        card = app.query_one(ChatApprovalCard)
        _set_payload(card, _owner_payload(count=2))
        await pilot.pause()
        assert all(not select.display for select in card.query(Select))
        more = card.query_one("#approval-more-options-batch", Button)
        _assert_painted(app, more)
        await pilot.click(more)
        select = list(card.query(Select))[0]
        select.focus()
        await pilot.press("enter", "down", "enter")
        assert select.value == "approve_session" and not app.decided
        await pilot.press("escape")
        assert app.focused is more and not select.display
        assert select.value == "approve_session" and not app.decided
        summary = card.query_one("#approval-count-scope-summary", Static)
        assert "Until Chatbook exits" in str(summary.content)
        await pilot.click(more)
        submit = card.query_one("#approval-submit", Button)
        _assert_painted(app, submit)
        await pilot.click(submit)
        await pilot.pause()
        assert app.decided == [{"key-0": "approve_session", "key-1": "approve_once"}]


@pytest.mark.asyncio
async def test_ineligible_bulk_routes_to_individual_review_without_accepting_raw_default():
    app = _PaintedCardHarness()
    async with app.run_test(size=(80, 24)) as pilot:
        card = app.query_one(ChatApprovalCard)
        card.set_batch(
            [*_sample_calls()[:1], _raw_shell_call("echo sample")],
            timeout_seconds=0,
            round_id="mixed",
        )
        await pilot.pause()
        review = card.query_one("#approval-approve-all", Button)
        assert str(review.label) == "Review individually" and not review.disabled
        _assert_painted(app, review)
        await pilot.click(review)
        assert app.focused in list(card.query(Select))
        assert all(select.display for select in card.query(Select))
        submit = card.query_one("#approval-submit", Button)
        _assert_painted(app, submit)
        await pilot.click(submit)
        assert not app.decided
        raw = list(card.query(Select))[1]
        raw.focus()
        await pilot.press("enter", "enter")
        await pilot.click(submit)
        await pilot.pause()
        assert app.decided == [
            {"mcp__srv_a__search": "approve_once", "raw-call-1": "deny"}
        ]


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["more", "review"])
async def test_batch_disclosure_gesture_cannot_open_a_replacement_snapshot(action):
    from dataclasses import replace

    app = _PaintedCardHarness()
    async with app.run_test(size=(80, 24)) as pilot:
        card = app.query_one(ChatApprovalCard)
        payload = _owner_payload(count=2)
        if action == "review":
            payload["view"] = replace(payload["view"], bulk_once=False)
        _set_payload(card, payload)
        await pilot.pause()
        selector = (
            "#approval-more-options-batch"
            if action == "more"
            else "#approval-approve-all"
        )
        old = card.query_one(selector, Button)
        old.press()
        payload["view"] = replace(payload["view"], revision=2)
        payload["presentation_revision"] = 2
        _set_payload(card, payload)
        await pilot.pause()
        assert not card._options_open
        assert all(not select.display for select in card.query(Select))
        assert not app.decided


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(80, 24), (120, 40)])
async def test_repeated_tool_withheld_scope_is_explained_only_after_disclosure(size):
    """Removing a legal choice must not silently hide its captured limitation."""
    from textual.widgets import Static

    payload = _owner_payload(count=2, stamp_domain="tool_name")
    assert all(row.withheld_scope_copy for row in payload["view"].rows)
    assert all(
        "allow_matching" not in row.legal_decisions for row in payload["view"].rows
    )
    app = _PaintedCardHarness()
    async with app.run_test(size=size) as pilot:
        card = app.query_one(ChatApprovalCard)
        _set_payload(card, payload)
        await pilot.pause()
        notes = list(card.query(".approval-row-withheld-scope"))
        assert len(notes) == 2
        assert all(not note.display for note in notes)
        assert not app.decided

        await pilot.click(card.query_one("#approval-more-options-batch", Button))
        await pilot.pause()
        for row, captured in zip(card.query(".approval-row"), payload["view"].rows):
            note = row.query_one(".approval-row-withheld-scope", Static)
            assert note.display
            assert str(note.content) == captured.withheld_scope_copy
            assert note.region.width > 0 and note.region.height > 0
            assert row.region.contains_region(note.region)
            note.scroll_visible(animate=False)
            await pilot.pause()
            _assert_painted(app, note)
            assert card.query_one(
                "#approval-batch-rows"
            ).content_region.contains_region(note.region)
            for action_id in (
                "#approval-more-options-batch",
                "#approval-submit",
                "#approval-deny-all",
            ):
                action = card.query_one(action_id, Button)
                _assert_painted(app, action)
                assert action.region.width >= len(action.label.plain)
            assert "Remember these inputs is unavailable" in unescape(
                app.export_screenshot()
            ).replace("\xa0", " ")
        assert all(select.value == "approve_once" for select in card.query(Select))
        assert not app.decided

        # Ordinary re-sync preserves the disclosed limitation and staged scopes.
        _set_payload(card, payload)
        await pilot.pause()
        assert card.has_class("approval-options-open")
        assert all(note.display for note in notes)
        await pilot.press("escape")
        assert all(not note.display for note in notes)
        assert not app.decided


@pytest.mark.asyncio
async def test_withheld_scope_copy_refreshes_in_reused_row_and_clears_when_unaffected():
    """A reused row must neither lose captured copy nor keep a prior limitation."""
    from dataclasses import replace
    from textual.widgets import Static

    original = _owner_payload()
    limited = dict(original)
    limited["view"] = replace(
        original["view"],
        rows=(
            replace(
                original["view"].rows[0],
                legal_decisions=("approve_once", "approve_session", "deny"),
                withheld_scope_copy="Captured limitation [literal]. Allow once remains available.",
            ),
        ),
    )
    app = _PaintedCardHarness()
    async with app.run_test(size=(120, 40)) as pilot:
        card = app.query_one(ChatApprovalCard)
        _set_payload(card, original)
        await pilot.pause()
        row = card.query_one(".approval-row")
        note = row.query_one(".approval-row-withheld-scope", Static)
        assert not note.display and not str(note.content)

        for round_id, payload in (
            ("limited", limited),
            ("limited-again", limited),
            ("restored", original),
        ):
            _set_payload(card, payload, round_id=round_id)
            await pilot.pause()
            assert card.query_one(".approval-row") is row
            assert row.query_one(".approval-row-withheld-scope", Static) is note
            assert not note.display
            await pilot.click(card.query_one(".approval-more-options", Button))
            await pilot.pause()
            expected = payload["view"].rows[0].withheld_scope_copy
            assert note.display == bool(expected)
            assert str(note.content) == expected
            if expected:
                _assert_painted(app, note)
                assert "[literal]" in app.export_screenshot()
            assert not app.decided
            await pilot.press("escape")
            assert not note.display
