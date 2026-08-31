from __future__ import annotations

from types import SimpleNamespace

import pytest
from textual.app import ComposeResult
from textual.widgets import Button, Static

from Tests.UI.consolidated_css import ConsolidatedCSSApp
from tldw_chatbook.Widgets.Settings_Widgets.personal_context_link_modal import (
    PersonalContextLinkModal,
)


class _Host(ConsolidatedCSSApp):
    def compose(self) -> ComposeResult:
        yield Static("host")


def _plan(*, blockers=(), collisions=1, unlinked=1):
    collision_rows = tuple(
        SimpleNamespace(
            decision_id=f"collision-{index}",
            record_ids=(f"local-{index}", f"remote-{index}"),
        )
        for index in range(collisions)
    )
    return SimpleNamespace(
        plan_id="plan-1",
        local_profile_id="profile-local",
        server_profile_id="profile-server",
        exact_record_ids=("record-exact",),
        local_only_record_ids=("record-local",),
        remote_only_record_ids=("record-remote",),
        version_conflicts=(),
        key_collisions=collision_rows,
        device_only_record_ids=("record-private",),
        unlinked_remote_scope_ids=tuple(
            f"scope-{index}" for index in range(unlinked)
        ),
        required_decision_ids=tuple(row.decision_id for row in collision_rows),
        attention_codes=tuple(blockers),
    )


@pytest.mark.asyncio
async def test_modal_shows_bounded_counts_and_requires_every_collision_decision() -> None:
    app = _Host()
    async with app.run_test(size=(100, 32)) as pilot:
        modal = PersonalContextLinkModal(_plan())
        await app.push_screen(modal)
        await pilot.pause()

        visible = " ".join(
            str(widget.renderable) for widget in modal.query(Static)
        )
        assert "1 local addition" in visible
        assert "1 server addition" in visible
        assert "1 collision" in visible
        assert "1 unlinked workspace" in visible
        assert modal.query_one("#personal-context-link-approve", Button).disabled

        await pilot.click("#personal-context-link-collision-0-keep-server")
        await pilot.pause()
        assert not modal.query_one(
            "#personal-context-link-approve", Button
        ).disabled


@pytest.mark.asyncio
async def test_modal_cancel_returns_none_and_attention_disables_approval() -> None:
    app = _Host()
    results = []
    async with app.run_test(size=(100, 32)) as pilot:
        modal = PersonalContextLinkModal(
            _plan(blockers=("schema_incompatible",), collisions=0)
        )
        await app.push_screen(modal, callback=results.append)
        await pilot.pause()

        assert modal.query_one("#personal-context-link-approve", Button).disabled
        assert "Review cannot continue" in str(
            modal.query_one("#personal-context-link-attention", Static).renderable
        )
        await pilot.click("#personal-context-link-cancel")
        await pilot.pause()

    assert results == [None]

