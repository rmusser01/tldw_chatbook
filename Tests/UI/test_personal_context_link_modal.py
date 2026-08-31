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


def _plan(*, blockers=(), collisions=1, unlinked=1, workspace=False):
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
        profile_adoption=("profile-local", "profile-server"),
        global_scope_mapping=("scope-local-global", "scope-server-global"),
        schema_outcome="compatible:3",
        quota_outcome="minimums_satisfied",
        purge_outcome="generation_matches:0",
        quota_outcomes=(("max_record_bytes", 16_384, 32_768, True),),
        record_outcomes=(
            ("record-exact", "exact", "local-v1", "server-v1"),
            ("record-local", "local_addition", "local-v2", None),
        ),
        proposal_outcomes=(
            ("proposal-exact", "exact"),
            ("proposal-diverged", "divergence_attention"),
        ),
        exact_record_ids=("record-exact",),
        exact_proposal_ids=("proposal-exact",),
        local_only_record_ids=("record-local",),
        remote_only_record_ids=("record-remote",),
        version_conflicts=(),
        key_collisions=collision_rows,
        device_only_record_ids=("record-private",),
        unlinked_remote_scope_ids=tuple(
            f"scope-{index}" for index in range(unlinked)
        ),
        local_workspace_scope_ids=("scope-local-workspace",) if workspace else (),
        workspace_new_scope_ids=(
            (("scope-local-workspace", "scope-workspace-reviewed"),)
            if workspace
            else ()
        ),
        workspace_mapping_conflicts=(
            (
                SimpleNamespace(
                    local_scope_id="scope-local-workspace",
                    remote_scope_id="scope-0",
                    record_ids=("record-local-workspace", "record-server-workspace"),
                ),
            )
            if workspace
            else ()
        ),
        required_decision_ids=(
            tuple(row.decision_id for row in collision_rows)
            + (("workspace:scope-local-workspace",) if workspace else ())
        ),
        attention_codes=tuple(blockers),
    )


@pytest.mark.asyncio
async def test_modal_shows_bounded_counts_and_requires_every_collision_decision() -> None:
    app = _Host()
    async with app.run_test(size=(100, 48)) as pilot:
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
        assert "Adopt the server profile identity" in visible
        assert "Map the global profile scope" in visible
        assert "Schema: compatible (version 3)" in visible
        assert "Server quotas satisfy required minimums" in visible
        assert "Purge generation: matches (0)" in visible
        assert "1 exact proposal" in visible
        assert "profile-local → profile-server" in visible
        assert "scope-local-global → scope-server-global" in visible
        assert "record-exact · exact · local local-v1 · server server-v1" in visible
        assert "proposal-diverged · divergence attention" in visible
        assert "max_record_bytes · required 16384 · server 32768 · satisfied" in visible
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


@pytest.mark.asyncio
async def test_modal_shows_reviewed_new_scope_and_disables_collision_mapping() -> None:
    app = _Host()
    async with app.run_test(size=(110, 48)) as pilot:
        modal = PersonalContextLinkModal(
            _plan(collisions=0, unlinked=1, workspace=True)
        )
        await app.push_screen(modal)
        await pilot.pause()

        visible = " ".join(str(widget.renderable) for widget in modal.query(Static))
        assert (
            "scope-local-workspace → new scope scope-workspace-reviewed" in visible
        )
        assert (
            "scope-local-workspace → scope-0 unavailable: "
            "record-local-workspace conflicts with record-server-workspace"
        ) in visible
        assert modal.query_one(
            "#personal-context-link-workspace-0-map-0", Button
        ).disabled
        assert modal.query_one("#personal-context-link-approve", Button).disabled

        await pilot.click("#personal-context-link-workspace-0-new")
        await pilot.pause()
        assert not modal.query_one(
            "#personal-context-link-approve", Button
        ).disabled
