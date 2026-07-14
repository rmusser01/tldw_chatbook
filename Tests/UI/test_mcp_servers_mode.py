# Tests/UI/test_mcp_servers_mode.py
from __future__ import annotations

import pytest
from textual.app import App, ComposeResult
from textual.widgets import DataTable, Static

from tldw_chatbook.MCP.readiness import (
    ReadinessSnapshot,
    ReadinessState,
    ReasonCode,
    builtin_readiness,
)
from tldw_chatbook.UI.MCP_Modules.mcp_servers_mode import MCPServersMode


def _snap(key: str, label: str, state=ReadinessState.READY, reasons=(), message="", **kw):
    return ReadinessSnapshot(
        server_key=key, label=label, source=key.split(":", 1)[0],
        state=state, reasons=reasons, message=message, **kw,
    )


class CanvasApp(App):
    def __init__(self) -> None:
        super().__init__()
        self.events: list[object] = []

    def compose(self) -> ComposeResult:
        yield MCPServersMode(id="mcp-mode-canvas-servers")

    def on_mcp_servers_mode_server_row_selected(self, event) -> None:
        self.events.append(event)


@pytest.mark.asyncio
async def test_overview_renders_aggregate_table_and_callouts():
    app = CanvasApp()
    async with app.run_test() as pilot:
        canvas = app.query_one(MCPServersMode)
        await canvas.update_overview(
            [
                _snap("local:docs", "docs", tool_count=4),
                _snap(
                    "local:web", "web",
                    state=ReadinessState.NEEDS_SETUP,
                    reasons=(ReasonCode.AUTH_MISSING,),
                    message="Missing environment variables: KEY.",
                ),
            ]
        )
        await pilot.pause()
        summary = app.query_one("#mcp-overview-summary", Static)
        assert "1 of 2" in str(summary.renderable)
        table = app.query_one("#mcp-servers-table", DataTable)
        assert table.row_count == 2
        callouts = list(app.query(".ds-recovery-callout"))
        assert len(callouts) == 1  # one problem row -> one callout


@pytest.mark.asyncio
async def test_table_row_selection_posts_server_key():
    app = CanvasApp()
    async with app.run_test() as pilot:
        canvas = app.query_one(MCPServersMode)
        await canvas.update_overview([_snap("local:docs", "docs")])
        await pilot.pause()
        table = app.query_one("#mcp-servers-table", DataTable)
        table.focus()
        await pilot.press("enter")
        await pilot.pause()
        assert app.events and app.events[-1].server_key == "local:docs"


@pytest.mark.asyncio
async def test_update_overview_survives_markup_like_labels_and_renders_plain():
    """I1 regression: DataTable cells go through `Text.from_markup` for
    plain-`str` values. A profile id like "[/bold]docs" (an unmatched
    closing tag) raises `rich.errors.MarkupError` -- crashing the app -- and
    "[red]x[/red]" would inject real styling. Label/auth_display/
    scope_display are user-controlled (local profile ids, server-reported
    names) and must render as literal, unstyled text.
    """
    app = CanvasApp()
    async with app.run_test() as pilot:
        canvas = app.query_one(MCPServersMode)
        await canvas.update_overview(
            [
                _snap("local:evil1", "[/bold]docs"),
                _snap(
                    "local:evil2", "safe-label",
                    auth_display="[red]x[/red]",
                    scope_display="[red]y[/red]",
                ),
            ]
        )
        await pilot.pause()
        table = app.query_one("#mcp-servers-table", DataTable)
        assert table.row_count == 2
        row0 = table.get_row_at(0)
        row1 = table.get_row_at(1)
        # Literal text, not interpreted as markup (no crash, no styling).
        assert str(row0[0]) == "[/bold]docs"
        assert str(row1[4]) == "[red]x[/red]"  # Auth column
        assert str(row1[5]) == "[red]y[/red]"  # Scope column


@pytest.mark.asyncio
async def test_row_selection_for_deduped_row_posts_canonical_server_key():
    """F3 regression: `update_overview` de-dupes a colliding row key with a
    `#N` suffix (see `test_update_overview_dedupes_colliding_row_keys_...`
    below), but that suffixed key is a table-internal identifier, not a real
    `server_key` -- posting it upstream leaves callers unable to resolve the
    selection back to a snapshot. Selecting the second (suffixed) row must
    still post the original, canonical `server_key`.
    """
    app = CanvasApp()
    async with app.run_test() as pilot:
        canvas = app.query_one(MCPServersMode)
        await canvas.update_overview(
            [
                _snap("local:unknown", "unknown-1"),
                _snap("local:unknown", "unknown-2"),
            ]
        )
        await pilot.pause()
        table = app.query_one("#mcp-servers-table", DataTable)
        assert table.row_count == 2
        table.focus()
        table.move_cursor(row=1)
        await pilot.press("enter")
        await pilot.pause()
        assert app.events and app.events[-1].server_key == "local:unknown"


@pytest.mark.asyncio
async def test_second_update_overview_leaves_only_latest_callouts():
    """F4 regression: `update_overview` must rebuild the callouts container
    the same awaited way `MCPInspector.update_readiness()` rebuilds its
    action buttons (remove_children awaited, then a single batched mount) --
    see the P0 fix in mcp_inspector.py. Driving two updates back to back
    with NO intervening `pilot.pause()` is the only way to prove the
    remove+mount cycle is fully serialized within one awaited call rather
    than merely "usually fast enough in practice": if the first call's
    callout removal is not awaited, a second call queued right behind it can
    interleave, leaving stale callouts from the first (superseded) snapshot
    list mounted alongside the new ones.
    """
    app = CanvasApp()
    async with app.run_test() as pilot:
        canvas = app.query_one(MCPServersMode)
        await canvas.update_overview(
            [
                _snap("local:a", "a", state=ReadinessState.NEEDS_SETUP,
                      reasons=(ReasonCode.AUTH_MISSING,), message="first-a"),
                _snap("local:b", "b", state=ReadinessState.NEEDS_SETUP,
                      reasons=(ReasonCode.AUTH_MISSING,), message="first-b"),
            ]
        )
        # No pilot.pause() here: the second call must start (and its own
        # remove+mount cycle must fully resolve) before this coroutine
        # returns.
        await canvas.update_overview(
            [
                _snap("local:c", "c", state=ReadinessState.NEEDS_SETUP,
                      reasons=(ReasonCode.AUTH_MISSING,), message="second-c"),
            ]
        )
        await pilot.pause()
        callouts = list(app.query(".ds-recovery-callout"))
        texts = [str(c.renderable) for c in callouts]
        assert texts == ["c: second-c"], texts


@pytest.mark.asyncio
async def test_update_overview_dedupes_colliding_row_keys_without_crashing():
    """I1 regression: two malformed records can both fall back to the same
    server_key (e.g. two local profiles missing `profile_id` both become
    "local:unknown" -- see `local_profile_readiness()`). `update_overview`
    must not let `DataTable.add_row(key=...)` raise `DuplicateKey`.
    """
    app = CanvasApp()
    async with app.run_test() as pilot:
        canvas = app.query_one(MCPServersMode)
        await canvas.update_overview(
            [
                _snap("local:unknown", "unknown"),
                _snap("local:unknown", "unknown"),
            ]
        )
        await pilot.pause()
        table = app.query_one("#mcp-servers-table", DataTable)
        assert table.row_count == 2


@pytest.mark.asyncio
async def test_detail_text_redacts_secret_query_params_in_base_url():
    """I1 regression: the detail pane's Base URL line must not leak a
    secret-looking query parameter value (e.g. `?api_key=...`)."""
    app = CanvasApp()
    async with app.run_test() as pilot:
        canvas = app.query_one(MCPServersMode)
        snap = _snap(
            "server:main", "main",
            detail={"base_url": "https://example.test/api?api_key=sk-super-secret&region=us"},
        )
        canvas.show_detail(snap)
        await pilot.pause()
        body = str(app.query_one("#mcp-detail-body", Static).renderable)
        assert "sk-super-secret" not in body
        assert "region=us" in body


@pytest.mark.asyncio
async def test_detail_text_marks_env_placeholder_missing_despite_whitespace():
    """F5 regression: the "(missing)" marker must use the same placeholder
    canonicalization as `env_placeholder_names()` (used to compute
    `missing_env` in the first place). A raw value with surrounding
    whitespace (e.g. " $MY_KEY ") previously compared
    `str(raw).strip("${}")` -- which only strips '$', '{', '}' characters,
    not whitespace -- against the canonical name in `missing`, so it never
    matched and the marker silently fell back to "set".
    """
    app = CanvasApp()
    async with app.run_test() as pilot:
        canvas = app.query_one(MCPServersMode)
        local = _snap(
            "local:docs", "docs",
            detail={
                "command": "python",
                "args": [],
                "env_placeholders": {"API_KEY": " $MY_KEY "},
                "missing_env": ["MY_KEY"],
                "discovery_snapshot": None,
            },
        )
        canvas.show_detail(local)
        await pilot.pause()
        body = str(app.query_one("#mcp-detail-body", Static).renderable)
        assert "API_KEY (missing)" in body


@pytest.mark.asyncio
async def test_detail_renders_redacted_config_and_builtin_snippet():
    app = CanvasApp()
    async with app.run_test() as pilot:
        canvas = app.query_one(MCPServersMode)
        local = _snap(
            "local:docs", "docs",
            detail={
                "command": "python",
                "args": ["--api-key", "sk-123"],
                "env_placeholders": {"API_KEY": "$MY_KEY"},
                "missing_env": [],
                "discovery_snapshot": {"tools": [{"name": "a"}], "resources": [], "prompts": []},
            },
        )
        canvas.show_detail(local)
        await pilot.pause()
        body = str(app.query_one("#mcp-detail-body", Static).renderable)
        assert "sk-123" not in body
        assert "python" in body

        canvas.show_detail(builtin_readiness(enabled=True))
        await pilot.pause()
        assert list(app.query("#mcp-detail-copy-snippet"))

        canvas.show_detail(None)
        await pilot.pause()
        assert app.query_one("#mcp-servers-overview").display


@pytest.mark.asyncio
async def test_builtin_detail_summarizes_exposed_capabilities_in_human_copy():
    """A3c: the builtin detail pane must read as prose ("Exposes · tools,
    resources") instead of dumping internal config flag names and raw
    booleans ("expose_tools · True").
    """
    app = CanvasApp()
    async with app.run_test() as pilot:
        canvas = app.query_one(MCPServersMode)
        canvas.show_detail(
            builtin_readiness(
                enabled=True, expose_tools=True, expose_resources=True, expose_prompts=False
            )
        )
        await pilot.pause()
        body = str(app.query_one("#mcp-detail-body", Static).renderable)
        assert "Exposes · tools, resources" in body
        assert "expose_tools" not in body
        assert "expose_resources" not in body
        assert "expose_prompts" not in body
        assert "True" not in body
        assert "False" not in body

        canvas.show_detail(
            builtin_readiness(
                enabled=True, expose_tools=False, expose_resources=False, expose_prompts=False
            )
        )
        await pilot.pause()
        body_none = str(app.query_one("#mcp-detail-body", Static).renderable)
        assert "Exposes · nothing" in body_none
