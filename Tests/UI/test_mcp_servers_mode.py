# Tests/UI/test_mcp_servers_mode.py
from __future__ import annotations

from pathlib import Path

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Button, Checkbox, DataTable, Static

import tldw_chatbook

from tldw_chatbook.MCP.readiness import (
    STATE_CSS_CLASSES,
    STATE_GLYPHS,
    HubAction,
    ReadinessSnapshot,
    ReadinessState,
    ReasonCode,
    builtin_readiness,
)
from tldw_chatbook.UI.MCP_Modules.mcp_inspector import MCPInspector
from tldw_chatbook.UI.MCP_Modules.mcp_profile_form import MCPImportPanel
from tldw_chatbook.UI.MCP_Modules.mcp_servers_mode import MCPServersMode

_BUNDLED_CSS_PATH = str(Path(tldw_chatbook.__file__).parent / "css" / "tldw_cli_modular.tcss")


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

    def on_mcp_servers_mode_delete_confirmed(self, event) -> None:
        self.events.append(event)

    def on_mcp_servers_mode_disconnect_requested(self, event) -> None:
        self.events.append(event)

    def on_mcp_inspector_hub_action_requested(self, event) -> None:
        self.events.append(event)

    def on_mcp_servers_mode_add_server_requested(self, event) -> None:
        self.events.append(event)

    def on_mcp_servers_mode_import_servers_requested(self, event) -> None:
        self.events.append(event)

    def on_mcp_servers_mode_builtin_flag_changed(self, event) -> None:
        self.events.append(event)


class CanvasAppWithBundledCSS(CanvasApp):
    """Same harness but with the real bundled stylesheet loaded, so rules
    that only exist in _agentic_terminal.tcss (e.g. `Button.mcp-callout`)
    resolve exactly as they do in the live app -- mirrors
    `RailAppWithBundledCSS` in test_mcp_rail.py."""

    CSS_PATH = _BUNDLED_CSS_PATH


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
        # Task 11: callouts are actionable one-line Buttons now, not inert
        # Statics.
        callouts = list(app.query(".mcp-callout"))
        assert len(callouts) == 1  # one problem row -> one callout
        assert "web" in str(callouts[0].label)
        assert "Missing environment variables" in str(callouts[0].label)


@pytest.mark.asyncio
async def test_overview_summary_glyph_carries_worst_state_class_sentence_stays_neutral():
    """A5: the aggregate summary row is a neutral sentence with a small
    colored glyph in front of it, not the whole line taking on the
    worst-state color (coloring an entire sentence red/orange reads as more
    alarming than the underlying signal warrants). `#mcp-overview-summary-
    glyph` carries the worst-state STATE_CSS_CLASSES class + STATE_GLYPHS
    text; `#mcp-overview-summary` (the sentence) stays a plain
    `ds-status-badge` Static with no status class at all.
    """
    app = CanvasApp()
    async with app.run_test() as pilot:
        canvas = app.query_one(MCPServersMode)
        await canvas.update_overview(
            [
                _snap("local:docs", "docs"),
                _snap(
                    "local:web", "web",
                    state=ReadinessState.NEEDS_ATTENTION,
                    reasons=(ReasonCode.AUTH_MISSING,),
                    message="Timed out",
                ),
            ]
        )
        await pilot.pause()
        glyph = app.query_one("#mcp-overview-summary-glyph", Static)
        summary = app.query_one("#mcp-overview-summary", Static)

        assert STATE_CSS_CLASSES[ReadinessState.NEEDS_ATTENTION] in glyph.classes
        assert str(glyph.renderable) == STATE_GLYPHS[ReadinessState.NEEDS_ATTENTION]

        for css_class in STATE_CSS_CLASSES.values():
            assert css_class not in summary.classes
        assert "ds-status-badge" in summary.classes
        assert "1 of 2" in str(summary.renderable)

        # A second call with a different worst state swaps the glyph's class
        # instead of stacking a second one alongside it.
        await canvas.update_overview(
            [
                _snap(
                    "local:docs", "docs",
                    state=ReadinessState.NEEDS_SETUP,
                    reasons=(ReasonCode.AUTH_MISSING,),
                    message="Missing environment variables: KEY.",
                ),
            ]
        )
        await pilot.pause()
        assert STATE_CSS_CLASSES[ReadinessState.NEEDS_SETUP] in glyph.classes
        assert STATE_CSS_CLASSES[ReadinessState.NEEDS_ATTENTION] not in glyph.classes
        assert str(glyph.renderable) == STATE_GLYPHS[ReadinessState.NEEDS_SETUP]
        for css_class in STATE_CSS_CLASSES.values():
            assert css_class not in summary.classes


@pytest.mark.asyncio
async def test_overview_callouts_are_left_aligned_with_bundled_css():
    """Button defaults BOTH `text-align` and `content-align` to center (see
    Textual's own Button.DEFAULT_CSS -- the exact lesson documented on
    `Button.mcp-rail-row` in MCPRail.DEFAULT_CSS and covered by
    test_mcp_rail.py's sibling test). `Button.mcp-callout` in
    _agentic_terminal.tcss must override both, or the one-line callout
    strips under the overview table render centered instead of flush-left.
    """
    app = CanvasAppWithBundledCSS()
    async with app.run_test(size=(80, 30)) as pilot:
        canvas = app.query_one(MCPServersMode)
        await canvas.update_overview(
            [
                _snap(
                    "local:web", "web",
                    state=ReadinessState.NEEDS_SETUP,
                    reasons=(ReasonCode.AUTH_MISSING,),
                    message="Missing environment variables: KEY.",
                ),
            ]
        )
        await pilot.pause()
        callout = app.query_one("#mcp-callout-0", Button)
        assert callout.styles.text_align == "left"
        assert callout.styles.content_align_horizontal == "left"


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

    Task 11: `source="server"` here (rather than the default "local") so
    the Scope column is still rendered -- Local overviews omit it (see
    `test_local_source_overview_omits_scope_column_and_uses_env_var_copy`),
    but this test's whole point is exercising markup-escaping across every
    column, Scope included.
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
            ],
            source="server",
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
        callouts = list(app.query(".mcp-callout"))
        texts = [str(c.label) for c in callouts]
        assert len(texts) == 1
        assert "second-c" in texts[0], texts


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
        await canvas.show_detail(snap)
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
        await canvas.show_detail(local)
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
        await canvas.show_detail(local)
        await pilot.pause()
        body = str(app.query_one("#mcp-detail-body", Static).renderable)
        assert "sk-123" not in body
        assert "python" in body

        await canvas.show_detail(builtin_readiness(enabled=True))
        await pilot.pause()
        assert list(app.query("#mcp-detail-copy-snippet"))

        await canvas.show_detail(None)
        await pilot.pause()
        assert app.query_one("#mcp-servers-overview").display


@pytest.mark.asyncio
async def test_builtin_detail_no_longer_dumps_raw_expose_flags_in_body_text():
    """A3c (carried forward from Task 6): the builtin detail body must not
    dump internal config flag names/raw booleans. Task 10 replaces the old
    "Exposes · tools, resources" prose line with the four checkboxes
    asserted below -- this only guards the body Static itself stays clean.
    """
    app = CanvasApp()
    async with app.run_test() as pilot:
        canvas = app.query_one(MCPServersMode)
        await canvas.show_detail(
            builtin_readiness(
                enabled=True, expose_tools=True, expose_resources=True, expose_prompts=False
            )
        )
        await pilot.pause()
        body = str(app.query_one("#mcp-detail-body", Static).renderable)
        assert "expose_tools" not in body
        assert "expose_resources" not in body
        assert "expose_prompts" not in body
        assert "True" not in body
        assert "False" not in body


@pytest.mark.asyncio
async def test_builtin_detail_shows_enable_expose_checkboxes_with_values_and_note():
    """Task 10 Step 1: the builtin detail view renders four Checkboxes
    (enabled/expose_tools/expose_resources/expose_prompts), each seeded from
    the snapshot's detail flags, plus the next-launch note. Every checkbox
    must carry a tooltip.
    """
    app = CanvasApp()
    async with app.run_test() as pilot:
        canvas = app.query_one(MCPServersMode)
        await canvas.show_detail(
            builtin_readiness(
                enabled=True, expose_tools=True, expose_resources=False, expose_prompts=True
            )
        )
        await pilot.pause()

        enabled_cb = app.query_one("#mcp-builtin-enabled", Checkbox)
        tools_cb = app.query_one("#mcp-builtin-expose-tools", Checkbox)
        resources_cb = app.query_one("#mcp-builtin-expose-resources", Checkbox)
        prompts_cb = app.query_one("#mcp-builtin-expose-prompts", Checkbox)
        assert enabled_cb.value is True
        assert tools_cb.value is True
        assert resources_cb.value is False
        assert prompts_cb.value is True
        for checkbox in (enabled_cb, tools_cb, resources_cb, prompts_cb):
            assert checkbox.tooltip, f"{checkbox.id} missing a tooltip"

        note = str(app.query_one("#mcp-builtin-toggles-note", Static).renderable)
        assert (
            "Applies to the next client launch — the built-in server reads config at start."
            in note
        )

        # Copy-snippet button survives the conversion to widget-composed rows.
        assert list(app.query("#mcp-detail-copy-snippet"))


@pytest.mark.asyncio
async def test_builtin_toggles_container_does_not_expand_past_content():
    """QA defect 1 (mcp-hub-phase2-2026-07 round, P3 cosmetic): `Vertical(id=
    "mcp-detail-builtin-toggles")` had no height override, so it silently
    inherited Textual's Vertical default (`height: 1fr`) and expanded to
    fill the rest of `#mcp-detail-scroll` -- pushing the sibling "Copy
    client config" Button roughly 800px below the four checkboxes it should
    render directly under. Guards both halves of the fix: the container is
    sized to its content (not the pane remainder), and the copy button
    renders immediately below it rather than far down the scroll region.
    """
    app = CanvasApp()
    async with app.run_test() as pilot:
        canvas = app.query_one(MCPServersMode)
        await canvas.show_detail(
            builtin_readiness(
                enabled=True, expose_tools=True, expose_resources=True, expose_prompts=True
            )
        )
        await pilot.pause()

        toggles = app.query_one("#mcp-detail-builtin-toggles")
        copy_button = app.query_one("#mcp-detail-copy-snippet", Button)

        # Four checkboxes + the next-launch note is a handful of rows --
        # nowhere near the height of an expanding 1fr container consuming
        # whatever vertical space the scroll pane has left.
        assert toggles.size.height < 12

        # The copy button must sit directly under the toggles container,
        # not dozens of rows further down the scroll pane.
        gap = copy_button.region.y - (toggles.region.y + toggles.region.height)
        assert 0 <= gap <= 2


@pytest.mark.asyncio
async def test_builtin_checkboxes_do_not_appear_for_non_builtin_detail():
    app = CanvasApp()
    async with app.run_test() as pilot:
        canvas = app.query_one(MCPServersMode)
        await canvas.show_detail(_snap("local:docs", "docs"))
        await pilot.pause()
        assert not list(app.query("#mcp-builtin-enabled"))
        assert not list(app.query("#mcp-builtin-toggles-note"))


@pytest.mark.asyncio
async def test_showing_builtin_detail_does_not_post_builtin_flag_changed():
    """Mount-echo guard: Checkbox's own constructor wraps its initial
    `value` set in `self.prevent(self.Changed)` and declares the reactive
    with `init=False` (see textual.widgets._toggle_button.ToggleButton), so
    composing these four Checkboxes with non-default values must not itself
    fire `Changed` -- and therefore must not post `BuiltinFlagChanged` --
    for any of them.
    """
    app = CanvasApp()
    async with app.run_test() as pilot:
        canvas = app.query_one(MCPServersMode)
        await canvas.show_detail(
            builtin_readiness(
                enabled=False, expose_tools=False, expose_resources=True, expose_prompts=False
            )
        )
        await pilot.pause()
        assert not app.events


@pytest.mark.asyncio
async def test_toggling_builtin_enabled_checkbox_posts_builtin_flag_changed():
    app = CanvasApp()
    async with app.run_test() as pilot:
        canvas = app.query_one(MCPServersMode)
        await canvas.show_detail(builtin_readiness(enabled=True))
        await pilot.pause()
        await pilot.click("#mcp-builtin-enabled")
        await pilot.pause()
        posted = [e for e in app.events if type(e).__name__ == "BuiltinFlagChanged"]
        assert posted, "expected a BuiltinFlagChanged event"
        assert posted[-1].key == "enabled"
        assert posted[-1].value is False


@pytest.mark.asyncio
async def test_toggling_builtin_expose_checkboxes_posts_matching_keys():
    app = CanvasApp()
    async with app.run_test() as pilot:
        canvas = app.query_one(MCPServersMode)
        await canvas.show_detail(
            builtin_readiness(
                enabled=True, expose_tools=True, expose_resources=True, expose_prompts=True
            )
        )
        await pilot.pause()
        for checkbox_id, key in (
            ("#mcp-builtin-expose-tools", "expose_tools"),
            ("#mcp-builtin-expose-resources", "expose_resources"),
            ("#mcp-builtin-expose-prompts", "expose_prompts"),
        ):
            app.events.clear()
            await pilot.click(checkbox_id)
            await pilot.pause()
            posted = [e for e in app.events if type(e).__name__ == "BuiltinFlagChanged"]
            assert posted, f"expected a BuiltinFlagChanged event for {checkbox_id}"
            assert posted[-1].key == key
            assert posted[-1].value is False


# -- T7: detail-view toolbar (Edit/Disconnect/Delete arm-then-confirm) ------


@pytest.mark.asyncio
async def test_delete_requires_arm_then_confirm():
    app = CanvasApp()
    async with app.run_test() as pilot:
        canvas = app.query_one(MCPServersMode)
        snap = _snap("local:docs", "docs",
                     detail={"command": "npx", "args": [], "env_placeholders": {},
                             "missing_env": [], "discovery_snapshot": None})
        await canvas.show_detail(snap)
        await pilot.pause()
        await pilot.click("#mcp-detail-delete")
        await pilot.pause()
        assert list(app.query("#mcp-detail-delete-confirm"))
        assert not app.events  # nothing posted yet
        await pilot.click("#mcp-detail-delete-cancel")
        await pilot.pause()
        assert list(app.query("#mcp-detail-delete"))  # disarmed
        await pilot.click("#mcp-detail-delete")
        await pilot.pause()
        await pilot.click("#mcp-detail-delete-confirm")
        await pilot.pause()
        confirmed = [e for e in app.events if type(e).__name__ == "DeleteConfirmed"]
        assert confirmed and confirmed[-1].server_key == "local:docs"


@pytest.mark.asyncio
async def test_builtin_detail_has_no_delete_toolbar():
    app = CanvasApp()
    async with app.run_test() as pilot:
        canvas = app.query_one(MCPServersMode)
        await canvas.show_detail(builtin_readiness(enabled=True))
        await pilot.pause()
        assert not list(app.query("#mcp-detail-delete"))


@pytest.mark.asyncio
async def test_server_source_detail_has_no_delete_toolbar():
    """Interfaces: "Built-in and server-source detail views do NOT render
    this toolbar" -- server-source is mutated server-side (Advanced), not
    from this detail pane, so it must be checked the same as builtin."""
    app = CanvasApp()
    async with app.run_test() as pilot:
        canvas = app.query_one(MCPServersMode)
        snap = _snap("server:main", "main", detail={"base_url": "https://example.test"})
        await canvas.show_detail(snap)
        await pilot.pause()
        assert not list(app.query("#mcp-detail-edit"))
        assert not list(app.query("#mcp-detail-disconnect"))
        assert not list(app.query("#mcp-detail-delete"))


@pytest.mark.asyncio
async def test_disconnect_button_gated_by_is_connected():
    """`snapshot.is_connected` gates the Disconnect button: shown for a
    connected local profile, absent for a disconnected (or never-checked)
    one -- disconnecting something that isn't running makes no sense."""
    app = CanvasApp()
    async with app.run_test() as pilot:
        canvas = app.query_one(MCPServersMode)
        connected = _snap("local:docs", "docs", is_connected=True)
        await canvas.show_detail(connected)
        await pilot.pause()
        assert list(app.query("#mcp-detail-disconnect"))

        disconnected = _snap("local:docs", "docs", is_connected=False)
        await canvas.show_detail(disconnected)
        await pilot.pause()
        assert not list(app.query("#mcp-detail-disconnect"))


@pytest.mark.asyncio
async def test_disconnect_button_posts_disconnect_requested():
    app = CanvasApp()
    async with app.run_test() as pilot:
        canvas = app.query_one(MCPServersMode)
        await canvas.show_detail(_snap("local:docs", "docs", is_connected=True))
        await pilot.pause()
        await pilot.click("#mcp-detail-disconnect")
        await pilot.pause()
        posted = [e for e in app.events if type(e).__name__ == "DisconnectRequested"]
        assert posted and posted[-1].server_key == "local:docs"


@pytest.mark.asyncio
async def test_edit_button_posts_edit_config_hub_action():
    """Edit reuses the existing EDIT_CONFIG path (Task 6's `show_form(record)`
    via the workbench's `on_mcp_inspector_hub_action_requested` handler)
    instead of duplicating the record lookup."""
    app = CanvasApp()
    async with app.run_test() as pilot:
        canvas = app.query_one(MCPServersMode)
        await canvas.show_detail(_snap("local:docs", "docs"))
        await pilot.pause()
        await pilot.click("#mcp-detail-edit")
        await pilot.pause()
        posted = [e for e in app.events if isinstance(e, MCPInspector.HubActionRequested)]
        assert posted and posted[-1].action is HubAction.EDIT_CONFIG
        assert posted[-1].server_key == "local:docs"


@pytest.mark.asyncio
async def test_every_detail_toolbar_button_has_a_tooltip():
    app = CanvasApp()
    async with app.run_test() as pilot:
        canvas = app.query_one(MCPServersMode)
        await canvas.show_detail(_snap("local:docs", "docs", is_connected=True))
        await pilot.pause()
        for button_id in ("#mcp-detail-edit", "#mcp-detail-disconnect", "#mcp-detail-delete"):
            button = app.query_one(button_id)
            assert button.tooltip, f"{button_id} missing a tooltip"
        await pilot.click("#mcp-detail-delete")
        await pilot.pause()
        for button_id in ("#mcp-detail-delete-confirm", "#mcp-detail-delete-cancel"):
            button = app.query_one(button_id)
            assert button.tooltip, f"{button_id} missing a tooltip"


# -- T8: mcpServers import button + show_import() hosting --------------------


@pytest.mark.asyncio
async def test_import_button_posts_import_servers_requested():
    app = CanvasApp()
    async with app.run_test() as pilot:
        await pilot.click("#mcp-import-server")
        await pilot.pause()
        posted = [e for e in app.events if isinstance(e, MCPServersMode.ImportServersRequested)]
        assert posted


@pytest.mark.asyncio
async def test_import_button_gated_off_under_server_source():
    """I3: Import always writes LOCAL profiles (`MCPWorkbench._apply_import()`
    calls `save_local_profile()` unconditionally), so it must be disabled
    under server source rather than silently writing somewhere invisible in
    the current view -- mirrors `_update_add_server_button()`'s
    disabled+tooltip gating pattern.
    """
    app = CanvasApp()
    async with app.run_test() as pilot:
        canvas = app.query_one(MCPServersMode)
        button = canvas.query_one("#mcp-import-server", Button)
        assert button.disabled is False

        await canvas.update_overview([], source="server")
        await pilot.pause()
        assert button.disabled is True
        assert button.tooltip == (
            "Import creates LOCAL server profiles — switch Source to Local."
        )

        await canvas.update_overview([], source="local")
        await pilot.pause()
        assert button.disabled is False
        assert button.tooltip == (
            "Import servers from a Claude-Desktop-style mcpServers JSON file or paste."
        )


@pytest.mark.asyncio
async def test_show_import_hides_overview_and_mounts_panel():
    app = CanvasApp()
    async with app.run_test() as pilot:
        canvas = app.query_one(MCPServersMode)
        await canvas.show_import({"docs"})
        await pilot.pause()
        assert not app.query_one("#mcp-servers-overview").display
        assert app.query_one("#mcp-servers-form").display
        panel = app.query_one(MCPImportPanel)
        assert panel._existing_ids == {"docs"}


@pytest.mark.asyncio
async def test_hide_form_closes_import_panel_and_restores_overview():
    app = CanvasApp()
    async with app.run_test() as pilot:
        canvas = app.query_one(MCPServersMode)
        await canvas.show_import(set())
        await pilot.pause()
        await canvas.hide_form()
        await pilot.pause()
        assert not list(app.query(MCPImportPanel))
        assert app.query_one("#mcp-servers-overview").display
        assert not app.query_one("#mcp-servers-form").display


# -- Task 11: breadcrumb navigation + selection restoration ------------------


@pytest.mark.asyncio
async def test_breadcrumb_posts_clear_and_returning_restores_table_cursor():
    """Task 11: `#mcp-detail-back` posts `ServerRowSelected(None)` (the same
    "clear the selection" contract the rail's "All servers" row uses), and
    the canvas's own `show_detail(None)` -- which the workbench calls once
    it applies that clear -- must put the DataTable cursor back on the row
    for whichever server was selected before, not reset to row 0.
    """
    app = CanvasApp()
    async with app.run_test() as pilot:
        canvas = app.query_one(MCPServersMode)
        await canvas.update_overview(
            [
                _snap("local:a", "a"),
                _snap("local:b", "b"),
                _snap("local:c", "c"),
            ]
        )
        await pilot.pause()
        # Select the third row (index 2) via the detail view, the same way
        # the workbench does after a table row click.
        await canvas.show_detail(_snap("local:c", "c"))
        await pilot.pause()
        assert not app.query_one("#mcp-servers-overview").display

        back_button = app.query_one("#mcp-detail-back", Button)
        assert back_button.tooltip == "Return to the overview table."
        back_button.press()
        await pilot.pause()

        assert app.events, "breadcrumb press should post ServerRowSelected"
        assert app.events[-1].server_key is None

        # The workbench's `_select_server_key(None)` path re-syncs the
        # overview and then calls `show_detail(None)` -- simulate that here
        # since this test drives the canvas directly, without a workbench.
        await canvas.show_detail(None)
        await pilot.pause()
        assert app.query_one("#mcp-servers-overview").display
        table = app.query_one("#mcp-servers-table", DataTable)
        assert table.cursor_row == 2


@pytest.mark.asyncio
async def test_returning_to_overview_without_a_prior_selection_leaves_cursor_alone():
    """No `_last_selected_key` recorded yet (e.g. the breadcrumb-equivalent
    fires before any row was ever selected) -- `_restore_overview_cursor()`
    must no-op rather than raise or force a move to row 0.
    """
    app = CanvasApp()
    async with app.run_test() as pilot:
        canvas = app.query_one(MCPServersMode)
        await canvas.update_overview([_snap("local:a", "a"), _snap("local:b", "b")])
        await pilot.pause()
        await canvas.show_detail(None)  # no prior show_detail(snapshot) call
        await pilot.pause()
        assert app.query_one("#mcp-servers-overview").display


# -- Task 11: actionable callouts ---------------------------------------------


@pytest.mark.asyncio
async def test_callout_click_posts_server_row_selected_with_its_key():
    app = CanvasApp()
    async with app.run_test() as pilot:
        canvas = app.query_one(MCPServersMode)
        await canvas.update_overview(
            [
                _snap("local:docs", "docs"),
                _snap(
                    "local:web", "web",
                    state=ReadinessState.NEEDS_SETUP,
                    reasons=(ReasonCode.AUTH_MISSING,),
                    message="Missing environment variables: KEY.",
                ),
            ]
        )
        await pilot.pause()
        callout = app.query_one("#mcp-callout-0", Button)
        assert callout.tooltip == "Open web."
        callout.press()
        await pilot.pause()
        assert app.events and app.events[-1].server_key == "local:web"


@pytest.mark.asyncio
async def test_callouts_cap_at_four_with_overflow_static():
    app = CanvasApp()
    async with app.run_test() as pilot:
        canvas = app.query_one(MCPServersMode)
        problem_snaps = [
            _snap(
                f"local:p{i}", f"p{i}",
                state=ReadinessState.NEEDS_SETUP,
                reasons=(ReasonCode.AUTH_MISSING,),
                message=f"problem {i}",
            )
            for i in range(6)
        ]
        await canvas.update_overview(problem_snaps)
        await pilot.pause()
        callouts = list(app.query(".mcp-callout"))
        assert len(callouts) == 4
        overflow = list(app.query("#mcp-overview-callouts .ds-recovery-callout"))
        assert len(overflow) == 1
        assert "+2 more" in str(overflow[0].renderable)
        assert "see the table above" in str(overflow[0].renderable)


# -- Task 11: per-source columns ----------------------------------------------


@pytest.mark.asyncio
async def test_local_source_overview_omits_scope_column_and_shows_env_var_copy():
    """Local source (built-in + local profiles) has no meaningful Scope --
    the overview table omits the column there rather than rendering a
    column of dashes. The Auth column copy for a local profile with one env
    placeholder is "1 env var" (see readiness.py's `local_profile_readiness`
    / `_env_auth_display`, which is what actually derives this for real
    profiles) -- pinned here at the table-rendering layer.
    """
    app = CanvasApp()
    async with app.run_test() as pilot:
        canvas = app.query_one(MCPServersMode)
        await canvas.update_overview(
            [_snap("local:docs", "docs", auth_display="1 env var")],
            source="local",
        )
        await pilot.pause()
        table = app.query_one("#mcp-servers-table", DataTable)
        columns = [str(col.label) for col in table.ordered_columns]
        assert columns == ["Name", "Transport", "Status", "Tools", "Auth"]
        row0 = table.get_row_at(0)
        assert len(row0) == 5
        assert str(row0[4]) == "1 env var"


@pytest.mark.asyncio
async def test_server_source_overview_keeps_scope_column():
    app = CanvasApp()
    async with app.run_test() as pilot:
        canvas = app.query_one(MCPServersMode)
        await canvas.update_overview(
            [_snap("server:main", "main", scope_display="Team")],
            source="server",
        )
        await pilot.pause()
        table = app.query_one("#mcp-servers-table", DataTable)
        columns = [str(col.label) for col in table.ordered_columns]
        assert columns == ["Name", "Transport", "Status", "Tools", "Auth", "Scope"]
        row0 = table.get_row_at(0)
        assert len(row0) == 6
        assert str(row0[5]) == "Team"


@pytest.mark.asyncio
async def test_switching_source_between_calls_rebuilds_columns():
    """Columns are rebuilt from scratch on every `update_overview()` call --
    a source switch (server -> local, the rail's actual round trip) must
    not leave a stale Scope column from the previous call's source behind.
    """
    app = CanvasApp()
    async with app.run_test() as pilot:
        canvas = app.query_one(MCPServersMode)
        await canvas.update_overview([_snap("server:main", "main")], source="server")
        await pilot.pause()
        table = app.query_one("#mcp-servers-table", DataTable)
        assert len(table.ordered_columns) == 6

        await canvas.update_overview([_snap("local:docs", "docs")], source="local")
        await pilot.pause()
        assert len(table.ordered_columns) == 5
