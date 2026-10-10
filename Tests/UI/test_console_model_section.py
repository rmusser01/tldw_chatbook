"""Console left-rail Model section tests."""

from __future__ import annotations

import pytest
from textual.containers import Horizontal
from textual.css.query import DOMQuery
from textual.widgets import Static

from Tests.private_profile import private_profile_test
from Tests.UI.app_factory import _build_test_app
from Tests.UI.console_fixture_ownership import owned_console_apps  # noqa: F401
from Tests.UI.test_console_dictation import _mounted_console
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
)
from tldw_chatbook.Chat.console_session_settings import ConsoleSettingsSummaryState


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(52, 20), (120, 50)])
@private_profile_test
async def test_model_values_update_hidden_rows_without_screen_wide_scans(
    request, monkeypatch, size
) -> None:
    """Model values stay live without evaluating three whole-screen queries."""
    host = ConsoleHarness(_build_test_app())
    async with host.run_test(size=size) as pilot:
        console = await _mounted_console(host, pilot)
        rows = [
            console.query_one(f"#console-model-section-{row_id}")
            for row_id in ("temperature", "max-tokens", "streaming")
        ]
        for row in rows:
            row.display = False
        values = [row.query_one(".console-model-section-value", Static) for row in rows]
        focused = host.focused
        native_nodes = DOMQuery.nodes.fget
        screen_scans = []

        def observe_nodes(query):
            if query.node is console and query._nodes is None:
                screen_scans.append(query)
            return native_nodes(query)

        for temperature, max_tokens, streaming in (
            ("0.42", "4096", "Off"),
            ("0.71", "8192", "On"),
            ("", "", ""),
        ):
            state = ConsoleSettingsSummaryState(
                model_row="Model: test",
                context_row="",
                sampling_row="Not parsed for values",
                identity_row="",
                temperature=temperature,
                max_tokens=max_tokens,
                streaming=streaming,
            )
            screen_scans.clear()
            with monkeypatch.context() as observer:
                observer.setattr(DOMQuery, "nodes", property(observe_nodes))
                console._context_spend._apply_console_settings_summary_state(state)
            assert [str(value.renderable).strip() for value in values] == [
                temperature or "—",
                max_tokens or "—",
                streaming or "—",
            ]
            assert all(not row.display for row in rows)
            assert host.focused is focused
            # The rail writes the values and the readiness word from its own
            # rows, so no lookup walks the whole screen (TASK-33628.5.2).
            assert len(screen_scans) == 0


@pytest.mark.asyncio
@private_profile_test
async def test_model_values_follow_missing_and_remounted_rows(request) -> None:
    """Missing rows skip safely, and native lookups follow replacement widgets."""
    from dataclasses import replace

    host = ConsoleHarness(_build_test_app())
    state = ConsoleSettingsSummaryState(
        model_row="Model: test",
        context_row="",
        sampling_row="Not parsed for values",
        identity_row="",
        temperature="0.42",
        max_tokens="4096",
        streaming="Off",
    )
    changed = replace(state, temperature="0.71", max_tokens="8192", streaming="On")
    sections = (
        ("temperature", "Temperature", "0.71"),
        ("max-tokens", "Max tokens", "8192"),
        ("streaming", "Streaming", "On"),
    )
    async with host.run_test(size=(120, 50)) as pilot:
        console = await _mounted_console(host, pilot)
        for row_id, label, expected in sections:
            row = console.query_one(f"#console-model-section-{row_id}")
            parent = row.parent
            position = parent.children.index(row)
            console._context_spend._apply_console_settings_summary_state(state)

            await row.query_one(".console-model-section-value", Static).remove()
            console._context_spend._apply_console_settings_summary_state(changed)
            for other_id, _, other_value in sections:
                if other_id != row_id:
                    assert (
                        str(
                            console.query_one(f"#console-model-section-{other_id}")
                            .query_one(".console-model-section-value", Static)
                            .renderable
                        ).strip()
                        == other_value
                    )

            replacement_value = Static(
                "stale", classes="console-model-section-value", markup=False
            )
            await row.mount(replacement_value)
            console._context_spend._apply_console_settings_summary_state(changed)
            assert str(replacement_value.renderable).strip() == expected

            await row.remove()
            console._context_spend._apply_console_settings_summary_state(state)
            replacement_value = Static(
                "stale", classes="console-model-section-value", markup=False
            )
            replacement_row = Horizontal(
                Static(label, classes="console-model-section-label", markup=False),
                replacement_value,
                id=f"console-model-section-{row_id}",
                classes="console-model-section-line",
            )
            await parent.mount(replacement_row, before=position)
            console._context_spend._apply_console_settings_summary_state(changed)
            assert (
                console.query_one(f"#console-model-section-{row_id}") is replacement_row
            )
            assert str(replacement_value.renderable).strip() == expected


@pytest.mark.asyncio
@private_profile_test
async def test_model_section_renders_the_parameters_only_it_shows(request) -> None:
    """The Model rail body shows the sampling rows and nothing duplicated.

    TASK-23196: it used to show Provider and Model too, which the persistent
    status bar and the Inspector's run recipe were both already rendering at
    the same moment -- three copies of two values, and this was the copy
    costing scarce rail rows. What remains is what is shown nowhere else.
    """
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 44)) as pilot:
        await pilot.pause(0.2)
        console = host.screen_stack[-1]
        assert console.query_one("#console-model-section-temperature")
        assert console.query_one("#console-model-section-max-tokens")
        assert not console.query("#console-model-section-provider")
        assert not console.query("#console-model-section-model")


@pytest.mark.asyncio
@private_profile_test
async def test_model_sync_updates_rows_with_the_actual_values(request) -> None:
    """A settings change must reach the rendered rows with the real value.

    TASK-32811.7: the updater queried two ids TASK-23196 had deleted
    (Provider, Model) and raised NoMatches on the first, so the temperature
    and max-token writes below never ran and the rows stayed frozen at their
    compose-time values -- while the summary reported a new temperature. And
    the values were regex-parsed out of the formatted `sampling_row` rather
    than read from the structured fields the state carries. This drives a
    specific temperature and max_tokens through the summary state and
    asserts the RENDERED rows show them, not merely that they are non-empty.
    """
    from tldw_chatbook.Chat.console_session_settings import (
        ConsoleSettingsSummaryState,
    )

    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 44)) as pilot:
        await pilot.pause(0.2)
        console = host.screen_stack[-1]

        state = ConsoleSettingsSummaryState(
            model_row="Model: test",
            context_row="",
            sampling_row="T 0.42 · max_tokens 4096",  # deliberately NOT parsed
            identity_row="",
            temperature="0.42",
            max_tokens="4096",
            streaming="Off",
        )
        console._context_spend._apply_console_settings_summary_state(state)
        await pilot.pause(0.2)

        temperature = console.query_one(
            "#console-model-section-temperature .console-model-section-value", Static
        )
        max_tokens = console.query_one(
            "#console-model-section-max-tokens .console-model-section-value", Static
        )
        assert str(temperature.renderable).strip() == "0.42"
        assert str(max_tokens.renderable).strip() == "4096"
        # TASK-33004.7 (task-338): Streaming is the third rendered value.
        streaming = console.query_one(
            "#console-model-section-streaming .console-model-section-value", Static
        )
        assert str(streaming.renderable).strip() == "Off"


def test_the_updater_reads_structured_values_and_no_deleted_ids() -> None:
    """Gate-free pin for TASK-32811.7's two invariants.

    The mounted cases use private interpreter/profile lifetimes. This also
    reads the updater's source to pin the two implementation invariants:

    * it must not query the Provider/Model section ids TASK-23196 deleted
      (querying them raised NoMatches and skipped the real writes), and
    * it must not regex-parse temperature/max_tokens out of `sampling_row`
      (the structured fields on the state are the source of truth).
    """
    import ast
    from pathlib import Path

    # TASK-33628.5.2 moved the updater from ChatScreen into the rail that
    # composes the rows, so it stops walking the whole screen per sync.
    source = (
        Path(__file__).resolve().parents[2]
        / "tldw_chatbook/UI/Console_Modules/left_rail.py"
    ).read_text()
    tree = ast.parse(source)
    body = None
    for node in ast.walk(tree):
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "sync_model_section_values"
        ):
            body = ast.get_source_segment(source, node) or ""
            break
    assert body, "updater not found"

    assert "console-model-section-provider" not in body
    assert "console-model-section-model " not in body
    assert "summary_state.temperature" in body
    assert "summary_state.max_tokens" in body
    assert "summary_state.streaming" in body
    assert 'search(r"T ' not in body and "search(r'T " not in body


@pytest.mark.asyncio
@private_profile_test
async def test_model_section_stays_within_its_15_row_cap(request) -> None:
    """TASK-33004.7 AC#3/AC#4: Streaming joins Temperature and Max tokens, the
    action reads "Change  Alt+M" under its old id, and the section's natural
    height stays within ADR-083's 15 rows even with every recovery row shown
    (each one's own copy and buttons): 15 exactly, so nothing scrolls."""
    from textual.widgets import Button

    from Tests.UI.console_rail_section_helpers import open_rail_section
    from tldw_chatbook.Widgets.Console.console_bounded_section import (
        ConsoleBoundedSection,
    )

    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(211, 44)) as pilot:
        await pilot.pause(0.2)
        console = host.screen_stack[-1]
        await open_rail_section(console, pilot, "model")
        section = console.query_one(
            "#console-bounded-section-model", ConsoleBoundedSection
        )
        rows = [
            str(row.query_one(".console-model-section-label", Static).render())
            for row in console.query(".console-model-section-line")
        ]
        assert rows == ["Temperature", "Max tokens", "Streaming"]
        change = console.query_one("#console-model-section-configure", Button)
        assert str(change.label) == "Change  Alt+M"

        await pilot.pause(0.3)
        normal = section.desired_content_lines
        assert 0 < normal <= 15, normal

        for selector in (
            "#console-model-section-recovery",
            "#console-generation-recovery-row",
            "#console-context-recovery-row",
            "#console-default-recovery-row",
        ):
            console.query_one(selector).styles.display = "block"
        console.query_one("#console-model-section-recovery", Static).update(
            "Not ready · no key"
        )
        console.query_one("#console-default-recovery-copy", Static).update(
            "Not saved: default"
        )
        section.request_reconcile()
        await pilot.pause(0.3)
        assert section.desired_content_lines <= 15, section.desired_content_lines
        assert not section._has_overflow
