"""Console left-rail Model section tests."""

from __future__ import annotations

import pytest
from textual.widgets import Static

from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
)
from Tests.UI.app_factory import _build_test_app


@pytest.mark.asyncio
async def test_model_section_renders_the_parameters_only_it_shows() -> None:
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
async def test_model_sync_updates_rows_with_the_actual_values() -> None:
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
        )
        console._apply_console_settings_summary_state(state)
        await pilot.pause(0.2)

        temperature = console.query_one(
            "#console-model-section-temperature .console-model-section-value", Static
        )
        max_tokens = console.query_one(
            "#console-model-section-max-tokens .console-model-section-value", Static
        )
        assert str(temperature.renderable).strip() == "0.42"
        assert str(max_tokens.renderable).strip() == "4096"


def test_the_updater_reads_structured_values_and_no_deleted_ids() -> None:
    """Gate-free pin for TASK-32811.7's two invariants.

    The harness tests above exercise the rendered rows but need a mounted
    app, which this worktree's storage-admission gate blocks; this reads the
    updater's source so the two regressions cannot come back silently:

    * it must not query the Provider/Model section ids TASK-23196 deleted
      (querying them raised NoMatches and skipped the real writes), and
    * it must not regex-parse temperature/max_tokens out of `sampling_row`
      (the structured fields on the state are the source of truth).
    """
    import ast
    from pathlib import Path

    source = (
        Path(__file__).resolve().parents[2]
        / "tldw_chatbook/UI/Screens/chat_screen.py"
    ).read_text()
    tree = ast.parse(source)
    body = None
    for node in ast.walk(tree):
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "_apply_console_settings_summary_state"
        ):
            body = ast.get_source_segment(source, node) or ""
            break
    assert body, "updater not found"

    assert "console-model-section-provider" not in body
    assert "console-model-section-model " not in body
    assert "summary_state.temperature" in body
    assert "summary_state.max_tokens" in body
    assert 'search(r"T ' not in body and "search(r'T " not in body
