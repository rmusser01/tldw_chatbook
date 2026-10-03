"""Switch model rows show each pair's context size (TASK-33004.4 AC#1).

Rewritten on purpose: the quick popover's context and compaction block, and
the serving-window resolver it awaited, are gone (the block lives in Chat
settings). Each row now prints the window the local model catalog knows,
marked ``~`` when it is only a provider or application estimate, and opening
the switcher starts no serving-capacity request.
"""

from __future__ import annotations

import pytest

from Tests.UI.test_console_model_switcher import (
    Recorder,
    SwitcherHarness,
    build_switcher,
    line_with,
    list_lines,
    open_switcher,
)
from tldw_chatbook.Utils.token_counter import resolve_context_window
from tldw_chatbook.Widgets.Console.console_model_popover import context_copy


def testcontext_copy_marks_estimates_and_shortens_sizes() -> None:
    assert context_copy(200_000, True) == "200k"
    assert context_copy(32_000, False) == "~32k"
    assert context_copy(1_048_576, True) == "1M"
    assert context_copy(512, True) == "512"


@pytest.mark.asyncio
async def test_rows_print_the_catalog_window_without_a_serving_request() -> None:
    recorder = Recorder()
    app = SwitcherHarness()
    async with app.run_test(size=(211, 44)) as pilot:
        switcher = await open_switcher(app, pilot, build_switcher(recorder))
        assert not hasattr(switcher, "_context_window_resolver")
        for provider, model in (
            ("anthropic", "claude-haiku-4-5"),
            ("llama_cpp", "model-b"),
        ):
            window = resolve_context_window(provider, model)
            expected = context_copy(window.tokens, window.verified)
            assert f" {expected}  " in line_with(list_lines(app, switcher), model)
