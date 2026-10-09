"""Switch model rows show each pair's context size (TASK-33004.4 AC#1).

Rewritten on purpose: the quick popover's context and compaction block, and
the serving-window resolver it awaited, are gone (the block lives in Chat
settings). Each row now prints the window the local model catalog knows,
and opening the switcher starts no serving-capacity request. A provider or
application fallback is a guess, so its row reads ``?`` (unknown), never a
size (TASK-33007 #12; it used to read ``~32k``).
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


def test_context_copy_calls_a_fallback_unknown_and_shortens_sizes() -> None:
    assert context_copy(200_000, True) == "200k"
    assert context_copy(32_000, False) == "?"
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


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(211, 44), (235, 52)])
async def test_a_fallback_window_reads_unknown_and_a_known_one_keeps_its_size(
    size: tuple[int, int],
) -> None:
    """TASK-33007 #12: the dense column never shows an assumed size as known."""
    assert resolve_context_window("openai", "gpt-4o").verified
    assert not resolve_context_window("openai", "gpt-5.6-terra").verified
    app = SwitcherHarness()
    async with app.run_test(size=size) as pilot:
        switcher = await open_switcher(
            app,
            pilot,
            build_switcher(
                Recorder(ready=frozenset({"openai"})),
                providers_models={"OpenAI": ["gpt-4o", "gpt-5.6-terra"]},
            ),
        )
        lines = list_lines(app, switcher)
        assert " 128k  " in line_with(lines, "gpt-4o")
        terra = line_with(lines, "gpt-5.6-terra")
        assert "    ?  " in terra
        assert "~" not in terra and "32k" not in terra
