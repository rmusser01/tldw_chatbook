"""Tests for tool-call diff rendering with textual-diff-view (TASK-1351)."""

import asyncio
import json
import time
from collections.abc import Callable

import pytest
from textual.app import App
from textual_diff_view import DiffView

from tldw_chatbook.Widgets import tool_message_widgets
from tldw_chatbook.Widgets.diff_widgets import (
    extract_diff_from_result,
    make_diff,
    set_default_diff_view_mode,
    strip_diff_contents,
)
from tldw_chatbook.Widgets.tool_message_widgets import (
    ToolExecutionWidget,
    ToolResultMessage,
)


async def wait_for_condition(
    predicate: Callable[[], bool], timeout: float = 5.0, interval: float = 0.02
) -> bool:
    """Poll ``predicate`` until it is true or ``timeout`` seconds elapse.

    Used instead of a fixed number of ``pilot.pause()`` calls, which is
    timing-dependent and flaky.

    Args:
        predicate: Zero-argument callable returning True when the awaited
            condition holds.
        timeout: Maximum seconds to poll before giving up.
        interval: Seconds between polls.

    Returns:
        True if the predicate became true before the deadline, False otherwise.
    """
    deadline = time.monotonic() + timeout
    while True:
        if predicate():
            return True
        if time.monotonic() >= deadline:
            return False
        await asyncio.sleep(interval)


class DiffTestApp(App):
    """Test app for mounting diff-related widgets."""

    def __init__(self, widget):
        super().__init__()
        self.test_widget = widget

    def compose(self):
        yield self.test_widget


@pytest.fixture(autouse=True)
def reset_diff_view_mode():
    """Restore the module-level default view mode after each test."""
    yield
    set_default_diff_view_mode("auto")


@pytest.fixture
def diff_tool_result():
    """A tool result carrying before/after file contents."""
    return [
        {
            "tool_call_id": "call_diff",
            "result": {
                "file_path": "/tmp/example.py",
                "action": "overwritten",
                "size_bytes": 20,
                "encoding": "utf-8",
                "lines_written": 2,
                "old_content": "def f():\n    return 1\n",
                "new_content": "def f():\n    return 2\n",
            },
        }
    ]


class TestExtractDiffFromResult:
    """extract_diff_from_result finds before/after content in tool results."""

    def test_extracts_diff_fields(self, diff_tool_result):
        extracted = extract_diff_from_result(diff_tool_result[0])
        assert extracted == (
            "/tmp/example.py",
            "def f():\n    return 1\n",
            "def f():\n    return 2\n",
        )

    def test_plain_result_returns_none(self):
        result = {"tool_call_id": "call_1", "result": {"answer": 42}}
        assert extract_diff_from_result(result) is None

    def test_error_result_returns_none(self):
        result = {"tool_call_id": "call_2", "error": "boom"}
        assert extract_diff_from_result(result) is None

    def test_missing_old_content_returns_none(self):
        result = {
            "tool_call_id": "call_3",
            "result": {"file_path": "/tmp/x", "new_content": "data"},
        }
        assert extract_diff_from_result(result) is None

    def test_non_dict_payload_returns_none(self):
        result = {"tool_call_id": "call_4", "result": "just a string"}
        assert extract_diff_from_result(result) is None


class TestStripDiffContents:
    """strip_diff_contents removes raw contents from outbound/stored payloads."""

    def test_strips_keys_from_copy(self, diff_tool_result):
        stripped = strip_diff_contents(diff_tool_result[0])

        assert stripped is not diff_tool_result[0]
        assert stripped["tool_call_id"] == "call_diff"
        assert "old_content" not in stripped["result"]
        assert "new_content" not in stripped["result"]
        # Other fields are preserved.
        assert stripped["result"]["file_path"] == "/tmp/example.py"
        assert stripped["result"]["action"] == "overwritten"

    def test_non_mutating(self, diff_tool_result):
        """The in-memory record keeps its contents for live UI rendering."""
        strip_diff_contents(diff_tool_result[0])

        assert diff_tool_result[0]["result"]["old_content"] == "def f():\n    return 1\n"
        assert diff_tool_result[0]["result"]["new_content"] == "def f():\n    return 2\n"

    def test_result_without_diff_keys_returned_as_is(self):
        result = {"tool_call_id": "call_1", "result": {"answer": 42}}
        assert strip_diff_contents(result) is result

    def test_error_result_returned_as_is(self):
        result = {"tool_call_id": "call_2", "error": "boom"}
        assert strip_diff_contents(result) is result

    def test_serialized_payload_drops_contents(self, diff_tool_result):
        """End-to-end: the stored/outbound JSON carries no raw contents."""
        payload = json.dumps([strip_diff_contents(r) for r in diff_tool_result])
        assert "old_content" not in payload
        assert "new_content" not in payload
        assert "return 1" not in payload
        assert "return 2" not in payload
        assert "overwritten" in payload


class TestMakeDiff:
    """make_diff builds a DiffView from a tool-call record's contents."""

    def test_paths_and_content(self):
        diff_view = make_diff("/tmp/a.py", "old\n", "new\n")

        assert isinstance(diff_view, DiffView)
        assert diff_view.path_original == "/tmp/a.py"
        assert diff_view.path_modified == "/tmp/a.py"
        assert diff_view.code_original == "old\n"
        assert diff_view.code_modified == "new\n"

    def test_none_content_becomes_empty(self):
        diff_view = make_diff("/tmp/a.py", None, "new\n")
        assert diff_view.code_original == ""

    def test_default_mode_is_auto(self):
        diff_view = make_diff("/tmp/a.py", "old\n", "new\n")
        assert diff_view.auto_split is True

    def test_mode_switching(self):
        set_default_diff_view_mode("split")
        assert make_diff("p", "a", "b").split is True
        assert make_diff("p", "a", "b").auto_split is False

        set_default_diff_view_mode("unified")
        assert make_diff("p", "a", "b").split is False
        assert make_diff("p", "a", "b").auto_split is False

        set_default_diff_view_mode("auto")
        assert make_diff("p", "a", "b").auto_split is True

    def test_invalid_mode_rejected(self):
        with pytest.raises(ValueError):
            set_default_diff_view_mode("sideways")


class TestAutoSplit:
    """Auto view mode flips between unified and split based on width."""

    @pytest.mark.asyncio
    async def test_wide_terminal_enables_split(self):
        diff_view = make_diff("/tmp/a.py", "a\nb\n", "a\nc\n")
        app = DiffTestApp(diff_view)

        async with app.run_test(size=(200, 50)) as pilot:
            await pilot.pause()
            assert diff_view.auto_split is True
            assert diff_view.split is True

    @pytest.mark.asyncio
    async def test_resize_flips_split_mode(self):
        """Resizing the terminal flips split off and back on (AC2)."""
        long_line = "x" * 60
        diff_view = make_diff("/tmp/a.py", f"{long_line}\nb\n", f"{long_line}\nc\n")
        app = DiffTestApp(diff_view)

        async with app.run_test(size=(200, 50)) as pilot:
            await pilot.pause()
            assert diff_view.split is True

            await pilot.resize_terminal(80, 24)
            await pilot.pause()
            assert diff_view.split is False

            await pilot.resize_terminal(200, 50)
            await pilot.pause()
            assert diff_view.split is True


class TestToolExecutionWidgetDiffs:
    """ToolExecutionWidget mounts DiffViews for results with diff content."""

    @pytest.mark.asyncio
    async def test_diff_mounted_after_prepare(self, monkeypatch, diff_tool_result):
        """Diff is prepared off the UI thread before being mounted (AC3)."""
        prepared = []
        real_make_diff = tool_message_widgets.make_diff

        def tracking_make_diff(path, old, new, **kwargs):
            diff_view = real_make_diff(path, old, new, **kwargs)
            original_prepare = diff_view.prepare

            async def tracked_prepare():
                prepared.append(path)
                await original_prepare()

            diff_view.prepare = tracked_prepare
            return diff_view

        monkeypatch.setattr(tool_message_widgets, "make_diff", tracking_make_diff)

        tool_calls = [
            {
                "function": {
                    "name": "write_file",
                    "arguments": json.dumps(
                        {"file_path": "/tmp/example.py", "content": "x"}
                    ),
                }
            }
        ]
        widget = ToolExecutionWidget(tool_calls=tool_calls)
        app = DiffTestApp(widget)

        async with app.run_test() as pilot:
            widget.update_results(diff_tool_result)

            found = await wait_for_condition(
                lambda: len(widget.query(DiffView)) == 1
            )
            assert found, "DiffView was not mounted within the timeout"
            await pilot.pause()

            diff_view = widget.query(DiffView).first()
            # prepare() ran before the widget was mounted
            assert prepared == ["/tmp/example.py"]
            assert diff_view.is_mounted
            assert diff_view.path_modified == "/tmp/example.py"
            assert diff_view.code_original == "def f():\n    return 1\n"
            assert diff_view.code_modified == "def f():\n    return 2\n"
            # Text result widget still renders alongside the diff
            assert widget.tool_result_widget is not None

    @pytest.mark.asyncio
    async def test_results_without_diff_content_render_as_before(self):
        """Plain tool results produce text only, no DiffView (AC, no regression)."""
        tool_calls = [
            {
                "function": {
                    "name": "calculator",
                    "arguments": json.dumps({"expression": "1+1"}),
                }
            }
        ]
        widget = ToolExecutionWidget(tool_calls=tool_calls)
        app = DiffTestApp(widget)

        async with app.run_test():
            widget.update_results(
                [{"tool_call_id": "call_1", "result": {"answer": 2}}]
            )

            mounted = await wait_for_condition(
                lambda: widget.tool_result_widget is not None
                and widget.tool_result_widget.is_mounted
            )
            assert mounted, "ToolResultMessage was not mounted within the timeout"
            # Bounded negative wait: no DiffView should ever appear.
            appeared = await wait_for_condition(
                lambda: len(widget.query(DiffView)) > 0, timeout=0.5
            )
            assert not appeared
            assert isinstance(widget.tool_result_widget, ToolResultMessage)
            assert "answer: 2" in widget.tool_result_widget.message_text

    @pytest.mark.asyncio
    async def test_results_passed_at_init_mount_diff_on_mount(self, diff_tool_result):
        """Results supplied to the constructor also get a DiffView."""
        tool_calls = [
            {
                "function": {
                    "name": "write_file",
                    "arguments": json.dumps({"file_path": "/tmp/example.py"}),
                }
            }
        ]
        widget = ToolExecutionWidget(
            tool_calls=tool_calls, tool_results=diff_tool_result
        )
        app = DiffTestApp(widget)

        async with app.run_test():
            found = await wait_for_condition(
                lambda: len(widget.query(DiffView)) == 1
            )
            assert found, "DiffView was not mounted within the timeout"

    @pytest.mark.asyncio
    async def test_repeated_update_results_does_not_double_render(
        self, diff_tool_result
    ):
        """A second update_results call replaces the diff, not duplicates it."""
        tool_calls = [
            {
                "function": {
                    "name": "write_file",
                    "arguments": json.dumps({"file_path": "/tmp/example.py"}),
                }
            }
        ]
        widget = ToolExecutionWidget(tool_calls=tool_calls)
        app = DiffTestApp(widget)

        async with app.run_test():
            widget.update_results(diff_tool_result)
            assert await wait_for_condition(
                lambda: len(widget.query(DiffView)) == 1
            )

            updated_result = [
                {
                    **diff_tool_result[0],
                    "result": {
                        **diff_tool_result[0]["result"],
                        "new_content": "def f():\n    return 3\n",
                    },
                }
            ]
            widget.update_results(updated_result)

            # Exactly one DiffView, showing the latest content (the old diff
            # may transiently coexist while its removal is processed, so wait
            # for the settled state).
            settled = await wait_for_condition(
                lambda: len(widget.query(DiffView)) == 1
                and widget.query(DiffView).first().code_modified
                == "def f():\n    return 3\n"
            )
            assert settled, "DiffView was not replaced within the timeout"

    def test_raw_contents_skipped_in_text_formatting(self, diff_tool_result):
        """old_content/new_content are not dumped into the text rendering."""
        result_widget = ToolResultMessage(tool_results=diff_tool_result)
        assert "return 1" not in result_widget.message_text
        assert "return 2" not in result_widget.message_text
        assert "overwritten" in result_widget.message_text
