"""Tests for tool-call diff rendering with textual-diff-view (TASK-1351)."""

import json

import pytest
from textual.app import App
from textual_diff_view import DiffView

from tldw_chatbook.Widgets.diff_widgets import (
    extract_diff_from_result,
    make_diff,
    set_default_diff_view_mode,
    strip_diff_contents,
)


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

        assert (
            diff_tool_result[0]["result"]["old_content"] == "def f():\n    return 1\n"
        )
        assert (
            diff_tool_result[0]["result"]["new_content"] == "def f():\n    return 2\n"
        )

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
