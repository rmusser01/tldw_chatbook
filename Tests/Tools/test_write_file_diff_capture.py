"""Tests for before/after content capture in file-writing tools (TASK-1351)."""

import pytest

from tldw_chatbook.Agents.tool_catalog import BuiltinToolProvider
from tldw_chatbook.Tools import file_operation_tools as fot
from tldw_chatbook.Tools import workspace_file_roots as wfr
from tldw_chatbook.Tools.file_operation_tools import (
    DIFF_CAPTURE_MAX_BYTES,
    WriteFileTool,
)


@pytest.fixture(autouse=True)
def _sandbox_only_roots(monkeypatch, tmp_path):
    """Point the file-tool sandbox at tmp_path, with no workspace roots.

    Same idiom as ``Tests/Tools/test_file_tool_sandbox.py``: raising from
    the registry factory drives ``allowed_file_roots`` into its documented
    fail-safe fallback (sandbox-only).
    """
    sandbox = (tmp_path / "tool_sandbox").resolve()
    sandbox.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(fot, "_tool_sandbox_root", lambda: sandbox)

    def _raise():
        raise RuntimeError("no workspace registry in this test")

    monkeypatch.setattr(wfr, "_registry_factory", _raise)
    return sandbox


@pytest.fixture
def tool():
    return WriteFileTool()


class TestWriteFileDiffCapture:
    """WriteFileTool results carry old_content/new_content for diff rendering."""

    @pytest.mark.asyncio
    async def test_new_file_has_empty_old_content(self, tool, _sandbox_only_roots):
        result = await tool.execute(file_path="new_file.txt", content="hello\nworld\n")

        assert "error" not in result
        assert result["action"] == "created"
        assert result["old_content"] == ""
        assert result["new_content"] == "hello\nworld\n"
        assert result["file_path"] == str(_sandbox_only_roots / "new_file.txt")

    @pytest.mark.asyncio
    async def test_overwrite_captures_old_content(self, tool, _sandbox_only_roots):
        target = _sandbox_only_roots / "existing.txt"
        target.write_text("line one\nline two\n", encoding="utf-8")

        result = await tool.execute(
            file_path="existing.txt", content="line one\nchanged\n"
        )

        assert "error" not in result
        assert result["action"] == "overwritten"
        assert result["old_content"] == "line one\nline two\n"
        assert result["new_content"] == "line one\nchanged\n"

    @pytest.mark.asyncio
    async def test_append_captures_combined_new_content(self, tool, _sandbox_only_roots):
        target = _sandbox_only_roots / "append.txt"
        target.write_text("start\n", encoding="utf-8")

        result = await tool.execute(
            file_path="append.txt", content="more\n", mode="append"
        )

        assert "error" not in result
        assert result["action"] == "appended to"
        assert result["old_content"] == "start\n"
        assert result["new_content"] == "start\nmore\n"

    @pytest.mark.asyncio
    async def test_undecodable_old_content_skips_capture(self, tool, _sandbox_only_roots):
        """Binary (undecodable) pre-existing files omit the capture keys —
        a fabricated "before" state would render a misleading diff."""
        target = _sandbox_only_roots / "binary.bin"
        target.write_bytes(b"\xff\xfe\x00\x01binary")

        result = await tool.execute(file_path="binary.bin", content="text\n")

        assert "error" not in result
        assert "old_content" not in result
        assert "new_content" not in result

    @pytest.mark.asyncio
    async def test_undecodable_append_skips_capture(self, tool, _sandbox_only_roots):
        """Append to an undecodable file omits the capture keys too."""
        target = _sandbox_only_roots / "binary.bin"
        target.write_bytes(b"\xff\xfe\x00\x01binary")

        result = await tool.execute(
            file_path="binary.bin", content="text\n", mode="append"
        )

        assert "error" not in result
        assert result["action"] == "appended to"
        assert "old_content" not in result
        assert "new_content" not in result

    @pytest.mark.asyncio
    async def test_oversized_existing_file_skips_capture(self, tool, _sandbox_only_roots):
        """Pre-existing files over DIFF_CAPTURE_MAX_BYTES are not captured."""
        target = _sandbox_only_roots / "big.txt"
        target.write_text("x" * (DIFF_CAPTURE_MAX_BYTES + 1), encoding="utf-8")

        result = await tool.execute(file_path="big.txt", content="small\n")

        assert "error" not in result
        assert result["action"] == "overwritten"
        assert "old_content" not in result
        assert "new_content" not in result
        # The write itself still happened.
        assert target.read_text(encoding="utf-8") == "small\n"

    @pytest.mark.asyncio
    async def test_oversized_new_content_skips_capture(self, tool, _sandbox_only_roots):
        """Writes whose new content exceeds the cap are not captured."""
        result = await tool.execute(
            file_path="new_big.txt", content="y" * (DIFF_CAPTURE_MAX_BYTES + 1)
        )

        assert "error" not in result
        assert result["action"] == "created"
        assert "old_content" not in result
        assert "new_content" not in result

    @pytest.mark.asyncio
    async def test_content_at_cap_is_captured(self, tool, _sandbox_only_roots):
        """Content exactly at the cap is still captured (boundary check)."""
        content = "z" * DIFF_CAPTURE_MAX_BYTES

        result = await tool.execute(file_path="at_cap.txt", content=content)

        assert "error" not in result
        assert result["old_content"] == ""
        assert result["new_content"] == content

    @pytest.mark.asyncio
    async def test_append_combined_size_over_cap_skips_capture(self, tool, _sandbox_only_roots):
        """Append mode materializes new_content as old + appended, so the
        combined size (not just the appended chunk) must stay within the cap."""
        target = _sandbox_only_roots / "append_big.txt"
        old_text = "o" * (DIFF_CAPTURE_MAX_BYTES - 10)
        target.write_text(old_text, encoding="utf-8")
        chunk = "c" * 2000  # under the cap alone, over it combined

        result = await tool.execute(
            file_path="append_big.txt", content=chunk, mode="append"
        )

        assert "error" not in result
        assert result["action"] == "appended to"
        assert "old_content" not in result
        assert "new_content" not in result
        # The append itself still happened.
        assert target.read_text(encoding="utf-8") == old_text + chunk

    @pytest.mark.asyncio
    async def test_error_results_have_no_diff_content(self, tool):
        result = await tool.execute(content="no path given")
        assert result == {"error": "No file path provided"}
        assert "old_content" not in result
        assert "new_content" not in result


class AllowAllGate:
    """Provider gate stub that permits every tool."""

    def check(self, tool, run_id):
        return None


class TestBuiltinProviderStripsDiffContents:
    """The provider seam never replays raw diff contents to the LLM/run log.

    ``BuiltinToolProvider.invoke`` is where a builtin tool's result dict
    becomes the JSON text that feeds both the model history and the on-disk
    run log (TASK-1351) — old_content/new_content must not survive it.
    """

    def test_invoke_strips_diff_content_keys(self, _sandbox_only_roots):
        provider = BuiltinToolProvider(gate=AllowAllGate())
        provider._tools["write_file"] = WriteFileTool()

        result = provider.invoke(
            "builtin:write_file",
            {"file_path": "note.txt", "content": "secret before/after text\n"},
        )

        assert result.ok
        assert "old_content" not in result.content
        assert "new_content" not in result.content
        assert "secret before/after text" not in result.content
        # The ordinary result fields are still reported to the model.
        assert "note.txt" in result.content
        assert "created" in result.content

    def test_invoke_leaves_plain_results_untouched(self, _sandbox_only_roots):
        provider = BuiltinToolProvider(gate=AllowAllGate())
        provider._tools["write_file"] = WriteFileTool()

        result = provider.invoke(
            "builtin:write_file",
            {"file_path": "plain.txt", "content": "hi\n"},
        )

        assert result.ok
        assert '"action": "created"' in result.content
        assert '"lines_written": 1' in result.content


class TestBuiltinProviderDiffSink:
    """TASK-1366: invoke hands raw diff contents to a UI-side sink.

    The sink is the narrow channel that lets the live Console render a
    diff: it fires at the strip seam, BEFORE old_content/new_content are
    removed from the dict that becomes the LLM/run-log-bound JSON, so the
    UI gets the raw contents while the strip guarantee is untouched.
    """

    def test_sink_receives_raw_contents_while_result_stays_stripped(
        self, _sandbox_only_roots
    ):
        captured = []
        provider = BuiltinToolProvider(gate=AllowAllGate(), diff_sink=captured.append)
        provider._tools["write_file"] = WriteFileTool()

        result = provider.invoke(
            "builtin:write_file",
            {"file_path": "note.txt", "content": "secret after text\n"},
        )

        assert result.ok
        assert captured == [
            (
                "write_file",
                str(_sandbox_only_roots / "note.txt"),
                "",
                "secret after text\n",
            )
        ]
        # The strip guarantee is unchanged: nothing raw in the LLM-bound text.
        assert "old_content" not in result.content
        assert "new_content" not in result.content
        assert "secret after text" not in result.content

    def test_sink_not_called_for_results_without_diff_content(
        self, _sandbox_only_roots
    ):
        captured = []
        provider = BuiltinToolProvider(gate=AllowAllGate(), diff_sink=captured.append)

        result = provider.invoke("builtin:calculator", {"expression": "6*7"})

        assert result.ok
        assert captured == []

    def test_sink_not_called_when_unset(self, _sandbox_only_roots):
        """Default construction (no sink) behaves exactly as before."""
        provider = BuiltinToolProvider(gate=AllowAllGate())
        provider._tools["write_file"] = WriteFileTool()

        result = provider.invoke(
            "builtin:write_file", {"file_path": "note.txt", "content": "hi\n"}
        )

        assert result.ok
        assert '"action": "created"' in result.content

    def test_raising_sink_never_breaks_invoke(self, _sandbox_only_roots):
        """A UI-side failure must never fail or corrupt the tool call."""

        def boom(_capture):
            raise RuntimeError("UI exploded")

        provider = BuiltinToolProvider(gate=AllowAllGate(), diff_sink=boom)
        provider._tools["write_file"] = WriteFileTool()

        result = provider.invoke(
            "builtin:write_file", {"file_path": "note.txt", "content": "hi\n"}
        )

        assert result.ok
        assert '"action": "created"' in result.content
        assert "old_content" not in result.content
