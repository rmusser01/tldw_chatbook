"""Real-file validation controls for the Console's explicit Markdown export."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import aiofiles
import pytest
from loguru import logger

from tldw_chatbook.UI.Console_Modules.wiring import build_console_row_actions_controller
from tldw_chatbook.Utils import path_validation


@pytest.fixture
def export_writer(monkeypatch):
    """Observe actual IO while preserving the real validator and filesystem."""
    notify = Mock()
    controller = build_console_row_actions_controller(
        SimpleNamespace(app=SimpleNamespace(notify=notify))
    )
    validated, created, opened = [], [], []
    real_validate = path_validation.validate_path
    real_mkdir = Path.mkdir
    real_open = aiofiles.open

    def validate(*args, **kwargs):
        result = real_validate(*args, **kwargs)
        validated.append(result)
        return result

    def mkdir(path, *args, **kwargs):
        created.append(path)
        return real_mkdir(path, *args, **kwargs)

    def open_file(path, *args, **kwargs):
        opened.append(path)
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(path_validation, "validate_path", validate)
    monkeypatch.setattr(Path, "mkdir", mkdir)
    monkeypatch.setattr(aiofiles, "open", open_file)
    return controller, notify, validated, created, opened


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "spelling", ["relative", "tilde", "new-parent", "hidden", "in-parent-link"]
)
async def test_export_passes_canonical_destination_to_real_io(
    tmp_path, monkeypatch, export_writer, spelling: str
) -> None:
    """Explicit destinations stay usable and both IO operations use validation."""
    controller, notify, validated, created, opened = export_writer
    destination = tmp_path / "export.md"
    if spelling == "relative":
        monkeypatch.chdir(tmp_path)
        requested = "export.md"
    elif spelling == "tilde":
        monkeypatch.setenv("HOME", str(tmp_path))
        requested = "~/export.md"
    elif spelling == "new-parent":
        destination = tmp_path / "new" / "nested" / "export.md"
        requested = str(destination)
    elif spelling == "hidden":
        destination = tmp_path / ".exports" / ".export.md"
        requested = str(destination)
    else:
        alias = tmp_path / "alias.md"
        alias.symlink_to(destination)
        requested = str(alias)
    expected = destination.resolve()
    markdown = "# Export\n\nA real UTF-8 transcript: café.\n"
    created.clear()

    await controller._write_console_markdown_file(requested, markdown)

    assert destination.read_text(encoding="utf-8") == markdown
    assert validated == [expected]
    assert created[0] == expected.parent
    assert opened == [expected]
    assert opened[0] is validated[0]
    notify.assert_called_once_with(f"Saved {expected.name}.")


@pytest.mark.asyncio
@pytest.mark.parametrize("existing", [False, True])
async def test_export_rejects_symlink_outside_selected_parent_without_io(
    tmp_path, export_writer, existing: bool
) -> None:
    """A stable final symlink must not create or overwrite an outside file."""
    controller, notify, validated, created, opened = export_writer
    selected_parent = tmp_path / "selected"
    selected_parent.mkdir()
    outside = tmp_path / "outside.md"
    if existing:
        outside.write_text("keep original", encoding="utf-8")
    alias = selected_parent / "export.md"
    alias.symlink_to(outside)
    created.clear()

    await controller._write_console_markdown_file(str(alias), "must not write")

    if existing:
        assert outside.read_text(encoding="utf-8") == "keep original"
    else:
        assert not outside.exists()
    assert alias.is_symlink()
    assert validated == created == opened == []
    notify.assert_called_once()
    assert notify.call_args.args[0].startswith("Invalid path:")
    assert notify.call_args.kwargs == {"severity": "error"}


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["bad\x00.md", "bad;name.md", "../../escape.md"])
async def test_export_preserves_lexical_rejection_without_io(
    tmp_path, export_writer, name: str
) -> None:
    """Canonicalization must not remove the existing lexical refusal boundary."""
    controller, notify, validated, created, opened = export_writer
    before = set(tmp_path.rglob("*"))
    created.clear()

    await controller._write_console_markdown_file(str(tmp_path / name), "rejected")

    assert set(tmp_path.rglob("*")) == before
    assert validated == created == opened == []
    notify.assert_called_once()
    assert notify.call_args.args[0].startswith("Invalid path:")
    assert notify.call_args.kwargs == {"severity": "error"}


@pytest.mark.asyncio
async def test_export_symlink_refusal_logs_no_selected_or_resolved_path(
    tmp_path, export_writer
) -> None:
    """Real validator diagnostics report refusal without private path content."""
    controller, notify, validated, created, opened = export_writer
    selected = tmp_path / "private-selected-directory"
    selected.mkdir()
    outside = tmp_path / "private-outside-transcript.md"
    alias = selected / "private-selected-transcript.md"
    alias.symlink_to(outside)
    created.clear()
    messages = []
    sink = logger.add(messages.append, format="{message}", level="WARNING")
    try:
        logger.warning("export-log-capture-canary")
        await controller._write_console_markdown_file(str(alias), "rejected")
    finally:
        logger.remove(sink)

    captured = "".join(messages)
    assert "export-log-capture-canary" in captured
    assert "Path traversal attempt detected." in captured
    assert "Path validation failed (category=ValueError)." in captured
    assert str(tmp_path) not in captured
    assert "private-selected" not in captured
    assert "private-outside" not in captured
    assert not outside.exists()
    assert validated == created == opened == []
    notify.assert_called_once_with(
        "Invalid path: Path is outside the allowed directory", severity="error"
    )
