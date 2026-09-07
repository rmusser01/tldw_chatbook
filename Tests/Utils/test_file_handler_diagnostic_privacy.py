"""Born-red runtime privacy contracts for attachment file diagnostics."""

from pathlib import Path

import pytest
from loguru import logger

from tldw_chatbook.Utils.file_handlers import FileHandlerRegistry, TextFileHandler
from tldw_chatbook.Utils.log_sanitizer import content_fingerprint


@pytest.mark.asyncio
async def test_text_handler_failure_logs_only_safe_path_metadata(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "task-19864-private-reading-list.txt"
    path.write_text("content that proves stat succeeds", encoding="utf-8")
    raw_exception = f"TASK-19864 read failure repeated path={path}"
    original_read_text = Path.read_text

    def fail_for_private_path(self: Path, *args: object, **kwargs: object) -> str:
        if self == path:
            raise OSError(raw_exception)
        return original_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", fail_for_private_path)
    records: list[str] = []
    sink_id = logger.add(lambda message: records.append(str(message)))
    try:
        result = await TextFileHandler().process(path)
    finally:
        logger.remove(sink_id)

    rendered = "".join(records)
    assert result.content == f"[Error reading file: {path.name}]"
    assert "Failed to read text file" in rendered
    assert f"path_sha256={content_fingerprint(str(path))}" in rendered
    assert "exception_type=OSError" in rendered
    assert str(path) not in rendered
    assert path.name not in rendered
    assert raw_exception not in rendered


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["not-found", "no-handler"])
async def test_registry_failure_logs_only_safe_path_metadata(
    failure: str, tmp_path: Path
) -> None:
    path = tmp_path / f"task-19864-private-{failure}.opaque"
    registry = FileHandlerRegistry()
    if failure == "no-handler":
        path.write_text("present but deliberately unclaimed", encoding="utf-8")
        registry.handlers = []

    records: list[str] = []
    sink_id = logger.add(lambda message: records.append(str(message)))
    try:
        result = await registry.process_file(path)
    finally:
        logger.remove(sink_id)

    rendered = "".join(records)
    event_label = "File not found" if failure == "not-found" else "No handler found"
    expected_content = (
        f"[File not found: {path.name}]"
        if failure == "not-found"
        else f"[No handler for: {path.name}]"
    )
    assert result.content == expected_content
    assert event_label in rendered
    assert f"path_sha256={content_fingerprint(str(path))}" in rendered
    assert str(path) not in rendered
    assert path.name not in rendered
