"""Storage failures retain useful context without persisting character content."""

from __future__ import annotations

import logging

import pytest

from Tests.MCP.test_character_authoring import character_tools  # noqa: F401
from Tests.test_persistent_diagnostic_boundary import (
    _all_generations,
    _real_private_sink,
)
from tldw_chatbook.Character_Chat.local_character_persona_service import (
    LocalCharacterPersonaService,
)

pytestmark = pytest.mark.bootstrap_profile


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["create_character", "update_character"])
async def test_storage_failure_persists_only_safe_operation_and_raise_site(
    character_tools,  # noqa: F811 -- shared fixture
    monkeypatch,
    tmp_path,
    operation,
):
    private_content = "PRIVATE-CARD-SENTINEL-sk-not-a-real-key"
    created = await character_tools.create_character("Existing")
    before = character_tools.chachanotes_db.list_character_cards()

    def fail_storage(*args, **kwargs):
        raise RuntimeError(f"/private/cards/{private_content}: database failed")

    monkeypatch.setattr(LocalCharacterPersonaService, operation, fail_storage)
    path = tmp_path / "diagnostics.log"
    sink = _real_private_sink(path)
    diagnostic_logger = logging.getLogger("tldw_chatbook.diagnostics.mcp")
    previous_level = diagnostic_logger.level
    diagnostic_logger.setLevel(logging.DEBUG)
    diagnostic_logger.addHandler(sink)
    try:
        if operation == "create_character":
            result = await character_tools.create_character(
                private_content, {"description": private_content}
            )
        else:
            result = await character_tools.update_character(
                created["id"], 1, {"description": private_content}
            )
        sink.flush()
    finally:
        diagnostic_logger.removeHandler(sink)
        diagnostic_logger.setLevel(previous_level)
        sink.close()

    assert result == {
        "error_code": "storage_error",
        "error": "Character could not be saved.",
    }
    assert character_tools.chachanotes_db.list_character_cards() == before
    persisted = _all_generations(path)
    assert f"operation={operation}" in persisted
    assert "exception_type=RuntimeError" in persisted
    assert "raise_function=fail_storage" in persisted
    assert "raise_line=" in persisted
    assert private_content not in persisted
    assert "/private/cards/" not in persisted
    assert "database failed" not in persisted
    assert "description" not in persisted
