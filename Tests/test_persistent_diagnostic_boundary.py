from __future__ import annotations

import ast
import logging
from pathlib import Path

from loguru import logger as loguru_logger

from tldw_chatbook.Logging_Config import (
    PrivateRotatingFileHandler,
    _forward_loguru_to_standard,
)
from tldw_chatbook.Utils.persistent_diagnostics import (
    PersistentDiagnosticFilter,
    log_persistent_metadata,
)

PRIVATE_SENTINEL = "PRIVATE-PROMPT-SENTINEL-sk-not-a-real-key"

_CONSTANT_DIAGNOSTIC_PREFIXES = {
    "tldw_chatbook/Agents/mcp_tool_provider.py": (
        "MCPToolProvider: persona_policy_provider failed",
    ),
    "tldw_chatbook/Agents/persona_policy.py": (
        "Dropping non-mapping persona policy rule",
        "Dropping malformed persona policy rule",
    ),
    "tldw_chatbook/Character_Chat/local_character_persona_service.py": (
        "Dropping malformed persona policy rule",
    ),
    "tldw_chatbook/UI/Screens/personas_screen.py": (
        "Error saving persona policy rules",
    ),
    "tldw_chatbook/Workspaces/agent_provisioning.py": (
        "Workspace agent provisioning failed",
        "Workspace agent backfill could not persist defaults",
    ),
    "tldw_chatbook/Workspaces/registry_service.py": (
        "Workspace agent provisioning hook failed",
        "Workspace agent provisioning returned no defaults",
        "Workspace agent defaults could not be persisted",
        "Ignoring malformed workspace assistant_defaults",
    ),
}


def _real_private_sink(path: Path) -> PrivateRotatingFileHandler:
    handler = PrivateRotatingFileHandler(
        path,
        maxBytes=256,
        backupCount=2,
        encoding="utf-8",
    )
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter("%(levelname)s %(name)s %(message)s"))
    handler.addFilter(PersistentDiagnosticFilter())
    return handler


def _all_generations(path: Path) -> str:
    return "\n".join(
        candidate.read_text(encoding="utf-8")
        for candidate in sorted(path.parent.glob(f"{path.name}*"))
        if candidate.is_file()
    )


def _emit_owned_loguru_payload(module_name: str, message: str) -> None:
    source = (
        Path(__file__).resolve().parents[1] / Path(*module_name.split("."))
    ).with_suffix(".py")
    code = compile("loguru_logger.debug(message)", str(source), "exec")
    exec(
        code,
        {
            "__name__": module_name,
            "loguru_logger": loguru_logger,
            "message": message,
        },
    )


def _diagnostic_template(node: ast.expr) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        return "".join(
            part.value
            for part in node.values
            if isinstance(part, ast.Constant) and isinstance(part.value, str)
        )
    return None


def test_persona_workspace_diagnostics_do_not_interpolate_private_values() -> None:
    """Persistent diagnostics name the failure category, never private values."""
    root = Path(__file__).resolve().parents[1]
    for relative_path, prefixes in _CONSTANT_DIAGNOSTIC_PREFIXES.items():
        tree = ast.parse((root / relative_path).read_text(encoding="utf-8"))
        matched: set[str] = set()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not node.args:
                continue
            template = _diagnostic_template(node.args[0])
            if template is None:
                continue
            for prefix in prefixes:
                if not template.startswith(prefix):
                    continue
                matched.add(prefix)
                assert isinstance(node.args[0], ast.Constant), (
                    f"{relative_path}: {prefix!r} must use a constant message"
                )
                assert len(node.args) == 1, (
                    f"{relative_path}: {prefix!r} must not format runtime values"
                )
        assert matched == set(prefixes), (
            f"{relative_path}: expected diagnostics were renamed or removed"
        )


def test_real_rotating_sink_rejects_owned_standard_payloads_but_keeps_metadata(
    tmp_path: Path,
) -> None:
    path = tmp_path / "application.log"
    handler = _real_private_sink(path)
    owned_logger = logging.getLogger("tldw_chatbook.Chat.privacy_test")
    owned_logger.handlers = [handler]
    owned_logger.propagate = False
    owned_logger.setLevel(logging.DEBUG)
    try:
        owned_logger.debug("prompt=%s", PRIVATE_SENTINEL)
        owned_logger.error("response=%s", PRIVATE_SENTINEL)
        log_persistent_metadata(
            owned_logger,
            logging.INFO,
            "provider_request",
            provider="openai",
            status="success",
            payload_length=len(PRIVATE_SENTINEL),
        )
    finally:
        handler.close()
        owned_logger.handlers.clear()

    persisted = _all_generations(path)
    assert PRIVATE_SENTINEL not in persisted
    assert "event=provider_request" in persisted
    assert "provider=openai" in persisted
    assert "payload_length=" in persisted


def test_real_rotating_sink_rejects_loguru_payload_from_owned_module(
    tmp_path: Path,
) -> None:
    path = tmp_path / "application.log"
    handler = _real_private_sink(path)
    root = logging.getLogger()
    old_level = root.level
    root.setLevel(logging.DEBUG)
    root.addHandler(handler)
    sink_id = loguru_logger.add(_forward_loguru_to_standard, level="DEBUG")
    try:
        _emit_owned_loguru_payload(
            "tldw_chatbook.Agents.builtin_tool_gate",
            f"tool={PRIVATE_SENTINEL}",
        )
    finally:
        loguru_logger.remove(sink_id)
        root.removeHandler(handler)
        root.setLevel(old_level)
        handler.close()

    persisted = _all_generations(path)
    assert PRIVATE_SENTINEL not in persisted


def test_metadata_helper_rejects_unregistered_fields_and_private_values(
    tmp_path: Path,
) -> None:
    path = tmp_path / "application.log"
    handler = _real_private_sink(path)
    owned_logger = logging.getLogger("tldw_chatbook.Tools.privacy_test")
    owned_logger.handlers = [handler]
    owned_logger.propagate = False
    owned_logger.setLevel(logging.DEBUG)
    try:
        try:
            log_persistent_metadata(
                owned_logger,
                logging.INFO,
                "tool_execution",
                raw_argument=PRIVATE_SENTINEL,
            )
        except ValueError as exc:
            assert "raw_argument" in str(exc)
        else:
            raise AssertionError("unregistered persistent metadata was accepted")

        log_persistent_metadata(
            owned_logger,
            logging.INFO,
            "tool_execution",
            tool_name=PRIVATE_SENTINEL,
            status="success",
        )
    finally:
        handler.close()
        owned_logger.handlers.clear()

    persisted = _all_generations(path)
    assert PRIVATE_SENTINEL not in persisted
    assert "tool_name=invalid" in persisted
