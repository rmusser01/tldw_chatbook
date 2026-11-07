from __future__ import annotations

import ast
import logging
from pathlib import Path

from loguru import logger as loguru_logger

from tldw_chatbook.Logging_Config import (
    PrivateRotatingFileHandler,
    RedactingFileFormatter,
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
            latency_ms=17,
        )
    finally:
        handler.close()
        owned_logger.handlers.clear()

    persisted = _all_generations(path)
    assert PRIVATE_SENTINEL not in persisted
    assert "event=provider_request" in persisted
    assert "provider=openai" in persisted
    assert "payload_length=" in persisted
    assert "latency_ms=17" in persisted


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


def test_forwarder_renders_only_a_plain_bound_exception_type() -> None:
    """TASK-38989: a bounded log binds its failure class, and the forwarder
    (the only production loguru sink) renders that one key, only as a plain
    class name; the redacting formatter still covers the whole line."""

    class _Capture(logging.Handler):
        def __init__(self) -> None:
            super().__init__(logging.DEBUG)
            self.records: list[logging.LogRecord] = []

        def emit(self, record: logging.LogRecord) -> None:
            self.records.append(record)

    handler = _Capture()
    root = logging.getLogger()
    old_level = root.level
    root.setLevel(logging.DEBUG)
    root.addHandler(handler)
    sink_id = loguru_logger.add(_forward_loguru_to_standard, level="DEBUG")
    try:
        loguru_logger.bind(exception_type="RuntimeError").debug("r10a_case_one")
        loguru_logger.bind(exception_type="Not /a/class").debug("r10a_case_two")
        loguru_logger.bind(other=PRIVATE_SENTINEL).debug("r10a_case_three")
        loguru_logger.bind(exception_type="ValueError").debug(
            "r10a_case_four token=" + PRIVATE_SENTINEL
        )
    finally:
        loguru_logger.remove(sink_id)
        root.removeHandler(handler)
        root.setLevel(old_level)

    ours = [record for record in handler.records if record.getMessage().startswith("r10a_")]
    assert [record.getMessage() for record in ours[:3]] == [
        "r10a_case_one: RuntimeError",
        "r10a_case_two",
        "r10a_case_three",
    ]
    formatted = RedactingFileFormatter("%(message)s").format(ours[3])
    assert formatted.startswith("r10a_case_four")
    assert PRIVATE_SENTINEL not in formatted



def test_persona_policy_failure_forwards_only_its_error_class() -> None:
    """TASK-39060: the raising-provider log names the class, through the real
    forwarder, and neither the tool name nor the error text."""
    import asyncio
    from types import SimpleNamespace

    from tldw_chatbook.Agents.mcp_tool_provider import MCPToolProvider

    class _Capture(logging.Handler):
        def __init__(self) -> None:
            super().__init__(logging.DEBUG)
            self.records: list[logging.LogRecord] = []

        def emit(self, record: logging.LogRecord) -> None:
            self.records.append(record)

    def _raise() -> None:
        raise RuntimeError(PRIVATE_SENTINEL)

    loop = asyncio.new_event_loop()
    provider = MCPToolProvider(
        service=object(), main_loop=loop, persona_policy_provider=_raise
    )
    state = object()
    handler = _Capture()
    root = logging.getLogger()
    old_level = root.level
    root.setLevel(logging.DEBUG)
    root.addHandler(handler)
    sink_id = loguru_logger.add(_forward_loguru_to_standard, level="DEBUG")
    try:
        tool = SimpleNamespace(name="tool-" + PRIVATE_SENTINEL)
        assert provider._persona_floor(state, tool) is state
    finally:
        loguru_logger.remove(sink_id)
        root.removeHandler(handler)
        root.setLevel(old_level)
        loop.close()

    ours = [
        record
        for record in handler.records
        if record.getMessage().startswith("MCPToolProvider: persona_policy_provider")
    ]
    assert [record.getMessage() for record in ours] == [
        "MCPToolProvider: persona_policy_provider failed: RuntimeError"
    ]
    formatted = RedactingFileFormatter("%(message)s").format(ours[0])
    assert "RuntimeError" in formatted
    assert PRIVATE_SENTINEL not in formatted
