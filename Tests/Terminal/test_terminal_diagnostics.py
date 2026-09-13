"""Content-free diagnostic characterization for persistent Terminal sessions."""

from __future__ import annotations

import ast
from collections.abc import Callable, Mapping
from dataclasses import asdict
from pathlib import Path

import pytest
from loguru import logger

from tldw_chatbook.Terminal.contracts import (
    AdmissionGate,
    BackendIdentity,
    CleanupAttempt,
    CleanupProof,
    TerminalLaunchRequest,
    TerminalLifecycle,
    TerminalReason,
)
from tldw_chatbook.Terminal.launch import build_terminal_environment
from tldw_chatbook.Terminal.protocol_gate import TerminalProtocolGate
from tldw_chatbook.Terminal.session_manager import TerminalSessionManager


REPO_ROOT = Path(__file__).resolve().parents[2]
TERMINAL_ROOT = REPO_ROOT / "tldw_chatbook" / "Terminal"
PRIVATE_SENTINELS = {
    "name": "TERM-DIAG-NAME-22512",
    "starting_path": "TERM-DIAG-PATH-22512",
    "input": "TERM-DIAG-INPUT-22512",
    "output": "TERM-DIAG-OUTPUT-22512",
    "environment": "TERM-DIAG-ENV-22512",
    "profile": "TERM-DIAG-PROFILE-22512",
    "rejected_paste": "TERM-DIAG-PASTE-22512",
    "parser_failure": "TERM-DIAG-PARSER-22512",
    "backend_failure": "TERM-DIAG-BACKEND-22512",
    "cleanup_unproven": "TERM-DIAG-CLEANUP-22512",
}
ALLOWED_DIAGNOSTIC_FIELDS = frozenset(
    {
        "session_id",
        "lifecycle",
        "timestamp",
        "duration_ms",
        "input_bytes",
        "output_bytes",
        "columns",
        "rows",
        "failure_category",
        "buffered_bytes",
        "discarding",
        "rejected_sequences",
        "ignored_sequences",
    }
)
LOG_METHODS = frozenset(
    {
        "trace",
        "debug",
        "info",
        "success",
        "warning",
        "error",
        "critical",
        "exception",
        "log",
        "catch",
    }
)


class _Screen:
    failure_reason = None

    def __init__(self) -> None:
        self.seen: list[bytes] = []

    def feed(self, data: bytes) -> None:
        self.seen.append(data)
        if PRIVATE_SENTINELS["parser_failure"].encode() in data:
            raise RuntimeError(PRIVATE_SENTINELS["parser_failure"])


class _Backend:
    def __init__(
        self,
        *,
        environment: Mapping[str, str] | None = None,
        start_failure: bool = False,
        cleanup_failure: bool = False,
    ) -> None:
        self.start_failure = start_failure
        self.cleanup_failure = cleanup_failure
        self.environment = dict(environment or {})
        self.profile = PRIVATE_SENTINELS["profile"]
        self.observed_private_launch_values: list[str] = []
        self.started_session_id = ""

    def start(
        self, request: TerminalLaunchRequest, admission: AdmissionGate
    ) -> BackendIdentity:
        self.started_session_id = admission.token
        self.observed_private_launch_values.extend(
            [
                request.name,
                request.start_directory,
                request.shell,
                *self.environment.values(),
                self.profile,
            ]
        )
        if self.start_failure:
            raise RuntimeError(PRIVATE_SENTINELS["backend_failure"])
        return BackendIdentity(session_id=admission.token)

    def write(self, data: bytes) -> None:
        del data

    def resize(self, columns: int, rows: int) -> None:
        del columns, rows

    def request_priority_close(self) -> None:
        return None

    def finalize_shutdown(self) -> None:
        return None

    def cleanup(self, attempt: CleanupAttempt) -> CleanupProof:
        del attempt
        if self.cleanup_failure:
            raise RuntimeError(PRIVATE_SENTINELS["cleanup_unproven"])
        return CleanupProof(True, True, True)


def _capture_loguru(action: Callable[[], None]) -> list[Mapping[str, object]]:
    captured: list[Mapping[str, object]] = []
    sink_id = logger.add(
        lambda message: captured.append(dict(message.record)),
        level="TRACE",
    )
    try:
        action()
    finally:
        logger.remove(sink_id)
    return captured


def _assert_private_diagnostics_absent(
    records: list[Mapping[str, object]], exported: list[Mapping[str, object]]
) -> None:
    rendered = "\n".join(repr(payload) for payload in [*records, *exported])
    for label, sentinel in PRIVATE_SENTINELS.items():
        assert sentinel not in rendered, f"{label} reached generic diagnostics"
    for payload in exported:
        assert set(payload) <= ALLOWED_DIAGNOSTIC_FIELDS
        if "lifecycle" in payload:
            assert payload["lifecycle"] in {
                lifecycle.value for lifecycle in TerminalLifecycle
            }
        if "failure_category" in payload:
            assert payload["failure_category"] in {
                reason.value for reason in TerminalReason
            }


def _manager(backend: _Backend, screens: list[_Screen]) -> TerminalSessionManager:
    def screen_factory(_columns: int, _rows: int) -> _Screen:
        screen = _Screen()
        screens.append(screen)
        return screen

    manager = TerminalSessionManager(
        lambda: True,
        lambda: backend,
        screen_model_factory=screen_factory,
    )
    assert manager.arm(acknowledge_disclosure=True).armed is True
    return manager


def _request(tmp_path: Path, *, name: str | None = None) -> TerminalLaunchRequest:
    start_directory = tmp_path / PRIVATE_SENTINELS["starting_path"]
    start_directory.mkdir(exist_ok=True)
    return TerminalLaunchRequest(
        name=name or PRIVATE_SENTINELS["name"],
        shell=PRIVATE_SENTINELS["profile"],
        start_directory=str(start_directory),
        columns=91,
        rows=37,
    )


def _scrubbed_environment(tmp_path: Path) -> dict[str, str]:
    home = tmp_path / "home"
    temporary = tmp_path / "tmp"
    binaries = tmp_path / "bin"
    for path in (home, temporary, binaries):
        path.mkdir(exist_ok=True)
    ambient = {
        "PATH": str(binaries),
        "PRIVATE_ENV": PRIVATE_SENTINELS["environment"],
    }
    environment = build_terminal_environment(
        platform_name="posix",
        ambient=ambient,
        account_reader=lambda: {
            "HOME": str(home),
            "USER": "diagnostic-user",
            "LOGNAME": "diagnostic-user",
            "SHELL": "/bin/sh",
        },
        system_reader=lambda: {"TMPDIR": str(temporary)},
        path_is_directory=lambda value: value == str(binaries),
    )
    assert PRIVATE_SENTINELS["environment"] in ambient.values()
    assert PRIVATE_SENTINELS["environment"] not in environment.values()
    return environment


def _exercise_private_failure_paths(tmp_path: Path) -> None:
    managers: list[TerminalSessionManager] = []
    screens: list[_Screen] = []
    backend = _Backend(environment=_scrubbed_environment(tmp_path))
    try:
        manager = _manager(backend, screens)
        managers.append(manager)
        created = manager.create_session(_request(tmp_path))
        assert created.projection is not None
        session_id = created.projection.session_id
        view = manager.attach_view()
        key = PRIVATE_SENTINELS["input"].encode()
        assert manager.send_key(session_id, key, view=view).accepted is True
        input_event = manager.take_input(session_id)
        assert input_event is not None and input_event.data == key
        rejected = manager.send_paste(
            session_id,
            PRIVATE_SENTINELS["rejected_paste"] + "\x1b",
            bracketed=True,
            view=view,
        )
        assert rejected.accepted is False
        output = PRIVATE_SENTINELS["output"].encode()
        assert manager.offer_output(session_id, output).accepted is True
        assert manager.process_output(session_id, visible=False) is not None
        assert output in screens[0].seen
        parser_failure = PRIVATE_SENTINELS["parser_failure"].encode()
        assert manager.offer_output(session_id, parser_failure).accepted is True
        assert manager.process_output(session_id, visible=False) is None
        assert manager.wait_for_cleanup(session_id, timeout_seconds=1)

        failed_backend = _Backend(start_failure=True)
        failed_manager = _manager(failed_backend, [])
        managers.append(failed_manager)
        failed = failed_manager.create_session(
            _request(tmp_path, name="TERM-DIAG-BACKEND-NAME-22512")
        )
        assert failed.reason is TerminalReason.SPAWN_FAILED
        assert failed_manager.wait_for_cleanup(
            failed_backend.started_session_id, timeout_seconds=1
        )

        cleanup_backend = _Backend(cleanup_failure=True)
        cleanup_manager = _manager(cleanup_backend, [])
        managers.append(cleanup_manager)
        cleanup = cleanup_manager.create_session(
            _request(tmp_path, name="TERM-DIAG-CLEANUP-NAME-22512")
        )
        assert cleanup.projection is not None
        cleanup_id = cleanup.projection.session_id
        cleanup_view = cleanup_manager.attach_view()
        assert cleanup_manager.close_session(cleanup_id, view=cleanup_view) is not None
        assert cleanup_manager.wait_for_cleanup(cleanup_id, timeout_seconds=1)
        retained = cleanup_manager.projection(cleanup_id)
        assert retained is not None
        assert retained.lifecycle is TerminalLifecycle.CLEANUP_UNPROVEN

        observed = "\n".join(backend.observed_private_launch_values)
        assert PRIVATE_SENTINELS["name"] in observed
        assert PRIVATE_SENTINELS["starting_path"] in observed
        assert PRIVATE_SENTINELS["environment"] not in observed
        assert PRIVATE_SENTINELS["profile"] in observed
    finally:
        for owned_manager in reversed(managers):
            owned_manager.disarm()
            for projection in owned_manager.projections():
                owned_manager.wait_for_cleanup(
                    projection.session_id,
                    timeout_seconds=1,
                )
            owned_manager.finalize_shutdown()


def _receiver_names(node: ast.AST) -> set[str]:
    if isinstance(node, ast.Name):
        return {node.id}
    if isinstance(node, ast.Attribute):
        return _receiver_names(node.value)
    if isinstance(node, ast.Call):
        return _receiver_names(node.func)
    return set()


def _logger_symbols(tree: ast.Module) -> set[str]:
    symbols: set[str] = set()
    factories: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            symbols.update(
                alias.asname or alias.name.split(".", 1)[0]
                for alias in node.names
                if alias.name in {"logging", "loguru"}
            )
        elif isinstance(node, ast.ImportFrom) and node.module == "loguru":
            symbols.update(alias.asname or alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module == "logging":
            for alias in node.names:
                name = alias.asname or alias.name
                if alias.name == "getLogger":
                    factories.add(name)
                else:
                    symbols.add(name)

    changed = True
    while changed:
        changed = False
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            value = node.value
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            target_names = {
                target.id for target in targets if isinstance(target, ast.Name)
            }
            if not target_names:
                continue
            is_logger_alias = isinstance(value, (ast.Name, ast.Attribute)) and bool(
                symbols.intersection(_receiver_names(value))
            )
            is_logger_factory = isinstance(value, ast.Call) and (
                isinstance(value.func, ast.Name) and value.func.id in factories
            )
            is_logger_derivation = isinstance(value, ast.Call) and (
                isinstance(value.func, ast.Attribute)
                and value.func.attr in {"bind", "getLogger", "opt", "patch"}
                and bool(symbols.intersection(_receiver_names(value.func.value)))
            )
            if is_logger_alias or is_logger_factory or is_logger_derivation:
                before = len(symbols)
                symbols.update(target_names)
                changed = changed or len(symbols) != before
    return symbols


def _production_diagnostic_sites(root: Path = TERMINAL_ROOT) -> list[str]:
    sites: list[str] = []
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        logger_symbols = _logger_symbols(tree)
        called_attributes = {
            id(node.func)
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                method = node.func.attr
                receiver = node.func.value
            elif (
                isinstance(node, ast.Attribute)
                and node.attr == "catch"
                and id(node) not in called_attributes
            ):
                method = node.attr
                receiver = node.value
            else:
                continue
            if method not in LOG_METHODS:
                continue
            receiver_names = _receiver_names(receiver)
            if logger_symbols.intersection(receiver_names) or any(
                name.casefold() in {"log", "logger", "logging", "loguru_logger"}
                or name.casefold().endswith("_logger")
                for name in receiver_names
            ):
                relative = path.relative_to(root).as_posix()
                sites.append(f"{relative}:{node.lineno}")
    return sites


def _capture_production_diagnostic_exports() -> list[Mapping[str, object]]:
    gate = TerminalProtocolGate()
    gate.feed(b"\x1b]" + PRIVATE_SENTINELS["output"].encode())
    live = asdict(gate.snapshot())
    finished = asdict(gate.finish())
    assert live["buffered_bytes"] == 0
    assert finished["rejected_sequences"] == 1
    return [live, finished]


def test_terminal_has_no_production_diagnostic_sites_to_inventory() -> None:
    assert _production_diagnostic_sites() == []


def test_diagnostic_site_scan_detects_chained_loguru_calls(tmp_path: Path) -> None:
    module = tmp_path / "nested" / "diagnostic_probe.py"
    module.parent.mkdir()
    module.write_text(
        "from loguru import logger as terminal_logger\n"
        "terminal_logger.opt(exception=True).warning('mutation probe')\n"
        "terminal_logger.log('INFO', 'second mutation probe')\n",
        encoding="utf-8",
    )

    assert _production_diagnostic_sites(tmp_path) == [
        "nested/diagnostic_probe.py:2",
        "nested/diagnostic_probe.py:3",
    ]


def test_diagnostic_scan_detects_logger_aliases_and_catch(tmp_path: Path) -> None:
    module = tmp_path / "diagnostic_aliases.py"
    module.write_text(
        "from logging import getLogger as make_logger\n"
        "from loguru import logger\n"
        "bound = logger.bind(component='terminal')\n"
        "bound.error('bound mutation probe')\n"
        "audit = make_logger(__name__)\n"
        "audit.warning('stdlib mutation probe')\n"
        "plain_alias = logger\n"
        "@plain_alias.catch\n"
        "def guarded():\n"
        "    return None\n",
        encoding="utf-8",
    )

    assert _production_diagnostic_sites(tmp_path) == [
        "diagnostic_aliases.py:4",
        "diagnostic_aliases.py:6",
        "diagnostic_aliases.py:8",
    ]


def test_private_terminal_paths_emit_no_generic_diagnostics(tmp_path: Path) -> None:
    records = _capture_loguru(lambda: _exercise_private_failure_paths(tmp_path))
    exported_diagnostics = _capture_production_diagnostic_exports()

    assert records == []
    assert exported_diagnostics
    _assert_private_diagnostics_absent(records, exported_diagnostics)


def test_diagnostic_guards_are_mutation_sensitive_for_every_private_value() -> None:
    for label, sentinel in PRIVATE_SENTINELS.items():
        records = _capture_loguru(
            lambda sentinel=sentinel: logger.bind(terminal_diagnostic_test=True).log(
                "TRACE", sentinel
            )
        )
        assert len(records) == 1 and sentinel in repr(records[0]), (
            f"Loguru mutation sink was not reached for {label}"
        )
        with pytest.raises(AssertionError, match=label):
            _assert_private_diagnostics_absent(records, [])

        exported = [{"session_id": sentinel}]
        assert exported[0]["session_id"] == sentinel
        with pytest.raises(AssertionError, match=label):
            _assert_private_diagnostics_absent([], exported)
