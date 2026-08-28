"""Privacy contracts for ChaChaNotes diagnostic exception handling."""

import ast
import sqlite3
from collections import Counter
from pathlib import Path
from textwrap import dedent

import pytest
from loguru import logger

from scripts.check_persistent_diagnostic_inventory import (
    _is_diagnostic_call,
    _logger_symbols,
)
import tldw_chatbook.DB.ChaChaNotes_DB as chachanotes_module
from tldw_chatbook.DB.ChaChaNotes_DB import (
    CharactersRAGDB,
    CharactersRAGDBError,
    SchemaError,
)
from tldw_chatbook.Utils.log_sanitizer import content_fingerprint


_CHACHANOTES_SOURCE_PATH = (
    Path(__file__).resolve().parents[2] / "tldw_chatbook/DB/ChaChaNotes_DB.py"
)


def _target_names(target: ast.AST) -> set[str]:
    if isinstance(target, ast.Name):
        return {target.id}
    if isinstance(target, (ast.Tuple, ast.List)):
        return {name for element in target.elts for name in _target_names(element)}
    if isinstance(target, ast.Starred):
        return _target_names(target.value)
    return set()


def _nearest_exception_handler(
    node: ast.AST, parent_by_node: dict[int, ast.AST]
) -> ast.ExceptHandler | None:
    ancestor = parent_by_node.get(id(node))
    while ancestor is not None:
        if isinstance(ancestor, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            return None
        if isinstance(ancestor, ast.ExceptHandler):
            return ancestor
        ancestor = parent_by_node.get(id(ancestor))
    return None


def _is_safe_exception_type_name(node: ast.AST, tainted_names: set[str]) -> bool:
    if not (
        isinstance(node, ast.Attribute)
        and node.attr == "__name__"
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "type"
        and len(node.value.args) == 1
        and not node.value.keywords
    ):
        return False
    return any(
        isinstance(child, ast.Name) and child.id in tainted_names
        for child in ast.walk(node.value.args[0])
    )


def _contains_raw_exception_payload(node: ast.AST, tainted_names: set[str]) -> bool:
    if _is_safe_exception_type_name(node, tainted_names):
        return False
    if isinstance(node, ast.Name) and node.id in tainted_names:
        return True
    return any(
        _contains_raw_exception_payload(child, tainted_names)
        for child in ast.iter_child_nodes(node)
    )


def _handler_derived_exception_names(
    handler: ast.ExceptHandler,
    parent_by_node: dict[int, ast.AST],
) -> set[str]:
    if handler.name is None:
        return set()
    tainted_names = {handler.name}
    assignments: list[tuple[set[str], ast.AST]] = []
    for node in ast.walk(handler):
        if (
            node is not handler
            and _nearest_exception_handler(node, parent_by_node) is not handler
        ):
            continue
        if isinstance(node, ast.Assign):
            targets = {
                name for target in node.targets for name in _target_names(target)
            }
            assignments.append((targets, node.value))
        elif isinstance(node, (ast.AnnAssign, ast.NamedExpr)):
            assignments.append((_target_names(node.target), node.value))

    changed = True
    while changed:
        changed = False
        for targets, value in assignments:
            if targets - tainted_names and _contains_raw_exception_payload(
                value, tainted_names
            ):
                tainted_names.update(targets)
                changed = True
    return tainted_names


def _enabled_exception_capture(call: ast.Call) -> str | None:
    if call.func.attr == "exception":
        return "logger.exception"

    for receiver_node in ast.walk(call.func.value):
        if not (
            isinstance(receiver_node, ast.Call)
            and isinstance(receiver_node.func, ast.Attribute)
            and receiver_node.func.attr == "opt"
        ):
            continue
        exception_option = next(
            (
                keyword.value
                for keyword in receiver_node.keywords
                if keyword.arg == "exception"
            ),
            None,
        )
        if exception_option is None:
            continue
        if isinstance(exception_option, ast.Constant) and exception_option.value in (
            False,
            None,
        ):
            continue
        return f"logger.opt(exception={ast.unparse(exception_option)})"
    return None


def _scan_exception_diagnostic_offenders(
    source: str, *, filename: str
) -> list[dict[str, object]]:
    """Return every diagnostic that can render a caught exception payload."""
    tree = ast.parse(source, filename=filename)
    logger_symbols = _logger_symbols(tree)
    parent_by_node = {
        id(child): parent
        for parent in ast.walk(tree)
        for child in ast.iter_child_nodes(parent)
    }
    handler_taint: dict[int, set[str]] = {
        id(handler): _handler_derived_exception_names(handler, parent_by_node)
        for handler in ast.walk(tree)
        if isinstance(handler, ast.ExceptHandler) and handler.name is not None
    }
    offenders: list[dict[str, object]] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not _is_diagnostic_call(
            node, logger_symbols
        ):
            continue

        reasons: list[str] = []
        capture = _enabled_exception_capture(node)
        if capture is not None:
            reasons.append(capture)

        handler = _nearest_exception_handler(node, parent_by_node)
        if handler is not None and _contains_raw_exception_payload(
            node, handler_taint.get(id(handler), set())
        ):
            reasons.append("raw_exception_payload")

        if reasons:
            offenders.append(
                {
                    "line": node.lineno,
                    "column": node.col_offset + 1,
                    "method": node.func.attr,
                    "reasons": tuple(reasons),
                    "call": ast.unparse(node),
                }
            )

    return sorted(
        offenders,
        key=lambda offender: (
            offender["line"],
            offender["column"],
            offender["method"],
        ),
    )


def _format_exception_diagnostic_offenders(
    offenders: list[dict[str, object]],
) -> str:
    reason_counts = Counter(
        reason for offender in offenders for reason in offender["reasons"]
    )
    categories = ", ".join(
        f"{reason}={count}" for reason, count in sorted(reason_counts.items())
    )
    details = "\n".join(
        f"  line {offender['line']}:{offender['column']} "
        f"{offender['method']} reasons={','.join(offender['reasons'])}: "
        f"{offender['call']}"
        for offender in offenders
    )
    return f"{len(offenders)} offending diagnostic calls ({categories})" + (
        f"\n{details}" if details else ""
    )


@pytest.mark.parametrize(
    "diagnostic",
    [
        pytest.param('logger.error(f"Failure: {exc}")', id="f-string"),
        pytest.param(
            'logger.error("Failure: {}", exc)', id="loguru-positional-argument"
        ),
        pytest.param(
            'logger.error("Failure: {error}", error=exc)',
            id="loguru-keyword-argument",
        ),
        pytest.param('logger.error("Failure: {}".format(exc))', id="dot-format"),
        pytest.param('logger.error("Failure: %s" % exc)', id="percent-format"),
        pytest.param('logger.error("Failure: {}", str(exc))', id="str"),
        pytest.param('logger.error("Failure: {}", repr(exc))', id="repr"),
        pytest.param('logger.error("Failure: {}", exc.args)', id="attribute"),
    ],
)
def test_exception_diagnostic_scanner_flags_raw_handler_payloads(
    diagnostic: str,
) -> None:
    source = dedent(
        f"""
        try:
            run_database_operation()
        except Exception as exc:
            {diagnostic}
        """
    )

    offenders = _scan_exception_diagnostic_offenders(source, filename="synthetic.py")

    assert len(offenders) == 1
    assert offenders[0]["reasons"] == ("raw_exception_payload",)


def test_exception_diagnostic_scanner_flags_derived_handler_payload() -> None:
    source = dedent(
        """
        try:
            run_database_operation()
        except Exception as exc:
            details = repr(exc)
            rendered = details
            logger.error("Failure: {}", rendered)
        """
    )

    offenders = _scan_exception_diagnostic_offenders(source, filename="derived.py")

    assert len(offenders) == 1
    assert offenders[0]["reasons"] == ("raw_exception_payload",)


@pytest.mark.parametrize("exception_option", ["True", "capture_exception"])
def test_exception_diagnostic_scanner_flags_enabled_or_dynamic_opt_capture(
    exception_option: str,
) -> None:
    source = dedent(
        f"""
        try:
            run_database_operation()
        except Exception:
            logger.opt(exception={exception_option}).error("database operation failed")
        """
    )

    offenders = _scan_exception_diagnostic_offenders(source, filename="capture.py")

    assert len(offenders) == 1
    assert offenders[0]["reasons"] == (f"logger.opt(exception={exception_option})",)


@pytest.mark.parametrize("exception_option", ["False", "None"])
def test_exception_diagnostic_scanner_allows_disabled_opt_capture(
    exception_option: str,
) -> None:
    source = dedent(
        f"""
        try:
            run_database_operation()
        except Exception:
            logger.opt(exception={exception_option}).error("database operation failed")
        """
    )

    assert _scan_exception_diagnostic_offenders(source, filename="disabled.py") == []


def test_exception_diagnostic_scanner_flags_logger_exception() -> None:
    source = 'logger.exception("database operation failed")'

    offenders = _scan_exception_diagnostic_offenders(source, filename="exception.py")

    assert len(offenders) == 1
    assert offenders[0]["reasons"] == ("logger.exception",)


def test_scanner_allows_type_only_metadata_and_non_logger_exception_use() -> None:
    source = dedent(
        """
        try:
            run_database_operation()
        except Exception as exc:
            details = str(exc)
            logger.error(
                "database operation failed exception_type={}",
                type(exc).__name__,
            )
            consume(details)
            raise CharactersRAGDBError(f"operation failed: {exc}") from exc
        """
    )

    assert _scan_exception_diagnostic_offenders(source, filename="safe.py") == []


def test_exception_diagnostic_scanner_reports_the_complete_offender_set() -> None:
    source = dedent(
        """
        try:
            first_operation()
        except Exception as first_exc:
            logger.opt(exception=True).error("first failed")
            logger.error("second failed: {}", first_exc)
            logger.opt(exception=dynamic).critical(
                "third failed: {error}", error=first_exc
            )
        """
    )

    offenders = _scan_exception_diagnostic_offenders(source, filename="complete.py")
    report = _format_exception_diagnostic_offenders(offenders)

    assert len(offenders) == 3
    assert "3 offending diagnostic calls" in report
    assert "logger.opt(exception=True)=1" in report
    assert "logger.opt(exception=dynamic)=1" in report
    assert "raw_exception_payload=2" in report
    assert report.count("line ") == 3


def test_chachanotes_source_has_no_exception_rendering_diagnostics() -> None:
    source = _CHACHANOTES_SOURCE_PATH.read_text(encoding="utf-8")

    offenders = _scan_exception_diagnostic_offenders(
        source, filename=str(_CHACHANOTES_SOURCE_PATH)
    )

    assert offenders == [], _format_exception_diagnostic_offenders(offenders)


def test_migration_failure_preserves_caller_error_and_logs_only_exception_type(
    monkeypatch, tmp_path: Path
) -> None:
    database_path = tmp_path / "task-19864-private-migration.sqlite"
    raw_exception = f"TASK-19864 migration failure repeated path={database_path}"
    database = CharactersRAGDB(database_path, client_id="task-19864")
    expected_ref = database._db_diagnostic_ref

    def fail_version_check(_connection: sqlite3.Connection) -> int:
        raise SchemaError(raw_exception)

    monkeypatch.setattr(database, "_get_db_version", fail_version_check)
    records: list[str] = []
    sink_id = logger.add(lambda message: records.append(str(message)))
    try:
        with pytest.raises(SchemaError) as exc_info:
            database._initialize_schema()
    finally:
        logger.remove(sink_id)
        database.close()

    error = exc_info.value
    assert str(error) == (
        f"Schema initialization/migration for '{database._SCHEMA_NAME}' failed: "
        f"{raw_exception}"
    )
    assert isinstance(error.__cause__, SchemaError)
    assert str(error.__cause__) == raw_exception

    rendered = "".join(records)
    assert "Schema initialization/migration failed" in rendered
    assert f"db_sha256={expected_ref}" in rendered
    assert "exception_type=SchemaError" in rendered
    assert str(database_path) not in rendered
    assert database_path.name not in rendered
    assert raw_exception not in rendered
    assert "Traceback (most recent call last)" not in rendered


def test_delegated_reconnect_failure_preserves_caller_error_without_relogging_path(
    monkeypatch, tmp_path: Path
) -> None:
    database_path = tmp_path / "task-19864-private-reconnect.sqlite"
    raw_exception = f"TASK-19864 reconnect failure repeated path={database_path}"
    database = CharactersRAGDB(database_path, client_id="task-19864")
    expected_ref = database._db_diagnostic_ref
    database.close_connection()

    def fail_reconnect(*_args: object, **_kwargs: object) -> sqlite3.Connection:
        raise sqlite3.OperationalError(raw_exception)

    monkeypatch.setattr(chachanotes_module, "connect_private_sqlite", fail_reconnect)
    records: list[str] = []
    sink_id = logger.add(lambda message: records.append(str(message)))
    try:
        with pytest.raises(CharactersRAGDBError) as exc_info:
            database.list_all_active_conversations()
    finally:
        logger.remove(sink_id)
        database.close()

    error = exc_info.value
    assert str(error) == (
        f"Failed to connect to database '{database.db_path_str}': {raw_exception}"
    )
    assert isinstance(error.__cause__, sqlite3.OperationalError)
    assert str(error.__cause__) == raw_exception

    rendered = "".join(records)
    assert "Failed to connect to database" in rendered
    assert "Database error listing all active conversations" in rendered
    assert f"db_sha256={expected_ref}" in rendered
    assert "exception_type=OperationalError" in rendered
    assert "exception_type=CharactersRAGDBError" in rendered
    assert str(database_path) not in rendered
    assert database_path.name not in rendered
    assert raw_exception not in rendered
    assert "Traceback (most recent call last)" not in rendered


def test_file_database_caches_one_diagnostic_fingerprint(
    monkeypatch, tmp_path: Path
) -> None:
    database_path = tmp_path / "task-19864-private-cached.sqlite"
    fingerprint_calls: list[object] = []

    def record_fingerprint(value: object) -> str:
        fingerprint_calls.append(value)
        return "cached-diagnostic-ref"

    monkeypatch.setattr(
        chachanotes_module,
        "content_fingerprint",
        record_fingerprint,
        raising=False,
    )
    database = CharactersRAGDB(database_path, client_id="task-19864")
    try:
        assert database.check_integrity() is True
        assert database._db_diagnostic_ref == "cached-diagnostic-ref"
    finally:
        database.close()
    assert fingerprint_calls == [str(database_path)]


def test_memory_database_uses_fixed_diagnostic_reference(
    monkeypatch,
) -> None:
    fingerprint_calls: list[object] = []

    def record_fingerprint(value: object) -> str:
        fingerprint_calls.append(value)
        return "must-not-be-used"

    monkeypatch.setattr(
        chachanotes_module,
        "content_fingerprint",
        record_fingerprint,
        raising=False,
    )
    database = CharactersRAGDB(":memory:", client_id="task-19864")
    try:
        assert database._db_diagnostic_ref == "memory"
    finally:
        database.close()
    assert fingerprint_calls == []


def test_integrity_failure_logs_stable_metadata_without_database_path(
    monkeypatch, tmp_path: Path
) -> None:
    database_path = tmp_path / "task-19864-private-integrity.sqlite"
    raw_exception = f"TASK-19864 integrity failure repeated path={database_path}"
    database = CharactersRAGDB(database_path, client_id="task-19864")
    expected_ref = database._db_diagnostic_ref

    def fail_integrity_connection() -> sqlite3.Connection:
        raise sqlite3.OperationalError(raw_exception)

    monkeypatch.setattr(database, "get_connection", fail_integrity_connection)
    records: list[str] = []
    sink_id = logger.add(lambda message: records.append(str(message)))
    try:
        assert database.check_integrity() is False
    finally:
        logger.remove(sink_id)
        database.close()

    rendered = "".join(records)
    assert "Failed to check database integrity" in rendered
    assert str(database_path) not in rendered
    assert database_path.name not in rendered
    assert raw_exception not in rendered
    assert "Traceback (most recent call last)" not in rendered
    assert expected_ref == content_fingerprint(str(database_path))
    assert f"db_sha256={expected_ref}" in rendered
    assert "exception_type=OperationalError" in rendered


def test_backup_failure_logs_stable_metadata_without_database_paths(
    monkeypatch, tmp_path: Path
) -> None:
    database_path = tmp_path / "task-19864-private-primary.sqlite"
    backup_path = tmp_path / "task-19864-private-backup.sqlite"
    raw_exception = f"TASK-19864 backup failed from {database_path} to {backup_path}"
    database = CharactersRAGDB(database_path, client_id="task-19864")

    def fail_backup(*_args: object, **_kwargs: object) -> None:
        raise sqlite3.OperationalError(raw_exception)

    monkeypatch.setattr(chachanotes_module, "backup_connection_to_private", fail_backup)
    records: list[str] = []
    sink_id = logger.add(lambda message: records.append(str(message)))
    try:
        assert database.backup_database(str(backup_path)) is False
    finally:
        logger.remove(sink_id)
        database.close()

    rendered = "".join(records)
    assert "Starting database backup" in rendered
    assert "SQLite error during database backup" in rendered
    assert f"db_sha256={content_fingerprint(str(database_path))}" in rendered
    assert f"backup_sha256={content_fingerprint(str(backup_path))}" in rendered
    assert "exception_type=OperationalError" in rendered
    assert str(database_path) not in rendered
    assert str(backup_path) not in rendered
    assert database_path.name not in rendered
    assert backup_path.name not in rendered
    assert raw_exception not in rendered


def test_vacuum_failure_preserves_caller_error_and_logs_only_safe_metadata(
    monkeypatch, tmp_path: Path
) -> None:
    database_path = tmp_path / "task-19864-private-vacuum.sqlite"
    raw_exception = f"TASK-19864 vacuum failure repeated path={database_path}"
    database = CharactersRAGDB(database_path, client_id="task-19864")
    expected_ref = database._db_diagnostic_ref
    connection = database.get_connection()

    class VacuumFailureConnection:
        def execute(self, statement: str, *args: object) -> sqlite3.Cursor:
            if statement == "VACUUM":
                raise sqlite3.OperationalError(raw_exception)
            return connection.execute(statement, *args)

    monkeypatch.setattr(
        database,
        "get_connection",
        lambda: VacuumFailureConnection(),
    )
    records: list[str] = []
    sink_id = logger.add(lambda message: records.append(str(message)))
    try:
        with pytest.raises(CharactersRAGDBError) as exc_info:
            database.vacuum()
    finally:
        logger.remove(sink_id)
        database.close()

    error = exc_info.value
    assert str(error) == f"Vacuum failed: {raw_exception}"
    assert isinstance(error.__cause__, sqlite3.OperationalError)
    assert str(error.__cause__) == raw_exception

    rendered = "".join(records)
    assert "Failed to vacuum database" in rendered
    assert str(database_path) not in rendered
    assert database_path.name not in rendered
    assert raw_exception not in rendered
    assert "Traceback (most recent call last)" not in rendered
    assert expected_ref == content_fingerprint(str(database_path))
    assert f"db_sha256={expected_ref}" in rendered
    assert "exception_type=OperationalError" in rendered
