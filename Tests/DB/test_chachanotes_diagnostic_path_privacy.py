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


def _enclosing_exception_handlers(
    node: ast.AST, parent_by_node: dict[int, ast.AST]
) -> tuple[ast.ExceptHandler, ...]:
    handlers: list[ast.ExceptHandler] = []
    ancestor = parent_by_node.get(id(node))
    while ancestor is not None:
        if isinstance(ancestor, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            break
        if isinstance(ancestor, ast.ExceptHandler):
            handlers.append(ancestor)
        ancestor = parent_by_node.get(id(ancestor))
    return tuple(handlers)


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


def _lexical_scope(
    node: ast.AST, parent_by_node: dict[int, ast.AST]
) -> ast.Module | ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef:
    ancestor: ast.AST | None = node
    while ancestor is not None:
        if isinstance(
            ancestor, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
        ):
            return ancestor
        ancestor = parent_by_node.get(id(ancestor))
    raise AssertionError("AST node has no lexical scope")


def _assignment_parts(node: ast.AST) -> tuple[set[str], ast.AST] | None:
    if isinstance(node, ast.Assign):
        targets = {name for target in node.targets for name in _target_names(target)}
        return targets, node.value
    if isinstance(node, (ast.AnnAssign, ast.NamedExpr)) and node.value is not None:
        return _target_names(node.target), node.value
    return None


def _ends_before(node: ast.AST, later: ast.AST) -> bool:
    end_line = getattr(node, "end_lineno", None) or node.lineno
    end_column = getattr(node, "end_col_offset", None) or node.col_offset
    return (end_line, end_column) <= (later.lineno, later.col_offset)


def _comes_from_mutually_exclusive_handler(
    node: ast.AST,
    call_handlers: tuple[ast.ExceptHandler, ...],
    parent_by_node: dict[int, ast.AST],
) -> bool:
    for handler in _enclosing_exception_handlers(node, parent_by_node):
        handler_try = parent_by_node.get(id(handler))
        if not isinstance(handler_try, (ast.Try, ast.TryStar)):
            continue
        if any(
            other is not handler and parent_by_node.get(id(other)) is handler_try
            for other in call_handlers
        ):
            return True
    return False


def _exception_payload_names_at_call(
    call: ast.Call,
    parent_by_node: dict[int, ast.AST],
) -> set[str]:
    """Return lexical exception aliases visible before one diagnostic call.

    The model is source-ordered and branch-insensitive, except that handlers on the
    same ``try`` are mutually exclusive. Handler targets are ephemeral inputs to
    assignments and are never exported; names derived from them remain tainted.
    """
    scope = _lexical_scope(call, parent_by_node)
    call_handlers = _enclosing_exception_handlers(call, parent_by_node)
    tainted_names: set[str] = set()
    assignments: list[ast.AST] = []

    for node in ast.walk(scope):
        if (
            _assignment_parts(node) is not None
            and _lexical_scope(node, parent_by_node) is scope
            and _ends_before(node, call)
            and not _comes_from_mutually_exclusive_handler(
                node, call_handlers, parent_by_node
            )
        ):
            assignments.append(node)

    for assignment in sorted(
        assignments, key=lambda node: (node.lineno, node.col_offset)
    ):
        parts = _assignment_parts(assignment)
        assert parts is not None
        targets, value = parts
        active_handler_names = {
            handler.name
            for handler in _enclosing_exception_handlers(assignment, parent_by_node)
            if handler.name is not None
        }
        if _contains_raw_exception_payload(value, tainted_names | active_handler_names):
            tainted_names.update(targets - active_handler_names)

    tainted_names.update(
        handler.name for handler in call_handlers if handler.name is not None
    )
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

        tainted_names = _exception_payload_names_at_call(node, parent_by_node)
        if _contains_raw_exception_payload(node, tainted_names):
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


def test_exception_diagnostic_scanner_flags_outer_payload_in_nested_handler() -> None:
    source = dedent(
        """
        try:
            run_outer_database_operation()
        except Exception as outer_exc:
            try:
                run_inner_database_operation()
            except Exception:
                logger.error("Outer failure: {}", outer_exc)
        """
    )

    offenders = _scan_exception_diagnostic_offenders(source, filename="nested_outer.py")

    assert len(offenders) == 1
    assert offenders[0]["reasons"] == ("raw_exception_payload",)


def test_exception_diagnostic_scanner_flags_outer_alias_in_nested_handler() -> None:
    source = dedent(
        """
        try:
            run_outer_database_operation()
        except Exception as outer_exc:
            try:
                run_inner_database_operation()
            except Exception as inner_exc:
                outer_details = repr(outer_exc)
                rendered_outer = outer_details
                logger.error("Outer failure: {}", rendered_outer)
        """
    )

    offenders = _scan_exception_diagnostic_offenders(
        source, filename="nested_outer_alias.py"
    )

    assert len(offenders) == 1
    assert offenders[0]["reasons"] == ("raw_exception_payload",)


def test_exception_diagnostic_scanner_counts_one_call_with_outer_and_inner_payloads() -> (
    None
):
    source = dedent(
        """
        try:
            run_outer_database_operation()
        except Exception as outer_exc:
            try:
                run_inner_database_operation()
            except Exception as inner_exc:
                logger.error("Outer: {} inner: {}", outer_exc, inner_exc)
                logger.error("Outer only: {}", outer_exc)
        """
    )

    offenders = _scan_exception_diagnostic_offenders(source, filename="nested_both.py")

    assert len(offenders) == 2
    assert offenders[0]["reasons"] == ("raw_exception_payload",)
    assert "outer_exc, inner_exc" in offenders[0]["call"]
    assert offenders[1]["reasons"] == ("raw_exception_payload",)
    assert "Outer only" in offenders[1]["call"]


@pytest.mark.parametrize(
    "source",
    [
        pytest.param(
            dedent(
                """
                try:
                    run_first_database_operation()
                except Exception as first_exc:
                    first_details = repr(first_exc)
                try:
                    run_second_database_operation()
                except Exception:
                    logger.error("Earlier details: {}", first_details)
                """
            ),
            id="module",
        ),
        pytest.param(
            dedent(
                """
                def run_operations():
                    try:
                        run_first_database_operation()
                    except Exception as first_exc:
                        first_details = repr(first_exc)
                    try:
                        run_second_database_operation()
                    except Exception:
                        logger.error("Earlier details: {}", first_details)
                """
            ),
            id="function",
        ),
    ],
)
def test_exception_diagnostic_scanner_flags_alias_from_earlier_sequential_handler(
    source: str,
) -> None:
    offenders = _scan_exception_diagnostic_offenders(
        source, filename="sequential_handlers.py"
    )

    assert len(offenders) == 1
    assert offenders[0]["reasons"] == ("raw_exception_payload",)
    assert "first_details" in offenders[0]["call"]


def test_exception_diagnostic_scanner_flags_transitive_persisted_aliases() -> None:
    source = dedent(
        """
        try:
            run_first_database_operation()
        except Exception as first_exc:
            first_details = repr(first_exc)
        rendered_details = first_details
        try:
            run_second_database_operation()
        except Exception:
            final_details = rendered_details
            logger.error("Earlier details: {}", final_details)
        """
    )

    offenders = _scan_exception_diagnostic_offenders(
        source, filename="transitive_sequential.py"
    )

    assert len(offenders) == 1
    assert offenders[0]["reasons"] == ("raw_exception_payload",)
    assert "final_details" in offenders[0]["call"]


def test_exception_diagnostic_scanner_keeps_bound_exception_only_while_enclosing() -> (
    None
):
    source = dedent(
        """
        try:
            run_outer_database_operation()
        except Exception as outer_exc:
            try:
                run_inner_database_operation()
            except Exception:
                logger.error("Live outer exception: {}", outer_exc)
        try:
            run_later_database_operation()
        except Exception:
            logger.error("Cleared outer exception: {}", outer_exc)
        """
    )

    offenders = _scan_exception_diagnostic_offenders(
        source, filename="cleared_exception_target.py"
    )

    assert len(offenders) == 1
    assert "Live outer exception" in offenders[0]["call"]
    assert "Cleared outer exception" not in offenders[0]["call"]


def test_exception_diagnostic_scanner_isolates_persisted_aliases_by_lexical_scope() -> (
    None
):
    source = dedent(
        """
        def log_from_function():
            try:
                run_function_database_operation()
            except Exception as function_exc:
                function_details = repr(function_exc)
            logger.error("Function details: {}", function_details)

        class LogFromClass:
            try:
                run_class_database_operation()
            except Exception as class_exc:
                class_details = repr(class_exc)
            logger.error("Class details: {}", class_details)

        logger.error(
            "Outer details: {} {}",
            function_details,
            class_details,
        )
        """
    )

    offenders = _scan_exception_diagnostic_offenders(
        source, filename="persisted_scope_boundaries.py"
    )

    assert len(offenders) == 2
    assert "Function details" in offenders[0]["call"]
    assert "Class details" in offenders[1]["call"]
    assert all("Outer details" not in offender["call"] for offender in offenders)


def test_exception_diagnostic_scanner_does_not_flow_taint_backward() -> None:
    source = dedent(
        """
        logger.error("Before handler: {}", later_details)
        try:
            run_database_operation()
        except Exception as exc:
            logger.error("Before assignment: {}", later_details)
            later_details = repr(exc)
            logger.error("After assignment: {}", later_details)
        """
    )

    offenders = _scan_exception_diagnostic_offenders(source, filename="ordered.py")

    assert len(offenders) == 1
    assert "After assignment" in offenders[0]["call"]


def test_exception_diagnostic_scanner_does_not_share_aliases_between_same_try_handlers() -> (
    None
):
    source = dedent(
        """
        try:
            run_database_operation()
        except FirstError as first_exc:
            first_details = repr(first_exc)
        except SecondError:
            logger.error("Mutually exclusive details: {}", first_details)
        """
    )

    assert (
        _scan_exception_diagnostic_offenders(
            source, filename="mutually_exclusive_handlers.py"
        )
        == []
    )


def test_exception_diagnostic_scanner_reports_one_call_for_persisted_alias_set() -> (
    None
):
    source = dedent(
        """
        try:
            run_first_database_operation()
        except Exception as first_exc:
            first_details = repr(first_exc)
            rendered_details = first_details
        try:
            run_second_database_operation()
        except Exception:
            logger.error(
                "Earlier details: {} {}",
                first_details,
                rendered_details,
            )
        """
    )

    offenders = _scan_exception_diagnostic_offenders(
        source, filename="persisted_whole_set.py"
    )
    report = _format_exception_diagnostic_offenders(offenders)

    assert len(offenders) == 1
    assert offenders[0]["reasons"] == ("raw_exception_payload",)
    assert "1 offending diagnostic calls" in report
    assert "raw_exception_payload=1" in report
    assert report.count("line ") == 1


def test_exception_diagnostic_scanner_stops_taint_at_nested_scope_boundaries() -> None:
    source = dedent(
        """
        try:
            run_database_operation()
        except Exception as outer_exc:
            try:
                run_nested_database_operation()
            except Exception:
                logger.error("Nested failure: {}", outer_exc)

            def log_from_function():
                logger.error("Function failure: {}", outer_exc)

            class LogFromClass:
                logger.error("Class failure: {}", outer_exc)
        """
    )

    offenders = _scan_exception_diagnostic_offenders(source, filename="scopes.py")

    assert len(offenders) == 1
    assert "Nested failure" in offenders[0]["call"]


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
