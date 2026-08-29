"""Born-red contracts for the TASK-19864 diagnostic path scanner."""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from textwrap import dedent

import pytest

from scripts.check_persistent_diagnostic_inventory import (
    _PathState,
    _expression_path_state,
    _is_diagnostic_call,
    _logger_symbols,
    _safe_transform_contexts,
    _scope_contexts,
    _scope_path_aliases,
    render_diff,
    scan_path_diagnostic_candidates,
)


TASK_19864_OWNER_PATHS = (
    "tldw_chatbook/Utils/file_handlers.py",
    "tldw_chatbook/DB/ChaChaNotes_DB.py",
    "tldw_chatbook/UI/Screens/change_review_screen.py",
    "tldw_chatbook/Widgets/Console/console_conversation_inspector.py",
    "tldw_chatbook/Workspaces/git_workspace.py",
)
_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _has_path_expression_label(labels: list[str], expected: str) -> bool:
    if expected != "row.get(root)":
        return expected in labels
    return any(re.fullmatch(r"row\.get\((['\"])root\1\)", label) for label in labels)


@pytest.mark.parametrize(
    ("source", "filename", "method", "scope", "path_expression"),
    [
        pytest.param(
            'logger.info(f"Opened {file_path}")',
            "f_string.py",
            "info",
            "<module>",
            "file_path",
            id="f-string",
        ),
        pytest.param(
            'logger.info("Opened {}", file_path)',
            "loguru_positional.py",
            "info",
            "<module>",
            "file_path",
            id="loguru-positional",
        ),
        pytest.param(
            'logger.warning("Workspace {root}", root=workspace_root)',
            "loguru_keyword.py",
            "warning",
            "<module>",
            "workspace_root",
            id="loguru-keyword",
        ),
        pytest.param(
            'logger.debug("Opened %s" % file_path)',
            "percent_format.py",
            "debug",
            "<module>",
            "file_path",
            id="percent-format",
        ),
        pytest.param(
            'logger.error("Opened {}".format(file_path))',
            "dot_format.py",
            "error",
            "<module>",
            "file_path",
            id="dot-format",
        ),
        pytest.param(
            dedent(
                """
                def emit(output_path):
                    logger.error(
                        "Output path: {}",
                        output_path,
                    )
                """
            ),
            "multiline.py",
            "error",
            "emit",
            "output_path",
            id="multiline-call",
        ),
        pytest.param(
            'logger.info("Workspace {}", row.get("root"))',
            "mapping_root.py",
            "info",
            "<module>",
            "row.get(root)",
            id="mapping-root-key",
        ),
        pytest.param(
            dedent(
                """
                class Store:
                    def emit(self):
                        logger.error("Database {}", self.db_path_str)
                """
            ),
            "database_path.py",
            "error",
            "Store.emit",
            "self.db_path_str",
            id="database-path-attribute",
        ),
    ],
)
def test_path_shaped_diagnostic_inputs_are_candidates(
    source: str,
    filename: str,
    method: str,
    scope: str,
    path_expression: str,
) -> None:
    candidates = scan_path_diagnostic_candidates(source, filename=filename)

    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate["method"] == method
    assert candidate["scope"] == scope
    assert candidate["status"] == "legacy_unreviewed"
    assert candidate["call_digest"]
    assert _has_path_expression_label(candidate["path_expressions"], path_expression)


@pytest.mark.parametrize(
    ("source", "path_expression"),
    [
        pytest.param(
            dedent(
                """
                def emit(configured):
                    raw = configured or os.getcwd()
                    logger.info("Workspace {}", raw)
                """
            ),
            "raw",
            id="configured-or-cwd",
        ),
        pytest.param(
            dedent(
                """
                def emit(raw):
                    target = validate_path_simple(raw)
                    logger.info("Target {}", target)
                """
            ),
            "target",
            id="validated-path-alias",
        ),
        pytest.param(
            dedent(
                """
                def emit(raw):
                    target = Path(raw)
                    logger.info("Target {}", target)
                """
            ),
            "target",
            id="path-constructor-alias",
        ),
        pytest.param(
            dedent(
                """
                def emit():
                    target = Path.home()
                    logger.info("Target {}", target)
                """
            ),
            "target",
            id="path-home-alias",
        ),
        pytest.param(
            dedent(
                """
                def emit(raw):
                    target = raw.resolve()
                    logger.info("Target {}", target)
                """
            ),
            "target",
            id="resolved-path-alias",
        ),
    ],
)
def test_simple_assignment_taint_reaches_diagnostic_arguments(
    source: str, path_expression: str
) -> None:
    candidates = scan_path_diagnostic_candidates(source, filename="assignment.py")

    assert len(candidates) == 1
    assert path_expression in candidates[0]["path_expressions"]


def test_assignment_taint_reaches_a_scope_local_fixed_point() -> None:
    source = dedent(
        """
        def emit(raw):
            final = intermediate
            intermediate: str = Path(raw)
            logger.info("Target {}", final)
        """
    )

    candidates = scan_path_diagnostic_candidates(source, filename="fixed_point.py")

    assert len(candidates) == 1
    assert candidates[0]["path_expressions"] == ["final"]


def test_assignment_taint_does_not_bleed_between_lexical_scopes() -> None:
    source = dedent(
        """
        def discover():
            value = os.getcwd()

        def emit(value):
            logger.info("Value {}", value)
        """
    )

    assert scan_path_diagnostic_candidates(source, filename="scopes.py") == []


def test_moving_candidate_within_the_same_scope_preserves_identity() -> None:
    original = dedent(
        """
        def emit(output_path):
            logger.info("Output {}", output_path)
        """
    )
    moved = dedent(
        """
        def emit(output_path):
            unrelated = 1
            logger.info("Output {}", output_path)
            return unrelated
        """
    )

    assert scan_path_diagnostic_candidates(
        original, filename="movement.py"
    ) == scan_path_diagnostic_candidates(moved, filename="movement.py")


@pytest.mark.parametrize(
    "source",
    [
        pytest.param(
            'logger.info("Path ref {}", content_fingerprint(path))',
            id="content-fingerprint",
        ),
        pytest.param(
            'logger.info(f"Path {redact_user_paths(path)}")',
            id="path-redaction",
        ),
        pytest.param(
            'logger.info("File kind {}", path.suffix)',
            id="suffix",
        ),
        pytest.param(
            'logger.info("Path count {}", len(paths))',
            id="cardinality",
        ),
        pytest.param(
            'logger.error("Failure type {}", type(exc).__name__)',
            id="exception-type",
        ),
        pytest.param(
            'logger.info("Workspace {root}", root=content_fingerprint(user_path))',
            id="hinted-content-fingerprint",
        ),
        pytest.param(
            dedent(
                """
                from tldw_chatbook.Utils import log_sanitizer

                logger.info("Path ref {}", log_sanitizer.content_fingerprint(path))
                """
            ),
            id="approved-module-qualified-fingerprint",
        ),
        pytest.param(
            dedent(
                """
                import tldw_chatbook.Utils.log_sanitizer as sanitizer

                logger.info("Path {}", sanitizer.redact_user_paths(path))
                """
            ),
            id="approved-module-alias-redaction",
        ),
    ],
)
def test_safe_path_transforms_are_not_candidates(source: str) -> None:
    assert scan_path_diagnostic_candidates(source, filename="safe.py") == []


@pytest.mark.parametrize(
    ("source", "scope"),
    [
        pytest.param(
            dedent(
                """
                def emit(type):
                    logger.error("Path kind {}", type(user_path).__name__)
                """
            ),
            "emit",
            id="parameter",
        ),
        pytest.param(
            dedent(
                """
                def emit(transform):
                    type = transform
                    logger.error("Path kind {}", type(user_path).__name__)
                """
            ),
            "emit",
            id="local-assignment",
        ),
        pytest.param(
            dedent(
                """
                def outer(type):
                    def emit():
                        logger.error("Path kind {}", type(user_path).__name__)
                """
            ),
            "outer.emit",
            id="enclosing-scope",
        ),
        pytest.param(
            dedent(
                """
                type = transform

                def emit():
                    logger.error("Path kind {}", type(user_path).__name__)
                """
            ),
            "emit",
            id="module-scope",
        ),
    ],
)
def test_shadowed_type_builtin_does_not_sanitize_path_argument(
    source: str, scope: str
) -> None:
    candidates = scan_path_diagnostic_candidates(
        source, filename="shadowed_type_builtin.py"
    )

    assert len(candidates) == 1
    assert candidates[0]["scope"] == scope
    assert candidates[0]["path_expressions"] == ["type(user_path).__name__"]


def test_module_sanitizer_alias_is_safe_in_an_unshadowed_function() -> None:
    source = dedent(
        """
        import tldw_chatbook.Utils.log_sanitizer as sanitizer

        def emit():
            logger.info("Path {}", sanitizer.content_fingerprint(user_path))
        """
    )

    assert scan_path_diagnostic_candidates(source, filename="module_alias.py") == []


def test_module_sanitizer_alias_shadowed_by_a_parameter_is_a_candidate() -> None:
    source = dedent(
        """
        import tldw_chatbook.Utils.log_sanitizer as sanitizer

        def emit(sanitizer):
            logger.info("Path {}", sanitizer.content_fingerprint(user_path))
        """
    )

    candidates = scan_path_diagnostic_candidates(
        source, filename="shadowed_module_alias.py"
    )

    assert len(candidates) == 1
    assert candidates[0]["scope"] == "emit"
    assert candidates[0]["path_expressions"] == [
        "sanitizer.content_fingerprint(user_path)"
    ]


def test_function_local_sanitizer_alias_does_not_bless_a_sibling_scope() -> None:
    source = dedent(
        """
        def safe_emit():
            import tldw_chatbook.Utils.log_sanitizer as sanitizer
            logger.info("Path {}", sanitizer.content_fingerprint(user_path))

        def unsafe_emit(sanitizer):
            logger.info("Path {}", sanitizer.content_fingerprint(user_path))
        """
    )

    candidates = scan_path_diagnostic_candidates(
        source, filename="function_local_alias.py"
    )

    assert len(candidates) == 1
    assert candidates[0]["scope"] == "unsafe_emit"
    assert candidates[0]["path_expressions"] == [
        "sanitizer.content_fingerprint(user_path)"
    ]


@pytest.mark.parametrize(
    "source",
    [
        pytest.param(
            dedent(
                """
                def emit(content_fingerprint):
                    logger.info("Path {}", content_fingerprint(user_path))
                """
            ),
            id="parameter",
        ),
        pytest.param(
            dedent(
                """
                def emit(transform):
                    content_fingerprint = transform
                    logger.info("Path {}", content_fingerprint(user_path))
                """
            ),
            id="assignment",
        ),
    ],
)
def test_shadowed_unqualified_safe_transform_is_a_candidate(source: str) -> None:
    candidates = scan_path_diagnostic_candidates(
        source, filename="shadowed_unqualified.py"
    )

    assert len(candidates) == 1
    assert candidates[0]["path_expressions"] == ["content_fingerprint(user_path)"]


@pytest.mark.parametrize(
    "source",
    [
        pytest.param(
            dedent(
                """
                def outer(content_fingerprint):
                    def emit():
                        logger.info("Path {}", content_fingerprint(user_path))
                """
            ),
            id="outer-parameter",
        ),
        pytest.param(
            dedent(
                """
                def outer(transform):
                    content_fingerprint = transform

                    def emit():
                        logger.info("Path {}", content_fingerprint(user_path))
                """
            ),
            id="outer-assignment",
        ),
        pytest.param(
            dedent(
                """
                def outer():
                    def content_fingerprint(value):
                        return value

                    def emit():
                        logger.info("Path {}", content_fingerprint(user_path))
                """
            ),
            id="outer-function-definition",
        ),
        pytest.param(
            dedent(
                """
                def outer():
                    class content_fingerprint:
                        pass

                    def emit():
                        logger.info("Path {}", content_fingerprint(user_path))
                """
            ),
            id="outer-class-definition",
        ),
    ],
)
def test_enclosing_function_shadow_is_a_candidate_in_nested_closure(
    source: str,
) -> None:
    candidates = scan_path_diagnostic_candidates(
        source, filename="enclosing_function_shadow.py"
    )

    assert len(candidates) == 1
    assert candidates[0]["scope"] == "outer.emit"
    assert candidates[0]["path_expressions"] == ["content_fingerprint(user_path)"]


def test_enclosing_function_shadow_does_not_leak_to_sibling_scope() -> None:
    source = dedent(
        """
        def shadowing_outer(content_fingerprint):
            def unsafe_emit():
                logger.info("Path {}", content_fingerprint(user_path))

        def clean_outer():
            def safe_emit():
                logger.info("Path {}", content_fingerprint(user_path))
        """
    )

    candidates = scan_path_diagnostic_candidates(
        source, filename="enclosing_function_siblings.py"
    )

    assert len(candidates) == 1
    assert candidates[0]["scope"] == "shadowing_outer.unsafe_emit"
    assert candidates[0]["path_expressions"] == ["content_fingerprint(user_path)"]


@pytest.mark.parametrize(
    "definition_template",
    [
        pytest.param(
            "def {name}(value):\n    return value",
            id="function-definition",
        ),
        pytest.param(
            "async def {name}(value):\n    return value",
            id="async-function-definition",
        ),
        pytest.param(
            "class {name}:\n    pass",
            id="class-definition",
        ),
    ],
)
@pytest.mark.parametrize(
    "transform_name",
    ["content_fingerprint", "redact_user_paths"],
)
def test_module_definition_shadows_unqualified_safe_transform(
    definition_template: str,
    transform_name: str,
) -> None:
    definition = definition_template.format(name=transform_name)
    source = f'{definition}\nlogger.info("Path {{}}", {transform_name}(user_path))\n'

    candidates = scan_path_diagnostic_candidates(
        source, filename="module_definition_shadow.py"
    )

    assert len(candidates) == 1
    assert candidates[0]["scope"] == "<module>"
    assert candidates[0]["path_expressions"] == [f"{transform_name}(user_path)"]


@pytest.mark.parametrize(
    "definition_template",
    [
        pytest.param(
            "    def {name}(value):\n        return value",
            id="function-definition",
        ),
        pytest.param(
            "    async def {name}(value):\n        return value",
            id="async-function-definition",
        ),
        pytest.param(
            "    class {name}:\n        pass",
            id="class-definition",
        ),
    ],
)
@pytest.mark.parametrize(
    "transform_name",
    ["content_fingerprint", "redact_user_paths"],
)
def test_nested_definition_shadows_safe_transform_in_enclosing_scope(
    definition_template: str,
    transform_name: str,
) -> None:
    definition = definition_template.format(name=transform_name)
    source = (
        f"def emit():\n{definition}\n"
        f'    logger.info("Path {{}}", {transform_name}(user_path))\n'
    )

    candidates = scan_path_diagnostic_candidates(
        source, filename="nested_definition_shadow.py"
    )

    assert len(candidates) == 1
    assert candidates[0]["scope"] == "emit"
    assert candidates[0]["path_expressions"] == [f"{transform_name}(user_path)"]


def test_shadowed_len_is_not_a_safe_cardinality_transform() -> None:
    source = dedent(
        """
        def emit(len):
            logger.info("Path count {path}", path=len(paths))
        """
    )

    candidates = scan_path_diagnostic_candidates(source, filename="shadowed_len.py")

    assert len(candidates) == 1
    assert candidates[0]["path_expressions"] == ["len(paths)"]


@pytest.mark.parametrize(
    "method",
    ["content_fingerprint", "redact_user_paths"],
)
def test_object_methods_named_like_safe_transforms_remain_candidates(
    method: str,
) -> None:
    source = f'logger.info("Path {{}}", passthrough.{method}(user_path))'

    candidates = scan_path_diagnostic_candidates(source, filename="lookalike.py")

    assert len(candidates) == 1
    assert candidates[0]["path_expressions"] == [f"passthrough.{method}(user_path)"]


def test_loguru_path_shaped_keyword_taints_a_generic_value() -> None:
    candidates = scan_path_diagnostic_candidates(
        'logger.info("Workspace {root}", root=value)',
        filename="loguru_keyword_hint.py",
    )

    assert len(candidates) == 1
    assert candidates[0]["path_expressions"] == ["root=value"]


def test_path_shaped_hint_preserves_recursively_proven_safe_value() -> None:
    source = 'logger.info("Workspace {root}", root=str(content_fingerprint(user_path)))'

    assert scan_path_diagnostic_candidates(source, filename="safe_wrapper.py") == []


@pytest.mark.parametrize(
    ("argument", "expression"),
    [
        pytest.param("None", "load(None)", id="none"),
        pytest.param("'workspace'", "load('workspace')", id="string"),
    ],
)
def test_path_shaped_hint_keeps_unknown_call_result_as_a_candidate(
    argument: str,
    expression: str,
) -> None:
    source = f'logger.info("Workspace {{root}}", root=load({argument}))'

    candidates = scan_path_diagnostic_candidates(source, filename="unknown_call.py")

    assert len(candidates) == 1
    assert candidates[0]["path_expressions"] == [f"root={expression}"]


def test_shadowed_str_does_not_prove_a_safe_value_remains_safe() -> None:
    source = dedent(
        """
        def emit(str):
            logger.info(
                "Workspace {root}",
                root=str(content_fingerprint(user_path)),
            )
        """
    )

    candidates = scan_path_diagnostic_candidates(source, filename="shadowed_str.py")

    assert len(candidates) == 1
    assert candidates[0]["path_expressions"] == [
        "root=str(content_fingerprint(user_path))"
    ]


def test_safe_receiver_method_with_safe_arguments_remains_safe() -> None:
    source = dedent(
        """
        logger.info(
            "Workspace {root}",
            root=content_fingerprint(user_path).removeprefix("sha256:"),
        )
        """
    )

    assert scan_path_diagnostic_candidates(source, filename="safe_method.py") == []


def test_tainted_child_dominates_an_ordinary_wrapper() -> None:
    candidates = scan_path_diagnostic_candidates(
        'logger.info("Workspace {root}", root=str(user_path))',
        filename="tainted_wrapper.py",
    )

    assert len(candidates) == 1
    assert candidates[0]["path_expressions"] == ["str(user_path)"]


def test_tainted_descendant_dominates_a_comprehension_wrapper() -> None:
    source = dedent(
        """
        def emit(raw):
            source_path = Path(raw)
            wrapped = [str(value) for value in source_path]
            logger.info("Wrapped {}", wrapped)
        """
    )

    candidates = scan_path_diagnostic_candidates(
        source, filename="tainted_comprehension.py"
    )

    assert len(candidates) == 1
    assert candidates[0]["path_expressions"] == ["wrapped"]


@pytest.mark.parametrize(
    "source",
    [
        'logger.info("Workspace {root}".format(root=value))',
        'logger.info("Workspace {}", "Workspace {root}".format(root=value))',
    ],
)
def test_str_format_path_shaped_keyword_taints_a_generic_value(source: str) -> None:
    candidates = scan_path_diagnostic_candidates(
        source, filename="str_format_keyword_hint.py"
    )

    assert len(candidates) == 1
    assert candidates[0]["path_expressions"] == ["root=value"]


def test_percent_mapping_path_shaped_key_taints_a_generic_value() -> None:
    candidates = scan_path_diagnostic_candidates(
        'logger.info("Workspace %(root)s" % {"root": value})',
        filename="percent_mapping_hint.py",
    )

    assert len(candidates) == 1
    assert candidates[0]["path_expressions"] == ["root=value"]


def test_path_like_substrings_without_bounded_path_identifiers_are_ignored() -> None:
    source = dedent(
        """
        logger.info("Root cause {}", root_cause)
        logger.info("Directory count {}", directory_count)
        """
    )

    assert scan_path_diagnostic_candidates(source, filename="negative_names.py") == []


def _inventory_with_path_candidates(
    rows: list[dict[str, object]], *, candidate_count: int
) -> dict[str, object]:
    return {
        "schema_version": 3,
        "scope": "tldw_chatbook/**/*.py",
        "classification_rules": {},
        "path_privacy_rules": {},
        "reviewed_exclusions": [],
        "summary": {"path_privacy_candidate_calls": candidate_count},
        "owners": [],
        "persistent_sink_topology": [],
        "path_privacy_candidates": rows,
    }


def _candidate_detail(candidate: dict[str, object]) -> str:
    labels = ", ".join(candidate["path_expressions"])
    return (
        f"{candidate['scope']}: {candidate['method']} "
        f"({candidate['call_digest']}) paths=[{labels}] "
        f"status={candidate['status']}"
    )


def _lexical_function_has_path_state(
    scope: ast.AST,
    assignments: dict[int, list[tuple[ast.AST, ast.AST]]],
    aliases: set[str],
    log_sanitizer_qualifiers: frozenset[tuple[str, ...]],
    shadowed_names: frozenset[str],
) -> bool:
    """Return whether one function frame can retain a raw path value."""
    if not isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return False

    arguments = scope.args
    parameters = [
        *arguments.posonlyargs,
        *arguments.args,
        *arguments.kwonlyargs,
    ]
    if arguments.vararg is not None:
        parameters.append(arguments.vararg)
    if arguments.kwarg is not None:
        parameters.append(arguments.kwarg)
    if any(
        _expression_path_state(
            ast.Name(id=parameter.arg, ctx=ast.Load()),
            aliases,
            log_sanitizer_qualifiers,
            shadowed_names,
        )
        is _PathState.TAINTED
        for parameter in parameters
    ):
        return True

    for target, value in assignments.get(id(scope), []):
        value_state = _expression_path_state(
            value,
            aliases,
            log_sanitizer_qualifiers,
            shadowed_names,
        )
        if value_state is _PathState.TAINTED:
            return True
        if value_state is _PathState.PROVEN_SAFE:
            continue
        if (
            _expression_path_state(
                target,
                aliases,
                log_sanitizer_qualifiers,
                shadowed_names,
            )
            is _PathState.TAINTED
        ):
            return True

    return False


def _traceback_capture_calls(source: str, *, filename: str) -> list[dict[str, object]]:
    """Return traceback captures in path-bearing exception regions."""
    tree = ast.parse(source, filename=filename)
    logger_symbols = _logger_symbols(tree)
    (
        scope_names,
        lexical_scopes,
        assignments,
        definition_parent_scopes,
    ) = _scope_contexts(tree)
    safe_transform_contexts = _safe_transform_contexts(
        tree, lexical_scopes, definition_parent_scopes
    )
    parent_by_node = {
        id(child): parent
        for parent in ast.walk(tree)
        for child in ast.iter_child_nodes(parent)
    }
    capture_calls: list[tuple[ast.Call, str]] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not _is_diagnostic_call(
            node, logger_symbols
        ):
            continue

        method = node.func.attr
        capture: str | None = None
        if method == "exception":
            capture = "logger.exception"
        elif isinstance(node.func.value, ast.Call):
            options_call = node.func.value
            if (
                isinstance(options_call.func, ast.Attribute)
                and options_call.func.attr == "opt"
            ):
                exception_option = next(
                    (
                        keyword.value
                        for keyword in options_call.keywords
                        if keyword.arg == "exception"
                    ),
                    None,
                )
                disabled = isinstance(
                    exception_option, ast.Constant
                ) and exception_option.value in (False, None)
                if exception_option is not None and not disabled:
                    capture = f"logger.opt(exception={ast.unparse(exception_option)})"

        if capture is not None:
            capture_calls.append((node, capture))

    active_scope_ids = {
        id(lexical_scopes[id(node)]) for node, _capture in capture_calls
    }
    aliases_by_scope = _scope_path_aliases(
        assignments,
        active_scope_ids,
        safe_transform_contexts,
    )
    captures: list[dict[str, object]] = []

    for node, capture in capture_calls:
        method = node.func.attr
        ancestor = parent_by_node.get(id(node))
        handler: ast.ExceptHandler | None = None
        operation: ast.Try | ast.TryStar | None = None
        while ancestor is not None:
            if isinstance(ancestor, ast.ExceptHandler):
                candidate_operation = parent_by_node.get(id(ancestor))
                if isinstance(candidate_operation, (ast.Try, ast.TryStar)):
                    handler = ancestor
                    operation = candidate_operation
                    break
            ancestor = parent_by_node.get(id(ancestor))
        if handler is None or operation is None:
            continue

        scope_id = id(lexical_scopes[id(node)])
        log_sanitizer_qualifiers, shadowed_names = safe_transform_contexts[scope_id]
        aliases = aliases_by_scope.get(scope_id, set())
        region = [*operation.body, *handler.body]
        region_has_path_state = any(
            _expression_path_state(
                statement,
                aliases,
                log_sanitizer_qualifiers,
                shadowed_names,
            )
            is _PathState.TAINTED
            for statement in region
        )
        function_has_path_state = _lexical_function_has_path_state(
            lexical_scopes[id(node)],
            assignments,
            aliases,
            log_sanitizer_qualifiers,
            shadowed_names,
        )
        if not region_has_path_state and not function_has_path_state:
            continue

        captures.append(
            {
                "line": node.lineno,
                "column": node.col_offset + 1,
                "scope": scope_names.get(id(node)) or "<module>",
                "method": method,
                "capture": capture,
            }
        )

    return sorted(
        captures,
        key=lambda entry: (entry["line"], entry["column"], entry["method"]),
    )


def test_traceback_capture_flags_path_bearing_loguru_opt() -> None:
    source = dedent(
        """
        def emit(root):
            try:
                provider.current_status(root)
            except Exception:
                logger.opt(exception=True).warning(
                    "status failed root_sha256={}", content_fingerprint(root)
                )
        """
    )

    captures = _traceback_capture_calls(source, filename="path_opt.py")

    assert [capture["capture"] for capture in captures] == [
        "logger.opt(exception=True)"
    ]


def test_traceback_capture_flags_path_bearing_logger_exception() -> None:
    source = dedent(
        """
        def emit(database_path):
            try:
                connect_private_sqlite(database_path)
            except Exception:
                logger.exception("database connection failed")
        """
    )

    captures = _traceback_capture_calls(source, filename="path_exception.py")

    assert [capture["capture"] for capture in captures] == ["logger.exception"]


def test_traceback_capture_flags_constructor_path_state_for_delegated_call() -> None:
    source = dedent(
        """
        class Store:
            def __init__(self, db_path):
                self.db_path_str = str(db_path)
                self._db_diagnostic_ref = content_fingerprint(self.db_path_str)
                try:
                    self._initialize_schema()
                except Exception:
                    logger.opt(exception=True).critical(
                        "database initialization failed db_sha256={}",
                        self._db_diagnostic_ref,
                    )
        """
    )

    captures = _traceback_capture_calls(source, filename="constructor.py")

    assert [capture["capture"] for capture in captures] == [
        "logger.opt(exception=True)"
    ]


def test_traceback_capture_flags_retained_path_parameter_for_delegated_call() -> None:
    source = dedent(
        """
        def emit(workspace_root):
            retained = workspace_root
            try:
                run_check()
            except Exception:
                logger.exception("check failed")
        """
    )

    captures = _traceback_capture_calls(source, filename="retained_parameter.py")

    assert [capture["capture"] for capture in captures] == ["logger.exception"]


def test_traceback_capture_ignores_path_state_in_nested_child_function() -> None:
    source = dedent(
        """
        def emit():
            def child(input_path):
                retained = input_path
                return retained

            try:
                run_check()
            except Exception:
                logger.exception("check failed")
        """
    )

    assert _traceback_capture_calls(source, filename="nested_child.py") == []


def test_traceback_capture_ignores_path_state_in_nested_child_class() -> None:
    source = dedent(
        """
        def emit():
            class Child:
                output_path = Path.home()

            try:
                run_check()
            except Exception:
                logger.exception("check failed")
        """
    )

    assert _traceback_capture_calls(source, filename="nested_class.py") == []


def test_traceback_capture_preserves_safe_local_path_transform() -> None:
    source = dedent(
        """
        def emit(value):
            workspace_root = content_fingerprint(value)
            try:
                run_check()
            except Exception:
                logger.exception("check failed")
        """
    )

    assert _traceback_capture_calls(source, filename="safe_local.py") == []


@pytest.mark.parametrize("exception_option", ["False", "None"])
def test_traceback_capture_ignores_disabled_loguru_options(
    exception_option: str,
) -> None:
    source = dedent(
        f"""
        def emit(root):
            try:
                provider.current_status(root)
            except Exception:
                logger.opt(exception={exception_option}).warning("status failed")
        """
    )

    assert _traceback_capture_calls(source, filename="disabled.py") == []


def test_traceback_capture_ignores_unrelated_url_only_failure_region() -> None:
    source = dedent(
        """
        def emit(url):
            retained_url = url
            try:
                app.open_url(url)
            except Exception:
                logger.opt(exception=True).warning("open_url failed")
        """
    )

    assert _traceback_capture_calls(source, filename="unrelated.py") == []


def test_traceback_capture_conservatively_flags_dynamic_enabled_option() -> None:
    source = dedent(
        """
        def emit(root, capture_exception):
            try:
                provider.pr_url(root)
            except Exception:
                logger.opt(exception=capture_exception).warning("PR link failed")
        """
    )

    captures = _traceback_capture_calls(source, filename="dynamic.py")

    assert [capture["capture"] for capture in captures] == [
        "logger.opt(exception=capture_exception)"
    ]


def test_traceback_capture_preserves_complete_path_bearing_evidence() -> None:
    source = dedent(
        """
        def commit(root):
            try:
                provider.current_status(root)
            except Exception:
                logger.opt(exception=True).warning("commit preflight failed")

        def connect(db_path):
            try:
                connect_private_sqlite(db_path)
            except Exception:
                logger.exception("database connection failed")

        def browse(url):
            try:
                app.open_url(url)
            except Exception:
                logger.exception("open_url failed")
        """
    )

    captures = _traceback_capture_calls(source, filename="complete.py")

    assert [capture["capture"] for capture in captures] == [
        "logger.opt(exception=True)",
        "logger.exception",
    ]


def test_path_candidate_report_preserves_all_files_and_duplicate_findings() -> None:
    duplicate_source = dedent(
        """
        logger.warning("Workspace root {}", workspace_root)
        logger.warning("Workspace root {}", workspace_root)
        """
    )
    other_source = 'logger.error("Database path {}", database_path)'
    duplicate_candidates = scan_path_diagnostic_candidates(
        duplicate_source, filename="alpha.py"
    )
    other_candidates = scan_path_diagnostic_candidates(other_source, filename="beta.py")

    assert len(duplicate_candidates) == 2
    assert len(other_candidates) == 1
    duplicate_digest = duplicate_candidates[0]["call_digest"]
    assert duplicate_candidates[1]["call_digest"] == duplicate_digest
    other_digest = other_candidates[0]["call_digest"]

    committed = _inventory_with_path_candidates([], candidate_count=0)
    rebuilt = _inventory_with_path_candidates(
        [
            {"path": "alpha.py", "candidates": duplicate_candidates},
            {"path": "beta.py", "candidates": other_candidates},
        ],
        candidate_count=3,
    )
    report = render_diff(json.dumps(committed), rebuilt)

    assert "alpha.py" in report
    assert "beta.py" in report
    assert duplicate_digest in report
    assert other_digest in report
    assert "x2" in report


def test_path_candidate_report_counts_additions_removals_and_changes() -> None:
    removed = scan_path_diagnostic_candidates(
        'logger.warning("Removed root {}", removed_root)', filename="removed.py"
    )[0]
    added = scan_path_diagnostic_candidates(
        'logger.error("Added path {}", added_path)', filename="added.py"
    )[0]
    changed_old = scan_path_diagnostic_candidates(
        'logger.info("Source directory {}", source_directory)',
        filename="changed.py",
    )[0]
    changed_new = scan_path_diagnostic_candidates(
        'logger.info("Destination folder {}", destination_folder)',
        filename="changed.py",
    )[0]
    multiplicity = scan_path_diagnostic_candidates(
        'logger.debug("Workspace roots {}", workspace_roots)',
        filename="multiplicity.py",
    )[0]
    committed = _inventory_with_path_candidates(
        [
            {"path": "removed.py", "candidates": [removed, removed]},
            {"path": "changed.py", "candidates": [changed_old]},
            {"path": "multiplicity.py", "candidates": [multiplicity, multiplicity]},
        ],
        candidate_count=5,
    )
    rebuilt = _inventory_with_path_candidates(
        [
            {"path": "added.py", "candidates": [added, added]},
            {"path": "changed.py", "candidates": [changed_new]},
            {"path": "multiplicity.py", "candidates": [multiplicity]},
        ],
        candidate_count=4,
    )

    report = render_diff(json.dumps(committed), rebuilt)

    assert "only in committed (candidate file removed): removed.py" in report
    assert f"      - {_candidate_detail(removed)} x2" in report
    assert "only in rebuild (NEW candidate file): added.py" in report
    assert f"      + {_candidate_detail(added)} x2" in report
    assert "changed candidates: changed.py (1 -> 1 calls)" in report
    assert f"      - {_candidate_detail(changed_old)} x1" in report
    assert f"      + {_candidate_detail(changed_new)} x1" in report
    assert "changed candidates: multiplicity.py (2 -> 1 calls)" in report
    assert f"      ~ {_candidate_detail(multiplicity)}: x2 -> x1 (-1)" in report
    assert "unresolved" in report
    assert "not approved" in report


def test_path_privacy_rules_are_inventory_metadata() -> None:
    committed = _inventory_with_path_candidates([], candidate_count=0)
    rebuilt = _inventory_with_path_candidates([], candidate_count=0)
    committed["path_privacy_rules"] = {"candidate_status": "old"}
    rebuilt["path_privacy_rules"] = {"candidate_status": "legacy_unreviewed"}

    report = render_diff(json.dumps(committed), rebuilt)

    assert "path_privacy_rules" in report
    assert "legacy_unreviewed" in report


def test_task_19864_owner_files_have_no_raw_path_diagnostics() -> None:
    """Every owner must reach zero path inputs and implicit traceback capture."""
    evidence_by_owner: dict[str, dict[str, list[dict[str, object]]]] = {}
    for relative_path in TASK_19864_OWNER_PATHS:
        source = (_REPOSITORY_ROOT / relative_path).read_text(encoding="utf-8")
        path_candidates = scan_path_diagnostic_candidates(
            source,
            filename=relative_path,
        )
        traceback_captures = _traceback_capture_calls(
            source,
            filename=relative_path,
        )
        if path_candidates or traceback_captures:
            evidence_by_owner[relative_path] = {
                "path_diagnostic_candidates": path_candidates,
                "traceback_captures": traceback_captures,
            }

    if evidence_by_owner:
        pytest.fail(
            "TASK-19864 owner path diagnostics or traceback capture remain; "
            "complete evidence sets:\n"
            + json.dumps(evidence_by_owner, indent=2, sort_keys=True),
            pytrace=False,
        )
