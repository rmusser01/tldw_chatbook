"""Every keyword the Study handlers pass to a scope service is one the REAL
service accepts (TASK-34000.6, S-05).

``flashcards_handler.py`` spread ``**self._scope_arguments()`` (``scope_type``,
``workspace_id``) into ``StudyScopeService.list_flashcards`` and
``create_flashcard``, neither of which takes them. Every test fake accepted
the keywords, so the ``TypeError`` -- an unhandled exception that exited the
app the moment a deck was selected -- lived from the parity merge
(54b9ca6a17, 2026-04-20) to the 2026-10-02 UX review.

This guard DERIVES the contract instead of listing it: an AST walk over both
handlers collects every ``service.<method>(...)`` call -- explicit keywords,
positional arguments, and the keys of every ``**`` spread, resolved through
the handler's own helper methods (``_scope_arguments``,
``_workspace_create_arguments``, ``_review_session_teardown_request``) -- and
checks each against ``inspect.signature`` of the real ``StudyScopeService`` /
``QuizScopeService`` method. A spread whose keys cannot be resolved fails the
guard too (fail closed), so a new helper cannot slip an unknown keyword past
it. ``test_guard_fails_on_a_reintroduced_scope_spread`` is the negative
control: the original bad call, re-inserted into a scratch copy, is reported
by method, keyword and line.
"""

from __future__ import annotations

import ast
import importlib
import inspect
from dataclasses import dataclass, field
from pathlib import Path

import pytest

_PACKAGE = Path(__file__).resolve().parents[2] / "tldw_chatbook"

#: (handler module path, service module, service class, local receiver name)
HANDLER_SERVICE_PAIRS: tuple[tuple[Path, str, str, str], ...] = (
    (
        _PACKAGE / "UI" / "Study_Modules" / "flashcards_handler.py",
        "tldw_chatbook.Study_Interop.study_scope_service",
        "StudyScopeService",
        "service",
    ),
    (
        _PACKAGE / "UI" / "Study_Modules" / "quizzes_handler.py",
        "tldw_chatbook.Study_Interop.quiz_scope_service",
        "QuizScopeService",
        "service",
    ),
)


@dataclass
class ServiceCall:
    """One ``service.<method>(...)`` call site and everything it passes."""

    line: int
    method: str
    keywords: set[str] = field(default_factory=set)
    positional: int = 0
    spreads: list[str] = field(default_factory=list)
    unresolved: list[str] = field(default_factory=list)


def _class_methods(
    tree: ast.Module,
) -> dict[str, ast.FunctionDef | ast.AsyncFunctionDef]:
    methods: dict[str, ast.FunctionDef | ast.AsyncFunctionDef] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    methods[item.name] = item
    return methods


def _is_self_call(node: ast.AST) -> str | None:
    """``self.<helper>()`` -> ``<helper>``."""
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "self"
        and not node.args
        and not node.keywords
    ):
        return node.func.attr
    return None


def _is_self_attribute(node: ast.AST) -> str | None:
    if (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "self"
    ):
        return node.attr
    return None


class _SpreadResolver:
    """Resolve the keys a ``**expr`` contributes, through the handler's helpers."""

    def __init__(self, tree: ast.Module) -> None:
        self.tree = tree
        self.methods = _class_methods(tree)

    def keys_of(
        self, expr: ast.AST, scope: ast.AST, visiting: frozenset[str] = frozenset()
    ) -> tuple[set[str], list[str]]:
        """Return ``(keys, unresolved descriptions)`` for ``expr``."""
        keys: set[str] = set()
        unresolved: list[str] = []
        if isinstance(expr, ast.Dict):
            for key, value in zip(expr.keys, expr.values):
                if key is None:  # ``**value`` inside the literal
                    inner, inner_unresolved = self.keys_of(value, scope, visiting)
                    keys |= inner
                    unresolved += inner_unresolved
                elif isinstance(key, ast.Constant) and isinstance(key.value, str):
                    keys.add(key.value)
                else:
                    unresolved.append(
                        f"non-literal dict key at line {getattr(key, 'lineno', '?')}"
                    )
            return keys, unresolved
        if isinstance(expr, ast.Constant) and expr.value is None:
            return keys, unresolved
        helper = _is_self_call(expr)
        if helper is not None:
            return self._keys_of_helper(helper, visiting)
        if isinstance(expr, ast.Name):
            return self._keys_of_local_name(expr.id, scope, visiting)
        if (
            isinstance(expr, ast.Call)
            and isinstance(expr.func, ast.Name)
            and expr.func.id == "dict"
            and len(expr.args) == 1
            and not expr.keywords
        ):
            attribute = _is_self_attribute(expr.args[0])
            if attribute is not None:
                return self._keys_of_self_attribute(attribute, visiting)
            return self.keys_of(expr.args[0], scope, visiting)
        unresolved.append(f"{ast.unparse(expr)} at line {getattr(expr, 'lineno', '?')}")
        return keys, unresolved

    def _keys_of_helper(
        self, helper: str, visiting: frozenset[str]
    ) -> tuple[set[str], list[str]]:
        if helper in visiting:
            return set(), []  # re-entrant (a helper feeding itself back)
        function = self.methods.get(helper)
        if function is None:
            return set(), [f"helper self.{helper}() is not defined on the handler"]
        keys: set[str] = set()
        unresolved: list[str] = []
        for node in ast.walk(function):
            if isinstance(node, ast.Return) and node.value is not None:
                inner, inner_unresolved = self.keys_of(
                    node.value, function, visiting | {helper}
                )
                keys |= inner
                unresolved += inner_unresolved
        return keys, unresolved

    def _keys_of_local_name(
        self, name: str, scope: ast.AST, visiting: frozenset[str]
    ) -> tuple[set[str], list[str]]:
        keys: set[str] = set()
        unresolved: list[str] = []
        found = False
        for node in ast.walk(scope):
            if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == name
                for target in node.targets
            ):
                found = True
                inner, inner_unresolved = self.keys_of(node.value, scope, visiting)
                keys |= inner
                unresolved += inner_unresolved
        if not found:
            unresolved.append(f"local name {name!r} has no resolvable assignment")
        return keys, unresolved

    def _keys_of_self_attribute(
        self, attribute: str, visiting: frozenset[str]
    ) -> tuple[set[str], list[str]]:
        keys: set[str] = set()
        unresolved: list[str] = []
        found = False
        for function in self.methods.values():
            for node in ast.walk(function):
                if isinstance(node, ast.Assign) and any(
                    _is_self_attribute(target) == attribute for target in node.targets
                ):
                    found = True
                    inner, inner_unresolved = self.keys_of(
                        node.value, function, visiting
                    )
                    keys |= inner
                    unresolved += inner_unresolved
        if not found:
            unresolved.append(f"self.{attribute} is never assigned on the handler")
        return keys, unresolved


def collect_service_calls(source: str, receiver: str = "service") -> list[ServiceCall]:
    """Every ``<receiver>.<method>(...)`` call in ``source`` with its arguments."""
    tree = ast.parse(source)
    resolver = _SpreadResolver(tree)
    calls: list[ServiceCall] = []
    for function in _class_methods(tree).values():
        for node in ast.walk(function):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == receiver
            ):
                continue
            call = ServiceCall(line=node.lineno, method=node.func.attr)
            call.positional = len(node.args)
            for keyword in node.keywords:
                if keyword.arg is not None:
                    call.keywords.add(keyword.arg)
                    continue
                call.spreads.append(ast.unparse(keyword.value))
                keys, unresolved = resolver.keys_of(keyword.value, function)
                call.keywords |= keys
                call.unresolved += unresolved
            calls.append(call)
    return calls


def check_handler_against_service(
    handler_path: Path, service_class: type, receiver: str = "service"
) -> list[str]:
    """Return one violation line per mismatch (empty means the contract holds)."""
    source = handler_path.read_text(encoding="utf-8")
    violations: list[str] = []
    calls = collect_service_calls(source, receiver)
    if not calls:
        violations.append(f"{handler_path.name}: found no {receiver}.<method>() calls")
    for call in calls:
        where = f"{handler_path.name}:{call.line} {receiver}.{call.method}("
        target = getattr(service_class, call.method, None)
        if target is None:
            violations.append(f"{where}): {service_class.__name__} has no such method")
            continue
        signature = inspect.signature(target)
        parameters = signature.parameters
        accepts_any = any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        )
        positional_slots = [
            name
            for name, parameter in parameters.items()
            if name != "self"
            and parameter.kind
            in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
        ]
        if call.positional > len(positional_slots):
            violations.append(
                f"{where}): passes {call.positional} positional argument(s); the real "
                f"signature takes {len(positional_slots)}: {signature}"
            )
        for description in call.unresolved:
            violations.append(
                f"{where}): cannot resolve the keys of a ** spread: {description}"
            )
        if not accepts_any:
            for keyword in sorted(call.keywords - set(parameters)):
                via = f" (via **{', **'.join(call.spreads)})" if call.spreads else ""
                violations.append(
                    f"{where}{keyword}=...): {service_class.__name__}.{call.method} does not "
                    f"accept {keyword!r}{via}; real signature: {signature}"
                )
        required = {
            name
            for name, parameter in parameters.items()
            if name != "self"
            and parameter.default is inspect.Parameter.empty
            and parameter.kind
            in (inspect.Parameter.KEYWORD_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        }
        if not call.unresolved:
            for keyword in sorted(required - call.keywords):
                violations.append(
                    f"{where}): required keyword {keyword!r} is never passed; "
                    f"real signature: {signature}"
                )
    return violations


def _service_class(module_name: str, class_name: str) -> type:
    return getattr(importlib.import_module(module_name), class_name)


@pytest.mark.parametrize(
    "handler_path, service_module, service_class_name, receiver",
    HANDLER_SERVICE_PAIRS,
    ids=[pair[0].stem for pair in HANDLER_SERVICE_PAIRS],
)
def test_study_handler_passes_only_keywords_the_real_service_accepts(
    handler_path: Path, service_module: str, service_class_name: str, receiver: str
) -> None:
    violations = check_handler_against_service(
        handler_path, _service_class(service_module, service_class_name), receiver
    )
    assert not violations, "\n".join(violations)


def test_guard_sees_the_spread_call_sites_it_exists_for() -> None:
    """The resolver must actually see through the handler's helpers: the
    ``**self._scope_arguments()`` spreads resolve to both scope keys, and the
    teardown spread resolves through its local name and the pending attribute."""
    source = HANDLER_SERVICE_PAIRS[0][0].read_text(encoding="utf-8")
    calls = {
        (call.method, tuple(call.spreads)): call
        for call in collect_service_calls(source)
    }
    list_decks = next(
        call for (method, _), call in calls.items() if method == "list_decks"
    )
    assert {"scope_type", "workspace_id", "mode"} <= list_decks.keywords
    assert not list_decks.unresolved
    end_review = next(
        call for (method, _), call in calls.items() if method == "end_review_session"
    )
    assert end_review.spreads == ["teardown_request"]
    assert {
        "mode",
        "scope_type",
        "workspace_id",
        "review_session_id",
    } <= end_review.keywords
    assert not end_review.unresolved, end_review.unresolved


def test_guard_fails_on_a_reintroduced_scope_spread(tmp_path: Path) -> None:
    """Negative control: the exact call that crashed the app, re-inserted into
    a scratch copy of the handler, is named by method, keyword and line."""
    handler_path = HANDLER_SERVICE_PAIRS[0][0]
    source = handler_path.read_text(encoding="utf-8")
    anchor = "        cards = await service.list_flashcards(\n            mode=self._current_mode(),\n"
    assert source.count(anchor) == 1, (
        "the refresh_cards call site moved; update the control"
    )
    bad = anchor + "            **self._scope_arguments(),\n"
    scratch = tmp_path / "flashcards_handler_bad.py"
    scratch.write_text(source.replace(anchor, bad, 1), encoding="utf-8")
    bad_line = source[: source.index(anchor)].count("\n") + 1

    violations = check_handler_against_service(
        scratch, _service_class(*HANDLER_SERVICE_PAIRS[0][1:3])
    )

    assert violations, "the guard accepted the call that crashed the app"
    assert any(
        f"flashcards_handler_bad.py:{bad_line} service.list_flashcards(scope_type=...)"
        in line
        and "does not accept 'scope_type'" in line
        and "**self._scope_arguments()" in line
        for line in violations
    ), violations
    assert any("does not accept 'workspace_id'" in line for line in violations), (
        violations
    )
    # And nothing else in the handler is reported: the control isolates the one bad call.
    assert all(f":{bad_line} " in line for line in violations), violations


def test_guard_fails_on_an_unresolvable_spread(tmp_path: Path) -> None:
    """Fail closed: a spread the resolver cannot see through is a violation,
    not a silent pass."""
    handler_path = HANDLER_SERVICE_PAIRS[0][0]
    source = handler_path.read_text(encoding="utf-8")
    anchor = "            decks = await service.list_decks(mode=mode, **self._scope_arguments())\n"
    assert source.count(anchor) == 1
    scratch = tmp_path / "flashcards_handler_opaque.py"
    scratch.write_text(
        source.replace(
            anchor,
            "            decks = await service.list_decks(mode=mode, **opaque_kwargs)\n",
            1,
        ),
        encoding="utf-8",
    )
    violations = check_handler_against_service(
        scratch, _service_class(*HANDLER_SERVICE_PAIRS[0][1:3])
    )
    assert any(
        "service.list_decks(" in line
        and "cannot resolve the keys" in line
        and "opaque_kwargs" in line
        for line in violations
    ), violations
