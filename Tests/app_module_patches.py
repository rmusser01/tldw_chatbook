"""Patch a module-level name that ``tldw_chatbook.app`` shares with its extracted modules.

TASK-33011 moves ``TldwCli`` code out of ``app.py`` into sibling modules. A
moved body looks names up in its NEW module's globals, so a patch on
``tldw_chatbook.app.<name>`` alone no longer reaches it. When ``app.py`` and
``app_service_wiring.py`` both read a name (``get_cli_setting``,
``get_user_data_dir``, ``get_subscriptions_db_path`` and so on), a test that
means "every read the app makes" must patch both modules, with ONE object so
call assertions still see every call. These two helpers do that. An extracted
module that does not bind the name (``app_speech`` has no
``get_user_data_dir``) cannot read it as a global, so it is skipped; the app
module itself is always patched, so a misspelt name still raises. Whether a
module binds a name is read from its SOURCE, not by importing it, so patching a
name a lazily imported module (``app_speech``) does not bind leaves it unloaded
and the test still exercises its production first-use import.

``Tests/Architecture/test_app_extracted_patch_targets.py`` fails on a bare
app-module patch of such a name, and points here.
"""

from __future__ import annotations

import ast
import functools
import importlib.util
from pathlib import Path
from typing import Any
from unittest.mock import DEFAULT, patch

import pytest

#: The app module first, then each extracted module that shares its reads. The
#: eager extractions (mixins and the palette providers, loaded with app.py anyway)
#: are all listed. Of the lazy ones only ``app_speech`` is: its moved bodies share
#: names with app.py that tests patch. ``app_entry`` and ``app_destinations`` stay
#: out -- listing them would import them for every ``load_settings`` patch, and a
#: patch of a name only THEY read is still caught by the guard's moved-only rule.
APP_GLOBAL_MODULES: tuple[str, ...] = (
    "tldw_chatbook.app",
    "tldw_chatbook.app_ingest_queue",
    "tldw_chatbook.app_service_wiring",
    "tldw_chatbook.app_speech",
    "tldw_chatbook.app_lifecycle",
    "tldw_chatbook.app_navigation",
    "tldw_chatbook.app_command_providers",
    "tldw_chatbook.app_feature_glue",
)


def _is_type_checking_guard(node: ast.If) -> bool:
    test = node.test
    return (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or (
        isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"
    )


def _statement_bindings(statements: list[ast.stmt]) -> set[str]:
    names: set[str] = set()
    for node in statements:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                bound = alias.asname or alias.name
                names.add(bound.split(".")[0] if isinstance(node, ast.Import) else bound)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                names.update(
                    sub.id for sub in ast.walk(target) if isinstance(sub, ast.Name)
                )
        elif isinstance(node, ast.If):
            # ``if TYPE_CHECKING:`` imports are not bound at runtime.
            if not _is_type_checking_guard(node):
                names |= _statement_bindings(node.body)
            names |= _statement_bindings(node.orelse)
        elif isinstance(node, ast.Try):
            names |= _statement_bindings(node.body)
            names |= _statement_bindings(node.orelse)
            names |= _statement_bindings(node.finalbody)
            for handler in node.handlers:
                names |= _statement_bindings(handler.body)
        elif isinstance(node, ast.With):
            names |= _statement_bindings(node.body)
    return names


@functools.cache
def module_scope_names(module: str) -> frozenset[str]:
    """Names ``module`` binds at module scope, read from its source.

    Args:
        module: Dotted module name, e.g. ``"tldw_chatbook.app_speech"``.

    Returns:
        Every name a top-level import, def, class or assignment binds (outside
        ``if TYPE_CHECKING:``). Reading the source keeps a lazily imported
        module unloaded; ``find_spec`` imports only its parent package.
    """
    spec = importlib.util.find_spec(module)
    if spec is None or spec.origin is None:
        raise ModuleNotFoundError(module)
    tree = ast.parse(Path(spec.origin).read_text(encoding="utf-8"))
    return frozenset(_statement_bindings(tree.body))


def _binding_extracted_modules(name: str) -> list[str]:
    """The ``APP_GLOBAL_MODULES[1:]`` entries that bind ``name`` at module scope."""
    return [
        module for module in APP_GLOBAL_MODULES[1:] if name in module_scope_names(module)
    ]


class _AppGlobalPatcher:
    """``unittest.mock.patch`` of one name on every ``APP_GLOBAL_MODULES`` entry.

    The first module's patch creates (or takes) the replacement; every other
    module gets that same object. Supports ``with`` and ``start``/``stop``.
    """

    def __init__(self, name: str, new: Any, kwargs: dict[str, Any]) -> None:
        self._name = name
        self._primary = patch(f"{APP_GLOBAL_MODULES[0]}.{name}", new, **kwargs)
        self._others: list[Any] = []

    def start(self) -> Any:
        value = self._primary.start()
        try:
            for module in _binding_extracted_modules(self._name):
                other = patch(f"{module}.{self._name}", value)
                other.start()
                self._others.append(other)
        except BaseException:
            self.stop()
            raise
        return value

    def stop(self) -> None:
        while self._others:
            self._others.pop().stop()
        self._primary.stop()

    def __enter__(self) -> Any:
        return self.start()

    def __exit__(self, *exc_info: object) -> bool:
        self.stop()
        return False


def patch_app_global(name: str, new: Any = DEFAULT, **kwargs: Any) -> _AppGlobalPatcher:
    """Like ``patch("tldw_chatbook.app.<name>", new, **kwargs)``, on every app module.

    Args:
        name: The module-level name to replace.
        new: The replacement; omitted, a ``MagicMock`` is created as usual.
        **kwargs: Passed to the first ``patch`` (``return_value``,
            ``side_effect`` and so on).

    Returns:
        A patcher usable as a context manager or via ``start``/``stop``.
    """
    return _AppGlobalPatcher(name, new, kwargs)


def set_app_global(monkeypatch: pytest.MonkeyPatch, name: str, value: Any) -> None:
    """``monkeypatch.setattr`` one name on every app module that binds it.

    Args:
        monkeypatch: The test's ``MonkeyPatch``.
        name: The module-level name to replace.
        value: The replacement, shared by every module.
    """
    monkeypatch.setattr(f"{APP_GLOBAL_MODULES[0]}.{name}", value)
    for module in _binding_extracted_modules(name):
        monkeypatch.setattr(f"{module}.{name}", value)
