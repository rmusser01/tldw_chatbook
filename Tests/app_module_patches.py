"""Patch a module-level name that ``tldw_chatbook.app`` shares with its extracted modules.

TASK-33011 moves ``TldwCli`` code out of ``app.py`` into sibling modules. A
moved body looks names up in its NEW module's globals, so a patch on
``tldw_chatbook.app.<name>`` alone no longer reaches it. When ``app.py`` and
``app_service_wiring.py`` both read a name (``get_cli_setting``,
``get_user_data_dir``, ``get_subscriptions_db_path`` and so on), a test that
means "every read the app makes" must patch both modules, with ONE object so
call assertions still see every call. These two helpers do that.

``Tests/Architecture/test_app_extracted_patch_targets.py`` fails on a bare
app-module patch of such a name, and points here.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import DEFAULT, patch

import pytest

#: The app module first, then each extracted module that shares its reads.
APP_GLOBAL_MODULES: tuple[str, ...] = (
    "tldw_chatbook.app",
    "tldw_chatbook.app_service_wiring",
)


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
            for module in APP_GLOBAL_MODULES[1:]:
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
    """``monkeypatch.setattr`` one name on every ``APP_GLOBAL_MODULES`` entry.

    Args:
        monkeypatch: The test's ``MonkeyPatch``.
        name: The module-level name to replace.
        value: The replacement, shared by every module.
    """
    for module in APP_GLOBAL_MODULES:
        monkeypatch.setattr(f"{module}.{name}", value)
