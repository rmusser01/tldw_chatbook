# tldw_chatbook/Chunking/_shims/testing.py
"""Replaces tldw_Server_API.app.core.testing (spec §5.3). ~20 lines upstream."""
import os

_TRUTHY = {"1", "true", "yes", "on", "y", "t"}
_FALSY = {"0", "false", "no", "off", "n", "f", ""}


def is_truthy(value) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    s = str(value).strip().lower()
    if s in _TRUTHY:
        return True
    if s in _FALSY:
        return False
    try:
        return bool(int(s))
    except ValueError:
        return False


def is_test_mode() -> bool:
    return os.getenv("PYTEST_CURRENT_TEST", "") != "" or os.getenv("TLDW_TEST_MODE", "") != ""
