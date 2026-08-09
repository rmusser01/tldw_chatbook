"""Wiring tests for the multi-instance boot warning (RAG-53 / task-7), plus
the sibling config-load-failure boot warning (TASK-13157).

``TldwCli`` is enormous to construct for real (it boots the whole app), so
``_maybe_warn_second_instance`` and ``_maybe_warn_config_load_failure`` are
both exercised unbound against a lightweight stub object -- the same pattern
used elsewhere in this suite for pinning a single method's behaviour without
paying for a full app instance (see e.g. ``inspect.getsource`` pins in
``Tests/UI/test_console_rail_sections.py`` and
``Tests/UI/test_chatbook_wizard_open_folder.py``).

A second test (per method) pins that ``_push_initial_screen`` actually calls
the warning method in its source, so the wiring can't silently regress (a
future refactor moving/removing the call would otherwise pass every other
test in this file while quietly disarming the warning at boot).
"""
from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

from tldw_chatbook.app import TldwCli
from tldw_chatbook.config import ConfigLoadFailure
from tldw_chatbook.Utils.instance_lock import InstanceLockStatus


def _stub_app(instance_lock_status=None, **extra_attrs):
    stub = SimpleNamespace(notify=MagicMock(), **extra_attrs)
    if instance_lock_status is not None:
        stub._instance_lock_status = instance_lock_status
    return stub


def test_warns_when_not_acquired():
    status = InstanceLockStatus(acquired=False, holder_pid=4242)
    stub = _stub_app(instance_lock_status=status)

    TldwCli._maybe_warn_second_instance(stub)

    stub.notify.assert_called_once()
    _args, kwargs = stub.notify.call_args
    assert kwargs.get("severity") == "warning"
    assert kwargs.get("timeout") == 10
    assert kwargs.get("title") == "Profile already open"
    message = _args[0] if _args else kwargs.get("message", "")
    assert "pid 4242" in message


def test_warns_without_pid_detail_when_holder_pid_unknown():
    status = InstanceLockStatus(acquired=False, holder_pid=None)
    stub = _stub_app(instance_lock_status=status)

    TldwCli._maybe_warn_second_instance(stub)

    stub.notify.assert_called_once()
    _args, kwargs = stub.notify.call_args
    message = _args[0] if _args else kwargs.get("message", "")
    assert "pid" not in message


def test_no_warning_when_acquired():
    status = InstanceLockStatus(acquired=True)
    stub = _stub_app(instance_lock_status=status)

    TldwCli._maybe_warn_second_instance(stub)

    stub.notify.assert_not_called()


def test_no_warning_when_status_attribute_absent():
    stub = _stub_app()  # no _instance_lock_status attribute at all

    TldwCli._maybe_warn_second_instance(stub)

    stub.notify.assert_not_called()


def test_no_warning_when_status_is_none():
    stub = _stub_app(instance_lock_status=None)
    # _stub_app skips setting the attr when None is passed; set explicitly.
    stub._instance_lock_status = None

    TldwCli._maybe_warn_second_instance(stub)

    stub.notify.assert_not_called()


def test_push_initial_screen_wires_the_warning_call():
    """Source-level pin: ``_push_initial_screen`` must call
    ``_maybe_warn_second_instance`` so the wiring can't be silently dropped
    by a future edit while every other (unit-level) test here still passes.
    """
    source = inspect.getsource(TldwCli._push_initial_screen)
    assert "_maybe_warn_second_instance" in source


def test_config_load_failure_warns_with_error_severity():
    failure = ConfigLoadFailure(
        path=Path("/home/user/.config/tldw_cli/config.toml"),
        message="Invalid TOML syntax at line 3",
    )
    stub = _stub_app(_config_load_failure=failure)

    TldwCli._maybe_warn_config_load_failure(stub)

    stub.notify.assert_called_once()
    _args, kwargs = stub.notify.call_args
    assert kwargs.get("severity") == "error"
    assert kwargs.get("timeout") == 60
    assert kwargs.get("title") == "Config file failed to load"
    message = _args[0] if _args else kwargs.get("message", "")
    assert str(failure.path) in message
    assert failure.message in message


def test_no_config_load_failure_warning_when_none():
    stub = _stub_app(_config_load_failure=None)

    TldwCli._maybe_warn_config_load_failure(stub)

    stub.notify.assert_not_called()


def test_no_config_load_failure_warning_when_attribute_absent():
    stub = _stub_app()  # no _config_load_failure attribute at all

    TldwCli._maybe_warn_config_load_failure(stub)

    stub.notify.assert_not_called()


def test_push_initial_screen_wires_the_config_load_failure_warning_call():
    """Source-level pin: ``_push_initial_screen`` must call
    ``_maybe_warn_config_load_failure`` so the wiring can't be silently
    dropped by a future edit while every other (unit-level) test here still
    passes.
    """
    source = inspect.getsource(TldwCli._push_initial_screen)
    assert "_maybe_warn_config_load_failure" in source
