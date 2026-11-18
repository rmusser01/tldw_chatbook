"""Failure-only acquisition diagnostics never expose native exception details."""

import asyncio
import errno
import json

import pytest

from Tests.Plugins.acquisition_process_support import (
    acquisition_failure_diagnostic,
    repository,
)


@pytest.mark.parametrize("stage", ("spawn", "quota", "read"))
def test_acquisition_diagnostic_preserves_native_stage_and_original_error(
    plugin_stack, native_package, tmp_path, monkeypatch, capsys, stage
):
    from Tests.Plugins.test_acquisition_limits import acquire
    from tldw_chatbook.Plugins import acquisition

    source = repository(native_package(), tmp_path / "source")
    native = OSError(errno.EIO, "private-native-message", "/private/native-path")

    async def fail_async(*args, **kwargs):
        raise native

    def fail_sync(*args, **kwargs):
        raise native

    if stage == "spawn":
        monkeypatch.setattr(acquisition.ProcessTreeController, "spawn", fail_async)
    elif stage == "quota":
        monkeypatch.setattr(acquisition._Producer, "quota", fail_sync)
    else:
        monkeypatch.setattr(asyncio.StreamReader, "read", fail_async)
    with pytest.raises(acquisition.AcquisitionError) as caught:
        with acquisition_failure_diagnostic():
            acquire(plugin_stack, source)
    assert caught.value.__context__ is native
    prefix, payload = capsys.readouterr().out.strip().split(" ", 1)
    assert prefix == "ACQUISITION_NATIVE_FAILURE"
    assert json.loads(payload) == {"stage": stage, "errno": errno.EIO}


@pytest.mark.parametrize("native_errno", (None, "private-native-errno", True))
def test_acquisition_diagnostic_emits_only_integer_or_null_errno(capsys, native_errno):
    from tldw_chatbook.Plugins.acquisition import AcquisitionError

    native = OSError("private-native-message", "/private/native-path")
    native.errno = native_errno
    failure = AcquisitionError("acquisition_process_unavailable")
    with pytest.raises(AcquisitionError) as caught:
        with acquisition_failure_diagnostic():
            try:
                raise native
            except OSError:
                raise failure from None
    assert caught.value is failure and failure.__context__ is native
    prefix, payload = capsys.readouterr().out.strip().split(" ", 1)
    assert prefix == "ACQUISITION_NATIVE_FAILURE"
    assert json.loads(payload) == {"stage": "unknown_io", "errno": None}


@pytest.mark.parametrize("context", (None, ValueError("private-context")))
def test_acquisition_diagnostic_is_silent_without_native_context(capsys, context):
    from tldw_chatbook.Plugins.acquisition import AcquisitionError

    failure = AcquisitionError("acquisition_process_unavailable")
    failure.__context__ = context
    with pytest.raises(AcquisitionError) as caught:
        with acquisition_failure_diagnostic():
            raise failure
    assert caught.value is failure
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("output_error", (OSError, KeyboardInterrupt))
def test_acquisition_diagnostic_output_failure_keeps_original(
    monkeypatch, output_error
):
    from tldw_chatbook.Plugins.acquisition import AcquisitionError

    native = OSError(errno.EIO, "private-native-message")
    failure = AcquisitionError("acquisition_process_unavailable")

    def fail_print(*args, **kwargs):
        raise output_error("private-output-message")

    monkeypatch.setattr("builtins.print", fail_print)
    with pytest.raises(AcquisitionError) as caught:
        with acquisition_failure_diagnostic():
            try:
                raise native
            except OSError:
                raise failure from None
    assert caught.value is failure and failure.__context__ is native
