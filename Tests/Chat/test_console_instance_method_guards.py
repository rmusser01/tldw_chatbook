"""Installed instance guards retain live descriptors and definition checks."""

from contextlib import contextmanager
from dataclasses import replace
import inspect
import sys

import pytest

from Tests.private_profile import private_profile_test


@contextmanager
def _observe_checks(owner):
    from tldw_chatbook.Chat.console_domain_services import _definition_current

    static, definitions = [], []
    previous = sys.getprofile()

    def observe(frame, event, _arg):
        if event != "call":
            return
        if frame.f_code is inspect.getattr_static.__code__:
            if frame.f_locals["obj"] is owner:
                static.append(frame.f_locals["attr"])
        elif frame.f_code is _definition_current.__code__:
            definitions.append(frame.f_locals["record"])

    sys.setprofile(observe)
    try:
        yield static, definitions
    finally:
        sys.setprofile(previous)


def _assert_original_inventory(module, owner, current):
    with _observe_checks(owner) as (static, definitions):
        assert current()
    assert static == []
    assert any(record is module._PLAIN_METHOD for record in definitions)
    for _name, record in module._DEFINITIONS:
        assert sum(checked is record for checked in definitions) == 1


def _probe_method():
    captured = object()

    def entry(self, positional=None, *, keyword=None):
        return captured, positional, keyword

    return entry


def _assert_mutations_refused(module, owner, current, monkeypatch):
    from tldw_chatbook.Chat.console_domain_services import _definition
    from tldw_chatbook.Utils import plain_method_lookup

    kind = type(owner)
    # Append one controlled record to the full original inventory. The
    # installed guard and all original records still execute on the real type.
    mutations = {
        "class": lambda patch, fn: patch.setattr(kind, "_guard_probe", object()),
        "shadow": lambda patch, fn: patch.setitem(vars(owner), "_guard_probe", object()),
        "dictionary": lambda patch, fn: patch.setattr(
            owner, "__dict__", {**vars(owner), "_guard_probe": object()}
        ),
        "code": lambda patch, fn: patch.setattr(fn, "__code__", fn.__code__.replace(co_name="changed")),
        "defaults": lambda patch, fn: patch.setattr(fn, "__defaults__", (object(),)),
        "keywords": lambda patch, fn: patch.setitem(fn.__kwdefaults__, "keyword", object()),
        "closure": lambda patch, fn: patch.setattr(fn.__closure__[0], "cell_contents", object()),
        "wrapper": lambda patch, fn: patch.setattr(fn, "__wrapped__", _probe_method(), raising=False),
    }
    for change in mutations.values():
        with monkeypatch.context() as patch:
            function = _probe_method()
            patch.setattr(kind, "_guard_probe", function, raising=False)
            patch.setattr(module, "_DEFINITIONS", (*module._DEFINITIONS, ("_guard_probe", _definition(function))))
            assert current()
            change(patch, function)
            assert not current()
        assert current()

    helper = plain_method_lookup.plain_instance_method
    for change in ("replacement", "code", "fallback"):
        with monkeypatch.context() as patch:
            if change == "replacement":
                patch.setattr(plain_method_lookup, "plain_instance_method", lambda *_: None)
            elif change == "code":
                patch.setattr(helper, "__code__", helper.__code__.replace(co_name="changed"))
            else:
                patch.setitem(helper.__kwdefaults__, "_static", lambda *_: None)
            assert not current()
        assert current()


def test_installed_presentation_guard_keeps_live_checks(monkeypatch):
    from tldw_chatbook.Chat import console_backend_presentation as module

    # This method only inspects definitions/dictionary. No resource owner is
    # constructed and this test makes no native-authority claim.
    owner = object.__new__(module.ConsoleBackendPresentation)
    current = owner.definitions_current
    _assert_original_inventory(module, owner, current)
    _assert_mutations_refused(module, owner, current, monkeypatch)


@private_profile_test
@pytest.mark.asyncio
async def test_actual_backend_graph_keeps_live_checks(request, monkeypatch):
    from Tests.Chat.test_console_execution_profile import _settled_profile
    from tldw_chatbook.Chat import console_execution_backend as module
    from tldw_chatbook.Chat.console_execution_profile import open_console_backend_profile

    path, directory, *_ = _settled_profile()
    profile = open_console_backend_profile(
        expected_config_path=path, expected_data_directory=directory,
    )
    backend = module.ConsoleExecutionBackend(
        profile, profile_epoch="profile", service_epoch="instance-method-guards",
    )

    def current():
        try:
            backend._require_source_graph()
        except RuntimeError as error:
            assert str(error) == "Console backend source association changed"
            return False
        return True

    try:
        backend.start()
        _assert_original_inventory(module, backend, current)
        _assert_mutations_refused(module, backend, current, monkeypatch)
        with monkeypatch.context() as patch:
            patch.setattr(backend, "closed", True)
            assert not current()
        with monkeypatch.context() as patch:
            patch.setattr(type(profile), "selection_is_current", lambda self: True)
            assert not current()
        for field in ("configuration", "chat_database", "prompt_database", "control_plane", "prompt_scope"):
            with monkeypatch.context() as patch:
                patch.setattr(backend.domain, "sources", replace(backend.domain.sources, **{field: object()}))
                assert not current()
            assert current()
    finally:
        # Every source/descriptor mutation is restored before original disposal.
        await backend.aclose()
