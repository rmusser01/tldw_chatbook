"""Mutation controls for the definition checks shared by selected owners."""

from __future__ import annotations

import pytest

from tldw_chatbook.Chat.console_domain_services import (
    _DOMAIN_DEFINITIONS,
    _DOMAIN_HELPER_DEFINITIONS,
    _definition,
    _definition_current,
)

pytestmark = pytest.mark.bootstrap_profile


def _closed_function():
    captured = object()
    default = object()
    keyword = object()

    def function(value=default, *, option=keyword):
        return value, option, captured

    return function


def test_original_definitions_and_empty_metadata_remain_current():
    for _, _, record in _DOMAIN_DEFINITIONS:
        assert _definition_current(record)
    for _, record in _DOMAIN_HELPER_DEFINITIONS:
        assert _definition_current(record)

    def empty():
        return None

    record = _definition(empty)
    assert _definition_current(record)
    empty.__kwdefaults__ = {}
    assert not _definition_current(record)
    record = _definition(empty)
    assert _definition_current(record)
    empty.__kwdefaults__["added"] = None
    assert not _definition_current(record)


@pytest.mark.parametrize("changed", [
    "code", "namespace", "defaults", "keyword_mapping", "keyword_value",
    "keyword_added", "keyword_removed", "closure_identity", "closure_value",
])
def test_definition_metadata_mutation_is_rejected(changed):
    function = _closed_function()
    record = _definition(function)
    assert _definition_current(record)
    if changed == "code":
        function.__code__ = function.__code__.replace(co_name="changed")
    elif changed == "namespace":
        record = (*record[:2], dict(record[2]), *record[3:])
    elif changed == "defaults":
        function.__defaults__ = tuple(list(function.__defaults__))
    elif changed == "keyword_mapping":
        function.__kwdefaults__ = dict(function.__kwdefaults__)
    elif changed == "keyword_value":
        function.__kwdefaults__["option"] = object()
    elif changed == "keyword_added":
        function.__kwdefaults__["added"] = None
    elif changed == "keyword_removed":
        del function.__kwdefaults__["option"]
    elif changed == "closure_identity":
        record = (*record[:6], tuple(list(record[6])), *record[7:])
    else:
        function.__closure__[0].cell_contents = object()
    assert not _definition_current(record)


def test_wrapped_definition_and_absence_are_checked_on_every_call():
    inner = _closed_function()

    def wrapper():
        return None

    absent = _definition(wrapper)
    wrapper.__wrapped__ = inner
    assert not _definition_current(absent)
    record = _definition(wrapper)
    assert _definition_current(record)
    inner.__kwdefaults__["option"] = object()
    assert not _definition_current(record)
    record = _definition(wrapper)
    wrapper.__wrapped__ = _closed_function()
    assert not _definition_current(record)
    del wrapper.__wrapped__
    assert not _definition_current(record)
    assert _definition_current(absent)


def test_nonempty_metadata_preserves_original_exception_boundaries():
    function = _closed_function()
    record = _definition(function)
    del function.__closure__[0].cell_contents
    with pytest.raises(ValueError, match="Cell is empty"):
        _definition_current(record)

    class ChangedMapping(dict):
        def get(self, key, default=None):
            raise StopIteration("changed mapping")

    function = _closed_function()
    function.__kwdefaults__ = ChangedMapping(function.__kwdefaults__)
    record = _definition(function)
    with pytest.raises(RuntimeError, match="generator raised StopIteration"):
        _definition_current(record)
