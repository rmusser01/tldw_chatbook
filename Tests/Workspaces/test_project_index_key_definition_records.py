"""Exact record shape and explicit alias slots survive the pure extraction."""

from functools import wraps

from tldw_chatbook.Utils.definition_records import (
    capture_definitions,
    definitions_current,
)


def test_definition_record_preserves_alias_wrappers_and_mutable_defaults():
    token = object()

    def inner(value=token, *, flag=token):
        return value, flag

    def wrap(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            return function(*args, **kwargs)

        return wrapped

    middle = wrap(inner)
    outer = wrap(middle)
    namespace = {"explicit_alias": outer}
    records = capture_definitions(((namespace, "explicit_alias", outer),))
    owner, name, original, functions = records[0]
    assert owner is namespace and name == "explicit_alias" and original is outer
    assert tuple(record[0] for record in functions) == (outer, middle, inner)
    for record, function in zip(functions, (outer, middle, inner), strict=True):
        assert record[1] is function.__code__
        assert record[2] is function.__defaults__
        assert record[3] is function.__kwdefaults__
        assert record[5] is function.__closure__
    assert functions[-1][4] == (("flag", token),)
    assert definitions_current(records)
    inner.__kwdefaults__["flag"] = object()
    assert not definitions_current(records)
    inner.__kwdefaults__["flag"] = token
    assert definitions_current(records)
    outer.__code__ = outer.__code__.replace()
    assert not definitions_current(records)
