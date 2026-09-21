"""coerce_bool_setting(None, default) must return the default (TASK-32808.4).

It delegated to _get_typed_value, which returns None unchanged for a None
value, so a `-> bool` helper returned None — "a feature that ships ON reads
OFF" whenever the key was absent. coerce_int_setting already special-cased
None; the bool helper now does too.
"""

from tldw_chatbook.config import coerce_bool_setting


def test_none_returns_the_default_not_none():
    assert coerce_bool_setting(None, False) is False
    assert coerce_bool_setting(None, True) is True
    # The annotation says -> bool; None must never leak through.
    assert coerce_bool_setting(None) is True  # default default is True


def test_string_and_number_spellings_still_coerce():
    assert coerce_bool_setting("false", True) is False
    assert coerce_bool_setting("true", False) is True
    assert coerce_bool_setting("0", True) is False
    assert coerce_bool_setting("1", False) is True
    assert coerce_bool_setting(1, False) is True
    assert coerce_bool_setting(0, True) is False


def test_actual_bools_pass_through():
    assert coerce_bool_setting(True, False) is True
    assert coerce_bool_setting(False, True) is False
