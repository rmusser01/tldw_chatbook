"""The shared wide boolean-flag coercer (TASK-32808.4).

`coerce_bool_flag` is the one implementation of the vocabulary that
`console_background_effects`, `Image_Generation.config` and
`character_expression_playback` had each copied verbatim. It is deliberately
DISTINCT from `config.coerce_bool_setting`, whose narrower fail-closed
vocabulary (no "on"; unrecognized reads False) backs config/security gates and
is pinned separately -- this test also guards that they stay different.
"""

from __future__ import annotations

import pytest

from tldw_chatbook.Utils.Utils import coerce_bool_flag


@pytest.mark.parametrize("raw", ["true", "True", "  TRUE  ", "1", "yes", "on", "ON"])
def test_true_vocabulary(raw):
    assert coerce_bool_flag(raw, False) is True


@pytest.mark.parametrize("raw", ["false", "False", " off ", "0", "no", "OFF"])
def test_false_vocabulary(raw):
    assert coerce_bool_flag(raw, True) is False


def test_real_bools_pass_through():
    assert coerce_bool_flag(True, False) is True
    assert coerce_bool_flag(False, True) is False


def test_integer_one_is_true_zero_is_false():
    assert coerce_bool_flag(1, False) is True
    assert coerce_bool_flag(0, True) is False


@pytest.mark.parametrize("raw", ["maybe", "", "enable", "sure", 2, 3.5, object(), None])
def test_unrecognized_returns_default(raw):
    assert coerce_bool_flag(raw, True) is True
    assert coerce_bool_flag(raw, False) is False


def test_stays_distinct_from_the_config_gate_vocabulary():
    """The security gate helper must NOT gain "on" as true, and this flag
    helper MUST read "on" as true -- pinning that the two are not merged."""
    from tldw_chatbook.config import coerce_bool_setting

    assert coerce_bool_flag("on", False) is True
    assert coerce_bool_setting("on", False) is False
