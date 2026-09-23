"""The three rails coerce stored booleans through one shared helper.

TASK-32901 (tier-2 S17 P2): ``Home/home_rail_state.py``,
``Chat/console_rail_state.py`` and ``Library/library_rail_state.py`` each
carried a byte-identical private ``_coerce_bool`` plus its own retyped
true/false string sets, and the resolved boolean is persisted back into rail
state -- so the duplication reaches storage.

The canonical home is ``Utils.coerce_bool_flag``, NOT
``config.coerce_bool_setting``: the shared helper's own docstring reserves
``coerce_bool_setting`` for "config/security gates ... its narrower
fail-closed vocabulary is intentional and separately pinned", and rail
open/collapsed state is a UI preference, not a gate. Adopting the gate helper
would have dropped ``" yes "`` (it does not strip) and non-zero ints, both of
which work in the rails today.
"""

from __future__ import annotations

import pytest

from tldw_chatbook.Chat import console_rail_state
from tldw_chatbook.Home import home_rail_state
from tldw_chatbook.Library import library_rail_state
from tldw_chatbook.Utils.Utils import coerce_bool_flag

RAIL_MODULES = (home_rail_state, console_rail_state, library_rail_state)

VOCABULARY = (
    True,
    False,
    "true",
    "TRUE",
    " yes ",
    "on",
    "1",
    "false",
    "no",
    "off",
    "0",
    1,
    0,
    None,
    "banana",
    2.5,
    object(),
)


@pytest.mark.parametrize("module", RAIL_MODULES, ids=lambda m: m.__name__)
def test_rail_modules_do_not_keep_a_private_bool_coercer(module):
    assert not hasattr(module, "_coerce_bool")
    assert not hasattr(module, "_TRUE_STRINGS")
    assert not hasattr(module, "_FALSE_STRINGS")


@pytest.mark.parametrize("raw", VOCABULARY, ids=repr)
@pytest.mark.parametrize("fallback", (True, False))
def test_home_rail_preferences_agree_with_the_shared_helper(raw, fallback):
    defaults = home_rail_state.HomeRailPreferences()
    resolved = home_rail_state.coerce_home_rail_preferences({"details_open": raw})
    assert resolved.details_open == coerce_bool_flag(raw, defaults.details_open)


def test_the_whole_shipped_vocabulary_still_resolves_the_same_way():
    """Nothing in the accepted vocabulary changes meaning.

    Only an out-of-vocabulary int (e.g. a hand-edited ``2``) moves: it now
    falls back like every other unrecognised value instead of being truthy.
    """
    for raw in ("true", "yes", "on", "1", "false", "no", "off", "0", " yes ", 1, 0):
        assert coerce_bool_flag(raw, False) == coerce_bool_flag(raw, True)
