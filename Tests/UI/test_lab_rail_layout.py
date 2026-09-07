"""Pure collapse state for the Lab frame's two rails."""

from __future__ import annotations

import pytest

from tldw_chatbook.UI.Lab_Modules.lab_rail_layout import (
    LAB_RAIL_INSPECTOR,
    LAB_RAIL_LEFT,
    LAB_RAILS,
    LabRailLayout,
)


def test_default_layout_has_nothing_collapsed():
    layout = LabRailLayout()
    assert layout.is_collapsed(LAB_RAIL_LEFT) is False
    assert layout.is_collapsed(LAB_RAIL_INSPECTOR) is False


def test_toggle_collapses_then_expands():
    layout = LabRailLayout()
    collapsed = layout.toggle(LAB_RAIL_LEFT)
    assert collapsed.is_collapsed(LAB_RAIL_LEFT) is True
    assert collapsed.is_collapsed(LAB_RAIL_INSPECTOR) is False
    assert collapsed.toggle(LAB_RAIL_LEFT).is_collapsed(LAB_RAIL_LEFT) is False


def test_toggle_returns_a_new_instance_and_leaves_the_original_alone():
    """Frozen means callers can hold an old layout without it mutating."""
    layout = LabRailLayout()
    other = layout.toggle(LAB_RAIL_INSPECTOR)
    assert other is not layout
    assert layout.is_collapsed(LAB_RAIL_INSPECTOR) is False


def test_the_two_rails_are_independent():
    layout = LabRailLayout().toggle(LAB_RAIL_LEFT).toggle(LAB_RAIL_INSPECTOR)
    assert layout.is_collapsed(LAB_RAIL_LEFT) is True
    assert layout.is_collapsed(LAB_RAIL_INSPECTOR) is True


@pytest.mark.parametrize("method", ["is_collapsed", "toggle"])
def test_unknown_rail_names_raise(method):
    """A typo'd rail must fail loudly, not collapse nothing forever."""
    layout = LabRailLayout()
    with pytest.raises(ValueError):
        getattr(layout, method)("sidebar")


def test_lab_rails_lists_both_rails_in_render_order():
    assert LAB_RAILS == (LAB_RAIL_LEFT, LAB_RAIL_INSPECTOR)
