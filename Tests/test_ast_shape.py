# test_ast_shape.py
# Description: Pins the cross-version stability of `Tests.ast_shape.stable_dump`
"""Guard the wrapper that keeps frozen `ast.dump()` inventories portable.

Without this, the failure mode is silent and delayed: someone runs the suite
on the interpreter the ledger was generated with, everything passes, and the
break only shows up for whoever upgrades Python -- which is exactly how five
tests in `test_summarization_diagnostic_privacy.py` came to fail on 3.14
while passing on the repo's 3.12 venv.
"""

from __future__ import annotations

import ast

import pytest

from Tests.ast_shape import stable_dump


def _call_node() -> ast.AST:
    """A call with no keyword arguments -- the shape whose rendering moved."""
    return ast.parse("type(data)").body[0].value


def test_stable_dump_renders_empty_fields_on_every_supported_version() -> None:
    """The load-bearing assertion.

    `ast.dump()` gained `show_empty` in 3.13 and defaulted it to False, so a
    no-keyword call renders as `Call(func=..., args=[...])` there and
    `Call(func=..., args=[...], keywords=[])` on 3.11/3.12. Committed
    inventories hold the latter. This fails on >=3.13 the moment the wrapper
    stops pinning it, and passes on 3.11/3.12 either way -- so it is the
    check that actually travels.
    """
    assert "keywords=[]" in stable_dump(_call_node())


def test_stable_dump_matches_plain_dump_where_nothing_is_empty() -> None:
    """The wrapper must not otherwise alter the rendering.

    A node with no empty fields has to dump identically through either path,
    or the wrapper would be silently rewriting shapes rather than pinning one
    version's rendering of them.
    """
    node = ast.parse("x + 1").body[0].value
    assert stable_dump(node) == ast.dump(node)


def test_stable_dump_forwards_its_keyword_arguments() -> None:
    """Callers pass `include_attributes=False`; it has to reach `ast.dump`."""
    node = _call_node()
    assert stable_dump(node, include_attributes=False) == stable_dump(node)
    with_attrs = stable_dump(node, include_attributes=True)
    assert "lineno=" in with_attrs


def test_stable_dump_rejects_an_explicit_show_empty() -> None:
    """Passing it back in would defeat the point, so it is refused loudly."""
    with pytest.raises(TypeError, match="show_empty"):
        stable_dump(_call_node(), show_empty=False)
