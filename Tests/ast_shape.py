# ast_shape.py
# Description: Version-stable `ast.dump()` for frozen-shape inventories
"""A single ``ast.dump()`` wrapper for tests that FREEZE dumped shapes.

Several guards in this suite record ``ast.dump()`` output in a committed,
individually-reviewed inventory (the summarization diagnostic ledger, the
persistent-diagnostic inventory, the Console wave-6 inventory) and compare
later runs against it. That only works if the dump is a pure function of
the source -- and by default it is not: it is also a function of the
interpreter.

The incident: Python 3.13 added ``show_empty`` to ``ast.dump()`` and
defaulted it to ``False``, so an empty field stopped being rendered.
``Call(func=..., args=[...], keywords=[])`` on <=3.12 became
``Call(func=..., args=[...])`` on >=3.13. Every frozen shape containing a
no-keyword call therefore stopped matching, and five tests in
``test_summarization_diagnostic_privacy.py`` failed on a 3.14 interpreter
while passing on the 3.12 venv the ledger was generated with -- a red that
looked like a product regression and was really a toolchain difference.

Measured on this repo, digesting all 229 diagnostic shapes in
``LLM_Calls/Local_Summarization_Lib.py`` with four real interpreters:

    3.11  digest=92a85a3a989ffd13   <- what the committed ledger holds
    3.12  digest=92a85a3a989ffd13
    3.13  digest=cd27a1b6c1fdf001   <- diverges exactly where show_empty landed
    3.14  digest=cd27a1b6c1fdf001

and with ``show_empty=True`` forced on >=3.13, all four produce
``92a85a3a989ffd13``. So pinning the OLD rendering (rather than
regenerating the ledger against the new one) is what makes the inventory
portable across the versions this project supports -- ``requires-python``
is ``>=3.11`` -- and it leaves the reviewed ledger rows untouched, which
matters because those rows are a privacy review, not a checksum.
"""

from __future__ import annotations

import ast
import sys
from typing import Any

#: ``show_empty`` exists from 3.13 on. Below that, empty fields are always
#: rendered, which IS the behaviour this module pins -- so no argument is
#: passed there and the call keeps working on the 3.11 floor.
_SUPPORTS_SHOW_EMPTY = sys.version_info >= (3, 13)


def stable_dump(node: ast.AST, **kwargs: Any) -> str:
    """``ast.dump(node, **kwargs)``, rendered the same on every version.

    Use this instead of ``ast.dump()`` anywhere the result is compared
    against a committed value. For a dump that is only ever produced and
    consumed within one process (an error message, a same-run comparison),
    plain ``ast.dump()`` is fine.

    Args:
        node: The node to dump.
        **kwargs: Passed through to ``ast.dump`` (e.g.
            ``include_attributes=False``). Passing ``show_empty`` yourself
            defeats the point and is rejected.

    Returns:
        The dump, with empty fields rendered on every supported version.

    Raises:
        TypeError: If ``show_empty`` is passed explicitly.
    """
    if "show_empty" in kwargs:
        raise TypeError(
            "stable_dump() pins show_empty itself -- passing it defeats the "
            "cross-version stability this wrapper exists to provide"
        )
    if _SUPPORTS_SHOW_EMPTY:
        return ast.dump(node, show_empty=True, **kwargs)
    return ast.dump(node, **kwargs)
