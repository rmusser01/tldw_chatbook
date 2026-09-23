"""The workspace registry's clock must not share the canonical helper's name.

TASK-32901 (tier-2 S13 P2): ``Workspaces/models.py`` defined ``utc_now_iso``
-- the same public name as the sanctioned ``Utils/timestamps.utc_now_iso`` --
returning a *different*, variable-width shape (``...+00:00``, microseconds
present or omitted per value) against the canonical fixed-width millisecond
``Z``. The registry's ``ORDER BY created_at ASC`` sorts those values lexically.

No mixed-shape column exists today, so the live hazard is the collision
itself: "fixing" the import to the shared helper silently rewrites every
future stored value and *creates* the mixed column. Renaming the local clock
removes the accident, and pinning its shape catches anyone who swaps it.
"""

from __future__ import annotations

import re

from tldw_chatbook.Utils.timestamps import utc_now_iso as canonical_utc_now_iso
from tldw_chatbook.Workspaces import models


def test_local_registry_clock_does_not_shadow_the_canonical_name():
    assert not hasattr(models, "utc_now_iso")
    assert callable(models.registry_now_iso)


def test_registry_clock_keeps_the_stored_aware_offset_shape():
    """Stored registry timestamps keep their committed shape.

    Changing this silently rewrites ``workspaces``/``workspace_memberships``
    rows into a second shape in the same lexically-ordered column.
    """
    value = models.registry_now_iso()

    assert value.endswith("+00:00")
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?\+00:00", value)
    # The canonical helper is a genuinely different shape -- that is the point.
    assert not canonical_utc_now_iso().endswith("+00:00")
