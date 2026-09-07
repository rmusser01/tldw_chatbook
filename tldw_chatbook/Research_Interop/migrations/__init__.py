"""Versioned schema migrations for the local research database.

Each module in this package upgrades the database from one
``PRAGMA user_version`` to the next and stamps the target version.
``LocalResearchService._init_schema`` applies them in order on open;
databases stamped by a newer service are refused rather than silently
downgraded (Qodo PR-1822 finding 7 -- the lease columns originally landed
as bare, unversioned startup ALTERs).
"""

from __future__ import annotations

import sqlite3
from typing import Callable

from . import v0_to_v1_run_lease_columns

#: Every migration, ordered by target version. A database at version N is
#: upgraded by applying each step whose target exceeds N, in order.
MIGRATIONS: tuple[tuple[int, Callable[[sqlite3.Connection], None]], ...] = (
    (v0_to_v1_run_lease_columns.TARGET_VERSION, v0_to_v1_run_lease_columns.apply),
)

__all__ = ["MIGRATIONS"]
