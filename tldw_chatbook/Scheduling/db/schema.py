"""Scheduling module database schema.

The canonical DDL lives in ``migrations/v0_to_v1.py`` so that schema creation
is versioned alongside later migrations. This module re-exports the initial
schema string for callers that need it at import time.

JSON columns are stored as TEXT and parsed/serialized in Python.
All datetime columns store UTC ISO-8601 strings.
"""

from tldw_chatbook.Scheduling.db.migrations.v0_to_v1 import CREATE_SCHEMA_SQL

__all__ = ["CREATE_SCHEMA_SQL"]
