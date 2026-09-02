"""Schedule-vocabulary translation at the client/server ownership boundary.

Pure module (no I/O, no imports of other Scheduling submodules): a local
automation definition's ``schedule`` dict is authored and previewed in
CLIENT field-name vocabulary (``Scheduling/schedule_compute.py``), but the
server's scheduling executor reads a DIFFERENT set of per-kind field names.
The server's own definition validator (``_validate_schedule`` in
``scheduled_task_automation_service.py``) checks only ``kind`` -- never the
per-kind fields -- so an untranslated client-vocab ``schedule`` dict passes
the server's create/update preview cleanly and then silently never arms,
because the server's REAL executor reads different field names (schedules-
handoff PR-5, final-review finding 6 for PR-4).

Rename table (client -> server), derived by reading both sides directly:

- Client: ``tldw_chatbook/Scheduling/schedule_compute.py`` module docstring
  and ``_compute_*`` field reads (``_compute_one_time``, ``_compute_interval``,
  ``_compute_daily``, ``_compute_weekly``, ``_compute_cron``).
- Server: ``tldw_Server_API/app/services/scheduled_task_automation_scheduler.py``,
  function ``build_trigger`` (its module docstring states the schedule dict
  conventions the executor actually reads; ``build_trigger`` itself pulls
  ``schedule["seconds"]``, ``schedule.get("at")``, etc.).

| kind         | client field         | server field                    | translation      |
|--------------|-----------------------|----------------------------------|-------------------|
| one_time     | ``run_at``            | ``run_at``                       | same, no rename   |
| interval     | ``every_seconds``     | ``seconds``                      | renamed           |
| interval     | (no client field)     | ``start_at`` (optional ISO)      | server-only, passes through untouched |
| daily/weekly | ``time_of_day``       | ``at``                           | renamed           |
| weekly       | ``weekday`` (int 0-6) | ``weekday`` (int 0-6 or day name)| same field name, passthrough |
| cron         | ``cron``              | ``cron``                         | same, no rename   |
| all kinds    | ``timezone``          | ``timezone``                     | same, no rename   |

Both ``to_server_schedule`` and ``to_local_schedule`` are pure and
non-mutating (always return a new dict), pass unknown keys through
unchanged, and return an unchanged copy for an unrecognized ``kind``.
``to_server_schedule`` is idempotent -- calling it twice is the same as
calling it once, since client and server field names never collide, so a
second pass finds nothing left to rename. That makes it safe to wire into
a single translation site that more than one caller may end up routing
through (schedules-handoff PR-5 task 3/4).
"""
from __future__ import annotations

from typing import Any

#: Per-kind client-field -> server-field renames. Kinds absent here
#: (``one_time``, ``cron``) use identical field names on both sides.
_CLIENT_TO_SERVER_FIELD_RENAMES: dict[str, dict[str, str]] = {
    "interval": {"every_seconds": "seconds"},
    "daily": {"time_of_day": "at"},
    "weekly": {"time_of_day": "at"},
}

_SERVER_TO_CLIENT_FIELD_RENAMES: dict[str, dict[str, str]] = {
    kind: {server_field: client_field for client_field, server_field in renames.items()}
    for kind, renames in _CLIENT_TO_SERVER_FIELD_RENAMES.items()
}


def _translate(schedule: dict[str, Any], rename_tables: dict[str, dict[str, str]]) -> dict[str, Any]:
    """Rename ``schedule``'s per-kind fields per ``rename_tables``.

    Always returns a new dict -- ``schedule`` itself is never mutated.
    """
    if not isinstance(schedule, dict):
        return schedule
    renames = rename_tables.get(schedule.get("kind"))
    if not renames:
        # Unrecognized kind, or a recognized kind with no field renames
        # (one_time/cron): unchanged copy.
        return dict(schedule)
    return {renames.get(key, key): value for key, value in schedule.items()}


def to_server_schedule(schedule: dict[str, Any]) -> dict[str, Any]:
    """Translate a ``schedule`` dict from client vocabulary to server vocabulary."""
    return _translate(schedule, _CLIENT_TO_SERVER_FIELD_RENAMES)


def to_local_schedule(schedule: dict[str, Any]) -> dict[str, Any]:
    """Translate a ``schedule`` dict from server vocabulary to client vocabulary."""
    return _translate(schedule, _SERVER_TO_CLIENT_FIELD_RENAMES)
