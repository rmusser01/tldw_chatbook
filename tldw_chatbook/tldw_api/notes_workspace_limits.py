"""Bound constants for the workspace-source owner projections.

Stdlib-only on purpose (TASK-23023). ``Research_Workspace/server_adapter.py``
enforces these bounds on every owner-rows projection, and it used to import
them from ``notes_workspace_schemas`` -- a 782-LOC, 26-model pydantic module
-- paying that module's whole import cost for one integer.
``notes_workspace_schemas`` re-imports these names from here (the
``chunking_engine_version.py`` / ``search_modes.py`` pattern), so there is
exactly one object per bound in the process and the value the adapter
enforces cannot drift from the one the schema fields validate against.
Guarded by ``Tests/Packaging/test_research_workspace_import_closure.py``.
"""

from __future__ import annotations

#: Maximum rows in one workspace-source page.
MAX_WORKSPACE_SOURCE_ROWS = 100

# GET sources/status are unpaged owner projections. This finite bound covers
# the public offset contract (10_000) plus one maximum page (100).
MAX_WORKSPACE_SOURCE_OWNER_ROWS = 10_100
