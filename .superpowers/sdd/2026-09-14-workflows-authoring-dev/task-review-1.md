Spec Compliance

- Issues found: JSON exchange can raw-open an alias of the live database, violating the binding SQLite constraint.
- Cannot verify: preserved-source byte identity and coordinator-owned compatibility/capture evidence were not independently rechecked.

## Strengths

- Storage remains a small authoring-only wrapper using the existing connection factory. App initialization is lazy and additive. Workflows_DB.py:20; app.py:19163.
- Revision saves check generation and head within one transaction; draft writes retain newer edits and survive caller cancellation. document_service.py:973; draft_session.py:219.
- Console refresh updates stable children independently of the editor. console_context.py:78.

## Important

Prevent database aliases from reaching file exchange. authoring.py:139 compares lexical paths only. A `.json` hard link to the live database passes this check. `open_private_binary` then opens the inode before checking its link count (private_paths.py:1047), rejects it, and closes the descriptor (private_paths.py:1092). On POSIX this can release SQLite’s process-owned locks during an outstanding draft transaction—even though import fails. Reject database/sidecar aliases before generic file opening and add a failed-import foreign-writer regression.

## Minor

Existing verification debt remains unwaived: 713 Ruff findings, five formatter failures, Requests warnings and Kokoro cleanup noise. These are reported baseline concerns, not new task regressions. Report:410.

## Assessment

Task quality: Needs fixes. No Critical findings. The authoring implementation has substantial behavioral coverage, but exchange bypasses the required live-database boundary.

Checks: Reviewed complete 11,388-line immutable package. Targeted outside-diff check: private-file helpers and path validation for database-alias risk. No mutations, subagents, app boots or test reruns. Supplied final evidence records 374 targeted passes.
