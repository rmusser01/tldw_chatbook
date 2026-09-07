# TASK-840 Console citation format normalization

## Goal

Normalize inherited Ruff formatting drift in the four Console files identified
by TASK-553.15 without changing behavior.

## Implementation

1. Reproduce the exact eleven-file Ruff format check from TASK-553.15.
2. Run Ruff format on only the four recorded failing files.
3. Review the complete diff for formatter-only changes.
4. Run the exact eleven-file format gate, Ruff check on the touched files,
   focused Console citation/UI tests, and `git diff --check`.

ADR required: no

ADR path: N/A

Reason: This is mechanical formatting normalization with no change to behavior,
interfaces, ownership, persistence, security, or architecture.
