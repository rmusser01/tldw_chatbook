---
id: TASK-2235
title: 'Library: one canonical label for the ingest CTA everywhere (R2)'
status: Done
assignee: []
created_date: '2026-08-04 16:18'
updated_date: '2026-08-04 20:26'
labels:
  - ux-review
  - library
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
'Add content…' (rail top) vs 'Import media' (hub + Import/Export row) open the same canvas — identity crisis for the landing's most important action. Post-fix re-review P1. See Docs/superpowers/qa/2026-08-03-library-roleplay-mcp-ux-review.md.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 One canonical label per destination used consistently across rail, hub, and palette,Tests updated
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no (label copy only; behavior/ids unchanged). Canonical label: 'Add content…' -- F-013's deliberate plain-language pick, already used by the rail-top primary button and the command palette; 'Import media' was the surviving older name on the hub action row and the Import/Export rail row. Steps: 1. RED tests: shell-state pin for the ingest row title -> 'Add content…'; hub action row label pin in test_library_shell.py. 2. library_shell_state.py row title + library_screen.py hub action tuple labels. 3. Sweep other 'Import media' label assertions (contract-layout test, user guide). 4. Run shell-state/shell/contract/parity tests + ruff.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Canonical label 'Add content…' (F-013's plain-language pick) now used on all four surfaces: rail-top primary button (already), command palette (already), landing hub action row (changed from 'Import media', tooltip aligned with the rail-top button), and the Import/Export rail row (changed). 'Import media' survives only inside the ingest flow (canvas header, file-picker title) per F-013's boundary. Behavior/ids unchanged. Files: library_shell_state.py (row title), library_screen.py (hub action tuple), Tests/Library/test_library_shell_state.py (row-title pin), Tests/UI/test_library_shell.py (new test_ingest_cta_uses_one_canonical_label_everywhere covering all four surfaces), contract-layout + two replay files (text assertions), Docs/User_Guide/library.md + import-and-export.md. Verified: new test RED->GREEN; consistency + palette + shell-state 87 passed; contract/replay/destination sweep 198 passed + 1 skip; full test_library_shell.py 329 passed. Ruff clean (1 pre-existing F401 in test_library_shell.py untouched). ADR: not required (label copy only). Commit 6aa679f09.
<!-- SECTION:NOTES:END -->
