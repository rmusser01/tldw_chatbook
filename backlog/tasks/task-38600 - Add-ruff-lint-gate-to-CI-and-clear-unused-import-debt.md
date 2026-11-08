---
id: TASK-38600
title: Add ruff lint gate to CI and clear unused-import debt
status: Done
created_date: 2026-11-02 14:28
labels:
- chore
- ci
priority: medium
updated_date: 2026-11-03 02:56
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
None of the 24 CI workflows runs a linter and the package carries 1219 findings under ruff F B ASYNC PLE rules including 126 unused imports; full list in /tmp/ruff_audit_full.txt. See bug-sweep-2026-11-02-dev.md section 4.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A CI workflow runs ruff with at least F B ASYNC and PLE selected
- [x] #2 Unused imports are fixed so the gate starts green
- [x] #3 Remaining findings are triaged or baselined explicitly
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
F-class findings fixed 146 to 0 across 65 files with reasoned noqas; ruff ratchet gate added (lint.yml, baseline 1064 findings over 26 rules, pinned 0.16.6) plus ADR-355. Commits d0bfa5a5fe and d4f8e61b94.

**Rebased onto dev 2026-11-06 (PR #3361 rebuild).**
- The ADR was renumbered 287 to 355: dev's ADR-287 is the Personas workbench decision.
- dev had gained 50 new F findings since the audit:
  - 30 are `library_screen.py`'s re-exports. Its moved support layer is pinned by `Tests/Architecture/test_library_support_layer_surface.py`, so the file gets an F401 per-file ignore in `pyproject.toml`.
  - 12 are annotation-only undefined names in `Workspaces/change_review_provider.py`, left by `93e7c53459`'s move out of the screen. They now come from a `TYPE_CHECKING` block, and the screen keeps only the four it uses.
  - The rest are dead imports and one dead local.
- Six imports that ruff flags but that other code still reaches are kept under reasoned `noqa`s:
  - `agent_service.FIND_TOOLS_NAME` and `LOAD_TOOLS_NAME`, read through the module;
  - `app.LocalCharacterPersonaService`;
  - `RAG_Search/simplified/config.load_cli_config_and_ensure_existence`, a test patch target;
  - the speech TTS panel's `get_cli_setting`, a test patch target.

  An AST scan of `Tests/`, `tldw_chatbook/` and `scripts/` for from-imports, module attributes, `setattr` targets and dotted strings found them.
- The B/ASYNC/PLE baseline was regenerated on the rebased tree with the pinned 0.16.6: 1106 findings over 26 rules. Most of the growth is dev's own since the audit (B905 186 to 213).
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
