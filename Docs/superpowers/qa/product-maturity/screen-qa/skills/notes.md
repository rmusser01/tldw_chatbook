# Skills Screenshot QA Notes

Date:
2026-05-11
Branch: `codex/screen-qa-skills`
Backlog task: TASK-14.11
Commit:
pending
Screen: Skills
Viewport:
2050x1240
Launch method:
`tldw-serve --host 127.0.0.1 --port 8831` with isolated HOME/XDG config and `default_tab = "skills"`
Screenshot method:
Playwright browser screenshot against textual-web at `http://127.0.0.1:8831`
Fallback reason:
None; actual textual-web screenshots were used.

## Baseline Screenshot

- Path: `Docs/superpowers/qa/product-maturity/screen-qa/skills/baseline-2026-05-11-skills.png`
- Defects: Skills inherited bulky destination spacing, lacked clearly defined three-column workbench structure, and the live textual-web screen stayed on `Loading local Agent Skills...` instead of resolving to the empty/local state.

## Interaction Smoke

- Goal: Verify the empty local Agent Skills state resolves and exposes disabled recovery actions without leaving the user in a loading state.
- Steps: Launched Skills as the default tab in textual-web, waited for local skills discovery, inspected the empty library/detail/inspector panes, and verified the import/attach actions are disabled with explanatory copy.
- Result: Empty state resolved to "No local Agent Skills are installed yet"; inspector reports that import is not wired in this shell yet and keeps import/attach actions disabled.

## Fixes

- Summary: Adapted Skills to the approved Textual-native destination pattern with compact header, explicit three-column labels, vertical dividers, and focused workbench sizing. Changed local skills refresh to a thread worker so textual-web does not remain stuck on the loading state.

## Final Screenshot

- Path: `Docs/superpowers/qa/product-maturity/screen-qa/skills/final-2026-05-11-skills.png`
- User approval: approved in Codex thread after actual rendered screenshot review.

## Verification

- Commands:
  - `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/UI/test_destination_shells.py::test_skills_destination_uses_three_column_workbench_contract Tests/UI/test_destination_shells.py::test_skills_destination_empty_state_disables_console_attach --tb=short`
  - `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/UI/test_destination_visual_parity_correction.py -k 'source_prep_loading_states_preserve_workbench_geometry and skills' --tb=short`
- Results:
  - `2 passed, 1 warning`
  - `1 passed, 77 deselected, 1 warning`

## Residual Risks

- Importing external Agent Skills is still not wired in this shell; the screen now presents that as an explicit disabled recovery state rather than a broken action.
