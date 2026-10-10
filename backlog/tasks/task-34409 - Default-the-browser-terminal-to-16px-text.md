---
id: TASK-34409
title: Default the browser terminal to 16px text
status: Done
assignee:
  - '@codex'
created_date: '2026-10-04 20:33'
updated_date: '2026-10-04 20:48'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Use the approved readable browser text size on fresh served launches and keep explicit user overrides.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Fresh served launches, generated config, and Settings use 16px by default.
- [x] #2 Saved font sizes and valid URL overrides retain their existing precedence.
- [x] #3 Focused tests and a real browser check verify the default and override behavior.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Update the default-behavior tests and observe the expected failures.
2. Set the server, generated-config, and Settings defaults to 16px; document pixels and override precedence.
3. Run focused tests and static checks, then verify Console and Watchlists in a real browser without a font-size override.
4. Review and open a PR against dev.

ADR required: no
ADR path: N/A
Reason: a default-value adjustment within the existing served shell and font-size setting; no new interface, runtime, or persistence boundary.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Set the server fallback, generated config, and canonical Settings appearance default to 16 CSS pixels. Preserve saved sizes and URL overrides; document units and how existing profiles adopt the default.

Verification: 47 focused viewport and Appearance tests pass. Default tests were observed failing at 12 before implementation. The fake-server helper supplies a disabled Canvas policy; pure Appearance helpers avoid the unrelated full-app catalog-refresh fixture. These local harness repairs address baseline config-owner admission failures without changing production permission behavior.

Real browser: a private config without font_size and http://127.0.0.1:8775/ without a query produce data-font-size=16. Visually inspected Console and the populated Watchlists Read screen at 1280x720 (142x37 cells); screenshots and a DOM receipt are retained in the local task artifact directory.

Ruff passes for modified tests, and format checks pass for both tests and the Settings helper. Production files add no lint diagnostics compared with HEAD (77 baseline findings). Diff whitespace and Backlog ID/path/frontmatter guards pass. Read-only code review approved.

ADR required: no; default-value adjustment within the existing setting and served shell, with no new boundary. Files: Web_Server/serve.py and README.md, config.py, settings_appearance_defaults.py, both focused test files, this task.
<!-- SECTION:NOTES:END -->
