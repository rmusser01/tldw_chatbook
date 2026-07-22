---
id: TASK-421
title: Skills trust - remediation guidance for non-reviewable quarantine states
status: Done
assignee:
  - '@claude'
created_date: '2026-07-21 15:19'
updated_date: '2026-07-21 16:49'
labels:
  - skills
  - ux
  - trust
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
P2 from the 2026-07-21 Skills UX/NNG review. quarantined_manifest_error and quarantined_unsupported_path are trust_blocked but excluded from review eligibility and not unlockable, so every trust button stays disabled and the panel offers no way forward. NNG heuristic 9 (help users recognize, diagnose, and recover from errors).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Both non-reviewable quarantine states render guidance explaining what happened and what to do next,The on-disk skill path is surfaced so the user can inspect or fix files externally,No trust state leaves the panel with zero enabled remediation or guidance,Covered by tests
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
New pure helper skill_trust_remediation_copy(trust_status, skill_path) in library_skills_canvas (re-exported via Widgets/Library): state-specific next-step guidance for the two trust_blocked-but-non-reviewable states - quarantined_manifest_error (inspect files + trust store at the named path; remove the trust store to start over via Set up skill trust) and quarantined_unsupported_path (open the named skill directory, remove/flatten unsupported paths, reopen to re-check); empty string for every state that already has in-panel remediation. Rendered as an always-present #library-skill-trust-remediation Static under the trust state line (empty when idle) so _render_library_skill_trust_panel patches it in place under the existing no-recompose contract. Screen computes the per-skill path via new _library_skill_on_disk_path (local_skills_service.skills_dir / selected name, defensive-empty when unavailable; copy helper falls back to generic wording). 2 new tests (pure helper covering both states + healthy-state emptiness, canvas render with path) watched fail first. Canvas 74 passed; Skills+state 142 passed.
<!-- SECTION:NOTES:END -->
