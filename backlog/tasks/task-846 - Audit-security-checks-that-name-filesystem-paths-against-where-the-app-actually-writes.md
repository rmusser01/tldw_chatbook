---
id: TASK-846
title: >-
  Audit security checks that name filesystem paths against where the app
  actually writes
status: Done
assignee: []
created_date: '2026-07-27 03:46'
updated_date: '2026-07-27 06:41'
labels:
  - security
  - tools
  - audit
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The agent file-tool denylist shipped broken on two branches: it hardcoded the permission store as ~/.config/tldw_cli/mcp_permissions.json while the app builds that file under get_user_data_dir(). The literal path never existed, so the check never matched once, and write_file was demonstrated overwriting the real permission store with global_default allow -- the exact gate bypass the denylist existed to prevent. Fifteen review passes confirmed 'credential and app-state paths are refused' without anyone checking that the paths named in the list were the paths the application uses. The fix derives the path from the app's own accessor and the regression test re-derives it the same way, so it cannot drift back to matching nothing. Other security checks in the codebase that name paths, config keys or database locations by literal may carry the same defect.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Every security-relevant check that names a filesystem path is inventoried,Each one is confirmed against the accessor the app really uses or corrected to derive from it,Tests for those checks derive their paths the way the app does rather than asserting literals,Any check found broken is recorded with what it failed to protect
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Audited 30 security-relevant checks that name filesystem paths, config keys or database locations by literal, and verified each against the accessor the application actually uses by resolving both sides on-disk. 27 were broken. Six Critical, including: all three config-encryption entry points writing to DEFAULT_CONFIG_PATH rather than the effective path, so 'Enable encryption' returns True while leaving the active profile's API key plaintext; detect_api_keys exact-matching six literals when every real key name is prefixed, so should_encrypt_config() never prompts; the skill script-grant store (which authorizes script execution) uncovered by the denylist; and two skills-trust containment checks that are true by construction. Findings and evidence in .superpowers/sdd/task-846-audit.md. Filed as tasks 851-862; recommendations 3 and 4 were folded into TASK-848 rather than duplicated, and recommendation 14's two test sites became task-862. TASK-848's fixes on this branch closed the two Critical denylist omissions.
<!-- SECTION:NOTES:END -->
