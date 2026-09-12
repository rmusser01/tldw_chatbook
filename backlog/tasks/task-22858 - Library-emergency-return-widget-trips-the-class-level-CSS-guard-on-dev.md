---
id: TASK-22858
title: Library emergency-return widget trips the class-level CSS guard on dev
status: Done
assignee: []
created_date: '2026-08-27 02:14'
updated_date: '2026-08-27 04:23'
labels:
  - ci
  - css
  - library
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Upstream commit 6161bd1fe ('feat(library): add narrow emergency return path', 2026-08-26) added Widgets/Library/library_emergency_return.py::LibraryEmergencyReturn.DEFAULT_CSS, which Tests/UI/test_widget_css_consolidation.py::test_class_level_css_stays_within_the_allowlist bars: each class-level DEFAULT_CSS registers another stylesheet source against Textual's 64-entry parse cache, past which every first mount of an unseen class re-pays a ~150-450ms cold parse. Verified failing on pristine origin/dev in a clean worktree (not introduced by PR #2101, which hit the same guard for its own widget and fixed it via BUNDLED_CSS).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 test_class_level_css_stays_within_the_allowlist passes on dev
- [ ] #2 LibraryEmergencyReturn CSS either rides the bundle as BUNDLED_CSS (regenerated sheets committed) or is added to the allowlist with a recorded reason
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Fixed in the TASK-21150 follow-up branch: LibraryEmergencyReturn.DEFAULT_CSS -> BUNDLED_CSS (a plain string literal, so the default sanctioned route applied cleanly) with build_css.py rerun and the regenerated widget_defaults_self.tcss / bundle committed. Tests/UI/test_widget_css_consolidation.py is green again (33 passed); the widget's own behavior tests still pass, and preflight's CSS bundle-sync check confirms the sheets reproduce from source.
<!-- SECTION:NOTES:END -->
