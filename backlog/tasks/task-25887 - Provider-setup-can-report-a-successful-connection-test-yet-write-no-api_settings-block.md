---
id: TASK-25887
title: >-
  Provider setup can report a successful connection test yet write no
  api_settings block
status: To Do
assignee: []
created_date: '2026-08-31 20:05'
labels:
  - console
  - ux-review
  - follow-up
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
During the 2026-08-30 Console review a first-run setup completed for a local provider whose connection test had just reported success, and the finished config contained no api_settings entry for it at all -- the summary screen reported 'no credentials or saved endpoint' for the provider it had just verified. TASK-25817 originally attributed this to the resume checkpoint dropping the endpoint; that was traced to code and disproved (the checkpoint is not the applied configuration, and the endpoint is deliberately scrubbed from step memory). The commit path itself was never examined, so the behaviour that produced an unusable configuration is still unexplained.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Completing the provider step for a local provider writes an api_settings entry that can reach that provider
- [ ] #2 A successful connection test cannot be followed by a summary reporting the same provider as unconfigured
- [ ] #3 The conditions under which provider commit is skipped are covered by a test
<!-- AC:END -->
