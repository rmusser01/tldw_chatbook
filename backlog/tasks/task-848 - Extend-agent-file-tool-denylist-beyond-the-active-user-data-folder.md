---
id: TASK-848
title: Extend agent file-tool denylist beyond the active user data folder
status: To Do
assignee: []
created_date: '2026-07-27 02:36'
labels:
  - tools
  - security
  - follow-up
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The denylist refuses files sitting directly in get_user_data_dir(), which covers the app's own state. Under a deliberately widened sandbox root, chromadb/chroma.sqlite3 (plaintext chunks of the same conversations and notes ChaChaNotes.db protects) and sibling profile folders remain readable. Reviewed as a disclosure asymmetry rather than a permission-gate bypass -- skill trust manifests are HMAC+keyring authenticated and script grants are digest-pinned, so tampering fails closed. Filed from the PR #953 review.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Vector-store and sibling-profile paths are refused under a widened sandbox root,The default sandbox configuration still works end to end,A test pins both directions
<!-- AC:END -->
