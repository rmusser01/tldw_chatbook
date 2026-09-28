---
id: TASK-33161
title: Console auto-speak consent crashes on mount when its service is not ready
status: To Do
assignee: []
created_date: '2026-09-28 02:11'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The perf-guard run on PR #2860 failed in Tests/Performance/test_textual_css_fastpath.py::test_ancestor_scoped_bare_type_rule_count_is_a_ratchet with AttributeError: 'NoneType' object has no attribute 'subscribe_message_completed' / 'active_session_id'. The raise site is tldw_chatbook/Widgets/Console/console_auto_speak_consent.py, mount() (about line 216), during ChatScreen mount, where self._store_accessor() returns None and .subscribe_message_completed is called on it. It passes on dev, so it is order- or race-dependent: a candidate latent app crash.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The crash is reproduced or its trigger is explained
- [ ] #2 mount() tolerates a not-yet-ready service (self._store_accessor() returning None) without raising
- [ ] #3 A regression test covers a mount with the service not yet ready
<!-- AC:END -->
