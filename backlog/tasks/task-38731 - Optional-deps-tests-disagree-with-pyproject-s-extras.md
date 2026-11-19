---
id: TASK-38731
title: "Optional-deps tests disagree with pyproject's extras"
status: To Do
created_date: 2026-11-04 02:52
labels:
- follow-up
- tests
- packaging
priority: low
references:
- Tests/Utils/test_optional_deps.py
- pyproject.toml
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
On dev d204ddce70 (2026-11-03), `Tests/Utils/test_optional_deps.py::test_optional_feature_metadata_covers_pyproject_extras` and `::test_subscriptions_extra_declares_beautifulsoup_dependency` fail when run alone. Either the optional-feature metadata or pyproject.toml's extras have drifted.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Both tests pass on dev, with whichever side drifted brought back in line
- [ ] #2 The task notes record which extras or features changed
<!-- AC:END -->
