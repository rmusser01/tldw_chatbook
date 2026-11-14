---
id: TASK-38736
title: "An installed wheel's tldw-serve crashes at import: the mobile static assets are not packaged"
status: To Do
created_date: 2026-11-04 03:10
labels:
- follow-up
- packaging
- web-server
- bug
priority: high
references:
- Tests/Packaging/test_installed_distribution.py
- tldw_chatbook/Web_Server/serve.py
- pyproject.toml
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
On dev d204ddce70 (2026-11-03), `Tests/Packaging/test_installed_distribution.py::test_installed_wheel_loaders_entry_points_and_assets_are_immutable` builds and installs the wheel, then runs `tldw-serve --help`, which exits 1. `Web_Server/serve.py:178` reads `_MOBILE_STATIC_ROOT.joinpath("index.html")` at import time, and the wheel does not contain `tldw_chatbook/Web_Server/static/mobile/`, so the import raises FileNotFoundError. Anyone running the served UI from an installed package (rather than a source checkout) cannot start it. It was already red on 2026-10-31.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The built wheel contains every file under tldw_chatbook/Web_Server/static/mobile/ that serve.py and the mobile routes read
- [ ] #2 `tldw-serve --help` exits 0 from an installed wheel, and the packaging test passes on dev
- [ ] #3 serve.py no longer reads static files at import time, or the task notes say why that read must stay
<!-- AC:END -->
