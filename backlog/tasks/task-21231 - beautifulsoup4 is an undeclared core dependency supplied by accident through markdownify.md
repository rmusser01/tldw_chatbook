---
id: TASK-21231
title: >-
  beautifulsoup4 is an undeclared core dependency supplied by accident through
  markdownify
status: To Do
assignee: []
created_date: '2026-08-23'
labels:
  - packaging
  - dependencies
  - technical-debt
dependencies: []
priority: medium
---

## Description

Source: close-out of the 2026-08-22 holistic performance review burn-down; honesty note raised
during the TASK-21104 implementation. Base evidence doc:
`Docs/Design/2026-08-22-holistic-perf-review.md`.

TASK-21104 was filed because `beautifulsoup4` is imported at boot but declared only in extras,
so a base install could not import the app. The guard it added is correct, but the underlying
declaration is still wrong in the other direction. Verified on dev `b2b1e2e0d`:
`pyproject.toml` declares `markdownify` in `[project] dependencies` (line 70) while
`beautifulsoup4` appears only under the `websearch` (line 107), `ebook` (176),
`subscriptions` (331) and `web` (375) extras. `markdownify` transitively pins
`beautifulsoup4<5,>=4.9`, so every healthy base install has bs4 present — by accident, not by
declaration.

Two consequences. The guard's "bs4 is not installed" branch and the extras' "install this to
get bs4" promise are both untestable against a real base install, because a real base install
has bs4. And a future `markdownify` release that drops the pin silently changes what a base
install of this app can do, with nothing in the repo recording that we relied on it.

## Acceptance Criteria

- [ ] The repo records an explicit decision on whether `beautifulsoup4` is a core dependency or an optional one, and `pyproject.toml` matches that decision
- [ ] If bs4 stays optional, an environment built from the declared core dependencies genuinely lacks bs4, so the TASK-21104 guard's not-installed branch is reachable in a real environment
- [ ] If bs4 becomes core, it is declared in `[project] dependencies` and the extras stop re-declaring it as though it were their own
- [ ] `Tests/Packaging/test_extras_import_closure.py` still passes
