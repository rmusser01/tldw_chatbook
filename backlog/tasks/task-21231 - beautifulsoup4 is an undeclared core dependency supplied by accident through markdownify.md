---
id: TASK-21231
title: >-
  beautifulsoup4 is an undeclared core dependency supplied by accident through
  markdownify
status: Done
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

## Implementation Plan

1. Record the decision as an ADR (dependency declaration is an ADR-required
   choice): bs4 is CORE, because `markdownify` (already core) transitively
   pins it and every base install has it.
2. Declare `beautifulsoup4` in `[project] dependencies`; remove the redundant
   re-declarations from the `websearch`, `ebook`, `subscriptions`, and `web`
   extras.
3. Update the TASK-21104 comment in `monitoring_engine.py` to describe the
   guard-as-safety-net reality.
4. Re-run `Tests/Packaging/test_extras_import_closure.py`.

## Implementation Notes (2026-10-11, dev `a3c47c6c72`)

- **Decision: CORE**, recorded as `backlog/decisions/225-beautifulsoup4-core-dependency.md`
  (ADR-225). Rationale: markdownify's pin already puts bs4 on every base
  install; declaring it makes that reliance explicit and testable, and a
  future markdownify dropping the pin can no longer silently change what a
  base install can do. The TASK-21104 lazy guard is kept unchanged as a
  safety net for damaged/partial installs.
- **pyproject.toml**: `beautifulsoup4` added to `[project] dependencies` with
  an ADR-225 comment; the four extras stopped re-declaring it. The
  `subscriptions` extras block comment updated ("bs4 ... a core dependency,
  ADR-225"). `markdownify` re-declarations inside `ebook`/`web` extras are
  likewise redundant now but are out of this task's scope.
- **Evidence**: `Tests/Packaging/test_extras_import_closure.py` 4 passed;
  TOML parses with core = True and no extras carrying bs4. AC#2 (genuinely
  bs4-less base env) is moot under the "core" branch -- the guard's
  not-installed branch remains as defense for broken installs, per the ADR's
  alternatives section.
