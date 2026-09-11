---
id: TASK-21601
title: Establish tldw-pydle fork provenance and isolated package
status: To Do
assignee: []
created_date: '2026-08-23 23:23'
updated_date: '2026-09-10 12:00'
labels:
  - irc
  - network-chat
  - dependency
  - packaging
dependencies: []
references:
  - backlog/decisions/148-network-chat-ircv3-and-tldw-pydle-boundary.md
  - backlog/decisions/149-network-chat-handoff-reliability-amendments.md
documentation:
  - Docs/superpowers/plans/2026-09-10-network-chat-pto-handoff.md
  - Docs/superpowers/plans/2026-08-23-tldw-pydle-first-release.md
  - Docs/superpowers/specs/2026-08-23-network-chat-ircv3-and-tldw-pydle-design.md
priority: high
type: task
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create a provenance-preserving, separately installable tldw-pydle foundation so
later reliability work has one stable repository, namespace, license boundary,
and reproducible package artifact shared by approved tldw client roles.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The fork retains the complete canonical Codeberg history through upstream commit `4efcc3b5096536668dfe772461f19a17f1ddd84e`, records that exact base in a machine-readable location, and retains an `upstream` remote plus a `develop` default branch.
- [ ] #2 Built metadata identifies distribution `tldw-pydle`, import package `tldw_pydle`, version `1.1.0.post1`, and Python 3.11 or newer without installing or shadowing a top-level `pydle` package.
- [ ] #3 The upstream BSD-3-Clause license text is preserved verbatim and both wheel and sdist contain a fork NOTICE identifying the canonical source, base commit, and tldw modification boundary.
- [ ] #4 The namespace migration covers code, tests, entry points, examples, and project metadata, and automated guards detect stale runtime `import pydle` references or upstream console-script names outside explicit provenance material.
- [ ] #5 Wheel and sdist build reproducibly, their members are inspected directly, and each artifact installs and imports successfully in a clean Python 3.11 environment alongside upstream pydle without collision.
- [ ] #6 The initial downstream commit series separates provenance/notice work from the mechanical namespace migration and contains no protocol-behavior change.
- [ ] #7 Fork governance documents the patch taxonomy, upstream synchronization procedure, release blocking policy, and quarterly/release/security review triggers from ADR-148.
<!-- AC:END -->

## Handoff

Read the [PTO entry point](../../Docs/superpowers/plans/2026-09-10-network-chat-pto-handoff.md) before claiming this task. Implementation lives in the separate fork; this record stays in Chatbook. ADR-148 records the approved boundary; ADR-149 contains the proposed review corrections. Add an Implementation Plan only after moving this task to In Progress. No implementation or release evidence is claimed by this documentation PR.
