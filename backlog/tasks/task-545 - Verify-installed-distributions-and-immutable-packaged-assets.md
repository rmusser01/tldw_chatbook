---
id: TASK-545
title: Verify installed distributions and immutable packaged assets
status: In Progress
assignee:
  - '@codex'
created_date: '2026-07-24 17:40'
labels:
  - packaging
  - reliability
  - security
dependencies: []
references:
  - backlog/decisions/025-immutable-installed-distribution-assets.md
documentation:
  - Docs/superpowers/specs/2026-07-24-installed-distribution-integrity-design.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Prove the built distributions outside the source checkout, include every runtime-owned asset and vendored license, and prevent installed commands from rebuilding generated package files at runtime.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Built sdist and wheel artifacts pass automated manifest and content verification from a temporary source copy
- [ ] #2 The wheel contains the compiled CSS bundle, RAG pipeline config, built-in chunking templates, eval configuration, and vendored Apache and MIT license files
- [ ] #3 An isolated wheel installation resolves tldw_chatbook from the installed target and runtime loaders consume the packaged configs and templates
- [ ] #4 Installed tldw-cli --help and tldw-serve --help exit successfully with private state contained under a temporary root
- [ ] #5 Installed commands do not rebuild or write generated assets inside the installed package
- [ ] #6 Regression coverage runs through the integration test gate without relying on source-checkout imports or network dependency resolution
<!-- AC:END -->
