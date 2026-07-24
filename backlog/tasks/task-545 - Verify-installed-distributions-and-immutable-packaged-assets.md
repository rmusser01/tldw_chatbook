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
- [ ] #1 Exactly one fresh sdist and wheel pass automated required-content and forbidden-content verification from an isolated temporary source and build tree
- [ ] #2 The wheel contains the compiled CSS bundle, RAG pipeline config, built-in chunking templates, eval configuration, and vendored Apache and MIT license files while sdist-only, example, test, cache, and OS files remain excluded
- [ ] #3 An isolated wheel installation resolves tldw_chatbook from the installed target and runtime loaders consume the packaged configs and templates
- [ ] #4 Installed tldw-cli --help and tldw-serve --help exit successfully with private state contained under a temporary root
- [ ] #5 Installed entry points and the application factory leave the complete installed-target file inventory and content hashes unchanged and do not attempt CSS rebuilds
- [ ] #6 Regression coverage runs through the integration test gate without relying on source-checkout imports or network dependency resolution
- [ ] #7 Built metadata uses the current AGPL-3.0-or-later SPDX license expression, declares LICENSE, and avoids the legacy license-table deprecation
<!-- AC:END -->
