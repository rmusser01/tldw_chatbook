---
id: TASK-16319
title: Bundle Samira character and full Visual Identity reaction pack
status: To Do
assignee: []
created_date: '2026-08-15 05:23'
labels: []
dependencies: []
references:
  - >-
    Docs/superpowers/specs/2026-08-14-task-16319-samira-builtin-visual-identity-pack-design.md
  - backlog/decisions/067-bundled-samira-visual-identity-pack.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Ship Samira "Sammy" Vadem as an included public character card and demonstrate character reactions through a complete, server-aligned Visual Identity expression pack without changing the default assistant or default persona.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Fresh and upgraded profiles receive one searchable Samira character card with the approved portrait, sanitized public metadata, and no VademHQ references.
- [ ] #2 Samira has a bound immutable Visual Identity pack containing the exact 28 standard GoEmotions labels plus thinking, speaking, and error companions, with deterministic server-aligned expression keys and fallbacks.
- [ ] #3 Users can browse and lazily preview the complete pack and manually select or clear a session-local reaction in Console without automatic emotion classification.
- [ ] #4 Existing idle, thinking, speaking, and error behavior and legacy character expression images continue to work through documented fallback behavior.
- [ ] #5 Seeding is idempotent and restart-safe across fresh installs, upgrades, collisions, renames, customization, deletion, and partial pack failure, and never overwrites or resurrects user data.
- [ ] #6 The JSON card and PNG-embedded V2 card are equivalent, every reaction is a valid 1024x1024 WebP, all bundled assets declare AGPL-3.0-or-later, and the installed wheel and sdist contain the verified bounded asset inventory.
- [ ] #7 Local persistence and normalization match the pinned tldw_server development contract while local IDs and message override state remain explicitly non-syncable until a later authenticated sync contract.
- [ ] #8 Automated database, resolver, packaging, and Textual UI tests plus real installed-artifact and live TUI verification pass.
<!-- AC:END -->
