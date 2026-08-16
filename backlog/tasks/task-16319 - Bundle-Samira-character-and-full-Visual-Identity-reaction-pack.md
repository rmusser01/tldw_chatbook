---
id: TASK-16319
title: Bundle Samira character and full Visual Identity reaction pack
status: Done
assignee: []
created_date: '2026-08-15 05:23'
updated_date: '2026-08-16 15:01'
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
Ship Samira "Sammy" Vadem as an included public character card and demonstrate character reactions through a complete, server-aligned Visual Identity expression pack without changing the default assistant or default persona. This is the umbrella task for TASK-16319.1 through TASK-16319.3.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Fresh and upgraded profiles receive one searchable Samira character card with the approved portrait, sanitized public metadata, and no VademHQ references.
- [x] #2 Samira has a bound immutable Visual Identity pack containing the exact 28 standard GoEmotions labels plus thinking, speaking, and error companions, with deterministic server-aligned expression keys and fallbacks.
- [x] #3 Users can browse and lazily preview the complete pack and manually select or clear a session-local reaction in Console without automatic emotion classification.
- [x] #4 Existing idle, thinking, speaking, and error behavior and legacy character expression images continue to work through documented fallback behavior.
- [x] #5 Seeding is idempotent and restart-safe across fresh installs, upgrades, collisions, renames, customization, deletion, and partial pack failure, and never overwrites or resurrects user data.
- [x] #6 The JSON card and PNG-embedded V2 card are equivalent, every reaction is a valid 1024x1024 WebP, all bundled assets declare AGPL-3.0-or-later, and the installed wheel and sdist contain the verified bounded asset inventory.
- [x] #7 Local persistence and normalization match the pinned tldw_server development contract while local IDs and message override state remain explicitly non-syncable until a later authenticated sync contract.
- [x] #8 Automated database, resolver, packaging, and Textual UI tests plus real installed-artifact and live TUI verification pass.
- [x] #9 Users can stage replacement or generated reactions for a bound pack; editing a built-in pack creates a private copy, and one Save creates one immutable version while cancellation or failure leaves the active version unchanged.
<!-- AC:END -->

## Implementation Plan

1. TASK-16319.1 — add the server-aligned local schema, immutable asset contract, bundled Samira card/reactions, and idempotent seeding.
2. TASK-16319.2 — resolve Visual Identity reactions in Console with fallback compatibility, lazy manual selection, and race-safe cache identities.
3. TASK-16319.3 — add Personas browsing and immutable profile-owned copy-on-write authoring, then complete isolated release verification.

ADR required: yes
ADR path: `backlog/decisions/067-bundled-samira-visual-identity-pack.md`
Reason: ADR-067 governs the schema, package/profile ownership, fallback, publication, and future-sync boundaries implemented by all three children.

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Completed TASK-16319.1-.3 under ADR-067. TASK-16319.1 delivered sanitized equivalent JSON/PNG cards, the exact 31-reaction AGPL-3.0-or-later package, server-aligned local persistence, and idempotent create-only seeding. TASK-16319.2 delivered deterministic resolver fallbacks, automatic idle/thinking/speaking/error behavior, lazy session-local Console selection/Clear, and race-safe cache invalidation. TASK-16319.3 delivered metadata-only Personas browsing and immutable profile-owned copy-on-write Replace/Generate/Clear/Save/Cancel authoring.
- Defaults remain unchanged; the delivery adds no default Persona, automatic classifier, durable reaction replay, server sync implementation, provider abstraction, or separate Persona Visual Packs system. Local identities and manual overrides retain the ADR-067 non-syncable boundary.
- Isolated evidence covered touched DB/migrations, manifest/contracts/lifecycle/resolution/publication, Console and Personas UIs, legacy behavior, CSS/architecture/diagnostic inventories, privacy/failure/orphan recovery, wheel/sdist and installed-resource behavior, and production-shaped Pilot UAT. The exact release inventory is 35 wheel files / 37 sdist entries (including two directories), with one fresh searchable Samira, unchanged Default Assistant, and 31 resolvable assets.
- Every reaction is a visually reviewed 1024x1024 WebP; all 31 total 3,366,092 bytes (largest 130,864), retain consistent identity/composition, and are content-safe. The labeled contact sheet is `/tmp/samira-reactions-contact-sheet-final.jpg` (SHA-256 `fcdd11e56d19a607ff850dd3a90b83b92383558c04aba5dc6c3e6e6f1bc3dc5f`). License/provenance audits found AGPL-3.0-or-later throughout and zero case-insensitive VademHQ references in source or installed assets.
- Verification used explicit isolated HOME/XDG/config/data paths. Real config and data fingerprints remained byte-identical before/after (`f1c67381c94b54519ab29958832fe88a07d2146cec1adc32d9b342e7c772ff8a` and `2d02afe67ebed4509c77b7545f92ec660bf4d8c997bc06dd2046cafbf94abdb7`). No additional user documentation or lessons entry was warranted.
- Verification scope deviation: at user direction, the full repository suite was not run; verification was limited to modified/touched components.
<!-- SECTION:NOTES:END -->
