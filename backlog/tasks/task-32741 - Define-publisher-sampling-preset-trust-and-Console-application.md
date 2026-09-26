---
id: TASK-32741
title: Define publisher sampling preset trust and Console application
status: To Do
assignee: []
created_date: '2026-09-17 17:36'
updated_date: '2026-09-17 17:36'
labels:
  - llamacpp
  - catapult-review
  - presets
  - design
dependencies:
  - TASK-32722
  - TASK-32707
documentation:
  - Docs/superpowers/plans/2026-09-17-llamacpp-management-roadmap.md
  - Docs/superpowers/reviews/2026-09-17-catapult-llamacpp-management-review.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Decide how untrusted publisher recommendations can be reviewed and applied without overriding user choices or gaining launch authority.

### Scope

One ADR and focused spec for exact-revision presets.ini retrieval, section selection, bounded numeric allowlists, provenance, user overrides and Console application/persistence.

### Explicit exclusions

No remote code, arbitrary config import, server command/path settings, automatic apply on model download or replacement Console defaults owner.

### Architecture gate

ADR required: yes. ADR path: N/A for this scoping record; publish a canonical decision before dependent code. Reason: remote configuration input, provenance and precedence cross existing settings boundaries.

### Affected areas

- `backlog/decisions/`
- `Docs/superpowers/specs/`
- `tldw_chatbook/Model_Artifacts/remote_huggingface.py`
- `tldw_chatbook/Chat/console_session_settings.py`
- `tldw_chatbook/Chat/console_settings_apply.py`
- `tldw_chatbook/Chat/console_settings_defaults.py`

### Validation contract

Review the ADR/spec against every acceptance criterion and the existing ownership ADRs. Check Markdown links and dependency ordering. The focused executable plan must name concrete module interfaces, test paths/commands and live-evidence gates. This design-only ticket does not need application tests.

### Execution boundary

This is a scoped To Do ticket. Read its dependencies and the linked roadmap before starting. Move it to In Progress, then add its detailed Implementation Plan (including the actual governing ADR path) before changing application code. Acceptance criteria and evidence govern completion; this planning pass authorizes no automatic execution.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The ADR limits retrieval to an explicitly selected immutable repository revision, fixes size/deadline/section/key/numeric bounds and rejects executable/path/network/credential authority.
- [ ] #2 Section selection, duplicates/default inheritance/interpolation and unknown-key policies are deterministic; values from distinct presets cannot silently blend.
- [ ] #3 Preview/apply/cancel and imported recommendation versus explicit user override precedence are defined through the canonical Console settings transaction, including provider/model changes.
- [ ] #4 Persistence and provenance rules state exactly what survives restart; applying a session recommendation cannot silently become Make default or alter launch profiles.
- [ ] #5 A focused executable plan pins parser and application interfaces, allowed field mapping, malformed/stale/conflict tests and docs without adding an alternate settings surface.
<!-- AC:END -->
