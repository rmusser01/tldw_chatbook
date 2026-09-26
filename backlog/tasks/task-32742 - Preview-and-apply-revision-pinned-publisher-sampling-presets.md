---
id: TASK-32742
title: Preview and apply revision-pinned publisher sampling presets
status: To Do
assignee: []
created_date: '2026-09-17 17:36'
updated_date: '2026-09-17 17:37'
labels:
  - llamacpp
  - catapult-review
  - presets
  - feature
dependencies:
  - TASK-32741
documentation:
  - Docs/superpowers/plans/2026-09-17-llamacpp-management-roadmap.md
  - Docs/superpowers/reviews/2026-09-17-catapult-llamacpp-management-review.md
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Reduce manual copying of publisher sampling recommendations while keeping section choice and every settings change reviewable.

### Scope

Exact-revision bounded presets.ini retrieval, strict section-aware parsing, an allowlisted settings diff and explicit application through existing Console settings with provenance.

### Explicit exclusions

No automatic main-branch fallback, auto-apply on download, command/executable/source settings, new launcher sampling fields or automatic durable defaults.

### Architecture gate

ADR required: yes, supplied by the TASK-32741 decision. ADR path: N/A until that decision is published; link the actual path before implementation. Reason: implement its remote-input and settings-ownership contract.

### Affected areas

- `tldw_chatbook/Model_Artifacts/remote_huggingface.py`
- `tldw_chatbook/LLM_Management/llamacpp_publisher_presets.py (new)`
- `tldw_chatbook/Chat/console_settings_apply.py`
- `tldw_chatbook/Chat/console_session_settings.py`
- `tldw_chatbook/Widgets/Console/console_settings_modal.py`

### Validation contract

Primary targeted run after implementing the scoped behavior:

```bash
.venv/bin/python -m pytest -q Tests/LLM_Management/test_llamacpp_publisher_presets.py Tests/Model_Artifacts/test_remote_huggingface.py Tests/Chat/test_console_settings_apply_store.py Tests/UI/test_llamacpp_publisher_presets.py
```

New test paths are planned deliverables. Run Ruff on changed Python files, `git diff --check`, and mounted/CSS governance checks for affected UI. Live cases are opt-in with isolated state and explicit assets; skips are not qualification. No full suite without user opt-in.

### Execution boundary

This is a scoped To Do ticket. Read its dependencies and the linked roadmap before starting. Move it to In Progress, then add its detailed Implementation Plan (including the actual governing ADR path) before changing application code. Acceptance criteria and evidence govern completion; this planning pass authorizes no automatic execution.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Retrieval uses the exact selected immutable revision and approved source/bounds; missing or malformed files leave settings unchanged and never fall back to main.
- [ ] #2 Distinct INI sections stay distinct; duplicate/default/interpolated/malformed/nonfinite/unknown values follow the documented strict policy and cannot inject commands, paths or network settings.
- [ ] #3 A preview identifies source revision, chosen section, accepted values and differences from explicit user overrides; cancel has no settings or persistence effects.
- [ ] #4 Apply uses the canonical Console transaction for the exact current provider/model; stale previews are refused, user overrides follow the approved precedence, and default persistence remains a separate ordinary action.
- [ ] #5 Parser/transport/transaction and mounted preview/apply/cancel tests pass, including restart/provenance behavior prescribed by the ADR; guides describe trust and ownership.
<!-- AC:END -->
