---
id: TASK-32738
title: Define llama.cpp launch fitting and profile comparison evidence
status: To Do
assignee: []
created_date: '2026-09-17 17:34'
updated_date: '2026-09-17 17:34'
labels:
  - llamacpp
  - catapult-review
  - tuning
  - design
dependencies:
  - TASK-32727
  - TASK-32733
documentation:
  - Docs/superpowers/plans/2026-09-17-llamacpp-management-roadmap.md
  - Docs/superpowers/reviews/2026-09-17-catapult-llamacpp-management-review.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Set clear limits on launch advice and repeatable measurements before exposing fitting controls or durable comparison results.

### Scope

One ADR and focused tuning spec covering runtime-assisted fit, observable effective settings, snapshot eligibility and explicit fixed-prompt A/B comparison provenance/retention.

### Explicit exclusions

No replacement of ADR-080 download scenarios, summed GPU memory, opaque automatic tuning, default speculative decoding or broad benchmarking framework.

### Architecture gate

ADR required: yes. ADR path: N/A for this scoping record; publish the canonical decision before dependent code. Reason: runtime-specific fitting semantics and durable benchmark evidence are new contracts.

### Affected areas

- `backlog/decisions/`
- `Docs/superpowers/specs/`
- `tldw_chatbook/Model_Artifacts/machine_memory.py`
- `tldw_chatbook/LLM_Management/snapshot_admission.py`
- `tldw_chatbook/LLM_Management/llamacpp_observations.py (new)`

### Validation contract

Review the ADR/spec against every acceptance criterion and the existing ownership ADRs. Check Markdown links and dependency ordering. The focused executable plan must name concrete module interfaces, test paths/commands and live-evidence gates. This design-only ticket does not need application tests.

### Execution boundary

This is a scoped To Do ticket. Read its dependencies and the linked roadmap before starting. Move it to In Progress, then add its detailed Implementation Plan (including the actual governing ADR path) before changing application code. Acceptance criteria and evidence govern completion; this planning pass authorizes no automatic execution.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The ADR separates provider-neutral advisory RAM scenarios from selected model/runtime/context/cache/slots/device launch guidance; unified memory is counted once and GPU devices stay separate.
- [ ] #2 Capability-gated fit semantics, user opt-in, configured-versus-effective values and manual fallback are specified; missing effective compatibility values disable snapshot Save/Restore.
- [ ] #3 Comparison protocol fixes a benign prompt, token budget, warmup/repetition count, cache conditions, run order and cancellation/ownership rules, with metric definitions and unavailable-evidence behavior.
- [ ] #4 A bounded private result schema records exact runtime/model/profile/effective-settings provenance, cache conditions and timestamps without conversation content, credentials, persistent external paths or hardware identifiers.
- [ ] #5 A focused executable plan sets interfaces, storage/retention/deletion, bias controls and targeted/live acceptance evidence; speculation stays an optional measured experiment, never an automatic faster preset.
<!-- AC:END -->
