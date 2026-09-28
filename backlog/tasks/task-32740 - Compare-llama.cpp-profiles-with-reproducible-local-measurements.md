---
id: TASK-32740
title: Compare llama.cpp profiles with reproducible local measurements
status: To Do
assignee: []
created_date: '2026-09-17 17:35'
updated_date: '2026-09-17 17:35'
labels:
  - llamacpp
  - catapult-review
  - tuning
  - feature
dependencies:
  - TASK-32728
  - TASK-32738
documentation:
  - Docs/superpowers/plans/2026-09-17-llamacpp-management-roadmap.md
  - Docs/superpowers/reviews/2026-09-17-catapult-llamacpp-management-review.md
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Let users judge two configurations using measured latency and throughput instead of assuming every tuning flag makes inference faster.

### Scope

An explicit bounded A/B workflow on a Chatbook-owned local runtime using the approved fixed benign prompt protocol and private result retention. Users choose two tuning drafts/profiles; launch and cleanup reuse exact process ownership.

### Explicit exclusions

No production-conversation prompts, external/shared-server benchmarking, automatic replacement of a running user server, optimizer, leaderboard or default speculative-decoding enablement.

### Architecture gate

ADR required: yes, supplied by the TASK-32738 decision. ADR path: N/A until that decision is published; execution must link its canonical path. Reason: measurement semantics, private durable results and run ownership.

### Affected areas

- `tldw_chatbook/LLM_Management/llamacpp_profile_comparison.py (new)`
- `tldw_chatbook/LLM_Management/llamacpp_comparison_store.py (new)`
- `tldw_chatbook/LLM_Management/llamacpp_profiles.py`
- `tldw_chatbook/LLM_Management/llamacpp_observations.py (new)`
- `tldw_chatbook/Event_Handlers/LLM_Management_Events/server_lifecycle.py`
- `tldw_chatbook/UI/LLM_Management/llamacpp_setup_view.py`

### Validation contract

Primary targeted run after implementing the scoped behavior:

```bash
.venv/bin/python -m pytest -q Tests/LLM_Management/test_llamacpp_profile_comparison.py Tests/LLM_Management/test_llamacpp_comparison_store.py Tests/UI/test_llamacpp_profile_comparison.py
```

New test paths are planned deliverables. Run Ruff on changed Python files, `git diff --check`, and mounted/CSS governance checks for affected UI. Live cases are opt-in with isolated state and explicit assets; skips are not qualification. No full suite without user opt-in.

### Execution boundary

This is a scoped To Do ticket. Read its dependencies and the linked roadmap before starting. Move it to In Progress, then add its detailed Implementation Plan (including the actual governing ADR path) before changing application code. Acceptance criteria and evidence govern completion; this planning pass authorizes no automatic execution.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Comparison starts only by explicit user action while no conflicting user-owned run is active; cancellation stops only the exact comparison process and settles its leases.
- [ ] #2 Both configurations use the approved equal workload/warmup/repetition/cache protocol and report time to first token, prompt/decode throughput and variation only where actually measurable.
- [ ] #3 Results include exact runtime/model/profile/effective settings and cache provenance; failed, unequal-workload, missing-counter and interrupted runs cannot rank a winner.
- [ ] #4 Retention/deletion obey the approved bounded private schema and omit external paths, credentials, real chat content and hardware identifiers.
- [ ] #5 Deterministic metric/lifecycle/store and mounted tests pass; isolated real A/B evidence exercises the production workflow and preserves uncertainty rather than promising a speedup.
<!-- AC:END -->
