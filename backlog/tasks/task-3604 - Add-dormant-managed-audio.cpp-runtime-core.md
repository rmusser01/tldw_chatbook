---
id: TASK-3604
title: Add dormant managed audio.cpp runtime core
status: In Progress
assignee:
  - '@codex'
created_date: '2026-08-08 15:02'
updated_date: '2026-08-08 15:23'
labels:
  - tts
  - audio-cpp
  - backend
  - lifecycle
dependencies:
  - TASK-3602
references:
  - backlog/decisions/023-tts-adapter-registry-and-audio-cpp-runtime-boundary.md
  - backlog/decisions/039-global-and-studio-tts-settings-ownership.md
documentation:
  - Docs/superpowers/specs/2026-08-02-audio-cpp-managed-lifecycle-design.md
  - >-
    Docs/superpowers/plans/2026-08-08-task-3604-audio-cpp-managed-runtime-core.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the provider-specific managed audio.cpp runtime core behind the existing native adapter so Chatbook can validate and supervise one user-supplied loopback audiocpp_server process through the current registry and service lifecycle without exposing Managed mode in the UI or changing External and legacy-provider behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 ML-AC-001 ML-AC-002 ML-AC-003: Existing configurations remain External by default and active-mode projection validates bounded Managed timing executable and strict loopback server.json inputs, including duplicate-key and non-finite-constant rejection, without launching or probing a process.
- [ ] #2 ML-AC-004 ML-AC-005: A deliberate managed operation lazily starts exactly one app-owned child and concurrent first use shares one retained startup that is not cancelled by an individual waiter.
- [ ] #3 ML-AC-006: Launch uses the exact user-approved executable and server.json without a shell, performs fail-closed port preflight and complete readiness validation, assigns one exit monitor as sole child reaper, and removes the exact child plus every generation task and parent pipe on pre-Running failure.
- [ ] #4 ML-AC-007 ML-AC-008: Generation-bound non-overlapping shared health probes and exit supervision report truthful Running Unhealthy and Unavailable states with a safe stable last failure, without automatic restart, and only a later deliberate operation may start one replacement from the latest eligible saved mode.
- [ ] #5 ML-AC-009 ML-AC-010 ML-AC-016: Every save atomically supersedes or clears older staging, and explicit restart shutdown or External application drains admitted work before changing or stopping only the owned child; an obsolete staged Managed mapping can never reappear after a newer External save.
- [ ] #6 ML-AC-011: Application shutdown uses one deadline, drains registry leases before child termination, joins any in-flight generation health probe before its HTTP client closes, is bounded and idempotent, and cannot report completion while an owned child generation-bound HTTP client startup task health task exit monitor or output drain remains.
- [ ] #7 ML-AC-012: Child output is continuously drained into a bounded memory-only best-effort-sanitized diagnostic snapshot with truthful truncation and no propagation to general logs configuration artifacts or persistence; drain failure and inherited descendant pipe descriptors cannot strand a child or cleanup task.
- [ ] #8 ML-AC-014: Managed catalog discovery and synthesis reuse the existing native audio.cpp HTTP contract and return one bounded structurally validated complete WAV item through the asynchronous response interface.
- [ ] #9 ML-AC-001 ML-AC-017: External audio.cpp and every legacy bridge provider retain their existing behavior, passive capability reads never launch Managed mode, every deliberate service entry path uses one preparation seam, and deterministic CI requires no audio.cpp binary models download external network or audio hardware.
- [ ] #10 Slice boundary: The runtime core is reachable through service APIs and tests but no Managed selector lifecycle control diagnostic panel or other user-visible Slice 5 behavior is added.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Extend AudioCppConfig with backward-compatible active-mode projection and strict bounded Managed launch validation.
2. Add the fixed child-environment allowlist, credential-name inventory, and bounded sanitized in-memory diagnostics.
3. Implement one app-scoped AudioCppSupervisor with lifecycle-epoch launch fencing, shared lazy startup and health probes, sole process reaping, exact-child ownership, bounded pipe cleanup, readiness, exit, restart/stop, and terminal cleanup state machines.
4. Extend TTSAdapterRegistry with atomically superseded latest-wins staged configuration snapshots, retryable failed transitions, and an exclusive transition that publishes Draining, rejects new leases, drains admitted work, then performs lifecycle cleanup and optional promotion.
5. Bind every deliberate native audio.cpp adapter/service entry path to one provider-selection and preparation seam while keeping passive observation launch-free and preserving External, legacy, and complete-WAV behavior.
6. Prove the real subprocess boundary with a controlled local fixture and add negative UI regressions showing Slice 4 remains dormant.
7. Update the TTS module guide; run focused/full tests, Ruff, mypy, compileall, privacy/boundary checks, and complete task evidence.

ADR required: yes
ADR paths: backlog/decisions/023-tts-adapter-registry-and-audio-cpp-runtime-boundary.md; backlog/decisions/039-global-and-studio-tts-settings-ownership.md
Reason: this task implements the already-approved provider-runtime, process-ownership, staged-configuration, shutdown, and security boundaries; no new architectural decision is introduced.
<!-- SECTION:PLAN:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Focused configuration supervisor registry service adapter privacy shutdown staging-race passive-no-launch and repeated-subprocess tests pass.
- [ ] #2 The full Tests/TTS suite plus affected briefing spoken-feedback character-UAT and settings-ownership regressions pass or exact unchanged baseline failures are documented with matching evidence.
- [ ] #3 Ruff formatting compileall scoped typing exact operation-code boundary searches and cumulative origin/dev branch-diff checks pass.
- [ ] #4 ADR-023 ADR-039 the managed-lifecycle design the TTS module guide and task notes remain traceable and current.
- [ ] #5 Self-review confirms zero hidden launch automatic restart process adoption arbitrary arguments persistent diagnostics or Slice 5 UI leakage.
<!-- DOD:END -->
