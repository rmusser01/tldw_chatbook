---
id: TASK-37270
title: Gate real project-chat journeys with an executable UAT pilot
status: In Progress
assignee:
  - '@codex'
created_date: '2026-10-28 23:58'
updated_date: '2026-10-29 06:33'
labels: []
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Prevent project-chat feature failures from passing component-only checks by exercising the real application journey and retaining actionable evidence. Publish a wider workflow catalog and an initial deterministic project-chat pilot.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The reviewed catalog defines user actions, independent success criteria, and evidence for core journeys while clearly labeling unimplemented coverage.
- [x] #2 An isolated real-app pilot creates a workspace, binds a fixture folder, starts a scoped chat through the UI, and reads actual project evidence through the production provider and tool path.
- [x] #3 Fenced, native, compact schema recovery, and wrong-family argument recovery variants validate the final assistant answer and durable run; negative controls reject false success.
- [x] #4 Only the owned loopback provider can be contacted; private profile isolation, bounded waits, and failure/cleanup artifacts are verified.
- [x] #5 The exact pilot variants are enrolled in an existing required CI lane with complete no-skip result checking and retained diagnostic artifacts after targeted qualification.
- [x] #6 Documentation records exact commands and measured platform coverage, distinguishing scripted Windows/Linux runs from unperformed live Fedora qualification.
- [x] #7 All four project-chat UAT variants pass on Windows with the original authority boundaries and bounded waits intact.
- [x] #8 Local compatible-provider stream control frames never enter assistant prose, while genuine content, tools, reasoning and errors retain their behavior.
- [x] #9 Strata and PolyStrata are supported through verified local-provider configuration, discovery and inference paths with focused compatibility coverage and honest live-model limitations.
- [ ] #10 Actionable PR3288 review findings are resolved and every required check passes on the final head.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no; existing ADR-069/174/241/253 govern the original pilot, ADR-002/006/012/066/179 govern external provider support, and ADR-126 authority remains intact. Follow the linked project-chat UAT plan. Diagnose Windows timing with native evidence, remove notification polling churn, and correct the proven Windows-only 180-second agent-budget exhaustion using bounded measured test budgets; qualify all four variants and document performance limits. Preserve real native checks and evidence oracles. Fix invisible SSE control checkpoints and Stop, add independently configured Strata/PolyStrata discovery/inference with metadata-gated tools, resolve review findings and keep PR3288 open until final-head required checks pass. No merge is authorized.

Final documentation-head CI exposed a credential-round-trip fixture opening settings before its checked presentation revision settled. Reproduce and trace the existing owner guard, reuse the established readiness synchronization if confirmed, and keep the original modal/unsaved-edit assertions. Rerun targeted regression tests and final-head required CI. ADR required: no; test setup repair within existing ownership contracts.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented the critical-journey catalog and real TldwCli project-chat pilot. Four scripted-model variants exercise actual workspace/binding/chat UI, instruction disclosure, built-in approval, provider serialization, file tools, SQLite ownership and painted/saved README evidence. Exact no-skip reporting and a real missing-evidence negative control run in the required CI lane.

Local SSE comments/control frames remain invisible while preserving cancellation checkpoints. Strata and PolyStrata have independent configuration, discovery, credentials, reasoning/tool metadata and local cost classification. Review fixes cover explicit-keyless credential selection, tool-round reasoning ownership and keep-alive-only Stop. Storage initialization waiters use existing state-change notifications instead of 10ms polling; native authority remains intact. The pilot waits for a distinct new chat and composer, and uses bounded Windows functional budgets after reproducing test-only agent-budget exhaustion.

Source head 3660a6caff: all four Windows journeys passed and their aggregate report verified exactly four successes with complete receipts. Body times: 743.657s fenced, 571.203s native, 493.250s compact recovery, 509.734s wrong family. The Windows missing-evidence control had exactly one intended failure, zero errors/skips, 399.400s parent time; verifier confirmed actual fs_read evidence caused rejection. Linux run 40817423013 passed 59 cases in 183.711s; its real negative control failed as intended in 38.888s. Both downloaded verifiers passed.

The prior required aggregate failed only diagnostic inventory drift. Statement-level review found one added log interpolating a raw rejected effort. Follow-up 133b18c2cf removes that unnecessary diagnostic; two regression cases failed before the fix, then all 48 Strata tests passed in 17.38s. Exact statements again match the committed inventory without regeneration. Earlier focused provider/thinking/pricing/ratchet, six-case streaming/Stop, native admission and 90 CI-contract checks passed. Scoped Ruff/format and task-metadata checks pass; no full-suite sweep.

Independent code/documentation review findings are addressed. Qodo is unavailable for workspace credits and CodeRabbit auto-review skips dev; neither is claimed as external approval. Required final-head CI remains the closeout gate. No merge is authorized.

Windows native validation remains slow: this is functional qualification, not a latency claim. Live Fedora/Python 3.14, real inference engines and native-terminal paint remain untested. Other catalog journeys remain planned. ADR required: no; existing ADR-069/174/241/253 govern the pilot, ADR-002/006/012/066/179 provider integration, and ADR-126 storage authority. Commands and evidence are in the linked UAT spec/plan; testing lessons record the consent, ownership and Stop-fixture traps.

Final source validation: required check and full Derived Artifacts run 40822404188 passed on 133b18c2cf3f029fc248987ca4a2108c0b4d6b13. Linux UAT rerun: 59 passed in 190.245s; its expected negative control and downloaded report verification passed. The complete local diagnostic inventory passed unchanged. No review threads are unresolved. The final documentation-only commit retains the same required CI gate.

The documentation-only fc6bb555c9 CI run40825594322 exposed a pre-existing credential fixture race: settings revision2 opened while a mount-time revision0 readiness read was pending. Both parameter cases reproduced on Windows; temporary diagnostics captured the unchanged exact owner and stale pending revision. The two real-app credential setups now reuse the existing bounded _sync_console_control_bar_now helper after direct settings replacement. Production guards and original modal/unsaved-edit assertions remain intact. Targeted native Windows verification: 10 passed,25 deselected in286.39s, covering all three credential cases and seven helper controls; independent review found no actionable issues. Ruff lint and diff checks pass; formatter differences are pre-existing. Final-head required CI is pending.
<!-- SECTION:NOTES:END -->
