---
id: TASK-31226
title: Prove and package the strict-zero-egress Canvas runtime
status: In Progress
assignee: []
created_date: '2026-09-03'
updated_date: '2026-09-03'
labels: [canvas, security, runtime]
dependencies: []
priority: high
---

## Description

Establish the trusted compiler and virtual JavaScript execution foundation that lets Canvas render interactive self-contained documents without giving generated code a native browser or network capability. This implements the release-blocking runtime boundary in ADR-115 before any Canvas tool or UI is exposed.

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A reviewed, version-pinned WebAssembly ECMAScript engine is reproducibly bundled with license and integrity metadata
- [ ] #2 Complete HTML documents compile into a versioned render plan without handing untrusted markup or scripts to native browser evaluation sinks
- [ ] #3 Generated scripts can use the documented V1 DOM, event, timer, SVG, and bridge-request facade inside a bounded worker
- [ ] #4 Unsupported markup, CSS, browser APIs, and runtime profiles fail closed with bounded compatibility diagnostics
- [ ] #5 CPU, memory, stack, timer, listener, job, DOM-patch, and mutation-rate limits terminate or refuse abusive documents without freezing the trusted shell
- [ ] #6 A real-browser adversarial suite observes zero generated-code egress for computed URLs, navigation, forms, CSS/SVG, redirects, beacons, workers, and downloads
- [ ] #7 Generated JavaScript never executes in a native browser realm and no fallback can enable native execution
- [ ] #8 Focused unit, property, package-wheel, and browser tests pass with runtime assets available offline
- [ ] #9 Runtime compatibility and security limitations are documented for Canvas authors and model tool guidance
<!-- AC:END -->

## Related Design

- `Docs/superpowers/specs/2026-09-03-chatbook-canvas-design.md`
- `Docs/superpowers/plans/2026-09-03-chatbook-canvas-implementation.md`
- `backlog/decisions/115-local-versioned-canvas-artifacts-and-browser-sandbox.md`

## Implementation Plan

- ADR required: yes
- ADR path: `backlog/decisions/115-local-versioned-canvas-artifacts-and-browser-sandbox.md`
- Reason: this task implements and records the runtime/dependency boundary already accepted by ADR-115; the chosen engine and packaging details will be added as an ADR addendum.

1. Define the versioned render-plan and bridge wire models plus shared hard-limit validators through strict red/green TDD.
2. Parse complete HTML/CSS into a closed allowlisted render plan and reject every native execution, navigation, and resource-fetch surface with bounded diagnostics.
3. Review, pin, reproducibly vendor, license, package, and integrity-check the selected WebAssembly ECMAScript runtime.
4. Build the worker-hosted virtual DOM and trusted patch renderer, then prove strict zero egress with adversarial real-browser tests.
5. Run the targeted unit/property/package/browser suites, complete a security-focused review, record evidence and limitations, and stop the rollout if zero egress is not demonstrated.
