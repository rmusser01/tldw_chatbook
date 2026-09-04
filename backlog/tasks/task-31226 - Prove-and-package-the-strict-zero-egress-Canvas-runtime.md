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
- [x] #1 A reviewed, version-pinned WebAssembly ECMAScript engine is reproducibly bundled with license and integrity metadata
- [x] #2 Complete HTML documents compile into a versioned render plan without handing untrusted markup or scripts to native browser evaluation sinks
- [x] #3 Generated scripts can use the documented V1 DOM, event, timer, SVG, and bridge-request facade inside a bounded worker
- [x] #4 Unsupported markup, CSS, browser APIs, and runtime profiles fail closed with bounded compatibility diagnostics
- [x] #5 CPU, memory, stack, timer, listener, job, DOM-patch, and mutation-rate limits terminate or refuse abusive documents without freezing the trusted shell
- [x] #6 A real-browser adversarial suite observes zero generated-code egress for computed URLs, navigation, forms, CSS/SVG, redirects, beacons, workers, and downloads
- [x] #7 Generated JavaScript never executes in a native browser realm and no fallback can enable native execution
- [x] #8 Focused unit, property, package-wheel, and browser tests pass with runtime assets available offline
- [x] #9 Runtime compatibility and security limitations are documented for Canvas authors and model tool guidance
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

## Implementation Notes

Implemented the compiler/runtime security foundation behind the disabled Canvas
seam. The packaged engine is the MIT-licensed `quickjs-emscripten-core@0.32.0`
plus `@jitl/quickjs-singlefile-browser-release-sync@0.32.0` and
`@jitl/quickjs-ffi-types@0.32.0`, reproducibly bundled with
`esbuild-wasm@0.25.9`; the manifest now integrity-covers the engine, worker,
renderer, and notices. Prior delivery evidence includes deterministic archive
regeneration and offline installed-wheel loading.

The browser runtime creates a fresh 32 MiB / 512 KiB QuickJS runtime/context in
a dedicated worker, with 250 ms startup and 50 ms event interrupts, 100 pending
jobs, 64 timers, 100 timer firings/second, 500 listeners, a 100-event queue,
1,000 patches/operation, 2,000 mutations/second, and 16 typed bridge requests
per operation / 32 per second. The trusted renderer owns
DOM/CSSOM construction, passive-image object URLs, typed bridge forwarding, and
worker termination. It runs in an opaque `sandbox="allow-scripts"` iframe with
the ADR-115 CSP and no native-JavaScript fallback.

Passive images are limited to 64 static PNG/JPEG/GIF/WebP assets, 1 MiB each / 4
MiB aggregate encoded bytes, 4,096 per dimension, 4,194,304 pixels each /
16,777,216 pixels aggregate, and a one-second-per-image / three-second-total
native decode deadline. Renderer-side container parsing rejects malformed or
animated assets and decoded dimensions must match metadata before an object URL
is exposed.

The real Chromium 145.0.7632.6 corpus covers literal/computed URLs, redirects,
resource attributes, beacons, media/font/CSS, popups/workers, DOM clobbering,
prototype pollution, encoded CSS, active SVG, forbidden browser capabilities,
blob/data navigation, native-download attempts, bridge spoofing, syntax errors,
memory/stack/CPU/jobs/timers/listeners/patch/mutation/JSON limits, image
signature/dimension/pixel/frame/decode/count boundaries, and event storms. All
four supported static image types decoded successfully in the benign fixture.
Every generated phase recorded zero HTTP(S), WebSocket, navigation,
popup, download, or worker observations and the owned egress server received
zero requests. Native shell/renderer/worker sentinels were unchanged. A native
CSP control probe observed its image attempt in browser instrumentation but the
egress server received nothing; parent DOM, storage, and inline native script
were blocked. Firefox and WebKit were not installed and are explicit skips.

Focused verification:
`pytest Tests/Canvas/test_compiler.py Tests/Canvas/test_runtime_assets.py Tests/Canvas/browser/test_canvas_zero_egress.py -q`
reported `91 passed, 3 skipped, 1 pre-existing dependency warning in 9.43s`.
Compatibility and exact budgets are documented in
`Docs/Canvas/V1_RUNTIME_COMPATIBILITY.md`; architecture and qualification
results are recorded in ADR-115. Detailed RED/GREEN and request evidence is in
`.superpowers/sdd/2026-09-03-chatbook-canvas-implementation/task-1.4-report.md`.

The task intentionally remains **In Progress** until the delivery controller's
independent security-focused review is clean. No Canvas product tool, gateway,
persistence, confirmation effect, or UI was enabled in this delivery.
