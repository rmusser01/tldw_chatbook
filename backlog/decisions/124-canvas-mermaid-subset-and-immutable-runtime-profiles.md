# ADR-124: Canvas Mermaid subset and immutable runtime profiles

Status: Proposed — written design awaiting user review
Date: 2026-09-06
Related Task: TASK-31933
Extends: [ADR-121](121-local-versioned-canvas-artifacts-and-browser-sandbox.md)

## Context

Canvas V1 provides immutable conversation-owned HTML revisions and strict
zero-egress QuickJS-WASM execution. The next requested library is Mermaid.
The [compatibility spike](../../Docs/Canvas/V2_MERMAID_SPIKE.md) found that full
and Tiny Mermaid bundles exceed current script limits and their renderer needs
native browser facilities. A small grammar-only prototype rendered two simple
examples inside the existing boundary. It did not qualify general diagrams,
portability or adversarial safety.

The current revision/archive model already preserves a bounded runtime-profile
identifier, including unknown profiles as inert data. A second dependency
database or executable archive format would add authority without solving a
demonstrated storage problem.

## Decision

1. Ship a version-pinned **Mermaid syntax subset with a Canvas-specific renderer**,
   initially acyclic TD/LR flowcharts and simple participant/message/note sequence
   diagrams. The [design spec](../../Docs/superpowers/specs/2026-09-06-chatbook-canvas-v2-mermaid-design.md)
   is the normative syntax, lifecycle, quota and qualification contract. No full
   Mermaid browser API, React or D3 catalog entry is implied.
2. Author diagrams as text-only `pre[data-canvas-diagram="mermaid"]` declarations.
   Mermaid fences use the existing Open in Canvas import workflow with text-node
   escaping. All declarations parse, validate and prepare once at startup before
   authored scripts. No mutation observer or public diagram-update API is added.
3. Reuse verified upstream grammar inputs where practical, with a closed semantic
   adapter and bounded Canvas layout. Accepting the upstream grammar is not
   accepting every feature: reject unsupported operations/configuration explicitly
   before drawing. Layout guarantees semantic/geometry determinism under pinned
   text metrics, not full Mermaid aesthetics or cross-platform pixel equality.
4. Execute parser, adapter and layout only in the existing QuickJS-WASM worker.
   Load the fixed verified packaged asset closure before generated execution;
   expose no native realm, new DOM/SVG privileges, network, filesystem, cookies,
   module loader or host API. Preserve the confirmed bridge and browser scopes.
   Library work and every diagram share document-wide script, memory, time, DOM
   and patch budgets. Additional diagram admission/work limits do not raise V1
   ceilings and are not a guarantee that all individual maxima fit together.
5. An immutable short `runtime_profile` ID references a packaged immutable manifest
   with exact grammar/build/adapter/layout identities, full integrity hashes and
   semantic quotas. Never reuse an ID for different bytes. One Canvas-domain
   resolver serves all creation, update, staging, promotion, delivery and load
   paths. New Mermaid content can create a V2 child of V1; existing V2 updates
   retain their exact profile, even when diagrams are removed. Renames and
   historical loads preserve profile identity. Derived-plan schemas are closed
   and profile-specific; V1's wire format stays unchanged.
6. Security policy overrides execution availability. Unknown, missing, tampered,
   retired or revoked profiles remain source-only without substitution, even if
   their immutable revision is valid. Source/history/export remain available.
   Installed policy changes invalidate active execution and bridge reservations;
   no online policy service is added. Repair under a current profile is an explicit
   new Canvas, not an automatic revision upgrade. Compiler security checks may
   tighten without rewriting a stored semantic profile; cache identity includes
   compiler/security policy as well as exact source and profile manifest.
7. Artifact acceptance and preview success are independent. Host validation can
   reject before staging, but browser parsing/layout may fail after commit. Keep
   that failed revision identifiable with source, a bounded error and View previous.
   Never label an old preview as a successful new revision. Runtime errors do not
   roll back committed history; turn cancellation retains ADR-121's atomicity.
8. Retain schema 68's profile storage and Canvas archive format 3.0. Exports carry
   source/profile identities, not executable library installers. Imports never
   fetch, trust an archive-provided catalog or infer a replacement profile.
   Temporary ownership, atomic promotion, branch history, deletion and local-only
   storage remain governed by ADR-121. Synchronization stays with TASK-31003.

## Alternatives considered

| Alternative | Reason not selected |
| --- | --- |
| Full/Tiny Mermaid with larger quotas or native APIs | Spike exposed both size and native-platform gaps; relaxing the sandbox violates the approved zero-egress requirement. |
| Hand-written full parser | Larger compatibility burden; reuse pinned upstream grammar with explicit semantic admission instead. Internal upstream grammar modules still require reproducible-build and upgrade tests. |
| Latest library on every load | Changes old revisions and fails offline reproducibility. Immutable profiles are explicit compatibility identities. |
| Always execute an available pinned profile | Reproducibility cannot justify executing a known-unsafe dependency. Source preservation and execution permission are separate. |
| Dependency database plus executable archive bundles | Existing profile storage is sufficient; imported executable bytes would create an unnecessary installation/trust boundary. |
| Host JavaScript preflight before every save | Adds a second evaluator/runtime boundary merely to conflate save with rendering. Keep honest independent statuses and browser validation. |
| Live declaration observation or new JS library API | Adds lifecycle, mutation and quota-reentry complexity outside the approved preview-first initial scope. |

## Consequences and release gate

The small surface can provide useful offline diagrams but rejects many valid
Mermaid documents. Documentation and errors must say so. Custom layout, grammar
admission and profile compatibility require fixtures and maintenance; internal
upstream generated modules are not a stable public integration API.

Older profiles may become non-executable while their source/history remain
portable. Source-only HTML exports with declarations are not standalone diagram
applications. No new storage migration or archive schema is planned.

Implementation requires reproducible/licensed assets, useful mixed-document
examples within unchanged budgets, adversarial real-browser qualification,
native/served workflow evidence, profile lifecycle tests and genuine archive
round-trips. The spike is feasibility evidence only. If the promised subset
cannot qualify within the limits, return to design review; do not silently weaken
the boundary. This proposed ADR does not authorize implementation before written
spec approval.

## Links

- [Design task](../tasks/task-31933%20-%20Design-Canvas-V2-Mermaid-subset-and-pinned-runtime-profiles.md)
- [Design spec](../../Docs/superpowers/specs/2026-09-06-chatbook-canvas-v2-mermaid-design.md)
- [V1 compatibility contract](../../Docs/Canvas/V1_RUNTIME_COMPATIBILITY.md)
- [Deferred Canvas synchronization contract](../tasks/task-31003%20-%20Define-server-synchronization-contract-for-Canvas-artifacts.md)
