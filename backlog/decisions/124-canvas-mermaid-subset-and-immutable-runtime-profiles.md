# ADR-124: Canvas Mermaid subset and immutable runtime profiles

Status: Accepted — user approved the written design on 2026-09-06
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
   Tool parameters stay unchanged; bounded, profile-aware model guidance includes
   tested flow/sequence examples and source-free repair hints. Historical updates
   receive their exact profile's guide, not the creation default. Browser errors
   never become automatic assistant submissions.
3. Reuse verified upstream grammar inputs where practical, with a closed semantic
   adapter and bounded Canvas layout. Accepting the upstream grammar is not
   accepting every feature: reject unsupported operations/configuration explicitly
   before drawing. Layout guarantees semantic/geometry determinism under pinned
   text metrics, not full Mermaid aesthetics or cross-platform pixel equality.
   Pin Unicode segmentation/width data and rules. Default 16 px monospace text
   with 24 px line spacing uses scrollable intrinsic-size output, not automatic
   shrink-to-fit. Cap diagram width, height and aggregate logical area as specified
   in the design; reject invalid/nonfinite geometry before patches. Authored CSS/JS
   overrides remain possible under V1 and are outside default-layout fidelity.
4. Execute parser, adapter and layout only in the existing QuickJS-WASM worker.
   Load the fixed verified packaged asset closure before generated execution;
   expose no native realm, new DOM/SVG privileges, network, filesystem, cookies,
   module loader or host API. Preserve the confirmed bridge and browser scopes.
   Library work and every diagram share document-wide script, memory, time, DOM
   and patch budgets. Additional diagram admission/work limits do not raise V1
   ceilings and are not a guarantee that all individual maxima fit together.
5. An immutable short `runtime_profile` ID references a packaged immutable manifest
   with exact grammar/build/adapter/layout identities, engine build, facade/plan
   compatibility versions, Unicode rules/data, full integrity hashes and semantic
   quotas. Changing pinned inputs requires a new profile even for an engine
   security fix; nonsemantic host/compiler security validation can tighten under
   separate build/policy identity with compatibility evidence. Never reuse an ID
   for different bytes. One Canvas-domain resolver serves all creation, update,
   staging, promotion, delivery and load
   paths. New Mermaid content can create a V2 child of V1; existing V2 updates
   retain their exact profile, even when diagrams are removed. Renames and
   historical loads preserve profile identity. Derived-plan schemas are closed
   and profile-specific; V1's wire format stays unchanged.
6. Security policy overrides execution availability. Unknown, missing, tampered,
   retired or revoked profiles remain source-only without substitution, even if
   their immutable revision is valid. Source/history/export remain available.
   Catalog/policy snapshots last for the process lifetime. Application and packaged
   policy updates require stopping/restarting the native host or served parent and
   all children; browser refresh is insufficient. Restart invalidates old loads and
   reservations, and parent/child snapshot mismatches fail closed. No hot-reload
   watcher or online policy service is added. The existing explicit Canvas-disable
   latch remains the immediate live containment action. Repair under a current
   profile is an explicit new Canvas, not an automatic revision upgrade. Compiler
   security checks may tighten without rewriting a stored semantic profile; cache identity includes
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

Task 4 implementation clarification: V2 owns separate
`canvas_runtime_worker_v2.js` and `canvas_renderer_v2.js` files and hashes;
the published V1 closure is not rewritten. The trusted parent supplies captured
inert library/manifest strings in a bounded private init/prepare envelope before
execution acknowledgement. Both consumers verify those bytes before QuickJS
startup. This preserves the existing CSP and avoids a native library module.
Product parent delivery is implemented in Task 6; the candidate stays disabled
until Task 8 qualification. Separate frozen files require per-profile maintenance.

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
| Hot-reload installed runtime/revocation policy | Introduces cross-process update races and mutable-asset/cache coordination. V2 requires an orderly host/server restart; explicit Canvas disable remains available for immediate containment. |

## Consequences and release gate

The small surface can provide useful offline diagrams but rejects many valid
Mermaid documents. Documentation and errors must say so. Custom layout, grammar
admission and profile compatibility require fixtures and maintenance; internal
upstream generated modules are not a stable public integration API.

Older profiles may become non-executable while their source/history remain
portable. Source-only HTML exports with declarations are not standalone diagram
applications. No new storage migration or archive schema is planned.

Security updates to pinned engine/semantic inputs may require new Canvases under
a new profile; old vulnerable profiles stay source-only rather than being silently
patched in place. Operators must restart the complete serving process group after
updates, not just reload a browser. Diagram defaults favor readable scrolling;
custom authored styles can change appearance without rerunning layout.

Implementation requires reproducible/licensed assets, useful mixed-document
examples within unchanged budgets, adversarial real-browser qualification,
native/served workflow evidence, profile lifecycle tests and genuine archive
round-trips. The spike is feasibility evidence only. If the promised subset
cannot qualify within the limits, return to design review; do not silently weaken
the boundary. Written-spec approval was received on 2026-09-06; the qualification
gates remain mandatory before product admission.

## Qualification clarification: trusted selection publication

Before first release, Task31941 removes the misleading `qualification` field
from the generated manifest's Mermaid provenance. Manifest/source/input hashes
describe immutable identity; the catalog alone describes execution policy. This
metadata-only correction changes the final manifest hash without changing any
executable, library, or V1 byte. Rebuilding retains checked-in admission or
revocation only when the regenerated manifest and complete library inventory
(including notices) exactly match that entry. A changed identity regenerates
disabled with no diagram default. There is no CLI/environment admission switch,
and a rebuild cannot silently un-revoke an unchanged profile. This clarification
precedes the first immutable admission; released identities still cannot change.

Task31941's separately owned parent/all-child restart gate exposed a normal
new-Canvas publication interval: the trusted child had selected its newly saved
Canvas while the parent's last snapshot still identified the old source-only
Canvas. Its event response correctly carried the new identity and was refused.
The existing parent snapshot reconciliation is now optionally constrained to the
captured child, conversation session, and existing shell. Those bindings must
remain current before and after the snapshot await; reconciliation cannot create
or rebind a shell for a replacement owner. Mismatched events are always discarded.
Only a proven advanced live selection uses the gateway's existing 409 response;
unchanged, malformed, disconnected, or cross-session responses remain refused.
This preserves the authority boundary without forwarding sibling events, granting
browser-selected authority, adding routes, or treating arbitrary 503s as retries.

## Links

- [Design task](../tasks/task-31933%20-%20Design-Canvas-V2-Mermaid-subset-and-pinned-runtime-profiles.md)
- [Design spec](../../Docs/superpowers/specs/2026-09-06-chatbook-canvas-v2-mermaid-design.md)
- [Implementation plan](../../Docs/superpowers/plans/2026-09-06-chatbook-canvas-v2-mermaid-implementation.md)
- [V1 compatibility contract](../../Docs/Canvas/V1_RUNTIME_COMPATIBILITY.md)
- [Deferred Canvas synchronization contract](../tasks/task-31003%20-%20Define-server-synchronization-contract-for-Canvas-artifacts.md)
