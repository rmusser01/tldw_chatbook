# Personal Context Profile Documentation Design

- **Date:** 2026-08-31
- **Status:** Approved design; shipped-behavior correction pending review
- **Last corrected:** 2026-09-02
- **Repositories:** `tldw_chatbook`, `tldw_server`

## Decision record check

- ADR required: no new ADR required; existing ADR applies
- ADR path: `backlog/decisions/102-personal-context-profile-authority-sync-and-encryption.md`
- Reason: This work documents the Personal Context architecture and behavior already governed by the existing Personal Context ADRs and merged implementation. It does not change storage, synchronization, security, authority, or user-interface contracts.

## Purpose

Publish accurate user and developer documentation for the Personal Context Profile feature in both repositories. The documentation must help people use, operate, extend, and troubleshoot the feature without creating two competing descriptions of the shared contract or presenting planned behavior as shipped behavior.

Chatbook remains the primary user-facing profile editor and interview experience. Server documentation explains the home-peer role, authenticated API, operations, storage, security, and synchronization boundary. Each repository owns its product-specific details while both repeat a short, deliberately identical statement of the shared identity and synchronization contract.

## Goals

- Make the existing Chatbook profile experience easy to discover and use.
- Explain global and workspace context, optional interviews, reviewable agent learning, profile editing, export, and deletion.
- Explain how a standalone Chatbook profile and a server profile become one logical profile only after successful reviewed first linking, identity adoption, and publication of the approved snapshot.
- State exactly what reviewed first linking publishes, what later changes only queue locally today, and which settings or secrets remain peer-local.
- Give developers a reliable map of the shared core package, peer-specific services and repositories, authority checks, encryption, synchronization, and tests.
- Give users and operators actionable recovery guidance for common locked, offline, conflict, compatibility, and purge states.
- Keep the documentation maintainable by assigning one owner to each body of detail and linking rather than duplicating it.

## Non-goals

- Redesigning or changing the Personal Context feature.
- Adding a standalone server WebUI profile editor.
- Reproducing the complete API reference in a user guide.
- Reproducing implementation plans or ADRs as end-user documentation.
- Publishing encryption keys, recovery secrets, raw interview drafts, or other private profile material in examples.
- Adding diagrams or other visual aids where prose and compact tables are clearer.

## Documentation ownership

### Shared contract

Both repositories will include a short matching contract statement:

- `tldw_profile_core` defines the versioned canonical profile object models, exact canonical bytes, interview/tool contracts, serialization, and validation used by both peers. Sync-v2 transport envelopes are a separate contract.
- After successful reviewed first linking, Chatbook and tldw_server converge on the same canonical manifest, scope, record, proposal, and version identities and bytes for the eligible snapshot the user approved.
- Sync V2 defines the `personal_context.manifest`, `personal_context.scope`, `personal_context.record`, `personal_context.proposal`, and content-free `personal_context.purge` domains. Reviewed first linking publishes the eligible Chatbook-approved snapshot. Later syncable Chatbook mutations create encrypted local outbox entries, but the current shipped app does not run an ongoing Personal Context sync cycle, so those post-link changes remain queued locally. Purge production and distribution are not wired end to end.
- Each peer retains its own at-rest ciphertext and keys, local database rows, runtime permissions, conflict-review metadata, acknowledgement tracking, and other operational state.

The shared statement is intentionally short. Detailed client behavior belongs in Chatbook; API and server behavior belongs in tldw_server.

### Chatbook-owned detail

Chatbook documentation owns:

- Setup-wizard and post-install interview workflows.
- Settings navigation and profile editing.
- Global versus workspace context from the user's perspective.
- Agent read, propose, review, direct-write, and promotion behavior.
- Context injection and the Console **Next Send** preview.
- Local encryption and unlock behavior.
- Chatbook outbox creation, reviewed first-link reconciliation, the missing ongoing Personal Context sync caller, and client-side recovery limits.
- Client architecture, services, repositories, UI boundaries, and client tests.

### Server-owned detail

tldw_server documentation owns:

- Home-peer configuration and authenticated access.
- Per-user storage and ownership boundaries.
- Master-key custody and locked-profile behavior.
- REST and Sync-v2 surfaces.
- Server repository and service boundaries.
- Conflict, idempotency, capability, export, and purge semantics.
- Operator troubleshooting and server tests.

## Architecture target versus shipped behavior

ADR-102 and the approved product design define the intended one-profile, ongoing-sync architecture. They remain the authority for future implementation decisions. They are not evidence that every lifecycle described there is reachable in the current products. User and developer guides must label protocol support, accepted architecture, and extension points separately from shipped controls and callers.

## Current implementation boundary

The documentation must distinguish the accepted architecture from the behavior available in the merged implementation:

- Reviewed first-link reconciliation and publication are available. A standalone Chatbook profile and an existing server profile remain separate until the user approves reconciliation and link completion succeeds. Successful completion publishes the eligible approved snapshot and establishes matching canonical identities and bytes on both peers.
- Later syncable Chatbook mutations atomically create encrypted Personal Context outbox entries. The transport and apply paths understand the five Personal Context domains, but no shipped startup, background, Settings, or other production caller runs an ongoing `LocalFirstSyncService.sync_once()` cycle for them. **Overview → Manual Sync** invokes only the Notes and Chat domains. Post-link Personal Context changes therefore remain queued locally; documentation must not promise ongoing convergence, tell users to retry Manual Sync, or claim that Settings exposes a Personal Context sync/outbox status.
- Ordinary server REST mutations update the server canonical copy but are not published into the Personal Context Sync log for linked Chatbook clients. Current post-link editing can therefore make the peers diverge in either direction.
- The fixed interview mode is local and makes no model call. Adaptive mode uses the configured default Console provider without tools. Each request sends the interview audience, allowed topics, attempt number, eligible agent-visible records from the exact selected scope, and—after the first answer—all prior answered turns including raw answer text. The interview screen shows the actual provider and model only after the first provider response completes, before answer entry becomes available. Guides must state this ordering and must not claim that raw answers stay local or that provider/model disclosure precedes the first egress.
- Chatbook accepts configured server URLs with either `http://` or `https://`; it does not enforce HTTPS for non-loopback hosts. Runtime requests verify TLS by default, but Settings → Data & Privacy → Network permits a custom CA or disabled verification. **Test Connection** uses httpx's default verification rather than the saved custom/off runtime policy. Documentation may recommend HTTPS and default verification, but must not describe either as enforced.
- First-link bootstrap exchanges authentication/capability, device-registration and public-key, display, schema/quota, and purge-generation metadata before the user approves reconciliation. It does not upload local profile record or proposal content until approval. Guides must disclose the pre-approval metadata exchange without implying that approval gates all network activity.
- Chatbook currently exposes **Remove local profile**. It removes local canonical profile data, link state, pending Personal Context outbox entries, and local conflict/quarantine rows; it neither deletes the server copy nor unregisters the device. Its encrypted recovery export includes canonical local heads, including device-only records, but no shipped UI or production caller imports/restores that export. Key deletion occurs after row removal and can fail; in that state Settings must direct the user to **Finish secure removal**. Chatbook does not expose a **Delete everywhere** action.
- The server exposes authenticated `POST /purge`. It advances a server-local canonical purge fence and leaves the server in `purge_pending`; it does not publish a `personal_context.purge` Sync envelope, and completion through device acknowledgement is not implemented. User documentation must present this as a current limitation, not a routine recoverable workflow.
- First-link collisions have a reviewed reconciliation surface. The transport can retain later conflicts as generic Sync conflict metadata, but the shipped app neither runs the ongoing Personal Context cycle that would ordinarily produce those conflicts nor provides a dedicated Personal Context status or conflict-resolution screen.

These gaps should be named as limitations and follow-up engineering work. Documentation must not imply that users can complete them through hidden controls or unsupported manual database changes.

### Desired future behavior

Future engineering may complete the accepted architecture by scheduling or exposing an ongoing Personal Context sync cycle, publishing server-origin mutations, surfacing queue/conflict state and resolution, enforcing an explicit production transport policy, disclosing the adaptive provider/model before first egress, adding recovery-import support, and completing purge-barrier distribution and acknowledgement. Until each path is implemented and verified at the user-facing surface, documentation must label it as desired future behavior rather than a current capability.

## Chatbook documentation changes

### User guide

Enhance the existing canonical guide:

`Docs/User_Guide/settings/personal-context-profile.md`

The current guide already contains the comprehensive feature reference. The change must preserve it as the canonical detailed user guide and add only the missing task-oriented material:

1. An **In five minutes** quick start covering profile creation, the optional fixed or adaptive interview, review, and first use. It must distinguish **Save only** from **Save and use with agents** and explain adaptive-mode egress before offering that mode.
2. Short common workflows for:
   - adding or editing a global preference;
   - adding workspace goals and long-term context;
   - reviewing something an agent learned;
   - rerunning an interview;
   - activating/authenticating a server and then linking it from **My Profile**;
   - exporting and removing a local copy, including the absence of a recovery-import control and the **Finish secure removal** recovery state;
   - understanding the separate server purge endpoint and its current `purge_pending` limitation.
3. A compact **What first linking publishes, and what does not sync afterward** table.
4. A transport warning that distinguishes accepted `http://` and `https://` URLs, runtime TLS verification modes, and **Test Connection** behavior.
5. A troubleshooting table covering the failure states defined below without inventing an ongoing Personal Context sync/status action.
6. Clear links to the server operator and API documentation.

The guide must not duplicate internal module descriptions or the complete server endpoint reference.

### Developer guide

Add:

`Docs/Development/personal-context-profile.md`

The guide will cover:

- Shared-core contract and compatibility boundary.
- Local encrypted repository and key custody.
- Service boundary and prohibition on direct profile-table access.
- Fixed and adaptive interview request contents, provider/model disclosure timing, draft, answer, proposal, and review lifecycle.
- Agent tool exposure and effective-authority calculation.
- Context selection and injection.
- Outbox creation, Sync-v2 adapters, reviewed first-link publication, the absent ongoing Personal Context sync caller, generic transport conflict storage, and purge generations.
- First-link bootstrap sequencing and metadata exchanged before review approval.
- Server URL validation, runtime TLS trust policy, and the different **Test Connection** verification path.
- Local removal transaction boundaries, post-row key cleanup, recovery-export contents, and the absent production restore caller.
- Extension checklist and targeted test map.

It will link to the existing Personal Context ADR/design and the generic Sync-v2 client reference instead of restating those documents.

### Discovery links

Add or confirm links from:

- `Docs/User_Guide/index.md`
- `Docs/User_Guide/settings.md`
- `Docs/Development/Developer_Guide.md`

Only concise index descriptions should be added.

## Server documentation changes

### User and operator guide

Add:

`Docs/User_Guides/Server/Personal_Context_Profile.md`

The guide will explain:

- What the server contributes to a shared profile.
- Required authentication and master-key setup.
- Linking Chatbook as the primary user interface.
- Inspecting or operating the profile through the authenticated API when needed.
- Export, local-client removal, and the server's current purge behavior.
- Reviewed first-link state, protocol-level conflict handling, the absence of ongoing client Personal Context synchronization, and operational recovery limits.
- The current product boundary: the server does not provide a complete standalone profile-editing WebUI.

Endpoint details will link to the existing API reference:

`Docs/API-related/Personal_Context_API.md`

### Developer guide

Add:

`Docs/Code_Documentation/Personal_Context_Developer_Guide.md`

The guide will cover:

- Authentication and per-user ownership.
- Shared-core version and compatibility boundary.
- Key custody, encryption, and locked states.
- API, service, repository, and Sync-v2 adapter responsibilities.
- Optimistic concurrency, idempotency, semantic collisions, first-link reconciliation/publication, generic Sync conflict metadata, and purge barriers.
- Logging and privacy constraints.
- Extension checklist and server conformance/test map.
- The missing ongoing Chatbook Personal Context sync caller and missing server-origin publication seam as explicit extension points and current limitations.

It will link to the existing server design, ADR, and API reference rather than duplicating them. Because `Docs/Design/` and `backlog/decisions/` are not published into the MkDocs tree, links to those source-only documents must use stable GitHub `blob/dev` URLs; links to published guides and API pages remain relative.

### Discovery and publication

Update:

- `Docs/User_Guides/index.md`
- `Docs/Code_Documentation/index.md`
- `Docs/Code_Documentation/README.md` when required by its existing organization
- `Docs/API-related/API_README.md` with a related-guide link
- `Docs/mkdocs.yml`

Canonical source remains under `Docs/`. `Docs/Published/` must be regenerated with `Helper_Scripts/refresh_docs_published.sh`; generated files must not be edited by hand.

## What first linking publishes

The user and developer documentation must distinguish the reviewed first-link snapshot, unsent later mutations, and peer-local state explicitly.

| Published during successful reviewed first linking when eligible | Not published by the shipped ongoing application lifecycle |
| --- | --- |
| Canonical manifest in the approved snapshot | Later syncable Chatbook mutations: encrypted outbox entries are created but no shipped ongoing Personal Context caller sends them |
| Required global and linked-workspace scopes in that snapshot | Ordinary server REST mutations: the server copy changes but no Personal Context Sync entry publishes them to Chatbook |
| Eligible record heads, tombstones, and proposal review state selected by reconciliation | Device-only or non-syncable records |
| Exact canonical object identities, versions, and bytes for those eligible objects | Runtime agent authority grants, tool availability, local workspace mappings, and enablement |
| — | Peer-local at-rest encryption/recovery keys, local undo data, caches, ciphertext, database row identities, conflict-review metadata, acknowledgement tracking, and other operational state |
| — | Interview drafts and raw answers do not enter profile Sync; adaptive interview requests nevertheless send prior raw answers to the configured provider as described below |

`personal_context.purge` exists at the protocol and adapter boundary, but the shipped Chatbook has no producer and server `POST /purge` does not publish that envelope. The server endpoint currently performs a server-local canonical purge and fence only; barrier distribution and acknowledgement completion are not a reachable end-to-end workflow.

Wording must not imply that a shared logical record means peers share at-rest ciphertext, recovery keys, or physical database rows. Developer documentation must separately explain that the home server owns the Sync integrity key and distributes it wrapped for an authenticated registered Chatbook device during bootstrap; that transport key exchange is not synchronization of profile content or at-rest key custody.

## User lifecycle

The documentation will describe the lifecycle in prose and small tables:

1. A user creates or edits context directly, or chooses an optional fixed or adaptive interview. Fixed mode stays local. Adaptive mode calls the configured default Console provider without tools and sends the bounded request data described in the current implementation boundary; after the first answer, that includes all prior raw answer text.
2. The adaptive screen reveals the actual provider/model after the first question call finishes and before answer entry is enabled. This is a current disclosure-timing limitation, not proof of pre-egress consent. Interview output always goes through user review. Inferred or newly learned agent context becomes a proposal.
3. Direct-write authority permits only the narrow update of an existing eligible record for an explicit correction whose exact evidence appears in the current persisted user message.
4. Accepted changes are stored as encrypted canonical records in the local peer repository. Syncable mutations also create encrypted local outbox entries atomically.
5. To link, the user first creates or unlocks the profile in **Settings → Data & Privacy → My Profile**, then activates and authenticates the server through **Settings → Overview → Advanced / Diagnostics → Switch Source / Server**, and finally returns to **My Profile → Link to home server**.
6. Bootstrap exchanges connection, capability, device/key, display, schema/quota, and purge-generation metadata before approval. It does not upload local record or proposal content. The user reviews reconciliation, then approval permits publication of the eligible snapshot and successful first-link convergence.
7. Later Chatbook changes remain in the encrypted local outbox because no shipped ongoing Personal Context sync cycle sends them. **Overview → Manual Sync** covers Notes and Chat only. Ordinary server REST mutations likewise do not publish to Chatbook. Guides must not describe post-link convergence as current behavior.
8. First-link collisions are resolved in the review. The transport can retain later conflicts as generic Sync metadata, but no ongoing Personal Context cycle or dedicated status/resolution UI is shipped.
9. Chatbook selects permitted global and current-workspace records for agent context and exposes the exact assembled body in **Next Send**.

## Failure and recovery guidance

Both user-facing guides must use consistent names and direct the reader to the owning peer for recovery.

| State | Meaning | Required guidance |
| --- | --- | --- |
| Profile locked | The peer cannot decrypt profile content because existing key material is unavailable or locked. | Unlock the configured key protector; do not recreate or overwrite encrypted data. Chatbook's recovery-export reader has no shipped import/restore caller or UI. |
| Adaptive interview privacy or provider failure | Adaptive mode sends bounded interview context to the default Console provider; the first request completes before the screen displays the actual provider/model. | Use fixed mode when no model egress is acceptable. If adaptive mode fails, continue ordinary setup and retry later or use fixed mode; do not claim the first request stayed local. |
| HTTP or altered TLS verification | Chatbook accepts HTTP as well as HTTPS, and runtime verification can use default trust, a custom CA, or disabled verification. **Test Connection** always uses default httpx verification. | Recommend HTTPS with verification enabled. Explain the saved Network policy and the probe/runtime mismatch; do not claim transport enforcement that the client does not perform. |
| Post-link change queued | Chatbook stored the local mutation and encrypted outbox entry, but the shipped app has no ongoing Personal Context sync caller. | Do not claim the server copy changed and do not direct the user to **Overview → Manual Sync**, which covers Notes and Chat only. Preserve the local profile; no supported Settings action currently drains this queue. |
| Capability not negotiated | Peer versions do not share the required profile capability during linking. | Upgrade the incompatible peer and retry first linking; do not bypass negotiation. |
| First-link publication interrupted | Reconciliation was approved but link completion did not finish. | Preserve both copies and retry the reviewed link flow. Do not treat the profiles as converged until completion succeeds. |
| Version conflict | Both peers changed the same canonical object from different bases at the transport boundary. | Preserve the generic Sync conflict metadata. A shipped ongoing Personal Context cycle and dedicated Settings resolver/status are both absent; do not claim that Settings can resolve it. |
| First-link semantic collision | Distinct local and server record identities describe the same scope/kind/key during reviewed linking. | Compare and resolve the presented records in the first-link reconciliation review before treating the profiles as one synchronized set. |
| Post-link semantic collision | Distinct record identities describe the same scope/kind/key after linking. The current products have no ongoing Personal Context cycle that reconciles them. | Preserve both peer copies. No dedicated Personal Context status or resolver is currently shipped; do not invent a Settings recovery path. |
| Local removal incomplete | Local rows were removed, but key cleanup failed. | Use **Finish secure removal**. Do not imply that the server copy or device registration was removed, or that the recovery export can currently be restored in the app. |
| Purge pending | The server purge barrier has advanced and ordinary mutations are blocked. | The current server cannot complete the acknowledgement workflow. State this limitation before an operator invokes `POST /purge`; do not promise that reconnecting devices will clear it. |

## Developer extension checklist

Both developer guides will contain a peer-specific version of this checklist:

1. Decide whether the change affects the shared contract or only one peer.
2. For shared canonical object changes, update `tldw_profile_core` schemas and compatibility behavior first; update Sync-v2 transport separately when its envelope or domain behavior changes.
3. Preserve canonical identities and explicit syncability.
4. Route reads and writes through the owning service; never access profile tables directly from UI, tools, or endpoints.
5. Enforce authority, scope, expiry, visibility, and secret-rejection checks at the service boundary.
6. Keep plaintext out of logs, diagnostics, outbox metadata, and unencrypted fixtures.
7. Add parity/conformance coverage in both repositories for shared behavior.
8. Add peer-specific migration, repository, service, API/UI, and recovery tests as applicable.
9. Update the governing ADR when storage, ownership, encryption, synchronization, or authority changes.
10. Update both documentation sets when the shared contract changes.

## Links and merge order

- Use relative links within a repository.
- Use stable GitHub `blob/dev` links for the counterpart repository.
- In server MkDocs pages, use stable GitHub `blob/dev` links for source-only `Docs/Design/` and `backlog/` documents that are not copied into `Docs/Published/`.
- User guides link to stable user/operator material, not implementation plans.
- Developer guides may link to accepted ADRs, designs, API references, and maintained implementation references.

The server documentation PR lands first because it can link to Chatbook's existing merged user guide. The Chatbook documentation branch is then rebased on current `dev` and finalized with links to the merged server guides. This avoids permanently landing cross-repository links whose target does not yet exist.

## Work breakdown and pull requests

Create one atomic Backlog task and one documentation PR per repository:

1. **tldw_server documentation PR against `dev`**
   - User/operator guide, developer guide, indexes, navigation, related API link, and generated published documentation.
2. **tldw_chatbook documentation PR against `dev`**
   - Focused improvements to the existing user guide, new developer guide, indexes, and final cross-repository links.

Implementation must occur in isolated worktrees so unrelated changes in existing checkouts are preserved.

## Verification

### tldw_server

- Run the repository's documentation-link validation if available.
- Run `Helper_Scripts/refresh_docs_published.sh`.
- Stage or inspect the generated output, then run the refresh script again and confirm it produces no diff.
- Run `mkdocs build --strict -f Docs/mkdocs.yml` in the supported documentation environment.
- Confirm the generated diff contains only expected pages, navigation, and indexes.
- Confirm published pages have no relative links to source-only `Docs/Design/` or `backlog/` paths.
- Confirm the API reference distinguishes implemented Personal Context Sync-v2 transport/domain support from the absent shipped ongoing client caller, absent server-origin publication, and incomplete purge acknowledgement.
- Confirm the `tldw-profile-core==0.1.0` boundary against the vendored server package, parity tests, current digest authority, and governing ADR.
- Confirm the guides distinguish protocol support for `personal_context.purge` from the absent producer/distribution/acknowledgement workflow.
- Confirm the guides limit cross-peer equality claims to successful reviewed first-link publication and do not tell users that later queued changes can be sent through Chatbook **Manual Sync**.
- Check Markdown formatting and repository diff hygiene.

### tldw_chatbook

- Run the repository's targeted documentation-link or Markdown validation if available.
- Verify every internal relative link and every cross-repository target.
- Confirm user instructions match the merged Settings and Console surfaces.
- Confirm developer paths, symbols, and test references exist on `dev`.
- Confirm the guides do not advertise ongoing post-link Personal Context synchronization, a Personal Context sync/status control, a Chatbook **Delete everywhere** control, completed purge acknowledgement, a dedicated post-link conflict resolver, or automatic publication of server API edits to linked clients.
- Confirm adaptive-interview guidance names every transmitted field, raw prior-answer egress, no-tools execution, default Console provider selection, and the actual provider/model display only after the first request completes; confirm fixed mode is described as local.
- Confirm transport guidance says Chatbook accepts HTTP and HTTPS, runtime TLS verification may be default/custom/off, and **Test Connection** uses default verification rather than the saved custom/off runtime policy.
- Confirm first-link guidance discloses pre-approval bootstrap metadata while stating that local record/proposal content is not uploaded until approval.
- Confirm removal guidance says the server copy and registration remain, pending Personal Context outbox and local conflict/quarantine state are deleted, recovery import is not shipped, and failed key cleanup requires **Finish secure removal**.
- Confirm the guides name the exact current Sync-v2 domains and distinguish them from Shared Core models.
- Confirm the `tldw-profile-core==0.1.0` pin and parity authority against current code, tests, and the governing ADR without freezing a stale implementation commit as user-facing truth.
- Check Markdown formatting and repository diff hygiene.

No full application test sweep is required for documentation-only changes unless repository checks expose an integration concern.

## Acceptance of the documentation set

The documentation work is complete when:

- Both repositories contain discoverable user and developer guidance appropriate to their roles.
- The guides agree that successful reviewed first linking establishes equal canonical identities and bytes for the approved eligible snapshot, while later mutations do not currently converge.
- The guides clearly distinguish first-link publication, unsent later outbox entries, adaptive-provider egress, and peer-local keys and settings.
- No guide promises a UI or operation that the merged products do not provide.
- Common failure states have actionable, consistent guidance, including an explicit statement when the current release has no completion or resolution surface.
- Cross-repository links resolve on `dev` after the ordered merges.
- Server published documentation is reproducible and strict site validation passes.
- Both PRs are based on current `dev`, contain only documentation/task artifacts, and have no unrelated changes.
