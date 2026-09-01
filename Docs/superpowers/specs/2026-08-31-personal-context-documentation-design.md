# Personal Context Profile Documentation Design

- **Date:** 2026-08-31
- **Status:** Approved for implementation planning
- **Repositories:** `tldw_chatbook`, `tldw_server`

## Decision record check

- ADR required: no
- ADR path: N/A
- Reason: This work documents the Personal Context architecture and behavior already governed by the existing Personal Context ADRs and merged implementation. It does not change storage, synchronization, security, authority, or user-interface contracts.

## Purpose

Publish accurate user and developer documentation for the Personal Context Profile feature in both repositories. The documentation must help people use, operate, extend, and troubleshoot the feature without creating two competing descriptions of the shared contract or presenting planned behavior as shipped behavior.

Chatbook remains the primary user-facing profile editor and interview experience. Server documentation explains the home-peer role, authenticated API, operations, storage, security, and synchronization boundary. Each repository owns its product-specific details while both repeat a short, deliberately identical statement of the shared identity and synchronization contract.

## Goals

- Make the existing Chatbook profile experience easy to discover and use.
- Explain global and workspace context, optional interviews, reviewable agent learning, profile editing, export, and deletion.
- Explain how a standalone Chatbook profile and a server profile become one logical profile only after successful reviewed linking, identity adoption, and convergence.
- State exactly which data synchronizes and which settings or secrets remain peer-local.
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
- After a successful reviewed link, Chatbook and tldw_server converge on the same canonical manifest, scope, record, proposal, and version identities and bytes for eligible shared objects.
- Sync V2 defines the `personal_context.manifest`, `personal_context.scope`, `personal_context.record`, `personal_context.proposal`, and content-free `personal_context.purge` domains. The current linked flow publishes eligible Chatbook-originated manifest, scope, record, and proposal changes; purge production and distribution are not wired end to end.
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
- Chatbook outbox, reconciliation, and client-side recovery.
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

## Current implementation boundary

The documentation must distinguish the accepted architecture from the behavior available in the merged implementation:

- Reviewed first-link reconciliation is available. A standalone Chatbook profile and an existing server profile remain separate until the user approves reconciliation and link completion succeeds.
- Eligible Chatbook-originated changes can use the Personal Context Sync-v2 domains after linking.
- Chatbook currently exposes **Remove local profile**. It does not expose a **Delete everywhere** action.
- The server exposes authenticated `POST /purge`. It advances a server-local canonical purge fence and leaves the server in `purge_pending`; it does not publish a `personal_context.purge` Sync envelope, and completion through device acknowledgement is not implemented. User documentation must present this as a current limitation, not a routine recoverable workflow.
- First-link collisions have a reviewed reconciliation surface. Normal post-link conflicts may be retained as generic Sync conflict metadata, but there is no dedicated Personal Context conflict-resolution screen.
- The server REST API operates the canonical server copy, but ordinary server-origin record and proposal mutations are not currently published into the Personal Context Sync log for already-linked Chatbook clients. Chatbook is therefore the supported editing surface for changes expected to flow through the current shared sync path.

These gaps should be named as limitations and follow-up engineering work. Documentation must not imply that users can complete them through hidden controls or unsupported manual database changes.

## Chatbook documentation changes

### User guide

Enhance the existing canonical guide:

`Docs/User_Guide/settings/personal-context-profile.md`

The current guide already contains the comprehensive feature reference. The change must preserve it as the canonical detailed user guide and add only the missing task-oriented material:

1. An **In five minutes** quick start covering profile creation, the optional interview, review, and first use.
2. Short common workflows for:
   - adding or editing a global preference;
   - adding workspace goals and long-term context;
   - reviewing something an agent learned;
   - rerunning an interview;
   - linking a home server;
   - exporting and removing a local copy;
   - understanding the separate server purge endpoint and its current `purge_pending` limitation.
3. A compact **What synchronizes?** table.
4. A troubleshooting table covering the failure states defined below.
5. Clear links to the server operator and API documentation.

The guide must not duplicate internal module descriptions or the complete server endpoint reference.

### Developer guide

Add:

`Docs/Development/personal-context-profile.md`

The guide will cover:

- Shared-core contract and compatibility boundary.
- Local encrypted repository and key custody.
- Service boundary and prohibition on direct profile-table access.
- Interview draft, answer, proposal, and review lifecycle.
- Agent tool exposure and effective-authority calculation.
- Context selection and injection.
- Outbox, Sync-v2 adapters, first-link reconciliation, conflicts, and purge generations.
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
- Sync status, conflicts, and operational recovery.
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
- Optimistic concurrency, idempotency, semantic collisions, first-link reconciliation, generic Sync conflict metadata, and purge barriers.
- Logging and privacy constraints.
- Extension checklist and server conformance/test map.
- The missing server-origin Sync publication seam as an explicit extension point and current limitation.

It will link to the existing server design, ADR, and API reference rather than duplicating them. Because `Docs/Design/` and `backlog/decisions/` are not published into the MkDocs tree, links to those source-only documents must use stable GitHub `blob/dev` URLs; links to published guides and API pages remain relative.

### Discovery and publication

Update:

- `Docs/User_Guides/index.md`
- `Docs/Code_Documentation/index.md`
- `Docs/Code_Documentation/README.md` when required by its existing organization
- `Docs/API-related/API_README.md` with a related-guide link
- `Docs/mkdocs.yml`

Canonical source remains under `Docs/`. `Docs/Published/` must be regenerated with `Helper_Scripts/refresh_docs_published.sh`; generated files must not be edited by hand.

## What synchronizes

The user and developer documentation must distinguish these categories explicitly.

| Shared through the current linked flow when eligible | Remains peer-local or is not currently published |
| --- | --- |
| Canonical manifest after successful reviewed linking | Peer-local at-rest encryption and recovery keys |
| Required global and linked-workspace scope objects | Raw interview answers and unfinished drafts |
| Records and tombstones whose controls permit synchronization | Runtime agent authority grants and tool availability |
| Eligible proposals and their canonical review state | Device-only records or records marked non-syncable |
| Exact canonical object identities, versions, and bytes for eligible shared objects | Local undo history, caches, ciphertext, database row identities, and other operational metadata |
| — | Conflict-review objects and acknowledgement tracking |

`personal_context.purge` exists at the protocol and adapter boundary, but the shipped Chatbook has no producer and server `POST /purge` does not publish that envelope. The server endpoint currently performs a server-local canonical purge and fence only; barrier distribution and acknowledgement completion are not a reachable end-to-end workflow.

Wording must not imply that a shared logical record means peers share at-rest ciphertext, recovery keys, or physical database rows. Developer documentation must separately explain that the home server owns the Sync integrity key and distributes it wrapped for an authenticated registered Chatbook device during bootstrap; that transport key exchange is not synchronization of profile content or at-rest key custody.

## User lifecycle

The documentation will describe the lifecycle in prose and small tables:

1. A user creates or edits context directly, or completes an optional interview.
2. Interview output always goes through user review. Inferred or newly learned agent context becomes a proposal.
3. Direct-write authority permits only the narrow update of an existing eligible record for an explicit correction whose exact evidence appears in the current persisted user message.
4. Accepted changes are stored as encrypted canonical records in the local peer repository.
5. Eligible Chatbook-originated changes enter its outbox and synchronize with the configured home peer. Ordinary server REST mutations do not currently publish back to linked Chatbook clients.
6. The peers transport eligible versions, tombstones, and proposals. First-link collisions are reviewed before linking; later conflicts remain generic Sync metadata until a dedicated resolution surface exists.
7. Chatbook selects permitted global and current-workspace records for agent context and exposes the exact assembled body in **Next Send**.

## Failure and recovery guidance

Both user-facing guides must use consistent names and direct the reader to the owning peer for recovery.

| State | Meaning | Required guidance |
| --- | --- | --- |
| Profile locked | The peer cannot decrypt profile content because key material is unavailable or locked. | Restore or unlock the configured key; do not recreate or overwrite encrypted data as a first response. |
| Offline or queued | Local changes are safe but have not reached the home peer. | Continue locally where supported, then retry sync and inspect the outbox/status. |
| Capability not negotiated | Peer versions do not share the required profile capability. | Upgrade the incompatible peer and retry; do not bypass negotiation. |
| Version conflict | Both peers changed the same canonical object from different bases. | Preserve the conflict and inspect the generic Sync status/metadata. A dedicated Personal Context post-link resolver is not currently shipped; do not claim that Settings can resolve it. |
| First-link semantic collision | Distinct local and server record identities describe the same scope/kind/key during reviewed linking. | Compare and resolve the presented records in the first-link reconciliation review before treating the profiles as one synchronized set. |
| Post-link semantic collision | Distinct record identities describe the same scope/kind/key after linking. | Preserve both sides and inspect generic Sync status/metadata. No dedicated Personal Context resolver is currently shipped; do not claim that Settings can resolve it. |
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
- Confirm the API reference no longer says Sync is wholly outside the server when the current Sync-v2 domains are present, while retaining the server-origin publication and purge-acknowledgement limitations.
- Confirm the `tldw-profile-core==0.1.0` boundary against the vendored server package, parity tests, current digest authority, and governing ADR.
- Confirm the guides distinguish protocol support for `personal_context.purge` from the absent producer/distribution/acknowledgement workflow.
- Check Markdown formatting and repository diff hygiene.

### tldw_chatbook

- Run the repository's targeted documentation-link or Markdown validation if available.
- Verify every internal relative link and every cross-repository target.
- Confirm user instructions match the merged Settings and Console surfaces.
- Confirm developer paths, symbols, and test references exist on `dev`.
- Confirm the guides do not advertise a Chatbook **Delete everywhere** control, completed purge acknowledgement, a dedicated post-link Personal Context conflict resolver, or automatic publication of server API edits to linked clients.
- Confirm the guides name the exact current Sync-v2 domains and distinguish them from Shared Core models.
- Confirm the `tldw-profile-core==0.1.0` pin and parity authority against current code, tests, and the governing ADR without freezing a stale implementation commit as user-facing truth.
- Check Markdown formatting and repository diff hygiene.

No full application test sweep is required for documentation-only changes unless repository checks expose an integration concern.

## Acceptance of the documentation set

The documentation work is complete when:

- Both repositories contain discoverable user and developer guidance appropriate to their roles.
- The guides agree about post-link canonical identity, encryption, syncability, authority, and the current deletion boundary.
- The guides clearly distinguish synchronized content from peer-local keys and settings.
- No guide promises a UI or operation that the merged products do not provide.
- Common failure states have actionable, consistent guidance, including an explicit statement when the current release has no completion or resolution surface.
- Cross-repository links resolve on `dev` after the ordered merges.
- Server published documentation is reproducible and strict site validation passes.
- Both PRs are based on current `dev`, contain only documentation/task artifacts, and have no unrelated changes.
