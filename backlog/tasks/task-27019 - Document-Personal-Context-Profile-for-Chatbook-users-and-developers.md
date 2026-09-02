---
id: TASK-27019
title: Document Personal Context Profile for Chatbook users and developers
status: In Progress
assignee:
  - '@codex'
created_date: '2026-09-01 14:45'
updated_date: '2026-09-02 13:37'
labels: []
dependencies: []
references:
  - Docs/superpowers/specs/2026-08-31-personal-context-documentation-design.md
  - >-
    backlog/decisions/102-personal-context-profile-authority-sync-and-encryption.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Publish accurate, discoverable Chatbook documentation for using and extending the Personal Context Profile while clearly separating shipped synchronization behavior from planned capabilities.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The canonical user guide includes a quick start, task-oriented workflows, the corrected first-link-only publication boundary, and the eleven shipped troubleshooting states.
- [ ] #2 A developer guide maps Shared Core, encrypted local storage, interviews, agent authority, exact bounded context ordering, Sync-v2 integration, current limitations, separate full-local-first and API-only extension responsibilities, and targeted tests.
- [ ] #3 User and developer indexes link to the guides, and stable links connect to merged server documentation.
- [ ] #4 Documentation does not advertise an ongoing Personal Context sync caller or status surface, Manual Sync support, server REST publication, recovery import, delete-everywhere, purge acknowledgement, or a dedicated post-link conflict resolver; adaptive-interview disclosure, TLS behavior, first-link review, and local-removal residual state—including separate interview and link-key custody—match shipped code.
- [ ] #5 Targeted UI, interview, TLS, Console, first-link, dispatcher, client, removal/export, contract, link, and diff checks pass after the final rebase.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Rebase/inventory shipped behavior, including the corrections merged in PR #2310.
2. Task-oriented user guide.
3. Focused developer guide.
4. Discovery/server links.
5. Final targeted contract/link/diff verification.
6. Complete notes/open docs-only PR.

ADR required: no new ADR required; existing ADR applies
ADR path: backlog/decisions/102-personal-context-profile-authority-sync-and-encryption.md
Reason: Documentation only; the existing Personal Context authority, Sync, and encryption ADR applies.
<!-- SECTION:PLAN:END -->

## Corrected shipped-behavior claim inventory

PR #2310 corrected the approved specification after the first documentation pass. The guides and their verification must now enforce these boundaries:

- A successful reviewed first link publishes the eligible snapshot produced by the approved content-free reconciliation plan. Later syncable Chatbook changes create encrypted Personal Context outbox entries, but no shipped ongoing Personal Context caller drains them; **Manual Sync** covers Notes and Chat only. Ordinary server REST changes are not published to Chatbook.
- Setup completes before the optional chained interview. Leaving **Get to know you after setup** unchecked is the setup-only opt-out and stores no interview answers. Within an interview, **Skip** skips only the current question. **Cancel** opens **Leave interview**, where **Keep draft**, **Discard draft**, and **Continue interview** determine exit and draft retention.
- The fixed interview is local. The adaptive interview uses the default Console provider and model with tools disabled. Its requests include the audience, coverage topics, attempt number, eligible records for the selected scope, and—after the first answer—all prior answered turns with raw answer text. The UI can show the actual provider/model only after the first provider response, before answer input.
- Interview draft and transcript objects are not Sync payloads. Approved answer text can become an ordinary canonical record and is then governed by that record's controls.
- Chatbook accepts HTTP and HTTPS server URLs. HTTP is unencrypted. HTTPS protects transport privacy when a valid server certificate is verified through default trust or a correctly configured custom CA. Disabling verification removes server authentication and permits interception. Runtime calls honor default verification, a custom CA bundle, or verification off; **Test Connection** uses the HTTP client's default certificate verification.
- Before link approval, bootstrap exchanges metadata and downloads eligible server records and proposals into transient memory. Review and durable state stay content-free, and no local profile content uploads before approval.
- **Remove local profile** deletes the canonical Personal Context repository and its canonical outbox, but the separate encrypted interview-draft database and per-session draft keys, separate Sync state and staged envelopes, and link-custody wrapping, staged-integrity, or dataset-staging keys can remain. Retained drafts may contain raw answers. It does not delete the server copy or unregister the device. **Finish secure removal** retries canonical profile-key deletion only; it is not a complete device-artifact purge. Recovery export has no shipped import or restore flow.
- Chatbook has no **Delete everywhere**. Server purge remains a server-local fence in `purge_pending`; Sync distribution and acknowledgement completion are not wired end to end.
- First-link version conflicts and semantic collisions use content-free **Keep this device** or **Keep server** lineage choices. Later version or semantic conflict metadata may remain, but there is no shipped ongoing Personal Context sync cycle, status surface, or dedicated Personal Context conflict resolver.
- The current Console preview route is **Ctrl+Shift+P** (**View context**) > **Conversation Inspector** > outer **Next Send** > inner **Next Send** payload tab.
- Effective context order is workspace corrections/constraints, other keyed workspace records, global corrections/constraints, relevant preferences/working context, then the remainder. Selection uses whole records under the 10-percent token and 12 KiB limits, so a record that does not fit is skipped and a later smaller record can still be selected.
- Extension guidance distinguishes a full local-first Sync peer—which separates device wrapping/unwrap custody, staged-integrity and dataset-staging custody, and active `ProfileKeyMaterial` custody—from an API-only client, which uses authenticated server APIs and does not need to implement Chatbook's local profile, interview, injection, or tool stack.
- The five-step quick start separates manual **Add**/**Edit** > **Save** from interview review > **Save only**/**Save and use with agents**, and the workflow and boundary tables keep complete claims in compact user-facing form.

The stable server guide URLs temporarily retain the older continuous-sync wording. Final cross-repository parity verification depends on merging the already-approved server documentation correction; Chatbook uses the corrected PR #2310 specification now. TASK-27019 remains **In Progress** through that ordered merge and final verification.

## Review correction scope and evidence

The quality-correction pass may modify only the focused developer guide, its implementation plan, this TASK-27019 record, and the documentation design's review-status metadata. The approved user guide and the TASK-28228 collision correction remain unchanged. The metadata correction records that PR #2310's shipped-behavior correction was reviewed and merged; it does not reopen or modify authoritative TASK-27016.

Evidence is the shipped owner boundary in `PersonalContextService.remove_local_profile()`, `finish_secure_removal()`, `PersonalContextRepository.destroy_profile_content()` and `apply_reviewed_link()`, `_draft_repository()`, `InterviewDraftRepository`, `KeyringPersonalContextWrappingKeyProvider`, `PersonalContextLinkKeyCustodian`, `ProfileKeyProtector.replace()`, and `ProfileContextService` ordering/whole-record budget code. The wrapping provider owns the device RSA private key and unwrap; the link custodian owns staged-integrity and dataset-staging keys; after reviewed rebaseline the active integrity key is stored in canonical `ProfileKeyMaterial`, and durable completion deletes the staged copy. Final verification must cover that custody lifecycle, removal residuals, exact priority order, separate integrator modes, specification status, stable-link-only guide prose, shared-block parity, privacy, task-ID uniqueness, diff, and allowed-file scope.

## Renumbering provenance

- Previous ID: TASK-26835
- Current ID: TASK-27019
- Reason: current `origin/dev` contains the older `task-26835 - Textual-batch-updates-leave-the-screen-frozen-until-the-next-input-event.md` record (created 2026-09-01 14:27); this documentation record was created at 2026-09-01 14:45 and therefore moved under the younger-task-renumbers rule.
- Inbound references: the filename, frontmatter ID, and Chatbook documentation implementation plan references moved together to TASK-27019; no older task was changed.
