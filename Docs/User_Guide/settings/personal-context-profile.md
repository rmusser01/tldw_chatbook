# Settings > My Profile — Personal Context

## What this screen is for

**My Profile** stores user-owned context that can help eligible agents work the
way you prefer. It supports global facts such as communication preferences and
workspace-specific context such as a project's goal, conventions, and current
working assumptions.

The profile is local-first and encrypted. Nothing is required during setup:
you may skip personalization, add records manually later, or run the optional
interview. Every action on this page applies immediately; there is no separate
Settings Save button.

## Getting there

Open **Settings** with **F9**, then choose **Data & Privacy > My Profile**. You
can also press **/** in Settings and search for `profile`, `personal context`,
or `interview`.

<!-- personal-context-quick-start:start -->
## In five minutes

1. Open **F9 > Data & Privacy > My Profile**.
2. Choose manual entry or the optional **Get to know you** interview. **Fixed local questions** stay on this device; **Adaptive provider questions** use your default Console provider. Skipping is supported and stores no answers.
3. Review every proposed value and its visibility and syncability controls.
4. Choose **Save only** or **Save and use with agents**, then inspect **Context > Next Send** in Console.
5. Activate and authenticate a supported home server under **Overview > Advanced / Diagnostics > Switch Source / Server**, then use **Server sync > Link to home server** only if you want to share.
<!-- personal-context-quick-start:end -->

## Everyday tasks

### Common workflows

#### Edit manually

Under **Profile records**, choose **Add**, select **Scope** inside the editor,
enter the value, review **Syncability** and **Visibility**, then choose **Save**.
Use the Global scope for preferences that should apply broadly. Use a linked
workspace for its goals, conventions, and working context. **Show** only filters
the list; it does not choose the scope for a new record. An existing record's
scope cannot be changed with **Edit**.

#### Run or rerun an interview

Setup can optionally run **Get to know you** with fixed local questions. To run
an interview later, select Global or a linked workspace with **Show**, choose
**Fixed local questions** or **Adaptive provider questions**, and select **Run
interview again**. Review each proposed row before choosing **Save only** or
**Save and use with agents**.

#### Review agent proposals

Open a row under **Proposed changes**, then choose **Accept**, **Accept
edited**, or **Reject**. New inferred facts remain proposals. **Direct write**
only updates an existing eligible record for an explicit correction evidenced
by the current persisted user message.

#### Export plaintext and recovery material

Set **Show** to the scope you want, then use **Export plaintext: _scope_** for a
readable copy. **Export recovery copy** creates a passphrase-encrypted snapshot
of the manifest, all scopes, current record heads and tombstones, and proposals,
including device-only records. Protect plaintext exports and keep the recovery
passphrase separately. Chatbook does not currently provide a recovery import or
restore action.

#### Remove the local copy

Choose **Remove local profile** only when you intend to destroy the canonical
profile on this device. Export anything you need first. This action also removes
the canonical Personal Context outbox, so queued post-link changes are discarded;
**Manual Sync** cannot send them.

#### Link a home server

First activate and authenticate the server at **Settings > Overview > Advanced /
Diagnostics > Switch Source / Server**. Return to **Data & Privacy > My Profile >
Server sync > Link to home server**. Bootstrap exchanges metadata and downloads
eligible server records and proposals into memory so Chatbook can build the
plan. The review shows content-free IDs, versions, counts, and outcomes—not
profile values. No local profile content uploads before **Approve and link**.

## Create or interview

On a new installation, ordinary application setup finishes first. The setup
summary can offer **Get to know you after setup**, an optional fixed local
interview of at most 20 questions. Choosing **Skip** completes setup without
storing answers or enabling agent use. The setup path does not offer adaptive
questions; that choice is available later in **My Profile**.

The interview offers two question styles:

- **Fixed local questions** make no model or network call.
- **Adaptive provider questions** use the default Console provider and model.
  Tools and streaming are disabled, and the model cannot write the profile.
  Each question request sends the interview audience, coverage topics, attempt
  number, and eligible agent-visible, syncable records from the selected scope.
  After you answer once, later requests also send every prior answered turn,
  including its raw answer text. Provider retention is controlled by that
  provider.

For an adaptive interview, answer entry remains disabled while the first
provider request runs. Chatbook shows the actual provider and model only after
that first response completes and before you can enter an answer. Review this
disclosure before continuing; use **Fixed local questions** if you do not want
the adaptive request to leave the device.

You may skip a question, finish early, keep an encrypted draft when protected
storage is available, or discard the interview. The draft and transcript
objects are local and are not Personal Context Sync payloads. Adaptive requests
still send the material described above to the configured provider. Drafts
expire after 30 days and are destroyed after a successful final review. If
protected storage is unavailable, the draft is memory-only and cannot be
resumed.

Finishing the questions opens a structured review. Only checked rows are saved.
You can edit each proposed value and choose its syncability and agent visibility
before committing. Approved answer text becomes ordinary canonical record
content, so a syncable record can be included in a later reviewed first-link
snapshot. **Save only** leaves runtime agent use as it is; **Save and use with
agents** also requests runtime enablement. Until this final commit, the
interview changes are not records and cannot affect an agent.

Use **Run interview again** at any time. Select Global or a linked workspace
with **Show** first, then choose the question style. A re-interview diffs against
current records; it does not blindly replace the profile.

After creating a workspace, Chatbook may offer **Define project context**.
That interview writes only to the new workspace scope. It cannot silently add
or replace global profile records.

## Records and scopes

Global records follow you across chats. Workspace records are considered only
for the explicitly mapped current workspace. When a workspace and global
record have the same structured key, the workspace value takes precedence for
that workspace; it does not overwrite the global record.

Use **Add**, **Edit**, **Archive/Restore**, and **Delete** to manage records.
Choose a Global or linked-workspace **Scope** inside **Add record**. Records
have a kind and subject rather than being one unstructured biography. Supported
kinds include preferences, identity, relationships, corrections, constraints,
goals, conventions, working context, and legacy notes.

Working context expires after 30 days by default unless you explicitly choose
no expiry. Archived records remain reviewable but are not injected into agent
context. Deleted records become content-free tombstones. Eligible tombstones can
be included in reviewed first-link convergence; a later tombstone remains queued
like other post-link changes.

Each record has two independent controls:

- **Agent visible** allows eligible runtimes to read the record. **User only**
  keeps it out of agent context, tools, search, adaptive interviews, and
  agent-derived artifacts.
- **Syncable** is eligible for the shared home-server profile. An authorized
  home server can read syncable content. **Device only** stays on this
  Chatbook device and never enters the sync outbox.

## Agent authority

Authority is runtime-local and configured separately for every scope:

- **Read only** permits bounded context plus search/get tools.
- **Propose changes** also lets an agent create a reviewable proposal. This is
  the default.
- **Direct write** additionally permits an explicit correction only when the
  exact evidence appears in the current persisted user message and the target
  version has not changed.

These grants do not synchronize to another Chatbook device or to the server.
An agent can never use them to read user-only records, enumerate unrelated
workspaces, change sync or visibility controls, approve its own proposal,
delete records, or purge the profile. Disabling agent use stops context and
tools but does not delete records, change their syncability, or clear queued
Personal Context outbox entries.

## Review proposed changes

Agent-inferred information is never trusted silently. Pending suggestions
appear under **Proposed changes** and remain outside model context. Open one to
see its agent provenance, scope, operation, proposed content, and a warning
that a similar private record may exist without exposing that private record.

Choose one of:

- **Accept** applies the proposal as a user-approved canonical mutation.
- **Accept edited** validates your edited content while keeping the proposal's
  scope and privacy controls fixed.
- **Reject** keeps no proposed value.

Workspace-to-global promotion is always a proposal. Accepting it creates a new
global record with provenance back to the workspace record; it never overwrites
the source or an existing global record.

Acceptance commits the record, manifest, required encrypted local outbox entry,
and terminal proposal receipt in one profile-database transaction. Accept, reject,
supersede, and expiry remove the proposal body and retain only a content-free
receipt. Chatbook also scrubs the obsolete encrypted proposal envelope from
the live profile database and journal artifacts; an already-running read
snapshot is allowed to finish before its pinned journal history is released.
Proposal review cannot be closed while that transaction is running. If the
record changed or the proposal expired, close the review; Settings reloads the
proposal list before another action can be taken.

## Chatbook and tldw_server sharing

Chatbook can reconcile its local profile with a supported, authenticated home
server during reviewed first linking. The two copies remain separate until you
approve the content-free plan and link completion succeeds. That successful
first link publishes the approved eligible snapshot and gives both peers the
same canonical identities and bytes for it. It does not start a shipped ongoing
Personal Context sync cycle.

### What first linking publishes, and what does not sync afterward

<!-- shared-personal-context-contract:start -->
- `tldw_profile_core` defines the versioned canonical profile object models, exact canonical bytes, interview/tool contracts, serialization, and validation used by both peers. Sync-v2 transport envelopes are a separate contract.
- After successful reviewed first linking, Chatbook and tldw_server converge on the same canonical manifest, scope, record, proposal, and version identities and bytes for the eligible snapshot resulting from the user-approved content-free reconciliation plan.
- Sync V2 defines the `personal_context.manifest`, `personal_context.scope`, `personal_context.record`, `personal_context.proposal`, and content-free `personal_context.purge` domains. Reviewed first linking publishes the eligible snapshot resulting from the user-approved content-free reconciliation plan. Later syncable Chatbook mutations create encrypted local outbox entries, but the current shipped app does not run an ongoing Personal Context sync cycle, so those post-link changes remain queued locally. Purge production and distribution are not wired end to end.
- Each peer retains its own at-rest ciphertext and keys, local database rows, runtime permissions, conflict-review metadata, acknowledgement tracking, and other operational state.
<!-- shared-personal-context-contract:end -->

<!-- personal-context-boundary-matrix:start -->
| Published during successful reviewed first linking when eligible | Not published by the shipped ongoing application lifecycle |
| --- | --- |
| Canonical manifest in the snapshot resulting from the user-approved content-free reconciliation plan | Later syncable Chatbook mutations: encrypted outbox entries are created but no shipped ongoing Personal Context caller sends them |
| Required global and linked-workspace scopes in that snapshot | Ordinary server REST mutations: the server copy changes but no Personal Context Sync entry publishes them to Chatbook |
| Eligible record heads, tombstones, and proposal review state selected by reconciliation, including approved interview answer content after it becomes a canonical record payload | Device-only or non-syncable records |
| Exact canonical object identities, versions, and bytes for those eligible objects | Runtime agent authority grants, tool availability, local workspace mappings, and enablement |
| — | Peer-local at-rest encryption/recovery keys, local undo data, caches, ciphertext, database row identities, conflict-review metadata, acknowledgement tracking, and other operational state |
| — | Encrypted interview draft and transcript objects are not Sync payloads as such; adaptive interview requests still send prior raw answers to the configured provider, while approved answer content may become a syncable canonical record as described at left |
<!-- personal-context-boundary-matrix:end -->

After first linking, a syncable Chatbook edit creates an encrypted local outbox
entry, but no shipped ongoing Personal Context caller sends it. **Manual Sync**
handles Notes and Chat only; it does not drain this outbox or provide Personal
Context status. Ordinary server REST edits are not published to linked Chatbook
clients either, so the copies can diverge in either direction after linking.

`personal_context.purge` exists at the protocol boundary, but Chatbook has no
producer and the server endpoint does not distribute it through Sync V2.

Sharing uses `server_trusted_v1`: the authenticated home server can read
syncable canonical content, then encrypt it with its own at-rest keys. Chatbook
accepts both HTTP and HTTPS server URLs; HTTPS is not enforced. Use HTTPS with
**Verify certificates (default)** for any non-loopback server. Runtime calls
honor **Settings > Data & Privacy > Network** choices for default verification,
a **Custom CA bundle**, or **Disable verification**. **Test Connection** always
uses the HTTP client's default certificate verification, so it does not test a
saved custom-CA or verification-off policy. For server key custody and TLS
deployment guidance, see the [server operator guide](https://github.com/rmusser01/tldw_server/blob/dev/Docs/User_Guides/Server/Personal_Context_Profile.md).

Chatbook remains usable without a server. Linking requires the server to
negotiate the required Personal Context capability, domains, schema, and quotas.
During bootstrap, Chatbook exchanges metadata and downloads the server's
sync-eligible record and proposal content into memory before approval. Durable
review state and the review screen remain content-free: they show identities,
versions, counts, and outcomes, not profile values. No local profile content
uploads before approval. Device-only records never enter the first-link snapshot,
and runtime agent authority remains local.

First-link semantic collisions are resolved in this reviewed plan. Generic Sync
metadata may retain a version or semantic conflict encountered by transport,
but Chatbook has no ongoing Personal Context cycle or dedicated Personal Context
status screen. Post-link conflicts retain generic Sync metadata, but there is no
dedicated Personal Context resolution screen.

## Export, removal, and deletion

**Export plaintext** writes the currently selected scope only after explicit
confirmation. It is readable sensitive data. **Export recovery copy** writes
an encrypted whole-profile snapshot protected by the passphrase you supply. It
contains the canonical manifest, scopes, current record heads and tombstones,
and proposals, including device-only records. It does not contain runtime
authority or separate Sync state. The passphrase cannot be recovered, and
Chatbook has no shipped recovery import or restore action.

**Remove local profile** destroys this device's canonical Personal Context
repository: the manifest, scopes, records, proposals, runtime policy, mappings,
local undo and quarantine data, and canonical encrypted outbox. It does not
delete the server copy or unregister this device. It can leave separate Sync
link/profile state, staged encrypted transport envelopes, heads and cursors,
conflict reviews and receipts, and dataset staging keys. Export anything you
need before removal; there is no supported way to drain queued Personal Context
changes first.

If canonical profile-key deletion fails after the rows are removed, Settings
shows **Finish secure removal**. That action retries only the old canonical
profile-key cleanup. It does not clear the separate Sync artifacts, staging
keys, server copy, or device registration.

Chatbook does not currently expose **Delete everywhere**. The authenticated
server purge endpoint creates a server-local purge fence and remains
`purge_pending`; distribution and acknowledgement completion are not wired end
to end. Reconnecting devices does not clear that state. See the [Personal
Context API reference](https://github.com/rmusser01/tldw_server/blob/dev/Docs/API-related/Personal_Context_API.md)
before considering this currently incomplete server operation.

### Troubleshooting

<!-- personal-context-troubleshooting:start -->
| State | Cause | Safe next action | Current limit |
| --- | --- | --- | --- |
| **Profile locked** | Protected key material is unavailable or Chatbook cannot decrypt the existing profile. | Preserve the encrypted data, unlock the configured key protector, and choose **Try again**. | There is no bypass, automatic key recreation, or shipped recovery-import path for existing ciphertext. |
| **Adaptive interview privacy or provider failure** | Adaptive mode sends bounded interview context to the default Console provider, or that provider did not return a usable question. | Use **Fixed local questions** when no model egress is acceptable. After a failure, continue without the interview, retry later, or use the fixed fallback. | The first provider request finishes before Chatbook displays the actual provider/model; do not assume a failed request stayed local. |
| **HTTP or altered TLS verification** | The server URL uses HTTP, or runtime verification uses a custom CA or is disabled. | Prefer HTTPS with **Verify certificates (default)**; review **Data & Privacy > Network** before connecting. | HTTP is accepted, and **Test Connection** always uses default certificate verification rather than the saved custom-CA or verification-off runtime policy. |
| **Post-link change queued** | A syncable local mutation created an encrypted outbox entry after first linking. | Preserve the local profile and export a copy if needed. Treat the server as unchanged. | No shipped Settings action drains this queue; **Manual Sync** covers Notes and Chat only, and there is no Personal Context status screen. |
| **Capability not negotiated** | The peers do not share the required Personal Context domains, schema support, or quotas. | Upgrade or correctly configure the incompatible peer, then retry first linking. | Linking cannot bypass capability negotiation or publish the profile while it is incompatible. |
| **First-link publication interrupted** | You approved reconciliation, but link completion did not finish. | Preserve both copies and retry the reviewed **Link to home server** flow. | Do not treat the copies as converged until completion succeeds. |
| **Version conflict** | The transport encountered changes to the same canonical object from different base versions. | Preserve both peer copies and any generic Sync conflict metadata before making more edits. | No ongoing Personal Context cycle, dedicated status, or dedicated Personal Context resolver is shipped. |
| **First-link semantic collision** | Different local and server record identities describe the same scope, kind, namespace, and subject during linking. | Use the presented content-free IDs, versions, outcomes, and local/server choices to select the lineage that remains active. | The review does not show profile values, and this resolution is available only during first linking. |
| **Post-link semantic collision** | Different record identities describe the same semantic key after linking. | Preserve both peer copies and avoid creating another duplicate. | Post-link conflicts may retain generic Sync metadata, but there is no ongoing Personal Context cycle or dedicated Personal Context resolution screen. |
| **Local removal incomplete or residual state** | Canonical rows were removed but canonical key deletion failed, or separate Sync state and staging keys remain. | Use **Finish secure removal** for canonical profile-key cleanup; preserve any needed evidence before other maintenance. | It does not clear separate Sync state, staging keys, the server copy, or device registration, and the recovery export cannot currently be restored in Chatbook. |
| **Purge pending** | The server purge fence advanced and ordinary profile mutations are blocked. | Treat the server profile as non-writable and consult the server guides before invoking or investigating purge. | Distribution and acknowledgement completion are not wired end to end, and reconnecting devices does not clear `purge_pending`. |
<!-- personal-context-troubleshooting:end -->

If an ordinary server REST edit does not appear in Chatbook, preserve the server
state and avoid creating a duplicate blindly. The server does not publish
ordinary REST edits into Chatbook, and **Manual Sync** cannot deliver that edit.
Preserve and manage the two copies separately; there is no shipped post-link
Personal Context merge or resolution action in Settings.

## Privacy notes

- Stored profile values, proposal bodies, drafts, conflict artifacts, and
  transport envelopes are encrypted at rest. Logs and diagnostics contain
  bounded status and IDs, not profile values or raw interview answers.
- Encryption does not hide all metadata, such as object counts and update
  timing. Under `server_trusted_v1`, the authorized home server can decrypt
  syncable content and re-encrypt it with its own keys. HTTPS protects that
  content in transit; an HTTP URL does not.
- Adaptive interview requests send the bounded selected-scope context and prior
  raw answers described above to the configured provider. Fixed questions do
  not make that provider call.
- **Context > Next Send** in Console is the only disposable preview of the
  exact profile block planned for the next request.
- Profile data is user-owned data, not instructions. It cannot override the
  current request, system instructions, safety rules, or tool permissions.

For what the Console actually sends and which tools each authority exposes,
see [Console chat basics](../console/chat-basics.md#personal-context-in-agent-requests).
