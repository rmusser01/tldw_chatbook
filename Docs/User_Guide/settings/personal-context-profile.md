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

During initial setup only, leave **Get to know you after setup** unchecked to
opt out. Setup finishes before an interview selected there opens.

<!-- personal-context-quick-start:start -->
## In five minutes

1. Open **F9 > Data & Privacy > My Profile**.
2. **Manual:** if needed, choose **Create profile**. Use **Add** for a new record or **Edit** for an existing one, review its scope, visibility, and syncability, then choose **Save**.
3. **Interview:** select a scope with **Show**, choose a **Question style**, and select **Run interview again**. Review every proposed row and its controls, then choose **Save only** or **Save and use with agents**.
4. After either path, agent use is optional. In Console, press **Ctrl+Shift+P** (**View context**) to open **Conversation Inspector**; select the outer **Next Send** tab, then the inner **Next Send** payload tab before sending.
5. Activate and authenticate a supported home server under **Overview > Advanced / Diagnostics > Switch Source / Server**, then use **Server sync > Link to home server** only if you want to share.
<!-- personal-context-quick-start:end -->

## Common workflows

### Edit manually

Under **Profile records**, choose **Add**, select **Scope** inside the editor,
enter the value, review **Syncability** and **Visibility**, then choose **Save**.
Use the Global scope for preferences that should apply broadly. Use a linked
workspace for its goals, conventions, and working context. **Show** only filters
the list; it does not choose the scope for a new record. An existing record's
scope cannot be changed with **Edit**.

### Run or rerun an interview

Setup can optionally run **Get to know you** with fixed local questions. To run
an interview later, select Global or a linked workspace with **Show**, choose
**Fixed local questions** or **Adaptive provider questions**, and select **Run
interview again**. Review each proposed row before choosing **Save only** or
**Save and use with agents**.

### Review agent proposals

Open a row under **Proposed changes**, then choose **Accept**, **Accept
edited**, or **Reject**. New inferred facts remain proposals. **Direct write**
only updates an existing eligible record for an explicit correction evidenced
by the current persisted user message.

### Export plaintext and recovery material

Set **Show** to the scope you want, then use **Export plaintext: _scope_** for a
readable copy. **Export recovery copy** creates a passphrase-encrypted snapshot
of the manifest, all scopes, current record heads and tombstones, and proposals,
including device-only records. Protect plaintext exports and keep the recovery
passphrase separately. Chatbook does not currently provide a recovery import or
restore action.

### Remove the local copy

Choose **Remove local profile** only when you intend to destroy the canonical
profile on this device. Export anything you need first. This action also removes
the canonical Personal Context outbox, so queued post-link changes are discarded;
**Manual Sync** cannot send them.

### Link a home server

First activate and authenticate the server at **Settings > Overview > Advanced /
Diagnostics > Switch Source / Server**. Return to **Data & Privacy > My Profile >
Server sync > Link to home server**. Bootstrap exchanges metadata and downloads
eligible server records and proposals into memory so Chatbook can build the
plan. The review shows content-free IDs, versions, counts, and outcomes—not
profile values. No local profile content uploads before **Approve and link**.

## Create or interview

On a new installation, the setup summary can offer **Get to know you after
setup**. Leave it unchecked to opt out without storing interview answers or
enabling agent use. Setup completes before the selected interview opens. The
chained interview uses at most 20 fixed local questions; adaptive questions are
available later in **My Profile**.

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

Within an interview, **Skip** skips only the current question. **Cancel** opens
**Leave interview**: choose **Continue interview** to return, **Keep draft** to
exit and retain the encrypted draft, or **Discard draft** to exit and destroy
its draft key. A memory-only interview cannot be kept, so it offers only
continue or discard. You may also finish early. The draft and transcript objects
are local and are not Personal Context Sync payloads. Adaptive requests still
send the material described above to the configured provider. Drafts expire
after 30 days and are destroyed after a successful final review. If protected
storage is unavailable, the draft is memory-only and cannot be resumed.

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
| Published at reviewed first link when eligible | Not published afterward or peer-local |
| --- | --- |
| Approved eligible canonical manifest | Later syncable Chatbook mutations, which remain queued locally |
| Required global and linked-workspace scopes | Ordinary server REST mutations |
| Controls-eligible record heads and tombstones; eligible proposals and canonical review state; approved interview answers after they are saved as records | Device-only or non-syncable records |
| Exact canonical IDs, versions, and bytes | Runtime agent authority, tool availability, workspace mappings, and enablement |
| — | At-rest and recovery keys; local undo, caches, ciphertext, database row IDs, conflict-review objects, acknowledgement tracking, and operational metadata |
| — | Interview draft and transcript objects; adaptive requests still send prior raw answers to the provider |
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
accepts both HTTP and HTTPS server URLs; HTTPS is not enforced. HTTP is
unencrypted. HTTPS protects transport privacy only when Chatbook verifies a
valid server certificate through **Verify certificates (default)** or a
correctly configured **Custom CA bundle**. With **Disable verification**,
Chatbook does not authenticate the server and an on-path attacker can intercept
the connection. Runtime calls honor the saved choice under **Settings > Data &
Privacy > Network**. **Test Connection** always uses the HTTP client's default
certificate verification, so it does not test a saved custom-CA or
verification-off policy. For server key custody and TLS deployment guidance,
see the [server operator guide](https://github.com/rmusser01/tldw_server/blob/dev/Docs/User_Guides/Server/Personal_Context_Profile.md).

Chatbook remains usable without a server. Linking requires the server to
negotiate the required Personal Context capability, domains, schema, and quotas.
During bootstrap, Chatbook exchanges metadata and downloads the server's
sync-eligible record and proposal content into memory before approval. Durable
review state and the review screen remain content-free: they show identities,
versions, counts, and outcomes, not profile values. No local profile content
uploads before approval. Device-only records never enter the first-link snapshot,
and runtime agent authority remains local.

**First-link conflicts.** A version conflict means the device and server object
versions follow different canonical lineages. The content-free review shows the
canonical ID and versions; choose **Keep this device** or **Keep server**.
First-link semantic collisions use the same choices for different record IDs
with one semantic key. The review shows IDs, versions, counts, and outcomes—not
profile values.

**Post-link conflicts.** Generic Sync metadata may retain later version or
semantic conflicts, but the shipped app has no ongoing Personal Context cycle,
status screen, or dedicated resolver. The first-link lineage choices are not a
post-link resolution tool.

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
| **Profile locked** | The profile key is unavailable or cannot decrypt the data. | Unlock the key protector, preserve the data, and choose **Try again**. | No bypass, automatic key recreation, or recovery import is shipped. |
| **Adaptive interview privacy or provider failure** | Adaptive mode contacted the default Console provider, or it returned no usable question. | Use **Fixed local questions**, continue without the interview, or retry later. | The first request finishes before provider/model disclosure; a failed request may have left the device. |
| **HTTP or altered TLS verification** | The URL is HTTP, or custom/disabled verification is selected. | Use HTTPS with **Verify certificates (default)** or a correctly configured custom CA. | HTTP is unencrypted. Disabled verification does not authenticate the server and permits interception. **Test Connection** always uses default verification. |
| **Post-link change queued** | A syncable local change created an encrypted outbox entry after linking. | Preserve the local profile and export it if needed; treat the server as unchanged. | No shipped action drains this queue; **Manual Sync** covers Notes and Chat only. |
| **Capability not negotiated** | Required domains, schema support, or quotas do not match. | Upgrade or configure the incompatible peer, then retry first linking. | Linking cannot bypass negotiation or publish an incompatible profile. |
| **First-link publication interrupted** | Approval occurred, but link completion failed. | Preserve both copies and retry the reviewed **Link to home server** flow. | The copies have not converged until completion succeeds. |
| **Version conflict** | At first link, device and server versions follow different canonical lineages. | Review the content-free ID and versions; choose **Keep this device** or **Keep server**. | This lineage choice is first-link only; no profile values are shown. |
| **First-link semantic collision** | Different record IDs have the same scope, kind, namespace, and subject. | Review the content-free IDs, versions, and outcomes; choose **Keep this device** or **Keep server**. | The choice exists only during first linking; no profile values are shown. |
| **Post-link semantic collision** | Later records use the same semantic key, or later versions conflict. | Preserve both peer copies and avoid another duplicate. | Generic Sync metadata may remain, but no ongoing Personal Context cycle, status, or resolver is shipped. |
| **Local removal incomplete or residual state** | Key deletion failed, or separate Sync artifacts remain. | Use **Finish secure removal** for profile-key cleanup; preserve needed evidence. | It does not clear separate Sync state, staging keys, server data, or device registration; recovery import is not shipped. |
| **Purge pending** | The server purge fence blocks ordinary profile mutations. | Treat the server profile as non-writable and consult the server guides. | Distribution and acknowledgements are incomplete; reconnecting does not clear `purge_pending`. |
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
  content in transit only when Chatbook verifies a valid server certificate;
  HTTP is unencrypted, and disabled verification allows interception without
  server authentication.
- Adaptive interview requests send the bounded selected-scope context and prior
  raw answers described above to the configured provider. Fixed questions do
  not make that provider call.
- In Console, press **Ctrl+Shift+P** (**View context**) to open **Conversation
  Inspector**, then select the outer **Next Send** tab and the inner **Next
  Send** payload tab. It is the only disposable preview of the exact profile
  block planned for the next request.
- Profile data is user-owned data, not instructions. It cannot override the
  current request, system instructions, safety rules, or tool permissions.

For what the Console actually sends and which tools each authority exposes,
see [Console chat basics](../console/chat-basics.md#personal-context-in-agent-requests).
