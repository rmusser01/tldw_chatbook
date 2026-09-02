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
2. Choose **Create profile**, then **Add** for manual entry, or use the optional **Get to know you** interview. **Skip** is supported and stores no answers.
3. Review every proposed value and its visibility and syncability controls.
4. Choose **Save only** or **Save and use with agents**, then inspect **Context > Next Send** in Console.
5. Choose **Server sync > Link to home server** only if you want to share with a supported home server.
<!-- personal-context-quick-start:end -->

## Everyday tasks

### Common workflows

#### Edit manually

Under **Profile records**, set **Show** to Global, choose **Add** or **Edit**,
enter the preference, review **Syncability** and **Visibility**, then choose
**Save**. For project context, set **Show** to a linked workspace and add its
goals or conventions there instead.

#### Run or rerun an interview

Use the optional **Get to know you** interview during setup, or set **Show** to
Global or a linked workspace, choose a **Question style**, and select **Run
interview again**. Review each proposed row before choosing **Save only** or
**Save and use with agents**.

#### Review agent proposals

Open a row under **Proposed changes**, then choose **Accept**, **Accept
edited**, or **Reject**. New inferred facts remain proposals. **Direct write**
only updates an existing eligible record for an explicit correction evidenced
by the current persisted user message.

#### Export plaintext and recovery material

Set **Show** to the scope you want, then use **Export plaintext: _scope_** for a
readable copy or **Export recovery copy** for an encrypted whole-profile copy.
Protect the plaintext file and keep the recovery passphrase separately.

#### Remove the local copy

Choose **Remove local profile** only when you intend to destroy this device's
readable copy and local keys. Export or synchronize wanted changes first.

#### Link a home server

Under **Server sync**, choose **Link to home server**. Review every identity,
record, proposal, collision, and workspace outcome, then choose **Approve and
link**. Nothing is uploaded before this reviewed first-link step succeeds.

## Create or interview

On a new installation, ordinary application setup finishes first. The setup
wizard may then offer **Get to know you**, an optional interview of at most 20
questions. Choosing **Skip** completes setup without creating answers or
enabling agent use.

The interview offers two question styles:

- **Fixed local questions** make no model or network call.
- **Adaptive provider questions** show the selected provider and model before
  the first answer. The interview model receives no tools and cannot write the
  profile.

You may skip a question, finish early, save an encrypted draft when protected
storage is available, or discard the interview. Raw questions and answers stay
local, are never synchronized, expire after 30 days, and are destroyed after a
successful final review. If protected storage is unavailable, the draft is
memory-only and cannot be resumed.

Finishing the questions opens a structured review. Only checked rows are
saved. You can edit each proposed value and choose its syncability and agent
visibility before committing. **Save only** leaves runtime agent use as it is;
**Save and use with agents** also requests runtime enablement. Until this final
commit, the interview changes are not records and cannot affect an agent.

Use **Run interview again** at any time. Select Global or a linked workspace
with the scope filter first. A re-interview diffs against current records; it
does not blindly replace the profile.

After creating a workspace, Chatbook may offer **Define project context**.
That interview writes only to the new workspace scope. It cannot silently add
or replace global profile records.

## Records and scopes

Global records follow you across chats. Workspace records are considered only
for the explicitly mapped current workspace. When a workspace and global
record have the same structured key, the workspace value takes precedence for
that workspace; it does not overwrite the global record.

Use **Add**, **Edit**, **Archive/Restore**, and **Delete** to manage records.
Records have a kind and subject rather than being one unstructured biography.
Supported kinds include preferences, identity, relationships, corrections,
constraints, goals, conventions, working context, and legacy notes.

Working context expires after 30 days by default unless you explicitly choose
no expiry. Archived records remain reviewable but are not injected into agent
context. Deleted records become content-free tombstones so deletion can safely
converge between replicas.

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
tools but does not delete records or silently stop synchronization.

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

Acceptance commits the record, manifest, required sync outbox, and terminal
proposal receipt in one profile-database transaction. Accept, reject,
supersede, and expiry remove the proposal body and retain only a content-free
receipt. Chatbook also scrubs the obsolete encrypted proposal envelope from
the live profile database and journal artifacts; an already-running read
snapshot is allowed to finish before its pinned journal history is released.
Proposal review cannot be closed while that transaction is running. If the
record changed or the proposal expired, close the review; Settings reloads the
proposal list before another action can be taken.

## Chatbook and tldw_server sharing

The shared Chatbook/tldw_server contract defines one logical Personal Context
Profile, not separate app-specific records that must be manually reconciled.
When a server supports and negotiates that contract, syncable records keep the
same canonical profile, scope, record, version, and provenance identities on
both peers; each peer encrypts the canonical bytes with its own at-rest keys.
This is also the contract for converging multiple Chatbook devices through one
home server.

### What currently synchronizes

<!-- shared-personal-context-contract:start -->
- `tldw_profile_core` defines the versioned canonical profile object models, exact canonical bytes, interview/tool contracts, serialization, and validation used by both peers. Sync-v2 transport envelopes are a separate contract.
- After a successful reviewed link, Chatbook and tldw_server converge on the same canonical manifest, scope, record, proposal, and version identities and bytes for eligible shared objects.
- Sync V2 defines the `personal_context.manifest`, `personal_context.scope`, `personal_context.record`, `personal_context.proposal`, and content-free `personal_context.purge` domains. The current linked flow publishes eligible Chatbook-originated manifest, scope, record, and proposal changes; purge production and distribution are not wired end to end.
- Each peer retains its own at-rest ciphertext and keys, local database rows, runtime permissions, conflict-review metadata, acknowledgement tracking, and other operational state.
<!-- shared-personal-context-contract:end -->

<!-- personal-context-boundary-matrix:start -->
| Shared through the current linked flow when eligible | Remains peer-local or is not currently published |
| --- | --- |
| Canonical manifest after successful reviewed linking | Peer-local at-rest encryption and recovery keys |
| Required global and linked-workspace scope objects | Raw interview answers and unfinished drafts |
| Records and tombstones whose controls permit synchronization | Runtime agent authority grants and tool availability |
| Eligible proposals and their canonical review state | Device-only records or records marked non-syncable |
| Exact canonical object identities, versions, and bytes for eligible shared objects | Local undo history, caches, ciphertext, database row identities, and other operational metadata |
| — | Conflict-review objects and acknowledgement tracking |
<!-- personal-context-boundary-matrix:end -->

Ordinary server REST edits are not currently published to linked Chatbook clients.
`personal_context.purge` exists at the protocol boundary, but Chatbook has no producer and the server endpoint does not distribute it through Sync V2.

Sharing uses `server_trusted_v1`: the authenticated home server can read
syncable canonical content over TLS, then encrypt it with its own at-rest keys.
For server key custody and TLS guidance, see the [server operator
guide](https://github.com/rmusser01/tldw_server/blob/dev/Docs/User_Guides/Server/Personal_Context_Profile.md).

Chatbook remains usable without a server. Before a home server advertises and
negotiates Personal Context sync, syncable records remain local and queued
rather than being sent through the older server-personalization UI. Device-only
records never acquire a server representation. Runtime agent authority remains
local even after record synchronization.

Under that shared contract, conflicting concurrent edits are retained for user
review; neither peer uses last-write-wins for profile content. Pending
proposals may synchronize for review, but raw interview answers, device-only
records, undo data, and local authority grants do not.

## Export, removal, and deletion

**Export plaintext** writes the currently selected scope only after explicit
confirmation. It is readable sensitive data. **Export recovery copy** writes
an encrypted whole-profile snapshot protected by the passphrase you supply;
the passphrase cannot be recovered.

**Remove local profile** destroys this device's readable local copy. It is not
the same as deleting the shared profile from the home server. When a linked
profile has unsynchronized changes, export or synchronize them before local
removal unless you intentionally choose to discard them.

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
| **Profile locked** | Chatbook cannot decrypt the profile because protected key material is unavailable or locked. | Preserve the encrypted profile, unlock or restore the configured key, then retry. | There is no bypass or automatic key-recreation path for existing ciphertext. |
| **Offline or queued** | Local changes remain in Chatbook's outbox because the home server is unreachable or authentication failed. | Continue locally, restore connectivity and credentials, then retry Sync and inspect its outbox/status. | The server cannot inspect a device-local queue until Chatbook delivers it. |
| **Capability not negotiated** | The peers do not share the required Personal Context domains, schema support, or quotas. | Upgrade or correctly configure the incompatible peer, then negotiate again. | There is no supported bypass; linking and upload remain blocked. |
| **Version conflict** | Both peers changed the same canonical object from different base versions. | Preserve the conflict and inspect generic Sync status and metadata before editing again. | No dedicated Personal Context post-link resolver is currently shipped. |
| **First-link semantic collision** | Different local and server record identities describe the same scope, kind, namespace, and subject during linking. | Compare the presented records and choose the outcome in the first-link reconciliation review. | This resolver is available only during reviewed first linking. |
| **Post-link semantic collision** | Different record identities describe the same semantic key after linking. | Preserve both sides and inspect generic Sync status and metadata. | Post-link conflicts retain generic Sync metadata; there is no dedicated Personal Context resolution screen. |
| **Purge pending** | The server purge fence advanced and ordinary profile mutations are blocked. | Preserve operational evidence and treat the server profile as non-writable; consult the server guides before acting. | Distribution and acknowledgement completion are not wired end to end, and reconnecting devices does not clear `purge_pending`. |
<!-- personal-context-troubleshooting:end -->

If an ordinary server REST edit does not appear in Chatbook, preserve the server
state and avoid creating a duplicate blindly. The current server does not
publish ordinary REST edits into the linked Chatbook Sync path, so retrying Sync
cannot deliver that edit. Use Chatbook for future edits that must travel through
the linked Sync path.

## Privacy notes

- Profile values, proposal bodies, drafts, conflicts, and exact sync snapshots
  are encrypted at rest. Logs and diagnostics contain bounded status and IDs,
  not profile values or raw interview answers.
- Encryption does not hide all metadata, such as object counts and update
  timing. Syncable content is decrypted over authenticated TLS by the
  authorized home server and re-encrypted there.
- **Context > Next Send** in Console is the only disposable preview of the
  exact profile block planned for the next request.
- Profile data is user-owned data, not instructions. It cannot override the
  current request, system instructions, safety rules, or tool permissions.

For what the Console actually sends and which tools each authority exposes,
see [Console chat basics](../console/chat-basics.md#personal-context-in-agent-requests).
