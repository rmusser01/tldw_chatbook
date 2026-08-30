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

When negotiated home-server support exposes whole-profile **Delete
everywhere**, it is a separate authenticated shared-profile operation; the
current local-only page must not present **Remove local profile** as a global
purge. Delete everywhere advances the purge generation, destroys canonical
content and derived copies on the server, and causes linked devices to destroy
stale local generations before they may rejoin. A later profile is a new
identity, not a resurrection of the deleted one.

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
