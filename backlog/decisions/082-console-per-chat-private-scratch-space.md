# ADR-082: Give every Console chat private temporary scratch space

Status: Accepted
Date: 2026-08-23
Related Task: TASK-21161

## Decision

Each live Console chat session owns one unpredictably named, owner-only temporary
scratch directory. The directory is process-local authority: its locator is
never persisted, synchronized, derived from a session or conversation ID, or
reattached when a saved conversation is reopened.

The Console runtime owns a scratch-space manager. Allocation is lazy. A turn
captures an immutable scratch snapshot containing the canonical root and an
opaque generation token; every Chatbook-managed local filesystem provider
receives that explicit snapshot instead of resolving a global default. The
manager validates that the snapshot is still live before new filesystem work
starts.

Default-workspace conversations, presented to users as **Chats**, receive only
their private scratch root. Console composition no longer treats
`[console] workspace_root` or the process working directory as filesystem
authority for these sessions. Those compatibility settings may remain for
non-Console consumers, but they do not widen a Console chat.

Named Workspace sessions also receive private scratch. Their existing explicit
folder bindings remain additional roots for the built-in multi-root file tools.
The single-root local `fs_*` and Git provider uses private scratch unless the
session has explicitly selected a project-instruction folder binding; when it
has, the selected binding remains that provider's root and keeps the existing
locator-fingerprint, read-only/read-write, and path-target guards from ADR-069.
No implicit "first folder" or common-ancestor root is invented for a
multi-binding Workspace.

All Console paths that previously fell back to the shared file-tool sandbox
must consume the owning chat's scratch snapshot. This includes built-in file
tool validation and invocation, local `fs_*`/Git composition when no project
root is selected, retained skill-script output, and fallback agent run logs.
Non-filesystem local tools keep their current behavior.

Permission review remains unchanged. Private temporary storage is a
confinement boundary, not permission to bypass Ask/Allow/Off decisions,
sensitive-path checks, tool kill switches, or read-only Workspace bindings.
Third-party MCP servers, attachments, Library/RAG records, generated media,
and provider-side tools keep their independent authority contracts; Console
copy must describe **local file tools** rather than claiming universal
sandboxing.

Closing a chat tombstones its scratch space before session state is removed.
Tombstoning rejects new leases immediately. Every consumer that reads or writes
inside scratch—including file tools, retained skill output, and fallback run
logs—holds a lease while doing so. Normal cleanup runs off the Textual event
loop when the last lease drains. A timed-out or cancelled tool
may continue on an abandoned daemon thread, so close must never reuse or
reattach the directory and must defer deletion rather than claiming the thread
was killed. Application disposal tombstones every remaining scratch space and
performs best-effort cleanup. Ordinary navigation away from the Console does
not close sessions and therefore does not delete their scratch spaces.

Deletion is ordinary recursive cleanup, not secure erase. A hard process or OS
crash can leave an unreferenced operating-system temporary directory. A later
Chatbook process never discovers or attaches it, so it cannot become another
chat's authority. Strong crash-residue reclamation would require a separate
cross-process ownership/locking design.

The Console presents Default Chats as `Local file tools: Private scratch` and
named Workspaces as private scratch plus their explicit folder access. A path
denial outside all allowed roots explains that an external folder can be added
through a named Workspace; it never says a folder is required to chat.

## Context

ADR-027 intentionally hides the implementation-only Default workspace behind
the user-facing Chats section so ordinary conversation does not require
Workspace vocabulary. ADR-028 says Default is sandbox-only, while named
Workspaces may add explicitly bound folder roots. The implementation drifted
from that model in two directions:

1. `file_operation_tools._tool_sandbox_root()` resolves one shared durable
   `<user-data>/tool_sandbox`, allowing files to outlive and be reachable from
   unrelated chats.
2. `ConsoleChatController._compose_local_provider()` uses
   `[console] workspace_root` or `os.getcwd()` when no project root is selected,
   so an ordinary Chat can receive broader local `fs_*`/Git access than the
   built-in sandboxed tools.

Presentation then obscures both facts by rendering `File tools: Off in
Default`, while path-denial recovery tells users to bind a folder. The result
looks like a folder is required for chatting even though session creation and
provider dispatch do not require one.

The Console agent runtime also abandons timed-out tool calls on daemon threads
because Python cannot kill a thread. Immediate unconditional directory removal
on close would therefore race real work. Lifecycle fencing and leases are part
of the authority decision, not an optional cleanup refinement.

Open PR #1657 / TASK-16316 proposes a required-folder Console Workspace modal.
That policy does not solve cross-chat sandbox leakage or the cwd fallback and
must not be treated as this decision's implementation.

## Alternatives Considered

| Option | Why rejected |
| --- | --- |
| Keep the shared durable sandbox and only change Console copy | Cross-chat file discovery remains possible and "private" would be false. |
| Persist one sandbox per conversation | Contradicts the temporary lifecycle, creates cleanup and synchronization semantics, and lets reopening recover old scratch files. |
| Use the live session ID as the directory name | Exposes durable-looking identifiers in filesystem paths and logs without adding useful authority. |
| Keep `[console] workspace_root`/cwd for legacy-disabled sessions | Leaves the central policy violation intact: Chats retain implicit host-folder access. |
| Give local `fs_*` an arbitrary first Workspace binding | Makes relative paths silently retarget when bindings reorder and bypasses explicit project-root selection. |
| Refactor local `fs_*`/Git to a new multi-root namespace | Considerably enlarges the security surface and relative-path grammar; built-in tools already own multi-root access. |
| Delete immediately when a chat closes | Races abandoned daemon tool threads and falsely assumes cancellation kills filesystem work. |
| Treat temporary scratch as approval-free | Conflates path confinement with user authorization and weakens the existing permission boundary. |
| Securely erase files | Filesystem and SSD behavior cannot guarantee erasure; truthful best-effort deletion is the supportable contract. |

## Consequences

- A new runtime-owned scratch manager and immutable scratch snapshot become the
  single Console authority source for temporary local files.
- A live session, not a persisted conversation, is the ownership unit. Two
  simultaneous tabs for the same saved conversation still receive different
  scratch spaces.
- Subagents and concurrent turns belonging to one live chat share its scratch
  space; unrelated live chats never do.
- Default Chats lose implicit `[console] workspace_root`/cwd access. This is an
  intentional security correction and may expose workflows that relied on the
  undocumented widening.
- Named Workspaces retain current folder-binding and project-instruction
  behavior. A Workspace with no folder binding still works with scratch only.
- Cleanup needs thread-safe tombstone, lease, deferred-delete, off-loop removal,
  and idempotent disposal behavior. Cleanup failures are bounded diagnostics and
  never restore authority.
- No database migration or synchronized state is introduced.
- Live DeepSeek UAT is required for provider-facing confidence, but deterministic
  authority and lifecycle tests remain the acceptance evidence because model
  tool selection is nondeterministic.

## Links

- [Design spec](../../Docs/superpowers/specs/2026-08-23-console-per-chat-private-scratch-space-design.md)
- [TASK-21161](../tasks/task-21161%20-%20Console-per-chat-private-scratch-space-and-workspace-file-authority.md)
- [ADR-027: Default-workspace conversations live in Chats](027-default-workspace-chats-in-chats-section.md)
- [ADR-028: Settings workspaces and folder roots](028-settings-workspaces-category-and-folder-roots.md)
- [ADR-032: Local agent tool permission boundary](032-local-agent-tool-permission-boundary.md)
- [ADR-069: Console project-instruction local state and preflight](069-console-project-instruction-local-state-and-preflight.md)
- [Conflicting open PR #1657](https://github.com/rmusser01/tldw_chatbook/pull/1657)
