# Console per-chat private scratch space design

Date: 2026-08-23
Status: Approved
Task: TASK-21161
ADR: ADR-081
Baseline: `origin/dev` at `7363592020076c9508fe4a0eee0c1a1679ec7851`

## Summary

Every live Console chat receives an independent temporary scratch directory.
Ordinary Chats require no folder and have no implicit access to the process
working directory or `[console] workspace_root`. Named Workspaces retain the
same private scratch space and may add user folders only through explicit
Workspace bindings.

The change is an authority correction, not only a copy change. It replaces the
shared durable file-tool sandbox for Console runs, removes the local-provider
cwd fallback, routes sandbox-adjacent outputs through the owning chat, and adds
lifecycle fencing that remains correct when a timed-out tool thread finishes
late.

## Problem and evidence

Latest `dev` has four conflicting behaviors:

1. `Tools/file_operation_tools.py` creates one shared durable
   `<user-data>/tool_sandbox` and includes it as the first root for every
   built-in file-tool call.
2. `ConsoleChatController._compose_local_provider()` roots the local `fs_*` and
   Git provider at `[console] workspace_root` or `os.getcwd()` unless project
   instructions select a folder. An ordinary Chat can therefore receive host
   directory access that its built-in tools do not have.
3. `Skills_Interop/local_skills_service.py` retains script output under the
   shared sandbox, and `Agents/run_log.py` uses the same sandbox as its no-
   Workspace fallback.
4. `ConsoleWorkspaceDetails` says `File tools: Off in Default`, while path
   recovery says to bind a folder. The data model permits and tests folderless
   Chats, but the UX presents them as incomplete.

The open PR #1657 / TASK-16316 makes a folder mandatory in the Console's New
Workspace modal. It neither isolates Chats nor removes the local-provider cwd
fallback. It is a conflicting policy, not an implementation dependency.

The relevant targeted baseline on this branch produced 55 passes and one
failure. The failure is test-only: a root-swap regression calls a review hook
without the `run_id` argument that the hook required before that test was
written. The implementation plan must restore that test before using the suite
as verification evidence.

## Goals

- Make ordinary Chats immediately usable without Workspace or folder setup.
- Prevent Chatbook-managed local file operations from crossing live-chat
  scratch boundaries.
- Preserve explicit Workspace folder access, read-only enforcement, project-
  instruction authority, and permission review.
- Make close, late tool completion, reopening, and application shutdown have
  truthful and testable cleanup semantics.
- Verify the user-visible flow with DeepSeek while retaining deterministic
  authority tests as the acceptance gate.

## Non-goals

- A synchronized, persisted, or reopenable per-conversation filesystem.
- A secure-erase guarantee after deletion.
- A new multi-root namespace for local `fs_*` or Git tools.
- Sandboxing third-party MCP servers or provider-hosted tools.
- Changing attachment, Library/RAG, generated-media, or database ownership.
- Changing existing Ask/Allow/Off permissions or sensitive-path policy.
- Making a folder mandatory when creating either a Chat or a Workspace.

## Terminology and ownership

**Private scratch space** means a temporary directory owned by one live
`ConsoleChatSession`. It is shared with that session's turns and subagents but
not with any other live session, including another tab that resumes the same
saved conversation.

**Temporary conversation** continues to mean `ConsoleChatSession.ephemeral`, an
unsaved conversation with additional durability restrictions. The scratch
feature must not reuse `ephemeral` in API or UI names because every Console
session—durable or temporary—gets private scratch.

**Workspace folder access** means only active, validated folder bindings stored
by `LocalWorkspaceRegistryService`. The implementation-only Default workspace
continues to appear as Chats and cannot hold bindings.

The application-owned `ConsoleRuntime` owns the scratch-space manager. The
`ConsoleChatStore` remains a pure session/message owner and does not gain paths,
cleanup callbacks, or filesystem I/O.

## Authority matrix

| Live session state | Built-in file roots | Local `fs_*` / Git root | Folder required |
| --- | --- | --- | --- |
| Chat in Default | Private scratch only | Private scratch only | No |
| Named Workspace, no project selection | Private scratch plus explicit Workspace bindings | Private scratch only | No |
| Named Workspace, selected project binding | Private scratch plus explicit Workspace bindings | Selected binding root, preserving read/write and fingerprint guards | No; selection requires an existing binding |
| Non-Console caller | Existing configured behavior | Existing configured behavior | Unchanged |

Relative file paths resolve inside private scratch when scratch is the working
root. Existing absolute-path handling for explicitly bound Workspace folders is
preserved. The design does not invent an implicit primary binding when a
Workspace has several folders.

## Components

### `ConsoleScratchSpaceManager`

A new Textual-free, thread-safe runtime service owns the mapping from live
session IDs to scratch records. It provides four conceptual operations:

- `snapshot(session_id)`: lazily allocate and return an immutable live snapshot;
- `lease(snapshot)`: reject a tombstoned or replaced snapshot, otherwise count
  one active filesystem operation until the context exits;
- `close(session_id)`: tombstone immediately and delete after the last lease;
- `dispose()`: tombstone all records and perform bounded best-effort cleanup.

Allocation uses an OS temporary parent and an unpredictable directory
name that contains no session or conversation identifier. The manager verifies
the created directory is a real directory, captures canonical identity, and
sets owner-only permissions. Records never leave process memory.

The immutable snapshot contains the canonical root plus an opaque generation
or identity token. A stale snapshot cannot become valid for a new record even
if an operating system later reuses a path.

Tombstoning is synchronous and cheap. Recursive deletion is filesystem I/O and
must never run on Textual's event loop: the manager schedules it through an
owned bounded cleanup worker, while application disposal drains that worker for
a bounded interval. A cleanup failure leaves the record tombstoned and eligible
for a later disposal retry; it never restores access.

### Run-scoped file authority

`ConsoleTurnExecutionContext` captures the owning session's scratch snapshot in
the same owning-session step that already captures provider and Workspace
configuration. The snapshot is detached from viewed-tab state and remains
stable for the turn.

The Console passes this explicit authority into each filesystem consumer:

- `BuiltinToolProvider` and its path precheck use the captured scratch root
  instead of calling the global `_tool_sandbox_root()`;
- `ConsoleChatController._compose_local_provider()` uses scratch when there is
  no explicitly selected project root, never config/cwd;
- retained skill-script output receives the run's scratch root;
- agent run-log fallback receives the run's scratch root.

The non-Console default `_tool_sandbox_root()` remains available for consumers
outside a Console run. No process-global mutation changes its meaning.

### Workspace roots

Built-in file tools keep `run_workspace(workspace_id)` and call-time binding
resolution. Their roots become `(chat_scratch, *live_workspace_bindings)`.
Registry failure still degrades to scratch-only. Deleted, drifted, symlinked, or
read-only bindings retain existing fail-closed behavior.

When project instructions are enabled and an explicit binding is selected,
ADR-069 remains authoritative: the local provider uses that one selected root,
checks its captured locator identity, and omits write tools for a read-only
binding. Scratch does not silently become a second local-provider root.

### Lease integration and late workers

Python cannot kill the daemon thread used for a timed-out tool invocation. File
authority therefore needs a lease around actual filesystem work, not only a
check when composing the catalog.

Every filesystem consumer of the scratch root holds a manager lease for the
duration of its actual access, including built-in/local tools, retained skill
output, and fallback run-log reads and writes. Closing a session proceeds in
this order:

1. tombstone scratch authority, preventing new leases;
2. tombstone/cancel prompt queues, active runs, and subagents through existing
   controller paths;
3. remove the Console session state;
4. delete scratch when its active lease count reaches zero.

If a worker is abandoned, the mapping stays tombstoned and unreachable. Its
eventual lease release triggers deletion. Application disposal retries any
remaining cleanup. Cleanup is idempotent and a cleanup error cannot reactivate
authority.

Ordinary navigation away from Console uses `leave_console`, not session close;
it must not delete scratch belonging to surviving runtime sessions.

## Sandbox-adjacent artifacts

Retained skill-script output currently defaults to a shared
`skill_script_output` directory under the global sandbox. A Console skill run
must instead retain output under the owning chat scratch root. An explicitly
configured non-Console scratch root remains unchanged.

Agent run logs currently prefer a writable Workspace binding and fall back to
the global sandbox. Workspace behavior remains: logs may use the validated
writable binding under existing hidden-directory rules. The fallback becomes
the owning chat scratch root so Default Chat logs cannot accumulate in a
cross-chat container.

No scratch root or raw contents may be added to ordinary payload logs. Existing
UI affordances that read a run's full retained result must receive the owning
run/session authority instead of searching another chat's fallback root.

## UX and recovery

The Chats status row should use scoped, truthful copy:

- label: `Local file tools`
- Default value: `Private scratch`
- named Workspace value: `Private scratch + N folder(s)` when bindings exist

Creating a Chat never opens a folder picker or blocks on Workspace state. A
named Workspace may still be created without a folder; it behaves as scratch-
only until a folder is explicitly bound.

When a tool requests a path outside scratch and all bound roots, recovery copy
should explain: use or create a named Workspace and bind that external folder
if access is intended. It must not describe the current Chat as broken or say a
folder is required for conversation.

The copy is deliberately limited to **local file tools**. MCP servers and
provider-hosted tools may have separately configured capabilities that this
boundary does not control.

## Security and privacy invariants

- Scratch paths are random and contain no chat, session, conversation, title,
  provider, or user identifiers.
- Root directories are owner-only and are never symlinks.
- Snapshots are generation-fenced; stale snapshots fail closed.
- Chat A's provider, precheck, tool invocation, skill output, and fallback run
  log never resolve Chat B's scratch root.
- Scratch is always an allowed root; Workspace bindings can add authority but
  never replace or widen it by common-ancestor inference.
- Permission decisions, kill switches, sensitive-path exclusions, and
  read-only filtering execute as they do today.
- The implementation passes the exact manager-owned root into existing path
  validation; it does not add a broad `/tmp` or OS-temporary sensitive-path
  exemption.
- Cleanup logs bounded categories and opaque identifiers, not raw paths or
  filenames.
- Normal deletion is not represented as secure erase.

## Persistence and crash behavior

No scratch locator enters `ConsoleChatSession`, conversation metadata, SQLite,
sync payloads, snapshots, exports, or app configuration. A saved conversation
resumed after close or restart receives a new empty scratch directory.

Clean close and app shutdown perform best-effort deletion. A hard crash may
leave an OS-temporary directory. A later process does not enumerate or attach
old scratch directories, preventing cross-chat reuse. Cross-process stale-root
reclamation is deferred because safe deletion would require process ownership
and locking semantics beyond this task.

## Test strategy

Implementation follows red-green TDD with real filesystem operations.

### Manager and authority tests

- two sessions receive distinct canonical owner-only roots;
- relative writes in A are absent from B and direct B access to A is refused;
- a stale/tombstoned snapshot cannot acquire a new lease;
- close schedules off-loop deletion with no lease and defers cleanup while an active lease remains;
- last lease release deletes a tombstoned root;
- close/dispose are idempotent and cleanup failure never restores access;
- reopening the same persisted conversation receives a fresh empty root.

### Provider and artifact tests

- Default Console local-provider composition uses scratch, not config/cwd;
- built-in precheck and invocation use the identical captured scratch root;
- named Workspace built-ins receive scratch plus current bindings;
- registry failure produces scratch-only roots;
- selected project root and read-only behavior remain unchanged;
- skill-script output and fallback run logs use the owning chat root;
- two simultaneous runs cannot resolve each other's retained outputs/logs;
- non-Console global-sandbox callers retain existing behavior.

The stale `test_selected_root_swap_fails_closed_before_local_invoke` invocation
must pass an explicit run ID and continue proving that a retargeted project root
is refused before file contents are exposed.

### UI tests

- New Chat in Chats creates a session without a modal or folder requirement;
- status copy says `Private scratch` rather than `Off in Default`;
- Workspace copy reports only actual bindings;
- recovery copy presents folder binding as optional external-folder access.

### Live DeepSeek UAT

Use an isolated application profile and latest-`dev` worktree. When the user's
DeepSeek credential lives in config rather than an environment variable, copy
the config without printing it into a private mode-0600 temporary profile,
point `TLDW_CONFIG_PATH` at that copy, and delete the profile after UAT. Record
the original config hash before and after; never place the credential in a
command argument, repository file, screenshot, or log.

1. Select DeepSeek and start a Chat under Chats with no folder.
2. Send a plain prompt and observe a successful streamed response.
3. Ask DeepSeek to write and read a unique relative scratch file, approving the
   normal local file-tool request.
4. Start a second Chat and verify the first marker cannot be listed or read.
5. Create or select a named Workspace bound to a disposable UAT folder; verify
   an allowed read and, for an explicit read-write binding, a write.
6. Return to a Chat and verify the Workspace file is outside its authority.
7. Close the first Chat, reopen its saved conversation, and verify the old
   scratch marker is absent.
8. Confirm no folder prompt appeared for either Chat and the user config hash
   is unchanged.

The live model's exact tool-selection sequence is observational evidence, not a
deterministic assertion. Scripted provider tests exercise the same production
composition when a repeatable tool sequence is required.

## Documentation impact

Update the Console user guide, Settings/Workspace guidance, tool-calling docs,
and AGENTS.md's Console tool-authority summary. Explicitly state that
`[console] workspace_root` is no longer a Console Chat authority fallback.

## ADR check

ADR required: yes

ADR path: `backlog/decisions/081-console-per-chat-private-scratch-space.md`

Reason: the change establishes filesystem authority, temporary-data ownership,
provider composition, and teardown behavior across Console, Workspaces, tools,
skills, and run logs.

## Open risks

- DeepSeek may answer a tool prompt without invoking a tool; live UAT may need
  a more explicit instruction, while deterministic tests remain authoritative.
- Timed-out third-party or non-cooperative code may hold a lease until process
  exit. Tombstoning prevents reuse, but cleanup cannot force the thread to die.
- A hard crash can leave OS-temporary residue. No later chat can discover it,
  but secure crash cleanup is outside this design.
- Workflows that unknowingly relied on Console's cwd fallback will lose that
  access and must opt into a named Workspace binding. That break is intentional
  and should be called out in release notes.
- This task preserves existing per-call content and run-budget limits but does
  not introduce a cumulative per-chat disk quota. A separate quota policy would
  need its own UX and cleanup decision; existing permission prompts and bounded
  run tool counts remain the immediate abuse controls.
