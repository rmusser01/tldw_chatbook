# ADR-119: llama.cpp Prompt-cache Snapshot Ownership

Status: Accepted — reviewed design and approved review amendments

Date: 2026-09-04

Task: [TASK-31552](../tasks/task-31552%20-%20llama.cpp-manual-prompt-cache-snapshot-manager.md)

Design: [Manual prompt-cache snapshots](../../Docs/superpowers/specs/2026-09-04-llamacpp-slot-snapshots-design.md)

Allocation: local heads, remote-tracking refs, and 64 worktrees held ADRs through
118 at drafting. Recheck before integration; this is not a remote reservation.

## Context

llama.cpp PR 26640 enables multimodal slot persistence through its existing
management API. Chatbook already launches local llama-server processes but has
no slot management client, persistent cache catalog, or conversation-to-slot
mapping. The user selected a manual manager first, timestamp-generated names,
and configurable retention with a default of 10.

The API stores model state, not Chatbook history. Busy operations may be deferred;
a failed restore can clear the destination; an acknowledged restore does not
prove subsequent prefix reuse. New serialization is not readable by old servers.
These semantics require explicit lifecycle and disk ownership beyond widget state.

## Decision

1. Compose one app-owned snapshot service for Chatbook-launched llama.cpp servers,
   following ADR-036. Bind operations to immutable launch descriptors and exact
   process claims. The first transport envelope is direct loopback HTTP with
   normal API paths and optional launch-resolved bearer authentication.
   The management client disables environment proxies and redirects, connects
   only to an admitted numeric loopback address, and sends credentials only there.
2. Present **Prompt-cache snapshots** and preserve normal chat history sends.
   Snapshot actions never reopen conversations or promise next-request reuse.
   Per-conversation routing and automatic persistence remain separate work.
3. Store profile-local private binaries and versioned sidecars outside conversation
   storage, sync and export. Apply ADR-029. Use generated basenames, verified file
   ownership, and cross-process catalog locking with the existing portalocker.
4. Expose only a private per-launch working directory to llama-server. Publish
   completed saves into the retained catalog; restore from private working copies.
   This isolates active/uncertain I/O from retained-file pruning and later servers.
   Verify byte length and SHA-256 during restore staging before any Restore POST;
   damaged input must not touch the destination slot. Clean working files after
   acknowledged completion or proven pre-submission failure once local handles
   close. Surface cleanup failures and residual bytes separately; uncertain
   operations retain their files until completion or stop is established.
5. Treat atomic sidecar publication after validated, flushed binary output as the
   catalog commit point. Unacknowledged or incomplete output never triggers pruning
   or becomes implicitly restorable. Reconcile owned interrupted records honestly.
6. Retain the newest 10 committed snapshots across all models per profile by
   default. A canonical config setting controls a positive bounded keep count.
   Prune only after a completed save, by catalog publication order, under the
   store lock. Report cleanup failure separately. Manual deletion is explicit
   permanent deletion with confirmation; there is no automatic pre-restore save.
   Show the effective cross-model keep count beside Save, including narrow layouts.
7. Compare actual model/projector/runtime identity and state-affecting configuration
   before restore. Block known mismatches and missing required evidence in v1.
   Disable Save when required compatibility evidence is missing, and revalidate
   it before publication. Invalidated evidence means no publication or pruning;
   an unusable new save must not evict usable older snapshots.
   Matching configuration is not a guarantee of binary portability or cache reuse.
   The server owns decoding; Chatbook does not implement the packed binary format.
8. Refresh slot observations before acting but make no atomic idle-reservation
   promise. Serialize server Save/Restore operations. Use five-second probe
   deadlines and separate explicit Save/Restore deadlines: five-second connection/
   pool, 30-second write, ten-minute read inactivity and overall submission deadline.
   Show preparation stages and elapsed time without server polling. A timeout
   after possible submission is an unknown outcome, not cancellation, and must
   not cause automatic retries or cleanup.
   Keep Save/Restore disabled for the uncertain generation until resolved/stopped;
   catalog browsing and confirmed deletion remain available.
9. Preserve the source snapshot on restore failure and describe possible loss of
   destination cache. Keep saved records inspectable/deletable while the server
   is unavailable. Show explicit status and recovery without raw response bodies.
10. Require isolated real-server save/restart/restore and measured matching-image
    prefix reuse before declaring the feature complete. Targeted automated tests
    cover storage, compatibility, transport, lifecycle, and production TUI paths.

## Alternatives considered

| Alternative | Reason not selected |
| --- | --- |
| Raw slot ID / filename API form | Moves filename/path risk and retention work to users. |
| Manage every file in a user-selected directory | Cannot establish Chatbook ownership for automatic deletion. |
| Store snapshots in conversation rows | Conflates expendable model cache with durable chat history and sync. |
| Automatic conversation slot binding now | Adds send/branch/routing ownership beyond the selected manual first release. |
| Share one exposed directory across launches and retained files | Late server writes and pruning can race active files or replacement generations. |
| New snapshot database | Small versioned file records and catalog locking suffice for the retained set. |
| Trust file names and HTTP success | Names do not prove model identity; successful restore does not prove reuse. |
| Automatic retry or rollback after timeout | The original server operation may still run; rollback is not atomic. |
| Prune before saving to make room | A failed new save could remove the last useful retained snapshot. |
| Retain and prune for saves with unknown compatibility | Can evict usable snapshots for a new record that v1 refuses to restore. |
| Use shared HTTP client defaults | Proxy routing can violate local-only management; short read deadlines can abandon valid large operations. |
| Continuous slot polling | Entry/action refresh plus observation timestamps serve the manual workflow. |

## Consequences

Users obtain a bounded-count cache manager with understandable save, restore,
and cleanup behavior. Binaries remain local private artifacts and management
does not alter the Console request contract. Service/store/client separation
provides reusable operations if conversation automation is designed later.

Restores may be unavailable after changing runtime/model configuration, and Save
is unavailable until required compatibility evidence is complete. Files
can be large; the keep count is not a byte quota, and active copies require extra
space. Successful operations release their working copies promptly; cleanup
failure remains visible independently of retention. Integrity checks during the
already-required staging copy avoid a second full source-file read. Uncertain
operations can require stopping the managed server, and orphaned
working files remain visible until their writer is known to have stopped.
The v1 transport envelope excludes advanced TLS/prefix/router/shared-network
launches without changing their ordinary launcher support. Windows privacy
reporting retains ADR-029's limitations instead of claiming POSIX-equivalent ACLs.

## References

- [ADR-029: Local private data](029-local-private-data-boundary.md)
- [ADR-036: Application service composition](036-application-service-composition-lifecycle.md)
- [PR 26640](https://github.com/ggml-org/llama.cpp/pull/26640)
- [Reviewed server source](https://github.com/ggml-org/llama.cpp/blob/427291b5b34cd914a31b3fd3b61a68f6184f4b9f/tools/server/server-context.cpp)
- [Slot path issue 26315](https://github.com/ggml-org/llama.cpp/issues/26315)
