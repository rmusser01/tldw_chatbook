# Lasting Notes folder sync

Lasting folder sync connects one or more local folders to Library Notes. It is
the only production Notes-to-files sync owner. The retired single-folder
engine, its Library panel, its five-minute timer, and its config writer are no
longer present.

For the user workflow, see
[Library Notes — Add from files and lasting sync](../User_Guide/library/notes.md#add-from-files-and-lasting-sync).

## Safety model

Folder sync follows [ADR-059](../../backlog/decisions/059-notes-folder-import-and-device-local-sync-ownership.md)
and [ADR-073](../../backlog/decisions/073-notes-sync-round-trip-and-interoperability-constraints.md):

- **Check is read-only.** Scanning produces a bounded reviewed plan and an
  observation token. It does not create a root, note, folder membership, or
  file.
- **Apply is exact.** Activation accepts only the current reviewed token and
  the reviewed safe actions. Stale content requires **Check again**.
- **Attention is explicit.** Conflicts, deletion-like effects, partial work,
  offline roots, and recovery pressure are shown instead of resolved by a
  global winner policy.
- **Unsupported controls stay inert.** Conflict/deletion resolution, Retarget,
  and Disconnect remain visibly disabled in this release; they never imply a
  mutation path the runtime does not own.
- **Authority is private and fenced.** Absolute paths and observation tokens
  stay out of ordinary snapshots and diagnostics. A process lease prevents two
  Chatbook processes from owning the same root.
- **Server setup fails closed.** Server-backed folder sync remains disabled
  until the separately versioned server capability exists.

## Entry points

The Notes toolbar exposes **Add from files…**. Its first choice is made before
any source is read:

- **Import once** delegates to the existing reviewed import workflow and ends
  after its receipt.
- **Keep a folder synced** opens lasting-sync setup. The user chooses a folder,
  direction, and local destination, then checks and reviews the exact effects
  before **Activate reviewed root** becomes available.

When roots or migrated candidates exist, **Manage sync folders** opens their
status and contextual controls. Manual **Sync now** runs the same complete
reconciliation path as startup. Pause, resume, migration review, and recovery
are root-scoped operations. **Retarget** and **Disconnect** are visibly disabled
in this release and do not invoke a runtime mutation.

## Runtime and storage owners

| Component | Responsibility |
|---|---|
| `tldw_chatbook/Notes/notes_sync_runtime.py` | App-owned startup barrier, root lifecycle, reviewed check/activation, control operations, watcher ownership, and bounded public snapshots. |
| `tldw_chatbook/Notes/notes_device_state_store.py` | Private root, binding, operation, action, receipt, setting, and recovery transactions. |
| `tldw_chatbook/Notes/notes_sync_reconciler.py` | Mutation-free comparison of observed files, notes, bindings, and baselines. |
| `tldw_chatbook/Notes/notes_sync_executor.py` | Durable ordered execution of the exact reviewed safe actions with recovery-aware settlement. |
| `tldw_chatbook/Notes/notes_sync_legacy.py` | The sole compatibility reader for already-present legacy evidence; it creates paused review candidates only. |
| `tldw_chatbook/UI/Library_Modules/library_notes_sync_controller.py` | Converts typed canvas messages into runtime calls and republishes immutable UI projections. |

The private sync store owns device-local paths, baselines, operation state, and
recovery data. ChaChaNotes owns note content, logical folders, and memberships.
A successful new-root activation creates the managed Notes folder first and
persists the returned folder identity; it never invents or parses an opaque
root ID as a folder ID.

## Restart-only legacy cutover

Cutover happens during a normal application restart, never by hot-swapping two
writers:

1. The release contains no legacy engine constructor, timer, handler, or config
   writer.
2. After the private store initializes, the runtime reads the cutover marker.
3. An unknown or future marker fails closed without invoking migration or
   changing migration tables.
4. With no marker, legacy evidence is read into paused candidates. Only after a
   successful migration is the exact `notes-sync-cutover-v1` marker written.
5. The lasting runtime admits work only when that marker is present and this is
   the sole process for the profile.

Close older Chatbook versions before activating folder sync. If another profile
process is open, activation remains disabled with: **Close the other Chatbook
process and restart before activating folder sync**.

Migrated candidates do not inherit a legacy conflict winner or automatic-sync
choice. They remain paused until **Review migration** produces a current dry
run and the user explicitly activates it. Candidate bindings become active in
the same private-store transaction as the root.

## Compatibility data

Legacy config keys, note columns, `sync_sessions`, and `sync_conflicts` remain
read-only migration/history inputs. New installs do not emit legacy sync
defaults, and production code outside `notes_sync_legacy.py` neither reads nor
writes those inputs. Do not edit the private store or ChaChaNotes with direct
SQL; use the typed store, runtime, and Notes scope service transactions.

## Recovery and diagnostics

Operations are journaled before execution. Completed actions are idempotent;
partial or interrupted work remains visible with the nearest valid recovery
action. A root is not reported **Up to date** until its reviewed actions have
durably completed and its bindings/memberships have settled.

Public snapshots expose bounded counts, status codes, opaque IDs, and safe next
actions—not absolute paths, source content, observation tokens, or raw exception
text. Diagnostics use the same bounded runtime projection.

## Related documentation

- [Library Notes user guide](../User_Guide/library/notes.md)
- [Folder Files user guide](../User_Guide/library/file-notes.md)
- [Notes folder import and lasting-sync design](../superpowers/specs/2026-08-12-notes-folder-import-sync-design.md)
- [Library Notes reviewed-sync surface design](../superpowers/specs/2026-08-19-library-notes-files-reviewed-sync-redesign-design.md)
