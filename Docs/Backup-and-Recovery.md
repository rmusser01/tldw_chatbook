# Local backup and recovery

Chatbook's **Backup & Restore** screen opens with F9. A backup is a local
recovery archive, separate from a Chatbook content export. It covers selected
Chatbook-owned data; it does not back up a remote tldw server, the operating
system, or the application installer.

Complete backup and replacement have separate availability checks. They require
a matching qualified platform, the Python encryption backend, and supported storage
locations. F9 shows availability alongside coverage and required space. Follow the
operation's actual review and refusal messages; platform support alone does not
establish that a particular selection can be captured or restored.

## Create a backup

1. Open F9 and choose **Create backup**. Review the listed profile configurations;
   add another configuration explicitly if it is not listed. The default selection
   includes known owned profile data.
2. Choose a new output file. Use `.tldw-backup.zip` for a plaintext archive or
   `.tldw-backup.zip.age` for an encrypted archive.
3. Select optional external folders, model IDs, temporary media, or diagnostic
   history if wanted. These are explicit additions. Arbitrary documents, external
   files, and diagnostic history can contain secrets that managed-credential
   filtering does not recognize.
4. Choose encryption and enter the password twice. **Include supported
   credentials** requires encryption. Credentials are excluded by default;
   unavailable or unsupported credentials are shown for review.
5. Choose **Review**, inspect coverage and exclusions, then confirm the reviewed
   backup. Acknowledging partial coverage does not make a partial archive complete.

Selected external files have per-file stability checks. Their presence makes the
archive **Partial**, even when every selected file is included. Review and explicitly
acknowledge Partial creation. Partial archives support validated isolated recovery
and file extraction; they cannot replace existing installation data.

Keep the password separately. Chatbook does not retain it, and a lost password
cannot be recovered. Encryption does not eliminate plaintext working files:
capture, decryption, validation, and restore use private local staging with
best-effort cleanup. Allow space for staging, extracted data, and retained recovery
copies in addition to the output archive.

Supported credentials means readable values handled by Chatbook's credential
owners. Environment variable names are setup hints, not exports of the process
environment. An unreadable encrypted value is not a usable credential backup.
Inspect the reported omissions and re-enter missing credentials after recovery.

## Inspect and restore separately

Choose **Inspect / restore**, select the archive, supply its password when needed,
and choose **Inspect**. Inspection verifies the archive before destination review.
Archive paths never authorize writes to those locations on your computer.

For **Restore as isolated profile**, choose new local destination roots and profile
names using the displayed slots. Choose **Review restore**, read the affected data
and setup requirements, then **Confirm reviewed restore**. After successful
validation and publication, use **Restored profiles** to open the profile in a
separate process. Opening it does not switch the running application's storage.

For custom tokenizers, the default tokenizer directory may already exist only if
it is empty, owned by you, and private. An isolated restore refuses an occupied
tokenizer destination.

An isolated restore retains included credentials in an encrypted recovery copy;
they do not need a destination slot. The restored configuration is sanitized, and
credentials still require the applicable local setup before use.

These results mean different things:

| Result | Meaning |
| --- | --- |
| Verified archive | The archive passed inspection; no installation is implied. |
| Validated restore | The restored installation passed the applicable checks. |
| Opened profile | The separate application launch completed its local read checks. |
| Needs setup | Restored capabilities or missing assets still need local review. |

For manual file recovery, use **Review extraction** and **Confirm inert
extraction** with selected groups and a new directory. Extracted files are inert
bytes: this does not register, migrate, or open a working profile.

## Replace existing stored data

Choose **Replace selected stored data** and identify the existing local profile
configuration. From the normal application, **Continue in recovery mode** closes
Chatbook before replacement review. Enter passwords and inspect the archive again
in the fresh recovery screen; approval does not carry across the restart.

Review the exact restore, retirement, and preservation lists. Preserved items can
be selected explicitly for inclusion in the safety copy where the review offers
them. Such a selection preserves their current bytes for recovery; it does not
add them to the replacement payload.

Supply and confirm a password for the encrypted rollback copy, even if the incoming
archive is plaintext. Replacement first requires a verified recovery copy of the
affected existing data. It must refuse if that copy or the reviewed target cannot
be safely established.

If an attempt reports credential omissions, review each displayed issue. An
untouched prepared replacement may need **Recovery copies → Abort untouched
replacement** before returning to inspection, choosing the explicit omission
acknowledgements, and creating a fresh reviewed plan. Do not treat an omission as
covered by the credential-recovery guarantee.

Target changes invalidate review. Re-inspect and review the actual current state
instead of trying to reuse a stale confirmation.

First-time replacement may refuse a configuration folder that contains recovery
controls or unrelated files. The verified custom-folder layout keeps profile data
under a private configuration folder, with recovery controls elsewhere. Its storage
binding includes future lock and temporary files created inside that folder.

## Interrupted recovery and retained copies

Use **Recovery copies** to inspect recorded operations and available recovery
actions. The permitted action depends on the operation's durable state. Abort is
available for an untouched replacement; once publication starts, recovery must
finish or roll back at a safe boundary. Keep the rollback password available.

Recovery copies remain until explicit deletion. Deleting a copy removes that
later-rollback option; the UI asks for acknowledgement and protects copies needed
by active recovery. There is no automatic retention cleanup in this version.

Later rollback replaces changes made since the selected copy while first
preserving those current changes in a new verified encrypted safety copy. Select
the copy and current profile configuration, enter the old copy's password, and
choose **Review later rollback**. Review any reported credential omissions. If
offered, use **Abort untouched replacement**, then explicitly acknowledge the
omissions and request a fresh review. Changing the selection or acknowledgements
invalidates the reviewed plan.

Confirm the reviewed consequences, re-enter the old password, and provide matching
passwords for the new safety copy before executing later rollback. A refused
review is not a completed rollback; retain the existing copy and operation records.

## Manage recovered media

Open **Restored profiles → Recovered media details** for the current profile.
The list shows each asset's catalog state, recorded size, references, and recovery
holds. Recorded sizes remain visible after deletion; they are not a measurement
of current disk usage.

Choose **Review asset details** to see every affected reference, including aliases
from earlier restores or other profiles. Confirm the review before choosing
**Delete reviewed asset**. Deletion removes the payload and retains its deletion
history. Recovery holds prevent deletion. **Clean up reviewed orphan** is available
only when the asset has no references or holds.

Changes to the selected profile, asset references, or recovery state require a new
review. Cancel dismisses an unaccepted review; leaving the screen after accepting
an action does not cancel the operation.

## Recovery when normal startup is unavailable

The installed command is `tldw-cli`. Its recovery commands run before normal
configuration loading and application services. For example:

```console
tldw-cli recovery inspect /absolute/path/archive.tldw-backup.zip
tldw-cli recovery inspect /absolute/path/archive.tldw-backup.zip.age --ask-password
tldw-cli recovery copies
tldw-cli recovery copies --inspect OPERATION_ID
tldw-cli recovery recover OPERATION_ID
tldw-cli recovery recover OPERATION_ID --rollback --ask-password
tldw-cli recovery profiles
tldw-cli recovery profiles --open PROFILE_ID
```

Use IDs returned by the commands. `recover` without an action inspects the recorded
operation; `--finish`, `--rollback`, or `--abort` requests its corresponding recovery
action. This is distinct from starting a later rollback of a completed replacement.
Add `--ask-password` to `--finish` or `--rollback` when the recorded recovery needs
its encrypted safety archive; without that flag no recovery password is requested.
Abort does not require that password flag.
Use `tldw-cli recovery restore --help` or `extract --help` for explicit destination
and selection arguments. Passwords are entered at prompts, never as command-line
arguments. If a nondefault control root was used, supply its actual path with
`tldw-cli recovery --control-root /absolute/path ...`.

## Setup after restore and compatibility

Open **Restored profiles** to see the **Current profile** setup summary. It follows
the current recovery generation, including replacement and later rollback, separately
from the list of isolated profiles. Missing or damaged local evidence is reported as
unavailable; it is not treated as completed setup. Historical recovery copies retain
their own operation status.

Restored data can be inspected locally while execution and reconnection remain
inactive. Opening a profile or restarting does not approve provider connections,
sync, schedules, MCP, skill scripts, model processes, downloads, or automatic
refresh. Use the owning feature's explicit recovery review when available; one
feature's approval does not approve another. Unavailable review controls leave the
corresponding capability inactive.

Re-enter missing credentials, review external-folder bindings against their current
local contents, and explicitly rebuild derived indexes when required. Missing model
files are setup requirements; restore does not automatically download them. Historic
queued work and old permissions do not authorize replay.

Older chat-context sidecars retain their message relationships and historical
citation metadata. They keep the app's existing legacy provenance status.

Saved TTS reference audio is retained with its profile and recipe metadata. Loose
voice catalogs can still contain historical audio paths; their copied WAV files
need explicit setup before synthesis can use them.

Saved generated images return to the restored profile's image storage. Their files
can be read locally; generating new images still requires the applicable setup.

Saved RAG profiles can relocate known absolute index-directory settings when the
corresponding captured projection root is selected for restore. Missing or mismatched
roots are refused. Retained profiles, tracking records and index files still require
owner review and index validation before retrieval can use them.

For an existing local HF embedding model, use **Settings → Library/RAG** with the
active RAG profile selected and any draft changes saved or discarded:

1. Choose **Review local model**. Check the displayed model directory, file count,
   and recovery generations, then choose **Approve local model**. This records
   only the model review; it does not load or download a model or rebuild an index.
2. Choose **Review recovery**, inspect the RAG owners, sources, and prerequisites,
   then choose **Approve RAG owners**. Model permission and RAG permission are
   separate; neither approval completes index validation.
3. When the required local sources and model are available, choose
   **Reconcile / rebuild** and confirm that action. This explicitly rebuilds and
   verifies the index before recovered semantic retrieval can become available.

Changing the model or active settings invalidates the displayed review. Request a
fresh review rather than reusing an earlier confirmation.

The reviewed local-HF rebuild and retrieval route is qualified within the running
process. Reusing its saved projection after a fresh application launch remains
unavailable: a valid model-review receipt alone does not verify the saved vectors.
The app must retain that refusal until the projection prerequisites are met through
an explicit qualified reconciliation or rebuild. Restarting does not automatically
rebuild or approve the index; local source-content inspection remains available.

Archive format, owner schema, encryption, and native filesystem capabilities are checked
independently. A newer archive is not automatically compatible with an older app.
Unsupported formats may permit selected inert extraction without permitting an
installation restore. Keep a compatible application release with important archives;
do not use protocol-unaware older launchers against an interrupted recovery.

The currently enabled platform is macOS Darwin 25.5.0, arm64, Python 3.12.11, with
native protocol 2 and the qualified APFS configurations recorded in the
[release evidence](../backlog/docs/backup-recovery-release-evidence.md). Complete
capture uses the host configuration (filesystem flags 76583040). Replacement also
supports publication on the tested APFS image configuration (flags 76583448), while
its recovery authority remains on qualified host storage. The image is not qualified
for Complete capture or cooperative storage admission.

Linux, Windows, other platform/filesystem combinations, and upgrade pairs remain
unqualified. Source, editable, wheel and source-distribution installs include the
Python encryption worker, which uses the existing PyCryptodome dependency. No Go
installation or separate encryption executable is required. Filesystem qualification
is still required for backup and replacement; missing capabilities produce a refusal.
The evidence does not establish physical power-loss or device-detachment
recovery; retain recovery copies and operation records when resolving an interruption.

The controlling contracts are the [approved design](superpowers/specs/2026-09-07-complete-local-backup-restore-design.md),
the [Python encryption correction](superpowers/specs/2026-09-12-python-backup-encryption-design.md),
and [release qualification plan](superpowers/plans/2026-09-07-backup-recovery-06-release-evidence.md).
