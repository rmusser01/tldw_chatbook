# Local backup and recovery

Chatbook's **Backup & Restore** screen opens from **F9 → Settings → Backup & Restore**, or **Ctrl+P → backup**. A backup is a local
recovery archive, separate from a Chatbook content export. It covers selected
Chatbook-owned data; it does not back up a remote tldw server, the operating
system, or the application installer.

Complete backup and replacement have separate availability checks. They require
a matching qualified platform, the Python encryption backend, and supported storage
locations. The screen shows availability alongside coverage and required space. Follow the
operation's actual review and refusal messages; platform support alone does not
establish that a particular selection can be captured or restored.

## Create a backup

1. Open **Backup & Restore** and choose **Create backup**. Review the listed profile configurations;
   add another configuration explicitly if it is not listed. The default selection
   is **Everything**, which includes known owned profile data. To save specific
   modules, select **Choose groups** and check the groups you want.
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

Groups contain whole data stores. Conversations, notes, and personas share a
database and are backed up together. Required linked groups appear in the review.
**Complete selected coverage** means the chosen groups and their required data
are included; the archive does not include the deliberately unselected groups.
Configuration needed to interpret selected data is included as support, even
when Settings is unselected. Changing the selection requires a new review.

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

Restore defaults to every available group in the archive. Choose groups to
restore a subset, including from an older full backup. Review lists any linked
groups that must accompany the selection. Destination fields follow the selected
groups.

For **Restore as isolated profile**, choose one new local folder and name for each
profile, plus a destination for each included external folder. Chatbook derives
the configuration and database paths, keeping shared databases together.
Place new profile folders beneath an existing private restore directory that is
separate from Chatbook's recovery storage. Review identifies unsuitable parents
before you confirm.
Choose **Review restore**, read the affected data
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
extraction** with selected groups and a new directory. Group labels show file
counts and sizes; choose a group containing files to recover saved content.
Extracted files are inert
bytes: this does not register, migrate, or open a working profile.

## Replace existing stored data

Choose **Replace selected stored data** and identify the existing local profile
configuration. From the normal application, **Continue in recovery mode** closes
Chatbook before replacement review. Enter passwords and inspect the archive again
in the fresh recovery screen; approval does not carry across the restart.

Selected groups replace their current data; individual records are not merged.
Unselected groups are preserved. If **Settings** is unselected, existing local
settings determine the destinations and remain unchanged. If Settings is selected,
restore also checks that unselected data remains reachable at its current paths.

An empty selected data source replaces its current contents with that empty state;
the safety copy retains the removed data. If source and target locations cannot
be matched safely, replacement refuses. Restore into a separate profile to recover
that selection without guessing which existing files to remove.

When **Files needing setup** is shown, choose an existing private folder outside
profile and recovery storage. Chatbook places those inactive recovered files there
without mixing them into the running profile's configuration.

Review the exact restore, retirement, and preservation lists. Preserved items can
be selected explicitly for inclusion in the safety copy where the review offers
them. If review requires related safety-copy files, choose **Select required
safety-copy files**, then **Review restore** again. Other preserved additions are
optional. These selections preserve their current bytes for recovery; they do not
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
The `restore` command accepts repeatable `--group` arguments, such as
`--group prompts --group library`; omitting them restores all available groups.
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

The implementation checks native protocol 2 capabilities on writable local APFS
on macOS, ext4 on Linux, and NTFS on Windows. Private ownership, containment,
exclusive publication, locking and persistence checks still apply to the actual
storage used by each operation. Missing capabilities produce a refusal.

The full installed backup, restore, replacement and retained-copy rollback flows
have passed on macOS, Linux and Windows. Native TTS reference backup, isolated
restore and fresh readback are also verified on all three platforms.
See the [cross-platform verification record](Development/backup-cross-platform-verification-2026-09-12.md)
for exact revisions, environments and results. Other filesystems and application
upgrade pairs have not been qualified.

Source, editable, wheel and source-distribution installs include the Python
encryption worker, which uses the existing PyCryptodome dependency. No Go
installation or separate encryption executable is required.
The evidence does not establish physical power-loss or device-detachment
recovery; retain recovery copies and operation records when resolving an interruption.

The controlling contracts are the [approved design](superpowers/specs/2026-09-07-complete-local-backup-restore-design.md),
the [Python encryption correction](superpowers/specs/2026-09-12-python-backup-encryption-design.md),
and [release qualification plan](superpowers/plans/2026-09-07-backup-recovery-06-release-evidence.md).
The [cross-platform correction](superpowers/specs/2026-09-12-cross-platform-backup-correction.md)
updates the platform operations without changing those recovery workflows.

## Preview legacy mobile history

To inspect legacy mobile history before consolidation, run:

```bash
python -m tldw_chatbook.Backup_Recovery consolidate-mobile \
  --source-config /absolute/source/config.toml \
  --target-config /absolute/target/config.toml --preview
```

The preview lists the physical source, explicit canonical destination, counts,
record identities, dependencies and backup scope. It does not import history or
create missing databases. The source is the selected user's `mobile` directory;
canonical database overrides do not relocate that legacy source. If several known
configurations share it, the preview requires a destination review instead of
assigning history to the launching profile. Matching titles do not establish that
records are equivalent.

Normal backup includes recognized legacy chat, workspace and run stores plus
qualified referenced artifacts, using the existing coherent capture and
credential policy. Unknown contents, unsupported schemas, missing dependencies or
unsafe aliases must be resolved before capture; partial capture cannot silently
omit blocked legacy history. Physical dictionary/persona artifact dependencies
without a qualified legacy binding remain blocked. Historical note locator fields
are a distinct case: native mobile capture can retain them as database content.
That does not capture the external note file or grant active file-sync authority;
import and App-owned candidate visibility still require qualification. Source runs and claims are
never reconciled during preview or capture. The local public service and command
also implement explicit apply and same-operation recovery; full Task8 integration
qualification is still in progress.

The native import implementation also refuses two specific trace histories before
any import effect: a populated detached trace whose deleted original conversation
cannot be lawfully reconstructed, and calls recorded as `post_dispatch_promoted`
through the current Voice authorization path. An empty detached trace is not
blocked merely because it is detached. With the exact original conversation proved
and retained, preflight qualifies an attached/replay/one-way-detach schedule in one
Chat transaction. Each historical owner releases the unique conversation slot
before a later owner attaches; final rows and detach timestamps remain exact.
These restrictions preserve native parentage
and call provenance; they do not discard those records or import the rest of a
blocked profile. Keep the original stores and authenticated recovery captures with
the exact owner/call diagnostic. Do not edit provenance flags, create replacement
parents, remove history or use an older binary to force an import. The public
apply command preserves these same preflight refusals.


Imported unfinished response recovery remains visible but cannot retry or resume
old work. When its committed provenance and current checkpoint are valid, the
existing explicit Discard action retains the user message and permits a later
fresh Send. Invalid provenance leaves recovery blocked without an action. Imported
queued work and fleet messages remain history and do not restart a queue or consume
fresh delivery capacity. Discard does not release imported execution permissions.

Interrupted first publication of the canonical history directory may resume only
from its original authenticated capture and recorded native directory identities.
Changed or additional files remain retained and block qualification. Do not remove
pending recovery records or modify source/provenance to force progress. Public
cutover and ordinary access to the produced target require durable native
completion, legitimate pending retirement and physical owner closure. Public
apply and explicit recovery compose this workflow.


The explicit native `Remove payloads` action can erase local private Goal bodies
when the complete committed receipt and unchanged current allowance family prove
that unsettled facts are imported history. Missing or changed accounting, current
work, incomplete receipts, or missing/reviewed holds still refuse. Historical
uncertainty and all execution holds remain; this does not settle remote work or
release old authority. Previously erased body rows are never restored from
receipts. This consumer proof does not replace ordinary access to a produced
profile after the pending operation is legitimately completed.


An imported active provider continuation without a real dispatch checkpoint stays
paused as read-only history. There is no supported local recovery action for that
retained owner in this increment. The explicit Discard path for a validated actual
imported dispatch checkpoint does not apply to a checkpoint-free continuation;
no checkpoint, branch, current permission or release is synthesized.

Failed change-tracking snapshots retain their recorded error, counters, tips and
original project locator; their native review path displays the failure without
opening Git. Other snapshot rows require actual installed physical Git proof,
including proof of a genuinely pruned/unavailable state. This mobile operation
currently has no admitted shadow-Git reader/publisher or current selected writable
worktree authority. Otherwise valid profiles containing those unresolved physical
dependencies therefore block the whole consolidation before publication or owner
writes. A path, hash or recorded SHA is not physical authority; original history,
files and statuses are retained, and no failure/pruned status is manufactured.


### Local mobile apply and recovery

Close all application instances using the selected source or target before apply.
Use the exact profile reviewed by preview. Applying obtains and shows a fresh
preview, asks for `consolidate`, then prompts for a new encrypted recovery-copy
password; keep that password for an interrupted operation.

```bash
python -m tldw_chatbook.Backup_Recovery consolidate-mobile \
  --source-config /absolute/source/config.toml \
  --target-config /absolute/target/config.toml --apply
python -m tldw_chatbook.Backup_Recovery recover OPERATION_ID
python -m tldw_chatbook.Backup_Recovery recover OPERATION_ID --finish --ask-password
```

Use the same `--control-root` before the subcommand if an explicit control root
was selected. The status command lists actions actually available for that
operation. A pre-cutover rollback uses `recover OPERATION_ID --rollback
--ask-password` only when offered. It restores the authenticated coherent canonical
pre-import snapshot/state, retains displaced current objects and leaves legacy
source bytes untouched. It does not promise the original uncaptured WAL bytes or
inodes. After committed cutover, restoring a retained canonical copy instead uses
a fresh native replacement review and Safety Copy.

Verified empty or absent legacy source does not require creating a legacy database.
Required canonical owners must already be capturable: initialize the canonical
application explicitly with its selected `--config` and valid private storage
parents, retire it, then obtain a new preview. Missing required canonical owners
refuse before preparation. Never initialize them silently from a stale preview.

A public start or recovery currently has one fixed 4 GiB cumulative materialization
allowance coupled to the original native 16 MiB proof allowance. These are bounded
reservation counters, not heap measurements or a promise that every profile fits.
A late refusal can retain pending recovery after partial native commits; use the
explicit available same-operation recovery action with the original password.
There is no automatic hidden retry or allowance reset within one call. Configured
but unreadable or required missing credentials remain blocking review issues.

Current targeted evidence includes fresh populated cutover, no-data completion,
fresh-process continuation after a late bounded refusal, and ordinary RunLog,
command history, trace and Goal privacy APIs on the same produced targets. Native
Console hydration/controller Discard also preserves the mapped user and execution
holds without provider dispatch. The absent-marker Notes/Personas flow also passes:
a real App rename survives public import and two cold App starts, the default
assistant remains ID 1, and imported linked-note metadata becomes a paused
candidate with pending migration review. Source databases and external files
remain unchanged; this grants no active file-sync authority. The present-marker
flow also passes with receipt-backed disjoint candidates, preserved existing
Notes state, two stable cold starts and all observed Notes handles closed. The
mounted Console manual turn separately passes with one scripted generation,
zero tool dispatches and unchanged imported holds/accounting through reopen.
These are startup and scripted-provider proofs, not active file sync or hardware
qualification. The explicitly approved direct post-replacement observer and
4 GiB/early terminal-pricing changes are implemented. Their finite component
checks are recorded separately. Fresh restored-generation checks, native data and
theme saves, refusal controls, and rollback-to-reviewed-successor integration now
pass. The same public Workspace operation also finishes in a new process after a
durable-commit interruption, with exact receipts/rows/holds and no replay. Native packaged-artwork copy/hash/member controls and their
composition with existing native source namespaces have explicit human approval
and preserve profile bindings. Full included-artwork recovery now passes actual
native staging, SafetyCopy capture and authentication, publication and restoration.
The verified SafetyCopy retains the later edit while the restored target returns
to its captured contents; final source/package and guard checks pass. The ordinary
mobile gate now accepts that authenticated direct restoration and preserves its
source, activation, physical, schema and retained-capture checks. Earlier
parse and scratch-admission failures remain historical; the human-approved native
scratch composition also passes its scoped mounted manual flow.
Do not treat a generic replacement success as mobile write-gate acceptance.

### Native-owner qualification environment

The saved native-owner integration tests emitted a Pydantic warning from
`AudioCppAcceptedPackage.managed_artifact` in `TTS/audio_cpp_guided_config.py`:
`Field(exclude_if=...)` was treated as an extra keyword by the recorded Pydantic
2.11 environment (`pydantic/fields.py:1093`). This declaration predates the owner
fix and belongs to the audio configuration serialization contract. Its compatible
correction remains tracked there; changing or suppressing it is outside the
native-owner fix. The already installed Pydantic 2.12.5 environment used for FIX1
has an explicit `exclude_if` parameter. Those newer results do not erase the older
warning or establish compatibility with older Pydantic versions. No dependency
was installed or updated for FIX1.


The mounted Console qualification now completes a new manual turn on the actual public-produced target and reopens its persisted response after native shutdown. The human-approved R135 composition retains the native scratch manager/writer and independent worker leases while preserving profile admission, imported holds and tool permissions. The test recorded one scripted provider generation, no tool dispatches and unchanged imported authority/accounting. It uses the native character-estimate tokenizer fallback; real providers, hardware and main-entrypoint startup remain unqualified. R129/R130 received explicit authorization; their current restored-profile and public recovery checks pass.


Notes device-state lock files are local control metadata. R136 adds an exact-shape exclusion for the native private `notes_sync_locks` directory; malformed or foreign contents block Complete discovery. This does not activate imported sync roots or reset migration markers. Native absent, empty and acquired/released retained-lock Complete checks and scoped refusal/accounting controls pass. The present-marker App flow now passes with the separate R137 receipt-backed candidate path; lock exclusion itself grants no candidate or sync authority.


R134 held-preview accounting clarification: the fixed authenticated mobile canonical-copy path carries one original ordinary16MiB allowance explicitly through its new source-prefix, selector/mapping, bootstrap/journal and framed-observation representations, and its already budgeted package read/copy/view/slice work. The actual native `_capture_bound_sources` context remains the sole source-read authority; generic capture admission is unchanged. No PreviewScope may be issued during maintenance and no generic role is added; R141 adds only a private byte workspace, not authority or an allowance. Nested fixed callbacks reuse the exact allowance without reset. Original generic verify_sealed/plaintext and encrypted hash passes, Zip/config/native planner internals retain their existing ArchiveLimits and behavior; this counter is not a whole-engine or RSS bound. Factor the existing authenticated candidate prefix and preserve positive current nonpackage profile bindings, complete target/member/selector/store checks, cancellation and physical retirement. Optional private forwarding defaults preserve existing callers; component identity/refusal/cancel/exhaustion/default checks precede another full replacement. The separately authorized R129/R130 changes are now qualified by their targeted public integrations.


The R137 Notes startup extension is implemented and qualified by a fresh present-marker App flow: with an existing current cutover marker, it inspects validated canonical import receipts for unrepresented linked notes and offers only new disjoint paused candidates. Two cold starts preserve the existing marker/root/binding/review baseline and mapped note ID/version/digest; all seven observed Notes handles close each time, including the callback store. Overlapping or unreadable file links remain unadded with a separate visible warning; the note stays available. Existing sync folders and their authority remain unchanged, and activation still needs the native fresh review. The earlier PRESENT run imported note content successfully but did not expose its candidate; that failure remains recorded.


Current artwork-recovery qualification includes the full native replacement and
verified SafetyCopy path. Ordinary and held preview, native staging and all three
fresh source checks complete under the original limits. Parent handles remain
held through current-name checks, and distinct write paths are checked without
collapsing logical owner mappings. The final full run restores the captured chat
and retains the newer note in its authenticated SafetyCopy; final source and
package facts remain unchanged and both guards are clear. Target data changes are
the expected restoration, not a claim of full-run physical nonmutation.

The historical qualification above returned native `mobile_object_changed` at
the ordinary mobile gate. The approved direct-restored-generation observer is now
implemented and passes its focused integration qualification. It requires native
replacement provenance and current schema/physical checks; it does not recreate
imported receipts or release execution holds. The public allowance is now 4 GiB;
the complete ordinary cutover coordinator uses a fixed 64 MiB allowance, while
generic trace/Notes and private native defaults remain 16 MiB. No hidden retry, universal-fit
or whole-engine memory claim is made. Earlier failed epochs and the staging-only sentinel remain
separate evidence.


The ordinary mobile cutover check uses a separate fixed64MiB cumulative coordinator allowance for its complete source, native restoration, schema and retained-evidence observation (R146), replacing its former whole-call16MiB allowance. It grants no ownership or tool access and cannot renew its allowance during a call. Generic trace/Notes and private native migration defaults remain16MiB; public migration remains4GiB with its original native16MiB child. These counters are not RSS measurements or guarantees that every profile fits. Fresh restored-profile, native edit/theme-save and included-artwork recovery checks pass under this allowance. The public Workspace crash/reconnect proof passes separately; the retained private 16 MiB cold-case refusal is not claimed as fixed.

### Served qualification after cutover

The serving parent and native app child must select the same canonical profile,
including its configured data directory and legacy mobile source. The parent
now derives these paths from one authoritative config snapshot. A fresh NoData
profile has completed an actual terminal prompt and phone continuation without
pre-creating an optional tokenizer directory.

Cold phone Create followed by immediate Send now uses native read-only
completion transactions, so concurrent app writes do not require a stable byte
copy of the live database. Preview and maintenance keep their stricter copy
checks. This does not repeat a completed migration or grant new authority.
The exact current evidence and
remaining release checks are in the [Task11 qualification packet](superpowers/qa/2026-10-25-shared-console-mobile-supervision/README.md).
