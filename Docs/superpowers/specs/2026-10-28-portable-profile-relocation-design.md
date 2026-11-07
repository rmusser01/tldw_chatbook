# Portable profile relocation after the original is unavailable

Date: 2026-10-28
Task: TASK-37205
Status: Approved by the user, 2026-10-28

ADR required: yes
ADR path: backlog/decisions/256-detached-portable-profile-import.md
Reason: a new recovery input boundary, independent destination authority and
preservation of copied publication evidence.

## Intent and agreed constraints

The user wants to move an already used Windows portable Chatbook folder first,
then launch it without access to the original location. A source-side Prepare
to move step is not acceptable. Recovery checks must remain effective. Terminal
IPC/runtime files must stay in Data; ADR-255 covers that independent change.

The copied profile is a detached input. Native ancestry receipts bind physical
identities and absolute paths and cannot become valid by changing their text.
Copied payloads, configuration and approval history also cannot establish fresh
destination execution permission. The original copied bytes must be retained.

## Approaches

1. A reviewed, destination-only detached import into a fresh isolated generation
   is the proposed approach. It uses installed schema/owner validators and the
   existing journaled isolated-restore and activation machinery.
2. Automatic clean-shutdown backup checkpoints could eventually carry a stronger
   coherent-transfer artifact. They do not solve existing copies that have no
   checkpoint and therefore are not a prerequisite for this work.
3. Rewriting or removing copied ancestry, namespace, journal or activation records
   is rejected: it would conceal the state that made startup refuse.

The user approved destination-only support and the reviewed import and credential
exclusion behavior below.

## User-visible flow

1. At startup the launcher notices a location change or copied recovery ancestry.
   It does not import config or open the copied profile as an ordinary live app.
2. A local recovery process presents a relocation review: copied content included,
   unsupported or unresolved content, pending-operation findings, required storage,
   credential consequences, and the new destination. No original path is opened.
3. Explicit confirmation creates a new isolated generation inside Data. The copied
   source remains unchanged and retained. Closing/cancelling the review leaves it
   untouched and does not select the installed Chatbook profile as a fallback.
4. Existing isolated restore validates, journals and publishes the new profile.
   Its fresh installation/profile IDs and activation requirements are produced by
   the installed recovery executor. The importer never approves restored owners.
5. The launcher starts the validated profile in a fresh process. It shows the
   existing restored-content review/setup requirements. Subsequent starts validate
   the committed profile descriptor rather than trusting a JSON pointer alone.

After another move the same destination-only import can be repeated. Previously
retained source generations remain inert; they are not recursively imported or
silently discarded. Their cleanup is an explicit user operation outside this work.

## Source boundary and acquisition

Introduce a distinct detached-profile importer owned by Backup_Recovery, called
from the portable launcher. It is not a live capture operation. It must not call
adapter.capture, copy_capture_file, copy_private_sqlite, or manufacture a
MaintenanceSession around an old profile.

Select a current local NTFS source directory explicitly. Use pinned no-follow
handles and bounded member enumeration. Refuse reparse points, symlinks, hard
links, device/non-regular members, unsafe ownership/ACL posture, escapes,
case/Unicode collisions, resource limits and ambiguous aliases. Bind review and
acquisition to the complete member list, file identities, sizes and digests.
Acquire private staged copies and recheck all source observations before sealing.
Source files and sidecars are never opened for writing.

The old location marker and recorded paths are untrusted syntax. They may map an
already validated lexical relative suffix beneath the selected copied root; they
grant no permission and are never resolved, stat-ed or opened at the old location.
Arbitrary path-like strings in user content are not rewritten.

## Installed ownership and validation

Successful coverage is the complete base portable profile: every installed
recovery owner represented by durable content inside copied Data, including
Notes/conversations, settings, base SQLite stores, supported owned definitions,
history and stored assets. Importing only the core Notes database while silently
leaving other owned durable content behind is not a successful relocation.
Optional owners absent from the bundle, unknown durable files, and unsupported
custom layouts refuse with an explicit report; they are never silently dropped.
External workspaces are outside the copied Data payload: retain their locator
metadata as inactive/unresolved references requiring review, without opening old
targets or automatically granting the new profile access to them.

Use an explicit installed detached-layout resolver/census. Ordinary
inventory.discover and _preview_reads are not the confinement boundary: discovery
can enumerate DB/config-referenced external paths, while _PreviewScope intentionally
has no root restriction. Each supported owner needs a declaration that resolves
only to acquired staged members, with its installed dependency/relocation policy.

Validate staged SQLite through validated_schema_version and restricted
validate_candidate with migration disabled during preview. Use installed raw and
semantic validators for other content, plus owner-specific dependency/reference
validation. Unsupported owners, unknown durable files, schema mismatches or required
assets outside the copied source block publication; never substitute empty defaults
or silently omit them. The review names those blockers without displaying secrets.

Installed immutable built-in assets are an explicit support exception: the
bundled installed asset owner may stage its exact built-in bytes only when the
imported identifiers, hashes and relative references satisfy its installed
dependency policy. These are shipped application assets, not old user locators.
Unknown/mismatched references and custom asset substitutions still refuse. The
partial archive report identifies this installed support source.

Default SQLite stores absent because of lazy initialization may be declared
unused only with no explicit custom selector or residual main/sidecar/lock.
The review explicitly lists those absences; no empty payload is synthesized.
Installed validators on included stores must still refuse every required missing
reference/dependency. This detached partial-input policy does not alter live
capture's required-core declarations or claim historical completeness.

Process SQLite main/WAL/SHM as one acquired set. The immutable validation connection
must not be pointed directly at a main file while ignoring its WAL. A bounded
normalizer may replay only private staged copies and produce a standalone image;
then run the restricted installed validator on that image. Loading extensions and
arbitrary SQL are forbidden. Unsupported or ambiguous rollback/sidecar states
refuse. Normalization never checkpoints or repairs the retained copied source.

## Copied recovery evidence

Add a bounded pathless parser for copied control and operation evidence. Factor
existing strict event/schema/hash-chain parsers where needed. Do not instantiate
Journal on copied evidence: its lock path is writable and its constructor can
create storage. Do not call bootstrap._control_records to bless copied ancestry.

Reject pending-*.json, activation-update intents, registry.pending.json, unsettled
creation intents, registry pending/proposed state, malformed/duplicate-key records,
corrupt event chains and unknown versions. Every copied publication journal must
have a recognized terminal sequence: committed, fully rolled back, or aborted
before publication. Missing/ambiguous evidence needed to decide completion refuses.
Recorded selectors/control roots/candidates/targets remain inert text.

Keep copied recovery records and old approvals as retained evidence, excluded
from restored payloads. Never use a record label alone to waive native admission.

## Archive and fresh destination authority

Create a distinct detached-import result with a strict ArchiveManifest marked
consistency="partial". A raw copied directory cannot prove coherent multi-store
capture, even if every file validates. Do not relabel it as a coherent backup.

Use existing archive packaging and archive_reader acquisition/verification, then
resolve_destinations(mode="isolated") and restore_isolated. Partial isolated
restore is an existing contract; partial replace remains forbidden. Existing
stage_restore performs bounded migrations, reference relocation, dependency
validation and installed-copy validation. Existing journal/publication/catalog
durability must finish before the launcher selects the new profile.

The new recovery host and profile live in fresh generation directories inside
Data, disjoint from the retained source and staging. An active-profile locator is
only a hint: select_profile/_launch_state must validate the actual committed
association, catalog, installation ID and activation generation on each start.
The outer relocation transaction records acquisition/review/publication/selection
so crashes either remain on retained inert source or resume the same existing
restore operation. It must never start an uncommitted destination or automatically
retry publication under a new operation ID after an ambiguous result.

## Credentials and executable state

For the first detached importer, use the existing credential exclusion policy on
staged copies and refuse every returned format/coverage issue. Managed provider
credentials need setup again in the newly active profile. No original keyring,
include/rollback credential capture, inherited API environment or plaintext
fallback is consulted. The retained copied source still preserves the original
bytes, including encrypted content. Arbitrary notes/media are not claimed free of
secrets. Supporting authenticated credential inclusion is a separate reviewed
extension, not an implicit weakening of the existing archive policy.

Fresh isolated restore requires review for all installed activation-required
owners. Copied tools/MCP/hooks/schedulers/approvals stay historical data and cannot
activate themselves. The importer must never call ActivationStore.approve.

## Verification and limits

Test real Notes/conversations/config and representative owned assets; move/copy
the folder, make the original unavailable, review/import, launch and reopen again.
Verify retained source digests and that old paths were never accessed. Use spaces
and Unicode in both locations. Show that malformed schemas, changed source bytes,
pending publications, corrupt chains, reparse/hardlink escapes, unsupported owners,
WAL data, credential formats and activation history fail or follow the exact
specified normalization/review behavior. Exercise interruption before/after each
publication/selection boundary and two concurrent relocation attempts.

Re-run targeted existing archive/planner/isolated-restore/activation/admission
tests, plus packaged real-terminal journeys and ADR-255 host-path containment.
No full test sweep without the user's opt-in.

Initial scope is local NTFS under the same Windows account. FAT/exFAT, network
shares, cross-account ACL repair, original historical authenticity, guaranteed
original cross-store coherence and automatic external-workspace relocation are
not claimed. Refusals preserve the copied bytes and explain the specific issue.
