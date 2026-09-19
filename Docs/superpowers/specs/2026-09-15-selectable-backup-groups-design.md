# Selectable backup and recovery data groups

Task: TASK-32628. Amends ADR-126. User authorization: migrate the existing
backup system to selectable data groups; replace selected groups and preserve
unselected groups. This change remains Python on macOS, Linux, and Windows.

## User contract

Everything remains the default. Choose groups exposes installed, named groups
for the selected profiles. Each group contains whole declared storage owners;
owners sharing one physical database cannot be split into record subsets.
Conversations, notes, personas, study, and quizzes in ChaChaNotes therefore form
one group. Required linked groups are added to the review with an explanation.
Changing the selection invalidates the review. Empty and unknown selections
refuse. The archive contains one coherent backup set with explicit group scope;
each group's payloads remain independently identifiable and restorable.

Selected-group completeness is distinct from whole-profile coverage. A backup
that deliberately excludes groups can be coherent; missing required selected
data is still incomplete. Existing version-one full archives remain readable.
Installed owner policy, never archive-provided executable rules, determines
membership and dependencies. Existing explicit external-folder, model-artifact,
temporary-media, credential, encryption, and diagnostics options remain.

Restore defaults to all available groups. Users can restore a subset from full
or selective archives into a separate location or an existing profile. Existing
profile restoration creates and verifies a safety copy first, replaces only
reviewed selected and required groups, and preserves other stored data. There
is no record merging or conflict resolution. Reverse dependencies in the target
must be included in the review or cause refusal; silently retiring unselected
projections or orphaning retained records is not preservation.

## Implementation boundaries

Reuse the installed owner adapters, capture coordinator, archive container,
encryption, destination planner, native filesystem operations, admission,
journaling, and recovery UI. Add a small installed group catalog and selection
resolver rather than independent per-module backup executors. Group closure
includes whole owners, declared dependencies, shared stores, and tree topology.
Configuration required to interpret a group is support data, not implicit
permission to replace Settings. Discovery and maintenance remain conservative;
this change does not optimize how many services are paused.

Version two adds typed requested/effective group scope and support-member IDs.
The reader validates this metadata against installed group policy and the
authenticated producer inventory. Version one retains its existing checks and
is projected onto the installed catalog for selection. No manifest metadata
grants local filesystem authority. Existing full capture remains available
through the same service.

For existing-target restores without Settings, the local plan records an
explicit imported-config dependency to verified local-config association.
Local configuration determines the destinations and supplies a bounded private
copy to semantic validators under separately proven local provenance. Incoming
settings never silently change local paths or preferences. Invalid or changed
local configuration refuses. The association is bound to the plan digest,
saved plan and receipt, staging, native namespaces, publication, Finish, Abort,
and later rollback. Retained configuration is never a publish or retire row.
Safety-copy support does not authorize restoring that support as user settings.

Selecting Settings also requires proving that retained owners remain reachable
at their independently observed locations. If proposed settings cannot preserve
that condition, require the affected groups or refuse before publication.
Finalization validates the installed profile and advances activation/control
metadata even when user configuration bytes remain unchanged.

Refinement explicitly approved by the user on 2026-09-15 after static review:
deleting an empty selected source can leave a redundant file namespace beneath
an already registered profile directory. Effective physical-root validation may
omit that missing alias only when its historical canonical slot is covered by
another native-verified directory in the same declared scope. Every profile
referencing the alias must independently own that directory authority. Serialized
root lists, namespace names, and historical lock tokens remain unchanged; the
missing alias grants no source-read authority. Standalone, foreign, redirected,
and uncovered missing roots still refuse. This also permits external deletion of
such a redundant child inside the already owned directory; data discovery and
actual source reads continue to validate their own required content and identity.
Pending recovery retains the existing journal proof. Admission and finalization
must verify exhaustive profile coverage before normal operation can resume.

## Acceptance and exclusions

Behavioral tests cover exact selection, forward/shared/reverse dependencies,
strict archive scope, changed review refusal, preservation of unselected bytes
and identity, retained-config validation, safety copies, interruption recovery,
and later rollback. Native installed flows and first-time/power-user UAT run on
macOS, actual Linux SSH, and the authorized Windows Actions runner. Full/legacy,
encryption, and credential workflows retain their existing protections.

No cloud storage, scheduling, deduplication, remote server backup, new language,
record merging, unrelated Models fixtures, or replacement of the native recovery
engine is part of this migration. The prior backup UAT evidence is historical
evidence, not evidence that this migration has passed.

Design review reference: `/private/tmp/uat-selective-replacement-config-design-review.md`.
