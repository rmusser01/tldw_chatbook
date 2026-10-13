# Structured Notes editor integration qualification

Task34656 implements ADR194 through the canonical local Database Notes session
and existing File Notes workspace. The app owns one DocumentWorker service;
editor adapters own transient check leases and native history only. They cannot
save. Database authority is the captured canonical DB-path identity, notes user
and explicit global/null workspace, not the selected remote profile. The separate
server_notes_workspace_service does not become a local validation/save port.

Mode, Check, Format and keyboard Problems are mounted on both real surfaces.
Database mode is durable; File mode is temporary per open. Mode changes preserve
source. Initial/manual/mode checks are immediate; native typing retires prior
results and schedules the existing400ms lane. Full identity, live source and
immutable bridge capture fence application. Losing authority, conflict,
editability, visibility or attachment permanently retires the controller; a
restored host acquires a new lease. Actual app teardown closes the service.

Ruling43 makes secondary Research Quick Notes Source preview exact and
markup-disabled for every mode because its read contract omits language
presence. This costs rendered Markdown in that secondary surface. Main Notes
retains rendered null/Markdown preview and escaped structured/unknown source.
Ruling44 maps only File CRLF proposals to the incumbent LF view, preserving
original BOM/newline/disk serializer facts; a mapped equality is unchanged and
creates no history/revision/autosave. Ruling45 fences a Database body that cannot
native-round-trip exactly: source preview remains exact and title/mode/save use
the original canonical body; native body edit/check/format are unavailable.
Existing copy/export capture canonical fields. Supported LF/CRLF/lone-CR native
profiles remain separate from engine admission. No DB normalization or second
persisted body was added.

Mounted controls exercise real SQLite/disk saves, syntax-error saveability,
native A/B/C/D→C→B→C→B→A history, stale/manual/mode/authority/conflict controls,
BOM/CRLF no-op and bytes, nonzero viewport/caret/undo/redo,200k excerpts/hash
conflicts, emoji/tab/combining/CRLF/EOF navigation and diagnostic truncation.
The compact body is checked in its actual composited content rows, with typing
and keyboard Problems return; source tokens and the builder own all CSS.

Qualification: targeted covering selection213 passed; subsequent ancestor-guard
native-host selection34 passed; final rendered-body/CSS selection16 passed.
Exact source manifests and command/output hashes are retained in the durable
Task12 evidence directory. These are separate revisions, not one combined count.
Only a final File type-import blank line changed after the34 selection; the
whole AST and before/after hashes prove that formatting transfer.

New production modules pass Ruff/Black/Bandit/mypy; all four new files pass
Ruff/Black and Bandit with only intentional test assertions excluded. Expanding
mypy to the test imports hits the inherited Screen '# type: since…' comment parse
barrier. Legacy per-owner checks retain their existing debt with no additions;
legacy formatter differences do not grow. No broad cleanup was made.

The213/34 selections retain FD warnings323/328 respectively. The count-only
controls show growth with validation disabled and enabled. Immutable factory and
BaseDB bytes plus identical allocator ASTs attribute constructor allocations to
existing Collections/Workspace/Subscriptions DB owners (3 descriptors each),
not an active parser. This is partial owner attribution, not proof that every
remaining descriptor is accounted for. Independent actual app teardown retires
its source lease and releases its process. No host-wide resource repair landed.

Physical OS IME, screen-reader/hardware behavior, all supported platforms and a
full clean-installed parent remain downstream qualifications. E4 R39–R42 and I1
limits remain; no pure package/bootstrap/engine/worker/bridge contract changed.
Task34656 stays In Progress for independent controller review. No full suite,
server implementation, new export labels, provider/config/store/editor replacement,
shared installation rebind, push or merge was performed.


Fix1 qualification (Ruling46, original review Important I1): Quick Notes' native
TextArea normalizes some mixed line endings. The secondary view now fences those
bodies using canonical loaded source for preview, snapshots, download and title/
metadata/save/navigation flush. Body editing and Clear are unavailable, visibly
explained in its existing status; supported drafts regain native editing and
preview/save current edits. No language-presence, persisted owner or normalization
contract changed. ADR194 records this secondary cost before adoption.

The new RED measured mixed JSON/Markdown preview loss and a title-only save
request's normalized body; it stopped before a write. Corrected mounted controls
use actual Save-button/controller/LocalResearchWorkspaceAdapter/NotesScopeService/
NotesInteropService and dedicated SQLite databases, retaining exact mixed source
and known json metadata through title-only Save and navigation flush. Supported
native typing/preview/Save, queued old Changed and owner navigation, native delete
refusal and exact homogeneous CRLF/lone-CR previews are positive controls. Both
Quick Notes test files pass85 tests; a subsequent fixture-owned interop connection
cleanup is separately covered by its SQLite test (1 passed), with production
bytes unchanged. Immutable cb6e6e5 comparisons add no scoped
Ruff/mypy/formatter debt. Fix1 commands/hashes are separate in task-12-evidence/fix1;
the original522-artifact index remains unchanged. Original limits and all three
Minor findings remain open for final branch triage; no physical input/resource or
whole-branch qualification is implied by this Important-only fix.
