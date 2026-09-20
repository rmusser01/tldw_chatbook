# Workspace Binding Exclusions — Design

Date: 2026-09-20
Status: Draft (pending review)
Related ADR: ADR-172 (to be created with this design), amends the enforcement seam shared by ADR-101/102; interacts with ADR-028, ADR-069, ADR-079

## Problem

A named Workspace may bind local folders that give the Console's local path
tools (`fs_*`, read-only Git) access to a subtree. Users need a way to mark
specific files or folders inside a binding as explicitly off-limits: the agent
must not read, write, list, glob, grep, or Git-inspect them. Today the only
equivalent is the global, code-defined sensitive-path denylist
(`Utils/sensitive_paths.py`), which users cannot extend per workspace.

## Requirements

- A user can exclude any exact file or folder path under a workspace folder
  binding, and remove that exclusion later.
- "Excluded" means **fully invisible to the agent**: direct tool access is
  refused, and enumeration tools (`fs_list`, `fs_glob`, `fs_grep`, Git)
  omit the entry. The agent cannot learn the path exists.
- The user's own surfaces (Console file inspector, Settings) remain
  direct-user authority (ADR-079): excluded entries stay visible to the
  user, badged, and un-excludable.
- Mid-conversation exclusion additions take effect on the next tool call.
  Removals take effect on the next run (authority never expands mid-run).

## Non-goals

- Pattern matching (gitignore-style globs, bare names). Exact literal paths
  only. A pattern engine may arrive later as a separate task.
- Per-path read-only carve-outs (exclude = no access at all, not "read but
  don't write").
- Hiding excluded entries from the user's own inspector.
- Filtering the user's own ingestion paths (media import, Notes sync, manual
  RAG ingestion). If the user feeds an excluded file into RAG through their
  own authority, that is a user action outside the agent boundary.
- Any effect on the default workspace (it has no bindings) or private
  per-chat scratch (ADR-082).

## Decision summary

Exclusions are user-managed, per-binding data enforced by merging into the
existing sensitive-path exclusion pipeline — the same choke point, the same
per-candidate enumeration filters, the same Git pathspec translation, and the
same serialized `sensitive_exclusions` field in ADR-101's one-shot pinned
worker request. No new matching engine, no new enforcement seam, no worker
protocol change.

Rejected alternatives:

- **Permission-store deny rules** (ADR-032 store): the store is keyed by
  tool/server state, not paths; it can refuse calls but cannot remove
  entries from listing results; the pinned worker never consults the store.
- **Negative bindings** in root resolution (`Tools/workspace_file_roots.py`):
  binding validation deliberately forbids nesting/overlap, so excluded
  subtrees cannot be expressed as bindings; would also require a second
  implementation seam across the builtin and admitted-roots tool families.

## Detailed design

### 1. Data model and storage

- Persisted in `workspace_runtime_bindings.metadata_json` under a new
  `exclusions` key, alongside the existing `access` key:
  `{"path": "<binding-relative POSIX path>", "kind": "file"|"directory",
  "added_at": "<ISO-8601 UTC>"}`.
  No schema migration. `LocalWorkspaceRegistryService` remains the single
  serialized writer. Trade-off accepted: no per-row audit trail for
  exclusion edits in v1.
- New registry API, mirroring existing binding CRUD style:
  - `add_binding_exclusion(workspace_id, binding_id, path, kind)`
  - `remove_binding_exclusion(workspace_id, binding_id, path)`
  - `list_binding_exclusions(binding_id) -> tuple[...]`
- Add-time validation (reusing existing path machinery):
  - relative path only: rejects absolute paths, `..`, any component escaping
    the binding root, symlink escapes;
  - rejects excluding the entire binding root (remove the binding instead);
  - deduplicates via the denylist `_compare_key` casefold discipline;
  - hard cap of 200 entries per binding (bounds the serialized worker
    request);
  - the target need not exist on disk (pre-excluding `build/` is valid).
  - `kind` is display metadata only — enforcement always matches "this path
    and everything under it" (a file simply has nothing under it). When the
    target exists, `kind` is inferred from disk; when it does not (Settings
    text input), it defaults to `directory`.
- Naming: user-facing "Excluded"; internal `binding exclusions` /
  `user_exclusions`. The words "denylist" (means `sensitive_paths`),
  "SensitiveExclusion" (existing dataclass), and "deny" (a permission-store
  state) are deliberately avoided.

### 2. Enforcement

- **Run admission**: `capture_run_admitted_workspace_roots` attaches the
  frozen exclusion snapshot (paths + fingerprint) to each
  `RunAdmittedWorkspaceRoot`, following ADR-102's freeze-at-admission
  discipline.
- **Per-call merge**: `WorkspaceToolExecutor.execute` already captures
  `sensitive_exclusions_under(root, context)` on every call and serializes
  them into the pinned worker request. User exclusions merge into that same
  `sensitive_exclusions` request field; the closed worker protocol schema is
  unchanged, and the pinned worker enforces per-candidate as it does today.
- **Mid-run semantics**: the executor keeps a per-run high-water mark.
  Effective set = frozen snapshot ∪ high-water mark of live registry sets
  observed at each call. Consequences:
  - an exclusion added mid-run is enforced on the next tool call
    (fail-closed shrink, the same principle as ADR-102's binding-removal
    refusal);
  - an exclusion removed mid-run stays enforced for the rest of the run
    (authority never expands mid-run);
  - add-then-remove within one run stays excluded (monotone mark).
  - if the live registry read fails, the last-known effective set is reused
    (protection never shrinks on a read glitch); log and continue.
- **In-process choke point**: `resolve_workspace_path`'s deny set becomes
  `sensitive paths ∪ effective binding exclusions`. The
  `Tests/Tools/test_local_tool_sensitive_paths.py` tripwire contract extends;
  no third path-resolution seam is introduced.
- **Enumeration**: `fs_list` / `fs_glob` / `fs_grep` per-candidate filters
  omit excluded entries; an excluded directory prunes its subtree.
- **Git tools**: excluded paths append `:(exclude,literal)` pathspecs via
  the existing `_denylist_pathspecs` translation.
- **Project instructions**: `AGENTS.md` / `AGENTS.override.md` discovery and
  activation under an excluded path is skipped (ADR-069 ledger never
  activates instructions from excluded paths).
- **Model-facing opacity**: refusals hitting an excluded path reuse the
  exact sensitive-path refusal message shape and tokens — the model cannot
  distinguish a user exclusion from the system denylist, and never learns
  the path exists. User-facing surfaces (events, logs, Settings) state
  "excluded in workspace settings" explicitly.
- **Prompt transparency**: `workspace_context_note` does not name excluded
  paths (naming them would leak their existence). The tool catalog is
  unchanged — tools stay advertised; they simply cannot reach those paths.

### 3. UX

- **Console files modal** (`Widgets/Console/console_workspace_files_modal.py`
  over `Workspaces/file_inspector.py`): entries carry an excluded state,
  rendered dimmed with an "excluded" badge (composed from `$ds-*` tokens per
  ADR-150; no ad-hoc literals). A single-letter key action (ADR-031
  conventions, footer hint only if implemented) toggles exclude/unexclude on
  the selected entry. The inspector's own listing and reads are **not**
  filtered — only badged (direct-user authority, ADR-079).
- **Settings screen** (`UI/Screens/settings_screen.py`,
  `_render_workspace_folder_bindings`): under each binding row, an
  exclusions sub-list (path, file/folder kind, remove action) and an add
  input validated by the same registry rules. This is the canonical config
  surface; the modal is the in-context convenience surface. Both call the
  same registry API.
- Binding gone missing: exclusions remain listed in Settings with the
  binding's missing status; they are inert (no admitted root enforces them).

### 4. Edge cases

- Overlapping entries (`docs/` and `docs/secrets.md`): both apply; harmless
  redundancy; dedup remains exact-path only.
- Renamed or moved targets: the exclusion goes inert (matches nothing) and
  stays in the list until the user removes it.
- Case-insensitive comparison applies to exclusion matching only (denylist
  discipline); confinement checks remain case-sensitive — the exclusion set
  extends only the deny side, never the confinement side.
- Cross-workspace: two workspaces may bind overlapping roots; exclusions
  are per-binding and enforced only against their own admitted root.
- Worker request size: capped at 200 literal entries per binding; literal
  pathspecs are cheap to serialize and apply.
- RAG: workspace folder bindings are not auto-indexed into the vector store;
  the only agent-side routes into binding files are the fs/Git tools and
  project instructions, all covered above.

### 5. Testing

Targeted runs only (repo policy; no full sweep unless requested):

- Registry: CRUD round-trips; validation (absolute, `..`, root escape,
  symlink escape, whole-root, duplicate casefold, cap, non-existent OK).
  Location: `Tests/Workspaces/` next to existing registry tests.
- Choke point: user exclusions refuse `fs_read`/`fs_write`/`fs_edit`/
  `fs_patch` and are invisible in `fs_list`/`fs_glob`/`fs_grep` results.
  Location: `Tests/Tools/test_local_tool_*` family.
- Pinned worker: user exclusions serialized in `sensitive_exclusions` and
  enforced inside the pinned root.
- Git: `:(exclude,literal)` pathspec coverage for excluded paths.
- Run admission: snapshot freeze; mid-run addition refuses the next call;
  mid-run removal stays excluded for the run; registry-read failure reuses
  last-known set.
- Project instructions: an excluded folder's `AGENTS.md` never activates.
- UI: Settings exclusion list render + add/remove actions; files-modal
  badge + toggle action; design-token governance test stays green.

## ADR plan

ADR required: yes.
ADR path: `backlog/decisions/172-workspace-binding-exclusions.md`
Reason: security/permission boundary decision (agent file-access restriction)
plus a storage decision (metadata_json vs. new table), interacting with
ADR-028, ADR-069, ADR-079, ADR-101, and ADR-102. Created before
implementation begins and linked from the backlog task.

## Open questions

None — semantics, matching, surfaces, and mid-run behavior were all settled
during brainstorming (fully-invisible semantics; exact paths only; both
Settings and files-modal surfaces; additions immediate / removals next-run).
