# Workspace Create Modal + Project Skills (.SKILLS/) Discovery — Design

- **Date:** 2026-08-17
- **Status:** Proposed (awaiting owner review)
- **Supersedes (partially):** `2026-07-26-settings-workspaces-category-design.md` §1 — that spec locked
  "Library keeps create as an in-context quick action" with zero-input creation. This design replaces
  instant creation on **all three** surfaces (Console rail, Settings ▸ Workspaces, Library) with a
  shared creation modal. Everything else in that spec (Settings as management home, folder-binding
  editor, Alt+W switcher scope) stands.
- **Related ADRs:** ADR-028 (folder bindings are file-tool access roots, ro default, Default
  workspace stays tool-less), ADR-009 (local skill trust boundary). This design adds a new ADR for
  the project-skills folder convention (see §8).
- **Related backlog:** task-713 (silent workspace creation), task-714 (forced "Workspace N" names) —
  both partially addressed by Feature 1.

## 1. Problem

Workspace creation today collects nothing: Console's `New` button and Library's create button
auto-name "Workspace N" with zero input (`UI/Console_Modules/workspace.py:837`,
`UI/Screens/library_screen.py:19853`); Settings collects an optional name only
(`UI/Screens/settings_screen.py:14415`). Folder binding — the thing that makes a workspace *do*
anything for agents — is a separate, post-creation, Settings-only, type-the-path-by-hand action.
Users are left with a new entry and no idea what it changes.

Separately, a user resuming work on an existing project has no way to bring that project's skills
into the app short of importing them one directory at a time through Library ▸ Skills. There is no
project-local skills convention and no cwd awareness at startup.

## 2. Goals

1. Creating a workspace from any surface prompts for a **name** and optional **folder path(s)** in a
   modal that explains, truthfully, what a workspace is and why a folder is being requested.
2. A directory containing a **`.SKILLS/`** folder (per-skill subdirectories with `SKILL.md`, or
   loose `*.md` skill files) triggers a **prompt-driven, never silent** offer to import those skills:
   on app startup in that directory, and when a workspace is created with that directory bound.

## 3. Non-goals

- **Live-loading** skills from `.SKILLS/` without importing (would cross the ADR-009 trust
  boundary and requires per-workspace skill scoping that does not exist). Possible later phase.
- **Bulk trust approval.** Imported skills land quarantined exactly as today; trusting stays the
  per-skill Review → Approve → passphrase flow. One-click-trusting a freshly cloned repo's skills is
  what the boundary exists to prevent.
- Recognizing `.claude/skills/` / `.agents/skills/` layouts (same shape, large ecosystem) — tempting
  follow-up task, not v1.
- Changing what folder bindings mean (ADR-028 semantics untouched), the Alt+W switcher, or the
  Settings folder-binding editor.

## 4. Feature 1 — shared workspace creation modal

### 4.1 Component

New `tldw_chatbook/Widgets/workspace_create_modal.py`:
`WorkspaceCreateModal(ModalScreen[WorkspaceCreateResult | None])`, patterned on
`ConsoleWorkspaceRenameModal` (escape binding, Save/Cancel, `Input.Submitted`). Layout top to
bottom:

1. **Educational header** (Static, ~4 sentences). Draft copy — must stay truthful to shipped
   behavior:
   > *A workspace scopes the Console to one project. Conversations started in it are grouped
   > together, agents' project file access comes only from the folders you bind here (read-only
   > unless you grant write in Settings), and retrieval can be narrowed to the workspace's items
   > via its RAG Scope.
   > Binding your project's folder is what makes a workspace more than a label — without one, agents
   > have no file access. You can add or change folders later in Settings ▸ Workspaces.*

   Note the precision: folder bindings drive **file-tool access roots** (ADR-028); RAG scoping is
   driven by the separate per-workspace item picker (live and enforced —
   `Chat/rag_scope.py:219` → allowlists in `RAG_Search/pipeline_functions_simple.py`). The copy
   must not imply binding a folder scopes RAG.
2. **Name input**, prefilled from `next_local_workspace_identity(registry)`
   (`Workspaces/registry_service.py:1185`) so completion stays Enter-cheap. The workspace **id** is
   always the generated one regardless of the typed name (matches Settings today).
3. **Folder list**: path `Input` + **Browse…** `Button` (pushes the existing
   `Third_Party/textual_fspicker/SelectDirectory`) + **Add** — added folders render as removable
   rows. Multiple folders supported (schema is N bindings). Folders are **optional**; leaving the
   list empty is valid and the header copy states what is lost.
4. **"Switch to this workspace"** checkbox, **default checked** — makes activation behavior uniform
   across surfaces (today Console/Library activate, Settings silently doesn't).
5. **Create / Cancel** buttons. `escape` = Cancel = dismiss with `None`, nothing created.

### 4.2 Validation (before Create, not after)

Extract a **pure path validator** from `add_folder_binding`
(`Workspaces/registry_service.py:659-726`) — e.g.
`validate_folder_binding_path(path, existing_locators) -> Path` raising a typed error — covering:
expanduser/resolve, must be an existing real directory, not filesystem root, not `$HOME`, sensitive-
path denylist (`find_root_binding_conflict`), and duplicate/nested/parent overlap **among the
modal's own entries** (a new workspace has no prior bindings, so intra-modal overlap is the only
conflict class). `add_folder_binding` is refactored to call the same validator so the rules cannot
drift. The modal runs it **as each folder is added**, surfacing rejection inline immediately.

Name collisions: `workspace_records` has a unique index on `lower(name)` (non-archived). A collision
on Create renders inline in the modal (modal stays open); optionally pre-checked against
`list_workspaces` on blur.

### 4.3 Ownership split: modal creates, surfaces sync

The modal receives the registry service and **owns the service calls** on Create:
`create_workspace(...)` then `add_folder_binding(...)` per folder (pre-validated; residual TOCTOU
failures render inline as per-folder warnings — the workspace still exists, matching a partial
result in `WorkspaceCreateResult`). Create-level failure keeps the modal open with an inline error.

`WorkspaceCreateResult` carries: `workspace_id`, `name`, `bound_folders`, `failed_folders`,
`make_active`, and `project_skills` (see §5.5). Each surface keeps its **own** dismissal callback
for post-create UI sync:

- **Console** (`UI/Console_Modules/workspace.py:837` `_create_console_workspace` becomes
  push-modal + result handler): when `make_active` is set, replicates today's exact sequence —
  `set_active_workspace` →
  `_sync_console_chat_core_state()` → `_activate_console_session_for_workspace(workspace_id)` →
  `_sync_console_workspace_context()` → `run_worker(_sync_native_console_chat_ui(),
  exclusive=True, group="console-sync")` → the toast. When `make_active` is unchecked, only
  `_sync_console_workspace_context()` runs, with a "Created <name>." toast and no session switch.
  The toast **stays** despite the modal in both cases: task-713's comment marks it load-bearing
  when the status row is scrolled out of view.
- **Settings** (`settings_screen.py`): the inline name-input + Create row
  (`:11886-11893`, handler `:14415`) is **replaced** by a single "Create workspace…" button that
  pushes the modal; on result, honor `make_active` then refresh via the existing
  `mutate_reactive(SettingsScreen.active_category)` recompose.
- **Library** (`library_screen.py:19853`): same button → modal; existing activate/recompose/toast
  behavior preserved, honoring `make_active`.

All three surfaces then run the shared project-skills chaining helper (§5.5).

### 4.4 Implementation risk — spike first

Pushing a screen from inside a `ModalScreen` is established (6 sites, e.g.
`briefing_preset_modal.py:610`, which documents the snapshot-before-await discipline), but **no
site has ever pushed the fspicker from a modal** — every `SelectDirectory` push is from a full
screen. The Browse-from-modal interaction (focus return, callback delivery, escape handling) is the
first thing implementation verifies, before the rest of the modal is built. Fallback if it
misbehaves: modal temporarily dismisses to a picker and re-pushes itself prefilled (the
`document_generation_modal.py:301` chaining shape).

## 5. Feature 2 — `.SKILLS/` discovery and prompt-driven import

### 5.1 Convention

A project-local skills folder is a directory named `.SKILLS/` (or `.skills/`; if a case-sensitive
filesystem somehow has both, `.SKILLS/` wins and the other is ignored with a log line) at a
project root, containing:

- **per-skill subdirectories** with a top-level `SKILL.md` → imported via
  `import_skill_directory` (skill name = directory name), and/or
- **loose `*.md` files** → imported via `import_skill_file` (name derived from filename).

Entries that match neither (subdir without `SKILL.md`, non-markdown files) are reported as skipped
with a reason, never silently ignored.

### 5.2 Discovery module (pure)

New `tldw_chatbook/Skills_Interop/project_skills_discovery.py`:
`discover_project_skills(root: Path) -> ProjectSkillsDiscovery` — no side effects, no execution.
Hardening (untrusted repo input):

- Refuse a **symlinked** `.SKILLS/` directory and skip symlinked entries (mirrors
  `_iter_bundle_files` discipline in `local_skills_service.py`).
- Caps before any parse: max 50 entries enumerated (remainder reported as "N more not
  shown"), max 64 KiB read per `SKILL.md`/`*.md` for frontmatter, top level only — no
  recursive scan below the per-skill dirs at discovery time (the importer walks them later under
  its own caps).
- Frontmatter via the existing `_parse_front_matter`/`yaml.safe_load` grammar; a file that fails to
  parse is listed as "invalid" rather than aborting discovery.
- Derived skill names are pre-checked against the skills name grammar
  (`local_skills_service.py:82`); an entry whose directory/file name cannot normalize to a valid
  name (e.g. `My_Skill/`) is listed as "invalid — name must be lowercase-kebab" at discovery time
  instead of failing later inside the importer.
- `.SKILLS/` and `.skills/` candidates are deduplicated by resolved path (on case-insensitive
  filesystems both names match the same directory).
- Fingerprint: stable hash over sorted (entry name, size, mtime) of the recognized skill files —
  used by the ledger (§5.3).
- **Names/descriptions are untrusted display text**: rendered with Rich markup escaped everywhere
  they appear (modal rows, toasts, logs), so a hostile `description` cannot inject console markup.

### 5.3 Prompt ledger (nag-avoidance)

`<user_data_dir>/skills/project_prompts.json` (same atomic-replace pattern as the skills index):
`{version: 1, entries: {<resolved dir>: {decision: "imported"|"declined"|"never", fingerprint,
timestamp}}}`. Pure gating function, mirroring `first_run_setup_state.should_offer_wizard`:

> offer ⇔ prompt enabled **and** `.SKILLS/` present **and** (no entry, **or** decision ≠ "never"
> **and** fingerprint changed since the recorded one).

So "Not now" suppresses re-prompting until the skill set actually changes (which then usefully
reads as "2 new skills appeared in this project"); "Never for this folder" is permanent; a global
kill-switch `[skills] project_skills_prompt_enabled = true` is declared in `config.py` defaults.
Both triggers (§5.4) write the same ledger, so declining in one place silences the other.

### 5.4 Triggers

**Startup:** new `_maybe_offer_project_skills_import()` in `app.py`'s `_post_mount_setup`, next to
`_maybe_offer_first_run_wizard()` (`app.py:7878`). Rules:

- If the first-run wizard was offered this launch, **skip** (defer to next launch) — no modal
  stacking at startup.
- `Path.cwd()` guarded (the launch directory can be deleted out from under the process); resolved
  before ledger lookup. If cwd has no `.SKILLS/`, a **bounded upward walk** checks each ancestor,
  stopping after the first one containing `.git` (the project root) or on reaching `$HOME` or the
  filesystem root — so launching from a project *subdirectory* still finds the project's skills.
  The ledger is keyed by the directory where `.SKILLS/` was found. Launches from unrelated
  directories are covered by the second trigger.
- Discovery runs in a worker thread (startup path stays unblocked), then
  `call_from_thread`/`call_after_refresh` pushes the import modal, matching the wizard's push
  pattern.

**Workspace creation:** after `WorkspaceCreateModal` dismisses successfully, the surface's result
handler checks `result.project_skills` (the modal runs discovery on each folder as it is added — a
row in the folder list gains a "contains N project skills" annotation) and, after post-create sync
completes, pushes the import modal via a shared helper
`maybe_offer_project_skills_import(app, discoveries)` used by all three surfaces. When several
bound folders each carry a `.SKILLS/` directory (rare), import modals are offered **sequentially,
one per discovery**, each writing its own ledger entry — no merged multi-source modal. The workspace is
created regardless of what the user chooses here — skills are a **global** store; the workspace
trigger is a discovery moment, not a linkage (no workspace↔skill coupling exists or is added).

### 5.5 Import modal

New `tldw_chatbook/Widgets/project_skills_import_modal.py` (`ModalScreen`):

- **Header** states provenance ("found in `<dir>/.SKILLS/`") and sets the trust expectation up
  front: *imported skills require a one-time trust review in Library ▸ Skills before they can
  run.* This is load-bearing: quarantined skills are excluded from the Console skill picker
  entirely and `$mentions` are refused (`console_chat_controller.py:5692`), so without this framing
  the feature reads as broken.
- **Rows**: checkbox + name + status — `new` (checked by default), `already installed` (unchecked;
  never silently overwritten), `invalid` (unselectable, with reason). Skipped-entry and
  "N more not shown" lines included. All repo-sourced text escaped.
- **Buttons**: **Import selected** / **Not now** / **Never for this folder**. Escape = Not now.
- On import: iterate `import_skill_directory` / `import_skill_file` with
  `trust_approved=False` (unchanged security posture); collect per-skill outcomes; result view
  lists them with a **"Review in Library ▸ Skills"** button navigating via the existing `skills`
  Library sub-route (`UI/Navigation/shell_destinations.py:65`), plus Close. If the trust store was
  never bootstrapped, Library's adaptive trust header already routes through setup first — the
  result-view copy mentions this ("set up skill trust, then approve") rather than assuming a
  passphrase exists. Ledger updated with the decision + current fingerprint.

## 6. Error handling summary

- Modal-time validation failures (path, name) render inline; the modal never dismisses on error.
- Post-validation binding failures: workspace exists, failed folders listed inline as warnings and
  carried in the result (`failed_folders`) so surfaces can toast.
- Registry unavailable: Console's `New` is already disabled by `display_state`; Settings/Library
  buttons render the existing inline status error instead of pushing the modal.
- Discovery/import failures per-skill, never aborting the batch; import errors reuse the Library
  import flow's error surfaces.
- RAG-scope fails-closed behavior, trust flow, and folder-root call-time re-validation are all
  untouched.

## 7. Testing

- **Pure units** (no Textual): the extracted folder-path validator (each rejection class +
  intra-modal overlap); `discover_project_skills` against fixtures — happy layouts, loose files,
  symlinked `.SKILLS/`, symlinked entries, junk entries, oversized frontmatter, markup-hostile
  names, fingerprint stability; ledger gating function truth table (no entry / declined+same /
  declined+changed / never / kill-switch).
- **Pilot tests** (Textual): create modal — prefilled name Enter-Enter fast path, escape creates
  nothing, invalid path inline error, duplicate name inline error, Browse round-trip (the §4.4
  spike graduates into this test), result plumbed to each surface's handler; import modal —
  selection defaults, Never writes ledger, statuses render escaped.
- **Integration**: startup gating (wizard-offered ⇒ skipped; ledger honored), Console post-create
  sequence still activates session + resyncs rail (assert against the same seams the existing
  workspace tests use).
- Per repo rules: targeted test selection, real in-memory SQLite for registry tests, and a live
  `verify`-skill pass on the three surfaces before PR.

## 8. Documentation / decision records

- **New ADR**: project-skills folder convention — layout, both triggers, import-copy (not
  live-load) rationale, quarantine posture, relation to ADR-009's boundary (import-copy stays
  inside it; live-load would not).
- **Supersession note** added to `2026-07-26-settings-workspaces-category-design.md` §1 pointing
  here.
- **User Guide** pages for Console (workspace section), Settings ▸ Workspaces, and Library ▸ Skills
  updated (CLAUDE.md rule), including the `.SKILLS/` convention as user-facing documentation.

## 9. Phasing

Two independently valuable PR-sized stages, foundations first:

1. **PR A — creation modal**: validator extraction + `WorkspaceCreateModal` + three-surface wiring
   (includes the §4.4 spike). Ships alone; addresses task-713/714 concerns.
2. **PR B — project skills**: discovery module + ledger + import modal + startup trigger +
   create-modal chaining (the only touch on PR A is reading `project_skills` off the result).

Backlog tasks to be filed per repo hygiene (IDs swept against origin/dev + all remotes/worktrees).

## 10. Decisions taken (owner may override)

| # | Decision | Default taken |
|---|----------|---------------|
| 1 | Folder at creation | Optional, encouraged by copy — organizational-only workspaces stay legal |
| 2 | Activation | Uniform "Switch to this workspace" checkbox, default on |
| 3 | `.SKILLS/` handling | Import-copy through existing importer; quarantined; no bulk trust |
| 4 | Existing skill names | Skipped by default, shown as "already installed", explicit overwrite only |
| 5 | Convention names | `.SKILLS/` and `.skills/`; `.claude/skills/` deferred to a follow-up task |
| 6 | Escape semantics | Cancel — nothing created/imported; "Not now" for the import prompt |
| 7 | Startup detection scope | cwd plus a bounded walk up to the first `.git` ancestor (covers subdirectory launches); stops at `$HOME`/root |
