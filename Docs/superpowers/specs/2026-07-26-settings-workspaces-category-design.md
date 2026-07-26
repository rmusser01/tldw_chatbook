# Settings ▸ Workspaces — management category with folder access roots

Date: 2026-07-26
Status: approved design, pending implementation
Supersedes: the Stage-5 "sync & workspace status in Settings is read-only"
boundary (`Docs/superpowers/plans/2026-05-29-settings-configuration-hub.md`);
a new ADR (see §8) records the supersession.
Builds on: workspace UX review + fix campaign (PRs #928/#932, tasks 712-723,
ADR-027).

## 1. Goal and decisions (locked with the user)

A dedicated **Settings ▸ Workspaces** category that hosts workspace
management outright — create, rename, archive/unarchive, set active — and
lets users modify existing workspaces by **adding/removing folders**.

- **Folders are file-tool access roots.** A folder bound to a workspace
  defines where that workspace's agent file tools (`read_file`,
  `list_directory`, `write_file`) may operate, activating the dormant
  `workspace_runtime_bindings` machinery (`kind=local-filesystem`). The
  Details tray's "File tools: N ready, M missing" copy becomes true.
- **Coexistence:** Settings is the management home; the Alt+W switcher keeps
  switch/rename/archive and Library keeps create as in-context quick actions.
- **Two stacked PRs, merged as one train** (§7): PR1 = service + enforcement,
  PR2 = the Settings page. Never ship management UI for behavior that does
  not exist yet (the aspirational-UI failure the UX review condemned).

## 2. Service layer (PR1)

`LocalWorkspaceRegistryService` (`tldw_chatbook/Workspaces/registry_service.py`)
gains:

- `unarchive_workspace(workspace_id)` — clears `archived`; does **not**
  auto-activate. `WorkspaceNotFound` for unknown/not-archived; Default is
  never archived so needs no special case.
- `add_folder_binding(workspace_id, path, *, allow_write=False)` — thin
  wrapper over the existing `save_runtime_binding`:
  - Path normalization: `expanduser()` + `resolve()` (canonical locator; a
    symlinked root is stored as its target).
  - Validation: must exist and be a directory; **deny** the filesystem root
    and the user's home directory itself (`WorkspaceRegistryServiceError`
    with explicit copy); deny duplicates and roots nested inside an existing
    binding of the same workspace (and vice versa — an existing child root
    is reported so the user can remove it first).
  - Default workspace: rejected (`"The Default workspace cannot have folder
    bindings."`) — preserves the existing design where
    `_delete_default_runtime_bindings` keeps Default tool-less.
  - Stored fields: `binding_kind=LOCAL_FILESYSTEM`, `locator=<resolved
    path>`, `label=<basename>`, `status=ready|missing` (display-only, see
    §3), `metadata={"access": "rw"|"ro"}` — **`ro` is the default**; write
    access is per-folder opt-in.
- `remove_runtime_binding(binding_id)` — deletes the row; raises a new
  `BindingNotFound(WorkspaceRegistryServiceError)` for unknown ids.
- `list_folder_bindings(workspace_id)` — `list_runtime_bindings` filtered to
  `LOCAL_FILESYSTEM`, with status recomputed from the filesystem at read
  time (a stored `ready` is never trusted for display either).
- `rename_workspace` / `create_workspace` (Settings path): add a
  case-insensitive duplicate-name check across non-archived workspaces
  (`WorkspaceRegistryServiceError: "A workspace named X already exists."`).
  The switcher's disambiguation depends on distinct names. (Applies to the
  service so Alt+W rename gets it too.)

## 3. Enforcement (PR1) — folders become real

`tldw_chatbook/Tools/file_operation_tools.py` currently confines every
operation to one global sandbox root (`[tools] file_sandbox_root`, default
`<user data dir>/tool_sandbox`), resolved **at call time** inside each
tool's `execute()`. Changes:

- New module-level seam `_allowed_roots(write: bool) -> tuple[Path, ...]`:
  the global sandbox root **plus** the bound folders of the *resolved
  workspace* (see below). For `write=True`, only folders with
  `metadata.access == "rw"`. Each folder is included only if the directory
  **exists at call time** — stored status is never trusted; a deleted folder
  drops out of the allowed set immediately.
- New `validate_path_multi(path, roots)` beside the existing
  `validate_path`: succeeds if the resolved path is within any root; its
  error copy names the roots that were consulted (so a denial is
  actionable). Existing single-root callers are untouched.
- `ReadFileTool`/`ListDirectoryTool` use `_allowed_roots(write=False)`;
  `WriteFileTool` uses `_allowed_roots(write=True)`.
- **Workspace resolution — the run, not the mouse.** The tool catalog
  (`Agents/tool_catalog.py`) is where tool instances are constructed for a
  run; it injects a roots-provider closure bound to the run's workspace id
  (from the run/session context the agent engine already carries). Only
  when a run has no workspace context does the provider fall back to a
  call-time `get_active_workspace()` read. This prevents a mid-run Alt+W
  switch from silently retargeting where a running agent may write.
  `WorkspaceDB` opens a fresh connection per call, so worker-thread reads
  are safe. If inspection during implementation shows the engine truly has
  no per-run workspace seam, the fallback becomes the behavior and the spec
  note is updated — do not invent a parallel context channel for this.
- **Gates unchanged:** the existing built-in tool gates (Allow/Ask/Off,
  `read_file_enabled` etc.) still decide *whether* a tool runs; folders
  only widen *where*. Default workspace: no bindings can exist, so tools
  keep today's sandbox-only behavior there — as designed.
- Net effect with zero bindings: byte-for-byte today's behavior. Safe to
  merge PR1 alone.

## 4. Settings category (PR2)

**Integration touchpoints** (all in `UI/Screens/`): add `WORKSPACES` to
`SettingsCategoryId` (`settings_config_models.py`); category summary +
"Data & Privacy" group (`_category_summaries`, `_category_groups`); detail
dispatch (`_render_detail_pane` → new `_render_workspaces_detail`); add to
the Save/Revert **suppression** set alongside Theme/Splash (management
actions are immediate, not draft-based; the Scope Inspector shows an
explanatory guided-action message instead); ownership record (owns workspace
lifecycle + folder bindings; names Console/Library quick actions);
deep-link navigation target. Do **not** add it to
`GUIDED_SETTINGS_MUTATION_CATEGORIES`.

**Page layout** (single detail pane, list-then-detail):

- Workspace list: name, id, `(active)` marker, folder count, archived badge;
  a "Show archived" checkbox (off by default). Selection is in-pane state.
- Detail card for the selected workspace:
  - Rename: Input + Apply (duplicate names rejected with the service copy).
  - Set active button (disabled-with-inline-reason when already active).
  - Archive / Unarchive (ConfirmationDialog for archive, same copy contract
    as the switcher: conversations stay saved and visible in Library;
    archiving the active workspace falls back to Default; unarchive never
    auto-activates).
  - **Folders**: one row per binding — locator, ro/rw tag, live
    ready/missing status; a per-row write-access toggle; per-row Remove.
    Below: Add-folder Input (`~` expansion) + Add button; service
    validation errors surface inline next to the input, not tooltip-only
    (TASK-716 lesson: a disabled control with a tooltip explanation is
    unreachable — blocked states explain themselves inline).
  - Default workspace: rename/archive/folder controls replaced by a static
    inline explanation ("The built-in Default workspace keeps its identity
    and stays tool-less; create a workspace to bind folders.").
- Create: name Input + Create button at the top of the list pane (id stays
  generated `workspace-local-N`; the name is free-form — first surface
  without forced "Workspace N" names).
- **Freshness:** the list re-reads the registry on screen resume and after
  every action (Console/Library mutate the registry concurrently). Follow
  the existing settings recompose-hygiene patterns (coalesced refresh +
  focus restoration, task-290; no watcher-built content that a deferred
  recompose can wipe).

**Cross-surface copy (PR2):** Overview's workspace rows and recovery copy
name Settings ▸ Workspaces as the management home (replacing the
"Library > Details > Workspace" pointer shipped in TASK-719); the Details
tray's file-tools row stays as-is (it now reports real state); Console/
Library quick-action copy unchanged.

## 5. Security model (summary)

- Read-only by default; write is per-folder opt-in (`metadata.access`).
- Deny `/` and `$HOME` as roots; locators stored canonical (resolved).
- Symlinks inside a bound root that escape it are already denied by the
  existing resolve-then-compare check (`_is_within`); `validate_path_multi`
  keeps that property per root.
- Existence re-checked at call time; gates still control tool availability;
  Default workspace remains tool-less.

## 6. Testing

- **Service (PR1):** unarchive (incl. not-auto-activating), folder add
  (validation matrix: missing path, file-not-dir, `/`, `$HOME`, duplicate,
  nested-either-direction, Default rejection, ro default), remove, status
  recompute, duplicate-name rename/create.
- **Enforcement (PR1):** read allowed inside a bound folder of the run's
  workspace; write denied for `ro` folder and allowed after `rw` toggle;
  denied outside all roots with error copy naming roots; deleted-folder
  drop-out; sandbox-only fallback with zero bindings byte-identical to
  today (regression pin); other-workspace folders not consulted.
- **UI (PR2):** mounted Settings tests for CRUD flows, Default protections,
  archived filtering, inline validation errors, refresh-on-action; a
  Save/Revert-suppression assertion; cross-surface copy test for the
  Overview pointer.
- Suites to keep green: settings hub, Workspaces, workspace lifecycle/
  keyboard, agent tool-gate tests, native-chat-flow.

## 7. PR phasing

- **PR1 `feat/workspace-folder-bindings-enforcement`**: §2 service + §3
  enforcement + §5 + their tests + the ADR (§8). Zero-binding behavior
  identical to today.
- **PR2 `feat/settings-workspaces-category`** (stacked on PR1): §4 UI +
  cross-surface copy + UI tests.
- Merge as one train (PR1 then PR2, same session) so no released state has
  management UI without enforcement or enforcement without a management
  surface.

## 8. ADR

`backlog/decisions/028-settings-workspaces-category-and-folder-roots.md`:
records (a) superseding the Stage-5 read-only boundary — Settings now owns
workspace lifecycle + folder bindings, with Console/Library keeping quick
actions; (b) folder semantics: file-tool access roots, read-only default,
call-time existence validation, run-bound workspace resolution, Default
stays tool-less.

## 9. Out of scope

Non-filesystem binding kinds (git-worktree, container, VM, remote, ACP);
per-folder permissions beyond ro/rw; folder pickers (text input only);
watching/ingesting folder contents (that is Library/watchlists territory);
server sync of bindings.
