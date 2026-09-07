# Research Workspace Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:executing-plans` to execute this plan, use
> `superpowers:test-driven-development` for every behavior change, and use
> `superpowers:verification-before-completion` before each commit. Do not
> delegate unless the user explicitly requests subagents. Apply `impeccable`
> immediately before UI implementation tasks.

**Goal:** Add the Research shell destination and a real, authority-explicit,
responsive Workspace screen that safely hosts later Sources, Chat, and Studio
features.

**Architecture:** Keep `research` as the existing Research Runs route while a
new `research_workspace` route becomes the Research destination's primary
screen. A headless controller holds only normalized authority-qualified state;
two concrete adapters call the existing Local workspace registry or Server
workspace service. A private device overlay owns presentation preferences. A
pure layout reducer separates persisted preferences from effective responsive
pane state.

**Tech Stack:** Python 3.11+, Textual 8.x, frozen dataclasses/Protocols,
existing workspace and server services, private atomic JSON helpers, TCSS,
pytest.

**Spec:**
`Docs/superpowers/specs/2026-08-23-research-workspace-design.md`

**Backlog:** `TASK-21507`

## Global constraints

- Local and Server are explicit data sources. Never call the other adapter as
  fallback and never use `WorkspaceAuthority` as this discriminator.
- Preserve the direct `research` route and all thirteen current shortcut
  owners. Research Workspace alone receives F10.
- Keep controller/domain code Textual-free. Region widgets own pixels and post
  events; they do not call databases or HTTP clients directly.
- Use the existing `private_paths` primitives; do not persist overlay content
  through a default-0644 JSON helper.
- The visible collapse labels are exactly four characters: Sources collapse
  `<---`, Sources reveal `--->`, Studio collapse `--->`, Studio reveal `<---`.
- Do not add a dependency or a generic app-wide adapter framework.
- Run targeted tests only. Repository-wide pytest requires a separate user
  opt-in and cannot be claimed by this task.

## ADR check

ADR required: no new ADR

ADR path:
`backlog/decisions/078-research-workspace-authority-and-screen-boundaries.md`

Reason: ADR-078 already fixes the shell, route, authority, adapter, overlay,
and responsive pane boundaries. If implementation changes a canonical owner,
permits cross-authority fallback, or moves canonical content into the overlay,
stop and write a new ADR first.

## Task 1: Pin route and shortcut ownership

**Files:**

- Modify: `tldw_chatbook/UI/Navigation/shell_destinations.py`
- Modify: `tldw_chatbook/UI/Navigation/screen_registry.py`
- Modify: `tldw_chatbook/Constants.py`
- Modify: `tldw_chatbook/app.py`
- Test: `Tests/UI/test_shell_destinations.py`
- Test: `Tests/UI/test_master_shell_navigation.py`
- Test: `Tests/UI/test_command_palette_shell_routes.py`
- Test: `Tests/UI/test_ux_batch4.py`

1. Add RED tests proving the destination order is Home, Console, Library,
   Research, Artifacts, then the unchanged remainder; `research_workspace`
   resolves to Research Workspace; `research` resolves to the existing Runs
   screen; and the palette exposes one Research destination with all approved
   aliases.
2. Replace positional shortcut ownership with one immutable ID-keyed mapping:

   ```python
   SHELL_DESTINATION_SHORTCUTS = {
       "home": "ctrl+1",
       "console": "ctrl+2",
       "library": "ctrl+3",
       "artifacts": "ctrl+4",
       "personas": "ctrl+5",
       "watchlists_collections": "ctrl+6",
       "schedules": "ctrl+7",
       "workflows": "ctrl+8",
       "mcp": "ctrl+9",
       "acp": "ctrl+0",
       "lab": "f7",
       "logs": "f8",
       "settings": "f9",
       "research": "f10",
   }
   ```

3. Extend `ShellDestination` with explicit related real routes (not legacy
   aliases). Build route entries so `research` remains canonical to itself even
   though the destination ID is also `research`; palette/destination commands
   use `primary_route="research_workspace"`.
4. Change Textual bindings and `action_shell_destination` to accept a stable
   destination ID. Keep a narrowly documented integer compatibility path only
   if an existing public caller outside these tests requires it; new bindings
   and tests must be ID-based.
5. Register `TAB_RESEARCH_WORKSPACE`, `ResearchWorkspaceScreen`, and the new
   route without replacing `ResearchScreen`.
6. Run:

   ```bash
   .venv/bin/python -m pytest -q Tests/UI/test_shell_destinations.py Tests/UI/test_master_shell_navigation.py Tests/UI/test_command_palette_shell_routes.py Tests/UI/test_ux_batch4.py
   ```

7. Commit only this task's files:

   ```bash
   git commit -m "feat: add stable Research shell routes"
   ```

## Task 2: Define authority-qualified contracts and fail-closed adapters

**Files:**

- Create: `tldw_chatbook/Research_Workspace/__init__.py`
- Create: `tldw_chatbook/Research_Workspace/contracts.py`
- Create: `tldw_chatbook/Research_Workspace/local_adapter.py`
- Create: `tldw_chatbook/Research_Workspace/server_adapter.py`
- Create: `tldw_chatbook/Research_Workspace/controller.py`
- Test: `Tests/Research_Workspace/test_contracts.py`
- Test: `Tests/Research_Workspace/test_workspace_adapters.py`
- Test: `Tests/Research_Workspace/test_controller.py`

1. Add RED validation tests for blank IDs, Server refs without profile identity,
   secret-looking identity metadata, mismatched result refs, unknown
   capabilities, and Local/Server no-cross-call inverses.
2. Add the smallest normalized contracts:

   ```python
   class WorkspaceDataSource(StrEnum):
       LOCAL = "local"
       SERVER = "server"

   @dataclass(frozen=True, slots=True)
   class QualifiedWorkspaceRef:
       data_source: WorkspaceDataSource
       workspace_id: str
       server_profile_id: str = ""
       principal_id: str = ""

   @dataclass(frozen=True, slots=True)
   class ResearchCapability:
       available: bool
       reason_code: str
       user_message: str
       owner: str
       recovery_action: str = ""
       capability_revision: str = ""

   class ResearchWorkspacePort(Protocol):
       async def list_workspaces(self, *, include_archived: bool = False) -> tuple[ResearchWorkspaceSummary, ...]: ...
       async def get_workspace(self, ref: QualifiedWorkspaceRef) -> ResearchWorkspaceSummary | None: ...
       async def create_workspace(self, *, name: str, description: str = "", template_id: str = "") -> ResearchWorkspaceSummary: ...
       async def update_workspace(self, ref: QualifiedWorkspaceRef, *, name: str | None = None, expected_version: int | None = None) -> ResearchWorkspaceSummary: ...
       async def duplicate_workspace(self, ref: QualifiedWorkspaceRef, *, name: str) -> ResearchWorkspaceSummary: ...
       async def archive_workspace(self, ref: QualifiedWorkspaceRef, *, expected_version: int | None = None) -> ResearchWorkspaceSummary: ...
       async def restore_workspace(self, ref: QualifiedWorkspaceRef, *, expected_version: int | None = None) -> ResearchWorkspaceSummary: ...
       async def delete_workspace(self, ref: QualifiedWorkspaceRef, *, expected_version: int | None = None) -> bool: ...
       async def capabilities(self, ref: QualifiedWorkspaceRef) -> Mapping[str, ResearchCapability]: ...
   ```

3. Include `ResearchWorkspaceSummary`, `ResearchSourceSummary`,
   `ProcessingRoute`, and a bounded page result. Every row returned by an
   adapter carries its `QualifiedWorkspaceRef`; no UI cache key uses a raw ID.
4. Implement the lifecycle methods with owner capability checks. A method that
   the selected owner or current permission does not support returns the exact
   unavailable capability/recovery; it never calls the other adapter. Local
   destructive delete remains Settings-owned even though the port names the
   operation.
5. Implement `LocalResearchWorkspaceAdapter` over
   `LocalWorkspaceRegistryService`, running file-backed SQLite calls via
   `asyncio.to_thread`. Exclude `workspace-default` from Research notebook
   selection while leaving it valid elsewhere in Chatbook.
6. Implement `ServerResearchWorkspaceAdapter` over
   `ServerNotesWorkspaceService` and the active server-context provider. Missing
   profile, auth, network, or capability returns explicit unavailable state and
   does not instantiate/call the Local adapter.
7. Implement `ResearchWorkspaceController` with a monotonically increasing
   context revision. Every request captures qualified ref + capability
   revision + context revision; stale results may update their canonical owner
   but cannot replace visible controller state.
8. Run:

   ```bash
   .venv/bin/python -m pytest -q Tests/Research_Workspace/test_contracts.py Tests/Research_Workspace/test_workspace_adapters.py Tests/Research_Workspace/test_controller.py
   ```

9. Commit:

   ```bash
   git commit -m "feat: add Research workspace authority adapters"
   ```

## Task 3: Persist private presentation state and derive responsive layout

**Files:**

- Create: `tldw_chatbook/Research_Workspace/layout_state.py`
- Create: `tldw_chatbook/Research_Workspace/overlay_store.py`
- Test: `Tests/Research_Workspace/test_layout_state.py`
- Test: `Tests/Research_Workspace/test_overlay_store.py`
- Test: `Tests/Utils/test_private_paths.py`

1. Add RED table tests at 160, 150, 149, 120, 100, 99, 84, 80, and 60
   columns. Cover both side preferences, preferred companion replacement,
   Chat-only medium, exactly-one narrow pane, width restoration, and both
   explicit-toggle directions.
2. Implement a pure reducer with these owners:

   ```python
   @dataclass(frozen=True, slots=True)
   class ResearchPanePreferences:
       sources_open: bool = True
       studio_open: bool = True
       preferred_companion: Literal["sources", "studio"] = "sources"

   @dataclass(frozen=True, slots=True)
   class ResearchPaneLayout:
       mode: Literal["wide", "medium", "narrow"]
       visible_panes: tuple[Literal["sources", "chat", "studio"], ...]
       sources_forced_closed: bool
       studio_forced_closed: bool
   ```

   Wide is `>=150`, medium is `100..149`, narrow is `<100`. Effective state
   never mutates `ResearchPanePreferences`.
3. Add overlay schema v1 with only qualified key, revision, pane preferences,
   preferred companion, and timestamps. Bound the file to 1 MiB, records to
   512, strings to explicit field limits, and reject secret/path/body-shaped
   keys.
4. Read and write through `secure_private_directory` and
   `atomic_private_write_text`. Use optimistic compare-before-replace. Decode
   records independently so one corrupt record is quarantined/exportable and
   canonical workspace loading continues.
5. Add inverse tests: persist effective forced collapse (must fail restoration),
   key a Server overlay by display name (must fail identity isolation), and
   place source content or a token in a v1 preference/unknown field (must fail
   validation).
6. Run:

   ```bash
   .venv/bin/python -m pytest -q Tests/Research_Workspace/test_layout_state.py Tests/Research_Workspace/test_overlay_store.py
   ```

7. Commit:

   ```bash
   git commit -m "feat: add private Research workspace layout state"
   ```

## Task 4: Compose the two Research screens and accessible pane shell

**Files:**

- Create: `tldw_chatbook/UI/Research_Workspace_Modules/__init__.py`
- Create: `tldw_chatbook/UI/Research_Workspace_Modules/mode_bar.py`
- Create: `tldw_chatbook/UI/Research_Workspace_Modules/header_region.py`
- Create: `tldw_chatbook/UI/Research_Workspace_Modules/workspace_menu.py`
- Create: `tldw_chatbook/UI/Research_Workspace_Modules/pane_handle.py`
- Create: `tldw_chatbook/UI/Research_Workspace_Modules/sources_region.py`
- Create: `tldw_chatbook/UI/Research_Workspace_Modules/chat_region.py`
- Create: `tldw_chatbook/UI/Research_Workspace_Modules/studio_region.py`
- Create: `tldw_chatbook/UI/Screens/research_workspace_screen.py`
- Modify: `tldw_chatbook/UI/Screens/research_screen.py`
- Create: `tldw_chatbook/css/features/_research_workspace.tcss`
- Modify generated sheets through: `tldw_chatbook/css/build_css.py`
- Test: `Tests/UI/test_research_mode_strip.py`
- Test: `Tests/UI/test_research_workspace_screen.py`
- Test: `Tests/UI/test_research_workspace_geometry.py`
- Modify: `Tests/UI/test_research_screen.py`
- Modify: `Tests/UI/test_screen_navigation.py`

1. Add mounted RED tests proving both real screens mount the same mode bar and
   navigate with `NavigateToScreen`; neither embeds the other.
2. Implement `ResearchModeStrip` using the `LabModeStrip` precedent with
   Workspace -> `research_workspace` and Runs -> `research`.
3. Compose the Workspace screen once: pinned header, Sources region/handle,
   dominant Chat region, Studio region/handle, and one status row. Foundation
   regions may show honest loading/empty/recovery copy and real owner links,
   but no inert Add/Generate/Send buttons.
4. Add `ResearchPaneHandle` with exact visible labels and full tooltip/access
   names. Collapse moves focus to its reveal arrow. Expansion moves focus to
   the revealed region root/heading. Hidden panes and hidden buttons set
   `display=False` and cannot remain tabbable.
5. On resize, call the pure reducer, patch region/handle display and grid
   classes in place, and preserve semantic focus. If reflow hides focus, move
   to the visible pane-mode button and announce the change.
6. At medium/narrow width mount the approved Sources/Chat/Studio mode strip;
   an explicit medium reveal replaces the companion visibly even when the wide
   preference remains open for both.
7. Add CSS only in `_research_workspace.tcss`, then run:

   ```bash
   .venv/bin/python tldw_chatbook/css/build_css.py
   ```

8. In the geometry harness mount the production screen hierarchy with
   `TldwCli.CSS_PATH`. Test 160x40, 120x30, 100x30, 84x24, 80x24, and 60x20;
   assert rendered frames and compositor containment, not only style values.
9. Run:

   ```bash
   .venv/bin/python -m pytest -q Tests/UI/test_research_mode_strip.py Tests/UI/test_research_workspace_screen.py Tests/UI/test_research_workspace_geometry.py Tests/UI/test_research_screen.py Tests/UI/test_screen_navigation.py
   ```

10. Commit:

   ```bash
   git commit -m "feat: add responsive Research workspace screen"
   ```

## Task 5: Wire app services, user guidance, and closeout evidence

**Files:**

- Modify: `tldw_chatbook/app.py`
- Create: `Docs/User_Guide/research_workspace.md`
- Modify: `backlog/tasks/task-21507 - Add-Research-shell-authority-and-responsive-workspace-foundation.md`
- Test: `Tests/UI/test_research_workspace_app_wiring.py`
- Test: `Tests/UI/test_shell_chrome_contract.py`

1. Add RED app-wiring tests showing adapters, controller, and overlay store are
   late-bound from the active local/server services and unavailable services
   yield recovery rather than constructor crashes.
2. Wire only the foundation services. Do not construct Source, Chat, Studio,
   sharing, or transfer coordinators from future phases.
3. Document the Research destination, F10, Workspace/Runs distinction,
   Local/Server selector, device-only pane preferences, and exact collapse
   controls.
4. Run the focused tests from Tasks 1-4 plus:

   ```bash
   .venv/bin/python -m pytest -q Tests/UI/test_research_workspace_app_wiring.py Tests/UI/test_shell_chrome_contract.py
   .venv/bin/python -m ruff check tldw_chatbook/Research_Workspace tldw_chatbook/UI/Research_Workspace_Modules tldw_chatbook/UI/Screens/research_workspace_screen.py
   git diff --check
   ```

5. Run the configured UI detector on the new module and stylesheet:

   ```bash
   node .agents/skills/impeccable/scripts/detect.mjs tldw_chatbook/UI/Research_Workspace_Modules tldw_chatbook/css/features/_research_workspace.tcss
   ```

   Treat findings as defect evidence and verify them against the rendered TUI;
   a clean detector result is not visual proof.
6. Perform spec, no-placeholder, no-cross-authority, keybinding, focus, and
   rendered-geometry self-review. Update TASK-21507 ACs/notes only with fresh
   evidence and state explicitly that full pytest was not run.
7. Commit:

   ```bash
   git commit -m "docs: complete Research workspace foundation"
   ```

## Required inverse checks

1. Put `research` back under Library or map it to `research_workspace`; the
   preserved Runs-route test must fail.
2. Generate bindings by destination index; the unchanged-shortcut test must
   fail after Research is inserted.
3. Let Server adapter call Local after a server error; the no-cross-call test
   must fail.
4. Persist a forced responsive collapse; width-restoration test must fail.
5. Leave focus in a hidden pane; the mounted focus-cycle test must fail.
6. Render `←`/`→` instead of the exact ASCII labels; arrow contract test must
   fail.

## Focused verification boundary

Run only the files named above, CSS build/parity when CSS changes, Ruff on the
changed Python inventory, and `git diff --check`. No repository-wide test claim
is permitted.
