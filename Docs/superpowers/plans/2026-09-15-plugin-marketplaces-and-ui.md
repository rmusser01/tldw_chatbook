# Plugin adapters marketplaces and UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make approved native/vendor capabilities discoverable and manageable through Git catalogs, selected app imports and a dedicated Plugins destination.

**Architecture:** Normalize package/catalog dialects into the existing plugin coordinator, acquire immutable bounded snapshots and render authority-free UI projections. Integrate all components into existing context/tool services and qualify each compatibility claim with recorded evidence.

**Tech Stack:** Python 3.12+, Textual 8.2.8, Pydantic 2, private SQLite, httpx, portalocker and existing trust/credential primitives.

**Spec:** [Plugin spec](../specs/2026-09-15-managed-plugins-design.md) and [hook spec](../specs/2026-09-15-expanded-hook-runtime-design.md). Read both; this plan covers its assigned subsystem within the complete [delivery plan](2026-09-15-managed-plugins-delivery.md).

## Global Constraints

- Python >=3.12; current checkout pins Textual 8.2.8, Pydantic >=2.4,<3 and portalocker 3.2.0. Preserve these pins; use the existing SQLite/httpx/crypto/keyring seams.
- One installation and selected revision exist per user-data directory.
- Global default activation starts disabled. Importing a catalog does not install or enable its entries.
- All approved hook additions are in scope. Delivery stages are ordering, not deferral.
- No parallel agent/permission runtime, package build/install execution, vendor grants or per-workspace package versions.
- Required constraints never disappear because parsing, configuration, hooks or persistence fail. Native/foreign instructions remain attributed untrusted context.
- Package activation grants no filesystem binding, tool permission, network credential or trusted project status. Console local tools retain scratch/explicit-binding authority.
- Apply current authority before injection/launch/dispatch/result acceptance. Workspace disable preserves other authorized scopes; namespace marker changes alone do not cancel all work.
- Full-suite runs require explicit user opt-in. Every task runs its exact feature/regression files and a successful control on the same production entry.
- Implementation uses an isolated execution worktree and profile; do not repoint a shared editable environment. Verify child interpreter package provenance as well as pytest cwd.
- Every To Do task must move In Progress and receive its Implementation Plan via Backlog CLI before code changes. Add Implementation Notes and mark Done only after its acceptance criteria, review, targeted tests and static checks pass.

ADR required: yes
ADR paths: [ADR-162](../../../backlog/decisions/162-managed-agent-plugins.md); [ADR-163](../../../backlog/decisions/163-expanded-console-hook-runtime.md)
Reason: Implements the accepted storage, trust, runtime and UI contracts. No additional ADR is needed unless implementation changes one of those decisions.

---

## Execution and evidence

This is an implementation plan, not implemented code or passing runtime evidence.
The code blocks below are small invariant/RED-test sketches, not a complete
implementation to paste blindly. Each task must also exercise its named production
entry and failure/control matrix. Preserve the stated interfaces across tasks;
when current library behavior contradicts a sketch, establish the real RED failure
and correct the sketch/test before implementation, as the repository's
[testing-evidence lesson](../../../backlog/docs/lessons-testing-evidence.md) requires.

Read [live verification](../../../backlog/docs/lessons-live-verification.md) before
running the app. Isolate config, data, credential and child-process roots before
importing runtime code, and verify the isolation. Use sys.executable for controlled
children. A disabled path failing from an unrelated event-loop error is not evidence.

Each task is one independently reviewable deliverable. Its checklist is the
sequence of small test/implementation increments; repeat the RED/GREEN cycle for
each listed failure/control case. Do not implement the entire subsystem before
running its first integration test. File roles and public contracts below define
the decomposition; no unrelated broad refactoring is part of these plans.

## File ownership map

| Task | New implementation units | Existing integration boundaries |
| --- | --- | --- |
| I1 | `tldw_chatbook/Plugins/components.py`, `tldw_chatbook/Plugins/commands.py`, `tldw_chatbook/Plugins/agent_presets.py` | `tldw_chatbook/Plugins/context.py`, `tldw_chatbook/Plugins/skill_provider.py`, `tldw_chatbook/Chat/console_skill_resolver.py`, `tldw_chatbook/Chat/console_agent_bridge.py`, `tldw_chatbook/Agents/tool_catalog.py` |
| I2 | `tldw_chatbook/Plugins/adapters/codex.py`, `tldw_chatbook/Plugins/adapters/cursor.py`, `tldw_chatbook/Plugins/adapters/vendor_hooks.py` | `tldw_chatbook/Plugins/inspection.py` |
| I3 | `tldw_chatbook/Plugins/acquisition.py`, `tldw_chatbook/Plugins/git_source.py` | `tldw_chatbook/Plugins/package_files.py` |
| I4 | `tldw_chatbook/Plugins/catalogs.py`, `tldw_chatbook/Plugins/imports.py`, `tldw_chatbook/Plugins/source_service.py` | `tldw_chatbook/Plugins/registry.py`, `tldw_chatbook/Plugins/coordinator.py` |
| I5 | `tldw_chatbook/UI/Screens/plugins_screen.py`, `tldw_chatbook/UI/Plugins_Modules/__init__.py`, `tldw_chatbook/UI/Plugins_Modules/browse_state.py`, `tldw_chatbook/UI/Plugins_Modules/details.py`, `tldw_chatbook/css/features/_plugins.tcss` | `tldw_chatbook/UI/Navigation/screen_registry.py`, `tldw_chatbook/UI/Navigation/main_navigation.py`, `tldw_chatbook/UI/stable_command_palette.py`, `tldw_chatbook/UI/Screens/settings_screen.py`, `tldw_chatbook/css/build_css.py`, `tldw_chatbook/css/tldw_cli_modular.tcss` |
| I6 | `tldw_chatbook/UI/Plugins_Modules/review.py`, `tldw_chatbook/UI/Plugins_Modules/operations.py`, `tldw_chatbook/Widgets/plugin_review_modal.py` | `tldw_chatbook/UI/Screens/plugins_screen.py`, `tldw_chatbook/UI/Library_Modules/library_skills_state.py`, `tldw_chatbook/UI/MCP_Modules/mcp_profile_form.py` |
| I7 | Qualification fixtures/docs | No existing runtime entry changed |

## I1: Register all native plugin capability types

**Backlog:** [TASK-32686](../../../backlog/tasks/task-32686%20-%20Register-all-native-plugin-capability-types.md). **Requires:** [TASK-32672](../../../backlog/tasks/task-32672%20-%20Admit-native-plugin-skills-through-existing-Console-authority.md), [TASK-32685](../../../backlog/tasks/task-32685%20-%20Invoke-MCP-backed-hooks-through-normal-tool-authority.md), [TASK-32684](../../../backlog/tasks/task-32684%20-%20Expose-owned-plugin-MCP-tools-with-scoped-connection-leases.md).

**Deliverable:** Expose commands, rules, agents, hooks and complete skill metadata consistently through Chatbook runtime services.

**Files:**

- Create: `tldw_chatbook/Plugins/components.py`
- Create: `tldw_chatbook/Plugins/commands.py`
- Create: `tldw_chatbook/Plugins/agent_presets.py`
- Modify: `tldw_chatbook/Plugins/context.py`
- Modify: `tldw_chatbook/Plugins/skill_provider.py`
- Modify: `tldw_chatbook/Chat/console_skill_resolver.py`
- Modify: `tldw_chatbook/Chat/console_agent_bridge.py`
- Modify: `tldw_chatbook/Agents/tool_catalog.py`
- Test: `Tests/Plugins/test_native_components.py`
- Test: `Tests/Plugins/test_native_skill_flow.py`
- Test: `Tests/Chat/test_console_skill_substitution.py`

**Interfaces**

- Consumes: F5 owned skill/context admission, M4 plugin tools and H6 complete hook runtime.
- Produces: PluginComponents.project(snapshot: RunPluginSnapshot) -> tuple[ComponentRecord, ...]. render_command(component: ComponentRecord, arguments: dict[str, str]) -> tuple[dict, ...] returns separately attributed instruction and argument blocks. resolve_agent_tools(value: str | tuple[str, ...], eligible: frozenset[str]) -> frozenset[str] preserves inherit versus empty. register_hook_definitions(snapshot: RunPluginSnapshot) -> tuple[HookHandler, ...] supplies owned definitions and current-authority validation.

- [x] **1. Create the first behavioral test and its local fixtures.** Use the fixture contract in the implementation increments below; initial import/behavior must fail for the missing feature.

```python
def test_empty_agent_tool_list_never_inherits():
    from tldw_chatbook.Plugins.agent_presets import resolve_agent_tools
    eligible = frozenset({"mcp:repository:read"})
    assert resolve_agent_tools((), eligible) == frozenset()
    assert resolve_agent_tools("inherit", eligible) == eligible
```

- [x] **2. Establish RED through the intended entry.** Run `python -m pytest Tests/Plugins/test_native_components.py -q`. Expected: the named new behavior fails, while any positive precondition/control succeeds. Resolve test harness/API errors before changing production code.

- [x] **3. Implement the smallest invariant, then integrate the real owner.** This kernel states the ordering/data rule; the following increments supply the complete behavior and limits.

```python
def resolve_agent_tools(value, eligible: frozenset[str]) -> frozenset[str]:
    if value == "inherit":
        return eligible
    requested = frozenset(value)
    if not requested <= eligible:
        raise ValueError("plugin_agent_tools_unavailable")
    return requested
```

  - [x] 3.1. Normalize command/rule/agent definitions already inventoried by F1 into existing composer/context/AgentDefinition seams. Give aliases stable installation identity; no display-name tool lookup or recursive command expansion.
  - [x] 3.2. Keep rule order stable by installation/component ID, support only always/manual activation and require explicit manual adaptation for file/model-selected modes. Reject oversized required blocks whole.
  - [x] 3.3. Preserve skill manual-only, inline/fork, explicit tool restrictions and model mapping semantics through final provider input; plugin prose never enters an internal system-authority lane.
  - [x] 3.4. Supply owned v2 hook sets and dependency requirements at the next run snapshot; report partial readiness per component and never auto-select newly supported items. Bind the H2 HookProcessOwner port to F2 before any plugin command launch, preserving pending-launch provenance and root usage.

**Failure and successful-control matrix:** Empty/inherit/unmapped tools, mapped/unknown model, arguments containing command syntax, namespace collisions, forked context authority, whole-block overflow, mid-run enablement, disabled required hook and successful unrelated component.

- [x] **4. Establish GREEN and preserve the neighboring path.** Run the exact files below. Expected: all execute and pass, with no silent coroutine/platform skips used as qualification. Inspect real resources/output, not source-string matches.

```bash
python -m pytest Tests/Plugins/test_native_components.py Tests/Plugins/test_native_skill_flow.py Tests/Chat/test_console_skill_substitution.py -q
```

- [x] **5. Review, record evidence and commit the task.** Update its ACs/notes and the relevant authoring/operation documentation; record platform limits. Run `git diff --check`, Python syntax checks and the verification-environment formatter/linter on the changed Python files as specified in the delivery plan. Stage the exact task-owned files, including any test fixtures and generated CSS, and commit; never stage unrelated work.

```bash
backlog task task-32686 --plain
git diff --check
```

## I2: Qualify Cursor and Codex package and hook adapters

**Backlog:** [TASK-32687](../../../backlog/tasks/task-32687%20-%20Qualify-Cursor-and-Codex-package-and-hook-adapters.md). **Requires:** [TASK-32686](../../../backlog/tasks/task-32686%20-%20Register-all-native-plugin-capability-types.md).

**Deliverable:** Make supported foreign packages useful without claiming runtime equivalence for unsupported semantics.

**Files:**

- Create: `tldw_chatbook/Plugins/adapters/codex.py`
- Create: `tldw_chatbook/Plugins/adapters/cursor.py`
- Create: `tldw_chatbook/Plugins/adapters/vendor_hooks.py`
- Create: `Tests/Plugins/fixtures/interop/README.md`
- Modify: `tldw_chatbook/Plugins/inspection.py`
- Test: `Tests/Plugins/test_codex_adapter.py`
- Test: `Tests/Plugins/test_cursor_adapter.py`
- Test: `Tests/Plugins/test_vendor_hook_adapters.py`

**Interfaces**

- Consumes: F1 normalized inventory and I1/H6 capability contracts. Read the linked primary packaging/hook references from both approved specs; adapter versions and fixture upstream commits are recorded at implementation.
- Produces: inspect_codex(root: Path, catalog_overlay: dict | None) -> PackageInspection; inspect_cursor(root: Path, catalog_overlay: dict | None) -> PackageInspection; normalize_vendor_hook(value: dict, dialect_version: str) -> dict (non-executable qualification proposal). overlay_for_openai(inline: dict | None, compatibility: dict | None) -> dict selects wholesale, never merges. Unsupported adaptations carry reason and affected dependency IDs.

- [x] **1. Create the first behavioral test and its local fixtures.** Use the fixture contract in the implementation increments below; initial import/behavior must fail for the missing feature.

```python
def test_inline_overlay_replaces_compatibility_wholesale():
    from tldw_chatbook.Plugins.adapters.codex import overlay_for_openai
    assert overlay_for_openai({"skills": ["new"]}, {"skills": ["old"], "hooks": "legacy"}) == {"skills": ["new"]}
    assert overlay_for_openai(None, {"skills": ["old"]}) == {"skills": ["old"]}
```

- [x] **2. Establish RED through the intended entry.** Run `python -m pytest Tests/Plugins/test_codex_adapter.py -q`. Expected: the named new behavior fails, while any positive precondition/control succeeds. Resolve test harness/API errors before changing production code.

- [x] **3. Implement the smallest invariant, then integrate the real owner.** This kernel states the ordering/data rule; the following increments supply the complete behavior and limits.

```python
def overlay_for_openai(inline: dict | None, compatibility: dict | None) -> dict:
    return dict(inline if inline is not None else (compatibility or {}))
```

  - [x] 3.1. Pin independently curated fixtures with upstream commit, original URL, license and expected inventory; use controlled runnable examples separately from parser fixtures. Native Chatbook extension constraints still apply when portable inventory survives malformed input.
  - [x] 3.2. Implement inline OpenAI precedence, standalone vendor candidates, root portable identity/location rules and Cursor explicit-path replacement. Catalog executable overlays participate in effective digest and review.
  - [x] 3.3. Normalize supported variable and metadata subsets explicitly. Conflicting manual-only declarations, unavailable rules/models/constraints and unknown source behaviors remain invalid/unsupported rather than inherited defaults.
  - [x] 3.4. Qualify each hook mapping across timing, matcher, payload, result, cwd, variable expansion and limits. Parse only tested simple platform-specific argv forms; reject shell operators/substitutions and unsupported regex semantics. Unknown guard scope blocks executable/automatic components until reviewed narrowing.

**Failure and successful-control matrix:** Portable plus both vendor manifests, invalid root schema, empty explicit component paths, inline overlay removing compatibility fields, foreign allow bypass, guard versus transformer, source success-only post-event, file rule manual adaptation and parser/behavior evidence separation.

- [x] **4. Establish GREEN and preserve the neighboring path.** Run the exact files below. Expected: all execute and pass, with no silent coroutine/platform skips used as qualification. Inspect real resources/output, not source-string matches.

```bash
python -m pytest Tests/Plugins/test_codex_adapter.py Tests/Plugins/test_cursor_adapter.py Tests/Plugins/test_vendor_hook_adapters.py -q
```

- [x] **5. Review, record evidence and commit the task.** Update its ACs/notes and the relevant authoring/operation documentation; record platform limits. Run `git diff --check`, Python syntax checks and the verification-environment formatter/linter on the changed Python files as specified in the delivery plan. Stage the exact task-owned files, including any test fixtures and generated CSS, and commit; never stage unrelated work.

```bash
backlog task task-32687 --plain
git diff --check
```

Implementation adjustment: the prototype HookHandler return would falsely imply
runtime qualification. All foreign hook contracts remain unsupported proposals:
Chatbook cannot yet supply complete vendor payload/cwd/permission/transcript,
output, timing and timeout semantics together. Required or unknown guard scope
fences the affected package; known optional observers remain visible. No source
hook is handed to the native scheduler. This is the accepted conservative subset,
not an original-host compatibility claim.

Evidence and exact limits: [I2 integration record](../reviews/2026-09-30-expanded-hooks-integration.md).
The chosen dialect now reaches ordinary materialization; retained executable
catalog fields use the existing immutable snapshot/review/recovery owner. No
acquisition, marketplace or UI work from I3+ is included.

## I3: Acquire bounded Git and local plugin snapshots

**Backlog:** [TASK-32688](../../../backlog/tasks/task-32688%20-%20Acquire-bounded-Git-and-local-plugin-snapshots.md). **Requires:** [TASK-32668](../../../backlog/tasks/task-32668%20-%20Inspect-immutable-native-plugin-packages.md), [TASK-32674](../../../backlog/tasks/task-32674%20-%20Drain-plugin-work-before-applying-updates-and-rollback.md).

**Deliverable:** Fetch inspectable immutable package content from user-selected Git hosts without running repository-controlled checkout behavior.

**Files:**

- Create: `tldw_chatbook/Plugins/acquisition.py`
- Create: `tldw_chatbook/Plugins/git_source.py`
- Modify: `tldw_chatbook/Plugins/package_files.py`
- Test: `Tests/Plugins/test_git_acquisition.py`
- Test: `Tests/Plugins/test_source_locators.py`
- Test: `Tests/Plugins/test_acquisition_limits.py`

**Interfaces**

- Consumes: F1 bounded materializer and F7 quota/retention; installed Git executable with argv invocation and existing host credential helper/SSH agent authority.
- Produces: SourceLocator is closed with transport, origin, repository, ref and subdirectory; parse_source_locator(value: str) -> SourceLocator. async acquire_snapshot(source: SourceLocator, operation_id: str, destination: Path) -> AcquiredSnapshot returns immutable commit/tree provenance, bounded content and acquisition diagnostics. AcquiredSnapshot is a frozen model in Plugins/models.py; it does not imply trust or activation.

- [ ] **1. Create the first behavioral test and its local fixtures.** Use the fixture contract in the implementation increments below; initial import/behavior must fail for the missing feature.

```python
def test_git_ext_transport_is_rejected_before_spawn():
    import pytest
    from tldw_chatbook.Plugins.git_source import parse_source_locator
    with pytest.raises(ValueError):
        parse_source_locator("ext::sh -c whoami")
    assert parse_source_locator("https://example.test/team/plugins.git").transport == "https"
```

- [ ] **2. Establish RED through the intended entry.** Run `python -m pytest Tests/Plugins/test_git_acquisition.py -q`. Expected: the named new behavior fails, while any positive precondition/control succeeds. Resolve test harness/API errors before changing production code.

- [ ] **3. Implement the smallest invariant, then integrate the real owner.** This kernel states the ordering/data rule; the following increments supply the complete behavior and limits.

```python
ALLOWED_GIT_TRANSPORTS = frozenset({"https", "ssh", "local"})

def require_transport(transport: str) -> None:
    if transport not in ALLOWED_GIT_TRANSPORTS:
        raise ValueError("plugin_source_transport_unsupported")
```

  - [ ] 3.1. Parse GitHub shorthand and explicit origin/ref/subdirectory independently of display metadata. A selected local source is distinct from a remote catalog string that names a host file.
  - [ ] 3.2. Fetch without worktree checkout, recursive submodules or LFS; read the selected tree with Git plumbing and materialize validated regular files through F1. Disable repository hooks/filters and package-selected helpers/SSH commands while preserving explicitly configured host authentication.
  - [ ] 3.3. Apply two-acquisition, 120-second, 500 MiB staging, 100 MiB expanded package and file/depth limits during fetch/materialization. Own the Git process, quota monitor and staging cleanup through cancellation.
  - [ ] 3.4. Use temporary controlled Git repositories and safe fake authentication endpoints for origin/ref changes. Pin commit before review; moving a branch later must not retarget the reviewed snapshot.

**Failure and successful-control matrix:** Hostile config, ext protocol, escaping link, internal regular-file link, directory link, normalized duplicate path, submodule-required material, missing Git, full disk, credential sentinel, redirect to unrelated origin and responsive cancellation under bounded worst-case input.

- [ ] **4. Establish GREEN and preserve the neighboring path.** Run the exact files below. Expected: all execute and pass, with no silent coroutine/platform skips used as qualification. Inspect real resources/output, not source-string matches.

```bash
python -m pytest Tests/Plugins/test_git_acquisition.py Tests/Plugins/test_source_locators.py Tests/Plugins/test_acquisition_limits.py -q
```

- [ ] **5. Review, record evidence and commit the task.** Update its ACs/notes and the relevant authoring/operation documentation; record platform limits. Run `git diff --check`, Python syntax checks and the verification-environment formatter/linter on the changed Python files as specified in the delivery plan. Stage the exact task-owned files, including any test fixtures and generated CSS, and commit; never stage unrelated work.

```bash
backlog task task-32688 --plain
git diff --check
```

## I4: Browse Git catalogs and import selected marketplace sources

**Backlog:** [TASK-32689](../../../backlog/tasks/task-32689%20-%20Browse-Git-catalogs-and-import-selected-marketplace-sources.md). **Requires:** [TASK-32687](../../../backlog/tasks/task-32687%20-%20Qualify-Cursor-and-Codex-package-and-hook-adapters.md), [TASK-32688](../../../backlog/tasks/task-32688%20-%20Acquire-bounded-Git-and-local-plugin-snapshots.md).

**Deliverable:** Let users discover packages and copy selected existing app sources while keeping discovery separate from installation and authority.

**Files:**

- Create: `tldw_chatbook/Plugins/catalogs.py`
- Create: `tldw_chatbook/Plugins/imports.py`
- Create: `tldw_chatbook/Plugins/source_service.py`
- Modify: `tldw_chatbook/Plugins/registry.py`
- Modify: `tldw_chatbook/Plugins/coordinator.py`
- Test: `Tests/Plugins/test_catalogs.py`
- Test: `Tests/Plugins/test_app_imports.py`
- Test: `Tests/Plugins/test_source_service.py`

**Interfaces**

- Consumes: I2 dialect/overlay inventory and I3 immutable acquisition. Supported paths are .agents/plugins/marketplace.json, .cursor-plugin/marketplace.json and .claude-plugin/marketplace.json.
- Produces: SourceService.refresh(source_id: str) -> Awaitable[CatalogSnapshot]; search(query: str, *, offset: int, limit: int = 50) -> tuple[CatalogEntry, ...]; check_updates(installation_id: str) -> Awaitable[tuple[PackageInspection, ...]]; remove(source_id: str) -> Awaitable[None]. inspect_app_import(app: str, selected_root: Path) -> ImportPreview returns allowlisted source/package references and exclusions; commit_import(preview: ImportPreview) records only the reviewed copy. Frozen CatalogSnapshot fields are source_id, catalog_commit, digest, fetched_at, entries and refresh_error; CatalogEntry contains source-qualified identity, package locator, overlays, support and diagnostics. ImportPreview binds the selected root identity, candidate references, exclusions and reviewed digest.

- [ ] **1. Create the first behavioral test and its local fixtures.** Use the fixture contract in the implementation increments below; initial import/behavior must fail for the missing feature.

```python
def test_catalog_local_entry_resolves_against_catalog_root(tmp_path):
    from tldw_chatbook.Plugins.catalogs import catalog_local_path
    root = tmp_path / "repo"
    root.mkdir()
    assert catalog_local_path(root, "./plugins/review") == root / "plugins" / "review"
```

- [ ] **2. Establish RED through the intended entry.** Run `python -m pytest Tests/Plugins/test_catalogs.py -q`. Expected: the named new behavior fails, while any positive precondition/control succeeds. Resolve test harness/API errors before changing production code.

- [ ] **3. Implement the smallest invariant, then integrate the real owner.** This kernel states the ordering/data rule; the following increments supply the complete behavior and limits.

```python
def catalog_local_path(root, relative: str):
    from pathlib import Path
    candidate = (root / relative).resolve()
    if Path(relative).is_absolute() or not candidate.is_relative_to(root.resolve()):
        raise ValueError("catalog_path_outside_root")
    return candidate
```

  - [ ] 3.1. Normalize the three catalog shapes and explicit selection among multiple catalogs. Pin package and executable catalog overlays together; reject conflicting ref/SHA selectors and show unsupported npm/unknown entries without dropping valid siblings.
  - [ ] 3.2. Implement cached source-qualified search with 50-row pages and 5 MiB/10,000-entry/50-source limits. Refresh changes discovery only; preserve last successful cache and a bounded failure status.
  - [ ] 3.3. Import only explicitly selected Cursor/Codex paths via allowlisted config reads; copy managed snapshots/source locators and preview exclusions. Never import credentials, trust, approval state, enablement or partial vendor runtime cache overlays.
  - [ ] 3.4. Remove source records without uninstalling their packages. Keep provenance and selected update locators so source refresh, check updates and reviewed update remain distinct operations.

**Failure and successful-control matrix:** Catalog root versus metadata directory, private/local origin authority, duplicate source-qualified names, unknown entry type, stale cache, malformed sibling, source deletion with installed packages, modified foreign cache and no automatic connection/authentication on import.

- [ ] **4. Establish GREEN and preserve the neighboring path.** Run the exact files below. Expected: all execute and pass, with no silent coroutine/platform skips used as qualification. Inspect real resources/output, not source-string matches.

```bash
python -m pytest Tests/Plugins/test_catalogs.py Tests/Plugins/test_app_imports.py Tests/Plugins/test_source_service.py -q
```

- [ ] **5. Review, record evidence and commit the task.** Update its ACs/notes and the relevant authoring/operation documentation; record platform limits. Run `git diff --check`, Python syntax checks and the verification-environment formatter/linter on the changed Python files as specified in the delivery plan. Stage the exact task-owned files, including any test fixtures and generated CSS, and commit; never stage unrelated work.

```bash
backlog task task-32689 --plain
git diff --check
```

## I5: Add Plugins navigation and source-qualified browsing

**Backlog:** [TASK-32693](../../../backlog/tasks/task-32693%20-%20Add-Plugins-navigation-and-source-qualified-browsing.md). **Requires:** [TASK-32689](../../../backlog/tasks/task-32689%20-%20Browse-Git-catalogs-and-import-selected-marketplace-sources.md).

**Deliverable:** Give users an accessible plugin destination with stable browsing state and understandable partial readiness.

**Files:**

- Create: `tldw_chatbook/UI/Screens/plugins_screen.py`
- Create: `tldw_chatbook/UI/Plugins_Modules/__init__.py`
- Create: `tldw_chatbook/UI/Plugins_Modules/browse_state.py`
- Create: `tldw_chatbook/UI/Plugins_Modules/details.py`
- Create: `tldw_chatbook/css/features/_plugins.tcss`
- Modify: `tldw_chatbook/UI/Navigation/screen_registry.py`
- Shared metadata: `tldw_chatbook/UI/Navigation/shell_destinations.py` and `UI/Navigation/screen_registry.py` (R13/R97); generic navigation/palette consumers remain unchanged.
- Modify: `tldw_chatbook/UI/Screens/settings_screen.py`
- Modify: `tldw_chatbook/css/build_css.py`
- Modify: `tldw_chatbook/css/tldw_cli_modular.tcss`
- Test: `Tests/UI/test_plugins_navigation.py`
- Test: `Tests/UI/test_plugins_browse.py`
- Test: `Tests/UI/test_command_palette_shell_routes.py`
- Test: `Tests/UI/test_workbench_route_inventory.py`
- Test: `Tests/UI/test_design_token_governance.py`

**Interfaces**

- Consumes: I4 SourceService and the existing ScreenRoute lazy-navigation/command-palette surface. Read backlog/docs/design-language.md before changing any UI.
- Produces: PluginsScreen(app) uses screen route plugins. PluginBrowseState stores view, query, page, selected_source_id, selected_installation_id and request_generation. accept_response(request_generation: int, current_generation: int) -> bool ignores stale responses. The one lazy app-owned SourceService wraps PluginService; mutations use public facades; projected UI state is never authority.

- [ ] **1. Create the first behavioral test and its local fixtures.** Use the fixture contract in the implementation increments below; initial import/behavior must fail for the missing feature.

```python
def test_stale_search_cannot_retarget_selection():
    from tldw_chatbook.UI.Plugins_Modules.browse_state import accept_response
    assert not accept_response(3, 4)
    assert accept_response(4, 4)
```

- [ ] **2. Establish RED through the intended entry.** Run `python -m pytest Tests/UI/test_plugins_navigation.py -q`. Expected: the named new behavior fails, while any positive precondition/control succeeds. Resolve test harness/API errors before changing production code.

- [ ] **3. Implement the smallest invariant, then integrate the real owner.** This kernel states the ordering/data rule; the following increments supply the complete behavior and limits.

```python
def accept_response(request_generation: int, current_generation: int) -> bool:
    return request_generation == current_generation
```

  - [ ] 3.1. Register a lazy Plugins destination, visible navigation and palette entries, plus canonical Settings link; derive shortcut labels from existing route ownership. Use workers with exclusive ownership for I/O over 100 ms and no startup catalog scan.
  - [ ] 3.2. Build Installed/Browse/Marketplaces from standard token-backed lists, detail panes, status areas and buttons. Wide view uses list/details; narrow view uses full-width detail with Back, preserving query/page/stable selection and focus.
  - [ ] 3.3. Show workspace activation separately from device installation, partial component readiness, support/selection/evidence and provenance. Provide visible install-from-repository/import/add-source empty-state actions.
  - [ ] 3.4. Sanitize bounded untrusted readmes and labels, omit automatic remote images, and reject stale async responses before selection/application. Edit source CSS modules and rebuild the generated bundle; do not hand-edit it.

**Failure and successful-control matrix:** Mounted route/palette entry, 80x24 and 120x35 keyboard flow, narrow Back, row reorder while response pending, changed workspace, no-sources/no-matches/failure/stale cache, malicious control sequences and readable disabled/focus states. Live isolated-profile evidence accompanies mounted assertions.

- [ ] **4. Establish GREEN and preserve the neighboring path.** Run the exact files below. Expected: all execute and pass, with no silent coroutine/platform skips used as qualification. Inspect real resources/output, not source-string matches.

```bash
python -m pytest Tests/UI/test_plugins_navigation.py Tests/UI/test_plugins_browse.py Tests/UI/test_command_palette_shell_routes.py Tests/UI/test_workbench_route_inventory.py Tests/UI/test_design_token_governance.py -q
```

- [ ] **5. Review, record evidence and commit the task.** Update its ACs/notes and the relevant authoring/operation documentation; record platform limits. Run `git diff --check`, Python syntax checks and the verification-environment formatter/linter on the changed Python files as specified in the delivery plan. Stage the exact task-owned files, including any test fixtures and generated CSS, and commit; never stage unrelated work.

```bash
backlog task task-32693 --plain
git diff --check
```

### I5 implemented public seam and I6 handoff

Plugins is appended as F8; Settings retains F4 and exposes a Plugins link.
`PluginService.management_snapshot(workspace_id, *, query="", offset=0, limit=50)`
returns immutable display metadata with activation, trust, selection, support,
evidence and current component eligibility from the existing admission owner.
`management_readme(installation_id, revision_digest, *, workspace_id)` reads only
an explicitly selected current retained revision and returns a bounded, marked
64 KiB UTF-8 preview. Neither display API grants a token, lease or run admission.
ADR: [162](../../../backlog/decisions/162-managed-agent-plugins.md).

Repository acquisition, catalog-entry preview and explicit Cursor/Codex import
inspection retain actual snapshot/review/preview objects in `PluginBrowseState`.
I6 supplies exact review/commit and operation-management panels. Its concrete
service/test extension is protected terminal candidate reference enumeration
and reviewed reference discard in `Plugins/source_service.py` and
`Tests/Plugins/test_source_service.py`. Compact persisted references cannot be
inflated into fabricated full `AcquiredSnapshot` objects; discard must check the
full protected dependency union and boot-qualified custody. Missing in-session
objects or empty maps never prove abandonment; pending/reboot custody is not
adopted. This is management increment scope, not an I5 discard implementation.

## I6: Review and manage plugin operations across UI surfaces

**Backlog:** [TASK-32694](../../../backlog/tasks/task-32694%20-%20Review-and-manage-plugin-operations-across-UI-surfaces.md). **Requires:** [TASK-32693](../../../backlog/tasks/task-32693%20-%20Add-Plugins-navigation-and-source-qualified-browsing.md), [TASK-32675](../../../backlog/tasks/task-32675%20-%20Delete-plugin-data-only-after-exact-root-users-drain.md).

**Deliverable:** Complete install/configure/update/remove workflows with review-bound actions and honest recovery status.

**Files:**

- Create: `tldw_chatbook/UI/Plugins_Modules/review.py`
- Create: `tldw_chatbook/UI/Plugins_Modules/operations.py`
- Create: `tldw_chatbook/Widgets/plugin_review_modal.py`
- Modify: `tldw_chatbook/UI/Screens/plugins_screen.py`
- Modify: `tldw_chatbook/UI/Library_Modules/library_skills_state.py`
- Modify: `tldw_chatbook/UI/MCP_Modules/mcp_profile_form.py`
- Test: `Tests/UI/test_plugin_review.py`
- Test: `Tests/UI/test_plugin_operations.py`
- Test: `Tests/Library/test_library_skills_state.py`
- Test: `Tests/MCP/test_control_plane_lifecycle.py`

**Interfaces**

- Consumes: F4 exact PluginReview, F6 independent stop/persistence outcomes, F7 update drain, F8 root deletion and I5 navigation state.
- Produces: PluginReviewModal(review: PluginReview) returns explicit Install disabled or Install and enable in the captured workspace. PluginOperationViewModel(receipt: OperationReceipt, runtime_status: dict) preserves installation/component/workspace identity and distinguishes submitted phase, durable result and cleanup. operation_label(*, committed: bool, stopped: bool, cleanup_pending: bool) -> str is a display projection, never a state mutation.

- [ ] **1. Create the first behavioral test and its local fixtures.** Use the fixture contract in the implementation increments below; initial import/behavior must fail for the missing feature.

```python
def test_persistence_failure_is_not_a_durable_disable():
    from tldw_chatbook.UI.Plugins_Modules.operations import operation_label
    assert "not saved" in operation_label(committed=False, stopped=True, cleanup_pending=False).lower()
    assert "pending" in operation_label(committed=True, stopped=False, cleanup_pending=True).lower()
```

- [ ] **2. Establish RED through the intended entry.** Run `python -m pytest Tests/UI/test_plugin_review.py -q`. Expected: the named new behavior fails, while any positive precondition/control succeeds. Resolve test harness/API errors before changing production code.

- [ ] **3. Implement the smallest invariant, then integrate the real owner.** This kernel states the ordering/data rule; the following increments supply the complete behavior and limits.

```python
def operation_label(*, committed: bool, stopped: bool, cleanup_pending: bool) -> str:
    if not committed:
        return "Disable not saved (blocked for this session)"
    if cleanup_pending or not stopped:
        return "Disabled - cleanup pending"
    return "Disabled"
```

  - [ ] 3.1. Implement review/configuration panels bound to exact immutable identity and current authority. Show executable privileges, adaptations, dependencies, endpoints, secret-reference readiness and affected workspaces before mutation.
  - [ ] 3.2. Wire install/update/rollback/disable/uninstall/delete-data through coordinator operations. Surface wait/cancel/explicit affected-work cancellation separately, and keep runtime stopping, durable receipt, deletion pending and Recovery required visible.
  - [ ] 3.3. Preserve in-session drafts and focus on Close/Back; explicit Discard removes them. Revalidate on return, reject deleted/changed workspaces and retain submitted operations across navigation without retargeting.
  - [ ] 3.4. Add ownership labels and round-trip links from Library Skills and MCP; configuration save remains non-executing and explicit connection/hook tests use normal review. Use mounted event/keyboard tests and live isolated profile rather than direct action-only coverage.

**Failure and successful-control matrix:** Pending review plus source refresh, reordered rows, nested modal dismissal, secret draft handling, cross-workspace impact, session-only failure, surviving writer, unknown remote cancellation, data deletion after partial failure and independent component mutation controls.

- [ ] **4. Establish GREEN and preserve the neighboring path.** Run the exact files below. Expected: all execute and pass, with no silent coroutine/platform skips used as qualification. Inspect real resources/output, not source-string matches.

```bash
python -m pytest Tests/UI/test_plugin_review.py Tests/UI/test_plugin_operations.py Tests/Library/test_library_skills_state.py Tests/MCP/test_control_plane_lifecycle.py -q
```

- [ ] **5. Review, record evidence and commit the task.** Update its ACs/notes and the relevant authoring/operation documentation; record platform limits. Run `git diff --check`, Python syntax checks and the verification-environment formatter/linter on the changed Python files as specified in the delivery plan. Stage the exact task-owned files, including any test fixtures and generated CSS, and commit; never stage unrelated work.

```bash
backlog task task-32694 --plain
git diff --check
```

## I7: Qualify plugin lifecycle interoperability and authoring examples

**Backlog:** [TASK-32695](../../../backlog/tasks/task-32695%20-%20Qualify-plugin-lifecycle-interoperability-and-authoring-examples.md). **Requires:** [TASK-32694](../../../backlog/tasks/task-32694%20-%20Review-and-manage-plugin-operations-across-UI-surfaces.md).

**Deliverable:** Publish evidence-backed compatibility and authoring guidance after each production path has been exercised under supported platform constraints.

**Files:**

- Create: `Tests/Plugins/test_plugin_end_to_end.py`
- Create: `Tests/Plugins/fixtures/native_review/plugin.json`
- Create: `Tests/Plugins/fixtures/native_review/skills/review/SKILL.md`
- Create: `Docs/Plugins/authoring.md`
- Create: `Docs/Plugins/compatibility.md`
- Create: `Docs/Plugins/operations.md`
- Create: `.github/workflows/plugin-qualification.yml`
- Modify: `pyproject.toml`
- Test: `Tests/Plugins/test_plugin_end_to_end.py`
- Test: `Tests/Plugins/test_recovery.py`
- Test: `Tests/Plugins/test_data_cleanup.py`
- Test: `Tests/Agents/test_hooks_v2_mcp_execution.py`
- Test: `Tests/Performance/test_app_startup_performance.py`
- Test: `Tests/DB/test_private_sqlite_inventory.py`

**Interfaces**

- Consumes: Every preceding task and production entry. Compatibility evidence describes the exact checked revision/adapter/platform/configuration; no parser-only blanket claim.
- Produces: A reproducible controlled example and evidence matrix covering native components, selected vendor adaptations, all hook events, both transports and lifecycle failure states. Qualification workflow runs only explicit feature/regression paths on Linux/macOS/Windows; unknown/skipped combinations remain unqualified in Docs/Plugins/compatibility.md.

- [ ] **1. Create the first behavioral test and its local fixtures.** Use the fixture contract in the implementation increments below; initial import/behavior must fail for the missing feature.

```python
import pytest

@pytest.mark.asyncio
async def test_install_use_disable_uses_production_paths(end_to_end_case):
    case = end_to_end_case
    await case.install_and_enable()
    assert await case.invoke_review_skill() == "review-complete"
    await case.disable_here()
    assert not case.skill_advertised()
    assert case.no_owned_live_children()
```

- [ ] **2. Establish RED through the intended entry.** Run `python -m pytest Tests/Plugins/test_plugin_end_to_end.py -q`. Expected: the named new behavior fails, while any positive precondition/control succeeds. Resolve test harness/API errors before changing production code.

- [ ] **3. Implement the smallest invariant, then integrate the real owner.** This kernel states the ordering/data rule; the following increments supply the complete behavior and limits.

```python
EVIDENCE_LEVELS = ("parsed", "connection_exercised", "behavior_exercised", "original_host_compared")

def evidence_valid(record: dict, current: dict) -> bool:
    keys = ("revision", "adapter", "platform", "configuration")
    return record["level"] in EVIDENCE_LEVELS and all(record[key] == current[key] for key in keys)
```

  - [ ] 3.1. Create end_to_end_case with isolated config/data/credential/process roots, a controlled Git catalog and local stdio/HTTP fixtures. Its methods invoke the real install/coordinator/Console paths and wait for confirmed cleanup; prove child Python imports this checkout.
  - [ ] 3.2. Run integrated scenarios for local and Git installs, catalogs/imports, configure/test, owned tool use, update/rollback, scoped/global disable, uninstall and data deletion. Include every amendment scenario from both specs; do not postpone integration tests from earlier tasks to this final gate.
  - [ ] 3.3. Add a targeted three-platform qualification matrix using sys.executable for children and explicit test paths; include startup responsiveness, private DB inventory, bounded input and real kill/reap/crash-recovery evidence. Record missing prerequisites as unqualified, not passing skips.
  - [ ] 3.4. Publish executable authoring examples and a per-component/dialect evidence table with known adaptations, unsupported authentication and process-containment limits. Document recovery and partial readiness with product wording, update package data only for shipped examples, and retain license/provenance.

**Failure and successful-control matrix:** All twelve main-spec acceptance groups and all hook events. Include legacy standalone Skills/MCP, v1 hooks, historical registry/profile migrations, empty/default configs, privacy sentinels, no startup fetch, optional dependencies absent and exact source-bound authority. Full-suite execution still requires user opt-in.

- [ ] **4. Establish GREEN and preserve the neighboring path.** Run the exact files below. Expected: all execute and pass, with no silent coroutine/platform skips used as qualification. Inspect real resources/output, not source-string matches.

```bash
python -m pytest Tests/Plugins/test_plugin_end_to_end.py Tests/Plugins/test_recovery.py Tests/Plugins/test_data_cleanup.py Tests/Agents/test_hooks_v2_mcp_execution.py Tests/Performance/test_app_startup_performance.py Tests/DB/test_private_sqlite_inventory.py -q
```

- [ ] **5. Review, record evidence and commit the task.** Update its ACs/notes and the relevant authoring/operation documentation; record platform limits. Run `git diff --check`, Python syntax checks and the verification-environment formatter/linter on the changed Python files as specified in the delivery plan. Stage the exact task-owned files, including any test fixtures and generated CSS, and commit; never stage unrelated work.

```bash
backlog task task-32695 --plain
git diff --check
```

## Shared package limits

These exact approved limits apply wherever this plan handles the corresponding resource.

| Resource | V1 limit | Exhaustion behavior |
| --- | --- | --- |
| Manifest or hooks definition JSON | 256 KiB each; depth 32 | Reject that document with a bounded diagnostic. |
| Catalog snapshot | 5 MiB; 10,000 entries; 50 sources | Reject oversized refresh; retain previous catalog. |
| Package snapshot | 100 MiB expanded; 10,000 files; 10 MiB/file; path depth 32 | Abort materialization; preserve current installation. |
| Normalized component inventory | 512/package | Reject inspection; do not drop arbitrary tail components. |
| Concurrent acquisitions | 2; 120 s overall per acquisition | Queue visibly or cancel with timeout; no plugin execution. |
| Git staging including object data | 500 MiB/operation | Terminate fetch at quota check; discard staging. This is host acquisition control, not an OS disk quota. |
| Managed package/cache storage | 2 GiB total; require estimated new bytes plus 100 MiB free reserve | Prune eligible cache or refuse before commit. |
| Inactive revisions | 2 most recent per installation; 30-day age target | Prune only unleased/unreferenced revisions; current and recovery records are protected. |
| Abandoned staging | 24 hours | Remove only after journal reconciliation and ownership checks. |
| Plugin instruction blocks | 8 KiB/block, 32 KiB combined per send, also bounded by remaining model context | Reject oversized selected material whole; explain affected components. |
| Listing page | 50 rows | Paginate; search remains over cached metadata. |
| Display metadata | 256 characters/name; 2,000/summary; 64 KiB README preview | Sanitize and mark display truncation; preserve immutable source for explicit file review. |
| Operation receipts | 1,000 terminal receipts or 30 days | Drop oldest eligible terminal receipts; never delete recovery authority. |


### Native component ownership and canonical restrictions (R74–R75)

Ruling: R74 — native command hooks select their exact existing process-owner adapter from the qualified handler at dispatch and retain that adapter on the original CommandJob through reserve, publish and terminal settlement; the shared HookEvent stays unchanged — concurrent handlers can belong to different installations, so one event-wide owner cannot identify their root/revision custody — cost if wrong: revise the narrow per-job selection seam; owner lookup failure must refuse before launch, and cancellation, late cleanup or replacement must never settle another handler’s custody.

Ruling: R75 — explicit canonical builtin/local tool IDs in native declarations are immutable narrowing constraints resolved against the actual eligible parent registry at use, while noncanonical tokens require reviewed exact owner mappings — a declaration can restrict existing rights without creating a durable generic tool grant or another permission owner — cost if wrong: narrow supported declarations or revise the resolver; missing or changed identities refuse, explicit empty stays empty, recovery creates no grants, and display-name fallback never authorizes a tool.


### Native reviewed component tool restrictions (R24/R75)

The existing configuration review may map a component's noncanonical declared token to an exact same-installation/revision MCP tool mapping in that same reviewed complete set. Existing configuration rows bind the component definition and target mapping digests. The normal MCP owner revalidates those references at publication, admission, recovery and use; unavailable or changed references refuse. This declaration narrows the actual eligible parent registry and cannot recreate a permission. Canonical builtin/local IDs remain immutable runtime narrowing constraints with no additional durable tool grant.


### Native agent invocation owner (R76)

Ruling: R76 — native agent components adapt to the existing AgentDefinition child spawn/resume owner and captured roster; I1 does not introduce a primary-run preset mode for @alias:agent — the accepted contract names AgentDefinition, which is a sub-agent template, and the existing path already owns child budgets, permissions and hooks — cost if wrong: add separately specified primary-preset routing later; native agents remain usable as children, and unsupported selectors must refuse visibly rather than silently changing the primary run or falling back to another component.

Original branch I1 evidence (historical): I1 completed with implementation `b6c4baa693` and reviewed Fix1 `bc1817cd70`: native projection/context/child presets use `Plugins/{components,commands,agent_presets}.py`; persistent component schema and installation-local mapping IDs remain unchanged. Actual Console/`TldwCli` compose the existing hook/MCP/model owners. Fix1 preserves manual-rule references through initial/resumed sends, scopes mapping recovery to installation/revision, and qualifies same-event distinct-installation/revision process/root custody. Fresh scoped review accepted all three items with no new findings. Final ordinary75/affected114/controller17/ownedMCP7 tests and static checks pass; counts overlap. Initial import-headroom/POSIX timing observations and H6 FD growth remain explicit I7/final-review handoffs; other-platform/external-server/management UI qualification is not claimed. These original review/test counts do not qualify the rebased source. Current integration preserves NativeHooks, ComponentRecord with explicit snapshot, unified available_skills and the existing AgentDefinition owner; TASK32695 AC15 requires final new-base checks and independent review.

### Task20 implementation handoff

I2 uses the existing native inventory, review/recovery, HookHandler, matcher and
command executor seams. R77 stores one bounded canonical public catalog carrier;
R78 selects closed version/event codecs without changing runtime ownership.
Supported source pre-guards remain installation-wide dependencies; mixed
transformers cannot qualify final-argument guards. Qualification and exact pinned
fixture provenance are recorded in the Task20 implementation report and interop
README. Original-host execution, other operating systems, external vendor services
and full-suite qualification remain outside this increment.


Historical original-branch I2 qualification follows; these counts and execution approvals do not qualify the rebased source. TASK32695 AC15 remains pending until the new targeted checks and independent review.

I2 accepted with implementation `9de71a8694` and reviewed Fix1 `38daa5315c`: version-pinned portable/OpenAI/Cursor inventories, public catalog replay and pure vendor command-hook codecs use the existing owners. Fix1 preserves usable MCP siblings through inspection, selected-good publication, authenticated recovery and actual admission, with deterministic diagnostic identities for malformed source names (R82). Fresh scoped review closed I1 with no new findings. Final affected113adapter/136owner tests pass; unchanged36component-context/238hook results remain qualified in their recorded contexts. Only the component/context pair required approved normal-host execution after the exact native boot probe returned EPERM in the tool sandbox. Parser captures do not claim original-host execution; other platforms, external services/keychain/UI/full-suite and prior resource/startup/timing observations remain explicit later-task/final-review boundaries. Existing ADR162/163 and R77/R78/R82 govern these contracts.

### I3 implemented owner/API handoff

R83/R84 and TASK-32688's recorded CLI2/CLI4 scope add existing
`Plugins/service.py`, `runtime_owner.py`, `authority_store.py`, `authority.py`
(the acquisition-only purpose constant), `models.py`, and the repository-owned
`Tests/Plugins/acquisition_process_support.py` fixture to the I3 paths above.
No installed schema, authority transition, runtime/process controller or F7
retention implementation is replaced. F1 accepts optional internal checkpoints;
unchanged callers retain their validation.

The service exposes `acquisition_destination`, `acquire_snapshot`,
`release_acquired_snapshot` and `cleanup_acquisitions`. The original worker and
owner retain uncertain process/thread cleanup even after a waiter cancels;
shutdown refuses until it drains. Acquired results retain a recovery reference;
call release only after consuming or discarding the result. Separate origin,
commit and selected-tree fields are not yet persisted installation provenance.
I4 must carry them into authenticated selected-source review/recovery rather
than reduce the result to a path. See the design's **I3 acquisition operation
contract (R83/R84)** for supported sources/authentication, limits, no-replay
eligible R85 receipt retirement/refusal, conservative restart behavior and qualification boundaries.

R86 restores the shared native_console fixture to its canonical `marker` path,
matching unchanged retained restart consumers. The five ordinary neighbor files
and two recorded native fixture-consumer nodes qualify this one-line correction;
missing-marker failures are distinct from native boot-probe restrictions.


Historical original-branch I3 qualification follows; new-base targeted checks and review remain TASK32695 AC15.

I3 accepted at initial `de3707eedd` plus reviewed Fix1 `9e2ddcaf9b`.
Final acquisition verification passes78 cases, with232 targeted neighbor passes
and2 native fixture-consumer controls; changed-file static checks and frozen
source/provenance evidence pass. Fix1 adds an unthrottled retained-worker final
staging/managed/reserve scan before success; failed or cancelled output has no
success reference and remains subject to exact terminal/age-qualified cleanup.
Independent initial/scoped reviews leave no open findings. I4 must still consume
the complete acquired provenance and bind exact catalog/update origins through
authenticated review/recovery. macOS/APFS-only qualification, conservative host
auth support, polled disk/native containment and protected lost/reboot custody
limits remain explicit; no broader platform/full-suite qualification is claimed.

## I4 delivered source and import contract (TASK-32689)

`SourceService(plugin_service)` uses the original PluginService storage worker.
Every mutation and disk read is async; `search(query, offset=..., limit=50)` and
`list_sources()` read immutable memory only. Call `await load_cached()` on opening
the browser, including while locked. It returns valid CatalogSnapshots and also
populates SourceSummary rows for all configured sources, including never-refreshed,
failed and unusable caches. Display metadata is untrusted; it grants no source,
host authentication, installation or activation authority.

- `await add(locator, catalog_path=None)` creates an immutable selected source ID.
  A different origin requires a new source. `await refresh(source_id)` fetches that
  signed source, retaining the last good listing/error when refresh fails. A first
  failed refresh has no successful cache; a successful empty catalog does.
- `await review_entry(source_id, entry_identity, selection=..., workspace_id=...,
  selected_locator=...)` acquires the actual retained entry and returns an ordinary
  install review. For remote entries the UI must pass the exact displayed normalized
  locator; mutable names never select an alternate authentication context. Submit
  the returned review through `PluginService.commit`.
- For direct repository/local installs, use `PluginService.acquire_snapshot`, then
  `await SourceService.review_acquired(snapshot, selection=..., workspace_id=...)`.
  This verifies the full producer result and binds compact origin/ref/commit/tree
  and byte provenance. Passing a materialized Path to `review_install` does not
  provide this direct-source update contract.
- `await check_updates(installation_id)` returns candidates only from the exact
  retained original locator. `await review_update(installation_id, candidate)`
  returns an ordinary revision review; `PluginService.apply_revision` performs the
  existing drain and mutation. `await remove(source_id)` pauses catalog updates
  without uninstalling. Explicit `await review_rebind(installation_id, source_id,
  entry_identity, selected_locator=...)` reviews a newly selected source; applying
  it preserves old immutable provenance even when package bytes are unchanged.
- `await discard_candidate(snapshot_or_inspection)` invalidates candidate reviews
  and releases only unshared producer references. Use this facade for its candidates,
  including after commit; installed/history references remain protected. Original
  I3 low-level release remains caller-controlled and is not a source-aware discard.
- `inspect_app_import(app, selected_root)` reads only known catalog paths and one
  explicitly selected complete package root. Cursor uses `.cursor-plugin`, Codex
  uses `.agents/plugins` / `.claude-plugin` catalogs and supported package manifests.
  It does not scan vendor profile/cache layouts or infer undocumented configuration.
  `await commit_import(preview)` returns copied source IDs, AcquiredSnapshot package
  candidates, exclusions and bounded per-item failures. Copies have local managed
  provenance; review/install and upstream rebinding are separate explicit actions.
  Root/preview mutation, acquisition, authority, quota and storage failures remain
  hard refusals. Only local allowlisted preparation/copy failures are per-item.

Schema5 adds independent marketplace_sources; historical migrations are unchanged.
Compact optional catalog_binding/acquisition_reference fields are mutually exclusive
and omitted for legacy records through serialization, projection and reconstruction.
New source-bound revision digests include stable selected provenance, excluding
operation/nonce/staging identities. Repeated acquisition is stable; explicit source
rebinding creates a distinct immutable revision without changing old unbound digests.

Retained interpretation verifies signed successful producer references, exact physical
incarnation and actual bytes without requiring boot equality. Cleanup retains its
original same-boot destructive qualification and 24-hour age. Signed current and
retained authority, sources, reviews and persisted candidate dependencies protect
catalog/import bases across restart; unknown or pending custody refuses cleanup.
Registry loss can reconstruct installed provenance but cannot invent lost discovery
sources; catalog updates remain paused until explicit reviewed rebinding.

Qualification uses independently authored temporary packages, real SQLite, private
profiles, controlled Git repositories/loopback TLS, actual worker/coordinator/recovery
and existing pinned inert I2 vendor fixtures. Changed-boot evidence is a simulation,
not a real reboot or cross-platform qualification. No foreign grants are imported.

Update candidates are bound to the requested installation identity; a candidate for
one installation cannot be submitted as another installation's update. Explicit
rebinding remains a separate reviewed action. During active acquisition only,
quota monitoring tolerates a vanished descendant inside its physically qualified
staging leaf. Final acceptance and all non-acquisition capacity scans remain strict;
this exception does not grant deletion authority or suppress other storage errors.

### I5 R98 budget qualification

The exact Task23 BASE full-registry control already exceeded ADR097 at515 modules;
the initial Plugins implementation added4. R98 defers the twelve measured
Settings category import groups plus Plugins private display/state modules at
first use, retaining existing export/class/patch identities and event handlers.
Actual full-registry qualification reaches499 modules under the unchanged500 cap;
headroom warnings remain visible. The existing Settings category sweep at both
terminal sizes and targeted category/constructor controls qualify first use.
This does not identify the original branch-wide cause of the BASE breach.

### Task23 Fix1 — R99/R100 authenticated recorded display and actual category use

Passive Installed listing and explicit README metadata lookup use recorded inspection metadata authenticated in the same registry SQLite projection that is compared with protected current authority and the published snapshot. This replaces the earlier contradictory `capture(fresh=False)` display requirement: that call still fully reinspects package bodies and remains mandatory for execution, not passive display. The registry collector is untrusted until coordinator equality gates succeed. Only requested current revision metadata is returned; unprotected evidence/availability/diagnostics/candidates are stripped and dependency edges derive from authenticated component constraints. No schema, store, owner or authority format changes.

Admission shares its existing activation/trust/selection/mapping/dependency/revocation/live-fence checks between full capture and recorded eligibility. Display returns only component IDs, never RunPluginSnapshot, tokens, leases or run admission. Execution retains complete retained-byte verification, drain gates, component ceilings, data-root epochs and current authority checks. UI labels recorded eligibility and explicitly says package bytes are not reverified. Explicit README opens only the selected current root's no-follow regular README, reads at most65537 bytes including sentinel and returns at most65536 UTF8 bytes including truncation marker; it is a bounded literal preview, not proof of whole-package integrity.

R100 requires actual cold Overview and unrelated shared handlers to leave unopened category panels/models deferred. Symbol resolution follows its matching category/field/ID/saved-content/nonempty-query branch; genuine shared Overview model use must be identified and tested. Original class/export/assigned-module-patch identity and saved drafts are preserved. ADR097 caps and all routes remain unchanged.

Fix1 controls cover zero package-member reads for list/search, entire selected README read bound, authenticated-record tampering, identical shared eligibility/runtime gates with full runtime verification, cold Overview then actual category/draft behavior, held-load Back/form/new-owner cleanup and scoped token-backed title/placeholder contrast. See task-23-fix1-proposal.md and CLI5 plan; root owns independent re-review and closeout.


### Current dev navigation ownership (2026-10-17 integration)

The removed `UI/Workbench/route_inventory.py` is not restored. Shared route and shell coverage uses current `UI/Navigation/screen_registry.py` and `shell_destinations.py`, including Meetings, Research and the new Plugins destination. Plugins uses F8; all existing shortcuts, including Settings F4, remain. The lazy source facade follows the current app skill-wiring owner. This preserves the route-coverage outcome of historical I5 AC5 through current owners; rebased-source checks remain TASK32695 AC15.


### I5 route-owned Plugins styling (2026-10-17 integration)

Apply the existing ADR097 first-navigation stylesheet owner. The current builder splits only `plugins`-prefixed browser selectors from `features/_plugins.tcss` into generated `screen_feature_plugins.tcss`; the app's existing route CSS map loads it on Plugins entry. Shared-selector, singular `plugin-review` and type-only modal rules stay in the normal boot remainder, so standalone plugin review/authority/status modals retain first-open styling. Current classifier/cascade rules and all startup caps remain unchanged. There is no new loader, Screen.CSS_PATH feature duplication, modal owner, token or cap exception. Existing source builder alone generates the sheets; qualify source synchronization, the full boot budget, first navigation and standalone modal first use at both terminal sizes.

### Task24 / I6 exact management owners (R101–R103)

The management UI consumes issued immutable reviews and the single app-owned
PluginService/SourceService. Acquired format selection authenticates the unchanged
full producer before reinspection; reference-discard separately authenticates
terminal custody and the complete protected dependency union. Install-and-enable
records only a named workspace intent, atomically with the default-disabled,
untrusted installation. Trust and configuration remain separate gates.

R101 adds the existing select result kind to the public reviewed selection path.
It drains through the revision owner, commits exact selection/current revision and
advances installation authority without changing trust, mappings or activation.

Operation history derives from authenticated transition/current-marker evidence,
never SQLite hints. App-owned UI awaiters preserve submitted work across navigation;
restart resolves durable evidence, and session-only failure or uncertain cleanup
never becomes durable success or confirmed stop. Nonsecret drafts survive Close;
secrets remain exclusively in the owning ephemeral credential flow.

R102 permits explicit hook/connection tests only through a lazy Console-owned
review/reservation adapter for an existing real session in the captured workspace.
It uses normal current authority, permission/persona/parent ceilings, process and
MCP owners, budgets and cleanup. Review executes nothing; explicit execution
discloses initialization, makes no model request and displays test effects without
injecting them into conversation context. No UI-fabricated caller context.

R103 exposes explicit reviewed reboot reconciliation only with a separately
protected namespace/current-marker-matching clean checkpoint, exact native root
identities and drained current users. The existing root owner retains that
positive proof across its own fenced dirty publication and rechecks it at apply.
Changed boot alone, SQL row absence, operator assertion and released locks never
prove whole-root quiescence. Dirty, unknown, legacy, reconstructed, reduced-backend,
remote-uncertain and pending-rebind cases remain Recovery required without proof;
show specific reasons and available actual stop/wait/retry/resume actions. There
is no automatic rebind, new persistent proof store or universal recovery claim.

Concrete APIs/scope and controls: `.superpowers/sdd/2026-09-15-managed-plugins-delivery/`
`task-24-seam-proposal-e817ec8df5.md`, its root-proof addendum and root seam decision.
ADR required: yes; this amends ADR162 and the ADR163 Console execution handoff.

### Reviewed management removal (R105)

Management device-wide disable/uninstall obtains an actual issued PluginReview
through PluginService.review_removal before confirmation. The existing
coordinator binds current revision, authority and consequences.
PluginService.begin_reviewed_removal validates the retained original object,
kind, expiry and installation from minimal facade custody, then uses the existing
immediate fence/cancel request and worker persistence. It carries the original
review and operation ID; worker revalidation cannot replace them with newer
state. Refusal after sealing remains a conservative session-only block with an
honest persistence error, without deletion or automatic reopening. Scoped
disable uses review_activation(intent="disabled") and commit. Existing emergency
begin_disable/begin_uninstall behavior is unchanged. Same-request retry retains
custody; navigation never retargets the reviewed installation or workspace.

### I6 R107 / CLI6 bounded startup guard closeout

ADR required: no new ADR. ADR097 directly governs first-use import deferral with
unchanged caps; ADR150/162/163 contracts remain unchanged. After exactBASE/current
origin-qualified975-module set equality, pin side-chat/generation absence, move
both real eager edges to canonical first-use functions (runtime constructors and
isinstance use real classes), migrate incidental imported-alias test patches to
the canonical generation owner, and run cold+later-use/style/side-chat/generation/
H3/attachment controls. Disable only optional splash in the named preimport
ordering fixture via existing save_setting_to_cli_config in its private profile (R108/CLI7); retain5s and synchronous order
assertion. Preserve Freeze05 and all failures; freeze semanticPhase06 and renew
changed coverage/static/ratchets/live under the same private-profile policy.

## Task24 Fix1 — bounded saved-root impact (R110 / CLI9)

RootUsage.blocker_status(roots, *, limit=50, offset=0) returns the existing
one-pass owner projection as blockers plus blockers_complete. Existing blockers
keeps its tuple/page contract. Completeness means offset==0 and the entire
collected projection fits the requested limit, never atomic quiescence or a
permission grant. Preserve short live-lock then separate storage access; never
hold the live fence lock over database reads. PluginService.data_status adds
this completeness field without a new runtime owner or raised page cap.

Before saved-root confirmation, disclose exact IDs/generations/paths and actual
owner/workspace/reason/recovery/cleanup metadata. Unknown coverage, missing or
false completeness, or oversized disclosure disables confirmation. Reread exact
refs and the displayed projection before submit; changed users or generations
require renewed review. Normal owner revalidation/fencing/drain still governs
users arriving later. Wait, cancel operation and cancel affected work remain
distinct. Uninstalled packages remain absent; history routes by original exact
installation ID to their retained roots through existing data owners. Mixed
PluginReview/RootReview acquisition protection retains original dependencies,
root custody and fail-closed behavior for unknown review kinds.

ADR required: existing ADR162 amendment for the additive status projection; no
new ADR/schema/owner. ADR163/150/097 remain unchanged. Fix1 retains all initial
failed phases and qualifies the three Important review findings independently.


### Fix1 exact retained-owner pages (R111 / CLI10)

PluginService.data_status accepts optional installation_id and filters the
authenticated original root owner before page slicing and exact-ref equality.
The row need not still be installed. Missing IDs return no roots; mismatching
exact refs refuse. Existing no-filter and installed attached-root exact-ref
callers retain behavior. The manager's256-root review bound counts only the
exact target installation, with one bounded lookahead; unrelated roots cannot
strand retained cleanup. No name matching, fabricated install, raised page cap
or new authority. ADR162 records this additive read-only facade argument.

### I6 current Console test admission (2026-10-17 integration)

Explicit managed-plugin tests use current asynchronous configuration capture and the actual originating Console session, runtime and received-turn/reservation owners. A shared private nonawaiting publication helper preserves the ordinary submission's existing admission checks. The explicit test path first refuses pending received turns/claims and busy or full capacity, then reserves without an intervening await; it never claims or consumes a received turn. Recapture/currentness, consent, persona/profile/parent ceilings, scope cancellation and original process/root terminal custody remain mandatory. Synthetic initialization and test effects stay outside later conversation context; test-only cleanup cannot retire another turn's or live session's resources.

The existing private MCP composer may retain its actual empty owned-provider group only on the explicit discovery path. That flag does not bypass its kill switch, source/currentness, permissions or captured maximums; normal callers still receive no empty catalog. Discovery uses the exact selected owned component and existing ConnectionOwnership, grants no tools, and detaches only leases acquired by the test's original owner. Review makes no connection/model request. The actual lazy app mixin owns management operations. Existing R4/R12 closed reference forms and complete-set configuration remain authoritative; do not import a later prefill helper into this increment. Current route-owned Plugins CSS and boot-loaded standalone modal rules stay under R15, with unchanged caps.

This directly integrates existing ADR162/163 contracts through current private seams. Original validation counts are historical; TASK32695 AC15 qualifies the latest dev source with real Console capture/reservation, empty discovery and normal-run controls, cancellation/cleanup, complete configuration and both modal visit orders.
