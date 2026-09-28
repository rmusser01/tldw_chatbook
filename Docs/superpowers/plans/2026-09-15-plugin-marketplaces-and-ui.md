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
- Test: `Tests/Plugins/test_plugin_context.py`
- Test: `Tests/Chat/test_console_skill_substitution.py`

**Interfaces**

- Consumes: F5 owned skill/context admission, M4 plugin tools and H6 complete hook runtime.
- Produces: PluginComponents.project(snapshot: RunPluginSnapshot) -> tuple[ComponentRecord, ...]. render_command(component: ComponentRecord, arguments: dict[str, str]) -> tuple[dict, ...] returns separately attributed instruction and argument blocks. resolve_agent_tools(value: str | tuple[str, ...], eligible: frozenset[str]) -> frozenset[str] preserves inherit versus empty. register_hook_definitions(snapshot: RunPluginSnapshot) -> tuple[HookHandler, ...] supplies owned definitions and current-authority validation.

- [ ] **1. Create the first behavioral test and its local fixtures.** Use the fixture contract in the implementation increments below; initial import/behavior must fail for the missing feature.

```python
def test_empty_agent_tool_list_never_inherits():
    from tldw_chatbook.Plugins.agent_presets import resolve_agent_tools
    eligible = frozenset({"mcp:repository:read"})
    assert resolve_agent_tools((), eligible) == frozenset()
    assert resolve_agent_tools("inherit", eligible) == eligible
```

- [ ] **2. Establish RED through the intended entry.** Run `python -m pytest Tests/Plugins/test_native_components.py -q`. Expected: the named new behavior fails, while any positive precondition/control succeeds. Resolve test harness/API errors before changing production code.

- [ ] **3. Implement the smallest invariant, then integrate the real owner.** This kernel states the ordering/data rule; the following increments supply the complete behavior and limits.

```python
def resolve_agent_tools(value, eligible: frozenset[str]) -> frozenset[str]:
    if value == "inherit":
        return eligible
    requested = frozenset(value)
    if not requested <= eligible:
        raise ValueError("plugin_agent_tools_unavailable")
    return requested
```

  - [ ] 3.1. Normalize command/rule/agent definitions already inventoried by F1 into existing composer/context/AgentDefinition seams. Give aliases stable installation identity; no display-name tool lookup or recursive command expansion.
  - [ ] 3.2. Keep rule order stable by installation/component ID, support only always/manual activation and require explicit manual adaptation for file/model-selected modes. Reject oversized required blocks whole.
  - [ ] 3.3. Preserve skill manual-only, inline/fork, explicit tool restrictions and model mapping semantics through final provider input; plugin prose never enters an internal system-authority lane.
  - [ ] 3.4. Supply owned v2 hook sets and dependency requirements at the next run snapshot; report partial readiness per component and never auto-select newly supported items. Bind the H2 HookProcessOwner port to F2 before any plugin command launch, preserving pending-launch provenance and root usage.

**Failure and successful-control matrix:** Empty/inherit/unmapped tools, mapped/unknown model, arguments containing command syntax, namespace collisions, forked context authority, whole-block overflow, mid-run enablement, disabled required hook and successful unrelated component.

- [ ] **4. Establish GREEN and preserve the neighboring path.** Run the exact files below. Expected: all execute and pass, with no silent coroutine/platform skips used as qualification. Inspect real resources/output, not source-string matches.

```bash
python -m pytest Tests/Plugins/test_native_components.py Tests/Plugins/test_plugin_context.py Tests/Chat/test_console_skill_substitution.py -q
```

- [ ] **5. Review, record evidence and commit the task.** Update its ACs/notes and the relevant authoring/operation documentation; record platform limits. Run `git diff --check`, Python syntax checks and the verification-environment formatter/linter on the changed Python files as specified in the delivery plan. Stage the exact task-owned files, including any test fixtures and generated CSS, and commit; never stage unrelated work.

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
- Produces: inspect_codex(root: Path, catalog_overlay: dict | None) -> PackageInspection; inspect_cursor(root: Path, catalog_overlay: dict | None) -> PackageInspection; normalize_vendor_hook(value: dict, dialect_version: str) -> HookHandler. overlay_for_openai(inline: dict | None, compatibility: dict | None) -> dict selects wholesale, never merges. Unsupported adaptations carry reason and affected dependency IDs.

- [ ] **1. Create the first behavioral test and its local fixtures.** Use the fixture contract in the implementation increments below; initial import/behavior must fail for the missing feature.

```python
def test_inline_overlay_replaces_compatibility_wholesale():
    from tldw_chatbook.Plugins.adapters.codex import overlay_for_openai
    assert overlay_for_openai({"skills": ["new"]}, {"skills": ["old"], "hooks": "legacy"}) == {"skills": ["new"]}
    assert overlay_for_openai(None, {"skills": ["old"]}) == {"skills": ["old"]}
```

- [ ] **2. Establish RED through the intended entry.** Run `python -m pytest Tests/Plugins/test_codex_adapter.py -q`. Expected: the named new behavior fails, while any positive precondition/control succeeds. Resolve test harness/API errors before changing production code.

- [ ] **3. Implement the smallest invariant, then integrate the real owner.** This kernel states the ordering/data rule; the following increments supply the complete behavior and limits.

```python
def overlay_for_openai(inline: dict | None, compatibility: dict | None) -> dict:
    return dict(inline if inline is not None else (compatibility or {}))
```

  - [ ] 3.1. Pin independently curated fixtures with upstream commit, original URL, license and expected inventory; use controlled runnable examples separately from parser fixtures. Native Chatbook extension constraints still apply when portable inventory survives malformed input.
  - [ ] 3.2. Implement inline OpenAI precedence, standalone vendor candidates, root portable identity/location rules and Cursor explicit-path replacement. Catalog executable overlays participate in effective digest and review.
  - [ ] 3.3. Normalize supported variable and metadata subsets explicitly. Conflicting manual-only declarations, unavailable rules/models/constraints and unknown source behaviors remain invalid/unsupported rather than inherited defaults.
  - [ ] 3.4. Qualify each hook mapping across timing, matcher, payload, result, cwd, variable expansion and limits. Parse only tested simple platform-specific argv forms; reject shell operators/substitutions and unsupported regex semantics. Unknown guard scope blocks executable/automatic components until reviewed narrowing.

**Failure and successful-control matrix:** Portable plus both vendor manifests, invalid root schema, empty explicit component paths, inline overlay removing compatibility fields, foreign allow bypass, guard versus transformer, source success-only post-event, file rule manual adaptation and parser/behavior evidence separation.

- [ ] **4. Establish GREEN and preserve the neighboring path.** Run the exact files below. Expected: all execute and pass, with no silent coroutine/platform skips used as qualification. Inspect real resources/output, not source-string matches.

```bash
python -m pytest Tests/Plugins/test_codex_adapter.py Tests/Plugins/test_cursor_adapter.py Tests/Plugins/test_vendor_hook_adapters.py -q
```

- [ ] **5. Review, record evidence and commit the task.** Update its ACs/notes and the relevant authoring/operation documentation; record platform limits. Run `git diff --check`, Python syntax checks and the verification-environment formatter/linter on the changed Python files as specified in the delivery plan. Stage the exact task-owned files, including any test fixtures and generated CSS, and commit; never stage unrelated work.

```bash
backlog task task-32687 --plain
git diff --check
```

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
- Produces: SourceService.refresh(source_id: str) -> Awaitable[CatalogSnapshot]; search(query: str, *, offset: int, limit: int = 50) -> tuple[CatalogEntry, ...]; check_updates(installation_id: str) -> Awaitable[tuple[PackageInspection, ...]]; remove(source_id: str) -> None. inspect_app_import(app: str, selected_root: Path) -> ImportPreview returns allowlisted source/package references and exclusions; commit_import(preview: ImportPreview) records only the reviewed copy. Frozen CatalogSnapshot fields are source_id, catalog_commit, digest, fetched_at, entries and refresh_error; CatalogEntry contains source-qualified identity, package locator, overlays, support and diagnostics. ImportPreview binds the selected root identity, candidate references, exclusions and reviewed digest.

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
- Modify: `tldw_chatbook/UI/Navigation/main_navigation.py`
- Modify: `tldw_chatbook/UI/stable_command_palette.py`
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
- Produces: PluginsScreen(source_service: SourceService, coordinator: PluginCoordinator) uses screen route plugins. PluginBrowseState stores view, query, page, selected_source_id, selected_installation_id and request_generation. accept_response(request_generation: int, current_generation: int) -> bool ignores stale responses. All mutations remain coordinator calls; projected UI state is never authority.

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
