# Application Service Composition Lifecycle Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Writing and Chat conversation service composition single-pass, bind Writing to the application-owned server-context provider, bind Chat and Media to the application-owned Sync scope at initial construction, and prove those contracts in both the source checkout and a clean installed wheel.

**Architecture:** Keep `TldwCli` as the one composition root and retain the first dependency-ready call site for each affected service graph. Remove only the duplicate calls, pass existing long-lived dependencies directly into the affected constructors, and add narrow AST plus real-production-app sentinels; do not add a container, make private provider wiring reentrant, or use a surrogate application.

**Tech Stack:** Python 3.11+, Textual, pytest/pytest-asyncio, Python `ast`, `build`, pip target installs, Ruff.

---

## Inputs and constraints

- Backlog task: [`TASK-1538`](../../../backlog/tasks/task-1538%20-%20Enforce-single-pass-service-composition-and-runtime-dependency-binding.md)
- Approved design: [`2026-07-28-application-service-composition-lifecycle-design.md`](../specs/2026-07-28-application-service-composition-lifecycle-design.md)
- Governing decision: [`ADR-036`](../../../backlog/decisions/036-application-service-composition-lifecycle.md)
- Reviewed baseline: `origin/dev` at `61960f436`
- Tests must use the real production `TldwCli` or an app-independent pure function. No test `App`, simplified app, `SimpleNamespace`, `MagicMock`, `object.__new__(TldwCli)`, or unbound `TldwCli` method is acceptable as application integration evidence.
- Preserve the existing post-construction Sync reassignment loop in `_wire_watchlists_and_notifications_services()`. The separate server-context provider helper remains non-reentrant.
- Keep public `from_config(...)` compatibility constructors importable.
- Do not use `Tests/Performance/test_app_startup_performance.py::test_conversation_wiring_attaches_migration_before_existing_artifact_coordinator` as evidence; it is a surrogate/unbound-method test. Replace its relevant citation-ordering evidence with identities observed on a full production app.
- The installed RED run verified that `ChaChaNotes_DB.py` reads `tldw_chatbook/DB/migrations/chachanotes_v26_to_v27_citation_provenance.sql` at runtime while the current wheel omits it. Current `dev` subsequently added the exact `tldw_chatbook/DB/migrations/chachanotes_v27_to_v28_character_authority.sql` runtime dependency. Package and enforce both exact assets under ADR-032; do not broaden package data to a recursive catch-all.
- Do not commit the intentionally red state. Commit tests and the minimal implementation together after green verification.

## ADR check

ADR required: yes

ADR path: `backlog/decisions/036-application-service-composition-lifecycle.md`

Reason: The change defines application service construction, runtime provider ownership, Sync dependency binding, and the boundary at which a future service container remains unjustified.

## File map

- Create `Tests/ProductionApp/test_service_composition_lifecycle.py`: narrow AST sentinel plus full `TldwCli` construction/mount/unmount identity proof.
- Modify `Tests/Packaging/test_installed_distribution.py`: extend the existing isolated installed-wheel child probe with the same call-count and identity contract.
- Modify `MANIFEST.in`, `pyproject.toml`, and `Packaging/check_manifest.py`: declare and enforce the exact runtime citation-provenance and character-authority migrations in both artifacts.
- Modify `Packaging/PACKAGING_CHECKLIST.md`: document the database migration runtime-data obligation.
- Modify `tldw_chatbook/app.py`: remove the two duplicate calls, use `server_context_provider` for Writing, and inject Sync into Chat and Media scopes.
- Modify `Docs/superpowers/specs/2026-07-28-application-service-composition-lifecycle-design.md`: record final implementation status and verification evidence.
- Modify `Docs/superpowers/plans/2026-07-28-application-service-composition-lifecycle.md`: check completed steps and record exact command evidence.
- Modify `backlog/tasks/task-1538 - Enforce-single-pass-service-composition-and-runtime-dependency-binding.md`: check acceptance criteria, add concise implementation notes, and set Done only after every required check succeeds.

### Task 1: Add red source, production-app, and installed-wheel contracts

**Files:**

- Create: `Tests/ProductionApp/test_service_composition_lifecycle.py`
- Modify: `Tests/Packaging/test_installed_distribution.py:41-454`
- Test: `Tests/ProductionApp/test_service_composition_lifecycle.py`
- Test: `Tests/ProductionApp/test_reactive_ownership_maturity.py`
- Test: `Tests/Packaging/test_installed_distribution.py`

- [x] **Step 1: Create the narrow AST and full-production-app test**

Create `Tests/ProductionApp/test_service_composition_lifecycle.py` with the
following complete contract:

```python
from __future__ import annotations

from collections import Counter
import ast
import logging
from pathlib import Path

import pytest

import tldw_chatbook.app as app_module
from tldw_chatbook.app import TldwCli


WIRING_METHODS = (
    "_wire_writing_services",
    "_wire_chat_conversation_services",
)
EXPECTED_WIRING_CALLS = Counter({name: 1 for name in WIRING_METHODS})
SERVICE_ATTRIBUTES = (
    "local_writing_service",
    "server_writing_service",
    "writing_scope_service",
    "local_chat_conversation_service",
    "conversation_local_marks_service",
    "server_chat_conversation_service",
    "chat_conversation_scope_service",
    "citation_trace_repository",
    "citation_legacy_migration_service",
    "citation_artifact_ownership_coordinator",
    "media_reading_scope_service",
    "sync_scope_service",
)
APP_PATH = Path(app_module.__file__).resolve()


def _constructor_wiring_calls() -> Counter[str]:
    tree = ast.parse(
        APP_PATH.read_text(encoding="utf-8"),
        filename=str(APP_PATH),
    )
    app_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "TldwCli"
    )
    init_method = next(
        node
        for node in app_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )
    return Counter(
        node.func.attr
        for node in ast.walk(init_method)
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "self"
            and node.func.attr in WIRING_METHODS
        )
    )


def _disable_splash(monkeypatch: pytest.MonkeyPatch) -> None:
    real_get_cli_setting = app_module.get_cli_setting

    def get_cli_setting_without_splash(section, key=None, default=None):
        if section == "splash_screen" and key == "enabled":
            return False
        return real_get_cli_setting(section, key, default)

    monkeypatch.setattr(app_module, "get_cli_setting", get_cli_setting_without_splash)


def _service_identities(app: TldwCli) -> tuple[object, ...]:
    return tuple(getattr(app, name) for name in SERVICE_ATTRIBUTES)


def _assert_service_identities(
    app: TldwCli,
    expected: tuple[object, ...],
) -> None:
    current = _service_identities(app)
    assert len(current) == len(expected)
    assert all(
        actual is original for actual, original in zip(current, expected, strict=True)
    )


def _assert_service_graph(app: TldwCli) -> None:
    assert app.writing_scope_service.local_service is app.local_writing_service
    assert app.writing_scope_service.server_service is app.server_writing_service
    assert app.server_writing_service.client_provider is app.server_context_provider
    assert (
        app.chat_conversation_scope_service.local_service
        is app.local_chat_conversation_service
    )
    assert (
        app.chat_conversation_scope_service.server_service
        is app.server_chat_conversation_service
    )
    assert app.chat_conversation_scope_service.sync_scope_service is app.sync_scope_service
    assert app.media_reading_scope_service.sync_scope_service is app.sync_scope_service
    assert (
        app.local_chat_conversation_service.citation_legacy_migration
        is app.citation_legacy_migration_service
    )
    assert (
        app.citation_artifact_ownership_coordinator.trace_repository
        is app.citation_trace_repository
    )
    assert (
        app.citation_artifact_ownership_coordinator.artifact_store
        is app.local_chatbook_service
    )


async def _close_production_app(app: TldwCli) -> None:
    try:
        if app._rich_log_handler:
            await app._rich_log_handler.stop_processor()
            logging.getLogger().removeHandler(app._rich_log_handler)
            app._rich_log_handler.close()
        await app.on_shutdown_request()
        await app.on_unmount()
    except Exception:
        pass


def test_constructor_contains_one_call_for_each_guarded_composition_helper() -> None:
    assert _constructor_wiring_calls() == EXPECTED_WIRING_CALLS


@pytest.mark.asyncio
async def test_production_app_composes_one_stable_dependency_graph(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: Counter[str] = Counter()
    for method_name in WIRING_METHODS:
        original = getattr(TldwCli, method_name)

        def counted(
            self: TldwCli,
            _original=original,
            _method_name=method_name,
        ) -> None:
            calls[_method_name] += 1
            _original(self)

        monkeypatch.setattr(TldwCli, method_name, counted)

    _disable_splash(monkeypatch)
    app = TldwCli()
    app.app_config["_first_run"] = False
    identities = _service_identities(app)

    try:
        assert calls == EXPECTED_WIRING_CALLS
        _assert_service_graph(app)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            assert calls == EXPECTED_WIRING_CALLS
            _assert_service_identities(app, identities)
            _assert_service_graph(app)
        assert calls == EXPECTED_WIRING_CALLS
        _assert_service_identities(app, identities)
        _assert_service_graph(app)
    finally:
        await _close_production_app(app)
```

The member-by-member `is` assertions are intentional: value equality is not
evidence that the composition root retained the same service objects.

- [x] **Step 2: Verify the source sentinel fails for the duplicate calls**

Run:

```bash
../../.venv/bin/python -m pytest Tests/ProductionApp/test_service_composition_lifecycle.py::test_constructor_contains_one_call_for_each_guarded_composition_helper -q
```

Expected: FAIL; the actual `Counter` reports two calls for each guarded helper
while the expected counter reports one.

- [x] **Step 3: Verify the full production app exposes the current defects**

Run:

```bash
../../.venv/bin/python -m pytest Tests/ProductionApp/test_service_composition_lifecycle.py::test_production_app_composes_one_stable_dependency_graph -q
```

Expected: FAIL before implementation. The first failing assertion must be an
observed production defect: duplicate helper count, Writing provider mismatch,
or absent Chat/Media Sync binding. A collection/setup failure is not an
acceptable red result.

- [x] **Step 4: Extend the existing installed-wheel probe**

In the `INSTALLED_PROBE` raw string, add `Counter` to the child imports and,
after importing `TldwCli` but before `app = get_app()`, install wrappers that
still invoke the real installed methods:

```python
from collections import Counter

# ... existing imports and AST ownership checks ...

wiring_methods = (
    "_wire_writing_services",
    "_wire_chat_conversation_services",
)
expected_wiring_calls = Counter({name: 1 for name in wiring_methods})
wiring_calls = Counter()
for method_name in wiring_methods:
    original = getattr(TldwCli, method_name)

    def counted(
        self,
        _original=original,
        _method_name=method_name,
    ):
        wiring_calls[_method_name] += 1
        _original(self)

    setattr(TldwCli, method_name, counted)


def service_identities(app):
    return tuple(
        getattr(app, name)
        for name in (
            "local_writing_service",
            "server_writing_service",
            "writing_scope_service",
            "local_chat_conversation_service",
            "conversation_local_marks_service",
            "server_chat_conversation_service",
            "chat_conversation_scope_service",
            "citation_trace_repository",
            "citation_legacy_migration_service",
            "citation_artifact_ownership_coordinator",
            "media_reading_scope_service",
            "sync_scope_service",
        )
    )


def assert_service_identities(app, expected):
    current = service_identities(app)
    assert len(current) == len(expected)
    assert all(
        actual is original
        for actual, original in zip(current, expected, strict=True)
    )


def assert_service_graph(app):
    assert app.writing_scope_service.local_service is app.local_writing_service
    assert app.writing_scope_service.server_service is app.server_writing_service
    assert app.server_writing_service.client_provider is app.server_context_provider
    assert (
        app.chat_conversation_scope_service.local_service
        is app.local_chat_conversation_service
    )
    assert (
        app.chat_conversation_scope_service.server_service
        is app.server_chat_conversation_service
    )
    assert (
        app.chat_conversation_scope_service.sync_scope_service
        is app.sync_scope_service
    )
    assert app.media_reading_scope_service.sync_scope_service is app.sync_scope_service
    assert (
        app.local_chat_conversation_service.citation_legacy_migration
        is app.citation_legacy_migration_service
    )
    assert (
        app.citation_artifact_ownership_coordinator.trace_repository
        is app.citation_trace_repository
    )
    assert (
        app.citation_artifact_ownership_coordinator.artifact_store
        is app.local_chatbook_service
    )


app = get_app()
assert isinstance(app, TldwCli)
assert wiring_calls == expected_wiring_calls
assert_service_graph(app)
initial_service_identities = service_identities(app)
```

Inside the existing `app.run_test(...)` block and again immediately after
`asyncio.run(exercise_production_app())`, add:

```python
assert wiring_calls == expected_wiring_calls
assert_service_identities(app, initial_service_identities)
assert_service_graph(app)
```

Keep every existing installed-root, resource, entry-point, loaded-module,
immutability, and Home-to-Chat assertion unchanged.

- [x] **Step 5: Verify the installed-wheel probe fails for the production defect**

Run:

```bash
../../.venv/bin/python -m pytest Tests/Packaging/test_installed_distribution.py::test_installed_wheel_loaders_entry_points_and_assets_are_immutable -q
```

Expected on the reviewed baseline: FAIL at the service-composition call-count
assertion. The child also logs a deterministic missing-file failure for
`chachanotes_v26_to_v27_citation_provenance.sql`; Task 2 repairs that verified
installed-runtime prerequisite before the application wiring change.

### Task 2: Repair the installed migration-asset contracts

**Files:**

- Modify: `Tests/Packaging/test_installed_distribution.py`
- Modify: `MANIFEST.in`
- Modify: `pyproject.toml`
- Modify: `Packaging/check_manifest.py`
- Modify: `Packaging/PACKAGING_CHECKLIST.md`

- [x] **Step 1: Add the exact migrations to artifact expectations**

Define this test constant:

```python
CITATION_MIGRATION_PATH = (
    "tldw_chatbook/DB/migrations/"
    "chachanotes_v26_to_v27_citation_provenance.sql"
)
CHARACTER_AUTHORITY_MIGRATION_PATH = (
    "tldw_chatbook/DB/migrations/"
    "chachanotes_v27_to_v28_character_authority.sql"
)
RUNTIME_MIGRATION_PATHS = {
    CITATION_MIGRATION_PATH,
    CHARACTER_AUTHORITY_MIGRATION_PATH,
}
```

Add `RUNTIME_MIGRATION_PATHS` to both `required_sdist` and `required_wheel` in
`test_built_artifacts_match_distribution_contract`.

- [x] **Step 2: Add a release-checker removal regression**

Add a focused parameterized test that copies the built distributions, rewrites
the wheel without each member of `RUNTIME_MIGRATION_PATHS`, runs
`Packaging/check_manifest.py`, and asserts return code `1` plus the exact
missing path in output. Use the same standard-library archive rewrite pattern
as `test_release_checker_rejects_missing_runtime_data`.

- [x] **Step 3: Verify the artifact expectation is RED**

Run:

```bash
../../.venv/bin/python -m pytest Tests/Packaging/test_installed_distribution.py::test_built_artifacts_match_distribution_contract -q
```

Expected: FAIL because the exact migrations are absent from both freshly built
artifacts.

- [x] **Step 4: Declare the exact runtime files**

Add this exact root-manifest entry:

```text
include tldw_chatbook/DB/migrations/chachanotes_v26_to_v27_citation_provenance.sql
include tldw_chatbook/DB/migrations/chachanotes_v27_to_v28_character_authority.sql
```

Add this exact setuptools package-data owner:

```toml
"tldw_chatbook.DB" = [
    "migrations/chachanotes_v26_to_v27_citation_provenance.sql",
    "migrations/chachanotes_v27_to_v28_character_authority.sql",
]
```

Do not enable `include-package-data` and do not package every SQL or migration
file.

- [x] **Step 5: Verify the fresh artifacts now contain the migrations**

Run the Step 3 command again.

Expected: PASS.

- [x] **Step 6: Verify the release checker is still RED**

Run:

```bash
../../.venv/bin/python -m pytest Tests/Packaging/test_installed_distribution.py::test_release_checker_rejects_missing_database_migration -q
```

Expected: FAIL because the checker does not yet require the removed files.

- [x] **Step 7: Enforce the migrations in the reusable checker**

Add both exact migration paths to `REQUIRED_SDIST_PATHS` and
`REQUIRED_WHEEL_PATHS` in `Packaging/check_manifest.py`. Update
`Packaging/PACKAGING_CHECKLIST.md` to name the packaged runtime migrations.

- [x] **Step 8: Verify the packaging contracts**

Run:

```bash
../../.venv/bin/python -m pytest \
  Tests/Packaging/test_installed_distribution.py::test_built_artifacts_match_distribution_contract \
  Tests/Packaging/test_installed_distribution.py::test_release_checker_accepts_fresh_artifacts \
  Tests/Packaging/test_installed_distribution.py::test_release_checker_rejects_missing_database_migration \
  -q
```

Expected: `3 passed`.

- [x] **Step 9: Re-run installed RED without a migration failure**

Run:

```bash
../../.venv/bin/python -m pytest Tests/Packaging/test_installed_distribution.py::test_installed_wheel_loaders_entry_points_and_assets_are_immutable -q
```

Expected: FAIL only on the guarded helper count; the installed ChaChaNotes
migrations complete and no missing migration path appears in child output.

### Task 3: Apply the minimal composition-root repair

**Files:**

- Modify: `tldw_chatbook/app.py:3566-3584`
- Modify: `tldw_chatbook/app.py:3665-3680`
- Modify: `tldw_chatbook/app.py:4243-4340`
- Test: `Tests/ProductionApp/test_service_composition_lifecycle.py`
- Test: `Tests/Packaging/test_installed_distribution.py`

- [x] **Step 1: Bind Media to the existing Sync scope**

Keep Media's local and server service construction before the Library,
workspace, prompt/Chatbook, and watchlist/notification helpers. Move only the
existing `MediaReadingScopeService` construction immediately after
`_wire_watchlists_and_notifications_services()`, where Sync now exists, and
change it to:

```python
self.media_reading_scope_service = MediaReadingScopeService(
    local_service=self.local_media_reading_service,
    server_service=self.server_media_reading_service,
    policy_enforcer=self.service_policy_enforcer,
    sync_scope_service=self.sync_scope_service,
)
```

Do not call or move `_wire_server_context_provider()`. Do not remove or alter
the Sync reassignment loop inside
`_wire_watchlists_and_notifications_services()`.

- [x] **Step 2: Remove only the later duplicate calls**

Keep the first `_wire_writing_services()` call after watchlist/notification
composition and the first `_wire_chat_conversation_services()` call after
`chachanotes_db` resolution.

Delete only these later calls:

```python
self._wire_evaluation_services()
self._wire_study_services()
self._wire_writing_services()  # delete
self._wire_research_services()
self._wire_character_persona_services()
self._wire_chat_conversation_services()  # delete
```

Do not add idempotence flags, retry branches, or a new mega-wiring helper.

- [x] **Step 3: Bind Chat to the existing Sync scope**

Change the `ChatConversationScopeService` construction to:

```python
self.chat_conversation_scope_service = ChatConversationScopeService(
    local_service=self.local_chat_conversation_service,
    server_service=self.server_chat_conversation_service,
    policy_enforcer=self.service_policy_enforcer,
    sync_scope_service=self.sync_scope_service,
)
```

Leave citation repository/migration construction and
`_wire_citation_artifact_ownership()` ordering unchanged.

- [x] **Step 4: Bind Writing to the long-lived provider**

Replace the compatibility constructor and unreachable `ValueError` fallback
with:

```python
self.server_writing_service = (
    ServerWritingService.from_server_context_provider(
        self.server_context_provider,
        policy_enforcer=self.service_policy_enforcer,
    )
)
```

Keep the existing guarded local Writing construction and
`WritingScopeService` composition unchanged. Do not modify or remove
`ServerWritingService.from_config(...)`.

- [x] **Step 5: Run the new source and real-app contracts**

Run:

```bash
../../.venv/bin/python -m pytest Tests/ProductionApp/test_service_composition_lifecycle.py -q
```

Expected: `2 passed`.

- [x] **Step 6: Run the installed-wheel contract**

Run:

```bash
../../.venv/bin/python -m pytest Tests/Packaging/test_installed_distribution.py::test_installed_wheel_loaders_entry_points_and_assets_are_immutable -q
```

Expected: PASS, with the child still loading only from the pip target and the
wheel target hash remaining unchanged.

- [x] **Step 7: Commit the implementation and regression tests**

```bash
git add \
  tldw_chatbook/app.py \
  Tests/ProductionApp/test_service_composition_lifecycle.py \
  Tests/Packaging/test_installed_distribution.py
git commit -m "fix: stabilize application service composition"
```

### Task 4: Verify adjacent contracts and the remaining inventory

**Files:**

- Test: `Tests/ProductionApp/test_service_composition_lifecycle.py`
- Test: `Tests/ProductionApp/test_reactive_ownership_maturity.py`
- Test: `Tests/RuntimePolicy/test_runtime_policy_full_app.py`
- Test: `Tests/Writing_Interop/test_server_writing_service.py`
- Test: `Tests/Chat/test_chat_conversation_scope_service.py`
- Test: `Tests/Chat/test_citation_service_factory.py`
- Test: `Tests/Media/test_media_reading_scope_service.py`
- Test: `Tests/Packaging/test_installed_distribution.py`
- Test: `Packaging/check_manifest.py`

- [x] **Step 1: Prove the no-surrogate policy still covers the new module**

Run:

```bash
../../.venv/bin/python -m pytest \
  Tests/ProductionApp/test_service_composition_lifecycle.py \
  Tests/ProductionApp/test_reactive_ownership_maturity.py::test_production_app_tests_contain_no_surrogate_application_patterns \
  -q
```

Expected: `3 passed`; no allowlist or guard change.

- [x] **Step 2: Verify long-lived runtime-provider behavior**

Run:

```bash
../../.venv/bin/python -m pytest \
  Tests/RuntimePolicy/test_runtime_policy_full_app.py::test_full_app_wires_one_runtime_context_to_long_lived_consumers \
  Tests/RuntimePolicy/test_runtime_policy_full_app.py::test_full_app_wiring_uses_unavailable_store_when_secure_store_is_missing \
  Tests/Writing_Interop/test_server_writing_service.py::test_server_writing_service_from_server_context_provider_is_lazy \
  Tests/Writing_Interop/test_server_writing_service.py::test_server_writing_service_re_resolves_provider_without_service_local_client_cache \
  -q
```

Expected: `4 passed`. The private provider-rewiring test is retained as an
existing focused behavior; it is not evidence that the entire application
graph is reentrant.

- [x] **Step 3: Verify direct Chat, Media, and citation contracts**

Run:

```bash
../../.venv/bin/python -m pytest \
  Tests/Chat/test_chat_conversation_scope_service.py::test_scope_service_routes_chat_metadata_sync_mirror_report_to_sync_scope \
  Tests/Media/test_media_reading_scope_service.py::test_media_scope_service_routes_sync_mirror_report_to_sync_scope \
  Tests/Chat/test_citation_service_factory.py \
  -q
```

Expected: all selected tests PASS. These are direct service/factory tests, not
application substitutes; the production citation ownership evidence remains
in the new full-app test.

- [x] **Step 4: Run all ProductionApp and Packaging tests**

Run:

```bash
../../.venv/bin/python -m pytest Tests/ProductionApp Tests/Packaging -q
```

Expected: all tests PASS. Preserve the exact count and duration in the plan's
implementation evidence.

- [x] **Step 5: Reconcile verified current-dev test and sentinel drift**

The rebased full-suite runs exposed contracts already stale on clean
`origin/dev`. Reconcile only the verified failures:

- remove references to retired `StreamDone`, `TabState`, and root-owned
  streaming behavior without restoring them or adding a test application;
- patch provider tests through `get_runtime_config_snapshot()` rather than
  deleted module-level `settings`;
- create fixture-owned trusted directories before constructing services or
  retargeting the real config loader;
- wait for dynamic Textual controls to have a positive rendered region before
  clicking them in the real production app;
- review the semantic diagnostic-call delta and persistent sink topology from
  the last inventory update, verify the metadata-only sink boundary, then
  regenerate the checked inventory.

Each failure must first reproduce on a detached worktree at current
`origin/dev`. Do not weaken production path validation, restore retired worker
state, or retain an obsolete app-shaped streaming suite.

- [ ] **Step 6: Run the repository test suite for regression safety**

Run:

```bash
../../.venv/bin/python -m pytest -q
```

Expected: all tests PASS. This is general regression evidence required by the
repository Definition of Done; it does not replace the focused full-app and
installed-wheel architecture evidence above.

- [x] **Step 7: Record the executable legacy-provider inventory**

Run:

```bash
../../.venv/bin/python - <<'PY'
import ast
from pathlib import Path

path = Path("tldw_chatbook/app.py")
tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
calls = sorted(
    (node.lineno, node.func.value.id)
    for node in ast.walk(tree)
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "from_config"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id.startswith("Server")
        and node.func.value.id.endswith("Service")
    )
)
print(f"executable Server*Service.from_config calls: {len(calls)}")
for line, name in calls:
    print(f"{line}: {name}.from_config")
PY
```

Expected after the repair: `32` executable calls and no
`ServerWritingService.from_config` entry. Record the list as a follow-up
inventory only; do not claim that the remaining services share Writing's
provider or shutdown semantics.

- [ ] **Step 8: Run compile and scoped Ruff checks**

Run:

```bash
../../.venv/bin/python -m compileall \
  tldw_chatbook/app.py \
  Tests/ProductionApp/test_service_composition_lifecycle.py \
  Tests/Packaging/test_installed_distribution.py
../../.venv/bin/ruff check \
  tldw_chatbook/app.py \
  Tests/ProductionApp/test_service_composition_lifecycle.py \
  Tests/Packaging/test_installed_distribution.py
../../.venv/bin/ruff format --check \
  tldw_chatbook/app.py \
  Tests/ProductionApp/test_service_composition_lifecycle.py \
  Tests/Packaging/test_installed_distribution.py
git diff --check
```

Expected: every command exits `0`.

- [ ] **Step 9: Self-review the complete diff**

Run:

```bash
git diff --stat origin/dev...HEAD
git diff origin/dev...HEAD -- \
  tldw_chatbook/app.py \
  Tests/ProductionApp/test_service_composition_lifecycle.py \
  Tests/Packaging/test_installed_distribution.py \
  Docs/superpowers/specs/2026-07-28-application-service-composition-lifecycle-design.md \
  Docs/superpowers/plans/2026-07-28-application-service-composition-lifecycle.md \
  backlog/decisions/036-application-service-composition-lifecycle.md \
  "backlog/tasks/task-1538 - Enforce-single-pass-service-composition-and-runtime-dependency-binding.md"
```

Confirm:

- the first dependency-ready Writing and Chat calls remain;
- only the later duplicate calls were removed;
- all dependency assertions use exact identity (`is`);
- no test surrogate or app-shaped substitute was introduced;
- no public compatibility API was removed;
- no container, reentrancy, generic lifecycle, or global provider-closure
  claim entered the change;
- the installed child still executes outside the checkout/build source and
  verifies the unchanged target hash.

### Task 5: Close the specification and Backlog record

**Files:**

- Modify: `Docs/superpowers/specs/2026-07-28-application-service-composition-lifecycle-design.md`
- Modify: `Docs/superpowers/plans/2026-07-28-application-service-composition-lifecycle.md`
- Modify: `backlog/tasks/task-1538 - Enforce-single-pass-service-composition-and-runtime-dependency-binding.md`

- [ ] **Step 1: Record implementation evidence**

Append a concise implementation-evidence section to the design and this plan
containing:

- the implementation commit;
- the exact focused and aggregate pytest results;
- the explicit sdist/wheel/runtime-checker result for the packaged
  citation-provenance and character-authority migrations;
- compile/Ruff/diff-check results;
- the post-change count of 32 executable
  `Server*Service.from_config(...)` app calls;
- the explicit statement that the remaining provider inventory and private
  provider-wiring reentrancy are follow-up work.

- [ ] **Step 2: Complete the Backlog task without overstating scope**

Use the Backlog CLI to:

1. check all six acceptance criteria;
2. add `## Implementation Notes` summarizing the single-pass repair, provider
   and Sync binding, production/installed tests, and ADR-036;
3. record the verification commands/results and the 32-call remaining
   inventory;
4. set TASK-1538 to Done only after all tests and static checks are green.

Do not hand-edit generated Backlog front matter and do not mark Done before
the Definition of Done is actually satisfied.

- [ ] **Step 3: Commit closeout documentation**

```bash
git add \
  Docs/superpowers/specs/2026-07-28-application-service-composition-lifecycle-design.md \
  Docs/superpowers/plans/2026-07-28-application-service-composition-lifecycle.md \
  "backlog/tasks/task-1538 - Enforce-single-pass-service-composition-and-runtime-dependency-binding.md"
git commit -m "docs: close TASK-1538 service composition lifecycle"
```

### Task 6: Rebase, review the PR, and merge

**Files:**

- No planned source changes; any review-driven change must remain within
  TASK-1538 acceptance criteria or update the task before implementation.

- [ ] **Step 1: Rebase onto the latest remote development branch**

```bash
git fetch origin
git rebase origin/dev
```

Resolve conflicts semantically. Re-run the Task 4 focused and aggregate checks
after any conflict resolution. Do not preserve stale line placement merely to
make a mechanical rebase succeed.

- [ ] **Step 2: Verify the rebased head**

Repeat:

```bash
../../.venv/bin/python -m pytest Tests/ProductionApp Tests/Packaging -q
../../.venv/bin/python -m pytest \
  Tests/RuntimePolicy/test_runtime_policy_full_app.py::test_full_app_wires_one_runtime_context_to_long_lived_consumers \
  Tests/RuntimePolicy/test_runtime_policy_full_app.py::test_full_app_wiring_uses_unavailable_store_when_secure_store_is_missing \
  Tests/Writing_Interop/test_server_writing_service.py::test_server_writing_service_from_server_context_provider_is_lazy \
  Tests/Writing_Interop/test_server_writing_service.py::test_server_writing_service_re_resolves_provider_without_service_local_client_cache \
  Tests/Chat/test_chat_conversation_scope_service.py::test_scope_service_routes_chat_metadata_sync_mirror_report_to_sync_scope \
  Tests/Media/test_media_reading_scope_service.py::test_media_scope_service_routes_sync_mirror_report_to_sync_scope \
  Tests/Chat/test_citation_service_factory.py \
  -q
../../.venv/bin/ruff check \
  tldw_chatbook/app.py \
  Tests/ProductionApp/test_service_composition_lifecycle.py \
  Tests/Packaging/test_installed_distribution.py
../../.venv/bin/ruff format --check \
  tldw_chatbook/app.py \
  Tests/ProductionApp/test_service_composition_lifecycle.py \
  Tests/Packaging/test_installed_distribution.py
git diff --check origin/dev...HEAD
```

Expected: all commands exit `0`.

- [ ] **Step 3: Push and open a ready PR**

Push `codex/task-1538-service-composition` and create a non-draft pull request
against `dev`. The PR body must link TASK-1538 and ADR-036, summarize the
verified defect and bounded repair, list the real-production-app and
installed-wheel evidence, and state the 32-call legacy-provider follow-up
inventory.

- [ ] **Step 4: Address every review thread and required check**

Inspect all top-level comments, inline review threads, review decisions, and
required CI checks. For each comment:

1. verify it against the current rebased code;
2. implement only technically valid, in-scope corrections;
3. add or adjust a real-app/direct-function test before behavior changes;
4. reply with the exact commit/evidence;
5. resolve the thread only after the pushed fix is visible.

If a requested change expands TASK-1538, update its acceptance criteria before
implementation or create a separate Backlog task.

- [ ] **Step 5: Rebase once more and merge**

Fetch and rebase onto the latest `origin/dev` again after review. Re-run the
required checks, force-push with lease if the rebase changed commits, wait for
required CI/reviews to be green, then merge the PR into `dev`. Confirm the
merge commit is reachable from the latest remote `dev` and report the merged
PR URL and commit.
