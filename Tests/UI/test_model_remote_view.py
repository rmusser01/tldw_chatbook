"""Focused behavior tests for lazy managed Remote model discovery (TASK-1914).

``RemoteView`` also has a discovery-flow-only concern set: metadata search
and repository resolution (``_search_remote``/``_resolve_remote``) stay
owned by this view -- a read-only listing concern, mirroring
``CuratedView._load_curated``, which TASK-1803 also left in place. Most of
this file's coverage of that half is unchanged by TASK-1914.

TASK-1914 moved this view's preflight/provision workers to ``LLMScreen``,
mirroring TASK-1803's move of the equivalent ``CuratedView`` workers.
Tests that used to drive this view's own ``_preflight_model``/
``_confirm_install``/``_apply_preflight_result``/``_provision_model``/
``_apply_provision_result`` directly (the plan-resolution, consent-modal-
push, activation, and failure-logging coverage) moved to
``test_llm_screen_lab_adoption.py``, against ``LLMScreen``, which now owns
that logic; what belongs here instead is ``RemoteView``'s own render-only
contract: confirming a selected candidate posts ``InstallRequested`` and,
once told the outcome, calls ``cancel_pending_install()``/
``finish_install()``/``apply_progress()``.
"""

from __future__ import annotations

import ast
from collections.abc import Callable
from threading import Event
from unittest.mock import MagicMock

import httpx
import pytest
from textual import on

# Harness apps load the consolidated widget CSS the real app loads
# (TASK-15450); without it the widgets under test mount unstyled.
from Tests.UI.consolidated_css import ConsolidatedCSSApp
from textual.app import App, ComposeResult
from textual.css.query import NoMatches
from textual.widgets import Button, Input, Select, Static

from tldw_chatbook.Model_Artifacts.machine_memory import (
    AcceleratorMemoryObservation,
    AcceleratorSource,
    AcceleratorState,
    GIB,
    MachineMemorySnapshot,
    MemoryKind,
    ProbeReason,
    SystemMemoryState,
)
from tldw_chatbook.Model_Artifacts.remote_huggingface import (
    HuggingFaceRemoteAdapter,
    RemoteDiscoveryError,
    RemoteGGUFCandidate,
    RemoteGGUFFile,
    RemoteModelSummary,
    ResolvedRemoteModel,
    build_remote_catalog,
)


_COMMIT = "a" * 40
_DIGEST = "b" * 64


# ---------------------------------------------------------------------------
# Shared AST-based module-scope import check (TASK-1914 fix round 1).
#
# The original version of this check (see git history) scanned for three
# literal substrings in the module's source text. That missed the
# package-then-attribute bypass -- `from tldw_chatbook.Model_Artifacts
# import acquisition` is a real, eager, module-scope import of the
# acquisition runtime, but contains none of the three forbidden substrings
# (no ".acquisition import", no "from .acquisition import", no "import
# tldw_chatbook.Model_Artifacts.acquisition"). This is the exact class of
# gap this workstream's own Task 2 no-subclass test caught and fixed with
# an MRO check instead of a substring scan; the fix here is the same shape
# of fix, applied to imports instead of subclassing.
#
# ``test_model_curated_view.py`` imports this helper rather than
# duplicating it -- both modules are held to the identical rule, and one
# AST walker gets to be the single implementation both tests exercise.
# ---------------------------------------------------------------------------


def _is_type_checking_test(test: ast.expr) -> bool:
    """True for an ``if`` test that is (or ends in) ``TYPE_CHECKING``.

    Covers both ``if TYPE_CHECKING:`` (a bare ``Name``) and
    ``if typing.TYPE_CHECKING:`` (an ``Attribute``) -- this codebase only
    ever uses the former, but the check is cheap to make either way.
    """
    return (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or (
        isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"
    )


def _module_dotted_suffix_is(module: str | None, suffix: str) -> bool:
    """True if ``module`` (an import's dotted path, absolute or relative)
    ends in ``suffix`` as its own final component -- e.g. both
    ``"tldw_chatbook.Model_Artifacts.acquisition"`` and
    ``"Model_Artifacts.acquisition"`` (a relative import's ``module``,
    which never includes the leading dots ``ast`` strips into ``level``)
    end in ``"acquisition"``, but ``"acquisition_helpers"`` does not.
    """
    if not module:
        return False
    return module.rsplit(".", 1)[-1] == suffix


def module_scope_forbidden_acquisition_imports(source: str) -> list[str]:
    """Find real, module-scope imports of ``Model_Artifacts.acquisition``/``.fetch``.

    "Module scope" here means reachable by simply importing the module --
    NOT nested inside any function/method body (those run lazily, on
    demand, which is exactly what the "acquisition/fetch only inside
    functions" rule requires) and NOT inside an ``if TYPE_CHECKING:``
    guard (``False`` at runtime, so that branch never executes).

    Catches both import forms a violation could take:

    - ``from tldw_chatbook.Model_Artifacts.acquisition import X`` (or the
      relative ``from ...Model_Artifacts.acquisition import X``) -- the
      import's own ``module`` ends in ``"acquisition"``/``"fetch"``.
    - ``from tldw_chatbook.Model_Artifacts import acquisition`` -- the
      package-then-attribute bypass a plain substring scan on import text
      misses entirely: ``module`` ends in ``"Model_Artifacts"``, but one
      of the imported *names* is ``"acquisition"``/``"fetch"``.

    Args:
        source: The module's full source text (e.g. from
            ``inspect.getsource``).

    Returns:
        Human-readable descriptions of every forbidden import found, one
        per finding; empty when the module is clean.
    """
    tree = ast.parse(source)
    findings: list[str] = []

    def visit(node: ast.AST, in_function: bool, in_type_checking: bool) -> None:
        for child in ast.iter_child_nodes(node):
            child_in_function = in_function or isinstance(
                child, (ast.FunctionDef, ast.AsyncFunctionDef)
            )
            child_in_type_checking = in_type_checking or (
                isinstance(child, ast.If) and _is_type_checking_test(child.test)
            )
            if (
                isinstance(child, (ast.Import, ast.ImportFrom))
                and not in_function
                and not in_type_checking
            ):
                if isinstance(child, ast.Import):
                    for alias in child.names:
                        if _module_dotted_suffix_is(
                            alias.name, "acquisition"
                        ) or _module_dotted_suffix_is(alias.name, "fetch"):
                            findings.append(f"line {child.lineno}: import {alias.name}")
                else:
                    module = child.module
                    if _module_dotted_suffix_is(
                        module, "acquisition"
                    ) or _module_dotted_suffix_is(module, "fetch"):
                        findings.append(
                            f"line {child.lineno}: from {module!r} import ..."
                        )
                    elif _module_dotted_suffix_is(module, "Model_Artifacts"):
                        for alias in child.names:
                            if alias.name in {"acquisition", "fetch"}:
                                findings.append(
                                    f"line {child.lineno}: from {module!r} "
                                    f"import {alias.name}"
                                )
            visit(child, child_in_function, child_in_type_checking)

    visit(tree, False, False)
    return findings


class _Resolver:
    def __init__(
        self, calls: list[str], token: str | None = "configured-token"
    ) -> None:
        self.calls = calls
        self.token = token

    def resolve(self, repository: str) -> str | None:
        self.calls.append(repository)
        return self.token


class _Adapter:
    def __init__(
        self,
        *,
        search_result: tuple[RemoteModelSummary, ...] = (),
        resolved: ResolvedRemoteModel | None = None,
    ) -> None:
        self.search_result = search_result
        self.resolved = resolved or _resolved()
        self.search_calls: list[tuple[str, str | None]] = []
        self.resolve_calls: list[tuple[str, str | None]] = []

    async def search(
        self,
        query: str,
        *,
        token: str | None = None,
    ) -> tuple[RemoteModelSummary, ...]:
        self.search_calls.append((query, token))
        return self.search_result

    async def resolve(
        self,
        repository: str,
        *,
        token: str | None = None,
    ) -> ResolvedRemoteModel:
        self.resolve_calls.append((repository, token))
        return self.resolved


def _summary(repository: str = "owner/repository") -> RemoteModelSummary:
    return RemoteModelSummary(
        repository=repository,
        private=False,
        gated="none",
        downloads=12,
        likes=3,
        last_modified="2026-08-01T00:00:00Z",
    )


def _candidate(label: str = "owner/repository · model-q4.gguf") -> RemoteGGUFCandidate:
    return RemoteGGUFCandidate(
        label=label,
        files=(RemoteGGUFFile("model-q4.gguf", 1024, _DIGEST),),
        total_bytes=1024,
    )


def _resolved(
    repository: str = "owner/repository",
    *,
    license_id: str = "apache-2.0",
    warnings: tuple[str, ...] = (),
) -> ResolvedRemoteModel:
    return ResolvedRemoteModel(
        repository=repository,
        commit=_COMMIT,
        license_id=license_id,
        review_url=f"https://huggingface.co/{repository}/tree/{_COMMIT}",
        candidates=(_candidate(f"{repository} · model-q4.gguf"),),
        total_candidate_count=1,
        warnings=warnings,
    )


def _memory_resolved() -> ResolvedRemoteModel:
    """Return one realistic 4 GiB candidate for scenario/pressure rendering."""
    candidate = RemoteGGUFCandidate(
        label="owner/repository · model-q4.gguf",
        files=(RemoteGGUFFile("model-q4.gguf", 4 * GIB, _DIGEST),),
        total_bytes=4 * GIB,
    )
    return ResolvedRemoteModel(
        repository="owner/repository",
        commit=_COMMIT,
        license_id="apache-2.0",
        review_url=f"https://huggingface.co/owner/repository/tree/{_COMMIT}",
        candidates=(candidate,),
        total_candidate_count=1,
        warnings=(),
    )


def _variant_resolved() -> ResolvedRemoteModel:
    """Return deliberately unsorted known and unknown variant filenames."""
    candidates = tuple(
        RemoteGGUFCandidate(
            label=f"owner/repository · {filename}",
            files=(RemoteGGUFFile(filename, size, _DIGEST),),
            total_bytes=size,
        )
        for filename, size in (
            ("model-Q8_0.gguf", 80 * 1024 * 1024),
            ("model-Q4_K_M.gguf", 40 * 1024 * 1024),
            ("experimental.gguf", 60 * 1024 * 1024),
        )
    )
    return ResolvedRemoteModel(
        repository="owner/repository",
        commit=_COMMIT,
        license_id="apache-2.0",
        review_url=(f"https://huggingface.co/owner/repository/tree/{_COMMIT}"),
        candidates=candidates,
        total_candidate_count=len(candidates),
        warnings=(),
    )


def _catalog(*, license_id: str = "apache-2.0"):
    resolved = _resolved(license_id=license_id)
    return build_remote_catalog(resolved, resolved.candidates[0])


class _RemoteApp(ConsolidatedCSSApp):
    def __init__(self, view) -> None:
        self.view = view
        super().__init__()

    def compose(self) -> ComposeResult:
        yield self.view


def _view(
    *,
    adapter_factory: Callable[[], object],
    resolver_factory: Callable[[], object] | None = None,
    service_factory: Callable[[], object] = MagicMock,
):
    from tldw_chatbook.UI.Screens.model_remote_view import RemoteView

    kwargs: dict[str, object] = {
        "adapter_factory": adapter_factory,
        "service_factory": service_factory,
    }
    if resolver_factory is not None:
        kwargs["credential_resolver_factory"] = resolver_factory
    return RemoteView(**kwargs)


async def _submit(app: _RemoteApp, pilot, query: str) -> None:
    app.view.query_one("#remote-model-query", Input).value = query
    await pilot.click("#remote-model-search")
    await app.workers.wait_for_complete()
    await pilot.pause()


def _text(view) -> str:
    return "\n".join(str(item.renderable) for item in view.query(Static))


def _memory_snapshot(
    *,
    total_gib: int = 32,
    available_gib: int | None = 10,
    system_state: SystemMemoryState = SystemMemoryState.OBSERVED,
    system_reason: ProbeReason | None = None,
    accelerator_state: AcceleratorState = AcceleratorState.NOT_OBSERVED,
    accelerator_reason: ProbeReason | None = None,
    accelerators: tuple[AcceleratorMemoryObservation, ...] = (),
    memory_kind: MemoryKind = MemoryKind.SYSTEM,
    platform: str = "linux",
    architecture: str = "x86_64",
) -> MachineMemorySnapshot:
    """Build complete trusted memory evidence for mounted Remote tests."""
    has_capacity = system_state in {
        SystemMemoryState.OBSERVED,
        SystemMemoryState.PARTIAL,
    }
    return MachineMemorySnapshot(
        platform=platform,
        architecture=architecture,
        system_state=system_state,
        accelerator_state=accelerator_state,
        total_bytes=total_gib * GIB if has_capacity else None,
        available_bytes=(
            available_gib * GIB if has_capacity and available_gib is not None else None
        ),
        memory_kind=memory_kind if has_capacity else MemoryKind.UNKNOWN,
        accelerators=accelerators,
        system_reason=system_reason,
        accelerator_reason=accelerator_reason,
    )


def _memory_presentation(
    snapshot: MachineMemorySnapshot | None,
    *,
    active: bool = False,
    observed_at_label: str | None = None,
    failure: ProbeReason | None = None,
):
    from tldw_chatbook.UI.Screens.model_memory_presenter import (
        build_machine_memory_presentation,
    )

    return build_machine_memory_presentation(
        snapshot,
        active=active,
        observed_at_label=observed_at_label,
        failure=failure,
    )


@pytest.mark.asyncio
async def test_compose_and_mount_create_no_remote_dependencies_or_io() -> None:
    """An eager parent mount cannot instantiate any I/O-bearing dependency."""
    adapter_factory = MagicMock()
    resolver_factory = MagicMock()
    service_factory = MagicMock()
    view = _view(
        adapter_factory=adapter_factory,
        resolver_factory=resolver_factory,
        service_factory=service_factory,
    )
    app = _RemoteApp(view)

    async with app.run_test() as pilot:
        await pilot.pause()

    adapter_factory.assert_not_called()
    resolver_factory.assert_not_called()
    service_factory.assert_not_called()


@pytest.mark.asyncio
async def test_exact_repository_submission_resolves_without_searching() -> None:
    """Changing exact-ID classification must not add an unnecessary search request."""
    adapter = _Adapter(resolved=_resolved())
    resolver_calls: list[str] = []
    view = _view(
        adapter_factory=lambda: adapter,
        resolver_factory=lambda: _Resolver(resolver_calls),
    )
    app = _RemoteApp(view)

    async with app.run_test() as pilot:
        await _submit(app, pilot, "owner/repository")
        rendered = _text(view)

    assert adapter.search_calls == []
    assert adapter.resolve_calls == [("owner/repository", "configured-token")]
    assert resolver_calls == ["owner/repository"]
    assert "owner/repository" in rendered
    assert "model-q4.gguf" in rendered


@pytest.mark.asyncio
async def test_successful_repository_resolution_requests_machine_memory_once() -> None:
    """Omitting the intent would leave every candidate in an unknown state."""
    from tldw_chatbook.UI.Screens.model_remote_view import RemoteView

    requests: list[bool] = []
    view = _view(
        adapter_factory=lambda: _Adapter(resolved=_memory_resolved()),
        resolver_factory=lambda: _Resolver([]),
    )

    class _App(_RemoteApp):
        @on(RemoteView.MachineMemoryRequested)
        def _capture(self, event: RemoteView.MachineMemoryRequested) -> None:
            requests.append(event.force)

    app = _App(view)
    async with app.run_test(size=(100, 30)) as pilot:
        await _submit(app, pilot, "owner/repository")

    assert requests == [False]


@pytest.mark.asyncio
async def test_machine_memory_update_preserves_candidate_identity_and_focus() -> None:
    """Rebuilding candidate rows on refresh would break keyboard continuity."""
    view = _view(
        adapter_factory=lambda: _Adapter(resolved=_memory_resolved()),
        resolver_factory=lambda: _Resolver([]),
    )
    app = _RemoteApp(view)
    snapshot = _memory_snapshot(total_gib=32, available_gib=10)

    async with app.run_test(size=(100, 30)) as pilot:
        await _submit(app, pilot, "owner/repository")
        candidate = view.query_one(".remote-candidate", Button)
        outcome = view.query_one("#remote-fit-outcome-0", Static)
        details = view.query_one("#remote-fit-details-0", Static)
        advisory = view.query_one("#remote-fit-advisory-0", Static)
        candidate.focus()
        view.apply_machine_memory_state(
            _memory_presentation(snapshot),
            snapshot,
        )
        await pilot.pause()

        assert view.query_one(".remote-candidate", Button) is candidate
        assert view.query_one("#remote-fit-outcome-0", Static) is outcome
        assert view.query_one("#remote-fit-details-0", Static) is details
        assert view.query_one("#remote-fit-advisory-0", Static) is advisory
        assert app.focused is candidate
        assert "64K scenario within RAM budget" in _text(view)
        assert "64K may need more free RAM now" in _text(view)
        assert (
            "model-context support and runtime compatibility remain unverified"
            in _text(view)
        )
        assert outcome._render_markup is False
        assert details._render_markup is False
        assert advisory._render_markup is False


@pytest.mark.asyncio
async def test_memory_scenario_panel_recheck_and_details_toggle_are_explicit() -> None:
    """Losing labeled controls would hide recovery and exact estimate evidence."""
    from tldw_chatbook.UI.Screens.model_remote_view import RemoteView

    requests: list[bool] = []
    snapshot = _memory_snapshot(total_gib=32, available_gib=10)
    view = _view(
        adapter_factory=lambda: _Adapter(resolved=_resolved()),
        resolver_factory=lambda: _Resolver([]),
    )

    class _App(_RemoteApp):
        @on(RemoteView.MachineMemoryRequested)
        def _capture(self, event: RemoteView.MachineMemoryRequested) -> None:
            requests.append(event.force)

    app = _App(view)
    async with app.run_test(size=(100, 30)) as pilot:
        await _submit(app, pilot, "owner/repository")
        view.apply_machine_memory_state(_memory_presentation(snapshot), snapshot)
        await pilot.pause()
        details = view.query_one("#remote-machine-estimate-details", Static)
        toggle = view.query_one("#remote-machine-details-toggle", Button)
        assert details.display is True
        assert str(toggle.label) == "Hide estimate details"
        assert "VRAM not used in this rating" in str(details.renderable)
        assert "VRAM not observed · not used in this rating" in _text(view)

        toggle.press()
        await pilot.pause()
        assert details.display is False
        assert str(toggle.label) == "Show estimate details"

        recheck = view.query_one("#remote-machine-recheck", Button)
        recheck.press()
        await pilot.pause()
        assert requests == [False, True]
        assert str(recheck.label) == "Checking…"
        assert recheck.disabled is True


@pytest.mark.asyncio
async def test_memory_scenario_unavailable_and_retained_failure_copy_is_rendered() -> (
    None
):
    """Flattening failure states would make a retained estimate look freshly observed."""
    view = _view(
        adapter_factory=lambda: _Adapter(resolved=_resolved()),
        resolver_factory=lambda: _Resolver([]),
    )
    app = _RemoteApp(view)
    unavailable = _memory_snapshot(
        system_state=SystemMemoryState.PERMISSION_DENIED,
        system_reason=ProbeReason.PERMISSION_DENIED,
        available_gib=None,
    )

    async with app.run_test(size=(100, 30)) as pilot:
        await _submit(app, pilot, "owner/repository")
        view.apply_machine_memory_state(
            _memory_presentation(unavailable),
            unavailable,
        )
        await pilot.pause()
        assert "Memory access was denied · filename guidance still applies" in _text(
            view
        )
        assert "Memory estimate unavailable · memory access denied" in _text(view)
        recheck = view.query_one("#remote-machine-recheck", Button)
        recheck.press()
        await pilot.pause()
        assert str(recheck.label) == "Checking…"
        assert recheck.disabled is True

        accepted = _memory_snapshot()
        view.apply_machine_memory_state(
            _memory_presentation(
                accepted,
                observed_at_label="09:41",
                failure=ProbeReason.MEMORY_UNAVAILABLE,
            ),
            accepted,
        )
        await pilot.pause()
        assert "Recheck failed · using memory observed at 09:41" in _text(view)


@pytest.mark.asyncio
async def test_memory_scenario_accelerator_details_keep_all_bounded_labels() -> None:
    """Capping compact copy must not make bounded device facts inaccessible."""
    devices = tuple(
        AcceleratorMemoryObservation(
            vendor="nvidia",
            label=("長い GPU مثال " + str(index)).ljust(96, "x"),
            total_bytes=(index + 1) * GIB,
            shared=False,
            source=AcceleratorSource.NVIDIA_SMI,
        )
        for index in range(3)
    )
    snapshot = _memory_snapshot(
        accelerator_state=AcceleratorState.OBSERVED,
        accelerators=devices,
    )
    view = _view(
        adapter_factory=lambda: _Adapter(resolved=_resolved()),
        resolver_factory=lambda: _Resolver([]),
    )
    app = _RemoteApp(view)

    async with app.run_test(size=(100, 40)) as pilot:
        await _submit(app, pilot, "owner/repository")
        view.apply_machine_memory_state(_memory_presentation(snapshot), snapshot)
        await pilot.pause()
        panel = view.query_one("#remote-machine-evidence", Static)
        details = view.query_one("#remote-machine-estimate-details", Static)

        assert "VRAM observed on 3 devices · show estimate details" in str(
            panel.renderable
        )
        assert all(device.label in str(details.renderable) for device in devices)
        assert panel._render_markup is False
        assert details._render_markup is False


@pytest.mark.asyncio
@pytest.mark.parametrize("device_count", [2, 16])
async def test_memory_scenario_two_and_sixteen_accelerators_remain_inspectable(
    device_count: int,
) -> None:
    """Changing compact/detail thresholds must not drop a bounded device fact."""
    devices = tuple(
        AcceleratorMemoryObservation(
            vendor="nvidia",
            label=f"GPU {index}",
            total_bytes=8 * GIB,
            shared=False,
            source=AcceleratorSource.NVIDIA_SMI,
        )
        for index in range(device_count)
    )
    snapshot = _memory_snapshot(
        accelerator_state=AcceleratorState.OBSERVED,
        accelerators=devices,
    )
    view = _view(
        adapter_factory=lambda: _Adapter(resolved=_resolved()),
        resolver_factory=lambda: _Resolver([]),
    )
    app = _RemoteApp(view)

    async with app.run_test(size=(100, 40)) as pilot:
        await _submit(app, pilot, "owner/repository")
        view.apply_machine_memory_state(_memory_presentation(snapshot), snapshot)
        await pilot.pause()
        compact = str(view.query_one("#remote-machine-evidence", Static).renderable)
        expanded = str(
            view.query_one("#remote-machine-estimate-details", Static).renderable
        )

    if device_count == 2:
        assert "NVIDIA GPU 0 8.0 GiB" in compact
        assert "NVIDIA GPU 1 8.0 GiB" in compact
    else:
        assert "VRAM observed on 16 devices · show estimate details" in compact
        assert "NVIDIA GPU 15 8.0 GiB" not in compact
    assert all(device.label in expanded for device in devices)


@pytest.mark.asyncio
async def test_drill_down_at_71_cells_restores_exact_repository_focus() -> None:
    """Using two squeezed panes or rebuilding results would break narrow browsing."""
    adapter = _Adapter(
        search_result=(_summary(),),
        resolved=_memory_resolved(),
    )
    view = _view(
        adapter_factory=lambda: adapter,
        resolver_factory=lambda: _Resolver([]),
    )
    app = _RemoteApp(view)

    async with app.run_test(size=(71, 30)) as pilot:
        await _submit(app, pilot, "test model")
        results_pane = view.query_one(".remote-results-pane")
        detail_pane = view.query_one(".remote-detail-pane")
        repository = view.query_one(".remote-result", Button)
        assert view.has_class("-single-pane")
        assert results_pane.display is True
        assert detail_pane.display is False

        repository.focus()
        repository.press()
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert results_pane.display is False
        assert detail_pane.display is True
        back = view.query_one("#remote-back-to-results", Button)
        assert str(back.label) == "Back to repositories"

        back.press()
        await pilot.pause()
        assert results_pane.display is True
        assert detail_pane.display is False
        assert view.query_one(".remote-result", Button) is repository
        assert app.focused is repository


@pytest.mark.asyncio
async def test_drill_down_new_search_returns_to_results_and_details_start_collapsed() -> (
    None
):
    """Retaining detail on a new narrow search would hide the new result set."""
    adapter = _Adapter(
        search_result=(_summary(),),
        resolved=_memory_resolved(),
    )
    view = _view(
        adapter_factory=lambda: adapter,
        resolver_factory=lambda: _Resolver([]),
    )
    app = _RemoteApp(view)
    snapshot = _memory_snapshot()

    async with app.run_test(size=(71, 35)) as pilot:
        await _submit(app, pilot, "test model")
        view.query_one(".remote-result", Button).press()
        await app.workers.wait_for_complete()
        await pilot.pause()
        view.apply_machine_memory_state(_memory_presentation(snapshot), snapshot)
        await pilot.pause()
        details = view.query_one("#remote-machine-estimate-details", Static)
        toggle = view.query_one("#remote-machine-details-toggle", Button)
        assert details.display is False
        assert str(toggle.label) == "Show estimate details"

        query = view.query_one("#remote-model-query", Input)
        query.value = "another model"
        view.query_one("#remote-model-search", Button).press()
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert view.query_one(".remote-results-pane").display is True
        assert view.query_one(".remote-detail-pane").display is False


@pytest.mark.asyncio
async def test_memory_scenario_pressure_advisory_stays_visible_when_71_details_collapse() -> (
    None
):
    """Collapsing exact inputs must not hide a current free-memory warning."""
    view = _view(
        adapter_factory=lambda: _Adapter(resolved=_memory_resolved()),
        resolver_factory=lambda: _Resolver([]),
    )
    app = _RemoteApp(view)
    snapshot = _memory_snapshot(available_gib=10)

    async with app.run_test(size=(71, 35)) as pilot:
        await _submit(app, pilot, "owner/repository")
        view.apply_machine_memory_state(_memory_presentation(snapshot), snapshot)
        await pilot.pause()
        exact_inputs = view.query_one("#remote-fit-details-0", Static)
        advisory = view.query_one("#remote-fit-advisory-0", Static)

        assert exact_inputs.display is False
        assert advisory.display is True
        assert "64K may need more free RAM now" in str(advisory.renderable)


@pytest.mark.asyncio
async def test_memory_scenario_retained_failure_stays_visible_when_71_details_collapse() -> (
    None
):
    """Collapsing exact inputs must not hide that a refresh used retained facts."""
    view = _view(
        adapter_factory=lambda: _Adapter(resolved=_memory_resolved()),
        resolver_factory=lambda: _Resolver([]),
    )
    app = _RemoteApp(view)
    snapshot = _memory_snapshot(available_gib=32)

    async with app.run_test(size=(71, 35)) as pilot:
        await _submit(app, pilot, "owner/repository")
        view.apply_machine_memory_state(
            _memory_presentation(
                snapshot,
                observed_at_label="09:41",
                failure=ProbeReason.MEMORY_UNAVAILABLE,
            ),
            snapshot,
        )
        await pilot.pause()
        exact_inputs = view.query_one("#remote-fit-details-0", Static)
        advisory = view.query_one("#remote-fit-advisory-0", Static)

        assert exact_inputs.display is False
        assert advisory.display is True
        assert "Recheck failed · using memory observed at 09:41" in str(
            advisory.renderable
        )


@pytest.mark.asyncio
async def test_memory_scenario_new_71_search_resets_expanded_details_to_collapsed() -> (
    None
):
    """A prior repository's toggle choice must not expand a new narrow result."""
    adapter = _Adapter(
        search_result=(_summary(),),
        resolved=_memory_resolved(),
    )
    view = _view(
        adapter_factory=lambda: adapter,
        resolver_factory=lambda: _Resolver([]),
    )
    app = _RemoteApp(view)
    snapshot = _memory_snapshot()

    async with app.run_test(size=(71, 40)) as pilot:
        await _submit(app, pilot, "first model")
        view.query_one(".remote-result", Button).press()
        await app.workers.wait_for_complete()
        await pilot.pause()
        view.apply_machine_memory_state(_memory_presentation(snapshot), snapshot)
        await pilot.pause()
        toggle = view.query_one("#remote-machine-details-toggle", Button)
        toggle.press()
        await pilot.pause()
        assert view.query_one("#remote-fit-details-0", Static).display is True

        query = view.query_one("#remote-model-query", Input)
        query.value = "second model"
        view.query_one("#remote-model-search", Button).press()
        await app.workers.wait_for_complete()
        await pilot.pause()
        view.query_one(".remote-result", Button).press()
        await app.workers.wait_for_complete()
        await pilot.pause()

        assert view.query_one("#remote-fit-details-0", Static).display is False
        assert (
            str(view.query_one("#remote-machine-details-toggle", Button).label)
            == "Show estimate details"
        )


@pytest.mark.asyncio
async def test_drill_down_controls_are_reachable_by_tab() -> None:
    """A narrow-only display rule must not remove estimate or install controls."""
    adapter = _Adapter(
        search_result=(_summary(),),
        resolved=_memory_resolved(),
    )
    view = _view(
        adapter_factory=lambda: adapter,
        resolver_factory=lambda: _Resolver([]),
    )
    app = _RemoteApp(view)
    snapshot = _memory_snapshot()

    async with app.run_test(size=(71, 40)) as pilot:
        await _submit(app, pilot, "test model")
        view.query_one(".remote-result", Button).press()
        await app.workers.wait_for_complete()
        await pilot.pause()
        view.apply_machine_memory_state(_memory_presentation(snapshot), snapshot)
        await pilot.pause()
        view.query_one(".remote-candidate", Button).press()
        await pilot.pause()

        seen_ids: set[str] = set()
        saw_candidate = False
        for _ in range(18):
            await pilot.press("tab")
            focused = app.focused
            if focused is None:
                continue
            if focused.id is not None:
                seen_ids.add(focused.id)
            saw_candidate |= focused.has_class("remote-candidate")

        assert {
            "remote-back-to-results",
            "remote-machine-recheck",
            "remote-machine-details-toggle",
            "remote-variant-filter",
            "remote-variant-sort",
            "remote-model-install",
        } <= seen_ids
        assert saw_candidate


@pytest.mark.asyncio
async def test_two_pane_layout_starts_at_72_cells_with_details_expanded() -> None:
    """Moving the exact breakpoint would regress the approved 72-cell contract."""
    view = _view(
        adapter_factory=lambda: _Adapter(resolved=_memory_resolved()),
        resolver_factory=lambda: _Resolver([]),
    )
    app = _RemoteApp(view)
    snapshot = _memory_snapshot()

    async with app.run_test(size=(72, 35)) as pilot:
        await _submit(app, pilot, "owner/repository")
        view.apply_machine_memory_state(_memory_presentation(snapshot), snapshot)
        await pilot.pause()

        assert not view.has_class("-single-pane")
        assert view.query_one(".remote-results-pane").display is True
        assert view.query_one(".remote-detail-pane").display is True
        assert (
            view.query_one("#remote-machine-estimate-details", Static).display is True
        )
        assert (
            str(view.query_one("#remote-machine-details-toggle", Button).label)
            == "Hide estimate details"
        )


@pytest.mark.asyncio
async def test_no_user_visible_string_contains_artifact() -> None:
    """No rendered Static text says "artifact" -- the UI says "model" throughout."""
    adapter = _Adapter(resolved=_resolved())
    view = _view(
        adapter_factory=lambda: adapter,
        resolver_factory=lambda: _Resolver([]),
    )
    app = _RemoteApp(view)

    async with app.run_test() as pilot:
        await _submit(app, pilot, "owner/repository")
        rendered = _text(view)

    assert "artifact" not in rendered.lower()


@pytest.mark.asyncio
async def test_free_text_search_resolves_only_after_result_selection() -> None:
    """Free text must remain a search and selection must resolve that exact result."""
    adapter = _Adapter(search_result=(_summary(),), resolved=_resolved())
    resolver_calls: list[str] = []
    view = _view(
        adapter_factory=lambda: adapter,
        resolver_factory=lambda: _Resolver(resolver_calls),
    )
    app = _RemoteApp(view)

    async with app.run_test() as pilot:
        await _submit(app, pilot, "quantized model")
        assert adapter.search_calls == [("quantized model", "configured-token")]
        assert adapter.resolve_calls == []
        assert "owner/repository" in _text(view)

        await pilot.click(".remote-result")
        await app.workers.wait_for_complete()
        await pilot.pause()
        rendered = _text(view)

    assert adapter.resolve_calls == [("owner/repository", "configured-token")]
    assert resolver_calls == ["quantized model", "owner/repository"]
    assert "Runtime compatibility has not been verified." in rendered
    assert "Local integrity recorded" in rendered


@pytest.mark.asyncio
async def test_repository_selection_keeps_results_and_metadata_visible_in_two_panes() -> (
    None
):
    """Collapsing discovery into the selected model must not destroy browse context."""
    selected = RemoteModelSummary(
        repository="owner/repository",
        private=False,
        gated="none",
        downloads=12_400,
        likes=81,
        last_modified="2026-08-18T14:05:00Z",
    )
    other = RemoteModelSummary(
        repository="other/repository",
        private=True,
        gated="manual",
        downloads=None,
        likes=None,
        last_modified=None,
    )
    adapter = _Adapter(search_result=(selected, other), resolved=_resolved())
    view = _view(
        adapter_factory=lambda: adapter,
        resolver_factory=lambda: _Resolver([]),
    )
    app = _RemoteApp(view)

    async with app.run_test(size=(100, 36)) as pilot:
        await _submit(app, pilot, "quantized model")
        result_buttons = list(view.query(".remote-result").results(Button))
        assert len(result_buttons) == 2

        result_buttons[0].press()
        await app.workers.wait_for_complete()
        await pilot.pause()

        rendered = _text(view)
        result_buttons = list(view.query(".remote-result").results(Button))
        candidate_buttons = list(view.query(".remote-candidate").results(Button))
        result_x = result_buttons[0].region.x
        candidate_x = candidate_buttons[0].region.x

    assert len(result_buttons) == 2
    assert len(candidate_buttons) == 1
    assert "owner/repository" in rendered
    assert "other/repository" in rendered
    assert "12.4K downloads · 81 likes" in rendered
    assert "Updated 2026-08-18" in rendered
    assert "Public · Gated: none" in rendered
    assert "Private · Gated: manual" in rendered
    assert result_x < candidate_x


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "last_modified",
    ("updated", "2026-08-18\x1b[31m"),
)
async def test_result_metadata_rejects_malformed_or_nonprintable_update_dates(
    last_modified: str,
) -> None:
    """Untrusted bounded text must not impersonate a normalized update date."""
    summary = RemoteModelSummary(
        repository="owner/repository",
        private=False,
        gated="none",
        downloads=1,
        likes=2,
        last_modified=last_modified,
    )
    adapter = _Adapter(search_result=(summary,), resolved=_resolved())
    view = _view(
        adapter_factory=lambda: adapter,
        resolver_factory=lambda: _Resolver([]),
    )
    app = _RemoteApp(view)

    async with app.run_test() as pilot:
        await _submit(app, pilot, "quantized model")
        rendered = _text(view)

    assert "Updated —" in rendered
    assert last_modified not in rendered


@pytest.mark.asyncio
async def test_repository_selection_preserves_result_focus_while_inspecting() -> None:
    """Opening details must not replace the result row under the keyboard."""
    adapter = _Adapter(search_result=(_summary(),), resolved=_resolved())
    view = _view(
        adapter_factory=lambda: adapter,
        resolver_factory=lambda: _Resolver([]),
    )
    app = _RemoteApp(view)

    async with app.run_test() as pilot:
        await _submit(app, pilot, "quantized model")
        result = view.query_one(".remote-result", Button)
        app.screen.set_focus(result)
        await pilot.pause()
        assert app.focused is result

        result.press()
        await app.workers.wait_for_complete()
        await pilot.pause()

        assert app.focused is result
        assert view.query_one(".remote-candidate", Button).region.x > result.region.x


@pytest.mark.asyncio
async def test_candidate_selection_enables_one_contextual_install_action() -> None:
    """Selecting a file must not start acquisition before the explicit install action."""
    resolved = ResolvedRemoteModel(
        repository="owner/repository",
        commit=_COMMIT,
        license_id="apache-2.0",
        review_url=f"https://huggingface.co/owner/repository/tree/{_COMMIT}",
        candidates=(
            RemoteGGUFCandidate(
                label="owner/repository · model-q5-k-m.gguf",
                files=(
                    RemoteGGUFFile(
                        "model-q5-k-m.gguf",
                        661_191_781,
                        _DIGEST,
                    ),
                ),
                total_bytes=661_191_781,
            ),
        ),
        total_candidate_count=1,
        warnings=(),
    )
    view = _view(adapter_factory=MagicMock(), resolver_factory=MagicMock())
    app = _capturing_app(view)

    async with app.run_test() as pilot:
        query = view.query_one("#remote-model-query", Input)
        query.value = "owner/repository"
        view._resolve_generation = 1
        view._apply_resolve_result(
            1,
            "owner/repository",
            "owner/repository",
            resolved,
            None,
        )
        await pilot.pause()

        candidate_button = view.query_one(".remote-candidate", Button)
        assert str(candidate_button.label) == "Select variant"
        candidate_button.press()
        await pilot.pause()

        assert app.requests == []
        assert str(candidate_button.label) == "Selected variant"
        install = view.query_one("#remote-model-install", Button)
        assert install.disabled is False
        assert "630.6 MiB" in _text(view)

        install.press()
        await pilot.pause()

    assert len(app.requests) == 1
    assert app.requests[0].candidate == resolved.candidates[0]


@pytest.mark.asyncio
async def test_candidate_selection_preserves_keyboard_focus() -> None:
    """Selecting a variant must not replace the focused row under the keyboard."""
    resolved = _resolved()
    view = _view(adapter_factory=MagicMock(), resolver_factory=MagicMock())
    app = _RemoteApp(view)

    async with app.run_test() as pilot:
        query = view.query_one("#remote-model-query", Input)
        query.value = resolved.repository
        view._resolve_generation = 1
        view._apply_resolve_result(
            1,
            resolved.repository,
            resolved.repository,
            resolved,
            None,
        )
        await pilot.pause()

        candidate = view.query_one(".remote-candidate", Button)
        app.screen.set_focus(candidate)
        await pilot.pause()
        assert app.focused is candidate

        candidate.press()
        await pilot.pause()

        assert app.focused is candidate
        assert view.query_one("#remote-model-install", Button).disabled is False


@pytest.mark.asyncio
async def test_variant_rows_explain_filename_derived_guidance_without_fit_claims() -> (
    None
):
    """Every row must expose exact facts while unknown names stay explicitly unknown."""
    resolved = _variant_resolved()
    view = _view(adapter_factory=MagicMock(), resolver_factory=MagicMock())
    app = _RemoteApp(view)

    async with app.run_test() as pilot:
        view.query_one("#remote-model-query", Input).value = resolved.repository
        view._resolve_generation = 1
        view._apply_resolve_result(
            1,
            resolved.repository,
            resolved.repository,
            resolved,
            None,
        )
        await pilot.pause()
        rendered = _text(view)
        filenames = tuple(
            str(widget.renderable)
            for widget in view.query(".remote-variant-filename").results(Static)
        )

    assert filenames == (
        "model-Q8_0.gguf",
        "model-Q4_K_M.gguf",
        "experimental.gguf",
    )
    assert "Filename-derived general guidance" in rendered
    assert (
        "model-context support and runtime compatibility remain unverified" in rendered
    )
    assert "Quantization: Q4_K_M · 1 file · 40.0 MiB" in rendered
    assert "Quantization: Not identified · 1 file · 60.0 MiB" in rendered
    assert "No recognized quantization token in the filename" in rendered
    assert rendered.index("Available GGUF files") < rendered.index("Source review page")


@pytest.mark.asyncio
async def test_variant_rows_use_exact_file_authority_for_long_paths_and_shards() -> (
    None
):
    """Bounded or synthetic candidate labels must never replace exact file paths."""
    long_path = f"nested/{'long-name-' * 18}Q4_K_M.gguf"
    shard_paths = (
        "nested/model-Q5_K_M-00001-of-00002.gguf",
        "nested/model-Q5_K_M-00002-of-00002.gguf",
    )
    candidates = (
        RemoteGGUFCandidate(
            label=f"owner/repository · {long_path}"[:160],
            files=(RemoteGGUFFile(long_path, 40, _DIGEST),),
            total_bytes=40,
        ),
        RemoteGGUFCandidate(
            label="owner/repository · nested/model-Q5_K_M",
            files=tuple(RemoteGGUFFile(path, 50, _DIGEST) for path in shard_paths),
            total_bytes=100,
        ),
    )
    resolved = ResolvedRemoteModel(
        repository="owner/repository",
        commit=_COMMIT,
        license_id="apache-2.0",
        review_url=(f"https://huggingface.co/owner/repository/tree/{_COMMIT}"),
        candidates=candidates,
        total_candidate_count=2,
        warnings=(),
    )
    view = _view(adapter_factory=MagicMock(), resolver_factory=MagicMock())
    app = _RemoteApp(view)

    async with app.run_test() as pilot:
        view.query_one("#remote-model-query", Input).value = resolved.repository
        view._resolve_generation = 1
        view._apply_resolve_result(
            1,
            resolved.repository,
            resolved.repository,
            resolved,
            None,
        )
        await pilot.pause()

        filenames = tuple(
            str(widget.renderable)
            for widget in view.query(".remote-variant-filename").results(Static)
        )
        rendered = _text(view)
        assert filenames == (long_path, shard_paths[0])
        assert "Quantization: Q4_K_M · 1 file" in rendered
        assert "Quantization: Q5_K_M · 2 shards" in rendered

        view.query_one("#remote-variant-filter", Input).value = "00002"
        await pilot.pause()
        visible = list(view.query(".remote-candidate").results(Button))

    assert len(visible) == 1
    assert getattr(visible[0], "candidate") == candidates[1]


@pytest.mark.asyncio
async def test_variant_filter_is_local_and_clears_a_hidden_selection() -> None:
    """Filtering must not fetch again or leave an invisible variant installable."""
    resolved = _variant_resolved()
    adapter = _Adapter(resolved=resolved)
    view = _view(
        adapter_factory=lambda: adapter,
        resolver_factory=lambda: _Resolver([]),
    )
    app = _RemoteApp(view)

    async with app.run_test() as pilot:
        await _submit(app, pilot, resolved.repository)
        candidates = list(view.query(".remote-candidate").results(Button))
        candidates[0].press()
        await pilot.pause()
        assert view.query_one("#remote-model-install", Button).disabled is False

        variant_filter = view.query_one("#remote-variant-filter", Input)
        app.screen.set_focus(variant_filter)
        variant_filter.value = "q4_k_m"
        await pilot.pause()

        visible = list(view.query(".remote-candidate").results(Button))
        assert len(visible) == 1
        assert getattr(visible[0], "candidate") == resolved.candidates[1]
        assert view._selected_candidate is None
        assert view.query_one("#remote-model-install", Button).disabled is True
        assert app.focused is variant_filter
        assert adapter.resolve_calls == [(resolved.repository, "configured-token")]

        variant_filter.value = "does-not-exist"
        await pilot.pause()

        assert "No GGUF variants match this filter" in _text(view)
        assert list(view.query(".remote-candidate").results(Button)) == []
        assert adapter.resolve_calls == [(resolved.repository, "configured-token")]


@pytest.mark.asyncio
async def test_variant_sort_preserves_selection_and_control_focus() -> None:
    """Reordering must retain exact candidate identity and the active control."""
    resolved = _variant_resolved()
    view = _view(adapter_factory=MagicMock(), resolver_factory=MagicMock())
    app = _RemoteApp(view)

    async with app.run_test() as pilot:
        view.query_one("#remote-model-query", Input).value = resolved.repository
        view._resolve_generation = 1
        view._apply_resolve_result(
            1,
            resolved.repository,
            resolved.repository,
            resolved,
            None,
        )
        await pilot.pause()
        selected = resolved.candidates[0]
        view.query_one(".remote-candidate", Button).press()
        await pilot.pause()

        sort = view.query_one("#remote-variant-sort", Select)
        app.screen.set_focus(sort)
        sort.value = "size-asc"
        await pilot.pause()

        filenames = tuple(
            str(widget.renderable)
            for widget in view.query(".remote-variant-filename").results(Static)
        )
        selected_buttons = tuple(
            button
            for button in view.query(".remote-candidate").results(Button)
            if str(button.label) == "Selected variant"
        )

        assert filenames == (
            "model-Q4_K_M.gguf",
            "experimental.gguf",
            "model-Q8_0.gguf",
        )
        assert view._selected_candidate == selected
        assert len(selected_buttons) == 1
        assert getattr(selected_buttons[0], "candidate") == selected
        assert view.query_one("#remote-model-install", Button).disabled is False
        assert app.focused is sort


@pytest.mark.asyncio
async def test_two_pane_layout_and_install_action_paint_at_eighty_columns() -> None:
    """The supported narrow terminal must paint both panes and the final action."""
    adapter = _Adapter(search_result=(_summary(),), resolved=_resolved())
    view = _view(
        adapter_factory=lambda: adapter,
        resolver_factory=lambda: _Resolver([]),
    )
    app = _RemoteApp(view)

    async with app.run_test(size=(80, 24)) as pilot:
        await _submit(app, pilot, "quantized model")
        view.query_one(".remote-result", Button).press()
        await app.workers.wait_for_complete()
        await pilot.pause()

        candidate = view.query_one(".remote-candidate", Button)
        app.screen.set_focus(candidate)
        candidate.scroll_visible(animate=False, immediate=True, force=True)
        await pilot.pause()
        candidate.press()
        await pilot.pause()

        results_pane = view.query_one(".remote-results-pane")
        detail_pane = view.query_one(".remote-detail-pane")
        variant_filter = view.query_one("#remote-variant-filter", Input)
        variant_sort = view.query_one("#remote-variant-sort", Select)
        install = view.query_one("#remote-model-install", Button)
        widget_at_install, _offset = app.get_widget_at(*install.region.center)
        painted = "\n".join(
            "".join(segment.text for segment in strip)
            for strip in app.screen._compositor.render_strips()
        )

        assert results_pane.region.width > 0
        assert detail_pane.region.width > 0
        assert results_pane.region.right <= detail_pane.region.x
        assert variant_filter.region.width > 0
        assert variant_sort.region.width > 0
        assert variant_filter.region.right <= variant_sort.region.x
        assert variant_sort.region.right <= detail_pane.region.right
        assert install.region.width > 0
        assert install.region.bottom <= view.region.bottom
        assert widget_at_install is install
        assert "Repositories" in painted
        assert "Review and install" in painted


@pytest.mark.asyncio
async def test_completion_actions_paint_and_tab_at_eighty_columns() -> None:
    """Both post-download handoffs remain visible and keyboard reachable."""
    from tldw_chatbook.Model_Artifacts.service import ArtifactRef

    resolved = _resolved()
    reference = ArtifactRef("owner-repository", "a" * 40, "q4_k_m")
    view = _view(adapter_factory=MagicMock(), resolver_factory=MagicMock())
    app = _RemoteApp(view)

    async with app.run_test(size=(80, 24)) as pilot:
        view.query_one("#remote-model-query", Input).value = resolved.repository
        view._resolve_generation = 1
        view._apply_resolve_result(
            1,
            resolved.repository,
            resolved.repository,
            resolved,
            None,
        )
        await pilot.pause()
        view.query_one(".remote-candidate", Button).press()
        view.finish_install("done", completed_reference=reference)
        await pilot.pause()

        open_installed = view.query_one("#remote-model-open-installed", Button)
        configure = view.query_one("#remote-model-configure-runtime", Button)
        for button in (open_installed, configure):
            button.scroll_visible(animate=False, immediate=True, force=True)
            await pilot.pause()
            widget, _offset = app.get_widget_at(*button.region.center)
            assert button in app.screen._compositor.visible_widgets
            assert button.region.bottom <= view.region.bottom
            assert widget is button

        app.screen.set_focus(open_installed)
        await pilot.press("tab")
        assert app.focused is configure


@pytest.mark.asyncio
async def test_stale_search_and_resolve_completions_cannot_replace_newer_results() -> (
    None
):
    """Removing either generation check must let an older completion overwrite state."""
    view = _view(adapter_factory=MagicMock(), resolver_factory=MagicMock())
    app = _RemoteApp(view)
    older_summary = _summary("old/result")
    newer_summary = _summary("new/result")
    older_resolved = _resolved("old/result")
    newer_resolved = _resolved("new/result")

    async with app.run_test() as pilot:
        query = view.query_one("#remote-model-query", Input)
        query.value = "new query"
        view._search_generation = 2
        view._apply_search_result(1, "old query", (older_summary,), None)
        view._apply_search_result(2, "new query", (newer_summary,), None)
        await pilot.pause()
        assert "new/result" in _text(view)
        assert "old/result" not in _text(view)

        view._resolve_generation = 4
        view._apply_resolve_result(
            3,
            "old/result",
            "old query",
            older_resolved,
            None,
        )
        view._apply_resolve_result(
            4,
            "new/result",
            "new query",
            newer_resolved,
            None,
        )
        await pilot.pause()
        rendered = _text(view)

    assert "new/result" in rendered
    assert "old/result" not in rendered
    assert "model-q4.gguf" in rendered


@pytest.mark.asyncio
async def test_same_generation_resolve_rejects_a_different_repository_response() -> (
    None
):
    """An adapter identity mismatch must never expose an installable candidate."""
    adapter = _Adapter(resolved=_resolved("other/repository"))
    view = _view(
        adapter_factory=lambda: adapter,
        resolver_factory=lambda: _Resolver([]),
    )
    app = _RemoteApp(view)

    async with app.run_test() as pilot:
        await _submit(app, pilot, "owner/repository")
        rendered = _text(view)
        candidate_buttons = list(view.query(".remote-candidate").results(Button))
        search_disabled = view.query_one("#remote-model-search", Button).disabled

    assert "other/repository" not in rendered
    assert candidate_buttons == []
    assert view._operation_reference is None
    assert search_disabled is False


@pytest.mark.asyncio
async def test_same_generation_resolve_rejects_when_repository_input_changes() -> None:
    """Input drift during one request must not make its completion installable."""
    started = Event()
    release = Event()

    class _WaitingAdapter(_Adapter):
        async def resolve(
            self,
            repository: str,
            *,
            token: str | None = None,
        ) -> ResolvedRemoteModel:
            self.resolve_calls.append((repository, token))
            started.set()
            release.wait(timeout=2)
            return self.resolved

    adapter = _WaitingAdapter(resolved=_resolved("owner/repository"))
    view = _view(
        adapter_factory=lambda: adapter,
        resolver_factory=lambda: _Resolver([]),
    )
    app = _RemoteApp(view)

    async with app.run_test() as pilot:
        query = view.query_one("#remote-model-query", Input)
        query.value = "owner/repository"
        await pilot.click("#remote-model-search")
        for _attempt in range(20):
            if started.is_set():
                break
            await pilot.pause()
        assert started.is_set(), "resolve worker did not reach the adapter"

        query.value = "new/repository"
        release.set()
        await app.workers.wait_for_complete()
        await pilot.pause()
        rendered = _text(view)
        candidate_buttons = list(view.query(".remote-candidate").results(Button))
        search_disabled = view.query_one("#remote-model-search", Button).disabled

    assert "model-q4.gguf" not in rendered
    assert candidate_buttons == []
    assert view._operation_reference is None
    assert search_disabled is False


@pytest.mark.asyncio
async def test_resolution_mismatch_removes_stale_rendered_candidate_controls() -> None:
    """Cleared retained state must also remove previously mounted candidates."""
    view = _view(adapter_factory=MagicMock(), resolver_factory=MagicMock())
    app = _RemoteApp(view)
    old_resolved = _resolved("old/repository")
    new_resolved = _resolved("new/repository")

    async with app.run_test() as pilot:
        query = view.query_one("#remote-model-query", Input)
        query.value = "old/repository"
        view._resolve_generation = 1
        view._apply_resolve_result(
            1,
            "old/repository",
            "old/repository",
            old_resolved,
            None,
        )
        await pilot.pause()
        assert list(view.query(".remote-candidate").results(Button))

        view._resolve_remote = MagicMock()
        query.value = "new/repository"
        view._search_submitted()
        assert "Inspecting repository" in _text(view)

        query.value = "changed/repository"
        view._apply_resolve_result(
            2,
            "new/repository",
            "new/repository",
            new_resolved,
            None,
        )
        await pilot.pause()

        stale_controls = list(
            view.query(".remote-result, .remote-candidate").results(Button)
        )
        status = str(view.query_one("#remote-model-status", Static).renderable)
        search_disabled = view.query_one("#remote-model-search", Button).disabled
        query_disabled = query.disabled

    assert stale_controls == []
    assert view._operation_reference is None
    assert "Inspecting repository" not in status
    assert "Press Search" in status
    assert search_disabled is False
    assert query_disabled is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error", "expected"),
    (
        (
            RemoteDiscoveryError("authentication_required"),
            "Configure or verify Hugging Face access, then Retry.",
        ),
        (RemoteDiscoveryError("rate_limited", retryable=True), "Retry."),
        (RemoteDiscoveryError("network_error", retryable=True), "Retry."),
        (
            RemoteDiscoveryError("response_too_large"),
            "cannot be safely inspected",
        ),
        (
            RemoteDiscoveryError(
                "no_eligible_gguf",
                details=("owner/repository · model missing 00002",),
            ),
            "LFS-backed with size and SHA-256 metadata",
        ),
    ),
)
async def test_discovery_errors_render_sanitized_retry_guidance(
    error: RemoteDiscoveryError,
    expected: str,
) -> None:
    """Raw upstream details must never displace bounded recovery guidance."""
    view = _view(adapter_factory=MagicMock(), resolver_factory=MagicMock())
    app = _RemoteApp(view)

    async with app.run_test() as pilot:
        view.query_one("#remote-model-query", Input).value = "owner/repository"
        view._resolve_generation = 1
        view._apply_resolve_result(
            1,
            "owner/repository",
            "owner/repository",
            None,
            error,
        )
        await pilot.pause()
        rendered = _text(view)

    assert expected in rendered
    assert repr(error) not in rendered


@pytest.mark.asyncio
async def test_resolve_error_keeps_results_without_stale_inspecting_copy() -> None:
    """A failed detail request must return to browse context instead of looking busy."""
    summary = _summary()
    view = _view(adapter_factory=MagicMock(), resolver_factory=MagicMock())
    app = _RemoteApp(view)

    async with app.run_test() as pilot:
        query = view.query_one("#remote-model-query", Input)
        query.value = "quantized model"
        view._results = (summary,)
        view._selected_repository = summary.repository
        view._resolve_generation = 1
        view._apply_resolve_result(
            1,
            summary.repository,
            "quantized model",
            None,
            RemoteDiscoveryError("network_error", retryable=True),
        )
        await pilot.pause()
        rendered = _text(view)

    assert "owner/repository" in rendered
    assert "Remote request failed. Retry." in rendered
    assert "Inspecting owner/repository" not in rendered


@pytest.mark.asyncio
async def test_no_eligible_error_renders_bounded_incomplete_shard_details() -> None:
    """Validated missing-shard recovery details must follow the generic LFS rule."""
    detail = "owner/repository · model-q4 missing 00002 00004"
    error = RemoteDiscoveryError("no_eligible_gguf", details=(detail,))
    view = _view(adapter_factory=MagicMock(), resolver_factory=MagicMock())
    app = _RemoteApp(view)

    async with app.run_test() as pilot:
        view.query_one("#remote-model-query", Input).value = "owner/repository"
        view._resolve_generation = 1
        view._apply_resolve_result(
            1,
            "owner/repository",
            "owner/repository",
            None,
            error,
        )
        await pilot.pause()
        status = view.query_one("#remote-model-status", Static)
        rendered = str(status.renderable)

    generic = "Files must be LFS-backed with size and SHA-256 metadata."
    assert generic in rendered
    assert detail in rendered
    assert rendered.index(generic) < rendered.index(detail)
    assert status._render_markup is False


@pytest.mark.asyncio
async def test_oversized_lfs_size_recovers_without_rendering_a_candidate() -> None:
    """Hostile declared sizes must be rejected before the Remote view formats them."""

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "sha": _COMMIT,
                "siblings": [
                    {
                        "rfilename": "huge.gguf",
                        "lfs": {"size": 2**63, "sha256": _DIGEST},
                    }
                ],
                "cardData": None,
            },
        )

    adapter = HuggingFaceRemoteAdapter(
        client_factory=lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler))
    )
    view = _view(
        adapter_factory=lambda: adapter,
        resolver_factory=lambda: _Resolver([]),
    )
    app = _RemoteApp(view)

    async with app.run_test() as pilot:
        await _submit(app, pilot, "owner/repository")
        rendered = _text(view)
        candidates = list(view.query(".remote-candidate").results(Button))

    assert "No eligible GGUF files were found" in rendered
    assert str(2**63) not in rendered
    assert candidates == []


@pytest.mark.asyncio
async def test_incomplete_shard_warnings_render_bounded_candidate_and_indexes() -> None:
    """Dropping resolution warnings must hide the actionable missing-shard evidence."""
    warning = "owner/repository · model-q4 missing 00002 00004"
    resolved = _resolved(warnings=(warning,))
    view = _view(adapter_factory=MagicMock(), resolver_factory=MagicMock())
    app = _RemoteApp(view)

    async with app.run_test() as pilot:
        view.query_one("#remote-model-query", Input).value = "owner/repository"
        view._resolve_generation = 1
        view._apply_resolve_result(
            1,
            "owner/repository",
            "owner/repository",
            resolved,
            None,
        )
        await pilot.pause()
        rendered = _text(view)

    assert warning in rendered


@pytest.mark.asyncio
async def test_candidate_cap_discloses_deterministic_first_hundred() -> None:
    """A truncated candidate list must disclose its deterministic upstream order."""
    candidates = tuple(
        RemoteGGUFCandidate(
            label=f"owner/repository · {index:03d}.gguf",
            files=(RemoteGGUFFile(f"{index:03d}.gguf", 1, _DIGEST),),
            total_bytes=1,
        )
        for index in range(100)
    )
    resolved = ResolvedRemoteModel(
        repository="owner/repository",
        commit=_COMMIT,
        license_id="apache-2.0",
        review_url=f"https://huggingface.co/owner/repository/tree/{_COMMIT}",
        candidates=candidates,
        total_candidate_count=137,
        warnings=(),
    )
    view = _view(adapter_factory=MagicMock(), resolver_factory=MagicMock())
    app = _RemoteApp(view)

    async with app.run_test() as pilot:
        view._resolved = resolved
        view._refresh_with_status("Select one GGUF candidate.")
        await pilot.pause()
        rendered = _text(view)

    assert "First 100 of 137, sorted by upstream path" in rendered


# ---------------------------------------------------------------------------
# Install-request flow: this view posts the intent and stops (TASK-1914).
# ---------------------------------------------------------------------------


def _capturing_app(view) -> App:
    """Build an App that captures ``RemoteView.InstallRequested`` events."""
    from tldw_chatbook.UI.Screens.model_remote_view import RemoteView

    class _App(ConsolidatedCSSApp):
        def __init__(self) -> None:
            self.view = view
            self.requests: list = []
            super().__init__()

        def compose(self) -> ComposeResult:
            yield self.view

        @on(RemoteView.InstallRequested)
        def _capture(self, event: RemoteView.InstallRequested) -> None:
            self.requests.append(event)

    return _App()


@pytest.mark.asyncio
async def test_contextual_install_posts_requested_with_the_resolved_service_and_resolver() -> (
    None
):
    """Real candidate selection plus the contextual install action
    posts ``RemoteView.InstallRequested`` carrying the exact catalog,
    candidate, service, and credential resolver the host screen needs to
    resolve a plan itself (TASK-1914: this view no longer performs that
    resolution; ``LLMScreen`` does). See
    ``test_llm_screen_lab_adoption.py``'s remote-install tests for the
    end-to-end coverage of what happens once ``LLMScreen`` receives this
    message.
    """
    resolved = _resolved()
    service = object()
    # A working `.resolve()` stand-in, not a bare `object()`: this same
    # factory also backs the metadata search/resolve this test drives
    # through `_submit` first, so it must behave like a real resolver, not
    # just be identity-comparable.
    resolver = _Resolver([])
    adapter = _Adapter(resolved=resolved)
    view = _view(
        adapter_factory=lambda: adapter,
        resolver_factory=lambda: resolver,
        service_factory=lambda: service,
    )
    app = _capturing_app(view)

    async with app.run_test() as pilot:
        await _submit(app, pilot, "owner/repository")

        candidate_button = view.query_one(".remote-candidate", Button)
        candidate_button.press()
        await pilot.pause()
        assert app.requests == []

        install_button = view.query_one("#remote-model-install", Button)
        install_button.press()
        await pilot.pause()

        assert len(app.requests) == 1
        event = app.requests[0]
        expected_catalog = build_remote_catalog(resolved, resolved.candidates[0])
        assert event.catalog == expected_catalog
        assert event.candidate == resolved.candidates[0]
        assert event.service is service
        assert event.credential_resolver is resolver

        # The selected candidate's own row re-disables immediately (the
        # long-standing "cannot double-click install" contract, unrelated
        # to whether LLMScreen has even received the message yet).
        assert view.query_one(".remote-candidate", Button).disabled is True
        assert view.query_one("#remote-model-search", Button).disabled is True
        assert view.query_one("#remote-variant-filter", Input).disabled is True
        assert view.query_one("#remote-variant-sort", Select).disabled is True


@pytest.mark.asyncio
@pytest.mark.parametrize("failing_dependency", ("service", "credential resolver"))
async def test_install_dependency_failure_keeps_the_selected_candidate_retryable(
    failing_dependency: str,
) -> None:
    """A dependency factory failure must not strand the view in-flight.

    This catches moving ``_operation_reference``/control disabling ahead of
    dependency construction without a rollback path. The dependency is
    allowed to fail only after repository resolution so the test exercises
    the install boundary, not metadata search.
    """
    resolved = _resolved()
    adapter = _Adapter(resolved=resolved)
    resolver = _Resolver([])
    fail_now = False

    def service_factory():
        if fail_now and failing_dependency == "service":
            raise RuntimeError("private service detail")
        return object()

    def resolver_factory():
        if fail_now and failing_dependency == "credential resolver":
            raise RuntimeError("private resolver detail")
        return resolver

    view = _view(
        adapter_factory=lambda: adapter,
        resolver_factory=resolver_factory,
        service_factory=service_factory,
    )
    app = _capturing_app(view)
    notifications: list[tuple[str, str]] = []

    async with app.run_test() as pilot:
        view.notify = lambda message, *, severity: notifications.append(
            (message, severity)
        )
        await _submit(app, pilot, resolved.repository)
        view.query_one(".remote-candidate", Button).press()
        await pilot.pause()
        fail_now = True

        error: Exception | None = None
        try:
            view._install_pressed(
                Button.Pressed(view.query_one("#remote-model-install", Button))
            )
        except Exception as exc:  # assertion below exposes the production leak
            error = exc
        await pilot.pause()

        assert error is None, "dependency construction escaped the UI boundary"
        assert app.requests == []
        assert view._operation_reference is None
        assert view.query_one("#remote-model-install", Button).disabled is False
        assert view.query_one("#remote-model-search", Button).disabled is False
        assert "private" not in str(
            view.query_one("#remote-model-status", Static).renderable
        )

    assert notifications == [
        (
            "Could not prepare the managed install. Check model storage "
            "settings and try again.",
            "error",
        )
    ]


@pytest.mark.asyncio
async def test_default_credential_resolver_factory_builds_env_config_resolver_for_the_posted_intent() -> (
    None
):
    """The production path must not silently fall back to no credential resolver at all."""
    from tldw_chatbook.Model_Artifacts.acquisition import EnvConfigCredentialResolver

    resolved = _resolved()
    adapter = _Adapter(resolved=resolved)
    view = _view(adapter_factory=lambda: adapter, service_factory=lambda: object())
    app = _capturing_app(view)

    async with app.run_test() as pilot:
        await _submit(app, pilot, "owner/repository")
        view.query_one(".remote-candidate", Button).press()
        await pilot.pause()
        view.query_one("#remote-model-install", Button).press()
        await pilot.pause()

    assert len(app.requests) == 1
    assert isinstance(app.requests[0].credential_resolver, EnvConfigCredentialResolver)


@pytest.mark.asyncio
async def test_stale_candidate_press_is_rejected_at_the_ui_boundary() -> None:
    """A queued button event from an old resolution must not post an intent."""
    old_resolved = _resolved("old/repository")
    current_resolved = _resolved("current/repository")
    view = _view(adapter_factory=MagicMock(), resolver_factory=MagicMock())
    app = _capturing_app(view)

    async with app.run_test() as pilot:
        view._resolved = old_resolved
        view._refresh_with_status("Old resolution")
        await pilot.pause()
        stale_button = view.query_one(".remote-candidate", Button)

        view._resolved = current_resolved
        view._refresh_with_status("Current resolution")
        await pilot.pause()
        view._candidate_pressed(Button.Pressed(stale_button))
        await pilot.pause()

        assert view._selected_candidate is None
        install = view.query_one("#remote-model-install", Button)
        assert install.disabled is True
        install.press()
        await pilot.pause()

    assert app.requests == []
    assert view._operation_reference is None


@pytest.mark.asyncio
async def test_candidate_press_notifies_and_does_not_post_when_the_catalog_cannot_be_built() -> (
    None
):
    """A candidate that fails ``build_remote_catalog`` must never reach the host screen."""
    resolved = _resolved()
    bad_candidate = RemoteGGUFCandidate(label="bad", files=(), total_bytes=0)
    tampered = ResolvedRemoteModel(
        repository=resolved.repository,
        commit=resolved.commit,
        license_id=resolved.license_id,
        review_url=resolved.review_url,
        candidates=(bad_candidate,),
        total_candidate_count=1,
        warnings=(),
    )
    view = _view(adapter_factory=MagicMock(), resolver_factory=MagicMock())
    app = _capturing_app(view)
    notifications: list[tuple[str, str]] = []

    async with app.run_test() as pilot:
        view.notify = lambda message, *, severity: notifications.append(
            (message, severity)
        )
        view._resolved = tampered
        view._refresh_with_status("Select one GGUF candidate.")
        await pilot.pause()
        view.query_one(".remote-candidate", Button).press()
        await pilot.pause()
        view.query_one("#remote-model-install", Button).press()
        await pilot.pause()

    assert app.requests == []
    assert view._operation_reference is None
    assert notifications and notifications[0][1] == "error"


# ---------------------------------------------------------------------------
# Render-only outcomes: apply_progress() / cancel_pending_install() /
# finish_install().
#
# TASK-1914: the host screen (LLMScreen) is the only caller of any of
# these -- apply_progress for a live tick, cancel_pending_install after a
# preflight failure or an explicit consent-modal decline, finish_install
# once provisioning completes, successfully or not.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_apply_progress_renders_and_retains_the_tick() -> None:
    """A live tick updates the progress widget and is retained for hydration."""
    from tldw_chatbook.Model_Artifacts.acquisition import AcquisitionProgress
    from tldw_chatbook.Model_Artifacts.service import ArtifactRef
    from tldw_chatbook.Widgets.ModelArtifacts.install_progress import (
        ModelInstallProgress,
    )

    view = _view(adapter_factory=MagicMock(), resolver_factory=MagicMock())
    app = _RemoteApp(view)
    reference = ArtifactRef("owner-repository", "a" * 40, "q4_k_m")
    progress = AcquisitionProgress("fetch", reference, "model-q4.gguf", 512, 1024)

    async with app.run_test() as pilot:
        view.apply_progress(progress)
        await pilot.pause()

        widget = view.query_one("#remote-model-install-progress", ModelInstallProgress)
        detail_pane = view.query_one(".remote-detail-pane")
        assert widget.display is True
        assert widget.region.x >= detail_pane.region.x
        assert widget.region.right <= detail_pane.region.right
        assert view._progress == progress


def test_apply_progress_tolerates_a_recompose_gap() -> None:
    """A progress event is retained while its widget is temporarily absent."""
    from tldw_chatbook.UI.Screens.model_remote_view import RemoteView

    view = RemoteView(adapter_factory=MagicMock(), service_factory=MagicMock())
    view.query_one = MagicMock(side_effect=NoMatches)
    view.refresh = MagicMock()
    progress = object()

    view.apply_progress(progress)

    assert view._progress is progress
    view.refresh.assert_called_once_with(recompose=True)


@pytest.mark.asyncio
async def test_cancel_pending_install_clears_the_indicator_and_reenables_controls() -> (
    None
):
    """A preflight failure or a decline releases the indicator without reloading."""
    resolved = _resolved()
    adapter = _Adapter(resolved=resolved)
    view = _view(
        adapter_factory=lambda: adapter, resolver_factory=lambda: _Resolver([])
    )
    app = _RemoteApp(view)

    async with app.run_test() as pilot:
        await _submit(app, pilot, "owner/repository")
        view.query_one(".remote-candidate", Button).press()
        await pilot.pause()
        view.query_one("#remote-model-install", Button).press()
        await pilot.pause()
        assert view._operation_reference is not None
        assert view.query_one("#remote-model-search", Button).disabled is True

        view.cancel_pending_install("Sanitized failure.")
        await pilot.pause()

        assert view._operation_reference is None
        assert view.query_one("#remote-model-search", Button).disabled is False
        assert view.query_one(".remote-candidate", Button).disabled is False
        status = str(view.query_one("#remote-model-status", Static).renderable)

    assert status == "Sanitized failure."


@pytest.mark.asyncio
async def test_cancel_pending_install_with_no_message_restores_the_default_status() -> (
    None
):
    """An explicit consent-modal decline restores ordinary status copy."""
    resolved = _resolved()
    adapter = _Adapter(resolved=resolved)
    view = _view(
        adapter_factory=lambda: adapter, resolver_factory=lambda: _Resolver([])
    )
    app = _RemoteApp(view)

    async with app.run_test() as pilot:
        await _submit(app, pilot, "owner/repository")
        view.query_one(".remote-candidate", Button).press()
        await pilot.pause()
        view.query_one("#remote-model-install", Button).press()
        await pilot.pause()

        view.cancel_pending_install()
        await pilot.pause()
        status = str(view.query_one("#remote-model-status", Static).renderable)

    assert "Select one GGUF candidate." in status


@pytest.mark.asyncio
async def test_finish_install_clears_the_indicator_progress_and_shows_the_given_message() -> (
    None
):
    """``finish_install`` always hides progress, even mid-recompose."""
    from tldw_chatbook.Model_Artifacts.acquisition import AcquisitionProgress
    from tldw_chatbook.Model_Artifacts.service import ArtifactRef
    from tldw_chatbook.Widgets.ModelArtifacts.install_progress import (
        ModelInstallProgress,
    )

    resolved = _resolved()
    adapter = _Adapter(resolved=resolved)
    view = _view(
        adapter_factory=lambda: adapter, resolver_factory=lambda: _Resolver([])
    )
    app = _RemoteApp(view)
    reference = ArtifactRef("owner-repository", "a" * 40, "q4_k_m")

    async with app.run_test() as pilot:
        await _submit(app, pilot, "owner/repository")
        view.query_one(".remote-candidate", Button).press()
        await pilot.pause()
        view.query_one("#remote-model-install", Button).press()
        await pilot.pause()
        view.apply_progress(
            AcquisitionProgress("fetch", reference, "model-q4.gguf", 512, 1024)
        )
        await pilot.pause()

        view.finish_install("Model downloaded and managed.")
        await pilot.pause()

        assert view._operation_reference is None
        assert view._progress is None
        progress_widget = view.query_one(
            "#remote-model-install-progress", ModelInstallProgress
        )
        assert progress_widget.display is False
        status = str(view.query_one("#remote-model-status", Static).renderable)

    assert status == "Model downloaded and managed."


@pytest.mark.asyncio
async def test_successful_install_exposes_provider_attributed_adoption_actions() -> (
    None
):
    """A verified Remote download ends in explicit next actions, not another install."""
    from tldw_chatbook.Model_Artifacts.service import ArtifactRef

    resolved = _resolved()
    view = _view(adapter_factory=MagicMock(), resolver_factory=MagicMock())
    app = _RemoteApp(view)
    reference = ArtifactRef("owner-repository", "a" * 40, "q4_k_m")

    async with app.run_test() as pilot:
        query = view.query_one("#remote-model-query", Input)
        query.value = resolved.repository
        view._resolve_generation = 1
        view._apply_resolve_result(
            1,
            resolved.repository,
            resolved.repository,
            resolved,
            None,
        )
        await pilot.pause()
        view.query_one(".remote-candidate", Button).press()
        await pilot.pause()

        view.finish_install(
            "Model downloaded and managed.",
            completed_reference=reference,
        )
        await pilot.pause()

        rendered = _text(view)
        open_installed = view.query_one("#remote-model-open-installed", Button)
        configure = view.query_one("#remote-model-configure-runtime", Button)

        assert "Source: Hugging Face" in rendered
        assert "Downloaded · Verified · Managed · Not active" in rendered
        assert open_installed.disabled is False
        assert configure.disabled is False
        assert list(view.query("#remote-model-install")) == []


@pytest.mark.asyncio
async def test_open_installed_posts_the_exact_completed_reference() -> None:
    """The completion action carries managed identity without filename recovery."""
    from tldw_chatbook.Model_Artifacts.service import ArtifactRef
    from tldw_chatbook.UI.Screens.model_remote_view import RemoteView

    resolved = _resolved()
    reference = ArtifactRef("owner-repository", "a" * 40, "q4_k_m")
    view = _view(adapter_factory=MagicMock(), resolver_factory=MagicMock())

    class _AdoptionApp(ConsolidatedCSSApp):
        def __init__(self) -> None:
            self.opened: list[ArtifactRef] = []
            super().__init__()

        def compose(self) -> ComposeResult:
            yield view

        @on(RemoteView.OpenInstalledRequested)
        def _opened(self, event: RemoteView.OpenInstalledRequested) -> None:
            self.opened.append(event.reference)

    app = _AdoptionApp()
    async with app.run_test() as pilot:
        query = view.query_one("#remote-model-query", Input)
        query.value = resolved.repository
        view._resolve_generation = 1
        view._apply_resolve_result(
            1,
            resolved.repository,
            resolved.repository,
            resolved,
            None,
        )
        await pilot.pause()
        view.query_one(".remote-candidate", Button).press()
        await pilot.pause()
        view.finish_install(
            "Model downloaded and managed.",
            completed_reference=reference,
        )
        await pilot.pause()

        view.query_one("#remote-model-open-installed", Button).press()
        await pilot.pause()

    assert app.opened == [reference]


@pytest.mark.asyncio
async def test_configure_runtime_posts_the_exact_completed_reference() -> None:
    """Remote delegates runtime choice while preserving exact managed identity."""
    from tldw_chatbook.Model_Artifacts.service import ArtifactRef
    from tldw_chatbook.UI.Screens.model_remote_view import RemoteView

    resolved = _resolved()
    reference = ArtifactRef("owner-repository", "a" * 40, "q4_k_m")
    view = _view(adapter_factory=MagicMock(), resolver_factory=MagicMock())

    class _AdoptionApp(ConsolidatedCSSApp):
        def __init__(self) -> None:
            self.configured: list[ArtifactRef] = []
            super().__init__()

        def compose(self) -> ComposeResult:
            yield view

        @on(RemoteView.ConfigureRuntimeRequested)
        def _configured(self, event: RemoteView.ConfigureRuntimeRequested) -> None:
            self.configured.append(event.reference)

    app = _AdoptionApp()
    async with app.run_test() as pilot:
        query = view.query_one("#remote-model-query", Input)
        query.value = resolved.repository
        view._resolve_generation = 1
        view._apply_resolve_result(
            1,
            resolved.repository,
            resolved.repository,
            resolved,
            None,
        )
        await pilot.pause()
        view.query_one(".remote-candidate", Button).press()
        await pilot.pause()
        view.finish_install(
            "Model downloaded and managed.",
            completed_reference=reference,
        )
        await pilot.pause()

        view.query_one("#remote-model-configure-runtime", Button).press()
        await pilot.pause()

    assert app.configured == [reference]


@pytest.mark.asyncio
async def test_new_search_clears_the_prior_completion_actions() -> None:
    """A new discovery cannot retain adoption actions for a stale managed root."""
    from tldw_chatbook.Model_Artifacts.service import ArtifactRef

    resolved = _resolved()
    adapter = _Adapter(search_result=(_summary("new/repository"),), resolved=resolved)
    reference = ArtifactRef("owner-repository", "a" * 40, "q4_k_m")
    view = _view(
        adapter_factory=lambda: adapter,
        resolver_factory=lambda: _Resolver([]),
    )
    app = _RemoteApp(view)

    async with app.run_test() as pilot:
        query = view.query_one("#remote-model-query", Input)
        query.value = resolved.repository
        view._resolve_generation = 1
        view._apply_resolve_result(
            1,
            resolved.repository,
            resolved.repository,
            resolved,
            None,
        )
        await pilot.pause()
        view.query_one(".remote-candidate", Button).press()
        await pilot.pause()
        view.finish_install(
            "Model downloaded and managed.",
            completed_reference=reference,
        )
        await pilot.pause()

        await _submit(app, pilot, "new model")

        assert view._query_value == "new model"
        assert adapter.search_calls == [("new model", "configured-token")]
        assert [result.repository for result in view._results] == ["new/repository"]
        assert list(view.query("#remote-model-open-installed")) == []
        assert list(view.query("#remote-model-configure-runtime")) == []
        assert view.query_one("#remote-model-install", Button).disabled is True
        assert "new/repository" in _text(view)


@pytest.mark.asyncio
async def test_new_discovery_notifies_the_host_to_clear_durable_completion() -> None:
    """Starting a search invalidates the screen owner's prior adoption target."""
    from tldw_chatbook.UI.Screens.model_remote_view import RemoteView

    adapter = _Adapter(search_result=(_summary("new/repository"),))
    view = _view(
        adapter_factory=lambda: adapter,
        resolver_factory=lambda: _Resolver([]),
    )

    class _DiscoveryApp(ConsolidatedCSSApp):
        def __init__(self) -> None:
            self.view = view
            self.started: list[str] = []
            super().__init__()

        def compose(self) -> ComposeResult:
            yield view

        @on(RemoteView.DiscoveryStarted)
        def _started(self, event: RemoteView.DiscoveryStarted) -> None:
            self.started.append(event.query)

    app = _DiscoveryApp()
    async with app.run_test() as pilot:
        await _submit(app, pilot, "new model")

    assert app.started == ["new model"]


@pytest.mark.asyncio
async def test_finish_install_tolerates_a_missing_progress_widget() -> None:
    """Missing progress markup mid-recompose must not skip indicator cleanup
    or the status update -- only the progress widget lookup is tolerated
    (mirroring ``apply_progress``'s own tolerance for the same underlying
    reason: ``ModelInstallProgress`` may not have finished composing its
    own children yet), not every widget on the view.
    """
    from tldw_chatbook.Model_Artifacts.service import ArtifactRef

    view = _view(adapter_factory=MagicMock(), resolver_factory=MagicMock())
    app = _RemoteApp(view)

    async with app.run_test() as pilot:
        view._operation_reference = ArtifactRef("model-a", "a" * 40, "q4_k_m")
        view._progress = object()
        original_query_one = view.query_one

        def _flaky_query_one(selector, *args, **kwargs):
            if selector == "#remote-model-install-progress":
                raise NoMatches("missing widget")
            return original_query_one(selector, *args, **kwargs)

        view.query_one = _flaky_query_one

        view.finish_install("Model downloaded and managed.")
        await pilot.pause()

        assert view._operation_reference is None
        assert view._progress is None
        status = str(view.query_one("#remote-model-status", Static).renderable)

    assert status == "Model downloaded and managed."


# ---------------------------------------------------------------------------
# Module-scope import boundary (TASK-1914, AC #3).
# ---------------------------------------------------------------------------


def test_remote_view_does_not_import_acquisition_at_module_scope() -> None:
    """``RemoteView`` posts intents; only ``LLMScreen``'s worker methods
    (and this module's own lazily-invoked ``_default_credential_resolver``)
    import ``Model_Artifacts.acquisition``.

    Uses the AST-based :func:`module_scope_forbidden_acquisition_imports`
    (TASK-1914 fix round 1) rather than a text/substring scan: a substring
    scan for ``"from tldw_chatbook.Model_Artifacts.acquisition import"``
    (etc.) passes right over ``from tldw_chatbook.Model_Artifacts import
    acquisition`` -- a real, eager, module-scope import of the acquisition
    runtime via the package-then-attribute form -- because that exact
    substring never appears in it.
    """
    import inspect

    from tldw_chatbook.UI.Screens import model_remote_view as module

    source = inspect.getsource(module)
    assert "class RemoteView(Widget):" in source
    findings = module_scope_forbidden_acquisition_imports(source)
    assert findings == [], (
        f"model_remote_view.py imports acquisition/fetch at module scope: {findings}"
    )
