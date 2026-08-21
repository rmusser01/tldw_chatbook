import asyncio
import os
from dataclasses import replace
from types import SimpleNamespace

import pytest
from textual import on
from textual.app import App
from textual.widgets import Checkbox, Static

from tldw_chatbook.Skills_Interop.project_skills_discovery import (
    discover_project_skills,
)
from tldw_chatbook.Skills_Interop.project_skills_prompt import (
    ProjectSkillsPromptLedger,
)
from tldw_chatbook.UI.Navigation.main_navigation import NavigateToScreen
from tldw_chatbook.Widgets import project_skills_import_modal as _offer_module
from tldw_chatbook.Widgets.project_skills_import_modal import (
    ProjectSkillsImportModal,
    _project_skills_importer,
    maybe_offer_project_skills_import,
)


def _discovery(tmp_path, names=("alpha-skill", "beta-skill")):
    for name in names:
        d = tmp_path / ".SKILLS" / name
        d.mkdir(parents=True)
        (d / "SKILL.md").write_text(
            f"---\ndescription: [red]desc[/red] for {name}\n---\nBody\n",
            encoding="utf-8",
        )
    return discover_project_skills(tmp_path)


class _HarnessApp(App[None]):
    def __init__(self, modal):
        super().__init__()
        self._modal = modal
        self.result = "unset"

    def on_mount(self) -> None:
        def _done(result):
            self.result = result

        self.push_screen(self._modal, _done)


class _RecorderApp:
    """Minimal app double (not a real Textual App): records
    ``push_screen``/``run_worker`` calls without ever running an event
    loop -- used for Finding 1/6 tests where ``maybe_offer_project_skills_
    import`` must return BEFORE touching either, so there is nothing to
    pump.
    """

    def __init__(self, skills_scope_service="sentinel"):
        self.skills_scope_service = skills_scope_service
        self.pushed: list = []
        self.workers: list = []

    def push_screen(self, screen, callback=None):
        self.pushed.append(screen)

    def run_worker(self, coro, **kwargs):
        self.workers.append(coro)


def _modal(tmp_path, installed=frozenset(), imported=None, fail=()):
    imported = imported if imported is not None else []

    async def importer(entry):
        if entry.name in fail:
            raise ValueError("import exploded")
        imported.append(entry.name)

    return (
        ProjectSkillsImportModal(
            discovery=_discovery(tmp_path),
            installed_names=installed,
            importer=importer,
        ),
        imported,
    )


@pytest.mark.asyncio
async def test_new_rows_checked_installed_rows_unchecked(tmp_path):
    modal, _ = _modal(tmp_path, installed=frozenset({"beta-skill"}))
    app = _HarnessApp(modal)
    async with app.run_test() as pilot:
        await pilot.pause()
        boxes = {
            box.id: box.value for box in modal.query(Checkbox)
        }
        assert boxes["project-skill-row-0"] is True   # alpha-skill: new
        assert boxes["project-skill-row-1"] is False  # beta-skill: installed


@pytest.mark.asyncio
async def test_escape_means_not_now(tmp_path):
    modal, _ = _modal(tmp_path)
    app = _HarnessApp(modal)
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("escape")
        await pilot.pause()
    assert app.result == ("not_now", None)


@pytest.mark.asyncio
async def test_never_button(tmp_path):
    modal, _ = _modal(tmp_path)
    app = _HarnessApp(modal)
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.click("#project-skills-never")
        await pilot.pause()
    assert app.result == ("never", None)


@pytest.mark.asyncio
async def test_import_selected_runs_importer_and_reports(tmp_path):
    modal, imported = _modal(tmp_path, fail=("beta-skill",))
    app = _HarnessApp(modal)
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.click("#project-skills-import")
        await pilot.pause()
        # results phase: Close dismisses with outcomes
        await pilot.click("#project-skills-close")
        await pilot.pause()
    assert imported == ["alpha-skill"]
    decision, outcomes = app.result
    assert decision == "imported"
    assert ("alpha-skill", "imported") in outcomes
    assert any(name == "beta-skill" and "exploded" in msg for name, msg in outcomes)


@pytest.mark.asyncio
async def test_double_click_import_runs_importer_once_per_entry(tmp_path):
    """A rapid double-press of 'Import selected' must not double-run imports.

    A slow (real-yielding) importer keeps the "Import selected" button
    mounted across both clicks -- ``pilot.click`` only waits for the app's
    message queue to idle, not for the background import worker to finish,
    so a second real click can land on the SAME button while the first
    import is still in flight. The modal's one-shot ``_committed`` guard is
    what keeps that second click a no-op instead of a duplicate import run.
    """
    imported: list[str] = []

    async def slow_importer(entry):
        await asyncio.sleep(0.05)
        imported.append(entry.name)

    modal = ProjectSkillsImportModal(
        discovery=_discovery(tmp_path),
        installed_names=frozenset(),
        importer=slow_importer,
    )
    app = _HarnessApp(modal)
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.click("#project-skills-import")
        await pilot.click("#project-skills-import")
        await pilot.pause(0.2)
    assert sorted(imported) == ["alpha-skill", "beta-skill"]


@pytest.mark.asyncio
async def test_never_during_inflight_import_is_inert(tmp_path):
    """Finding 2: while an import is running (``_committed`` and no
    outcomes yet), "Never for this folder" must be inert -- it must not
    dismiss the modal mid-import -- and the flow must still land on the
    results phase once the import finishes.
    """
    imported: list[str] = []

    async def slow_importer(entry):
        await asyncio.sleep(0.1)
        imported.append(entry.name)

    modal = ProjectSkillsImportModal(
        discovery=_discovery(tmp_path),
        installed_names=frozenset(),
        importer=slow_importer,
    )
    app = _HarnessApp(modal)
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.click("#project-skills-import")
        await pilot.pause()
        # The slow importer is still in flight here (0.1s sleep, no pause
        # duration given) -- Never must be a no-op, not a dismissal.
        await pilot.click("#project-skills-never")
        await pilot.pause()
        assert app.result == "unset"  # still open -- no dismissal happened
        assert app.screen is modal
        await pilot.pause(0.2)  # let the slow importer finish
        assert modal._outcomes is not None  # landed on the results phase
    assert sorted(imported) == ["alpha-skill", "beta-skill"]


@pytest.mark.asyncio
async def test_escape_on_results_phase_dismisses_as_imported(tmp_path):
    """Minor 6: escape on the RESULTS phase must dismiss like "Close"
    (``("imported", outcomes)``), not like a cancel (``("not_now", None)``)
    -- the import already ran; there is nothing left to cancel.
    """
    modal, _imported = _modal(tmp_path)
    app = _HarnessApp(modal)
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.click("#project-skills-import")
        await pilot.pause()
        assert modal._outcomes is not None  # sanity: results phase reached
        await pilot.press("escape")
        await pilot.pause()
    decision, outcomes = app.result
    assert decision == "imported"
    assert outcomes is not None and len(outcomes) == 2


# ---------------------------------------------------------------------------
# TASK-17964 follow-ups
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_offer_footer_renders_skipped_and_truncated_lines(tmp_path):
    """``_offer_footer_lines``'s two branches -- neither exercised by any
    existing test before now: a discovery with ``skipped`` entries renders
    ``Skipped: <names>``, and ``truncated > 0`` renders "N more not shown".
    """
    base_discovery = _discovery(tmp_path, names=("alpha-skill",))
    discovery = replace(
        base_discovery,
        skipped=(("weird-file", "no SKILL.md"), ("sneaky", "symlink")),
        truncated=4,
    )

    async def _importer(entry):
        raise AssertionError("importer should not run in this footer-only test")

    modal = ProjectSkillsImportModal(
        discovery=discovery, installed_names=frozenset(), importer=_importer
    )
    app = _HarnessApp(modal)
    async with app.run_test() as pilot:
        await pilot.pause()
        footer = modal.query_one("#project-skills-footer", Static)
        text = str(footer.renderable)

    assert "Skipped: weird-file, sneaky" in text
    assert "4 more not shown" in text


@pytest.mark.asyncio
async def test_offer_phase_checkbox_label_escapes_markup_literally(tmp_path):
    """A markup-hostile skill ``description`` must render as literal text
    in the Checkbox label -- ``escape_markup()`` at the
    ``_compose_offer_phase`` call site is the only thing defending this,
    with no assertion pinning it before now.

    The description must be YAML-quoted: unquoted brackets break the
    frontmatter's YAML grammar and degrade to an EMPTY description
    entirely (this file's shared ``_discovery()`` helper's unquoted
    ``[red]desc[/red] for {name}`` hits exactly that, so it can't be
    reused here) -- see the discovery-layer fixtures already covering this
    in ``Tests/Skills/test_project_skills_discovery.py``
    (``test_hostile_description_survives_as_plain_data``,
    ``test_unparseable_frontmatter_degrades_to_empty_description``).
    """
    skill_dir = tmp_path / ".SKILLS" / "alpha-skill"
    skill_dir.mkdir(parents=True)
    skill_dir.joinpath("SKILL.md").write_text(
        '---\ndescription: "[red]evil[/red]"\n---\nBody\n', encoding="utf-8"
    )
    discovery = discover_project_skills(tmp_path)
    assert discovery is not None
    assert discovery.entries[0].description == "[red]evil[/red]"  # sanity

    async def _importer(entry):
        raise AssertionError("importer should not run in this test")

    modal = ProjectSkillsImportModal(
        discovery=discovery, installed_names=frozenset(), importer=_importer
    )
    app = _HarnessApp(modal)
    async with app.run_test() as pilot:
        await pilot.pause()
        checkbox = modal.query_one("#project-skill-row-0", Checkbox)
        label = checkbox.label

    assert "[red]evil[/red]" in label.plain
    # No "red" style span was produced -- the brackets were never
    # interpreted as a markup tag.
    assert not any(span.style == "red" for span in label.spans)


class _StubSkillsScopeService:
    """Records import calls; ``alpha-skill`` reports as already installed."""

    def __init__(self):
        self.import_calls: list[tuple] = []

    async def get_context(self, *, mode=None):
        del mode
        return {
            "available_skills": [{"name": "alpha-skill"}],
            "blocked_skills": [],
        }

    async def import_skill_directory(self, path, *, mode, name, trust_approved):
        self.import_calls.append(("directory", str(path), mode, name, trust_approved))

    async def import_skill_file(
        self, data, *, mode, filename, content_type, trust_approved
    ):
        self.import_calls.append(
            ("file", data, mode, filename, content_type, trust_approved)
        )


class _OfferHarnessApp(App[None]):
    def __init__(self, service, discoveries):
        super().__init__()
        self.skills_scope_service = service
        self._discoveries = discoveries
        self.navigated: list[str] = []

    def on_mount(self) -> None:
        maybe_offer_project_skills_import(self, self._discoveries)

    @on(NavigateToScreen)
    def _record_navigation(self, message: NavigateToScreen) -> None:
        self.navigated.append(message.screen_name)


@pytest.mark.asyncio
async def test_maybe_offer_chains_installed_names_ledger_and_navigation(
    tmp_path, monkeypatch
):
    """End-to-end exercise of the shared offer helper across two discoveries.

    Covers everything the modal-only tests above don't: installed_names
    built from ``app.skills_scope_service.get_context``, the injected
    importer's exact ``import_skill_directory`` call shape, sequential
    chaining (discovery B's modal only appears after A's dismisses), the
    ledger decision mapping, and the "review" -> NavigateToScreen("skills")
    post.
    """
    ledger_dir = tmp_path / "data"
    monkeypatch.setattr(_offer_module, "get_user_data_dir", lambda: ledger_dir)

    root_a = tmp_path / "repo-a"
    root_b = tmp_path / "repo-b"
    discovery_a = _discovery(root_a, names=("alpha-skill",))
    discovery_b = _discovery(root_b, names=("beta-skill",))

    service = _StubSkillsScopeService()
    app = _OfferHarnessApp(service, (discovery_a, discovery_b))
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.pause()

        # First discovery: alpha-skill is reported "installed" by the stub
        # context, so its row must render unchecked.
        assert isinstance(app.screen, ProjectSkillsImportModal)
        boxes = {box.id: box.value for box in app.screen.query(Checkbox)}
        assert boxes["project-skill-row-0"] is False
        await pilot.click("#project-skills-never")
        await pilot.pause()
        await pilot.pause()

        # Second discovery only appears after the first dismisses.
        assert isinstance(app.screen, ProjectSkillsImportModal)
        await pilot.click("#project-skills-import")
        await pilot.pause()
        await pilot.click("#project-skills-review")
        await pilot.pause()

    assert service.import_calls == [
        ("directory", str(root_b / ".SKILLS" / "beta-skill"), "local", "beta-skill", False)
    ]
    assert app.navigated == ["skills"]

    ledger = ProjectSkillsPromptLedger(ledger_dir)
    assert ledger.decision_for(discovery_a.root) == ("never", discovery_a.fingerprint)
    assert ledger.decision_for(discovery_b.root) == (
        "imported",
        discovery_b.fingerprint,
    )


@pytest.mark.asyncio
async def test_maybe_offer_reentrancy_guard_blocks_concurrent_calls(
    tmp_path, monkeypatch
):
    """Two back-to-back offer calls while the first modal is still open.

    ``_OfferHarnessApp.on_mount`` fires the first (and only, in this test)
    call automatically; a second call while that modal is up must be a
    no-op -- no second modal stacked on top -- and the re-entrancy flag
    must clear once the (single-discovery) chain's last modal dismisses.
    """
    ledger_dir = tmp_path / "data"
    monkeypatch.setattr(_offer_module, "get_user_data_dir", lambda: ledger_dir)

    discovery = _discovery(tmp_path, names=("alpha-skill",))
    service = _StubSkillsScopeService()
    app = _OfferHarnessApp(service, (discovery,))
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.pause()

        assert isinstance(app.screen, ProjectSkillsImportModal)
        first_modal = app.screen
        assert app._project_skills_offer_active is True

        # Second call while the flow is still active: must be a no-op.
        maybe_offer_project_skills_import(app, (discovery,))
        await pilot.pause()
        await pilot.pause()
        assert app.screen is first_modal

        await pilot.click("#project-skills-never")
        await pilot.pause()
        await pilot.pause()

    assert getattr(app, "_project_skills_offer_active", False) is False

    ledger = ProjectSkillsPromptLedger(ledger_dir)
    assert ledger.decision_for(discovery.root) == ("never", discovery.fingerprint)


@pytest.mark.asyncio
async def test_on_dismiss_bookkeeping_failure_still_clears_flag_and_does_not_raise(
    tmp_path, monkeypatch
):
    """A raising ledger write inside ``_on_dismiss`` must not crash the app.

    ``_on_dismiss`` runs synchronously out of Textual's own dismiss
    handling -- NOT a worker -- so an uncaught exception there reaches
    ``App._handle_exception`` on the main thread and exits the app.
    Monkeypatches the ledger-record call to raise and drives a real
    dismissal through the pilot: the app must survive to the end of the
    ``async with`` block, and the re-entrancy flag must still end False
    (never stuck True) since this is the chain's last (only) discovery.
    """
    ledger_dir = tmp_path / "data"
    monkeypatch.setattr(_offer_module, "get_user_data_dir", lambda: ledger_dir)

    def _raise(*_args, **_kwargs):
        raise RuntimeError("ledger exploded")

    monkeypatch.setattr(_offer_module, "_record_project_skills_decision", _raise)

    discovery = _discovery(tmp_path, names=("alpha-skill",))
    service = _StubSkillsScopeService()
    app = _OfferHarnessApp(service, (discovery,))
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.pause()
        assert isinstance(app.screen, ProjectSkillsImportModal)

        await pilot.click("#project-skills-not-now")
        await pilot.pause()
        await pilot.pause()

        # The app is still alive and responsive -- an uncaught exception
        # in the un-fixed `_on_dismiss` would have exited it here.
        assert not app._exit

    assert getattr(app, "_project_skills_offer_active", False) is False


# ---------------------------------------------------------------------------
# Finding 6 (Qodo review, PR #1810): a failed import's unchanged fingerprint
# must not be recorded as "imported" in the ledger -- the .SKILLS/ folder on
# disk is untouched by a failed import, so recording "imported" anyway would
# make the ledger's unchanged-fingerprint check permanently suppress every
# future offer for a root that never actually finished importing.
# ---------------------------------------------------------------------------


class _NoInstalledSkillsScopeService(_StubSkillsScopeService):
    """Like ``_StubSkillsScopeService`` but reports nothing as installed --
    so every row defaults checked and gets attempted on Import."""

    async def get_context(self, *, mode=None):
        del mode
        return {"available_skills": [], "blocked_skills": []}


class _PartialFailureSkillsScopeService(_NoInstalledSkillsScopeService):
    """Fails ``import_skill_directory`` for one configured skill name;
    every other entry imports normally."""

    def __init__(self, fail_name: str):
        super().__init__()
        self._fail_name = fail_name

    async def import_skill_directory(self, path, *, mode, name, trust_approved):
        if name == self._fail_name:
            raise RuntimeError("import exploded")
        await super().import_skill_directory(
            path, mode=mode, name=name, trust_approved=trust_approved
        )


@pytest.mark.asyncio
async def test_partial_import_failure_leaves_ledger_untouched(tmp_path, monkeypatch):
    ledger_dir = tmp_path / "data"
    monkeypatch.setattr(_offer_module, "get_user_data_dir", lambda: ledger_dir)

    discovery = _discovery(tmp_path, names=("alpha-skill", "beta-skill"))
    service = _PartialFailureSkillsScopeService(fail_name="beta-skill")
    app = _OfferHarnessApp(service, (discovery,))
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.pause()
        assert isinstance(app.screen, ProjectSkillsImportModal)

        await pilot.click("#project-skills-import")
        await pilot.pause()
        await pilot.click("#project-skills-close")
        await pilot.pause()
        await pilot.pause()

    ledger = ProjectSkillsPromptLedger(ledger_dir)
    assert ledger.decision_for(discovery.root) is None


@pytest.mark.asyncio
async def test_fully_successful_import_records_imported_decision(
    tmp_path, monkeypatch
):
    ledger_dir = tmp_path / "data"
    monkeypatch.setattr(_offer_module, "get_user_data_dir", lambda: ledger_dir)

    discovery = _discovery(tmp_path, names=("alpha-skill",))
    service = _NoInstalledSkillsScopeService()
    app = _OfferHarnessApp(service, (discovery,))
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.pause()
        assert isinstance(app.screen, ProjectSkillsImportModal)

        await pilot.click("#project-skills-import")
        await pilot.pause()
        await pilot.click("#project-skills-close")
        await pilot.pause()
        await pilot.pause()

    ledger = ProjectSkillsPromptLedger(ledger_dir)
    assert ledger.decision_for(discovery.root) == ("imported", discovery.fingerprint)


@pytest.mark.asyncio
async def test_zero_selected_import_records_nothing(tmp_path, monkeypatch):
    """The "outcomes empty because the import never ran" half of Finding 6:
    pressing Import with nothing checked must not record "imported" either
    -- nothing actually happened, so the ledger must stay untouched."""
    ledger_dir = tmp_path / "data"
    monkeypatch.setattr(_offer_module, "get_user_data_dir", lambda: ledger_dir)

    discovery = _discovery(tmp_path, names=("alpha-skill",))
    service = _NoInstalledSkillsScopeService()
    app = _OfferHarnessApp(service, (discovery,))
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.pause()
        assert isinstance(app.screen, ProjectSkillsImportModal)

        app.screen.query_one("#project-skill-row-0", Checkbox).value = False
        await pilot.click("#project-skills-import")
        await pilot.pause()
        await pilot.click("#project-skills-close")
        await pilot.pause()
        await pilot.pause()

    ledger = ProjectSkillsPromptLedger(ledger_dir)
    assert ledger.decision_for(discovery.root) is None


# ---------------------------------------------------------------------------
# Finding 1: the create-path trigger must respect the SAME gating as the
# startup path -- kill-switch, "Never", and fingerprint gating -- not just
# scan-and-offer unconditionally. ``maybe_offer_project_skills_import`` is
# the one entry point every call site (startup, Console/Settings/Library
# create-modal chaining) routes through, so fixing gating there closes all
# of them at once.
# ---------------------------------------------------------------------------


def test_never_for_folder_silences_create_trigger(tmp_path, monkeypatch):
    """A "never" decision recorded for a discovery's root (e.g. via the
    startup offer) must silence a LATER create-path offer for the same
    folder too -- spec §5.3's "declining in one place silences the other".
    """
    ledger_dir = tmp_path / "data"
    monkeypatch.setattr(_offer_module, "get_user_data_dir", lambda: ledger_dir)
    monkeypatch.setattr(_offer_module, "get_cli_setting", lambda *a, **k: True)

    discovery = _discovery(tmp_path, names=("alpha-skill",))
    ledger = ProjectSkillsPromptLedger(ledger_dir)
    ledger.record(discovery.root, "never", discovery.fingerprint)

    app = _RecorderApp()
    maybe_offer_project_skills_import(app, (discovery,))

    assert app.pushed == []
    assert app.workers == []
    assert getattr(app, "_project_skills_offer_active", False) is False


def test_kill_switch_suppresses_create_offer(tmp_path, monkeypatch):
    """The ``[skills] project_skills_prompt_enabled`` kill-switch must be
    consulted at the top of ``maybe_offer_project_skills_import`` itself --
    not only by the startup path's own pre-filtering -- so every call site
    that routes through this one helper is covered.
    """
    monkeypatch.setattr(_offer_module, "get_cli_setting", lambda *a, **k: False)

    discovery = _discovery(tmp_path, names=("alpha-skill",))
    app = _RecorderApp()

    maybe_offer_project_skills_import(app, (discovery,))

    assert app.pushed == []
    assert app.workers == []
    assert getattr(app, "_project_skills_offer_active", False) is False


def test_missing_skills_scope_service_suppresses_offer(tmp_path, monkeypatch):
    """Finding 6: without ``app.skills_scope_service`` an offer can only
    fail (there is nothing to import into) -- suppress it entirely instead
    of pushing a modal whose "Import selected" is guaranteed to error.
    """
    monkeypatch.setattr(_offer_module, "get_cli_setting", lambda *a, **k: True)

    discovery = _discovery(tmp_path, names=("alpha-skill",))
    app = _RecorderApp(skills_scope_service=None)

    maybe_offer_project_skills_import(app, (discovery,))

    assert app.pushed == []
    assert app.workers == []
    assert getattr(app, "_project_skills_offer_active", False) is False


# ---------------------------------------------------------------------------
# Finding 7 (Qodo review, PR #1810): TOCTOU symlink swap between discovery
# and import. Discovery accepts a loose ``.md`` file, but the import-time
# read happens later (after the user reviews and presses "Import selected")
# -- a hostile/racing actor can swap the accepted file for a symlink to any
# readable file in between. The loose-file importer must re-validate
# symlink/regular-file status immediately before reading, not trust
# discovery's earlier check.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_loose_file_symlink_swap_before_import_is_refused(tmp_path):
    skill_path = tmp_path / ".SKILLS" / "swapped-skill.md"
    skill_path.parent.mkdir(parents=True)
    skill_path.write_text("---\ndescription: real.\n---\nBody\n", encoding="utf-8")
    discovery = discover_project_skills(tmp_path)
    assert discovery is not None
    entry = next(e for e in discovery.entries if e.name == "swapped-skill")

    secret = tmp_path / "secret.txt"
    secret.write_text("TOP SECRET CONTENTS", encoding="utf-8")

    # Swap the discovery-accepted regular file for a symlink pointing at an
    # arbitrary (readable) file, simulating a race between discovery and
    # the user pressing "Import selected".
    skill_path.unlink()
    os.symlink(secret, skill_path)

    calls: list = []

    class _Service:
        async def import_skill_file(self, data, **kwargs):
            calls.append(data)

        async def import_skill_directory(self, *args, **kwargs):
            raise AssertionError("directory import should not be reached")

    app = SimpleNamespace(skills_scope_service=_Service())
    importer = _project_skills_importer(app)

    with pytest.raises(ValueError, match="skill file changed on disk"):
        await importer(entry)

    assert calls == []  # the symlink target must never reach the importer


@pytest.mark.asyncio
async def test_loose_file_unswapped_import_still_succeeds(tmp_path):
    """Sanity companion to the swap test above: the re-validation added for
    Finding 7 must not break the ordinary (unswapped) loose-file import
    path."""
    skill_path = tmp_path / ".SKILLS" / "steady-skill.md"
    skill_path.parent.mkdir(parents=True)
    skill_path.write_text("---\ndescription: real.\n---\nBody\n", encoding="utf-8")
    discovery = discover_project_skills(tmp_path)
    assert discovery is not None
    entry = next(e for e in discovery.entries if e.name == "steady-skill")

    calls: list = []

    class _Service:
        async def import_skill_file(self, data, **kwargs):
            calls.append((data, kwargs))

    app = SimpleNamespace(skills_scope_service=_Service())
    importer = _project_skills_importer(app)
    await importer(entry)

    assert len(calls) == 1
    data, kwargs = calls[0]
    assert data == skill_path.read_bytes()
    assert kwargs["filename"] == "steady-skill.md"


# ---------------------------------------------------------------------------
# Finding 10 (Qodo review, PR #1810): the INITIAL ``run_worker`` call in
# ``maybe_offer_project_skills_import`` was unguarded -- unlike the
# continuation's own ``run_worker`` call in ``_on_dismiss``, a scheduling
# failure here (e.g. the app's task runner rejecting the call) left
# ``_project_skills_offer_active`` stuck True forever, permanently
# silencing every future offer for the app's whole session.
# ---------------------------------------------------------------------------


class _RaisingOnceRunWorkerApp(_RecorderApp):
    """``run_worker`` raises on its first call only; succeeds after."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._fail_next = True

    def run_worker(self, coro, **kwargs):
        if self._fail_next:
            self._fail_next = False
            # Deliberately does NOT close `coro` here -- closing it is the
            # fix's own responsibility (Finding 10), not this double's; a
            # correct fix closes it in its `except` branch so no
            # "coroutine was never awaited" warning escapes this test.
            raise RuntimeError("scheduling exploded")
        return super().run_worker(coro, **kwargs)


def test_initial_scheduling_failure_clears_flag_and_allows_retry(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(_offer_module, "get_cli_setting", lambda *a, **k: True)
    monkeypatch.setattr(_offer_module, "get_user_data_dir", lambda: tmp_path / "data")

    discovery = _discovery(tmp_path, names=("alpha-skill",))
    app = _RaisingOnceRunWorkerApp()

    maybe_offer_project_skills_import(app, (discovery,))

    # The failed scheduling attempt must not propagate, and must not leave
    # the re-entrancy flag stuck True.
    assert app.workers == []
    assert getattr(app, "_project_skills_offer_active", False) is False

    # A second call must not be blocked by a stuck flag -- it schedules
    # normally now that run_worker no longer raises.
    maybe_offer_project_skills_import(app, (discovery,))
    assert len(app.workers) == 1
    # `_RecorderApp.run_worker` only records the coroutine, it never
    # awaits it (no event loop here) -- close it explicitly so pytest's
    # gc-based leak detector doesn't flag it as never-awaited.
    app.workers[0].close()
