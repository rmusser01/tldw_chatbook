import asyncio

import pytest
from textual import on
from textual.app import App
from textual.widgets import Checkbox

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
