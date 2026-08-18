"""Project-skills import modal + shared offer helper (spec 2026-08-17 §5.5).

``ProjectSkillsImportModal`` presents one project's ``.SKILLS/`` discovery
(Task 1) and lets the user pick which entries to import; import execution
itself is injected (``importer``) so this module never talks to a skills
store directly -- the modal only decides WHAT to import and reports WHAT
happened.

``maybe_offer_project_skills_import`` is the one entry point other modules
call: it wires the injected importer/``installed_names`` from the live
``app.skills_scope_service``, chains one modal per discovery, and records
each dismissal in the prompt ledger (Task 2).
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable, Sequence
from typing import Any

from loguru import logger
from rich.markup import escape as escape_markup
from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Checkbox, Static

from tldw_chatbook.config import get_user_data_dir
from tldw_chatbook.Skills_Interop.project_skills_discovery import (
    ProjectSkillEntry,
    ProjectSkillsDiscovery,
)
from tldw_chatbook.Skills_Interop.project_skills_prompt import (
    ProjectSkillsPromptLedger,
)
from tldw_chatbook.UI.Navigation.main_navigation import NavigateToScreen
from tldw_chatbook.Widgets.modal_dismissal import SafeModalDismissMixin

#: Verbatim trust-expectation line, spec 2026-08-17 §5.5.
_TRUST_LINE = (
    "Imported skills require a one-time trust review in Library ▸ Skills "
    "before they can run."
)
#: Bootstrap-aware trust framing shown on the post-import results phase.
_POST_IMPORT_TRUST_LINE = (
    "Set up skill trust if this is your first skill, then approve each one."
)

#: ``async (entry) -> None``; raises on failure (caught + reported per-entry).
Importer = Callable[[ProjectSkillEntry], Awaitable[None]]
#: ``(name, "imported" | error text)``.
ImportOutcome = tuple[str, str]
#: ``("imported" | "not_now" | "never" | "review", outcomes-or-None)``.
ImportDecision = tuple[str, tuple[ImportOutcome, ...] | None]

#: Decision -> ledger verb (spec §5.3/§5.5): review still counts as imported
#: (the entries were already copied in; "review" only changes what happens
#: after dismissal, not whether the import itself is recorded).
_LEDGER_DECISIONS = {
    "imported": "imported",
    "review": "imported",
    "not_now": "declined",
    "never": "never",
}


class ProjectSkillsImportModal(SafeModalDismissMixin, ModalScreen[ImportDecision]):
    """Offer to import a discovered project's ``.SKILLS/`` folder.

    Args:
        discovery: The project-skills discovery to offer.
        installed_names: Skill names already installed locally -- their rows
            render unchecked and labeled "(already installed)".
        importer: Injected ``async (entry) -> None`` that performs one
            entry's import, raising on failure. Never called directly by
            this modal for anything except an "Import selected" press.
    """

    DEFAULT_CSS = """
    ProjectSkillsImportModal {
        align: center middle;
    }

    #project-skills-modal {
        width: 76;
        height: auto;
        max-height: 34;
        border: tall gray;
        background: black;
        padding: 1 2;
    }

    #project-skills-provenance {
        color: $text-muted;
        margin: 0 0 1 0;
    }

    #project-skills-trust-line {
        color: $text-muted;
        margin: 0 0 1 0;
    }

    #project-skills-post-trust-line {
        color: $text-muted;
        margin: 0 0 1 0;
    }

    #project-skills-rows {
        height: auto;
        max-height: 18;
        margin: 0 0 1 0;
    }

    #project-skills-results {
        height: auto;
        max-height: 18;
        margin: 0 0 1 0;
    }

    #project-skills-footer {
        color: $text-muted;
        height: auto;
    }

    #project-skills-actions {
        height: auto;
        margin: 1 0 0 0;
        align-horizontal: right;
    }
    """

    SAFE_MODAL_CONTENT = "#project-skills-modal"
    BINDINGS = [("escape", "request_safe_cancel", "Cancel")]
    AUTO_FOCUS = "#project-skills-import"

    def __init__(
        self,
        *,
        discovery: ProjectSkillsDiscovery,
        installed_names: frozenset[str],
        importer: Importer,
    ) -> None:
        super().__init__()
        self._discovery = discovery
        self._installed_names = installed_names
        self._importer = importer
        # One-shot guard: a double "Import selected" press (rapid double
        # click, or Enter-Enter on the focused button) must run the injected
        # importer at most once per entry -- see WorkspaceCreateModal's
        # ``_committed`` precedent for this exact shape.
        self._committed = False
        self._outcomes: tuple[ImportOutcome, ...] | None = None

    def compose(self) -> ComposeResult:
        with Vertical(id="project-skills-modal"):
            yield Static(
                "Project skills found", classes="console-modal-header", markup=False
            )
            if self._outcomes is None:
                yield from self._compose_offer_phase()
            else:
                yield from self._compose_results_phase()

    def _compose_offer_phase(self) -> ComposeResult:
        yield Static(
            f"Found in {self._discovery.skills_dir}",
            id="project-skills-provenance",
            markup=False,
        )
        yield Static(_TRUST_LINE, id="project-skills-trust-line", markup=False)
        with Vertical(id="project-skills-rows"):
            for index, entry in enumerate(self._discovery.entries):
                if entry.status == "ok":
                    installed = entry.name in self._installed_names
                    label = f"{entry.name} — {entry.description}"
                    if installed:
                        label += " (already installed)"
                    # Checkbox labels render markup; every piece of that
                    # label is a repo-sourced string (name/description), so
                    # it must be escaped before Textual interprets it.
                    yield Checkbox(
                        escape_markup(label),
                        not installed,
                        id=f"project-skill-row-{index}",
                    )
                else:
                    yield Static(
                        f"{entry.name} — invalid: {entry.reason}",
                        markup=False,
                    )
        footer_lines = self._offer_footer_lines()
        if footer_lines:
            yield Static(
                "\n".join(footer_lines), id="project-skills-footer", markup=False
            )
        with Horizontal(id="project-skills-actions"):
            yield Button("Import selected", id="project-skills-import", compact=True)
            yield Button("Not now", id="project-skills-not-now", compact=True)
            yield Button(
                "Never for this folder", id="project-skills-never", compact=True
            )

    def _offer_footer_lines(self) -> list[str]:
        lines: list[str] = []
        if self._discovery.skipped:
            names = ", ".join(name for name, _reason in self._discovery.skipped)
            lines.append(f"Skipped: {names}")
        if self._discovery.truncated:
            lines.append(f"{self._discovery.truncated} more not shown")
        return lines

    def _compose_results_phase(self) -> ComposeResult:
        assert self._outcomes is not None
        with Vertical(id="project-skills-results"):
            for name, message in self._outcomes:
                yield Static(f"{name}: {message}", markup=False)
        yield Static(
            _POST_IMPORT_TRUST_LINE,
            id="project-skills-post-trust-line",
            markup=False,
        )
        with Horizontal(id="project-skills-actions"):
            yield Button(
                "Review in Library ▸ Skills",
                id="project-skills-review",
                compact=True,
            )
            yield Button("Close", id="project-skills-close", compact=True)

    async def _perform_safe_cancel(self, *, source: str) -> None:
        # Escape / backdrop-click / "Not now" all route here, and must all
        # dismiss with the same ("not_now", None) decision tuple.
        del source
        self.dismiss_safe_once(("not_now", None))

    @on(Button.Pressed, "#project-skills-not-now")
    async def _not_now(self, event: Button.Pressed) -> None:
        event.stop()
        await self.request_safe_cancel(source="button")

    @on(Button.Pressed, "#project-skills-never")
    def _never(self, event: Button.Pressed) -> None:
        event.stop()
        self.dismiss_safe_once(("never", None))

    def _selected_entries(self) -> tuple[ProjectSkillEntry, ...]:
        selected: list[ProjectSkillEntry] = []
        for index, entry in enumerate(self._discovery.entries):
            if entry.status != "ok":
                continue
            checkbox = self.query_one(f"#project-skill-row-{index}", Checkbox)
            if checkbox.value:
                selected.append(entry)
        return tuple(selected)

    @on(Button.Pressed, "#project-skills-import")
    def _import_selected(self, event: Button.Pressed) -> None:
        event.stop()
        if self._committed:
            return
        self._committed = True
        selected = self._selected_entries()
        self.run_worker(self._run_import(selected), exclusive=True)

    async def _run_import(self, entries: tuple[ProjectSkillEntry, ...]) -> None:
        outcomes: list[ImportOutcome] = []
        for entry in entries:
            try:
                await self._importer(entry)
            except Exception as exc:  # noqa: BLE001 - reported per-entry, not raised
                outcomes.append((entry.name, str(exc)))
            else:
                outcomes.append((entry.name, "imported"))
        self._outcomes = tuple(outcomes)
        if self.is_mounted:
            self.refresh(recompose=True)

    @on(Button.Pressed, "#project-skills-review")
    def _review(self, event: Button.Pressed) -> None:
        event.stop()
        if self._outcomes is None:
            return
        self.dismiss_safe_once(("review", self._outcomes))

    @on(Button.Pressed, "#project-skills-close")
    def _close(self, event: Button.Pressed) -> None:
        event.stop()
        if self._outcomes is None:
            return
        self.dismiss_safe_once(("imported", self._outcomes))


async def _project_skills_installed_names(app: Any) -> frozenset[str]:
    """Best-effort local skill-name set from ``app.skills_scope_service``.

    Degrades to an empty set (every row renders as "new") on any lookup
    failure -- an offer to import must never crash because the installed-
    names lookup failed.
    """
    service = getattr(app, "skills_scope_service", None)
    get_context = getattr(service, "get_context", None)
    if not callable(get_context):
        return frozenset()
    try:
        payload = get_context(mode="local")
        if inspect.isawaitable(payload):
            payload = await payload
    except Exception:
        logger.opt(exception=True).debug(
            "project-skills installed-name lookup failed"
        )
        return frozenset()
    if not isinstance(payload, dict):
        return frozenset()
    names: set[str] = set()
    for key in ("available_skills", "blocked_skills"):
        for item in payload.get(key) or ():
            if isinstance(item, dict):
                name = item.get("name")
                if name:
                    names.add(str(name))
    return frozenset(names)


def _project_skills_importer(app: Any) -> Importer:
    """Build the injected importer from ``app.skills_scope_service``.

    Mirrors the exact call shapes the Library skills-import flow uses
    (``library_screen.py``'s ``_run_library_skills_import``/
    ``_import_library_skill_from_loose_file``): a directory entry imports
    via ``import_skill_directory`` (preserving the whole tree faithfully),
    a loose ``.md`` file entry imports via ``import_skill_file``. Both land
    TRUST-PENDING (``trust_approved=False``).
    """
    service = getattr(app, "skills_scope_service", None)
    import_skill_directory = getattr(service, "import_skill_directory", None)
    import_skill_file = getattr(service, "import_skill_file", None)

    async def _importer(entry: ProjectSkillEntry) -> None:
        if entry.kind == "directory":
            if not callable(import_skill_directory):
                raise RuntimeError("Skill import is unavailable.")
            await import_skill_directory(
                entry.path,
                mode="local",
                name=entry.name,
                trust_approved=False,
            )
            return
        if not callable(import_skill_file):
            raise RuntimeError("Skill import is unavailable.")
        data = await asyncio.to_thread(entry.path.read_bytes)
        await import_skill_file(
            data,
            mode="local",
            filename=entry.path.name,
            content_type="text/markdown",
            trust_approved=False,
        )

    return _importer


def _record_project_skills_decision(
    discovery: ProjectSkillsDiscovery, decision: str
) -> None:
    ledger_decision = _LEDGER_DECISIONS.get(decision)
    if ledger_decision is None:
        return
    ledger = ProjectSkillsPromptLedger(get_user_data_dir())
    ledger.record(discovery.root, ledger_decision, discovery.fingerprint)


async def _offer_next_project_skills_discovery(
    app: Any, discoveries: tuple[ProjectSkillsDiscovery, ...]
) -> None:
    discovery, remaining = discoveries[0], discoveries[1:]
    installed_names = await _project_skills_installed_names(app)
    importer = _project_skills_importer(app)
    modal = ProjectSkillsImportModal(
        discovery=discovery,
        installed_names=installed_names,
        importer=importer,
    )

    def _on_dismiss(result: ImportDecision) -> None:
        decision, _outcomes = result
        _record_project_skills_decision(discovery, decision)
        if decision == "review":
            app.post_message(NavigateToScreen("skills"))
        # Continue the SAME already-active chain directly rather than
        # routing back through the public entry point below -- the
        # re-entrancy guard there would see ``_project_skills_offer_active``
        # still True (it stays True for the whole chain, cleared only at
        # the terminal step here) and mistake this continuation for a
        # second concurrent caller, dropping it.
        if not remaining:
            app._project_skills_offer_active = False
            return
        app.run_worker(
            _run_project_skills_offer_chain(app, remaining),
            exclusive=False,
        )

    app.push_screen(modal, _on_dismiss)


async def _run_project_skills_offer_chain(
    app: Any, discoveries: tuple[ProjectSkillsDiscovery, ...]
) -> None:
    """Worker body for one offer flow; clears the re-entrancy flag on failure.

    An exception here (e.g. ``push_screen`` raising) would otherwise leave
    ``_project_skills_offer_active`` stuck True forever, permanently
    blocking every future offer for this app -- this is the "on worker
    failure" half of the re-entrancy guard's clear condition.
    """
    try:
        await _offer_next_project_skills_discovery(app, discoveries)
    except Exception:
        app._project_skills_offer_active = False
        logger.opt(exception=True).debug("project-skills offer chain failed")


def maybe_offer_project_skills_import(
    app: Any, discoveries: Sequence[ProjectSkillsDiscovery]
) -> None:
    """Offer imports for each discovery in turn (spec §5.5).

    The one entry point other call sites use (startup, workspace-create
    chaining). Builds ``installed_names``/an ``importer`` from the live
    ``app.skills_scope_service``, then pushes one modal per discovery --
    the next discovery's modal is only pushed once the current one
    dismisses. Every dismissal is recorded in the prompt ledger
    (``ProjectSkillsPromptLedger``, keyed by ``discovery.root``), and a
    "review" decision also posts a navigation request to the Skills screen.

    Re-entrancy: while one offer flow is already running for ``app``
    (tracked via ``app._project_skills_offer_active``, set here when the
    flow starts and cleared when the chain's last modal dismisses or the
    worker fails), a second call is a no-op. This stops two independent
    callers -- e.g. the startup offer and a workspace-create offer -- from
    both trying to push a modal for the same app at once.

    Args:
        app: The running ``TldwCli`` app (or a test double exposing
            ``skills_scope_service``, ``push_screen``, ``post_message``, and
            ``run_worker``).
        discoveries: Discoveries to offer, one modal at a time.
    """
    if getattr(app, "_project_skills_offer_active", False):
        return
    discoveries = tuple(discoveries)
    if not discoveries:
        return
    app._project_skills_offer_active = True
    app.run_worker(
        _run_project_skills_offer_chain(app, discoveries),
        exclusive=False,
    )
