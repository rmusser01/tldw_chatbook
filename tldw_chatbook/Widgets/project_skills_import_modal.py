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
import os
import stat
from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path
from typing import Any

from loguru import logger
from rich.markup import escape as escape_markup
from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Checkbox, Static

from tldw_chatbook.config import get_cli_setting, get_user_data_dir
from tldw_chatbook.Skills_Interop.project_skills_discovery import (
    ProjectSkillEntry,
    ProjectSkillsDiscovery,
)
from tldw_chatbook.Skills_Interop.project_skills_prompt import (
    ProjectSkillsPromptLedger,
    should_offer_project_skills_prompt,
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

#: Decision -> ledger verb for the two decisions whose mapping does NOT
#: depend on per-entry outcomes (spec §5.3/§5.5). "imported"/"review" are
#: handled separately in ``_record_project_skills_decision`` below: review
#: still counts as imported when the import fully succeeded (the entries
#: were already copied in; "review" only changes what happens after
#: dismissal), but -- Qodo finding 6, PR #1810 -- NEITHER is recorded when
#: any attempted entry failed, or none were attempted at all. A failed
#: import leaves the ``.SKILLS/`` folder's fingerprint unchanged, so
#: recording "imported" anyway would make the ledger's unchanged-
#: fingerprint check (``should_offer_project_skills_prompt``) permanently
#: suppress every future offer for a root that never actually finished
#: importing.
_LEDGER_DECISIONS = {
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

    def _import_in_flight(self) -> bool:
        """True only while a committed import hasn't produced outcomes yet.

        Finding 2 (final review 2026-08-17): the offer-phase buttons stay
        mounted while ``_run_import`` is running (the modal only recomposes
        into the results phase once ``self._outcomes`` is set), so a click
        can still land on "Never"/"Not now", or Escape can still fire,
        while an import the user already committed to is mid-flight. All
        three must be inert then -- dismissing (or re-running) would either
        discard a partial import silently or race the in-flight worker.

        TASK-17964 (dismissal posture, decided and NOT changed): the
        consequence of this guard is that a permanently-hung importer
        (``self._importer`` never returning/raising) leaves no in-modal
        exit -- Escape, backdrop-click, and "Not now"/"Never" all stay
        inert for as long as ``_import_in_flight()`` is true, with no
        timeout. This is an accepted trade-off, not an oversight: imports
        here are local-filesystem only (directory copy or loose-file read,
        see ``_project_skills_importer``), so a hung importer implies a
        hung filesystem -- the same failure class the off-thread-discovery
        decision above names for ``_add_folder``. Discarding a partial
        import silently (what an escape hatch would have to do while the
        worker is still running) is worse than leaving no exit for a
        pathological, previously-unseen case: it risks a half-imported
        skill directory with no way for the user to know it happened. If a
        real hang is ever observed in practice, a bounded timeout on the
        importer call is the fix to file -- not reopening the guard here.
        """
        return self._committed and self._outcomes is None

    async def _perform_safe_cancel(self, *, source: str) -> None:
        # Escape / backdrop-click / "Not now" all route here.
        del source
        if self._import_in_flight():
            return
        if self._outcomes is not None:
            # Results phase (minor 6): escape/backdrop must dismiss exactly
            # like "Close" -- the import already ran, there is nothing left
            # to cancel, and mislabeling it "not_now" would make the
            # already-completed import look declined to the ledger.
            self.dismiss_safe_once(("imported", self._outcomes))
            return
        self.dismiss_safe_once(("not_now", None))

    @on(Button.Pressed, "#project-skills-not-now")
    async def _not_now(self, event: Button.Pressed) -> None:
        event.stop()
        await self.request_safe_cancel(source="button")

    @on(Button.Pressed, "#project-skills-never")
    def _never(self, event: Button.Pressed) -> None:
        event.stop()
        if self._import_in_flight():
            return
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
        # Belt-and-suspenders alongside the `_import_in_flight()` guards
        # above: visibly disable the offer-phase buttons too, so a
        # still-mounted "Never"/"Not now" doesn't even look clickable while
        # the import it would discard is running.
        for button_id in (
            "#project-skills-import",
            "#project-skills-not-now",
            "#project-skills-never",
        ):
            try:
                self.query_one(button_id, Button).disabled = True
            except Exception:  # noqa: BLE001 - purely cosmetic, never fatal
                pass
        selected = self._selected_entries()
        self.run_worker(
            self._run_import(selected),
            exclusive=True,
            group="project-skills-run-import",
        )

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


def _read_loose_skill_file_sync(path: Path) -> bytes:
    """Re-validate and read a loose-file skill body at import time.

    Finding 7 (Qodo review, PR #1810): discovery accepts a loose ``.md``
    file well before the user presses "Import selected" -- in between, a
    hostile or racing actor can swap the accepted regular file for a
    symlink pointing at any other readable file on disk. Reading via
    ``entry.path.read_bytes()`` at import time would silently follow that
    swap and ingest the symlink's target instead of the file the user
    actually reviewed.

    This re-checks symlink/regular-file status via ``os.lstat`` (which,
    unlike ``stat``, does not itself follow a symlink) immediately before
    reading, then opens with ``O_NOFOLLOW`` where the platform supports it
    (macOS and Linux both do) so even a symlink planted in the TOCTOU
    window between the ``lstat`` check and the ``open`` call is refused by
    the kernel rather than followed.

    Args:
        path: The loose skill file's path, as recorded on the
            ``ProjectSkillEntry`` at discovery time.

    Returns:
        The file's raw bytes.

    Raises:
        ValueError: The path is now a symlink, is no longer a regular
            file, or otherwise could not be opened/read for import --
            reported uniformly as "skill file changed on disk" so a caller
            never has to distinguish a swap from an ordinary race with
            deletion.
    """
    try:
        info = os.lstat(path)
    except OSError as exc:
        raise ValueError("skill file changed on disk") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise ValueError("skill file changed on disk")
    o_nofollow = getattr(os, "O_NOFOLLOW", None)
    if o_nofollow is None:
        # Platform without O_NOFOLLOW: the lstat check above is the only
        # protection against a same-window swap, but a regular open() is
        # still correct for the (overwhelmingly common) unswapped case.
        try:
            return path.read_bytes()
        except OSError as exc:
            raise ValueError("skill file changed on disk") from exc
    try:
        fd = os.open(path, os.O_RDONLY | o_nofollow)
    except OSError as exc:
        # ELOOP (symlink) or ENOENT/others (deleted/replaced) alike: same
        # clean, uniform error either way.
        raise ValueError("skill file changed on disk") from exc
    try:
        with os.fdopen(fd, "rb") as handle:
            return handle.read()
    except OSError as exc:
        raise ValueError("skill file changed on disk") from exc


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
        data = await asyncio.to_thread(_read_loose_skill_file_sync, entry.path)
        await import_skill_file(
            data,
            mode="local",
            filename=entry.path.name,
            content_type="text/markdown",
            trust_approved=False,
        )

    return _importer


def _record_project_skills_decision(
    discovery: ProjectSkillsDiscovery,
    decision: str,
    outcomes: tuple[ImportOutcome, ...] | None,
) -> None:
    """Record one discovery's dismissal in the prompt ledger, or suppress it.

    Finding 6 (Qodo review, PR #1810): "imported"/"review" are recorded as
    ``"imported"`` ONLY when every attempted entry actually succeeded
    (``outcome == "imported"`` for all of them). When any attempted entry
    failed, or nothing was attempted at all (``outcomes`` empty or
    ``None`` -- e.g. the user pressed Import with nothing checked),
    NOTHING is recorded for this root: the ``.SKILLS/`` folder's
    fingerprint is unchanged by a failed/no-op import, so recording
    ``"imported"`` anyway would make the ledger's unchanged-fingerprint
    check permanently suppress every future offer even though nothing
    actually succeeded. Leaving prior ledger state untouched means the
    next trigger re-offers -- entries that DID succeed naturally render
    "(already installed)" on that re-offer.

    "not_now"/"never" are unaffected by ``outcomes`` -- see
    ``_LEDGER_DECISIONS``.

    Args:
        discovery: The discovery being dismissed.
        decision: One of ``"imported"``, ``"review"``, ``"not_now"``,
            ``"never"``.
        outcomes: The per-entry ``(name, "imported" | error text)`` results
            from the modal's import run, or ``None`` when no import ran
            (declined/never decisions).
    """
    if decision in ("imported", "review"):
        if not outcomes or any(message != "imported" for _name, message in outcomes):
            return
        ledger_decision = "imported"
    else:
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
        try:
            decision, _outcomes = result
            _record_project_skills_decision(discovery, decision, _outcomes)
            if decision == "review":
                app.post_message(NavigateToScreen("skills"))
        except Exception:
            # The ledger write and the review-navigation post are both
            # best-effort: this callback runs synchronously out of
            # Textual's own dismiss handling (NOT a worker), so an
            # uncaught exception here would reach App._handle_exception on
            # the main thread and exit the whole app over what is, at
            # worst, a lost ledger record or a missed navigation hint.
            logger.opt(exception=True).debug(
                "project-skills dismissal bookkeeping failed"
            )
        finally:
            # The "last discovery -> clear flag" vs "recurse to next"
            # decision lives in `finally`, not the `try` above, so the
            # re-entrancy flag can NEVER stay stuck True -- it clears (or
            # the chain continues) even when the bookkeeping above raised.
            #
            # Continuing here calls `_run_project_skills_offer_chain`
            # directly rather than routing back through the public entry
            # point below -- that guard would see
            # ``_project_skills_offer_active`` still True (it stays True
            # for the whole chain, cleared only at this terminal step) and
            # mistake this continuation for a second concurrent caller,
            # dropping it.
            if not remaining:
                app._project_skills_offer_active = False
            else:
                # The coroutine is created before `run_worker` sees it (it
                # has to be -- it's the call's own argument), but a
                # `run_worker` failure must not leak it as a never-awaited
                # coroutine (Finding 7, final review 2026-08-17): keep the
                # reference so the except branch can close it explicitly
                # instead of letting Python warn-and-leak an orphaned one.
                continuation = _run_project_skills_offer_chain(app, remaining)
                try:
                    app.run_worker(continuation, exclusive=False)
                except Exception:
                    continuation.close()
                    app._project_skills_offer_active = False
                    logger.opt(exception=True).debug(
                        "project-skills offer chain continuation failed to start"
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

    The one entry point EVERY call site uses -- startup and every
    workspace-create surface (Console, Settings, Library) -- and therefore
    the one choke point where gating lives (final review 2026-08-17,
    Finding 1). Previously only the startup path pre-filtered discoveries
    through the kill-switch/ledger before calling here, so a "Never for
    this folder" or a disabled kill-switch recorded via one trigger did not
    silence the other, violating spec §5.3's "declining in one place
    silences the other". This function now applies the SAME gating no
    matter which trigger calls it:

    1. The ``[skills] project_skills_prompt_enabled`` kill-switch is
       checked first -- off means no offer, full stop.
    2. Without a usable ``app.skills_scope_service`` an offer can only
       fail (Finding 6): suppressed rather than pushing a modal whose
       "Import selected" is guaranteed to error.
    3. Each discovery is re-checked against the prompt ledger
       (``should_offer_project_skills_prompt`` -- never-seen, or
       changed-and-not-"never"); a discovery already declined "never", or
       unchanged since it was last shown, is dropped. If none remain,
       nothing is offered.

    For any survivors: builds ``installed_names``/an ``importer`` from the
    live ``app.skills_scope_service``, then pushes one modal per discovery
    -- the next discovery's modal is only pushed once the current one
    dismisses. Every dismissal is passed through
    ``_record_project_skills_decision`` (``ProjectSkillsPromptLedger``,
    keyed by ``discovery.root``): "not_now"/"never" record
    "declined"/"never" unconditionally, but "imported"/"review" record
    "imported" ONLY when every attempted entry actually succeeded --
    otherwise (any entry failed, or nothing was attempted) NOTHING is
    recorded, leaving prior ledger state untouched so the next trigger
    re-offers (Qodo finding 6, PR #1810: a failed import's unchanged
    ``.SKILLS/`` fingerprint must not permanently suppress retries). A
    "review" decision also posts a navigation request to the Skills
    screen. (The startup path's own pre-filtering in
    ``startup_discovery_for`` becomes redundant with this, but idempotent
    -- harmless double gating.)

    Re-entrancy: while one offer flow is already running for ``app``
    (tracked via ``app._project_skills_offer_active``, set here when the
    flow starts), a second call is a no-op. This stops two independent
    callers -- e.g. the startup offer and a workspace-create offer -- from
    both trying to push a modal for the same app at once. The flag is
    guaranteed to clear -- it can never stay stuck True and permanently
    silence the feature for the rest of the session -- because every path
    that can end the flow clears it: the chain worker's own failure
    (``_run_project_skills_offer_chain``'s ``except``), a failure
    scheduling the next discovery's worker, a failure scheduling this
    INITIAL worker (Qodo finding 10, PR #1810: mirrors the continuation's
    own guard -- the coroutine is created up front so a raising
    ``run_worker`` can still be closed explicitly instead of leaking a
    never-awaited coroutine), and (via a ``finally``, so it runs even if
    the ledger write or the review-navigation post above it raised) the
    dismissal callback's terminal step when the last discovery in the
    chain dismisses.

    Args:
        app: The running ``TldwCli`` app (or a test double exposing
            ``skills_scope_service``, ``push_screen``, ``post_message``, and
            ``run_worker``).
        discoveries: Discoveries to offer, one modal at a time.
    """
    if not get_cli_setting("skills", "project_skills_prompt_enabled", True):
        logger.debug(
            "project-skills offer suppressed: "
            "[skills] project_skills_prompt_enabled is off"
        )
        return
    if getattr(app, "_project_skills_offer_active", False):
        return
    discoveries = tuple(discoveries)
    if not discoveries:
        return
    if getattr(app, "skills_scope_service", None) is None:
        logger.debug(
            "project-skills offer suppressed: app.skills_scope_service is unavailable"
        )
        return
    ledger = ProjectSkillsPromptLedger(get_user_data_dir())
    gated_discoveries = tuple(
        discovery
        for discovery in discoveries
        if should_offer_project_skills_prompt(
            True, ledger.decision_for(discovery.root), discovery.fingerprint
        )
    )
    if not gated_discoveries:
        return
    app._project_skills_offer_active = True
    # The coroutine is created before `run_worker` sees it (same shape as
    # the continuation's own guard in `_on_dismiss`): a `run_worker`
    # failure here must not leave `_project_skills_offer_active` stuck
    # True forever (Finding 10, Qodo review 2026-08-17) -- close the
    # never-started coroutine explicitly instead of letting Python
    # warn-and-leak an orphaned one, clear the flag, and never propagate.
    coroutine = _run_project_skills_offer_chain(app, gated_discoveries)
    try:
        app.run_worker(coroutine, exclusive=False)
    except Exception:
        coroutine.close()
        app._project_skills_offer_active = False
        logger.opt(exception=True).debug(
            "project-skills offer chain failed to start"
        )
