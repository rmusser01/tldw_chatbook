"""Census: every modal that refuses or guards its own close has a Ctrl+Q decision.

TASK-33622.15. Ctrl+Q is a priority binding (TASK-33622.10), so the quit flow
runs while any modal is open and asks each one's ``confirm_quit`` first. A
modal that refuses Escape while it works, or asks before Escape discards
something, protects work that Ctrl+Q would otherwise quit straight past.
Every review round on PR #2949 found another such modal by hand, so this
census finds them statically instead and makes each one carry a decision:

* ``CONFIRM_QUIT`` -- the modal answers the quit walk itself
  (``refuse_quit_while_working`` while an operation runs, the
  ``confirm_quit_discarding_edits`` prompt before a discard); or
* an exemption naming why quitting loses nothing the guard protects.

**What counts as a guard.** A modal's close entry points are its Escape
binding's action and, on ``SafeModalDismissMixin`` modals, ``_perform_safe_
cancel`` (Escape, a backdrop click and most Cancel buttons route there). An
entry point GUARDS the close when some path through it -- following
``self.`` calls into the class's own methods -- returns without dismissing,
or pushes another screen (an "are you sure?"). Guards are keyed by the
class that WRITES the guarding method, so one row covers every subclass
that inherits it; a ``CONFIRM_QUIT`` row is checked on every concrete modal
that inherits the guard.

**What it does not see.** A guard written only in a Button handler or an
``on_key`` override, outside both entry points, is invisible here; so is a
dismissal through a helper outside the class, and a class whose header the
cheap pre-pass cannot read (a base expression containing ``)``). Those are
for review. The pre-pass exists because parsing all ~2,700 modules took
over a minute; it keeps the ~150 that define a modal or an ancestor of one,
and finds exactly the guards a full parse found when it was written.

The test fails on a guard with no row (a new one appeared), a row with no
guard (stale), and a ``CONFIRM_QUIT`` row whose modal has no
``confirm_quit``.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
import re

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE = REPO_ROOT / "tldw_chatbook"

#: The decision for a guard whose modal answers the quit walk itself.
CONFIRM_QUIT = "confirm_quit"
#: Calls that close the modal.
_DISMISSALS = frozenset({"dismiss", "dismiss_safe_once", "pop_screen"})
#: Calls that put another screen (a question) over the modal.
_PUSHES = frozenset({"push_screen", "push_screen_wait"})
#: The mixin entry point every one of its modals routes Escape/backdrop to.
_SAFE_CANCEL = "_perform_safe_cancel"
#: Escape actions that are the mixin's request, which ends in _SAFE_CANCEL.
_SAFE_REQUESTS = frozenset({"request_safe_cancel", "action_request_safe_cancel"})

#: Every guarded close, keyed ``path:Class.method`` where the guard is
#: written, with its Ctrl+Q decision. Add a row for a new guard: prefer
#: ``CONFIRM_QUIT``; an exemption must say why quitting loses nothing.
GUARDED_CLOSE_DECISIONS: dict[str, str] = {
    # --- Refuse to close while an operation runs: Ctrl+Q says "still
    # working" and stays (refuse_quit_while_working, TASK-33622.15).
    "tldw_chatbook/Widgets/Console/console_fork_chat_modal.py:"
    "ConsoleForkChatModal._perform_safe_cancel": CONFIRM_QUIT,
    # TASK-33628.5: the message-Delete receipt while the delete or its Undo
    # is still being saved off the event loop.
    "tldw_chatbook/Widgets/Console/console_message_delete_receipt.py:"
    "ConsoleMessageDeleteReceiptModal._perform_safe_cancel": CONFIRM_QUIT,
    "tldw_chatbook/Widgets/Console/console_capture_policy_dialog.py:"
    "ConsoleCapturePolicyDialog._perform_safe_cancel": CONFIRM_QUIT,
    "tldw_chatbook/Widgets/Console/console_capture_policy_dialog.py:"
    "ConsoleTracePrivacyDialog._perform_safe_cancel": CONFIRM_QUIT,
    "tldw_chatbook/Widgets/project_skills_import_modal.py:"
    "ProjectSkillsImportModal._perform_safe_cancel": CONFIRM_QUIT,
    "tldw_chatbook/Widgets/Console/console_exchange_export_dialog.py:"
    "ConsoleExchangeExportDialog._perform_safe_cancel": CONFIRM_QUIT,
    "tldw_chatbook/Widgets/Console/trace_export_dialog.py:"
    "TraceExportDialog._perform_safe_cancel": CONFIRM_QUIT,
    "tldw_chatbook/UI/Library_Modules/prompt_collection_manager_modal.py:"
    "PromptCollectionManagerModal._perform_safe_cancel": CONFIRM_QUIT,
    "tldw_chatbook/Widgets/Persona_Widgets/buddy_character_review.py:"
    "BuddyCharacterReviewDialog._perform_safe_cancel": CONFIRM_QUIT,
    "tldw_chatbook/Widgets/Settings_Widgets/personal_context_review_modal.py:"
    "PersonalContextProposalReviewModal._perform_safe_cancel": CONFIRM_QUIT,
    "tldw_chatbook/Widgets/Settings_Widgets/personal_context_review_modal.py:"
    "PersonalContextReviewModal._perform_safe_cancel": CONFIRM_QUIT,
    "tldw_chatbook/UI/Watchlists_Modules/bulk_sources_modal.py:"
    "BulkSourcesModal.action_cancel": CONFIRM_QUIT,
    "tldw_chatbook/Widgets/Console/console_session_switcher_modal.py:"
    "ConsoleSessionSwitcherModal._perform_safe_cancel": CONFIRM_QUIT,
    # --- Ask before Escape discards something: Ctrl+Q asks the same
    # (confirm_quit_discarding_edits, TASK-33622.10 / .14 / .15).
    "tldw_chatbook/UI/Screens/scheduling/forms/reminder_form.py:"
    "ReminderForm.action_dismiss": CONFIRM_QUIT,
    "tldw_chatbook/UI/Screens/scheduling/forms/automation_definition_form.py:"
    "AutomationDefinitionForm.action_dismiss": CONFIRM_QUIT,
    "tldw_chatbook/Widgets/Console/console_library_access_modal.py:"
    "ConsoleLibraryAccessModal._perform_safe_cancel": CONFIRM_QUIT,
    "tldw_chatbook/Widgets/Console/console_prompts_modal.py:"
    "ConsolePromptsModal._perform_safe_cancel": CONFIRM_QUIT,
    "tldw_chatbook/Widgets/Console/console_settings_unsaved.py:"
    "ConsoleSettingsUnsavedGuardMixin._perform_safe_cancel": CONFIRM_QUIT,
    "tldw_chatbook/Widgets/Console/console_model_popover.py:"
    "ConsoleModelPopover._perform_safe_cancel": CONFIRM_QUIT,
    "tldw_chatbook/Widgets/Console/console_video_capacity_modal.py:"
    "ConsoleVideoCapacityModal._perform_safe_cancel": CONFIRM_QUIT,
    "tldw_chatbook/UI/Screens/profile_interview_screen.py:"
    "ProfileInterviewScreen._perform_safe_cancel": CONFIRM_QUIT,
    # --- Exempt: the guard holds nothing quitting could lose.
    "tldw_chatbook/Third_Party/textual_fspicker/base_dialog.py:"
    "FileSystemPickerScreen._perform_safe_cancel": (
        "Escape first peels the picker's own path bar, search and recent "
        "list; the picker holds no work of its own (a caller whose result "
        "matters, like the generated video's, subclasses it with a hook)"
    ),
    "tldw_chatbook/Widgets/enhanced_file_picker.py:"
    "EnhancedFileDialog.action_smart_dismiss": (
        "Escape first peels the picker's path bar, search, recent and "
        "bookmark overlays; nothing is held that quitting would lose"
    ),
    "tldw_chatbook/UI/Screens/trajectory_screen.py:TrajectoryScreen.action_dismiss": (
        "a read-only viewer: Escape first leaves the search box or clears a "
        "selected range"
    ),
    "tldw_chatbook/Widgets/Console/console_review_notes_modal.py:"
    "ConsoleReviewNotesModal._perform_safe_cancel": (
        "Escape itself drops an in-progress note edit without asking, so "
        "quitting loses no more than Escape does; saved notes are stored"
    ),
    "tldw_chatbook/UI/Wizards/FirstRunSetupWizard.py:"
    "_SettlingGuardedConfirmationDialog.action_cancel_dialog_if_settled": (
        "only absorbs a reflexive second Escape before the dialog has "
        "painted; the wizard keeps every step already completed"
    ),
}


# --- The scan -----------------------------------------------------------------

_FunctionNode = ast.FunctionDef | ast.AsyncFunctionDef


@dataclass(eq=False)
class _Class:
    """One parsed class; compared by identity (its fields hold AST nodes)."""

    label: str
    module: str
    name: str
    node: ast.ClassDef
    base_names: list[str]
    imports: dict[str, str]
    methods: dict[str, _FunctionNode] = field(default_factory=dict)
    escape_actions: list[str] = field(default_factory=list)
    bases: list[_Class] = field(default_factory=list)


def _base_name(node: ast.expr) -> str:
    if isinstance(node, ast.Subscript):
        node = node.value
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Name):
        return node.id
    return ""


def _escape_actions(cls: ast.ClassDef) -> list[str]:
    """The action names bound to Escape in ``cls``'s own BINDINGS."""
    actions: list[str] = []
    for statement in cls.body:
        if not (
            isinstance(statement, (ast.Assign, ast.AnnAssign))
            and any(
                isinstance(target, ast.Name) and target.id == "BINDINGS"
                for target in (
                    statement.targets
                    if isinstance(statement, ast.Assign)
                    else [statement.target]
                )
            )
            and isinstance(statement.value, (ast.List, ast.Tuple))
        ):
            continue
        for entry in statement.value.elts:
            args: list[ast.expr] = []
            if isinstance(entry, ast.Tuple):
                args = list(entry.elts)
            elif isinstance(entry, ast.Call):
                args = list(entry.args)
                keywords = {kw.arg: kw.value for kw in entry.keywords}
                if not args and "key" in keywords:
                    args = [keywords["key"], keywords.get("action")]
                elif len(args) == 1 and "action" in keywords:
                    args.append(keywords["action"])
            if len(args) < 2 or not all(
                isinstance(arg, ast.Constant) and isinstance(arg.value, str)
                for arg in args[:2]
            ):
                continue
            keys = {key.strip() for key in args[0].value.split(",")}
            if "escape" in keys:
                actions.append(re.sub(r"\(.*", "", args[1].value))
    return actions


def _module_name(path: Path) -> str:
    return ".".join(path.relative_to(REPO_ROOT).with_suffix("").parts)


def _imports(tree: ast.Module, module: str) -> dict[str, str]:
    """Map each name imported at module level to its ``module.Name``."""
    package = module.rsplit(".", 1)[0]
    names: dict[str, str] = {}
    for node in tree.body:
        if not isinstance(node, ast.ImportFrom) or node.module is None:
            continue
        source = node.module
        if node.level:
            parts = package.split(".")
            source = ".".join(parts[: len(parts) - node.level + 1] + [node.module])
        for alias in node.names:
            names[alias.asname or alias.name] = f"{source}.{alias.name}"
    return names


def collect_classes(sources: dict[str, str]) -> dict[str, _Class]:
    """Parse ``{label: source}`` modules; return every class, bases resolved.

    Args:
        sources: Module source text keyed by a ``path/like/label.py``.

    Returns:
        Classes keyed ``module.Name``, each with its in-scan bases resolved.
    """
    classes: dict[str, _Class] = {}
    by_name: dict[str, list[_Class]] = {}
    for label, source in sources.items():
        module = ".".join(Path(label).with_suffix("").parts)
        tree = ast.parse(source)
        imports = _imports(tree, module)
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            info = _Class(
                label=label,
                module=module,
                name=node.name,
                node=node,
                base_names=[_base_name(base) for base in node.bases],
                imports=imports,
                methods={
                    item.name: item
                    for item in node.body
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                },
                escape_actions=_escape_actions(node),
            )
            classes[f"{module}.{node.name}"] = info
            by_name.setdefault(node.name, []).append(info)
    for info in classes.values():
        for base in info.base_names:
            local = classes.get(f"{info.module}.{base}")
            imported = classes.get(info.imports.get(base, ""))
            candidates = by_name.get(base, [])
            resolved = (
                local or imported or (candidates[0] if len(candidates) == 1 else None)
            )
            if resolved is not None and resolved is not info:
                info.bases.append(resolved)
    return classes


def _mro(info: _Class) -> list[_Class]:
    """A depth-first, left-to-right linearization of the in-scan ancestry."""
    order: list[_Class] = []

    def visit(node: _Class) -> None:
        if node in order:
            return
        order.append(node)
        for base in node.bases:
            visit(base)

    visit(info)
    return order


def _is_modal(info: _Class) -> bool:
    return any("ModalScreen" in node.base_names for node in _mro(info))


def _lookup(mro: list[_Class], name: str, after: _Class | None = None):
    """The first ``(class, method)`` named ``name`` in ``mro`` (after ``after``)."""
    start = mro.index(after) + 1 if after is not None and after in mro else 0
    for node in mro[start:]:
        if name in node.methods:
            return node, node.methods[name]
    return None


class _Analyzer:
    """Whether a close entry point always closes the modal, per class."""

    def __init__(self, mro: list[_Class]) -> None:
        self._mro = mro
        self._dismisses: dict[tuple[str, str], bool] = {}

    def _call_dismisses(self, call: ast.Call, owner: _Class) -> bool:
        func = call.func
        if not isinstance(func, ast.Attribute):
            return False
        if func.attr in _DISMISSALS:
            return True
        receiver = func.value
        if (
            isinstance(receiver, ast.Call)
            and isinstance(receiver.func, ast.Name)
            and receiver.func.id == "super"
        ):
            found = _lookup(self._mro, func.attr, after=owner)
            return found is not None and self.always_dismisses(*found)
        if isinstance(receiver, ast.Name) and receiver.id == "self":
            name = _SAFE_CANCEL if func.attr in _SAFE_REQUESTS else func.attr
            found = _lookup(self._mro, name)
            return found is not None and self.always_dismisses(*found)
        return False

    def _expr_dismisses(self, node: ast.AST | None, owner: _Class) -> bool:
        if node is None:
            return False
        return any(
            isinstance(child, ast.Call) and self._call_dismisses(child, owner)
            for child in ast.walk(node)
        )

    def _block(self, statements: list[ast.stmt], owner: _Class) -> bool | None:
        """True: every path closes. False: some path returns open. None: falls through."""
        for statement in statements:
            if isinstance(statement, ast.Return):
                return self._expr_dismisses(statement.value, owner)
            if isinstance(statement, ast.Raise):
                return False
            if isinstance(statement, ast.If):
                if self._expr_dismisses(statement.test, owner):
                    return True
                body = self._block(statement.body, owner)
                orelse = self._block(statement.orelse, owner)
                if body is False or orelse is False:
                    return False
                if body is True and orelse is True:
                    return True
                continue
            if isinstance(statement, (ast.Try, ast.With, ast.AsyncWith)):
                inner = self._block(statement.body, owner)
                if inner is not None:
                    return inner
                continue
            if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if self._expr_dismisses(statement, owner):
                return True
        return None

    def always_dismisses(self, owner: _Class, method: _FunctionNode) -> bool:
        key = (owner.module + "." + owner.name, method.name)
        if key not in self._dismisses:
            self._dismisses[key] = False  # a recursive path does not close
            self._dismisses[key] = self._block(method.body, owner) is True
        return self._dismisses[key]

    def pushes_a_screen(self, owner: _Class, method: _FunctionNode) -> bool:
        """Whether ``method``, or a method of the class it reaches, pushes a screen."""
        seen: set[str] = set()
        pending = [method]
        while pending:
            current = pending.pop()
            if current.name in seen:
                continue
            seen.add(current.name)
            for node in ast.walk(current):
                if isinstance(node, ast.Attribute) and node.attr in _PUSHES:
                    return True
                if (
                    isinstance(node, ast.Attribute)
                    and isinstance(node.value, ast.Name)
                    and node.value.id == "self"
                ):
                    found = _lookup(self._mro, node.attr)
                    if found is not None:
                        pending.append(found[1])
        return False


def _entry_points(mro: list[_Class]) -> list[str]:
    names: list[str] = []
    for node in mro:
        if node.escape_actions:
            for action in node.escape_actions:
                if action.startswith(("app.", "screen.")):
                    continue
                names.append(
                    _SAFE_CANCEL if action in _SAFE_REQUESTS else f"action_{action}"
                )
            break
    if any(node.name == "SafeModalDismissMixin" for node in mro):
        names.append(_SAFE_CANCEL)
    return list(dict.fromkeys(names))


@dataclass(frozen=True)
class Guard:
    """One guarded close: where it is written, and the modals that inherit it."""

    key: str
    modals: tuple[str, ...]
    modals_without_confirm_quit: tuple[str, ...]


def find_guards(classes: dict[str, _Class]) -> dict[str, Guard]:
    """Every guarded close entry point among ``classes``' modals.

    Args:
        classes: From ``collect_classes``.

    Returns:
        Guards keyed ``label:Class.method`` where the guarding method is
        written.
    """
    modals: dict[str, list[str]] = {}
    missing: dict[str, list[str]] = {}
    for info in classes.values():
        if not _is_modal(info) or info.name == "SafeModalDismissMixin":
            continue
        mro = _mro(info)
        analyzer = _Analyzer(mro)
        has_confirm_quit = _lookup(mro, CONFIRM_QUIT) is not None
        for entry in _entry_points(mro):
            found = _lookup(mro, entry)
            if found is None:
                continue
            owner, method = found
            if owner.name == "SafeModalDismissMixin":
                continue
            if analyzer.always_dismisses(owner, method) and not (
                analyzer.pushes_a_screen(owner, method)
            ):
                continue
            key = f"{owner.label}:{owner.name}.{method.name}"
            modal = f"{info.label}:{info.name}"
            modals.setdefault(key, []).append(modal)
            if not has_confirm_quit:
                missing.setdefault(key, []).append(modal)
    return {
        key: Guard(
            key,
            tuple(sorted(set(names))),
            tuple(sorted(set(missing.get(key, [])))),
        )
        for key, names in modals.items()
    }


def _package_sources() -> dict[str, str]:
    return {
        str(path.relative_to(REPO_ROOT)): path.read_text(encoding="utf-8")
        for path in sorted(PACKAGE.rglob("*.py"))
    }


_CLASS_HEADER = re.compile(
    r"^[ \t]*class[ \t]+(\w+)[ \t]*(?:\(([^)]*)\))?[ \t]*:", re.M
)


def modal_family_sources(sources: dict[str, str]) -> dict[str, str]:
    """Only the modules that define a modal or one of its ancestors.

    Parsing every module of the package costs tens of seconds; a regex over
    class headers finds, by name, every ModalScreen subclass and every class
    it inherits from, and only their modules are parsed.

    Args:
        sources: Module source text keyed by label.

    Returns:
        The subset of ``sources`` the census needs.
    """
    bases_by_name: dict[str, set[str]] = {}
    files_by_name: dict[str, set[str]] = {}
    for label, source in sources.items():
        for name, bases in _CLASS_HEADER.findall(source):
            # A subscript may span lines and hold commas: keep what precedes
            # its "[" (stray pieces from inside one are harmless names).
            names = {
                base.split("[")[0].strip().split(".")[-1]
                for base in bases.split(",")
                if base.strip()
            }
            bases_by_name.setdefault(name, set()).update(names)
            files_by_name.setdefault(name, set()).add(label)
    modal = {"ModalScreen"}
    grew = True
    while grew:
        found = {name for name, bases in bases_by_name.items() if bases & modal} - modal
        grew = bool(found)
        modal |= found
    wanted = set(modal)
    pending = list(modal)
    while pending:
        for base in bases_by_name.get(pending.pop(), ()):
            if base not in wanted:
                wanted.add(base)
                pending.append(base)
    labels = set().union(*(files_by_name.get(name, set()) for name in wanted))
    return {label: sources[label] for label in sorted(labels)}


@lru_cache(maxsize=1)
def _package_guards() -> dict[str, Guard]:
    """The live census, scanned once per session."""
    return find_guards(collect_classes(modal_family_sources(_package_sources())))


def _format(rows: list[str]) -> str:
    return "\n  ".join(rows)


def test_every_guarded_modal_close_has_a_ctrl_q_decision() -> None:
    guards = _package_guards()
    undecided = sorted(set(guards) - set(GUARDED_CLOSE_DECISIONS))
    assert not undecided, (
        "A modal refuses or guards its own close, and nothing decides what "
        "Ctrl+Q does there. Ctrl+Q is a priority binding (TASK-33622.10): "
        "without a confirm_quit it quits straight past the guard. Add a "
        "confirm_quit -- refuse_quit_while_working (Widgets/quit_while_working"
        ".py) while an operation runs, confirm_quit_discarding_edits before a "
        "discard -- and a CONFIRM_QUIT row in GUARDED_CLOSE_DECISIONS, or an "
        "exemption row saying why quitting loses nothing:\n  "
        + _format(
            f"{key} (modals: {', '.join(guards[key].modals)})" for key in undecided
        )
    )


def test_every_census_row_still_names_a_guarded_close() -> None:
    """A row whose guard is gone (or moved) is stale and must be updated."""
    guards = _package_guards()
    stale = sorted(set(GUARDED_CLOSE_DECISIONS) - set(guards))
    assert not stale, (
        "These GUARDED_CLOSE_DECISIONS rows no longer match a guarded close; "
        "update or remove them:\n  " + _format(stale)
    )


def test_every_confirm_quit_decision_is_backed_by_a_hook() -> None:
    """A CONFIRM_QUIT row holds for every concrete modal inheriting the guard."""
    guards = _package_guards()
    missing = sorted(
        f"{key}: {', '.join(guards[key].modals_without_confirm_quit)}"
        for key, decision in GUARDED_CLOSE_DECISIONS.items()
        if decision == CONFIRM_QUIT
        and key in guards
        and guards[key].modals_without_confirm_quit
    )
    assert not missing, (
        "These modals inherit a guard recorded as CONFIRM_QUIT but define no "
        "confirm_quit:\n  " + _format(missing)
    )


def test_every_exemption_says_why() -> None:
    vague = sorted(
        key
        for key, decision in GUARDED_CLOSE_DECISIONS.items()
        if decision != CONFIRM_QUIT and len(decision.split()) < 6
    )
    assert not vague, f"an exemption must say why quitting loses nothing: {vague}"


# --- Negative controls ----------------------------------------------------------

_FIXTURE = {
    "pkg/mixin.py": """
class SafeModalDismissMixin:
    async def _perform_safe_cancel(self, *, source):
        self.dismiss_safe_once(None)

    async def request_safe_cancel(self, *, source):
        if self._pending:
            return
        await self._perform_safe_cancel(source=source)
""",
    "pkg/modals.py": """
from pkg.mixin import SafeModalDismissMixin


class RefusesWhileBusy(SafeModalDismissMixin, ModalScreen[None]):
    async def _perform_safe_cancel(self, *, source):
        if self._busy:
            return
        self.dismiss_safe_once(None)


class ImplicitRefusal(SafeModalDismissMixin, ModalScreen[None]):
    async def _perform_safe_cancel(self, *, source):
        if not self._busy:
            self.dismiss_safe_once(None)


class AsksFirst(SafeModalDismissMixin, ModalScreen[None]):
    async def _perform_safe_cancel(self, *, source):
        self.call_after_refresh(self._ask)

    def _ask(self):
        self.app.push_screen(Prompt(), callback=self._answered)


class AlwaysCloses(SafeModalDismissMixin, ModalScreen[None]):
    async def _perform_safe_cancel(self, *, source):
        await self.run_effect()
        if self._created:
            await super()._perform_safe_cancel(source=source)
            return
        self.dismiss_safe_once("partial")


class PlainEscape(ModalScreen[None]):
    BINDINGS = [("escape", "cancel", "Cancel")]

    def action_cancel(self):
        if self._posted:
            self.show("still working")
            return
        self.dismiss(None)


class PlainEscapeThatCloses(ModalScreen[None]):
    BINDINGS = [Binding("escape", "close", "Close", show=False)]

    def action_close(self):
        self.dismiss(None)


class Hooked(RefusesWhileBusy):
    async def confirm_quit(self):
        return not self._busy


class Inherits(RefusesWhileBusy):
    pass


class NotAModal:
    async def _perform_safe_cancel(self, *, source):
        if self._busy:
            return
""",
    "pkg/elsewhere.py": """
from pkg.modals import RefusesWhileBusy


class MultiLineHeader(
    RefusesWhileBusy,
    ModalScreen[
        "Result | None"
    ],
):
    pass


class Unrelated:
    pass
""",
    "pkg/unrelated.py": """
class Helper(Base):
    def _perform_safe_cancel(self, *, source):
        return
""",
}


def test_negative_control_the_census_finds_each_kind_of_guard() -> None:
    sources = modal_family_sources(_FIXTURE)
    # The cheap header pass keeps a paren-less mixin and a multi-line
    # header, and drops the module with no modal in its family.
    assert sorted(sources) == ["pkg/elsewhere.py", "pkg/mixin.py", "pkg/modals.py"]
    guards = find_guards(collect_classes(sources))
    assert sorted(guards) == [
        "pkg/modals.py:AsksFirst._perform_safe_cancel",
        "pkg/modals.py:ImplicitRefusal._perform_safe_cancel",
        "pkg/modals.py:PlainEscape.action_cancel",
        "pkg/modals.py:RefusesWhileBusy._perform_safe_cancel",
    ]
    busy = guards["pkg/modals.py:RefusesWhileBusy._perform_safe_cancel"]
    # One row for the guard, every modal that inherits it, and only the
    # subclass with no hook of its own is missing one.
    assert busy.modals == (
        "pkg/elsewhere.py:MultiLineHeader",
        "pkg/modals.py:Hooked",
        "pkg/modals.py:Inherits",
        "pkg/modals.py:RefusesWhileBusy",
    )
    assert busy.modals_without_confirm_quit == (
        "pkg/elsewhere.py:MultiLineHeader",
        "pkg/modals.py:Inherits",
        "pkg/modals.py:RefusesWhileBusy",
    )


def test_negative_control_the_live_census_reaches_known_guards() -> None:
    """A scan that found nothing would pass vacuously."""
    guards = _package_guards()
    for key in (
        "tldw_chatbook/Widgets/Console/console_fork_chat_modal.py:"
        "ConsoleForkChatModal._perform_safe_cancel",
        "tldw_chatbook/UI/Screens/scheduling/forms/reminder_form.py:"
        "ReminderForm.action_dismiss",
        "tldw_chatbook/Widgets/Console/console_settings_unsaved.py:"
        "ConsoleSettingsUnsavedGuardMixin._perform_safe_cancel",
    ):
        assert key in guards, key
    settings = guards[
        "tldw_chatbook/Widgets/Console/console_settings_unsaved.py:"
        "ConsoleSettingsUnsavedGuardMixin._perform_safe_cancel"
    ]
    assert (
        "tldw_chatbook/Widgets/Console/console_settings_modal.py:ConsoleSettingsModal"
        in settings.modals
    )
