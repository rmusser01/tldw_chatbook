"""Every prompt the quit flow owns is awaited through ``await_quit_prompt``.

TASK-33622.10. Ctrl+Q is a priority binding, so the quit flow pushes its
prompts over any open modal, and in Textual 8.2.8 ``Screen.dismiss()`` pops
the TOP screen -- so a covered modal closing itself pops the quit prompt
unanswered. A bare ``push_screen_wait`` (or ``push_screen(...,
wait_for_dismiss=True)``) on that prompt never returns: the quit worker hangs,
``_quit_in_progress`` stays set and Ctrl+Q is dead for the session.
``await_quit_prompt`` (``Widgets/confirmation_dialog.py``) is the one choke
point that cannot hang, and ADR-031's task-33622.10 refinement makes it the
rule.

Nothing else enforces that rule: a new screen adding ``async def
confirm_quit(self): return await self.app.push_screen_wait(...)`` would bring
the permanent hang straight back with every behavioural test still green. So
this scans, statically:

* every zero-argument ``confirm_quit`` / ``prepare_for_quit`` /
  ``prepare_quit`` method in ``tldw_chatbook/`` -- the quit walk calls those
  with no arguments, so a two-argument ``confirm_quit(self, controller)`` is
  not reachable from it -- plus every method of the same class it reaches
  through ``self.<name>(...)``;
* every method of ``app_lifecycle.py`` whose name mentions ``quit``, with the
  same ``self.`` closure;
* every module-level function whose name mentions ``quit`` (except
  ``await_quit_prompt`` itself) in any module that defines or names a walk
  hook -- ``confirmation_dialog.py``'s prompt helpers, and the delegates
  TASK-33622.15 added beside the modals that call them
  (``refuse_quit_while_working``, ``confirm_quit_discarding_generated_video``
  ...), which a ``self.`` closure cannot follow;
* every module-level function of ``Persona_Modules/roleplay_draft_guard.py``
  (TASK-33622.14). ``PersonasScreen.confirm_quit`` delegates there, so the
  scan cannot follow it through ``self.``; the module's flow is shared with
  app navigation and is handed its ``ask``, so no function in it may wait on
  a pushed screen at all.

and fails if any of them waits on a pushed screen other than through the
choke point. Calls into other classes are not followed; the rule is pinned
where a quit-flow prompt is written, which is these methods.
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE = REPO_ROOT / "tldw_chatbook"
APP_LIFECYCLE = PACKAGE / "app_lifecycle.py"
#: Roleplay's quit hook delegates here; every function in it is scanned.
ROLEPLAY_DRAFT_GUARD = PACKAGE / "UI" / "Persona_Modules" / "roleplay_draft_guard.py"

#: Hooks the quit walk (and the workflow authoring owner) call with no args.
WALK_HOOKS = frozenset({"confirm_quit", "prepare_for_quit", "prepare_quit"})
#: The choke point itself is the one place allowed to wait on a pushed screen.
CHOKE_POINT = "await_quit_prompt"

_FunctionNode = ast.FunctionDef | ast.AsyncFunctionDef


def _waits_on_a_pushed_screen(function: _FunctionNode) -> list[int]:
    """Line numbers where ``function`` waits on a screen it pushed."""
    lines: list[int] = []
    for node in ast.walk(function):
        if isinstance(node, ast.Attribute) and node.attr == "push_screen_wait":
            lines.append(node.lineno)
        elif isinstance(node, ast.Call) and any(
            keyword.arg == "wait_for_dismiss"
            and not (
                isinstance(keyword.value, ast.Constant) and keyword.value.value is False
            )
            for keyword in node.keywords
        ):
            lines.append(node.lineno)
    return lines


def _self_calls(function: _FunctionNode) -> set[str]:
    return {
        node.func.attr
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "self"
    }


def _takes_no_arguments(function: _FunctionNode) -> bool:
    args = function.args
    return (
        len(args.posonlyargs) + len(args.args) == 1
        and not args.kwonlyargs
        and args.vararg is None
        and args.kwarg is None
    )


def _closure(methods: dict[str, _FunctionNode], roots: list[str]) -> list[str]:
    seen: list[str] = []
    pending = list(roots)
    while pending:
        name = pending.pop()
        if name in seen or name not in methods:
            continue
        seen.append(name)
        pending.extend(sorted(_self_calls(methods[name])))
    return seen


def scan_source(
    source: str, label: str, *, quit_named_roots: bool
) -> tuple[list[str], list[str]]:
    """Scan one module.

    Args:
        source: The module's source text.
        label: How offences name the module.
        quit_named_roots: Also treat every class method whose name mentions
            ``quit`` as a root (the app's own quit flow module).

    Returns:
        ``(roots, offences)``: every scanned root as ``label:Class.method``,
        and every ``label:Class.method:line`` that waits on a pushed screen.
    """
    tree = ast.parse(source)
    roots_seen: list[str] = []
    offences: list[str] = []
    for cls in (node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)):
        methods = {
            node.name: node
            for node in cls.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        roots = [
            name
            for name, node in methods.items()
            if (name in WALK_HOOKS and _takes_no_arguments(node))
            or (quit_named_roots and "quit" in name)
        ]
        roots_seen.extend(f"{label}:{cls.name}.{name}" for name in roots)
        for name in _closure(methods, roots):
            offences.extend(
                f"{label}:{cls.name}.{name}:{line}"
                for line in _waits_on_a_pushed_screen(methods[name])
            )
    return roots_seen, offences


def scan_module_functions(
    source: str, label: str, *, quit_named_only: bool = True
) -> tuple[list[str], list[str]]:
    """Scan a module's module-level functions.

    Args:
        source: The module's source text.
        label: How offences name the module.
        quit_named_only: Scan only functions whose name mentions ``quit``
            (the choke point's own module); False scans every function.

    Returns:
        ``(roots, offences)`` as ``scan_source`` returns them.
    """
    tree = ast.parse(source)
    roots: list[str] = []
    offences: list[str] = []
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name == CHOKE_POINT:
            continue
        if quit_named_only and "quit" not in node.name:
            continue
        roots.append(f"{label}:{node.name}")
        offences.extend(
            f"{label}:{node.name}:{line}" for line in _waits_on_a_pushed_screen(node)
        )
    return roots, offences


def _scan_tree() -> tuple[list[str], list[str]]:
    roots: list[str] = []
    offences: list[str] = []
    for path in sorted(PACKAGE.rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        if path != APP_LIFECYCLE and not any(hook in source for hook in WALK_HOOKS):
            continue
        label = str(path.relative_to(REPO_ROOT))
        found_roots, found_offences = scan_source(
            source, label, quit_named_roots=path == APP_LIFECYCLE
        )
        roots.extend(found_roots)
        offences.extend(found_offences)
        if path == ROLEPLAY_DRAFT_GUARD:
            continue  # scanned whole below
        # TASK-33622.15: a hook's module-level quit helpers (the prompt
        # helpers in confirmation_dialog.py among them).
        found_roots, found_offences = scan_module_functions(source, label)
        roots.extend(found_roots)
        offences.extend(found_offences)
    # Missing on a tree without TASK-33622.14: the reach test then names it.
    if ROLEPLAY_DRAFT_GUARD.exists():
        found_roots, found_offences = scan_module_functions(
            ROLEPLAY_DRAFT_GUARD.read_text(encoding="utf-8"),
            str(ROLEPLAY_DRAFT_GUARD.relative_to(REPO_ROOT)),
            quit_named_only=False,
        )
        roots.extend(found_roots)
        offences.extend(found_offences)
    return roots, offences


def test_quit_flow_prompts_are_awaited_only_through_the_choke_point() -> None:
    roots, offences = _scan_tree()
    assert offences == [], (
        "A quit-flow prompt waits on a pushed screen directly. A covered "
        "modal's dismiss() pops the TOP screen, so that wait can never "
        "return and Ctrl+Q dies for the session (TASK-33622.10, ADR-031). "
        "Await it through await_quit_prompt (Widgets/confirmation_dialog.py) "
        "instead:\n  " + "\n  ".join(offences)
    )


def test_the_scan_reaches_the_quit_flow_it_guards() -> None:
    """A scan that finds no roots would pass vacuously."""
    roots, _offences = _scan_tree()
    expected = {
        "tldw_chatbook/UI/Screens/settings_screen.py:SettingsScreen.confirm_quit",
        "tldw_chatbook/UI/Screens/scheduling/forms/reminder_form.py:"
        "ReminderForm.confirm_quit",
        "tldw_chatbook/UI/Screens/chunking_lab_screen.py:"
        "ChunkingLabScreen.prepare_for_quit",
        "tldw_chatbook/Workflows/authoring.py:WorkflowAuthoring.prepare_quit",
        "tldw_chatbook/app_lifecycle.py:LifecycleMixin._confirm_and_quit",
        "tldw_chatbook/app_lifecycle.py:LifecycleMixin._await_quit_prompt",
        "tldw_chatbook/Widgets/confirmation_dialog.py:confirm_quit_discarding_edits",
        # TASK-33622.14: Roleplay's hook, and the shared flow it delegates to.
        "tldw_chatbook/UI/Screens/personas_screen.py:PersonasScreen.confirm_quit",
        "tldw_chatbook/UI/Persona_Modules/roleplay_draft_guard.py:"
        "confirm_roleplay_quit",
        "tldw_chatbook/UI/Persona_Modules/roleplay_draft_guard.py:"
        "confirm_roleplay_drafts",
        # TASK-33622.15: in-flight refusals, the video save screens, and the
        # module-level delegates their hooks call.
        "tldw_chatbook/Widgets/Console/console_fork_chat_modal.py:"
        "ConsoleForkChatModal.confirm_quit",
        "tldw_chatbook/Widgets/quit_while_working.py:refuse_quit_while_working",
        "tldw_chatbook/Widgets/Console/console_video_save_screens.py:"
        "GeneratedVideoFileSave.confirm_quit",
        "tldw_chatbook/Widgets/Console/console_video_capacity_modal.py:"
        "confirm_quit_discarding_generated_video",
        "tldw_chatbook/Widgets/Console/console_capture_policy_dialog.py:"
        "_quit_unless_applying",
    }
    missing = expected - set(roots)
    assert not missing, f"the scan no longer reaches: {sorted(missing)}"
    # session.py's confirm_quit(self, controller) is not called by the walk.
    assert not any(
        root.endswith(
            "Console_Modules/session.py:ConsoleSessionController.confirm_quit"
        )
        for root in roots
    )


_OFFENDING = """
class Direct:
    async def confirm_quit(self):
        return await self.app.push_screen_wait(Prompt())

class Transitive:
    async def confirm_quit(self):
        return await self._ask()

    async def _ask(self):
        return await self.app.push_screen(Prompt(), wait_for_dismiss=True)

class NotOnTheWalk:
    async def confirm_quit(self, controller):
        return await self.app.push_screen_wait(Prompt())

class Fine:
    async def confirm_quit(self):
        return await confirm_quit_discarding_edits(self, "edits")
"""


def test_negative_control_the_scan_catches_a_direct_and_a_transitive_wait() -> None:
    roots, offences = scan_source(_OFFENDING, "fixture", quit_named_roots=False)
    assert offences == [
        "fixture:Direct.confirm_quit:4",
        "fixture:Transitive._ask:11",
    ]
    assert "fixture:NotOnTheWalk.confirm_quit" not in roots
    assert "fixture:Fine.confirm_quit" in roots


def test_negative_control_module_functions_except_the_choke_point() -> None:
    source = """
async def await_quit_prompt(app, prompt):
    return app.push_screen(prompt, wait_for_dismiss=True)

async def confirm_quit_with_a_bare_wait(app, prompt):
    return await app.push_screen_wait(prompt)
"""
    roots, offences = scan_module_functions(source, "fixture")
    assert roots == ["fixture:confirm_quit_with_a_bare_wait"]
    assert offences == ["fixture:confirm_quit_with_a_bare_wait:6"]


def test_negative_control_every_function_of_a_delegate_module() -> None:
    """The Roleplay flow's shared helper is scanned whatever its name."""
    source = """
async def confirm_drafts(screen, ask):
    return await ask(Prompt())

async def _save_then_ask(screen):
    return await screen.app.push_screen_wait(Prompt())
"""
    roots, offences = scan_module_functions(source, "fixture", quit_named_only=False)
    assert roots == ["fixture:confirm_drafts", "fixture:_save_then_ask"]
    assert offences == ["fixture:_save_then_ask:6"]
