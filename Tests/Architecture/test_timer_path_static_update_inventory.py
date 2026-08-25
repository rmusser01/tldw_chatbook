"""Census: no timer-path ``.update(`` may default to ``layout=True`` unnoticed.

``Static.update`` is ``update(content="", *, layout: bool = True)`` and ends in
``self.refresh(layout=layout)``. **The default lays out.** A repaint on a clock
therefore arms a whole ``Screen._refresh_layout`` / ``Compositor.reflow`` on
every tick unless the caller opts out. Three instances have been found the hard
way:

* TASK-21692 -- the Console composer cursor blink: 396 ``Widget.arrange`` calls
  per 6 ticks, ~1.9 whole-screen reflows/second on an *idle* focused composer,
  in a method whose own docstring said it must not do that.
* TASK-21134 item 7 -- the media-viewer match-nav: 10 layout messages -> 0.
* TASK-21595 (this census) -- ``SplashScreen._update_animation`` at 10-100 fps
  during startup, and ``PersonaBuddyWidget._paint_frame`` at the pet's frame
  rate.

Three instances is a pattern, so this module rebuilds the census on every run
instead of trusting a one-off sweep. It walks the package AST, collects every
*repeating* clock root, follows the intra-package call graph from each, and
requires that every ``.update(`` call it can reach is **classified**: either the
call passes ``layout=`` explicitly, or it carries an entry in
``CLASSIFIED_SITES`` saying why it does not need to.

Adding a new timer-path repaint therefore fails this test until its author says
which of the two it is. That is the whole point -- the cost is invisible to
every other test, because nothing else counts layout operations.

Runtime evidence for the two sites this census fixed (measured layout-pass
counts against a measured idle floor, plus geometry-equivalence A/Bs) lives in
``Tests/UI/test_timer_path_layout_cost.py``.
"""

from __future__ import annotations

import ast
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Iterator

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "tldw_chatbook"
REPO_ROOT = PACKAGE_ROOT.parent

#: How deep to follow the call graph out from a clock root. Six hops covers the
#: real chains (``_poll_transcript`` -> ... -> a leaf widget's ``sync_*``)
#: without the graph degenerating into "everything".
MAX_CALL_DEPTH = 6

#: Textual's own repeating-clock constructors. ``set_timer`` is a one-shot, so
#: it only counts as a clock when the callback re-arms *itself* -- an interval
#: spelled as a chain of one-shots, which is how the Persona Buddy frame timer
#: and several Console ticks are written.
REPEATING_CLOCK = "set_interval"
ONE_SHOT_CLOCK = "set_timer"


# ---------------------------------------------------------------------------
# classification
# ---------------------------------------------------------------------------
#
# Key: (path relative to the repo root, enclosing `Class.method`, receiver
#       expression as written).
# Value: why this call does not need `layout=False`.
#
# Four kinds of reason appear, and the distinction matters:
#
#   NOT-A-WIDGET  -- `dict.update` / `set.update`, which shares its name with
#                    `Static.update` and is indistinguishable in the AST.
#   NEEDS-LAYOUT  -- the rendered size genuinely depends on the content (a
#                    `height: auto` box, or a `width: auto` strip), so skipping
#                    the layout pass would leave stale geometry on screen. This
#                    is a real behaviour change, not an optimisation.
#   NOT-PER-TICK  -- the tick reaches the call only through an equality gate or
#                    a branch a tick cannot take, so it is not a per-tick cost.
#   UNREACHABLE   -- the module has no importer anywhere in the repo.
#
CLASSIFIED_SITES: dict[tuple[str, str, str], str] = {
    # -- tldw_chatbook/UI/Console_Modules/prompt_queue.py
    (
        "tldw_chatbook/UI/Console_Modules/prompt_queue.py",
        "ConsolePromptQueueRegion.sync_presentation",
        "preview",
    ): "NEEDS-LAYOUT: queue preview grows with the queued prompt.",
    (
        "tldw_chatbook/UI/Console_Modules/prompt_queue.py",
        "ConsolePromptQueueRegion.sync_presentation",
        "summary",
    ): "NEEDS-LAYOUT: queue summary is height:auto.",
    # -- tldw_chatbook/UI/Console_Modules/workspace.py
    (
        "tldw_chatbook/UI/Console_Modules/workspace.py",
        "ConsoleWorkspaceController.apply_workspace_membership_snapshot",
        "owner_by_conversation",
    ): "NOT-A-WIDGET: local dict merge, not a Static.",
    (
        "tldw_chatbook/UI/Console_Modules/workspace.py",
        "ConsoleWorkspaceController.apply_workspace_membership_snapshot",
        "self._canonical_owner_observations",
    ): "NOT-A-WIDGET: dict of owner observations, not a Static.",
    # -- tldw_chatbook/UI/Research_Window.py
    (
        "tldw_chatbook/UI/Research_Window.py",
        "ResearchWindow._update_detail",
        "self.query_one('#research-run-detail', Static)",
    ): (
        "NEEDS-LAYOUT: run detail is a height:auto block that grows with "
        "events."
    ),
    # -- tldw_chatbook/UI/Screens/chat_screen.py
    (
        "tldw_chatbook/UI/Screens/chat_screen.py",
        "ChatScreen._prune_console_rail_preferences",
        "live",
    ): "NOT-A-WIDGET: set of live rail preference keys, not a Static.",
    (
        "tldw_chatbook/UI/Screens/chat_screen.py",
        "ChatScreen._sync_console_agent_section",
        "fleet_summary",
    ): (
        "NEEDS-LAYOUT: height:auto summary whose line count tracks the "
        "fleet size."
    ),
    (
        "tldw_chatbook/UI/Screens/chat_screen.py",
        "ChatScreen._sync_console_agent_section",
        "self.query_one('#console-agent-section-status', Static)",
    ): "NEEDS-LAYOUT: height:auto agent status line.",
    (
        "tldw_chatbook/UI/Screens/chat_screen.py",
        "ChatScreen._sync_console_agent_section",
        "self.query_one('#console-agent-section-steps', Static)",
    ): "NEEDS-LAYOUT: height:auto agent steps line.",
    (
        "tldw_chatbook/UI/Screens/chat_screen.py",
        "ChatScreen._sync_console_mode_bar",
        "mode_bar",
    ): (
        "NEEDS-LAYOUT: the mode bar's chip row wraps, so its height is "
        "content-driven."
    ),
    (
        "tldw_chatbook/UI/Screens/chat_screen.py",
        "ChatScreen._sync_console_rail_system_line",
        "system_line",
    ): (
        "NOT-PER-TICK: equality-gated on _console_rail_system_line_last "
        "(TASK-251), so the 0.2 s tick does not repaint it."
    ),
    (
        "tldw_chatbook/UI/Screens/chat_screen.py",
        "ChatScreen._sync_console_settings_summary",
        "recovery",
    ): (
        "NEEDS-LAYOUT: the readiness row is display-toggled on the same "
        "path."
    ),
    (
        "tldw_chatbook/UI/Screens/chat_screen.py",
        "ChatScreen._sync_console_settings_summary",
        "self.query_one('#console-model-section-max-tokens .console-model-section-value', Static)",
    ): (
        "NEEDS-LAYOUT: as above -- shares the wrapped .console-model- "
        "section-value rule."
    ),
    (
        "tldw_chatbook/UI/Screens/chat_screen.py",
        "ChatScreen._sync_console_settings_summary",
        "self.query_one('#console-model-section-model .console-model-section-value', Static)",
    ): "NEEDS-LAYOUT: as above -- wrapped value, auto height capped at 3.",
    (
        "tldw_chatbook/UI/Screens/chat_screen.py",
        "ChatScreen._sync_console_settings_summary",
        "self.query_one('#console-model-section-provider .console-model-section-value', Static)",
    ): (
        "NEEDS-LAYOUT: .console-model-section-value is text-wrap:wrap with "
        "max-height:3, so the painted row count changes with the model "
        "name."
    ),
    (
        "tldw_chatbook/UI/Screens/chat_screen.py",
        "ChatScreen._sync_console_settings_summary",
        "self.query_one('#console-model-section-temperature .console-model-section-value', Static)",
    ): (
        "NEEDS-LAYOUT: as above -- shares the wrapped .console-model- "
        "section-value rule."
    ),
    (
        "tldw_chatbook/UI/Screens/chat_screen.py",
        "ChatScreen._sync_native_console_transcript",
        "self._console_image_preparing",
    ): "NOT-A-WIDGET: set of in-flight image ids, not a Static.",
    # -- tldw_chatbook/UI/Screens/trajectory_screen.py
    (
        "tldw_chatbook/UI/Screens/trajectory_screen.py",
        "TrajectoryScreen._refresh_state",
        "self.query_one('#trajectory-state', Static)",
    ): (
        "NEEDS-LAYOUT: #trajectory-state is height:auto with max-height:4, "
        "so the state copy really does change its row count."
    ),
    # -- tldw_chatbook/UI/Screens/video_player_screen.py
    (
        "tldw_chatbook/UI/Screens/video_player_screen.py",
        "VideoPlayerScreen._refresh_status",
        "self.query_one('#video-player-status', Static)",
    ): (
        "NOT-PER-TICK: #video-player-status is height:1, but the status "
        "timer only runs while a video is actually playing and the line "
        "changes on every tick anyway (position advances), so the repaint "
        "is real work rather than idle burn. Left alone to avoid a "
        "behaviour change for a gain that is one coalesced layout pass per "
        "second."
    ),
    # -- tldw_chatbook/Widgets/Console/console_composer_bar.py
    (
        "tldw_chatbook/Widgets/Console/console_composer_bar.py",
        "ConsoleComposerBar._refresh_visible_draft",
        "self.query_one('#console-command-visible-text', Static)",
    ): (
        "NEEDS-LAYOUT: _refresh_visible_draft is the draft-mutation path "
        "and must recompute the composer height. TASK-21692 split the blink "
        "tick off into _render_visible_draft_only, which passes "
        "layout=False."
    ),
    (
        "tldw_chatbook/Widgets/Console/console_composer_bar.py",
        "ConsoleComposerBar._sync_collapsed_presentation",
        "status",
    ): "NEEDS-LAYOUT: collapsed-composer status line is height:auto.",
    (
        "tldw_chatbook/Widgets/Console/console_composer_bar.py",
        "ConsoleComposerBar._sync_send_disabled_reason",
        "strip",
    ): (
        "NEEDS-LAYOUT: the Send disabled-reason strip is width:auto and the "
        "same call toggles display/width/height inline -- the layout pass "
        "is the point, not an accident."
    ),
    (
        "tldw_chatbook/Widgets/Console/console_composer_bar.py",
        "ConsoleComposerBar.set_pending_attachment_label",
        "indicator",
    ): (
        "NEEDS-LAYOUT: attachment indicator is width:auto and display- "
        "toggled."
    ),
    (
        "tldw_chatbook/Widgets/Console/console_composer_bar.py",
        "ConsoleComposerBar.set_voice_status",
        "chip",
    ): (
        "NEEDS-LAYOUT: the voice chip's width is computed and assigned "
        "inline on the same path (styles.width / display), so its box "
        "changes with the content it is being handed."
    ),
    # -- tldw_chatbook/Widgets/Console/console_prompt_queue_modal.py
    (
        "tldw_chatbook/Widgets/Console/console_prompt_queue_modal.py",
        "ConsolePromptQueueModal._apply_snapshot",
        "state",
    ): "NEEDS-LAYOUT: modal state block is height:auto.",
    # -- tldw_chatbook/Widgets/Console/console_session_surface.py
    (
        "tldw_chatbook/Widgets/Console/console_session_surface.py",
        "ConsoleSessionSurface.set_session_title",
        "header",
    ): (
        "NEEDS-LAYOUT: session title width is auto and tracks the title "
        "text."
    ),
    (
        "tldw_chatbook/Widgets/Console/console_session_surface.py",
        "ConsoleSessionSurface.show_fleet_coachmark",
        "content",
    ): (
        "NEEDS-LAYOUT: the fleet coachmark is a display-toggled auto-sized "
        "callout."
    ),
    (
        "tldw_chatbook/Widgets/Console/console_session_surface.py",
        "ConsoleSessionSurface.sync_inline_guidance",
        "title",
    ): "NEEDS-LAYOUT: inline guidance is height:auto and display-toggled.",
    # -- tldw_chatbook/Widgets/Console/console_status_chips.py
    (
        "tldw_chatbook/Widgets/Console/console_status_chips.py",
        "ConsoleStatusChips.sync_cost_state",
        "chip",
    ): (
        "NEEDS-LAYOUT: chips are width:auto -- the label length IS the "
        "width."
    ),
    (
        "tldw_chatbook/Widgets/Console/console_status_chips.py",
        "ConsoleStatusChips.sync_run_chip",
        "chip",
    ): (
        "NEEDS-LAYOUT: chips are width:auto -- the label length IS the "
        "width."
    ),
    # -- tldw_chatbook/Widgets/Console/console_transcript.py
    (
        "tldw_chatbook/Widgets/Console/console_transcript.py",
        "ConsoleTranscript.sync_jump_indicator",
        "pill",
    ): "NEEDS-LAYOUT: the jump pill is width:auto and display-toggled.",
    # -- tldw_chatbook/Widgets/Console/console_video_preview.py
    (
        "tldw_chatbook/Widgets/Console/console_video_preview.py",
        "ConsoleVideoPreview._update_frame",
        "frame",
    ): (
        "NEEDS-LAYOUT: .console-video-preview-frame is height:auto and the "
        "Pixels renderable's row count varies with the scaled image (and "
        "with the poster-text fallback), so the box really does resize per "
        "frame."
    ),
    # -- tldw_chatbook/Widgets/Console/console_workspace_context.py
    (
        "tldw_chatbook/Widgets/Console/console_workspace_context.py",
        "ConsoleWorkspaceContextTray._update_workspace_tree_selection_context",
        "context",
    ): "NEEDS-LAYOUT: the context tray line count tracks the selection.",
    # -- tldw_chatbook/Widgets/Library/library_file_notes_workspace.py
    (
        "tldw_chatbook/Widgets/Library/library_file_notes_workspace.py",
        "LibraryFileNotesWorkspace._apply_opened_document",
        "self.query_one('#file-notes-breadcrumb', Static)",
    ): (
        "NEEDS-LAYOUT: the breadcrumb wraps, so its row count tracks the "
        "path."
    ),
    (
        "tldw_chatbook/Widgets/Library/library_file_notes_workspace.py",
        "LibraryFileNotesWorkspace._dismiss_reload_confirmation",
        "self.query_one('#file-notes-reload-confirm-copy', Static)",
    ): (
        "NEEDS-LAYOUT: the reload-confirm callout is display-toggled and "
        "auto-sized."
    ),
    (
        "tldw_chatbook/Widgets/Library/library_file_notes_workspace.py",
        "LibraryFileNotesWorkspace._render_session_git_label",
        "authority",
    ): (
        "NEEDS-LAYOUT: the authority line is width:auto and display- "
        "toggled."
    ),
    (
        "tldw_chatbook/Widgets/Library/library_file_notes_workspace.py",
        "LibraryFileNotesWorkspace._set_action_status",
        "self.query_one('#file-notes-action-status', Static)",
    ): "NEEDS-LAYOUT: height:auto action status line.",
    (
        "tldw_chatbook/Widgets/Library/library_file_notes_workspace.py",
        "LibraryFileNotesWorkspace._set_save_state",
        "self.query_one('#file-notes-authority', Static)",
    ): "NEEDS-LAYOUT: width:auto authority chip.",
    (
        "tldw_chatbook/Widgets/Library/library_file_notes_workspace.py",
        "LibraryFileNotesWorkspace._set_save_state",
        "status",
    ): (
        "NEEDS-LAYOUT: the save/preview status lines are height:auto and "
        "display-toggled; the poll is a 1.5 s worker-backed reconcile, not "
        "an animation clock."
    ),
    (
        "tldw_chatbook/Widgets/Library/library_file_notes_workspace.py",
        "LibraryFileNotesWorkspace._sync_large_file_preview",
        "status",
    ): (
        "NEEDS-LAYOUT: the save/preview status lines are height:auto and "
        "display-toggled; the poll is a 1.5 s worker-backed reconcile, not "
        "an animation clock."
    ),
    (
        "tldw_chatbook/Widgets/Library/library_file_notes_workspace.py",
        "LibraryFileNotesWorkspace._update_controls",
        "self.query_one('#file-notes-action-status', Static)",
    ): "NEEDS-LAYOUT: height:auto action status line.",
    (
        "tldw_chatbook/Widgets/Library/library_file_notes_workspace.py",
        "LibraryFileNotesWorkspace._update_controls",
        "self.query_one('#file-notes-path-label', Static)",
    ): (
        "NEEDS-LAYOUT: the path label wraps, so its row count tracks the "
        "path."
    ),
    (
        "tldw_chatbook/Widgets/Library/library_file_notes_workspace.py",
        "LibraryFileNotesWorkspace._update_root_surface",
        "authority",
    ): (
        "NEEDS-LAYOUT: the authority line is width:auto and display- "
        "toggled."
    ),
    (
        "tldw_chatbook/Widgets/Library/library_file_notes_workspace.py",
        "LibraryFileNotesWorkspace._update_root_surface",
        "status",
    ): (
        "NEEDS-LAYOUT: the save/preview status lines are height:auto and "
        "display-toggled; the poll is a 1.5 s worker-backed reconcile, not "
        "an animation clock."
    ),
    # -- tldw_chatbook/Widgets/audio_troubleshooting_dialog.py
    (
        "tldw_chatbook/Widgets/audio_troubleshooting_dialog.py",
        "AudioTroubleshootingDialog._update_level_meter",
        "level_text",
    ): (
        "NEEDS-LAYOUT: 'Level: N%' is width:auto and its length changes "
        "with the digit count; the tick also only runs during an explicit "
        "user-started mic test, not at idle."
    ),
    (
        "tldw_chatbook/Widgets/audio_troubleshooting_dialog.py",
        "AudioTroubleshootingDialog._update_level_meter",
        "meter",
    ): (
        "NOT-A-WIDGET: ProgressBar.update(progress=...) has no layout "
        "kwarg."
    ),
    # -- tldw_chatbook/Widgets/detailed_progress.py
    (
        "tldw_chatbook/Widgets/detailed_progress.py",
        "DetailedProgressBar._update_metrics",
        "self.query_one('#elapsed-time', Static)",
    ): (
        "UNREACHABLE: Widgets/detailed_progress.py has no importer (prod or "
        "tests)."
    ),
    (
        "tldw_chatbook/Widgets/detailed_progress.py",
        "DetailedProgressBar._update_metrics",
        "self.query_one('#memory-usage', Static)",
    ): (
        "UNREACHABLE: Widgets/detailed_progress.py has no importer (prod or "
        "tests)."
    ),
    (
        "tldw_chatbook/Widgets/detailed_progress.py",
        "DetailedProgressBar._update_metrics",
        "self.query_one('#remaining-time', Static)",
    ): (
        "UNREACHABLE: Widgets/detailed_progress.py has no importer (prod or "
        "tests)."
    ),
    (
        "tldw_chatbook/Widgets/detailed_progress.py",
        "DetailedProgressBar._update_metrics",
        "self.query_one('#speed-metric', Static)",
    ): (
        "UNREACHABLE: Widgets/detailed_progress.py has no importer (prod or "
        "tests)."
    ),
    # -- tldw_chatbook/Widgets/loading_states.py
    (
        "tldw_chatbook/Widgets/loading_states.py",
        "InlineLoader._update_dots",
        "self",
    ): (
        "UNREACHABLE: Widgets/loading_states.py has no importer (prod or "
        "tests)."
    ),
    # -- tldw_chatbook/Widgets/splash_screen.py
    (
        "tldw_chatbook/Widgets/splash_screen.py",
        "SplashScreen._display_static_fallback",
        "display",
    ): (
        "NOT-PER-TICK: _display_static_fallback runs once, on the error "
        "edge that also stops the animation timer. The per-frame repaint in "
        "_update_animation passes layout=False (TASK-21595)."
    ),
    (
        "tldw_chatbook/Widgets/splash_screen.py",
        "SplashScreen._update_animation",
        "self.effect_handler",
    ): (
        "NOT-A-WIDGET: the effect handler's own frame producer, not a "
        "Static."
    ),
    # -- tldw_chatbook/Widgets/status_dashboard.py
    (
        "tldw_chatbook/Widgets/status_dashboard.py",
        "StatusDashboard._update_time_display",
        "time_display",
    ): (
        "UNREACHABLE: Widgets/status_dashboard.py has no importer (prod or "
        "tests)."
    ),
}


# ---------------------------------------------------------------------------
# census machinery
# ---------------------------------------------------------------------------


class _Module:
    __slots__ = ("path", "tree", "dotted", "classes", "methods", "funcs", "enclosing")

    def __init__(self, path: Path, tree: ast.Module, dotted: str) -> None:
        self.path = path
        self.tree = tree
        self.dotted = dotted
        self.classes: dict[str, ast.ClassDef] = {}
        self.methods: dict[tuple[str, str], ast.AST] = {}
        self.funcs: dict[str, ast.AST] = {}
        self.enclosing: dict[ast.AST, tuple[str | None, str | None]] = {}


def _callee(node: ast.Call) -> str:
    func = node.func
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return ""


def _callback_names(arg: ast.AST) -> list[str]:
    """Method/function names a timer callback argument can refer to."""
    names: list[str] = []
    if isinstance(arg, ast.Attribute):
        names.append(arg.attr)
    elif isinstance(arg, ast.Name):
        names.append(arg.id)
    elif isinstance(arg, ast.Lambda):
        names.extend(_callee(n) for n in ast.walk(arg) if isinstance(n, ast.Call))
    elif isinstance(arg, ast.Call) and _callee(arg) == "partial" and arg.args:
        names.extend(_callback_names(arg.args[0]))
    return [name for name in names if name]


def _index_enclosing(tree: ast.Module) -> dict[ast.AST, tuple[str | None, str | None]]:
    out: dict[ast.AST, tuple[str | None, str | None]] = {}

    def walk(node: ast.AST, cls: str | None, fn: str | None) -> None:
        for child in ast.iter_child_nodes(node):
            child_cls, child_fn = cls, fn
            if isinstance(child, ast.ClassDef):
                child_cls, child_fn = child.name, None
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                child_fn = child.name
            out[child] = (child_cls, child_fn)
            walk(child, child_cls, child_fn)

    walk(tree, None, None)
    return out


@lru_cache(maxsize=1)
def _load_package() -> dict[str, _Module]:
    modules: dict[str, _Module] = {}
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (SyntaxError, UnicodeDecodeError):  # pragma: no cover
            continue
        dotted = ".".join(path.relative_to(REPO_ROOT).with_suffix("").parts)
        module = _Module(path, tree, dotted)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                module.classes[node.name] = node
                for sub in ast.walk(node):
                    if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        module.methods[(node.name, sub.name)] = sub
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                module.funcs[node.name] = node
        module.enclosing = _index_enclosing(tree)
        modules[dotted] = module
    return modules


class _CallGraph:
    def __init__(self, modules: dict[str, _Module]) -> None:
        self.modules = modules
        self.by_method: dict[str, list[tuple[str, str]]] = defaultdict(list)
        self.bases: dict[tuple[str, str], list[str]] = {}
        for dotted, module in modules.items():
            for cls, method in module.methods:
                self.by_method[method].append((dotted, cls))
            for cls, node in module.classes.items():
                names = []
                for base in node.bases:
                    if isinstance(base, ast.Name):
                        names.append(base.id)
                    elif isinstance(base, ast.Attribute):
                        names.append(base.attr)
                self.bases[(dotted, cls)] = names

    def resolve(self, dotted: str, cls: str | None, name: str):
        module = self.modules.get(dotted)
        if module is None:
            return None
        if cls and (cls, name) in module.methods:
            return (dotted, cls, name, module.methods[(cls, name)])
        for base in self.bases.get((dotted, cls or ""), []):
            if (base, name) in module.methods:
                return (dotted, base, name, module.methods[(base, name)])
        if name in module.funcs:
            return (dotted, None, name, module.funcs[name])
        return None

    def reachable(self, dotted: str, cls: str | None, name: str) -> Iterator[tuple]:
        start = self.resolve(dotted, cls, name)
        if start is None:
            return
        seen: set[tuple[str, str | None, str]] = set()
        frontier = [(start, 1)]
        while frontier:
            nxt = []
            for (mod, klass, meth, node), depth in frontier:
                key = (mod, klass, meth)
                if key in seen:
                    continue
                seen.add(key)
                yield (mod, klass, meth, node)
                if depth >= MAX_CALL_DEPTH:
                    continue
                for call in (n for n in ast.walk(node) if isinstance(n, ast.Call)):
                    callee = _callee(call)
                    if not callee:
                        continue
                    target = self.resolve(mod, klass, callee)
                    if target is None:
                        candidates = self.by_method.get(callee, [])
                        if len(candidates) == 1:
                            target = self.resolve(
                                candidates[0][0], candidates[0][1], callee
                            )
                    if target and (target[0], target[1], target[2]) not in seen:
                        nxt.append((target, depth + 1))
            frontier = nxt


def _clock_roots(modules: dict[str, _Module]) -> list[tuple[str, str, str | None, str]]:
    """Every *repeating* clock in the package.

    Returns ``(kind, module, class, callback_name)``. Two kinds:
    ``set_interval`` (a real repeating timer) and ``rearming-set_timer`` (a
    method that schedules itself again -- an interval written as a chain of
    one-shots, which is how the Persona Buddy frame clock is spelled).
    """
    roots: list[tuple[str, str, str | None, str]] = []
    for dotted, module in modules.items():
        for node in ast.walk(module.tree):
            if isinstance(node, ast.Call) and _callee(node) == REPEATING_CLOCK:
                cls, _fn = module.enclosing.get(node, (None, None))
                names = _callback_names(node.args[1]) if len(node.args) > 1 else []
                for keyword in node.keywords:
                    if keyword.arg == "callback":
                        names.extend(_callback_names(keyword.value))
                for name in names:
                    roots.append((REPEATING_CLOCK, dotted, cls, name))
        for (cls, method), node in module.methods.items():
            for call in (n for n in ast.walk(node) if isinstance(n, ast.Call)):
                if _callee(call) != ONE_SHOT_CLOCK:
                    continue
                names = _callback_names(call.args[1]) if len(call.args) > 1 else []
                for keyword in call.keywords:
                    if keyword.arg == "callback":
                        names.extend(_callback_names(keyword.value))
                if method in names:
                    roots.append((f"rearming-{ONE_SHOT_CLOCK}", dotted, cls, method))
    return roots


def _receiver(call: ast.Call) -> str | None:
    if not isinstance(call.func, ast.Attribute):
        return None
    try:
        return ast.unparse(call.func.value)
    except Exception:  # pragma: no cover
        return None


def census() -> dict[tuple[str, str, str], dict]:
    """Every ``.update(`` reachable from a repeating clock.

    Keyed ``(file, enclosing Class.method, receiver expression)``. The
    enclosing function is part of the key on purpose: a first draft keyed only
    ``(file, receiver)`` and the un-fixed ``SplashScreen._update_animation``
    site SURVIVED its mutation, because ``_display_static_fallback`` in the
    same module writes to a local also named ``display`` and its allowlist
    entry covered both.
    """
    modules = _load_package()
    graph = _CallGraph(modules)
    found: dict[tuple[str, str, str], dict] = {}
    for kind, dotted, cls, callback in _clock_roots(modules):
        for mod, klass, meth, node in graph.reachable(dotted, cls, callback):
            module = modules[mod]
            rel = str(module.path.relative_to(REPO_ROOT))
            for call in (n for n in ast.walk(node) if isinstance(n, ast.Call)):
                if _callee(call) != "update":
                    continue
                receiver = _receiver(call)
                if receiver is None:
                    continue
                explicit = any(kw.arg == "layout" for kw in call.keywords)
                record = found.setdefault(
                    (rel, f"{klass}.{meth}", receiver),
                    {"lines": set(), "explicit": True, "roots": set()},
                )
                record["lines"].add(call.lineno)
                record["explicit"] = record["explicit"] and explicit
                record["roots"].add(f"[{kind}] {dotted.split('.')[-1]}:{cls}.{callback}")
    return found


@pytest.fixture(scope="module")
def timer_path_census() -> dict[tuple[str, str, str], dict]:
    return census()


# ---------------------------------------------------------------------------
# the guard
# ---------------------------------------------------------------------------


def test_census_actually_finds_the_known_clock_roots() -> None:
    """The census must be proven to see, or the guard below passes vacuously.

    A silently-broken AST walker that returns nothing would make every other
    assertion in this module green. This pins the floor and names three roots
    that must be found: an app-wide interval, the animation clock TASK-21595
    fixed, and the self-rearming one-shot chain that the naive
    "grep for set_interval" version of this census would miss entirely.
    """
    modules = _load_package()
    roots = _clock_roots(modules)
    assert len(roots) >= 30, f"census collapsed to {len(roots)} clock roots"

    flattened = {(dotted.split(".")[-1], cls, name) for _kind, dotted, cls, name in roots}
    assert ("splash_screen", "SplashScreen", "_update_animation") in flattened
    assert (
        "console_composer_bar",
        "ConsoleComposerBar",
        "_toggle_cursor_blink",
    ) in flattened, "the TASK-21692 blink clock disappeared from the census"

    rearming = {r for r in roots if r[0].startswith("rearming-")}
    assert rearming, (
        "no self-rearming set_timer found -- an interval spelled as a chain of "
        "one-shots is exactly the shape a set_interval-only sweep misses"
    )


def test_no_timer_path_update_defaults_to_layout_true(timer_path_census) -> None:
    """Every clock-reachable ``.update(`` is classified.

    Either the call passes ``layout=`` explicitly, or ``CLASSIFIED_SITES`` says
    why it does not have to. A new repaint on a timer lands here unclassified
    and fails, which is the only signal this cost ever gets -- no other test in
    the suite counts layout operations.
    """
    unclassified = sorted(
        (path, qualname, receiver, sorted(rec["lines"]), sorted(rec["roots"]))
        for (path, qualname, receiver), rec in timer_path_census.items()
        if not rec["explicit"] and (path, qualname, receiver) not in CLASSIFIED_SITES
    )
    assert not unclassified, (
        "These `.update(` calls are reachable from a repeating clock and do not "
        "pass `layout=`. Static.update defaults to layout=True, so each of "
        "these arms a whole screen reflow per tick.\n\n"
        "Fix by either passing `layout=False` (only where the rendered size "
        "genuinely cannot change -- prove it with a geometry-equivalence A/B, "
        "see Tests/UI/test_timer_path_layout_cost.py), or by adding an entry to "
        "CLASSIFIED_SITES saying why it does not need to.\n\n"
        + "\n".join(
            f"  {path}:{lines} in {qualname} recv={receiver!r}\n"
            f"      via {roots[:2]}"
            for path, qualname, receiver, lines, roots in unclassified
        )
    )


def test_classified_sites_are_not_stale(timer_path_census) -> None:
    """Every classification still names a real ``.update(`` call.

    A refactor that deletes or renames one of these should retire its entry
    rather than leave folklore behind. Deliberately lenient about
    *reachability* (a call chain moving around is not a finding), strict about
    the call still existing.
    """
    stale: list[str] = []
    for (rel, qualname, receiver), reason in sorted(CLASSIFIED_SITES.items()):
        path = REPO_ROOT / rel
        if not path.exists():
            stale.append(f"  {rel}: file no longer exists ({reason})")
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:  # pragma: no cover
            continue
        enclosing = _index_enclosing(tree)
        live = {
            (
                "{}.{}".format(*enclosing.get(node, (None, None))),
                _receiver(node),
            )
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and _callee(node) == "update"
        }
        if (qualname, receiver) not in live:
            stale.append(
                f"  {rel}: no `.update(` on {receiver!r} in {qualname} any more "
                f"({reason})"
            )
    assert not stale, "Stale CLASSIFIED_SITES entries:\n" + "\n".join(stale)


def test_every_classification_states_a_kind() -> None:
    """Reasons must carry one of the four kinds, not just prose.

    A reason that does not say *which* kind of exemption it is claiming cannot
    be reviewed, and "documented" would degrade into "mentioned".
    """
    kinds = ("NOT-A-WIDGET:", "NEEDS-LAYOUT:", "NOT-PER-TICK:", "UNREACHABLE:")
    bad = [
        f"  {rel} / {qualname} / {receiver}: {reason}"
        for (rel, qualname, receiver), reason in sorted(CLASSIFIED_SITES.items())
        if not reason.startswith(kinds)
    ]
    assert not bad, (
        f"Classifications must start with one of {kinds}:\n" + "\n".join(bad)
    )
