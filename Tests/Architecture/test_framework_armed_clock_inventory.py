"""Census: repeating clocks that TEXTUAL arms, which no timer sweep can see.

The timer-path census (``test_timer_path_static_update_inventory.py``) walks
package sources for ``set_interval``-family calls. Some repeating clocks are
armed inside ``textual/dom.py`` instead, by widgets the package merely
CONSTRUCTS -- no package file spells a timer at all (TASK-23028):

* ``ProgressBar(total=None)`` (or ``total`` omitted) makes Textual's ``Bar``
  indeterminate: ``_start_indeterminate`` sets ``auto_refresh = 1/15`` -- a
  **15 Hz repeating clock for as long as the widget is mounted**.
* ``LoadingIndicator`` sets ``auto_refresh = 1/16`` in ``_on_mount`` -- a
  16 Hz clock.
* Any direct ``widget.auto_refresh = <interval>`` assignment or
  ``auto_refresh=`` constructor kwarg arms the same mechanism by hand.

**``display = False`` does not stop any of these.** Textual gates only the
repaint on ``is_on_screen``, never the timer -- the 2026-08-27 perf review
measured four invisible indeterminate ProgressBars burning 88% of the Lab
screen's idle CPU (960 of 1018 timer fires in 15 s changing zero pixels),
plus hidden 16 Hz LoadingIndicators on Library/Personas surfaces.

This census therefore rebuilds the inventory on every run: every
indeterminate ``ProgressBar`` construction and every ``LoadingIndicator``
construction must carry a row saying what bounds its clock, and any
``auto_refresh`` arming in package code (currently ZERO -- asserted, not
assumed) must be classified the same way.

Row kinds:

  ``BOUNDED:``              the widget is only mounted while a real operation
                            runs and is unmounted/replaced after, so the
                            clock's lifetime is the operation's.
  ``HIDDEN-WHILE-MOUNTED:`` an ACKNOWLEDGED live idle clock: the widget stays
                            mounted and is merely hidden. This kind is an
                            acknowledgement with an owner, never an
                            absolution -- the row must name the task that
                            owns the instance fix.
  ``UNREACHABLE:``          the module has no importer (prod or tests).

Instance fixes are owned by TASK-23022 (PR #2156, which also carries its own
exemption-free mechanism guard, ``test_progress_widget_clock_guard.py``);
this file deliberately ships none of them and stays construction-shaped so
the two complement rather than collide. When that task lands, its rows go
stale here and the staleness test says to retire them -- that hand-off is
the intended collision.

Boundary, stated rather than implied: a ``ProgressBar(total=<expression>)``
whose expression might evaluate to ``None`` at runtime is not censused (the
AST cannot price it); only the omitted/``None``-literal shapes are. The
``Widget.loading = True`` property (which mounts a transient
``LoadingIndicator``) is likewise out of scope here.
"""

from __future__ import annotations

import ast
from functools import lru_cache

import pytest

from Tests.Architecture.test_timer_path_static_update_inventory import (
    REPO_ROOT,
    _callee,
    _load_package,
    _Module,
    _modules_from_source,
)

#: Widgets whose construction alone arms a repeating framework clock.
INDETERMINATE_PROGRESS = "ProgressBar"
LOADING_INDICATOR = "LoadingIndicator"


# ---------------------------------------------------------------------------
# inventory
# ---------------------------------------------------------------------------
#
# Key: (path relative to the repo root, enclosing `Class.method`, the
#       construction as unparsed source).
# Value: what bounds this clock -- one of the kinds in the module docstring.
#
FRAMEWORK_ARMED_CLOCK_ROWS: dict[tuple[str, str, str], str] = {
    (
        "tldw_chatbook/UI/CCP_Modules/ccp_loading_indicators.py",
        "CCPLoadingWidget.compose",
        "LoadingIndicator()",
    ): (
        "HIDDEN-WHILE-MOUNTED: CCPLoadingManager.setup() mounts this widget "
        "onto the Personas screen permanently and toggles a 'visible' CSS "
        "class -- the 16 Hz indicator clock runs the whole time Personas is "
        "mounted (the 08-27 perf review's 'Personas at 16 Hz'). Instance fix "
        "owned by TASK-23022; retire this row with it."
    ),
    (
        "tldw_chatbook/UI/CodeRepoCopyPasteWindow.py",
        "CodeRepoCopyPasteWindow.compose",
        "LoadingIndicator()",
    ): (
        "UNREACHABLE: CodeRepoCopyPasteWindow has no importer outside "
        "css/build_css.py and no screen_registry route."
    ),
    (
        "tldw_chatbook/UI/Screens/stats_screen.py",
        "StatsScreen.compose_content",
        "LoadingIndicator()",
    ): (
        "BOUNDED: initial 'Initializing statistics...' placeholder; "
        "refresh_stats_display() clears #stats-content (remove_children) as "
        "soon as data or an error arrives, unmounting the indicator."
    ),
    (
        "tldw_chatbook/UI/Screens/stats_screen.py",
        "StatsScreen.refresh_stats_display",
        "LoadingIndicator()",
    ): (
        "BOUNDED: mounted only while is_loading, inside a container the same "
        "method removes and rebuilds when the state leaves loading."
    ),
    (
        "tldw_chatbook/Widgets/Console/console_conversation_inspector.py",
        "ConsoleConversationInspector.compose",
        "LoadingIndicator(id='console-inspector-next-send-loading')",
    ): (
        "HIDDEN-WHILE-MOUNTED: display:none by default, '.loading' class "
        "toggles it -- the 16 Hz clock runs whenever the inspector is "
        "mounted, loading or not. Real idle clock found by this census "
        "(TASK-23028); instance fix owned by TASK-23022's hidden-clock "
        "family. Retire this row with the fix."
    ),
    (
        "tldw_chatbook/Widgets/ModelArtifacts/install_progress.py",
        "ModelInstallProgress.compose",
        "ProgressBar(total=None, show_eta=False, id='model-install-progress-bar')",
    ): (
        "HIDDEN-WHILE-MOUNTED: the Lab screen's model rows each mount one "
        "of these; total=None keeps the Bar indeterminate, a 15 Hz clock "
        "per instance even at display:none -- the 08-27 perf review "
        "measured four of them as 88% of the Lab screen's idle CPU. "
        "Instance fix owned by TASK-23022; retire this row with it."
    ),
    (
        "tldw_chatbook/Widgets/audio_troubleshooting_dialog.py",
        "AudioTroubleshootingDialog.compose",
        "LoadingIndicator(id='status-loading')",
    ): (
        "BOUNDED: transient troubleshooting dialog. Honest caveat: the init "
        "path only sets display=False on completion, which hides the "
        "indicator but does NOT stop its 16 Hz timer -- it ticks for the "
        "dialog's remaining (user-bounded) lifetime."
    ),
    (
        "tldw_chatbook/Widgets/detailed_progress.py",
        "DetailedProgressBar.compose",
        "ProgressBar(id='main-progress', show_eta=False, show_percentage=True)",
    ): (
        "UNREACHABLE: Widgets/detailed_progress.py has no importer (prod or "
        "tests) -- same verdict as its rows in the timer-path census."
    ),
    (
        "tldw_chatbook/Widgets/detailed_progress.py",
        "DetailedProgressBar.compose",
        "ProgressBar(id='stage-progress', show_eta=False, show_percentage=True, "
        "classes='stage-progress')",
    ): (
        "UNREACHABLE: Widgets/detailed_progress.py has no importer (prod or "
        "tests) -- same verdict as its rows in the timer-path census."
    ),
    (
        "tldw_chatbook/Widgets/document_generation_modal.py",
        "DocumentGenerationModal.compose",
        "LoadingIndicator()",
    ): (
        "BOUNDED: transient modal; #loading-container.display is toggled "
        "with the generation. Same caveat as the audio dialog: hiding does "
        "not stop the 16 Hz timer, which runs while the modal is open."
    ),
    (
        "tldw_chatbook/Widgets/enhanced_sidebar.py",
        "SidebarSection.compose",
        "LoadingIndicator(classes='section-loading hidden')",
    ): (
        "UNREACHABLE: Widgets/enhanced_sidebar.py has no importer (prod or "
        "tests). Two identical constructions share this key; both are in "
        "the same unreachable compose."
    ),
    (
        "tldw_chatbook/Widgets/loading_states.py",
        "LoadingState.compose",
        "LoadingIndicator()",
    ): (
        "UNREACHABLE: Widgets/loading_states.py has no importer (prod or "
        "tests) -- same verdict as its rows in the timer-path census."
    ),
}


# ---------------------------------------------------------------------------
# detectors
# ---------------------------------------------------------------------------


def _qualname(module: _Module, node: ast.AST) -> str:
    cls, fn = module.enclosing.get(node, (None, None))
    return f"{cls}.{fn}"


def _is_none_literal(node: ast.AST | None) -> bool:
    return isinstance(node, ast.Constant) and node.value is None


def _progress_bar_is_indeterminate(call: ast.Call) -> bool:
    """total omitted, or literally None -- the shapes that arm the 15 Hz Bar."""
    positional_total = None
    if call.args and not isinstance(call.args[0], ast.Starred):
        positional_total = call.args[0]
    keyword_total = None
    has_keyword_total = False
    for keyword in call.keywords:
        if keyword.arg == "total":
            has_keyword_total = True
            keyword_total = keyword.value
        elif keyword.arg is None:
            # `ProgressBar(**kwargs)` MIGHT carry a total; the AST cannot
            # tell. Treat as not-indeterminate (documented boundary).
            has_keyword_total = True
    if positional_total is not None:
        return _is_none_literal(positional_total)
    if has_keyword_total:
        return _is_none_literal(keyword_total)
    return True


def framework_clock_constructions(
    modules: dict[str, _Module] | None = None,
) -> dict[tuple[str, str, str], str]:
    """Every construction that arms a framework clock, keyed like the rows.

    Values are short labels: ``indeterminate-progress-bar`` or
    ``loading-indicator``.
    """
    if modules is None:
        return _package_framework_clocks()
    found: dict[tuple[str, str, str], str] = {}
    for module in modules.values():
        rel = str(module.path.relative_to(REPO_ROOT))
        for node in ast.walk(module.tree):
            if not isinstance(node, ast.Call):
                continue
            callee = _callee(node)
            if callee == INDETERMINATE_PROGRESS:
                if _progress_bar_is_indeterminate(node):
                    key = (rel, _qualname(module, node), ast.unparse(node))
                    found[key] = "indeterminate-progress-bar"
            elif callee == LOADING_INDICATOR:
                key = (rel, _qualname(module, node), ast.unparse(node))
                found[key] = "loading-indicator"
    return found


def auto_refresh_armings(
    modules: dict[str, _Module] | None = None,
) -> dict[tuple[str, str, str], str]:
    """Every hand-armed ``auto_refresh`` in package code.

    Both spellings: ``something.auto_refresh = interval`` and an
    ``auto_refresh=`` constructor kwarg. ``auto_refresh = None`` (disarming)
    is not a clock and is skipped.
    """
    if modules is None:
        return _package_auto_refresh_armings()
    found: dict[tuple[str, str, str], str] = {}
    for module in modules.values():
        rel = str(module.path.relative_to(REPO_ROOT))
        for node in ast.walk(module.tree):
            if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                targets = (
                    node.targets if isinstance(node, ast.Assign) else [node.target]
                )
                value = getattr(node, "value", None)
                for target in targets:
                    if (
                        isinstance(target, ast.Attribute)
                        and target.attr == "auto_refresh"
                        and not _is_none_literal(value)
                    ):
                        key = (rel, _qualname(module, node), ast.unparse(node))
                        found[key] = "auto_refresh-assignment"
            elif isinstance(node, ast.Call):
                for keyword in node.keywords:
                    if keyword.arg == "auto_refresh" and not _is_none_literal(
                        keyword.value
                    ):
                        key = (rel, _qualname(module, node), ast.unparse(node))
                        found[key] = "auto_refresh-kwarg"
    return found


@lru_cache(maxsize=1)
def _package_framework_clocks() -> dict[tuple[str, str, str], str]:
    return framework_clock_constructions(_load_package())


@lru_cache(maxsize=1)
def _package_auto_refresh_armings() -> dict[tuple[str, str, str], str]:
    return auto_refresh_armings(_load_package())


@pytest.fixture(scope="module")
def framework_clocks() -> dict[tuple[str, str, str], str]:
    """The construction census, built once for the module."""
    return framework_clock_constructions()


# ---------------------------------------------------------------------------
# the guard
# ---------------------------------------------------------------------------


def test_every_framework_armed_clock_is_classified(framework_clocks) -> None:
    """Constructing one of these widgets IS arming a repeating clock.

    A new indeterminate ProgressBar or LoadingIndicator lands here
    unclassified and fails until its author says what bounds the clock --
    exactly the acknowledgement the six instances the 08-27 perf review found
    never got.
    """
    unclassified = sorted(
        (key, label)
        for key, label in framework_clocks.items()
        if key not in FRAMEWORK_ARMED_CLOCK_ROWS
    )
    assert not unclassified, (
        "These constructions arm a repeating framework clock (15-16 Hz via "
        "textual/dom.py auto_refresh) with no row saying what bounds it. "
        "display=False does NOT stop the timer. Either give the ProgressBar "
        "a real total, mount the indicator only while its operation runs, or "
        "add a FRAMEWORK_ARMED_CLOCK_ROWS entry:\n\n"
        + "\n".join(
            f"  [{label}] {rel} in {qualname}\n      {construction}"
            for (rel, qualname, construction), label in unclassified
        )
    )


def test_no_unclassified_auto_refresh_arming() -> None:
    """Package code arms ``auto_refresh`` nowhere -- asserted, not implied.

    The 08-27 review noted the timer census could never see the ProgressBar
    clocks because "no package file assigns auto_refresh" -- which was true
    only by accident. This makes it a contract: the first hand-armed
    auto_refresh in package code must land a classification row here.
    """
    armings = auto_refresh_armings()
    unclassified = {
        key: label
        for key, label in armings.items()
        if key not in FRAMEWORK_ARMED_CLOCK_ROWS
    }
    assert not unclassified, (
        "Hand-armed auto_refresh clocks without a FRAMEWORK_ARMED_CLOCK_ROWS "
        "entry:\n"
        + "\n".join(
            f"  [{label}] {rel} in {qualname}: {source}"
            for (rel, qualname, source), label in sorted(unclassified.items())
        )
    )


def test_inventory_rows_are_not_stale(framework_clocks) -> None:
    """Every row still matches a live construction (or arming).

    When TASK-23022 fixes an instance (gives the ProgressBar a total, bounds
    an indicator's mount), its row here goes red with this message: retire
    the row in the same diff. A row that outlives its site would silently
    re-classify a future clock re-introduced at the same spot.
    """
    live = set(framework_clocks) | set(auto_refresh_armings())
    stale = sorted(set(FRAMEWORK_ARMED_CLOCK_ROWS) - live)
    assert not stale, (
        "FRAMEWORK_ARMED_CLOCK_ROWS entries with no matching live "
        "construction -- the instance was fixed or moved (TASK-23022?); "
        "retire these rows:\n"
        + "\n".join(f"  {row}" for row in stale)
    )


def test_every_row_states_a_kind() -> None:
    """Rows must carry one of the three kinds, not just prose."""
    kinds = ("BOUNDED:", "HIDDEN-WHILE-MOUNTED:", "UNREACHABLE:")
    bad = [
        f"  {key}: {reason}"
        for key, reason in sorted(FRAMEWORK_ARMED_CLOCK_ROWS.items())
        if not reason.startswith(kinds)
    ]
    assert not bad, (
        f"Classifications must start with one of {kinds}:\n" + "\n".join(bad)
    )


def test_hidden_rows_name_an_owner() -> None:
    """An acknowledged live idle clock must point at the task that owns it.

    HIDDEN-WHILE-MOUNTED is the one kind that concedes a real per-frame cost
    is shipping. Without an owner it would quietly become permanent.
    """
    bad = [
        f"  {key}"
        for key, reason in sorted(FRAMEWORK_ARMED_CLOCK_ROWS.items())
        if reason.startswith("HIDDEN-WHILE-MOUNTED:") and "TASK-" not in reason
    ]
    assert not bad, (
        "HIDDEN-WHILE-MOUNTED rows must cite the TASK that owns the instance "
        "fix:\n" + "\n".join(bad)
    )


# ---------------------------------------------------------------------------
# mutation fixtures: the blind spot, reintroduced, must go RED
# ---------------------------------------------------------------------------


def test_hidden_indeterminate_progress_bar_is_detected() -> None:
    """TASK-23028 regression pin: the exact Lab-screen shape.

    ``ProgressBar(total=None)`` inside a widget that is display-toggled --
    the construction that put a permanent 15 Hz clock behind ``display =
    False`` four times over on the Lab screen. The census must find it with
    no timer spelled anywhere in the module.
    """
    modules = _modules_from_source(
        """
        class InstallProgressRow:
            def compose(self):
                bar = ProgressBar(total=None, show_eta=False, id="install-bar")
                bar.display = False
                yield bar
        """
    )
    found = framework_clock_constructions(modules)
    assert list(found.values()) == ["indeterminate-progress-bar"]
    ((rel, qualname, construction),) = found.keys()
    assert qualname == "InstallProgressRow.compose"
    assert "total=None" in construction


def test_omitted_total_is_indeterminate_and_literal_total_is_not() -> None:
    """``ProgressBar()`` arms the clock; ``ProgressBar(total=100)`` does not.

    Also pins the boundary: a dynamic ``total=expr`` is NOT censused --
    weakening `_progress_bar_is_indeterminate` toward flagging everything
    would bury the real findings in noise, and toward flagging nothing would
    reopen the blind spot.
    """
    modules = _modules_from_source(
        """
        class Rows:
            def a(self):
                yield ProgressBar()

            def b(self):
                yield ProgressBar(total=100)

            def c(self):
                yield ProgressBar(total=self.expected_total)

            def d(self):
                yield ProgressBar(100)
        """
    )
    found = framework_clock_constructions(modules)
    qualnames = sorted(qualname for (_rel, qualname, _src) in found)
    assert qualnames == ["Rows.a"], (
        "indeterminate detection drifted: only the omitted-total construction "
        f"should be censused, got {qualnames}"
    )


def test_loading_indicator_construction_is_detected() -> None:
    """Every LoadingIndicator construction is a 16 Hz clock while mounted."""
    modules = _modules_from_source(
        """
        class Overlay:
            def compose(self):
                yield LoadingIndicator(classes="loading-overlay hidden")
        """
    )
    found = framework_clock_constructions(modules)
    assert list(found.values()) == ["loading-indicator"]


def test_auto_refresh_arming_is_detected_in_both_spellings() -> None:
    """Assignment and constructor-kwarg armings are both censused."""
    modules = _modules_from_source(
        """
        class HandArmed:
            def on_mount(self):
                self.auto_refresh = 1 / 15

            def build(self):
                return SomeWidget(auto_refresh=0.5)

            def disarm(self):
                self.auto_refresh = None
        """
    )
    found = auto_refresh_armings(modules)
    labels = sorted(found.values())
    assert labels == ["auto_refresh-assignment", "auto_refresh-kwarg"], (
        "disarming (auto_refresh = None) must not be censused; both arming "
        f"spellings must be. Got {labels}"
    )
