"""Pure contracts of the Roleplay frame state module (frame slice B1: the header).

No Textual app is mounted here (spec 5.7.1, "pure unit"): every rule of the
one-row header -- the unsaved predicate, the item fit, the degrade order, the
escaping of untrusted text -- is a plain function of plain values.
"""

from __future__ import annotations

import ast
import itertools
from pathlib import Path
from types import SimpleNamespace

import pytest
from rich.cells import cell_len

import tldw_chatbook.app  # noqa: F401  -- collection-time import (lessons-testing-evidence: Tests/UI RecoveryRequired at setup)
from Tests.UI.python_style_inventory import inventory_styles
from tldw_chatbook.UI.Navigation.character_conversation_navigation import (
    RoleplayDraftSnapshot,
)
from tldw_chatbook.UI.Persona_Modules import roleplay_frame_state as fs
from tldw_chatbook.Widgets.glyph_fallback import set_ascii_glyph_mode

MODULE = Path(fs.__file__)
ROOT = Path(__file__).resolve().parents[2]
MODES = ("characters", "personas", "dictionaries", "lore")

#: Every production module B1 changes. The repo-wide Python-style ratchet is
#: red on dev for other modules, so a failure-set diff could not see a NEW
#: offender here; this pin can (ADR-161's hard floor; B0's convention).
B1_PRODUCTION_MODULES = (
    "tldw_chatbook/UI/Persona_Modules/roleplay_frame_state.py",
    "tldw_chatbook/UI/Persona_Modules/personas_preview_controller.py",
    "tldw_chatbook/UI/Persona_Modules/roleplay_draft_guard.py",
    "tldw_chatbook/UI/Screens/personas_screen.py",
    "tldw_chatbook/UI/Workbench/workbench_widgets.py",
)


@pytest.fixture
def ascii_mode():
    """Run one test with ASCII mode on and ALWAYS restore the off default."""
    set_ascii_glyph_mode(True)
    try:
        yield
    finally:
        set_ascii_glyph_mode(False)


def _snapshot(**overrides) -> RoleplayDraftSnapshot:
    values = {
        "form_dirty": False,
        "character_visual_dirty": False,
        "persona_visual_dirty": False,
        "attachments_dirty": False,
        "inflight_save_domains": (),
    }
    values.update(overrides)
    return RoleplayDraftSnapshot(**values)


@pytest.mark.parametrize(
    "overrides",
    [
        {},
        {"form_dirty": True},
        {"character_visual_dirty": True},
        {"persona_visual_dirty": True},
        {"attachments_dirty": True},
        {"inflight_save_domains": ("character form",)},
        {"inflight_save_domains": ("Persona form",)},
    ],
    ids=lambda o: ",".join(o) or "clean",
)
def test_unsaved_predicate_is_the_aggregate_including_inflight_saves(overrides):
    """R24: one predicate, the aggregate snapshot; an in-flight save counts."""
    snapshot = _snapshot(**overrides)
    assert fs.roleplay_has_unsaved_work(snapshot) is (not snapshot.is_clean)
    assert fs.roleplay_has_unsaved_work(snapshot) is bool(overrides)


def test_the_module_imports_no_textual():
    """Pure (spec 5.11): no ``textual`` import anywhere in the module."""
    tree = ast.parse(MODULE.read_text(encoding="utf-8"))
    imported = {
        node.module.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    } | {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    assert "textual" not in imported


def test_kind_names_every_mode_and_the_new_item_while_creating():
    assert [fs.header_kind(mode) for mode in MODES] == [
        "Characters",
        "Personas",
        "Dictionaries",
        "Lore",
    ]
    creating = fs.RoleplayHeaderInputs(mode="personas", edit_mode="create")
    assert fs.header_item(creating) == "New persona"
    assert (
        fs.header_item(fs.RoleplayHeaderInputs(mode="characters", edit_mode="create"))
        == "New character"
    )
    viewing = fs.RoleplayHeaderInputs(mode="lore", item_name="Blackreach")
    assert fs.header_item(viewing) == "Blackreach"


def test_item_fit_keeps_the_whole_line_when_it_fits():
    assert fs.fit_header_item(("Detective Sam", False), 40) == "› Detective Sam"
    assert (
        fs.fit_header_item(("Detective Sam", True), 40) == "› Detective Sam · editing"
    )
    assert fs.fit_header_item(("", False), 40) == ""
    # ``FittedText``'s own default value, before the first paint pushes one.
    assert fs.fit_header_item("", 10) == ""


def test_item_fit_cuts_the_name_first_and_keeps_the_state_word():
    assert fs.fit_header_item(("Detective Sam", True), 20) == "› Detecti… · editing"
    # No room for even one character of the name: the state word survives,
    # so the header still reads "Characters · editing".
    assert fs.fit_header_item(("Detective Sam", True), 12) == "· editing"
    assert fs.fit_header_item(("Detective Sam", True), 5) == ""
    assert fs.fit_header_item(("Detective Sam", False), 8) == "› Detec…"
    assert fs.fit_header_item(("Detective Sam", False), 3) == ""


@pytest.mark.parametrize(
    "name",
    [
        "探偵サム・スペード",
        "Sam 🕵️ Spade",
        "Zero\u200bwidth\u200bname",
        "[/]",
        "x" * 300,
    ],
)
def test_item_fit_never_paints_past_its_width(name):
    for width, editing in itertools.product(range(0, 60), (False, True)):
        fitted = fs.fit_header_item((name, editing), width)
        assert cell_len(fitted) <= width, (name, width, fitted)


def test_item_fit_uses_ascii_markers_in_ascii_mode(ascii_mode):
    assert (
        fs.fit_header_item(("Detective Sam", True), 40) == "> Detective Sam - editing"
    )
    assert fs.fit_header_item(("Detective Sam", False), 10) == "> Detec..."


def test_ellipsize_measures_cells_and_resolves_the_glyph():
    assert fs.ellipsize_cells("abcdef", 6) == "abcdef"
    assert fs.ellipsize_cells("abcdef", 4) == "abc…"
    assert fs.ellipsize_cells("abcdef", 1) == ""
    assert fs.ellipsize_cells("探偵サム", 5) == "探偵…"


def _all_inputs():
    for mode, edit_mode, unsaved, blocked, source in itertools.product(
        MODES, ("view", "edit"), (False, True), (False, True), ("local", "server")
    ):
        yield fs.RoleplayHeaderInputs(
            mode=mode,
            edit_mode=edit_mode,
            item_name="A rather long character name for the header",
            unsaved=unsaved,
            provider_blocked=blocked,
            runtime_source=source,
            server_label="home-tldw.example.internal:8000"
            if source == "server"
            else "",
        )


def _required(view: fs.RoleplayHeaderView) -> int:
    return fs._required_cells(
        view.state.subtitle, (view.unsaved_chip, view.blocked_chip), view.status_plain
    )


def test_header_never_says_ready_and_always_names_the_kind():
    """RP-067: the false "Ready" badge is gone; the status is the data source."""
    for inputs in _all_inputs():
        for width in (40, 80, 120, 160, 220):
            view = fs.build_header_view(inputs, width)
            assert view.state.title == "Roleplay"
            assert view.state.subtitle == fs.header_kind(inputs.mode)
            assert view.status_plain != "Ready"
            assert view.state.status_label  # never empty: "Ready" never paints
            assert view.status_plain.startswith(
                "Local" if inputs.runtime_source == "local" else "Server"
            )
    # The composed state, before any input is gathered, already names both.
    for mode, source in itertools.product(MODES, ("local", "server")):
        state = fs.initial_header_state(mode, source)
        assert (state.title, state.subtitle) == ("Roleplay", fs.header_kind(mode))
        assert state.status_label == ("Local" if source == "local" else "Server")


def test_header_fits_from_65_columns_degrading_in_order():
    """Everything fits from 65 columns (the worst case's floor), and the
    degrade steps only ever shorten: status label, Settings, then "Server"."""
    for inputs in _all_inputs():
        previous = None
        for width in range(65, 261):
            view = fs.build_header_view(inputs, width)
            assert _required(view) <= width, (inputs, width, view)
            if previous is not None:
                # Wider never paints less.
                assert cell_len(view.status_plain) >= cell_len(previous.status_plain)
                assert cell_len(view.blocked_chip) >= cell_len(previous.blocked_chip)
            previous = view


def test_unsaved_chip_is_short_below_100_columns_and_absent_when_clean():
    dirty = fs.RoleplayHeaderInputs(mode="characters", unsaved=True)
    assert fs.build_header_view(dirty, 99).unsaved_chip == "Unsaved"
    assert fs.build_header_view(dirty, 100).unsaved_chip == "Unsaved changes"
    clean = fs.RoleplayHeaderInputs(mode="characters")
    assert fs.build_header_view(clean, 160).unsaved_chip == ""


def test_blocked_chip_shows_only_for_a_blocked_destination():
    blocked = fs.RoleplayHeaderInputs(mode="characters", provider_blocked=True)
    assert fs.build_header_view(blocked, 160).blocked_chip == (
        "No chat provider · Settings ›"
    )
    assert fs.build_header_view(blocked, 66).blocked_chip in {
        "No chat provider · Settings ›",
        "No chat provider ›",
    }
    ready = fs.RoleplayHeaderInputs(mode="characters")
    assert fs.build_header_view(ready, 160).blocked_chip == ""


def test_server_status_is_escaped_for_the_markup_header_but_measured_plain():
    """R33: the label is cut as plain text, then escaped for the markup-on chip."""
    inputs = fs.RoleplayHeaderInputs(
        mode="characters", runtime_source="server", server_label="[/]home"
    )
    view = fs.build_header_view(inputs, 160)
    assert view.status_plain == "Server: [/]home · read-only"
    assert view.state.status_label == "Server: \\[/]home · read-only"
    long_label = fs.RoleplayHeaderInputs(
        mode="characters", runtime_source="server", server_label="h" * 60
    )
    status = fs.build_header_view(long_label, 220).status_plain
    assert (
        cell_len(status) == cell_len("Server:  · read-only") + fs.SERVER_LABEL_MAX_CELLS
    )


def test_runtime_server_label_reads_the_label_then_the_id():
    def app(**state):
        return SimpleNamespace(
            runtime_policy=SimpleNamespace(state=SimpleNamespace(**state))
        )

    assert (
        fs.runtime_server_label(
            app(last_known_server_label=" home ", active_server_id="t-7")
        )
        == "home"
    )
    assert (
        fs.runtime_server_label(
            app(last_known_server_label=None, active_server_id="t-7")
        )
        == "t-7"
    )
    assert fs.runtime_server_label(app(last_known_server_label=object())) == ""
    assert fs.runtime_server_label(object()) == ""


def test_purpose_line_keeps_the_pre_move_copy():
    """Moved verbatim from the screen in B1 (F-033); B6 retires it."""
    assert fs.purpose_line("characters", 2) == "Characters — who the AI plays · 2"
    assert fs.purpose_line("personas", None) == "Personas — who you play in the chat."
    assert fs.mode_descriptor("unknown") == "unknown"


def test_the_pane_class_names_are_the_shell_parts_in_field_order():
    """(shell, nav, items, work, grip): B7 builds ``AdaptivePaneClasses`` from them."""
    assert fs.ROLEPLAY_PANE_CLASS_NAMES == (
        "roleplay-shell",
        "roleplay-nav",
        "roleplay-items",
        "roleplay-shell-work",
        "roleplay-shell-grip",
    )


def test_b1_production_modules_carry_no_python_style_violation():
    violations = {
        module: [
            (write.line, write.property, write.form)
            for write in inventory_styles((ROOT / module).read_text(encoding="utf-8"))
            if write.violation
        ]
        for module in B1_PRODUCTION_MODULES
    }
    assert {module: found for module, found in violations.items() if found} == {}
