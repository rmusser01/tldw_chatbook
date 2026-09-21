"""Repo-wide responsive wide tier for modal surfaces: contracts + spot geometry.

Two guards over the single shared mechanism (the rollout of the per-surface
tiers shipped for the Conversation settings modal in PR #2670 and the Alt+M
model popover in PR #2672):

* Coverage contract -- every anchor registered in
  ``Tests/UI/modal_wide_tier_registry.py`` has an ``App.-wide-viewport`` rule
  in the built bundle (and the source module) with the registered cap, every
  rule in the block is registered, and every anchor still targets real code
  (the owner module names its id/class/type). A renamed modal id without a
  tier update fails here instead of silently losing the wide tier.
* Spot geometry -- one representative per cap tier, mounted in a
  consolidated-CSS harness with the REAL app toggle
  (``WideViewportTierMixin`` from ``tldw_chatbook.app``), asserting the
  tier engages at >= 150 viewport columns, produces
  ``min(cap, viewport * 85%)``, leaves base geometry intact below the
  threshold, and re-syncs across a live resize in both directions.
"""

from __future__ import annotations

import ast
import re
import tempfile
from pathlib import Path

import pytest

from Tests.UI.consolidated_css import APP_STYLESHEETS, ConsolidatedCSSApp
from Tests.UI.modal_wide_tier_registry import (
    MODAL_WIDE_TIER,
    MODAL_WIDE_TIER_SKIPPED,
    MODAL_WIDE_TIER_SKIPPED_MODULES,
    WIDE_TIER_CAPS,
    WIDE_TIER_WIDTH_PERCENT,
    WIDE_VIEWPORT_COLUMNS,
)
from tldw_chatbook.Library.library_prompts_state import (
    begin_prompt_collection_catalog,
)
from tldw_chatbook.Notes.recovery_review import NotesRecoveryReview
from tldw_chatbook.Skills_Interop.recovery_activation import RecoveryReview
from tldw_chatbook.UI.ChatbookCreationWindow import ChatbookCreationWindow
from tldw_chatbook.UI.ChatbookExportManagementWindow import (
    ChatbookExportManagementWindow,
)
from tldw_chatbook.UI.ChatbookTemplatesWindow import ChatbookTemplatesWindow
from tldw_chatbook.UI.Library_Modules.prompt_collection_manager_modal import (
    PromptCollectionManagerModal,
)
from tldw_chatbook.UI.Screens.skills_screen import SkillRecoveryReviewModal
from tldw_chatbook.Widgets.Console.console_prompts_modal import ConsolePromptsModal
from tldw_chatbook.Widgets.Console.console_reaction_picker_modal import (
    ConsoleReactionPickerModal,
)
from tldw_chatbook.Widgets.Console.console_system_prompt_modal import (
    ConsoleSystemPromptModal,
)
from tldw_chatbook.Widgets.Library.notes_recovery_dialog import NotesRecoveryDialog
from tldw_chatbook.Widgets.Persona_Widgets.buddy_management_modal import (
    BuddyManagementModal,
)
from tldw_chatbook.Widgets.template_selector import TemplateSelectorDialog

REPO_ROOT = Path(__file__).resolve().parents[2]
CSS_ROOT = REPO_ROOT / "tldw_chatbook/css"

#: The source module that owns the wide-tier block.
WIDE_TIER_SOURCE = CSS_ROOT / "components/_agentic_terminal.tcss"

#: The boot bundle the app parses (rebuilt from the modules by build_css).
WIDE_TIER_BUNDLE = CSS_ROOT / "tldw_cli_modular.tcss"

#: One ``App.-wide-viewport <anchor> { width: $ds-percent-85; max-width: N }``.
WIDE_RULE_RE = re.compile(
    r"App\.-wide-viewport\s+([^\n{}]+?)\s*\{\s*"
    r"width:\s*\$ds-percent-85;\s*"
    r"max-width:\s*(\$ds-[a-z0-9-]+|\d+);\s*"
    r"\}",
)

#: Comment banners the block is wrapped in, so drift in the block's position
#: inside the module is still visible in failures.
WIDE_TIER_BANNER = "Responsive wide tier (repo-wide)"


def _token_values() -> dict[str, str]:
    """Parse ``$ds-*`` definitions from the token file (integers as str)."""
    text = (CSS_ROOT / "core/_variables.tcss").read_text(encoding="utf-8")
    return dict(re.findall(r"^\s*(\$ds-[a-z0-9-]+)\s*:\s*([^;]+);", text, re.M))


def _wide_rules(text: str) -> dict[str, int]:
    """Map anchor -> effective max-width cap for every wide-tier rule."""
    tokens = _token_values()

    def resolve(value: str) -> int:
        if value.startswith("$"):
            return int(tokens[value])
        return int(value)

    return {
        anchor.strip(): resolve(cap)
        for anchor, cap in WIDE_RULE_RE.findall(text)
    }


def test_wide_tier_block_exists_in_source_module() -> None:
    """The block lands in the owning module, banner and all."""
    source = WIDE_TIER_SOURCE.read_text(encoding="utf-8")
    assert WIDE_TIER_BANNER in source
    assert "App.-wide-viewport" in source


@pytest.mark.parametrize("path", [WIDE_TIER_SOURCE, WIDE_TIER_BUNDLE])
def test_every_registered_anchor_has_a_wide_tier_rule(path: Path) -> None:
    """The registry and the shipped block agree, anchor for anchor."""
    rules = _wide_rules(path.read_text(encoding="utf-8"))
    expected = {anchor: cap for anchor, cap, _owner in MODAL_WIDE_TIER}
    assert rules, f"no App.-wide-viewport rules parsed from {path.name}"
    assert rules == expected


def test_every_wide_tier_rule_is_registered() -> None:
    """No orphan rules: an unregistered rule means the registry is stale."""
    for path in (WIDE_TIER_SOURCE, WIDE_TIER_BUNDLE):
        rules = _wide_rules(path.read_text(encoding="utf-8"))
        registered = {anchor for anchor, _cap, _owner in MODAL_WIDE_TIER}
        assert set(rules) <= registered, (path.name, set(rules) - registered)


@pytest.mark.parametrize(["anchor", "cap", "owner"], MODAL_WIDE_TIER)
def test_every_anchor_targets_real_code(
    anchor: str, cap: int, owner: str
) -> None:
    """Each registered anchor resolves inside its owner module.

    A renamed modal id (or a moved file) fails here naming the owner, so the
    tier cannot silently keep styling an id nothing mounts.
    """
    owner_path = REPO_ROOT / "tldw_chatbook" / owner
    assert owner_path.exists(), owner
    owner_text = owner_path.read_text(encoding="utf-8")
    for token in re.findall(
        r"(?<!\w)([#.][A-Za-z0-9_-]+|[A-Z][A-Za-z0-9]+)(?!\w)", anchor
    ):
        assert token.lstrip("#.") in owner_text, (anchor, token, owner)


# --------------------------------------------------------------------------
# Reverse sweep: no ModalScreen subclass may sit OUTSIDE the tier silently.
#
# The forward contract above only sees anchors someone remembered to
# register; PR #2742's review found eight substantial modals the inventory
# never scoped. This sweep walks the AST for every ModalScreen subclass
# (direct or inherited), marks one "anchored" when a registry anchor's tokens
# resolve inside the class's own source segment (or an anchored base class),
# and fails naming every class that is neither anchored nor explicitly
# skipped. Stale skip entries (class gone, or module without unanchored
# modals) fail too, so the skip lists cannot rot into folklore.
# --------------------------------------------------------------------------

#: Bases that terminate modal-resolution: Textual/widget plumbing that never
#: carries modal geometry of its own.
_NON_MODAL_BASES = frozenset(
    {
        "ModalScreen",
        "Screen",
        "Widget",
        "App",
        "object",
        "Vertical",
        "Horizontal",
        "Container",
        "VerticalScroll",
        "HorizontalScroll",
        "ScrollableContainer",
        "CenterMiddle",
        "Static",
        "ListView",
        "DataTable",
        "Grid",
    }
)

_ANCHOR_TOKENS = [
    re.findall(r"(?<!\w)([#.][A-Za-z0-9_-]+|[A-Z][A-Za-z0-9]+)(?!\w)", anchor)
    for anchor, _cap, _owner in MODAL_WIDE_TIER
]


def _scan_modal_classes() -> dict[str, tuple[str, list[str], str]]:
    """Map class name -> (module relpath, base names, source segment).

    AST-walks every non-vendored module under ``tldw_chatbook`` collecting
    class definitions; only the ModalScreen subset survives filtering by the
    callers (this keeps one parse for both the modal check and anchoring).
    """
    classes: dict[str, tuple[str, list[str], str]] = {}
    for py in (REPO_ROOT / "tldw_chatbook").rglob("*.py"):
        if "Third_Party" in py.parts:
            continue  # vendored code (e.g. textual_fspicker) -- out of scope
        try:
            text = py.read_text(encoding="utf-8")
            tree = ast.parse(text)
        except SyntaxError:
            continue
        rel = py.relative_to(REPO_ROOT / "tldw_chatbook").as_posix()
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue

            def base_name(base: ast.expr) -> str:  # noqa: ANN001 - ast node
                if isinstance(base, ast.Name):
                    return base.id
                if isinstance(base, ast.Attribute):
                    return base.attr
                if isinstance(base, ast.Subscript):
                    return base_name(base.value)
                return ""

            classes[node.name] = (
                rel,
                [base_name(base) for base in node.bases],
                ast.get_source_segment(text, node) or "",
            )
    return classes


def _is_modal(
    name: str,
    classes: dict[str, tuple[str, list[str], str]],
    _seen: frozenset[str] | None = None,
) -> bool:
    """True when ``name`` reaches ModalScreen through its base chain."""
    if name == "ModalScreen":
        return True
    if name in _NON_MODAL_BASES or name not in classes:
        return False
    seen = _seen or frozenset()
    if name in seen:
        return False
    seen = seen | {name}
    return any(_is_modal(base, classes, seen) for base in classes[name][1])


def _is_anchored(
    name: str,
    classes: dict[str, tuple[str, list[str], str]],
    _seen: frozenset[str] | None = None,
) -> bool:
    """True when a registry anchor resolves inside ``name``'s own segment.

    An anchor matches when the class name is one of the anchor's type tokens
    (``FileExtractionDialog > Vertical``) or every id/class token of the
    anchor appears in the class's source segment and the anchor carries no
    type token from a different class. Anchored bases propagate: a modal
    subclassing an anchored modal inherits its tier rule.
    """
    if name not in classes:
        return False
    _bases, segment = classes[name][1], classes[name][2]
    for tokens in _ANCHOR_TOKENS:
        types = [t for t in tokens if not t.startswith(("#", "."))]
        sels = [t.lstrip("#.") for t in tokens if t.startswith(("#", "."))]
        if name in types and all(sel in segment for sel in sels):
            return True
        if not types and sels and all(sel in segment for sel in sels):
            return True
    seen = _seen or frozenset()
    if name in seen:
        return False
    return any(
        _is_anchored(base, classes, seen | {name}) for base in _bases if base in classes
    )


def test_every_modalscreen_class_is_tiered_or_explicitly_skipped() -> None:
    """The reverse sweep: no modal surface may be an unlisted gap.

    Fails naming any ModalScreen subclass that no registry anchor reaches
    and no skip entry covers -- exactly how the PR #2742 review found the
    eight modals the inventory missed -- and fails on stale skip entries so
    the lists stay honest.
    """
    classes = _scan_modal_classes()
    modals = sorted(name for name in classes if _is_modal(name, classes))

    uncovered = {
        name: classes[name][0]
        for name in modals
        if not _is_anchored(name, classes)
    }

    stale_class_skips = sorted(set(MODAL_WIDE_TIER_SKIPPED) - set(modals))
    assert not stale_class_skips, (
        f"skip entries name non-modal/missing classes: {stale_class_skips}"
    )

    missing = sorted(
        f"{name} ({module})"
        for name, module in uncovered.items()
        if name not in MODAL_WIDE_TIER_SKIPPED
        and module not in MODAL_WIDE_TIER_SKIPPED_MODULES
    )
    assert not missing, (
        "ModalScreen subclasses outside the wide tier and absent from both "
        "skip lists (register an anchor or document a skip):\n  "
        + "\n  ".join(missing)
    )

    stale_module_skips = sorted(
        module
        for module in MODAL_WIDE_TIER_SKIPPED_MODULES
        if not any(
            unanchored_module == module
            for unanchored_module in uncovered.values()
        )
    )
    assert not stale_module_skips, (
        f"module skip entries with no unanchored modal left: {stale_module_skips}"
    )


class WideTierHarness:  # placeholder until the mixin exists; replaced below
    """Fallback that keeps collection working pre-implementation."""


try:  # The real toggle, from the app module (heavy import, done once).
    from tldw_chatbook.app import TldwCli, WideViewportTierMixin

    class WideTierHarness(WideViewportTierMixin, ConsolidatedCSSApp[None]):  # type: ignore[no-redef]
        """Consolidated-CSS app carrying the REAL app's wide-tier toggle."""

        CSS_PATH = [str(path) for path in APP_STYLESHEETS]

        # ChatbookCreationWindow.on_mount reads self.app.config_data (the
        # real TldwCli carries one); the empty dict is the no-user-name path.
        config_data: dict = {}  # noqa: RUF012 - class-level default is the point

    def test_real_app_carries_the_wide_viewport_toggle() -> None:
        """TldwCli inherits the shared toggle, not just the test harness."""
        assert WideViewportTierMixin in TldwCli.__mro__

except ImportError:  # RED phase: mixin not shipped yet.

    def test_real_app_carries_the_wide_viewport_toggle() -> None:
        pytest.fail("WideViewportTierMixin not importable from tldw_chatbook.app")


def _build_prompts_modal(_app: object) -> ConsolePromptsModal:
    return ConsolePromptsModal(
        capabilities=lambda _source: object(),
        list_page=lambda _source, _page: [],
        search=lambda _source, _query: [],
        detail=lambda _source, _identifier: {},
        save=lambda **_payload: {},
    )


def _build_system_prompt_modal(_app: object) -> ConsoleSystemPromptModal:
    async def _save_to_library(_name: str, _text: str) -> str:
        return "saved"

    return ConsoleSystemPromptModal(
        system_prompt="You are concise.",
        save_to_library=_save_to_library,
    )


def _build_reaction_picker_modal(_app: object) -> ConsoleReactionPickerModal:
    return ConsoleReactionPickerModal(options=[])


def _build_buddy_management_modal(_app: object) -> BuddyManagementModal:
    return BuddyManagementModal()


def _build_notes_recovery_modal(_app: object) -> NotesRecoveryDialog:
    async def approve(_review: object) -> None:
        return None

    review = NotesRecoveryReview(
        "notes.sync_bindings",
        "fixture",
        Path("/tmp"),
        (("note.md", "retained"),),
        (),
        (),
    )
    return NotesRecoveryDialog(review, current=lambda: True, approve=approve)


def _build_skills_recovery_modal(_app: object) -> SkillRecoveryReviewModal:
    review = RecoveryReview("fixture", (), ("/tmp", 0, 0), (), (), (), "fixture")
    return SkillRecoveryReviewModal(review, Path("/tmp"), lambda: True)


def _build_prompt_collection_modal(_app: object) -> PromptCollectionManagerModal:
    async def load(**_kwargs: object) -> object:
        return begin_prompt_collection_catalog(query="", request_token=1)

    async def create(_name: str) -> object:
        return begin_prompt_collection_catalog(query="", request_token=2)

    async def rename(_collection_id: int, _name: str) -> object:
        return begin_prompt_collection_catalog(query="", request_token=3)

    return PromptCollectionManagerModal(
        mode="browse",
        selected_collection_id=None,
        staged_collection_ids=(),
        load_catalog=load,
        create_collection=create,
        rename_collection=rename,
    )


def _build_chatbook_creation_modal(_app: object) -> ChatbookCreationWindow:
    # Two constructor dependencies load the CLI config, which trips this
    # machine's pre-existing ``RecoveryRequired: raw_source_selection_changed``
    # profile condition when config first loads mid-test-session (see
    # .superpowers/wide-tier-rollout-report.md, environment note). Both are
    # stubbed just around __init__ and the window only stores the results:
    # get_chatbook_database_paths -> paths that do not exist (the fresh-
    # install empty-database branch on_mount reads from self.db_paths), and
    # ChatbookCreator -> None (self.creator is only touched by the create
    # button's action, never on the mount/geometry path).
    # TASK-32829 removed the constructor's ``self.app`` assignment (and its
    # ``app_instance`` parameter), so the shipped class mounts directly now;
    # the ``_MountedCreationWindow`` subclass that used to shadow the
    # read-only ``Widget.app`` property is gone.
    import tldw_chatbook.UI.ChatbookCreationWindow as creation_module

    empty_dir = Path(tempfile.mkdtemp(prefix="tldw-wide-tier-spot-"))
    real_paths = creation_module.get_chatbook_database_paths
    real_creator = creation_module.ChatbookCreator
    creation_module.get_chatbook_database_paths = lambda: {
        "ChaChaNotes": str(empty_dir / "chachanotes.db"),
        "Prompts": str(empty_dir / "prompts.db"),
        "Media": str(empty_dir / "media.db"),
    }
    creation_module.ChatbookCreator = lambda *_args, **_kwargs: None
    try:
        return ChatbookCreationWindow()
    finally:
        creation_module.get_chatbook_database_paths = real_paths
        creation_module.ChatbookCreator = real_creator


def _build_chatbook_export_modal(app: object) -> ChatbookExportManagementWindow:
    # app_instance is only stored; every mount-path read goes through
    # getattr-with-defaults, so the harness app exercises the empty-list
    # fallbacks. get_private_chatbooks_dir is stubbed around __init__ for
    # the same config-load reason as the creation window above; the empty
    # temp dir is the no-exports-yet branch.
    import tldw_chatbook.UI.ChatbookExportManagementWindow as export_module

    empty_dir = Path(tempfile.mkdtemp(prefix="tldw-wide-tier-spot-"))
    real_dir = export_module.get_private_chatbooks_dir
    export_module.get_private_chatbooks_dir = lambda: empty_dir
    try:
        return ChatbookExportManagementWindow(app)  # type: ignore[arg-type]
    finally:
        export_module.get_private_chatbooks_dir = real_dir


def _build_chatbook_templates_modal(app: object) -> ChatbookTemplatesWindow:
    return ChatbookTemplatesWindow(app)  # type: ignore[arg-type]


def _build_template_selector_modal(_app: object) -> TemplateSelectorDialog:
    # on_mount's get_eval_templates import failure is pre-existing and
    # swallowed by the widget (existing logger.error, empty template list);
    # the dialog mounts either way.
    return TemplateSelectorDialog()


#: One representative per cap tier: anchor -> (cap, base width rule).
#: Base geometry is pinned here so the tier provably leaves it untouched:
#: prompts ``90% / 104``, system prompt ``84 / 95%``, reaction ``76 / 100%``.
#: ``#buddy-management`` (PR #2742 review addition) additionally pins one of
#: the newly registered surfaces: ``78 / 96%`` base, 120 cap.
#: The wave-2 entries pin every surface that wave registered (PR #2748
#: review: committed spot-mount coverage for all of them, not just one):
#: ``#notes-recovery-dialog`` ``85% / 100`` base, 170 cap;
#: ``#skills-recovery-review`` ``90% / 110``, 170; the three Chatbook work
#: windows ``90% / 120`` and ``80% / 100``, 170;
#: ``#prompt-collection-manager`` fixed ``96``, 150;
#: ``TemplateSelectorDialog .template-selector-dialog`` (base rule shipped
#: with wave 2) ``95% / 76``, 120.
SPOT_REPRESENTATIVES: dict[str, tuple[int, tuple[int, int | None]]] = {
    WIDE_TIER_CAPS[170]: (170, (104, 90)),  # base max-width, base percent
    WIDE_TIER_CAPS[150]: (150, (84, 95)),
    WIDE_TIER_CAPS[120]: (120, (76, 100)),
    "#buddy-management": (120, (78, 96)),
    "#notes-recovery-dialog": (170, (100, 85)),
    "#skills-recovery-review": (170, (110, 90)),
    "ChatbookCreationWindow > Container": (170, (120, 90)),
    "ChatbookExportManagementWindow > Container": (170, (120, 90)),
    "ChatbookTemplatesWindow > Container": (170, (100, 80)),
    "#prompt-collection-manager": (150, (96, None)),
    "TemplateSelectorDialog .template-selector-dialog": (120, (76, 95)),
}
SPOT_BUILDERS = {
    WIDE_TIER_CAPS[170]: _build_prompts_modal,
    WIDE_TIER_CAPS[150]: _build_system_prompt_modal,
    WIDE_TIER_CAPS[120]: _build_reaction_picker_modal,
    "#buddy-management": _build_buddy_management_modal,
    "#notes-recovery-dialog": _build_notes_recovery_modal,
    "#skills-recovery-review": _build_skills_recovery_modal,
    "ChatbookCreationWindow > Container": _build_chatbook_creation_modal,
    "ChatbookExportManagementWindow > Container": _build_chatbook_export_modal,
    "ChatbookTemplatesWindow > Container": _build_chatbook_templates_modal,
    "#prompt-collection-manager": _build_prompt_collection_modal,
    "TemplateSelectorDialog .template-selector-dialog": _build_template_selector_modal,
}


def _expected_base_width(
    viewport_width: int, base_max: int, base_percent: int | None
) -> int:
    if base_percent is None:
        return base_max
    return min(base_max, viewport_width * base_percent // 100)


@pytest.mark.parametrize(
    ("anchor", "size"),
    (
        *[
            (anchor, (200, 50))
            for anchor in SPOT_REPRESENTATIVES
        ],
        *[
            (anchor, (120, 40))
            for anchor in SPOT_REPRESENTATIVES
        ],
    ),
    ids=lambda value: (
        value[1:] if isinstance(value, str) else f"{value[0]}x{value[1]}"
    ),
)
@pytest.mark.asyncio
async def test_spot_modal_geometry_per_tier(anchor: str, size: tuple[int, int]) -> None:
    """A representative per cap tier scales up wide and holds base narrow."""
    cap, (base_max, base_percent) = SPOT_REPRESENTATIVES[anchor]
    app = WideTierHarness()
    modal = SPOT_BUILDERS[anchor](app)

    async with app.run_test(size=size) as pilot:
        await app.push_screen(modal)
        await pilot.pause()
        await pilot.pause()

        container = modal.query_one(anchor)
        wide = app.size.width >= WIDE_VIEWPORT_COLUMNS
        assert app.has_class("-wide-viewport") is wide
        if wide:
            assert container.region.width == min(
                cap, app.size.width * WIDE_TIER_WIDTH_PERCENT // 100
            )
        else:
            assert container.region.width == _expected_base_width(
                app.size.width, base_max, base_percent
            )
        assert 0 < container.region.width <= app.size.width
        assert 0 < container.region.height <= app.size.height


@pytest.mark.parametrize(
    ("start_size", "end_size"),
    (((140, 40), (200, 50)), ((200, 50), (120, 40))),
    ids=["grow-past-threshold", "shrink-below-threshold"],
)
@pytest.mark.parametrize("anchor", list(SPOT_REPRESENTATIVES))
@pytest.mark.asyncio
async def test_spot_modal_wide_tier_tracks_live_resize(
    anchor: str, start_size: tuple[int, int], end_size: tuple[int, int]
) -> None:
    """An open modal re-syncs its tier when the viewport crosses the breakpoint."""
    cap, (base_max, base_percent) = SPOT_REPRESENTATIVES[anchor]
    app = WideTierHarness()
    modal = SPOT_BUILDERS[anchor](app)

    async with app.run_test(size=start_size) as pilot:
        await app.push_screen(modal)
        await pilot.pause()
        await pilot.pause()

        container = modal.query_one(anchor)
        assert app.has_class("-wide-viewport") is (
            start_size[0] >= WIDE_VIEWPORT_COLUMNS
        )

        await pilot.resize_terminal(*end_size)
        await pilot.pause()
        await pilot.pause()

        wide_after = end_size[0] >= WIDE_VIEWPORT_COLUMNS
        assert app.has_class("-wide-viewport") is wide_after
        if wide_after:
            assert container.region.width == min(
                cap, end_size[0] * WIDE_TIER_WIDTH_PERCENT // 100
            )
        else:
            assert container.region.width == _expected_base_width(
                end_size[0], base_max, base_percent
            )
