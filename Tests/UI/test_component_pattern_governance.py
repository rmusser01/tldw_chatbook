"""Component-pattern governance (ADR-161, spec 3.7).

Mechanical enforcement of the component-pattern layer, reading the one
registry artifact ``css/patterns.json`` (task 3) and importing
``CSS_MODULES`` from ``build_css.py`` as the single source of truth for
bundle membership:

1. **Single-definition rule** — a rule whose selector is exactly a
   ``Canonical`` class (optionally with state pseudo-selectors, which
   belong to the owner per the spec 2.7 contract) may exist only in that
   class's owning sheet. Compound/scoped selectors using the class are
   composition and are legal anywhere.
2. **Deprecated-name ratchet** — Python use sites of ``Deprecated`` names
   may only decrease below their registry ceiling.
3. **Sync checks** — every ``Canonical`` class appears in the catalog doc
   (``backlog/docs/component-patterns.md``) and in the pattern gallery
   (``tldw_chatbook/Widgets/pattern_gallery.py``).
4. **Fallback-ban ratchet** — per-screen bundled CSS blocks may not add
   new local ``$ds-*`` fallback definitions (the TASK-15993 trap); the
   blocks that already carried them when this test landed are
   grandfathered in ``FALLBACK_ALLOWED`` (empty at pinning time).
5. **Dimension-literal floor (hard zero)** — no hand-written sheet may
   carry a raw numeric ``padding``/``margin``/``width``/``height``
   declaration. ADR-161 task 11 completed the migration (3,514 -> 0), so
   the former baseline ratchet is now a flat ban. ``core/_variables.tcss``
   is exempt: raw values are legal ONLY in token definitions by design
   (ADR-150's "raw hex legal only in token definitions", same principle
   for dimensions) — the counting regex is name-blind and would otherwise
   match ``height: 3`` inside ``$ds-control-height: 3;``.
6. **Python ad-hoc style floor (hard zero)** — a shared AST inventory
   covers assignments, set_styles and setattr, including runtime exceptions.
   Only documented nonliteral runtime values and None resets are exempt.

The dimension regex matches ``pin_pattern_ratchets.py``; the Python checker
is imported from the same pure helper by governance and pinning.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import tldw_chatbook
from Tests.UI.python_style_inventory import inventory_styles
from tldw_chatbook.css.build_css import CSS_MODULES
from tldw_chatbook.css.widget_css import (
    SCREEN_ATTR,
    WIDGET_ATTR,
    iter_blocks,
    split_scoped_css,
)

ROOT = Path(__file__).resolve().parents[2]
CSS = ROOT / "tldw_chatbook/css"
PKG = Path(tldw_chatbook.__file__).parent
REGISTRY = json.loads((CSS / "patterns.json").read_text(encoding="utf-8"))
CATALOG = ROOT / "backlog/docs/component-patterns.md"
GALLERY = PKG / "Widgets/pattern_gallery.py"

_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)
_RULE = re.compile(r"([^{}]+)\{([^{}]*)\}")
_DIM = re.compile(
    r"\b(?:padding|margin|width|height)(?:-(?:top|right|bottom|left))?\s*:\s*(?:[^;{}\s]+\s+)*[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)"
)

#: A local ``$ds-*:`` definition inside a bundled screen block -- the
#: TASK-15993 trap this test bans going forward.
_DS_FALLBACK = re.compile(r"\$ds-[a-z0-9-]+\s*:")

#: Per-screen bundled blocks that already carried a local ``$ds-*``
#: fallback when this test landed (2026-09-13). The scan that produced the
#: manifest found none -- ``isolate_local_variables`` had already inlined
#: the historical offenders -- so new blocks start from a flat ban.
FALLBACK_ALLOWED: frozenset[str] = frozenset()


def _sheet_sources() -> dict[str, str]:
    """Comment-stripped text of every hand-written source sheet, by its
    path relative to ``tldw_chatbook/css``."""
    out: dict[str, str] = {}
    for sub in ("core", "layout", "components", "features", "utilities"):
        for p in sorted((CSS / sub).glob("*.tcss")):
            out[str(p.relative_to(CSS))] = _COMMENT.sub(
                "", p.read_text(encoding="utf-8")
            )
    return out


def _bundled_sources() -> dict[str, str]:
    """Bundled blocks in their **emitted** form, comment-stripped.

    The build lifts each ``BUNDLED_CSS``/``BUNDLED_SCREEN_CSS`` block into
    a generated sheet with every top-level selector prefixed by the
    declaring class's type name (``scope_every_selector=True`` since
    TASK-15450/15998), so a bare ``.foo`` written in a class body reaches
    the app as ``ClassName .foo`` -- composition, not a definition.  The
    governance question is what the cascade actually sees, so the scan
    applies the same scoping via ``split_scoped_css`` rather than reading
    the raw class-body text (which would flag eight false positives, e.g.
    Chatbooks' ``.section-header``).
    """
    out: dict[str, str] = {}
    for attr in (WIDGET_ATTR, SCREEN_ATTR):
        for block in iter_blocks(PKG, attr):
            self_css, scoped_css = split_scoped_css(
                block.css, block.class_name, scope_every_selector=True
            )
            key = f"{block.module}:{block.class_name}:{attr}"
            out[key] = _COMMENT.sub("", self_css + scoped_css)
    return out


def _bundled_raw_screen_sources() -> dict[str, str]:
    """Raw class-body text of the per-screen blocks (fallback scan).

    Unlike the emitted form, this preserves the block's own top-level
    ``$ds-*:`` fallback declarations, which the build inlines and strips
    -- exactly what the fallback ban needs to see.
    """
    out: dict[str, str] = {}
    for block in iter_blocks(PKG, SCREEN_ATTR):
        key = f"{block.module}:{block.class_name}:{SCREEN_ATTR}"
        out[key] = block.css
    return out


def _bare_definitions(css: str) -> set[str]:
    """Classes defined by a bare selector (one class token + pseudo suffixes)."""
    found: set[str] = set()
    for m in _RULE.finditer(css):
        for part in m.group(1).split(","):
            sel = part.strip()
            if re.fullmatch(r"\.[A-Za-z_][\w-]*(?::[a-zA-Z-]+)*", sel):
                found.add(sel.split(":")[0][1:])
    return found


def _canonical() -> dict[str, str]:
    """Canonical class -> owning sheet, honoring per-class ``owning_sheet``
    overrides (e.g. ``nav-button`` -> ``components/_navigation.tcss``).

    Draft classes (``ds_primitives`` until task 9 flips them) contribute
    nothing -- they are not yet canonical and must not be flagged.
    """
    out: dict[str, str] = {}
    for spec_ in REGISTRY["families"].values():
        for cls, meta in spec_["classes"].items():
            if meta["status"] == "canonical":
                out[cls] = meta.get("owning_sheet", spec_["owning_sheet"])
    return out


def test_canonical_classes_defined_only_in_owning_sheet() -> None:
    """A bare Canonical-class rule may exist only in its owning sheet."""
    canonical = _canonical()
    sources = {**_sheet_sources(), **_bundled_sources()}
    offenders: dict[str, list[str]] = {}
    for cls, owner in canonical.items():
        for src, css in sources.items():
            if cls in _bare_definitions(css) and src != owner:
                offenders.setdefault(cls, []).append(src)
    assert not offenders, (
        "Canonical classes redefined outside their owning sheet "
        f"(ADR-161 spec 3.7; scope or fold per the migration rule): {offenders}"
    )


def test_canonical_owning_sheets_are_bundled() -> None:
    """Every sheet that owns a canonical class must be a bundle member.

    ``CSS_MODULES`` is the single source of truth for bundle membership;
    an owning sheet that drifted out of the manifest would style nothing.
    Draft-only families (``ds_primitives`` until task 9) are exempt.
    """
    owners = sorted(set(_canonical().values()))
    missing = [sheet for sheet in owners if sheet not in CSS_MODULES]
    assert not missing, (
        f"Owning sheets missing from CSS_MODULES (build_css.py): {missing}"
    )


def test_deprecated_names_ratchet_down() -> None:
    """Python and sheet use sites of Deprecated names may only decrease.

    Counted with CSS-token boundaries (``(?<![\\w-])name(?![\\w-])``), not
    regex word boundaries: live prefixed variant classes such as
    ``speech-setting-label`` / ``library-prompt-field-label`` contain the
    deprecated names as hyphen-suffixes, and ``\\b`` matches at a hyphen --
    those variants would count against the ceiling (ADR-161 task 7 pinned
    ``field-label``'s word-boundary count at ~76 of pure variant hits) and
    hand the ratchet 76 units of false headroom. Token semantics match the
    integrity test's relocated-class matcher.

    ADR-161 task 8 (Task 7 review, pre-step 3): the scan also covers the
    hand-written ``.tcss`` sources, comment-stripped, with the same token
    boundaries and the same ceilings -- a renamed class must not survive as
    a stylesheet definition any more than as a Python compose site (comment
    mentions stay legal; ``_sheet_sources`` strips comments).
    """
    for cls, meta in REGISTRY["deprecated"].items():
        count = 0
        for p in PKG.rglob("*.py"):
            count += len(
                re.findall(
                    rf"(?<![\w-]){re.escape(cls)}(?![\w-])",
                    p.read_text(encoding="utf-8", errors="ignore"),
                )
            )
        ceiling = meta["ceiling"]
        assert count <= ceiling, f"{cls}: {count} Python uses > ceiling {ceiling}"
        sheet_count = 0
        offenders: list[str] = []
        for src, css in _sheet_sources().items():
            hits = len(re.findall(rf"(?<![\w-]){re.escape(cls)}(?![\w-])", css))
            if hits:
                offenders.append(f"{src}x{hits}")
            sheet_count += hits
        assert sheet_count <= ceiling, (
            f"{cls}: {sheet_count} comment-stripped .tcss uses > ceiling "
            f"{ceiling} (offenders: {offenders})"
        )


def test_catalog_and_gallery_sync() -> None:
    """Every canonical class is documented and rendered somewhere."""
    doc = CATALOG.read_text(encoding="utf-8")
    gallery = GALLERY.read_text(encoding="utf-8")
    for cls in _canonical():
        assert cls in doc, f"{cls} missing from catalog doc"
        assert cls in gallery, f"{cls} missing from pattern gallery"


def test_no_new_local_ds_fallbacks() -> None:
    """Per-screen bundled blocks may not add local $ds-* fallbacks."""
    for key, css in _bundled_raw_screen_sources().items():
        if not key.endswith(SCREEN_ATTR):
            continue
        module = key.split(":")[0]
        if _DS_FALLBACK.search(css):
            assert module in FALLBACK_ALLOWED, (
                f"{module} adds a local $ds-* fallback (TASK-15993 trap); "
                "rely on the app variables instead."
            )


#: The one sheet where raw dimension VALUES are legal: token definitions.
#: Raw values live only here by design (ADR-150's raw-hex rule, same
#: principle for dimensions); the counting regex is name-blind and would
#: otherwise match ``height: 3`` inside ``$ds-control-height: 3;``.
TOKENS_SHEET = "core/_variables.tcss"


def test_dimension_literal_ratchet() -> None:
    """The dimension floor is a HARD ZERO (ADR-161 task 11 close-out).

    Every baseline entry reached zero during the task-11 migration, so the
    baseline-pinned allowance ratchet was replaced by a flat ban: no
    hand-written sheet outside the tokens file may carry a raw numeric
    padding/margin/width/height declaration. ``core/_variables.tcss`` alone
    is exempt -- mirroring the hex ratchet's tokens-file exemption in
    ``test_design_token_governance.py``.
    """
    offenders: list[str] = []
    for sheet, css in _sheet_sources().items():
        if sheet == TOKENS_SHEET:
            continue
        count = len(_DIM.findall(css))
        if count:
            offenders.append(f"{sheet}: {count}")
    assert not offenders, (
        "Raw numeric dimension literals remain (ADR-161 spec 3.10: the "
        "floor is hard zero; raw values are legal only in token "
        "definitions, which live in core/_variables.tcss):\n  " + "\n  ".join(offenders)
    )


def test_python_style_ratchet() -> None:
    """No token-covered static or unmarked Python visual writes remain."""
    offenders: list[str] = []
    for path in sorted(PKG.rglob("*.py")):
        relative = path.relative_to(PKG)
        for write in inventory_styles(path.read_text(encoding="utf-8")):
            if write.violation:
                offenders.append(
                    f"{relative}:{write.line}: {write.form} {write.property} "
                    f"({write.value_kind})"
                )
    assert not offenders, (
        "Ad-hoc Python visual writes remain (ADR-161 spec 3.10, HARD ZERO). "
        "Compose token-backed classes. Only nonliteral runtime geometry/user "
        "previews with an adjacent '# ds-runtime: <specific reason>' and "
        "None resets are exempt:\n  " + "\n  ".join(offenders)
    )


#: Spec 3.5's size ceiling for hand-authored source sheets. Load-bearing
#: comment essays are not punished (comments stripped before counting), so
#: the ceiling measures the rules a sheet carries, not its documentation.
#: Data at the pin: next-largest sheet after the task-10 carve is
#: features/_wizards.tcss at ~1,090 active lines -- the ceiling has real
#: headroom by design, it exists to stop the next _agentic_terminal.tcss
#: (12,240 total lines, ~8,700 active, TASK-24451).
SHEET_ACTIVE_LINE_CEILING = 2_000


def test_sheet_size_ceiling() -> None:
    """No CSS_MODULES sheet exceeds 2,000 comment-stripped active lines.

    TASK-24451 close-out (ADR-161 spec 3.5): the ceiling ships in the same
    PR that completed the agentic-terminal carve-up -- no temporary
    allowance that can outlive the project. A sheet over the ceiling must
    split by vocabulary, exactly as the task-10 carve did.
    """
    for rel in CSS_MODULES:
        source = CSS / rel
        assert source.is_file(), f"{rel} missing from css/"
        text = _COMMENT.sub("", source.read_text(encoding="utf-8"))
        lines = [ln for ln in text.splitlines() if ln.strip()]
        assert len(lines) <= SHEET_ACTIVE_LINE_CEILING, (
            f"{rel}: {len(lines)} active lines > "
            f"{SHEET_ACTIVE_LINE_CEILING} (spec 3.5 size ceiling, "
            "TASK-24451). Split the sheet by vocabulary."
        )


def test_ratchet_regexes_match_pin_script() -> None:
    """The counting regexes must stay byte-identical to the pin script's."""
    pin_source = (Path(__file__).parent / "pin_pattern_ratchets.py").read_text(
        encoding="utf-8"
    )
    for pattern in (_DIM.pattern,):
        assert pattern in pin_source, (
            f"Ratchet regex drifted from pin_pattern_ratchets.py: {pattern!r}. "
            "Update both files together or the baseline stops matching the test."
        )


@pytest.mark.parametrize(
    "declaration",
    [
        "padding: $ds-space-1 0 0 0;",
        "margin: $ds-space-stack 0;",
        "margin: $ds-space-0 -2;",
        "padding: $ds-space-1 .5;",
        "width: -1.5;",
        "height: +.25;",
    ],
)
def test_dimension_floor_and_pin_reject_every_numeric_slot(
    declaration, tmp_path, monkeypatch
):
    """Mixed shorthand and signed/decimal literals cannot bypass either floor."""
    source = "Widget { " + declaration + " }"
    monkeypatch.setattr(
        sys.modules[__name__],
        "_sheet_sources",
        lambda: {"features/_probe.tcss": source},
    )
    with pytest.raises(AssertionError, match="Raw numeric dimension literals"):
        test_dimension_literal_ratchet()

    test_dir = tmp_path / "Tests" / "UI"
    test_dir.mkdir(parents=True)
    for name in ("pin_pattern_ratchets.py", "python_style_inventory.py"):
        shutil.copy(Path(__file__).parent / name, test_dir / name)
    sheet = tmp_path / "tldw_chatbook" / "css" / "features" / "_probe.tcss"
    sheet.parent.mkdir(parents=True)
    sheet.write_text(source)
    result = subprocess.run(
        [sys.executable, str(test_dir / "pin_pattern_ratchets.py")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "refusing to pin: raw numeric dimension literals" in result.stderr
    assert not (test_dir / "pattern_ratchet_baseline.json").exists()


def test_dimension_floor_ignores_tokens_and_token_definitions(monkeypatch):
    monkeypatch.setattr(
        sys.modules[__name__],
        "_sheet_sources",
        lambda: {
            "core/_variables.tcss": "$ds-size-1: 1; $ds-space-half: .5;",
            "features/_probe.tcss": (
                "Widget { padding: $ds-space-1 $ds-space-0; "
                "width: auto; height: $ds-height-fill; }"
            ),
        },
    )
    test_dimension_literal_ratchet()
