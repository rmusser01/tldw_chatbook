"""User Guide pages agree with the shipped nav map and carry self-contained images.

TASK-38969: the index's screen table and F-key lists stopped at F7 after F8
Plugins and F9 Kanban shipped, and two captures still fetched fonts from
cdnjs. TASK-38970: two pages lost their screenshots; every image link must
resolve.
"""

from __future__ import annotations

import re
from pathlib import Path

from Tests.Docs.test_console_library_controls_docs import missing_local_links
from tldw_chatbook.UI.Navigation.shell_destinations import (
    ARTIFACTS_COMPATIBILITY_SHORTCUT,
    MCP_COMPATIBILITY_SHORTCUT,
    SHELL_DESTINATION_SHORTCUTS,
    get_shell_destination,
)

GUIDE = Path(__file__).resolve().parents[2] / "Docs" / "User_Guide"
INDEX = GUIDE / "index.md"
IMAGE_LINK_RE = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")
TABLE_KEY_RE = re.compile(r"Ctrl\+\d|F\d+")
LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)#]+)")
# A run of two or more F-keys joined only by list separators: "F2, F3",
# "F2–F5 / F7", "F2 / F3 / F4". Ranges ("F2–F5") are expanded.
F_KEY_RUN_RE = re.compile(r"\bF\d+(?:\s*(?:,|/|–|and|or)\s*F\d+)+\b")


def _nav_table() -> dict[str, str]:
    """Map each hotkey row of the index's "The screens" table to its screen cell."""
    section = INDEX.read_text(encoding="utf-8").split("## The screens", 1)[1]
    rows: dict[str, str] = {}
    for line in section.split("\n## ", 1)[0].splitlines():
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) >= 2 and TABLE_KEY_RE.fullmatch(cells[0]):
            rows[cells[0].lower()] = cells[1]
    return rows


def test_index_nav_table_matches_the_shell_shortcut_map() -> None:
    table = _nav_table()
    expected = dict(SHELL_DESTINATION_SHORTCUTS)
    shipped_keys = set(expected.values()) | {
        ARTIFACTS_COMPATIBILITY_SHORTCUT,
        MCP_COMPATIBILITY_SHORTCUT,
    }
    assert set(table) == shipped_keys, (
        f"index.md nav table keys differ from the shipped map: missing "
        f"{sorted(shipped_keys - set(table))}, extra {sorted(set(table) - shipped_keys)}"
    )
    for destination_id, key in expected.items():
        label = get_shell_destination(destination_id).label
        assert label in table[key], f"{key} row should name {label}: {table[key]!r}"
    for key, cell in table.items():
        links = LINK_RE.findall(cell)
        assert links, f"{key} row has no page link: {cell!r}"
        for _text, target in links:
            assert (INDEX.parent / target).is_file(), f"{key} row links to {target}"


def test_every_f_key_list_on_the_index_names_the_whole_f_row() -> None:
    shipped = sorted(
        (key.upper() for key in SHELL_DESTINATION_SHORTCUTS.values() if key[0] == "f"),
        key=lambda key: int(key[1:]),
    )
    prose = re.sub(r"[*`]", "", INDEX.read_text(encoding="utf-8"))
    runs = F_KEY_RUN_RE.findall(prose)
    assert len(runs) >= 4, f"expected the index's four F-key lists, found {runs}"
    for run in runs:
        keys: list[str] = []
        for part in re.split(r"\s*(?:,|/|and|or)\s*", run):
            first, _, last = part.partition("–")
            low, high = int(first[1:]), int((last or first)[1:])
            keys += [f"F{n}" for n in range(low, high + 1)]
        assert keys == shipped, f"index.md F-key list {run!r} should cover {shipped}"


def test_user_guide_svgs_load_no_remote_fonts() -> None:
    remote = sorted(
        str(svg.relative_to(GUIDE))
        for svg in GUIDE.rglob("*.svg")
        if "cdnjs.cloudflare.com" in svg.read_text(encoding="utf-8")
    )
    assert not remote, f"User Guide SVGs still fetch cdnjs fonts: {remote}"


def test_every_user_guide_image_link_resolves() -> None:
    missing = {
        str(page.relative_to(GUIDE)): broken
        for page in sorted(GUIDE.rglob("*.md"))
        if (broken := missing_local_links(page, IMAGE_LINK_RE))
    }
    assert not missing, f"User Guide image links name missing files: {missing}"
