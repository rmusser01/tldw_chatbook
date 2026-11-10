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
# A run of F-keys joined only by list separators: "F2, F3", "F2–F5 / F7",
# "F2-F5/F7". Ranges (en dash or hyphen) are expanded; "Shift+F6" never starts
# one. Runs of fewer than three keys ("F1, F6", finding ids like "F1/F2") are
# not destination lists.
F_KEY_RUN_RE = re.compile(r"(?<![+\w])F\d+(?:\s*(?:,|/|–|-|and|or)\s*F\d+)+\b")
# Two or more quoted nav labels: "F2 Lab", "F3 Logs", ...
QUOTED_RUN_RE = re.compile(r'"F\d+ [A-Z]\w*"(?:,\s*(?:and\s+)?"F\d+ [A-Z]\w*")+')


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


def _f_keys(run: str) -> list[str]:
    """Expand a run such as "F2–F5 / F7" to F2, F3, F4, F5, F7."""
    keys: list[str] = []
    for part in re.split(r"\s*(?:,|/|and|or)\s*", run):
        ends = [int(key[1:]) for key in re.findall(r"F\d+", part)]
        keys += [f"F{n}" for n in range(ends[0], ends[-1] + 1)]
    return keys


def test_every_f_key_list_in_the_guide_names_the_whole_f_row() -> None:
    shipped = sorted(
        (key for key in SHELL_DESTINATION_SHORTCUTS.values() if key[0] == "f"),
        key=lambda key: int(key[1:]),
    )
    keys = [key.upper() for key in shipped]
    owner = {key: dest for dest, key in SHELL_DESTINATION_SHORTCUTS.items()}
    labels = [
        f'"{key} {get_shell_destination(owner[key.lower()]).label}"' for key in keys
    ]
    stale: list[str] = []
    index_runs = 0
    for page in sorted(GUIDE.rglob("*.md")):
        prose = " ".join(re.sub(r"[*`]", "", page.read_text(encoding="utf-8")).split())
        name = str(page.relative_to(GUIDE))
        for run in F_KEY_RUN_RE.findall(prose):
            expanded = _f_keys(run)
            if len(expanded) < 3 or not set(expanded) & set(keys):
                continue
            index_runs += page == INDEX
            if expanded != keys:
                stale.append(f"{name}: {run!r}")
        for run in QUOTED_RUN_RE.findall(prose):
            index_runs += page == INDEX
            if re.findall(r'"[^"]+"', run) != labels:
                stale.append(f"{name}: {run!r}")
    assert index_runs >= 5, "the index's F-key lists were not found"
    assert not stale, f"F-key lists that differ from {keys}: {stale}"


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


async def test_capture_driver_builds_its_gateway_on_the_uat_planner() -> None:
    """A refactor of the mounted UAT's planner that breaks the driver fails here."""
    from scripts.capture_user_guide import guide_gateway

    gateway = guide_gateway(["https://public.example/feed-1.xml"], None)
    first = [chunk async for chunk in gateway.stream_chat(None, [{"content": ""}])]
    assert "find_tools" in "".join(first) and gateway.stage == 1
    assert gateway.cached_context_window(None) is None
