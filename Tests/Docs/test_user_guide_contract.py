"""User Guide pages agree with the shipped nav map and carry self-contained images.

TASK-38969: the index's screen table and F-key lists stopped at F7 after F8
Plugins and F9 Kanban shipped, and two captures still fetched fonts from
cdnjs. TASK-38970: two pages lost their screenshots; every image link must
resolve. TASK-40329: the README's destinations table drifted the same way,
out of reach of the guide-only checks. TASK-41101: pages carry no
"Verified against" stamps; that record belongs in a task's notes.
"""

from __future__ import annotations

import re
from pathlib import Path, PurePath, PureWindowsPath

import pytest

from Tests.Docs.test_console_library_controls_docs import missing_local_links
from tldw_chatbook.UI.Navigation.shell_destinations import (
    ARTIFACTS_COMPATIBILITY_SHORTCUT,
    MCP_COMPATIBILITY_SHORTCUT,
    SHELL_DESTINATION_SHORTCUTS,
    get_shell_destination,
    resolve_shell_route,
)

GUIDE = Path(__file__).resolve().parents[2] / "Docs" / "User_Guide"
INDEX = GUIDE / "index.md"
README = GUIDE.parents[1] / "README.md"
README_KEY_RE = re.compile(r"\*\*(Ctrl\+\d|F\d+)\*\*")
IMAGE_LINK_RE = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")
TABLE_KEY_RE = re.compile(r"Ctrl\+\d|F\d+")
LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)#]+)")
# A run of F-keys joined only by list separators: "F2, F3", "F2–F5 / F7",
# "F2-F5/F7", "F5, then F7". Ranges (en dash or hyphen) are expanded;
# "Shift+F6" never starts one. Runs of fewer than three keys ("F1, F6",
# finding ids like "F1/F2") are not destination lists.
_SEP = r"\s*(?:,|/|–|-|\band\b|\bor\b|\bthen\b)"
F_KEY_RUN_RE = re.compile(rf"(?<![+\w])F\d+(?:(?:{_SEP})+\s*F\d+)+\b")
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


def _readme_destination_table() -> dict[str, str]:
    """Map each row of the README's "Application destinations" table to its destination cell."""
    section = README.read_text(encoding="utf-8").split("## Application destinations", 1)[1]
    rows: dict[str, str] = {}
    for line in section.split("\n## ", 1)[0].splitlines():
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) >= 2 and (key := README_KEY_RE.fullmatch(cells[0])):
            rows[key.group(1).lower()] = cells[1]
    return rows


def test_readme_destination_table_matches_the_shell_shortcut_map() -> None:
    table = _readme_destination_table()
    expected = {
        key: get_shell_destination(destination_id).label
        for destination_id, key in SHELL_DESTINATION_SHORTCUTS.items()
    }
    # The compatibility keys land in another destination; the cell names it first.
    for key, route in (
        (ARTIFACTS_COMPATIBILITY_SHORTCUT, "artifacts"),
        (MCP_COMPATIBILITY_SHORTCUT, "mcp"),
    ):
        expected[key] = get_shell_destination(resolve_shell_route(route).destination_id).label
    wrong = {
        key: (label, table.get(key))
        for key, label in expected.items()
        if (table.get(key) or "").split(" ▸ ")[0] != label
    }
    assert set(table) == set(expected) and not wrong, (
        f"README destinations table differs from the shipped map: expected/found "
        f"{wrong}, extra rows {sorted(set(table) - set(expected))}"
    )


def test_readme_prose_hotkeys_match_the_shell_shortcut_map() -> None:
    shortcut = {
        get_shell_destination(destination_id).label: key
        for destination_id, key in SHELL_DESTINATION_SHORTCUTS.items()
    }
    prose = " ".join(README.read_text(encoding="utf-8").split())
    mentions = re.findall(r"\*\*(\w+)\*\* with \*\*(Ctrl\+\d|F\d+)\*\*", prose)
    assert mentions, "the README's 'open X with KEY' sentences were not found"
    wrong = [(label, key) for label, key in mentions if shortcut.get(label) != key.lower()]
    assert not wrong, f"README names the wrong key for a destination: {wrong}"


def _f_keys(run: str) -> list[str]:
    """Expand a run such as "F2–F5 / F7" to F2, F3, F4, F5, F7."""
    keys: list[str] = []
    for part in re.split(r"(?:\s*(?:,|/|\band\b|\bor\b|\bthen\b)\s*)+", run):
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


# A verification stamp: "Verified against <ref> — <date>", "Re-stamped
# against", "verified on <branch> from <sha>", "checked against <sha> on
# <date>": a date, an @sha or a bare sha follows within 200 characters of the
# same sentence (stamps wrap across lines, so pages are matched with newlines
# flattened). "Re-verified live <date>" and a line opening "*Verified" are
# stamps too. Prose such as "is verified against a live call" is not.
_STAMP_REF = r"\b20\d\d-\d\d-\d\d\b|@\s*[0-9a-f]{7,}|\b(?=[0-9a-f]*\d)[0-9a-f]{7,40}\b"
STAMP_RE = re.compile(
    r"\b(?:re-?)?(?:verified|stamped|checked)(?:\s+live)?\s+(?:against|on)\b"
    rf"(?=(?:(?![.;]\s).){{0,200}}?(?:{_STAMP_REF}))"
    r"|\b(?:re-?)?verified\s+live\s+20\d\d-\d\d-\d\d\b",
    re.IGNORECASE,
)
ITALIC_STAMP_RE = re.compile(r"^\*(?:re-?)?verified\b", re.IGNORECASE | re.MULTILINE)
#: Pages whose stamps other tasks remove: TASK-34100.15 AC#7 and
#: TASK-33910.18 AC#4. Drop a name here when its task lands.
STAMP_EXEMPT_PAGES = frozenset({
    "First_Run_Setup.md",
    "roleplay-chat-dictionaries.md",
    "roleplay-chat-dictionaries/characters-and-personas.md",
    "roleplay-chat-dictionaries/chat-dictionaries.md",
    "roleplay-chat-dictionaries/lore-books.md",
})


def _stamp_lines(page: Path) -> list[int]:
    text = page.read_text(encoding="utf-8")
    flat = text.replace("\n", " ")
    starts = [m.start() for m in STAMP_RE.finditer(flat)]
    starts += [m.start() for m in ITALIC_STAMP_RE.finditer(text)]
    return sorted({text.count("\n", 0, start) + 1 for start in starts})


def _guide_name(page: PurePath, guide: PurePath = GUIDE) -> str:
    """A page's name as STAMP_EXEMPT_PAGES spells it, "/" on every OS."""
    return page.relative_to(guide).as_posix()


def test_user_guide_pages_carry_no_verified_against_stamps() -> None:
    """CLAUDE.md and _template.md: no stamps on a page (TASK-41101)."""
    pages = {_guide_name(page): page for page in sorted(GUIDE.rglob("*.md"))}
    stamped = {
        name: lines
        for name, page in pages.items()
        if name not in STAMP_EXEMPT_PAGES and (lines := _stamp_lines(page))
    }
    assert not stamped, (
        "User Guide stamps (record verification in the task's Implementation "
        f"Notes instead), page: lines: {stamped}"
    )
    landed = sorted(
        name for name in STAMP_EXEMPT_PAGES if not _stamp_lines(pages[name])
    )
    assert not landed, f"Stamp-free now; remove from STAMP_EXEMPT_PAGES: {landed}"


def test_stamp_exemptions_match_windows_page_names() -> None:
    """A Windows path names a Roleplay sub-page with "/" too (r15doc-rev1)."""
    guide = PureWindowsPath(r"C:\repo\Docs\User_Guide")
    page = guide / "roleplay-chat-dictionaries" / "lore-books.md"
    assert _guide_name(page, guide) in STAMP_EXEMPT_PAGES


@pytest.mark.parametrize(
    ("text", "stamped"),
    [
        ("*Verified against dev @ 0123456789 — 2026-11-15 (planted).*", True),
        ("*Verified\nagainst dev\n@ abcdef1 — 2026-09-17*", True),
        ("Re-verified live 2026-08-11 with the server.", True),
        ("*Verified against dev 0123456789 (TASK-1).*", True),
        ("*Verified against `fix/foo` (task-123: something).*", True),
        ("The key is verified against a live call.", False),
        ("Each file is verified against that hash. Its id 0123456789 stays.", False),
        ("It is checked against the allowlist; a reminder due 2026-01-05 fires.", False),
    ],
)
def test_stamp_pattern_tells_stamps_from_prose(text: str, stamped: bool) -> None:
    """Review r15doc-rev1: a bare sha, "verified live <date>" and an italic
    "*Verified ..." line are stamps too; prose that verifies is not."""
    found = bool(STAMP_RE.search(text.replace("\n", " ")) or ITALIC_STAMP_RE.search(text))
    assert found is stamped


async def test_capture_driver_builds_its_gateway_on_the_uat_planner() -> None:
    """A refactor of the mounted UAT's planner that breaks the driver fails here."""
    from scripts.capture_user_guide import guide_gateway

    gateway = guide_gateway(["https://public.example/feed-1.xml"], None)
    first = [chunk async for chunk in gateway.stream_chat(None, [{"content": ""}])]
    assert "find_tools" in "".join(first) and gateway.stage == 1
    assert gateway.cached_context_window(None) is None
