#!/usr/bin/env python3
"""Report guide sentences that quote a UI string no source file emits.

task-32558. A user-guide sentence that quotes what the app prints is
falsifiable in one command, and nothing in this repo was running that
command. The wave-4 sweep ran it by hand over five pages and found **nine**
false claims in roughly three hundred -- on pages already carrying dozens of
"Verified against" stamps, two of which were live walks of the very panel
whose heading, scope line and keyboard guide had all three been rewritten a
month earlier (commit 67fec3f350, 2026-08-11) and never re-read.

The failure mode is specific and worth naming: a stamp verifies the claim it
names, not the chapter it sits in. So a false sentence can survive any number
of honest verifications while a `grep -rF` of its own quoted string returns
nothing at all.

This is the tool that finds them, not yet the gate. It PRINTS candidates and
always exits 0 by default: the output is a read-list, not a verdict, because
a guide legitimately quotes strings no source emits -- historical "(Was ...)"
clauses, composed examples, the reader's own input. Turning the judgement
into a pass/fail check needs an allowlist of the reviewed exceptions, in the
idiom `EXPECTED_CHACHANOTES_INDEXES` and `REVIEWED_METADATA_ONLY_DIAGNOSTICS`
already use here. Pass ``--fail-on-miss`` to get a non-zero exit once such a
list exists.

Usage::

    python scripts/check_guide_claim_strings.py Docs/User_Guide/library/notes.md
    python scripts/check_guide_claim_strings.py --end-line 1223 Docs/...
    python scripts/check_guide_claim_strings.py --self-test
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SOURCE_ROOT = REPO_ROOT / "tldw_chatbook"

#: Extensions worth grepping. A UI string lives in Python or in a stylesheet's
#: generated content; everything else is noise that slows the scan down.
SOURCE_GLOBS = ("*.py", "*.tcss", "*.json")

#: Quoted fragments that are never UI copy.
SKIP_PREFIXES = ("http", "../", "./", "`")

#: A literal run shorter than this matches half the tree and proves nothing.
MIN_LITERAL_RUN = 12

#: Split points for a composed line: interpolation braces, placeholder angle
#: brackets, digit runs, and the two ellipsis spellings.
INTERPOLATION = re.compile(r"[<>{}]|\b\d[\d,]*\b|…|\.\.\.")


def quoted_fragments(text: str) -> list[tuple[int, str]]:
    """Return ``(line_number, fragment)`` for every quoted or bolded run.

    Both forms are collected because guides quote UI copy either way: a
    status line goes in double quotes, a button label in ``**bold**``.
    """
    found: list[tuple[int, str]] = []
    for pattern in (r'"([^"\n]{3,200})"', r"\*\*([^*\n]{3,120})\*\*"):
        for match in re.finditer(pattern, text):
            line = text.count("\n", 0, match.start()) + 1
            found.append((line, match.group(1)))
    return found


def normalise(fragment: str) -> str:
    """Undo the guide's markdown escaping and collapse whitespace.

    Guides write ``\\<name\\>`` and ``\\|`` to survive markdown and tables;
    the app emits neither backslash. A quoted string also wraps across source
    lines, so internal runs of whitespace are not comparable.
    """
    for escaped, plain in (("\\<", "<"), ("\\>", ">"), ("\\|", "|")):
        fragment = fragment.replace(escaped, plain)
    return re.sub(r"\s+", " ", fragment).strip()


def emitted_somewhere(fragment: str, source_root: Path = SOURCE_ROOT) -> bool:
    """True when some source file contains ``fragment`` literally.

    ponytail: this probe is REPO-WIDE and RAW, and it needs to be BOTH
    surface-scoped AND literal-only before it can falsify a claim like
    "press Export… **in Notes**". Neither half alone is enough. Measured on
    that exact incident -- the Notes toolbar ships a bare ``"Export"``
    (``library_notes_canvas.py:1842``) while the guide said ``"Export…"`` --
    with "scoped" meaning the 12 ``library_notes*`` modules and "AST"
    meaning non-docstring ``ast.Constant`` string values:

        probe                "Export…"   "Export"
        repo-wide + raw            35       1431   <- what runs today
        repo-wide + AST             9        373   <- literals only: MISSES
        scoped    + raw             1         31   <- scoping only: MISSES
        scoped    + AST             0         13   <- CATCHES

    Read the two middle rows before "improving" this function. Literals
    alone still find 9, because ``Export…`` is a live ``Button`` label on
    Media, Conversations, Prompts, Meetings, Artifacts and the Console
    inspector -- six surfaces the claim did not name. Scoping alone still
    finds 1: ``library_notes_controller.py:5406``, a stale DOCSTRING on
    ``handle_library_notes_export`` -- the handler for the very button that
    ships bare -- so prose *about* the Notes export action, sitting inside a
    Notes module, satisfies a raw scoped probe.

    Only the pair works, and the bottom row also shows the corrected
    sentence still passing (``"Export"``, scoped + AST -> 13), which is the
    half a one-directional check would miss.

    Upgrade path: bind a page (or a page section) to the modules owning the
    surface it documents, and probe those with an ``ast.Constant`` pass that
    excludes docstrings. That binding is the real work and belongs with the
    allowlist in task-32589 (AC#7). A raw tree-wide grep is what makes this
    script runnable in seconds today, and it still catches the larger class
    -- a string no source file contains anywhere.
    """
    command = ["grep", "-rqF"]
    command += [f"--include={glob}" for glob in SOURCE_GLOBS]
    command += [fragment, str(source_root)]
    return subprocess.run(command, capture_output=True).returncode == 0


def is_falsified(fragment: str, source_root: Path = SOURCE_ROOT) -> bool:
    """True when neither the whole fragment nor any literal run of it is emitted.

    A composed line ("Saved 12:47", "Create a Library note (56)") is never
    present whole, so the interpolated parts are split out and the surviving
    literal runs probed individually. One hit is enough: the guide's sentence
    is then at least anchored to something the app really says.
    """
    if emitted_somewhere(fragment, source_root):
        return False
    runs = [run.strip() for run in INTERPOLATION.split(fragment)]
    return not any(
        len(run) >= MIN_LITERAL_RUN and emitted_somewhere(run, source_root)
        for run in runs
    )


def scan(page: Path, end_line: int | None = None) -> tuple[int, list[tuple[int, str]]]:
    """Return ``(checked, misses)`` for one guide page."""
    text = page.read_text()
    seen: set[str] = set()
    misses: list[tuple[int, str]] = []
    checked = 0
    for line, raw in sorted(quoted_fragments(text)):
        if end_line is not None and line > end_line:
            continue
        fragment = normalise(raw)
        if not fragment or fragment in seen or fragment.startswith(SKIP_PREFIXES):
            continue
        seen.add(fragment)
        checked += 1
        if is_falsified(fragment):
            misses.append((line, fragment))
    return checked, misses


def self_test() -> None:
    """Prove the parts that decide anything, against this repo's real source."""
    assert normalise("File name\\<x\\>  or\npath") == "File name<x> or path"
    assert [f for _, f in quoted_fragments('a "hello there" b')] == ["hello there"]
    assert [f for _, f in quoted_fragments("a **Select folder** b")] == [
        "Select folder"
    ]
    assert quoted_fragments('"ab"') == [], "runs under 3 chars are not claims"

    # A real shipped string, and one that cannot exist.
    assert emitted_somewhere("Empty file — nothing to import.")
    assert not emitted_somewhere("Prepare session for commit")

    # The composed-line path: this exact sentence is never present whole
    # (the count is interpolated), but its literal run is.
    assert is_falsified("Prepare session for commit")
    assert not is_falsified("60 applied · listed under Receipts")
    print("self-test OK")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("pages", nargs="*", type=Path, help="guide pages to scan")
    parser.add_argument(
        "--end-line",
        type=int,
        default=None,
        help="ignore quotes below this line (use it to skip a stamp section)",
    )
    parser.add_argument(
        "--fail-on-miss",
        action="store_true",
        help="exit 1 when any candidate is found (for a gate with an allowlist)",
    )
    parser.add_argument("--self-test", action="store_true", help="check the parts")
    args = parser.parse_args(argv)

    if args.self_test:
        self_test()
        return 0
    if not args.pages:
        parser.error("give at least one page, or --self-test")

    total_misses = 0
    for page in args.pages:
        checked, misses = scan(page, args.end_line)
        total_misses += len(misses)
        print(f"{page}: {checked} quoted strings checked, {len(misses)} not emitted")
        for line, fragment in misses:
            print(f"  :{line}\t{fragment[:150]}")
    return 1 if (args.fail_on_miss and total_misses) else 0


if __name__ == "__main__":
    sys.exit(main())
