"""TASK-16688 AC#5 -- the continuation (`next_offset`) walk, outside unit tests.

TASK-16174's live oracle run (Phase E) expanded 8 documents of 289-4,924
chars against `DEFAULT_MAX_CHARS = 8000`, so every call recorded
`expand_truncated: false` and returned a whole document. The measured claim
was "expansion opens a label", never "expansion NAVIGATES a long document":
the window/continuation half of the contract was unit-tested only.

This script closes that half on a real profile. It seeds ONE note larger
than the budget through the app's own writer (`CharactersRAGDB.add_note`),
expands it with the production tool object at the DEFAULT budget, then walks
`next_offset` to exhaustion and checks the walk REASSEMBLES the document
byte-for-byte. No TUI, no LLM, no network, no embeddings -- the note seam
needs none of them, which is why this is a walk and not a retrieval probe.

**Failability.** A walk that merely "returns some text each time" would pass
against a tool that re-served the head forever, so three checks are what
make the reading real: the windows must be CONTIGUOUS (`start[i] ==
end[i-1]`, no gap and no re-served overlap), their concatenation must equal
the stored document exactly, and two planted markers -- one in the first
budget, one in the last -- must each appear in exactly ONE window. The
head-only failure mode fails all three.

Isolation follows `backlog/docs/lessons-live-verification.md`: a scratch
HOME/XDG/`TLDW_CONFIG_PATH` set BEFORE any `tldw_chatbook` import (importing
`config.py` resolves the profile's paths at import time), asserted after the
import, with the REAL config's sha256 recorded before and after so a write
that escaped the scratch profile would be visible.

Usage
-----
    continuation_walk.py [<scratch-root>]

`<scratch-root>` must resolve OUTSIDE this repository (the same containment
rule as TASK-16588's `route_probe.py`); omitted, a fresh temp directory is
used. The report is written next to this file -- a fixed, non-operator path,
so no CLI string can ever address a tracked file.
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import random
import sys
import tempfile

REPORT_PATH = pathlib.Path(__file__).resolve().parent / "report.md"


# ---------------------------------------------------------------------------
# Scratch environment. This block MUST run before any tldw_chatbook import --
# `tldw_chatbook.config` resolves the data/config directories at import time,
# so hoisting the project imports above it would bind the REAL profile and
# silently invalidate every isolation claim in the report (the `data_dir`
# assert in `_assert_isolated` is what catches that mistake).
# ---------------------------------------------------------------------------
def _validated_scratch_path(raw: str) -> pathlib.Path:
    """Resolve an operator-supplied path and refuse unsafe targets.

    The `route_probe.py` precedent (Qodo PR-1729 finding 1): CLI paths flow
    into ``mkdir``/``write_text``, and this script may only ever write
    outside the repository tree, so the containment check that matters is
    "never inside the repo checkout".

    Args:
        raw: The path string exactly as passed on the command line.

    Returns:
        The fully resolved path.

    Raises:
        SystemExit: If the resolved path falls inside this repository.
    """
    resolved = pathlib.Path(raw).resolve()
    repo_root = pathlib.Path(__file__).resolve().parents[3]
    if resolved == repo_root or repo_root in resolved.parents:
        raise SystemExit(f"refusing repo-tree target {resolved} (repo: {repo_root})")
    return resolved


SCRATCH = _validated_scratch_path(
    sys.argv[1]
    if len(sys.argv) > 1
    else tempfile.mkdtemp(prefix="task16688-continuation-walk-")
)
USER_NAME = "walk16688"
PROFILE_ROOT = SCRATCH / "profile"

# Captured BEFORE HOME is overwritten: the real config's digest is the
# isolation proof, and `expanduser("~")` reads os.environ["HOME"].
_REAL_HOME = pathlib.Path(os.path.expanduser("~"))
_REAL_CONFIG = _REAL_HOME / ".config/tldw_cli/config.toml"


def _digest(path: pathlib.Path) -> str:
    if not path.exists():
        return "ABSENT"
    return hashlib.sha256(path.read_bytes()).hexdigest()


_REAL_CONFIG_SHA_BEFORE = _digest(_REAL_CONFIG)

_config_dir = PROFILE_ROOT / "home/.config/tldw_cli"
_data_dir = PROFILE_ROOT / "data"
_config_dir.mkdir(parents=True, exist_ok=True)
_data_dir.mkdir(parents=True, exist_ok=True)
_config_path = _config_dir / "config.toml"
_config_path.write_text(
    "\n".join(
        [
            "[general]",
            f'users_name = "{USER_NAME}"',
            'default_tab = "chat"',
            "",
            "[paths]",
            f'data_dir = "{_data_dir}"',
            "",
            "[first_run]",
            "setup_started = true",
            "setup_completed = true",
            "",
            "[splash_screen]",
            "enabled = false",
            "",
        ]
    ),
    encoding="utf-8",
)

os.environ["HOME"] = str(PROFILE_ROOT / "home")
os.environ["XDG_CONFIG_HOME"] = str(PROFILE_ROOT / "home/.config")
os.environ["XDG_DATA_HOME"] = str(PROFILE_ROOT / "home/.local/share")
os.environ["XDG_CACHE_HOME"] = str(PROFILE_ROOT / "home/.cache")
os.environ["TLDW_CONFIG_PATH"] = str(_config_path)

import asyncio  # noqa: E402 -- after the scratch environment, on purpose

# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------

#: Generic maintenance vocabulary, filler ONLY: it shares no token with the
#: two markers, so a marker cannot be produced by the filler by accident.
_FILLER_WORDS = (
    "inspection interval routine schedule technician logged replacement gasket "
    "fastener torque wrench grease fitting bearing housing pump seal valve "
    "flange bolt washer shim plate bracket panel cover guard rail ladder "
    "platform walkway handrail signage label tag record sheet binder cabinet "
    "storeroom spare consumable filter element cartridge hose clamp fitting "
    "coupling adapter reducer elbow tee union nipple bushing sleeve gland "
    "packing lantern ring stuffing box shaft sleeve wear plate liner"
).split()

#: The note must exceed this so the default budget cannot swallow it whole.
MIN_DOCUMENT_CHARS = 20000

HEAD_MARKER = "HEADWINDOW-ONLY-KESTREL-CLAMP-9317"
TAIL_MARKER = "TAILWINDOW-ONLY-MARROWVANE-SPINDLE-4482"


def _filler(seed: int, target_chars: int) -> str:
    """Deterministic topical filler of at least ``target_chars`` characters."""
    rng = random.Random(seed)
    sentences: list[str] = []
    total = 0
    while total < target_chars:
        words = [rng.choice(_FILLER_WORDS) for _ in range(rng.randint(12, 20))]
        sentence = " ".join(words).capitalize() + "."
        sentences.append(sentence)
        total += len(sentence) + 1
    return " ".join(sentences)


def build_long_note_body() -> str:
    """One note body past `MIN_DOCUMENT_CHARS`, with a head and a tail marker.

    The markers are the per-window controls: HEAD sits inside the first
    budget and TAIL inside the last, so "each marker appears in exactly one
    window" fails immediately for a tool that re-serves the head or that
    slides its window backwards.

    Returns:
        The note body.

    Raises:
        AssertionError: If the built body would not exercise continuation
            (too short, or a marker landing in the wrong window).
    """
    body = (
        f"{HEAD_MARKER}\n\n"
        f"{_filler(11, 9000)}\n\n"
        f"{_filler(12, 9000)}\n\n"
        f"{_filler(13, 7000)}\n\n"
        f"{TAIL_MARKER}"
    )
    if len(body) <= MIN_DOCUMENT_CHARS:
        raise AssertionError(
            f"corpus design failure: {len(body)} chars, needs > "
            f"{MIN_DOCUMENT_CHARS} or the walk has nothing to walk"
        )
    if body.count(HEAD_MARKER) != 1 or body.count(TAIL_MARKER) != 1:
        raise AssertionError("corpus design failure: a marker is not unique")
    return body


# ---------------------------------------------------------------------------
# The walk
# ---------------------------------------------------------------------------


def _assert_isolated() -> pathlib.Path:
    from tldw_chatbook.config import get_user_data_dir

    data_dir = pathlib.Path(get_user_data_dir())
    if not str(data_dir).startswith(str(SCRATCH)):
        raise SystemExit(f"NOT ISOLATED: data_dir={data_dir} scratch={SCRATCH}")
    return data_dir


def _seed_note(body: str) -> str:
    """Write the note through the production writer the tool will read back."""
    from tldw_chatbook.config import get_chachanotes_db_lazy

    db = get_chachanotes_db_lazy()
    if db is None:
        raise SystemExit("FATAL: the ChaChaNotes DB could not be opened")
    note_id = db.add_note(title="Continuation walk corpus", content=body)
    if not note_id:
        raise SystemExit("FATAL: the note writer returned no id")
    stored = db.get_note_by_id(str(note_id))
    if not stored or stored.get("content") != body:
        raise SystemExit("FATAL: the stored note is not the body that was written")
    return str(note_id)


async def _walk(note_id: str, body: str) -> dict:
    """Expand at the default budget, then follow `next_offset` to exhaustion."""
    from tldw_chatbook.Tools.document_expansion_tool import (
        DEFAULT_MAX_CHARS,
        ExpandDocumentTool,
    )

    tool = ExpandDocumentTool()
    windows: list[dict] = []
    pieces: list[str] = []
    calls = 0
    offset: int | None = 0

    while offset is not None:
        # The first call passes no `offset` at all -- exactly what an agent
        # holding a label-only row can send (`source_type` + `source_id`).
        kwargs = {"source_type": "note", "source_id": note_id}
        if calls:
            kwargs["offset"] = offset
        result = await tool.execute(**kwargs)
        calls += 1
        if result["status"] != "ok":
            raise SystemExit(f"FATAL: call {calls} returned {result['status']!r}")
        window = result["window"]
        windows.append(
            {
                "call": calls,
                "start": window["start"],
                "end": window["end"],
                "chars": len(result["text"]),
                "truncated": result["truncated"],
                "next_offset": result["next_offset"],
                "head_marker": HEAD_MARKER in result["text"],
                "tail_marker": TAIL_MARKER in result["text"],
            }
        )
        pieces.append(result["text"])
        offset = result["next_offset"]
        if calls > 64:  # a walk that will not terminate is a finding, not a hang
            raise SystemExit("FATAL: the walk did not terminate within 64 calls")

    reassembled = "".join(pieces)
    contiguous = all(
        windows[i]["start"] == windows[i - 1]["end"] for i in range(1, len(windows))
    )
    return {
        "budget": DEFAULT_MAX_CHARS,
        "document_chars": len(body),
        "total_size_reported": result["total_size"],
        "calls": calls,
        "windows": windows,
        "first_call_truncated": windows[0]["truncated"],
        "first_next_offset": windows[0]["next_offset"],
        "last_next_offset": windows[-1]["next_offset"],
        "starts_at_zero": windows[0]["start"] == 0,
        "ends_at_total": windows[-1]["end"] == len(body),
        "contiguous": contiguous,
        "reassembles": reassembled == body,
        "coverage_chars": len(reassembled),
        "coverage_pct": round(100.0 * len(reassembled) / len(body), 4),
        "head_marker_windows": [w["call"] for w in windows if w["head_marker"]],
        "tail_marker_windows": [w["call"] for w in windows if w["tail_marker"]],
    }


def _checks(walk: dict) -> list[tuple[str, bool, str]]:
    """The pre-registered pass/fail conditions, each with its reading."""
    return [
        (
            "document exceeds the default budget",
            walk["document_chars"] > walk["budget"],
            f"{walk['document_chars']} chars vs budget {walk['budget']}",
        ),
        (
            "first call reports truncated + a next_offset",
            walk["first_call_truncated"] is True
            and walk["first_next_offset"] == walk["budget"],
            f"truncated={walk['first_call_truncated']}, "
            f"next_offset={walk['first_next_offset']}",
        ),
        (
            "walk terminates (final next_offset is None)",
            walk["last_next_offset"] is None,
            f"last next_offset={walk['last_next_offset']}",
        ),
        (
            "windows are contiguous (no gap, no re-served overlap)",
            walk["contiguous"] and walk["starts_at_zero"] and walk["ends_at_total"],
            f"contiguous={walk['contiguous']}, starts_at_zero="
            f"{walk['starts_at_zero']}, ends_at_total={walk['ends_at_total']}",
        ),
        (
            "concatenated windows reassemble the document byte-for-byte",
            walk["reassembles"],
            f"{walk['coverage_chars']} / {walk['document_chars']} chars "
            f"({walk['coverage_pct']}%)",
        ),
        (
            "total_size describes the whole document",
            walk["total_size_reported"] == walk["document_chars"],
            f"reported {walk['total_size_reported']}",
        ),
        (
            "head marker appears in exactly one window (the first)",
            walk["head_marker_windows"] == [1],
            f"windows {walk['head_marker_windows']}",
        ),
        (
            "tail marker appears in exactly one window (the last)",
            walk["tail_marker_windows"] == [walk["calls"]],
            f"windows {walk['tail_marker_windows']}",
        ),
    ]


def _write_report(walk: dict, checks: list, data_dir: pathlib.Path, note_id: str):
    rows = "\n".join(
        f"| {w['call']} | {w['start']} | {w['end']} | {w['chars']} | "
        f"{str(w['truncated']).lower()} | {w['next_offset']} | "
        f"{'HEAD' if w['head_marker'] else ''}"
        f"{'TAIL' if w['tail_marker'] else ''} |"
        for w in walk["windows"]
    )
    check_rows = "\n".join(
        f"| {name} | {reading} | {'**PASS**' if ok else '**FAIL**'} |"
        for name, ok, reading in checks
    )
    verdict = "PASS" if all(ok for _n, ok, _r in checks) else "FAIL"
    REPORT_PATH.write_text(
        f"""# TASK-16688 AC#5 -- continuation walk over a long document

Generated by `continuation_walk.py` (this directory). No TUI, no LLM, no
network, no embeddings: one seeded note, the production
`ExpandDocumentTool`, and `next_offset` followed to exhaustion.

**Verdict: {verdict}** ({walk["calls"]} calls, {walk["coverage_pct"]}% coverage).

## Why this run exists

TASK-16174's Phase E oracle run expanded 8 documents of 289-4,924 chars
against a {walk["budget"]}-char budget, so all 8 recorded
`expand_truncated: false` and the continuation half of the contract was
never exercised outside unit tests (finding 16). This walk is that
exercise: a {walk["document_chars"]:,}-char note, expanded at the DEFAULT
budget and then followed.

## Corpus and profile

| item | value |
|---|---|
| seam | `note` (`add_note` -> `get_note_by_id`, the production writer/reader pair) |
| note id | `{note_id}` |
| document chars | {walk["document_chars"]:,} (> 20,000 by design) |
| budget | {walk["budget"]} (`DEFAULT_MAX_CHARS`, unchanged -- no `max_chars` passed) |
| scratch data_dir | `{data_dir}` |
| real config sha256 before / after | `{_REAL_CONFIG_SHA_BEFORE[:16]}` / `{_digest(_REAL_CONFIG)[:16]}` |

The document carries two unique markers -- `HEADWINDOW-...` in the first
budget, `TAILWINDOW-...` in the last -- built from a vocabulary the filler
never uses, so "which window contains which marker" is a real reading.

## The walk

| call | window start | window end | chars | truncated | next_offset | marker |
|---|---|---|---|---|---|---|
{rows}

Coverage: **{walk["coverage_chars"]:,} of {walk["document_chars"]:,} chars
({walk["coverage_pct"]}%)** across **{walk["calls"]} calls** -- the
concatenation of the windows, compared against the stored note body with
`==`, not a length check.

`truncated` reads `true` on the LAST call too, and that is the tool's
contract rather than an unfinished walk: it means "this window is not the
whole document" (`start > 0 or end < total`), so a payload can never be
mistaken for a complete read. The signal that the walk is DONE is
`next_offset: None`.

## Checks

| check | reading | verdict |
|---|---|---|
{check_rows}

## What this does and does not show

It shows the walk terminates, covers the document exactly once, and reports
its own incompleteness at every step but the last. It does NOT show an
agent choosing to continue -- that is a model behaviour, and the Phase E
oracle run is the instrument for it. The `chunk_start`-anchored window (the
other way into a long document) is TASK-16588's probe, which planted a
marker past this same budget and got 22/22 anchored windows containing it
against 0/22 head windows.

## Reproduce

```
.venv/bin/python Docs/superpowers/qa/2026-08-16-expansion-residue/continuation_walk.py
```

Raw walk data:

```json
{json.dumps(walk, indent=2)}
```
""",
        encoding="utf-8",
    )


def main() -> int:
    data_dir = _assert_isolated()
    body = build_long_note_body()
    note_id = _seed_note(body)
    walk = asyncio.run(_walk(note_id, body))
    checks = _checks(walk)
    _write_report(walk, checks, data_dir, note_id)

    for name, ok, reading in checks:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {reading}")
    print(f"windows={walk['calls']} coverage={walk['coverage_pct']}% report={REPORT_PATH}")
    after = _digest(_REAL_CONFIG)
    if after != _REAL_CONFIG_SHA_BEFORE:
        print(f"ISOLATION BREACH: real config changed ({_REAL_CONFIG_SHA_BEFORE} -> {after})")
        return 2
    return 0 if all(ok for _n, ok, _r in checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
