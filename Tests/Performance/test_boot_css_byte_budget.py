"""Boot-parsed CSS byte budget (TASK-22222, finding 22222).

Every byte in this census is parsed by Textual BEFORE first paint: the three
``TldwCli.CSS_PATH`` files are read at app start, and the two consolidated
widget-defaults sheets are registered as sources by ``app._get_default_css``
on the same boot path. The 2026-08-24 holistic perf review measured that
total growing 770,285 -> 813,605 bytes between pins with no guard noticing,
while warm boot-to-ready regressed ~11%.

The design tension this budget makes visible (NOT forbidden): the
TASK-15450/21115 consolidation ratchet
(``Tests/UI/test_widget_css_consolidation.py::test_class_level_css_stays_within_the_allowlist``)
deliberately forces new widget/screen CSS into these eagerly-parsed generated
sheets, because per-class ``DEFAULT_CSS`` sources overflow Textual's
64-source parse cache (past the cliff, every first mount of a new widget
class cost a measured 127-378 ms fully-cold reparse). That trade buys
parse-cache-cliff safety with eager boot bytes. Both halves are real costs;
the ratchet counts SOURCES and this budget counts BYTES, so between them the
trade is priced in both currencies. Growing the bundle is expected, normal
maintenance -- growing it SILENTLY is what this guard forbids.

RATCHET (TASK-23029 / ADR-097,
``backlog/decisions/097-boot-budget-ratchets.md``):
``MAX_BOOT_PARSED_CSS_BYTES`` never rises. On a breach, trim or defer the
styles that grew (a rarely-opened modal's large sheet does not need to ride
the boot bundle) or shed equivalent bytes elsewhere in the same PR; the
only other path is an explicit owner exception recorded in the ADR's
exception ledger. The breach message diffs the per-source AND per-segment
census (every ``/* ===== MODULE|WIDGET: ... ===== */`` block in the five
boot-parsed sources) against the pinned snapshot
(``boot_budget_snapshots/boot_css_bytes.json``) so the grown component is
named. Refresh the snapshot only via
``scripts/update_boot_budget_snapshots.py``. When a trim drops the measured
total well below the limit, LOWER the limit to measured + standard slack
(ADR-097's tightening convention) in that same PR.

Measured 833,841 bytes on 2026-08-25 (this branch, base dev f0e896122):
screen_css_scoped 12,615 + tldw_cli_modular 640,599 + screen_css_self 2,418
+ widget_defaults_self 89,127 + widget_defaults_scoped 89,082.
Re-measured 854,720 on 2026-08-28 (dev b5eaa9cf64, TASK-23029): headroom
5,280 -- the tightest of the four ratchets. One segment,
``components/_agentic_terminal.tcss``, is 270,217 B (42% of the bundle).

Documented blind spots (what a byte count cannot see):

* Bytes, not parse time: a selector-heavy kilobyte costs more than a
  comment-heavy one, and this census weighs them the same. Only a TTI probe
  (the review's interleaved method) sees the time.
* Only the boot-registered sources are counted. Widget classes still using
  per-class ``DEFAULT_CSS`` (Textual's own builtins included) register
  separate sources at class registration, and anything calling
  ``stylesheet.add_source`` at runtime is invisible here -- the source-count
  ratchet above is the guard for that direction.
* The census reads the checked-in generated sheets. If they are stale
  against their ``BUNDLED_CSS`` sources the count is stale too; the CSS
  bundle-sync check in ``./scripts/preflight.sh`` / CI owns that gap.
* ``TieAwareStylesheet`` full reparses (see ``css/tie_aware_stylesheet.py``)
  re-pay this whole parse per tie-breaker lowering; this guard prices the
  bundle, not the reparse count.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from tldw_chatbook.css import build_css

#: Drift budget for the total bytes of CSS parsed on the boot path.
#: Measured 833,841 on 2026-08-25. RATCHET (ADR-097): this constant never
#: rises -- see the module docstring before touching it.
#:
#: TIGHTENED 2026-08-31 (TASK-25812): the agentic-terminal split moved the
#: Console/Library/Settings rules off the bundle, taking the boot census
#: from a BREACHED 879,439 B to 679,726 B. The CONSOLE sheet was then put
#: back on the boot path deliberately (see ``TldwCli.CSS_PATH``: loading it
#: at first Console mount cost a one-time full-app restyle and destabilised
#: the `_ui_ready` census), landing at a measured 780,368 B. Pinned at
#: measured + the guard's standard 25,000 B slack per ADR-097's tightening
#: convention -- still banking ~54 KB against the pre-split 860,000.
#: Lowering needs no ledger row -- only raises do.
#:
#: TIGHTENED 2026-09-04 (TASK-24459): dev re-breached the 806,000 pin at
#: 826,956 B four days after the last paydown. The split machinery was
#: generalized to two more screen-owned modules -- `features/_evals.tcss`
#: (39,695 B moved to `screen_feature_evals.tcss`, EvalsScreen) and
#: `features/_scheduling.tcss` (7,936 B moved to
#: `screen_feature_scheduling.tcss`, SchedulesWorkbench) -- landing at a
#: measured 779,320 B. Pinned at measured + the standard 25,000 B slack,
#: rounded down. This guard now also runs in `perf-guard.yml` (task-24461's
#: join step), so the NEXT breach fails the PR that causes it, in minutes.
MAX_BOOT_PARSED_CSS_BYTES = 804_000

#: Anti-vacuity floor: the app bundle alone is ~470 KB post-split, so a
#: census that comes in under this did not measure the real boot-parsed set
#: (a renamed file, an empty generated sheet, a broken source list) and must
#: fail loudly rather than "pass" on a hollow measurement. Re-pinned below
#: the post-split reality on 2026-08-31 (was 700,000 against a pre-split
#: ~854 KB census).
MIN_BOOT_PARSED_CSS_BYTES = 600_000


#: The generated sheets' internal separators: ``/* ===== MODULE: x ===== */``
#: in the bundle, ``/* ===== WIDGET: X (path) ===== */`` in the lifted sheets.
_SEGMENT_MARKER = re.compile(r"/\* ===== [A-Z]+: (?P<label>.+?) ===== \*/")


def _boot_parsed_css_sources() -> dict[str, bytes]:
    """Content of every CSS source parsed on the boot path.

    The ``CSS_PATH`` members come from the app class itself (not a copied
    file list), so a new ``CSS_PATH`` entry is counted automatically; the
    widget-defaults sheets come from the same ``build_css`` helper
    ``app._get_default_css`` calls.

    Returns:
        Mapping of source name to its UTF-8 content bytes.
    """
    # Deferred import: pulls in the app module (conftest.py isolates
    # HOME/XDG/TLDW_CONFIG_PATH for the whole pytest session, so this
    # import cannot touch a live config).
    import tldw_chatbook.app
    from tldw_chatbook.app import TldwCli

    sources: dict[str, bytes] = {}
    for entry in TldwCli.CSS_PATH:
        path = Path(entry)
        assert path.is_file(), (
            f"CSS_PATH member missing on disk: {path} -- the census cannot "
            "be trusted (and neither can the app boot)."
        )
        sources[path.name] = path.read_bytes()

    # The same directory app._get_default_css derives.
    css_dir = Path(tldw_chatbook.app.__file__).parent / "css"
    widget_sources = build_css.widget_defaults_sources(css_dir)
    assert len(widget_sources) == 2, (
        f"expected the two consolidated widget-defaults sheets, got "
        f"{len(widget_sources)} -- app._get_default_css treats this as a "
        "build/packaging bug and so does this census."
    )
    for (_, filename), css, _tie, _scope in widget_sources:
        sources[filename] = css.encode("utf-8")
    return sources


def _boot_parsed_css_census() -> dict[str, int]:
    """Byte census of every CSS source parsed on the boot path."""
    return {name: len(data) for name, data in _boot_parsed_css_sources().items()}


def _boot_parsed_css_segment_census() -> dict[str, int]:
    """Byte census of every marked segment inside the boot-parsed sources.

    Each generated sheet is a concatenation of ``/* ===== KIND: label =====
    */`` blocks (bundle modules, lifted widget/screen declarations). Splitting
    on those markers attributes the bytes to the component that owns them, so
    a budget breach can name the widget or css module that grew rather than a
    640 KB monolith. Bytes before the first marker land in ``(header)``.

    Returns:
        Mapping of ``source::label`` to that segment's size in bytes
        (marker line included).
    """
    segments: dict[str, int] = {}
    for name, data in _boot_parsed_css_sources().items():
        text = data.decode("utf-8")
        matches = list(_SEGMENT_MARKER.finditer(text))
        if not matches:
            segments[f"{name}::(whole file)"] = len(data)
            continue
        boundaries = [0] + [m.start() for m in matches] + [len(text)]
        labels = ["(header)"] + [m.group("label").strip() for m in matches]
        for label, start, end in zip(labels, boundaries[:-1], boundaries[1:]):
            key = f"{name}::{label}"
            segments[key] = segments.get(key, 0) + len(
                text[start:end].encode("utf-8")
            )
    return segments


@pytest.mark.unit
def test_boot_parsed_css_bytes_stay_within_budget(ratchet) -> None:
    """Total bytes of boot-parsed CSS stay within the pinned budget.

    Args:
        ratchet: shared ratchet helper (see ``conftest.py``).
    """
    census = _boot_parsed_css_census()
    total = sum(census.values())
    lines = "\n".join(f"  {name}: {size:,} B" for name, size in census.items())

    assert total >= MIN_BOOT_PARSED_CSS_BYTES, (
        f"boot-parsed CSS census came to only {total:,} B:\n{lines}\n"
        "That is below the anti-vacuity floor "
        f"({MIN_BOOT_PARSED_CSS_BYTES:,} B) -- the census is measuring a "
        "hollow source list, not a real boot."
    )
    if total > MAX_BOOT_PARSED_CSS_BYTES:
        snapshot = ratchet.load_json_snapshot("boot-css-bytes")
        per_source_diff = ratchet.format_byte_diff(
            census, snapshot.get("per_source", {}), "source"
        )
        per_segment_diff = ratchet.format_byte_diff(
            _boot_parsed_css_segment_census(),
            snapshot.get("per_segment", {}),
            "segment",
        )
        raise AssertionError(
            f"boot-parsed CSS grew to {total:,} B "
            f"(ratchet limit {MAX_BOOT_PARSED_CSS_BYTES:,} B):\n{lines}\n"
            "Every one of these bytes is parsed before first paint. "
            "Vs pinned snapshot boot_budget_snapshots/boot_css_bytes.json:\n"
            f"{per_source_diff}\n{per_segment_diff}\n"
            f"{ratchet.ratchet_policy('MAX_BOOT_PARSED_CSS_BYTES')}\n"
            f"Deliberate snapshot refresh: `{ratchet.SNAPSHOT_REFRESH}`"
        )
    ratchet.emit_headroom(
        ratchet.headroom_line(
            "boot-css-bytes", [("bytes", total, MAX_BOOT_PARSED_CSS_BYTES)]
        )
    )
