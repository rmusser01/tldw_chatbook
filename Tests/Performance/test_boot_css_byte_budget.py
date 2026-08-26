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

Raising the budget: re-measure (the failure message prints the per-source
byte census), name the widget/screen/feature whose CSS grew the total and
check the growth is styles it actually needs at boot (a rarely-opened
modal's large sheet is a candidate for trimming, not for a raise), then
update ``MAX_BOOT_PARSED_CSS_BYTES`` and this docstring with the new
measured number and the cause, in the same commit. A raise without a named
cause is the failure mode this guard exists to catch.

Measured 833,841 bytes on 2026-08-25 (this branch, base dev f0e896122):
screen_css_scoped 12,615 + tldw_cli_modular 640,599 + screen_css_self 2,418
+ widget_defaults_self 89,127 + widget_defaults_scoped 89,082.

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

from pathlib import Path

import pytest

from tldw_chatbook.css import build_css

#: Drift budget for the total bytes of CSS parsed on the boot path.
#: Measured 833,841 on 2026-08-25; see the module docstring's raise
#: procedure before touching this number.
MAX_BOOT_PARSED_CSS_BYTES = 860_000

#: Anti-vacuity floor: the app bundle alone is ~640 KB, so a census that
#: comes in under this did not measure the real boot-parsed set (a renamed
#: file, an empty generated sheet, a broken source list) and must fail
#: loudly rather than "pass" on a hollow measurement.
MIN_BOOT_PARSED_CSS_BYTES = 700_000


def _boot_parsed_css_census() -> dict[str, int]:
    """Byte census of every CSS source parsed on the boot path.

    The ``CSS_PATH`` members come from the app class itself (not a copied
    file list), so a new ``CSS_PATH`` entry is counted automatically; the
    widget-defaults sheets come from the same ``build_css`` helper
    ``app._get_default_css`` calls.

    Returns:
        Mapping of source name to its size in bytes.
    """
    # Deferred import: pulls in the app module (conftest.py isolates
    # HOME/XDG/TLDW_CONFIG_PATH for the whole pytest session, so this
    # import cannot touch a live config).
    import tldw_chatbook.app
    from tldw_chatbook.app import TldwCli

    census: dict[str, int] = {}
    for entry in TldwCli.CSS_PATH:
        path = Path(entry)
        assert path.is_file(), (
            f"CSS_PATH member missing on disk: {path} -- the census cannot "
            "be trusted (and neither can the app boot)."
        )
        census[path.name] = len(path.read_bytes())

    # The same directory app._get_default_css derives.
    css_dir = Path(tldw_chatbook.app.__file__).parent / "css"
    widget_sources = build_css.widget_defaults_sources(css_dir)
    assert len(widget_sources) == 2, (
        f"expected the two consolidated widget-defaults sheets, got "
        f"{len(widget_sources)} -- app._get_default_css treats this as a "
        "build/packaging bug and so does this census."
    )
    for (_, filename), css, _tie, _scope in widget_sources:
        census[filename] = len(css.encode("utf-8"))
    return census


@pytest.mark.unit
def test_boot_parsed_css_bytes_stay_within_budget() -> None:
    """Total bytes of boot-parsed CSS stay within the pinned budget."""
    census = _boot_parsed_css_census()
    total = sum(census.values())
    lines = "\n".join(f"  {name}: {size:,} B" for name, size in census.items())

    assert total >= MIN_BOOT_PARSED_CSS_BYTES, (
        f"boot-parsed CSS census came to only {total:,} B:\n{lines}\n"
        "That is below the anti-vacuity floor "
        f"({MIN_BOOT_PARSED_CSS_BYTES:,} B) -- the census is measuring a "
        "hollow source list, not a real boot."
    )
    assert total <= MAX_BOOT_PARSED_CSS_BYTES, (
        f"boot-parsed CSS grew to {total:,} B "
        f"(budget {MAX_BOOT_PARSED_CSS_BYTES:,} B):\n{lines}\n"
        "Every one of these bytes is parsed before first paint. Name what "
        "grew (diff the generated sheets / css modules against a clean "
        "checkout), decide whether those styles must really ride the boot "
        "bundle, then raise the budget per the module docstring's "
        "procedure -- with the cause named in the same commit."
    )
