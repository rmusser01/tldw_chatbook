#!/usr/bin/env python3
"""Guard: the generated CSS bundle must reproduce from its source modules.

TASK-395: ``778f75813`` hand-edited the generated ``tldw_cli_modular.tcss``
instead of a source module; the desync shipped unnoticed for a day because every
rebuild (including the app's boot-time rebuild) silently stripped the live
styling. This checker rebuilds the bundle into a temp file (non-destructive) and
compares it to the committed one, ignoring only the ``Generated:`` timestamp
line, and names any drifted MODULE block so the fix is obvious.

Run locally or in CI:  ``python tldw_chatbook/css/check_bundle_sync.py``
Exits 0 when in sync, 1 (with ``::error::`` annotations) when drifted.

Stdlib-only and imported as a sibling module, so it runs without installing the
app's dependencies (matching the standalone backlog-guard workflow).
"""

import contextlib
import io
import re
import sys
import tempfile
from pathlib import Path

try:
    # Normal package import (tests, `python -m`) -- no global side effects.
    from . import build_css as _build_css
except ImportError:  # pragma: no cover - only when run as a bare script
    # Direct script execution (`python tldw_chatbook/css/build_css.py`'s dir on
    # sys.path) has no package context; add the sibling dir just for this path.
    sys.path.insert(0, str(Path(__file__).parent))
    import build_css as _build_css  # noqa: E402

_TIMESTAMP_RE = re.compile(r"^ \* Generated:.*$", re.MULTILINE)
_MODULE_SPLIT_RE = re.compile(r"/\* ===== MODULE: (.+?) ===== \*/")


def _strip_timestamp(text: str) -> str:
    """Blank the non-deterministic ``Generated:`` header line."""
    return _TIMESTAMP_RE.sub(" * Generated: <timestamp>", text)


def _modules(text: str) -> dict[str, str]:
    """Split a bundle into ``{module_name: block_body}`` by its MODULE markers."""
    parts = _MODULE_SPLIT_RE.split(text)
    return {parts[i]: parts[i + 1] for i in range(1, len(parts), 2)}


def drifted_modules(committed: str, rebuilt: str) -> list[str] | None:
    """Return the drifted module names, or None when the bundle is in sync.

    Args:
        committed: The committed bundle text.
        rebuilt: The freshly rebuilt bundle text.

    Returns:
        ``None`` when the two match once the timestamp line is ignored; otherwise
        the sorted names of MODULE blocks that differ (``["<header>"]`` when only
        the non-module header region drifted).
    """
    committed, rebuilt = _strip_timestamp(committed), _strip_timestamp(rebuilt)
    if committed == rebuilt:
        return None
    committed_modules, rebuilt_modules = _modules(committed), _modules(rebuilt)
    drifted = sorted(
        name
        for name in set(committed_modules) | set(rebuilt_modules)
        if committed_modules.get(name) != rebuilt_modules.get(name)
    )
    return drifted or ["<header / non-module region>"]


def main() -> int:
    """Rebuild the bundle to a temp file and compare it to the committed one.

    Returns:
        ``0`` when the committed bundle reproduces from its sources; ``1`` when it
        has drifted or cannot be rebuilt (with ``::error::`` annotations naming
        the drifted modules or the missing source).
    """
    css_dir = Path(__file__).parent
    committed = (css_dir / "tldw_cli_modular.tcss").read_text(encoding="utf-8")
    with tempfile.TemporaryDirectory() as tmp:
        rebuilt_path = Path(tmp) / "rebuilt.tcss"
        try:
            # build_css narrates each module; silence it so CI logs stay clean.
            with contextlib.redirect_stdout(io.StringIO()):
                _build_css.build_css(css_dir, rebuilt_path)
        except FileNotFoundError as exc:
            # A declared module was removed without updating the manifest -- a
            # desync worth failing on, but reported clearly rather than as a
            # traceback (Qodo #835).
            print(
                f"::error::CSS bundle could not be rebuilt: {exc}. Update the "
                "CSS_MODULES manifest in build_css.py to match the source tree."
            )
            return 1
        rebuilt = rebuilt_path.read_text(encoding="utf-8")

    drifted = drifted_modules(committed, rebuilt)
    if drifted is None:
        print("CSS bundle reproduces from its source modules.")
        return 0

    print(
        "::error::CSS bundle is out of sync with its source modules. Run "
        "`python tldw_chatbook/css/build_css.py` and commit the regenerated "
        "tldw_cli_modular.tcss."
    )
    for module in drifted:
        print(f"::error::drifted module: {module}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
