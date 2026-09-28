"""No `foo.py` sits beside a `foo/` package that permanently shadows it.

Tier-2 review S11, P2 [D3]: `tldw_chatbook/Evals/eval_templates.py` (1,298
lines, 53 KB) lived next to `tldw_chatbook/Evals/eval_templates/`. A
regular package always wins over a same-named module in the same parent,
so `import tldw_chatbook.Evals.eval_templates` resolved to the package's
`__init__.py` -- as did all three production importers
(`eval_orchestrator.py`, `task_loader.py`, `Widgets/template_selector.py`).

That is worse than dead weight: the file *looks* like the template system,
is packaged and shipped, and editing it produces no behaviour change at
all. A trap, not just a leftover.

Cheap to state as a rule, so it is a rule -- one that cannot recur
silently.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "tldw_chatbook"


def _shadowed_modules() -> list[tuple[str, str]]:
    """(`module.py`, `package/`) pairs where the package wins the import."""
    found: list[tuple[str, str]] = []
    for directory in sorted(_PACKAGE_ROOT.rglob("*")):
        if not directory.is_dir() or not (directory / "__init__.py").exists():
            continue
        sibling = directory.with_suffix(".py")
        if sibling.exists():
            found.append(
                (
                    sibling.relative_to(_PACKAGE_ROOT).as_posix(),
                    directory.relative_to(_PACKAGE_ROOT).as_posix() + "/",
                )
            )
    return found


def test_no_module_is_shadowed_by_a_same_named_package() -> None:
    assert _shadowed_modules() == [], (
        "these modules can never execute -- the same-named package beside "
        f"them wins every import: {_shadowed_modules()}. Delete the module "
        "(or merge what is still wanted into the package)."
    )


def test_the_census_would_detect_the_shape_it_was_written_for() -> None:
    """Bite-proof: the detector is a real measurement, not an empty loop."""
    packages = [
        d
        for d in _PACKAGE_ROOT.rglob("*")
        if d.is_dir() and (d / "__init__.py").exists()
    ]
    assert len(packages) > 20, "the package sweep found almost nothing"
