"""Read the actual caller and callee for guards that inspect application steps.

No expressions, dependencies or verdicts are rewritten. Raw-graph tests check
admission separately; the callee metadata remains available to output guards.
"""

from pathlib import Path

import yaml


def derived_workflow(root: Path) -> dict:
    """Load both physical documents, exposing their disjoint jobs for selection."""
    directory = root / ".github/workflows"
    parent = yaml.safe_load(
        (directory / "derived-artifacts.yml").read_text(encoding="utf-8")
    )
    child = yaml.safe_load(
        (directory / "ci-qualification.yml").read_text(encoding="utf-8")
    )
    assert not (
        parent["jobs"].keys() & child["jobs"].keys()
    ), "Duplicate caller/callee job IDs"
    return {
        **parent,
        "jobs": {**parent["jobs"], **child["jobs"]},
        "qualification_workflow": child,
    }
