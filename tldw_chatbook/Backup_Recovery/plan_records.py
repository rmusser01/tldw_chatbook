"""Bounded local RestorePlan persistence; imported archives cannot supply authority."""

import json
from pathlib import Path

from pydantic import TypeAdapter

from .restore_plan import RestorePlan

_PLAN = TypeAdapter(RestorePlan)
_LIMIT = 16 * 1024 * 1024


def save_plan(journal, parent, plan):
    """Write the actual checked local plan before its candidate receipt."""
    from .admission import Admission
    from .archive_reader import _regular
    from .journal import observe_artifact

    if type(plan) is not RestorePlan:
        raise ValueError("recovery_plan_invalid")
    encoded = _PLAN.dump_json(plan)
    if len(encoded) > _LIMIT:
        raise ValueError("recovery_plan_limit")
    path = journal.root / "restore-plan.json"
    try:
        Admission._write_new_record(parent, path.name, encoded)
    except FileExistsError:
        with _regular(path) as stream:
            if stream.read(_LIMIT + 1) != encoded:
                raise ValueError("recovery_plan_changed") from None
    return observe_artifact(path)


def load_plan(journal):
    """Reconstruct the existing typed plan from receipt-bound private bytes."""
    from .archive_reader import _regular
    from .journal import _CandidateReceipt, _matches
    from .publication import _plan_digest

    with journal._locked(exclusive=False) as parent:
        rows = journal._records(parent)
        if not rows or rows[0].event != "candidate_staged":
            raise ValueError("recovery_plan_missing")
        receipt = _CandidateReceipt.model_validate(rows[0].evidence)
        expected = receipt.local_plan
        path = journal.root / "restore-plan.json"
        if (
            expected is None
            or expected.path != str(path)
            or not _matches(expected, str(path))
        ):
            raise ValueError("recovery_plan_changed")
        with _regular(path) as stream:
            encoded = stream.read(_LIMIT + 1)
        if len(encoded) > _LIMIT:
            raise ValueError("recovery_plan_limit")
        plan = _PLAN.validate_json(encoded, strict=True)
        if (
            json.loads(_PLAN.dump_json(plan)) != json.loads(encoded)
            or _plan_digest(plan) != receipt.plan_digest
        ):
            raise ValueError("recovery_plan_changed")
        for values in (
            plan.restore,
            plan.retire,
            plan.preserve,
            plan.destinations,
            plan.selectors,
            plan.containers,
        ):
            if any(
                not isinstance(path, Path)
                or not path.is_absolute()
                or ".." in path.parts
                for _, path in values
            ):
                raise ValueError("recovery_plan_invalid")
        if not _matches(expected, str(path)):
            raise ValueError("recovery_plan_changed")
        return plan
