"""Bounded local RestorePlan persistence; imported archives cannot supply authority."""

import json
from pathlib import Path

from pydantic import TypeAdapter

from .restore_plan import RestorePlan

_PLAN = TypeAdapter(RestorePlan)
_LIMIT = 16 * 1024 * 1024


def _encode_plan(plan):
    """Keep pre-selection plan bytes/digests canonical when new fields are absent."""
    defaults = {
        "retained_configs": (),
        "requested_groups": None,
        "effective_groups": (),
        "required_groups": (),
    }
    return _PLAN.dump_json(
        plan,
        exclude={
            key for key, default in defaults.items() if getattr(plan, key) == default
        },
    )


def save_plan(journal, parent, plan):
    """Write the actual checked local plan before its candidate receipt."""
    from .admission import Admission
    from .archive_reader import _regular
    from .journal import observe_artifact

    if type(plan) is not RestorePlan:
        raise ValueError("recovery_plan_invalid")
    encoded = _encode_plan(plan)
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
    with journal._locked(exclusive=False) as parent:
        return _load_plan(journal, journal._records(parent))


def _load_plan(journal, rows):
    """Use already-read records while the caller retains its journal lock."""
    from .archive_reader import _regular
    from .journal import _CandidateReceipt, _matches
    from .publication import _plan_digest

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
        json.loads(_encode_plan(plan)) != json.loads(encoded)
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
        tuple(
            (row.archive_config_id, row.config_path) for row in plan.retained_configs
        ),
    ):
        if any(
            not isinstance(path, Path) or not path.is_absolute() or ".." in path.parts
            for _, path in values
        ):
            raise ValueError("recovery_plan_invalid")
    if not _matches(expected, str(path)):
        raise ValueError("recovery_plan_changed")
    return plan
