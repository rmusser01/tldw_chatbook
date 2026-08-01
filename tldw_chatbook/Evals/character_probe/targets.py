"""Resolving an ``eval_models`` row into what a probe run actually needs.

A "target" reaches this package as a raw ``eval_models`` row, exactly as
``EvalsDB.get_model``/``list_models`` hand it back. Two of the three things a
run needs from it are NOT top-level columns:

* the provider-side model name is the row's ``model_id`` column, while the
  row's ``id`` is the target's own identity (what a run and a conversation
  are keyed by) -- confusing the two sends a UUID to the provider;
* the target's steering lives inside the free-form ``config`` JSON column,
  never at the top level. Reading ``row["system_prompt"]`` finds nothing on
  any row the database has ever produced, which is precisely how a whole
  branch of this package came to drop every target's steering silently while
  a test built its own row shape and passed.

The steering read itself is ``word_bench.storage.model_steering`` -- the one
existing home for that convention (task-1611), reused rather than
reimplemented so the two bench types can never drift into disagreeing about
what a row means. Only that reader is used; none of word_bench's measurement
code is touched.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

from ..word_bench.storage import model_steering


@dataclass(frozen=True)
class ResolvedTarget:
    """One validated target: who it is, what to call, and how it is steered.

    ``id`` is the ``eval_models`` row id -- the value a ``Conversation``'s
    ``target_id`` carries and the value ``EvalsDB.create_run`` stores as its
    ``model_id`` column. ``model_id`` is the provider-side model name that
    goes out on the wire. They are deliberately separate fields with the
    confusable names kept, so a reader comparing this against a raw row can
    see which column each came from.
    """

    id: str
    model_id: str
    name: str = ""
    provider: str = ""
    steering: Optional[str] = None


def _required_text(target: Mapping[str, Any], key: str) -> str:
    value = target.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"Target row {target.get('id')!r} (name {target.get('name')!r}) "
            f"has no usable {key!r}: {value!r}. A character probe cannot run "
            "it -- an absent model_id reaches the provider as None, and an "
            "absent id becomes the literal string 'None' on every "
            "conversation it produces."
        )
    return value.strip()


def resolve_target(target: Mapping[str, Any]) -> ResolvedTarget:
    """Validate one ``eval_models`` row and read its steering.

    Args:
        target: A row as returned by ``EvalsDB.get_model``/``list_models``,
            with ``config`` already parsed into a mapping by those methods.

    Returns:
        ResolvedTarget: The row's identity, provider-side model name, and
        chat-mode steering (``None`` when the row is unsteered).

    Raises:
        ValueError: If ``id`` or ``model_id`` is missing, not a string, or
            blank -- naming the offending row, per this package's
            fail-loudly convention, rather than letting ``None`` reach the
            provider and ``"None"`` become a conversation's target.
        ValueError: If the row is steered with a raw-completion ``prefix``.
            A probe is chat-shaped -- a system message followed by
            alternating turns -- and has no slot a literal prefix could be
            prepended to, so honouring such a target is impossible. This
            REJECTS rather than ignoring on purpose: quietly running it
            would evaluate an unsteered model while the bench's own record
            claims a steered one, which is the exact failure this whole
            module exists to prevent. Steering is immutable per
            ``eval_models`` row (there is no ``update_model``), so the
            remedy is a new row carrying ``system_prompt`` instead.
        ValueError: Propagated from ``model_steering`` for a row whose
            ``config`` is corrupt or carries both steering kinds at once.
    """
    target_id = _required_text(target, "id")
    model_name = _required_text(target, "model_id")
    prefix, system_prompt = model_steering(target)
    if prefix:
        raise ValueError(
            f"Target {target_id!r} (name {target.get('name')!r}) is steered "
            f"with a raw-completion prefix ({prefix!r}), which a character "
            "probe cannot honour: a probe sends a system message and "
            "alternating chat turns, with no place to prepend a literal "
            "prefix. Refusing rather than dropping it -- running this "
            "target would evaluate an UNSTEERED model while the run claims "
            "otherwise. Steering cannot be edited on an eval_models row, so "
            "use (or create) a target that carries system_prompt instead."
        )
    return ResolvedTarget(
        id=target_id,
        model_id=model_name,
        name=str(target.get("name") or ""),
        provider=str(target.get("provider") or ""),
        steering=system_prompt,
    )


def resolve_targets(targets: Sequence[Mapping[str, Any]]) -> list[ResolvedTarget]:
    """Validate a whole target list in ONE pass, before anything runs.

    Resolving up front rather than per cell means a malformed or
    prefix-steered target is reported before the first provider call, not
    after part of a grid has already been paid for.

    Args:
        targets: ``eval_models`` rows.

    Returns:
        list[ResolvedTarget]: One per row, in the order given.

    Raises:
        ValueError: If ``targets`` is empty -- a run with no targets can
            never produce a cell.
        ValueError: If two rows share an ``id``. Everything downstream is
            keyed by target id (``run_ids`` in storage, ``target_id`` on
            every conversation), so a duplicate would silently collapse two
            columns into one -- the same guard, for the same reason, that
            ``word_bench.storage.create_run_group`` applies.
        ValueError: Propagated from :func:`resolve_target` for any single
            malformed row.
    """
    if not targets:
        raise ValueError("A character probe run needs at least one target.")
    resolved = [resolve_target(target) for target in targets]
    ids = [item.id for item in resolved]
    if len(set(ids)) != len(ids):
        duplicates = sorted({tid for tid in ids if ids.count(tid) > 1})
        raise ValueError(
            f"targets must have unique ids, got duplicates: {duplicates!r}"
        )
    return resolved
