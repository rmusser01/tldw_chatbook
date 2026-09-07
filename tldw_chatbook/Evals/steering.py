"""Read a target's steering out of its ``eval_models`` row.

Shared home for ``model_steering``/``_steering_field`` (task-1611's original
reader, relocated here by task-1754). Both are pure functions of a row
mapping -- they touch nothing else, not even ``Evals_DB`` -- but they
originally lived inside ``word_bench.storage``, a module whose own imports
reach ``capture_client``, ``normalizer``, and ``httpx``.
``character_probe.targets`` needs the exact same reader: reusing it there is
deliberate, not incidental -- the alternative (a second, private steering
reader) is exactly the failure mode that produced Critical C1 of the phase 1
whole-branch review, where a hand-rolled reader looked for
``row["system_prompt"]``, a key no ``eval_models`` row has ever carried, and
every real run silently dropped its target's steering while a hand-built
test fixture agreed with the bug. Duplicating this function is the thing to
avoid, not importing it.

The cost of importing it from ``word_bench.storage``, though, was dragging
that module's own imports in transitively -- ``storage`` -> ``capture_client``
-> ``normalizer`` -> ``httpx`` -- into a package (``character_probe``) that
must never acquire distribution vocabulary, even in its import graph. Living
here instead, at the ``Evals`` package's own top level with no imports
beyond the standard library, lets both ``word_bench`` and ``character_probe``
read a target's steering through the one function without either dragging
the other's dependency stack in.

``word_bench.storage`` re-exports ``model_steering`` under its original name
(``from ..steering import model_steering``) so every existing caller
(``UI/Evals/bench_editor.py``, ``UI/Evals/sample_bench.py``, and
``Tests/Evals/word_bench/test_storage.py``, all of which spell it
``word_bench.storage.model_steering``) keeps working unchanged. That
re-export is safe precisely because it is backward-only: ``storage.py``
still imports ``capture_client`` anyway (for ``NEUTRAL_SAMPLER``, used by its
own ``_snapshot``), so re-exporting costs it nothing further, and
``character_probe.targets`` imports this module directly rather than through
``word_bench.storage`` -- see that module's own docstring -- so the
transitive pull never reaches it.
"""

from __future__ import annotations

import json
from typing import Any, Mapping, Optional


def model_steering(model_row: Mapping[str, Any]) -> tuple[Optional[str], Optional[str]]:
    """Read a target's steering out of its ``eval_models`` row.

    ``eval_models.config`` (a free-form JSON column -- see
    ``Evals_DB.create_model``/``get_model``, both of which already parse it
    into a ``dict`` before this function ever sees it) is this app's ONE
    home for a target's steering: ``config["prefix"]`` for a raw-mode
    target, ``config["system_prompt"]`` for a chat-mode one -- the same
    split ``word_bench.models.Target``/``word_bench.capture_client._build_request``
    already enforce (see ``Target``'s own docstring: raw mode prepends a
    literal prefix, chat mode has no prefix slot and instead sends a system
    message). Steering is immutable per row: ``Evals_DB`` has no
    ``update_model``, so a differently-steered variant of a target (e.g.
    the same underlying model with a different prefix) is always a NEW
    ``eval_models`` row, never an edit of an existing one.

    Args:
        model_row: An ``eval_models`` row as returned by
            ``EvalsDB.get_model``/``list_models`` -- ``config`` already
            parsed into a mapping by those methods. A row with no
            ``config`` key at all (every ``eval_models`` row written before
            this convention existed), or an explicit SQL ``NULL``
            (``config`` present and ``None``), is treated the same as an
            explicit ``{}``. Every OTHER non-mapping value -- INCLUDING
            falsy ones like ``0``, ``[]``, ``""``, or ``False`` -- is
            corrupt, not lenient-unsteered; see Raises. Only genuine
            absence carries no information, and only a real empty mapping
            (``{}``) is evidence of "deliberately unsteered"; every other
            shape is evidence something wrote a non-config value into this
            column.

    Returns:
        ``(prefix, system_prompt)``. An unset key or an empty-string value
        both read as ``None`` for that field alike -- a form field left
        blank must never be distinguished from one that was cleared back
        to ``""``. At most one of the pair is ever non-``None`` on a
        successful return; see Raises for the case where the stored row
        itself violates that.

    Raises:
        ValueError: If ``config`` has BOTH ``prefix`` and ``system_prompt``
            set to a non-empty value. ``word_bench.models.Target.__post_init__``
            already rejects constructing a ``Target`` with both set, but a
            row that reached this state some other way (e.g. hand-edited
            JSON) must be surfaced as the corrupt row it is -- naming the
            model id -- rather than have this function silently pick one
            field over the other and hide the inconsistency.
        ValueError: If ``config`` (once present and non-``None``) is
            anything other than a JSON object -- e.g. hand-edited into a
            list, a bare number, a bool, or an empty string -- naming the
            model id, same as the both-set case above, rather than raising
            an opaque ``AttributeError`` out of the ``.get()`` calls below,
            or (for a falsy value) silently reading as "unsteered" as an
            earlier version of this function did. See Args above: falsy is
            NOT a synonym for absent here.
        ValueError: If a present ``prefix`` or ``system_prompt`` value is
            not itself a string (e.g. ``{"prefix": 5}``) -- naming both the
            model id and the offending field, so a non-string never reaches
            ``Target.prefix``/``Target.system_prompt`` and then
            ``capture_client._build_request``'s string concatenation/
            message-building as an untyped value.
    """
    _unset = object()
    raw_config = model_row.get("config", _unset)
    if raw_config is _unset or raw_config is None:
        # Genuine absence (no "config" key at all -- every eval_models row
        # written before this convention existed) or an explicit SQL NULL.
        # Both carry no information about steering and read as unsteered.
        # Nothing else does -- see the type check below.
        config: Any = {}
    elif isinstance(raw_config, str):
        # Defensive: get_model/list_models always hand back an
        # already-parsed value, so a caller going through them never lands
        # here with genuine unparsed JSON text -- this accommodates a
        # caller passing a raw sqlite row instead (config still literal
        # JSON text). A value that fails to parse as JSON at all (e.g. the
        # literal empty string, once ALREADY parsed by get_model out of a
        # stored `""` config) falls straight through as the original
        # string, to be rejected by the non-mapping check below with a
        # message naming the model id -- rather than leaking json.loads's
        # own JSONDecodeError, which never mentions the row at all.
        try:
            config = json.loads(raw_config)
        except ValueError:
            config = raw_config
    else:
        config = raw_config

    if not isinstance(config, dict):
        raise ValueError(
            f"eval_models row {model_row.get('id')!r} has a non-mapping "
            "config; expected an object with optional 'prefix'/"
            "'system_prompt'"
        )

    prefix = _steering_field(config, "prefix", model_row.get("id"))
    system_prompt = _steering_field(config, "system_prompt", model_row.get("id"))
    if prefix and system_prompt:
        raise ValueError(
            f"eval_models row {model_row.get('id')!r} has both prefix and "
            "system_prompt set in its config; a target belongs to exactly "
            "one prompt mode."
        )
    return prefix, system_prompt


def _steering_field(config: dict, field: str, model_id: Any) -> Optional[str]:
    """One steering field out of an already-validated ``config`` mapping,
    type-checked and empty-string-normalized. Shared by ``model_steering``
    for both ``"prefix"`` and ``"system_prompt"`` so the two can never
    silently drift in how they validate or normalize.

    Args:
        config: The row's ``config``, already confirmed to be a ``dict`` by
            ``model_steering``'s own check.
        field: ``"prefix"`` or ``"system_prompt"``.
        model_id: The owning row's id, only for the error message below.

    Returns:
        The field's string value, or ``None`` if the key is absent, its
        value is ``None``, or its value is an empty string.

    Raises:
        ValueError: If the key is present with a non-``None``,
            non-``str`` value (e.g. ``{"prefix": 5}``) -- naming both
            ``model_id`` and ``field`` so the corrupt row and the specific
            offending key are both legible from the message alone.
    """
    value = config.get(field)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(
            f"eval_models row {model_id!r} has a non-string {field!r} in "
            "its config; steering values must be strings."
        )
    return value or None
