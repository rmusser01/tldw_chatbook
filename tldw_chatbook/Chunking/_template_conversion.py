"""Pure v6→v7 chunking-template row conversion (spec §5.3/§5.4, task-7).

The Media DB v6→v7 migration rebuilds ``ChunkingTemplates``; this module
holds the per-row mechanics so they are unit-testable in isolation from the
DB layer:

- pipeline→flat conversion: ``{base_method, pipeline: [...]}`` →
  ``{preprocessing, chunking: {method, config}, postprocessing}``;
- operation spelling rewrite: ``{type, params}`` → ``{operation, config}``;
- method-name repair: ``structural``/``hierarchical`` → ``structure_aware``
  (the latter with ``config.hierarchical = true``), ``contextual`` →
  ``sentences``;
- operation mapping/loss: ``section_detection`` → ``extract_sections``;
  ``extract_metadata``, ``code_block_detection``, ``add_context`` are
  dropped and recorded in ``metadata._dropped_operations``;
- tags extraction: the top-level ``tags`` (else ``metadata.tags``) list
  moves to the ``tags`` column and is removed from the JSON body;
- quarantine: a row that cannot be converted (unparseable JSON, or no chunk
  stage and no ``base_method``) is soft-deleted, renamed
  ``<name> (needs review)``, and its original body preserved under
  ``metadata._unconverted`` — never silently re-pointed at a default method.

Everything here is pure: no DB, no engine imports — only stdlib. The
migration (``DB/Client_Media_DB_v2._apply_migration_v6_to_v7``) decides the
drop-vs-convert precedence and calls :func:`convert_template_row` per row
inside its ADR-030 transaction.
"""

from __future__ import annotations

import copy
import json
import logging
import uuid as uuid_module
from typing import Any, Callable, Dict, Optional

__all__ = [
    "DEFAULT_METHOD",
    "QUARANTINE_SUFFIX",
    "DROPPED_OPERATIONS",
    "convert_template_row",
    "convert_template_body",
]

#: Chunking method a quarantined row's repair block names (spec §5.4: the
#: row keeps "a chunking block naming the default method" so it stays a
#: parseable, editable template).
DEFAULT_METHOD = "words"

#: Appended to a quarantined row's name so the original live name is freed.
QUARANTINE_SUFFIX = " (needs review)"

#: v6-registered operations with no counterpart in the vendored processor's
#: registry (spec §5.4: four are lost; ``section_detection`` maps instead).
DROPPED_OPERATIONS = frozenset(
    {
        "extract_metadata",
        "code_block_detection",
        "add_context",
    }
)

#: ``section_detection`` → ``extract_sections`` (the one intent-matching map).
MAPPED_OPERATIONS = {"section_detection": "extract_sections"}

#: Method-name repair for methods the old seeds used but the live registry
#: rejects (``InvalidChunkingMethodError``); hierarchical additionally flags
#: ``config.hierarchical = true`` so the intent survives the rename.
METHOD_REPAIRS = {
    "structural": "structure_aware",
    "hierarchical": "structure_aware",
    "contextual": "sentences",
}

_REPAIRED_METHOD_FLAG = {"hierarchical": ("hierarchical", True)}


def convert_template_row(
    row: Dict[str, Any],
    *,
    uuid_factory: Callable[[], uuid_module.UUID] = uuid_module.uuid4,
) -> Dict[str, Any]:
    """Convert one v6 ``ChunkingTemplates`` row to the v7 column set.

    Args:
        row: A v6-shaped row dict (``name``, ``description``,
            ``template_json`` as JSON string or dict, ``is_system``,
            ``created_at``/``updated_at`` optional). The input is not
            mutated.
        uuid_factory: Injects the uuid source (tests pin determinism).

    Returns:
        A dict with the v7 insert column values: ``uuid``, ``name``,
        ``description``, ``template_json`` (JSON string), ``tags`` (JSON
        list string or ``None``), ``is_builtin``, ``version`` (1),
        ``deleted``, ``created_at``, ``updated_at``.
    """
    name = str(row.get("name") or "template")
    raw_body = row.get("template_json")
    description = row.get("description")

    body, unconverted = _load_body(name, raw_body)
    pristine = copy.deepcopy(body)
    dropped: list = []
    if unconverted is None:
        assert isinstance(body, dict)
        tags, body = _extract_tags(body)
        converted_body = convert_template_body(body, dropped_operations=dropped)
        # §5.4: no chunk stage and no base_method is unconvertible, not a
        # template silently re-pointed at the default method.
        if converted_body["chunking"]["method"] is None:
            unconverted = pristine
    if unconverted is not None:
        converted_body = _quarantine_body(name, description, unconverted)
        deleted = True
        out_name = name + QUARANTINE_SUFFIX
        tags = None
        logging.warning(
            "Chunking template %r could not be converted to the v7 flat "
            "shape; quarantined as %r with the original body preserved "
            "under metadata._unconverted",
            name,
            out_name,
        )
    else:
        deleted = False
        out_name = name

    return {
        "uuid": str(uuid_factory()),
        "name": out_name,
        "description": description,
        "template_json": json.dumps(converted_body),
        "tags": json.dumps(tags) if tags is not None else None,
        "is_builtin": bool(row.get("is_system")),
        "version": 1,
        "deleted": deleted,
        "created_at": _timestamp(row.get("created_at")),
        "updated_at": _timestamp(row.get("updated_at")),
    }


def convert_template_body(
    body: Dict[str, Any], *, dropped_operations: Optional[list] = None
) -> Dict[str, Any]:
    """Convert a parsed v6 pipeline body to the flat v7 shape (spec §5.4).

    ``{base_method, pipeline: [{stage, method, options, operations}]}`` →
    ``{preprocessing: [...], chunking: {method, config}, postprocessing:
    [...]}`` with operation spellings rewritten, methods repaired, and lost
    operations appended to ``dropped_operations`` (a caller-owned list so
    the migration can surface the loss).

    Already-flat bodies (no ``pipeline``/``base_method``) are returned with
    method repair and operation rewrite still applied — a no-op for bodies
    that were flat all along.
    """
    repaired = copy.deepcopy(body)
    dropped = dropped_operations if dropped_operations is not None else []

    chunk_method: Optional[str] = None
    chunk_options: Dict[str, Any] = {}
    preprocessing: list = []
    postprocessing: list = []
    base_method = repaired.pop("base_method", None)

    pipeline = repaired.pop("pipeline", None)
    if pipeline is not None or base_method is not None:
        for stage in pipeline or []:
            if not isinstance(stage, dict):
                continue
            stage_name = stage.get("stage")
            if stage_name == "preprocess":
                preprocessing.extend(_convert_operations(stage.get("operations"), dropped))
            elif stage_name == "chunk":
                if chunk_method is None and isinstance(stage.get("method"), str):
                    chunk_method = stage["method"]
                options = stage.get("options")
                if isinstance(options, dict):
                    chunk_options.update(options)
            elif stage_name == "postprocess":
                postprocessing.extend(_convert_operations(stage.get("operations"), dropped))
            # unknown stages are dropped silently — nothing ever produced them
    else:
        # Already flat (no pipeline/base_method): the existing chunking block
        # is the method source; only repairs and the operation rewrite apply.
        chunking = repaired.pop("chunking", None)
        if isinstance(chunking, dict) and isinstance(chunking.get("method"), str):
            chunk_method = chunking["method"]
            config = chunking.get("config")
            if isinstance(config, dict):
                chunk_options.update(config)
        preprocessing.extend(_convert_operations(repaired.pop("preprocessing", None), dropped))
        postprocessing.extend(_convert_operations(repaired.pop("postprocessing", None), dropped))

    if chunk_method is None:
        chunk_method = base_method if isinstance(base_method, str) else None

    chunk_method, chunk_options = _repair_method(chunk_method, chunk_options)

    converted: Dict[str, Any] = {}
    for key in ("name", "description"):
        if key in repaired:
            converted[key] = repaired[key]
    if preprocessing:
        converted["preprocessing"] = preprocessing
    converted["chunking"] = {"method": chunk_method, "config": chunk_options}
    if postprocessing:
        converted["postprocessing"] = postprocessing
    metadata = repaired.get("metadata")
    converted["metadata"] = metadata if isinstance(metadata, dict) else {}
    if dropped:
        converted["metadata"]["_dropped_operations"] = sorted(set(dropped))
    return converted


def _load_body(name: str, raw: Any) -> tuple[Optional[dict], Optional[Any]]:
    """Parse ``template_json``; on failure return ``(None, original)``."""
    if isinstance(raw, dict):
        return raw, None
    if isinstance(raw, str):
        try:
            body = json.loads(raw)
        except json.JSONDecodeError:
            return None, raw
        if isinstance(body, dict):
            return body, None
        return None, raw
    return None, raw


def _quarantine_body(
    name: str, description: Any, unconverted: Any
) -> Dict[str, Any]:
    """The repairable flat body a quarantined row keeps (spec §5.4)."""
    body: Dict[str, Any] = {"name": name + QUARANTINE_SUFFIX}
    if description is not None:
        body["description"] = description
    body["chunking"] = {"method": DEFAULT_METHOD, "config": {}}
    body["metadata"] = {"_unconverted": unconverted}
    return body


def _extract_tags(body: dict) -> tuple[Optional[list], dict]:
    """Move ``tags`` (top-level, else ``metadata.tags``) out of the body."""
    tags: Optional[list] = None
    if isinstance(body.get("tags"), list):
        tags = body.pop("tags")
    metadata = body.get("metadata")
    if isinstance(metadata, dict) and isinstance(metadata.get("tags"), list):
        if tags is None:
            tags = metadata.pop("tags")
        else:
            del metadata["tags"]
        if not metadata:
            body.pop("metadata", None)
    return tags, body


def _convert_operations(operations: Any, dropped: list) -> list:
    """Rewrite ``{type, params}`` → ``{operation, config}``, mapping or
    dropping the four v6-only operations."""
    converted = []
    for operation in operations or []:
        if not isinstance(operation, dict):
            converted.append(operation)
            continue
        op_name = operation.get("operation") or operation.get("type")
        if not op_name:
            converted.append(operation)
            continue
        if op_name in MAPPED_OPERATIONS:
            op_name = MAPPED_OPERATIONS[op_name]
        elif op_name in DROPPED_OPERATIONS:
            dropped.append(op_name)
            continue
        params = operation.get("config")
        if params is None:
            params = operation.get("params")
        converted.append(
            {
                "operation": op_name,
                **({"config": params} if params is not None else {}),
            }
        )
    return converted


def _repair_method(
    method: Optional[str], config: Dict[str, Any]
) -> tuple[Optional[str], Dict[str, Any]]:
    """Apply the §5.4 method-name repairs to ``method``/``config``."""
    if method in METHOD_REPAIRS:
        repaired = METHOD_REPAIRS[method]
        flag = _REPAIRED_METHOD_FLAG.get(method)
        if flag is not None:
            key, value = flag
            config = {**config, key: value}
        return repaired, config
    return method, config


def _timestamp(value: Any) -> Any:
    """Normalize a copied timestamp to the storage string form."""
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return str(value)
