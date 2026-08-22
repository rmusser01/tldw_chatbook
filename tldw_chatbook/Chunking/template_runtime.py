"""The chatbook seam over the vendored template processor (spec §6.2).

This module is the ONE place that maps the server's flat template shape
(spec §4.1) onto the vendored ``ChunkingTemplate`` dataclass, the ONE place
that resolves a template name against the Media DB, and the single entry
point that runs a template and normalizes its output to chatbook's flat
chunk contract (spec §6.4).

Why here and only here: the upstream server implements the flat→dataclass
mapping three times, two of which raise bare ``KeyError`` on a template
missing ``chunking`` (spec §4.3). chatbook implements it once, guarded.

Fencing note (spec §6.3): the vendored ``engine/templates.py`` also contains
``TemplateManager`` (whose constructor mkdirs a ``template_library``
directory and carries a divergent in-memory store), ``TemplateClassifier``,
and ``TemplateLearner``. chatbook consumes ``TemplateProcessor`` and the two
dataclasses only; ``Tests/Chunking/test_template_runtime.py`` pins that no
production module constructs the fenced classes.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from loguru import logger

from .Chunk_Lib import _synthesize_flat_offsets
from .engine.exceptions import TemplateError
from .engine.templates import ChunkingTemplate, TemplateProcessor, TemplateStage

__all__ = [
    "template_from_record",
    "resolve_template",
    "resolve_ingest_template",
    "materialize_template_chunk_options",
    "apply_template",
    "TemplateResolutionError",
]

_SOURCE_BASIS = "source"


class TemplateResolutionError(Exception):
    """An ingest/re-chunk template choice no longer resolves (spec §9.1).

    Raised by :func:`resolve_ingest_template` when a NON-EMPTY choice
    (picker, batch, stored per-media, or the configured
    ``[chunking] default_template``) names a template that is absent or
    soft-deleted. Named on purpose (AC 37): a template that silently fell
    through to plain chunking is how a library gets chunked two ways
    without the user knowing. The ingest dispatch fails the item on this
    error; the re-chunk worker (PR E) skips-and-counts on it.
    """


def template_from_record(record: Dict[str, Any]) -> ChunkingTemplate:
    """Map a flat (server-shape, spec §4.1) template record to the vendored
    ``ChunkingTemplate``.

    This is the single flat→internal mapper in the codebase (spec §4.3).
    ``record`` is either the flat template dict itself (it has a top-level
    ``chunking`` block) or a DB-style record carrying the flat body under
    ``template_json`` (a JSON string or an already-parsed dict). The
    processor's tolerance for the ``{type, params}`` operation spelling and
    the stage-based shape is inherited for free via the operations lists,
    which are passed through verbatim.

    Args:
        record: Flat template dict, or ``{"name": ..., "template_json": ...}``.

    Returns:
        The mapped ``ChunkingTemplate`` (preprocess/chunk/postprocess stages
        built from the flat blocks, ``chunking.config`` as default options).

    Raises:
        TemplateError: If ``record`` is not a dict, ``template_json`` is not
            valid JSON, the required ``chunking`` block is missing/empty, or
            ``chunking.method`` is missing. Never ``KeyError`` (spec §4.3).
    """
    if not isinstance(record, dict):
        raise TemplateError(
            f"Template record must be a dict, got {type(record).__name__}"
        )

    name = str(record.get("name") or "template")
    body = record
    if "template_json" in record:
        raw = record["template_json"]
        if isinstance(raw, str):
            try:
                body = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise TemplateError(
                    f"Template '{name}' has invalid template_json: {exc}"
                ) from exc
        elif isinstance(raw, dict):
            body = raw
        else:
            raise TemplateError(
                f"Template '{name}' template_json must be a JSON string or dict, "
                f"got {type(raw).__name__}"
            )
    if not isinstance(body, dict):
        raise TemplateError(
            f"Template '{name}' body must be an object, got {type(body).__name__}"
        )

    chunking_cfg = body.get("chunking")
    if not isinstance(chunking_cfg, dict) or not chunking_cfg:
        raise TemplateError(
            f"Template '{name}' is missing the required 'chunking' block"
        )
    method = chunking_cfg.get("method")
    if not isinstance(method, str) or not method:
        raise TemplateError(f"Template '{name}' is missing 'chunking.method'")

    config = chunking_cfg.get("config") or {}
    if not isinstance(config, dict):
        raise TemplateError(
            f"Template '{name}' chunking.config must be an object, "
            f"got {type(config).__name__}"
        )

    stages: List[TemplateStage] = []
    if "preprocessing" in body:
        operations = body["preprocessing"]
        if not isinstance(operations, list):
            raise TemplateError(
                f"Template '{name}' preprocessing must be a list, "
                f"got {type(operations).__name__}"
            )
        stages.append(TemplateStage(name="preprocess", operations=operations))
    stages.append(TemplateStage(name="chunk", operations=[chunking_cfg]))
    if "postprocessing" in body:
        operations = body["postprocessing"]
        if not isinstance(operations, list):
            raise TemplateError(
                f"Template '{name}' postprocessing must be a list, "
                f"got {type(operations).__name__}"
            )
        stages.append(TemplateStage(name="postprocess", operations=operations))

    metadata = body.get("metadata")
    if metadata is None:
        metadata = {}
    if not isinstance(metadata, dict):
        metadata = {}

    return ChunkingTemplate(
        name=name,
        description=str(body.get("description") or record.get("description") or ""),
        base_method=method,
        stages=stages,
        default_options=dict(config),
        metadata=metadata,
    )


def resolve_template(db: Any, name: str) -> Optional[Dict[str, Any]]:
    """Resolve a template name to its flat template dict.

    The ONLY name→template resolution in the codebase (spec §6.2). Queries
    just the columns that are stable across Media DB v6/v7 — ``(name,
    template_json)`` — and filters ``deleted = 0`` (the filter was
    re-assigned here from Task 5 by Task 8's review: PR B's v7 schema added
    the column, and runtime resolution must not resurrect soft-deleted
    templates).

    Args:
        db: Media DB handle exposing ``get_connection()``.
        name: Template name (the table's UNIQUE column).

    Returns:
        The parsed template dict with the authoritative ``name`` column set,
        or ``None`` when the name is unknown, soft-deleted, or the stored
        JSON is corrupt.
    """
    if not isinstance(name, str) or not name:
        return None
    conn = db.get_connection()
    # Clause ORDER is load-bearing: ``FROM ChunkingTemplates WHERE name``
    # is the resolver guard's fingerprint for genuine name→body
    # resolution sites (the CRUD layer's fetches put ``deleted = 0``
    # first), so the name predicate stays first with the deleted filter
    # ANDed after it.
    row = conn.execute(
        "SELECT name, template_json FROM ChunkingTemplates "
        "WHERE name = ? AND deleted = 0",
        (name,),
    ).fetchone()
    if row is None:
        return None
    try:
        row_name = row["name"]
        raw = row["template_json"]
    except (IndexError, KeyError, TypeError):
        row_name = row[0]
        raw = row[1]
    if raw is None:
        return None
    if isinstance(raw, str):
        try:
            body = json.loads(raw)
        except json.JSONDecodeError as exc:
            logger.warning(f"Template '{name}' has corrupt template_json: {exc}")
            return None
    else:
        body = raw
    if not isinstance(body, dict):
        logger.warning(
            f"Template '{name}' template_json is not an object "
            f"(got {type(body).__name__})"
        )
        return None
    resolved = dict(body)
    # The name column is authoritative (it is what the UNIQUE index guards).
    resolved["name"] = row_name if isinstance(row_name, str) else name
    return resolved


def resolve_ingest_template(
    db: Any,
    picker_choice: Optional[str] = None,
    *,
    per_media: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Resolve the template a chunking run should use (spec §9.1, AC 34).

    One resolution helper for both paths, because they differ ONLY in what
    sits at the top of the order:

    * **ingest** (no media row exists yet): ``picker_choice`` (the Library
      ingest picker / batch default) → config ``[chunking]
      default_template`` → ``None`` (plain method/size/overlap options —
      today's behavior).
    * **re-chunk** (``per_media`` given): the stored per-media choice
      (``Media.chunking_config["template"]``) → config default → ``None``.

    Not-found is NEVER a silent fallback (AC 37, spec §9.1): a NON-EMPTY
    choice at any level that no longer resolves (soft-deleted, renamed)
    raises :class:`TemplateResolutionError` instead of falling through to
    the next level or to plain chunking — the ingest dispatch fails the
    item with it; the re-chunk worker (PR E) catches it and
    skips-and-counts.

    Stored-invalid bodies are refused here too (AC-24b ingest half): the
    resolved template is run through the server-parity validator so an
    unexecutable template fails with the NAMED ``InvalidTemplateError``
    rather than an unnamed engine error mid-chunk.

    Args:
        db: Media DB handle exposing ``get_connection()`` (the template
            store). ``None`` means no local store: resolution returns
            ``None`` unless a choice was actually made, in which case the
            named error fires (a made choice must never silently vanish).
        picker_choice: The ingest picker/batch choice, if any.
        per_media: The stored per-media template name (re-chunk path).

    Returns:
        The resolved flat template dict (name + body), or ``None`` when no
        choice exists at any level and no config default is set (plain
        options).

    Raises:
        TemplateResolutionError: A non-empty choice (or config default)
            does not resolve.
        InvalidTemplateError: The resolved stored body fails validation
            (imported from ``chunking_interop_library`` lazily).
    """
    if per_media is not None:
        choice, source = str(per_media).strip(), "stored per-media"
    else:
        choice, source = str(picker_choice or "").strip(), "picker/batch"
    if not choice:
        # config import at call time: module scope would make this
        # import-light runtime module depend on the config machinery.
        from ..config import get_cli_setting

        configured = get_cli_setting("chunking", "default_template")
        choice = str(configured or "").strip()
        source = "config [chunking] default_template"
    if not choice:
        return None
    if db is None:
        raise TemplateResolutionError(
            f"Template '{choice}' (from {source}) cannot be resolved: "
            "no template store is available."
        )
    resolved = resolve_template(db, choice)
    if resolved is None:
        raise TemplateResolutionError(
            f"Template '{choice}' (from {source}) no longer resolves "
            "(deleted or renamed); it was refused instead of silently "
            "falling back to different chunking."
        )
    _refuse_invalid_body(resolved)
    return resolved


def _refuse_invalid_body(template: Dict[str, Any]) -> None:
    """Validate a resolved template body, refusing it with the NAMED error.

    AC-24b: the apply/ingest paths must never surface unnamed engine
    errors for a stored-invalid body — the server-parity validator (Task 6)
    decides before any chunking runs. Mirrors
    ``ChunkingInteropService._validate_body`` (name/description never enter
    the validated body, §7.1 carve-out).
    """
    # Lazy: both imports would be circular at module scope
    # (RAG_Admin.template_validation → ... → Chunking; interop → DB).
    from ..RAG_Admin.template_validation import validate_template
    from .chunking_interop_library import InvalidTemplateError

    body = {
        key: value
        for key, value in template.items()
        if key not in ("name", "description")
    }
    result = validate_template(body)
    if not result["valid"]:
        summary = "; ".join(
            f"{issue['field']}: {issue['message']}"
            for issue in result["errors"][:3]
        )
        raise InvalidTemplateError(
            f"Template '{template.get('name', 'template')}' failed "
            f"validation and was refused: {summary}"
        )


def materialize_template_chunk_options(
    chunk_options: Dict[str, Any], template: Dict[str, Any]
) -> None:
    """Setdefault the template's chunk-stage options into ``chunk_options``
    (in place).

    The precedence fix's second half (spec §9.1/§9.2): the ingest builder
    strips its DEFAULTS when a template is resolved, but every downstream
    seam re-injects its own defaults via ``setdefault`` — ``process_pdf``
    (sentences/500/100), ``process_epub``/``process_fb2``
    (ebook_chapters/1500/200), the audio/video key-by-key re-projection,
    the plain-text tail's fresh three-key dict. Those re-injected defaults
    would arrive at ``Chunker`` as EXPLICIT options, and the Chunker's
    merge order (defaults ← template ← explicit) would let them beat the
    template — the inert-picker trap. Materializing the template's
    chunk-stage options HERE, once, before any branch dispatch, occupies
    those keys so every downstream ``setdefault`` is a no-op and the
    template's values are what travel.

    ``setdefault`` (not overwrite) preserves the other half of the ruling:
    a user-changed form value the builder kept in ``chunk_options``
    overrides the template.

    The ``size`` alias: the audio/video re-projection reads the ``size``
    spelling while the flat template contract uses ``max_size``; both are
    filled from the template's ``max_size``.

    Args:
        chunk_options: The parse's chunking options dict (mutated in
            place; it is per-job data owned by this parse).
        template: The resolved flat template dict.
    """
    chunking = template.get("chunking")
    if not isinstance(chunking, dict):
        return  # invalid bodies are refused upstream (resolve_ingest_template)
    method = chunking.get("method")
    if isinstance(method, str) and method:
        chunk_options.setdefault("method", method)
    config = chunking.get("config")
    if isinstance(config, dict):
        for key, value in config.items():
            chunk_options.setdefault(key, value)
        if "max_size" in config:
            chunk_options.setdefault("size", config["max_size"])


def apply_template(
    template: Dict[str, Any],
    text: str,
    options: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Run a template (pre + chunk + post) and synthesize the flat chunk
    contract (spec §6.4).

    The vendored processor returns ``{'text': c, 'metadata': {}}`` on every
    path — offsets, indices, and counts do not exist unless supplied here.
    This function reconstructs them:

    1. The preprocessing stage runs FIRST and separately, so the transformed
       text the chunks actually come from is captured, along with which
       operation rewrote it. Offsets are computed by
       ``Chunk_Lib._synthesize_flat_offsets`` against that TRANSFORMED text.
    2. Each chunk carries ``metadata.offset_basis``: ``"source"`` when no
       preprocessing op rewrote the text, else ``"preprocessed:<first-op>"``.
    3. When a postprocessing op rewrote/deleted/merged chunks the offset
       mapping is lost, so the offset keys are OMITTED entirely — never
       present-and-``None`` (which raises ``TypeError`` in
       ``vector_store.py`` citation building at search time).
    4. ``chunk_index`` (0-based), ``total_chunks``, and ``word_count`` are
       reconstructed so template chunks are shape-identical to plain ones.

    Args:
        template: Flat template dict (spec §4.1) or a record carrying it
            under ``template_json``.
        text: Source text.
        options: Optional runtime overrides forwarded to the processor
            (e.g. ``{"method": "words"}``).

    Returns:
        Flat chunk dicts: ``text``, ``word_count``, ``chunk_index``,
        ``total_chunks``, ``metadata`` (always containing ``offset_basis``),
        and ``start_char``/``end_char`` when synthesizable.

    Raises:
        TemplateError: If the template is not a valid flat shape.
    """
    if not isinstance(text, str) or not text.strip():
        return []

    mapped = template_from_record(template)
    opts = dict(options or {})
    processor = TemplateProcessor()
    final_options = {**mapped.default_options, **opts}

    pre_stage = next((s for s in mapped.stages if s.name == "preprocess"), None)
    chunk_stage = next(s for s in mapped.stages if s.name == "chunk")
    post_stage = next((s for s in mapped.stages if s.name == "postprocess"), None)

    # 1. Pre-stage separately: capture the transformed text and the basis.
    pre_text = text
    offset_basis = _SOURCE_BASIS
    if pre_stage is not None:
        pre_text, offset_basis = _run_pre_operations(
            processor, pre_stage.operations, text, final_options
        )

    # 2. Chunk (+ postprocess) on the transformed text.
    exec_stages = [chunk_stage] + ([post_stage] if post_stage is not None else [])
    processed = processor.process_template(
        pre_text, _with_stages(mapped, exec_stages), **opts
    )
    final_texts = [_chunk_text_of(chunk) for chunk in processed]

    # 3. Offsets against the transformed text — only when postprocessing
    #    left the chunk list untouched. Any rewrite/deletion/merge loses the
    #    mapping, so the keys are omitted (never emitted as None).
    offsets = None
    if post_stage is None or not post_stage.operations:
        offsets = _synthesize_flat_offsets(pre_text, final_texts)
    else:
        chunk_only = processor.process_template(
            pre_text, _with_stages(mapped, [chunk_stage]), **opts
        )
        if [_chunk_text_of(chunk) for chunk in chunk_only] == final_texts:
            offsets = _synthesize_flat_offsets(pre_text, final_texts)

    # 4. Normalize to the flat contract (the one place guaranteeing no
    #    present-but-None offset).
    total = len(final_texts)
    output: List[Dict[str, Any]] = []
    for index, chunk in enumerate(processed):
        chunk_text = _chunk_text_of(chunk)
        metadata = dict(chunk.get("metadata") or {})
        metadata["offset_basis"] = offset_basis
        record: Dict[str, Any] = {
            "text": chunk_text,
            "word_count": len(chunk_text.split()),
            "chunk_index": index,
            "total_chunks": total,
            "metadata": metadata,
        }
        if offsets is not None:
            span = offsets[index]
            start_char = span.get("start_char")
            end_char = span.get("end_char")
            if start_char is not None and end_char is not None:
                record["start_char"] = int(start_char)
                record["end_char"] = int(end_char)
        output.append(record)
    return output


def _with_stages(
    template: ChunkingTemplate, stages: List[TemplateStage]
) -> ChunkingTemplate:
    """Copy ``template`` with its stages replaced by ``stages``."""
    return ChunkingTemplate(
        name=template.name,
        description=template.description,
        base_method=template.base_method,
        stages=stages,
        default_options=template.default_options,
        metadata=template.metadata,
    )


def _chunk_text_of(chunk: Any) -> str:
    """Extract the text of a processor chunk (dict or bare string)."""
    if isinstance(chunk, dict):
        return str(chunk.get("text", ""))
    return str(chunk)


def _run_pre_operations(
    processor: TemplateProcessor,
    operations: List[Dict[str, Any]],
    text: str,
    base_options: Dict[str, Any],
) -> tuple[str, str]:
    """Run preprocessing operations individually to capture (a) the fully
    transformed text and (b) the name of the first operation that rewrote
    it — the two facts ``TemplateProcessor.process_template`` does not
    surface (it only returns chunks).

    Reuses the processor's own operation registry (``processor._operations``,
    the seam this module owns) and mirrors ``_run_preprocess_stage``'s
    handling of both operation spellings and of dict results, so behavior is
    identical to running the stage through the processor — with the rewrite
    detection added. Unknown operations are skipped, exactly as upstream does.

    Returns:
        ``(transformed_text, offset_basis)`` where ``offset_basis`` is
        ``"source"`` or ``"preprocessed:<first-rewriting-op>"``.
    """
    transformed = text
    basis = _SOURCE_BASIS
    for operation in operations or []:
        if not isinstance(operation, dict):
            continue
        op_name = operation.get("type") or operation.get("operation")
        params = operation.get("params")
        if params is None:
            params = operation.get("config")
        op_options = {**base_options, **(params or {})}
        func = processor._operations.get(op_name) if op_name else None
        if func is None:
            continue
        result = func(transformed, op_options)
        if isinstance(result, str):
            new_text = result
        elif isinstance(result, dict):
            new_text = result.get("text", transformed)
        else:
            new_text = transformed
        if basis == _SOURCE_BASIS and new_text != transformed:
            basis = f"preprocessed:{op_name}"
        transformed = new_text
    return transformed, basis
