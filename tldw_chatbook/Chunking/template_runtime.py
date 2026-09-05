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
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from loguru import logger

from .Chunk_Lib import _synthesize_flat_offsets
from .engine.chunker import Chunker
from .engine.exceptions import TemplateError
from .engine.templates import ChunkingTemplate, TemplateProcessor, TemplateStage
from .lab_models import ExecutionReport, PreparedRecipe

if TYPE_CHECKING:  # runtime import is lazy (circular with auto_selection)
    from .auto_selection import AutoDecision

__all__ = [
    "template_from_record",
    "resolve_template",
    "resolve_ingest_template",
    "resolve_for_rechunk",
    "materialize_template_chunk_options",
    "apply_template",
    "execute_prepared",
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
    media_type: Optional[str] = None,
    title: Optional[str] = None,
    filename: Optional[str] = None,
    url: Optional[str] = None,
) -> Optional[Union[Dict[str, Any], "AutoDecision"]]:
    """Resolve the template a chunking run should use (spec §9.1, AC 34;
    auto-selection spec §4.3, ACs 8/11).

    One resolution helper for both paths, because they differ ONLY in what
    sits at the top of the order:

    * **ingest** (no media row exists yet): ``picker_choice`` (the Library
      ingest picker / batch default) → config ``[chunking]
      default_template`` → ``None`` (plain method/size/overlap options —
      today's behavior).
    * **re-chunk** (``per_media`` given): the stored per-media choice
      (``Media.chunking_config["template"]``) → config default → ``None``.

    **The Auto sentinel** (auto-selection spec §4.3): a ``picker_choice``
    equal to :data:`~tldw_chatbook.Chunking.auto_selection.AUTO_SENTINEL`
    (``"auto"``) routes through
    :func:`~tldw_chatbook.Chunking.auto_selection.resolve_auto` and returns
    its :class:`AutoDecision` — whichever tier won. The sentinel fires at
    the PICKER tier only: it is the user's terminating choice (never
    falling through to the config default), and the config
    ``default_template`` NEVER triggers auto (AC 11 — a configured sentinel
    value is an unresolvable template name and raises, exactly as any other
    configured name that no longer resolves). The stored per-media path is
    untouched: a stored ``mode: "auto"`` is ``resolve_for_rechunk``'s to
    re-resolve, and a stored *name* keeps this function's #2 behavior.

    Not-found is NEVER a silent fallback (AC 37, spec §9.1): a NON-EMPTY
    choice at any level that no longer resolves (soft-deleted, renamed)
    raises :class:`TemplateResolutionError` instead of falling through to
    the next level or to plain chunking — the ingest dispatch fails the
    item with it; the re-chunk worker (PR E) catches it and
    skips-and-counts. The Auto choice itself is exempt by construction: it
    never names a template, and ``resolve_auto`` always terminates (plan or
    plain) without raising.

    Stored-invalid bodies are refused here too (AC-24b ingest half): the
    resolved template is run through the server-parity validator so an
    unexecutable template fails with the NAMED ``InvalidTemplateError``
    rather than an unnamed engine error mid-chunk.

    Args:
        db: Media DB handle exposing ``get_connection()`` (the template
            store). ``None`` means no local store: resolution returns
            ``None`` unless a choice was actually made, in which case the
            named error fires (a made choice must never silently vanish);
            the Auto sentinel still decides (tier 1 vacuous, the planner
            answers).
        picker_choice: The ingest picker/batch choice, if any. The reserved
            value ``"auto"`` selects the Auto decision path.
        per_media: The stored per-media template name (re-chunk path).
        media_type: The ingest job's media-type string (Auto tier 1's
            classifier input; the planner's type switch). Optional —
            callers without metadata still get a decision (the generic
            plan).
        title: The item's title, if known (classifier ``title_regex``).
        filename: The item's filename, if known (``filename_regex``).
        url: The item's URL, if known (``url_regex``).

    Returns:
        The resolved flat template dict (name + body), or an
        :class:`AutoDecision` when the picker sentinel fired, or ``None``
        when no choice exists at any level and no config default is set
        (plain options).

    Raises:
        TemplateResolutionError: A non-empty choice (or config default)
            does not resolve.
        InvalidTemplateError: The resolved stored body fails validation
            (imported from ``chunking_interop_library`` lazily).
    """
    if per_media is None:
        picked = str(picker_choice or "").strip()
        if picked == _auto_sentinel():
            # Lazy: auto_selection imports this module at ITS module scope,
            # so this import must stay inside the function.
            from .auto_selection import resolve_auto

            return resolve_auto(
                db,
                media_type=media_type,
                title=title,
                filename=filename,
                url=url,
            )
        choice, source = picked, "picker/batch"
    else:
        choice, source = str(per_media).strip(), "stored per-media"
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


def resolve_for_rechunk(
    db: Any,
    chunking_config: Union[Dict[str, Any], str, None],
    *,
    media_type: Optional[str] = None,
    title: Optional[str] = None,
    filename: Optional[str] = None,
    url: Optional[str] = None,
) -> Optional[Union[Dict[str, Any], "AutoDecision"]]:
    """Resolve the chunking a RE-CHUNK run should use (auto-selection spec
    §4.3 re-chunk half, AC 10).

    A stored ``mode: "auto"`` RE-RESOLVES — the decision is re-derived from
    the current template store (a classifier block added since ingest flips
    the tier), never replayed from the stored ``auto_tier``. Anything else
    keeps #2's behavior byte-for-byte: the stored ``template`` name runs
    through :func:`resolve_ingest_template`'s per-media path (config
    default fallback, named refusals, invalid-body refusal).

    Args:
        db: Media DB handle exposing ``get_connection()`` (the template
            store). ``None`` still decides for ``mode: "auto"`` (tier 1
            vacuous, the planner answers).
        chunking_config: The item's stored ``Media.chunking_config`` — a
            dict, its JSON-string spelling, or ``None``/absent.
        media_type: The media row's type (``Media.type``) — the re-resolved
            decision's classifier input and planner type switch.
        title: The media row's title, if known.
        filename: The item's filename, if known (the Media table carries no
            filename column; pass ``None`` unless a caller knows better).
        url: The media row's URL, if known.

    Returns:
        An :class:`AutoDecision` for a stored ``mode: "auto"``; otherwise
        whatever :func:`resolve_ingest_template` returns for the stored
        name (flat template dict, or ``None`` for plain options).

    Raises:
        TemplateResolutionError / InvalidTemplateError: A stored (or
            configured) explicit name that does not resolve or fails
            validation — exactly #2's re-chunk refusal behavior.
    """
    config: Optional[Dict[str, Any]] = None
    if isinstance(chunking_config, dict):
        config = chunking_config
    elif isinstance(chunking_config, str):
        try:
            parsed = json.loads(chunking_config)
        except (TypeError, ValueError):
            parsed = None
        config = parsed if isinstance(parsed, dict) else None
    if config is not None and str(config.get("mode") or "").strip() == (
        _auto_sentinel()
    ):
        from .auto_selection import resolve_auto

        return resolve_auto(
            db,
            media_type=media_type,
            title=title,
            filename=filename,
            url=url,
        )
    stored_name: Optional[str] = None
    if config is not None:
        value = config.get("template")
        name = str(value).strip() if value else ""
        stored_name = name or None
    return resolve_ingest_template(db, per_media=stored_name)


def _auto_sentinel() -> str:
    """The reserved picker-sentinel name (lazy: circular import with
    ``auto_selection``, which imports this module at ITS module scope)."""
    from .auto_selection import AUTO_SENTINEL

    return AUTO_SENTINEL


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

    Shares structured execution with Lab, retaining engine metadata and
    provenance. This legacy adapter reconstructs its established flat fields:

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
        structured ``provenance``,
        and ``start_char``/``end_char`` when synthesizable.

    Raises:
        TemplateError: If the template is not a valid flat shape.
    """
    if not isinstance(text, str) or not text.strip():
        return []

    report, offset_basis, unchanged = _execute_report(template, text, options)
    # Compatibility adapter: legacy consumers use whitespace-tolerant offsets.
    # Reports carry only exact verified spans and never inherit these guesses.
    offsets = (
        _synthesize_flat_offsets(
            report.transformed_text, [c["text"] for c in report.chunks]
        )
        if unchanged
        else None
    )
    total = len(report.chunks)
    output: List[Dict[str, Any]] = []
    for index, chunk in enumerate(report.chunks):
        chunk_text = _chunk_text_of(chunk)
        metadata = dict(chunk.get("metadata") or {})
        metadata["offset_basis"] = offset_basis
        record: Dict[str, Any] = {
            "text": chunk_text,
            "word_count": len(chunk_text.split()),
            "chunk_index": index,
            "total_chunks": total,
            "metadata": metadata,
            "provenance": deepcopy(chunk["provenance"]),
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


def execute_prepared(recipe: PreparedRecipe, text: str) -> ExecutionReport:
    """Execute a validated immutable Lab snapshot using the saved-apply seam.

    Call inside the bounded preview worker, never on Textual's event loop.
    Restored/tampered snapshots and changed runtime identities must be prepared
    again before execution; the effective document alone grants no admission.
    """
    from .lab_preflight import PreviewUnsupportedError, prepare_recipe

    checked = prepare_recipe(json.loads(recipe.authored_json), runtime=recipe.runtime)
    if checked != recipe:
        raise PreviewUnsupportedError(
            "snapshot",
            "Prepared snapshot does not match its authored recipe and runtime",
        )
    if not isinstance(text, str):
        raise TypeError("Sample text must be a string")
    report, _, _ = _execute_report(json.loads(recipe.effective_json), text)
    return report


def registered_template_operations() -> frozenset[str]:
    """Expose live operation names without exporting the vendor mapper types."""
    return frozenset(TemplateProcessor()._operations)


def run_template_preprocessing_operation(
    text: str, operation: str, config: dict[str, Any]
) -> str | dict[str, Any]:
    """Run one already-preflighted preprocessing operation for Lab admission.

    Args:
        text: Current preprocessing text.
        operation: Registered preprocessing operation name.
        config: Preflighted effective operation configuration.

    Returns:
        The vendor operation's unchanged string or structured result.
    """
    return TemplateProcessor()._operations[operation](text, config)


class _ReportingChunker(Chunker):
    """Observe the engine's actual sanitation without copying its algorithm."""

    sanitized_text: str | None = None
    resolved_method: str | None = None

    def _sanitize_input(self, text: str, **kwargs: Any) -> str:
        result = super()._sanitize_input(text, **kwargs)
        if self.sanitized_text is None:
            self.sanitized_text = result
        return result

    def _resolve_method(
        self, method: Any, language: str | None, options: dict | None = None
    ) -> str:
        result = super()._resolve_method(method, language, options)
        self.resolved_method = result
        return result


def sanitize_template_input(text: str) -> str:
    """Apply the engine's real input sanitation without a security event.

    Args:
        text: Preprocessed template input.

    Returns:
        The engine-sanitized text used for resource admission.
    """
    return _ReportingChunker()._sanitize_input(text, suppress_security_log=True)


def _execute_report(
    template: dict, text: str, options: dict | None = None
) -> tuple[ExecutionReport, str, bool]:
    mapped = template_from_record(template)
    chunker = _ReportingChunker()
    processor = TemplateProcessor(chunker=chunker)
    final_options = {**mapped.default_options, **(options or {})}
    pre_stage = next((s for s in mapped.stages if s.name == "preprocess"), None)
    chunk_stage = next(s for s in mapped.stages if s.name == "chunk")
    post_stage = next((s for s in mapped.stages if s.name == "postprocess"), None)
    pre_text, basis, pre_events = _run_pre_operations(
        processor, pre_stage.operations if pre_stage else [], text, final_options
    )
    data = processor._run_chunk_stage(
        {"text": pre_text, "chunks": [], "metadata": {}},
        chunk_stage,
        final_options,
        mapped.base_method,
    )
    transformed = (
        chunker.sanitized_text if chunker.sanitized_text is not None else pre_text
    )
    if transformed != pre_text:
        if basis == _SOURCE_BASIS:
            basis = "preprocessed:engine_sanitize"
        pre_events.append(
            {"operation": "engine_sanitize", "changed_text": True, "metadata": {}}
        )
    word_spacing_changed = (
        chunker.resolved_method == "words"
        and " ".join(transformed.split()) != transformed
    )
    records = []
    for index, raw in enumerate(data.get("chunks", [])):
        raw = {"text": raw, "metadata": {}} if isinstance(raw, str) else deepcopy(raw)
        chunk_text = _chunk_text_of(raw)
        metadata = raw.get("metadata", {})
        if not isinstance(metadata, dict):
            raise TemplateError("Engine chunk metadata must be an object")
        contributor = {
            "index": index,
            "metadata": metadata,
            "fields": {k: v for k, v in raw.items() if k not in ("text", "metadata")},
        }
        record = {
            "text": chunk_text,
            "metadata": metadata,
            "provenance": {
                "contributors": [contributor],
                "preprocessing": pre_events,
                "operations": [],
            },
        }
        # Word normalization can make one window equal a different source window.
        # Until the processor exposes originating windows, refuse these maps even
        # when the normalized output has one exact match elsewhere in the source.
        start = (
            transformed.find(chunk_text)
            if chunk_text and not word_spacing_changed
            else -1
        )
        if start >= 0 and transformed.find(chunk_text, start + 1) < 0:
            record["span"] = {
                "start": start,
                "end": start + len(chunk_text),
                "coordinate_space": "source" if transformed == text else "transformed",
            }
            record["provenance"]["mapping"] = {"status": "exact"}
        else:
            record["provenance"]["mapping"] = {
                "status": "unavailable",
                "reason": (
                    "Word chunking normalizes source whitespace; originating windows are unavailable"
                    if word_spacing_changed
                    else "Output is not a unique exact substring of processed text"
                ),
            }
        records.append(record)
    original_texts = [record["text"] for record in records]
    for stage_index, operation in enumerate(
        post_stage.operations if post_stage else []
    ):
        records = _run_post_operation(
            processor, records, operation, final_options, stage_index
        )
    for index, record in enumerate(records):
        record["provenance"].update(
            chunk_index=index,
            total_chunks=len(records),
            word_count=len(record["text"].split()),
        )
    report = ExecutionReport(
        chunks=tuple(records),
        transformed_text=transformed,
        diagnostics=({"kind": "preprocessing", "operations": pre_events},),
    )
    return report, basis, original_texts == [record["text"] for record in records]


def _run_post_operation(
    processor: TemplateProcessor,
    records: list[dict],
    operation: dict,
    options: dict,
    stage_index: int,
) -> list[dict]:
    """Execute the original text operation and attribute its structured effects.

    Attribution checks output against ordered inputs; it does not implement the
    operation's splitting, filtering, formatting, or merge threshold algorithm.
    """
    name = operation.get("type") or operation.get("operation")
    func = processor._operations.get(name)
    if func is None:
        return records  # legacy apply keeps parity's tolerant admission policy
    params = operation.get("params")
    config = {
        **options,
        **((operation.get("config") if params is None else params) or {}),
    }
    texts = func([record["text"] for record in records], config)
    if not isinstance(texts, list) or any(not isinstance(item, str) for item in texts):
        raise TemplateError(f"postprocessing.{name}: expected a list of text chunks")
    output = []
    cursor = 0
    for index, text in enumerate(texts):
        if name == "filter_empty":
            while cursor < len(records) and records[cursor]["text"] != text:
                cursor += 1
            if cursor >= len(records):
                raise TemplateError(
                    "postprocessing.filter_empty: cannot attribute output"
                )
            sources = [records[cursor]]
            cursor += 1
        elif name == "merge_small":
            start = cursor
            joined = ""
            while cursor < len(records):
                joined = (
                    joined + config.get("separator", "\n\n") if cursor > start else ""
                ) + records[cursor]["text"]
                cursor += 1
                if joined == text:
                    break
            if joined != text:
                raise TemplateError(
                    "postprocessing.merge_small: cannot attribute output without losing records"
                )
            sources = records[start:cursor]
        elif name in ("add_overlap", "add_metadata", "format_chunks") and len(
            texts
        ) == len(records):
            sources = (
                [records[index - 1], records[index]]
                if name == "add_overlap" and index > 0 and config.get("size", 50) > 0
                else [records[index]]
            )
        else:
            raise TemplateError(
                f"postprocessing.{name}: structured attribution unavailable"
            )
        primary = sources[-1] if name == "add_overlap" else sources[0]
        record = deepcopy(primary)
        record["text"] = text
        if len(sources) > 1:
            contributors = {
                item["index"]: item
                for source in sources
                for item in source["provenance"]["contributors"]
            }
            record["provenance"]["contributors"] = list(contributors.values())
            if name == "merge_small":
                record["metadata"] = {}  # each contributor retains its own metadata
            history = {
                (event["stage_index"], event["output_index"]): event
                for source in sources
                for event in source["provenance"]["operations"]
            }
            record["provenance"]["operations"] = [
                deepcopy(history[key]) for key in sorted(history)
            ]
        event = {
            "operation": name,
            "config": deepcopy(config),
            "stage_index": stage_index,
            "output_index": index,
            "input_indices": [
                item["index"]
                for source in sources
                for item in source["provenance"]["contributors"]
            ],
        }
        if name == "add_overlap" and len(sources) > 1:
            event["inserted_text"] = (
                text[: -len(primary["text"])] if primary["text"] else text
            )
        record["provenance"]["operations"].append(event)
        if len(sources) > 1 or text != primary["text"]:
            record.pop("span", None)
            record["provenance"]["mapping"] = {
                "status": "unavailable",
                "reason": f"postprocessing.{name} changed or combined text",
            }
        output.append(record)
    return output


def _chunk_text_of(chunk: Any) -> str:
    """Extract the text of a processor chunk (dict or bare string)."""
    if isinstance(chunk, dict):
        chunk = chunk.get("text")
    if not isinstance(chunk, str):
        raise TemplateError(
            "Engine chunk text must be a string; structured chunks cannot be stringified"
        )
    return chunk


def _run_pre_operations(
    processor: TemplateProcessor,
    operations: List[Dict[str, Any]],
    text: str,
    base_options: Dict[str, Any],
) -> tuple[str, str, list[dict]]:
    """Capture transformed text, first rewrite, and each operation's metadata.

    Reuses the processor's own operation registry (``processor._operations``,
    the seam this module owns) and mirrors ``_run_preprocess_stage``'s
    handling of both operation spellings and of dict results, so behavior is
    identical to running the stage through the processor — with the rewrite
    detection added. Unknown operations are skipped, exactly as upstream does.

    Returns:
        ``(transformed_text, offset_basis, events)`` where ``offset_basis`` is
        ``"source"`` or ``"preprocessed:<first-rewriting-op>"``.
    """
    transformed = text
    basis = _SOURCE_BASIS
    events = []
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
        events.append(
            {
                "operation": op_name,
                "changed_text": new_text != transformed,
                "metadata": deepcopy(result.get("metadata", {}))
                if isinstance(result, dict)
                else {},
            }
        )
        transformed = new_text
    return transformed, basis, events
