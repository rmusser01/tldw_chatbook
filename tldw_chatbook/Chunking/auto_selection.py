"""The auto-selection decision engine (spec §4.2) — the only module that
decides what "Auto" chunking means for a media item.

Three tiers, in order (ruling §8.2):

1. **Template tier** — score every live template's classifier block via the
   vendored ``TemplateClassifier``; the best strictly-positive score wins and
   the winning template is resolved through ``template_runtime``.
2. **Plan tier** — the vendored ``plan_auto_chunking`` derives method/size
   options from media type and goal.
3. **Plain tier** — ``chunk_options=None``; the caller keeps today's
   defaults. Auto cannot fail; it can only explain why it declined.

Where chatbook deliberately diverges from upstream (spec §0.2, rulings
§8.2/§8.6/§8.7 — the three load-bearing divergences):

1. **A winning template runs in full** — preprocessing, chunking,
   postprocessing — through the #2 template engine, exactly as a
   manually-picked template. Upstream's auto/explicit apply paths extract
   only the hierarchical block and silently drop every other stage
   (``UPSTREAM_DEFECTS.md`` #16); this module returns the whole resolved
   template so the apply path is indistinguishable from a manual pick.
2. **Auto rides the picker's ``chunk_template`` slot with the reserved
   sentinel name** ``"auto"`` (``AUTO_SENTINEL``) rather than upstream's
   separate form flag. The name is reserved at create/rename
   (``chunking_interop_library``), so no user template can be shadowed;
   a legacy row that already holds the name is flagged by the listing
   decoration (``name_reserved``) and skipped here by name.
3. **The template tier suppresses the planner** — a selected template's
   chunk-stage config *is* the plan. Upstream runs the planner and the
   template-apply helper as separate mechanisms; chatbook's chain is a
   composition, pinned by the never-runs test in
   ``Tests/Chunking/test_auto_selection.py``.

Threshold semantics (spec §0.1 correction): candidacy requires a
**strictly positive** score. The vendored ``score`` clamps internally with
the block's ``min_score`` (default 0.0), so a template with no classifier
block at all scores 0.0 and is never auto-selected (the six #2 built-ins
included — the opt-in), while a present block with absent ``min_score``
selects at any positive score, matching upstream parity.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Literal

from .engine.auto_planner import AutoChunkingDecision, plan_auto_chunking
from .engine.templates import TemplateClassifier
from .template_runtime import resolve_template

__all__ = [
    "AUTO_SENTINEL",
    "KNOWN_INGEST_MEDIA_TYPES",
    "MEDIA_TYPE_MAP",
    "AutoDecision",
    "resolve_auto",
]

#: The reserved name the picker's "Auto" option travels under in the
#: ``chunk_template`` slot (spec §4.3, ruling §8.7). Refused at
#: create/rename by ``chunking_interop_library``; a legacy row holding it
#: is flagged ``name_reserved`` by the listing decoration and never a
#: tier-1 candidate.
AUTO_SENTINEL = "auto"

#: Chatbook ingest media-type string → the planner's normalized vocabulary
#: (spec §5, implementation item §6.9). The frozen, verified table (Task 3):
#: every string chatbook's ingest paths can produce appears here — identity
#: entries included — so the planner's per-type plans can never be silently
#: disabled by a vocabulary mismatch. Anything NOT in the table still rides
#: through unchanged via the ``.get(media_type, media_type)`` call.
#:
#: The planner's own expectations (vendored ``engine/auto_planner.py``):
#: ``_normalize_media_type`` folds ``web_document/webpage/article/html`` to
#: ``"web"``; ``_choose_method`` then switches on ``ebook`` (chapter-based
#: ``ebook_chapters``), ``email`` (message-friendly sentences AND a fixed
#: 1000/150 sizing override), ``audio``/``video`` (transcript sentences),
#: and ``document``/``pdf``/``web`` (structure-aware → semantic → sentence
#: fallback); every other value gets the generic sentence plan. Mapping
#: targets are therefore exactly: those recognized types, or identity.
#:
#: Sources for every entry are named inline; the companion
#: ``Tests/Chunking/test_media_type_vocabulary.py`` fails loudly when either
#: side drifts.
MEDIA_TYPE_MAP: dict[str, str] = {
    # --- Local-file ingest family -------------------------------------------
    # Local_Ingestion/local_file_ingestion.py: detect_file_type() /
    # classify_ingest_source() results, persisted verbatim as Media.type by
    # the persist seam ("media_type": file_type). The planner-recognized
    # types pass through; html/article ride the planner's own web
    # normalization (mapped here explicitly so the table is total).
    "pdf": "pdf",
    "document": "document",
    "ebook": "ebook",
    "audio": "audio",
    "video": "video",
    "html": "web",
    "article": "web",
    "plaintext": "plaintext",  # no planner specialization → generic plan
    "image": "image",  # OCR text; no planner specialization → generic plan
    # --- Web-content aliases ------------------------------------------------
    # Names chatbook gives web-extracted content that the planner's own
    # normalization set does NOT cover — the load-bearing entries (spec §5:
    # without them these types silently get the generic plan forever):
    # "web_article" is the config-table key + URL-article job input family
    # (Media/local_media_reading_service.py accepts
    # article/web_article/webpage/html); "web_scraping" is the local
    # web-scraping result label; "webpage"/"web_document" are the planner's
    # own alias names (web_document kept from the Task-2 seed so a
    # planner-side name arriving via stored metadata also normalizes).
    "web_article": "web",
    "web_scraping": "web",
    "webpage": "web",
    "web_document": "web",
    # --- Media reading service family ---------------------------------------
    # Media/local_media_reading_service.py. "email" is planner-recognized
    # (message_boundaries + the 1000/150 sizing override); the rest have no
    # planner specialization and map to themselves intentionally:
    "email": "email",
    "code": "code",  # process_code
    "mediawiki_page": "mediawiki_page",  # ingest_mediawiki_dump persist
    "mediawiki_dump": "mediawiki_dump",  # dump-processing job/result label
    "reading": "reading",  # reading-import job label (rows keep origin_type)
    "media": "media",  # reprocess/sync job label (sink_type fallback)
    "unknown": "unknown",  # add_media/add_media_to_database normalization floor
    # --- Book family ---------------------------------------------------------
    # Local_Ingestion/Book_Ingestion_Lib.py: "book" is the text-file-as-book
    # persist; "zip" is the epub-container result-row label (the extracted
    # epubs persist as "ebook"). Neither has a planner equivalent → identity.
    "book": "book",
    "zip": "zip",
    # --- XML family -----------------------------------------------------------
    # Local_Ingestion/XML_Ingestion.py persists "xml_document"; "xml" is the
    # local-ingest config/dispatch family label (currently unreachable —
    # detect_file_type never returns it and the xml branch raises). No
    # planner equivalent → identity.
    "xml_document": "xml_document",
    "xml": "xml",
    # --- Notes-import family ---------------------------------------------------
    # DB/Client_Media_DB_v2.py save_obsidian_note() persists "obsidian_note"
    # through the same add_media_with_keywords seam; the Library viewer
    # recognizes the type (_MARKDOWN_MEDIA_TYPES). No planner equivalent →
    # identity (markdown-ish prose rides the generic sentence plan).
    "obsidian_note": "obsidian_note",
    # NOTE: Book_Ingestion_Lib._process_markup_or_plain_text's result dict
    # names "opml"/"text" (else "document") for markup family labels, but
    # that helper is dead code today — if it is ever revived, those strings
    # arrive unmapped and ride the identity fallback unchanged (generic
    # plan), which is the correct behavior; add explicit identity entries
    # only if that revival ever happens.
}

#: Every media-type string chatbook's own ingest code can produce or label
#: (the map's keys minus ``web_document``, which is the planner's own alias,
#: not a chatbook producer). Frozen (Task 3): a new ingest family without a
#: ``MEDIA_TYPE_MAP`` entry fails
#: ``Tests/Chunking/test_media_type_vocabulary.py`` loudly instead of
#: silently disabling tier-2 per-type plans (spec §6.9). Values not in this
#: tuple can still arrive via the passthrough surfaces (``add_media(media_type=…)``,
#: reading-import ``origin_type`` rows, chatbook-export imports) — those are
#: arbitrary caller data and ride the map's identity fallback by design.
KNOWN_INGEST_MEDIA_TYPES: tuple[str, ...] = (
    # local-file family
    "pdf",
    "document",
    "ebook",
    "audio",
    "video",
    "html",
    "article",
    "plaintext",
    "image",
    # web-content aliases
    "web_article",
    "web_scraping",
    "webpage",
    # media reading service family
    "email",
    "code",
    "mediawiki_page",
    "mediawiki_dump",
    "reading",
    "media",
    "unknown",
    # book family
    "book",
    "zip",
    # xml family
    "xml_document",
    "xml",
    # notes-import family
    "obsidian_note",
)

#: Spec §4.2 asks for the embeddings config's enabled state. No cheap
#: reader exists at this layer: the planner's ``semantic`` METHOD is the
#: engine's local ``SemanticChunkingStrategy``, registered unconditionally
#: in ``Chunker`` (never gated on an embeddings provider), and the
#: embeddings-config surfaces (``[embedding_config]``, RAG simplified
#: config) key on provider settings, not chunking-method availability.
#: Default True — the honest value for a local engine that always has the
#: strategy — and noted here for revisit if a config gate ever appears.
SEMANTIC_AVAILABLE_DEFAULT = True


@dataclass
class AutoDecision:
    """The outcome of one auto-selection (spec §4.2).

    Attributes:
        tier: Which tier won — ``"template"`` (a classifier block opted in
            and scored), ``"plan"`` (the vendored planner's options), or
            ``"plain"`` (no options; the caller keeps today's defaults).
        template: The resolved flat template dict — set ONLY on the
            template tier; the apply path runs it in full (divergence 1).
        chunk_options: The planner's derived options — set ONLY on the
            plan tier. ``None`` on template/plain tiers (a winning
            template IS the plan; plain means "change nothing").
        rationale: Short human-readable explanation lines of what won.
        fallback_reasons: Machine-ish tokens explaining every decline
            along the way (skipped candidates, tier falls).
    """

    tier: Literal["template", "plan", "plain"]
    template: dict[str, Any] | None = None
    chunk_options: dict[str, Any] | None = None
    rationale: list[str] = field(default_factory=list)
    fallback_reasons: list[str] = field(default_factory=list)


def resolve_auto(
    db: Any,
    *,
    media_type: str | None,
    title: str | None,
    filename: str | None,
    url: str | None,
    goal: str = "balanced",
) -> AutoDecision:
    """Decide the Auto chunking outcome for one media item (spec §4.2).

    Never raises for a selection outcome: every decline is a plain-tier
    decision with an explanation. Inputs come from the ingest job's
    already-known metadata — nothing re-reads file contents here.

    Args:
        db: Media DB handle exposing ``get_connection()`` (the template
            store). ``None`` means no store: tier 1 is vacuous.
        media_type: The item's chatbook media-type string.
        title: The item's title, if known (classifier ``title_regex``).
        filename: The item's filename, if known (``filename_regex``).
        url: The item's URL, if known (``url_regex``).
        goal: The planner goal; hardcoded ``"balanced"`` at the call sites
            (ruling §8.4 — no UI, no config key).

    Returns:
        The :class:`AutoDecision` for the item.
    """
    reasons: list[str] = []
    winner = _select_template(
        db, media_type=media_type, title=title, filename=filename, url=url, reasons=reasons
    )
    if winner is not None:
        score, priority, name = winner
        resolved = resolve_template(db, name)
        if resolved is not None:
            return AutoDecision(
                tier="template",
                template=resolved,
                chunk_options=None,
                rationale=[
                    (
                        f"Template '{name}' selected by its classifier block "
                        f"(score={score:.3f}, priority={priority}); a selected "
                        "template is the plan, so the planner does not run."
                    )
                ],
                fallback_reasons=reasons,
            )
        # Cannot happen for a validity-decorated survivor (the decoration
        # already parsed the body); guarded anyway — auto never raises.
        reasons.append(f"template_unresolvable:{name}")
    return _plan_or_plain(media_type=media_type, goal=goal, reasons=reasons)


def _select_template(
    db: Any,
    *,
    media_type: str | None,
    title: str | None,
    filename: str | None,
    url: str | None,
    reasons: list[str],
) -> tuple[float, int, str] | None:
    """Tier 1: pick the best live, valid, opted-in template.

    Iterates #2's listing surface (``LocalRAGAdminService.list_templates``
    — the interop's deleted-filtered listing decorated with the AC-24a
    ``template_valid`` flag and the ``name_reserved`` sentinel flag).
    Excludes stored-invalid rows (ruling §8.8) and reserved names, scores
    each survivor individually guarded, and keeps the best
    ``(score, priority)`` under strictly-greater comparison — ties keep
    the first-listed row (the listing is ordered ``is_builtin DESC,
    name ASC``, so the effective order is priority, then builtin, then
    name; spec §0.2 states the coupling).

    Returns:
        ``(score, priority, name)`` of the winner, or ``None`` when no
        candidate qualifies (declines appended to ``reasons``).
    """
    if db is None:
        reasons.append("template_store_unavailable")
        return None
    try:
        # Lazy: keeps this decision module out of RAG_Admin's import graph
        # until a decision is actually made (the template_runtime pattern).
        from ..RAG_Admin.local_rag_admin_service import LocalRAGAdminService

        listing = LocalRAGAdminService(db).list_templates()
    except Exception as exc:  # noqa: BLE001 — auto declines, never raises
        reasons.append(f"template_listing_error:{type(exc).__name__}")
        return None
    if not listing:
        reasons.append("template_store_empty")
        return None

    best: tuple[tuple[float, int], str] | None = None
    for record in listing:
        name = str(record.get("name") or "")
        if name == AUTO_SENTINEL:
            # Legacy sentinel-named row: flagged by the decoration, never
            # a candidate (never selected, never shadowed — AC 14).
            reasons.append(f"template_name_reserved:{name}")
            continue
        if record.get("template_valid") is False:
            # Ruling 8.8: auto must never pick a body the apply path
            # would then refuse.
            reasons.append(f"template_invalid:{name}")
            continue
        try:
            raw = record.get("template_json")
            body = json.loads(raw) if isinstance(raw, str) else raw
            if not isinstance(body, dict):
                raise TypeError(f"body is a {type(body).__name__}, not an object")
            score = float(
                TemplateClassifier.score(
                    body, media_type=media_type, title=title, url=url, filename=filename
                )
            )
            priority = _classifier_priority(body)
        except Exception as exc:  # noqa: BLE001 — one malformed candidate
            # is skipped with a reason, never fatal (spec §4.2/§5).
            reasons.append(f"template_classifier_error:{name}:{type(exc).__name__}")
            continue
        if score <= 0:
            # Spec §0.1 correction 1 + upstream's own guard: no-block
            # templates score 0.0 and are never candidates.
            continue
        key = (score, priority)
        if best is None or key > best[0]:
            best = (key, name)
    if best is None:
        reasons.append("no_positive_template_score")
        return None
    (score, priority), name = best
    return score, priority, name


def _classifier_priority(body: dict[str, Any]) -> int:
    """The classifier block's ``priority`` (absent/invalid → 0).

    Mirrors the classifier-block lookup ``TemplateClassifier.score`` uses
    (top-level ``classifier`` first, then ``chunking.config.classifier``).
    """
    classifier = body.get("classifier")
    if not isinstance(classifier, dict):
        chunking = body.get("chunking")
        config = chunking.get("config") if isinstance(chunking, dict) else None
        classifier = config.get("classifier") if isinstance(config, dict) else None
    if isinstance(classifier, dict):
        try:
            return int(classifier.get("priority") or 0)
        except (TypeError, ValueError):
            return 0
    return 0


def _plan_or_plain(
    *, media_type: str | None, goal: str, reasons: list[str]
) -> AutoDecision:
    """Tiers 2 and 3: the vendored planner, then the plain floor.

    The planner call pins the caller contract (spec §4.2 / AC 5):
    ``perform_chunking=True`` + ``chunking_mode="auto"``, goal rides
    through, ``requested_llm=False`` and ``llm_available=False`` (LLM
    boundary work is #6's, not ours), ``semantic_available`` per
    ``SEMANTIC_AVAILABLE_DEFAULT`` above. Template-status args stay unset —
    no template reached the planner.
    """
    try:
        decision: AutoChunkingDecision = plan_auto_chunking(
            perform_chunking=True,
            chunking_mode="auto",
            goal=goal,
            media_type=MEDIA_TYPE_MAP.get(media_type, media_type),
            requested_llm=False,
            llm_available=False,
            semantic_available=SEMANTIC_AVAILABLE_DEFAULT,
        )
    except Exception as exc:  # noqa: BLE001 — auto declines, never raises
        reasons.append(f"planner_error:{type(exc).__name__}")
        return AutoDecision(
            tier="plain",
            rationale=["The auto planner failed; keeping current chunking defaults."],
            fallback_reasons=reasons,
        )
    if decision.chunk_options is None:
        reasons.append("planner_declined")
        return AutoDecision(
            tier="plain",
            rationale=[
                (
                    "The auto planner declined to produce options; "
                    "keeping current chunking defaults."
                )
            ],
            fallback_reasons=reasons,
        )
    plan_meta = decision.chunking_plan or {}
    rationale: list[str] = []
    plan_rationale = plan_meta.get("rationale")
    if plan_rationale:
        rationale.append(str(plan_rationale))
    fallback_reason = plan_meta.get("fallback_reason")
    if fallback_reason:
        reasons.extend(part for part in str(fallback_reason).split(";") if part)
    return AutoDecision(
        tier="plan",
        chunk_options=dict(decision.chunk_options),
        rationale=rationale,
        fallback_reasons=reasons,
    )
