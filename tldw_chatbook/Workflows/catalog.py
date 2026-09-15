"""Five explicit local operation subsets; preservation is not adapter support."""

from dataclasses import dataclass

CONTRACT_REVISION = "local-file-to-note/1"

# Pinned server dev 6cd2745f696af04668a61c20b84ab8a9e69ca5e4;
# 2026-09-08 approved parity matrix. Inventory only, never execution support.
DISCOVERY = (
    ("media_ingest", "Media and documents", "V1 minimum"),
    ("prompt", "Templates and text/data", "V1 minimum"),
    ("llm", "LLM and generated content", "V1 minimum"),
    ("rag_search", "Search and RAG", "V1 minimum"),
    ("kanban", "Knowledge and domain records", "Sequential parity candidate"),
    ("mcp_tool", "Tools and external integrations", "V1 minimum"),
    ("acp_stage", "Tools and external integrations", "Separate lifecycle design"),
    ("tts", "Audio/video", "V1 minimum"),
    ("webhook", "Tools and external integrations", "Sequential parity candidate"),
    ("delay", "Runtime and orchestration", "V1 minimum"),
    ("log", "Runtime and orchestration", "V1 minimum"),
    ("wait_for_human", "Runtime and orchestration", "V1 minimum"),
    ("wait_for_approval", "Runtime and orchestration", "V1 minimum"),
    ("branch", "Runtime and orchestration", "V2"),
    ("map", "Runtime and orchestration", "V3"),
    ("process_media", "Media and documents", "Sequential parity candidate"),
    ("policy_check", "LLM and generated content", "Sequential parity candidate"),
    ("rss_fetch", "Tools and external integrations", "Sequential parity candidate"),
    ("atom_fetch", "Tools and external integrations", "Sequential parity candidate"),
    ("embed", "Search and RAG", "Sequential parity candidate"),
    ("translate", "LLM and generated content", "Sequential parity candidate"),
    ("stt_transcribe", "Audio/video", "V1 minimum"),
    ("notify", "Tools and external integrations", "Sequential parity candidate"),
    ("diff_change_detector", "Templates and text/data", "Sequential parity candidate"),
    ("notes", "Knowledge and domain records", "V1 minimum"),
    ("prompts", "Knowledge and domain records", "V1 minimum"),
    ("chunking", "Search and RAG", "V1 minimum"),
    ("web_search", "Search and RAG", "Sequential parity candidate"),
    ("collections", "Knowledge and domain records", "Sequential parity candidate"),
    ("chatbooks", "Knowledge and domain records", "Sequential parity candidate"),
    ("evaluations", "Knowledge and domain records", "Sequential parity candidate"),
    ("claims_extract", "Knowledge and domain records", "Sequential parity candidate"),
    ("character_chat", "LLM and generated content", "Sequential parity candidate"),
    ("moderation", "LLM and generated content", "Sequential parity candidate"),
    ("sandbox_exec", "Tools and external integrations", "Separate lifecycle design"),
    ("image_gen", "LLM and generated content", "Sequential parity candidate"),
    ("summarize", "LLM and generated content", "Sequential parity candidate"),
    ("query_expand", "Search and RAG", "Sequential parity candidate"),
    ("rerank", "Search and RAG", "Sequential parity candidate"),
    ("citations", "Search and RAG", "Sequential parity candidate"),
    ("ocr", "Media and documents", "Sequential parity candidate"),
    ("pdf_extract", "Media and documents", "V1 minimum"),
    ("voice_intent", "Audio/video", "Sequential parity candidate"),
    ("query_rewrite", "Search and RAG", "Sequential parity candidate"),
    ("hyde_generate", "Search and RAG", "Sequential parity candidate"),
    ("semantic_cache_check", "Search and RAG", "Sequential parity candidate"),
    ("search_aggregate", "Search and RAG", "Sequential parity candidate"),
    ("entity_extract", "Search and RAG", "Sequential parity candidate"),
    ("bibliography_generate", "Search and RAG", "Sequential parity candidate"),
    ("document_table_extract", "Media and documents", "Sequential parity candidate"),
    ("audio_diarize", "Audio/video", "Sequential parity candidate"),
    ("flashcard_generate", "LLM and generated content", "Sequential parity candidate"),
    ("quiz_generate", "LLM and generated content", "Sequential parity candidate"),
    ("quiz_evaluate", "LLM and generated content", "Sequential parity candidate"),
    ("outline_generate", "LLM and generated content", "Sequential parity candidate"),
    ("glossary_extract", "LLM and generated content", "Sequential parity candidate"),
    ("mindmap_generate", "LLM and generated content", "Sequential parity candidate"),
    ("eval_readability", "LLM and generated content", "Sequential parity candidate"),
    ("json_transform", "Templates and text/data", "Sequential parity candidate"),
    ("json_validate", "Templates and text/data", "V1 minimum"),
    ("csv_to_json", "Templates and text/data", "V1 minimum"),
    ("json_to_csv", "Templates and text/data", "V1 minimum"),
    ("regex_extract", "Templates and text/data", "V1 minimum"),
    ("text_clean", "Templates and text/data", "V1 minimum"),
    ("xml_transform", "Templates and text/data", "Sequential parity candidate"),
    ("template_render", "Templates and text/data", "V1 minimum"),
    ("batch", "Templates and text/data", "Sequential parity candidate"),
    ("workflow_call", "Runtime and orchestration", "Separate lifecycle design"),
    ("parallel", "Runtime and orchestration", "V3"),
    ("cache_result", "Runtime and orchestration", "Separate lifecycle design"),
    ("retry", "Runtime and orchestration", "Separate lifecycle design"),
    ("checkpoint", "Runtime and orchestration", "Separate lifecycle design"),
    ("s3_upload", "Tools and external integrations", "Sequential parity candidate"),
    ("s3_download", "Tools and external integrations", "Sequential parity candidate"),
    (
        "github_create_issue",
        "Tools and external integrations",
        "Sequential parity candidate",
    ),
    (
        "podcast_rss_publish",
        "Tools and external integrations",
        "Sequential parity candidate",
    ),
    ("llm_with_tools", "LLM and generated content", "Sequential parity candidate"),
    ("llm_critique", "LLM and generated content", "Sequential parity candidate"),
    ("context_build", "LLM and generated content", "Sequential parity candidate"),
    ("document_merge", "Templates and text/data", "Sequential parity candidate"),
    ("document_diff", "Templates and text/data", "Sequential parity candidate"),
    ("markdown_to_html", "Templates and text/data", "Sequential parity candidate"),
    ("html_to_markdown", "Templates and text/data", "Sequential parity candidate"),
    ("keyword_extract", "Templates and text/data", "Sequential parity candidate"),
    ("sentiment_analyze", "Templates and text/data", "Sequential parity candidate"),
    ("language_detect", "Templates and text/data", "Sequential parity candidate"),
    ("topic_model", "Templates and text/data", "Sequential parity candidate"),
    ("token_count", "Templates and text/data", "Sequential parity candidate"),
    ("context_window_check", "Templates and text/data", "Sequential parity candidate"),
    ("llm_compare", "LLM and generated content", "Sequential parity candidate"),
    ("image_describe", "LLM and generated content", "Sequential parity candidate"),
    ("report_generate", "LLM and generated content", "Sequential parity candidate"),
    ("newsletter_generate", "LLM and generated content", "Sequential parity candidate"),
    (
        "audio_briefing_compose",
        "LLM and generated content",
        "Sequential parity candidate",
    ),
    ("slides_generate", "LLM and generated content", "Sequential parity candidate"),
    ("diagram_generate", "LLM and generated content", "Sequential parity candidate"),
    ("email_send", "Tools and external integrations", "Sequential parity candidate"),
    (
        "screenshot_capture",
        "Tools and external integrations",
        "Sequential parity candidate",
    ),
    ("schedule_workflow", "Runtime and orchestration", "Separate lifecycle design"),
    ("timing_start", "Runtime and orchestration", "Sequential parity candidate"),
    ("timing_stop", "Runtime and orchestration", "Sequential parity candidate"),
    ("multi_voice_tts", "Audio/video", "Sequential parity candidate"),
    ("audio_normalize", "Audio/video", "Sequential parity candidate"),
    ("audio_concat", "Audio/video", "Sequential parity candidate"),
    ("audio_trim", "Audio/video", "Sequential parity candidate"),
    ("audio_convert", "Audio/video", "Sequential parity candidate"),
    ("audio_extract", "Audio/video", "Sequential parity candidate"),
    ("audio_mix", "Audio/video", "Sequential parity candidate"),
    ("video_thumbnail", "Audio/video", "Sequential parity candidate"),
    ("video_trim", "Audio/video", "Sequential parity candidate"),
    ("video_concat", "Audio/video", "Sequential parity candidate"),
    ("video_convert", "Audio/video", "Sequential parity candidate"),
    ("video_extract_frames", "Audio/video", "Sequential parity candidate"),
    ("subtitle_generate", "Audio/video", "Sequential parity candidate"),
    ("subtitle_translate", "Audio/video", "Sequential parity candidate"),
    ("subtitle_burn", "Audio/video", "Sequential parity candidate"),
    ("arxiv_search", "Research", "Sequential parity candidate"),
    ("arxiv_download", "Research", "Sequential parity candidate"),
    ("pubmed_search", "Research", "Sequential parity candidate"),
    ("semantic_scholar_search", "Research", "Sequential parity candidate"),
    ("google_scholar_search", "Research", "Sequential parity candidate"),
    ("patent_search", "Research", "Sequential parity candidate"),
    ("doi_resolve", "Research", "Sequential parity candidate"),
    ("reference_parse", "Research", "Sequential parity candidate"),
    ("bibtex_generate", "Research", "Sequential parity candidate"),
    ("literature_review", "Research", "Sequential parity candidate"),
    ("deep_research", "Research", "Sequential parity candidate"),
    ("deep_research_wait", "Research", "Sequential parity candidate"),
    ("deep_research_load_bundle", "Research", "Sequential parity candidate"),
    ("deep_research_select_bundle_fields", "Research", "Sequential parity candidate"),
)


@dataclass(frozen=True)
class StepContract:
    """Bundled local-subset field and effect description, not execution proof.

    Attributes:
        step_type: Canonical server step type.
        fields: Configuration fields exposed by the local subset.
        output_fields: Named outputs described by that subset.
        effects: Effect categories requiring execution-time admission.
    """

    step_type: str
    fields: tuple[str, ...]
    output_fields: tuple[str, ...]
    effects: tuple[str, ...]


STEP_CONTRACTS = (
    StepContract(
        "media_ingest",
        ("sources", "extraction"),
        ("text", "media_ids", "metadata", "transcripts", "rag_indexed"),
        ("file_read",),
    ),
    StepContract("prompt", ("template",), ("text",), ()),
    StepContract(
        "llm",
        ("provider", "model", "prompt", "max_tokens"),
        ("text",),
        ("local_model",),
    ),
    StepContract(
        "wait_for_human",
        ("instructions", "assigned_to_user_id", "timeout_seconds"),
        ("text", "decision"),
        ("human_review",),
    ),
    StepContract(
        "notes", ("action", "title", "content"), ("note", "success"), ("note_create",)
    ),
)


@dataclass(frozen=True)
class FieldSpec:
    """Canonical config pointer relative to a step, with documented value type."""

    path: str
    label: str
    value_type: str = "string"
    section: str = "action"
    required: bool = True
    multiline: bool = False


FIELDS = {
    "media_ingest": (
        FieldSpec("config/sources/0/uri", "UTF-8 text file URI", section="inputs"),
        FieldSpec("config/extraction/extract_text", "Extract text (true)", "boolean"),
    ),
    "prompt": (FieldSpec("config/template", "Template", multiline=True),),
    "llm": (
        FieldSpec("config/provider", "Provider", section="inputs"),
        FieldSpec("config/model", "Model", section="inputs"),
        FieldSpec("config/prompt", "Prompt", multiline=True),
        FieldSpec("config/max_tokens", "Maximum output tokens", "integer"),
    ),
    "wait_for_human": (
        FieldSpec("config/assigned_to_user_id", "Review actor", section="inputs"),
        FieldSpec("config/instructions", "Review instructions", multiline=True),
        FieldSpec(
            "config/timeout_seconds", "Human response deadline (seconds)", "integer"
        ),
    ),
    "notes": (
        FieldSpec("config/action", "Action (create)"),
        FieldSpec("config/title", "Note title"),
        FieldSpec("config/content", "Note content", multiline=True),
    ),
}

_OUTPUT_TYPES = {
    # Pinned adapter fixture outputs, not types inferred from arbitrary values.
    "media_ingest": (
        ("text", "string"),
        ("media_ids", "array"),
        ("metadata", "object"),
        ("transcripts", "array"),
        ("rag_indexed", "boolean"),
    ),
    "prompt": (("text", "string"),),
    "llm": (("text", "string"),),
    # The actor supplies edited_fields; text is NOT guaranteed by the server or
    # the local decision contract. Expose it with explicit runtime validation.
    "wait_for_human": (("text", "unverified"), ("decision", "string")),
    "notes": (("note", "object"), ("success", "boolean")),
}


def output_types(step_type: str) -> tuple[tuple[str, str], ...]:
    """Documented output paths/types; an unknown contract promises no fields."""
    return _OUTPUT_TYPES.get(step_type, ())


def new_step(step_type: str) -> dict:
    """Create only one of the five authorable subsets; missing setup stays blank."""
    from copy import deepcopy

    configs = {
        "media_ingest": {
            "sources": [{"uri": ""}],
            "extraction": {"extract_text": True},
        },
        "prompt": {"template": ""},
        "llm": {"provider": "", "model": "", "prompt": "", "max_tokens": 512},
        "wait_for_human": {
            "instructions": "",
            "assigned_to_user_id": "",
            "timeout_seconds": 3600,
        },
        "notes": {"action": "create", "title": "", "content": ""},
    }
    if step_type not in configs:
        raise ValueError("This step type is unavailable for local authoring")
    return {
        "type": step_type,
        "retry": 0,
        "timeout_seconds": 300,
        "config": deepcopy(configs[step_type]),
    }


@dataclass(frozen=True)
class DiscoveryEntry:
    """Searchable presentation of a bundled canonical step type.

    Attributes:
        step_type: Canonical server step type.
        family: Task-oriented group used by the chooser.
        disposition: Current local support classification.
        label: Human-readable action name.
        example: Brief example of the action's inputs and outputs.
        available: Whether the authoring chooser permits inserting this type.
    """

    step_type: str
    family: str
    disposition: str
    label: str
    example: str
    available: bool


_LABELS = {
    "media_ingest": ("Read a local text file", "file URI → text", "Sources and data"),
    "prompt": ("Render text", "template → text", "Text and models"),
    "llm": ("Call model", "prompt → generated text", "Text and models"),
    "wait_for_human": (
        "Review with a human",
        "instructions → reviewed fields",
        "Run controls",
    ),
    "notes": ("Create a note", "title + content → note", "Tools and outputs"),
}


def discover(*, show_all: bool = False, query: str = "") -> tuple[DiscoveryEntry, ...]:
    """Read bundled pinned inventory; discovery never performs I/O or admission."""
    entries = []
    for step_type, family, disposition in DISCOVERY:
        available = step_type in _LABELS
        label, example, group = _LABELS.get(
            step_type,
            (step_type.replace("_", " ").capitalize(), "Schema unverified", family),
        )
        if (show_all or available) and query.casefold() in (
            label + " " + step_type + " " + group
        ).casefold():
            entries.append(
                DiscoveryEntry(step_type, group, disposition, label, example, available)
            )
    return tuple(
        sorted(entries, key=lambda item: (not item.available, item.family, item.label))
    )
