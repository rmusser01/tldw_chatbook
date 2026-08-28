"""Setting-level search index for the Settings hub.

task-1715 introduced field-level "/" search (typing a setting's visible
label surfaces its category and Enter focuses the field). TASK-23109 moved
the index here and completed its coverage: every rendered value-editing
control in a category's detail pane must have a row, and the drift test
``test_every_rendered_setting_is_in_the_search_index`` (in
``Tests/UI/test_settings_search_index.py``) fails when a rendered setting
is missing, so the hand-maintained table cannot silently rot.

Labels mirror the visible row labels; a field may carry several rows when
users know it by more than one name.
"""

from __future__ import annotations

from .settings_config_models import SettingsCategoryId

#: Library reader destinations rendered as per-destination Items pane/width
#: rows on the Appearance category (shared with settings_screen).
LIBRARY_READER_DESTINATIONS = (
    ("media", "Media"),
    ("conversations", "Conversations"),
    ("notes", "Notes"),
    ("prompts", "Prompts"),
    ("skills", "Skills"),
)

# Task 3 (541 v2 UX AC3): RAG widget id -> guidance-group key. Mirrors the
# ids `_library_rag_field_selector` and the LIBRARY_RAG compose branch mint
# (search around "settings-library-rag-" in settings_screen.py). Used by
# `_rag_field_guidance_rows()` so the Scope Inspector follows the focused
# field; falls back to `_active_rag_scope_group` (the last-expanded
# Collapsible) when the focused widget isn't one of these.
RAG_FIELD_GROUP_BY_ID: dict[str, str] = {
    "settings-library-rag-search-mode": "search",
    "settings-library-rag-default-top-k": "search",
    "settings-library-rag-fts-top-k": "search",
    "settings-library-rag-vector-top-k": "search",
    "settings-library-rag-hybrid-alpha": "search",
    "settings-library-rag-score-threshold": "search",
    "settings-library-rag-include-citations": "search",
    "settings-library-rag-direct-library-tools": "search",
    "settings-library-rag-citation-style": "search",
    "settings-library-rag-snippet-max-chars": "search",
    "settings-library-rag-max-context-size": "search",
    "settings-library-rag-embedding-model": "embedding",
    "settings-library-rag-embedding-device": "embedding",
    "settings-library-rag-embedding-batch-size": "embedding",
    "settings-library-rag-embedding-max-length": "embedding",
    "settings-library-rag-chunk-size": "chunking",
    "settings-library-rag-chunk-overlap": "chunking",
    "settings-library-rag-chunking-method": "chunking",
    "settings-library-rag-distance-metric": "vector_store",
    "settings-library-rag-enable-reranking": "reranking",
    "settings-library-rag-reranker-provider": "reranking",
    "settings-library-rag-reranker-model": "reranking",
    "settings-library-rag-reranker-top-k": "reranking",
    "settings-library-rag-profile-select": "profile",
    "settings-library-rag-profile-set-active": "profile",
    "settings-library-rag-profile-clone": "profile",
    "settings-library-rag-profile-rename": "profile",
    "settings-library-rag-profile-delete": "profile",
    "settings-library-rag-index-backfill": "index",
}


def _rag_field_search_label(field_id: str) -> str:
    """Human label for a RAG field id (task-1715 field-level search).

    Args:
        field_id: A ``settings-library-rag-*`` widget id.

    Returns:
        The id suffix as a spaced title, e.g. "hybrid alpha".
    """
    return field_id.removeprefix("settings-library-rag-").replace("-", " ")


#: task-1715: field-level search index -- "/" previously matched only
#: category names/descriptions/owned keys, so "threshold" found nothing
#: on a 23-category screen (critique r4 P1). Labels mirror the visible
#: row labels; Enter focuses the matched field.
FIELD_SEARCH_INDEX: dict[SettingsCategoryId, tuple[tuple[str, str], ...]] = {}


def _backend_field_entries(
    id_prefix: str,
    backend_labels: dict[str, str],
    field_schema: dict[str, tuple],
) -> tuple[tuple[str, str], ...]:
    """Index rows for a generation category's per-backend editor fields.

    Derived from the same ``FIELD_SCHEMA``/``BACKEND_LABELS`` tables the form
    builders render from (TASK-23109's durability requirement), so a new
    backend or field becomes searchable without touching this module.

    Args:
        id_prefix: "imagegen" or "videogen" (the widget-id family).
        backend_labels: ``BACKEND_LABELS`` from the category's helper module.
        field_schema: ``FIELD_SCHEMA`` from the same module.

    Returns:
        ``(widget_id, label)`` rows, backend label prefixed for scope.
    """
    entries: list[tuple[str, str]] = []
    for backend_id, backend_label in backend_labels.items():
        entries.append(
            (
                f"settings-{id_prefix}-enabled-{backend_id}",
                f"{backend_label} backend enabled",
            )
        )
        for spec in field_schema.get(backend_id, ()):
            entries.append(
                (
                    f"settings-{id_prefix}-field-{backend_id}-{spec.toml_key}",
                    f"{backend_label} {spec.label}",
                )
            )
    return tuple(entries)


def build_field_search_index() -> None:
    """(Re)build ``FIELD_SEARCH_INDEX`` from its per-category tables."""
    from ...LLM_Provider_Catalog.model_catalog_settings import (
        AUTO_REFRESH_PROVIDER_LIST_KEYS,
    )
    from .settings_image_gen_defaults import (
        _GLOBAL_FLOAT_FIELD_SPECS as _imagegen_float_specs,
        _GLOBAL_INT_FIELD_SPECS as _imagegen_int_specs,
        BACKEND_LABELS as _imagegen_backend_labels,
        FIELD_SCHEMA as _imagegen_field_schema,
    )
    from .settings_storage_defaults import STORAGE_FIELD_LABELS as _labels
    from .settings_video_gen_defaults import (
        _GLOBAL_INT_FIELD_SPECS as _videogen_int_specs,
        BACKEND_LABELS as _videogen_backend_labels,
        FIELD_SCHEMA as _videogen_field_schema,
    )

    FIELD_SEARCH_INDEX.update(
        {
            SettingsCategoryId.CONSOLE_BEHAVIOR: (
                (
                    "settings-console-show-model-thinking",
                    "Show model thinking",
                ),
                (
                    "settings-console-rail-layout-scope",
                    "Rail layout scope",
                ),
                (
                    "settings-console-rail-layout-scope",
                    "Global per workspace layout scope",
                ),
                (
                    "settings-console-stack-collapsed-rail-labels",
                    "Stack collapsed rail labels",
                ),
                (
                    "settings-console-stack-collapsed-rail-labels",
                    "Rail handle presentation",
                ),
                (
                    "settings-console-stack-collapsed-rail-labels",
                    "Stacked vertical Context Inspector",
                ),
                (
                    "settings-console-status-row-position-toggle",
                    "Status row placement",
                ),
                (
                    "settings-console-status-row-position-toggle",
                    "Status chips above below composer",
                ),
                ("settings-console-paste-collapse-threshold", "Threshold (chars)"),
                ("settings-console-max-parallel-runs", "Max parallel agent runs"),
                ("settings-console-tool-result-display-chars", "Display cap (chars)"),
                ("settings-console-sidechat-model", "Side chat model"),
                (
                    "settings-console-sidechat-prompt-template",
                    "Side chat prompt template",
                ),
                (
                    "settings-console-sidechat-prompt-template",
                    "More Details prompt",
                ),
                (
                    "settings-console-context-budget-mode",
                    "Conversation budget strategy",
                ),
                ("settings-console-context-budget-tokens", "Conversation max tokens"),
                ("settings-console-context-compaction-mode", "When limit nears"),
                (
                    "settings-console-context-compaction-representation",
                    "Compaction representation",
                ),
                (
                    "settings-console-context-trigger-percent",
                    "Compact at percent",
                ),
                (
                    "settings-console-context-target-percent",
                    "Reduce conversation to percent",
                ),
                (
                    "settings-console-context-summary-max-tokens",
                    "Summary response max tokens",
                ),
                (
                    "settings-console-context-failure-behavior",
                    "If compaction fails",
                ),
                (
                    "settings-console-context-carry-forward-mode",
                    "Keep after compaction",
                ),
                # TASK-23109 completion sweep: labels mirror the rendered rows.
                (
                    "settings-console-exchange-capture-enabled",
                    "Capture future provider exchanges",
                ),
                ("settings-console-exchange-capture-detail", "Capture detail"),
                (
                    "settings-console-collapse-large-pastes-toggle",
                    "Collapse large pastes",
                ),
                ("settings-console-agent-max-total-tokens", "Token budget (per run)"),
                (
                    "settings-console-agent-max-wall-seconds",
                    "Wall-clock limit (seconds)",
                ),
                (
                    "settings-console-agent-max-tool-call-seconds",
                    "Per-tool-call limit (seconds)",
                ),
                ("settings-console-agent-max-model-turns", "Model turns (backstop)"),
                ("settings-console-agent-max-steps", "Steps (backstop)"),
                (
                    "settings-console-default-user-display-name",
                    "Default chat display name",
                ),
                ("settings-console-default-streaming", "Streaming"),
                ("settings-console-default-temperature", "Temperature"),
                ("settings-console-default-top-p", "Top P"),
                ("settings-console-default-min-p", "Min P"),
                ("settings-console-default-top-k", "Top K"),
                ("settings-console-default-max-tokens", "Response max tokens"),
                ("settings-console-default-seed", "Seed"),
                ("settings-console-default-presence-penalty", "Presence penalty"),
                ("settings-console-default-frequency-penalty", "Frequency penalty"),
                ("settings-console-default-reasoning-effort", "Reasoning effort"),
                ("settings-console-default-reasoning-summary", "Reasoning summary"),
                ("settings-console-default-verbosity", "Verbosity"),
                ("settings-console-default-thinking-effort", "Thinking effort"),
                (
                    "settings-console-default-thinking-budget-tokens",
                    "Thinking budget tokens",
                ),
                (
                    "settings-console-background-effect-enabled",
                    "Enable background effects",
                ),
                ("settings-console-background-effect-type", "Background effect"),
                ("settings-console-background-effect-scope", "Background effect scope"),
                (
                    "settings-console-background-effect-intensity",
                    "Background effect intensity",
                ),
                ("settings-console-background-effect-fps", "Background effect frame rate"),
            ),
            SettingsCategoryId.APPEARANCE: (
                ("settings-appearance-theme", "Theme"),
                ("settings-appearance-palette-theme-limit", "Palette limit (themes)"),
                ("settings-appearance-font-size", "Web font size (px)"),
                ("settings-appearance-density", "Density"),
                ("settings-appearance-transcript-style", "Console transcript"),
                ("settings-appearance-animations-enabled", "Animations"),
                # TASK-23109: the setting the critique could not find by name.
                ("settings-appearance-reduce-motion", "Reduce motion"),
                ("settings-appearance-ascii-glyphs", "ASCII glyphs"),
                ("settings-appearance-smooth-scrolling", "Smooth scrolling"),
                (
                    "settings-appearance-library-media-library-open",
                    "Shared Library rail",
                ),
                (
                    "settings-appearance-library-media-custom-widths",
                    "Shared Library rail width mode",
                ),
                (
                    "settings-appearance-library-media-library-width",
                    "Preferred Library rail width",
                ),
                *(
                    (
                        f"settings-appearance-library-{destination}-items-{suffix}",
                        f"{label} Items {description}",
                    )
                    for destination, label in LIBRARY_READER_DESTINATIONS
                    for suffix, description in (
                        ("open", "pane"),
                        ("width", "width"),
                    )
                ),
            ),
            SettingsCategoryId.PROVIDERS_MODELS: (
                ("settings-provider-value", "Provider"),
                ("settings-provider-api-mode", "API mode"),
                (
                    "settings-provider-api-mode",
                    "api_settings.<provider>.api_mode",
                ),
                ("settings-model-value", "Model"),
                ("settings-provider-endpoint-value", "Endpoint"),
                ("settings-provider-api-key", "API key"),
                ("settings-provider-credential-env-var", "Credential env var"),
                ("settings-model-context-window", "Model context window tokens"),
                # TASK-23109 completion sweep: model-catalog refresh controls.
                (
                    "settings-model-catalog-auto-refresh",
                    "Auto-refresh model lists on startup",
                ),
                ("settings-model-catalog-stale-hours", "Refresh after (hours)"),
                *(
                    entry
                    for provider in AUTO_REFRESH_PROVIDER_LIST_KEYS
                    for entry in (
                        (
                            f"settings-mc-auto-{provider.lower()}",
                            f"{provider} auto-refresh model list",
                        ),
                        (
                            f"settings-mc-write-{provider.lower()}",
                            f"{provider} save fetched models to config",
                        ),
                    )
                ),
                # Model profile (per provider+model) sampling overrides.
                ("settings-model-profile-temperature", "Temperature"),
                ("settings-model-profile-top-p", "Top P"),
                ("settings-model-profile-min-p", "Min P"),
                ("settings-model-profile-top-k", "Top K"),
                ("settings-model-profile-max-tokens", "Response max tokens"),
                ("settings-model-profile-seed", "Seed"),
                ("settings-model-profile-presence-penalty", "Presence penalty"),
                ("settings-model-profile-frequency-penalty", "Frequency penalty"),
                ("settings-model-profile-reasoning-effort", "Reasoning effort"),
                ("settings-model-profile-reasoning-summary", "Reasoning summary"),
                ("settings-model-profile-verbosity", "Verbosity"),
                ("settings-model-profile-thinking-effort", "Thinking effort"),
                (
                    "settings-model-profile-thinking-budget-tokens",
                    "Thinking budget tokens",
                ),
                ("settings-model-profile-streaming", "Streaming"),
            ),
            SettingsCategoryId.SPEECH_TTS: (
                ("settings-speech-default-provider", "Default TTS Provider"),
                ("settings-speech-model-value", "TTS model"),
                ("settings-speech-voice-value", "TTS voice"),
                ("settings-speech-configure-provider", "audio.cpp audio_cpp"),
                ("settings-speech-configure-provider", "OpenAI"),
                ("settings-speech-configure-provider", "ElevenLabs"),
                ("settings-speech-configure-provider", "Kokoro"),
                ("settings-speech-configure-provider", "Chatterbox"),
                ("settings-speech-configure-provider", "Higgs"),
                ("settings-speech-configure-provider", "AllTalk"),
                # TASK-23109 completion sweep: labels mirror the rendered rows.
                ("settings-speech-default-profile", "Default voice profile"),
                ("settings-speech-model-policy", "Model policy"),
                ("settings-speech-voice-policy", "Voice policy"),
                ("settings-speech-output-format", "Output format"),
                ("settings-speech-speed", "Speech speed"),
                ("settings-speech-openai-authentication-mode", "OpenAI authentication"),
                ("settings-speech-openai-base-url", "OpenAI TTS base URL"),
                ("settings-speech-openai-organization-id", "OpenAI organization ID"),
                (
                    "settings-speech-realtime-enabled",
                    "Enable realtime voice engine",
                ),
                ("settings-speech-realtime-provider", "Realtime provider"),
                ("settings-speech-realtime-model", "Realtime model"),
                ("settings-speech-realtime-voice", "Realtime voice"),
                (
                    "settings-speech-realtime-idle-timeout-minutes",
                    "Realtime idle timeout (minutes)",
                ),
                ("settings-speech-realtime-turn-detection", "Turn detection"),
                ("settings-speech-realtime-vad-threshold", "VAD threshold"),
                (
                    "settings-speech-realtime-vad-silence-ms",
                    "End-of-turn silence (ms)",
                ),
                ("settings-speech-realtime-handsfree-engine", "Hands-free engine"),
            ),
            SettingsCategoryId.STORAGE: tuple(
                (f"settings-storage-{name.replace('_', '-')}", label)
                for name, label in _labels.items()
            ),
            SettingsCategoryId.PRIVACY_SECURITY: (
                ("settings-raw-cli-permitted", "Allow raw CLI host access"),
            ),
            SettingsCategoryId.LIBRARY_RAG: (
                *(
                    (field_id, _rag_field_search_label(field_id))
                    for field_id in RAG_FIELD_GROUP_BY_ID
                ),
                # TASK-23109: Console-integration toggles rendered above the
                # profile editor (not part of the profile field-group map).
                (
                    "settings-library-rag-auto-retrieve-default",
                    "Automatic retrieval",
                ),
                (
                    "settings-library-rag-assistant-access-default",
                    "Assistant Library access",
                ),
            ),
            SettingsCategoryId.SPLASH_SCREEN: (
                ("settings-splash-enabled", "Splash screen enabled"),
                ("settings-splash-default-select", "Default splash card"),
                ("settings-splash-show-progress", "Show progress"),
                ("settings-splash-skip-on-keypress", "Skip on keypress"),
                ("settings-splash-duration", "Splash duration (s)"),
                ("settings-splash-animation-speed", "Animation speed"),
            ),
            SettingsCategoryId.THEME: (
                ("settings-theme-name", "Theme name"),
                ("settings-theme-dark-mode", "Dark theme"),
                ("settings-theme-color-primary", "Primary color"),
                ("settings-theme-color-secondary", "Secondary color"),
                ("settings-theme-color-accent", "Accent color"),
                ("settings-theme-color-background", "Background color"),
                ("settings-theme-color-surface", "Surface color"),
                ("settings-theme-color-panel", "Panel color"),
                ("settings-theme-color-foreground", "Foreground color"),
                ("settings-theme-color-success", "Success color"),
                ("settings-theme-color-warning", "Warning color"),
                ("settings-theme-color-error", "Error color"),
            ),
            SettingsCategoryId.IMAGE_GENERATION: (
                ("settings-imagegen-default_backend", "Default image backend"),
                *_backend_field_entries(
                    "imagegen", _imagegen_backend_labels, _imagegen_field_schema
                ),
                ("settings-imagegen-context_llm_enabled", "Context LLM"),
                *(
                    (f"settings-imagegen-{key}", label)
                    for key, label, _minimum in (
                        *_imagegen_int_specs,
                        *_imagegen_float_specs,
                    )
                ),
            ),
            SettingsCategoryId.VIDEO_GENERATION: (
                ("settings-videogen-default_backend", "Default video backend"),
                *_backend_field_entries(
                    "videogen", _videogen_backend_labels, _videogen_field_schema
                ),
                ("settings-videogen-retention", "Video retention"),
                (
                    "settings-videogen-confirm_cost_estimate",
                    "Confirm cost before paid generation",
                ),
                *(
                    (f"settings-videogen-{key}", label)
                    for key, label, _minimum in _videogen_int_specs
                ),
            ),
        }
    )


build_field_search_index()
