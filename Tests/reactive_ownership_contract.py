"""Reviewed TldwCli reactive ownership contract shared by test sentinels."""

from __future__ import annotations


RETAINED_TLDW_REACTIVES = frozenset({"current_tab", "splash_screen_active"})
RETIRED_TLDW_REACTIVES = frozenset(
    {
        "ccp_active_view",
        "chat_api_provider_value",
        "ccp_api_provider_value",
        "rag_expansion_provider_value",
        "current_editing_character_id",
        "current_editing_character_data",
        "chat_sidebar_collapsed",
        "chat_right_sidebar_collapsed",
        "chat_right_sidebar_width",
        "conv_char_sidebar_left_collapsed",
        "conv_char_sidebar_right_collapsed",
        "evals_sidebar_collapsed",
        "media_active_view",
        "current_selected_note_id",
        "current_selected_note_version",
        "current_selected_note_title",
        "current_selected_note_content",
        "notes_sort_by",
        "notes_sort_ascending",
        "notes_preview_mode",
        "notes_auto_save_enabled",
        "notes_auto_save_timer",
        "notes_last_save_time",
        "chat_sidebar_selected_prompt_id",
        "chat_sidebar_selected_prompt_system",
        "chat_sidebar_selected_prompt_user",
        "current_chat_is_ephemeral",
        "current_chat_conversation_id",
        "current_conv_char_tab_conversation_id",
        "current_chat_active_character_data",
        "current_ccp_character_details",
        "active_chat_tab_id",
        "chat_sessions",
        "chat_sidebar_loaded_prompt_id",
        "chat_sidebar_loaded_prompt_title_text",
        "chat_sidebar_loaded_prompt_system_text",
        "chat_sidebar_loaded_prompt_user_text",
        "chat_sidebar_loaded_prompt_keywords_text",
        "chat_sidebar_prompt_display_visible",
        "current_prompt_id",
        "current_prompt_uuid",
        "current_prompt_name",
        "current_prompt_author",
        "current_prompt_details",
        "current_prompt_system",
        "current_prompt_user",
        "current_prompt_keywords_str",
        "current_prompt_version",
        "_initial_media_view_slug",
        "current_media_type_filter_slug",
        "current_media_type_filter_display_name",
        "media_current_page",
        "current_loaded_media_item",
        "chat_settings_mode",
        "chat_settings_search_query",
        "search_active_sub_tab",
        "ingest_active_view",
        "tools_settings_active_view",
        "llm_active_view",
    }
)


assert len(RETAINED_TLDW_REACTIVES) == 2
assert len(RETIRED_TLDW_REACTIVES) == 59
assert RETAINED_TLDW_REACTIVES.isdisjoint(RETIRED_TLDW_REACTIVES)
