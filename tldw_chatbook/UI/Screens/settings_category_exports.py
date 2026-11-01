"""Closed lazy category exports; importing this table loads no category."""

from typing import Any

CATEGORY_EXPORTS = {
    "TERMINAL_DISCLOSURE_LINES": ("...Terminal.contracts", "TERMINAL_DISCLOSURE_LINES"),
    "CustomEndpointReferenceStore": (
        ".settings_provider_view_model",
        "CustomEndpointReferenceStore",
    ),
    "ProviderPickerGroup": (".settings_provider_view_model", "ProviderPickerGroup"),
    "SettingsOverviewPresentation": (
        ".settings_provider_view_model",
        "SettingsOverviewPresentation",
    ),
    "build_entry_edit_mutation": (
        ".settings_provider_view_model",
        "build_entry_edit_mutation",
    ),
    "build_provider_picker_groups": (
        ".settings_provider_view_model",
        "build_provider_picker_groups",
    ),
    "build_settings_overview": (
        ".settings_provider_view_model",
        "build_settings_overview",
    ),
    "conversations_referencing_endpoint": (
        ".settings_provider_view_model",
        "conversations_referencing_endpoint",
    ),
    "convert_slot_to_named_endpoint": (
        ".settings_provider_view_model",
        "convert_slot_to_named_endpoint",
    ),
    "custom_endpoint_rows": (".settings_provider_view_model", "custom_endpoint_rows"),
    "detach_and_delete_entry": (
        ".settings_provider_view_model",
        "detach_and_delete_entry",
    ),
    "SettingsSplashScreenViewer": (
        "...Widgets.settings_splash_screen_viewer",
        "SettingsSplashScreenViewer",
    ),
    "AgentsSettingsPanel": ("...Widgets.settings_agents_panel", "AgentsSettingsPanel"),
    "WEB_SEARCH_TERMS": (".settings_web_search", "SEARCH_TERMS"),
    "WebSearchSettings": (".settings_web_search", "WebSearchSettings"),
    "WebSearchSettingsPanel": (
        "...Widgets.settings_web_search_panel",
        "WebSearchSettingsPanel",
    ),
    "AdvancedConfigSettings": (".settings_advanced_config", "AdvancedConfigSettings"),
    "AdvancedConfigPanel": (
        "...Widgets.settings_advanced_config_panel",
        "AdvancedConfigPanel",
    ),
    "ImageGenSettingsPanel": (
        "...Widgets.settings_image_gen_panel",
        "ImageGenSettingsPanel",
    ),
    "_image_gen_key_source_line": (
        "...Widgets.settings_image_gen_panel",
        "_key_source_line",
    ),
    "_image_gen_secret_placeholder": (
        "...Widgets.settings_image_gen_panel",
        "_secret_placeholder",
    ),
    "VideoGenSettingsPanel": (
        "...Widgets.settings_video_gen_panel",
        "VideoGenSettingsPanel",
    ),
    "SettingsAppearanceDefaults": (
        ".settings_appearance_defaults",
        "SettingsAppearanceDefaults",
    ),
    "build_appearance_save_sections": (
        ".settings_appearance_defaults",
        "build_appearance_save_sections",
    ),
    "load_appearance_defaults": (
        ".settings_appearance_defaults",
        "load_appearance_defaults",
    ),
    "validate_appearance_defaults": (
        ".settings_appearance_defaults",
        "validate_appearance_defaults",
    ),
    "SettingsPrivacyPosture": (".settings_privacy_security", "SettingsPrivacyPosture"),
    "build_privacy_posture_rows": (
        ".settings_privacy_security",
        "build_privacy_posture_rows",
    ),
    "build_settings_privacy_posture": (
        ".settings_privacy_security",
        "build_settings_privacy_posture",
    ),
    "env_var_summary": (".settings_privacy_security", "env_var_summary"),
    "skill_trust_display": (".settings_privacy_security", "skill_trust_display"),
    "configured_provider_keys": (
        ".settings_provider_view_model",
        "configured_provider_keys",
    ),
    "provider_picker_summary": (
        ".settings_provider_view_model",
        "provider_picker_summary",
    ),
    "safe_skill_trust_status": (
        ".settings_privacy_security",
        "safe_skill_trust_status",
    ),
}


def resolve_category_symbol(name: str) -> Any:
    "Keep category exports lazy and preserve their original object identity."
    from importlib import import_module

    if name not in CATEGORY_EXPORTS:
        raise AttributeError(name)
    module, member = CATEGORY_EXPORTS[name]
    return getattr(import_module(module, __package__), member)
