"""Category support is shared while guidance-only shortcuts stay bound."""

from types import SimpleNamespace

from Tests.private_profile import private_profile_test
from tldw_chatbook.UI.Screens.settings_config_models import SettingsCategoryId
from tldw_chatbook.UI.Screens.settings_screen import (
    GUIDED_SETTINGS_MUTATION_CATEGORIES,
    SettingsScreen,
)


@private_profile_test
async def test_category_action_support_matches_the_existing_help_contract(request):
    screen = SettingsScreen(SimpleNamespace(unified_mcp_service=None))
    for category in SettingsCategoryId:
        _assert_category_support(screen, category)


def _assert_category_support(screen, category):
    supports = SettingsScreen._category_action_supported
    assert supports(category, "save") == (
        category in GUIDED_SETTINGS_MUTATION_CATEGORIES
        or category is SettingsCategoryId.NETWORK
    )
    assert supports(category, "revert") == (
        category in GUIDED_SETTINGS_MUTATION_CATEGORIES
        or category is SettingsCategoryId.ADVANCED_CONFIG
    )
    assert supports(category, "test") == (
        category in SettingsScreen.TESTABLE_SETTINGS_CATEGORIES
    )
    for action in ("set_active", "clone", "backfill"):
        assert supports(category, action) == (
            category is SettingsCategoryId.LIBRARY_RAG
        )
    for action in ("add_record", "edit_record", "delete_record", "export_profile"):
        assert supports(category, action) == (
            category is SettingsCategoryId.PERSONAL_CONTEXT
        )
    assert not supports(category, "unknown")
    hints = dict(screen._category_footer_shortcuts(category, include_owned_actions=False))
    for key, action in (("s", "save"), ("r", "revert"), ("t", "test")):
        assert (key in hints) == supports(category, action)
