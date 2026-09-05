"""Pure Console policies stay with the controllers that consume them."""

import inspect

import pytest

from tldw_chatbook.UI.Console_Modules.commands import ConsoleCommandsController
from tldw_chatbook.UI.Console_Modules.settings_navigation import (
    ConsoleSettingsNavigationController,
)
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen


@pytest.mark.parametrize(
    ("owner", "method_name"),
    [
        (ConsoleCommandsController, "_console_rewind_summary_disabled_reason"),
        (ConsoleSettingsNavigationController, "_console_settings_initial_draft"),
    ],
)
def test_pure_policy_is_owned_not_injected_through_the_screen(owner, method_name):
    assert isinstance(owner.__dict__.get(method_name), staticmethod)
    assert method_name not in inspect.signature(owner).parameters
    assert method_name not in ChatScreen.__dict__
