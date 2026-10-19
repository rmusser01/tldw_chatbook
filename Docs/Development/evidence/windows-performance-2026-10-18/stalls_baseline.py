"""Restore exactly the four presentation modules preceding TASK-35612."""

import presentation_baseline

BASE_REVISION = "5629f0bdc52be71a4152fdc4c6e9c6336006a9f1"
presentation_baseline.BASE_REVISION = BASE_REVISION
presentation_baseline.MODULES = {
    "tldw_chatbook.UI.Screens.home_screen",
    "tldw_chatbook.UI.Screens.chat_screen",
    "tldw_chatbook.UI.Console_Modules.console_spend_projection",
    "tldw_chatbook.UI.Console_Modules.config_sync",
}
