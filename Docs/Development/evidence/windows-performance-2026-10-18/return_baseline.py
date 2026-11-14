"""Exact pre-TASK-35614 modules without changing the checkout on disk."""

import presentation_baseline

BASE_REVISION = "191fb7d7a3368062d8c4adea2f053ba4588d4f45"
presentation_baseline.BASE_REVISION = BASE_REVISION
presentation_baseline.MODULES = {
    "tldw_chatbook.UI.Screens.chat_screen",
    "tldw_chatbook.UI.Console_Modules.view_workers",
    "tldw_chatbook.Widgets.Console.console_composer_bar",
}
