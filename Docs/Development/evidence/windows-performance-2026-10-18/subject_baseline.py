"""Exact pre-TASK-35613 code, loaded without swapping checkout files."""

import presentation_baseline

BASE_REVISION = "f152dfc4a6c4fbf2fcd277543757a46a0d2874e2"
presentation_baseline.BASE_REVISION = BASE_REVISION
presentation_baseline.MODULES = {
    "tldw_chatbook.UI.Screens.chat_screen",
    "tldw_chatbook.Utils.textual_css_fastpath",
}
