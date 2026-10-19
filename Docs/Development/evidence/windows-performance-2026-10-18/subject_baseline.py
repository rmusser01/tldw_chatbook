"""Exact pre-TASK-35613 code, loaded without swapping checkout files."""

import presentation_baseline

BASE_REVISION = "aeaac674e3b19e2af5989be2a145a0316856248a"
presentation_baseline.BASE_REVISION = BASE_REVISION
presentation_baseline.MODULES = {
    "tldw_chatbook.UI.Screens.chat_screen",
    "tldw_chatbook.Utils.textual_css_fastpath",
}
