"""Restore only the two transcript-adoption modules at the preceding PR head."""

import presentation_baseline as original

BASE_REVISION = "d6dffdf7270bcc9e31f8b048e86581e576cd9ae7"
original.BASE_REVISION = BASE_REVISION
original.MODULES = {
    "tldw_chatbook.UI.Screens.chat_screen",
    "tldw_chatbook.UI.Console_Modules.console_spend_projection",
}
OriginalLoader = original.OriginalLoader
