"""Pin only the preceding display-projection module for control-bar profiling."""

import presentation_baseline as original

BASE_REVISION = "473c32b4edbcf4c17e982045a749b5e7c6bc49b7"
original.BASE_REVISION = BASE_REVISION
original.MODULES = {"tldw_chatbook.UI.Console_Modules.console_spend_projection"}
OriginalLoader = original.OriginalLoader
