"""Pin only Home to the exact preceding PR head for native-wait comparisons."""

import presentation_baseline as original

BASE_REVISION = "d24919ab402d503ca8210dc5ee6b0fd186adbd64"
original.BASE_REVISION = BASE_REVISION
original.MODULES = {"tldw_chatbook.UI.Screens.home_screen"}
