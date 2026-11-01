"""Reuse the in-memory loader for the exact pre-matching CSS module."""

import presentation_baseline

BASE_REVISION = "f9cb81835d80c272a2eaba665bdba95f92f020a8"
presentation_baseline.BASE_REVISION = BASE_REVISION
presentation_baseline.MODULES = {"tldw_chatbook.Utils.textual_css_fastpath"}
