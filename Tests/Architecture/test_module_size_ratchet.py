"""A size ratchet for the largest ungoverned modules — stop them growing.

**Why this test exists.** The core-runtime code review (2026-09-17,
`qa/core-code-review-2026-09-17/report.md`) found that several of the
repository's largest modules had no size ratchet row at all, across
directories that neither `test_screen_size_ratchet.py` (screens) nor
`test_library_modules_size_ratchet.py` (`Library_Modules/*_controller.py`)
covers. Without a row, nothing stops them growing — the same silent creep
those two files were written to catch, on the modules that need it most.

**This is a ratchet, not a limit.** Each budget below is pinned at the
module's exact current line count (`len(path.read_text().splitlines())`,
the expression `_measure` uses). The numbers may only ever go DOWN. When
you shrink one of these files, lower its row to the new measurement in the
same commit. If you are here because CI failed, the fix is to put your new
code somewhere else — a controller, a widget, a helper module — never to
raise the number, which re-opens the hole this test exists to close.

**Scope.** These are hand-picked god modules, not a directory family, so
(unlike the Library controller ratchet) there is no glob that auto-adds new
files. `settings_screen.py` is deliberately absent: its split and its
missing ratchet row are already tracked by task-1378 / task-31202 — linked,
not duplicated here (core-review TASK-32809.2 AC#2).

First recorded 2026-09-19 by core-review TASK-32809.2, each row at its exact
measured size as of `origin/dev`.

**Second pass, 2026-09-21 (tier-2 review, task-32901).** TASK-32809.2's
hand-picked list missed four modules LARGER than three of the seven rows it
did pick — the tier-2 review found them in four separate slices, each slice
independently reporting "god module with no ratchet row":

* `UI/Wizards/FirstRunSetupWizard.py` (10,404) — the largest module in the
  repo with no row at all (S21). `backlog/docs/size-decomposition-candidates-
  2026-09-18.md` says "Size ratchets now guard all of these" and does not
  list `UI/Wizards/` anywhere.
* `UI/Screens/watchlists_collections_screen.py` (14,319) — one class, 392
  methods, roughly double the method count of the biggest budgeted
  non-`chat`/`library` class (S18). Neither `test_screen_size_ratchet.py`
  (which holds only `chat_screen` and `library_screen`) nor this file had it.
* `UI/Screens/llm_screen.py` (5,180) and `UI/Screens/change_review_screen.py`
  (4,967) — the other two S18 named.

They are budgeted HERE rather than in `test_screen_size_ratchet.py` because
that file's rows also pin a method count per named class, which needs a
decomposition plan to be meaningful; `personas_screen.py` is the standing
precedent for a screen living in this file. No split is proposed by adding a
row — `backlog/docs/library-decomposition-recipe.md` governs any split.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]

#: path -> max line count. LOWER these when a module shrinks. Never raise
#: them to silence a failure — see the module docstring.
_BUDGETS: dict[str, int] = {
    # TASK-33011 decomposition: lowered as each extraction PR lands
    # (entry tail -> app_entry.py: 21,234 measured -> 20,484; destinations
    # D/K2/K3 -> app_destinations.py: 20,484 -> 19,658; task-33081 on dev
    # +24 -> 19,682; dev growth (theme tail, Dreams wiring) +191 -> 19,873;
    # Library ingest queue -> app_ingest_queue.py: 19,873 -> 14,930; dev
    # growth (theme final wave, SSH sessions) +42 -> 14,972; service
    # composition C/E/O -> app_service_wiring.py: 14,972 -> 11,506; speech
    # N/U/U2 -> app_speech.py: 11,506 -> 10,601; lifecycle, shutdown and
    # quit flow -> app_lifecycle.py: 10,601 -> 8,524; screen navigation ->
    # app_navigation.py: 8,524 -> 7,467; command-palette providers ->
    # app_command_providers.py: 7,467 -> 6,403; per-feature glue ->
    # app_feature_glue.py: 6,403 -> 5,712).
    "tldw_chatbook/app.py": 5712,
    # TASK-33011: the Library ingest queue moved verbatim out of app.py. At
    # 4,999 lines it is larger than two rows below, so it is born governed.
    # 2026-10-03 (PR #2993): TASK-20973's provenance block added 16 lines;
    # the signature probe moved beside the poller's other pure helpers in
    # Library/server_ingest_reconcile.py: 5,015 -> 4,985.
    "tldw_chatbook/app_ingest_queue.py": 4985,
    # TASK-33011: TldwCli's service composition moved verbatim out of app.py.
    # It was governed there; without its own row, wiring code could regrow in
    # the mixin unchecked.
    "tldw_chatbook/app_service_wiring.py": 3601,
    # TASK-33011: TldwCli's lifecycle, shutdown and quit flow moved verbatim
    # out of app.py (LifecycleMixin); governed there, so governed here.
    # TASK-33621.13: 2,140 -> 2,134 (keep-alive dead-pump handling moved
    # into app_keep_alive.py).
    "tldw_chatbook/app_lifecycle.py": 2134,
    # TASK-33011: TldwCli's screen navigation moved verbatim out of app.py
    # (NavigationMixin); governed there, so governed here.
    "tldw_chatbook/app_navigation.py": 1110,
    # TASK-33011: the command-palette providers moved verbatim out of app.py;
    # governed there, so governed here.
    "tldw_chatbook/app_command_providers.py": 1123,
    # TASK-33011: TldwCli's per-feature glue moved verbatim out of app.py
    # (FeatureGlueMixin); governed there, so governed here.
    "tldw_chatbook/app_feature_glue.py": 742,
    "tldw_chatbook/Chat/console_chat_controller.py": 29367,
    "tldw_chatbook/Chat/console_chat_store.py": 22344,
    # TASK-33622.14: the aggregate Roleplay draft guard moved to
    # UI/Persona_Modules/roleplay_draft_guard.py (dev had grown to 16,533,
    # over this row; the move brings it to 16,397).
    # 2026-10-03 (PR #2993, owner decision): +128 is formatter reflow only
    # (TASK-26000 series, Ruff 0.15.22; the file is AST-identical to the
    # 16,397-line version), so the row is re-measured, not grown.
    "tldw_chatbook/UI/Screens/personas_screen.py": 16525,
    "tldw_chatbook/Widgets/Console/console_transcript.py": 8353,
    # TASK-33003 (Phase 3) ends at 7,764 measured, down from 7,802: .2 moved
    # the control-height rules to app CSS, .4/.5/.8 spent part of that.
    # TASK-33006.1 lowers it to 7,516: the Model view's field rows, Source
    # words and open focus moved to console_settings_field_row.py.
    # TASK-33006.2 lowers it to 7,466: the support sync moved there too, and
    # the modal's own support table and "no effect" copy are gone. Its
    # review fix lowers it to 7,445: the required check and the control
    # support reads moved there as well. TASK-33006.3 lowers it to 7,410:
    # the Request estimate and name disclosures are built there now.
    # TASK-33006.4 lowers it to 6,201: the provider picker, the model search
    # and their adapters, Custom model and Keep unverified are gone; the
    # model changes only through Switch model's pick mode.
    "tldw_chatbook/Widgets/Console/console_settings_modal.py": 6268,
    "tldw_chatbook/UI/MCP_Modules/mcp_workbench.py": 6760,
    # Tier-2 review S03/S04: the two largest TTS modules had no row at all,
    # though both are larger in CLASS terms than every row above them
    # (TTSProfileRepository 4,880 lines / 118 methods; TTSService 2,825 / 93)
    # and profile_repository.py is within 456 file lines of the smallest
    # governed row. Pinned at their exact measurement, like every row here.
    "tldw_chatbook/TTS/profile_repository.py": 6304,
    "tldw_chatbook/TTS/TTS_Generation.py": 4046,
    # Added by the tier-2 review (S06 P2 [D3]): the largest module in the
    # repo with no row -- 1,217 methods on one class, rank 8 repo-wide,
    # while rank 9 (`personas_screen.py`) was already pinned. The
    # hand-picked list above missed it. 1,086 of those methods are a single
    # `_request(...)` -> `Model.model_validate` shape, so the honest
    # decomposition is per-API-namespace delegates -- `MCPUnifiedClient` is
    # the precedent already in the package.
    "tldw_chatbook/tldw_api/client.py": 16687,
    # Added 2026-09-21 (task-32901), each at its exact measured size as of
    # `origin/dev` 9e33252708 — see the module docstring's second pass.
    "tldw_chatbook/UI/Screens/watchlists_collections_screen.py": 14324,
    # TASK-33921: 10,854 on dev 2026-10-03 (over by 450); the Voice step moved
    # to UI/Wizards/first_run_voice_step.py.
    "tldw_chatbook/UI/Wizards/FirstRunSetupWizard.py": 9866,
    "tldw_chatbook/UI/Screens/llm_screen.py": 5180,
    "tldw_chatbook/UI/Screens/change_review_screen.py": 4967,
}

#: Same tolerance as the Library controller ratchet: loose enough that
#: ordinary in-file edits do not fail CI, tight enough that a real shrink
#: which forgot to lower its row is still caught.
_SLACK_TOLERANCE_LINES = 50


@lru_cache(maxsize=None)
def _measure(rel_path: str) -> int:
    """Line count of a module via ``str.splitlines()``.

    Cached because both tests below parametrize over the same paths.

    Raises:
        AssertionError: If the module is missing — the budget entry is
            stale and must be updated deliberately, not silently skipped.
    """
    path = _REPO_ROOT / rel_path
    assert path.exists(), f"{rel_path} not found; the budget entry is stale."
    return len(path.read_text(encoding="utf-8").splitlines())


@pytest.mark.unit
@pytest.mark.parametrize("rel_path", sorted(_BUDGETS))
def test_module_does_not_grow_past_its_budget(rel_path: str) -> None:
    """The ceiling itself: a budgeted module may not exceed its pin."""
    max_lines = _BUDGETS[rel_path]
    lines = _measure(rel_path)

    assert lines <= max_lines, (
        f"{rel_path} grew to {lines} lines (budget {max_lines}, "
        f"+{lines - max_lines}).\n\n"
        f"{rel_path} is under a size ratchet "
        f"(Tests/Architecture/test_module_size_ratchet.py). Put new code in "
        f"a controller/widget/helper module — do NOT raise the budget to "
        f"make this pass. Lower it when the file shrinks."
    )


@pytest.mark.unit
@pytest.mark.parametrize("rel_path", sorted(_BUDGETS))
def test_budget_is_not_left_slack(rel_path: str) -> None:
    """The recorded budget should track reality, not drift above it."""
    max_lines = _BUDGETS[rel_path]
    lines = _measure(rel_path)

    assert max_lines - lines <= _SLACK_TOLERANCE_LINES, (
        f"{rel_path} is {max_lines - lines} lines under its budget "
        f"({lines} vs {max_lines}). Set it to {lines} so the real "
        f"measurement is what's pinned."
    )
