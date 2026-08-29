"""Census: no timer-path ``.update(`` may default to ``layout=True`` unnoticed.

``Static.update`` is ``update(content="", *, layout: bool = True)`` and ends in
``self.refresh(layout=layout)``. **The default lays out.** A repaint on a clock
therefore arms a whole ``Screen._refresh_layout`` / ``Compositor.reflow`` on
every tick unless the caller opts out. Three instances have been found the hard
way:

* TASK-21692 -- the Console composer cursor blink: 396 ``Widget.arrange`` calls
  per 6 ticks, ~1.9 whole-screen reflows/second on an *idle* focused composer,
  in a method whose own docstring said it must not do that.
* TASK-21134 item 7 -- the media-viewer match-nav: 10 layout messages -> 0.
* TASK-21595 (this census) -- ``SplashScreen._update_animation`` at 10-100 fps
  during startup, and ``PersonaBuddyWidget._paint_frame`` at the pet's frame
  rate.

Three instances is a pattern, so this module rebuilds the census on every run
instead of trusting a one-off sweep. It walks the package AST, collects every
*repeating* clock root, follows the intra-package call graph from each, and
requires that every ``.update(`` call it can reach is **classified**: either the
call passes ``layout=`` explicitly, or it carries an entry in
``CLASSIFIED_SITES`` saying why it does not need to.

Adding a new timer-path repaint therefore fails this test until its author says
which of the two it is. That is the whole point -- the cost is invisible to
every other test, because nothing else counts layout operations.

Runtime evidence for the two sites this census fixed (measured layout-pass
counts against a measured idle floor, plus geometry-equivalence A/Bs) lives in
``Tests/UI/test_timer_path_layout_cost.py``.

TASK-23028 hardened the census after it was caught green while blind to the
two largest idle clocks in the app:

* **Renamed/injected clock spellings.** The root matcher was the exact callee
  name ``set_interval``, so ``self._set_interval(0.1, ...)`` (a 10 Hz clock in
  ``UI/Console_Modules/realtime.py``, injected through a constructor kwarg)
  and ``self._create_interval(...)`` (``fleet.py``) left the census silently.
  The matcher is now the wrapper family ``^_?(create|set)_interval$`` -- that
  naming pattern is the census's contract: spell an interval wrapper inside it
  or the pass-through detector below fails your build by name.
* **Silent unresolvable roots.** Two roots resolved to nothing and nothing
  noticed (a wiring lambda's *parameter* name; a ``call_later`` deferral shim
  that hid the real callback). Every root must now resolve into the call
  graph, be a recognized pass-through wrapper whose exposed name itself
  matches the clock pattern, or carry a ``CLASSIFIED_ROOTS`` row --
  otherwise ``test_clock_roots_all_resolve_loudly`` fails naming the site.
* **Root-count stability is not evidence.** The blind window had a net-zero
  root count (35 -> 35) with two real changes underneath. The root *set* is
  now pinned (``EXPECTED_CLOCK_ROOTS``): adding a clock means acknowledging
  it here, on purpose, in the same diff.
* **Receiver typing.** dict/set ``.update()`` shares its name with
  ``Static.update`` in the AST; per-site NOT-A-WIDGET rows had started to rot
  (three dict/set sites red on dev). A receiver is now auto-classified as
  not-a-widget ONLY when every binding of that name in scope (or of
  ``self.<attr>`` across the class) is provably a dict/set constructor;
  anything unprovable still requires a row. The inference can only guess
  toward *red*, never toward silence.

Framework-armed clocks (``ProgressBar(total=None)``'s 15 Hz repaint,
``LoadingIndicator``'s 16 Hz, ``auto_refresh``) are armed inside
``textual/dom.py`` and are structurally invisible to this walk of package
sources; they are censused separately in
``Tests/Architecture/test_framework_armed_clock_inventory.py``.

Known remaining limitation, on purpose: the intra-package call graph still
cannot cross a constructor-injected *non-clock* callable (e.g. a
``repaint_chip=lambda: screen._repaint...`` seam), so reach out of the
Console_Modules family understates. Clock ROOTS through such seams are what
TASK-23028 made loud; general graph crossing stays out of scope.
"""

from __future__ import annotations

import ast
import re
import textwrap
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Iterator

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "tldw_chatbook"
REPO_ROOT = PACKAGE_ROOT.parent

#: How deep to follow the call graph out from a clock root. Six hops covers the
#: real chains (``_poll_transcript`` -> ... -> a leaf widget's ``sync_*``)
#: without the graph degenerating into "everything".
MAX_CALL_DEPTH = 6

#: Textual's own repeating-clock constructors. ``set_timer`` is a one-shot, so
#: it only counts as a clock when the callback re-arms *itself* -- an interval
#: spelled as a chain of one-shots, which is how the Persona Buddy frame timer
#: and several Console ticks are written.
REPEATING_CLOCK = "set_interval"
ONE_SHOT_CLOCK = "set_timer"

#: The census's naming contract for repeating-clock constructors (TASK-23028).
#: ``set_interval`` itself plus the injected-wrapper spellings the Console
#: wiring uses (``self._set_interval``, ``self._create_interval``). A wrapper
#: that forwards to ``set_interval`` under a name OUTSIDE this family is
#: caught by the pass-through detector and fails the census by name.
CLOCK_CALLEE_RE = re.compile(r"^_?(?:create|set)_interval$")

#: Message-pump deferral shims: ``set_interval(s, lambda: app.call_later(cb))``
#: schedules ``cb`` per tick -- the shim's ARGUMENT is the real callback, and
#: recording the shim's own name is how the db_status_manager root resolved to
#: nothing for a month.
DEFERRAL_SHIMS = frozenset(
    {"call_later", "call_after_refresh", "call_next", "call_from_thread"}
)


# ---------------------------------------------------------------------------
# classification
# ---------------------------------------------------------------------------
#
# Key: (path relative to the repo root, enclosing `Class.method`, receiver
#       expression as written).
# Value: why this call does not need `layout=False`.
#
# Four kinds of reason appear, and the distinction matters:
#
#   NOT-A-WIDGET  -- `dict.update` / `set.update`, which shares its name with
#                    `Static.update` and is indistinguishable in the AST.
#   NEEDS-LAYOUT  -- the rendered size genuinely depends on the content (a
#                    `height: auto` box, or a `width: auto` strip), so skipping
#                    the layout pass would leave stale geometry on screen. This
#                    is a real behaviour change, not an optimisation.
#   NOT-PER-TICK  -- the tick reaches the call only through an equality gate or
#                    a branch a tick cannot take, so it is not a per-tick cost.
#   UNREACHABLE   -- the module has no importer anywhere in the repo.
#
CLASSIFIED_SITES: dict[tuple[str, str, str], str] = {
    # -- tldw_chatbook/UI/Screens/scheduling/schedules_workbench.py
    (
        "tldw_chatbook/UI/Screens/scheduling/schedules_workbench.py",
        "SchedulesWorkbench._update_static_content",
        "target",
    ): (
        "NOT-PER-TICK: equality-gated on Static.content, so unchanged relative-"
        "time refreshes do not repaint or lay out; changed height:auto copy "
        "keeps the required default layout pass."
    ),
    # -- tldw_chatbook/UI/Console_Modules/prompt_queue.py
    (
        "tldw_chatbook/UI/Console_Modules/prompt_queue.py",
        "ConsolePromptQueueRegion.sync_presentation",
        "preview",
    ): "NEEDS-LAYOUT: queue preview grows with the queued prompt.",
    (
        "tldw_chatbook/UI/Console_Modules/prompt_queue.py",
        "ConsolePromptQueueRegion.sync_presentation",
        "summary",
    ): "NEEDS-LAYOUT: queue summary is height:auto.",
    # -- tldw_chatbook/UI/Research_Window.py
    (
        "tldw_chatbook/UI/Research_Window.py",
        "ResearchWindow._update_detail",
        "self.query_one('#research-run-detail', Static)",
    ): (
        "NEEDS-LAYOUT: run detail is a height:auto block that grows with "
        "events."
    ),
    # -- tldw_chatbook/UI/Screens/chat_screen.py
    #    (dict/set receivers formerly carried NOT-A-WIDGET rows here; they are
    #    now auto-classified by _receiver_is_provably_collection, TASK-23028)
    (
        "tldw_chatbook/UI/Screens/chat_screen.py",
        "ChatScreen._sync_console_agent_section",
        "fleet_summary",
    ): (
        "NEEDS-LAYOUT: height:auto summary whose line count tracks the "
        "fleet size."
    ),
    (
        "tldw_chatbook/UI/Screens/chat_screen.py",
        "ChatScreen._sync_console_agent_section",
        "self.query_one('#console-agent-section-status', Static)",
    ): "NEEDS-LAYOUT: height:auto agent status line.",
    (
        "tldw_chatbook/UI/Screens/chat_screen.py",
        "ChatScreen._sync_console_agent_section",
        "self.query_one('#console-agent-section-steps', Static)",
    ): "NEEDS-LAYOUT: height:auto agent steps line.",
    (
        "tldw_chatbook/UI/Screens/chat_screen.py",
        "ChatScreen._sync_console_mode_bar",
        "mode_bar",
    ): (
        "NEEDS-LAYOUT: the mode bar's chip row wraps, so its height is "
        "content-driven."
    ),
    (
        "tldw_chatbook/UI/Screens/chat_screen.py",
        "ChatScreen._sync_console_rail_system_line",
        "system_line",
    ): (
        "NOT-PER-TICK: equality-gated on _console_rail_system_line_last "
        "(TASK-251), so the 0.2 s tick does not repaint it."
    ),
    (
        "tldw_chatbook/UI/Screens/chat_screen.py",
        "ChatScreen._sync_console_settings_summary",
        "recovery",
    ): (
        "NEEDS-LAYOUT: the readiness row is display-toggled on the same "
        "path."
    ),
    (
        "tldw_chatbook/UI/Screens/chat_screen.py",
        "ChatScreen._sync_console_settings_summary",
        "self.query_one('#console-model-section-max-tokens .console-model-section-value', Static)",
    ): (
        "NEEDS-LAYOUT: as above -- shares the wrapped .console-model- "
        "section-value rule."
    ),
    (
        "tldw_chatbook/UI/Screens/chat_screen.py",
        "ChatScreen._sync_console_settings_summary",
        "self.query_one('#console-model-section-model .console-model-section-value', Static)",
    ): "NEEDS-LAYOUT: as above -- wrapped value, auto height capped at 3.",
    (
        "tldw_chatbook/UI/Screens/chat_screen.py",
        "ChatScreen._sync_console_settings_summary",
        "self.query_one('#console-model-section-provider .console-model-section-value', Static)",
    ): (
        "NEEDS-LAYOUT: .console-model-section-value is text-wrap:wrap with "
        "max-height:3, so the painted row count changes with the model "
        "name."
    ),
    (
        "tldw_chatbook/UI/Screens/chat_screen.py",
        "ChatScreen._sync_console_settings_summary",
        "self.query_one('#console-model-section-temperature .console-model-section-value', Static)",
    ): (
        "NEEDS-LAYOUT: as above -- shares the wrapped .console-model- "
        "section-value rule."
    ),
    # -- tldw_chatbook/UI/Screens/trajectory_screen.py
    (
        "tldw_chatbook/UI/Screens/trajectory_screen.py",
        "TrajectoryScreen._refresh_state",
        "self.query_one('#trajectory-state', Static)",
    ): (
        "NEEDS-LAYOUT: #trajectory-state is height:auto with max-height:4, "
        "so the state copy really does change its row count."
    ),
    # -- tldw_chatbook/UI/Screens/video_player_screen.py
    (
        "tldw_chatbook/UI/Screens/video_player_screen.py",
        "VideoPlayerScreen._refresh_status",
        "self.query_one('#video-player-status', Static)",
    ): (
        "NOT-PER-TICK: #video-player-status is height:1, but the status "
        "timer only runs while a video is actually playing and the line "
        "changes on every tick anyway (position advances), so the repaint "
        "is real work rather than idle burn. Left alone to avoid a "
        "behaviour change for a gain that is one coalesced layout pass per "
        "second."
    ),
    # -- tldw_chatbook/Widgets/Console/console_composer_bar.py
    (
        "tldw_chatbook/Widgets/Console/console_composer_bar.py",
        "ConsoleComposerBar._refresh_visible_draft",
        "self.query_one('#console-command-visible-text', Static)",
    ): (
        "NEEDS-LAYOUT: _refresh_visible_draft is the draft-mutation path "
        "and must recompute the composer height. TASK-21692 split the blink "
        "tick off into _render_visible_draft_only, which passes "
        "layout=False."
    ),
    (
        "tldw_chatbook/Widgets/Console/console_composer_bar.py",
        "ConsoleComposerBar._sync_collapsed_presentation",
        "status",
    ): "NEEDS-LAYOUT: collapsed-composer status line is height:auto.",
    (
        "tldw_chatbook/Widgets/Console/console_composer_bar.py",
        "ConsoleComposerBar._sync_send_disabled_reason",
        "strip",
    ): (
        "NEEDS-LAYOUT: the Send disabled-reason strip is width:auto and the "
        "same call toggles display/width/height inline -- the layout pass "
        "is the point, not an accident."
    ),
    (
        "tldw_chatbook/Widgets/Console/console_composer_bar.py",
        "ConsoleComposerBar.set_pending_attachment_label",
        "indicator",
    ): (
        "NEEDS-LAYOUT: attachment indicator is width:auto and display- "
        "toggled."
    ),
    (
        "tldw_chatbook/Widgets/Console/console_composer_bar.py",
        "ConsoleComposerBar.set_voice_status",
        "chip",
    ): (
        "NEEDS-LAYOUT: the voice chip's width is computed and assigned "
        "inline on the same path (styles.width / display), so its box "
        "changes with the content it is being handed."
    ),
    # -- tldw_chatbook/Widgets/Console/console_prompt_queue_modal.py
    (
        "tldw_chatbook/Widgets/Console/console_prompt_queue_modal.py",
        "ConsolePromptQueueModal._apply_snapshot",
        "state",
    ): "NEEDS-LAYOUT: modal state block is height:auto.",
    # -- tldw_chatbook/Widgets/Console/console_session_surface.py
    (
        "tldw_chatbook/Widgets/Console/console_session_surface.py",
        "ConsoleSessionSurface.set_session_title",
        "header",
    ): (
        "NEEDS-LAYOUT: session title width is auto and tracks the title "
        "text."
    ),
    (
        "tldw_chatbook/Widgets/Console/console_session_surface.py",
        "ConsoleSessionSurface.show_fleet_coachmark",
        "content",
    ): (
        "NEEDS-LAYOUT: the fleet coachmark is a display-toggled auto-sized "
        "callout."
    ),
    (
        "tldw_chatbook/Widgets/Console/console_session_surface.py",
        "ConsoleSessionSurface.sync_inline_guidance",
        "title",
    ): "NEEDS-LAYOUT: inline guidance is height:auto and display-toggled.",
    # -- tldw_chatbook/Widgets/Console/console_status_chips.py
    (
        "tldw_chatbook/Widgets/Console/console_status_chips.py",
        "ConsoleStatusChips.sync_cost_state",
        "chip",
    ): (
        "NEEDS-LAYOUT: chips are width:auto -- the label length IS the "
        "width."
    ),
    (
        "tldw_chatbook/Widgets/Console/console_status_chips.py",
        "ConsoleStatusChips.sync_run_chip",
        "chip",
    ): (
        "NEEDS-LAYOUT: chips are width:auto -- the label length IS the "
        "width."
    ),
    # -- tldw_chatbook/Widgets/Console/console_transcript.py
    (
        "tldw_chatbook/Widgets/Console/console_transcript.py",
        "ConsoleTranscript.sync_jump_indicator",
        "pill",
    ): "NEEDS-LAYOUT: the jump pill is width:auto and display-toggled.",
    # -- tldw_chatbook/Widgets/Console/console_video_preview.py
    (
        "tldw_chatbook/Widgets/Console/console_video_preview.py",
        "ConsoleVideoPreview._update_frame",
        "frame",
    ): (
        "NEEDS-LAYOUT: .console-video-preview-frame is height:auto and the "
        "Pixels renderable's row count varies with the scaled image (and "
        "with the poster-text fallback), so the box really does resize per "
        "frame."
    ),
    # -- tldw_chatbook/Widgets/Console/console_workspace_context.py
    (
        "tldw_chatbook/Widgets/Console/console_workspace_context.py",
        "ConsoleWorkspaceContextTray._update_workspace_tree_selection_context",
        "context",
    ): "NEEDS-LAYOUT: the context tray line count tracks the selection.",
    # -- tldw_chatbook/Widgets/Library/library_file_notes_workspace.py
    (
        "tldw_chatbook/Widgets/Library/library_file_notes_workspace.py",
        "LibraryFileNotesWorkspace._update_static_content",
        "target",
    ): (
        "NOT-PER-TICK: equality-gated on Static.content, so an unchanged file "
        "poll does not repaint or lay out; changed auto-height status/path copy "
        "keeps the required default layout pass."
    ),
    (
        "tldw_chatbook/Widgets/Library/library_file_notes_workspace.py",
        "LibraryFileNotesWorkspace._apply_opened_document",
        "self.query_one('#file-notes-breadcrumb', Static)",
    ): (
        "NEEDS-LAYOUT: the breadcrumb wraps, so its row count tracks the "
        "path."
    ),
    (
        "tldw_chatbook/Widgets/Library/library_file_notes_workspace.py",
        "LibraryFileNotesWorkspace._dismiss_reload_confirmation",
        "self.query_one('#file-notes-reload-confirm-copy', Static)",
    ): (
        "NEEDS-LAYOUT: the reload-confirm callout is display-toggled and "
        "auto-sized."
    ),
    (
        "tldw_chatbook/Widgets/Library/library_file_notes_workspace.py",
        "LibraryFileNotesWorkspace._set_action_status",
        "self.query_one('#file-notes-action-status', Static)",
    ): "NEEDS-LAYOUT: height:auto action status line.",
    (
        "tldw_chatbook/Widgets/Library/library_file_notes_workspace.py",
        "LibraryFileNotesWorkspace._sync_large_file_preview",
        "status",
    ): (
        "NEEDS-LAYOUT: the save/preview status lines are height:auto and "
        "display-toggled; the poll is a 1.5 s worker-backed reconcile, not "
        "an animation clock."
    ),
    (
        "tldw_chatbook/Widgets/Library/library_file_notes_workspace.py",
        "LibraryFileNotesWorkspace._update_controls",
        "self.query_one('#file-notes-action-status', Static)",
    ): "NEEDS-LAYOUT: height:auto action status line.",
    (
        "tldw_chatbook/Widgets/Library/library_file_notes_workspace.py",
        "LibraryFileNotesWorkspace._update_root_surface",
        "status",
    ): (
        "NEEDS-LAYOUT: the save/preview status lines are height:auto and "
        "display-toggled; the poll is a 1.5 s worker-backed reconcile, not "
        "an animation clock."
    ),
    # -- tldw_chatbook/Widgets/audio_troubleshooting_dialog.py
    (
        "tldw_chatbook/Widgets/audio_troubleshooting_dialog.py",
        "AudioTroubleshootingDialog._update_level_meter",
        "level_text",
    ): (
        "NEEDS-LAYOUT: 'Level: N%' is width:auto and its length changes "
        "with the digit count; the tick also only runs during an explicit "
        "user-started mic test, not at idle."
    ),
    (
        "tldw_chatbook/Widgets/audio_troubleshooting_dialog.py",
        "AudioTroubleshootingDialog._update_level_meter",
        "meter",
    ): (
        "NOT-A-WIDGET: ProgressBar.update(progress=...) has no layout "
        "kwarg."
    ),
    # -- tldw_chatbook/Widgets/detailed_progress.py
    (
        "tldw_chatbook/Widgets/detailed_progress.py",
        "DetailedProgressBar._update_metrics",
        "self.query_one('#elapsed-time', Static)",
    ): (
        "UNREACHABLE: Widgets/detailed_progress.py has no importer (prod or "
        "tests)."
    ),
    (
        "tldw_chatbook/Widgets/detailed_progress.py",
        "DetailedProgressBar._update_metrics",
        "self.query_one('#memory-usage', Static)",
    ): (
        "UNREACHABLE: Widgets/detailed_progress.py has no importer (prod or "
        "tests)."
    ),
    (
        "tldw_chatbook/Widgets/detailed_progress.py",
        "DetailedProgressBar._update_metrics",
        "self.query_one('#remaining-time', Static)",
    ): (
        "UNREACHABLE: Widgets/detailed_progress.py has no importer (prod or "
        "tests)."
    ),
    (
        "tldw_chatbook/Widgets/detailed_progress.py",
        "DetailedProgressBar._update_metrics",
        "self.query_one('#speed-metric', Static)",
    ): (
        "UNREACHABLE: Widgets/detailed_progress.py has no importer (prod or "
        "tests)."
    ),
    # -- tldw_chatbook/Widgets/loading_states.py
    (
        "tldw_chatbook/Widgets/loading_states.py",
        "InlineLoader._update_dots",
        "self",
    ): (
        "UNREACHABLE: Widgets/loading_states.py has no importer (prod or "
        "tests)."
    ),
    # -- tldw_chatbook/Widgets/splash_screen.py
    (
        "tldw_chatbook/Widgets/splash_screen.py",
        "SplashScreen._display_static_fallback",
        "display",
    ): (
        "NOT-PER-TICK: _display_static_fallback runs once, on the error "
        "edge that also stops the animation timer. The per-frame repaint in "
        "_update_animation passes layout=False (TASK-21595)."
    ),
    (
        "tldw_chatbook/Widgets/splash_screen.py",
        "SplashScreen._update_animation",
        "self.effect_handler",
    ): (
        "NOT-A-WIDGET: the effect handler's own frame producer, not a "
        "Static."
    ),
    # -- tldw_chatbook/Widgets/status_dashboard.py
    (
        "tldw_chatbook/Widgets/status_dashboard.py",
        "StatusDashboard._update_time_display",
        "time_display",
    ): (
        "UNREACHABLE: Widgets/status_dashboard.py has no importer (prod or "
        "tests)."
    ),
}


# ---------------------------------------------------------------------------
# root classification (TASK-23028)
# ---------------------------------------------------------------------------
#
# Key: (path relative to the repo root, enclosing `Class.method`, the clock
#       constructor call as unparsed source).
# Value: why this repeating-clock root is allowed to resolve to nothing.
#
# Empty on the current tree, deliberately: every root either resolves into
# the call graph or is a pass-through wrapper exposed inside the
# ^_?(create|set)_interval$ family. A row here is a last resort -- it means
# the census agrees to walk NOTHING out of that clock, so the reason must
# say where the callback's `.update(` cost is accounted for instead.
CLASSIFIED_ROOTS: dict[tuple[str, str, str], str] = {}


#: The full expected clock-root set, pinned (TASK-23028). Root-COUNT
#: stability is worthless evidence: in the very window that motivated this
#: task, one root left the census and another arrived, 35 -> 35, and nobody
#: saw either. Adding a repeating clock now means acknowledging it here in
#: the same diff -- after checking what its callback repaints (that is what
#: the census walks out of it).
EXPECTED_CLOCK_ROOTS: frozenset[tuple[str, str, str | None, str]] = frozenset(
    {
        # kind, file, class, callback
        (
            "_create_interval",
            "tldw_chatbook/UI/Console_Modules/fleet.py",
            "ConsoleFleetLifecycleController",
            "_console_fleet_survivor_tick",
        ),
        (
            "_set_interval",
            "tldw_chatbook/UI/Console_Modules/realtime.py",
            "ConsoleRealtimeController",
            "_tick_console_realtime",
        ),
        (
            "rearming-set_timer",
            "tldw_chatbook/UI/Screens/model_installed_view.py",
            "InstalledView",
            "_focus_import_after_recompose",
        ),
        (
            "rearming-set_timer",
            "tldw_chatbook/UI/Screens/model_installed_view.py",
            "InstalledView",
            "_focus_revealed_after_recompose",
        ),
        (
            "rearming-set_timer",
            "tldw_chatbook/Widgets/Console/console_session_surface.py",
            "ConsoleSessionSurface",
            "_scroll_active_tab_into_view",
        ),
        (
            "set_interval",
            "tldw_chatbook/UI/Console_Modules/dictation.py",
            "ConsoleDictationController",
            "_tick_console_dictation_elapsed",
        ),
        (
            "set_interval",
            "tldw_chatbook/UI/Console_Modules/hands_free.py",
            "ConsoleHandsFreeController",
            "_tick_console_hands_free",
        ),
        (
            "set_interval",
            "tldw_chatbook/UI/LLM_Management_Window.py",
            "LLMManagementWindow",
            "_schedule_ollama_api_state",
        ),
        (
            "set_interval",
            "tldw_chatbook/UI/Navigation/main_navigation.py",
            "MainNavigationBar",
            "_update_overflow_hints",
        ),
        (
            "set_interval",
            "tldw_chatbook/UI/Research_Window.py",
            "ResearchWindow",
            "_auto_refresh_selected_run",
        ),
        (
            "set_interval",
            "tldw_chatbook/UI/Screens/chat_screen.py",
            "ChatScreen",
            "_poll_transcript",
        ),
        (
            "set_interval",
            "tldw_chatbook/UI/Screens/chat_screen.py",
            "ChatScreen",
            "_sync_console_cost_chip",
        ),
        (
            "set_interval",
            "tldw_chatbook/UI/Screens/llm_screen.py",
            "LLMScreen",
            "refresh_lab_status",
        ),
        (
            "set_interval",
            "tldw_chatbook/UI/Screens/scheduling/schedules_workbench.py",
            "SchedulesWorkbench",
            "_refresh_next_run_rendering",
        ),
        (
            "set_interval",
            "tldw_chatbook/UI/Screens/trajectory_screen.py",
            "TrajectoryScreen",
            "_poll_revision",
        ),
        (
            "set_interval",
            "tldw_chatbook/UI/Screens/video_player_screen.py",
            "VideoPlayerScreen",
            "_refresh_status",
        ),
        (
            "set_interval",
            "tldw_chatbook/UI/Speech/speech_playground_pane.py",
            "SpeechPlaygroundPane",
            "_poll_audio_cpp_runtime_observation",
        ),
        (
            "set_interval",
            "tldw_chatbook/Utils/db_status_manager.py",
            "DBStatusManager",
            "update_db_sizes",
        ),
        (
            "set_interval",
            "tldw_chatbook/Widgets/Console/console_background_effect.py",
            "ConsoleBackgroundEffect",
            "_advance_frame",
        ),
        (
            "set_interval",
            "tldw_chatbook/Widgets/Console/console_composer_bar.py",
            "ConsoleComposerBar",
            "_toggle_cursor_blink",
        ),
        (
            "set_interval",
            "tldw_chatbook/Widgets/Console/console_prompt_queue_modal.py",
            "ConsolePromptQueueModal",
            "_poll_snapshot",
        ),
        # ConsoleSetupBackdrop._tick retired by TASK-23021 (snow is a still frame).
        (
            "set_interval",
            "tldw_chatbook/Widgets/Console/console_assistant_turn.py",
            "ConsoleActivityHeader",
            "_tick_raw_cli_elapsed",
        ),
        (
            "set_interval",
            "tldw_chatbook/Widgets/Console/console_transcript.py",
            "ConsoleMessageHeader",
            "_tick_raw_cli_elapsed",
        ),
        (
            "set_interval",
            "tldw_chatbook/Widgets/Console/console_video_preview.py",
            "ConsoleVideoPreview",
            "_pause_if_offscreen",
        ),
        (
            "set_interval",
            "tldw_chatbook/Widgets/Library/library_file_notes_workspace.py",
            "LibraryFileNotesWorkspace",
            "_start_poll",
        ),
        (
            "set_interval",
            "tldw_chatbook/Widgets/Persona_Widgets/persona_buddy_widget.py",
            "PersonaBuddyWidget",
            "refresh_from_controller",
        ),
        (
            "set_interval",
            "tldw_chatbook/Widgets/Tamagotchi/base_tamagotchi.py",
            "BaseTamagotchi",
            "_periodic_update",
        ),
        (
            "set_interval",
            "tldw_chatbook/Widgets/Tamagotchi/base_tamagotchi.py",
            "BaseTamagotchi",
            "next_frame",
        ),
        (
            "set_interval",
            "tldw_chatbook/Widgets/activity_log.py",
            "ActivityLogWidget",
            "_update_timestamps",
        ),
        (
            "set_interval",
            "tldw_chatbook/Widgets/audio_troubleshooting_dialog.py",
            "AudioTroubleshootingDialog",
            "_update_level_meter",
        ),
        (
            "set_interval",
            "tldw_chatbook/Widgets/detailed_progress.py",
            "DetailedProgressBar",
            "_update_metrics",
        ),
        (
            "set_interval",
            "tldw_chatbook/Widgets/loading_states.py",
            "InlineLoader",
            "_update_dots",
        ),
        (
            "set_interval",
            "tldw_chatbook/Widgets/splash_screen.py",
            "SplashScreen",
            "_update_animation",
        ),
        (
            "set_interval",
            "tldw_chatbook/Widgets/status_dashboard.py",
            "StatusDashboard",
            "_update_time_display",
        ),
        (
            "set_interval",
            "tldw_chatbook/app.py",
            "TldwCli",
            "_perform_change_review_retention",
        ),
        (
            "set_interval",
            "tldw_chatbook/app.py",
            "TldwCli",
            "_reconcile_boot_worker_slots",
        ),
        (
            "set_interval",
            "tldw_chatbook/app.py",
            "TldwCli",
            "_record_ui_heartbeat",
        ),
        (
            "set_interval",
            "tldw_chatbook/app.py",
            "TldwCli",
            "perform_media_cleanup",
        ),
    }
)


# ---------------------------------------------------------------------------
# census machinery
# ---------------------------------------------------------------------------


class _Module:
    __slots__ = ("path", "tree", "dotted", "classes", "methods", "funcs", "enclosing")

    def __init__(self, path: Path, tree: ast.Module, dotted: str) -> None:
        self.path = path
        self.tree = tree
        self.dotted = dotted
        self.classes: dict[str, ast.ClassDef] = {}
        self.methods: dict[tuple[str, str], ast.AST] = {}
        self.funcs: dict[str, ast.AST] = {}
        self.enclosing: dict[ast.AST, tuple[str | None, str | None]] = {}


def _callee(node: ast.Call) -> str:
    func = node.func
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return ""


def _callback_names(arg: ast.AST) -> list[str]:
    """Method/function names a timer callback argument can refer to."""
    names: list[str] = []
    if isinstance(arg, ast.Attribute):
        names.append(arg.attr)
    elif isinstance(arg, ast.Name):
        names.append(arg.id)
    elif isinstance(arg, ast.Lambda):
        for call in (n for n in ast.walk(arg) if isinstance(n, ast.Call)):
            names.append(_callee(call))
            # `lambda: app.call_later(self.update_db_sizes)` runs
            # update_db_sizes per tick; the shim's argument is the real
            # callback (TASK-23028).
            if _callee(call) in DEFERRAL_SHIMS and call.args:
                names.extend(_callback_names(call.args[0]))
    elif isinstance(arg, ast.Call) and _callee(arg) == "partial" and arg.args:
        names.extend(_callback_names(arg.args[0]))
    return [name for name in names if name]


def _index_enclosing(tree: ast.Module) -> dict[ast.AST, tuple[str | None, str | None]]:
    out: dict[ast.AST, tuple[str | None, str | None]] = {}

    def walk(node: ast.AST, cls: str | None, fn: str | None) -> None:
        for child in ast.iter_child_nodes(node):
            child_cls, child_fn = cls, fn
            if isinstance(child, ast.ClassDef):
                child_cls, child_fn = child.name, None
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                child_fn = child.name
            out[child] = (child_cls, child_fn)
            walk(child, child_cls, child_fn)

    walk(tree, None, None)
    return out


def _parent_map(tree: ast.Module) -> dict[ast.AST, ast.AST]:
    """child -> parent for every node. Built lazily, per clock-bearing module."""
    parents: dict[ast.AST, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    return parents


def _enclosing_callables(node: ast.AST, parents: dict[ast.AST, ast.AST]):
    """Innermost-first lambdas/defs enclosing *node*."""
    chain = []
    current: ast.AST | None = node
    while (current := parents.get(current)) is not None:
        if isinstance(current, (ast.Lambda, ast.FunctionDef, ast.AsyncFunctionDef)):
            chain.append(current)
    return chain


def _param_names(fn) -> set[str]:
    args = fn.args
    names = {a.arg for a in (*args.posonlyargs, *args.args, *args.kwonlyargs)}
    if args.vararg:
        names.add(args.vararg.arg)
    if args.kwarg:
        names.add(args.kwarg.arg)
    return names


def _exposed_name(fn, parents: dict[ast.AST, ast.AST]) -> str | None:
    """The name a pass-through wrapper's CALLERS use for it.

    A ``def`` is exposed as its own name. A lambda is exposed as the keyword
    argument it is passed as (``create_interval=lambda ...`` -- the Console
    wiring shape), or the assignment target it is bound to. Anything else
    (a bare positional lambda, a lambda in a data structure) has no name the
    census could match call sites against, so it returns None and the caller
    fails loudly.
    """
    if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return fn.name
    binder = parents.get(fn)
    if isinstance(binder, ast.keyword):
        return binder.arg
    if isinstance(binder, ast.Assign) and len(binder.targets) == 1:
        target = binder.targets[0]
        if isinstance(target, ast.Name):
            return target.id
        if isinstance(target, ast.Attribute):
            return target.attr
    if isinstance(binder, ast.AnnAssign):
        if isinstance(binder.target, ast.Name):
            return binder.target.id
        if isinstance(binder.target, ast.Attribute):
            return binder.target.attr
    return None


# ---------------------------------------------------------------------------
# receiver typing (TASK-23028)
# ---------------------------------------------------------------------------
#
# dict/set `.update()` is indistinguishable from `Static.update()` by callee
# name alone. A receiver is auto-classified NOT-A-WIDGET only on positive
# proof: EVERY binding of that name in the relevant scope is a dict/set
# constructor expression (or carries a dict/set-rooted annotation), and at
# least one such binding exists. One unprovable binding, a tuple-unpack, a
# loop target, a bare parameter -- and the site needs a CLASSIFIED_SITES row
# instead. The inference can therefore only err toward RED.

_COLLECTION_CONSTRUCTORS = frozenset(
    {"dict", "set", "frozenset", "defaultdict", "Counter", "OrderedDict", "ChainMap"}
)
_COLLECTION_ANNOTATION_ROOTS = frozenset(
    {
        "dict", "Dict", "defaultdict", "DefaultDict", "OrderedDict", "Counter",
        "MutableMapping", "Mapping", "ChainMap",
        "set", "Set", "frozenset", "FrozenSet", "MutableSet", "AbstractSet",
    }
)


def _annotation_root(node: ast.AST | None) -> str | None:
    while isinstance(node, ast.Subscript):
        node = node.value
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value.split("[", 1)[0].strip().rsplit(".", 1)[-1]
    return None


def _provable_collection_value(node: ast.AST | None) -> bool:
    if isinstance(node, (ast.Dict, ast.Set, ast.DictComp, ast.SetComp)):
        return True
    if isinstance(node, ast.Call) and _callee(node) in _COLLECTION_CONSTRUCTORS:
        return True
    return False


def _provable_collection_annotation(node: ast.AST | None) -> bool:
    return _annotation_root(node) in _COLLECTION_ANNOTATION_ROOTS


def _name_bound_in(target: ast.AST, name: str) -> bool:
    """Does *target* REBIND the bare name?

    Store-context only: ``x[k] = v`` and ``x.y = v`` mutate ``x`` (its Name
    loads), they do not rebind it -- treating them as bindings made the
    inference reject every dict a function ever writes a key into.
    """
    return any(
        isinstance(sub, ast.Name)
        and sub.id == name
        and isinstance(sub.ctx, ast.Store)
        for sub in ast.walk(target)
    )


def _local_receiver_is_collection(func_node: ast.AST, name: str) -> bool:
    """True iff every binding of local *name* in *func_node* proves dict/set."""
    provable = 0
    for arg in (
        *func_node.args.posonlyargs,
        *func_node.args.args,
        *func_node.args.kwonlyargs,
    ):
        if arg.arg == name:
            if _provable_collection_annotation(arg.annotation):
                provable += 1
            else:
                return False
    for extra in (func_node.args.vararg, func_node.args.kwarg):
        if extra is not None and extra.arg == name:
            return False
    for node in ast.walk(func_node):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    if _provable_collection_value(node.value):
                        provable += 1
                    else:
                        return False
                elif not isinstance(target, ast.Name) and _name_bound_in(
                    target, name
                ):
                    return False  # tuple/star unpack: type unknowable
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and node.target.id == name:
                if _provable_collection_value(
                    node.value
                ) or _provable_collection_annotation(node.annotation):
                    provable += 1
                else:
                    return False
        elif isinstance(node, ast.NamedExpr):
            if isinstance(node.target, ast.Name) and node.target.id == name:
                if _provable_collection_value(node.value):
                    provable += 1
                else:
                    return False
        elif isinstance(node, (ast.For, ast.AsyncFor)):
            if _name_bound_in(node.target, name):
                return False
        elif isinstance(node, ast.comprehension):
            if _name_bound_in(node.target, name):
                return False
        elif isinstance(node, (ast.With, ast.AsyncWith)):
            for item in node.items:
                if item.optional_vars is not None and _name_bound_in(
                    item.optional_vars, name
                ):
                    return False
        elif isinstance(node, ast.ExceptHandler):
            if node.name == name:
                return False
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            if name in node.names:
                return False
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                if (alias.asname or alias.name.split(".")[0]) == name:
                    return False
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node is not func_node and node.name == name:
                return False
    return provable > 0


def _is_self_attr_store(node: ast.AST, attr: str) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "self"
        and node.attr == attr
        and isinstance(node.ctx, ast.Store)
    )


def _self_attr_is_collection(class_node: ast.ClassDef | None, attr: str) -> bool:
    """True iff every ``self.<attr>`` binding in the class proves dict/set.

    ``self.<attr>[k] = v`` is a mutation, not a rebinding (the Attribute
    loads), so it neither proves nor disproves. A rebinding buried in a
    tuple-unpack target has an unknowable type and disqualifies.
    """
    if class_node is None:
        return False
    provable = 0
    for node in ast.walk(class_node):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if _is_self_attr_store(target, attr):
                    if _provable_collection_value(node.value):
                        provable += 1
                    else:
                        return False
                elif any(
                    _is_self_attr_store(sub, attr) for sub in ast.walk(target)
                ):
                    return False  # tuple/star unpack: type unknowable
        elif isinstance(node, ast.AnnAssign):
            target = node.target
            hit = _is_self_attr_store(target, attr) or (
                isinstance(target, ast.Name) and target.id == attr
            )
            if hit:
                if _provable_collection_value(
                    node.value
                ) or _provable_collection_annotation(node.annotation):
                    provable += 1
                else:
                    return False
    return provable > 0


def _receiver_is_provably_collection(
    module: "_Module", klass: str | None, func_node: ast.AST, call: ast.Call
) -> bool:
    """dict/set proof for the ``.update(`` receiver, or False (needs a row)."""
    receiver = call.func.value if isinstance(call.func, ast.Attribute) else None
    if isinstance(receiver, ast.Name):
        return _local_receiver_is_collection(func_node, receiver.id)
    if (
        isinstance(receiver, ast.Attribute)
        and isinstance(receiver.value, ast.Name)
        and receiver.value.id == "self"
        and klass is not None
    ):
        return _self_attr_is_collection(module.classes.get(klass), receiver.attr)
    return False


def _index_module(path: Path, tree: ast.Module, dotted: str) -> _Module:
    module = _Module(path, tree, dotted)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            module.classes[node.name] = node
            for sub in ast.walk(node):
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    module.methods[(node.name, sub.name)] = sub
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            module.funcs[node.name] = node
    module.enclosing = _index_enclosing(tree)
    return module


@lru_cache(maxsize=1)
def _load_package() -> dict[str, _Module]:
    modules: dict[str, _Module] = {}
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (SyntaxError, UnicodeDecodeError):  # pragma: no cover
            continue
        dotted = ".".join(path.relative_to(REPO_ROOT).with_suffix("").parts)
        modules[dotted] = _index_module(path, tree, dotted)
    return modules


def _modules_from_source(
    source: str, filename: str = "timer_census_fixture.py"
) -> dict[str, _Module]:
    """A one-module package for fixture/mutation tests of the census machinery.

    The path is synthesized under the real PACKAGE_ROOT so relative-path
    computation matches production; nothing is written to disk.
    """
    tree = ast.parse(textwrap.dedent(source))
    path = PACKAGE_ROOT / filename
    dotted = ".".join(path.relative_to(REPO_ROOT).with_suffix("").parts)
    return {dotted: _index_module(path, tree, dotted)}


class _CallGraph:
    def __init__(self, modules: dict[str, _Module]) -> None:
        self.modules = modules
        self.by_method: dict[str, list[tuple[str, str]]] = defaultdict(list)
        self.bases: dict[tuple[str, str], list[str]] = {}
        for dotted, module in modules.items():
            for cls, method in module.methods:
                self.by_method[method].append((dotted, cls))
            for cls, node in module.classes.items():
                names = []
                for base in node.bases:
                    if isinstance(base, ast.Name):
                        names.append(base.id)
                    elif isinstance(base, ast.Attribute):
                        names.append(base.attr)
                self.bases[(dotted, cls)] = names

    def resolve(self, dotted: str, cls: str | None, name: str):
        module = self.modules.get(dotted)
        if module is None:
            return None
        if cls and (cls, name) in module.methods:
            return (dotted, cls, name, module.methods[(cls, name)])
        for base in self.bases.get((dotted, cls or ""), []):
            if (base, name) in module.methods:
                return (dotted, base, name, module.methods[(base, name)])
        if name in module.funcs:
            return (dotted, None, name, module.funcs[name])
        return None

    def resolve_anywhere(self, dotted: str, cls: str | None, name: str):
        """resolve(), then the unique-global-candidate fallback."""
        target = self.resolve(dotted, cls, name)
        if target is None:
            candidates = self.by_method.get(name, [])
            if len(candidates) == 1:
                target = self.resolve(candidates[0][0], candidates[0][1], name)
        return target

    def reachable(self, dotted: str, cls: str | None, name: str) -> Iterator[tuple]:
        start = self.resolve_anywhere(dotted, cls, name)
        if start is None:
            return
        seen: set[tuple[str, str | None, str]] = set()
        frontier = [(start, 1)]
        while frontier:
            nxt = []
            for (mod, klass, meth, node), depth in frontier:
                key = (mod, klass, meth)
                if key in seen:
                    continue
                seen.add(key)
                yield (mod, klass, meth, node)
                if depth >= MAX_CALL_DEPTH:
                    continue
                for call in (n for n in ast.walk(node) if isinstance(n, ast.Call)):
                    callee = _callee(call)
                    if not callee:
                        continue
                    target = self.resolve_anywhere(mod, klass, callee)
                    if target and (target[0], target[1], target[2]) not in seen:
                        nxt.append((target, depth + 1))
            frontier = nxt


def _collect_clock_roots(
    modules: dict[str, _Module], graph: "_CallGraph"
) -> tuple[list[tuple[str, str, str | None, str]], dict[tuple[str, str, str], str]]:
    """Every *repeating* clock in the package, plus every root that FAILED.

    Returns ``(roots, problems)``.

    ``roots`` entries are ``(kind, module, class, callback_name)``. Kinds:
    the interval-family callee as spelled (``set_interval``,
    ``_set_interval``, ``_create_interval``, ...) and ``rearming-set_timer``
    (a method that schedules itself again -- an interval written as a chain
    of one-shots, which is how the Persona Buddy frame clock is spelled).

    ``problems`` entries are keyed ``(relpath, Class.method, call_text)`` and
    describe a clock constructor call whose callback the census could NOT
    resolve into the call graph. Silence here was TASK-23028's defect: two
    such roots contributed nothing for weeks and nothing noticed. The one
    legitimate unresolvable shape -- a pass-through wrapper
    (``lambda seconds, callback: screen.set_interval(seconds, callback)``,
    or a ``*args`` forwarder) -- is recognized and excused ONLY when the
    wrapper is exposed under a name inside the ``CLOCK_CALLEE_RE`` family,
    because that is what makes its own call sites census-visible.
    """
    roots: list[tuple[str, str, str | None, str]] = []
    problems: dict[tuple[str, str, str], str] = {}
    for dotted, module in modules.items():
        parents: dict[ast.AST, ast.AST] | None = None
        for node in ast.walk(module.tree):
            if not (
                isinstance(node, ast.Call) and CLOCK_CALLEE_RE.match(_callee(node))
            ):
                continue
            kind = _callee(node)
            cls, fn = module.enclosing.get(node, (None, None))
            names: list[str] = []
            if len(node.args) > 1:
                names.extend(_callback_names(node.args[1]))
            for keyword in node.keywords:
                if keyword.arg == "callback":
                    names.extend(_callback_names(keyword.value))
            if parents is None:
                parents = _parent_map(module.tree)
            enclosing = _enclosing_callables(node, parents)
            # A callback name that is a PARAMETER of an enclosing callable is
            # that parameter, not a package symbol: barring it from graph
            # resolution is load-bearing. The wiring lambda
            # `lambda seconds, callback: screen.set_interval(seconds, callback)`
            # once resolved `callback` to the single package method of that
            # name -- a bogus root that also swallowed the pass-through check.
            shadowed = {
                name
                for name in names
                if any(name in _param_names(c) for c in enclosing)
            }
            resolved = [
                name
                for name in names
                if name not in shadowed
                and graph.resolve_anywhere(dotted, cls, name)
            ]
            if resolved:
                for name in resolved:
                    roots.append((kind, dotted, cls, name))
                continue

            # Nothing resolved. The only excusable shape is a pass-through
            # wrapper whose exposed name is itself in the clock family.
            rel = str(module.path.relative_to(REPO_ROOT))
            where = f"{cls}.{fn}"
            call_text = ast.unparse(node)
            starred = {
                arg.value.id
                for arg in node.args
                if isinstance(arg, ast.Starred) and isinstance(arg.value, ast.Name)
            }
            wrapper = None
            for candidate in enclosing:
                params = _param_names(candidate)
                if names and set(names) <= params:
                    wrapper = candidate
                    break
                if not names and starred and starred <= params:
                    wrapper = candidate
                    break
            if wrapper is not None:
                exposed = _exposed_name(wrapper, parents)
                if exposed is not None and CLOCK_CALLEE_RE.match(exposed):
                    continue  # its call sites are clock roots in their own right
                problems[(rel, where, call_text)] = (
                    f"pass-through interval wrapper exposed as {exposed!r}: the "
                    "census can only see its call sites if the wrapper's name "
                    "matches ^_?(create|set)_interval$. Rename the wrapper into "
                    "that family, or add a CLASSIFIED_ROOTS row saying why its "
                    "callbacks need no census."
                )
                continue
            problems[(rel, where, call_text)] = (
                "repeating-clock callback resolves to NOTHING in the package "
                f"call graph (candidate names: {sorted(set(names)) or '<none>'}). "
                "An unresolved root silently censuses nothing -- name the real "
                "callback so the census can follow it, or add a "
                "CLASSIFIED_ROOTS row saying why it cannot."
            )
        for (cls, method), node in module.methods.items():
            for call in (n for n in ast.walk(node) if isinstance(n, ast.Call)):
                if _callee(call) != ONE_SHOT_CLOCK:
                    continue
                names = _callback_names(call.args[1]) if len(call.args) > 1 else []
                for keyword in call.keywords:
                    if keyword.arg == "callback":
                        names.extend(_callback_names(keyword.value))
                if method in names:
                    roots.append((f"rearming-{ONE_SHOT_CLOCK}", dotted, cls, method))
    return roots, problems


def _receiver(call: ast.Call) -> str | None:
    if not isinstance(call.func, ast.Attribute):
        return None
    try:
        return ast.unparse(call.func.value)
    except Exception:  # pragma: no cover
        return None


@lru_cache(maxsize=1)
def _package_graph() -> "_CallGraph":
    return _CallGraph(_load_package())


@lru_cache(maxsize=1)
def _package_clock_roots():
    """(roots, problems) for the real package, computed once per process."""
    return _collect_clock_roots(_load_package(), _package_graph())


def census(
    modules: dict[str, _Module] | None = None,
) -> dict[tuple[str, str, str], dict]:
    """Every ``.update(`` reachable from a repeating clock.

    Keyed ``(file, enclosing Class.method, receiver expression)``. The
    enclosing function is part of the key on purpose: a first draft keyed only
    ``(file, receiver)`` and the un-fixed ``SplashScreen._update_animation``
    site SURVIVED its mutation, because ``_display_static_fallback`` in the
    same module writes to a local also named ``display`` and its allowlist
    entry covered both.

    Receivers the AST PROVES to be dict/set collections (see
    ``_receiver_is_provably_collection``) are excluded here rather than
    carried as NOT-A-WIDGET rows; every unprovable receiver still lands in
    the census and must be classified.

    Args:
        modules: Override for fixture/mutation tests; defaults to the parsed
            package.
    """
    if modules is None:
        modules = _load_package()
        graph = _package_graph()
        roots, _problems = _package_clock_roots()
    else:
        graph = _CallGraph(modules)
        roots, _problems = _collect_clock_roots(modules, graph)
    found: dict[tuple[str, str, str], dict] = {}
    for kind, dotted, cls, callback in roots:
        for mod, klass, meth, node in graph.reachable(dotted, cls, callback):
            module = modules[mod]
            rel = str(module.path.relative_to(REPO_ROOT))
            for call in (n for n in ast.walk(node) if isinstance(n, ast.Call)):
                if _callee(call) != "update":
                    continue
                receiver = _receiver(call)
                if receiver is None:
                    continue
                if _receiver_is_provably_collection(module, klass, node, call):
                    continue
                explicit = any(kw.arg == "layout" for kw in call.keywords)
                record = found.setdefault(
                    (rel, f"{klass}.{meth}", receiver),
                    {"lines": set(), "explicit": True, "roots": set()},
                )
                record["lines"].add(call.lineno)
                record["explicit"] = record["explicit"] and explicit
                record["roots"].add(f"[{kind}] {dotted.split('.')[-1]}:{cls}.{callback}")
    return found


@pytest.fixture(scope="module")
def timer_path_census() -> dict[tuple[str, str, str], dict]:
    """Build the census once for the whole module.

    Module-scoped because :func:`census` parses every file in the package
    (1,889 modules at the time of writing) and every test below asks the same
    question of the same answer.

    Returns:
        The mapping :func:`census` returns -- see its docstring for the key
        and the per-site record.
    """
    return census()


# ---------------------------------------------------------------------------
# the guard
# ---------------------------------------------------------------------------


def test_census_actually_finds_the_known_clock_roots() -> None:
    """The census must be proven to see, or the guard below passes vacuously.

    A silently-broken AST walker that returns nothing would make every other
    assertion in this module green. This pins the floor and names three roots
    that must be found: an app-wide interval, the animation clock TASK-21595
    fixed, and the self-rearming one-shot chain that the naive
    "grep for set_interval" version of this census would miss entirely.
    """
    roots, _problems = _package_clock_roots()
    assert len(roots) >= 30, f"census collapsed to {len(roots)} clock roots"

    flattened = {(dotted.split(".")[-1], cls, name) for _kind, dotted, cls, name in roots}
    assert ("splash_screen", "SplashScreen", "_update_animation") in flattened
    assert (
        "console_composer_bar",
        "ConsoleComposerBar",
        "_toggle_cursor_blink",
    ) in flattened, "the TASK-21692 blink clock disappeared from the census"

    # TASK-23028 regression pins: the clocks the exact-name matcher missed.
    assert (
        "realtime",
        "ConsoleRealtimeController",
        "_tick_console_realtime",
    ) in flattened, (
        "the 10 Hz Console realtime clock (spelled `self._set_interval(...)` "
        "through a constructor-injected callable) left the census again"
    )
    assert (
        "fleet",
        "ConsoleFleetLifecycleController",
        "_console_fleet_survivor_tick",
    ) in flattened, (
        "the fleet survivor tick (spelled `self._create_interval(...)`) left "
        "the census again"
    )
    assert (
        "db_status_manager",
        "DBStatusManager",
        "update_db_sizes",
    ) in flattened, (
        "the DB-size poll's real callback (hidden behind a "
        "`lambda: app.call_later(update_db_sizes)` deferral shim) left the "
        "census again"
    )

    rearming = {r for r in roots if r[0].startswith("rearming-")}
    assert rearming, (
        "no self-rearming set_timer found -- an interval spelled as a chain of "
        "one-shots is exactly the shape a set_interval-only sweep misses"
    )


def test_no_timer_path_update_defaults_to_layout_true(timer_path_census) -> None:
    """Every clock-reachable ``.update(`` is classified.

    Either the call passes ``layout=`` explicitly, or ``CLASSIFIED_SITES`` says
    why it does not have to. A new repaint on a timer lands here unclassified
    and fails, which is the only signal this cost ever gets -- no other test in
    the suite counts layout operations.
    """
    unclassified = sorted(
        (path, qualname, receiver, sorted(rec["lines"]), sorted(rec["roots"]))
        for (path, qualname, receiver), rec in timer_path_census.items()
        if not rec["explicit"] and (path, qualname, receiver) not in CLASSIFIED_SITES
    )
    assert not unclassified, (
        "These `.update(` calls are reachable from a repeating clock and do not "
        "pass `layout=`. Static.update defaults to layout=True, so each of "
        "these arms a whole screen reflow per tick.\n\n"
        "Fix by either passing `layout=False` (only where the rendered size "
        "genuinely cannot change -- prove it with a geometry-equivalence A/B, "
        "see Tests/UI/test_timer_path_layout_cost.py), or by adding an entry to "
        "CLASSIFIED_SITES saying why it does not need to.\n\n"
        + "\n".join(
            f"  {path}:{lines} in {qualname} recv={receiver!r}\n"
            f"      via {roots[:2]}"
            for path, qualname, receiver, lines, roots in unclassified
        )
    )


def test_classified_sites_are_not_stale(timer_path_census) -> None:
    """Every classification still names a real ``.update(`` call.

    A refactor that deletes or renames one of these should retire its entry
    rather than leave folklore behind. Deliberately lenient about
    *reachability* (a call chain moving around is not a finding), strict about
    the call still existing.
    """
    stale: list[str] = []
    for (rel, qualname, receiver), reason in sorted(CLASSIFIED_SITES.items()):
        path = REPO_ROOT / rel
        if not path.exists():
            stale.append(f"  {rel}: file no longer exists ({reason})")
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:  # pragma: no cover
            continue
        enclosing = _index_enclosing(tree)
        live = {
            (
                "{}.{}".format(*enclosing.get(node, (None, None))),
                _receiver(node),
            )
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and _callee(node) == "update"
        }
        if (qualname, receiver) not in live:
            stale.append(
                f"  {rel}: no `.update(` on {receiver!r} in {qualname} any more "
                f"({reason})"
            )
    assert not stale, "Stale CLASSIFIED_SITES entries:\n" + "\n".join(stale)


def test_every_classification_states_a_kind() -> None:
    """Reasons must carry one of the four kinds, not just prose.

    A reason that does not say *which* kind of exemption it is claiming cannot
    be reviewed, and "documented" would degrade into "mentioned".
    """
    kinds = ("NOT-A-WIDGET:", "NEEDS-LAYOUT:", "NOT-PER-TICK:", "UNREACHABLE:")
    bad = [
        f"  {rel} / {qualname} / {receiver}: {reason}"
        for (rel, qualname, receiver), reason in sorted(CLASSIFIED_SITES.items())
        if not reason.startswith(kinds)
    ]
    assert not bad, (
        f"Classifications must start with one of {kinds}:\n" + "\n".join(bad)
    )


# ---------------------------------------------------------------------------
# the guard, part 2: clock roots themselves (TASK-23028)
# ---------------------------------------------------------------------------


def test_clock_roots_all_resolve_loudly() -> None:
    """A clock root the census cannot follow FAILS -- silence is the defect.

    Two roots resolved to nothing for weeks and nothing noticed: the wiring
    lambda recorded its own *parameter* name, and the db-status root recorded
    ``call_later`` instead of the real callback. Both walked zero methods and
    censused zero repaints, while the module stayed green.
    """
    _roots, problems = _package_clock_roots()
    unexplained = {
        key: detail for key, detail in problems.items() if key not in CLASSIFIED_ROOTS
    }
    assert not unexplained, (
        "Repeating-clock roots the census cannot follow (each one censuses "
        "NOTHING until fixed):\n\n"
        + "\n".join(
            f"  {rel} in {where}\n    {call_text}\n    -> {detail}"
            for (rel, where, call_text), detail in sorted(unexplained.items())
        )
    )
    stale = set(CLASSIFIED_ROOTS) - set(problems)
    assert not stale, (
        "CLASSIFIED_ROOTS rows whose root now resolves (or no longer exists) "
        "-- retire them:\n" + "\n".join(f"  {row}" for row in sorted(stale))
    )


def test_clock_root_set_is_pinned() -> None:
    """The root SET is the evidence, never the root count.

    In the window that motivated TASK-23028 the count held at 35 while one
    root silently left the census and another arrived. Equality on the full
    set makes both directions loud.
    """
    modules = _load_package()
    roots, _problems = _package_clock_roots()
    rel_of = {
        dotted: str(module.path.relative_to(REPO_ROOT))
        for dotted, module in modules.items()
    }
    actual = frozenset(
        (kind, rel_of[dotted], cls, name) for kind, dotted, cls, name in roots
    )
    new = actual - EXPECTED_CLOCK_ROOTS
    gone = EXPECTED_CLOCK_ROOTS - actual
    message = []
    if new:
        message.append(
            "NEW repeating-clock roots. Before adding each to "
            "EXPECTED_CLOCK_ROOTS, check what its callback repaints: every "
            "`.update(` it reaches must pass layout= or carry a "
            "CLASSIFIED_SITES row.\n"
            + "\n".join(f"  + {root}" for root in sorted(new))
        )
    if gone:
        message.append(
            "Clock roots DISAPPEARED from the census. If the timer was "
            "genuinely removed, retire its pin; if it was renamed or "
            "re-wired, make sure the census still sees it under the new "
            "spelling before touching the pin.\n"
            + "\n".join(f"  - {root}" for root in sorted(gone))
        )
    assert actual == EXPECTED_CLOCK_ROOTS, "\n\n".join(message)


# ---------------------------------------------------------------------------
# mutation fixtures: each TASK-23028 blind spot, reintroduced, must go RED
# ---------------------------------------------------------------------------
#
# Each fixture is the minimal module that exhibits one of the shapes the
# census was blind to. The assertions are phrased as "the detector fires",
# so deleting or weakening a detector turns exactly its fixture red.


def test_wrapper_spelled_interval_is_censused() -> None:
    """Defect 1: `self._set_interval(...)` is a clock, not a mystery callable.

    This is the exact shape of the 10 Hz Console realtime clock that left the
    census silently: a constructor-injected callable named into the
    ``_set_interval`` family. Reverting CLOCK_CALLEE_RE to the exact name
    ``set_interval`` turns this red.
    """
    modules = _modules_from_source(
        """
        class InjectedClockWidget:
            def __init__(self, set_interval):
                self._set_interval = set_interval

            def start(self):
                self._set_interval(0.1, self._tick)

            def _tick(self):
                self.status.update("tick")
        """
    )
    roots, problems = _collect_clock_roots(modules, _CallGraph(modules))
    assert not problems
    assert [(k, c, n) for k, _d, c, n in roots] == [
        ("_set_interval", "InjectedClockWidget", "_tick")
    ]
    sites = census(modules)
    key = (
        "tldw_chatbook/timer_census_fixture.py",
        "InjectedClockWidget._tick",
        "self.status",
    )
    assert key in sites and not sites[key]["explicit"], (
        "the wrapper-spelled clock's repaint did not reach the census"
    )


def test_deferral_shim_callback_resolves_through() -> None:
    """Defect 3, shape A: `lambda: app.call_later(cb)` censuses ``cb``.

    The db_status_manager root recorded the shim's name (``call_later``) and
    censused nothing. The shim's ARGUMENT is the callback.
    """
    modules = _modules_from_source(
        """
        class ShimClockManager:
            def start(self):
                self.app.set_interval(5.0, lambda: self.app.call_later(self.reconcile))

            def reconcile(self):
                self.status.update("n")
        """
    )
    roots, problems = _collect_clock_roots(modules, _CallGraph(modules))
    assert not problems
    assert ("set_interval", roots[0][1], "ShimClockManager", "reconcile") in roots
    key = (
        "tldw_chatbook/timer_census_fixture.py",
        "ShimClockManager.reconcile",
        "self.status",
    )
    assert key in census(modules)


def test_unresolvable_root_fails_loudly() -> None:
    """Defect 3, shape B: a root that resolves to nothing is a FAILURE.

    The pre-23028 census silently yielded zero reachable methods for such a
    root; deleting the `problems` bookkeeping turns this red.
    """
    modules = _modules_from_source(
        """
        class GhostClock:
            def start(self):
                self.set_interval(1.0, self._ghost)
        """
    )
    roots, problems = _collect_clock_roots(modules, _CallGraph(modules))
    assert not roots
    assert len(problems) == 1
    ((_rel, where, _call), detail) = next(iter(problems.items()))
    assert where == "GhostClock.start"
    assert "resolves to NOTHING" in detail
    assert "_ghost" in detail


def test_unregistered_passthrough_wrapper_fails_loudly() -> None:
    """Defect 3, shape C: a wrapper OUTSIDE the naming family is refused.

    ``arm_refresh(seconds, callback)`` forwards to set_interval, but its call
    sites are invisible to CLOCK_CALLEE_RE -- so accepting it silently would
    recreate the realtime blind spot under a different name. The failure
    must tell the author what to rename it to.
    """
    modules = _modules_from_source(
        """
        class RenamedWrapper:
            def arm_refresh(self, seconds, callback):
                return self.set_interval(seconds, callback)
        """
    )
    roots, problems = _collect_clock_roots(modules, _CallGraph(modules))
    assert not roots
    assert len(problems) == 1
    detail = next(iter(problems.values()))
    assert "pass-through interval wrapper" in detail
    assert "'arm_refresh'" in detail
    assert "_?(create|set)_interval" in detail


def test_clock_family_passthrough_wrappers_are_excused() -> None:
    """The Console wiring shape stays green -- by name, not by silence.

    Both live wiring spellings: the named-parameter forwarder bound to a
    ``create_interval=`` keyword, and the ``*args`` forwarder bound to
    ``set_interval=``. Their call sites are clock roots in their own right
    (CLOCK_CALLEE_RE), so the lambdas themselves census nothing ON PURPOSE.
    """
    modules = _modules_from_source(
        """
        def wire(screen, controller_cls):
            return controller_cls(
                create_interval=lambda seconds, callback: screen.set_interval(
                    seconds, callback
                ),
                set_interval=lambda *args, **kwargs: screen.set_interval(
                    *args, **kwargs
                ),
            )
        """
    )
    roots, problems = _collect_clock_roots(modules, _CallGraph(modules))
    assert not roots
    assert not problems


def test_provable_dict_receiver_is_auto_classified() -> None:
    """Defect 4: a receiver every binding proves to be a dict needs no row."""
    modules = _modules_from_source(
        """
        class DictTickWidget:
            def start(self):
                self.set_interval(1.0, self._tick)

            def _tick(self):
                merged: dict[str, str] = {}
                merged["k"] = "v"
                merged.update(self._pending)
        """
    )
    sites = census(modules)
    assert not any(receiver == "merged" for (_p, _q, receiver) in sites), (
        "a provably-dict receiver still demands a CLASSIFIED_SITES row"
    )


def test_self_attr_set_receiver_is_auto_classified() -> None:
    """Defect 4: `self._seen: set[str] = set()` proves the attribute."""
    modules = _modules_from_source(
        """
        class AttrSetWidget:
            def __init__(self):
                self._seen: set[str] = set()

            def start(self):
                self.set_interval(1.0, self._tick)

            def _tick(self):
                self._seen.update(("a",))
        """
    )
    sites = census(modules)
    assert not any(receiver == "self._seen" for (_p, _q, receiver) in sites)


def test_query_one_receiver_is_never_auto_classified() -> None:
    """The inference must still catch a real Static -- proven, not assumed.

    A guard that cannot fail is decorative (lessons-testing-evidence): this
    is the discrimination proof for _receiver_is_provably_collection. If the
    inference ever guesses toward silence, this fixture stops being censused
    and goes red.
    """
    modules = _modules_from_source(
        """
        class StaticTickWidget:
            def start(self):
                self.set_interval(1.0, self._tick)

            def _tick(self):
                label = self.query_one("#x", Static)
                label.update("y")
        """
    )
    sites = census(modules)
    key = (
        "tldw_chatbook/timer_census_fixture.py",
        "StaticTickWidget._tick",
        "label",
    )
    assert key in sites and not sites[key]["explicit"], (
        "a query_one receiver was auto-classified -- the inference is "
        "guessing toward silence"
    )


def test_mixed_binding_receiver_is_never_auto_classified() -> None:
    """One unprovable binding disqualifies the whole receiver."""
    modules = _modules_from_source(
        """
        class MixedTickWidget:
            def start(self):
                self.set_interval(1.0, self._tick)

            def _tick(self):
                target = {}
                if self.flag:
                    target = self.query_one("#x", Static)
                target.update("y")
        """
    )
    sites = census(modules)
    assert any(receiver == "target" for (_p, _q, receiver) in sites), (
        "a dict-then-widget receiver was auto-classified as a collection"
    )


def test_rebound_self_attr_is_never_auto_classified() -> None:
    """An attribute rebound to a widget anywhere in the class needs a row."""
    modules = _modules_from_source(
        """
        class ReboundAttrWidget:
            def __init__(self):
                self._panel = {}

            def rebind(self):
                self._panel = self.query_one("#p", Static)

            def start(self):
                self.set_interval(1.0, self._tick)

            def _tick(self):
                self._panel.update("y")
        """
    )
    sites = census(modules)
    assert any(receiver == "self._panel" for (_p, _q, receiver) in sites)
