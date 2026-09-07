from dataclasses import dataclass, replace

import pytest

from tldw_chatbook.Chat.console_display_state import (
    ConsoleInspectorState,
    ConsoleStagedContextState,
)
import tldw_chatbook.Chat.console_rail_state as console_rail_state_module
from tldw_chatbook.Chat.console_rail_state import (
    CONSOLE_INSPECTOR_AUTO_OPEN_MAX_COLUMNS,
    CONSOLE_INSPECTOR_AUTO_OPEN_MIN_COLUMNS,
    CONSOLE_RAIL_LEFT_OPEN_EXPLICIT_KEY,
    ConsoleRailPreferences,
    _INACTIVE_STAGED_SUMMARIES,
    _normalized_inactive_text,
    build_console_context_rail_badge,
    build_console_inspector_rail_badge,
    build_console_rail_preference_key,
    build_console_rail_state,
    coerce_console_rail_preferences,
    console_context_reveal_preferences,
    console_rail_left_open_explicit,
    console_rail_width_band,
    resolve_console_rail_priority,
    serialize_console_rail_preferences,
)


@dataclass(frozen=True)
class Row:
    label: str
    status: str = "ready"
    value: str = ""
    text: str = ""


def test_console_rail_state_uses_first_start_defaults():
    key = build_console_rail_preference_key(
        workspace_id="workspace-1",
        session_id="session-1",
    )
    state = build_console_rail_state(preference_key=key)

    assert state.left_open is True
    assert state.right_open is False
    assert state.preferred_left_open is True
    assert state.preferred_right_open is False
    assert state.persistence_key == "console_rail_state:global:shared-layout-v1"


def test_console_rail_state_restores_stored_preferences():
    key = build_console_rail_preference_key(
        workspace_id="workspace-1",
        session_id="session-1",
    )

    state = build_console_rail_state(
        preference_key=key,
        stored_preferences={"left_open": False, "right_open": True},
        available_columns=220,
    )

    assert state.left_open is False
    assert state.right_open is True


def test_console_rail_state_invalid_stored_preferences_fall_back_to_defaults():
    key = build_console_rail_preference_key(
        workspace_id="workspace-1",
        session_id="session-1",
    )

    for invalid_preferences in (
        None,
        "bad",
        {"left_open": "bad"},
        {"right_open": []},
    ):
        state = build_console_rail_state(
            preference_key=key,
            stored_preferences=invalid_preferences,
        )

        assert state.left_open is True
        assert state.right_open is False


def test_console_rail_state_coerces_integer_preferences():
    preferences = coerce_console_rail_preferences(
        {"left_open": 0, "right_open": 1},
    )

    assert preferences.left_open is False
    assert preferences.right_open is True


def test_console_rail_preference_key_workspace_scope_is_per_workspace_only():
    """TASK-718: layout preferences are keyed per workspace. Conversation and
    session ids are accepted for API compatibility but must not shape the key -
    per-conversation keys multiplied config entries and reset section layouts
    on every new chat."""
    key = build_console_rail_preference_key(
        workspace_id="workspace 1",
        conversation_id="conv:1",
        session_id="session:1",
        layout_scope="workspace",
    )
    bare_key = build_console_rail_preference_key(
        workspace_id="workspace 1", layout_scope="workspace"
    )

    assert key.value == "console_rail_state:workspace_1:layout"
    assert bare_key.value == key.value
    # Legacy per-workspace ':global' keys are the one-time migration source.
    assert key.fallback_value == "console_rail_state:workspace_1:global"


def test_console_rail_preference_key_scope_inputs_never_leak_into_key():
    for conversation_id, session_id in ((0, 0), ("   ", 0), ("conv", None)):
        key = build_console_rail_preference_key(
            workspace_id="workspace",
            conversation_id=conversation_id,
            session_id=session_id,
            layout_scope="workspace",
        )
        assert key.value == "console_rail_state:workspace:layout"
        assert key.fallback_value == "console_rail_state:workspace:global"


def test_console_rail_layout_scope_normalizes_to_global_by_default():
    normalize = console_rail_state_module.normalize_console_rail_layout_scope

    class WorkspaceImpostor:
        def __str__(self) -> str:
            return "workspace"

    assert normalize(None) == "global"
    assert normalize("bogus") == "global"
    assert normalize({"workspace": True}) == "global"
    assert normalize(WorkspaceImpostor()) == "global"
    assert normalize(["workspace"]) == "global"
    assert normalize(1) == "global"
    assert normalize(True) == "global"
    assert normalize("  WoRkSpAcE  ") == "workspace"


def test_console_rail_preference_key_global_scope_uses_reserved_shared_key():
    global_key = build_console_rail_preference_key(
        workspace_id="Research Lab", layout_scope="global"
    )
    default_key = build_console_rail_preference_key(workspace_id="Other Workspace")

    assert global_key.value == "console_rail_state:global:shared-layout-v1"
    assert global_key.workspace_id == "global"
    assert global_key.scope_id == "shared-layout-v1"
    assert global_key.fallback_value is None
    assert default_key == global_key


def test_console_rail_preference_key_workspace_scope_keeps_legacy_fallback():
    workspace_key = build_console_rail_preference_key(
        workspace_id="Research Lab", layout_scope="workspace"
    )

    assert workspace_key.value == "console_rail_state:Research_Lab:layout"
    assert workspace_key.scope_id == "layout"
    assert workspace_key.fallback_value == "console_rail_state:Research_Lab:global"


def test_console_context_rail_badge_reflects_workspace_and_session_only():
    """Left badge summarizes workspace/session only (task-400)."""
    # Task-400: staged-context signals moved to the Inspector badge with the
    # staged-sources section; the left badge is workspace/session only.
    assert (
        build_console_context_rail_badge(workspace_label="Research workspace")
        == "workspace"
    )
    assert build_console_context_rail_badge(session_label="Conversation 1") == "session"
    assert build_console_context_rail_badge() == ""


def test_console_inspector_rail_badge_surfaces_staged_context():
    """Inspector badge carries staged context below action-required signals."""
    assert build_console_inspector_rail_badge(staged_source_count=3) == "3 staged"
    assert (
        build_console_inspector_rail_badge(
            staged_source_count="bad",
            staged_summary="Ready staged citations",
        )
        == "staged"
    )
    # Action-required signals keep precedence over staged context...
    assert (
        build_console_inspector_rail_badge(
            staged_source_count=3,
            approval_count=1,
        )
        == "1 approval"
    )
    assert (
        build_console_inspector_rail_badge(staged_source_count=3, tool_count=2)
        == "tools"
    )
    # ...while staged context outranks the informational readiness fallbacks.
    assert (
        build_console_inspector_rail_badge(
            staged_source_count=3,
            can_save_chatbook=True,
        )
        == "3 staged"
    )
    assert (
        build_console_inspector_rail_badge(
            staged_summary="Ready staged citations",
            inspector_rows=(Row("RAG/source", value="available"),),
        )
        == "staged"
    )


def test_console_context_rail_badge_ignores_workspace_fallback_labels():
    for workspace_label in ("", "local", "default", "no workspace", "No-workspace"):
        assert (
            build_console_context_rail_badge(
                workspace_label=workspace_label,
                session_label="Conversation 1",
            )
            == "session"
        )


def test_console_inspector_rail_badge_ignores_empty_staged_summary():
    """Empty and legacy empty-state summaries never trigger the staged badge."""
    # Task-400: the empty state carries no summary line (the tray renders its
    # own guidance copy), so the badge treats it as inactive by emptiness.
    empty_summary = ConsoleStagedContextState.empty().summary
    assert empty_summary == ""

    # Drift guard for legacy payloads: the retired empty-state copy must stay
    # in the inactive set so stored summaries never re-trigger the badge bug
    # where the empty-state summary was treated as "active" staged context.
    assert _normalized_inactive_text("No sources attached.") in (
        _INACTIVE_STAGED_SUMMARIES
    )

    assert build_console_inspector_rail_badge(staged_summary=empty_summary) == ""
    assert (
        build_console_inspector_rail_badge(staged_summary="No sources attached.") == ""
    )


def test_console_context_rail_badge_ignores_default_workspace_display_labels():
    for workspace_label in (
        "No workspace selected",
        "Workspace: Local Default",
        "Workspace: Default",
    ):
        assert (
            build_console_context_rail_badge(
                workspace_label=workspace_label,
                session_label="Conversation 1",
            )
            == "session"
        )
        assert build_console_context_rail_badge(workspace_label=workspace_label) == ""


def test_console_inspector_rail_badge_prioritizes_run_and_review_state():
    assert build_console_inspector_rail_badge(run_status="failed") == "failed"
    assert (
        build_console_inspector_rail_badge(
            run_status="failed",
            inspector_rows=(Row("Policy", status="blocked"),),
        )
        == "failed"
    )
    assert (
        build_console_inspector_rail_badge(
            run_status="streaming",
            inspector_rows=(Row("Policy", status="blocked"),),
            approval_count=2,
        )
        == "blocked"
    )
    assert build_console_inspector_rail_badge(approval_count=1) == "1 approval"
    assert build_console_inspector_rail_badge(approval_count=2) == "2 approvals"
    assert (
        build_console_inspector_rail_badge(
            tool_count=3,
            inspector_rows=(Row("Artifacts", value="Chatbook artifact available"),),
        )
        == "tools"
    )
    assert (
        build_console_inspector_rail_badge(
            inspector_rows=(Row("Artifacts", value="Chatbook artifact available"),),
        )
        == "artifact"
    )
    assert (
        build_console_inspector_rail_badge(
            inspector_rows=(Row("RAG/source", value="3 sources staged"),),
        )
        == "source"
    )
    assert build_console_inspector_rail_badge() == ""


def test_console_inspector_rail_badge_detects_blocked_from_row_fields():
    assert (
        build_console_inspector_rail_badge(
            inspector_rows=(Row("Policy blocked"),),
        )
        == "blocked"
    )
    assert (
        build_console_inspector_rail_badge(
            inspector_rows=(Row("Policy", value="BLOCKED by workspace"),),
        )
        == "blocked"
    )


def test_console_inspector_rail_badge_names_provider_setup_blockers():
    assert (
        build_console_inspector_rail_badge(
            run_status="blocked",
            inspector_rows=(Row("Provider", status="blocked"),),
        )
        == "setup"
    )
    assert (
        build_console_inspector_rail_badge(
            inspector_rows=(Row("Model", value="Missing", text="blocked"),),
        )
        == "setup"
    )


def test_console_inspector_rail_badge_detects_failed_from_row_fields():
    assert (
        build_console_inspector_rail_badge(
            inspector_rows=(Row("Run", text="failed"),),
        )
        == "failed"
    )
    assert (
        build_console_inspector_rail_badge(
            inspector_rows=(Row("FAILED run"),),
        )
        == "failed"
    )


def test_console_inspector_rail_badge_ignores_idle_inspector_rows():
    state = ConsoleInspectorState.from_values(
        provider_ready=True,
        rag_status="not staged",
        artifact_status="unavailable",
    )

    assert build_console_inspector_rail_badge(inspector_rows=state.rows) == ""


def test_console_inspector_rail_badge_ignores_not_requested_source_rows():
    state = ConsoleInspectorState.from_values(
        provider_ready=True,
        rag_status="not requested",
        artifact_status="not available for this item",
    )

    assert build_console_inspector_rail_badge(inspector_rows=state.rows) == ""


def test_console_inspector_rail_badge_detects_positive_artifact_and_source_readiness():
    assert build_console_inspector_rail_badge(can_save_chatbook=True) == "artifact"
    assert (
        build_console_inspector_rail_badge(
            inspector_rows=(Row("Artifacts", value="available"),),
        )
        == "artifact"
    )
    assert (
        build_console_inspector_rail_badge(
            inspector_rows=(Row("Artifacts", value="Chatbook artifact available"),),
        )
        == "artifact"
    )
    assert (
        build_console_inspector_rail_badge(
            inspector_rows=(Row("RAG/source", value="available"),),
        )
        == "source"
    )
    assert (
        build_console_inspector_rail_badge(
            inspector_rows=(Row("RAG/source", value="staged from Library Search/RAG"),),
        )
        == "source"
    )
    assert (
        build_console_inspector_rail_badge(
            inspector_rows=(Row("RAG/source", value="staged"),),
        )
        == "source"
    )
    assert (
        build_console_inspector_rail_badge(
            inspector_rows=(Row("RAG/source", status="ready"),),
        )
        == "source"
    )


def test_console_inspector_rail_badge_does_not_treat_staged_as_source_category():
    assert (
        build_console_inspector_rail_badge(
            inspector_rows=(Row("Review", value="staged"),),
        )
        == ""
    )


def test_console_inspector_rail_badge_requires_label_category_for_artifact_and_source():
    assert (
        build_console_inspector_rail_badge(
            inspector_rows=(Row("Review", value="source available"),),
        )
        == ""
    )
    assert (
        build_console_inspector_rail_badge(
            inspector_rows=(Row("Review", value="artifact available"),),
        )
        == ""
    )
    assert (
        build_console_inspector_rail_badge(
            inspector_rows=(Row("Artifacts", value="available"),),
        )
        == "artifact"
    )
    assert (
        build_console_inspector_rail_badge(
            inspector_rows=(Row("RAG/source", value="available"),),
        )
        == "source"
    )


def test_console_rail_preferences_accept_boolean_strings_case_insensitively():
    for raw_value in ("true", "yes", "1", "on", "TRUE", "Yes", "ON"):
        preferences = coerce_console_rail_preferences({"left_open": raw_value})

        assert preferences.left_open is True

    for raw_value in ("false", "no", "0", "off", "FALSE", "No", "OFF"):
        preferences = coerce_console_rail_preferences({"right_open": raw_value})

        assert preferences.right_open is False


def test_console_rail_preferences_serialize_to_public_dict_shape():
    assert serialize_console_rail_preferences(
        ConsoleRailPreferences(left_open=False, right_open=True),
    ) == {
        "left_open": False,
        "right_open": True,
        "workspace_open": False,
        "conversations_open": True,
        "model_open": False,
        "details_open": False,
        "agent_open": False,
        "character_open": False,
        "inspector_more_open": False,
        "environment_open": True,
        "tasks_open": True,
    }


def test_console_rail_badges_do_not_mutate_open_booleans():
    key = build_console_rail_preference_key(
        workspace_id="workspace-1",
        session_id="session-1",
    )

    state = build_console_rail_state(
        preference_key=key,
        stored_preferences={"left_open": False, "right_open": False},
        staged_source_count=2,
        run_status="blocked",
        inspector_rows=(Row("Provider", status="blocked"),),
    )

    assert state.left_open is False
    assert state.right_open is False
    # Task-400: staged context no longer badges the left handle; the blocked
    # provider row still wins the right badge over the staged count.
    assert state.left_badge == ""
    assert state.right_badge == "setup"


def test_console_rail_state_routes_staged_context_to_inspector_badge():
    """Rail-state build surfaces staged context on the right badge only."""
    key = build_console_rail_preference_key(
        workspace_id="workspace-1",
        session_id="session-1",
    )

    state = build_console_rail_state(
        preference_key=key,
        stored_preferences={"left_open": False, "right_open": False},
        staged_source_count=2,
    )

    assert state.left_badge == ""
    assert state.right_badge == "2 staged"


def test_console_rail_state_compact_width_collapses_right_rail_by_default():
    """The 150-col compact collapse is the responsive DEFAULT (TASK-2154.2):
    with no explicit toggle stored, the right rail renders closed below the
    threshold exactly as before."""
    key = build_console_rail_preference_key(
        workspace_id="workspace-1",
        session_id="session-1",
    )

    for no_explicit_toggle in (None, {}, {"left_open": True}, {"right_open": False}):
        state = build_console_rail_state(
            preference_key=key,
            stored_preferences=no_explicit_toggle,
            available_columns=120,
        )

        assert state.left_open is True
        assert state.right_open is False
        assert state.right_forced_collapsed is True
        assert state.right_compact_override is False
        assert state.compact_override is False


def test_console_rail_state_compact_width_honors_explicit_right_open():
    """TASK-2154.2 (LY-11): an explicit right_open=True is honored below the
    150-col threshold -- the force rule is the default, not a hard block --
    and the compact override flags report it for the min-width waiver."""
    key = build_console_rail_preference_key(
        workspace_id="workspace-1",
        session_id="session-1",
    )

    state = build_console_rail_state(
        preference_key=key,
        stored_preferences={"left_open": True, "right_open": True},
        available_columns=120,
    )

    assert state.left_open is True
    assert state.right_open is True
    assert state.preferred_right_open is True
    assert state.right_forced_collapsed is False
    assert state.right_compact_override is True
    assert state.left_compact_override is False
    assert state.compact_override is True


def test_console_rail_state_narrow_width_collapses_left_rail_by_default():
    """TASK-2154.1 (LY-08) + TASK-2154.2: below 100 cols the left rail
    force-collapses as the responsive default while the user has never
    explicitly toggled it; the (absent) stored preference stays untouched.

    A stored ``left_open`` value WITHOUT the explicit-toggle marker (e.g. a
    legacy full-payload write from toggling the OTHER rail) must keep the
    collapse: every write serializes all keys, so key presence alone can
    never mark explicitness (the 2026-08-05 UAT regression this guards).
    """
    key = build_console_rail_preference_key(
        workspace_id="workspace-1",
        session_id="session-1",
    )

    for no_explicit_toggle in (
        None,
        {},
        {"left_open": True},
        {"right_open": True},
        {"left_open": True, "left_open_explicit": False},
    ):
        state = build_console_rail_state(
            preference_key=key,
            stored_preferences=no_explicit_toggle,
            available_columns=90,
        )

        assert state.left_open is False
        assert state.left_forced_collapsed is True
        assert state.preferred_left_open is True
        assert state.left_compact_override is False
        assert state.single_pane is False


def test_console_rail_state_narrow_width_honors_explicit_left_open():
    """TASK-2154.2: a marker-flagged explicit left_open=True is honored
    below 100 cols -- the left default is OPEN, so only the marker written
    alongside the toggle gesture can tell this apart from the
    never-toggled (or legacy full-payload) default above."""
    key = build_console_rail_preference_key(
        workspace_id="workspace-1",
        session_id="session-1",
    )

    state = build_console_rail_state(
        preference_key=key,
        stored_preferences={"left_open": True, "left_open_explicit": True},
        available_columns=90,
    )

    assert state.left_open is True
    assert state.left_forced_collapsed is False
    assert state.preferred_left_open is True
    assert state.left_compact_override is True
    assert state.compact_override is True
    assert state.single_pane is False


@pytest.mark.parametrize(
    ("width", "expected_open", "expected_override"),
    [(99, False, False), (100, True, True), (101, True, False)],
)
def test_console_rail_state_default_left_boundary(
    width: int,
    expected_open: bool,
    expected_override: bool,
):
    key = build_console_rail_preference_key(
        workspace_id="workspace-1",
        session_id="session-1",
    )

    state = build_console_rail_state(
        preference_key=key,
        stored_preferences={"left_open": True},
        available_columns=width,
    )

    assert state.left_open is expected_open
    assert state.preferred_left_open is True
    assert state.left_compact_override is expected_override
    assert state.compact_override is expected_override


def test_console_rail_state_single_pane_below_84_columns():
    """TASK-2154.1 (LY-09): below 84 cols the workspace drops to one pane.

    TASK-2154.2 originally honored an explicitly opened rail even in
    single-pane mode. task-18911 (2026-08-19 mobile audit) narrowed that:
    below the width budget (single-pane floor + rail min-width) the collapse
    is a rendering override the explicit marker cannot buy past -- at 60
    cols an honored 30-col rail left a 14-col transcript. Above the budget
    the marker is still honored (see the *_width_budget tests below).
    """
    key = build_console_rail_preference_key(
        workspace_id="workspace-1",
        session_id="session-1",
    )

    state = build_console_rail_state(
        preference_key=key,
        available_columns=60,
    )

    assert state.single_pane is True
    assert state.left_open is False
    assert state.left_forced_collapsed is True
    assert state.compact_override is False

    explicit = build_console_rail_state(
        preference_key=key,
        stored_preferences={
            "left_open": True,
            "left_open_explicit": True,
            "right_open": True,
        },
        available_columns=60,
    )

    assert explicit.single_pane is True
    assert explicit.left_open is False
    assert explicit.right_open is False
    assert explicit.left_forced_collapsed is True
    assert explicit.right_forced_collapsed is True
    assert explicit.left_compact_override is False
    assert explicit.right_compact_override is False
    assert explicit.compact_override is False


def test_console_rail_state_no_responsive_overrides_without_width():
    key = build_console_rail_preference_key(
        workspace_id="workspace-1",
        session_id="session-1",
    )

    state = build_console_rail_state(
        preference_key=key,
        stored_preferences={"left_open": True, "right_open": True},
    )

    assert state.left_open is True
    assert state.right_open is True
    assert state.left_forced_collapsed is False
    assert state.right_forced_collapsed is False
    assert state.left_compact_override is False
    assert state.right_compact_override is False
    assert state.compact_override is False
    assert state.single_pane is False


def test_console_rail_left_open_explicit_marker_helper():
    """The explicit-toggle marker is the ONLY thing the helper reads, and
    it tolerates legacy/hand-edited payloads."""
    assert CONSOLE_RAIL_LEFT_OPEN_EXPLICIT_KEY == "left_open_explicit"
    assert console_rail_left_open_explicit(None) is False
    assert console_rail_left_open_explicit("bad") is False
    assert console_rail_left_open_explicit({}) is False
    assert console_rail_left_open_explicit({"left_open": True}) is False
    assert console_rail_left_open_explicit({"left_open_explicit": True}) is True
    assert console_rail_left_open_explicit({"left_open_explicit": "true"}) is True
    assert console_rail_left_open_explicit({"left_open_explicit": 0}) is False


@pytest.mark.parametrize(
    ("width", "expected_left", "expected_right", "expected_override"),
    [
        (99, True, True, True),
        (100, False, True, True),
        (149, False, True, True),
        (150, True, True, False),
    ],
)
def test_console_rail_priority_resolves_two_open_rails(
    width: int,
    expected_left: bool,
    expected_right: bool,
    expected_override: bool,
):
    key = build_console_rail_preference_key(workspace_id="workspace-1")
    state = build_console_rail_state(
        preference_key=key,
        stored_preferences={
            "left_open": True,
            "left_open_explicit": True,
            "right_open": True,
        },
        available_columns=width,
    )
    state = replace(
        state,
        left_badge="workspace",
        right_badge="blocked",
        persistence_key="sentinel-key",
        details_open=True,
    )
    snapshot = replace(state)

    resolved = resolve_console_rail_priority(state, width)

    assert state == snapshot
    if 100 <= width < 150:
        # TASK-23197 added two fields to the eviction: it now records that
        # the rail was FORCED closed (not merely closed) and replaces the
        # stub's badge with the reason, so the user is not left watching a
        # panel vanish with no explanation.
        assert resolved == replace(
            snapshot,
            left_open=False,
            left_compact_override=False,
            right_compact_override=True,
            compact_override=True,
            left_forced_collapsed=True,
        )
    else:
        assert resolved is state
    assert resolved.left_open is expected_left
    assert resolved.right_open is expected_right
    assert resolved.preferred_left_open is True
    assert resolved.right_compact_override is expected_override
    assert resolved.compact_override is expected_override


@pytest.mark.parametrize(
    ("width", "right_open", "expected"),
    [
        (99, True, {"left_open": True}),
        (100, True, {"left_open": True, "right_open": False}),
        (149, True, {"left_open": True, "right_open": False}),
        (150, True, {"left_open": True}),
        (120, False, {"left_open": True}),
    ],
)
def test_console_context_reveal_preferences_switches_from_effective_inspector(
    width: int,
    right_open: bool,
    expected: dict[str, bool],
):
    key = build_console_rail_preference_key(workspace_id="workspace-1")
    state = build_console_rail_state(
        preference_key=key,
        stored_preferences={"left_open": False, "right_open": right_open},
        available_columns=width,
    )

    assert console_context_reveal_preferences(state, width) == expected


def test_console_inspector_auto_open_bounds_are_shared_contracts():
    assert CONSOLE_INSPECTOR_AUTO_OPEN_MIN_COLUMNS == 118
    assert CONSOLE_INSPECTOR_AUTO_OPEN_MAX_COLUMNS == 128


@pytest.mark.parametrize(
    ("width", "expected"),
    [
        (None, "standard"),
        (60, "single-pane"),
        (83, "single-pane"),
        (84, "narrow"),
        (99, "narrow"),
        (100, "exact-left-boundary"),
        (101, "compact-before-auto-open"),
        (117, "compact-before-auto-open"),
        (118, "compact-auto-open"),
        (128, "compact-auto-open"),
        (129, "compact-after-auto-open"),
        (149, "compact-after-auto-open"),
        (150, "standard"),
        (160, "standard"),
    ],
)
def test_console_rail_width_band_buckets(width: int | None, expected: str):
    assert console_rail_width_band(width) == expected


def test_console_rail_state_wide_default_layout_unchanged():
    """TASK-2154.2 AC: the 160-col default (left open, right closed) is
    untouched by the explicit-toggle override semantics."""
    key = build_console_rail_preference_key(
        workspace_id="workspace-1",
        session_id="session-1",
    )

    state = build_console_rail_state(preference_key=key, available_columns=160)

    assert state.left_open is True
    assert state.right_open is False
    assert state.left_forced_collapsed is False
    assert state.right_forced_collapsed is False
    assert state.compact_override is False


def test_console_rail_state_explicit_right_open_at_threshold_renders_standard():
    """At/above 150 cols an explicit right_open=True is a STANDARD open:
    no force-collapse and no compact override (min widths stay 56/34)."""
    key = build_console_rail_preference_key(
        workspace_id="workspace-1",
        session_id="session-1",
    )

    state = build_console_rail_state(
        preference_key=key,
        stored_preferences={"right_open": True},
        available_columns=150,
    )

    assert state.right_open is True
    assert state.right_forced_collapsed is False
    assert state.right_compact_override is False
    assert state.compact_override is False


def test_console_rail_section_defaults():
    from tldw_chatbook.Chat.console_rail_state import (
        CONSOLE_RAIL_PREFERENCE_DISCLOSURE_IDS,
        CONSOLE_RAIL_SECTION_IDS,
    )

    prefs = ConsoleRailPreferences()
    # Task-400: "context" (staged sources) is no longer a left-rail section;
    # it renders in the Inspector rail instead. P3c added "character".
    assert CONSOLE_RAIL_SECTION_IDS == (
        "workspace",
        "conversations",
        "model",
        "details",
        "agent",
        "character",
    )
    # TASK-23193: only the two sections a user navigates by ship open. A
    # 2026-08-29 audit measured the previous five-open default at 51 rows
    # against a 32-row viewport at 160x48 -- it overflowed at every one of
    # ten terminal geometries, including 200x60, hiding three sections
    # entirely on a fresh install.
    # TASK-23199 retired the Sessions section; session_open survives only as
    # a legacy migration seed inside coerce_console_rail_preferences.
    assert not hasattr(prefs, "session_open")
    assert prefs.workspace_open is False
    assert prefs.conversations_open is True
    assert prefs.model_open is False
    assert prefs.details_open is False
    assert prefs.character_open is False
    assert prefs.inspector_more_open is False
    # TASK-8: Environment and Tasks are Inspector-rail disclosures (not
    # left-rail sections), and default open.
    assert prefs.environment_open is True
    assert prefs.tasks_open is True
    assert CONSOLE_RAIL_PREFERENCE_DISCLOSURE_IDS == (
        *CONSOLE_RAIL_SECTION_IDS,
        "inspector_more",
        "environment",
        "tasks",
    )


def test_coerce_console_rail_preferences_reads_section_fields():
    coerced = coerce_console_rail_preferences(
        {
            "left_open": True,
            "details_open": "true",
            "model_open": "off",
            "workspace_open": "false",
            "conversations_open": "on",
        }
    )
    assert coerced.details_open is True
    assert coerced.model_open is False
    assert coerced.workspace_open is False
    assert coerced.conversations_open is True


def test_coerce_console_rail_preferences_migrates_legacy_session_collapse():
    """A pre-TASK-14810 payload still seeds the sections that outlived it.

    Before that split there was one mixed "Session" body, so an old payload
    carries only ``session_open``. TASK-23199 then retired the Sessions
    section itself -- but the seed must keep working, or users whose stored
    layout predates the split silently lose their collapsed rail.
    """
    collapsed = coerce_console_rail_preferences({"session_open": False})
    assert collapsed.workspace_open is False
    assert collapsed.conversations_open is False
    assert not hasattr(collapsed, "session_open")

    expanded = coerce_console_rail_preferences({"session_open": True})
    assert expanded.workspace_open is True
    assert expanded.conversations_open is True

    # A modern explicit flag still beats the legacy seed.
    mixed = coerce_console_rail_preferences(
        {"session_open": True, "workspace_open": False}
    )
    assert mixed.workspace_open is False
    assert mixed.conversations_open is True


def test_serialize_console_rail_preferences_round_trips_sections():
    prefs = ConsoleRailPreferences(
        details_open=True,
        model_open=False,
        conversations_open=False,
        inspector_more_open=True,
    )
    serialized = serialize_console_rail_preferences(prefs)
    assert serialized["details_open"] is True
    assert serialized["model_open"] is False
    assert serialized["conversations_open"] is False
    assert serialized["inspector_more_open"] is True
    assert "context_open" not in serialized
    assert coerce_console_rail_preferences(serialized) == prefs


def test_coerce_console_rail_preferences_ignores_legacy_context_key():
    """Stored payloads with the retired context_open key coerce cleanly."""
    # Task-400 migration path: payloads persisted while the rail still had a
    # Context section keep a context_open key; it must be ignored, not fail.
    with_legacy_key = coerce_console_rail_preferences(
        {"left_open": False, "context_open": False, "details_open": True}
    )
    without_legacy_key = coerce_console_rail_preferences(
        {"left_open": False, "details_open": True}
    )
    assert with_legacy_key == without_legacy_key


def test_console_rail_preferences_ignore_transient_view_state():
    transient = {
        "left_open": False,
        "inspector_more_open": "yes",
        "left_scroll_offset": 19,
        "right_scroll_offset": 23,
        "focused_widget_id": "console-workspace-tree",
        "workspace_search": "research",
        "conversation_search": "draft",
        "selected_workspace_id": "workspace-1",
        "selected_conversation_id": "conversation-2",
        "tooltip_text": "A long workspace label",
        "tooltip_target_id": "workspace-1",
    }

    preferences = coerce_console_rail_preferences(transient)
    serialized = serialize_console_rail_preferences(preferences)

    assert preferences.inspector_more_open is True
    assert serialized["inspector_more_open"] is True
    assert not (
        set(serialized)
        & {
            "left_scroll_offset",
            "right_scroll_offset",
            "focused_widget_id",
            "workspace_search",
            "conversation_search",
            "selected_workspace_id",
            "selected_conversation_id",
            "tooltip_text",
            "tooltip_target_id",
        }
    )


def test_build_console_rail_state_carries_section_flags():
    key = build_console_rail_preference_key(workspace_id="ws", session_id="s")
    state = build_console_rail_state(
        preference_key=key,
        stored_preferences={
            "details_open": True,
            "workspace_open": False,
            "conversations_open": True,
        },
    )
    assert state.details_open is True
    assert state.workspace_open is False
    assert state.conversations_open is True
    # Unlisted flags fall back to the shipped default rather than to True.
    assert state.model_open is ConsoleRailPreferences().model_open
    assert state.inspector_more_open is False


def test_build_console_rail_state_carries_inspector_more_preference():
    key = build_console_rail_preference_key(layout_scope="global")

    state = build_console_rail_state(
        preference_key=key,
        stored_preferences={"inspector_more_open": True},
    )

    assert state.inspector_more_open is True


# --- task-18911: width-budget rule (explicit toggles vs usable transcript) ---


def test_explicit_left_open_honored_when_width_affords_it():
    """30 + 40 = 70: exactly the left budget, explicit open is honored."""
    key = build_console_rail_preference_key(workspace_id="w", session_id="s")
    state = build_console_rail_state(
        preference_key=key,
        stored_preferences={"left_open": True, "left_open_explicit": True},
        available_columns=70,
    )
    assert state.left_open is True
    assert state.left_forced_collapsed is False


def test_explicit_left_open_collapsed_below_width_budget():
    """69 cols (inside the compact-collapse zone) with the explicit marker:
    ADR-043 honored the open; task-18911 collapses it because 69 < 30+40 --
    rail + usable transcript does not fit. At 70+ the marker is honored
    again (the honored-at-budget test above pins that side)."""
    key = build_console_rail_preference_key(workspace_id="w", session_id="s")
    state = build_console_rail_state(
        preference_key=key,
        stored_preferences={"left_open": True, "left_open_explicit": True},
        available_columns=69,
    )
    assert state.left_open is False
    assert state.left_forced_collapsed is True
    assert state.single_pane is True  # 69 < 84: single-pane floor applies too


def test_explicit_right_open_collapsed_below_width_budget():
    """73 cols: right rail open by explicit value, but 73 < 34+40 -- an
    honored 34-col rail would leave a 39-col transcript."""
    key = build_console_rail_preference_key(workspace_id="w", session_id="s")
    state = build_console_rail_state(
        preference_key=key,
        stored_preferences={"right_open": True},
        available_columns=73,
    )
    assert state.right_open is False
    assert state.right_forced_collapsed is True


def test_explicit_right_open_honored_at_budget():
    key = build_console_rail_preference_key(workspace_id="w", session_id="s")
    state = build_console_rail_state(
        preference_key=key,
        stored_preferences={"right_open": True},
        available_columns=74,
    )
    assert state.right_open is True
    assert state.right_forced_collapsed is False


def test_phone_width_explicit_left_open_collapsed():
    """The 2026-08-19 mobile-audit repro: phone width (48 cols) with the
    explicit marker stored -- transcript must get the full width."""
    key = build_console_rail_preference_key(workspace_id="w", session_id="s")
    state = build_console_rail_state(
        preference_key=key,
        stored_preferences={"left_open": True, "left_open_explicit": True},
        available_columns=48,
    )
    assert state.single_pane is True
    assert state.left_open is False
    assert state.left_forced_collapsed is True


# --- task-8: persisted Environment/Tasks section-collapse booleans ---


def test_environment_and_tasks_open_default_true_and_round_trip():
    defaults = ConsoleRailPreferences()
    assert defaults.environment_open is True
    assert defaults.tasks_open is True
    serialized = serialize_console_rail_preferences(defaults)
    assert serialized["environment_open"] is True and serialized["tasks_open"] is True
    coerced = coerce_console_rail_preferences(
        {"environment_open": False, "tasks_open": False}
    )
    assert coerced.environment_open is False and coerced.tasks_open is False


def test_disclosure_ids_accept_environment_and_tasks():
    from tldw_chatbook.Chat.console_rail_state import (
        CONSOLE_RAIL_PREFERENCE_DISCLOSURE_IDS,
    )

    assert "environment" in CONSOLE_RAIL_PREFERENCE_DISCLOSURE_IDS
    assert "tasks" in CONSOLE_RAIL_PREFERENCE_DISCLOSURE_IDS


def test_coerce_garbage_falls_back_to_defaults():
    coerced = coerce_console_rail_preferences({"environment_open": "banana"})
    assert coerced.environment_open is True
