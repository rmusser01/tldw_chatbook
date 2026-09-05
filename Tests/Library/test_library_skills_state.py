from tldw_chatbook.Library.library_skills_state import (
    SkillBrowseScope,
    apply_skill_browse_result,
    begin_skill_browse,
    build_skill_browse_result,
    coerce_skill_editor_mode,
    reconcile_skill_allowed_tools,
    SkillEditorSupportingFile,
    build_skill_editor_state,
    build_skills_list_state,
    classify_skill_save_error,
    compose_skill_markdown,
    save_marks_needs_review,
    skill_allowed_tools_sequence,
    skill_flags_line,
    skill_invocation_copy,
    skill_name_shadows_builtin,
    skill_trust_requires_details,
)
from tldw_chatbook.Skills_Interop.skill_trust_models import SkillTrustBlockedError


def _ctx(available=(), blocked=()):
    return {"available_skills": list(available), "blocked_skills": list(blocked)}


def _summary(name, **over):
    base = {
        "name": name,
        "description": f"{name} desc",
        "argument_hint": None,
        "user_invocable": True,
        "disable_model_invocation": False,
        "trust_status": "trusted",
        "trust_blocked": False,
    }
    base.update(over)
    return base


def test_list_renders_trusted_and_blocked_with_glyphs():
    state = build_skills_list_state(
        _ctx(
            available=[_summary("alpha")],
            blocked=[
                _summary(
                    "bravo", trust_status="quarantined_modified", trust_blocked=True
                )
            ],
        ),
        query="",
        sort="name",
    )
    by_name = {r.name: r for r in state.rows}
    assert by_name["alpha"].trust_glyph == "✓" and by_name["alpha"].blocked is False
    assert by_name["bravo"].trust_glyph == "⚠" and by_name["bravo"].blocked is True
    assert state.count == 2


def test_list_marks_the_selected_skill_independently_of_focus():
    state = build_skills_list_state(
        _ctx(available=[_summary("alpha"), _summary("bravo")]),
        query="",
        sort="name",
        selected_name="bravo",
    )

    by_name = {row.name: row for row in state.rows}
    assert by_name["alpha"].selected is False
    assert by_name["bravo"].selected is True


def test_status_sort_puts_needs_review_first():
    state = build_skills_list_state(
        _ctx(
            available=[_summary("zeta")],
            blocked=[_summary("aardvark", trust_blocked=True)],
        ),
        query="",
        sort="status",
    )
    assert [r.name for r in state.rows] == ["aardvark", "zeta"]


def test_query_matches_name_and_description():
    state = build_skills_list_state(
        _ctx(
            available=[
                _summary("code-review", description="Review pull requests"),
                _summary("summarize", description="Shorten text"),
            ]
        ),
        query="pull",
        sort="name",
    )
    assert [r.name for r in state.rows] == ["code-review"]


def test_skill_browse_scope_is_local_bounded_and_fingerprints_full_scope():
    scope = SkillBrowseScope(query="  Needle  ", sort="STATUS", page=2)

    assert scope.backend == "local"
    assert scope.query == "Needle"
    assert scope.sort == "status"
    assert scope.page_size == 20
    assert scope.fingerprint != SkillBrowseScope(query="Needle", page=1).fingerprint


def test_skill_browse_result_validates_exact_page_and_source_wide_trust():
    scope = SkillBrowseScope(query="needle", sort="status", page=2)
    result = build_skill_browse_result(
        scope,
        {
            "skills": [_summary("skill-20", trust_blocked=True)],
            "count": 1,
            "total": 21,
            "limit": 20,
            "offset": 20,
            "blocked_total": 4,
            "first_blocked_skill_name": "blocked-alpha",
        },
        request_token=7,
    )

    assert result.page == 2
    assert result.total_items == 21
    assert result.blocked_total == 4
    assert result.first_blocked_skill_name == "blocked-alpha"
    assert result.items[0]["name"] == "skill-20"


def test_skill_browse_result_rejects_duplicate_or_inexact_page_rows():
    scope = SkillBrowseScope(page=1)
    duplicate = _summary("same")

    for skills, count, total in (
        ([duplicate, duplicate], 2, 2),
        ([_summary("only")], 1, 2),
    ):
        try:
            build_skill_browse_result(
                scope,
                {
                    "skills": skills,
                    "count": count,
                    "total": total,
                    "limit": 20,
                    "offset": 0,
                    "blocked_total": 0,
                    "first_blocked_skill_name": None,
                },
            )
        except ValueError:
            pass
        else:
            raise AssertionError("malformed Skill page must fail closed")


def test_skill_browse_result_rejects_contradictory_blocked_recovery_metadata():
    scope = SkillBrowseScope()

    try:
        build_skill_browse_result(
            scope,
            {
                "skills": [],
                "count": 0,
                "total": 0,
                "limit": 20,
                "offset": 0,
                "blocked_total": 0,
                "first_blocked_skill_name": "blocked-alpha",
            },
        )
    except ValueError:
        pass
    else:
        raise AssertionError("zero blocked total cannot expose a review target")


def test_skill_browse_result_applies_only_to_matching_loading_generation():
    scope = SkillBrowseScope()
    loading = begin_skill_browse(scope, request_token=3)
    late = build_skill_browse_result(
        scope,
        {
            "skills": [],
            "count": 0,
            "total": 0,
            "limit": 20,
            "offset": 0,
            "blocked_total": 0,
            "first_blocked_skill_name": None,
        },
        request_token=2,
    )

    assert apply_skill_browse_result(loading, late) is loading


def test_flags_line_variants():
    # task-418 copy pass: spell the invocability out instead of the bare
    # "user · agent" tokens (no legend existed anywhere in the UI).
    assert skill_flags_line(True, False) == "invocable: user & agent"
    assert skill_flags_line(True, True) == "invocable: user only"
    assert skill_flags_line(False, False) == "invocable: agent only"
    assert skill_flags_line(False, True) == "not invocable"


def test_skill_editor_mode_defaults_to_basic_and_accepts_only_known_values():
    assert coerce_skill_editor_mode(None) == "basic"
    assert coerce_skill_editor_mode("") == "basic"
    assert coerce_skill_editor_mode("expert") == "basic"
    assert coerce_skill_editor_mode("basic") == "basic"
    assert coerce_skill_editor_mode("advanced") == "advanced"


def test_skill_invocation_copy_treats_user_and_agent_as_independent_choices():
    assert (
        skill_invocation_copy(True, False) == "You and the agent can invoke this Skill."
    )
    assert skill_invocation_copy(True, True) == "Only you can invoke this Skill."
    assert (
        skill_invocation_copy(False, False) == "Only the agent can invoke this Skill."
    )
    assert skill_invocation_copy(False, True) == (
        "Reference only — neither you nor the agent can invoke this Skill."
    )


def test_skill_trust_details_expand_only_for_actionable_safety_state():
    assert skill_trust_requires_details("trusted", False, ()) is False
    assert skill_trust_requires_details("pending_review", False, ()) is True
    assert skill_trust_requires_details("trusted", True, ()) is True
    assert skill_trust_requires_details("trusted", False, ("scripts/check.py",)) is True


def test_skill_allowed_tools_sequence_preserves_order_duplicates_and_unknowns():
    assert skill_allowed_tools_sequence(
        "read_file, mystery, read_file, calculator"
    ) == (
        "read_file",
        "mystery",
        "read_file",
        "calculator",
    )


def test_skill_allowed_tools_stay_exact_until_the_picker_is_explicitly_edited():
    captured = ("read_file", "mystery", "read_file", "calculator")
    assert (
        reconcile_skill_allowed_tools(
            captured,
            selected=("calculator",),
            catalog_order=("calculator", "read_file", "write_file"),
            picker_changed=False,
        )
        == captured
    )


def test_skill_allowed_tools_reconcile_only_known_user_edits_losslessly():
    captured = ("read_file", "mystery", "read_file", "calculator")
    assert reconcile_skill_allowed_tools(
        captured,
        selected=("read_file", "write_file"),
        catalog_order=("calculator", "read_file", "write_file"),
        picker_changed=True,
    ) == ("read_file", "mystery", "read_file", "write_file")


def test_shadow_predicate():
    assert skill_name_shadows_builtin("calculator") == "calculator"
    assert skill_name_shadows_builtin("skills") == "skills"
    assert skill_name_shadows_builtin("code-review") is None


def test_save_marks_needs_review_only_when_currently_trusted():
    assert save_marks_needs_review("trusted", False) is True
    assert save_marks_needs_review("quarantined_modified", True) is False


def test_editor_state_splits_frontmatter_and_body():
    detail = {
        "name": "code-review",
        "description": "Review code",
        "argument_hint": "[path]",
        "allowed_tools": ["calculator"],
        "user_invocable": True,
        "disable_model_invocation": False,
        "context": "inline",
        "model": None,
        "version": 3,
        "trust_status": "trusted",
        "trust_blocked": False,
        "supporting_files": {"notes.md": "hello"},
        "content": "---\nname: code-review\ndescription: Review code\n---\nReview {{args}} now.",
    }
    state = build_skill_editor_state(detail)
    assert state.name == "code-review" and state.argument_hint == "[path]"
    assert state.allowed_tools_csv == "calculator"
    assert state.body.strip() == "Review {{args}} now."
    # No bundle_files on this detail -> falls back to supporting_files
    # (task-11: nested/binary listing only kicks in when bundle_files is
    # present), reduced to SkillEditorSupportingFile rows (is_text=True,
    # the fallback path's default -- it only ever carries decoded text).
    assert state.supporting_files == (
        SkillEditorSupportingFile(name="notes.md", size=5, is_text=True),
    )
    assert state.version == 3


def test_editor_lists_nested_and_marks_binary():
    from tldw_chatbook.Library.library_skills_state import build_skill_editor_state

    detail = {
        "name": "demo",
        "content": "body",
        "version": 1,
        "supporting_files": {"references/api.md": "# api\n"},
        "bundle_files": [
            {
                "path": "references/api.md",
                "size": 6,
                "executable": False,
                "is_text": True,
            },
            {
                "path": "assets/logo.png",
                "size": 2048,
                "executable": False,
                "is_text": False,
            },
        ],
        "trust_status": "trusted",
        "trust_blocked": False,
    }
    state = build_skill_editor_state(detail)
    names = [f.name for f in state.supporting_files]
    assert "references/api.md" in names
    assert "assets/logo.png" in names  # binary listed
    binary = next(f for f in state.supporting_files if f.name == "assets/logo.png")
    assert binary.is_text is False  # view-only marker


def test_compose_roundtrips_through_frontmatter_grammar():
    detail = {
        "name": "code-review",
        "description": "Review code",
        "argument_hint": None,
        "allowed_tools": None,
        "user_invocable": True,
        "disable_model_invocation": False,
        "context": "fork",
        "model": None,
        "version": 1,
        "trust_status": "trusted",
        "trust_blocked": False,
        "supporting_files": None,
        "content": "---\nname: code-review\ndescription: Review code\n---\nBody here.",
    }
    state = build_skill_editor_state(detail)
    text = compose_skill_markdown(state, body="New body {{args}}")
    assert text.startswith("---\n") and "name: code-review" in text
    assert text.rstrip().endswith("New body {{args}}")


def test_classify_outcomes():
    assert classify_skill_save_error(None, "local_skill_exists:x", None) == "exists"
    assert (
        classify_skill_save_error(None, "local_skill_version_conflict:x", None)
        == "version-conflict"
    )
    assert (
        classify_skill_save_error(
            None,
            "",
            SkillTrustBlockedError(
                skill_name="x",
                reason_code="skill_modified",
                trust_status="quarantined_modified",
            ),
        )
        == "trust-blocked"
    )
    assert classify_skill_save_error({"name": "x"}, "", None) == "ok"


def test_pure_module_has_no_forbidden_imports():
    import tldw_chatbook.Library.library_skills_state as mod

    with open(mod.__file__, encoding="utf-8") as handle:
        src = handle.read()
    for forbidden in (
        "textual",
        "sqlite3",
        "tldw_chatbook.DB",
        "tldw_chatbook.app",
        "httpx",
        "requests",
    ):
        assert forbidden not in src


def test_shadow_name_set_stays_in_sync_with_real_sources():
    """Drift guard: _SHADOWED_BUILTIN_NAMES must cover all real builtin/command names.

    This test fails when a new builtin tool/command isn't added to the shadow set
    -- update _SHADOWED_BUILTIN_NAMES in library_skills_state.py when it fires.

    TASK-13214 AC#3: the four sources are collected FIRST and asserted ONCE,
    so one source's gap can no longer mask another's -- the original
    three-assert version short-circuited in order, and the fleet-tools gap
    hid the video-command gap which hid the /research gap, serially, across
    three sightings.

    TASK-13214/F6: the builtin source is the GATE TABLE
    (`gateable_builtin_tools()`), not the live catalog -- `list_catalog()`
    omits gated-OFF tools, which is exactly how `expand_document` stayed
    invisible to this guard while every gate defaults OFF.
    """
    from tldw_chatbook.Agents.agent_models import RUNTIME_TOOL_NAMES
    from tldw_chatbook.Agents.tool_catalog import (
        BuiltinToolProvider,
        gateable_builtin_tools,
    )
    from tldw_chatbook.Chat.console_command_grammar import default_console_registry
    from tldw_chatbook.Library.library_skills_state import _SHADOWED_BUILTIN_NAMES

    sources = {
        "RUNTIME_TOOL_NAMES": set(RUNTIME_TOOL_NAMES),
        "BuiltinToolProvider catalog": {
            e.name for e in BuiltinToolProvider().list_catalog()
        },
        "gateable builtins (gate table, incl. gated-off)": {
            gate.tool_name for gate in gateable_builtin_tools()
        },
        "ConsoleCommandRegistry": set(default_console_registry().available_names()),
    }
    uncovered = {
        source: names - _SHADOWED_BUILTIN_NAMES
        for source, names in sources.items()
        if names - _SHADOWED_BUILTIN_NAMES
    }
    assert not uncovered, (
        "Shadow-guard drift -- ALL gaps across ALL sources (nothing masked): "
        f"{ {source: sorted(names) for source, names in uncovered.items()} }. "
        "Add them to _SHADOWED_BUILTIN_NAMES in "
        "tldw_chatbook/Library/library_skills_state.py -- do not accept this "
        "as a baseline failure (task-580)."
    )


def test_build_editor_state_marks_derived_description_and_keeps_field_empty():
    """task-419: when the SKILL.md frontmatter has NO description, the
    service derives one from the first body line for list display -- the
    editor must not echo that into the Description field as if the user
    had written it (a later save would ratchet it into the frontmatter)."""
    from tldw_chatbook.Library.library_skills_state import build_skill_editor_state

    state = build_skill_editor_state(
        {
            "name": "demo",
            "description": "First body line.",
            "content": "---\nname: demo\n---\nFirst body line.\nMore.",
        }
    )
    assert state.description == ""
    assert state.description_derived is True


def test_build_editor_state_keeps_real_frontmatter_description():
    from tldw_chatbook.Library.library_skills_state import build_skill_editor_state

    state = build_skill_editor_state(
        {
            "name": "demo",
            "description": "Real description.",
            "content": "---\nname: demo\ndescription: Real description.\n---\nBody.",
        }
    )
    assert state.description == "Real description."
    assert state.description_derived is False


def test_skill_trust_header_line_maps_postures():
    from tldw_chatbook.Library.library_skills_state import skill_trust_header_line

    assert skill_trust_header_line("needs_setup", 0)[1] == "setup"
    assert "isn't set up" in skill_trust_header_line("needs_setup", 0)[0]
    assert skill_trust_header_line("needs_resetup", 0)[1] == "resetup"
    assert "again after an update" in skill_trust_header_line("needs_resetup", 0)[0]
    assert skill_trust_header_line("unavailable", 0)[1] == "retry"
    assert skill_trust_header_line("locked", 0)[1] == "unlock"
    # ready + blocked skills -> review; ready + none -> quiet 'ready'
    assert skill_trust_header_line("ready", 3)[1] == "review"
    assert "3 skill" in skill_trust_header_line("ready", 3)[0]
    assert skill_trust_header_line("ready", 0)[1] == ""
    # error posture (corrupt/tampered manifest) -> still a header, with a
    # list-level recovery action (reuses "resetup" -- reset-then-bootstrap).
    assert skill_trust_header_line("error", 0)[1] == "resetup"
    assert "can't be verified" in skill_trust_header_line("error", 0)[0]
    # disabled/empty posture -> hidden
    assert skill_trust_header_line("", 0) is None


def test_console_command_names_are_treated_as_shadowing():
    """task-580 (AC#3): a skill named after a console command shadows it.

    `rewind` and `generate-image` were missing from the set, so a skill by
    either name was silently NOT recognised as shadowing a built-in, unlike
    every other command. Pinned by name rather than by re-deriving the
    registry, so this keeps failing if someone removes them.
    """
    from tldw_chatbook.Library.library_skills_state import skill_name_shadows_builtin

    for name in ("rewind", "generate-image"):
        assert skill_name_shadows_builtin(name) == name
        assert skill_name_shadows_builtin(f"  {name.upper()} ") == name


def test_new_runtime_and_console_names_are_treated_as_shadowing():
    """Pin the names uncovered by the resumed dev sweep independently of registries."""
    for name in (
        "discard_agent_worktree",
        "merge_agent_worktree",
        "prepare_managed_skill_promotion",
        "context",
        "doctor",
        "emergency-stop",
        "help",
        "model",
        "new",
        "redirect",
        "sessions",
        "settings",
        "steer",
        "temp",
        "workspace",
    ):
        assert skill_name_shadows_builtin(name) == name
        assert skill_name_shadows_builtin(f"  {name.upper()} ") == name
        assert skill_name_shadows_builtin(f"my-{name}-skill") is None
