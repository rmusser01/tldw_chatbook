from tldw_chatbook.Library.library_skills_state import (
    build_skill_editor_state, build_skills_list_state, classify_skill_save_error,
    compose_skill_markdown, save_marks_needs_review, skill_flags_line,
    skill_name_shadows_builtin,
)
from tldw_chatbook.Skills_Interop.skill_trust_models import SkillTrustBlockedError


def _ctx(available=(), blocked=()):
    return {"available_skills": list(available), "blocked_skills": list(blocked)}


def _summary(name, **over):
    base = {"name": name, "description": f"{name} desc", "argument_hint": None,
            "user_invocable": True, "disable_model_invocation": False,
            "trust_status": "trusted", "trust_blocked": False}
    base.update(over)
    return base


def test_list_renders_trusted_and_blocked_with_glyphs():
    state = build_skills_list_state(
        _ctx(available=[_summary("alpha")],
             blocked=[_summary("bravo", trust_status="quarantined_modified", trust_blocked=True)]),
        query="", sort="name")
    by_name = {r.name: r for r in state.rows}
    assert by_name["alpha"].trust_glyph == "✓" and by_name["alpha"].blocked is False
    assert by_name["bravo"].trust_glyph == "⚠" and by_name["bravo"].blocked is True
    assert state.count == 2


def test_status_sort_puts_needs_review_first():
    state = build_skills_list_state(
        _ctx(available=[_summary("zeta")],
             blocked=[_summary("aardvark", trust_blocked=True)]),
        query="", sort="status")
    assert [r.name for r in state.rows] == ["aardvark", "zeta"]


def test_query_matches_name_and_description():
    state = build_skills_list_state(
        _ctx(available=[_summary("code-review", description="Review pull requests"),
                        _summary("summarize", description="Shorten text")]),
        query="pull", sort="name")
    assert [r.name for r in state.rows] == ["code-review"]


def test_flags_line_variants():
    assert skill_flags_line(True, False) == "user · agent"
    assert skill_flags_line(True, True) == "user"
    assert skill_flags_line(False, False) == "agent"
    assert skill_flags_line(False, True) == "not invocable"


def test_shadow_predicate():
    assert skill_name_shadows_builtin("calculator") == "calculator"
    assert skill_name_shadows_builtin("skills") == "skills"
    assert skill_name_shadows_builtin("code-review") is None


def test_save_marks_needs_review_only_when_currently_trusted():
    assert save_marks_needs_review("trusted", False) is True
    assert save_marks_needs_review("quarantined_modified", True) is False


def test_editor_state_splits_frontmatter_and_body():
    detail = {"name": "code-review", "description": "Review code",
              "argument_hint": "[path]", "allowed_tools": ["calculator"],
              "user_invocable": True, "disable_model_invocation": False,
              "context": "inline", "model": None, "version": 3,
              "trust_status": "trusted", "trust_blocked": False,
              "supporting_files": {"notes.md": "hello"},
              "content": "---\nname: code-review\ndescription: Review code\n---\nReview {{args}} now."}
    state = build_skill_editor_state(detail)
    assert state.name == "code-review" and state.argument_hint == "[path]"
    assert state.allowed_tools_csv == "calculator"
    assert state.body.strip() == "Review {{args}} now."
    assert state.supporting_files == (("notes.md", 5),)
    assert state.version == 3


def test_compose_roundtrips_through_frontmatter_grammar():
    detail = {"name": "code-review", "description": "Review code", "argument_hint": None,
              "allowed_tools": None, "user_invocable": True, "disable_model_invocation": False,
              "context": "fork", "model": None, "version": 1, "trust_status": "trusted",
              "trust_blocked": False, "supporting_files": None,
              "content": "---\nname: code-review\ndescription: Review code\n---\nBody here."}
    state = build_skill_editor_state(detail)
    text = compose_skill_markdown(state, body="New body {{args}}")
    assert text.startswith("---\n") and "name: code-review" in text
    assert text.rstrip().endswith("New body {{args}}")


def test_classify_outcomes():
    assert classify_skill_save_error(None, "local_skill_exists:x", None) == "exists"
    assert classify_skill_save_error(None, "local_skill_version_conflict:x", None) == "version-conflict"
    assert classify_skill_save_error(None, "", SkillTrustBlockedError(
        skill_name="x", reason_code="skill_modified", trust_status="quarantined_modified")) == "trust-blocked"
    assert classify_skill_save_error({"name": "x"}, "", None) == "ok"


def test_pure_module_has_no_forbidden_imports():
    import tldw_chatbook.Library.library_skills_state as mod
    src = open(mod.__file__, encoding="utf-8").read()
    for forbidden in ("textual", "sqlite3", "tldw_chatbook.DB",
                      "tldw_chatbook.app", "httpx", "requests"):
        assert forbidden not in src
