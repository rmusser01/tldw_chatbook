"""Static analyzer: sub-scores, dimension mapping, anti-pattern findings."""
from tldw_chatbook.Evals.skill_eval.models import SkillSubject
from tldw_chatbook.Evals.skill_eval.static_analyzer import analyze_static

TOOLS = frozenset({"fs_read", "fs_list"})
RESERVED = frozenset({"read_file"})


def _subject(**over) -> SkillSubject:
    base = dict(
        name="csv-cleaner",
        description="Use when tidying messy CSV exports before import. " + "x" * 40,
        body="# CSV cleaner\nRead inputs with fs_read. Apply references/rules.md.\n",
        allowed_tools=("fs_read",), script_paths=(),
        referenced_files=("references/rules.md",),
        bundle_paths=("references/rules.md",),
        source_kind="store", source_path="s", trust_status="trusted",
        digest="d" * 64, line_count=30,
    )
    base.update(over)
    return SkillSubject(**base)


def _codes(result):
    return {f.code for f in result.findings}


def test_clean_subject_scores_full_without_findings():
    res = analyze_static(_subject(), builtin_tool_names=TOOLS,
                         local_tool_names=frozenset(), reserved_names=RESERVED)
    assert res.findings == ()
    assert res.dimension_scores["triggering_accuracy"] == 1.0
    assert res.dimension_scores["tool_surface_sanity"] == 1.0
    assert res.dimension_scores["output_quality"] is None


def test_empty_description_and_missing_trigger_flagged():
    res = analyze_static(_subject(description="   "),
                         builtin_tool_names=TOOLS,
                         local_tool_names=frozenset(), reserved_names=RESERVED)
    assert {"EMPTY_DESCRIPTION", "MISSING_TRIGGER"} <= _codes(res)


def test_unknown_tools_includes_dead_namespaced_grants():
    res = analyze_static(_subject(allowed_tools=("fs_read", "local:fs_read",
                                                 "mcp__db__query")),
                         builtin_tool_names=TOOLS,
                         local_tool_names=frozenset(), reserved_names=RESERVED)
    assert "UNKNOWN_TOOLS" in _codes(res)
    assert res.dimension_scores["tool_surface_sanity"] < 1.0


def test_over_constrained_bloated_orphan_and_collision():
    body = "MUST do. ALWAYS check. NEVER skip.\n" * 6 + "see references/missing.md"
    res = analyze_static(
        _subject(body=body, line_count=900, referenced_files=("references/missing.md",),
                 bundle_paths=(), name="read_file"),
        builtin_tool_names=TOOLS, local_tool_names=frozenset(),
        reserved_names=RESERVED,
    )
    assert {"OVER_CONSTRAINED", "BLOATED_SKILL", "ORPHAN_REFERENCE",
            "NAME_COLLISION"} <= _codes(res)
