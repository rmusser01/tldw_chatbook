from tldw_chatbook.Chat.console_command_grammar import KIND_FALLBACK
from tldw_chatbook.Chat.console_skill_resolver import (
    SKILLS_EMPTY_LIST_ROW,
    SkillCommandCandidate,
    cap_skill_args,
    format_skills_list,
    make_skill_fallback_resolver,
    resolve_skill_command,
)


def _cands(*names):
    return tuple(SkillCommandCandidate(n, f"{n} desc") for n in names)


def test_exact_case_insensitive():
    r = resolve_skill_command("Code-Review", "x", _cands("code-review", "summarize"))
    assert (r.kind, r.name) == ("resolved", "code-review")


def test_unique_prefix():
    r = resolve_skill_command("summ", "", _cands("summarize", "code-review"))
    assert (r.kind, r.name) == ("resolved", "summarize")


def test_ambiguous_prefix():
    r = resolve_skill_command("s", "", _cands("summarize", "scan"))
    assert r.kind == "ambiguous" and set(r.matches) == {"summarize", "scan"}


def test_no_match():
    assert resolve_skill_command("zzz", "", _cands("summarize")).kind == "none"


def test_exact_wins_over_prefix():
    r = resolve_skill_command("scan", "", _cands("scan", "scanner"))
    assert (r.kind, r.name) == ("resolved", "scan")


def test_cap_args():
    assert len(cap_skill_args("x" * 10000)) == 4000


def test_format_list_empty():
    assert format_skills_list(()) == SKILLS_EMPTY_LIST_ROW


def test_format_list_lines_include_name_and_desc():
    text = format_skills_list(_cands("summarize"))
    assert "summarize" in text and "summarize desc" in text


def test_fallback_claims_matching_word_only():
    resolver = make_skill_fallback_resolver(lambda: _cands("summarize"))
    claimed = resolver("summ", "the doc")
    assert claimed is not None and claimed.kind == KIND_FALLBACK and claimed.name == "summ"
    assert resolver("unknownword", "x") is None
