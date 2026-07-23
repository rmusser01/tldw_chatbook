from tldw_chatbook.Chat.console_command_grammar import KIND_FALLBACK
from tldw_chatbook.Chat.console_skill_resolver import (
    SKILLS_EMPTY_LIST_ROW,
    SkillCommandCandidate,
    SkillMention,
    cap_skill_args,
    find_embedded_mentions,
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


def test_empty_word_never_matches_any_candidate():
    """Gemini M finding (PR #636 bot review): a bare `/` draft splits into
    an empty command word. Every string ``.startswith("")`` is True, so
    without a guard this fell through to the prefix-match branch and
    "matched" every candidate -- resolving to a single skill (one
    candidate) or reporting "ambiguous" (2+ candidates) for a draft that
    named no skill at all."""
    assert resolve_skill_command("", "", _cands("summarize")).kind == "none"


def test_whitespace_only_word_never_matches_any_candidate():
    """A `/ ` (slash + trailing space) draft also splits into an
    effectively-empty word; whitespace-only must be treated the same as
    truly empty."""
    assert resolve_skill_command("   ", "", _cands("summarize")).kind == "none"


def test_empty_word_is_none_even_with_multiple_candidates():
    assert resolve_skill_command("", "", _cands("summarize", "scan")).kind == "none"


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
    assert (
        claimed is not None and claimed.kind == KIND_FALLBACK and claimed.name == "summ"
    )
    assert resolver("unknownword", "x") is None


def test_fallback_does_not_claim_a_bare_slash_draft():
    """A `/` or `/ ` draft (console_command_grammar splits it into an
    empty word) must fall through to the unknown-command hint, never
    open a picker or silently resolve to whatever skill happens to exist."""
    resolver = make_skill_fallback_resolver(lambda: _cands("summarize"))
    assert resolver("", "") is None
    assert resolver("", " ") is None


_NAMES = frozenset({"code-review", "style-guide", "path"})


def test_embedded_mention_found_mid_prose():
    text = "please $style-guide this draft"
    mentions = find_embedded_mentions(text, _NAMES)
    assert mentions == (SkillMention(start=7, end=19, name="style-guide"),)
    assert text[7:19] == "$style-guide"


def test_embedded_mention_trailing_punctuation_stays_prose():
    mentions = find_embedded_mentions("run $style-guide.", _NAMES)
    assert mentions[0].name == "style-guide"
    assert mentions[0].end == 16  # the "." is not part of the token


def test_case_sensitive_exact_match_only():
    # $PATH stays literal even though a skill named "path" exists.
    assert find_embedded_mentions("echo $PATH", _NAMES) == ()
    assert find_embedded_mentions("echo $path", _NAMES)[0].name == "path"
    # prefix / unknown / numeric stay literal
    assert find_embedded_mentions("$style", _NAMES) == ()
    assert find_embedded_mentions("$5 and $100", _NAMES) == ()


def test_multiple_mentions_all_found_in_order():
    text = "$code-review then $style-guide"
    names = [m.name for m in find_embedded_mentions(text, _NAMES)]
    assert names == ["code-review", "style-guide"]


def test_code_spans_are_skipped():
    fenced = "look:\n```sh\necho $path\n```\nand $path here"
    mentions = find_embedded_mentions(fenced, _NAMES)
    assert len(mentions) == 1
    assert fenced[mentions[0].start :].startswith("$path here"[:5])
    inline = "use `$path` literally but $path expands"
    inline_mentions = find_embedded_mentions(inline, _NAMES)
    assert len(inline_mentions) == 1
    assert inline_mentions[0].start == inline.rindex("$path")


def test_skills_list_rows_use_dollar_sigil():
    listing = format_skills_list(
        (SkillCommandCandidate(name="code-review", description="d"),)
    )
    assert "$code-review" in listing
    assert "/code-review" not in listing


def test_odd_backtick_line_masks_entirely():
    """A line with an ODD backtick count is unparseable inline code — no
    pairing scheme is reliable (greedy pairing would let a stray tick
    consume a real opening tick and un-mask a user-guarded span). Fail
    SAFE: mask the whole line, mirroring the unclosed-fence philosophy."""
    assert find_embedded_mentions("a ` b then `$path` here", frozenset({"path"})) == ()
    # Well-formed control on the same skill: prose mentions still expand.
    found = find_embedded_mentions("plain $path here", frozenset({"path"}))
    assert [m.name for m in found] == ["path"]


def test_unclosed_fence_masks_to_eof():
    """An opening ``` fence with no closer masks everything after it — a
    $mention on a later line stays literal (pins existing behavior)."""
    text = "before $code-review\n```\necho $path on a later line"
    mentions = find_embedded_mentions(text, _NAMES)
    assert [m.name for m in mentions] == ["code-review"]
    assert not any(m.name == "path" for m in mentions)
