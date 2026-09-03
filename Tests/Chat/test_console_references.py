"""TASK-26020: @-references for files, folders and diffs."""

from __future__ import annotations

from tldw_chatbook.Chat.console_references import (
    find_reference_candidates,
    expand_references,
    ReferenceExpansion,
)


# --- parsing: emails/decorators left untouched (AC#7) ---

def test_email_is_not_a_reference():
    cands = find_reference_candidates("mail me at bob@example.com please")
    assert cands == []


def test_at_preceded_by_word_char_is_not_a_reference():
    cands = find_reference_candidates("user@host and a@b.c")
    assert cands == []


def test_file_reference_is_found():
    cands = find_reference_candidates("see @src/main.py for details")
    assert len(cands) == 1
    assert cands[0].token == "src/main.py"


def test_file_reference_with_line_range_is_found():
    cands = find_reference_candidates("look at @src/main.py#L10-20 here")
    assert len(cands) == 1
    assert cands[0].token == "src/main.py#L10-20"


def test_special_tokens_found():
    cands = find_reference_candidates("review @diff and @staged now")
    tokens = {c.token for c in cands}
    assert "diff" in tokens and "staged" in tokens


# --- expansion with injected deps ---

def _resolver_for(mapping):
    # mapping: token-path -> ("file"|"folder", content-or-listing) or None if not allowed/exists
    def resolve(path_token):
        return mapping.get(path_token)
    return resolve


def test_decorator_left_untouched_when_it_does_not_resolve():
    """AC#7: @property (a decorator) resolves to no file -> literal text kept."""
    text = "@property\ndef x(self): ..."
    result = expand_references(
        text,
        resolve=lambda t: None,   # nothing resolves
        git_runner=lambda kind: "",
    )
    assert result.expanded_text == text  # untouched
    assert result.records == []


def test_file_reference_expands_with_content(  ):
    """AC#1."""
    def resolve(token):
        if token == "a.py":
            return ("file", "print('hi')\n", None)
        return None
    result = expand_references("show @a.py", resolve=resolve, git_runner=lambda k: "")
    assert "print('hi')" in result.expanded_text
    assert any(r.kind == "file" and r.ok for r in result.records)


def test_file_reference_with_line_range():
    """AC#1: optional line range."""
    lines = "".join(f"line{i}\n" for i in range(1, 11))
    def resolve(token):
        # resolver returns full content + parsed range
        if token.startswith("a.py"):
            return ("file", lines, (2, 4))
        return None
    result = expand_references("see @a.py#L2-4", resolve=resolve, git_runner=lambda k: "")
    assert "line2" in result.expanded_text and "line4" in result.expanded_text
    assert "line1" not in result.expanded_text and "line5" not in result.expanded_text


def test_folder_reference_expands_to_listing():
    """AC#2."""
    def resolve(token):
        if token == "src/":
            return ("folder", "main.py\nutil.py\n", None)
        return None
    result = expand_references("@src/", resolve=resolve, git_runner=lambda k: "")
    assert "main.py" in result.expanded_text and "util.py" in result.expanded_text


def test_diff_and_staged_expand_via_git():
    """AC#2."""
    def git_runner(kind):
        return f"DIFF({kind})"
    result = expand_references("@diff", resolve=lambda t: None, git_runner=git_runner)
    assert "DIFF(diff)" in result.expanded_text
    result2 = expand_references("@staged", resolve=lambda t: None, git_runner=git_runner)
    assert "DIFF(staged)" in result2.expanded_text


def test_outside_roots_reference_is_refused_not_read():
    """AC#3: cannot read what the tools cannot read -> resolver returns a refusal."""
    def resolve(token):
        if token == "/etc/passwd":
            return ("refused", "outside the allowed workspace roots", None)
        return None
    result = expand_references("@/etc/passwd", resolve=resolve, git_runner=lambda k: "")
    # refusal is shown, content is NOT injected
    assert "passwd" not in result.expanded_text or "refused" in str(result.records).lower()
    rec = [r for r in result.records if not r.ok]
    assert rec and "root" in rec[0].detail.lower()


def test_binary_and_oversized_refused():
    """AC#4."""
    def resolve(token):
        if token == "big.bin":
            return ("refused", "file is binary", None)
        if token == "huge.txt":
            return ("refused", "file exceeds the size limit", None)
        return None
    r1 = expand_references("@big.bin", resolve=resolve, git_runner=lambda k: "")
    assert any(not r.ok and "binary" in r.detail for r in r1.records)
    r2 = expand_references("@huge.txt", resolve=resolve, git_runner=lambda k: "")
    assert any(not r.ok and "size" in r.detail for r in r2.records)


# --- resolver against a real tmp workspace (real is_within, AC#3/#4) ---

from pathlib import Path
from tldw_chatbook.Chat.console_references import (
    resolve_reference, parse_token, MAX_REFERENCE_BYTES,
)


def test_parse_token_line_ranges():
    assert parse_token("a.py") == ("a.py", None)
    assert parse_token("a.py#L5") == ("a.py", (5, 5))
    assert parse_token("a.py#L10-20") == ("a.py", (10, 20))
    assert parse_token("a.py#Lbad") == ("a.py#Lbad", None)  # not a valid range


def test_resolver_reads_allowed_file(tmp_path):
    (tmp_path / "a.py").write_text("hello\nworld\n")
    kind, payload, rng = resolve_reference("a.py", roots=(tmp_path,))
    assert kind == "file" and "hello" in payload


def test_resolver_folder_listing(tmp_path):
    (tmp_path / "a.py").write_text("x")
    (tmp_path / "sub").mkdir()
    kind, payload, rng = resolve_reference(".", roots=(tmp_path,))
    assert kind == "folder"
    assert "a.py" in payload and "sub/" in payload


def test_resolver_refuses_outside_roots(tmp_path):
    outside = tmp_path.parent / "outside_root_file.txt"
    outside.write_text("secret")
    try:
        result = resolve_reference(str(outside), roots=(tmp_path,))
        assert result is not None and result[0] == "refused"
        assert "root" in result[1].lower() or "sensitive" in result[1].lower()
    finally:
        outside.unlink(missing_ok=True)


def test_resolver_refuses_binary(tmp_path):
    (tmp_path / "b.bin").write_bytes(b"\x00\x01\x02BINARY")
    kind, payload, rng = resolve_reference("b.bin", roots=(tmp_path,))
    assert kind == "refused" and "binary" in payload


def test_resolver_refuses_oversized(tmp_path):
    (tmp_path / "big.txt").write_text("A" * (MAX_REFERENCE_BYTES + 1))
    kind, payload, rng = resolve_reference("big.txt", roots=(tmp_path,))
    assert kind == "refused" and "size" in payload


def test_resolver_nonexistent_is_literal(tmp_path):
    """AC#7: a decorator-like token that resolves to nothing -> None (literal)."""
    assert resolve_reference("property", roots=(tmp_path,)) is None


# --- lane-8 review C1/M2: dotfile parity with ReadFileTool ---

def test_c1_dotfile_is_refused_not_read(tmp_path):
    """C1: @.env must be refused like ReadFileTool refuses hidden files, and
    its content must never be returned for injection."""
    (tmp_path / ".env").write_text("SECRET_KEY=sk-not-a-real-secret-value-000\n")
    result = resolve_reference(".env", roots=(tmp_path,))
    assert result is not None and result[0] == "refused"
    assert "SECRET_KEY" not in str(result)


def test_c1_git_config_is_refused(tmp_path):
    """C1: @.git/config (a hidden component) is refused, not read."""
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("[remote]\n url = https://u:tok@h/r\n")
    result = resolve_reference(".git/config", roots=(tmp_path,))
    assert result is not None and result[0] == "refused"
    assert "tok@h" not in str(result)


def test_m2_folder_listing_hides_dotfiles(tmp_path):
    """M2: a folder listing must not disclose hidden entry names."""
    (tmp_path / "visible.py").write_text("x")
    (tmp_path / ".env").write_text("secret")
    (tmp_path / ".ssh").mkdir()
    kind, payload, rng = resolve_reference(".", roots=(tmp_path,))
    assert kind == "folder"
    assert "visible.py" in payload
    assert ".env" not in payload and ".ssh" not in payload
