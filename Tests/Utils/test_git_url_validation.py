import pytest

from tldw_chatbook.Utils.input_validation import (
    ValidationError,
    validate_git_repo_url,
    validate_git_ref,
)


@pytest.mark.parametrize("url", [
    "https://github.com/owner/repo.git",
    "https://gitlab.example.com/a/b",
    "ssh://git@github.com/owner/repo.git",
])
def test_valid_repo_urls_pass(url):
    validate_git_repo_url(url)  # no raise


@pytest.mark.parametrize("url", [
    "ext::sh -c 'touch /tmp/pwn'",          # RCE transport
    "ext::git-upload-pack",
    "file:///etc/passwd",
    "file::/etc/passwd",
    "fd::17",
    "git://example.com/repo.git",           # unauthenticated transport, not allowlisted
    "http://example.com/repo.git",          # http not allowlisted (https only)
    "git@github.com:owner/repo.git",        # scp-shorthand rejected (ambiguous) — use ssh://
    "-upload-pack=/bin/sh",                 # leading dash (arg injection)
    "--upload-pack=x",
    "  https://x/y ",                        # whitespace
    "https://exa\\mple.com/y",              # backslash
    "/local/path/repo",                      # no scheme
    "repo",                                  # no scheme
    "",                                       # empty
])
def test_malicious_or_disallowed_repo_urls_raise(url):
    with pytest.raises(ValidationError):
        validate_git_repo_url(url)


@pytest.mark.parametrize("ref", ["main", "v1.2.3", "feature/new-thing", "release_1"])
def test_valid_refs_pass(ref):
    validate_git_ref(ref)  # no raise


@pytest.mark.parametrize("ref", [
    "--upload-pack=/bin/sh",   # leading dash
    "-b",
    "a b",                      # whitespace
    "a\tb",
    "..",                       # traversal-ish / invalid ref
    "a..b",
    "a\nb",                     # control char
    "",
])
def test_malicious_or_invalid_refs_raise(ref):
    with pytest.raises(ValidationError):
        validate_git_ref(ref)
