"""PR compare-URL builder for change-review git modes (TASK-16801 arc B, T4).

`_parse_remote_url` is a pure function (no I/O) and is tested as such.
`pr_compare_url` is pure for github.com/gitlab.com/bitbucket.org, but the
codeberg.org (Gitea-family) branch needs a local
`refs/remotes/<remote>/HEAD` symref to resolve the default branch, so
those two cases drive a REAL tiny repo rather than mocking git (AC #5, no
mocked git) -- `test_pr_url_codeberg_remote_name_with_slash` in particular
pins the "never derive by splitting on '/'" rule (spec §2 probe 6 applied
to the codeberg lookup): the remote is literally named `a/b`.
"""
import subprocess
from pathlib import Path
from urllib.parse import quote

import pytest

from tldw_chatbook.Workspaces.git_workspace import (
    GitWorkspaceInfo,
    GitWorkspaceRefusal,
    _parse_remote_url,
    pr_compare_url,
)


def _git(cwd: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=True
    )
    return proc.stdout.strip()


def _info(**overrides) -> GitWorkspaceInfo:
    defaults = dict(
        root=Path("/repo"),
        repo_root=Path("/repo"),
        branch="feat/x",
        detached=False,
        unborn=False,
        upstream="origin/feat/x",
        upstream_remote="origin",
        remotes=(("origin", "https://github.com/o/r.git"),),
        ahead=0,
        behind=0,
    )
    defaults.update(overrides)
    return GitWorkspaceInfo(**defaults)


# ---------------------------------------------------------------------------
# _parse_remote_url -- pure.
# ---------------------------------------------------------------------------

_REMOTE_URL_SHAPES = [
    # github.com
    ("https://github.com/o/r.git", ("github.com", "o", "r")),
    ("ssh://git@github.com/o/r", ("github.com", "o", "r")),
    ("git@github.com:o/r.git", ("github.com", "o", "r")),
    # gitlab.com
    ("https://gitlab.com/o/r.git", ("gitlab.com", "o", "r")),
    ("ssh://git@gitlab.com/o/r", ("gitlab.com", "o", "r")),
    ("git@gitlab.com:o/r.git", ("gitlab.com", "o", "r")),
    # bitbucket.org
    ("https://bitbucket.org/o/r.git", ("bitbucket.org", "o", "r")),
    ("ssh://git@bitbucket.org/o/r", ("bitbucket.org", "o", "r")),
    ("git@bitbucket.org:o/r.git", ("bitbucket.org", "o", "r")),
    # codeberg.org
    ("https://codeberg.org/o/r.git", ("codeberg.org", "o", "r")),
    ("ssh://git@codeberg.org/o/r", ("codeberg.org", "o", "r")),
    ("git@codeberg.org:o/r.git", ("codeberg.org", "o", "r")),
]


@pytest.mark.parametrize("url,expected", _REMOTE_URL_SHAPES)
def test_parse_remote_url_shapes(url, expected):
    assert _parse_remote_url(url) == expected


def test_parse_remote_url_git_suffix_optional():
    assert _parse_remote_url("https://github.com/o/r") == ("github.com", "o", "r")
    assert _parse_remote_url("https://github.com/o/r.git") == ("github.com", "o", "r")


def test_parse_remote_url_gitlab_subgroup():
    assert _parse_remote_url("https://gitlab.com/g/sub/r.git") == (
        "gitlab.com",
        "g/sub",
        "r",
    )


def test_parse_remote_url_scp_like_subgroup():
    assert _parse_remote_url("git@gitlab.com:g/sub/r.git") == (
        "gitlab.com",
        "g/sub",
        "r",
    )


def test_parse_remote_url_unrecognized_shape_returns_none():
    assert _parse_remote_url("not a url") is None
    assert _parse_remote_url("") is None
    assert _parse_remote_url("https://github.com/repo-only.git") is None


# ---------------------------------------------------------------------------
# pr_compare_url -- github/gitlab/bitbucket are pure; codeberg needs a
# local symref.
# ---------------------------------------------------------------------------


def test_pr_url_no_upstream_refused(tmp_path):
    info = _info(upstream=None, upstream_remote=None)
    result = pr_compare_url(tmp_path, info)
    assert isinstance(result, GitWorkspaceRefusal)
    assert "push the branch first" in result.reason


def test_pr_url_github(tmp_path):
    info = _info(remotes=(("origin", "https://github.com/o/r.git"),))
    url = pr_compare_url(tmp_path, info)
    assert url == "https://github.com/o/r/compare/feat/x?expand=1"


def test_pr_url_github_ssh_remote(tmp_path):
    info = _info(remotes=(("origin", "ssh://git@github.com/o/r.git"),))
    url = pr_compare_url(tmp_path, info)
    assert url == "https://github.com/o/r/compare/feat/x?expand=1"


def test_pr_url_github_scp_like_remote(tmp_path):
    info = _info(remotes=(("origin", "git@github.com:o/r.git"),))
    url = pr_compare_url(tmp_path, info)
    assert url == "https://github.com/o/r/compare/feat/x?expand=1"


def test_pr_url_gitlab_branch_slash_encoded_in_query(tmp_path):
    info = _info(remotes=(("origin", "https://gitlab.com/o/r.git"),))
    url = pr_compare_url(tmp_path, info)
    assert url == (
        "https://gitlab.com/o/r/-/merge_requests/new"
        "?merge_request%5Bsource_branch%5D=feat%2Fx"
    )


def test_pr_url_gitlab_subgroup(tmp_path):
    info = _info(remotes=(("origin", "https://gitlab.com/g/sub/r.git"),))
    url = pr_compare_url(tmp_path, info)
    assert url == (
        "https://gitlab.com/g/sub/r/-/merge_requests/new"
        "?merge_request%5Bsource_branch%5D=feat%2Fx"
    )


def test_pr_url_bitbucket(tmp_path):
    info = _info(remotes=(("origin", "https://bitbucket.org/o/r.git"),))
    url = pr_compare_url(tmp_path, info)
    assert url == "https://bitbucket.org/o/r/pull-requests/new?source=feat%2Fx"


def test_pr_url_unicode_branch_percent_encoded(tmp_path):
    branch = "feat/日本語"
    info = _info(
        branch=branch,
        upstream=f"origin/{branch}",
        remotes=(("origin", "https://github.com/o/r.git"),),
    )
    url = pr_compare_url(tmp_path, info)
    assert url == f"https://github.com/o/r/compare/{quote(branch, safe='/')}?expand=1"

    info_gitlab = _info(
        branch=branch,
        upstream=f"origin/{branch}",
        remotes=(("origin", "https://gitlab.com/o/r.git"),),
    )
    url_gitlab = pr_compare_url(tmp_path, info_gitlab)
    assert url_gitlab == (
        "https://gitlab.com/o/r/-/merge_requests/new"
        f"?merge_request%5Bsource_branch%5D={quote(branch, safe='')}"
    )


def test_pr_url_unsupported_host_names_four_hosts(tmp_path):
    info = _info(remotes=(("origin", "https://example.com/o/r.git"),))
    result = pr_compare_url(tmp_path, info)
    assert isinstance(result, GitWorkspaceRefusal)
    for host in ("github.com", "gitlab.com", "bitbucket.org", "codeberg.org"):
        assert host in result.reason


def test_pr_url_unparseable_remote_names_four_hosts(tmp_path):
    info = _info(remotes=(("origin", "not a url"),))
    result = pr_compare_url(tmp_path, info)
    assert isinstance(result, GitWorkspaceRefusal)
    for host in ("github.com", "gitlab.com", "bitbucket.org", "codeberg.org"):
        assert host in result.reason


def _make_repo_with_default_branch_symref(root: Path, remote_name: str) -> None:
    root.mkdir()
    _git(root, "init", "-q", "-b", "main")
    _git(root, "config", "user.email", "t@t")
    _git(root, "config", "user.name", "t")
    (root / "a.txt").write_text("x\n")
    _git(root, "add", "-A")
    _git(root, "commit", "-qm", "base")
    sha = _git(root, "rev-parse", "HEAD")
    _git(root, "update-ref", f"refs/remotes/{remote_name}/main", sha)
    _git(
        root,
        "symbolic-ref",
        f"refs/remotes/{remote_name}/HEAD",
        f"refs/remotes/{remote_name}/main",
    )


def test_pr_url_codeberg_with_resolvable_default_branch(tmp_path):
    root = tmp_path / "repo"
    _make_repo_with_default_branch_symref(root, "origin")
    info = _info(
        root=root,
        repo_root=root,
        remotes=(("origin", "https://codeberg.org/o/r.git"),),
    )
    url = pr_compare_url(root, info)
    assert url == "https://codeberg.org/o/r/compare/main...feat/x"


def test_pr_url_codeberg_without_resolvable_default_branch_refused(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q", "-b", "main")
    _git(root, "config", "user.email", "t@t")
    _git(root, "config", "user.name", "t")
    info = _info(
        root=root,
        repo_root=root,
        remotes=(("origin", "https://codeberg.org/o/r.git"),),
    )
    result = pr_compare_url(root, info)
    assert isinstance(result, GitWorkspaceRefusal)
    assert "default branch" in result.reason


def test_pr_url_codeberg_remote_name_with_slash(tmp_path):
    # Regression pin: remote names CAN contain "/" (spec §2 probe 6); the
    # default-branch prefix must be stripped by the KNOWN remote name's
    # length, never by splitting the symref value on "/".
    root = tmp_path / "repo"
    _make_repo_with_default_branch_symref(root, "a/b")
    info = _info(
        root=root,
        repo_root=root,
        upstream="a/b/feat/x",
        upstream_remote="a/b",
        remotes=(("a/b", "https://codeberg.org/o/r.git"),),
    )
    url = pr_compare_url(root, info)
    assert url == "https://codeberg.org/o/r/compare/main...feat/x"
