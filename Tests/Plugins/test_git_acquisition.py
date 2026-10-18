"""Controlled Git acquisition through the public source and snapshot entries."""

from pathlib import Path

import pytest

from Tests.private_profile import private_profile_test


@pytest.fixture(autouse=True)
def private_host_auth(monkeypatch, tmp_path):
    import os

    from tldw_chatbook.Plugins import acquisition

    monkeypatch.setattr(acquisition, "HOST_GIT_SYSTEM", tmp_path / "host-git-system")
    monkeypatch.setattr(acquisition, "HOST_SSH_SYSTEM", tmp_path / "host-ssh-system")
    for key in tuple(os.environ):
        if key.startswith(("GIT_", "SSH_")) or "proxy" in key.lower():
            monkeypatch.delenv(key)


def test_git_ext_transport_is_rejected_before_spawn():
    from tldw_chatbook.Plugins.git_source import parse_source_locator

    with pytest.raises(ValueError, match="plugin_source_transport_unsupported"):
        parse_source_locator("ext::sh -c whoami")
    assert (
        parse_source_locator("https://example.test/team/plugins.git").transport
        == "https"
    )


def test_local_git_pins_commit_without_repository_commands(
    plugin_stack, native_package, tmp_path
):
    from Tests.Plugins.acquisition_process_support import git, repository
    from tldw_chatbook.Plugins.acquisition import acquire_snapshot
    from tldw_chatbook.Plugins.git_source import select_local_source

    source = repository(native_package(), tmp_path / "git-source")
    (source / ".gitattributes").write_text("* filter=evil\n")
    git(source, "add", ".gitattributes")
    git(source, "commit", "-m", "checkout filter declaration")
    commit = git(source, "rev-parse", "HEAD").decode().strip()
    sentinel = tmp_path / "repository-command-ran"
    git(source, "config", "core.sshCommand", "touch " + str(sentinel))
    git(source, "config", "uploadpack.packObjectsHook", "touch " + str(sentinel))
    git(source, "config", "credential.helper", "!touch " + str(sentinel))
    git(source, "config", "filter.evil.smudge", "touch " + str(sentinel))
    owner = plugin_stack.owner
    destination = owner.acquisition_destination("git-local")
    result = plugin_stack.call(
        lambda: acquire_snapshot(
            select_local_source(source),
            "git-local",
            destination,
            owner=owner,
            authority=plugin_stack.authority,
        )
    )
    assert result.commit == commit
    assert result.tree == git(source, "rev-parse", "HEAD^{tree}").decode().strip()
    assert not (destination / ".git").exists()
    assert not sentinel.exists()
    before = (destination / "plugin.json").read_bytes()
    (source / "plugin.json").write_text("changed after review")
    git(source, "add", ".")
    git(source, "commit", "-m", "move branch")
    assert git(source, "rev-parse", "HEAD").decode().strip() != result.commit
    assert (destination / "plugin.json").read_bytes() == before


@pytest.mark.parametrize(
    "case",
    ["internal", "escape", "directory", "submodule", "lfs", "collision", "alternates"],
)
def test_git_tree_boundary_and_successful_control(
    plugin_stack, native_package, tmp_path, case
):
    from Tests.Plugins.acquisition_process_support import git, repository
    from Tests.Plugins.test_acquisition_limits import acquire

    source = repository(native_package(), tmp_path / "source")
    if case in {"internal", "escape", "directory"}:
        target = {
            "internal": "plugin.json",
            "escape": "../outside",
            "directory": "skills",
        }[case]
        (source / "linked").symlink_to(target)
        git(source, "add", "linked")
    elif case == "lfs":
        (source / "payload").write_text(
            "version https://git-lfs.github.com/spec/v1\noid sha256:"
            + "0" * 64
            + "\nsize 1\n"
        )
        git(source, "add", "payload")
    elif case == "submodule":
        oid = git(source, "rev-parse", "HEAD").decode().strip()
        git(source, "update-index", "--add", "--cacheinfo", "160000," + oid + ",nested")
    elif case == "collision":
        oid = (
            git(source, "hash-object", "-w", "--stdin", input=b"collision")
            .decode()
            .strip()
        )
        git(
            source,
            "update-index",
            "--add",
            "--cacheinfo",
            "100644," + oid + ",PLUGIN.json",
        )
    else:
        (source / ".git/objects/info/alternates").write_text(str(tmp_path / "outside"))
    if case != "alternates":
        git(source, "commit", "-m", "boundary")
    if case == "internal":
        result = acquire(plugin_stack, source)
        output = Path(result.materialized_identity)
        assert (output / "linked").is_file() and not (output / "linked").is_symlink()
        assert (output / "linked").read_bytes() == (output / "plugin.json").read_bytes()
        assert result.inspection.link_targets == {"linked": "plugin.json"}
        with pytest.raises(TypeError):
            result.inspection.link_targets["other"] = "plugin.json"
    else:
        with pytest.raises((ValueError, OSError)):
            acquire(plugin_stack, source)
    assert acquire(plugin_stack, native_package(), "positive").inspection.content_digest


def test_packed_refs_and_explicit_subdirectory(plugin_stack, native_package, tmp_path):
    import shutil

    from Tests.Plugins.acquisition_process_support import git
    from tldw_chatbook.Plugins.acquisition import acquire_snapshot
    from tldw_chatbook.Plugins.git_source import select_local_source

    source = tmp_path / "monorepo"
    source.mkdir()
    shutil.copytree(native_package(), source / "nested")
    git(source, "init", "-b", "main")
    git(source, "add", ".")
    git(source, "commit", "-m", "nested package")
    git(source, "tag", "release")
    git(source, "pack-refs", "--all")
    git(source, "repack", "-ad")
    assert tuple((source / ".git/objects/pack").glob("*.pack"))
    owner = plugin_stack.owner
    result = plugin_stack.call(
        lambda: acquire_snapshot(
            select_local_source(source, ref="refs/tags/release", subdirectory="nested"),
            "packed",
            owner.acquisition_destination("packed"),
            owner=owner,
            authority=plugin_stack.authority,
        )
    )
    assert (Path(result.materialized_identity) / "plugin.json").exists()
    assert result.tree == git(source, "rev-parse", "release:nested").decode().strip()


def test_missing_git_is_redacted_and_preserves_existing_package(
    plugin_stack, native_package, tmp_path, monkeypatch
):
    from Tests.Plugins.acquisition_process_support import repository
    from Tests.Plugins.test_acquisition_limits import acquire
    from tldw_chatbook.Plugins import acquisition

    source = repository(native_package(), tmp_path / "source")
    previous = acquire(plugin_stack, native_package(), "previous")
    original = acquisition.shutil.which
    monkeypatch.setattr(
        acquisition.shutil,
        "which",
        lambda name: None if name == "git" else original(name),
    )
    with pytest.raises(ValueError, match="plugin_git_missing"):
        acquire(plugin_stack, source)
    assert Path(previous.materialized_identity).exists()


@private_profile_test
@pytest.mark.loopback_network
@pytest.mark.parametrize("redirect", [False, True])
def test_https_host_helpers_tls_and_exact_origin(
    request, plugin_stack, native_package, tmp_path, redirect
):
    import json
    import os
    import shlex

    from Tests.hooks_v2_process_support import child_argv
    from Tests.Plugins.acquisition_process_support import git, https_git, repository
    from tldw_chatbook.Plugins.acquisition import acquire_snapshot
    from tldw_chatbook.Plugins.git_source import parse_source_locator

    source = repository(native_package(), tmp_path / "source")
    contexts = tmp_path / "helper-contexts"

    def helper(label, answer):
        code = (
            "import sys;from pathlib import Path;context=sys.stdin.read();"
            f"f=Path({str(contexts)!r});"
            f"f.write_text((f.read_text() if f.exists() else '')+{label!r}+'\\n'+context+'---\\n') if sys.argv[-1]=='get' else None;"
            f"print({answer!r}) if sys.argv[-1]=='get' else None"
        )
        return "!" + shlex.join(child_argv(code))

    helpers = [
        helper("USER", "username=fixture"),
        helper("PASSWORD", "password=private-credential-sentinel"),
    ]
    home = Path(os.environ["HOME"])
    include = home / "host-auth"
    with https_git(
        tmp_path, redirect="https://unrelated.invalid/retarget" if redirect else None
    ) as (origin, ca, requests):
        include.write_text(
            "[credential]\n helper = "
            + ("\n helper = ".join(json.dumps(item) for item in helpers))
            + "\n useHttpPath = true\n[http]\n sslCAInfo = "
            + json.dumps(str(ca))
            + "\n"
        )
        (home / ".gitconfig").write_text(
            "[include]\n path = " + json.dumps(str(include)) + "\n"
        )
        locator = parse_source_locator(origin + "/source/.git#ref=refs/heads/main")
        owner = plugin_stack.owner
        call = lambda: plugin_stack.call(
            lambda: acquire_snapshot(
                locator,
                "https",
                owner.acquisition_destination("https"),
                owner=owner,
                authority=plugin_stack.authority,
            )
        )
        if redirect:
            with pytest.raises(ValueError) as caught:
                call()
            assert "private-credential-sentinel" not in str(caught.value)
            assert len(requests) == 1
            assert not contexts.exists()
        else:
            result = call()
            assert result.commit == git(source, "rev-parse", "HEAD").decode().strip()
            assert result.origin == origin
            assert "private-credential-sentinel" not in result.model_dump_json()
            context = contexts.read_text()
            assert context.index("USER") < context.index("PASSWORD")
            assert "username=fixture" in context.split("PASSWORD", 1)[1]
            assert "protocol=https" in context and "path=source/.git" in context
            assert "host=" + origin.removeprefix("https://") in context
            assert any(value is not None for _, _, value in requests)


@pytest.mark.parametrize(
    "host_config",
    [
        '[url "https://other.invalid/"]\n insteadOf = https://selected.invalid/\n',
        "[http]\n proxy = http://other.invalid\n",
        "[http]\n extraHeader = Authorization: sentinel\n",
        "[core]\n sshCommand = touch sentinel\n",
    ],
)
def test_unsupported_host_retargeting_refuses_before_network(
    plugin_stack, native_package, host_config
):
    import os

    from tldw_chatbook.Plugins.acquisition import acquire_snapshot
    from tldw_chatbook.Plugins.git_source import parse_source_locator

    (Path(os.environ["HOME"]) / ".gitconfig").write_text(host_config)
    owner = plugin_stack.owner
    with pytest.raises(ValueError, match="host_auth_unsupported"):
        plugin_stack.call(
            lambda: acquire_snapshot(
                parse_source_locator("https://selected.invalid/repository"),
                "host-rejected",
                owner.acquisition_destination("host-rejected"),
                owner=owner,
                authority=plugin_stack.authority,
            )
        )


@private_profile_test
@pytest.mark.parametrize(
    "extra",
    [
        "",
        "Include other",
        "Match exec touch-sentinel",
        "HostName unrelated.invalid",
        "ProxyCommand touch-sentinel",
        "Port 2222",
        "GSSAPIAuthentication yes",
    ],
)
def test_ssh_host_projection_and_explicit_refusals(
    request, plugin_stack, native_package, tmp_path, monkeypatch, extra
):
    import json
    import os

    from Tests.Plugins.acquisition_process_support import fake_ssh, git, repository
    from tldw_chatbook.Plugins import acquisition
    from tldw_chatbook.Plugins.git_source import parse_source_locator

    source = repository(native_package(), tmp_path / "source")
    home = Path(os.environ["HOME"])
    config = home / ".ssh/config"
    config.parent.mkdir()
    config.write_text(
        "Host example.test\n IdentityFile "
        + str(tmp_path / "selected-key")
        + "\n CertificateFile "
        + str(tmp_path / "selected-certificate")
        + "\n IdentitiesOnly yes\n IdentityAgent none\n PreferredAuthentications publickey\n StrictHostKeyChecking yes\n "
        + extra
        + "\n"
    )
    with config.open("a") as stream:
        stream.write(
            " UserKnownHostsFile "
            + str(tmp_path / "known_hosts")
            + "\n GlobalKnownHostsFile "
            + str(tmp_path / "global_known_hosts")
            + "\n"
        )
    log = tmp_path / "ssh-log.json"
    executable = fake_ssh(tmp_path / "controlled-ssh", source, log)
    original = acquisition.shutil.which
    monkeypatch.setattr(
        acquisition.shutil,
        "which",
        lambda name: str(executable) if name == "ssh" else original(name),
    )
    owner = plugin_stack.owner
    call = lambda: plugin_stack.call(
        lambda: acquisition.acquire_snapshot(
            parse_source_locator("ssh://git@example.test/team/repo#ref=main"),
            "ssh",
            owner.acquisition_destination("ssh"),
            owner=owner,
            authority=plugin_stack.authority,
        )
    )
    if extra:
        with pytest.raises(ValueError, match="ssh_config_unsupported"):
            call()
        assert not log.exists()
    else:
        result = call()
        assert result.commit == git(source, "rev-parse", "HEAD").decode().strip()
        sent = json.loads(log.read_text())
        assert (
            "identityfile " + json.dumps(str(tmp_path / "selected-key"))
            in sent["config"]
        )
        assert (
            "certificatefile " + json.dumps(str(tmp_path / "selected-certificate"))
            in sent["config"]
        )
        assert "identitiesonly yes" in sent["config"]
        assert "preferredauthentications publickey" in sent["config"]
        assert "git@example.test" in sent["args"]


@pytest.mark.parametrize("kind", ["blob", "tree", "commit"])
def test_local_git_mismatched_object_content_cannot_claim_commit(
    plugin_stack, native_package, tmp_path, kind
):
    import zlib

    from Tests.Plugins.acquisition_process_support import git, repository
    from Tests.Plugins.test_acquisition_limits import acquire

    source = repository(native_package(), tmp_path / "source")
    (source / "data.txt").write_text("original")
    git(source, "add", ".")
    git(source, "commit", "-m", "data")
    assert acquire(plugin_stack, source, "valid").commit
    selected = {"blob": "HEAD:data.txt", "tree": "HEAD^{tree}", "commit": "HEAD"}[kind]
    oid = git(source, "rev-parse", selected).decode().strip()
    content = git(source, "cat-file", kind, oid)
    if kind == "blob":
        content = b"tampered"
    elif kind == "tree":
        content = content.replace(b"data.txt", b"dAta.txt")
    else:
        content = content.replace(b"\n\ndata\n", b"\n\nDATA\n")
    # Store the modified bytes under their real ID too: Git accepts their shape.
    actual_oid = (
        git(source, "hash-object", "-t", kind, "-w", "--stdin", input=content)
        .decode()
        .strip()
    )
    assert actual_oid != oid
    assert git(source, "cat-file", kind, actual_oid) == content
    object_path = source / ".git/objects" / oid[:2] / oid[2:]
    object_path.chmod(0o600)
    object_path.write_bytes(
        zlib.compress(f"{kind} {len(content)}\0".encode() + content)
    )
    with pytest.raises(ValueError, match="object_identity|acquisition_git_failed"):
        acquire(plugin_stack, source)


def test_git_links_specials_and_dynamic_configuration_cannot_escape(
    plugin_stack, native_package, tmp_path, monkeypatch
):
    import os

    from Tests.Plugins.acquisition_process_support import repository
    from Tests.Plugins.test_acquisition_limits import acquire

    source = repository(native_package(), tmp_path / "source")
    sentinel = tmp_path / "command-ran"
    monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
    monkeypatch.setenv("GIT_CONFIG_KEY_0", "core.sshCommand")
    monkeypatch.setenv("GIT_CONFIG_VALUE_0", "touch " + str(sentinel))
    assert acquire(plugin_stack, source, "control").commit
    assert not sentinel.exists()
    objects = source / ".git/objects"
    os.mkfifo(objects / "special")
    with pytest.raises(ValueError, match="local_git_unsupported"):
        acquire(plugin_stack, source, "special")
    (objects / "special").unlink()
    (objects / "external").symlink_to(tmp_path)
    with pytest.raises(ValueError, match="local_git_unsupported"):
        acquire(plugin_stack, source, "link")


def test_verified_ancestor_and_chained_internal_links(
    plugin_stack, native_package, tmp_path
):
    import zlib

    from Tests.Plugins.acquisition_process_support import git, repository
    from tldw_chatbook.Plugins.acquisition import acquire_snapshot
    from tldw_chatbook.Plugins.git_source import select_local_source

    source = tmp_path / "source"
    source.mkdir()
    import shutil

    package = source / "nested/plugin"
    shutil.copytree(native_package(), package)
    (package / "value").write_text("safe")
    (package / "first").symlink_to("value")
    (package / "second").symlink_to("first")
    repository(source, tmp_path / "repo")
    source = tmp_path / "repo"
    owner = plugin_stack.owner

    def acquire(operation):
        return plugin_stack.call(
            lambda: acquire_snapshot(
                select_local_source(source, subdirectory="nested/plugin"),
                operation,
                owner.acquisition_destination(operation),
                owner=owner,
                authority=plugin_stack.authority,
            )
        )

    result = acquire("good")
    assert (Path(result.materialized_identity) / "second").read_text() == "safe"
    assert result.inspection.link_targets["second"] == "value"
    oid = git(source, "rev-parse", "HEAD^{tree}").decode().strip()
    raw = git(source, "cat-file", "tree", oid).replace(b"nested", b"NESTED")
    valid_oid = (
        git(source, "hash-object", "-t", "tree", "-w", "--stdin", input=raw)
        .decode()
        .strip()
    )
    assert valid_oid != oid and git(source, "cat-file", "tree", valid_oid) == raw
    target = source / ".git/objects" / oid[:2] / oid[2:]
    target.chmod(0o600)
    target.write_bytes(zlib.compress(f"tree {len(raw)}\0".encode() + raw))
    with pytest.raises(ValueError, match="object_identity|acquisition_git_failed"):
        acquire("bad-parent")
