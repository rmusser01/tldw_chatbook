"""Focused spike checks: path isolation, relocation and archive containment."""

import importlib.util
import json
import shutil
import sys
import zipfile
from pathlib import Path

import pytest

SOURCE = Path(__file__).absolute().parents[2] / "Packaging" / "windows" / "portable"


def load(name):
    spec = importlib.util.spec_from_file_location(name, SOURCE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_portable_environment_drops_host_profile_and_secrets(tmp_path):
    launcher = load("portable")
    inherited = {
        "SystemRoot": r"C:\Windows",
        "HOME": "host",
        "TLDW_CONFIG_PATH": "host.toml",
        "OPENAI_API_KEY": "not-a-real-key",
        "PYTHONPATH": "checkout",
        "PATH": "host-bin",
        "CHATBOOK_WEZTERM_RUNTIME_DIR": "host-terminal",
    }
    root = tmp_path / "Chatbook space ü"
    env = launcher.portable_environment(root, inherited)
    launcher.prepare_data(root, env)
    assert Path(env["HOME"]).is_relative_to(root / "Data")
    assert Path(env["TLDW_CONFIG_PATH"]).is_relative_to(root / "Data")
    assert env["SYSTEMROOT"] == r"C:\Windows"
    assert env["PYTHON_KEYRING_BACKEND"] == "keyring.backends.null.Keyring"
    assert "OPENAI_API_KEY" not in env and "PYTHONPATH" not in env
    assert "host-bin" not in env["PATH"]
    assert inherited["HOME"] == "host"
    for name in (
        "CHATBOOK_WEZTERM_HOME",
        "CHATBOOK_WEZTERM_DATA_DIR",
        "CHATBOOK_WEZTERM_RUNTIME_DIR",
    ):
        assert Path(env[name]).is_relative_to(root / "Data")
        assert Path(env[name]).is_dir()


def test_relocation_derives_new_paths_and_launches_exact_bundled_runtime(tmp_path):
    launcher = load("portable")
    old, new = tmp_path / "first", tmp_path / "moved space ü"
    env = launcher.portable_environment(new, {"HOME": str(old / "Data" / "home")})
    assert str(old) not in str(env)
    command = launcher.launch_command(new, ["--help"])
    assert command[0] == str(new / "terminal" / "wezterm-gui.exe")
    assert str(new / "runtime" / "python.exe") in command
    assert "--always-new-process" in command and command[-1] == "--help"


def test_data_file_is_refused_instead_of_using_host_profile(tmp_path):
    launcher = load("portable")
    (tmp_path / "Data").write_text("occupied", encoding="utf-8")
    with pytest.raises(ValueError, match="not a directory"):
        launcher.prepare_data(tmp_path, launcher.portable_environment(tmp_path, {}))


@pytest.mark.parametrize(
    "member", ["../escape", "/absolute", "C:/escape", r"..\escape"]
)
def test_archive_rejects_escape_before_writing(tmp_path, member):
    builder = load("build_portable")
    archive = tmp_path / "input.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("first.txt", "must not be written")
        bundle.writestr(member, "escape")
    with pytest.raises(ValueError, match="Unsafe archive"):
        builder.extract_archive(archive, tmp_path / "unpacked")
    assert not (tmp_path / "unpacked").exists()


def test_archive_checksum_refuses_wrong_cached_bytes(tmp_path):
    builder = load("build_portable")
    (tmp_path / "python.zip").write_bytes(b"wrong")
    with pytest.raises(ValueError, match="checksum mismatch"):
        builder.verified_archive(tmp_path, "python", {"sha256": "0" * 64})


def test_distribution_never_contains_portable_user_state(tmp_path):
    builder = load("build_portable")
    payload = tmp_path / "payload"
    (payload / "Data").mkdir(parents=True)
    (payload / "Data" / "config.toml").write_text("private", encoding="utf-8")
    (payload / "portable.py").write_text("launcher", encoding="utf-8")
    output = tmp_path / "fresh.zip"
    builder.write_pristine_zip(payload, output)
    with zipfile.ZipFile(output) as archive:
        assert archive.namelist() == ["payload/portable.py"]


def test_moved_used_profile_is_refused_without_resetting_its_data(tmp_path):
    launcher = load("portable")
    original = tmp_path / "original"
    launcher.prepare_data(original, launcher.portable_environment(original, {}))
    launcher.check_location(original)
    (original / "Data" / "saved-note.txt").write_text("keep me", encoding="utf-8")
    copied = tmp_path / "copied space ü"
    shutil.copytree(original, copied)
    with pytest.raises(launcher.PortableDataMovedError, match="has moved"):
        launcher.check_location(copied)
    assert (copied / "Data" / "saved-note.txt").read_text(encoding="utf-8") == "keep me"
    launcher.check_location(original)


def test_moved_launch_routes_to_review_before_app_startup(tmp_path):
    launcher = load("portable")
    root = tmp_path / "moved"
    launcher.prepare_data(root, launcher.portable_environment(root, {}))
    (root / "Data" / "portable-location.json").write_text(
        json.dumps({"version": 1, "root": str(tmp_path / "original")})
    )
    assert launcher.app_arguments(root, ["--help"]) == ["--portable-relocate", "--help"]
    assert json.loads((root / "Data" / "portable-location.json").read_text())[
        "root"
    ] == str(tmp_path / "original")


def native_build_fixture(builder, root):
    binaries = root / "src" / "target" / "release"
    binaries.mkdir(parents=True)
    names = ("wezterm.exe", "wezterm-gui.exe", "wezterm-mux-server.exe")
    for name in names:
        (binaries / name).write_bytes(name.encode("ascii"))
    for name in ("LICENSE.md", "Cargo.lock", ".tag"):
        (root / "src" / name).write_text(name, encoding="utf-8")
    notices = root / "third-party-licenses"
    notices.mkdir()
    (notices / "notice.txt").write_bytes(b"license notice")
    (notices / "index.json").write_text(
        json.dumps(
            {
                "packages": [],
                "notices": [
                    {
                        "source": "upstream/LICENSE",
                        "file": "notice.txt",
                        "sha256": builder.sha256(notices / "notice.txt"),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    pins = json.loads((SOURCE / "wezterm" / "inputs.json").read_text(encoding="utf-8"))
    provenance = {
        "status": "experimental-modified-wezterm",
        "inputs": pins,
        "patch_sha256": builder.sha256(SOURCE / "wezterm" / "portable-roots.patch"),
        "executables": {name: builder.sha256(binaries / name) for name in names},
        "cargo_lock_sha256": builder.sha256(root / "src" / "Cargo.lock"),
        "third_party_license_index_sha256": builder.sha256(notices / "index.json"),
        "licenses": {
            name: builder.sha256(root / "src" / name)
            for name in ("LICENSE.md", "Cargo.lock", ".tag")
        },
    }
    (root / "provenance.json").write_text(json.dumps(provenance), encoding="utf-8")
    return binaries, provenance


def test_native_gui_cli_and_mux_must_match_build_provenance(tmp_path):
    builder = load("build_portable")
    binaries, provenance = native_build_fixture(builder, tmp_path)
    assert builder.verified_wezterm_build(tmp_path) == provenance
    (binaries / "wezterm.exe").write_bytes(b"stale stock executable")
    with pytest.raises(ValueError, match="executable checksum mismatch"):
        builder.verified_wezterm_build(tmp_path)


def test_native_license_notice_bytes_must_match_index(tmp_path):
    builder = load("build_portable")
    native_build_fixture(builder, tmp_path)
    (tmp_path / "third-party-licenses" / "notice.txt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="license notice checksum"):
        builder.verified_wezterm_build(tmp_path)


def test_native_patch_identity_must_match_tracked_source(tmp_path):
    builder = load("build_portable")
    _, provenance = native_build_fixture(builder, tmp_path)
    provenance["patch_sha256"] = "0" * 64
    (tmp_path / "provenance.json").write_text(json.dumps(provenance), encoding="utf-8")
    with pytest.raises(ValueError, match="portable patch provenance mismatch"):
        builder.verified_wezterm_build(tmp_path)


def test_terminal_uses_system_conpty_instead_of_path_limited_sideload(tmp_path):
    builder = load("build_portable")
    work = tmp_path / "build"
    native_build_fixture(builder, work)
    terminal = tmp_path / "terminal"
    terminal.mkdir()
    for name in ("conpty.dll", "OpenConsole.exe", "libGLESv2.dll"):
        (terminal / name).write_bytes(name.encode("ascii"))
    builder.install_wezterm_build(work, terminal)
    assert not (terminal / "conpty.dll").exists()
    assert not (terminal / "OpenConsole.exe").exists()
    assert (terminal / "libGLESv2.dll").read_bytes() == b"libGLESv2.dll"
    assert (terminal / "wezterm-gui.exe").read_bytes() == b"wezterm-gui.exe"
