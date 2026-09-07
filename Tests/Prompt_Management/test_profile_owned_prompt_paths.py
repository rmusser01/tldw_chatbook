"""Profile ownership regressions for prompt import defaults."""

from pathlib import Path

import pytest

import tldw_chatbook.Prompt_Management.Prompts_Interop as prompts_interop


@pytest.mark.parametrize("profile_name", ["alpha", "beta"])
def test_prompt_import_default_directory_follows_effective_profile(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    profile_name: str,
) -> None:
    """Prompt imports validate against the selected profile's prompt directory."""
    config_path = tmp_path / profile_name / "config.toml"
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))
    monkeypatch.setattr(prompts_interop, "is_initialized", lambda: True)
    captured_bases: list[Path] = []

    def capture_default_base(file_path: str, base_directory: Path) -> Path:
        captured_bases.append(base_directory)
        raise ValueError("stop after observing the default base directory")

    monkeypatch.setattr(prompts_interop, "validate_path", capture_default_base)

    prompts_interop.import_prompts_from_files(tmp_path / "import.json")

    assert captured_bases == [config_path.parent / "prompts"]


def test_prompt_import_preserves_explicit_base_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Prompt imports retain a caller-selected validation base directory."""
    selected_base_directory = str(tmp_path / "caller-selected")
    captured_bases: list[str] = []
    monkeypatch.setattr(prompts_interop, "is_initialized", lambda: True)

    def capture_explicit_base(file_path: str, base_directory: str) -> Path:
        captured_bases.append(base_directory)
        raise ValueError("stop after observing the explicit base directory")

    monkeypatch.setattr(prompts_interop, "validate_path", capture_explicit_base)

    prompts_interop.import_prompts_from_files(
        tmp_path / "import.json", base_directory=selected_base_directory
    )

    assert captured_bases == [selected_base_directory]
