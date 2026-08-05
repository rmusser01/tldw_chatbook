"""UI-root regression coverage for the shared pytest data sandbox."""

from pathlib import Path

from tldw_chatbook import config


def test_ui_root_uses_shared_application_data_isolation(
    isolate_test_environment: Path,
) -> None:
    """Nested UI pytest runs must not resolve application data under HOME."""
    assert config.get_user_data_dir().is_relative_to(isolate_test_environment)
