"""Recovered provider config reads retain native private-file checks."""

import pytest

from tldw_chatbook.LLM_Calls.recovery_review import ProviderReconnectRequired, _config


def test_recovered_provider_reads_private_config(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text('[api_settings.openai]\nmodel="synthetic"\n', encoding="utf-8")
    path.chmod(0o600)
    assert _config(path) == {"api_settings": {"openai": {"model": "synthetic"}}}


def test_recovered_provider_refuses_linked_config(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text("[general]\n", encoding="utf-8")
    path.chmod(0o600)
    (tmp_path / "alias.toml").hardlink_to(path)
    with pytest.raises(ProviderReconnectRequired):
        _config(path)


def test_provider_review_reads_through_selected_native_parent(tmp_path, monkeypatch):
    import importlib
    from contextlib import contextmanager
    from types import SimpleNamespace

    from tldw_chatbook.Backup_Recovery import native_files
    from tldw_chatbook.LLM_Calls import recovery_review
    from tldw_chatbook.Utils import platform_files

    path = tmp_path / "config.toml"
    path.write_text("[general]\n", encoding="utf-8")
    path.chmod(0o600)
    parent = object()
    original = platform_files.os
    namespace = recovery_review.__dict__.copy()

    @contextmanager
    def pinned(selected):
        assert selected == path.parent
        yield parent

    def native_open(name, flags, *, dir_fd):
        assert name == path.name and dir_fd is parent
        return original.open(path, flags)

    backend = SimpleNamespace(
        open=native_open,
        O_RDONLY=original.O_RDONLY,
        O_NOFOLLOW=original.O_NOFOLLOW,
        O_NONBLOCK=original.O_NONBLOCK,
        fstat=original.fstat,
        geteuid=original.geteuid,
        fdopen=original.fdopen,
        close=original.close,
    )
    try:
        monkeypatch.setattr(platform_files, "os", backend)
        monkeypatch.setattr(native_files, "pinned_directory", pinned)
        importlib.reload(recovery_review)
        assert recovery_review._config(path) == {"general": {}}
    finally:
        monkeypatch.undo()
        recovery_review.__dict__.clear()
        recovery_review.__dict__.update(namespace)
