"""Focused contracts for the supported macOS builders."""

from __future__ import annotations

from types import SimpleNamespace

from Packaging.macos import build_app


def test_nuitka_build_includes_offline_tiktoken_assets(monkeypatch) -> None:
    builder = build_app.MacOSBuilder(build_mode="minimal", use_nuitka=True)
    captured: dict[str, object] = {}

    monkeypatch.setattr(builder, "create_app_icon", lambda: None)

    def fake_run(args, *, cwd):
        captured["args"] = args
        captured["cwd"] = cwd
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(build_app.subprocess, "run", fake_run)

    assert builder.build_with_nuitka() is True
    source = builder.project_root / "tldw_chatbook" / "assets" / "tiktoken_cache"
    assert (
        f"--include-data-dir={source}=tldw_chatbook/assets/tiktoken_cache"
        in captured["args"]
    )
