"""Tests for the `tldw-serve --pair` CLI flag (ADR-233 task 6).

Covers `_print_pair_qr()`: ASCII QR + URL when public_url is configured,
and a skip notice when it is not.
"""

import pytest

from tldw_chatbook.Web_Server.serve import _print_pair_qr


class _Stub:
    public_url = "https://chatbook.example"

    def issue_browser_bootstrap_url(self) -> str:
        return "https://chatbook.example/auth/bootstrap?nonce=abc123"


def test_pair_prints_qr_and_url(capsys) -> None:
    pytest.importorskip("qrcode")
    assert _print_pair_qr(_Stub()) is True
    out = capsys.readouterr().out
    assert "https://chatbook.example/auth/bootstrap?nonce=abc123" in out
    assert "█" in out or "▀" in out  # ascii qr block output


def test_pair_skips_without_public_url(capsys) -> None:
    class _NoUrl(_Stub):
        public_url = None

    assert _print_pair_qr(_NoUrl()) is False
    assert "public_url" in capsys.readouterr().out.lower()
