"""Tests for the `tldw-serve --pair` CLI flag (ADR-233 task 6).

Covers `_print_pair_qr()`: ASCII QR + URL when public_url is configured,
a skip notice when it is not, and a skip when the URL's host is a wildcard
or unspecified (e.g. the auto-derived TLS URL for a 0.0.0.0 bind).
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


def test_pair_skips_for_wildcard_public_url_host(capsys) -> None:
    # create_server auto-derives https://<bind-host>:<port> when TLS is set
    # without an explicit public_url; a wildcard bind must not be QR'd.
    class _WildcardHost(_Stub):
        public_url = "https://0.0.0.0:8000"

    assert _print_pair_qr(_WildcardHost()) is False
    out = capsys.readouterr().out.lower()
    assert "0.0.0.0" in out
    assert "public_url" in out
