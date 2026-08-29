"""task-18908: `python -m tldw_chatbook.app --serve --port N` must route
host/port/title/debug into run_web_server instead of ignoring them."""

from unittest.mock import patch

from tldw_chatbook.app import _build_arg_parser


def test_serve_flags_parse_on_shared_parser():
    # Both entry points share _build_arg_parser; serve flags must exist.
    args = _build_arg_parser().parse_args(
        ["--serve", "--port", "8765", "--host", "0.0.0.0"]
    )
    assert args.serve is True
    assert args.port == 8765
    assert args.host == "0.0.0.0"


def test_main_module_routes_serve_args(monkeypatch):
    """The __main__ serve branch must pass parsed args to run_web_server.

    Runs the module's __main__ block via runpy with the serve flags; the
    real run_web_server is patched so the test asserts the routing, not a
    real bind. runpy with run_name="__main__" executes the whole module
    (imports, early logging) the same way `python -m` does.
    """
    import runpy
    import sys

    captured = {}

    def _fake_run_web_server(host=None, port=None, title=None, debug=None):
        # Stands in for the whole serve loop; recording the routed kwargs
        # IS the assertion surface (a real bind is covered by the live
        # check in the task's AC).
        captured.update(host=host, port=port, title=title, debug=debug)

    def _fake_check():
        return True

    monkeypatch.setattr(
        "tldw_chatbook.Web_Server.serve.run_web_server", _fake_run_web_server
    )
    monkeypatch.setattr(
        "tldw_chatbook.Web_Server.serve.check_web_server_available", _fake_check
    )
    monkeypatch.setattr(sys, "argv", ["tldw-cli", "--serve", "--port", "8765"])

    # The module entry point may decide generated CSS is stale. Do not let
    # this routing test rewrite shared tracked stylesheets while xdist peers
    # are reading them; CSS rebuilding has its own dedicated tests.
    with patch("subprocess.run") as subprocess_run:
        subprocess_run.return_value.returncode = 0
        subprocess_run.return_value.stderr = ""
        try:
            runpy.run_module(
                "tldw_chatbook.app", run_name="__main__", alter_sys=True
            )
        except SystemExit as exc:
            # The serve branch exits 0 after the (patched) server returns.
            assert exc.code in (0, None)

    assert captured.get("port") == 8765, captured
