"""Matched body-free edge timestamps and a total three-POST test allowance."""

import json
import os
import threading
import time
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

if os.environ.get("TEXTUAL_DRIVER") == "textual.drivers.web_driver:WebDriver":
    import requests

    _output = Path(os.environ["PR3023_HTTP_LOG"])
    assert _output.parent.parent == Path("/private/tmp/pr3023-live-current-r2")
    _lock = threading.Lock()
    _count = 3 if _output.exists() else 0
    _send = requests.Session.send

    def _record(value):
        value["monotonic_ns"] = time.monotonic_ns()
        value["wall_ns"] = time.time_ns()
        with _lock:
            with _output.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(value) + "\n")

    def _observed_send(session, request, **kwargs):
        global _count
        target = urlsplit(request.url)
        if target.hostname == "127.0.0.1":
            assert target.port != 9519, "Direct fixture access bypasses observation"
            return _send(session, request, **kwargs)
        if (
            request.method != "POST"
            or target.scheme != "https"
            or target.netloc != "api.deepseek.com"
            or target.path != "/chat/completions"
            or target.query
            or target.fragment
        ):
            raise RuntimeError("Network target outside the first-send fixture")
        request.url = urlunsplit(("http", "127.0.0.1:9519", target.path, "", ""))
        assert not kwargs.get("proxies"), "Fixture traffic must stay on loopback"
        kwargs["allow_redirects"] = False
        with _lock:
            if _count >= 3:
                raise RuntimeError("Disposable provider test budget exhausted")
            _count += 1
            number = _count
        _record(
            {
                "request": number,
                "event": "send",
                "path": target.path,
                "stream": bool(kwargs.get("stream")),
            }
        )
        try:
            response = _send(session, request, **kwargs)
        except BaseException as error:
            _record(
                {
                    "request": number,
                    "event": "send_exception",
                    "type": type(error).__name__,
                }
            )
            raise
        _record(
            {
                "request": number,
                "event": "response_open",
                "status": response.status_code,
            }
        )
        close = response.close

        def observed_close():
            try:
                return close()
            finally:
                _record({"request": number, "event": "response_closed"})

        response.close = observed_close
        return response

    requests.Session.send = _observed_send
