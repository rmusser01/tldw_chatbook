"""looks_attachable must not re-read the attachment config per path (TASK-32804.9).

_supported_patterns() (called once per candidate path by looks_attachable) read
attachment_filter_specs() every time -- ~5 ms/path, 98 ms for a 20-file
clipboard paste, on the event loop. It is now memoized; the supported formats do
not change within a session.
"""

import tldw_chatbook.Chat.console_paste_attach as cpa


def test_supported_patterns_reads_the_config_once(monkeypatch):
    calls = {"n": 0}

    def _specs():
        calls["n"] += 1
        return [("Images", "*.png;*.jpg")]

    monkeypatch.setattr(cpa, "attachment_filter_specs", _specs)
    cpa._supported_patterns.cache_clear()
    try:
        first = cpa._supported_patterns()
        again = cpa._supported_patterns()
        third = cpa._supported_patterns()
        assert first == again == third == ("*.png", "*.jpg")
        assert calls["n"] == 1, "attachment config was re-read; not memoized"
    finally:
        cpa._supported_patterns.cache_clear()  # don't leak the fake to other tests
