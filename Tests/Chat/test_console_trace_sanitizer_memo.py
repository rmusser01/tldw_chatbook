"""TASK-40608: the credential filter remembers results without keeping text.

Every provider request sanitizes the whole conversation and tool list again,
so the filter's cost grew with the conversation. Results are now remembered
by a keyed content digest. These cases pin that the remembered path returns
exactly what the filter computes, that an entry holds no input text and no
known credential, and that sanitizers with different inputs to the
computation never share an entry.
"""

from __future__ import annotations

import pytest

from tldw_chatbook.Chat import console_trace_redaction as redaction
from tldw_chatbook.Chat.console_trace_redaction import CredentialSanitizer

SECRET = "sk-secret98765432109876543210"
CORPUS = [
    "A plain paragraph of prose that is long enough to be remembered by the filter.",
    f"Authorization: Bearer {SECRET} was pasted into this long line by mistake, sorry.",
    "See https://user:hunter2@example.com/path?token=abc#frag for the deployment notes.",
    "-----BEGIN PRIVATE KEY-----\nMIIEvQIBADANBgkqhkiG9w0BAQEFAASC\n-----END PRIVATE KEY-----",
    "api_key=abcdef0123456789abcdef0123456789 appears in the middle of a longer sentence.",
    "héllo — 🚀 mixed scripts and a lone surrogate \ud800 in a string that is long enough.",
    "x" * 5000,
]


@pytest.fixture(autouse=True)
def empty_memo():
    redaction.forget_sanitized_text()
    yield
    redaction.forget_sanitized_text()


def _fresh(sanitizer: CredentialSanitizer, text: str, *, include_known: bool = True) -> str:
    return sanitizer._sanitize_text_now(text, include_known=include_known)


@pytest.mark.parametrize("text", CORPUS)
@pytest.mark.parametrize("include_known", [True, False])
def test_a_remembered_result_is_the_computed_result(text, include_known):
    sanitizer = CredentialSanitizer(known_credentials=(SECRET,))
    expected = _fresh(sanitizer, text, include_known=include_known)
    first = sanitizer._sanitize_text(text, include_known=include_known)
    second = sanitizer._sanitize_text(text, include_known=include_known)
    assert first == second == expected


def test_a_repeated_payload_is_filtered_once(monkeypatch):
    calls = []
    real = redaction._CREDENTIAL_PROJECTIONS[redaction.CREDENTIAL_FILTER_VERSION]

    def counted(text):
        calls.append(len(text))
        return real(text)

    monkeypatch.setitem(
        redaction._CREDENTIAL_PROJECTIONS, redaction.CREDENTIAL_FILTER_VERSION, counted
    )
    payload = {"messages": [{"role": "user", "content": text} for text in CORPUS]}
    sanitizer = CredentialSanitizer(known_credentials=(SECRET,))
    first = sanitizer.sanitize(payload)
    after_first = len(calls)
    second = CredentialSanitizer(known_credentials=(SECRET,)).sanitize(payload)
    assert first == second
    long_strings = sum(
        1 for text in CORPUS if len(text) >= redaction._SANITIZED_TEXT_MIN_LENGTH
    )
    assert after_first - (len(calls) - after_first) >= long_strings - 1, (
        "the second pass re-filtered strings the first pass had remembered"
    )


SHORT = ["type", "description", "", "fs_read", SECRET, "token=" + SECRET, "sk_" "live_0123456789abcdef"]


def test_an_entry_holds_no_input_text_and_no_known_credential():
    sanitizer = CredentialSanitizer(known_credentials=(SECRET,))
    for text in CORPUS + SHORT:
        sanitizer._sanitize_text(text)
    assert redaction._sanitized_text, "nothing was remembered"
    assert redaction._sanitized_short_text, "no short string was remembered"
    entries = [*redaction._sanitized_text.items(), *redaction._sanitized_short_text.items()]
    for key, (projection, value) in entries:
        assert type(key) is bytes and len(key) == 32
        assert projection is None or callable(projection)
        assert value is None or type(value) is str
        if value is not None:
            assert value not in CORPUS, "an unchanged input was kept as a value"
            assert SECRET not in value and "hunter2" not in value
        for text in CORPUS + [text for text in SHORT if text]:
            assert text.encode("utf-8", "surrogatepass") not in key
        assert SECRET.encode() not in key


def test_sanitizers_with_different_known_values_do_not_share_results():
    text = f"The deployment used the value {SECRET} in its environment file, in the clear."
    without = CredentialSanitizer(version="credentials-v1")
    with_known = CredentialSanitizer(known_credentials=("deployment used",), version="credentials-v1")
    first = without._sanitize_text(text)
    second = with_known._sanitize_text(text)
    assert first == _fresh(without, text)
    assert second == _fresh(with_known, text)
    assert first != second


@pytest.mark.parametrize("version", sorted(redaction.READABLE_CREDENTIAL_FILTER_VERSIONS))
def test_each_filter_version_keeps_its_own_results(version):
    text = "whsec_" "0123456789abcdef0123456789abcdef and sk_" "live_0123456789abcdef0123456789 here"
    sanitizer = CredentialSanitizer(version=version)
    assert sanitizer._sanitize_text(text) == _fresh(sanitizer, text)
    for other in sorted(redaction.READABLE_CREDENTIAL_FILTER_VERSIONS):
        peer = CredentialSanitizer(version=other)
        assert peer._sanitize_text(text) == _fresh(peer, text)


def test_a_replaced_projection_never_serves_another_projections_result(monkeypatch):
    """A result computed under a weaker filter is not reused under the real one.

    ``Tests/Chat/test_console_dispatch_continuation_handoff.py`` replays a
    turn traced before an upgrade by pointing the current version's table
    entry at the v1 filter, then restores it. Keyed by version string alone,
    the memo served v1's "unchanged" for a bare key after the restore, and
    the key reached a capture unmasked.
    """
    from tldw_chatbook.Utils.log_sanitizer import sanitize_trace_credentials_v1

    text = "env STRIPE=" + "sk" + "_live_" + "DONOTUSEEXAMPLEONLY" + "a1B2c3D4e5" * 2
    current = CredentialSanitizer()
    expected = _fresh(current, text)
    assert "sk_live_" not in expected
    with monkeypatch.context() as weaker:
        weaker.setitem(
            redaction._CREDENTIAL_PROJECTIONS,
            redaction.CREDENTIAL_FILTER_VERSION,
            sanitize_trace_credentials_v1,
        )
        assert "sk_live_" in CredentialSanitizer()._sanitize_text(text), (
            "the control: v1 leaves this bare key in place"
        )
    assert CredentialSanitizer()._sanitize_text(text) == expected
    assert current._sanitize_text(text) == expected
    # And the other way round: the real filter's result is not served to v1.
    with monkeypatch.context() as weaker:
        weaker.setitem(
            redaction._CREDENTIAL_PROJECTIONS,
            redaction.CREDENTIAL_FILTER_VERSION,
            sanitize_trace_credentials_v1,
        )
        assert "sk_live_" in CredentialSanitizer()._sanitize_text(text)


@pytest.mark.parametrize("include_known", [True, False])
def test_short_strings_are_remembered_in_their_own_table(include_known):
    sanitizer = CredentialSanitizer(known_credentials=(SECRET,))
    for chunk in SHORT:
        expected = _fresh(sanitizer, chunk, include_known=include_known)
        assert sanitizer._sanitize_text(chunk, include_known=include_known) == expected
        assert sanitizer._sanitize_text(chunk, include_known=include_known) == expected
    assert not redaction._sanitized_text
    assert len(redaction._sanitized_short_text) == len(SHORT)


def test_a_repeated_tool_list_filters_each_short_string_once(monkeypatch):
    """What the journey measured: the same schema words in every request."""
    calls = []
    real = redaction._CREDENTIAL_PROJECTIONS[redaction.CREDENTIAL_FILTER_VERSION]

    def counted(text):
        calls.append(text)
        return real(text)

    monkeypatch.setitem(
        redaction._CREDENTIAL_PROJECTIONS, redaction.CREDENTIAL_FILTER_VERSION, counted
    )
    schema = {
        "type": "function",
        "function": {
            "name": "fs_read",
            "description": "Read a file.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "description": "The file."}},
                "required": ["path"],
            },
        },
    }
    payload = {"model": "m", "tools": [dict(schema) for _ in range(20)]}
    first = CredentialSanitizer().sanitize(payload)
    distinct = len(calls)
    # A word that is both a key and a value ("function") is filtered once as
    # each: keys are filtered without the known values, so they do not share.
    assert max(calls.count(text) for text in set(calls)) <= 2
    assert distinct < 30, "the first pass filtered repeated schema words again"
    second = CredentialSanitizer().sanitize(payload)
    assert first == second and first.value == payload
    assert len(calls) == distinct, "the second request filtered remembered strings again"


def test_one_off_short_strings_never_push_out_a_long_one(monkeypatch):
    monkeypatch.setattr(redaction, "_SANITIZED_SHORT_TEXT_MAX", 8)
    sanitizer = CredentialSanitizer()
    long_text = CORPUS[0]
    sanitizer._sanitize_text(long_text)
    for index in range(40):
        sanitizer._sanitize_text(f"chunk {index}")
    assert len(redaction._sanitized_short_text) == 8
    assert len(redaction._sanitized_text) == 1
    calls = []
    monkeypatch.setattr(
        CredentialSanitizer,
        "_sanitize_text_now",
        lambda self, value, **options: calls.append(value) or value,
    )
    assert sanitizer._sanitize_text(long_text) == long_text
    assert not calls, "the long string was filtered again"


def test_a_short_string_computed_under_a_replaced_projection_is_not_reused(monkeypatch):
    """The projection rule of the long table holds for the short one."""
    from tldw_chatbook.Utils.log_sanitizer import sanitize_trace_credentials_v1

    text = "sk" + "_live_" + "DONOTUSEEXAMPLEONLY" + "a1B2c3D4e5"
    assert len(text) < redaction._SANITIZED_TEXT_MIN_LENGTH
    expected = _fresh(CredentialSanitizer(), text)
    assert "sk_live_" not in expected
    with monkeypatch.context() as weaker:
        weaker.setitem(
            redaction._CREDENTIAL_PROJECTIONS,
            redaction.CREDENTIAL_FILTER_VERSION,
            sanitize_trace_credentials_v1,
        )
        assert "sk_live_" in CredentialSanitizer()._sanitize_text(text), (
            "the control: v1 leaves this bare key in place"
        )
    assert CredentialSanitizer()._sanitize_text(text) == expected


def test_the_memo_is_bounded(monkeypatch):
    monkeypatch.setattr(redaction, "_SANITIZED_TEXT_MAX", 8)
    monkeypatch.setattr(redaction, "_SANITIZED_SHORT_TEXT_MAX", 8)
    sanitizer = CredentialSanitizer()
    for index in range(40):
        sanitizer._sanitize_text(f"{index:04d} " + "filler text that is long enough to remember " * 2)
        sanitizer._sanitize_text(f"short {index:04d}")
    assert len(redaction._sanitized_text) == 8
    assert len(redaction._sanitized_short_text) == 8


@pytest.mark.parametrize(
    "key", ["api_key", "Authorization", "content", "x-api-key", "https://example.com/api_key", "role"]
)
def test_credential_key_answers_match_the_uncached_rule(key):
    expected = "://" not in key and CredentialSanitizer._is_credential_key(key)
    assert CredentialSanitizer._credential_key(key) is expected
    assert CredentialSanitizer._credential_key(key) is expected
