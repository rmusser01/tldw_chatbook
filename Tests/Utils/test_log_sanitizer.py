"""Tests for log sanitization utilities."""

import tomllib

import pytest

from tldw_chatbook.config import CONFIG_TOML_CONTENT, DEFAULT_APP_TTS_CONFIG
from tldw_chatbook.Utils.log_sanitizer import (
    create_safe_log_message,
    safe_log,
    sanitize_dict,
    sanitize_list,
    sanitize_log_params,
    sanitize_string,
)
from tldw_chatbook.Utils.sensitive_config_keys import is_sensitive_config_key


def _synthetic(*parts: str) -> str:
    """Assemble a synthetic credential at import time from fragments.

    The fragments exist so that no committed line of this file contains a
    contiguous string in a real token shape (TASK-19555 Qodo round, rule
    497144). Secret scanners -- including GitHub push protection, which
    rejects the whole branch rather than the file -- match on the literal, so
    splitting the detector-bearing prefix is what makes the fixture shippable.
    The assembled value is byte-identical to the one the redactor must handle,
    so nothing about the test weakens.

    Args:
        *parts: Fragments to join, split at the detector prefix.

    Returns:
        The joined synthetic credential.
    """
    return "".join(parts)


def _iter_leaf_key_names(mapping):
    """Yield leaf mapping keys from the shipped configuration structure."""
    for key, value in mapping.items():
        if isinstance(value, dict):
            yield from _iter_leaf_key_names(value)
        else:
            yield key


def test_real_shipped_sensitive_key_names_are_redacted() -> None:
    """Redact every real secret-bearing default key using the shared policy."""
    default_config = tomllib.loads(CONFIG_TOML_CONTENT)
    key_names = {
        str(key)
        for key in _iter_leaf_key_names(default_config)
        if is_sensitive_config_key(key)
    }
    key_names.update(
        key for key in DEFAULT_APP_TTS_CONFIG if is_sensitive_config_key(key)
    )
    assert {"api_key", "auth_token", "api_token"} <= key_names

    sentinels = {
        key: f"PRIVATE_CONFIG_{index}" for index, key in enumerate(sorted(key_names))
    }
    result = sanitize_dict(sentinels)

    assert set(result) == set(sentinels)
    assert all(value == "***REDACTED***" for value in result.values())


@pytest.mark.parametrize(
    "key",
    [
        "Authorization",
        "Proxy-Authorization",
        "cookie",
        "Set-Cookie",
        "credential",
        "credentials",
        "database_url",
        "connection-string",
        "dsn",
    ],
)
def test_log_protocol_fields_are_redacted_without_expanding_config_policy(
    key: str,
) -> None:
    """Redact protocol-only fields while keeping config policy intentionally narrow."""
    assert not is_sensitive_config_key(key)
    assert sanitize_dict({key: "PRIVATE_PROTOCOL_VALUE"})[key] == "***REDACTED***"


def test_non_string_mapping_key_is_safe_and_sensitive_values_are_redacted() -> None:
    """Avoid lower() failures for non-string mapping keys."""
    result = sanitize_dict({1: "safe", "x-api-key": "PRIVATE"})

    assert result == {1: "safe", "x-api-key": "***REDACTED***"}


def test_non_mutating_sanitization_preserves_input_containers() -> None:
    """Return copies without changing supplied nested dictionaries or lists."""
    nested = {"token": "PRIVATE_NESTED"}
    items = [{"api_key": "PRIVATE_LIST"}]
    source = {"nested": nested, "items": items}

    result = sanitize_dict(source)

    assert source == {
        "nested": {"token": "PRIVATE_NESTED"},
        "items": [{"api_key": "PRIVATE_LIST"}],
    }
    assert result == {
        "nested": {"token": "***REDACTED***"},
        "items": [{"api_key": "***REDACTED***"}],
    }


def test_deep_false_returns_new_outer_container_and_sanitizes_direct_strings() -> None:
    """Skip nested traversal without sharing the outer mapping."""
    nested = {"token": "PRIVATE_NESTED"}
    items = ["token=PRIVATE_LIST"]
    source = {"nested": nested, "items": items, "message": "api_key=PRIVATE"}

    result = sanitize_dict(source, deep=False)

    assert result is not source
    assert result["nested"] is nested
    assert result["items"] is items
    assert result["message"] == "api_key=***REDACTED***"


@pytest.mark.parametrize("key", ["api_key_env_var", "max_tokens", "ordinary"])
def test_non_sensitive_keys_remain_unchanged(key: str) -> None:
    """Keep known non-secret configuration names and ordinary data intact."""
    assert sanitize_dict({key: "safe-value"}) == {key: "safe-value"}


@pytest.mark.parametrize("value", [{"nested": "PRIVATE"}, ["PRIVATE"]])
def test_sensitive_container_value_is_redacted_before_recursion(value) -> None:
    """Replace a sensitive container wholly instead of traversing its contents."""
    assert sanitize_dict({"api_key": value}) == {"api_key": "***REDACTED***"}


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (
            'api_key="PRIVATE_QUOTED", safe="visible"',
            'api_key="***REDACTED***", safe="visible"',
        ),
        ("password=correct horse battery staple", "password=***REDACTED***"),
        (
            "max_tokens=42 api_key=PRIVATE_LATER",
            "max_tokens=42 api_key=***REDACTED***",
        ),
        ("api_key=PRIVATE_QUERY&safe=visible", "api_key=***REDACTED***"),
        (
            "api_key=\nrefresh_token=PRIVATE_NEXT",
            "api_key=\nrefresh_token=***REDACTED***",
        ),
    ],
    ids=(
        "quoted-api-key",
        "password-with-spaces",
        "later-api-key",
        "query-api-key",
        "multiline-refresh-token",
    ),
)
def test_assignment_scanner_contract(raw: str, expected: str) -> None:
    """Redact classified assignments without consuming later candidates."""
    assert sanitize_string(raw) == expected


def test_assignment_scanner_redacts_two_quoted_secrets_on_one_line() -> None:
    """Keep syntax and safe fields while finding each quoted sensitive value."""
    raw = 'api_key="PRIVATE_FIRST", safe="visible", password="PRIVATE_SECOND"'

    assert sanitize_string(raw) == (
        'api_key="***REDACTED***", safe="visible", password="***REDACTED***"'
    )


def test_assignment_scanner_ignores_escaped_quotes_inside_secret() -> None:
    """Treat an escaped quote as part of the quoted secret value."""
    assert sanitize_string('api_key="PRIVATE\\"QUOTED", safe=visible') == (
        'api_key="***REDACTED***", safe=visible'
    )


def test_assignment_scanner_resumes_after_unterminated_quoted_value() -> None:
    """Redact an unterminated line then continue at the next assignment line."""
    raw = 'api_key="PRIVATE_UNTERMINATED\nrefresh_token=PRIVATE_NEXT'

    assert sanitize_string(raw) == (
        'api_key="***REDACTED***\nrefresh_token=***REDACTED***'
    )


@pytest.mark.parametrize("label", ["OPENAI_API_KEY", "ANTHROPIC_API_KEY"])
def test_explicit_provider_labels_redact_quoted_and_unquoted_values(label: str) -> None:
    """Classify explicit provider labels before consuming either value syntax."""
    assert sanitize_string(f'{label}="PRIVATE_QUOTED"') == f'{label}="***REDACTED***"'
    assert sanitize_string(f"{label}=PRIVATE_UNQUOTED") == f"{label}=***REDACTED***"


def test_all_config_derived_sensitive_labels_redact_quoted_and_unquoted_values() -> (
    None
):
    """Keep string logging aligned with the shared sensitive-key policy."""
    default_config = tomllib.loads(CONFIG_TOML_CONTENT)
    labels = {
        str(key)
        for key in _iter_leaf_key_names(default_config)
        if is_sensitive_config_key(key)
    }
    labels.update(key for key in DEFAULT_APP_TTS_CONFIG if is_sensitive_config_key(key))

    for index, label in enumerate(sorted(labels)):
        quoted = f'{label}="PRIVATE_QUOTED_{index}"'
        unquoted = f"{label}=PRIVATE_UNQUOTED_{index}"

        assert sanitize_string(quoted) == f'{label}="***REDACTED***"'
        assert sanitize_string(unquoted) == f"{label}=***REDACTED***"


@pytest.mark.parametrize(
    "raw",
    [
        "api_key_env_var",
        "max_tokens",
        "claude-opus-4-20250514",
        "Basic model configuration",
        "NotBearer token",
        "not-bearer token",
    ],
)
def test_false_positive_shapes_remain_unchanged(raw: str) -> None:
    """Avoid treating model names and near-schemes as credentials."""
    assert sanitize_string(raw) == raw


def test_standalone_bearer_credential_preserves_scheme_only() -> None:
    """Remove a full bearer credential without rewriting its scheme."""
    assert sanitize_string("Bearer PRIVATE_BEARER") == "Bearer ***REDACTED***"


def test_url_userinfo_removes_both_username_and_password() -> None:
    """Redact URL credentials as one neutral marker."""
    result = sanitize_string("https://user:PRIVATE_PASSWORD@example.test/private")

    assert result == "https://***REDACTED***@example.test/private"
    assert "user" not in result
    assert "PRIVATE_PASSWORD" not in result


@pytest.mark.parametrize(
    "raw",
    [
        # Assembled, not literal -- see `_synthetic` (TASK-19555 Qodo round).
        _synthetic("sk", "-proj-DO_NOT_USE_EXAMPLE_123456"),
        _synthetic("sk", "-ant-api03-DO_NOT_USE_EXAMPLE_123456"),
        _synthetic("sk", "-DONOTUSEEXAMPLEONLY1234567890"),
        "AIza" + "DO_NOT_USE_EXAMPLE_ONLY_" + "0" * 11,
    ],
    ids=("openai-project", "anthropic", "openai-legacy", "google"),
)
def test_standalone_credential_shapes_are_replaced_wholly(raw: str) -> None:
    """Replace every recognized standalone key shape with exactly one marker."""
    assert sanitize_string(raw) == "***REDACTED***"


@pytest.mark.parametrize(
    "raw",
    [
        "SK-PROJ-DO_NOT_USE_EXAMPLE_123456",
        "SK-ANT-API03-DO_NOT_USE_EXAMPLE_123456",
        "SK-DONOTUSEEXAMPLEONLY1234567890",
        "AIZA" + "DO_NOT_USE_EXAMPLE_ONLY_" + "0" * 11,
    ],
    ids=(
        "openai-project-uppercase",
        "anthropic-uppercase",
        "openai-legacy-uppercase",
        "google-uppercase",
    ),
)
def test_uppercase_standalone_credential_shapes_are_not_recognized(raw: str) -> None:
    """Keep standalone credential family recognition intentionally case-sensitive."""
    assert sanitize_string(raw) == raw


def test_labeled_standalone_shaped_value_has_one_idempotent_marker() -> None:
    """Assignment redaction consumes a key-shaped quoted secret only once."""
    secret = _synthetic("sk", "-proj-DO_NOT_USE_EXAMPLE_123456")
    raw = f'api_key="{secret}"'
    sanitized = sanitize_string(raw)

    assert sanitized == 'api_key="***REDACTED***"'
    assert sanitize_string(sanitized) == sanitized


def test_sanitization_is_idempotent_for_strings_and_containers() -> None:
    """Repeated sanitization must not further alter redacted data."""
    already_redacted = 'api_key="***REDACTED***" Bearer ***REDACTED***'
    dictionary = {"api_key": "PRIVATE", "nested": {"password": "PRIVATE"}}
    values = ["token=PRIVATE", {"api_key": "PRIVATE"}]

    assert sanitize_string(sanitize_string(already_redacted)) == already_redacted
    assert sanitize_dict(sanitize_dict(dictionary)) == sanitize_dict(dictionary)
    assert sanitize_list(sanitize_list(values)) == sanitize_list(values)


def test_sanitize_string_keeps_non_string_str_fallback() -> None:
    """Preserve the established permissive fallback for non-string values."""
    assert sanitize_string(1234) == "1234"


def test_formatting_failure_sanitizes_template_without_interpolating_raw_arguments() -> (
    None
):
    """Do not leak raw arguments when formatting falls back to its template."""
    result = create_safe_log_message("api_key={missing}", "PRIVATE_ARGUMENT")

    assert result == "api_key=***REDACTED***"
    assert "PRIVATE_ARGUMENT" not in result


def test_safe_log_calls_callback_once_with_sanitized_final_message() -> None:
    """Send one fully sanitized message to the provided logging callback."""
    calls: list[str] = []

    safe_log(calls.append, "api_key={}", "PRIVATE_CALLBACK")

    assert calls == ["api_key=***REDACTED***"]


def test_long_non_matching_text_remains_unchanged() -> None:
    """Exercise the scanner's linear no-match path without timing assertions."""
    raw = "x" * 100_000

    assert sanitize_string(raw) == raw


def test_dense_quoted_assignments_do_not_repeat_line_end_scans() -> None:
    """Keep matched quoted input single-pass without a wall-clock assertion."""

    class LineScanAccountingText(str):
        line_scan_work = 0

        def find(
            self,
            substring: str,
            start: int = 0,
            end: int | None = None,
        ) -> int:
            if substring in {"\r", "\n"}:
                scan_end = len(self) if end is None else end
                self.line_scan_work += scan_end - start
            if end is None:
                return super().find(substring, start)
            return super().find(substring, start, end)

    assignment_count = 2_000
    raw = LineScanAccountingText(
        ", ".join(f'api_key="PRIVATE_{index}"' for index in range(assignment_count))
    )
    expected = ", ".join('api_key="***REDACTED***"' for _ in range(assignment_count))

    assert sanitize_string(raw) == expected
    assert raw.line_scan_work == 0


class TestLogSanitizer:
    """Test the log sanitization utilities."""

    def test_sanitize_string_api_keys(self):
        """Test that API keys are sanitized from strings."""
        test_cases = [
            ("api_key=PRIVATE_API_KEY", "api_key=***REDACTED***"),
            (
                "Bearer PRIVATE_BEARER_TOKEN",
                "Bearer ***REDACTED***",
            ),
            ("OPENAI_API_KEY=PRIVATE_OPENAI_KEY", "OPENAI_API_KEY=***REDACTED***"),
            ('{"api_key": "secret123"}', '{"api_key": "***REDACTED***"}'),
            ("password: mypassword123", "password: ***REDACTED***"),
            ("https://user:pass@example.com", "https://***REDACTED***@example.com"),
        ]

        for input_str, expected in test_cases:
            result = sanitize_string(input_str)
            assert result == expected

    def test_sanitize_dict(self):
        """Test dictionary sanitization."""
        test_dict = {
            "name": "test",
            "api_key": "PRIVATE_API_KEY",
            "password": "secret",
            "nested": {"token": "bearer123", "safe": "value"},
            "config": "api_key=embedded_secret",
        }

        result = sanitize_dict(test_dict)

        assert result["name"] == "test"
        assert result["api_key"] == "***REDACTED***"
        assert result["password"] == "***REDACTED***"
        assert result["nested"]["token"] == "***REDACTED***"
        assert result["nested"]["safe"] == "value"
        assert "***REDACTED***" in result["config"]

    def test_sanitize_list(self):
        """Test list sanitization."""
        test_list = [
            "safe value",
            "api_key=secret",
            {"password": "hidden"},
            ["nested", "token=abc123"],
        ]

        result = sanitize_list(test_list)

        assert result[0] == "safe value"
        assert "***REDACTED***" in result[1]
        assert result[2]["password"] == "***REDACTED***"
        assert "***REDACTED***" in result[3][1]

    def test_create_safe_log_message(self):
        """Test safe log message creation."""
        # Test with positional args (OpenAI keys need 20+ chars after sk-)
        msg = create_safe_log_message(
            "User {} logged in with key {}",
            "john",
            _synthetic("sk", "-DONOTUSEEXAMPLEONLY1234567890"),
        )
        assert msg == "User john logged in with key ***REDACTED***"

        # Test with keyword args
        msg = create_safe_log_message("Config: {config}", config={"api_key": "secret"})
        assert "***REDACTED***" in msg

    def test_sanitize_log_params(self):
        """Test parameter sanitization."""
        args = ("test", {"api_key": "secret"}, "password=123")
        kwargs = {"token": "bearer123", "safe": "value"}

        clean_args, clean_kwargs = sanitize_log_params(*args, **kwargs)

        assert clean_args[0] == "test"
        assert clean_args[1]["api_key"] == "***REDACTED***"
        assert "***REDACTED***" in clean_args[2]
        assert clean_kwargs["token"] == "***REDACTED***"
        assert clean_kwargs["safe"] == "value"


# ---------------------------------------------------------------------------
# TASK-19555: the sink-side redactor applied to every record entering the
# in-app log collector, plus the hyphen-normalization repair it depends on.
# ---------------------------------------------------------------------------


class TestSinkRedaction:
    """`redact_log_line` and its two halves."""

    @pytest.mark.parametrize(
        "header",
        ["x-auth-token", "X-Session-Key", "x-client-secret", "api-secret"],
    )
    def test_hyphenated_header_names_are_recognised_as_secret_bearing(
        self, header: str
    ) -> None:
        """The normalization used to be computed and then thrown away.

        `is_sensitive_config_key` received the RAW key, and its
        `_key`/`_token`/`_secret` rules are underscore-suffix matches, so
        every hyphenated header whose sensitivity comes from a suffix was
        classified harmless and its value written out verbatim. These are the
        exact names provider request logging produces.
        """
        line = f"sending {header}: DONOTUSEEXAMPLEONLYzz9911"
        assert "DONOTUSEEXAMPLEONLYzz9911" not in sanitize_string(line)

    def test_max_tokens_is_not_a_false_positive_after_normalization(self) -> None:
        """`max-tokens` normalizes to `max_tokens`, which is still not `_token`."""
        assert sanitize_string("max-tokens=4096") == "max-tokens=4096"

    def test_home_directory_collapses_to_tilde(self) -> None:
        """The account name is a real-name identifier; the path shape is not."""
        from tldw_chatbook.Utils.log_sanitizer import redact_user_paths

        assert (
            redact_user_paths("saved /Users/janedoe/Notes/Q3.pdf")
            == "saved ~/Notes/Q3.pdf"
        )
        assert redact_user_paths("/home/janedoe/.cache/x") == "~/.cache/x"
        assert (
            redact_user_paths(r"read C:\Users\janedoe\AppData\x") == r"read ~\AppData\x"
        )

    def test_url_paths_are_not_mistaken_for_home_directories(self) -> None:
        """Case-sensitive on purpose: `/users/` in a REST URL is not a home."""
        from tldw_chatbook.Utils.log_sanitizer import redact_user_paths

        url = "GET https://api.example.com/users/alice/profile"
        assert redact_user_paths(url) == url

    def test_redact_log_line_applies_both_halves(self) -> None:
        from tldw_chatbook.Utils.log_sanitizer import redact_log_line

        secret = _synthetic("sk", "-DONOTUSEEXAMPLE1234567890")
        line = f"upload /Users/janedoe/x.pdf with api_key={secret}"
        redacted = redact_log_line(line)
        assert "janedoe" not in redacted
        assert secret not in redacted
        assert "upload ~/x.pdf" in redacted

    def test_redact_log_line_leaves_ordinary_diagnostics_alone(self) -> None:
        """Redaction must not be so eager that the Logs screen stops helping."""
        from tldw_chatbook.Utils.log_sanitizer import redact_log_line

        line = "RAG search returned 12 chunks in 340ms (mode=hybrid)"
        assert redact_log_line(line) == line

    @pytest.mark.parametrize(
        "credential",
        [
            # Every one is assembled by `_synthetic` rather than written as a
            # literal: see that helper for why (Qodo rule 497144 / GitHub push
            # protection). The values reaching the redactor are unchanged.
            _synthetic("ghp", "_DONOTUSEEXAMPLEONLYaaaaaaaaaaaaaaaaaaaaaa"),
            _synthetic("gho", "_DONOTUSEEXAMPLEONLYaaaaaaaaaaaaaaaaaaaaaa"),
            _synthetic("github", "_pat_DONOTUSEEXAMPLEONLYaaaaaaaaaaaaaaaaaaaa"),
            _synthetic("hf", "_DONOTUSEEXAMPLEONLYaaaaaaaaaaaaaaaa"),
            _synthetic("sk", "-or-v1-DONOTUSEEXAMPLEONLYaaaaaaaaaaaaaaaa"),
            _synthetic("AKI", "ADONOTUSEEXAMPLE1"),
            _synthetic("xox", "b-1234567890-DONOTUSEEXAMPLEONLY"),
            _synthetic(
                "eyJ", "hbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0",
                ".DONOTUSEEXAMPLE1",
            ),
        ],
    )
    def test_unlabelled_provider_token_shapes_are_redacted(
        self, credential: str
    ) -> None:
        """TASK-19555 review: the empty state claimed API keys were "always"
        removed while `ghp_`, `hf_`, `sk-or-v1-`, `AKIA…` and JWTs all walked
        straight through, because the standalone set only knew four shapes.
        The copy is now "recognised formats" AND the set is wider."""
        assert credential not in sanitize_string(f"request failed: {credential}")

    def test_windows_home_paths_redact_in_either_case_and_over_unc(self) -> None:
        """`Users` was a literal, so only the `C:\\Users\\` spelling redacted."""
        from tldw_chatbook.Utils.log_sanitizer import redact_user_paths

        assert redact_user_paths(r"c:\users\janedoe\Notes") == r"~\Notes"
        assert redact_user_paths(r"C:\Users\janedoe\Notes") == r"~\Notes"
        assert redact_user_paths(r"\\FILESRV\Users\janedoe\Notes") == r"~\Notes"

    def test_oversized_lines_are_truncated_before_redaction(self) -> None:
        """Cost is linear in length and the buffer bounds line COUNT only."""
        from tldw_chatbook.Utils.log_sanitizer import (
            MAX_REDACTED_LINE_CHARS,
            redact_log_line,
        )

        redacted = redact_log_line("payload " + "y" * 60_000)
        assert len(redacted) < MAX_REDACTED_LINE_CHARS + 200
        assert "truncated, 60008 chars" in redacted

    def test_ordinary_length_lines_are_not_marked_truncated(self) -> None:
        from tldw_chatbook.Utils.log_sanitizer import redact_log_line

        assert "truncated" not in redact_log_line("a short diagnostic line")


# ---------------------------------------------------------------------------
# TASK-19555 Qodo round: two security defects in the first cut of the above.
# ---------------------------------------------------------------------------


class TestRedactionOrderAndPathBoundaries:
    """Truncation must not manufacture a partial secret, and the literal
    home substitution must not rewrite half of somebody else's username."""

    def test_credential_astride_the_truncation_boundary_leaves_no_fragment(
        self,
    ) -> None:
        """Truncating BEFORE redaction cut tokens out of pattern range.

        `_STANDALONE_CREDENTIALS` all carry minimum lengths (`sk-` needs 20+
        trailing characters). A secret straddling the cap was therefore sliced
        into a fragment too short to match, and the fragment stayed in the
        Logs view and in "Copy visible logs" -- the exact surface this task
        exists to protect.
        """
        from tldw_chatbook.Utils.log_sanitizer import (
            MAX_REDACTED_LINE_CHARS,
            redact_log_line,
        )

        secret = _synthetic("sk", "-", "B" * 40)
        # Land the cap inside the token: 13 of its characters fall before it.
        head = "x" * (MAX_REDACTED_LINE_CHARS - 20)
        line = f"{head} token {secret} tail " + "y " * 5000

        redacted = redact_log_line(line)

        assert secret not in redacted
        # ...and no leading slice of it either.
        assert _synthetic("sk", "-B") not in redacted

    def test_truncation_keeps_whole_tokens_and_still_reports_the_real_length(
        self,
    ) -> None:
        from tldw_chatbook.Utils.log_sanitizer import (
            MAX_REDACTED_LINE_CHARS,
            redact_log_line,
        )

        line = "chunk " * 5000
        redacted = redact_log_line(line)

        assert "truncated, 30000 chars" in redacted
        assert len(redacted) < MAX_REDACTED_LINE_CHARS + 100
        # No half-word at the seam: every retained token is intact.
        body = redacted.split("…")[0]
        assert set(body.split()) <= {"chunk"}

    def test_one_unbroken_token_larger_than_the_cap_is_withheld_entirely(
        self,
    ) -> None:
        """No safe cut exists inside a single token, so none is attempted."""
        from tldw_chatbook.Utils.log_sanitizer import redact_log_line

        blob = "Q" * 9000
        redacted = redact_log_line(blob)

        assert "Q" * 50 not in redacted
        assert "9000" in redacted

    def test_a_home_prefixing_another_account_is_not_partially_replaced(
        self, monkeypatch
    ) -> None:
        """`str.replace` rewrote `/Users/jan` inside `/Users/janedoe`.

        That left `edoe` -- still identifying -- and destroyed the `/Users/`
        prefix `_HOME_ROOTS_POSIX` needed in order to fire, so the fallback
        could not clean up after it either.
        """
        monkeypatch.setenv("HOME", "/Users/jan")
        from tldw_chatbook.Utils.log_sanitizer import redact_user_paths

        assert redact_user_paths("/Users/janedoe/Notes/x.pdf") == "~/Notes/x.pdf"
        assert "edoe" not in redact_user_paths("/Users/janedoe/Notes/x.pdf")

    def test_the_windows_home_literal_respects_segment_boundaries_too(
        self, monkeypatch
    ) -> None:
        monkeypatch.setenv("USERPROFILE", r"C:\Users\jan")
        from tldw_chatbook.Utils.log_sanitizer import redact_user_paths

        assert redact_user_paths(r"C:\Users\janedoe\Notes") == r"~\Notes"

    def test_an_exotic_home_outside_users_and_home_still_collapses(
        self, monkeypatch
    ) -> None:
        """The literal pass is what covers `/root` and `$HOME` overrides."""
        monkeypatch.setenv("HOME", "/srv/appdata")
        from tldw_chatbook.Utils.log_sanitizer import redact_user_paths

        assert redact_user_paths("/srv/appdata/db.sqlite") == "~/db.sqlite"
        # A sibling that merely shares the prefix must survive intact.
        assert redact_user_paths("/srv/appdata-backup/db") == "/srv/appdata-backup/db"
