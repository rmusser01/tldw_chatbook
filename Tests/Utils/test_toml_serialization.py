"""Literal Windows argv survives serialization without changing ordinary codecs."""

from __future__ import annotations

import datetime as dt
import tomllib

import pytest
import toml


@pytest.mark.parametrize(
    "value",
    [
        r"C:\hostedtoolcache\windows\Python\3.12.10\x64\python.exe",
        r"C:\x64\python.exe",
        r"\\server\share\x64\python.exe",
        r"--literal=\x41",
        "quoted ' and \" and literal \\x; Unicode ☃ 😀",
        r"C:\repeated\\x64\python.exe",
    ],
)
def test_literal_backslash_x_values_roundtrip_in_nested_argv(value):
    from tldw_chatbook.Utils.toml_serialization import dumps_cli_config

    original = {
        "hooks": {"hook": [{"id": "same", "command": [value, "-c", "pass"]}]},
        "quoted ordinary.key": {"value": value},
        "enabled": True,
    }
    assert tomllib.loads(dumps_cli_config(original)) == original


def test_ordinary_values_types_and_quoted_keys_keep_original_codec_bytes():
    from tldw_chatbook.Utils.toml_serialization import dumps_cli_config

    original = {
        "quoted key.with.dot": {"value": 'Unicode ☃; quote "; tab\t; newline\n'},
        "ordinary_windows": r"C:\Users\fixture\python.exe",
        "uppercase_X": r"C:\X64\python.exe",
        "numbers": [1, 2, 3],
        "float": 1.25,
        "boolean": False,
        "date": dt.date(2026, 10, 5),
        "time": dt.time(12, 34, 56),
        "when": dt.datetime(2026, 10, 5, 12, 34, 56, tzinfo=dt.timezone.utc),
    }
    assert dumps_cli_config(original) == toml.dumps(original)
    assert tomllib.loads(dumps_cli_config(original)) == tomllib.loads(
        toml.dumps(original)
    )


def test_one_affected_value_does_not_reencode_ordinary_control_value():
    from tldw_chatbook.Utils.toml_serialization import dumps_cli_config

    # Preserve the original codec outside the qualified branch. This task
    # does not extend the legacy writer's parseability promise to full fidelity.
    ordinary = "unchanged\x00\x1b ordinary"
    baseline = tomllib.loads(toml.dumps({"ordinary": ordinary}))["ordinary"]
    actual = tomllib.loads(
        dumps_cli_config({"affected": r"C:\x64\python.exe", "ordinary": ordinary})
    )
    assert actual["ordinary"] == baseline
    assert actual["affected"] == r"C:\x64\python.exe"


def test_cycle_retains_original_failure():
    from tldw_chatbook.Utils.toml_serialization import dumps_cli_config

    original = {"affected": r"C:\x64\python.exe"}
    original["cycle"] = original
    with pytest.raises(ValueError):
        dumps_cli_config(original)


def test_affected_value_keeps_other_original_codec_bytes():
    from tldw_chatbook.Utils.toml_serialization import dumps_cli_config

    ordinary = {
        "boolean": True,
        "integer": 42,
        "float": 1.5,
        "date": dt.date(2026, 10, 5),
        "time": dt.time(12, 34, 56),
        "when": dt.datetime(2026, 10, 5, tzinfo=dt.timezone.utc),
        "quoted key.with.dot": 'quote " and Unicode ☃',
        "numbers": [1, 2],
    }
    actual = dumps_cli_config({**ordinary, "affected": r"C:\x64\python.exe"})
    for line in toml.dumps(ordinary).splitlines():
        assert line in actual.splitlines()
    parsed = tomllib.loads(actual)
    assert {key: parsed[key] for key in ordinary} == tomllib.loads(toml.dumps(ordinary))


def test_ordinary_input_keeps_single_argument_serialization_fault(monkeypatch):
    from tldw_chatbook.Utils.toml_serialization import dumps_cli_config

    original = {"ordinary": r"C:\Users\fixture\python.exe"}
    error = ValueError("original serialization fault")

    def fail(value):
        assert value is original
        raise error

    monkeypatch.setattr(toml, "dumps", fail)
    with pytest.raises(ValueError) as raised:
        dumps_cli_config(original)
    assert raised.value is error


def test_affected_invalid_unicode_scalar_still_fails_parse_back():
    from tldw_chatbook.Utils.toml_serialization import dumps_cli_config

    serialized = dumps_cli_config({"affected": "literal \\x and \ud800"})
    with pytest.raises(tomllib.TOMLDecodeError):
        tomllib.loads(serialized)
