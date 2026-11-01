"""R104 fixed native selector and predecode TOML materialization boundary."""

import stat

import pytest

from Tests.Backup_Recovery.test_mobile_content_graph import content_reads
from Tests.Backup_Recovery.test_mobile_sync_plan import _version_default


@pytest.mark.parametrize(
    "raw",
    [
        b'[header."quoted.part".nested]\nfirst=1\nsecond={ child.key = [1,2] }\n',
        b'a.b.c = {nested = [1, 2, {"quoted.key" = "# [not syntax]"}]}\n',
        b"[\"quoted.table\".child]\nvalue = '''multi\n# literal . [ ]\nline'''\n",
        b'[[items]]\nname="first"\n[items.child]\nx=1\n[[items]]\nname="second"\n',
        b'basic = """multi\n# text \\"quote\\"\n"""\n# comment [fake] a.b.c\n',
    ],
)
def test_native_selector_toml_semantics_survive_predecode_accounting(raw):
    from tldw_chatbook.Backup_Recovery.inventory import _parse_profile_config
    from tldw_chatbook.Chat.console_trace_read_budget import _TraceReadBudget

    assert _parse_profile_config(
        raw, _read_budget=_TraceReadBudget()
    ) == _parse_profile_config(raw)


@pytest.mark.parametrize(
    "raw",
    [
        b"values=[" + b"0," * 120_000 + b"]",
        b".".join([b"prefix"] * 80_000) + b"=1",
    ],
    ids=["array-expansion", "dotted-prefix-expansion"],
)
def test_native_toml_expansion_refuses_before_decoder(raw, monkeypatch):
    from tldw_chatbook.Backup_Recovery import inventory
    from tldw_chatbook.Chat.console_trace_read_budget import (
        _TraceReadBudget,
        _TraceReadLimit,
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("TOML decoder reached before expansion admission")

    monkeypatch.setattr(inventory.tomllib, "loads", forbidden)
    with pytest.raises(_TraceReadLimit):
        inventory._parse_profile_config(raw, _read_budget=_TraceReadBudget())


@pytest.mark.parametrize("fault", ["ordinary", "short", "cancel", "expansion"])
def test_actual_history_selector_parses_once_with_native_ordinary_mode(
    tmp_path, monkeypatch, fault
):
    from tldw_chatbook.Agents import recovery as native_recovery
    from tldw_chatbook.Backup_Recovery import (
        inventory,
        mobile_runs_history,
        mobile_runs_selector,
    )
    from tldw_chatbook.Chat.console_trace_read_budget import _TraceReadLimit

    def native(db, purpose):
        _version_default(db, purpose)
        selector = tmp_path / "profile" / "config.toml"
        selector.chmod(0o644)
        if purpose == "canonical" and fault == "expansion":
            selector.write_text(
                selector.read_text()
                + "\n[preflight_expansion]\nvalues=["
                + "0," * 120_000
                + "]\n"
            )

    with content_reads(
        tmp_path, monkeypatch, lambda c, p: None, native_seed=native
    ) as reads:
        original = mobile_runs_history._qualify_history
        parse = inventory._parse_profile_config
        seen = []
        hits = []
        name_budgets = []
        native_names = native_recovery._run_log_names

        def names(config, *, _read_budget=None):
            assert _read_budget is not None
            name_budgets.append(_read_budget)
            assert len(name_budgets) == 1
            return native_names(config, _read_budget=_read_budget)

        original_read = mobile_runs_selector.os.read
        selector_identity = reads.captured.preview.source_selector.stat()

        def read(fd, amount):
            info = mobile_runs_selector.os.fstat(fd)
            raw = original_read(fd, amount)
            if amount == info.st_size + 1 and (info.st_dev, info.st_ino) == (
                selector_identity.st_dev,
                selector_identity.st_ino,
            ):
                hits.append(amount)
                if fault == "short":
                    return raw[:-1]
                if fault == "cancel":
                    reads.cancel.set()
            return raw

        def parsed(payload, *, _read_budget=None):
            if _read_budget is not None:
                seen.append(_read_budget)
            return parse(payload, _read_budget=_read_budget)

        def forbidden(*args, **kwargs):
            raise AssertionError("pure history must not reread config unbounded")

        def qualify(*args, **kwargs):
            with monkeypatch.context() as patch:
                patch.setattr(inventory, "read_profile_config", forbidden)
                patch.setattr(inventory, "_parse_profile_config", parsed)
                patch.setattr(native_recovery, "_run_log_names", names)
                if fault in {"short", "cancel"}:
                    patch.setattr(mobile_runs_selector.os, "read", read)
                if fault == "expansion":
                    patch.setattr(inventory.tomllib, "loads", forbidden)
                result = original(*args, **kwargs)
                assert seen == [kwargs["_read_budget"]]
                assert name_budgets == seen
                return result

        monkeypatch.setattr(mobile_runs_history, "_qualify_history", qualify)
        if fault == "ordinary":
            reads.runs_preflight()
        else:
            expected = {
                "short": ValueError,
                "cancel": InterruptedError,
                "expansion": _TraceReadLimit,
            }[fault]
            with pytest.raises(expected):
                reads.runs_preflight()
            if fault in {"short", "cancel"}:
                assert hits == [selector_identity.st_size + 1]
        assert (
            stat.S_IMODE(reads.captured.preview.source_selector.stat().st_mode) == 0o644
        )
    assert not reads.active and not reads.roots


@pytest.mark.parametrize(
    "configured,environment,expected",
    [
        ("logs", None, {"logs", ".logs"}),
        (".private", None, {".private"}),
        ("ignored", " custom ", {"custom", ".custom"}),
        ("../escape", None, {"agent-runs", ".agent-runs"}),
        ("custom", "..", {"agent-runs", ".agent-runs"}),
    ],
)
def test_installed_runlog_name_fallback_is_shared(
    configured, environment, expected, monkeypatch
):
    from tldw_chatbook.Agents.recovery import _run_log_names

    monkeypatch.delenv("TLDW_AGENTS_RUN_LOG_DIR_NAME", raising=False)
    if environment is not None:
        monkeypatch.setenv("TLDW_AGENTS_RUN_LOG_DIR_NAME", environment)
    assert _run_log_names({"agents": {"run_log_dir_name": configured}}) == expected


def test_repeated_native_header_prefix_is_charged_before_decoder(monkeypatch):
    from tldw_chatbook.Backup_Recovery import inventory
    from tldw_chatbook.Chat.console_trace_read_budget import (
        _TraceReadBudget,
        _TraceReadLimit,
    )

    raw = (
        "["
        + ".".join(["prefix"] * 1000)
        + "]\n"
        + "\n".join(f"key{n}=0" for n in range(1000))
    ).encode()

    def forbidden(*args, **kwargs):
        raise AssertionError("repeated native header/key tuples were not admitted")

    monkeypatch.setattr(inventory.tomllib, "loads", forbidden)
    with pytest.raises(_TraceReadLimit):
        inventory._parse_profile_config(raw, _read_budget=_TraceReadBudget())


def test_native_default_model_fields_are_charged_before_validation(monkeypatch):
    from tldw_chatbook.Backup_Recovery import inventory
    from tldw_chatbook.Chat.console_trace_read_budget import (
        _TraceReadBudget,
        _TraceReadLimit,
    )

    ordinary = inventory._parse_profile_config(b"")
    assert len(ordinary) == 7 and all(value == {} for value in ordinary.values())
    budget = _TraceReadBudget()
    budget.reserve(16 * 1024 * 1024 - 1024)  # Prior consumers of the same allowance.

    def forbidden(*args, **kwargs):
        raise AssertionError("native default dictionaries were not admitted")

    monkeypatch.setattr(inventory._ProfileSelectors, "model_validate", forbidden)
    with pytest.raises(_TraceReadLimit):
        inventory._parse_profile_config(b"", _read_budget=budget)


@pytest.mark.parametrize(
    "literal",
    [
        '"plain"',
        '"\\u0001"',
        "42",
        "-123456789",
        "1.7976931348623157e308",
        "true",
        '["a", "\\u0001", [1, 2]]',
        '{a=[1, "b"]}',
        "1979-05-27",
        "07:32:00.999999",
        "1979-05-27T07:32:00.999999-07:00",
    ],
)
def test_native_log_names_budget_preserves_toml_value_semantics(literal, monkeypatch):
    import tomllib

    from tldw_chatbook.Agents.recovery import _run_log_names
    from tldw_chatbook.Chat.console_trace_read_budget import _TraceReadBudget

    monkeypatch.delenv("TLDW_AGENTS_RUN_LOG_DIR_NAME", raising=False)
    config = tomllib.loads("[agents]\nrun_log_dir_name=" + literal)
    expected = _run_log_names(config)
    assert _run_log_names(config, _read_budget=_TraceReadBudget()) == expected


def test_native_log_names_environment_is_admitted_before_path_construction(monkeypatch):
    from tldw_chatbook.Agents import recovery
    from tldw_chatbook.Chat.console_trace_read_budget import (
        _TraceReadBudget,
        _TraceReadLimit,
    )

    monkeypatch.setenv("TLDW_AGENTS_RUN_LOG_DIR_NAME", "a" * (1024 * 1024))

    def forbidden(*args, **kwargs):
        raise AssertionError("native path construction preceded name admission")

    monkeypatch.setattr(recovery, "Path", forbidden)
    with pytest.raises(_TraceReadLimit):
        recovery._run_log_names({}, _read_budget=_TraceReadBudget())
