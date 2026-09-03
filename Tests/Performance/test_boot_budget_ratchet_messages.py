"""The ratchet guards' failure paths, proven cheap (TASK-23029 / ADR-097).

The four boot-budget guards are subprocess probes (seconds to minutes per
run), so their message plumbing gets its discrimination proof here, at unit
speed: the policy footer says what the ratchet forbids, the diff formatters
name a planted mutant, the headroom line keeps its stable format, and the
checked-in snapshots stay real (an empty or truncated snapshot would turn
every "names the culprit" promise vacuous). The full-path mutants -- a
synthetic module planted on the boot path, pins edited under a live probe --
were run and captured in TASK-23029's implementation notes; these tests keep
the pieces from rotting between such runs.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
ADR_PATH = REPO_ROOT / "backlog" / "decisions" / "097-boot-budget-ratchets.md"

#: The four guard files whose budget assertions must carry the ratchet footer.
GUARD_FILES = (
    "test_app_import_weight.py",
    "test_ui_ready_module_census.py",
    "test_boot_css_byte_budget.py",
    "test_screen_preimport_payload_budget.py",
)


@pytest.mark.unit
def test_ratchet_policy_states_the_rule_and_the_three_responses(ratchet) -> None:
    """The footer must forbid raising and name the three legitimate moves."""
    text = ratchet.ratchet_policy("MAX_TLDW_MODULE_COUNT")
    assert "MAX_TLDW_MODULE_COUNT never rises" in text
    assert "(a) defer" in text
    assert "(b) shed" in text
    assert "(c) an explicit owner exception" in text
    assert "exception ledger" in text
    assert "ADR-097" in text
    assert "Raising the constant is NOT one of the options" in text
    assert "rejected in review" in text


@pytest.mark.unit
def test_adr_097_exists_with_an_exception_ledger() -> None:
    """The failure messages point at ADR-097; the reference must not dangle."""
    assert ADR_PATH.is_file(), f"missing: {ADR_PATH}"
    text = ADR_PATH.read_text(encoding="utf-8")
    assert "Exception ledger" in text
    assert "never rise" in text
    assert "tightening convention" in text.lower()


@pytest.mark.unit
def test_headroom_line_format_is_stable(ratchet) -> None:
    """CI-log consumers grep this exact shape; pin it."""
    assert (
        ratchet.headroom_line("boot-import-weight", [("modules", 650, 660)])
        == "boot-import-weight: 650/660 modules (headroom 10)"
    )
    assert ratchet.headroom_line(
        "preimport-payload", [("modules", 488, 500), ("LOC", 374697, 380000)]
    ) == (
        "preimport-payload: 488/500 modules (headroom 12); "
        "374697/380000 LOC (headroom 5303)"
    )


@pytest.mark.unit
def test_emit_headroom_prints_and_warns(ratchet, capsys) -> None:
    """The one-liner must reach BOTH channels: stdout (-s) and the warnings
    summary (the only per-test channel a default CI invocation shows for a
    passing test)."""
    line = "boot-css-bytes: 1/2 bytes (headroom 1)"
    with pytest.warns(UserWarning, match="headroom 1"):
        returned = ratchet.emit_headroom(line)
    assert returned == line
    assert line in capsys.readouterr().out


@pytest.mark.unit
def test_module_diff_names_a_planted_mutant_against_the_real_snapshot(
    ratchet,
) -> None:
    """Mutant proof for the breach-naming path, against the checked-in pin.

    A synthetic module added to the measured set must surface as ``+ name``
    under the NEW-modules heading, and a module withheld from the measured
    set must surface as ``- name`` -- the TASK-23028 directional house
    pattern, driven by the real snapshot artifact so a broken/empty snapshot
    fails here too.
    """
    pinned = ratchet.load_module_snapshot("boot-import-weight")
    assert len(pinned) >= 600, (
        f"boot-import-weight snapshot looks hollow ({len(pinned)} names) -- "
        "the breach diff would name nothing."
    )
    victim = sorted(pinned)[0]
    live = (set(pinned) | {"tldw_chatbook.zz_ratchet_mutant_probe"}) - {victim}
    report = ratchet.format_module_diff(live, "boot-import-weight")
    assert "+ tldw_chatbook.zz_ratchet_mutant_probe" in report
    assert f"- {victim}" in report
    assert "NEW modules (1)" in report
    assert "boot_import_modules.txt" in report


@pytest.mark.unit
def test_byte_diff_names_grown_new_and_removed_keys(ratchet) -> None:
    """Per-key delta formatting: signed sizes, largest first, +/- for churn."""
    pinned = {"a.tcss::X": 100, "a.tcss::Y": 50, "a.tcss::GONE": 10}
    live = {"a.tcss::X": 400, "a.tcss::Y": 45, "a.tcss::NEWCOMER": 77}
    report = ratchet.format_byte_diff(live, pinned, "segment")
    assert "a.tcss::X: 100 -> 400 (+300)" in report
    assert "a.tcss::Y: 50 -> 45 (-5)" in report
    assert "+ a.tcss::NEWCOMER: 77" in report
    assert "- a.tcss::GONE: was 10" in report
    # Largest |delta| first.
    assert report.index("a.tcss::X") < report.index("a.tcss::Y")


@pytest.mark.unit
def test_name_delta_is_directional(ratchet) -> None:
    """The unpinned variant keeps the same +/- discipline."""
    report = ratchet.format_name_delta({"m.a", "m.b"}, {"m.b", "m.c"}, "module")
    assert "+ m.a" in report
    assert "- m.c" in report
    identical = ratchet.format_name_delta({"m.a"}, {"m.a"}, "module")
    assert "identical" in identical


@pytest.mark.unit
def test_snapshots_are_real_not_hollow(ratchet) -> None:
    """Anti-vacuity for every checked-in snapshot.

    A truncated or accidentally emptied snapshot would keep the guards green
    while gutting their culprit-naming; pin minimum populations and internal
    consistency instead.
    """
    boot = ratchet.load_module_snapshot("boot-import-weight")
    assert len(boot) >= 600
    assert all(name.startswith("tldw_chatbook") for name in boot)

    ready = ratchet.load_module_snapshot("ui-ready-census")
    assert len(ready) >= 900
    assert all(name.startswith("tldw_chatbook") for name in ready)
    # The mount leg strictly extends the import leg's residency.
    assert len(ready) > len(boot)

    css = ratchet.load_json_snapshot("boot-css-bytes")
    assert len(css["per_source"]) == 6, "expected the six boot-parsed sources"
    assert len(css["per_segment"]) >= 150
    assert css["total"] == sum(css["per_source"].values())
    assert css["total"] == sum(css["per_segment"].values())

    pre = ratchet.load_json_snapshot("preimport-payload")
    routes = pre["routes"]
    assert "library" in routes and "settings" in routes
    union: set[str] = set()
    for row in routes.values():
        union |= set(row["modules"])
    assert len(union) == pre["pass_added_modules"]
    assert sum(row["loc"] for row in routes.values()) == pre["pass_added_loc"]


@pytest.mark.unit
def test_every_guard_wires_the_ratchet_footer_and_headroom_line() -> None:
    """Deleting the policy footer or the headroom emit from any guard is a
    silent policy rollback; keep both wired in all four files."""
    here = Path(__file__).parent
    for filename in GUARD_FILES:
        source = (here / filename).read_text(encoding="utf-8")
        assert "ratchet_policy(" in source, f"{filename} lost the policy footer"
        assert "emit_headroom(" in source, f"{filename} lost the headroom line"
        assert "SNAPSHOT_REFRESH" in source, (
            f"{filename} lost the snapshot-refresh hint"
        )


@pytest.mark.unit
def test_guards_never_write_their_snapshots() -> None:
    """Snapshots must be impossible to regenerate accidentally: only
    ``scripts/update_boot_budget_snapshots.py`` writes them.

    Two pins: the shared helper (the only code that knows the snapshot
    paths) is entirely write-free, and no guard file reaches for
    ``SNAPSHOT_DIR`` itself -- guards go through the helper's ``load_*`` /
    ``format_*`` API only. (Guards DO write probe fixtures like a scratch
    ``config.toml``; that is fine and not what this pins.)
    """
    here = Path(__file__).parent
    helper = (here / "boot_budget_ratchet.py").read_text(encoding="utf-8")
    for needle in ("write_text", "write_bytes", "json.dump(", "open("):
        assert needle not in helper, (
            f"boot_budget_ratchet.py contains {needle} -- the helper must "
            "stay write-free (ADR-097)."
        )
    for filename in GUARD_FILES:
        source = (here / filename).read_text(encoding="utf-8")
        assert "SNAPSHOT_DIR" not in source, (
            f"{filename} touches SNAPSHOT_DIR directly -- guards must only "
            "read snapshots through the helper (ADR-097)."
        )


@pytest.mark.unit
def test_update_script_exists_and_refuses_over_budget_pins() -> None:
    """The documented one-liner must exist and keep its refusal semantics."""
    script = REPO_ROOT / "scripts" / "update_boot_budget_snapshots.py"
    assert script.is_file()
    source = script.read_text(encoding="utf-8")
    assert "REFUSING to pin" in source
    assert "--force" in source
    assert "ADR-097" in source
