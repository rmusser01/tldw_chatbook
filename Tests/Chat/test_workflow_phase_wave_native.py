"""Real accepted native phases share a finite Change Review wave."""

from pathlib import Path

import pytest

from Tests.Backup_Recovery.test_home_citation_retirement import _run

pytestmark = pytest.mark.bootstrap_profile


def test_two_native_bounded_tasks_overlap_inside_one_actual_capture_wave(tmp_path):
    _run(
        tmp_path,
        "overlap",
        "success",
        script=Path(__file__).with_name("workflow_wave_probe.py").read_text(),
        timeout=60,
    )


def test_control_race_after_actual_baseline_closes_unused_without_child(tmp_path):
    _run(
        tmp_path,
        "cancel-after-baseline",
        "success",
        script=Path(__file__).with_name("workflow_wave_probe.py").read_text(),
        timeout=60,
    )


def test_shutdown_waits_for_original_held_models_and_final_snapshot(tmp_path):
    _run(
        tmp_path,
        "shutdown-held",
        "success",
        script=Path(__file__).with_name("workflow_wave_probe.py").read_text(),
        timeout=60,
    )


def test_actual_tool_finish_return_precedes_native_provider_unbind(tmp_path):
    _run(
        tmp_path,
        "held-tool-return",
        "success",
        script=Path(__file__).with_name("workflow_wave_probe.py").read_text(),
        timeout=60,
    )


def test_capture_off_tasks_keep_original_source_until_native_retirement(tmp_path):
    _run(
        tmp_path,
        "capture-off",
        "success",
        script=Path(__file__).with_name("workflow_wave_probe.py").read_text(),
        timeout=60,
    )


@pytest.mark.parametrize("failure", ["publisher-first-fails", "publisher-both-fail"])
def test_actual_required_e_publication_failures_keep_truthful_custody(
    tmp_path, failure
):
    _run(
        tmp_path,
        failure,
        "success",
        script=Path(__file__).with_name("workflow_wave_probe.py").read_text(),
        timeout=60,
    )


def test_real_manual_gap_closes_e_then_takes_fresh_b(tmp_path):
    _run(
        tmp_path,
        "manual-gap",
        "success",
        script=Path(__file__).with_name("workflow_wave_probe.py").read_text(),
        timeout=90,
    )


def test_actual_native_window_archive_rejects_missing_required_reference(tmp_path):
    _run(
        tmp_path,
        "archive-controls",
        "success",
        script=Path(__file__).with_name("workflow_wave_probe.py").read_text(),
        timeout=60,
    )


def test_native_wave_complete_archive_restore_and_rebackup(tmp_path):
    from Tests.Chat.workflow_wave_archive import restore_and_rebackup

    original = tmp_path / "original"
    original.mkdir()
    _run(
        original,
        "complete-archive",
        "success",
        script=Path(__file__).with_name("workflow_wave_probe.py").read_text(),
        timeout=90,
    )
    restore_and_rebackup(original)


def test_failed_actual_native_creation_returns_original_close_obligation(tmp_path):
    _run(
        tmp_path,
        "native-create-fails",
        "success",
        script=Path(__file__).with_name("workflow_wave_probe.py").read_text(),
        timeout=45,
    )


def test_required_native_refs_survive_pruning_until_settled_payload_removal(tmp_path):
    _run(
        tmp_path,
        "retention",
        "success",
        script=Path(__file__).with_name("workflow_wave_probe.py").read_text(),
        timeout=60,
    )


def test_actual_publisher_return_retains_wave_after_immutable_task_checkpoint(tmp_path):
    _run(
        tmp_path,
        "publisher-return",
        "success",
        script=Path(__file__).with_name("workflow_wave_probe.py").read_text(),
        timeout=80,
    )


@pytest.mark.parametrize(
    "ordering", ["earlier-manual", "manual-before-child", "ready-queue"]
)
def test_original_manual_admission_and_open_wave_ordering(tmp_path, ordering):
    _run(
        tmp_path,
        ordering,
        "success",
        script=Path(__file__).with_name("workflow_wave_probe.py").read_text(),
        timeout=70,
    )


def test_changed_actual_alias_handler_refuses_before_undeclared_effect(tmp_path):
    _run(
        tmp_path,
        "alias-handler",
        "success",
        script=Path(__file__).with_name("workflow_wave_probe.py").read_text(),
        timeout=60,
    )


def test_known_window_rollback_retires_before_native_ordinary_send(tmp_path):
    _run(
        tmp_path,
        "window-refused",
        "success",
        script=Path(__file__).with_name("workflow_wave_probe.py").read_text(),
        timeout=60,
    )


@pytest.mark.parametrize("failure", ["window-ack-unknown", "window-unwind-unknown"])
def test_unknown_window_publication_or_unwind_keeps_original_protection(
    tmp_path, failure
):
    _run(
        tmp_path,
        failure,
        "success",
        script=Path(__file__).with_name("workflow_wave_probe.py").read_text(),
        timeout=60,
    )


def test_changed_receiving_handler_after_original_permission_return_is_not_invoked(
    tmp_path,
):
    _run(
        tmp_path,
        "alias-handler-after-permission",
        "success",
        script=Path(__file__).with_name("workflow_wave_probe.py").read_text(),
        timeout=60,
    )


def test_suppressed_original_owned_workspace_close_keeps_unpublished_wave(tmp_path):
    import json
    import os
    import subprocess
    import sys

    for name in ("home", "config", "data"):
        (tmp_path / name).mkdir(mode=0o700)
    environment = dict(
        os.environ,
        HOME=str(tmp_path / "home"),
        USERPROFILE=str(tmp_path / "home"),
        XDG_CONFIG_HOME=str(tmp_path / "config"),
        XDG_DATA_HOME=str(tmp_path / "data"),
        TLDW_CONFIG_PATH=str(tmp_path / "config/config.toml"),
        TLDW_TEST_MODE="1",
        TLDW_DISABLE_CONFIG_WATCH="1",
    )
    log = tmp_path / "owned-close-output.log"
    with log.open("w") as output:
        result = subprocess.run(
            [
                sys.executable,
                "-X",
                "utf8",
                "-c",
                Path(__file__).with_name("workflow_wave_probe.py").read_text(),
                "window-owned-close-unknown",
                "retained",
            ],
            cwd=Path(__file__).resolve().parents[2],
            env=environment,
            stdout=output,
            stderr=subprocess.STDOUT,
            timeout=60,
            check=False,
        )
    stdout = log.read_text()
    assert result.returncode == 0, stdout[-3000:]
    line = next(
        line
        for line in stdout.splitlines()
        if line.startswith("WAVE_OWNED_CLOSE_FACTS ")
    )
    print(line)
    facts = json.loads(line.removeprefix("WAVE_OWNED_CLOSE_FACTS "))
    assert facts["fault_count"] == facts["unpublished_count"] == 1
    assert all(
        facts[name]
        for name in (
            "original_workspace_receiver",
            "live_handle",
            "cached_original",
            "retained_lease",
            "retained_participant",
        )
    )
    assert facts["calls"] == {"left": 0, "right": 0}
    assert (
        facts["owners"]
        == facts["claims"]
        == facts["windows"]
        == facts["native_runs"]
        == 0
    )
    assert facts["waves"] == facts["fences"] == 1, (
        "unknown original owned close released its wave"
    )
    assert facts["sources"] == facts["preparations"] == 2
    assert facts["repeat_close_refused"], (
        "an early closed flag bypassed original return proof"
    )
