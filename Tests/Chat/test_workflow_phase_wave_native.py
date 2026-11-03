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
    import os
    import subprocess
    import sys

    from Tests.Backup_Recovery.test_temporary_media_capture import _RESTORE
    from Tests.Chat.workflow_wave_archive import REOPEN

    original = tmp_path / "original"
    original.mkdir()
    _run(
        original,
        "complete-archive",
        "success",
        script=Path(__file__).with_name("workflow_wave_probe.py").read_text(),
        timeout=90,
    )
    restored = tmp_path / "restored"
    restored.mkdir()
    restore = _RESTORE.replace(
        "source.rename(source.with_name('source-home-removed'))",
        "(home/'wave.json').write_bytes((source/'wave.json').read_bytes())\nsource.rename(source.with_name('source-home-removed'))",
    )
    restore = restore.replace(
        "'agents.history':'tool_sandbox'",
        "'workspaces.change_tracking':'change_review','tokenizers.custom':'tokenizers','agents.history':'tool_sandbox'",
    )
    restore = restore.replace(
        "mapping[f'profile:{profile}:paths.data_dir']",
        "(home / '.config/tldw_cli/tokenizers').mkdir(parents=True, mode=0o700, exist_ok=True)\nfor row in doc.directories:\n    if row.parent_id is None and not row.synthetic and producer[row.logical_id].owner_id == 'tokenizers.custom':\n        mapping[row.logical_id] = home / '.config/tldw_cli/tokenizers'\nmapping[f'profile:{profile}:paths.data_dir']",
    )
    _run(restored, str(original / "home"), "isolated", script=restore, timeout=60)
    environment = dict(
        os.environ,
        HOME=str(restored / "home"),
        USERPROFILE=str(restored / "home"),
        XDG_CONFIG_HOME=str(restored / "config"),
        XDG_DATA_HOME=str(restored / "data"),
        TLDW_CONFIG_PATH=str(restored / "config/config.toml"),
        TLDW_TEST_MODE="1",
        TLDW_DISABLE_CONFIG_WATCH="1",
    )
    result = subprocess.run(
        [sys.executable, "-c", REOPEN],
        cwd=Path(__file__).resolve().parents[2],
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    (tmp_path / "wave-rebackup.log").write_text(result.stdout + result.stderr)
    assert result.returncode == 0, result.stdout[-1000:] + result.stderr[-4000:]
    assert (restored / "home/second.tldw-backup.zip").is_file()


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
