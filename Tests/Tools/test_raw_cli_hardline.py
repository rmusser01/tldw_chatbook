"""TASK-25905: the unbypassable hardline command floor.

A small catastrophic set is refused at the request-validation boundary --
before permission state, before any session grant, for BOTH callers. This
is a floor under the approval card, not a replacement for it, and it is
deliberately not configurable off (AC#5): the rule tuple is a module
constant with no config read anywhere near it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tldw_chatbook.Tools.raw_cli_executor import (
    RawCliHardlineViolation,
    RawCliRequest,
    hardline_violation,
    validate_raw_cli_request,
)


HARDLINE_CASES = [
    # rule 1: recursive root delete, plus trivial obfuscation (AC#3)
    ("rm -rf /", "recursive-root-delete"),
    ("rm -fr /", "recursive-root-delete"),
    ("rm   -rf    /", "recursive-root-delete"),
    ("rm -'r'f /", "recursive-root-delete"),
    ('rm -"rf" "/"', "recursive-root-delete"),
    ("rm \\-rf /", "recursive-root-delete"),
    ("sudo rm -rf /", "recursive-root-delete"),
    ("$DELETER -rf /", "recursive-root-delete"),
    ("cd /tmp && rm -rf /", "recursive-root-delete"),
    ("rm -rf /*", "recursive-root-delete"),
    ("rm -rf ~", "recursive-root-delete"),
    ("rm -rf $HOME", "recursive-root-delete"),
    ("rm --recursive --force /", "recursive-root-delete"),
    # review 2026-09-01: split flags + more root forms + variable command
    ("rm -r -f /", "recursive-root-delete"),
    ("rm -f -r /", "recursive-root-delete"),
    ("rm -rf //", "recursive-root-delete"),
    ("rm -rf -- /", "recursive-root-delete"),
    ("FOO=1 rm -rf /", "recursive-root-delete"),
    ("(rm -rf /)", "recursive-root-delete"),
    ("$DELETER -rf /", "recursive-root-delete"),
    ("rm -rf /; reboot", "recursive-root-delete"),
    # rule 2: filesystem format
    ("mkfs /dev/sda1", "filesystem-format"),
    ("mkfs.ext4 /dev/nvme0n1", "filesystem-format"),
    ("sudo mkfs.xfs -f /dev/sdb", "filesystem-format"),
    # rule 3: dd onto a block device
    ("dd if=/dev/zero of=/dev/sda", "dd-to-block-device"),
    ("dd if=image.iso of=/dev/disk2 bs=4m", "dd-to-block-device"),
    ("sudo dd of=/dev/nvme0n1 if=payload", "dd-to-block-device"),
    # review: macOS raw-disk device + RAID/mapper were missing
    ("dd if=/dev/zero of=/dev/rdisk2", "dd-to-block-device"),
    ("dd of=/dev/md0 if=x", "dd-to-block-device"),
    ("dd of=/dev/mapper/vg-root if=x", "dd-to-block-device"),
    # rule 4: fork bomb
    (":(){ :|:& };:", "fork-bomb"),
    (":() { : | : & } ; :", "fork-bomb"),
    ("bomb(){ bomb|bomb& };bomb", "fork-bomb"),
    # rule 5: shutdown/poweroff in command position
    ("shutdown -h now", "system-shutdown"),
    ("sudo shutdown -r +5", "system-shutdown"),
    ("poweroff", "system-shutdown"),
    ("true; reboot", "system-shutdown"),
    ("sudo halt", "system-shutdown"),
    # review: init runlevel + systemctl verb were missing
    ("init 6", "system-shutdown"),
    ("init 0", "system-shutdown"),
    ("systemctl poweroff", "system-shutdown"),
    ("sudo systemctl reboot", "system-shutdown"),
]

ORDINARY_CASES = [
    # AC#6: the developer corpus that must pass unaffected
    "git status",
    "git commit -m 'fix the shutdown handler race'",
    "git log --oneline | head -20",
    "npm install --save-dev vitest",
    "pytest Tests/Chat -k 'compaction' -q",
    "rm build/output.log",
    "rm -rf node_modules",
    "rm -rf ./dist",
    "rm -r /tmp/scratch-workdir",
    "make clean && make -j8",
    "cargo build --release",
    "docker compose up -d",
    "dd if=backup.img of=restored.img bs=1M",
    "python -c 'print(1)'",
    "grep -rf patterns.txt src/",
    "tar -cf halt.tar docs/",
    "echo 'never poweroff mid-write' >> notes.md",
    # review 2026-09-01: quoted separators + non-rm recursive flags must pass
    'git commit -m "fix crash; reboot handling"',
    "git commit -m 'works & shutdown flow tested'",
    "echo 'step1; halt if bad' >> runbook.md",
    "git commit -m \"test; dd of=/dev/sda1 notes\"",
    "ls -laRF /",
    "cp -rf / /mnt/backup",
    "ls -R /",
    "find / -name '*.log'",
    "systemctl status nginx",
    "init_project.sh",
    "dd if=/dev/zero of=disk.img bs=1M count=100",
]


@pytest.mark.parametrize(("command", "rule"), HARDLINE_CASES)
def test_catastrophic_commands_name_their_rule(command: str, rule: str) -> None:
    assert hardline_violation(command) == rule


@pytest.mark.parametrize("command", ORDINARY_CASES)
def test_ordinary_developer_commands_pass(command: str) -> None:
    assert hardline_violation(command) is None


def test_validation_boundary_refuses_before_any_permission_state(tmp_path: Path) -> None:
    """AC#1/#4: the floor fires inside request validation itself, with a
    typed error naming the rule -- distinguishable from a user denial."""
    request = RawCliRequest(
        invocation_id="inv-1",
        caller="model",
        command="rm -rf /",
        shell="auto",
        initial_directory=tmp_path,
        timeout_seconds=30,
        console_session_id="console-1",
    )
    with pytest.raises(RawCliHardlineViolation) as excinfo:
        validate_raw_cli_request(request)
    assert excinfo.value.rule == "recursive-root-delete"
    assert "hardline" in str(excinfo.value)

    # the same boundary applies to the USER caller -- the floor has no
    # caller exemption
    with pytest.raises(RawCliHardlineViolation):
        validate_raw_cli_request(
            RawCliRequest(
                invocation_id="inv-2",
                caller="user",
                command="mkfs.ext4 /dev/sda",
                shell="auto",
                initial_directory=tmp_path,
                timeout_seconds=30,
                console_session_id="console-1",
            )
        )


# --- TASK-26006: actionable failure hints -----------------------------------


def test_failure_hint_maps_known_shapes_first_match_wins() -> None:
    from tldw_chatbook.Tools.raw_cli_executor import failure_hint

    hint = failure_hint(127, "bash: pyest: command not found")
    assert hint is not None
    assert hint.startswith("[tool hint]"), "must be marked tool-generated"
    assert "PATH" in hint

    # first match wins even when several shapes appear (AC#3)
    combined = failure_hint(
        1, "bash: x: command not found\nPermission denied"
    )
    assert combined is not None
    assert "PATH" in combined
    assert len(combined) < 300, "hints are bounded"


def test_failure_hint_only_on_nonzero_exit() -> None:
    from tldw_chatbook.Tools.raw_cli_executor import failure_hint

    assert failure_hint(0, "command not found") is None
    assert failure_hint(None, "command not found") is None
    assert failure_hint(1, "some novel failure text") is None


def test_failure_hint_table_is_data() -> None:
    """AC#6: adding a shape is a table row, not control flow."""
    from tldw_chatbook.Tools import raw_cli_executor

    table = raw_cli_executor.FAILURE_HINT_TABLE
    assert len(table) >= 6
    assert all(len(row) == 2 for row in table)
