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
    # rule 2: filesystem format
    ("mkfs /dev/sda1", "filesystem-format"),
    ("mkfs.ext4 /dev/nvme0n1", "filesystem-format"),
    ("sudo mkfs.xfs -f /dev/sdb", "filesystem-format"),
    # rule 3: dd onto a block device
    ("dd if=/dev/zero of=/dev/sda", "dd-to-block-device"),
    ("dd if=image.iso of=/dev/disk2 bs=4m", "dd-to-block-device"),
    ("sudo dd of=/dev/nvme0n1 if=payload", "dd-to-block-device"),
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
