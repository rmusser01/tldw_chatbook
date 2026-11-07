"""Shared native Git qualification and physical transport (ADR-285)."""

from __future__ import annotations

import asyncio
import os
import re
import subprocess


async def run_native_git(producer, cmd, env, payload, *, cwd, timeout):
    """Join the original worker's streams and managed tree before it may return."""
    from tldw_chatbook.Notes.git_process_containment import (
        ProcessTreeAdmissionError,
        ProcessTreeController,
    )

    controller = ProcessTreeController()
    tree = communication = None
    try:
        try:
            tree = await controller.spawn(
                *cmd,
                cwd=cwd,
                environment=env,
                stdin=True if payload is not None else None,
            )
        except ProcessTreeAdmissionError as error:
            tree = error.tree
            raise
        communication = asyncio.create_task(tree.process.communicate(payload))
        deadline = asyncio.get_running_loop().time() + timeout
        try:
            stdout, stderr = await asyncio.wait_for(
                asyncio.shield(communication), timeout
            )
        except TimeoutError:
            raise subprocess.TimeoutExpired(cmd, timeout) from None
        if not await controller.wait(
            tree, timeout=max(0, deadline - asyncio.get_running_loop().time())
        ):
            raise subprocess.TimeoutExpired(cmd, timeout)
        controller.close(tree)
        return tree.process.returncode, stdout, stderr
    except BaseException:
        if tree is None:
            raise
        producer.operation.mark_stopping()
        if communication is None:
            communication = asyncio.create_task(tree.process.communicate(payload))
        try:
            for stop in (controller.terminate, controller.kill):
                try:
                    stop(tree)
                except OSError:
                    pass  # The same retained tree still needs positive wait proof.
                if await controller.wait(tree, timeout=0.25):
                    break
            else:
                if not await controller.wait(tree, timeout=float("inf")):
                    raise OSError("native Git cleanup proof unavailable")
            await asyncio.shield(asyncio.gather(communication, return_exceptions=True))
            controller.close(tree)
        except BaseException:  # noqa: BLE001 -- no proof permits original scopes to return
            producer.owner.mark_cleanup_unproven()
            # No proof mechanism remains: keep this original worker/storage alive.
            never = asyncio.get_running_loop().create_future()
            while True:
                try:
                    await asyncio.shield(never)
                except asyncio.CancelledError:
                    continue
        raise


def qualified_git_command(
    executable, repository_prefix, args, env, run_process, unavailable
):
    """Apply native Git version/config policy without exposing private config."""
    allowed = {
        "init",
        "config",
        "rev-parse",
        "cat-file",
        "update-index",
        "ls-files",
        "ls-tree",
        "add",
        "rm",
        "diff",
        "commit",
        "show",
        "check-ignore",
    }
    if not args or args[0] not in allowed:
        raise unavailable("native change-review Git command unsupported")
    pins = (
        "core.hooksPath=" + os.devnull,
        "commit.gpgsign=false",
        "tag.gpgsign=false",
        "log.showsignature=false",
        "core.fsmonitor=false",
        "gc.auto=0",
        "gc.autoDetach=false",
        "maintenance.auto=false",
        "maintenance.autoDetach=false",
    )
    prefix = [executable, "--no-pager", *repository_prefix]
    for pin in pins:
        prefix.extend(("-c", pin))
    try:
        version = run_process([executable, "--version"], env)
        match = re.match(r"git version (\d+)\.(\d+)", version.stdout)
        if (
            version.returncode
            or match is None
            or tuple(map(int, match.groups())) < (2, 36)
        ):
            raise unavailable("native change review requires Git >=2.36")
        config = run_process(
            [
                *prefix,
                "config",
                "--includes",
                "--null",
                "--name-only",
                "--get-regexp",
                r"^(filter\..*\.(clean|process|smudge)|remote\..*\.promisor|extensions\.partialclone)$",
            ],
            env,
            binary=True,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise unavailable(
            "native change-review Git qualification unavailable"
        ) from error
    if config.returncode == 0:
        # Names stay private; even cat-file -e may lazily spawn a fetch.
        if not any(name.startswith(b"filter.") for name in config.stdout.split(b"\0")):
            raise unavailable(
                "native change review unavailable: configured partial-clone remotes"
            )
        raise unavailable("native change review unavailable: configured filters")
    if config.returncode != 1:
        raise unavailable("native change-review Git configuration unavailable")
    if args[0] in {"diff", "show"}:
        args = (args[0], "--no-ext-diff", "--no-textconv", *args[1:])
    return [*prefix, *args]
