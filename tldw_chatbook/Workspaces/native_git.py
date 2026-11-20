"""Shared native Git qualification and physical transport (ADR-285)."""

from __future__ import annotations

import asyncio
import os
import re
import subprocess


async def run_native_git(producer, cmd, env, payload, *, cwd, timeout_seconds):
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
        deadline = asyncio.get_running_loop().time() + timeout_seconds
        try:
            stdout, stderr = await asyncio.wait_for(
                asyncio.shield(communication), timeout_seconds
            )
        except TimeoutError:
            raise subprocess.TimeoutExpired(cmd, timeout_seconds) from None
        if not await controller.wait(
            tree, timeout=max(0, deadline - asyncio.get_running_loop().time())
        ):
            raise subprocess.TimeoutExpired(cmd, timeout_seconds)
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
            # No proof mechanism remains: this daemon worker keeps the original
            # storage until restart; its waiters stop waiting (TASK-40716).
            producer.operation.retain_unproven("native Git").hold()
        raise


def scrubbed_git_environment() -> dict[str, str]:
    """Environment for native Git: ``GIT_*`` scrubbed, prompts disabled."""
    # Case-INSENSITIVE match: Windows environment variables are
    # case-insensitive, so `Git_Index_File` reaches git just as
    # GIT_INDEX_FILE does. Harmless over-scrub on POSIX.
    env = {k: v for k, v in os.environ.items() if not k.upper().startswith("GIT_")}
    env["GIT_TERMINAL_PROMPT"] = "0"
    return env


# Invocations that can hand work-tree content to a clean/smudge/process filter;
# a configured driver a reachable path selects refuses them. Refusal decides the
# outcome; it is not the guarantee -- every configured driver is also emptied for
# the command itself (`_neutralize_filters`), so Git never runs one.
_FILTER_CAPABLE = frozenset({"add", "update-index", "rm", "commit", "diff"})


def _filter_capable(args) -> bool:
    if args[0] == "cat-file":
        return "--filters" in args
    if args[0] == "ls-files":
        return any(arg in ("-m", "--modified") for arg in args)
    return args[0] in _FILTER_CAPABLE


def _neutralize_filters(env, drivers) -> None:
    """Empty every configured driver's commands for this one invocation.

    Through GIT_CONFIG_COUNT rather than `-c`, so driver names stay out of
    argv; Git children (submodule status) inherit it. A selected path whose
    driver is `required` then fails the command instead of running anything.
    """
    count = int(env.get("GIT_CONFIG_COUNT") or 0)
    for driver in sorted(drivers):
        name = os.fsdecode(driver)
        for part in ("clean", "smudge", "process"):
            env[f"GIT_CONFIG_KEY_{count}"] = f"filter.{name}.{part}"
            env[f"GIT_CONFIG_VALUE_{count}"] = ""
            count += 1
    env["GIT_CONFIG_COUNT"] = str(count)


def _selects_configured_filter(prefix, args, env, run_process, drivers, payload):
    """True when a path this invocation can reach selects a configured driver.

    Git decides selection itself (`check-attr`, macros included) over every
    tracked and non-ignored untracked path, plus the exact paths an
    ``update-index --stdin`` stages by name, which may be ignored (TASK-42101).
    """
    # ponytail: O(paths) per filter-capable invocation, and only when a filter
    # is configured; never cached, because attributes can change between
    # invocations the same way config can.
    listing = run_process(
        [*prefix, "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        env,
        binary=True,
    )
    paths = listing.stdout
    if args[0] == "update-index" and "--stdin" in args and payload:
        paths += os.fsencode(payload) if isinstance(payload, str) else payload
    attributes = run_process(
        [*prefix, "check-attr", "-z", "--stdin", "filter"],
        env,
        binary=True,
        input_data=paths,
    )
    if listing.returncode or attributes.returncode:
        return None
    return any(value in drivers for value in attributes.stdout.split(b"\0")[2::3])


def qualified_git_command(
    executable, repository_prefix, args, env, run_process, unavailable, payload=None
):
    """Apply native Git version/config policy without exposing private config.

    A configured clean/process/smudge filter refuses the invocation only when a
    path it can reach selects that driver (TASK-42101); partial-clone remotes
    always refuse. When filters are configured, ``env`` (this invocation's own
    copy) also gets every configured driver emptied for the returned command.
    """
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
    if (args[0] == "cat-file" and "--textconv" in args) or (
        # Detection reads stdin paths NUL-delimited; quoted lines would not match.
        args[0] == "update-index" and "--stdin" in args and "-z" not in args
    ):
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
    if config.returncode not in (0, 1):
        raise unavailable("native change-review Git configuration unavailable")
    # Names stay private; even cat-file -e may lazily spawn a fetch.
    names = [name for name in config.stdout.split(b"\0") if name]
    if any(not name.startswith(b"filter.") for name in names):
        raise unavailable(
            "native change review unavailable: configured partial-clone remotes"
        )
    if names and _filter_capable(args):
        drivers = {name[len(b"filter.") : name.rindex(b".")] for name in names}
        try:
            selected = _selects_configured_filter(
                prefix, args, env, run_process, drivers, payload
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise unavailable(
                "native change-review Git qualification unavailable"
            ) from error
        if selected is None:
            raise unavailable("native change-review Git configuration unavailable")
        if selected:
            raise unavailable(
                "native change review unavailable: configured filters selected by"
                " Git attributes (remove the filter= attribute or its filter config)"
            )
    if names:
        _neutralize_filters(
            env, {name[len(b"filter.") : name.rindex(b".")] for name in names}
        )
    if args[0] == "diff":
        # A dirty-submodule check runs `git status` inside the submodule, whose
        # own config this qualification never reads.
        args = (
            "diff",
            "--no-ext-diff",
            "--no-textconv",
            "--ignore-submodules=dirty",
            *args[1:],
        )
    elif args[0] == "show":
        args = ("show", "--no-ext-diff", "--no-textconv", *args[1:])
    return [*prefix, *args]
