"""Verify real Import success, retry and restart with disposable local files.

Usage: native_success_check.py PROFILE TMUX_SOCKET SESSION success|interrupt|recover|reopen
Run phases in order as separate processes. No parser, pool or writer is replaced.
"""

import asyncio
import hashlib
import json
import os
import runpy
import subprocess
import sys
import traceback
from pathlib import Path


def main():
    root = Path(sys.argv[1]).resolve()
    socket, session, phase = sys.argv[2:5]
    if phase not in {"success", "interrupt", "recover", "reopen"}:
        raise ValueError("Unknown phase")
    runpy.run_path(str(Path(__file__).with_name("native_check.py")))[
        "validate_profile"
    ](root)
    evidence = root / phase
    evidence.mkdir(exist_ok=False)
    os.environ["TLDW_CONFIG_PATH"] = str(root / "config.toml")
    os.environ["XDG_DATA_HOME"] = str(root / "data")
    os.environ["XDG_CONFIG_HOME"] = str(root / "config")
    os.environ["PYTHON_KEYRING_BACKEND"] = "keyring.backends.null.Keyring"
    os.environ.pop("NO_COLOR", None)

    from textual.css.query import NoMatches, QueryError
    from textual.widgets import Button, Checkbox, Input
    from textual_image._terminal import probe_terminal

    import tldw_chatbook.app as app_module
    from tldw_chatbook.app import TldwCli
    from tldw_chatbook.Widgets.Library.library_media_content import (
        LibraryMediaContentBody,
    )

    expected_app = Path(__file__).resolve().parents[4] / "tldw_chatbook/app.py"
    assert Path(app_module.__file__).resolve() == expected_app
    probe_terminal()
    app = TldwCli()
    result = {
        "pid": os.getpid(),
        "phase": phase,
        "steps": [],
        "app_module": str(expected_app),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}

    def record():
        (evidence / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    def painted(widget):
        region = widget.region
        strips = list(app.screen._compositor.render_strips())
        return "\n".join(
            strips[y].crop(region.x, region.right).text
            for y in range(max(0, region.y), min(region.bottom, len(strips)))
        )

    async def tmux(*args):
        return await asyncio.to_thread(
            subprocess.run,
            ["/opt/homebrew/bin/tmux", "-L", socket, *args],
            check=True,
            text=True,
            capture_output=True,
        )

    async def journey(pilot):
        registry = app.library_ingest_jobs

        async def wait_for(predicate, label, *, settle=True):
            result["waiting_for"] = label
            record()
            deadline = asyncio.get_running_loop().time() + 90
            while asyncio.get_running_loop().time() < deadline:
                try:
                    if predicate():
                        if settle:
                            await pilot.pause()
                        return
                except (NoMatches, QueryError):
                    pass
                await pilot.pause(0.03)
            raise AssertionError(f"{label}: focus={app.screen.focused!r}")

        async def press(selector):
            await wait_for(
                lambda selector=selector: bool(app.screen.query(selector)), selector
            )
            button = app.screen.query_one(selector, Button)
            assert button.display and not button.disabled, selector
            button.focus()
            await wait_for(lambda: app.screen.focused is button, "focus " + selector)
            if selector.startswith("#library-ingest-open-"):
                await wait_for(
                    lambda: str(button.label) in painted(button), "painted Open action"
                )
            await pilot.press("enter")

        async def enter_import():
            if app.screen.query("#library-ingest-path"):
                return
            if app.screen.query("#library-hub-action-import"):
                await press("#library-hub-action-import")
            else:
                grip = app.screen.query_one("#library-browse-library-grip", Button)
                if str(grip.tooltip).startswith("Expand"):
                    await press("#library-browse-library-grip")
                await press("#library-ingest-top-button")
            await wait_for(
                lambda: bool(app.screen.query("#library-ingest-path")), "Import"
            )

        async def stage(name, *, chunk=False):
            path = root / "sources" / name
            assert path.resolve().is_relative_to(root / "sources")
            field = app.screen.query_one("#library-ingest-path", Input)
            field.focus()
            field.value = str(path)
            await wait_for(
                lambda: (
                    app.screen._ingest_state.form.path == str(path)
                    and app.screen._ingest_state.form.preflight is not None
                    and not app.screen._ingest_state.form.preflight_checking
                    and bool(
                        app.screen._ingest_state.form.preflight.type_groups.get(
                            "generic"
                        )
                    )
                ),
                "preflight " + name,
            )
            for option in ("analyze", "generate_embeddings", "chunk"):
                app.screen.query_one("#opt-generic-" + option, Checkbox).value = (
                    chunk if option == "chunk" else False
                )
            await wait_for(
                lambda: (
                    app.screen._ingest_state.form.type_options["generic"].get(
                        "generate_embeddings"
                    )
                    is False
                ),
                "embeddings disabled",
            )
            await wait_for(
                lambda: (
                    not app.screen.query_one("#library-ingest-start", Button).disabled
                ),
                "Start",
            )
            return path

        async def capture(name):
            await wait_for(lambda: not app.screen.query("Toast"), "notices expired")
            app.save_screenshot(name + ".svg", path=str(evidence))

        async def resize(width, height):
            await tmux(
                "resize-window", "-t", session, "-x", str(width), "-y", str(height)
            )
            await wait_for(lambda: tuple(app.size) == (width, height), "resize")

        def count_media():
            return app.media_db.execute_query("SELECT COUNT(*) FROM Media").fetchone()[
                0
            ]

        def check_saved(job_id, name):
            job = registry.get_job(job_id)
            assert job.state.value == "done", job
            row = app.media_db.get_media_by_id(job.media_id)
            expected = (root / "sources" / name).read_text()
            assert row["content"] == expected
            assert not row["is_trash"]
            return row

        async def submitted(name, *, chunk=False, deny_read=False):
            path = await stage(name, chunk=chunk)
            before = {j.job_id for j in registry.jobs()}
            if deny_read:
                path.chmod(0)
            await press("#library-ingest-start")
            await wait_for(
                lambda: len(registry.jobs()) == len(before) + 1,
                "job created",
                settle=False,
            )
            return next(j.job_id for j in registry.jobs() if j.job_id not in before)

        async def settled(job_id):
            await wait_for(
                lambda: (
                    registry.get_job(job_id).state.value
                    in {"done", "failed", "skipped"}
                ),
                "job settled",
            )
            assert not app.screen.query(f"#library-ingest-cancel-{job_id}")
            return registry.get_job(job_id)

        async def open_saved(job_id, name):
            await press(f"#library-ingest-open-{job_id}")
            expected_id = (
                registry.get_job(job_id).media_id or manifest["alpha_media_id"]
            )
            await wait_for(
                lambda: (
                    str(app.screen._media_state.reader_session.loaded_backing_id)
                    == str(expected_id)
                ),
                "media reader identity",
            )
            await wait_for(
                lambda: bool(app.screen.query(LibraryMediaContentBody)), "media content"
            )
            body = app.screen.query_one(LibraryMediaContentBody)
            assert body.content == (root / "sources" / name).read_text().strip()
            marker = {
                "alpha.txt": "Alpha import verification.",
                "beta.md": "Beta import verification",
                "permission.txt": "Permission recovered.",
            }[name]
            await wait_for(
                lambda: marker in " ".join(painted(body).split()),
                "painted saved content",
            )
            if phase == "reopen" and name == "beta.md":
                await capture(f"reader-{app.theme}-{app.size.width}")
            result["steps"].append(
                {
                    "opened": name,
                    "media_id": expected_id,
                    "theme": app.theme,
                    "size": list(app.size),
                    "body_matches_normalized_presentation": True,
                    "visible_content_marker": marker,
                }
            )
            await enter_import()

        async def matrix(job_id, kind, *, capture_prefix):
            for theme in ("textual-dark", "textual-light"):
                app.theme = theme
                for width, height in ((170, 48), (80, 24)):
                    await resize(width, height)
                    selector = f"#library-ingest-{kind}-{job_id}"
                    await wait_for(
                        lambda selector=selector: bool(app.screen.query(selector)),
                        selector,
                    )
                    button = app.screen.query_one(selector, Button)
                    button.focus()
                    await wait_for(
                        lambda button=button: str(button.label) in painted(button),
                        "painted action",
                    )
                    await capture(f"{capture_prefix}-{theme}-{width}")
                    if kind == "open":
                        await open_saved(job_id, "alpha.txt")
                        await open_saved(manifest["beta.md"], "beta.md")
            await resize(170, 48)
            record()

        try:
            assert app._instance_lock_status.acquired
            result["exclusive_profile"] = True
            await wait_for(lambda: getattr(app, "_ui_ready", False), "app ready")
            await wait_for(lambda: registry._store is not None, "durable registry")
            result["driver"] = type(app._driver).__name__
            assert result["driver"] == "LinuxDriver"
            await resize(170, 48)
            await pilot.press("ctrl+3")
            await wait_for(
                lambda: type(app.screen).__name__ == "LibraryScreen", "Library"
            )
            await enter_import()
            if phase != "success":
                original = registry.get_job(manifest["permission_failed"])
                retried = registry.get_job(manifest["permission.txt"])
                assert original.state.value == "failed" and original.superseded
                assert "Permission denied" in original.error and not original.permanent
                assert (
                    retried.retry_of_job_id == original.job_id
                    and retried.retry_count == 1
                )
                duplicate = registry.get_job(manifest["duplicate"])
                assert duplicate.state.value == "done"
                assert duplicate.media_id in {None, manifest["alpha_media_id"]}
                assert (
                    duplicate.progress and "already" in str(duplicate.progress).lower()
                )
                result["steps"].append(
                    {
                        "permission_lineage_survived_restart": True,
                        "duplicate_identity_survived_restart": True,
                    }
                )
            if phase == "reopen":
                original = registry.get_job(manifest["interrupted"])
                retried = registry.get_job(manifest["interrupt.txt"])
                assert original.state.value == "failed" and original.superseded
                assert (
                    original.error == "Interrupted by app restart"
                    and not original.permanent
                )
                assert (
                    retried.retry_of_job_id == original.job_id
                    and retried.retry_count == 1
                )
                result["steps"].append(
                    {"interruption_lineage_survived_second_restart": True}
                )
            if phase == "success":
                await stage("alpha.txt")
                await press("#library-ingest-clear-path")
                await wait_for(
                    lambda: (
                        app.screen.query_one("#library-ingest-path", Input).value == ""
                    ),
                    "clear",
                )
                assert not registry.jobs() and count_media() == 0
                result["steps"].append(
                    {"clear_unsubmitted": True, "jobs": 0, "media": 0}
                )
                for name in ("alpha.txt", "beta.md"):
                    job_id = await submitted(name)
                    manifest[name] = job_id
                    await settled(job_id)
                    check_saved(job_id, name)
                manifest["alpha_media_id"] = registry.get_job(
                    manifest["alpha.txt"]
                ).media_id
                assert count_media() == 2
                await matrix(manifest["alpha.txt"], "open", capture_prefix="completed")
                duplicate_id = await submitted("alpha.txt")
                duplicate = await settled(duplicate_id)
                assert duplicate.state.value == "done", duplicate
                assert count_media() == 2
                assert (
                    duplicate.progress and "already" in str(duplicate.progress).lower()
                ), duplicate
                manifest["duplicate"] = duplicate_id
                await open_saved(duplicate_id, "alpha.txt")
                result["steps"].append(
                    {
                        "duplicate": duplicate_id,
                        "media_count": 2,
                        "resolved_original": True,
                    }
                )
                denied_id = await submitted("permission.txt", deny_read=True)
                manifest["permission_failed"] = denied_id
                failed = await settled(denied_id)
                assert failed.state.value == "failed" and not failed.permanent, failed
                assert "Permission denied" in failed.error, failed.error
                assert count_media() == 2
                await matrix(denied_id, "retry", capture_prefix="permission-retry")
                (root / "sources/permission.txt").chmod(0o600)
                await press(f"#library-ingest-retry-{denied_id}")
                await wait_for(
                    lambda: any(
                        j.retry_of_job_id == denied_id for j in registry.jobs()
                    ),
                    "permission retry",
                )
                retry = next(
                    j.job_id for j in registry.jobs() if j.retry_of_job_id == denied_id
                )
                await settled(retry)
                check_saved(retry, "permission.txt")
                assert registry.get_job(denied_id).superseded
                assert registry.get_job(retry).retry_count == 1
                assert count_media() == 3
                manifest["permission.txt"] = retry
                result["steps"].append(
                    {
                        "permission_failure": denied_id,
                        "successful_retry": retry,
                        "lineage_retained": True,
                    }
                )
                await open_saved(retry, "permission.txt")
            elif phase == "interrupt":
                assert count_media() == 3
                for name in ("alpha.txt", "beta.md", "permission.txt"):
                    check_saved(manifest[name], name)
                job_id = await submitted("interrupt.txt", chunk=True)
                manifest["interrupted"] = job_id
                await wait_for(
                    lambda: bool(registry.get_job(job_id).progress),
                    "real worker progress",
                    settle=False,
                )
                job = registry.get_job(job_id)
                assert job.state.value == "parsing", job
                assert not app.screen.query(f"#library-ingest-cancel-{job_id}")
                result["steps"].append(
                    {
                        "quit_while": job.state.value,
                        "worker_progress": job.progress,
                        "job_id": job_id,
                        "pool_worker_pids": [
                            p.pid for p in app._ingest_parse_pool._pool
                        ],
                        "cancel_not_offered": True,
                    }
                )
            elif phase == "recover":
                interrupted = registry.get_job(manifest["interrupted"])
                assert interrupted.state.value == "failed" and not interrupted.permanent
                assert interrupted.error == "Interrupted by app restart", interrupted
                assert count_media() == 3
                await matrix(
                    interrupted.job_id, "retry", capture_prefix="interrupted-retry"
                )
                await press(f"#library-ingest-retry-{interrupted.job_id}")
                await wait_for(
                    lambda: any(
                        j.retry_of_job_id == interrupted.job_id for j in registry.jobs()
                    ),
                    "interrupted retry",
                )
                retry = next(
                    j.job_id
                    for j in registry.jobs()
                    if j.retry_of_job_id == interrupted.job_id
                )
                await settled(retry)
                check_saved(retry, "interrupt.txt")
                assert registry.get_job(interrupted.job_id).superseded
                assert registry.get_job(retry).retry_count == 1
                assert count_media() == 4
                manifest["interrupt.txt"] = retry
                result["steps"].append(
                    {
                        "interrupted_reconciled": True,
                        "successful_retry": retry,
                        "lineage_retained": True,
                        "media_count": 4,
                    }
                )
            elif phase == "reopen":
                assert count_media() == 4
                for name in ("alpha.txt", "beta.md", "permission.txt", "interrupt.txt"):
                    check_saved(manifest[name], name)
                for theme in ("textual-dark", "textual-light"):
                    app.theme = theme
                    for width, height in ((170, 48), (80, 24)):
                        await resize(width, height)
                        for name in ("alpha.txt", "beta.md", "permission.txt"):
                            await open_saved(manifest[name], name)
                result["steps"].append(
                    {
                        "fresh_process_reopened": True,
                        "media_count": 4,
                        "jobs": len(registry.jobs()),
                    }
                )
            result["passed"] = True
            record()
            await tmux("send-keys", "-t", session, "C-q")
        except Exception:  # noqa: BLE001 - preserve failure before normal shutdown
            (root / "sources/permission.txt").chmod(0o600)
            result["passed"] = False
            result["error"] = traceback.format_exc()
            app.save_screenshot("failed-state.svg", path=str(evidence))
            record()
            await tmux("send-keys", "-t", session, "C-q")

    app.run(auto_pilot=journey, size=(170, 48))
    result["app_run_returned"] = True
    record()
    raise SystemExit(0 if result.get("passed") else 1)


if __name__ == "__main__":
    main()
