"""Verify the observed Import pool-allocation failure with a private profile.

Usage: native_check.py PROFILE TMUX_SOCKET SESSION submit|restart
Run submit then restart in separate app processes on the same fresh profile.
This failure-specific runner is expected to fail once host allocation recovers;
it does not qualify successful import. No parser, pool or writer is substituted.
The main guard is essential: the real parse pool uses multiprocessing spawn.
"""

import asyncio
import json
import os
import subprocess
import sys
import tomllib
import traceback
from pathlib import Path


def validate_profile(root: Path) -> None:
    """Reject missing or escaping data paths before app imports can write."""
    root = root.resolve()
    config_path = root / "config.toml"
    if not config_path.resolve().is_relative_to(root):
        raise ValueError("Config must be inside the private profile")
    config = tomllib.loads(config_path.read_text())
    database = config["database"]
    directories = [config["paths"]["data_dir"], database["USER_DB_BASE_DIR"]]
    files = [value for key, value in database.items() if key.endswith("_db_path")]
    for raw in directories + files:
        if not isinstance(raw, str) or not raw or not Path(raw).is_absolute():
            raise ValueError("Private paths must be explicit and absolute")
        path = Path(raw).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Data and database paths must stay inside the profile")
        parent = path if raw in directories else path.parent
        if not parent.is_dir():
            raise ValueError("Private data directories must exist before launch")


def main():
    root = Path(sys.argv[1]).resolve()
    socket, session, phase = sys.argv[2:5]
    if phase not in {"submit", "restart"}:
        raise ValueError("Choose submit or restart")
    validate_profile(root)
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

    from tldw_chatbook.app import TldwCli

    probe_terminal()
    app = TldwCli()
    result = {"pid": os.getpid(), "phase": phase, "steps": []}
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

        async def wait_for(predicate, label):
            deadline = asyncio.get_running_loop().time() + 45
            while asyncio.get_running_loop().time() < deadline:
                try:
                    if predicate():
                        await pilot.pause()
                        return
                except (NoMatches, QueryError):
                    pass
                await pilot.pause(0.03)
            raise AssertionError(f"{label}: focus={app.screen.focused!r}")

        async def press(selector):
            await wait_for(lambda: bool(app.screen.query(selector)), selector)
            button = app.screen.query_one(selector, Button)
            assert not button.disabled, selector
            button.focus()
            await wait_for(lambda: app.screen.focused is button, "focus " + selector)
            await pilot.press("enter")

        async def enter_import():
            # Escape leaves the reader/Import destination for the Library hub.
            for _ in range(3):
                if app.screen.query("#library-hub-action-import"):
                    break
                await pilot.press("escape")
                await pilot.pause()
            await press("#library-hub-action-import")
            await wait_for(
                lambda: bool(app.screen.query("#library-ingest-path")), "Import"
            )

        async def stage(name):
            path = root / "sources" / name
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
            for option in ("analyze", "generate_embeddings"):
                app.screen.query_one("#opt-generic-" + option, Checkbox).value = False
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

        async def matrix(job_id, kind):
            for theme in ("textual-dark", "textual-light"):
                app.theme = theme
                for width, height in ((170, 48), (80, 24)):
                    await tmux(
                        "resize-window",
                        "-t",
                        session,
                        "-x",
                        str(width),
                        "-y",
                        str(height),
                    )
                    await wait_for(
                        lambda width=width, height=height: (
                            tuple(app.size) == (width, height)
                        ),
                        "native resize",
                    )
                    button = app.screen.query_one(
                        f"#library-ingest-{kind}-{job_id}", Button
                    )
                    button.focus()
                    await wait_for(
                        lambda button=button: str(button.label) in painted(button),
                        "painted action",
                    )
                    await capture(f"{kind}-{theme}-{width}")
            result["steps"].append({"matrix": kind, "themes": 2, "sizes": 2})
            record()

        try:
            assert app._instance_lock_status.acquired
            result["exclusive_profile"] = True
            await wait_for(lambda: getattr(app, "_ui_ready", False), "app ready")
            await wait_for(lambda: registry._store is not None, "durable registry")
            result["driver"] = type(app._driver).__name__
            assert result["driver"] == "LinuxDriver"
            await pilot.press("ctrl+3")
            await wait_for(
                lambda: type(app.screen).__name__ == "LibraryScreen", "Library"
            )
            await enter_import()
            if phase == "submit":
                await stage("alpha.txt")
                await press("#library-ingest-clear-path")
                await wait_for(
                    lambda: (
                        app.screen.query_one("#library-ingest-path", Input).value == ""
                    ),
                    "staged clear",
                )
                assert not registry.jobs()
                assert (
                    app.media_db.execute_query("SELECT COUNT(*) FROM Media").fetchone()[
                        0
                    ]
                    == 0
                )
                result["steps"].append(
                    {"clear_unsubmitted": True, "jobs": 0, "media": 0}
                )
                await stage("alpha.txt")
                await press("#library-ingest-start")
                await wait_for(lambda: len(registry.jobs()) == 1, "submitted")
                job_id = registry.jobs()[0].job_id
                manifest["alpha.txt"] = job_id
                await wait_for(
                    lambda: registry.get_job(job_id).state.value in {"done", "failed"},
                    "settled",
                )
                failed = registry.get_job(job_id)
                assert failed.state.value == "failed" and not failed.permanent
                assert failed.error.startswith("Parse pool could not start:")
                assert "[Errno 28]" in failed.error
                assert failed.media_id is None
                result["steps"].append(
                    {"pool_unavailable": True, "error": failed.error}
                )
            elif phase == "restart":
                failed = registry.get_job(manifest["alpha.txt"])
                assert failed.state.value == "failed" and not failed.permanent
                assert failed.error.startswith("Parse pool could not start:")
                assert "[Errno 28]" in failed.error
                result["steps"].append(
                    {"failure_survived_restart": True, "error": failed.error}
                )
                await matrix(failed.job_id, "retry")
                await press(f"#library-ingest-retry-{failed.job_id}")
                await wait_for(
                    lambda: any(
                        j.retry_of_job_id == failed.job_id for j in registry.jobs()
                    ),
                    "retry created",
                )
                successor = next(
                    j.job_id
                    for j in registry.jobs()
                    if j.retry_of_job_id == failed.job_id
                )
                await wait_for(
                    lambda: registry.get_job(successor).state.value == "failed",
                    "pool failure retried",
                )
                retried = registry.get_job(successor)
                assert retried.error.startswith("Parse pool could not start:")
                assert not retried.permanent and retried.retry_count == 1
                assert registry.get_job(failed.job_id).superseded
                assert (
                    app.media_db.execute_query("SELECT COUNT(*) FROM Media").fetchone()[
                        0
                    ]
                    == 0
                )
                result["steps"].append(
                    {
                        "retry_of": failed.job_id,
                        "successor": successor,
                        "retryable_failure_retained": True,
                        "media": 0,
                    }
                )
            else:
                raise ValueError(phase)
            result["passed"] = True
            record()
            await tmux("send-keys", "-t", session, "C-q")
        except Exception:  # noqa: BLE001 - preserve failure before normal shutdown
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
