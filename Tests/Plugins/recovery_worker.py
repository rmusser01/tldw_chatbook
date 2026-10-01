"""Independent controlled owner, isolated before any application import.

Invoked as a module by recovery tests. Only the parent may terminate this exact
process at a flushed durable-boundary message; no plugin subprocesses are run.
"""

import argparse
import asyncio
import json
import os
import sys
import threading
from concurrent.futures import Future
from dataclasses import asdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("install", "recover"))
    parser.add_argument("root", type=Path)
    parser.add_argument("--package", type=Path)
    parser.add_argument("--barrier")
    parser.add_argument("--operation", default="child-install")
    args = parser.parse_args()
    root = args.root.resolve()
    config = root / "config.toml"
    if Path(os.environ["TLDW_CONFIG_PATH"]) != config:
        raise RuntimeError("child config isolation missing before import")
    for name in ("XDG_DATA_HOME", "XDG_CONFIG_HOME", "XDG_CACHE_HOME"):
        if not Path(os.environ[name]).is_relative_to(root):
            raise RuntimeError("child XDG isolation missing before import")
    if not config.is_file():
        raise RuntimeError("explicit child profile config required")
    worktree = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(worktree))
    from Tests.hooks_v2_process_support import bootstrap

    bootstrap(str(worktree), {}, str(root / "data" / "crash-test"))
    result = Future()

    def worker():
        from tldw_chatbook import config as app_config
        from tldw_chatbook.Plugins import coordinator as coordinator_module
        from tldw_chatbook.Plugins.authority_store import (
            FilePluginMarkerStore,
            PluginAuthorityStore,
        )
        from tldw_chatbook.Plugins.coordinator import PluginCoordinator
        from tldw_chatbook.Plugins.inspection import inspect_package
        from tldw_chatbook.Plugins.registry import PluginRegistry
        from tldw_chatbook.Plugins.runtime_owner import PluginRuntimeOwner

        assert Path(coordinator_module.__file__).is_relative_to(worktree)
        profile = app_config.get_user_data_dir()
        assert profile.resolve() == (root / "data" / "crash-test").resolve()
        owner = PluginRuntimeOwner(profile / "plugins")
        registry = None
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            assert owner.try_acquire()
            registry = PluginRegistry(owner.root / "registry.sqlite3", owner=owner)
            authority = PluginAuthorityStore(
                root / "trust" / "plugins",
                FilePluginMarkerStore(root / "marker"),
                accept_reduced_protection=True,
            )
            coordinator = PluginCoordinator(registry, authority, owner)

            async def run():
                if authority.load_marker() is None:
                    coordinator.bootstrap("controlled child passphrase")
                else:
                    authority.unlock("controlled child passphrase")
                if args.command == "install":
                    await coordinator.recover()
                    review = coordinator.review(
                        inspect_package(args.package),
                        selection=("skill:review",),
                        workspace_id=None,
                    )

                    def progress(phase):
                        if phase == args.barrier:
                            print(
                                json.dumps(
                                    {
                                        "barrier": phase,
                                        "pid": os.getpid(),
                                        "operation_id": args.operation,
                                    }
                                ),
                                flush=True,
                            )
                            threading.Event().wait()

                    coordinator.progress = progress
                    receipts = (await coordinator.commit(review, args.operation),)
                else:
                    receipts = await coordinator.recover()
                snapshot = authority.verify_current()
                blocked = []
                for item in snapshot["installations"]:
                    try:
                        token = owner.reserve_launch(
                            "probe",
                            item["installation_id"],
                            None,
                            item["revision_digest"],
                        )
                    except PermissionError:
                        blocked.append(item["installation_id"])
                    else:
                        owner.settle_process(token, confirmed=True)
                return {
                    "receipts": [asdict(item) for item in receipts],
                    "installations": list(
                        registry.list_installations(limit=50, offset=0)
                    ),
                    "marker_generation": authority.load_marker().generation,
                    "blocked": blocked,
                    "processes": list(owner.list_processes(limit=50, offset=0)),
                    "module": coordinator_module.__file__,
                    "profile": str(profile),
                    "config": str(config),
                    "pid": os.getpid(),
                }

            result.set_result(loop.run_until_complete(run()))
        except BaseException as error:  # noqa: BLE001
            result.set_exception(error)
        finally:
            if registry is not None:
                registry.close()
            owner.close()
            loop.close()

    def guarded_worker():
        try:
            worker()
        except BaseException as error:  # noqa: BLE001
            if not result.done():
                result.set_exception(error)

    thread = threading.Thread(target=guarded_worker, name="plugin-crash-storage")
    thread.start()
    thread.join()
    print(json.dumps(result.result()), flush=True)


if __name__ == "__main__":
    main()
