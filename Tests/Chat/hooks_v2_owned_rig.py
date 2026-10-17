"""Dispose the runtime and databases owned by hook lifecycle test rigs."""

import asyncio
from contextlib import asynccontextmanager


@asynccontextmanager
async def _owned_hook_rig_resources(controller, runs_db, db):
    try:
        yield
    finally:
        try:
            await controller.shutdown()
        finally:
            try:
                runs_db.close()
            finally:
                from Tests.conftest import _close_database_instance

                _close_database_instance(db)


@asynccontextmanager
async def disposed_hook_rig(runtime, controller, runs_db, db):
    """Settle runtime work before case assertions and owned resource release."""
    async with _owned_hook_rig_resources(controller, runs_db, db):
        await runtime.dispose()
        yield


@asynccontextmanager
async def disposed_mounted_hook_rig(app, runtime, controller, runs_db, db):
    """Also settle the mounted app's producers and its additional owners."""
    # Current mounted consumers retain their app/database/lock inventory in
    # _close_mounted_app_owners; settle producers here before their assertions.
    async with _owned_hook_rig_resources(controller, runs_db, db):
        try:
            await app._shutdown_app_owned_lifecycles()
        finally:
            await runtime.dispose()
        yield
