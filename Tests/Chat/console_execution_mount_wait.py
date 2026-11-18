"""Wait for the actual selected surface inside the caller's existing deadline."""

import asyncio


async def wait_for_selected_composer(app) -> None:
    """Observe mount and reconciliation without an unrelated all-widget barrier."""
    while True:
        if app._exception is not None:
            raise app._exception
        screen = app.screen
        if (
            screen.is_mounted and screen.is_attached
            and getattr(screen, "_console_process_surface_reconciled", False)
            and any(
                composer.is_mounted and composer.is_attached
                for composer in screen.query("#console-native-composer")
            )
        ):
            return
        await asyncio.sleep(0.01)
