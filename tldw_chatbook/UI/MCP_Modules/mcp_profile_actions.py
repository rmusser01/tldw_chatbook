"""Current MCP profile worker bodies; the workbench retains its live state."""

from typing import Any


async def _delete_local_profile(self, server_key: str, profile_id: str) -> None:
    from . import mcp_workbench as _owner

    try:
        if (self._catalog_records.get(profile_id) or {}).get(
            "plugin_owner"
        ) is not None:
            self.app.notify("Manage this owned profile in Plugins.", severity="warning")
            return
        service = self._service()
        if service is None:
            return
        try:
            await service.delete_local_profile(profile_id)
        except Exception as exc:
            _owner.logger.warning(
                "{}", _owner._safe_diagnostic_message("MCP profile delete failed", exc)
            )
            self.app.notify(_owner._toast(f"Delete failed: {exc}"), severity="error")
            return
        self.app.notify(_owner._toast(f"Deleted {profile_id}."))
        if self._selected_server_key == server_key:
            self._selected_server_key = None
        self._snapshots = await self._collect_snapshots()
        await self._sync_children()
    finally:
        self._profile_delete_in_flight = False


async def _save_local_profile(
    self,
    payload: dict[str, Any],
    warning: str | None = None,
    *,
    connect_after: bool = False,
) -> None:
    """Run one profile save; on success, also re-surface the form's args
    secret-lint `warning` as a toast (I4 follow-up). The in-form
    `#mcp-form-args-warning` Static only survives FAILED saves -- the
    success path's `hide_form()` below unmounts the whole form
    sub-second after the warning rendered, so without this toast the
    user would never see it on exactly the path where the secret
    actually got persisted into a profile's args.

    Wave C (F7a): `connect_after=True` (the form's "Save and connect")
    dispatches the connect lifecycle for the just-saved profile once
    the save and resync land -- the saved->connected journey used to
    require finding the new row and its inspector Connect action.
    A connect failure surfaces through the lifecycle's own
    notification/readiness path (e.g. a missing env placeholder).
    """
    from . import mcp_workbench as _owner

    try:
        service = self._service()
        if service is None:
            return
        try:
            if (self._catalog_records.get(payload.get("profile_id")) or {}).get(
                "plugin_owner"
            ) is not None:
                raise ValueError("plugin_owned_profile_use_plugins")
            await service.save_local_profile(payload)
        except ValueError as exc:
            # Store-validation copy is user-ready. If the form is gone
            # (e.g. cancelled while the save was in flight), the failure
            # must still surface -- never vanish silently.
            form = self._form_or_none()
            if form is not None:
                form.show_error(str(exc))
            else:
                self.app.notify(_owner._toast(str(exc)), severity="error")
            return
        except Exception as exc:
            _owner.logger.warning(
                "{}", _owner._safe_diagnostic_message("MCP profile save failed", exc)
            )
            # Route through show_error when possible: it also re-enables
            # the form's Save button (disabled at submit) for a retry.
            form = self._form_or_none()
            if form is not None:
                form.show_error(f"Save failed: {exc}")
            else:
                self.app.notify(_owner._toast(f"Save failed: {exc}"), severity="error")
            return
        canvas = self.query_one(_owner.MCPServersMode)
        await canvas.hide_form()
        self.app.notify(_owner._toast(f"Saved {payload.get('profile_id')}."))
        if warning:
            self.app.notify(warning, severity="warning")
        self._snapshots = await self._collect_snapshots()
        await self._sync_children()
        if connect_after and payload.get("profile_id"):
            profile_id = str(payload["profile_id"])
            self._start_lifecycle(f"local:{profile_id}", profile_id, "connect")
    finally:
        self._profile_save_in_flight = False
