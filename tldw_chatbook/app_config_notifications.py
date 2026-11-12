"""Existing config warning presentation, loaded only for a detected warning."""

from __future__ import annotations

from typing import Any


def notify_config_load_failure(self, failure: Any) -> None:
    """Warn (never block) when boot's config load fell back to defaults.

    TASK-13157: a config.toml that fails to parse was previously a
    completely silent failure -- `load_settings()`/`load_cli_config_and_
    ensure_existence()` both return bare in-memory defaults with no
    signal, which a live-verification incident showed can silently
    resolve the data directory to the `default_user` profile instead of
    the configured one, with no error, toast, or log line a normal user
    would ever see. `self._config_load_failure` was snapshotted in
    `__init__` (before the UI existed to notify through); this surfaces
    it once the initial screen is up, naming the exact file and parse
    error so the user knows their saved settings are NOT the ones
    currently in effect. `timeout=None` only reaches Textual's own
    5-second default (`App.NOTIFICATION_TIMEOUT`), not "persistent", so
    this passes an explicit long timeout instead -- this is not a
    transient event and must not be missed the way the silent fallback
    it replaces was.
    """
    self.notify(
        f"Your configuration file could not be parsed and was NOT used "
        f"this session -- running on built-in defaults instead (this may "
        f"include the wrong user profile). File: {failure.path}  "
        f"Error: {failure.message}",
        title="Config file failed to load",
        severity="error",
        timeout=60,
        markup=False,
    )


def notify_config_schema_conflict(self, conflict: Any) -> None:
    """Warn (never block) when the config is from a newer app version.

    TASK-26040 AC#5: a config carrying a schema version newer than this
    build understands is served untouched rather than migrated (a
    downgrade could silently drop keys). This surfaces the detected
    conflict once the UI is up so the user knows why newer settings may
    not take effect, mirroring `_maybe_warn_config_load_failure`.
    """
    self.notify(
        f"Your configuration was written by a newer version of this "
        f"application and was left unchanged (not migrated). Some newer "
        f"settings may not take effect until you upgrade. {conflict}",
        title="Config is from a newer version",
        severity="warning",
        timeout=60,
        markup=False,
    )
