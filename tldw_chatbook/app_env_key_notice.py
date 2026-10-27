"""TldwCli's env-key first-launch notice: ``EnvKeyNoticeMixin``.

TASK-21147 (UAT E-1) said once, at launch, that exported provider keys
skipped the setup wizard. TASK-34100.2 (AC#7, AC#11) made that notice honest
and ordered it: its copy comes from the shared readiness verdict
(``build_env_key_first_launch_notice``), and it is held until the model-list
consent question is answered, so it never appears on top of that modal or
expires behind it. Moved out of ``app.py`` whole (TASK-34100.2 pass 2) so the
app module stays inside its size ratchet; new app-level glue belongs in a
mixin like this one, not in ``app.py``.

Every method is called class-qualified by tests with a stub ``self``
(``TldwCli._show_pending_env_key_notice(fake)``), so none of them uses
``super()``.
"""

from __future__ import annotations

from loguru import logger

from tldw_chatbook.app_feature_glue import FeatureGlueMixin


class EnvKeyNoticeMixin:
    """Hold, show and record the one-time env-key launch notice."""

    def _show_pending_env_key_notice(self) -> None:
        """Show the held env-key notice once, then record it (TASK-34100.2)."""
        notice = getattr(self, "_pending_env_key_notice", None)
        self._pending_env_key_notice = None
        if notice is None:
            return
        self.notify(
            notice.message,
            title=notice.title,
            timeout=10 if notice.ready else 20,
            severity="information" if notice.ready else "warning",
        )
        self._env_key_notice_message = notice.message  # F10: 'Use X' dismisses it
        self._persist_env_key_notice_flag()

    def _schedule_catalog_refresh_then_env_key_notice(self) -> bool:
        """AC#11: schedule the catalog pass, then show the held env-key notice
        unless that pass opened the consent question. One call at mount, so
        the order cannot be swapped there (review round 2, R2-5).

        Returns:
            Whether the model-list consent question is still open (its answer
            then flushes the notice, ``_handle_model_catalog_consent``).
        """
        self._schedule_startup_model_catalog_refresh()
        consent_pending = getattr(
            self, "_startup_model_catalog_consent_required", False
        )
        if self.is_headless or not consent_pending:  # headless: no modal to wait on
            self._show_pending_env_key_notice()
        return consent_pending

    async def _handle_model_catalog_consent(self, allowed: bool | None) -> None:
        """AC#11: the consent decision comes first, then the env-key notice."""
        try:
            await FeatureGlueMixin._handle_model_catalog_consent(self, allowed)
        finally:
            EnvKeyNoticeMixin._show_pending_env_key_notice(self)

    def _persist_env_key_notice_flag(self) -> None:
        """Record the one-time env-key notice (TASK-21147, UAT E-1)."""

        from tldw_chatbook.UI.Wizards.first_run_setup_state import (
            ENV_KEY_NOTICE_KEY,
            WIZARD_STATE_SECTION,
        )

        app_config = self.app_config
        if isinstance(app_config, dict):
            app_config.setdefault(WIZARD_STATE_SECTION, {})[ENV_KEY_NOTICE_KEY] = True

        def _write() -> None:
            from tldw_chatbook.config import save_settings_to_cli_config

            try:
                saved = save_settings_to_cli_config(
                    {WIZARD_STATE_SECTION: {ENV_KEY_NOTICE_KEY: True}}
                )
            except Exception as exc:
                logger.warning(
                    "Failed to persist env-key notice flag "
                    f"(category=persistence, error_type={type(exc).__name__})"
                )
                return
            if not saved:
                logger.warning(
                    "Failed to persist env-key notice flag "
                    "(category=persistence, error_type=save_returned_false)"
                )

        self.run_worker(
            _write,
            thread=True,
            group="first-run-env-key-notice-flag",
            exit_on_error=False,
        )
