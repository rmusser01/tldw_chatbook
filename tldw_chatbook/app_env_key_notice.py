"""TldwCli's env-key first-launch notice bodies: ``EnvKeyNoticeMixin``.

TASK-21147 (UAT E-1) said once, at launch, that exported provider keys
skipped the setup wizard. TASK-34100.2 (AC#7, AC#11) made that notice honest
and ordered it: its copy comes from the shared readiness verdict
(``build_env_key_first_launch_notice``), and it is held until the model-list
consent question is answered, so it never appears on top of that modal or
expires behind it. Moved out of ``app.py`` (TASK-34100.2 pass 2) so the app
module stays inside its size ratchet.

Not a ``TldwCli`` base (PR #3517's round-24 port, ADR-097): ``app.py`` keeps
the mount-time scheduling and the consent wrapper, its show and record stubs
import this module only once a notice is held or shown, and the Get started
card imports it when an offer commits, so a normal launch never loads it.
Every method is called class-qualified
(``EnvKeyNoticeMixin._show_pending_env_key_notice(app)``), so none of them
uses ``super()``.
"""

from __future__ import annotations

from loguru import logger


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
            markup=False,  # TASK-1513: env var and model names are text, never markup
        )
        self._env_key_notice_message = notice.message  # F10: 'Use X' dismisses it
        self._persist_env_key_notice_flag()

    def dismiss_env_key_notice(self) -> None:
        """Remove the env-key launch toast, and only that toast (review F10).

        Called when a Get started card offer commits, so the toast stops
        naming the old unkeyed default. Textual has no public way to remove
        one toast, so this reads the app's notification collection and is a
        no-op if that ever changes; a stale toast never fails the commit.
        """
        message = getattr(self, "_env_key_notice_message", None)
        notifications = getattr(self, "_notifications", None)
        unnotify = getattr(self, "_unnotify", None)
        if not message or notifications is None or not callable(unnotify):
            return
        try:
            for notification in [n for n in notifications if n.message == message]:
                unnotify(notification)
        except Exception:  # noqa: BLE001 - a stale toast never fails the commit
            logger.debug("env_key_notice_dismiss_failed")

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
