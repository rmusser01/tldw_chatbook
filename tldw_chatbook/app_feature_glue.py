"""TldwCli's per-feature glue: ``FeatureGlueMixin``.

Moved verbatim from ``app.py`` (TASK-33011, PR-I). App-level message forwarders and the small
per-feature seams that used to grow ``app.py`` with every feature:

- reminders and automation (``on_reminder_dispatched`` and the automation-definition handler);
- the subscription-items and ChaChaNotes-messages FTS backfills;
- server-parity state repositories and the server-notification event scope;
- the persona buddy overlay and its forwarders (``on_persona_buddy_changed``,
  ``on_base_app_screen_contents_rebuilt``, ``on_character_card_changed``);
- model-catalog refresh, startup scheduling and the consent modal.

Name-based ``on_*`` handlers dispatch from a mixin like this one; the ``@on``-decorated
``on_model_catalog_refreshed`` stays on ``TldwCli``, since Textual only dispatches decorated
handlers defined on Textual classes. New app-level glue of this kind belongs here, not in
``app.py``.

Patch the names this code reads HERE: the bodies resolve free names through this module's
globals, so a patch on ``tldw_chatbook.app`` alone no longer reaches them. Where ``app.py`` still
reads the same name, patch both modules (``Tests/app_module_patches.py``).
``Tests/Architecture/test_app_extracted_patch_targets.py`` fails on an app-module patch that can
only have been meant for code that moved out.
"""

import asyncio
import os
from typing import TYPE_CHECKING, Any, Mapping  # noqa: UP035

from loguru import logger

from tldw_chatbook.app_service_wiring import TldwCli  # class proxy (see its docstring)
from tldw_chatbook.config import CLI_APP_CLIENT_ID, get_chachanotes_db_lazy
from tldw_chatbook.Constants import MODEL_CATALOG_REFRESH_WORKER_GROUP
from tldw_chatbook.DB.Subscriptions_DB import SubscriptionsDB
from tldw_chatbook.LLM_Provider_Catalog.model_auto_refresh import ModelCatalogRefreshed
from tldw_chatbook.Notifications import EventStateRepository
from tldw_chatbook.runtime_policy.server_event_scope import (
    event_principal_id_from_active_context,
)
from tldw_chatbook.runtime_policy.server_parity_state import (
    ServerParityStateRepositories,
    build_server_parity_state_repositories,
)
from tldw_chatbook.Sync_Interop import SyncStateRepository

from .config import (
    get_cli_setting,
    get_subscriptions_db_path,
    get_user_data_dir,
    load_settings,
)
from .Scheduling.constants import HANDLER_TIMEOUT_SECONDS

if TYPE_CHECKING:  # pragma: no cover - typing only
    from tldw_chatbook.LLM_Provider_Catalog.model_discovery_disk_cache import (
        ModelCatalogDiskStore,
    )


_SETUP_STARTUP_NETWORKING_ACTIONS = frozenset({"offer", "prompt", "home"})

def setup_owns_startup_networking(
    app_config: Mapping[str, Any], environ: Mapping[str, str]
) -> bool:
    """Return whether first-run setup owns automatic networking this startup."""
    from tldw_chatbook.UI.Wizards.first_run_setup_state import (
        setup_recovery_action,
    )

    return (
        setup_recovery_action(app_config, environ) in _SETUP_STARTUP_NETWORKING_ACTIONS
    )


class FeatureGlueMixin:
    """``TldwCli`` members moved from ``app.py`` (TASK-33011); see the module docstring."""

    def _get_automation_definition_handler(self) -> Any:
        """Lazily construct and memoize the automation-definition handler.

        ADR-097 (boot-census ratchet): `AutomationDefinitionHandler`'s
        import chain (the handler module + `schedule_compute` +
        `slot_keys`) stays OFF the boot path -- built on first use, then
        cached on `self` so every later call (scheduled dispatch via
        `_dispatch_automation_definition` above, and manual dispatch via
        `SchedulingService.run_automation_now`'s injected
        `automation_handler_getter`) reuses the SAME instance. The
        overlap-claim guard (`_claimed`/`_pending` on the handler) only
        works across calls if it is the same object each time; a fresh
        handler per call would silently defeat it -- and would let a
        manual run race a scheduled one for the same definition.
        """
        handler = getattr(self, "_automation_definition_handler", None)
        if handler is None:
            from .Scheduling.scheduler.handlers.automation_handler import (
                AutomationDefinitionHandler,
            )

            handler = AutomationDefinitionHandler(
                db=self.scheduling_service.db,
                app_getter=lambda: self,
                dispatch_service=self.notification_dispatch_service,
                handler_timeout_seconds=get_cli_setting(
                    "scheduling", "handler_timeout_seconds", HANDLER_TIMEOUT_SECONDS
                ),
                # task-18937/Finding D precedent (`on_queue_changed` above):
                # a schedule advance written here would otherwise sit
                # unseen by the live queue until its periodic ~30-minute
                # reload. Same lazy-getattr discipline -- `scheduler_loop`
                # is constructed after `self.scheduling_service`, so this
                # lambda must not resolve it eagerly.
                on_queue_changed=lambda: getattr(self, "scheduler_loop", None)
                and self.scheduler_loop.request_reload(),
            )
            self._automation_definition_handler = handler
        return handler

    def _post_reminder_dispatched(self, task_id: str) -> None:
        """`SchedulerLoop.on_reminder_dispatched` callback (finding 3a).

        Lazy import: `Scheduling.events` pulls in `Widgets.
        detail_value_row` (ADR-097 boot-census ratchet), and this only
        runs once a reminder has actually fired -- long after boot.
        Posting to `self` (the App) is the only route available: the
        loop is UI-agnostic and holds no screen reference, so
        `on_reminder_dispatched` below relays this to whichever
        `SchedulesWorkbench` is currently on the screen stack, if any.
        """
        from .Scheduling.events import ReminderDispatched

        self.post_message(ReminderDispatched(task_id))

    def on_reminder_dispatched(self, message: Any) -> None:
        """Relay a scheduler-fired reminder into the live Schedules
        workbench, if one is mounted (finding 3a).

        A purely local scheduler fire has no other route to the UI --
        the only existing push path is the server SSE observer, which
        requires a server. Same `screen_stack` scan idiom as
        `_mounted_chat_screen` above; a full `_request_tasks_refresh()`
        rather than a targeted repaint, since the fired row's own
        bucket/next-run text both change together.
        """
        from .UI.Screens.scheduling.schedules_workbench import SchedulesWorkbench

        for screen in reversed(tuple(getattr(self, "screen_stack", ()))):
            if isinstance(screen, SchedulesWorkbench):
                screen._request_tasks_refresh(refresh_definitions=False)
                return

    def _backfill_subscription_items_fts(self) -> None:
        from tldw_chatbook.Backup_Recovery.activation import execution_scope

        with execution_scope(
            ("db.subscriptions",), get_subscriptions_db_path()
        ) as allowed:
            if allowed:
                TldwCli._backfill_subscription_items_fts_owned(self)

    def _backfill_subscription_items_fts_owned(self) -> None:
        """Worker body: index subscription_items rows that predate the FTS
        index (task-688). Started from ``on_mount`` via
        ``run_worker(thread=True)`` so a large backlog never blocks app
        startup or screen mount.

        Uses the app's single ``SubscriptionsDB`` (task-15463). It used to
        construct its own, on the theory that a thread-local connection
        cannot be shared -- but thread-locality is exactly what makes sharing
        the INSTANCE safe: this worker thread gets its own connection from it.
        Constructing a second instance re-ran ``_initialize_schema`` -- a
        ~52-statement ``executescript`` plus migrations, measured at 238 ms --
        on a worker thread *while the app was already serving screens*, and
        any connection opened during that window cached a schema view without
        the tables it was rewriting. That is not theoretical: with per-call
        database construction it showed up as the intermittent
        ``OperationalError: no such table: subscription_items`` documented in
        ``Tests/UI/test_watchlists_inspector.py``, self-healing on retry
        because the next call built a new connection; against a held instance
        the poisoned connection survives, and the write that lands on it just
        fails. One instance, one schema initialization, no window.

        ``close()`` below stays, and what it does is worth stating exactly.
        ``SubscriptionsDB.close`` closes only the CALLING thread's connection
        and clears that thread's slot. This body runs on a **pooled** thread
        (Textual's thread workers run on asyncio's default executor, which is
        shared with every ``asyncio.to_thread`` hop in the app), so the
        connection it closes belongs to a pool thread that will later serve
        other watchlists work on this same shared instance. That is safe for
        exactly one reason: the ``conn`` property re-opens lazily, so the next
        hop scheduled onto that thread gets a fresh connection instead of a
        closed one. It is not safe to "improve" this into a close of the
        instance itself.

        TASK-22215: the driver paces itself between chunks (the TASK-22200
        treatment, now shared) and this worker hands it the Textual worker's
        cancellation flag -- pacing makes the run longer, and a thread worker
        that never polls ``is_cancelled`` would make shutdown wait out every
        remaining pause. Stopping is safe: the resume frontier lives in the
        database.
        """
        from textual.worker import NoActiveWorker, get_current_worker

        from tldw_chatbook.Subscriptions.fts_backfill import (
            FTSBackfillError,
            backfill_subscription_items_fts,
        )

        try:
            worker = get_current_worker()
        except NoActiveWorker:
            worker = None  # direct calls in tests/harnesses run un-cancellable
        should_abort = (lambda: worker.is_cancelled) if worker is not None else None

        db = None
        db_path = get_subscriptions_db_path()
        try:
            db = getattr(self, "subscriptions_db", None)
            if db is None:
                # Only a harness that skipped service wiring gets here.
                db = SubscriptionsDB(db_path, CLI_APP_CLIENT_ID)
            backfill_subscription_items_fts(db, should_abort=should_abort)
        except FTSBackfillError as exc:
            logger.opt(exception=True).error(
                "Subscription items FTS backfill failed for database {} "
                "after indexing {} row(s) this run; some pre-existing "
                "items may remain unsearchable until the app is restarted.",
                db_path,
                exc.rows_indexed,
            )
        except Exception:
            logger.opt(exception=True).error(
                "Subscription items FTS backfill failed for database {}; "
                "some pre-existing items may remain unsearchable until the "
                "app is restarted.",
                db_path,
            )
        finally:
            if db is not None:
                try:
                    db.close()
                except Exception:
                    logger.opt(exception=True).warning(
                        "Failed to close SubscriptionsDB {} after FTS backfill.",
                        db_path,
                    )

    def _backfill_chachanotes_messages_fts(self) -> None:
        """Worker body: reinsert messages the v45->v46 migration no longer
        indexes inline (task-21100). Started from ``on_mount`` via
        ``run_worker(thread=True)`` so an upgraded profile's index rebuild
        never blocks boot or first paint; each chunk commits in its own
        transaction, so a kill at any point leaves a consistent, resumable
        index (state = ``messages_fts_docsize`` membership, in the DB
        itself).

        Uses the app's shared ``CharactersRAGDB`` singleton -- thread-local
        connections are exactly what makes that safe from a worker thread
        (see ``_backfill_subscription_items_fts`` for the incident that
        taught this). Unlike that worker, no ``close()`` here: pooled threads
        serve ChaChaNotes work constantly, and the thread-local connection
        this run opens is the same one later hops on this thread reuse.

        On an up-to-date database the loop's first chunk finds nothing and
        the whole call is one indexed scan -- cheap, and it doubles as
        self-healing for any run interrupted before completion.

        task-22200: the driver paces itself (inter-chunk sleep + backoff on
        lock-queue timeouts) so this run yields the write lock to foreground
        UI writes instead of convoying against them for the whole first
        post-upgrade session. The worker's own cancellation flag is passed
        through as ``should_abort`` -- pacing makes the run longer, and a
        thread worker that never polls ``is_cancelled`` would make shutdown
        wait out every remaining pause; the driver polls it between chunks
        and inside every sleep, and stopping is safe because the resume
        frontier lives in the database.
        """
        from textual.worker import NoActiveWorker, get_current_worker

        from tldw_chatbook.DB.chachanotes_fts_backfill import (
            ChaChaNotesFTSBackfillError,
            backfill_chachanotes_messages_fts,
        )

        try:
            worker = get_current_worker()
        except NoActiveWorker:
            worker = None  # direct calls in tests/harnesses run un-cancellable
        should_abort = (lambda: worker.is_cancelled) if worker is not None else None

        try:
            db = get_chachanotes_db_lazy()
            if db is None:
                logger.debug(
                    "ChaChaNotes messages FTS backfill skipped: no database instance."
                )
                return
            from .DB.base_db import operation_owned_connection

            with operation_owned_connection(db):
                backfill_chachanotes_messages_fts(db, should_abort=should_abort)
        except ChaChaNotesFTSBackfillError as exc:
            logger.opt(exception=True).error(
                "ChaChaNotes messages FTS backfill failed after indexing {} "
                "row(s) this run; older messages may be missing from search "
                "until the next app start resumes it.",
                exc.rows_indexed,
            )
        except Exception:
            logger.opt(exception=True).error(
                "ChaChaNotes messages FTS backfill failed; older messages may "
                "be missing from search until the next app start resumes it."
            )

    def _wire_server_parity_state_repositories(self) -> None:
        try:
            self.server_parity_state = build_server_parity_state_repositories(
                data_dir=get_user_data_dir(),
                client_id=CLI_APP_CLIENT_ID,
                local_notifications_db=self.client_notifications_db,
            )
        except Exception as exc:
            logger.opt(exception=True).error(
                "Failed to initialize server parity state repositories; using in-memory stores: {}",
                exc,
            )
            self.server_parity_state = ServerParityStateRepositories(
                local_notifications_db=self.client_notifications_db,
                event_state_repository=EventStateRepository(
                    ":memory:", CLI_APP_CLIENT_ID
                ),
                sync_state_repository=SyncStateRepository(
                    ":memory:", CLI_APP_CLIENT_ID
                ),
            )
        self.event_state_repository = self.server_parity_state.event_state_repository
        self.sync_state_repository = self.server_parity_state.sync_state_repository

    def _server_notification_event_scope(self) -> dict[str, str | None]:
        runtime_policy = getattr(self, "runtime_policy", None)
        runtime_state = runtime_policy.state if runtime_policy is not None else None
        active_server_id = getattr(runtime_state, "active_server_id", None)
        authenticated_principal_id = None
        server_context_provider = getattr(self, "server_context_provider", None)
        get_active_context = getattr(
            server_context_provider, "get_active_context", None
        )
        if callable(get_active_context):
            try:
                authenticated_principal_id = event_principal_id_from_active_context(
                    get_active_context()
                )
            except Exception:
                authenticated_principal_id = None
        return {
            "server_profile_id": str(active_server_id) if active_server_id else None,
            "authenticated_principal_id": authenticated_principal_id,
            "stream_instance_id": "global",
        }

    @staticmethod
    def _persona_buddy_authority(controller: Any, snapshot: Any) -> tuple[Any, ...]:
        """Return the exact app-lifetime authority for one visual decision."""

        return (
            id(controller),
            snapshot.generation,
            snapshot.selection,
            snapshot.preferences_generation,
            snapshot.profile_generation,
        )

    def is_persona_buddy_confirmed_unavailable(
        self, controller: Any, snapshot: Any
    ) -> bool:
        """Query and clear the app-owned unavailable marker by exact authority."""

        authority = self._persona_buddy_authority(controller, snapshot)
        marker = getattr(self, "_persona_buddy_unavailable_authority", None)
        if marker is not None and marker != authority:
            self._persona_buddy_unavailable_authority = None
            return False
        return marker == authority

    def confirm_persona_buddy_unavailable(
        self,
        *,
        screen: Any,
        view: Any,
        view_generation: int,
        controller: Any,
        snapshot: Any,
        visual: Any,
    ) -> bool:
        """Publish unavailable only for the exact current app/screen/view authority."""

        current_controller = getattr(self, "persona_buddy_controller", None)
        try:
            current_screen = self.screen
        except Exception:
            return False
        if (
            controller is not current_controller
            or current_screen is not screen
            or not screen.is_attached
            or getattr(self, "_persona_buddy_overlay", None) is None
            or not self._persona_buddy_overlay.is_current(view)
            or self._persona_buddy_overlay.generation != view_generation
            or not view.is_attached
        ):
            return False
        current = controller.snapshot()
        if (
            self._persona_buddy_authority(controller, current)
            != self._persona_buddy_authority(controller, snapshot)
            or current.visual is not visual
            or visual is None
            or visual.available
        ):
            return False
        self._persona_buddy_unavailable_authority = self._persona_buddy_authority(
            controller, current
        )
        return True

    async def reconcile_persona_buddy_view(self) -> bool:
        """Reconcile the active screen and report whether its Buddy is absent."""

        from .UI.Navigation.base_app_screen import BaseAppScreen

        try:
            screen = self.screen
        except Exception:
            return False
        if not isinstance(screen, BaseAppScreen) or not screen.is_active:
            return False
        owner = getattr(self, "_persona_buddy_overlay", None)
        if owner is None:
            from .UI.Navigation.persona_buddy_overlay import PersonaBuddyOverlay

            owner = self._persona_buddy_overlay = PersonaBuddyOverlay(self)
        return await owner.reconcile()

    def _start_persona_buddy_overlay(self) -> None:
        """Subscribe once to native screen changes and reconcile after mount."""
        if getattr(self, "_persona_buddy_overlay_started", False):
            return
        self._persona_buddy_overlay_started = True
        self.screen_change_signal.subscribe(self, self._schedule_persona_buddy_overlay)
        self.call_after_refresh(self._schedule_persona_buddy_overlay)

    def _notify_persona_buddy_changed(self) -> None:
        """Post a content-free notification safely from any controller thread."""
        from .UI.Navigation.persona_buddy_overlay import PersonaBuddyChanged

        self.post_message(PersonaBuddyChanged())

    def on_persona_buddy_changed(self, message: Any) -> None:
        """Reconcile the latest controller generation on the app event loop."""
        self._schedule_persona_buddy_overlay()

    def on_base_app_screen_contents_rebuilt(self, message: Any) -> None:
        """Restore app-owned presentation after the active screen rebuilds."""
        if self.screen_stack and message.screen is self.screen:
            self._schedule_persona_buddy_overlay()

    def on_character_card_changed(self, message: Any) -> None:
        """Forward a Console character-card save to the active Personas screen.

        Textual delivers an App-posted message to App handlers only (it
        never bubbles down into a Screen's own handler -- see
        ``forward_model_catalog_refreshed`` for the identical constraint),
        so the screen's handler is called directly instead. Walks the full
        screen stack, not just ``self.screen``, so a modal sitting on top of
        Personas (e.g. an unsaved-changes confirm dialog) does not silently
        drop the notification (fix round 1, review point 4).
        ``exit_on_error=False``: a background notification must never crash
        the app (review point 1).
        """
        from tldw_chatbook.UI.Screens.personas_screen import PersonasScreen

        for screen in reversed(tuple(getattr(self, "screen_stack", ()))):
            if isinstance(screen, PersonasScreen):
                screen.run_worker(
                    screen._on_character_card_changed(message),
                    group="personas-character-changed",
                    exclusive=True,
                    exit_on_error=False,
                )
                return

    def _schedule_persona_buddy_overlay(self, _screen: Any = None) -> None:
        """Skip disabled work and coalesce presentation updates on the app."""
        if not self.screen_stack:
            return
        owner = getattr(self, "_persona_buddy_overlay", None)
        if owner is not None and owner.closed:
            return
        controller = getattr(self, "persona_buddy_controller", None)
        snapshot = controller.snapshot() if controller is not None else None
        if (snapshot is None or not snapshot.enabled) and (
            owner is None or owner.view is None
        ):
            sync = getattr(self.screen, "sync_persona_buddy_reconciled_state", None)
            if callable(sync):
                sync()
            return
        if owner is None:
            from .UI.Navigation.persona_buddy_overlay import PersonaBuddyOverlay

            owner = self._persona_buddy_overlay = PersonaBuddyOverlay(self)
        owner.request()

    def _init_model_catalog_disk_store(self) -> "ModelCatalogDiskStore | None":
        """Build the disk-backed model catalog cache for startup (ADR-020).

        Returns None (with a log line) when the cache path cannot be resolved,
        fails validation against the user data dir, or the on-disk cache cannot
        be loaded; startup continues without persistence in those cases.
        """
        from tldw_chatbook.LLM_Provider_Catalog.model_discovery_disk_cache import (
            ModelCatalogDiskStore,
        )
        from tldw_chatbook.Utils.path_validation import get_safe_relative_path

        try:
            user_data_dir = get_user_data_dir()
            cache_path = user_data_dir / "model_catalog_cache.json"
        except Exception as exc:
            logger.error(
                f"Failed to resolve model catalog cache path: {type(exc).__name__}"
            )
            return None
        # get_safe_relative_path (not is_safe_path): the default data dir lives
        # under ~/.local, which validate_path's hidden-component rule rejects.
        if get_safe_relative_path(cache_path, user_data_dir) is None:
            logger.warning(
                f"Ignoring model catalog cache outside the user data dir: {cache_path}"
            )
            return None
        try:
            store = ModelCatalogDiskStore(cache_path)
            store.load_into(self.local_llm_provider_catalog_service.discovery_cache)
        except Exception as exc:
            # No traceback: the log file sink runs with diagnose=True, which
            # would dump frame locals (including the app's config) into the log.
            logger.error(
                f"Failed to load model catalog disk cache {cache_path}: "
                f"{type(exc).__name__}"
            )
            return None
        return store

    async def _refresh_model_catalogs(self) -> None:
        from tldw_chatbook.Backup_Recovery.activation import execution_scope

        with execution_scope(("config",)) as allowed:
            if allowed:
                await TldwCli._refresh_model_catalogs_owned(self)

    async def _refresh_model_catalogs_owned(self) -> None:
        """ADR-020 startup auto-refresh; never blocks or crashes startup."""
        try:
            from tldw_chatbook.LLM_Provider_Catalog.model_auto_refresh import (
                format_refresh_notification,
            )
            from tldw_chatbook.LLM_Provider_Catalog.model_catalog_settings import (
                AUTO_REFRESH_PROVIDER_LIST_KEYS,
                load_model_catalog_settings,
            )

            if self.model_catalog_disk_store is None:
                return
            catalog_settings = load_model_catalog_settings(load_settings())
            if not catalog_settings.auto_refresh_enabled:
                return
            if not catalog_settings.refresh_consent_recorded:
                # ADR-020 amendment: the startup check is confirm-first.
                # Scheduling-side consent gate normally intercepts this
                # before the worker spawns; keep the check here so the
                # refresh never runs unconsented by any other path.
                return
            report = await self.local_llm_provider_catalog_service.refresh_stale_configured_providers(
                catalog_settings=catalog_settings,
                disk_store=self.model_catalog_disk_store,
                on_config_saved=self._init_providers_models,
            )
            refreshed = {
                outcome.provider_list_key
                for outcome in report.outcomes
                if outcome.status in {"refreshed", "baseline"}
            }
            if refreshed:
                self.post_message(ModelCatalogRefreshed(providers=refreshed))
            message = format_refresh_notification(report)
            has_failure = report.disk_write_failed or any(
                outcome.status == "failed" or outcome.write_failed
                for outcome in report.outcomes
            )
            # TASK-34100.5 AC#11: the pass setup released is news only if it failed.
            if message and (has_failure or not getattr(self, "_model_catalog_notice_quiet", False)):
                severity = "warning" if has_failure else "information"
                self.notify(message, title="Model catalog", severity=severity)
        except Exception as exc:
            # No traceback: the log file sink runs with diagnose=True, which
            # would dump frame locals (potentially API keys) into the log file.
            logger.error(
                "Model catalog auto-refresh failed "
                f"({', '.join(AUTO_REFRESH_PROVIDER_LIST_KEYS)}): "
                f"{type(exc).__name__}"
            )

    def _schedule_startup_model_catalog_refresh(
        self,
        *,
        after_setup_completion: bool = False,
        environ: Mapping[str, str] | None = None,
    ) -> bool:
        """Schedule the automatic catalog pass once when setup releases it.

        ADR-020 amendment (confirm-first): when the user has never answered
        the consent question, a modal is shown instead of the refresh; the
        refresh itself is only scheduled from the consent callback.
        """
        from tldw_chatbook.Backup_Recovery.activation import execution_allowed

        if not execution_allowed(("config",)):
            return False
        if getattr(self, "_startup_model_catalog_refresh_scheduled", False):
            return False
        if not after_setup_completion and setup_owns_startup_networking(
            self.app_config,
            os.environ if environ is None else environ,
        ):
            return False

        try:
            from tldw_chatbook.LLM_Provider_Catalog.model_catalog_settings import (
                load_model_catalog_settings,
            )

            catalog_settings = load_model_catalog_settings(load_settings())
        except Exception as exc:
            logger.error(
                "Failed to load model catalog settings for startup refresh "
                f"scheduling (after_setup_completion={after_setup_completion}): "
                f"{type(exc).__name__}"
            )
            return False
        if catalog_settings.auto_refresh_enabled and (
            not catalog_settings.refresh_consent_recorded
        ):
            self._startup_model_catalog_refresh_scheduled = True
            self._startup_model_catalog_consent_required = True
            self.call_after_refresh(self._push_model_catalog_consent_modal)
            return True

        self._startup_model_catalog_refresh_scheduled = True
        self._model_catalog_notice_quiet = after_setup_completion
        self.run_worker(
            self._refresh_model_catalogs,
            exclusive=True,
            group=MODEL_CATALOG_REFRESH_WORKER_GROUP,
        )
        return True

    def _push_model_catalog_consent_modal(self) -> None:
        """Show the one-time consent dialog for online model-list checks."""
        if self.is_headless:
            # Headless/embedded runs have no user to answer a modal; stay
            # unconsented (no refresh) rather than blocking startup behind
            # an unanswerable dialog.
            return
        try:
            from tldw_chatbook.UI.Screens.model_catalog_consent import (
                ModelCatalogConsentModal,
            )
        except Exception as exc:
            logger.error(
                "Failed to import the model catalog consent modal "
                f"(screen=model_catalog_consent): {type(exc).__name__}"
            )
            return
        self.push_screen(ModelCatalogConsentModal(), self._handle_model_catalog_consent)

    async def _handle_model_catalog_consent(self, allowed: bool | None) -> None:
        """Persist the consent answer; on allow, run the startup refresh."""
        # Only the boolean singleton True counts as consent — truthy garbage
        # (e.g. a non-bool reaching this callback) falls through to the deny
        # path, mirroring the settings parser's strict validator.
        allowed = allowed is True
        try:
            from tldw_chatbook.config import save_settings_to_cli_config

            section = {"refresh_consent_recorded": True}
            if not allowed:
                section["auto_refresh_enabled"] = False
            saved = await asyncio.to_thread(
                save_settings_to_cli_config, {"model_catalog": section}
            )
        except Exception as exc:
            # No traceback: the log file sink runs with diagnose=True, which
            # would dump frame locals (including the app's config) into the log.
            logger.error(
                "Failed to persist model catalog consent "
                f"(allowed={allowed!r}, section=model_catalog): "
                f"{type(exc).__name__}"
            )
            saved = False
        if allowed:
            if not saved:
                self.notify(
                    "Your choice couldn't be saved; you'll be asked again next launch.",
                    title="Model catalog",
                    severity="warning",
                )
            self.run_worker(
                self._refresh_model_catalogs,
                exclusive=True,
                group=MODEL_CATALOG_REFRESH_WORKER_GROUP,
            )
        else:
            self.notify(
                "Online model-list checks stay off. You can enable them any "
                "time in Settings.",
                title="Model catalog",
            )

    def refresh_model_catalogs_now(self) -> None:
        """Run the provider catalog refresh immediately (TASK-21150).

        The public seam behind the wizard Summary's model-list consent, so
        answering "yes" there refreshes this session exactly as answering
        "yes" to the Console consent modal does — same worker, same
        exclusive group, so the two paths can never run concurrently.
        """
        self._startup_model_catalog_refresh_scheduled = True
        self._model_catalog_notice_quiet = True  # setup's own choice (AC#11)
        self.run_worker(
            self._refresh_model_catalogs,
            exclusive=True,
            group=MODEL_CATALOG_REFRESH_WORKER_GROUP,
        )
