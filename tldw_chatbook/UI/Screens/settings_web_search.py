"""Session-lifetime draft and worker state for guided web-search setup (ADR-012)."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from copy import deepcopy

from ... import config
from ...Web_Scraping import search_backend_settings as catalog
from .settings_config_models import SettingsDraft

DEFAULT_KEY = "SearchSettings.search_provider_default"
APPLICATION_DEFAULT = "application-default"
SEARCH_TERMS = " ".join(f"{key} {spec.label}" for key, spec in catalog.BACKENDS.items())


class WebSearchSettings:
    """Keep staged credentials and async results independent of panel remounts."""

    def __init__(self, draft: Callable[[], SettingsDraft], changed: Callable[[], None]):
        """Bind one retained draft and load its saved configuration baseline.

        Args:
            draft: Factory called once to obtain the session's staged settings.
            changed: Callback notifying the owning Settings screen of state changes.
        """
        self._draft = draft()
        self.changed = changed
        self.capture_input: Callable[[], None] | None = None
        self.view_changed: Callable[[], None] | None = None
        self.raw = config.load_cli_config_and_ensure_existence()
        self.path = config.get_cli_config_path()
        self.backend = self.default_backend
        if self.backend not in catalog.BACKENDS:
            self.backend = "duckduckgo"
        self.saving = False
        self.testing = False
        self.revision = 0
        self.save_status = ""
        self.test_status = "Not tested in this session."

    @property
    def draft(self) -> SettingsDraft:
        """Return the live staged values retained across disposable panel views.

        Returns:
            The session-owned mutable draft, including originals for conflict checks.
        """
        return self._draft

    def _saved(self, key: str):
        section, field = key.split(".", 1)
        return self.raw.get(section, {}).get(field)

    @property
    def default_backend(self) -> str:
        """Return the staged shared default, falling back to its saved selection.

        Returns:
            Backend identifier, or APPLICATION_DEFAULT when no local default is set.
        """
        value = self.draft.values.get(DEFAULT_KEY, self._saved(DEFAULT_KEY))
        return str(value) if value else APPLICATION_DEFAULT

    def _emit(self) -> None:
        self.changed()
        if self.view_changed:
            self.view_changed()

    def capture_pending_input(self) -> None:
        """Include values whose Input.Changed events have not reached the model."""
        if self.capture_input is not None:
            self.capture_input()

    def invalidate_test(self) -> None:
        """Advance the view/settings revision and invalidate any in-flight probe.

        A running request finishes normally, but its old result cannot become the
        current view's test status. This method does not emit a change notification.
        """
        self.revision += 1
        self.test_status = (
            "Previous search test is still finishing; its result will be discarded."
            if self.testing
            else "Not tested for the current view and settings."
        )

    def select_backend(self, backend: str) -> None:
        """Select the backend to edit without changing the shared search default.

        Args:
            backend: Catalog identifier; unknown or unchanged selections are ignored.
        """
        if backend in catalog.BACKENDS and backend != self.backend:
            self.backend = backend
            self.invalidate_test()
            self._emit()

    def _stage(self, key: str, value: object) -> None:
        if self.saving:
            return
        original = self.draft.originals.get(key, self._saved(key))
        self.draft.set_value(key, original, value)
        self.invalidate_test()
        self.save_status = ""
        self._emit()

    def set_default(self, backend: str) -> None:
        """Stage the shared default and invalidate prior search-test evidence.

        Args:
            backend: Catalog identifier or APPLICATION_DEFAULT to remove the local
                preference. Unknown identifiers and edits during saving are ignored.
        """
        if backend in catalog.BACKENDS or backend == APPLICATION_DEFAULT:
            self._stage(
                DEFAULT_KEY, None if backend == APPLICATION_DEFAULT else backend
            )

    def edit(self, key: str, value: str) -> None:
        """Stage a field replacement without writing configuration.

        Empty secret input keeps the saved value; explicit deletion uses ``clear``.
        Other values are trimmed, and empty nonsecret input stages deletion. Edits
        during saving are ignored.

        Args:
            key: Canonical field key from the backend catalog.
            value: Current editor text, including an empty replacement.

        Raises:
            StopIteration: The key is not a field in the backend catalog.
        """
        field = next(
            field
            for spec in catalog.BACKENDS.values()
            for field in spec.fields
            if field.key == key
        )
        full_key = f"SearchEngines.{key}"
        if field.secret and not value:
            if self.saving:
                return
            # Empty replacement means keep saved. Clear is an explicit action.
            self.draft.values.pop(full_key, None)
            self.draft.originals.pop(full_key, None)
            self.invalidate_test()
            self.save_status = ""
            self._emit()
        else:
            self._stage(full_key, value.strip() or None)

    def clear(self, key: str) -> None:
        """Capture pending input and stage removal of a local field and its aliases.

        Environment overrides remain effective. No changes are made during saving.

        Args:
            key: Canonical backend field key to remove on the next Save.
        """
        self.capture_pending_input()
        self._stage(f"SearchEngines.{key}", None)
        # Delete aliases too: otherwise a legacy key would become effective again.
        for alias in catalog.LEGACY_FIELD_ALIASES.get(key, ()):
            if self._saved(f"SearchEngines.{alias}") is not None:
                self._stage(f"SearchEngines.{alias}", None)

    def input_value(self, field: catalog.FieldSpec) -> str:
        """Return editable local text while keeping saved credentials masked.

        Args:
            field: Catalog field whose editor is being rendered.

        Returns:
            Staged replacement, saved nonsecret value, or empty text for saved secrets.
            Environment overrides are described separately rather than inserted.
        """
        key = f"SearchEngines.{field.key}"
        if key in self.draft.values:
            return str(self.draft.values[key] or "")
        if field.secret:
            return ""
        # Edit the saved local value; the source line identifies any env override.
        return catalog.saved_field_value(field, self.raw)

    def field_status(self, field: catalog.FieldSpec) -> str:
        """Explain a field's effective source and any staged replacement or clear.

        Args:
            field: Catalog field whose source guidance is being rendered.

        Returns:
            Credential-safe guidance, including environment precedence when relevant.
        """
        source = catalog.field_source(field, self.raw)
        key = f"SearchEngines.{field.key}"
        staged = ""
        if key in self.draft.dirty_keys:
            staged = (
                "Local value will be cleared. "
                if self.draft.values[key] is None
                else "Replacement staged. "
            )
        if source.startswith("Environment:"):
            return staged + source + " takes precedence over local values."
        return (
            staged
            + source
            + (
                ". Leave blank to keep; Clear removes the local value."
                if field.secret
                else "."
            )
        )

    def preview(self) -> dict:
        """Return a separate local-config preview with all staged changes applied.

        Returns:
            Deep copy of the saved baseline, including replacements and deletions.
            It may contain credentials and must not be logged or persisted as UI state.
        """
        raw = deepcopy(self.raw)
        for key in self.draft.dirty_keys:
            section, field = key.split(".", 1)
            table = raw.setdefault(section, {})
            value = self.draft.values[key]
            if value is None:
                table.pop(field, None)
            else:
                table[field] = value
        return raw

    @property
    def default_status(self) -> str:
        """Describe missing setup for the staged shared default without a request.

        Returns:
            Credential-safe blockers, or empty text when required setup is present.
        """
        backend = (
            "duckduckgo"
            if self.default_backend == APPLICATION_DEFAULT
            else self.default_backend
        )
        issues = catalog.setup_issues(backend, self.preview())
        return "Default setup incomplete: " + " ".join(issues) if issues else ""

    @property
    def setup_status(self) -> str:
        """Describe offline readiness of the backend currently being edited.

        Returns:
            Credential-safe blockers or confirmation that required settings are present.
        """
        issues = catalog.setup_issues(self.backend, self.preview())
        if issues:
            return "Setup incomplete: " + " ".join(issues)
        return "Required settings present."

    @property
    def can_test(self) -> bool:
        """Return whether an explicit probe can use the current saved setup.

        Returns:
            True only with complete saved setup, no dirty draft, and no save or test
            already running. Readiness does not establish network connectivity.
        """
        return not (
            self.saving
            or self.testing
            or self.draft.is_dirty
            or catalog.setup_issues(self.backend, self.raw)
        )

    def revert(self) -> None:
        """Discard staged values and reload the effective local config baseline.

        Pending input is captured first. Saving blocks the operation; otherwise
        previous test evidence is invalidated and views are notified of the reload.
        The caller owns confirmation before discarding unsaved work.
        """
        self.capture_pending_input()
        if self.saving:
            return
        self.draft.values.clear()
        self.draft.originals.clear()
        self.raw = config.load_cli_config_and_ensure_existence(force_reload=True)
        self.path = config.get_cli_config_path()
        self.invalidate_test()
        self.save_status = "Draft discarded; showing saved settings."
        self._emit()

    async def save(self) -> None:
        """Commit staged fields off the UI thread with per-field conflict checks.

        Pending input joins the submission. Duplicate or clean saves are ignored,
        and edits are blocked while saving. A committed write clears the draft even
        if runtime refresh fails; conflicts and pre-write failures retain it.
        Completion updates ``save_status`` with credential-safe recovery guidance
        and notifies whichever view is attached to this retained session.
        """
        self.capture_pending_input()
        if self.saving or not self.draft.is_dirty:
            return
        self.saving = True
        self.invalidate_test()
        self.save_status = "Saving…"
        self._emit()
        changes = {key: self.draft.values[key] for key in self.draft.dirty_keys}
        originals = {key: self.draft.originals[key] for key in changes}
        path = self.path
        sets: dict[str, dict] = {}
        deletes: dict[str, list[str]] = {}
        for key, value in changes.items():
            section, field = key.split(".", 1)
            if value is None:
                deletes.setdefault(section, []).append(field)
            else:
                sets.setdefault(section, {})[field] = value

        def unchanged(snapshot) -> bool:
            return config.get_cli_config_path() == path and all(
                snapshot.values.get(key.split(".", 1)[0], {}).get(key.split(".", 1)[1])
                == originals[key]
                for key in changes
            )

        try:
            result = await asyncio.to_thread(
                config.apply_settings_mutation_to_cli_config,
                sets,
                delete_keys=deletes,
                locked_snapshot_precondition=unchanged,
            )
            if result.file_replaced:
                # The write is committed even if the subsequent read fails.
                # Keep the committed values as the next editing baseline.
                self.raw = self.preview()
                self.draft.values.clear()
                self.draft.originals.clear()
                refreshed = result.caches_reloaded
                try:
                    self.raw = await asyncio.to_thread(
                        config.load_cli_config_and_ensure_existence, force_reload=True
                    )
                except Exception:  # noqa: BLE001 - never expose config credentials
                    refreshed = False
                self.save_status = (
                    "Saved. New searches use these settings."
                    if refreshed
                    else "Saved to disk. Restart the app before searching; runtime refresh failed."
                )
            elif result.conflict:
                self.save_status = "Saved settings changed elsewhere. Revert to reload them, then reapply your edits."
            else:
                self.save_status = "Could not save. Your draft is retained; check config file access and try again."
        except Exception:  # noqa: BLE001 - config errors must never expose credentials
            self.save_status = "Could not save. Your draft is retained; check config file access and try again."
        finally:
            self.saving = False
            self._emit()

    async def test_saved(self) -> None:
        """Run one explicit sample query using saved settings off the UI thread.

        Pending input is captured before checking readiness. Dirty, incomplete,
        saving or already-testing states block the request with status guidance.
        Editing or switching backends invalidates its result; only the original
        revision receives the credential-safe completion in ``test_status``.
        """
        self.capture_pending_input()
        if not self.can_test:
            self.test_status = (
                "Save or revert edits and complete setup before testing saved settings."
            )
            self._emit()
            return
        self.testing = True
        self.test_status = "Testing saved settings…"
        backend, revision = self.backend, self.revision
        self._emit()
        try:
            result = await asyncio.to_thread(catalog.probe_saved_backend, backend)
            self.capture_pending_input()
            if revision == self.revision:
                self.test_status = result.message
        except Exception:  # noqa: BLE001 - closed provider diagnostic boundary
            if revision == self.revision:
                self.test_status = (
                    "Search test failed. Check the saved setup and try again."
                )
        finally:
            self.testing = False
            if revision != self.revision:
                self.test_status = "Not tested for the current view and settings."
            self._emit()
