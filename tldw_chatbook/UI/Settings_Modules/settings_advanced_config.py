"""Guarded in-memory raw TOML drafts, owned by Settings (ADR-033)."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field

from ... import config
from .settings_config_adapter import SettingsConfigAdapter


@dataclass
class RawConfigDraft:
    """Ephemeral editor state; never serialize this body to durable UI metadata."""

    text: str = field(default="", repr=False)
    baseline_text: str = field(default="", repr=False)
    snapshot: config.ConfigFileSnapshot | None = None
    revision: int = 0
    validated_revision: int | None = None
    file_changed: bool = False

    @property
    def is_dirty(self) -> bool:
        """Whether the editor differs from its last accepted baseline."""
        return self.text != self.baseline_text


class AdvancedConfigSettings:
    """Retain work across views and apply async results only to their revision."""

    def __init__(
        self,
        changed: Callable[[], None],
        applied: Callable[[dict], None],
        state: RawConfigDraft | None = None,
    ):
        self.adapter = SettingsConfigAdapter()
        self.state = state or RawConfigDraft()
        self.changed = changed
        self.applied = applied
        self.view_changed: Callable[[], None] | None = None
        self.read_editor: Callable[[], str] | None = None
        self._initial_load = state is None
        self.busy = "Loading config…" if self._initial_load else ""
        self.result = ""

    def _capture_editor(self) -> None:
        """Include input whose TextArea.Changed message has not bubbled yet."""
        if self.read_editor is not None:
            self.edit(self.read_editor())

    def _emit(self) -> None:
        self.changed()
        if self.view_changed:
            self.view_changed()

    def _accept_snapshot(self, snapshot: config.ConfigFileSnapshot) -> None:
        self.state.snapshot = snapshot
        self.state.text = snapshot.serialized or ""
        self.state.baseline_text = self.state.text
        self.state.revision += 1
        self.state.validated_revision = None
        self.state.file_changed = False

    def edit(self, text: str) -> None:
        """Retain exact text and invalidate validation for older revisions."""
        if text == self.state.text:
            return
        self.state.text = text
        self.state.revision += 1
        self.result = ""
        self._emit()

    @property
    def validation_status(self) -> str:
        if self.state.validated_revision is None:
            return "Not validated."
        if self.state.validated_revision == self.state.revision:
            return "Current text validated."
        return "Text changed; validate again."

    @property
    def can_save(self) -> bool:
        return (
            not self.busy
            and self.state.snapshot is not None
            and self.state.validated_revision == self.state.revision
            and not self.state.file_changed
        )

    @property
    def status(self) -> str:
        if self.state.file_changed:
            if self.state.snapshot is None:
                return (
                    f"{self.result} Copy any edits, then Revert to reload the saved "
                    "file before saving again."
                ).strip()
            return "Config changed elsewhere. Copy your draft, then Revert to reload before saving."
        return self.result

    async def inspect_current(self) -> None:
        """Read off the UI thread, refreshing only a still-current clean view."""
        if self.state.file_changed and self.state.snapshot is None:
            # A lost post-commit baseline is not an initial load. Navigation must
            # not adopt an external edit as authority for the retained draft.
            return
        initial_load = self._initial_load
        if self.busy and not initial_load:
            return
        self._initial_load = False
        original = self.state.snapshot
        revision = self.state.revision
        try:
            current = await asyncio.to_thread(self.adapter.read_snapshot)
        except (OSError, RuntimeError, ValueError):
            if self.state.snapshot != original or self.state.revision != revision:
                return
            self.result = "Cannot read current config. Draft kept; check file access, then Revert to reload."
            self.state.file_changed = original is not None
        else:
            self._capture_editor()
            if self.state.snapshot != original:
                return
            if current != original:
                if original is None and (
                    self.state.is_dirty or self.state.revision != revision
                ):
                    # A draft created before the first read completes still
                    # belongs to this baseline; never overwrite that text.
                    self.state.snapshot = current
                    self.state.baseline_text = current.serialized or ""
                elif self.state.is_dirty or self.state.revision != revision:
                    self.state.file_changed = True
                else:
                    self._accept_snapshot(current)
                    self.result = "" if initial_load else "Reloaded current config."
        finally:
            if initial_load:
                self.busy = ""
            self._emit()

    async def validate(self) -> None:
        """Validate a captured revision without granting Save to later edits."""
        self._capture_editor()
        if self.busy:
            return
        self.busy = "Validating…"
        text, revision = self.state.text, self.state.revision
        self._emit()
        try:
            result = await asyncio.to_thread(self.adapter.validate_raw_toml, text)
            self._capture_editor()
            if revision == self.state.revision:
                self.state.validated_revision = revision if result.valid else None
                self.result = (
                    "Valid TOML."
                    if result.valid
                    else "Invalid TOML. Check syntax and top-level table, then validate again."
                )
        except Exception:  # noqa: BLE001 - parser errors may contain raw credentials
            if revision == self.state.revision:
                self.state.validated_revision = None
                self.result = "Validation failed. Draft kept; check TOML and try again."
        finally:
            self.busy = ""
            self._emit()

    async def save(self) -> None:
        """Persist the validated revision while retaining any newer editor work."""
        self._capture_editor()
        if not self.can_save:
            self.result = (
                "Save blocked. Revert to reload the config file first."
                if self.state.snapshot is None
                else "Save blocked. Validate current text before saving."
            )
            self._emit()
            return
        text, revision, snapshot = (
            self.state.text,
            self.state.revision,
            self.state.snapshot,
        )
        self.busy = "Saving…"
        self._emit()
        try:
            refreshed = True
            try:
                loaded, backup, saved = await asyncio.to_thread(
                    self.adapter.replace_snapshot, text, snapshot
                )
            except config.ConfigPostCommitError as error:
                backup, saved = error.backup_path, error.snapshot
                refreshed = False
            self._capture_editor()
            self.state.snapshot = saved
            self.state.baseline_text = text
            if saved is None:
                self.state.validated_revision = None
                self.state.file_changed = True
            elif self.state.revision == revision:
                # Show the representation actually saved, including encryption
                # and protected-section normalization from the config owner.
                self._accept_snapshot(saved)
                self.state.validated_revision = self.state.revision
            if refreshed:
                try:
                    self.applied(loaded)
                except Exception:  # noqa: BLE001 - view refresh cannot undo the write
                    refreshed = False
            if refreshed:
                self.result = (
                    "Saved; backup created."
                    if backup
                    else "Saved; no previous file to back up."
                )
            else:
                self.result = (
                    "Saved to disk. Restart the app because post-save refresh failed."
                )
            if self.state.is_dirty:
                self.result += " Newer edits remain unsaved."
                if not refreshed:
                    self.result += " Copy them before restarting."
        except config.ConfigSnapshotConflictError:
            self.state.file_changed = True
        except Exception:  # noqa: BLE001 - closed disk/encryption diagnostic boundary
            self.result = (
                "Save failed; draft kept. Reload to check disk before retrying."
            )
        finally:
            self.busy = ""
            self._emit()

    async def replace_draft(self, source: str, revision: int) -> None:
        """Apply a confirmed replacement only if no later edit superseded it."""
        self._capture_editor()
        if self.busy or revision != self.state.revision:
            return
        self.busy = "Loading backup…" if source == "backup" else "Reloading…"
        self._emit()
        try:
            if source == "backup":
                text = await asyncio.to_thread(self.adapter.read_backup_serialized)
                self._capture_editor()
                if revision == self.state.revision:
                    self.state.text = text
                    self.state.revision += 1
                    self.state.validated_revision = None
                    self.result = "Backup loaded as a draft. Validate before saving."
                else:
                    self.result = "Newer edits kept; replacement was not applied."
            else:
                snapshot = await asyncio.to_thread(self.adapter.read_snapshot)
                self._capture_editor()
                if revision == self.state.revision:
                    self._accept_snapshot(snapshot)
                    self.result = "Reloaded current config; draft discarded."
                else:
                    self.result = "Newer edits kept; replacement was not applied."
        except FileNotFoundError:
            self.result = "No backup found. Draft kept."
        except (OSError, RuntimeError, ValueError):
            self.result = "Could not load config text. Draft kept; check file access and encoding."
        finally:
            self.busy = ""
            self._emit()
