"""Literal, disposable provenance inspection for user-owned Settings surfaces."""

from __future__ import annotations

from collections.abc import Callable
from time import monotonic
from typing import Any
from unicodedata import category

from rich.text import Text
from textual import on, work
from textual.app import App
from textual.timer import Timer
from textual.widgets import Button, Collapsible, Static

from ...Personal_Context.service import PersonalContextService
from ...Personal_Context.settings_provenance import (
    SettingsProvenanceProjection,
    SettingsProvenanceResult,
    SettingsProvenanceSubject,
)

_REFRESH_SECONDS = 1.0
_LEASE_SECONDS = 2.0
_MAX_REFERENCES = 8


def literal_metadata(value: str, *, limit: int = 160) -> str:
    """Remove terminal/directional controls and explicitly clip long metadata."""
    cleaned = "".join(
        " " if category(char) in {"Cc", "Cf", "Zl", "Zp"} else char for char in value
    )
    return cleaned if len(cleaned) <= limit else cleaned[: max(0, limit - 1)] + "…"


def _render_projection(projection: SettingsProvenanceProjection) -> Text:
    history = {"Edit history", "Inference classification"}
    lines = [
        f"{field.label}: {literal_metadata(field.value)}"
        for field in projection.fields
        if field.label not in history
    ]
    lines.append(projection.reference_status)
    for label, values in (
        ("Source references", projection.source_references),
        ("Hashes (unverified)", projection.source_hashes),
    ):
        if values:
            lines.append(
                f"{label} — showing {min(len(values), _MAX_REFERENCES)} of {len(values)}"
            )
            lines.extend(literal_metadata(value) for value in values[:_MAX_REFERENCES])
    lines.extend(field.value for field in projection.fields if field.label in history)
    lines.append(
        "Recorded approval does not verify the source or current wording. Earlier content may no longer exist."
    )
    return Text("\n".join(lines))


class PersonalContextProvenanceDetails(Collapsible):
    """Inspect one selected version; disposal never controls mutation workers."""

    def __init__(
        self,
        subject: SettingsProvenanceSubject,
        service_loader: Callable[[], PersonalContextService],
        *,
        reload_details: Callable[[], None] | None = None,
        **kwargs: Any,
    ) -> None:
        self._subject = subject
        self._service_loader = service_loader
        self._reload_details = reload_details
        self._content = Static(
            Text("Expand to inspect recorded metadata."),
            markup=False,
            classes="personal-context-provenance-content",
        )
        self._reload = Button(
            "Reload details", classes="personal-context-provenance-reload"
        )
        self._reload.display = False
        super().__init__(
            self._content, self._reload, title="Recorded provenance", **kwargs
        )
        self.add_class("personal-context-provenance")
        self._generation = 0
        self._pending = False
        self._latched = False
        self._owner: PersonalContextService | None = None
        self._timer: Timer | None = None
        self._lease_timer: Timer | None = None
        self._deadline = 0.0

    def on_mount(self) -> None:
        self.call_after_refresh(self.resume)

    def on_unmount(self) -> None:
        self.invalidate()
        if self._timer is not None:
            self._timer.stop()

    def _eligible(self) -> bool:
        return (
            self.is_mounted
            and not self.collapsed
            and self.screen.is_current
            and not self._latched
        )

    @on(Collapsible.Expanded)
    def _expanded(self, event: Collapsible.Expanded) -> None:
        if event.control is self:
            event.stop()
            self.call_after_refresh(self.resume)

    @on(Collapsible.Collapsed)
    def _collapsed(self, event: Collapsible.Collapsed) -> None:
        if event.control is self:
            event.stop()
            self.suspend()

    def suspend(self) -> None:
        """Discard metadata and pause reads while its owning surface is hidden."""
        self._generation += 1
        if self._timer is not None:
            self._timer.pause()
        if self._lease_timer is not None:
            self._lease_timer.stop()
            self._lease_timer = None
        self._deadline = 0.0
        self._content.update(Text("Provenance unavailable. Expand to inspect again."))
        self._reload.display = False

    def invalidate(self) -> None:
        """Permanently fence this captured selection until its host reloads it."""
        self.suspend()
        self._latched = True

    def resume(self) -> None:
        """Start selected-item reads after mount or a screen resume."""
        if not self._eligible():
            return
        if self._timer is None:
            self._timer = self.set_interval(_REFRESH_SECONDS, self._tick)
        else:
            self._timer.resume()
        self._tick()

    def _tick(self) -> None:
        if not self._eligible():
            return
        if self._deadline and monotonic() >= self._deadline:
            self._expire()
        if self._pending:
            return
        self._pending = True
        self._read(self._generation, self._subject, self.app)

    @work(thread=True, group="personal-context-provenance-read", exit_on_error=False)
    def _read(
        self, generation: int, subject: SettingsProvenanceSubject, app: App
    ) -> None:
        deadline = monotonic() + _LEASE_SECONDS
        owner = None
        try:
            owner = self._service_loader()
            result = owner.settings_provenance(subject)
        except Exception:  # noqa: BLE001 — optional diagnostics fail closed without personal logs.
            # Diagnostics must fail closed without logging personal data.
            result = SettingsProvenanceResult("unavailable")
        app.call_from_thread(self._finish, generation, subject, owner, deadline, result)

    def _finish(
        self,
        generation: int,
        subject: SettingsProvenanceSubject,
        owner: PersonalContextService | None,
        deadline: float,
        result: SettingsProvenanceResult,
    ) -> None:
        self._pending = False
        if (
            generation != self._generation
            or subject != self._subject
            or not self._eligible()
        ):
            return
        if self._owner is not None and owner is not self._owner:
            result = SettingsProvenanceResult("changed")
        if owner is not None:
            self._owner = owner
        if result.state == "changed":
            self.invalidate()
            self._content.update(
                Text(
                    "Record changed; reload details"
                    if subject.object_type == "record"
                    else "Proposal changed; close and reopen review."
                )
            )
            self._reload.display = self._reload_details is not None
            return
        if (
            deadline <= monotonic()
            or result.state != "available"
            or result.projection is None
            or result.projection.subject != subject
        ):
            self._content.update(Text("Provenance unavailable."))
            return
        self._content.update(_render_projection(result.projection))
        self._deadline = deadline
        if self._lease_timer is not None:
            self._lease_timer.stop()
        self._lease_timer = self.set_timer(deadline - monotonic(), self._expire)

    def _expire(self) -> None:
        if self._deadline and monotonic() >= self._deadline:
            self._content.update(
                Text("Provenance unavailable. Refreshing recorded metadata…")
            )
            self._deadline = 0.0

    @on(Button.Pressed, ".personal-context-provenance-reload")
    def _reload_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        if self._reload_details is not None:
            self._reload_details()
