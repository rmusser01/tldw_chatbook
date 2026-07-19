"""Automatic startup refresh for cloud provider model catalogs (ADR-019)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from textual.message import Message

RefreshStatus = Literal[
    "refreshed",
    "baseline",
    "skipped_disabled",
    "skipped_not_ready",
    "skipped_fresh",
    "failed",
]


@dataclass(frozen=True)
class ProviderRefreshOutcome:
    """Result of one provider's auto-refresh attempt."""

    provider_list_key: str
    status: RefreshStatus
    new_model_ids: tuple[str, ...] = ()
    saved_model_ids: tuple[str, ...] = ()
    error_kind: str | None = None
    write_failed: bool = False  # cache updated but config write-through failed


@dataclass(frozen=True)
class RefreshReport:
    """Aggregated result of one startup auto-refresh pass."""

    outcomes: tuple[ProviderRefreshOutcome, ...] = ()


def format_refresh_notification(report: RefreshReport) -> str | None:
    """Build one consolidated user notification, or None when nothing changed."""
    parts: list[str] = []
    failures: list[str] = []
    write_failures: list[str] = []
    for outcome in report.outcomes:
        if outcome.status == "refreshed" and outcome.saved_model_ids:
            parts.append(f"{outcome.provider_list_key}: {len(outcome.saved_model_ids)} new saved")
        elif (
            outcome.status == "refreshed"
            and outcome.new_model_ids
            and not outcome.write_failed
        ):
            # Suppressed on write failure: the save-failed clause already covers it.
            parts.append(f"{outcome.provider_list_key}: {len(outcome.new_model_ids)} new cached")
        elif outcome.status == "baseline":
            if outcome.new_model_ids:
                # Baseline suppressed the write; the diff is still reported as cached.
                parts.append(f"{outcome.provider_list_key}: {len(outcome.new_model_ids)} new cached")
            else:
                parts.append(f"{outcome.provider_list_key}: catalog cached")
        if outcome.write_failed:
            write_failures.append(outcome.provider_list_key)
        if outcome.status == "failed":
            failures.append(outcome.provider_list_key)
    if write_failures:
        parts.append(
            f"config save failed for {', '.join(write_failures)} (models cached instead)"
        )
    if not parts and not failures:
        return None
    message = f"Model lists updated — {', '.join(parts)}" if parts else ""
    if failures:
        message += ("; " if message else "") + (
            f"refresh failed for {', '.join(failures)} (using cached list)"
        )
    return message or None


class ModelCatalogRefreshed(Message):
    """Posted when startup refresh updated one or more provider catalogs."""

    def __init__(self, *, providers: set[str]) -> None:
        super().__init__()
        self.providers = frozenset(providers)


async def forward_model_catalog_refreshed(app: Any, event: ModelCatalogRefreshed) -> bool:
    """Forward the event to a mounted screen exposing a refresh handler.

    Textual messages bubble UP only: App.post_message() never reaches a Screen's
    @on handler (verified against Textual 8.2.7). The App-level handler calls this
    to reach the chat screen via duck typing; returns False when no mounted screen
    handles it (e.g. chat tab never opened — options build fresh on mount anyway).
    """
    for screen in reversed(getattr(app, "screen_stack", ())):  # pragma: no branch
        handler = getattr(screen, "handle_model_catalog_refreshed", None)
        if callable(handler):
            await handler(event)
            return True
    return False
