"""Settings > Video Gen panel (task-3401.12, AC1/AC3).

Self-contained editor pattern mirroring ``settings_image_gen_panel.py``:
compose-only against the live ``VideoGenerationConfig`` + the raw
``[video_generation]`` section (via ``load_user_video_generation_table()``
for input values, ``effective_placeholder`` for placeholders -- the
set-vs-default blur rule is unchanged). Secrets are never echoed; the
source line reports where the effective secret came from. The screen owns
staging/saving (``settings-videogen-`` ids); this widget only renders.

The Diagnostics section (AC3) reports per-backend configured status, the
minimax key source, and playback-tool availability (ffmpeg/ffplay/yt-dlp)
-- generation's own binary-level health read.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Button, Checkbox, Collapsible, Input, Select, Static

from tldw_chatbook.UI.Screens.settings_video_gen_defaults import (
    BACKEND_IDS,
    BACKEND_LABELS,
    DEFAULT_BACKEND_SELECT_ID,
    FIELD_SCHEMA,
    RETENTION_CHOICES,
    RETENTION_SELECT_ID,
    build_backend_rows,
    effective_placeholder,
    key_source_after_clear,
    load_user_video_generation_table,
    playback_tool_rows,
)
from tldw_chatbook.Video_Generation.config import get_video_generation_config
from tldw_chatbook.Video_Generation.video_templates import (
    BUILTIN_VIDEO_TEMPLATES,
    get_all_video_templates,
)
from tldw_chatbook.Widgets.settings_image_gen_panel import switch_word, toggle_label


_GENERATION_DEFAULT_FIELDS: tuple[tuple[str, str], ...] = (
    ("retention_ttl_hours", "Retention TTL (hours)"),
    ("max_store_mb", "Store cap (MB)"),
)

_STREAM_HINT_TEXT = "Stream URLs play via /stream-video <url> in the Console"


def _key_source_line(key_source: str) -> str:
    """Map a raw ``key_source`` value to the display text (image-panel strings)."""
    if key_source == "config":
        return "local config key saved"
    if key_source.startswith("env:"):
        return f"env: {key_source.split(':', 1)[1]}"
    if key_source == "keyring":
        return "keyring"
    return "missing"


def _secret_placeholder(key_source: str) -> str:
    if key_source == "config":
        return "Local config key saved; paste a replacement to change it"
    return "Paste a key/token to save locally in config"


def _advanced_keys_hint(backend_id: str) -> str:
    return (
        f"Advanced keys for {BACKEND_LABELS[backend_id]} live in config.toml -> "
        f"[video_generation.{backend_id}] (not editable here)."
    )


def _template_count_line() -> str:
    all_templates = get_all_video_templates()
    builtin_count = len(BUILTIN_VIDEO_TEMPLATES)
    user_count = max(len(all_templates) - builtin_count, 0)
    return (
        f"{builtin_count} built-in + {user_count} user styles · "
        "/generate-video @<style> · user styles via [video_generation.styles.<id>]"
    )


class VideoGenSettingsPanel(Vertical):
    """Browse + edit Video Gen backend defaults. Title is rendered by the screen."""

    def __init__(
        self,
        *args: Any,
        overlay: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        # Mirror of the image panel's overlay contract: the screen's staged
        # (unsaved) values, keyed like the draft ("" for nothing staged).
        self.overlay: Mapping[str, Any] = overlay or {}

    def compose(self) -> ComposeResult:
        cfg = get_video_generation_config(reload=True)
        raw_top: Mapping = load_user_video_generation_table()
        rows = build_backend_rows(cfg)
        overlay = self.overlay

        effective_default_backend = overlay.get("default_backend", cfg.default_backend)
        selected_backend = (
            effective_default_backend
            if effective_default_backend in BACKEND_IDS
            else BACKEND_IDS[0]
        )
        overlay_enabled_backends = overlay.get("enabled_backends")

        yield Static("Backends", classes="destination-section")
        with Horizontal(classes="settings-input-row settings-select-row"):
            yield Static("Default backend", classes="settings-input-label")
            yield Select(
                [(BACKEND_LABELS[backend_id], backend_id) for backend_id in BACKEND_IDS],
                value=(
                    effective_default_backend
                    if effective_default_backend in BACKEND_IDS
                    else Select.NULL
                ),
                id=DEFAULT_BACKEND_SELECT_ID,
                classes="settings-compact-select",
                allow_blank=True,
                compact=True,
            )
        for row in rows:
            is_enabled = (
                row.backend_id in overlay_enabled_backends
                if overlay_enabled_backends is not None
                else row.enabled
            )
            with Horizontal(
                id=f"settings-videogen-backend-{row.backend_id}",
                classes="settings-imagegen-backend-row",
            ):
                yield Static(row.label, classes="settings-input-label")
                yield Static(
                    "Configured" if row.configured else "Not configured",
                    id=f"settings-videogen-status-{row.backend_id}",
                    classes="settings-imagegen-badge",
                )
                yield Checkbox(
                    switch_word(is_enabled),
                    value=is_enabled,
                    id=f"settings-videogen-enabled-{row.backend_id}",
                    tooltip="Whether this backend participates in generation.",
                )
                yield Static(
                    "★ Default" if row.backend_id == selected_backend else "",
                    id=f"settings-videogen-default-marker-{row.backend_id}",
                    classes="settings-imagegen-default-marker",
                )

        yield Static("Backend settings", classes="destination-section")
        with VerticalScroll(id="settings-videogen-editor"):
            for row in rows:
                backend_id = row.backend_id
                raw_backend: Mapping = raw_top.get(backend_id) or {}
                with Collapsible(
                    title=BACKEND_LABELS[backend_id],
                    collapsed=(backend_id != selected_backend),
                    id=f"settings-videogen-editor-{backend_id}",
                ):
                    for spec in FIELD_SCHEMA[backend_id]:
                        field_overlay_key = f"field::{backend_id}::{spec.toml_key}"
                        if spec.kind == "secret":
                            cleared_key = f"cleared::{backend_id}::{spec.toml_key}"
                            key_source = (
                                key_source_after_clear(backend_id)
                                if cleared_key in overlay
                                else row.key_source
                            )
                        if spec.kind == "bool":
                            with Horizontal(classes="settings-input-row"):
                                yield Checkbox(
                                    toggle_label(
                                        spec.label,
                                        bool(
                                            overlay.get(
                                                field_overlay_key,
                                                raw_backend.get(spec.toml_key, False),
                                            )
                                        ),
                                    ),
                                    value=bool(
                                        overlay.get(
                                            field_overlay_key,
                                            raw_backend.get(spec.toml_key, False),
                                        )
                                    ),
                                    id=f"settings-videogen-field-{backend_id}-{spec.toml_key}",
                                    tooltip=(
                                        "Allow uploading locally generated images to this "
                                        "cloud backend as i2v first frames (privacy gate)."
                                    ),
                                )
                            continue
                        row_classes = (
                            "settings-imagegen-secret-row"
                            if spec.kind == "secret"
                            else "settings-input-row"
                        )
                        with Horizontal(classes=row_classes):
                            yield Static(spec.label, classes="settings-input-label")
                            if spec.kind == "secret":
                                yield Input(
                                    value=str(overlay.get(field_overlay_key, "")),
                                    id=f"settings-videogen-field-{backend_id}-{spec.toml_key}",
                                    classes="settings-compact-input",
                                    placeholder=_secret_placeholder(key_source),
                                    password=True,
                                )
                                yield Button(
                                    "Clear",
                                    id=f"settings-videogen-clear-{backend_id}-{spec.toml_key}",
                                    classes="settings-imagegen-clear-button",
                                    tooltip=(
                                        "Clears the locally saved key -- "
                                        "env/keyring sources still apply."
                                    ),
                                )
                            else:
                                raw_value = raw_backend.get(spec.toml_key)
                                default_value = "" if raw_value is None else str(raw_value)
                                yield Input(
                                    value=str(overlay.get(field_overlay_key, default_value)),
                                    id=f"settings-videogen-field-{backend_id}-{spec.toml_key}",
                                    classes="settings-compact-input",
                                    placeholder=effective_placeholder(
                                        cfg, backend_id, spec.toml_key
                                    ),
                                )
                        if spec.kind == "secret":
                            yield Static(
                                _key_source_line(key_source),
                                id=f"settings-videogen-key-source-{backend_id}",
                                classes="settings-imagegen-hint settings-imagegen-key-source",
                            )
                    yield Static(
                        _advanced_keys_hint(backend_id),
                        id=f"settings-videogen-advanced-hint-{backend_id}",
                        classes="settings-imagegen-hint",
                    )

        yield Static("Generation defaults", classes="destination-section")
        with Horizontal(classes="settings-input-row settings-select-row"):
            yield Static("Retention", classes="settings-input-label")
            retention_value = str(overlay.get("retention", cfg.retention))
            yield Select(
                [("session (wipe on app start)", "session"), ("ttl (keep N hours)", "ttl")],
                value=(
                    retention_value
                    if retention_value in RETENTION_CHOICES
                    else RETENTION_CHOICES[0]
                ),
                id=RETENTION_SELECT_ID,
                classes="settings-compact-select",
                allow_blank=False,
                compact=True,
            )
        yield Checkbox(
            toggle_label(
                "Confirm cost before paid generation",
                bool(overlay.get("confirm_cost_estimate", bool(cfg.confirm_cost_estimate))),
            ),
            value=bool(overlay.get("confirm_cost_estimate", bool(cfg.confirm_cost_estimate))),
            id="settings-videogen-confirm_cost_estimate",
            tooltip="Ask before spending on a paid (cloud) video generation.",
        )
        for key, label in _GENERATION_DEFAULT_FIELDS:
            with Horizontal(classes="settings-input-row"):
                yield Static(label, classes="settings-input-label")
                raw_value = raw_top.get(key)
                default_value = "" if raw_value is None else str(raw_value)
                yield Input(
                    value=str(overlay.get(key, default_value)),
                    id=f"settings-videogen-{key}",
                    classes="settings-compact-input",
                    placeholder=str(getattr(cfg, key)),
                )

        yield Static("Style templates", classes="destination-section")
        yield Static(
            _template_count_line(),
            id="settings-videogen-template-count",
            classes="settings-imagegen-hint",
        )
        yield Static(
            _STREAM_HINT_TEXT,
            id="settings-videogen-stream-hint",
            classes="settings-imagegen-hint",
        )

        yield Static("Diagnostics", classes="destination-section")
        for row in rows:
            yield Static(
                f"{row.label}: {'configured' if row.configured else 'not configured'}"
                + (
                    f" · key: {_key_source_line(row.key_source)}"
                    if row.backend_id == "minimax"
                    else ""
                ),
                id=f"settings-videogen-diag-{row.backend_id}",
                classes="settings-imagegen-hint",
            )
        for tool, found in playback_tool_rows():
            yield Static(
                f"{tool}: {'found' if found else 'MISSING (playback/streaming degraded)'}",
                id=f"settings-videogen-diag-tool-{tool}",
                classes=(
                    "settings-imagegen-hint"
                    if found
                    else "settings-imagegen-hint settings-imagegen-key-source"
                ),
            )

        with Horizontal(classes="settings-action-row"):
            yield Button("Save", id="settings-videogen-save", disabled=not overlay)
            yield Button("Revert", id="settings-videogen-revert", disabled=not overlay)
        yield Static(
            "",
            id="settings-videogen-save-result",
            classes="settings-imagegen-hint",
        )
