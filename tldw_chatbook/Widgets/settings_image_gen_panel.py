"""Settings > Image Gen panel (task 4 of the Settings > Image Gen plan).

Self-contained editor pattern (mirrors ``InternalPromptsPanel`` /
``SettingsThemeEditor``): owns its own compose against the live
``ImageGenerationConfig`` + the raw ``[image_generation]`` config section,
and posts nothing back to the screen this task -- Save/Revert/Test are
wired in tasks 5-6. The screen composes this widget with a single line
(see the ``IMAGE_GENERATION`` branch in ``settings_screen.py``'s
``_render_detail_pane``), keeping the 13k-line monolith's diff to a single
import + one compose branch.

READ-ONLY this task: every input renders a value (raw-config value when
set, ``effective_placeholder`` when unset) but stays ``disabled=True``.
Secrets are never echoed -- the input is always empty; a source line next
to it reports where the *effective* secret came from
(``ImageGenBackendRow.key_source``), reusing the exact three/four display
strings the design spec pins: ``"env: <VAR>"``, ``"local config key
saved"``, ``"keyring"``, ``"missing"``. SwarmUI's ``swarm_token`` is
optional for local installs, so its ``"missing"`` state renders with a
neutral (non-warning) CSS class per ``ImageGenBackendRow.secret_optional``
-- the text itself is unchanged, only the styling differs.
"""

from __future__ import annotations

from collections.abc import Mapping

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Button, Checkbox, Collapsible, Input, Select, Static

from tldw_chatbook.Image_Generation.config import get_image_generation_config
from tldw_chatbook.Media_Creation.generation_templates import (
    BUILTIN_TEMPLATES,
    get_all_templates,
)
from tldw_chatbook.UI.Screens.settings_config_adapter import SettingsConfigAdapter
from tldw_chatbook.UI.Screens.settings_image_gen_defaults import (
    BACKEND_IDS,
    BACKEND_LABELS,
    FIELD_SCHEMA,
    ImageGenBackendRow,
    build_backend_rows,
    effective_placeholder,
)


# Generation-defaults scalar fields, in display order. `context_llm_enabled`
# is rendered separately as a Checkbox (booleans have no placeholder analog).
_GENERATION_DEFAULT_FIELDS: tuple[tuple[str, str], ...] = (
    ("default_batch", "Default batch"),
    ("max_variants_per_message", "Max variants / message"),
    ("context_llm_turns", "Context LLM turns"),
    ("context_llm_timeout_seconds", "Context LLM timeout (s)"),
)

_DEMO_HINT_TEXT = "Test a generation end-to-end: command palette → Image Generation demo"


def _key_source_line(key_source: str) -> str:
    """Map a raw ``key_source`` value to the design spec's exact display text.

    One of ``"env: <VAR>"``, ``"local config key saved"``, ``"keyring"``, or
    ``"missing"`` -- never the raw ``"env:<VAR>"``/``"config"`` loader
    spelling verbatim.
    """
    if key_source == "config":
        return "local config key saved"
    if key_source.startswith("env:"):
        return f"env: {key_source.split(':', 1)[1]}"
    if key_source == "keyring":
        return "keyring"
    return "missing"


def _secret_placeholder(key_source: str) -> str:
    """Providers & Models convention: placeholder communicates saved-state.

    A masked secret input is never pre-filled with the saved value -- the
    placeholder says whether pasting will replace an existing local key or
    create a new one.
    """
    if key_source == "config":
        return "Local config key saved; paste a replacement to change it"
    return "Paste a key/token to save locally in config"


def _advanced_keys_hint(backend_id: str) -> str:
    return (
        f"Advanced keys for {BACKEND_LABELS[backend_id]} live in config.toml -> "
        f"[image_generation.{backend_id}] (not editable here)."
    )


def _template_count_line() -> str:
    all_templates = get_all_templates(reload=True)
    builtin_count = len(BUILTIN_TEMPLATES)
    user_count = max(len(all_templates) - builtin_count, 0)
    return (
        f"{builtin_count} built-in + {user_count} user templates · manage via "
        "[image_generation.styles.<id>] or <user_data_dir>/image_generation_styles/ "
        "(editing UI planned)"
    )


class ImageGenSettingsPanel(Vertical):
    """Browse Image Gen backend defaults. Title is rendered by the screen."""

    def compose(self) -> ComposeResult:
        cfg = get_image_generation_config(reload=True)
        raw_config: Mapping = SettingsConfigAdapter().load()
        raw_top: Mapping = raw_config.get("image_generation") or {}
        rows = build_backend_rows(cfg)
        rows_by_id: dict[str, ImageGenBackendRow] = {row.backend_id: row for row in rows}
        selected_backend = (
            cfg.default_backend if cfg.default_backend in BACKEND_IDS else BACKEND_IDS[0]
        )

        yield Static("Backends", classes="destination-section")
        with Horizontal(classes="settings-input-row settings-select-row"):
            yield Static("Default backend", classes="settings-input-label")
            yield Select(
                [(BACKEND_LABELS[backend_id], backend_id) for backend_id in BACKEND_IDS],
                value=cfg.default_backend if cfg.default_backend in BACKEND_IDS else Select.NULL,
                id="settings-imagegen-default_backend",
                classes="settings-compact-select",
                allow_blank=True,
                compact=True,
                disabled=True,
            )
        for row in rows:
            with Horizontal(
                id=f"settings-imagegen-backend-{row.backend_id}",
                classes="settings-imagegen-backend-row",
            ):
                yield Static(row.label, classes="settings-input-label")
                yield Static(
                    "Configured" if row.configured else "Not configured",
                    id=f"settings-imagegen-status-{row.backend_id}",
                    classes="settings-imagegen-badge",
                )
                yield Checkbox(
                    "Enabled",
                    value=row.enabled,
                    id=f"settings-imagegen-enabled-{row.backend_id}",
                    disabled=True,
                )
                yield Static(
                    "★ Default" if row.is_default else "",
                    id=f"settings-imagegen-default-marker-{row.backend_id}",
                    classes="settings-imagegen-default-marker",
                )
                yield Button(
                    "Test",
                    id=f"settings-imagegen-test-{row.backend_id}",
                    disabled=True,
                )

        yield Static("Backend settings", classes="destination-section")
        with VerticalScroll(id="settings-imagegen-editor"):
            for backend_id in BACKEND_IDS:
                row = rows_by_id[backend_id]
                raw_backend: Mapping = raw_top.get(backend_id) or {}
                with Collapsible(
                    title=BACKEND_LABELS[backend_id],
                    collapsed=(backend_id != selected_backend),
                    id=f"settings-imagegen-editor-{backend_id}",
                ):
                    for spec in FIELD_SCHEMA[backend_id]:
                        with Horizontal(classes="settings-input-row"):
                            yield Static(spec.label, classes="settings-input-label")
                            if spec.kind == "secret":
                                yield Input(
                                    value="",
                                    id=f"settings-imagegen-field-{backend_id}-{spec.toml_key}",
                                    classes="settings-compact-input",
                                    placeholder=_secret_placeholder(row.key_source),
                                    password=True,
                                    disabled=True,
                                )
                            else:
                                raw_value = raw_backend.get(spec.toml_key)
                                yield Input(
                                    value="" if raw_value is None else str(raw_value),
                                    id=f"settings-imagegen-field-{backend_id}-{spec.toml_key}",
                                    classes="settings-compact-input",
                                    placeholder=effective_placeholder(
                                        cfg, backend_id, spec.toml_key
                                    ),
                                    disabled=True,
                                )
                        if spec.kind == "secret":
                            source_classes = "settings-imagegen-hint settings-imagegen-key-source"
                            if row.secret_optional and row.key_source == "missing":
                                source_classes += " settings-imagegen-key-source-neutral"
                            yield Static(
                                _key_source_line(row.key_source),
                                id=f"settings-imagegen-key-source-{backend_id}",
                                classes=source_classes,
                            )
                        if backend_id == "openrouter" and spec.toml_key == "default_model":
                            yield Static(
                                "env OPENROUTER_IMAGE_MODEL overrides this",
                                classes="settings-imagegen-hint",
                            )
                    yield Static(
                        _advanced_keys_hint(backend_id),
                        id=f"settings-imagegen-advanced-hint-{backend_id}",
                        classes="settings-imagegen-hint",
                    )

        yield Static("Generation defaults", classes="destination-section")
        yield Checkbox(
            "Context LLM enabled",
            value=bool(cfg.context_llm_enabled),
            id="settings-imagegen-context_llm_enabled",
            tooltip=(
                "Whether the LLM-composed context prompt is attempted before "
                "falling back to the keyword extractor."
            ),
            disabled=True,
        )
        for key, label in _GENERATION_DEFAULT_FIELDS:
            with Horizontal(classes="settings-input-row"):
                yield Static(label, classes="settings-input-label")
                raw_value = raw_top.get(key)
                yield Input(
                    value="" if raw_value is None else str(raw_value),
                    id=f"settings-imagegen-{key}",
                    classes="settings-compact-input",
                    placeholder=str(getattr(cfg, key)),
                    disabled=True,
                )

        yield Static("Style templates", classes="destination-section")
        yield Static(
            _template_count_line(),
            id="settings-imagegen-template-count",
            classes="settings-imagegen-hint",
        )
        yield Static(
            _DEMO_HINT_TEXT,
            id="settings-imagegen-demo-hint",
            classes="settings-imagegen-hint",
        )

        with Horizontal(classes="settings-action-row"):
            yield Button("Save", id="settings-imagegen-save", disabled=True)
            yield Button("Revert", id="settings-imagegen-revert", disabled=True)
