# Config-Driven Attachment Filters and Image Caps (TASK-222) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Drive the attachment picker filters, image format allowlist, size cap, and resize dimension from `[chat.images]` config — one effective list for picker, routing, and pipeline — extending real support to `.tiff`/`.tif` (PIL + payload transcode) and `.svg` (cairosvg, capability-gated).

**Architecture:** Call-time policy functions in `Chat/attachment_core.py` (read via `get_cli_setting`) become the single source; a new `ChatImageHandler.prepare_image_payload` normalizes every image to a provider-safe format with a mime that matches the bytes (SVG rasterizes bounded, first); routing (`file_handlers`), paste gate (`console_paste_attach`), and both pickers consume the policy at call time. cairosvg is a new optional extra registered in `optional_deps.py` with a verified macOS Homebrew dyld fix.

**Tech Stack:** Python 3.11+, PIL/Pillow, cairosvg (optional, needs the cairo C library), defusedxml (ships with cairosvg), pytest + pytest-asyncio.

**Spec:** `Docs/superpowers/specs/2026-07-14-console-config-caps-design.md` (approved 2026-07-14).

## Global Constraints

- Work ONLY inside this worktree (`.claude/worktrees/console-config-caps-222`). NEVER touch `~/.config/tldw_cli/config.toml` (the live user config) or any file outside the worktree.
- NEVER use bare `git stash` (shared stash stack across sessions). Prefer a WIP commit if you must set work aside.
- Existing test files are READ-ONLY. All new tests go in NEW files. At branch end `git diff origin/dev --diff-filter=M --name-only -- Tests/` must print nothing.
- The legacy image regression gate must stay green with ZERO edits: `Tests/Event_Handlers/Chat_Events/test_chat_image_events.py`, `Tests/Event_Handlers/Chat_Events/test_chat_image_properties.py`, `Tests/UI/test_chat_image_attachment.py`, `Tests/unit/test_chat_image_unit.py`.
- Run pytest as `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest` (the `timeout` shell command is NOT available in this environment).
- Pinned literals that must NOT change: `ChatImageHandler.MAX_IMAGE_SIZE = 10 * 1024 * 1024`; `ChatImageHandler.SUPPORTED_FORMATS = {'.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp'}`; `_process_image_data(image_data, extension, mime_type) -> bytes` signature and bytes-only return.
- `DEFAULT_SUPPORTED_IMAGE_FORMATS` is exactly `(".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tiff", ".tif", ".svg")`.
- All config reads are call-time via `get_cli_setting("chat.images", key, default)` — never at import time, never through `app_config`.
- Tests must never read the live config: every test whose subject reads policy MUST monkeypatch `tldw_chatbook.config.get_cli_setting` (the live config pins the OLD 6-format list, so unpatched tiff/svg tests would fail falsely).
- NEVER pass `unsafe=True` to any cairosvg call.
- Error copy preserved: `"Unsupported image format: ..."`, `"Image file too large (...)"`, and the size line's `{cap / 1024 / 1024}MB` float rendering (`10.0MB` under defaults).

---

### Task 1: cairosvg optional dependency + `ensure_svg_rendering`

**Files:**
- Modify: `pyproject.toml` (after the `mcp` extra, lines 121-123)
- Modify: `tldw_chatbook/Utils/optional_deps.py` (imports at top; `DEPENDENCIES_AVAILABLE` dict at line 11; new functions after `check_dependency`, which ends near line 408)
- Test: `Tests/Utils/test_svg_rendering_dep.py` (new file)

**Interfaces:**
- Consumes: `check_dependency(module_name, feature_name)` and `DEPENDENCIES_AVAILABLE` (both already in `optional_deps.py`).
- Produces: `optional_deps.ensure_svg_rendering() -> bool` (cached after first call; performs the darwin dyld fix before the first cairosvg import). Later tasks call it through `attachment_core.svg_rendering_available()`.

**Background:** cairosvg needs the cairo C library. On macOS, Homebrew installs it in `/opt/homebrew/lib`, which neither `dlopen` nor `ctypes.util.find_library` searches by default. `find_library` (cairocffi's fallback resolver) re-reads `DYLD_FALLBACK_LIBRARY_PATH` from `os.environ` on every call, so an in-process append before the first import fixes resolution — verified live on this machine. When the variable is unset, macOS uses a default fallback chain; we reproduce it before appending so other libraries keep resolving.

- [ ] **Step 1: Write the failing tests**

Create `Tests/Utils/test_svg_rendering_dep.py`:

```python
"""Tests for optional_deps.ensure_svg_rendering (TASK-222)."""

import os

import pytest

from tldw_chatbook.Utils import optional_deps


@pytest.fixture(autouse=True)
def _reset_svg_cache(monkeypatch):
    """Each test starts from the unchecked state."""
    monkeypatch.setattr(optional_deps, "_svg_rendering_available", None)


def test_registered_in_dependencies_available():
    assert "svg_rendering" in optional_deps.DEPENDENCIES_AVAILABLE


def test_success_is_cached(monkeypatch):
    calls = []

    def fake_check(module, feature=None):
        calls.append(module)
        return True

    monkeypatch.setattr(optional_deps, "check_dependency", fake_check)
    assert optional_deps.ensure_svg_rendering() is True
    assert optional_deps.ensure_svg_rendering() is True
    assert calls == ["cairosvg"]  # second call served from cache


def test_failure_is_cached(monkeypatch):
    calls = []

    def fake_check(module, feature=None):
        calls.append(module)
        return False

    monkeypatch.setattr(optional_deps, "check_dependency", fake_check)
    assert optional_deps.ensure_svg_rendering() is False
    assert optional_deps.ensure_svg_rendering() is False
    assert calls == ["cairosvg"]


def test_darwin_sets_fallback_path_when_unset(monkeypatch):
    monkeypatch.setattr(optional_deps.sys, "platform", "darwin")
    monkeypatch.setattr(optional_deps, "check_dependency", lambda m, f=None: True)
    monkeypatch.setattr(
        optional_deps.os.path, "isdir", lambda p: p == "/opt/homebrew/lib"
    )
    monkeypatch.delenv("DYLD_FALLBACK_LIBRARY_PATH", raising=False)
    optional_deps.ensure_svg_rendering()
    value = os.environ["DYLD_FALLBACK_LIBRARY_PATH"]
    entries = value.split(":")
    assert entries[-1] == "/opt/homebrew/lib"
    # the dyld default fallback chain is preserved ahead of the append
    assert "/usr/local/lib" in entries and "/usr/lib" in entries


def test_darwin_appends_to_existing_value(monkeypatch):
    monkeypatch.setattr(optional_deps.sys, "platform", "darwin")
    monkeypatch.setattr(optional_deps, "check_dependency", lambda m, f=None: True)
    monkeypatch.setattr(
        optional_deps.os.path, "isdir", lambda p: p == "/opt/homebrew/lib"
    )
    monkeypatch.setenv("DYLD_FALLBACK_LIBRARY_PATH", "/custom/lib")
    optional_deps.ensure_svg_rendering()
    assert os.environ["DYLD_FALLBACK_LIBRARY_PATH"] == "/custom/lib:/opt/homebrew/lib"


def test_darwin_append_is_idempotent(monkeypatch):
    monkeypatch.setattr(optional_deps.sys, "platform", "darwin")
    monkeypatch.setattr(optional_deps, "check_dependency", lambda m, f=None: True)
    monkeypatch.setattr(
        optional_deps.os.path, "isdir", lambda p: p == "/opt/homebrew/lib"
    )
    monkeypatch.setenv("DYLD_FALLBACK_LIBRARY_PATH", "/custom/lib:/opt/homebrew/lib")
    optional_deps.ensure_svg_rendering()
    assert (
        os.environ["DYLD_FALLBACK_LIBRARY_PATH"] == "/custom/lib:/opt/homebrew/lib"
    )


def test_non_darwin_leaves_env_alone(monkeypatch):
    monkeypatch.setattr(optional_deps.sys, "platform", "linux")
    monkeypatch.setattr(optional_deps, "check_dependency", lambda m, f=None: True)
    monkeypatch.delenv("DYLD_FALLBACK_LIBRARY_PATH", raising=False)
    optional_deps.ensure_svg_rendering()
    assert "DYLD_FALLBACK_LIBRARY_PATH" not in os.environ
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Utils/test_svg_rendering_dep.py -v`
Expected: FAIL / ERROR with `AttributeError: ... has no attribute '_svg_rendering_available'` (the autouse fixture fails first).

- [ ] **Step 3: Implement**

In `pyproject.toml`, directly after the `mcp` extra block (lines 121-123), insert:

```toml
svg = [
    "cairosvg",  # SVG rasterization for image attachments; needs the cairo C library (e.g. `brew install cairo`)
]
```

In `tldw_chatbook/Utils/optional_deps.py`:

(a) Add `import os` to the imports (after `import sys`).

(b) In the `DEPENDENCIES_AVAILABLE` dict (starts line 11), add after the `# PDF processing` group's entries:

```python
    # Image/SVG rendering
    'svg_rendering': False,
```

(c) After `check_dependency` (ends near line 408), add:

```python
# SVG rendering (cairosvg) — checked lazily, cached after the first call.
_svg_rendering_available: Optional[bool] = None

_HOMEBREW_LIB = '/opt/homebrew/lib'


def _ensure_homebrew_dyld_path() -> None:
    """Make Homebrew's libcairo findable by ctypes.util.find_library on macOS.

    dlopen reads DYLD_* at process start, but ctypes.util.find_library
    (cairocffi's fallback resolver) re-reads os.environ on every call, so an
    in-process append is sufficient. When the variable is unset, macOS uses a
    default fallback chain — reproduce it first so other libraries keep
    resolving exactly as before.
    """
    if not os.path.isdir(_HOMEBREW_LIB):
        return
    current = os.environ.get('DYLD_FALLBACK_LIBRARY_PATH')
    if current is None:
        default_chain = (
            os.path.expanduser('~/lib'), '/usr/local/lib', '/lib', '/usr/lib',
        )
        os.environ['DYLD_FALLBACK_LIBRARY_PATH'] = ':'.join(
            default_chain + (_HOMEBREW_LIB,)
        )
        return
    if _HOMEBREW_LIB not in current.split(':'):
        os.environ['DYLD_FALLBACK_LIBRARY_PATH'] = f"{current}:{_HOMEBREW_LIB}"


def ensure_svg_rendering() -> bool:
    """Return whether cairosvg-based SVG rasterization is available.

    Applies the macOS Homebrew dyld fix before the first import attempt.
    The result is cached for the life of the process.
    """
    global _svg_rendering_available
    if _svg_rendering_available is not None:
        return _svg_rendering_available
    if sys.platform == 'darwin':
        _ensure_homebrew_dyld_path()
    _svg_rendering_available = check_dependency('cairosvg', 'svg_rendering')
    return _svg_rendering_available
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Utils/test_svg_rendering_dep.py -v`
Expected: 7 passed.

- [ ] **Step 5: Sanity-check the real import on this machine**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -c "from tldw_chatbook.Utils.optional_deps import ensure_svg_rendering; print(ensure_svg_rendering())"`
Expected: `True` (cairo + cairosvg are installed on this machine).

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml tldw_chatbook/Utils/optional_deps.py Tests/Utils/test_svg_rendering_dep.py
git commit -m "feat(deps): cairosvg optional extra with cached availability check and macOS Homebrew dyld fix"
```

---

### Task 2: Policy functions, extended defaults, drift-by-construction

**Files:**
- Modify: `tldw_chatbook/Chat/attachment_core.py` (constants area lines 25-38; new functions after `_format_size`)
- Modify: `tldw_chatbook/config.py` (`CONFIG_TOML_CONTENT` template, the `supported_formats` line ≈2057)
- Modify: `tldw_chatbook/UI/Tools_Settings_Window.py:1890` (fallback literal)
- Test: `Tests/Chat/test_attachment_policy.py` (new file)

**Interfaces:**
- Consumes: `optional_deps.ensure_svg_rendering() -> bool` (Task 1); `get_cli_setting(section, key=None, default=None)` from `tldw_chatbook.config`.
- Produces (Tasks 3-4 rely on these exact names):
  - `DEFAULT_SUPPORTED_IMAGE_FORMATS: tuple[str, ...]`
  - `DEFAULT_RESIZE_MAX_DIMENSION: int = 2048`
  - `supported_image_formats() -> tuple[str, ...]`
  - `max_image_bytes() -> int`
  - `image_resize_max_dimension() -> int`
  - `svg_rendering_available() -> bool` (the seam tests monkeypatch)
  - `attachment_filter_specs() -> tuple[tuple[str, str], ...]`
- Note: the old `ATTACHMENT_FILTER_SPECS` constant stays in place this task (its three consumers are rewired and the constant deleted in Task 4).

- [ ] **Step 1: Write the failing tests**

Create `Tests/Chat/test_attachment_policy.py`:

```python
"""Policy-function and drift-by-construction tests (TASK-222)."""

import tomllib

import pytest

import tldw_chatbook.config as config_mod
from tldw_chatbook.Chat import attachment_core
from tldw_chatbook.Chat.attachment_core import (
    DEFAULT_RESIZE_MAX_DIMENSION,
    DEFAULT_SUPPORTED_IMAGE_FORMATS,
    MAX_IMAGE_BYTES,
    attachment_filter_specs,
    image_resize_max_dimension,
    max_image_bytes,
    supported_image_formats,
)


@pytest.fixture
def defaults_config(monkeypatch):
    """Simulate a config with no [chat.images] overrides (never read live config)."""
    monkeypatch.setattr(
        config_mod, "get_cli_setting",
        lambda section, key=None, default=None: default,
    )


@pytest.fixture
def svg_on(monkeypatch):
    monkeypatch.setattr(attachment_core, "svg_rendering_available", lambda: True)


@pytest.fixture
def config_override(monkeypatch):
    """Return a setter that overrides one [chat.images] key, defaults elsewhere."""

    def _set(key, value):
        monkeypatch.setattr(
            config_mod, "get_cli_setting",
            lambda section, k=None, default=None: (
                value if section == "chat.images" and k == key else default
            ),
        )

    return _set


class TestSupportedImageFormats:
    def test_defaults(self, defaults_config, svg_on):
        assert supported_image_formats() == DEFAULT_SUPPORTED_IMAGE_FORMATS

    def test_svg_dropped_when_unavailable(self, defaults_config, monkeypatch):
        monkeypatch.setattr(attachment_core, "svg_rendering_available", lambda: False)
        formats = supported_image_formats()
        assert ".svg" not in formats
        assert formats == tuple(
            f for f in DEFAULT_SUPPORTED_IMAGE_FORMATS if f != ".svg"
        )

    def test_normalization(self, config_override, svg_on):
        config_override(
            "supported_formats", ["PNG", "jpg", ".JPEG", "png", 42, "  .webp "]
        )
        assert supported_image_formats() == (".png", ".jpg", ".jpeg", ".webp")

    def test_invalid_value_falls_back_to_defaults(self, config_override, svg_on):
        config_override("supported_formats", "not-a-list")
        assert supported_image_formats() == DEFAULT_SUPPORTED_IMAGE_FORMATS

    def test_empty_list_falls_back_to_defaults(self, config_override, svg_on):
        config_override("supported_formats", [])
        assert supported_image_formats() == DEFAULT_SUPPORTED_IMAGE_FORMATS


class TestCaps:
    def test_max_image_bytes_default(self, defaults_config):
        assert max_image_bytes() == MAX_IMAGE_BYTES

    def test_max_image_bytes_override(self, config_override):
        config_override("max_size_mb", 2.5)
        assert max_image_bytes() == int(2.5 * 1024 * 1024)

    def test_max_image_bytes_invalid(self, config_override):
        config_override("max_size_mb", -3)
        assert max_image_bytes() == MAX_IMAGE_BYTES

    def test_max_image_bytes_non_numeric(self, config_override):
        config_override("max_size_mb", "lots")
        assert max_image_bytes() == MAX_IMAGE_BYTES

    def test_resize_dimension_default(self, defaults_config):
        assert image_resize_max_dimension() == DEFAULT_RESIZE_MAX_DIMENSION

    def test_resize_dimension_override(self, config_override):
        config_override("resize_max_dimension", 512)
        assert image_resize_max_dimension() == 512

    def test_resize_dimension_invalid(self, config_override):
        config_override("resize_max_dimension", 0)
        assert image_resize_max_dimension() == DEFAULT_RESIZE_MAX_DIMENSION


class TestFilterSpecsDrift:
    def test_image_row_derives_from_formats(self, defaults_config, svg_on):
        specs = attachment_filter_specs()
        expected = ";".join(f"*{ext}" for ext in supported_image_formats())
        assert specs[1] == ("Image Files", expected)

    def test_all_files_row_leads_with_image_patterns(self, defaults_config, svg_on):
        specs = attachment_filter_specs()
        image_patterns = ";".join(f"*{ext}" for ext in supported_image_formats())
        assert specs[0][0] == "All Supported Files"
        assert specs[0][1].startswith(image_patterns + ";")
        # non-image tail preserved verbatim from the legacy literal
        assert specs[0][1].endswith("*.epub;*.mobi;*.azw;*.azw3;*.fb2")

    def test_specs_follow_config_narrowing(self, config_override, svg_on):
        config_override("supported_formats", [".png"])
        specs = attachment_filter_specs()
        assert specs[1] == ("Image Files", "*.png")

    def test_non_image_rows_unchanged(self, defaults_config, svg_on):
        labels = [label for label, _ in attachment_filter_specs()]
        assert labels == [
            "All Supported Files", "Image Files", "Document Files",
            "E-book Files", "Text Files", "Code Files", "Data Files",
        ]


def test_config_template_matches_policy_default():
    """CONFIG_TOML_CONTENT's [chat.images].supported_formats == the policy default."""
    from tldw_chatbook.config import CONFIG_TOML_CONTENT

    parsed = tomllib.loads(CONFIG_TOML_CONTENT)
    assert parsed["chat"]["images"]["supported_formats"] == list(
        DEFAULT_SUPPORTED_IMAGE_FORMATS
    )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Chat/test_attachment_policy.py -v`
Expected: collection ERROR — `ImportError: cannot import name 'DEFAULT_RESIZE_MAX_DIMENSION'`.

- [ ] **Step 3: Implement `attachment_core.py` policy layer**

Add after line 27 (`DEFAULT_MAX_HISTORY_IMAGES = 10 ...`), leaving the existing `ATTACHMENT_FILTER_SPECS` constant (lines 29-38) in place for now:

```python
DEFAULT_RESIZE_MAX_DIMENSION = 2048  # matches ChatImageHandler's legacy literal

DEFAULT_SUPPORTED_IMAGE_FORMATS: tuple[str, ...] = (
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tiff", ".tif", ".svg",
)

# Non-image picker rows; the image rows are derived at call time by
# attachment_filter_specs(). The "All Supported Files" non-image tail is the
# legacy literal verbatim (it was never the union of the rows below — do not
# "fix" that here).
_ALL_FILES_NON_IMAGE_PATTERNS = (
    "*.txt;*.md;*.log;*.py;*.js;*.ts;*.java;*.cpp;*.c;*.h;*.cs;*.rb;*.go;*.rs;"
    "*.json;*.yaml;*.yml;*.csv;*.tsv;*.pdf;*.doc;*.docx;*.rtf;*.odt;"
    "*.epub;*.mobi;*.azw;*.azw3;*.fb2"
)
_NON_IMAGE_FILTER_SPECS: tuple[tuple[str, str], ...] = (
    ("Document Files", "*.pdf;*.doc;*.docx;*.rtf;*.odt"),
    ("E-book Files", "*.epub;*.mobi;*.azw;*.azw3;*.fb2"),
    ("Text Files", "*.txt;*.md;*.log;*.text;*.rst"),
    ("Code Files", "*.py;*.js;*.ts;*.java;*.cpp;*.c;*.h;*.cs;*.rb;*.go;*.rs;*.swift;*.kt;*.php;*.r;*.m;*.lua;*.sh;*.bash;*.ps1;*.sql;*.html;*.css;*.xml"),
    ("Data Files", "*.json;*.yaml;*.yml;*.csv;*.tsv"),
)


def svg_rendering_available() -> bool:
    """Capability seam for the SVG gate; tests monkeypatch this name."""
    from tldw_chatbook.Utils.optional_deps import ensure_svg_rendering

    return ensure_svg_rendering()


def supported_image_formats() -> tuple[str, ...]:
    """Effective image extension allowlist from [chat.images].supported_formats.

    Entries are normalized (lowercased, dotted, deduped in order); .svg is
    dropped when cairosvg is unavailable. Invalid or empty config values fall
    back to DEFAULT_SUPPORTED_IMAGE_FORMATS.
    """
    from tldw_chatbook.config import get_cli_setting

    raw = get_cli_setting(
        "chat.images", "supported_formats", list(DEFAULT_SUPPORTED_IMAGE_FORMATS)
    )
    formats: list[str] = []
    if isinstance(raw, (list, tuple)):
        for entry in raw:
            if not isinstance(entry, str) or not entry.strip():
                logger.warning(
                    f"[chat.images].supported_formats: ignoring entry {entry!r}"
                )
                continue
            ext = entry.strip().lower()
            if not ext.startswith("."):
                ext = f".{ext}"
            if ext not in formats:
                formats.append(ext)
    if not formats:
        logger.warning(
            "[chat.images].supported_formats invalid or empty; using defaults"
        )
        formats = list(DEFAULT_SUPPORTED_IMAGE_FORMATS)
    if ".svg" in formats and not svg_rendering_available():
        formats.remove(".svg")
    return tuple(formats)


def max_image_bytes() -> int:
    """Image byte cap from [chat.images].max_size_mb (default 10 MB)."""
    from tldw_chatbook.config import get_cli_setting

    raw = get_cli_setting("chat.images", "max_size_mb", MAX_IMAGE_BYTES / (1024 * 1024))
    try:
        value = float(raw)
    except (TypeError, ValueError):
        value = 0.0
    if value <= 0:
        logger.warning(f"[chat.images].max_size_mb invalid ({raw!r}); using 10.0")
        return MAX_IMAGE_BYTES
    return int(value * 1024 * 1024)


def image_resize_max_dimension() -> int:
    """Resize bound from [chat.images].resize_max_dimension (default 2048)."""
    from tldw_chatbook.config import get_cli_setting

    raw = get_cli_setting(
        "chat.images", "resize_max_dimension", DEFAULT_RESIZE_MAX_DIMENSION
    )
    try:
        value = int(raw)
    except (TypeError, ValueError):
        value = 0
    if value <= 0:
        logger.warning(
            f"[chat.images].resize_max_dimension invalid ({raw!r}); using "
            f"{DEFAULT_RESIZE_MAX_DIMENSION}"
        )
        return DEFAULT_RESIZE_MAX_DIMENSION
    return value


def attachment_filter_specs() -> tuple[tuple[str, str], ...]:
    """Picker filter rows with image patterns derived from the effective formats."""
    image_patterns = ";".join(f"*{ext}" for ext in supported_image_formats())
    return (
        ("All Supported Files", f"{image_patterns};{_ALL_FILES_NON_IMAGE_PATTERNS}"),
        ("Image Files", image_patterns),
        *_NON_IMAGE_FILTER_SPECS,
    )
```

- [ ] **Step 4: Extend the config template default**

In `tldw_chatbook/config.py` (inside `CONFIG_TOML_CONTENT`, line ≈2057), change:

```toml
supported_formats = [".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"]
```

to:

```toml
supported_formats = [".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tiff", ".tif", ".svg"]
```

- [ ] **Step 5: Point the Settings fallback at the policy default**

In `tldw_chatbook/UI/Tools_Settings_Window.py:1890`, change:

```python
        formats = chat_images_config.get("supported_formats", [".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"])
```

to:

```python
        from tldw_chatbook.Chat.attachment_core import DEFAULT_SUPPORTED_IMAGE_FORMATS
        formats = chat_images_config.get("supported_formats", list(DEFAULT_SUPPORTED_IMAGE_FORMATS))
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Chat/test_attachment_policy.py -v`
Expected: 17 passed.

- [ ] **Step 7: Run the neighboring suites**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Chat/test_attachment_core.py Tests/Chat/test_console_paste_attach.py -v`
Expected: all pass (nothing consumed the new functions yet).

- [ ] **Step 8: Commit**

```bash
git add tldw_chatbook/Chat/attachment_core.py tldw_chatbook/config.py tldw_chatbook/UI/Tools_Settings_Window.py Tests/Chat/test_attachment_policy.py
git commit -m "feat(chat): call-time attachment policy functions driven by [chat.images] config"
```

---

### Task 3: `prepare_image_payload` — SVG rasterize, payload-safe transcode, truthful mime

**Files:**
- Modify: `tldw_chatbook/Event_Handlers/Chat_Events/chat_image_events.py` (imports; module constants after the class docstring area; `process_image_file` lines 30-80; `_process_image_data` lines 82-129)
- Modify: `tldw_chatbook/Chat/attachment_core.py` (`process_attachment_bytes` lines 141-199)
- Test: `Tests/Chat/test_image_payload.py` (new file)

**Interfaces:**
- Consumes (Task 2): `supported_image_formats()`, `max_image_bytes()`, `image_resize_max_dimension()`, `svg_rendering_available()` from `tldw_chatbook.Chat.attachment_core`.
- Produces:
  - `chat_image_events.PAYLOAD_SAFE_FORMATS: set[str]` and `chat_image_events.PAYLOAD_FORMAT_MIME: dict[str, str]` (module level)
  - `ChatImageHandler.prepare_image_payload(image_data: bytes, extension: str) -> Tuple[bytes, str]` (async staticmethod; returned mime always matches returned bytes)
  - `ChatImageHandler._svg_raster_kwargs(svg_bytes: bytes, cap: int) -> dict` (staticmethod)
  - `_process_image_data` keeps its EXACT pinned signature, now a thin adapter.

- [ ] **Step 1: Write the failing tests**

Create `Tests/Chat/test_image_payload.py`:

```python
"""Payload pipeline: config-driven caps, tiff/svg support, truthful mime (TASK-222)."""

from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image as PILImage

import tldw_chatbook.config as config_mod
from tldw_chatbook.Chat import attachment_core
from tldw_chatbook.Event_Handlers.Chat_Events.chat_image_events import (
    PAYLOAD_FORMAT_MIME,
    ChatImageHandler,
)


def _svg_ready() -> bool:
    from tldw_chatbook.Utils.optional_deps import ensure_svg_rendering

    return ensure_svg_rendering()


svg_required = pytest.mark.skipif(not _svg_ready(), reason="cairosvg unavailable")

SVG_RED_RECT = (
    b'<svg xmlns="http://www.w3.org/2000/svg" width="40" height="20">'
    b'<rect width="40" height="20" fill="red"/></svg>'
)


@pytest.fixture
def defaults_config(monkeypatch):
    """Simulate a config with no [chat.images] overrides (never read live config)."""
    monkeypatch.setattr(
        config_mod, "get_cli_setting",
        lambda section, key=None, default=None: default,
    )


def _write_image(tmp_path: Path, name: str, fmt: str, size=(64, 64), mode="RGB") -> Path:
    path = tmp_path / name
    PILImage.new(mode, size, color="red").save(path, format=fmt)
    return path


@pytest.mark.asyncio
async def test_tiff_end_to_end_transcodes_to_png(tmp_path, defaults_config):
    tiff = _write_image(tmp_path, "photo.tiff", "TIFF")
    data, mime = await ChatImageHandler.process_image_file(str(tiff))
    assert mime == "image/png"
    assert PILImage.open(BytesIO(data)).format == "PNG"


@pytest.mark.asyncio
async def test_small_bmp_transcodes_to_png(tmp_path, defaults_config):
    bmp = _write_image(tmp_path, "icon.bmp", "BMP")
    data, mime = await ChatImageHandler.process_image_file(str(bmp))
    assert mime == "image/png"
    assert PILImage.open(BytesIO(data)).format == "PNG"


@pytest.mark.asyncio
async def test_small_png_passes_through_unchanged(tmp_path, defaults_config):
    png = _write_image(tmp_path, "pic.png", "PNG")
    original = png.read_bytes()
    data, mime = await ChatImageHandler.process_image_file(str(png))
    assert data == original
    assert mime == "image/png"


@pytest.mark.asyncio
async def test_large_gif_resizes_with_matching_mime(tmp_path, defaults_config):
    gif = _write_image(tmp_path, "big.gif", "GIF", size=(3000, 1500))
    data, mime = await ChatImageHandler.process_image_file(str(gif))
    img = PILImage.open(BytesIO(data))
    assert max(img.size) <= 2048
    assert PAYLOAD_FORMAT_MIME[img.format] == mime  # mime matches actual bytes


@pytest.mark.asyncio
async def test_cmyk_tiff_transcodes_without_crash(tmp_path, defaults_config):
    tiff = _write_image(tmp_path, "print.tiff", "TIFF", mode="CMYK")
    data, mime = await ChatImageHandler.process_image_file(str(tiff))
    assert mime == "image/png"
    assert PILImage.open(BytesIO(data)).format == "PNG"


@pytest.mark.asyncio
async def test_custom_resize_dimension_honored(tmp_path, monkeypatch):
    monkeypatch.setattr(
        config_mod, "get_cli_setting",
        lambda section, key=None, default=None: (
            512 if key == "resize_max_dimension" else default
        ),
    )
    png = _write_image(tmp_path, "wide.png", "PNG", size=(1024, 800))
    data, _mime = await ChatImageHandler.process_image_file(str(png))
    assert max(PILImage.open(BytesIO(data)).size) <= 512


@pytest.mark.asyncio
async def test_custom_size_cap_rejects_through_real_path(tmp_path, monkeypatch):
    monkeypatch.setattr(
        config_mod, "get_cli_setting",
        lambda section, key=None, default=None: (
            0.0001 if key == "max_size_mb" else default
        ),
    )
    png = _write_image(tmp_path, "pic.png", "PNG")
    with pytest.raises(ValueError, match="too large"):
        await ChatImageHandler.process_image_file(str(png))


@pytest.mark.asyncio
async def test_custom_formats_reject_through_real_path(tmp_path, monkeypatch):
    monkeypatch.setattr(
        config_mod, "get_cli_setting",
        lambda section, key=None, default=None: (
            [".png"] if key == "supported_formats" else default
        ),
    )
    gif = _write_image(tmp_path, "anim.gif", "GIF")
    with pytest.raises(ValueError, match="Unsupported image format"):
        await ChatImageHandler.process_image_file(str(gif))


@pytest.mark.asyncio
async def test_process_attachment_bytes_mime_matches_bytes(defaults_config):
    buffer = BytesIO()
    PILImage.new("RGB", (32, 32), "blue").save(buffer, format="PNG")
    pending = await attachment_core.process_attachment_bytes(
        buffer.getvalue(), display_name="clip.png", mime_type="image/png"
    )
    assert pending.mime_type == "image/png"
    assert pending.data is not None


@pytest.mark.asyncio
async def test_process_attachment_bytes_fallback_probes_mime(defaults_config, monkeypatch):
    async def boom(*args, **kwargs):
        raise RuntimeError("simulated processing failure")

    monkeypatch.setattr(ChatImageHandler, "prepare_image_payload", boom)
    buffer = BytesIO()
    PILImage.new("RGB", (32, 32), "blue").save(buffer, format="PNG")
    pending = await attachment_core.process_attachment_bytes(
        buffer.getvalue(), display_name="clip.png", mime_type="image/jpeg"  # caller lies
    )
    assert pending.data == buffer.getvalue()  # fallback keeps original bytes
    assert pending.mime_type == "image/png"  # probed from the actual bytes


@pytest.mark.asyncio
async def test_svg_rejected_when_capability_absent(tmp_path, monkeypatch, defaults_config):
    monkeypatch.setattr(attachment_core, "svg_rendering_available", lambda: False)
    svg = tmp_path / "logo.svg"
    svg.write_bytes(SVG_RED_RECT)
    with pytest.raises(ValueError, match="Unsupported image format"):
        await ChatImageHandler.process_image_file(str(svg))


@svg_required
@pytest.mark.asyncio
async def test_svg_end_to_end_rasterizes_to_png(tmp_path, defaults_config):
    svg = tmp_path / "logo.svg"
    svg.write_bytes(SVG_RED_RECT)
    data, mime = await ChatImageHandler.process_image_file(str(svg))
    assert mime == "image/png"
    img = PILImage.open(BytesIO(data))
    assert img.format == "PNG"
    assert img.size == (40, 20)


@svg_required
@pytest.mark.asyncio
async def test_svg_oversize_declaration_is_bounded(tmp_path, defaults_config):
    svg = tmp_path / "bomb.svg"
    svg.write_bytes(
        b'<svg xmlns="http://www.w3.org/2000/svg" width="100000" height="100000">'
        b'<rect width="100000" height="100000" fill="red"/></svg>'
    )
    data, _mime = await ChatImageHandler.process_image_file(str(svg))
    assert max(PILImage.open(BytesIO(data)).size) <= 2048


@svg_required
@pytest.mark.asyncio
async def test_svg_xml_entities_rejected(tmp_path, defaults_config):
    svg = tmp_path / "xxe.svg"
    svg.write_bytes(
        b'<?xml version="1.0"?>'
        b'<!DOCTYPE svg [<!ENTITY x SYSTEM "file:///etc/hosts">]>'
        b'<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10">'
        b"<text>&x;</text></svg>"
    )
    with pytest.raises(ValueError, match="Could not render SVG"):
        await ChatImageHandler.process_image_file(str(svg))


@svg_required
@pytest.mark.asyncio
async def test_svg_viewbox_only_preserves_aspect(tmp_path, defaults_config):
    svg = tmp_path / "vb.svg"
    svg.write_bytes(
        b'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 300 150">'
        b'<rect width="300" height="150" fill="blue"/></svg>'
    )
    data, _mime = await ChatImageHandler.process_image_file(str(svg))
    assert PILImage.open(BytesIO(data)).size == (300, 150)


@svg_required
@pytest.mark.asyncio
async def test_svg_unparseable_aspect_hard_bounds(tmp_path, defaults_config):
    svg = tmp_path / "pct.svg"
    svg.write_bytes(
        b'<svg xmlns="http://www.w3.org/2000/svg" width="100%" height="100%">'
        b'<rect width="10" height="10" fill="red"/></svg>'
    )
    data, _mime = await ChatImageHandler.process_image_file(str(svg))
    assert PILImage.open(BytesIO(data)).size == (2048, 2048)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Chat/test_image_payload.py -v`
Expected: collection ERROR — `ImportError: cannot import name 'PAYLOAD_FORMAT_MIME'`.

- [ ] **Step 3: Implement `chat_image_events.py`**

(a) Add `import re` to the standard-library imports (after `import mimetypes`).

(b) Add module constants immediately BEFORE `class ChatImageHandler:`:

```python
# Formats vision providers accept as payloads; anything else transcodes to PNG.
PAYLOAD_SAFE_FORMATS = {"PNG", "JPEG", "WEBP", "GIF"}
PAYLOAD_FORMAT_MIME = {
    "PNG": "image/png",
    "JPEG": "image/jpeg",
    "WEBP": "image/webp",
    "GIF": "image/gif",
}
```

(c) Replace the BODY of `process_image_file` (keep the decorator, signature, and docstring; replace everything from `path = Path(...)` through the final `return image_data, mime_type`) with:

```python
        path = Path(file_path).expanduser().resolve()

        # Validate file exists
        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {file_path}")

        from tldw_chatbook.Chat.attachment_core import (
            max_image_bytes,
            supported_image_formats,
        )

        # Check file extension against the effective (config-driven) allowlist
        effective_formats = supported_image_formats()
        if path.suffix.lower() not in effective_formats:
            raise ValueError(
                f"Unsupported image format: {path.suffix}. "
                f"Supported formats: {', '.join(effective_formats)}"
            )

        # Check file size against the config-driven cap
        file_size = path.stat().st_size
        size_cap = max_image_bytes()
        if file_size > size_cap:
            raise ValueError(
                f"Image file too large ({file_size / 1024 / 1024:.1f}MB). "
                f"Maximum size: {size_cap / 1024 / 1024}MB"
            )

        image_data = path.read_bytes()
        extension = path.suffix.lower()

        if extension == '.svg':
            # No usable fallback for un-rasterized SVG bytes — errors reject.
            return await ChatImageHandler.prepare_image_payload(image_data, extension)

        mime_type = mimetypes.guess_type(str(path))[0] or 'image/png'
        try:
            return await ChatImageHandler.prepare_image_payload(image_data, extension)
        except Exception as e:
            logging.warning(f"Failed to process image, using original: {e}")
            # If processing fails, use original data
            return image_data, mime_type
```

(d) Add two new staticmethods to `ChatImageHandler` (place them directly after `process_image_file`):

```python
    @staticmethod
    def _svg_raster_kwargs(svg_bytes: bytes, cap: int) -> dict:
        """Bounded svg2png output kwargs: longer rendered side ≤ cap.

        The byte-size cap guards the SVG *source* only; a small file can
        declare a huge canvas, and cairo allocates that surface during
        render — before any PIL bomb guard runs. Aspect comes from viewBox,
        else numeric width/height; an unparseable aspect falls back to a
        both-dims hard bound (may distort that degenerate case; logged).
        """
        intrinsic = None  # (width, height) in user units
        try:
            from defusedxml import ElementTree as SafeET  # ships with cairosvg

            root = SafeET.fromstring(svg_bytes)
            view_box = root.get("viewBox")
            if view_box:
                parts = view_box.replace(",", " ").split()
                if len(parts) == 4:
                    vb_w, vb_h = float(parts[2]), float(parts[3])
                    if vb_w > 0 and vb_h > 0:
                        intrinsic = (vb_w, vb_h)
            if intrinsic is None:
                w_attr, h_attr = root.get("width"), root.get("height")
                if w_attr and h_attr:
                    w = float(re.sub(r"px\s*$", "", w_attr.strip(), count=1))
                    h = float(re.sub(r"px\s*$", "", h_attr.strip(), count=1))
                    if w > 0 and h > 0:
                        intrinsic = (w, h)
        except Exception:
            intrinsic = None
        if intrinsic is None:
            logging.warning(
                "SVG has no parseable aspect; rasterizing with a hard "
                f"{cap}x{cap} bound"
            )
            return {"output_width": cap, "output_height": cap}
        w, h = intrinsic
        target = max(1, int(round(min(max(w, h), cap))))
        if w >= h:
            return {"output_width": target}
        return {"output_height": target}

    @staticmethod
    async def prepare_image_payload(image_data: bytes, extension: str) -> Tuple[bytes, str]:
        """Normalize image bytes for provider payloads.

        SVG rasterizes to PNG first (bounded — see _svg_raster_kwargs);
        rasters larger than [chat.images].resize_max_dimension shrink;
        anything outside PAYLOAD_SAFE_FORMATS transcodes to PNG. The
        returned mime always matches the returned bytes.

        Args:
            image_data: Raw image bytes.
            extension: Lowercased dotted extension ('' when unknown, e.g.
                clipboard bytes — the SVG branch then never triggers).

        Returns:
            Tuple of (payload_bytes, mime_type).

        Raises:
            ValueError: If SVG rendering is required but unavailable, or the
                SVG cannot be rendered.
        """
        from tldw_chatbook.Chat.attachment_core import (
            image_resize_max_dimension,
            svg_rendering_available,
        )

        if extension == '.svg':
            if not svg_rendering_available():
                raise ValueError(
                    "SVG attachments require the optional cairosvg dependency "
                    "(pip install tldw_chatbook[svg])."
                )
            import cairosvg

            kwargs = ChatImageHandler._svg_raster_kwargs(
                image_data, image_resize_max_dimension()
            )
            try:
                # NOTE: cairosvg's `unsafe` parameter stays at its default
                # (False): XML entities are hard-blocked and external file
                # references are not read. Never pass unsafe=True.
                image_data = cairosvg.svg2png(bytestring=image_data, **kwargs)
            except Exception as exc:
                raise ValueError(f"Could not render SVG: {exc}") from exc

        pil_image = PILImage.open(BytesIO(image_data))
        actual_format = (pil_image.format or "").upper()
        max_dimension = image_resize_max_dimension()
        needs_resize = (
            pil_image.width > max_dimension or pil_image.height > max_dimension
        )
        needs_transcode = actual_format not in PAYLOAD_SAFE_FORMATS
        if not needs_resize and not needs_transcode:
            return image_data, PAYLOAD_FORMAT_MIME[actual_format]

        if needs_resize:
            pil_image.thumbnail(
                (max_dimension, max_dimension), PILImage.Resampling.LANCZOS
            )
        if actual_format == 'JPEG':
            save_format, save_kwargs = 'JPEG', {'optimize': True, 'quality': 85}
        elif actual_format == 'WEBP':
            save_format, save_kwargs = 'WEBP', {'quality': 85}
        else:
            # PNG stays PNG; GIF and every non-payload-safe format transcode
            # to PNG (the legacy else-branch, now with a truthful mime).
            save_format, save_kwargs = 'PNG', {'optimize': True}
        if save_format == 'PNG' and pil_image.mode not in (
            '1', 'L', 'LA', 'I', 'P', 'RGB', 'RGBA'
        ):
            pil_image = pil_image.convert('RGB')  # e.g. CMYK — PNG can't encode it
        buffer = BytesIO()
        pil_image.save(buffer, format=save_format, **save_kwargs)
        return buffer.getvalue(), PAYLOAD_FORMAT_MIME[save_format]
```

(e) Replace `_process_image_data`'s body (keep the EXACT signature `async def _process_image_data(image_data: bytes, extension: str, mime_type: str) -> bytes:` and its decorator) with:

```python
        """Legacy adapter: bytes-only view of prepare_image_payload.

        Signature and return shape are pinned by existing callers and tests;
        mime-aware callers use prepare_image_payload directly.
        """
        processed_data, _mime = await ChatImageHandler.prepare_image_payload(
            image_data, extension
        )
        return processed_data
```

- [ ] **Step 4: Rewire `attachment_core.process_attachment_bytes`**

Replace the function body from the local imports through the `return PendingAttachment(...)` (lines 164-199) with:

```python
    from io import BytesIO

    from PIL import Image as PILImage

    from tldw_chatbook.Event_Handlers.Chat_Events.chat_image_events import (
        PAYLOAD_FORMAT_MIME,
        ChatImageHandler,
    )

    size_cap = max_image_bytes()
    if len(data) > size_cap:
        raise ValueError(
            f"Image too large ({len(data) / 1024 / 1024:.1f}MB). "
            f"Maximum size: {size_cap / 1024 / 1024:.0f}MB"
        )
    try:
        probe = PILImage.open(BytesIO(data))
        probe.verify()
        probed_format = (probe.format or "").upper()
    except Exception as exc:
        raise ValueError("Clipboard data is not a valid image.") from exc
    extension = ".png" if "png" in mime_type else ".jpg"
    try:
        processed, mime_type = await ChatImageHandler.prepare_image_payload(
            data, extension
        )
    except Exception:
        logger.opt(exception=True).warning(
            "Failed to process clipboard image data, using original bytes."
        )
        processed = data
        mime_type = PAYLOAD_FORMAT_MIME.get(probed_format, mime_type)
    return PendingAttachment(
        file_path="",
        display_name=display_name,
        file_type="image",
        insert_mode="attachment",
        data=processed,
        mime_type=mime_type,
        text_content=None,
        original_size=len(data),
        processed_size=len(processed),
    )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Chat/test_image_payload.py -v`
Expected: 16 passed (all SVG tests run — cairosvg works on this machine).

- [ ] **Step 6: Run the legacy regression gate (ZERO edits allowed)**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Event_Handlers/Chat_Events/test_chat_image_events.py Tests/Event_Handlers/Chat_Events/test_chat_image_properties.py Tests/UI/test_chat_image_attachment.py Tests/unit/test_chat_image_unit.py -v`
Expected: all pass. If ANY gate test fails, fix the implementation — do NOT touch the test files; if you believe a gate test itself must change, STOP and report BLOCKED.

- [ ] **Step 7: Run the attachment-core suite**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Chat/test_attachment_core.py Tests/Chat/test_attachment_policy.py -v`
Expected: all pass.

- [ ] **Step 8: Commit**

```bash
git add tldw_chatbook/Event_Handlers/Chat_Events/chat_image_events.py tldw_chatbook/Chat/attachment_core.py Tests/Chat/test_image_payload.py
git commit -m "feat(chat): prepare_image_payload — bounded SVG rasterize, payload-safe transcode, truthful mime"
```

---

### Task 4: Call-time routing and picker rewires; retire `ATTACHMENT_FILTER_SPECS`

**Files:**
- Modify: `tldw_chatbook/Utils/file_handlers.py:56-62` (ImageFileHandler)
- Modify: `tldw_chatbook/Chat/console_paste_attach.py` (import line 24; `_SUPPORTED_PATTERNS` lines 27-31; `looks_attachable` line 159)
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py:7757,7769` (picker)
- Modify: `tldw_chatbook/UI/Chat_Modules/chat_attachment_handler.py:77,80` (picker)
- Modify: `tldw_chatbook/Chat/attachment_core.py:29-38` (delete the old constant)
- Test: `Tests/Chat/test_attachment_routing.py` (new file)

**Interfaces:**
- Consumes (Task 2): `supported_image_formats()`, `attachment_filter_specs()`, `svg_rendering_available` (monkeypatch target) from `tldw_chatbook.Chat.attachment_core`.
- Produces: `console_paste_attach._supported_patterns() -> tuple[str, ...]`. Deletes: `attachment_core.ATTACHMENT_FILTER_SPECS`, `ImageFileHandler.SUPPORTED_EXTENSIONS` (verified: no test references either).

- [ ] **Step 1: Write the failing tests**

Create `Tests/Chat/test_attachment_routing.py`:

```python
"""Routing/picker drift-by-construction tests (TASK-222)."""

from pathlib import Path

import pytest

import tldw_chatbook.config as config_mod
from tldw_chatbook.Chat import attachment_core, console_paste_attach
from tldw_chatbook.Utils.file_handlers import ImageFileHandler


@pytest.fixture
def defaults_config(monkeypatch):
    """Simulate a config with no [chat.images] overrides (never read live config)."""
    monkeypatch.setattr(
        config_mod, "get_cli_setting",
        lambda section, key=None, default=None: default,
    )


def test_routing_matches_effective_formats(defaults_config, monkeypatch):
    monkeypatch.setattr(attachment_core, "svg_rendering_available", lambda: True)
    handler = ImageFileHandler()
    for ext in attachment_core.supported_image_formats():
        assert handler.can_handle(Path(f"pic{ext}")) is True
    assert handler.can_handle(Path("pic.xcf")) is False


def test_routing_respects_config_narrowing(monkeypatch):
    monkeypatch.setattr(
        config_mod, "get_cli_setting",
        lambda section, key=None, default=None: (
            [".png"] if key == "supported_formats" else default
        ),
    )
    handler = ImageFileHandler()
    assert handler.can_handle(Path("pic.png")) is True
    assert handler.can_handle(Path("pic.gif")) is False


def test_svg_routing_gated_by_capability(defaults_config, monkeypatch):
    handler = ImageFileHandler()
    monkeypatch.setattr(attachment_core, "svg_rendering_available", lambda: False)
    assert handler.can_handle(Path("logo.svg")) is False
    monkeypatch.setattr(attachment_core, "svg_rendering_available", lambda: True)
    assert handler.can_handle(Path("logo.svg")) is True


def test_looks_attachable_follows_effective_formats(tmp_path, defaults_config, monkeypatch):
    monkeypatch.setattr(console_paste_attach, "is_safe_path", lambda p, r: True)
    svg = tmp_path / "logo.svg"
    svg.write_text("<svg/>")
    monkeypatch.setattr(attachment_core, "svg_rendering_available", lambda: False)
    assert console_paste_attach.looks_attachable(str(svg)) is False
    monkeypatch.setattr(attachment_core, "svg_rendering_available", lambda: True)
    assert console_paste_attach.looks_attachable(str(svg)) is True


def test_looks_attachable_still_takes_tiff(tmp_path, defaults_config, monkeypatch):
    monkeypatch.setattr(console_paste_attach, "is_safe_path", lambda p, r: True)
    tiff = tmp_path / "scan.tiff"
    tiff.write_bytes(b"II*\x00")
    assert console_paste_attach.looks_attachable(str(tiff)) is True


def test_old_module_constants_are_gone():
    assert not hasattr(attachment_core, "ATTACHMENT_FILTER_SPECS")
    assert not hasattr(ImageFileHandler, "SUPPORTED_EXTENSIONS")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Chat/test_attachment_routing.py -v`
Expected: `test_old_module_constants_are_gone`, `test_svg_routing_gated_by_capability`, `test_routing_respects_config_narrowing`, and `test_looks_attachable_follows_effective_formats` FAIL (constants still exist; routing/patterns still static supersets). The two remaining tests pass already (the old superset covers the default formats) — that's expected.

- [ ] **Step 3: Rewire `ImageFileHandler`**

In `tldw_chatbook/Utils/file_handlers.py`, replace lines 56-62 (the `SUPPORTED_EXTENSIONS` attribute and `can_handle`) so the class starts:

```python
class ImageFileHandler(FileHandler):
    """Handler for image files - maintains existing functionality."""

    def can_handle(self, file_path: Path) -> bool:
        # Lazy import: attachment_core imports this module at module level
        # (same cycle-avoidance as the lazy import in process() below).
        from ..Chat.attachment_core import supported_image_formats

        return file_path.suffix.lower() in supported_image_formats()
```

(`process()` stays unchanged.)

- [ ] **Step 4: Rewire `console_paste_attach`**

In `tldw_chatbook/Chat/console_paste_attach.py`:

(a) Change line 24 from `from tldw_chatbook.Chat.attachment_core import ATTACHMENT_FILTER_SPECS` to:

```python
from tldw_chatbook.Chat.attachment_core import attachment_filter_specs
```

(b) Replace the `_SUPPORTED_PATTERNS` module constant (lines 27-31) with:

```python
def _supported_patterns() -> tuple[str, ...]:
    """Glob patterns for attachable files, from the call-time picker specs."""
    return tuple(
        pattern
        for _label, patterns in attachment_filter_specs()
        for pattern in patterns.split(";")
    )
```

(c) In `looks_attachable` (line 159), change:

```python
    return any(fnmatch(name, pattern) for pattern in _SUPPORTED_PATTERNS)
```

to:

```python
    return any(fnmatch(name, pattern) for pattern in _supported_patterns())
```

- [ ] **Step 5: Rewire both pickers**

In `tldw_chatbook/UI/Screens/chat_screen.py` (`_handle_console_attach_context`): change line 7757 `from tldw_chatbook.Chat.attachment_core import ATTACHMENT_FILTER_SPECS` to `from tldw_chatbook.Chat.attachment_core import attachment_filter_specs`, and line 7769 to:

```python
            *[(label, create_filter(patterns)) for label, patterns in attachment_filter_specs()],
```

In `tldw_chatbook/UI/Chat_Modules/chat_attachment_handler.py`: change line 77 `from ...Chat.attachment_core import ATTACHMENT_FILTER_SPECS` to `from ...Chat.attachment_core import attachment_filter_specs`, and line 80 to:

```python
            *[(label, create_filter(patterns)) for label, patterns in attachment_filter_specs()],
```

- [ ] **Step 6: Delete the old constant**

In `tldw_chatbook/Chat/attachment_core.py`, delete the entire `ATTACHMENT_FILTER_SPECS` tuple (the original lines 29-38, starting with the `# (label, semicolon-separated glob patterns)` comment).

- [ ] **Step 7: Verify no stragglers**

Run: `git grep -nE "ATTACHMENT_FILTER_SPECS|_SUPPORTED_PATTERNS" -- tldw_chatbook Tests`
Expected: no output (the new function is lowercase `_supported_patterns`, so the uppercase names must be fully gone).

- [ ] **Step 8: Run tests to verify they pass**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Chat/test_attachment_routing.py -v`
Expected: 6 passed.

- [ ] **Step 9: Run the full affected suites + legacy gate**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Chat/ Tests/Utils/test_svg_rendering_dep.py Tests/Event_Handlers/Chat_Events/test_chat_image_events.py Tests/Event_Handlers/Chat_Events/test_chat_image_properties.py Tests/UI/test_chat_image_attachment.py Tests/unit/test_chat_image_unit.py -v`
Expected: all pass (fix any failures in implementation code only).

- [ ] **Step 10: Verify existing test files untouched**

Run: `git diff origin/dev --diff-filter=M --name-only -- Tests/`
Expected: no output (only NEW test files exist under Tests/).

- [ ] **Step 11: Commit**

```bash
git add tldw_chatbook/Utils/file_handlers.py tldw_chatbook/Chat/console_paste_attach.py tldw_chatbook/UI/Screens/chat_screen.py tldw_chatbook/UI/Chat_Modules/chat_attachment_handler.py tldw_chatbook/Chat/attachment_core.py Tests/Chat/test_attachment_routing.py
git commit -m "feat(chat): call-time routing and picker filters from attachment policy; retire static filter specs"
```

---

## After the plan (not plan tasks)

Final whole-branch review → live QA (textual-serve captures: extended picker rows, `.tiff` attach→chip→send, `.svg` attach rasterized) → user screenshot gate → PR to dev. Release note: existing configs pin the old 6-format list; enabling tiff/svg is a one-line `supported_formats` edit (config.toml or Settings → Chat Images).
