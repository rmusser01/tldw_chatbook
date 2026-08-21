# tldw_chatbook/Chunking/_shims/config.py
"""Replaces tldw_Server_API.app.core.config for the vendored engine (spec §5.3).

The engine reads chunking toggles via load_comprehensive_config() (a
config-parser-like object whose section is named 'Chunking' -- see the
has_section('Chunking') guards in engine/base.py, chunker.py, regex_safety.py,
strategies/json_xml.py and strategies/ebook_chapters.py) and
load_and_log_configs() (a dict). Both delegate to chatbook's TOML config.

Deviation from the phase-A brief (documented in task-2-report.md):
``get_cli_setting("chunking", None, None)`` cannot fetch a whole section --
with ``key=None`` and no dot in the section name, config.py returns the
default unconditionally (config.py:5670-5672). The section is therefore read
directly from ``load_cli_config_and_ensure_existence()``, the same merged
dict chatbook's own ``load_settings`` reads via ``get_toml_section("Chunking")``
(config.py:1275). Both the capitalised ``[Chunking]`` table (chatbook's
existing convention, which the engine's lookups also use) and a lowercase
``[chunking]`` table are accepted and merged.
"""
import configparser
from typing import Any, Dict

from ...config import load_cli_config_and_ensure_existence


def _chunking_section() -> Dict[str, Any]:
    """Return chatbook's chunking TOML table as a flat dict (possibly empty)."""
    config = load_cli_config_and_ensure_existence()
    merged: Dict[str, Any] = {}
    # [chunking] first, then [Chunking]: the capitalized table is applied
    # last, so it wins on key conflict (review M-4).
    for name in ("chunking", "Chunking"):
        section = config.get(name)
        if isinstance(section, dict):
            merged.update(section)
    return merged


class _ChunkingConfigParser(configparser.ConfigParser):
    """Parser-like view over chatbook's [Chunking] TOML section.

    When the user has no such table the parser stays empty, so every engine
    ``has_section('Chunking')`` guard is False and engine defaults apply.
    """

    def __init__(self) -> None:
        # interpolation=None: raw pass-through of '%' in values. With the
        # default BasicInterpolation, any '%' in a [Chunking]/[chunking]
        # TOML value raised ValueError("invalid interpolation syntax") here
        # at construction time (review I-1) — the engine catches only
        # ImportError around this call, so every Chunker construction died,
        # even for keys the engine never reads. '%(name)s'-style values
        # previously raised InterpolationMissingOptionError at .get() time,
        # which no engine exception tuple covers.
        super().__init__(interpolation=None)
        chunking = _chunking_section()
        if chunking:
            self.read_dict({"Chunking": {str(k): str(v) for k, v in chunking.items()}})


def load_comprehensive_config() -> _ChunkingConfigParser:
    return _ChunkingConfigParser()


def load_and_log_configs() -> Dict[str, Any]:
    return {"chunking_config": {k: v for k, v in _chunking_section().items()
            if isinstance(v, (str, int, float, bool))}}
