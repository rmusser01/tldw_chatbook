"""task-21124: `set_buffer_duration` must not write config.toml itself.

`Dictation_Window_Improved.on_input_changed` calls
`LazyLiveDictationService.set_buffer_duration` once per parsing keystroke of
the buffer-duration input, and `_initialize_service` calls it once on every
service init. The service used to run a synchronous
`save_setting_to_cli_config` inside it -- a full config read-rewrite-reload
cycle per keystroke on the event loop, duplicating the persistence the
owning widget already batches through its debounced task-15470 settings
snapshot (which `Tests/UI/test_dictation_settings_debounce.py` pins,
`buffer_duration_ms` included).

These tests pin the service side: the in-memory value updates (with the
documented clamp) and the config file is not touched.
"""

from __future__ import annotations

from tldw_chatbook.Audio.dictation_service_lazy import LazyLiveDictationService
from tldw_chatbook.config import _get_effective_config_path


def _config_file_fingerprint() -> tuple[bool, bytes]:
    config_path = _get_effective_config_path()
    if not config_path.exists():
        return (False, b"")
    return (True, config_path.read_bytes())


def test_set_buffer_duration_does_not_write_config():
    service = LazyLiveDictationService()
    before = _config_file_fingerprint()

    service.set_buffer_duration(300)

    assert service.buffer_duration_ms == 300
    assert _config_file_fingerprint() == before, (
        "set_buffer_duration wrote to config.toml; persistence belongs to "
        "the owning widget's debounced batched snapshot (task-21124)"
    )


def test_set_buffer_duration_clamps_without_config_write():
    service = LazyLiveDictationService()
    before = _config_file_fingerprint()

    service.set_buffer_duration(50)
    assert service.buffer_duration_ms == 100

    service.set_buffer_duration(99999)
    assert service.buffer_duration_ms == 2000

    assert _config_file_fingerprint() == before
