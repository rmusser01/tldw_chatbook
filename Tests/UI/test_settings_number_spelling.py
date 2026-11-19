"""Settings spells a number one way (review of PR #3057, R1 #29; ruling C6).

The Inspector's "Next new chat will use" line wrote a temperature of one as
"T 1" while the Model defaults row beside it said "inherits 1.0". Both go
through ``format_value`` now. The Console's own summary line is not Settings'.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tldw_chatbook.UI.Settings_Modules import providers_models_card as card
from tldw_chatbook.UI.Settings_Modules import settings_field_rows as rows

# Importing the card imports the Settings screen, whose config-participant
# admission the per-test sandbox refuses; keep the collection-time profile.
pytestmark = pytest.mark.bootstrap_profile


@pytest.mark.parametrize(
    ("temperature", "spelled"), [(1, "1.0"), (0.4, "0.4"), (2.0, "2.0")]
)
def test_the_inspector_and_the_row_spell_a_temperature_alike(temperature, spelled):
    config = {
        "chat_defaults": {
            "provider": "openai",
            "model": "gpt-4.1",
            "temperature": temperature,
            "max_tokens": 2048,
            "streaming": False,
        },
        "api_settings": {"openai": {}},
    }
    screen = SimpleNamespace(
        _app_config_mapping=lambda: config,
        _provider_display_label=lambda provider: str(provider),
    )

    _pair, values = card.next_new_chat_lines(screen)
    inherited = rows.inherited_values(config, "openai", "gpt-4.1")
    _word, help_line = rows.row_copy("temperature", "", False, inherited["temperature"])

    assert help_line == f"inherits {spelled}"
    assert values == f"T {spelled} · max 2048 · stream Off"
