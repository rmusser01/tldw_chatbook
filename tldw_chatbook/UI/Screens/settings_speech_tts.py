"""Compatibility alias: this module moved to ``tldw_chatbook.UI.Speech.settings_speech_tts``.

TASK-36944.8: Widgets must not import ``UI/Screens``, so this code moved out
of the Screens package. The old dotted path aliases the new module object, so
``monkeypatch.setattr`` on either path hits the same namespace. Delete this
file once no importer uses the old path.
"""

import sys
from importlib import import_module

sys.modules[__name__] = import_module("tldw_chatbook.UI.Speech.settings_speech_tts")
