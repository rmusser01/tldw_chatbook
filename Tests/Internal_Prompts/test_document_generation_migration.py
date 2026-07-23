# Tests/Internal_Prompts/test_document_generation_migration.py
"""Overrides must reach the document-generation LLM payloads; legacy TOML
customization still outranks the shipped default. Fakes only at the
provider-dispatch seam inside `_call_llm` (`self.provider_functions[<name>]`)
so `generate_timeline`/`generate_study_guide`/`generate_briefing`'s real
prompt-assembly code runs for real and the assembled `messages` list (built
at document_generator.py:506-509) is captured.

Adaptations from the brief's skeleton:

- `DocumentGenerator.__init__(db_path, client_id="document_generator")`
  constructs a real `CharactersRAGDB`. An in-memory DB
  (`DocumentGenerator(":memory:", "test-client")`) is cheap and matches the
  project's established in-memory-DB test pattern (see
  Tests/UI/test_home_screen.py, Tests/UI/test_library_shell.py) — no faking
  needed for `__init__` itself, and `__init__` reads config for the
  temperature/max_tokens dicts, so it must run AFTER `scratch_config(...)`
  writes (per the brief).
- The brief's target seam ("the actual outbound call around ~507 where
  messages are assembled and sent") is `_call_llm`'s
  `chat_function(messages=messages, ...)` call, where
  `chat_function = self.provider_functions.get(provider_lower)`. Rather than
  monkeypatching a module-level `chat_with_openai` (which `_call_llm` never
  references directly — it only reads the dict), the tests below replace the
  `"openai"` entry in the generator's own `self.provider_functions` dict
  with a capturing fake. This is the exact callable `_call_llm` invokes, so
  `generate_*`'s system/user prompt assembly (the migrated code under test)
  and `_call_llm`'s `messages = [...]` construction both run unmodified.
- `get_conversation_context` calls
  `self.db.get_messages_by_conversation_id(...)`, a method that does not
  exist anywhere on `CharactersRAGDB` (confirmed by repo-wide grep —
  pre-existing, unrelated bug, out of scope for this migration and left
  untouched). The call always raises `AttributeError`, which
  `get_conversation_context` catches and logs, returning `[]`. This means
  `context` is always `""` for any `conversation_id` passed to the
  `generate_*` methods in the tests below, including a nonexistent one — no
  conversation/message fixtures are needed to reach the prompt-assembly seam
  under test.
"""

import inspect

import pytest

from tldw_chatbook.Chat.document_generator import DocumentGenerator
from tldw_chatbook.Internal_Prompts import get_internal_prompt


def _make_generator() -> DocumentGenerator:
    return DocumentGenerator(":memory:", "test-client")


def _capture_messages(generator: DocumentGenerator) -> list:
    """Install a capturing fake at the `_call_llm` provider-dispatch seam
    (`self.provider_functions["openai"]`, the callable `_call_llm` actually
    invokes) and return the list of captured call kwargs (each has a
    "messages" key holding the real assembled [system, user] list)."""
    captured = []

    def fake_chat(**kwargs):
        captured.append(kwargs)
        return "generated content"

    generator.provider_functions["openai"] = fake_chat
    return captured


def _system_and_user(captured_kwargs: dict) -> tuple:
    messages = captured_kwargs["messages"]
    system_message = next(m["content"] for m in messages if m["role"] == "system")
    user_message = next(m["content"] for m in messages if m["role"] == "user")
    return system_message, user_message


# --- task 453: real conversation context reaches the prompt ----------------


def test_generation_includes_real_conversation_context(scratch_config):
    # Regression for the get_messages_by_conversation_id (nonexistent method)
    # bug that made every generated document silently contextless.
    scratch_config("")
    generator = _make_generator()
    conv_id = generator.db.add_conversation(
        {"title": "Trip planning", "client_id": "test-client"}
    )
    generator.db.add_message(
        {
            "conversation_id": conv_id,
            "sender": "user",
            "content": "We leave for Tokyo on March 3rd.",
        }
    )
    generator.db.add_message(
        {
            "conversation_id": conv_id,
            "sender": "assistant",
            "content": "Noted -- Tokyo, departing March 3rd.",
        }
    )

    captured = _capture_messages(generator)
    generator.generate_timeline(
        conversation_id=conv_id, provider="openai", model="gpt-4", api_key="k"
    )

    assert captured, "provider dispatch fake was never invoked"
    _system, user_message = _system_and_user(captured[0])
    # Real message content reaches the payload (not an empty context)...
    assert "We leave for Tokyo on March 3rd." in user_message
    assert "Noted -- Tokyo, departing March 3rd." in user_message
    # ...the normalized 'role' column drives the label (not "UNKNOWN")...
    assert "USER:" in user_message
    assert "ASSISTANT:" in user_message
    # ...and the context is chronological (DESC fetch reversed back to ASC).
    assert user_message.index("Tokyo on March 3rd") < user_message.index(
        "Noted -- Tokyo"
    )
    assert not user_message.endswith("\n\nConversation Context:\n")


def test_generation_empty_context_degrades_gracefully(scratch_config):
    # A missing conversation must not crash — context is simply empty.
    scratch_config("")
    generator = _make_generator()
    captured = _capture_messages(generator)
    generator.generate_timeline(
        conversation_id="does-not-exist", provider="openai", model="gpt-4", api_key="k"
    )
    assert captured
    _system, user_message = _system_and_user(captured[0])
    assert user_message.endswith("\n\nConversation Context:\n")


# --- (a) no override: shipped TOML user text + hardcoded system text ------


def test_timeline_no_override_uses_shipped_toml_user_and_hardcoded_system(
    scratch_config,
):
    scratch_config("")
    generator = _make_generator()
    captured = _capture_messages(generator)

    generator.generate_timeline(
        conversation_id="nonexistent", provider="openai", model="gpt-4", api_key="k"
    )

    assert captured, "provider dispatch fake was never invoked"
    system_message, user_message = _system_and_user(captured[0])

    assert system_message == get_internal_prompt("document_generation.timeline_system")
    assert system_message == (
        "You are an expert at creating clear, chronological timelines from "
        "conversations and content."
    )
    assert user_message.startswith(
        get_internal_prompt("document_generation.timeline_user")
    )
    assert user_message.startswith(
        "Create a detailed text-based timeline based on our "
        "conversation/materials being referenced."
    )
    # Code-side concatenation (string, not token substitution) is untouched.
    assert user_message.endswith("\n\nConversation Context:\n")


@pytest.mark.parametrize(
    ("method_name", "type_prefix"),
    [
        ("generate_timeline", "timeline"),
        ("generate_study_guide", "study_guide"),
        ("generate_briefing", "briefing"),
    ],
)
def test_all_three_generators_thread_system_and_user_registry_ids(
    scratch_config, method_name, type_prefix
):
    """Proves all three migrated generator methods (not just timeline) route
    both their system and user prompts through the registry."""
    scratch_config("")
    generator = _make_generator()
    captured = _capture_messages(generator)

    getattr(generator, method_name)(
        conversation_id="nonexistent", provider="openai", model="gpt-4", api_key="k"
    )

    system_message, user_message = _system_and_user(captured[0])
    assert system_message == get_internal_prompt(
        f"document_generation.{type_prefix}_system"
    )
    assert user_message.startswith(
        get_internal_prompt(f"document_generation.{type_prefix}_user")
    )


# --- (b) [internal_prompts.document_generation] override ------------------


def test_timeline_internal_prompts_override_reaches_payload(scratch_config):
    scratch_config(
        "[internal_prompts.document_generation]\n"
        'timeline_system = "OVERRIDE SYSTEM TEXT"\n'
        'timeline_user = "OVERRIDE USER TEXT"\n'
    )
    generator = _make_generator()
    captured = _capture_messages(generator)

    generator.generate_timeline(
        conversation_id="nonexistent", provider="openai", model="gpt-4", api_key="k"
    )

    system_message, user_message = _system_and_user(captured[0])
    assert system_message == "OVERRIDE SYSTEM TEXT"
    assert user_message.startswith("OVERRIDE USER TEXT")


# --- (c) customized legacy [prompts.document_generation.timeline].prompt --


def test_timeline_legacy_customized_toml_wins_over_shipped_default(scratch_config):
    scratch_config(
        "[prompts.document_generation.timeline]\n"
        'prompt = "MY CUSTOM LEGACY TIMELINE PROMPT"\n'
        "temperature = 0.3\n"
        "max_tokens = 2000\n"
    )
    generator = _make_generator()
    captured = _capture_messages(generator)

    generator.generate_timeline(
        conversation_id="nonexistent", provider="openai", model="gpt-4", api_key="k"
    )

    system_message, user_message = _system_and_user(captured[0])
    assert user_message.startswith("MY CUSTOM LEGACY TIMELINE PROMPT")
    # System prompt is a separate spec entirely and is unaffected by the
    # legacy user-prompt customization.
    assert system_message == (
        "You are an expert at creating clear, chronological timelines from "
        "conversations and content."
    )


def test_timeline_legacy_untouched_toml_does_not_shadow_shipped_default(
    scratch_config,
):
    """Sanity counterpart to the legacy-wins case: writing back the exact
    shipped default under the legacy key must NOT be treated as a
    customization (resolver's differs-from-shipped rule)."""
    shipped = get_internal_prompt("document_generation.timeline_user")
    scratch_config(
        "[prompts.document_generation.timeline]\n"
        f'prompt = "{shipped}"\n'
        "temperature = 0.3\n"
        "max_tokens = 2000\n"
    )
    generator = _make_generator()
    captured = _capture_messages(generator)

    generator.generate_timeline(
        conversation_id="nonexistent", provider="openai", model="gpt-4", api_key="k"
    )

    _system_message, user_message = _system_and_user(captured[0])
    assert user_message.startswith(shipped)


# --- temperature/max_tokens reads (untouched by this migration) -----------


def test_temperature_and_max_tokens_still_read_from_config_dict(scratch_config):
    """Confirms the migration left `self.timeline_config`'s temperature/
    max_tokens reads untouched — only the ['prompt'] usage was removed."""
    scratch_config(
        "[prompts.document_generation.timeline]\n"
        'prompt = "irrelevant for this assertion"\n'
        "temperature = 0.11\n"
        "max_tokens = 1234\n"
    )
    generator = _make_generator()
    captured = _capture_messages(generator)

    generator.generate_timeline(
        conversation_id="nonexistent", provider="openai", model="gpt-4", api_key="k"
    )

    kwargs = captured[0]
    assert kwargs["temperature"] == 0.11
    assert kwargs["max_tokens"] == 1234


# --- grep-guard equivalent: hardcoded literals must be gone ---------------


def test_hardcoded_system_literals_removed_from_document_generator():
    from tldw_chatbook.Chat import document_generator as dg

    source = inspect.getsource(dg)
    assert (
        "You are an expert at creating clear, chronological timelines from "
        "conversations and content." not in source
    )
    assert (
        "You are an educational expert specializing in creating "
        "comprehensive study guides." not in source
    )
    assert (
        "You are an expert at creating executive briefing documents with "
        "actionable insights." not in source
    )
    assert source.count('get_internal_prompt("document_generation.') == 3
    assert source.count("get_internal_prompt('document_generation.") == 3
    assert "self.timeline_config['prompt']" not in source
    assert "self.study_guide_config['prompt']" not in source
    assert "self.briefing_config['prompt']" not in source
    # temperature/max_tokens reads must remain untouched.
    assert 'self.timeline_config.get("temperature"' in source
    assert 'self.timeline_config.get("max_tokens"' in source
    assert 'self.study_guide_config.get("temperature"' in source
    assert 'self.study_guide_config.get("max_tokens"' in source
    assert 'self.briefing_config.get("temperature"' in source
    assert 'self.briefing_config.get("max_tokens"' in source
