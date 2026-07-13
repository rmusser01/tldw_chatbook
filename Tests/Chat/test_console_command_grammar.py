from tldw_chatbook.Chat.console_command_grammar import (
    CommandParse,
    ConsoleCommand,
    ConsoleCommandRegistry,
    default_console_registry,
)


def test_registered_command_parses_name_and_args():
    registry = default_console_registry()

    assert registry.parse("/prompt release note") == CommandParse(
        "command", "prompt", "release note"
    )


def test_registered_command_match_is_case_insensitive():
    registry = default_console_registry()

    assert registry.parse("/PROMPT x") == CommandParse("command", "prompt", "x")


def test_unmatched_leading_token_with_embedded_slashes_is_unknown():
    # Tokenizer rule: token = chars up to first whitespace; name = token[1:].
    # "/usr/bin/thing" has no whitespace at all, so the whole thing is one
    # token and the entire remainder (including the embedded slashes) is the
    # name — it is NOT re-split on "/".
    registry = default_console_registry()

    assert registry.parse("/usr/bin/thing") == CommandParse(
        "unknown", "usr/bin/thing", ""
    )


def test_plain_text_without_leading_slash_is_not_command():
    registry = default_console_registry()

    assert registry.parse("hello") == CommandParse("not-command")


def test_bare_system_command_has_empty_args():
    registry = default_console_registry()

    assert registry.parse("/system") == CommandParse("command", "system", "")


def test_draft_containing_a_paste_token_marker_is_not_command_even_with_leading_slash():
    # A caller could hand this module the composer's canonical draft text
    # while a large paste is staged; if the draft happens to carry the
    # composer's collapsed-paste display marker, treat the whole draft as
    # not-command per the grammar rule, regardless of the leading slash.
    registry = default_console_registry()

    assert registry.parse("/prompt Pasted Text: 512 Characters") == CommandParse(
        "not-command"
    )
    assert registry.parse("/prompt Unfurl? more text") == CommandParse("not-command")


def test_fallback_resolver_claiming_a_word_wins_over_unknown():
    registry = default_console_registry()

    def resolver(word: str, rest: str):
        if word == "myskill":
            return CommandParse("fallback", word, rest)
        return None

    registry.register_fallback_resolver(resolver)

    assert registry.parse("/myskill do the thing") == CommandParse(
        "fallback", "myskill", "do the thing"
    )


def test_fallback_resolver_returning_none_falls_through_to_unknown():
    registry = default_console_registry()
    calls = []

    def resolver(word: str, rest: str):
        calls.append((word, rest))
        return None

    registry.register_fallback_resolver(resolver)

    assert registry.parse("/ghost text") == CommandParse("unknown", "ghost", "")
    assert calls == [("ghost", "text")]


def test_second_fallback_resolver_is_consulted_after_first_declines():
    registry = default_console_registry()
    registry.register_fallback_resolver(lambda word, rest: None)

    def second(word: str, rest: str):
        return CommandParse("fallback", word, rest) if word == "found" else None

    registry.register_fallback_resolver(second)

    assert registry.parse("/found here") == CommandParse("fallback", "found", "here")


def test_default_console_registry_registers_prompt_and_system_with_stable_ids():
    registry = default_console_registry()

    assert registry.available_names() == ("prompt", "system")
    assert registry.parse("/prompt") == CommandParse("command", "prompt", "")
    assert registry.parse("/system") == CommandParse("command", "system", "")


def test_register_adds_a_new_command_available_by_name():
    registry = ConsoleCommandRegistry()
    registry.register(ConsoleCommand("foo", "[bar]", "do-foo"))

    assert registry.available_names() == ("foo",)
    assert registry.parse("/foo baz") == CommandParse("command", "foo", "baz")


def test_empty_registry_has_no_available_names_and_unknown_falls_through():
    registry = ConsoleCommandRegistry()

    assert registry.available_names() == ()
    assert registry.parse("/anything") == CommandParse("unknown", "anything", "")
