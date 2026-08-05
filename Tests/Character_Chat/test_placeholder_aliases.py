from tldw_chatbook.Character_Chat.Character_Chat_Lib import replace_placeholders


def test_new_character_side_aliases():
    out = replace_placeholders("Hi {{user}}, I am {{char}}/{{character}}/{{persona}}.", "Ada", "Sam")
    assert out == "Hi Sam, I am Ada/Ada/Ada."


def test_user_side_tokens_never_get_character_name():
    # THE brainstorm correction: user-side tokens carry the USER's name only.
    out = replace_placeholders("{{user}} {{random_user}} <USER>", "Ada", "Sam")
    assert out == "Sam Sam Sam"
    assert "Ada" not in out


def test_token_free_text_byte_identical():
    assert replace_placeholders("plain text", "Ada", "Sam") == "plain text"


def test_defaults_unchanged():
    out = replace_placeholders("{{user}} meets {{persona}}", None, None)
    assert out == "User meets Character"
