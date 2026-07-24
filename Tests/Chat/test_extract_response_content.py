from tldw_chatbook.Chat.Chat_Functions import extract_response_content


def test_openai_shape():
    resp = {"choices": [{"message": {"role": "assistant", "content": "Hello there"}}]}
    assert extract_response_content(resp) == "Hello there"


def test_flat_content_shape():
    # Shape code_audit_tool historically read via resp.get("content")
    assert extract_response_content({"content": "flat text"}) == "flat text"


def test_openai_shape_wins_over_flat():
    resp = {"choices": [{"message": {"content": "nested"}}], "content": "flat"}
    assert extract_response_content(resp) == "nested"


def test_empty_choices_list_no_indexerror():
    assert extract_response_content({"choices": []}) == ""


def test_choices_missing_message():
    assert extract_response_content({"choices": [{}]}) == ""


def test_null_content_coerced_to_empty_string():
    resp = {"choices": [{"message": {"content": None}}]}
    assert extract_response_content(resp) == ""


def test_non_dict_input():
    assert extract_response_content("already a string") == "already a string"
    assert extract_response_content(None) == ""


def test_missing_everything():
    assert extract_response_content({}) == ""
