from tldw_chatbook.Chat.sampling_params import (
    KNOWN_SAMPLING_PARAM_KEYS,
    params_to_dict,
    params_to_tuple,
    validate_sampling_params,
)

def test_known_keys_accepted():
    assert validate_sampling_params(
        {"temperature": 0.2, "top_k": 40, "reasoning_effort": "low"}
    ) == []

def test_unknown_key_rejected_as_typo_guard():
    errors = validate_sampling_params({"temprature": 0.2})
    assert len(errors) == 1 and "unknown" in errors[0].lower()

def test_bool_rejected_for_numeric_key():
    # bool IS an int subclass; without an explicit guard `seed: true`
    # would pass an int check.
    assert validate_sampling_params({"seed": True}) != []

def test_wrong_type_rejected():
    assert validate_sampling_params({"temperature": "hot"}) != []
    assert validate_sampling_params({"max_tokens": 1.5}) != []

def test_streaming_is_not_a_sampling_param():
    assert "streaming" not in KNOWN_SAMPLING_PARAM_KEYS

def test_params_tuple_round_trip_sorted():
    pairs = params_to_tuple({"top_p": 0.9, "temperature": 0.2})
    assert pairs == (("temperature", 0.2), ("top_p", 0.9))
    assert params_to_dict(pairs) == {"temperature": 0.2, "top_p": 0.9}
