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

def test_float_bounds_enforced():
    assert validate_sampling_params({"temperature": 99}) != []
    assert validate_sampling_params({"temperature": -0.1}) != []
    assert validate_sampling_params({"top_p": 1.5}) != []
    assert validate_sampling_params({"min_p": -0.5}) != []
    assert validate_sampling_params({"presence_penalty": 2.5}) != []
    # Boundary values are inclusive.
    assert validate_sampling_params(
        {"temperature": 2.0, "top_p": 1.0, "min_p": 0.0,
         "frequency_penalty": -2.0}
    ) == []

def test_non_finite_floats_rejected():
    assert validate_sampling_params({"temperature": float("nan")}) != []
    assert validate_sampling_params({"top_p": float("inf")}) != []
    assert validate_sampling_params({"min_p": float("-inf")}) != []

def test_int_minima_enforced():
    assert validate_sampling_params({"max_tokens": 0}) != []
    assert validate_sampling_params({"seed": -1}) != []
    assert validate_sampling_params({"top_k": -1}) != []
    # Provider floor for thinking budgets.
    assert validate_sampling_params({"thinking_budget_tokens": 512}) != []
    assert validate_sampling_params(
        {"max_tokens": 1, "seed": 0, "top_k": 0,
         "thinking_budget_tokens": 1024}
    ) == []

def test_enum_membership_enforced():
    assert validate_sampling_params({"reasoning_effort": "ultra"}) != []
    assert validate_sampling_params({"reasoning_summary": "verbose"}) != []
    assert validate_sampling_params({"verbosity": "loud"}) != []
    assert validate_sampling_params({"thinking_effort": "medium-high"}) != []
    assert validate_sampling_params({
        "reasoning_effort": "xhigh", "reasoning_summary": "concise",
        "verbosity": "medium", "thinking_effort": "max",
    }) == []
