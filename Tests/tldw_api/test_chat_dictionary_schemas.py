"""Default-value pins for the chat dictionary entry request schemas.

An omitted ``case_sensitive`` field must not silently create a
case-sensitive entry: the UI/engine default is case-INsensitive, so the
API request schema's default has to match it.
"""

from tldw_chatbook.tldw_api.chat_dictionary_schemas import (
    DictionaryEntryCreateRequest,
    DictionaryEntryUpdateRequest,
)


class TestDictionaryEntryCreateRequestDefaults:
    def test_case_sensitive_defaults_to_false(self):
        request = DictionaryEntryCreateRequest(pattern="Ada", replacement="Dr. Ada")
        assert request.case_sensitive is False

    def test_enabled_defaults_to_true(self):
        request = DictionaryEntryCreateRequest(pattern="Ada", replacement="Dr. Ada")
        assert request.enabled is True


class TestDictionaryEntryUpdateRequestDefaults:
    def test_optional_fields_default_to_none_and_do_not_force_a_value(self):
        # An update request's unset fields must stay None (meaning "leave
        # unchanged"), not resolve to a hardcoded server-side default.
        request = DictionaryEntryUpdateRequest()
        assert request.case_sensitive is None
        assert request.enabled is None
