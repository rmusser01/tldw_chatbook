from tldw_chatbook.Character_Chat.persona_list_paging import page_persona_profiles

PROFILES = [
    {"id": "1", "name": "Alice", "description": "hero", "created_at": "2026-01-01", "last_modified": "2026-01-03"},
    {"id": "2", "name": "bob", "description": "villain", "created_at": "2026-01-02", "last_modified": "2026-01-02"},
    {"id": "3", "name": "Carol", "description": "hero mage", "created_at": "2026-01-03", "last_modified": "2026-01-01"},
]


def test_search_matches_name_and_description_case_insensitive():
    rows, total = page_persona_profiles(PROFILES, search_term="HERO", sort_key="name_asc", offset=0, page_size=50)
    assert {r["name"] for r in rows} == {"Alice", "Carol"}
    assert total == 2


def test_sort_name_asc_case_insensitive():
    rows, total = page_persona_profiles(PROFILES, search_term=None, sort_key="name_asc", offset=0, page_size=50)
    assert [r["name"] for r in rows] == ["Alice", "bob", "Carol"]
    assert total == 3


def test_sort_created_desc():
    rows, _ = page_persona_profiles(PROFILES, search_term=None, sort_key="created_desc", offset=0, page_size=50)
    assert [r["name"] for r in rows] == ["Carol", "bob", "Alice"]


def test_sort_modified_desc():
    rows, _ = page_persona_profiles(PROFILES, search_term=None, sort_key="modified_desc", offset=0, page_size=50)
    assert [r["name"] for r in rows] == ["Alice", "bob", "Carol"]


def test_pagination_window_and_total():
    profiles = [{"id": str(i), "name": f"p{i:03d}", "description": "", "created_at": "x", "last_modified": "x"} for i in range(120)]
    page2, total = page_persona_profiles(profiles, search_term=None, sort_key="name_asc", offset=50, page_size=50)
    assert total == 120 and len(page2) == 50 and page2[0]["name"] == "p050"


def test_unknown_sort_key_falls_back_to_name():
    rows, _ = page_persona_profiles(PROFILES, search_term=None, sort_key="relevance", offset=0, page_size=50)
    assert [r["name"] for r in rows] == ["Alice", "bob", "Carol"]
