CREATE TABLE character_portrait_regions (
    character_id INTEGER NOT NULL REFERENCES character_cards(id) ON DELETE CASCADE,
    expression_key TEXT NOT NULL CHECK(length(expression_key) > 0),
    source_sha256 TEXT NOT NULL CHECK(length(source_sha256) = 64),
    source_kind TEXT NOT NULL CHECK(source_kind IN ('portrait','legacy_expression','visual_identity')),
    source_ref_json TEXT NOT NULL CHECK(json_valid(source_ref_json) AND length(source_ref_json) <= 128),
    recipe_json TEXT NOT NULL CHECK(json_valid(recipe_json) AND length(CAST(recipe_json AS BLOB)) <= 2048),
    revision INTEGER NOT NULL CHECK(revision > 0),
    PRIMARY KEY(character_id, expression_key)
);
ALTER TABLE character_portrait_performances ADD COLUMN composition_recipe_json TEXT
    CHECK(composition_recipe_json IS NULL OR (json_valid(composition_recipe_json) AND length(CAST(composition_recipe_json AS BLOB)) <= 2048));
