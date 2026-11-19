CREATE INDEX IF NOT EXISTS idx_dream_tracked_origin_story
            ON dream_tracked_items(origin_story_id)
        ;
DROP TABLE IF EXISTS dream_interest_profile_v3;


            CREATE TABLE dream_interest_profile_v3 (
                id INTEGER PRIMARY KEY,
                facet TEXT NOT NULL CHECK(facet IN ('topic', 'goal')),
                text TEXT NOT NULL,
                weight REAL NOT NULL DEFAULT 1.0,
                searchable INTEGER NOT NULL DEFAULT 1,
                source TEXT NOT NULL
                    CHECK(source IN ('user', 'seed', 'personal_context', 'notes',
                                     'media', 'guardian')),
                query_angle TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                last_boosted_at TEXT,
                UNIQUE(facet, text)
            )
            ;

INSERT INTO dream_interest_profile_v3 SELECT * FROM dream_interest_profile;

DROP TABLE dream_interest_profile;

ALTER TABLE dream_interest_profile_v3 RENAME TO dream_interest_profile;

INSERT OR IGNORE INTO schema_version(version) VALUES (3);
