ALTER TABLE guardian_alerts ADD COLUMN crisis_provenance INTEGER NOT NULL DEFAULT -1 CHECK(crisis_provenance IN (-1,0,1));
DELETE FROM schema_version;
INSERT INTO schema_version (version) VALUES (2);
