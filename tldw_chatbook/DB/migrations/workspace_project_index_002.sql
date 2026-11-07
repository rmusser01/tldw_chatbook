DROP TABLE schema_version;
CREATE TABLE schema_version(version INTEGER PRIMARY KEY CHECK(version=2));
INSERT INTO schema_version VALUES(2);
CREATE TABLE authority_witness(singleton INTEGER PRIMARY KEY CHECK(singleton=1), revision INTEGER NOT NULL);
CREATE TABLE scan_seen(path TEXT PRIMARY KEY);
CREATE TABLE legacy_retirement(singleton INTEGER PRIMARY KEY CHECK(singleton=1));
INSERT INTO legacy_retirement SELECT 1 WHERE EXISTS(SELECT 1 FROM active_job);
