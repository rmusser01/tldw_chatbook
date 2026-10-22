DROP TABLE chunk_fts;
INSERT INTO chunks(path,text,start_char,end_char,line_start,line_end)
SELECT path,'',0,0,1,1 FROM files WHERE path NOT IN (SELECT path FROM chunks);
CREATE VIRTUAL TABLE chunk_fts USING fts5(path,text);
INSERT INTO chunk_fts(rowid,path,text) SELECT id,path,text FROM chunks;
ALTER TABLE directories ADD COLUMN attempt_revision INTEGER NOT NULL DEFAULT 0;
CREATE INDEX directories_eligible ON directories(complete,attempt_revision,path);
DROP TABLE schema_version;
CREATE TABLE schema_version(version INTEGER PRIMARY KEY CHECK(version=3));
INSERT INTO schema_version VALUES(3);
