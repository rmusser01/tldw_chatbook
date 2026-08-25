CREATE TABLE research_quick_note_owner_proofs(
  note_id     TEXT PRIMARY KEY NOT NULL
              REFERENCES notes(id) ON DELETE CASCADE ON UPDATE CASCADE,
  owner_proof TEXT NOT NULL CHECK (
      length(owner_proof) = 64
      AND owner_proof = lower(owner_proof)
      AND owner_proof NOT GLOB '*[^0-9a-f]*'
  ),
  created_at  DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);

INSERT OR IGNORE INTO research_quick_note_owner_proofs (note_id, owner_proof)
SELECT nk.note_id,
       substr(k.keyword, length('research-receipt-proof:') + 1)
  FROM note_keywords AS nk
  JOIN keywords AS k ON k.id = nk.keyword_id
 WHERE k.keyword LIKE 'research-receipt-proof:%'
   AND length(substr(k.keyword, length('research-receipt-proof:') + 1)) = 64
   AND substr(k.keyword, length('research-receipt-proof:') + 1)
       = lower(substr(k.keyword, length('research-receipt-proof:') + 1))
   AND substr(k.keyword, length('research-receipt-proof:') + 1)
       NOT GLOB '*[^0-9a-f]*';

DELETE FROM sync_log
 WHERE entity = 'note_keywords'
   AND EXISTS (
       SELECT 1
         FROM keywords AS k
        WHERE k.keyword LIKE 'research-receipt-proof:%'
          AND CAST(json_extract(sync_log.payload, '$.keyword_id') AS INTEGER) = k.id
   );

DELETE FROM sync_log
 WHERE entity = 'keywords'
   AND entity_id IN (
       SELECT CAST(id AS TEXT)
         FROM keywords
        WHERE keyword LIKE 'research-receipt-proof:%'
   );

DELETE FROM keywords WHERE keyword LIKE 'research-receipt-proof:%';
