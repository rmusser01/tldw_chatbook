-- ChaChaNotes v53 -> v54: local explicit-before-first Console cursor.
UPDATE db_schema_version
   SET version = 54
 WHERE schema_name = 'rag_char_chat_schema'
   AND version = 53;
