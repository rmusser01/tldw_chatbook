# ToDo for tldw-cli

### ToDo List
- [ ] Manual for use
- [x] Add tests
- [ ] Add examples of use
- **General UI/UX**
  - [x] Setup Emoji support/handling/detection if the user's terminal supports it
  - 
- **Chat Functionality**
  - [ ] Support for uploading files in chat
- **Character Chat Functionality**
  - [ ] Add support for multiple files
- **Media Endpoint Functionality**
  - [ ] Add support for media endpoint
  - [ ] Add support for searching media endpoint for specific media
  - [ ] Add support for reviewing full details of found items/media items
  - [ ] Add support for ingestion of media files into the tldw `/media/add` endpoint
  - [ ] Add support for viewing versions of media files
  - [ ] Add support for modifying or deleting versions of media files
  - [x] Add support for processing of media files without remote ingestion.
- **RAG Search Endpoint Functionality**
  - [ ] Add support for RAG search endpoint
  - [ ] Add support for searching against RAG endpoint
- **Stats & Logs Functionality**
  - [ ] Add support for local usage statistics
    - This should capture total tokens used, tokens per endpoint/API, and tokens per character
    - Also things like cost per endpoint/API, and cost per character (maybe? low priority)
  - [ ] Add support for logging of usage
    - This should capture at least the same information as the stats, but more
    - So requests, responses, errors, etc.
    - Also a way to file bug reports if one is encountered. (maybe from main menu?)
- **Local DB Functionality**
  - [x] Allow for ingestion of media files that have been processed by the tldw API (process-* endpoints)
  - [ ] Allow for editing/modifications/deletion of locally stored media files/character cards/chats


https://github.com/paulrobello/parllama


