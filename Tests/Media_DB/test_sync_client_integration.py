# tests/test_sync_client_integration.py
# Description: Integration tests for the ClientSyncEngine class using a real test server
#
# Imports
from datetime import datetime, timedelta
import pytest
import pytest_asyncio
import json
import os
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import List, Dict, Any
import tempfile

# 3rd-Party Imports
import requests

# Local Imports
from .test_media_db_v2 import get_entity_version
from tldw_chatbook.DB.Sync_Client import ClientSyncEngine
from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase

# Test marker for integration tests
pytestmark = pytest.mark.integration

#######################################################################################################################
#
# Test Server Implementation

class MockSyncServer(BaseHTTPRequestHandler):
    """A mock sync server that mimics the real sync server behavior"""
    
    # Class variables to store server state
    server_logs: List[Dict[str, Any]] = []
    server_log_counter: int = 0
    client_states: Dict[str, int] = {}  # Track last processed log ID per client
    
    def log_message(self, format, *args):
        """Suppress default HTTP server logging during tests"""
        pass
    
    def do_POST(self):
        """Handle POST requests for sync operations"""
        if self.path == "/sync/push":
            self._handle_push()
        else:
            self.send_error(404)
    
    def do_GET(self):
        """Handle GET requests for sync operations"""
        if self.path.startswith("/sync/pull"):
            self._handle_pull()
        else:
            self.send_error(404)
    
    def _handle_push(self):
        """Handle push requests from clients"""
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        
        try:
            data = json.loads(post_data.decode('utf-8'))
            client_id = data['client_id']
            changes = data['changes']
            last_processed = data.get('last_processed_server_id', 0)
            
            # Store client's last processed server ID
            self.__class__.client_states[client_id] = last_processed
            
            # Process each change
            for change in changes:
                # Add server metadata
                self.__class__.server_log_counter += 1
                server_change = {
                    **change,
                    'server_log_id': self.__class__.server_log_counter,
                    'server_timestamp': datetime.utcnow().isoformat()
                }
                self.__class__.server_logs.append(server_change)
            
            # Send success response
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            response = {
                'status': 'success',
                'processed_count': len(changes),
                'server_log_id': self.__class__.server_log_counter
            }
            self.wfile.write(json.dumps(response).encode('utf-8'))
            
        except Exception as e:
            self.send_error(400, str(e))
    
    def _handle_pull(self):
        """Handle pull requests from clients"""
        # Parse query parameters
        query_parts = self.path.split('?')
        params = {}
        if len(query_parts) > 1:
            for param in query_parts[1].split('&'):
                key, value = param.split('=')
                params[key] = value
        
        client_id = params.get('client_id', 'unknown')
        after_id = int(params.get('after', '0'))
        
        # Get changes after the specified ID, excluding changes from this client
        changes = [
            log for log in self.__class__.server_logs
            if log['server_log_id'] > after_id and log['client_id'] != client_id
        ]
        
        # Send response
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        response = {
            'changes': changes,
            'latest_server_log_id': self.__class__.server_log_counter
        }
        self.wfile.write(json.dumps(response).encode('utf-8'))
    
    @classmethod
    def reset_server_state(cls):
        """Reset server state between tests"""
        cls.server_logs = []
        cls.server_log_counter = 0
        cls.client_states = {}


# --- Fixtures ---

@pytest.fixture
def test_server():
    """Start a test HTTP server in a separate thread"""
    # Reset server state
    MockSyncServer.reset_server_state()
    
    # Create and start server
    server = HTTPServer(('localhost', 0), MockSyncServer)  # Port 0 = auto-assign
    server_thread = threading.Thread(target=server.serve_forever)
    server_thread.daemon = True
    server_thread.start()
    
    # Get the actual port
    port = server.server_port
    server_url = f"http://localhost:{port}"
    
    yield server_url
    
    # Cleanup
    server.shutdown()
    server_thread.join(timeout=1)


@pytest.fixture
def client_db():
    """Provides a fresh client DB instance"""
    tmp = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
    db = MediaDatabase(db_path=tmp.name, client_id="client_test_integration")
    yield db
    db.close()
    os.unlink(tmp.name)


@pytest.fixture
def client_state_file():
    """Provides path to temp state file"""
    tmp = tempfile.NamedTemporaryFile(suffix='.json', delete=False)
    tmp.close()
    yield tmp.name
    if os.path.exists(tmp.name):
        os.unlink(tmp.name)


@pytest.fixture
def sync_engine(client_db, client_state_file, test_server):
    """Provides an initialized ClientSyncEngine instance with real server"""
    engine = ClientSyncEngine(
        db_instance=client_db,
        server_api_url=test_server,
        client_id=client_db.client_id,
        state_file=client_state_file
    )
    return engine


# Helper function to create entities in database
def create_test_entities(db: MediaDatabase) -> Dict[str, Any]:
    """Create test entities and return their info"""
    # Create keyword
    kw_id, kw_uuid = db.add_keyword("test_keyword")
    
    # Create media item
    media_id = db.insert_media_item(
        title="Test Media",
        content="Test content",
        media_type="article",
        author="Test Author"
    )
    media_info = db.get_media_by_id(media_id)
    
    return {
        'keyword': {'id': kw_id, 'uuid': kw_uuid},
        'media': {'id': media_id, 'uuid': media_info['uuid']}
    }


# --- Test Classes ---

class TestClientSyncEngineIntegration:
    """Integration tests with real HTTP server"""
    
    def test_push_pull_basic_flow(self, sync_engine, client_db):
        """Test basic push and pull flow with real server"""
        # Create local changes
        entities = create_test_entities(client_db)
        
        # Push changes to server
        sync_engine._push_local_changes()
        
        # Verify state was updated
        assert sync_engine.last_local_log_id_sent > 0
        
        # Verify server received the changes
        time.sleep(0.1)  # Give server time to process
        assert len(MockSyncServer.server_logs) == 2  # keyword + media
        
        # Create another client and pull changes
        tmp_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        client2_db = MediaDatabase(db_path=tmp_db.name, client_id="client2")
        tmp_state = tempfile.NamedTemporaryFile(suffix='.json', delete=False)
        
        client2_engine = ClientSyncEngine(
            db_instance=client2_db,
            server_api_url=sync_engine.server_api_url,
            client_id="client2",
            state_file=tmp_state.name
        )
        
        # Pull changes
        client2_engine._pull_and_apply_server_changes()
        
        # Verify client2 received the changes
        keywords = client2_db.get_keywords()
        assert len(keywords) == 1
        assert keywords[0]['keyword'] == "test_keyword"
        
        # Cleanup
        client2_db.close()
        os.unlink(tmp_db.name)
        os.unlink(tmp_state.name)
    
    def test_conflict_resolution_lww(self, sync_engine, client_db, test_server):
        """Test Last-Write-Wins conflict resolution with real server"""
        # Create a keyword
        kw_id, kw_uuid = client_db.add_keyword("original")
        
        # Push to server
        sync_engine._push_local_changes()
        
        # Create second client and pull the keyword
        tmp_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        client2_db = MediaDatabase(db_path=tmp_db.name, client_id="client2")
        tmp_state = tempfile.NamedTemporaryFile(suffix='.json', delete=False)
        
        client2_engine = ClientSyncEngine(
            db_instance=client2_db,
            server_api_url=test_server,
            client_id="client2",
            state_file=tmp_state.name
        )
        
        client2_engine._pull_and_apply_server_changes()
        
        # Both clients update the same keyword
        client_db.update_keyword(kw_id, "client1_update")
        
        # Get client2's keyword ID
        client2_keywords = client2_db.get_keywords()
        client2_kw_id = client2_keywords[0]['id']
        client2_db.update_keyword(client2_kw_id, "client2_update")
        
        # Client1 pushes first
        sync_engine._push_local_changes()
        
        # Client2 pushes (should create a conflict)
        client2_engine._push_local_changes()
        
        # Client1 pulls - should see client2's update win (assuming it's newer)
        sync_engine._pull_and_apply_server_changes()
        
        # Check the final state
        final_keywords = client_db.get_keywords()
        # The exact winner depends on timestamps, but both clients should converge
        assert len(final_keywords) == 1
        assert final_keywords[0]['keyword'] in ["client1_update", "client2_update"]
        
        # Cleanup
        client2_db.close()
        os.unlink(tmp_db.name)
        os.unlink(tmp_state.name)
    
    def test_sync_with_deletions(self, sync_engine, client_db):
        """Test syncing deletion operations"""
        # Create and push some entities
        entities = create_test_entities(client_db)
        sync_engine._push_local_changes()
        
        # Delete the keyword
        client_db.delete_keyword(entities['keyword']['id'])
        
        # Push deletion
        sync_engine._push_local_changes()
        
        # Create another client and sync
        tmp_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        client2_db = MediaDatabase(db_path=tmp_db.name, client_id="client2")
        tmp_state = tempfile.NamedTemporaryFile(suffix='.json', delete=False)
        
        client2_engine = ClientSyncEngine(
            db_instance=client2_db,
            server_api_url=sync_engine.server_api_url,
            client_id="client2",
            state_file=tmp_state.name
        )
        
        # Pull all changes including deletion
        client2_engine._pull_and_apply_server_changes()
        
        # Verify the keyword doesn't exist in client2
        keywords = client2_db.get_keywords()
        assert len(keywords) == 0
        
        # But media should still exist
        media_items = client2_db.get_all_media()
        assert len(media_items) == 1
        
        # Cleanup
        client2_db.close()
        os.unlink(tmp_db.name)
        os.unlink(tmp_state.name)
    
    def test_network_error_handling(self, client_db, client_state_file):
        """Test handling of network errors"""
        # Create engine with invalid server URL
        engine = ClientSyncEngine(
            db_instance=client_db,
            server_api_url="http://localhost:99999",  # Invalid port
            client_id="test_client",
            state_file=client_state_file
        )
        
        # Create some local changes
        client_db.add_keyword("test")
        
        # Try to push - should handle connection error gracefully
        try:
            engine._push_local_changes()
        except requests.exceptions.ConnectionError:
            pass  # Expected
        
        # State should not be updated on failure
        assert engine.last_local_log_id_sent == 0
        
        # Try to pull - should also handle error gracefully
        try:
            engine._pull_and_apply_server_changes()
        except requests.exceptions.ConnectionError:
            pass  # Expected
        
        assert engine.last_server_log_id_processed == 0
    
    def test_idempotent_operations(self, sync_engine, client_db):
        """Test that operations are idempotent"""
        # Create changes
        kw_id, kw_uuid = client_db.add_keyword("idempotent_test")
        
        # Push multiple times
        sync_engine._push_local_changes()
        initial_server_count = len(MockSyncServer.server_logs)
        
        sync_engine._push_local_changes()
        sync_engine._push_local_changes()
        
        # Server should not have duplicate entries
        assert len(MockSyncServer.server_logs) == initial_server_count
        
        # Create another client and pull multiple times
        tmp_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        client2_db = MediaDatabase(db_path=tmp_db.name, client_id="client2")
        tmp_state = tempfile.NamedTemporaryFile(suffix='.json', delete=False)
        
        client2_engine = ClientSyncEngine(
            db_instance=client2_db,
            server_api_url=sync_engine.server_api_url,
            client_id="client2",
            state_file=tmp_state.name
        )
        
        # Pull multiple times
        client2_engine._pull_and_apply_server_changes()
        client2_engine._pull_and_apply_server_changes()
        client2_engine._pull_and_apply_server_changes()
        
        # Should still have exactly one keyword
        keywords = client2_db.get_keywords()
        assert len(keywords) == 1
        assert keywords[0]['keyword'] == "idempotent_test"
        
        # Cleanup
        client2_db.close()
        os.unlink(tmp_db.name)
        os.unlink(tmp_state.name)


class TestSyncEnginePerformance:
    """Performance-related integration tests"""
    
    def test_large_batch_sync(self, sync_engine, client_db):
        """Test syncing a large number of changes"""
        # Create many keywords
        num_items = 100
        for i in range(num_items):
            client_db.add_keyword(f"keyword_{i}")
        
        # Measure push time
        start_time = time.time()
        sync_engine._push_local_changes()
        push_time = time.time() - start_time
        
        # Should complete in reasonable time
        assert push_time < 5.0  # 5 seconds for 100 items
        
        # Verify all items were pushed
        assert len(MockSyncServer.server_logs) == num_items
        
        # Create another client and pull
        tmp_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        client2_db = MediaDatabase(db_path=tmp_db.name, client_id="client2")
        tmp_state = tempfile.NamedTemporaryFile(suffix='.json', delete=False)
        
        client2_engine = ClientSyncEngine(
            db_instance=client2_db,
            server_api_url=sync_engine.server_api_url,
            client_id="client2",
            state_file=tmp_state.name
        )
        
        # Measure pull time
        start_time = time.time()
        client2_engine._pull_and_apply_server_changes()
        pull_time = time.time() - start_time
        
        # Should complete in reasonable time
        assert pull_time < 5.0
        
        # Verify all items were received
        keywords = client2_db.get_keywords()
        assert len(keywords) == num_items
        
        # Cleanup
        client2_db.close()
        os.unlink(tmp_db.name)
        os.unlink(tmp_state.name)