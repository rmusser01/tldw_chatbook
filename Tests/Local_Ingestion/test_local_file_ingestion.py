"""
Tests for programmatic local file ingestion.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

from tldw_chatbook.Local_Ingestion.local_file_ingestion import (
    ingest_local_file,
    batch_ingest_files,
    ingest_directory,
    get_supported_extensions,
    get_supported_media_types,
    SUPPORTED_EXTENSIONS
)


@pytest.fixture
def mock_media_db():
    """Create a mock MediaDatabase instance."""
    db = Mock()
    db.add_media_with_keywords = Mock(return_value=(123, "test-uuid", "Success"))
    return db


@pytest.fixture
def temp_pdf_file():
    """Create a temporary PDF file for testing."""
    with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as f:
        f.write(b'%PDF-1.4\ntest content')
        temp_path = Path(f.name)
    yield temp_path
    temp_path.unlink()


@pytest.fixture
def temp_text_file():
    """Create a temporary text file for testing."""
    with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as f:
        f.write(b'This is test content for ingestion.')
        temp_path = Path(f.name)
    yield temp_path
    temp_path.unlink()


class TestLocalFileIngestion:
    """Test the local file ingestion functionality."""
    
    def test_get_supported_extensions(self):
        """Test getting supported extensions."""
        extensions = get_supported_extensions()
        assert isinstance(extensions, list)
        assert len(extensions) > 0
        assert '.pdf' in extensions
        assert '.docx' in extensions
        assert '.epub' in extensions
        assert '.txt' in extensions
    
    def test_get_supported_media_types(self):
        """Test getting supported media types."""
        media_types = get_supported_media_types()
        assert isinstance(media_types, list)
        assert 'pdf' in media_types
        assert 'document' in media_types
        assert 'ebook' in media_types
        assert 'plaintext' in media_types
    
    def test_ingest_nonexistent_file(self, mock_media_db):
        """Test ingesting a file that doesn't exist."""
        result = ingest_local_file(
            file_path="/nonexistent/file.pdf",
            media_db=mock_media_db
        )
        
        assert result['success'] is False
        assert 'not found' in result['message']
        assert result['media_id'] is None
    
    def test_ingest_unsupported_file(self, mock_media_db):
        """Test ingesting an unsupported file type."""
        with tempfile.NamedTemporaryFile(suffix='.xyz', delete=False) as f:
            temp_path = Path(f.name)
        
        try:
            result = ingest_local_file(
                file_path=temp_path,
                media_db=mock_media_db
            )
            
            assert result['success'] is False
            assert 'Unsupported file type' in result['message']
        finally:
            temp_path.unlink()
    
    @patch('tldw_chatbook.Local_Ingestion.local_file_ingestion.process_pdf')
    def test_ingest_pdf_success(self, mock_process_pdf, mock_media_db, temp_pdf_file):
        """Test successful PDF ingestion."""
        # Mock the PDF processor
        mock_process_pdf.return_value = {
            'status': 'Success',
            'content': 'Extracted PDF content',
            'metadata': {'title': 'Test PDF', 'author': 'Test Author'},
            'chunks': [{'text': 'chunk 1'}, {'text': 'chunk 2'}],
            'analysis': 'Summary of PDF'
        }
        
        result = ingest_local_file(
            file_path=temp_pdf_file,
            media_db=mock_media_db,
            keywords=['test', 'pdf']
        )
        
        assert result['success'] is True
        assert result['media_id'] == 123
        assert result['media_uuid'] == "test-uuid"
        assert result['media_type'] == 'pdf'
        assert 'Successfully ingested' in result['message']
        
        # Verify processor was called
        mock_process_pdf.assert_called_once()
        
        # Verify database was called
        mock_media_db.add_media_with_keywords.assert_called_once()
        call_args = mock_media_db.add_media_with_keywords.call_args[1]
        assert call_args['content'] == 'Extracted PDF content'
        assert call_args['title'] == 'Test PDF'
        assert call_args['keywords'] == ['test', 'pdf']
    
    @patch('tldw_chatbook.Local_Ingestion.local_file_ingestion.process_plain_text_file')
    def test_ingest_text_file_success(self, mock_process_text, mock_media_db, temp_text_file):
        """Test successful text file ingestion."""
        mock_process_text.return_value = {
            'status': 'Success',
            'content': 'This is test content for ingestion.',
            'metadata': {}
        }
        
        result = ingest_local_file(
            file_path=temp_text_file,
            media_db=mock_media_db,
            title="My Text File"
        )
        
        assert result['success'] is True
        assert result['media_type'] == 'plaintext'
        assert result['title'] == 'My Text File'
    
    @patch('tldw_chatbook.Local_Ingestion.local_file_ingestion.process_pdf')
    def test_ingest_with_processing_error(self, mock_process_pdf, mock_media_db, temp_pdf_file):
        """Test handling of processing errors."""
        mock_process_pdf.return_value = {
            'status': 'Error',
            'error': 'Failed to parse PDF'
        }
        
        result = ingest_local_file(
            file_path=temp_pdf_file,
            media_db=mock_media_db
        )
        
        assert result['success'] is False
        assert 'Processing failed' in result['message']
        mock_media_db.add_media_with_keywords.assert_not_called()
    
    @patch('tldw_chatbook.Local_Ingestion.local_file_ingestion.process_pdf')
    def test_ingest_with_database_error(self, mock_process_pdf, mock_media_db, temp_pdf_file):
        """Test handling of database errors."""
        mock_process_pdf.return_value = {
            'status': 'Success',
            'content': 'PDF content'
        }
        
        # Mock database failure
        mock_media_db.add_media_with_keywords.return_value = (None, None, "Database error")
        
        result = ingest_local_file(
            file_path=temp_pdf_file,
            media_db=mock_media_db
        )
        
        assert result['success'] is False
        assert 'Database storage failed' in result['message']
    
    @patch('tldw_chatbook.Local_Ingestion.local_file_ingestion.ingest_local_file')
    def test_batch_ingest_files(self, mock_ingest, mock_media_db):
        """Test batch file ingestion."""
        # Mock individual file results
        mock_ingest.side_effect = [
            {'success': True, 'media_id': 1, 'message': 'Success'},
            {'success': False, 'media_id': None, 'message': 'Failed'},
            {'success': True, 'media_id': 2, 'message': 'Success'}
        ]
        
        file_paths = ['/path/1.pdf', '/path/2.pdf', '/path/3.pdf']
        results = batch_ingest_files(
            file_paths=file_paths,
            media_db=mock_media_db,
            common_keywords=['batch']
        )
        
        assert len(results) == 3
        assert results[0]['success'] is True
        assert results[1]['success'] is False
        assert results[2]['success'] is True
        
        # Verify common keywords were passed
        for call in mock_ingest.call_args_list:
            assert 'batch' in call[1]['keywords']
    
    @patch('tldw_chatbook.Local_Ingestion.local_file_ingestion.ingest_local_file')
    def test_batch_ingest_stop_on_error(self, mock_ingest, mock_media_db):
        """Test batch ingestion with stop_on_error=True."""
        mock_ingest.side_effect = [
            {'success': True, 'media_id': 1, 'message': 'Success'},
            {'success': False, 'media_id': None, 'message': 'Failed'}
        ]
        
        file_paths = ['/path/1.pdf', '/path/2.pdf', '/path/3.pdf']
        results = batch_ingest_files(
            file_paths=file_paths,
            media_db=mock_media_db,
            stop_on_error=True
        )
        
        # Should stop after first error
        assert len(results) == 2
        assert mock_ingest.call_count == 2
    
    def test_ingest_directory_nonexistent(self, mock_media_db):
        """Test ingesting from nonexistent directory."""
        results = ingest_directory(
            directory_path="/nonexistent/directory",
            media_db=mock_media_db
        )
        
        assert results == []
    
    @patch('tldw_chatbook.Local_Ingestion.local_file_ingestion.batch_ingest_files')
    def test_ingest_directory_recursive(self, mock_batch_ingest, mock_media_db):
        """Test recursive directory ingestion."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create test files
            (temp_path / 'file1.pdf').write_text('pdf1')
            (temp_path / 'file2.txt').write_text('text')
            (temp_path / 'subdir').mkdir()
            (temp_path / 'subdir' / 'file3.pdf').write_text('pdf2')
            (temp_path / 'ignore.xyz').write_text('ignored')
            
            mock_batch_ingest.return_value = [
                {'success': True} for _ in range(3)
            ]
            
            results = ingest_directory(
                directory_path=temp_path,
                media_db=mock_media_db,
                recursive=True,
                file_extensions=['.pdf', '.txt']
            )
            
            # Should find 3 files (2 PDFs + 1 TXT)
            mock_batch_ingest.assert_called_once()
            file_paths = mock_batch_ingest.call_args[1]['file_paths']
            assert len(file_paths) == 3
            
            # Check file types
            extensions = [p.suffix for p in file_paths]
            assert extensions.count('.pdf') == 2
            assert extensions.count('.txt') == 1
    
    @patch('tldw_chatbook.Local_Ingestion.local_file_ingestion.batch_ingest_files')
    def test_ingest_directory_non_recursive(self, mock_batch_ingest, mock_media_db):
        """Test non-recursive directory ingestion."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create test files
            (temp_path / 'file1.pdf').write_text('pdf1')
            (temp_path / 'subdir').mkdir()
            (temp_path / 'subdir' / 'file2.pdf').write_text('pdf2')
            
            mock_batch_ingest.return_value = [{'success': True}]
            
            results = ingest_directory(
                directory_path=temp_path,
                media_db=mock_media_db,
                recursive=False
            )
            
            # Should only find 1 file (not in subdir)
            file_paths = mock_batch_ingest.call_args[1]['file_paths']
            assert len(file_paths) == 1
            assert file_paths[0].name == 'file1.pdf'