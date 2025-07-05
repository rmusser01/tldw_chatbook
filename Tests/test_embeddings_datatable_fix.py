#!/usr/bin/env python3
"""Test to verify the embeddings window DataTable fix."""

import pytest
from textual.widgets import DataTable
from textual.app import App
from tldw_chatbook.UI.Embeddings_Window import EmbeddingsWindow
from tldw_chatbook.app import TldwCli


class TestEmbeddingsDataTableFix:
    """Test that the DataTable update_cell usage is correct."""
    
    def test_update_cell_syntax(self):
        """Test that update_cell is called with correct parameters."""
        # Read the file and check for the correct pattern
        with open("tldw_chatbook/UI/Embeddings_Window.py", "r") as f:
            content = f.read()
        
        # Check that we're not using update_cell_at
        assert "update_cell_at(" not in content, "Found update_cell_at usage - should use update_cell instead"
        
        # Check that we're not using get_row_index
        assert "get_row_index(" not in content, "Found get_row_index usage - should use row_key directly"
        
        # Check that update_cell is used with the correct parameters
        # Pattern: table.update_cell(row_key, "✓", "✓") or table.update_cell(row_key, "✓", "")
        import re
        update_cell_pattern = r'table\.update_cell\(row_key, "\u2713", (""|"\u2713")\)'
        matches = re.findall(update_cell_pattern, content)
        assert len(matches) > 0, "Could not find proper update_cell usage pattern"
        
        print(f"✅ Found {len(matches)} correct update_cell usages")
        
    def test_datatable_row_selection_pattern(self):
        """Test the row selection pattern is correct."""
        with open("tldw_chatbook/UI/Embeddings_Window.py", "r") as f:
            content = f.read()
        
        # Check for the correct row selection pattern
        assert "on_row_selected(self, event: DataTable.RowSelected)" in content
        assert "row_key = event.row_key" in content
        assert "table.get_row(row_key)" in content
        
        print("✅ Row selection pattern is correct")
        
    def test_select_all_pattern(self):
        """Test the select all pattern is correct."""
        with open("tldw_chatbook/UI/Embeddings_Window.py", "r") as f:
            content = f.read()
        
        # Check for the correct select all pattern
        assert "for row_key in table.rows:" in content
        assert 'table.update_cell(row_key, "✓", "✓")' in content
        
        print("✅ Select all pattern is correct")
        
    def test_clear_selection_pattern(self):
        """Test the clear selection pattern is correct."""
        with open("tldw_chatbook/UI/Embeddings_Window.py", "r") as f:
            content = f.read()
        
        # Check for the correct clear selection pattern
        assert 'table.update_cell(row_key, "✓", "")' in content
        
        print("✅ Clear selection pattern is correct")


if __name__ == "__main__":
    test = TestEmbeddingsDataTableFix()
    test.test_update_cell_syntax()
    test.test_datatable_row_selection_pattern()
    test.test_select_all_pattern()
    test.test_clear_selection_pattern()
    print("\n✅ All tests passed!")