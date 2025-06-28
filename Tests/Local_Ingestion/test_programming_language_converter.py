"""
Tests for ProgrammingLanguageConverter
"""
import pytest
from pathlib import Path
import tempfile
import shutil
from tldw_chatbook.Local_Ingestion.format_converters import (
    ProgrammingLanguageConverter, get_format_registry, can_convert_file, convert_file
)


class TestProgrammingLanguageConverter:
    """Test the ProgrammingLanguageConverter class"""
    
    @pytest.fixture
    def converter(self):
        """Create a ProgrammingLanguageConverter instance"""
        return ProgrammingLanguageConverter()
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files"""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    def test_supported_extensions(self, converter):
        """Test that converter reports correct supported extensions"""
        extensions = converter.supported_extensions
        assert '.py' in extensions
        assert '.js' in extensions
        assert '.java' in extensions
        assert '.cs' in extensions
        assert '.cpp' in extensions
        assert '.go' in extensions
        assert '.rs' in extensions
        assert len(extensions) > 30  # Should support many languages
    
    def test_can_convert_programming_files(self, converter, temp_dir):
        """Test can_convert method for various programming files"""
        # Test files that should be convertible
        test_files = [
            'test.py', 'script.js', 'Main.java', 'Program.cs',
            'main.cpp', 'server.go', 'lib.rs', 'app.rb'
        ]
        
        for filename in test_files:
            file_path = temp_dir / filename
            file_path.touch()
            assert converter.can_convert(file_path), f"Should be able to convert {filename}"
        
        # Test files that should NOT be convertible
        non_code_files = ['doc.pdf', 'image.png', 'data.db', 'archive.zip']
        for filename in non_code_files:
            file_path = temp_dir / filename
            file_path.touch()
            assert not converter.can_convert(file_path), f"Should not convert {filename}"
    
    def test_convert_python_file(self, converter, temp_dir):
        """Test converting a Python file"""
        python_code = '''#!/usr/bin/env python3
"""Test module for demonstration"""
import os
import sys
from pathlib import Path

# This is a comment
class TestClass:
    """A test class"""
    def __init__(self):
        self.value = 42
    
    def test_method(self):
        """A test method"""
        return self.value * 2

def main():
    """Main function"""
    obj = TestClass()
    print(obj.test_method())

if __name__ == "__main__":
    main()
'''
        
        file_path = temp_dir / 'test_module.py'
        file_path.write_text(python_code)
        
        content, metadata = converter.convert(file_path)
        
        # Check content formatting
        assert '# test_module.py' in content
        assert '# Language: Python' in content
        assert '```py' in content or '```python' in content
        assert python_code in content
        
        # Check metadata
        assert metadata['format'] == 'code'
        assert metadata['language'] == 'Python'
        assert metadata['extension'] == '.py'
        assert metadata['filename'] == 'test_module.py'
        assert metadata['line_count'] == 24
        assert metadata['import_count'] == 3
        assert 'functions' in metadata
        assert 'main' in metadata['functions']
        assert 'classes' in metadata
        assert 'TestClass' in metadata['classes']
        assert metadata['comment_lines'] == 2  # Two comment lines starting with #
    
    def test_convert_javascript_file(self, converter, temp_dir):
        """Test converting a JavaScript file"""
        js_code = '''// JavaScript test file
import React from 'react';
import { useState } from 'react';
const axios = require('axios');

// Function to fetch data
async function fetchData(url) {
    try {
        const response = await axios.get(url);
        return response.data;
    } catch (error) {
        console.error('Error:', error);
    }
}

class DataProcessor {
    constructor() {
        this.data = [];
    }
    
    process(item) {
        return item * 2;
    }
}

export default DataProcessor;
'''
        
        file_path = temp_dir / 'data_processor.js'
        file_path.write_text(js_code)
        
        content, metadata = converter.convert(file_path)
        
        # Check content
        assert '# data_processor.js' in content
        assert '# Language: JavaScript' in content
        assert '```js' in content
        assert js_code in content
        
        # Check metadata
        assert metadata['language'] == 'JavaScript'
        assert metadata['import_count'] == 3
        assert 'fetchData' in metadata.get('functions', [])
        assert 'DataProcessor' in metadata.get('classes', [])
        assert metadata['comment_lines'] == 2
    
    def test_convert_java_file(self, converter, temp_dir):
        """Test converting a Java file"""
        java_code = '''package com.example;

import java.util.List;
import java.util.ArrayList;

public class Example {
    private List<String> items;
    
    public Example() {
        this.items = new ArrayList<>();
    }
    
    public void addItem(String item) {
        items.add(item);
    }
    
    public static void main(String[] args) {
        Example ex = new Example();
        ex.addItem("test");
    }
}
'''
        
        file_path = temp_dir / 'Example.java'
        file_path.write_text(java_code)
        
        content, metadata = converter.convert(file_path)
        
        # Check metadata
        assert metadata['language'] == 'Java'
        assert metadata['import_count'] == 2
        assert 'Example' in metadata.get('classes', [])
        assert any('addItem' in func for func in metadata.get('functions', []))
    
    def test_encoding_detection(self, converter, temp_dir):
        """Test that converter handles different encodings"""
        # Create file with non-ASCII characters
        content_utf8 = 'def función():\n    return "Ñoño"  # Español'
        file_path = temp_dir / 'unicode_test.py'
        file_path.write_text(content_utf8, encoding='utf-8')
        
        converted_content, metadata = converter.convert(file_path)
        assert 'función' in converted_content
        assert 'Ñoño' in converted_content
    
    def test_test_file_detection(self, converter, temp_dir):
        """Test that test files are properly detected"""
        test_files = ['test_module.py', 'module_test.js', 'spec_helper.rb']
        
        for filename in test_files:
            file_path = temp_dir / filename
            file_path.write_text('# Test file')
            _, metadata = converter.convert(file_path)
            assert metadata['is_test'] is True, f"{filename} should be detected as test file"
        
        # Non-test file
        regular_file = temp_dir / 'regular_module.py'
        regular_file.write_text('# Regular file')
        _, metadata = converter.convert(regular_file)
        assert metadata['is_test'] is False
    
    def test_global_registry_integration(self, temp_dir):
        """Test that ProgrammingLanguageConverter is registered globally"""
        registry = get_format_registry()
        
        # Create a Python file
        py_file = temp_dir / 'test.py'
        py_file.write_text('print("Hello, World!")')
        
        # Test via global functions
        assert can_convert_file(py_file)
        
        content, metadata = convert_file(py_file)
        assert metadata['language'] == 'Python'
        assert 'print("Hello, World!")' in content
    
    def test_multiple_language_files(self, converter, temp_dir):
        """Test converting files of different programming languages"""
        test_cases = [
            ('script.sh', '#!/bin/bash\necho "Hello"', 'Shell'),
            ('main.go', 'package main\nimport "fmt"\nfunc main() {}', 'Go'),
            ('lib.rs', 'fn main() {\n    println!("Hello");\n}', 'Rust'),
            ('app.rb', 'class App\n  def initialize\n  end\nend', 'Ruby'),
            ('style.css', 'body { color: red; }', None),  # Should not be supported
        ]
        
        for filename, code, expected_lang in test_cases:
            file_path = temp_dir / filename
            file_path.write_text(code)
            
            if expected_lang:
                assert converter.can_convert(file_path)
                content, metadata = converter.convert(file_path)
                assert metadata['language'] == expected_lang
                assert code in content
            else:
                assert not converter.can_convert(file_path)