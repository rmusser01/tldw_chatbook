# test_exporters.py
# Description: Unit tests for the unified exporters module
#
"""
Test Evaluation Exporters
-------------------------

Tests for the consolidated export functionality.
"""

import pytest
import json
import csv
from pathlib import Path
from unittest.mock import Mock, patch
from datetime import datetime

from tldw_chatbook.Evals.exporters import EvaluationExporter, export_to_csv, export_to_json


class MockABTestResult:
    """Mock ABTestResult for testing."""
    def __init__(self):
        self.test_id = "test_001"
        self.test_name = "Model Comparison Test"
        self.model_a_name = "Model A"
        self.model_b_name = "Model B"
        self.sample_size = 100
        self.winner = "Model A"
        self.timestamp = datetime.now().isoformat()
        self.model_a_metrics = {
            'accuracy': 0.85,
            'f1': 0.82,
            'latency': 150.5
        }
        self.model_b_metrics = {
            'accuracy': 0.78,
            'f1': 0.75,
            'latency': 200.3
        }
        self.statistical_tests = {
            'accuracy': {
                'difference': 0.07,
                'p_value': 0.03,
                'is_significant': True
            },
            'f1': {
                'difference': 0.07,
                'p_value': 0.04,
                'is_significant': True
            }
        }
        self.model_a_latency = 150.5
        self.model_b_latency = 200.3
        self.model_a_cost = 0.05
        self.model_b_cost = 0.08
        self.confidence = 0.95
        self.recommendations = [
            "Model A shows better performance",
            "Consider Model A for production"
        ]


class TestEvaluationExporter:
    """Test suite for EvaluationExporter."""
    
    @pytest.fixture
    def exporter(self):
        """Create an exporter instance."""
        return EvaluationExporter()
    
    @pytest.fixture
    def ab_test_result(self):
        """Create a mock A/B test result."""
        return MockABTestResult()
    
    @pytest.fixture
    def standard_run_data(self):
        """Create mock standard run data."""
        return {
            'run_id': 'run_123',
            'model': 'test-model',
            'task': 'test-task',
            'status': 'completed',
            'total_cost': 0.15,
            'completed_samples': 50,
            'duration_seconds': 120.5,
            'metrics': {
                'accuracy': 0.88,
                'f1': 0.85,
                'precision': 0.87,
                'recall': 0.83
            },
            'results': [
                {
                    'id': '1',
                    'input': 'Test input 1',
                    'output': 'Test output 1',
                    'score': 0.9
                },
                {
                    'id': '2',
                    'input': 'Test input 2',
                    'output': 'Test output 2',
                    'score': 0.85
                }
            ]
        }
    
    def test_export_dispatch_ab_test(self, exporter, ab_test_result, tmp_path):
        """Test that export correctly dispatches A/B test results."""
        output_path = tmp_path / "ab_test.csv"
        
        exporter.export(ab_test_result, output_path, format='csv')
        
        assert output_path.exists()
        
        # Read and verify CSV content
        with open(output_path, 'r') as f:
            content = f.read()
            assert "A/B Test Results" in content
            assert "Model A" in content
            assert "Model B" in content
    
    def test_export_dispatch_standard_run(self, exporter, standard_run_data, tmp_path):
        """Test that export correctly dispatches standard run data."""
        output_path = tmp_path / "standard_run.json"
        
        exporter.export(standard_run_data, output_path, format='json')
        
        assert output_path.exists()
        
        # Read and verify JSON content
        with open(output_path, 'r') as f:
            data = json.load(f)
            assert data['run_id'] == 'run_123'
            assert data['model'] == 'test-model'
    
    def test_export_ab_test_csv(self, exporter, ab_test_result, tmp_path):
        """Test exporting A/B test results to CSV."""
        output_path = tmp_path / "ab_test.csv"
        
        exporter._export_ab_test_csv(ab_test_result, output_path)
        
        assert output_path.exists()
        
        # Read CSV and verify structure
        with open(output_path, 'r') as f:
            reader = csv.reader(f)
            rows = list(reader)
            
            # Check header
            assert rows[0][0] == "A/B Test Results Export"
            
            # Check for key information
            content = str(rows)
            assert "Model A" in content
            assert "Model B" in content
            assert "0.85" in content  # accuracy value
    
    def test_export_ab_test_json(self, exporter, ab_test_result, tmp_path):
        """Test exporting A/B test results to JSON."""
        output_path = tmp_path / "ab_test.json"
        
        exporter._export_ab_test_json(ab_test_result, output_path, {'include_raw_outputs': False})
        
        assert output_path.exists()
        
        with open(output_path, 'r') as f:
            data = json.load(f)
            
            assert data['test_id'] == 'test_001'
            assert data['configuration']['model_a'] == 'Model A'
            assert data['metrics']['model_a']['accuracy'] == 0.85
            assert data['conclusion']['winner'] == 'Model A'
    
    def test_export_ab_test_markdown(self, exporter, ab_test_result, tmp_path):
        """Test exporting A/B test results to Markdown."""
        output_path = tmp_path / "ab_test.md"
        
        exporter._export_ab_test_markdown(ab_test_result, output_path)
        
        assert output_path.exists()
        
        with open(output_path, 'r') as f:
            content = f.read()
            
            assert "# A/B Test Report" in content
            assert "## Executive Summary" in content
            assert "## Metrics Comparison" in content
            assert "Model A" in content
            assert "Model B" in content
            assert "|" in content  # Table formatting
    
    def test_export_ab_test_latex(self, exporter, ab_test_result, tmp_path):
        """Test exporting A/B test results to LaTeX."""
        output_path = tmp_path / "ab_test.tex"
        
        exporter._export_ab_test_latex(ab_test_result, output_path)
        
        assert output_path.exists()
        
        with open(output_path, 'r') as f:
            content = f.read()
            
            assert "\\documentclass{article}" in content
            assert "\\begin{table}" in content
            assert "\\begin{tabular}" in content
            assert "Model A" in content
    
    def test_export_standard_run_csv(self, exporter, standard_run_data, tmp_path):
        """Test exporting standard run to CSV."""
        output_path = tmp_path / "run.csv"
        
        exporter._export_run_csv(standard_run_data, output_path, {})
        
        assert output_path.exists()
        
        with open(output_path, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            
            assert len(rows) == 2  # Two result samples
            assert rows[0]['id'] == '1'
            assert rows[0]['input'] == 'Test input 1'
    
    def test_export_standard_run_json(self, exporter, standard_run_data, tmp_path):
        """Test exporting standard run to JSON."""
        output_path = tmp_path / "run.json"
        
        exporter._export_run_json(
            standard_run_data,
            output_path,
            {'include_raw_outputs': True, 'include_metadata': True}
        )
        
        assert output_path.exists()
        
        with open(output_path, 'r') as f:
            data = json.load(f)
            
            assert data['run_id'] == 'run_123'
            assert data['metrics']['accuracy'] == 0.88
            assert len(data['results']) == 2
    
    def test_export_standard_run_markdown(self, exporter, standard_run_data, tmp_path):
        """Test exporting standard run to Markdown."""
        output_path = tmp_path / "run.md"
        
        exporter._export_run_markdown(standard_run_data, output_path)
        
        assert output_path.exists()
        
        with open(output_path, 'r') as f:
            content = f.read()
            
            assert "# Evaluation Report" in content
            assert "## Summary" in content
            assert "## Metrics" in content
            assert "run_123" in content
            assert "0.88" in content  # accuracy value
    
    def test_export_invalid_format(self, exporter, standard_run_data, tmp_path):
        """Test exporting with invalid format raises error."""
        output_path = tmp_path / "output.xyz"
        
        with pytest.raises(ValueError) as exc_info:
            exporter.export(standard_run_data, output_path, format='invalid')
        
        assert "Unsupported export format" in str(exc_info.value)
    
    def test_export_empty_results(self, exporter, tmp_path):
        """Test exporting empty results."""
        empty_data = {
            'run_id': 'empty_run',
            'model': 'test',
            'task': 'test',
            'status': 'completed',
            'total_cost': 0,
            'completed_samples': 0,
            'results': []
        }
        
        output_path = tmp_path / "empty.csv"
        
        exporter._export_run_csv(empty_data, output_path, {})
        
        assert output_path.exists()
        
        with open(output_path, 'r') as f:
            reader = csv.reader(f)
            rows = list(reader)
            
            # Should have summary row
            assert len(rows) >= 2
            assert "empty_run" in str(rows)


class TestLegacyFunctions:
    """Test backward compatibility functions."""
    
    def test_export_to_csv_legacy(self, tmp_path):
        """Test legacy export_to_csv function."""
        run_data = {
            'run_id': 'legacy_csv',
            'results': [
                {'id': '1', 'value': 'test'}
            ]
        }
        
        output_path = tmp_path / "legacy.csv"
        
        export_to_csv(run_data, output_path, {})
        
        assert output_path.exists()
    
    def test_export_to_json_legacy(self, tmp_path):
        """Test legacy export_to_json function."""
        run_data = {
            'run_id': 'legacy_json',
            'metrics': {'test': 1.0}
        }
        
        output_path = tmp_path / "legacy.json"
        
        export_to_json(run_data, output_path, {})
        
        assert output_path.exists()
        
        with open(output_path, 'r') as f:
            data = json.load(f)
            assert data['run_id'] == 'legacy_json'


class TestExportOptions:
    """Test various export options."""
    
    @pytest.fixture
    def exporter(self):
        return EvaluationExporter()
    
    def test_json_export_without_raw_outputs(self, exporter, tmp_path):
        """Test JSON export excluding raw outputs."""
        data = {
            'run_id': 'test',
            'results': [
                {
                    'id': '1',
                    'input': 'test',
                    'output': 'result',
                    'raw_output': 'This should be excluded'
                }
            ]
        }
        
        output_path = tmp_path / "no_raw.json"
        
        exporter._export_run_json(
            data,
            output_path,
            {'include_raw_outputs': False}
        )
        
        with open(output_path, 'r') as f:
            exported = json.load(f)
            
            # raw_output should be removed
            assert 'raw_output' not in exported['results'][0]
            assert exported['results'][0]['output'] == 'result'
    
    def test_json_export_without_metadata(self, exporter, tmp_path):
        """Test JSON export excluding metadata."""
        data = {
            'run_id': 'test',
            'metadata': {'should': 'be_excluded'},
            'results': []
        }
        
        output_path = tmp_path / "no_metadata.json"
        
        exporter._export_run_json(
            data,
            output_path,
            {'include_metadata': False}
        )
        
        with open(output_path, 'r') as f:
            exported = json.load(f)
            
            assert 'metadata' not in exported
            assert exported['run_id'] == 'test'