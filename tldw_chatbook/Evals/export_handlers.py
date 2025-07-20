# export_handlers.py
# Description: Export handlers for evaluation results
#
"""
Export Handlers for Evaluation Results
-------------------------------------

Provides export functionality for evaluation results in various formats:
- CSV: Tabular data suitable for spreadsheets
- JSON: Complete data with metadata
- PDF: Formatted reports (placeholder for now)
"""

import csv
import json
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime
from loguru import logger


def export_to_csv(
    run_data: Dict[str, Any],
    output_path: Path,
    options: Dict[str, Any]
) -> None:
    """
    Export evaluation results to CSV format.
    
    Args:
        run_data: Complete run data including results
        output_path: Path to save the CSV file
        options: Export options (include_raw, include_metrics, etc.)
    """
    try:
        # Extract results
        results = run_data.get('results', [])
        
        if not results:
            # Create a summary CSV
            with open(output_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Run ID', 'Model', 'Task', 'Status', 'Total Cost', 'Samples'])
                writer.writerow([
                    run_data.get('run_id', ''),
                    run_data.get('model', ''),
                    run_data.get('task', ''),
                    run_data.get('status', ''),
                    run_data.get('total_cost', 0),
                    run_data.get('completed_samples', 0)
                ])
        else:
            # Create detailed results CSV
            with open(output_path, 'w', newline='') as f:
                # Determine columns based on first result
                first_result = results[0] if results else {}
                columns = list(first_result.keys())
                
                writer = csv.DictWriter(f, fieldnames=columns)
                writer.writeheader()
                writer.writerows(results)
        
        logger.info(f"Exported CSV to {output_path}")
        
    except Exception as e:
        logger.error(f"Failed to export CSV: {e}")
        raise


def export_to_json(
    run_data: Dict[str, Any],
    output_path: Path,
    options: Dict[str, Any]
) -> None:
    """
    Export evaluation results to JSON format.
    
    Args:
        run_data: Complete run data including results
        output_path: Path to save the JSON file
        options: Export options
    """
    try:
        # Build export data based on options
        export_data = {
            'metadata': {
                'run_id': run_data.get('run_id'),
                'exported_at': datetime.now().isoformat(),
                'model': run_data.get('model'),
                'task': run_data.get('task'),
                'dataset': run_data.get('dataset')
            }
        }
        
        if options.get('include_config', True):
            export_data['config'] = run_data.get('config', {})
        
        if options.get('include_metrics', True):
            export_data['metrics'] = run_data.get('metrics', {})
        
        if options.get('include_raw', True):
            export_data['results'] = run_data.get('results', [])
        
        if options.get('add_summary', True):
            export_data['summary'] = {
                'total_samples': run_data.get('total_samples', 0),
                'completed_samples': run_data.get('completed_samples', 0),
                'total_cost': run_data.get('total_cost', 0),
                'duration': run_data.get('duration', 'N/A'),
                'status': run_data.get('status', 'unknown')
            }
        
        # Write JSON with proper formatting
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        logger.info(f"Exported JSON to {output_path}")
        
    except Exception as e:
        logger.error(f"Failed to export JSON: {e}")
        raise


def export_to_pdf(
    run_data: Dict[str, Any],
    output_path: Path,
    options: Dict[str, Any]
) -> None:
    """
    Export evaluation results to PDF format.
    
    NOTE: This is a placeholder. Full PDF generation would require
    additional dependencies like reportlab or weasyprint.
    
    Args:
        run_data: Complete run data including results
        output_path: Path to save the PDF file
        options: Export options
    """
    try:
        # For now, create a text file as placeholder
        with open(output_path.with_suffix('.txt'), 'w') as f:
            f.write("EVALUATION REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Run ID: {run_data.get('run_id', 'N/A')}\n")
            f.write(f"Model: {run_data.get('model', 'N/A')}\n")
            f.write(f"Task: {run_data.get('task', 'N/A')}\n")
            f.write(f"Dataset: {run_data.get('dataset', 'N/A')}\n")
            f.write(f"Status: {run_data.get('status', 'N/A')}\n")
            f.write(f"Duration: {run_data.get('duration', 'N/A')}\n")
            f.write(f"Total Cost: ${run_data.get('total_cost', 0):.2f}\n")
            f.write(f"Samples: {run_data.get('completed_samples', 0)} / {run_data.get('total_samples', 0)}\n")
            
            if run_data.get('metrics'):
                f.write("\nMETRICS\n")
                f.write("-" * 30 + "\n")
                for metric, value in run_data['metrics'].items():
                    f.write(f"{metric}: {value}\n")
            
            f.write("\n" + "=" * 50 + "\n")
            f.write("Note: PDF generation not yet implemented. This is a text placeholder.\n")
        
        logger.info(f"Exported text report to {output_path.with_suffix('.txt')}")
        
    except Exception as e:
        logger.error(f"Failed to export PDF: {e}")
        raise