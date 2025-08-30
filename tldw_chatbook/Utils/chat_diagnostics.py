#!/usr/bin/env python3
"""
Chat Widget Structure Diagnostic Tool

This tool helps diagnose the actual structure of chat widgets
to understand how to properly save and restore state.
"""

from typing import Any, Dict, List, Optional, Set
from datetime import datetime
from loguru import logger
from textual.widget import Widget
from textual.widgets import TextArea, Button, Static, Label
from textual.containers import Container

logger = logger.bind(module="ChatDiagnostics")


class ChatDiagnostics:
    """Diagnostic tool for inspecting chat widget structure."""
    
    def __init__(self):
        self.report = []
        self.widget_count = {}
        self.text_areas_found = []
        self.containers_found = []
        self.input_widgets = []
        
    def inspect_widget_tree(self, root_widget: Widget, max_depth: int = 10) -> Dict[str, Any]:
        """
        Recursively inspect the widget tree and build a diagnostic report.
        
        Args:
            root_widget: The root widget to start inspection from
            max_depth: Maximum depth to traverse
            
        Returns:
            A diagnostic report dictionary
        """
        logger.info("Starting chat widget structure inspection")
        self.report = []
        self.widget_count = {}
        self.text_areas_found = []
        self.containers_found = []
        self.input_widgets = []
        
        # Start recursive inspection
        self._inspect_recursive(root_widget, depth=0, max_depth=max_depth)
        
        # Build summary report
        report = {
            "timestamp": datetime.now().isoformat(),
            "root_widget": {
                "class": root_widget.__class__.__name__,
                "id": root_widget.id,
                "has_tabs": self._check_for_tabs(root_widget)
            },
            "widget_counts": self.widget_count,
            "text_areas": self._summarize_text_areas(),
            "input_widgets": self._summarize_input_widgets(),
            "containers": self._summarize_containers(),
            "chat_structure": self._analyze_chat_structure(),
            "recommendations": self._generate_recommendations(),
            "detailed_tree": self.report[:100]  # Limit to first 100 entries
        }
        
        return report
    
    def _inspect_recursive(self, widget: Widget, depth: int, max_depth: int, parent_path: str = "") -> None:
        """Recursively inspect widgets and collect information."""
        if depth > max_depth:
            return
            
        # Build path
        widget_id = widget.id or f"unnamed_{widget.__class__.__name__}"
        current_path = f"{parent_path}/{widget_id}" if parent_path else widget_id
        
        # Count widget types
        widget_type = widget.__class__.__name__
        self.widget_count[widget_type] = self.widget_count.get(widget_type, 0) + 1
        
        # Collect specific widget info
        widget_info = {
            "path": current_path,
            "depth": depth,
            "type": widget_type,
            "id": widget.id,
            "classes": list(widget.classes) if hasattr(widget, 'classes') else [],
            "children_count": len(widget.children) if hasattr(widget, 'children') else 0
        }
        
        # Special handling for TextArea
        if isinstance(widget, TextArea):
            text_area_info = {
                **widget_info,
                "has_text": bool(widget.text if hasattr(widget, 'text') else False),
                "text_preview": (widget.text[:100] + "...") if hasattr(widget, 'text') and widget.text else "",
                "is_disabled": widget.disabled if hasattr(widget, 'disabled') else False,
                "is_visible": widget.styles.display != "none" if hasattr(widget, 'styles') else True
            }
            self.text_areas_found.append(text_area_info)
            
            # Check if this might be an input widget
            if widget.id and ('input' in widget.id.lower() or 'message' in widget.id.lower()):
                self.input_widgets.append(text_area_info)
        
        # Special handling for Containers
        if isinstance(widget, Container):
            container_info = {
                **widget_info,
                "might_be_chat_log": any(keyword in str(widget.id).lower() for keyword in ['chat', 'log', 'message', 'history']) if widget.id else False,
                "might_be_tab_container": any(keyword in str(widget.id).lower() for keyword in ['tab', 'session']) if widget.id else False
            }
            self.containers_found.append(container_info)
        
        # Add to report
        self.report.append(widget_info)
        
        # Recurse to children
        if hasattr(widget, 'children'):
            for child in widget.children:
                self._inspect_recursive(child, depth + 1, max_depth, current_path)
    
    def _check_for_tabs(self, root_widget: Widget) -> bool:
        """Check if the interface appears to have tabs."""
        # Look for tab-related widgets
        tab_indicators = ['ChatTabContainer', 'ChatTabBar', 'TabPane', 'TabbedContent']
        
        for widget in root_widget.walk_children():
            if widget.__class__.__name__ in tab_indicators:
                return True
            if widget.id and 'tab' in widget.id.lower():
                # Check if it's actually a tab widget, not just named with 'tab'
                if 'container' in widget.id.lower() or 'bar' in widget.id.lower():
                    return True
        
        return False
    
    def _summarize_text_areas(self) -> Dict[str, Any]:
        """Summarize found TextArea widgets."""
        return {
            "count": len(self.text_areas_found),
            "with_text": sum(1 for ta in self.text_areas_found if ta.get('has_text')),
            "likely_input": len(self.input_widgets),
            "details": self.text_areas_found[:5]  # First 5 for debugging
        }
    
    def _summarize_input_widgets(self) -> List[Dict[str, Any]]:
        """Summarize widgets that appear to be input fields."""
        return self.input_widgets
    
    def _summarize_containers(self) -> Dict[str, Any]:
        """Summarize container widgets."""
        chat_containers = [c for c in self.containers_found if c.get('might_be_chat_log')]
        tab_containers = [c for c in self.containers_found if c.get('might_be_tab_container')]
        
        return {
            "total_count": len(self.containers_found),
            "chat_containers": len(chat_containers),
            "tab_containers": len(tab_containers),
            "chat_container_ids": [c['id'] for c in chat_containers if c.get('id')],
            "tab_container_ids": [c['id'] for c in tab_containers if c.get('id')]
        }
    
    def _analyze_chat_structure(self) -> Dict[str, str]:
        """Analyze and determine the chat structure type."""
        has_tabs = any(c for c in self.containers_found if c.get('might_be_tab_container'))
        has_multiple_text_areas = len(self.text_areas_found) > 1
        has_chat_containers = any(c for c in self.containers_found if c.get('might_be_chat_log'))
        
        if has_tabs:
            structure_type = "tabbed"
            description = "Detected tabbed chat interface with multiple sessions"
        elif has_multiple_text_areas:
            structure_type = "multi-input"
            description = "Multiple input areas detected, possibly split interface"
        elif has_chat_containers:
            structure_type = "single"
            description = "Single chat interface with message container"
        else:
            structure_type = "unknown"
            description = "Could not determine chat structure type"
        
        return {
            "type": structure_type,
            "description": description,
            "confidence": "high" if has_tabs or has_chat_containers else "low"
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on the inspection."""
        recommendations = []
        
        # Check for input widgets
        if not self.input_widgets:
            recommendations.append("No clear input widgets found - check for TextArea with different IDs")
        elif len(self.input_widgets) > 1:
            recommendations.append(f"Multiple input widgets found ({len(self.input_widgets)}) - determine which is primary")
        
        # Check for tab structure
        has_tabs = any(c for c in self.containers_found if c.get('might_be_tab_container'))
        if has_tabs:
            recommendations.append("Tabbed interface detected - use ChatTabContainer methods for state")
        else:
            recommendations.append("Non-tabbed interface - save state directly from widgets")
        
        # Check for chat log
        has_chat_log = any(c for c in self.containers_found if c.get('might_be_chat_log'))
        if not has_chat_log:
            recommendations.append("No clear chat log container found - may need alternative message extraction")
        
        # TextArea recommendations
        if self.text_areas_found:
            visible_areas = [ta for ta in self.text_areas_found if ta.get('is_visible', True)]
            if visible_areas:
                recommendations.append(f"Focus on {len(visible_areas)} visible TextArea widgets for state capture")
        
        return recommendations
    
    def print_report(self, report: Dict[str, Any]) -> None:
        """Print a formatted diagnostic report."""
        print("\n" + "="*60)
        print("CHAT WIDGET STRUCTURE DIAGNOSTIC REPORT")
        print("="*60)
        print(f"Timestamp: {report['timestamp']}")
        print(f"Root Widget: {report['root_widget']['class']} (id={report['root_widget']['id']})")
        print(f"Has Tabs: {report['root_widget']['has_tabs']}")
        
        print("\n" + "-"*40)
        print("WIDGET COUNTS:")
        for widget_type, count in sorted(report['widget_counts'].items()):
            print(f"  {widget_type}: {count}")
        
        print("\n" + "-"*40)
        print("TEXT AREAS:")
        print(f"  Total: {report['text_areas']['count']}")
        print(f"  With Text: {report['text_areas']['with_text']}")
        print(f"  Likely Input: {report['text_areas']['likely_input']}")
        
        if report['text_areas']['details']:
            print("  Examples:")
            for ta in report['text_areas']['details'][:3]:
                print(f"    - {ta['id']} at {ta['path']}")
                if ta.get('text_preview'):
                    print(f"      Text: '{ta['text_preview'][:50]}...'")
        
        print("\n" + "-"*40)
        print("CONTAINERS:")
        print(f"  Total: {report['containers']['total_count']}")
        print(f"  Chat Containers: {report['containers']['chat_containers']}")
        if report['containers']['chat_container_ids']:
            print(f"    IDs: {', '.join(report['containers']['chat_container_ids'])}")
        print(f"  Tab Containers: {report['containers']['tab_containers']}")
        if report['containers']['tab_container_ids']:
            print(f"    IDs: {', '.join(report['containers']['tab_container_ids'])}")
        
        print("\n" + "-"*40)
        print("CHAT STRUCTURE ANALYSIS:")
        print(f"  Type: {report['chat_structure']['type']}")
        print(f"  Description: {report['chat_structure']['description']}")
        print(f"  Confidence: {report['chat_structure']['confidence']}")
        
        print("\n" + "-"*40)
        print("RECOMMENDATIONS:")
        for i, rec in enumerate(report['recommendations'], 1):
            print(f"  {i}. {rec}")
        
        print("\n" + "="*60)
    
    @staticmethod
    def run_diagnostic(chat_window: Widget) -> Dict[str, Any]:
        """
        Convenience method to run diagnostics on a chat window.
        
        Args:
            chat_window: The chat window widget to diagnose
            
        Returns:
            Diagnostic report dictionary
        """
        diagnostics = ChatDiagnostics()
        report = diagnostics.inspect_widget_tree(chat_window)
        
        # Log summary
        logger.info(f"Diagnostic complete: {report['chat_structure']['type']} structure with {report['text_areas']['count']} TextAreas")
        
        # Log recommendations
        for rec in report['recommendations']:
            logger.info(f"Recommendation: {rec}")
        
        return report


def diagnose_chat_screen(screen) -> Dict[str, Any]:
    """
    Run diagnostics on a ChatScreen instance.
    
    Args:
        screen: ChatScreen instance
        
    Returns:
        Diagnostic report
    """
    if not hasattr(screen, 'chat_window') or not screen.chat_window:
        logger.error("ChatScreen has no chat_window")
        return {"error": "No chat window found"}
    
    diagnostics = ChatDiagnostics()
    report = diagnostics.inspect_widget_tree(screen.chat_window)
    
    # Also print to console for debugging
    diagnostics.print_report(report)
    
    return report