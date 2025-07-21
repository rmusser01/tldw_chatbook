"""
Event handlers for chunking template operations.
"""

from textual.message import Message


class TemplateEvent(Message):
    """Base class for template-related events."""
    pass


class TemplateDeleteConfirmationEvent(TemplateEvent):
    """Event to request confirmation before deleting a template."""
    
    def __init__(self, template_id: int, template_name: str):
        super().__init__()
        self.template_id = template_id
        self.template_name = template_name


class TemplateDeletedEvent(TemplateEvent):
    """Event fired after a template is successfully deleted."""
    
    def __init__(self, template_id: int):
        super().__init__()
        self.template_id = template_id


class TemplateCreatedEvent(TemplateEvent):
    """Event fired after a template is successfully created."""
    
    def __init__(self, template_id: int, template_name: str):
        super().__init__()
        self.template_id = template_id
        self.template_name = template_name


class TemplateUpdatedEvent(TemplateEvent):
    """Event fired after a template is successfully updated."""
    
    def __init__(self, template_id: int, template_name: str):
        super().__init__()
        self.template_id = template_id
        self.template_name = template_name


class TemplateImportedEvent(TemplateEvent):
    """Event fired after a template is successfully imported."""
    
    def __init__(self, template_name: str):
        super().__init__()
        self.template_name = template_name


class TemplateExportedEvent(TemplateEvent):
    """Event fired after a template is successfully exported."""
    
    def __init__(self, template_name: str, export_path: str):
        super().__init__()
        self.template_name = template_name
        self.export_path = export_path