# file_operation_hooks.py
"""
File Operation Hooks for Claude Code Audit System

This module provides integration hooks that can be injected into Claude Code's
file operation functions to automatically monitor and audit all file changes.
"""

import asyncio
import functools
import inspect
from typing import Any, Callable, Optional, Dict
from pathlib import Path

from loguru import logger

from .code_audit_tool import record_file_operation, set_user_prompt


class FileOperationMonitor:
    """Monitors and hooks into file operations for auditing."""
    
    def __init__(self):
        self.current_user_prompt: Optional[str] = None
        self.hooked_functions = []
    
    def set_user_context(self, prompt: str):
        """Set the current user context/prompt."""
        self.current_user_prompt = prompt
        set_user_prompt(prompt)
        logger.info(f"File audit context set: {prompt[:100]}...")
    
    def hook_function(self, original_func: Callable, operation_type: str) -> Callable:
        """
        Create a wrapper that hooks into a file operation function.
        
        Args:
            original_func: The original function to hook
            operation_type: Type of operation (Read, Write, Edit, MultiEdit)
        """
        if asyncio.iscoroutinefunction(original_func):
            @functools.wraps(original_func)
            async def async_wrapper(*args, **kwargs):
                return await self._monitor_async_operation(
                    original_func, operation_type, args, kwargs
                )
            return async_wrapper
        else:
            @functools.wraps(original_func)
            def sync_wrapper(*args, **kwargs):
                return asyncio.run(self._monitor_sync_operation(
                    original_func, operation_type, args, kwargs
                ))
            return sync_wrapper
    
    async def _monitor_async_operation(self, func: Callable, operation_type: str, 
                                     args: tuple, kwargs: dict) -> Any:
        """Monitor an async file operation."""
        # Extract file path and content from arguments
        file_path, content_before, content_after = self._extract_operation_details(
            func, operation_type, args, kwargs
        )
        
        # Call the original function
        try:
            result = await func(*args, **kwargs)
            
            # Record the operation for audit
            if file_path:
                await self._record_operation(
                    operation_type, file_path, content_before, content_after, 
                    self.current_user_prompt, {"function": func.__name__, "args_count": len(args)}
                )
            
            return result
            
        except Exception as e:
            # Still record failed operations
            if file_path:
                await self._record_operation(
                    f"{operation_type}_FAILED", file_path, content_before, None,
                    self.current_user_prompt, {"function": func.__name__, "error": str(e)}
                )
            raise
    
    async def _monitor_sync_operation(self, func: Callable, operation_type: str,
                                    args: tuple, kwargs: dict) -> Any:
        """Monitor a sync file operation."""
        # Extract file path and content from arguments
        file_path, content_before, content_after = self._extract_operation_details(
            func, operation_type, args, kwargs
        )
        
        # Call the original function in thread
        try:
            result = await asyncio.to_thread(func, *args, **kwargs)
            
            # Record the operation for audit
            if file_path:
                await self._record_operation(
                    operation_type, file_path, content_before, content_after,
                    self.current_user_prompt, {"function": func.__name__, "args_count": len(args)}
                )
            
            return result
            
        except Exception as e:
            # Still record failed operations
            if file_path:
                await self._record_operation(
                    f"{operation_type}_FAILED", file_path, content_before, None,
                    self.current_user_prompt, {"function": func.__name__, "error": str(e)}
                )
            raise
    
    def _extract_operation_details(self, func: Callable, operation_type: str, 
                                 args: tuple, kwargs: dict) -> tuple[Optional[str], Optional[str], Optional[str]]:
        """
        Extract file path and content from function arguments.
        
        Returns:
            Tuple of (file_path, content_before, content_after)
        """
        file_path = None
        content_before = None
        content_after = None
        
        # Get function signature
        sig = inspect.signature(func)
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()
        
        # Extract file_path (common parameter names)
        for param_name in ['file_path', 'path', 'filepath']:
            if param_name in bound_args.arguments:
                file_path = str(bound_args.arguments[param_name])
                break
        
        # Extract content based on operation type
        if operation_type == "Read":
            # For Read operations, we'll get content_after from the result
            # This is handled in the calling wrapper
            pass
        elif operation_type == "Write":
            content_after = bound_args.arguments.get('content', bound_args.arguments.get('data'))
        elif operation_type in ["Edit", "MultiEdit"]:
            content_before = self._try_read_existing_file(file_path)
            if operation_type == "Edit":
                # For Edit, we need to reconstruct the new content
                old_string = bound_args.arguments.get('old_string', '')
                new_string = bound_args.arguments.get('new_string', '')
                if content_before and old_string and new_string:
                    content_after = content_before.replace(old_string, new_string)
            elif operation_type == "MultiEdit":
                # For MultiEdit, apply all edits
                content_after = self._apply_multi_edits(content_before, bound_args.arguments.get('edits', []))
        
        return file_path, content_before, content_after
    
    def _try_read_existing_file(self, file_path: Optional[str]) -> Optional[str]:
        """Try to read existing file content."""
        if not file_path:
            return None
        
        try:
            path = Path(file_path)
            if path.exists() and path.is_file():
                return path.read_text(encoding='utf-8', errors='ignore')
        except Exception as e:
            logger.debug(f"Could not read existing file {file_path}: {e}")
        
        return None
    
    def _apply_multi_edits(self, content: Optional[str], edits: list) -> Optional[str]:
        """Apply multiple edits to content."""
        if not content or not edits:
            return content
        
        result = content
        try:
            for edit in edits:
                old_string = edit.get('old_string', '')
                new_string = edit.get('new_string', '')
                replace_all = edit.get('replace_all', False)
                
                if old_string and new_string is not None:
                    if replace_all:
                        result = result.replace(old_string, new_string)
                    else:
                        result = result.replace(old_string, new_string, 1)
            
            return result
        except Exception as e:
            logger.warning(f"Failed to apply multi-edits: {e}")
            return content
    
    async def _record_operation(self, operation_type: str, file_path: str,
                              content_before: Optional[str], content_after: Optional[str],
                              user_prompt: Optional[str], operation_details: Dict[str, Any]):
        """Record the file operation for audit."""
        try:
            await record_file_operation(
                operation_type=operation_type,
                file_path=file_path,
                content_before=content_before,
                content_after=content_after,
                user_prompt=user_prompt,
                operation_details=operation_details
            )
        except Exception as e:
            logger.error(f"Failed to record file operation: {e}")


# Global monitor instance
_monitor = FileOperationMonitor()


def install_claude_code_hooks():
    """
    Install file operation hooks into Claude Code functions.
    
    This function attempts to hook into the main file operation functions
    used by Claude Code tools.
    """
    try:
        # Import the actual functions we want to hook
        # Note: These imports may fail if the modules don't exist yet
        
        # Hook the primary tool functions if they exist
        hook_attempts = []
        
        # Try to hook Read tool
        try:
            import tldw_chatbook.Tools.file_operation_tools as file_tools
            if hasattr(file_tools, 'ReadFileTool'):
                original_read = file_tools.ReadFileTool.execute
                file_tools.ReadFileTool.execute = _monitor.hook_function(original_read, "Read")
                hook_attempts.append("ReadFileTool.execute")
        except ImportError:
            logger.debug("ReadFileTool not available for hooking")
        
        # Try to hook Write tool  
        try:
            import tldw_chatbook.Tools.file_operation_tools as file_tools
            if hasattr(file_tools, 'WriteFileTool'):
                original_write = file_tools.WriteFileTool.execute
                file_tools.WriteFileTool.execute = _monitor.hook_function(original_write, "Write")
                hook_attempts.append("WriteFileTool.execute")
        except ImportError:
            logger.debug("WriteFileTool not available for hooking")
        
        # The actual Edit and MultiEdit functions would be in Claude Code's internals
        # Since we can't directly access them, we'll provide instructions for manual integration
        
        logger.info(f"File operation hooks installed: {', '.join(hook_attempts)}")
        
        if not hook_attempts:
            logger.warning("No file operations were hooked. Manual integration may be required.")
            
        return len(hook_attempts)
        
    except Exception as e:
        logger.error(f"Failed to install file operation hooks: {e}")
        return 0


def set_user_context(prompt: str):
    """Set the current user context for file auditing."""
    _monitor.set_user_context(prompt)


def get_monitor() -> FileOperationMonitor:
    """Get the global file operation monitor."""
    return _monitor


# Integration instructions for manual hooking
INTEGRATION_INSTRUCTIONS = """
FILE OPERATION AUDIT INTEGRATION INSTRUCTIONS

To manually integrate file operation auditing into Claude Code:

1. In the main Claude Code request processing loop, call:
   ```python
   from tldw_chatbook.Tools.file_operation_hooks import set_user_context
   set_user_context(user_prompt)
   ```

2. Before any file operations (Read, Write, Edit, MultiEdit), call:
   ```python
   from tldw_chatbook.Tools.code_audit_tool import record_file_operation
   
   # For file reads:
   await record_file_operation("Read", file_path, content_after=file_content)
   
   # For file writes:
   await record_file_operation("Write", file_path, content_after=new_content)
   
   # For file edits:
   await record_file_operation("Edit", file_path, content_before=old_content, content_after=new_content)
   ```

3. Register the audit tool in tool_executor.py by adding:
   ```python
   from .code_audit_tool import CodeAuditTool
   _global_executor.register_tool(CodeAuditTool())
   ```
"""