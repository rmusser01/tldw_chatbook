"""
Password Dialog for Config Encryption
------------------------------------

Provides modal dialogs for entering master password for config file encryption.
Supports both initial password setup and password entry for decryption.
"""

from typing import Optional, Callable, Literal
from textual import on
from textual.app import ComposeResult
from textual.containers import Container, Vertical, Horizontal
from textual.screen import ModalScreen
from textual.widgets import Button, Label, Input, Static
from textual.validation import Length
from loguru import logger


class PasswordDialog(ModalScreen):
    """Dialog for entering master password for config encryption."""
    
    DEFAULT_CSS = """
    PasswordDialog {
        align: center middle;
    }
    
    PasswordDialog > Container {
        background: $surface;
        border: thick $primary;
        padding: 1 2;
        width: 60;
        height: auto;
        max-height: 25;
    }
    
    PasswordDialog .dialog-title {
        text-align: center;
        text-style: bold;
        margin-bottom: 1;
        color: $primary;
    }
    
    PasswordDialog .dialog-message {
        margin-bottom: 1;
        color: $text;
    }
    
    PasswordDialog .password-input {
        margin-bottom: 1;
        width: 100%;
    }
    
    PasswordDialog .error-message {
        color: $error;
        margin-bottom: 1;
        display: none;
    }
    
    PasswordDialog .error-message.visible {
        display: block;
    }
    
    PasswordDialog .button-container {
        align: center middle;
        margin-top: 1;
    }
    
    PasswordDialog Button {
        margin: 0 1;
        min-width: 12;
    }
    
    PasswordDialog .strength-indicator {
        height: 1;
        margin-bottom: 1;
    }
    
    PasswordDialog .strength-weak {
        color: $error;
    }
    
    PasswordDialog .strength-medium {
        color: $warning;
    }
    
    PasswordDialog .strength-strong {
        color: $success;
    }
    """
    
    def __init__(
        self,
        mode: Literal["setup", "unlock", "change"] = "unlock",
        title: Optional[str] = None,
        message: Optional[str] = None,
        on_submit: Optional[Callable[[str], None]] = None,
        on_cancel: Optional[Callable[[], None]] = None,
        name: Optional[str] = None,
    ):
        """
        Initialize the password dialog.
        
        Args:
            mode: The dialog mode - "setup" for initial setup, "unlock" for decryption, "change" for changing password
            title: Custom title for the dialog
            message: Custom message to display
            on_submit: Callback when password is submitted
            on_cancel: Callback when dialog is cancelled
            name: Name for the screen
        """
        super().__init__(name=name)
        self.mode = mode
        self.custom_title = title
        self.custom_message = message
        self.on_submit_callback = on_submit
        self.on_cancel_callback = on_cancel
        
        # Set default titles and messages based on mode
        if not self.custom_title:
            if mode == "setup":
                self.custom_title = "Setup Master Password"
            elif mode == "unlock":
                self.custom_title = "Enter Master Password"
            elif mode == "change":
                self.custom_title = "Change Master Password"
        
        if not self.custom_message:
            if mode == "setup":
                self.custom_message = "Create a master password to encrypt your API keys and sensitive configuration data."
            elif mode == "unlock":
                self.custom_message = "Enter your master password to decrypt the configuration file."
            elif mode == "change":
                self.custom_message = "Enter your current master password to change it."
    
    def compose(self) -> ComposeResult:
        """Create the dialog layout."""
        with Container():
            with Vertical():
                yield Label(self.custom_title, classes="dialog-title")
                yield Static(self.custom_message, classes="dialog-message")
                
                # Password input
                yield Input(
                    placeholder="Enter password",
                    password=True,
                    id="password-input",
                    classes="password-input",
                    validators=[Length(minimum=8, failure_description="Password must be at least 8 characters")]
                )
                
                # Confirm password for setup/change modes
                if self.mode in ["setup", "change"]:
                    yield Input(
                        placeholder="Confirm password",
                        password=True,
                        id="confirm-input",
                        classes="password-input"
                    )
                    
                    # Password strength indicator
                    yield Static("", id="strength-indicator", classes="strength-indicator")
                
                # Error message container
                yield Static("", id="error-message", classes="error-message")
                
                # Buttons
                with Horizontal(classes="button-container"):
                    yield Button("Cancel", variant="default", id="cancel-button")
                    yield Button("Submit", variant="primary", id="submit-button")
    
    def check_password_strength(self, password: str) -> tuple[str, str]:
        """
        Check password strength and return strength level and message.
        
        Returns:
            Tuple of (strength_class, strength_message)
        """
        if len(password) < 8:
            return "strength-weak", "Too short"
        
        has_upper = any(c.isupper() for c in password)
        has_lower = any(c.islower() for c in password)
        has_digit = any(c.isdigit() for c in password)
        has_special = any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password)
        
        strength_score = sum([has_upper, has_lower, has_digit, has_special])
        
        if strength_score <= 1:
            return "strength-weak", "Weak password"
        elif strength_score == 2:
            return "strength-medium", "Medium strength"
        else:
            return "strength-strong", "Strong password"
    
    @on(Input.Changed, "#password-input")
    def on_password_changed(self, event: Input.Changed) -> None:
        """Update password strength indicator when password changes."""
        if self.mode in ["setup", "change"]:
            strength_indicator = self.query_one("#strength-indicator", Static)
            if event.value:
                strength_class, strength_msg = self.check_password_strength(event.value)
                strength_indicator.update(f"Password strength: {strength_msg}")
                strength_indicator.remove_class("strength-weak", "strength-medium", "strength-strong")
                strength_indicator.add_class(strength_class)
            else:
                strength_indicator.update("")
    
    def show_error(self, message: str) -> None:
        """Display an error message."""
        error_widget = self.query_one("#error-message", Static)
        error_widget.update(message)
        error_widget.add_class("visible")
    
    def hide_error(self) -> None:
        """Hide the error message."""
        error_widget = self.query_one("#error-message", Static)
        error_widget.remove_class("visible")
    
    @on(Button.Pressed, "#submit-button")
    def on_submit(self) -> None:
        """Handle submit button press."""
        self.hide_error()
        
        password_input = self.query_one("#password-input", Input)
        password = password_input.value
        
        # Validate password
        if not password:
            self.show_error("Password cannot be empty")
            return
        
        if len(password) < 8:
            self.show_error("Password must be at least 8 characters")
            return
        
        # For setup/change modes, check password confirmation
        if self.mode in ["setup", "change"]:
            confirm_input = self.query_one("#confirm-input", Input)
            confirm_password = confirm_input.value
            
            if password != confirm_password:
                self.show_error("Passwords do not match")
                return
        
        # Call the callback if provided
        if self.on_submit_callback:
            try:
                self.on_submit_callback(password)
                self.dismiss(password)
            except Exception as e:
                logger.error(f"Error in password submit callback: {e}")
                self.show_error(str(e))
        else:
            self.dismiss(password)
    
    @on(Button.Pressed, "#cancel-button")
    def on_cancel(self) -> None:
        """Handle cancel button press."""
        if self.on_cancel_callback:
            self.on_cancel_callback()
        self.dismiss(None)
    
    @on(Input.Submitted)
    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle Enter key in input fields."""
        if event.input.id == "password-input" and self.mode in ["setup", "change"]:
            # Move focus to confirm input
            self.query_one("#confirm-input", Input).focus()
        else:
            # Submit the form
            self.on_submit()


class EncryptionSetupDialog(ModalScreen):
    """Dialog for setting up config encryption with API key detection."""
    
    DEFAULT_CSS = """
    EncryptionSetupDialog {
        align: center middle;
    }
    
    EncryptionSetupDialog > Container {
        background: $surface;
        border: thick $primary;
        padding: 2;
        width: 70;
        height: auto;
        max-height: 30;
    }
    
    EncryptionSetupDialog .dialog-title {
        text-align: center;
        text-style: bold;
        margin-bottom: 1;
        color: $primary;
    }
    
    EncryptionSetupDialog .info-section {
        margin-bottom: 1;
        padding: 1;
        background: $boost;
        border: solid $primary;
    }
    
    EncryptionSetupDialog .warning-section {
        margin-bottom: 1;
        padding: 1;
        background: $warning 20%;
        border: solid $warning;
    }
    
    EncryptionSetupDialog .api-key-list {
        margin: 1 0;
        padding-left: 2;
    }
    
    EncryptionSetupDialog .button-container {
        align: center middle;
        margin-top: 2;
    }
    
    EncryptionSetupDialog Button {
        margin: 0 1;
        min-width: 12;
    }
    """
    
    def __init__(
        self,
        detected_providers: list[str],
        on_proceed: Optional[Callable[[], None]] = None,
        on_cancel: Optional[Callable[[], None]] = None,
        name: Optional[str] = None,
    ):
        """
        Initialize the encryption setup dialog.
        
        Args:
            detected_providers: List of providers with detected API keys
            on_proceed: Callback when user proceeds with encryption
            on_cancel: Callback when user cancels
            name: Name for the screen
        """
        super().__init__(name=name)
        self.detected_providers = detected_providers
        self.on_proceed_callback = on_proceed
        self.on_cancel_callback = on_cancel
    
    def compose(self) -> ComposeResult:
        """Create the dialog layout."""
        with Container():
            with Vertical():
                yield Label("Config Encryption Setup", classes="dialog-title")
                
                # Info section
                with Vertical(classes="info-section"):
                    yield Static("🔐 API Keys Detected!")
                    yield Static(f"\nFound API keys for {len(self.detected_providers)} provider(s):")
                    
                    # List detected providers
                    api_key_list = "\n".join(f"• {provider}" for provider in self.detected_providers)
                    yield Static(api_key_list, classes="api-key-list")
                
                # Warning section
                with Vertical(classes="warning-section"):
                    yield Static("⚠️  Important:")
                    yield Static("• You'll need to enter this password every time you start the app")
                    yield Static("• If you forget the password, you'll need to re-enter your API keys")
                    yield Static("• Make sure to use a strong, memorable password")
                
                # Buttons
                with Horizontal(classes="button-container"):
                    yield Button("Not Now", variant="default", id="cancel-button")
                    yield Button("Setup Encryption", variant="primary", id="proceed-button")
    
    @on(Button.Pressed, "#proceed-button")
    def on_proceed(self) -> None:
        """Handle proceed button press."""
        if self.on_proceed_callback:
            self.on_proceed_callback()
        self.dismiss(True)
    
    @on(Button.Pressed, "#cancel-button")
    def on_cancel(self) -> None:
        """Handle cancel button press."""
        if self.on_cancel_callback:
            self.on_cancel_callback()
        self.dismiss(False)