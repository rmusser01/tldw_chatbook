# distribution_manager.py
# Description: Distribution manager for briefings via email, webhooks, and cloud storage
#
# This module handles distributing briefings through various channels:
# - Email (SMTP)
# - Webhooks (Discord, Slack, generic)
# - Cloud storage (Dropbox, Google Drive, OneDrive)
# - Read later services (Pocket, Instapaper)
#
# Imports
import os
import json
import asyncio
import smtplib
import mimetypes
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List, Union, Tuple
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
#
# Third-Party Imports
import httpx
from loguru import logger
#
# Local Imports
from ..config import get_user_config
from ..Security.config_encryption import ConfigEncryption
from ..Metrics.metrics_logger import log_counter, log_histogram
#
########################################################################################################################
#
# Distribution Manager
#
########################################################################################################################

class DistributionManager:
    """Manages distributing briefings through various channels."""
    
    def __init__(self):
        """Initialize distribution manager."""
        self.config = get_user_config()
        self.encryption = ConfigEncryption()
        self._load_credentials()
    
    def _load_credentials(self):
        """Load and decrypt distribution credentials."""
        self.smtp_config = self._get_encrypted_config('smtp', {})
        self.webhook_config = self._get_encrypted_config('webhooks', {})
        self.cloud_config = self._get_encrypted_config('cloud_storage', {})
    
    def _get_encrypted_config(self, section: str, default: dict) -> dict:
        """Get and decrypt configuration section."""
        try:
            config_data = self.config.get(f'distribution.{section}', default)
            
            # Decrypt sensitive fields
            if section == 'smtp' and 'password' in config_data:
                config_data['password'] = self.encryption.decrypt_value(config_data['password'])
            
            return config_data
        except Exception as e:
            logger.error(f"Error loading {section} config: {str(e)}")
            return default
    
    async def distribute(self,
                        content: str,
                        format: str,
                        metadata: Dict[str, Any],
                        channels: List[str],
                        attachments: Optional[List[Path]] = None) -> Dict[str, Any]:
        """
        Distribute content through specified channels.
        
        Args:
            content: Content to distribute
            format: Content format (markdown, html, pdf, etc.)
            metadata: Content metadata
            channels: List of distribution channels
            attachments: Optional file attachments
            
        Returns:
            Distribution results
        """
        results = {}
        
        for channel in channels:
            try:
                if channel == 'email':
                    results['email'] = await self._distribute_email(content, format, metadata, attachments)
                elif channel.startswith('webhook:'):
                    webhook_type = channel.split(':', 1)[1]
                    results[channel] = await self._distribute_webhook(content, format, metadata, webhook_type)
                elif channel.startswith('cloud:'):
                    cloud_type = channel.split(':', 1)[1]
                    results[channel] = await self._distribute_cloud(content, format, metadata, attachments, cloud_type)
                else:
                    logger.warning(f"Unknown distribution channel: {channel}")
                    results[channel] = {'success': False, 'error': 'Unknown channel'}
            
            except Exception as e:
                logger.error(f"Error distributing to {channel}: {str(e)}")
                results[channel] = {'success': False, 'error': str(e)}
        
        return results
    
    async def _distribute_email(self,
                              content: str,
                              format: str,
                              metadata: Dict[str, Any],
                              attachments: Optional[List[Path]] = None) -> Dict[str, Any]:
        """Distribute via email."""
        start_time = datetime.now()
        
        try:
            # Get SMTP configuration
            smtp_host = self.smtp_config.get('host', 'smtp.gmail.com')
            smtp_port = self.smtp_config.get('port', 587)
            smtp_user = self.smtp_config.get('username')
            smtp_pass = self.smtp_config.get('password')
            from_addr = self.smtp_config.get('from_address', smtp_user)
            to_addrs = self.smtp_config.get('to_addresses', [])
            
            if not smtp_user or not smtp_pass:
                return {'success': False, 'error': 'SMTP credentials not configured'}
            
            if not to_addrs:
                return {'success': False, 'error': 'No recipient addresses configured'}
            
            # Create message
            msg = MIMEMultipart('mixed')
            msg['From'] = from_addr
            msg['To'] = ', '.join(to_addrs)
            msg['Subject'] = f"{metadata.get('name', 'Briefing')} - {datetime.now().strftime('%Y-%m-%d')}"
            
            # Add metadata headers
            msg['X-Briefing-Generated'] = metadata.get('generated_at', datetime.now().isoformat())
            msg['X-Briefing-Items'] = str(metadata.get('item_count', 0))
            msg['X-Briefing-Sources'] = str(metadata.get('source_count', 0))
            
            # Create email body based on format
            if format == 'html':
                body = MIMEText(content, 'html')
            else:
                # Convert markdown to simple HTML for better email rendering
                html_content = self._markdown_to_email_html(content, metadata)
                body = MIMEText(html_content, 'html')
            
            msg.attach(body)
            
            # Add attachments
            if attachments:
                for attachment_path in attachments:
                    if attachment_path.exists():
                        self._attach_file(msg, attachment_path)
            
            # Send email
            with smtplib.SMTP(smtp_host, smtp_port) as server:
                server.starttls()
                server.login(smtp_user, smtp_pass)
                server.send_message(msg)
            
            # Log metrics
            duration = (datetime.now() - start_time).total_seconds()
            log_histogram("distribution_duration", duration, labels={"channel": "email"})
            log_counter("distributions_sent", labels={"channel": "email", "status": "success"})
            
            logger.info(f"Email sent to {len(to_addrs)} recipients")
            return {
                'success': True,
                'recipients': to_addrs,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            log_counter("distributions_sent", labels={"channel": "email", "status": "error"})
            logger.error(f"Error sending email: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    async def _distribute_webhook(self,
                                content: str,
                                format: str,
                                metadata: Dict[str, Any],
                                webhook_type: str) -> Dict[str, Any]:
        """Distribute via webhook."""
        try:
            webhook_url = self.webhook_config.get(f'{webhook_type}_url')
            if not webhook_url:
                return {'success': False, 'error': f'No webhook URL configured for {webhook_type}'}
            
            # Format payload based on webhook type
            if webhook_type == 'discord':
                payload = self._format_discord_payload(content, metadata)
            elif webhook_type == 'slack':
                payload = self._format_slack_payload(content, metadata)
            else:
                # Generic webhook
                payload = {
                    'content': content,
                    'format': format,
                    'metadata': metadata
                }
            
            # Send webhook
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(webhook_url, json=payload)
                response.raise_for_status()
            
            log_counter("distributions_sent", labels={"channel": f"webhook_{webhook_type}", "status": "success"})
            
            return {
                'success': True,
                'webhook_type': webhook_type,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            log_counter("distributions_sent", labels={"channel": f"webhook_{webhook_type}", "status": "error"})
            logger.error(f"Error sending {webhook_type} webhook: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def _format_discord_payload(self, content: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Format payload for Discord webhook."""
        # Discord has a 2000 character limit for content
        if len(content) > 1900:
            content = content[:1900] + "...\n*[Content truncated]*"
        
        # Create embed for rich formatting
        embed = {
            "title": metadata.get('name', 'Briefing'),
            "description": f"Generated on {metadata.get('generated_at', datetime.now().isoformat())}",
            "color": 3447003,  # Blue color
            "fields": [
                {
                    "name": "Items",
                    "value": str(metadata.get('item_count', 0)),
                    "inline": True
                },
                {
                    "name": "Sources",
                    "value": str(metadata.get('source_count', 0)),
                    "inline": True
                }
            ],
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        return {
            "content": content,
            "embeds": [embed],
            "username": "Briefing Bot"
        }
    
    def _format_slack_payload(self, content: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Format payload for Slack webhook."""
        # Convert markdown to Slack's mrkdwn format
        slack_content = content.replace('**', '*')  # Bold
        slack_content = slack_content.replace('__', '_')  # Italic
        
        return {
            "text": metadata.get('name', 'Briefing'),
            "blocks": [
                {
                    "type": "header",
                    "text": {
                        "type": "plain_text",
                        "text": metadata.get('name', 'Briefing')
                    }
                },
                {
                    "type": "context",
                    "elements": [
                        {
                            "type": "mrkdwn",
                            "text": f"Generated: {metadata.get('generated_at', datetime.now().isoformat())}"
                        }
                    ]
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": slack_content[:3000]  # Slack limit
                    }
                },
                {
                    "type": "context",
                    "elements": [
                        {
                            "type": "mrkdwn",
                            "text": f"*Items:* {metadata.get('item_count', 0)} | *Sources:* {metadata.get('source_count', 0)}"
                        }
                    ]
                }
            ]
        }
    
    async def _distribute_cloud(self,
                              content: str,
                              format: str,
                              metadata: Dict[str, Any],
                              attachments: Optional[List[Path]],
                              cloud_type: str) -> Dict[str, Any]:
        """Distribute to cloud storage."""
        # This would require implementing specific cloud provider APIs
        # For now, return a placeholder
        logger.warning(f"Cloud storage distribution not yet implemented for {cloud_type}")
        return {
            'success': False,
            'error': f'Cloud storage {cloud_type} not yet implemented'
        }
    
    def _markdown_to_email_html(self, content: str, metadata: Dict[str, Any]) -> str:
        """Convert markdown to email-friendly HTML."""
        import markdown
        
        # Convert markdown to HTML
        html_body = markdown.markdown(content, extensions=['extra'])
        
        # Create email-friendly HTML
        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body {{
            font-family: -apple-system, Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 600px;
            margin: 0 auto;
            padding: 20px;
        }}
        h1, h2, h3 {{
            color: #2c3e50;
            margin-top: 20px;
        }}
        h1 {{
            font-size: 24px;
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            font-size: 20px;
        }}
        h3 {{
            font-size: 18px;
        }}
        a {{
            color: #3498db;
            text-decoration: none;
        }}
        ul, ol {{
            padding-left: 30px;
        }}
        li {{
            margin-bottom: 5px;
        }}
        .footer {{
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
            color: #666;
            font-size: 14px;
        }}
        .metadata {{
            background-color: #f5f5f5;
            padding: 10px;
            border-radius: 5px;
            margin-bottom: 20px;
            font-size: 14px;
        }}
    </style>
</head>
<body>
    <h1>{metadata.get('name', 'Briefing')}</h1>
    <div class="metadata">
        <strong>Generated:</strong> {metadata.get('generated_at', datetime.now().isoformat())}<br>
        <strong>Items:</strong> {metadata.get('item_count', 0)} | <strong>Sources:</strong> {metadata.get('source_count', 0)}
    </div>
    {html_body}
    <div class="footer">
        <p>This is an automated briefing generated by your subscription system.</p>
    </div>
</body>
</html>"""
        
        return html
    
    def _attach_file(self, msg: MIMEMultipart, file_path: Path):
        """Attach a file to email message."""
        try:
            # Guess content type
            ctype, encoding = mimetypes.guess_type(str(file_path))
            if ctype is None or encoding is not None:
                ctype = 'application/octet-stream'
            
            maintype, subtype = ctype.split('/', 1)
            
            # Read file
            with open(file_path, 'rb') as f:
                attachment = MIMEBase(maintype, subtype)
                attachment.set_payload(f.read())
            
            # Encode
            encoders.encode_base64(attachment)
            
            # Add header
            attachment.add_header(
                'Content-Disposition',
                f'attachment; filename="{file_path.name}"'
            )
            
            msg.attach(attachment)
            
        except Exception as e:
            logger.error(f"Error attaching file {file_path}: {str(e)}")
    
    def configure_smtp(self,
                      host: str,
                      port: int,
                      username: str,
                      password: str,
                      from_address: Optional[str] = None,
                      to_addresses: Optional[List[str]] = None) -> bool:
        """
        Configure SMTP settings.
        
        Args:
            host: SMTP server host
            port: SMTP server port
            username: SMTP username
            password: SMTP password
            from_address: From email address
            to_addresses: List of recipient addresses
            
        Returns:
            Success status
        """
        try:
            # Encrypt password
            encrypted_password = self.encryption.encrypt_value(password)
            
            # Update configuration
            self.smtp_config = {
                'host': host,
                'port': port,
                'username': username,
                'password': encrypted_password,
                'from_address': from_address or username,
                'to_addresses': to_addresses or []
            }
            
            # Save to config
            config = get_user_config()
            config['distribution'] = config.get('distribution', {})
            config['distribution']['smtp'] = self.smtp_config
            
            # Save config file
            # TODO: Implement config save
            
            return True
            
        except Exception as e:
            logger.error(f"Error configuring SMTP: {str(e)}")
            return False
    
    def configure_webhook(self, webhook_type: str, url: str) -> bool:
        """
        Configure webhook URL.
        
        Args:
            webhook_type: Type of webhook (discord, slack, generic)
            url: Webhook URL
            
        Returns:
            Success status
        """
        try:
            self.webhook_config[f'{webhook_type}_url'] = url
            
            # Save to config
            config = get_user_config()
            config['distribution'] = config.get('distribution', {})
            config['distribution']['webhooks'] = self.webhook_config
            
            # TODO: Implement config save
            
            return True
            
        except Exception as e:
            logger.error(f"Error configuring webhook: {str(e)}")
            return False


# End of distribution_manager.py