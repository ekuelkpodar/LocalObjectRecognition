"""
Configuration management module for loading and validating settings.
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any


class Config:
    """Configuration manager for the object recognition system."""

    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize configuration from YAML file.

        Args:
            config_path: Path to the configuration YAML file
        """
        self.config_path = config_path
        self.config = self._load_config()
        self._validate_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)

        return config

    def _validate_config(self):
        """Validate that required configuration sections exist."""
        required_sections = ['camera', 'model', 'processing', 'prompts', 'logging']
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required configuration section: {section}")

    @property
    def camera(self) -> Dict[str, Any]:
        """Get camera configuration."""
        return self.config['camera']

    @property
    def model(self) -> Dict[str, Any]:
        """Get model configuration."""
        return self.config['model']

    @property
    def processing(self) -> Dict[str, Any]:
        """Get processing configuration."""
        return self.config['processing']

    @property
    def prompts(self) -> Dict[str, str]:
        """Get prompt templates."""
        return self.config['prompts']

    @property
    def logging(self) -> Dict[str, Any]:
        """Get logging configuration."""
        return self.config['logging']

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value by key.

        Args:
            key: Configuration key (supports nested keys with dot notation)
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        keys = key.split('.')
        value = self.config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def reload(self):
        """Reload configuration from file."""
        self.config = self._load_config()
        self._validate_config()


class PromptTemplateManager:
    """Manages prompt templates from configuration file."""

    def __init__(self, templates_path: str = "prompts/templates.yaml"):
        """
        Initialize prompt template manager.

        Args:
            templates_path: Path to prompt templates YAML file
        """
        self.templates_path = templates_path
        self.templates = self._load_templates()
        self.current_template_key = None

    def _load_templates(self) -> Dict[str, Dict[str, str]]:
        """Load prompt templates from YAML file."""
        if not os.path.exists(self.templates_path):
            return {}

        with open(self.templates_path, 'r') as f:
            data = yaml.safe_load(f)

        return data.get('templates', {})

    def get_template(self, template_key: str) -> str:
        """
        Get a specific prompt template.

        Args:
            template_key: Key of the template to retrieve

        Returns:
            Prompt template string
        """
        if template_key in self.templates:
            return self.templates[template_key]['prompt']
        return ""

    def get_template_by_number(self, number: int) -> str:
        """
        Get template by number key (1-5).

        Args:
            number: Number key (1-5)

        Returns:
            Prompt template string
        """
        for key, template in self.templates.items():
            if template.get('key') == str(number):
                self.current_template_key = key
                return template['prompt']
        return ""

    def list_templates(self) -> Dict[str, Dict[str, str]]:
        """Get all available templates."""
        return self.templates

    def get_default_prompt(self) -> str:
        """Get the default prompt template."""
        if self.templates:
            first_key = list(self.templates.keys())[0]
            return self.templates[first_key]['prompt']
        return "Describe all objects you see in this image."
