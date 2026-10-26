"""
OpenAI client wrapper for AI agent interactions.
"""

import logging
from typing import List, Dict, Any, Optional
from openai import OpenAI

logger = logging.getLogger(__name__)


class AIClient:
    """
    Wrapper around OpenAI client for AI agent interactions.
    Handles API calls and response processing.
    """

    def __init__(self, api_key: str, client: Optional[OpenAI] = None):
        """
        Initialize the AI client.

        Args:
            client: Optional OpenAI client instance. If None, creates a new one.
        """
        self.client = client if client is not None else OpenAI(api_key=api_key)

    def create_response(self, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]], model: str = "gpt-4o", tool_choice: str = "auto"):
        """
        Create a response from the OpenAI API.

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            tools: List of tool definitions for function calling
            model: Model name to use (default: "gpt-4o")
            tool_choice: Tool choice strategy (default: "auto")

        Returns:
            Response object from OpenAI API
        """
        return self.client.responses.create(
            model=model,
            input=messages,
            tools=tools,
            tool_choice=tool_choice,
        )
