"""
Tool Registry for centralized tool management.

This module provides a centralized registry for all tools used by the AI agent.
It allows tools to be registered with their schemas, handlers, and metadata,
making it easier to add, remove, or modify tools without scattered changes.
"""

from typing import Dict, Callable, Any, Optional, Type
from pydantic import BaseModel
import logging

logger = logging.getLogger(__name__)


class ToolMetadata:
    """Metadata for a registered tool."""

    def __init__(
        self,
        name: str,
        input_schema: Type[BaseModel],
        handler: Callable,
        description: str = "",
        module: Optional[str] = None,  # "education", "age_gender", or None for both
    ):
        self.name = name
        self.input_schema = input_schema
        self.handler = handler
        self.description = description
        self.module = module  # None means available for both modules


class ToolRegistry:
    """
    Centralized registry for all tools.

    Usage:
        registry = ToolRegistry()
        registry.register_tool(...)
        handler = registry.get_handler("tool_name")
        schema = registry.get_input_schema("tool_name")
    """

    def __init__(self):
        self._tools: Dict[str, ToolMetadata] = {}

    def register_tool(
        self,
        name: str,
        input_schema: Type[BaseModel],
        handler: Callable,
        description: str = "",
        module: Optional[str] = None,
    ) -> None:
        """
        Register a tool with the registry.

        Args:
            name: Tool name (must match the function name used in OpenAI API)
            input_schema: Pydantic model for validating tool input
            handler: Callable that handles the tool execution
            description: Optional description of the tool
            module: Optional module restriction ("education", "age_gender", or None for both)
        """

        if name in self._tools:
            logger.warning(f"Tool '{name}' is already registered. Overwriting.")

        self._tools[name] = ToolMetadata(
            name=name,
            input_schema=input_schema,
            handler=handler,
            description=description,
            module=module,
        )
        logger.debug(f"Registered tool: {name}")

    def get_handler(self, name: str) -> Optional[Callable]:
        """Get the handler function for a tool."""
        tool = self._tools.get(name)
        return tool.handler if tool else None

    def get_input_schema(self, name: str) -> Optional[Type[BaseModel]]:
        """Get the input schema class for a tool."""
        tool = self._tools.get(name)
        return tool.input_schema if tool else None

    def is_registered(self, name: str) -> bool:
        """Check if a tool is registered."""
        return name in self._tools

    def get_all_tool_names(self, module: Optional[str] = None) -> list[str]:
        """
        Get all registered tool names, optionally filtered by module.

        Args:
            module: Optional module filter ("education", "age_gender", or None for all)

        Returns:
            List of tool names
        """
        if module is None:
            return list(self._tools.keys())

        return [name for name, tool in self._tools.items() if tool.module is None or tool.module == module]


# Global registry instance
_registry = ToolRegistry()


def get_registry() -> ToolRegistry:
    """Get the global tool registry instance."""
    return _registry


def register_all_tools(agent_instance: Any) -> None:
    """
    Register all tools with the global registry.

    This function should be called during agent initialization to register
    all available tools with their handlers from the agent instance.

    Args:
        agent_instance: Instance of CorrectionAgent with handler methods
    """
    from adoption_forecaster.AI_adjustments.schemas import (
        ListCountriesInput,
        ListIndividualsInput,
        ListAllCombinationsInput,
        AnalyzeCountryTrendInput,
        AdjustCountryProjectionsInput,
    )

    registry = get_registry()

    # Register tools available for both modules
    registry.register_tool(
        name="list_countries",
        input_schema=ListCountriesInput,
        handler=agent_instance._handle_list_countries,
        description="List all available countries in the dataset",
    )

    # Register tools specific to age_gender module
    registry.register_tool(
        name="list_individuals",
        input_schema=ListIndividualsInput,
        handler=agent_instance._handle_list_individuals,
        description="List all distinct gender-age groups in the dataset",
        module="age_gender",
    )

    registry.register_tool(
        name="list_all_combinations",
        input_schema=ListAllCombinationsInput,
        handler=agent_instance._handle_list_all_combinations,
        description="List all country-individual combinations that need to be processed",
        module="age_gender",
    )

    # Register tools available for both modules (with different handlers)
    registry.register_tool(
        name="analyze_country_trend",
        input_schema=AnalyzeCountryTrendInput,
        handler=agent_instance._handle_analyze_trend,
        description="Analyze country trends and identify outliers",
    )

    registry.register_tool(
        name="adjust_country_projections",
        input_schema=AdjustCountryProjectionsInput,
        handler=agent_instance._handle_adjust_projections,
        description="Adjust projection columns for a country",
    )
