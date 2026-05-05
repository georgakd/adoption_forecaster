"""
Schema models for structured input/output and validation.

This module defines Pydantic models for:
- Tool input schemas (what the AI sends to tools)
- Tool output schemas (what tools return)
- Agent-level input/output schemas

This schemas file supports both Education and Age_Gender modules.
"""

from pydantic import BaseModel, Field, field_validator
from typing import Dict, List, Optional, Union


# ============================================================================
# TOOL INPUT SCHEMAS (for validating tool call arguments)
# ============================================================================


class ListCountriesInput(BaseModel):
    """Input schema for list_countries tool (no parameters)."""

    pass  # This tool takes no parameters


class ListIndividualsInput(BaseModel):
    """Input schema for list_gender_age_groups tool (no parameters)."""

    pass  # This tool takes no parameters


class ListAllCombinationsInput(BaseModel):
    """Input schema for list_all_combinations tool (no parameters)."""

    pass  # This tool takes no parameters


class AnalyzeCountryTrendInput(BaseModel):
    """Input schema for analyze_country_trend tool (supports both Education and Age_Gender modules)."""

    country: str = Field(..., description="The country name to analyze")
    individual: Optional[str] = Field(
        default=None, description="The individual (gender-age group) name to analyze. Required for Age_Gender module, not used for Education module."
    )

    @field_validator("country")
    @classmethod
    def country_not_empty(cls, v: str) -> str:
        """Checks if country name is empty. Raises ValueError if it is"""
        if not v or not v.strip():
            raise ValueError("Country name cannot be empty")
        return v.strip()

    @field_validator("individual")
    @classmethod
    def individual_not_empty(cls, v: Optional[str]) -> Optional[str]:
        """Checks if Individual name is empty. Raises ValueError if it is"""
        if v is not None and (not v or not v.strip()):
            raise ValueError("Individual name cannot be empty if provided")
        return v.strip() if v else None


class AdjustCountryProjectionsInput(BaseModel):
    """Input schema for adjust_country_projections tool (supports both Education and Age_Gender modules)."""

    country: str = Field(..., description="The country name to adjust")
    individual: Optional[str] = Field(
        default=None, description="The individual (gender-age group) name to adjust. Required for Age_Gender module, not used for Education module."
    )
    adjustments: Dict[str, float] = Field(
        ...,
        description="Dictionary mapping projection column names to new values. For Education module: ONLY adjust SECONDARY columns (e.g., secondary_2025, secondary_2026). Tertiary columns are automatically calculated. Format: {'secondary_2025': 0.55, 'secondary_2026': 0.56}",
    )
    reason: str = Field(default="", description="Reason for the adjustments")

    @field_validator("country")
    @classmethod
    def country_not_empty(cls, v: str) -> str:
        """Checks if country column is empty. Raises ValueError if it is"""
        if not v or not v.strip():
            raise ValueError("Country name cannot be empty")
        return v.strip()

    @field_validator("individual")
    @classmethod
    def individual_not_empty(cls, v: Optional[str]) -> Optional[str]:
        """Checks if Individual name column is empty. Raises ValueError if it is"""
        if v is not None and (not v or not v.strip()):
            raise ValueError("Individual name cannot be empty if provided")
        return v.strip() if v else None

    @field_validator("adjustments")
    @classmethod
    def validate_adjustments(cls, v: Dict[str, float]) -> Dict[str, float]:
        """Validates adjustments types"""
        if not v:
            raise ValueError("Adjustments dictionary cannot be empty")
        for col, value in v.items():
            if not isinstance(col, str):
                raise ValueError(f"Column name must be a string, got {type(col)}")
            if not isinstance(value, (int, float)):
                raise ValueError(f"Adjustment value must be a number, got {type(value)}")
            if not (0 <= value <= 1):
                raise ValueError(f"Adjustment value must be between 0 and 1, got {value}")
        return v


# ============================================================================
# TOOL OUTPUT SCHEMAS (for structured tool results)
# ============================================================================


class ListCountriesOutput(BaseModel):
    """Output schema for list_countries tool."""

    success: bool = Field(..., description="Whether the operation succeeded")
    countries: List[str] = Field(..., description="List of available country names")
    count: int = Field(..., description="Total number of countries")
    error: Optional[str] = Field(default=None, description="Error message if failed")


class ListIndividualsOutput(BaseModel):
    """Output schema for list_gender_age_groups tool."""

    success: bool = Field(..., description="Whether the operation succeeded")
    individuals: List[str] = Field(..., description="List of available individual (gender-age group) names")
    count: int = Field(..., description="Total number of individuals")
    error: Optional[str] = Field(default=None, description="Error message if failed")


class CombinationInfo(BaseModel):
    """Information about a single country-individual combination."""

    number: int = Field(..., description="Combination number")
    country: str = Field(..., description="Country name")
    individual: str = Field(..., description="Individual (gender-age group) name")


class ListAllCombinationsOutput(BaseModel):
    """Output schema for list_all_combinations tool."""

    success: bool = Field(..., description="Whether the operation succeeded")
    total_combinations: int = Field(..., description="Total number of combinations")
    countries_count: int = Field(..., description="Number of countries")
    individuals_count: int = Field(..., description="Number of individuals")
    combinations: List[CombinationInfo] = Field(..., description="List of all combinations")
    error: Optional[str] = Field(default=None, description="Error message if failed")


class AnalysisDetail(BaseModel):
    """Detailed analysis for a single projection column."""

    column: str = Field(..., description="Projection column name")
    type: str = Field(..., description="Column type (e.g., extracted from column name)")
    previous_year_column: str = Field(..., description="Previous year column name")
    previous_year_value: float = Field(..., description="Previous year value")
    current_value: float = Field(..., description="Current projection value")
    change: float = Field(..., description="Absolute change (current - previous)")
    change_pct: float = Field(..., description="Percentage change")
    is_outlier: bool = Field(..., description="Whether this exceeds the 1% threshold")
    threshold: float = Field(default=1.0, description="Threshold percentage")
    error: Optional[str] = Field(default=None, description="Error message if analysis failed")
    feedback: Optional[str] = Field(default=None, description="Global researcher feedback text")


# ============================================================================
# PURE TOOL OUTPUT SCHEMAS (what tools actually return)
# ============================================================================


class AnalyzeCountryTrendToolOutput(BaseModel):
    """Pure tool output - what analyze_country_trend() actually returns.

    This schema represents ONLY what the tool function returns, without any
    agent-level orchestration metadata. Use this to validate tool output.
    """

    success: bool = Field(..., description="Whether the operation succeeded")
    country: str = Field(..., description="Country that was analyzed")
    individual: Optional[str] = Field(default=None, description="Individual (gender-age group) that was analyzed")
    analyses: Dict[str, AnalysisDetail] = Field(default_factory=dict, description="Analysis details for each projection column")
    outliers: List[str] = Field(default_factory=list, description="List of column names that are outliers")
    error: Optional[str] = Field(default=None, description="Error message if failed")


# ============================================================================
# AGENT ORCHESTRATION METADATA (agent-level concerns)
# ============================================================================


class AgentOrchestrationMetadata(BaseModel):
    """Agent-level orchestration metadata.

    This schema represents metadata added by the agent for workflow enforcement,
    progress tracking, and guidance. These fields are NOT returned by tools.
    """

    workflow_warning: Optional[str] = Field(default=None, description="[AGENT] Warning about workflow violations")
    next_step: Optional[str] = Field(default=None, description="[AGENT] Explicit instruction for the next step")
    critical_action: Optional[str] = Field(default=None, description="[AGENT] Critical action that must be taken immediately")
    action_required: Optional[str] = Field(default=None, description="[AGENT] Type of action required: 'adjust_before_next' or 'continue_to_next'")
    progress: Optional[Dict[str, Union[int, str, bool]]] = Field(
        default=None, description="[AGENT] Progress information about processed countries/combinations"
    )
    reminder: Optional[str] = Field(default=None, description="[AGENT] Reminder message for the agent")


# ============================================================================
# COMBINED OUTPUT SCHEMAS (for agent consumption - flat structure)
# ============================================================================


class AnalyzeCountryTrendOutput(BaseModel):
    """Combined output: tool result + orchestration metadata (flat structure for agent consumption).

    This schema combines the pure tool output with agent orchestration metadata
    into a single flat structure that the LLM can easily consume. The tool fields
    come from AnalyzeCountryTrendToolOutput, and orchestration fields come from
    AgentOrchestrationMetadata.

    Use AnalyzeCountryTrendOutput.compose() to create instances from separate
    tool output and orchestration metadata.
    """

    # Tool fields (from AnalyzeCountryTrendToolOutput)
    success: bool = Field(..., description="Whether the operation succeeded")
    country: str = Field(..., description="Country that was analyzed")
    individual: Optional[str] = Field(default=None, description="Individual (gender-age group) that was analyzed")
    analyses: Dict[str, AnalysisDetail] = Field(default_factory=dict, description="Analysis details for each projection column")
    outliers: List[str] = Field(default_factory=list, description="List of column names that are outliers")
    error: Optional[str] = Field(default=None, description="Error message if failed")

    # Orchestration fields (from AgentOrchestrationMetadata)
    workflow_warning: Optional[str] = Field(default=None, description="[AGENT] Warning about workflow violations")
    next_step: Optional[str] = Field(default=None, description="[AGENT] Explicit instruction for the next step")
    critical_action: Optional[str] = Field(default=None, description="[AGENT] Critical action that must be taken immediately")
    action_required: Optional[str] = Field(default=None, description="[AGENT] Type of action required: 'adjust_before_next' or 'continue_to_next'")
    progress: Optional[Dict[str, Union[int, str, bool]]] = Field(
        default=None, description="[AGENT] Progress information about processed countries/combinations"
    )
    reminder: Optional[str] = Field(default=None, description="[AGENT] Reminder message for the agent")

    @classmethod
    def compose(cls, tool_output: AnalyzeCountryTrendToolOutput, orchestration: AgentOrchestrationMetadata) -> "AnalyzeCountryTrendOutput":
        """Compose tool output with orchestration metadata into a flat structure.

        Args:
            tool_output: Pure tool output from analyze_country_trend()
            orchestration: Agent-level orchestration metadata

        Returns:
            Combined output with flat structure for LLM consumption
        """
        return cls(**tool_output.model_dump(), **orchestration.model_dump(exclude_none=True))


class AdjustmentDetail(BaseModel):
    """Detail of a single adjustment made."""

    column: str = Field(..., description="Column name that was adjusted")
    old_value: float = Field(..., ge=0, le=1, description="Original value")
    new_value: float = Field(..., ge=0, le=1, description="New value")
    reason: str = Field(default="", description="Reason for adjustment")
    auto_adjusted: bool = Field(
        default=False, description="Whether this was auto-adjusted to maintain constraint (e.g., tertiary + secondary = 1 for Education)"
    )


class AdjustCountryProjectionsOutput(BaseModel):
    """Output schema for adjust_country_projections tool (supports both Education and Age_Gender modules)."""

    success: bool = Field(..., description="Whether the operation succeeded")
    country: str = Field(..., description="Country that was adjusted")
    individual: Optional[str] = Field(default=None, description="Individual (gender-age group) that was adjusted")
    adjustments_made: Dict[str, AdjustmentDetail] = Field(default_factory=dict, description="Dictionary of adjustments made, keyed by column name")
    error: Optional[str] = Field(default=None, description="Error message if failed")


# ============================================================================
# AGENT-LEVEL SCHEMAS
# ============================================================================


class ToolCallSummary(BaseModel):
    """Summary of a tool call."""

    tool_name: str = Field(..., description="Name of the tool called")
    success: bool = Field(..., description="Whether the call succeeded")
    iteration: int = Field(..., description="Iteration number when called")


# ============================================================================
# VALIDATION HELPERS
# ============================================================================


def validate_tool_call_input(tool_name: str, args: dict) -> tuple[bool, Optional[str], Optional[BaseModel]]:
    """
    Validate tool call arguments against the appropriate input schema.

    Uses the tool registry to get the appropriate schema for validation.

    Args:
        tool_name: Name of the tool being called
        args: Dictionary of arguments from the AI tool call

    Returns:
        Tuple of (is_valid, error_message, validated_schema_instance)
        - is_valid: True if validation passed, False otherwise
        - error_message: Error message if validation failed, None if successful
        - validated_schema_instance: Pydantic model instance with validated data, None if validation failed

    Example:
        >>> is_valid, error, validated = validate_tool_call_input(
        ...     "analyze_country_trend",
        ...     {"country": "Austria"}
        ... )
        >>> if is_valid:
        ...     print(f"Validated country: {validated.country}")
    """
    # Use tool registry for validation
    from tool_registry import get_registry

    registry = get_registry()

    # Fallback to direct schema mapping if registry is not initialized
    # (for backward compatibility or when registry hasn't been set up yet)
    if not registry.is_registered(tool_name):
        # Fallback schema map for backward compatibility
        schema_map = {
            "list_countries": ListCountriesInput,
            "list_gender_age_groups": ListIndividualsInput,
            "list_individuals": ListIndividualsInput,
            "list_all_combinations": ListAllCombinationsInput,
            "analyze_country_trend": AnalyzeCountryTrendInput,
            "adjust_country_projections": AdjustCountryProjectionsInput,
        }

        if tool_name not in schema_map:
            return False, f"Unknown tool: {tool_name}", None

        schema_class = schema_map[tool_name]
    else:
        schema_class = registry.get_input_schema(tool_name)
        if schema_class is None:
            return False, f"No input schema found for tool: {tool_name}", None

    try:
        validated = schema_class(**args)
        return True, None, validated
    except Exception as e:
        return False, str(e), None
