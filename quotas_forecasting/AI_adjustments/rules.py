"""
Correction rules for AI agent adjustments.

This module defines the correction logic that guides the AI agent in making
adjustments to projections. These rules are customizable and focus on WHAT
to correct and HOW to correct, not the workflow (which is defined in prompts.py).
"""


def get_education_rules() -> str:
    """
    Get the correction rules for the Education module.

    Returns:
        str: Correction rules for making adjustments to education projections
    """
    return """Apply corrections to ensure projections are realistic and consistent:
- Review each country's projections for outliers (year-over-year changes > 1%)
- Adjust projections that show unrealistic jumps or drops
- Ensure tertiary + secondary always sum to 1.0
- Maintain consistency with historical trends
- Make conservative adjustments when in doubt
- ⚠️ CRITICAL: ONLY adjust SECONDARY columns (e.g., secondary_2025, secondary_2026). Tertiary columns are automatically calculated as: tertiary = 1 - secondary
- ⚠️ CRITICAL WORKFLOW: Adjust years SEQUENTIALLY:
  1. FIRST adjust secondary_2025 (if it has outliers) - compare it to secondary_2024
  2. THEN adjust secondary_2026 - compare it to the ADJUSTED secondary_2025 value (not the original), ensuring realistic progression from the adjusted 2025 value
- This sequential approach ensures that 2026 adjustments account for the corrected 2025 baseline"""


def get_age_gender_rules() -> str:
    """
    Get the correction rules for the Age_Gender module.

    Returns:
        str: Correction rules for making adjustments to age/gender projections
    """
    return """Apply corrections to ensure projections are realistic and consistent:
- Review each country-individual combination's projections for outliers (year-over-year changes > 1%)
- Adjust projections that show unrealistic jumps or drops
- Maintain consistency with historical trends
- Make conservative downward adjustments when in doubt
- Always decrease the bigger projection (never increase smaller one)
- Ensure projections remain within [0, 1] bounds"""


def get_rules(module: str, count: int = None) -> str:
    """
    Get correction rules for a specific module.

    Args:
        module: Module name ("education" or "age_gender")
        count: Deprecated parameter (kept for backward compatibility, not used)

    Returns:
        str: Correction rules for making adjustments

    Raises:
        ValueError: If module is not recognized
    """
    if module == "education":
        return get_education_rules()
    elif module == "age_gender":
        return get_age_gender_rules()
    else:
        raise ValueError(f"Invalid module: {module}. Must be 'education' or 'age_gender'")
