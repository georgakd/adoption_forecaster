"""
Workflow enforcement for the correction agent.

Centralizes "adjust before analyzing next" logic: pending adjustments,
processed items, and all nudge/violation messages in one place.
"""

import json
import logging
from typing import Any, Optional

from quotas_forecasting.AI_adjustments.schemas import (
    AgentOrchestrationMetadata,
    AnalyzeCountryTrendOutput,
    AnalyzeCountryTrendToolOutput,
)

logger = logging.getLogger(__name__)


class WorkflowEnforcer:
    """Owns pending_adjustments and processed_items; exposes workflow checks and message builders."""

    def __init__(self, module: str, total_items: int):
        self.module = module
        self.total_items = total_items
        self.processed_items: set = set()
        self.pending_adjustments: dict = {}

    @property
    def max_calls_per_iteration(self) -> Optional[int]:
        """When there are pending adjustments, allow only one call per response."""
        return 1 if self.pending_adjustments else None

    def has_pending(self) -> bool:
        return len(self.pending_adjustments) > 0

    def remaining(self) -> int:
        return self.total_items - len(self.processed_items)

    # --- Item keys and names ---

    def item_key(self, country: str, individual: Optional[str] = None) -> Any:
        if self.module == "education":
            return country
        return (country, individual)

    def format_entity_name(self, country: str, individual: Optional[str] = None) -> str:
        if self.module == "education":
            return country
        return f"{country} - {individual}"

    def _first_pending_info(self) -> Optional[dict]:
        if not self.pending_adjustments:
            return None
        return next(iter(self.pending_adjustments.values()))

    def _first_pending_entity_name(self) -> Optional[str]:
        info = self._first_pending_info()
        if not info:
            return None
        if self.module == "education":
            return info["country"]
        return f"{info['country']} - {info['individual']}"

    def _adjust_instruction(self, country: str, individual: Optional[str] = None) -> str:
        if self.module == "education":
            return f"Call adjust_country_projections(country='{country}', adjustments={{...}}, reason='...') NOW."
        return f"Call adjust_country_projections(country='{country}', individual='{individual}', adjustments={{...}}, reason='...') NOW."

    # --- Recording ---

    def record_analyzed(self, item_key: Any) -> None:
        self.processed_items.add(item_key)

    def record_outliers(self, item_key: Any, country: str, individual: Optional[str], outliers: list) -> None:
        self.pending_adjustments[item_key] = {"country": country, "outliers": outliers}
        if individual is not None:
            self.pending_adjustments[item_key]["individual"] = individual

    def record_adjustment(self, item_key: Any) -> None:
        if item_key in self.pending_adjustments:
            del self.pending_adjustments[item_key]

    # --- Checks ---

    def check_can_analyze(self, country: str, individual: Optional[str] = None) -> bool:
        """True if the model is allowed to analyze this item (no pending adjustments blocking it)."""
        return not self.pending_adjustments

    def should_reject_multiple_calls(self, num_calls: int) -> bool:
        return num_calls > 1 and self.has_pending()

    # --- Violation: tried to analyze while pending exist ---

    def get_violation_response(self, country: str, individual: Optional[str] = None) -> tuple[str, str]:
        """
        When analyze_country_trend was called but there are pending adjustments.
        Returns (tool_response_json, user_message_to_append).
        """
        pending_list = self._pending_list()
        entity = self.format_entity_name(country, individual)
        violation_msg = (
            f"  WORKFLOW VIOLATION: You attempted to analyze {entity} while "
            f"{len(self.pending_adjustments)} previous "
            + ("country/countries" if self.module == "education" else "combination(s)")
            + f" have pending adjustments: {', '.join(pending_list)}. "
            + "You MUST call adjust_country_projections for these "
            + ("countries" if self.module == "education" else "combinations")
            + " BEFORE analyzing a new one."
        )
        logger.warning(f"  [WORKFLOW WARNING] {violation_msg}")

        first = self._first_pending_info()
        next_step = self._adjust_instruction(first["country"], first.get("individual"))
        next_step += (
            " Do NOT call analyze_country_trend for any "
            + ("country" if self.module == "education" else "combination")
            + " until all pending adjustments are complete."
        )

        tool_output = AnalyzeCountryTrendToolOutput(
            success=False,
            country=country,
            individual=individual,
            analyses={},
            outliers=[],
            error="Pending adjustments must be completed before analyzing a new " + ("country." if self.module == "education" else "combination."),
        )
        orchestration = AgentOrchestrationMetadata(workflow_warning=violation_msg, next_step=next_step)
        final_response = AnalyzeCountryTrendOutput.compose(tool_output, orchestration)
        response_json = json.dumps(final_response.model_dump(), indent=2)

        user_msg = (
            f"  CRITICAL WORKFLOW VIOLATION DETECTED: You attempted to analyze {entity} while "
            f"{len(self.pending_adjustments)} "
            + ("country/countries" if self.module == "education" else "combination(s)")
            + " have pending adjustments. You MUST call adjust_country_projections for the pending "
            + ("country/countries" if self.module == "education" else "combinations")
            + " FIRST. Do NOT make multiple function calls in parallel. Process ONE "
            + ("country" if self.module == "education" else "combination")
            + " at a time: analyze → adjust (if needed) → then next."
        )
        return response_json, user_msg

    def _pending_list(self) -> list:
        if self.module == "education":
            return [info["country"] for info in self.pending_adjustments.values()]
        return [f"{info['country']} - {info['individual']}" for info in self.pending_adjustments.values()]

    # --- Blocked analyze (in loop: model tried to analyze while pending) ---

    def get_blocked_analyze_error_response(self, attempted_country: str, attempted_individual: Optional[str]) -> dict:
        """Error response dict for the blocked analyze_country_trend call."""
        entity_name = self._first_pending_entity_name()
        return {
            "success": False,
            "country": attempted_country,
            "individual": attempted_individual,
            "analyses": {},
            "outliers": [],
            "error": f"Cannot analyze new country while {entity_name} has pending adjustments. "
            f"You MUST call adjust_country_projections for {entity_name} first.",
        }

    def get_blocked_analyze_enforcement_message(self, attempted_country: str, attempted_individual: Optional[str]) -> str:
        """User message to append when we blocked an analyze call."""
        first = self._first_pending_info()
        entity_name = self._first_pending_entity_name()
        if self.module == "education":
            return (
                f"  WORKFLOW ENFORCEMENT: You attempted to analyze {attempted_country} while {entity_name} has pending adjustments. "
                f"You MUST call adjust_country_projections(country='{first['country']}', adjustments={{...}}, reason='...') NOW. "
                f"Do NOT call analyze_country_trend for any country until all pending adjustments are complete."
            )
        return (
            f"  WORKFLOW ENFORCEMENT: You attempted to analyze {attempted_country} - {attempted_individual} while {entity_name} has pending adjustments. "
            f"You MUST call adjust_country_projections(country='{first['country']}', individual='{first['individual']}', adjustments={{...}}, reason='...') NOW. "
            f"Do NOT call analyze_country_trend until all pending adjustments are complete."
        )

    def get_blocked_analyze_log_message(self) -> str:
        entity_name = self._first_pending_entity_name()
        return (
            f"  [WORKFLOW] BLOCKED analyze_country_trend: Cannot analyze while {len(self.pending_adjustments)} pending adjustment(s) exist. "
            f"Must adjust {entity_name} first."
        )

    # --- Multiple calls in one response ---

    def get_reject_multiple_calls_message(self, num_calls: int) -> str:
        """User message when model made multiple tool calls while pending adjustments exist."""
        entity_name = self._first_pending_entity_name()
        first = self._first_pending_info()
        instruction = self._adjust_instruction(first["country"], first.get("individual"))
        return (
            f"  WORKFLOW VIOLATION: You made {num_calls} function calls in one response while {entity_name} has pending adjustments. "
            f"You MUST make ONLY ONE function call per API response. {instruction} "
            f"Do NOT make multiple function calls in parallel."
        )

    def get_reject_multiple_calls_log_message(self, num_calls: int) -> str:
        entity_name = self._first_pending_entity_name()
        return (
            f"  [WORKFLOW] Rejected response with {num_calls} function calls while {entity_name} has pending adjustments. "
            f"Only ONE function call allowed per response when adjustments are pending."
        )

    # --- Limit reached (one call per iteration when pending) ---

    def get_limit_reached_message(self) -> str:
        entity_name = "country/countries" if self.module == "education" else "combination(s)"
        return (
            f"  WORKFLOW ENFORCEMENT: You have {len(self.pending_adjustments)} pending adjustment(s). "
            f"You MUST call adjust_country_projections for the pending {entity_name} NOW. "
            f"Do NOT make multiple function calls. Make ONLY ONE call: adjust_country_projections for the pending {entity_name}."
        )

    def get_limit_reached_log_message(self) -> str:
        return (
            f"  [WORKFLOW] Stopping function call processing: {len(self.pending_adjustments)} "
            f"pending adjustment(s) require immediate attention. Only one function call should be "
            f"made per iteration when adjustments are pending."
        )

    def should_stop_processing_calls(self, processed_this_iteration: int) -> bool:
        max_calls = self.max_calls_per_iteration
        return max_calls is not None and processed_this_iteration >= max_calls

    # --- Orchestration metadata (progress, reminder, critical action) ---

    def get_critical_action_message(self, country: str, individual: Optional[str] = None) -> str:
        if self.module == "education":
            return (
                f"  CRITICAL: Outliers found for {country}. You MUST call adjust_country_projections NOW "
                f"before moving to the next country. DO NOT proceed to analyze another country until adjustments are made."
            )
        return (
            f"  CRITICAL: Outliers found for {country} - {individual}. You MUST call adjust_country_projections NOW "
            f"before moving to the next combination. DO NOT proceed to analyze another combination until adjustments are made."
        )

    def get_orchestration_metadata(
        self, country: str, individual: Optional[str], critical_action: Optional[str] = None
    ) -> AgentOrchestrationMetadata:
        """Build progress/reminder/action_required for the analyze response."""
        item_key = self.item_key(country, individual)
        item_name = self.format_entity_name(country, individual)
        remaining = self.remaining()

        if item_key in self.pending_adjustments:
            action_required = "adjust_before_next"
            if self.module == "education":
                progress_msg = (
                    f"PROGRESS: {len(self.processed_items)}/{self.total_items} countries analyzed. "
                    f"  CRITICAL: You MUST call adjust_country_projections for {country} NOW before analyzing the next country."
                )
            else:
                progress_msg = (
                    f"PROGRESS: {len(self.processed_items)}/{self.total_items} combinations analyzed. "
                    f"  CRITICAL: You MUST call adjust_country_projections for {item_name} NOW before the next combination."
                )
        else:
            action_required = "continue_to_next"
            if self.module == "education":
                progress_msg = (
                    f"PROGRESS: {len(self.processed_items)}/{self.total_items} countries processed. {remaining} remaining. "
                    f"You can proceed to analyze the next country."
                )
            else:
                progress_msg = (
                    f"PROGRESS: {len(self.processed_items)}/{self.total_items} combinations processed. {remaining} remaining. "
                    f"You can proceed to analyze the next combination."
                )

        reminder = None
        if remaining > 0:
            if item_key in self.pending_adjustments:
                if self.module == "education":
                    reminder = (
                        f"  CRITICAL: You found outliers for {country}. You MUST call adjust_country_projections NOW. "
                        f"DO NOT analyze the next country until adjustments are complete."
                    )
                else:
                    reminder = (
                        f"  CRITICAL: You found outliers for {item_name}. You MUST call adjust_country_projections NOW. "
                        f"DO NOT analyze the next combination until adjustments are complete."
                    )
            else:
                if self.module == "education":
                    reminder = f"  IMPORTANT: {remaining} countries still need analysis. Continue with analyze_country_trend for the next country."
                else:
                    reminder = f"  IMPORTANT: {remaining} combinations still need analysis. Continue with analyze_country_trend for the next country-individual pair."

        return AgentOrchestrationMetadata(
            critical_action=critical_action,
            action_required=action_required,
            progress={
                "processed": len(self.processed_items),
                "total": self.total_items,
                "remaining": remaining,
                "message": progress_msg,
                "has_pending_adjustments": item_key in self.pending_adjustments,
            },
            reminder=reminder,
        )
