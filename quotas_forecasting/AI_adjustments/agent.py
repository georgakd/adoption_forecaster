"""
AI agent orchestration for processing corrections - Class-based implementation
"""

import json
import logging
import pandas as pd
from typing import List, Dict, Any
from client import AIClient
from tools import list_countries, list_gender_age_groups, analyze_country_trend, adjust_country_projections, list_all_combinations
from schemas import (
    validate_tool_call_input,
    ListCountriesOutput,
    ListIndividualsOutput,
    ListAllCombinationsOutput,
    AnalyzeCountryTrendOutput,
    AnalyzeCountryTrendToolOutput,
    AdjustCountryProjectionsOutput,
)
from utils import create_tools_schema
from prompts import get_education_system_prompt, get_age_gender_system_prompt
from rules import get_rules
from tool_registry import get_registry, register_all_tools
from message_history import MessageHistory
from workflow_enforcer import WorkflowEnforcer

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


cosine_score = 0.75


class CorrectionAgent:
    """
    AI agent that orchestrates projection corrections using tool calling.

    Analyzes projections for outliers (>1% YoY change) and applies adjustments
    based on hard rules and market researcher feedback.
    """

    def __init__(
        self,
        df,
        projection_columns,
        country_col,
        adjusted_year,
        client,
        model,
        instruction,
        individual_col=None,
        module="education",
        feedback_retriever=None,
    ):
        """Initialize agent with DataFrame, projection columns, and AI client."""
        self.df = df
        self.projection_columns = projection_columns
        self.country_col = country_col
        self.individual_col = individual_col
        self.adjusted_year = adjusted_year
        self.client = client
        self.model = model
        self.instruction = instruction
        self.module = module
        self.feedback_retriever = feedback_retriever
        self.df_working = None
        self.total_items = 0
        self.message_history = None
        self.tools = None
        self.max_iterations = 500  # max iterations to prevent infinite loop
        self.adjustment_log = []

    def run(self) -> pd.DataFrame:
        """Main entry point."""
        self._setup()
        self._run_loop()
        return self.df_working, self.adjustment_log

    def _setup(self):
        """Build system prompt, tools, initial messages."""
        self.df_working = self.df.copy()
        # Register all tools with the registry
        register_all_tools(self)
        self.registry = get_registry()
        countries = self.df[self.country_col].unique().tolist()
        countries_str = ", ".join(countries)
        projection_cols_str = ", ".join(self.projection_columns)
        if self.module == "education":
            self.total_items = len(countries)
            system_prompt = get_education_system_prompt(
                adjusted_year=self.adjusted_year,
                countries_str=countries_str,
                projection_cols_str=projection_cols_str,
                num_countries=self.total_items,
            )
            self.tools = create_tools_schema(
                self.projection_columns, self.country_col, individual_col=None, adjusted_year=self.adjusted_year, module="education"
            )
        elif self.module == "age_gender":
            individuals = self.df[self.individual_col].unique().tolist()
            individuals_str = ", ".join(individuals)
            combinations = (self.df[self.country_col] + ", " + self.df[self.individual_col]).unique().tolist()
            combinations_str = ", ".join(combinations)
            self.total_items = len(countries) * len(individuals)
            system_prompt = get_age_gender_system_prompt(
                adjusted_year=self.adjusted_year,
                countries_str=countries_str,
                individuals_str=individuals_str,
                combinations_str=combinations_str,
                projection_cols_str=projection_cols_str,
                num_combinations=self.total_items,
            )
            self.tools = create_tools_schema(
                self.projection_columns, self.country_col, self.individual_col, adjusted_year=self.adjusted_year, module="age_gender"
            )
        else:
            raise ValueError(f"Invalid module: {self.module}")
        if self.instruction:
            rules_text = self.instruction
        else:
            rules_text = get_rules(self.module, self.total_items)
        if self.module == "education":
            user_prompt = f"""Review the predictions for every country and apply corrections based on this instruction: {rules_text}"""
        else:
            user_prompt = (
                f"""Review the predictions for every country-individual combination and apply corrections based on this instruction: {rules_text}"""
            )
        self.message_history = MessageHistory(system_prompt, user_prompt)
        self.workflow = WorkflowEnforcer(module=self.module, total_items=self.total_items)

    def _run_loop(self):
        """
        Main processing loop: iterate until all items processed or max iterations reached.

        Workflow:
        1. Get AI response (may include function calls)
        2. Store function calls in message history (required before outputs)
        3. Process each function call (validate, execute handler, get result)
        4. Store function call outputs in message history
        5. Repeat until all countries/combinations are processed
        """
        for iteration in range(self.max_iterations):
            self.message_history.validate_pairs()

            # trim before  API call to prevent context overflow
            # age_gender has 264 combinations so will need to trim more aggressively
            trim_threshold = 100 if self.module == "age_gender" else 150
            keep_recent_count = 50 if self.module == "age_gender" else 75
            if len(self.message_history) > trim_threshold:
                self.message_history.trim(keep_recent=keep_recent_count)

            response = self.client.create_response(
                model=self.model,
                messages=self.message_history.get_messages(),
                tools=self.tools,
                tool_choice="auto",
            )

            # do a pre-check before processing to prevent violations
            function_calls_in_response = [item for item in response.output if item.type == "function_call"]

            num_calls = len(function_calls_in_response)

            if self.workflow.should_reject_multiple_calls(num_calls):
                logger.warning(self.workflow.get_reject_multiple_calls_log_message(num_calls))
                self.message_history.append_user(self.workflow.get_reject_multiple_calls_message(num_calls))
                continue

            saw_tool_call = False
            processed_calls_this_iteration = 0

            saw_tool_call, _, output_items = self.process_response_output(response, self.message_history)

            workflow_enforcement_added = False

            for item in output_items:
                if item.type == "function_call":
                    function_name = item.name

                    if self.workflow.has_pending() and function_name == "analyze_country_trend":
                        try:
                            function_args = json.loads(item.arguments)
                        except json.JSONDecodeError:
                            function_args = {}
                        attempted_country = function_args.get("country", "unknown")
                        attempted_individual = function_args.get("individual") if self.module == "age_gender" else None
                        error_response = self.workflow.get_blocked_analyze_error_response(attempted_country, attempted_individual)
                        result = json.dumps(error_response, indent=2)
                        if not workflow_enforcement_added:
                            logger.warning(self.workflow.get_blocked_analyze_log_message())
                        self.message_history.append_function_call(item.call_id, function_name, item.arguments)
                        self.message_history.append_function_output(item.call_id, result)
                        if not workflow_enforcement_added:
                            self.message_history.append_user(
                                self.workflow.get_blocked_analyze_enforcement_message(attempted_country, attempted_individual)
                            )
                            workflow_enforcement_added = True
                        break

                    # Check call count limit BEFORE adding to message history
                    if self.workflow.should_stop_processing_calls(processed_calls_this_iteration):
                        logger.warning(self.workflow.get_limit_reached_log_message())
                        self.message_history.append_user(self.workflow.get_limit_reached_message())
                        break

                    # Now safe to add to message history (workflow check passed)
                    self.message_history.append_function_call(item.call_id, function_name, item.arguments)
                    saw_tool_call = True
                    processed_calls_this_iteration += 1

                    # Parse arguments (only if workflow check passed)
                    try:
                        function_args = json.loads(item.arguments)
                    except json.JSONDecodeError as e:
                        result = f"Error: Invalid JSON in arguments for {function_name}: {str(e)}"
                        logger.warning(f"  [WARNING] {result}")
                        self.message_history.append_function_output(item.call_id, result)
                        continue

                    logger.info("\n" + "=" * 60)
                    logger.info(f"function_name: {function_name}")
                    logger.info(f"function_args: {function_args}")
                    logger.info("=" * 60)

                    # Validate tool call input
                    is_valid, error_message, validated_args = validate_tool_call_input(function_name, function_args)
                    if not is_valid:
                        result = f"Error: Validation failed for {function_name}: {error_message}"
                        logger.warning(f"  [WARNING] {result}")
                        self.message_history.append_function_output(item.call_id, result)
                        continue

                    # Get handler
                    handler = self.registry.get_handler(function_name)
                    if handler is None:
                        result = f"Error: Unknown function: {function_name}"
                        logger.warning(f"  [WARNING] {result}")
                        self.message_history.append_function_output(item.call_id, result)
                    else:
                        # Execute handler (workflow check already passed above)
                        if function_name == "analyze_country_trend":
                            # No need to check workflow violation again - already checked above
                            result, workflow_violation_detected = handler(item, validated_args)
                        else:
                            result = handler(item, validated_args)
                            workflow_violation_detected = False
                        self.message_history.append_function_output(item.call_id, result)
                        if workflow_violation_detected:
                            break

            # Check completion
            if len(self.workflow.processed_items) == self.total_items:
                if response.output_text:
                    logger.info(f"\n[OK] {response.output_text}")
                logger.info(f"[COMPLETE] All {self.total_items} items have been processed.")
                break
            if response.output_text and not saw_tool_call:
                remaining = self.workflow.remaining()
                if self.module == "education":
                    logger.warning(
                        f"\n[WARNING] Model provided text output but only {len(self.workflow.processed_items)}/{self.total_items} countries processed. {remaining} remaining."
                    )
                    self.message_history.append_user(
                        f"You cannot finish yet. You have only processed {len(self.workflow.processed_items)}/{self.total_items} countries. {remaining} countries still need to be analyzed. You MUST continue calling analyze_country_trend for each remaining country. Do NOT provide a final summary until ALL {self.total_items} countries have been analyzed."
                    )
                else:
                    logger.warning(
                        f"\n[WARNING] Model provided text output but only {len(self.workflow.processed_items)}/{self.total_items} combinations processed. {remaining} remaining."
                    )
                    self.message_history.append_user(
                        f"You cannot finish yet. You have only processed {len(self.workflow.processed_items)}/{self.total_items} combinations. {remaining} combinations still need to be analyzed. You MUST continue calling analyze_country_trend for each remaining country-individual combination. Do NOT provide a final summary until ALL {self.total_items} combinations have been analyzed."
                    )

    # --- Tool handlers ---
    def _handle_list_countries(self, item, args):
        """Handle list_countries tool call."""
        result_dict = list_countries(self.df_working, self.country_col)
        validated = ListCountriesOutput(**result_dict)
        logger.info(f"  [INFO] Found {validated.count} countries")
        result_dict = validated.model_dump()
        return json.dumps(result_dict, indent=2)

    def _handle_list_individuals(self, item, args):
        """Handle list_individuals tool call."""
        result_dict = list_gender_age_groups(self.df_working, self.individual_col)
        validated = ListIndividualsOutput(**result_dict)
        logger.info(f"  [INFO] Found {validated.count} individuals")
        result_dict = validated.model_dump()
        return json.dumps(result_dict, indent=2)

    def _handle_list_all_combinations(self, item, args):
        """Handle list_all_combinations tool call."""
        result_dict = list_all_combinations(self.df_working, self.country_col, self.individual_col)
        validated = ListAllCombinationsOutput(**result_dict)
        logger.info(f"  [INFO] Total combinations: {validated.total_combinations}")
        result_dict = validated.model_dump()
        return json.dumps(result_dict, indent=2)

    def _handle_analyze_trend(self, item, args):
        """Handle analyze_country_trend tool call.

        Returns:
            tuple: (result_string, workflow_violation_detected)
        """
        item_key = self.workflow.item_key(args.country, getattr(args, "individual", None))
        individual = getattr(args, "individual", None)
        if not self.workflow.check_can_analyze(args.country, individual):
            response_json, user_msg = self.workflow.get_violation_response(args.country, individual)
            self.message_history.append_user(user_msg)
            return response_json, True
        self.workflow.record_analyzed(item_key)
        if self.module == "education":
            result_dict = analyze_country_trend(
                self.df_working,
                args.country,
                self.country_col,
                args.individual,
                None,
                self.projection_columns,
                module="education",
                feedback_retriever=self.feedback_retriever,
            )
        else:
            result_dict = analyze_country_trend(
                self.df_working,
                args.country,
                self.country_col,
                args.individual,
                self.individual_col,
                self.projection_columns,
                module="age_gender",
                feedback_retriever=self.feedback_retriever,
            )
        tool_output = AnalyzeCountryTrendToolOutput(**result_dict)
        outliers = []
        critical_action = None
        if tool_output.success:
            outliers = tool_output.outliers
            if outliers:
                logger.info(f"  [ANALYSIS] Found {len(outliers)} outlier(s): {', '.join(outliers)}")
                self.workflow.record_outliers(item_key, args.country, individual, outliers)
                critical_action = self.workflow.get_critical_action_message(args.country, individual)
            else:
                logger.info(f"  [ANALYSIS] No outliers found for {self.workflow.format_entity_name(args.country, individual)}")
        else:
            logger.warning(f"  [WARNING] Analysis failed: {tool_output.error}")
        orchestration = self.workflow.get_orchestration_metadata(args.country, individual, critical_action)
        final_response = AnalyzeCountryTrendOutput.compose(tool_output, orchestration)
        if final_response.progress:
            logger.debug(final_response.progress["message"])
        return json.dumps(final_response.model_dump(), indent=2), False

    def _handle_adjust_projections(self, item, args):
        """Handle adjust_country_projections tool call."""
        item_key = self.workflow.item_key(args.country, getattr(args, "individual", None))

        if self.module == "education":
            result_dict, self.df_working = adjust_country_projections(
                self.df_working,
                args.country,
                self.country_col,
                None,
                None,
                self.projection_columns,
                args.adjustments,
                module="education",
                reason=args.reason,
            )
        else:
            result_dict, self.df_working = adjust_country_projections(
                self.df_working,
                args.country,
                self.country_col,
                args.individual,
                self.individual_col,
                self.projection_columns,
                args.adjustments,
                module="age_gender",
                reason=args.reason,
            )
        validated = AdjustCountryProjectionsOutput(**result_dict)
        result_dict = validated.model_dump()
        if validated.success:
            adjustments_count = len(validated.adjustments_made)
            logger.info(
                f"  [ADJUSTED] Made {adjustments_count} adjustment(s) for {self.workflow.format_entity_name(args.country, getattr(args, 'individual', None))}"
            )

            for col, adj_detail in validated.adjustments_made.items():
                log_entry = {"country": args.country, "column": col}
                if self.module == "age_gender":
                    log_entry["individual"] = args.individual

                log_entry.update(
                    {
                        "old_value": adj_detail.old_value,
                        "new_value": adj_detail.new_value,
                        "reason": adj_detail.reason,
                        "auto_adjusted": adj_detail.auto_adjusted,
                    }
                )

                self.adjustment_log.append(log_entry)

            if item_key in self.workflow.pending_adjustments:
                self.workflow.record_adjustment(item_key)
                logger.debug(f"  [WORKFLOW] Cleared pending_adjustments for {item_key}. Remaining: {list(self.workflow.pending_adjustments.keys())}")
            else:
                logger.debug(
                    f"  [WORKFLOW WARNING] item_key {item_key} succeeded but not found in pending_adjustments. "
                    f" This is expected for sequential adjustments (e.g. adjusting 2025 and then 2026)."
                    f"Current pending: {list(self.workflow.pending_adjustments.keys())}. "
                )
        else:
            logger.warning(f"  [WARNING] Adjustment failed: {validated.error}")
            logger.warning(f"  [WORKFLOW] Keeping {item_key} in pending_adjustments due to adjustment failure")
        return json.dumps(result_dict, indent=2)

    # --- Helpers ---

    def process_response_output(self, response, messages: List[Dict[str, Any]]) -> tuple[bool, int, List]:
        """
        Process the output items from a response. Only appends non-function-call items to messages.
        Function call items are stored separately and will be added to messages during processing.

        Note: OpenAI responses API format:
        - Text outputs: {"role": "assistant", "content": "..."}
        - Function calls: {"type": "function_call", "call_id": "...", "name": "...", "arguments": "..."}
        - Function outputs: {"type": "function_call_output", "call_id": "...", "output": "..."}

        Function calls must be stored BEFORE their outputs in the message history.

        Args:
            response: Response object from OpenAI API
            messages: List of messages to append response items to (only non-function-call items)

        Returns:
            Tuple of (saw_tool_call, processed_calls_count, output_items)
            - saw_tool_call: True if any function calls were made
            - processed_calls_count: Number of function calls found
            - output_items: List of all output items for further processing
        """
        saw_tool_call = False
        processed_calls_count = 0
        output_items = []

        for item in response.output:
            # Only add non-function-call items to messages immediately
            # Function calls will be added when we process them
            if item.type != "function_call":
                content = getattr(item, "content", None) or getattr(item, "text", None) or getattr(item, "message", None) or str(item)
                if content is None:
                    content = ""
                self.message_history.append_assistant(content)
            output_items.append(item)
            if item.type == "function_call":
                saw_tool_call = True
                processed_calls_count += 1
        return saw_tool_call, processed_calls_count, output_items


# --- Backward compatibility wrapper ---
def apply_corrections(
    df: pd.DataFrame,
    projection_columns: list,
    country_col: str,
    individual_col: str,
    adjusted_year: int,
    client: AIClient,
    module,
    model: str = "gpt-4o",
    instruction: str = None,
    feedback_retriever=None,
) -> pd.DataFrame:
    """
    Use AI agent with tool calling to modify the DataFrame.

    This is a wrapper function that maintains backward compatibility with the old function-based API.
    It creates a CorrectionAgent instance and runs it.

    Args:
        df: DataFrame to modify
        projection_columns: List of projection column names
        country_col: Name of country column
        individual_col: Name of individual column (None for education module)
        adjusted_year: Target year for adjustments
        client: AIClient instance
        module: Module name ("education" or "age_gender")
        model: Model name (default: "gpt-4o")
        instruction: Optional custom instruction. If None, uses rules from rules.py

    Returns:
        Modified DataFrame
    """
    agent = CorrectionAgent(
        df=df,
        projection_columns=projection_columns,
        country_col=country_col,
        adjusted_year=adjusted_year,
        client=client,
        model=model,
        instruction=instruction,
        individual_col=individual_col,
        module=module,
        feedback_retriever=feedback_retriever,
    )
    return agent.run()
