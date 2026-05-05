"""
Message history management for OpenAI conversations.

Handles message formatting, trimming, and validation for OpenAI API interactions,
with special handling for function_call/function_call_output pairs.
"""

from quotas_forecasting.config.logging_config import get_logger


logger = get_logger(__name__)


def _collect_call_ids_with_outputs(messages: list) -> set:
    """Collect call_ids that have a function_call_output in the history."""
    out = set()
    for msg in messages:
        if msg.get("type") == "function_call_output":
            cid = msg.get("call_id")
            if cid:
                out.add(cid)
    return out


def _find_pair_positions(messages: list, call_ids_with_outputs: set) -> list:
    """Return list of (call_idx, output_idx, call_id) for each complete pair, ordered by output index (recent first)."""
    pair_positions = []
    for call_id in call_ids_with_outputs:
        call_idx = output_idx = None
        for i, msg in enumerate(messages):
            if msg.get("type") == "function_call" and msg.get("call_id") == call_id:
                call_idx = i
            elif msg.get("type") == "function_call_output" and msg.get("call_id") == call_id:
                output_idx = i
        if call_idx is not None and output_idx is not None:
            pair_positions.append((call_idx, output_idx, call_id))
    pair_positions.sort(key=lambda x: x[1], reverse=True)
    return pair_positions


def _select_pairs_to_keep(pair_positions: list, keep_recent: int) -> list:
    """Keep only the most recent N pairs (N = max(5, keep_recent // 4))."""
    max_pairs_to_keep = max(5, keep_recent // 4)
    return pair_positions[:max_pairs_to_keep]


def _drop_orphaned_function_calls(
    messages: list,
    call_ids_with_outputs: set,
    pairs_to_keep: list,
    keep_recent: int,
    original_length: int,
) -> tuple[list, int]:
    """Remove old function_calls that have no output; return (cleaned_messages, earliest_index_to_keep)."""
    call_ids_with_calls = set()
    for msg in messages:
        if msg.get("type") == "function_call":
            cid = msg.get("call_id")
            if cid:
                call_ids_with_calls.add(cid)
    orphaned_call_ids = call_ids_with_calls - call_ids_with_outputs
    if not orphaned_call_ids:
        earliest = original_length
        for call_idx, _, _ in pairs_to_keep:
            earliest = min(earliest, call_idx)
        return messages, earliest

    cutoff = original_length - keep_recent
    cleaned = [
        msg for i, msg in enumerate(messages) if not (msg.get("type") == "function_call" and msg.get("call_id") in orphaned_call_ids and i < cutoff)
    ]
    earliest = original_length
    for _, _, call_id in pairs_to_keep:
        for i, msg in enumerate(cleaned):
            if msg.get("type") == "function_call" and msg.get("call_id") == call_id:
                earliest = min(earliest, i)
                break
    return cleaned, earliest


def _build_trimmed_messages(
    system_message: dict | None,
    messages: list,
    earliest_index: int,
    keep_recent: int,
    original_length: int,
) -> list:
    """Build final list: system + slice from start_idx (or fallback to last keep_recent)."""
    result = []
    if system_message:
        result.append(system_message)
    if earliest_index < original_length and earliest_index > 0:
        start_idx = min(earliest_index, original_length - keep_recent)
        result.extend(messages[start_idx:])
    else:
        result.extend(messages[-keep_recent:])
    return result


def _drop_orphaned_outputs(messages: list) -> list:
    """Remove any function_call_output whose function_call is no longer in the list; log if any removed."""
    output_call_ids = set()
    call_call_ids = set()
    for msg in messages:
        if msg.get("type") == "function_call_output":
            cid = msg.get("call_id")
            if cid:
                output_call_ids.add(cid)
        elif msg.get("type") == "function_call":
            cid = msg.get("call_id")
            if cid:
                call_call_ids.add(cid)
    missing_calls = output_call_ids - call_call_ids
    if missing_calls:
        logger.warning(f"[CONTEXT] WARNING: Found {len(missing_calls)} function_call_outputs without corresponding function_calls: {missing_calls}")
        return [msg for msg in messages if not (msg.get("type") == "function_call_output" and msg.get("call_id") in missing_calls)]
    return messages


class MessageHistory:

    def __init__(self, system_prompt: str, user_prompt: str = None):
        "Initialize message history with system prompt"

        self.messages = []
        self.append_system(system_prompt)
        if user_prompt:
            self.append_user(user_prompt)

    def append_function_call(self, call_id: str, name: str, arguments: str):
        """Append a function call message"""
        self.messages.append({"type": "function_call", "call_id": call_id, "name": name, "arguments": arguments})

    def append_function_output(self, call_id: str, output: str):
        """Append a function call output message"""
        self.messages.append({"type": "function_call_output", "call_id": call_id, "output": output})

    def append_user(self, content: str):
        "Append user message"
        self.messages.append({"role": "user", "content": content})

    def append_assistant(self, content: str):
        """Append assistant prompt"""
        self.messages.append({"role": "assistant", "content": content})

    def pop(self):
        return self.messages.pop() if self.messages else None

    def append_system(self, content: str):
        """Append system prompt"""
        if self.messages and self.messages[0].get("role") == "system":
            self.messages[0]["content"] = content
        else:
            self.messages.insert(0, {"role": "system", "content": content})

    def trim(self, keep_recent: int = 20):
        """
        Trim message history to prevent context window overflow.

        OpenAI API has token limits, so we periodically trim old messages while preserving:
        - System prompt (always kept)
        - Recent function calls and their outputs (kept together to maintain call_id matching)
        - Current state (last N messages)

        IMPORTANT: Ensures function_call and function_call_output pairs are kept together.
        If a function_call_output exists anywhere in messages, its corresponding function_call
        must also be kept.

        Args:
            keep_recent: Number of recent messages to keep (default: 20)
        """
        if len(self.messages) <= keep_recent:
            return
        original_length = len(self.messages)
        working = self.messages.copy()
        system_message = working[0] if working else None

        call_ids_with_outputs = _collect_call_ids_with_outputs(working)
        pair_positions = _find_pair_positions(working, call_ids_with_outputs)
        pairs_to_keep = _select_pairs_to_keep(pair_positions, keep_recent)
        working, earliest_index = _drop_orphaned_function_calls(working, call_ids_with_outputs, pairs_to_keep, keep_recent, original_length)
        self.messages = _build_trimmed_messages(system_message, working, earliest_index, keep_recent, original_length)
        self.messages = _drop_orphaned_outputs(self.messages)
        logger.debug(f"[CONTEXT] Trimmed messages: {original_length} -> {len(self.messages)} messages")

    def validate_pairs(self):
        """
        Validate that all function_call_outputs have their corresponding function_calls.
        This prevents API errors from orphaned outputs.
        """
        output_call_ids = set()
        call_call_ids = set()

        for msg in self.messages:
            if msg.get("type") == "function_call_output":
                call_id = msg.get("call_id")
                if call_id:
                    output_call_ids.add(call_id)
            elif msg.get("type") == "function_call":
                call_id = msg.get("call_id")
                if call_id:
                    call_call_ids.add(call_id)

        missing_calls = output_call_ids - call_call_ids
        if missing_calls:
            logger.error(f"[ERROR] Found {len(missing_calls)} function_call_outputs without corresponding function_calls: {missing_calls}")
            # Remove orphaned outputs to prevent API errors
            self.messages = [msg for msg in self.messages if not (msg.get("type") == "function_call_output" and msg.get("call_id") in missing_calls)]
            logger.warning(f"[FIXED] Removed {len(missing_calls)} orphaned function_call_outputs")

    def get_messages(self) -> list[dict]:
        """Get current messages list."""
        return self.messages

    def __len__(self) -> int:
        """Return number of messages"""
        return len(self.messages)
