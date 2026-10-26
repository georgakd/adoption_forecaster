"""Thin LLM client: a Protocol, the NVIDIA OpenAI-compatible endpoint, and a scripted stub for tests.

The API key is read from NVIDIA_API_KEY only and is never stored on the client, logged, or put in exception text.
"""

import copy
import json
import logging
import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

logger = logging.getLogger(__name__)

NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
_BACKOFF_BASE_SECONDS = 1.0


class MissingApiKeyError(RuntimeError):
    """NVIDIA_API_KEY is not set."""


class LLMUnavailableError(RuntimeError):
    """The LLM could not be reached or rejected the request; the message carries no request or key details."""


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict
    parse_error: str | None = None


@dataclass
class LLMResponse:
    text: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)


class LLMClient(Protocol):
    def chat(self, messages: list[dict], tools: list[dict]) -> LLMResponse: ...


def _status_code(exc: Exception) -> int | None:
    status = getattr(exc, "status_code", None)
    return status if isinstance(status, int) else None


def _is_retryable(exc: Exception) -> bool:
    status = _status_code(exc)
    if status is not None:
        return status == 429 or status >= 500
    # no HTTP status: connection errors and timeouts (builtin and the openai SDK's APIConnectionError/APITimeoutError)
    return isinstance(exc, (ConnectionError, TimeoutError)) or type(exc).__name__ in {"APIConnectionError", "APITimeoutError"}


def _safe_message(exc: Exception) -> str:
    status = _status_code(exc)
    suffix = f" (HTTP {status})" if status is not None else ""
    return f"LLM request failed: {type(exc).__name__}{suffix}"


def _parse_tool_call(raw: Any) -> ToolCall:
    try:
        arguments = json.loads(raw.function.arguments or "{}")
        if not isinstance(arguments, dict):
            raise ValueError("arguments JSON is not an object")
        parse_error = None
    except (ValueError, TypeError) as exc:
        arguments, parse_error = {}, f"invalid tool arguments: {type(exc).__name__}"
    return ToolCall(id=raw.id, name=raw.function.name, arguments=arguments, parse_error=parse_error)


class NvidiaClient:
    def __init__(
        self,
        model: str,
        base_url: str = NVIDIA_BASE_URL,
        max_retries: int = 3,
        sdk_client: Any | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.model = model
        self.base_url = base_url
        self.max_retries = max_retries
        self._sleep = sleep
        if sdk_client is None:
            api_key = os.environ.get("NVIDIA_API_KEY")
            if not api_key:
                raise MissingApiKeyError("NVIDIA_API_KEY environment variable is not set")
            from openai import OpenAI

            sdk_client = OpenAI(base_url=base_url, api_key=api_key)
        self._sdk = sdk_client

    def __repr__(self) -> str:
        return f"NvidiaClient(model={self.model!r}, base_url={self.base_url!r}, max_retries={self.max_retries})"

    def chat(self, messages: list[dict], tools: list[dict]) -> LLMResponse:
        kwargs: dict[str, Any] = {"model": self.model, "messages": messages}
        if tools:
            kwargs["tools"] = tools
        failure = ""
        for attempt in range(self.max_retries + 1):
            try:
                completion = self._sdk.chat.completions.create(**kwargs)
                break
            except Exception as exc:  # noqa: BLE001 - every SDK failure is wrapped with a scrubbed message
                failure = _safe_message(exc)
                if attempt < self.max_retries and _is_retryable(exc):
                    logger.info("%s; retrying (%d/%d)", failure, attempt + 1, self.max_retries)
                    self._sleep(_BACKOFF_BASE_SECONDS * 2**attempt)
                    continue
            # raised outside the except block so the SDK exception (which may echo request details) is not chained
            logger.warning(failure)
            raise LLMUnavailableError(failure)
        msg = completion.choices[0].message
        return LLMResponse(text=msg.content, tool_calls=[_parse_tool_call(tc) for tc in (msg.tool_calls or [])])


class ScriptedClient:
    """Test stub: replays canned responses in order and records every call."""

    def __init__(self, responses: list[LLMResponse]):
        self._responses = list(responses)
        self.calls: list[dict] = []

    def chat(self, messages: list[dict], tools: list[dict]) -> LLMResponse:
        if len(self.calls) >= len(self._responses):
            raise AssertionError("script exhausted")
        self.calls.append({"messages": copy.deepcopy(messages), "tools": copy.deepcopy(tools)})
        return self._responses[len(self.calls) - 1]


def get_llm_client(config: dict) -> LLMClient:
    correction = config["correction"]
    if not correction.get("model"):
        raise ValueError("correction.model is not set in the config; choose an NVIDIA model id before running the agent")
    return NvidiaClient(model=correction["model"], max_retries=correction["max_llm_retries"])
