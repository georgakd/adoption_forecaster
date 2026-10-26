import json
import logging
from types import SimpleNamespace

import pytest

from adoption_forecaster.correction.llm import (
    LLMResponse,
    LLMUnavailableError,
    MissingApiKeyError,
    NvidiaClient,
    ScriptedClient,
    ToolCall,
    get_llm_client,
)

SECRET = "nvapi-test-secret"


class FakeHttpError(Exception):
    def __init__(self, status_code, text=""):
        super().__init__(text)
        self.status_code = status_code


def _completion(text=None, tool_calls=None):
    message = SimpleNamespace(content=text, tool_calls=tool_calls)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def _tool_call(arguments, call_id="c1", name="lookup"):
    return SimpleNamespace(id=call_id, function=SimpleNamespace(name=name, arguments=arguments))


class FakeSdk:
    """Fake OpenAI SDK client: each outcome is either an exception to raise or a completion to return."""

    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _client(outcomes, max_retries=3):
    sleeps = []
    sdk = FakeSdk(outcomes)
    return NvidiaClient(model="m", max_retries=max_retries, sdk_client=sdk, sleep=sleeps.append), sdk, sleeps


def test_missing_key_raises_clear_error_without_leaking_anything(monkeypatch):
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    with pytest.raises(MissingApiKeyError) as exc_info:
        NvidiaClient(model="m")
    assert isinstance(exc_info.value, RuntimeError)
    assert "NVIDIA_API_KEY" in str(exc_info.value)
    assert "nvapi" not in str(exc_info.value)


def test_key_never_in_repr_or_exception_text(monkeypatch, caplog):
    monkeypatch.setenv("NVIDIA_API_KEY", SECRET)
    sdk = FakeSdk([FakeHttpError(401, f"bad key {SECRET}")])
    client = NvidiaClient(model="m", sdk_client=sdk, sleep=lambda s: None)
    assert SECRET not in repr(client)
    with caplog.at_level(logging.DEBUG), pytest.raises(LLMUnavailableError) as exc_info:
        client.chat([{"role": "user", "content": "hi"}], [])
    exc = exc_info.value
    assert SECRET not in str(exc)
    assert "FakeHttpError" in str(exc) and "401" in str(exc)
    assert exc.__cause__ is None and exc.__context__ is None
    assert SECRET not in caplog.text


def test_tool_call_json_is_parsed_into_arguments():
    client, _, _ = _client([_completion(tool_calls=[_tool_call(json.dumps({"a": 1}))])])
    response = client.chat([], [])
    assert response.text is None
    assert response.tool_calls == [ToolCall(id="c1", name="lookup", arguments={"a": 1})]


def test_malformed_tool_arguments_become_empty_dict_with_parse_error_flag():
    client, _, _ = _client([_completion(tool_calls=[_tool_call("{not json")])])
    (call,) = client.chat([], []).tool_calls
    assert call.arguments == {} and call.parse_error


def test_non_object_tool_arguments_flag_parse_error():
    client, _, _ = _client([_completion(tool_calls=[_tool_call("[1, 2]")])])
    (call,) = client.chat([], []).tool_calls
    assert call.arguments == {} and call.parse_error


def test_retries_on_429_then_succeeds():
    client, sdk, sleeps = _client([FakeHttpError(429), FakeHttpError(503), _completion(text="ok")])
    assert client.chat([], []).text == "ok"
    assert len(sdk.calls) == 3
    assert sleeps == [1.0, 2.0]


def test_gives_up_after_max_retries_and_raises_llm_unavailable():
    client, sdk, sleeps = _client([FakeHttpError(500)] * 3, max_retries=2)
    with pytest.raises(LLMUnavailableError):
        client.chat([], [])
    assert len(sdk.calls) == 3 and len(sleeps) == 2


def test_no_retry_on_auth_error():
    client, sdk, sleeps = _client([FakeHttpError(400), _completion(text="ok")])
    with pytest.raises(LLMUnavailableError):
        client.chat([], [])
    assert len(sdk.calls) == 1 and sleeps == []


def test_retries_on_connection_error():
    client, _, sleeps = _client([ConnectionError("down"), _completion(text="ok")])
    assert client.chat([], []).text == "ok"
    assert len(sleeps) == 1


def test_tools_omitted_when_empty_and_passed_when_given():
    client, sdk, _ = _client([_completion(text="a"), _completion(text="b")])
    client.chat([{"role": "user", "content": "x"}], [])
    client.chat([], [{"type": "function"}])
    assert "tools" not in sdk.calls[0] and sdk.calls[1]["tools"] == [{"type": "function"}]


def test_get_llm_client_requires_model(monkeypatch):
    monkeypatch.setenv("NVIDIA_API_KEY", SECRET)
    with pytest.raises(ValueError, match="model"):
        get_llm_client({"correction": {"model": None, "max_llm_retries": 3}})
    client = get_llm_client({"correction": {"model": "some/model", "max_llm_retries": 5}})
    assert isinstance(client, NvidiaClient) and client.max_retries == 5


def test_scripted_client_replays_in_order():
    r1, r2 = LLMResponse(text="a", tool_calls=[]), LLMResponse(text=None, tool_calls=[ToolCall("1", "t", {})])
    client = ScriptedClient([r1, r2])
    messages = [{"role": "user", "content": "q"}]
    assert client.chat(messages, []) is r1
    messages.append({"role": "assistant", "content": "later mutation"})
    assert client.chat([], [{"x": 1}]) is r2
    with pytest.raises(AssertionError, match="script exhausted"):
        client.chat([], [])
    assert client.calls[0]["messages"] == [{"role": "user", "content": "q"}]
    assert client.calls[1]["tools"] == [{"x": 1}]
