import asyncio
import json
import logging
from unittest.mock import AsyncMock

import pytest
from agents import MaxTurnsExceeded, ModelResponse, OpenAIChatCompletionsModel, Usage
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion
from mcp.types import CallToolResult, TextContent
from openai.types.responses import (
    ResponseFunctionToolCall,
    ResponseOutputMessage,
    ResponseOutputText,
)

from app.agent import INVALID_MODEL_OUTPUT, NO_LIVE_DATA, requires_live_data, run_agent
import app.agent as agent_module
from app.config import MAX_TURNS
from app.prompts import SYSTEM_PROMPT


def answer(text):
    return ModelResponse(
        output=[ResponseOutputMessage(
            id="message", role="assistant", status="completed",
            content=[ResponseOutputText(text=text, type="output_text", annotations=[])],
            type="message",
        )],
        usage=Usage(), response_id=None,
    )


def call(name, arguments):
    return ModelResponse(
        output=[ResponseFunctionToolCall(
            name=name, arguments=json.dumps(arguments), call_id=f"call_{name}",
            type="function_call",
        )],
        usage=Usage(), response_id=None,
    )


def exposed_name(tools, server, original_name):
    matches = [tool.name for tool in tools
               if server in tool.name and tool.name.endswith(original_name)]
    assert len(matches) == 1
    return matches[0]


@pytest.fixture
def model(monkeypatch, mcp_boundary):
    # Fail immediately if any test accidentally reaches the HTTP client.
    monkeypatch.setattr(AsyncOpenAI, "request", AsyncMock(
        side_effect=AssertionError("Tests must not use the network")
    ))
    mock = AsyncMock()
    monkeypatch.setattr(OpenAIChatCompletionsModel, "get_response", mock)
    return mock


def test_final_answer_without_tools(model, config, mcp_boundary):
    model.return_value = answer("I can check pods and workflows.")
    result = asyncio.run(run_agent("What can you check?", config))
    assert isinstance(result, str) and result
    assert model.await_count == 1
    assert model.call_args.kwargs["system_instructions"] == SYSTEM_PROMPT
    assert model.call_args.kwargs["tracing"].is_disabled()
    exposed = model.call_args.kwargs["tools"]
    discovered = [(server, tool) for server, session in mcp_boundary.sessions.items()
                  for tool in session.list_tools.return_value.tools]
    assert len(exposed) == len(discovered)
    for server, source in discovered:
        name = exposed_name(exposed, server, source.name)
        actual = next(tool for tool in exposed if tool.name == name)
        assert actual.description == source.description
        assert actual.params_json_schema == source.model_dump(by_alias=True)["inputSchema"]
    assert "test-openshift-token" not in repr(model.call_args_list)


@pytest.mark.parametrize("namespace", ["payments", "staging"])
def test_tool_arguments_and_result_influence_answer(model, namespace, config, mcp_boundary):
    def respond(**kwargs):
        if isinstance(kwargs["input"], str) or len(kwargs["input"]) == 1:
            return call(exposed_name(kwargs["tools"], "openshift", "get_pod_count"),
                        {"namespace": namespace})
        output = kwargs["input"][-1]
        assert output["type"] == "function_call_output"
        data = json.loads(output["output"][0]["text"])
        assert data == {"namespace": namespace, "pod_count": 3, "fake": True}
        return answer(f"{data['namespace']}: {data['pod_count']} fake pods")

    model.side_effect = respond
    result = asyncio.run(run_agent(f"How many pods are in {namespace}?", config))
    assert namespace in result and "3" in result
    assert model.await_count == 2
    mcp_boundary.sessions["openshift"].call_tool.assert_awaited_once_with(
        "get_pod_count", {"namespace": namespace}
    )


def test_sequential_tool_calls(model, config, mcp_boundary):
    def respond(**kwargs):
        if model.await_count == 1:
            return call(exposed_name(kwargs["tools"], "openshift", "get_pod_count"),
                        {"namespace": "payments"})
        if model.await_count == 2:
            return call(exposed_name(kwargs["tools"], "aap", "get_workflow_status"),
                        {"workflow_id": "deploy-42"})
        mcp_boundary.sessions["openshift"].call_tool.assert_awaited_once()
        mcp_boundary.sessions["aap"].call_tool.assert_awaited_once()
        return answer("Payments: 3 pods. Workflow deploy-42: succeeded.")

    model.side_effect = respond
    asyncio.run(run_agent("Check payments pods and workflow deploy-42.", config))
    assert model.await_count == 3
    for session in mcp_boundary.sessions.values():
        session.list_tools.assert_awaited_once()
    history = model.call_args.kwargs["input"]
    outputs = [json.loads(item["output"][0]["text"]) for item in history
               if isinstance(item, dict) and item.get("type") == "function_call_output"]
    assert outputs == [
        {"namespace": "payments", "pod_count": 3, "fake": True},
        {"workflow_id": "deploy-42", "status": "succeeded", "fake": True},
    ]


def test_turn_limit(model, config):
    model.side_effect = lambda **kwargs: call(
        exposed_name(kwargs["tools"], "openshift", "get_pod_count"), {"namespace": "payments"}
    )
    with pytest.raises(MaxTurnsExceeded):
        asyncio.run(run_agent("Check payments pods.", config))
    assert model.await_count == MAX_TURNS


def test_missing_information_asks_without_calling_tools(model, config, mcp_boundary):
    model.return_value = answer("Which namespace should I check?")
    result = asyncio.run(run_agent("How many pods are there?", config))
    assert result == NO_LIVE_DATA and model.await_count == 1
    for session in mcp_boundary.sessions.values():
        session.call_tool.assert_not_awaited()
    for tool in model.call_args.kwargs["tools"]:
        required = tool.params_json_schema["required"]
        assert len(required) == 1
        assert "default" not in tool.params_json_schema["properties"][required[0]]


def test_missing_tool_argument_is_not_defaulted(model, config, mcp_boundary):
    def respond(**kwargs):
        if model.await_count == 1:
            return call(exposed_name(kwargs["tools"], "openshift", "get_pod_count"), {})
        output = kwargs["input"][-1]["output"]
        assert "could not be retrieved" in output
        assert '"pod_count":' not in output
        return answer("Which namespace should I check?")

    model.side_effect = respond
    assert asyncio.run(run_agent("How many pods are there?", config)) == NO_LIVE_DATA
    assert model.await_count == 2

    mcp_boundary.sessions["openshift"].call_tool.assert_not_awaited()


def test_duplicate_tool_names_route_to_original_server(model, config, mcp_boundary):
    for server in ("openshift", "aap"):
        mcp_boundary.definitions[server] = ("get_status", "resource")
    names = {}

    def respond(**kwargs):
        for server in ("openshift", "aap"):
            name = exposed_name(kwargs["tools"], server, "get_status")
            if server in names:
                assert name == names[server]
            names[server] = name
        assert names["openshift"] != names["aap"]
        if model.await_count == 1:
            return call(names["openshift"], {"resource": "payments"})
        if model.await_count == 2:
            return call(names["aap"], {"resource": "deploy-42"})
        results = [json.loads(item["output"][0]["text"]) for item in kwargs["input"]
                   if isinstance(item, dict) and item.get("type") == "function_call_output"]
        assert results == [
            {"resource": "payments", "server": "openshift", "status": "ready"},
            {"resource": "deploy-42", "server": "aap", "status": "ready"},
        ]
        return answer("Both systems returned their status.")

    model.side_effect = respond
    assert asyncio.run(run_agent("Check payments and deploy-42 status.", config))
    assert model.await_count == 3
    mcp_boundary.sessions["openshift"].call_tool.assert_awaited_once_with(
        "get_status", {"resource": "payments"}
    )
    mcp_boundary.sessions["aap"].call_tool.assert_awaited_once_with(
        "get_status", {"resource": "deploy-42"}
    )
    mcp_boundary.sessions["itsm"].call_tool.assert_not_awaited()


@pytest.mark.parametrize("fabrication", [
    "There are 2 pods in namespace pepe.",
    "I ran kubectl get pods -n pepe. The command returned 2 pods.",
    "I called the MCP tool and checked the API: everything is healthy.",
])
def test_operational_answer_without_execution_is_blocked(model, config, mcp_boundary, fabrication, caplog):
    caplog.set_level(logging.INFO, logger="app")
    model.return_value = answer(fabrication)
    result = asyncio.run(run_agent("How many pods are in namespace pepe?", config))
    assert result == NO_LIVE_DATA
    assert fabrication not in result
    assert model.call_args.kwargs["model_settings"].tool_choice == "auto"
    for session in mcp_boundary.sessions.values():
        session.call_tool.assert_not_awaited()
    assert "tool_executed=False" in caplog.text


def test_operational_tool_result_is_authoritative(model, config, mcp_boundary, caplog):
    caplog.set_level(logging.INFO, logger="app")
    mcp_boundary.tool_results["openshift"] = CallToolResult(content=[
        TextContent(type="text", text=json.dumps({"namespace": "pepe", "count": 7})),
    ])

    def respond(**kwargs):
        if model.await_count == 1:
            assert kwargs["model_settings"].tool_choice == "auto"
            return call(exposed_name(kwargs["tools"], "openshift", "get_pod_count"),
                        {"namespace": "pepe"})
        assert kwargs["model_settings"].tool_choice != "required"
        mcp_boundary.sessions["openshift"].call_tool.assert_awaited_once_with(
            "get_pod_count", {"namespace": "pepe"}
        )
        data = json.loads(kwargs["input"][-1]["output"][0]["text"])
        assert data["count"] == 7
        return answer(f"There are {data['count']} pods in {data['namespace']}.")

    model.side_effect = respond
    result = asyncio.run(run_agent("How many pods are in namespace pepe?", config))
    assert "7" in result and "2" not in result
    assert result != NO_LIVE_DATA
    assert "get_pod_count" in caplog.text and "status=success" in caplog.text
    assert "status=starting" in caplog.text
    assert "MCP connected" in caplog.text
    assert "MCP tool discovery starting" in caplog.text
    assert "MCP tools exposed count=" in caplog.text
    assert "test-openshift-token" not in caplog.text


@pytest.mark.parametrize("failure", [
    CallToolResult(isError=True, content=[TextContent(type="text", text="Query failed")]),
    RuntimeError("Authorization: Bearer test-secret"),
])
def test_failed_tool_cannot_support_fabricated_answer(model, config, mcp_boundary, failure, caplog):
    caplog.set_level(logging.INFO, logger="app")
    mcp_boundary.tool_results["openshift"] = failure

    def respond(**kwargs):
        if model.await_count == 1:
            return call(exposed_name(kwargs["tools"], "openshift", "get_pod_count"),
                        {"namespace": "pepe"})
        return answer("There are 2 pods in namespace pepe.")

    model.side_effect = respond
    result = asyncio.run(run_agent("How many pods are in namespace pepe?", config))
    assert result == NO_LIVE_DATA
    mcp_boundary.sessions["openshift"].call_tool.assert_awaited_once()
    assert "get_pod_count" in caplog.text
    assert "status=error" in caplog.text or "status=failed" in caplog.text
    assert "test-secret" not in caplog.text + result + repr(model.call_args_list)


@pytest.mark.parametrize("message", ["hello", "what can you do?", "explain what a Kubernetes pod is"])
def test_general_questions_allow_direct_answers(model, config, mcp_boundary, message):
    model.return_value = answer("I can help with that general question.")
    result = asyncio.run(run_agent(message, config))
    assert result != NO_LIVE_DATA
    assert model.call_args.kwargs["model_settings"].tool_choice == "auto"
    for session in mcp_boundary.sessions.values():
        session.call_tool.assert_not_awaited()


@pytest.mark.parametrize("message", [
    "Hello, how many pods are running?", "What can you do? Check my tickets.",
    "Explain why my workflow failed", "Show current system health", "Is it working?",
])
def test_unknown_and_mixed_requests_require_tools(message):
    assert requires_live_data(message)


def test_no_tools_returns_controlled_response(model, config):
    from dataclasses import replace

    result = asyncio.run(run_agent("How many pods are in pepe?", replace(config, mcp_servers=())))
    assert result == NO_LIVE_DATA
    model.assert_not_awaited()


def test_provider_rejecting_required_tool_choice_can_answer_with_auto(model, config, mcp_boundary):
    def respond(**kwargs):
        if kwargs["model_settings"].tool_choice == "required":
            raise RuntimeError("Provider does not support required tool choice")
        if model.await_count == 1:
            return call(exposed_name(kwargs["tools"], "openshift", "get_pod_count"),
                        {"namespace": "pepe"})
        return answer("There are 3 pods in pepe.")

    model.side_effect = respond
    result = asyncio.run(run_agent("How many pods do we have?", config))
    assert result != NO_LIVE_DATA
    mcp_boundary.sessions["openshift"].call_tool.assert_awaited_once_with(
        "get_pod_count", {"namespace": "pepe"}
    )


@pytest.mark.parametrize("textual_call", [False, True])
def test_real_model_adapter_handles_reasoning_field_after_tool_call(config, mcp_boundary, monkeypatch, textual_call, caplog):
    """Keep the SDK model adapter real: the old boolean fails on its second turn."""
    caplog.set_level(logging.INFO, logger="app")
    def completion(**kwargs):
        if completions.await_count == 1:
            assert kwargs["tool_choice"] == "auto"
            tool = next(tool["function"]["name"] for tool in kwargs["tools"]
                        if "openshift" in tool["function"]["name"]
                        and tool["function"]["name"].endswith("get_pod_count"))
            message = {
                "role": "assistant", "content": None,
                "reasoning_content": "synthetic-test-field",
                "tool_calls": [{"id": "call_1", "type": "function", "function": {
                    "name": tool, "arguments": '{"namespace":"pepe"}',
                }}],
            }
            if textual_call:
                message = {
                    "role": "assistant",
                    "content": f'<|start|>assistant<|channel|>commentary to=functions.{tool} {{"namespace":"pepe"}}',
                }
        else:
            assert "synthetic-test-field" not in repr(kwargs["messages"])
            assert any(item["role"] == "tool" for item in kwargs["messages"])
            message = {"role": "assistant", "content": "There are 3 pods in pepe."}
        return ChatCompletion.model_validate({
            "id": "test", "created": 0, "model": "test-model", "object": "chat.completion",
            "choices": [{"index": 0, "message": message,
                         "finish_reason": "tool_calls" if completions.await_count == 1 and not textual_call else "stop"}],
        })

    completions = AsyncMock(side_effect=completion)

    def client(**kwargs):
        instance = AsyncOpenAI(**kwargs)
        monkeypatch.setattr(instance.chat.completions, "create", completions)
        return instance

    monkeypatch.setattr(agent_module, "AsyncOpenAI", client)
    monkeypatch.setattr(AsyncOpenAI, "request", AsyncMock(side_effect=AssertionError("No network")))
    result = asyncio.run(run_agent("How many pods do we have?", config))
    if textual_call:
        assert result == INVALID_MODEL_OUTPUT
        assert completions.await_count == 1
        assert "('message', ['output_text'])" in caplog.text
        assert "structured_tool_call=False" in caplog.text
        assert "protocol_in_assistant_text=True" in caplog.text
        for session in mcp_boundary.sessions.values():
            session.call_tool.assert_not_awaited()
        return
    assert "3" in result and result != NO_LIVE_DATA
    assert completions.await_count == 2
    mcp_boundary.sessions["openshift"].call_tool.assert_awaited_once_with(
        "get_pod_count", {"namespace": "pepe"}
    )
    assert "structured_tool_call=True" in caplog.text


@pytest.mark.parametrize("message", ["Hello", "How many pods are in pepe?"])
def test_protocol_text_is_rejected_even_for_general_questions(model, config, mcp_boundary, caplog, message):
    caplog.set_level(logging.INFO, logger="app")
    text = '<|start|>assistant<|channel|>commentary to=functions.fake {"secret":"sensitive-test-payload"}'
    model.return_value = answer(text)
    result = asyncio.run(run_agent(message, config))
    assert result == INVALID_MODEL_OUTPUT
    assert "<|" not in result
    assert text not in caplog.text
    assert "sensitive-test-payload" not in caplog.text
    assert "protocol_in_assistant_text=True" in caplog.text
    for session in mcp_boundary.sessions.values():
        session.call_tool.assert_not_awaited()


def test_protocol_text_after_real_tool_is_not_rendered(model, config, mcp_boundary):
    def respond(**kwargs):
        if model.await_count == 1:
            return call(exposed_name(kwargs["tools"], "openshift", "get_pod_count"),
                        {"namespace": "pepe"})
        return answer('<|start|>assistant<|channel|>commentary to=functions.fake {}')

    model.side_effect = respond
    assert asyncio.run(run_agent("How many pods are in pepe?", config)) == INVALID_MODEL_OUTPUT
    mcp_boundary.sessions["openshift"].call_tool.assert_awaited_once()
    mcp_boundary.sessions["aap"].call_tool.assert_not_awaited()


def test_plain_text_function_call_is_never_executed(model, config, mcp_boundary):
    model.return_value = answer('functions.get_pod_count(namespace="pepe")')
    assert asyncio.run(run_agent("How many pods are in pepe?", config)) == NO_LIVE_DATA
    for session in mcp_boundary.sessions.values():
        session.call_tool.assert_not_awaited()


def test_count_and_names_come_from_results(model, config, mcp_boundary):
    mcp_boundary.tool_results["openshift"] = CallToolResult(content=[TextContent(
        type="text", text=json.dumps({"count": 2, "names": ["payments-a", "payments-b"]}),
    )])

    def respond(**kwargs):
        if model.await_count == 1:
            return call(exposed_name(kwargs["tools"], "openshift", "get_pod_count"),
                        {"namespace": "payments"})
        data = json.loads(kwargs["input"][-1]["output"][0]["text"])
        return answer(f"{data['count']} pods: {', '.join(data['names'])}.")

    model.side_effect = respond
    result = asyncio.run(run_agent("Count and name payments pods.", config))
    assert "2" in result and "payments-a" in result and "payments-b" in result
    assert "kubectl" not in result and "I ran" not in result


def test_workflow_distinctions_remain_in_single_prompt(model, config):
    model.return_value = answer("Do you mean workflow job templates or workflow executions?")
    asyncio.run(run_agent("How many workflows are there?", config))
    prompt = model.call_args.kwargs["system_instructions"]
    assert prompt == SYSTEM_PROMPT
    for term in ("workflow job templates", "workflow jobs", "workflow executions"):
        assert term in prompt
    assert "clarification question" in prompt
