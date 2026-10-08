import asyncio
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
from agents import MaxTurnsExceeded

from app import web
from app.mcp import MCPConnectionError


def request(method, path, **kwargs):
    async def send():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=web.app), base_url="http://test"
        ) as client:
            return await client.request(method, path, **kwargs)
    return asyncio.run(send())


@pytest.fixture
def agent(monkeypatch, config):
    monkeypatch.setattr(web, "load_config", lambda: config)
    mock = AsyncMock(return_value="The systems are available.")
    monkeypatch.setattr(web, "run_agent", mock)
    return mock


def test_page_does_not_call_agent(agent):
    response = request("GET", "/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Operations Agent" in response.text
    assert "autofocus" in response.text
    agent.assert_not_awaited()


# --- Procedure UI markers ----------------------------------------------------
# This project has no JS test runner, so (as with the assertions above) these
# check the rendered page for the required procedure-mode building blocks.

def test_page_includes_procedure_input_mode(agent):
    page = request("GET", "/").text
    assert 'id="procedure-chip"' in page
    assert "PROCEDURE_PATTERN" in page
    assert "enterProcedureMode" in page
    assert "procedure-input" in page


def test_page_includes_distinct_procedure_message_styling(agent):
    page = request("GET", "/").text
    assert "procedure-user" in page
    assert "appendProcedureMessage" in page


def test_page_includes_transient_procedure_progress(agent):
    page = request("GET", "/").text
    assert "procedure-progress" in page
    assert "procedureProgress" in page
    assert "Searching KB" in page
    # The permanent card/heading from the previous iteration must be gone.
    assert "procedure-card" not in page
    assert "createProcedureCard" not in page


def test_page_includes_animated_pending_indicator(agent):
    page = request("GET", "/").text
    assert "ELLIPSIS_FRAMES" in page
    assert "'...'" in page


def test_page_hides_progress_before_showing_the_final_message(agent):
    page = request("GET", "/").text
    # Both the success and the error path hide the transient bubble before
    # rendering the outcome as a normal chat message, never both at once.
    assert "procedureProgress.hide();\n        if (isProcedure && data.type === 'procedure_status') {\n          appendMessage(data.message, 'assistant');" in page
    assert "procedureProgress.hide();\n        appendMessage(`Error: ${error.message}`, 'error');" in page


def test_page_skips_progress_for_empty_procedure_command(agent):
    page = request("GET", "/").text
    assert "if (isProcedure && procedureQuery) procedureProgress.show('Searching KB');" in page


def test_chat_reuses_agent(agent, config):
    response = request("POST", "/api/chat", json={"message": "  Check pods  "})
    assert response.status_code == 200
    assert response.json() == {"response": agent.return_value}
    agent.assert_awaited_once_with("Check pods", config)
    assert config.model_api_key not in response.text


@pytest.mark.parametrize("body", [{}, {"message": ""}, {"message": "   "}, {"message": None}])
def test_invalid_message_does_not_call_agent(agent, body):
    assert request("POST", "/api/chat", json=body).status_code == 422
    agent.assert_not_awaited()


@pytest.mark.parametrize("error, status, detail", [
    (MCPConnectionError("MCP connection failed: aap"), 502, "aap"),
    (MaxTurnsExceeded("limit"), 422, "turn limit"),
    (RuntimeError("Authorization: Bearer test-secret"), 502, "request failed"),
])
def test_agent_errors_are_safe(agent, error, status, detail, caplog):
    agent.side_effect = error
    response = request("POST", "/api/chat", json={"message": "Check pods"})
    assert response.status_code == status
    assert detail in response.json()["detail"]
    assert "test-secret" not in response.text + caplog.text


def test_configuration_error(agent, monkeypatch):
    monkeypatch.setattr(web, "load_config", Mock(side_effect=ValueError("test-secret")))
    response = request("POST", "/api/chat", json={"message": "Check pods"})
    assert response.status_code == 503
    assert "test-secret" not in response.text
    agent.assert_not_awaited()


def test_provider_error_is_diagnostic_only_on_server(agent, config, caplog):
    agent.side_effect = RuntimeError(
        "Provider does not support tool_choice=required\n"
        f"API key: {config.model_api_key}\n"
        f"Authorization: Bearer {config.mcp_servers[0].token}"
    )
    response = request("POST", "/api/chat", json={"message": "How many pods do we have?"})
    assert response.status_code == 502
    assert "RuntimeError" in caplog.text
    assert "does not support tool_choice=required" in caplog.text
    assert "does not support" not in response.text
    assert config.model_api_key not in caplog.text + response.text
    assert config.mcp_servers[0].token not in caplog.text + response.text
    assert "Authorization" not in caplog.text


def test_error_logging_omits_response_payload(agent, caplog):
    error = RuntimeError("raw response contains sensitive-payload")
    error.body = {"error": {"message": "Unsupported tool choice"}, "request": "sensitive-payload"}
    agent.side_effect = error
    response = request("POST", "/api/chat", json={"message": "Check pods"})
    assert response.status_code == 502
    assert "Unsupported tool choice" in caplog.text
    assert "sensitive-payload" not in caplog.text + response.text


# --- Deterministic /procedure routing ---------------------------------------
# No LLM classifies these messages: only the literal "/procedure" prefix
# diverts from the existing OperationsAgent chat path.

@pytest.fixture
def procedure(monkeypatch):
    from app.procedure.models import ProcedureStatus
    mock = AsyncMock(return_value=ProcedureStatus(state="kb_found", message="KB found: Test"))
    monkeypatch.setattr(web, "handle_procedure", mock)
    return mock


def test_procedure_message_uses_procedure_handler(agent, procedure, config):
    response = request("POST", "/api/chat", json={"message": "/procedure inspect namespace health"})
    assert response.status_code == 200
    assert response.json() == {"type": "procedure_status", "state": "kb_found", "message": "KB found: Test"}
    procedure.assert_awaited_once_with("/procedure inspect namespace health", config)
    agent.assert_not_awaited()


def test_normal_message_uses_existing_agent_path(agent, procedure, config):
    response = request("POST", "/api/chat", json={"message": "inspect namespace health"})
    assert response.status_code == 200
    assert response.json() == {"response": agent.return_value}
    agent.assert_awaited_once_with("inspect namespace health", config)
    procedure.assert_not_awaited()


def test_natural_language_mention_of_procedure_uses_existing_agent_path(agent, procedure, config):
    message = "what is the procedure for inspecting namespace health?"
    response = request("POST", "/api/chat", json={"message": message})
    assert response.status_code == 200
    assert response.json() == {"response": agent.return_value}
    agent.assert_awaited_once_with(message, config)
    procedure.assert_not_awaited()
