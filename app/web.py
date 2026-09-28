from pathlib import Path
import logging
from typing import Annotated

from agents import MaxTurnsExceeded
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, StringConstraints

from app.agent import run_agent
from app.config import load_config
from app.diagnostics import configure_logging, log_failure
from app.mcp import MCPConnectionError


configure_logging()
app = FastAPI()
PAGE = Path(__file__).with_name("index.html").read_text(encoding="utf-8")


class ChatRequest(BaseModel):
    message: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    return PAGE


@app.post("/api/chat")
async def chat(request: ChatRequest) -> dict[str, str]:
    try:
        config = load_config()
    except ValueError:
        raise HTTPException(503, "Check the server's environment configuration.") from None
    try:
        response = await run_agent(request.message, config)
    except MCPConnectionError as error:
        # This application exception contains only server names and error categories.
        raise HTTPException(502, str(error)) from None
    except MaxTurnsExceeded:
        raise HTTPException(422, "The agent reached its turn limit. Try a more specific request.") from None
    except Exception as error:
        log_failure(logging.getLogger(__name__), "Chat request failed", error,
                    (config.model_api_key, *(server.token for server in config.mcp_servers)))
        raise HTTPException(502, "The agent request failed. Check model and MCP connectivity.") from None
    return {"response": response}
