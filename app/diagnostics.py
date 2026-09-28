"""Small, credential-safe exception diagnostics shared by the entry points."""

import logging
import os
import re


class ProtocolTextError(RuntimeError):
    """The model emitted reserved protocol delimiters in user-facing text."""


def log_model_response(logger, response) -> None:
    structure = []
    protocol_text = False
    for item in response.output:
        content = getattr(item, "content", []) or []
        structure.append((item.type, [part.type for part in content]))
        if item.type == "message":
            for part in content:
                if part.type == "output_text" and re.search(r"<\|[^<>\r\n]+\|>", part.text):
                    protocol_text = True
    logger.info(
        "Model response structure=%s structured_tool_call=%s protocol_in_assistant_text=%s",
        structure, any(item.type == "function_call" for item in response.output), protocol_text,
    )
    if protocol_text:
        # Reject the whole response. Never strip tokens, decode a native protocol,
        # extract arguments, or execute calls represented only in text.
        raise ProtocolTextError(
            "Provider returned protocol delimiters as normal assistant text; "
            "check provider tool-call formatting."
        )


def configure_logging() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s %(message)s")
    logging.getLogger("app").setLevel(logging.INFO)


def log_failure(logger: logging.Logger, phase: str, error: Exception, secrets=()) -> None:
    # API errors may embed the entire response body; retain only its error message.
    body = getattr(error, "body", None)
    if isinstance(body, dict):
        detail = body.get("error", body)
        message = str(detail.get("message", type(error).__name__)) if isinstance(detail, dict) else str(detail)
    else:
        message = str(error)
    values = [*secrets, *(value for key, value in os.environ.items()
                         if any(word in key.upper() for word in ("KEY", "TOKEN", "SECRET", "PASSWORD")))]
    for value in sorted((value for value in values if value), key=len, reverse=True):
        message = message.replace(value, "[redacted]")
    # Also redact credentials not present in configuration (for example echoed headers).
    message = re.sub(
        r"(?i)(?:authorization|api[_-]?key|token|password|secret)[\"']?\s*[:=][^\r\n]*",
        "[redacted credential]", message,
    )
    message = re.sub(r"(?i)\bBearer\s+\S+", "[redacted credential]", message)
    logger.error("%s: %s: %s", phase, type(error).__name__, message.replace("\n", " ").replace("\r", " "))
