import asyncio
import sys

from agents import MaxTurnsExceeded

from app.agent import run_agent
from app.config import load_config
from app.diagnostics import configure_logging
from app.mcp import MCPConnectionError


def main() -> None:
    configure_logging()
    try:
        config = load_config()
        message = input("> ").strip()
        if message:
            print(asyncio.run(run_agent(message, config)))
    except (ValueError, MCPConnectionError) as error:
        sys.exit(str(error))
    except MaxTurnsExceeded:
        sys.exit("The agent reached its turn limit. Try a more specific request.")
    except (EOFError, KeyboardInterrupt):
        pass


if __name__ == "__main__":
    main()
