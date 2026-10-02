import asyncio
import sys

from agents import MaxTurnsExceeded

from app.agent import run_agent
from app.config import load_config
from app.diagnostics import configure_logging
from app.mcp import MCPConnectionError
from app.procedures import ProcedureRetrievalError, procedure_query, run_procedure


def main() -> None:
    configure_logging()
    try:
        config = load_config()
        message = input("> ").strip()
        if message:
            query = procedure_query(message)
            runner = run_agent if query is None else run_procedure
            print(asyncio.run(runner(message if query is None else query, config)))
    except (ValueError, MCPConnectionError, ProcedureRetrievalError) as error:
        sys.exit(str(error))
    except MaxTurnsExceeded:
        sys.exit("The agent reached its turn limit. Try a more specific request.")
    except (EOFError, KeyboardInterrupt):
        pass


if __name__ == "__main__":
    main()
